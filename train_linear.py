import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.models.vision_transformer import (
    VIT_EMBED_DIMS,
    vit_base,
    vit_giant,
    vit_huge,
    vit_large,
    vit_small,
    vit_tiny,
)


VIT_BUILDERS = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_huge": vit_huge,
    "vit_giant": vit_giant,
}


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


class LinearClassifierModel(nn.Module):
    """I-JEPA ViT encoder + linear classifier head."""

    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder returns token embeddings with shape [B, N, D].
        tokens = self.encoder(x)
        if tokens.ndim == 3:
            feats = tokens.mean(dim=1)
        elif tokens.ndim == 2:
            feats = tokens
        else:
            raise RuntimeError(f"Unexpected encoder output shape: {tokens.shape}")
        logits = self.head(feats)
        return logits


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, num_classes: int, device: torch.device):
    if args.arch not in VIT_BUILDERS:
        raise ValueError(f"Unknown architecture: {args.arch}. Choices: {list(VIT_BUILDERS)}")

    encoder = VIT_BUILDERS[args.arch](
        img_size=[args.image_size],
        patch_size=args.patch_size,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "encoder" not in ckpt:
        raise KeyError("Checkpoint does not contain key 'encoder'.")

    missing, unexpected = encoder.load_state_dict(ckpt["encoder"], strict=False)
    if missing:
        print(f"[Warning] Missing encoder keys: {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected encoder keys: {len(unexpected)}")

    model = LinearClassifierModel(
        encoder=encoder,
        embed_dim=VIT_EMBED_DIMS[args.arch],
        num_classes=num_classes,
    )

    if args.mode == "linear":
        for p in model.encoder.parameters():
            p.requires_grad = False

    model = model.to(device)
    if torch.cuda.device_count() > 1 and not args.no_data_parallel:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    return model


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            k = min(k, output.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k * (100.0 / output.size(0)))
        return res


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval=50):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()

    start = time.time()
    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        acc1, _ = accuracy(logits, targets, topk=(1, 5))
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        top1_meter.update(acc1.item(), bs)

        if step % log_interval == 0 or step == len(loader):
            print(
                f"Epoch [{epoch}] Step [{step}/{len(loader)}] "
                f"Loss: {loss_meter.avg:.4f} Top1: {top1_meter.avg:.2f}"
            )

    elapsed = time.time() - start
    return {"loss": loss_meter.avg, "top1": top1_meter.avg, "time_sec": elapsed}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        top1_meter.update(acc1.item(), bs)
        top5_meter.update(acc5.item(), bs)

    return {"loss": loss_meter.avg, "top1": top1_meter.avg, "top5": top5_meter.avg}


def build_dataloaders(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if args.val_dir:
        train_set = datasets.ImageFolder(args.train_dir, transform=train_tf)
        val_set = datasets.ImageFolder(args.val_dir, transform=val_tf)
    else:
        full = datasets.ImageFolder(args.train_dir, transform=train_tf)
        val_len = int(len(full) * args.val_split)
        train_len = len(full) - val_len
        train_set, val_set = random_split(
            full,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(args.seed),
        )
        # override validation transform
        val_set.dataset = datasets.ImageFolder(args.train_dir, transform=val_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    num_classes = len(train_set.dataset.classes) if hasattr(train_set, "dataset") else len(train_set.classes)
    return train_loader, val_loader, num_classes


def parse_args():
    p = argparse.ArgumentParser("I-JEPA linear probe / finetune")
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--val-dir", type=str, default="")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--arch", type=str, default="vit_base", choices=list(VIT_BUILDERS.keys()))
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--resize-size", type=int, default=256)
    p.add_argument("--mode", type=str, default="linear", choices=["linear", "finetune"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"])
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"])
    p.add_argument("--step-size", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="outputs_linear")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--no-data-parallel", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes = build_dataloaders(args)
    model = build_model(args, num_classes=num_classes, device=device)

    criterion = nn.CrossEntropyLoss().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_top1 = -math.inf

    if args.eval_only:
        val_stats = evaluate(model, val_loader, criterion, device)
        print(f"Eval | Loss {val_stats['loss']:.4f} | Top1 {val_stats['top1']:.2f} | Top5 {val_stats['top5']:.2f}")
        return

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_stats = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_stats['loss']:.4f} Top1 {train_stats['top1']:.2f} | "
            f"Val Loss {val_stats['loss']:.4f} Top1 {val_stats['top1']:.2f} Top5 {val_stats['top5']:.2f}"
        )

        state = {
            "epoch": epoch,
            "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "val_top1": val_stats["top1"],
            "val_top5": val_stats["top5"],
        }

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_name = (
            f"last-ep{epoch:03d}-top1_{val_stats['top1']:.2f}-"
            f"top5_{val_stats['top5']:.2f}-{timestamp}.pt"
        )
        torch.save(state, os.path.join(args.output_dir, ckpt_name))

        if val_stats["top1"] > best_top1:
            best_top1 = val_stats["top1"]
            best_name = (
                f"best-ep{epoch:03d}-top1_{val_stats['top1']:.2f}-"
                f"top5_{val_stats['top5']:.2f}-{timestamp}.pt"
            )
            torch.save(state, os.path.join(args.output_dir, best_name))

    print(f"Training complete. Best Val Top1: {best_top1:.2f}")


if __name__ == "__main__":
    main()
