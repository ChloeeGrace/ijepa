#!/usr/bin/env python3
import argparse
import logging
import os
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import src.models.vision_transformer as vit


logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class IJEPALinearClassifier(nn.Module):
    """I-JEPA encoder + linear head for classification evaluation."""

    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int, pool: str = 'mean'):
        super().__init__()
        if pool not in {'mean', 'cls'}:
            raise ValueError("pool must be one of {'mean', 'cls'}")
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)
        self.pool = pool

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(images)  # [B, N, D]
        if self.pool == 'cls':
            feats = tokens[:, 0]
        else:
            feats = tokens.mean(dim=1)
        logits = self.head(feats)
        return logits


def setup_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    is_distributed = world_size > 1

    if not is_distributed:
        return False, 0, 1, 0

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    return True, rank, world_size, local_rank


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith('module.') for k in state_dict.keys()):
        return state_dict
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}


def load_encoder_weights(encoder: nn.Module, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'encoder' in checkpoint:
        encoder_state = checkpoint['encoder']
    else:
        encoder_state = checkpoint

    encoder_state = strip_module_prefix(encoder_state)
    msg = encoder.load_state_dict(encoder_state, strict=False)
    logger.info(f'Loaded encoder checkpoint: {ckpt_path}')
    logger.info(f'Encoder load msg: {msg}')


def load_classifier_weights(model: nn.Module, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        else:
            state = checkpoint
    else:
        raise ValueError('Unsupported classifier checkpoint format')

    state = strip_module_prefix(state)
    msg = model.load_state_dict(state, strict=False)
    logger.info(f'Loaded classifier/finetuned checkpoint: {ckpt_path}')
    logger.info(f'Classifier load msg: {msg}')


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        k_eff = min(k, output.size(1))
        correct_k = correct[:k_eff].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def build_dataloader(data_path: str, image_size: int, batch_size: int, num_workers: int,
                     is_distributed: bool, rank: int, world_size: int):
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, is_distributed: bool):
    model.eval()
    top1_sum = torch.zeros(1, device=device)
    top5_sum = torch.zeros(1, device=device)
    n_sum = torch.zeros(1, device=device)

    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(images)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            batch_size = images.size(0)
            top1_sum += acc1 * batch_size / 100.0
            top5_sum += acc5 * batch_size / 100.0
            n_sum += batch_size

    if is_distributed:
        dist.all_reduce(top1_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(top5_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_sum, op=dist.ReduceOp.SUM)

    top1 = (top1_sum / n_sum * 100.0).item()
    top5 = (top5_sum / n_sum * 100.0).item()
    return top1, top5


def parse_args():
    parser = argparse.ArgumentParser('I-JEPA linear evaluation / inference script')
    parser.add_argument('--data-path', type=str, required=True, help='ImageFolder root for val/test split')
    parser.add_argument('--checkpoint', type=str, required=True, help='Pretrained or finetuned checkpoint path')
    parser.add_argument('--checkpoint-type', type=str, default='pretrained', choices=['pretrained', 'finetuned'])
    parser.add_argument('--model-name', type=str, default='vit_huge', choices=sorted(vit.VIT_EMBED_DIMS.keys()))
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--pool', type=str, default='mean', choices=['mean', 'cls'])
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    is_distributed, rank, world_size, local_rank = setup_distributed()

    use_cuda = args.device.startswith('cuda') and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f'cuda:{local_rank}' if is_distributed else args.device)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    if rank == 0:
        logger.info(f'Distributed: {is_distributed}, world_size={world_size}')

    encoder = vit.__dict__[args.model_name](
        img_size=[args.image_size],
        patch_size=args.patch_size,
    )

    model = IJEPALinearClassifier(
        encoder=encoder,
        embed_dim=vit.VIT_EMBED_DIMS[args.model_name],
        num_classes=args.num_classes,
        pool=args.pool,
    )

    if args.checkpoint_type == 'pretrained':
        load_encoder_weights(model.encoder, args.checkpoint)
    else:
        load_classifier_weights(model, args.checkpoint)

    model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    dataset, loader = build_dataloader(
        args.data_path,
        args.image_size,
        args.batch_size,
        args.num_workers,
        is_distributed,
        rank,
        world_size,
    )

    top1, top5 = evaluate(model, loader, device, is_distributed)

    if rank == 0:
        logger.info(f'Dataset samples: {len(dataset)}')
        logger.info(f'Top-1 Accuracy: {top1:.4f}%')
        logger.info(f'Top-5 Accuracy: {top5:.4f}%')

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
