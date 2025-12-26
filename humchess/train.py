"""
Training script with DDP support.

Usage:
    Single GPU:
        python -m humchess.train --config configs/model.yml

    Multi-GPU:
        torchrun --nproc_per_node=4 -m humchess.train --config configs/model.yml
"""

import argparse
import glob
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

from .data.pgn_dataset import PGNDataset, collate_fn
from .model.transformer import ChessTransformer


def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy with legality masking.

    Args:
        logits: Raw logits, shape (batch, num_classes)
        targets: Target class indices, shape (batch,)
        mask: Boolean mask, True for legal moves, shape (batch, num_classes)

    Returns:
        Scalar loss.
    """
    # Apply mask: set illegal logits to -inf
    masked_logits = logits.clone()
    masked_logits[~mask] = float('-inf')

    return nn.functional.cross_entropy(masked_logits, targets)


def load_config(path: Path) -> dict:
    """Load full config from YAML file."""
    with path.open('r') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping.")

    # Validate model config
    model_cfg = data.get('model', {})
    if not isinstance(model_cfg, dict):
        raise ValueError("Config 'model' must be a YAML mapping.")

    required = ['d_model', 'n_heads', 'n_layers', 'd_ff']
    missing = [key for key in required if key not in model_cfg]
    if missing:
        raise ValueError(f"Model config missing keys: {missing}")

    return data


def expand_globs(patterns: list[str]) -> list[Path]:
    """Expand glob patterns and directories to file paths."""
    paths = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.parquet")))
        elif '*' in pattern or '?' in pattern:
            paths.extend(sorted(Path(m) for m in glob.glob(pattern)))
        else:
            paths.append(p)
    return paths


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str,
    scaler: torch.cuda.amp.GradScaler | None,
    rank: int = 0,
    log_interval: int = 100,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_move_loss = 0.0
    total_promo_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_promo_samples = 0
    total_promo_correct = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        tokens = batch['tokens'].to(device)
        move_targets = batch['move_id'].to(device)
        promo_targets = batch['promo_id'].to(device)
        legal_mask = batch['legal_mask'].to(device)
        is_promotion = batch['is_promotion'].to(device)

        # Forward pass
        if precision == 'bf16':
            autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        elif precision == 'fp16':
            autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16)
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            outputs = model(tokens)
            move_logits = outputs['move_logits']
            promo_logits = outputs['promo_logits']

            # Move loss (always computed)
            move_loss = masked_cross_entropy(move_logits, move_targets, legal_mask)

            # Promotion loss (only for promotion moves)
            promo_loss = torch.tensor(0.0, device=device)
            if is_promotion.any():
                promo_mask = is_promotion
                promo_loss = nn.functional.cross_entropy(
                    promo_logits[promo_mask],
                    promo_targets[promo_mask],
                )

            # Total loss
            loss = move_loss + promo_loss

        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Stats
        batch_size = tokens.size(0)
        total_loss += loss.item() * batch_size
        total_move_loss += move_loss.item() * batch_size
        total_samples += batch_size

        # Move accuracy (with masking)
        with torch.no_grad():
            masked_logits = move_logits.clone()
            masked_logits[~legal_mask] = float('-inf')
            preds = masked_logits.argmax(dim=-1)
            total_correct += (preds == move_targets).sum().item()

            # Promotion accuracy
            if is_promotion.any():
                promo_mask = is_promotion
                promo_preds = promo_logits[promo_mask].argmax(dim=-1)
                total_promo_correct += (promo_preds == promo_targets[promo_mask]).sum().item()
                total_promo_samples += promo_mask.sum().item()
                total_promo_loss += promo_loss.item() * promo_mask.sum().item()

        # Logging
        if rank == 0 and (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed
            avg_loss = total_loss / total_samples
            move_acc = total_correct / total_samples * 100

            print(f"  Batch {batch_idx + 1}: loss={avg_loss:.4f}, "
                  f"move_acc={move_acc:.1f}%, "
                  f"speed={samples_per_sec:.0f} samples/s")

    # Final stats
    stats = {
        'loss': total_loss / total_samples,
        'move_loss': total_move_loss / total_samples,
        'move_accuracy': total_correct / total_samples,
    }

    if total_promo_samples > 0:
        stats['promo_loss'] = total_promo_loss / total_promo_samples
        stats['promo_accuracy'] = total_promo_correct / total_promo_samples

    return stats


def main():
    parser = argparse.ArgumentParser(description='Train HumChess model')
    parser.add_argument('--config', type=str, default='configs/model.yml',
                        help='Path to YAML config file')
    # CLI overrides (optional)
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--num-workers', type=int, help='Override num workers')
    parser.add_argument('--checkpoint-dir', type=str, help='Override checkpoint dir')
    parser.add_argument('--precision', type=str, choices=['bf16', 'fp16', 'fp32'],
                        help='Override training precision')
    parser.add_argument('--tf32', action='store_true',
                        help='Enable TF32 matmul on CUDA')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    config = load_config(config_path)

    # Extract sections with defaults
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})

    # Apply CLI overrides
    batch_size = args.batch_size or train_cfg.get('batch_size', 256)
    lr = args.lr or train_cfg.get('lr', 1e-4)
    epochs = args.epochs or train_cfg.get('epochs', 1)
    num_workers = args.num_workers or train_cfg.get('num_workers', 4)
    checkpoint_dir = Path(args.checkpoint_dir or train_cfg.get('checkpoint_dir', 'checkpoints'))
    precision = args.precision or train_cfg.get('precision', 'bf16')
    log_interval = train_cfg.get('log_interval', 100)

    # Setup distributed
    rank, world_size, local_rank, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f"Config: {config_path}")
        print(f"Training with {world_size} GPU(s)")
        print(f"Batch size: {batch_size} (per GPU)")
        print(f"Learning rate: {lr}")
        print(f"Precision: {precision}")
        print(f"TF32: {args.tf32}")

    # Create model
    model_params = {key: int(model_cfg[key]) for key in ['d_model', 'n_heads', 'n_layers', 'd_ff']}
    model = ChessTransformer(**model_params)
    model = model.to(device)

    if rank == 0:
        print(f"Model params: {model_params}")
        print(f"Model parameters: {model.count_parameters():,}")

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = args.tf32

    scaler = None
    if precision == 'fp16' and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # Create dataset
    parquet_patterns = data_cfg.get('parquet', [])
    pgn_patterns = data_cfg.get('pgn', [])

    if parquet_patterns:
        parquet_paths = expand_globs(parquet_patterns)
        if rank == 0:
            print(f"Parquet files: {len(parquet_paths)} shards")
        dataset = PGNDataset.from_parquet(parquet_paths=parquet_paths)
    elif pgn_patterns:
        pgn_paths = expand_globs(pgn_patterns)
        if rank == 0:
            print(f"PGN files: {pgn_paths}")
        dataset = PGNDataset(
            pgn_paths=pgn_paths,
            min_elo=data_cfg.get('min_elo', 0),
            max_elo=data_cfg.get('max_elo', 4000),
        )
    else:
        raise ValueError("Config must specify data.parquet or data.pgn")

    # Create dataloader
    # Note: IterableDataset doesn't use sampler in the traditional sense
    # Workers handle sharding internally via get_worker_info()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{epochs}")

        stats = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            precision=precision,
            scaler=scaler,
            rank=rank,
            log_interval=log_interval,
        )

        if rank == 0:
            print(f"Epoch {epoch + 1} complete:")
            print(f"  Loss: {stats['loss']:.4f}")
            print(f"  Move accuracy: {stats['move_accuracy']*100:.1f}%")
            if 'promo_accuracy' in stats:
                print(f"  Promo accuracy: {stats['promo_accuracy']*100:.1f}%")

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f'epoch_{epoch + 1}.pt'
            model_to_save = model.module if is_distributed else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stats': stats,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    cleanup_distributed()

    if rank == 0:
        print("\nTraining complete!")


if __name__ == '__main__':
    main()
