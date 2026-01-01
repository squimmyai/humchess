"""
Training script with DDP support.

Usage:
    Single GPU:
        uv run python -m humchess.train --config configs/model.yml

    Multi-GPU:
        uv run torchrun --nproc_per_node=4 -m humchess.train --config configs/model.yml
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .data.pgn_dataset import PGNDataset
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


def get_samples_per_shard(paths: list[Path]) -> int:
    """Get samples per shard by reading metadata from first file."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(paths[0]).metadata.num_rows


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str,
    scaler: torch.cuda.amp.GradScaler | None,
    rank: int = 0,
    log_interval: int = 100,
    total_shards: int | None = None,
    samples_per_shard: int | None = None,
    use_wandb: bool = False,
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

    seen = 0  # for debug logging

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

        with torch.no_grad():
            target_is_legal = legal_mask.gather(1, move_targets.unsqueeze(1)).squeeze(1)
            illegal_target = ~target_is_legal
            bad_idx = torch.where(illegal_target)[0]

            if bad_idx.numel() > 0 and rank == 0:
                for i in bad_idx[:10].tolist():
                    sample_num = seen + i
                    print(f"[illegal_target] batch={batch_idx} in_batch={i} sample_num={sample_num}")

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

            progress_str = ""
            if total_shards and samples_per_shard:
                shards_done = total_samples / samples_per_shard
                pct = shards_done / total_shards * 100
                remaining_shards = total_shards - shards_done
                remaining_samples = remaining_shards * samples_per_shard
                eta_sec = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
                eta_min = eta_sec / 60
                if eta_min >= 60:
                    eta_str = f"{eta_min/60:.1f}h"
                else:
                    eta_str = f"{eta_min:.1f}m"
                progress_str = f" | shard ~{shards_done:.1f}/{total_shards} ({pct:.1f}%) ETA {eta_str}"

            print(f"  Batch {batch_idx + 1}: loss={avg_loss:.4f}, "
                  f"move_acc={move_acc:.1f}%, "
                  f"speed={samples_per_sec:.0f} samples/s{progress_str}")

            if use_wandb:
                log_dict = {
                    'train/loss': avg_loss,
                    'train/move_loss': total_move_loss / total_samples,
                    'train/move_accuracy': total_correct / total_samples,
                    'train/samples_per_sec': samples_per_sec,
                    'train/samples': total_samples,
                }
                if total_shards and samples_per_shard:
                    shards_done = total_samples / samples_per_shard
                    log_dict['train/shards_done'] = shards_done
                    log_dict['train/progress_pct'] = shards_done / total_shards * 100
                if total_promo_samples > 0:
                    log_dict['train/promo_loss'] = total_promo_loss / total_promo_samples
                    log_dict['train/promo_accuracy'] = total_promo_correct / total_promo_samples
                wandb.log(log_dict)

        seen += tokens.size(0)  # for debug logging

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
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster training')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='humchess',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name (auto-generated if not set)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
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
    lr = args.lr or float(train_cfg.get('lr', 1e-4))
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

    # Compile model for faster training
    if args.compile:
        if rank == 0:
            print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Initialize wandb
    use_wandb = args.wandb and rank == 0
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: --wandb specified but wandb not installed. Run: pip install wandb")
            use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    'model': model_params,
                    'model_parameters': model.count_parameters(),
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'epochs': epochs,
                    'precision': precision,
                    'tf32': args.tf32,
                    'num_workers': num_workers,
                    'world_size': world_size,
                },
            )
            print(f"W&B run: {wandb.run.url}")

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = args.tf32

    scaler = None
    if precision == 'fp16' and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    if is_distributed:
        # find_unused_parameters=True needed because promo_head is unused
        # when batch has no promotions
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Create dataset
    parquet_patterns = data_cfg.get('parquet', [])
    pgn_patterns = data_cfg.get('pgn', [])

    # Handle resume
    resume_checkpoint = None
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            start_epoch = resume_checkpoint.get('epoch', 0)
            if rank == 0:
                print(f"Resuming from: {resume_path}")
                print(f"  Starting from epoch {start_epoch + 1}")
        else:
            if rank == 0:
                print(f"Warning: Resume checkpoint not found: {resume_path}")

    total_shards = None
    samples_per_shard = None
    if parquet_patterns:
        parquet_paths = expand_globs(parquet_patterns)
        if rank == 0:
            print(f"Parquet files: {len(parquet_paths)} shards")
            samples_per_shard = get_samples_per_shard(parquet_paths)
            # Each rank sees ~1/world_size of the shards
            total_shards = len(parquet_paths) // world_size
            print(f"Total shards: {len(parquet_paths)} ({total_shards} per rank, ~{samples_per_shard:,} samples/shard)")
        dataset = PGNDataset.from_parquet(
            parquet_paths=parquet_paths,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
        )
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
    # Dataset yields pre-batched dicts, so batch_size=1 and no collation needed
    prefetch_factor = train_cfg.get('prefetch_factor', 4)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Dataset yields pre-batched data
        num_workers=num_workers,
        collate_fn=lambda x: x[0],  # Unwrap the single-item list
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)

    # Load model/optimizer state if resuming
    if resume_checkpoint:
        model_to_load = model.module if is_distributed else model
        model_to_load.load_state_dict(resume_checkpoint['model_state_dict'])
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if rank == 0:
            print("Loaded model and optimizer state from checkpoint")

    # Training loop
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(start_epoch, epochs):
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
            total_shards=total_shards,
            samples_per_shard=samples_per_shard,
            use_wandb=use_wandb,
        )

        if rank == 0:
            print(f"Epoch {epoch + 1} complete:")
            print(f"  Loss: {stats['loss']:.4f}")
            print(f"  Move accuracy: {stats['move_accuracy']*100:.1f}%")
            if 'promo_accuracy' in stats:
                print(f"  Promo accuracy: {stats['promo_accuracy']*100:.1f}%")

            if use_wandb:
                epoch_log = {
                    'epoch': epoch + 1,
                    'epoch/loss': stats['loss'],
                    'epoch/move_accuracy': stats['move_accuracy'],
                }
                if 'promo_accuracy' in stats:
                    epoch_log['epoch/promo_accuracy'] = stats['promo_accuracy']
                wandb.log(epoch_log)

            # Save end-of-epoch checkpoint
            # epoch is 0-based, so epoch+1 marks "completed epoch 1, ready for epoch 2"
            checkpoint_path = checkpoint_dir / f'epoch_{epoch + 1}.pt'
            model_to_save = model.module if is_distributed else model
            torch.save({
                'epoch': epoch + 1,  # Next epoch to start from
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stats': stats,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    cleanup_distributed()

    if use_wandb:
        wandb.finish()

    if rank == 0:
        print("\nTraining complete!")


if __name__ == '__main__':
    main()
