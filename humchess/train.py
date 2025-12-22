"""
Training script with DDP support.

Usage:
    Single GPU:
        python -m humchess.train --pgn data/*.pgn

    Multi-GPU:
        torchrun --nproc_per_node=4 -m humchess.train --pgn data/*.pgn
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data.pgn_dataset import PGNDataset, collate_fn
from .data.vocabulary import NUM_MOVE_CLASSES, NUM_PROMO_CLASSES
from .model.transformer import ChessTransformer, create_model


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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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
    parser.add_argument('--pgn', type=str, nargs='+', required=True,
                        help='Path(s) to PGN files')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--min-elo', type=int, default=0)
    parser.add_argument('--max-elo', type=int, default=4000)
    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f"Training with {world_size} GPU(s)")
        print(f"Model size: {args.model_size}")
        print(f"Batch size: {args.batch_size} (per GPU)")
        print(f"Learning rate: {args.lr}")
        print(f"PGN files: {args.pgn}")

    # Create model
    model = create_model(args.model_size)
    model = model.to(device)

    if rank == 0:
        print(f"Model parameters: {model.count_parameters():,}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # Create dataset
    dataset = PGNDataset(
        pgn_paths=args.pgn,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
    )

    # Create dataloader
    # Note: IterableDataset doesn't use sampler in the traditional sense
    # Workers handle sharding internally via get_worker_info()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

        stats = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            rank=rank,
            log_interval=args.log_interval,
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
