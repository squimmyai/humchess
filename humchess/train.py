"""
Training script with DDP support.

Usage:
    Single GPU:
        uv run python -m humchess.train --config configs/model.yml

    Multi-GPU:
        uv run torchrun --nproc_per_node=4 -m humchess.train --config configs/model.yml
"""

import argparse
import datetime
import glob
import os
import sys
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

from .data.pgn_dataset import PGNDataset, expand_s3_glob
from .model.transformer import ChessTransformer


# Default timeout for collective operations. Without this, a rank that exits
# early (or crashes) leaves its peers blocked forever in a collective. With it,
# the blocked ranks raise instead, turning silent hangs into visible errors.
DEFAULT_PG_TIMEOUT = datetime.timedelta(minutes=30)


def setup_distributed(timeout: datetime.timedelta = DEFAULT_PG_TIMEOUT):
    """Initialize distributed training if available."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                'nccl',
                device_id=torch.device(f'cuda:{local_rank}'),
                timeout=timeout,
            )
        else:
            # CPU fallback (e.g. CI / local multi-process tests). Uses gloo so
            # the DDP logic can be exercised without GPUs.
            dist.init_process_group('gloo', timeout=timeout)

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
    # Apply mask: set illegal logits to -inf (masked_fill avoids clone)
    masked_logits = logits.masked_fill(~mask, float('-inf'))

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


def expand_globs(patterns: list[str]) -> list[Path | str]:
    """Expand glob patterns and directories to file paths.

    Supports both local paths and S3 URLs (s3://bucket/path/*.parquet).
    Local paths are returned as Path objects, S3 URLs as strings.
    """
    paths: list[Path | str] = []
    for pattern in patterns:
        if pattern.startswith('s3://'):
            # S3 URL - use S3 glob expansion
            if '*' in pattern or '?' in pattern:
                paths.extend(expand_s3_glob(pattern))
            else:
                paths.append(pattern)
        else:
            # Local path
            p = Path(pattern)
            if p.is_dir():
                paths.extend(sorted(p.glob("*.parquet")))
            elif '*' in pattern or '?' in pattern:
                paths.extend(sorted(Path(m) for m in glob.glob(pattern)))
            else:
                paths.append(p)
    return paths


def get_samples_per_shard(paths: list[Path | str]) -> int:
    """Get samples per shard by reading metadata from first file.

    Supports both local paths and S3 URLs.
    """
    import pyarrow.parquet as pq
    from .data.pgn_dataset import _get_s3_filesystem, _parse_s3_path

    first_path = str(paths[0])
    if first_path.startswith('s3://'):
        fs = _get_s3_filesystem()
        bucket, key = _parse_s3_path(first_path)
        # Use read_table for R2 compatibility (avoids range request issues)
        table = pq.read_table(f"{bucket}/{key}", filesystem=fs)
        return table.num_rows
    else:
        return pq.ParquetFile(paths[0]).metadata.num_rows


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    completed_shards: set[str],
    is_distributed: bool,
    rank: int = 0,
    world_size: int = 1,
    gather_shards: bool = True,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    keep_last: int | None = None,
) -> None:
    """Save checkpoint with model, optimizer, and completed shards.

    Args:
        gather_shards: If True, gather completed_shards from all ranks (requires
            all ranks to call this function). If False, only rank 0 saves using
            its local completed_shards (safe for mid-epoch checkpoints).
        keep_last: If set, retain only the N most recent ``checkpoint_shards_*.pt``
            files in the checkpoint directory, deleting older ones. A full epoch
            writes one checkpoint per ``checkpoint_every_n_shards`` shards; without
            rotation these accumulate to far more than the disk holds (model +
            optimizer state is ~0.4 GB each).
    """
    # Gather completed shards from all ranks (only if requested and distributed)
    if gather_shards and is_distributed and world_size > 1:
        all_completed: list[list[str]] = [None] * world_size
        dist.all_gather_object(all_completed, list(completed_shards))
        # Flatten into single set
        all_shards = set()
        for rank_shards in all_completed:
            all_shards.update(rank_shards)
    else:
        all_shards = completed_shards

    # Only rank 0 saves (all ranks still participate in the gather above)
    if rank == 0 and all_shards:
        model_to_save = model.module if is_distributed else model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'completed_shards': list(all_shards),
        }
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path} ({len(all_shards)} shards completed)")

        # Rotate old checkpoints so a full epoch cannot exhaust the disk.
        if keep_last is not None and keep_last > 0:
            existing = sorted(
                path.parent.glob('checkpoint_shards_*.pt'),
                key=lambda p: int(p.stem.rsplit('_', 1)[-1]),
            )
            for stale in existing[:-keep_last]:
                try:
                    stale.unlink()
                    print(f"  Removed old checkpoint: {stale}")
                except OSError as e:
                    print(f"  Warning: could not remove {stale}: {e}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str,
    scaler: torch.cuda.amp.GradScaler | None,
    rank: int = 0,
    world_size: int = 1,
    log_interval: int = 100,
    total_shards: int | None = None,
    samples_per_shard: int | None = None,
    use_wandb: bool = False,
    max_batches: int | None = None,
    profiler: torch.profiler.profile | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every_n_shards: int | None = None,
    is_distributed: bool = False,
    initial_completed_shards: set[str] | None = None,
    debug_batches: bool = False,
    val_loader=None,
    eval_interval: int | None = None,
    val_max_batches: int = 20,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_grad_norm: float | None = 1.0,
    keep_last_n_checkpoints: int | None = None,
    grad_accum_steps: int = 1,
) -> dict:
    """Train for one epoch."""
    model.train()

    # Accumulate stats on GPU to avoid CPU-GPU sync every batch
    total_loss_t = torch.tensor(0.0, device=device)
    total_move_loss_t = torch.tensor(0.0, device=device)
    total_promo_loss_t = torch.tensor(0.0, device=device)
    total_samples_t = torch.tensor(0, device=device, dtype=torch.long)
    total_correct_t = torch.tensor(0, device=device, dtype=torch.long)
    total_promo_samples_t = torch.tensor(0, device=device, dtype=torch.long)
    total_promo_correct_t = torch.tensor(0, device=device, dtype=torch.long)

    # Shard-based checkpointing (include previously completed shards from resume)
    completed_shards: set[str] = set(initial_completed_shards or [])
    initial_shard_count = len(completed_shards)  # For computing new shards in all_reduce
    last_checkpoint_shard_count = initial_shard_count

    start_time = time.time()

    # ------------------------------------------------------------------
    # Equal-step loop.
    #
    # Ranks receive different shards, and shards have different row counts,
    # so ranks would otherwise run a different number of batches. Any
    # collective issued from inside the loop (logging all_reduce, checkpoint
    # all_reduce, save_checkpoint's all_gather_object) would then be called a
    # different number of times per rank and deadlock.
    #
    # Fix: run a fixed number of steps on every rank. A rank whose data runs
    # out keeps stepping on the last real batch with the loss zeroed, so it
    # still performs the full forward/backward (keeping DDP's gradient
    # allreduce in lockstep) but contributes zero gradient. Ranks stop only
    # when a collective check says *all* are exhausted, so the iteration count
    # is identical across ranks by construction.
    #
    # model.join() is intentionally NOT used: it only shadows DDP's own
    # gradient allreduces, not the user-level collectives in this loop.
    # ------------------------------------------------------------------
    data_iter = iter(dataloader)
    dummy_batch: dict | None = None
    exhausted = False
    step = 0

    # Autocast context is fixed for the run, so build it once.
    if precision == 'bf16':
        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    elif precision == 'fp16':
        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    while True:
        # ------------------------------------------------------------------
        # One optimizer step = grad_accum_steps micro-batches.
        #
        # The effective per-GPU batch is micro_batch * grad_accum_steps; each
        # micro-batch is forwarded and backwarded in turn, with gradients
        # summed into .grad, and the optimizer steps once at the end. The
        # equal-step invariant (every rank runs the same number of optimizer
        # steps and issues the same collectives) is preserved: a rank that runs
        # out of data pads *micro-batches* with its last real batch and a zeroed
        # loss, so the boundary count is identical on every rank.
        # ------------------------------------------------------------------
        optimizer.zero_grad()
        # Stats accumulated on-device (set to None here, summed then consumed)
        # so no GPU->CPU sync happens inside the micro-batch loop.
        step_loss_t = None
        step_move_loss_t = None
        step_promo_loss_t = None
        step_samples = 0
        step_promos_t = None
        step_correct_t = None
        step_promo_correct_t = None
        boundary_shard_complete = False
        boundary_shard_path = None

        for micro_idx in range(grad_accum_steps):
            if not exhausted:
                batch = next(data_iter, None)
                if batch is None:
                    exhausted = True
            if exhausted:
                if dummy_batch is None:
                    # This rank never produced a single batch. Topology
                    # validation in main() should prevent this; fail loudly
                    # rather than hang.
                    raise RuntimeError(
                        f"rank {rank} received no data batches; check world_size vs shard count"
                    )
                # Single-process run: nothing to synchronize with, so stop here
                # rather than doing pointless zero-loss padded micro-batches.
                if not (is_distributed and world_size > 1):
                    break
                batch = dummy_batch
            else:
                dummy_batch = batch

            tokens = batch['tokens'].to(device, non_blocking=True)
            move_targets = batch['move_id'].to(device, non_blocking=True)
            promo_targets = batch['promo_id'].to(device, non_blocking=True)
            legal_mask = batch['legal_mask'].to(device, non_blocking=True)
            is_promotion = batch['is_promotion'].to(device, non_blocking=True)

            # Debug logging (gated behind --debug-batches flag due to perf cost)
            if debug_batches:
                shard_path = batch.get('shard_path', 'unknown')
                shard_name = shard_path.split('/')[-1] if isinstance(shard_path, str) and '/' in shard_path else shard_path
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                print(f"[BATCH] rank={rank} step={step} micro={micro_idx} shard={shard_name}",
                      file=sys.stderr, flush=True)

            with autocast_ctx:
                outputs = model(tokens)
                move_logits = outputs['move_logits']
                promo_logits = outputs['promo_logits']

                # Move loss (always computed)
                move_loss = masked_cross_entropy(move_logits, move_targets, legal_mask)

                # Promotion loss (only for promotion moves)
                safe_promo_targets = promo_targets.clamp(0, 3)
                promo_loss_per_sample = nn.functional.cross_entropy(
                    promo_logits,
                    safe_promo_targets,
                    reduction='none',
                )
                num_promos = is_promotion.sum().clamp(min=1)
                promo_loss = (promo_loss_per_sample * is_promotion.float()).sum() / num_promos

                loss = move_loss + promo_loss

            # Zero the loss on padded micro-batches so an exhausted rank
            # contributes no gradient while still participating in DDP's
            # gradient allreduce.
            if exhausted:
                loss = loss * 0.0

            # Divide by grad_accum_steps so the summed gradient equals the mean
            # over the full effective batch (what a single large batch would
            # give), keeping gradient clipping and the optimizer math identical.
            loss = loss / grad_accum_steps

            # DDP allreduce only on the final micro-batch; earlier ones
            # accumulate local gradients via no_sync().
            sync_now = (micro_idx == grad_accum_steps - 1) or not (is_distributed and world_size > 1)
            if scaler is not None:
                with (model.no_sync() if not sync_now else nullcontext()):
                    scaler.scale(loss).backward()
            else:
                with (model.no_sync() if not sync_now else nullcontext()):
                    loss.backward()

            # Accumulate stats on-device for the whole effective batch. `loss`
            # is already divided by grad_accum_steps, so multiply back to recover
            # the per-micro-batch mean for a correct average.
            if not exhausted:
                with torch.no_grad():
                    b = tokens.size(0)
                    step_samples += b
                    np_t = is_promotion.sum()
                    masked_logits = move_logits.masked_fill(~legal_mask, float('-inf'))
                    promo_preds = promo_logits.argmax(dim=-1)
                    contribs = (
                        (loss.detach() * grad_accum_steps + 0.0) * b,
                        move_loss.detach() * b,
                        promo_loss.detach() * np_t,
                        np_t,
                        (masked_logits.argmax(dim=-1) == move_targets).sum(),
                        ((promo_preds == promo_targets) & is_promotion).sum(),
                    )
                    if step_loss_t is None:
                        (
                            step_loss_t, step_move_loss_t, step_promo_loss_t,
                            step_promos_t, step_correct_t, step_promo_correct_t,
                        ) = contribs
                    else:
                        (
                            step_loss_t, step_move_loss_t, step_promo_loss_t,
                            step_promos_t, step_correct_t, step_promo_correct_t,
                        ) = [a + c for a, c in zip(
                            (step_loss_t, step_move_loss_t, step_promo_loss_t,
                             step_promos_t, step_correct_t, step_promo_correct_t),
                            contribs,
                        )]
                # Track shard completion across the boundary.
                if batch.get('shard_complete', False):
                    boundary_shard_complete = True
                    boundary_shard_path = batch.get('shard_path')

        else:
            # for-loop completed all micro-batches (not broken by single-process
            # exhaustion): do the optimizer step.
            if scaler is not None:
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Fold the accumulated boundary stats into the GPU stat tensors
            # (stays on-device; the big sync happens at the logging boundary).
            if step_samples > 0:
                total_loss_t += step_loss_t
                total_move_loss_t += step_move_loss_t
                total_promo_loss_t += step_promo_loss_t
                total_samples_t += step_samples
                total_correct_t += step_correct_t
                total_promo_samples_t += step_promos_t
                total_promo_correct_t += step_promo_correct_t

            if boundary_shard_complete and boundary_shard_path:
                completed_shards.add(boundary_shard_path)

        # Single-process exhaustion broke out of the micro loop; stop the epoch.
        if exhausted and not (is_distributed and world_size > 1):
            break

        # One-time guard: fp32 master weights means the optimizer must never
        # hold bf16 state. Catches accidental re-introduction of a model.to(bf16).
        if step == 0:
            p0 = next(iter(optimizer.state.values()), None)
            if p0 is not None:
                assert all(
                    t.dtype == torch.float32
                    for t in p0.values()
                    if isinstance(t, torch.Tensor)
                ), "optimizer state is not fp32; fp32 master weights broken"

        # Step profiler
        if profiler is not None:
            profiler.step()

        step += 1

        # Early stop for testing. max_batches is a global step budget, so all
        # ranks stop at the same step (equal iteration count preserved).
        if max_batches is not None and step >= max_batches:
            if rank == 0:
                print(f"Stopping after {max_batches} steps (--max-batches)")
            break

        # Logging / checkpointing / termination check, every log_interval
        # steps. Every rank reaches these boundaries because steps are
        # equalized by the padding above, so all collectives below are matched.
        if step % log_interval != 0:
            continue

        # Collective termination check: stop only when every rank is exhausted.
        if is_distributed and world_size > 1:
            done_t = torch.tensor(1 if exhausted else 0, device=device)
            dist.all_reduce(done_t, op=dist.ReduceOp.MAX)
            all_done = bool(done_t.item())
        else:
            all_done = exhausted
        if all_done:
            break

        # Reduce sample count across all ranks for accurate global throughput
        global_samples_t = total_samples_t.clone()
        if world_size > 1:
            dist.all_reduce(global_samples_t, op=dist.ReduceOp.SUM)

        # Checkpoint based on global shard count (all ranks sync here)
        if checkpoint_every_n_shards is not None and checkpoint_dir is not None:
            # Only count NEW shards (exclude initial to avoid double-counting on resume)
            local_new_shards = torch.tensor(len(completed_shards) - initial_shard_count, device=device)
            if world_size > 1:
                dist.all_reduce(local_new_shards, op=dist.ReduceOp.SUM)
            global_shard_count = initial_shard_count + local_new_shards.item()

            # All ranks see same global_shard_count, make same decision
            if global_shard_count >= last_checkpoint_shard_count + checkpoint_every_n_shards:
                checkpoint_path = checkpoint_dir / f'checkpoint_shards_{int(global_shard_count)}.pt'
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    completed_shards=completed_shards,
                    is_distributed=is_distributed,
                    rank=rank,
                    world_size=world_size,
                    gather_shards=True,
                    scheduler=scheduler,
                    keep_last=keep_last_n_checkpoints,
                )
                last_checkpoint_shard_count = global_shard_count

        if rank == 0:
            # Single sync point: pull all stats from GPU
            global_samples = global_samples_t.item()
            total_samples = total_samples_t.item()
            total_loss = total_loss_t.item()
            total_move_loss = total_move_loss_t.item()
            total_correct = total_correct_t.item()
            total_promo_samples = total_promo_samples_t.item()
            total_promo_correct = total_promo_correct_t.item()
            total_promo_loss = total_promo_loss_t.item()

            elapsed = time.time() - start_time
            samples_per_sec = global_samples / elapsed
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            move_acc = total_correct / total_samples * 100 if total_samples > 0 else 0.0

            progress_str = ""
            if total_shards and samples_per_shard:
                shards_done = global_samples / samples_per_shard
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

            print(f"  Step {step}: loss={avg_loss:.4f}, "
                f"move_acc={move_acc:.1f}%, "
                f"speed={samples_per_sec:.0f} samples/s{progress_str}")

            if use_wandb and total_samples > 0:
                log_dict = {
                    'train/loss': avg_loss,
                    'train/move_loss': total_move_loss / total_samples,
                    'train/move_accuracy': total_correct / total_samples,
                    'train/samples_per_sec': samples_per_sec,
                    'train/samples': global_samples,
                }
                if total_shards and samples_per_shard:
                    shards_done = global_samples / samples_per_shard
                    log_dict['train/shards_done'] = shards_done
                    log_dict['train/progress_pct'] = shards_done / total_shards * 100
                if total_promo_samples > 0:
                    log_dict['train/promo_loss'] = total_promo_loss / total_promo_samples
                    log_dict['train/promo_accuracy'] = total_promo_correct / total_promo_samples
                wandb.log(log_dict)

            # Periodic validation (rank 0 only; evaluate() has no collectives,
            # so other ranks simply wait at the next collective boundary).
            if val_loader is not None and eval_interval and step % eval_interval == 0:
                val_stats = evaluate(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    precision=precision,
                    max_batches=val_max_batches,
                )
                if val_stats:
                    print(f"  [val] Step {step}: loss={val_stats['val_loss']:.4f}, "
                        f"move_acc={val_stats['val_move_accuracy']*100:.1f}%")
                    if use_wandb:
                        wandb.log({
                            'val/loss': val_stats['val_loss'],
                            'val/move_loss': val_stats['val_move_loss'],
                            'val/move_accuracy': val_stats['val_move_accuracy'],
                        })

    # Final stats - sync from GPU once at end
    total_samples = total_samples_t.item()
    total_loss = total_loss_t.item()
    total_move_loss = total_move_loss_t.item()
    total_correct = total_correct_t.item()
    total_promo_samples = total_promo_samples_t.item()
    total_promo_correct = total_promo_correct_t.item()
    total_promo_loss = total_promo_loss_t.item()

    denom = max(total_samples, 1)
    stats = {
        'loss': total_loss / denom,
        'move_loss': total_move_loss / denom,
        'move_accuracy': total_correct / denom,
        'completed_shards': completed_shards,
    }

    if total_promo_samples > 0:
        stats['promo_loss'] = total_promo_loss / total_promo_samples
        stats['promo_accuracy'] = total_promo_correct / total_promo_samples

    return stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    precision: str = 'bf16',
    max_batches: int | None = 20,
) -> dict:
    """Evaluate on held-out data.

    Runs a bounded number of batches, so validation cost is fixed and small
    relative to an epoch. Performs no collectives - callers run it on a single
    rank - so it cannot affect the distributed step alignment of training.
    """
    model.eval()

    total_loss = 0.0
    total_move_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        tokens = batch['tokens'].to(device, non_blocking=True)
        move_targets = batch['move_id'].to(device, non_blocking=True)
        legal_mask = batch['legal_mask'].to(device, non_blocking=True)

        if precision == 'bf16':
            autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        elif precision == 'fp16':
            autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16)
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            outputs = model(tokens)
            move_loss = masked_cross_entropy(outputs['move_logits'], move_targets, legal_mask)

        n = tokens.size(0)
        total_loss += move_loss.item() * n
        total_move_loss += move_loss.item() * n
        masked_logits = outputs['move_logits'].masked_fill(~legal_mask, float('-inf'))
        total_correct += (masked_logits.argmax(dim=-1) == move_targets).sum().item()
        total_samples += n

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    model.train()

    if total_samples == 0:
        return {}
    return {
        'val_loss': total_loss / total_samples,
        'val_move_loss': total_move_loss / total_samples,
        'val_move_accuracy': total_correct / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Train HumChess model')
    parser.add_argument('--config', type=str, default='configs/model.yml',
                        help='Path to YAML config file')
    # CLI overrides (optional)
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--num-workers', type=int, help='Override num workers')
    parser.add_argument('--checkpoint-dir', type=str, help='Override checkpoint dir')
    parser.add_argument('--keep-last-n-checkpoints', type=int, default=None,
                        help='Retain only the N most recent shard checkpoints (disk safety)')
    parser.add_argument('--checkpoint-every-n-shards', type=int, default=None,
                        help='Save checkpoint every N shards (default: from config or 100)')
    parser.add_argument('--precision', type=str, choices=['bf16', 'fp16', 'fp32'],
                        help='Override training precision')
    parser.add_argument('--tf32', action='store_true',
                        help='Enable TF32 matmul on CUDA')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster training')
    parser.add_argument('--compile-mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='humchess',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name (auto-generated if not set)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Stop after N batches (for testing)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable torch.profiler (outputs Chrome trace)')
    parser.add_argument('--profile-batches', type=int, default=20,
                        help='Number of batches to profile (default: 20)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Recompute block activations in backward to fit a larger batch')
    parser.add_argument('--grad-accum-steps', type=int, default=None,
                        help='Split the per-GPU batch into this many micro-batches '
                             '(effective batch size unchanged; lower peak memory)')
    parser.add_argument('--debug-batches', action='store_true',
                        help='Log every batch with CUDA sync (slow, for debugging crashes)')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    config = load_config(config_path)

    # Extract sections with defaults
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})

    # Apply CLI overrides (use 'is not None' for values where 0 is valid)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get('batch_size', 256)
    lr = args.lr if args.lr is not None else float(train_cfg.get('lr', 1e-4))
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get('num_workers', 4)
    checkpoint_dir = Path(args.checkpoint_dir or train_cfg.get('checkpoint_dir', 'checkpoints'))
    precision = args.precision or train_cfg.get('precision', 'bf16')
    log_interval = train_cfg.get('log_interval', 100)
    checkpoint_every_n_shards = args.checkpoint_every_n_shards if args.checkpoint_every_n_shards is not None else train_cfg.get('checkpoint_every_n_shards', 100)
    keep_last_n_checkpoints = args.keep_last_n_checkpoints if args.keep_last_n_checkpoints is not None else train_cfg.get('keep_last_n_checkpoints', 3)
    # Gradient accumulation: `batch_size` is the effective per-GPU batch (so the
    # LR schedule is defined in terms of it), split into `grad_accum_steps`
    # micro-batches that are forwarded/backwarded in sequence with gradients
    # summed in .grad, then a single optimizer step. This decouples the memory
    # footprint (set by the micro-batch) from the effective batch size.
    grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps is not None else int(train_cfg.get('grad_accum_steps', 1))
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    if batch_size % grad_accum_steps != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by grad_accum_steps ({grad_accum_steps})"
        )
    micro_batch = batch_size // grad_accum_steps
    warmup_steps = int(train_cfg.get('warmup_steps', 0))
    min_lr_ratio = float(train_cfg.get('min_lr_ratio', 0.1))
    max_grad_norm = float(train_cfg.get('max_grad_norm', 1.0))
    # Total optimizer steps, needed for cosine decay. Estimated from the number
    # of training shards; if unknown (PGN mode) fall back to a large constant.
    max_steps = train_cfg.get('max_steps')

    # Setup distributed
    rank, world_size, local_rank, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f"Config: {config_path}")
        print(f"Training with {world_size} GPU(s)")
        print(f"Batch size: {batch_size} (per GPU, effective)")
        if grad_accum_steps > 1:
            print(f"Gradient accumulation: {grad_accum_steps} micro-batches of {micro_batch}")
        print(f"Learning rate: {lr}")
        print(f"Precision: {precision}")
        print(f"TF32: {args.tf32}")
        print(f"Checkpoint every: {checkpoint_every_n_shards} shards")

    # Create model
    model_params = {key: int(model_cfg[key]) for key in ['d_model', 'n_heads', 'n_layers', 'd_ff']}
    model_params['d_head'] = int(model_cfg.get('d_head', 64))
    model_params['use_adaln'] = bool(model_cfg.get('use_adaln', True))
    model_params['d_cond'] = int(model_cfg.get('d_cond', 128))
    model_params['ffn_type'] = str(model_cfg.get('ffn_type', 'silu'))
    model_params['gradient_checkpointing'] = bool(
        args.gradient_checkpointing or model_cfg.get('gradient_checkpointing', False)
    )
    model = ChessTransformer(**model_params)
    model = model.to(device)

    # NOTE: the model is intentionally kept in fp32 ("fp32 master weights").
    # bf16 autocast casts activations/params to bf16 per-op for compute, but the
    # parameters and optimizer state stay fp32. Permanently casting the model to
    # bf16 (as before) also made AdamW's moment buffers bf16; with weight
    # magnitudes ~0.02 the per-step update (~lr=1e-4) is roughly one bf16 ulp,
    # so informative updates were rounded away and the optimizer carried ~0.4%
    # relative error. The custom RMSNorm computes in fp32, so there is no
    # autocast dtype mismatch.

    if rank == 0:
        print(f"Model params: {model_params}")
        print(f"Model parameters: {model.count_parameters():,}")

    # Compile model for faster training
    if args.compile:
        compile_mode = args.compile_mode
        # CUDA-graph modes ("reduce-overhead"/"max-autotune") reuse static output
        # buffers across replays. With gradient accumulation we call the model
        # several times per optimizer step and keep references to earlier
        # outputs until the step ends, so a subsequent replay overwrites them
        # ("accessing tensor output of CUDAGraphs that has been overwritten").
        # "default" (Inductor, no CUDA graphs) has no such constraint and,
        # measured here, is just as fast for this model.
        if grad_accum_steps > 1 and compile_mode in ('reduce-overhead', 'max-autotune'):
            if rank == 0:
                print(f"Note: grad_accum_steps={grad_accum_steps} is incompatible with "
                      f"torch.compile mode '{compile_mode}'; using 'default' instead.")
            compile_mode = 'default'
        if rank == 0:
            print(f"Compiling model with torch.compile(mode='{compile_mode}')...")
        model = torch.compile(model, mode=compile_mode, fullgraph=True)

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
                    'checkpoint_every_n_shards': checkpoint_every_n_shards,
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
        model = DDP(model, device_ids=[local_rank])

    # Create dataset
    parquet_patterns = data_cfg.get('parquet', [])
    pgn_patterns = data_cfg.get('pgn', [])

    # Handle resume
    resume_checkpoint = None
    skip_shards: set[str] = set()
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            skip_shards = set(resume_checkpoint.get('completed_shards', []))
            if rank == 0:
                print(f"Resuming from: {resume_path}")
                print(f"  Skipping {len(skip_shards)} completed shards")
        else:
            if rank == 0:
                print(f"Warning: Resume checkpoint not found: {resume_path}")

    total_shards = None
    samples_per_shard = None
    val_loader = None
    if parquet_patterns:
        parquet_paths = expand_globs(parquet_patterns)

        # Hold out a fraction of shards for validation. Without this the loop
        # only ever reports training metrics, so every architecture/optimizer
        # A/B would measure fit rather than generalization. The split is taken
        # deterministically from the end of the sorted shard list and is the
        # same on every rank (so sharding stays consistent).
        val_fraction = float(train_cfg.get('val_fraction', 0.0))
        val_shards = []
        if val_fraction > 0:
            n_val = max(1, int(len(parquet_paths) * val_fraction))
            if n_val >= len(parquet_paths):
                raise ValueError(
                    f"val_fraction={val_fraction} would hold out all {len(parquet_paths)} shards"
                )
            val_shards = parquet_paths[-n_val:]
            parquet_paths = parquet_paths[:-n_val]

        # Topology validation: every rank must receive at least one shard after
        # rank sharding, and every rank's shards must cover at least one worker.
        # Otherwise a rank produces zero batches and can never join the
        # collectives, which previously manifested as a hang.
        n_files = len(parquet_paths)
        if world_size > n_files:
            raise ValueError(
                f"world_size={world_size} > number of training parquet shards={n_files}; "
                f"some ranks would receive no data and deadlock. "
                f"Reduce world_size or add shards."
            )
        # Note: idle DataLoader workers (when num_workers > shards-per-rank) are
        # harmless - they simply yield nothing - so only the rank-level check
        # above is a hard requirement.
        # Computed on ALL ranks (not just rank 0) because the LR schedule and
        # throughput/ETA estimates must be identical across ranks.
        total_shards = len(parquet_paths)
        samples_per_shard = get_samples_per_shard(parquet_paths)
        if rank == 0:
            print(f"Parquet files: {len(parquet_paths)} train shards, {len(val_shards)} val shards")
            remaining_shards = total_shards - len(skip_shards)
            print(f"Total shards: {total_shards} (~{samples_per_shard:,} samples/shard)")
            if skip_shards:
                print(f"Remaining shards: {remaining_shards}")

        if val_shards:
            # Validation uses the micro-batch size so a forward pass cannot OOM
            # when the effective batch is large but only the micro-batch fits.
            val_dataset = PGNDataset.from_parquet(
                parquet_paths=val_shards,
                rank=0,
                world_size=1,
                batch_size=micro_batch,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                num_workers=0,
                collate_fn=lambda x: x[0],
            )

        dataset = PGNDataset.from_parquet(
            parquet_paths=parquet_paths,
            rank=rank,
            world_size=world_size,
            batch_size=micro_batch,
            skip_shards=skip_shards,
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

    # LR schedule: linear warmup then cosine decay to min_lr_ratio * lr.
    # Transformers benefit from warmup (early gradients are large/unstable) and
    # from decay (large constant LR late in training adds noise near the
    # optimum). max_steps is estimated from shard count when not given.
    scheduler = None
    if warmup_steps > 0 or min_lr_ratio < 1.0:
        if max_steps is None and parquet_patterns:
            # Approximate number of optimizer steps this rank will run:
            # (training shards / world_size) * (rows/shard / batch_size).
            try:
                est_batches_per_shard = max(1, samples_per_shard // batch_size) if samples_per_shard else 1
                shards_per_rank = max(1, total_shards // world_size)
                max_steps = max(1, shards_per_rank * est_batches_per_shard)
            except Exception:
                max_steps = 1_000_000
        if max_steps is None:
            max_steps = 1_000_000

        import math

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if rank == 0:
            print(f"LR schedule: warmup {warmup_steps} steps, cosine to "
                  f"{min_lr_ratio:.0%} of {lr:g} over ~{max_steps} steps")

    # Load model/optimizer state if resuming
    if resume_checkpoint:
        model_to_load = model.module if is_distributed else model
        model_to_load.load_state_dict(resume_checkpoint['model_state_dict'])
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if scaler is not None and 'scaler_state_dict' in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        if rank == 0:
            print("Loaded model and optimizer state from checkpoint")

    # Training loop
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Setup profiler if requested
    profiler = None
    if args.profile and rank == 0:
        # Skip first few batches (warmup), then profile
        wait_batches = 5
        warmup_batches = 3
        active_batches = args.profile_batches
        profile_dir = Path('profiler_output')
        profile_dir.mkdir(exist_ok=True)

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait_batches,
                warmup=warmup_batches,
                active=active_batches,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()
        print(f"Profiler enabled: skip {wait_batches}, warmup {warmup_batches}, profile {active_batches} batches")
        print(f"Output: {profile_dir}/")
        # Override max_batches to stop after profiling
        if args.max_batches is None:
            args.max_batches = wait_batches + warmup_batches + active_batches + 5

    if rank == 0:
        print(f"\nStarting training (checkpoint every {checkpoint_every_n_shards} shards)")

    # If any rank raises (corrupt data, OOM, NCCL error), tear down the process
    # group so peers fail fast instead of blocking forever in a collective.
    try:
        stats = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            precision=precision,
            scaler=scaler,
            rank=rank,
            world_size=world_size,
            log_interval=log_interval,
            total_shards=total_shards,
            samples_per_shard=samples_per_shard,
            use_wandb=use_wandb,
            max_batches=args.max_batches,
            profiler=profiler,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_n_shards=checkpoint_every_n_shards,
            is_distributed=is_distributed,
            initial_completed_shards=skip_shards,
            debug_batches=args.debug_batches,
            val_loader=val_loader,
            eval_interval=train_cfg.get('eval_interval'),
            val_max_batches=train_cfg.get('val_max_batches', 20),
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
            grad_accum_steps=grad_accum_steps,
        )
    except Exception:
        if is_distributed:
            try:
                dist.abort()
            except Exception:
                pass
        raise

    if rank == 0:
        print(f"\nTraining pass complete:")
        print(f"  Loss: {stats['loss']:.4f}")
        print(f"  Move accuracy: {stats['move_accuracy']*100:.1f}%")
        if 'promo_accuracy' in stats:
            print(f"  Promo accuracy: {stats['promo_accuracy']*100:.1f}%")
        print(f"  Shards completed: {len(stats['completed_shards'])}")

    # Save final checkpoint. ALL ranks must call this unconditionally because
    # save_checkpoint performs an all_gather_object across ranks; only rank 0
    # actually writes, and it skips writing if no shards completed. Guarding
    # the call itself with `if completed_shards` (as before) would hang any
    # rank whose set happened to be empty.
    final_checkpoint_path = checkpoint_dir / 'checkpoint_final.pt'
    save_checkpoint(
        path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        completed_shards=stats['completed_shards'],
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        scheduler=scheduler,
    )

    if profiler is not None:
        profiler.stop()
        print(f"\nProfiler trace saved to: profiler_output/")
        print("View with: tensorboard --logdir=profiler_output")

    cleanup_distributed()

    if use_wandb:
        wandb.finish()

    if rank == 0:
        print("\nTraining complete!")


if __name__ == '__main__':
    main()
