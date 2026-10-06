# Agent guide

See `CLAUDE.md` for the project overview and key files. This file covers
**how to run training on a fresh (GPU) server** and the operational gotchas we
hit doing it. Read this before launching anything on a new box.

## TL;DR (4x A100)

```bash
# on the server, from the repo root
source .venv/bin/activate                       # or any env with torch (CUDA build)
set -a; source .env.local; set +a               # R2 + W&B creds (gitignored)
mkdir -p checkpoints
tmux new -s train                                # survive SSH drops
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=4 -m humchess.train \
    --config configs/model.yml --num-workers 12 --compile --wandb
```

## 1. Prerequisites

- **GPUs + CUDA.** Confirmed on 4x A100-SXM4-40GB (driver 595.91, CUDA 12.8,
  torch 2.11.0+cu128, NCCL 2.28.9). The `torch` build must have kernels for the
  GPU architecture — A100 is compute capability 8.0 (any CUDA 12.x wheel works);
  Blackwell (cc >= 10.0) needs CUDA >= 12.8 wheels. A mismatched wheel installs
  fine then dies at the first GPU op with `no kernel image is available`.
- **Python.** `pyproject.toml` requires `>=3.13`; we have run it on 3.12, but see
  the packaging note in §2.
- **RAM/disk.** Data is streamed from S3/R2, so little disk is needed — but see
  §7 (checkpoints). Several hundred GB of host RAM is plenty but not required.

## 2. Install dependencies

`torch` comes from the environment (many GPU images preinstall it). Install the
rest:

```bash
python -m venv .venv && source .venv/bin/activate
pip install "pyarrow>=18" wandb python-chess zstandard tqdm
# or, if the env is uv-managed (Vast images):  uv pip install pyarrow wandb python-chess zstandard tqdm
```

`pip install -e .` is the canonical path, but PyArrow's S3 support is bundled in
the wheel, so **boto3/s3fs are NOT required**. If `install -e .` trips on the
`requires-python` pin, install the deps above directly instead.

## 3. Credentials (`.env.local`)

The code reads credentials straight from the environment — there is **no
dotenv loader**, so you must export them before running:

```bash
set -a; source .env.local; set +a
```

`.env.local` is gitignored. It needs:

```
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
WANDB_API_KEY=...            # optional, only for --wandb
```

`chmod 600 .env.local`. `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` are accepted
as fallbacks.

## 4. Data

Shards live at, and are streamed directly from:

```
s3://lightning-ai-storage/tokenized-2022-01/*.parquet
```

10,211 shards, ~590 GB total, ~664,577 rows/shard. **Nothing is downloaded to
disk** — each shard is read, decoded, and iterated. Config key is
`data.parquet` (a glob). Verify access with:

```bash
python -c "from humchess.data.pgn_dataset import expand_s3_glob as g; print(len(g('s3://lightning-ai-storage/tokenized-2022-01/*.parquet')))"
# -> 10211
```

## 5. Launch

Single GPU:

```bash
python -m humchess.train --config configs/model.yml
```

Multi-GPU (DDP via `torchrun`; run under `tmux`/`nohup`):

```bash
torchrun --nproc_per_node=4 -m humchess.train \
    --config configs/model.yml --num-workers 12 --compile --wandb
```

Useful flags:

| flag | why |
|---|---|
| `--compile` | `torch.compile` — **~2.7x faster** here, essentially required |
| `--compile-mode default\|reduce-overhead\|max-autotune` | default `reduce-overhead` |
| `--gradient-checkpointing` | fit a larger per-GPU batch (see §6); also config `model.gradient_checkpointing` |
| `--num-workers` | DataLoader workers per rank |
| `--keep-last-n-checkpoints` | rotate old checkpoints (see §7) |
| `--wandb --wandb-project humchess` | W&B logging (rank 0) |
| `--resume <ckpt.pt>` | resume model/optimizer/scheduler/progress |
| `--max-batches N` | stop early (smoke tests) |

Notes:
- `torchrun` sets `OMP_NUM_THREADS=1` per rank. That mostly matters for the CPU
  side of `torch.compile` (compilation is slower); steady-state is GPU-bound.
- `--compile` compiles once per rank; expect ~5-8 min of startup before step 1.
- Both the LR schedule length and validation split depend on the **shard count**,
  so keep `data.parquet` pointing at the full glob unless you know what you're doing.

## 6. Fitting the model in memory

The current 30.66M config (`d_model=384, n_heads=6, n_layers=14, d_ff=1536`,
SwiGLU + AdaLN) **OOMs at the intended per-GPU batch of 3000 on a 40 GB A100**
without gradient checkpointing. It fits comfortably with it:

| batch/GPU | checkpointing | peak VRAM |
|---|---|---|
| 1000 | off | 34.4 GB |
| 3000 | **on** | **13.3 GB** |

So `model.gradient_checkpointing: true` is set in `configs/model.yml`. It trades
~30% extra compute for a large memory cut. Increasing the batch beyond 3000 did
**not** improve throughput-per-sample for this model, so don't bother unless you
want fewer steps.

## 7. Checkpoints and disk

Checkpoints are written every `checkpoint_every_n_shards` shards (default 100)
and contain model + optimizer + scaler + scheduler + the completed-shard set
(resumable). Each is ~0.4 GB (fp32 params + AdamW state).

**A full epoch writes ~100 of them (~40 GB).** On a small instance disk this
fills up mid-run, so `keep_last_n_checkpoints` (config, default 3) deletes older
`checkpoint_shards_*.pt` after each save. Keep it set, or ensure `checkpoint_dir`
has tens of GB.

## 8. Expected performance (4x A100-40GB, 30.66M model)

- **~1.4 steps/s → ~17,000 samples/s** at global batch 12,000; every GPU at 100%.
- **~110 h (~4.6 days) per epoch** over the 10,211 shards (~6.8B positions).
- Effective MFU is only ~17% — the model is small (384-wide), so matmuls don't
  saturate an A100. This is expected; a bigger model would get better MFU.
- **`torch.compile` is the single biggest lever:** per-GPU 2,307 → 6,177
  samples/s. Without it the epoch is ~8.6 days.
- **Data loading is not the bottleneck.** Raw shard reads scale to >300k rows/s
  with 8 workers; converted batches cost ~40 ms/3000 rows. If GPUs aren't at
  100%, suspect compile/CPU before the data path.

## 9. Troubleshooting

- **CUDA OOM at startup** → enable gradient checkpointing (§6), and set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Disk fills mid-epoch** → `keep_last_n_checkpoints` (§7).
- **`[c10d] ... cannot be initialized to connect to [localhost]:...`** → harmless
  IPv6 fallback warning; training proceeds. Ignore it.
- **`ModuleNotFoundError: No module named 'humchess'`** → run from the repo root
  (or set `PYTHONPATH`). A bare `python /tmp/script.py` won't find the package.
- **`no kernel image is available`** → torch wheel built for the wrong GPU
  architecture (§1).
- **NCCL hangs / collective errors** → the trainer uses an equal-step loop so
  uneven shard counts don't deadlock, and validates `world_size <= n_shards`.
  If it still hangs, check that all ranks launched together and the default
  process-group timeout (30 min) is enough for your slowest startup.

## 10. Vast.ai specifics (if that's the host)

- **"stop" preserves the whole filesystem; "recycle"/"destroy" wipe it.** Only a
  mounted volume survives a recycle. Check
  `vast-capabilities | jq '.instance.workspace_is_volume'` — if `false`, nothing
  on disk is safe across recycle/destroy.
- `python` is not on `PATH` in non-interactive shells — `source /venv/main/bin/activate`.
- The instance prints an agent guide to read first: `/etc/vast-agents-guide.md`
  (supervisor, Caddy/portal ports, provisioning, CUDA notes).
