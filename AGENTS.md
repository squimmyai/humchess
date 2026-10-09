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
| `--grad-accum-steps N` | split the per-GPU batch into N micro-batches (see §6); also config `training.grad_accum_steps` |
| `--gradient-checkpointing` | recompute block activations (alternative to accumulation; costs compute) |
| `--num-workers` | DataLoader workers per rank |
| `--keep-last-n-checkpoints` | rotate old checkpoints (see §7) |
| `--wandb --wandb-project humchess` | W&B logging (rank 0) |
| `--resume <ckpt.pt>` | resume model/optimizer/scheduler/progress |
| `--anneal-steps N` | on resume, cosine from the checkpoint's current LR down to `min_lr_ratio` over N steps, then hold (shortens the tail; pair with `--max-batches N`) |
| `--max-batches N` | stop early (smoke tests / anneal end; also switches the progress line to run-based) |

Notes:
- `torchrun` sets `OMP_NUM_THREADS=1` per rank. That mostly matters for the CPU
  side of `torch.compile` (compilation is slower); steady-state is GPU-bound.
- `--compile` compiles once per rank; expect ~5-8 min of startup before step 1.
- Both the LR schedule length and validation split depend on the **shard count**,
  so keep `data.parquet` pointing at the full glob unless you know what you're doing.

## 6. Fitting the model in memory (gradient accumulation)

The 30.66M config (`d_model=384, n_heads=6, n_layers=14, d_ff=1536`, SwiGLU +
AdaLN) **cannot forward a per-GPU batch of 3000 on a 40 GB A100** — activations
peak above 39 GB. Measured (single A100, no checkpointing):

| batch/GPU | peak VRAM |
|---|---|
| 800 | 27.9 GB |
| 1000 | 34.7 GB |
| 1200 | 41.5 GB (OOM) |
| 1500 | OOM |

Two ways to fit it, and we chose the second:

1. **Gradient checkpointing** (`gradient_checkpointing: true`): recompute block
   activations in backward. Fits batch 3000 at 13 GB, but costs **~33% compute**.
2. **Gradient accumulation** (`grad_accum_steps: N`): split the effective batch
   into N micro-batches, summing gradients and stepping once. Peak memory is set
   by the *micro*-batch, and there is **no recompute cost**.

`configs/model.yml` uses **`batch_size: 3000, grad_accum_steps: 3`** →
micro-batch 1000, which fits, with `gradient_checkpointing: false`. The optimizer
sees the same effective batch and the same LR schedule either way. Measured
per-GPU throughput (batch 3000, `--compile`):

Throughput depends mainly on the **micro-batch size** (GPU efficiency), not on
the accumulation count. Measured global throughput on 4x A100-PCIE at effective
batch 12000, `--compile-mode default`:

| micro-batch | accum | global samples/s | peak/GPU |
|---|---|---|---|
| 750 | 4 | 14,300 | 14.0 GB |
| 1000 | 3 | 17,000 | 18.5 GB |
| **1500** | **2 (chosen)** | **20,600** | **27.4 GB** |

The accumulation count barely matters (accum 3/6/12 at micro 1000 all ~17k), so
the DDP allreduce is not the bottleneck — split as little as memory allows.
Micro-batch 2000 reaches ~23k/s but leaves only ~3 GB headroom and changes the
effective batch; 1500/2 is the safe pick that preserves the recipe.

`--compile-mode reduce-overhead` / `max-autotune` use CUDA graphs that reuse
output buffers; with accumulation (several forwards per step whose outputs are
all read at the step boundary) a replay overwrites an earlier output and raises
`accessing tensor output of CUDAGraphs that has been overwritten`. The trainer
auto-downgrades to `default` when `grad_accum_steps > 1`, and `default` is the
default mode. Left as a note here because it isn't obvious from the error.

`grad_accum_steps` must divide `batch_size`. `accum=1` reproduces the plain
single-batch path exactly. Gradient equivalence is covered by
`tests/test_grad_accum.py`.

## 7. Checkpoints and disk

Checkpoints are written every `checkpoint_every_n_shards` shards (default 100)
and contain model + optimizer + scaler + scheduler + the completed-shard set
(resumable). Each is ~0.4 GB (fp32 params + AdamW state).

**A full epoch writes ~100 of them (~40 GB).** On a small instance disk this
fills up mid-run, so `keep_last_n_checkpoints` (config, default 3) deletes older
`checkpoint_shards_*.pt` after each save. Keep it set, or ensure `checkpoint_dir`
has tens of GB.

## 8. Expected performance (4x A100-40GB, 30.66M model)

- **~20,600 samples/s** at global batch 12,000 (micro 1500 x accum 2); every GPU
  at 100%. (An earlier SXM box measured ~17,000 with the checkpointing design.)
- **~91 h (~3.8 days) per epoch** over the 10,211 shards (~6.8B positions).
- Throughput is bounded by GPU efficiency on small GEMMs, not by data loading,
  allreduce, or checkpointing. Bigger micro-batches are the lever (see §6).
- Effective MFU is only ~17% — the model is small (384-wide), so matmuls don't
  saturate an A100. This is expected; a bigger model would get better MFU.
- **`torch.compile` is the single biggest lever:** ~2,800 → ~7,600 samples/s
  per GPU (mode `default`). Without it the epoch is ~10 days.
- **Data loading is not the bottleneck.** Raw shard reads scale to >300k rows/s
  with 8 workers; converted batches cost ~40 ms/3000 rows. If GPUs aren't at
  100%, suspect compile/CPU before the data path.

### When to stop: the model is capacity-bound, not data-bound

Validation loss falls logarithmically and flattens hard (~1.72 → 1.46 → 1.40 →
1.38 by ~180k steps). The model has 30.66M params and the epoch is ~500B tokens,
so it is ~100x+ past the compute-optimal data point: extra data buys almost
nothing. What remains is mostly the LR anneal, not new data — so it is cheaper
to **shorten the schedule** (`--anneal-steps N --max-batches N`) than to run the
full epoch. To actually use the 6.8B positions, scale the model instead.

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
- **`FailOnRecompileLimitHit: Hard failure due to fullgraph=True` / `Dynamo hit
  config.recompile_limit (8)`** → `torch.compile` saw too many distinct input
  shapes. Parquet shards have different row counts, so their trailing partial
  batches differ in size and each new size builds a fresh graph; with
  `fullgraph=True` the 9th graph is fatal (it killed a run at step 116,400).
  Fixed by `drop_last=True` in `PGNDataset.from_parquet` (set for both train and
  val datasets in `train.py`), which emits only full `micro_batch`-sized batches.
  If you ever bypass that (new dataset path), either force one shape or raise
  `torch._dynamo.config.recompile_limit` / use `dynamic=True` — but note
  `dynamic=True` still recompiled on `pad_mm` guards in testing, so a static
  shape is the reliable fix. Do **not** just raise the limit and hope: recompiles
  also leak memory. Resume with `--resume checkpoints/checkpoint_shards_<N>.pt`.

## 10. Vast.ai specifics (if that's the host)

- **"stop" preserves the whole filesystem; "recycle"/"destroy" wipe it.** Only a
  mounted volume survives a recycle. Check
  `vast-capabilities | jq '.instance.workspace_is_volume'` — if `false`, nothing
  on disk is safe across recycle/destroy.
- `python` is not on `PATH` in non-interactive shells — `source /venv/main/bin/activate`.
- The instance prints an agent guide to read first: `/etc/vast-agents-guide.md`
  (supervisor, Caddy/portal ports, provisioning, CUDA notes).
