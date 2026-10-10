# GigaAM Fine-tuning (CTC / RNNT)

For training and evaluation examples and a more detailed description of manifests, see [`example.ipynb`](./example.ipynb).

## Setup

From the repository root, install the training dependencies:

```bash
pip install -e ".[train]"

cd train_utils
```

## Data format

TSV manifest (tab-separated columns): `path`, `duration`, and optionally `transcription`. Paths may be absolute or relative to the manifest directory. `transcription` may be omitted for audio-only manifests.

```
path	duration	transcription
audio/0001.wav	3.21	привет как дела
```

## Training

```bash
python train.py \
    --model_name v3_e2e_ctc \
    --train_manifest /path/to/manifest_train.tsv \
    --val_manifest /path/to/manifest_val.tsv \
    --max_epochs 3 \
    --val_check_interval 0.5 \
    --batch_size 64 \
    --eval_batch_size 64 \
    --lr 8e-5 \
    --activation_checkpointing

python train.py \
    --model_name v3_rnnt \
    --train_manifest /path/to/manifest_train.tsv \
    --val_manifest /path/to/manifest_val.tsv \
    --raw_text \
    --max_epochs 2 \
    --val_check_interval 0.5 \
    --batch_size 8 \
    --accumulate_grad_batches 2 \
    --rnnt_subbatch_size 2 \
    --eval_batch_size 64 \
    --lr 2e-5 \
    --val_first_batches 50
```

### Arguments

#### Model and data

| Argument | Default | Description |
|---|---|---|
| `--model_name` | required | Pretrained GigaAM model name |
| `--train_manifest` | required | TSV manifest for training |
| `--val_manifest` | required | TSV manifest for validation |
| `--raw_text` | off | For non-E2E setups: lowercase text, drop punctuation, restrict to the character vocabulary |
| `--max_duration` | `20.0` | Maximum audio length in seconds (dataset filter) |
| `--min_duration` | `0.1` | Minimum audio length in seconds (dataset filter) |

#### Scheduling (epochs vs steps)

| Argument | Default | Description |
|---|---|---|
| `--max_epochs` | `None` | Train for this many epochs (omit if using `--max_steps`) |
| `--val_check_interval` | `1.0` | Run validation every N-th fraction of an epoch |
| `--max_steps` | `None` | Train for this many steps (requires `--val_check_steps`) |
| `--val_check_steps` | `None` | With `--max_steps`: validate every N training steps |
| `--val_first_batches` | `None` | If set, run validation on only the first N batches |

#### Batching and memory

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | `8` | Per-device training batch size |
| `--eval_batch_size` | `32` | Validation batch size |
| `--rnnt_subbatch_size` | `0` | RNNT loss sub-batches (`0` disables) |
| `--num_workers` | `4` | Number of `DataLoader` workers |
| `--precision` | `32` | Lightning precision (`16`, `bf16`, `32`, ...) |
| `--accumulate_grad_batches` | `1` | Gradient accumulation steps |
| `--accelerator` | `auto` | Lightning accelerator (`auto`, `cpu`, `gpu`, ...) |
| `--devices` | `1` | Number of devices (DDP when `> 1`) |
| `--activation_checkpointing` | off | Activation checkpointing for each Conformer layer |
| `--freeze_encoder` | off | Freeze encoder weights |

#### Optimizer

| Argument | Default | Description |
|---|---|---|
| `--lr` | `2e-5` | Peak learning rate (AdamW) |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--warmup_ratio` | `0.1` | Linear warmup fraction before cosine decay |
| `--gradient_clip_val` | `1.0` | Global gradient norm clipping |
| `--seed` | `42` | Passed to `pl.seed_everything` |

#### SpecAugment

| Argument | Default | Description |
|---|---|---|
| `--freq_masks` | `2` | Number of frequency masks |
| `--freq_width` | `27` | Maximum width of each frequency mask (bins) |
| `--time_masks` | `2` | Number of time masks |
| `--time_width` | `20` | Maximum width of each time mask (frames) |
| `--disable_spec_augment` | off | Disable SpecAugment (enabled by default) |

#### Outputs, logging and resuming

| Argument | Default | Description |
|---|---|---|
| `--output_dir` | `./checkpoints` | Checkpoints under `models/<exp_name>/`, TensorBoard under `tb_logs/` |
| `--exp_name` | auto | Run directory name; if omitted, derived from hyperparameters |
| `--log_every_n_steps` | `25` | Logging interval in steps |
| `--save_top_k` | `2` | Keep this many best checkpoints by `val_wer` |
| `--disable_tqdm` | off | Disable progress bars |
| `--skip_initial_validation` | off | Skip `trainer.validate` before `fit` |
| `--resume_from_checkpoint` | `None` | Path to a Lightning `.ckpt` file to resume training from |

## Streaming fine-tuning

A base model patched into streaming mode (`gigaam.enable_streaming`) runs correctly but transcribes poorly: its weights never saw causal convolutions, chunked attention or a bounded context window. Streaming quality comes from a fine-tune in streaming mode that distills the offline model into the streaming one. The whole recipe for one H100 is scripted:

```bash
bash train_utils/streaming/setup_h100.sh        # uv venv, PyTorch 2.14.1 (CUDA 13.0), train extras
source .venv-h100/bin/activate
SMOKE=1 bash train_utils/streaming/run_pipeline.sh   # ~10 min end-to-end check on a tiny subset
bash train_utils/streaming/run_pipeline.sh           # full run; see the header for the knobs
```

The pipeline has four stages, selectable with `STAGES="data pseudo train eval"`:

1. **data** — `prepare_golos.py` materializes [govnejri/golos_mfa_punctuation](https://huggingface.co/datasets/govnejri/golos_mfa_punctuation) (970k clips, ~1044 h, 16 kHz) to WAV + manifests, shards in parallel, with the last shard held out for validation. `--train_hours` bounds a subset and reads only the shards it needs.
2. **pseudo** — `pseudo_label.py` transcribes the training audio with the offline model, so the student is trained to restore the offline model rather than to fit the dataset's transcripts.
3. **train** — `train.py --streaming` for each config in `CONFIGS` (default `rc1_c8 rc0_c16`, see below) with `--kd_alpha 0.5` (per-frame KL from the frozen offline teacher plus CTC on its pseudo-labels), `--precision bf16-mixed`, `--compile`, and `--bucket_by_duration`.
4. **eval** — `eval_streaming.py` reports the restoration gap (WER/CER of the streaming model against the offline model's output), absolute WER against the dataset text, and the batched real-time factor.

Streaming flags of `train.py`:

| Argument | Default | Description |
|---|---|---|
| `--streaming` | off | Fine-tune a streaming encoder (causal subsampling/convs + chunked attention). Adds no parameters; the checkpoint restores these settings on `gigaam.load_model`. |
| `--chunk_size` | 8 | Chunk in encoder frames of 40 ms. A token is emitted once its chunk is complete, so latency is `chunk_size * 40 ms` (plus one chunk with look-ahead). |
| `--left_context_chunks` | 16 | Attention window into the past, in chunks (16 = 5.1 s at chunk 8). Bounds the inference cache. `-1` = unlimited. |
| `--right_context_chunks` | 0 | `1` adds one chunk of hard-copied (Emformer-style) look-ahead: latency doubles, the training forward is exactly the chunked inference computation, encoder cost ~2x. |
| `--kd_alpha` | 0 | Distillation weight: `kd_alpha * KL(teacher || student) + (1 - kd_alpha) * CTC`. CTC models only; RNN-T streaming fine-tunes use the transducer loss on the teacher's pseudo-labels. |
| `--init_from` | – | Warm-start the student from a fine-tuned checkpoint. |
| `--compile`, `--compile_backend` | off, inductor | `torch.compile` the encoder and the teacher with dynamic shapes. Both streaming forwards capture as a single graph with no breaks. |
| `--bucket_by_duration` | off | Batch clips of similar duration (shuffled buckets); typically 1.5-2x throughput on Golos-like data. |

Configs in the pipeline: `rc1_c8` (320 ms chunks + one chunk of look-ahead, 640 ms latency), `rc0_c16` (640 ms chunks, strictly causal, same latency at half the training cost) and `rc0_c8` (320 ms, strictly causal, lowest latency). Train the first two and pick by the restoration gap; as a reference, FP8 quantization of the same models disagrees with fp16 at about 1.4% WER.

Rough budget on one H100 with the defaults (bf16, compiled, bucketed, batch 32, 2 epochs over ~1000 h): on the order of 45k steps, a few hours per config. The fine-tuned checkpoint drops straight into `gigaam.StreamingSession` / `gigaam.StreamingBatch`.

## Evaluation

Evaluate a fine-tuned checkpoint:

```bash
python eval.py \
    --checkpoint ./checkpoints/models/<exp_name>/gigaam-*.ckpt \
    --eval_manifest /path/to/manifest.tsv
```

Evaluate a pretrained GigaAM model:

```bash
python eval.py --model_name v3_e2e_ctc --eval_manifest /path/to/manifest.tsv
```

This writes `preds.jsonl` and prints WER. Predictions are saved under `predictions/<manifest_stem>/<exp_name>/step_<step>/preds.jsonl` (`step_<step>` is omitted for pretrained models) next to the manifest. WER is reported on the original transcripts (end-to-end) and on raw texts.

## Loading fine-tuned checkpoints

`gigaam.load_model` accepts a path to a Lightning `.ckpt` file, so fine-tuned models can be loaded the same way as pretrained ones:

```python
model = gigaam.load_model("./checkpoints/models/<exp_name>/gigaam-*.ckpt")
```
