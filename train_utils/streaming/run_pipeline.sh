#!/usr/bin/env bash
# Streaming fine-tune pipeline for one H100: data -> teacher pseudo-labels ->
# streaming student(s) with distillation -> restoration-gap evaluation.
#
#   bash train_utils/streaming/run_pipeline.sh            # full run (hours)
#   SMOKE=1 bash train_utils/streaming/run_pipeline.sh    # ~10 min end-to-end check
#   STAGES="train eval" bash train_utils/streaming/run_pipeline.sh
#
# Tunables (environment variables, defaults in brackets):
#   DATA_DIR   [./data/golos]      where WAVs + manifests go
#   OUT_DIR    [./checkpoints]     experiments (models/<exp>/..., tb logs)
#   MODEL      [v3_e2e_ctc]        base model to convert (v3_ctc, v3_e2e_ctc, v3_rnnt, v3_e2e_rnnt)
#   TRAIN_HOURS [all]              cap on training audio
#   CONFIGS    ["rc1_c8 rc0_c16"]  streaming configs to train (see below)
#   EPOCHS     [2]  BATCH [32]  LR [1e-4]  KD_ALPHA [0.5]  WORKERS [16]
#   COMPILE_BACKEND [inductor]     torch.compile backend
#   ACCELERATOR [auto]             Lightning accelerator (cpu for a local smoke run)
#
# Configs: rc1_c8  = chunk 8 (320 ms) + 1 chunk hard-copied look-ahead, 640 ms latency
#          rc0_c16 = chunk 16 (640 ms), strictly causal, same latency, ~half the training cost
#          rc0_c8  = chunk 8 (320 ms), strictly causal, lowest latency
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"

DATA_DIR="${DATA_DIR:-$ROOT/data/golos}"
OUT_DIR="${OUT_DIR:-$ROOT/checkpoints}"
MODEL="${MODEL:-v3_e2e_ctc}"
CONFIGS="${CONFIGS:-rc1_c8 rc0_c16}"
STAGES="${STAGES:-data pseudo train eval}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-32}"
LR="${LR:-1e-4}"
KD_ALPHA="${KD_ALPHA:-0.5}"
WORKERS="${WORKERS:-16}"
COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}"
LEFT_CONTEXT="${LEFT_CONTEXT:-16}"
ACCELERATOR="${ACCELERATOR:-auto}"
PY="${PYTHON:-python}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  DATA_DIR="${DATA_DIR}_smoke"
  TRAIN_HOURS="${TRAIN_HOURS:-0.05}"
  VAL_HOURS="0.02"
  TRAIN_ARGS="--max_steps 4 --val_check_steps 4 --val_first_batches 2 --log_every_n_steps 1"
  EVAL_LIMIT="--limit 8"
  BATCH=4
else
  TRAIN_HOURS="${TRAIN_HOURS:-}"
  VAL_HOURS="${VAL_HOURS:-2}"
  TRAIN_ARGS="--max_epochs $EPOCHS --val_check_interval 0.25 --val_first_batches 200"
  EVAL_LIMIT="${EVAL_LIMIT:-}"
fi

case "$MODEL" in
  *e2e*) TEXT_ARGS="" ;;
  *)     TEXT_ARGS="--raw_text" ;;   # character-level vocab: strip punctuation/case
esac

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# ---------------------------------------------------------------- 1. data
if has_stage data; then
  echo "== data -> $DATA_DIR"
  $PY train_utils/prepare_golos.py --out_dir "$DATA_DIR" --workers "$WORKERS" \
      --val_hours "$VAL_HOURS" ${TRAIN_HOURS:+--train_hours "$TRAIN_HOURS"}
fi

# ------------------------------------------------- 2. teacher pseudo-labels
# The student is trained to restore the OFFLINE model, so its targets are the
# teacher's own transcripts (plus the per-frame KL term in training).
TRAIN_MANIFEST="$DATA_DIR/manifest_train.teacher.tsv"
if has_stage pseudo; then
  echo "== pseudo-label with offline $MODEL"
  (cd train_utils && $PY pseudo_label.py --teacher "$MODEL" \
      --in_manifest "$DATA_DIR/manifest_train.tsv" --out_manifest "$TRAIN_MANIFEST" \
      --batch_size 64)
fi

# ------------------------------------------------------------- 3. training
config_args() {
  case "$1" in
    rc1_c8)  echo "--chunk_size 8  --right_context_chunks 1" ;;
    rc0_c16) echo "--chunk_size 16 --right_context_chunks 0" ;;
    rc0_c8)  echo "--chunk_size 8  --right_context_chunks 0" ;;
    *) echo "unknown config $1" >&2; exit 1 ;;
  esac
}

if has_stage train; then
  for cfg in $CONFIGS; do
    EXP="stream_${MODEL}_${cfg}"
    echo "== train $EXP"
    (cd train_utils && $PY train.py \
        --model_name "$MODEL" \
        --train_manifest "$TRAIN_MANIFEST" \
        --val_manifest "$DATA_DIR/manifest_val.tsv" \
        --output_dir "$OUT_DIR" --exp_name "$EXP" \
        --streaming $(config_args "$cfg") --left_context_chunks "$LEFT_CONTEXT" \
        --kd_alpha "$KD_ALPHA" \
        --batch_size "$BATCH" --eval_batch_size 64 --bucket_by_duration \
        --num_workers "$WORKERS" --precision bf16-mixed --accelerator "$ACCELERATOR" \
        --compile --compile_backend "$COMPILE_BACKEND" \
        --lr "$LR" --warmup_ratio 0.05 --gradient_clip_val 1.0 \
        --skip_initial_validation --save_top_k 2 \
        $TEXT_ARGS $TRAIN_ARGS)
  done
fi

# ----------------------------------------------------------------- 4. eval
if has_stage eval; then
  for cfg in $CONFIGS; do
    EXP="stream_${MODEL}_${cfg}"
    CKPT=$(ls -t "$OUT_DIR/models/$EXP"/*.ckpt 2>/dev/null | head -1 || true)
    if [[ -z "$CKPT" ]]; then echo "no checkpoint for $EXP"; continue; fi
    echo "== eval $EXP ($CKPT)"
    (cd train_utils && $PY eval_streaming.py --streaming_ckpt "$CKPT" --offline "$MODEL" \
        --manifest "$DATA_DIR/manifest_val.tsv" --batch_size 32 $EVAL_LIMIT)
  done
fi
echo "done"
