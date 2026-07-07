# FP8 quantization for GigaAM (`gigaam.fp8`)

Quantize the Conformer encoder's Linear layers to **FP8 E4M3** and emit a
runtime-agnostic artifact for native FP8 execution (e.g. AMD MI300X Matrix Core).

The dev box simulates FP8 via a faithful quantize-dequant round trip, so **WER
measured here equals WER on the FP8 target hardware** (same E4M3 rounding, same
scales). The exported artifact is `uint8` weight bytes + per-channel float32
scales + per-tensor activation scales — a custom FP8 runtime casts the bytes to
FP8 and issues MFMA.

## What gets quantized

- **FP8 (E4M3):** all `nn.Linear` in the Conformer encoder — FFN1/FFN2, attention
  q/k/v/out, the subsampling output projection. ~87% of params/compute.
- **Kept fp16/fp32:** preprocessor (Log-Mel DSP), subsampling input convs, all
  LayerNorm/BatchNorm, attention softmax + score matmul, rotary cos/sin, the CTC
  head (1×1 Conv1d, ~0.03% of params), and the final log_softmax.

The decision is encoded in `quantize.should_quantize` and is auditable.

## Two stages

FP8 E4M3 PTQ is near-lossless on GigaAM v3 (~97–98% transcript agreement with the
fp16 model on held-out audio), so the pipeline is just **calibrate → export**.
QAT + logit knowledge distillation was evaluated and did **not** shrink the
(already tiny) FP8-vs-fp16 gap — training perturbs the near-optimal PTQ weights —
so it is not part of the flow.

### 1. Calibration + PTQ WER

```bash
uv run python -m gigaam.fp8.calibrate \
    --manifest path/to/manifest_val.tsv \
    --model_name v3_e2e_ctc \
    --n_batches 64 --batch_size 16 \
    --export_dir fp8_out
```

Loads the fp16 model, quantizes it, collects per-tensor activation `amax` over
`n_batches`, bakes the scales, and reports fp16 vs. FP8-PTQ WER. Writes
`activation_scales.json` + the exported artifact.

### 2. Export

```python
import gigaam
from gigaam.fp8 import quantize_encoder, export_fp8_model

model = gigaam.load_model_from_hf("v3_e2e_ctc")   # or load_model if CDN is reachable
quantize_encoder(model, compute_dtype=torch.float16)
export_fp8_model(model, out_dir="fp8_out", model_name="v3_e2e_ctc_fp8")
```

Artifact:
- `v3_e2e_ctc_fp8.safetensors` — FP8 weights as `uint8` (E4M3 bitcast) + `<w>_scale`
  per-channel float32; unquantized params in native dtype.
- `activation_scales.json` — per-tensor activation `amax` per FP8Linear.
- `model_meta.json` — FP8 module list, dtype map, granularity, full model config.

## Loading models from HuggingFace

If the default CDN is unreachable, use the HuggingFace mirror:

```python
import gigaam
model = gigaam.load_model_from_hf("v3_e2e_ctc")  # revision under ai-sage/GigaAM-v3
```

## Materializing a dataset subset

```bash
uv run python -m gigaam.fp8.data --out_dir data/sub --n_samples 1024
```

Streams a bounded sample from `govnejri/golos_mfa_punctuation` to a TSV manifest
that `gigaam.utils.AudioDataset` reads directly.
