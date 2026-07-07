"""FP8 (E4M3) quantization for GigaAM, targeting AMD MI300X.

This package quantizes the Conformer encoder's Linear layers to FP8 E4M3, using a
faithful quantize-dequant simulation on hardware without native FP8 support. The
exported artifact (uint8 weight bytes + per-channel weight scales + per-tensor
activation scales) is consumed directly by a custom FP8 runtime.

Pipeline
--------
1. ``quantize_encoder(model)``        : swap eligible Linears -> FP8Linear (PTQ).
2. ``calibrate.py`` / KD fine-tuning  : set activation scales / recover accuracy.
3. ``export_fp8_model(model, dir)``   : emit the runtime-agnostic artifact.

See ``gigaam/fp8/quantize.py`` for the per-module quantization decision.
"""

from .export import export_fp8_model
from .export_native import model_to_safetensors
from .linear import E4M3_MAX, FP8Linear
from .quantize import (
    bake_activation_scales,
    collect_fp8_linears,
    count_targets,
    quantize_encoder,
    quantize_head,
    quantize_model,
    should_quantize,
    should_quantize_head,
)

__all__ = [
    "E4M3_MAX",
    "FP8Linear",
    "bake_activation_scales",
    "collect_fp8_linears",
    "count_targets",
    "export_fp8_model",
    "model_to_safetensors",
    "quantize_encoder",
    "quantize_head",
    "quantize_model",
    "should_quantize",
    "should_quantize_head",
]
