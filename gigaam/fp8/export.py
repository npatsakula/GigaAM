"""Export a FP8-quantized GigaAM model to a runtime-agnostic artifact.

The artifact is a directory containing:

- ``<model_name>.safetensors`` : the FP8 weights as ``uint8`` bytes (E4M3 bitcast)
  plus a ``<name>_scale`` float32 tensor per weight (per-output-channel), and the
  unquantized fp16 parameters (norms, biases, embeddings, head) as-is.
- ``activation_scales.json`` : per-tensor activation ``amax`` for each FP8Linear,
  keyed by encoder-relative dotted path.
- ``model_meta.json`` : which modules are FP8, the dtype map, and the full model
  config (mirroring the ``to_onnx`` config-saving pattern in gigaam/model.py).

A custom runtime (e.g. targeting AMD MI300X FP8 Matrix Core) reads the uint8
weight bytes, casts them to FP8, applies the per-channel scale, and issues FP8
MFMA instructions using the activation scales.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from .linear import FP8Linear
from .quantize import collect_fp8_linears


def _flatten_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Flatten ``model.state_dict()`` but emit FP8 weight bytes as uint8.

    For each FP8Linear ``weight`` the export emits:
      - ``<path>.weight`` : uint8 E4M3 bitcast
      - ``<path>.weight_scale`` : float32 per-output-channel scale
    All other parameters/buffers are emitted in their current dtype.
    """
    fp8_modules: Dict[str, FP8Linear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, FP8Linear):
            fp8_modules[f"{name}.weight"] = mod

    tensors: Dict[str, torch.Tensor] = {}
    # Emit unquantized params/buffers from the standard state dict.
    for full_name, tensor in model.state_dict().items():
        if full_name in fp8_modules:
            continue  # emitted below as uint8 + scale
        tensors[full_name] = tensor.contiguous()

    # Emit FP8 latent weights, quantized to E4M3 bytes + per-channel scale.
    for full_name, mod in fp8_modules.items():
        weight_q, weight_scale = mod.quantize_weight_bytes()
        tensors[full_name] = weight_q.contiguous().to("cpu")
        tensors[f"{full_name}_scale"] = weight_scale.contiguous().to("cpu")
    return tensors


def export_fp8_model(
    model: nn.Module,
    out_dir: str,
    model_name: Optional[str] = None,
) -> str:
    """Export a FP8-quantized GigaAM model to ``out_dir``.

    Parameters
    ----------
    model : nn.Module
        A GigaAM model whose encoder has been quantized (FP8Linears in place).
    out_dir : str
        Destination directory (created if absent).
    model_name : str, optional
        Base name for the safetensors file. Defaults to ``model.cfg.model_name``.

    Returns
    -------
    str
        Path to the written safetensors file.
    """
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = getattr(model.cfg, "model_name", "gigaam_fp8")

    # 1. Weights (+ fp8 bytes + scales) -> safetensors.
    tensors = {k: v.contiguous().cpu() for k, v in _flatten_state(model).items()}
    st_path = out / f"{model_name}.safetensors"
    save_file(tensors, str(st_path))

    # 2. Activation scales.
    act_scales: Dict[str, float] = {}
    for name, mod in collect_fp8_linears(model).items():
        if mod.act_scale is not None:
            act_scales[name] = float(mod.act_scale)
    (out / "activation_scales.json").write_text(
        json.dumps(act_scales, indent=2), encoding="utf-8"
    )

    # 3. Metadata: FP8 module list, dtype map, full config.
    fp8_modules = sorted(collect_fp8_linears(model).keys())
    dtype_map = {
        name: str(t.dtype).removeprefix("torch.") for name, t in tensors.items()
    }
    meta = {
        "model_name": model_name,
        "quant_format": "fp8_e4m3",
        "weight_granularity": "per_output_channel",
        "activation_granularity": "per_tensor",
        "fp8_modules": fp8_modules,
        "e4m3_max": 448.0,
        "dtype_map": dtype_map,
        "config": _cfg_to_serializable(model.cfg),
    }
    (out / "model_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Exported FP8 model -> {st_path}")
    print(
        f"  {len(fp8_modules)} FP8 modules, "
        f"{len(act_scales)} activation scales, {len(tensors)} tensors total"
    )
    return str(st_path)


def _cfg_to_serializable(cfg) -> dict:
    """Convert an omegaconf DictConfig (or similar) to a plain JSON dict."""
    try:
        import omegaconf

        return omegaconf.OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        if hasattr(cfg, "items"):
            return {k: _cfg_to_serializable(v) for k, v in cfg.items()}
        return cfg
