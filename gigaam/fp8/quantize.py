"""Quantize a GigaAM encoder's Linear layers to FP8.

``quantize_encoder`` walks ``model.encoder`` and replaces the target ``nn.Linear``
modules with ``FP8Linear``. The decision of which Linears to swap is made by the
``should_quantize`` predicate below, which encodes the analysis of which modules are
safe / beneficial to quantize and which must stay in higher precision.

Why each excluded class is excluded (see gigaam/encoder.py for the architecture):

- ``pre_encode.conv.*`` : the strided subsampling input convolutions run on raw
  log-mel features; a quantization error at the input boundary propagates through
  every downstream layer. They are convolutions, not the target GEMMs anyway.
- ``layers.*.conv.*`` : the Conformer convolution module (pointwise -> GLU ->
  depthwise -> BN -> SiLU -> pointwise). The GLU gate is precision-sensitive and
  the batch-norm running stats need range. Kept fp16.
- attention ``linear_pos`` (only present for rel_pos, not rotary): tiny, position
  projection; kept fp16 for parity with the rotary cos/sin path.
- Everything that is not an ``nn.Linear`` (LayerNorms, BatchNorms, embeddings,
  softmax, rotary buffers, pos_bias_u/v) is left untouched by construction.

The CTC head (``model.head``) is intentionally NOT quantized: it is a single 1x1
Conv1d of ~60k params (~0.03% of the model) feeding a log_softmax. Not worth the
risk for the negligible memory/compute savings.
"""

from typing import Dict, Optional

import torch
from torch import nn

from .linear import FP8Linear


def should_quantize(name: str) -> bool:
    """Return True if the Linear at dotted ``name`` should be FP8-quantized.

    ``name`` is the path relative to the *encoder* root, e.g.
    ``layers.0.feed_forward1.linear1``. For head modules use
    :func:`should_quantize_head`.
    """
    # Subsampling output projection: a real GEMM, safe to quantize.
    if name == "pre_encode.out":
        return True
    # Subsampling input convs: excluded (handled by conv branch below), but the
    # convs are Conv2d, not Linear, so they never reach here anyway.
    if name.startswith("pre_encode."):
        return False
    # Conformer convolution module: excluded (GLU gate + BatchNorm precision).
    if ".conv." in name:
        return False
    # Feed-forward and attention projection Linears: the prime targets.
    if ".feed_forward1." in name or ".feed_forward2." in name:
        return True
    if ".self_attn.linear_" in name:  # linear_q/k/v/out
        return True
    if name.endswith(".self_attn.linear_pos"):  # rel_pos variant only
        return False
    # Any other Linear we did not explicitly classify: leave it alone.
    return False


def should_quantize_head(name: str) -> bool:
    """Return True if a head Linear at dotted ``name`` should be FP8-quantized.

    ``name`` is the path relative to ``model.head``, e.g. ``joint.enc``.

    RNNT joint: the ``enc`` and ``pred`` projection Linears are GEMMs and safe to
    quantize. The ``joint_net`` final vocab projection and the LSTM decoder /
    embedding stay fp16 (output logits / recurrent / gather — see analysis).
    The CTC head is a single 1x1 Conv1d (~0.03% of params) and is left fp16.
    """
    # RNNT joint input projections -> FP8.
    if name in ("joint.enc", "joint.pred"):
        return True
    # RNNT joint final vocab projection -> fp16 (output logits precision).
    if name.startswith("joint.joint_net"):
        return False
    # Everything else in the head (decoder LSTM, embedding, CTC conv) -> fp16.
    return False


def _set_submodule(root: nn.Module, dotted: str, module: nn.Module) -> None:
    """Replace the submodule at ``dotted`` path within ``root``."""
    *parents, leaf = dotted.split(".")
    parent = root
    for p in parents:
        parent = getattr(parent, p)
    setattr(parent, leaf, module)


def quantize_encoder(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.float16,
) -> Dict[str, nn.Module]:
    """Replace eligible Linears in ``model.encoder`` with ``FP8Linear`` in place.

    Parameters
    ----------
    model : nn.Module
        A GigaAM model (must expose ``.encoder``).
    compute_dtype : torch.dtype
        Dtype the simulated matmul runs in.

    Returns
    -------
    dict
        Mapping of quantized dotted-path -> the new ``FP8Linear`` module, for
        introspection / scale-baking by the calibration step.
    """
    encoder = model.encoder
    replaced: Dict[str, nn.Module] = {}
    # Snapshot names first since we mutate the tree while iterating.
    linear_items = [
        (name, mod)
        for name, mod in encoder.named_modules()
        if isinstance(mod, nn.Linear)
    ]
    for name, linear in linear_items:
        if not should_quantize(name):
            continue
        fp8 = FP8Linear.from_linear(linear, compute_dtype=compute_dtype)
        _set_submodule(encoder, name, fp8)
        replaced[name] = fp8
    return replaced


def quantize_head(
    model: nn.Module,
    compute_dtype: Optional[torch.dtype] = None,
) -> Dict[str, nn.Module]:
    """Replace eligible Linears in ``model.head`` with ``FP8Linear`` in place.

    Only the RNNT joint's ``enc``/``pred`` projections are quantized; the CTC head
    (Conv1d) has no Linears and is left untouched. See :func:`should_quantize_head`.

    ``compute_dtype`` defaults to the head's native parameter dtype (fp32 for the
    RNNT head, which ``load_model`` leaves un-halved), so the FP8 matmul matches
    the surrounding fp32 ops in the joint/decoder.
    """
    head = model.head
    if compute_dtype is None:
        compute_dtype = next(head.parameters()).dtype
    replaced: Dict[str, nn.Module] = {}
    linear_items = [
        (name, mod) for name, mod in head.named_modules() if isinstance(mod, nn.Linear)
    ]
    for name, linear in linear_items:
        if not should_quantize_head(name):
            continue
        fp8 = FP8Linear.from_linear(linear, compute_dtype=compute_dtype)
        _set_submodule(head, name, fp8)
        replaced[f"head.{name}"] = fp8
    return replaced


def quantize_model(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.float16,
) -> Dict[str, nn.Module]:
    """Quantize both the encoder and (if applicable) the RNNT head.

    The encoder uses ``compute_dtype`` (fp16, matching the fp16 encoder). The head
    auto-detects its own native dtype (fp32) so the RNNT joint stays internally
    consistent. Returns a mapping of dotted-path -> FP8Linear.

    Note: the key convention is intentionally *inconsistent* between the two
    sub-maps — encoder keys are bare (``layers.0...``) while head keys carry a
    ``head.`` prefix (``head.joint.enc``). This mirrors how
    :func:`bake_activation_scales` / :func:`collect_fp8_linears` key by
    model-root ``named_modules()`` paths (encoder.* / head.*) only for the head.
    The return value is for introspection only; never key amax/scale lookups off
    it — re-walk ``model.named_modules()`` instead.
    """
    replaced = quantize_encoder(model, compute_dtype=compute_dtype)
    if hasattr(model, "head"):
        replaced.update(quantize_head(model, compute_dtype=None))
    return replaced


def bake_activation_scales(model: nn.Module, amax_dict: Dict[str, float]) -> int:
    """Apply per-tensor activation ``amax`` values to the model's FP8Linears.

    Covers both encoder and head FP8Linears. Keys in ``amax_dict`` are model-root
    paths: encoder keys are prefixed ``encoder.`` and head keys ``head.``.
    """
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, FP8Linear) and name in amax_dict:
            mod.bake_activation_scale(amax_dict[name])
            n += 1
    return n


def collect_fp8_linears(model: nn.Module) -> Dict[str, FP8Linear]:
    """Return all ``FP8Linear`` modules in the model keyed by model-root dotted path.

    Includes encoder Linears (keyed ``encoder.<...>``) and head Linears
    (keyed ``head.<...>``).
    """
    return {
        name: mod for name, mod in model.named_modules() if isinstance(mod, FP8Linear)
    }


def count_targets(model: nn.Module) -> Dict[str, int]:
    """Diagnostic: FP8Linears vs remaining nn.Linears across encoder and head."""
    linear = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
    fp8 = sum(1 for _, m in model.named_modules() if isinstance(m, FP8Linear))
    return {"fp8": fp8, "fp16_linear": linear, "total": fp8 + linear}
