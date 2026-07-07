"""Quantize a GigaAM encoder's (and RNNT head's) Linear layers to FP8.

``quantize_encoder`` walks ``model.encoder`` and replaces the target ``nn.Linear``
modules with ``FP8Linear``. The decision of which Linears to swap is made by the
``should_quantize`` predicate below, which encodes the analysis of which modules are
safe / beneficial to quantize and which must stay in higher precision.

Why each excluded class is excluded (see gigaam/encoder.py for the architecture):

- ``pre_encode.conv.*`` : the strided subsampling input convolutions run on raw
  log-mel features; a quantization error at the input boundary propagates through
  every downstream layer. They are convolutions, not the target GEMMs, so they are
  also skipped by the ``isinstance(mod, nn.Linear)`` filter in the walk.
- ``layers.*.conv.*`` : the Conformer convolution module (pointwise -> GLU ->
  depthwise -> BN/LN -> SiLU -> pointwise). The GLU gate is precision-sensitive and
  the norm running stats need range. Kept in the module's native dtype.
- attention ``linear_pos`` (only present for rel_pos, not rotary): tiny position
  projection; kept unquantized for parity with the rotary cos/sin path.
- Everything that is not an ``nn.Linear`` (LayerNorms, BatchNorms, embeddings,
  softmax, rotary buffers, pos_bias_u/v) is left untouched by construction.

The CTC head (``model.head``) is intentionally NOT quantized: it is a single 1x1
Conv1d of ~60k params (~0.03% of the model) feeding a log_softmax. Not worth the
risk for the negligible memory/compute savings.

``compute_dtype`` defaults to ``None`` (auto-detect the module's native parameter
dtype in :meth:`FP8Linear.from_linear`) so the quantized layers keep the dtype of
the surrounding un-quantized ops — fp32 during QAT (fp32 encoder), fp16 in the
halved eval/deploy path. Forcing a fixed fp16 while the encoder is fp32 makes the
first un-quantized ``matmul``/``conv`` raise on a fp16/fp32 mix.
"""

import warnings
from typing import Dict, Optional

import torch
from torch import nn

from .linear import FP8Linear


def should_quantize(name: str) -> bool:
    """Return True if the Linear at dotted ``name`` should be FP8-quantized.

    ``name`` is the path relative to the *encoder* root, e.g.
    ``layers.0.feed_forward1.linear1``. For head modules use
    :func:`should_quantize_head`.

    Selection anchors on the *parent* module suffix and the exact leaf name rather
    than loose substring tests, so ``self_attn.linear_pos`` (rel_pos only) is not
    accidentally swept in by a ``self_attn.linear_`` substring — the previous
    implementation returned True for ``linear_pos`` because that check fired before
    the dedicated exclusion could run.
    """
    # Subsampling output projection: a real GEMM, safe to quantize.
    if name == "pre_encode.out":
        return True
    # Conformer convolution module: excluded (GLU gate + norm precision).
    if ".conv." in name:
        return False

    parent, _, leaf = name.rpartition(".")
    # Feed-forward projections: the prime targets.
    if parent.endswith(("feed_forward1", "feed_forward2")):
        return leaf in ("linear1", "linear2")
    # Attention q/k/v/out projections -> FP8; linear_pos (rel_pos) stays unquantized.
    if parent.endswith("self_attn"):
        return leaf in ("linear_q", "linear_k", "linear_v", "linear_out")
    # Any other Linear we did not explicitly classify: leave it alone.
    return False


def should_quantize_head(name: str) -> bool:
    """Return True if a head Linear at dotted ``name`` should be FP8-quantized.

    ``name`` is the path relative to ``model.head``, e.g. ``joint.enc``.

    RNNT joint: the ``enc`` and ``pred`` projection Linears are GEMMs and safe to
    quantize. The ``joint_net`` final vocab projection and the LSTM decoder /
    embedding stay unquantized (output logits / recurrent / gather). The CTC head is
    a single 1x1 Conv1d and has no Linears.
    """
    return name in ("joint.enc", "joint.pred")


def _set_submodule(root: nn.Module, dotted: str, module: nn.Module) -> None:
    """Replace the submodule at ``dotted`` path within ``root``."""
    *parents, leaf = dotted.split(".")
    parent = root
    for p in parents:
        parent = getattr(parent, p)
    setattr(parent, leaf, module)


def quantize_encoder(
    model: nn.Module,
    compute_dtype: Optional[torch.dtype] = None,
) -> Dict[str, FP8Linear]:
    """Replace eligible Linears in ``model.encoder`` with ``FP8Linear`` in place.

    Parameters
    ----------
    model : nn.Module
        A GigaAM model (must expose ``.encoder``).
    compute_dtype : torch.dtype, optional
        Dtype the simulated matmul runs in. ``None`` (default) auto-detects each
        Linear's native dtype; pass an explicit dtype to override.

    Returns
    -------
    dict
        Mapping of *model-root* dotted-path (``encoder.<...>``) -> the new
        ``FP8Linear`` module, matching the keys used by
        :func:`bake_activation_scales` / :func:`collect_fp8_linears`.
    """
    encoder = model.encoder
    replaced: Dict[str, FP8Linear] = {}
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
        replaced[f"encoder.{name}"] = fp8
    return replaced


def quantize_head(
    model: nn.Module,
    compute_dtype: Optional[torch.dtype] = None,
) -> Dict[str, FP8Linear]:
    """Replace eligible Linears in ``model.head`` with ``FP8Linear`` in place.

    Only the RNNT joint's ``enc``/``pred`` projections are quantized; the CTC head
    (Conv1d) has no Linears and is left untouched. See :func:`should_quantize_head`.
    ``compute_dtype`` defaults to auto-detecting the head's native dtype (fp32 for
    the RNNT head, which ``load_model`` leaves un-halved).
    """
    head = model.head
    replaced: Dict[str, FP8Linear] = {}
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
    compute_dtype: Optional[torch.dtype] = None,
) -> Dict[str, FP8Linear]:
    """Quantize both the encoder and (if applicable) the RNNT head in place.

    Both sub-maps are keyed by model-root ``named_modules()`` paths (``encoder.*`` /
    ``head.*``), so the returned dict can be used directly for introspection and its
    keys match :func:`collect_fp8_linears` and the calibration amax dict.
    """
    replaced = quantize_encoder(model, compute_dtype=compute_dtype)
    if hasattr(model, "head"):
        # The head is never halved by load_model (it halves the encoder only), so
        # the RNNT joint's enc/pred GEMMs must follow the head's own native dtype
        # (fp32) — matching the un-quantized joint_net / decoder — regardless of the
        # encoder's compute_dtype. Passing the encoder's fp16 here would make the
        # joint enc/pred emit fp16 into the fp32 joint_net and crash. Always
        # auto-detect (None) for the head.
        replaced.update(quantize_head(model, compute_dtype=None))
    return replaced


def bake_activation_scales(model: nn.Module, amax_dict: Dict[str, float]) -> int:
    """Apply per-tensor activation ``amax`` values to the model's FP8Linears.

    Keys in ``amax_dict`` are model-root paths: encoder keys are prefixed
    ``encoder.`` and head keys ``head.`` (as produced by the calibrator, which walks
    the same ``named_modules()`` tree). Returns the number of scales baked.

    Fails loudly on a total key-namespace mismatch (e.g. a stale calibration file
    with bare ``layers.*`` keys) instead of silently baking nothing and leaving the
    run on dynamic scaling; warns on partial mismatches so a wrong/outdated file is
    visible rather than a silent train/deploy scale discrepancy.
    """
    fp8 = {
        name: mod
        for name, mod in model.named_modules()
        if isinstance(mod, FP8Linear)
    }
    matched = [name for name in fp8 if name in amax_dict]
    for name in matched:
        fp8[name].bake_activation_scale(amax_dict[name])

    if amax_dict and not matched:
        example_model = sorted(fp8)[:2]
        example_keys = sorted(amax_dict)[:2]
        raise ValueError(
            f"bake_activation_scales matched 0 of {len(fp8)} FP8Linear modules "
            f"against {len(amax_dict)} calibration keys — key-namespace mismatch. "
            f"Model paths look like {example_model}; calibration keys look like "
            f"{example_keys}. Expected model-root paths (encoder.* / head.*)."
        )

    stale = sorted(set(amax_dict) - set(fp8))
    if stale:
        warnings.warn(
            f"bake_activation_scales: {len(stale)} calibration key(s) matched no "
            f"FP8Linear (e.g. {stale[:2]}); they were ignored.",
            stacklevel=2,
        )
    missing = sorted(set(fp8) - set(amax_dict))
    if missing:
        warnings.warn(
            f"bake_activation_scales: {len(missing)} FP8Linear(s) received no static "
            f"scale and will use dynamic scaling (e.g. {missing[:2]}).",
            stacklevel=2,
        )
    return len(matched)


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
