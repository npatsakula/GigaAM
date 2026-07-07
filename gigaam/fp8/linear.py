"""E4M3-simulated FP8 Linear layer.

This module provides ``FP8Linear``, a drop-in replacement for ``nn.Linear`` that
simulates FP8 E4M3 inference via a faithful quantize -> dequant round trip. The
simulation is numerically identical to what a real FP8 execution unit (e.g. the
AMD MI300X Matrix Core) computes: weights and activations are rounded to the E4M3
grid (max representable value ``E4M3_MAX = 448.0``) before the matmul. WER measured
with this path therefore equals WER on the target FP8 hardware.

Design
------
- The layer keeps a *float32 latent weight* as an ``nn.Parameter`` (so QAT /
  knowledge-distillation can update it). The exported artifact quantizes this
  latent weight to FP8 bytes.
- The forward pass applies *fake quantization* (quantize -> dequant) to both the
  weight and the activation before the matmul. The straight-through estimator
  (STE) makes the non-differentiable rounding pass gradients through as the
  identity: the latent weight learns to compensate for the E4M3 grid.
- Weights use *per-output-channel* scales: ``scale = amax_per_channel / E4M3_MAX``.
  Per-tensor FP8 weights lose noticeable accuracy on Conformer outlier channels.
- Activations use a *per-tensor* scale: ``scale = amax / E4M3_MAX``. In training it
  is recomputed dynamically from the live tensor; in eval it is taken from a scale
  baked in by calibration (or falls back to dynamic if unset).

``compute_dtype`` is the dtype the dequantized operands are represented in for the
matmul. It defaults to the *source* Linear's dtype (via ``from_linear``): fp32 when
the surrounding module is fp32 (QAT), fp16 when it has been halved (eval/deploy).
This avoids feeding fp16 activations into an otherwise-fp32 module (which raises on
the first un-quantized ``matmul``/``conv``), while the E4M3 rounding — the only thing
that determines FP8 accuracy — is identical either way.

Export (see export.py) emits the latent weight quantized to ``uint8`` bytes plus a
per-channel float32 scale; a custom runtime casts the bytes to FP8 and issues MFMA.
"""

from typing import Optional

import torch
from torch import Tensor, nn

# Largest finite value representable in float8_e4m3fn.
E4M3_MAX = 448.0


def _to_e4m3(x: Tensor) -> Tensor:
    """Cast a float tensor to E4M3, returning the bit pattern as uint8."""
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def _from_e4m3(x_u8: Tensor, dtype: torch.dtype) -> Tensor:
    """Interpret uint8 bytes as E4M3 and cast to ``dtype``."""
    return x_u8.view(torch.float8_e4m3fn).to(dtype)


def _quant_scale(amax: Tensor) -> Tensor:
    """Per-element E4M3 scale ``amax / E4M3_MAX``, computed in float32 and safe
    against degenerate (all-zero) channels/tensors.

    The scale is deliberately kept in float32. The previous implementation clamped
    the scale to ``1e-12`` and then cast it to ``compute_dtype``; in fp16 anything
    below ~6e-8 collapses to exactly 0.0, so a dead weight channel or an all-silent
    activation produced ``x / 0 -> NaN`` weights and a NaN loss. Here a zero ``amax``
    maps to a scale of 1.0, so the quantize step becomes ``0 / 1 = 0`` (an exact,
    NaN-free round trip) and every other scale stays representable in float32.
    """
    amax = amax.detach().to(torch.float32)
    scale = amax / E4M3_MAX
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def _fake_quant(x: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    """Faithful E4M3 quantize-dequantize of ``x`` with a straight-through estimator.

    ``scale`` (float32, broadcastable to ``x``) is applied on a *detached* copy so
    the non-differentiable fp8 cast stays out of the autograd graph; the result is
    bridged with ``x + (xq - x).detach()`` so the forward equals the E4M3-rounded
    value (in ``dtype``) while the backward is the identity. The round trip itself
    runs in float32 to avoid scale underflow; only the returned value is cast to
    ``dtype``.
    """
    x32 = x.detach().to(torch.float32)
    x_q = _from_e4m3(_to_e4m3((x32 / scale).clamp(-E4M3_MAX, E4M3_MAX)), torch.float32)
    x_q = (x_q * scale).to(dtype)
    x_d = x.to(dtype)
    return x_d + (x_q - x_d).detach()


def fake_quant_per_tensor(x: Tensor, dtype: torch.dtype) -> Tensor:
    """E4M3 per-tensor quantize-dequantize of ``x`` (STE)."""
    scale = _quant_scale(x.detach().abs().amax())
    return _fake_quant(x, scale, dtype)


def fake_quant_per_channel(x: Tensor, dtype: torch.dtype, dim: int = 0) -> Tensor:
    """Per-channel E4M3 quantize-dequantize along ``dim`` (default: output rows)."""
    reduce_dims = [i for i in range(x.dim()) if i != dim]
    scale = _quant_scale(x.detach().abs().amax(dim=reduce_dims, keepdim=True))
    return _fake_quant(x, scale, dtype)


class FP8Linear(nn.Module):
    """``nn.Linear`` with E4M3-simulated FP8 weights and activations.

    Parameters
    ----------
    in_features, out_features : int
        As ``nn.Linear``.
    bias : bool
        Whether to include a bias term (kept in float32, unquantized, trainable).
    compute_dtype : torch.dtype
        Dtype the dequantized operands / matmul run in. Prefer building via
        :meth:`from_linear`, which defaults this to the source Linear's dtype.

    Attributes
    ----------
    weight : nn.Parameter (float32)
        Latent weight, shape ``[out, in]``. Updated by the optimizer during QAT;
        fake-quantized to E4M3 in forward. Quantized to uint8 bytes on export.
    bias : nn.Parameter (float32) or None
        Unquantized bias, kept in float32 and **trainable** (like the original
        ``nn.Linear.bias``). It is stored as a Parameter — not a buffer — so
        fine-tuning can still update it; a buffer would silently freeze it.
    act_scale : Tensor or None
        Per-tensor activation scale baked by calibration. Registered as a
        *persistent* buffer so it (a) moves with the module on ``.to(device)`` and
        (b) round-trips through Lightning checkpoints — resuming a calibrated run no
        longer silently reverts to dynamic scaling. A ``None`` scale is omitted from
        ``state_dict()`` (uncalibrated models are unaffected), and the activation
        scale is computed dynamically from each input tensor when unset.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        # Activation scale: None -> dynamic; set by calibration -> static (eval).
        # Persistent so baked scales survive checkpoints (see class docstring).
        self.register_buffer("act_scale", None, persistent=True)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, compute_dtype: Optional[torch.dtype] = None
    ) -> "FP8Linear":
        """Build an ``FP8Linear`` initialised from a pretrained ``nn.Linear``.

        ``compute_dtype`` defaults to the source ``linear``'s parameter dtype, so a
        fp32 module yields a fp32-compute FP8Linear and a halved (fp16) module yields
        a fp16-compute one. This keeps the quantized layer's output dtype consistent
        with the surrounding un-quantized ops (LayerNorm/conv/residual), which
        otherwise raise on the first mixed-dtype matmul when the encoder is fp32 but
        the FP8Linear was forced to fp16. The fp32 latent weight is copied verbatim;
        fake quantization is applied in ``forward``.
        """
        if compute_dtype is None:
            compute_dtype = linear.weight.dtype
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=compute_dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight.detach().to(torch.float32))
            if linear.bias is not None:
                layer.bias.copy_(linear.bias.detach().to(torch.float32))
        return layer.to(linear.weight.device)

    def quantize_weight_bytes(self) -> "tuple[Tensor, Tensor]":
        """Quantize the latent weight to E4M3 bytes + per-channel scale.

        Returns ``(weight_q [uint8, out x in], weight_scale [float32, out])``. This is
        exactly what the export emits and what the MI300X runtime consumes.
        """
        w = self.weight.detach().to(torch.float32)
        scale = _quant_scale(w.abs().amax(dim=-1, keepdim=True))  # [out, 1]
        w_q = (w / scale).clamp(-E4M3_MAX, E4M3_MAX)
        return _to_e4m3(w_q), scale.squeeze(-1)

    def dequantized_weight(self) -> Tensor:
        """Return the weight in ``compute_dtype`` after E4M3 fake-quantization."""
        return fake_quant_per_channel(self.weight, self.compute_dtype, dim=0)

    def _quantize_activation(self, x: Tensor) -> Tensor:
        """E4M3 quantize-dequantize of the activation, per-tensor (STE).

        Uses the baked static ``act_scale`` when present (eval/calibrated), else a
        dynamic per-tensor scale computed from ``x`` (training / uncalibrated eval).
        """
        if self.act_scale is not None:
            scale = self.act_scale.to(torch.float32).to(x.device)
            return _fake_quant(x, scale, self.compute_dtype)
        return fake_quant_per_tensor(x, self.compute_dtype)

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype != self.compute_dtype:
            x = x.to(self.compute_dtype)
        x_sim = self._quantize_activation(x)
        w_sim = self.dequantized_weight()  # STE via detached scale in fake-quant
        out = torch.nn.functional.linear(x_sim, w_sim, bias=None)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, *args
    ) -> None:
        # ``act_scale`` is a persistent buffer that starts as ``None``; a None buffer
        # has no state_dict entry, so the default loader can neither offer a slot to
        # load a *baked* scale into (fresh module) nor reconcile one it no longer
        # needs. Align this module's buffer slot with the checkpoint before the
        # standard copy so a calibrated checkpoint round-trips on resume.
        key = prefix + "act_scale"
        val = state_dict.get(key, None)
        if val is not None and self.act_scale is None:
            self.act_scale = torch.empty_like(val, dtype=torch.float32)
        elif val is None and self.act_scale is not None:
            self.act_scale = None
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, *args
        )

    def bake_activation_scale(self, amax: float) -> None:
        """Set a static per-tensor activation scale from calibration ``amax``."""
        amax_t = torch.tensor(float(amax), dtype=torch.float32)
        # Assign into the registered buffer so it travels with .to(device)/.cuda()
        # and is serialised with the checkpoint.
        self.act_scale = _quant_scale(amax_t)

    def extra_repr(self) -> str:
        dyn = (
            "dynamic"
            if self.act_scale is None
            else f"scale={float(self.act_scale):.4e}"
        )
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, compute_dtype={self.compute_dtype}, "
            f"act={dyn}"
        )
