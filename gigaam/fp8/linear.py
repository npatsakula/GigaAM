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
  weight and the activation before the matmul, in ``compute_dtype`` (fp16 on GPU).
  The straight-through estimator (STE) makes the non-differentiable rounding pass
  gradients through as the identity: the latent weight learns to compensate for
  the E4M3 grid.
- Weights use *per-output-channel* scales: ``scale = amax_per_channel / E4M3_MAX``.
  Per-tensor FP8 weights lose noticeable accuracy on Conformer outlier channels.
- Activations use a *per-tensor* scale: ``scale = amax / E4M3_MAX``. In training it
  is recomputed dynamically from the live tensor; in eval it is taken from a scale
  baked in by calibration (or falls back to dynamic if unset).

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


def fake_quant_per_tensor(x: Tensor, dtype: torch.dtype) -> Tensor:
    """Faithful E4M3 per-tensor quantize-dequantize of ``x`` (STE).

    The quantize-dequant value is computed on a detached tensor (so the
    non-differentiable fp8 cast stays out of the autograd graph), then bridged
    with the straight-through estimator ``x + (xq - x).detach()``: the forward
    equals ``xq`` (the rounded value) while the backward is the identity.
    """
    amax = x.detach().abs().amax()
    scale = (amax / E4M3_MAX).clamp(min=1e-12).to(dtype)
    x_d = x.detach().to(dtype)
    x_q = _from_e4m3(_to_e4m3((x_d / scale).clamp(-E4M3_MAX, E4M3_MAX)), dtype) * scale
    return x.to(dtype) + (x_q - x.to(dtype)).detach()


def fake_quant_per_channel(x: Tensor, dtype: torch.dtype, dim: int = 0) -> Tensor:
    """Per-channel E4M3 quantize-dequantize along ``dim`` (default: output rows)."""
    reduce_dims = [i for i in range(x.dim()) if i != dim]
    amax = x.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = (amax / E4M3_MAX).clamp(min=1e-12).to(dtype)
    x_d = x.detach().to(dtype)
    x_q = _from_e4m3(_to_e4m3((x_d / scale).clamp(-E4M3_MAX, E4M3_MAX)), dtype) * scale
    return x.to(dtype) + (x_q - x.to(dtype)).detach()


class FP8Linear(nn.Module):
    """``nn.Linear`` with E4M3-simulated FP8 weights and activations.

    Parameters
    ----------
    in_features, out_features : int
        As ``nn.Linear``.
    bias : bool
        Whether to include a bias term (kept in float32, unquantized).
    compute_dtype : torch.dtype
        Dtype the simulated matmul runs in (float16 on GPU, float32 on CPU).

    Attributes
    ----------
    weight : nn.Parameter (float32)
        Latent weight, shape ``[out, in]``. Updated by the optimizer during QAT;
        fake-quantized to E4M3 in forward. Quantized to uint8 bytes on export.
    act_scale : Tensor or None
        Per-tensor activation scale baked by calibration. Registered as a
        non-persistent buffer so it moves with the module but is excluded from
        ``state_dict()`` (scales are serialised via the activation_scales.json
        sidecar on export). When ``None`` the activation scale is computed
        dynamically from each input tensor.
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
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)  # type: ignore[arg-type]

        # Activation scale: None -> dynamic; set by calibration -> static (eval).
        # Registered as a non-persistent buffer so it moves with .to(device)/.cuda()
        # (a plain attribute would be stranded on CPU after a device move) but is
        # deliberately excluded from state_dict() — activation scales are serialised
        # separately to the activation_scales.json sidecar on export, not via the
        # weight checkpoint.
        self.register_buffer("act_scale", None, persistent=False)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, compute_dtype: torch.dtype = torch.float16
    ) -> "FP8Linear":
        """Build an ``FP8Linear`` initialised from a pretrained ``nn.Linear``.

        The fp32 latent weight is copied verbatim; fake quantization is applied in
        ``forward``. Export quantizes the latent weight to E4M3 bytes.
        """
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
        layer = layer.to(linear.weight.device)
        return layer

    def quantize_weight_bytes(self) -> tuple[Tensor, Tensor]:
        """Quantize the latent weight to E4M3 bytes + per-channel scale.

        Returns ``(weight_q [uint8], weight_scale [float32, out])``. This is exactly
        what the export emits and what the MI300X runtime consumes.
        """
        w = self.weight.detach().to(torch.float32)
        amax = w.abs().amax(dim=-1)  # [out]
        scale = (amax / E4M3_MAX).clamp(min=1e-12)
        w_q = (w / scale.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX)
        return _to_e4m3(w_q), scale

    def dequantized_weight(self) -> Tensor:
        """Return the weight in ``compute_dtype`` after E4M3 fake-quantization."""
        return fake_quant_per_channel(self.weight, self.compute_dtype, dim=0)

    def _quantize_activation(self, x: Tensor) -> Tensor:
        """E4M3 quantize-dequantize of the activation, per-tensor (STE)."""
        dtype = self.compute_dtype
        if self.act_scale is not None:
            scale = self.act_scale.to(dtype).to(x.device)
            x_d = x.detach().to(dtype)
            x_q = (
                _from_e4m3(_to_e4m3((x_d / scale).clamp(-E4M3_MAX, E4M3_MAX)), dtype)
                * scale
            )
            return x.to(dtype) + (x_q - x.to(dtype)).detach()
        # Dynamic scaling (training / uncalibrated eval).
        return fake_quant_per_tensor(x, dtype)

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype != self.compute_dtype:
            x = x.to(self.compute_dtype)
        x_sim = self._quantize_activation(x)
        w_sim = self.dequantized_weight()  # STE via detached scale in fake-quant
        out = torch.nn.functional.linear(x_sim, w_sim, bias=None)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def bake_activation_scale(self, amax: float) -> None:
        """Set a static per-tensor activation scale from calibration ``amax``."""
        scale = max(float(amax), 1e-12) / E4M3_MAX
        # Assign into the registered buffer (not a bare attribute) so the scale
        # travels with .to(device)/.cuda() and stays out of state_dict().
        self.act_scale = torch.tensor(scale, dtype=torch.float32)

    def extra_repr(self) -> str:
        dyn = (
            "dynamic"
            if self.act_scale is None
            else f"scale={float(self.act_scale):.4e}"
        )
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"compute_dtype={self.compute_dtype}, act={dyn}"
        )
