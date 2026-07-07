"""Export a GigaAM model to a safetensors in the native framework convention.

Mirrors ``pt_to_safetensors.py``: keys are the raw PyTorch state-dict names (no
``model.`` prefix), dtypes preserved. For a FP8-quantized model the FP8 weights are
emitted as ``uint8`` (E4M3 bitcast) + a ``<key>_scale`` per-channel float32 tensor;
all other parameters keep their native dtype and key name.

The result is a single self-contained ``.safetensors`` loadable by a custom runtime
that casts the uint8 bytes to FP8 and issues MFMA using the per-channel scales and
the per-tensor activation scales (written alongside as JSON).

Usage (original, unquantized model)::

    uv run python -m gigaam.fp8.export_native \\
        --model_name e2e_ctc --from_hf --out original_e2e_ctc.safetensors

Usage (FP8 PTQ, quantized + calibrated)::

    uv run python -m gigaam.fp8.export_native \\
        --model_name e2e_ctc --from_hf --fp8 \\
        --manifest data/sub/manifest_train.tsv --n_calibration_batches 32 \\
        --out e2e_ctc_fp8.safetensors
"""

import argparse
import json
import os
from typing import Dict, Optional

import torch
from torch import nn


def model_to_safetensors(
    model: nn.Module,
    out_path: str,
    activation_scales: Optional[Dict[str, float]] = None,
    fp8_as_dequant: bool = False,
) -> None:
    """Save ``model.state_dict()`` to safetensors.

    Keys are the raw PyTorch names (no prefix). Non-FP8 params keep their native
    dtype. FP8Linear weights are written according to ``fp8_as_dequant``:

    - ``False`` (default): the FP8 weight as a ``uint8`` tensor (E4M3 bitcast) at
      the same key plus a ``<key>_scale`` per-channel float32 tensor. This is the
      native FP8 artifact a custom runtime consumes.
    - ``True``: the FP8 weight *dequantized* to its native compute dtype (the
      E4M3-rounded value, what the layer actually computes), with no scale
      tensor. Use this for the fp16 ``model.safetensors`` parity file.

    If ``activation_scales`` is given, it is written as a sibling JSON file
    (``<out>.activation_scales.json``) keyed by model-root module path.
    """
    from safetensors.torch import save_file
    from gigaam.fp8.linear import FP8Linear

    fp8_paths = {name for name, m in model.named_modules() if isinstance(m, FP8Linear)}
    tensors: Dict[str, torch.Tensor] = {}

    for full_name, tensor in model.state_dict().items():
        path = (
            full_name[: -len(".weight")] if full_name.endswith(".weight") else full_name
        )
        if path in fp8_paths and full_name.endswith(".weight"):
            mod = model.get_submodule(path)
            if fp8_as_dequant:
                # E4M3-rounded weight in its compute dtype (no scale tensor).
                tensors[full_name] = mod.dequantized_weight().contiguous().cpu()
            else:
                wq, wscale = mod.quantize_weight_bytes()
                tensors[full_name] = wq.contiguous().cpu()  # uint8
                tensors[full_name + "_scale"] = wscale.contiguous().cpu()  # float32
        else:
            tensors[full_name] = tensor.contiguous().cpu()

    save_file(tensors, out_path)
    n_fp8 = sum(1 for k in fp8_paths if k + ".weight" in model.state_dict())
    sz_mb = os.path.getsize(out_path) / 1e6
    tag = "dequant" if fp8_as_dequant else "uint8+scales"
    print(f"Wrote {out_path} ({len(tensors)} tensors, {n_fp8} FP8 [{tag}], {sz_mb:.1f} MB)")

    if activation_scales is not None:
        sidecar = out_path + ".activation_scales.json"
        with open(sidecar, "w") as f:
            json.dump(activation_scales, f, indent=2)
        print(f"Wrote {sidecar} ({len(activation_scales)} activation scales)")


def main():
    p = argparse.ArgumentParser(
        description="Export a GigaAM model to native-convention safetensors."
    )
    p.add_argument("--model_name", required=True)
    p.add_argument(
        "--from_hf",
        action="store_true",
        help="Load from the HuggingFace mirror instead of the CDN.",
    )
    p.add_argument("--out", required=True, help="Output .safetensors path.")
    p.add_argument(
        "--fp8",
        action="store_true",
        help="Quantize to FP8 E4M3 (PTQ) before exporting.",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="TSV manifest for activation calibration (FP8 mode).",
    )
    p.add_argument("--n_calibration_batches", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    import gigaam
    from torch.utils.data import DataLoader
    from gigaam.utils import AudioDataset

    if args.from_hf:
        model = gigaam.load_model_from_hf(
            args.model_name, fp16_encoder=True, device=args.device
        )
    else:
        model = gigaam.load_model(
            args.model_name, fp16_encoder=True, device=args.device
        )
    model.eval()
    # Match the matrix-run convention: the encoder ships as fp16 (non-FP8 params
    # are fp16, FP8 weights are uint8). load_model_from_hf only halves on a CUDA
    # device, so force it here to stay correct on CPU and for the FP8 path.
    model.encoder = model.encoder.half()

    activation_scales = None
    if args.fp8:
        from gigaam.fp8 import bake_activation_scales, quantize_model
        from gigaam.fp8.calibrate import ActivationCalibrator

        quantize_model(model, compute_dtype=torch.float16)
        if args.manifest:
            assert os.path.exists(args.manifest), f"manifest not found: {args.manifest}"
            ds = AudioDataset(args.manifest)
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=AudioDataset.collate,
                num_workers=4,
            )
            calib = ActivationCalibrator(model)
            with torch.no_grad():
                for i, (wavs, lens) in enumerate(dl):
                    model(wavs.to(args.device), lens.to(args.device))
                    if i + 1 >= args.n_calibration_batches:
                        break
            calib.remove()
            bake_activation_scales(model, calib.amax)
            activation_scales = calib.amax
            print(f"Calibrated {len(activation_scales)} activation scales")
        else:
            print("FP8 export without --manifest: activation scales will be dynamic")

    model_to_safetensors(model, args.out, activation_scales=activation_scales)


if __name__ == "__main__":
    main()
