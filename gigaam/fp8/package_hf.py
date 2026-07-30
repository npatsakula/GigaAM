"""Package a FP8-fine-tuned GigaAM checkpoint for HuggingFace.

Produces a directory ready to push as an HF model repo / revision, with two weight
formats plus config and tokenizer:

1. ``pytorch_model.bin`` (fp16, ``model.`` prefix) — the *dequantized* student,
   runnable through the stock ``transformers`` / ``modeling_gigaam.py`` pipeline for
   verification parity. The FP8 latent weights are dequantized to fp16 so this file
   reproduces exactly what the FP8 student computes (E4M3-rounded weights).

2. ``model_fp8.safetensors`` (uint8 E4M3 bytes + per-channel float32 scales) — the
   native FP8 artifact consumed by a custom FP8 runtime (e.g. AMD MI300X). Weights
   are bitcast E4M3; activation scales live in ``activation_scales.json``.

3. ``config.json`` (HF format), ``modeling_gigaam.py`` (copied from the base repo),
   ``tokenizer.model`` — so both formats load under the same config.

Usage
-----
    uv run python -m gigaam.fp8.package_hf \
        --checkpoint path/to/best.ckpt \
        --out_dir hf_package \
        --model_name v3_e2e_ctc_fp8_kd \
        --base_revision e2e_ctc
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import torch

HF_REPO = "ai-sage/GigaAM-v3"


def _load_student_state(ckpt_path: str) -> tuple[dict, dict]:
    """Load a Lightning checkpoint and return (student state_dict, hyper_parameters).

    Strips the frozen teachers (``teacher_encoder.*`` and, for RNNT KD,
    ``teacher_head.*``) and optimizer state, keeping only the student model
    weights. The FP8 Linears are stored as fp32 latent ``Parameter``s in the
    checkpoint.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    student = {
        k: v
        for k, v in sd.items()
        if not k.startswith("teacher_encoder.") and not k.startswith("teacher_head.")
    }
    hp = ck.get("hyper_parameters", {})
    return student, hp


def _streaming_meta_from_hp(hp: dict) -> "Optional[dict]":
    """Build a top-level ``streaming`` config block from checkpoint hyper-parameters,
    or None for a non-streaming checkpoint.

    Mirrors the FP8 ``fp8`` block convention: streaming mode is recorded at the top
    level of config.json, NOT inside the encoder cfg, so the stock transformers
    ``modeling_gigaam.py`` still instantiates the encoder (whose ctor has no
    streaming args). ``gigaam.load_model_from_hf`` reads this block and rebuilds the
    encoder in streaming mode after loading. See gigaam.model.enable_streaming.
    """
    if not hp.get("streaming"):
        return None
    return {
        "enabled": True,
        "chunk_size": hp.get("chunk_size", 8),
        "left_context_chunks": hp.get("left_context_chunks", -1),
        "latency_ms": hp.get("chunk_size", 8) * 40,
        "note": (
            "Cache-aware chunked Conformer (causal convs/subsampling + block "
            "attention). Reconstructed at load by gigaam.load_model_from_hf / "
            "gigaam.enable_streaming. Streaming adds no parameters."
        ),
    }


def _quantize_fp8_weights(state_dict: dict) -> tuple[dict, dict]:
    """Split state_dict into (fp8-bytes+scales dict, plain fp16 dict).

    FP8 latent weights -> uint8 E4M3 + per-channel scale; everything else stays in
    its original dtype (cast to fp16 for the parity bin).
    """
    from gigaam.fp8.linear import _to_e4m3, _from_e4m3, E4M3_MAX
    from gigaam.fp8.quantize import should_quantize, should_quantize_head

    fp8_artifact: dict = {}
    fp16_parity: dict = {}

    for key, tensor in state_dict.items():
        if not key.endswith(".weight"):
            fp16_parity[key] = tensor
            continue
        # Decide whether this weight is an FP8 target. Keys are model-root paths:
        # encoder.* -> strip "encoder." for should_quantize; head.* -> strip "head."
        # for should_quantize_head.
        path = key[: -len(".weight")]
        if path.startswith("encoder."):
            is_fp8 = should_quantize(path[len("encoder.") :])
        elif path.startswith("head."):
            is_fp8 = should_quantize_head(path[len("head.") :])
        else:
            is_fp8 = False

        if is_fp8:
            w = tensor.detach().to(torch.float32)
            amax = w.abs().amax(dim=-1)  # [out]
            scale = (amax / E4M3_MAX).clamp(min=1e-12)
            # Quantize to E4M3 grid, then dequantize: this is exactly what the FP8
            # student computes in forward and what the parity bin must reproduce.
            w_scaled = (w / scale.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX)
            w_deq = _from_e4m3(_to_e4m3(w_scaled), torch.float32) * scale.unsqueeze(-1)
            fp8_artifact[key] = _to_e4m3(w_scaled)  # uint8 E4M3 bytes
            fp8_artifact[key + "_scale"] = scale  # float32 per-channel [out]
            # Parity keeps the E4M3-rounded value in the param's native dtype
            # (fp16 for encoder weights, fp32 for head weights).
            native_dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
            fp16_parity[key] = w_deq.to(native_dtype)
        else:
            fp16_parity[key] = tensor

    return fp8_artifact, fp16_parity


def _fetch_base_config(base_revision: str, out_dir: str) -> dict:
    """Download config.json + modeling_gigaam.py + tokenizer from the base revision."""
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(HF_REPO, filename="config.json", revision=base_revision)
    modeling_path = hf_hub_download(
        HF_REPO, filename="modeling_gigaam.py", revision=base_revision
    )
    shutil.copy(cfg_path, out_dir)
    shutil.copy(modeling_path, out_dir)
    raw_cfg = json.load(open(cfg_path))
    # tokenizer may not exist for all revisions
    try:
        tok_path = hf_hub_download(
            HF_REPO, filename="tokenizer.model", revision=base_revision
        )
        shutil.copy(tok_path, out_dir)
    except Exception:
        pass
    return raw_cfg


def _write_package(
    out_dir: str,
    fp8_artifact: dict,
    fp16_parity: dict,
    model_name: str,
    base_revision: str,
    activation_scales_path: "Optional[str]",
    kd_alpha: "Optional[float]" = None,
    trained_steps: "Optional[int]" = None,
    streaming_meta: "Optional[dict]" = None,
) -> None:
    """Write the HF-ready package artifacts (shared by checkpoint & model paths)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    # 1. parity bin (model. prefix). Preserve each param's native dtype.
    parity_prefixed = {f"model.{k}": v for k, v in fp16_parity.items()}
    print(
        f"Writing parity -> {out / 'pytorch_model.bin'} ({len(parity_prefixed)} tensors)"
    )
    torch.save(parity_prefixed, out / "pytorch_model.bin")

    # 2. native FP8 safetensors (FP8 weights + per-channel scales ONLY).
    fp8_tensors = {f"model.{k}": v.contiguous().cpu() for k, v in fp8_artifact.items()}
    save_file(fp8_tensors, str(out / "model_fp8.safetensors"))
    print(f"Writing native FP8 -> {out / 'model_fp8.safetensors'}")

    # 3. activation scales (copy unless src is already the destination)
    if activation_scales_path and os.path.exists(activation_scales_path):
        dst = out / "activation_scales.json"
        if os.path.abspath(activation_scales_path) != os.path.abspath(dst):
            shutil.copy(activation_scales_path, dst)
        print(f"  activation scales at {dst}")

    # 4. config + modeling + tokenizer from base revision
    print(f"Fetching base config/code from {HF_REPO}@{base_revision} ...")
    raw_cfg = _fetch_base_config(base_revision, str(out))
    fp8_meta = {
        "quant_format": "fp8_e4m3",
        "weight_granularity": "per_output_channel",
        "activation_granularity": "per_tensor",
        "e4m3_max": 448.0,
        "fp8_weight_file": "model_fp8.safetensors",
        "activation_scales_file": "activation_scales.json",
        "kd_alpha": kd_alpha,
        "base_model": f"{HF_REPO}@{base_revision}",
        "trained_steps": trained_steps,
    }
    raw_cfg["fp8"] = fp8_meta
    if streaming_meta:
        raw_cfg["streaming"] = streaming_meta
    raw_cfg["cfg"]["model"]["cfg"]["model_name"] = model_name
    with open(out / "config.json", "w") as f:
        json.dump(raw_cfg, f, indent=2, ensure_ascii=False)
    print("  wrote config.json (with fp8 metadata" + (
        " + streaming" if streaming_meta else "") + ")")

    # 5. index map
    fp8_weight_keys = sorted(
        k for k in fp8_tensors if not k.endswith("_scale") and "_scale" not in k
    )
    index = {
        "quant_format": "fp8_e4m3",
        "fp8_weights": {
            "file": "model_fp8.safetensors",
            "weights": fp8_weight_keys,
            "scales_suffix": "_scale",
        },
        "fp16_params": {
            "file": "pytorch_model.bin",
            "note": "all non-FP8 parameters (norms, biases, convs, embedding, head, preprocessor)",
        },
        "activation_scales_file": "activation_scales.json",
    }
    (out / "fp8_index.json").write_text(json.dumps(index, indent=2))
    print("  wrote fp8_index.json")

    print("\n=== Package ready ===")
    for f in sorted(os.listdir(out)):
        sz = os.path.getsize(out / f)
        print(f"  {f:30s} {sz/1e6:8.1f} MB")
    print(f"\nPush to HF with:")
    print(f"  huggingface-cli upload <your-repo> {out} --revision {model_name}")


def _fetch_config_and_meta(
    out_dir: str,
    model_name: str,
    base_revision: str,
    kd_alpha: "Optional[float]" = None,
    trained_steps: "Optional[int]" = None,
    streaming_meta: "Optional[dict]" = None,
) -> dict:
    """Fetch config/modeling/tokenizer from the base revision and write config.json.

    Shared by the HF-convention and native-convention packagers. Returns the raw
    config dict (already written to ``out_dir/config.json`` with FP8 metadata,
    and a streaming block when ``streaming_meta`` is set).
    """
    out = Path(out_dir)
    print(f"Fetching base config/code from {HF_REPO}@{base_revision} ...")
    raw_cfg = _fetch_base_config(base_revision, str(out))
    fp8_meta = {
        "quant_format": "fp8_e4m3",
        "weight_granularity": "per_output_channel",
        "activation_granularity": "per_tensor",
        "e4m3_max": 448.0,
        "fp8_weight_file": "model_fp8.safetensors",
        "activation_scales_file": "activation_scales.json",
        "kd_alpha": kd_alpha,
        "base_model": f"{HF_REPO}@{base_revision}",
        "trained_steps": trained_steps,
    }
    raw_cfg["fp8"] = fp8_meta
    if streaming_meta:
        raw_cfg["streaming"] = streaming_meta
    raw_cfg["cfg"]["model"]["cfg"]["model_name"] = model_name
    with open(out / "config.json", "w") as f:
        json.dump(raw_cfg, f, indent=2, ensure_ascii=False)
    print("  wrote config.json (with fp8 metadata" + (
        " + streaming" if streaming_meta else "") + ")")
    return raw_cfg


def package_checkpoint(
    ckpt_path: str,
    out_dir: str,
    model_name: str,
    base_revision: str = "e2e_ctc",
) -> None:
    """Package a FP8 fine-tuned Lightning checkpoint into an HF-ready directory."""
    print(f"Loading checkpoint {ckpt_path} ...")
    student_sd, hp = _load_student_state(ckpt_path)
    print(f"  student state_dict: {len(student_sd)} tensors")
    print("Splitting FP8 weights (uint8+scales) from fp16 parity ...")
    fp8_artifact, fp16_parity = _quantize_fp8_weights(student_sd)
    n_fp8 = len([k for k in fp8_artifact if not k.endswith("_scale")])
    print(f"  FP8 weights: {n_fp8} | parity tensors: {len(fp16_parity)}")
    _write_package(
        out_dir=out_dir,
        fp8_artifact=fp8_artifact,
        fp16_parity=fp16_parity,
        model_name=model_name,
        base_revision=base_revision,
        activation_scales_path=hp.get("calibration_scales"),
        kd_alpha=hp.get("kd_alpha"),
        trained_steps=hp.get("max_steps"),
        streaming_meta=_streaming_meta_from_hp(hp),
    )


def _to_shipped_dtype(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """Cast a student param to the dtype convention of the shipped artifacts.

    The KD checkpoint stores the student in fp32 (FP8Linear latent weights are
    fp32 by design, and the encoder is loaded fp32 for training). The shipped
    native-convention ``model.safetensors`` uses fp16 for the encoder and fp32
    for the head/preprocessor (matching ``load_model_from_hf``, which only halves
    the encoder). Apply that policy so the parity file matches in size/dtype.
    """
    if not tensor.is_floating_point():
        return tensor
    if key.startswith("encoder."):
        return tensor.to(torch.float16)
    return tensor  # head.* / preprocessor.* stay fp32


def package_checkpoint_native(
    ckpt_path: str,
    out_dir: str,
    model_name: str,
    base_revision: str,
    activation_scales_path: "Optional[str]" = None,
    kd_alpha: "Optional[float]" = None,
    trained_steps: "Optional[int]" = None,
) -> None:
    """Package a FP8 KD checkpoint in the *native* convention (no ``model.`` prefix).

    Mirrors :func:`package_checkpoint` (same student state loading and FP8/parity
    split) but emits the file set a custom runtime consumes directly:

    - ``model.safetensors`` — full fp16 student (parity; FP8 weights dequantized
      to their native dtype). No prefix.
    - ``model_fp8.safetensors`` — FP8 weights as ``uint8`` + per-channel
      ``<key>_scale`` float32, plus the non-FP8 params in their native dtype so
      the file is self-contained. No prefix. Smaller than ``model.safetensors``.
    - ``model_fp8.safetensors.activation_scales.json`` — per-tensor activation
      amax (copied from ``activation_scales_path``).
    - ``config.json``, ``modeling_gigaam.py``, ``tokenizer.model`` from the base.

    Keys are the raw PyTorch state-dict names, matching ``pt_to_safetensors.py``.
    """
    print(f"Loading checkpoint {ckpt_path} ...")
    student_sd, hp = _load_student_state(ckpt_path)
    print(f"  student state_dict: {len(student_sd)} tensors")
    print("Splitting FP8 weights (uint8+scales) from fp16 parity ...")
    fp8_artifact, fp16_parity = _quantize_fp8_weights(student_sd)
    n_fp8 = len([k for k in fp8_artifact if not k.endswith("_scale")])
    print(f"  FP8 weights: {n_fp8} | parity tensors: {len(fp16_parity)}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    # 1. model.safetensors — full student (no prefix). Cast to the shipped dtype
    #    convention (encoder fp16, head/preprocessor fp32) so the file matches the
    #    PTQ artifacts in size and dtype.
    model_tensors = {
        k: _to_shipped_dtype(k, v).contiguous().cpu() for k, v in fp16_parity.items()
    }
    save_file(model_tensors, str(out / "model.safetensors"))
    print(f"Writing model.safetensors -> {out / 'model.safetensors'} ({len(model_tensors)} tensors)")

    # 2. model_fp8.safetensors — self-contained FP8 file: the FP8 weights as uint8
    #    + per-channel ``<key>_scale`` float32, PLUS all non-FP8 params in their
    #    native dtype. This matches the convention export_native.model_to_safetensors
    #    uses and what the shipped matrix artifacts were verified to contain: U8 for
    #    FP8 weights, F16 for non-FP8 encoder params, F32 for head/scales.
    fp8_weight_keys = {k for k in fp8_artifact if not k.endswith("_scale")}
    fp8_tensors = {k: v.contiguous().cpu() for k, v in fp8_artifact.items()}
    for k, v in fp16_parity.items():
        # Skip the fp16 copy of any weight that is represented as uint8 here.
        if k in fp8_weight_keys:
            continue
        fp8_tensors[k] = _to_shipped_dtype(k, v).contiguous().cpu()
    save_file(fp8_tensors, str(out / "model_fp8.safetensors"))
    print(f"Writing model_fp8.safetensors -> {out / 'model_fp8.safetensors'} ({len(fp8_tensors)} tensors)")

    # 3. activation scales sidecar (named after the fp8 file per the export convention).
    if activation_scales_path and os.path.exists(activation_scales_path):
        dst = out / "model_fp8.safetensors.activation_scales.json"
        if os.path.abspath(activation_scales_path) != os.path.abspath(dst):
            shutil.copy(activation_scales_path, dst)
        print(f"  activation scales at {dst}")

    # 4. config + modeling + tokenizer from base revision.
    _fetch_config_and_meta(
        str(out),
        model_name=model_name,
        base_revision=base_revision,
        kd_alpha=kd_alpha,
        trained_steps=trained_steps,
        streaming_meta=_streaming_meta_from_hp(hp),
    )

    print("\n=== Native package ready ===")
    for f in sorted(os.listdir(out)):
        sz = os.path.getsize(out / f)
        print(f"  {f:42s} {sz/1e6:8.1f} MB")


def package_from_model(
    model,
    out_dir: str,
    model_name: str,
    base_revision: str,
    activation_scales_path: "Optional[str]" = None,
) -> None:
    """Package a live, already-quantized model (PTQ path) into an HF-ready directory.

    Unlike :func:`package_checkpoint` (which reads a Lightning checkpoint), this
    operates on a model whose FP8Linears are already in place. It reads the FP8
    bytes/scales directly from each FP8Linear and the unquantized params from the
    model's state_dict.
    """
    from gigaam.fp8.linear import FP8Linear

    # Build the model-root state_dict, replacing FP8 latent weights with their
    # quantized bytes + scales, and keeping everything else in native dtype.
    fp8_artifact: dict = {}
    fp16_parity: dict = {}
    fp8_paths = {name for name, m in model.named_modules() if isinstance(m, FP8Linear)}
    for full_name, tensor in model.state_dict().items():
        path = (
            full_name[: -len(".weight")] if full_name.endswith(".weight") else full_name
        )
        if path in fp8_paths and full_name.endswith(".weight"):
            mod = model.get_submodule(path)
            wq, wscale = mod.quantize_weight_bytes()
            fp8_artifact[full_name] = wq
            fp8_artifact[full_name + "_scale"] = wscale
            fp16_parity[full_name] = mod.dequantized_weight().to(tensor.dtype)
        else:
            fp16_parity[full_name] = tensor
    n_fp8 = len([k for k in fp8_artifact if not k.endswith("_scale")])
    print(f"  FP8 weights: {n_fp8} | parity tensors: {len(fp16_parity)}")
    _write_package(
        out_dir=out_dir,
        fp8_artifact=fp8_artifact,
        fp16_parity=fp16_parity,
        model_name=model_name,
        base_revision=base_revision,
        activation_scales_path=activation_scales_path,
    )


def main():
    p = argparse.ArgumentParser(description="Package a FP8 checkpoint for HuggingFace.")
    p.add_argument("--checkpoint", required=True, help="Path to the .ckpt file.")
    p.add_argument("--out_dir", required=True, help="Output directory for the package.")
    p.add_argument("--model_name", default="v3_e2e_ctc_fp8_kd")
    p.add_argument(
        "--base_revision",
        default="e2e_ctc",
        help="HF revision to copy config/modeling/tokenizer from.",
    )
    args = p.parse_args()
    package_checkpoint(
        ckpt_path=args.checkpoint,
        out_dir=args.out_dir,
        model_name=args.model_name,
        base_revision=args.base_revision,
    )


if __name__ == "__main__":
    main()
