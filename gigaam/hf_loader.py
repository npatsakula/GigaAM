"""Load a GigaAM model from the HuggingFace mirror (``ai-sage/GigaAM-v3``).

The repo is in transformers format: each variant is a git *revision*
(``e2e_ctc``, ``e2e_rnnt``, ``ctc``, ``rnnt``, ``ssl``, ...) containing
``pytorch_model.bin`` + ``config.json`` + ``modeling_gigaam.py``.

The state dict is directly compatible with this package's ``GigaAMASR`` /
``GigaAMEmo`` / ``GigaAM`` modules: keys are identical save for a leading
``model.`` prefix. We download the weights + config, translate the hydra ``_target_``
class paths back to this package, strip the prefix, and ``load_state_dict``.

This is an alternative entry point for environments where the default CDN
(``gigaam.load_model``) is unreachable.
"""

import os
from typing import Optional, Tuple, Union

import omegaconf
import torch

from .model import GigaAM, GigaAMASR, GigaAMEmo

HF_REPO = "ai-sage/GigaAM-v3"

# Variant name -> HF revision.
REVISIONS = {
    "v1_ctc": "ctc",
    "v1_rnnt": "rnnt",
    "v1_ssl": "ssl",
    "v2_ctc": "v2_ctc",
    "v2_rnnt": "v2_rnnt",
    "v2_ssl": "v2_ssl",
    "v3_ctc": "v3_ctc",
    "v3_rnnt": "v3_rnnt",
    "v3_e2e_ctc": "e2e_ctc",
    "v3_e2e_rnnt": "e2e_rnnt",
    "v3_ssl": "ssl_v3",
    "emo": "emo",
    # Short aliases (as accepted by gigaam.load_model). The HF repo carries both a
    # charwise ``rnnt`` revision (num_classes=34) and a SentencePiece ``e2e_rnnt``
    # revision (num_classes=1025); the alias must point to the matching revision.
    "ctc": "e2e_ctc",
    "rnnt": "rnnt",
    "e2e_ctc": "e2e_ctc",
    "e2e_rnnt": "e2e_rnnt",
    "ssl": "ssl_v3",
}

# Map the remote modeling_gigaam _target_ paths to this package's classes.
_TARGET_REMAP = {
    "modeling_gigaam.FeatureExtractor": "gigaam.preprocess.FeatureExtractor",
    "modeling_gigaam.ConformerEncoder": "gigaam.encoder.ConformerEncoder",
    "modeling_gigaam.CTCHead": "gigaam.decoder.CTCHead",
    "modeling_gigaam.RNNTHead": "gigaam.decoder.RNNTHead",
    "modeling_gigaam.RNNTDecoder": "gigaam.decoder.RNNTDecoder",
    "modeling_gigaam.RNNTJoint": "gigaam.decoder.RNNTJoint",
    "modeling_gigaam.CTCGreedyDecoding": "gigaam.decoding.CTCGreedyDecoding",
    "modeling_gigaam.RNNTGreedyDecoding": "gigaam.decoding.RNNTGreedyDecoding",
    "modeling_gigaam.GigaAMASR": "gigaam.model.GigaAMASR",
    "modeling_gigaam.GigaAMEmo": "gigaam.model.GigaAMEmo",
    "modeling_gigaam.GigaAM": "gigaam.model.GigaAM",
}


def _remap_targets(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Rewrite hydra ``_target_`` values to point at this package."""

    def _walk(node):
        if isinstance(node, omegaconf.DictConfig):
            if "_target_" in node and node["_target_"] in _TARGET_REMAP:
                node["_target_"] = _TARGET_REMAP[node["_target_"]]
            for k in list(node.keys()):
                _walk(node[k])
        elif isinstance(node, omegaconf.ListConfig):
            for i in range(len(node)):
                _walk(node[i])

    _walk(cfg)
    return cfg


def _strip_prefix(sd: dict, prefix: str = "model.") -> dict:
    out = {}
    for k, v in sd.items():
        out[k[len(prefix) :] if k.startswith(prefix) else k] = v
    return out


def load_model_from_hf(
    model_name: str,
    fp16_encoder: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    download_root: Optional[str] = None,
) -> Union[GigaAM, GigaAMEmo, GigaAMASR]:
    """Load a GigaAM model from the HuggingFace mirror.

    Parameters mirror ``gigaam.load_model``. ``model_name`` may be a full name
    (``v3_e2e_ctc``) or a short alias (``ctc``); it is mapped to an HF revision.
    """
    from huggingface_hub import hf_hub_download

    if model_name not in REVISIONS:
        raise ValueError(f"Unknown model '{model_name}'. Known: {sorted(REVISIONS)}")
    revision = REVISIONS[model_name]
    # The canonical v3_* name is derived from the loaded config's model_name field
    # (authoritative) rather than from the REVISIONS map, which uses short HF
    # branch names that may not have a v3_-prefixed counterpart (e.g. branch
    # "rnnt" with no "v3_rnnt" key). Fall back to the input name if not a v3 model.
    resolved_name = model_name

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cfg_path = hf_hub_download(HF_REPO, filename="config.json", revision=revision)
    import json

    raw = json.load(open(cfg_path))
    # Nested as config.json ["cfg"]["model"]["cfg"] -> the actual hydra config.
    cfg = omegaconf.OmegaConf.create(raw["cfg"]["model"]["cfg"])

    # The canonical model name (e.g. v3_rnnt, v3_e2e_ctc) is authoritative for
    # SSL/emo class selection and for cfg.model_name. The REVISIONS map uses short
    # HF branch names that may not carry a v3_ prefix, so prefer the config field.
    cfg_model_name = cfg.get("model_name")
    if cfg_model_name:
        resolved_name = cfg_model_name

    # tokenizer path: relative file shipped in the revision.
    if "decoding" in cfg and cfg.decoding.get("model_path") == "tokenizer.model":
        tok_path = hf_hub_download(
            HF_REPO, filename="tokenizer.model", revision=revision
        )
        cfg.decoding.model_path = tok_path

    cfg = _remap_targets(cfg)

    # Instantiate the right model class.
    model_cls_name = cfg.get("model_class", "")
    if "ssl" in resolved_name:
        model = GigaAM(cfg)
    elif "emo" in resolved_name:
        model = GigaAMEmo(cfg)
    else:
        model = GigaAMASR(cfg)

    sd_path = hf_hub_download(HF_REPO, filename="pytorch_model.bin", revision=revision)
    sd = torch.load(sd_path, map_location="cpu", weights_only=True)
    sd = _strip_prefix(sd, "model.")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The preprocessor mel filters/window are buffers that may differ in naming;
    # report only genuine parameter mismatches.
    real_missing = [
        k for k in missing if not k.startswith("preprocessor.featurizer.0.")
    ]
    real_unexpected = [
        k for k in unexpected if not k.startswith("preprocessor.featurizer.0.")
    ]
    if real_missing or real_unexpected:
        print(
            f"  load_state_dict: missing={real_missing[:5]} unexpected={real_unexpected[:5]}"
        )

    model = model.eval()
    if fp16_encoder and device.type != "cpu":
        model.encoder = model.encoder.half()

    # Streaming checkpoints record their mode at the top level of config.json
    # (a ``streaming`` block mirroring the FP8 ``fp8`` block -- NOT in the encoder
    # cfg, which stays pristine so the stock transformers ``modeling_gigaam.py``
    # still instantiates it). Rebuild the encoder in streaming mode from that block.
    # See gigaam.model.enable_streaming and gigaam.fp8.package_hf.
    streaming_meta = raw.get("streaming") if isinstance(raw, dict) else None
    if streaming_meta and not getattr(model.encoder, "streaming", False):
        from .model import enable_streaming

        enable_streaming(
            model,
            chunk_size=streaming_meta.get("chunk_size", 8),
            left_context_chunks=streaming_meta.get("left_context_chunks", -1),
            right_context_chunks=streaming_meta.get("right_context_chunks", 0),
        )

    cfg.model_name = resolved_name
    model.cfg = cfg
    return model.to(device)
