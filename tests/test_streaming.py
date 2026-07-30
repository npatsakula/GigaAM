"""Streaming-encoder correctness tests.

The first three tests build small randomly-initialised encoders (or a synthetic
CTC decoder) and need no network / weights. The last one is marked ``partial``
and exercises the real v3_ctc checkpoint end-to-end.
"""
import logging

import pytest
import torch

from gigaam.decoding import CTCGreedyDecoding
from gigaam.encoder import ConformerEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# v3_ctc / v3_e2e_ctc encoder architecture (shared): conv1d k=5, layer_norm, rotary.
ENC_KW = dict(
    feat_in=64,
    n_layers=4,
    d_model=768,
    subsampling="conv1d",
    subs_kernel_size=5,
    subsampling_factor=4,
    self_attention_model="rotary",
    n_heads=16,
    conv_kernel_size=5,
    conv_norm_type="layer_norm",
    flash_attn=False,
)
CHUNK = 8  # encoder frames -> 320 ms
SUB = 4  # subsampling factor -> feature frames per encoder frame


def _streaming_encoder():
    torch.manual_seed(0)
    return ConformerEncoder(
        **ENC_KW, streaming=True, chunk_size=CHUNK, left_context_chunks=-1
    ).eval()


def test_zero_new_params():
    """Streaming must add no parameters: an offline and a streaming encoder share
    an identical state_dict."""
    torch.manual_seed(0)
    off = ConformerEncoder(**ENC_KW).eval()
    torch.manual_seed(0)
    strm = ConformerEncoder(
        **ENC_KW, streaming=True, chunk_size=CHUNK, left_context_chunks=-1
    ).eval()
    strm.load_state_dict(off.state_dict(), strict=True)  # raises if keys differ


def test_no_future_leak():
    """Perturbing the audio tail past a chunk boundary must not change any encoder
    output strictly before that boundary (validates causal conv/subsampling +
    block attention). An offline encoder, by contrast, must change."""
    enc = _streaming_encoder()
    torch.manual_seed(1)
    off = ConformerEncoder(**ENC_KW).eval()
    off.load_state_dict(enc.state_dict(), strict=True)

    feat = 200
    x = torch.randn(1, 64, feat)
    lens = torch.tensor([feat])
    with torch.inference_mode():
        ys, _ = enc(x, lens)
        yo, _ = off(x, lens)
        k = 3  # perturb from encoder frame k*CHUNK onward
        bnd_enc = k * CHUNK
        x2 = x.clone()
        x2[:, :, bnd_enc * SUB :] = torch.randn_like(x2[:, :, bnd_enc * SUB :])
        ys2, _ = enc(x2, lens)
        yo2, _ = off(x2, lens)

    leak = (ys[:, :, :bnd_enc] - ys2[:, :, :bnd_enc]).abs().max().item()
    off_change = (yo[:, :, :bnd_enc] - yo2[:, :, :bnd_enc]).abs().max().item()
    assert leak < 1e-5, f"future leaked into past frames: {leak}"
    assert off_change > 1e-3, "offline sanity: perturbation should have mattered"


@pytest.mark.parametrize("feat_frames", [128, 200, 256, 259, 323])
def test_encoder_cache_parity(feat_frames):
    """Chunk-by-chunk cache inference == full masked forward, at chunk-aligned and
    non-multiple lengths."""
    enc = _streaming_encoder()
    x = torch.randn(1, 64, feat_frames)
    lens = torch.tensor([feat_frames])
    with torch.inference_mode():
        full, _ = enc(x, lens)
        state = enc.init_streaming_state()
        outs = []
        for s in range(0, feat_frames, CHUNK * SUB):
            outs.append(enc.forward_streaming_chunk(x[:, :, s : s + CHUNK * SUB], state))
        stream = torch.cat(outs, dim=2)
    assert full.shape == stream.shape
    diff = (full - stream).abs().max().item()
    assert diff < 1e-4, f"cache parity broke: max|diff|={diff}"


@pytest.mark.parametrize("left_context_chunks", [1, 2, 4])
@pytest.mark.parametrize("feat_frames", [128, 200, 323])
def test_bounded_left_context_cache_parity(left_context_chunks, feat_frames):
    """A *bounded* left-context window must still reproduce the full masked
    forward. The streaming attention cache must be evicted to the last
    ``left_context_chunks`` chunks; without eviction it silently keeps the whole
    past and diverges from the trained block mask."""
    torch.manual_seed(0)
    enc = ConformerEncoder(
        **ENC_KW,
        streaming=True,
        chunk_size=CHUNK,
        left_context_chunks=left_context_chunks,
    ).eval()
    x = torch.randn(1, 64, feat_frames)
    lens = torch.tensor([feat_frames])
    with torch.inference_mode():
        full, _ = enc(x, lens)
        state = enc.init_streaming_state()
        outs = []
        for s in range(0, feat_frames, CHUNK * SUB):
            outs.append(enc.forward_streaming_chunk(x[:, :, s : s + CHUNK * SUB], state))
        stream = torch.cat(outs, dim=2)
    assert full.shape == stream.shape
    diff = (full - stream).abs().max().item()
    assert diff < 1e-4, (
        f"bounded left_context={left_context_chunks}: cache parity broke: "
        f"max|diff|={diff} (cache not evicted to the trained window?)"
    )


def test_incremental_collapse_parity():
    """Incremental cross-chunk CTC collapse == offline whole-sequence collapse on
    the same (synthetic) logits."""
    torch.manual_seed(0)
    vocab = [chr(ord("a") + i) for i in range(12)]
    dec = CTCGreedyDecoding(vocab)  # blank_id = len(vocab) = 12
    C = len(vocab) + 1
    T = 137
    log_probs = torch.log_softmax(torch.randn(1, T, C), dim=-1)

    # offline reference: reuse the same collapse rule as decode()
    labels = log_probs.argmax(-1)[0]
    keep = labels != dec.blank_id
    keep[1:] &= labels[1:] != labels[:-1]
    ref_ids = labels[keep].tolist()

    ids, last = [], dec.blank_id
    for s in range(0, T, CHUNK):
        new_ids, _, last = dec.decode_chunk(log_probs[:, s : s + CHUNK, :], last, s)
        ids.extend(new_ids)
    assert ids == ref_ids
    assert dec.tokenizer.decode(ids) == dec.tokenizer.decode(ref_ids)


@pytest.mark.partial
def test_real_model_streaming_parity():
    """End-to-end: on real v3_ctc weights patched into streaming mode, the
    chunk-by-chunk driver reproduces the full streaming-masked forward's text."""
    import gigaam

    model = gigaam.load_model("v3_ctc", device="cpu", fp16_encoder=False).eval()
    gigaam.enable_streaming(model, chunk_size=CHUNK, left_context_chunks=-1)

    # deterministic ~4 s tone so the test needs no data files
    import numpy as np

    sr = 16000
    t = np.linspace(0, 4.0, sr * 4, dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    import soundfile as sf
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        path = f.name

    full = model.transcribe(path).text
    stream = model.transcribe_streaming(path).text
    assert full == stream, f"driver text parity: {full!r} != {stream!r}"


@pytest.mark.parametrize("T", [37, 64, 65])
def test_rnnt_decode_chunk_parity(T):
    """Chunk-by-chunk RNN-T greedy decode == offline whole-sequence decode, at
    chunk-aligned and non-multiple lengths. Isolates decode correctness (predictor
    state + last label carried across chunk boundaries) from the encoder."""
    from gigaam.decoder import RNNTHead
    from gigaam.decoding import RNNTGreedyDecoding

    D = 16
    head = RNNTHead(
        decoder={"pred_hidden": D, "pred_rnn_layers": 1, "num_classes": 8},
        joint={"enc_hidden": D, "pred_hidden": D, "joint_hidden": D, "num_classes": 8},
    ).eval()
    dec = RNNTGreedyDecoding(vocabulary=[chr(97 + i) for i in range(7)])  # blank=7

    torch.manual_seed(0)
    encoded = torch.randn(1, D, T)  # channels-first [1, D, T]
    enc_len = torch.tensor([T])

    with torch.inference_mode():
        ref_ids = dec.decode(head, encoded, enc_len)[0][1]

        ids, dec_state, last_label = [], None, dec.blank_id
        for off in range(0, T, CHUNK):
            chunk = encoded[:, :, off : off + CHUNK]
            new_ids, _, dec_state, last_label = dec.decode_chunk(
                head, chunk, dec_state, last_label, off
            )
            ids.extend(new_ids)

    assert ids == ref_ids, f"RNN-T decode parity broke: {ids[:10]} != {ref_ids[:10]}"


@pytest.mark.parametrize("revision", ["v3_rnnt", "v3_e2e_rnnt"])
@pytest.mark.partial
def test_real_model_rnnt_streaming_parity(revision):
    """End-to-end: on real RNN-T weights patched into streaming mode, the
    chunk-by-chunk driver reproduces the streaming-masked forward's text. Covers
    v3_e2e_rnnt (punctuation/normalization) with the same code path for free."""
    import gigaam

    model = gigaam.load_model(revision, device="cpu", fp16_encoder=False).eval()
    gigaam.enable_streaming(model, chunk_size=CHUNK, left_context_chunks=-1)

    import numpy as np
    import soundfile as sf
    import tempfile

    sr = 16000
    t = np.linspace(0, 4.0, sr * 4, dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        path = f.name

    full = model.transcribe(path).text
    stream = model.transcribe_streaming(path).text
    assert full == stream, f"[{revision}] driver text parity: {full!r} != {stream!r}"


def test_enable_streaming_refuses_fp8_encoder():
    """enable_streaming rebuilds the encoder fresh (plain nn.Linear), so it must
    run BEFORE FP8 quantization. Calling it on an already-quantized encoder would
    silently replace every FP8Linear with a plain Linear (act_scale is None while
    uncalibrated, hence omitted from the state_dict, so the strict load succeeds
    while wiping the quantization). It must refuse loudly instead."""
    from gigaam.fp8.linear import FP8Linear
    from gigaam.model import enable_streaming

    enc = _streaming_encoder()
    # Swap one Linear for an FP8Linear to simulate a post-quantize encoder.
    enc.layers[0].feed_forward1.linear1 = FP8Linear.from_linear(
        enc.layers[0].feed_forward1.linear1, compute_dtype=torch.float32
    )

    class _StubModel:
        def __init__(self, encoder):
            self.encoder = encoder

    with pytest.raises(RuntimeError, match="BEFORE gigaam.fp8.quantize_model"):
        enable_streaming(_StubModel(enc), chunk_size=CHUNK, left_context_chunks=-1)


@pytest.mark.parametrize("left_context_chunks", [-1, 2])
def test_fp8_streaming_composes(left_context_chunks):
    """FP8 quantization composed with streaming (enable_streaming -> quantize,
    the only valid order) must run end-to-end through the cache-aware path:
    the streaming FP8 encoder produces finite, correctly-shaped output for both
    bounded and unbounded left context.

    Note this deliberately does NOT assert chunk-vs-full parity: per-tensor FP8
    activation quantization is applied to different input distributions (a chunk
    vs the whole sequence), so the chunked and full FP8 forwards legitimately
    differ. The pure-encoder cache parity (no FP8) is covered by the other
    parity tests; this test covers the FP8+streaming *integration*."""
    from gigaam.fp8 import FP8Linear, quantize_model

    torch.manual_seed(0)
    enc = ConformerEncoder(
        **ENC_KW,
        streaming=True,
        chunk_size=CHUNK,
        left_context_chunks=left_context_chunks,
    ).eval()
    # Wrap in a minimal structure so quantize_model can find ``.encoder``.
    import torch.nn as nn

    class _Wrap(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

    quantize_model(_Wrap(enc), compute_dtype=torch.float32)
    assert any(isinstance(m, FP8Linear) for _, m in enc.named_modules()), (
        "quantize_model swapped in no FP8Linear modules"
    )

    feat_frames = 200
    x = torch.randn(1, 64, feat_frames)
    lens = torch.tensor([feat_frames])
    with torch.inference_mode():
        full, _ = enc(x, lens)
        state = enc.init_streaming_state()
        outs = []
        for s in range(0, feat_frames, CHUNK * SUB):
            outs.append(enc.forward_streaming_chunk(x[:, :, s : s + CHUNK * SUB], state))
        stream = torch.cat(outs, dim=2)
    assert full.shape == stream.shape
    assert torch.isfinite(stream).all(), "streaming FP8 output has NaN/Inf"
    # The two paths track each other in magnitude (same order, finite).
    rel = (full - stream).abs().max().item() / full.abs().max().item()
    assert rel < 0.5, f"FP8 chunked path diverged wildly from full: rel={rel:.2f}"


def test_hf_streaming_config_round_trip():
    """A streaming checkpoint published to HF round-trips: package_hf records the
    mode as a top-level ``streaming`` block (NOT in the encoder cfg, so the stock
    transformers modeling_gigaam.py still instantiates it), and load_model_from_hf's
    restore logic rebuilds streaming mode from that block. No network needed: this
    exercises the config helper + the loader's restore branch on a built config."""
    from gigaam.fp8.package_hf import _streaming_meta_from_hp
    from gigaam.hf_loader import _remap_targets

    # --- packaging side: a streaming checkpoint's hyper-parameters -> config block ---
    hp = {"streaming": True, "chunk_size": 8, "left_context_chunks": -1}
    streaming_meta = _streaming_meta_from_hp(hp)
    assert streaming_meta is not None and streaming_meta["enabled"] is True
    assert streaming_meta["chunk_size"] == 8

    # Non-streaming checkpoint yields no block.
    assert _streaming_meta_from_hp({"model_name": "v3_ctc"}) is None

    # --- the published config.json: base config + top-level streaming block ---
    import omegaconf

    import gigaam

    raw = {
        "cfg": {"model": {"cfg": omegaconf.OmegaConf.to_container(
            gigaam.load_model("v3_ctc", device="cpu", fp16_encoder=False).cfg,
            resolve=True,
        )}},
        "streaming": streaming_meta,
    }
    # Encoder cfg must stay pristine (no streaming ctor arg).
    enc_cfg = raw["cfg"]["model"]["cfg"]["encoder"]
    assert "streaming" not in enc_cfg, "encoder cfg must not carry streaming args"

    # --- loader side: instantiate + restore (mirrors load_model_from_hf) ---
    cfg = _remap_targets(omegaconf.OmegaConf.create(raw["cfg"]["model"]["cfg"]))
    model = gigaam.GigaAMASR(cfg).eval()
    assert not getattr(model.encoder, "streaming", False), "should load offline first"

    sm = raw.get("streaming")
    if sm and not getattr(model.encoder, "streaming", False):
        gigaam.enable_streaming(
            model,
            chunk_size=sm.get("chunk_size", 8),
            left_context_chunks=sm.get("left_context_chunks", -1),
        )
    assert model.encoder.streaming is True
    assert hasattr(model, "transcribe_streaming")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
