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



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
