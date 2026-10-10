"""StreamingSession tests: live PCM push API == streaming model's full forward.

The tiny-model tests build a random GigaAMASR from an in-memory config (no
network / weights). Tests marked ``partial`` run the real v3 checkpoints.
"""

import omegaconf
import pytest
import torch

import gigaam
from gigaam.streaming import StreamingSession

VOCAB = list("абвгдежзийклмнопрстуфхцчшщъыьэюя ")  # 33 symbols, blank = 33
SR = 16000


def _tiny_model(mode: str = "ctc", right_context_chunks: int = 0, chunk_size: int = 8):
    d = 128
    cfg = {
        "model_name": f"tiny_{mode}",
        "preprocessor": {
            "_target_": "gigaam.preprocess.FeatureExtractor",
            "sample_rate": SR,
            "features": 64,
            "win_length": 320,
            "hop_length": 160,
            "n_fft": 320,
            "center": False,
        },
        "encoder": {
            "_target_": "gigaam.encoder.ConformerEncoder",
            "feat_in": 64,
            "n_layers": 2,
            "d_model": d,
            "subsampling": "conv1d",
            "subs_kernel_size": 5,
            "subsampling_factor": 4,
            "self_attention_model": "rotary",
            "n_heads": 4,
            "conv_kernel_size": 5,
            "conv_norm_type": "layer_norm",
            "flash_attn": False,
            "streaming": True,
            "chunk_size": chunk_size,
            "left_context_chunks": 2,
            "right_context_chunks": right_context_chunks,
        },
    }
    if mode == "ctc":
        cfg["head"] = {
            "_target_": "gigaam.decoder.CTCHead",
            "feat_in": d,
            "num_classes": len(VOCAB) + 1,
        }
        cfg["decoding"] = {
            "_target_": "gigaam.decoding.CTCGreedyDecoding",
            "vocabulary": VOCAB,
        }
    else:
        cfg["head"] = {
            "_target_": "gigaam.decoder.RNNTHead",
            "decoder": {
                "pred_hidden": 32,
                "pred_rnn_layers": 1,
                "num_classes": len(VOCAB) + 1,
            },
            "joint": {
                "enc_hidden": d,
                "pred_hidden": 32,
                "joint_hidden": 32,
                "num_classes": len(VOCAB) + 1,
            },
        }
        cfg["decoding"] = {
            "_target_": "gigaam.decoding.RNNTGreedyDecoding",
            "vocabulary": VOCAB,
        }
    torch.manual_seed(0)
    return gigaam.GigaAMASR(omegaconf.OmegaConf.create(cfg)).eval()


def _audio(seconds: float = 2.37, seed: int = 1):
    torch.manual_seed(seed)
    return 0.1 * torch.randn(int(seconds * SR))


def _reference(model, wav):
    """Streaming model's full forward + offline decode: (ids, frames, text)."""
    with torch.inference_mode():
        enc, enc_len = model.forward(wav.unsqueeze(0), torch.tensor([wav.numel()]))
        text, ids, frames = model.decoding.decode(model.head, enc, enc_len)[0]
    return ids, frames, text


def _run_session(model, wav, push, word_timestamps=False):
    session = StreamingSession(model, word_timestamps=word_timestamps)
    n_tokens = []
    for s in range(0, wav.numel(), push):
        session.push(wav[s : s + push])
        n_tokens.append(len(session._ids))
    session.finish()
    return session, n_tokens


@pytest.mark.parametrize("push", [16000 * 3, 1600, 333, 7])
def test_causal_features_match_offline(push):
    """Frames produced incrementally from arbitrary-sized PCM pieces must equal
    the offline feature matrix (center=False: frame i = samples [i*hop, i*hop+win))."""
    model = _tiny_model()
    wav = _audio()
    session = StreamingSession(model)
    session._drain = lambda final: None  # keep every frame buffered
    for s in range(0, wav.numel(), push):
        session.push(wav[s : s + push])
    session._featurize()
    with torch.inference_mode():
        ref, ref_len = model.preprocessor(wav.unsqueeze(0), torch.tensor([wav.numel()]))
    assert session._feats.shape == ref.shape, (session._feats.shape, ref.shape)
    assert session._feats.shape[-1] == ref_len.item()
    assert torch.allclose(session._feats, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["ctc", "rnnt"])
@pytest.mark.parametrize("right_context_chunks", [0, 1])
@pytest.mark.parametrize("push", [16000 * 3, 5120, 1000, 37])
def test_session_matches_full_forward(mode, right_context_chunks, push):
    """Pushing audio in any piece size yields the token ids and frames of the
    streaming model's full forward over the whole utterance, and partial results
    only ever grow."""
    model = _tiny_model(mode, right_context_chunks)
    wav = _audio()
    ref_ids, ref_frames, ref_text = _reference(model, wav)
    assert len(ref_ids) > 0, "random model should emit tokens"
    session, n_tokens = _run_session(model, wav, push)
    assert session._ids == ref_ids, f"{mode} rc={right_context_chunks} push={push}"
    assert session._frames == ref_frames
    assert session.text == ref_text
    assert n_tokens == sorted(n_tokens), "partial hypothesis shrank"
    assert session.finished


def test_word_timestamps():
    model = _tiny_model("ctc")
    wav = _audio(3.0)
    session, _ = _run_session(model, wav, 4000, word_timestamps=True)
    words = session.result().words
    assert words, "expected words"
    duration = wav.numel() / SR
    prev_end = 0.0
    for w in words:
        assert w.text
        assert 0.0 <= w.start <= w.end <= duration + session.frame_shift
        assert w.start >= prev_end - session.frame_shift  # non-overlapping, in order
        prev_end = w.end
    # offline timestamps of the same (streaming) model agree on word boundaries
    with torch.inference_mode():
        enc, enc_len = model.forward(wav.unsqueeze(0), torch.tensor([wav.numel()]))
    _, ref_words = model._decode(enc, enc_len, torch.tensor([wav.numel()]), True)[0]
    assert [w.text for w in words] == [w.text for w in ref_words]
    for a, b in zip(words, ref_words):
        assert abs(a.start - b.start) < 2 * session.frame_shift
        assert abs(a.end - b.end) < 2 * session.frame_shift


def test_latency_and_reset():
    model = _tiny_model("ctc", right_context_chunks=1)
    session = StreamingSession(model)
    assert abs(session.chunk_seconds - 0.32) < 1e-9
    assert abs(session.latency_seconds - 0.64) < 1e-9
    wav = _audio(1.0)
    session.push(wav)
    first = session.finish()
    assert session.finish() == first  # idempotent
    with pytest.raises(RuntimeError):
        session.push(wav)
    session.reset()
    assert session.text == "" and not session.finished
    session.push(wav)
    assert session.finish() == first  # a reused session reproduces the result


def test_rejects_bad_inputs():
    model = _tiny_model("ctc")
    session = StreamingSession(model)
    with pytest.raises(ValueError):
        session.push(_audio(0.5), sample_rate=8000)
    with pytest.raises(ValueError):
        session.push(torch.zeros(2, 100))  # stereo
    # int16 input is accepted and scaled
    session.push((_audio(0.5) * 32767).to(torch.int16))
    off = gigaam.GigaAMASR(model.cfg)  # same cfg, but patch streaming off
    off.encoder.streaming = False
    with pytest.raises(ValueError):
        StreamingSession(off)


def test_transcribe_streaming_uses_session(tmp_path):
    """The file helper is a thin wrapper over the session and must agree with it
    for every push size, including word timestamps."""
    import soundfile as sf

    model = _tiny_model("ctc")
    wav = _audio(2.0)
    path = tmp_path / "a.wav"
    sf.write(str(path), wav.numpy(), SR)
    wav_loaded = gigaam.load_audio(str(path))
    session, _ = _run_session(model, wav_loaded, 777, word_timestamps=True)
    for push_seconds in (0.02, 0.1, 5.0):
        res = model.transcribe_streaming(
            str(path), word_timestamps=True, push_seconds=push_seconds
        )
        assert res.text == session.text
        assert res.words == session.result().words


@pytest.mark.parametrize("name", ["v3_ctc", "v3_rnnt", "v3_e2e_ctc", "v3_e2e_rnnt"])
@pytest.mark.partial
def test_real_model_session_parity(name):
    """Real weights patched into streaming mode: the live session (pushed in
    small pieces) reproduces the streaming model's full forward on real speech,
    and the result does not depend on the push size.

    Note: a base model patched zero-shot into streaming mode emits little or
    nothing (its weights never saw causal convs / chunked attention), so this
    only checks agreement, not content; see test_streaming.py for the
    encoder-level parity on real weights. Quality needs a streaming fine-tune."""
    model = gigaam.load_model(name, device="cpu", fp16_encoder=False).eval()
    gigaam.enable_streaming(model)
    path = gigaam.utils.download_short_audio()
    wav = gigaam.load_audio(path)
    ref_ids, ref_frames, ref_text = _reference(model, wav)
    for push in (1600, 16000):
        session, _ = _run_session(model, wav, push, word_timestamps=True)
        assert session.text == ref_text, f"[{name}] push={push}: {session.text!r}"
        assert session._ids == ref_ids
        assert session._frames == ref_frames
    assert model.transcribe_streaming(path).text == ref_text


@pytest.mark.partial
def test_real_model_right_context_session_parity():
    """Real v3_ctc weights with hard-copied look-ahead: the deferred-emission
    session reproduces the hard-copy full forward at 16 layers."""
    model = gigaam.load_model("v3_ctc", device="cpu", fp16_encoder=False).eval()
    gigaam.enable_streaming(model, right_context_chunks=1)
    wav = gigaam.load_audio(gigaam.utils.download_short_audio())
    _, _, ref_text = _reference(model, wav)
    session, _ = _run_session(model, wav, 4000)
    assert session.text == ref_text


@pytest.mark.partial
def test_streaming_finetune_round_trip(tmp_path):
    """Two optimizer steps of a streaming fine-tune (hard-copy look-ahead +
    offline-teacher distillation) through train.py on CPU; the checkpoint must
    reload with its streaming settings restored and stream with session parity.
    This is the train -> checkpoint -> live-session path a production model
    takes."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import soundfile as sf

    pytest.importorskip("lightning")
    train_utils = Path(__file__).resolve().parents[1] / "train_utils"
    base = gigaam.load_model("v3_ctc", device="cpu", fp16_encoder=False).eval()
    audio_path = gigaam.utils.download_short_audio()
    text = base.transcribe(audio_path).text
    del base
    manifest = tmp_path / "manifest.tsv"
    duration = sf.info(audio_path).duration
    manifest.write_text(
        "path\tduration\ttranscription\n"
        f"{os.path.abspath(audio_path)}\t{duration:.3f}\t{text}\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "artifacts"
    cmd = [
        sys.executable,
        "train.py",
        "--model_name",
        "v3_ctc",
        "--train_manifest",
        str(manifest),
        "--val_manifest",
        str(manifest),
        "--output_dir",
        str(out_dir),
        "--exp_name",
        "pytest_streaming",
        "--batch_size",
        "1",
        "--eval_batch_size",
        "1",
        "--num_workers",
        "0",
        "--precision",
        "32",
        "--accelerator",
        "cpu",
        "--max_steps",
        "2",
        "--val_check_steps",
        "2",
        "--disable_tqdm",
        "--log_every_n_steps",
        "1",
        "--skip_initial_validation",
        "--save_top_k",
        "1",
        "--lr",
        "1e-5",
        "--streaming",
        "--chunk_size",
        "8",
        "--left_context_chunks",
        "4",
        "--right_context_chunks",
        "1",
        "--kd_alpha",
        "0.5",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    run = subprocess.run(cmd, cwd=train_utils, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    ckpts = sorted((out_dir / "models" / "pytest_streaming").glob("*.ckpt"))
    assert ckpts, "no checkpoint written"

    model = gigaam.load_model(str(ckpts[0]), device="cpu", fp16_encoder=False).eval()
    enc = model.encoder
    assert enc.streaming and enc.chunk_size == 8
    assert enc.left_context_chunks == 4 and enc.right_context_chunks == 1
    wav = gigaam.load_audio(audio_path)
    ref_ids, ref_frames, ref_text = _reference(model, wav)
    session, _ = _run_session(model, wav, 4000, word_timestamps=True)
    assert session._ids == ref_ids and session._frames == ref_frames
    assert session.text == ref_text
    assert abs(session.latency_seconds - 0.64) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
