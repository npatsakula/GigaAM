from typing import Dict, List, Optional, Tuple

import hydra
import omegaconf
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .preprocess import SAMPLE_RATE, load_audio
from .types import LongformTranscriptionResult, Segment, TranscriptionResult, Word
from .utils import AudioDataset, onnx_converter

LONGFORM_THRESHOLD = 25 * SAMPLE_RATE


def enable_streaming(
    model: "GigaAM",
    chunk_size: int = 8,
    left_context_chunks: int = 16,
    right_context_chunks: int = 0,
) -> "GigaAM":
    """
    Rebuild ``model.encoder`` in streaming mode (chunked block attention + causal
    subsampling/convs) in place, preserving weights. Streaming adds no parameters,
    so the existing encoder state_dict loads strict. Records the config on
    ``model.cfg.encoder`` so it round-trips through checkpoints.

    ``chunk_size`` is in encoder frames (40 ms each; 8 = 320 ms of audio per
    chunk). ``left_context_chunks`` bounds the attention window into the past
    (16 chunks = 5.1 s at chunk 8); it bounds the streaming cache and per-chunk
    cost and keeps rotary distances inside the trained range. ``-1`` means
    unlimited past: cost and memory then grow with stream length, and positions
    extrapolate beyond training lengths on long streams -- avoid it for live
    use. ``right_context_chunks=1`` adds one chunk of hard-copied look-ahead
    (Emformer style): latency doubles to two chunks, and the training forward
    is exactly the chunked inference computation (no depth compounding).

    A base (offline) model patched this way streams correctly but with degraded
    accuracy, because its weights never saw causal convs or chunked attention.
    Fine-tune in streaming mode (``train_utils/train.py --streaming``) for
    production quality; the fine-tune's checkpoint restores these settings on
    load.

    Order matters with FP8 quantization: this rebuilds the encoder fresh from
    ``cfg.encoder`` (plain ``nn.Linear``), so it must run *before*
    ``gigaam.fp8.quantize_model``. Calling it on an already-quantized encoder
    would silently rebuild plain Linears and discard every ``FP8Linear`` (its
    ``act_scale`` buffer is ``None`` while uncalibrated, hence absent from the
    state_dict, so the strict load succeeds while wiping the quantization).
    """
    # Guard: refuse to wipe an already-FP8-quantized encoder. Lazy import keeps
    # fp8 an optional dependency.
    try:
        from .fp8.linear import FP8Linear
    except Exception:  # fp8 extras not installed -> nothing to guard against
        FP8Linear = None  # type: ignore[assignment,misc]
    if FP8Linear is not None:
        has_fp8 = any(
            isinstance(m, FP8Linear) for _, m in model.encoder.named_modules()
        )
        if has_fp8:
            raise RuntimeError(
                "enable_streaming must be called BEFORE gigaam.fp8.quantize_model. "
                "The encoder is already FP8-quantized; rebuilding it here would "
                "silently replace every FP8Linear with a plain nn.Linear (the "
                "uncalibrated act_scale buffer is omitted from the state_dict, so "
                "the strict load succeeds while discarding the quantization). "
                "Re-enable streaming on the unquantized base, then quantize."
            )

    cfg = model.cfg
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.encoder.streaming = True
    cfg.encoder.chunk_size = chunk_size
    cfg.encoder.left_context_chunks = left_context_chunks
    cfg.encoder.right_context_chunks = right_context_chunks
    device = next(model.encoder.parameters()).device
    dtype = next(model.encoder.parameters()).dtype
    old_sd = model.encoder.state_dict()
    keep_ac = getattr(model.encoder, "activation_checkpointing", False)
    new_enc = hydra.utils.instantiate(cfg.encoder)
    new_enc.load_state_dict(old_sd, strict=True)
    new_enc.activation_checkpointing = keep_ac
    model.encoder = new_enc.to(device=device, dtype=dtype)
    return model


class GigaAM(nn.Module):
    """
    Giga Acoustic Model: Self-Supervised Model for Speech Tasks
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        self.preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
        self.encoder = hydra.utils.instantiate(self.cfg.encoder)

    def forward(
        self, features: Tensor, feature_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform forward pass through the preprocessor and encoder.
        """
        features, feature_lengths = self.preprocessor(features, feature_lengths)
        if self._device.type == "cpu":
            return self.encoder(features, feature_lengths)
        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            return self.encoder(features, feature_lengths)

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def prepare_wav(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Prepare an audio file for processing by loading it onto
        the correct device and converting its format.
        """
        wav = load_audio(wav_file)
        wav = wav.to(self._device).to(self._dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self._device)
        return wav, length

    def embed_audio(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Extract audio representations using the GigaAM model.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, encoded_len = self.forward(wav, length)
        return encoded, encoded_len

    def to_onnx(self, dir_path: str = ".", dtype: torch.dtype = torch.float32) -> None:
        """
        Export onnx model encoder to the specified dir.
        """
        with self.encoder.onnx_export_mode():
            self._to_onnx(dir_path, dtype=dtype)
        omegaconf.OmegaConf.save(self.cfg, f"{dir_path}/{self.cfg.model_name}.yaml")

    def _to_onnx(self, dir_path: str = ".", dtype: torch.dtype = torch.float32) -> None:
        """
        Export onnx model encoder to the specified dir.
        """
        onnx_converter(
            model_name=f"{self.cfg.model_name}_encoder",
            out_dir=dir_path,
            module=self.encoder,
            dynamic_axes=self.encoder.dynamic_axes(),
            export_dtype=dtype,
        )


class GigaAMASR(GigaAM):
    """
    Giga Acoustic Model for Speech Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.decoding = hydra.utils.instantiate(self.cfg.decoding)

    def _decode(
        self,
        encoded: Tensor,
        encoded_len: Tensor,
        wav_lens: Tensor,
        word_timestamps: bool = False,
    ) -> List[Tuple[str, Optional[List[Word]]]]:
        decoded = self.decoding.decode(self.head, encoded, encoded_len)
        if not word_timestamps:
            return [(t, None) for t, _, _ in decoded]
        from .timestamps_utils import compute_frame_shift, frames_to_words

        out: List[Tuple[str, Optional[List[Word]]]] = []
        for i, (text, token_ids, token_frames) in enumerate(decoded):
            frame_shift = compute_frame_shift(
                int(wav_lens[i].item()), int(encoded_len[i].item())
            )
            out.append(
                (
                    text,
                    frames_to_words(
                        self.decoding.tokenizer,
                        token_ids,
                        token_frames,
                        frame_shift,
                    ),
                )
            )
        return out

    @torch.inference_mode()
    def transcribe(
        self, wav_file: str, word_timestamps: bool = False
    ) -> TranscriptionResult:
        """
        Transcribes a short audio file into text.
        Returns TranscriptionResult with optional word-level timestamps.
        """
        wav, length = self.prepare_wav(wav_file)
        if length.item() > LONGFORM_THRESHOLD:
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")

        encoded, encoded_len = self.forward(wav, length)
        text, words = self._decode(encoded, encoded_len, length, word_timestamps)[0]
        return TranscriptionResult(text=text, words=words)

    @torch.inference_mode()
    def transcribe_streaming(
        self,
        wav_file: str,
        word_timestamps: bool = False,
        push_seconds: float = 0.1,
        head_padding_seconds: float = 0.0,
        tail_padding_seconds: float = 0.0,
        padding_noise: float = 0.0,
    ) -> TranscriptionResult:
        """
        Transcribe a file through the live streaming path: the audio is pushed
        into a :class:`gigaam.streaming.StreamingSession` in ``push_seconds``
        pieces, exactly as a microphone or socket would deliver it, and the
        session's final result is returned. Requires a model in streaming mode
        (a streaming fine-tune, or ``gigaam.enable_streaming``). Supports CTC and
        RNN-T models. The result equals the streaming model's full forward over
        the whole file, whatever ``push_seconds`` is.
        """
        from .streaming import StreamingSession

        session = StreamingSession(
            self,
            word_timestamps=word_timestamps,
            head_padding_seconds=head_padding_seconds,
            tail_padding_seconds=tail_padding_seconds,
            padding_noise=padding_noise,
        )
        wav = load_audio(wav_file)
        step = max(1, int(push_seconds * SAMPLE_RATE))
        for s in range(0, wav.numel(), step):
            session.push(wav[s : s + step])
        return session.finish()

    @torch.inference_mode()
    def transcribe_streaming_batch(
        self,
        wav_files: List[str],
        word_timestamps: bool = False,
        push_seconds: float = 0.1,
        max_batch_size: int = 64,
        head_padding_seconds: float = 0.0,
        tail_padding_seconds: float = 0.0,
        padding_noise: float = 0.0,
    ) -> List[TranscriptionResult]:
        """
        Transcribe several files through the live streaming path at once: every
        file is a :class:`gigaam.streaming.StreamingSession` in one
        :class:`gigaam.streaming.StreamingBatch`; audio is pushed in
        ``push_seconds`` pieces in lockstep and the batch steps after every
        round, so each chunk round is one batched encoder call. Each result
        equals :meth:`transcribe_streaming` on that file alone.
        """
        from .streaming import StreamingBatch

        batch = StreamingBatch(
            self,
            word_timestamps=word_timestamps,
            max_batch_size=max_batch_size,
            head_padding_seconds=head_padding_seconds,
            tail_padding_seconds=tail_padding_seconds,
            padding_noise=padding_noise,
        )
        wavs = [load_audio(f) for f in wav_files]
        sessions = [batch.new_session() for _ in wavs]
        step = max(1, int(push_seconds * SAMPLE_RATE))
        longest = max((w.numel() for w in wavs), default=0)
        for s0 in range(0, longest, step):
            for wav, session in zip(wavs, sessions):
                if s0 < wav.numel():
                    session.push(wav[s0 : s0 + step])
            batch.step()
            for wav, session in zip(wavs, sessions):
                if s0 + step >= wav.numel() and not session.finished:
                    session.finish()
        return [s.finish() for s in sessions]

    def forward_for_export(
        self, features: Tensor, feature_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        encoded, encoded_len = self.encoder(features, feature_lengths)
        return self.head(encoded), encoded_len

    def _to_onnx(self, dir_path: str = ".", dtype: torch.dtype = torch.float32) -> None:
        """
        Export onnx ASR model.
        `ctc`:  exported entirely in encoder-decoder format.
        `rnnt`: exported in encoder/decoder/joint parts separately.
        """
        if "ctc" in self.cfg.model_name:
            saved_forward = self.forward
            self.forward = self.forward_for_export  # type: ignore[assignment, method-assign]
            try:
                onnx_converter(
                    model_name=self.cfg.model_name,
                    out_dir=dir_path,
                    module=self,
                    inputs=self.encoder.input_example(),
                    input_names=["features", "feature_lengths"],
                    output_names=["log_probs", "encoded_lengths"],
                    dynamic_axes={
                        "features": {0: "batch_size", 2: "seq_len"},
                        "feature_lengths": {0: "batch_size"},
                        "log_probs": {0: "batch_size", 1: "seq_len"},
                        "encoded_lengths": {0: "batch_size"},
                    },
                    export_dtype=dtype,
                )
            finally:
                self.forward = saved_forward  # type: ignore[assignment, method-assign]
        else:
            super()._to_onnx(dir_path, dtype=dtype)
            onnx_converter(
                model_name=f"{self.cfg.model_name}_decoder",
                out_dir=dir_path,
                module=self.head.decoder,
                dynamic_axes=self.head.decoder.dynamic_axes(),
                export_dtype=dtype,
            )
            onnx_converter(
                model_name=f"{self.cfg.model_name}_joint",
                out_dir=dir_path,
                module=self.head.joint,
                dynamic_axes=self.head.joint.dynamic_axes(),
                export_dtype=dtype,
            )

    @torch.inference_mode()
    def transcribe_longform(
        self,
        wav_file: str,
        word_timestamps: bool = False,
        fr_batch_size: int = 16,
        fr_num_workers: int = 0,
        **kwargs,
    ) -> LongformTranscriptionResult:
        """
        Transcribes a long audio file by splitting it into segments and
        then transcribing each segment (batched inference via AudioDataset).
        Use fr_batch_size and fr_num_workers to control the batched inference.
        Returns LongformTranscriptionResult with segments containing optional word-level timestamps.
        """
        from .vad_utils import segment_audio_file

        segments, boundaries = segment_audio_file(
            wav_file, SAMPLE_RATE, device=self._device, **kwargs
        )

        if not segments:
            return LongformTranscriptionResult(segments=[])

        ds = AudioDataset(segments, tokenizer=None)
        dl = DataLoader(
            ds,
            batch_size=fr_batch_size,
            shuffle=False,
            collate_fn=AudioDataset.collate,
            num_workers=fr_num_workers,
        )

        result_segments: List[Segment] = []
        idx = 0
        for wav_pad, wav_lens in dl:
            wav_pad = wav_pad.to(self._device).to(self._dtype)
            wav_lens = wav_lens.to(self._device)
            encoded, encoded_len = self.forward(wav_pad, wav_lens)
            for text, words in self._decode(
                encoded, encoded_len, wav_lens, word_timestamps
            ):
                seg_start, seg_end = boundaries[idx]
                idx += 1
                if word_timestamps:
                    result_segments.append(
                        Segment(
                            text=text,
                            start=seg_start,
                            end=seg_end,
                            words=[
                                Word(
                                    text=w.text,
                                    start=round(w.start + seg_start, 3),
                                    end=round(w.end + seg_start, 3),
                                )
                                for w in words or []
                            ],
                        )
                    )
                else:
                    result_segments.append(
                        Segment(text=text, start=seg_start, end=seg_end)
                    )
        return LongformTranscriptionResult(segments=result_segments)


class GigaAMEmo(GigaAM):
    """
    Giga Acoustic Model for Emotion Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.id2name = cfg.id2name

    def get_probs(self, wav_file: str) -> Dict[str, float]:
        """
        Calculate probabilities for each emotion class based on the provided audio file.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, _ = self.forward(wav, length)
        encoded_pooled = nn.functional.avg_pool1d(
            encoded, kernel_size=encoded.shape[-1]
        ).squeeze(-1)

        logits = self.head(encoded_pooled)[0]
        probs = nn.functional.softmax(logits, dim=-1).detach().tolist()

        return {self.id2name[i]: probs[i] for i in range(len(self.id2name))}

    def forward_for_export(self, features: Tensor, feature_lengths: Tensor) -> Tensor:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        encoded, _ = self.encoder(features, feature_lengths)
        enc_pooled = encoded.mean(dim=-1)
        return nn.functional.softmax(self.head(enc_pooled), dim=-1)

    def _to_onnx(self, dir_path: str = ".", dtype: torch.dtype = torch.float32) -> None:
        """
        Export onnx Emo model.
        """
        saved_forward = self.forward
        self.forward = self.forward_for_export  # type: ignore[assignment, method-assign]
        try:
            onnx_converter(
                model_name=self.cfg.model_name,
                out_dir=dir_path,
                module=self,
                inputs=self.encoder.input_example(),
                input_names=["features", "feature_lengths"],
                output_names=["probs"],
                dynamic_axes={
                    "features": {0: "batch_size", 2: "seq_len"},
                    "feature_lengths": {0: "batch_size"},
                    "probs": {0: "batch_size", 1: "seq_len"},
                },
                export_dtype=dtype,
            )
        finally:
            self.forward = saved_forward  # type: ignore[assignment, method-assign]
