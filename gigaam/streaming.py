"""
Live streaming ASR session.

:class:`StreamingSession` accepts raw 16 kHz PCM in arbitrary-sized pieces as it
arrives, featurizes it causally, drives the cache-aware streaming encoder one
chunk at a time, decodes incrementally (CTC or RNN-T) and returns the running
hypothesis after every push. The result is identical to running the streaming
model's full forward over the whole utterance, regardless of how the audio was
split across pushes.

Usage::

    model = gigaam.load_model("v3_ctc")          # or a streaming fine-tune
    gigaam.enable_streaming(model)               # no-op for a streaming ckpt
    session = gigaam.StreamingSession(model, word_timestamps=True)
    for pcm in microphone():                     # float32 / int16, 16 kHz, mono
        partial = session.push(pcm)
        print(partial.text)
    final = session.finish()

Latency: a token can be emitted once the encoder chunk containing it is
complete, i.e. ``chunk_size * 40 ms`` of audio after it, plus one more chunk
when the model uses right context (``right_context_chunks=1``). See
:attr:`StreamingSession.latency_seconds`.
"""

import contextlib
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .preprocess import SAMPLE_RATE
from .types import TranscriptionResult, Word

AudioLike = Union[Tensor, np.ndarray, List[float]]


class StreamingSession:
    """
    One utterance / one audio stream. Not thread-safe; create one session per
    stream. Batch size is 1. Call :meth:`reset` to reuse a session for a new
    stream.
    """

    def __init__(self, model, word_timestamps: bool = False):
        from .decoding import CTCGreedyDecoding, RNNTGreedyDecoding

        encoder = model.encoder
        if not getattr(encoder, "streaming", False):
            raise ValueError(
                "model is not in streaming mode; load a streaming fine-tune or call "
                "gigaam.enable_streaming(model) first"
            )
        self._is_ctc = isinstance(model.decoding, CTCGreedyDecoding)
        self._is_rnnt = isinstance(model.decoding, RNNTGreedyDecoding)
        if not (self._is_ctc or self._is_rnnt):
            raise ValueError("StreamingSession supports CTC and RNN-T models only")
        fe = model.preprocessor
        if getattr(fe, "center", False):
            raise ValueError(
                "streaming needs center=False log-mel features (the GigaAM v3 "
                "preprocessor); centered STFT frames read future samples"
            )

        self.model = model
        self.encoder = encoder
        self.head = model.head
        self.decoding = model.decoding
        self.word_timestamps = word_timestamps

        self.hop_length: int = fe.hop_length
        self.win_length: int = fe.win_length
        self.subsampling: int = 2**encoder.pre_encode._sampling_num
        self.chunk_size: int = encoder.chunk_size  # encoder frames per chunk
        self.feat_per_chunk: int = self.chunk_size * self.subsampling
        # Seconds per encoder frame (exact: hop * subsampling / sample rate).
        self.frame_shift: float = self.hop_length * self.subsampling / SAMPLE_RATE

        self._device = model._device
        self._enc_dtype = next(encoder.parameters()).dtype
        self._head_dtype = next(model.head.parameters()).dtype
        self._encoder_ctx = (
            torch.autocast(device_type=self._device.type, dtype=torch.float16)
            if self._device.type != "cpu"
            else contextlib.nullcontext()
        )
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Forget everything and start a new stream."""
        self._state = self.encoder.init_streaming_state()
        self._pcm = torch.zeros(0, dtype=torch.float32)  # unconsumed samples (CPU)
        self._feats: Optional[Tensor] = None  # featurized, not yet encoded [1,F,t]
        self._ids: List[int] = []
        self._frames: List[int] = []
        self._last_label: int = self.decoding.blank_id
        self._dec_state: Optional[Tuple[Tensor, Tensor]] = None  # RNN-T predictor
        self._enc_frames: int = 0  # encoder frames decoded so far
        self._finished = False
        self.samples_pushed: int = 0

    @property
    def chunk_seconds(self) -> float:
        """Audio duration of one encoder chunk."""
        return self.chunk_size * self.frame_shift

    @property
    def latency_seconds(self) -> float:
        """Algorithmic latency: audio that must arrive after a frame before the
        encoder can emit it (one chunk, plus one more with right context)."""
        return self.chunk_seconds * (1 + self.encoder.right_context_chunks)

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def text(self) -> str:
        """Current hypothesis text."""
        return self.decoding.tokenizer.decode(self._ids)

    @property
    def words(self) -> Optional[List[Word]]:
        """Current word-level timestamps (``None`` unless ``word_timestamps``).
        The last word may still be growing until :meth:`finish`."""
        if not self.word_timestamps:
            return None
        from .timestamps_utils import frames_to_words

        return frames_to_words(
            self.decoding.tokenizer, self._ids, self._frames, self.frame_shift
        )

    def result(self) -> TranscriptionResult:
        return TranscriptionResult(text=self.text, words=self.words)

    # ------------------------------------------------------------------ input
    @staticmethod
    def _to_pcm(audio: AudioLike) -> Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(np.ascontiguousarray(audio))
        elif not isinstance(audio, Tensor):
            audio = torch.tensor(audio)
        audio = audio.detach().cpu()
        if audio.dim() == 2 and audio.shape[0] == 1:
            audio = audio[0]
        if audio.dim() != 1:
            raise ValueError(
                f"expected mono audio [n] or [1, n], got {tuple(audio.shape)}"
            )
        if audio.dtype == torch.int16:
            return audio.float() / 32768.0
        if audio.dtype == torch.int32:
            return audio.float() / 2147483648.0
        return audio.float()

    @torch.inference_mode()
    def push(
        self, audio: AudioLike, sample_rate: int = SAMPLE_RATE
    ) -> TranscriptionResult:
        """
        Feed the next piece of audio (any length, including empty) and return
        the running hypothesis. ``audio`` is mono PCM: float in [-1, 1] or
        int16/int32, as a tensor, numpy array or list, at 16 kHz.
        """
        if self._finished:
            raise RuntimeError("session is finished; call reset() for a new stream")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"audio must be {SAMPLE_RATE} Hz (got {sample_rate}); resample before pushing"
            )
        pcm = self._to_pcm(audio)
        self.samples_pushed += pcm.numel()
        self._pcm = torch.cat([self._pcm, pcm]) if self._pcm.numel() else pcm
        self._featurize()
        self._drain(final=False)
        return self.result()

    @torch.inference_mode()
    def finish(self) -> TranscriptionResult:
        """End of stream: encode whatever is buffered (including a partial final
        chunk), flush any deferred look-ahead chunk and return the final result."""
        if not self._finished:
            self._featurize()
            self._drain(final=True)
            with self._encoder_ctx:
                tail = self.encoder.flush_streaming(self._state)
            self._consume_encoded(tail)
            self._finished = True
        return self.result()

    # --------------------------------------------------------------- pipeline
    def _featurize(self) -> None:
        """Turn every *complete* STFT window in the sample buffer into log-mel
        frames (frame i spans samples [i*hop, i*hop + win)), keeping the
        trailing ``win - hop`` samples for the next frame. Frames are computed
        exactly as the offline preprocessor computes them, so the stream of
        frames equals the offline feature matrix however the audio is split."""
        n = self._pcm.numel()
        if n < self.win_length:
            return
        n_frames = (n - self.win_length) // self.hop_length + 1
        use = (n_frames - 1) * self.hop_length + self.win_length
        seg = self._pcm[:use].to(self._device).unsqueeze(0)
        feats = self.model.preprocessor.featurizer(seg)  # [1, F, n_frames]
        assert feats.shape[-1] == n_frames, (feats.shape, n_frames)
        self._pcm = self._pcm[n_frames * self.hop_length :]
        self._feats = (
            feats if self._feats is None else torch.cat([self._feats, feats], dim=2)
        )

    def _drain(self, final: bool) -> None:
        """Encode every complete chunk of buffered frames; with ``final`` also
        the remaining partial chunk (the only chunk allowed to be partial)."""
        while self._feats is not None and self._feats.shape[-1] >= self.feat_per_chunk:
            block = self._feats[:, :, : self.feat_per_chunk]
            self._feats = self._feats[:, :, self.feat_per_chunk :]
            self._encode(block)
        if final and self._feats is not None and self._feats.shape[-1] > 0:
            block, self._feats = self._feats, None
            self._encode(block)

    def _encode(self, feat_block: Tensor) -> None:
        with self._encoder_ctx:
            enc = self.encoder.forward_streaming_chunk(
                feat_block.to(self._enc_dtype), self._state
            )
        self._consume_encoded(enc)

    def _consume_encoded(self, enc_chunk: Tensor) -> None:
        if enc_chunk.shape[-1] == 0:
            return
        enc_chunk = enc_chunk.to(self._head_dtype)
        if self._is_ctc:
            log_probs = self.head(encoder_output=enc_chunk)  # [1, n, C]
            ids, frames, self._last_label = self.decoding.decode_chunk(
                log_probs, self._last_label, self._enc_frames
            )
        else:
            ids, frames, self._dec_state, self._last_label = self.decoding.decode_chunk(
                self.head,
                enc_chunk,
                self._dec_state,
                self._last_label,
                self._enc_frames,
            )
        self._ids.extend(ids)
        self._frames.extend(frames)
        self._enc_frames += enc_chunk.shape[-1]
