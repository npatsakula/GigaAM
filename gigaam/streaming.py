"""
Live streaming ASR sessions.

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

:class:`StreamingBatch` serves many concurrent streams with one batched encoder
(and RNN-T decoder) call per chunk round. Sessions created through it only
buffer on ``push``; ``batch.step()`` encodes every full chunk of every session
together, and each session still finishes on its own::

    batch = gigaam.StreamingBatch(model)
    a, b = batch.new_session(), batch.new_session()
    a.push(pcm_a); b.push(pcm_b)
    batch.step()                                 # one encoder call for both
    print(a.text, b.text)
    a.finish()                                   # drains a's tail, leaves the batch

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
    stream. Call :meth:`reset` to reuse a session for a new stream. A session
    created by :class:`StreamingBatch` encodes only when the batch steps.
    """

    def __init__(
        self,
        model,
        word_timestamps: bool = False,
        batch: Optional["StreamingBatch"] = None,
    ):
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
        self._batch = batch

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
        if self._batch is not None:
            self._batch._attach(self)

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
        int16/int32, as a tensor, numpy array or list, at 16 kHz. In a
        :class:`StreamingBatch` the audio is only buffered and featurized here;
        the hypothesis advances when the batch steps.
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
        if self._batch is None:
            self._drain(final=False)
        return self.result()

    @torch.inference_mode()
    def finish(self) -> TranscriptionResult:
        """End of stream: encode whatever is buffered (including a partial final
        chunk), flush any deferred look-ahead chunk and return the final result.
        A batched session leaves its batch here."""
        if not self._finished:
            self._featurize()
            self._drain(final=True)
            with self._encoder_ctx:
                tail = self.encoder.flush_streaming(self._state)
            self._consume_encoded(tail)
            self._finished = True
            if self._batch is not None:
                self._batch._detach(self)
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

    def _chunk_ready(self) -> bool:
        return (
            not self._finished
            and self._feats is not None
            and self._feats.shape[-1] >= self.feat_per_chunk
        )

    def _take_chunk(self) -> Tensor:
        """Pop one full chunk of buffered frames ``[1, F, feat_per_chunk]``."""
        assert self._feats is not None
        block = self._feats[:, :, : self.feat_per_chunk]
        self._feats = self._feats[:, :, self.feat_per_chunk :]
        return block

    def _drain(self, final: bool) -> None:
        """Encode every complete chunk of buffered frames; with ``final`` also
        the remaining partial chunk (the only chunk allowed to be partial)."""
        while self._chunk_ready():
            self._encode(self._take_chunk())
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
            self._consume_log_probs(log_probs)
        else:
            ids, frames, self._dec_state, self._last_label = self.decoding.decode_chunk(
                self.head,
                enc_chunk,
                self._dec_state,
                self._last_label,
                self._enc_frames,
            )
            self._append(ids, frames, enc_chunk.shape[-1])

    def _consume_log_probs(self, log_probs: Tensor) -> None:
        """CTC: collapse one chunk of ``[1, n, C]`` log-probs into tokens."""
        ids, frames, self._last_label = self.decoding.decode_chunk(
            log_probs, self._last_label, self._enc_frames
        )
        self._append(ids, frames, log_probs.shape[1])

    def _append(self, ids: List[int], frames: List[int], n_frames: int) -> None:
        self._ids.extend(ids)
        self._frames.extend(frames)
        self._enc_frames += n_frames


class StreamingBatch:
    """
    Serve many independent streams with batched compute. Create sessions with
    :meth:`new_session`; their ``push`` only buffers audio. :meth:`step` then
    encodes every session that has a full chunk ready in one batched encoder
    call (and one batched RNN-T decoder call), repeating until no session has a
    full chunk left. Sessions may join at any time and leave via ``finish()``,
    which drains that session's partial tail on its own. Results are identical
    to running each session alone.

    With ``right_context_chunks=1`` a round is split into two batched calls:
    sessions still waiting for their first look-ahead chunk, and the rest.
    ``max_batch_size`` caps the number of streams per encoder call.
    """

    def __init__(self, model, word_timestamps: bool = False, max_batch_size: int = 64):
        self.model = model
        self.word_timestamps = word_timestamps
        self.max_batch_size = max(1, int(max_batch_size))
        self._sessions: List[StreamingSession] = []

    @property
    def sessions(self) -> List[StreamingSession]:
        """Active (unfinished) sessions, in creation order."""
        return list(self._sessions)

    def new_session(self, word_timestamps: Optional[bool] = None) -> StreamingSession:
        wt = self.word_timestamps if word_timestamps is None else word_timestamps
        return StreamingSession(self.model, word_timestamps=wt, batch=self)

    def _attach(self, session: StreamingSession) -> None:
        if session not in self._sessions:
            self._sessions.append(session)

    def _detach(self, session: StreamingSession) -> None:
        if session in self._sessions:
            self._sessions.remove(session)

    @torch.inference_mode()
    def step(self) -> int:
        """Encode and decode every full chunk buffered in every active session,
        batched. Returns the number of batched encoder calls made."""
        calls = 0
        while True:
            ready = [s for s in self._sessions if s._chunk_ready()]
            if not ready:
                return calls
            for group in self._group(ready):
                for i in range(0, len(group), self.max_batch_size):
                    self._step_group(group[i : i + self.max_batch_size])
                    calls += 1

    def _group(self, ready: List[StreamingSession]) -> List[List[StreamingSession]]:
        enc = self.model.encoder
        if enc.right_context_chunks == 0:
            return [ready]
        # Look-ahead: streams without a pending chunk only buffer this round;
        # streams with one all hold a full chunk (a partial pending chunk can
        # only arise in finish(), which runs on its own), so each group is
        # homogeneous as forward_streaming_chunk_batched requires.
        fresh = [s for s in ready if s._state.pending_feat is None]
        pending = [s for s in ready if s._state.pending_feat is not None]
        return [g for g in (fresh, pending) if g]

    def _step_group(self, group: List[StreamingSession]) -> None:
        ref = group[0]
        blocks = torch.cat([s._take_chunk() for s in group], dim=0)
        with ref._encoder_ctx:
            enc = self.model.encoder.forward_streaming_chunk_batched(
                blocks.to(ref._enc_dtype), [s._state for s in group]
            )
        if enc.shape[-1] == 0:
            return
        enc = enc.to(ref._head_dtype)
        n = enc.shape[-1]
        if ref._is_ctc:
            log_probs = self.model.head(encoder_output=enc)  # [B, n, C]
            for b, s in enumerate(group):
                s._consume_log_probs(log_probs[b : b + 1])
        else:
            ids, frames, states, labels = self.model.decoding.decode_chunk_batched(
                self.model.head,
                enc,
                [s._dec_state for s in group],
                [s._last_label for s in group],
                [s._enc_frames for s in group],
            )
            for b, s in enumerate(group):
                s._dec_state, s._last_label = states[b], labels[b]
                s._append(ids[b], frames[b], n)
