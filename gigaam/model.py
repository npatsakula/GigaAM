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
    model: "GigaAM", chunk_size: int = 8, left_context_chunks: int = -1
) -> "GigaAM":
    """
    Rebuild ``model.encoder`` in streaming mode (chunked block attention + causal
    subsampling/convs) in place, preserving weights. Streaming adds no parameters,
    so the existing encoder state_dict loads strict. ``chunk_size`` is in encoder
    frames (40 ms each; 8 = 320 ms latency); ``left_context_chunks=-1`` means
    unlimited past. Also records the config on ``model.cfg.encoder`` so it round-
    trips through checkpoints.
    """
    cfg = model.cfg
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.encoder.streaming = True
    cfg.encoder.chunk_size = chunk_size
    cfg.encoder.left_context_chunks = left_context_chunks
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
    def transcribe_streaming(self, wav_file: str) -> TranscriptionResult:
        """
        Cache-aware streaming transcription (CTC models only). Encodes the audio
        chunk-by-chunk through the streaming encoder -- reproducing the offline
        result within numerical tolerance -- and collapses CTC output incrementally
        across chunk boundaries. Requires a model loaded/fine-tuned in streaming
        mode (``encoder.streaming``); see ``gigaam.enable_streaming``.

        This reference driver preprocesses the whole file, then feeds fixed
        feature blocks to emulate real-time arrival; a live driver would push
        audio as it comes and keep a one-hop overlap for the ``center=False``
        window. Output is identical either way.
        """
        from .decoding import CTCGreedyDecoding

        assert getattr(self.encoder, "streaming", False), (
            "model is not in streaming mode; load a streaming fine-tune or call "
            "gigaam.enable_streaming(model)"
        )
        assert isinstance(self.decoding, CTCGreedyDecoding), (
            "transcribe_streaming supports CTC models only"
        )

        wav, length = self.prepare_wav(wav_file)
        features, _ = self.preprocessor(wav.float(), length)  # [1, feat, T_feat]
        enc_dtype = next(self.encoder.parameters()).dtype
        head_dtype = next(self.head.parameters()).dtype
        features = features.to(enc_dtype)

        sub_factor = 2 ** self.encoder.pre_encode._sampling_num
        feat_per_chunk = self.encoder.chunk_size * sub_factor

        state = self.encoder.init_streaming_state()
        t_feat = features.shape[-1]
        ids: List[int] = []
        frames: List[int] = []
        last_label = self.decoding.blank_id
        enc_frame = 0
        for s in range(0, t_feat, feat_per_chunk):
            fc = features[:, :, s : s + feat_per_chunk]
            enc_chunk = self.encoder.forward_streaming_chunk(fc, state)
            log_probs = self.head(encoder_output=enc_chunk.to(head_dtype))  # [1, n, C]
            new_ids, new_fr, last_label = self.decoding.decode_chunk(
                log_probs, last_label, enc_frame
            )
            ids.extend(new_ids)
            frames.extend(new_fr)
            enc_frame += enc_chunk.shape[-1]

        text = self.decoding.tokenizer.decode(ids)
        return TranscriptionResult(text=text, words=None)

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
