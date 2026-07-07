import copy
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import editdistance
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torchaudio.functional import rnnt_loss as _ta_rnnt_loss
from torchaudio.transforms import FrequencyMasking, TimeMasking

from gigaam.decoder import CTCHead, RNNTHead
from gigaam.model import GigaAMASR


class GigaAMFineTuner(pl.LightningModule):

    def __init__(
        self,
        model: GigaAMASR,
        blank_id: int,
        lr: float = 1e-4,
        freeze_encoder: bool = False,
        rnnt_subbatch_size: int = 0,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        log_every_n_steps: int = 10,
        spec_augment: bool = False,
        freq_masks: int = 2,
        freq_width: int = 27,
        time_masks: int = 2,
        time_width: int = 20,
        fp8: bool = False,
        kd_alpha: float = 0.5,
        calibration_scales: Optional[str] = None,
        cli_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "cli_args"])
        if cli_args is not None:
            self.save_hyperparameters(cli_args)
        self.preprocessor = model.preprocessor
        self.encoder = model.encoder
        self.head = model.head
        self._decoding = model.decoding
        self._tokenizer = model.decoding.tokenizer
        self._blank_id = blank_id
        self._rnnt_subbatch_size = rnnt_subbatch_size
        self._freeze_encoder = freeze_encoder

        self._spec_augment = spec_augment
        if spec_augment:
            self._freq_aug = nn.ModuleList(
                [FrequencyMasking(freq_width) for _ in range(freq_masks)]
            )
            self._time_aug = nn.ModuleList(
                [TimeMasking(time_width) for _ in range(time_masks)]
            )

        # Head / loss selection (must run before the FP8 block, which checks mode).
        self._rnnt_vocab_size = 0
        if isinstance(self.head, CTCHead):
            self.mode = "ctc"
            self._ctc = nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=True)
        elif isinstance(self.head, RNNTHead):
            self.mode = "rnnt"
            self._rnnt_vocab_size = int(model.cfg.head.joint.num_classes)
            assert (
                blank_id + 1 == self._rnnt_vocab_size
            ), f"blank_id={blank_id} != joint V={self._rnnt_vocab_size}"
        else:
            raise ValueError(f"Unsupported head: {type(self.head)}")

        self._lr, self._wd = lr, weight_decay
        self._warmup_ratio = warmup_ratio
        self._log_every, self._last_log_step = log_every_n_steps, -1

        for p in self.preprocessor.parameters():
            p.requires_grad = False
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # --- FP8 quantization-aware training + logit knowledge distillation ---
        # The student has its GEMM Linears swapped for FP8Linear (E4M3 simulated,
        # STE). A frozen copy of the *original* (pre-quantization) model acts as
        # the teacher for logit KD:
        #   - CTC:  teacher = fp16 encoder copy. The CTC head (1x1 Conv1d) is NOT
        #           quantized, so teacher and student share self.head.
        #   - RNNT: teacher = fp16 encoder copy + fp32 head copy. The RNNT joint
        #           enc/pred Linears ARE quantized in the student and the decoder
        #           LSTM drifts during training, so the teacher needs its own
        #           full-precision joint + decoder.
        # Both teachers are registered as submodules so Lightning moves them to
        # the accelerator device; all their params are frozen so they are never
        # optimised. The teacher copies are taken BEFORE quantizing the student.
        self._fp8 = fp8
        self._kd_alpha = kd_alpha
        if fp8:
            from gigaam.fp8 import bake_activation_scales
            from gigaam.fp8.quantize import quantize_encoder, quantize_model

            # Frozen fp16 teacher encoder (used by both CTC and RNNT KD).
            teacher = copy.deepcopy(self.encoder).half()
            for p in teacher.parameters():
                p.requires_grad = False
            self.teacher_encoder = teacher

            if self.mode == "ctc":
                # CTC head is not quantized -> shared between teacher and student.
                self.teacher_head = None
                # compute_dtype auto-detects each Linear's native dtype (fp32 here,
                # since train.py loads fp16_encoder=False). Forcing fp16 would make
                # the quantized layers emit fp16 into the otherwise-fp32 encoder,
                # crashing on the first un-quantized conv/matmul for conv2d/rel_pos
                # models; auto-detect keeps the whole encoder in one dtype.
                quantize_encoder(self)
            elif self.mode == "rnnt":
                # RNNT joint enc/pred are quantized in the student -> the teacher
                # needs its own full-precision head (decoder LSTM + joint). The
                # head stays fp32 (its native dtype under load_model_from_hf).
                teacher_head = copy.deepcopy(self.head)
                for p in teacher_head.parameters():
                    p.requires_grad = False
                self.teacher_head = teacher_head
                # quantize_model quantizes the encoder and the RNNT joint enc/pred,
                # each following its own module's native dtype (fp32 here).
                quantize_model(self)
            else:  # pragma: no cover - guarded above
                raise ValueError(f"FP8 QAT + KD does not support mode={self.mode!r}")

            # Bake activation scales from calibration if provided.
            if calibration_scales is not None:
                import json

                with open(calibration_scales) as f:
                    amax = json.load(f)
                n = bake_activation_scales(self, amax)
                print(f"  FP8: baked {n} activation scales from {calibration_scales}")
            head_note = (
                " + teacher head" if self.teacher_head is not None else ""
            )
            print(
                f"  FP8 QAT enabled: student quantized, teacher"
                f" encoder{head_note}, kd_alpha={kd_alpha}"
            )
        else:
            self.teacher_encoder = None
            self.teacher_head = None
            if calibration_scales is not None:
                warnings.warn(
                    "calibration_scales was provided but fp8=False; FP8 QAT and KD "
                    "are disabled, so the calibration file is ignored. Pass --fp8 to "
                    "enable quantization-aware training.",
                    stacklevel=2,
                )

        self._val_errors = self._val_words = 0

    def train(self, mode: bool = True) -> "GigaAMFineTuner":
        super().train(mode)
        # We keep it verbose for the freezed preprocessor
        self.preprocessor.eval()
        if self._freeze_encoder:
            self.encoder.eval()
        # The KD teachers are always frozen (eval), even when the student trains.
        if self.teacher_encoder is not None:
            self.teacher_encoder.eval()
        if self.teacher_head is not None:
            self.teacher_head.eval()
        return self

    def on_train_epoch_start(self):
        self.train()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # The KD teachers are frozen copies of the pretrained encoder/head,
        # deterministically reconstructed in __init__, so serialising them into
        # every checkpoint just doubles its size. Strip them on save; on_load
        # re-injects the freshly-built copies so a strict state_dict load still
        # sees no missing keys.
        sd = checkpoint.get("state_dict")
        if sd is not None:
            for k in list(sd):
                if k.startswith(("teacher_encoder.", "teacher_head.")):
                    del sd[k]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        sd = checkpoint.get("state_dict")
        if sd is not None:
            for k, v in self.state_dict().items():
                if k.startswith(("teacher_encoder.", "teacher_head.")) and k not in sd:
                    sd[k] = v

    def _ctc_loss(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lens: Tensor,
        target_lens: Tensor,
    ) -> Tensor:
        return self._ctc(
            log_probs.transpose(0, 1),
            targets.long(),
            input_lens.long(),
            target_lens.long(),
        ).mean()

    def _rnnt_loss(
        self, logits: Tensor, targets: Tensor, logit_lens: Tensor, target_lens: Tensor
    ) -> Tensor:
        return _ta_rnnt_loss(
            logits=logits.float(),
            targets=targets.int(),
            logit_lengths=logit_lens.int(),
            target_lengths=target_lens.int(),
            blank=self._blank_id,
            reduction="mean",
            fused_log_softmax=True,
        )

    def _ctc_kd_loss(
        self,
        features: Tensor,
        feat_lens: Tensor,
        student_log_probs: Tensor,
        enc_lens: Tensor,
    ) -> Tensor:
        """Per-frame CTC knowledge-distillation loss: ``KL(teacher || student)``.

        The frozen teacher encodes the same mel ``features`` and produces CTC
        log-probs through the (shared) head. We minimise the KL divergence of the
        student's per-frame distribution toward the teacher's soft distribution.
        Soft per-frame distributions carry "dark knowledge" about confusable phones,
        recovering most of the WER gap that PTQ FP8 costs.

        The KL is summed over valid frames and averaged over the batch, matching the
        scale of ``_ctc_loss`` (``nn.CTCLoss(reduction="none")`` gives a per-sequence
        *summed* NLL, then ``.mean()`` over the batch). Normalising by the number of
        frames instead — as the first version did — makes the KD term ~2 orders of
        magnitude smaller than the CE term, so ``kd_alpha`` has almost no effect.

        ``student_log_probs`` already includes a log_softmax (from CTCHead); the
        teacher path is recomputed here under no_grad.
        """
        assert self.teacher_encoder is not None, "KD requires a teacher encoder"
        dev = student_log_probs.device
        teacher = self.teacher_encoder
        t_dev = next(teacher.parameters()).device
        t_dtype = next(teacher.parameters()).dtype
        with torch.no_grad():
            t_features = features.to(t_dev).to(t_dtype)
            t_enc, _ = teacher(t_features, feat_lens.to(t_dev))
            teacher_log_probs = self.head(t_enc.float().to(dev))
        # Teacher and student share an identical architecture and the same input,
        # so their logit shapes must match frame-for-frame. If they ever don't the
        # per-frame KL below would silently broadcast/misalign, so guard it.
        assert teacher_log_probs.shape == student_log_probs.shape, (
            f"teacher/student logit shape mismatch: "
            f"{tuple(teacher_log_probs.shape)} vs {tuple(student_log_probs.shape)}"
        )
        # KL(teacher || student) per frame via F.kl_div(log_target=True): computes
        # exp(log_q) * (log_q - log_p) fused, without materialising exp() of the full
        # [B, T, V] teacher tensor. Gradient flows through the student only.
        B, T = student_log_probs.shape[0], student_log_probs.size(1)
        kl = torch.nn.functional.kl_div(
            student_log_probs, teacher_log_probs, reduction="none", log_target=True
        ).sum(dim=-1)  # [B, T]
        frame_mask = (
            torch.arange(T, device=dev)[None, :] < enc_lens.to(dev)[:, None]
        ).to(kl.dtype)
        return (kl * frame_mask).sum() / max(B, 1)

    def _encode(
        self, wavs: Tensor, wav_lens: Tensor, return_features: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        amp_dev = next(self.parameters()).device.type
        with torch.amp.autocast(amp_dev, enabled=False):
            features, feat_lens = self.preprocessor(wavs.float(), wav_lens)
        if self.training and self._spec_augment:
            for aug in self._freq_aug:
                features = aug(features)
            for aug in self._time_aug:
                features = aug(features)
        encoded, enc_lens = self.encoder(features, feat_lens)
        if return_features:
            return encoded, enc_lens, features, feat_lens
        return encoded, enc_lens

    def _rnnt_joint(
        self,
        encoded: Tensor,
        tokens: Tensor,
        head: "Optional[RNNTHead]" = None,
    ) -> Tensor:
        head = self.head if head is None else head
        assert isinstance(head, RNNTHead)
        enc_t = encoded.float().transpose(1, 2)
        dec, jnt = head.decoder, head.joint
        bos = torch.zeros(
            enc_t.size(0), 1, dec.pred_hidden, device=enc_t.device, dtype=enc_t.dtype
        )
        pred_out, _ = dec.lstm(
            torch.cat([bos, dec.embed(tokens)], dim=1).transpose(0, 1)
        )
        return jnt.joint_net(
            jnt.enc(enc_t).unsqueeze(2)
            + jnt.pred(pred_out.transpose(0, 1)).unsqueeze(1)
        )

    def _rnnt_grid_kl(
        self,
        teacher_logits: Tensor,
        student_logits: Tensor,
        enc_lens: Tensor,
        tok_lens: Tensor,
    ) -> Tensor:
        """RNNT action-based knowledge distillation: grid-level ``KL(teacher || student)``.

        Both inputs are the raw pre-softmax joint logits over the (T, U+1)
        transducer lattice, shape ``[b, T, U+1, V]`` (as produced by
        :meth:`_rnnt_joint`). We minimise the KL divergence of the student's
        per-cell posterior toward the teacher's, mirroring the per-frame CTC KD
        but over the 2-D alignment lattice instead of a single frame axis.

        ``teacher_logits`` must be detached (computed under ``no_grad``); gradients
        flow only through ``student_logits``. Cells are masked to valid encoder
        frames (``t < enc_lens``) and valid prediction positions (``u <= tok_lens``,
        i.e. columns ``0..U`` including the final blank). Temperature is fixed at
        1, matching the CTC KD.

        The masked KL is summed over valid lattice cells and averaged over the
        (sub-)batch so it shares the scale of ``_rnnt_loss`` (``reduction="mean"``
        over the batch of per-sequence transducer NLLs). Dividing by the number of
        cells instead made the KD term negligible next to the CE term.
        """
        dev = student_logits.device
        teacher_logits = teacher_logits.to(dev)
        # KL(teacher || student) = sum p_t * (log p_t - log p_s), computed via
        # F.kl_div with log_target=True so the teacher distribution stays in
        # log-space (numerically stable, no exp of the grid).
        log_p = torch.log_softmax(student_logits, dim=-1)  # carries grad
        with torch.no_grad():
            log_q = torch.log_softmax(teacher_logits, dim=-1)
        kl = torch.nn.functional.kl_div(
            log_p, log_q, reduction="none", log_target=True
        ).sum(dim=-1)  # [b, T, U+1]
        T, U1 = kl.shape[1], kl.shape[2]
        t_mask = torch.arange(T, device=dev)[None, :] < enc_lens.to(dev)[:, None]
        # Valid prediction positions are 0..U inclusive; the U-th column holds the
        # final blank emission (max_u1 = tok_lens.max() + 1).
        u_mask = torch.arange(U1, device=dev)[None, :] <= tok_lens.to(dev)[:, None]
        cell_mask = (t_mask.unsqueeze(2) & u_mask.unsqueeze(1)).to(kl.dtype)
        batch = student_logits.size(0)
        return (kl * cell_mask).sum() / max(batch, 1)

    def _rnnt_forward(
        self,
        encoded: Tensor,
        enc_lens: Tensor,
        tokens: Tensor,
        tok_lens: Tensor,
        kd_ctx: "Optional[Dict[str, Tensor]]" = None,
    ) -> Tensor:
        """RNNT loss, optionally blended with grid-level KD against the teacher.

        When ``kd_ctx`` (``{"features", "feat_lens"}``) is provided the frozen fp16
        teacher encoder is run once on the shared mel features, and within the
        existing sub-batch loop a teacher joint grid is produced and compared to
        the student grid via :meth:`_rnnt_grid_kl`. The per-sub-batch loss becomes
        ``alpha * CE + (1 - alpha) * KD``. The teacher grid is computed and freed
        inside the loop (never materialised for the whole batch) to stay within
        the same memory budget as pure CE.
        """
        B = encoded.size(0)
        subb = self._rnnt_subbatch_size or B
        use_kd = kd_ctx is not None and self.teacher_encoder is not None

        # Ensure int32-safe indexing in _ta_rnnt_loss
        max_t = int(enc_lens.max().item())
        max_u1 = int(tok_lens.max().item()) + 1
        vocab_size = self._rnnt_vocab_size
        while subb > 1 and subb * max_t * max_u1 * vocab_size >= (1 << 31):
            subb = max(1, subb // 2)
        # KD materialises two grids (teacher + student) plus log_softmax buffers per
        # sub-batch (~4x the CE footprint). Halve the bound so e2e_rnnt (V=1025)
        # stays within a 12 GB dev GPU; charwise rnnt (V=34) is ~30x lighter.
        if use_kd:
            while subb > 1 and subb * max_t * max_u1 * vocab_size >= (1 << 30):
                subb = max(1, subb // 2)

        # Teacher encoder: run once on shared mel features (cheap vs. the grid).
        t_encoded: "Optional[Tensor]" = None
        if use_kd:
            assert self.teacher_head is not None, "RNNT KD requires a teacher head"
            t_param = next(self.teacher_encoder.parameters())
            t_dev = t_param.device
            with torch.no_grad():
                t_features = kd_ctx["features"].to(t_dev).to(t_param.dtype)
                t_feat_lens = kd_ctx["feat_lens"].to(t_dev)
                t_encoded, _ = self.teacher_encoder(t_features, t_feat_lens)

        losses: List[torch.Tensor] = []
        kd_terms: List[torch.Tensor] = []
        weights: List[int] = []
        for i in range(0, B, subb):
            s = slice(i, min(i + subb, B))
            enc_i, enc_lens_i = encoded[s], enc_lens[s]
            tok_i, tok_lens_i = tokens[s], tok_lens[s]
            sub_t = int(enc_lens_i.max().item())
            enc_i = enc_i[:, :, :sub_t].contiguous()
            tok_i = tok_i[:, : int(tok_lens_i.max().item())].contiguous()
            logits_i = self._rnnt_joint(enc_i, tok_i)  # [b, T, U+1, V]
            T, U1 = logits_i.shape[1:3]
            enc_lens_i = enc_lens_i.clamp(min=1, max=T)
            tok_lens_i = tok_lens_i.clamp(min=1, max=U1 - 1)
            loss_i = self._rnnt_loss(logits_i, tok_i, enc_lens_i, tok_lens_i)

            if use_kd:
                # Teacher grid for this sub-batch (trimmed to the same T, freed
                # after the KL). Teacher output is [b, D, T]; match the trim.
                t_enc_i = t_encoded[s][:, :, :sub_t].float()  # type: ignore[index]
                with torch.no_grad():
                    t_logits_i = self._rnnt_joint(t_enc_i, tok_i, head=self.teacher_head)
                # Teacher is frozen and detached; move to student device in case
                # of a device split (they normally match).
                kd_i = self._rnnt_grid_kl(
                    t_logits_i.detach(), logits_i, enc_lens_i, tok_lens_i
                )
                loss_i = self._kd_alpha * loss_i + (1.0 - self._kd_alpha) * kd_i
                kd_terms.append(kd_i.detach() * enc_i.size(0))
                del t_logits_i, t_enc_i

            losses.append(loss_i * (enc_i.size(0)))
            weights.append(enc_i.size(0))
            del logits_i

        if kd_terms and self.training:
            self.log(
                "train/kd", torch.stack(kd_terms).sum() / sum(weights), prog_bar=True
            )
        return torch.stack(losses).sum() / sum(weights)

    def _batch_wer(
        self, hyps: List[str], tokens: Tensor, tok_lens: Tensor
    ) -> Tuple[int, int]:
        errors = words = 0
        for i, hyp in enumerate(hyps):
            ref = self._tokenizer.decode(tokens[i, : tok_lens[i]].tolist()).split()
            hyp_w = hyp.split()
            errors += editdistance.eval(ref, hyp_w)
            words += max(len(ref), 1)
        return errors, words

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        wavs, wav_lens, tokens, tok_lens = batch
        # Both CTC and RNNT KD need the shared mel features to feed the frozen
        # teacher encoder on the same input the student saw.
        need_feats = self._fp8 and self.teacher_encoder is not None
        if need_feats:
            encoded, enc_lens, features, feat_lens = self._encode(
                wavs, wav_lens, return_features=True
            )
        else:
            encoded, enc_lens = self._encode(wavs, wav_lens)

        if self.mode == "ctc":
            log_probs = self.head(encoded)
            loss = self._ctc_loss(log_probs, tokens, enc_lens, tok_lens)
            # FP8 knowledge distillation: add a KL term on per-frame CTC logits
            # against the frozen fp16 teacher's logits. Both heads see their own
            # encoder output so the soft distributions align frame-by-frame.
            if need_feats:
                kd = self._ctc_kd_loss(features, feat_lens, log_probs, enc_lens)
                loss = self._kd_alpha * loss + (1.0 - self._kd_alpha) * kd
                self.log("train/kd", kd, prog_bar=True)
        else:
            kd_ctx = (
                {"features": features, "feat_lens": feat_lens} if need_feats else None
            )
            loss = self._rnnt_forward(encoded, enc_lens, tokens, tok_lens, kd_ctx=kd_ctx)
        self.log("train/loss", loss, prog_bar=True)

        if (
            self.global_step % self._log_every == 0
            and self.global_step != self._last_log_step
        ):
            self._last_log_step = self.global_step
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr)
            with torch.no_grad():
                res = self._decoding.decode(
                    self.head, encoded.detach().float(), enc_lens
                )
                errs, wds = self._batch_wer([h[0] for h in res], tokens, tok_lens)
                train_wer = errs / max(wds, 1)
                self.log("train/wer", train_wer)
        return loss

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int):
        wavs, wav_lens, tokens, tok_lens = batch
        encoded, enc_lens = self._encode(wavs, wav_lens)

        if self.mode == "ctc":
            log_probs = self.head(encoded)
            loss = self._ctc_loss(log_probs, tokens, enc_lens, tok_lens)
        else:
            loss = self._rnnt_forward(encoded, enc_lens, tokens, tok_lens)

        res = self._decoding.decode(self.head, encoded, enc_lens)
        errs, wds = self._batch_wer([h[0] for h in res], tokens, tok_lens)
        self._val_errors += errs
        self._val_words += wds
        self.log("val/loss", loss, sync_dist=True)

    def on_validation_epoch_end(self):
        errs_t = torch.tensor(self._val_errors, device=self.device, dtype=torch.float64)
        wds_t = torch.tensor(self._val_words, device=self.device, dtype=torch.float64)
        if self.trainer.world_size > 1 and dist.is_initialized():
            dist.all_reduce(errs_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(wds_t, op=dist.ReduceOp.SUM)
        self._val_errors = self._val_words = 0
        if wds_t.item() <= 0:
            return
        wer = (errs_t / wds_t).item()
        # All ranks must log val_wer so ModelCheckpoint sees monitor on every process.
        self.log("val/wer", wer, sync_dist=False)
        self.log("val_wer", wer, logger=False, sync_dist=False)
        if self.trainer.is_global_zero:
            print(
                f"  [val] step={self.global_step} epoch={self.current_epoch}  "
                f"val/loss={self.trainer.callback_metrics.get('val/loss', 0):.6f}  "
                f"val/wer={wer:.4f}"
            )

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self._lr,
            weight_decay=self._wd,
        )
        total = self.trainer.estimated_stepping_batches
        warmup = max(1, int(self._warmup_ratio * total))
        decay = max(1, total - warmup)
        print(f"  LR: {warmup} warmup + {decay} cosine = {total} steps")

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / decay)))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sch, "interval": "step"}]
