import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn import flash_attn_func

    IMPORT_FLASH = True
except Exception as err:
    IMPORT_FLASH = False
    IMPORT_FLASH_ERR = err

from .utils import apply_masked_flash_attn, apply_rotary_pos_emb


def _conformer_layer_fwd(
    layer: nn.Module,
    x: Tensor,
    pos_emb: Union[Tensor, List[Tensor]],
    att_mask: Optional[Tensor],
    pad_mask: Optional[Tensor],
    rc_layout: Optional[Tuple[int, int]] = None,
) -> Tensor:
    return layer(
        x=x, pos_emb=pos_emb, att_mask=att_mask, pad_mask=pad_mask, rc_layout=rc_layout
    )


class StreamingState:
    """
    Mutable per-utterance cache for cache-aware streaming inference. Holds
    everything needed so that feeding an utterance chunk-by-chunk reproduces the
    full masked forward exactly: the rotary position offset, per-subsampling-stage
    left-context frames, and per-Conformer-layer attention KV and depthwise-conv
    caches.

    ``pending_feat`` is used only by the right-context (``right_context_chunks ==
    1``) defer path: it buffers ONE chunk's worth of *subsampled* features (the
    layer-0 input) whose attention is still missing its 1-chunk look-ahead. The
    next call pairs it with the freshly subsampled chunk so its queries can attend
    to the look-ahead chunk's keys, after which it is emitted. It is always at
    rotary offset ``self.processed`` and is ``None`` while empty (first call, and
    after :meth:`flush_streaming`).

    Memory: with a bounded ``left_context_chunks`` every cache is bounded, so a
    state is O(left_context_chunks * chunk_size * n_layers) regardless of stream
    length. With ``left_context_chunks=-1`` the attention cache grows linearly
    with the stream and per-chunk cost becomes O(prefix) -- use a bounded window
    for long-running streams.
    """

    def __init__(self, n_layers: int, n_subs_convs: int):
        self.processed = 0  # encoder frames emitted so far (rotary offset)
        self.subs_cache: List[Optional[Tensor]] = [None] * n_subs_convs
        self.kv: List[Optional[Tuple[Tensor, Tensor]]] = [None] * n_layers
        self.conv: List[Optional[Tensor]] = [None] * n_layers
        self.pending_feat: Optional[Tensor] = None


class StridingSubsampling(nn.Module):
    """
    Strided Subsampling layer used to reduce the sequence length.
    """

    def __init__(
        self,
        subsampling: str,
        kernel_size: int,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        conv_channels: int,
        causal: bool = False,
    ):
        super().__init__()
        self.subsampling_type = subsampling
        assert self.subsampling_type in ["conv1d", "conv2d"]
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = kernel_size
        self.causal = causal
        # Causal streaming pads only on the left (kernel-1, applied manually in
        # forward) so an output frame never reads future input; symmetric
        # otherwise. Both schemes sum to the same total padding (kernel-1 for
        # odd kernels), so calc_output_length -- which uses _pad_total -- is
        # identical either way and no length bookkeeping changes.
        self._padding = 0 if causal else (self._kernel_size - 1) // 2
        self._pad_total = self._kernel_size - 1 if causal else 2 * self._padding

        layers: List[nn.Module] = []
        in_channels = 1 if self.subsampling_type == "conv2d" else feat_in
        subs_conv_class = (
            torch.nn.Conv2d if self.subsampling_type == "conv2d" else torch.nn.Conv1d
        )
        for _ in range(self._sampling_num):
            layers.append(
                subs_conv_class(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                )
            )
            layers.append(nn.ReLU())
            in_channels = conv_channels

        out_length = self.calc_output_length(torch.tensor(feat_in))
        if self.subsampling_type == "conv2d":
            self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def calc_output_length(
        self, lengths: Tensor, num_stages: Optional[int] = None
    ) -> Tensor:
        """
        Valid length after applying ``num_stages`` strided subsampling conv
        stages (defaults to all of them, i.e. the full subsampling output).
        """
        if num_stages is None:
            num_stages = self._sampling_num
        add_pad = self._pad_total - self._kernel_size
        lengths = lengths.to(torch.float)
        for _ in range(num_stages):
            lengths = torch.floor((lengths + add_pad) / self._stride + 1.0)
        return lengths.to(dtype=torch.int)

    def _mask_time(self, x: Tensor, lengths: Tensor) -> Tensor:
        """
        Zero out the padded tail along the time axis (dim 2). The subsampling
        convolutions are strided and have a receptive field wider than the
        stride, so the padded frames of shorter samples leak into the last
        valid frames. Left unmasked, the padding is the log-mel floor
        (``log(1e-9) ~= -20.7``) of zero-padded audio, not zero, so a batched
        short sample sees a different boundary than the same sample run alone
        (where conv zero-padding applies instead). Re-zeroing after every conv
        stage keeps the valid frames of batched inference aligned with the
        batch-size-1 result.
        """
        time = torch.arange(x.size(2), device=x.device)
        pad = time[None, :] >= lengths[:, None]  # [b, t]
        pad = pad[:, None]  # add channel dim -> [b, 1, t]
        if x.dim() == 4:
            pad = pad[..., None]  # add feature dim for conv2d -> [b, 1, t, 1]
        return x.masked_fill(pad, 0.0)

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        if self.subsampling_type == "conv2d":
            x = x.unsqueeze(1)
        else:
            x = x.transpose(1, 2)

        cur_len = lengths
        x = self._mask_time(x, cur_len)
        for module in self.conv:
            if self.causal and isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                # Left-pad the time axis (last dim for both conv1d [b,c,t] and
                # conv2d [b,c,t,f]) so the strided conv only sees past frames.
                pad = (
                    (0, 0, self._kernel_size - 1, 0)
                    if x.dim() == 4
                    else (
                        self._kernel_size - 1,
                        0,
                    )
                )
                x = F.pad(x, pad)
            x = module(x)
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                cur_len = self.calc_output_length(cur_len, 1)
                x = self._mask_time(x, cur_len)

        if self.subsampling_type == "conv2d":
            b, _, t, _ = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        else:
            x = x.transpose(1, 2)
        return x, self.calc_output_length(lengths)

    def forward_streaming(self, x: Tensor, state: "StreamingState") -> Tensor:
        """
        Causal subsampling for one streaming chunk (conv1d only). ``x`` is
        channels-first ``[b, feat_in, t]``. Each conv stage prepends its cached
        last ``kernel-1`` input frames (zeros on the first chunk, which matches
        the full forward's causal left-pad) instead of zero-padding, so the
        concatenation of all chunks reproduces the full forward exactly.
        """
        assert self.subsampling_type == "conv1d" and self.causal
        out = x
        ci = 0
        left = self._kernel_size - 1
        for module in self.conv:
            if isinstance(module, torch.nn.Conv1d):
                cache = state.subs_cache[ci]
                if cache is None:
                    cache = torch.zeros(
                        out.shape[0],
                        out.shape[1],
                        left,
                        device=out.device,
                        dtype=out.dtype,
                    )
                inp = torch.cat([cache, out], dim=2)
                state.subs_cache[ci] = inp[:, :, inp.shape[2] - left :]
                out = module(inp)
                ci += 1
            else:
                out = module(out)  # ReLU
        return out  # [b, d_model, t_enc]


class MultiHeadAttention(nn.Module, ABC):
    """
    Base class of Multi-Head Attention Mechanisms.
    """

    def __init__(
        self, n_head: int, n_feat: int, flash_attn=False, torch_sdpa_attn=False
    ):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.flash_attn = flash_attn
        self.torch_sdpa_attn = torch_sdpa_attn
        if self.flash_attn and not IMPORT_FLASH:
            raise RuntimeError(
                f"flash_attn_func was imported with err {IMPORT_FLASH_ERR}. "
                "Please install flash_attn or use --no_flash flag. "
                "If you have already done this, "
                "--force-reinstall flag might be useful"
            )

    def forward_qkv(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Projects the inputs into queries, keys, and values for multi-head attention.
        """
        b = query.size(0)
        q = self.linear_q(query).view(b, -1, self.h, self.d_k)
        k = self.linear_k(key).view(b, -1, self.h, self.d_k)
        v = self.linear_v(value).view(b, -1, self.h, self.d_k)
        if self.flash_attn:
            return q, k, v
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward_attention(
        self, value: Tensor, scores: Tensor, mask: Optional[Tensor]
    ) -> Tensor:
        """
        Computes the scaled dot-product attention given the projected values and scores.
        """
        b = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, value)
        x = x.transpose(1, 2).reshape(b, -1, self.h * self.d_k)
        return self.linear_out(x)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """
    Relative Position Multi-Head Attention module.
    """

    def __init__(self, n_head: int, n_feat: int):
        super().__init__(n_head, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))

    def rel_shift(self, x: Tensor) -> Tensor:
        b, h, qlen, pos_len = x.size()
        x = torch.nn.functional.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)
        pos_emb = pos_emb.to(dtype=self.linear_pos.weight.dtype)
        p = self.linear_pos(pos_emb)
        p = p.view(pos_emb.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RotaryPositionMultiHeadAttention(MultiHeadAttention):
    """
    Rotary Position Multi-Head Attention module.
    """

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: List[Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        b, t, _ = value.size()
        query = query.transpose(0, 1).view(t, b, self.h, self.d_k)
        key = key.transpose(0, 1).view(t, b, self.h, self.d_k)
        value = value.transpose(0, 1).view(t, b, self.h, self.d_k)

        cos, sin = pos_emb
        query, key = apply_rotary_pos_emb(query, key, cos, sin, offset=0)

        q, k, v = self.forward_qkv(
            query.view(t, b, self.h * self.d_k).transpose(0, 1),
            key.view(t, b, self.h * self.d_k).transpose(0, 1),
            value.view(t, b, self.h * self.d_k).transpose(0, 1),
        )

        if self.flash_attn:
            if mask is None:
                scores = flash_attn_func(q, k, v)
            else:
                scores = apply_masked_flash_attn(q, k, v, mask, self.h, self.d_k)
            scores = scores.view(b, -1, self.h * self.d_k)
            return self.linear_out(scores)
        elif self.torch_sdpa_attn:
            attn_mask = None if mask is None else ~mask.unsqueeze(1)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
            )
            attn_output = attn_output.transpose(1, 2).reshape(b, t, self.h * self.d_k)
            return self.linear_out(attn_output)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(self.d_k))
            return self.forward_attention(v, scores, mask)

    def forward_streaming(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: Optional[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Cache-aware self-attention for one chunk. ``x`` is ``[b, n, d]``; ``cos``/
        ``sin`` are the rotary tables sliced to this chunk's absolute positions
        ``[J, J+n)`` (so ``offset=0`` here). New keys/values are appended to the
        running cache and the query attends to the full left context + current
        chunk with no mask -- which reproduces the training block mask under
        unlimited left context. Keys are cached post-rotation; cached keys are
        never re-rotated (RoPE relative-phase property preserves parity).
        """
        b, n, _ = x.shape
        xh = x.transpose(0, 1).view(n, b, self.h, self.d_k)
        xq, xk = apply_rotary_pos_emb(xh, xh, cos, sin, offset=0)
        q, k, v = self.forward_qkv(
            xq.view(n, b, self.h * self.d_k).transpose(0, 1),
            xk.view(n, b, self.h * self.d_k).transpose(0, 1),
            x,
        )  # q, k, v: [b, h, n, d_k]
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).reshape(b, n, self.h * self.d_k)
        return self.linear_out(attn_output), (k, v)


class PositionalEncoding(nn.Module, ABC):
    """
    Base class of Positional Encodings.
    """

    def __init__(self, dim: int, base: int):
        super().__init__()
        self.dim = dim
        self.base = base

    @abstractmethod
    def create_pe(self, length: int, device: torch.device) -> Optional[Tensor]:
        pass

    def extend_pe(self, length: int, device: torch.device):
        """
        Extends the positional encoding buffer to process longer sequences.
        """
        pe = self.create_pe(length, device)
        if pe is None:
            return
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)


class RelPositionalEmbedding(PositionalEncoding):
    """
    Relative Positional Embedding module.
    """

    def create_pe(self, length: int, device: torch.device) -> Optional[Tensor]:
        """
        Creates the relative positional encoding matrix.
        """
        if hasattr(self, "pe") and self.pe.shape[1] >= 2 * length - 1:
            return None
        positions = torch.arange(length - 1, -length, -1, device=device).unsqueeze(1)
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.dim, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=pe.device)
            * -(math.log(10000.0) / self.dim)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        input_len = x.size(1)
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        return x, self.pe[:, start_pos:end_pos]


class RotaryPositionalEmbedding(PositionalEncoding):
    """
    Rotary Positional Embedding module.
    """

    def create_pe(self, length: int, device: torch.device) -> Optional[Tensor]:
        """
        Creates or extends the rotary positional encoding matrix.
        """
        if hasattr(self, "pe") and self.pe.size(0) >= 2 * length:
            return None
        positions = torch.arange(0, length, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        t = torch.arange(length, device=positions.device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(positions.device)
        return torch.cat([emb.cos()[:, None, None, :], emb.sin()[:, None, None, :]])

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, List[Tensor]]:
        cos_emb = self.pe[0 : x.shape[1]]
        half_pe = self.pe.shape[0] // 2
        sin_emb = self.pe[half_pe : half_pe + x.shape[1]]
        return x, [cos_emb, sin_emb]


class ConformerConvolution(nn.Module):
    """
    Conformer Convolution module.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        norm_type: str,
        causal: bool = False,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        assert norm_type in ["batch_norm", "layer_norm"]
        self.norm_type = norm_type
        self.causal = causal
        self.kernel_size = kernel_size
        # Causal depthwise conv left-pads by kernel-1 (applied in forward) so a
        # frame only sees itself and the past; symmetric padding looks ahead by
        # (kernel-1)//2 and would compound that lookahead across every layer.
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=0 if causal else (kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        self.batch_norm = (
            nn.BatchNorm1d(d_model)
            if norm_type == "batch_norm"
            else nn.LayerNorm(d_model)
        )
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        if self.causal:
            x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.depthwise_conv(x)
        if self.norm_type == "batch_norm":
            x = self.batch_norm(x)
        else:
            x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)

    def forward_streaming(
        self,
        x: Tensor,
        conv_cache: Optional[Tensor],
        lookahead_len: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Causal conv for one chunk. ``x`` is ``[b, t, d]``. The depthwise conv's
        left context is supplied by ``conv_cache`` (its last ``kernel-1`` input
        frames; zeros on the first chunk, matching the training left-pad) instead
        of zero-padding. Returns the output and the updated cache.

        ``lookahead_len`` serves the right-context (defer=1) path. When ``x``
        holds two chunks concatenated ``[pending, lookahead]`` and we only EMIT the
        pending chunk, the cache must retain the pending chunk's tail -- NOT the
        look-ahead chunk's tail -- so the next call's pending chunk is not
        re-counted as its own left context. ``lookahead_len`` frames are skipped
        before taking the trailing ``kernel-1`` cache slice.
        """
        assert self.causal
        x = x.transpose(1, 2)  # [b, d, t]
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        left = self.kernel_size - 1
        if conv_cache is None:
            conv_cache = torch.zeros(
                x.shape[0], x.shape[1], left, device=x.device, dtype=x.dtype
            )
        inp = torch.cat([conv_cache, x], dim=2)
        if lookahead_len > 0:
            # inp ends with [..., pending (C), lookahead (lookahead_len)]; keep
            # the ``left`` frames ending right before the lookahead region.
            new_cache = inp[:, :, -(lookahead_len + left) : -lookahead_len]
        else:
            new_cache = inp[:, :, inp.shape[2] - left :]
        x = self.depthwise_conv(inp)
        if self.norm_type == "batch_norm":
            x = self.batch_norm(x)
        else:
            x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2), new_cache

    def forward_hardcopy(
        self,
        x: Tensor,
        pad_mask: Optional[Tensor],
        n_chunks: int,
        chunk_size: int,
    ) -> Tensor:
        """
        Causal conv over the hard-copied right-context layout used by the
        streaming *training* forward (see ``ConformerEncoder.forward``). ``x`` is
        ``[b, 2*N*C, d]``: the first ``N*C`` frames are the main sequence (chunk
        ``c`` at ``[c*C, (c+1)*C)``), the last ``N*C`` are the right-context
        copies (copy ``c`` is a fresh copy of chunk ``c+1`` that is evolved under
        chunk ``c``'s restricted context).

        The main sequence gets the ordinary causal conv. Copy ``c`` gets, as its
        left context, the tail of *main* chunk ``c`` -- exactly what the chunked
        inference path supplies to the look-ahead chunk (``forward_streaming``
        with ``lookahead_len``: conv over ``[cache, pending, lookahead]`` where the
        lookahead's left context is the pending chunk's tail).
        """
        assert self.causal
        b, l2, _ = x.shape
        nc = n_chunks * chunk_size
        assert l2 == 2 * nc, f"hard-copy layout expects 2*{nc} frames, got {l2}"
        x = x.transpose(1, 2)  # [b, d, 2NC]
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        d = x.shape[1]
        left = self.kernel_size - 1
        main, copies = x[:, :, :nc], x[:, :, nc:]

        main_padded = F.pad(main, (left, 0))  # [b, d, left + NC]
        main_out = self.depthwise_conv(main_padded)  # [b, d, NC]

        # Tail of main chunk c = padded frames [(c+1)*C, (c+1)*C + left). Windows
        # of ``left`` frames with stride C starting at padded offset C give
        # exactly those N tails, in chunk order.
        tails = main_padded[:, :, chunk_size:].unfold(2, left, chunk_size)
        tails = tails[:, :, :n_chunks]  # [b, d, N, left]
        copies_r = copies.view(b, d, n_chunks, chunk_size)  # [b, d, N, C]
        inp = torch.cat([tails, copies_r], dim=3)  # [b, d, N, left + C]
        inp = inp.permute(0, 2, 1, 3).reshape(b * n_chunks, d, left + chunk_size)
        copies_out = self.depthwise_conv(inp)  # [b*N, d, C]
        copies_out = copies_out.view(b, n_chunks, d, chunk_size).permute(0, 2, 1, 3)
        copies_out = copies_out.reshape(b, d, nc)

        x = torch.cat([main_out, copies_out], dim=2)  # [b, d, 2NC]
        if self.norm_type == "batch_norm":
            x = self.batch_norm(x)
        else:
            x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ConformerFeedForward(nn.Module):
    """
    Conformer Feed Forward module.
    """

    def __init__(self, d_model: int, d_ff: int, use_bias=True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class ConformerLayer(nn.Module):
    """
    Conformer Layer module.
    This module combines several submodules including feed forward networks,
    depthwise separable convolution, and multi-head self-attention
    to form a single Conformer block.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        self_attention_model: str,
        n_heads: int = 16,
        conv_norm_type: str = "batch_norm",
        conv_kernel_size: int = 31,
        flash_attn: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.fc_factor = 0.5
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            causal=causal,
        )
        self.norm_self_att = nn.LayerNorm(d_model)
        if self_attention_model == "rotary":
            self.self_attn: nn.Module = RotaryPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                flash_attn=flash_attn,
                torch_sdpa_attn=not flash_attn,
            )
        else:
            assert not flash_attn, "Not supported flash_attn for rel_pos"
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
            )
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        pos_emb: Union[Tensor, List[Tensor]],
        att_mask: Optional[Tensor] = None,
        pad_mask: Optional[Tensor] = None,
        rc_layout: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """``rc_layout=(n_chunks, chunk_size)`` selects the hard-copied
        right-context layout for the conv module (streaming training with
        ``right_context_chunks=1``); attention and the feed-forwards are layout
        agnostic (the caller supplies a matching ``att_mask``/``pos_emb``)."""
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x * self.fc_factor

        x = self.norm_self_att(residual)
        x = self.self_attn(x, x, x, pos_emb, mask=att_mask)
        residual = residual + x

        x = self.norm_conv(residual)
        if rc_layout is None:
            x = self.conv(x, pad_mask=pad_mask)
        else:
            x = self.conv.forward_hardcopy(x, pad_mask, *rc_layout)
        residual = residual + x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + x * self.fc_factor

        x = self.norm_out(residual)
        return x

    def forward_streaming(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: Optional[Tuple[Tensor, Tensor]],
        conv_cache: Optional[Tensor],
        lookahead_len: int = 0,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        """Cache-aware Conformer block for one chunk. Mirrors ``forward`` exactly
        but threads the attention KV-cache and depthwise-conv cache.

        ``lookahead_len`` enables the right-context (defer=1) path: ``x`` then
        holds two concatenated chunks ``[pending, lookahead]`` whose attention is
        computed together (the pending chunk's queries see the look-ahead chunk's
        keys), and the depthwise-conv cache retains the pending chunk's tail. The
        attention KV cache is returned UN-trimmed (cache + pending + lookahead);
        the caller drops the look-ahead frames so the persistent cache holds only
        emitted (left-context) keys."""
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x * self.fc_factor

        x = self.norm_self_att(residual)
        x, new_kv = self.self_attn.forward_streaming(x, cos, sin, kv_cache)
        residual = residual + x

        x = self.norm_conv(residual)
        x, new_conv = self.conv.forward_streaming(
            x, conv_cache, lookahead_len=lookahead_len
        )
        residual = residual + x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + x * self.fc_factor

        x = self.norm_out(residual)
        return x, new_kv, new_conv


class ConformerEncoder(nn.Module):
    """
    Conformer Encoder module.
    This module encapsulates the entire Conformer encoder architecture,
    consisting of a StridingSubsampling layer, positional embeddings, and
    a stack of Conformer Layers.
    It serves as the main component responsible for processing speech features.
    """

    def __init__(
        self,
        feat_in: int = 64,
        n_layers: int = 16,
        d_model: int = 768,
        subsampling: str = "conv2d",
        subs_kernel_size: int = 3,
        subsampling_factor: int = 4,
        ff_expansion_factor: int = 4,
        self_attention_model: str = "rotary",
        n_heads: int = 16,
        pos_emb_max_len: int = 5000,
        conv_norm_type: str = "batch_norm",
        conv_kernel_size: int = 31,
        flash_attn: bool = False,
        activation_checkpointing: bool = False,
        streaming: bool = False,
        chunk_size: int = 8,
        left_context_chunks: int = 16,
        right_context_chunks: int = 0,
    ):
        super().__init__()
        self.feat_in = feat_in
        self.d_model = d_model
        self.activation_checkpointing = activation_checkpointing
        # Streaming turns the encoder into a chunked model: causal subsampling/
        # conv + block attention. A query in chunk c attends to chunks
        # [c - left_context_chunks, c] plus, with right_context_chunks=1, a
        # hard copy of chunk c+1 (Emformer-style). left=-1 means unlimited past.
        # Both forward() (training / offline) and the chunked inference path
        # compute the identical function, so a model fine-tuned with forward()
        # runs chunk-by-chunk with no train/inference mismatch. Adds no params.
        self.streaming = streaming
        self.chunk_size = chunk_size
        self.left_context_chunks = left_context_chunks
        self.right_context_chunks = right_context_chunks
        if streaming:
            assert (
                self_attention_model == "rotary"
            ), "streaming supports only rotary attention"
            assert chunk_size >= 1, "chunk_size must be >= 1"
            assert right_context_chunks in (0, 1), (
                "right_context_chunks must be 0 (strictly causal) or 1 (one chunk "
                "of hard-copied look-ahead)"
            )
            # Block attention masks can't be expressed through the flash-attn
            # path; force torch sdpa, which accepts an arbitrary bool att_mask.
            flash_attn = False
        assert self_attention_model in [
            "rotary",
            "rel_pos",
        ], f"Not supported attn = {self_attention_model}"

        self.pre_encode = StridingSubsampling(
            subsampling=subsampling,
            kernel_size=subs_kernel_size,
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=d_model,
            causal=streaming,
        )

        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rotary":
            self.pos_enc: PositionalEncoding = RotaryPositionalEmbedding(
                d_model // n_heads, pos_emb_max_len
            )
        else:
            self.pos_enc = RelPositionalEmbedding(d_model, pos_emb_max_len)

        # Build the positional-encoding buffer eagerly, in this (normal) context.
        # Creating it lazily inside forward() makes its dtype/device and, under
        # modern PyTorch, its *inference-ness* depend on the first call: if that
        # call happens under torch.inference_mode() (e.g. Lightning's initial
        # validation), ``pe`` becomes an inference tensor and every later
        # grad-enabled forward — i.e. training — raises "Inference tensors cannot
        # be saved for backward". A pe built here is a normal buffer that moves
        # with .to()/.half() and never poisons autograd.
        self.pos_enc.extend_pe(pos_emb_max_len, torch.device("cpu"))

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_model * ff_expansion_factor,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_norm_type=conv_norm_type,
                conv_kernel_size=conv_kernel_size,
                flash_attn=flash_attn,
                causal=streaming,
            )
            self.layers.append(layer)

    def input_example(
        self,
        batch_size: int = 8,
        seqlen: int = 200,
    ) -> Tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        features = torch.randn(batch_size, self.feat_in, seqlen)
        feature_lengths = torch.randint(1, seqlen + 1, (batch_size,))
        feature_lengths[0] = seqlen
        return features.float().to(device), feature_lengths.to(device)

    def input_names(self) -> List[str]:
        return ["audio_signal", "length"]

    def output_names(self) -> List[str]:
        return ["encoded", "encoded_len"]

    @contextmanager
    def onnx_export_mode(self):
        saved = []
        for layer in self.layers:
            attn = layer.self_attn
            saved.append((attn.flash_attn, attn.torch_sdpa_attn))
            attn.flash_attn = False
            attn.torch_sdpa_attn = False
        try:
            yield
        finally:
            for layer, (fa, sdpa) in zip(self.layers, saved):
                layer.self_attn.flash_attn = fa
                layer.self_attn.torch_sdpa_attn = sdpa

    def dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "audio_signal": {0: "batch_size", 2: "seq_len"},
            "length": {0: "batch_size"},
            "encoded": {0: "batch_size", 1: "seq_len"},
            "encoded_len": {0: "batch_size"},
        }

    def _chunk_att_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Strictly causal block (chunk-limited) attention mask in encoder-frame
        space, using the codebase convention True = disallowed. A query in chunk
        ``c(q)=q//chunk`` may attend to a key in chunk ``c(k)`` iff ``c(k) <= c(q)``
        and, for a bounded window, ``c(k) >= c(q) - left_context_chunks``. Used
        by the ``right_context_chunks=0`` streaming forward; the look-ahead
        variant lives in :meth:`_hardcopy_att_mask`.
        """
        idx = torch.arange(seq_len, device=device)
        cq = (idx // self.chunk_size).unsqueeze(1)  # [T, 1]
        ck = (idx // self.chunk_size).unsqueeze(0)  # [1, T]
        disallow = ck > cq
        if self.left_context_chunks >= 0:
            disallow = disallow | (ck < (cq - self.left_context_chunks))
        return disallow  # [T, T], True = masked

    def _hardcopy_att_mask(self, n_chunks: int, device: torch.device) -> Tensor:
        """
        Attention mask over the hard-copied right-context layout ``[main (N*C) |
        copies (N*C)]`` where copy ``c`` is a copy of chunk ``c+1``. Any query
        belonging to chunk ``c`` (main or copy) may attend to main keys in chunks
        ``[c - left_context_chunks, c]`` and to copy keys of chunk ``c`` only.
        That is exactly the key set the chunked inference path exposes to the
        pending chunk and to its look-ahead chunk, so training == inference at
        every depth (the copies never see *their* own look-ahead, which is what
        a naive ``c(k) <= c(q)+1`` mask would let compound through the layers).
        """
        c = self.chunk_size
        nc = n_chunks * c
        chunk_of = torch.arange(nc, device=device) // c
        cidx = torch.cat([chunk_of, chunk_of])  # [2NC]
        is_copy = torch.cat(
            [
                torch.zeros(nc, dtype=torch.bool, device=device),
                torch.ones(nc, dtype=torch.bool, device=device),
            ]
        )
        cq = cidx.unsqueeze(1)  # [2NC, 1]
        ck = cidx.unsqueeze(0)  # [1, 2NC]
        main_disallow = ck > cq
        if self.left_context_chunks >= 0:
            main_disallow = main_disallow | (ck < (cq - self.left_context_chunks))
        copy_disallow = ck != cq
        return torch.where(is_copy.unsqueeze(0), copy_disallow, main_disallow)

    def forward(self, audio_signal: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        if not hasattr(self.pos_enc, "pe"):
            self.pos_enc.extend_pe(self.pos_emb_max_len, audio_signal.device)

        audio_signal, length = self.pre_encode(
            x=audio_signal.transpose(1, 2), lengths=length
        )
        if self.streaming and self.right_context_chunks == 1:
            return self._forward_hardcopy(audio_signal, length)

        max_len = audio_signal.size(1)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)

        pad_mask = torch.arange(0, max_len, device=audio_signal.device).expand(
            length.size(0), -1
        ) < length.unsqueeze(-1)

        att_mask = None
        if self.streaming:
            # Always build a chunked mask (even at batch 1). Mask padded *keys*
            # and always let a query see itself, so no softmax row is ever fully
            # masked (a fully masked row is NaN under sdpa and would poison the
            # whole batch through the next layer's keys). Padded query rows are
            # garbage but finite; they are masked as keys and zeroed in the conv.
            block = self._chunk_att_mask(max_len, audio_signal.device)  # [T,T]
            key_pad = ~pad_mask  # [B, T], True where key is padding
            att_mask = block.unsqueeze(0) | key_pad.unsqueeze(1)  # [B, T, T]
            att_mask = att_mask & ~torch.eye(
                max_len, dtype=torch.bool, device=att_mask.device
            )
        elif audio_signal.shape[0] > 1:
            att_mask = pad_mask.unsqueeze(1).repeat([1, max_len, 1])
            att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))
            att_mask = ~att_mask

        pad_mask = ~pad_mask

        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                audio_signal = checkpoint(
                    _conformer_layer_fwd,
                    layer,
                    audio_signal,
                    pos_emb,
                    att_mask,
                    pad_mask,
                    use_reentrant=False,
                )
            else:
                audio_signal = layer(
                    x=audio_signal,
                    pos_emb=pos_emb,
                    att_mask=att_mask,
                    pad_mask=pad_mask,
                )

        return audio_signal.transpose(1, 2), length

    def _forward_hardcopy(self, x: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Streaming forward with one chunk of hard-copied right context (Emformer
        style). ``x`` is the subsampled sequence ``[B, T, d]``.

        The sequence is padded to ``N`` whole chunks and a second copy of every
        chunk ``c+1`` is appended as the look-ahead block of chunk ``c``:
        ``[main (N*C) | copies (N*C)]``. Copies carry the rotary positions of the
        frames they copy (``t + C``). Main chunk ``c`` and its copy both attend
        to main chunks ``[c-L, c]`` and to the copy -- never to anything later --
        and the copy's depthwise conv takes main chunk ``c``'s tail as its left
        context. This is, frame for frame, the computation the chunked inference
        path performs on ``[cache, pending, look-ahead]``, so a model trained
        here streams with zero mismatch (verified by parity tests at every
        depth). The copies are dropped from the output. Cost is roughly 2x the
        causal forward (twice the frames, 4x the attention area, all masked).
        """
        b, t, d = x.shape
        c = self.chunk_size
        n_chunks = max(1, -(-t // c))
        nc = n_chunks * c
        device = x.device
        if nc > t:
            x = F.pad(x, (0, 0, 0, nc - t))
        copies = torch.cat([x[:, c:], x.new_zeros(b, c, d)], dim=1)  # [B, NC, d]
        aug = torch.cat([x, copies], dim=1)  # [B, 2NC, d]

        # Rotary tables: main frame t at position t; copy frame t at t + C.
        pe = self.pos_enc.pe
        half = pe.shape[0] // 2
        if nc + c > half:
            self.pos_enc.extend_pe(nc + c, device)
            pe = self.pos_enc.pe
            half = pe.shape[0] // 2
        cos = torch.cat([pe[:nc], pe[c : nc + c]], dim=0)
        sin = torch.cat([pe[half : half + nc], pe[half + c : half + nc + c]], dim=0)
        pos_emb = [cos, sin]

        # Padding: a main frame t is valid iff t < length; copy frame t copies
        # frame t + C, so it is valid iff t + C < length.
        pos = torch.arange(nc, device=device)
        valid_main = pos.unsqueeze(0) < length.unsqueeze(1)  # [B, NC]
        valid_copy = (pos + c).unsqueeze(0) < length.unsqueeze(1)
        key_pad = ~torch.cat([valid_main, valid_copy], dim=1)  # [B, 2NC], True=pad

        block = self._hardcopy_att_mask(n_chunks, device)  # [2NC, 2NC]
        att_mask = block.unsqueeze(0) | key_pad.unsqueeze(1)  # [B, 2NC, 2NC]
        att_mask = att_mask & ~torch.eye(2 * nc, dtype=torch.bool, device=device)
        rc_layout = (n_chunks, c)

        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                aug = checkpoint(
                    _conformer_layer_fwd,
                    layer,
                    aug,
                    pos_emb,
                    att_mask,
                    key_pad,
                    rc_layout,
                    use_reentrant=False,
                )
            else:
                aug = layer(
                    x=aug,
                    pos_emb=pos_emb,
                    att_mask=att_mask,
                    pad_mask=key_pad,
                    rc_layout=rc_layout,
                )

        return aug[:, :t].transpose(1, 2), length

    def init_streaming_state(self) -> StreamingState:
        """Fresh per-utterance cache for :meth:`forward_streaming_chunk`."""
        n_subs = sum(1 for m in self.pre_encode.conv if isinstance(m, torch.nn.Conv1d))
        return StreamingState(len(self.layers), n_subs)

    def _rotary_for(
        self, j: int, n: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Cos/sin tables for ``n`` encoder frames starting at absolute position
        ``j``, growing the rotary buffer when the stream exceeds ``pos_emb_max_len``."""
        pe = self.pos_enc.pe
        half = pe.shape[0] // 2
        # Grow the rotary buffer for streams longer than pos_emb_max_len (default
        # 5000 -> half=2500 encoder frames ~= 100 s at 25 fps). Rotary cos/sin
        # are position-wise, so extending the buffer does not change the values
        # already emitted for earlier positions; it only appends higher indices.
        if j + n > half:
            self.pos_enc.extend_pe(j + n, device)
            pe = self.pos_enc.pe
            half = pe.shape[0] // 2
        return pe[j : j + n], pe[half + j : half + j + n]

    def _evict_left_context(self, state: StreamingState) -> None:
        """Bound each layer's attention cache to the training left-context window.
        The next chunk's queries may only see keys at positions >= (j+n) -
        left_context_chunks*chunk_size, so keep that many trailing frames. With
        left_context_chunks=-1 (unlimited) keep everything."""
        if self.left_context_chunks >= 0:
            keep = self.left_context_chunks * self.chunk_size
            state.kv = [self._trim_kv(kv, keep) for kv in state.kv]

    def forward_streaming_chunk(
        self, feat_chunk: Tensor, state: StreamingState
    ) -> Tensor:
        """
        Encode one streaming chunk of features, updating ``state`` in place.
        ``feat_chunk`` is channels-first ``[b, feat_in, t_feat]`` (raw log-mel,
        e.g. 32 feature frames -> 8 encoder frames at chunk_size=8). Returns the
        encoded chunk ``[b, d_model, n_enc]``.

        Every call except the last must carry a whole number of encoder frames,
        i.e. ``t_feat`` a multiple of the subsampling factor (and normally exactly
        ``chunk_size * factor``); the strided subsampling caches align output
        frames to the chunk boundary, so an odd-sized intermediate chunk would
        shift the frame grid relative to ``forward``. The final chunk may be any
        size. :class:`gigaam.streaming.StreamingSession` enforces this.

        With ``right_context_chunks == 0`` (strictly causal) each chunk is emitted
        immediately and concatenating outputs reproduces ``forward`` exactly
        (verified by parity test).

        With ``right_context_chunks == 1`` emission is deferred by one chunk: a
        query in chunk ``c`` must attend to keys in chunk ``c+1``. The most
        recently subsampled chunk is held in ``state.pending_feat``; the next call
        pairs it with the fresh chunk so the pending chunk's queries see the fresh
        chunk's keys, and the pending chunk is then emitted. The first call
        therefore emits nothing (it only buffers), and the final pending chunk is
        drained by :meth:`flush_streaming`. Concatenating outputs reproduces the
        hard-copy ``forward`` exactly at any depth (verified by parity test).
        Per-call cost is O(chunk_size * n_layers) with a bounded left context.
        """
        assert self.streaming, "forward_streaming_chunk requires streaming=True"
        if not hasattr(self.pos_enc, "pe"):
            self.pos_enc.extend_pe(self.pos_emb_max_len, feat_chunk.device)

        if self.right_context_chunks == 0:
            return self._forward_streaming_chunk_causal(feat_chunk, state)
        return self._forward_streaming_chunk_defer(feat_chunk, state)

    def _forward_streaming_chunk_causal(
        self, feat_chunk: Tensor, state: StreamingState
    ) -> Tensor:
        """Strictly-causal (right_context_chunks=0) streaming path. Each chunk is
        emitted immediately; concatenating outputs reproduces ``forward``."""
        x = self.pre_encode.forward_streaming(feat_chunk, state)  # [b, d, n]
        x = x.transpose(1, 2)  # [b, n, d]
        n = x.shape[1]
        if n == 0:
            return x.transpose(1, 2)

        j = state.processed
        cos, sin = self._rotary_for(j, n, feat_chunk.device)

        for i, layer in enumerate(self.layers):
            x, state.kv[i], state.conv[i] = layer.forward_streaming(
                x, cos, sin, state.kv[i], state.conv[i]
            )
        self._evict_left_context(state)
        state.processed += n
        return x.transpose(1, 2)  # [b, d, n]

    def _forward_streaming_chunk_defer(
        self, feat_chunk: Tensor, state: StreamingState
    ) -> Tensor:
        """Right-context (defer=1) streaming path.

        Subsample the new chunk only (the causal subsampling cache advances by
        exactly this chunk -- no overlap, no double-count). If a pending chunk is
        buffered, run the whole layer stack over ``[pending, new]`` so the pending
        chunk's queries attend to the new chunk's keys (the 1-chunk look-ahead),
        then emit the pending chunk and keep the new chunk as the next pending.
        The attention KV cache holds only EMITTED (left-context) keys: the new
        chunk's keys are used transiently and then trimmed off; the depthwise-conv
        cache retains the pending chunk's tail (see ``lookahead_len``). All work is
        over two chunks, so cost is O(chunk_size * n_layers) for a bounded left
        context.

        The look-ahead chunk is computed here with exactly the context the
        hard-copy training forward gives its copy (left context + pending +
        itself), so this path reproduces ``forward`` exactly at any depth."""
        s_new = self.pre_encode.forward_streaming(feat_chunk, state)  # [b, d, n_new]
        s_new = s_new.transpose(1, 2)  # [b, n_new, d]
        n_new = s_new.shape[1]
        if n_new == 0:
            return s_new.transpose(1, 2)  # nothing subsampled; emit nothing

        if state.pending_feat is None:
            # First chunk: no look-ahead available yet. Buffer it and emit nothing.
            state.pending_feat = s_new
            return s_new.new_zeros(s_new.shape[0], s_new.shape[2], 0)

        s_pend = state.pending_feat  # [b, n_pend, d]
        n_pend = s_pend.shape[1]
        x = torch.cat([s_pend, s_new], dim=1)  # [b, n_pend + n_new, d]
        j = state.processed  # absolute offset of the pending chunk
        cos, sin = self._rotary_for(j, n_pend + n_new, feat_chunk.device)

        for i, layer in enumerate(self.layers):
            x, full_kv, new_conv = layer.forward_streaming(
                x, cos, sin, state.kv[i], state.conv[i], lookahead_len=n_new
            )
            # The attention appended BOTH chunks' keys to the cache. Drop the new
            # (look-ahead) chunk's keys so only emitted left-context persists; the
            # new chunk is fed fresh next call (as the pending half).
            state.kv[i] = self._drop_lookahead_kv(full_kv, n_new)
            state.conv[i] = new_conv

        self._evict_left_context(state)
        out_pend = x[:, :n_pend, :]  # emit the (now look-ahead-complete) chunk
        state.processed += n_pend
        state.pending_feat = s_new  # the fresh chunk awaits its own look-ahead
        return out_pend.transpose(1, 2)  # [b, d, n_pend]

    @staticmethod
    def _drop_lookahead_kv(
        kv: Optional[Tuple[Tensor, Tensor]], lookahead_len: int
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Drop the last ``lookahead_len`` time-frames of a cached (k, v) pair
        (dim 2), so the persistent cache holds only emitted left-context keys."""
        if kv is None or lookahead_len == 0:
            return kv
        k, v = kv
        keep = k.size(2) - lookahead_len
        return (k[:, :, :keep, :], v[:, :, :keep, :])

    def flush_streaming(self, state: StreamingState) -> Tensor:
        """
        Drain the final buffered chunk at end of stream and return its encoding
        ``[b, d_model, n]``. With ``right_context_chunks == 0`` nothing is ever
        buffered, so this returns an empty tensor. With ``right_context_chunks >
        0`` the last pending chunk has no real look-ahead (the stream ended), so it
        is run alone -- exactly matching the training mask, whose look-ahead chunk
        for the final query chunk simply does not exist. Safe to call once per
        stream after the last :meth:`forward_streaming_chunk`; idempotent
        (clears ``pending_feat``).
        """
        if self.right_context_chunks == 0 or state.pending_feat is None:
            state.pending_feat = None
            p = next(self.parameters())
            return torch.zeros(1, self.d_model, 0, device=p.device, dtype=p.dtype)

        s_pend = state.pending_feat  # [b, n_pend, d]
        n_pend = s_pend.shape[1]
        j = state.processed
        cos, sin = self._rotary_for(j, n_pend, s_pend.device)
        x = s_pend
        for i, layer in enumerate(self.layers):
            x, state.kv[i], state.conv[i] = layer.forward_streaming(
                x, cos, sin, state.kv[i], state.conv[i]
            )
        self._evict_left_context(state)
        state.processed += n_pend
        state.pending_feat = None
        return x.transpose(1, 2)  # [b, d, n_pend]

    @staticmethod
    def _trim_kv(
        kv: Optional[Tuple[Tensor, Tensor]], keep: int
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Keep only the last ``keep`` time-frames of a cached (k, v) pair (dim 2)."""
        if kv is None:
            return None
        k, v = kv
        if k.size(2) <= keep:
            return kv
        return (k[:, :, -keep:, :], v[:, :, -keep:, :])
