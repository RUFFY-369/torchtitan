# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import ClassVar, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

try:
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,
        _score_mod_signature,
        BlockMask,
        create_block_mask,
        flex_attention,
    )
except ImportError:
    _mask_mod_signature = _score_mod_signature = Callable
    BlockMask = create_block_mask = flex_attention = None

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None

from torch.types import Number


class VarlenMetadata(NamedTuple):
    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    max_q: Number
    max_k: Number


class VarlenAttentionWrapper(nn.Module):
    \"\"\"Wrapper for varlen_attn with torch.compile support.\"\"\"

    _compiled_varlen_attn: ClassVar[Callable] = varlen_attn

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
    ) -> torch.Tensor:
        # xq, xk, xv shape: [batch_size, seq_len, num_heads, head_dim]
        # varlen_attn expects packed input: [total_seq_len, num_heads, head_dim]
        xq_packed = xq.transpose(1, 2).flatten(0, 1)
        xk_packed = xk.transpose(1, 2).flatten(0, 1)
        xv_packed = xv.transpose(1, 2).flatten(0, 1)

        if VarlenAttentionWrapper._compiled_varlen_attn is None:
            raise ImportError(
                \"varlen_attn is not available in this torch version. \"
                \"Please upgrade to a version that supports it.\"
            )

        return VarlenAttentionWrapper._compiled_varlen_attn(
            xq_packed,
            xk_packed,
            xv_packed,
            cu_seqlens_q=attention_masks.cu_seq_q,
            cu_seqlens_k=attention_masks.cu_seq_k,
            max_seqlen_q=attention_masks.max_q,
            max_seqlen_k=attention_masks.max_k,
            scale=scale,
            # Current implementation assumes causal mask
            window_size=(-1, 0),
        )


class FlexAttentionWrapper(nn.Module):
    \"\"\"Wrapper for flex_attention with torch.compile support.\"\"\"

    _compiled_flex_attn: ClassVar[Callable] = flex_attention

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        score_mod: _score_mod_signature | None = None,
        block_mask: BlockMask | None = None,
        scale: float | None = None,
        return_lse: bool = False,
        enable_gqa: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if FlexAttentionWrapper._compiled_flex_attn is None:
            raise ImportError(
                \"flex_attention is not available in this torch version. \"
                \"Please upgrade to a version that supports it.\"
            )

        return FlexAttentionWrapper._compiled_flex_attn(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=return_lse,
        )


class ScaledDotProductAttentionWrapper(nn.Module):
    \"\"\"Wrapper for scaled_dot_product_attention with backend selection.\"\"\"

    sdpa_backends = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
    ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
    ) -> torch.Tensor:
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                is_causal=is_causal,
                enable_gqa=enable_gqa,
            )


def get_causal_mask_mod() -> _mask_mod_signature:
    \"\"\"Returns a mask_mod that implements a standard causal mask.\"\"\"

    def causal_mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return causal_mask_mod


# Helper for block mask creation
_compiled_create_block_mask = create_block_mask


def create_attention_mask(*args, **kwargs) -> BlockMask:
    if _compiled_create_block_mask is None:
        raise ImportError(
            \"create_block_mask is not available in this torch version. \"
            \"Please upgrade to a version that supports it.\"
        )
    return _compiled_create_block_mask(*args, **kwargs)


def create_varlen_metadata_from_sequence_lengths(
    sequence_lengths: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> VarlenMetadata:
    \"\"\"
    Create VarlenMetadata from sequence lengths for varlen_attn.
    \"\"\"
    batch_size = len(sequence_lengths)
    cu_seqlens_list = []
    all_seq_lengths = []
    offset = 0

    for b in range(batch_size):
        sample_seq_lens = sequence_lengths[b]
        sample_cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                torch.cumsum(sample_seq_lens.to(torch.int32), dim=0),
            ]
        )
        all_seq_lengths.append(sample_seq_lens)
        cu_seqlens_list.append((sample_cu_seqlens[:-1] + offset).to(torch.int32))
        offset += seq_len

    packed_cu_seqlens = torch.cat(
        cu_seqlens_list + [torch.tensor([offset], dtype=torch.int32, device=device)]
    ).to(torch.int32)

    max_seqlen = torch.cat(all_seq_lengths).max().item() if all_seq_lengths else 0

    return VarlenMetadata(
        cu_seq_q=packed_cu_seqlens,
        cu_seq_k=packed_cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )
