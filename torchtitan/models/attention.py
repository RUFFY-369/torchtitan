# Copyright (c) Meta Platforms, Inc. and affiliates.
from collections.abc import Callable
from typing import ClassVar, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
try:
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature, _score_mod_signature, BlockMask, create_block_mask, flex_attention,
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
    _compiled_varlen_attn: ClassVar[Callable] = varlen_attn
    def forward(self, xq, xk, xv, attention_masks, scale=None):
        xq_packed = xq.transpose(1, 2).flatten(0, 1)
        xk_packed = xk.transpose(1, 2).flatten(0, 1)
        xv_packed = xv.transpose(1, 2).flatten(0, 1)
        if VarlenAttentionWrapper._compiled_varlen_attn is None:
             raise ImportError("varlen_attn is not available")
        return VarlenAttentionWrapper._compiled_varlen_attn(
            xq_packed, xk_packed, xv_packed,
            cu_seqlens_q=attention_masks.cu_seq_q, cu_seqlens_k=attention_masks.cu_seq_k,
            max_seqlen_q=attention_masks.max_q, max_seqlen_k=attention_masks.max_k,
            scale=scale, window_size=(-1, 0),
        )
class FlexAttentionWrapper(nn.Module):
    _compiled_flex_attn: ClassVar[Callable] = flex_attention
    def forward(self, q, k, v, score_mod=None, block_mask=None, scale=None, return_lse=False, enable_gqa=False):
        if FlexAttentionWrapper._compiled_flex_attn is None:
             raise ImportError("flex_attention is not available")
        return FlexAttentionWrapper._compiled_flex_attn(
            q, k, v, score_mod=score_mod, block_mask=block_mask, scale=scale, enable_gqa=enable_gqa, return_lse=return_lse,
        )
class ScaledDotProductAttentionWrapper(nn.Module):
    sdpa_backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
    def forward(self, q, k, v, *, scale=None, enable_gqa=False, is_causal=True):
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa)
def get_causal_mask_mod():
    return lambda b, h, q_idx, kv_idx: q_idx >= kv_idx
_compiled_create_block_mask = create_block_mask
def create_attention_mask(*args, **kwargs):
    if _compiled_create_block_mask is None:
         raise ImportError("create_block_mask is not available")
    return _compiled_create_block_mask(*args, **kwargs)
def create_varlen_metadata_from_sequence_lengths(sequence_lengths, seq_len, device):
    batch_size = len(sequence_lengths)
    cu_seqlens_list, all_seq_lengths, offset = [], [], 0
    for b in range(batch_size):
        sample_seq_lens = sequence_lengths[b]
        sample_cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32, device=device), torch.cumsum(sample_seq_lens.to(torch.int32), dim=0)])
        all_seq_lengths.append(sample_seq_lens)
        cu_seqlens_list.append((sample_cu_seqlens[:-1] + offset).to(torch.int32))
        offset += seq_len
    packed_cu_seqlens = torch.cat(cu_seqlens_list + [torch.tensor([offset], dtype=torch.int32, device=device)]).to(torch.int32)
    max_seqlen = torch.cat(all_seq_lengths).max().item() if all_seq_lengths else 0
    return VarlenMetadata(cu_seq_q=packed_cu_seqlens, cu_seq_k=packed_cu_seqlens, max_q=max_seqlen, max_k=max_seqlen)
