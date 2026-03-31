from collections.abc import Sequence
from typing import Any, cast, Optional
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
try:
    from torch.distributed.tensor.experimental._attention import (
        _context_parallel_shard, _ContextParallel, _enable_context_parallel_dispatcher, _HeadTailLoadBalancer, _PTRRLoadBalancer,
    )
except ImportError:
    _context_parallel_shard = lambda **kwargs: kwargs.get("buffers", None)
    class _ContextParallel:
        class AttentionType:
            FLEX = "flex"
            SDPA = "sdpa"
        def __init__(self, *args, **kwargs): pass
    _enable_context_parallel_dispatcher = lambda: None
    _HeadTailLoadBalancer = _PTRRLoadBalancer = lambda *args, **kwargs: None
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention.flex_attention import BlockMask
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.tools.logging import logger
def apply_cp_to_attention_module(attention_modules, cp_mesh, attention_type):
    match attention_type:
        case "flex": cp_plan = _ContextParallel(seq_dim=2, attention_type=_ContextParallel.AttentionType.FLEX)
        case "sdpa":
            _enable_context_parallel_dispatcher()
            cp_plan = _ContextParallel(seq_dim=2, attention_type=_ContextParallel.AttentionType.SDPA)
        case "varlen": raise NotImplementedError("Variable-length attention CP is not yet supported")
        case _: raise ValueError(f"Invalid attention_type {attention_type}")
    for attention_module in attention_modules:
        parallelize_module(module=attention_module, device_mesh=cp_mesh, parallelize_plan=cp_plan)
    logger.info("Applied Context Parallel to the model")
def prepare_context_parallel_input(inputs, labels, extra_kwargs, cp_mesh, device, load_balancer_type="headtail"):
    attention_masks = extra_kwargs.get("attention_masks", None)
    positions = extra_kwargs.get("positions", None)
    if positions is None:
        positions = torch.arange(0, inputs.shape[1], dtype=torch.int32, device=device).expand(inputs.shape)
    (inputs, labels, positions), attention_masks = cp_shard(cp_mesh, (inputs, labels, positions), attention_masks, load_balancer_type)
    extra_kwargs["positions"] = positions
    if attention_masks is not None: extra_kwargs["attention_masks"] = attention_masks
    return inputs, labels, extra_kwargs
def cp_shard(cp_mesh, inputs, attention_masks, load_balancer_type="headtail", input_seq_dim=1):
    seq_len = inputs[0].size(input_seq_dim)
    cp_world_size = cp_mesh.size(0)
    load_balancer = None
    if load_balancer_type:
        match load_balancer_type:
            case "headtail": load_balancer = _HeadTailLoadBalancer(seq_len, cp_world_size, cp_mesh.device_type)
            case "ptrr": load_balancer = _PTRRLoadBalancer(attention_masks, cp_world_size)
    res = _context_parallel_shard(mesh=cp_mesh, buffers=inputs, seq_dims=tuple(input_seq_dim for _ in inputs), load_balancer=load_balancer)
    inputs = cast(tuple[torch.Tensor, ...], res)
    MASK_Q_SEQ_DIM = 2
    if attention_masks is not None:
        masks = [attention_masks] if isinstance(attention_masks, BlockMask) else list(attention_masks.values())
        masks = _context_parallel_shard(mesh=cp_mesh, buffers=masks, seq_dims=(MASK_Q_SEQ_DIM,) * len(masks), load_balancer=load_balancer)
        attention_masks = cast((BlockMask | dict[str, BlockMask]), masks[0] if isinstance(attention_masks, BlockMask) else {k: v for k, v in zip(attention_masks.keys(), masks)})
    return inputs, attention_masks
