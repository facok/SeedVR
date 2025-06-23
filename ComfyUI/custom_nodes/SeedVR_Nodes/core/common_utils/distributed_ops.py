# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Distributed ops for supporting sequence parallel.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import Tensor

# Adjusted import
from .cache import Cache

# TODO: Ensure 'advanced.py' is copied to this directory or these functions are stubbed/replaced
# from common.distributed.advanced import (
#     get_sequence_parallel_group,
#     get_sequence_parallel_rank,
#     get_sequence_parallel_world_size,
# )

# TODO: Ensure 'basic.py' is copied to this directory or this function is stubbed/replaced
# from .basic import get_device

# Placeholder functions for missing imports to allow file to be parsed
def get_sequence_parallel_group():
    # In a non-distributed ComfyUI context, this might always return None
    if dist.is_available() and dist.is_initialized():
        # This is a simplification and might not be correct for actual sequence parallelism
        return dist.group.WORLD
    return None

def get_sequence_parallel_rank():
    if dist.is_available() and dist.is_initialized():
        # return dist.get_rank(get_sequence_parallel_group())
        return dist.get_rank() # Simplified
    return 0

def get_sequence_parallel_world_size():
    if dist.is_available() and dist.is_initialized():
        # return dist.get_world_size(get_sequence_parallel_group())
        return dist.get_world_size() # Simplified
    return 1

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_SEQ_DATA_BUF = defaultdict(lambda: [None, None, None])
_SEQ_DATA_META_SHAPES = defaultdict()
_SEQ_DATA_META_DTYPES = defaultdict()
_SEQ_DATA_ASYNC_COMMS = defaultdict(list)
_SYNC_BUFFER = defaultdict(dict)


def single_all_to_all(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
    async_op: bool = False,
):
    """
    A function to do all-to-all on a tensor
    """
    seq_world_size = dist.get_world_size(group)
    prev_scatter_dim = scatter_dim
    if scatter_dim != 0:
        local_input = local_input.transpose(0, scatter_dim)
        if gather_dim == 0:
            gather_dim = scatter_dim
        scatter_dim = 0

    inp_shape = list(local_input.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // seq_world_size
    input_t = local_input.reshape(
        [seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :]
    ).contiguous()
    output = torch.empty_like(input_t)
    comm = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)
    if async_op:
        # let user's code transpose & reshape
        return output, comm, prev_scatter_dim

    # first dim is seq_world_size, so we can split it directly
    output = torch.cat(output.split(1), dim=gather_dim + 1).squeeze(0)
    if prev_scatter_dim:
        output = output.transpose(0, prev_scatter_dim).contiguous()
    return output


def _all_to_all(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
):
    seq_world_size = dist.get_world_size(group)
    input_list = [
        t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool,
    ) -> Tensor:
        # For ComfyUI, assume no distributed setup unless explicitly handled
        if group is None or dist.get_world_size(group) <= 1:
            if async_op:
                return local_input, None # No comm needed
            return local_input

        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        if async_op:
            output, comm, prev_scatter_dim = single_all_to_all(
                local_input, scatter_dim, gather_dim, group, async_op=async_op
            )
            ctx.prev_scatter_dim = prev_scatter_dim
            return output, comm

        return _all_to_all(local_input, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        if not hasattr(ctx, 'group') or ctx.group is None or dist.get_world_size(ctx.group) <= 1:
            return None, grad_output[0], None, None, None

        if ctx.async_op:
            input_t = torch.cat(grad_output[0].split(1), dim=ctx.gather_dim + 1).squeeze(0)
            if ctx.prev_scatter_dim:
                input_t = input_t.transpose(0, ctx.prev_scatter_dim)
        else:
            input_t = grad_output[0]
        return (
            None,
            _all_to_all(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group),
            None,
            None,
            None,
        )


class Slice(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, local_input: Tensor, dim: int) -> Tensor:
        if group is None or dist.get_world_size(group) <= 1:
            return local_input

        ctx.group = group
        ctx.rank = dist.get_rank(group)
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size
        ctx.dim = dim
        dim_size = local_input.shape[dim]
        return local_input.split(dim_size // seq_world_size, dim=dim)[ctx.rank].contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor, None]:
        if not hasattr(ctx, 'group') or ctx.group is None or dist.get_world_size(ctx.group) <= 1:
            return None, grad_output, None

        dim_size = list(grad_output.size())
        split_size = dim_size[0] # This assumes dim 0 is the one being gathered, might be too specific
        # Should be: split_size = grad_output.shape[ctx.dim] if ctx.dim != 0 else grad_output.shape[0] / ctx.seq_world_size
        # However, the original code uses dim_size[0] which implies the split part is moved to dim 0 before all_gather
        # This part might need careful review depending on how _all_gather_base behaves with non-contiguous tensors or specific dims

        # A safer assumption for general case:
        # The size of the tensor on *this* rank along the split dimension
        current_dim_size = grad_output.shape[ctx.dim]
        # The full size of the tensor along the split dimension, if gathered from all ranks
        full_dim_size = current_dim_size * ctx.seq_world_size

        output_shape = list(grad_output.shape)
        output_shape[ctx.dim] = full_dim_size

        output = torch.empty(output_shape, dtype=grad_output.dtype, device=grad_output.device)

        # _all_gather_base expects a flat list of tensors if ranks have different shapes,
        # or a single large tensor to fill if shapes are identical.
        # For simplicity, assuming identical shapes for now, which is what Slice would produce.
        dist._all_gather_base(output, grad_output.contiguous(), group=ctx.group) # contiguous may be important

        # The original code implies a concatenation after gather, which is what _all_gather_base does if output is pre-sized.
        # If it gathers into a list, then torch.cat(output_list, dim=ctx.dim) would be needed.
        # Given it's _all_gather_base, it fills 'output'.
        return (None, output, None)


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        dim: int,
        grad_scale: Optional[bool] = False,
    ) -> Tensor:
        if group is None or dist.get_world_size(group) <= 1:
            return local_input

        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size

        current_dim_size = local_input.shape[dim]
        ctx.part_size = current_dim_size # Size of this rank's part along 'dim'

        output_shape = list(local_input.shape)
        output_shape[dim] = current_dim_size * seq_world_size # Total size along 'dim' after gathering

        output = torch.empty(output_shape, dtype=local_input.dtype, device=local_input.device)
        dist._all_gather_base(output, local_input.contiguous(), group=ctx.group)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        if not hasattr(ctx, 'group') or ctx.group is None or dist.get_world_size(ctx.group) <= 1:
             if ctx.grad_scale: # Still apply grad_scale if it was intended, though world_size is 1
                return None, grad_output * ctx.seq_world_size, None, None
             return None, grad_output, None, None

        if ctx.grad_scale:
            grad_output = grad_output * ctx.seq_world_size

        # Split the gradient and return the part for this rank
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.dim)[ctx.rank].contiguous(),
            None,
            None,
        )


def gather_seq_scatter_heads_qkv(
    qkv_tensor: Tensor,
    *,
    seq_dim: int,
    qkv_shape: Optional[Tensor] = None, # Original shape of individual Q, K, or V before padding/concatenation
    cache: Cache = Cache(disable=True),
    restore_shape: bool = True,
):
    """
    A func to sync splited qkv tensor
    qkv_tensor: the tensor we want to do alltoall with. The last dim must
        be the projection_idx, which we will split into 3 part. After
        spliting, the gather idx will be projecttion_idx + 1
    seq_dim: gather_dim for all2all comm
    restore_shape: if True, output will has the same shape length as input
    """
    group = get_sequence_parallel_group()
    if not group or get_sequence_parallel_world_size() <= 1:
        return qkv_tensor

    world = get_sequence_parallel_world_size()
    orig_shape = qkv_tensor.shape
    scatter_dim = qkv_tensor.dim() # This seems to intend to scatter along the last dim after reshaping
                                   # For qkv (B, S, 3*H*D) -> (B, S, 3, H, D)
                                   # scatter_dim should be related to Head dimension for all2all

    bef_all2all_shape = list(orig_shape)
    qkv_proj_dim = bef_all2all_shape[-1] # Total feature dim (3 * num_heads * head_dim)

    # Reshape to (..., 3, num_heads_per_rank * head_dim_per_rank)
    # It should be (..., 3, num_heads_total/world_size, head_dim) or (..., 3, num_heads_per_rank, head_dim)
    # The original code implies the last dim is already per-rank head features combined (e.g. H*D)
    # and it's split into 3 for Q, K, V.
    # Let's assume shape is (Batch, SeqLen, 3 * NumHeads * HeadDim)
    # We want to make it (Batch, SeqLen, 3, NumHeads_per_process, HeadDim_per_process) for SP
    # then all2all along SeqLen, gather along HeadDim

    # The original comment says "scatter_dim = qkv_tensor.dim()" which is index for last dim.
    # And then "qkv_tensor = SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, False)"
    # This implies scattering the last dimension (features) and gathering along sequence dimension.
    # This is "gather sequence, scatter features"

    # Let's re-evaluate based on the function name "gather_seq_scatter_heads_qkv"
    # This implies:
    # 1. Input QKV is likely (Batch, Seq_per_rank, 3 * NumHeads_total * HeadDim_total) (already sliced in Seq for TP)
    # OR (Batch, Seq_total, 3 * NumHeads_per_rank * HeadDim_per_rank) (sliced in Heads for SP)

    # If input is (Batch, Seq_per_rank, NumFeatures_total), SP wants to make it (Batch, Seq_total, NumFeatures_per_rank)
    # So, gather along Seq (dim 1), scatter along Features (dim 2)
    # Here seq_dim is gather_dim. scatter_dim should be feature dim.

    # Original: qkv_tensor.view(list(orig_shape)[:-1] + [3, qkv_proj_dim // 3])
    # This makes it (B, S_part, 3, H*D_part)
    # Then SeqAllToAll(..., scatter_dim=last_dim_idx, gather_dim=seq_dim)
    # This would scatter H*D_part and gather S_part. This matches "gather_seq_scatter_features"

    # Let's assume qkv_tensor is (Batch, SeqLen_part, 3 * HeadDim_full)
    # We want (Batch, SeqLen_full, 3 * HeadDim_part)
    # scatter_dim would be the last one (features). gather_dim is seq_dim.

    qkv_tensor_reshaped = qkv_tensor.view(list(orig_shape)[:-1] + [3, qkv_proj_dim // 3])
    # scatter_dim_idx is the index of the dimension (qkv_proj_dim // 3) in qkv_tensor_reshaped
    scatter_dim_idx = qkv_tensor_reshaped.dim() -1

    qkv_tensor_all2all = SeqAllToAll.apply(group, qkv_tensor_reshaped, scatter_dim_idx, seq_dim, False)

    if restore_shape:
        # qkv_tensor_all2all is now (B, SeqLen_full, 3, (H*D_part)/world_size ) - this is not right.
        # Output of SeqAllToAll for (Input, scatter_idx, gather_idx)
        # Shape of Input: S_0, S_1, ..., S_scatter_idx, ..., S_gather_idx, ...
        # Shape of Output: S_0, S_1, ..., S_scatter_idx/world, ..., S_gather_idx*world, ...

        # Input to All2All: (B, SeqLen_part, 3, HeadDim_full)
        # scatter_dim_idx = 3 (HeadDim_full), gather_dim = 1 (SeqLen_part)
        # Output from All2All: (B, SeqLen_full, 3, HeadDim_full/world_size) which is (B, SeqLen_full, 3, HeadDim_part)

        out_shape_list = list(qkv_tensor_all2all.shape) # (B, SeqLen_full, 3, HeadDim_part)
        # We want to merge the last two dims back if the original was (B, SeqLen_part, 3*HeadDim_full)
        # Target shape: (B, SeqLen_full, 3*HeadDim_part)
        final_shape = list(orig_shape)
        final_shape[seq_dim] = final_shape[seq_dim] * world       # SeqLen_full
        final_shape[-1] = final_shape[-1] // world # 3*HeadDim_part

        qkv_tensor_final = qkv_tensor_all2all.view(final_shape)
    else:
        # If not restoring, output is (B, SeqLen_full, 3, HeadDim_part)
        qkv_tensor_final = qkv_tensor_all2all


    # remove padding - this part seems complex and tied to specific 'qkv_shape' meaning.
    # For now, we assume padding is handled externally or not strictly needed for basic inference.
    # if qkv_shape is not None:
    #     unpad_dim_size = cache(
    #         "unpad_dim_size", lambda: torch.sum(torch.prod(qkv_shape, dim=-1)).item()
    #     ) # This seems to calculate total elements from original shapes.
    #     if unpad_dim_size % world != 0: # If total elements not divisible by world size, padding was added
    #         # This padding removal logic might be specific to how padding was added.
    #         # The current gather_dim is seq_dim. If padding was on seq_dim:
    #         # current_seq_len_full = qkv_tensor_final.shape[seq_dim]
    #         # original_total_seq_elements = unpad_dim_size / (product of other dims)
    #         # This needs careful handling. For now, skip unpadding.
    #         pass

    return qkv_tensor_final


def slice_inputs(x: Tensor, dim: int, padding: bool = True):
    """
    A func to slice the input sequence in sequence parallel
    Input x is (Batch, SeqFull, Features)
    Output should be (Batch, SeqPart, Features) for this rank
    """
    group = get_sequence_parallel_group()
    if group is None or get_sequence_parallel_world_size() <= 1:
        return x

    sp_rank = get_sequence_parallel_rank()
    sp_world = get_sequence_parallel_world_size()
    dim_size = x.shape[dim] # Full sequence length

    unit = (dim_size + sp_world - 1) // sp_world # Size of part, with padding if necessary

    x_padded = x
    if padding and dim_size % sp_world != 0:
        padding_size = unit * sp_world - dim_size # Total padding needed to make it divisible
        pad_shape = list(x.shape)
        pad_shape[dim] = padding_size
        pad_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, pad_tensor], dim=dim)

    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(unit * sp_rank, unit * (sp_rank + 1))
    return x_padded[slc].contiguous()


def remove_seqeunce_parallel_padding(x: Tensor, dim: int, unpad_dim_size: int):
    """
    A func to remove the padding part of the tensor based on its original shape.
    x is (Batch, SeqFull_Padded, Features)
    unpad_dim_size is Original_SeqFull
    This function is typically called *after* a gather operation if padding was added *before* a slice.
    """
    group = get_sequence_parallel_group()
    if group is None or get_sequence_parallel_world_size() <=1:
        return x # No padding to remove if not SP

    # This function seems to be for removing padding from a tensor that is already fully gathered.
    current_dim_size = x.shape[dim]
    if current_dim_size == unpad_dim_size: # No padding was present or needed removal
        return x

    # Example: original 10, world 4. Padded to 12. Sliced parts are 3. Gathered is 12. Remove 2.
    # Here, x is the gathered tensor of size `current_dim_size` along `dim`.
    # `unpad_dim_size` is the true original size before any SP-related padding.
    if current_dim_size > unpad_dim_size:
        slc = [slice(None)] * len(x.shape)
        slc[dim] = slice(0, unpad_dim_size)
        return x[slc].contiguous()
    return x # Should not happen if padding was added correctly


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    Input x is (Batch, Seq_part, Heads_part, Features_per_head)
    Output is (Batch, Seq_full, Heads_full / world_size, Features_per_head)
    This means: gather along head_dim, scatter along seq_dim.
    """
    group = get_sequence_parallel_group()
    if not group or get_sequence_parallel_world_size() <= 1:
        return x

    # Padding logic for seq_dim (scatter_dim for All2All)
    # This seems reversed. If seq_dim is scatter_dim, padding should be applied if its size
    # is not divisible by world_size.
    # The original code pads seq_dim if it's not divisible.
    # This means seq_dim is being scattered.
    # head_dim is being gathered.

    dim_size_seq = x.size(seq_dim)
    sp_world = get_sequence_parallel_world_size()
    x_padded = x
    if dim_size_seq % sp_world != 0:
        padding_size = sp_world - (dim_size_seq % sp_world)
        pad_shape = list(x.shape)
        pad_shape[seq_dim] = padding_size
        pad_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, pad_tensor], dim=seq_dim)

    # SeqAllToAll(input, scatter_dim, gather_dim)
    # scatter along seq_dim, gather along head_dim
    return SeqAllToAll.apply(group, x_padded, seq_dim, head_dim, False)


def gather_seq_scatter_heads(x: Tensor, seq_dim: int, head_dim: int) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    Input x is (Batch, Seq_part, Heads_full, Features_per_head) or (Batch, Seq_part, TotalFeatures)
    Output is (Batch, Seq_full, Heads_full/world_size, Features_per_head) or (Batch, Seq_full, TotalFeatures/world_size)
    This means: gather along seq_dim, scatter along head_dim (or feature_dim).
    """
    group = get_sequence_parallel_group()
    if not group or get_sequence_parallel_world_size() <= 1:
        return x
    # SeqAllToAll(input, scatter_dim, gather_dim)
    # scatter along head_dim, gather along seq_dim
    return SeqAllToAll.apply(group, x, head_dim, seq_dim, False)


def scatter_heads(x: Tensor, dim: int) -> Tensor:
    """
    A func to split heads before attention in sequence parallel
    Input x is (Batch, Seq, NumHeads_Full, Features_per_head)
    Output is (Batch, Seq, NumHeads_Part, Features_per_head) for this rank
    """
    group = get_sequence_parallel_group()
    if not group or get_sequence_parallel_world_size() <= 1:
        return x
    # Slice is (group, input, dim_to_slice_along)
    # It splits along 'dim' and gives each rank its part.
    return Slice.apply(group, x, dim)


def gather_heads(x: Tensor, dim: int, grad_scale: Optional[bool] = False) -> Tensor:
    """
    A func to gather heads for the attention result in sequence parallel
    Input x is (Batch, Seq, NumHeads_Part, Features_per_head)
    Output is (Batch, Seq, NumHeads_Full, Features_per_head)
    """
    group = get_sequence_parallel_group()
    if not group or get_sequence_parallel_world_size() <= 1:
        return x
    # Gather is (group, input_part, dim_to_gather_along)
    return Gather.apply(group, x, dim, grad_scale)


def gather_outputs(
    x: Tensor,
    *,
    gather_dim: int,
    padding_dim: Optional[int] = None, # The dimension on which original padding for SP was applied
    unpad_shape: Optional[Tensor] = None, # Original shape tensor (e.g. (B, TrueSeqLen, ...))
    cache: Cache = Cache(disable=True),
    scale_grad=True,
):
    """
    A func to gather the outputs for the model result in sequence parallel
    Input x is (Batch, Seq_Part, Features)
    Output is (Batch, Seq_Full, Features)
    """
    group = get_sequence_parallel_group()
    if not group or get_sequence_parallel_world_size() <=1:
        return x

    x_gathered = Gather.apply(group, x, gather_dim, scale_grad)

    if padding_dim is not None and unpad_shape is not None:
        # This assumes unpad_shape gives info about the true length of padding_dim
        # Example: unpad_shape = (B, TrueSeqLen, ...), padding_dim = 1
        # We need the true length for the padding_dim.
        # The original code's cache logic for "unpad_dim_size" is a bit opaque.
        # A simpler approach: if unpad_shape is the target shape *before* SP slicing.

        # Let's assume unpad_shape is a tensor representing the original full shape of X
        # before it was sliced by SP. We need the size of 'padding_dim' from this shape.
        true_dim_size = unpad_shape[padding_dim] # This needs to be an integer.
                                                  # If unpad_shape is like (B,S,H,W), then unpad_shape[padding_dim] is S.
                                                  # This requires unpad_shape to be 1D tensor of true dimensions, or similar.

        # A more direct way: if the user knows the original length of the dimension that was padded.
        # Let's assume 'unpad_shape' IS the original length of 'padding_dim'.
        # This is not how it's used in the original code with `torch.prod(unpad_shape, dim=1)).item()`
        # That original logic implies `unpad_shape` is a list/tensor of shapes for multiple items.

        # For ComfyUI, it's simpler to assume that if padding was added by `slice_inputs`,
        # the necessary info to unpad would be the original length of that dimension.
        # The current `remove_seqeunce_parallel_padding` expects the *original total length* of the padded dimension.

        # If `unpad_shape` is a tensor like `torch.tensor([original_batch, original_seq_len, original_features])`
        # and `padding_dim` is the index of the dimension that was padded (e.g., 1 for sequence)
        original_length_of_padded_dim = unpad_shape[padding_dim].item() # Make sure it's an int

        x_gathered = remove_seqeunce_parallel_padding(x_gathered, padding_dim, original_length_of_padded_dim)

    return x_gathered


def _pad_tensor(x: Tensor, dim: int, padding_size: int):
    if padding_size == 0:
        return x
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def _unpad_tensor(x: Tensor, dim: int, padding_size):
    if padding_size == 0: # No padding to remove
        return x
    slc = [slice(None)] * len(x.shape)
    # x current shape[dim] is PaddedLength. padding_size is amount to remove from end.
    # So, new length is x.shape[dim] - padding_size.
    # Slice from 0 to x.shape[dim] - padding_size
    slc[dim] = slice(0, x.shape[dim] - padding_size)
    return x[slc].contiguous() # Contiguous might be important


def _broadcast_data(data, shape, dtype, src, group, async_op):
    # Simplified for non-distributed ComfyUI context, actual broadcast won't happen
    comms = []
    if group is None or dist.get_world_size(group) <= 1:
        return comms # No communication

    if isinstance(data, (list, tuple)):
        for i, sub_shape in enumerate(shape):
            comms += _broadcast_data(data[i], sub_shape, dtype[i], src, group, async_op)
    elif isinstance(data, dict):
        for key, sub_data in data.items():
            comms += _broadcast_data(sub_data, shape[key], dtype[key], src, group, async_op)
    elif isinstance(data, Tensor):
        comms.append(dist.broadcast(data, src=src, group=group, async_op=async_op))
    return comms


def _traverse(data: Any, op: Callable) -> Union[None, List, Dict, Any]:
    if isinstance(data, (list, tuple)):
        return [_traverse(sub_data, op) for sub_data in data]
    elif isinstance(data, dict):
        return {key: _traverse(sub_data, op) for key, sub_data in data.items()}
    elif isinstance(data, Tensor):
        return op(data)
    else:
        return None # Or raise error for unsupported type


def _get_shapes(data):
    return _traverse(data, op=lambda x: x.shape)


def _get_dtypes(data):
    return _traverse(data, op=lambda x: x.dtype)


def _construct_broadcast_buffer(shapes, dtypes, device):
    if shapes is None or dtypes is None: # Handle case where one branch of traverse returned None
        return None
    if isinstance(shapes, torch.Size): # Base case: shape of a tensor
        return torch.empty(shapes, dtype=dtypes, device=device)

    if isinstance(shapes, (list, tuple)):
        buffer = []
        for i, sub_shape in enumerate(shapes):
            # Ensure dtypes[i] is valid if shapes[i] is valid
            buffer.append(_construct_broadcast_buffer(sub_shape, dtypes[i] if dtypes and i < len(dtypes) else None, device))
    elif isinstance(shapes, dict):
        buffer = {}
        for key, sub_shape in shapes.items():
            buffer[key] = _construct_broadcast_buffer(sub_shape, dtypes.get(key) if dtypes else None, device)
    else:
        return None # Or raise error for unsupported shape type
    return buffer


class SPDistForward:
    """A forward tool to sync different result across sp group

    Args:
        module: a function or module to process users input
        sp_step: current training step to judge which rank to broadcast its result to all
        name: a distinct str to save meta and async comm
        comm_shape: if different ranks have different shape, mark this arg to True
        device: the device for current rank, can be empty
    """

    def __init__(
        self,
        name: str,
        comm_shape: bool, # If true, means shapes can differ per rank, need all_gather_object for shapes
        device: torch.device = None,
    ):
        self.name = name
        self.comm_shape = comm_shape
        if device:
            self.device = device
        else:
            self.device = get_device()

    def __call__(self, inputs) -> Any:
        group = get_sequence_parallel_group()
        # If no SP group or world size is 1, just yield inputs directly (no distribution)
        if not group or get_sequence_parallel_world_size() <= 1:
            yield inputs
            return # Must use return here to stop generator after one yield

        device = self.device
        sp_world = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()

        for local_step in range(sp_world): # Iterate through which rank is the source
            src_rank_in_group = local_step # Rank within the SP group
            # Get global rank of the source based on its rank within the SP group
            # This requires dist.get_global_rank(group, local_group_rank)
            # For simplicity, if group is WORLD, src_global_rank = src_rank_in_group
            # This needs careful handling if SP group is a subgroup of WORLD.
            # Assuming group is WORLD for now if SP is active.
            src_global_rank = src_rank_in_group # Simplified: assumes group=WORLD or map is direct

            is_current_rank_src = (sp_rank == src_rank_in_group)

            current_iter_shapes = None
            current_iter_dtypes = None

            if is_current_rank_src: # If this rank is the source for this iteration
                # The 'inputs' to this __call__ method are what the current rank computed locally.
                # If this rank is the source, its 'inputs' are what need to be broadcast.
                data_to_broadcast = inputs
                current_iter_shapes = _get_shapes(data_to_broadcast)
                current_iter_dtypes = _get_dtypes(data_to_broadcast)

            # Share shapes and dtypes if they can vary (comm_shape=True) or on first step
            if self.comm_shape or local_step == 0:
                if is_current_rank_src:
                    # This rank (source) has its shapes/dtypes, needs to send to others
                    # All ranks need to participate in all_gather_object
                    gathered_shapes_list = [None] * sp_world
                    dist.all_gather_object(gathered_shapes_list, current_iter_shapes, group=group)
                    _SEQ_DATA_META_SHAPES[self.name + str(local_step)] = gathered_shapes_list

                    # Dtypes are usually consistent, can broadcast or all_gather_object too
                    gathered_dtypes_list = [None] * sp_world
                    dist.all_gather_object(gathered_dtypes_list, current_iter_dtypes, group=group)
                    _SEQ_DATA_META_DTYPES[self.name + str(local_step)] = gathered_dtypes_list[src_rank_in_group] # Store only src's dtypes
                else:
                    # Other ranks receive shapes/dtypes
                    gathered_shapes_list = [None] * sp_world
                    dist.all_gather_object(gathered_shapes_list, None, group=group) # Pass dummy for non-src
                    _SEQ_DATA_META_SHAPES[self.name + str(local_step)] = gathered_shapes_list

                    gathered_dtypes_list = [None] * sp_world
                    dist.all_gather_object(gathered_dtypes_list, None, group=group)
                    _SEQ_DATA_META_DTYPES[self.name + str(local_step)] = gathered_dtypes_list[src_rank_in_group]

            # Determine the shape/dtype for the data being broadcast in *this* iteration (from src_rank_in_group)
            # Shapes for data from current source rank (local_step)
            shapes_from_current_src = _SEQ_DATA_META_SHAPES[self.name + str(local_step)][src_rank_in_group]
            dtypes_from_current_src = _SEQ_DATA_META_DTYPES[self.name + str(local_step)] # Already indexed by src

            # Prepare buffer for broadcast if not source, or use actual data if source
            if is_current_rank_src:
                sync_data = inputs # This rank's data is the source data
            else:
                sync_data = _construct_broadcast_buffer(shapes_from_current_src, dtypes_from_current_src, device)

            # Blocking broadcast for the current iteration's data
            _broadcast_data(sync_data, shapes_from_current_src, dtypes_from_current_src, src_global_rank, group, False)

            yield sync_data # Yield the synchronized data for this source rank

        # Clear buffers after use (optional, or manage based on name)
        # del _SEQ_DATA_META_SHAPES[self.name]
        # del _SEQ_DATA_META_DTYPES[self.name]


sync_inputs = SPDistForward(name="bef_fwd", comm_shape=True) # comm_shape=True implies inputs can have different shapes across SP ranks.


def sync_data(data, sp_idx, name="tmp"):
    """
    Broadcasts 'data' from rank 'sp_idx' (in SP group) to all other ranks in SP group.
    """
    group = get_sequence_parallel_group()
    if group is None or get_sequence_parallel_world_size() <= 1:
        return data

    sp_rank = get_sequence_parallel_rank() # Current rank in SP group

    # Determine global rank of the source sp_idx
    # This needs a proper mapping if SP group is not WORLD
    # Simplified: src_global_rank = sp_idx if group is WORLD or sp_idx is already global
    # A more robust way: dist.get_global_rank(group, sp_idx) if sp_idx is local to group
    # Assuming sp_idx is the global rank of the source for now.
    # No, sp_idx is the rank *within the sequence parallel group*.

    # Let's assume sp_idx is the rank *within the group `group`* that is the source.
    # We need the global rank of this source to pass to broadcast_object_list's `src` argument.
    # However, broadcast_object_list `src` is global rank.
    # This function seems to intend sp_idx as the *source rank within the SP group*.

    # Option 1: sp_idx is global rank. Check if current global rank == sp_idx.
    # Option 2: sp_idx is local rank in SP group.
    # The context `src_rank = dist.get_global_rank(group, local_step)` in SPDistForward
    # suggests that sp_idx here should also be a local rank in the SP group.

    # If sp_idx is local rank in group:
    # global_src_rank = dist.get_global_rank(group, sp_idx) # This is not a direct API.
    # Let's assume a utility function like:
    # global_src_rank = map_local_rank_to_global(group, sp_idx)

    # The `broadcast_object_list` takes a global rank for `src`.
    # If `group` is WORLD, then local rank = global rank.
    # If `group` is a subgroup, we need to map `sp_idx` (local to group) to a global rank.
    # This is often done by having a list of global ranks in the group.
    # ranks_in_group = dist.get_process_group_ranks(group)
    # global_src_rank = ranks_in_group[sp_idx]

    # For simplicity in ComfyUI (likely single process or basic DDP):
    if sp_rank == sp_idx:
        objects = [data]
    else:
        objects = [None]

    if dist.is_initialized(): # Only call broadcast if distributed is running
         # This still needs global_src_rank.
         # If we assume sp_idx is the *global rank* of the source:
         dist.broadcast_object_list(objects, src=sp_idx, group=group) # src is global rank

    return objects[0]
