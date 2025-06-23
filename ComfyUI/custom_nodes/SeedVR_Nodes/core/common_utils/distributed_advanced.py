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
Advanced distributed functions for sequence parallel.
Simplified for ComfyUI.
"""

from typing import Optional, List
import torch
import torch.distributed as dist
# from torch.distributed.device_mesh import DeviceMesh, init_device_mesh # Commented out
# from torch.distributed.fsdp import ShardingStrategy # Commented out

# Adjusted imports to use placeholders from distributed_ops.py
from .distributed_ops import get_sequence_parallel_rank as get_global_rank
from .distributed_ops import get_sequence_parallel_world_size as get_world_size


_DATA_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_CPU_GROUP = None
_MODEL_SHARD_CPU_INTER_GROUP = None
_MODEL_SHARD_CPU_INTRA_GROUP = None
_MODEL_SHARD_INTER_GROUP = None
_MODEL_SHARD_INTRA_GROUP = None
_SEQUENCE_PARALLEL_GLOBAL_RANKS = None


def get_data_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get data parallel process group.
    """
    # Simplified: In ComfyUI, likely no separate data parallel group from sequence parallel group
    return _DATA_PARALLEL_GROUP if _DATA_PARALLEL_GROUP is not None else get_sequence_parallel_group()


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get sequence parallel process group.
    """
    # This might be initialized by some ComfyUI distributed setup if available, otherwise None
    if dist.is_available() and dist.is_initialized() and _SEQUENCE_PARALLEL_GROUP is None:
        # Fallback to WORLD if no specific SP group is set up by this module's init
        # return dist.group.WORLD # This might be too broad.
        # For now, let it be None unless initialized by init_sequence_parallel
        pass
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_cpu_group() -> Optional[dist.ProcessGroup]:
    """
    Get sequence parallel CPU process group.
    """
    return _SEQUENCE_PARALLEL_CPU_GROUP


def get_data_parallel_rank() -> int:
    """
    Get data parallel rank.
    """
    # Simplified: Assumes data parallel is either full world or same as SP if SP is active.
    # If SP is not active (world_size=1), global_rank is 0.
    # If SP is active, this should be rank within the DP group.
    # For ComfyUI, usually a single process, so 0.
    if _DATA_PARALLEL_GROUP:
        return dist.get_rank(_DATA_PARALLEL_GROUP)
    return 0 # Default to 0 if no group


def get_data_parallel_world_size() -> int:
    """
    Get data parallel world size.
    """
    if _DATA_PARALLEL_GROUP:
        return dist.get_world_size(_DATA_PARALLEL_GROUP)
    return 1 # Default to 1


def get_sequence_parallel_rank() -> int:
    """
    Get sequence parallel rank.
    """
    group = get_sequence_parallel_group()
    # return dist.get_rank(group) if group else 0 # Original line
    if group and dist.is_initialized():
        try:
            return dist.get_rank(group)
        except RuntimeError: # Group not in this process
            return 0
    return 0


def get_sequence_parallel_world_size() -> int:
    """
    Get sequence parallel world size.
    Modified to simply return 1 for ComfyUI.
    """
    return 1
    # group = get_sequence_parallel_group()
    # return dist.get_world_size(group) if group else 1 # Original logic


def get_model_shard_cpu_intra_group() -> Optional[dist.ProcessGroup]:
    return _MODEL_SHARD_CPU_INTRA_GROUP


def get_model_shard_cpu_inter_group() -> Optional[dist.ProcessGroup]:
    return _MODEL_SHARD_CPU_INTER_GROUP


def get_model_shard_intra_group() -> Optional[dist.ProcessGroup]:
    return _MODEL_SHARD_INTRA_GROUP


def get_model_shard_inter_group() -> Optional[dist.ProcessGroup]:
    return _MODEL_SHARD_INTER_GROUP


def init_sequence_parallel(sequence_parallel_size: int):
    """
    Initialize sequence parallel.
    Simplified for ComfyUI: This level of setup is unlikely.
    """
    global _DATA_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_CPU_GROUP
    global _SEQUENCE_PARALLEL_GLOBAL_RANKS

    if not (dist.is_available() and dist.is_initialized()):
        # print("[distributed_advanced] Distributed not initialized. Skipping init_sequence_parallel.")
        return

    if sequence_parallel_size <= 1:
        # print("[distributed_advanced] Sequence parallel size <= 1. Skipping init_sequence_parallel.")
        _SEQUENCE_PARALLEL_GROUP = None # Explicitly None
        _DATA_PARALLEL_GROUP = None # Or WORLD if DDP is used without SP. For now, None.
        _SEQUENCE_PARALLEL_GLOBAL_RANKS = [get_global_rank()] if dist.is_initialized() else [0]
        return

    # The following original logic is for multi-GPU distributed setups.
    # It's highly unlikely to be applicable directly in ComfyUI without major infra.
    # For now, we'll assume if ComfyUI ever runs this, it's in a context where
    # groups might already exist or this init is a no-op.
    # Keeping the logic commented out for reference.

    # world_size = dist.get_world_size()
    # rank = dist.get_rank()
    # data_parallel_size = world_size // sequence_parallel_size
    # if world_size % sequence_parallel_size != 0:
    #     print(f"[distributed_advanced] WARNING: world_size ({world_size}) not divisible by sequence_parallel_size ({sequence_parallel_size}). SP may not work correctly.")
    #     return

    # for i in range(data_parallel_size):
    #     start_rank = i * sequence_parallel_size
    #     end_rank = (i + 1) * sequence_parallel_size
    #     ranks = list(range(start_rank, end_rank)) # Ensure it's a list
    #     # Check if ranks are valid for new_group
    #     if not all(r < world_size for r in ranks):
    #         print(f"[distributed_advanced] Invalid ranks for new_group: {ranks}. Skipping SP group creation.")
    #         continue

    #     # Attempt to create groups. This can fail if groups overlap or ranks are bad.
    #     try:
    #         group = dist.new_group(ranks=ranks)
    #         cpu_group = dist.new_group(ranks=ranks, backend="gloo") # Gloo for CPU group
    #         if rank in ranks:
    #             _SEQUENCE_PARALLEL_GROUP = group
    #             _SEQUENCE_PARALLEL_CPU_GROUP = cpu_group
    #             _SEQUENCE_PARALLEL_GLOBAL_RANKS = list(ranks)
    #             print(f"[distributed_advanced] Rank {rank} part of SP group {ranks}")
    #     except RuntimeError as e:
    #         print(f"[distributed_advanced] Error creating SP group for ranks {ranks}: {e}")
    #         # If a group for these ranks already exists, new_group might fail.
    #         # This needs robust handling in a real distributed app.
    #         # For ComfyUI, assume this init path is less critical.
    #         pass

    # Fallback if SP group not set (e.g. if single process or error)
    if _SEQUENCE_PARALLEL_GROUP is None and dist.is_initialized():
         _SEQUENCE_PARALLEL_GLOBAL_RANKS = [dist.get_rank()]
    elif not dist.is_initialized():
         _SEQUENCE_PARALLEL_GLOBAL_RANKS = [0]


def init_model_shard_group(
    *,
    sharding_strategy = None, # ShardingStrategy type hinted, but simplified
    device_mesh = None, # DeviceMesh type hinted, but simplified
):
    """
    Initialize process group of model sharding.
    Simplified for ComfyUI.
    """
    global _MODEL_SHARD_INTER_GROUP
    global _MODEL_SHARD_INTRA_GROUP
    global _MODEL_SHARD_CPU_INTER_GROUP
    global _MODEL_SHARD_CPU_INTRA_GROUP

    # print("[distributed_advanced] init_model_shard_group called but is a no-op in this simplified version for ComfyUI.")
    # This function is highly dependent on torch.distributed.device_mesh and FSDP ShardingStrategy,
    # which are complex distributed training features. For ComfyUI inference, these are typically not used.
    # Setting groups to None.
    _MODEL_SHARD_INTER_GROUP = None
    _MODEL_SHARD_INTRA_GROUP = None
    _MODEL_SHARD_CPU_INTER_GROUP = None
    _MODEL_SHARD_CPU_INTRA_GROUP = None


def get_sequence_parallel_global_ranks() -> List[int]:
    """
    Get all global ranks of the sequence parallel process group
    that the caller rank belongs to.
    """
    if _SEQUENCE_PARALLEL_GLOBAL_RANKS is None:
        # Initialize with current rank if not set by init_sequence_parallel
        # This implies no sequence parallelism if init wasn't called or was ineffective.
        if dist.is_available() and dist.is_initialized():
            return [dist.get_rank()]
        else:
            return [0] # Default for non-distributed context
    return _SEQUENCE_PARALLEL_GLOBAL_RANKS


def get_next_sequence_parallel_rank() -> int:
    """
    Get the next global rank of the sequence parallel process group
    that the caller rank belongs to.
    Simplified for ComfyUI (assumes SP world size is 1).
    """
    # sp_global_ranks = get_sequence_parallel_global_ranks()
    # sp_rank = get_sequence_parallel_rank() # This is local rank in SP group
    # sp_size = get_sequence_parallel_world_size() # This is now 1
    # # Original logic: return sp_global_ranks[(sp_rank + 1) % sp_size]
    # Since sp_size is 1, (sp_rank + 1) % 1 is always 0.
    # This would return sp_global_ranks[0], which is the current rank's global rank.
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_prev_sequence_parallel_rank() -> int:
    """
    Get the previous global rank of the sequence parallel process group
    that the caller rank belongs to.
    Simplified for ComfyUI (assumes SP world size is 1).
    """
    # sp_global_ranks = get_sequence_parallel_global_ranks()
    # sp_rank = get_sequence_parallel_rank() # Local rank in SP group
    # sp_size = get_sequence_parallel_world_size() # This is now 1
    # # Original logic: return sp_global_ranks[(sp_rank + sp_size - 1) % sp_size]
    # Since sp_size is 1, (sp_rank + 1 - 1) % 1 is always 0.
    # Returns sp_global_ranks[0], current rank's global rank.
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0
