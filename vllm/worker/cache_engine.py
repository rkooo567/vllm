"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available
from vllm.distributed.parallel_state import (
    get_stage_model_parallel_next_rank, get_stage_model_parallel_prev_rank)

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(model_config.dtype)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, "cuda")
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _get_kv_cache_shape(self):
        return self.attn_backend.get_kv_cache_shape(self.num_gpu_blocks,
                                                    self.block_size,
                                                    self.num_heads,
                                                    self.head_size)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total

    def _get_kv_cache_to_send_recv(
            self, layer_idx) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int]]:
        key_cache = self.gpu_cache[layer_idx][0]
        value_cache = self.gpu_cache[layer_idx][1]
        return key_cache, value_cache

    def send_blocks(self, block_ids: List[int]) -> None:
        # torch futures.
        reqs = []
        keys: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        for block_id in block_ids:
            for i in range(self.num_layers):
                key_cache, value_cache = self._get_kv_cache_to_send_recv(i)
                keys.append(key_cache[block_id])
                values.append(value_cache[block_id])

        key_tensor = torch.stack(tuple(keys))
        value_tensor = torch.stack(tuple(values))
        reqs.append(
            torch.distributed.isend(key_tensor,
                                    dst=get_stage_model_parallel_next_rank()))
        reqs.append(
            torch.distributed.isend(value_tensor,
                                    dst=get_stage_model_parallel_next_rank()))
        return reqs

    def recv_blocks(self, block_ids: List[int]) -> None:
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            self.num_gpu_blocks, self.block_size, self.num_heads,
            self.head_size)
        reqs = []
        # SANG-TODO hacky fix it.
        key_tensor: torch.Tensor = torch.empty(
            size=(self.num_layers * len(block_ids), *kv_cache_shape[2:]),
            dtype=self.dtype,
            device=self.gpu_cache[0].device,
        )
        value_tensor: torch.Tensor = torch.empty(
            size=(self.num_layers * len(block_ids), *kv_cache_shape[2:]),
            dtype=self.dtype,
            device=self.gpu_cache[0].device,
        )

        reqs.append(
            torch.distributed.irecv(key_tensor,
                                    src=get_stage_model_parallel_prev_rank()))
        reqs.append(
            torch.distributed.irecv(value_tensor,
                                    src=get_stage_model_parallel_prev_rank()))

        for req in reqs:
            req.wait()

        offset = 0
        for block_id in block_ids:
            for i in range(self.num_layers):
                self.gpu_cache[i][0][block_id].copy_(key_tensor[offset])
                self.gpu_cache[i][1][block_id].copy_(value_tensor[offset])
                offset += 1

                # print(f"{self.gpu_cache[i][0][block_id]=}")
                # print(f"{self.gpu_cache[i][1][block_id]=}")


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
