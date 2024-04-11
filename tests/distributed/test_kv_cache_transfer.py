"""Test the KV cache communication operators.

Run `python test_kvcache_comm.py`.
"""
from typing import List
import torch
import ray

from vllm import EngineArgs, LLMEngine
from vllm.utils import get_open_port
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.worker import init_worker_distributed_environment
from vllm.utils import set_cuda_visible_devices
from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync, RayGPUExecutor
from vllm.engine.ray_utils import initialize_ray_cluster
from vllm.sequence import SequenceGroupMetadata, SequenceData
from vllm import SamplingParams
import math


def run_all_workers(engine: LLMEngine, method: str, *args):
    """Run all the workers."""
    executor = engine.model_executors[0]
    ray_worker_outputs = [
        worker.execute_method.remote(method, *args)
        for worker in engine.workers
    ]
    _ = getattr(engine.driver_worker, method)(*args)
    ray.get(ray_worker_outputs)


def test_cache_transfer():
    try:
        engine_args = EngineArgs("facebook/opt-125m")
        config = engine_args.create_engine_config()

        @ray.remote
        class CacheWorker:

            def __init__(self, config, rank, world_size, master_port):
                set_cuda_visible_devices([0, 1])
                torch.cuda.set_device(torch.device(f"cuda:{rank}"))
                config.cache_config.num_gpu_blocks = 64
                config.cache_config.num_cpu_blocks = 0
                config.parallel_config.worker_use_ray = True
                config.parallel_config.world_size = 2
                config.parallel_config.enable_disaggregated_prefill = True
                self.config = config
                self.cache = CacheEngine(
                    config.cache_config,
                    config.model_config,
                    config.parallel_config,
                )
                self.rank = rank
                self.world_size = world_size
                self.master_port = master_port

            def init_distributed(self):
                distributed_init_method = f"tcp://localhost:{self.master_port}"
                init_worker_distributed_environment(
                    self.config.parallel_config,
                    self.rank,
                    distributed_init_method=distributed_init_method,
                    local_rank=self.rank)

            def write(self, block_ids: List[int], val: int):
                self.cache.gpu_cache
                block_shape = self.cache._get_kv_cache_shape()[2:]
                for i in range(self.cache.num_layers):
                    k, v = self.cache._get_kv_cache_to_send_recv(i)
                    for block_id in block_ids:
                        k[block_id].copy_(
                            torch.full(block_shape, val, device="cuda"))
                        v[block_id].copy_(
                            torch.full(block_shape, val, device="cuda"))

            def read(self):
                return self.cache.gpu_cache

            def send(self, block_ids):
                self.cache.send_blocks(block_ids)

            def recv(self, block_ids):
                self.cache.recv_blocks(block_ids)

            def num_layers(self):
                return self.cache.num_layers

        distributed_init_port = get_open_port()
        cache_a = CacheWorker.remote(config, 0, 2, distributed_init_port)
        cache_b = CacheWorker.remote(config, 1, 2, distributed_init_port)
        ray.get([
            cache_a.init_distributed.remote(),
            cache_b.init_distributed.remote(),
        ])

        blocks_to_write = [3, 5, 7]
        blocks_to_read = [2, 4, 6]
        ray.get(cache_a.write.remote(blocks_to_write, 2))
        kv_cache_a = ray.get(cache_a.read.remote())
        ray.get([
            cache_a.send.remote(blocks_to_write),
            cache_b.recv.remote(blocks_to_read),
        ])
        kv_cache_b = ray.get(cache_b.read.remote())
        for i in range(ray.get(cache_a.num_layers.remote())):
            for send_block_id, recv_block_id in zip(blocks_to_write,
                                                    blocks_to_read):
                torch.allclose(kv_cache_a[i][0][send_block_id].cpu(),
                               kv_cache_b[i][0][recv_block_id].cpu())
                torch.allclose(kv_cache_a[i][1][send_block_id].cpu(),
                               kv_cache_b[i][0][recv_block_id].cpu())

    finally:
        ray.shutdown()


def test_executor_kv_cache_transfer():
    try:
        engine_args = EngineArgs("facebook/opt-125m", worker_use_ray=True)
        config = engine_args.create_engine_config()
        initialize_ray_cluster(config.parallel_config)
        executor = RayGPUExecutor(
            config.model_config,
            config.cache_config,
            config.parallel_config,
            config.scheduler_config,
            config.device_config,
            config.lora_config,
            config.vision_language_config,
            config.speculative_config,
        )
        executor.check_health()

        seq_group_metadata_list = []
        block_size = config.cache_config.block_size
        prompt_len = 30
        data = SequenceData(prompt_token_ids=[2] * prompt_len)
        num_blocks = math.ceil(len(data.prompt_token_ids) / block_size)

        seq_id = 0
        block_tables = {seq_id: list(range(num_blocks))}
        print(block_tables)
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"1",
            is_prompt=True,
            seq_data={seq_id: data},
            sampling_params=SamplingParams(temperature=0),
            block_tables=block_tables,
            token_chunk_size=len(data.prompt_token_ids),
        )
        seq_group_metadata_list.append(seq_group_metadata)
        output = executor.execute_model(seq_group_metadata_list, {}, {}, {})
        print(output)
    finally:
        ray.shutdown()
