import asyncio
import copy
import os
import pickle
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, SpeculativeConfig,
                         VisionLanguageConfig)
from vllm.engine.ray_utils import RayWorkerVllm, ray
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async, set_cuda_visible_devices)

if ray is not None:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

# If the env var is set, it uses the Ray's compiled DAG API
# which optimizes the control plane overhead.
# Run vLLM with VLLM_USE_RAY_COMPILED_DAG=1 to enable it.
USE_RAY_COMPILED_DAG = bool(os.getenv("VLLM_USE_RAY_COMPILED_DAG", 0))


class DisaggRayGpuExecutor(ExecutorBase):

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
        speculative_config: Optional[SpeculativeConfig],
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.vision_language_config = vision_language_config
        assert (not speculative_config
                ), "Speculative decoding not yet supported for RayGPU backend."
        assert self.parallel_config.enable_disaggregated_prefill

        assert self.parallel_config.worker_use_ray
        placement_group = self.parallel_config.placement_group
        # SANG-TODO parallel config should be applied per worker groups

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        ranks = list(range(parallel_config.world_size))
        if len(ranks) % 2 != 0:
            raise AssertionError("only N:N supported")
        half = len(ranks) // 2
        prefill_ranks = ranks[:half]
        decode_ranks = ranks[half:]

        prefill_workers, decode_workers = self._init_workers_ray(
            placement_group, ranks, prefill_ranks, decode_ranks)

        self.prefill_executor = RayGPUExecutor(
            prefill_ranks,
            prefill_workers,
        )
        self.decode_executor = RayGPUExecutor(
            decode_ranks,
            decode_workers,
        )
        self._run_workers("init_device")
        self._run_workers(
            "load_model",
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
        )

    def _init_workers_ray(self, placement_group: "PlacementGroup", ranks,
                          prefill_ranks, decode_ranks, **ray_remote_kwargs):
        if self.parallel_config.tensor_parallel_size == 1:
            # For single GPU case, we use a ray worker with constrained memory.
            num_gpus = self.cache_config.gpu_memory_utilization
        else:
            # Otherwise, the ray workers are allocated with a full GPU.
            num_gpus = 1

        # The remaining workers are the actual ray actors.
        workers: List[RayWorkerVllm] = []

        # Create the workers.
        driver_ip = get_ip()
        for bundle_id in ranks:
            bundle = placement_group.bundle_specs[bundle_id]
            if not bundle.get("GPU", 0):
                continue

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )
            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerVllm).remote(self.model_config.trust_remote_code)
            # Else, added to the list of workers.
            workers.append(worker)

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = ray.get(
            [worker.get_node_and_gpu_ids.remote() for worker in workers])

        # node -> list of worker ranks
        node_workers = defaultdict(list)
        node_gpus = defaultdict(list)

        for i, (node_id, gpu_ids) in zip(ranks, worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        # Set CUDA_VISIBLE_DEVICES for all workers.
        for worker, (node_id, _) in zip(workers, worker_node_and_gpu_ids):
            worker.set_cuda_visible_devices.remote(node_gpus[node_id])

        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())

        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        device_config = copy.deepcopy(self.device_config)
        lora_config = copy.deepcopy(self.lora_config)
        cache_config = copy.deepcopy(self.cache_config)
        vision_language_config = copy.deepcopy(self.vision_language_config)

        # Initialize the actual workers with the Worker class.
        for rank, (worker, (node_id,
                            _)) in zip(ranks,
                                       zip(workers, worker_node_and_gpu_ids)):
            local_rank = node_workers[node_id].index(rank)
            if rank in prefill_ranks:
                driver_worker_rank = prefill_ranks[0]
            else:
                driver_worker_rank = decode_ranks[0]
            worker.init_worker.remote(
                lambda rank=rank, local_rank=local_rank: Worker(
                    model_config=model_config,
                    parallel_config=parallel_config,
                    scheduler_config=scheduler_config,
                    device_config=device_config,
                    cache_config=cache_config,
                    local_rank=local_rank,
                    rank=rank,
                    distributed_init_method=distributed_init_method,
                    lora_config=lora_config,
                    vision_language_config=vision_language_config,
                    driver_worker_rank=driver_worker_rank,
                ))

        return (
            [workers[rank] for rank in prefill_ranks],
            [workers[rank] for rank in decode_ranks],
        )

    # Run the same method across all workers in 2 executors.
    def _run_workers(self, method, *args, **kwargs):
        refs = self.prefill_executor._run_workers_async(
            method, *args, **kwargs)
        refs.extend(
            self.decode_executor._run_workers_async(method, *args, **kwargs))
        return ray.get(refs)

    def determine_num_available_blocks(self) -> tuple[int, int]:
        num_blocks = self._run_workers("determine_num_available_blocks", )
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache in all workers.
        """

        # NOTE: We log here to avoid multiple logs when number of workers is
        # greater than one. We could log in the engine, but not all executors
        # have GPUs.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._run_workers("initialize_cache",
                          num_gpu_blocks=num_gpu_blocks,
                          num_cpu_blocks=num_cpu_blocks)

    def execute_model(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            blocks_to_swap_in: Dict[int, int],
            blocks_to_swap_out: Dict[int, int],
            blocks_to_copy: Dict[int, List[int]],
            blocks_to_send: Optional[List[int]] = None,
            blocks_to_recv: Optional[List[int]] = None) -> SamplerOutput:
        if seq_group_metadata_list[0].is_prompt:
            assert blocks_to_recv is None
            print("SANG-TODO execute prefill")
            output = self.prefill_executor.execute_model(
                seq_group_metadata_list,
                blocks_to_swap_in,
                blocks_to_swap_out,
                blocks_to_copy,
                blocks_to_send=blocks_to_send,
                blocks_to_recv=blocks_to_recv)
        else:
            assert blocks_to_send is None
            print("SANG-TODO execute decodee")
            output = self.decode_executor.execute_model(
                seq_group_metadata_list,
                blocks_to_swap_in,
                blocks_to_swap_out,
                blocks_to_copy,
                blocks_to_send=blocks_to_send,
                blocks_to_recv=blocks_to_recv)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplemented

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplemented

    def list_loras(self) -> List[int]:
        raise NotImplemented

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        self.prefill_executor.check_health()
        self.decode_executor.check_health()


class RayGPUExecutor(ExecutorBase):

    def __init__(self, ranks: List[int], workers) -> None:
        self.workers = workers
        self.driver_worker = self.workers.pop(0)
        self.ranks = ranks
        self.driver_rank = self.ranks[0]

        self.forward_dag = None
        if USE_RAY_COMPILED_DAG:
            assert False
            self.forward_dag = self._compiled_ray_dag()

    def determine_num_available_blocks(self) -> tuple[int, int]:
        """Determine the number of available KV blocks.

        This invokes `determine_num_available_blocks` on each worker and takes
        the min of the results, guaranteeing that the selected cache sizes are
        compatible with all workers.

        Returns:
            - tuple[num_gpu_blocks, num_cpu_blocks]
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers("determine_num_available_blocks", )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache in all workers.
        """

        # NOTE: We log here to avoid multiple logs when number of workers is
        # greater than one. We could log in the engine, but not all executors
        # have GPUs.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._run_workers("initialize_cache",
                          num_gpu_blocks=num_gpu_blocks,
                          num_cpu_blocks=num_cpu_blocks)

    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        blocks_to_send: Optional[List[int]],
        blocks_to_recv: Optional[List[int]],
    ) -> SamplerOutput:
        all_outputs = self._run_workers(
            "execute_model",
            driver_kwargs={
                "seq_group_metadata_list": seq_group_metadata_list,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
                "blocks_to_send": blocks_to_send,
                "blocks_to_recv": blocks_to_recv,
            },
            use_ray_compiled_dag=USE_RAY_COMPILED_DAG)

        # Only the driver worker returns the sampling results.
        output = all_outputs[0]
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "add_lora",
            lora_request=lora_request,
        )

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "remove_lora",
            lora_id=lora_id,
        )

    def list_loras(self) -> List[int]:
        return self._run_workers("list_loras")

    def _run_workers(
        self,
        method: str,
        *args,
        driver_args: Optional[List[Any]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_workers: Optional[int] = None,
        use_ray_compiled_dag: bool = False,
        **kwargs,
    ) -> List["ray.ObjectRef"]:
        return ray.get(
            self._run_workers_async(
                method,
                *args,
                driver_args=driver_args,
                driver_kwargs=driver_kwargs,
                max_concurrent_workers=max_concurrent_workers,
                use_ray_compiled_dag=use_ray_compiled_dag,
                **kwargs))

    def _run_workers_async(
        self,
        method: str,
        *args,
        driver_args: Optional[List[Any]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_workers: Optional[int] = None,
        use_ray_compiled_dag: bool = False,
        **kwargs,
    ) -> List["ray.ObjectRef"]:
        """Runs the given method on all workers."""
        assert use_ray_compiled_dag is False
        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        if use_ray_compiled_dag:
            # Right now, compiled DAG can only accept a single
            # input. TODO(sang): Fix it.
            output_channels = self.forward_dag.execute(1)
        else:
            # Start the ray workers first.
            ray_worker_outputs = [
                worker.execute_method.remote(method, *args, **kwargs)
                for worker in self.workers
            ]

        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        # Start the driver worker after all the ray workers.
        driver_worker_output = self.driver_worker.execute_method.remote(
            method, *driver_args, **driver_kwargs)
        refs = [driver_worker_output]
        refs.extend(ray_worker_outputs)
        return refs

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        self._check_if_any_actor_is_dead()

    def _check_if_any_actor_is_dead(self):
        if not self.workers:
            return

        dead_actors = []
        for actor in self.workers:
            actor_state = ray.state.actors(actor._ray_actor_id.hex())  # pylint: disable=protected-access
            if actor_state["State"] == "DEAD":
                dead_actors.append(actor)
        if dead_actors:
            raise RuntimeError("At least one Worker is dead. "
                               f"Dead Workers: {dead_actors}. ")


class RayGPUExecutorAsync(RayGPUExecutor, ExecutorAsyncBase):

    async def _run_workers_async(
        self,
        method: str,
        *args,
        driver_args: Optional[List[Any]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        coros = []

        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        # Run the driver worker asynchronously.
        driver_executor = make_async(getattr(self.driver_worker, method))
        coros.append(driver_executor(*driver_args, **driver_kwargs))

        # Run the ray workers asynchronously.
        for worker in self.workers:
            coros.append(worker.execute_method.remote(method, *args, **kwargs))

        all_outputs = await asyncio.gather(*coros)
        return all_outputs

    async def execute_model_async(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        all_outputs = await self._run_workers_async(
            "execute_model",
            driver_kwargs={
                "seq_group_metadata_list": seq_group_metadata_list,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
            })

        # Only the driver worker returns the sampling results.
        output = all_outputs[0]
        return output

    async def check_health_async(self) -> None:
        """Raises an error if engine is unhealthy."""
        self._check_if_any_actor_is_dead()
