from collections import deque
import enum
import time
from typing import Deque, Dict, Iterable, List, Optional, Tuple, Union, Set

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.block_manager import AllocStatus, BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.lora.request import LoRARequest
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)

logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: Iterable[SequenceGroup],
        num_chunked_prefill_groups: int,
        num_prompt_groups: int,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
        lora_enabled: bool = False,
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.num_chunked_prefill_groups = num_chunked_prefill_groups
        self.num_prompt_groups = num_prompt_groups
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

        if lora_enabled:
            self.num_loras = len(self.lora_requests)
            self._sort_by_lora_ids()

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self) -> bool:
        self.scheduled_seq_groups = sorted(self.scheduled_seq_groups,
                                           key=lambda g:
                                           (g.lora_int_id, g.request_id))

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {g.lora_request for g in self.scheduled_seq_groups}


class SchedulerDecodeOutputs:
    """Outputs of the decoding phase of the scheduler.

    Attributes:
        decoding_seq_groups: Selected sequence groups for decoding.
        num_preempted_seqs: The number of preempted sequences.
        blocks_to_swap_in: The blocks to swap in.
        blocks_to_swap_out: The blocks to swap out.
        blocks_to_copy: The blocks to copy.
    """

    def __init__(
        self,
        decoding_seq_groups: List[SequenceGroup],
        num_preempted_seqs: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        self.decoding_seq_groups = decoding_seq_groups
        self.num_preempted_seqs = num_preempted_seqs
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy

    @staticmethod
    def create_empty() -> "SchedulerDecodeOutputs":
        return SchedulerDecodeOutputs([], 0, [], [], [])

    def num_decoding_seqs(self):
        return sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.decoding_seq_groups)


class SchedulePrefillOutputs:
    """Outputs of the prefilling phase of the scheduler.

    Attributes:
        num_batched_tokens: The number of batched tokens.
        chunk_prefilling_seq_groups: Selected sequence groups for chunked
            prefilling.
        prompting_seq_groups: Selected sequence groups for prompting.
        ignored_seq_groups: Ignored sequence groups.
    """

    def __init__(
        self,
        chunk_prefilling_seq_groups: List[SequenceGroup],
        prompting_seq_groups: List[SequenceGroup],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.chunk_prefilling_seq_groups = chunk_prefilling_seq_groups
        self.prompting_seq_groups = prompting_seq_groups
        self.ignored_seq_groups = ignored_seq_groups

    def num_prompting_groups(self):
        return len(self.prompting_seq_groups)

    def num_chunk_prefilling_groups(self):
        return len(self.chunk_prefilling_seq_groups)

    def num_selected_groups(self):
        return len(self.chunk_prefilling_seq_groups) + len(
            self.prompting_seq_groups)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)
        self.chunked_prefill_enabled = \
            self.scheduler_config.max_chunked_prefill_len >= 0
        if self.chunked_prefill_enabled:
            self.max_chunked_prefill_len = \
                scheduler_config.max_chunked_prefill_len
            logger.info(
                f"chunked prefill enabled, {self.max_chunked_prefill_len=}"
                f", {self.scheduler_config.max_num_prompt_seqs=}"
                f", { self.scheduler_config.max_num_batched_tokens=}")
            assert not self.lora_enabled, \
                "chunked prefilling is not supported with LoRA"
        else:
            self.max_chunked_prefill_len = 1000_000_000

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the CHUNKED PREFILLING state.
        self.chunked_prefilling: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        self.swapped: Deque[SequenceGroup] = deque()

        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        logger.debug(f"add_seq_group {seq_group.request_id}")
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [
                self.waiting, self.running, self.swapped,
                self.chunked_prefilling
        ]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity .
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped or self.chunked_prefilling

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule_decoding(
            self, token_budget: int) -> Tuple[int, SchedulerDecodeOutputs]:
        """Schedule sequence groups for decoding.
        First schedule the sequence groups in the RUNNING state.
        Then schedule the sequence groups in the SWAPPED state.
        Args:
            num_batched_decoding_tokens: The number of batched decoding tokens.
            token_budget: The number of available token slots.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        decoding_seq_groups: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        num_batched_decoding_tokens = 0

        # Fix the current time.
        now = time.time()
        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Step 1: Schedule as many decoding requests as possible.
        # If we run out of token budget, stop.
        # If we run out of available slots, try to preempt
        # the lowest-priority sequence groups.
        while self.running:
            if token_budget - num_batched_decoding_tokens <= 0:
                break

            seq_group = self.running.popleft()
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop()
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                logger.debug(f"append slot for {seq_group}")
                self._append_slot(seq_group, blocks_to_copy)
                decoding_seq_groups.append(seq_group)
                logger.debug(f"scheduled r -> r {seq_group.request_id}")
                num_batched_decoding_tokens += (
                    seq_group.num_seqs(status=SequenceStatus.RUNNING) *
                    self.num_decoding_tokens_per_seq)

        # If any sequence group is preempted, do not swap in any sequence group.
        if preempted:
            return num_batched_decoding_tokens, SchedulerDecodeOutputs(
                decoding_seq_groups, len(preempted), blocks_to_swap_in,
                blocks_to_swap_out, blocks_to_copy)

        # Step 2: Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)

        num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                            for seq_group in self.running)
        curr_loras = set(
            seq_group.lora_int_id
            for seq_group in self.running) if self.lora_enabled else None

        leftover_swapped = deque()

        while self.swapped:
            seq_group = self.swapped[0]
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                if (lora_int_id > 0 and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    self.swapped.popleft()
                    continue

            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            self.swapped.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            logger.debug(f"scheduled s -> r {seq_group.request_id}")
            num_curr_seqs += num_new_seqs
            decoding_seq_groups.append(seq_group)
            num_batched_decoding_tokens += (
                seq_group.num_seqs(status=SequenceStatus.RUNNING) *
                self.num_decoding_tokens_per_seq)

        self.swapped.extendleft(leftover_swapped)

        return num_batched_decoding_tokens, SchedulerDecodeOutputs(
            decoding_seq_groups, len(preempted), blocks_to_swap_in,
            blocks_to_swap_out, blocks_to_copy)

    def _chunk_prefill_sequence_group(
            self, seq_group: SequenceGroup, token_budget: int,
            chunk_prefilling_seq_groups: List[SequenceGroup],
            prompting_seq_groups: List[SequenceGroup]) -> int:
        """Chunked prefilling one sequence_group.

            If a seq_group is a chunked prefill, chunk_prefilling_seq_groups
            is updated in-place (appended). Otherwise, prompting_seq_groups is
            updated in-place (appended).

        Args:
            seq_group: The sequence to be chunk prefilled.
            token_budget: The number of available token slots.

        Returns:
            num_tokens: The number of tokens to be prefilled from
                the sequence group.
        """
        # This API is available after https://github.com/vllm-project/vllm/pull/3538 is merged.
        num_uncomputed_tokens = seq_group.get_num_uncomputed_tokens()
        to_advance = min(num_uncomputed_tokens, token_budget,
                         self.max_chunked_prefill_len)

        # This API is available after https://github.com/vllm-project/vllm/pull/3538 is merged.
        seq_group.advance_prefill_range(to_advance)
        # If the sequence group is not fully prefilled, put it into the
        # chunked prefilling queue.
        # This API is available after https://github.com/vllm-project/vllm/pull/3538 is merged.
        if seq_group.get_num_uncomputed_tokens() > 0:
            logger.debug(f"scheduled p -> p {seq_group.request_id}")
            chunk_prefilling_seq_groups.append(seq_group)
        else:
            logger.debug(f"scheduled p -> r {seq_group.request_id}")
            prompting_seq_groups.append(seq_group)

        return to_advance

    def _schedule_prefilling(
            self, token_budget: int,
            num_curr_seqs: int) -> Tuple[int, SchedulePrefillOutputs]:
        """Schedule sequence groups for (chunked) prefilling.

        Args:
            token_budget: The number of available token slots.
            num_curr_seqs: The number of sequences already scheduled.

        Returns:
            num_batched_tokens: The number of batched prefill tokens.
            SchedulePrefillOutputs: The outputs of the prefilling phase.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        prompting_seq_groups: List[SequenceGroup] = []
        chunk_prefilling_seq_groups: List[SequenceGroup] = []
        num_batched_tokens = 0

        # If any request in swapped state, try not schedule any prefilling.
        if self.swapped:
            return 0, SchedulePrefillOutputs(chunk_prefilling_seq_groups,
                                             prompting_seq_groups,
                                             ignored_seq_groups)

        # Step 1: Continue schedule those requests are in chunked prefilling.
        # This is called only if chunked prefilling is enabled.
        while self.chunked_prefilling:
            if not self.chunked_prefill_enabled:
                raise AssertionError(
                    "can't reach here since chunk prefill is disabled")

            if token_budget - num_batched_tokens <= 0:
                break

            seq_group = self.chunked_prefilling.popleft()
            num_prefilled_tokens = self._chunk_prefill_sequence_group(
                seq_group, token_budget, chunk_prefilling_seq_groups,
                prompting_seq_groups)

            num_batched_tokens += num_prefilled_tokens
            num_curr_seqs += seq_group.get_max_num_running_seqs()

        # Step 2: Schedule the new requests (WAITING).
        # The total number of sequences on the fly, including the
        # requests in the generation phase.
        num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                            for seq_group in self.running)
        curr_loras = set(
            seq_group.lora_int_id
            for seq_group in self.running) if self.lora_enabled else None

        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        leftover_waiting_sequences = deque()
        while self._passed_delay(time.time()) and self.waiting:
            seq_group = self.waiting[0]

            if token_budget - num_batched_tokens <= 0:
                leftover_waiting_sequences.appendleft(seq_group)
                self.waiting.popleft()
                break

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")

            num_prompt_tokens = waiting_seqs[0].get_len()
            if num_prompt_tokens > self.prompt_limit:
                logger.warning(
                    f"Input prompt ({num_prompt_tokens} tokens) is too long"
                    f" and exceeds limit of {self.prompt_limit}")
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    f"Input prompt ({num_prompt_tokens} tokens) is too long"
                    f" and exceeds the capacity of block_manager")
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                if (curr_loras is not None and self.lora_config is not None
                        and lora_int_id > 0 and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    self.waiting.popleft()
                    continue

            if num_batched_tokens + num_prompt_tokens > token_budget:
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            self.waiting.popleft()
            self._allocate(seq_group)

            num_prefilled_tokens = self._chunk_prefill_sequence_group(
                seq_group, token_budget, chunk_prefilling_seq_groups,
                prompting_seq_groups)

            num_batched_tokens += num_prefilled_tokens
            num_curr_seqs += num_new_seqs

        self.waiting.extendleft(leftover_waiting_sequences)
        if len(prompting_seq_groups) > 0:
            self.prev_prompt = True

        return num_batched_tokens, SchedulePrefillOutputs(
            prompting_seq_groups, ignored_seq_groups)

    def _schedule(self) -> SchedulerOutputs:
        token_budget = self.scheduler_config.max_num_batched_tokens

        if self.chunked_prefill_enabled:
            # Chunked prefilling is enabled.
            # We first schedule as many decoding requests as possible,
            # and then schedule chunked prefilling requests.
            num_batched_decoding_tokens, decoding_outputs = self._schedule_decoding(
                token_budget)

            num_batched_prefill_tokens, prefilling_outputs = self._schedule_prefilling(
                token_budget, decoding_outputs.num_decoding_seqs())
        else:
            # First schedule as many prefilling requests as possible,
            # then schedule decoding requests.
            num_curr_seqs = sum(
                seq_group.num_seqs(status=SequenceStatus.RUNNING)
                for seq_group in self.running)
            (num_batched_prefill_tokens,
             prefilling_outputs) = self._schedule_prefilling(
                 token_budget, num_curr_seqs)
            token_budget -= num_batched_prefill_tokens

            if len(prefilling_outputs.prompting_seq_groups) > 0:
                decoding_outputs = SchedulerDecodeOutputs.create_empty()
                num_batched_decoding_tokens = 0
            else:
                (num_batched_decoding_tokens,
                 decoding_outputs) = self._schedule_decoding(token_budget)

        num_batched_tokens = (num_batched_prefill_tokens +
                              num_batched_decoding_tokens)
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=prefilling_outputs.chunk_prefilling_seq_groups
            + prefilling_outputs.prompting_seq_groups +
            decoding_outputs.decoding_seq_groups,
            num_chunked_prefill_groups=prefilling_outputs.
            num_chunk_prefilling_groups(),
            num_prompt_groups=prefilling_outputs.num_selected_groups(),
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=decoding_outputs.blocks_to_swap_in,
            blocks_to_swap_out=decoding_outputs.blocks_to_swap_out,
            blocks_to_copy=decoding_outputs.blocks_to_copy,
            ignored_seq_groups=prefilling_outputs.ignored_seq_groups,
            lora_enabled=self.lora_enabled,
        )

        self.chunked_prefilling.extend(
            prefilling_outputs.chunk_prefilling_seq_groups)
        self.running.extend(prefilling_outputs.prompting_seq_groups)
        self.running.extend(decoding_outputs.decoding_seq_groups)
        return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()
        now = time.time()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, seq_group in enumerate(scheduler_outputs.scheduled_seq_groups):
            seq_group.maybe_set_first_scheduled_time(now)

            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            # It assumes the scheduled_seq_groups is ordered by
            # chunked prefill < prefill < decoding.
            is_prompt = i < scheduler_outputs.num_prompt_groups
            is_chunked_prefill = \
                i < scheduler_outputs.num_chunked_prefill_groups

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                is_chunked_prefill=is_chunked_prefill,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                lora_request=seq_group.lora_request,
                computed_block_nums=self.block_manager.
                get_common_computed_block_ids(seq_group),
                state=seq_group.state,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = deque(seq_group for seq_group in self.running
                             if not seq_group.is_finished())

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.appendleft(seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        self.block_manager.mark_blocks_as_computed(seq_group)

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay
