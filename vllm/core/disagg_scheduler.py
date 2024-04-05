import copy

from typing import Optional, Union, Iterable

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceGroup, SequenceStatus)

logger = init_logger(__name__)


class DisaggScheduleOutputs:

    def __init__(
        self,
        prefill_schedule=None,
        decode_schedule=None,
        send_blocks=None,
        recv_blocks=None,
    ):
        self.prefill_schedule = prefill_schedule or (
            [], SchedulerOutputs.create_empty())
        self.decode_schedule = decode_schedule or (
            [], SchedulerOutputs.create_empty())
        self.send_blocks = send_blocks or []
        self.recv_blocks = recv_blocks or []

    @property
    def has_prefill(self):
        return self.prefill_schedule[0] or self.send_blocks

    @property
    def has_decode(self):
        return self.decode_schedule[0] or self.recv_blocks


class DisaggScheduler:
    """A Scheduler for prefill-decode disaggregation.

    It runs to two schedulers under the hood, with one for prefilling and
    another for decoding. Both schedulers manage the same number of
    workers each.

    Prefill scheduler can only do prefill, no decoding is allowed.
    The finished prefill requests will added into decode scheduler
    for prefill and decoding.

    Decode scheduler do both prefill and decode. However, the
    prefill will be treated as block transfer instead.

    There can be only 1 concurrent prefill and decoding at any time possible.

    The caller should trigger scheduling when a new prefill/decoding requests
    are finished.
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.prefill_scheduler = Scheduler(scheduler_config=scheduler_config,
                                           cache_config=cache_config,
                                           lora_config=lora_config)

        self.decode_scheduler = Scheduler(scheduler_config=scheduler_config,
                                          cache_config=cache_config,
                                          lora_config=lora_config)

        # scheduling state
        self.prefilling = False
        self.decoding = False

        # track ongoing prefill requests
        self.ongoing_prefill_requests = []
        self.ongoing_prefill_requests_meta = []

        # track kv_cache_blocks in the prefill workers
        self.prefill_request_blocks = {}

        # track requests scheduled for transfer
        self.transfering_request_ids = []

    @property
    def lora_enabled(self):
        return self.prefill_scheduler.lora_enabled

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        return self.prefill_scheduler.num_decoding_tokens_per_seq

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        self.prefill_scheduler.add_seq_group(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> int:
        """Returns the number of actually aborted seq groups."""
        return self.prefill_scheduler.abort_seq_group(request_id) + \
            self.decode_scheduler.abort_seq_group(request_id)

    def has_unfinished_seqs(self) -> bool:
        return self.prefill_scheduler.has_unfinished_seqs() or \
            self.decode_scheduler.has_unfinished_seqs()

    def get_num_unfinished_seq_groups(self) -> int:
        return self.prefill_scheduler.get_num_unfinished_seq_groups() + \
            self.decode_scheduler.get_num_unfinished_seq_groups()

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        assert False, "not implemented"

    def free_seq(self, seq: Sequence) -> None:
        self.prefill_scheduler.free_seq(seq)
        self.decode_scheduler.free_seq(seq)

    def free_finished_seq_groups(self) -> None:
        self.prefill_scheduler.free_finished_seq_groups()
        self.decode_scheduler.free_finished_seq_groups()

    def _schedule_prefill(self):
        assert not self.prefilling, "prefilling in progress"
        meta, output = self.prefill_scheduler.schedule(enable_decode=False)
        if meta:
            self.prefilling = True
        self.ongoing_prefill_requests = output.scheduled_seq_groups
        self.ongoing_prefill_requests_meta = meta
        return meta, output

    def _duplicate_seq_group(self, seq_group: SequenceGroup) -> SequenceGroup:
        seq_group = copy.copy(seq_group)
        seq_group.seqs_dict = copy.deepcopy(seq_group.seqs_dict)
        for seq in seq_group.seqs_dict.values():
            seq.status = SequenceStatus.WAITING
        return seq_group

    def on_prefill_finish(self):
        assert self.prefilling, "prefilling not scheduled"
        self.prefilling = False
        finished_request_ids = set()
        for seq_group in self.ongoing_prefill_requests:
            if seq_group.is_finished():
                finished_request_ids.add(seq_group.request_id)
            else:
                self.decode_scheduler.add_seq_group(
                    self._duplicate_seq_group(seq_group))
        for meta in self.ongoing_prefill_requests_meta:
            if meta.request_id in finished_request_ids:
                continue
            self.prefill_request_blocks[meta.request_id] = meta.block_tables

        self.ongoing_prefill_requests = []
        self.ongoing_prefill_requests_meta = []

    def _schedule_decode(self, transfer_new_blocks: bool):
        assert not self.decoding, "decoding in progress"

        # first scheduling decoding
        decoding_meta_list, decoding_output = self.decode_scheduler.schedule(
            enable_prefill=False,
            enable_decode=True)

        if decoding_meta_list:
            self.decoding = True

        if not transfer_new_blocks:
            return decoding_meta_list, decoding_output, [], []

        # if we have new blocks to transfer, schedule prefill
        # for new blocks transfer
        transfer_meta_list, transfer_output = self.decode_scheduler.schedule(
            enable_prefill=True,
            enable_decode=False)

        if transfer_meta_list:
            self.decoding = True

        send_blocks = []
        recv_blocks = []

        for meta in transfer_meta_list:
            prefill_blocks = self.prefill_request_blocks[meta.request_id]
            decode_blocks = meta.block_tables
            for seq_id, _send in prefill_blocks.items():
                # recv_blocks might contain one more block due to
                # the new token generated after prefill
                _recv = decode_blocks[seq_id][:len(_send)]
                send_blocks.extend(_send)
                recv_blocks.extend(_recv)

        for seq_group in transfer_output.scheduled_seq_groups:
            # assume 1 to 1 mapping between seq_group and seq
            if seq_group.get_num_unprefilled() > 0:
                self.transfering_request_ids.\
                    append(seq_group.request_id)

        return decoding_meta_list, decoding_output, send_blocks, recv_blocks

    def has_pending_transfer(self):
        return len(self.decode_scheduler.waiting) > 0

    def on_decode_finish(self):
        assert self.decoding, "decoding not scheduled"
        self.decoding = False
        if self.transfering_request_ids:
            self.prefill_scheduler.abort_seq_group(
                self.transfering_request_ids)
        self.transfering_request_ids = None

    def schedule(self):
        if self.decoding and self.prefilling:
            return DisaggScheduleOutputs()

        if self.decoding and not self.prefilling:
            if self.has_pending_transfer():
                # delay prefill if there are transfers pending.
                return DisaggScheduleOutputs()

            return DisaggScheduleOutputs(
                prefill_schedule=self._schedule_prefill())

        if self.prefilling and not self.decoding:
            meta_list, output, send_blocks, recv_blocks = self._schedule_decode(
                transfer_new_blocks=False)
            return DisaggScheduleOutputs(decode_schedule=(meta_list, output),
                                         send_blocks=send_blocks,
                                         recv_blocks=recv_blocks)

        meta_list, output, send_blocks, recv_blocks = self._schedule_decode(
            transfer_new_blocks=self.has_pending_transfer())
        prefill_meta_list, prefill_output = self._schedule_prefill()
        if send_blocks:
            self.prefilling = True
            self.decoding = True
        return DisaggScheduleOutputs(
            prefill_schedule=(prefill_meta_list, prefill_output),
            decode_schedule=(meta_list, output),
            send_blocks=send_blocks,
            recv_blocks=recv_blocks,
        )