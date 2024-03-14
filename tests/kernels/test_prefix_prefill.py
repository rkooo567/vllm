from asyncio.selector_events import BaseSelectorEventLoop
import random
from termios import BSDLY
import pytest
import time

import torch
from vllm.model_executor.layers.attention.ops.prefix_prefill import (
    context_attention_fwd)
from xformers import ops as xops
from allclose_default import get_default_atol, get_default_rtol
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask, BlockDiagonalCausalMask, LowerTriangularFromBottomRightMask, BlockDiagonalCausalLocalAttentionFromBottomRightMask

NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 64]
HEAD_SIZES = [128]
DTYPES = [torch.float16, torch.half, torch.float32]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.set_default_device(device)
    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    subquery_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(subquery_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(subquery_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    k = torch.zeros(sum(subquery_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(subquery_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[:BS * max_block_per_request].view(
        BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + subquery_lens[:-1],
                                            dtype=torch.long),
                               dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1],
                                                dtype=torch.long),
                                   dim=0)
    for i in range(BS):
        for j in range(subquery_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] +
                                            j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] +
                                              b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = k_cache.view(-1, block_size, num_kv_heads, head_size // 8,
                           8).permute(0, 2, 3, 1, 4).contiguous()
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = v_cache.view(-1, block_size, num_kv_heads,
                           head_size).permute(0, 2, 3, 1).contiguous()

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    context_attention_fwd(query, k, v, output, k_cache, v_cache, block_table,
                          b_start_loc, b_seq_len, b_ctx_len, max_input_len)
    torch.cuda.synchronize()
    start_time = time.time()
    context_attention_fwd(query, k, v, output, k_cache, v_cache, block_table,
                          b_start_loc, b_seq_len, b_ctx_len, max_input_len)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")

    scale = float(1.0 / (head_size**0.5))

    attn_op = xops.fmha.cutlass.FwOp()

    if num_kv_heads != num_heads:
        # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
        # project the key and value tensors to the desired number of
        # heads.
        #
        # see also: vllm/model_executor/layers/attention.py
        query = query.view(query.shape[0], num_kv_heads, num_queries_per_kv,
                           query.shape[-1])
        key = key[:, :, None, :].expand(key.shape[0], num_kv_heads,
                                        num_queries_per_kv, key.shape[-1])
        value = value[:, :,
                      None, :].expand(value.shape[0], num_kv_heads,
                                      num_queries_per_kv, value.shape[-1])
    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)
    attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        subquery_lens, seq_lens)
    output_ref = xops.memory_efficient_attention_forward(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    )
    torch.cuda.synchronize()
    start_time = time.time()
    output_ref = xops.memory_efficient_attention_forward(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"xformers Time: {(end_time - start_time)*1000:.2f} ms")
    output_ref = output_ref.squeeze(0, 2)
    assert torch.allclose(output_ref, output, atol=get_default_atol(output), rtol=get_default_rtol(output))


def setup_test(device: str, seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)


def prepare_inputs(
        max_subquery_len, max_ctx_len, batch_size,
        num_heads, num_kv_heads, head_size, dtype):
    subquery_lens = [random.randint(0, max_subquery_len) for _ in range(batch_size)]
    ctx_lens = [random.randint(0, max_ctx_len) for _ in range(batch_size)]
    seq_lens = [a + b for a, b in zip(subquery_lens, ctx_lens)]

    num_tokens = sum(seq_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_subquery_start_loc = torch.cumsum(torch.tensor([0] + subquery_lens[:-1],
                                            dtype=torch.long),
                               dim=0)
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1],
                                                dtype=torch.long),
                                   dim=0)
    return query, key, value, subquery_lens, seq_lens, b_seq_len, b_ctx_len, b_subquery_start_loc, b_seq_start_loc


def fill_kv_cache(
        key,
        value,
        k_cache,
        v_cache,
        cache_size,
        block_size,
        num_kv_heads,
        head_size,
        batch_size,
        max_block_per_request,
        b_seq_start_loc,
        b_ctx_len):
    block_table_values = torch.arange(0, cache_size, dtype=torch.long)
    block_table_values = block_table_values[torch.randperm(cache_size)]
    block_table = block_table_values[:batch_size * max_block_per_request].view(
        batch_size, max_block_per_request)
    for i in range(batch_size):
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = k_cache.view(-1, block_size, num_kv_heads, head_size // 8,
                           8).permute(0, 2, 3, 1, 4).contiguous()
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = v_cache.view(-1, block_size, num_kv_heads,
                           head_size).permute(0, 2, 3, 1).contiguous()
    return k_cache, v_cache, block_table


def xformer_attention(
        query, key, value, attn_bias, scale,
        num_kv_heads, num_heads, num_queries_per_kv):
    # Run xformer attention.
    attn_op = xops.fmha.cutlass.FwOp()
    if num_kv_heads != num_heads:
        # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
        # project the key and value tensors to the desired number of
        # heads.
        #
        # see also: vllm/model_executor/layers/attention.py
        query = query.view(query.shape[0], num_kv_heads, num_queries_per_kv,
                           query.shape[-1])
        key = key[:, :, None, :].expand(key.shape[0], num_kv_heads,
                                        num_queries_per_kv, key.shape[-1])
        value = value[:, :,
                      None, :].expand(value.shape[0], num_kv_heads,
                                      num_queries_per_kv, value.shape[-1])
    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)
    output_ref = xops.memory_efficient_attention_forward(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    ).squeeze(0, 2)
    return output_ref


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("device", ["cuda"])
@torch.inference_mode()
def test_contexted_kv_attention_xformer(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    setup_test("cuda:0", seed=0)

    # Prepare metadata.
    num_kv_heads = num_heads // num_queries_per_kv
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    max_input_len = MAX_SEQ_LEN

    (query, key, value, subquery_lens, seq_lens,
     b_seq_len, b_ctx_len,
     b_subquery_start_loc, b_seq_start_loc) = prepare_inputs(
        MAX_SEQ_LEN, MAX_CTX_LEN, BS, num_heads, num_kv_heads, head_size, dtype)

    num_tokens = sum(b_seq_len)
    num_subquery_tokens = sum(subquery_lens)

    # Fill kv caches.
    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    k_cache, v_cache, block_table = fill_kv_cache(
         key, value, k_cache, v_cache, cache_size, block_size,
         num_kv_heads, head_size, BS, max_block_per_request,
         b_seq_start_loc, b_ctx_len)

    # Copy the subquery's key value to k and v.
    k = torch.zeros(sum(subquery_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(subquery_lens), num_kv_heads, head_size, dtype=dtype)
    for i in range(BS):
        for j in range(subquery_lens[i]):
            k[b_subquery_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] +
                                            j])
            v[b_subquery_start_loc[i] + j].copy_(value[b_seq_start_loc[i] +
                                              b_ctx_len[i] + j])

    # context attn subquery vs xformer subquery.
    context_subquery_output = torch.empty(num_subquery_tokens, num_heads, head_size, dtype=dtype)
    subquery = query[:num_subquery_tokens]
    context_attention_fwd(subquery, k, v, context_subquery_output, k_cache, v_cache, block_table,
                          b_subquery_start_loc, b_seq_len, b_ctx_len, max_input_len)
    torch.cuda.synchronize()

    # Full query
    attn_bias = BlockDiagonalCausalMask.from_seqlens(seq_lens)
    scale = float(1.0 / (head_size**0.5))
    output_ref = xformer_attention(query, key, value, attn_bias, scale,
        num_kv_heads, num_heads, num_queries_per_kv)
    torch.cuda.synchronize()

    atol = get_default_atol(context_subquery_output)
    rtol = get_default_rtol(context_subquery_output)

    # Attention with subquery.
    attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(subquery_lens, seq_lens)
    scale = float(1.0 / (head_size**0.5))
    xformer_subquery_output = xformer_attention(subquery, key, value, attn_bias, scale,
        num_kv_heads, num_heads, num_queries_per_kv)
    torch.cuda.synchronize()

    # # Context attention subquery vs xformer subquery.
    # assert torch.allclose(context_subquery_output, xformer_subquery_output, atol=atol, rtol=rtol)
    # # Xformer query vs xformer subquery
    # assert torch.allclose(output_ref[-num_subquery_tokens:], xformer_subquery_output, atol=atol, rtol=rtol)
    # # Xformer subquery vs context subquery
    # assert torch.allclose(output_ref[-num_subquery_tokens:], context_subquery_output, atol=atol, rtol=rtol)

    # Test multi batches
    output_ref_truncated = output_ref
    context_subquery_output_truncated = context_subquery_output
    for i in range(BS):
        subquery_len = subquery_lens[i]
        seqlen = seq_lens[i]

        output_ref_truncated = output_ref_truncated[seqlen - subquery_len:]
        assert torch.allclose(output_ref_truncated[:subquery_len], context_subquery_output_truncated[:subquery_len], atol=atol, rtol=rtol)
        output_ref_truncated = output_ref_truncated[subquery_len:]
        context_subquery_output_truncated = context_subquery_output_truncated[subquery_len:]

    output_ref_truncated = output_ref
    context_subquery_output_truncated = xformer_subquery_output
    for i in range(BS):
        subquery_len = subquery_lens[i]
        seqlen = seq_lens[i]

        output_ref_truncated = output_ref_truncated[seqlen - subquery_len:]
        print("Correct!")
        assert torch.allclose(output_ref_truncated[:subquery_len], context_subquery_output_truncated[:subquery_len], atol=atol, rtol=rtol)
        output_ref_truncated = output_ref_truncated[subquery_len:]
        context_subquery_output_truncated = context_subquery_output_truncated[subquery_len:]


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("device", ["cuda"])
@torch.inference_mode()
def test_contexted_kv_attention_no_kv_cache(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    MAX_SEQ_LEN = 1024
    # Meaning, kv cache is not used.
    MAX_CTX_LEN = 0
    BS = 10
    setup_test("cuda:0", seed=0)

    # Prepare metadata.
    num_kv_heads = num_heads // num_queries_per_kv
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    max_input_len = MAX_SEQ_LEN

    (query, key, value, subquery_lens, seq_lens,
     b_seq_len, b_ctx_len,
     b_subquery_start_loc, b_seq_start_loc) = prepare_inputs(
        MAX_SEQ_LEN, MAX_CTX_LEN, BS, num_heads, num_kv_heads, head_size, dtype)

    num_tokens = sum(b_seq_len)

    # Fill kv caches.
    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    k_cache, v_cache, block_table = fill_kv_cache(
         key, value, k_cache, v_cache, cache_size, block_size,
         num_kv_heads, head_size, BS, max_block_per_request,
         b_seq_start_loc, b_ctx_len)

    # Run context attention fwd full query.
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    context_attention_fwd(query, key, value, output, k_cache, v_cache, block_table,
                          b_subquery_start_loc, b_seq_len, b_ctx_len, max_input_len)
    torch.cuda.synchronize()

    # Run xformer.
    attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        subquery_lens, seq_lens)
    # attn_bias = LowerTriangularFromBottomRightMask()
    scale = float(1.0 / (head_size**0.5))
    output_ref = xformer_attention(query, key, value, attn_bias, scale,
        num_kv_heads, num_heads, num_queries_per_kv)
    torch.cuda.synchronize()

    atol = get_default_atol(output)
    rtol = get_default_rtol(output)
    assert torch.allclose(output_ref, output, atol=atol, rtol=rtol)
