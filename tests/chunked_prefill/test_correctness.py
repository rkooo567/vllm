import gc

import pytest
import torch

from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

MODELS = [
    "JackFram/llama-68m",
    # "facebook/opt-125m",
]

TEST_PROMPTS = [
    # pylint: disable=line-too-long
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
    "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    # Different between page attention and flash attention.
    # "Describe the basic components of a neural network and how it can be trained.",
    "Write a short story about a robot that dreams for the first time.",
    "Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.",
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.",
    "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches the worm.'",
]


# TODO(sang): Add chunked prefill parameters.
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("max_chunked_prefill_len", [16])
@pytest.mark.parametrize("max_num_prompt_seqs", [1, 2, 100])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_models(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
    max_tokens: int,
    max_chunked_prefill_len: int,
    max_num_prompt_seqs: int,
    block_size: int,
    tensor_parallel_size: int,
) -> None:
    """ verify the flash attention has the same output
    as page attention """
    if torch.cuda.device_count() < tensor_parallel_size:
        pytest.skip(
            f"{torch.cuda.device_count()=} is smaller than {tensor_parallel_size=}"
        )
    print("loading page attention models..")
    pg_model = vllm_runner(model, dtype=dtype)
    expected_outputs = []

    print("generating tokens...")
    expected_outputs.extend(pg_model.generate_greedy(TEST_PROMPTS, max_tokens))
    print("generating tokens finished")

    del pg_model

    for i in range(len(TEST_PROMPTS)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = expected_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")

    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()

    flash_attn_output_by_batches = []
    flash_attn_model = vllm_runner(model,
                                   dtype=dtype,
                                   block_size=block_size,
                                   flash_style=True,
                                   tensor_parallel_size=tensor_parallel_size)
    for i in range(10):
        prompts = [TEST_PROMPTS[j % len(TEST_PROMPTS)] for j in range(i)]
        flash_attn_output_by_batches.append(
            flash_attn_model.generate_greedy(prompts, max_tokens))

    del flash_attn_model
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()

    for flash_attn_outputs in flash_attn_output_by_batches:
        for i in range(len(flash_attn_outputs)):
            fa_output_ids, fa_output_str = flash_attn_outputs[i]
            vllm_output_ids, vllm_output_str = expected_outputs[
                i % len(expected_outputs)]
            print(vllm_output_str)
            assert fa_output_ids == vllm_output_ids, (
                f"Test{i}:\flash ids: {fa_output_ids}\nvLLM ids: {vllm_output_ids}"
                f"Test{i}:\nflash output: {fa_output_str!r}\nvLLM output: {vllm_output_str!r}"
            )
