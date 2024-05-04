"""Compare the outputs of HF and vLLM for Mistral models using greedy sampling.

Run `pytest tests/models/test_mistral.py`.
"""
import pytest

from tests.models.utils import check_logprobs_close

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_long_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    # NOTE: example_long_prompts has 4096+ tokens which is bigger than
    # mistral's sliding window size.
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy_logprobs_limit(
        example_long_prompts, max_tokens, num_logprobs)
    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_outputs = vllm_model.generate_greedy_logprobs(example_long_prompts,
                                                       max_tokens,
                                                       num_logprobs)
    del vllm_model
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
