from vllm import LLM, SamplingParams

BATCH_SIZE = 1024
PROMPT_LEN = 1000
OUTPUT_LEN = 1
ENABLE_CUDA = False
MAX_BATCH_PER_STEP = None

llm = LLM(
    model="codellama/CodeLlama-7b-hf",
    max_num_batched_tokens=MAX_BATCH_PER_STEP,
    tokenizer=None,
    trust_remote_code=True,
    dtype="half",
    swap_space=0,
    disable_log_stats=True,
    tensor_parallel_size=2,
    # max_model_len=max_model_len,
    # block_size=block_size,
    enable_chunked_prefill=False,
    enforce_eager=not ENABLE_CUDA,
    # **kwargs,
)
import time
s = time.time()
llm.generate(
    sampling_params=SamplingParams(min_tokens=OUTPUT_LEN, max_tokens=OUTPUT_LEN),
    prompt_token_ids=[[1 * PROMPT_LEN] for _ in range(BATCH_SIZE)])
print(f"e2e takes {time.time() -s}")
