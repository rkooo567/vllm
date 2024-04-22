from vllm import LLM, SamplingParams
import random
import argparse

parser = argparse.ArgumentParser(
    description='')
parser.add_argument('--model', type=str, default='codellama/CodeLlama-7b-hf')
parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=2)
parser.add_argument('--input-len', type=int, default=512)
parser.add_argument('--output-len', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--enforce-eager',
                    action='store_true',
                    help='enforce eager mode and disable CUDA graph')
args = parser.parse_args()


BATCH_SIZE = args.batch_size
PROMPT_LEN = args.input_len
OUTPUT_LEN = args.output_len
ENABLE_CUDA = not args.enforce_eager
TP = args.tensor_parallel_size
print(args)
print(f"\n=======\n{BATCH_SIZE=} {PROMPT_LEN=} {OUTPUT_LEN=} {ENABLE_CUDA=} {TP=}\n=======\n")

llm = LLM(
    model=args.model,
    tokenizer=None,
    trust_remote_code=True,
    dtype="half",
    swap_space=0,
    disable_log_stats=False,
    tensor_parallel_size=TP,
    # max_model_len=max_model_len,
    # block_size=block_size,
    enable_chunked_prefill=False,
    enforce_eager=not ENABLE_CUDA,
    # **kwargs,
)
import time
print("warn up")
llm.generate(
    sampling_params=SamplingParams(min_tokens=OUTPUT_LEN, max_tokens=OUTPUT_LEN),
    prompt_token_ids=[[random.randint(0, 30000) for _ in range(PROMPT_LEN)] for _ in range(BATCH_SIZE)])
prompt_token_ids = [[random.randint(0, 30000) for _ in range(PROMPT_LEN)] for _ in range(BATCH_SIZE)]
s = time.time()
llm.generate(
    sampling_params=SamplingParams(min_tokens=OUTPUT_LEN, max_tokens=OUTPUT_LEN),
    prompt_token_ids=prompt_token_ids)
print(f"e2e takes {time.time() -s}")
llm.llm_engine.model_executor.print_perf()
