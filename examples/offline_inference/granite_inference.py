# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Inference script for IBM Granite 3.3-8B model on Spyre.

Note: Granite model has limited support on Spyre compared to micro models.
Increase parameters carefully and test thoroughly.
"""

import os

# Environment variables must be set BEFORE importing vLLM
os.environ["VLLM_PLUGINS"] = "spyre_inference"
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1800")

import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        default="ibm-granite/granite-3.3-8b-instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=128,
        dest="max_model_len",
        help="Maximum model context length",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=2,
        dest="max_num_seqs",
        help="Maximum batch size (sequences in flight)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=64,
        dest="max_num_batched_tokens",
        help="Maximum tokens processed per batch",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=2,
        help="Number of prompts to generate",
    )
    parser.add_argument(
        "--max-tokens",
        type=str,
        default="32",
        dest="max_tokens",
        help="Comma-separated max tokens per prompt (cycled if shorter than num_prompts)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        dest="enforce_eager",
        help="Skip torch.compile, run in eager mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs

    prompts = [
        "What is machine learning?",
        "Explain artificial intelligence in one sentence.",
        "Define neural networks.",
        "What is deep learning?",
    ]
    prompts = prompts[: args.num_prompts]

    max_tokens = [int(v) for v in args.max_tokens.split(",")]
    max_tokens = max_tokens * (args.num_prompts // len(max_tokens) + 1)
    max_tokens = max_tokens[: args.num_prompts]

    print(f"Initializing LLM with model: {args.model}")
    print(f"  max_model_len: {args.max_model_len}")
    print(f"  max_num_seqs: {args.max_num_seqs}")
    print(f"  max_num_batched_tokens: {args.max_num_batched_tokens}")
    print(f"  num_prompts: {args.num_prompts}")
    print(f"  max_tokens: {max_tokens}")

    # Use EngineArgs for proper initialization (matches vllm bench approach)
    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        dtype="float16",
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
        enable_prefix_caching=False,
    )
    llm = LLM.from_engine_args(engine_args)

    # Sampling params: temperature=1.0, top_p=1.0 (matches vllm bench defaults)
    sampling_params = [
        SamplingParams(
            max_tokens=m,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
        )
        for m in max_tokens
    ]

    print("\n=============== GENERATE")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(f"Generated {total_tokens} tokens in {elapsed:.2f} sec")
    print("===============\n")

    for i, output in enumerate(outputs):
        print(f"Prompt {i}: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}\n")


if __name__ == "__main__":
    main()
