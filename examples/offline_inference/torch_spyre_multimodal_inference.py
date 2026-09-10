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
Offline inference on Spyre through the vision path: image+text (ChartQA) prompts
against a vision tower + text decoder. Supports Pixtral/Ministral (Mistral-format
config) and Gemma 4 (HF-format config, e.g. google/gemma-4-26B-A4B).

See torch_spyre_inference.py for the text-only equivalent.

By default this runs with torch.compile (STOCK_TORCH_COMPILE).
Use --enforce-eager to skip torch.compile and run in eager mode.
"""

import os

# Environment variables must be set BEFORE importing vLLM
# (if not already in environment or to correct other env variables)
os.environ["VLLM_PLUGINS"] = "spyre_inference"

import argparse
import multiprocessing as mp
import platform
import time


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="mistralai/Ministral-3-14B-Instruct-2512-BF16")
    parser.add_argument("--max-model-len", type=int, default=2048, dest="max_model_len")
    parser.add_argument("--max-num-seqs", type=int, default=2, dest="max_num_seqs")
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=512, dest="max_num_batched_tokens"
    )
    parser.add_argument(
        "--num-gpu-blocks-override", type=int, default=None, dest="num_gpu_blocks_override"
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("-n", "--num-prompts", type=int, default=3, dest="num_prompts")
    parser.add_argument(
        "--max-tokens",
        type=str,
        default="450",
        dest="max_tokens",
        help="Comma-separated list of max tokens; the largest is used for the batch",
    )
    parser.add_argument(
        "--compare-with-cpu",
        action="store_true",
        dest="compare_with_cpu",
        help="Compare results with HuggingFace CPU inference",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        dest="enforce_eager",
        help="Skip torch.compile (whole model and attention kernel), run in eager mode",
    )
    parser.add_argument(
        "--config-format",
        type=str,
        default=None,
        dest="config_format",
        help=(
            "vLLM config_format for --model. Defaults to 'mistral' for a "
            "mistral/ministral/pixtral model name, 'hf' otherwise (explicit, not "
            "'auto': 'auto's mistral probe is a live Hub call)."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help=(
            "Leave as auto: the platform picks float16, or bfloat16 for checkpoints "
            "that overflow it"
        ),
    )
    return parser.parse_args()


def _default_config_format(model: str) -> str:
    return "mistral" if any(name in model.lower() for name in ("mistral", "pixtral")) else "hf"


_CHARTQA = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/test/png/"

# (image, question, expected answer) triples. The expected answer is printed for
# grading by eye; it never goes into the prompt.
MULTIMODAL_CASES = [
    (
        f"{_CHARTQA}1201.png",
        "What percentage of adults aged 65+ see COVID-19 as a major threat to "
        "their personal health?",
        "49%",
    ),
    (
        f"{_CHARTQA}15008.png",
        "In which year did the percentage saying there is solid evidence that "
        "the Earth is warming reach 77%?",
        "2006",
    ),
    (
        f"{_CHARTQA}41699051005347.png",
        "Which food commodity has the highest price index in the chart?",
        "Lamb",
    ),
]

# The ChartQA benchmark prompt verbatim, so offline and served runs are comparable.
# `{}` takes the question; the rest pins the answer format.
MULTIMODAL_QUESTION = (
    "{} \n Analyze the image and question "
    "carefully, using step-by-step reasoning. \n First, describe any image provided "
    "in detail. Then, present your reasoning. And finally your final answer in this "
    "format: \n Final Answer: <answer> \n where <answer> follows the following "
    "instructions: \n - <answer> should should be a single phrase or number. \n "
    "- <answer> should not paraphrase or reformat the text in the image. \n - If "
    "<answer> is a ratio, it should be a decimal value like 0.25 instead of 1:4. \n "
    "- If the question is a Yes/No question, <answer> should be Yes/No. \n - If "
    "<answer> is a number, it should not contain any units. \n - If <answer> is a "
    "percentage, it should include a % sign. \n - If <answer> is an entity, it should "
    "include the full label from the graph. \n IMPORTANT: Remember, to end your "
    "answer with Final Answer: <answer>."
)


def run_multimodal(args):
    """Run --num-prompts image+text prompts through the vision path as one batch.

    Images are passed as base64 data URIs so `llm.chat` places the image tokens.
    """
    import base64
    import urllib.request

    from vllm import LLM, SamplingParams

    def _fetch(url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "spyre-inference"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            image_bytes = resp.read()
        print(f"Downloaded {len(image_bytes)} bytes from {url}")
        return "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")

    cases = [MULTIMODAL_CASES[i % len(MULTIMODAL_CASES)] for i in range(args.num_prompts)]

    prepared: list[tuple[str, str, str]] = []
    for url, question, expected in cases:
        try:
            prepared.append((_fetch(url), question, expected))
        except Exception as exc:  # noqa: BLE001 - any download failure is non-fatal
            # Reuse the first image rather than shrink the batch, which is the
            # variable under test.
            if not prepared:
                raise
            print(f"WARNING: {url} failed ({exc}); reusing the first image.")
            prepared.append((prepared[0][0], question, expected))

    config_format = args.config_format or _default_config_format(args.model)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        config_format=config_format,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tp,
        max_num_batched_tokens=args.max_num_batched_tokens,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        limit_mm_per_prompt={"image": 1},
    )

    # One conversation per image; llm.chat runs max_num_seqs of them concurrently.
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": MULTIMODAL_QUESTION.format(question)},
                ],
            }
        ]
        for uri, question, _ in prepared
    ]
    # Pixtral expands one image into thousands of tokens, so --max-model-len must
    # leave room for those plus the reasoning prompt.
    sampling_params = SamplingParams(
        max_tokens=max(int(v) for v in args.max_tokens.split(",")),
        temperature=0.0,
    )

    print(f"=============== GENERATE (multimodal, {len(conversations)} prompt(s))")
    t0 = time.time()
    outputs = llm.chat(conversations, sampling_params)
    elapsed = time.time() - t0
    print(f"Time elapsed: {elapsed:.2f} sec")
    print("===============")
    for i, output in enumerate(outputs):
        # The full wrapped prompt: the ChartQA formatting rules shape the answer.
        print(f"\n[{i}] Prompt sent to model:\n{MULTIMODAL_QUESTION.format(prepared[i][1])}")
        print(f"\nExpected answer:\n {prepared[i][2]!r}")
        print(f"\nGenerated text:\n {output.outputs[0].text!r}\n")
        print("-----------------------------------")

    if args.compare_with_cpu:
        compare_multimodal_with_cpu(args, prepared, outputs, sampling_params.max_tokens)


def compare_multimodal_with_cpu(args, prepared, outputs, max_tokens):
    """Re-run the same image+question pairs through HuggingFace on CPU.

    Both texts are printed rather than compared: free-form answers rarely match
    token-for-token, and HF differs from vLLM in preprocessing and template too.
    """
    import base64
    import io

    print("Comparing multimodal results with HF on cpu")
    print("===============")

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(args.model)
        # Match the Spyre run's float16 weights, as the text path does.
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float16)
    except Exception as exc:  # noqa: BLE001 - a missing HF-format config is not fatal
        # mistral-format repos may carry no HF processor config, leaving no CPU oracle.
        print(f"Cannot load {args.model} with transformers ({exc}); skipping CPU comparison.")
        return

    for i, (uri, question, expected) in enumerate(prepared):
        image = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": MULTIMODAL_QUESTION.format(question)},
                ],
            }
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=text, images=image, return_tensors="pt")

        # vLLM ignores EOS, so force HF to emit exactly max_tokens too.
        hf_output = model.generate(
            **inputs,
            do_sample=False,
            min_new_tokens=max_tokens,
            max_new_tokens=max_tokens,
        )
        hf_text = processor.batch_decode(
            hf_output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]

        spyre_text = outputs[i].outputs[0].text
        match = "MATCH" if hf_text == spyre_text else "DIFFER"
        print(f"\n[{i}] {match} — Prompt sent to model:\n{MULTIMODAL_QUESTION.format(question)}")
        print(f"\nExpected answer:\n {expected!r}")
        print(f"\nSpyre generated text:\n {spyre_text!r}\n")
        print(f"\nCPU generated text:\n {hf_text!r}\n")
        print("-----------------------------------")


def main():
    args = parse_args()

    if platform.machine() == "arm64":
        print(
            "Detected arm64 running environment. "
            "Setting HF_HUB_OFFLINE=1 otherwise vllm tries to download a "
            "different version of the model using HF API which might not work "
            "locally on arm64."
        )
        os.environ["HF_HUB_OFFLINE"] = "1"

    run_multimodal(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
