"""Compare greedy generations between load formats (e.g. safetensors vs spyre).

Run each format in its own process, one at a time (never two Spyre jobs at once):

    uv run --no-sync python scripts/compare_load_formats.py run --load-format safetensors --out /tmp/gen_st.json
    uv run --no-sync python scripts/compare_load_formats.py run --load-format spyre       --out /tmp/gen_sp.json
    python3 scripts/compare_load_formats.py compare /tmp/gen_st.json /tmp/gen_sp.json
"""

import argparse
import json
import os

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain in one sentence why the sky is blue:",
    "1, 2, 3, 4, 5,",
]


def run(args):
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "64")
    os.environ.setdefault("SPYRE_DEVICES", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="float16",
        load_format=args.load_format,
        max_model_len=128,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=1)
    result = []
    # One prompt at a time so batching cannot differ between the two runs.
    for prompt in PROMPTS:
        out = llm.generate([prompt], params)[0].outputs[0]
        result.append(
            {
                "prompt": prompt,
                "text": out.text,
                "token_ids": list(out.token_ids),
                "logprobs": [
                    lp[tid].logprob for tid, lp in zip(out.token_ids, out.logprobs)
                ],
            }
        )
    with open(args.out, "w") as f:
        json.dump({"load_format": args.load_format, "results": result}, f, indent=1)
    print(f"wrote {args.out}")


def compare(args):
    a = json.load(open(args.a))
    b = json.load(open(args.b))
    print(f"A = {a['load_format']}   B = {b['load_format']}")
    all_match = True
    for ra, rb in zip(a["results"], b["results"]):
        ta, tb = ra["token_ids"], rb["token_ids"]
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
            None if len(ta) == len(tb) else min(len(ta), len(tb)),
        )
        n = min(len(ta), len(tb))
        max_dlp = max((abs(x - y) for x, y in zip(ra["logprobs"][:n], rb["logprobs"][:n])), default=0.0)
        status = "MATCH" if first_diff is None else f"DIFF at token {first_diff}"
        all_match &= first_diff is None
        print(f"- {ra['prompt']!r}: {status}; max |dlogprob| = {max_dlp:.4f}")
        if first_diff is not None:
            print(f"    A: {ra['text']!r}")
            print(f"    B: {rb['text']!r}")
    print("RESULT:", "all identical" if all_match else "outputs differ")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--load-format", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default="ibm-ai-platform/micro-g3.3-8b-instruct-1b")
    r.add_argument("--max-tokens", type=int, default=32)
    r.set_defaults(fn=run)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(fn=compare)
    args = p.parse_args()
    args.fn(args)

