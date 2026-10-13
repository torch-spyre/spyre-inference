"""Dump model weights after loading (per load format) and diff them per parameter.

The dump itself happens in the worker (spyre_model_runner.py, enabled by
SPYRE_DUMP_WEIGHTS_DIR). Run one format at a time, never in parallel:

    uv run --no-sync python scripts/diff_loaded_weights.py dump --load-format safetensors --out /tmp/w_st
    uv run --no-sync python scripts/diff_loaded_weights.py dump --load-format spyre       --out /tmp/w_sp
    uv run --no-sync python scripts/diff_loaded_weights.py diff /tmp/w_st /tmp/w_sp
"""

import argparse
import os


def dump(args):
    os.environ["SPYRE_DUMP_WEIGHTS_DIR"] = args.out
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "64")
    os.environ.setdefault("SPYRE_DEVICES", "0")
    from vllm import LLM

    LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="float16",
        load_format=args.load_format,
        max_model_len=128,
    )
    print(f"weights in {args.out}/weights.pt")


def diff(args):
    import torch

    a = torch.load(os.path.join(args.a, "weights.pt"))
    b = torch.load(os.path.join(args.b, "weights.pt"))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a:
        print("only in A:", only_a)
    if only_b:
        print("only in B:", only_b)

    rows, n_equal = [], 0
    for name in sorted(set(a) & set(b)):
        x, y = a[name], b[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            rows.append(
                (
                    float("inf"),
                    name,
                    f"shape/dtype {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}",
                )
            )
            continue
        d = (x.float() - y.float()).abs()
        m = d.max().item() if d.numel() else 0.0
        if m == 0.0:
            n_equal += 1
        else:
            rel = m / max(x.float().abs().max().item(), 1e-12)
            rows.append(
                (
                    m,
                    name,
                    f"max_abs={m:.3e} rel_to_max={rel:.3e} "
                    f"n_diff={(d > 0).sum().item()}/{d.numel()} dtype={x.dtype}",
                )
            )

    print(f"{n_equal} tensors bit-identical, {len(rows)} differ, of {len(set(a) & set(b))} common")
    for _, name, info in sorted(rows, reverse=True)[: args.top]:
        print(f"  {name}: {info}")
    if not rows and not only_a and not only_b:
        print("RESULT: weights identical -> any output difference comes from compute, not loading")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--load-format", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--model", default="ibm-ai-platform/micro-g3.3-8b-instruct-1b")
    d.set_defaults(fn=dump)
    f = sub.add_parser("diff")
    f.add_argument("a")
    f.add_argument("b")
    f.add_argument("--top", type=int, default=40)
    f.set_defaults(fn=diff)
    args = p.parse_args()
    args.fn(args)

