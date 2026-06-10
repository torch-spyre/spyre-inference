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

"""Connector-level KV smoke test for InMemorySpyreConnector.

Drives the real connector path — not nixl._api directly:

    register_kv_caches -> bind_connector_metadata
        -> wait_for_save (prefill) / start_load_kv (decode)
        -> verify KV block contents

The synthetic KV cache uses the Spyre paged layout: each layer cache is
``(k_pages, v_pages)`` with per-page tensors ``[num_kv_heads, block_size,
head_dim]``, so ``SpyrePagedKVCacheAccessor`` is active.

Modes:

* In-process roundtrip (no NIXL): both halves share the global store.
      python examples/kv_connector/spyre_connector_nixl_smoke.py --role both

* Two-process / two-pod NIXL: the producer saves through
  ``_save_request_nixl`` (non-blocking), exposes pending transfers, and
  keeps serving LIST/PULL requests; the consumer pulls through
  ``_load_saved_requests_nixl`` then loads pages via ``start_load_kv``.
      # pod 1
      python ... --role prefill --nixl --listen-port 9100 \
          --ready-file /shared/prefill_ready.json --expected-json /shared/expected.json
      # pod 2
      python ... --role decode --nixl --prefill-ip <pod1-ip> --prefill-port 9100 \
          --ready-file /shared/prefill_ready.json --expected-json /shared/expected.json

Grep-friendly markers:
    CONNECTOR_PREFILL_READY / CONNECTOR_SAVE_DONE / CONNECTOR_NIXL_READY
    CONNECTOR_PREFILL_KEEPALIVE_DONE
    CONNECTOR_DECODE_START / CONNECTOR_NIXL_PULL_DONE / CONNECTOR_LOAD_DONE
    CONNECTOR_CONTENT_MATCH true|false
    CONNECTOR_SMOKE_SUCCESS true|false

This module's helpers (CLI parser, paged cache builder, checksums) import
without vLLM; the connector itself is imported lazily inside main().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
from typing import Any

import torch

MARK_PREFILL_READY = "CONNECTOR_PREFILL_READY"
MARK_SAVE_DONE = "CONNECTOR_SAVE_DONE"
MARK_NIXL_READY = "CONNECTOR_NIXL_READY"
MARK_KEEPALIVE_DONE = "CONNECTOR_PREFILL_KEEPALIVE_DONE"
MARK_DECODE_START = "CONNECTOR_DECODE_START"
MARK_NIXL_PULL_DONE = "CONNECTOR_NIXL_PULL_DONE"
MARK_LOAD_DONE = "CONNECTOR_LOAD_DONE"
MARK_CONTENT_MATCH = "CONNECTOR_CONTENT_MATCH"
MARK_SUCCESS = "CONNECTOR_SMOKE_SUCCESS"

PREFILL_REQ_ID = "smoke-prefill-0"
DECODE_REQ_ID = "smoke-decode-0"
DEFAULT_NIXL_PORT = 9100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", choices=("prefill", "decode", "both"), required=True)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--num-pages", type=int, default=4)
    parser.add_argument("--block-ids", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", default="cpu", help="page tensor device (cpu for smoke)")
    parser.add_argument(
        "--result-file",
        "--result-json",
        dest="result_file",
        default="spyre_connector_smoke_result.json",
        help="where to write the JSON result (--result-json is an alias)",
    )
    # Split-process / two-pod NIXL mode.
    parser.add_argument(
        "--nixl",
        action="store_true",
        help="use the connector NIXL transport (split prefill/decode processes)",
    )
    parser.add_argument("--prefill-ip", default="127.0.0.1", help="producer IP (decode side)")
    parser.add_argument(
        "--prefill-port", type=int, default=DEFAULT_NIXL_PORT, help="producer port (decode side)"
    )
    parser.add_argument(
        "--listen-port", type=int, default=DEFAULT_NIXL_PORT, help="producer port (prefill side)"
    )
    parser.add_argument("--source-request-id", default=PREFILL_REQ_ID)
    parser.add_argument("--request-id", default=DECODE_REQ_ID)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--ready-file",
        default="",
        help="prefill writes this file when NIXL is ready; decode waits for it",
    )
    parser.add_argument(
        "--expected-json",
        default="",
        help="prefill writes expected checksums here; decode verifies against it",
    )
    parser.add_argument(
        "--keepalive-s",
        type=float,
        default=60.0,
        help="prefill stays alive serving pull requests for this many seconds",
    )
    return parser


def layer_names(num_layers: int) -> list[str]:
    return [f"model.layers.{i}.self_attn.attn" for i in range(num_layers)]


def deterministic_block(
    layer_idx: int,
    kv_kind: str,
    block_id: int,
    *,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Known-value page [num_kv_heads, block_size, head_dim], fp16."""
    base = layer_idx * 1000.0 + block_id * 10.0 + (0.0 if kv_kind == "k" else 5.0)
    numel = num_kv_heads * block_size * head_dim
    ramp = torch.arange(numel, dtype=torch.float16) * 0.125
    return (base + ramp).reshape(num_kv_heads, block_size, head_dim).to(torch.float16)


def build_paged_kv_caches(
    *,
    num_layers: int,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    num_pages: int,
    fill_block_ids: list[int] | None = None,
    device: str = "cpu",
) -> dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Synthetic Spyre paged caches; pages in fill_block_ids get known values."""
    caches: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for layer_idx, name in enumerate(layer_names(num_layers)):
        pages = {}
        for kind in ("k", "v"):
            pages[kind] = [
                torch.zeros(num_kv_heads, block_size, head_dim, dtype=torch.float16, device=device)
                for _ in range(num_pages)
            ]
            for block_id in fill_block_ids or []:
                pages[kind][block_id] = deterministic_block(
                    layer_idx,
                    kind,
                    block_id,
                    num_kv_heads=num_kv_heads,
                    block_size=block_size,
                    head_dim=head_dim,
                ).to(device)
        caches[name] = (pages["k"], pages["v"])
    return caches


def block_checksum(page: torch.Tensor) -> str:
    raw = page.to("cpu").contiguous().view(-1).view(torch.uint8)
    return hashlib.sha256(bytes(raw.tolist())).hexdigest()


def cache_checksums(
    caches: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]],
    block_ids: list[int],
) -> dict[str, str]:
    """Checksum per layer/kind/block: {'layer|k|1': sha256}."""
    sums: dict[str, str] = {}
    for name, (k_pages, v_pages) in sorted(caches.items()):
        for kind, pages in (("k", k_pages), ("v", v_pages)):
            for block_id in block_ids:
                sums[f"{name}|{kind}|{block_id}"] = block_checksum(pages[block_id])
    return sums


def checksums_match(expected: dict[str, str], actual: dict[str, str]) -> bool:
    return bool(expected) and expected == actual


def expected_checksums(args: argparse.Namespace) -> dict[str, str]:
    filled = build_paged_kv_caches(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        block_size=args.block_size,
        head_dim=args.head_dim,
        num_pages=args.num_pages,
        fill_block_ids=list(args.block_ids),
    )
    return cache_checksums(filled, list(args.block_ids))


def geometry_dict(args: argparse.Namespace) -> dict[str, int]:
    return {
        "num_layers": args.num_layers,
        "num_kv_heads": args.num_kv_heads,
        "block_size": args.block_size,
        "head_dim": args.head_dim,
        "num_pages": args.num_pages,
    }


def split_env(args: argparse.Namespace) -> dict[str, str]:
    """Environment that selects the connector NIXL path for a split role.

    Producer runs non-blocking so wait_for_save returns after exposing the
    pending transfer; the LIST/PULL handler then serves the consumer.
    """
    env: dict[str, str] = {"VLLM_SPYRE_ENABLE_NIXL_TRANSFER": "1"}
    if args.role == "prefill":
        env["VLLM_SPYRE_KV_ROLE"] = "kv_producer"
        env["VLLM_SPYRE_NIXL_BLOCKING_TRANSFER"] = "0"
    else:
        env["VLLM_SPYRE_KV_ROLE"] = "kv_consumer"
        env["VLLM_SPYRE_NIXL_REMOTE_IP"] = args.prefill_ip
    return env


def _apply_split_env(args: argparse.Namespace) -> None:
    os.environ.update(split_env(args))
    # NIXL_PORT is a module constant; override it for non-default ports.
    port = args.listen_port if args.role == "prefill" else args.prefill_port
    if port != DEFAULT_NIXL_PORT:
        from spyre_inference.distributed.kv_transfer.kv_connector.v1 import (
            inmemory_spyre_connector as mod,
        )

        mod.NIXL_PORT = port


def _wait_for_file(path: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    target = pathlib.Path(path)
    while time.monotonic() < deadline:
        if target.is_file():
            return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for {path}")


def _pending_count(connector) -> int:
    lock = connector._pending_transfers_lock
    if lock is None:
        return len(connector._pending_transfers)
    with lock:
        return len(connector._pending_transfers)


def _make_connector(role_name: str):
    """Build an InMemorySpyreConnector without a full engine.

    KVConnectorBase_V1 requires `vllm_config.kv_transfer_config` to be set,
    so we attach a minimal KVTransferConfig that selects the Spyre connector
    and a role that mirrors the script role.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        InMemorySpyreConnector,
    )

    role = KVConnectorRole.WORKER
    kv_role = "kv_producer" if role_name == "prefill" else "kv_consumer"
    try:
        from vllm.config import KVTransferConfig, VllmConfig

        kv_cfg = KVTransferConfig(kv_connector="InMemorySpyreConnector", kv_role=kv_role)
        vllm_config = VllmConfig(kv_transfer_config=kv_cfg)
    except Exception:
        # Minimal duck-typed config: the connector reads cache_config.block_size
        # and kv_transfer_config.{kv_role,kv_connector} at construction time.
        from types import SimpleNamespace

        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=0),
            kv_transfer_config=SimpleNamespace(
                kv_role=kv_role,
                kv_connector="InMemorySpyreConnector",
                engine_id="smoke-engine",
                kv_buffer_device="cpu",
                kv_buffer_size=int(1e9),
                kv_rank=None,
                kv_parallel_size=1,
                kv_ip="127.0.0.1",
                kv_port=14579,
                kv_connector_extra_config={},
                kv_connector_module_path=None,
                enable_permute_local_kv=False,
                kv_load_failure_policy="fail",
            ),
        )
    return InMemorySpyreConnector(vllm_config, role)


def _make_meta(
    args: argparse.Namespace,
    *,
    is_store: bool,
    block_ids: list[int] | None = None,
):
    from spyre_inference.distributed.kv_transfer.kv_connector.v1.metadata import (
        SpyreConnectorMeta,
    )

    meta = SpyreConnectorMeta(
        layer_names=layer_names(args.num_layers),
        block_size=args.block_size,
        dtype="torch.float16",
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )
    block_ids = list(block_ids if block_ids is not None else args.block_ids)
    token_count = len(block_ids) * args.block_size
    if is_store:
        meta.add_store_request(args.source_request_id, block_ids, token_count=token_count)
    else:
        meta.add_load_request(
            args.request_id,
            block_ids,
            source_req_id=args.source_request_id,
            token_count=token_count,
            block_mapping=[(b, b) for b in block_ids],
        )
    return meta


def run_prefill(args: argparse.Namespace, result: dict[str, Any]) -> None:
    connector = _make_connector("prefill")
    caches = build_paged_kv_caches(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        block_size=args.block_size,
        head_dim=args.head_dim,
        num_pages=args.num_pages,
        fill_block_ids=list(args.block_ids),
        device=args.device,
    )
    connector.register_kv_caches(caches)
    if not connector.paged_kv_active():
        raise RuntimeError("paged accessor not active for synthetic page lists")
    print(MARK_PREFILL_READY, flush=True)

    connector.bind_connector_metadata(_make_meta(args, is_store=True))
    connector.wait_for_save()
    connector.clear_connector_metadata()
    result["connector_stats"]["prefill"] = connector.get_cumulative_metrics()
    print(MARK_SAVE_DONE, flush=True)

    if not args.nixl:
        return

    # Non-blocking save exposed the transfer; wait until it is registered.
    deadline = time.monotonic() + args.timeout_s
    while _pending_count(connector) == 0:
        if time.monotonic() > deadline:
            raise TimeoutError("pending NIXL transfer was not registered before timeout")
        time.sleep(0.05)
    result["nixl_pending_requests"] = _pending_count(connector)
    print(MARK_NIXL_READY, flush=True)

    if args.expected_json:
        pathlib.Path(args.expected_json).write_text(json.dumps(result["expected_checksums"]))
    if args.ready_file:
        pathlib.Path(args.ready_file).write_text(
            json.dumps(
                {
                    "req_id": args.source_request_id,
                    "listen_port": args.listen_port,
                    "block_ids": list(args.block_ids),
                    "geometry": geometry_dict(args),
                }
            )
        )

    # Serve LIST/PULL requests while the consumer connects and pulls.
    deadline = time.monotonic() + args.keepalive_s
    while time.monotonic() < deadline:
        time.sleep(0.5)
    result["nixl_pending_requests_after_keepalive"] = _pending_count(connector)
    print(MARK_KEEPALIVE_DONE, flush=True)


def run_decode(args: argparse.Namespace, result: dict[str, Any]) -> tuple[bool, bool]:
    print(MARK_DECODE_START, flush=True)
    if args.nixl and args.ready_file:
        _wait_for_file(args.ready_file, args.timeout_s)

    connector = _make_connector("decode")
    caches = build_paged_kv_caches(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        block_size=args.block_size,
        head_dim=args.head_dim,
        num_pages=args.num_pages,
        fill_block_ids=None,  # decode starts empty
        device=args.device,
    )
    connector.register_kv_caches(caches)
    if not connector.paged_kv_active():
        raise RuntimeError("paged accessor not active for synthetic page lists")

    load_block_ids = list(args.block_ids)
    if args.nixl:
        records = connector._load_saved_requests_nixl()
        if not records:
            raise RuntimeError("NIXL pull returned no saved requests")
        by_id = {record.req_id: record for record in records}
        record = by_id.get(args.source_request_id, records[-1])
        result["nixl_pulled_request_ids"] = sorted(by_id)
        load_block_ids = list(record.block_ids)
        print(MARK_NIXL_PULL_DONE, flush=True)

    connector.bind_connector_metadata(_make_meta(args, is_store=False, block_ids=load_block_ids))
    connector.start_load_kv(None)
    load_errors = connector.get_block_ids_with_load_errors()
    connector.clear_connector_metadata()
    result["connector_stats"]["decode"] = connector.get_cumulative_metrics()
    result["load_error_block_ids"] = sorted(load_errors)
    print(MARK_LOAD_DONE, flush=True)

    actual = cache_checksums(caches, load_block_ids)
    result["actual_checksums"] = actual
    match = checksums_match(result["expected_checksums"], actual) and not load_errors
    return match, bool(load_errors)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: dict[str, Any] = {
        "role": args.role,
        "layout": "list_of_pages",
        "nixl": bool(args.nixl),
        "request_ids": {"prefill": args.source_request_id, "decode": args.request_id},
        "block_ids": list(args.block_ids),
        "geometry": geometry_dict(args),
        "connector_stats": {},
        "expected_checksums": expected_checksums(args),
        "actual_checksums": {},
        "content_match": None,
        "success": False,
        "error": None,
    }

    try:
        if max(args.block_ids) >= args.num_pages or min(args.block_ids) < 0:
            raise ValueError("--block-ids must be within [0, --num-pages)")
        if args.nixl and args.role == "both":
            raise ValueError("--nixl requires split --role prefill / --role decode processes")
        if args.nixl:
            _apply_split_env(args)
        if args.nixl and args.role == "decode" and args.expected_json:
            result["expected_checksums"] = json.loads(pathlib.Path(args.expected_json).read_text())

        if args.role in ("prefill", "both"):
            run_prefill(args, result)
        if args.role in ("decode", "both"):
            match, _ = run_decode(args, result)
            result["content_match"] = match
            print(f"{MARK_CONTENT_MATCH} {str(match).lower()}", flush=True)
            result["success"] = match
        else:
            result["success"] = True  # prefill-only success = save (+ NIXL expose) done
    except Exception as exc:  # noqa: BLE001 — smoke must always emit a verdict
        result["error"] = f"{type(exc).__name__}: {exc}"

    print(f"{MARK_SUCCESS} {str(result['success']).lower()}", flush=True)
    pathlib.Path(args.result_file).write_text(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
