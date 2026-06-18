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

"""Tests for the connector-level NIXL smoke script.

Helper coverage (CLI parser, paged cache builder, checksums) runs with
torch only. The full in-process roundtrip needs vLLM and skips cleanly
when it is absent. No NIXL or AIU hardware is required anywhere.
"""

import importlib.util
import json
import pathlib

import pytest

torch = pytest.importorskip("torch", reason="torch required for smoke helper tests")

REPO = pathlib.Path(__file__).resolve().parents[1]
CONNECTOR_DIR = REPO / "spyre_inference" / "distributed" / "kv_transfer" / "kv_connector" / "v1"


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Loading by file path keeps these imports vLLM-free: the smoke script's
# helpers are torch-only, and spyre_inference/__init__.py would pull vLLM in.
smoke = _load_by_path(
    "spyre_connector_nixl_smoke",
    REPO / "examples" / "kv_connector" / "spyre_connector_nixl_smoke.py",
)
accessor_mod = _load_by_path(
    "spyre_paged_kv_accessor", CONNECTOR_DIR / "spyre_paged_kv_accessor.py"
)


def test_parser_roles_and_defaults():
    args = smoke.build_parser().parse_args(["--role", "both"])
    assert args.role == "both"
    assert args.num_layers == 2
    assert args.num_kv_heads == 2
    assert args.block_size == 4
    assert args.head_dim == 8
    assert args.num_pages == 4
    assert args.block_ids == [1, 2]
    assert args.device == "cpu"

    args = smoke.build_parser().parse_args(
        ["--role", "decode", "--block-ids", "0", "3", "--num-pages", "6"]
    )
    assert args.role == "decode"
    assert args.block_ids == [0, 3]
    assert args.num_pages == 6


def test_parser_requires_valid_role():
    with pytest.raises(SystemExit):
        smoke.build_parser().parse_args([])
    with pytest.raises(SystemExit):
        smoke.build_parser().parse_args(["--role", "scheduler"])


def test_parser_split_mode_flags():
    args = smoke.build_parser().parse_args(
        [
            "--role",
            "decode",
            "--nixl",
            "--prefill-ip",
            "10.1.2.3",
            "--prefill-port",
            "9200",
            "--listen-port",
            "9300",
            "--source-request-id",
            "req-src",
            "--request-id",
            "req-dst",
            "--timeout-s",
            "30",
            "--ready-file",
            "/tmp/ready.json",
            "--expected-json",
            "/tmp/expected.json",
            "--keepalive-s",
            "5",
        ]
    )
    assert args.nixl is True
    assert args.prefill_ip == "10.1.2.3"
    assert args.prefill_port == 9200
    assert args.listen_port == 9300
    assert args.source_request_id == "req-src"
    assert args.request_id == "req-dst"
    assert args.timeout_s == 30.0
    assert args.ready_file == "/tmp/ready.json"
    assert args.expected_json == "/tmp/expected.json"
    assert args.keepalive_s == 5.0

    defaults = smoke.build_parser().parse_args(["--role", "both"])
    assert defaults.nixl is False
    assert defaults.prefill_port == smoke.DEFAULT_NIXL_PORT
    assert defaults.listen_port == smoke.DEFAULT_NIXL_PORT
    assert defaults.source_request_id == smoke.PREFILL_REQ_ID
    assert defaults.request_id == smoke.DECODE_REQ_ID


def test_result_json_aliases_result_file():
    a = smoke.build_parser().parse_args(["--role", "both", "--result-json", "/tmp/a.json"])
    b = smoke.build_parser().parse_args(["--role", "both", "--result-file", "/tmp/a.json"])
    assert a.result_file == b.result_file == "/tmp/a.json"


def test_split_env_role_configs():
    prefill = smoke.build_parser().parse_args(["--role", "prefill", "--nixl"])
    env = smoke.split_env(prefill)
    assert env["VLLM_SPYRE_ENABLE_NIXL_TRANSFER"] == "1"
    assert env["VLLM_SPYRE_KV_ROLE"] == "kv_producer"
    assert env["VLLM_SPYRE_NIXL_BLOCKING_TRANSFER"] == "0"
    assert "VLLM_SPYRE_NIXL_REMOTE_IP" not in env

    decode = smoke.build_parser().parse_args(
        ["--role", "decode", "--nixl", "--prefill-ip", "10.9.8.7"]
    )
    env = smoke.split_env(decode)
    assert env["VLLM_SPYRE_KV_ROLE"] == "kv_consumer"
    assert env["VLLM_SPYRE_NIXL_REMOTE_IP"] == "10.9.8.7"
    assert "VLLM_SPYRE_NIXL_BLOCKING_TRANSFER" not in env


def test_expected_json_roundtrip(tmp_path):
    args = smoke.build_parser().parse_args(["--role", "both"])
    expected = smoke.expected_checksums(args)
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(expected))
    assert smoke.checksums_match(expected, json.loads(path.read_text()))


def test_nixl_with_role_both_rejected(tmp_path):
    result_file = tmp_path / "result.json"
    rc = smoke.main(["--role", "both", "--nixl", "--result-file", str(result_file)])
    assert rc == 1
    result = json.loads(result_file.read_text())
    assert result["success"] is False
    assert "split" in result["error"]


def test_split_helpers_import_without_vllm():
    """Split-mode helper logic must not require vLLM at module import time."""
    src = (REPO / "examples" / "kv_connector" / "spyre_connector_nixl_smoke.py").read_text()
    top = src.split("def _apply_split_env", 1)[0]
    assert "import vllm" not in top
    assert "from vllm" not in top
    # module already imported above without vLLM
    assert smoke.split_env is not None


def test_nixl_receive_path_is_dtype_aware():
    """Decode receive buffers must use the registered cache dtype, not fp32.

    fp32 sizing halves the element count for fp16 pages (a 128-byte page is
    64 fp16 elements, not 32 fp32) and NIXL rejects the transfer with
    NIXL_ERR_INVALID_PARAM.
    """
    src = (CONNECTOR_DIR / "inmemory_spyre_connector.py").read_text()
    body = src.split("def _load_saved_requests_nixl", 1)[1].split("def _save_request_nixl", 1)[0]
    assert "torch.float32.itemsize" not in body
    assert "dtype=torch.float32" not in body
    assert "recv_dtype = self._nixl_receive_dtype()" in body
    assert "desc_len_bytes // recv_itemsize" in body
    assert "torch.zeros(kv_block_shape, dtype=recv_dtype" in body

    helper = src.split("def _nixl_receive_dtype", 1)[1].split("def _load_saved_requests_nixl", 1)[0]
    assert "self._paged_accessor.dtype" in helper
    assert "return torch.float32" in helper  # only as the unregistered fallback


def test_nixl_receive_uses_block_layout_buffers_for_paged():
    """Paged NIXL receive buffers must match the producer's registered layout.

    The producer registers store-resident blocks in heap-block layout
    [block_size, num_kv_heads, head_dim] (from _save_kv_bulk via
    SpyrePagedKVCacheAccessor.read_block). Receiving into page-shaped
    buffers and permuting afterwards transposes cell positions and breaks
    checksums, so the receive shape is block_shape and tensors are stored
    directly, with no paged-only permute.
    """
    src = (CONNECTOR_DIR / "inmemory_spyre_connector.py").read_text()
    body = src.split("def _load_saved_requests_nixl", 1)[1].split("def _save_request_nixl", 1)[0]
    assert "kv_block_shape = self._paged_accessor.block_shape" in body
    assert "self._paged_accessor.page_shape" not in body
    assert "key_tensor.permute" not in body
    assert "value_tensor.permute" not in body
    assert "self._store.put(key_store_key, key_tensor)" in body
    assert "self._store.put(value_store_key, value_tensor)" in body


def test_connector_nonblocking_save_does_not_wait_for_client():
    """Producer must expose pending transfers without a connected client."""
    src = (CONNECTOR_DIR / "inmemory_spyre_connector.py").read_text()
    body = src.split("def _save_request_nixl", 1)[1].split("def _save_request_record", 1)[0]
    assert body.count('check_remote_metadata("client")') == 1
    wait = body.index('check_remote_metadata("client")')
    blocking_branch = body.index("BLOCKING MODE: wait for the client")
    assert blocking_branch < wait, "client wait must be inside the blocking-mode branch"
    assert "_pending_transfers[record.req_id]" in body


def test_paged_cache_builder_activates_accessor():
    caches = smoke.build_paged_kv_caches(
        num_layers=2, num_kv_heads=2, block_size=4, head_dim=8, num_pages=4
    )
    accessor = accessor_mod.SpyrePagedKVCacheAccessor.try_from_kv_caches(caches)
    assert accessor is not None
    assert accessor.num_layers == 2
    assert accessor.num_pages == 4
    assert accessor.page_shape == (2, 4, 8)
    assert accessor.dtype == torch.float16


def test_deterministic_blocks_distinct_and_reproducible():
    kw = {"num_kv_heads": 2, "block_size": 4, "head_dim": 8}
    a = smoke.deterministic_block(0, "k", 1, **kw)
    assert a.shape == (2, 4, 8)
    assert a.dtype == torch.float16
    assert torch.equal(a, smoke.deterministic_block(0, "k", 1, **kw))
    assert not torch.equal(a, smoke.deterministic_block(0, "v", 1, **kw))
    assert not torch.equal(a, smoke.deterministic_block(1, "k", 1, **kw))
    assert not torch.equal(a, smoke.deterministic_block(0, "k", 2, **kw))


def test_checksum_match_helper():
    filled = smoke.build_paged_kv_caches(
        num_layers=2, num_kv_heads=2, block_size=4, head_dim=8, num_pages=4, fill_block_ids=[1, 2]
    )
    empty = smoke.build_paged_kv_caches(
        num_layers=2, num_kv_heads=2, block_size=4, head_dim=8, num_pages=4
    )
    expected = smoke.cache_checksums(filled, [1, 2])
    assert len(expected) == 2 * 2 * 2  # layers * kinds * blocks
    assert smoke.checksums_match(expected, smoke.cache_checksums(filled, [1, 2]))
    assert not smoke.checksums_match(expected, smoke.cache_checksums(empty, [1, 2]))
    assert not smoke.checksums_match({}, {})


def test_expected_checksums_match_builder():
    args = smoke.build_parser().parse_args(["--role", "both"])
    assert smoke.expected_checksums(args) == smoke.cache_checksums(
        smoke.build_paged_kv_caches(
            num_layers=2,
            num_kv_heads=2,
            block_size=4,
            head_dim=8,
            num_pages=4,
            fill_block_ids=[1, 2],
        ),
        [1, 2],
    )


def test_inprocess_roundtrip_without_nixl(tmp_path, capsys, monkeypatch):
    """Full prefill -> store -> decode roundtrip through the real connector."""
    pytest.importorskip("vllm", reason="vLLM required for connector roundtrip")
    monkeypatch.delenv("VLLM_SPYRE_ENABLE_NIXL_TRANSFER", raising=False)
    monkeypatch.delenv("VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE", raising=False)

    from spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        reset_global_store,
    )

    reset_global_store()
    result_file = tmp_path / "result.json"
    rc = smoke.main(["--role", "both", "--result-file", str(result_file)])

    out = capsys.readouterr().out
    for marker in (
        smoke.MARK_PREFILL_READY,
        smoke.MARK_SAVE_DONE,
        smoke.MARK_DECODE_START,
        smoke.MARK_LOAD_DONE,
        f"{smoke.MARK_CONTENT_MATCH} true",
        f"{smoke.MARK_SUCCESS} true",
    ):
        assert marker in out

    result = json.loads(result_file.read_text())
    assert rc == 0
    assert result["success"] is True
    assert result["content_match"] is True
    assert result["error"] is None
    assert result["layout"] == "list_of_pages"
    assert result["expected_checksums"] == result["actual_checksums"]
    assert result["load_error_block_ids"] == []
