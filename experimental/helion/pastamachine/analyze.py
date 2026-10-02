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
pastamachine.analyze
~~~~~~~~~~~~~~~~~~~~

Analysis utilities for inspecting SDSC metadata produced by the Spyre
compiler.  Works with :class:`~pastamachine.util.TranspileMeta` objects
returned by :func:`~pastamachine.compile_helion_to_spyre` when
``return_meta=True``.
"""

from __future__ import annotations

from typing import Any

from pastamachine._logging import getLogger
from pastamachine.util import TranspileMeta

log = getLogger("analyze")


def extract_key_sdsc_info(sdsc_data: dict[str, Any]) -> dict[str, Any]:
    """Extract key information from a parsed SDSC JSON structure.

    Parameters
    ----------
    sdsc_data : dict
        The parsed content of an ``sdsc.json`` file (as returned by
        :meth:`TranspileMeta.load_sdsc`).

    Returns
    -------
    dict
        A flat dictionary with the most relevant fields for quick inspection:
        ``operation``, ``numCoresUsed``, ``coreIdToDsc``,
        ``numWkSlicesPerDim``, and (when present) per-DSC core/corelet info
        and compute operation details.
    """
    info: dict[str, Any] = {}

    op_keys = list(sdsc_data.keys())
    if not op_keys:
        return info

    op_key = op_keys[0]
    op_data = sdsc_data[op_key]

    info["operation"] = op_key
    info["numCoresUsed"] = op_data.get("numCoresUsed_", "N/A")
    info["coreIdToDsc"] = op_data.get("coreIdToDsc_", {})
    info["numWkSlicesPerDim"] = op_data.get("numWkSlicesPerDim_", {})

    if "dscs_" in op_data and len(op_data["dscs_"]) > 0:
        dsc = op_data["dscs_"][0]
        dsc_key = list(dsc.keys())[0]
        dsc_data = dsc[dsc_key]

        info["dsc_numCoresUsed"] = dsc_data.get("numCoresUsed_", "N/A")
        info["dsc_numCoreletsUsed"] = dsc_data.get("numCoreletsUsed_", "N/A")
        info["dsc_coreIdsUsed"] = dsc_data.get("coreIdsUsed_", [])

        if "computeOp_" in dsc_data and len(dsc_data["computeOp_"]) > 0:
            compute_op = dsc_data["computeOp_"][0]
            info["compute_exUnit"] = compute_op.get("exUnit", "N/A")
            info["compute_opFuncName"] = compute_op.get("opFuncName", "N/A")
            info["compute_location"] = compute_op.get("location", "N/A")

    return info


def summarize_meta(meta: TranspileMeta) -> list[dict[str, Any]]:
    """Return a list of key-info dicts, one per SDSC file in *meta*.

    This is a convenience wrapper that loads each SDSC file referenced by the
    :class:`TranspileMeta` and runs :func:`extract_key_sdsc_info` on it.
    """
    summaries = []
    for i, path in enumerate(meta.sdsc_paths):
        sdsc_data = meta.load_sdsc(i)
        info = extract_key_sdsc_info(sdsc_data)
        info["sdsc_path"] = path
        summaries.append(info)
    return summaries


def print_meta_summary(meta: TranspileMeta) -> None:
    """Pretty-print a summary of all SDSC files referenced by *meta*.

    When ``meta.tile_dim_to_block_value`` is populated (i.e. a helion config was
    used), each SDSC kernel is checked against the expected core division:
    ``numCoresUsed`` should equal ``prod(numWkSlicesPerDim values)``, and
    the block_sizes that were applied are reported for reference.
    """
    import math

    summaries = summarize_meta(meta)
    if not summaries:
        log.info("No SDSC files found in meta.")
        return

    # tile_dim_to_block_value: {transpiled_node_name: {host_dim: block_size_value}}
    dim_to_bs = getattr(meta, "tile_dim_to_block_value", {}) or {}

    for i, info in enumerate(summaries):
        log.info("--- SDSC kernel %d ---", i)
        for key, value in info.items():
            log.info("  %s: %s", key, value)

        if not dim_to_bs:
            continue

        op_name = info.get("operation", "")
        num_cores_used = info.get("numCoresUsed", "N/A")
        wk_slices = info.get("numWkSlicesPerDim", {})

        # Match SDSC operation to affected FX nodes.
        # SDSC operation (e.g. "add") is a substring of transpiled node
        # names (e.g. "add", "add_1").
        matched = {name: dims for name, dims in dim_to_bs.items() if op_name and op_name in name}

        if not matched:
            log.info("  [config check] not an affected op — used default division")
            continue

        # Gather block_sizes applied to this op
        all_dim_bs: dict[int, int] = {}
        for dims in matched.values():
            all_dim_bs.update(dims)

        expected_from_slices = math.prod(wk_slices.values()) if wk_slices else 1

        log.info("  [config check] block_sizes applied: %s", all_dim_bs)
        log.info("  [config check] numWkSlicesPerDim: %s", wk_slices)

        if num_cores_used == "N/A":
            continue

        if num_cores_used == expected_from_slices:
            log.info(
                "  [config check] PASS: numCoresUsed=%d == prod(numWkSlicesPerDim)=%d",
                num_cores_used,
                expected_from_slices,
            )
        else:
            log.info(
                "  [config check] MISMATCH: numCoresUsed=%d != prod(numWkSlicesPerDim)=%d",
                num_cores_used,
                expected_from_slices,
            )


# Made with Bob
