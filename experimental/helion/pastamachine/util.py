# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
pastamachine.util
~~~~~~~~~~~~~~~~~

Metadata container returned by transpilation when ``return_meta=True``.
"""

from __future__ import annotations

import contextlib
import glob
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

if TYPE_CHECKING:
    import torch.fx


@dataclass
class TranspileMeta:
    """Metadata collected during a :func:`compile_helion_to_spyre` call.

    Attributes
    ----------
    sdsc_paths : list[str]
        Filesystem paths to the ``sdsc.json`` files produced by the Spyre
        compiler for each fused kernel in the compiled FX graph.
    graph_module : torch.fx.GraphModule | None
        The intermediate FX graph module (aten ops) before Spyre compilation.
    block_size_nodes : list[torch.fx.Node]
        FX nodes representing Helion block-size parameters in the graph.
    tile_dim_to_block_index : dict[str, dict[int, int]]
        Unresolved mapping from transpiled FX node name to
        ``{host_dim: block_size_index}``, as extracted from the helion AST.
    tile_dim_to_block_value : dict[str, dict[int, int]]
        Resolved mapping from transpiled FX node name to
        ``{host_dim: block_size_value}``, after merging with the helion config.
    """

    sdsc_paths: List[str] = field(default_factory=list)
    graph_module: Optional[torch.fx.GraphModule] = None
    block_size_nodes: List[torch.fx.Node] = field(default_factory=list)
    tile_dim_to_block_index: Dict[str, Dict[int, int]] = field(default_factory=dict)
    tile_dim_to_block_value: Dict[str, Dict[int, int]] = field(default_factory=dict)

    def load_sdsc(self, index: int = 0) -> Dict[str, Any]:
        """Load and return the parsed JSON of the *index*-th SDSC file.

        With the new torch-spyre, each kernel produces ``sdsc_0.json``,
        ``sdsc_1.json``, … directly in the kernel output directory.
        """
        import json

        with open(self.sdsc_paths[index], "r") as f:
            return json.load(f)


@contextlib.contextmanager
def _capture_sdsc_paths() -> Generator[TranspileMeta, None, None]:
    """Context manager that monkey-patches
    :meth:`SpyreAsyncCompile.sdsc` to record every ``sdsc.json`` path
    written during compilation.

    Yields a :class:`TranspileMeta` whose ``sdsc_paths`` list is populated
    by the time the context exits.  Also ensures torch-spyre inductor logging
    is enabled so that compilation artifacts are emitted regardless of the
    caller's environment.
    """
    from torch_spyre.execution.async_compile import SpyreAsyncCompile

    meta = TranspileMeta()
    original_sdsc = SpyreAsyncCompile.sdsc

    # Enable torch-spyre inductor logging for the duration of compilation
    # so that SDSC generation proceeds with full diagnostics.
    prev_log = os.environ.get("SPYRE_INDUCTOR_LOG")
    prev_level = os.environ.get("SPYRE_INDUCTOR_LOG_LEVEL")
    os.environ["SPYRE_INDUCTOR_LOG"] = "1"
    os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "INFO")

    def _patched_sdsc(self, kernel_name, specs):
        runner = original_sdsc(self, kernel_name, specs)
        code_dir = getattr(runner, "code_dir", None)
        if code_dir is not None:
            for path in sorted(glob.glob(os.path.join(code_dir, "sdsc_*.json"))):
                meta.sdsc_paths.append(path)
        return runner

    SpyreAsyncCompile.sdsc = _patched_sdsc
    try:
        yield meta
    finally:
        SpyreAsyncCompile.sdsc = original_sdsc
        if prev_log is None:
            os.environ.pop("SPYRE_INDUCTOR_LOG", None)
        else:
            os.environ["SPYRE_INDUCTOR_LOG"] = prev_log
        if prev_level is None:
            os.environ.pop("SPYRE_INDUCTOR_LOG_LEVEL", None)
        else:
            os.environ["SPYRE_INDUCTOR_LOG_LEVEL"] = prev_level

# Made with Bob
