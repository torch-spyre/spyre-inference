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
        """Load and return the parsed JSON of the *index*-th SDSC file."""
        import json

        with open(self.sdsc_paths[index], "r") as f:
            return json.load(f)


@contextlib.contextmanager
def _capture_sdsc_paths() -> Generator[TranspileMeta, None, None]:
    """Context manager that monkey-patches
    :meth:`SpyreAsyncCompile.sdsc` to record every ``sdsc.json`` path
    written during compilation.

    Yields a :class:`TranspileMeta` whose ``sdsc_paths`` list is populated
    by the time the context exits.
    """
    from torch_spyre.execution.async_compile import SpyreAsyncCompile

    meta = TranspileMeta()
    original_sdsc = SpyreAsyncCompile.sdsc

    def _patched_sdsc(self, kernel_name, ks):
        runner = original_sdsc(self, kernel_name, ks)
        # The original writes sdsc.json under code_dir/execute/<name>/
        code_dir = getattr(runner, "code_dir", None)
        if code_dir is not None:
            sdsc_path = os.path.join(
                code_dir, "execute", kernel_name, "sdsc.json"
            )
            if os.path.isfile(sdsc_path):
                meta.sdsc_paths.append(sdsc_path)

        return runner

    SpyreAsyncCompile.sdsc = _patched_sdsc
    try:
        yield meta
    finally:
        SpyreAsyncCompile.sdsc = original_sdsc

# Made with Bob
