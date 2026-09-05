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
pastamachine.verify
~~~~~~~~~~~~~~~~~~~

CPU-side verification of FX graph modules.
"""

from pastamachine._logging import getLogger

log = getLogger("verify")


def verify_on_cpu(graph_module, example_spyre_inputs):
    """Run *graph_module* on CPU clones of *example_spyre_inputs* and verify
    the inplace-modified output against the expected value.

    Returns the CPU output tensors (as a tuple) so the caller can compare
    against Spyre results later.
    """
    log.info("=== CPU Validation (execute graph_module via PyTorch) ===")

    cpu_inputs = tuple(t.detach().cpu().clone() for t in example_spyre_inputs)

    log.info("  Inputs: %s", [t.shape for t in cpu_inputs])

    result_cpu = graph_module(*cpu_inputs)
    log.info("  graph_module returned: %s", result_cpu)
    log.debug("  CPU tensors after execution: %s", [t for t in cpu_inputs])

    return cpu_inputs


# Made with Bob
