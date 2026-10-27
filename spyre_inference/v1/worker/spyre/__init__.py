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

"""Spyre Model Runner V2 (MRV2) implementation. MRV2 ONLY.

Subclasses vLLM's MRV2 runner and its components (vllm/v1/worker/gpu/).
Nothing here is used by the V1 runner, and outside this package only
spyre_inference/__init__.py and platform.py import it, for the Triton gate.
The V1 runner is spyre_model_runner.py / spyre_worker.py, next
to this package; those files are deleted when V2 becomes the default (#906).
"""
