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

"""Encoder / pooling helpers kept out of ``TorchSpyreModelRunner``."""

from spyre_inference.v1.pool.spyre_pooler import (
    TOKEN_POOLING_TASKS,
    configure_pooling_for_spyre,
    copy_pooler_output_to_cpu,
    select_rows,
)

__all__ = [
    "TOKEN_POOLING_TASKS",
    "configure_pooling_for_spyre",
    "copy_pooler_output_to_cpu",
    "select_rows",
]
