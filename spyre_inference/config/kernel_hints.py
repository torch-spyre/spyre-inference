# Copyright 2026 The Spyre-Inference Authors.
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

"""Resolve model-configured kernel hints before compilation."""

import math

from spyre_inference import config as model_config


def _row_hints(table, rank, cores):
    if not isinstance(table, dict):
        raise ValueError("Kernel hints must map row counts to core divisions")
    result = {}
    for rows, value in table.items():
        parts = (value,) if rank == 1 else value
        if (
            type(rows) is not int
            or rows <= 0
            or not isinstance(parts, (list, tuple))
            or len(parts) != rank
            or any(type(part) is not int or part <= 0 for part in parts)
            or rows % parts[0]
        ):
            raise ValueError(f"Invalid kernel hint: {rows!r}: {value!r}")
        if math.prod(parts) <= cores:
            result[rows] = tuple(parts)
    return result


def configure_kernel_hints(model, vllm_config):
    config = vllm_config.model_config
    if config is None or getattr(config, "quantization", None) is not None:
        return
    entry = model_config.model_registry().get(config.model)
    if entry is None or not entry.supports_platform(model_config.current_platform()):
        return
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    tables = [c.device_config.kernel_hints for c in entry.serving_configs if c.tp_size == tp_size]
    if not tables:
        return
    if any(table != tables[0] for table in tables[1:]):
        raise ValueError("Kernel hints must agree across context lengths for one model and TP size")
    if not tables[0]:
        return

    from torch_spyre._inductor import config as spyre_config

    from spyre_inference.custom_ops.linear import SpyreTransposedWeightMethod
    from spyre_inference.custom_ops.rotary_embedding import _SpyreRotaryMixin

    hints = tables[0]
    linear = {
        role: _row_hints(table, 2, spyre_config.sencores)
        for role, table in hints.get("linear", {}).items()
    }
    rope = {
        rows: parts[0]
        for rows, parts in _row_hints(hints.get("rope", {}), 1, spyre_config.sencores).items()
    }
    for name, layer in model.named_modules():
        method = getattr(layer, "quant_method", None)
        if isinstance(method, SpyreTransposedWeightMethod):
            requested = linear.get(name.rsplit(".", 1)[-1], {})
            if not requested:
                continue
            weight = getattr(layer, method.WEIGHT_T_ATTR)
            layer.spyre_work_div = {
                rows: parts for rows, parts in requested.items() if weight.shape[1] % parts[1] == 0
            }
        elif isinstance(layer, _SpyreRotaryMixin):
            layer.spyre_work_div = dict(rope)
