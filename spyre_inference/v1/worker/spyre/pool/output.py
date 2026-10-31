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

"""MRV2: synchronous pooler output, replacing
vllm/v1/worker/gpu/async_utils.py:AsyncPoolingOutput.

AsyncPoolingOutput defers the device-to-host copy behind a torch.cuda.Event for
later polling; Spyre has no async copy stream worth deferring to, so pool_sync()
does the copy inline and returns the result directly.
"""

from typing import TYPE_CHECKING

from vllm.v1.outputs import ModelRunnerOutput

from spyre_inference.v1.pool.spyre_pooler import copy_pooler_output_to_cpu
from spyre_inference.v1.worker.spyre import device_shims

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


def pool_sync(runner: "SpyreModelRunnerV2") -> ModelRunnerOutput | None:
    """GPUModelRunner.pool, with the pooler output copied to the host inline."""
    state = runner.execute_model_state
    if state is None:
        # The prior execute_model call must have failed.
        return None
    runner.execute_model_state = None
    input_batch = state.input_batch
    kv_connector_output = runner.kv_connector.post_forward(state.finished_req_ids)

    # No `not is_last_pp_rank` branch as upstream has: Spyre rejects PP>1.
    assert runner.pooling_runner is not None and state.hidden_states is not None
    pooler_output, finished_mask = runner.pooling_runner.pool(
        state.hidden_states, input_batch, runner.req_states
    )
    output = ModelRunnerOutput(
        req_ids=input_batch.req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
        kv_connector_output=kv_connector_output,
        ec_connector_output=state.ec_connector_output,
    )
    if pooler_output is None or not any(finished_mask):
        output.pooler_output = [None] * input_batch.num_reqs
    else:
        # V1's Spyre-safe D2H, then upstream's batch-size check and finished-mask split.
        output.pooler_output = copy_pooler_output_to_cpu(pooler_output, finished_mask)
    device_shims.spyre_fence()
    runner.postprocess_num_computed_tokens(input_batch)
    return output
