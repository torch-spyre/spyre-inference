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

from __future__ import annotations

from typing import TYPE_CHECKING

import spyre_inference.envs as envs_spyre
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
    )
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class SpyreKVConnectorBridge:
    def __init__(self, vllm_config: VllmConfig):
        self._vllm_config = vllm_config
        self._kv_connector: KVConnectorBase_V1 | None = None
        self._active = False
        self._output: KVConnectorOutput | None = None
        self._try_acquire_connector()

    def _try_acquire_connector(self) -> bool:
        if self._kv_connector is not None:
            return True
        if not has_kv_transfer_group():
            return False

        connector = get_kv_transfer_group()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )

        if not isinstance(connector, KVConnectorBase_V1):
            logger.warning(
                "KV connector is not a V1 connector (got %s). Bridge disabled.",
                type(connector).__name__,
            )
            return False

        self._kv_connector = connector
        return True

    @property
    def is_available(self) -> bool:
        return self._kv_connector is not None or self._try_acquire_connector()

    @property
    def uses_heap_kv(self) -> bool:
        if not self.is_available:
            return False
        assert self._kv_connector is not None
        active_fn = getattr(self._kv_connector, "heap_kv_active", None)
        if callable(active_fn):
            return bool(active_fn())
        return bool(getattr(self._kv_connector, "uses_heap_kv", False))

    def begin_step(self, scheduler_output: SchedulerOutput) -> bool:
        self._active = False
        self._output = None

        if not self.is_available:
            return False

        assert self._kv_connector is not None
        if scheduler_output.kv_connector_metadata is None:
            return False

        # vLLM 0.20.x: handle_preemptions takes the connector metadata,
        # not the preempted request ids.
        preempted = getattr(scheduler_output, "preempted_req_ids", None)
        if preempted:
            self._kv_connector.handle_preemptions(scheduler_output.kv_connector_metadata)

        self._active = True
        return True

    def before_forward(self, scheduler_output: SchedulerOutput) -> None:
        if not self._active:
            return

        assert self._kv_connector is not None
        assert scheduler_output.kv_connector_metadata is not None

        self._kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

        if is_forward_context_available():
            self._kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self._vllm_config):
                self._kv_connector.start_load_kv(get_forward_context())

    def after_forward(
        self,
        scheduler_output: SchedulerOutput,
        wait_for_save: bool = True,
    ) -> None:
        if not self._active:
            return

        assert self._kv_connector is not None
        output = KVConnectorOutput()
        if wait_for_save:
            self._kv_connector.wait_for_save()

        output.finished_sending, output.finished_recving = self._kv_connector.get_finished(
            scheduler_output.finished_req_ids
        )
        output.invalid_block_ids = self._kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self._kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self._kv_connector.get_kv_connector_kv_cache_events()
        self._output = output

    def finish_step(self) -> KVConnectorOutput | None:
        if not self._active:
            return None

        assert self._kv_connector is not None
        self._kv_connector.clear_connector_metadata()
        output = self._output
        self._output = None
        self._active = False
        return output

    def no_forward(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorOutput | None:
        if not self.begin_step(scheduler_output):
            return None
        self.before_forward(scheduler_output)
        self.after_forward(scheduler_output, wait_for_save=False)
        return self.finish_step()


def maybe_create_bridge(
    vllm_config: VllmConfig,
) -> SpyreKVConnectorBridge | None:
    if not envs_spyre.VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE:
        return None

    if (
        vllm_config.kv_transfer_config is None
        or not vllm_config.kv_transfer_config.is_kv_transfer_instance
    ):
        return None

    return SpyreKVConnectorBridge(vllm_config)
