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


from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier

# M1-S1: In progress on other branch:


logger = init_logger(__name__)


class SpyreOffloadingWorker(OffloadingWorker):
    def __init__(self, kv_caches, pool):
        self._finished_jobs: list[TransferResult] = []
        self._kv_caches = kv_caches
        self._pool = pool
        self._copier = SpyreKvDmaCopier()

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """
        Start an async copy for device to host.
        """
        for device_blk_id, slot_id in zip(src_spec.block_ids, dst_spec.block_ids):
            self._copier.copy_d2h(self._kv_caches[device_blk_id], self._pool, slot_id)

        self._finished_jobs.append(TransferResult(job_id=job_id, success=True))
        return True

    def submit_load(self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec) -> bool:
        """
        Start an async copy for host to device.
        """
        for slot_id, device_blk_id in zip(src_spec.block_ids, dst_spec.block_ids):
            self._copier.copy_h2d(self._kv_caches[device_blk_id], self._pool, slot_id)

        self._finished_jobs.append(TransferResult(job_id=job_id, success=True))
        return True

    def get_finished(self) -> list[TransferResult]:
        """
        Returns all the TransferResults the jobs that are completed.
        """
        finished_jobs = self._finished_jobs
        self._finished_jobs = []
        return finished_jobs

    def wait(self, job_ids: set[int]) -> None:
        """
        Block until those specific job ids are done.
        """
        # In this implementation, we assume that all jobs are completed immediately
        # after submission, so we don't need to do any waiting.
        pass
