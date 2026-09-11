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

import torch
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier

logger = init_logger(__name__)


class SpyreOffloadingWorker(OffloadingWorker):
    def __init__(self, kv_caches, pool):
        self._finished_jobs: list[TransferResult] = []
        self._kv_caches = kv_caches
        self._pool = pool
        self._copier = SpyreKvDmaCopier()

        # TODO: We are currently using a temporary tensor to store the data in
        # kv cache because copy h2d or d2h doesn't support copying directly
        # from/to the kv cache tensor. The direct copy is being worked on.
        self._temp = torch.empty_like(kv_caches.tensors[0].tensor[0], device="spyre")

    def _slot_id(self, block_id: int, tensor_id: int) -> int:
        """
        Get the slot id for a given block id.
        """
        return block_id * len(self._kv_caches.tensors) + tensor_id

    def _transfer(
        self, host_spec: LoadStoreSpec, gpu_spec: GPULoadStoreSpec, to_device: bool
    ) -> None:
        """
        Copy for host to device or device to host.
        """
        for dev_blk_id, host_blk_id in zip(gpu_spec.block_ids, host_spec.block_ids):
            for tensor_id, cache in enumerate(self._kv_caches.tensors):
                slot_id = self._slot_id(host_blk_id, tensor_id)
                if to_device:
                    self._copier.copy_h2d(self._temp, self._pool, slot_id)
                    cache.tensor[dev_blk_id].copy_(self._temp)
                else:
                    self._temp.copy_(cache.tensor[dev_blk_id])
                    self._copier.copy_d2h(self._temp, self._pool, slot_id)

    def _run(
        self, job_id: int, host_spec: LoadStoreSpec, gpu_spec: GPULoadStoreSpec, to_device: bool
    ) -> bool:
        """
        Run the transfer job and record the result.
        """
        try:
            self._transfer(host_spec, gpu_spec, to_device=to_device)
            self._finished_jobs.append(TransferResult(job_id=job_id, success=True))

        except Exception:
            logger.exception("Failed to run job %d", job_id)
            self._finished_jobs.append(TransferResult(job_id=job_id, success=False))

        # Report the job was accepted. We would only return False if the job was
        # rejected due to resource constraints, but this implementation does not
        # have such constraints.
        return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """
        Start an async copy for device to host.
        """
        return self._run(job_id, dst_spec, src_spec, to_device=False)

    def submit_load(self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec) -> bool:
        """
        Start an async copy for host to device.
        """
        return self._run(job_id, src_spec, dst_spec, to_device=True)

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
