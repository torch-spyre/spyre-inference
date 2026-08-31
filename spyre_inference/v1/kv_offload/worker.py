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

# M1-S1: In progress on other branch:


logger = init_logger(__name__)


class SpyreOffloadingWorker(OffloadingWorker):
    def submit_store(job_id, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec) -> bool:
        """
        Start an async copy for device to host.
        """
        pass

    def submit_load(job_id, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec) -> bool:
        """
        Start an async copy for host to device.
        """
        pass

    def get_finished(self) -> list[TransferResult]:
        """
        Returns all the TransferResults the jobs that are completed.
        """
        pass

    def wait(job_ids: set[int]) -> None:
        """
        Block until those specific job ids are done.
        """
        pass
