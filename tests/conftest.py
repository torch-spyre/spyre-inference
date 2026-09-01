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

import pytest

from spyre_inference import envs


@pytest.fixture(autouse=True)
def _clear_env_cache():
    """envs.py caches each SPYRE_* value on first read; drop the cache around
    every test so monkeypatched vars take effect and don't leak between tests."""
    envs.clear_env_cache()
    yield
    envs.clear_env_cache()
