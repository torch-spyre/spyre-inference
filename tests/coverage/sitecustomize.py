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

# Under COVERAGE=1 the Makefile puts this dir on PYTHONPATH so every interpreter
# (pytest + spawned vLLM workers) starts coverage. process_startup() is a no-op
# unless COVERAGE_PROCESS_START is set, so this is inert otherwise. The bare
# except is load-bearing: a raise here would break every interpreter, not just
# coverage.
try:
    import coverage

    coverage.process_startup()
except Exception:
    pass
