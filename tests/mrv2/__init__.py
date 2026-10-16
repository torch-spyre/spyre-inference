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

# Makes tests/mrv2 a real package, unlike every other tests/ subdirectory.
# Without it, pytest imports this conftest.py under the same bare module
# name ("conftest") as the real tests/conftest.py, since neither has a
# package ancestor; the second import silently replaces the first in
# sys.modules, and anything that depends on the original being the active
# plugin for new sessions started in-process (e.g. tests/test_tags.py's
# pytester-based self-test of tests/conftest.py's own hooks) breaks.
