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

import json
from json.decoder import JSONDecodeError
from typing import Any


def read_benchmark_results(filepath: str) -> list[dict[str, Any]]:
    """Read benchmark results from a JSON file (standard or JSONEachRow format)."""
    results = []
    with open(filepath) as f:
        try:
            r = json.load(f)
            if isinstance(r, dict):
                results.append(r)
            elif isinstance(r, list):
                results = r
        except JSONDecodeError:
            f.seek(0)
            for line in f:
                try:
                    r = json.loads(line)
                    if isinstance(r, dict):
                        results.append(r)
                    elif isinstance(r, list):
                        results.extend(r)
                except JSONDecodeError:
                    pass
    return results
