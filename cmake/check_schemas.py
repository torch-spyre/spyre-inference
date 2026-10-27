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
"""Usage: check_schemas.py <ours.cpp> <vllm csrc/cpu/torch_bindings.cpp>"""

import re
import sys

# Adjacent string literals concatenate, as in C++.
_DEF = re.compile(r'ops\.def\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)')
_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')


def schemas(path: str) -> dict[str, str]:
    with open(path) as f:
        src = f.read()
    out = {}
    for m in _DEF.finditer(src):
        schema = "".join(_LITERAL.findall(m.group(1))).replace('\\"', '"')
        out[schema.split("(", 1)[0]] = schema
    return out


def main(ours_path: str, upstream_path: str) -> int:
    ours, upstream = schemas(ours_path), schemas(upstream_path)
    errors = [
        f"  {name}:\n    ours:   {schema}\n    vLLM's: {upstream.get(name, '<missing>')}"
        for name, schema in ours.items()
        if upstream.get(name) != schema
    ]
    if not ours:
        errors.append(f"  no ops.def(...) found in {ours_path}")
    if errors:
        print("Op schemas differ from vLLM's:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
