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

"""``spyre_available()`` allocates on the card, opening /dev/vfio in the caller.

A test module that calls it at import time (a ``skipif`` condition, most
temptingly) claims the card during collection, before the plugin's
subprocess-first ordering can run the engine tests; their workers then fail to
open the device ("Device or resource busy") in every shard. The guard belongs
inside the test or a fixture, where it runs after the subprocess tests."""

import ast
from pathlib import Path

TESTS_DIR = Path(__file__).parent


def _import_time_calls(tree: ast.AST, name: str) -> list[ast.Call]:
    """Call nodes to ``name`` that evaluate while the module imports: module and
    class bodies, plus function decorators and argument defaults. A call inside a
    function body only runs when the test does, which is the safe place."""
    hits: list[ast.Call] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            hits.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                for decorator in child.decorator_list:
                    visit(decorator)
                args = child.args
                for default in [
                    *args.defaults,
                    *[d for d in args.kw_defaults if d is not None],
                ]:
                    visit(default)
            elif isinstance(child, ast.Lambda):
                continue
            else:
                visit(child)

    visit(tree)
    return hits


def test_no_test_module_claims_the_spyre_card_at_import_time():
    offenders = []
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        calls = _import_time_calls(ast.parse(path.read_text()), "spyre_available")
        if calls:
            offenders.append(f"{path.relative_to(TESTS_DIR)}:{calls[0].lineno}")
    assert not offenders, (
        "spyre_available() called at import time (claims the Spyre card during "
        "collection, before subprocess tests can run) in: " + ", ".join(offenders)
    )
