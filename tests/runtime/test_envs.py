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

"""The SPYRE_* environment registry (spyre_inference/envs.py).

Every component reads its knobs through module attributes here, and half the suite
patterns a test as ``monkeypatch.setenv`` + ``envs.clear_env_cache()`` (the autouse
``_clear_env_cache`` fixture in tests/conftest.py does it around every test), yet the
registry's own contract is tested nowhere: values are computed on first read and then
frozen, ``clear_env_cache`` is the only way to re-read, unknown names raise, and the
``TYPE_CHECKING`` declarations the docstring demands stay in sync with the registry.
"""

import ast
from pathlib import Path

import pytest

from spyre_inference import envs


def test_every_registry_entry_has_a_matching_type_checking_declaration():
    """The module docstring demands the declarations stay in sync with the registry.

    A registry key without a declaration loses type-checker and IDE resolution for
    ``envs.SPYRE_...`` reads; a declaration without a key resolves to an attribute
    that raises at runtime. Parse the source so the test names the drift, not just a
    mismatch count.
    """
    source = Path(envs.__file__).read_text()
    tree = ast.parse(source)
    # The `if TYPE_CHECKING:` block holds one `NAME: type = default` per registry key.
    declared = {
        stmt.target.id
        for node in tree.body
        if isinstance(node, ast.If)
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    assert declared == set(envs.environment_variables), (
        f"TYPE_CHECKING declarations {sorted(declared)} and registry keys "
        f"{sorted(envs.environment_variables)} drifted apart"
    )


def test_a_value_is_computed_on_first_read_and_then_cached(monkeypatch):
    """The documented contract: a value is frozen at first read, not re-read per access.

    The serving path relies on this to keep a knob stable across a run even if the
    environment is mutated underneath it.
    """
    monkeypatch.setenv("SPYRE_MOE_CHUNKS", "2")
    assert envs.SPYRE_MOE_CHUNKS == 2
    monkeypatch.setenv("SPYRE_MOE_CHUNKS", "5")
    assert envs.SPYRE_MOE_CHUNKS == 2, "a cached value must not follow the environment"


def test_clear_env_cache_drops_the_cached_values(monkeypatch):
    """The suite's setenv-then-clear pattern only works if the clear really re-reads."""
    monkeypatch.setenv("SPYRE_MOE_CHUNKS", "2")
    assert envs.SPYRE_MOE_CHUNKS == 2
    monkeypatch.setenv("SPYRE_MOE_CHUNKS", "5")
    envs.clear_env_cache()
    assert envs.SPYRE_MOE_CHUNKS == 5


def test_dir_lists_the_registry_keys():
    assert set(dir(envs)) == set(envs.environment_variables)


def test_an_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="module 'spyre_inference.envs'"):
        envs.SPYRE_NOT_A_KNOB  # noqa: B018


@pytest.mark.parametrize(
    ("name", "unset", "value", "parsed"),
    [
        # A str default behind `or`.
        ("SPYRE_MRV2_KERNELS", "host", "triton", "triton"),
        # Optional str: None when unset.
        ("SPYRE_ATTN_QUERY_BUCKETS", None, "1,512", "1,512"),
        # `bool(int(...))`: the default can be truthy.
        ("SPYRE_JAGGED_DECODE", True, "0", False),
        # `int(...)` with a numeric default.
        ("SPYRE_ATTN_MAX_CORES", 0, "7", 7),
        # String compare against "1", not int parsing.
        ("SPYRE_KERNEL_CACHE", False, "1", True),
        # Walrus int-or-None.
        ("SPYRE_MOE_CHUNKS", None, "3", 3),
    ],
)
def test_entries_parse_their_documented_types(monkeypatch, name, unset, value, parsed):
    """One entry per parsing idiom in the registry: default when unset, parsed when set."""
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) == unset
    monkeypatch.setenv(name, value)
    envs.clear_env_cache()
    assert getattr(envs, name) == parsed
