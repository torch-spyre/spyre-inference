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

"""Guards for tests/mrv2: an autouse fixture makes vllm.LLM.__init__() raise,
and a time budget (1s/test summed across setup+call+teardown, ~20s/directory,
exempting tests marked ``triton_cpu`` or ``subprocess_isolated``) fails the
run if exceeded. The budget alone catches an accidental real torch.compile()
too (its fixed tracing/codegen overhead alone exceeds the per-test budget),
without a dedicated torch.compile guard: monkeypatching it doesn't un-bake
what a module-scope ``@torch.compile`` decorator already returned into
whatever module first got imported during a tests/mrv2 run, so a naive guard
leaks into unrelated, later tests in the same session.

Only applies to items collected under tests/mrv2, so a bare `pytest` from
the repo root never has the budget hook constrain the rest of the suite.
"""

import pathlib
import time

import pytest
from vllm import LLM  # eager: vllm.LLM is a lazy import that would bill the first test's setup

_MRV2_DIR = pathlib.Path(__file__).parent
_PER_TEST_BUDGET_S = 1.0
_TOTAL_BUDGET_S = 20.0

_total_duration = 0.0
_total_budget_reported = False
_item_elapsed: dict[str, float] = {}
_session_fixture_s = 0.0


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(_SessionFixtureTimer(), "mrv2-session-fixture-timer")
    config.addinivalue_line(
        "markers",
        "triton_cpu: JIT-compiles Triton on a cold cache (seconds per kernel); "
        "exempt from the tests/mrv2 time budget",
    )
    config.addinivalue_line(
        "markers",
        "subprocess_isolated: spawns a fresh interpreter so a before/after check "
        "isn't vacuous from an already-warm module cache; exempt from the budget",
    )


@pytest.fixture(autouse=True)
def _no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_llm_init(*args: object, **kwargs: object) -> None:
        raise RuntimeError("tests/mrv2 must not construct vllm.LLM(); use fakes instead")

    monkeypatch.setattr(LLM, "__init__", _raise_llm_init)


def _is_mrv2_item(item: pytest.Item) -> bool:
    return _MRV2_DIR in item.path.parents


class _SessionFixtureTimer:
    """Times session-scoped fixtures, which set up once in whichever test runs first
    (the testing plugin's _spyre_session_config builds a ModelConfig), so that test
    is not billed for the whole session's setup.

    Registered as a global plugin: pytest only calls pytest_fixture_setup on the
    plugins and conftests visible where the fixture is defined, so a hook in this
    conftest never sees the testing plugin's fixtures.
    """

    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self, fixturedef: pytest.FixtureDef):
        start = time.perf_counter()
        yield
        if fixturedef.scope == "session":
            global _session_fixture_s
            _session_fixture_s += time.perf_counter() - start


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Fails the offending test's own report on a budget overrun, rather
    than only flipping the session exit code: a shard's JUnit XML must show
    which test went over, not every test green next to a red job. Sums
    setup+call+teardown, not just call: a slow fixture escapes the budget
    exactly as easily as a slow test body, and only judges once all three
    phases are in."""
    outcome = yield
    report = outcome.get_result()
    global _session_fixture_s
    session_setup, _session_fixture_s = _session_fixture_s, 0.0
    if not _is_mrv2_item(item):
        return
    if "triton_cpu" in item.keywords or "subprocess_isolated" in item.keywords:
        return

    duration = report.duration - session_setup if report.when == "setup" else report.duration
    _item_elapsed[item.nodeid] = _item_elapsed.get(item.nodeid, 0.0) + duration
    if report.when != "teardown":
        return
    elapsed = _item_elapsed.pop(item.nodeid, 0.0)

    global _total_duration, _total_budget_reported
    _total_duration += elapsed

    if elapsed > _PER_TEST_BUDGET_S:
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid} took {elapsed:.2f}s across setup/call/teardown "
            f"(tests/mrv2 budget is {_PER_TEST_BUDGET_S}s per test)"
        )
    elif _total_duration > _TOTAL_BUDGET_S and not _total_budget_reported:
        _total_budget_reported = True
        report.outcome = "failed"
        report.longrepr = (
            f"tests/mrv2 took {_total_duration:.2f}s total "
            f"(budget is {_TOTAL_BUDGET_S}s for the directory)"
        )
