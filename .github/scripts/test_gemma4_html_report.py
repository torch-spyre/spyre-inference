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

"""gemma4-benchmark's html_report.py: the chart-data script embedding. Kept here so
the pre-commit ingest-script-tests job runs it (it is not spyre-inference product code).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / ".claude/skills/gemma4-benchmark-detailed/scripts/html_report.py"
)

_spec = importlib.util.spec_from_file_location("gemma4_html_report", _SCRIPT)
html_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(html_report)


def test_charts_js_escapes_closing_script_tag():
    payload = {"rows": [{"name": "branch</script><img src=x onerror=alert(1)>"}]}
    embedded = html_report.charts_js(payload)
    assert "</script>" not in embedded
    assert "<\\/" in embedded
    # The escape is JSON-decodable back to the original payload.
    assert json.loads(embedded.replace("<\\/", "</")) == payload


def test_charts_js_safe_labels_unchanged():
    payload = {"rows": [{"name": "main"}, {"name": "hf-adapters"}]}
    assert html_report.charts_js(payload) == '{"rows": [{"name": "main"}, {"name": "hf-adapters"}]}'


def _fixture_suite(top: pathlib.Path, branch_name: str) -> pathlib.Path:
    """One minimal run_1102.sh-shaped run whose branch arm carries the payload."""
    out = top / "suite" / "run0" / "out_8"
    out.mkdir(parents=True)
    (out.parent / "workload.json").write_text('{"tp": 1, "itl": false}')
    (out / "meta.json").write_text('{"arm": "vllm", "rep": 0, "output_len": 8}')
    # The branch name reaches report.labels() -> the plain label -> CHARTS.
    (out.parent / "env_branch.json").write_text(
        json.dumps({"spyre-inference": {"branch": branch_name, "commit": "0123456789abcdef"}})
    )
    (out / "latency.json").write_text('{"latencies": [1.0, 1.0, 1.0]}')
    return top / "suite"


def test_page_embeds_escaped_charts(tmp_path):
    branch_name = "evil</script><img src=x onerror=alert(1)>"
    suite = _fixture_suite(tmp_path, branch_name)
    page = html_report.page(suite, {"title": "t", "footer": "f"})
    script_start = page.index("var CHARTS = ")
    script_end = page.index("</script>", script_start)
    charts = page[script_start:script_end]
    # The payload label is inside CHARTS, but never as a raw closing tag.
    assert branch_name not in charts
    assert "</script>" not in charts
    body = charts[len("var CHARTS = ") :]
    data, _ = json.JSONDecoder().raw_decode(body)
    assert any(
        branch_name in str(v)
        for chart in data.values()
        for row in chart
        for r in row["rows"]
        for v in r.values()
    )
