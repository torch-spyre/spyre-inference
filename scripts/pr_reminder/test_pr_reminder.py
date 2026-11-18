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

"""Standalone tests for pr_reminder's Slack message rendering.

pr_reminder.py imports requests/slack_sdk at module load for its posting paths,
which the render-only assertions here do not exercise, so those modules are
stubbed before the script is loaded by path (mirroring .github/scripts tests).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_pr_reminder():
    for name in ("requests", "slack_sdk", "slack_sdk.errors", "slack_sdk.webhook"):
        sys.modules.setdefault(name, types.ModuleType(name))
    # Attributes pr_reminder references at import time.
    sys.modules["requests"].Session = object
    sys.modules["slack_sdk"].WebClient = object
    sys.modules["slack_sdk.errors"].SlackClientError = type("SlackClientError", (Exception,), {})
    sys.modules["slack_sdk.webhook"].WebhookClient = object
    spec = importlib.util.spec_from_file_location("pr_reminder", _HERE / "pr_reminder.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pr_reminder = _load_pr_reminder()


def _pr(labels: list[str]) -> dict:
    return {
        "title": "a title",
        "number": 7,
        "html_url": "https://example.com/pr/7",
        "user": {"login": "someone"},
        "updated_at": "2026-01-01T00:00:00Z",
        "labels": [{"name": n} for n in labels],
    }


def test_label_names_are_escaped_for_the_code_span():
    now = datetime(2026, 1, 2, tzinfo=UTC)
    # A label carrying every mrkdwn-significant character: the backtick would
    # close the inline-code span, and <...|...> would render as a Slack link.
    hostile = "`</evil|x>&"
    message = pr_reminder.render_message([_pr([hostile])], total=1, now=now)

    # The raw, unescaped label must not appear verbatim.
    assert f"`{hostile}`" not in message
    # The backtick is dropped so the inline-code span cannot be closed early.
    assert "evil" in message  # the text survives, only the markup is neutralised
    assert "`</evil|x>&`" not in message
    # Link/entity characters are HTML-escaped exactly as titles already are.
    assert "&lt;" in message and "&gt;" in message and "&amp;" in message
    assert "</evil|x>" not in message


def test_plain_label_name_round_trips():
    now = datetime(2026, 1, 2, tzinfo=UTC)
    message = pr_reminder.render_message([_pr(["needs-review"])], total=1, now=now)
    assert "`needs-review`" in message
