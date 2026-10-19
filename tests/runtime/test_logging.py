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

"""Spyre loggers under vLLM's lazy logging configuration.

Each case runs in a fresh interpreter: ``dictConfig`` rewrites process-wide logging
state that the rest of the suite (caplog included) depends on.
"""

import json
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent(
    """
    import sys

    import spyre_inference  # noqa: F401
    from vllm.config.logging import LoggingConfig
    from vllm.logger import configure_logging, init_logger

    logger = init_logger("spyre_inference.platform")
    configure_logging(LoggingConfig(pylogging_config_file=sys.argv[1] or None))
    assert not logger.disabled, "spyre_inference.platform was disabled"
    logger.warning("spyre-marker")
    """
)


def _run(config_file: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, config_file],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout + proc.stderr


def test_default_config_routes_spyre_records_through_vllm_handler() -> None:
    out = _run("")
    line = next(ln for ln in out.splitlines() if "spyre-marker" in ln)
    # The default format needs the field vLLM's record factory adds; a record formatted
    # without it would surface as a "--- Logging error ---" instead.
    assert "Logging error" not in out
    assert "WARNING" in line


@pytest.mark.parametrize("names_spyre_logger", [False, True])
def test_custom_config_keeps_spyre_loggers_enabled(tmp_path, names_spyre_logger: bool) -> None:
    # No disable_existing_loggers key: dictConfig defaults it to True and disables every
    # existing logger outside the configured names' dotted namespaces.
    config = {
        "version": 1,
        "formatters": {"plain": {"format": "%(name)s %(levelname)s %(message)s"}},
        "handlers": {
            "out": {
                "class": "logging.StreamHandler",
                "formatter": "plain",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {"vllm": {"handlers": ["out"], "level": "INFO", "propagate": False}},
    }
    if names_spyre_logger:
        # `propagate` left at its default: our own handler must not also feed vllm's.
        config["loggers"]["spyre_inference"] = {"handlers": ["out"], "level": "INFO"}
    path = tmp_path / "logging.json"
    path.write_text(json.dumps(config))

    assert _run(str(path)).count("spyre_inference.platform WARNING spyre-marker") == 1
