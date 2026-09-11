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

"""Unit tests for spyre_inference/__init__.py registration functions.

Tests cover register(), register_ops(), and register_hf_adapters() — verifying
that they don't crash, handle errors gracefully, and produce expected side
effects.
"""

import importlib
from unittest.mock import patch, MagicMock

import pytest


class TestRegisterFunction:
    """Tests for register() — platform registration."""

    def test_returns_platform_class_path(self):
        """register() returns the TorchSpyrePlatform class path."""
        import spyre_inference

        result = spyre_inference.register()
        assert result == "spyre_inference.platform.TorchSpyrePlatform"

    def test_return_type_is_string(self):
        """register() returns a string."""
        import spyre_inference

        result = spyre_inference.register()
        assert isinstance(result, str)


class TestRegisterOps:
    """Tests for register_ops() — custom op registration."""

    def test_register_ops_does_not_raise(self):
        """register_ops() runs without error."""
        import spyre_inference

        # Should not raise
        spyre_inference.register_ops()

    def test_register_ops_idempotent(self):
        """register_ops() can be called multiple times safely."""
        import spyre_inference

        spyre_inference.register_ops()
        spyre_inference.register_ops()  # No error


class TestRegisterHfAdapters:
    """Tests for register_hf_adapters() — HF adapters registration."""

    def test_registers_successfully(self):
        """register_hf_adapters() calls ModelRegistry.register_model."""
        import spyre_inference

        with patch(
            "spyre_inference.ModelRegistry.register_model"
        ) as mock_register:
            spyre_inference.register_hf_adapters()
            mock_register.assert_called_once_with(
                "TransformersForCausalLM",
                "spyre_inference.hf_adapters:HfAdaptersForCausalLM",
            )

    def test_exception_is_swallowed_with_warning(self):
        """register_hf_adapters() catches exceptions and warns (no crash)."""
        import spyre_inference

        with patch(
            "spyre_inference.ModelRegistry.register_model",
            side_effect=RuntimeError("test error"),
        ):
            # Should NOT raise — the exception is caught internally
            spyre_inference.register_hf_adapters()

    def test_import_error_swallowed(self):
        """ImportError in ModelRegistry import is caught."""
        import spyre_inference

        with patch(
            "spyre_inference.ModelRegistry.register_model",
            side_effect=ImportError("no module"),
        ):
            # Should NOT raise
            spyre_inference.register_hf_adapters()


class TestVersionAttribute:
    """Tests for package metadata."""

    def test_version_is_set(self):
        """__version__ is set to a non-empty string."""
        import spyre_inference

        assert hasattr(spyre_inference, "__version__")
        assert isinstance(spyre_inference.__version__, str)
        assert len(spyre_inference.__version__) > 0


class TestAutoloadEnvVar:
    """Tests for TORCH_DEVICE_BACKEND_AUTOLOAD deferral."""

    def test_autoload_disabled(self):
        """TORCH_DEVICE_BACKEND_AUTOLOAD is set to 0."""
        import os

        # The import of spyre_inference sets this
        assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
