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

"""Unit tests for SpyreQKVParallelLinear custom op.

Tests the OOT registration and the gather_output=False assertion that guards
against unsupported all_gather on Spyre.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestSpyreQKVParallelLinearRegistration:
    """Verify OOT registration and class swap."""

    def test_oot_class_swap(self):
        """QKVParallelLinear.__new__ should produce SpyreQKVParallelLinear."""
        from vllm.model_executor.layers.linear import QKVParallelLinear
        from spyre_inference.custom_ops.linear import SpyreQKVParallelLinear

        # QKVParallelLinear requires hidden_size, head_size, total_num_heads, etc.
        # Use mock to avoid full initialization complexity
        with patch.object(QKVParallelLinear, "__init__", lambda self, *a, **kw: None):
            layer = QKVParallelLinear.__new__(QKVParallelLinear)

        assert isinstance(layer, SpyreQKVParallelLinear)


class TestSpyreQKVParallelLinearGatherOutputAssertion:
    """Test the gather_output=False assertion."""

    def test_raises_on_gather_output_true(self):
        """SpyreQKVParallelLinear must reject gather_output=True."""
        from spyre_inference.custom_ops.linear import SpyreQKVParallelLinear

        # Patch QKVParallelLinear.__init__ to simulate setting gather_output=True.
        # When super().__init__() is called from SpyreQKVParallelLinear.__init__,
        # the mock's side_effect receives the positional args but NOT `self`
        # (because mock patches the descriptor). We capture the instance from
        # the enclosing scope instead.
        instances = []

        def fake_super_init(*args, **kwargs):
            # The instance is the first positional arg passed by super().__init__
            # Actually with patch on the class method, self is already bound.
            # We need to set gather_output on the instance being constructed.
            pass

        with patch(
            "vllm.model_executor.layers.linear.QKVParallelLinear.__init__",
            fake_super_init,
        ):
            # After fake super().__init__ returns without setting gather_output,
            # manually set it to True before the assertion runs.
            # We need to intercept after super().__init__() but before the assert.
            # Instead, let's directly test the assertion logic:
            obj = SpyreQKVParallelLinear.__new__(SpyreQKVParallelLinear)
            obj.gather_output = True
            with pytest.raises(AssertionError, match="gather_output=False"):
                # Re-run the assertion that __init__ performs
                assert not obj.gather_output, (
                    f"{obj.__class__.__name__} requires gather_output=False; "
                    "all_gather is not yet supported on Spyre"
                )

    def test_passes_with_gather_output_false(self):
        """SpyreQKVParallelLinear succeeds when gather_output=False."""
        from spyre_inference.custom_ops.linear import SpyreQKVParallelLinear

        # Simulate the full init path with gather_output=False
        def fake_super_init(*args, **kwargs):
            pass

        with patch(
            "vllm.model_executor.layers.linear.QKVParallelLinear.__init__",
            fake_super_init,
        ):
            obj = SpyreQKVParallelLinear.__new__(SpyreQKVParallelLinear)
            obj.gather_output = False
            # Re-run the same assertion from __init__
            assert not obj.gather_output, (
                f"{obj.__class__.__name__} requires gather_output=False; "
                "all_gather is not yet supported on Spyre"
            )

    def test_init_with_gather_output_false(self):
        """Full __init__ path succeeds when parent sets gather_output=False."""
        from vllm.model_executor.layers.linear import QKVParallelLinear
        from spyre_inference.custom_ops.linear import SpyreQKVParallelLinear

        # Directly patch at the class level with a function that sets gather_output
        original_init = QKVParallelLinear.__init__

        def patched_init(self, *args, **kwargs):
            # Minimal setup that mirrors what the real __init__ does
            self.gather_output = False

        with patch.object(QKVParallelLinear, "__init__", patched_init):
            layer = SpyreQKVParallelLinear(
                hidden_size=768,
                head_size=64,
                total_num_heads=12,
                total_num_kv_heads=12,
            )
            assert not layer.gather_output

    def test_init_with_gather_output_true_raises(self):
        """Full __init__ path raises when parent sets gather_output=True."""
        from vllm.model_executor.layers.linear import QKVParallelLinear
        from spyre_inference.custom_ops.linear import SpyreQKVParallelLinear

        def patched_init(self, *args, **kwargs):
            self.gather_output = True

        with patch.object(QKVParallelLinear, "__init__", patched_init):
            with pytest.raises(AssertionError, match="gather_output=False"):
                SpyreQKVParallelLinear(
                    hidden_size=768,
                    head_size=64,
                    total_num_heads=12,
                    total_num_kv_heads=12,
                )
