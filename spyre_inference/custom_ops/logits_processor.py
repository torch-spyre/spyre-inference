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


from vllm.model_executor.layers.logits_processor import LogitsProcessor

from .utils import convert


@LogitsProcessor.register_oot(name="LogitsProcessor")
class SpyreLogitsProcessor(LogitsProcessor):
    def _get_logits(self, hidden_states, lm_head, embedding_bias):
        """Ensure logits are returned on CPU for downstream sampling.

        We override _get_logits rather than _gather_logits because upstream
        only calls _gather_logits when lm_head.tp_size > 1. With TP=1 the
        gather is skipped entirely, leaving logits on Spyre. The sampler
        then crashes on logits.to(torch.float32) since Spyre does not
        support that dtype cast. By overriding at this level we guarantee
        the D2H transfer regardless of TP configuration.
        """
        return convert(super()._get_logits(hidden_states, lm_head, embedding_bias), device="cpu")
