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

"""Worker instrumentation for serial, real-model speculative decoding tests."""

import time
import warnings
from functools import wraps

import torch

from spyre_inference.v1.spec_decode.dflash import accepted_context_length
from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker


class ValidationWorker(TorchSpyreWorker):
    def init_device(self):
        super().init_device()
        from torch_spyre.ops.fallbacks import FallbackWarning

        warnings.simplefilter("error", FallbackWarning)

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        self.mode = "real"
        self.references = []
        self.reject_at = -1
        self.steps = []
        self.verification_steps = []
        self._native_propose = self.model_runner.propose_draft_token_ids
        self.model_runner.propose_draft_token_ids = self._propose
        self._native_bookkeeping = self.model_runner._bookkeeping_sync
        self.model_runner._bookkeeping_sync = self._bookkeeping
        self._stats: dict[str, dict[str, float | int]] = {
            "drafter": dict.fromkeys(
                (
                    "proposal_calls",
                    "proposal_seconds",
                    "context_tokens",
                    "context_seconds",
                    "draft_forward_seconds",
                ),
                0,
            ),
            "model": dict.fromkeys(
                (
                    "selection_calls",
                    "device_to_host_seconds",
                    "host_argmax_seconds",
                    "candidate_gather_seconds",
                    "refiner_seconds",
                    "logits_seconds",
                    "logits_transfer_bytes",
                ),
                0,
            ),
        }
        drafter = self.model_runner.drafter
        self._time_call(
            drafter, "propose_spyre", "drafter", "proposal_seconds", count="proposal_calls"
        )
        model = drafter.model
        for name in ("combine_hidden_states", "precompute_and_store_context_kv"):
            self._time_call(model, name, "drafter", "context_seconds")
        self._time_call(model, "forward", "drafter", "draft_forward_seconds")
        self._time_call(model, "compute_device_logits", "model", "logits_seconds")
        self._time_call(
            model,
            "_copy_logits_to_cpu",
            "model",
            "device_to_host_seconds",
            transferred="logits_transfer_bytes",
        )
        self._time_call(
            model, "_argmax_on_cpu", "model", "host_argmax_seconds", count="selection_calls"
        )
        for name, metric in (
            ("_hidden_cache", "refiner_seconds"),
            ("_refine", "refiner_seconds"),
            ("_refine_candidates", "refiner_seconds"),
            ("_gather_readout", "candidate_gather_seconds"),
        ):
            if hasattr(model, name):
                self._time_call(model, name, "model", metric)

    def _time_call(self, owner, name, group, metric, *, count=None, transferred=None):
        original = getattr(owner, name)
        stats = self._stats[group]

        @wraps(original)
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            finally:
                stats[metric] += time.perf_counter() - started
            if count is not None:
                stats[count] += 1
            if transferred is not None:
                stats[transferred] += result.numel() * result.element_size()
            return result

        setattr(owner, name, timed)

    def configure_proposer(self, mode, references=None, reject_at=-1, passes=None, tail_salt=0):
        self.mode = mode
        self.references = references or []
        self.reject_at = reject_at
        self.tail_salt = tail_salt
        self.steps = []
        self.verification_steps = []
        self._last_blocks = None
        self._draft_context_end = 0
        self._last_proposals = []
        if passes is not None:
            self.model_runner.drafter.model.xpress_num_passes = passes

    def _bookkeeping(self, scheduler, *args, **kwargs):
        result = self._native_bookkeeping(scheduler, *args, **kwargs)
        # This also sees final rounds where the next draft would exceed max length.
        for request, sampled in zip(result[5], result[3]):
            self.verification_steps.append(
                dict(
                    request=request,
                    drafts=len(scheduler.scheduled_spec_decode_tokens.get(request, [])),
                    sampled=list(sampled),
                    position=int(
                        self.model_runner._get_positions(scheduler.total_num_scheduled_tokens)[0]
                    ),
                    finished_requests=sorted(scheduler.finished_req_ids),
                )
            )
        return result

    def _propose(
        self, scheduler, sampled, sampling, hidden, sample_hidden, aux, spec, common, slots
    ):
        runner = self.model_runner
        positions = runner._get_positions(scheduler.total_num_scheduled_tokens)
        drafts = spec.num_draft_tokens[0] if spec is not None else 0
        valid = accepted_context_length(len(positions), drafts, len(sampled[0]))
        self._last_blocks = common.block_table_tensor[0].clone()
        row = dict(
            request=runner.input_batch.req_ids[0],
            position=int(positions[0]),
            scheduled=len(positions),
            drafts=drafts,
            sampled=list(sampled[0]),
            committed_end=int(positions[valid - 1]) + 1,
            blocks=self._last_blocks.tolist(),
            previous_proposals=list(self._last_proposals),
        )
        if self.mode in ("real", "real_scripted"):
            result = self._native_propose(
                scheduler, sampled, sampling, hidden, sample_hidden, aux, spec, common, slots
            )
            self._draft_context_end = row["committed_end"]
            self._stats["drafter"]["context_tokens"] += valid
            row["native_proposals"] = list(result[0])
        if (
            self.mode == "none"
            or not sampled[0]
            or row["committed_end"] + runner.num_spec_tokens + 1
            > runner.effective_drafter_max_model_len
        ):
            result = [[]]
        elif self.mode in ("scripted", "real_scripted"):
            count = int(runner.input_batch.num_tokens_no_spec[0])
            tokens = runner.input_batch.token_ids_cpu[0, :count].tolist()
            candidates = [
                r for r in self.references if tokens[: len(r["prompt_ids"])] == r["prompt_ids"]
            ]
            assert candidates
            reference = max(candidates, key=lambda r: len(r["prompt_ids"]))
            offset = len(tokens) - len(reference["prompt_ids"])
            known = min(offset, len(reference["token_ids"]))
            assert tokens[len(reference["prompt_ids"]) :][:known] == reference["token_ids"][:known]
            size = runner.num_spec_tokens
            draft = reference["token_ids"][offset : offset + size]
            draft += [0] * (size - len(draft))
            if 0 <= self.reject_at < size:
                draft[self.reject_at] = (
                    draft[self.reject_at] + 1
                ) % runner.model_config.get_vocab_size()
                for i in range(self.reject_at + 1, size):
                    draft[i] = (draft[i] + self.tail_salt) % runner.model_config.get_vocab_size()
            result = [draft]
        row["proposals"] = result[0]
        self._last_proposals = list(result[0])
        self.steps.append(row)
        return result

    def validation_state(self):
        memory = torch.spyre.memory.memory_stats(0)
        return dict(
            steps=self.steps,
            verification_steps=self.verification_steps,
            runner_requests=sorted(self.model_runner.requests),
            draft_context_end=self._draft_context_end,
            memory={
                "allocated_bytes": memory["allocated_bytes.all.current"],
                "peak_allocated_bytes": memory["allocated_bytes.all.peak"],
            },
            drafter=dict(self._stats["drafter"]),
            model=dict(self._stats["model"]),
        )

    def save_cache_prefix(self, path, length, include_draft=False):
        runner = self.model_runner
        caches = runner._spyre_kv_caches
        draft_names = runner.drafter._draft_attn_layer_names
        order = lambda name: int(name.split(".")[2])
        target = sorted(caches.keys() - draft_names, key=order)
        draft = sorted(draft_names, key=order) if include_draft else []
        chosen = [(name, length) for name in target]
        if draft:
            assert self._draft_context_end > 0
            chosen.extend((name, min(length, self._draft_context_end)) for name in draft)
        result = {}
        for name, n in chosen:
            cache = caches[name]
            block_size = cache.k_pages.shape[2]
            ids = self._last_blocks[: (n + block_size - 1) // block_size].long()
            result[name] = tuple(
                pages.to(device="cpu", dtype=torch.float32)[ids]
                .permute(0, 2, 1, 3)
                .flatten(0, 1)[:n]
                .contiguous()
                for pages in cache
            )
        torch.save(result, path)
        return dict(lengths=dict(chosen), draft_context_end=self._draft_context_end)
