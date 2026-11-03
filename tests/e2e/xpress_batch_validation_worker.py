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

"""Observe packed request boundaries and capture caches before finished pages are freed."""

import hashlib
from pathlib import Path

import torch
from xpress_validation_worker import ValidationWorker

from spyre_inference.v1.spec_decode.dflash import accepted_context_length


class BatchValidationWorker(ValidationWorker):
    def configure_proposer(
        self, mode, references=None, reject_at=-1, passes=None, tail_salt=0, capture_dir=None
    ):
        super().configure_proposer(mode, references, reject_at, passes, tail_salt)
        self.batch_steps = []
        self.snapshots = {}
        self._prompt_caches = {}
        self.capture_dir = Path(capture_dir) if capture_dir else None
        if self.capture_dir:
            self.capture_dir.mkdir(parents=True, exist_ok=True)

    def _propose(
        self, scheduler, sampled, sampling, hidden, sample_hidden, aux, spec, common, slots
    ):
        runner = self.model_runner
        positions = runner._get_positions(scheduler.total_num_scheduled_tokens)
        starts = common.query_start_loc_cpu.tolist()
        rows = []
        for index, (request_id, tokens) in enumerate(zip(runner.input_batch.req_ids, sampled)):
            start, end = starts[index : index + 2]
            drafts = spec.num_draft_tokens[index] if spec is not None else 0
            valid = accepted_context_length(end - start, drafts, len(tokens))
            request = runner.requests[request_id]
            params = request.sampling_params
            rows.append(
                dict(
                    request=request_id,
                    index=index,
                    position=int(positions[start]),
                    scheduled=end - start,
                    drafts=drafts,
                    committed_rows=valid,
                    sampled=list(tokens),
                    committed_end=int(positions[start + valid - 1]) + 1,
                    blocks=common.block_table_tensor[index].tolist(),
                    prompt_ids=list(request.prompt_token_ids),
                    output_count=len(request.output_token_ids),
                    output_limit=params.max_tokens,
                )
            )
        if self.capture_dir:
            for row in rows:
                prompt_length = len(row["prompt_ids"])
                if (
                    row["committed_end"] >= prompt_length
                    and row["request"] not in self._prompt_caches
                ):
                    # Preserve target prompt rows before the first draft can touch any cache.
                    self._prompt_caches[row["request"]] = self._read_cache(
                        row, prompt_length, include_draft=False
                    )
        result = [[] for _ in rows]
        if self.mode in ("real", "real_scripted"):
            result = self._native_propose(
                scheduler, sampled, sampling, hidden, sample_hidden, aux, spec, common, slots
            )
            self._stats["drafter"]["context_tokens"] += sum(row["committed_rows"] for row in rows)
            for row, proposal in zip(rows, result, strict=True):
                row["native_proposals"] = list(proposal)
        if self.mode in ("scripted", "real_scripted"):
            for row in rows:
                if (
                    not row["sampled"]
                    or row["committed_end"] + runner.num_spec_tokens + 1
                    > runner.effective_drafter_max_model_len
                ):
                    continue
                if row["output_count"] >= row["output_limit"]:
                    continue
                reference = next(r for r in self.references if r["prompt_ids"] == row["prompt_ids"])
                index = row["index"]
                count = int(runner.input_batch.num_tokens_no_spec[index])
                tokens = runner.input_batch.token_ids_cpu[index, :count].tolist()
                offset = count - len(reference["prompt_ids"])
                known = min(offset, len(reference["token_ids"]))
                actual_prefix = tokens[len(reference["prompt_ids"]) :][:known]
                assert actual_prefix == reference["token_ids"][:known]
                draft = reference["token_ids"][offset : offset + runner.num_spec_tokens]
                draft += [0] * (runner.num_spec_tokens - len(draft))
                reject_at = reference.get("reject_at", self.reject_at)
                if 0 <= reject_at < len(draft):
                    draft[reject_at] = (draft[reject_at] + 1) % runner.model_config.get_vocab_size()
                    for i in range(reject_at + 1, len(draft)):
                        draft[i] = (
                            draft[i] + self.tail_salt
                        ) % runner.model_config.get_vocab_size()
                result[index] = draft
        for row, proposal in zip(rows, result):
            row["proposals"] = list(proposal)
            if (
                self.capture_dir
                and row["output_count"] >= row["output_limit"]
                and row["request"] not in self.snapshots
            ):
                self._capture(row)
        self.steps.extend(rows)
        self.batch_steps.append(rows)
        return result

    def _read_cache(self, row, length, *, include_draft):
        runner = self.model_runner
        cache = {}
        for name, pages in runner._spyre_kv_caches.items():
            if name in runner.drafter._draft_attn_layer_names and not include_draft:
                continue
            block_size = pages.k_pages.shape[2]
            ids = torch.tensor(row["blocks"][: (length + block_size - 1) // block_size])
            cache[name] = tuple(
                tensor.to(device="cpu", dtype=torch.float32)[ids]
                .permute(0, 2, 1, 3)
                .flatten(0, 1)[:length]
                .contiguous()
                for tensor in pages
            )
        return cache

    def _capture(self, row):
        prompt_length = len(row["prompt_ids"])
        length = min(prompt_length + row["output_limit"] - 1, row["committed_end"])
        cache = self._read_cache(row, length, include_draft=self.mode in ("real", "real_scripted"))
        for name, pair in self._prompt_caches[row["request"]].items():
            for label, actual, expected in zip(("key", "value"), cache[name], pair, strict=True):
                assert torch.equal(actual[:prompt_length], expected), (
                    row["request"],
                    name,
                    label,
                    "prompt cache changed during decoding",
                )
        path = self.capture_dir / (hashlib.sha256(row["request"].encode()).hexdigest() + ".pt")
        torch.save(cache, path)
        self.snapshots[row["request"]] = dict(
            path=str(path), length=length, prompt_rows_checked=prompt_length
        )

    def validation_state(self):
        state = super().validation_state()
        state.update(batch_steps=self.batch_steps, snapshots=self.snapshots)
        batch_sizes = [0] * 5
        for batch in self.batch_steps:
            active = sum(bool(row.get("native_proposals")) for row in batch)
            if active:
                batch_sizes[active] += 1
        state["drafter"]["proposal_batch_sizes"] = batch_sizes
        return state
