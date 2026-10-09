# Model-ops YAML

Op-level test configs (`test_suite_config`, one case per distinct op signature) recorded while a model
runs through spyre-inference, so the ops are the ones this plugin compiles and not the ones in stock
Hugging Face modeling code. The generator scripts live in
[hf-adapters](https://github.com/torch-spyre/hf-adapters) (`utils/model_ops/`); only the generated YAML
is kept here.

## Files

| File | Model | Cases | Distinct ops |
|---|---|---|---|
| `gemma-4-26B-A4B-it.yaml` | `google/gemma-4-26B-A4B-it` | 1090 | 45 |

## How `gemma-4-26B-A4B-it.yaml` was generated

Driver: `utils/model_ops/models/gemma4-26b-a4b/run_spyre_inference.py` in hf-adapters, added by
[hf-adapters#651](https://github.com/torch-spyre/hf-adapters/pull/651) (generated from its head commit
`f977452`).

- Image: `icr.io/ai_sw_accel/2.0/prod/spyre-inference:ci-cd-tech-preview-v3`, vLLM 0.28.0, one Spyre
  card, tensor parallel size 1, float16, default compilation (`STOCK_TORCH_COMPILE`).
- Engine settings: `max_model_len=3072`, `max_num_seqs=8`; one request, prompt
  `Say hello in one word.`, 8 new tokens (the parameter header at the top of the file records batch 1,
  6 input tokens, 8 output tokens).
- Collector output: 49 ops traced, 45 with test configs, 1090 test cases.

To regenerate, run the driver from `utils/model_ops/` inside the spyre-inference image on a Spyre host
(see the hf-adapters `utils/model_ops/README.md`) and copy the resulting `gemma-4-26B-A4B-it.yaml` here.

## Notes

- float16 is expected: spyre-inference forces float16 on Spyre, so these cases are float16. The YAML in
  torch-spyre was recorded from stock Hugging Face on CUDA with bfloat16.
- torch-spyre internal ops (`torch.ops.spyre.*`, `torch_spyre._monkey_patch.*`, `torch._C._autograd.*`)
  are skipped by the collector and do not appear in the file.
- The file is otherwise raw collector output. It still includes lowered `torch.ops.aten.*` cases (35) and
  other ops that the torch-spyre test run skips as unregistered (for example `torch.unflatten` and
  `torch.index_select`).
- Attention (`unified_attention_with_output`) and MoE (`moe_forward`) are traced but yield no test case,
  because their arguments include opaque objects.
