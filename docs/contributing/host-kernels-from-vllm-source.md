# Host kernels from vLLM's source (plan and context for the "option B" prototype)

Status: phase 1 and 2 of section 5 prototyped on vLLM 0.32.0 (2026-10-20); see
"Prototype status" at the end. The rest was written 2026-10-16 from a survey of vLLM
`main`, `v0.31.0` and vllm#61995; re-check the facts marked *(verify)* before building on
them.

## 1. Goal and invariants

Consume vLLM's host-side C++ kernels **as upstream writes them**, compiled from vLLM's
own source, instead of copying their sources into `csrc/`.

Invariants (decided):

- **Use upstream's C++.** No pure-Python reimplementations of these kernels.
- **No copied kernel sources** in this repository. Today's `csrc/cpu/` copies (#1111) go
  away once this lands.
- **Selective compilation.** Build only the kernels we need, not vLLM's whole CPU
  extension: a full CPU build is prohibitively slow for us (see 3).
- **Keep building vLLM with `VLLM_TARGET_DEVICE=empty`** (no `vllm._C`), unless that
  decision is explicitly revisited.

Longer term ("option C"), propose upstream that these kernels become their own small
build target. See 7. This prototype should be shaped so it shrinks to almost nothing
once that lands.

## 2. What exists today (#1111, on `main`)

| Piece | Where | Notes |
|---|---|---|
| Kernel sources (copied from vLLM) | `csrc/cpu/sampling_kernels.cpp`, `csrc/cpu/cpu_types{,_x86,_vsx,_vxe}.hpp`, `csrc/core/registration.h` | Headers say "Portions adapted from vLLM". `sampling_kernels.cpp` matches upstream `main` after vllm#60221 (SplitMix64 noise) |
| Bindings (ours) | `csrc/cpu/torch_bindings.cpp` | `TORCH_LIBRARY(_spyre_C, ops)`: `fused_gumbel_argmax(Tensor logits, Tensor seeds) -> Tensor`, `greedy_argmax(Tensor logits) -> Tensor` |
| Build | `CMakeLists.txt`, `cmake/cpu_extension.cmake`, `cmake/utils.cmake`, `setup.py` | One `SPYRE_EXT_SRC` list. x86: `_C` (AVX-512) + `_C_AVX2`; Power10+: `-mvsx -mcpu=power10`; s390x: `-mvx -mzvector -march=z15`. Links torch's libgomp (`prepare_torch_gomp_shim`) |
| Build-failure policy | `setup.py` `cmake_build_ext` | **Optional**: on failure it warns and installs without kernels |
| Prebuilt reuse in CI | `setup.py` `_extract_from_prebuilt_wheel`, `SPYRE_KERNELS_WHEEL_DIR`; `.github/workflows/_test_matrix.yaml` job `build_kernels` | CI builds once and every test job extracts the `.so`s |
| Loader | `spyre_inference/v1/sample/sampling_kernels.py` | Imports `spyre_inference._C` / `_C_AVX2` (AVX-512 check); `has_sampling_kernels()` |
| Consumers | `spyre_inference/v1/sample/{sampler.py,topk_topp_sampler.py}`; tests `tests/runtime/test_sampling_{kernels,fallback}.py` | V1 sampler calls `torch.ops._spyre_C.*`; PyTorch fallback when the kernels are absent |

A second consumer already exists without C++: the V1 runner reimplements vLLM's slot
mapping kernel in pure Python (`spyre_model_runner.py`: `_compute_slot_mapping_impl`,
`_SlotMappingKernelShim`, `_patch_compute_slot_mapping`). Upstream has a C++ version
(`compute_slot_mapping_kernel_impl`, in `csrc/cpu/utils.cpp`, present at `v0.31.0`),
which vLLM's own `vllm/utils/cpu_triton_utils.py` calls as
`torch.ops._C.compute_slot_mapping_kernel_impl`.

## 3. Facts that rule out the alternatives

- **A full CPU-target vLLM build** (`VLLM_TARGET_DEVICE=cpu`):
    - Always fetches and builds oneDNN (on x86, Power9+, s390x and Arm; plus ACL on
    Arm) and every CPU kernel, with several ISA variants on x86. There is no switch to
    build a subset (`cmake/cpu_extension.cmake`).
    - Changes vLLM's behaviour: the `+cpu` version string makes `vllm.triton_utils`
    accept triton-cpu, and `vllm._C` gets loaded.
    - Its `requirements/cpu.txt` pins `numba==0.65.0` with no architecture marker.
    numba and llvmlite publish no ppc64le/s390x wheels *(verify)*.
- **Prebuilt vLLM CPU wheels**: v0.31.0 publishes `+cpu` wheels for x86_64, aarch64 and
  macOS only. Not usable for Power/Z production.
- **Runtime JIT** (`torch.utils.cpp_extension.load`): vLLM wheels don't include
  `csrc/`, and production would need a compiler at runtime.

## 4. Design for option B

### 4.1 Getting the source without copying it

Use a CMake `FetchContent` (or a git submodule) of `https://github.com/vllm-project/vllm`
at a **pinned commit**. Only the listed files are compiled; nothing is copied into the
repository.

- Pin to a full SHA, not a branch. Record it in one place (for example a CMake cache
  variable `SPYRE_VLLM_KERNELS_REV`, read from a small file or `pyproject.toml`).
- Offline/air-gapped builds: honour `FETCHCONTENT_SOURCE_DIR_VLLM`, so a pre-fetched
  checkout can be used. Prefer a sparse/shallow fetch of `csrc/` only if CMake allows
  it cleanly.

### 4.2 Version coupling: two kinds of consumer

This is the main subtlety. It decides the pin rule.

| Consumer | Example ops | Called by | Pin rule |
|---|---|---|---|
| Our Python | `fused_gumbel_argmax`, `greedy_argmax` | `spyre_inference/v1/sample/*` | Any vLLM commit whose source we have reviewed; independent of the installed vLLM |
| vLLM's Python | `compute_slot_mapping_kernel_impl` (`cpu_triton_utils.py`), and later #61995's `*_cpu` ops (`vllm/_custom_ops.py`) | the installed vLLM | **Must be the same commit as the installed vLLM**, or the schemas and semantics can drift |

Facts *(verify at implementation time)*:

- At **`v0.31.0`** (our pin), `csrc/cpu/sampling_kernels.cpp` **does not exist**. It
  landed on `main` with vllm#53913 (2026-09-30) and was fixed by vllm#60221
  (2026-10-08). So the two ops #1111 uses are only available from a post-0.31 commit,
  or from 0.32.
- At `v0.31.0` these do exist: `csrc/cpu/utils.cpp` (`compute_slot_mapping_kernel_impl`;
  NUMA code guarded by `#ifndef VLLM_NUMA_DISABLED`), `csrc/cpu/spec_decode_utils.cpp`,
  `csrc/cpu/mamba_cpu.cpp`, `csrc/cpu/cpu_types*.hpp`, `csrc/cpu/utils.hpp`.
- vllm#61995 (fadara01, `fadara01:mrv2-cpu`, head
  `df4daa43a3185cf8083aed18ac0284c506befc72` on 2026-10-16) adds
  `csrc/cpu/{block_table.cpp,input_batch.cpp,sampling/*.h}`, extends
  `sampling_kernels.cpp` and `mamba_cpu.cpp`, and registers 11 new `torch.ops._C.*_cpu`
  ops. Unmerged.

Suggested pin rule:

- Ops called by vLLM's Python use the **installed vLLM's commit**.
- Ops called by our own Python may use a separate reviewed commit until our vLLM pin
  catches up (for example the #1111 ops before 0.32). Once it does, collapse to one pin.
- A test asserts that the "vLLM-called" pin equals the `vllm` source in `pyproject.toml`
  (`[tool.uv.sources]`, currently `rev = "v0.31.0"`).

### 4.3 What to compile (category H only)

Category H means C++ host versions of the shared runners' Triton kernels, or host
kernels our Python calls. They depend only on torch, `cpu_types.hpp` and `utils.hpp`;
no oneDNN or ACL.

| Upstream file | Ops we register | Phase | Notes |
|---|---|---|---|
| `csrc/cpu/sampling_kernels.cpp` | `fused_gumbel_argmax`, `greedy_argmax` | 1 | Replaces #1111's copy. Needs a post-0.31 commit (see 4.2) |
| `csrc/cpu/utils.cpp` | `compute_slot_mapping_kernel_impl` | 1 | Compile with `-DVLLM_NUMA_DISABLED`. Lets the V1 runner drop its pure-Python slot mapping (2) and go through vLLM's own dispatcher path instead |
| `csrc/cpu/spec_decode_utils.cpp` | rejection sampler (×3), `expand`, `eagle_*` (×3), `copy_and_expand_{eagle,dflash}_inputs` | optional | Only needed with spec decode, which Spyre rejects today |
| `csrc/cpu/{block_table,input_batch}.cpp`, `csrc/cpu/sampling/*.h`, `csrc/cpu/mamba_cpu.cpp` | #61995's `*_cpu` ops | 2 | After #61995 merges and our pin includes it; needed for MRV2 on Power/Z |

Headers needed on the include path: `csrc/` (for `cpu/utils.hpp`'s
`#include "cpu/cpu_types.hpp"`), `csrc/cpu/`, and `csrc/core/registration.h` (or our
own `REGISTER_EXTENSION`).

Do **not** compile CPU-compute files (`activation`, `layernorm`, `cpu_attn*`,
`dnnl_*`, `sgl-kernels/*`, ...), or `shm.cpp`. See `mrv2-upstream-outlook.md` section 4
on the `mrv2-full` branch for the full categorised inventory.

### 4.4 Registration

- **Namespace `_C`, as a fragment:** `TORCH_LIBRARY_FRAGMENT(_C, ops)` in a small
  bindings file of ours. Then `vllm/_custom_ops.py` and `cpu_triton_utils.py` resolve
  `torch.ops._C.<op>` unmodified, and our own Python calls `torch.ops._C.*` too (the
  `_spyre_C` namespace goes away). Under `empty` there is no real `vllm._C` to collide
  with.
- **Collision guard:** if `vllm._C` is importable (a CPU-built vLLM), don't load ours;
  upstream's registration wins.
- **Our bindings file duplicates only the op schema strings** (one `ops.def(...)` and
  `ops.impl(...)` per op). That's the only remaining duplication, and 4.6's parity test
  guards it.
- **Load order:** the extension must be imported before anything launches one of these
  ops. For V1, import it at plugin/worker start (where `sampling_kernels.py` imports it
  today). For MRV2, import it in `SpyreWorkerV2.init_device`, before `kernels.bind()`.

### 4.5 Build integration

- Reuse today's ISA flags and targets from `cmake/cpu_extension.cmake`: x86 `_C` +
  `_C_AVX2`, Power10+ VSX, s390x VXE, and the torch libgomp shim. `cpu_types.hpp`
  selects the ISA header from compiler macros, exactly as now.
- Upstream's `utils.hpp` includes `<ATen/cpu/Utils.h>`. Check that it exists in our
  torch 2.13 on all three architectures.
- Keep the `SPYRE_KERNELS_WHEEL_DIR` prebuilt path and the CI `build_kernels` job.
- **Failure policy needs a decision.** Today the build is optional with a PyTorch
  fallback. Under the "use upstream's C++" invariant, phase 1's V1 sampling fallback can
  stay as is (it exists today). Phase 2's MRV2 ops have no fallback, so a missing
  extension must fail loudly at worker start under `SPYRE_MODEL_RUNNER=v2`. Consider
  making the build mandatory on Power/Z.

### 4.6 Tests

- **Schema parity (CPU, fast):** fetch or parse upstream `csrc/cpu/torch_bindings.cpp`
  at the pinned commit and assert every schema we register is byte-identical to
  upstream's. Upstream uses two forms; handle both:
    - `ops.def("name(...) -> ...");` followed by `ops.impl("name", torch::kCPU, &fn);`
    - `ops.def("name(...) -> ...", &fn);`, with the schema often split across adjacent
    string literals (for example `compute_slot_mapping_kernel_impl(Tensor
    query_start_loc, Tensor positions, Tensor block_table, Tensor(a3!) slot_mapping,
    SymInt block_size) -> ()`).
- **Pin parity:** the vLLM-called pin equals the `vllm` rev in `pyproject.toml`.
- **Op availability:** after import, `torch.ops._C.<op>` exists for every registered op.
- **Behaviour:** keep `tests/runtime/test_sampling_kernels.py` and
  `test_sampling_fallback.py` (updated for `torch.ops._C`). Add slot mapping parity
  against the current pure-Python `_compute_slot_mapping_impl` before deleting it.
- **On hardware:** a V1 smoke test (greedy and seeded sampling token-exact before and
  after) on x86, and on Power/Z where available.

## 5. Suggested steps

1. Add the `FetchContent` of vLLM at a pinned SHA. Compile `sampling_kernels.cpp` from
   it, plus our `_C`-fragment bindings for `fused_gumbel_argmax` / `greedy_argmax`.
   Switch the V1 sampler from `torch.ops._spyre_C.*` to `torch.ops._C.*`. Delete
   `csrc/cpu/sampling_kernels.cpp` and the copied `cpu_types*.hpp` /
   `core/registration.h`.
2. Add `utils.cpp` (`-DVLLM_NUMA_DISABLED`) and register
   `compute_slot_mapping_kernel_impl`. Replace the V1 runner's
   `_SlotMappingKernelShim` / `_compute_slot_mapping_impl` with vLLM's own CPU path:
   `CpuPlatform.register_triton_kernel_overrides`, which `TorchSpyrePlatform` inherits,
   already routes the slot mapping kernel to `cpu_triton_utils` when Triton is absent.
3. Schema-parity and pin-parity tests; CI wiring.
4. After vllm#61995 merges and the vLLM pin includes it, add its files and ops for MRV2
   on Power/Z (the `mrv2-full` spike branch on the fork holds the V2 runner).

## 6. Open questions

- One pin or two until our vLLM pin includes `sampling_kernels.cpp` (4.2)?
- `FetchContent` or a git submodule (offline builds, CI caching, how a bump PR reads)?
- Mandatory or optional build (4.5)?
- Does upstream's `utils.cpp` compile cleanly with `-DVLLM_NUMA_DISABLED` on Power/Z, and
  does it pull in anything else beyond `utils.hpp`?
- Does #61995's `sampling_kernels.cpp` keep `fused_gumbel_argmax` / `greedy_argmax`
  unchanged (schema and RNG semantics: SplitMix64 noise, vllm#60221)?

## 7. Path to option C (upstream)

Propose that vLLM build "the runner's host kernels" (category H) as their own small
target, built for every device target including `empty`. Rationale (not
Spyre-specific):

1. They are host implementations of the shared V1/MRV2 runners' Triton kernels, as
   device-agnostic as the runner. Users: CPU builds without Triton (Power, Z, Arm,
   RISC-V; the motivation for #61995), and OOT platforms built with `empty`.
2. Clean dependency boundary: torch, `cpu_types.hpp` and `utils.hpp` only.
3. Precedent: `spinloop` and `fs_io_C` are already device-independent extensions built
   for every target (above the device branch in `CMakeLists.txt`). `empty` currently
   drops even those (`setup.py`: `if _no_device(): ext_modules = []`).
4. Restructuring needed:
   - Split `utils.cpp` (slot mapping vs NUMA setup).
   - Move the H registrations out of the monolithic `csrc/cpu/torch_bindings.cpp`
     (1,072 lines on `main`) into their own `TORCH_LIBRARY_FRAGMENT(_C)` file.

Once that lands, option B reduces to "depend on vLLM's host-kernels target". Our
bindings file and the schema-parity test go away.

## 8. Prototype status (2026-10-20, vLLM 0.32.0)

Built on top of the 0.32.0 bump. Steps 1 and 2 of section 5 are done; 3 is partly done.

What landed:

- `cmake/cpu_extension.cmake` reads the `vllm` rev from `[tool.uv.sources]` in
  `pyproject.toml` (`v0.32.0`), so the kernels can't drift from the installed vLLM
  (`-DSPYRE_VLLM_REV=<rev>` overrides it). It fetches that rev with `FetchContent` from
  GitHub's source tarball (`SOURCE_SUBDIR` points
  at a missing directory, so vLLM's own CMake is never configured;
  `FETCHCONTENT_SOURCE_DIR_VLLM` overrides it for offline builds). It compiles
  `csrc/cpu/sampling_kernels.cpp` and `csrc/cpu/utils.cpp` (`-DVLLM_NUMA_DISABLED`) with
  the unchanged ISA targets and flags. `csrc/` now holds only our `torch_bindings.cpp`.
- `csrc/torch_bindings.cpp` registers `fused_gumbel_argmax`, `greedy_argmax` and
  `compute_slot_mapping_kernel_impl` with `TORCH_LIBRARY_FRAGMENT(_C, ops)`, using
  vLLM's schemas. `core/registration.h` comes from vLLM's tree.
- Schema parity runs at configure time (`cmake/check_schemas.py`, which handles schemas
  split across string literals) rather than as a pytest: the build has vLLM's
  `torch_bindings.cpp` on disk, and a test would have to fetch it.
- No separate pin, so no pin-parity test (4.2's two-pin question is moot from 0.32 on).
  `tests/runtime/test_host_kernels.py` checks op availability and the slot mapping
  through vLLM's real `BlockTable` (including the `PAD_ID` tail).
- The V1 sampler calls `torch.ops._C.*`; `_spyre_C` is gone.
  `spyre_inference/v1/sample/sampling_kernels.py` became `spyre_inference/v1/host_kernels.py`.
- The pure-Python slot mapping (`_compute_slot_mapping_impl`, `_SlotMappingKernelShim`,
  `_patch_compute_slot_mapping`) and its test are deleted. The runner calls
  `register_slot_mapping_kernel()`, which binds the C++ op to vLLM's Triton kernel through
  vLLM's own dispatcher (`vllm.triton_utils.dispatcher.register_kernels`).

Findings that changed the plan:

- **The 0.31 shim is broken on 0.32.** 0.32 dropped `BLOCKS_PER_KV_BLOCK` from the
  slot mapping kernel's launch, so the shim's positional parameters shift and the first
  forward fails ("size of tensor a (2) must match … (1024)"). The 0.32 upgrade has to
  fix or replace it either way.
- **vLLM 0.32's own CPU override is broken too**, the same way:
  `cpu_triton_utils._compute_slot_mapping_kernel_impl` still has the 0.31 signature,
  so `CpuPlatform.register_triton_kernel_overrides()` raises on its first launch. Fixed
  upstream in vllm#61992 (2026-10-17), after 0.32.0. Hence our own small Python adapter
  for now (parameter names match the Triton kernel, so the dispatcher forwards by name).
  Once the pin includes #61992, it can be replaced by calling
  `CpuPlatform.register_triton_kernel_overrides()` (which also brings the spec-decode
  overrides; those need `spec_decode_utils.cpp` in the build).
- The upstream C++ op writes only the scheduled tokens. The Triton kernel also pads
  `slot_mapping[num_tokens:max_num_tokens]` with `PAD_ID`, so the adapter does too.
- `sampling_kernels.cpp` at 0.32.0 has the same semantics as the #1111 copy (SplitMix64
  noise; the copy was a refactor of the same loop), so outputs are unchanged.
- Fetching costs ~45 MB (full-source tarball, a few seconds normally, about a minute on a
  slow link); the compile of both x86 variants is about 50 s, as before.

Verified on x86 (Spyre host): the build through `uv sync`; `test_host_kernels.py`,
`test_sampling_kernels.py`, `test_sampling_fallback.py` and `test_spyre_topk_sampler.py`
(74 passed); and an end-to-end `LLM.generate` on `micro-g3.3-8b-instruct-1b`, greedy
and seeded sampling over 4 prompts × 24 tokens, token-identical to `main`
(vLLM 0.31 + the copied kernels + the Python slot mapping).

Not done:

- Power/Z builds (`utils.hpp` includes `<ATen/cpu/Utils.h>`; present in x86 torch 2.13,
  unchecked on the ppc64le source build).
- The failure policy (4.5) is unchanged: the build stays optional. Sampling falls back to
  PyTorch, but the slot mapping now has no fallback, so a missing extension fails at
  runner construction with a clear error. Making the build mandatory is the natural next
  step.
- CI's `build_kernels` job is untouched; it needs network access to GitHub at configure
  time now (or a pre-fetched `FETCHCONTENT_SOURCE_DIR_VLLM`).
- Phase 2 of 4.3 (#61995's MRV2 ops).
