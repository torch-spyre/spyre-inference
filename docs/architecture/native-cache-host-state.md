# Native cache placement metadata

The `spyre_native` loader owns construction, traversal, cache-identity computation,
FP8 model exclusion and fallback.
fastsafetensors' Spyre backend owns all persisted metadata keys and formats,
including the cache envelope, transfer/CPU placement recipes, payload I/O, path
resolution and alias-preserving tensor restoration. Models have no cache-specific
methods or construction changes.

## Writing

The private `_HostModelLoader` reads source checkpoints onto CPU. By default,
fastsafetensors distributes file reads across ranks and broadcasts full tensors
before the model's TP slicing (`all_local=False`). Set
`SPYRE_NATIVE_HOST_ALL_LOCAL=1` before launch to make each rank read independently;
unset or `0` uses distributed reading. Use the same value on all ranks. This only
controls source-checkpoint loading on cache miss or normal-load fallback, not
rank-local native-cache restoration.

1. Build/load/post-process the model on CPU as usual.
2. `TransferRecorder.capture_host_state(model)` records existing tensor paths.
3. Run `move_model_to_spyre(..., transfer=recorder)`, using the ordinary
   Attention exclusion and module `_apply` implementations. The pinned torch-spyre
   retains Parameter identity, so sink and MoE references need no repair.
4. `recorder.checkpoint_metadata(model)` returns tensor payload entries and
   backend-owned `torch_spyre.transfers` / `torch_spyre.host` metadata. Only
   original paths that still hold CPU tensors enter the host recipe. Newly
   derived rotation caches are regenerated during replay, not saved as host state.
5. `cache_metadata(key)` encodes cache identity with backend-owned
   `torch_spyre.cache.format`, `.key` and `.fp8` keys.
   Its repeated and shard-zero metadata are passed to `ParallelSaver`, merging
   the latter with the transfer/host metadata. The loader does not define cache
   format strings or manually encode metadata.

The supported state containers are module attributes (parameters/buffers
included), dictionaries, lists/tuples, dataclasses and `SimpleNamespace`.
Cycles are skipped; arbitrary object internals are not traversed. This includes
Attention's CPU scales, decoder and Transformers RoPE sources, vision padding,
Gemma vision-tail weights/standardization, and Granite packing newline/order
indices without per-model adapters. Tensor bodies remain safetensors payloads.

## Reading

Build a meta skeleton and validate the envelope with
`read_cache_metadata(metadata, key)`, then call `TransferRestorer.from_metadata` and
`restore_host_state(model)`. The reader resolves recorded paths directly;
it does not rediscover host tensors or evaluate Python. All host paths,
shape/dtype/device and object-alias groups are checked before the first swap.
The swap retains subclasses, Python attributes and existing object aliases.
Distinct views sharing storage are not restored as shared storage.

Restore CPU tensors **before** `move_model_to_spyre(..., transfer=restorer)` so
the existing `_apply` can derive sources from real CPU values. Check record
exhaustion and remaining meta state, then fill raw destinations
with the existing `ParallelLoader`.

`SkeletonLoader` uses a scoped dispatch mode to keep meta-to-CPU copies symbolic
during construction/post-processing. For example Transformers RoPE can request
`inv_freq.to("cpu")` without trying to read a valueless meta tensor. The mode
does not invent values, affect ordinary CPU loading or persist after construction;
the cache supplies actual CPU values afterwards. Data-dependent construction is
not supported by this mechanism.

The native payload convention/package versions are unchanged. Source digests
invalidate old cache keys; recipes without the backend-owned keys are rejected
and use the existing normal-load fallback. No `spyre_inference.*` metadata keys
are written or interpreted. The envelope is now `torch_spyre-cache/3`, distinct
from the physical `torch_spyre/1` payload format; schema revision 3 is retained.

Compatibility relies on the existing cache key (model configuration, checkpoint,
environment and source digests) and tensor-restoration checks. Plain non-tensor
model attributes are not collected, persisted, compared or restored.
