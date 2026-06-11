# KV Connector Configuration

`InMemorySpyreConnector` is the Spyre paged-KV connector for vLLM
prefill/decode disaggregation. It resolves its runtime settings from two
sources:

1. **vLLM `kv_transfer_config`** — set on the command line via
   `vllm serve --kv-transfer-config '{...}'`.
2. **Environment variables** — for manual or baked-image deployments.

An explicitly set environment variable always overrides the
corresponding config value, so existing env-driven deployments keep
working unchanged.

## Settings

| Setting | Environment variable | Config fallback | Default |
|---|---|---|---|
| KV role | `VLLM_SPYRE_KV_ROLE` | `kv_transfer_config.kv_role` | `""` |
| Enable NIXL | `VLLM_SPYRE_ENABLE_NIXL_TRANSFER` | `kv_connector_extra_config["use_nixl"]` | `false` |
| Remote IP | `VLLM_SPYRE_NIXL_REMOTE_IP` | `kv_connector_extra_config["nixl_remote_ip"]`, then `kv_transfer_config.kv_ip` | `10.130.2.89` |
| NIXL port | (none) | `kv_connector_extra_config["nixl_port"]` | `9100` |

Notes:

- The KV role is `kv_producer` (prefill) or `kv_consumer` (decode). A
  non-empty `VLLM_SPYRE_KV_ROLE` wins; otherwise the connector uses
  `kv_role` from the transfer config.
- `kv_port` from the transfer config is **not** used as the NIXL port
  fallback: vLLM auto-assigns it, which would silently change the listen
  port. Set `nixl_port` in `kv_connector_extra_config` for a non-default
  port.

## Examples

### Config-driven (no connector env vars needed)

Producer / prefill — enable NIXL and listen on a chosen port:

```bash
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "InMemorySpyreConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {"use_nixl": true, "nixl_port": 9100}
  }'
```

Consumer / decode — enable NIXL and point at the producer entirely
through `kv_connector_extra_config`, so no `VLLM_SPYRE_NIXL_REMOTE_IP`
env var is required:

```bash
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "InMemorySpyreConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "use_nixl": true,
      "nixl_remote_ip": "<producer-host>",
      "nixl_port": 9100
    }
  }'
```

### Env-driven (overrides config when both are present)

```bash
export VLLM_SPYRE_KV_ROLE=kv_consumer
export VLLM_SPYRE_ENABLE_NIXL_TRANSFER=1
export VLLM_SPYRE_NIXL_REMOTE_IP=<producer-host>
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"InMemorySpyreConnector"}'
```
