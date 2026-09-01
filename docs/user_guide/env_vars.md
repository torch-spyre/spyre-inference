# Environment Variables

Spyre Inference reads the following `SPYRE_*` environment variables to configure
the plugin. Each is evaluated the first time it is read, so set it before
launching vLLM. The list below is generated from `spyre_inference/envs.py`; the
comment above each entry documents its effect and default.

```python
--8<-- "spyre_inference/envs.py:env-vars-definition"
```
