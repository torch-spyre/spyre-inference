#!/usr/bin/env bash

# Activate the spyre-inference venv
source /opt/spyre-inference/bin/activate

# Required for Spyre backend
export VLLM_PLUGINS=spyre_inference

# Check for kineto-patched torch wheel (required for AIU device events in traces)
_torch_ver=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
if echo "$_torch_ver" | grep -q "+aiu.kineto"; then
    echo "[kineto] Patched torch installed: $_torch_ver — AIU device events will be captured."
else
    echo "[kineto] WARNING: stock torch detected ($_torch_ver). No Spyre device events will appear in traces." >&2
    echo "[kineto] Install the patched wheel matching your torch version: uv pip install --no-deps --force-reinstall <torch-VERSION+aiu.kineto.*>.whl" >&2
fi
unset _torch_ver

# Pin all CPU thread pools to 1 thread (keeps BLAS/OMP out of AIU dispatch).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
