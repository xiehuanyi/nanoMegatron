#!/bin/bash
# Build the Megatron-pinned Transformer Engine PyTorch extension in this environment.

set -euo pipefail

PYTHON=${PYTHON:-/ibex/project/c2334/huanyi/conda_env/finetuning/bin/python}
PIP=${PIP:-$(dirname "$PYTHON")/pip}
CUDA_HOME=${CUDA_HOME:-/sw/rl9g/cuda/12.4.1/rl9_binary}
TE_VERSION=${TE_VERSION:-2.9.0}
MAX_JOBS=${MAX_JOBS:-4}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}

site_packages=$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')
nvidia_dir="$site_packages/nvidia"
test -d "$nvidia_dir"
test -x "$CUDA_HOME/bin/nvcc"

nvidia_includes=$(find "$nvidia_dir" -maxdepth 3 -type d -name include -print | paste -sd:)
nvidia_libs=$(find "$nvidia_dir" -maxdepth 3 -type d -name lib -print | paste -sd:)

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$nvidia_includes:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$nvidia_includes:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$nvidia_libs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$nvidia_libs:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS
export NVTE_FRAMEWORK=pytorch
export TORCH_CUDA_ARCH_LIST

"$PIP" install -v --no-build-isolation "transformer-engine[pytorch]==$TE_VERSION"
