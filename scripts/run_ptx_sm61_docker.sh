#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found"
  exit 1
fi

if [[ ! -c /dev/nvidia0 ]]; then
  echo "no NVIDIA device nodes found (expected /dev/nvidia0)"
  exit 1
fi

LIBCUDA="$(
  readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null \
    || readlink -f /lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null \
    || true
)"
PTXJIT="$(
  readlink -f /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 2>/dev/null \
    || readlink -f /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 2>/dev/null \
    || true
)"

if [[ -z "${LIBCUDA}" || ! -f "${LIBCUDA}" ]]; then
  echo "host libcuda.so.1 not found"
  exit 1
fi
if [[ -z "${PTXJIT}" || ! -f "${PTXJIT}" ]]; then
  echo "host libnvidia-ptxjitcompiler.so.1 not found"
  exit 1
fi

CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.2.0-devel-ubuntu22.04}"

docker run --rm \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools \
  --device /dev/nvidia-modeset \
  -e LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
  -v "${LIBCUDA}:/usr/local/nvidia/lib64/$(basename "${LIBCUDA}")":ro \
  -v "${PTXJIT}:/usr/local/nvidia/lib64/$(basename "${PTXJIT}")":ro \
  -v "${ROOT}:/work" \
  -w /work \
  "${CUDA_IMAGE}" \
  bash -lc '
    set -euo pipefail
    mkdir -p /usr/local/nvidia/lib64
    ln -sf "$(basename "'"${LIBCUDA}"'")" /usr/local/nvidia/lib64/libcuda.so.1
    ln -sf libcuda.so.1 /usr/local/nvidia/lib64/libcuda.so
    ln -sf "$(basename "'"${PTXJIT}"'")" /usr/local/nvidia/lib64/libnvidia-ptxjitcompiler.so.1
    ln -sf libnvidia-ptxjitcompiler.so.1 /usr/local/nvidia/lib64/libnvidia-ptxjitcompiler.so
    nvcc --version
    apt-get update -y >/dev/null
    apt-get install -y python3 >/dev/null
    python3 -m unittest tests.test_backend_ptx_run_sm61_optional -v
  '

