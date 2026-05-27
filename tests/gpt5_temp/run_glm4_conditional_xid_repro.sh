#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

# This is a narrow repro for the 2026-05-27 GLM4 conditional Xid 62/45 event.
# It intentionally runs only the failing category, with CUDA synchronization
# hints and full logging. Use only after saving work, because prior runs hard
# locked the NVIDIA driver.

export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt5_systematic_language_v2_driver595_cuda121_xid_repro}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING:-1}"
export PROBE_TORCH_DTYPE="${PROBE_TORCH_DTYPE:-float16}"
export TOKENIZERS_PARALLELISM=false

tests/gpt5_temp/run_logged_language_category.sh glm4 conditional 10
