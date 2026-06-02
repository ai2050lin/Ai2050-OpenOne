#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_USE_CONSERVATIVE_ENV="${OPENONE_USE_CONSERVATIVE_ENV:-1}"
export OPENONE_CONSERVATIVE_ENV="${OPENONE_CONSERVATIVE_ENV:-openone-cu130-py312}"
export CASES_PER_CATEGORY="${CASES_PER_CATEGORY:-10}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt5_systematic_language_v2_conservative_stage${CASES_PER_CATEGORY}}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec tests/gpt5_temp/run_stage10_logged_sequence.sh "$@"
