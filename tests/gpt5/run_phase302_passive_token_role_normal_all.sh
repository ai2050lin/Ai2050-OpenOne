#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt5_phase302_passive_token_role_closure}"
export MAX_BASES="${MAX_BASES:-24}"
export COMMON_ARGS="${COMMON_ARGS:---max-bases ${MAX_BASES} --train-fraction 0.5 --modules resid_in,resid_out,mlp_out --alphas 0,1.0 --progress-every 4}"

echo "=== Phase302 all-model normal run ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "output_dir=${OUTPUT_DIR}"
echo "max_bases=${MAX_BASES}"
echo "common_args=${COMMON_ARGS}"
echo

echo "=== Run qwen3 ==="
MAX_SECONDS="${QWEN3_MAX_SECONDS:-7200}" OUTPUT_DIR="$OUTPUT_DIR" \
  tests/gpt5/run_phase302_normal.sh qwen3 \
    --layers "${QWEN3_LAYERS:-0,1,2,3,4,5,6,7,8}" \
    ${COMMON_ARGS}

echo "=== Run glm4 ==="
MAX_SECONDS="${GLM4_MAX_SECONDS:-10800}" OUTPUT_DIR="$OUTPUT_DIR" \
  tests/gpt5/run_phase302_normal.sh glm4 \
    --layers "${GLM4_LAYERS:-0,1,2,3,4,5,6,7,8}" \
    ${COMMON_ARGS}

echo "=== Run deepseek7b ==="
MAX_SECONDS="${DEEPSEEK7B_MAX_SECONDS:-7200}" OUTPUT_DIR="$OUTPUT_DIR" \
  tests/gpt5/run_phase302_normal.sh deepseek7b \
    --layers "${DEEPSEEK7B_LAYERS:-20,21,22,23,24,25,26,27}" \
    ${COMMON_ARGS}

echo "=== Summarize ==="
python tests/gpt5/phase302_passive_token_role_summary.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "=== Done Phase302 all-model normal run ==="
date '+%Y-%m-%d %H:%M:%S %Z'
