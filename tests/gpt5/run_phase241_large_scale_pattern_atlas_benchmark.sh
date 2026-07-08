#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-large_scale_pattern_atlas_benchmark}"
SAMPLES_PER_MODE="${SAMPLES_PER_MODE:-12}"
MODE_LIMIT="${MODE_LIMIT:-}"
FAMILIES="${FAMILIES:-}"
MAX_JOBS="${MAX_JOBS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --samples-per-mode "${SAMPLES_PER_MODE}"
  --max-jobs "${MAX_JOBS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --batch-size "${BATCH_SIZE}"
)

if [[ -n "${MODE_LIMIT}" ]]; then
  COMMON_ARGS+=(--mode-limit "${MODE_LIMIT}")
fi

if [[ -n "${FAMILIES}" ]]; then
  COMMON_ARGS+=(--families "${FAMILIES}")
fi

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase241 ${MODEL} =="
  python tests/gpt5/phase241_large_scale_pattern_atlas_benchmark.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

echo "== Phase241 summarize =="
python tests/gpt5/phase241_large_scale_pattern_atlas_benchmark.py \
  --summarize \
  --round-name "${ROUND_NAME}"
