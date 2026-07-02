#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

case "${ROUND_NAME}" in
  smoke)
    MAX_SOURCES=2
    MAX_NEW_TOKENS=6
    ;;
  main)
    MAX_SOURCES=6
    MAX_NEW_TOKENS=8
    ;;
  confirm)
    MAX_SOURCES=10
    MAX_NEW_TOKENS=8
    EXTRA_ARGS=()
    ;;
  transfer)
    MAX_SOURCES=2
    MAX_NEW_TOKENS=8
    EXTRA_ARGS=(
      --prompt-variants natural_question,natural_category
      --target-domains geometry,animal,tool,color,material,abstract
      --max-target-cases-per-domain 1
    )
    ;;
  *)
    echo "Unknown round: ${ROUND_NAME}" >&2
    exit 2
    ;;
esac

if [[ "${ROUND_NAME}" != "transfer" ]]; then
  EXTRA_ARGS=(
    --prompt-variants natural_question,natural_category,object_only
  )
fi

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase857 ${ROUND_NAME}: ${MODEL}"
  python tests/glm5/phase857_prompt_gated_causal_gear_rollout_validation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --source-round confirm \
    --max-sources "${MAX_SOURCES}" \
    --max-necessary-conditions-per-source 1 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --topk-tokens 20 \
    --log-every 2 \
    "${EXTRA_ARGS[@]}"
done

python tests/glm5/phase857_prompt_gated_causal_gear_rollout_validation.py \
  --round-name "${ROUND_NAME}" \
  --summarize-round

echo "{\"status\":\"complete\",\"round\":\"${ROUND_NAME}\",\"models\":[\"qwen3\",\"glm4\",\"deepseek7b\"]}"
