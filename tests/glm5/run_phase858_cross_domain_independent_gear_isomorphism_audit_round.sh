#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

case "${ROUND_NAME}" in
  smoke)
    DOMAINS="geometry,animal,tool"
    MAX_CASES_PER_DOMAIN=1
    PROMPTS="natural_question"
    LAYERS="26,28,30"
    PER_PROMPT_TOP=48
    MAX_NEG=1
    MAX_POS=1
    MAX_COMBO=2
    MAX_NEW_TOKENS=6
    ;;
  main)
    DOMAINS="geometry,animal,tool,color,material,abstract"
    MAX_CASES_PER_DOMAIN=1
    PROMPTS="natural_question,natural_category"
    LAYERS="24,26,28,30,32"
    PER_PROMPT_TOP=80
    MAX_NEG=2
    MAX_POS=1
    MAX_COMBO=2
    MAX_NEW_TOKENS=8
    ;;
  confirm)
    DOMAINS="geometry,animal,tool,color,material,abstract,plant,object"
    MAX_CASES_PER_DOMAIN=2
    PROMPTS="natural_question,natural_category"
    LAYERS="24,26,27,28,29,30,31,32"
    PER_PROMPT_TOP=96
    MAX_NEG=2
    MAX_POS=1
    MAX_COMBO=2
    MAX_NEW_TOKENS=8
    ;;
  *)
    echo "Unknown round: ${ROUND_NAME}" >&2
    exit 2
    ;;
esac

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase858 ${ROUND_NAME}: ${MODEL}"
  python tests/glm5/phase858_cross_domain_independent_gear_isomorphism_audit.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --domains "${DOMAINS}" \
    --max-cases-per-domain "${MAX_CASES_PER_DOMAIN}" \
    --prompt-variants "${PROMPTS}" \
    --layer-indices "${LAYERS}" \
    --per-prompt-top-channels "${PER_PROMPT_TOP}" \
    --max-negative-gears "${MAX_NEG}" \
    --max-positive-gears "${MAX_POS}" \
    --max-combo-gears "${MAX_COMBO}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --topk-tokens 20 \
    --log-every 1
done

python tests/glm5/phase858_cross_domain_independent_gear_isomorphism_audit.py \
  --round-name "${ROUND_NAME}" \
  --summarize-round

echo "{\"status\":\"complete\",\"round\":\"${ROUND_NAME}\",\"models\":[\"qwen3\",\"glm4\",\"deepseek7b\"]}"
