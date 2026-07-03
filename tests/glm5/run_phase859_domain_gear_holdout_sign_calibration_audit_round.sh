#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-holdout}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

case "${ROUND_NAME}" in
  smoke)
    DOMAINS="geometry,animal,tool"
    MAX_HOLDOUT=1
    PROMPTS="natural_question"
    MAX_NEW_TOKENS=6
    SHARED_ARGS=()
    ;;
  holdout|confirm)
    DOMAINS="geometry,animal,tool,color,material,abstract,plant,object"
    MAX_HOLDOUT=2
    PROMPTS="natural_question,natural_category"
    MAX_NEW_TOKENS=8
    SHARED_ARGS=(--include-shared-exact-probe)
    ;;
  *)
    echo "Unknown round: ${ROUND_NAME}" >&2
    exit 2
    ;;
esac

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase859 ${ROUND_NAME}: ${MODEL}"
  python tests/glm5/phase859_domain_gear_holdout_sign_calibration_audit.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --source-round confirm \
    --domains "${DOMAINS}" \
    --max-holdout-cases-per-domain "${MAX_HOLDOUT}" \
    --prompt-variants "${PROMPTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --topk-tokens 20 \
    "${SHARED_ARGS[@]}"
done

python tests/glm5/phase859_domain_gear_holdout_sign_calibration_audit.py \
  --round-name "${ROUND_NAME}" \
  --summarize-round

echo "{\"status\":\"complete\",\"round\":\"${ROUND_NAME}\",\"models\":[\"qwen3\",\"glm4\",\"deepseek7b\"]}"
