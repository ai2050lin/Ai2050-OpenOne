#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-replicate}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

case "${ROUND_NAME}" in
  smoke)
    MAX_CASES=3
    PROMPTS="natural_question"
    MAX_NEW_TOKENS=6
    ;;
  replicate|confirm)
    MAX_CASES=5
    PROMPTS="natural_question,natural_category,classification"
    MAX_NEW_TOKENS=8
    ;;
  *)
    echo "Unknown round: ${ROUND_NAME}" >&2
    exit 2
    ;;
esac

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase860 ${ROUND_NAME}: ${MODEL}"
  python tests/glm5/phase860_replicated_domain_gear_evidence_ladder.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --source-phase858-round confirm \
    --source-phase859-round holdout \
    --min-phase859-clear-gain 1 \
    --max-cases-per-domain "${MAX_CASES}" \
    --prompt-variants "${PROMPTS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --topk-tokens 20
done

python tests/glm5/phase860_replicated_domain_gear_evidence_ladder.py \
  --round-name "${ROUND_NAME}" \
  --summarize-round

echo "{\"status\":\"complete\",\"round\":\"${ROUND_NAME}\",\"models\":[\"qwen3\",\"glm4\",\"deepseek7b\"]}"
