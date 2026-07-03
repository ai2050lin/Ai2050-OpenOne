#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-replication}"
MAX_CASES_PER_DOMAIN="${2:-6}"
PROMPT_VARIANTS="${3:-replication_direct,replication_sentence,replication_form}"
INCLUDE_CONTROLS="${4:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS=(
  --round-name "$ROUND_NAME"
  --max-cases-per-domain "$MAX_CASES_PER_DOMAIN"
  --prompt-variants "$PROMPT_VARIANTS"
  --attn-implementations "flash_attention_2,sdpa"
)

if [[ "$INCLUDE_CONTROLS" == "1" ]]; then
  COMMON_ARGS+=(--include-non-clean-controls)
fi

for MODEL in qwen3 glm4 deepseek7b; do
  echo "===== Phase 873 model: ${MODEL} ====="
  python tests/glm5/phase873_output_gate_external_replication.py --model "$MODEL" "${COMMON_ARGS[@]}"
  python - <<'PY'
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
except Exception as exc:
    print(f"cuda cleanup skipped: {exc}")
gc.collect()
PY
done

python tests/glm5/phase873_output_gate_external_replication.py --round-name "$ROUND_NAME" --summarize-round
