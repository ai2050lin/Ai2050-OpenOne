#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-coverage_stable_boundary}"
EVAL_DOMAINS="${2:-geometry,animal,tool,color,material,abstract,plant,object}"
MAX_CASES_PER_DOMAIN="${3:-4}"
MAX_CANDIDATES_PER_MODEL="${4:-3}"
PROMPT_VARIANTS="${5:-natural_question,natural_category,object_only,classification}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS=(
  --round-name "$ROUND_NAME"
  --eval-domains "$EVAL_DOMAINS"
  --max-cases-per-domain "$MAX_CASES_PER_DOMAIN"
  --max-candidates-per-model "$MAX_CANDIDATES_PER_MODEL"
  --prompt-variants "$PROMPT_VARIANTS"
  --min-source-atlas-score "0.0"
  --exclude-labels "candidate_source_no_repair,cross_domain_side_effect,observed_pair_not_minimal,same_layer_random_control"
  --controls "same_layer_random"
  --include-subsets
  --attn-implementations "flash_attention_2,sdpa"
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "===== Phase 884 model: ${MODEL} ====="
  python tests/glm5/phase884_atlas_coverage_expansion_stable_boundary_search.py --model "$MODEL" "${COMMON_ARGS[@]}"
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

python tests/glm5/phase884_atlas_coverage_expansion_stable_boundary_search.py --round-name "$ROUND_NAME" --summarize-round
