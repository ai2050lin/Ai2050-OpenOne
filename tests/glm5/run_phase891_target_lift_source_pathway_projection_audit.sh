#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-target_lift_source_projection}"

python tests/glm5/phase891_target_lift_source_pathway_projection_audit.py --model qwen3 --round-name "$ROUND_NAME"
python tests/glm5/phase891_target_lift_source_pathway_projection_audit.py --model glm4 --round-name "$ROUND_NAME"
python tests/glm5/phase891_target_lift_source_pathway_projection_audit.py --model deepseek7b --round-name "$ROUND_NAME"
python tests/glm5/phase891_target_lift_source_pathway_projection_audit.py --round-name "$ROUND_NAME" --summarize-round
