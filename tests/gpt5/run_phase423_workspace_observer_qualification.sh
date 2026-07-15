#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VENDOR="$ROOT/tests/gpt5_temp/phase423_vendor/jacobian-lens"
COMMIT="581d398613e5602a5af361e1c34d3a92ea82ba8e"
if [[ ! -d "$VENDOR/.git" ]]; then
  rm -rf "$VENDOR"
  git clone --depth 1 https://github.com/anthropics/jacobian-lens.git "$VENDOR"
fi
if [[ "$(git -C "$VENDOR" rev-parse HEAD)" != "$COMMIT" ]]; then
  git -C "$VENDOR" fetch --depth 1 origin "$COMMIT"
  git -C "$VENDOR" checkout --detach "$COMMIT"
fi
export PYTHONPATH="$VENDOR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PROBE_TORCH_DTYPE=bfloat16

python tests/gpt5/phase423_workspace_observer_protocol.py \
  --official-root "$VENDOR" --reuse-frozen

python tests/gpt5/phase423_workspace_observer_fit.py --model qwen3
python tests/gpt5/phase423_workspace_observer_evaluate.py --model qwen3

python tests/gpt5/phase423_workspace_observer_fit.py --model glm4
python tests/gpt5/phase423_workspace_observer_evaluate.py --model glm4

python tests/gpt5/phase423_workspace_observer_fit.py --model deepseek7b
python tests/gpt5/phase423_workspace_observer_evaluate.py --model deepseek7b

python tests/gpt5/phase423_workspace_observer_analysis.py
python -m unittest tests.gpt5.test_phase423_workspace_observer_qualification
