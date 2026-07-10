#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase330_registered_causal_audit.py --register
for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase330_registered_causal_audit.py --model "$model" --max-new-tokens 8
done
python tests/gpt5/phase330_registered_causal_audit.py --collect
