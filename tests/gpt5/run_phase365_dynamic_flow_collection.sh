#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase365_dynamic_flow_collection.py --model qwen3
python tests/gpt5/phase365_dynamic_flow_collection.py --model glm4
python tests/gpt5/phase365_dynamic_flow_collection.py --model deepseek7b
python tests/gpt5/phase365_dynamic_flow_collection_analysis.py
