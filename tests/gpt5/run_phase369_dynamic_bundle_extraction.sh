#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase365_dynamic_bundle_extraction.py \
  --models qwen3 glm4 deepseek7b \
  --device cuda \
  --resume \
  --collection-root tests/gpt5/result/phase369_raw_topology_flow/raw_collection \
  --output-root tests/gpt5/result/phase369_raw_topology_flow/dynamic_bundle_extraction
