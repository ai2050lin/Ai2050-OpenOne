#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CASE_FILE="tests/gpt5/result/phase369_raw_topology_flow/raw_collection_freeze/private/phase369_collection_execution_cases.jsonl"
OUTPUT_ROOT="tests/gpt5/result/phase369_raw_topology_flow/raw_collection"
for MODEL in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase365_dynamic_flow_collection.py \
    --model "$MODEL" \
    --case-file "$CASE_FILE" \
    --output-root "$OUTPUT_ROOT" \
    --expected-case-count 112
done
