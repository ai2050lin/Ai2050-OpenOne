#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CASE_FILE="tests/gpt5/result/phase365_dynamic_flow_instrumentation/phase366_remaining_collection/private/phase366_remaining_execution_cases.jsonl"
OUT="tests/gpt5/result/phase365_dynamic_flow_instrumentation/engineering_collection"

python tests/gpt5/phase366_remaining_collection_freeze.py
python tests/gpt5/phase365_dynamic_flow_collection.py --model qwen3 --case-file "$CASE_FILE" --output-root "$OUT" --expected-case-count 64
python tests/gpt5/phase365_dynamic_flow_collection.py --model glm4 --case-file "$CASE_FILE" --output-root "$OUT" --expected-case-count 64
python tests/gpt5/phase365_dynamic_flow_collection.py --model deepseek7b --case-file "$CASE_FILE" --output-root "$OUT" --expected-case-count 64
python tests/gpt5/phase366_merge_collection_manifests.py
