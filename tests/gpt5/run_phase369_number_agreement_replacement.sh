#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CASE_FILE="tests/gpt5/result/phase369_raw_topology_flow/raw_topology_preregister_number_agreement_replacement/private/phase369_number_agreement_execution_cases.jsonl"
for MODEL in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase369_behavior_qualification.py \
    --model "$MODEL" \
    --case-file "$CASE_FILE" \
    --run-tag number_agreement_replacement \
    --expected-case-count 36
done
python tests/gpt5/phase369_behavior_qualification_final.py
