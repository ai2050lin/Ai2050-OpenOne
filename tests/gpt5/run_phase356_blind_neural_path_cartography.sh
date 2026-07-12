#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase356_blind_trace_schema.py
python tests/gpt5/phase356_blind_motif_discovery.py
python tests/gpt5/phase356_posthoc_motif_validation.py
