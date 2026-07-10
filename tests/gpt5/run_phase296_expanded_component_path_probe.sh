#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CASES="${1:-9}"
python tests/gpt5/phase296_expanded_component_path_probe.py --cases-per-model "$CASES"
