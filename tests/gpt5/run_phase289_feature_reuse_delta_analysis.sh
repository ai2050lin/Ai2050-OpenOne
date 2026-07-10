#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 tests/gpt5/phase289_feature_reuse_delta_analysis.py
