#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python tests/gpt5/phase295_feature_algorithm_v3_after_expansion.py
