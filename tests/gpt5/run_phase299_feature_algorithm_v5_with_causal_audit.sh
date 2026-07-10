#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python tests/gpt5/phase299_feature_algorithm_v5_with_causal_audit.py
