#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python tests/gpt5/phase292_feature_analysis_algorithm_v2.py
