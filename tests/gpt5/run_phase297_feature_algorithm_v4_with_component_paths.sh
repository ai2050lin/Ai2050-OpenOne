#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python tests/gpt5/phase297_feature_algorithm_v4_with_component_paths.py
