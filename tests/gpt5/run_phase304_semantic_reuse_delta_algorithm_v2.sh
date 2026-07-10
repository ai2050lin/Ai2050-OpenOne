#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python tests/gpt5/phase304_semantic_reuse_delta_algorithm_v2.py
