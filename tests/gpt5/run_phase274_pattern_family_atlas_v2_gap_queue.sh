#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python3 tests/gpt5/phase274_pattern_family_atlas_v2_gap_queue.py
