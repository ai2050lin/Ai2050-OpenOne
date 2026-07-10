#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python tests/gpt5/phase300_pattern_atlas_v3_candidate_synthesis.py
