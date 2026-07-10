#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python tests/gpt5/phase308_three_position_semantic_atlas_update.py
