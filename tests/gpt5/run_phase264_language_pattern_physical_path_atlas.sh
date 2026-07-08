#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase264_language_pattern_physical_path_atlas.py \
  --round-name language_pattern_physical_path_atlas
