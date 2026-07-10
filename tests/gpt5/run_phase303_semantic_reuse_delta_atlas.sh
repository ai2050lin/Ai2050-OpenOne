#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python tests/gpt5/phase303_semantic_reuse_delta_atlas.py
