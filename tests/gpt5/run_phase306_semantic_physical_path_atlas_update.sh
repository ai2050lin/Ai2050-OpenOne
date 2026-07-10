#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python tests/gpt5/phase306_semantic_physical_path_atlas_update.py
