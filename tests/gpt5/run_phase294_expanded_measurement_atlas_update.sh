#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python tests/gpt5/phase294_expanded_measurement_atlas_update.py
