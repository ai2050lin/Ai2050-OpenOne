#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python3 tests/gpt5/phase280_gap_recalibration_after_phase279.py
