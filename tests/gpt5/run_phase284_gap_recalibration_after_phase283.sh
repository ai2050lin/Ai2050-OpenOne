#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 tests/gpt5/phase284_gap_recalibration_after_phase283.py
