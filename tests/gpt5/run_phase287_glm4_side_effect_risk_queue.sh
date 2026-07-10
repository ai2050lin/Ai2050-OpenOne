#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 tests/gpt5/phase287_glm4_side_effect_risk_queue.py
