#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase365_repeat_noise_format_gate.py --model qwen3
python tests/gpt5/phase365_repeat_noise_format_gate.py --model glm4
python tests/gpt5/phase365_repeat_noise_format_gate.py --model deepseek7b
python tests/gpt5/phase365_repeat_noise_analysis.py
