#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase369_head_neuron_topology.py --model qwen3
python tests/gpt5/phase369_head_neuron_topology.py --model glm4
python tests/gpt5/phase369_head_neuron_topology.py --model deepseek7b
python tests/gpt5/phase369_head_neuron_topology.py --summarize
