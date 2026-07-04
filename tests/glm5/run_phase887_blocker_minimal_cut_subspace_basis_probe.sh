#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/glm5/phase887_blocker_minimal_cut_subspace_basis_probe.py --model qwen3
python tests/glm5/phase887_blocker_minimal_cut_subspace_basis_probe.py --model glm4
python tests/glm5/phase887_blocker_minimal_cut_subspace_basis_probe.py --model deepseek7b
python tests/glm5/phase887_blocker_minimal_cut_subspace_basis_probe.py --summarize
