#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/glm5/phase886_local_subspace_boundary_decomposition.py --model qwen3
python tests/glm5/phase886_local_subspace_boundary_decomposition.py --model glm4
python tests/glm5/phase886_local_subspace_boundary_decomposition.py --model deepseek7b
python tests/glm5/phase886_local_subspace_boundary_decomposition.py --summarize
