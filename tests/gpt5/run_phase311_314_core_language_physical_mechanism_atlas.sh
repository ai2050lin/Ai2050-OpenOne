#!/usr/bin/env bash
set -euo pipefail

ROUND_311="${1:-core_language_physical_atlas}"
ROUND_313="${2:-heldout_component_interaction_audit}"

cd "$(dirname "$0")/../.."

python tests/gpt5/phase311_core_language_physical_atlas.py --prepare
python tests/gpt5/phase311_core_language_physical_atlas.py --model qwen3 --round-name "${ROUND_311}"
python tests/gpt5/phase311_core_language_physical_atlas.py --model glm4 --round-name "${ROUND_311}"
python tests/gpt5/phase311_core_language_physical_atlas.py --model deepseek7b --round-name "${ROUND_311}"
python tests/gpt5/phase311_core_language_physical_atlas.py --summarize --round-name "${ROUND_311}"

python tests/gpt5/phase312_matched_path_feature_analysis.py

python tests/gpt5/phase313_heldout_component_interaction_audit.py --model qwen3 --round-name "${ROUND_313}"
python tests/gpt5/phase313_heldout_component_interaction_audit.py --model glm4 --round-name "${ROUND_313}"
python tests/gpt5/phase313_heldout_component_interaction_audit.py --model deepseek7b --round-name "${ROUND_313}"
python tests/gpt5/phase313_heldout_component_interaction_audit.py --summarize --round-name "${ROUND_313}"

python tests/gpt5/phase314_core_mechanism_atlas_synthesis.py
