#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase273_pattern_family_atlas_v2_system_build.py
