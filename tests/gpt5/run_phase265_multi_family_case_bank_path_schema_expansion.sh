#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase265_multi_family_case_bank_path_schema_expansion.py \
  --round-name multi_family_case_bank_path_schema_expansion
