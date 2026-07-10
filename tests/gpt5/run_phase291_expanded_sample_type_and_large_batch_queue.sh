#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python tests/gpt5/phase291_expanded_sample_type_and_large_batch_queue.py
