#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python tests/gpt5/phase290_readout_competition_channel_decomposition.py
