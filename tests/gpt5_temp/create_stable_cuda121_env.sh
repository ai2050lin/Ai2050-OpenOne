#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-openone-cuda121}"

echo "=== Create stable CUDA 12.1 Python 3.11 env ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "env: ${ENV_NAME}"
echo

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment already exists: ${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" python=3.11 pip
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

# Keep this environment deliberately conservative. The active base env is
# Python 3.13 + torch cu124 + transformers 5.x, which is too new for the
# current GPU stability investigation.
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  transformers==4.52.4 \
  accelerate==1.8.1 \
  huggingface_hub==0.33.0 \
  numpy==1.26.4 \
  safetensors==0.5.3 \
  sentencepiece==0.2.0 \
  protobuf==5.29.5 \
  einops==0.8.1 \
  tqdm==4.67.1

python -m pip install -e . --no-deps
python -m pip install \
  beartype==0.14.1 \
  better-abc==0.0.3 \
  datasets==2.21.0 \
  fancy-einsum==0.0.3 \
  jaxtyping==0.2.38 \
  pandas==2.2.3 \
  rich==13.9.4 \
  transformers-stream-generator==0.0.5 \
  typeguard==4.4.2 \
  wandb==0.17.9

python - <<'PY'
import sys
import torch
import numpy
import transformers
import accelerate
import transformer_lens
print("python", sys.version)
print("numpy", numpy.__version__)
print("torch", torch.__version__, "runtime", torch.version.cuda)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("transformer_lens", getattr(transformer_lens, "__version__", "local-editable"))
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo
echo "Use it with:"
echo "  conda activate ${ENV_NAME}"
