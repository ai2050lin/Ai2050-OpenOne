from __future__ import annotations

import json

import torch
import transformers
import accelerate
import huggingface_hub
import transformer_lens

from hf_probe_env import local_model_status


def main() -> None:
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            if torch.cuda.is_available()
            else 0
        ),
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "transformer_lens": getattr(transformer_lens, "__version__", "local-editable"),
        "models": local_model_status(),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
