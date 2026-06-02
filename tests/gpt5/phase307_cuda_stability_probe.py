from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from hf_probe_env import load_probe_model, release_loaded
from phase289_contract_scan import tokenize


REPO_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[phase307] {message}", flush=True)


def cuda_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        out.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
                "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            }
        )
    return out


def torch_tensor_probe(size: int, repeats: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda:0")
    start = time.time()
    last_norm = 0.0
    for idx in range(repeats):
        x = torch.randn((size, size), device=device, dtype=torch.float16)
        y = x @ x.T
        last_norm = float(y.float().norm().detach().cpu())
        del x, y
        torch.cuda.synchronize()
        log(f"tensor repeat={idx + 1}/{repeats} norm={last_norm:.4f}")
    return {"mode": "tensor", "size": size, "repeats": repeats, "last_norm": last_norm, "elapsed": time.time() - start}


def model_load_probe(model_key: str) -> dict[str, Any]:
    loaded = None
    start = time.time()
    try:
        loaded = load_probe_model(model_key)
        return {
            "mode": "model_load",
            "model": model_key,
            "class": type(loaded.model).__name__,
            "input_device": str(loaded.input_device),
            "elapsed": time.time() - start,
            "snapshot": cuda_snapshot(),
        }
    finally:
        release_loaded(loaded)


def model_forward_probe(model_key: str, prompt: str, max_seq_len: int, repeats: int) -> dict[str, Any]:
    loaded = None
    start = time.time()
    rows = []
    try:
        loaded = load_probe_model(model_key)
        seq_len = min(max(len(loaded.tokenizer.encode(prompt, add_special_tokens=True)), 8), max_seq_len)
        batch = tokenize(loaded, prompt, seq_len)
        for idx in range(repeats):
            with torch.no_grad():
                out = loaded.model(**batch)
            logits = out.logits[0, -1, :].detach().float()
            finite = bool(torch.isfinite(logits).all().item())
            max_logit = float(logits.max().cpu())
            rows.append({"repeat": idx + 1, "finite": finite, "max_logit": max_logit})
            log(f"forward repeat={idx + 1}/{repeats} finite={finite} max_logit={max_logit:.4f}")
            del out, logits
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return {
            "mode": "model_forward",
            "model": model_key,
            "class": type(loaded.model).__name__,
            "prompt": prompt,
            "seq_len": seq_len,
            "rows": rows,
            "elapsed": time.time() - start,
            "snapshot": cuda_snapshot(),
        }
    finally:
        release_loaded(loaded)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"mode={args.mode} model={args.model}")
    before = cuda_snapshot()
    if args.mode == "tensor":
        result = torch_tensor_probe(args.tensor_size, args.repeats)
    elif args.mode == "model_load":
        result = model_load_probe(args.model)
    elif args.mode == "model_forward":
        result = model_forward_probe(args.model, args.prompt, args.max_seq_len, args.repeats)
    else:
        raise ValueError(f"unknown mode={args.mode}")
    data = {
        "complete": True,
        "mode": args.mode,
        "model": args.model,
        "before": before,
        "result": result,
        "after": cuda_snapshot(),
        "env": {
            "PROBE_TORCH_DTYPE": os.environ.get("PROBE_TORCH_DTYPE"),
            "PROBE_ATTN_IMPLEMENTATION": os.environ.get("PROBE_ATTN_IMPLEMENTATION"),
            "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            "PYTORCH_NO_CUDA_MEMORY_CACHING": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
        },
    }
    out_file = output_dir / f"phase307_{args.mode}_{args.model}.json"
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log(f"saved {out_file}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tensor", "model_load", "model_forward"], required=True)
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase307_cuda_stability_probe"))
    parser.add_argument("--tensor-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prompt", default="The dog chased the cat. The agent is")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        data = run(args)
        log(f"done complete={data['complete']}")
    finally:
        if args.hard_exit_after_model:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
