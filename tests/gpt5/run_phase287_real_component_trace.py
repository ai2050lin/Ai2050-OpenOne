#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "gpt5" / "phase287_real_component_trace.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase287 one model at a time to protect GPU memory")
    parser.add_argument("--models", default="qwen3,glm4,deepseek7b")
    parser.add_argument("--prompt", default="A red cube is placed on the table. The color of the cube is")
    parser.add_argument("--target-label", default="red")
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()
    results = []
    for model in [part.strip() for part in args.models.split(",") if part.strip()]:
        child_env = dict(os.environ)
        if model in {"glm4", "deepseek7b"}:
            child_env["PROBE_DEVICE_MAP_AUTO_MODELS"] = model
            child_env.setdefault("PROBE_MAX_GPU_MEMORY", "11GiB" if model == "glm4" else "12GiB")
            child_env.setdefault("PROBE_MAX_CPU_MEMORY", "64GiB")
            child_env["PROBE_TORCH_DTYPE"] = "bfloat16"
        command = [
            sys.executable,
            str(SCRIPT),
            "--model",
            model,
            "--prompt",
            args.prompt,
            "--target-label",
            args.target_label,
            "--top-k",
            str(args.top_k),
            "--round-name",
            f"phase287_{model}_{args.target_label}_component_trace",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False, env=child_env)
        results.append({"model": model, "return_code": completed.returncode})
        if completed.returncode != 0:
            break
    print(json.dumps({"phase": 287, "results": results}, ensure_ascii=False, indent=2))
    if any(row["return_code"] != 0 for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
