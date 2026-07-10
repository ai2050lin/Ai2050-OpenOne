from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "gpt5" / "phase288_color_single_unit_heldout.py"
MODELS = ("qwen3", "glm4", "deepseek7b")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    for model in models:
        if model not in MODELS:
            raise SystemExit(f"Unknown model: {model}")
        env = dict(os.environ)
        if model == "glm4":
            env.update(PROBE_DEVICE_MAP_AUTO_MODELS="glm4", PROBE_MAX_GPU_MEMORY="11GiB")
        elif model == "deepseek7b":
            env.update(PROBE_DEVICE_MAP_AUTO_MODELS="deepseek7b", PROBE_MAX_GPU_MEMORY="12GiB")
        command = [sys.executable, str(SCRIPT), model, "--batch-size", str(args.batch_size)]
        if args.smoke:
            command.append("--smoke")
        print(f"[Phase288 runner] starting {model}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"Phase288 failed for {model}: exit={completed.returncode}")
        print(f"[Phase288 runner] released {model}", flush=True)


if __name__ == "__main__":
    main()
