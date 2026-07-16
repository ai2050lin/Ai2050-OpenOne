#!/usr/bin/env python3
"""Phase 961 single-model runner — runs one model at a time to avoid timeout."""
import sys, gc, torch, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase961_mode_head_mechanism import (
    run_model, task6_cross_model, ensure_dir, RESULT_DIR, log
)

model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
log(f"Phase 961 single-model runner: {model_name}")

ensure_dir(RESULT_DIR)
mr = run_model(model_name)

# Save model result
save_path = RESULT_DIR / f"{model_name}_result.json"
save_path.write_text(json.dumps(mr, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
log(f"Saved: {save_path}")
log(f"Done: {model_name}")
