"""Non-evidentiary CUDA smoke test for the Phase1272 vectorized mask executor."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1266_c014_free_transformer_population_certificate as p1266
import phase1269_c017_causal_support_funnel_confirmation as p1269
import phase1272_c020_cross_seed_layer_coalition as main


def run() -> None:
    device = torch.device("cuda")
    seed = 1_271_699_993
    config = main.ARCHITECTURES["deep8"]
    p1266.set_seed(seed)
    model, training = p1266.task_module.train_model(config, seed, device)
    rows = p1269.sample_worlds("development", 256, 1_272_999_001)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = main.evaluate_masks(model, rows, main.all_masks(config.layers), device)
    torch.cuda.synchronize()
    payload = {
        "status": "development_only",
        "seed": seed,
        "training_accuracy": training["accuracy_overall"],
        "masks": len(result),
        "rows": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "empty": result[0],
        "full": result[-1],
    }
    output = ROOT / "tests/glm5_temp/phase1272_c020_vectorized_mask_smoke.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    run()
