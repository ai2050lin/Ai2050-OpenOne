from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1274_c021_multitask_free_response_isomorphism as base  # noqa: E402


CHECKPOINTS = (0, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5500, 7000)
CELLS = (
    ("xor", "shallow4"),
    ("cyclic", "deep8"),
    ("context_lookup", "shallow4"),
)


def evaluate(model, task: str, seed: int, device: torch.device) -> dict[str, float]:
    inputs, labels = base.random_batch(task, 4096, seed)
    with torch.inference_mode():
        logits = model(inputs.to(device))[:, -1, base.CANDIDATE_SLICE].float()
        labels_device = labels.to(device)
        loss = F.cross_entropy(logits, labels_device)
        accuracy = (logits.argmax(-1) == labels_device).float().mean()
    return {"loss": float(loss.item()), "accuracy": float(accuracy.item())}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    rows: list[dict[str, object]] = []
    for task, architecture in CELLS:
        for seed_index in range(3):
            seed = base.MODEL_SEEDS[f"{task}.{architecture}.s{seed_index}"]
            base.set_seed(seed)
            model = base.TinyCausalTransformer(base.ARCHITECTURES[architecture]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-3)
            checkpoint_set = set(CHECKPOINTS)
            for step in range(max(CHECKPOINTS) + 1):
                if step in checkpoint_set:
                    model.eval()
                    metrics = evaluate(model, task, seed + 91_000_000 + step, device)
                    row = {
                        "task": task,
                        "architecture": architecture,
                        "seed_index": seed_index,
                        "seed": seed,
                        "step": step,
                        **metrics,
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
                    model.train()
                if step == max(CHECKPOINTS):
                    break
                if step == 3500:
                    optimizer.param_groups[0]["lr"] = 2.0e-4
                inputs, labels = base.random_batch(task, base.TRAINING_BATCH, seed + 10_000 + step)
                logits = model(inputs.to(device))[:, -1, base.CANDIDATE_SLICE].float()
                loss = F.cross_entropy(logits, labels.to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            del model, optimizer
            torch.cuda.empty_cache()
    output = ROOT / "tests/glm5_temp/phase1277_c022_cutoff_probe.json"
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
