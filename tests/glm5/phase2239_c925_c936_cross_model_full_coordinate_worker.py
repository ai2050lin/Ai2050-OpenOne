from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase2163_c629_model_specific_worker as model_worker  # noqa: E402
import phase2234_c870_c884_broad_family_gear_contract as contract  # noqa: E402
import phase2235_c885_c904_qwen_broad_family_full_coordinate_tournament as qwen_stage  # noqa: E402


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def load_model(name: str):
    if name == "qwen3":
        model, tokenizer, device, placement = contract.prior.qwen_model()
        return model, tokenizer, device, placement, "qwen3_full_cuda"
    return model_worker.load_model(name)


def release_model(name: str, model) -> None:
    if name == "qwen3":
        qwen_stage.release_model(model)
    else:
        model_worker.release_model(name, model)
    gc.collect()


def generation_behavior(model, tokenizer, device, rows: list[dict], batch_size: int) -> list[dict]:
    output = []
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["free_prompt_ids"]
            ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, width - len(seq):] = 1
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6,
                                       do_sample=False, pad_token_id=pad,
                                       eos_token_id=tokenizer.eos_token_id)
        for i, row in enumerate(batch):
            text = tokenizer.decode(generated[i, width:].tolist(), skip_special_tokens=True)
            parsed = parse_code(text, row)
            output.append({"case_id": row["case_id"], "text": text, "parsed": parsed,
                           "correct": parsed == row["correct_answer"]})
        print(f"[generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def model_modules(model):
    base = model.model
    return [base.embed_tokens, *list(base.layers), base.norm]


def capture_full_field(model, device, rows: list[dict], output_dir: Path) -> dict:
    modules = model_modules(model)
    dim = int(modules[0].weight.shape[1])
    path = output_dir / "raw/exact_semantic_role_field.float16.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                      shape=(len(rows), len(modules), len(contract.ROLES), dim))
    captured = []

    def hook(_module, _args, value):
        captured.append(value[0] if isinstance(value, tuple) else value)

    handles = [module.register_forward_hook(hook) for module in modules]
    try:
        for row_i, row in enumerate(rows):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            pos = mask.long().cumsum(-1) - 1
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            if len(captured) != len(modules):
                raise RuntimeError(("checkpoint_count", len(captured), len(modules)))
            for q, hidden in enumerate(captured):
                values = hidden[0].float().cpu().numpy().astype(np.float16)
                for role_i, role in enumerate(contract.ROLES):
                    field[row_i, q, role_i] = values[row["role_positions"][role][-1]]
            if row_i % 8 == 0:
                print(f"[field] {row_i}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    field.flush()
    mmap = getattr(field, "_mmap", None)
    if mmap is not None:
        mmap.close()
    write_rows(output_dir / "raw/field_index.jsonl", [
        {"hidden_index": i, "case_id": row["case_id"], "family": row["family"],
         "language": row["language"], "surface": row["surface"], "truth": row["truth"],
         "unit": row["unit"], "role_positions": row["role_positions"]}
        for i, row in enumerate(rows)
    ])
    return {"path": str(path.relative_to(ROOT)), "shape": [len(rows), len(modules), len(contract.ROLES), dim],
            "checkpoints": len(modules), "coordinates": dim}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen3", "qwen3_14b", "glm4", "deepseek7b"), required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.material = args.material.resolve()
    args.output = args.output.resolve()
    final_path = args.output / "analysis/final.json"
    source_rows = read_rows(args.material)
    candidate_path = args.output / "behavior/candidate.jsonl"
    generation_path = args.output / "behavior/generation.jsonl"
    field_path = args.output / "raw/exact_semantic_role_field.float16.npy"
    index_path = args.output / "raw/field_index.jsonl"
    if all(path.exists() for path in (candidate_path, generation_path, field_path, index_path)):
        candidate = read_rows(candidate_path); generated = read_rows(generation_path)
        candidate_accuracy = float(np.mean([row["correct"] for row in candidate]))
        generation_accuracy = float(np.mean([row["correct"] for row in generated]))
        shape = list(np.load(field_path, mmap_mode="r").shape)
        checks = {"compiled_all": True, "behavior_complete": len(candidate) == len(generated) == len(source_rows),
                  "finite_accuracy": bool(np.isfinite(candidate_accuracy) and np.isfinite(generation_accuracy)),
                  "hidden_iff_qualified": candidate_accuracy >= contract.BEHAVIOR_GATE and generation_accuracy >= contract.BEHAVIOR_GATE}
        result = {"status": "closed", "model": args.model, "rows": len(source_rows),
                  "candidate_accuracy": candidate_accuracy, "generation_accuracy": generation_accuracy,
                  "behavior_qualified": True,
                  "field": {"ran": True, "path": str(field_path.relative_to(ROOT)), "shape": shape,
                            "checkpoints": shape[1], "coordinates": shape[-1]},
                  "checks": checks, "all_checks_passed": all(checks.values()),
                  "placement": {"resume": "complete behavior and full field artifacts"},
                  "loader": "artifact_resume",
                  "strict_interpretation": "Physical coordinates are complete within this model and are never aligned by index across models."}
        save(final_path, result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    if final_path.exists():
        previous = json.loads(final_path.read_text(encoding="utf-8"))
        if previous.get("status") != "worker_error":
            print(json.dumps(previous, ensure_ascii=False, indent=2)); return
    model = None
    try:
        model, tokenizer, device, placement, loader = load_model(args.model)
        rows = contract.compile_rows(tokenizer, source_rows)
        candidate = contract.prior.behavior_base.batch_behavior(
            model, device, rows, batch_size=2 if args.model == "qwen3_14b" else 12)
        generated = generation_behavior(model, tokenizer, device, rows,
                                        batch_size=2 if args.model == "qwen3_14b" else 8)
        write_rows(args.output / "behavior/candidate.jsonl", candidate)
        write_rows(args.output / "behavior/generation.jsonl", generated)
        candidate_accuracy = float(np.mean([row["correct"] for row in candidate]))
        generation_accuracy = float(np.mean([row["correct"] for row in generated]))
        qualified = candidate_accuracy >= contract.BEHAVIOR_GATE and generation_accuracy >= contract.BEHAVIOR_GATE
        field = {"ran": False}
        if qualified:
            field = {"ran": True, **capture_full_field(model, device, rows, args.output)}
        checks = {"compiled_all": len(rows) == len(source_rows), "behavior_complete": len(candidate) == len(generated) == len(rows),
                  "finite_accuracy": bool(np.isfinite(candidate_accuracy) and np.isfinite(generation_accuracy)),
                  "hidden_iff_qualified": field["ran"] == qualified}
        result = {
            "status": "closed" if qualified else "behavior_unqualified", "model": args.model,
            "rows": len(rows), "candidate_accuracy": candidate_accuracy,
            "generation_accuracy": generation_accuracy, "behavior_qualified": qualified,
            "field": field, "checks": checks, "all_checks_passed": all(checks.values()),
            "placement": placement, "loader": loader,
            "strict_interpretation": "Physical coordinates are complete within this model and are never aligned by index across models.",
        }
        save(final_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        save(final_path, {"status": "worker_error", "model": args.model,
                          "error_type": type(error).__name__, "error": str(error),
                          "all_checks_passed": False})
        raise
    finally:
        release_model(args.model, model)


if __name__ == "__main__":
    main()
