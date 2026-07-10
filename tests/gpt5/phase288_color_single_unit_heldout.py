from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

GPT5_DIR = Path(__file__).resolve().parent
if str(GPT5_DIR) not in sys.path:
    sys.path.insert(0, str(GPT5_DIR))

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb


ROOT = Path(__file__).resolve().parents[2]
KERNEL_ROOT = ROOT / "tests" / "result" / "research_kernel"
RESULT_ROOT = ROOT / "tests" / "result" / "phase288_color_single_unit_heldout"

COLORS = ("red", "blue", "green", "yellow", "orange", "purple", "brown", "black", "white", "gray", "silver", "pink")
OBJECTS = (
    "cube", "sphere", "box", "card", "flag", "cup", "book", "chair", "door", "bottle",
    "lamp", "plate", "bag", "ribbon", "stone", "key", "phone", "pencil", "shirt", "tile",
)
TEMPLATES = (
    "A {color} {object} is on the table. The color of the {object} is",
    "The {object} was painted {color}. Its color is",
    "Question: what color is the {object}? Context: the {object} is {color}. Answer:",
    "In the scene, there is a {color} {object}. The {object}'s color is",
    "Remember that the {object} is {color}. Complete: The {object} is",
    "Color fact: {object} = {color}. Therefore, the color is",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidate_path(model: str) -> Path:
    return KERNEL_ROOT / "runs" / f"phase286_color_real_units_{model}" / "unit_evidence.jsonl"


def select_preregistered_candidates(model: str, colors: tuple[str, ...]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows = read_jsonl(candidate_path(model))
    selected: dict[str, dict[str, Any]] = {}
    for color in colors:
        matches = [row for row in rows if row.get("target_label") == color]
        if matches:
            selected[color] = max(matches, key=lambda row: float(row.get("candidate_score") or 0.0))
    missing = [color for color in colors if color not in selected]
    return selected, missing


def build_cases(colors: tuple[str, ...], object_count: int, template_count: int) -> list[dict[str, Any]]:
    rows = []
    for color in colors:
        for object_index, object_name in enumerate(OBJECTS[:object_count]):
            for template_index, template in enumerate(TEMPLATES[:template_count]):
                rows.append({
                    "case_id": f"heldout:{color}:{object_name}:t{template_index}",
                    "color": color,
                    "object": object_name,
                    "object_index": object_index,
                    "template_index": template_index,
                    "prompt": template.format(color=color, object=object_name),
                    "split": "heldout_phase288",
                })
    return rows


def first_token_id(tokenizer: Any, label: str) -> int:
    candidates = []
    for text in (label, f" {label}"):
        ids = tokenizer(text, add_special_tokens=False).get("input_ids") or []
        if ids:
            candidates.append(int(ids[0]))
    if not candidates:
        raise ValueError(f"Could not tokenize {label!r}")
    return candidates[-1]


def last_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.long().sum(dim=1) - 1


def batch_inputs(loaded: Any, prompts: list[str]) -> dict[str, torch.Tensor]:
    batch = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=96)
    return {key: value.to(loaded.input_device) for key, value in batch.items()}


def down_projection(loaded: Any, layer: int) -> Any:
    layers = get_layers(loaded.model)
    if not (0 <= layer < len(layers)):
        raise ValueError(f"Layer {layer} outside 0..{len(layers) - 1}")
    module = getattr(getattr(layers[layer], "mlp", None), "down_proj", None)
    if module is None:
        raise TypeError(f"No MLP down_proj at layer {layer}")
    return module


def forward_logits(loaded: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        output = loaded.model(**inputs, use_cache=False, return_dict=True)
    positions = last_positions(inputs["attention_mask"])
    batch_indices = torch.arange(positions.shape[0], device=output.logits.device)
    return output.logits[batch_indices, positions.to(output.logits.device)].detach().float().cpu()


def baseline_with_product(loaded: Any, inputs: dict[str, torch.Tensor], layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, module_inputs: tuple[Any, ...]) -> None:
        captured["product"] = module_inputs[0].detach().float().cpu()

    handle = down_projection(loaded, layer).register_forward_pre_hook(hook)
    try:
        logits = forward_logits(loaded, inputs)
    finally:
        handle.remove()
    if "product" not in captured:
        raise RuntimeError("MLP product activation was not captured")
    return logits, captured["product"]


def intervened_logits(
    loaded: Any,
    inputs: dict[str, torch.Tensor],
    layer: int,
    unit_indices: list[int],
    scale: float,
) -> torch.Tensor:
    positions = last_positions(inputs["attention_mask"]).tolist()

    def hook(_module: Any, module_inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        product = module_inputs[0].clone()
        for row, (position, unit_index) in enumerate(zip(positions, unit_indices)):
            product[row, int(position), int(unit_index)] *= scale
        return (product,) + module_inputs[1:]

    handle = down_projection(loaded, layer).register_forward_pre_hook(hook)
    try:
        return forward_logits(loaded, inputs)
    finally:
        handle.remove()


def choose_matched_units(product: torch.Tensor, positions: torch.Tensor, candidate_index: int, case_ids: list[str]) -> list[int]:
    width = product.shape[-1]
    matched = []
    for row, (position, case_id) in enumerate(zip(positions.tolist(), case_ids)):
        target_abs = float(product[row, int(position), candidate_index].abs())
        seed = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16)
        pool = [int((seed + 104729 * offset) % width) for offset in range(1, min(width, 257))]
        pool = [index for index in pool if index != candidate_index]
        best = min(pool, key=lambda index: abs(float(product[row, int(position), index].abs()) - target_abs))
        matched.append(best)
    return matched


def color_margin(logits: torch.Tensor, target_color: str, color_ids: dict[str, int]) -> tuple[float, int]:
    target = float(logits[color_ids[target_color]])
    competitors = [(color, float(logits[token_id])) for color, token_id in color_ids.items() if color != target_color]
    winner_color, winner_logit = max(competitors, key=lambda item: item[1])
    rank = 1 + sum(float(logits[token_id]) > target for token_id in color_ids.values())
    return target - winner_logit, rank


def arithmetic_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def evaluate_group(
    loaded: Any,
    cases: list[dict[str, Any]],
    candidate: dict[str, Any],
    color_ids: dict[str, int],
    batch_size: int,
    run_id: str,
) -> list[dict[str, Any]]:
    rows = []
    layer = int(candidate["layer"])
    candidate_index = int(candidate["unit_index"])
    for start in range(0, len(cases), batch_size):
        batch_cases = cases[start:start + batch_size]
        inputs = batch_inputs(loaded, [case["prompt"] for case in batch_cases])
        baseline, product = baseline_with_product(loaded, inputs, layer)
        positions = last_positions(inputs["attention_mask"]).cpu()
        candidate_units = [candidate_index] * len(batch_cases)
        matched_units = choose_matched_units(product, positions, candidate_index, [case["case_id"] for case in batch_cases])
        zeroed = intervened_logits(loaded, inputs, layer, candidate_units, 0.0)
        half = intervened_logits(loaded, inputs, layer, candidate_units, 0.5)
        matched = intervened_logits(loaded, inputs, layer, matched_units, 0.0)
        for index, case in enumerate(batch_cases):
            baseline_margin, baseline_rank = color_margin(baseline[index], case["color"], color_ids)
            zero_margin, zero_rank = color_margin(zeroed[index], case["color"], color_ids)
            half_margin, _ = color_margin(half[index], case["color"], color_ids)
            random_margin, _ = color_margin(matched[index], case["color"], color_ids)
            rows.append({
                "schema_version": "single_unit_intervention.v1",
                "run_id": run_id,
                **case,
                "model": loaded.key,
                "layer": layer,
                "component": "mlp_product",
                "unit_kind": "mlp_product_neuron",
                "unit_index": candidate_index,
                "matched_control_unit_index": matched_units[index],
                "token_position": int(positions[index]),
                "candidate_activation": float(product[index, int(positions[index]), candidate_index]),
                "baseline_margin": baseline_margin,
                "zero_margin": zero_margin,
                "half_margin": half_margin,
                "matched_random_margin": random_margin,
                "zero_margin_delta": zero_margin - baseline_margin,
                "half_margin_delta": half_margin - baseline_margin,
                "matched_random_margin_delta": random_margin - baseline_margin,
                "effect_beyond_matched_control": (zero_margin - baseline_margin) - (random_margin - baseline_margin),
                "baseline_color_rank": baseline_rank,
                "zero_color_rank": zero_rank,
                "causal_scope": "single_physical_unit",
                "evidence_level": "L4" if "smoke" in run_id else "L5_candidate_requires_replication",
                "source_candidate_run": candidate["run_id"],
                "source_candidate_artifact": candidate["source_artifact"],
            })
        print(f"[Phase288] {loaded.key} color={cases[0]['color']} {min(start + batch_size, len(cases))}/{len(cases)}", flush=True)
    return rows


def clean_side_effect(loaded: Any, candidates: dict[str, dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    rows = []
    prompt = "The capital city of France is"
    for color, candidate in candidates.items():
        inputs = batch_inputs(loaded, [prompt])
        baseline = forward_logits(loaded, inputs)[0]
        zeroed = intervened_logits(
            loaded,
            inputs,
            int(candidate["layer"]),
            [int(candidate["unit_index"])],
            0.0,
        )[0]
        top_ids = torch.topk(baseline, k=32).indices
        mean_abs_delta = float((zeroed[top_ids] - baseline[top_ids]).abs().mean())
        baseline_prob = F.softmax(baseline, dim=-1)
        kl = float(F.kl_div(F.log_softmax(zeroed, dim=-1), baseline_prob, reduction="sum"))
        rows.append({
            "schema_version": "single_unit_clean_control.v1",
            "run_id": run_id,
            "model": loaded.key,
            "color_candidate": color,
            "prompt": prompt,
            "layer": int(candidate["layer"]),
            "unit_index": int(candidate["unit_index"]),
            "top32_mean_abs_logit_delta": mean_abs_delta,
            "kl_from_baseline": kl,
        })
    return rows


def summarize(rows: list[dict[str, Any]], clean_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_color: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_color[row["color"]].append(row)
    clean_by_color = {row["color_candidate"]: row for row in clean_rows}
    colors = {}
    for color, color_rows in sorted(by_color.items()):
        total = len(color_rows)
        negative = sum(row["zero_margin_delta"] < 0 for row in color_rows)
        beyond_control = sum(row["effect_beyond_matched_control"] < 0 for row in color_rows)
        monotonic = sum(row["zero_margin_delta"] <= row["half_margin_delta"] <= 0 for row in color_rows)
        baseline_rank1 = sum(row["baseline_color_rank"] == 1 for row in color_rows)
        clean = clean_by_color.get(color, {})
        colors[color] = {
            "cases": total,
            "baseline_rank1_count": baseline_rank1,
            "negative_effect_count": negative,
            "stronger_than_matched_control_count": beyond_control,
            "monotonic_half_zero_count": monotonic,
            "mean_zero_margin_delta": arithmetic_mean([row["zero_margin_delta"] for row in color_rows]),
            "mean_half_margin_delta": arithmetic_mean([row["half_margin_delta"] for row in color_rows]),
            "mean_matched_random_margin_delta": arithmetic_mean([row["matched_random_margin_delta"] for row in color_rows]),
            "clean_top32_mean_abs_logit_delta": clean.get("top32_mean_abs_logit_delta"),
            "passes_preregistered_screen": (
                total > 0
                and negative / total >= 0.8
                and beyond_control / total >= 0.7
                and monotonic / total >= 0.7
                and float(clean.get("top32_mean_abs_logit_delta") or math.inf) <= 0.1
            ),
        }
    return {
        "colors": colors,
        "screen_pass_count": sum(bool(row["passes_preregistered_screen"]) for row in colors.values()),
        "tested_color_count": len(colors),
        "total_case_count": len(rows),
        "boundary": "A pass is only an L5 candidate for next-token color-margin necessity. It is not sufficiency, rollout stability, or clean mechanism closure.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode = "smoke" if args.smoke else "full"
    run_id = f"phase288_{args.model}_color_single_unit_{mode}"
    run_dir = RESULT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_colors = tuple(COLORS[:2] if args.smoke else COLORS)
    candidates, missing_colors = select_preregistered_candidates(args.model, requested_colors)
    tested_colors = tuple(color for color in requested_colors if color in candidates)
    object_count = 2 if args.smoke else len(OBJECTS)
    template_count = 1 if args.smoke else len(TEMPLATES)
    cases = build_cases(tested_colors, object_count, template_count)
    experiment = {
        "schema_version": "experiment_spec.v1",
        "run_id": run_id,
        "created_at": now(),
        "model": args.model,
        "mode": mode,
        "prediction_frozen_before_measurement": True,
        "candidate_source": str(candidate_path(args.model).relative_to(ROOT)).replace("\\", "/"),
        "requested_colors": list(requested_colors),
        "tested_colors": list(tested_colors),
        "missing_preregistered_colors": missing_colors,
        "case_design": {"objects": object_count, "templates": template_count, "cases": len(cases)},
        "interventions": ["single_unit_zero", "single_unit_half", "matched_activation_same_layer_zero"],
        "acceptance_counts": {"negative": "at least 80%", "beyond_matched_control": "at least 70%", "monotonic": "at least 70%"},
        "scientific_boundary": "Tests necessity of one preregistered MLP product unit for local color-candidate margin only.",
    }
    write_json(run_dir / "experiment.json", experiment)
    write_jsonl(run_dir / "cases.jsonl", cases)
    write_json(run_dir / "preregistered_candidates.json", candidates)
    if args.dry_run:
        return experiment

    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(args.model)
        model_config = loaded.model.config.to_dict()
        color_ids = {color: first_token_id(loaded.tokenizer, color) for color in tested_colors}
        rows = []
        for color in tested_colors:
            color_cases = [case for case in cases if case["color"] == color]
            rows.extend(evaluate_group(loaded, color_cases, candidates[color], color_ids, args.batch_size, run_id))
        clean_rows = clean_side_effect(loaded, candidates, run_id)
        summary = summarize(rows, clean_rows)
        allocated, reserved = vram_gb()
        write_jsonl(run_dir / "intervention_rows.jsonl", rows)
        write_jsonl(run_dir / "clean_control_rows.jsonl", clean_rows)
        write_jsonl(run_dir / "unit_evidence.jsonl", rows)
        write_json(run_dir / "model_snapshot.json", {
            "model": args.model,
            "model_class": type(loaded.model).__name__,
            "config": model_config,
            "color_token_ids": color_ids,
        })
        write_json(run_dir / "summary.json", summary)
        manifest = {
            "schema_version": "agi_research_bundle.v2",
            "run_id": run_id,
            "phase": 288,
            "status": "complete",
            "mode": mode,
            "model": args.model,
            "duration_seconds": round(time.monotonic() - started, 3),
            "vram_gb": {"allocated": allocated, "reserved": reserved},
            "summary": summary,
            "artifacts": {},
        }
        for name in ("experiment.json", "cases.jsonl", "preregistered_candidates.json", "intervention_rows.jsonl", "clean_control_rows.jsonl", "unit_evidence.jsonl", "model_snapshot.json", "summary.json"):
            path = run_dir / name
            manifest["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps({"run_id": run_id, "summary": summary, "vram_gb": manifest["vram_gb"]}, ensure_ascii=False, indent=2), flush=True)
        return manifest
    finally:
        release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("qwen3", "glm4", "deepseek7b"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
