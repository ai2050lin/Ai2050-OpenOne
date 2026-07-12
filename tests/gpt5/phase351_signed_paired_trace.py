#!/usr/bin/env python3
"""Collect signed target-versus-competitor component trajectories for Phase351."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase334_natural_contrast_survey import role_positions  # noqa: E402
from phase338_block_causal_screen import continuation_ids, get_layers, prompt_ids  # noqa: E402
from phase347_three_core_natural_trace import depth_lookup, install_capture_hooks  # noqa: E402
from phase351_signed_paired_trace_case_bank import (  # noqa: E402
    OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


MODELS = ("qwen3", "glm4", "deepseek7b")
ROLES = ("source", "query", "answer_start")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def unit_direction(weight: torch.Tensor, token_id: int) -> torch.Tensor:
    vector = weight[token_id].detach().float()
    return vector / vector.norm().clamp_min(1e-8)


@torch.inference_mode()
def run_model(model: str, round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    cases = [row for row in read_jsonl(root / "phase351_registered_cases.jsonl") if row["model"] == model]
    loaded = None
    handles: list[Any] = []
    rows = []
    try:
        loaded = load_probe_model(model)
        depth_by_layer = depth_lookup(len(get_layers(loaded.model)))
        output_weight = loaded.model.get_output_embeddings().weight.detach()
        state: dict[str, Any] = {"captures": [], "positions": None}
        handles = install_capture_hooks(loaded, state)
        for index, case in enumerate(cases, 1):
            ids = prompt_ids(loaded, case)
            role_map = role_positions(loaded, case, ids)
            state["positions"] = torch.tensor([role_map[role][0] for role in ROLES], dtype=torch.long, device=loaded.input_device)
            state["captures"] = []
            state["pre_inputs"] = {}
            target_id = continuation_ids(loaded, case, case["target"])[0]
            distractor_ids = [continuation_ids(loaded, case, value)[0] for value in case["distractors"]]
            target_direction = unit_direction(output_weight, target_id)
            distractor_directions = torch.stack([unit_direction(output_weight, token_id) for token_id in distractor_ids])
            input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.input_device)
            output = loaded.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, return_dict=True)
            for layer_index, component, captured in state["captures"]:
                vectors = captured.float()
                norms = vectors.norm(dim=-1).clamp_min(1e-8)
                target_cosine = torch.mv(vectors, target_direction.to(vectors.device)) / norms
                competitor_cosines = torch.mm(vectors, distractor_directions.to(vectors.device).T) / norms[:, None]
                best_competitor = competitor_cosines.max(dim=-1).values
                margins = target_cosine - best_competitor
                for role_index, role in enumerate(ROLES):
                    values = (norms[role_index], target_cosine[role_index], best_competitor[role_index], margins[role_index])
                    finite = all(torch.isfinite(value).item() for value in values)
                    rows.append({
                        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                        "model": model, "case_id": case["case_id"],
                        "contrast_group_id": case["contrast_group_id"], "family_id": case["family_id"],
                        "mechanism_id": case["mechanism_id"], "item_index": case["item_index"],
                        "split": case["split"], "template_id": case["template_id"],
                        "contrast_condition": case["contrast_condition"], "lexical_set": case["lexical_set"],
                        "operation_demanded": case["operation_demanded"],
                        "generation_step": 0, "component": component, "layer_index": layer_index,
                        "depth_bin": depth_by_layer[layer_index], "position_role": role,
                        "component_l2_norm": round(float(values[0].item()), 7) if finite else None,
                        "signed_target_cosine": round(float(values[1].item()), 7) if finite else None,
                        "signed_best_competitor_cosine": round(float(values[2].item()), 7) if finite else None,
                        "signed_competition_margin": round(float(values[3].item()), 7) if finite else None,
                        "finite": finite, "natural_trace_only": True,
                        "physical_heldout": False, "causal_sealed": False,
                        "single_unit_causal": False,
                    })
            del output, input_ids
            if index % 48 == 0 or index == len(cases):
                print(f"[{model}] {index}/{len(cases)}", flush=True)
        model_root = root / "models" / model
        write_jsonl(model_root / "phase351_signed_trace_rows.jsonl", rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "registered_case_count": len(cases), "trace_row_count": len(rows),
            "nonfinite_trace_row_count": sum(not row["finite"] for row in rows),
            "physical_heldout_trace_count": sum(row["physical_heldout"] for row in rows),
            "causal_sealed_trace_count": sum(row["causal_sealed"] for row in rows),
            "valid": len(cases) == 192 and bool(rows),
        }
        write_json(model_root / "complete.json", complete)
        return complete
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run_model(args.model, args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
