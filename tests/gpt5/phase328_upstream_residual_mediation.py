#!/usr/bin/env python3
"""Phase328 registered upstream-residual to frozen-carrier mediation audit."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase327_natural_retrieval_case_bank as case_bank  # noqa: E402
import phase327_natural_retrieval_path as phase327  # noqa: E402
from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402


PHASE = "Phase328"
SCHEMA_VERSION = "6.1.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
ROUND_DEFAULT = "upstream_residual_mediation"
OUT = ROOT / "tests/gpt5/result/phase328_upstream_residual_mediation"
PHASE326 = ROOT / "tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas"
PHASE327 = ROOT / "tests/gpt5/result/phase327_natural_retrieval_path/natural_retrieval_path"
MECHANISM = "category_retrieval"
ROLE = "query"
CONDITIONS = (
    "recipient_baseline",
    "correct_residual_transplant",
    "same_target_residual_transplant",
    "same_semantic_wrong_residual_transplant",
    "unrelated_residual_transplant",
    "correct_residual_wrong_layer_transplant",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_mean(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def freeze_residual_selection(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row["model"] == model
        and row["mechanism_id"] == MECHANISM
        and row["split"] == "registered_primary"
        and row["position_role"] == ROLE
    ]
    max_layer = max(int(row["layer"]) for row in eligible)
    layer_limit = round(max_layer * 0.70)
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"same": [], "wrong": []})
    for row in eligible:
        layer = int(row["layer"])
        if layer > layer_limit:
            continue
        bucket = "same" if row["comparison_variant"] == "same_target_object" else "wrong"
        grouped[layer][bucket].append(float(row["residual_rms_delta"]))
    candidates = []
    for layer, values in grouped.items():
        same = safe_mean(values["same"])
        wrong = safe_mean(values["wrong"])
        candidates.append({
            "layer": layer,
            "same_target_rms_delta": same,
            "wrong_target_rms_delta": wrong,
            "target_identity_specificity": round(wrong - same, 6),
            "same_target_observation_count": len(values["same"]),
            "wrong_target_observation_count": len(values["wrong"]),
        })
    selected = max(candidates, key=lambda row: (row["target_identity_specificity"], row["layer"]))
    if selected["layer"] <= 0:
        raise ValueError("Residual input transplantation requires a selected layer above layer zero")
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "family_id": "content_knowledge",
        "mechanism_id": MECHANISM,
        "position_role": ROLE,
        **selected,
        "selection_split": "phase327_registered_primary_only",
        "validation_split": "phase327_registered_confirmation_only",
        "layer_search_limit": layer_limit,
        "selection_updates_allowed": False,
        "selection_boundary": (
            "Maximum within the first 70% of layers; a boundary maximum is not interpreted as a local mechanism peak."
        ),
    }


def carrier_similarity(reference: dict[str, torch.Tensor], current: dict[str, torch.Tensor]) -> float:
    keys = sorted(set(reference) & set(current))
    return safe_mean([phase327.cosine(reference[key], current[key]) for key in keys])


def global_metrics(loaded: Any, result: dict[str, Any], target: str) -> dict[str, Any]:
    logits = result["first_logits"]
    target_id = phase327.target_token_ids(loaded.tokenizer, target)[0]
    target_logit = float(logits[target_id].item())
    rank = int((logits > target_logit).sum().item()) + 1
    top = torch.topk(logits, k=5).indices.tolist()
    return {
        "global_target_rank": rank,
        "global_target_top1": int(top[0]) == target_id,
        "global_top5_token_ids": [int(value) for value in top],
        "global_top5_tokens": [loaded.tokenizer.decode([int(value)]) for value in top],
    }


def donor_state(
    loaded: Any,
    case: dict[str, Any],
    variant: str,
    layer: int,
    carrier_specs: list[dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probe_case = phase327.variant_case(case, variant)
    result = phase327.forward_probe(
        loaded,
        probe_case,
        case["target"],
        case["distractors"],
        capture_specs=carrier_specs or [],
        capture_hidden=True,
    )
    key = f"L{layer - 1}:{ROLE}"
    if key not in result["hidden_vectors"]:
        raise KeyError(f"Missing donor residual state {key}")
    return result["hidden_vectors"][key], result["captures"]


def validation_cases() -> list[dict[str, Any]]:
    return [
        case for case in case_bank.build_cases()
        if case["mechanism_id"] == MECHANISM and case["split"] == "registered_confirmation"
    ]


def run_model(model_key: str, round_name: str) -> dict[str, Any]:
    output = OUT / round_name
    phase327_rows = read_jsonl(PHASE327 / f"phase327_{model_key}_residual_path_rows.jsonl")
    selection = freeze_residual_selection(model_key, phase327_rows)
    carrier_selections = read_jsonl(PHASE326 / f"phase326_{model_key}_carrier_sets.jsonl")
    cases = validation_cases()
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        n_layers = len(get_layers(loaded.model))
        selected_layer = int(selection["layer"])
        wrong_layer = min(n_layers - 1, selected_layer + max(1, n_layers // 5))
        rows: list[dict[str, Any]] = []
        for number, case in enumerate(cases, start=1):
            carrier_specs = phase327.selections_for_case(carrier_selections, case)
            recipient = phase327.variant_case(case, "same_semantic_wrong_target")
            correct_state, correct_carriers = donor_state(
                loaded, case, "correct_object", selected_layer, carrier_specs
            )
            same_target_state, _ = donor_state(loaded, case, "same_target_object", selected_layer)
            wrong_state, _ = donor_state(loaded, case, "same_semantic_wrong_target", selected_layer)
            unrelated_state, _ = donor_state(loaded, case, "unrelated_wrong_target", selected_layer)
            correct_wrong_layer_state, _ = donor_state(
                loaded, case, "correct_object", wrong_layer
            )
            settings = {
                "recipient_baseline": None,
                "correct_residual_transplant": {
                    "layer": selected_layer, "position_role": ROLE, "value": correct_state,
                },
                "same_target_residual_transplant": {
                    "layer": selected_layer, "position_role": ROLE, "value": same_target_state,
                },
                "same_semantic_wrong_residual_transplant": {
                    "layer": selected_layer, "position_role": ROLE, "value": wrong_state,
                },
                "unrelated_residual_transplant": {
                    "layer": selected_layer, "position_role": ROLE, "value": unrelated_state,
                },
                "correct_residual_wrong_layer_transplant": {
                    "layer": wrong_layer, "position_role": ROLE, "value": correct_wrong_layer_state,
                },
            }
            baseline = phase327.forward_probe(
                loaded, recipient, case["target"], case["distractors"], capture_specs=carrier_specs
            )
            baseline_global = global_metrics(loaded, baseline, case["target"])
            baseline_similarity = carrier_similarity(correct_carriers, baseline["captures"])
            for condition in CONDITIONS:
                result = baseline if condition == "recipient_baseline" else phase327.forward_probe(
                    loaded,
                    recipient,
                    case["target"],
                    case["distractors"],
                    capture_specs=carrier_specs,
                    residual_patch=settings[condition],
                )
                global_result = baseline_global if condition == "recipient_baseline" else global_metrics(
                    loaded, result, case["target"]
                )
                similarity = carrier_similarity(correct_carriers, result["captures"])
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "model": model_key,
                    "case_id": case["case_id"],
                    "base_case_id": case["base_case_id"],
                    "family_id": case["family_id"],
                    "mechanism_id": case["mechanism_id"],
                    "split": case["split"],
                    "template_id": case["template_id"],
                    "condition": condition,
                    "selected_residual_layer": selected_layer,
                    "intervention_layer": None if settings[condition] is None else settings[condition]["layer"],
                    "position_role": ROLE,
                    "recipient_subject": recipient["subject"],
                    "recipient_natural_target": recipient["natural_target"],
                    "evaluation_target": case["target"],
                    "baseline_phrase_logprob_sum": baseline["phrase_logprob_sum"],
                    "phrase_logprob_sum": result["phrase_logprob_sum"],
                    "phrase_logprob_gain": round(
                        result["phrase_logprob_sum"] - baseline["phrase_logprob_sum"], 6
                    ),
                    "baseline_target_margin": baseline["metrics"]["target_margin"],
                    "target_margin": result["metrics"]["target_margin"],
                    "target_margin_gain": round(
                        result["metrics"]["target_margin"] - baseline["metrics"]["target_margin"], 6
                    ),
                    "baseline_global_target_rank": baseline_global["global_target_rank"],
                    **global_result,
                    "global_target_rank_gain": baseline_global["global_target_rank"] - global_result["global_target_rank"],
                    "correct_donor_carrier_similarity": similarity,
                    "baseline_correct_donor_carrier_similarity": baseline_similarity,
                    "carrier_similarity_gain": round(similarity - baseline_similarity, 6),
                    "js_divergence_from_recipient": 0.0 if condition == "recipient_baseline" else phase327.phase326.js_divergence(
                        baseline["first_logits"], result["first_logits"]
                    ),
                    "causal_scope": "residual_state_to_frozen_carrier_set" if condition != "recipient_baseline" else "none",
                    "single_unit_causal": False,
                })
            if number % 4 == 0 or number == len(cases):
                print(f"[{model_key}] mediation {number}/{len(cases)}", flush=True)
        audit = audit_model(model_key, selection, rows)
        phase326_summary = read_json(PHASE326 / f"phase326_{model_key}_summary.json")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model_key,
            "model_name_or_path": phase326_summary["model_name_or_path"],
            "registered_validation_prompt_count": len(cases),
            "registered_validation_object_count": len({case["base_case_id"] for case in cases}),
            "intervention_row_count": len(rows),
            "selection": selection,
            "audit": audit,
            "single_unit_causal_count": 0,
        }
        write_json(output / f"phase328_{model_key}_residual_selection.json", selection)
        write_jsonl(output / f"phase328_{model_key}_mediation_rows.jsonl", rows)
        write_json(output / f"phase328_{model_key}_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def audit_model(model: str, selection: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in CONDITIONS
    }

    def values(condition: str, field: str) -> list[float]:
        return [float(row[field]) for row in by_condition[condition]]

    correct_phrase = values("correct_residual_transplant", "phrase_logprob_gain")
    same_phrase = values("same_target_residual_transplant", "phrase_logprob_gain")
    wrong_phrase = values("same_semantic_wrong_residual_transplant", "phrase_logprob_gain")
    unrelated_phrase = values("unrelated_residual_transplant", "phrase_logprob_gain")
    wrong_layer_phrase = values("correct_residual_wrong_layer_transplant", "phrase_logprob_gain")
    correct_carrier = values("correct_residual_transplant", "carrier_similarity_gain")
    same_carrier = values("same_target_residual_transplant", "carrier_similarity_gain")
    wrong_carrier = values("same_semantic_wrong_residual_transplant", "carrier_similarity_gain")
    unrelated_carrier = values("unrelated_residual_transplant", "carrier_similarity_gain")
    wrong_layer_carrier = values("correct_residual_wrong_layer_transplant", "carrier_similarity_gain")
    positive_phrase = min(safe_mean(correct_phrase), safe_mean(same_phrase))
    positive_carrier = min(safe_mean(correct_carrier), safe_mean(same_carrier))
    phrase_specificity = round(
        positive_phrase - max(safe_mean(wrong_phrase), safe_mean(unrelated_phrase)), 6
    )
    carrier_specificity = round(
        positive_carrier - max(safe_mean(wrong_carrier), safe_mean(unrelated_carrier)), 6
    )
    wrong_layer_phrase_specificity = round(safe_mean(correct_phrase) - safe_mean(wrong_layer_phrase), 6)
    wrong_layer_carrier_specificity = round(safe_mean(correct_carrier) - safe_mean(wrong_layer_carrier), 6)
    phrase_consistency = round(sum(value > 0 for value in correct_phrase) / len(correct_phrase), 6)
    carrier_consistency = round(sum(value > 0 for value in correct_carrier) / len(correct_carrier), 6)
    correct_rank_gain = safe_mean(values("correct_residual_transplant", "global_target_rank_gain"))
    baseline_top1 = safe_mean([
        float(row["global_target_top1"]) for row in by_condition["recipient_baseline"]
    ])
    correct_top1 = safe_mean([
        float(row["global_target_top1"]) for row in by_condition["correct_residual_transplant"]
    ])
    control_top1 = max(
        safe_mean([float(row["global_target_top1"]) for row in by_condition[condition]])
        for condition in (
            "same_semantic_wrong_residual_transplant",
            "unrelated_residual_transplant",
            "correct_residual_wrong_layer_transplant",
        )
    )
    mediation_pass = (
        len(correct_phrase) == 12
        and positive_phrase > 0.0 and phrase_specificity > 0.0
        and positive_carrier > 0.0 and carrier_specificity > 0.0
        and wrong_layer_phrase_specificity > 0.0 and wrong_layer_carrier_specificity > 0.0
        and phrase_consistency >= 0.65 and carrier_consistency >= 0.65
        and correct_rank_gain > 0.0
    )
    generation_unlock = correct_top1 > baseline_top1 and correct_top1 > control_top1
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "family_id": "content_knowledge",
        "mechanism_id": MECHANISM,
        "selected_residual_layer": selection["layer"],
        "position_role": ROLE,
        "registered_validation_prompt_count": len(correct_phrase),
        "correct_phrase_gain": safe_mean(correct_phrase),
        "same_target_phrase_gain": safe_mean(same_phrase),
        "wrong_donor_phrase_gain": safe_mean(wrong_phrase),
        "unrelated_donor_phrase_gain": safe_mean(unrelated_phrase),
        "wrong_layer_phrase_gain": safe_mean(wrong_layer_phrase),
        "phrase_donor_specificity": phrase_specificity,
        "phrase_wrong_layer_specificity": wrong_layer_phrase_specificity,
        "correct_carrier_similarity_gain": safe_mean(correct_carrier),
        "same_target_carrier_similarity_gain": safe_mean(same_carrier),
        "wrong_donor_carrier_similarity_gain": safe_mean(wrong_carrier),
        "unrelated_donor_carrier_similarity_gain": safe_mean(unrelated_carrier),
        "wrong_layer_carrier_similarity_gain": safe_mean(wrong_layer_carrier),
        "carrier_donor_specificity": carrier_specificity,
        "carrier_wrong_layer_specificity": wrong_layer_carrier_specificity,
        "phrase_positive_consistency": phrase_consistency,
        "carrier_positive_consistency": carrier_consistency,
        "global_target_rank_gain": correct_rank_gain,
        "recipient_baseline_top1_rate": baseline_top1,
        "correct_residual_top1_rate": correct_top1,
        "best_control_top1_rate": control_top1,
        "upstream_mediation_pass": mediation_pass,
        "natural_generation_unlock_pass": generation_unlock,
        "model_causal_edge_candidate": mediation_pass and generation_unlock,
        "single_unit_causal": False,
    }


def collect(round_name: str) -> dict[str, Any]:
    output = OUT / round_name
    summaries = [read_json(output / f"phase328_{model}_summary.json") for model in MODELS]
    audits = [row["audit"] for row in summaries]
    rows = [
        row for model in MODELS
        for row in read_jsonl(output / f"phase328_{model}_mediation_rows.jsonl")
    ]
    selections = [row["selection"] for row in summaries]
    mediation_models = [row["model"] for row in audits if row["upstream_mediation_pass"]]
    unlock_models = [row["model"] for row in audits if row["natural_generation_unlock_pass"]]
    causal_models = [row["model"] for row in audits if row["model_causal_edge_candidate"]]
    cross_causal = len(causal_models) >= 2
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "family_id": "content_knowledge",
        "mechanism_id": MECHANISM,
        "selection_prompt_count_per_model": 24,
        "selection_independent_object_count_per_model": 12,
        "validation_prompt_count_per_model": 12,
        "validation_independent_object_count_per_model": 6,
        "intervention_row_count": len(rows),
        "upstream_mediation_pass_models": mediation_models,
        "natural_generation_unlock_pass_models": unlock_models,
        "model_causal_edge_candidate_models": causal_models,
        "cross_model_causal_edge_replicated": cross_causal,
        "l5_promoted": False,
        "single_unit_causal_count": 0,
        "audits": audits,
    }
    edges = []
    for audit in audits:
        edges.append({
            "schema_version": "pattern_family_physical_edge.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "edge_id": f"phase328:{audit['model']}:{MECHANISM}:query_residual_to_frozen_carrier",
            "family_id": "content_knowledge",
            "mechanism_id": MECHANISM,
            "model": audit["model"],
            "source_stage": f"L{audit['selected_residual_layer']}_query_residual_state",
            "target_stage": "phase326_frozen_carrier_set",
            "relation": "registered_residual_state_mediation",
            "causal": bool(cross_causal and audit["model_causal_edge_candidate"]),
            "causal_scope": "pooled_residual_state_to_distributed_component_set",
            "single_unit_causal": False,
            "evidence_boundary": (
                "Cross-model residual-state mediation and natural top-1 unlock replicated."
                if cross_causal else
                "No cross-model replication of both downstream mediation and natural top-1 unlock."
            ),
        })
    write_jsonl(output / "phase328_residual_selections.jsonl", selections)
    write_jsonl(output / "phase328_mediation_rows.jsonl", rows)
    write_jsonl(output / "phase328_model_audits.jsonl", audits)
    write_json(output / "phase328_cross_model_summary.json", result)
    write_jsonl(output / "phase328_atlas_edges.jsonl", edges)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect(args.round)
        return
    if not args.model:
        parser.error("--model is required unless --collect is used")
    run_model(args.model, args.round)


if __name__ == "__main__":
    main()
