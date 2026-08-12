#!/usr/bin/env python3
"""Natural four-state factorial binding-identity matrix.

The operator was positively calibrated in Phases1143-1144.  This phase moves
it, unchanged, to untouched temporal-binding items.  It first tests the donor
matrix for the panel-opposed binding term.  Only a two-model identity result
authorizes the matched common-time donor matrix, and binding identity must then
exceed common-route identity before confirmation is allowed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers  # noqa: E402
from phase1023_fp16_utils import quantization_audit, release_fp16  # noqa: E402
import phase1135_temporal_binding_intervention as source  # noqa: E402
import phase1138_temporal_residual_onset as phase1138  # noqa: E402
import phase1140_temporal_position_scope_sufficiency as phase1140  # noqa: E402
import phase1141_first_lexical_divergence_boundary as phase1141  # noqa: E402
import phase1142_causal_donor_response_matrix as phase1142  # noqa: E402
import phase1143_ground_truth_mechanism_calibration as phase1143  # noqa: E402
import phase1144_symmetric_factorial_operator_calibration as phase1144  # noqa: E402


PHASE = 1145
SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1145_natural_factorial_binding_identity"
MODELS = phase1142.MODELS
SPLITS = phase1142.SPLITS
ROUTES = ("binding", "common_control")
PANELS = {
    "original": ("original_pre", "old", "new", 1.0),
    "swapped": ("swapped_pre", "new", "old", -1.0),
}
PROPERTIES = phase1142.PROPERTIES
COHORT_SIZE = 12
REQUESTED_FRACTION = 0.7
ALPHAS = (0.0, 1.0)
EXPECTED_COMPARISONS = COHORT_SIZE * (COHORT_SIZE - 1) * len(PANELS)
EXPECTED_RECORDS = EXPECTED_COMPARISONS * 4
EPSILON = 1e-8

THRESHOLDS = {
    "finite_fraction": 0.99,
    "paired_alpha0_max_abs_margin_difference": 0.005,
    "diagonal_baseline_valid_fraction": 0.99,
    "diagonal_endpoint_flip_fraction": 0.95,
    "diagonal_positive_change_fraction": 0.99,
    "item_advantage_median": 0.20,
    "item_advantage_positive_fraction": 0.75,
    "item_advantage_same_relation_median": 0.15,
    "item_advantage_cross_relation_median": 0.15,
    "diagonal_minus_offdiagonal_flip_fraction": 0.20,
    "diagonal_top1_fraction": 0.60,
    "per_property_item_advantage_median": 0.10,
    "binding_minus_common_advantage_median": 0.10,
    "binding_minus_common_positive_fraction": 0.75,
    "binding_minus_common_same_relation_median": 0.05,
    "binding_minus_common_cross_relation_median": 0.05,
    "per_property_binding_minus_common_median": 0.05,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else None


def fraction(rows: Iterable[dict[str, Any]], key: str) -> float:
    rows = list(rows)
    return sum(bool(row[key]) for row in rows) / max(len(rows), 1)


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists():
        raise RuntimeError("refusing to rewrite Phase1145 after model output exists")
    p1142 = read_json(phase1142.OUT_ROOT / "protocol/preregistration.json")
    f1142 = read_json(phase1142.OUT_ROOT / "analysis/final.json")
    f1143 = read_json(phase1143.OUT_ROOT / "analysis/final.json")
    a1143 = read_json(phase1143.OUT_ROOT / "audit/independent_result_audit.json")
    f1144 = read_json(phase1144.OUT_ROOT / "analysis/final.json")
    a1144 = read_json(phase1144.OUT_ROOT / "audit/independent_result_audit.json")
    cohorts = {
        "discovery": list(p1142["material"]["cohorts"]["confirmation"]),
        "confirmation": list(p1142["material"]["reserve_item_ids"]),
    }
    token_meta = phase1141.token_metadata()
    items = {str(row["item_id"]): row for row in read_jsonl(source.SOURCE)}
    property_counts = {
        split: {
            prop: sum(str(token_meta[item_id]["property_id"]) == prop for item_id in cohort)
            for prop in PROPERTIES
        }
        for split, cohort in cohorts.items()
    }
    checks = {
        "phase1142_stopped_before_confirmation": bool(
            f1142["confirmation_run"] is False and f1142["auto_continue"] is False
        ),
        "phase1143_calibrated_and_audited": bool(f1143["calibration_passed"] and a1143["all_checks_passed"]),
        "phase1144_calibrated_and_audited": bool(f1144["calibration_passed"] and a1144["all_checks_passed"]),
        "twelve_items_each_split": all(len(cohort) == COHORT_SIZE for cohort in cohorts.values()),
        "cohorts_disjoint": set(cohorts["discovery"]).isdisjoint(cohorts["confirmation"]),
        "all_items_exist": all(item_id in items for cohort in cohorts.values() for item_id in cohort),
        "four_per_property": all(property_counts[split][prop] == 4 for split in SPLITS for prop in PROPERTIES),
        "all_lcp_zero": all(int(token_meta[item_id]["common_prefix_length"]) == 0 for cohort in cohorts.values() for item_id in cohort),
        "all_first_tokens_informative": all(bool(token_meta[item_id]["first_token_informative"]) for cohort in cohorts.values() for item_id in cohort),
        "no_prefix_containment": all(not bool(token_meta[item_id]["prefix_containment"]) for cohort in cohorts.values() for item_id in cohort),
        "two_models_only_by_frozen_hidden_gate": MODELS == ("qwen3_4b", "qwen3_14b"),
        "glm4_denied_by_phase1135": True,
        "deepseek7b_denied_by_phase1135": True,
        "binding_route_first": ROUTES[0] == "binding",
        "common_route_gated": True,
        "confirmation_locked": True,
        "component_search_forbidden": True,
        "same_depth_and_position_as_phase1142": True,
        "fp16_only": True,
        "machine_consensus_scope_retained": True,
    }
    prereg = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "natural symmetric-factorial binding identity matrix",
        "objective": "Test whether the panel-opposed binding term has donor identity beyond a matched common-time route.",
        "epistemic_scope": "Qwen3-4B/14B FP16 whole-residual sufficiency at relative depth 0.7 on untouched machine-consensus temporal templates; no component, necessity, natural-generation, semantic-vector, or cross-architecture claim.",
        "source": {
            "items_sha256": sha256_file(source.SOURCE),
            "phase1142_protocol_digest": p1142["protocol_digest"],
            "phase1142_final_digest": f1142["final_digest"],
            "phase1143_final_digest": f1143["final_digest"],
            "phase1143_audit_digest": a1143["audit_digest"],
            "phase1144_final_digest": f1144["final_digest"],
            "phase1144_audit_digest": a1144["audit_digest"],
            "script_sha256": sha256_file(SCRIPT),
        },
        "models": p1142["models"],
        "cross_architecture_status": p1142["cross_architecture_status"],
        "material": {"cohorts": cohorts, "property_counts": property_counts, "selection": "untouched Phase1142 confirmation becomes discovery; untouched reserve becomes confirmation"},
        "intervention": {
            "depth_fraction": REQUESTED_FRACTION,
            "position": "first lexical divergence; all LCP=0",
            "component": "whole residual",
            "alphas": list(ALPHAS),
            "routes": {
                "binding": "target_pre + target_common + panel_sign * donor_binding",
                "common_control": "target_pre + donor_common + panel_sign * target_binding",
            },
            "common": "0.5*((Opost-Opre)+(Spost-Spre))",
            "binding": "0.5*((Opost-Opre)-(Spost-Spre))",
            "matrix_comparisons_per_route": EXPECTED_COMPARISONS,
            "records_per_model_split_route": EXPECTED_RECORDS,
        },
        "thresholds": THRESHOLDS,
        "decision_rule": {
            "stage1": "run full binding matrix in both models; both must pass item identity",
            "stage2": "only then run common-control matrix in both models; binding-minus-common specificity must pass in both",
            "confirmation": "repeat both frozen routes on untouched confirmation items",
        },
        "hard_stops": [
            "do not run common-control route unless both discovery binding routes pass",
            "do not run confirmation unless both discovery specificity routes pass",
            "do not alter depth, position, items, routes, models, precision, or thresholds",
            "do not search attention, MLP, head, neuron, SAE, Jacobian, or necessity",
            "do not upgrade machine consensus to human gold",
            "do not call an exact diagonal reconstruction payload evidence without off-diagonal and common-route specificity",
        ],
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1145 protocol checks failed: {checks}")
    body = dict(prereg)
    prereg["protocol_digest"] = digest(body)
    write_json(OUT_ROOT / "protocol/preregistration.json", prereg)
    write_json(OUT_ROOT / "protocol/audit.json", {"checks": checks, "check_count": len(checks), "passed_count": sum(checks.values()), "all_checks_passed": all(checks.values()), "protocol_digest": prereg["protocol_digest"]})
    print(canonical({"protocol_digest": prereg["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(prereg)
    stored = body.pop("protocol_digest")
    if digest(body) != stored or sha256_file(SCRIPT) != prereg["source"]["script_sha256"]:
        raise RuntimeError("Phase1145 frozen protocol mismatch")
    if not read_json(OUT_ROOT / "protocol/audit.json")["all_checks_passed"]:
        raise RuntimeError("Phase1145 protocol audit failed")
    return prereg


def entry(route: str, comparison_id: str, arm: str, item_id: str, source_item_id: str, property_id: str, source_property_id: str, panel: str, alpha: float) -> dict[str, Any]:
    target_state, base_key, desired_key, _ = PANELS[panel]
    curve_id = f"{comparison_id}|route={route}|arm={arm}"
    return {
        "phase": PHASE,
        "route": route,
        "comparison_id": comparison_id,
        "arm": arm,
        "curve_id": curve_id,
        "record_id": f"{curve_id}|alpha={alpha:.2f}",
        "item_id": item_id,
        "source_item_id": source_item_id,
        "property_id": property_id,
        "source_property_id": source_property_id,
        "same_relation": property_id == source_property_id,
        "panel": panel,
        "target_state": target_state,
        "target_case_id": source.state_id(item_id, target_state),
        "base_key": base_key,
        "desired_key": desired_key,
        "alpha": alpha,
    }


def build_blocks(route: str, item_ids: list[str], token_meta: dict[str, dict[str, Any]]) -> list[list[dict[str, Any]]]:
    blocks = []
    ordered = phase1142.stable_order(f"phase1145|{route}|targets", item_ids)
    for item_id in ordered:
        prop = str(token_meta[item_id]["property_id"])
        donors = phase1142.stable_order(f"phase1145|{route}|{item_id}", [value for value in ordered if value != item_id])
        for panel in PANELS:
            for donor in donors:
                donor_prop = str(token_meta[donor]["property_id"])
                comparison_id = f"matrix|{item_id}|{panel}|donor={donor}"
                block = []
                for arm, source_item_id, source_prop in (("paired_correct_donor", item_id, prop), ("challenger_donor", donor, donor_prop)):
                    for alpha in ALPHAS:
                        block.append(entry(route, comparison_id, arm, item_id, source_item_id, prop, source_prop, panel, alpha))
                blocks.append(block)
    if len(blocks) != EXPECTED_COMPARISONS or any(len(block) != 4 for block in blocks):
        raise RuntimeError("Phase1145 block count drift")
    return blocks


def vector_at(vectors: dict[str, torch.Tensor], item_id: str, state: str, candidate_key: str) -> torch.Tensor:
    key = phase1140.vector_key(source.state_id(item_id, state), candidate_key)
    value = vectors[key]
    if value.shape[0] < 1:
        raise RuntimeError("empty candidate path vector")
    return value[:1]


def composed_source(vectors: dict[str, torch.Tensor], entry_row: dict[str, Any], candidate_key: str) -> torch.Tensor:
    target = str(entry_row["item_id"])
    donor = str(entry_row["source_item_id"])
    panel = str(entry_row["panel"])
    panel_sign = float(PANELS[panel][3])

    def terms(item_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        opre = vector_at(vectors, item_id, "original_pre", candidate_key)
        opost = vector_at(vectors, item_id, "original_post", candidate_key)
        spre = vector_at(vectors, item_id, "swapped_pre", candidate_key)
        spost = vector_at(vectors, item_id, "swapped_post", candidate_key)
        common = 0.5 * ((opost - opre) + (spost - spre))
        binding = 0.5 * ((opost - opre) - (spost - spre))
        return common, binding

    target_common, target_binding = terms(target)
    donor_common, donor_binding = terms(donor)
    target_pre = vector_at(vectors, target, str(entry_row["target_state"]), candidate_key)
    if str(entry_row["route"]) == "binding":
        return target_pre + target_common + panel_sign * donor_binding
    if str(entry_row["route"]) == "common_control":
        return target_pre + donor_common + panel_sign * target_binding
    raise RuntimeError("unknown route")


def score_batch(model, layer, entries: list[dict[str, Any]], cases: dict[str, dict[str, Any]], vectors: dict[str, torch.Tensor], tokenizer, pad_id: int, device: torch.device) -> list[dict[str, Any]]:
    expanded = []
    owners = []
    sources = []
    positions = []
    alpha_rows = []
    for row in entries:
        case = cases[str(row["target_case_id"])]
        for candidate_key in ("old", "new"):
            token_row = source.tokenize_case(tokenizer, case, candidate_key)
            if int(token_row["prompt_length"]) < 1:
                raise RuntimeError("empty prompt")
            token_row["decision_offset"] = 0
            expanded.append(token_row)
            owners.append(row)
            sources.append(composed_source(vectors, row, candidate_key))
            positions.append(torch.tensor([int(token_row["prompt_length"]) - 1], dtype=torch.long, device=device))
            alpha_rows.append(float(row["alpha"]))
    input_ids, attention_mask = source.pad_sequences(expanded, pad_id, device)
    alphas = torch.tensor(alpha_rows, dtype=torch.float32, device=device)
    with torch.inference_mode():
        with phase1140.LivePathInterpolation(layer, positions, sources, alphas) as patch:
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        if patch.calls != 1:
            raise RuntimeError("interpolation hook call drift")
        scores = phase1141.score_token_paths(output.logits, expanded)
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for token_row, owner, score in zip(expanded, owners, scores):
        grouped[str(owner["record_id"])][str(token_row["candidate_key"])] = score
    result = []
    for row in entries:
        pair = grouped[str(row["record_id"])]
        base, desired = str(row["base_key"]), str(row["desired_key"])
        finite = bool(pair[base]["finite"] and pair[desired]["finite"])
        margin = pair[desired]["full_logp_mean"] - pair[base]["full_logp_mean"] if finite else None
        record = dict(row)
        record.update({"finite": finite, "full_oriented_margin": margin, "scores": pair if finite else None})
        result.append(record)
    del output, input_ids, attention_mask, alphas
    return result


def curves_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["curve_id"])].append(row)
    curves = []
    for curve_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["alpha"]))
        if [float(row["alpha"]) for row in ordered] != list(ALPHAS):
            raise RuntimeError("alpha drift")
        base, endpoint = ordered
        finite = bool(base["finite"] and endpoint["finite"])
        before, after = base["full_oriented_margin"], endpoint["full_oriented_margin"]
        change = float(after) - float(before) if finite and before is not None and after is not None else None
        curves.append({"curve_id": curve_id, "comparison_id": base["comparison_id"], "route": base["route"], "arm": base["arm"], "model": base["model"], "split": base["split"], "item_id": base["item_id"], "source_item_id": base["source_item_id"], "property_id": base["property_id"], "source_property_id": base["source_property_id"], "same_relation": bool(base["same_relation"]), "panel": base["panel"], "finite": finite, "baseline_margin": before, "endpoint_margin": after, "margin_change": change, "baseline_valid": bool(finite and before is not None and float(before) < 0.0), "endpoint_flip": bool(finite and after is not None and float(after) > 0.0), "positive_change": bool(change is not None and change > 0.0)})
    return curves


def comparisons_from_curves(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in curves:
        grouped[str(row["comparison_id"])][str(row["arm"])] = row
    comparisons = []
    for comparison_id, arms in sorted(grouped.items()):
        if set(arms) != {"paired_correct_donor", "challenger_donor"}:
            raise RuntimeError("arm drift")
        reference, challenger = arms["paired_correct_donor"], arms["challenger_donor"]
        finite = bool(reference["finite"] and challenger["finite"])
        advantage = float(reference["margin_change"]) - float(challenger["margin_change"]) if finite else None
        alpha0 = abs(float(reference["baseline_margin"]) - float(challenger["baseline_margin"])) if finite else None
        comparisons.append({"comparison_id": comparison_id, "route": reference["route"], "model": reference["model"], "split": reference["split"], "item_id": reference["item_id"], "source_item_id": challenger["source_item_id"], "property_id": reference["property_id"], "source_property_id": challenger["source_property_id"], "same_relation": bool(challenger["same_relation"]), "panel": reference["panel"], "finite": finite, "paired_alpha0_abs_margin_difference": alpha0, "diagonal_change": reference["margin_change"], "challenger_change": challenger["margin_change"], "diagonal_endpoint_flip": bool(reference["endpoint_flip"]), "challenger_endpoint_flip": bool(challenger["endpoint_flip"]), "diagonal_positive_change": bool(reference["positive_change"]), "challenger_positive_change": bool(challenger["positive_change"]), "diagonal_baseline_valid": bool(reference["baseline_valid"]), "diagonal_advantage": advantage, "diagonal_advantage_positive": bool(advantage is not None and advantage > 0.0)})
    return comparisons


def target_panel_metrics(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[(str(row["item_id"]), str(row["panel"]))].append(row)
    result = []
    for (item_id, panel), rows in sorted(grouped.items()):
        if len(rows) != COHORT_SIZE - 1:
            raise RuntimeError("target donor count drift")
        diagonal = median(row["diagonal_change"] for row in rows)
        donors = [float(row["challenger_change"]) for row in rows]
        rank = 1 + sum(value >= float(diagonal) for value in donors) if diagonal is not None else None
        result.append({"item_id": item_id, "panel": panel, "property_id": rows[0]["property_id"], "diagonal_rank": rank, "diagonal_top1": rank == 1, "diagonal_top3": rank is not None and rank <= 3})
    return result


def analyze_route(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    curves = curves_from_records(records)
    comparisons = comparisons_from_curves(curves)
    if len(comparisons) != EXPECTED_COMPARISONS:
        raise RuntimeError("comparison count drift")
    same = [row for row in comparisons if row["same_relation"]]
    cross = [row for row in comparisons if not row["same_relation"]]
    targets = target_panel_metrics(comparisons)
    per_property = {prop: median(row["diagonal_advantage"] for row in comparisons if row["property_id"] == prop) for prop in PROPERTIES}
    metrics = {
        "finite_fraction": fraction(comparisons, "finite"),
        "paired_alpha0_max_abs_margin_difference": max(float(row["paired_alpha0_abs_margin_difference"]) for row in comparisons),
        "diagonal_baseline_valid_fraction": fraction(comparisons, "diagonal_baseline_valid"),
        "diagonal_endpoint_flip_fraction": fraction(comparisons, "diagonal_endpoint_flip"),
        "diagonal_positive_change_fraction": fraction(comparisons, "diagonal_positive_change"),
        "item_advantage_median": median(row["diagonal_advantage"] for row in comparisons),
        "item_advantage_positive_fraction": fraction(comparisons, "diagonal_advantage_positive"),
        "item_advantage_same_relation_median": median(row["diagonal_advantage"] for row in same),
        "item_advantage_cross_relation_median": median(row["diagonal_advantage"] for row in cross),
        "diagonal_minus_offdiagonal_flip_fraction": fraction(comparisons, "diagonal_endpoint_flip") - fraction(comparisons, "challenger_endpoint_flip"),
        "diagonal_top1_fraction": fraction(targets, "diagonal_top1"),
        "diagonal_top3_fraction": fraction(targets, "diagonal_top3"),
        "diagonal_median_rank": median(row["diagonal_rank"] for row in targets),
        "per_property_item_advantage_median": per_property,
    }
    checks = {
        "finite": metrics["finite_fraction"] >= THRESHOLDS["finite_fraction"],
        "alpha0": metrics["paired_alpha0_max_abs_margin_difference"] <= THRESHOLDS["paired_alpha0_max_abs_margin_difference"],
        "baseline": metrics["diagonal_baseline_valid_fraction"] >= THRESHOLDS["diagonal_baseline_valid_fraction"],
        "diagonal_flip": metrics["diagonal_endpoint_flip_fraction"] >= THRESHOLDS["diagonal_endpoint_flip_fraction"],
        "diagonal_positive": metrics["diagonal_positive_change_fraction"] >= THRESHOLDS["diagonal_positive_change_fraction"],
        "advantage": metrics["item_advantage_median"] is not None and metrics["item_advantage_median"] >= THRESHOLDS["item_advantage_median"],
        "positive_fraction": metrics["item_advantage_positive_fraction"] >= THRESHOLDS["item_advantage_positive_fraction"],
        "same_relation": metrics["item_advantage_same_relation_median"] is not None and metrics["item_advantage_same_relation_median"] >= THRESHOLDS["item_advantage_same_relation_median"],
        "cross_relation": metrics["item_advantage_cross_relation_median"] is not None and metrics["item_advantage_cross_relation_median"] >= THRESHOLDS["item_advantage_cross_relation_median"],
        "flip_advantage": metrics["diagonal_minus_offdiagonal_flip_fraction"] >= THRESHOLDS["diagonal_minus_offdiagonal_flip_fraction"],
        "top1": metrics["diagonal_top1_fraction"] >= THRESHOLDS["diagonal_top1_fraction"],
        "each_property": all(per_property[prop] is not None and float(per_property[prop]) >= THRESHOLDS["per_property_item_advantage_median"] for prop in PROPERTIES),
    }
    metrics["route_gate_checks"] = checks
    metrics["route_qualified"] = all(checks.values())
    metrics["curve_digest"] = digest(curves)
    metrics["comparison_digest"] = digest(comparisons)
    return metrics, curves, comparisons


def run_command(model_name: str, split: str, route: str) -> None:
    prereg = verify_protocol()
    if model_name not in MODELS or split not in SPLITS or route not in ROUTES:
        raise RuntimeError("invalid Phase1145 endpoint")
    if route == "common_control":
        gate = read_json(OUT_ROOT / f"analysis/{split}_binding_selection.json")
        if not gate["common_control_authorized"]:
            raise RuntimeError("common-control route denied")
    if split == "confirmation":
        discovery = read_json(OUT_ROOT / "analysis/discovery_specificity_selection.json")
        if not discovery["confirmation_authorized"]:
            raise RuntimeError("confirmation denied")
    output_root = OUT_ROOT / "runs" / split / route / model_name
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite {output_root}")
    selected_ids = list(prereg["material"]["cohorts"][split])
    items = read_jsonl(source.SOURCE)
    selected_items, cases = source.causal_cases(items, split, selected_ids)
    token_meta_all = phase1141.token_metadata()
    token_meta = {item_id: token_meta_all[item_id] for item_id in selected_ids}
    blocks = build_blocks(route, selected_ids, token_meta)
    model = None
    records = []
    started = time.time()
    try:
        model, tokenizer, device, placement = phase1138.load_model(model_name, prereg)
        precision = quantization_audit(model)
        expected = prereg["models"][model_name]
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != int(expected["expected_parameter_count"]):
            raise RuntimeError("parameter count mismatch")
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("FP16/no-quantization gate failed")
        layers = get_layers(model)
        depth_rows = [row for row in phase1138.depth_rows_for_model(len(layers)) if math.isclose(float(row["requested_fraction"]), REQUESTED_FRACTION, abs_tol=1e-12)]
        if len(depth_rows) != 1:
            raise RuntimeError("depth map drift")
        depth = int(depth_rows[0]["depth"])
        layer = layers[depth - 1]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        batch_size = int(expected["batch_size"])
        vectors = phase1140.capture_candidate_paths(model, layer, cases, tokenizer, int(pad_id), device, batch_size)
        blocks_per_forward = max(1, batch_size // 8)
        for start in range(0, len(blocks), blocks_per_forward):
            current_blocks = blocks[start : start + blocks_per_forward]
            entries = [row for block in current_blocks for row in block]
            for row in entries:
                row.update({"model": model_name, "split": split, "depth": depth, "relative_depth": depth / len(layers), "requested_fraction": REQUESTED_FRACTION})
            records.extend(score_batch(model, layer, entries, cases, vectors, tokenizer, int(pad_id), device))
            completed = min(start + blocks_per_forward, len(blocks))
            if completed % 25 == 0 or completed == len(blocks):
                print(canonical({"phase": PHASE, "model": model_name, "split": split, "route": route, "comparisons": completed, "records": len(records)}), flush=True)
        if len(records) != EXPECTED_RECORDS:
            raise RuntimeError(f"record count drift: {len(records)}")
        metrics, curves, comparisons = analyze_route(records)
        core = {"phase": PHASE, "model": model_name, "split": split, "route": route, "protocol_digest": prereg["protocol_digest"], "precision": precision, "parameter_count": parameter_count, "placement": placement, "layer_count": len(layers), "depth": depth, "relative_depth": depth / len(layers), "item_count": len(selected_items), "record_count": len(records), "metrics": metrics, "elapsed_seconds": time.time() - started, "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "record_digest": digest(records), "evidence_scope": "calibrated_symmetric_factorial_whole_residual_sufficiency"}
        summary = dict(core)
        summary["summary_digest"] = digest(core)
        write_jsonl(output_root / "records.jsonl", records)
        write_jsonl(output_root / "curves.jsonl", curves)
        write_jsonl(output_root / "comparisons.jsonl", comparisons)
        write_json(output_root / "summary.json", summary)
        print(canonical({"model": model_name, "split": split, "route": route, "qualified": metrics["route_qualified"], "item_advantage": metrics["item_advantage_median"], "top1": metrics["diagonal_top1_fraction"], "summary_digest": summary["summary_digest"]}))
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def binding_selection_command(split: str) -> None:
    prereg = verify_protocol()
    summaries = {model: read_json(OUT_ROOT / "runs" / split / "binding" / model / "summary.json") for model in MODELS}
    passed = all(bool(summary["metrics"]["route_qualified"]) for summary in summaries.values())
    selection = {"phase": PHASE, "split": split, "protocol_digest": prereg["protocol_digest"], "both_models_binding_qualified": passed, "common_control_authorized": passed, "model_results": {model: bool(summary["metrics"]["route_qualified"]) for model, summary in summaries.items()}, "summary_digests": {model: summary["summary_digest"] for model, summary in summaries.items()}}
    selection["selection_digest"] = digest(selection)
    write_json(OUT_ROOT / f"analysis/{split}_binding_selection.json", selection)
    print(canonical(selection))


def specificity_metrics(binding: list[dict[str, Any]], common: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["comparison_id"]: row for row in binding}
    right = {row["comparison_id"]: row for row in common}
    if set(left) != set(right):
        raise RuntimeError("route comparison keys drift")
    rows = []
    for key in sorted(left):
        b, c = left[key], right[key]
        value = float(b["diagonal_advantage"]) - float(c["diagonal_advantage"])
        rows.append({"comparison_id": key, "item_id": b["item_id"], "property_id": b["property_id"], "same_relation": bool(b["same_relation"]), "panel": b["panel"], "binding_advantage": b["diagonal_advantage"], "common_advantage": c["diagonal_advantage"], "binding_minus_common": value, "positive": value > 0.0})
    same = [row for row in rows if row["same_relation"]]
    cross = [row for row in rows if not row["same_relation"]]
    per_property = {prop: median(row["binding_minus_common"] for row in rows if row["property_id"] == prop) for prop in PROPERTIES}
    metrics = {"binding_minus_common_advantage_median": median(row["binding_minus_common"] for row in rows), "binding_minus_common_positive_fraction": fraction(rows, "positive"), "binding_minus_common_same_relation_median": median(row["binding_minus_common"] for row in same), "binding_minus_common_cross_relation_median": median(row["binding_minus_common"] for row in cross), "per_property_binding_minus_common_median": per_property}
    checks = {"median": metrics["binding_minus_common_advantage_median"] is not None and metrics["binding_minus_common_advantage_median"] >= THRESHOLDS["binding_minus_common_advantage_median"], "positive_fraction": metrics["binding_minus_common_positive_fraction"] >= THRESHOLDS["binding_minus_common_positive_fraction"], "same_relation": metrics["binding_minus_common_same_relation_median"] is not None and metrics["binding_minus_common_same_relation_median"] >= THRESHOLDS["binding_minus_common_same_relation_median"], "cross_relation": metrics["binding_minus_common_cross_relation_median"] is not None and metrics["binding_minus_common_cross_relation_median"] >= THRESHOLDS["binding_minus_common_cross_relation_median"], "each_property": all(per_property[prop] is not None and float(per_property[prop]) >= THRESHOLDS["per_property_binding_minus_common_median"] for prop in PROPERTIES)}
    metrics["checks"] = checks
    metrics["qualified"] = all(checks.values())
    metrics["row_digest"] = digest(rows)
    return {"metrics": metrics, "rows": rows}


def specificity_selection_command(split: str) -> None:
    prereg = verify_protocol()
    results = {}
    for model in MODELS:
        binding_summary = read_json(OUT_ROOT / "runs" / split / "binding" / model / "summary.json")
        common_summary = read_json(OUT_ROOT / "runs" / split / "common_control" / model / "summary.json")
        binding = read_jsonl(OUT_ROOT / "runs" / split / "binding" / model / "comparisons.jsonl")
        common = read_jsonl(OUT_ROOT / "runs" / split / "common_control" / model / "comparisons.jsonl")
        specific = specificity_metrics(binding, common)
        qualified = bool(binding_summary["metrics"]["route_qualified"] and specific["metrics"]["qualified"])
        results[model] = {"binding_qualified": bool(binding_summary["metrics"]["route_qualified"]), "common_route_qualified": bool(common_summary["metrics"]["route_qualified"]), "specificity": specific["metrics"], "qualified": qualified}
        write_jsonl(OUT_ROOT / f"analysis/{split}_{model}_specificity_rows.jsonl", specific["rows"])
    both = all(bool(row["qualified"]) for row in results.values())
    selection = {"phase": PHASE, "split": split, "protocol_digest": prereg["protocol_digest"], "models": results, "both_models_specificity_qualified": both, "confirmation_authorized": bool(split == "discovery" and both), "final_confirmation_passed": bool(split == "confirmation" and both)}
    selection["selection_digest"] = digest(selection)
    write_json(OUT_ROOT / f"analysis/{split}_specificity_selection.json", selection)
    print(canonical(selection))


def finalize_command() -> None:
    prereg = verify_protocol()
    discovery_binding = read_json(OUT_ROOT / "analysis/discovery_binding_selection.json")
    discovery_specific_path = OUT_ROOT / "analysis/discovery_specificity_selection.json"
    discovery_specific = read_json(discovery_specific_path) if discovery_specific_path.exists() else None
    confirmation_path = OUT_ROOT / "analysis/confirmation_specificity_selection.json"
    confirmation = read_json(confirmation_path) if confirmation_path.exists() else None
    confirmed = bool(confirmation and confirmation["final_confirmation_passed"])
    if not discovery_binding["both_models_binding_qualified"]:
        outcome = "discovery_binding_identity_failed"
    elif not discovery_specific or not discovery_specific["both_models_specificity_qualified"]:
        outcome = "discovery_binding_not_above_common_control"
    elif not confirmed:
        outcome = "confirmation_failed_or_not_run"
    else:
        outcome = "factorial_binding_identity_confirmed"
    final = {"phase": PHASE, "protocol_digest": prereg["protocol_digest"], "outcome": outcome, "discovery_binding_passed": bool(discovery_binding["both_models_binding_qualified"]), "discovery_specificity_ran": discovery_specific is not None, "discovery_specificity_passed": bool(discovery_specific and discovery_specific["both_models_specificity_qualified"]), "confirmation_ran": confirmation is not None, "confirmed": confirmed, "component_search_authorized": confirmed, "necessity_authorized": False, "semantic_payload_claim_authorized": False, "auto_continue": confirmed, "claim_scope": "At most whole-residual factorial binding sufficiency beyond a common-route control; exact diagonal reconstruction alone is an algebraic instrument check."}
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    run = sub.add_parser("run")
    run.add_argument("--model", choices=MODELS, required=True)
    run.add_argument("--split", choices=SPLITS, required=True)
    run.add_argument("--route", choices=ROUTES, required=True)
    select = sub.add_parser("select")
    select.add_argument("--split", choices=SPLITS, required=True)
    select.add_argument("--stage", choices=("binding", "specificity"), required=True)
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_command(args.model, args.split, args.route)
    elif args.command == "select" and args.stage == "binding":
        binding_selection_command(args.split)
    elif args.command == "select":
        specificity_selection_command(args.split)
    elif args.command == "finalize":
        finalize_command()


if __name__ == "__main__":
    main()
