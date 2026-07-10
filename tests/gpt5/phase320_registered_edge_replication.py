#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
sys.path.insert(0, str(ROOT / "tests/glm5"))
sys.stdout.reconfigure(encoding="utf-8")

import phase318_natural_source_state_transfer as p318  # noqa: E402
import phase319_heldout_component_mediation as p319  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402


PHASE = "Phase320"
SCHEMA_VERSION = "4.3.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUND_DEFAULT = "registered_edge_replication"
OUT = ROOT / "tests/gpt5/result/phase320_registered_edge_replication"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"
PHASE318_ROOT = ROOT / "tests/gpt5/result/phase318_natural_source_state_transfer/natural_source_state_transfer"
PHASE319_ROOT = ROOT / "tests/gpt5/result/phase319_heldout_component_mediation/heldout_component_mediation"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def aliases(value: str) -> list[str]:
    return list(dict.fromkeys([value, value.lower(), value.capitalize()]))


REPLICATION_ITEMS = {
    "category_binding": [
        ("dolphin", "animal", ["tool", "plant", "material"]),
        ("pliers", "tool", ["animal", "plant", "material"]),
        ("bamboo", "plant", ["animal", "tool", "material"]),
        ("steel", "material", ["animal", "tool", "plant"]),
    ],
    "material_binding": [
        ("vase", "glass", ["wood", "paper", "rubber"]),
        ("desk", "wood", ["glass", "paper", "rubber"]),
        ("leaflet", "paper", ["glass", "wood", "rubber"]),
        ("hose", "rubber", ["glass", "wood", "paper"]),
    ],
}


def build_replication_bank() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    for mechanism, items in REPLICATION_ITEMS.items():
        label = "category" if mechanism == "category_binding" else "material"
        for index, (obj, target, distractors) in enumerate(items):
            prompt = f"In a temporary inventory, the entry for {obj} lists its {label} as {target}. Return only that recorded {label} for {obj}. Reply:"
            cases.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "case_id": f"phase320:{mechanism}:{index}",
                    "family_id": "content_knowledge",
                    "mechanism_id": mechanism,
                    "split": "registered_replication",
                    "template_id": "template_d_independent",
                    "prompt": prompt,
                    "source_surface": target,
                    "query_surface": label,
                    "target": target,
                    "target_aliases": aliases(target),
                    "distractor_aliases": distractors,
                    "object": obj,
                    "independent_case": True,
                }
            )
    by_mechanism = {mechanism: [r for r in cases if r["mechanism_id"] == mechanism] for mechanism in REPLICATION_ITEMS}
    pairs: list[dict[str, Any]] = []
    for mechanism, rows in by_mechanism.items():
        other_mechanism = "material_binding" if mechanism == "category_binding" else "category_binding"
        unrelated_rows = by_mechanism[other_mechanism]
        for index, recipient in enumerate(rows):
            donor = rows[(index + 1) % len(rows)]
            unrelated = unrelated_rows[index % len(unrelated_rows)]
            pairs.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "pair_id": f"phase320:pair:{mechanism}:{index}",
                    "family_id": "content_knowledge",
                    "mechanism_id": mechanism,
                    "split": "registered_replication",
                    "template_id": "template_d_independent",
                    "recipient_case_id": recipient["case_id"],
                    "donor_case_id": donor["case_id"],
                    "unrelated_control_case_id": unrelated["case_id"],
                    "recipient_target": recipient["target"],
                    "donor_target": donor["target"],
                    "targets_differ": True,
                    "selection_frozen_before_replication": True,
                }
            )
    return cases, pairs


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    cases, pairs = build_replication_bank()
    case_map = {str(r["case_id"]): r for r in cases}
    source_rows = read_jsonl(PHASE318_ROOT / f"phase318_{args.model}_source_layer_selection_rows.jsonl")
    component_rows = read_jsonl(PHASE319_ROOT / f"phase319_{args.model}_component_selection_rows.jsonl")
    source_map = {(str(r["family_id"]), str(r["mechanism_id"])): int(r["selected_source_layer"]) for r in source_rows}
    component_map = {(str(r["family_id"]), str(r["mechanism_id"]), str(r["component_type"])): r for r in component_rows}
    if not source_map or not component_map:
        raise FileNotFoundError("Phase318/319 frozen selections are required")
    model_obj = tokenizer = None
    condition_rows: list[dict[str, Any]] = []
    replication_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for index, pair in enumerate(pairs, 1):
            try:
                recipient = case_map[str(pair["recipient_case_id"])]
                donor = case_map[str(pair["donor_case_id"])]
                unrelated = case_map[str(pair["unrelated_control_case_id"])]
                key = (str(pair["family_id"]), str(pair["mechanism_id"]))
                source_layer = source_map[key]
                attention_spec = dict(component_map[(key[0], key[1], "attention_head_input")])
                product_spec = dict(component_map[(key[0], key[1], "mlp_product_group")])
                rec_encoded, rec_positions = p318.encode_case(tokenizer, recipient, device)
                donor_encoded, donor_positions = p318.encode_case(tokenizer, donor, device)
                unrelated_encoded, unrelated_positions = p318.encode_case(tokenizer, unrelated, device)
                _, donor_states = p318.forward_states(model_obj, donor_encoded, donor_positions)
                _, unrelated_states = p318.forward_states(model_obj, unrelated_encoded, unrelated_positions)
                source_position = int(rec_positions["source"]["token_position"])
                query_position = int(rec_positions["query"]["token_position"])
                attention_spec["token_position"] = int(rec_positions[str(attention_spec["position_role"])]["token_position"])
                product_spec["token_position"] = int(rec_positions[str(product_spec["position_role"])]["token_position"])
                layers = sorted({int(attention_spec["component_layer"]), int(product_spec["component_layer"])})
                base_logits, base_attn, base_product = p319.capture_component_vectors(model_obj, rec_encoded, rec_positions, layers)
                source_vector = donor_states["source"][source_layer]
                source_logits, _, _ = p319.capture_component_vectors(
                    model_obj,
                    rec_encoded,
                    rec_positions,
                    layers,
                    install_extra=lambda: [p319.source_patch_handle(model_obj, source_layer, source_position, source_vector)],
                )
                attn_key = (int(attention_spec["component_layer"]), str(attention_spec["position_role"]))
                product_key = (int(product_spec["component_layer"]), str(product_spec["position_role"]))
                attention_baseline = base_attn[attn_key]
                product_baseline = base_product[product_key]
                conditions = {
                    "baseline": base_logits,
                    "source_replace": source_logits,
                    "source_attention_head_restore": p319.forward_condition(model_obj, rec_encoded, source_layer, source_position, source_vector, attention_spec, attention_baseline),
                    "source_mlp_product_restore": p319.forward_condition(model_obj, rec_encoded, source_layer, source_position, source_vector, product_spec=product_spec, product_baseline=product_baseline),
                    "source_both_restore": p319.forward_condition(model_obj, rec_encoded, source_layer, source_position, source_vector, attention_spec, attention_baseline, product_spec, product_baseline),
                    "unrelated_replace": p319.forward_condition(model_obj, rec_encoded, source_layer, source_position, unrelated_states["source"][source_layer]),
                    "wrong_position_replace": p319.forward_condition(model_obj, rec_encoded, source_layer, query_position, source_vector),
                }
                rows_by_condition = {}
                for condition, logits in conditions.items():
                    row = p319.condition_row(args.model, pair, condition, source_layer, base_logits, logits, donor, recipient, tokenizer)
                    row["phase_id"] = PHASE
                    condition_rows.append(row)
                    rows_by_condition[condition] = row
                source_shift = safe_float(rows_by_condition["source_replace"]["donor_transfer_shift"])
                unrelated_shift = safe_float(rows_by_condition["unrelated_replace"]["donor_transfer_shift"])
                wrong_shift = safe_float(rows_by_condition["wrong_position_replace"]["donor_transfer_shift"])
                attention_loss = source_shift - safe_float(rows_by_condition["source_attention_head_restore"]["donor_transfer_shift"])
                product_loss = source_shift - safe_float(rows_by_condition["source_mlp_product_restore"]["donor_transfer_shift"])
                joint_loss = source_shift - safe_float(rows_by_condition["source_both_restore"]["donor_transfer_shift"])
                corrected = source_shift - max(unrelated_shift, wrong_shift)
                passes = (
                    source_shift > 0.5
                    and corrected > 0.5
                    and bool(rows_by_condition["source_replace"]["patched_donor_wins"])
                    and attention_loss > max(0.5, 0.2 * abs(source_shift))
                    and product_loss > max(0.5, 0.1 * abs(source_shift))
                    and joint_loss > max(attention_loss, product_loss)
                )
                replication_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": args.model,
                        "pair_id": pair["pair_id"],
                        "family_id": pair["family_id"],
                        "mechanism_id": pair["mechanism_id"],
                        "source_layer": source_layer,
                        "source_transfer_shift": round(source_shift, 6),
                        "control_corrected_transfer": round(corrected, 6),
                        "attention_mediation_loss": round(attention_loss, 6),
                        "mlp_product_mediation_loss": round(product_loss, 6),
                        "joint_mediation_loss": round(joint_loss, 6),
                        "donor_winner": rows_by_condition["source_replace"]["patched_donor_wins"],
                        "registered_edge_criterion_passed": passes,
                        "evidence_level": "L5_replicated_candidate" if passes else "L4_intervention_effect",
                        "attention_spec": attention_spec,
                        "mlp_product_spec": product_spec,
                    }
                )
                base_text = p319.generate_text(model_obj, tokenizer, rec_encoded, source_layer, source_position, None, args.rollout_tokens)
                patched_text = p319.generate_text(model_obj, tokenizer, rec_encoded, source_layer, source_position, source_vector, args.rollout_tokens)
                rollout_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "model": args.model,
                        "pair_id": pair["pair_id"],
                        "mechanism_id": pair["mechanism_id"],
                        "recipient_target": recipient["target"],
                        "donor_target": donor["target"],
                        "baseline_rollout": base_text,
                        "source_patched_rollout": patched_text,
                        "rollout_changed": base_text != patched_text,
                        "patched_starts_with_donor": patched_text.strip().lower().startswith(str(donor["target"]).lower()),
                    }
                )
                print(f"{args.model}: registered edge replication {index}/{len(pairs)}", flush=True)
            except Exception as exc:  # noqa: BLE001
                missing.append({"phase_id": PHASE, "model": args.model, "pair_id": pair["pair_id"], "reason": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = summarize_model(args.model, condition_rows, replication_rows, rollout_rows, missing, attn_impl)
        out_dir = OUT / args.round_name
        write_json(out_dir / f"phase320_{args.model}_summary.json", summary)
        write_jsonl(out_dir / f"phase320_{args.model}_condition_rows.jsonl", condition_rows)
        write_jsonl(out_dir / f"phase320_{args.model}_replication_rows.jsonl", replication_rows)
        write_jsonl(out_dir / f"phase320_{args.model}_rollout_rows.jsonl", rollout_rows)
        write_jsonl(out_dir / f"phase320_{args.model}_missing_rows.jsonl", missing)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def summarize_model(model: str, conditions: list[dict[str, Any]], replications: list[dict[str, Any]], rollouts: list[dict[str, Any]], missing: list[dict[str, Any]], attn_impl: str) -> dict[str, Any]:
    source = [r for r in conditions if r["condition"] == "source_replace"]
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if not missing else "complete_with_missing",
        "model": model,
        "attention_implementation": attn_impl,
        "registered_replication_cases": len(replications),
        "condition_rows": len(conditions),
        "rollout_cases": len(rollouts),
        "missing_cases": len(missing),
        "source_transfer_mean": mean_safe([safe_float(r["donor_transfer_shift"]) for r in source]),
        "control_corrected_transfer_mean": mean_safe([safe_float(r["control_corrected_transfer"]) for r in replications]),
        "donor_win_rate": mean_safe([1.0 if r["patched_donor_wins"] else 0.0 for r in source]),
        "attention_mediation_loss_mean": mean_safe([safe_float(r["attention_mediation_loss"]) for r in replications]),
        "mlp_product_mediation_loss_mean": mean_safe([safe_float(r["mlp_product_mediation_loss"]) for r in replications]),
        "joint_mediation_loss_mean": mean_safe([safe_float(r["joint_mediation_loss"]) for r in replications]),
        "registered_pass_count": sum(1 for r in replications if r["registered_edge_criterion_passed"]),
        "patched_donor_start_rate": mean_safe([1.0 if r["patched_starts_with_donor"] else 0.0 for r in rollouts]),
    }


def collect(round_name: str) -> dict[str, Any]:
    cases, pairs = build_replication_bank()
    out_dir = OUT / round_name
    summaries = []
    conditions: list[dict[str, Any]] = []
    replications: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        path = out_dir / f"phase320_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        conditions.extend(read_jsonl(out_dir / f"phase320_{model}_condition_rows.jsonl"))
        replications.extend(read_jsonl(out_dir / f"phase320_{model}_replication_rows.jsonl"))
        rollouts.extend(read_jsonl(out_dir / f"phase320_{model}_rollout_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase320_{model}_missing_rows.jsonl"))
    source = [r for r in conditions if r["condition"] == "source_replace"]
    mechanism_pass = {}
    for mechanism in REPLICATION_ITEMS:
        rows = [r for r in replications if r["mechanism_id"] == mechanism]
        mechanism_pass[mechanism] = {
            "cases": len(rows),
            "pass_count": sum(1 for r in rows if r["registered_edge_criterion_passed"]),
            "pass_rate": mean_safe([1.0 if r["registered_edge_criterion_passed"] else 0.0 for r in rows]),
            "models_with_any_pass": sorted({r["model"] for r in rows if r["registered_edge_criterion_passed"]}),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if len(summaries) == len(MODELS) and not missing else "partial_or_missing",
        "frozen_replication_base_cases": len(cases),
        "frozen_replication_pairs": len(pairs),
        "registered_model_cases": len(replications),
        "condition_rows": len(conditions),
        "rollout_cases": len(rollouts),
        "missing_cases": len(missing),
        "source_transfer_mean": mean_safe([safe_float(r["donor_transfer_shift"]) for r in source]),
        "control_corrected_transfer_mean": mean_safe([safe_float(r["control_corrected_transfer"]) for r in replications]),
        "donor_win_rate": mean_safe([1.0 if r["patched_donor_wins"] else 0.0 for r in source]),
        "attention_mediation_loss_mean": mean_safe([safe_float(r["attention_mediation_loss"]) for r in replications]),
        "mlp_product_mediation_loss_mean": mean_safe([safe_float(r["mlp_product_mediation_loss"]) for r in replications]),
        "joint_mediation_loss_mean": mean_safe([safe_float(r["joint_mediation_loss"]) for r in replications]),
        "registered_pass_count": sum(1 for r in replications if r["registered_edge_criterion_passed"]),
        "registered_pass_rate": mean_safe([1.0 if r["registered_edge_criterion_passed"] else 0.0 for r in replications]),
        "mechanism_replication": mechanism_pass,
        "promoted_l5_edge_count": 0,
        "promotion_rule": "A mechanism must pass across parallel objects and at least two models; individual passes are not promoted.",
        "model_summaries": summaries,
    }
    for base in [V2, LEGACY_V2]:
        write_jsonl(base / "phase320_registered_replication_case_bank.jsonl", cases)
        write_jsonl(base / "phase320_registered_replication_pair_bank.jsonl", pairs)
        write_jsonl(base / "phase320_registered_condition_rows.jsonl", conditions)
        write_jsonl(base / "phase320_registered_edge_replication_rows.jsonl", replications)
        write_jsonl(base / "phase320_registered_rollout_rows.jsonl", rollouts)
        write_jsonl(base / "phase320_missing_rows.jsonl", missing)
        write_json(base / "phase320_registered_edge_replication_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--rollout-tokens", type=int, default=8)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        collect(args.round_name)
    elif args.model:
        run_model(args)
    else:
        raise SystemExit("use --model or --summarize")


if __name__ == "__main__":
    main()
