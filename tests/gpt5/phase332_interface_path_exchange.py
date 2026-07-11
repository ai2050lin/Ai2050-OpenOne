#!/usr/bin/env python3
"""Run registered raw/aligned interface path exchanges on Phase332 heldout items."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase330_nine_family_case_bank import MODELS  # noqa: E402
import phase326_distributed_carrier_atlas as phase326  # noqa: E402
from phase331_refined_mechanism_audit import capture_values, run_condition  # noqa: E402
from phase332_interface_branch_case_bank import EXCHANGE_CONDITIONS, ROUND_DEFAULT  # noqa: E402


PHASE = "Phase332"
SCHEMA_VERSION = "10.0.0"
OUT = ROOT / "tests/gpt5/result/phase332_interface_branch_atlas"
TRACE_CONDITIONS = {
    "baseline", "shared_plus_branch_correct",
    "shared_plus_branch_wrong_item", "matched_random_units_correct",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd", row_group_size=32768)


def intervention_specs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for source in rows:
        row = dict(source)
        row["position_role"] = "last" if row["position_role"] == "answer_start" else row["position_role"]
        key = (
            row["component_type"], row["component_layer"], row["position_role"],
            row["component_index"], row["component_start"], row["component_end"],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def exchange_case(recipient: dict[str, Any], direction: str) -> dict[str, Any]:
    return {
        **recipient,
        "audit_case_id": (
            f"phase332_exchange_{recipient['model']}_{recipient['family_id']}_{recipient['mechanism_id']}_"
            f"{recipient['item_index']:02d}_{recipient['template_id']}_{direction}"
        ),
        "exchange_direction": direction,
    }


def run_model(model: str, round_name: str, max_new_tokens: int) -> dict[str, Any]:
    root = OUT / round_name
    model_dir = root / "exchange" / model
    complete_path = model_dir / "complete.json"
    if complete_path.exists():
        return read_json(complete_path)
    cases = [
        row for row in read_jsonl(root / "phase332_registered_cases.jsonl")
        if row["model"] == model and row["split"] == "heldout"
    ]
    lookup = {
        (row["family_id"], row["mechanism_id"], row["item_index"], row["template_id"], row["interface"]): row
        for row in cases
    }
    member_rows = read_jsonl(root / "survey" / model / "member_sets.jsonl")
    by_set: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in member_rows:
        by_set[(row["family_id"], row["mechanism_id"], row["set_type"], row["interface"])].append(row)
    mechanisms = sorted({(row["family_id"], row["mechanism_id"], row["cohort"]) for row in cases})
    registered = []
    for family, mechanism, cohort in mechanisms:
        for item_index in range(4, 8):
            wrong_item = 4 + ((item_index - 4 + 1) % 4)
            for template_id in ("template_a", "template_b", "template_c"):
                for direction, donor_interface, recipient_interface in (
                    ("raw_to_answer_aligned", "raw_completion", "answer_aligned_chat"),
                    ("answer_aligned_to_raw", "answer_aligned_chat", "raw_completion"),
                ):
                    recipient = exchange_case(
                        lookup[(family, mechanism, item_index, template_id, recipient_interface)], direction
                    )
                    donor = lookup[(family, mechanism, item_index, template_id, donor_interface)]
                    wrong = lookup[(family, mechanism, wrong_item, template_id, donor_interface)]
                    registered.append({
                        "recipient": recipient,
                        "donor": donor,
                        "wrong_donor": wrong,
                        "cohort": cohort,
                        "donor_interface": donor_interface,
                    })
    if len(registered) != 96:
        raise RuntimeError(f"Expected 96 exchange cases for {model}, got {len(registered)}")
    loaded = None
    condition_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    registry_rows = []
    try:
        loaded = load_probe_model(model)
        for case_index, entry in enumerate(registered, 1):
            recipient = entry["recipient"]
            donor = entry["donor"]
            wrong_donor = entry["wrong_donor"]
            family = recipient["family_id"]
            mechanism = recipient["mechanism_id"]
            donor_interface = entry["donor_interface"]
            shared = intervention_specs(by_set[(family, mechanism, "shared_skeleton", "shared_all_unique_interfaces")])
            branch = intervention_specs(by_set[(family, mechanism, "interface_branch", donor_interface)])
            combined = intervention_specs([*shared, *branch])
            random_specs = phase326.randomize_specs(loaded.model, combined) if combined else []
            shared_values = capture_values(loaded, donor, shared) if shared else {}
            branch_values = capture_values(loaded, donor, branch) if branch else {}
            combined_values = capture_values(loaded, donor, combined) if combined else {}
            wrong_values = capture_values(loaded, wrong_donor, combined) if combined else {}
            random_values = capture_values(loaded, donor, random_specs) if random_specs else {}
            plan = {
                "baseline": ([], {}),
                "shared_skeleton_correct": (shared, shared_values),
                "interface_branch_correct": (branch, branch_values),
                "shared_plus_branch_correct": (combined, combined_values),
                "shared_plus_branch_wrong_item": (combined, wrong_values),
                "matched_random_units_correct": (random_specs, random_values),
            }
            registry_rows.append({
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "exchange_case_id": recipient["audit_case_id"],
                "model": model,
                "family_id": family,
                "mechanism_id": mechanism,
                "cohort": recipient["cohort"],
                "item_index": recipient["item_index"],
                "template_id": recipient["template_id"],
                "exchange_direction": recipient["exchange_direction"],
                "donor_case_id": donor["case_id"],
                "recipient_case_id": recipient["case_id"],
                "wrong_donor_case_id": wrong_donor["case_id"],
                "shared_member_count": len(shared),
                "interface_branch_member_count": len(branch),
                "combined_member_count": len(combined),
                "condition_count": len(EXCHANGE_CONDITIONS),
                "selection_updates_allowed": False,
            })
            case_results = []
            for condition in EXCHANGE_CONDITIONS:
                specs, values = plan[condition]
                trace = recipient["template_id"] == "template_c" and condition in TRACE_CONDITIONS
                row, new_paths, new_units = run_condition(
                    loaded, recipient, condition, combined, [], specs, values,
                    max_new_tokens, trace, force_generation=True,
                )
                row.update({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "exchange_case_id": recipient["audit_case_id"],
                    "exchange_direction": recipient["exchange_direction"],
                    "donor_interface": donor_interface,
                    "recipient_interface": recipient["interface"],
                    "shared_member_count": len(shared),
                    "interface_branch_member_count": len(branch),
                    "combined_member_count": len(combined),
                    "evidence_level": "L4_registered_interface_path_exchange",
                })
                for path in new_paths:
                    path.update(phase_id=PHASE, schema_version=SCHEMA_VERSION, exchange_direction=recipient["exchange_direction"])
                for unit in new_units:
                    unit.update(phase_id=PHASE, schema_version=SCHEMA_VERSION, exchange_direction=recipient["exchange_direction"])
                case_results.append(row)
                path_rows.extend(new_paths)
                unit_rows.extend(new_units)
            baseline = next(row for row in case_results if row["condition"] == "baseline")
            for row in case_results:
                row["delta_target_margin_vs_baseline"] = round(row["target_margin"] - baseline["target_margin"], 7)
                row["delta_phrase_logprob_vs_baseline"] = round(
                    row["target_phrase_logprob"] - baseline["target_phrase_logprob"], 7
                )
                row["behavior_lost_vs_baseline"] = bool(
                    baseline["behavior_success"] and not row["behavior_success"]
                )
                row["behavior_gained_vs_baseline"] = bool(
                    not baseline["behavior_success"] and row["behavior_success"]
                )
                row["protocol_lost_vs_baseline"] = bool(
                    baseline["protocol_success_answer_segment"]
                    and not row["protocol_success_answer_segment"]
                )
                row["generation_changed_vs_baseline"] = (
                    row["generated_token_ids"] != baseline["generated_token_ids"]
                )
            condition_rows.extend(case_results)
            if case_index % 8 == 0:
                print(json.dumps({
                    "quality_only": True, "model": model,
                    "exchange_cases": case_index, "total_cases": len(registered),
                    "condition_rows": len(condition_rows), "path_rows": len(path_rows),
                }), flush=True)
        write_jsonl(model_dir / "registered_exchange_cases.jsonl", registry_rows)
        write_jsonl(model_dir / "exchange_rows.jsonl", condition_rows)
        write_parquet(model_dir / "exchange_rows.parquet", condition_rows)
        write_parquet(model_dir / "exchange_path_rows.parquet", path_rows)
        write_parquet(model_dir / "exchange_unit_rows.parquet", unit_rows)
        quality = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "exchange_case_count": len(registered),
            "condition_row_count": len(condition_rows),
            "generation_row_count": sum(row["generation_executed"] for row in condition_rows),
            "trace_condition_count": sum(row["trace_executed"] for row in condition_rows),
            "exchange_path_row_count": len(path_rows),
            "exchange_unit_row_count": len(unit_rows),
            "empty_shared_case_count": sum(row["shared_member_count"] == 0 for row in registry_rows),
            "empty_branch_case_count": sum(row["interface_branch_member_count"] == 0 for row in registry_rows),
            "selection_updates_allowed": False,
            "single_unit_intervention_gate_open": False,
            "valid": len(registered) == 96 and len(condition_rows) == 576,
        }
        write_json(complete_path, quality)
        return quality
    finally:
        release_loaded(loaded)
        gc.collect()


def collect(round_name: str) -> dict[str, Any]:
    root = OUT / round_name
    survey_quality = []
    exchange_quality = []
    baseline_rows = []
    member_rows = []
    exchange_rows = []
    registry_rows = []
    path_tables = []
    unit_tables = []
    exchange_path_tables = []
    exchange_unit_tables = []
    for model in MODELS:
        survey_dir = root / "survey" / model
        exchange_dir = root / "exchange" / model
        survey_quality.append(read_json(survey_dir / "complete.json"))
        exchange_quality.append(read_json(exchange_dir / "complete.json"))
        baseline_rows.extend(read_jsonl(survey_dir / "baseline_rows.jsonl"))
        member_rows.extend(read_jsonl(survey_dir / "member_sets.jsonl"))
        exchange_rows.extend(read_jsonl(exchange_dir / "exchange_rows.jsonl"))
        registry_rows.extend(read_jsonl(exchange_dir / "registered_exchange_cases.jsonl"))
        path_tables.append(pq.read_table(survey_dir / "natural_path_rows.parquet"))
        unit_tables.append(pq.read_table(survey_dir / "natural_unit_rows.parquet"))
        exchange_path_tables.append(pq.read_table(exchange_dir / "exchange_path_rows.parquet"))
        exchange_unit_tables.append(pq.read_table(exchange_dir / "exchange_unit_rows.parquet"))
    write_jsonl(root / "phase332_baseline_rows.jsonl", baseline_rows)
    write_jsonl(root / "phase332_member_sets.jsonl", member_rows)
    write_jsonl(root / "phase332_registered_exchange_cases.jsonl", registry_rows)
    write_jsonl(root / "phase332_exchange_rows.jsonl", exchange_rows)
    write_parquet(root / "phase332_baseline_rows.parquet", baseline_rows)
    write_parquet(root / "phase332_exchange_rows.parquet", exchange_rows)
    pq.write_table(
        pa.concat_tables(path_tables, promote_options="permissive"),
        root / "phase332_natural_path_rows.parquet", compression="zstd",
    )
    pq.write_table(
        pa.concat_tables(unit_tables, promote_options="permissive"),
        root / "phase332_natural_unit_rows.parquet", compression="zstd",
    )
    pq.write_table(
        pa.concat_tables(exchange_path_tables, promote_options="permissive"),
        root / "phase332_exchange_path_rows.parquet", compression="zstd",
    )
    pq.write_table(
        pa.concat_tables(exchange_unit_tables, promote_options="permissive"),
        root / "phase332_exchange_unit_rows.parquet", compression="zstd",
    )
    quality = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model_count": 3,
        "registered_interface_case_count": len(baseline_rows),
        "baseline_generation_count": sum(row["generated_text"] is not None for row in baseline_rows),
        "natural_path_row_count": sum(row["natural_path_row_count"] for row in survey_quality),
        "natural_unit_row_count": sum(row["natural_unit_row_count"] for row in survey_quality),
        "member_set_row_count": len(member_rows),
        "registered_exchange_case_count": len(registry_rows),
        "exchange_condition_row_count": len(exchange_rows),
        "exchange_generation_count": sum(row["generation_executed"] for row in exchange_rows),
        "exchange_path_row_count": sum(row["exchange_path_row_count"] for row in exchange_quality),
        "exchange_unit_row_count": sum(row["exchange_unit_row_count"] for row in exchange_quality),
        "all_survey_valid": all(row["valid"] for row in survey_quality),
        "all_exchange_valid": all(row["valid"] for row in exchange_quality),
        "selection_updates_allowed": False,
        "single_unit_intervention_gate_open": False,
    }
    quality["valid"] = (
        len(baseline_rows) == 1152 and len(registry_rows) == 288
        and len(exchange_rows) == 1728 and quality["all_survey_valid"] and quality["all_exchange_valid"]
    )
    write_json(root / "phase332_execution_quality.json", quality)
    return quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.model:
        result = run_model(args.model, args.round, args.max_new_tokens)
    elif args.collect:
        result = collect(args.round)
    else:
        raise SystemExit("Use --model MODEL or --collect")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
