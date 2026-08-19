#!/usr/bin/env python3
"""Phase1438: one-shot C072 exhaustive 24-permutation response spectrum."""
from __future__ import annotations

import inspect
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1392_c062_full_field_camera as batcher
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1438, "C072"
CONTRACT = TESTS / "result/phase1435_c072_permutation_spectrum_contract"
BEHAVIOR = TESTS / "result/phase1436_c072_behavior"
CAMERA = TESTS / "result/phase1437_c072_permutation_camera"
OUT = TESTS / "result/phase1438_c072_permutation_spectrum"
ROLES = ("record_target", "record_family", "query_target", "query_family")
DIRECTIONS = ("true_to_false", "false_to_true")
SPLITS = ("confirmation", "lockbox")
CONTROLS = ("self", "same_surface_identity", "wrong_cross_surface_identity")


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def margin(logits: torch.Tensor, row: dict) -> float:
    return float(logits[row["candidate_ids"][0][0]].float() - logits[row["candidate_ids"][1][0]].float())


def transfer_surfaces(transfer: str) -> tuple[str, str]:
    if transfer == "memo_contains_to_circle_roll":
        return "memo_contains", "circle_roll"
    if transfer == "circle_roll_to_memo_contains":
        return "circle_roll", "memo_contains"
    raise ValueError(transfer)


def direction_rows(case: dict, compiled: dict[str, dict], source_surface: str, target_surface: str, direction: str) -> tuple[dict, dict, dict, dict, float]:
    if direction == "true_to_false":
        return (
            compiled[case[f"{target_surface}_true_recipient"]],
            compiled[case[f"{target_surface}_false_donor"]],
            compiled[case[f"{source_surface}_false_donor"]],
            compiled[case[f"{source_surface}_true_donor"]], -1.0,
        )
    return (
        compiled[case[f"{target_surface}_false_recipient"]],
        compiled[case[f"{target_surface}_true_donor"]],
        compiled[case[f"{source_surface}_true_donor"]],
        compiled[case[f"{source_surface}_false_donor"]], 1.0,
    )


@torch.inference_mode()
def run_context(model, pad: int, device: torch.device, supports: bool, case: dict, transfer: str,
                direction: str, compiled: dict[str, dict], registry: list[dict], state_index: int) -> list[dict]:
    source_surface, target_surface = transfer_surfaces(transfer)
    recipient, same_donor, cross_donor, wrong_donor, orientation = direction_rows(case, compiled, source_surface, target_surface, direction)
    rows = [recipient, same_donor, cross_donor, wrong_donor, recipient, recipient, recipient] + [recipient for _ in registry]
    ids, mask, position_ids, offsets = batcher.make_batch(rows, pad, device)

    def points(row: dict, batch_index: int) -> dict[str, int]:
        return {role: batcher.points(row, offsets[batch_index], role)[0] for role in ROLES}

    def hook(_module, args):
        original = args[0]
        value = original.clone()
        recipient_points = points(recipient, 0)
        same_points = points(same_donor, 1)
        cross_points = points(cross_donor, 2)
        wrong_points = points(wrong_donor, 3)
        self_points = points(recipient, 4)
        same_arm_points = points(recipient, 5)
        wrong_arm_points = points(recipient, 6)
        for role in ROLES:
            value[4, self_points[role]] = original[0, recipient_points[role]]
            value[5, same_arm_points[role]] = original[1, same_points[role]]
            value[6, wrong_arm_points[role]] = original[3, wrong_points[role]]
        for offset, permutation in enumerate(registry, start=7):
            arm_points = points(recipient, offset)
            for role in ROLES:
                value[offset, arm_points[role]] = original[2, cross_points[permutation["mapping"][role]]]
        return (value,) + args[1:]

    handle = model.model.layers[state_index].register_forward_pre_hook(hook)
    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": position_ids, "use_cache": False, "return_dict": True}
    if supports:
        kwargs["logits_to_keep"] = 1
    try:
        output = model(**kwargs)
    finally:
        handle.remove()
    recipient_margin = margin(output.logits[0, -1], recipient)
    natural_margins = {"same": margin(output.logits[1, -1], same_donor), "cross": margin(output.logits[2, -1], cross_donor), "wrong": margin(output.logits[3, -1], wrong_donor)}
    base = {
        "set_id": case["set_id"], "partition": case["partition"], "family": case["family"], "donor_family": case["donor_family"],
        "surface_transfer": transfer, "source_surface": source_surface, "target_surface": target_surface,
        "direction": direction, "state_index": state_index, "source_length": len(cross_donor["prompt_ids"]), "target_length": len(recipient["prompt_ids"]),
        "recipient_margin": recipient_margin, "same_donor_margin": natural_margins["same"], "cross_donor_margin": natural_margins["cross"], "wrong_donor_margin": natural_margins["wrong"],
    }
    records = []
    for row_index, control in enumerate(CONTROLS, start=4):
        changed = margin(output.logits[row_index, -1], recipient)
        records.append({
            **base, "record_type": "control", "arm": control, "permutation_id": None,
            "swap_margin": changed, "oriented_gain": orientation * (changed - recipient_margin),
            "desired_sign": changed < 0 if direction == "true_to_false" else changed > 0,
            "wrong_expected_sign": changed > 0 if direction == "true_to_false" else changed < 0,
            "recipient_output_max_abs_diff": float((output.logits[row_index, -1].float() - output.logits[0, -1].float()).abs().max()),
        })
    for row_index, permutation in enumerate(registry, start=7):
        changed = margin(output.logits[row_index, -1], recipient)
        records.append({
            **base, "record_type": "permutation", "arm": f"perm_{permutation['permutation_id']}",
            "permutation_id": permutation["permutation_id"], "identity": permutation["identity"],
            "fixed_points": permutation["fixed_points"], "parity": permutation["parity"], "cycle_type": permutation["cycle_type"],
            "preserves_entity_family_kind": permutation["preserves_entity_family_kind"], "preserves_record_query_axis": permutation["preserves_record_query_axis"],
            "swap_margin": changed, "oriented_gain": orientation * (changed - recipient_margin),
            "desired_sign": changed < 0 if direction == "true_to_false" else changed > 0,
            "wrong_expected_sign": changed > 0 if direction == "true_to_false" else changed < 0,
            "recipient_output_max_abs_diff": float((output.logits[row_index, -1].float() - output.logits[0, -1].float()).abs().max()),
        })
    del output, ids, mask, position_ids
    return records


def control_metrics(rows: list[dict]) -> dict:
    arms = {arm: [row for row in rows if row["arm"] == arm] for arm in CONTROLS}
    return {
        "count": len(arms["self"]),
        "self_output_max_abs_diff": max(row["recipient_output_max_abs_diff"] for row in arms["self"]),
        "same_surface_desired_sign_fraction": sum(row["desired_sign"] for row in arms["same_surface_identity"]) / len(arms["same_surface_identity"]),
        "wrong_expected_sign_fraction": sum(row["wrong_expected_sign"] for row in arms["wrong_cross_surface_identity"]) / len(arms["wrong_cross_surface_identity"]),
        "same_surface_gain_median": med([row["oriented_gain"] for row in arms["same_surface_identity"]]),
    }


def permutation_metrics(rows: list[dict], registry: list[dict]) -> dict[str, dict]:
    result = {}
    for permutation in registry:
        values = [row for row in rows if row["permutation_id"] == permutation["permutation_id"]]
        result[permutation["permutation_id"]] = {
            "count": len(values), "desired_sign_fraction": sum(row["desired_sign"] for row in values) / len(values),
            "oriented_gain_median": med([row["oriented_gain"] for row in values]), "margin_median": med([row["swap_margin"] for row in values]),
        }
    return result


def compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(p[q[index]] for index in range(4))


def inverse(p: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * 4
    for index, value in enumerate(p):
        result[value] = index
    return tuple(result)


def subgroup(values: set[tuple[int, ...]]) -> bool:
    identity = tuple(range(4))
    return identity in values and all(inverse(value) in values for value in values) and all(compose(a, b) in values for a in values for b in values)


def descriptive_strata(rows: list[dict]) -> dict:
    fields = ("fixed_points", "parity", "cycle_type", "preserves_entity_family_kind", "preserves_record_query_axis")
    output = {}
    for field in fields:
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[field]).lower()].append(row)
        output[field] = {key: {"count": len(values), "desired_sign_fraction": sum(row["desired_sign"] for row in values) / len(values), "oriented_gain_median": med([row["oriented_gain"] for row in values])} for key, values in sorted(groups.items())}
    return output


def classify(aggregate: dict, family: dict, registry: list[dict], gate: dict, transfers: list[str]) -> tuple[dict, str]:
    identity = next(row["permutation_id"] for row in registry if row["identity"])
    tuple_by_id = {row["permutation_id"]: tuple(row["source_indices_by_target"]) for row in registry}
    all_ids = set(tuple_by_id)
    cells = {}
    for transfer in transfers:
        cells[transfer] = {}
        for direction in DIRECTIONS:
            controls = aggregate[transfer][direction]["controls"]
            same_families = [name for name, values in family[transfer][direction].items() if all(values[split]["controls"]["same_surface_desired_sign_fraction"] >= gate["family_sign_fraction_min"] for split in SPLITS)]
            wrong_families = [name for name, values in family[transfer][direction].items() if all(values[split]["controls"]["wrong_expected_sign_fraction"] >= gate["family_sign_fraction_min"] for split in SPLITS)]
            executor = all(controls[split]["self_output_max_abs_diff"] <= gate["self_max_abs_diff"] and controls[split]["same_surface_desired_sign_fraction"] >= gate["same_surface_desired_sign_fraction_min"] and controls[split]["wrong_expected_sign_fraction"] >= gate["wrong_expected_sign_fraction_min"] for split in SPLITS) and len(same_families) >= gate["minimum_family_breadth"] and len(wrong_families) >= gate["minimum_family_breadth"]
            qualified = []
            breadth = {}
            for permutation in registry:
                pid = permutation["permutation_id"]
                qualifying_families = [name for name, values in family[transfer][direction].items() if all(values[split]["permutations"][pid]["desired_sign_fraction"] >= gate["family_sign_fraction_min"] for split in SPLITS)]
                breadth[pid] = qualifying_families
                if all(aggregate[transfer][direction]["permutations"][split][pid]["desired_sign_fraction"] >= gate["permutation_desired_sign_fraction_min"] and aggregate[transfer][direction]["permutations"][split][pid]["oriented_gain_median"] >= gate["permutation_oriented_gain_median_min"] for split in SPLITS) and len(qualifying_families) >= gate["minimum_family_breadth"]:
                    qualified.append(pid)
            gap = {}
            symmetry = {}
            for split in SPLITS:
                pm = aggregate[transfer][direction]["permutations"][split]
                nonidentity = [pid for pid in all_ids if pid != identity]
                identity_rows = {row["set_id"]: row for row in aggregate[transfer][direction]["raw_by_split"][split] if row["permutation_id"] == identity}
                paired_gaps = []
                for pid in nonidentity:
                    other = {row["set_id"]: row for row in aggregate[transfer][direction]["raw_by_split"][split] if row["permutation_id"] == pid}
                    paired_gaps.append(med([identity_rows[set_id]["oriented_gain"] - other[set_id]["oriented_gain"] for set_id in identity_rows]))
                gains = [pm[pid]["oriented_gain_median"] for pid in all_ids]
                gap[split] = {"sign": pm[identity]["desired_sign_fraction"] - max(pm[pid]["desired_sign_fraction"] for pid in nonidentity), "paired_gain": min(paired_gaps)}
                symmetry[split] = (max(gains) - min(gains)) / (abs(pm[identity]["oriented_gain_median"]) + 1e-12)
            cells[transfer][direction] = {"executor_pass": executor, "same_control_families": same_families, "wrong_control_families": wrong_families, "qualified_permutations": sorted(qualified), "qualified_count": len(qualified), "family_breadth": breadth, "identity_vs_best_nonidentity": gap, "symmetric_gain_range_ratio": symmetry}
    flat = [cells[transfer][direction] for transfer in transfers for direction in DIRECTIONS]
    executor_all = all(cell["executor_pass"] for cell in flat)
    qualified_sets = [set(cell["qualified_permutations"]) for cell in flat]
    identity_only = executor_all and all(values == {identity} for values in qualified_sets)
    identity_gap = all(cell["identity_vs_best_nonidentity"][split]["sign"] >= gate["identity_vs_best_nonidentity_sign_gap_min"] and cell["identity_vs_best_nonidentity"][split]["paired_gain"] >= gate["identity_vs_best_nonidentity_gain_gap_median_min"] for cell in flat for split in SPLITS)
    all_permutations = executor_all and all(values == all_ids for values in qualified_sets)
    symmetric = all(cell["symmetric_gain_range_ratio"][split] <= gate["symmetric_gain_range_ratio_max"] for cell in flat for split in SPLITS)
    stable_set = len({frozenset(values) for values in qualified_sets}) == 1
    common = qualified_sets[0] if stable_set else set()
    proper_subgroup = executor_all and stable_set and 1 < len(common) < 24 and subgroup({tuple_by_id[pid] for pid in common})
    if identity_only and identity_gap:
        overall = "role_order_selective"
    elif all_permutations and symmetric:
        overall = "permutation_symmetric_multiset"
    elif proper_subgroup:
        overall = "subgroup_structured"
    else:
        overall = "heterogeneous_or_executor_failed"
    return cells, overall


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1438 exists")
    camera_final = core.load(CAMERA / "analysis/final.json")
    camera_audit = core.load(CAMERA / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    if camera_final["authorization"] != "run_phase1438_c072_permutation_spectrum" or not camera_audit["all_checks_passed"]:
        raise RuntimeError("camera gate missing")
    selected = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    holdouts = [row for row in selected if row["partition"] in SPLITS]
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    registry = core.rows(CONTRACT / "material/permutation_registry.jsonl")
    gate = protocol["mechanism"]
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for case in holdouts:
            for transfer in gate["surface_transfers"]:
                for direction in gate["directions"]:
                    records.extend(run_context(model, pad, device, supports, case, transfer, direction, compiled, registry, protocol["state_index"]))
        core.write_rows(OUT / "raw/permutation_spectrum.jsonl", records)
        aggregate = {}
        family = {}
        strata = {}
        for transfer in gate["surface_transfers"]:
            aggregate[transfer], family[transfer], strata[transfer] = {}, {}, {}
            for direction in DIRECTIONS:
                aggregate[transfer][direction] = {
                    "controls": {split: control_metrics([row for row in records if row["surface_transfer"] == transfer and row["direction"] == direction and row["partition"] == split and row["record_type"] == "control"]) for split in SPLITS},
                    "permutations": {split: permutation_metrics([row for row in records if row["surface_transfer"] == transfer and row["direction"] == direction and row["partition"] == split and row["record_type"] == "permutation"], registry) for split in SPLITS},
                    "raw_by_split": {split: [row for row in records if row["surface_transfer"] == transfer and row["direction"] == direction and row["partition"] == split and row["record_type"] == "permutation"] for split in SPLITS},
                }
                family[transfer][direction] = {
                    name: {
                        split: {
                            "controls": control_metrics([row for row in records if row["surface_transfer"] == transfer and row["direction"] == direction and row["partition"] == split and row["family"] == name and row["record_type"] == "control"]),
                            "permutations": permutation_metrics([row for row in records if row["surface_transfer"] == transfer and row["direction"] == direction and row["partition"] == split and row["family"] == name and row["record_type"] == "permutation"], registry),
                        }
                        for split in SPLITS
                    }
                    for name in behavior_final["qualified_families"]
                }
                strata[transfer][direction] = {split: descriptive_strata(aggregate[transfer][direction]["raw_by_split"][split]) for split in SPLITS}
        cells, overall = classify(aggregate, family, registry, gate, gate["surface_transfers"])
        compact_aggregate = {transfer: {direction: {"controls": aggregate[transfer][direction]["controls"], "permutations": aggregate[transfer][direction]["permutations"]} for direction in DIRECTIONS} for transfer in gate["surface_transfers"]}
        checks = {
            "holdouts": len(holdouts) == 48 and all(sum(row["partition"] == split for row in holdouts) == 24 for split in SPLITS),
            "record_count": len(records) == 48 * 2 * 2 * 27,
            "control_count": sum(row["record_type"] == "control" for row in records) == 48 * 2 * 2 * 3,
            "permutation_count": sum(row["record_type"] == "permutation" for row in records) == 48 * 2 * 2 * 24,
            "balance": all(sum(row["partition"] == split and row["surface_transfer"] == transfer and row["direction"] == direction and row["permutation_id"] == permutation["permutation_id"] for row in records) == 24 for split in SPLITS for transfer in gate["surface_transfers"] for direction in DIRECTIONS for permutation in registry),
            "holdout_only": {row["partition"] for row in records} == set(SPLITS), "state16": {row["state_index"] for row in records} == {16},
            "different_shapes": all(row["source_length"] != row["target_length"] for row in records),
            "finite": all(math.isfinite(row[key]) for row in records for key in ("recipient_margin", "swap_margin", "oriented_gain", "recipient_output_max_abs_diff")),
            "bf16": quant["has_bf16_parameters"], "not_quantized": not quant["has_quantized_modules"],
        }
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "holdout_set_count": len(holdouts), "record_count": len(records),
            "aggregate_metrics": compact_aggregate, "family_metrics": family, "descriptive_strata": strata,
            "cell_results": cells, "overall_classification": overall, "checks": checks, "all_execution_checks_passed": all(checks.values()),
            "contract_sha256": protocol["contract_sha256"], "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/permutation_spectrum_summary.json", summary)
        core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_execution_checks_passed": summary["all_execution_checks_passed"], "overall_classification": overall, "cell_results": cells, "authorization": "run_phase1439_c072_campaign_closure"})
        print(json.dumps({key: value for key, value in summary.items() if key not in ("family_metrics", "aggregate_metrics")}, indent=2))
        print(json.dumps(compact_aggregate, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
