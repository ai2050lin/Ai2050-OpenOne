#!/usr/bin/env python3
"""Phase1621 / C113: fourth-lexicon coordinate and role-coalition interventions."""
from __future__ import annotations

import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1618_c113_fourth_lexicon_role_lattice_replication"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1610_c110_frozen_transport_comparison as transport


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def med(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


@torch.inference_mode()
def main() -> None:
    contract = core.load(OUT / "protocol/preregistration.json")
    field = core.load(OUT / "analysis/field_adjudication.json")
    field_audit = core.load(OUT / "audit/independent_field_adjudication_audit.json")
    if contract["authorization"] != "execute_phase1619_c113_exact_field_capture" or field["authorization"] != "execute_phase1621_c113_coordinate_and_role_interventions_regardless_of_field_gate" or not field_audit["all_checks_passed"]:
        raise RuntimeError("C113 intervention authorization missing")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    pairs = transport.build_pairs(rows, contract)
    for index, pair in enumerate(pairs):
        pair["pair_id"] = f"c113-pair-{index:04d}"
    if len(pairs) != 192:
        raise RuntimeError(len(pairs))
    all_roles = tuple(contract["roles"])
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for pair in pairs:
        lengths = tuple(len(pair["recipient"]["role_positions"][role]) for role in all_roles)
        grouped[(pair["family"], pair["partition"], pair["code"], lengths)].append(pair)
    results = []
    model = None
    first_repeat = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for (family, partition, code, lengths), group in sorted(grouped.items()):
            support_name = "attribute_binding_k256" if family == "attribute_binding" else "agent_patient_k128"
            support = torch.tensor(contract["supports"][support_name], dtype=torch.long, device=device)
            permutations = [torch.tensor(values, dtype=torch.long, device=device) for values in contract["movement_permutations"][family]]
            for start in range(0, len(group), int(contract["numeric"]["batch_size"])):
                batch = group[start:start + int(contract["numeric"]["batch_size"])]
                recipients = [pair["recipient"] for pair in batch]
                donors = [pair["donor"] for pair in batch]
                recipient_logits, recipient_states = transport.forward_with_roles(model, recipients, all_roles, 19, pad, device, 224)
                donor_logits, donor_states = transport.forward_with_roles(model, donors, all_roles, 19, pad, device, 224)
                if first_repeat is None:
                    first_repeat = (recipients, recipient_logits.detach().clone(), {role: value.detach().clone() for role, value in recipient_states.items()})
                rq = recipient_states["query_anchor"]
                dq = donor_states["query_anchor"]
                target_delta = dq[..., support] - rq[..., support]
                target_norm = torch.sqrt(torch.sum(target_delta.float() ** 2, dim=(1, 2))).clamp_min(1e-12)
                patches: dict[str, dict[str, torch.Tensor]] = {}
                value = rq.clone()
                value[..., support] = dq[..., support]
                patches["frozen_support"] = {"query_anchor": value}
                permutation_norms = []
                for permutation_index, permutation in enumerate(permutations):
                    value = rq.clone()
                    value[..., support] = (rq[..., support].float() + target_delta.float()[..., permutation]).to(rq.dtype)
                    patches[f"movement_permutation_{permutation_index}"] = {"query_anchor": value}
                    permutation_norms.append(torch.sqrt(torch.sum((value[..., support] - rq[..., support]).float() ** 2, dim=(1, 2))).clamp_min(1e-12))
                for role in all_roles:
                    patches[f"single_{role}"] = {role: donor_states[role].clone()}
                for name, roles in contract["role_coalitions"].items():
                    patches[f"coalition_{name}"] = {role: donor_states[role].clone() for role in roles}
                patched_logits = {mode: transport.forward_patched_roles(model, recipients, values, 19, pad, device, 224) for mode, values in patches.items()}
                for local, pair in enumerate(batch):
                    base = transport.margin(recipient_logits[local], recipients[local])
                    donor_margin = transport.margin(donor_logits[local], donors[local])
                    mode_results = {}
                    for mode, logits in patched_logits.items():
                        patched = transport.margin(logits[local], recipients[local])
                        gain = patched - base
                        mode_results[mode] = {
                            "yes_minus_no": patched, "truth_direction_gain": gain, "code_aligned_task_gain": code * gain,
                            "truth_flip": base <= 0.0 < patched, "task_flip": code * base <= 0.0 < code * patched,
                        }
                    results.append({
                        "pair_id": pair["pair_id"], "unit_id": pair["unit_id"], "family": family,
                        "partition": partition, "code": code, "surface_factor": pair["surface_factor"],
                        "distractor_factor": pair["distractor_factor"], "recipient_yes_minus_no": base,
                        "donor_yes_minus_no": donor_margin, "target_movement_l2": float(target_norm[local]),
                        "permutation_l2_relative_errors": [float(torch.abs(norm[local] - target_norm[local]) / target_norm[local]) for norm in permutation_norms],
                        "role_l2": {role: float(torch.sqrt(torch.sum((donor_states[role][local] - recipient_states[role][local]).float() ** 2))) for role in all_roles},
                        "modes": mode_results,
                    })
                print(f"[phase1621] {family}/{partition}/code={code}/lengths={lengths} {start + len(batch)}/{len(group)}", flush=True)
        repeat_rows, old_logits, old_states = first_repeat
        new_logits, new_states = transport.forward_with_roles(model, repeat_rows, all_roles, 19, pad, device, 224)
        repeat_logits = float(torch.max(torch.abs(new_logits - old_logits)))
        repeat_hidden = max(float(torch.max(torch.abs(new_states[role] - old_states[role]))) for role in all_roles)
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect()
    result_path = OUT / "analysis/intervention_results.jsonl"
    core.write_rows(result_path, results)
    summaries = []
    path = "coalition_record_to_query_path"
    all_mode = "coalition_all_registered_roles"
    for family in contract["families"]:
        for partition in contract["partitions"]:
            for code in (1, -1):
                selected = [row for row in results if row["family"] == family and row["partition"] == partition and row["code"] == code]
                correct = med([row["modes"]["frozen_support"]["truth_direction_gain"] for row in selected])
                permutation_medians = [med([row["modes"][f"movement_permutation_{index}"]["truth_direction_gain"] for row in selected]) for index in range(8)]
                role_medians = {role: med([row["modes"][f"single_{role}"]["truth_direction_gain"] for row in selected]) for role in all_roles}
                coalition_medians = {name: med([row["modes"][f"coalition_{name}"]["truth_direction_gain"] for row in selected]) for name in contract["role_coalitions"]}
                leave_effects = {role: coalition_medians["record_to_query_path"] - coalition_medians[f"path_without_{role}"] for role in ("focus_record", "focus_post", "query_focus", "query_anchor")}
                summaries.append({
                    "family": family, "partition": partition, "code": code, "pairs": len(selected),
                    "independent_units": len({row["unit_id"] for row in selected}),
                    "frozen_support_median_gain": correct, "movement_permutation_median_gains": permutation_medians,
                    "movement_permutation_median_of_medians": med(permutation_medians),
                    "frozen_support_gt_permutation_median": correct > med(permutation_medians),
                    "frozen_support_gt_all_permutation_medians": all(correct > value for value in permutation_medians),
                    "single_role_median_gains": role_medians, "coalition_median_gains": coalition_medians,
                    "leave_one_path_median_losses": leave_effects,
                    "record_path_gt_query": coalition_medians["record_to_query_path"] > role_medians["query_anchor"],
                    "all_roles_gt_path": coalition_medians["all_registered_roles"] > coalition_medians["record_to_query_path"],
                    "leave_query_anchor_lowers": leave_effects["query_anchor"] > 0,
                    "leave_query_focus_lowers": leave_effects["query_focus"] > 0,
                    "staged_increments": {
                        "code_over_path": coalition_medians["path_plus_code"] - coalition_medians["record_to_query_path"],
                        "boundary_over_path_code": coalition_medians["path_plus_code_boundary"] - coalition_medians["path_plus_code"],
                        "all_over_path_code_boundary": coalition_medians["all_registered_roles"] - coalition_medians["path_plus_code_boundary"],
                    },
                    "truth_flip_rates": {mode: float(np.mean([row["modes"][mode]["truth_flip"] for row in selected])) for mode in contract["modes"]},
                })
    summary_path = OUT / "analysis/intervention_summary.jsonl"
    core.write_rows(summary_path, summaries)
    attr = [row for row in summaries if row["family"] == "attribute_binding"]
    agent = [row for row in summaries if row["family"] == "agent_patient"]
    predictions = {
        "attribute_frozen_gt_all_permutation_cells": int(sum(row["frozen_support_gt_all_permutation_medians"] for row in attr)),
        "agent_record_path_gt_query_cells": int(sum(row["record_path_gt_query"] for row in agent)),
        "agent_all_roles_gt_path_cells": int(sum(row["all_roles_gt_path"] for row in agent)),
        "agent_leave_query_anchor_lowers_cells": int(sum(row["leave_query_anchor_lowers"] for row in agent)),
        "agent_leave_query_focus_lowers_cells": int(sum(row["leave_query_focus_lowers"] for row in agent)),
    }
    max_l2_error = max(error for row in results for error in row["permutation_l2_relative_errors"])
    checks = {
        "rows": len(results) == 192,
        "modes": all(set(row["modes"]) == set(contract["modes"]) for row in results),
        "summary": len(summaries) == 8 and all(row["pairs"] == 24 and row["independent_units"] == 6 for row in summaries),
        "l2_preserved": max_l2_error <= float(contract["numeric"]["movement_permutation_actual_l2_relative_tolerance"]),
        "finite": all(math.isfinite(row["modes"][mode]["truth_direction_gain"]) for row in results for mode in contract["modes"]),
        "repeat_hidden": repeat_hidden == 0.0, "repeat_logits": repeat_logits == 0.0,
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "sources_unchanged": all(core.sha(Path(contract["source_paths"][name])) == digest for name, digest in contract["source_hashes"].items()),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "max_l2_error": max_l2_error})
    report = {
        "phase": 1621, "campaign": "C113", "created_at_utc": now(), "status": "fourth_lexicon_coordinate_role_interventions_complete",
        "checks": checks, "max_permutation_l2_relative_error": max_l2_error, "predictions": predictions,
        "summaries": summaries, "runtime": {"placement": placement, "quantization": quant},
        "producer_sha256": core.sha(Path(__file__)), "results_sha256": core.sha(result_path), "summary_sha256": core.sha(summary_path),
        "claim_boundary": contract["claim_boundary"], "authorization": "run_phase1622_c113_synthesis_heatmap_and_closure",
    }
    core.save(OUT / "analysis/intervention_adjudication.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"summaries", "runtime"}}, indent=2))


if __name__ == "__main__":
    main()
