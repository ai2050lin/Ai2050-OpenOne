#!/usr/bin/env python3
"""C131 repaired-interface two-hop precedence typed-transition campaign."""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1665_c131_composed_precedence_repaired_transition"
C130 = RESULT / "phase1664_c130_composed_precedence_typed_transition"
C129 = RESULT / "phase1663_c129_direct_precedence_typed_transition"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1661_c127_typed_transition_language_family as c127
import phase1664_c130_composed_precedence_typed_transition as base

PHASE = 1665
CAMPAIGN = "C131"
PARTITIONS = base.PARTITIONS
ROLES = base.ROLES
CHECKPOINTS = base.CHECKPOINTS
DIM = base.DIM
SUPPORT_K = base.SUPPORT_K
WIDTH = base.WIDTH
SYLLABLES = ("bex", "cog", "dyr", "fal", "gim", "hov", "jup", "kes", "lin", "mox", "nur", "pav", "qit", "res", "syl", "tob")

# Reuse the already audited numeric executor while replacing only campaign identity and output root.
base.OUT = OUT
base.PHASE = PHASE
base.CAMPAIGN = CAMPAIGN


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def values_for(index: int) -> tuple[str, str, str, str, str]:
    a = SYLLABLES[index % len(SYLLABLES)]
    b = SYLLABLES[(index * 9 + 3) % len(SYLLABLES)]
    return tuple(f"Deka{a}{b}{index:02d}{suffix}" for suffix in ("a", "b", "c", "d", "e"))


def prompt_for(values: tuple[str, ...], truth: int, surface: int, distractor: int) -> tuple[str, str, str]:
    source, bridge, target, extra_left, extra_right = values
    first_forward = truth == 1 or distractor == -1
    second_forward = truth == 1 or distractor == 1
    first_left, first_right = (source, bridge) if first_forward else (bridge, source)
    second_left, second_right = (bridge, target) if second_forward else (target, bridge)
    extra_a, extra_b = (extra_left, extra_right) if distractor == 1 else (extra_right, extra_left)
    if surface == 1:
        prefix_a, prefix_b, prefix_extra = "Route record", "Continuation", "Separate record"
    else:
        prefix_a, prefix_b, prefix_extra = "Schedule note", "Next link", "Unrelated note"
    prompt = (
        f"Route rule: a claim is established only when the listed before-links form a directed path. "
        f"{prefix_a}: {first_left} comes before {first_right}. "
        f"{prefix_b}: {second_left} comes before {second_right}. "
        f"{prefix_extra}: {extra_a} comes before {extra_b}. "
        f"Question: Does the record establish that {source} comes before {target}? Reply exactly yes or no."
    )
    return prompt, source, target


def material() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for unit_index in range(32):
        values = values_for(unit_index)
        partition = PARTITIONS[unit_index // 16]
        unit = {"unit_id": f"c131-{unit_index:02d}", "family": "two_hop_precedence_repaired_interface", "partition": partition, "world": "controlled_synthetic_two_hop_precedence", "values": list(values)}
        units.append(unit)
        for truth, surface, distractor in itertools.product((1, -1), repeat=3):
            prompt, query_left, query_right = prompt_for(values, truth, surface, distractor)
            cases.append({**unit, "case_id": f"c131-{len(cases):04d}", "truth_factor": truth, "surface_factor": surface, "distractor_factor": distractor, "truth": truth == 1, "output_yes": truth == 1, "gold_position": 0 if truth == 1 else 1, "query_left": query_left, "query_right": query_right, "prompt": prompt})
    return units, cases


def historical_values() -> set[str]:
    result: set[str] = set()
    for path in RESULT.glob("phase*/material/units.jsonl"):
        for row in core.rows(path):
            result.update(str(value).casefold() for value in row.get("values", []))
    return result


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C131 already exists: {OUT}")
    parent_closure = core.load(C130 / "analysis/closure.json")
    parent_audit = core.load(C130 / "audit/independent_behavior_failure_audit.json")
    c129_freeze = core.load(C129 / "protocol/frozen_discovery_nomination.json")
    units, cases = material()
    compiled = base.compile_rows(graph_base.tokenizer(), cases)
    fresh = [str(value).casefold() for row in units for value in row["values"]]
    old = historical_values()
    cells = Counter((row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"]) for row in cases)
    zero = {
        "always_yes": float(np.mean([row["truth"] for row in cases])),
        "always_no": float(np.mean([not row["truth"] for row in cases])),
        "surface_only": float(np.mean([(row["surface_factor"] == 1) == row["truth"] for row in cases])),
        "distractor_only": float(np.mean([(row["distractor_factor"] == 1) == row["truth"] for row in cases])),
        "first_link_only": float(np.mean([((row["truth_factor"] == 1 or row["distractor_factor"] == -1) == row["truth"]) for row in cases])),
        "second_link_only": float(np.mean([((row["truth_factor"] == 1 or row["distractor_factor"] == 1) == row["truth"]) for row in cases])),
    }
    checks = {
        "authorization": parent_audit["all_checks_passed"] and parent_audit["authorization"] == "start_c131_repaired_interface" and parent_closure["next_authorization"].startswith("C131 may freeze"),
        "units": len(units) == 32,
        "cases": len(cases) == 256,
        "factorial": cells == {(partition, *cell): 16 for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=3)},
        "freshness": not (set(fresh) & old) and len(fresh) == len(set(fresh)),
        "unique_prompts": len({row["prompt"] for row in cases}) == 256,
        "zero_models": all(value <= 0.75 for value in zero.values()) and zero["first_link_only"] == 0.75 and zero["second_link_only"] == 0.75,
        "candidate_ids": all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "semantic_uniqueness": all(row["values"][0] != row["values"][2] and row["query_left"] == row["values"][0] and row["query_right"] == row["values"][2] for row in cases),
        "machine_naturalness": all("comes before" in row["prompt"] and "appears earlier" not in row["prompt"] and row["prompt"].endswith("Reply exactly yes or no.") for row in cases),
        "typed_reference": c129_freeze["nominee"]["role"] == "boundary" and c129_freeze["nominee"]["transition_index"] == 35,
        "threshold_preservation": core.load(C130 / "protocol/preregistration.json")["behavior_gate"] == {"global_accuracy_min": 0.95, "partition_accuracy_min": 0.90, "truth_accuracy_min": 0.90, "surface_accuracy_min": 0.90, "global_margin_over_best_single_link_min": 0.20},
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(set(fresh) & old)})
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    source_paths = {
        "c130_closure": C130 / "analysis/closure.json",
        "c130_failure_audit": C130 / "audit/independent_behavior_failure_audit.json",
        "c129_nomination": C129 / "protocol/frozen_discovery_nomination.json",
        "c129_vector": C129 / "analysis/discovery_nominee_increment.float32.npy",
    }
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "repaired_two_hop_precedence_cross_family_contract_frozen",
        "object": "behavior-qualified two-hop precedence truth-response trajectory with relation wording held at comes-before and discourse-frame surface variation",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "units": 32,
        "cases": 256,
        "partitions": list(PARTITIONS),
        "factors": ["truth", "discourse_surface", "which_false_link"],
        "roles": list(ROLES),
        "checkpoints": list(CHECKPOINTS),
        "activation_coordinates": DIM,
        "zero_models": zero,
        "behavior_gate": {"global_accuracy_min": 0.95, "partition_accuracy_min": 0.90, "truth_accuracy_min": 0.90, "surface_accuracy_min": 0.90, "global_margin_over_best_single_link_min": 0.20},
        "discovery_rule": {"partition": "discovery", "unit_split": "first eight versus last eight", "score": "max(0,split_half_cosine)*min(split_half_L2_norms)", "support_k": SUPPORT_K},
        "within_family_confirmation_gates": {"cosine_min": 0.90, "top256_overlap_min": 0.50, "support_sign_agreement_min": 0.75, "coordinate_clock_within_one_min": 0.70, "wrong_state_margin_gt": 0.0, "wrong_role_margin_gt": 0.0},
        "cross_family_frozen_candidate": {"source_family": "C129 direct_precedence", "role": "boundary", "transition_index": 35, "from_checkpoint": CHECKPOINTS[35], "to_checkpoint": CHECKPOINTS[36], "reference_vector_sha256": core.sha(C129 / "analysis/discovery_nominee_increment.float32.npy")},
        "cross_family_confirmation_gates": {"cosine_min": 0.90, "top256_overlap_min": 0.50, "support_sign_agreement_min": 0.75},
        "composition_residual_confirmation_gates": {"cosine_min": 0.90, "top256_overlap_min": 0.50, "support_sign_agreement_min": 0.75, "residual_l2_min": 2.5, "residual_fraction_min": 0.05},
        "residual_rule": "alpha_D=<C_D,D>/<D,D>; U_D=C_D-alpha_D*D; U_C=C_C-alpha_D*D",
        "stop_conditions": ["behavior failure forbids HiddenState capture", "numeric capture failure", "confirmation failures close only their named claim route without reselection or threshold changes"],
        "observation_policy": "full 2560 activation coordinates; no PCA/SVD, attention, MLP, or weight analysis",
        "naturalness_scope": "deterministic machine grammar audit only; no independent human naturalness lock",
        "claim_boundary": "controlled two-hop precedence activation observations; cross-family similarity can indicate shared truth/output preparation and is not by itself a composition operator, semantic neuron, causal path, or new mathematics",
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_hashes": {name: core.sha(path) for name, path in source_paths.items()},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_c130_behavior",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "max_width": max(len(row["prompt_ids"]) for row in compiled), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior_report = core.load(OUT / "analysis/behavior_gate.json")
    confirmation = core.load(OUT / "analysis/confirmation.json")
    fields = np.load(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", mmap_mode="r")
    raw = np.load(OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy", mmap_mode="r")
    payload = core.load(PUBLIC)
    nominee = confirmation["within_family_nominee"]
    role_index = int(nominee["role_index"])
    effect_rows, profiles = [], []
    for partition_index, partition in enumerate(PARTITIONS):
        mean = np.mean(np.asarray(fields[partition_index * 16:(partition_index + 1) * 16, role_index], dtype=np.float32), axis=0, dtype=np.float32)
        increments = mean[1:] - mean[:-1]
        profiles.append({"partition": partition, "role": nominee["role"], "values": [float(np.linalg.norm(value)) for value in increments]})
        effect_rows.extend({"partition": partition, "role": nominee["role"], "kind": "truth_response", "checkpoint": CHECKPOINTS[index], "checkpoint_index": index, "values": mean[index].tolist()} for index in range(38))
        effect_rows.extend({"partition": partition, "role": nominee["role"], "kind": "truth_response_increment", "from_checkpoint": CHECKPOINTS[index], "to_checkpoint": CHECKPOINTS[index + 1], "transition_index": index, "values": increments[index].tolist()} for index in range(37))
    residual_rows = []
    for label, path in (
        ("c129_direct_reference", OUT / "analysis/c129_direct_reference_increment.float32.npy"),
        ("c131_composed_discovery", OUT / "analysis/discovery_composed_fixed_increment.float32.npy"),
        ("c131_composed_confirmation", OUT / "analysis/confirmation_composed_fixed_increment.float32.npy"),
        ("composition_residual_discovery", OUT / "analysis/discovery_composition_residual.float32.npy"),
        ("composition_residual_confirmation", OUT / "analysis/confirmation_composition_residual.float32.npy"),
    ):
        residual_rows.append({"label": label, "role": "boundary", "transition_index": 35, "values": np.load(path).astype(np.float32).tolist()})
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    representative = []
    for local_role_index, role in enumerate(ROLES):
        for checkpoint_index in (0, 8, 16, 24, 32, 36, 37):
            representative.append({"case_id": compiled[0]["case_id"], "role": role, "checkpoint": CHECKPOINTS[checkpoint_index], "checkpoint_index": checkpoint_index, "token_positions": compiled[0]["role_positions"][role], "values": c127.decode(raw[0, local_role_index, checkpoint_index]).tolist()})
    payload["c131_composed_precedence_repaired_transition_batch"] = {"protocol": protocol, "behavior": behavior_report["summary"], "confirmation": confirmation, "profiles": profiles, "effect_rows": effect_rows, "cross_family_and_residual_rows": residual_rows, "representative_raw_rows": representative}
    payload.update({"phase": PHASE, "campaign": "C109-C131", "title": "Role-State Atlas + Repaired Direct/Composed Precedence Comparison", "claim_boundary": "C131 adds a behavior-qualified two-hop precedence trajectory with exact checkpoint types and full coordinates. Shared direct/composed late truth response is separated from a frozen residual test; neither is a causal language operator."})
    core.save(PUBLIC, payload)
    heatmap = {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC), "effect_rows": len(effect_rows), "residual_rows": len(residual_rows), "representative_raw_rows": len(representative), "activation_coordinates": DIM}
    within_pass = confirmation["within_family"]["all_gates_passed"]
    cross_pass = confirmation["cross_family_common_response"]["all_gates_passed"]
    residual_pass = confirmation["composition_residual"]["all_gates_passed"]
    puzzles = {
        "K320": "A behavior-qualified two-hop precedence family has an exact-checkpoint full-coordinate truth-response trajectory with untouched lexical confirmation.",
        "K321": "A prospectively frozen C129 direct-precedence boundary increment is adjudicated against C131 composition and typed only as a shared late truth/output component if it passes.",
    }
    if residual_pass:
        puzzles["K322_candidate"] = "After subtracting the frozen scalar direct component, a nontrivial composition residual repeats across lexical partitions; it remains an observational candidate rather than an operator."
    closure = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "repaired_two_hop_typed_cross_family_stage_closed",
        "headline": {"behavior": behavior_report["summary"], "within_family_passed": within_pass, "cross_family_common_response_passed": cross_pass, "composition_residual_passed": residual_pass},
        "results": confirmation,
        "new_puzzles": puzzles,
        "theory_update": "The typed transition atlas now explicitly separates late truth/output convergence from a composition-specific residual test under a behavior-qualified two-hop task.",
        "unified_formula": "D=DeltaE_truth^direct(boundary,block34->35); C=DeltaE_truth^composed(boundary,block34->35); alpha=<C_D,D>/<D,D>; U=C-alpha D.",
        "problems": ["controlled synthetic English", "Qwen3 only", "truth aligned with yes/no output polarity", "registered roles rather than every token", "machine naturalness audit only", "no intervention, attention, MLP, or weight evidence"],
        "claim_boundary": protocol["claim_boundary"],
        "heatmap": heatmap,
        "next_authorization": "C132 may use a new language family to test whether the frozen late common component generalizes and whether any C131 residual is family-specific; it must preserve exact checkpoint and contrast types.",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "behavior": core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"], "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"], "discovery": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"], "confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_integrity_checks_passed"], "heatmap": heatmap["effect_rows"] == 150 and heatmap["residual_rows"] == 5 and heatmap["representative_raw_rows"] == 49}
    core.save(OUT / "audit/internal_closure_audit.json", {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gates": {"behavior": behavior_report["gate_passed"], "within_family": within_pass, "cross_family_common_response": cross_pass, "composition_residual": residual_pass}, "asset_sha256": heatmap["sha256"], "authorization": "run_independent_c131_audit_and_integrate_client"})
    print(json.dumps(closure, indent=2))


def close_failure() -> None:
    behavior = core.load(OUT / "analysis/behavior_gate.json")
    if behavior["gate_passed"]:
        raise RuntimeError("C131 behavior passed")
    closure = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "repaired_interface_behavior_failed", "headline": behavior["summary"], "result": "C131 failed behavior qualification; no HiddenState was captured.", "theory_update": "No internal update.", "problems": ["two-hop interface remains behavior-ineligible"], "claim_boundary": "behavior only; no HiddenState or mechanism result", "heatmap": None, "next_authorization": "C132 must change the behavior object rather than retune C131."}
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"behavior_failed": not behavior["gate_passed"], "no_hiddenstate": not (OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy").exists()}
    core.save(OUT / "audit/internal_closure_audit.json", {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": False, "authorization": "independent_failure_audit"})
    print(json.dumps(closure, indent=2))


def main() -> None:
    modes = {"contract": contract, "behavior": base.behavior, "capture": base.capture, "discover": base.discover, "validate": base.validate, "synthesize": synthesize, "close-failure": close_failure}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit(f"usage: {Path(__file__).name} {{{'|'.join(modes)}}}")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
