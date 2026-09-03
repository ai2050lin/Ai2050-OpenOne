#!/usr/bin/env python3
"""C132 fixed-main-frame two-hop precedence campaign."""
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
OUT = RESULT / "phase1666_c132_fixed_frame_composed_precedence"
C131 = RESULT / "phase1665_c131_composed_precedence_repaired_transition"
C129 = RESULT / "phase1663_c129_direct_precedence_typed_transition"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1661_c127_typed_transition_language_family as c127
import phase1664_c130_composed_precedence_typed_transition as base

PHASE = 1666
CAMPAIGN = "C132"
PARTITIONS = base.PARTITIONS
ROLES = base.ROLES
CHECKPOINTS = base.CHECKPOINTS
DIM = base.DIM
SUPPORT_K = base.SUPPORT_K
WIDTH = base.WIDTH
SYLLABLES = ("bov", "caz", "dit", "fup", "ger", "hix", "jol", "kum", "ler", "miv", "nox", "pud", "qas", "rim", "sov", "tux")

base.OUT = OUT
base.PHASE = PHASE
base.CAMPAIGN = CAMPAIGN


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def values_for(index: int) -> tuple[str, str, str, str, str]:
    a = SYLLABLES[index % len(SYLLABLES)]
    b = SYLLABLES[(index * 11 + 1) % len(SYLLABLES)]
    return tuple(f"Ennea{a}{b}{index:02d}{suffix}" for suffix in ("a", "b", "c", "d", "e"))


def prompt_for(values: tuple[str, ...], truth: int, label_factor: int, false_link_factor: int) -> tuple[str, str, str]:
    source, bridge, target, extra_left, extra_right = values
    first_forward = truth == 1 or false_link_factor == -1
    second_forward = truth == 1 or false_link_factor == 1
    first_left, first_right = (source, bridge) if first_forward else (bridge, source)
    second_left, second_right = (bridge, target) if second_forward else (target, bridge)
    extra_a, extra_b = (extra_left, extra_right) if false_link_factor == 1 else (extra_right, extra_left)
    extra_label = "Separate record" if label_factor == 1 else "Independent record"
    prompt = (
        "Route rule: a claim is established only when the listed before-links form a directed path. "
        f"Route record: {first_left} comes before {first_right}. "
        f"Continuation: {second_left} comes before {second_right}. "
        f"{extra_label}: {extra_a} comes before {extra_b}. "
        f"Question: Does the route record establish that {source} comes before {target}? Reply exactly yes or no."
    )
    return prompt, source, target


def material() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for unit_index in range(32):
        values = values_for(unit_index)
        partition = PARTITIONS[unit_index // 16]
        unit = {"unit_id": f"c132-{unit_index:02d}", "family": "two_hop_precedence_fixed_main_frame", "partition": partition, "world": "controlled_synthetic_two_hop_precedence", "values": list(values)}
        units.append(unit)
        for truth, label_factor, false_link_factor in itertools.product((1, -1), repeat=3):
            prompt, query_left, query_right = prompt_for(values, truth, label_factor, false_link_factor)
            cases.append({**unit, "case_id": f"c132-{len(cases):04d}", "truth_factor": truth, "surface_factor": label_factor, "distractor_factor": false_link_factor, "truth": truth == 1, "output_yes": truth == 1, "gold_position": 0 if truth == 1 else 1, "query_left": query_left, "query_right": query_right, "prompt": prompt})
    return units, cases


def historical_values() -> set[str]:
    result: set[str] = set()
    for path in RESULT.glob("phase*/material/units.jsonl"):
        for row in core.rows(path):
            result.update(str(value).casefold() for value in row.get("values", []))
    return result


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C132 already exists: {OUT}")
    parent_closure = core.load(C131 / "analysis/closure.json")
    parent_audit = core.load(C131 / "audit/independent_behavior_failure_audit.json")
    c129_freeze = core.load(C129 / "protocol/frozen_discovery_nomination.json")
    units, cases = material()
    compiled = base.compile_rows(graph_base.tokenizer(), cases)
    fresh = [str(value).casefold() for row in units for value in row["values"]]
    old = historical_values()
    cells = Counter((row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"]) for row in cases)
    zero = {
        "always_yes": float(np.mean([row["truth"] for row in cases])),
        "always_no": float(np.mean([not row["truth"] for row in cases])),
        "label_only": float(np.mean([(row["surface_factor"] == 1) == row["truth"] for row in cases])),
        "false_link_selector_only": float(np.mean([(row["distractor_factor"] == 1) == row["truth"] for row in cases])),
        "first_link_only": float(np.mean([((row["truth_factor"] == 1 or row["distractor_factor"] == -1) == row["truth"]) for row in cases])),
        "second_link_only": float(np.mean([((row["truth_factor"] == 1 or row["distractor_factor"] == 1) == row["truth"]) for row in cases])),
    }
    checks = {
        "authorization": parent_audit["all_checks_passed"] and parent_audit["authorization"] == "start_c132_fixed_main_frame" and parent_closure["next_authorization"].startswith("C132 must change"),
        "units": len(units) == 32,
        "cases": len(cases) == 256,
        "factorial": cells == {(partition, *cell): 16 for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=3)},
        "freshness": not (set(fresh) & old) and len(fresh) == len(set(fresh)),
        "unique_prompts": len({row["prompt"] for row in cases}) == 256,
        "zero_models": all(value <= 0.75 for value in zero.values()) and zero["first_link_only"] == 0.75 and zero["second_link_only"] == 0.75,
        "candidate_ids": all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "fixed_main_frame": all("Route record:" in row["prompt"] and "Continuation:" in row["prompt"] and "comes before" in row["prompt"] and "Schedule note" not in row["prompt"] for row in cases),
        "semantic_uniqueness": all(row["query_left"] == row["values"][0] and row["query_right"] == row["values"][2] for row in cases),
        "typed_reference": c129_freeze["nominee"]["role"] == "boundary" and c129_freeze["nominee"]["transition_index"] == 35,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(set(fresh) & old)})
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    source_paths = {"c131_closure": C131 / "analysis/closure.json", "c131_failure_audit": C131 / "audit/independent_behavior_failure_audit.json", "c129_nomination": C129 / "protocol/frozen_discovery_nomination.json", "c129_vector": C129 / "analysis/discovery_nominee_increment.float32.npy"}
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "fixed_main_frame_two_hop_cross_family_contract_frozen",
        "object": "two-hop precedence truth-response trajectory with fixed main route frame and irrelevant-record-label variation",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "units": 32,
        "cases": 256,
        "partitions": list(PARTITIONS),
        "factors": ["truth", "irrelevant_record_label", "which_false_link"],
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


def close_failure() -> None:
    behavior = core.load(OUT / "analysis/behavior_gate.json")
    if behavior["gate_passed"]:
        raise RuntimeError("C132 behavior passed")
    results = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    cells = {}
    for truth in (1, -1):
        for label_factor in (1, -1):
            for false_link_factor in (1, -1):
                subset = [row for row in results if row["truth_factor"] == truth and row["surface_factor"] == label_factor and row["distractor_factor"] == false_link_factor]
                cells[f"truth={truth},label={label_factor},false_link={false_link_factor}"] = base.group_accuracy(subset)
    closure = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "fixed_frame_behavior_failed", "headline": behavior["summary"], "cell_accuracy": cells, "result": "C132 narrowly failed three frozen behavior gates; no HiddenState was captured.", "theory_update": "No internal update. Strong two-hop behavior was observed, but it did not satisfy the preregistered qualification contract needed for typed-state comparison.", "problems": ["global accuracy 0.94921875 is below 0.95", "false accuracy 0.8984375 is below 0.90", "margin over the 0.75 single-link null is 0.19921875 below 0.20", "Qwen3 only", "controlled synthetic English"], "claim_boundary": "behavior only; no embeddings, HiddenStates, C129 transfer, composition residual, or mechanism result", "heatmap": None, "next_authorization": "Close the direct-versus-composed branch. A future composition campaign must pre-register a substantively different behavior object or use an independently established natural benchmark; it may not retune C132 thresholds."}
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"behavior_failed": not behavior["gate_passed"], "three_named_gates_failed": behavior["summary"]["global_accuracy"] < 0.95 and behavior["summary"]["by_truth"]["-1"] < 0.90 and behavior["summary"]["margin_over_best_single_link"] < 0.20, "cell_count": len(cells) == 8, "no_hiddenstate": not (OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy").exists(), "no_confirmation": not (OUT / "analysis/confirmation.json").exists()}
    core.save(OUT / "audit/internal_closure_audit.json", {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": False, "authorization": "independent_failure_audit"})
    print(json.dumps(closure, indent=2))


def main() -> None:
    modes = {"contract": contract, "behavior": base.behavior, "capture": base.capture, "discover": base.discover, "validate": base.validate, "close-failure": close_failure}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit(f"usage: {Path(__file__).name} {{{'|'.join(modes)}}}")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
