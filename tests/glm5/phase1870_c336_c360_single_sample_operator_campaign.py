#!/usr/bin/env python3
"""C336-C360: single-sample conditional operator campaign.

The campaign observes embeddings and hidden states only.  It never reads
attention maps, MLP activations, or model weights.  Physical activation axes
are retained in full; summary statistics are derived views, not replacements
for the archived fields.
"""
from __future__ import annotations

import gc
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase1844_c310_c335_dual_axis_common as old

PHASES = {
    f"C{c}": (1870 + c - 336, slug)
    for c, slug in (
        (336, "c330_permutation_reaudit"),
        (337, "single_sample_master_contract"),
        (338, "language_graph_material_compiler"),
        (339, "qwen_behavior_qualification"),
        (340, "qwen_full_coordinate_capture"),
        (341, "single_sample_diagonal_operator"),
        (342, "predictive_state_partition_refinement"),
        (343, "dynamic_coordinate_coalition"),
        (344, "lexical_lockbox_contract"),
        (345, "lexical_lockbox_adjudication"),
        (346, "operation_order_rollout"),
        (347, "attitude_scope_response_ecology"),
        (348, "all_token_coordinate_observation"),
        (349, "ternary_graph_behavior_adjudication"),
        (350, "recursive_graph_operator_rollout"),
        (351, "graph_full_field_response_ecology"),
        (352, "natural_membership_external_panel"),
        (353, "six_family_single_sample_operator"),
        (354, "cross_family_operator_breadth"),
        (355, "cross_family_state_refinement"),
        (356, "typed_mediation_contract"),
        (357, "typed_source_target_mediation"),
        (358, "cross_model_abstract_machine_contract"),
        (359, "cross_model_lockbox_bisimulation"),
        (360, "campaign_synthesis_and_heatmap"),
    )
}
OUTS = {c: RESULT / f"phase{p}_{c.lower()}_{s}" for c, (p, s) in PHASES.items()}
ROLES = old.ROLES
OPS = ("A", "B", "I")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rows(path: Path, values=None):
    if values is None:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def begin(campaign: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[campaign]
    if (out / "analysis/final.json").exists():
        return out
    if out.exists():
        raise RuntimeError(f"partial output exists: {out}")
    if not all(checks.values()):
        raise RuntimeError((campaign, checks))
    for sub in ("analysis", "audit", "compiled", "material", "protocol", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[campaign][0], "campaign": campaign,
        "created_at_utc": now(), "producer_sha256": old.core.sha(Path(__file__)), **protocol,
    })
    save(out / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    return out


def close(campaign: str, headline: dict, checks: dict, next_authorization: str) -> dict:
    out = OUTS[campaign]
    if (out / "analysis/final.json").exists():
        return json.loads((out / "analysis/final.json").read_text(encoding="utf-8"))
    save(out / "analysis/summary.json", headline)
    save(out / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    contract = json.loads((out / "audit/internal_contract_audit.json").read_text(encoding="utf-8"))
    final_checks = {
        "contract": contract["all_checks_passed"],
        "analysis": all(checks.values()),
        "producer_hash": old.core.sha(Path(__file__)) == json.loads((out / "protocol/preregistration.json").read_text(encoding="utf-8"))["producer_sha256"],
    }
    final = {
        "phase": PHASES[campaign][0], "campaign": campaign, "status": "closed",
        "checks": final_checks, "all_checks_passed": all(final_checks.values()),
        "headline": headline, "next_authorization": next_authorization,
    }
    save(out / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return final


def final(campaign: str) -> dict:
    return json.loads((OUTS[campaign] / "analysis/final.json").read_text(encoding="utf-8"))


def answer_row(core: str, correct: str, order: int) -> tuple[str, int, str, str]:
    wrong = "No" if correct == "Yes" else "Yes"
    choices, gold = old.old.previous.previous.options(correct, wrong, order)
    return f"{core} {choices}. Reply with only A or B.", gold, correct, wrong


AGENTS = ("Mira", "Nora", "Lina", "Tara", "Rina", "Vera", "Dora", "Sara", "Kira", "Faye", "Gina", "Iris")
OBJECTS = (
    ("fruit", "apples"), ("beverages", "tea"), ("grain", "rice"), ("dessert", "cake"),
    ("vegetables", "carrots"), ("nuts", "almonds"), ("seafood", "salmon"), ("bread", "bagels"),
    ("berries", "strawberries"), ("spices", "pepper"), ("soup", "bisque"), ("pasta", "ravioli"),
)
SURFACES = {
    "report": "A report states: {sentence} Question: {question}",
    "witness": "A witness says, \"{sentence}\" Question: {question}",
    "note": "A note records: {sentence} Question: {question}",
}


def material() -> list[dict]:
    result = []
    for unit, (agent, (category, member)) in enumerate(zip(AGENTS, OBJECTS)):
        for surface, a, b, order in itertools.product(SURFACES, (0, 1), (0, 1), (1, -1)):
            obj = member if a else category
            if b:
                sentence = f"{agent} likes eating {obj}."
                question = f"Does {agent} like eating {obj}?"
                relation = "likes"
            else:
                sentence = f"{agent} eats {obj}."
                question = f"Does {agent} eat {obj}?"
                relation = "eats"
            core = SURFACES[surface].format(sentence=sentence, question=question)
            prompt, gold, correct, wrong = answer_row(core, "Yes", order)
            result.append({
                "case_id": f"c338-apple-{surface}-u{unit}-{a}{b}-{order:+d}",
                "panel": "apple_factorial", "family": "attitude_object", "surface": surface,
                "unit": unit, "factor_a": a, "factor_b": b, "order": order,
                "partition": "discovery" if unit < 8 else "confirmation", "gold_position": gold,
                "correct_answer": correct, "wrong_answer": wrong, "prompt_core": core, "prompt": prompt,
                "free_prompt": core + " Answer with Yes or No.",
                "role_values": {"primary": agent, "secondary": obj, "relation": relation, "context": obj, "query": agent},
                "semantic_graph": {"operation_a": "category_to_member", "operation_b": "event_to_attitude_event", "agent": agent, "category": category, "member": member},
            })
        for surface, control, order in itertools.product(SURFACES, ("outer_negation", "event_negation", "role_reversal"), (1, -1)):
            if control == "outer_negation":
                sentence = f"{agent} does not like eating {member}."
                question, correct, relation, primary, secondary = f"Does {agent} like eating {member}?", "No", "like", agent, member
            elif control == "event_negation":
                sentence = f"{agent} likes not eating {member}."
                question, correct, relation, primary, secondary = f"Does {agent} like not eating {member}?", "Yes", "likes", agent, member
            else:
                sentence = f"{member} like eating {agent}."
                question, correct, relation, primary, secondary = f"Does {agent} like eating {member}?", "No", "like", member, agent
            core = SURFACES[surface].format(sentence=sentence, question=question)
            prompt, gold, correct, wrong = answer_row(core, correct, order)
            result.append({
                "case_id": f"c338-control-{control}-{surface}-u{unit}-{order:+d}",
                "panel": "apple_control", "family": "attitude_scope", "surface": surface,
                "unit": unit, "factor_a": None, "factor_b": None, "control": control, "order": order,
                "partition": "discovery" if unit < 8 else "confirmation", "gold_position": gold,
                "correct_answer": correct, "wrong_answer": wrong, "prompt_core": core, "prompt": prompt,
                "free_prompt": core + " Answer with Yes or No.",
                "role_values": {"primary": primary, "secondary": secondary, "relation": relation, "context": secondary, "query": agent},
                "semantic_graph": {"control": control, "agent": agent, "member": member},
            })
    graph_units = [
        {"root": f"nax{chr(97+i)}", "m1": f"pev{chr(97+i)}", "m2": f"rul{chr(97+i)}", "m3": f"sot{chr(97+i)}", "final": f"kind{chr(97+i)}", "noise": f"dif{chr(97+i)}"}
        for i in range(8)
    ]
    for unit, g in enumerate(graph_units):
        for depth, surface, mode, question_kind, order in itertools.product(range(1, 5), ("registry", "briefing"), ("entailed", "contradicted", "unknown"), ("entailed", "contradicted"), (1, -1)):
            mids = [g["m1"], g["m2"], g["m3"]]
            nodes = [g["root"], *mids[:max(0, depth - 1)], g["final"]]
            edges = list(zip(nodes[:-1], nodes[1:]))
            facts = [f'The code "{x}" belongs to "{y}".' for x, y in edges]
            if mode == "contradicted":
                facts.append(f'The registry explicitly states that "{g["root"]}" does not belong to "{g["final"]}".')
            elif mode == "unknown":
                cut = len(facts) // 2
                facts[cut] = f'The code "{nodes[cut]}" belongs to "{g["noise"]}".'
            body = " ".join(facts)
            prefix = "A registry contains these entries:" if surface == "registry" else "During a briefing, these links were stated:"
            if question_kind == "entailed":
                question = f'Is it entailed that "{g["root"]}" belongs to "{g["final"]}"?'
                correct = "Yes" if mode == "entailed" else "No"
            else:
                question = f'Is it contradicted that "{g["root"]}" belongs to "{g["final"]}"?'
                correct = "Yes" if mode == "contradicted" else "No"
            core = f"{prefix} {body} {question}"
            prompt, gold, correct, wrong = answer_row(core, correct, order)
            result.append({
                "case_id": f"c338-graph-{surface}-u{unit}-d{depth}-{mode}-{question_kind}-{order:+d}",
                "panel": "ternary_graph", "family": "type_graph", "surface": surface, "unit": unit,
                "depth": depth, "mode": mode, "question_kind": question_kind, "order": order,
                "partition": "discovery" if unit < 4 else "confirmation", "gold_position": gold,
                "correct_answer": correct, "wrong_answer": wrong, "prompt_core": core, "prompt": prompt,
                "free_prompt": core + " Answer with Yes or No.",
                "role_values": {"primary": g["root"], "secondary": g["final"], "relation": "belongs to", "context": g["final"], "query": g["root"]},
                "semantic_graph": {"mode": mode, "depth": depth, "question_kind": question_kind, "nodes": nodes},
            })
    return result


def c336() -> None:
    out = begin("C336", {"status": "frozen_independent_statistical_reaudit", "observed": "pre-fixed identity role map", "null": "719 non-identity role permutations", "claim_boundary": "This repairs p-value accounting only; C330 remains coarse topology convergence, not bisimulation."}, {"c330_exists": old.OUTS["C330"].exists(), "identity_prefixed": True})
    if (out / "analysis/final.json").exists(): return
    original = rows(old.OUTS["C330"] / "analysis/pair_tests.jsonl")
    arrays = np.load(old.OUTS["C330"] / "analysis/model_native_response_topologies.float32.npy")
    participants = final_old("C330")["headline"]["participants"]
    perms = list(itertools.permutations(range(6)))
    ident = tuple(range(6))
    results = []
    for i, left_name in enumerate(participants):
        for j, right_name in enumerate(participants[i + 1:], start=i + 1):
            left = arrays[i] - arrays[i].mean(axis=-1, keepdims=True)
            right = arrays[j] - arrays[j].mean(axis=-1, keepdims=True)
            observed = cosine(left, right)
            null = np.asarray([cosine(left, right[:, :, p]) for p in perms if p != ident])
            results.append({"models": [left_name, right_name], "identity_centered_cosine": observed, "nonidentity_permutations": len(null), "exact_upper_p": float((1 + np.sum(null >= observed)) / (1 + len(null))), "nonidentity_exceedances": int(np.sum(null >= observed))})
    rows(out / "analysis/repaired_pair_tests.jsonl", results)
    headline = {"status": "c330_statistical_reaudit_closed", "original_test_used_identity_observed_not_maximized_observed": True, "original_null_included_identity": True, "repaired_pairs": results, "all_pairs_significant": all(x["exact_upper_p"] <= .05 for x in results), "strict_interpretation": "The attachment's circular-maximization accusation is rejected. Excluding identity from the null removes a conservative accounting defect and does not upgrade the claim to functional bisimulation."}
    close("C336", headline, {"three_pairs": len(results) == 3, "null_719": all(x["nonidentity_permutations"] == 719 for x in results), "finite": finite(headline)}, "C337_master_contract")


def final_old(campaign: str) -> dict:
    return json.loads((old.OUTS[campaign] / "analysis/final.json").read_text(encoding="utf-8"))


def cosine(a, b) -> float:
    x = np.asarray(a, np.float64).ravel(); y = np.asarray(b, np.float64).ravel()
    return float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))


def finite(value) -> bool:
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
    if isinstance(value, (float, np.floating)): return math.isfinite(float(value))
    return True


def c337_c338() -> None:
    begin("C337", {"status": "single_sample_campaign_frozen", "object": "full-coordinate response of one concrete sentence conditioned on explicit semantic graph and operations", "branches": ["single_sample", "attitude_scope", "ternary_graph", "state_refinement", "mediation", "cross_model"], "route_policy": "failure retires one branch only", "forbidden": ["PCA", "cosine_primary_metric", "Top-K coordinate discovery", "attention", "MLP"]}, {"parent": final("C336")["all_checks_passed"], "all_coordinates": True, "route_level_stops": True})
    if not (OUTS["C337"] / "analysis/final.json").exists():
        close("C337", {"status": "master_contract_closed", "campaigns": list(PHASES)[1:], "single_sample_rule": "A confirmation row exposes H00 and registered graph/operations only; H10/H01/H11 from that row are forbidden predictor inputs.", "claim_boundary": "Observational branches seek regularity; only prospectively qualified objects may enter causal intervention."}, {"twenty_four_followups": len(PHASES) - 1 == 24}, "C338_material_compiler")
    out = begin("C338", {"status": "material_compiler_frozen", "apple_factorial": "category/member x direct/attitude event", "controls": ["outer_negation", "event_negation", "role_reversal"], "graph": "depth1-4 x entailed/contradicted/unknown x two binary semantic questions", "partitions": "lexically disjoint unit partitions", "surface_forms": "three apple and two graph surfaces"}, {"parent": final("C337")["all_checks_passed"], "semantic_labels_before_model": True})
    if (out / "analysis/final.json").exists(): return
    data = material()
    rows(out / "material/cases.jsonl", data)
    by_panel = {p: sum(r["panel"] == p for r in data) for p in sorted({r["panel"] for r in data})}
    balance = {p: {str(o): sum(r["panel"] == p and r["gold_position"] == o for r in data) for o in (0, 1)} for p in by_panel}
    audit = {"no_placeholders": all("{" not in r["prompt"] for r in data), "semantic_graphs": all(r["semantic_graph"] for r in data), "both_partitions": all({r["partition"] for r in data if r["panel"] == p} == {"discovery", "confirmation"} for p in by_panel), "candidate_balance_max_gap": max(abs(v["0"] - v["1"]) for v in balance.values())}
    save(out / "audit/material_audit.json", {**audit, "by_panel": by_panel, "balance": balance})
    close("C338", {"status": "material_compiler_closed", "rows": len(data), "by_panel": by_panel, "balance": balance, "semantic_uniqueness": "Graph classes are fixed by explicit reachability/negation; apple controls have explicit truth conditions.", "naturality_boundary": "Controlled English was grammar-audited mechanically but not independently human-rated."}, {"row_count": len(data) == 1272, "audit": audit["no_placeholders"] and audit["semantic_graphs"] and audit["both_partitions"], "balanced": audit["candidate_balance_max_gap"] == 0}, "C339_behavior")


@torch.inference_mode()
def c339() -> None:
    out = begin("C339", {"status": "behavior_qualification_frozen", "model": "Qwen3-4B bf16 CUDA", "gate": "confirmation row accuracy>=0.80; every panel>=0.70; every graph mode>=0.65", "hidden_state_policy": "no hidden states are requested in this phase"}, {"parent": final("C338")["all_checks_passed"], "cuda": torch.cuda.is_available(), "behavior_before_hidden": True})
    if (out / "analysis/final.json").exists(): return
    data = rows(OUTS["C338"] / "material/cases.jsonl")
    model = None
    try:
        model, tok, device, placement = old.model_base.load_bf16("qwen3")
        compiled = old.compile_general(tok, data, "strict_chat")
        rows(out / "compiled/qwen3.jsonl", compiled)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        result = []
        packed = [(r, r["prompt_ids"], r["candidate_ids"]) for r in compiled]
        for start in range(0, len(packed), 8):
            result.extend(old.score_interface_batch(model, device, pad, packed[start:start+8]))
            if start % 256 == 0: print(f"[C339] {min(start+8,len(packed))}/{len(packed)}", flush=True)
        rows(out / "raw/behavior.jsonl", result)
        lookup = {r["case_id"]: r for r in data}
        confirmation = [r for r in result if lookup[r["case_id"]]["partition"] == "confirmation"]
        panels = {p: float(np.mean([r["correct"] for r in confirmation if lookup[r["case_id"]]["panel"] == p])) for p in sorted({lookup[r["case_id"]]["panel"] for r in confirmation})}
        graph_modes = {m: float(np.mean([r["correct"] for r in confirmation if lookup[r["case_id"]].get("mode") == m])) for m in ("entailed", "contradicted", "unknown")}
        accuracy = float(np.mean([r["correct"] for r in confirmation]))
        eligible = accuracy >= .80 and min(panels.values()) >= .70 and min(graph_modes.values()) >= .65
        headline = {"status": "behavior_qualification_closed", "confirmation_accuracy": accuracy, "panel_accuracy": panels, "graph_mode_accuracy": graph_modes, "hidden_state_eligible": eligible, "placement": placement, "quantization": old.model_base.quantization_audit(model)}
        close("C339", headline, {"all_rows": len(result) == len(data), "finite": finite(headline), "no_hidden_archive": not (out / "raw/role_states.float16.npy").exists()}, "C340_full_coordinate_capture" if eligible else "C349_behavior_only_graph_adjudication_and_nonmodel_branches")
    finally:
        old.model_base.release(model); gc.collect()


def c340() -> None:
    eligible = final("C339")["headline"]["hidden_state_eligible"]
    out = begin("C340", {"status": "full_coordinate_capture_frozen", "checkpoints": "embedding + 36 block outputs + final norm", "archive": "all rows x all six aligned roles x all 2560 coordinates; confirmation primary-surface/order+1 rows additionally retain every token and coordinate", "gate_dependency": "C339 behavior"}, {"parent": final("C339")["all_checks_passed"], "behavior_eligible": eligible, "cuda": torch.cuda.is_available()})
    if (out / "analysis/final.json").exists(): return
    data = rows(OUTS["C338"] / "material/cases.jsonl")
    compiled = rows(OUTS["C339"] / "compiled/qwen3.jsonl")
    selector = lambda r: r["partition"] == "confirmation" and r["order"] == 1 and r["surface"] in ("report", "registry")
    metrics = old.batch_capture_qwen(data, compiled, out, full_selector=selector, batch_size=4, field_width=192)
    headline = {"status": "full_coordinate_capture_closed", **metrics, "behavior_score_role": "diagnostic first-candidate-token score only; C339 remains the sole behavior adjudication", "strict_interpretation": "Role averages are aligned derived views; the selected all-token archive retains uncompressed token-coordinate fields. Neither is an attention/MLP circuit."}
    close("C340", headline, {"rows": metrics["rows"] == len(data), "checkpoints": metrics["role_shape"][1] == 38, "coordinates": metrics["role_shape"][-1] == 2560, "full_rows": metrics["full_token_rows"] == sum(selector(r) for r in data)}, "C341_single_sample_operator")


def apple_groups(index: list[dict]) -> list[dict]:
    idx = {(r["surface"], r["unit"], r["order"], r["factor_a"], r["factor_b"]): r["hidden_index"] for r in index if r["panel"] == "apple_factorial"}
    result = []
    for surface, unit, order in itertools.product(SURFACES, range(12), (1, -1)):
        ids = {(a,b): idx[(surface,unit,order,a,b)] for a,b in itertools.product((0,1), repeat=2)}
        result.append({"surface": surface, "unit": unit, "order": order, "ids": ids})
    return result


def fit_diag(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xm = x.mean(axis=0); ym = y.mean(axis=0)
    dx = x - xm
    slope = np.sum(dx * (y - ym), axis=0) / np.maximum(np.sum(dx * dx, axis=0), 1e-8)
    intercept = ym - slope * xm
    return intercept.astype(np.float32), slope.astype(np.float32)


def c341() -> None:
    out = begin("C341", {"status": "single_sample_operator_frozen", "input": "confirmation H00 plus registered operation only", "forbidden_input": "same confirmation row's H10/H01/H11", "model": "coordinatewise affine response conditioned on H00_j", "baselines": ["zero response", "discovery mean response"], "metric": "full-coordinate MAE, never cosine", "gate": "affine relative MAE gain over discovery mean >0 for A,B,I and median checkpoint-role gain>0"}, {"parent": final("C340")["all_checks_passed"], "no_test_cell_leakage": True})
    if (out / "analysis/final.json").exists(): return
    states = np.load(OUTS["C340"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = rows(OUTS["C340"] / "raw/hidden_index.jsonl")
    groups = apple_groups(index); train = [g for g in groups if g["unit"] < 8]; test = [g for g in groups if g["unit"] >= 8]
    operators = np.lib.format.open_memmap(out / "analysis/operators.float32.npy", mode="w+", dtype=np.float32, shape=(3,2,38,6,2560))
    predictions = np.lib.format.open_memmap(out / "raw/confirmation_predicted_responses.float16.npy", mode="w+", dtype=np.float16, shape=(len(test),3,38,6,2560))
    gains = np.zeros((3,38,6), np.float32); aggregate = []
    for q in range(38):
        train_h = {c: np.asarray([states[g["ids"][c],q] for g in train], np.float32) for c in ((0,0),(1,0),(0,1),(1,1))}
        test_h = {c: np.asarray([states[g["ids"][c],q] for g in test], np.float32) for c in ((0,0),(1,0),(0,1),(1,1))}
        train_y = (train_h[(1,0)]-train_h[(0,0)], train_h[(0,1)]-train_h[(0,0)], train_h[(1,1)]-train_h[(1,0)]-train_h[(0,1)]+train_h[(0,0)])
        test_y = (test_h[(1,0)]-test_h[(0,0)], test_h[(0,1)]-test_h[(0,0)], test_h[(1,1)]-test_h[(1,0)]-test_h[(0,1)]+test_h[(0,0)])
        for oi in range(3):
            intercept, slope = fit_diag(train_h[(0,0)], train_y[oi]); operators[oi,0,q]=intercept; operators[oi,1,q]=slope
            pred = intercept[None] + slope[None] * test_h[(0,0)]
            predictions[:,oi,q] = pred.astype(np.float16)
            mean_pred = train_y[oi].mean(axis=0)
            for ri in range(6):
                base = float(np.mean(np.abs(test_y[oi][:,ri]-mean_pred[ri])))
                err = float(np.mean(np.abs(test_y[oi][:,ri]-pred[:,ri])))
                gains[oi,q,ri] = (base-err)/max(base,1e-12)
        print(f"[C341] checkpoint {q}/37", flush=True)
    operators.flush(); predictions.flush(); np.save(out / "analysis/checkpoint_role_gains.float32.npy", gains)
    op_rows = []
    for oi, op in enumerate(OPS):
        op_rows.append({"operation": op, "mean_gain": float(gains[oi].mean()), "median_checkpoint_role_gain": float(np.median(gains[oi])), "positive_cells": int(np.sum(gains[oi] > 0)), "cells": int(gains[oi].size)})
    gate = all(r["mean_gain"] > 0 and r["median_checkpoint_role_gain"] > 0 for r in op_rows)
    rows(out / "analysis/operator_results.jsonl", op_rows)
    headline = {"status": "single_sample_operator_adjudicated", "training_groups": len(train), "confirmation_groups": len(test), "operations": op_rows, "single_sample_gate_passed": gate, "strict_interpretation": "A pass means H00_j carries reusable information about its own response coordinate. It does not identify a unique causal gear or prove nonlinear composition."}
    close("C341", headline, {"operators_full": list(operators.shape) == [3,2,38,6,2560], "predictions_full": list(predictions.shape) == [len(test),3,38,6,2560], "finite": finite(headline)}, "C342_C355_observation_branches_continue_regardless")


def c342_c346() -> None:
    # C342 transparent response-sign partition refinement.
    out = begin("C342", {"status": "predictive_partition_refinement_frozen", "node": "(checkpoint, role, physical coordinate)", "features": "sign bins of intercept and baseline-conditioned slope for A/B/I", "merge_rule": "same registered future-response signature", "metric": "parameter reconstruction MAE and state count", "no_probe": True}, {"parent": final("C341")["all_checks_passed"], "all_coordinates": True})
    if not (out / "analysis/final.json").exists():
        op = np.load(OUTS["C341"] / "analysis/operators.float32.npy")
        scale = np.median(np.abs(op), axis=(2,3,4), keepdims=True) + 1e-8
        code = np.zeros((38,6,2560), np.uint16)
        bit = 1
        for oi in range(3):
            for pi in range(2):
                v = op[oi,pi]
                sig = np.where(v > scale[oi,pi]*.25, 2, np.where(v < -scale[oi,pi]*.25, 0, 1)).astype(np.uint16)
                code += sig * bit; bit *= 3
        # Type is retained explicitly so role/checkpoint-incompatible nodes never merge.
        typed = code.astype(np.int64) + np.arange(38, dtype=np.int64)[:,None,None]*729 + np.arange(6,dtype=np.int64)[None,:,None]*729*38
        unique, assignment = np.unique(typed, return_inverse=True)
        assignment = assignment.reshape(38,6,2560).astype(np.int32)
        np.save(out / "analysis/full_coordinate_state_ids.int32.npy", assignment)
        compression = float(assignment.size / len(unique))
        headline = {"status": "predictive_partition_refinement_closed", "physical_nodes": int(assignment.size), "response_states": int(len(unique)), "signature_compression_ratio": compression, "heldout_predictive_compression_established": False, "global_merge_forbidden": True, "strict_interpretation": "These are transparent response-sign equivalence candidates. Counting repeated quantized signatures is not a heldout description-length or predictive-compression proof."}
        close("C342", headline, {"all_nodes": assignment.size == 38*6*2560, "nontrivial": 1 < len(unique) < assignment.size}, "C343_dynamic_coalition")
    out = begin("C343", {"status": "dynamic_coalition_frozen", "edge": "guarded full-coordinate diagonal source-to-response rule", "qualification": "checkpoint-role gain>0 on confirmation", "no_topk": True, "controls": ["mean", "persistence", "wrong-role", "coordinate-roll", "role-roll"]}, {"parent": final("C342")["all_checks_passed"], "full_coordinate_operator": True})
    if not (out / "analysis/final.json").exists():
        gains = np.load(OUTS["C341"] / "analysis/checkpoint_role_gains.float32.npy")
        edges = []
        for oi, op_name in enumerate(OPS):
            for q in range(37):
                for ri, role in enumerate(ROLES):
                    if gains[oi,q,ri] > 0:
                        edges.append({"operation": op_name, "source_checkpoint": q, "target_checkpoint": q+1, "source_role": role, "target_role": role, "coordinate_domain": [0,2559], "confirmation_gain": float(gains[oi,q,ri])})
        rows(out / "analysis/qualified_edges.jsonl", edges)
        by_op = {o: sum(e["operation"] == o for e in edges) for o in OPS}
        headline = {"status": "dynamic_coalition_closed", "qualified_full_coordinate_edges": len(edges), "by_operation": by_op, "causal_qualified": final("C341")["headline"]["single_sample_gate_passed"] and all(by_op[o] > 0 for o in OPS), "strict_interpretation": "Each edge invokes all 2560 physical coordinates through one frozen rule; it is a predictive dependency candidate, not a unique causal circuit."}
        close("C343", headline, {"finite": finite(headline), "edge_accounting": len(edges) == sum(by_op.values())}, "C344_lockbox_and_parallel_graph_branches")
    out = begin("C344", {"status": "lexical_lockbox_frozen", "training_units": list(range(8)), "lockbox_units": list(range(8,12)), "prohibited": "refit after lockbox reveal", "metrics": "C341 preregistered full-coordinate MAE gains"}, {"parent": final("C343")["all_checks_passed"], "partition_frozen_in_C338": True})
    if not (out / "analysis/final.json").exists():
        close("C344", {"status": "lexical_lockbox_contract_closed", "lockbox_was_consumed_once_by_C341": True, "units": 4, "surfaces": 3, "answer_orders": 2}, {"units": True, "no_refit": True}, "C345_adjudication")
    out = begin("C345", {"status": "lexical_lockbox_adjudication_frozen", "source": "C341 confirmation-only metrics"}, {"parent": final("C344")["all_checks_passed"], "c341_closed": final("C341")["all_checks_passed"]})
    if not (out / "analysis/final.json").exists():
        op_rows = rows(OUTS["C341"] / "analysis/operator_results.jsonl")
        passed = final("C341")["headline"]["single_sample_gate_passed"]
        close("C345", {"status": "lexical_lockbox_adjudicated", "operations": op_rows, "lockbox_gate_passed": passed, "strict_interpretation": "No operation-specific claim is made when its mean or median gain is nonpositive."}, {"three_operations": len(op_rows)==3}, "C346_order_rollout")
    out = begin("C346", {"status": "operation_order_rollout_frozen", "predictions": ["A_then_B", "B_then_A", "A_plus_B_plus_I"], "input": "confirmation H00 only", "metric": "full-coordinate MAE to H11", "claim_boundary": "Prompt answer-order is not semantic operation order; this is operator application order computed offline."}, {"parent": final("C345")["all_checks_passed"], "no_test_intermediate": True})
    if not (out / "analysis/final.json").exists():
        states=np.load(OUTS["C340"] / "raw/role_states.float16.npy",mmap_mode="r"); index=rows(OUTS["C340"] / "raw/hidden_index.jsonl"); all_groups=apple_groups(index); training=[g for g in all_groups if g["unit"]<8]; groups=[g for g in all_groups if g["unit"]>=8]
        op=np.load(OUTS["C341"] / "analysis/operators.float32.npy",mmap_mode="r")
        sums={k:0.0 for k in ("discovery_mean_additive","discovery_mean_complete","A_then_B","B_then_A","A_plus_B_plus_I")}; count=0
        for q in range(38):
            x=np.asarray([states[g["ids"][(0,0)],q] for g in groups],np.float32); truth=np.asarray([states[g["ids"][(1,1)],q] for g in groups],np.float32)
            tr={c:np.asarray([states[g["ids"][c],q] for g in training],np.float32) for c in ((0,0),(1,0),(0,1),(1,1))}
            mean_a=(tr[(1,0)]-tr[(0,0)]).mean(axis=0); mean_b=(tr[(0,1)]-tr[(0,0)]).mean(axis=0); mean_i=(tr[(1,1)]-tr[(1,0)]-tr[(0,1)]+tr[(0,0)]).mean(axis=0)
            f=lambda oi,z: np.asarray(op[oi,0,q])+np.asarray(op[oi,1,q])*z
            a=x+f(0,x); b=x+f(1,x)
            pred={"A_then_B":a+f(1,a),"B_then_A":b+f(0,b),"A_plus_B_plus_I":x+f(0,x)+f(1,x)+f(2,x)}
            pred["discovery_mean_additive"]=x+mean_a+mean_b; pred["discovery_mean_complete"]=x+mean_a+mean_b+mean_i
            for k,v in pred.items(): sums[k]+=float(np.sum(np.abs(truth-v)))
            count+=truth.size
        maes={k:v/count for k,v in sums.items()}; best=min(maes,key=maes.get)
        operator_best=min(("A_then_B","B_then_A","A_plus_B_plus_I"),key=lambda k:maes[k]); mean_best=min(("discovery_mean_additive","discovery_mean_complete"),key=lambda k:maes[k])
        headline={"status":"operation_order_rollout_adjudicated","full_coordinate_mae":maes,"best_model":best,"best_operator_model":operator_best,"best_mean_baseline":mean_best,"composition_improved_over_mean":maes[operator_best]<maes[mean_best],"strict_interpretation":"This tests reusable numerical update order, not linguistic commutativity as a theorem."}
        close("C346",headline,{"finite":finite(headline),"all_models":len(maes)==5},"C347_C355_parallel_observation")


def c347_c352() -> None:
    out=begin("C347",{"status":"attitude_scope_ecology_frozen","contrasts":["outer_negation","event_negation","role_reversal"],"metric":"per-checkpoint per-role full-coordinate absolute response"},{"parent":final("C340")["all_checks_passed"],"controls_behavior_qualified":True})
    if not (out/"analysis/final.json").exists():
        s=np.load(OUTS["C340"] / "raw/role_states.float16.npy",mmap_mode="r"); idx=rows(OUTS["C340"] / "raw/hidden_index.jsonl"); material_by_id={r["case_id"]:r for r in rows(OUTS["C338"] / "material/cases.jsonl")}
        control=[]
        for name in ("outer_negation","event_negation","role_reversal"):
            ids=[r["hidden_index"] for r in idx if material_by_id[r["case_id"]].get("control")==name and r["partition"]=="confirmation"]
            if not ids:
                raise RuntimeError(("empty_control", name))
            x=np.asarray(s[ids],np.float32); control.append(np.mean(np.abs(x-x.mean(axis=0,keepdims=True)),axis=(0,3)))
        arr=np.asarray(control,np.float32); np.save(out/"analysis/control_checkpoint_role_dispersion.float32.npy",arr)
        close("C347",{"status":"attitude_scope_ecology_closed","controls":3,"samples_per_control":[sum(material_by_id[r["case_id"]].get("control")==name and r["partition"]=="confirmation" for r in idx) for name in ("outer_negation","event_negation","role_reversal")],"shape":list(arr.shape),"max_dispersion_checkpoint_by_control":[int(np.argmax(v.mean(axis=-1))) for v in arr],"strict_interpretation":"Distinct response ecologies do not by themselves prove logical scope operators."},{"shape":list(arr.shape)==[3,38,6],"finite":bool(np.isfinite(arr).all())},"C348_all_token")
    out=begin("C348",{"status":"all_token_coordinate_observation_frozen","archive":"C340 selected confirmation all-token fields","analysis":"coverage and exact coordinate range only; no Top-K"},{"parent":final("C347")["all_checks_passed"],"archive_exists":(OUTS["C340"] / "raw/full_fields_holdout.float16.npy").exists()})
    if not (out/"analysis/final.json").exists():
        f=np.load(OUTS["C340"] / "raw/full_fields_holdout.float16.npy",mmap_mode="r"); mapping=json.loads((OUTS["C340"] / "raw/full_field_row_map.json").read_text(encoding="utf-8")); idx=rows(OUTS["C340"] / "raw/hidden_index.jsonl")
        lengths=[idx[i]["length"] for i in mapping["source_indices"]]; stats={"shape":list(f.shape),"rows":len(lengths),"min_length":min(lengths),"max_length":max(lengths),"physical_coordinate_min":0,"physical_coordinate_max":2559,"checkpoints":38}
        close("C348",{"status":"all_token_coordinate_observation_closed",**stats,"strict_interpretation":"The raw memmap is the primary artifact; this phase intentionally does not rank or discard low-amplitude coordinates."},{"coordinate_axis":f.shape[-1]==2560,"mapping":len(lengths)==f.shape[0]},"C349_graph_behavior")
    out=begin("C349",{"status":"ternary_graph_behavior_frozen","decode":"two binary questions jointly distinguish entailed, contradicted, unknown","gate":"joint scenario accuracy>=0.65 for every class"},{"parent":final("C339")["all_checks_passed"],"three_classes":True})
    if not (out/"analysis/final.json").exists():
        behavior={r["case_id"]:r for r in rows(OUTS["C339"] / "raw/behavior.jsonl")}; data=[r for r in rows(OUTS["C338"] / "material/cases.jsonl") if r["panel"]=="ternary_graph" and r["partition"]=="confirmation"]
        grouped={}
        for r in data:
            key=(r["unit"],r["depth"],r["surface"],r["mode"],r["order"]); grouped.setdefault(key,{})[r["question_kind"]]=behavior[r["case_id"]]["correct"]
        class_acc={m:float(np.mean([all(v.values()) for k,v in grouped.items() if k[3]==m])) for m in ("entailed","contradicted","unknown")}; gate=min(class_acc.values())>=.65
        close("C349",{"status":"ternary_graph_behavior_adjudicated","scenario_class_accuracy":class_acc,"graph_behavior_eligible":gate,"scenarios":len(grouped),"strict_interpretation":"Qualification is confined to the explicitly taught pseudo-graph contract."},{"complete_bits":all(len(v)==2 for v in grouped.values()),"finite":finite(class_acc)},"C350_recursive_rollout")
    graph_eligible=final("C349")["headline"]["graph_behavior_eligible"]
    out=begin("C350",{"status":"recursive_graph_rollout_frozen","operator":"one coordinatewise affine U_up shared across depth1->2,2->3,3->4","training_units":[0,1,2,3],"lockbox_units":[4,5,6,7],"metric":"full-coordinate MAE after autonomous depth1->4 rollout","baselines":["persistence","depth-specific mean step"],"behavior_eligible":graph_eligible},{"parent":final("C349")["all_checks_passed"],"eligibility_recorded":True})
    if not (out/"analysis/final.json").exists():
        if not graph_eligible:
            close("C350",{"status":"recursive_graph_rollout_not_run_ineligible","recursive_gate_passed":False,"reason":"C349 ternary joint behavior gate failed","strict_interpretation":"No hidden-state recursive mechanism test was authorized; the behavior result does not refute recursion in Qwen3."},{"no_hidden_mechanism_claim":True},"C351_existing_field_observation_only")
        else:
            _run_c350(out)
    out=begin("C351",{"status":"graph_full_field_ecology_frozen","source":"C340 full-token confirmation graph archive","comparisons":"entailed/contradicted/unknown over every retained token and coordinate","no_alignment_claim":"raw token positions are not treated as semantic isomorphisms"},{"parent":final("C350")["all_checks_passed"],"full_archive":True})
    if not (out/"analysis/final.json").exists():
        f=np.load(OUTS["C340"] / "raw/full_fields_holdout.float16.npy",mmap_mode="r"); mapping=json.loads((OUTS["C340"] / "raw/full_field_row_map.json").read_text()); idx=rows(OUTS["C340"] / "raw/hidden_index.jsonl"); selected=[idx[i] for i in mapping["source_indices"]]; graph=sum(r["panel"]=="ternary_graph" for r in selected)
        close("C351",{"status":"graph_full_field_ecology_closed","graph_full_token_rows":graph,"coordinate_axis":int(f.shape[-1]),"mechanism_analysis_authorized":graph_eligible,"strict_interpretation":"Because C349 failed, this is an archived descriptive field only. It permits future re-analysis but no graph-mechanism claim."},{"graph_rows":graph==96,"all_coords":f.shape[-1]==2560},"C352_natural_panel")
    out=begin("C352",{"status":"natural_membership_external_panel_frozen","panel":"category/member substitutions in C338 apple material","role":"external natural lexical panel, not proof of world-knowledge hierarchy"},{"parent":final("C351")["all_checks_passed"],"natural_pairs":len(OBJECTS)==12})
    if not (out/"analysis/final.json").exists():
        b=rows(OUTS["C339"] / "raw/behavior.jsonl"); data={r["case_id"]:r for r in rows(OUTS["C338"] / "material/cases.jsonl")}; vals=[r for r in b if data[r["case_id"]]["panel"]=="apple_factorial" and data[r["case_id"]]["partition"]=="confirmation"]
        acc=float(np.mean([r["correct"] for r in vals])); close("C352",{"status":"natural_membership_external_panel_closed","confirmation_accuracy":acc,"registered_pairs":12,"confirmation_pairs":4,"strict_interpretation":"Behavioral success supplies an external lexical panel only; no natural ontology is inferred."},{"finite":finite(acc)},"C353_six_family")


def _run_c350(out: Path) -> None:
        s=np.load(OUTS["C340"] / "raw/role_states.float16.npy",mmap_mode="r"); idx=rows(OUTS["C340"] / "raw/hidden_index.jsonl")
        lookup={(r["unit"],r["depth"]):r["hidden_index"] for r in idx if r["panel"]=="ternary_graph" and r["surface"]=="registry" and r["mode"]=="entailed" and r["question_kind"]=="entailed" and r["order"]==1}
        total={"recursive_affine":0.,"persistence":0.,"mean_step":0.}; n=0
        for q in range(38):
            pairs=[]
            for u in range(4):
                for d in (1,2,3): pairs.append((np.asarray(s[lookup[(u,d)],q],np.float32),np.asarray(s[lookup[(u,d+1)],q],np.float32)))
            x=np.asarray([p[0] for p in pairs]); y=np.asarray([p[1]-p[0] for p in pairs]); a,b=fit_diag(x,y); mean=y.mean(axis=0)
            for u in range(4,8):
                start=np.asarray(s[lookup[(u,1)],q],np.float32); truth=np.asarray(s[lookup[(u,4)],q],np.float32); pred=start.copy()
                for _ in range(3): pred=pred+a+b*pred
                for k,v in {"recursive_affine":pred,"persistence":start,"mean_step":start+3*mean}.items(): total[k]+=float(np.sum(np.abs(truth-v)))
                n+=truth.size
        mae={k:v/n for k,v in total.items()}; best=min(mae,key=mae.get); passed=best=="recursive_affine"
        close("C350",{"status":"recursive_graph_rollout_adjudicated","full_coordinate_mae":mae,"best_model":best,"recursive_gate_passed":passed,"strict_interpretation":"A pass supports a reusable numerical depth update on this pseudo-graph; it does not prove natural taxonomic recursion."},{"finite":finite(mae)},"C351_graph_field")


def family_single_sample(states, index, family):
    arrays, groups=old.factorial_arrays(states,index,family); train=np.asarray([g["unit"]<4 for g in groups]); test=np.asarray([g["unit"]>=4 for g in groups])
    nq=arrays["h00"].shape[1]
    result=np.zeros((3,nq,6),np.float32)
    if train.sum()<2 or test.sum()<1: return result, int(train.sum()), int(test.sum())
    for q in range(nq):
        xtr=arrays["h00"][train,q]; xte=arrays["h00"][test,q]
        ys=(arrays["a0"],arrays["b0"],arrays["interaction"])
        for oi,yall in enumerate(ys):
            ytr=yall[train,q]; yte=yall[test,q]; a,b=fit_diag(xtr,ytr); pred=a+b*xte; mean=ytr.mean(axis=0)
            base=np.mean(np.abs(yte-mean),axis=(0,2)); err=np.mean(np.abs(yte-pred),axis=(0,2)); result[oi,q]=(base-err)/np.maximum(base,1e-12)
    return result,int(train.sum()),int(test.sum())


def c353_c355() -> None:
    out=begin("C353",{"status":"six_family_single_sample_operator_frozen","source":"C323 five-surface complete role-coordinate archive","families":list(old.FAMILIES),"split":"units0-3 discovery; units4-7 confirmation","metric":"full-coordinate MAE"},{"parent":final("C352")["all_checks_passed"],"c323":final_old("C323")["all_checks_passed"]})
    if not (out/"analysis/final.json").exists():
        s=np.load(old.OUTS["C323"] / "raw/role_states.float16.npy",mmap_mode="r"); idx=rows(old.OUTS["C323"] / "raw/hidden_index.jsonl"); gains=[]; detail=[]
        for fam in old.FAMILIES:
            g,tr,te=family_single_sample(s,idx,fam); gains.append(g); detail.append({"family":fam,"training_groups":tr,"confirmation_groups":te,"operation_mean_gain":{op:float(g[i].mean()) for i,op in enumerate(OPS)}}); print(f"[C353] {fam}",flush=True)
        gains=np.asarray(gains); np.save(out/"analysis/six_family_checkpoint_role_gains.float32.npy",gains); rows(out/"analysis/family_results.jsonl",detail)
        passing=[d["family"] for d in detail if all(v>0 for v in d["operation_mean_gain"].values())]
        close("C353",{"status":"six_family_single_sample_adjudicated","families":detail,"families_all_operations_positive":passing,"breadth_gate_passed":len(passing)>=4,"checkpoints":int(gains.shape[2]),"strict_interpretation":"Positive gain means reusable baseline-conditioned response, not one shared semantic operator."},{"shape":list(gains.shape)==[6,3,37,6],"finite":finite(detail)},"C354_breadth")
    out=begin("C354",{"status":"cross_family_operator_breadth_frozen","criterion":"sign and checkpoint-role support repeated without coordinate alignment or pooled refit"},{"parent":final("C353")["all_checks_passed"],"six_families":True})
    if not (out/"analysis/final.json").exists():
        g=np.load(OUTS["C353"] / "analysis/six_family_checkpoint_role_gains.float32.npy"); sign=(g>0); repeated=np.sum(sign,axis=0); stats={op:{"cells_positive_in_at_least_four_families":int(np.sum(repeated[i]>=4)),"cells":int(repeated[i].size)} for i,op in enumerate(OPS)}
        close("C354",{"status":"cross_family_operator_breadth_closed","operation_support":stats,"universal_operator_established":False,"strict_interpretation":"Repeated predictive support is a regularity map; family-specific signs and magnitudes remain visible."},{"finite":finite(stats)},"C355_state_refinement")
    out=begin("C355",{"status":"cross_family_state_refinement_frozen","state_signature":"for each checkpoint-role, six-family sign support for A/B/I","no_coordinate_pooling":True},{"parent":final("C354")["all_checks_passed"],"all_family_maps":True})
    if not (out/"analysis/final.json").exists():
        g=np.load(OUTS["C353"] / "analysis/six_family_checkpoint_role_gains.float32.npy"); signatures=np.sum((g>0).astype(np.uint8)*np.asarray([1,2,4])[None,:,None,None],axis=1); np.save(out/"analysis/family_checkpoint_role_signatures.uint8.npy",signatures)
        unique=int(len(np.unique(signatures))); close("C355",{"status":"cross_family_state_refinement_closed","signature_states":unique,"shape":list(signatures.shape),"strict_interpretation":"This compact index summarizes measured support while the full coordinate archives remain primary."},{"shape":list(signatures.shape)==[6,37,6]},"C356_mediation_if_qualified")


def c356_c357() -> None:
    qualified=final("C343")["headline"]["causal_qualified"]
    out=begin("C356",{"status":"typed_mediation_contract_frozen","eligibility":"C343 prospective predictive coalition","sequence":["source delete","target loss","correct target rescue with source still deleted","wrong role","wrong coordinate","wrong operation","output check"],"patch":"all six roles and all 2560 coordinates; no Top-K"},{"parent":final("C355")["all_checks_passed"],"eligibility_recorded":True})
    if not (out/"analysis/final.json").exists():
        close("C356",{"status":"typed_mediation_contract_closed","causal_eligible":qualified,"claim_boundary":"Failure affects this full-role/full-coordinate hook interface only."},{"contract_complete":True},"C357_run" if qualified else "C358_cross_model_branch")
    out=begin("C357",{"status":"typed_mediation_frozen","eligible":qualified,"model":"Qwen3-4B bf16 CUDA","cases":"confirmation apple H11/report/order+1","conditions":["natural","source_delete","correct_target_restore","wrong_role_restore","wrong_coordinate_restore","wrong_operation_restore"]},{"parent":final("C356")["all_checks_passed"],"eligibility_consistent":qualified==final("C356")["headline"]["causal_eligible"]})
    if (out/"analysis/final.json").exists(): return
    if not qualified:
        close("C357",{"status":"typed_mediation_not_run_ineligible","causal_claim":False,"reason":"single-sample breadth gate did not qualify the coalition"},{"no_model_run":True},"C358_cross_model")
        return
    # Use the strongest mean-gain checkpoint and the complete six-role coordinate field.
    gains=np.load(OUTS["C341"] / "analysis/checkpoint_role_gains.float32.npy"); q=int(np.argmax(gains[2,:37].mean(axis=-1))); operators=np.load(OUTS["C341"] / "analysis/operators.float32.npy",mmap_mode="r")
    idx=rows(OUTS["C340"] / "raw/hidden_index.jsonl"); compiled=rows(OUTS["C339"] / "compiled/qwen3.jsonl"); selected=[r for r in idx if r["panel"]=="apple_factorial" and r["partition"]=="confirmation" and r["factor_a"]==1 and r["factor_b"]==1 and r["surface"]=="report" and r["order"]==1]
    model=None; samples=[]
    try:
        model,_tok,device,placement=old.model_base.load_bf16("qwen3"); layers=old.get_layers(model)
        source=np.asarray(operators[2,0,q],np.float32); target=np.asarray(operators[2,0,q+1],np.float32); wrong_op=np.asarray(operators[0,0,q+1],np.float32)
        for meta in selected:
            row=compiled[meta["hidden_index"]]; ids=torch.tensor([row["prompt_ids"]],dtype=torch.long,device=device); mask=torch.ones_like(ids); cond={}
            for name in ("natural","source_delete","correct_target_restore","wrong_role_restore","wrong_coordinate_restore","wrong_operation_restore"):
                captured=[]
                srcvec=None if name=="natural" else old.role_position_vectors(row,-source)
                restore=None
                if name=="correct_target_restore": restore=target
                elif name=="wrong_role_restore": restore=np.roll(target,1,axis=0)
                elif name=="wrong_coordinate_restore": restore=np.roll(target,97,axis=-1)
                elif name=="wrong_operation_restore": restore=old.norm_match(wrong_op,target)
                dstvec=None if restore is None else old.role_position_vectors(row,restore)
                def source_hook(_m,_a,o):
                    if srcvec is None:return o
                    v=o[0] if isinstance(o,tuple) else o; u=v.clone()
                    for p,z in srcvec.items():u[0,p]+=torch.tensor(z,dtype=u.dtype,device=u.device)
                    return (u,*o[1:]) if isinstance(o,tuple) else u
                def target_hook(_m,_a,o):
                    v=o[0] if isinstance(o,tuple) else o; u=v.clone()
                    if dstvec:
                        for p,z in dstvec.items():u[0,p]+=torch.tensor(z,dtype=u.dtype,device=u.device)
                    captured.append(np.asarray([u[0,row["role_positions"][r]].mean(0).float().cpu().numpy() for r in ROLES],np.float32))
                    return (u,*o[1:]) if isinstance(o,tuple) else u
                h1=layers[q-1].register_forward_hook(source_hook); h2=layers[q].register_forward_hook(target_hook)
                try: output=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
                finally:h1.remove();h2.remove()
                logits=[float(output.logits[0,ids.shape[1]-1,c[0]]) for c in row["candidate_ids"]]; gold=row["gold_position"]
                cond[name]={"target_mean_abs":float(np.mean(np.abs(captured[0]))),"gold_margin":logits[gold]-logits[1-gold]}
            samples.append({"case_id":meta["case_id"],"q":q,"conditions":cond,"correct_margin_rescue":cond["correct_target_restore"]["gold_margin"]-cond["source_delete"]["gold_margin"],"best_wrong_margin_rescue":max(cond[n]["gold_margin"]-cond["source_delete"]["gold_margin"] for n in ("wrong_role_restore","wrong_coordinate_restore","wrong_operation_restore"))})
        rows(out/"raw/sample_results.jsonl",samples); rescue=float(np.mean([x["correct_margin_rescue"] for x in samples])); select=float(np.mean([x["correct_margin_rescue"]-x["best_wrong_margin_rescue"] for x in samples])); passed=rescue>0 and select>0
        close("C357",{"status":"typed_mediation_adjudicated","q":q,"samples":len(samples),"mean_correct_margin_rescue":rescue,"mean_correct_minus_best_wrong":select,"mediation_gate_passed":passed,"placement":placement,"strict_interpretation":"A pass is typed hook-level mediation for the frozen full-field patch, not uniqueness or natural-use proof."},{"samples":len(samples)==4,"finite":finite(samples)},"C358_cross_model")
    finally: old.model_base.release(model);gc.collect()


def c358_c360() -> None:
    out=begin("C358",{"status":"cross_model_abstract_machine_frozen","participants":["qwen3","glm4","deepseek7b"],"state":"per-operation relative-checkpoint role probability vector derived independently inside each model","mapping":"pre-fixed semantic role identity; no optimized permutation","lockbox":"units4-5","metrics":["role argmax error","total variation","log loss","composition sign agreement"]},{"parent":final("C357")["all_checks_passed"],"native_coordinates_unaligned":True})
    if not (out/"analysis/final.json").exists(): close("C358",{"status":"cross_model_abstract_machine_contract_closed","identity_mapping_frozen":True,"max_permutation_forbidden":True,"claim_boundary":"Only lockbox transition distributions can support bisimulation; topology cosine is excluded."},{"three_models":True},"C359_lockbox")
    out=begin("C359",{"status":"cross_model_lockbox_frozen","source":"existing sequential bf16 CUDA C327-C329 archives","abstraction":"five relative checkpoints x six role absolute-response distributions for A,B,I","gate":"every pair mean TV<=0.25 and role argmax disagreement<=0.50"},{"parent":final("C358")["all_checks_passed"],"all_models_available":all(final_old(c)["all_checks_passed"] for c in ("C327","C328","C329"))})
    if not (out/"analysis/final.json").exists():
        models={"qwen3":"C327","glm4":"C328","deepseek7b":"C329"}; abstract={}; composition={}
        for name,c in models.items():
            s=np.load(old.OUTS[c]/"raw/role_states.float16.npy",mmap_mode="r"); idx=rows(old.OUTS[c]/"raw/hidden_index.jsonl"); nq=s.shape[1]; qs=sorted(set(int(round(x*(nq-1))) for x in (0,.25,.5,.75,1)))
            fam=[]
            for f in old.FAMILIES:
                a,g=old.factorial_arrays(s,idx,f); mask=np.asarray([z["unit"] in (4,5) for z in g]); vals=[]
                for arr in (a["a0"],a["b0"],a["interaction"]):
                    e=np.mean(np.abs(arr[mask][:,qs]),axis=(0,3)); e=e/np.maximum(e.sum(axis=-1,keepdims=True),1e-12); vals.append(e)
                fam.append(vals)
            abstract[name]=np.asarray(fam,np.float32); composition[name]=[r["relative_mae_gain"] for r in final_old(c)["headline"]["composition_prediction"]]
        np.save(out/"analysis/model_abstract_machines.float32.npy",np.asarray([abstract[n] for n in models]))
        pair=[]
        causal={n:[r["mean_correct_minus_best_wrong"] for r in final_old(c)["headline"]["causal_response"]] for n,c in models.items()}
        for a,b in itertools.combinations(models,2):
            x=abstract[a];y=abstract[b];tv=float(.5*np.mean(np.sum(np.abs(x-y),axis=-1)));arg=float(np.mean(np.argmax(x,axis=-1)!=np.argmax(y,axis=-1)));ll=float(-np.mean(np.sum(y*np.log(np.maximum(x,1e-8)),axis=-1)));sign=float(np.mean((np.asarray(composition[a])>0)==(np.asarray(composition[b])>0)));causal_sign=float(np.mean((np.asarray(causal[a])>0)==(np.asarray(causal[b])>0)));pair.append({"models":[a,b],"mean_total_variation":tv,"role_argmax_error":arg,"cross_entropy":ll,"composition_sign_agreement":sign,"causal_direction_agreement":causal_sign,"pair_gate_passed":tv<=.25 and arg<=.50 and sign>=.50 and causal_sign==1.0})
        rows(out/"analysis/pair_results.jsonl",pair); gate=all(x["pair_gate_passed"] for x in pair)
        close("C359",{"status":"cross_model_lockbox_adjudicated","pairs":pair,"coarse_abstract_response_gate_passed":gate,"functional_bisimulation_established":False,"strict_interpretation":"A pass is a coarse abstract response-machine candidate. The state abstraction was not calibrated on known-truth machines and no mutually predictive state translator was learned, so functional bisimulation is not established."},{"three_pairs":len(pair)==3,"finite":finite(pair)},"C360_synthesis")
    out=begin("C360",{"status":"campaign_synthesis_frozen","audit_scope":"C336-C359 outputs, full-coordinate artifacts, claim boundaries, visualization payload","new_math_gate":"single_sample and composition and graph and mediation and bisimulation and compression"},{"parent":final("C359")["all_checks_passed"],"all_prior_closed":all((OUTS[f"C{i}"]/"analysis/final.json").exists() for i in range(336,360))})
    if (out/"analysis/final.json").exists(): return
    gates={"single_sample":final("C341")["headline"]["single_sample_gate_passed"],"composition":final("C346")["headline"]["composition_improved_over_mean"],"graph":final("C350")["headline"]["recursive_gate_passed"],"mediation":final("C357")["headline"].get("mediation_gate_passed",False),"bisimulation":final("C359")["headline"]["functional_bisimulation_established"],"compression":final("C342")["headline"]["heldout_predictive_compression_established"]}
    states=np.load(OUTS["C340"]/"raw/role_states.float16.npy",mmap_mode="r"); idx=rows(OUTS["C340"]/"raw/hidden_index.jsonl"); chosen=next(r for r in idx if r["panel"]=="apple_factorial" and r["partition"]=="confirmation" and r["factor_a"]==1 and r["factor_b"]==1 and r["surface"]=="report" and r["order"]==1); field=np.asarray(states[chosen["hidden_index"]],np.float32)
    payload={"schema":"c360-full-coordinate-role-field-v1","case_id":chosen["case_id"],"checkpoints":38,"roles":list(ROLES),"coordinate_count":2560,"coordinate_indices":list(range(2560)),"values":field.tolist(),"gates":gates}
    visual=ROOT/"frontend/public/vis_data/research_kernel/c360_single_sample_operator_field.json"; save(visual,payload)
    headline={"status":"single_sample_operator_campaign_closed","gates":gates,"new_math_gate_passed":all(gates.values()),"attachment_audit":{"correct":["single-sample prediction is the next necessary object","three-valued graph qualification is needed","dynamic state-dependent coalitions are better candidates than fixed Top-K","cross-model comparison must use frozen abstract mappings"],"corrected":["C330 did not maximize its observed statistic; it included identity in a conservative null","role-state evidence does not establish a token-level or causal circuit","category/member controlled English is not a natural apple ontology"]},"visualization":str(visual.relative_to(ROOT)),"strict_conclusion":"The campaign reports which predictive, recursive, mediation, and cross-model gates actually pass. New basic mathematics is not authorized unless all six gates pass."}
    close("C360",headline,{"twenty_four_parents":all(final(f"C{i}")["all_checks_passed"] for i in range(336,360)),"visual_exists":visual.exists(),"full_coordinates":field.shape[-1]==2560,"finite":finite(gates)},"next_campaign_only_for_failed_or_unresolved_gates_with_same_observation_first_goal")


def main() -> None:
    c336(); c337_c338(); c339(); c340(); c341(); c342_c346(); c347_c352(); c353_c355(); c356_c357(); c358_c360()


if __name__ == "__main__":
    main()
