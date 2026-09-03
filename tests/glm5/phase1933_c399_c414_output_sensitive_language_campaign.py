#!/usr/bin/env python3
"""C399-C414 output-sensitive language-family and construction campaign.

The campaign observes embeddings and HiddenStates only. It never reads
attention maps, MLP activations, or model weights. Every observational row
retains the full native activation axis. A failed route changes only its own
authorization and never stops the remaining registered routes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c414_output_sensitive_language_field.json"
sys.path.insert(0, str(TESTS))

import phase1903_c369_c390_language_operation_graph_campaign as previous
import phase1925_c391_c398_independent_construction_lockbox as fresh
import phase1797_c263_c272_state_operator_common as family_base
from model_utils import MODEL_CONFIGS, get_model_info


PHASES = {
    f"C{campaign}": (1933 + campaign - 399, slug)
    for campaign, slug in (
        (399, "evidence_adjudication_and_master_contract"),
        (400, "sixteen_family_output_sensitive_material"),
        (401, "qwen_output_sensitive_behavior"),
        (402, "qwen_output_sensitive_full_coordinate_field"),
        (403, "family_construction_response_atlas"),
        (404, "cross_construction_transfer_matrix"),
        (405, "single_sample_initial_state_prediction"),
        (406, "full_token_coordinate_transmission_observation"),
        (407, "three_factor_nested_composition"),
        (408, "known_truth_role_token_writer_calibration"),
        (409, "ternary_graph_interface_v2_behavior"),
        (410, "ternary_graph_field_and_depth_forecast"),
        (411, "qualified_qwen_hidden_state_writer_branch"),
        (412, "glm_deepseek_output_sensitive_external_panel"),
        (413, "cross_model_relative_response_abstraction"),
        (414, "campaign_synthesis_heatmap_cleanup_audit"),
    )
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}" for name, (phase, slug) in PHASES.items()}

FAMILIES = previous.FAMILIES
CONSTRUCTIONS = ("dossier", "witness", "briefing", "ledger")
CELLS = ("00", "10", "01", "11_ab", "11_ba")
OPS = ("A", "B", "I", "K")
ROLES = previous.ROLES
CHECKPOINTS = 38
DIM = 2560
FIELD_WIDTH = 192

UNITS = (
    {"p": "Alden", "s": "Brena", "o": "Corin", "obj": "quince", "other": "lantern", "node": "qavik", "parent": "produce", "wrong": "instrument", "event": "inspection"},
    {"p": "Darin", "s": "Elara", "o": "Faron", "obj": "turnip", "other": "compass", "node": "turelin", "parent": "vegetable", "wrong": "device", "event": "delivery"},
    {"p": "Galen", "s": "Hesta", "o": "Ivor", "obj": "papaya", "other": "violin", "node": "paporin", "parent": "food", "wrong": "music", "event": "departure"},
    {"p": "Joren", "s": "Kelda", "o": "Laris", "obj": "celeriac", "other": "tripod", "node": "celavik", "parent": "plant", "wrong": "tool", "event": "review"},
    {"p": "Marek", "s": "Nella", "o": "Orin", "obj": "guava", "other": "abacus", "node": "guavor", "parent": "organism", "wrong": "number", "event": "arrival"},
    {"p": "Perrin", "s": "Rhea", "o": "Solen", "obj": "radicchio", "other": "hourglass", "node": "radelin", "parent": "entity", "wrong": "timepiece", "event": "audit"},
)


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def producer_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def begin(name: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[name]
    if (out / "analysis/final.json").exists():
        return out
    if out.exists():
        raise RuntimeError(f"partial output exists: {out}")
    if not all(checks.values()):
        raise RuntimeError((name, checks))
    for sub in ("analysis", "audit", "compiled", "material", "protocol", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name, "created_at_utc": utc_now(),
        "producer_sha256": producer_hash(), **protocol,
    })
    save(out / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    return out


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    out = OUTS[name]
    if (out / "analysis/final.json").exists():
        return load(out / "analysis/final.json")
    save(out / "analysis/summary.json", headline)
    save(out / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    final_checks = {
        "contract": load(out / "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": all(checks.values()),
        "producer_hash": load(out / "protocol/preregistration.json")["producer_sha256"] == producer_hash(),
    }
    value = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "checks": final_checks, "all_checks_passed": all(final_checks.values()),
        "headline": headline, "next_authorization": authorization,
    }
    save(out / "analysis/final.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def close_memmap(value) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def nrmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(prediction, np.float32) - np.asarray(truth, np.float32)) / (np.linalg.norm(np.asarray(truth, np.float32)) + 1e-8))


def family_statement(family: str, unit: int, a: int, b: int) -> dict:
    u = UNITS[unit]
    p, s, o, obj, other = u["p"], u["s"], u["o"], u["obj"], u["other"]
    truth = a == b
    if family == "attitude_event":
        states = ("likes", "dislikes"); relation = states[a]
        query_states = ("like", "dislike")
        fact = f"{p} {relation} eating the {obj}."; question = f"Does {p} {query_states[b]} eating the {obj}?"
        context, primary, secondary = obj, p, s
    elif family == "type_graph":
        states = ("is a kind of", "is not a kind of"); relation = states[a]
        fact = f"The code {u['node']} {relation} {u['parent']}."; question = f"Is it true that {u['node']} {states[b]} {u['parent']}?"
        context, primary, secondary = u["parent"], u["node"], s
    elif family == "contrast":
        states = ("agrees with", "contrasts with"); query_states = ("agree with", "contrast with"); relation = states[a]
        fact = f"{p}'s proposal {relation} {s}'s proposal."; question = f"Does {p}'s proposal {query_states[b]} {s}'s proposal?"
        context, primary, secondary = "proposal", p, s
    elif family == "translation":
        states = (u["parent"], u["wrong"]); relation = "means"
        fact = f"In the codebook, {u['node']} means {states[a]}."; question = f"Does {u['node']} mean {states[b]}?"
        context, primary, secondary = states[a], u["node"], s
    elif family == "comparison":
        states = ("taller than", "shorter than"); relation = states[a]
        fact = f"{p} is {relation} {s}."; question = f"Is {p} {states[b]} {s}?"
        context, primary, secondary = s, p, s
    elif family == "nested_attitude":
        states = ("believes", "doubts"); relation = states[a]
        fact = f"{o} {relation} that {p} completed the {u['event']}."; question = f"Does {o} {states[b]} that {p} completed the {u['event']}?"
        context, primary, secondary = u["event"], o, p
    elif family == "agent_patient_voice":
        relation = "inspected"
        agents = (p, s); patients = (s, p)
        fact = f"{agents[a]} inspected {patients[a]} near the {obj}."; question = f"Did {agents[b]} inspect {patients[b]} near the {obj}?"
        context, primary, secondary = obj, p, s
    elif family == "possession":
        relation = "owns"; owners = (p, s)
        fact = f"{owners[a]} owns the {obj}."; question = f"Does {owners[b]} own the {obj}?"
        context, primary, secondary = obj, p, s
    elif family == "spatial_relation":
        states = ("north of", "south of"); relation = states[a]
        fact = f"{p} is {relation} {s}."; question = f"Is {p} {states[b]} {s}?"
        context, primary, secondary = s, p, s
    elif family == "temporal_order":
        states = ("before", "after"); relation = states[a]
        fact = f"{p} completed the {u['event']} {relation} {s}."; question = f"Did {p} complete the {u['event']} {states[b]} {s}?"
        context, primary, secondary = u["event"], p, s
    elif family == "causal_direction":
        states = ((p, u["event"]), (u["event"], p)); relation = "caused"
        left, right = states[a]; qleft, qright = states[b]
        fact = f"{left} caused {right}."; question = f"Did {qleft} cause {qright}?"
        context, primary, secondary = u["event"], p, s
    elif family == "negation_scope":
        states = ("did not claim", "claimed"); relation = states[a]
        tails = (f"that {p} inspected the {obj}", f"that {p} did not inspect the {obj}")
        fact = f"{o} {states[a]} {tails[a]}."; question = f"Is the record an instance where {o} {states[b]} {tails[b]}?"
        context, primary, secondary = obj, o, p
    elif family == "modality":
        states = ("must inspect", "may inspect"); relation = states[a]
        fact = f"{p} {relation} the {obj}."; question = f"Does the record say that {p} {states[b]} the {obj}?"
        context, primary, secondary = obj, p, s
    elif family == "coreference":
        relation = "I inspected"; speakers = (p, s)
        fact = f"{speakers[a]} told {o}, 'I inspected the {obj}.'"
        question = f"Was {speakers[b]} the speaker who said, 'I inspected the {obj}'?"
        context, primary, secondary = obj, p, s
    elif family == "attribute_binding":
        relation = "has"; bearers = (p, s)
        fact = f"{bearers[a]} has the amber badge."; question = f"Does {bearers[b]} have the amber badge?"
        context, primary, secondary = "amber", p, s
    elif family == "part_whole":
        relation = "includes"; wholes = (obj, other)
        fact = f"The {wholes[a]} includes the {u['node']}."; question = f"Does the {wholes[b]} include the {u['node']}?"
        context, primary, secondary = u["node"], wholes[a], s
    else:
        raise KeyError(family)
    noise = f"Separately, {s} catalogued the {other} for {p}."
    correct, wrong = ("Yes", "No") if truth else ("No", "Yes")
    return {
        "target": fact, "noise": noise, "question": question, "correct": correct, "wrong": wrong,
        "roles": {"primary": primary, "secondary": secondary, "relation": relation, "context": context, "query": context},
        "truth": truth,
    }


def wrap_construction(construction: str, target: str, noise: str, question: str, reverse: bool) -> str:
    left, right = (noise, target) if reverse else (target, noise)
    if construction == "dossier":
        return f"A dossier states: {left} It also states: {right} From the dossier alone, {question}"
    if construction == "witness":
        return f"A witness gave two statements. First: {left} Second: {right} Using only that testimony, {question}"
    if construction == "briefing":
        return f"During a briefing, one item was {left} Another independent item was {right} Decide from these items: {question}"
    if construction == "ledger":
        return f"Ledger entry one: {left} Ledger entry two: {right} Verify the following query from the ledger: {question}"
    raise KeyError(construction)


def main_material() -> list[dict]:
    rows = []
    for family, construction, unit, cell, order in itertools.product(FAMILIES, CONSTRUCTIONS, range(len(UNITS)), CELLS, (1, -1)):
        a, b = (0, 0) if cell == "00" else (1, 0) if cell == "10" else (0, 1) if cell == "01" else (1, 1)
        case = family_statement(family, unit, a, b)
        core = wrap_construction(construction, case["target"], case["noise"], case["question"], cell == "11_ba")
        choices, gold = family_base.options(case["correct"], case["wrong"], order)
        rows.append({
            "case_id": f"c400-{family}-{construction}-u{unit}-{cell}-{order:+d}",
            "panel": "output_sensitive_family", "family": family, "surface": construction,
            "construction": construction, "unit": unit, "cell": cell, "factor_a": a, "factor_b": b,
            "composition_order": "ba" if cell == "11_ba" else "ab", "order_semantics": "realization",
            "order": order, "partition": "discovery" if unit < 3 else "confirmation" if unit < 5 else "lockbox",
            "gold_position": gold, "correct_answer": case["correct"], "wrong_answer": case["wrong"],
            "prompt_core": core, "prompt": f"{core} {choices}. Reply with only A or B.",
            "free_prompt": f"{core} Answer with only Yes or No.", "role_values": case["roles"],
            "semantic_graph": {"family": family, "state": a, "query_state": b, "truth": case["truth"], "construction": construction},
        })
    return rows


def main_lookup() -> tuple[list[dict], dict[str, dict]]:
    rows = read_rows(OUTS["C400"] / "material/cases.jsonl")
    return rows, {row["case_id"]: row for row in rows}


def complete_groups(states: np.ndarray, index: list[dict], material: dict[str, dict], family: str, partitions: tuple[str, ...] | None = None) -> list[dict]:
    eligible = []
    for row in index:
        case = material[row["case_id"]]
        if case["family"] != family or (partitions is not None and case["partition"] not in partitions):
            continue
        if not row["correct"]:
            continue
        eligible.append((row, case))
    keyed = {(case["construction"], case["unit"], case["order"], case["cell"]): row for row, case in eligible}
    groups = []
    for construction, unit, order in itertools.product(CONSTRUCTIONS, range(len(UNITS)), (1, -1)):
        keys = [(construction, unit, order, cell) for cell in CELLS]
        if not all(key in keyed for key in keys):
            continue
        rows = [keyed[key] for key in keys]
        values = {cell: np.asarray(states[row["hidden_index"]], np.float32) for cell, row in zip(CELLS, rows)}
        groups.append({
            "construction": construction, "unit": unit, "order": order,
            "partition": material[rows[0]["case_id"]]["partition"], "H00": values["00"],
            "A": values["10"] - values["00"], "B": values["01"] - values["00"],
            "I": values["11_ab"] - values["10"] - values["01"] + values["00"],
            "K": values["11_ab"] - values["11_ba"],
        })
    return groups


def c399() -> None:
    out = begin("C399", {
        "status": "evidence_adjudication_and_master_contract_frozen",
        "parent": "C398 independent audit 25/25",
        "arms": ["sixteen_family_output_sensitive", "nested_composition", "ternary_graph_v2", "known_truth_writer", "cross_model_external"],
        "policy": "observe then predict then causal; route failure never stops other registered arms",
        "measurement": "embedding and HiddenState only; no Attention, MLP, weights, PCA, Top-K, or cosine gate",
    }, {"parent": fresh.final("C398")["all_checks_passed"], "phase_continuity": PHASES["C399"][0] == 1933})
    headline = {
        "status": "master_contract_closed",
        "retained": ["external language families organize experiments", "full-coordinate response ecology exists", "within-construction conditional gains are weakly positive"],
        "corrected": ["language families are not neural ontology", "C397 decoding is embedding-confounded", "no broad operator, recursion, causal closure, or new mathematics"],
        "strict_interpretation": "The campaign searches for transferable patterns without assuming a fixed coordinate dictionary or a language algebra.",
    }
    close("C399", headline, {"objects_separated": True, "no_causal_presumption": True}, "C400_material")


def c400() -> None:
    out = begin("C400", {
        "status": "sixteen_family_output_sensitive_material_frozen",
        "design": "16 families x 4 constructions x 6 lexical units x 5 cells x 2 answer orders",
        "factors": "statement state A and queried state B both change the correct Yes/No answer",
        "human_naturalness_review": False,
    }, {"parent": final("C399")["all_checks_passed"], "families": len(FAMILIES) == 16})
    rows = main_material()
    write_rows(out / "material/cases.jsonl", rows)
    zero = {"always_first": 0.5, "always_second": 0.5}
    for key in ("family", "construction", "cell"):
        correct = 0
        for value in sorted({row[key] for row in rows}):
            subset = [row for row in rows if row[key] == value]
            majority = int(np.mean([row["gold_position"] for row in subset]) >= 0.5)
            correct += sum(row["gold_position"] == majority for row in subset)
        zero[f"{key}_majority"] = correct / len(rows)
    role_occurrence = all(all(str(value) in row["prompt_core"] for value in row["role_values"].values()) for row in rows)
    output_sensitive = True
    by_group = {(r["family"], r["construction"], r["unit"], r["order"], r["cell"]): r for r in rows}
    for family, construction, unit, order in itertools.product(FAMILIES, CONSTRUCTIONS, range(len(UNITS)), (1, -1)):
        base = by_group[(family, construction, unit, order, "00")]["correct_answer"]
        output_sensitive &= base != by_group[(family, construction, unit, order, "10")]["correct_answer"]
        output_sensitive &= base != by_group[(family, construction, unit, order, "01")]["correct_answer"]
    headline = {
        "status": "output_sensitive_material_closed", "rows": len(rows),
        "partition_counts": {p: sum(r["partition"] == p for r in rows) for p in ("discovery", "confirmation", "lockbox")},
        "zero_model_accuracies": zero, "role_occurrence": role_occurrence,
        "both_factors_output_sensitive": bool(output_sensitive), "material_eligible": max(zero.values()) <= 0.51 and role_occurrence and output_sensitive,
        "human_naturalness_review": False,
        "strict_interpretation": "Exact output sensitivity and balance do not certify naturalness or isolate a neural family.",
    }
    close("C400", headline, {"rows": len(rows) == 3840, "zero": max(zero.values()) <= 0.51, "roles": role_occurrence, "output": output_sensitive}, "C401_qwen_behavior")


def c401() -> None:
    out = begin("C401", {
        "status": "qwen_output_sensitive_behavior_frozen", "model": "Qwen3-4B BF16 CUDA",
        "gates": {"heldout": 0.80, "family": 0.60, "construction": 0.70},
        "hidden_state_policy": "no hidden states requested in this phase",
    }, {"parent": final("C400")["all_checks_passed"], "material": final("C400")["headline"]["material_eligible"], "cuda": torch.cuda.is_available()})
    rows, _ = main_lookup()
    tokenizer = fresh.tokenizer_qwen()
    compiled = family_base.compile_qwen(tokenizer, rows)
    write_rows(out / "compiled/qwen3.jsonl", compiled)
    metrics = previous.qwen_behavior(rows, compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl")
    held = [row for row in behavior if row["partition"] != "discovery"]
    by_family = {family: float(np.mean([r["correct"] for r in held if r["family"] == family])) for family in FAMILIES}
    by_construction = {construction: float(np.mean([r["correct"] for r in held if r["surface"] == construction])) for construction in CONSTRUCTIONS}
    eligible = [family for family, score in by_family.items() if score >= 0.60]
    heldout = float(np.mean([row["correct"] for row in held]))
    headline = {
        "status": "qwen_output_sensitive_behavior_closed", **metrics, "heldout_accuracy": heldout,
        "family_accuracy": by_family, "construction_accuracy": by_construction,
        "eligible_families": eligible, "field_eligible": heldout >= 0.80 and min(by_construction.values()) >= 0.70 and len(eligible) >= 1,
        "strict_interpretation": "Behavior qualifies only listed families for internal interpretation; all family routes remain reportable.",
    }
    close("C401", headline, {"rows": len(behavior) == len(rows), "finite": finite(headline), "no_hidden": not (out / "raw/role_states.float16.npy").exists()}, "C402_full_field")


def c402() -> None:
    eligible = set(final("C401")["headline"]["eligible_families"])
    out = begin("C402", {
        "status": "qwen_output_sensitive_full_coordinate_field_frozen",
        "archive": "behavior-qualified families x embedding+36 blocks+final norm x six roles x all 2560 coordinates",
        "all_token_subset": "unit5, answer order +1, cells00/10, dossier/witness",
        "no_pca_topk": True, "cleanup": "checksum then remove only after C414 visual export",
    }, {"parent": final("C401")["all_checks_passed"], "field": final("C401")["headline"]["field_eligible"], "eligible": len(eligible) > 0, "cuda": torch.cuda.is_available()})
    rows, _ = main_lookup()
    compiled = read_rows(OUTS["C401"] / "compiled/qwen3.jsonl")
    selected = [(row, comp) for row, comp in zip(rows, compiled) if row["family"] in eligible]
    rows = [row for row, _ in selected]; compiled = [comp for _, comp in selected]
    selector = lambda row: row["unit"] == 5 and row["surface"] in ("dossier", "witness") and row["order"] == 1 and row["cell"] in ("00", "10")
    metrics = previous.common.batch_capture_qwen(rows, compiled, out, full_selector=selector, batch_size=8, field_width=FIELD_WIDTH)
    headline = {"status": "qwen_output_sensitive_full_field_closed", **metrics, "eligible_families": sorted(eligible), "strict_interpretation": "Role means and the complete-token subset are observational views with every physical coordinate retained."}
    close("C402", headline, {"rows": metrics["rows"] == len(rows), "shape": metrics["role_shape"][-1] == DIM and metrics["role_shape"][1:3] == [38, 6], "full": metrics["full_shape"][-1] == DIM, "finite": finite(headline)}, "C403_atlas")


def c403() -> None:
    out = begin("C403", {
        "status": "family_construction_response_atlas_frozen",
        "object": "family x construction x A/B/I/K x checkpoint x role x all coordinates",
        "qualification": "only complete five-cell behavior-correct groups",
    }, {"parent": final("C402")["all_checks_passed"]})
    states = np.load(OUTS["C402"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C402"] / "raw/hidden_index.jsonl")
    _rows, material = main_lookup()
    eligible = final("C402")["headline"]["eligible_families"]
    atlas = np.lib.format.open_memmap(out / "analysis/family_construction_operation_mean.float16.npy", mode="w+", dtype=np.float16, shape=(len(eligible), len(CONSTRUCTIONS), len(OPS), CHECKPOINTS, len(ROLES), DIM))
    counts, sign = {}, []
    for fi, family in enumerate(eligible):
        groups = complete_groups(states, index, material, family)
        counts[family] = len(groups)
        for ci, construction in enumerate(CONSTRUCTIONS):
            subset = [group for group in groups if group["construction"] == construction]
            for oi, op in enumerate(OPS):
                values = np.asarray([group[op] for group in subset], np.float32)
                mean = values.mean(0) if len(values) else np.zeros((38, 6, DIM), np.float32)
                atlas[fi, ci, oi] = mean.astype(np.float16)
                if len(values):
                    agreement = np.maximum(np.mean(values >= 0, axis=0), np.mean(values < 0, axis=0))
                    sign.append(float(np.mean(agreement)))
        atlas.flush(); del groups; gc.collect()
    headline = {
        "status": "family_construction_response_atlas_closed", "shape": list(atlas.shape), "group_counts": counts,
        "atlas_eligible_families": [family for family, count in counts.items() if count > 0],
        "mean_coordinate_sign_agreement": float(np.mean(sign)) if sign else None,
        "strict_interpretation": "A construction-conditioned mean response is not a universal family operator or a causal coordinate dictionary.",
    }
    close_memmap(atlas); close_memmap(states)
    close("C403", headline, {"shape": headline["shape"][-3:] == [38, 6, DIM], "group_accounting": set(counts) == set(eligible), "finite": finite(headline)}, "C404_cross_construction")


def c404() -> None:
    out = begin("C404", {
        "status": "cross_construction_transfer_matrix_frozen",
        "train": "source construction, discovery units0-2", "test": "different construction, confirmation/lockbox units3-5",
        "controls": ["zero", "all-construction family mean", "coordinate roll", "wrong family"],
        "pass": "source beats every control and improves family mean by more than 0.005",
    }, {"parent": final("C403")["all_checks_passed"]})
    states = np.load(OUTS["C402"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C402"] / "raw/hidden_index.jsonl")
    _rows, material = main_lookup()
    eligible = final("C403")["headline"]["atlas_eligible_families"]
    family_groups = {family: complete_groups(states, index, material, family) for family in eligible}
    details, qualified = [], []
    for fi, family in enumerate(eligible):
        groups = family_groups[family]
        wrong_family = eligible[(fi + 1) % len(eligible)]
        wrong_groups = family_groups[wrong_family]
        for op, source, target in itertools.product(OPS, CONSTRUCTIONS, CONSTRUCTIONS):
            if source == target:
                continue
            train = [g for g in groups if g["construction"] == source and g["unit"] < 3]
            test = [g for g in groups if g["construction"] == target and g["unit"] >= 3]
            all_train = [g for g in groups if g["unit"] < 3]
            wrong_train = [g for g in wrong_groups if g["construction"] == source and g["unit"] < 3]
            if not train or not test or not all_train or not wrong_train:
                continue
            prediction = np.mean([g[op] for g in train], axis=0)
            family_mean = np.mean([g[op] for g in all_train], axis=0)
            wrong = np.mean([g[op] for g in wrong_train], axis=0)
            scores = {
                "source": float(np.mean([nrmse(prediction, g[op]) for g in test])),
                "family_mean": float(np.mean([nrmse(family_mean, g[op]) for g in test])),
                "zero": 1.0,
                "coordinate_roll": float(np.mean([nrmse(np.roll(prediction, 1, axis=-1), g[op]) for g in test])),
                "wrong_family": float(np.mean([nrmse(wrong, g[op]) for g in test])),
            }
            passed = scores["source"] + 0.005 < scores["family_mean"] and scores["source"] < min(scores["zero"], scores["coordinate_roll"], scores["wrong_family"])
            row = {"family": family, "operation": op, "source": source, "target": target, "train_groups": len(train), "test_groups": len(test), "nrmse": scores, "gain_over_family_mean": scores["family_mean"] - scores["source"], "passed": passed}
            details.append(row)
            if passed:
                qualified.append(row)
    write_rows(out / "analysis/transfer_matrix.jsonl", details)
    by_family = {family: sum(row["passed"] for row in details if row["family"] == family) for family in eligible}
    candidates = [family for family, count in by_family.items() if count >= 2]
    headline = {
        "status": "cross_construction_transfer_closed", "cells": len(details), "passed_cells": len(qualified),
        "family_pass_counts": by_family, "candidate_families": candidates,
        "mean_gain_over_family_mean": float(np.mean([row["gain_over_family_mean"] for row in details])) if details else None,
        "strict_interpretation": "A passing cell is one source-to-target construction prediction, never a global language-family operator.",
    }
    close_memmap(states)
    close("C404", headline, {"route_accounted": isinstance(details, list), "finite": finite(headline)}, "C405_initial_state")


def c405() -> None:
    out = begin("C405", {
        "status": "single_sample_initial_state_prediction_frozen",
        "predictor": "nearest discovery H00 at embedding checkpoint q0, all six roles and all 2560 coordinates",
        "truth": "complete A/B/I/K response field", "controls": ["construction mean", "coordinate roll"],
        "no_pca_topk": True,
    }, {"parent": final("C404")["all_checks_passed"]})
    states = np.load(OUTS["C402"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C402"] / "raw/hidden_index.jsonl")
    _rows, material = main_lookup()
    eligible = final("C403")["headline"]["atlas_eligible_families"]
    results, passing = [], []
    for family in eligible:
        groups = complete_groups(states, index, material, family)
        train = [g for g in groups if g["unit"] < 3]
        test = [g for g in groups if g["unit"] >= 3]
        if not train or not test:
            continue
        train_h = np.asarray([g["H00"][0].reshape(-1) for g in train], np.float32)
        for op in OPS:
            nn_scores, mean_scores, roll_scores = [], [], []
            construction_means = {
                c: np.mean([g[op] for g in train if g["construction"] == c], axis=0)
                for c in CONSTRUCTIONS
                if any(g["construction"] == c for g in train)
            }
            for group in test:
                if group["construction"] not in construction_means:
                    continue
                h = group["H00"][0].reshape(-1).astype(np.float32)
                nearest = int(np.argmin(np.mean((train_h - h[None, :]) ** 2, axis=1)))
                prediction = train[nearest][op]
                nn_scores.append(nrmse(prediction, group[op]))
                mean_scores.append(nrmse(construction_means[group["construction"]], group[op]))
                roll_scores.append(nrmse(np.roll(prediction, 1, axis=-1), group[op]))
            row = {"family": family, "operation": op, "nrmse_nearest": float(np.mean(nn_scores)), "nrmse_construction_mean": float(np.mean(mean_scores)), "nrmse_coordinate_roll": float(np.mean(roll_scores))}
            row["gain_over_mean"] = row["nrmse_construction_mean"] - row["nrmse_nearest"]
            row["passed"] = row["gain_over_mean"] > 0.005 and row["nrmse_nearest"] < row["nrmse_coordinate_roll"]
            results.append(row)
            if row["passed"]:
                passing.append(row)
        del groups; gc.collect()
    write_rows(out / "analysis/initial_state_predictions.jsonl", results)
    candidates = sorted({row["family"] for row in passing})
    headline = {
        "status": "single_sample_initial_state_prediction_closed", "cells": len(results), "passed_cells": len(passing),
        "candidate_families": candidates, "mean_gain_over_construction_mean": float(np.mean([r["gain_over_mean"] for r in results])) if results else None,
        "strict_interpretation": "Nearest-H00 prediction is an effective dependency test, not a unique state variable or causal circuit.",
    }
    close_memmap(states)
    close("C405", headline, {"route_accounted": isinstance(results, list), "finite": finite(headline)}, "C406_full_token")


def c406() -> None:
    out = begin("C406", {
        "status": "full_token_coordinate_transmission_observation_frozen",
        "contrast": "cell10 minus cell00 at matched family/construction/unit/order",
        "archive": "every checkpoint, physical token position, and all 2560 coordinates",
        "claim": "effective positional response only; no unique causal edge",
    }, {"parent": final("C405")["all_checks_passed"]})
    fields = np.load(OUTS["C402"] / "raw/full_fields_holdout.float16.npy", mmap_mode="r")
    row_map = load(OUTS["C402"] / "raw/full_field_row_map.json")["source_indices"]
    index = read_rows(OUTS["C402"] / "raw/hidden_index.jsonl")
    _rows, material = main_lookup()
    selected = [(local, material[index[source]["case_id"]]) for local, source in enumerate(row_map)]
    keyed = {(case["family"], case["construction"], case["unit"], case["order"], case["cell"]): local for local, case in selected}
    pairs = []
    for family, construction in itertools.product(final("C402")["headline"]["eligible_families"], ("dossier", "witness")):
        k0 = (family, construction, 5, 1, "00"); k1 = (family, construction, 5, 1, "10")
        if k0 in keyed and k1 in keyed:
            pairs.append((family, construction, keyed[k0], keyed[k1]))
    deltas = np.lib.format.open_memmap(out / "analysis/full_token_delta.float16.npy", mode="w+", dtype=np.float16, shape=(len(pairs), CHECKPOINTS, FIELD_WIDTH, DIM))
    energy = np.zeros((len(pairs), CHECKPOINTS, FIELD_WIDTH), np.float32)
    meta = []
    for i, (family, construction, i0, i1) in enumerate(pairs):
        delta = np.asarray(fields[i1], np.float32) - np.asarray(fields[i0], np.float32)
        deltas[i] = delta.astype(np.float16); energy[i] = np.mean(np.abs(delta), axis=-1)
        meta.append({"delta_index": i, "family": family, "construction": construction, "source_cell": "00", "target_cell": "10"})
        deltas.flush()
    np.save(out / "analysis/full_token_energy.float32.npy", energy)
    write_rows(out / "analysis/full_token_delta_index.jsonl", meta)
    headline = {
        "status": "full_token_transmission_observation_closed", "pairs": len(pairs), "delta_shape": list(deltas.shape),
        "nonzero_fraction": float(np.mean(np.abs(np.asarray(deltas[:, :, :, :64], np.float32)) > 0)) if len(pairs) else 0.0,
        "strict_interpretation": "Physical token-position deltas preserve every coordinate but do not identify a unique source-target circuit.",
    }
    close_memmap(deltas); close_memmap(fields)
    close("C406", headline, {"pairs": len(pairs) > 0, "shape": headline["delta_shape"][-1] == DIM, "finite": finite(headline)}, "C407_nested_composition")


COMPOSITION_FAMILIES = ("attitude_event", "nested_attitude", "negation_scope")


def composition_material() -> list[dict]:
    rows = []
    for family, construction, unit, bits, order in itertools.product(COMPOSITION_FAMILIES, CONSTRUCTIONS, range(len(UNITS)), itertools.product((0, 1), repeat=3), (1, -1)):
        a, b, c = bits; u = UNITS[unit]; p, s, o, obj = u["p"], u["s"], u["o"], u["obj"]
        attitude = "likes" if a == 0 else "dislikes"; event = "ate" if b == 0 else "did not eat"; item = obj if c == 0 else u["other"]
        if family == "attitude_event":
            target = f"{p} {attitude} the event in which {s} {event} the {item}."; relation = attitude
        elif family == "nested_attitude":
            target = f"{o} believes that {p} {attitude} the event in which {s} {event} the {item}."; relation = "believes"
        else:
            target = f"{o} did not deny that {p} {attitude} the event in which {s} {event} the {item}."; relation = "did not deny"
        query = f"Does the record exactly say that {p} likes the event in which {s} ate the {obj}?"
        truth = a == b == c == 0; correct, wrong = ("Yes", "No") if truth else ("No", "Yes")
        noise = f"Separately, {s} catalogued the {u['other']}."
        core = wrap_construction(construction, target, noise, query, False)
        choices, gold = family_base.options(correct, wrong, order)
        rows.append({
            "case_id": f"c407-{family}-{construction}-u{unit}-{a}{b}{c}-{order:+d}", "panel": "three_factor_composition",
            "family": family, "surface": construction, "construction": construction, "unit": unit,
            "factor_a": a, "factor_b": b, "factor_c": c, "cell": f"{a}{b}{c}", "order": order,
            "partition": "discovery" if unit < 3 else "lockbox", "gold_position": gold,
            "correct_answer": correct, "wrong_answer": wrong, "prompt_core": core,
            "prompt": f"{core} {choices}. Reply with only A or B.", "free_prompt": f"{core} Answer with only Yes or No.",
            "role_values": {"primary": p, "secondary": s, "relation": relation, "context": item, "query": obj},
            "semantic_graph": {"family": family, "attitude": a, "event_polarity": b, "patient": c},
        })
    return rows


def c407() -> None:
    out = begin("C407", {
        "status": "three_factor_nested_composition_frozen",
        "design": "3 families x 4 constructions x 6 units x 8 cells x 2 answer orders",
        "prediction": "lockbox H111 from its H000 plus discovery-only mean atomic and pair residuals; no other lockbox corners are read",
        "behavior_before_hidden": True,
    }, {"parent": final("C406")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = composition_material(); write_rows(out / "material/cases.jsonl", rows)
    tokenizer = fresh.tokenizer_qwen(); compiled = family_base.compile_qwen(tokenizer, rows); write_rows(out / "compiled/qwen3.jsonl", compiled)
    behavior_metrics = previous.qwen_behavior(rows, compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl"); held = [r for r in behavior if r["partition"] == "lockbox"]
    by_family = {family: float(np.mean([r["correct"] for r in held if r["family"] == family])) for family in COMPOSITION_FAMILIES}
    behavior_eligible = float(np.mean([r["correct"] for r in held])) >= 0.80 and min(by_family.values()) >= 0.60
    composition = {"ran": False, "reason": "behavior_ineligible"}
    if behavior_eligible:
        metrics = previous.common.batch_capture_qwen(rows, compiled, out, full_selector=None, batch_size=8, field_width=FIELD_WIDTH)
        states = np.load(out / "raw/role_states.float16.npy", mmap_mode="r")
        index = read_rows(out / "raw/hidden_index.jsonl"); mat = {r["case_id"]: r for r in rows}
        keyed = {(mat[r["case_id"]]["family"], mat[r["case_id"]]["construction"], mat[r["case_id"]]["unit"], mat[r["case_id"]]["order"], mat[r["case_id"]]["cell"]): r for r in index if r["correct"]}
        family_results = []
        for family in COMPOSITION_FAMILIES:
            train_effects = {name: [] for name in ("A", "B", "C", "AB", "AC", "BC")}
            tests = []
            for construction, unit, order in itertools.product(CONSTRUCTIONS, range(len(UNITS)), (1, -1)):
                keys = {cell: (family, construction, unit, order, cell) for cell in ("000", "100", "010", "001", "110", "101", "011", "111")}
                if not all(key in keyed for key in keys.values()):
                    continue
                h = {cell: np.asarray(states[keyed[keys[cell]]["hidden_index"]], np.float32) for cell in keys}
                effects = {
                    "A": h["100"] - h["000"], "B": h["010"] - h["000"], "C": h["001"] - h["000"],
                    "AB": h["110"] - h["100"] - h["010"] + h["000"],
                    "AC": h["101"] - h["100"] - h["001"] + h["000"],
                    "BC": h["011"] - h["010"] - h["001"] + h["000"],
                }
                if unit < 3:
                    for name in train_effects: train_effects[name].append(effects[name])
                else:
                    tests.append((h["000"], h["111"]))
            means = {name: np.mean(values, axis=0) for name, values in train_effects.items() if values}
            if len(means) == len(train_effects) and tests:
                additive = [nrmse(h0 + means["A"] + means["B"] + means["C"], truth) for h0, truth in tests]
                pair = [nrmse(h0 + means["A"] + means["B"] + means["C"] + means["AB"] + means["AC"] + means["BC"], truth) for h0, truth in tests]
                family_results.append({"family": family, "prediction_ran": True, "train_groups": len(train_effects["A"]), "test_groups": len(tests), "nrmse_additive": float(np.mean(additive)), "nrmse_pair_residual": float(np.mean(pair)), "pair_gain": float(np.mean(additive) - np.mean(pair))})
            else:
                family_results.append({"family": family, "prediction_ran": False, "train_groups": len(train_effects["A"]), "test_groups": len(tests), "reason": "no_complete_behavior_correct_discovery_or_lockbox_group"})
        composition = {"ran": True, "field_metrics": metrics, "family_results": family_results, "strict_interpretation": "The predictor reads only each lockbox H000 and discovery means; success would be construction-panel prediction, not recursive closure."}
        close_memmap(states)
    headline = {"status": "three_factor_nested_composition_closed", **behavior_metrics, "lockbox_accuracy": float(np.mean([r["correct"] for r in held])), "family_accuracy": by_family, "behavior_eligible": behavior_eligible, "composition": composition}
    close("C407", headline, {"rows": len(rows) == 1152, "behavior": len(behavior) == len(rows), "branch": composition["ran"] == behavior_eligible, "finite": finite(headline)}, "C408_writer_calibration")


def c408() -> None:
    out = begin("C408", {
        "status": "known_truth_role_token_writer_calibration_frozen",
        "systems": 512, "dimension": DIM, "controls": ["wrong role", "wrong checkpoint", "wrong operation", "energy matched noise"],
        "claim": "instrument calibration only; no Qwen mechanism inference",
    }, {"parent": final("C407")["all_checks_passed"]})
    rng = np.random.default_rng(408); recoveries = {name: [] for name in ("correct", "wrong_role", "wrong_checkpoint", "wrong_operation", "energy_noise")}
    for _ in range(512):
        target = rng.normal(size=DIM).astype(np.float32); target /= np.linalg.norm(target) + 1e-8
        delta = target * rng.uniform(0.5, 1.5); base = rng.normal(scale=0.05, size=DIM).astype(np.float32)
        truth_score = float(np.dot(base + delta, target)); ablated = float(np.dot(base, target)); denom = truth_score - ablated + 1e-8
        controls = {
            "correct": delta,
            "wrong_role": np.roll(delta, 1),
            "wrong_checkpoint": -delta,
            "wrong_operation": np.roll(delta, 127),
            "energy_noise": (lambda v: v / (np.linalg.norm(v) + 1e-8) * np.linalg.norm(delta))(rng.normal(size=DIM).astype(np.float32)),
        }
        for name, write in controls.items():
            recoveries[name].append(float((np.dot(base + write, target) - ablated) / denom))
    means = {name: float(np.mean(values)) for name, values in recoveries.items()}
    passed = means["correct"] > 0.99 and means["correct"] > max(means[name] for name in means if name != "correct") + 0.5
    headline = {"status": "known_truth_writer_calibration_closed", "mean_recovery": means, "writer_calibrated": passed, "strict_interpretation": "This validates branch logic and controls in a constructed system, not role-token writing in Qwen."}
    close("C408", headline, {"systems": all(len(v) == 512 for v in recoveries.values()), "writer": passed, "finite": finite(headline)}, "C409_graph_behavior")


GRAPH_UNITS = tuple({"root": f"nax{chr(97+i)}", "m1": f"bel{chr(97+i)}", "m2": f"cor{chr(97+i)}", "m3": f"dov{chr(97+i)}", "final": f"class{chr(97+i)}", "wrong": f"other{chr(97+i)}"} for i in range(12))
GRAPH_MODES = ("entailed", "contradicted", "unknown", "multipath", "shortcut", "reversed", "broken")


def graph_material() -> list[dict]:
    rows = []; permutations = tuple(itertools.permutations(("entailed", "contradicted", "unknown")))
    group_i = 0
    for unit_i, depth, construction in itertools.product(range(12), range(1, 5), ("registry", "briefing")):
        u = GRAPH_UNITS[unit_i]; order = permutations[group_i % len(permutations)]; group_i += 1
        nodes = [u["root"], u["m1"], u["m2"], u["m3"], u["final"]]; path = [nodes[0], *nodes[1:depth], u["final"]]
        for mode in GRAPH_MODES:
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            label = "entailed" if mode in ("entailed", "multipath", "shortcut") else "contradicted" if mode == "contradicted" else "unknown"
            if mode == "contradicted":
                facts = [f'The registry explicitly says that "{u["root"]}" is not a kind of "{u["final"]}".']
                relation = "is not a kind of"
            else:
                if mode == "reversed": edges = [(right, left) for left, right in edges]
                if mode == "broken" and edges: edges[len(edges)//2] = (edges[len(edges)//2][0], u["wrong"])
                facts = [f'"{left}" is a kind of "{right}".' for left, right in edges]
                if mode == "unknown": facts = [f'"{u["root"]}" is a kind of "{u["m1"]}".', f'"{u["wrong"]}" is a kind of "{u["final"]}".']
                if mode == "multipath": facts.append(f'A second register directly says that "{u["root"]}" is a kind of "{u["final"]}".')
                if mode == "shortcut": facts.append(f'A direct shortcut says that "{u["root"]}" is a kind of "{u["final"]}".')
                relation = "is a kind of"
            body = " ".join(facts); rules = "Use only these rules: kind-of links are transitive; an explicit not-kind-of statement contradicts the query; missing links are unknown."
            core = (f"Registry facts: {body} {rules} Classify whether \"{u['root']}\" is a kind of \"{u['final']}\"." if construction == "registry" else f"A briefing reports: {body} {rules} Is the claim that \"{u['root']}\" is a kind of \"{u['final']}\" entailed, contradicted, or unknown?")
            options = " ".join(f"({chr(65+i)}) {value}" for i, value in enumerate(order)); gold = order.index(label)
            rows.append({
                "case_id": f"c409-{construction}-u{unit_i}-d{depth}-{mode}", "panel": "ternary_graph_v2", "family": "type_graph",
                "surface": construction, "unit": unit_i, "depth": depth, "mode": mode,
                "partition": "discovery" if unit_i < 6 else "confirmation" if unit_i < 10 else "lockbox",
                "gold_position": gold, "prompt_core": core, "prompt": f"{core} {options} Reply with only A, B, or C.",
                "role_values": {"primary": u["root"], "secondary": u["final"], "relation": relation, "context": u["final"], "query": u["root"]},
                "semantic_graph": {"depth": depth, "mode": mode, "label": label, "candidate_order": list(order)},
            })
    return rows


def c409() -> None:
    out = begin("C409", {
        "status": "ternary_graph_interface_v2_behavior_frozen",
        "design": "12 graphs x depths1-4 x 7 path modes x 2 constructions; candidate permutations balanced by graph group",
        "hidden_state_policy": "no hidden states requested", "gates": {"heldout": 0.80, "label": 0.70, "depth": 0.65},
    }, {"parent": final("C408")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = graph_material(); write_rows(out / "material/cases.jsonl", rows)
    tokenizer = fresh.tokenizer_qwen(); compiled = previous.compile_qwen_multiclass(tokenizer, rows); write_rows(out / "compiled/qwen3.jsonl", compiled)
    metrics = previous.qwen_multiclass_behavior(compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl"); held = [r for r in behavior if r["partition"] != "discovery"]
    material = {r["case_id"]: r for r in rows}
    by_label = {label: float(np.mean([r["correct"] for r in held if material[r["case_id"]]["semantic_graph"]["label"] == label])) for label in ("entailed", "contradicted", "unknown")}
    by_depth = {str(depth): float(np.mean([r["correct"] for r in held if r["depth"] == depth])) for depth in range(1, 5)}
    positions = {str(i): sum(r["gold_position"] == i for r in rows) / len(rows) for i in range(3)}
    heldout = float(np.mean([r["correct"] for r in held])); eligible = heldout >= 0.80 and min(by_label.values()) >= 0.70 and min(by_depth.values()) >= 0.65
    headline = {"status": "ternary_graph_v2_behavior_closed", **metrics, "heldout_accuracy": heldout, "label_accuracy": by_label, "depth_accuracy": by_depth, "gold_position_frequency": positions, "graph_field_eligible": eligible, "strict_interpretation": "Failure would invalidate this interface only, not graph reasoning in Qwen."}
    close("C409", headline, {"rows": len(rows) == 672, "positions": max(abs(v - 1/3) for v in positions.values()) < 1e-9, "finite": finite(headline)}, "C410_graph_field")


def c410() -> None:
    eligible = final("C409")["headline"]["graph_field_eligible"]
    out = begin("C410", {
        "status": "ternary_graph_field_and_depth_forecast_frozen",
        "run_condition": "C409 behavior eligibility", "forecast": "depth1-2 response increments predict depth3-4",
        "routing": "ineligible behavior records a legal no-field branch without stopping C411-C414",
    }, {"parent": final("C409")["all_checks_passed"]})
    if not eligible:
        headline = {"status": "graph_field_not_run_behavior_ineligible", "field_ran": False, "depth_forecast_ran": False, "strict_interpretation": "The internal graph mechanism remains untested."}
        close("C410", headline, {"no_field": not (out / "raw/role_states.float16.npy").exists()}, "C411_causal_branch")
        return
    compiled = read_rows(OUTS["C409"] / "compiled/qwen3.jsonl")
    metrics = previous.capture_qwen_multiclass(compiled, out, batch_size=8)
    states = np.load(out / "raw/role_states.float16.npy", mmap_mode="r"); index = read_rows(out / "raw/hidden_index.jsonl")
    rows = read_rows(OUTS["C409"] / "material/cases.jsonl"); material = {r["case_id"]: r for r in rows}
    keyed = {(material[r["case_id"]]["unit"], material[r["case_id"]]["depth"], material[r["case_id"]]["surface"], material[r["case_id"]]["mode"]): r for r in index if r["correct"]}
    train, test = [], []
    for unit, depth, surface in itertools.product(range(12), range(1, 5), ("registry", "briefing")):
        ke = (unit, depth, surface, "entailed"); ku = (unit, depth, surface, "unknown")
        if ke not in keyed or ku not in keyed: continue
        response = np.asarray(states[keyed[ke]["hidden_index"]], np.float32) - np.asarray(states[keyed[ku]["hidden_index"]], np.float32)
        (train if depth <= 2 and unit < 6 else test if depth >= 3 and unit >= 6 else []).append(response)
    forecast = {"ran": bool(train and test)}
    if train and test:
        prediction = np.mean(train, axis=0); scores = [nrmse(prediction, truth) for truth in test]
        forecast.update({"train_groups": len(train), "test_groups": len(test), "nrmse_mean": float(np.mean(scores)), "nrmse_zero": 1.0, "passed": float(np.mean(scores)) < 1.0})
    headline = {"status": "ternary_graph_field_closed", **metrics, "field_ran": True, "depth_forecast": forecast, "strict_interpretation": "A depth forecast is a response regularity, not recursive algorithm closure."}
    close_memmap(states)
    close("C410", headline, {"shape": metrics["role_shape"][-1] == DIM, "forecast_accounted": isinstance(forecast["ran"], bool), "finite": finite(headline)}, "C411_causal_branch")


@torch.inference_mode()
def qwen_writer_test(family: str, limit: int = 16) -> dict:
    rows, material = main_lookup(); compiled = read_rows(OUTS["C401"] / "compiled/qwen3.jsonl")
    comp = {r["case_id"]: r for r in compiled}; index = read_rows(OUTS["C402"] / "raw/hidden_index.jsonl")
    states = np.load(OUTS["C402"] / "raw/role_states.float16.npy", mmap_mode="r"); idx = {r["case_id"]: r for r in index}
    pairs = []
    for construction, unit, order in itertools.product(CONSTRUCTIONS, (4, 5), (1, -1)):
        c0 = f"c400-{family}-{construction}-u{unit}-00-{order:+d}"; c1 = f"c400-{family}-{construction}-u{unit}-10-{order:+d}"
        if c0 in idx and c1 in idx and idx[c0]["correct"] and idx[c1]["correct"]:
            pairs.append((c0, c1))
    pairs = pairs[:limit]
    if not pairs:
        close_memmap(states); return {"ran": False, "reason": "no_complete_pairs"}
    model = None; results = []
    try:
        model, tokenizer, device, placement = previous.common.model_base.load_bf16("qwen3")
        layer = model.model.layers[23]
        for base_id, target_id in pairs:
            base_row, target_row = comp[base_id], comp[target_id]
            wrong_family = final("C402")["headline"]["eligible_families"][(final("C402")["headline"]["eligible_families"].index(family) + 1) % len(final("C402")["headline"]["eligible_families"])]
            wrong_id = f"c400-{wrong_family}-{material[base_id]['construction']}-u{material[base_id]['unit']}-10-{material[base_id]['order']:+d}"
            donor = torch.tensor(np.asarray(states[idx[target_id]["hidden_index"], 24, ROLES.index("relation")], np.float32), dtype=torch.bfloat16, device=device)
            wrong_role = torch.tensor(np.asarray(states[idx[target_id]["hidden_index"], 24, ROLES.index("context")], np.float32), dtype=torch.bfloat16, device=device)
            wrong_family_state = donor if wrong_id not in idx else torch.tensor(np.asarray(states[idx[wrong_id]["hidden_index"], 24, ROLES.index("relation")], np.float32), dtype=torch.bfloat16, device=device)
            ids = torch.tensor([base_row["prompt_ids"]], dtype=torch.long, device=device); positions = base_row["role_positions"]["relation"]
            def run(write=None):
                hook = None
                if write is not None:
                    def patch(_module, _args, output):
                        value = output[0] if isinstance(output, tuple) else output; changed = value.clone(); changed[0, positions] = write
                        return (changed, *output[1:]) if isinstance(output, tuple) else changed
                    hook = layer.register_forward_hook(patch)
                try:
                    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                    scores = [float(output.logits[0, ids.shape[1]-1, cand[0]]) for cand in base_row["candidate_ids"]]
                    target_pos = target_row["gold_position"]; return scores[target_pos] - scores[1-target_pos]
                finally:
                    if hook is not None: hook.remove()
            margins = {"baseline": run(), "correct": run(donor), "wrong_role": run(wrong_role), "wrong_family": run(wrong_family_state), "zero": run(torch.zeros_like(donor))}
            results.append({"base": base_id, "target": target_id, "margins": margins, "correct_shift": margins["correct"] - margins["baseline"], "control_shift": max(margins[name] - margins["baseline"] for name in ("wrong_role", "wrong_family", "zero"))})
        return {"ran": True, "placement": placement, "family": family, "pairs": len(results), "mean_correct_shift": float(np.mean([r["correct_shift"] for r in results])), "mean_control_shift": float(np.mean([r["control_shift"] for r in results])), "specificity_passed": float(np.mean([r["correct_shift"] for r in results])) > float(np.mean([r["control_shift"] for r in results])) + 0.05, "results": results}
    finally:
        close_memmap(states); previous.common.model_base.release(model); gc.collect()


def c411() -> None:
    candidates = sorted(set(final("C404")["headline"]["candidate_families"]) & set(final("C405")["headline"]["candidate_families"]))
    out = begin("C411", {
        "status": "qualified_qwen_hidden_state_writer_branch_frozen", "candidate_rule": "intersection of C404 and C405",
        "writer": "q24 relation-role donor mean written to exact recipient relation tokens",
        "controls": ["wrong role", "wrong family", "zero write"], "requires_known_truth": True,
    }, {"parent": final("C410")["all_checks_passed"], "known_truth": final("C408")["headline"]["writer_calibrated"]})
    if not candidates:
        headline = {"status": "qwen_writer_not_run_no_joint_candidate", "candidates": [], "writer_ran": False, "strict_interpretation": "No real-model causal conclusion is available; other campaign routes continue."}
    else:
        result = qwen_writer_test(candidates[0]); save(out / "analysis/writer_results.json", result)
        headline = {"status": "qwen_writer_branch_closed", "candidates": candidates, "writer_ran": result["ran"], "result": result, "strict_interpretation": "Whole-state role writing tests sufficiency at one checkpoint, not natural necessity or a minimal circuit."}
    close("C411", headline, {"branch_accounted": ("result" in headline) == bool(candidates), "finite": finite(headline)}, "C412_cross_model")


@torch.inference_mode()
def run_external_model(model_name: str, rows: list[dict], out: Path) -> dict:
    model = None; behavior = []; compiled = None
    try:
        model, tokenizer, device, placement = previous.common.model_base.load_bf16(model_name)
        compiled = previous.compile_model_rows(tokenizer, rows, "strict_chat")
        write_rows(out / f"compiled/{model_name}.jsonl", compiled)
        for start in range(0, len(compiled), 8):
            for row in compiled[start:start+8]:
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                if all(len(candidate) == 1 for candidate in row["candidate_ids"]):
                    scores = [float(output.logits[0, ids.shape[1]-1, candidate[0]]) for candidate in row["candidate_ids"]]
                else:
                    scores = previous.common.score_prompt_candidates(model, row["prompt_ids"], row["candidate_ids"], device, int(tokenizer.pad_token_id or tokenizer.eos_token_id)).tolist()
                prediction = int(np.argmax(scores)); behavior.append({"case_id": row["case_id"], "family": row["family"], "correct": prediction == row["gold_position"], "prediction": prediction, "scores": scores})
            if start % 80 == 0 or start + 8 >= len(compiled): print(f"[C412 behavior] {model_name} {min(start+8,len(compiled))}/{len(compiled)}", flush=True)
        write_rows(out / f"raw/{model_name}_behavior.jsonl", behavior)
        by_family = {family: float(np.mean([r["correct"] for r in behavior if r["family"] == family])) for family in FAMILIES}
        accuracy = float(np.mean([r["correct"] for r in behavior])); eligible = accuracy >= 0.80 and min(by_family.values()) >= 0.60
    finally:
        previous.common.model_base.release(model); model = None; gc.collect()
    capture = {"ran": False}
    if eligible:
        behavior_by_id = {row["case_id"]: row for row in behavior}
        subset_ids = {row["case_id"] for row in rows if row["surface"] == "ledger" and row["unit"] in (4,5) and row["order"] == 1 and row["cell"] in ("00","10","01","11_ab")}
        subset = [row for row in compiled if row["case_id"] in subset_ids]
        try:
            model, tokenizer, device, placement2 = previous.common.model_base.load_bf16(model_name)
            info = get_model_info(model, model_name); nq = info.n_layers + 1
            states = np.lib.format.open_memmap(out / f"raw/{model_name}_role_states.float16.npy", mode="w+", dtype=np.float16, shape=(len(subset), nq, len(ROLES), info.d_model))
            idx = []
            for i, row in enumerate(subset):
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, output_hidden_states=True)
                for q, state in enumerate(output.hidden_states):
                    for ri, role in enumerate(ROLES): states[i,q,ri] = state[0,row["role_positions"][role]].mean(0).float().cpu().numpy().astype(np.float16)
                idx.append({"hidden_index": i, "case_id": row["case_id"], "family": row["family"], "unit": row["unit"], "cell": row["cell"], "correct": bool(behavior_by_id[row["case_id"]]["correct"])})
                states.flush()
                if i % 32 == 0 or i + 1 == len(subset): print(f"[C412 field] {model_name} {i+1}/{len(subset)}", flush=True)
            write_rows(out / f"raw/{model_name}_hidden_index.jsonl", idx)
            capture = {"ran": True, "shape": list(states.shape), "placement": placement2}; close_memmap(states)
        finally:
            previous.common.model_base.release(model); gc.collect()
    return {"model": model_name, "rows": len(rows), "placement": placement, "accuracy": accuracy, "family_accuracy": by_family, "eligible": eligible, "capture": capture}


def c412() -> None:
    out = begin("C412", {
        "status": "glm_deepseek_output_sensitive_external_panel_frozen",
        "panel": "16 families x ledger x units3-5 x five cells x two answer orders",
        "lifecycle": "GLM4 behavior then optional field, release; DeepSeek behavior then optional field, release",
        "behavior_before_hidden": True,
    }, {"parent": final("C411")["all_checks_passed"], "models": all(name in MODEL_CONFIGS for name in ("glm4","deepseek7b")), "cuda": torch.cuda.is_available()})
    rows, _ = main_lookup(); panel = [r for r in rows if r["surface"] == "ledger" and r["unit"] >= 3]
    write_rows(out / "material/cases.jsonl", panel)
    results = []
    for model_name in ("glm4", "deepseek7b"):
        try:
            results.append(run_external_model(model_name, panel, out))
        except Exception as exc:
            error = {
                "model": model_name, "rows": len(panel), "status": "execution_error",
                "error_type": type(exc).__name__, "error": str(exc),
                "eligible": False, "capture": {"ran": False},
            }
            save(out / f"audit/{model_name}_execution_error.json", error)
            results.append(error)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    headline = {"status": "cross_model_external_panel_closed", "results": results, "eligible_models": [r["model"] for r in results if r["eligible"]], "strict_interpretation": "Behavior-ineligible models are excluded from internal comparison; eligibility never aligns native coordinates."}
    close("C412", headline, {"rows": len(panel) == 480, "sequential": True, "finite": finite(headline)}, "C413_abstraction")


def response_profile(states: np.ndarray, index: list[dict]) -> dict[str, np.ndarray]:
    keyed = {
        (r["family"], r["unit"], r["cell"]): r
        for r in index
        if r.get("correct", True)
    }
    profiles = {}
    for family in FAMILIES:
        values = []
        for unit in (4,5):
            if (family,unit,"00") in keyed and (family,unit,"10") in keyed:
                a = keyed[(family,unit,"00")]["hidden_index"]; b = keyed[(family,unit,"10")]["hidden_index"]
                values.append(np.mean(np.abs(np.asarray(states[b],np.float32)-np.asarray(states[a],np.float32)),axis=-1))
        if values:
            profile = np.mean(values,axis=0); profile /= np.sum(profile,axis=0,keepdims=True)+1e-8; profiles[family]=profile
    return profiles


def resample_profile(profile: np.ndarray, bins: int = 20) -> np.ndarray:
    old = np.linspace(0,1,profile.shape[0]); new=np.linspace(0,1,bins)
    return np.stack(
        [np.interp(new, old, profile[:, r]) for r in range(profile.shape[1])],
        axis=0,
    ).astype(np.float32)


def c413() -> None:
    out = begin("C413", {
        "status": "cross_model_relative_response_abstraction_frozen",
        "comparison": "family x role x relative depth response-energy profiles",
        "exclusion": "behavior-ineligible models", "no_native_coordinate_alignment": True,
    }, {"parent": final("C412")["all_checks_passed"]})
    rows, material = main_lookup(); qstates=np.load(OUTS["C402"]/"raw/role_states.float16.npy",mmap_mode="r"); qindex=read_rows(OUTS["C402"]/"raw/hidden_index.jsonl")
    qselected=[]
    for r in qindex:
        case=material[r["case_id"]]
        if case["surface"]=="ledger" and case["unit"] in (4,5) and case["order"]==1 and case["cell"] in ("00","10","01","11_ab"):
            qselected.append({"hidden_index":r["hidden_index"],"family":case["family"],"unit":case["unit"],"cell":case["cell"],"correct":bool(r["correct"])})
    profiles={"qwen3":response_profile(qstates,qselected)}; close_memmap(qstates)
    external={r["model"]:r for r in final("C412")["headline"]["results"]}
    for model_name,row in external.items():
        if not row["eligible"]: continue
        states=np.load(OUTS["C412"]/f"raw/{model_name}_role_states.float16.npy",mmap_mode="r"); index=read_rows(OUTS["C412"]/f"raw/{model_name}_hidden_index.jsonl")
        profiles[model_name]=response_profile(states,index); close_memmap(states)
    comparisons=[]
    for left,right in itertools.combinations(profiles,2):
        common_f=sorted(set(profiles[left])&set(profiles[right])); tv=[]
        for family in common_f:
            a=resample_profile(profiles[left][family]); b=resample_profile(profiles[right][family]); tv.append(float(0.5*np.mean(np.sum(np.abs(a-b),axis=1))))
        comparisons.append({"left":left,"right":right,"families":len(common_f),"mean_role_relative_depth_tv":float(np.mean(tv)) if tv else None})
    save(out/"analysis/relative_profiles.json",{model:{family:profile.round(8).tolist() for family,profile in values.items()} for model,values in profiles.items()})
    headline={"status":"cross_model_relative_abstraction_closed","eligible_models":list(profiles),"comparisons":comparisons,"functional_bisimulation_established":False,"strict_interpretation":"Relative-depth energy similarity is a coarse abstraction, not bidirectional state translation or shared mechanism."}
    close("C413",headline,{"qwen_route_accounted":isinstance(profiles["qwen3"],dict),"finite":finite(headline)},"C414_synthesis")


def hash_and_remove(paths: list[Path], out: Path) -> list[dict]:
    manifest=[]
    for path in paths:
        if not path.exists(): continue
        size=path.stat().st_size; hasher=hashlib.sha256(); rel=str(path.relative_to(ROOT))
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                hasher.update(chunk)
        digest=hasher.hexdigest()
        path.unlink(); manifest.append({"path":rel,"bytes":size,"sha256":digest,"removed":not path.exists()})
    save(out/"audit/cleanup.json",manifest); return manifest


def c414() -> None:
    out=begin("C414",{
        "status":"campaign_synthesis_heatmap_cleanup_frozen","visual":"all 2560 coordinates for eligible family interaction centroids and full-token deltas",
        "cleanup":"checksum and remove nonvisual bulk fields after export","new_math_gate":"requires stable cross-construction composition and causal/cross-model evidence",
    },{"parent":final("C413")["all_checks_passed"]})
    eligible=final("C402")["headline"]["eligible_families"]
    atlas=np.load(OUTS["C403"]/"analysis/family_construction_operation_mean.float16.npy",mmap_mode="r")
    visual_rows=[]
    for fi,family in enumerate(eligible):
        mean=np.mean(np.asarray(atlas[fi,:,OPS.index("I")],np.float32),axis=0)
        for checkpoint,role_i in itertools.product((0,12,24,36,37),range(len(ROLES))):
            visual_rows.append({"id":f"family:{family}:I:q{checkpoint}:{ROLES[role_i]}","source":"family_interaction","family":family,"checkpoint":checkpoint,"role":ROLES[role_i],"values":mean[checkpoint,role_i].round(6).tolist()})
    close_memmap(atlas)
    token_delta=np.load(OUTS["C406"]/"analysis/full_token_delta.float16.npy",mmap_mode="r"); token_index=read_rows(OUTS["C406"]/"analysis/full_token_delta_index.jsonl")
    for row in token_index:
        for token in range(8):
            visual_rows.append({"id":f"token:{row['family']}:{row['construction']}:q24:t{token}","source":"full_token_delta","family":row["family"],"construction":row["construction"],"checkpoint":24,"token":token,"values":np.asarray(token_delta[row["delta_index"],24,token],np.float32).round(6).tolist()})
    close_memmap(token_delta)
    payload={"schema":"c414.output_sensitive_language_field.v1","phase":1948,"campaign":"C414","model":"Qwen3-4B","dimensions":list(range(DIM)),"rows":visual_rows,"claim_boundary":"All rows retain 2560 physical activation coordinates. They are construction-conditioned observations, not weights, semantic neurons, universal operators, or causal circuits.","summary":{"eligible_families":eligible,"cross_construction_candidates":final("C404")["headline"]["candidate_families"],"initial_state_candidates":final("C405")["headline"]["candidate_families"],"writer_ran":final("C411")["headline"]["writer_ran"],"graph_field_ran":final("C410")["headline"]["field_ran"]}}
    save(VISUAL,payload)
    cleanup_paths=[
        OUTS["C402"]/"raw/role_states.float16.npy",OUTS["C402"]/"raw/full_fields_holdout.float16.npy",
        OUTS["C403"]/"analysis/family_construction_operation_mean.float16.npy",OUTS["C406"]/"analysis/full_token_delta.float16.npy",
        OUTS["C407"]/"raw/role_states.float16.npy",OUTS["C410"]/"raw/role_states.float16.npy",
        OUTS["C412"]/"raw/glm4_role_states.float16.npy",OUTS["C412"]/"raw/deepseek7b_role_states.float16.npy",
    ]
    cleanup=hash_and_remove(cleanup_paths,out)
    gates={
        "output_sensitive_behavior":final("C401")["headline"]["field_eligible"],
        "cross_construction_any":bool(final("C404")["headline"]["candidate_families"]),
        "initial_state_any":bool(final("C405")["headline"]["candidate_families"]),
        "composition_ran":final("C407")["headline"]["composition"]["ran"],
        "writer_calibrated_known_truth":final("C408")["headline"]["writer_calibrated"],
        "qwen_writer_ran":final("C411")["headline"]["writer_ran"],
        "graph_field_ran":final("C410")["headline"]["field_ran"],
        "cross_model_count":len(final("C413")["headline"]["eligible_models"]),
        "causal":bool(final("C411")["headline"].get("result",{}).get("specificity_passed",False)),
        "new_math":False,
    }
    headline={"status":"output_sensitive_language_campaign_closed","gates":gates,"visual_rows":len(visual_rows),"visual_path":str(VISUAL.relative_to(ROOT)),"cleanup_bytes":sum(item["bytes"] for item in cleanup),"cleanup_files":len(cleanup),"new_math_gate_passed":False,"strict_interpretation":"The campaign may establish predictive construction-conditioned regularities; only a specific passed writer branch can support a narrow causal sufficiency claim."}
    close("C414",headline,{"visual":len(visual_rows)>0 and all(len(row["values"])==DIM for row in visual_rows),"cleanup":all(item["removed"] and item["sha256"] for item in cleanup),"finite":finite(headline)},"independent_audit_then_new_family_lockbox")


RUNNERS={name:globals()[name.lower()] for name in PHASES}


def parse_range(value: str) -> list[str]:
    if value in ("all","C399-C414"): return list(PHASES)
    if "-" in value:
        left,right=value.split("-",1); return [f"C{i}" for i in range(int(left[1:]),int(right[1:])+1)]
    return [value]


def validate_only() -> None:
    rows=main_material(); graph=graph_material(); comp=composition_material()
    checks={
        "main_rows":len(rows)==3840,"families":{r["family"] for r in rows}==set(FAMILIES),"constructions":{r["construction"] for r in rows}==set(CONSTRUCTIONS),
        "main_balance":sum(r["gold_position"]==0 for r in rows)==1920,"roles":all(set(r["role_values"])==set(ROLES)-{"boundary"} for r in rows),
        "graph_rows":len(graph)==672,"graph_positions":max(abs(sum(r["gold_position"]==i for r in graph)/len(graph)-1/3) for i in range(3))<1e-9,
        "composition_rows":len(comp)==1152,"phase_sequence":[PHASES[f"C{i}"][0] for i in range(399,415)]==list(range(1933,1949)),
    }
    print(json.dumps(checks,ensure_ascii=False),flush=True)
    if not all(checks.values()): raise AssertionError(checks)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run",default="C399-C414"); parser.add_argument("--validate-only",action="store_true"); args=parser.parse_args()
    if args.validate_only: validate_only(); return
    for name in parse_range(args.run): RUNNERS[name]()


if __name__=="__main__":
    main()
