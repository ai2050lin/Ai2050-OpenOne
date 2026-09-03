#!/usr/bin/env python3
"""C434-C445 guarded full-coordinate response graph campaign.

The campaign keeps embedding and HiddenState checkpoints as the only neural
objects.  Attention, MLP internals, model weights, PCA, and Top-K coordinate
selection are outside the registered object.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c445_guarded_response_graph.json"
sys.path.insert(0, str(TESTS))

import phase1933_c399_c414_output_sensitive_language_campaign as old
import phase1960_c426_c433_axis_lockbox_campaign as axis_old
import phase1844_c310_c335_dual_axis_common as common
import phase1797_c263_c272_state_operator_common as family_base


PHASES = {
    f"C{campaign}": (1968 + campaign - 434, slug)
    for campaign, slug in (
        (434, "evidence_adjudication_and_guarded_graph_contract"),
        (435, "fresh_language_graph_material_and_zero_models"),
        (436, "qwen_multifamily_behavior_qualification"),
        (437, "qualified_full_coordinate_and_token_field"),
        (438, "guarded_signed_event_hypergraph_discovery"),
        (439, "unseen_lexicon_construction_event_prediction"),
        (440, "typed_state_distance_tournament"),
        (441, "repaired_binary_graph_behavior_interface"),
        (442, "qualified_graph_field_and_depth_prediction"),
        (443, "expanded_known_truth_writer_calibration"),
        (444, "registered_cross_model_functional_topology"),
        (445, "campaign_synthesis_visual_cleanup_and_audit"),
    )
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

DIM = 2560
CHECKPOINTS = 38
ROLES = common.ROLES
FIELD_WIDTH = 192
BROAD_FAMILIES = (
    "attitude_event", "type_graph", "contrast", "translation",
    "comparison", "nested_attitude", "agent_patient_voice", "negation_scope",
)
AXIS_FAMILIES = ("attitude_event", "nested_attitude", "negation_scope")
AXES = ("outer", "attitude")
CONSTRUCTIONS = ("chronicle", "dispatch", "archive")
BROAD_CELLS = ("00", "10", "01", "11")
NEW_UNITS = (
    {"p": "Liora", "s": "Merek", "o": "Neris", "obj": "pomelo", "other": "compass", "node": "zefa", "parent": "clasa", "wrong": "clasa_alt", "event": "survey"},
    {"p": "Orin", "s": "Pavia", "o": "Quill", "obj": "radicchio", "other": "quadrant", "node": "zefb", "parent": "clasb", "wrong": "clasb_alt", "event": "inspection"},
    {"p": "Ravin", "s": "Sela", "o": "Torin", "obj": "tamarind", "other": "chronometer", "node": "zefc", "parent": "clasc", "wrong": "clasc_alt", "event": "review"},
    {"p": "Uma", "s": "Varen", "o": "Willa", "obj": "rutabaga", "other": "clinometer", "node": "zefd", "parent": "clasd", "wrong": "clasd_alt", "event": "audit"},
    {"p": "Xerin", "s": "Yara", "o": "Ziven", "obj": "persimmon", "other": "odometer", "node": "zefe", "parent": "clase", "wrong": "clase_alt", "event": "briefing"},
    {"p": "Aven", "s": "Bria", "o": "Cerin", "obj": "kohlrabi", "other": "altimeter", "node": "zeff", "parent": "clasf", "wrong": "clasf_alt", "event": "inventory"},
)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        "phase": PHASES[name][0], "campaign": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer_sha256": producer_hash(), **protocol,
    })
    save(out / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    return out


def finite(value: Any) -> bool:
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
    result = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "checks": final_checks, "all_checks_passed": all(final_checks.values()),
        "headline": headline, "next_authorization": authorization,
    }
    save(out / "analysis/final.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def nrmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, np.float32)
    truth = np.asarray(truth, np.float32)
    return float(np.linalg.norm(prediction - truth) / (np.linalg.norm(truth) + 1e-8))


def wrap(construction: str, target: str, noise: str, question: str) -> str:
    if construction == "chronicle":
        return f"A chronicle records this relevant entry: {target} It also contains an unrelated note: {noise} Using only the chronicle, {question}"
    if construction == "dispatch":
        return f"A dispatch gives two items. Relevant item: {target} Separate item: {noise} Based only on the dispatch, {question}"
    if construction == "archive":
        return f"An archive lists {target} Elsewhere it lists {noise} From the archive alone, {question}"
    raise KeyError(construction)


def partition(unit: int) -> str:
    return "discovery" if unit < 3 else "confirmation" if unit < 5 else "lockbox"


def broad_material() -> list[dict]:
    rows = []
    original = old.UNITS
    old.UNITS = NEW_UNITS
    try:
        for family, construction, unit, cell, order in itertools.product(
            BROAD_FAMILIES, CONSTRUCTIONS, range(len(NEW_UNITS)), BROAD_CELLS, (1, -1)
        ):
            a, b = {"00": (0, 0), "10": (1, 0), "01": (0, 1), "11": (1, 1)}[cell]
            case = old.family_statement(family, unit, a, b)
            core = wrap(construction, case["target"], case["noise"], case["question"])
            choices, gold = family_base.options(case["correct"], case["wrong"], order)
            rows.append({
                "case_id": f"c435-broad-{family}-{construction}-u{unit}-{cell}-{order:+d}",
                "panel": "broad_pair", "family": family, "surface": construction,
                "construction": construction, "unit": unit, "cell": cell,
                "factor_a": a, "factor_b": b, "order": order, "partition": partition(unit),
                "gold_position": gold, "correct_answer": case["correct"], "wrong_answer": case["wrong"],
                "prompt_core": core, "prompt": f"{core} {choices}. Reply with only A or B.",
                "free_prompt": f"{core} Answer with only Yes or No.", "role_values": case["roles"],
                "semantic_graph": {"family": family, "statement_state": a, "query_state": b},
            })
    finally:
        old.UNITS = original
    return rows


def axis_material() -> list[dict]:
    rows = []
    masks = tuple(range(8))
    for family, construction, unit_i, mask, axis, order in itertools.product(
        AXIS_FAMILIES, CONSTRUCTIONS, range(len(NEW_UNITS)), masks, AXES, (1, -1)
    ):
        d, a, b = ((mask >> bit) & 1 for bit in range(3))
        unit = NEW_UNITS[unit_i]
        p, s, o, obj = unit["p"], unit["s"], unit["o"], unit["obj"]
        attitude = ("likes", "dislikes")[a]
        event = ("ate", "did not eat")[b]
        outer = {
            "attitude_event": ("reported", "denied"),
            "nested_attitude": ("believes", "doubts"),
            "negation_scope": ("affirmed", "denied"),
        }[family][d]
        target = f"{o} {outer} that {p} {attitude} the event in which {s} {event} the {obj}."
        if axis == "outer":
            baseline = "report" if family == "attitude_event" else "believe" if family == "nested_attitude" else "affirm"
            question = f"Did {o} {baseline} that {p} {attitude} the event in which {s} {event} the {obj}?"
            relation = outer
            truth = d == 0
        else:
            question = f"According to the relevant statement, does {p} like the event in which {s} {event} the {obj}?"
            relation = attitude
            truth = a == 0
        noise = f"Separately, {s} catalogued the {unit['other']} for {p}."
        core = wrap(construction, target, noise, question)
        correct, wrong = ("Yes", "No") if truth else ("No", "Yes")
        choices, gold = family_base.options(correct, wrong, order)
        cell = f"{d}{a}{b}"
        rows.append({
            "case_id": f"c435-axis-{family}-{construction}-u{unit_i}-{cell}-{axis}-{order:+d}",
            "panel": "axis_composition", "family": family, "surface": construction,
            "construction": construction, "unit": unit_i, "cell": cell, "mask": mask,
            "query_axis": axis, "order": order, "partition": partition(unit_i),
            "gold_position": gold, "correct_answer": correct, "wrong_answer": wrong,
            "prompt_core": core, "prompt": f"{core} {choices}. Reply with only A or B.",
            "free_prompt": f"{core} Answer with only Yes or No.",
            "role_values": {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": obj},
            "semantic_graph": {"family": family, "outer": d, "attitude": a, "event": b, "query_axis": axis},
        })
    return rows


def all_material() -> list[dict]:
    return broad_material() + axis_material()


def material_lookup() -> tuple[list[dict], dict[str, dict]]:
    rows = read_rows(OUTS["C435"] / "material/cases.jsonl")
    return rows, {row["case_id"]: row for row in rows}


def behavior_metrics(behavior: list[dict], material: dict[str, dict]) -> dict:
    held = [row for row in behavior if material[row["case_id"]]["partition"] != "discovery"]
    broad = {}
    eligible_broad = []
    for family in BROAD_FAMILIES:
        subset = [row for row in held if material[row["case_id"]]["panel"] == "broad_pair" and row["family"] == family]
        construction = {
            value: float(np.mean([row["correct"] for row in subset if row["surface"] == value]))
            for value in CONSTRUCTIONS
        }
        score = float(np.mean([row["correct"] for row in subset]))
        eligible = score >= 0.68 and min(construction.values()) >= 0.58
        broad[family] = {"accuracy": score, "construction_accuracy": construction, "eligible": eligible}
        if eligible:
            eligible_broad.append(family)
    axes = {}
    eligible_axes = []
    for axis in AXES:
        subset = [row for row in held if material[row["case_id"]].get("query_axis") == axis]
        family_scores = {family: float(np.mean([row["correct"] for row in subset if row["family"] == family])) for family in AXIS_FAMILIES}
        construction = {value: float(np.mean([row["correct"] for row in subset if row["surface"] == value])) for value in CONSTRUCTIONS}
        score = float(np.mean([row["correct"] for row in subset]))
        eligible = score >= 0.70 and min(family_scores.values()) >= 0.63 and min(construction.values()) >= 0.63
        axes[axis] = {"accuracy": score, "family_accuracy": family_scores, "construction_accuracy": construction, "eligible": eligible}
        if eligible:
            eligible_axes.append(axis)
    return {
        "overall": float(np.mean([row["correct"] for row in behavior])),
        "heldout": float(np.mean([row["correct"] for row in held])),
        "broad": broad, "axes": axes,
        "eligible_broad": eligible_broad, "eligible_axes": eligible_axes,
    }


def c434() -> None:
    audit = load(ROOT / "tests/glm5/result/phase1967_c433_axis_campaign_synthesis_heatmap_cleanup/audit/independent_audit.json")
    out = begin("C434", {
        "status": "guarded_response_graph_master_contract_frozen",
        "registered_routes": ["fresh_multifamily", "signed_event_graph", "typed_distance", "repaired_graph", "known_truth_writer", "cross_model_topology"],
        "research_order": "full-field observation then rule discovery then prospective prediction then causal calibration",
        "route_policy": "failure eliminates one route only; all other registered routes continue",
        "neural_object": "embedding plus every post-block pre-norm HiddenState plus final norm, all roles/tokens and all coordinates",
        "excluded": ["attention", "MLP internals", "weights", "PCA", "Top-K", "coordinate pruning"],
    }, {"parent": audit["all_checks_passed"], "phase_continuity": PHASES["C434"][0] == 1968})
    headline = {
        "status": "evidence_adjudication_closed",
        "retained": ["axis-conditioned late-field regularity", "high-order mean candidates", "distributed low-amplitude coordinates matter"],
        "corrected": [
            "C430 did not confirm a high-order mechanism; it retained three mean candidates only",
            "full-coordinate storage is not mechanism recovery",
            "a language family registry is an experimental scaffold, not a neural ontology",
            "new mathematics is a possibility, not an observed necessity",
        ],
        "strict_interpretation": "The campaign searches for guarded signed response rules and keeps explicit abstention when they do not transfer.",
    }
    close("C434", headline, {"corrections": len(headline["corrected"]) == 4}, "C435_material")


def c435() -> None:
    out = begin("C435", {
        "status": "fresh_language_graph_material_frozen",
        "design": "8 pair families plus 3 composition families across 3 new constructions and 6 new lexicons",
        "partitions": {"discovery": [0, 1, 2], "confirmation": [3, 4], "lockbox": [5]},
        "semantic_uniqueness": "exact registered truth table and role-span occurrence",
        "naturalness": "controlled-English grammar audit only; independent human review absent",
        "zero_models": ["always A", "always B", "panel majority", "family majority", "surface majority"],
    }, {"parent": final("C434")["all_checks_passed"]})
    rows = all_material()
    write_rows(out / "material/cases.jsonl", rows)
    zero = {"always_a": float(np.mean([r["gold_position"] == 0 for r in rows])), "always_b": float(np.mean([r["gold_position"] == 1 for r in rows]))}
    for key in ("panel", "family", "surface"):
        correct = 0
        for value in sorted({r[key] for r in rows}):
            subset = [r for r in rows if r[key] == value]
            guess = int(np.mean([r["gold_position"] for r in subset]) >= 0.5)
            correct += sum(r["gold_position"] == guess for r in subset)
        zero[f"{key}_majority"] = correct / len(rows)
    role_occurrence = all(all(str(value) in row["prompt_core"] for value in row["role_values"].values()) for row in rows)
    headline = {
        "status": "fresh_language_graph_material_closed", "rows": len(rows),
        "panel_counts": {panel: sum(r["panel"] == panel for r in rows) for panel in ("broad_pair", "axis_composition")},
        "partition_counts": {part: sum(r["partition"] == part for r in rows) for part in ("discovery", "confirmation", "lockbox")},
        "zero_model_accuracies": zero, "role_occurrence": role_occurrence,
        "material_eligible": max(zero.values()) <= 0.51 and role_occurrence,
        "human_naturalness_review": False,
        "strict_interpretation": "Exact balance and role occurrence do not establish natural-language coverage.",
    }
    save(out / "material/language_family_graph.json", {
        "nodes": sorted({r["family"] for r in rows}),
        "operations": ["statement_state", "query_state", "outer", "attitude", "event"],
        "constructions": list(CONSTRUCTIONS), "axes": list(AXES),
    })
    close("C435", headline, {"rows": len(rows) == 2880, "zero": max(zero.values()) <= 0.51, "roles": role_occurrence}, "C436_behavior")


def c436() -> None:
    out = begin("C436", {
        "status": "qwen_multifamily_behavior_frozen", "model": "Qwen3-4B BF16 CUDA",
        "gates": {"broad_heldout": 0.68, "broad_construction": 0.58, "axis_heldout": 0.70, "axis_family_and_construction": 0.63},
        "hidden_state_policy": "none", "qualification": "per family and per axis",
    }, {"parent": final("C435")["all_checks_passed"], "material": final("C435")["headline"]["material_eligible"], "cuda": torch.cuda.is_available()})
    rows, material = material_lookup()
    tokenizer = axis_old.base.parent.fresh.tokenizer_qwen()
    compiled = family_base.compile_qwen(tokenizer, rows)
    write_rows(out / "compiled/qwen3.jsonl", compiled)
    run = axis_old.base.parent.previous.qwen_behavior(rows, compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl")
    metrics = behavior_metrics(behavior, material)
    headline = {
        "status": "qwen_multifamily_behavior_closed", **run, **metrics,
        "field_authorized": bool(metrics["eligible_broad"] or metrics["eligible_axes"]),
        "strict_interpretation": "Only qualified family/axis strata authorize HiddenState interpretation; failed strata remain behavioral observations.",
    }
    close("C436", headline, {"rows": len(behavior) == len(rows), "no_hidden": not (out / "raw/role_states.float16.npy").exists(), "finite": finite(headline)}, "C437_field")


def c437() -> None:
    eligible_broad = set(final("C436")["headline"]["eligible_broad"])
    eligible_axes = set(final("C436")["headline"]["eligible_axes"])
    out = begin("C437", {
        "status": "qualified_full_coordinate_field_frozen",
        "archive": "embedding plus 36 block outputs plus final norm x six roles x all 2560 coordinates",
        "all_token_subset": "ten frozen lockbox cases at all 38 checkpoints, 192 right-padded tokens and all coordinates",
        "selection": "behavior-qualified strata and answer order +1; no coordinate filtering",
        "no_pca_topk": True,
    }, {"parent": final("C436")["all_checks_passed"]})
    if not (eligible_broad or eligible_axes):
        close("C437", {"status": "field_not_run_no_qualified_stratum", "field_ran": False, "strict_interpretation": "No internal result."}, {"route_accounted": True}, "C438_hypergraph")
        return
    rows, _ = material_lookup()
    compiled = read_rows(OUTS["C436"] / "compiled/qwen3.jsonl")
    selected = []
    for row, comp in zip(rows, compiled):
        qualified = (row["panel"] == "broad_pair" and row["family"] in eligible_broad) or (row["panel"] == "axis_composition" and row.get("query_axis") in eligible_axes)
        if qualified and row["order"] == 1:
            selected.append((row, comp))
    rows2 = [row for row, _ in selected]
    compiled2 = [comp for _, comp in selected]
    def full_selector(row: dict) -> bool:
        if row["partition"] != "lockbox" or row["surface"] != "archive":
            return False
        if row["panel"] == "broad_pair":
            return row["family"] in BROAD_FAMILIES[:4] and row["cell"] == "10"
        return row["query_axis"] == "outer" and row["cell"] in ("000", "100")
    run = common.batch_capture_qwen(rows2, compiled2, out, full_selector=full_selector, batch_size=8, field_width=FIELD_WIDTH)
    headline = {
        "status": "qualified_full_coordinate_field_closed", **run, "field_ran": True,
        "eligible_broad": sorted(eligible_broad), "eligible_axes": sorted(eligible_axes),
        "strict_interpretation": "The archive preserves physical activation values, not model parameters or a decoded circuit.",
    }
    close("C437", headline, {"shape": run["role_shape"][1:] == [38, 6, 2560], "full": run["full_shape"][-1] == 2560, "finite": finite(headline)}, "C438_hypergraph")


def response_records(index: list[dict], material: dict[str, dict]) -> list[dict]:
    keyed = {row["case_id"]: row for row in index}
    records = []
    for family in BROAD_FAMILIES:
        for surface, unit in itertools.product(CONSTRUCTIONS, range(len(NEW_UNITS))):
            for op, right_cell in (("statement", "10"), ("query", "01")):
                left_id = f"c435-broad-{family}-{surface}-u{unit}-00-+1"
                right_id = f"c435-broad-{family}-{surface}-u{unit}-{right_cell}-+1"
                if left_id in keyed and right_id in keyed:
                    records.append({"group": f"broad:{family}:{op}", "family": family, "operation": op, "surface": surface, "unit": unit, "partition": partition(unit), "context": 0, "left": keyed[left_id]["hidden_index"], "right": keyed[right_id]["hidden_index"]})
    for family in AXIS_FAMILIES:
        for surface, unit, axis in itertools.product(CONSTRUCTIONS, range(len(NEW_UNITS)), AXES):
            for op, bit in (("outer", 0), ("attitude", 1)):
                for context in range(4):
                    other_bits = [i for i in range(3) if i != bit]
                    base_mask = sum(((context >> j) & 1) << other_bit for j, other_bit in enumerate(other_bits))
                    right_mask = base_mask | (1 << bit)
                    left_cell, right_cell = f"{base_mask:03b}"[::-1], f"{right_mask:03b}"[::-1]
                    left_id = f"c435-axis-{family}-{surface}-u{unit}-{left_cell}-{axis}-+1"
                    right_id = f"c435-axis-{family}-{surface}-u{unit}-{right_cell}-{axis}-+1"
                    if left_id in keyed and right_id in keyed:
                        records.append({"group": f"axis:{family}:{axis}:{op}", "family": family, "operation": op, "axis": axis, "surface": surface, "unit": unit, "partition": partition(unit), "context": context, "left": keyed[left_id]["hidden_index"], "right": keyed[right_id]["hidden_index"]})
    return records


def c438() -> None:
    out = begin("C438", {
        "status": "guarded_signed_event_hypergraph_discovery_frozen",
        "event": "checkpoint x role x coordinate signed response and residual interval",
        "guard": "panel, family, operation, query axis, construction class and initial response",
        "transition": "per-coordinate affine map from response at q to response at q+1",
        "fit": "elementary least squares independently at every physical coordinate",
        "no_coordinate_selection": True,
    }, {"parent": final("C437")["all_checks_passed"]})
    if not final("C437")["headline"]["field_ran"]:
        close("C438", {"status": "hypergraph_not_run_no_field", "ran": False}, {"route_accounted": True}, "C439_prediction")
        return
    states = np.load(OUTS["C437"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C437"] / "raw/hidden_index.jsonl")
    _, material = material_lookup()
    records = response_records(index, material)
    groups = sorted({row["group"] for row in records})
    shape = (len(groups), CHECKPOINTS - 1, len(ROLES), DIM)
    arrays = {
        name: np.lib.format.open_memmap(out / f"analysis/{name}.float16.npy", mode="w+", dtype=np.float16, shape=shape)
        for name in ("slope", "intercept", "mean_next", "residual_low", "residual_high")
    }
    counts = {}
    sign_agreement = []
    for gi, group in enumerate(groups):
        train = [row for row in records if row["group"] == group and row["partition"] == "discovery" and row["surface"] != "archive"]
        counts[group] = len(train)
        for q in range(CHECKPOINTS - 1):
            for role in range(len(ROLES)):
                x = np.stack([np.asarray(states[row["right"], q, role], np.float32) - np.asarray(states[row["left"], q, role], np.float32) for row in train])
                y = np.stack([np.asarray(states[row["right"], q + 1, role], np.float32) - np.asarray(states[row["left"], q + 1, role], np.float32) for row in train])
                xm, ym = x.mean(0), y.mean(0)
                centered = x - xm
                slope = np.sum(centered * (y - ym), axis=0) / (np.sum(centered * centered, axis=0) + 1e-6)
                intercept = ym - slope * xm
                residual = y - (x * slope + intercept)
                arrays["slope"][gi, q, role] = np.clip(slope, -64, 64).astype(np.float16)
                arrays["intercept"][gi, q, role] = intercept.astype(np.float16)
                arrays["mean_next"][gi, q, role] = ym.astype(np.float16)
                arrays["residual_low"][gi, q, role] = residual.min(0).astype(np.float16)
                arrays["residual_high"][gi, q, role] = residual.max(0).astype(np.float16)
                agreement = np.maximum(np.mean(y >= 0, axis=0), np.mean(y < 0, axis=0))
                sign_agreement.append(float(np.mean(agreement)))
        for value in arrays.values():
            value.flush()
        print(f"[C438] {gi + 1}/{len(groups)} {group}", flush=True)
    for value in arrays.values():
        close_memmap(value)
    close_memmap(states)
    write_rows(out / "analysis/response_records.jsonl", records)
    save(out / "analysis/groups.json", groups)
    headline = {
        "status": "guarded_signed_event_hypergraph_discovery_closed", "ran": True,
        "groups": len(groups), "records": len(records), "training_counts": counts,
        "coefficient_shape": list(shape), "mean_coordinate_sign_agreement": float(np.mean(sign_agreement)),
        "strict_interpretation": "The fitted edges are discovery estimators. They become empirical rules only if they beat frozen controls on unseen lexicons and construction.",
    }
    close("C438", headline, {"groups": len(groups) > 0, "training": min(counts.values()) >= 6, "shape": shape[-1] == DIM, "finite": finite(headline)}, "C439_prediction")


def accumulate_transition_metrics(states: np.ndarray, records: list[dict], gi: int, group: str, arrays: dict[str, np.ndarray], wrong_gi: int) -> dict:
    test = [row for row in records if row["group"] == group and row["surface"] == "archive" and row["partition"] in ("confirmation", "lockbox")]
    sums = {name: 0.0 for name in ("affine", "identity", "zero", "mean", "wrong", "truth")}
    signs = {name: [0, 0] for name in ("affine", "identity", "zero", "mean", "wrong")}
    covered = total = 0
    for row in test:
        for q in range(CHECKPOINTS - 1):
            for role in range(len(ROLES)):
                x = np.asarray(states[row["right"], q, role], np.float32) - np.asarray(states[row["left"], q, role], np.float32)
                y = np.asarray(states[row["right"], q + 1, role], np.float32) - np.asarray(states[row["left"], q + 1, role], np.float32)
                slope = np.asarray(arrays["slope"][gi, q, role], np.float32)
                intercept = np.asarray(arrays["intercept"][gi, q, role], np.float32)
                predictions = {
                    "affine": slope * x + intercept,
                    "identity": x,
                    "zero": np.zeros_like(x),
                    "mean": np.asarray(arrays["mean_next"][gi, q, role], np.float32),
                    "wrong": np.asarray(arrays["slope"][wrong_gi, q, role], np.float32) * x + np.asarray(arrays["intercept"][wrong_gi, q, role], np.float32),
                }
                sums["truth"] += float(np.sum(y * y))
                active = np.abs(y) > 1e-3
                for name, pred in predictions.items():
                    sums[name] += float(np.sum((pred - y) ** 2))
                    signs[name][0] += int(np.sum((pred[active] >= 0) == (y[active] >= 0)))
                    signs[name][1] += int(np.sum(active))
                residual = y - predictions["affine"]
                lo = np.asarray(arrays["residual_low"][gi, q, role], np.float32) - 1e-3
                hi = np.asarray(arrays["residual_high"][gi, q, role], np.float32) + 1e-3
                covered += int(np.sum((residual >= lo) & (residual <= hi)))
                total += residual.size
    denom = math.sqrt(sums["truth"]) + 1e-8
    return {
        "test_records": len(test),
        "nrmse": {name: math.sqrt(value) / denom for name, value in sums.items() if name != "truth"},
        "active_sign_accuracy": {name: (correct / count if count else None) for name, (correct, count) in signs.items()},
        "residual_interval_coverage": covered / max(total, 1),
    }


def c439() -> None:
    out = begin("C439", {
        "status": "unseen_lexicon_construction_event_prediction_frozen",
        "test": "archive construction and confirmation/lockbox lexicons unseen by fit",
        "controls": ["identity", "zero", "discovery mean", "wrong guarded family"],
        "pass": "affine NRMSE beats all controls, active sign accuracy beats all controls, residual interval coverage >=0.70",
        "active_coordinate": "absolute target response >1e-3; threshold frozen before reveal",
    }, {"parent": final("C438")["all_checks_passed"]})
    if not final("C438")["headline"]["ran"]:
        close("C439", {"status": "prediction_not_run_no_hypergraph", "ran": False, "candidate_groups": []}, {"route_accounted": True}, "C440_distance")
        return
    states = np.load(OUTS["C437"] / "raw/role_states.float16.npy", mmap_mode="r")
    records = read_rows(OUTS["C438"] / "analysis/response_records.jsonl")
    groups = load(OUTS["C438"] / "analysis/groups.json")
    arrays = {name: np.load(OUTS["C438"] / f"analysis/{name}.float16.npy", mmap_mode="r") for name in ("slope", "intercept", "mean_next", "residual_low", "residual_high")}
    results = []
    for gi, group in enumerate(groups):
        wrong_gi = (gi + 1) % len(groups)
        metric = accumulate_transition_metrics(states, records, gi, group, arrays, wrong_gi)
        n = metric["nrmse"]; s = metric["active_sign_accuracy"]
        metric["group"] = group
        metric["passed"] = metric["test_records"] > 0 and n["affine"] < min(n[k] for k in ("identity", "zero", "mean", "wrong")) and s["affine"] is not None and s["affine"] > max(s[k] or 0 for k in ("identity", "zero", "mean", "wrong")) and metric["residual_interval_coverage"] >= 0.70
        results.append(metric)
    write_rows(out / "analysis/group_metrics.jsonl", results)
    for value in arrays.values():
        close_memmap(value)
    close_memmap(states)
    candidates = [row["group"] for row in results if row["passed"]]
    headline = {
        "status": "unseen_event_prediction_closed", "ran": True, "groups": len(results),
        "candidate_groups": candidates, "candidate_count": len(candidates),
        "mean_affine_nrmse": float(np.mean([r["nrmse"]["affine"] for r in results])),
        "mean_best_control_nrmse": float(np.mean([min(r["nrmse"][k] for k in ("identity", "zero", "mean", "wrong")) for r in results])),
        "strict_interpretation": "A passing group is a local one-checkpoint prediction rule, not an autonomous full-trajectory simulator or causal circuit.",
    }
    close("C439", headline, {"accounting": len(results) == len(groups), "finite": finite(headline)}, "C440_distance")


def field_distance(target: np.ndarray, donor: np.ndarray, method: str, scale: np.ndarray | None = None) -> float:
    a, b = np.asarray(target, np.float32), np.asarray(donor, np.float32)
    if method == "euclidean":
        return float(np.mean((a - b) ** 2))
    if method == "cosine":
        return float(1.0 - np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    if method == "diagonal_standardized":
        return float(np.mean(((a - b) / (scale + 1e-3)) ** 2))
    if method == "signed_event":
        active = (np.abs(a) > 1e-3) | (np.abs(b) > 1e-3)
        return float(np.mean(np.signbit(a[active]) != np.signbit(b[active]))) if np.any(active) else 0.0
    if method == "typed_checkpoint_role":
        norms = np.sqrt(np.mean((a - b) ** 2, axis=-1))
        base = np.sqrt(np.mean(a * a, axis=-1)) + 1e-3
        return float(np.mean(norms / base))
    if method == "absolute_hellinger_exploratory":
        pa = np.abs(a).ravel(); pb = np.abs(b).ravel(); pa /= pa.sum() + 1e-8; pb /= pb.sum() + 1e-8
        return float(np.mean((np.sqrt(pa) - np.sqrt(pb)) ** 2))
    raise KeyError(method)


def c440() -> None:
    out = begin("C440", {
        "status": "typed_state_distance_tournament_frozen",
        "methods": ["euclidean", "cosine", "diagonal_standardized", "signed_event", "typed_checkpoint_role", "absolute_hellinger_exploratory"],
        "target": "one frozen archive lockbox response per guarded group",
        "donors": "discovery lexicons in chronicle/dispatch",
        "prediction": "nearest donor response compared with discovery mean response",
        "claim_boundary": "diagonal standardized is not full Mahalanobis; Hellinger is exploratory because activation magnitudes are not probabilities",
    }, {"parent": final("C439")["all_checks_passed"]})
    if not final("C438")["headline"]["ran"]:
        close("C440", {"status": "distance_not_run_no_field", "ran": False}, {"route_accounted": True}, "C441_graph")
        return
    states = np.load(OUTS["C437"] / "raw/role_states.float16.npy", mmap_mode="r")
    records = read_rows(OUTS["C438"] / "analysis/response_records.jsonl")
    methods = ["euclidean", "cosine", "diagonal_standardized", "signed_event", "typed_checkpoint_role", "absolute_hellinger_exploratory"]
    results = []
    for group in sorted({row["group"] for row in records}):
        donors = [row for row in records if row["group"] == group and row["partition"] == "discovery" and row["surface"] != "archive"]
        targets = [row for row in records if row["group"] == group and row["partition"] == "lockbox" and row["surface"] == "archive" and row["context"] == 0]
        if not donors or not targets:
            continue
        donor_base = [np.asarray(states[row["left"]], np.float32) for row in donors]
        donor_response = [np.asarray(states[row["right"]], np.float32) - np.asarray(states[row["left"]], np.float32) for row in donors]
        mean_response = np.mean(donor_response, axis=0)
        scale = np.std(np.stack(donor_base), axis=0)
        for target_row in targets[:1]:
            target_base = np.asarray(states[target_row["left"]], np.float32)
            truth = np.asarray(states[target_row["right"]], np.float32) - target_base
            mean_error = nrmse(mean_response, truth)
            for method in methods:
                distances = [field_distance(target_base, donor, method, scale) for donor in donor_base]
                choice = int(np.argmin(distances))
                error = nrmse(donor_response[choice], truth)
                results.append({"group": group, "method": method, "donor_index": choice, "distance": distances[choice], "donor_nrmse": error, "mean_nrmse": mean_error, "gain_over_mean": mean_error - error})
        del donor_base, donor_response, mean_response, scale
        gc.collect()
        print(f"[C440] {group}", flush=True)
    close_memmap(states)
    write_rows(out / "analysis/tournament.jsonl", results)
    summary = {}
    for method in methods:
        subset = [r for r in results if r["method"] == method]
        summary[method] = {"mean_gain": float(np.mean([r["gain_over_mean"] for r in subset])), "wins": sum(r["gain_over_mean"] > 0 for r in subset), "groups": len(subset)}
    candidates = [method for method, row in summary.items() if row["mean_gain"] > 0 and row["wins"] >= math.ceil(row["groups"] / 2)]
    headline = {
        "status": "typed_state_distance_tournament_closed", "ran": True, "summary": summary,
        "candidate_methods": candidates,
        "strict_interpretation": "Nearest-state success would identify an effective retrieval metric only; it would not establish manifold geometry or causality.",
    }
    close("C440", headline, {"methods": set(summary) == set(methods), "finite": finite(headline)}, "C441_graph")


GRAPH_UNITS = tuple({"root": f"nax{chr(97+i)}", "mid1": f"bir{chr(97+i)}", "mid2": f"cav{chr(97+i)}", "mid3": f"dox{chr(97+i)}", "final": f"group{chr(97+i)}", "wrong": f"other{chr(97+i)}", "noise": f"noise{chr(97+i)}"} for i in range(8))


def graph_facts(unit: dict, depth: int, mode: str) -> tuple[list[str], bool]:
    mids = [unit["mid1"], unit["mid2"], unit["mid3"]]
    nodes = [unit["root"], *mids[:max(depth - 1, 0)], unit["final"]]
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    entailed = mode in ("entailed_direct", "entailed_chain", "entailed_multipath", "entailed_shortcut")
    if mode == "entailed_direct":
        edges = [(unit["root"], unit["final"])]
    elif mode == "reversed":
        edges = [(right, left) for left, right in edges]
    elif mode == "broken" and edges:
        cut = len(edges) // 2; edges[cut] = (edges[cut][0], unit["noise"])
    elif mode == "sibling":
        edges = [(unit["root"], unit["mid1"]), (unit["noise"], unit["mid1"])]
    elif mode == "irrelevant":
        edges = [(unit["noise"], unit["wrong"])]
    facts = [f"The item {left} is a kind of {right}." for left, right in edges]
    if mode in ("entailed_multipath", "entailed_shortcut"):
        facts.append(f"A separate register directly states that {unit['root']} is a kind of {unit['final']}.")
    return facts, entailed


def graph_material() -> list[dict]:
    rows = []
    entail_modes = ("entailed_direct", "entailed_chain", "entailed_multipath", "entailed_shortcut", "reversed", "broken", "sibling", "irrelevant")
    contradiction_modes = ("explicit_negative", "explicit_negative_noise", "entailed_chain", "unknown_missing")
    surfaces = CONSTRUCTIONS
    for unit_i, depth, surface, channel, order in itertools.product(range(len(GRAPH_UNITS)), range(1, 5), surfaces, ("entailment", "contradiction"), (1, -1)):
        modes = entail_modes if channel == "entailment" else contradiction_modes
        for mode in modes:
            unit = GRAPH_UNITS[unit_i]
            base_mode = "entailed_chain" if mode in ("explicit_negative", "explicit_negative_noise", "unknown_missing") else mode
            facts, entailed = graph_facts(unit, depth, base_mode)
            contradiction = mode.startswith("explicit_negative")
            if contradiction:
                facts = [f"The registry explicitly states that {unit['root']} is not a kind of {unit['final']}."]
                if mode == "explicit_negative_noise":
                    facts.append(f"The unrelated item {unit['noise']} is a kind of {unit['wrong']}.")
            if mode == "unknown_missing":
                facts = [f"The item {unit['root']} is a kind of {unit['mid1']}." ]
            truth = entailed if channel == "entailment" else contradiction
            question = (f"Do these facts support the conclusion that {unit['root']} is a kind of {unit['final']}?" if channel == "entailment" else f"Do these facts explicitly state that {unit['root']} is not a kind of {unit['final']}?")
            core = wrap(surface, " ".join(facts), f"The item {unit['noise']} is catalogued beside {unit['wrong']}.", question)
            correct, wrong = ("Yes", "No") if truth else ("No", "Yes")
            choices, gold = family_base.options(correct, wrong, order)
            rows.append({
                "case_id": f"c441-{channel}-{mode}-{surface}-u{unit_i}-d{depth}-{order:+d}",
                "panel": "repaired_binary_graph", "family": "type_graph", "surface": surface, "construction": surface,
                "unit": unit_i, "depth": depth, "mode": mode, "channel": channel, "cell": mode,
                "order": order, "partition": "discovery" if unit_i < 4 else "confirmation" if unit_i < 6 else "lockbox",
                "gold_position": gold, "correct_answer": correct, "wrong_answer": wrong,
                "prompt_core": core, "prompt": f"{core} {choices}. Reply with only A or B.",
                "free_prompt": f"{core} Answer with only Yes or No.",
                "role_values": {"primary": unit["root"], "secondary": unit["final"], "relation": "kind of", "context": unit["final"], "query": unit["root"]},
                "semantic_graph": {"depth": depth, "mode": mode, "channel": channel, "truth": truth},
            })
    return rows


def c441() -> None:
    out = begin("C441", {
        "status": "repaired_binary_graph_behavior_interface_frozen",
        "channels": ["positive support", "explicit negative statement"],
        "change_from_C422": "removes negative-polarity entailment wording and balances each channel exactly",
        "modes": "direct, chain, multipath, shortcut, reversed, broken, sibling, irrelevant, explicit negative",
        "gates": {"heldout": 0.80, "channel": 0.75, "mode": 0.65, "depth": 0.70},
        "hidden_state_policy": "none until all behavior gates pass",
    }, {"parent": final("C440")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = graph_material()
    write_rows(out / "material/cases.jsonl", rows)
    zero = {channel: float(np.mean([r["correct_answer"] == "Yes" for r in rows if r["channel"] == channel])) for channel in ("entailment", "contradiction")}
    tokenizer = axis_old.base.parent.fresh.tokenizer_qwen()
    compiled = family_base.compile_qwen(tokenizer, rows)
    write_rows(out / "compiled/qwen3.jsonl", compiled)
    run = axis_old.base.parent.previous.qwen_behavior(rows, compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl"); material = {r["case_id"]: r for r in rows}
    held = [r for r in behavior if material[r["case_id"]]["partition"] != "discovery"]
    by_channel = {value: float(np.mean([r["correct"] for r in held if material[r["case_id"]]["channel"] == value])) for value in ("entailment", "contradiction")}
    by_mode = {value: float(np.mean([r["correct"] for r in held if material[r["case_id"]]["mode"] == value])) for value in sorted({m["mode"] for m in rows})}
    by_depth = {str(value): float(np.mean([r["correct"] for r in held if material[r["case_id"]]["depth"] == value])) for value in range(1, 5)}
    heldout = float(np.mean([r["correct"] for r in held]))
    eligible = heldout >= 0.80 and min(by_channel.values()) >= 0.75 and min(by_mode.values()) >= 0.65 and min(by_depth.values()) >= 0.70
    headline = {
        "status": "repaired_binary_graph_behavior_closed", **run, "rows": len(rows), "zero_yes_frequency": zero,
        "heldout_accuracy": heldout, "channel_accuracy": by_channel, "mode_accuracy": by_mode, "depth_accuracy": by_depth,
        "graph_field_authorized": eligible,
        "strict_interpretation": "Failure would invalidate this interface/material for internal study, not graph reasoning in Qwen3.",
    }
    close("C441", headline, {"rows": len(rows) == 2304, "balance": all(abs(v - 0.5) < 1e-12 for v in zero.values()), "no_hidden": not (out / "raw/role_states.float16.npy").exists(), "finite": finite(headline)}, "C442_graph_field")


def c442() -> None:
    out = begin("C442", {
        "status": "qualified_graph_field_and_depth_prediction_frozen",
        "qualification": "C441 all behavior gates",
        "field": "order +1, every model checkpoint, six roles and all coordinates",
        "prediction": "depth1/2 response increment predicts unseen lexicon depth3/4 response",
        "controls": ["zero", "depth2 mean"],
    }, {"parent": final("C441")["all_checks_passed"]})
    if not final("C441")["headline"]["graph_field_authorized"]:
        close("C442", {"status": "graph_field_not_run_behavior_ineligible", "field_ran": False, "strict_interpretation": "Internal graph mechanism remains untested."}, {"route_accounted": True}, "C443_writer")
        return
    rows = read_rows(OUTS["C441"] / "material/cases.jsonl")
    compiled = read_rows(OUTS["C441"] / "compiled/qwen3.jsonl")
    selected = [(r, c) for r, c in zip(rows, compiled) if r["order"] == 1]
    rows2, compiled2 = [r for r, _ in selected], [c for _, c in selected]
    selector = lambda row: row["partition"] == "lockbox" and row["surface"] == "archive" and row["depth"] in (1, 4) and row["mode"] in ("entailed_chain", "broken")
    run = common.batch_capture_qwen(rows2, compiled2, out, full_selector=selector, batch_size=8, field_width=FIELD_WIDTH)
    states = np.load(out / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(out / "raw/hidden_index.jsonl"); keyed = {r["case_id"]: r for r in index}
    discovery, targets = {1: [], 2: []}, {3: [], 4: []}
    for unit, surface, depth in itertools.product(range(8), CONSTRUCTIONS, range(1, 5)):
        a = f"c441-entailment-entailed_chain-{surface}-u{unit}-d{depth}-+1"
        b = f"c441-entailment-broken-{surface}-u{unit}-d{depth}-+1"
        if a not in keyed or b not in keyed:
            continue
        response = np.asarray(states[keyed[a]["hidden_index"]], np.float32) - np.asarray(states[keyed[b]["hidden_index"]], np.float32)
        bucket = discovery if unit < 4 and depth <= 2 else targets if unit >= 4 and depth >= 3 else None
        if bucket is not None:
            bucket[depth].append(response)
    m1, m2 = np.mean(discovery[1], axis=0), np.mean(discovery[2], axis=0)
    increment = m2 - m1
    metrics = {}
    for depth in (3, 4):
        pred = m2 + (depth - 2) * increment
        truth = np.asarray(targets[depth], np.float32)
        metrics[str(depth)] = {
            "linear_increment_nrmse": float(np.mean([nrmse(pred, value) for value in truth])),
            "depth2_mean_nrmse": float(np.mean([nrmse(m2, value) for value in truth])),
            "zero_nrmse": 1.0,
            "samples": len(truth),
        }
    candidate = all(row["linear_increment_nrmse"] < min(row["depth2_mean_nrmse"], row["zero_nrmse"]) for row in metrics.values())
    close_memmap(states)
    headline = {
        "status": "qualified_graph_field_depth_prediction_closed", **run, "field_ran": True,
        "depth_metrics": metrics, "depth_rule_candidate": candidate,
        "strict_interpretation": "A pass is an average response-depth rule for this interface, not a proof of transitive-reasoning circuitry.",
    }
    close("C442", headline, {"shape": run["role_shape"][1:] == [38, 6, 2560], "samples": all(row["samples"] > 0 for row in metrics.values()), "finite": finite(headline)}, "C443_writer")


def c443() -> None:
    out = begin("C443", {
        "status": "expanded_known_truth_writer_calibration_frozen",
        "systems": 96, "checkpoints": 4, "roles": 6, "coordinates": 2560,
        "operations": ["bind", "reverse", "scope"],
        "controls": ["wrong operation", "wrong role", "wrong checkpoint", "coordinate roll", "matched noise"],
        "pass": "correct recovery >=0.95 and every control <=0.35",
        "claim_boundary": "synthetic measurement calibration only; no Qwen causal claim",
    }, {"parent": final("C442")["all_checks_passed"]})
    rng = np.random.default_rng(4431968)
    systems = 96; checkpoints = 4; roles = 6; dim = DIM; operations = 3
    base = rng.normal(0, 0.3, size=(systems, checkpoints, roles, dim)).astype(np.float32)
    templates = rng.normal(0, 0.04, size=(operations, checkpoints, roles, dim)).astype(np.float32)
    gates = rng.uniform(0.4, 1.3, size=(systems, operations, 1, 1, 1)).astype(np.float32)
    responses = gates * templates[None]
    damaged = base[:, None] - responses
    def recovery(patch: np.ndarray) -> float:
        restored = damaged + patch
        numerator = np.linalg.norm(restored - damaged, axis=(-3, -2, -1))
        error = np.linalg.norm(restored - base[:, None], axis=(-3, -2, -1))
        return float(np.mean(1.0 - error / (numerator + 1e-8)))
    correct = recovery(responses)
    controls = {
        "wrong_operation": recovery(np.roll(responses, 1, axis=1)),
        "wrong_role": recovery(np.roll(responses, 1, axis=3)),
        "wrong_checkpoint": recovery(np.roll(responses, 1, axis=2)),
        "coordinate_roll": recovery(np.roll(responses, 257, axis=4)),
        "matched_noise": recovery(rng.normal(0, np.std(responses), size=responses.shape).astype(np.float32)),
    }
    calibrated = correct >= 0.95 and max(controls.values()) <= 0.35
    save(out / "raw/calibration_seed.json", {"seed": 4431968})
    headline = {
        "status": "expanded_known_truth_writer_calibration_closed", "correct_recovery": correct,
        "control_recovery": controls, "writer_calibrated": calibrated,
        "strict_interpretation": "The instrument distinguishes registered synthetic truth from five mismatches; transfer to a natural model is not authorized by calibration alone.",
    }
    close("C443", headline, {"calibrated": calibrated, "finite": finite(headline)}, "C444_cross_model")


def c444() -> None:
    out = begin("C444", {
        "status": "registered_cross_model_functional_topology_frozen",
        "sources": ["C412 sequential BF16 behavior", "C413 relative-depth profiles", "C437 fresh Qwen field"],
        "comparison": "relative checkpoint response energy and role topology only",
        "exclusion": "behavior-ineligible models and all same-coordinate comparisons",
        "reason_for_reuse": "registered data are sufficient; no duplicate GPU run is scientifically justified",
    }, {"parent": final("C443")["all_checks_passed"]})
    c412 = load(old.OUTS["C412"] / "analysis/final.json")["headline"]
    c413 = load(old.OUTS["C413"] / "analysis/final.json")["headline"]
    profiles_path = old.OUTS["C413"] / "analysis/relative_profiles.json"
    profiles = load(profiles_path) if profiles_path.exists() else {}
    eligible = list(c413["eligible_models"])
    comparisons = c413["comparisons"]
    headline = {
        "status": "registered_cross_model_functional_topology_closed",
        "sequential_behavior_results": c412["results"], "eligible_models": eligible,
        "relative_topology_comparisons": comparisons, "profile_models": sorted(profiles),
        "functional_bisimulation_established": False,
        "strict_interpretation": "Qwen3 and eligible GLM4 can be compared at coarse relative-depth/role topology. DeepSeek is behavior-ineligible; native coordinates are never aligned.",
    }
    close("C444", headline, {"registered": bool(c412["results"]), "profiles": set(eligible).issubset(profiles), "finite": finite(headline)}, "C445_synthesis")


def hash_and_remove(paths: list[Path], out: Path) -> list[dict]:
    manifest = []
    for path in paths:
        if not path.exists():
            continue
        digest = hashlib.sha256(); size = path.stat().st_size
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        row = {"path": str(path.relative_to(ROOT)), "bytes": size, "sha256": digest.hexdigest()}
        path.unlink(); row["removed"] = not path.exists(); manifest.append(row)
    save(out / "audit/cleanup.json", manifest)
    return manifest


def c445() -> None:
    out = begin("C445", {
        "status": "campaign_synthesis_visual_cleanup_audit_frozen",
        "visual": "selected signed means, guarded slopes and graph depth responses retain all 2560 physical coordinates",
        "cleanup": "checksum and remove bulk fields not directly displayed",
        "new_math_gate": "requires repeated prospective rules plus causal use and cross-model functional equivalence",
    }, {"parent": final("C444")["all_checks_passed"]})
    visual_rows = []
    if final("C438")["headline"]["ran"]:
        groups = load(OUTS["C438"] / "analysis/groups.json")
        mean = np.load(OUTS["C438"] / "analysis/mean_next.float16.npy", mmap_mode="r")
        slope = np.load(OUTS["C438"] / "analysis/slope.float16.npy", mmap_mode="r")
        for gi, group in enumerate(groups):
            for q, role in itertools.product((0, 8, 16, 24, 32, 36), range(len(ROLES))):
                visual_rows.append({"id": f"mean:{group}:q{q}:{ROLES[role]}", "source": "guarded_mean_next", "group": group, "checkpoint": q + 1, "role": ROLES[role], "values": np.asarray(mean[gi, q, role], np.float32).round(6).tolist()})
                visual_rows.append({"id": f"slope:{group}:q{q}:{ROLES[role]}", "source": "guarded_transition_slope", "group": group, "checkpoint": q, "role": ROLES[role], "values": np.asarray(slope[gi, q, role], np.float32).round(6).tolist()})
        close_memmap(mean); close_memmap(slope)
    if final("C442")["headline"].get("field_ran"):
        states = np.load(OUTS["C442"] / "raw/role_states.float16.npy", mmap_mode="r")
        index = read_rows(OUTS["C442"] / "raw/hidden_index.jsonl")
        chosen = [row for row in index if row["case_id"].startswith("c441-entailment-entailed_chain-archive-u7-d4")][:1]
        for row in chosen:
            for q, role in itertools.product((0, 8, 16, 24, 32, 37), range(len(ROLES))):
                visual_rows.append({"id": f"graph:{row['case_id']}:q{q}:{ROLES[role]}", "source": "graph_lockbox_state", "checkpoint": q, "role": ROLES[role], "values": np.asarray(states[row["hidden_index"], q, role], np.float32).round(6).tolist()})
        close_memmap(states)
    payload = {
        "schema": "c445.guarded-response-graph.v1", "phase": 1979, "campaign": "C445", "model": "Qwen3-4B",
        "dimensions": list(range(DIM)), "rows": visual_rows,
        "summary": {"behavior": final("C436")["headline"], "prediction": final("C439")["headline"], "distance": final("C440")["headline"], "graph": final("C442")["headline"], "writer": final("C443")["headline"], "cross_model": final("C444")["headline"]},
        "claim_boundary": "Coordinates are physical activation dimensions, not parameters, neurons, universal semantic atoms, or a causal circuit.",
    }
    save(VISUAL, payload)
    cleanup_paths = [
        OUTS["C437"] / "raw/role_states.float16.npy", OUTS["C437"] / "raw/full_fields_holdout.float16.npy",
        *[OUTS["C438"] / f"analysis/{name}.float16.npy" for name in ("slope", "intercept", "mean_next", "residual_low", "residual_high")],
        OUTS["C442"] / "raw/role_states.float16.npy", OUTS["C442"] / "raw/full_fields_holdout.float16.npy",
    ]
    cleanup = hash_and_remove(cleanup_paths, out)
    gates = {
        "qualified_broad": final("C436")["headline"]["eligible_broad"],
        "qualified_axes": final("C436")["headline"]["eligible_axes"],
        "guarded_prediction_candidates": final("C439")["headline"].get("candidate_groups", []),
        "distance_candidates": final("C440")["headline"].get("candidate_methods", []),
        "graph_field_ran": final("C442")["headline"].get("field_ran", False),
        "graph_depth_candidate": final("C442")["headline"].get("depth_rule_candidate", False),
        "known_truth_writer": final("C443")["headline"]["writer_calibrated"],
        "cross_model_bisimulation": final("C444")["headline"]["functional_bisimulation_established"],
        "causal_natural_model": False, "new_math": False,
    }
    next_same_goal = bool(gates["guarded_prediction_candidates"] or gates["distance_candidates"] or gates["graph_depth_candidate"])
    headline = {
        "status": "guarded_response_graph_campaign_closed", "gates": gates,
        "visual_rows": len(visual_rows), "visual_path": str(VISUAL.relative_to(ROOT)),
        "cleanup_files": len(cleanup), "cleanup_bytes": sum(row["bytes"] for row in cleanup),
        "next_stage_same_goal": next_same_goal,
        "next_authorization_detail": "prospective replication and then narrow natural-model writer" if next_same_goal else "broaden language constructions and discover a different full-coordinate rule family",
        "new_math_gate_passed": False,
        "strict_interpretation": "The campaign can establish empirical local rules and measurement calibration; it cannot infer a universal language algebra, unique circuit, or new mathematics without stronger prospective and causal evidence.",
    }
    close("C445", headline, {"visual": bool(visual_rows) and all(len(row["values"]) == DIM for row in visual_rows), "cleanup": all(row["removed"] and row["sha256"] for row in cleanup), "finite": finite(headline)}, "independent_audit_then_registered_next_stage")


RUNNERS = {name: globals()[name.lower()] for name in PHASES}


def parse_range(value: str) -> list[str]:
    if value in ("all", "C434-C445"):
        return list(PHASES)
    if "-" in value:
        left, right = value.split("-", 1)
        return [f"C{i}" for i in range(int(left[1:]), int(right[1:]) + 1)]
    return [value]


def validate_only() -> None:
    rows = all_material()
    graph = graph_material()
    checks = {
        "phase_sequence": [PHASES[f"C{i}"][0] for i in range(434, 446)] == list(range(1968, 1980)),
        "material_rows": len(rows) == 2880,
        "graph_rows": len(graph) == 2304,
        "answer_balance": sum(row["gold_position"] == 0 for row in rows) == len(rows) // 2,
        "graph_channel_balance": all(abs(np.mean([r["correct_answer"] == "Yes" for r in graph if r["channel"] == channel]) - 0.5) < 1e-12 for channel in ("entailment", "contradiction")),
        "roles": all(set(row["role_values"]) == set(ROLES) - {"boundary"} for row in rows + graph),
    }
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    if not all(checks.values()):
        raise AssertionError(checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="C434-C445")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_only(); return
    for name in parse_range(args.run):
        RUNNERS[name]()


if __name__ == "__main__":
    main()
