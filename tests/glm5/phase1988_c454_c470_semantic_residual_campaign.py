#!/usr/bin/env python3
"""C454-C470 shared propagation, semantic residual, graph, and writer campaign.

Only embeddings and HiddenState checkpoints are neural objects.  The campaign
keeps every physical activation coordinate and does not inspect Attention, MLP,
weights, PCA, Top-K selections, or compressed latent projections.
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
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c470_semantic_residual_graph.json"
sys.path.insert(0, str(TESTS))

import phase1980_c446_c453_fixed_codebook_replication as previous
import phase1968_c434_c445_guarded_response_graph_campaign as graph_base
import phase1933_c399_c414_output_sensitive_language_campaign as language_base
import phase1844_c310_c335_dual_axis_common as common
import phase1797_c263_c272_state_operator_common as compile_base
import phase1332_bf16_utils as model_base


PHASES = {
    f"C{campaign}": (1988 + campaign - 454, slug)
    for campaign, slug in (
        (454, "semantic_residual_evidence_adjudication_and_contract"),
        (455, "sixteen_family_surface_semantic_material"),
        (456, "material_zero_model_and_role_audit"),
        (457, "qwen_sixteen_family_behavior_qualification"),
        (458, "qualified_full_coordinate_semantic_surface_field"),
        (459, "typed_response_ledger"),
        (460, "identity_and_mean_propagation_baselines"),
        (461, "shared_checkpoint_role_propagation"),
        (462, "construction_conditioned_propagation"),
        (463, "operation_and_family_semantic_increment"),
        (464, "semantic_residual_lockbox_adjudication"),
        (465, "full_coordinate_neighbor_coupling_tournament"),
        (466, "autonomous_multistep_response_rollout"),
        (467, "fresh_graph_path_material_and_behavior"),
        (468, "graph_path_field_and_nonlinear_integration"),
        (469, "conditional_natural_semantic_writer"),
        (470, "campaign_synthesis_visual_cleanup_and_audit"),
    )
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}" for name, (phase, slug) in PHASES.items()}

DIM = 2560
CHECKPOINTS = 38
ROLES = common.ROLES
FIELD_WIDTH = 192
FAMILIES = tuple(language_base.FAMILIES)
CONSTRUCTIONS = ("register", "circular", "report")
OPERATIONS = ("surface", "statement", "query")
STYLES = (0, 1)
UNITS = (
    {"p": "Adren", "s": "Belis", "o": "Coren", "obj": "ackee", "other": "astrolabe", "node": "ruxa", "parent": "taxa", "wrong": "taxa_alt", "event": "alignment"},
    {"p": "Daris", "s": "Elwen", "o": "Felis", "obj": "breadfruit", "other": "caliper", "node": "ruxb", "parent": "taxb", "wrong": "taxb_alt", "event": "assembly"},
    {"p": "Garen", "s": "Helia", "o": "Iven", "obj": "chayote", "other": "dividers", "node": "ruxc", "parent": "taxc", "wrong": "taxc_alt", "event": "calibration"},
    {"p": "Jalen", "s": "Keris", "o": "Loren", "obj": "damson", "other": "goniometer", "node": "ruxd", "parent": "taxd", "wrong": "taxd_alt", "event": "classification"},
    {"p": "Maren", "s": "Neris", "o": "Orla", "obj": "endive", "other": "hypsometer", "node": "ruxe", "parent": "taxe", "wrong": "taxe_alt", "event": "comparison"},
    {"p": "Pavel", "s": "Quen", "o": "Risa", "obj": "feijoa", "other": "inclinometer", "node": "ruxf", "parent": "taxf", "wrong": "taxf_alt", "event": "diagnosis"},
    {"p": "Saren", "s": "Tovin", "o": "Ulia", "obj": "guava", "other": "micrometer", "node": "ruxg", "parent": "taxg", "wrong": "taxg_alt", "event": "evaluation"},
    {"p": "Varen", "s": "Welis", "o": "Xorin", "obj": "huckleberry", "other": "pantograph", "node": "ruxh", "parent": "taxh", "wrong": "taxh_alt", "event": "forecast"},
    {"p": "Yaren", "s": "Zelis", "o": "Acor", "obj": "ilama", "other": "planisphere", "node": "ruxi", "parent": "taxi", "wrong": "taxi_alt", "event": "inspection"},
    {"p": "Borin", "s": "Celia", "o": "Davor", "obj": "jackfruit", "other": "spherometer", "node": "ruxj", "parent": "taxj", "wrong": "taxj_alt", "event": "review"},
)
GRAPH_UNITS = tuple({
    "root": f"zen{chr(97+i)}", "mid1": f"pol{chr(97+i)}", "mid2": f"qir{chr(97+i)}",
    "mid3": f"sav{chr(97+i)}", "final": f"class{chr(97+i)}", "wrong": f"other{chr(97+i)}",
    "noise": f"noise{chr(97+i)}",
} for i in range(10))
GRAPH_MODES = ("chain", "shortcut", "direct", "broken", "reversed", "irrelevant")


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


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


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
    value = {"phase": PHASES[name][0], "campaign": name, "status": "closed", "checks": final_checks,
             "all_checks_passed": all(final_checks.values()), "headline": headline, "next_authorization": authorization}
    save(out / "analysis/final.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def partition(unit: int) -> str:
    return "discovery" if unit < 5 else "confirmation" if unit < 8 else "lockbox"


def fixed_prompt(core: str, truth: bool) -> tuple[str, int, str, str]:
    return f"{core} (A) Yes (B) No. Reply with only A or B.", 0 if truth else 1, "Yes" if truth else "No", "No" if truth else "Yes"


def wrap(construction: str, target: str, noise: str, question: str) -> str:
    if construction == "register":
        return f"A register marks the relevant statement: {target} A separate note says: {noise} Using only the register, {question}"
    if construction == "circular":
        return f"A circular contains two entries. Relevant entry: {target} Unrelated entry: {noise} From the circular alone, {question}"
    if construction == "report":
        return f"A report records {target} Elsewhere the report records {noise} Based only on the report, {question}"
    raise KeyError(construction)


def semantic_material() -> list[dict]:
    rows: list[dict] = []
    original = language_base.UNITS
    language_base.UNITS = UNITS
    try:
        for family, construction, unit, a, b, style in itertools.product(
            FAMILIES, CONSTRUCTIONS, range(len(UNITS)), (0, 1), (0, 1), STYLES
        ):
            case = language_base.family_statement(family, unit, a, b)
            noise = case["noise"] if style == 0 else case["noise"].replace(" catalogued ", " carefully catalogued ")
            core = wrap(construction, case["target"], noise, case["question"])
            prompt, gold, correct, wrong = fixed_prompt(core, bool(case["truth"]))
            case_id = f"c455-{family}-{construction}-u{unit}-a{a}b{b}-s{style}"
            rows.append({
                "case_id": case_id, "panel": "semantic_surface_factorial", "family": family,
                "surface": construction, "construction": construction, "unit": unit,
                "factor_a": a, "factor_b": b, "style": style, "cell": f"{a}{b}s{style}",
                "order": 1, "partition": partition(unit), "gold_position": gold,
                "correct_answer": correct, "wrong_answer": wrong, "prompt_core": core, "prompt": prompt,
                "free_prompt": f"{core} Answer with only Yes or No.", "role_values": case["roles"],
                "semantic_graph": {"family": family, "statement": a, "query": b, "style": style, "truth": bool(case["truth"])},
            })
    finally:
        language_base.UNITS = original
    return rows


def graph_facts(unit: dict, depth: int, mode: str) -> tuple[list[str], bool]:
    nodes = [unit["root"], *[unit["mid1"], unit["mid2"], unit["mid3"]][:max(depth - 1, 0)], unit["final"]]
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    truth = mode in ("chain", "shortcut", "direct")
    if mode == "direct":
        edges = [(unit["root"], unit["final"])]
    elif mode == "shortcut":
        edges.append((unit["root"], unit["final"]))
    elif mode == "broken":
        cut = len(edges) // 2
        edges[cut] = (edges[cut][0], unit["noise"])
    elif mode == "reversed":
        edges = [(right, left) for left, right in edges]
    elif mode == "irrelevant":
        edges = [(unit["noise"], unit["wrong"])]
    return [f"The item {left} is a kind of {right}." for left, right in edges], truth


def graph_material() -> list[dict]:
    rows = []
    for unit_i, depth, construction, mode in itertools.product(range(10), range(1, 5), CONSTRUCTIONS, GRAPH_MODES):
        unit = GRAPH_UNITS[unit_i]
        facts, truth = graph_facts(unit, depth, mode)
        question = f"Do these facts support the conclusion that {unit['root']} is a kind of {unit['final']}?"
        core = wrap(construction, " ".join(facts), f"The item {unit['noise']} is listed beside {unit['wrong']}.", question)
        prompt, gold, correct, wrong = fixed_prompt(core, truth)
        rows.append({
            "case_id": f"c467-{mode}-{construction}-u{unit_i}-d{depth}", "panel": "graph_path_factorial",
            "family": "type_graph", "surface": construction, "construction": construction, "unit": unit_i,
            "factor_a": None, "factor_b": None, "depth": depth, "mode": mode, "cell": mode, "order": 1,
            "partition": partition(unit_i), "gold_position": gold, "correct_answer": correct, "wrong_answer": wrong,
            "prompt_core": core, "prompt": prompt, "free_prompt": f"{core} Answer with only Yes or No.",
            "role_values": {"primary": unit["root"], "secondary": unit["final"], "relation": "kind of", "context": unit["final"], "query": unit["root"]},
            "semantic_graph": {"depth": depth, "mode": mode, "truth": truth},
        })
    return rows


def semantic_lookup() -> tuple[list[dict], dict[str, dict]]:
    rows = read_rows(OUTS["C455"] / "material/cases.jsonl")
    return rows, {row["case_id"]: row for row in rows}


def close_mmap(value) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def response_ledger(index: list[dict], eligible: set[str]) -> list[dict]:
    keyed = {row["case_id"]: row for row in index}
    records = []
    for family, construction, unit in itertools.product(sorted(eligible), CONSTRUCTIONS, range(len(UNITS))):
        for a, b in itertools.product((0, 1), repeat=2):
            left = f"c455-{family}-{construction}-u{unit}-a{a}b{b}-s0"
            right = f"c455-{family}-{construction}-u{unit}-a{a}b{b}-s1"
            if left in keyed and right in keyed:
                records.append({"family": family, "construction": construction, "unit": unit, "partition": partition(unit),
                                "operation": "surface", "context": f"a{a}b{b}", "left": keyed[left]["hidden_index"], "right": keyed[right]["hidden_index"]})
        for b in (0, 1):
            left = f"c455-{family}-{construction}-u{unit}-a0b{b}-s0"
            right = f"c455-{family}-{construction}-u{unit}-a1b{b}-s0"
            if left in keyed and right in keyed:
                records.append({"family": family, "construction": construction, "unit": unit, "partition": partition(unit),
                                "operation": "statement", "context": f"b{b}", "left": keyed[left]["hidden_index"], "right": keyed[right]["hidden_index"]})
        for a in (0, 1):
            left = f"c455-{family}-{construction}-u{unit}-a{a}b0-s0"
            right = f"c455-{family}-{construction}-u{unit}-a{a}b1-s0"
            if left in keyed and right in keyed:
                records.append({"family": family, "construction": construction, "unit": unit, "partition": partition(unit),
                                "operation": "query", "context": f"a{a}", "left": keyed[left]["hidden_index"], "right": keyed[right]["hidden_index"]})
    return records


def record_arrays(states, records: list[dict], q: int, role: int) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray([row["left"] for row in records], dtype=np.int64)
    right = np.asarray([row["right"] for row in records], dtype=np.int64)
    x = np.asarray(states[right, q, role], np.float32) - np.asarray(states[left, q, role], np.float32)
    y = np.asarray(states[right, q + 1, role], np.float32) - np.asarray(states[left, q + 1, role], np.float32)
    return x, y


def fit_diagonal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xm, ym = x.mean(0), y.mean(0)
    centered = x - xm
    slope = np.sum(centered * (y - ym), axis=0) / (np.sum(centered * centered, axis=0) + 1e-6)
    slope = np.clip(slope, -64, 64)
    return slope, ym - slope * xm


def new_accumulator() -> dict:
    return {"error": 0.0, "truth": 0.0, "samples": 0}


def add_metric(acc: dict, prediction: np.ndarray, truth: np.ndarray) -> None:
    diff = np.asarray(prediction, np.float32) - np.asarray(truth, np.float32)
    acc["error"] += float(np.sum(diff * diff, dtype=np.float64))
    acc["truth"] += float(np.sum(np.asarray(truth, np.float32) ** 2, dtype=np.float64))
    acc["samples"] += int(truth.shape[0])


def finish_metric(acc: dict) -> dict:
    return {"nrmse": math.sqrt(acc["error"] / (acc["truth"] + 1e-12)), "samples": acc["samples"]}


def metric_buckets() -> dict:
    return {"all": new_accumulator(), "by_operation": {}, "by_construction": {}, "by_partition": {}, "by_family_operation": {}}


def add_buckets(buckets: dict, records: list[dict], prediction: np.ndarray, truth: np.ndarray) -> None:
    add_metric(buckets["all"], prediction, truth)
    dimensions = {
        "by_operation": [row["operation"] for row in records],
        "by_construction": [row["construction"] for row in records],
        "by_partition": [row["partition"] for row in records],
        "by_family_operation": [f"{row['family']}::{row['operation']}" for row in records],
    }
    for dimension, labels in dimensions.items():
        for label in sorted(set(labels)):
            mask = np.asarray([value == label for value in labels])
            target = buckets[dimension].setdefault(label, new_accumulator())
            add_metric(target, prediction[mask], truth[mask])


def finish_buckets(buckets: dict) -> dict:
    return {"all": finish_metric(buckets["all"]), **{
        dimension: {label: finish_metric(value) for label, value in rows.items()}
        for dimension, rows in buckets.items() if dimension != "all"
    }}


def semantic_train_eval() -> tuple[list[dict], list[dict]]:
    records = read_rows(OUTS["C459"] / "analysis/response_records.jsonl")
    train = [row for row in records if row["partition"] == "discovery" and row["construction"] in CONSTRUCTIONS[:2]]
    evaluate = [row for row in records if row["partition"] != "discovery"]
    return train, evaluate


def c454() -> None:
    audit = load(previous.OUTS["C453"] / "audit/independent_audit.json")
    begin("C454", {
        "status": "semantic_residual_campaign_contract_frozen", "parent": "C446-C453 independent audit",
        "corrections": ["C450 fits separate coefficients per group; it does not identify one shared operator",
                        "semantic selection times local dynamics is a hypothesis, not a discovered mechanism",
                        "state-response transfer is not natural-state transition"],
        "routes": ["shared propagation baselines", "semantic increment", "cross-coordinate neighbor coupling",
                   "multistep rollout", "graph path integration", "conditional natural writer"],
        "route_policy": "route failure never stops another registered route",
        "measurement": "embedding and HiddenState only; every coordinate; no Attention, MLP, weights, PCA, Top-K, or cosine gate",
    }, {"parent": audit["all_checks_passed"], "continuity": PHASES["C454"][0] == 1988})
    close("C454", {"status": "contract_closed", "retained": ["broad adjacent-checkpoint predictability", "full-coordinate signed response"],
          "corrected": ["not one shared operator", "not semantic specificity", "not a natural causal circuit"],
          "strict_interpretation": "The campaign asks whether a semantic increment remains after explicit shared and surface baselines."},
          {"families": len(FAMILIES) == 16, "routes": True}, "C455_material")


def c455() -> None:
    out = begin("C455", {
        "status": "sixteen_family_surface_semantic_material_frozen", "families": list(FAMILIES),
        "factorial": "statement x query x irrelevant-surface-style", "constructions": list(CONSTRUCTIONS),
        "units": 10, "partitions": {"discovery": [0,1,2,3,4], "confirmation": [5,6,7], "lockbox": [8,9]},
        "codebook": "A always Yes; B always No", "naturalness": "controlled English; no independent human panel",
    }, {"parent": final("C454")["all_checks_passed"]})
    rows = semantic_material()
    write_rows(out / "material/cases.jsonl", rows)
    counts = {key: sum(row["partition"] == key for row in rows) for key in ("discovery", "confirmation", "lockbox")}
    close("C455", {"status": "material_closed", "rows": len(rows), "family_count": len({r['family'] for r in rows}),
          "partition_counts": counts, "truth_frequency": float(np.mean([r["gold_position"] == 0 for r in rows])),
          "strict_interpretation": "The factorial separates irrelevant surface edits from two balanced semantic edits; it is not a natural-language ontology."},
          {"rows": len(rows) == 3840, "families": len({r['family'] for r in rows}) == 16, "balance": sum(r["gold_position"] == 0 for r in rows) == len(rows)//2}, "C456_audit")


def c456() -> None:
    out = begin("C456", {"status": "material_zero_model_role_audit_frozen", "zero_models": ["always A", "always B", "family majority", "construction majority", "style majority"], "role_test": "every registered role string occurs in prompt"}, {"parent": final("C455")["all_checks_passed"]})
    rows, _ = semantic_lookup()
    zero = {
        "always_a": float(np.mean([r["gold_position"] == 0 for r in rows])),
        "always_b": float(np.mean([r["gold_position"] == 1 for r in rows])),
        "family_majority": float(np.mean([max(np.mean([x["gold_position"] == 0 for x in rows if x["family"] == f]), np.mean([x["gold_position"] == 1 for x in rows if x["family"] == f])) for f in FAMILIES])),
        "construction_majority": float(np.mean([max(np.mean([x["gold_position"] == 0 for x in rows if x["construction"] == c]), np.mean([x["gold_position"] == 1 for x in rows if x["construction"] == c])) for c in CONSTRUCTIONS])),
        "style_majority": float(np.mean([max(np.mean([x["gold_position"] == 0 for x in rows if x["style"] == s]), np.mean([x["gold_position"] == 1 for x in rows if x["style"] == s])) for s in STYLES])),
    }
    roles = all(all(str(value) in row["prompt_core"] for value in row["role_values"].values()) for row in rows)
    eligible = all(abs(value - 0.5) < 1e-12 for value in zero.values()) and roles
    close("C456", {"status": "material_audit_closed", "zero_model_accuracies": zero, "role_occurrence": roles,
          "material_eligible": eligible, "human_naturalness_review": False,
          "strict_interpretation": "Exact shortcut balance and role occurrence do not establish naturalness or semantic uniqueness."},
          {"balance": eligible}, "C457_behavior")


def c457() -> None:
    out = begin("C457", {"status": "qwen_sixteen_family_behavior_frozen", "model": "Qwen3-4B BF16 CUDA",
        "gates": {"overall_heldout": 0.75, "eligible_family": 0.60, "construction": 0.60, "minimum_eligible_families": 8},
        "policy": "only eligible families authorize internal interpretation; failed families do not stop eligible families"},
        {"parent": final("C456")["all_checks_passed"], "material": final("C456")["headline"]["material_eligible"], "cuda": torch.cuda.is_available()})
    rows, by_id = semantic_lookup()
    tokenizer = graph_base.axis_old.base.parent.fresh.tokenizer_qwen()
    compiled = compile_base.compile_qwen(tokenizer, rows)
    write_rows(out / "compiled/qwen3.jsonl", compiled)
    run = graph_base.axis_old.base.parent.previous.qwen_behavior(rows, compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl")
    held = [row for row in behavior if by_id[row["case_id"]]["partition"] != "discovery"]
    by_family = {family: float(np.mean([row["correct"] for row in held if by_id[row["case_id"]]["family"] == family])) for family in FAMILIES}
    by_construction = {c: float(np.mean([row["correct"] for row in held if by_id[row["case_id"]]["construction"] == c])) for c in CONSTRUCTIONS}
    by_style = {str(s): float(np.mean([row["correct"] for row in held if by_id[row["case_id"]]["style"] == s])) for s in STYLES}
    eligible = sorted([family for family, accuracy in by_family.items() if accuracy >= 0.60])
    heldout = float(np.mean([row["correct"] for row in held]))
    authorized = heldout >= 0.75 and min(by_construction.values()) >= 0.60 and len(eligible) >= 8
    headline = {"status": "behavior_closed", **run, "heldout_accuracy": heldout, "family_accuracy": by_family,
                "construction_accuracy": by_construction, "style_accuracy": by_style, "eligible_families": eligible,
                "field_authorized": authorized, "strict_interpretation": "Eligibility is family-specific and qualifies only this fixed-codebook interface."}
    close("C457", headline, {"rows": len(behavior) == len(rows), "no_hidden": not (out / "raw/role_states.float16.npy").exists(), "finite": finite(headline)}, "C458_field")


def c458() -> None:
    out = begin("C458", {"status": "qualified_semantic_surface_full_field_frozen",
        "archive": "all eligible cases x 38 checkpoints x six roles x all 2560 coordinates",
        "full_token": "32 deterministic lockbox report cases x all checkpoints x 192 token slots x all coordinates",
        "no_pca_topk": True}, {"parent": final("C457")["all_checks_passed"]})
    if not final("C457")["headline"]["field_authorized"]:
        close("C458", {"status": "field_not_run_behavior_ineligible", "field_ran": False}, {"route_accounted": True}, "C459_ledger")
        return
    rows, _ = semantic_lookup()
    eligible = set(final("C457")["headline"]["eligible_families"])
    compiled_all = {row["case_id"]: row for row in read_rows(OUTS["C457"] / "compiled/qwen3.jsonl")}
    selected_rows = [row for row in rows if row["family"] in eligible]
    selected_compiled = [compiled_all[row["case_id"]] for row in selected_rows]
    full_ids = set(row["case_id"] for row in sorted([r for r in selected_rows if r["partition"] == "lockbox" and r["construction"] == "report" and r["style"] == 0], key=lambda r: r["case_id"])[:32])
    run = common.batch_capture_qwen(selected_rows, selected_compiled, out, full_selector=lambda row: row["case_id"] in full_ids, batch_size=8, field_width=FIELD_WIDTH)
    headline = {"status": "full_field_closed", **run, "field_ran": True, "eligible_families": sorted(eligible),
                "strict_interpretation": "All coordinates are observations; surface/semantic factorial control does not itself establish semantic mechanism."}
    close("C458", headline, {"shape": run["role_shape"][1:] == [38,6,2560], "full": run["full_token_rows"] == 32, "finite": finite(headline)}, "C459_ledger")


def c459() -> None:
    out = begin("C459", {"status": "typed_response_ledger_frozen", "edits": list(OPERATIONS),
        "surface": "style0 to style1 at fixed statement/query truth cell", "statement": "a0 to a1 at fixed query factor", "query": "b0 to b1 at fixed statement factor"},
        {"parent": final("C458")["all_checks_passed"]})
    if not final("C458")["headline"].get("field_ran"):
        close("C459", {"status": "ledger_not_run_no_field", "ran": False}, {"route_accounted": True}, "C460_baselines")
        return
    index = read_rows(OUTS["C458"] / "raw/hidden_index.jsonl")
    eligible = set(final("C458")["headline"]["eligible_families"])
    records = response_ledger(index, eligible)
    write_rows(out / "analysis/response_records.jsonl", records)
    counts = {op: sum(row["operation"] == op for row in records) for op in OPERATIONS}
    headline = {"status": "typed_response_ledger_closed", "ran": True, "records": len(records), "operation_counts": counts,
                "families": len(eligible), "strict_interpretation": "Ledger entries are researcher-defined paired responses, not discovered neural edges."}
    close("C459", headline, {"operations": all(counts[op] > 0 for op in OPERATIONS), "balanced_semantics": counts["statement"] == counts["query"]}, "C460_baselines")


def c460() -> None:
    out = begin("C460", {"status": "identity_mean_baselines_frozen", "train": "discovery units0-4 in register/circular",
        "evaluation": "confirmation and lockbox; all eligible families, roles, checkpoints and coordinates", "models": ["identity", "training mean"]},
        {"parent": final("C459")["all_checks_passed"]})
    if not final("C459")["headline"].get("ran"):
        close("C460", {"status": "baseline_not_run_no_ledger", "ran": False}, {"route_accounted": True}, "C461_shared")
        return
    states = np.load(OUTS["C458"] / "raw/role_states.float16.npy", mmap_mode="r")
    train, evaluate = semantic_train_eval()
    means = np.lib.format.open_memmap(out / "analysis/mean_next.float16.npy", mode="w+", dtype=np.float16, shape=(37,6,DIM))
    identity, mean_metrics = metric_buckets(), metric_buckets()
    for q in range(37):
        for role in range(6):
            _, y_train = record_arrays(states, train, q, role)
            means[q, role] = y_train.mean(0).astype(np.float16)
            x, y = record_arrays(states, evaluate, q, role)
            add_buckets(identity, evaluate, x, y)
            add_buckets(mean_metrics, evaluate, np.broadcast_to(np.asarray(means[q, role], np.float32), y.shape), y)
        means.flush()
        print(f"[C460] q={q}", flush=True)
    close_mmap(states); close_mmap(means)
    metrics = {"identity": finish_buckets(identity), "mean": finish_buckets(mean_metrics)}
    save(out / "analysis/metrics.json", metrics)
    headline = {"status": "baseline_closed", "ran": True, "metrics": {name: row["all"] for name, row in metrics.items()},
                "strict_interpretation": "These are explicit continuity baselines, not null semantic models."}
    close("C460", headline, {"finite": finite(headline), "eval": len(evaluate) > 0}, "C461_shared")


def c461() -> None:
    out = begin("C461", {"status": "shared_checkpoint_role_propagation_frozen", "model": "one diagonal affine map per checkpoint and role shared by families, operations and constructions", "controls": ["identity", "mean"]}, {"parent": final("C460")["all_checks_passed"]})
    if not final("C460")["headline"].get("ran"):
        close("C461", {"status": "shared_not_run", "ran": False}, {"route_accounted": True}, "C462_construction")
        return
    states = np.load(OUTS["C458"] / "raw/role_states.float16.npy", mmap_mode="r")
    train, evaluate = semantic_train_eval()
    slope = np.lib.format.open_memmap(out / "analysis/slope.float16.npy", mode="w+", dtype=np.float16, shape=(37,6,DIM))
    intercept = np.lib.format.open_memmap(out / "analysis/intercept.float16.npy", mode="w+", dtype=np.float16, shape=(37,6,DIM))
    buckets = metric_buckets()
    for q in range(37):
        for role in range(6):
            x_train, y_train = record_arrays(states, train, q, role)
            a, b = fit_diagonal(x_train, y_train); slope[q,role] = a.astype(np.float16); intercept[q,role] = b.astype(np.float16)
            x, y = record_arrays(states, evaluate, q, role)
            pred = a * x + b
            add_buckets(buckets, evaluate, pred, y)
        slope.flush(); intercept.flush(); print(f"[C461] q={q}", flush=True)
    close_mmap(states); close_mmap(slope); close_mmap(intercept)
    metrics = finish_buckets(buckets); save(out / "analysis/metrics.json", metrics)
    controls = load(OUTS["C460"] / "analysis/metrics.json")
    candidate = metrics["all"]["nrmse"] < min(controls["identity"]["all"]["nrmse"], controls["mean"]["all"]["nrmse"])
    headline = {"status": "shared_propagation_closed", "ran": True, "nrmse": metrics["all"]["nrmse"],
                "identity_nrmse": controls["identity"]["all"]["nrmse"], "mean_nrmse": controls["mean"]["all"]["nrmse"],
                "shared_candidate": candidate, "strict_interpretation": "A pass identifies a shared diagonal response baseline, not semantic selection."}
    close("C461", headline, {"finite": finite(headline)}, "C462_construction")


def c462() -> None:
    out = begin("C462", {"status": "construction_conditioned_propagation_frozen", "model": "C461 plus per-construction mean residual learned only for register/circular", "unseen_construction": "report receives zero construction residual"}, {"parent": final("C461")["all_checks_passed"]})
    if not final("C461")["headline"].get("ran"):
        close("C462", {"status": "construction_not_run", "ran": False}, {"route_accounted": True}, "C463_semantic")
        return
    states = np.load(OUTS["C458"] / "raw/role_states.float16.npy", mmap_mode="r")
    slope = np.load(OUTS["C461"] / "analysis/slope.float16.npy", mmap_mode="r"); intercept = np.load(OUTS["C461"] / "analysis/intercept.float16.npy", mmap_mode="r")
    train, evaluate = semantic_train_eval()
    residual = np.lib.format.open_memmap(out / "analysis/construction_residual.float16.npy", mode="w+", dtype=np.float16, shape=(len(CONSTRUCTIONS),37,6,DIM))
    buckets = metric_buckets()
    for q in range(37):
        for role in range(6):
            x_train, y_train = record_arrays(states, train, q, role)
            base_train = np.asarray(slope[q,role], np.float32) * x_train + np.asarray(intercept[q,role], np.float32)
            for ci, construction in enumerate(CONSTRUCTIONS[:2]):
                mask = np.asarray([row["construction"] == construction for row in train])
                residual[ci,q,role] = (y_train[mask] - base_train[mask]).mean(0).astype(np.float16)
            residual[2,q,role] = 0
            x, y = record_arrays(states, evaluate, q, role)
            pred = np.asarray(slope[q,role],np.float32) * x + np.asarray(intercept[q,role],np.float32)
            for ci, construction in enumerate(CONSTRUCTIONS):
                mask = np.asarray([row["construction"] == construction for row in evaluate])
                pred[mask] += np.asarray(residual[ci,q,role], np.float32)
            add_buckets(buckets, evaluate, pred, y)
        residual.flush(); print(f"[C462] q={q}", flush=True)
    for value in (states,slope,intercept,residual): close_mmap(value)
    metrics = finish_buckets(buckets); save(out / "analysis/metrics.json", metrics)
    shared = load(OUTS["C461"] / "analysis/metrics.json")
    seen_gain = float(np.mean([shared["by_construction"][c]["nrmse"] - metrics["by_construction"][c]["nrmse"] for c in CONSTRUCTIONS[:2]]))
    headline = {"status": "construction_model_closed", "ran": True, "nrmse": metrics["all"]["nrmse"], "seen_construction_gain": seen_gain,
                "unseen_report_gain": shared["by_construction"]["report"]["nrmse"] - metrics["by_construction"]["report"]["nrmse"],
                "strict_interpretation": "Construction residuals are surface-conditioned baselines; report is a genuine abstention branch."}
    close("C462", headline, {"finite": finite(headline)}, "C463_semantic")


def c463() -> None:
    out = begin("C463", {"status": "operation_family_increment_frozen", "models": ["M2 construction baseline", "M3 operation increment", "M3 family-operation increment"],
        "heldout_family": "leave-one-family-out operation residual excludes target family", "claim": "semantic increment only if M3-LOO improves semantic edits beyond surface improvement"}, {"parent": final("C462")["all_checks_passed"]})
    if not final("C462")["headline"].get("ran"):
        close("C463", {"status": "semantic_model_not_run", "ran": False}, {"route_accounted": True}, "C464_adjudication")
        return
    eligible = final("C458")["headline"]["eligible_families"]; f_index = {name:i for i,name in enumerate(eligible)}; o_index = {name:i for i,name in enumerate(OPERATIONS)}; c_index={name:i for i,name in enumerate(CONSTRUCTIONS)}
    states=np.load(OUTS["C458"] / "raw/role_states.float16.npy",mmap_mode="r"); slope=np.load(OUTS["C461"] / "analysis/slope.float16.npy",mmap_mode="r"); intercept=np.load(OUTS["C461"] / "analysis/intercept.float16.npy",mmap_mode="r"); construction=np.load(OUTS["C462"] / "analysis/construction_residual.float16.npy",mmap_mode="r")
    train,evaluate=semantic_train_eval(); op_res=np.lib.format.open_memmap(out/"analysis/operation_residual.float16.npy",mode="w+",dtype=np.float16,shape=(len(OPERATIONS),37,6,DIM)); family_res=np.lib.format.open_memmap(out/"analysis/family_operation_residual.float16.npy",mode="w+",dtype=np.float16,shape=(len(eligible),len(OPERATIONS),37,6,DIM))
    m2b,m3loo,m3seen=metric_buckets(),metric_buckets(),metric_buckets(); node_rows=[]
    for q in range(37):
        for role in range(6):
            xt,yt=record_arrays(states,train,q,role); p2=np.asarray(slope[q,role],np.float32)*xt+np.asarray(intercept[q,role],np.float32)
            for ci,c in enumerate(CONSTRUCTIONS):
                mask=np.asarray([row["construction"]==c for row in train]); p2[mask]+=np.asarray(construction[ci,q,role],np.float32)
            residual2=yt-p2
            for oi,op in enumerate(OPERATIONS):
                om=np.asarray([row["operation"]==op for row in train]); mean=residual2[om].mean(0);op_res[oi,q,role]=mean.astype(np.float16)
                for fi,family in enumerate(eligible):
                    fm=np.asarray([row["operation"]==op and row["family"]==family for row in train]);family_res[fi,oi,q,role]=(residual2[fm].mean(0)-mean).astype(np.float16)
            x,y=record_arrays(states,evaluate,q,role); base=np.asarray(slope[q,role],np.float32)*x+np.asarray(intercept[q,role],np.float32)
            for ci,c in enumerate(CONSTRUCTIONS):
                mask=np.asarray([row["construction"]==c for row in evaluate]);base[mask]+=np.asarray(construction[ci,q,role],np.float32)
            pred_loo=base.copy();pred_seen=base.copy()
            for i,row in enumerate(evaluate):
                fi,oi=f_index[row["family"]],o_index[row["operation"]];op=np.asarray(op_res[oi,q,role],np.float32);fr=np.asarray(family_res[fi,oi,q,role],np.float32);loo=op-fr/max(len(eligible)-1,1);pred_loo[i]+=loo;pred_seen[i]+=op+fr
            add_buckets(m2b,evaluate,base,y);add_buckets(m3loo,evaluate,pred_loo,y);add_buckets(m3seen,evaluate,pred_seen,y)
            for family,op in itertools.product(eligible,OPERATIONS):
                mask=np.asarray([row["family"]==family and row["operation"]==op for row in evaluate]);truth=float(np.sum(y[mask]*y[mask],dtype=np.float64));e2=float(np.sum((base[mask]-y[mask])**2,dtype=np.float64));e3=float(np.sum((pred_loo[mask]-y[mask])**2,dtype=np.float64));node_rows.append({"family":family,"operation":op,"checkpoint":q,"role":ROLES[role],"m2_nrmse":math.sqrt(e2/(truth+1e-12)),"m3_loo_nrmse":math.sqrt(e3/(truth+1e-12)),"gain":math.sqrt(e2/(truth+1e-12))-math.sqrt(e3/(truth+1e-12))})
        op_res.flush();family_res.flush();print(f"[C463] q={q}",flush=True)
    for value in (states,slope,intercept,construction,op_res,family_res):close_mmap(value)
    metrics={"m2":finish_buckets(m2b),"m3_loo":finish_buckets(m3loo),"m3_seen":finish_buckets(m3seen)};save(out/"analysis/metrics.json",metrics);write_rows(out/"analysis/node_metrics.jsonl",node_rows)
    headline={"status":"semantic_increment_models_closed","ran":True,"eligible_families":eligible,"nrmse":{name:value["all"]["nrmse"] for name,value in metrics.items()},"strict_interpretation":"M3-LOO is the only cross-family semantic-selection test; M3-seen may contain family memorization."}
    close("C463",headline,{"finite":finite(headline),"nodes":len(node_rows)==len(eligible)*len(OPERATIONS)*37*6},"C464_adjudication")


def c464() -> None:
    out=begin("C464",{"status":"semantic_residual_lockbox_gate_frozen","pass":{"semantic_gain":0.02,"semantic_minus_surface":0.01,"family_wins":10,"unseen_report_gain":0.01},"semantic_operations":["statement","query"],"surface_control":"surface"},{"parent":final("C463")["all_checks_passed"]})
    if not final("C463")["headline"].get("ran"):
        close("C464",{"status":"semantic_gate_not_run","semantic_residual_candidate":False},{"route_accounted":True},"C465_coupling");return
    metrics=load(OUTS["C463"]/"analysis/metrics.json");m2,m3=metrics["m2"],metrics["m3_loo"]
    gains={op:m2["by_operation"][op]["nrmse"]-m3["by_operation"][op]["nrmse"] for op in OPERATIONS};semantic_gain=float(np.mean([gains["statement"],gains["query"]]));surface_gain=gains["surface"]
    families=final("C463")["headline"]["eligible_families"];family_gains={};wins=0
    for family in families:
        gain=float(np.mean([m2["by_family_operation"][f"{family}::{op}"]["nrmse"]-m3["by_family_operation"][f"{family}::{op}"]["nrmse"] for op in ("statement","query") ]));family_gains[family]=gain;wins+=gain>0.01
    report_gain=m2["by_construction"]["report"]["nrmse"]-m3["by_construction"]["report"]["nrmse"]
    candidate=semantic_gain>0.02 and semantic_gain>surface_gain+0.01 and wins>=min(10,len(families)) and report_gain>0.01
    headline={"status":"semantic_residual_adjudication_closed","semantic_gain":semantic_gain,"surface_gain":surface_gain,"operation_gains":gains,"family_gains":family_gains,"family_wins":wins,"unseen_report_gain":report_gain,"semantic_residual_candidate":candidate,"strict_interpretation":"A pass is a cross-family predictive increment after registered baselines, not a causal semantic circuit."}
    save(out/"analysis/gate.json",headline);close("C464",headline,{"finite":finite(headline)},"C465_coupling")


def c465() -> None:
    out=begin("C465",{"status":"neighbor_coordinate_coupling_frozen","model":"after M3-seen, predict residual with shared offsets -2,-1,0,+1,+2 for every physical coordinate","fit":"five scalar weights per operation/checkpoint/role using all coordinates","pass":"semantic NRMSE gain over M3-seen >=0.01","claim":"structured neighbor coupling only, not arbitrary full matrix"},{"parent":final("C464")["all_checks_passed"]})
    if not final("C463")["headline"].get("ran"):
        close("C465",{"status":"coupling_not_run","neighbor_coupling_candidate":False},{"route_accounted":True},"C466_rollout");return
    eligible=final("C463")["headline"]["eligible_families"];fi={f:i for i,f in enumerate(eligible)};oi={o:i for i,o in enumerate(OPERATIONS)};ci={c:i for i,c in enumerate(CONSTRUCTIONS)};offsets=(-2,-1,0,1,2)
    states=np.load(OUTS["C458"]/"raw/role_states.float16.npy",mmap_mode="r");slope=np.load(OUTS["C461"]/"analysis/slope.float16.npy",mmap_mode="r");intercept=np.load(OUTS["C461"]/"analysis/intercept.float16.npy",mmap_mode="r");construction=np.load(OUTS["C462"]/"analysis/construction_residual.float16.npy",mmap_mode="r");opres=np.load(OUTS["C463"]/"analysis/operation_residual.float16.npy",mmap_mode="r");famres=np.load(OUTS["C463"]/"analysis/family_operation_residual.float16.npy",mmap_mode="r");train,evaluate=semantic_train_eval();weights=np.zeros((3,37,6,5),np.float32);base_buckets,coupled_buckets=metric_buckets(),metric_buckets()
    for q in range(37):
        for role in range(6):
            xt,yt=record_arrays(states,train,q,role);base=np.asarray(slope[q,role],np.float32)*xt+np.asarray(intercept[q,role],np.float32)
            for i,row in enumerate(train):base[i]+=np.asarray(construction[ci[row["construction"]],q,role],np.float32)+np.asarray(opres[oi[row["operation"]],q,role],np.float32)+np.asarray(famres[fi[row["family"]],oi[row["operation"]],q,role],np.float32)
            for op in OPERATIONS:
                mask=np.asarray([row["operation"]==op for row in train]);xop=xt[mask];target=(yt-base)[mask];gram=np.zeros((5,5),np.float64);rhs=np.zeros(5,np.float64)
                features=[np.roll(xop,shift,axis=1) for shift in offsets]
                for a in range(5):
                    rhs[a]=float(np.sum(features[a]*target,dtype=np.float64))
                    for b in range(5):gram[a,b]=float(np.sum(features[a]*features[b],dtype=np.float64))
                weights[oi[op],q,role]=np.linalg.solve(gram+np.eye(5)*1e-3,rhs).astype(np.float32)
            x,y=record_arrays(states,evaluate,q,role);base_eval=np.asarray(slope[q,role],np.float32)*x+np.asarray(intercept[q,role],np.float32);coupled=base_eval.copy()
            for i,row in enumerate(evaluate):
                base_eval[i]+=np.asarray(construction[ci[row["construction"]],q,role],np.float32)+np.asarray(opres[oi[row["operation"]],q,role],np.float32)+np.asarray(famres[fi[row["family"]],oi[row["operation"]],q,role],np.float32)
            coupled[:]=base_eval
            for op in OPERATIONS:
                mask=np.asarray([row["operation"]==op for row in evaluate]);w=weights[oi[op],q,role]
                coupled[mask]+=sum(w[k]*np.roll(x[mask],offsets[k],axis=1) for k in range(5))
            add_buckets(base_buckets,evaluate,base_eval,y);add_buckets(coupled_buckets,evaluate,coupled,y)
        print(f"[C465] q={q}",flush=True)
    for value in (states,slope,intercept,construction,opres,famres):close_mmap(value)
    np.save(out/"analysis/neighbor_weights.float32.npy",weights);metrics={"m3_seen":finish_buckets(base_buckets),"neighbor":finish_buckets(coupled_buckets)};save(out/"analysis/metrics.json",metrics)
    gains={op:metrics["m3_seen"]["by_operation"][op]["nrmse"]-metrics["neighbor"]["by_operation"][op]["nrmse"] for op in OPERATIONS};semantic=float(np.mean([gains["statement"],gains["query"]]));candidate=semantic>=0.01
    headline={"status":"neighbor_coupling_closed","operation_gains":gains,"semantic_gain":semantic,"neighbor_coupling_candidate":candidate,"strict_interpretation":"The model tests only five translation-invariant coordinate offsets; failure does not exclude arbitrary cross-coordinate coupling."}
    close("C465",headline,{"finite":finite(headline)},"C466_rollout")


def c466() -> None:
    out=begin("C466",{"status":"autonomous_multistep_rollout_frozen","starts":[0,8,16,24],"steps":[1,2,4,8],"models":["identity hold","M3-seen recursive"],"evaluation":"all lockbox response records","pass":"M3 beats identity at steps2,4,8 and step8 error <=2 times step1"},{"parent":final("C465")["all_checks_passed"]})
    if not final("C463")["headline"].get("ran"):
        close("C466",{"status":"rollout_not_run","multistep_candidate":False},{"route_accounted":True},"C467_graph") ;return
    eligible=final("C463")["headline"]["eligible_families"];fi={f:i for i,f in enumerate(eligible)};oi={o:i for i,o in enumerate(OPERATIONS)};ci={c:i for i,c in enumerate(CONSTRUCTIONS)};records=[r for r in read_rows(OUTS["C459"]/"analysis/response_records.jsonl") if r["partition"]=="lockbox"]
    states=np.load(OUTS["C458"]/"raw/role_states.float16.npy",mmap_mode="r");slope=np.load(OUTS["C461"]/"analysis/slope.float16.npy",mmap_mode="r");intercept=np.load(OUTS["C461"]/"analysis/intercept.float16.npy",mmap_mode="r");construction=np.load(OUTS["C462"]/"analysis/construction_residual.float16.npy",mmap_mode="r");opres=np.load(OUTS["C463"]/"analysis/operation_residual.float16.npy",mmap_mode="r");famres=np.load(OUTS["C463"]/"analysis/family_operation_residual.float16.npy",mmap_mode="r")
    acc={k:{"m3":new_accumulator(),"identity":new_accumulator()} for k in (1,2,4,8)}
    left=np.asarray([r["left"] for r in records]);right=np.asarray([r["right"] for r in records])
    for start in (0,8,16,24):
        for role in range(6):
            current=np.asarray(states[right,start,role],np.float32)-np.asarray(states[left,start,role],np.float32);initial=current.copy()
            for step in range(1,9):
                q=start+step-1
                prediction=np.asarray(slope[q,role],np.float32)*current+np.asarray(intercept[q,role],np.float32)
                for i,row in enumerate(records):prediction[i]+=np.asarray(construction[ci[row["construction"]],q,role],np.float32)+np.asarray(opres[oi[row["operation"]],q,role],np.float32)+np.asarray(famres[fi[row["family"]],oi[row["operation"]],q,role],np.float32)
                current=prediction
                if step in acc:
                    truth=np.asarray(states[right,start+step,role],np.float32)-np.asarray(states[left,start+step,role],np.float32);add_metric(acc[step]["m3"],current,truth);add_metric(acc[step]["identity"],initial,truth)
        print(f"[C466] start={start}",flush=True)
    for value in (states,slope,intercept,construction,opres,famres):close_mmap(value)
    metrics={str(k):{name:finish_metric(value) for name,value in rows.items()} for k,rows in acc.items()};candidate=all(metrics[str(k)]["m3"]["nrmse"]<metrics[str(k)]["identity"]["nrmse"] for k in (2,4,8)) and metrics["8"]["m3"]["nrmse"]<=2*metrics["1"]["m3"]["nrmse"]
    headline={"status":"multistep_rollout_closed","metrics":metrics,"multistep_candidate":candidate,"strict_interpretation":"Recursive response rollout is not natural HiddenState generation; it tests stability of the fitted response law only."};save(out/"analysis/metrics.json",metrics);close("C466",headline,{"finite":finite(headline)},"C467_graph")


def c467() -> None:
    out=begin("C467",{"status":"fresh_graph_path_material_behavior_frozen","modes":list(GRAPH_MODES),"depths":[1,2,3,4],"constructions":list(CONSTRUCTIONS),"units":10,"codebook":"A Yes; B No","gates":{"heldout":0.75,"mode":0.60,"depth":0.65,"construction":0.65}},{"parent":final("C466")["all_checks_passed"],"cuda":torch.cuda.is_available()})
    rows=graph_material();write_rows(out/"material/cases.jsonl",rows);zero={mode:float(np.mean([r["gold_position"]==0 for r in rows if r["mode"]==mode])) for mode in GRAPH_MODES};tokenizer=graph_base.axis_old.base.parent.fresh.tokenizer_qwen();compiled=compile_base.compile_qwen(tokenizer,rows);write_rows(out/"compiled/qwen3.jsonl",compiled);run=graph_base.axis_old.base.parent.previous.qwen_behavior(rows,compiled,out,batch_size=12);behavior=read_rows(out/"raw/behavior.jsonl");lookup={r["case_id"]:r for r in rows};held=[r for r in behavior if lookup[r["case_id"]]["partition"]!="discovery"]
    by_mode={mode:float(np.mean([r["correct"] for r in held if lookup[r["case_id"]]["mode"]==mode])) for mode in GRAPH_MODES};by_depth={str(d):float(np.mean([r["correct"] for r in held if lookup[r["case_id"]]["depth"]==d])) for d in range(1,5)};by_construction={c:float(np.mean([r["correct"] for r in held if lookup[r["case_id"]]["construction"]==c])) for c in CONSTRUCTIONS};heldout=float(np.mean([r["correct"] for r in held]));authorized=heldout>=.75 and min(by_mode.values())>=.60 and min(by_depth.values())>=.65 and min(by_construction.values())>=.65
    headline={"status":"graph_behavior_closed",**run,"rows":len(rows),"truth_frequency":float(np.mean([r["gold_position"]==0 for r in rows])),"mode_truth_frequency":zero,"heldout_accuracy":heldout,"mode_accuracy":by_mode,"depth_accuracy":by_depth,"construction_accuracy":by_construction,"graph_field_authorized":authorized,"strict_interpretation":"Behavior qualifies only the registered graph interface."};close("C467",headline,{"rows":len(rows)==720,"finite":finite(headline)},"C468_graph_field")


def c468() -> None:
    out=begin("C468",{"status":"graph_path_field_nonlinear_integration_frozen","field":"all 720 cases x all checkpoints x six roles x all coordinates","response":"chain minus broken at each depth","model":"per-coordinate affine depth transition fit on discovery register/circular depth1->2 and2->3","controls":["identity","zero","training target mean"],"evaluation":"confirmation and lockbox, including unseen report construction"},{"parent":final("C467")["all_checks_passed"]})
    if not final("C467")["headline"]["graph_field_authorized"]:
        close("C468",{"status":"graph_field_not_run","field_ran":False,"depth_transition_candidate":False},{"route_accounted":True},"C469_writer");return
    rows=read_rows(OUTS["C467"]/"material/cases.jsonl");compiled=read_rows(OUTS["C467"]/"compiled/qwen3.jsonl");full_ids=set(r["case_id"] for r in rows if r["partition"]=="lockbox" and r["construction"]=="report" and r["mode"] in ("chain","broken") and r["depth"] in (1,4));run=common.batch_capture_qwen(rows,compiled,out,full_selector=lambda row:row["case_id"] in full_ids,batch_size=8,field_width=FIELD_WIDTH)
    states=np.load(out/"raw/role_states.float16.npy",mmap_mode="r");index=read_rows(out/"raw/hidden_index.jsonl");keyed={r["case_id"]:r for r in index};responses=[]
    for unit,c,d in itertools.product(range(10),CONSTRUCTIONS,range(1,5)):
        a=f"c467-chain-{c}-u{unit}-d{d}";b=f"c467-broken-{c}-u{unit}-d{d}"
        if a in keyed and b in keyed:responses.append({"unit":unit,"construction":c,"depth":d,"partition":partition(unit),"left":keyed[b]["hidden_index"],"right":keyed[a]["hidden_index"]})
    train_pairs=[]
    for unit,c,d in itertools.product(range(5),CONSTRUCTIONS[:2],(1,2)):
        left=next(r for r in responses if r["unit"]==unit and r["construction"]==c and r["depth"]==d);right=next(r for r in responses if r["unit"]==unit and r["construction"]==c and r["depth"]==d+1);train_pairs.append((left,right))
    eval_pairs=[]
    for unit,c,d in itertools.product(range(5,10),CONSTRUCTIONS,(2,3)):
        left=next(r for r in responses if r["unit"]==unit and r["construction"]==c and r["depth"]==d);right=next(r for r in responses if r["unit"]==unit and r["construction"]==c and r["depth"]==d+1);eval_pairs.append((left,right))
    slope=np.lib.format.open_memmap(out/"analysis/depth_slope.float16.npy",mode="w+",dtype=np.float16,shape=(37,6,DIM));intercept=np.lib.format.open_memmap(out/"analysis/depth_intercept.float16.npy",mode="w+",dtype=np.float16,shape=(37,6,DIM));depth_mean=np.lib.format.open_memmap(out/"analysis/depth_mean.float16.npy",mode="w+",dtype=np.float16,shape=(4,38,6,DIM));acc={name:new_accumulator() for name in ("affine","identity","zero","mean")}
    for depth in range(1,5):
        subset=[r for r in responses if r["depth"]==depth and r["partition"]=="discovery" and r["construction"] in CONSTRUCTIONS[:2]];left=np.asarray([r["left"] for r in subset]);right=np.asarray([r["right"] for r in subset]);depth_mean[depth-1]=np.mean(np.asarray(states[right],np.float32)-np.asarray(states[left],np.float32),axis=0).astype(np.float16);depth_mean.flush()
    for q in range(37):
        for role in range(6):
            xt=[];yt=[]
            for left_row,right_row in train_pairs:
                xl=np.asarray(states[left_row["right"],q,role],np.float32)-np.asarray(states[left_row["left"],q,role],np.float32);yl=np.asarray(states[right_row["right"],q+1,role],np.float32)-np.asarray(states[right_row["left"],q+1,role],np.float32);xt.append(xl);yt.append(yl)
            a,b=fit_diagonal(np.stack(xt),np.stack(yt));slope[q,role]=a.astype(np.float16);intercept[q,role]=b.astype(np.float16)
            xe=[];ye=[];means=[]
            for left_row,right_row in eval_pairs:
                xe.append(np.asarray(states[left_row["right"],q,role],np.float32)-np.asarray(states[left_row["left"],q,role],np.float32));ye.append(np.asarray(states[right_row["right"],q+1,role],np.float32)-np.asarray(states[right_row["left"],q+1,role],np.float32));means.append(np.asarray(depth_mean[right_row["depth"]-1,q+1,role],np.float32))
            x=np.stack(xe);y=np.stack(ye);add_metric(acc["affine"],a*x+b,y);add_metric(acc["identity"],x,y);add_metric(acc["zero"],np.zeros_like(y),y);add_metric(acc["mean"],np.stack(means),y)
        slope.flush();intercept.flush();print(f"[C468] q={q}",flush=True)
    for value in (states,slope,intercept,depth_mean):close_mmap(value)
    metrics={name:finish_metric(value) for name,value in acc.items()};candidate=metrics["affine"]["nrmse"]<min(metrics[k]["nrmse"] for k in ("identity","zero","mean"));headline={"status":"graph_path_integration_closed",**run,"field_ran":True,"response_pairs":len(responses),"metrics":metrics,"depth_transition_candidate":candidate,"strict_interpretation":"A pass is a response-depth predictor for this graph interface, not proof of recursive symbolic reasoning."};save(out/"analysis/metrics.json",metrics);write_rows(out/"analysis/response_records.jsonl",responses);close("C468",headline,{"shape":run["role_shape"][1:]==[38,6,2560],"finite":finite(headline)},"C469_writer")


@torch.inference_mode()
def score_with_patch(model,device,row:dict,layer_index:int|None,positions:list[int],delta:np.ndarray|None)->list[float]:
    hook=None
    if layer_index is not None and delta is not None:
        value=torch.tensor(delta,dtype=torch.float32,device=device)
        def patch(_module,_args,output):
            state=output[0] if isinstance(output,tuple) else output;changed=state.clone();changed[0,positions]+=value.to(changed.dtype);return (changed,*output[1:]) if isinstance(output,tuple) else changed
        hook=model.model.layers[layer_index].register_forward_hook(patch)
    try:
        ids=torch.tensor([row["prompt_ids"]],dtype=torch.long,device=device);output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True);return [float(output.logits[0,-1,c[0]]) for c in row["candidate_ids"]]
    finally:
        if hook:hook.remove()


def c469() -> None:
    out=begin("C469",{"status":"conditional_natural_semantic_writer_frozen","qualification":"C464 semantic residual candidate","selection":"maximum confirmation M3-LOO gain among semantic nodes, checkpoints8-30; deterministic lexical tie order","lockbox":"report units8-9","conditions":["natural base","natural target","predicted full response","actual full response","shared propagation","semantic residual only","surface residual","wrong family","coordinate roll","wrong role","wrong checkpoint"],"pass":"predicted shift positive, greater than shared and every mismatch, target rate >=0.60"},{"parent":final("C468")["all_checks_passed"]})
    if not final("C464")["headline"]["semantic_residual_candidate"]:
        close("C469",{"status":"writer_not_run_semantic_residual_ineligible","writer_ran":False,"specificity_passed":False,"strict_interpretation":"No natural-model causal result."},{"route_accounted":True},"C470_synthesis");return
    nodes=read_rows(OUTS["C463"]/"analysis/node_metrics.jsonl");eligible=[r for r in nodes if r["operation"] in ("statement","query") and 8<=r["checkpoint"]<=30 and r["gain"]>0];selection=max(eligible,key=lambda r:(r["gain"],r["family"],r["operation"],r["role"],-r["checkpoint"]));save(out/"protocol/writer_selection.json",selection)
    families=final("C463")["headline"]["eligible_families"];fi={f:i for i,f in enumerate(families)};oi={o:i for i,o in enumerate(OPERATIONS)};ci={c:i for i,c in enumerate(CONSTRUCTIONS)};q=selection["checkpoint"];role=ROLES.index(selection["role"]);states=np.load(OUTS["C458"]/"raw/role_states.float16.npy",mmap_mode="r");slope=np.load(OUTS["C461"]/"analysis/slope.float16.npy",mmap_mode="r");intercept=np.load(OUTS["C461"]/"analysis/intercept.float16.npy",mmap_mode="r");construction=np.load(OUTS["C462"]/"analysis/construction_residual.float16.npy",mmap_mode="r");opres=np.load(OUTS["C463"]/"analysis/operation_residual.float16.npy",mmap_mode="r");famres=np.load(OUTS["C463"]/"analysis/family_operation_residual.float16.npy",mmap_mode="r");records=read_rows(OUTS["C459"]/"analysis/response_records.jsonl");index=read_rows(OUTS["C458"]/"raw/hidden_index.jsonl");hidden={r["hidden_index"]:r for r in index};compiled={r["case_id"]:r for r in read_rows(OUTS["C457"]/"compiled/qwen3.jsonl")};targets=[r for r in records if r["family"]==selection["family"] and r["operation"]==selection["operation"] and r["partition"]=="lockbox" and r["construction"]=="report"]
    wrong_family=families[(fi[selection["family"]]+1)%len(families)];model=None;trials=[]
    try:
        model,_tok,device,_placement=model_base.load_bf16("qwen3")
        for record in targets:
            left_case=hidden[record["left"]]["case_id"];right_case=hidden[record["right"]]["case_id"];left=compiled[left_case];right=compiled[right_case];x=np.asarray(states[record["right"],q,role],np.float32)-np.asarray(states[record["left"],q,role],np.float32);actual=np.asarray(states[record["right"],q+1,role],np.float32)-np.asarray(states[record["left"],q+1,role],np.float32);shared=np.asarray(slope[q,role],np.float32)*x+np.asarray(intercept[q,role],np.float32)+np.asarray(construction[ci[record["construction"]],q,role],np.float32);semantic=np.asarray(opres[oi[record["operation"]],q,role],np.float32)+np.asarray(famres[fi[record["family"]],oi[record["operation"]],q,role],np.float32);surface=np.asarray(opres[oi["surface"],q,role],np.float32)+np.asarray(famres[fi[record["family"]],oi["surface"],q,role],np.float32);wrong=np.asarray(opres[oi[record["operation"]],q,role],np.float32)+np.asarray(famres[fi[wrong_family],oi[record["operation"]],q,role],np.float32);pred=shared+semantic;positions=left["role_positions"][selection["role"]];wrong_role=ROLES[(role+1)%6]
            conditions={"natural_base":(None,None,positions),"natural_target":(None,None,right["role_positions"][selection["role"]]),"predicted_full":(q,pred,positions),"actual_full":(q,actual,positions),"shared":(q,shared,positions),"semantic_only":(q,semantic,positions),"surface":(q,surface,positions),"wrong_family":(q,wrong,positions),"coordinate_roll":(q,np.roll(semantic,257),positions),"wrong_role":(q,semantic,left["role_positions"][wrong_role]),"wrong_checkpoint":(min(q+1,35),semantic,positions)};scores={name:score_with_patch(model,device,right if name=="natural_target" else left,layer,pos,delta) for name,(layer,delta,pos) in conditions.items()};base=scores["natural_base"][1]-scores["natural_base"][0];margins={name:value[1]-value[0] for name,value in scores.items()};trials.append({"left_case":left_case,"right_case":right_case,"margins":margins,"shifts":{name:value-base for name,value in margins.items() if name not in ("natural_base","natural_target")},"predicted_target_choice":int(np.argmax(scores["predicted_full"]))==right["gold_position"]})
    finally:model_base.release_bf16(model)
    for value in (states,slope,intercept,construction,opres,famres):close_mmap(value)
    write_rows(out/"raw/writer_trials.jsonl",trials);names=("predicted_full","actual_full","shared","semantic_only","surface","wrong_family","coordinate_roll","wrong_role","wrong_checkpoint");shifts={name:float(np.median([r["shifts"][name] for r in trials])) for name in names};rate=float(np.mean([r["predicted_target_choice"] for r in trials]));passed=shifts["predicted_full"]>0 and shifts["predicted_full"]>max(shifts[k] for k in ("shared","surface","wrong_family","coordinate_roll","wrong_role","wrong_checkpoint")) and rate>=.60
    headline={"status":"natural_semantic_writer_closed","writer_ran":True,"selection":selection,"trials":len(trials),"median_margin_shifts":shifts,"predicted_target_choice_rate":rate,"specificity_passed":passed,"strict_interpretation":"A pass is narrow sufficiency of one response writer; it is not necessity, uniqueness, or full semantic generation."};close("C469",headline,{"trials":len(trials)>=4,"finite":finite(headline)},"C470_synthesis")


def hash_remove(paths:list[Path],out:Path)->list[dict]:
    rows=[]
    for path in paths:
        if not path.exists():continue
        h=hashlib.sha256();size=path.stat().st_size
        with path.open("rb") as handle:
            while chunk:=handle.read(8*1024*1024):h.update(chunk)
        rows.append({"path":str(path.relative_to(ROOT)),"sha256":h.hexdigest(),"bytes":size,"deleted":True});path.unlink()
    save(out/"audit/cleanup.json",rows);return rows


def c470() -> None:
    out=begin("C470",{"status":"campaign_synthesis_visual_cleanup_audit_frozen","visual":"operation residuals plus graph depth means, every selected row keeps all 2560 coordinates","cleanup":"hash and delete nonvisual raw fields and large coefficient arrays after all dependent routes close","new_math_gate":"requires cross-family semantic residual, stable multistep prediction, path-specific causal evidence, graph-depth transfer, and a separately registered cross-model composition result; this campaign cannot satisfy the final cross-model clause"},{"parent":final("C469")["all_checks_passed"]})
    visual=[]
    if final("C463")["headline"].get("ran"):
        opres=np.load(OUTS["C463"]/"analysis/operation_residual.float16.npy",mmap_mode="r")
        for oi,op in enumerate(OPERATIONS):
            for q,role in itertools.product(range(37),range(6)):
                visual.append({"id":f"semantic:{op}:q{q}:{ROLES[role]}","source":"operation_residual","operation":op,"checkpoint":q+1,"role":ROLES[role],"values":np.asarray(opres[oi,q,role],np.float32).round(6).tolist()})
        close_mmap(opres)
    if final("C468")["headline"].get("field_ran"):
        depth=np.load(OUTS["C468"]/"analysis/depth_mean.float16.npy",mmap_mode="r")
        for d,q,role in itertools.product(range(4),range(38),range(6)):
            visual.append({"id":f"graph:depth{d+1}:q{q}:{ROLES[role]}","source":"graph_depth_response","depth":d+1,"checkpoint":q,"role":ROLES[role],"values":np.asarray(depth[d,q,role],np.float32).round(6).tolist()})
        close_mmap(depth)
    payload={"schema":"c470.semantic-residual-graph.v1","phase":2004,"campaign":"C454-C470","dimensions":list(range(DIM)),"rows":visual,
             "summary":{"behavior":final("C457")["headline"],"semantic_gate":final("C464")["headline"],"coupling":final("C465")["headline"],"rollout":final("C466")["headline"],"graph":final("C468")["headline"],"writer":final("C469")["headline"]},
             "claim_boundary":"Full-coordinate predictive responses are not model parameters, semantic neurons, or a unique causal circuit."};save(VISUAL,payload)
    cleanup_paths=[OUTS["C458"]/"raw/role_states.float16.npy",OUTS["C458"]/"raw/full_fields_holdout.float16.npy",OUTS["C460"]/"analysis/mean_next.float16.npy",OUTS["C461"]/"analysis/slope.float16.npy",OUTS["C461"]/"analysis/intercept.float16.npy",OUTS["C462"]/"analysis/construction_residual.float16.npy",OUTS["C463"]/"analysis/operation_residual.float16.npy",OUTS["C463"]/"analysis/family_operation_residual.float16.npy",OUTS["C468"]/"raw/role_states.float16.npy",OUTS["C468"]/"raw/full_fields_holdout.float16.npy",OUTS["C468"]/"analysis/depth_slope.float16.npy",OUTS["C468"]/"analysis/depth_intercept.float16.npy",OUTS["C468"]/"analysis/depth_mean.float16.npy"]
    cleanup=hash_remove(cleanup_paths,out);semantic=final("C464")["headline"]["semantic_residual_candidate"];multistep=final("C466")["headline"]["multistep_candidate"];causal=final("C469")["headline"]["specificity_passed"];graph_depth=final("C468")["headline"]["depth_transition_candidate"];cross_model_composition=False;new_math=bool(semantic and multistep and causal and graph_depth and cross_model_composition);next_same=bool(semantic and not causal)
    headline={"status":"semantic_residual_campaign_closed","gates":{"eligible_families":final("C457")["headline"]["eligible_families"],"shared_propagation":final("C461")["headline"]["shared_candidate"],"semantic_residual":semantic,"neighbor_coupling":final("C465")["headline"]["neighbor_coupling_candidate"],"multistep":multistep,"graph_depth_transition":graph_depth,"natural_writer":causal,"cross_model_composition":cross_model_composition},"visual_rows":len(visual),"visual_path":str(VISUAL.relative_to(ROOT)),"cleanup_files":len(cleanup),"cleanup_bytes":sum(r["bytes"] for r in cleanup),"new_math_gate_passed":new_math,"next_stage_same_goal":next_same,"strict_interpretation":"The campaign separates registered propagation baselines from predictive semantic increments; it cannot infer a universal language algebra, new mathematics, or a unique circuit."}
    close("C470",headline,{"visual_schema":payload["schema"].startswith("c470"),"full_coordinates":all(len(row["values"])==DIM for row in visual),"cleanup":all(not ROOT.joinpath(r["path"]).exists() for r in cleanup),"finite":finite(headline)},"independent_audit_then_registered_next_stage")


RUNNERS={name:globals()[name.lower()] for name in PHASES}


def validate_only() -> None:
    semantic=semantic_material();graph=graph_material();checks={
        "phase_sequence":[PHASES[f"C{i}"][0] for i in range(454,471)]==list(range(1988,2005)),
        "semantic_rows":len(semantic)==3840,"semantic_families":len({r["family"] for r in semantic})==16,
        "semantic_balance":sum(r["gold_position"]==0 for r in semantic)==len(semantic)//2,
        "semantic_codebook":all("(A) Yes (B) No" in r["prompt"] for r in semantic),
        "semantic_roles":all(all(str(v) in r["prompt_core"] for v in r["role_values"].values()) for r in semantic),
        "graph_rows":len(graph)==720,"graph_balance":sum(r["gold_position"]==0 for r in graph)==len(graph)//2,
        "graph_codebook":all("(A) Yes (B) No" in r["prompt"] for r in graph),
    }
    print(json.dumps(checks,ensure_ascii=False));assert all(checks.values())


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("campaign",nargs="?",choices=list(PHASES));parser.add_argument("--validate-only",action="store_true");args=parser.parse_args()
    if args.validate_only:validate_only();return
    names=[args.campaign] if args.campaign else list(PHASES)
    for name in names:RUNNERS[name]()


if __name__=="__main__":main()
