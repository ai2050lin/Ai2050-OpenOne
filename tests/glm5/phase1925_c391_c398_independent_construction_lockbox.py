#!/usr/bin/env python3
"""C391-C398 independent-construction lockbox for three local candidates.

This campaign observes embeddings and HiddenStates only. It keeps every Qwen
activation coordinate and tests fresh lexical units and constructions for the
three C377 candidates without stopping one route when another fails.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase1903_c369_c390_language_operation_graph_campaign as previous
import phase1797_c263_c272_state_operator_common as family_base
from model_utils import MODEL_CONFIGS


PHASES = {
    "C391": (1925, "independent_construction_master_contract"),
    "C392": (1926, "fresh_construction_material_and_zero_models"),
    "C393": (1927, "qwen_fresh_construction_behavior"),
    "C394": (1928, "qwen_fresh_construction_full_coordinate_field"),
    "C395": (1929, "three_candidate_conditional_interaction_lockbox"),
    "C396": (1930, "output_sensitive_negation_scope_order"),
    "C397": (1931, "cross_construction_response_ecology"),
    "C398": (1932, "independent_construction_campaign_synthesis"),
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}" for name, (phase, slug) in PHASES.items()}
FAMILIES = ("causal_direction", "negation_scope", "attribute_binding")
SURFACES = ("archive", "interview", "bulletin", "ledger")
CELLS = ("00", "10", "01", "11_ab", "11_ba")
ROLES = previous.ROLES
DIM = 2560

UNITS = (
    {"primary": "Tarin", "secondary": "Vela", "observer": "Wystan", "object": "saffron", "other": "pulley"},
    {"primary": "Yorin", "secondary": "Zella", "observer": "Adair", "object": "topaz", "other": "winch"},
    {"primary": "Bram", "secondary": "Cyra", "observer": "Della", "object": "fennel", "other": "goblet"},
    {"primary": "Eamon", "secondary": "Fiora", "observer": "Galen", "object": "jasper", "other": "tripod"},
    {"primary": "Hale", "secondary": "Isla", "observer": "Jorin", "object": "clover", "other": "compass"},
    {"primary": "Kellan", "secondary": "Luma", "observer": "Merek", "object": "agate", "other": "lantern"},
    {"primary": "Naren", "secondary": "Olya", "observer": "Perrin", "object": "anise", "other": "caliper"},
    {"primary": "Quill", "secondary": "Rhea", "observer": "Selka", "object": "beryl", "other": "abacus"},
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


def producer_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def wrap(surface: str, target: str, noise: str, question: str, reverse: bool) -> str:
    left, right = (noise, target) if reverse else (target, noise)
    if surface == "archive":
        return f"Archive entry: {left} Unrelated entry: {right} From the archive alone, {question}"
    if surface == "interview":
        return f"In an interview, the witness said: {left} The witness separately noted: {right} Answer from the interview: {question}"
    if surface == "bulletin":
        return f"Bulletin. {left} Separate item. {right} Question. {question}"
    if surface == "ledger":
        return f"A ledger records the following. First, {left} Second, {right} Using only these records, {question}"
    raise KeyError(surface)


def construction(family: str, unit: int, cell: str) -> dict:
    u = UNITS[unit]
    p, s, o, obj, other = u["primary"], u["secondary"], u["observer"], u["object"], u["other"]
    a, b = (0, 0) if cell == "00" else (1, 0) if cell == "10" else (0, 1) if cell == "01" else (1, 1)
    order = "ba" if cell == "11_ba" else "ab"
    if family == "causal_direction":
        outcome = "shutdown" if b == 0 else "warning"
        if a == 0:
            target, relation = f"{p}'s intervention brought about the {outcome}.", "brought about"
        else:
            target, relation = f"The {outcome} occurred as a consequence of {p}'s intervention.", "as a consequence of"
        noise = f"{s} moved the {other}."
        question, correct, wrong = f"Whose intervention produced the {outcome}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": outcome, "query": outcome}
    elif family == "attribute_binding":
        value = "amber" if b == 0 else "violet"
        if a == 0:
            target, relation = f"{p}'s badge carries the {value} mark.", "carries"
        else:
            target, relation = f"The {value} mark is assigned to {p}'s badge.", "is assigned to"
        noise = f"{s}'s badge carries the silver mark."
        question, correct, wrong = f"Whose badge has the {value} mark?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": value, "query": value}
    elif family == "negation_scope":
        if cell == "00":
            target, relation, truth = f"{o} affirmed that {p} inspected the {obj}.", "affirmed that", True
        elif cell == "10":
            target, relation, truth = f"It is false that {p} inspected the {obj}, according to {o}.", "is false that", False
        elif cell == "01":
            target, relation, truth = f"{o} affirmed that {p} did not inspect the {obj}.", "did not inspect", False
        elif cell == "11_ab":
            target, relation, truth = f"It is false that it is false that {p} inspected the {obj}; {o} confirms this wording.", "false that it is false", True
        else:
            target, relation, truth = f"{o} affirmed that it is false that {p} inspected the {obj}.", "affirmed that it is false", False
        noise = f"{s} adjusted the {other}."
        question = f"Is it affirmed that {p} inspected the {obj}?"
        correct, wrong = ("Yes", "No") if truth else ("No", "Yes")
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": obj}
    else:
        raise KeyError(family)
    return {
        "target": target, "noise": noise, "question": question, "correct": correct, "wrong": wrong,
        "roles": roles, "factor_a": a, "factor_b": b, "composition_order": order,
    }


def material() -> list[dict]:
    rows = []
    for family, surface, unit, cell, answer_order in itertools.product(FAMILIES, SURFACES, range(len(UNITS)), CELLS, (1, -1)):
        case = construction(family, unit, cell)
        reverse = cell == "11_ba" and family != "negation_scope"
        prompt_core = wrap(surface, case["target"], case["noise"], case["question"], reverse)
        choices, gold = family_base.options(case["correct"], case["wrong"], answer_order)
        rows.append({
            "case_id": f"c392-{family}-{surface}-u{unit}-{cell}-{answer_order:+d}",
            "panel": "independent_construction", "family": family, "surface": surface, "unit": unit,
            "cell": cell, "factor_a": case["factor_a"], "factor_b": case["factor_b"],
            "composition_order": case["composition_order"], "order": answer_order,
            "partition": "discovery" if unit < 4 else "confirmation" if unit < 6 else "lockbox",
            "gold_position": gold, "correct_answer": case["correct"], "wrong_answer": case["wrong"],
            "prompt_core": prompt_core, "prompt": f"{prompt_core} {choices}. Reply with only A or B.",
            "free_prompt": f"{prompt_core} Answer with only the answer word.", "role_values": case["roles"],
            "semantic_graph": {"family": family, "factor_a": case["factor_a"], "factor_b": case["factor_b"], "composition_order": case["composition_order"]},
        })
    return rows


def tokenizer_qwen():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def close_memmap(value) -> None:
    mapping = getattr(value, "_mmap", None)
    if mapping is not None:
        mapping.close()


def grouped(states: np.ndarray, index: list[dict], family: str, partitions: tuple[str, ...], surfaces: tuple[str, ...] | None = None) -> list[dict]:
    rows = [row for row in index if row["correct"] and row["family"] == family and row["partition"] in partitions and (surfaces is None or row["surface"] in surfaces)]
    lookup = {
        (row["surface"], row["unit"], row["order"], row.get("cell") or row["case_id"].split("-")[-2]): row["hidden_index"]
        for row in rows
    }
    result = []
    for surface, unit, answer_order in itertools.product(SURFACES, range(len(UNITS)), (1, -1)):
        keys = {(surface, unit, answer_order, cell) for cell in CELLS}
        if all(key in lookup for key in keys):
            h = {cell: np.asarray(states[lookup[(surface, unit, answer_order, cell)]], np.float32) for cell in CELLS}
            result.append({
                "surface": surface, "unit": unit, "order": answer_order, "h00": h["00"],
                "I": h["11_ab"] - h["10"] - h["01"] + h["00"],
                "K": h["11_ab"] - h["11_ba"],
            })
    return result


def nrmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    p, t = np.asarray(prediction, np.float32), np.asarray(truth, np.float32)
    return float(np.sqrt(np.square(p - t, dtype=np.float64).sum() / max(np.square(t, dtype=np.float64).sum(), 1e-30)))


def candidate_metrics(train: list[dict], test: list[dict], old_prediction: np.ndarray, wrong_prediction: np.ndarray) -> dict:
    train_h = np.asarray([row["h00"] for row in train], np.float32)
    train_i = np.asarray([row["I"] for row in train], np.float32)
    test_h = np.asarray([row["h00"] for row in test], np.float32)
    truth = np.asarray([row["I"] for row in test], np.float32)
    mean = train_i.mean(axis=0)
    threshold = np.median(train_h, axis=0)
    high = train_h >= threshold
    high_mean = (train_i * high).sum(axis=0) / np.maximum(high.sum(axis=0), 1)
    low_mean = (train_i * (~high)).sum(axis=0) / np.maximum((~high).sum(axis=0), 1)
    conditional = np.where(test_h >= threshold, high_mean, low_mean)
    predictions = {
        "zero": np.zeros_like(truth), "discovery_mean": np.broadcast_to(mean, truth.shape),
        "conditional": conditional, "old_atlas": np.broadcast_to(old_prediction, truth.shape),
        "coordinate_roll": np.broadcast_to(np.roll(mean, 137, axis=-1), truth.shape),
        "wrong_family": np.broadcast_to(wrong_prediction, truth.shape),
    }
    scores = {name: nrmse(value, truth) for name, value in predictions.items()}
    return {
        "train_groups": len(train), "test_groups": len(test), "nrmse": scores,
        "conditional_gain_over_mean": scores["discovery_mean"] - scores["conditional"],
        "conditional_control_advantage": min(scores["coordinate_roll"], scores["wrong_family"]) - scores["conditional"],
        "old_gain_over_zero": scores["zero"] - scores["old_atlas"],
        "conditional_passed": scores["conditional"] < scores["discovery_mean"] and scores["conditional"] < min(scores["coordinate_roll"], scores["wrong_family"]),
        "old_transfer_passed": scores["old_atlas"] < min(scores["zero"], scores["coordinate_roll"], scores["wrong_family"]),
    }


def c391() -> None:
    out = begin("C391", {
        "status": "independent_construction_lockbox_frozen",
        "candidates": list(FAMILIES),
        "design": "fresh vocabulary x four fresh constructions x output-sensitive negation scope x full coordinates",
        "policy": "failure removes one candidate only; all registered routes run",
        "no_attention_mlp_weights": True,
    }, {"parent": previous.final("C390")["all_checks_passed"], "phase_continuity": PHASES["C391"][0] == 1925})
    close("C391", {
        "status": "master_contract_closed", "families": list(FAMILIES),
        "strict_interpretation": "The campaign follows three local candidates; it does not treat them as established language operators.",
    }, {"families": len(FAMILIES) == 3, "surfaces": len(SURFACES) == 4}, "C392_material")


def c392() -> None:
    out = begin("C392", {
        "status": "fresh_material_zero_model_frozen", "rows": "3 x 4 x 8 x 5 x 2",
        "partitions": {"discovery": [0, 1, 2, 3], "confirmation": [4, 5], "lockbox": [6, 7]},
        "zero_model_gate": 0.51, "human_naturalness": "not independently certified",
    }, {"parent": final("C391")["all_checks_passed"]})
    rows = material()
    write_rows(out / "material/cases.jsonl", rows)
    accuracies = {"always_first": float(np.mean([row["gold_position"] == 0 for row in rows])), "always_second": float(np.mean([row["gold_position"] == 1 for row in rows]))}
    for key in ("family", "surface", "cell"):
        correct = 0
        for value in sorted({row[key] for row in rows}):
            subset = [row for row in rows if row[key] == value]
            majority = int(np.mean([row["gold_position"] for row in subset]) >= 0.5)
            correct += sum(row["gold_position"] == majority for row in subset)
        accuracies[f"{key}_majority"] = correct / len(rows)
    roles = all(all(str(value) in row["prompt_core"] for value in row["role_values"].values()) for row in rows)
    scope_answers = {row["correct_answer"] for row in rows if row["family"] == "negation_scope" and row["cell"] in ("11_ab", "11_ba")}
    counts = {part: sum(row["partition"] == part for row in rows) for part in ("discovery", "confirmation", "lockbox")}
    eligible = max(accuracies.values()) <= 0.51 and roles and scope_answers == {"Yes", "No"}
    headline = {
        "status": "fresh_material_zero_models_closed", "rows": len(rows), "partition_counts": counts,
        "zero_model_accuracies": accuracies, "role_occurrence": roles, "scope_order_is_output_sensitive": scope_answers == {"Yes", "No"},
        "material_eligible": eligible, "human_naturalness_review": False,
        "strict_interpretation": "Exact balancing and output sensitivity do not certify naturalness or isolate a neural operator.",
    }
    close("C392", headline, {"rows": len(rows) == 960, "balance": max(accuracies.values()) <= 0.51, "roles": roles, "scope": scope_answers == {"Yes", "No"}}, "C393_behavior")


def c393() -> None:
    out = begin("C393", {
        "status": "qwen_fresh_behavior_frozen", "model": "Qwen3-4B BF16 CUDA", "no_hidden": True,
        "gates": {"overall": 0.75, "per_family": 0.60, "per_surface": 0.60},
    }, {"parent": final("C392")["all_checks_passed"], "material": final("C392")["headline"]["material_eligible"], "cuda": torch.cuda.is_available()})
    rows = read_rows(OUTS["C392"] / "material/cases.jsonl")
    compiled = family_base.compile_qwen(tokenizer_qwen(), rows)
    write_rows(out / "compiled/qwen3.jsonl", compiled)
    metrics = previous.qwen_behavior(rows, compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl")
    held = [row for row in behavior if row["partition"] in ("confirmation", "lockbox")]
    by_family = {family: float(np.mean([row["correct"] for row in held if row["family"] == family])) for family in FAMILIES}
    by_surface = {surface: float(np.mean([row["correct"] for row in held if row["surface"] == surface])) for surface in SURFACES}
    overall = float(np.mean([row["correct"] for row in held]))
    eligible = [family for family, score in by_family.items() if score >= 0.60]
    headline = {
        "status": "qwen_fresh_behavior_closed", **metrics, "heldout_accuracy": overall,
        "family_accuracy": by_family, "surface_accuracy": by_surface, "eligible_families": eligible,
        "field_eligible": overall >= 0.75 and min(by_surface.values()) >= 0.60 and len(eligible) >= 2,
        "strict_interpretation": "Only qualified families enter family-specific mechanism claims; other routes continue descriptively.",
    }
    close("C393", headline, {"rows": len(behavior) == 960, "finite": previous.finite(headline)}, "C394_field")


def c394() -> None:
    out = begin("C394", {
        "status": "qwen_fresh_full_coordinate_field_frozen",
        "archive": "960 x 38 x 6 x 2560 plus 12 all-token lockbox rows",
        "no_pca_topk_cosine_gate": True, "cleanup_after_synthesis": True,
    }, {"parent": final("C393")["all_checks_passed"], "field_eligible": final("C393")["headline"]["field_eligible"], "cuda": torch.cuda.is_available()})
    rows = read_rows(OUTS["C392"] / "material/cases.jsonl")
    compiled = read_rows(OUTS["C393"] / "compiled/qwen3.jsonl")
    selector = lambda row: row["partition"] == "lockbox" and row["surface"] == "ledger" and row["order"] == 1 and row["cell"] in ("00", "11_ab")
    metrics = previous.common.batch_capture_qwen(rows, compiled, out, full_selector=selector, batch_size=8, field_width=192)
    headline = {"status": "qwen_fresh_full_coordinate_field_closed", **metrics, "strict_interpretation": "The archive is observational and retains the complete physical activation axis."}
    close("C394", headline, {"role_shape": metrics["role_shape"] == [960, 38, 6, 2560], "full_rows": metrics["full_token_rows"] == 12, "finite": previous.finite(headline)}, "C395_conditional_lockbox")


def c395() -> None:
    out = begin("C395", {
        "status": "three_candidate_conditional_interaction_frozen",
        "training": "fresh discovery only", "test": "fresh confirmation plus lockbox",
        "comparators": ["zero", "fresh mean", "old C375 atlas", "coordinate roll", "wrong family"],
    }, {"parent": final("C394")["all_checks_passed"]})
    states = np.load(OUTS["C394"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C394"] / "raw/hidden_index.jsonl")
    old = np.load(previous.OUTS["C375"] / "analysis/family_operation_mean_response.float16.npy", mmap_mode="r")
    results = []
    for family_i, family in enumerate(FAMILIES):
        train = grouped(states, index, family, ("discovery",))
        test = grouped(states, index, family, ("confirmation", "lockbox"))
        old_i = previous.FAMILIES.index(family)
        wrong_i = previous.FAMILIES.index(FAMILIES[(family_i + 1) % len(FAMILIES)])
        value = candidate_metrics(train, test, np.asarray(old[old_i, previous.OPS.index("I")], np.float32), np.asarray(old[wrong_i, previous.OPS.index("I")], np.float32))
        results.append({"family": family, **value})
    write_rows(out / "analysis/candidate_results.jsonl", results)
    close_memmap(states); close_memmap(old); del states, old; gc.collect()
    headline = {
        "status": "conditional_interaction_lockbox_closed", "results": results,
        "conditional_passed": [row["family"] for row in results if row["conditional_passed"]],
        "old_transfer_passed": [row["family"] for row in results if row["old_transfer_passed"]],
        "strict_interpretation": "A pass is construction-specific full-coordinate prediction, not a universal or causal operator.",
    }
    close("C395", headline, {"families": len(results) == 3, "finite": previous.finite(headline)}, "C396_scope_order")


def c396() -> None:
    out = begin("C396", {
        "status": "output_sensitive_scope_order_frozen", "family": "negation_scope",
        "truth": "11_ab and 11_ba have different correct answers", "comparators": ["zero", "discovery mean", "old K", "coordinate roll", "wrong family K"],
    }, {"parent": final("C395")["all_checks_passed"]})
    states = np.load(OUTS["C394"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C394"] / "raw/hidden_index.jsonl")
    old = np.load(previous.OUTS["C375"] / "analysis/family_operation_mean_response.float16.npy", mmap_mode="r")
    train = grouped(states, index, "negation_scope", ("discovery",))
    test = grouped(states, index, "negation_scope", ("confirmation", "lockbox"))
    truth = np.asarray([row["K"] for row in test], np.float32)
    mean = np.asarray([row["K"] for row in train], np.float32).mean(axis=0)
    old_k = np.asarray(old[previous.FAMILIES.index("negation_scope"), previous.OPS.index("K")], np.float32)
    wrong = np.asarray(old[previous.FAMILIES.index("causal_direction"), previous.OPS.index("K")], np.float32)
    scores = {
        "zero": nrmse(np.zeros_like(truth), truth), "discovery_mean": nrmse(np.broadcast_to(mean, truth.shape), truth),
        "old_k": nrmse(np.broadcast_to(old_k, truth.shape), truth), "coordinate_roll": nrmse(np.broadcast_to(np.roll(mean, 137, axis=-1), truth.shape), truth),
        "wrong_family_k": nrmse(np.broadcast_to(wrong, truth.shape), truth),
    }
    new_pass = scores["discovery_mean"] < min(scores["zero"], scores["coordinate_roll"], scores["wrong_family_k"])
    old_pass = scores["old_k"] < min(scores["zero"], scores["coordinate_roll"], scores["wrong_family_k"])
    close_memmap(states); close_memmap(old); del states, old; gc.collect()
    headline = {
        "status": "output_sensitive_scope_order_closed", "train_groups": len(train), "test_groups": len(test),
        "nrmse": scores, "fresh_scope_order_passed": new_pass, "old_scope_order_transfer_passed": old_pass,
        "strict_interpretation": "The fresh contrast changes the correct answer; any response transfer remains tied to these explicit constructions.",
    }
    close("C396", headline, {"groups": len(train) > 0 and len(test) > 0, "finite": previous.finite(headline)}, "C397_ecology")


def c397() -> None:
    out = begin("C397", {
        "status": "cross_construction_response_ecology_frozen",
        "train": "archive/interview discovery", "test": "bulletin/ledger confirmation+lockbox",
        "classifier": "nearest full-coordinate I centroid by NRMSE; descriptive only",
    }, {"parent": final("C396")["all_checks_passed"]})
    states = np.load(OUTS["C394"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C394"] / "raw/hidden_index.jsonl")
    train = {family: grouped(states, index, family, ("discovery",), ("archive", "interview")) for family in FAMILIES}
    test = {family: grouped(states, index, family, ("confirmation", "lockbox"), ("bulletin", "ledger")) for family in FAMILIES}
    centroids = {family: np.asarray([row["I"] for row in values], np.float32).mean(axis=0) for family, values in train.items()}
    predictions = []
    for truth_family, values in test.items():
        for row in values:
            scores = {family: nrmse(centroid, row["I"]) for family, centroid in centroids.items()}
            prediction = min(scores, key=scores.get)
            predictions.append({"truth": truth_family, "prediction": prediction, "correct": prediction == truth_family, "scores": scores})
    accuracy = float(np.mean([row["correct"] for row in predictions])) if predictions else 0.0
    energy = {family: np.mean(np.abs(centroid), axis=(1, 2)) for family, centroid in centroids.items()}
    first = None
    for checkpoint in range(38):
        values = [float(energy[family][checkpoint]) for family in FAMILIES]
        if max(values) - min(values) > 0.10 * max(float(np.mean(values)), 1e-12):
            first = checkpoint
            break
    write_rows(out / "analysis/predictions.jsonl", predictions)
    np.save(out / "analysis/family_i_centroids.float16.npy", np.asarray([centroids[family] for family in FAMILIES], np.float16))
    close_memmap(states); del states; gc.collect()
    headline = {
        "status": "cross_construction_response_ecology_closed", "test_groups": len(predictions),
        "family_accuracy": accuracy, "chance": 1 / 3, "first_energy_differentiation_checkpoint": first,
        "descriptive_candidate": accuracy >= 2 / 3,
        "strict_interpretation": "Family decoding can be confounded by answer type and construction; it is not a neural ontology or causal code.",
    }
    close("C397", headline, {"predictions": len(predictions) > 0, "finite": previous.finite(headline)}, "C398_synthesis")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def c398() -> None:
    out = begin("C398", {
        "status": "independent_construction_synthesis_frozen",
        "gates": ["behavior", "conditional I", "old transfer", "output-sensitive scope", "response ecology", "causal", "new math"],
        "visual": "three family I centroids x five checkpoints x six roles x all 2560 coordinates",
        "cleanup": "checksum bulk C394 fields then remove",
    }, {"parent": final("C397")["all_checks_passed"]})
    centroids = np.load(OUTS["C397"] / "analysis/family_i_centroids.float16.npy")
    rows = []
    for family_i, family in enumerate(FAMILIES):
        for checkpoint, role_i in itertools.product((0, 12, 24, 36, 37), range(len(ROLES))):
            rows.append({"id": f"{family}:I:q{checkpoint}:{ROLES[role_i]}", "family": family, "operation": "I", "checkpoint": checkpoint, "role": ROLES[role_i], "values": np.asarray(centroids[family_i, checkpoint, role_i], np.float32).round(6).tolist()})
    visual = {
        "schema": "c398.independent_construction_lockbox.v1", "phase": 1932, "campaign": "C398", "model": "Qwen3-4B",
        "dimensions": list(range(2560)), "rows": rows,
        "claim_boundary": "Full-coordinate fresh-construction I centroids are descriptive observations, not causal semantic coordinates.",
    }
    save(ROOT / "frontend/public/vis_data/research_kernel/c398_independent_construction_lockbox.json", visual)
    gates = {
        "behavior": final("C393")["headline"]["field_eligible"],
        "conditional_i_any": bool(final("C395")["headline"]["conditional_passed"]),
        "old_transfer_any": bool(final("C395")["headline"]["old_transfer_passed"]),
        "output_sensitive_scope": final("C396")["headline"]["fresh_scope_order_passed"],
        "old_scope_transfer": final("C396")["headline"]["old_scope_order_transfer_passed"],
        "response_ecology": final("C397")["headline"]["descriptive_candidate"],
        "causal": False,
    }
    gates["new_math"] = gates["conditional_i_any"] and gates["old_transfer_any"] and gates["output_sensitive_scope"] and gates["causal"]
    cleanup = []
    for path in (OUTS["C394"] / "raw/role_states.float16.npy", OUTS["C394"] / "raw/full_fields_holdout.float16.npy"):
        if path.exists():
            array = np.load(path, mmap_mode="r"); shape = list(array.shape); close_memmap(array); del array; gc.collect()
            cleanup.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": hash_file(path), "shape": shape, "status": "checksum_committed_pending_removal"})
    save(out / "audit/cleanup.provisional.json", cleanup)
    for item in cleanup:
        path = ROOT / item["path"]; path.unlink(); item["status"] = "checksum_committed_and_removed"; item["removed"] = not path.exists()
    save(out / "audit/cleanup.json", cleanup)
    headline = {
        "status": "independent_construction_campaign_closed", "gates": gates,
        "candidate_results": final("C395")["headline"]["results"], "scope_result": final("C396")["headline"],
        "ecology_result": final("C397")["headline"], "visual_rows": len(rows),
        "cleanup_bytes": sum(item["bytes"] for item in cleanup), "new_math_gate_passed": gates["new_math"],
        "strict_interpretation": "Fresh construction results update only their registered candidates. No causal operator or new mathematics is established.",
    }
    checks = {
        "phases": all(final(f"C{value}")["all_checks_passed"] for value in range(391, 398)),
        "visual": len(rows) == 90 and all(len(row["values"]) == 2560 for row in rows),
        "cleanup": all(item["removed"] for item in cleanup), "finite": previous.finite(headline),
    }
    close("C398", headline, checks, "independent_audit_then_broaden_output_sensitive_language_families")


FUNCTIONS = {name: globals()[name.lower()] for name in PHASES}


def parse_range(value: str) -> list[str]:
    if "-" not in value:
        return [value.upper()]
    left, right = value.upper().split("-", 1)
    return [f"C{number}" for number in range(int(left[1:]), int(right[1:]) + 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="C391-C398")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        rows = material()
        checks = {"rows": len(rows) == 960, "balance": sum(row["gold_position"] == 0 for row in rows) == 480, "families": {row["family"] for row in rows} == set(FAMILIES)}
        print(json.dumps(checks, indent=2)); raise SystemExit(0 if all(checks.values()) else 1)
    for name in parse_range(args.run):
        FUNCTIONS[name]()


if __name__ == "__main__":
    main()
