#!/usr/bin/env python3
"""C415-C425 balanced dynamic-composition and graph-interface campaign.

Only embeddings and HiddenStates are observed. Attention, MLP activations,
weights, PCA, Top-K selection, and cosine gates are outside the contract.
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
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c425_dynamic_composition_field.json"
sys.path.insert(0, str(TESTS))

import phase1933_c399_c414_output_sensitive_language_campaign as parent
import phase1797_c263_c272_state_operator_common as family_base


PHASES = {
    f"C{campaign}": (1949 + campaign - 415, slug)
    for campaign, slug in (
        (415, "balanced_dynamic_campaign_contract"),
        (416, "balanced_four_factor_composition_material"),
        (417, "qwen_four_factor_behavior"),
        (418, "qwen_four_factor_full_coordinate_field"),
        (419, "mobius_interaction_unseen_construction_prediction"),
        (420, "dynamic_h000_single_sample_donor_prediction"),
        (421, "full_token_dynamic_hyperedge_observation"),
        (422, "binary_decomposed_graph_behavior"),
        (423, "qualified_graph_field_depth_prediction"),
        (424, "qualified_dynamic_state_writer"),
        (425, "campaign_synthesis_heatmap_cleanup_audit"),
    )
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

FAMILIES = ("attitude_event", "nested_attitude", "negation_scope")
CONSTRUCTIONS = ("dossier", "witness", "briefing")
AXES = ("outer", "attitude", "event", "object")
ROLES = parent.ROLES
UNITS = parent.UNITS[:4]
CHECKPOINTS = 38
DIM = 2560
FIELD_WIDTH = 192


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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
    save(
        out / "protocol/preregistration.json",
        {
            "phase": PHASES[name][0],
            "campaign": name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "producer_sha256": producer_hash(),
            **protocol,
        },
    )
    save(
        out / "audit/internal_contract_audit.json",
        {"checks": checks, "all_checks_passed": True},
    )
    return out


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    out = OUTS[name]
    if (out / "analysis/final.json").exists():
        return load(out / "analysis/final.json")
    save(out / "analysis/summary.json", headline)
    save(
        out / "audit/internal_analysis_audit.json",
        {"checks": checks, "all_checks_passed": all(checks.values())},
    )
    final_checks = {
        "contract": load(out / "audit/internal_contract_audit.json")[
            "all_checks_passed"
        ],
        "analysis": all(checks.values()),
        "producer_hash": load(out / "protocol/preregistration.json")[
            "producer_sha256"
        ]
        == producer_hash(),
    }
    result = {
        "phase": PHASES[name][0],
        "campaign": name,
        "status": "closed",
        "checks": final_checks,
        "all_checks_passed": all(final_checks.values()),
        "headline": headline,
        "next_authorization": authorization,
    }
    save(out / "analysis/final.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def close_memmap(value) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def nrmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, np.float32)
    truth = np.asarray(truth, np.float32)
    return float(
        np.linalg.norm(prediction - truth) / (np.linalg.norm(truth) + 1e-8)
    )


def wrap(construction: str, target: str, noise: str, question: str) -> str:
    if construction == "dossier":
        return (
            f"A dossier states: {target} It also states: {noise} "
            f"From the dossier alone, {question}"
        )
    if construction == "witness":
        return (
            f"A witness gave two statements. First: {target} Second: {noise} "
            f"Using only that testimony, {question}"
        )
    if construction == "briefing":
        return (
            f"During a briefing, one item was {target} "
            f"Another independent item was {noise} Decide from these items: {question}"
        )
    raise KeyError(construction)


def composition_case(
    family: str, construction: str, unit: int, bits: tuple[int, ...], axis: str, order: int
) -> dict:
    d, a, b, c = bits
    u = UNITS[unit]
    p, s, o, obj, other = u["p"], u["s"], u["o"], u["obj"], u["other"]
    attitude = ("likes", "dislikes")[a]
    event = ("ate", "did not eat")[b]
    item = (obj, other)[c]
    if family == "attitude_event":
        outer = ("reported", "denied")[d]
        target = (
            f"{o} {outer} that {p} {attitude} the event in which "
            f"{s} {event} the {item}."
        )
        outer_question = (
            f"Did {o} report that {p} {attitude} the event in which "
            f"{s} {event} the {item}?"
        )
    elif family == "nested_attitude":
        outer = ("believes", "doubts")[d]
        target = (
            f"{o} {outer} that {p} {attitude} the event in which "
            f"{s} {event} the {item}."
        )
        outer_question = (
            f"Does {o} believe that {p} {attitude} the event in which "
            f"{s} {event} the {item}?"
        )
    elif family == "negation_scope":
        outer = ("affirmed", "denied")[d]
        target = (
            f"{o} {outer} that {p} {attitude} the event in which "
            f"{s} {event} the {item}."
        )
        outer_question = (
            f"Did {o} affirm that {p} {attitude} the event in which "
            f"{s} {event} the {item}?"
        )
    else:
        raise KeyError(family)
    questions = {
        "outer": outer_question,
        "attitude": (
            f"According to the statement, does {p} like the event in which "
            f"{s} {event} the {item}?"
        ),
        "event": (
            f"According to the statement, did {s} eat the {item} in the event "
            f"that {p} {attitude}?"
        ),
        "object": (
            f"According to the statement, was the event about the {obj} "
            f"while {p} {attitude} it?"
        ),
    }
    axis_i = AXES.index(axis)
    truth = bits[axis_i] == 0
    correct, wrong = ("Yes", "No") if truth else ("No", "Yes")
    noise = (
        f"Separately, {s} catalogued the {other} for {p} "
        f"while {o} reviewed the inventory."
    )
    core = wrap(construction, target, noise, questions[axis])
    choices, gold = family_base.options(correct, wrong, order)
    relation_values = {
        "outer": outer,
        "attitude": attitude,
        "event": event,
        "object": item,
    }
    query_value = obj if axis == "object" else item
    cell = "".join(str(v) for v in bits)
    return {
        "case_id": (
            f"c416-{family}-{construction}-u{unit}-{cell}-{axis}-{order:+d}"
        ),
        "panel": "balanced_four_factor_composition",
        "family": family,
        "surface": construction,
        "construction": construction,
        "unit": unit,
        "cell": cell,
        "mask": sum(bit << i for i, bit in enumerate(bits)),
        "query_axis": axis,
        "order": order,
        "partition": (
            "discovery" if unit < 2 else "confirmation" if unit == 2 else "lockbox"
        ),
        "gold_position": gold,
        "correct_answer": correct,
        "wrong_answer": wrong,
        "prompt_core": core,
        "prompt": f"{core} {choices}. Reply with only A or B.",
        "free_prompt": f"{core} Answer with only Yes or No.",
        "role_values": {
            "primary": p,
            "secondary": s,
            "relation": relation_values[axis],
            "context": item,
            "query": query_value,
        },
        "semantic_graph": {
            "family": family,
            "outer": d,
            "attitude": a,
            "event": b,
            "object": c,
            "query_axis": axis,
            "truth": truth,
        },
    }


def composition_material() -> list[dict]:
    return [
        composition_case(family, construction, unit, bits, axis, order)
        for family, construction, unit, bits, axis, order in itertools.product(
            FAMILIES,
            CONSTRUCTIONS,
            range(len(UNITS)),
            itertools.product((0, 1), repeat=4),
            AXES,
            (1, -1),
        )
    ]


def composition_lookup() -> tuple[list[dict], dict[str, dict]]:
    rows = read_rows(OUTS["C416"] / "material/cases.jsonl")
    return rows, {row["case_id"]: row for row in rows}


def c415() -> None:
    audit = load(parent.OUTS["C414"] / "audit/independent_audit.json")
    out = begin(
        "C415",
        {
            "status": "balanced_dynamic_campaign_contract_frozen",
            "parent": "C414 independent audit 24/24",
            "positive_base": "C407 pair residual improves unseen-lexicon prediction",
            "retired": ["q0 nearest H00", "unconditional cross-construction family mean"],
            "arms": [
                "balanced four-factor composition",
                "dynamic full-H000 donor",
                "full-token hyperedge observation",
                "binary decomposed graph interface",
                "qualified state writer",
            ],
            "route_policy": "predeclared missingness closes only its route",
        },
        {
            "parent": audit["all_checks_passed"],
            "phase_continuity": PHASES["C415"][0] == 1949,
        },
    )
    headline = {
        "status": "balanced_dynamic_contract_closed",
        "retained": [
            "second-order interaction residual as a predictive candidate",
            "full-coordinate distributed field",
            "language family as external experimental index",
        ],
        "corrected": [
            "C407 semantic labels were imbalanced across the eight corners",
            "relative-depth similarity is not state translation",
            "graph unknown interface remains unqualified",
        ],
        "strict_interpretation": (
            "The campaign asks whether balanced labels, unseen construction, and a "
            "full baseline trajectory preserve the local pair-interaction result."
        ),
    }
    close(
        "C415",
        headline,
        {"audit": audit["passed"] == audit["total"], "no_recycled_gate": True},
        "C416_material",
    )


def c416() -> None:
    out = begin(
        "C416",
        {
            "status": "balanced_four_factor_material_frozen",
            "design": (
                "3 families x 3 constructions x 4 lexical units x 16 corners x "
                "4 query axes x 2 answer orders"
            ),
            "factor_truth": "each query reads exactly one factor; Yes/No balanced per axis",
            "human_naturalness_review": False,
        },
        {"parent": final("C415")["all_checks_passed"]},
    )
    rows = composition_material()
    write_rows(out / "material/cases.jsonl", rows)
    zero = {
        "always_first": sum(row["gold_position"] == 0 for row in rows) / len(rows),
        "always_second": sum(row["gold_position"] == 1 for row in rows) / len(rows),
        "semantic_always_yes": sum(
            row["correct_answer"] == "Yes" for row in rows
        )
        / len(rows),
        "semantic_always_no": sum(
            row["correct_answer"] == "No" for row in rows
        )
        / len(rows),
    }
    per_axis_balance = {
        axis: sum(
            row["correct_answer"] == "Yes"
            for row in rows
            if row["query_axis"] == axis
        )
        / sum(row["query_axis"] == axis for row in rows)
        for axis in AXES
    }
    roles = all(
        all(str(value) in row["prompt_core"] for value in row["role_values"].values())
        for row in rows
    )
    headline = {
        "status": "balanced_four_factor_material_closed",
        "rows": len(rows),
        "partition_counts": {
            part: sum(row["partition"] == part for row in rows)
            for part in ("discovery", "confirmation", "lockbox")
        },
        "zero_model_accuracies": zero,
        "per_axis_yes_frequency": per_axis_balance,
        "role_occurrence": roles,
        "material_eligible": max(abs(value - 0.5) for value in zero.values())
        < 1e-12
        and max(abs(value - 0.5) for value in per_axis_balance.values()) < 1e-12
        and roles,
        "human_naturalness_review": False,
        "strict_interpretation": (
            "Semantic label balance repairs C407's dominant-No shortcut but does not "
            "certify natural-language representativeness."
        ),
    }
    close(
        "C416",
        headline,
        {
            "rows": len(rows) == 4608,
            "zero": max(abs(v - 0.5) for v in zero.values()) < 1e-12,
            "axis": max(abs(v - 0.5) for v in per_axis_balance.values()) < 1e-12,
            "roles": roles,
        },
        "C417_behavior",
    )


def c417() -> None:
    out = begin(
        "C417",
        {
            "status": "qwen_four_factor_behavior_frozen",
            "model": "Qwen3-4B CUDA BF16",
            "gates": {
                "heldout": 0.80,
                "family": 0.70,
                "query_axis": 0.70,
                "construction": 0.70,
            },
            "hidden_state_policy": "none",
        },
        {
            "parent": final("C416")["all_checks_passed"],
            "material": final("C416")["headline"]["material_eligible"],
            "cuda": torch.cuda.is_available(),
        },
    )
    rows, _ = composition_lookup()
    tokenizer = parent.fresh.tokenizer_qwen()
    compiled = family_base.compile_qwen(tokenizer, rows)
    write_rows(out / "compiled/qwen3.jsonl", compiled)
    metrics = parent.previous.qwen_behavior(rows, compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl")
    held = [row for row in behavior if row["partition"] != "discovery"]
    material = rows_by_id(rows)
    by_family = {
        family: float(np.mean([r["correct"] for r in held if r["family"] == family]))
        for family in FAMILIES
    }
    by_axis = {
        axis: float(
            np.mean(
                [
                    r["correct"]
                    for r in held
                    if material[r["case_id"]]["query_axis"] == axis
                ]
            )
        )
        for axis in AXES
    }
    by_construction = {
        construction: float(
            np.mean([r["correct"] for r in held if r["surface"] == construction])
        )
        for construction in CONSTRUCTIONS
    }
    heldout = float(np.mean([r["correct"] for r in held]))
    eligible = (
        heldout >= 0.80
        and min(by_family.values()) >= 0.70
        and min(by_axis.values()) >= 0.70
        and min(by_construction.values()) >= 0.70
    )
    headline = {
        "status": "qwen_four_factor_behavior_closed",
        **metrics,
        "heldout_accuracy": heldout,
        "family_accuracy": by_family,
        "query_axis_accuracy": by_axis,
        "construction_accuracy": by_construction,
        "field_eligible": eligible,
        "strict_interpretation": (
            "Balanced factor readout qualifies the internal field only if every "
            "registered family, axis, and construction clears its gate."
        ),
    }
    close(
        "C417",
        headline,
        {
            "rows": len(behavior) == len(rows),
            "finite": finite(headline),
            "no_hidden": not (out / "raw/role_states.float16.npy").exists(),
        },
        "C418_field",
    )


def rows_by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["case_id"]: row for row in rows}


def c418() -> None:
    eligible = final("C417")["headline"]["field_eligible"]
    out = begin(
        "C418",
        {
            "status": "qwen_four_factor_full_coordinate_field_frozen",
            "run_condition": "C417 behavior eligibility",
            "archive": "38 checkpoints x six roles x all 2560 coordinates",
            "full_token_subset": (
                "briefing lockbox unit3 order+1, cell0000 plus four singleton cells"
            ),
            "no_pca_topk": True,
        },
        {"parent": final("C417")["all_checks_passed"]},
    )
    if not eligible:
        headline = {
            "status": "four_factor_field_not_run_behavior_ineligible",
            "field_ran": False,
            "strict_interpretation": "The balanced four-factor internal mechanism remains untested.",
        }
        close(
            "C418",
            headline,
            {"no_field": not (out / "raw/role_states.float16.npy").exists()},
            "C419_mobius",
        )
        return
    rows, _ = composition_lookup()
    compiled = read_rows(OUTS["C417"] / "compiled/qwen3.jsonl")
    singleton_cells = {"0000", "1000", "0100", "0010", "0001"}
    selector = lambda row: (
        row["construction"] == "briefing"
        and row["unit"] == 3
        and row["order"] == 1
        and row["cell"] in singleton_cells
    )
    metrics = parent.previous.common.batch_capture_qwen(
        rows,
        compiled,
        out,
        full_selector=selector,
        batch_size=8,
        field_width=FIELD_WIDTH,
    )
    headline = {
        "status": "qwen_four_factor_full_field_closed",
        **metrics,
        "field_ran": True,
        "strict_interpretation": (
            "The archive is an observational full-coordinate field, not a "
            "factor-neuron dictionary."
        ),
    }
    close(
        "C418",
        headline,
        {
            "shape": metrics["role_shape"][1:] == [38, 6, 2560],
            "full": metrics["full_shape"][-1] == 2560,
            "finite": finite(headline),
        },
        "C419_mobius",
    )


def complete_groups(
    states: np.ndarray, index: list[dict], material: dict[str, dict], family: str, axis: str
) -> list[dict]:
    keyed = {}
    for row in index:
        case = material[row["case_id"]]
        if (
            row["correct"]
            and case["family"] == family
            and case["query_axis"] == axis
        ):
            keyed[
                (
                    case["construction"],
                    case["unit"],
                    case["order"],
                    case["mask"],
                )
            ] = row
    groups = []
    for construction, unit, order in itertools.product(
        CONSTRUCTIONS, range(len(UNITS)), (1, -1)
    ):
        keys = [(construction, unit, order, mask) for mask in range(16)]
        if not all(key in keyed for key in keys):
            continue
        h = {
            mask: np.asarray(states[keyed[(construction, unit, order, mask)]["hidden_index"]], np.float32)
            for mask in range(16)
        }
        effects = {}
        for mask in range(1, 16):
            total = np.zeros_like(h[0])
            subset = mask
            while True:
                sign = -1.0 if ((mask.bit_count() - subset.bit_count()) % 2) else 1.0
                total += sign * h[subset]
                if subset == 0:
                    break
                subset = (subset - 1) & mask
            effects[mask] = total
        groups.append(
            {
                "family": family,
                "axis": axis,
                "construction": construction,
                "unit": unit,
                "order": order,
                "H0": h[0],
                "H15": h[15],
                "effects": effects,
            }
        )
    return groups


def predict_from_effects(
    base: np.ndarray, effects: dict[int, np.ndarray], target_mask: int, max_order: int
) -> np.ndarray:
    prediction = np.asarray(base, np.float32).copy()
    for mask, value in effects.items():
        if mask & target_mask == mask and mask.bit_count() <= max_order:
            prediction += value
    return prediction


def c419() -> None:
    field_ran = final("C418")["headline"]["field_ran"]
    out = begin(
        "C419",
        {
            "status": "mobius_unseen_construction_prediction_frozen",
            "run_condition": "C418 field",
            "train": "dossier+witness discovery units0-1",
            "test": "briefing confirmation+lockbox units2-3",
            "target": "H1111 from its own H0000",
            "models": ["first order", "through pair", "through triple", "through quadruple"],
            "controls": ["wrong family pair", "coordinate roll", "zero response"],
            "pair_pass": "gain over additive >0.01 and beats all controls",
        },
        {"parent": final("C418")["all_checks_passed"]},
    )
    if not field_ran:
        headline = {
            "status": "mobius_prediction_not_run_no_field",
            "prediction_ran": False,
            "candidate_families": [],
            "strict_interpretation": "No internal composition conclusion is available.",
        }
        close("C419", headline, {"route_accounted": True}, "C420_dynamic_h000")
        return
    states = np.load(OUTS["C418"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C418"] / "raw/hidden_index.jsonl")
    _, material = composition_lookup()
    groups = {
        (family, axis): complete_groups(states, index, material, family, axis)
        for family, axis in itertools.product(FAMILIES, AXES)
    }
    means_path = out / "analysis/mobius_source_means.float16.npy"
    means_arr = np.lib.format.open_memmap(
        means_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(FAMILIES), len(AXES), 16, 38, 6, 2560),
    )
    means: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    for fi, family in enumerate(FAMILIES):
        for ai, axis in enumerate(AXES):
            train = [
                group
                for group in groups[(family, axis)]
                if group["construction"] in ("dossier", "witness") and group["unit"] < 2
            ]
            effect_mean = {
                mask: np.mean([group["effects"][mask] for group in train], axis=0)
                for mask in range(1, 16)
            } if train else {}
            means[(family, axis)] = effect_mean
            for mask, value in effect_mean.items():
                means_arr[fi, ai, mask] = value.astype(np.float16)
            means_arr.flush()
    results = []
    for fi, family in enumerate(FAMILIES):
        wrong_family = FAMILIES[(fi + 1) % len(FAMILIES)]
        for axis in AXES:
            train = [
                group
                for group in groups[(family, axis)]
                if group["construction"] in ("dossier", "witness") and group["unit"] < 2
            ]
            test = [
                group
                for group in groups[(family, axis)]
                if group["construction"] == "briefing" and group["unit"] >= 2
            ]
            effect_mean = means[(family, axis)]
            wrong_mean = means[(wrong_family, axis)]
            if not train or not test or not effect_mean or not wrong_mean:
                results.append(
                    {
                        "family": family,
                        "query_axis": axis,
                        "prediction_ran": False,
                        "train_groups": len(train),
                        "test_groups": len(test),
                        "reason": "incomplete_behavior_correct_source_or_target_groups",
                    }
                )
                continue
            scores = {name: [] for name in ("order1", "order2", "order3", "order4", "wrong_family", "coordinate_roll", "zero")}
            for group in test:
                truth = group["H15"]
                predictions = {
                    f"order{order}": predict_from_effects(group["H0"], effect_mean, 15, order)
                    for order in range(1, 5)
                }
                wrong_prediction = predict_from_effects(group["H0"], wrong_mean, 15, 2)
                pair_response = predictions["order2"] - group["H0"]
                control_roll = group["H0"] + np.roll(pair_response, 1, axis=-1)
                for name, prediction in predictions.items():
                    scores[name].append(nrmse(prediction, truth))
                scores["wrong_family"].append(nrmse(wrong_prediction, truth))
                scores["coordinate_roll"].append(nrmse(control_roll, truth))
                scores["zero"].append(nrmse(group["H0"], truth))
            values = {name: float(np.mean(v)) for name, v in scores.items()}
            pair_gain = values["order1"] - values["order2"]
            passed = (
                pair_gain > 0.01
                and values["order2"]
                < min(values["wrong_family"], values["coordinate_roll"], values["zero"])
            )
            results.append(
                {
                    "family": family,
                    "query_axis": axis,
                    "prediction_ran": True,
                    "train_groups": len(train),
                    "test_groups": len(test),
                    "nrmse": values,
                    "pair_gain": pair_gain,
                    "pair_passed": passed,
                }
            )
    write_rows(out / "analysis/mobius_predictions.jsonl", results)
    pass_counts = {
        family: sum(
            row.get("pair_passed", False)
            for row in results
            if row["family"] == family
        )
        for family in FAMILIES
    }
    candidates = [family for family, count in pass_counts.items() if count >= 2]
    headline = {
        "status": "mobius_unseen_construction_prediction_closed",
        "prediction_ran": True,
        "cells": len(results),
        "ran_cells": sum(row["prediction_ran"] for row in results),
        "pair_pass_counts": pass_counts,
        "candidate_families": candidates,
        "mean_pair_gain": float(
            np.mean([row["pair_gain"] for row in results if row["prediction_ran"]])
        )
        if any(row["prediction_ran"] for row in results)
        else None,
        "strict_interpretation": (
            "A passed pair term is an unseen-construction response predictor, not "
            "a recursive language algebra or causal hyperedge."
        ),
    }
    close_memmap(means_arr)
    close_memmap(states)
    close(
        "C419",
        headline,
        {"route_accounted": len(results) == len(FAMILIES) * len(AXES), "finite": finite(headline)},
        "C420_dynamic_h000",
    )


def mean_square_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean((np.asarray(left, np.float32) - np.asarray(right, np.float32)) ** 2))


def c420() -> None:
    out = begin(
        "C420",
        {
            "status": "dynamic_h000_single_sample_donor_frozen",
            "source": "complete 38x6x2560 H0000 trajectory only",
            "target": "H1111-H0000 response",
            "controls": ["q0-only nearest donor", "source mean", "coordinate roll"],
            "pass": "dynamic donor beats q0 and source mean by >0.005 and beats roll",
        },
        {"parent": final("C419")["all_checks_passed"]},
    )
    if not final("C419")["headline"]["prediction_ran"]:
        headline = {
            "status": "dynamic_h000_not_run_no_field",
            "prediction_ran": False,
            "candidate_families": [],
            "strict_interpretation": "The dynamic baseline hypothesis remains untested.",
        }
        close("C420", headline, {"route_accounted": True}, "C421_full_token")
        return
    states = np.load(OUTS["C418"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C418"] / "raw/hidden_index.jsonl")
    _, material = composition_lookup()
    results = []
    for family, axis in itertools.product(FAMILIES, AXES):
        groups = complete_groups(states, index, material, family, axis)
        train = [
            group
            for group in groups
            if group["construction"] in ("dossier", "witness") and group["unit"] < 2
        ]
        test = [
            group
            for group in groups
            if group["construction"] == "briefing" and group["unit"] >= 2
        ]
        if not train or not test:
            results.append(
                {
                    "family": family,
                    "query_axis": axis,
                    "prediction_ran": False,
                    "train_groups": len(train),
                    "test_groups": len(test),
                    "reason": "incomplete_groups",
                }
            )
            continue
        mean_response = np.mean([group["H15"] - group["H0"] for group in train], axis=0)
        scores = {name: [] for name in ("dynamic", "q0", "mean", "coordinate_roll")}
        for group in test:
            dynamic_i = int(
                np.argmin(
                    [
                        mean_square_distance(group["H0"], donor["H0"])
                        for donor in train
                    ]
                )
            )
            q0_i = int(
                np.argmin(
                    [
                        mean_square_distance(group["H0"][0], donor["H0"][0])
                        for donor in train
                    ]
                )
            )
            dynamic_response = train[dynamic_i]["H15"] - train[dynamic_i]["H0"]
            q0_response = train[q0_i]["H15"] - train[q0_i]["H0"]
            truth = group["H15"]
            predictions = {
                "dynamic": group["H0"] + dynamic_response,
                "q0": group["H0"] + q0_response,
                "mean": group["H0"] + mean_response,
                "coordinate_roll": group["H0"] + np.roll(dynamic_response, 1, axis=-1),
            }
            for name, prediction in predictions.items():
                scores[name].append(nrmse(prediction, truth))
        values = {name: float(np.mean(v)) for name, v in scores.items()}
        gain_mean = values["mean"] - values["dynamic"]
        gain_q0 = values["q0"] - values["dynamic"]
        passed = (
            gain_mean > 0.005
            and gain_q0 > 0.005
            and values["dynamic"] < values["coordinate_roll"]
        )
        results.append(
            {
                "family": family,
                "query_axis": axis,
                "prediction_ran": True,
                "train_groups": len(train),
                "test_groups": len(test),
                "nrmse": values,
                "gain_over_mean": gain_mean,
                "gain_over_q0": gain_q0,
                "passed": passed,
            }
        )
    write_rows(out / "analysis/dynamic_h000_predictions.jsonl", results)
    pass_counts = {
        family: sum(
            row.get("passed", False)
            for row in results
            if row["family"] == family
        )
        for family in FAMILIES
    }
    candidates = [family for family, count in pass_counts.items() if count >= 2]
    headline = {
        "status": "dynamic_h000_single_sample_prediction_closed",
        "prediction_ran": True,
        "cells": len(results),
        "ran_cells": sum(row["prediction_ran"] for row in results),
        "pass_counts": pass_counts,
        "candidate_families": candidates,
        "strict_interpretation": (
            "Nearest full-baseline-trajectory success would be an effective donor "
            "rule, not a unique state variable or causal circuit."
        ),
    }
    close_memmap(states)
    close(
        "C420",
        headline,
        {"route_accounted": len(results) == len(FAMILIES) * len(AXES), "finite": finite(headline)},
        "C421_full_token",
    )


def c421() -> None:
    out = begin(
        "C421",
        {
            "status": "full_token_dynamic_hyperedge_observation_frozen",
            "source": "C418 full-token briefing lockbox subset",
            "contrasts": "four singleton factor cells minus cell0000",
            "archive": "38 checkpoints x 192 positions x all 2560 coordinates",
            "claim": "effective response only",
        },
        {"parent": final("C420")["all_checks_passed"]},
    )
    if not final("C418")["headline"]["field_ran"]:
        headline = {
            "status": "full_token_hyperedge_not_run_no_field",
            "observation_ran": False,
            "pairs": 0,
            "strict_interpretation": "No token-level internal conclusion is available.",
        }
        close("C421", headline, {"route_accounted": True}, "C422_graph_behavior")
        return
    fields = np.load(OUTS["C418"] / "raw/full_fields_holdout.float16.npy", mmap_mode="r")
    row_map = load(OUTS["C418"] / "raw/full_field_row_map.json")["source_indices"]
    index = read_rows(OUTS["C418"] / "raw/hidden_index.jsonl")
    _, material = composition_lookup()
    selected = [
        (local, material[index[source]["case_id"]])
        for local, source in enumerate(row_map)
    ]
    keyed = {
        (case["family"], case["query_axis"], case["cell"]): local
        for local, case in selected
    }
    singleton = {
        "outer": "1000",
        "attitude": "0100",
        "event": "0010",
        "object": "0001",
    }
    pairs = []
    for family, query_axis, factor in itertools.product(FAMILIES, AXES, AXES):
        base = (family, query_axis, "0000")
        target = (family, query_axis, singleton[factor])
        if base in keyed and target in keyed:
            pairs.append((family, query_axis, factor, keyed[base], keyed[target]))
    delta_path = out / "analysis/full_token_singleton_delta.float16.npy"
    deltas = np.lib.format.open_memmap(
        delta_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(pairs), 38, FIELD_WIDTH, 2560),
    )
    energy = np.zeros((len(pairs), 38, FIELD_WIDTH), np.float32)
    meta = []
    for i, (family, query_axis, factor, base, target) in enumerate(pairs):
        delta = np.asarray(fields[target], np.float32) - np.asarray(fields[base], np.float32)
        deltas[i] = delta.astype(np.float16)
        energy[i] = np.mean(np.abs(delta), axis=-1)
        meta.append(
            {
                "delta_index": i,
                "family": family,
                "query_axis": query_axis,
                "factor": factor,
            }
        )
        deltas.flush()
    np.save(out / "analysis/full_token_singleton_energy.float32.npy", energy)
    write_rows(out / "analysis/full_token_singleton_index.jsonl", meta)
    headline = {
        "status": "full_token_dynamic_hyperedge_observation_closed",
        "observation_ran": True,
        "pairs": len(pairs),
        "shape": list(deltas.shape),
        "strict_interpretation": (
            "The tensor is a complete effective response field; without intervention "
            "it is not a unique directed hyperedge."
        ),
    }
    close_memmap(deltas)
    close_memmap(fields)
    close(
        "C421",
        headline,
        {"pairs": len(pairs) == 48, "shape": headline["shape"][-1] == 2560},
        "C422_graph_behavior",
    )


GRAPH_UNITS = parent.GRAPH_UNITS[:8]
GRAPH_MODES = parent.GRAPH_MODES
GRAPH_CHANNELS = ("entailment", "contradiction")
GRAPH_POLARITIES = ("positive", "negative")


def graph_core(unit: dict, depth: int, construction: str, mode: str) -> tuple[str, str]:
    nodes = [unit["root"], unit["m1"], unit["m2"], unit["m3"], unit["final"]]
    path = [nodes[0], *nodes[1:depth], unit["final"]]
    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    label = (
        "entailed"
        if mode in ("entailed", "multipath", "shortcut")
        else "contradicted"
        if mode == "contradicted"
        else "unknown"
    )
    if mode == "contradicted":
        facts = [
            f'The registry explicitly says that "{unit["root"]}" is not a kind of "{unit["final"]}".'
        ]
    else:
        if mode == "reversed":
            edges = [(right, left) for left, right in edges]
        if mode == "broken" and edges:
            edges[len(edges) // 2] = (edges[len(edges) // 2][0], unit["wrong"])
        facts = [f'"{left}" is a kind of "{right}".' for left, right in edges]
        if mode == "unknown":
            facts = [
                f'"{unit["root"]}" is a kind of "{unit["m1"]}".',
                f'"{unit["wrong"]}" is a kind of "{unit["final"]}".',
            ]
        if mode == "multipath":
            facts.append(
                f'A second register directly says that "{unit["root"]}" is a kind of "{unit["final"]}".'
            )
        if mode == "shortcut":
            facts.append(
                f'A direct shortcut says that "{unit["root"]}" is a kind of "{unit["final"]}".'
            )
    rules = (
        "Use only these rules: kind-of links are transitive; an explicit "
        "not-kind-of statement contradicts the query; missing links neither "
        "entail nor explicitly contradict it."
    )
    body = " ".join(facts)
    prefix = "Registry facts:" if construction == "registry" else "A briefing reports:"
    return f"{prefix} {body} {rules}", label


def graph_material() -> list[dict]:
    rows = []
    for unit_i, depth, construction, mode, channel, polarity, order in itertools.product(
        range(len(GRAPH_UNITS)),
        range(1, 5),
        ("registry", "briefing"),
        GRAPH_MODES,
        GRAPH_CHANNELS,
        GRAPH_POLARITIES,
        (1, -1),
    ):
        unit = GRAPH_UNITS[unit_i]
        core, label = graph_core(unit, depth, construction, mode)
        positive_truth = label == ("entailed" if channel == "entailment" else "contradicted")
        truth = positive_truth if polarity == "positive" else not positive_truth
        if channel == "entailment":
            question = (
                f'Do these facts entail that "{unit["root"]}" is a kind of "{unit["final"]}"?'
                if polarity == "positive"
                else f'Do these facts fail to entail that "{unit["root"]}" is a kind of "{unit["final"]}"?'
            )
        else:
            question = (
                f'Do these facts explicitly contradict that "{unit["root"]}" is a kind of "{unit["final"]}"?'
                if polarity == "positive"
                else f'Do these facts fail to explicitly contradict that "{unit["root"]}" is a kind of "{unit["final"]}"?'
            )
        correct, wrong = ("Yes", "No") if truth else ("No", "Yes")
        choices, gold = family_base.options(correct, wrong, order)
        prompt_core = f"{core} {question}"
        graph_id = f"u{unit_i}-d{depth}-{construction}-{mode}"
        rows.append(
            {
                "case_id": f"c422-{graph_id}-{channel}-{polarity}-{order:+d}",
                "graph_id": graph_id,
                "panel": "binary_decomposed_graph",
                "family": "type_graph",
                "surface": construction,
                "construction": construction,
                "unit": unit_i,
                "depth": depth,
                "mode": mode,
                "channel": channel,
                "polarity": polarity,
                "order": order,
                "partition": (
                    "discovery"
                    if unit_i < 4
                    else "confirmation"
                    if unit_i < 6
                    else "lockbox"
                ),
                "gold_position": gold,
                "correct_answer": correct,
                "wrong_answer": wrong,
                "prompt_core": prompt_core,
                "prompt": f"{prompt_core} {choices}. Reply with only A or B.",
                "free_prompt": f"{prompt_core} Answer with only Yes or No.",
                "role_values": {
                    "primary": unit["root"],
                    "secondary": unit["final"],
                    "relation": "kind of",
                    "context": unit["final"],
                    "query": unit["root"],
                },
                "semantic_graph": {
                    "label": label,
                    "mode": mode,
                    "channel": channel,
                    "polarity": polarity,
                    "truth": truth,
                },
            }
        )
    return rows


def c422() -> None:
    out = begin(
        "C422",
        {
            "status": "binary_decomposed_graph_behavior_frozen",
            "design": (
                "8 graphs x depths1-4 x 2 constructions x 7 modes x "
                "2 channels x 2 polarities x 2 answer orders"
            ),
            "interface": "unknown is represented by not-entailed AND not-contradicted",
            "gates": {
                "heldout": 0.80,
                "mode": 0.65,
                "channel": 0.75,
                "polarity": 0.75,
                "unknown_joint": 0.65,
            },
            "hidden_state_policy": "none",
        },
        {"parent": final("C421")["all_checks_passed"], "cuda": torch.cuda.is_available()},
    )
    rows = graph_material()
    write_rows(out / "material/cases.jsonl", rows)
    tokenizer = parent.fresh.tokenizer_qwen()
    compiled = family_base.compile_qwen(tokenizer, rows)
    write_rows(out / "compiled/qwen3.jsonl", compiled)
    metrics = parent.previous.qwen_behavior(rows, compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl")
    material = rows_by_id(rows)
    held = [row for row in behavior if row["partition"] != "discovery"]
    by_mode = {
        mode: float(
            np.mean([r["correct"] for r in held if material[r["case_id"]]["mode"] == mode])
        )
        for mode in GRAPH_MODES
    }
    by_channel = {
        channel: float(
            np.mean(
                [
                    r["correct"]
                    for r in held
                    if material[r["case_id"]]["channel"] == channel
                ]
            )
        )
        for channel in GRAPH_CHANNELS
    }
    by_polarity = {
        polarity: float(
            np.mean(
                [
                    r["correct"]
                    for r in held
                    if material[r["case_id"]]["polarity"] == polarity
                ]
            )
        )
        for polarity in GRAPH_POLARITIES
    }
    positive = {
        (material[r["case_id"]]["graph_id"], material[r["case_id"]]["channel"]): r
        for r in held
        if material[r["case_id"]]["polarity"] == "positive"
        and material[r["case_id"]]["order"] == 1
        and material[r["case_id"]]["mode"] == "unknown"
    }
    unknown_graphs = sorted({key[0] for key in positive})
    unknown_joint = float(
        np.mean(
            [
                positive[(graph_id, "entailment")]["correct"]
                and positive[(graph_id, "contradiction")]["correct"]
                for graph_id in unknown_graphs
                if (graph_id, "entailment") in positive
                and (graph_id, "contradiction") in positive
            ]
        )
    )
    heldout = float(np.mean([row["correct"] for row in held]))
    eligible = (
        heldout >= 0.80
        and min(by_mode.values()) >= 0.65
        and min(by_channel.values()) >= 0.75
        and min(by_polarity.values()) >= 0.75
        and unknown_joint >= 0.65
    )
    headline = {
        "status": "binary_decomposed_graph_behavior_closed",
        **metrics,
        "heldout_accuracy": heldout,
        "mode_accuracy": by_mode,
        "channel_accuracy": by_channel,
        "polarity_accuracy": by_polarity,
        "unknown_joint_accuracy": unknown_joint,
        "graph_field_eligible": eligible,
        "strict_interpretation": (
            "This interface tests separate entailment and contradiction judgments; "
            "failure would not negate graph reasoning in general."
        ),
    }
    close(
        "C422",
        headline,
        {
            "rows": len(rows) == 3584,
            "semantic_balance": sum(r["correct_answer"] == "Yes" for r in rows)
            == len(rows) // 2,
            "finite": finite(headline),
            "no_hidden": not (out / "raw/role_states.float16.npy").exists(),
        },
        "C423_graph_field",
    )


def c423() -> None:
    eligible = final("C422")["headline"]["graph_field_eligible"]
    out = begin(
        "C423",
        {
            "status": "qualified_graph_field_depth_prediction_frozen",
            "run_condition": "C422 graph behavior eligibility",
            "capture": "positive polarity, answer order +1 only",
            "forecast": "depth1-2 discovery response predicts depth3-4 heldout",
        },
        {"parent": final("C422")["all_checks_passed"]},
    )
    if not eligible:
        headline = {
            "status": "graph_field_not_run_behavior_ineligible",
            "field_ran": False,
            "forecast_ran": False,
            "strict_interpretation": "The repaired-interface graph mechanism remains untested.",
        }
        close("C423", headline, {"route_accounted": True}, "C424_writer")
        return
    rows = read_rows(OUTS["C422"] / "material/cases.jsonl")
    compiled = read_rows(OUTS["C422"] / "compiled/qwen3.jsonl")
    selected = [
        (row, comp)
        for row, comp in zip(rows, compiled)
        if row["polarity"] == "positive" and row["order"] == 1
    ]
    selected_rows = [row for row, _ in selected]
    selected_compiled = [comp for _, comp in selected]
    metrics = parent.previous.common.batch_capture_qwen(
        selected_rows,
        selected_compiled,
        out,
        full_selector=None,
        batch_size=8,
        field_width=FIELD_WIDTH,
    )
    states = np.load(out / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(out / "raw/hidden_index.jsonl")
    material = rows_by_id(selected_rows)
    keyed = {
        (
            material[row["case_id"]]["unit"],
            material[row["case_id"]]["depth"],
            material[row["case_id"]]["construction"],
            material[row["case_id"]]["mode"],
            material[row["case_id"]]["channel"],
        ): row
        for row in index
        if row["correct"]
    }
    forecasts = []
    for channel, positive_mode in (
        ("entailment", "entailed"),
        ("contradiction", "contradicted"),
    ):
        train, test = [], []
        for unit, depth, construction in itertools.product(
            range(len(GRAPH_UNITS)), range(1, 5), ("registry", "briefing")
        ):
            kp = (unit, depth, construction, positive_mode, channel)
            ku = (unit, depth, construction, "unknown", channel)
            if kp not in keyed or ku not in keyed:
                continue
            response = np.asarray(
                states[keyed[kp]["hidden_index"]], np.float32
            ) - np.asarray(states[keyed[ku]["hidden_index"]], np.float32)
            if depth <= 2 and unit < 4:
                train.append(response)
            elif depth >= 3 and unit >= 4:
                test.append(response)
        if train and test:
            prediction = np.mean(train, axis=0)
            score = float(np.mean([nrmse(prediction, truth) for truth in test]))
            forecasts.append(
                {
                    "channel": channel,
                    "prediction_ran": True,
                    "train_groups": len(train),
                    "test_groups": len(test),
                    "nrmse": score,
                    "zero_nrmse": 1.0,
                    "passed": score < 1.0,
                }
            )
        else:
            forecasts.append(
                {
                    "channel": channel,
                    "prediction_ran": False,
                    "train_groups": len(train),
                    "test_groups": len(test),
                    "reason": "incomplete_behavior_correct_pairs",
                }
            )
    write_rows(out / "analysis/depth_forecasts.jsonl", forecasts)
    headline = {
        "status": "qualified_graph_field_depth_prediction_closed",
        **metrics,
        "field_ran": True,
        "forecasts": forecasts,
        "strict_interpretation": (
            "A passed depth response forecast is not proof of a recursive graph algorithm."
        ),
    }
    close_memmap(states)
    close(
        "C423",
        headline,
        {"shape": metrics["role_shape"][-1] == 2560, "route_accounted": len(forecasts) == 2, "finite": finite(headline)},
        "C424_writer",
    )


@torch.inference_mode()
def writer_test(family: str, limit: int = 16) -> dict:
    rows, material = composition_lookup()
    compiled = read_rows(OUTS["C417"] / "compiled/qwen3.jsonl")
    comp = rows_by_id(compiled)
    index = read_rows(OUTS["C418"] / "raw/hidden_index.jsonl")
    idx = rows_by_id(index)
    states = np.load(OUTS["C418"] / "raw/role_states.float16.npy", mmap_mode="r")
    pairs = []
    for axis, unit, order in itertools.product(AXES, (2, 3), (1, -1)):
        base = f"c416-{family}-briefing-u{unit}-0000-{axis}-{order:+d}"
        target = f"c416-{family}-briefing-u{unit}-1111-{axis}-{order:+d}"
        if (
            base in idx
            and target in idx
            and idx[base]["correct"]
            and idx[target]["correct"]
        ):
            pairs.append((base, target))
    pairs = pairs[:limit]
    if not pairs:
        close_memmap(states)
        return {"ran": False, "reason": "no_complete_behavior_correct_pairs"}
    model = None
    results = []
    try:
        model, _tokenizer, device, placement = parent.previous.common.model_base.load_bf16("qwen3")
        layer = model.model.layers[23]
        for base_id, target_id in pairs:
            base_row, target_row = comp[base_id], comp[target_id]
            donor = torch.tensor(
                np.asarray(
                    states[
                        idx[target_id]["hidden_index"],
                        24,
                        ROLES.index("relation"),
                    ],
                    np.float32,
                ),
                dtype=torch.bfloat16,
                device=device,
            )
            wrong_role = torch.tensor(
                np.asarray(
                    states[
                        idx[target_id]["hidden_index"],
                        24,
                        ROLES.index("context"),
                    ],
                    np.float32,
                ),
                dtype=torch.bfloat16,
                device=device,
            )
            ids = torch.tensor([base_row["prompt_ids"]], dtype=torch.long, device=device)
            positions = base_row["role_positions"]["relation"]

            def run(write=None):
                hook = None
                if write is not None:
                    def patch(_module, _args, output):
                        value = output[0] if isinstance(output, tuple) else output
                        changed = value.clone()
                        changed[0, positions] = write
                        return (changed, *output[1:]) if isinstance(output, tuple) else changed
                    hook = layer.register_forward_hook(patch)
                try:
                    output = model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        use_cache=False,
                        return_dict=True,
                    )
                    scores = [
                        float(output.logits[0, ids.shape[1] - 1, candidate[0]])
                        for candidate in base_row["candidate_ids"]
                    ]
                    target_pos = target_row["gold_position"]
                    return scores[target_pos] - scores[1 - target_pos]
                finally:
                    if hook is not None:
                        hook.remove()

            margins = {
                "baseline": run(),
                "correct": run(donor),
                "wrong_role": run(wrong_role),
                "zero": run(torch.zeros_like(donor)),
            }
            results.append(
                {
                    "base": base_id,
                    "target": target_id,
                    "margins": margins,
                    "correct_shift": margins["correct"] - margins["baseline"],
                    "control_shift": max(
                        margins[name] - margins["baseline"]
                        for name in ("wrong_role", "zero")
                    ),
                }
            )
        correct = float(np.mean([row["correct_shift"] for row in results]))
        control = float(np.mean([row["control_shift"] for row in results]))
        return {
            "ran": True,
            "placement": placement,
            "family": family,
            "pairs": len(results),
            "mean_correct_shift": correct,
            "mean_control_shift": control,
            "specificity_passed": correct > control + 0.05,
            "results": results,
        }
    finally:
        close_memmap(states)
        parent.previous.common.model_base.release(model)
        gc.collect()


def c424() -> None:
    joint = sorted(
        set(final("C419")["headline"]["candidate_families"])
        & set(final("C420")["headline"]["candidate_families"])
    )
    out = begin(
        "C424",
        {
            "status": "qualified_dynamic_state_writer_frozen",
            "candidate_rule": "intersection of C419 and C420",
            "writer": "q24 relation-role H1111 donor written into matched H0000 recipient",
            "controls": ["wrong role", "zero"],
            "known_truth_parent": "C408 passed",
        },
        {
            "parent": final("C423")["all_checks_passed"],
            "known_truth": parent.final("C408")["headline"]["writer_calibrated"],
        },
    )
    if not joint:
        headline = {
            "status": "dynamic_writer_not_run_no_joint_candidate",
            "candidates": [],
            "writer_ran": False,
            "strict_interpretation": "No new real-model causal conclusion is available.",
        }
    else:
        result = writer_test(joint[0])
        save(out / "analysis/writer_results.json", result)
        headline = {
            "status": "qualified_dynamic_state_writer_closed",
            "candidates": joint,
            "writer_ran": result["ran"],
            "result": result,
            "strict_interpretation": (
                "A specific whole-state write tests narrow sufficiency, not necessity "
                "or a minimal coordinate circuit."
            ),
        }
    close(
        "C424",
        headline,
        {"branch_accounted": ("result" in headline) == bool(joint), "finite": finite(headline)},
        "C425_synthesis",
    )


def hash_and_remove(paths: list[Path], out: Path) -> list[dict]:
    manifest = []
    for path in paths:
        if not path.exists():
            continue
        hasher = hashlib.sha256()
        size = path.stat().st_size
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                hasher.update(chunk)
        relative = str(path.relative_to(ROOT))
        path.unlink()
        manifest.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": hasher.hexdigest(),
                "removed": not path.exists(),
            }
        )
    save(out / "audit/cleanup.json", manifest)
    return manifest


def c425() -> None:
    out = begin(
        "C425",
        {
            "status": "dynamic_campaign_synthesis_heatmap_cleanup_frozen",
            "visual": "pair Mobius means and full-token singleton deltas, all 2560 coordinates",
            "cleanup": "streaming SHA256 then remove nonvisual bulk fields",
            "new_math_gate": (
                "requires replicated unseen-construction composition plus causal or "
                "cross-model operator evidence"
            ),
        },
        {"parent": final("C424")["all_checks_passed"]},
    )
    visual_rows = []
    if final("C419")["headline"]["prediction_ran"]:
        means = np.load(
            OUTS["C419"] / "analysis/mobius_source_means.float16.npy",
            mmap_mode="r",
        )
        for fi, family in enumerate(FAMILIES):
            for ai, axis in enumerate(AXES):
                for mask in range(1, 16):
                    if mask.bit_count() != 2:
                        continue
                    for role_i, role in enumerate(ROLES):
                        visual_rows.append(
                            {
                                "id": f"mobius:{family}:{axis}:m{mask}:q24:{role}",
                                "source": "second_order_mobius",
                                "family": family,
                                "query_axis": axis,
                                "mask": mask,
                                "checkpoint": 24,
                                "role": role,
                                "values": np.asarray(
                                    means[fi, ai, mask, 24, role_i], np.float32
                                )
                                .round(6)
                                .tolist(),
                            }
                        )
        close_memmap(means)
    if final("C421")["headline"]["observation_ran"]:
        deltas = np.load(
            OUTS["C421"] / "analysis/full_token_singleton_delta.float16.npy",
            mmap_mode="r",
        )
        meta = read_rows(
            OUTS["C421"] / "analysis/full_token_singleton_index.jsonl"
        )
        for row in meta:
            for token in range(8):
                visual_rows.append(
                    {
                        "id": (
                            f"token:{row['family']}:{row['query_axis']}:"
                            f"{row['factor']}:q24:t{token}"
                        ),
                        "source": "full_token_singleton_delta",
                        "family": row["family"],
                        "query_axis": row["query_axis"],
                        "factor": row["factor"],
                        "checkpoint": 24,
                        "token": token,
                        "values": np.asarray(
                            deltas[row["delta_index"], 24, token], np.float32
                        )
                        .round(6)
                        .tolist(),
                    }
                )
        close_memmap(deltas)
    payload = {
        "schema": "c425.dynamic_composition_field.v1",
        "phase": 1959,
        "campaign": "C425",
        "model": "Qwen3-4B",
        "dimensions": list(range(2560)),
        "rows": visual_rows,
        "summary": {
            "mobius_candidates": final("C419")["headline"]["candidate_families"],
            "dynamic_h000_candidates": final("C420")["headline"][
                "candidate_families"
            ],
            "graph_field_ran": final("C423")["headline"]["field_ran"],
            "writer_ran": final("C424")["headline"]["writer_ran"],
        },
        "claim_boundary": (
            "Every row retains all 2560 physical activation coordinates. Rows are "
            "observational responses unless C424 explicitly passes its writer gate."
        ),
    }
    save(VISUAL, payload)
    cleanup_paths = [
        OUTS["C418"] / "raw/role_states.float16.npy",
        OUTS["C418"] / "raw/full_fields_holdout.float16.npy",
        OUTS["C419"] / "analysis/mobius_source_means.float16.npy",
        OUTS["C421"] / "analysis/full_token_singleton_delta.float16.npy",
        OUTS["C423"] / "raw/role_states.float16.npy",
    ]
    cleanup = hash_and_remove(cleanup_paths, out)
    gates = {
        "balanced_behavior": final("C417")["headline"]["field_eligible"],
        "mobius_candidates": final("C419")["headline"]["candidate_families"],
        "dynamic_h000_candidates": final("C420")["headline"]["candidate_families"],
        "graph_field": final("C423")["headline"]["field_ran"],
        "writer": final("C424")["headline"]["writer_ran"],
        "causal": bool(
            final("C424")["headline"].get("result", {}).get(
                "specificity_passed", False
            )
        ),
        "new_math": False,
    }
    headline = {
        "status": "dynamic_composition_campaign_closed",
        "gates": gates,
        "visual_rows": len(visual_rows),
        "visual_path": str(VISUAL.relative_to(ROOT)),
        "cleanup_files": len(cleanup),
        "cleanup_bytes": sum(item["bytes"] for item in cleanup),
        "new_math_gate_passed": False,
        "strict_interpretation": (
            "The campaign separates balanced behavior, unseen-construction Mobius "
            "prediction, dynamic-donor prediction, graph qualification, and causality."
        ),
    }
    field_ran = final("C418")["headline"]["field_ran"]
    close(
        "C425",
        headline,
        {
            "visual": (
                (not field_ran and len(visual_rows) == 0)
                or (
                    len(visual_rows) > 0
                    and all(len(row["values"]) == 2560 for row in visual_rows)
                )
            ),
            "cleanup": all(item["removed"] and item["sha256"] for item in cleanup),
            "finite": finite(headline),
        },
        "independent_audit_then_next_language_family_campaign",
    )


RUNNERS = {name: globals()[name.lower()] for name in PHASES}


def parse_range(value: str) -> list[str]:
    if value in ("all", "C415-C425"):
        return list(PHASES)
    if "-" in value:
        left, right = value.split("-", 1)
        return [f"C{i}" for i in range(int(left[1:]), int(right[1:]) + 1)]
    return [value]


def validate_only() -> None:
    rows = composition_material()
    graph = graph_material()
    checks = {
        "composition_rows": len(rows) == 4608,
        "semantic_balance": sum(r["correct_answer"] == "Yes" for r in rows)
        == len(rows) // 2,
        "axis_balance": all(
            sum(
                r["correct_answer"] == "Yes"
                for r in rows
                if r["query_axis"] == axis
            )
            * 2
            == sum(r["query_axis"] == axis for r in rows)
            for axis in AXES
        ),
        "roles": all(set(r["role_values"]) == set(ROLES) - {"boundary"} for r in rows),
        "graph_rows": len(graph) == 3584,
        "graph_semantic_balance": sum(
            r["correct_answer"] == "Yes" for r in graph
        )
        == len(graph) // 2,
        "phase_sequence": [PHASES[f"C{i}"][0] for i in range(415, 426)]
        == list(range(1949, 1960)),
    }
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    if not all(checks.values()):
        raise AssertionError(checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="C415-C425")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_only()
        return
    for name in parse_range(args.run):
        RUNNERS[name]()


if __name__ == "__main__":
    main()
