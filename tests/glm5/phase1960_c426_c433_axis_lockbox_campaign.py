#!/usr/bin/env python3
"""C426-C433 prospective axis lockbox for balanced language composition."""
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
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c433_axis_lockbox_field.json"
sys.path.insert(0, str(TESTS))

import phase1949_c415_c425_dynamic_composition_campaign as base
import phase1797_c263_c272_state_operator_common as family_base


PHASES = {
    f"C{c}": (1960 + c - 426, slug)
    for c, slug in (
        (426, "axis_lockbox_contract"),
        (427, "new_lexicon_construction_axis_material"),
        (428, "qwen_axis_lockbox_behavior"),
        (429, "qualified_axis_full_coordinate_field"),
        (430, "axis_mobius_unseen_construction_prediction"),
        (431, "axis_dynamic_h000_donor_prediction"),
        (432, "qualified_axis_state_writer"),
        (433, "axis_campaign_synthesis_heatmap_cleanup"),
    )
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

FAMILIES = base.FAMILIES
AXES = ("outer", "attitude")
CONSTRUCTIONS = ("transcript", "memorandum", "casefile")
ROLES = base.ROLES
DIM = 2560
CHECKPOINTS = 38
FIELD_WIDTH = 192
UNITS = (
    {"p": "Tavin", "s": "Ulna", "o": "Varek", "obj": "kumquat", "other": "sextant"},
    {"p": "Wren", "s": "Xara", "o": "Yorin", "obj": "endive", "other": "astrolabe"},
    {"p": "Zalen", "s": "Abria", "o": "Borin", "obj": "lychee", "other": "telescope"},
    {"p": "Ceris", "s": "Doval", "o": "Emina", "obj": "salsify", "other": "barometer"},
    {"p": "Felis", "s": "Grava", "o": "Halen", "obj": "plantain", "other": "metronome"},
    {"p": "Irena", "s": "Javin", "o": "Korin", "obj": "chicory", "other": "theodolite"},
)


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
    if construction == "transcript":
        return (
            f"A transcript contains two entries. Main entry: {target} "
            f"Unrelated entry: {noise} Based only on the transcript, {question}"
        )
    if construction == "memorandum":
        return (
            f"A memorandum records {target} It separately records {noise} "
            f"Using the memorandum alone, {question}"
        )
    if construction == "casefile":
        return (
            f"A case file lists one relevant statement: {target} "
            f"It also lists an independent note: {noise} From the case file, {question}"
        )
    raise KeyError(construction)


def material() -> list[dict]:
    rows = []
    for family, construction, unit_i, bits, axis, order in itertools.product(
        FAMILIES,
        CONSTRUCTIONS,
        range(len(UNITS)),
        itertools.product((0, 1), repeat=4),
        AXES,
        (1, -1),
    ):
        d, a, b, c = bits
        unit = UNITS[unit_i]
        p, s, o = unit["p"], unit["s"], unit["o"]
        obj, other = unit["obj"], unit["other"]
        attitude = ("likes", "dislikes")[a]
        event = ("ate", "did not eat")[b]
        item = (obj, other)[c]
        if family == "attitude_event":
            outer = ("reported", "denied")[d]
            outer_question = (
                f"Did {o} report that {p} {attitude} the event in which "
                f"{s} {event} the {item}?"
            )
        elif family == "nested_attitude":
            outer = ("believes", "doubts")[d]
            outer_question = (
                f"Does {o} believe that {p} {attitude} the event in which "
                f"{s} {event} the {item}?"
            )
        else:
            outer = ("affirmed", "denied")[d]
            outer_question = (
                f"Did {o} affirm that {p} {attitude} the event in which "
                f"{s} {event} the {item}?"
            )
        target = (
            f"{o} {outer} that {p} {attitude} the event in which "
            f"{s} {event} the {item}."
        )
        attitude_question = (
            f"According to the relevant statement, does {p} like the event "
            f"in which {s} {event} the {item}?"
        )
        question = outer_question if axis == "outer" else attitude_question
        truth = bits[AXES.index(axis)] == 0
        correct, wrong = ("Yes", "No") if truth else ("No", "Yes")
        noise = (
            f"{s} catalogued the {other} for {p} while {o} checked the archive."
        )
        core = wrap(construction, target, noise, question)
        choices, gold = family_base.options(correct, wrong, order)
        cell = "".join(str(bit) for bit in bits)
        rows.append(
            {
                "case_id": (
                    f"c427-{family}-{construction}-u{unit_i}-{cell}-{axis}-{order:+d}"
                ),
                "panel": "axis_lockbox",
                "family": family,
                "surface": construction,
                "construction": construction,
                "unit": unit_i,
                "cell": cell,
                "mask": sum(bit << i for i, bit in enumerate(bits)),
                "query_axis": axis,
                "order": order,
                "partition": (
                    "discovery"
                    if unit_i < 3
                    else "confirmation"
                    if unit_i < 5
                    else "lockbox"
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
                    "relation": outer if axis == "outer" else attitude,
                    "context": item,
                    "query": item,
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
        )
    return rows


def lookup() -> tuple[list[dict], dict[str, dict]]:
    rows = read_rows(OUTS["C427"] / "material/cases.jsonl")
    return rows, {row["case_id"]: row for row in rows}


def c426() -> None:
    audit = load(base.OUTS["C425"] / "audit/independent_audit.json")
    out = begin(
        "C426",
        {
            "status": "axis_lockbox_contract_frozen",
            "development_candidates": {
                "outer": base.final("C417")["headline"]["query_axis_accuracy"]["outer"],
                "attitude": base.final("C417")["headline"]["query_axis_accuracy"]["attitude"],
            },
            "confirmation": "six new lexicons and three new constructions",
            "axis_gate": {
                "heldout": 0.70,
                "family_within_axis": 0.65,
                "construction_within_axis": 0.65,
            },
            "policy": "each axis qualifies independently; no event/object retest",
        },
        {
            "parent": audit["all_checks_passed"],
            "phase_continuity": PHASES["C426"][0] == 1960,
        },
    )
    headline = {
        "status": "axis_lockbox_contract_closed",
        "candidates": list(AXES),
        "retired_for_this_campaign": ["event", "object"],
        "strict_interpretation": (
            "Outer and attitude are development candidates only; no previous "
            "HiddenState is reused for confirmation."
        ),
    }
    close("C426", headline, {"audit": audit["passed"] == audit["total"]}, "C427_material")


def c427() -> None:
    out = begin(
        "C427",
        {
            "status": "new_axis_lockbox_material_frozen",
            "design": (
                "3 families x 3 new constructions x 6 new lexicons x "
                "16 corners x 2 axes x 2 answer orders"
            ),
            "human_naturalness_review": False,
        },
        {"parent": final("C426")["all_checks_passed"]},
    )
    rows = material()
    write_rows(out / "material/cases.jsonl", rows)
    yes = sum(row["correct_answer"] == "Yes" for row in rows) / len(rows)
    position = sum(row["gold_position"] == 0 for row in rows) / len(rows)
    axis_yes = {
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
        "status": "new_axis_lockbox_material_closed",
        "rows": len(rows),
        "partition_counts": {
            part: sum(row["partition"] == part for row in rows)
            for part in ("discovery", "confirmation", "lockbox")
        },
        "yes_frequency": yes,
        "first_position_frequency": position,
        "axis_yes_frequency": axis_yes,
        "role_occurrence": roles,
        "material_eligible": abs(yes - 0.5) < 1e-12
        and abs(position - 0.5) < 1e-12
        and all(abs(v - 0.5) < 1e-12 for v in axis_yes.values())
        and roles,
        "human_naturalness_review": False,
        "strict_interpretation": "New lexicons and wrappers are synthetic controlled English.",
    }
    close(
        "C427",
        headline,
        {
            "rows": len(rows) == 3456,
            "balance": headline["material_eligible"],
        },
        "C428_behavior",
    )


def c428() -> None:
    out = begin(
        "C428",
        {
            "status": "qwen_axis_lockbox_behavior_frozen",
            "model": "Qwen3-4B CUDA BF16",
            "axis_gate": {
                "heldout": 0.70,
                "family_within_axis": 0.65,
                "construction_within_axis": 0.65,
            },
            "hidden_state_policy": "none",
        },
        {
            "parent": final("C427")["all_checks_passed"],
            "material": final("C427")["headline"]["material_eligible"],
            "cuda": torch.cuda.is_available(),
        },
    )
    rows, material_by_id = lookup()
    tokenizer = base.parent.fresh.tokenizer_qwen()
    compiled = family_base.compile_qwen(tokenizer, rows)
    write_rows(out / "compiled/qwen3.jsonl", compiled)
    metrics = base.parent.previous.qwen_behavior(rows, compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl")
    held = [row for row in behavior if row["partition"] != "discovery"]
    axis_results = {}
    eligible_axes = []
    for axis in AXES:
        subset = [
            row
            for row in held
            if material_by_id[row["case_id"]]["query_axis"] == axis
        ]
        family_accuracy = {
            family: float(
                np.mean([row["correct"] for row in subset if row["family"] == family])
            )
            for family in FAMILIES
        }
        construction_accuracy = {
            construction: float(
                np.mean(
                    [row["correct"] for row in subset if row["surface"] == construction]
                )
            )
            for construction in CONSTRUCTIONS
        }
        accuracy = float(np.mean([row["correct"] for row in subset]))
        eligible = (
            accuracy >= 0.70
            and min(family_accuracy.values()) >= 0.65
            and min(construction_accuracy.values()) >= 0.65
        )
        if eligible:
            eligible_axes.append(axis)
        axis_results[axis] = {
            "accuracy": accuracy,
            "family_accuracy": family_accuracy,
            "construction_accuracy": construction_accuracy,
            "eligible": eligible,
        }
    headline = {
        "status": "qwen_axis_lockbox_behavior_closed",
        **metrics,
        "heldout_accuracy": float(np.mean([row["correct"] for row in held])),
        "axis_results": axis_results,
        "eligible_axes": eligible_axes,
        "field_ran_authorized": bool(eligible_axes),
        "strict_interpretation": (
            "Only prospectively confirmed axes are authorized for internal observation."
        ),
    }
    close(
        "C428",
        headline,
        {
            "rows": len(behavior) == len(rows),
            "finite": finite(headline),
            "no_hidden": not (out / "raw/role_states.float16.npy").exists(),
        },
        "C429_field",
    )


def c429() -> None:
    axes = final("C428")["headline"]["eligible_axes"]
    out = begin(
        "C429",
        {
            "status": "qualified_axis_full_coordinate_field_frozen",
            "qualified_axes": axes,
            "archive": "38 checkpoints x six roles x all 2560 coordinates",
            "full_token_subset": "casefile lockbox base plus singleton factors",
            "no_pca_topk": True,
        },
        {"parent": final("C428")["all_checks_passed"]},
    )
    if not axes:
        headline = {
            "status": "axis_field_not_run_no_confirmed_axis",
            "field_ran": False,
            "qualified_axes": [],
            "strict_interpretation": "No axis-level internal conclusion is available.",
        }
        close("C429", headline, {"route_accounted": True}, "C430_mobius")
        return
    rows, _ = lookup()
    compiled = read_rows(OUTS["C428"] / "compiled/qwen3.jsonl")
    selected = [
        (row, comp)
        for row, comp in zip(rows, compiled)
        if row["query_axis"] in axes
    ]
    rows2 = [row for row, _ in selected]
    compiled2 = [comp for _, comp in selected]
    singleton = {"0000", "1000", "0100", "0010", "0001"}
    selector = lambda row: (
        row["construction"] == "casefile"
        and row["unit"] == 5
        and row["order"] == 1
        and row["cell"] in singleton
    )
    metrics = base.parent.previous.common.batch_capture_qwen(
        rows2,
        compiled2,
        out,
        full_selector=selector,
        batch_size=8,
        field_width=FIELD_WIDTH,
    )
    headline = {
        "status": "qualified_axis_full_coordinate_field_closed",
        **metrics,
        "field_ran": True,
        "qualified_axes": axes,
        "strict_interpretation": (
            "All rows on a qualified axis are retained for observation; correctness "
            "is preserved as metadata rather than used as a coordinate filter."
        ),
    }
    close(
        "C429",
        headline,
        {
            "shape": metrics["role_shape"][1:] == [38, 6, 2560],
            "full": metrics["full_shape"][-1] == 2560,
            "finite": finite(headline),
        },
        "C430_mobius",
    )


def complete_groups(
    states: np.ndarray,
    index: list[dict],
    material_by_id: dict[str, dict],
    family: str,
    axis: str,
) -> list[dict]:
    keyed = {}
    for row in index:
        case = material_by_id[row["case_id"]]
        if case["family"] == family and case["query_axis"] == axis:
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
        rows = [keyed[key] for key in keys]
        h = {
            mask: np.asarray(
                states[keyed[(construction, unit, order, mask)]["hidden_index"]],
                np.float32,
            )
            for mask in range(16)
        }
        effects = {}
        for mask in range(1, 16):
            total = np.zeros_like(h[0])
            subset = mask
            while True:
                sign = (
                    -1.0
                    if (mask.bit_count() - subset.bit_count()) % 2
                    else 1.0
                )
                total += sign * h[subset]
                if subset == 0:
                    break
                subset = (subset - 1) & mask
            effects[mask] = total
        groups.append(
            {
                "construction": construction,
                "unit": unit,
                "order": order,
                "H0": h[0],
                "H15": h[15],
                "target_correct": keyed[(construction, unit, order, 15)][
                    "correct"
                ],
                "effects": effects,
            }
        )
    return groups


def predict(base_state: np.ndarray, effects: dict[int, np.ndarray], order: int):
    value = np.asarray(base_state, np.float32).copy()
    for mask, effect in effects.items():
        if mask.bit_count() <= order:
            value += effect
    return value


def c430() -> None:
    axes = final("C429")["headline"]["qualified_axes"]
    out = begin(
        "C430",
        {
            "status": "axis_mobius_unseen_construction_prediction_frozen",
            "train": "transcript+memorandum discovery units0-2",
            "test": "casefile confirmation+lockbox units3-5",
            "primary_panel": "all rows on behavior-qualified axis",
            "secondary_panel": "behavior-correct target rows",
            "pass": (
                "pair gain >0.01 on both panels and pair beats wrong-family, "
                "coordinate-roll, and zero controls"
            ),
        },
        {"parent": final("C429")["all_checks_passed"]},
    )
    if not final("C429")["headline"]["field_ran"]:
        headline = {
            "status": "axis_mobius_not_run_no_field",
            "prediction_ran": False,
            "candidate_families": [],
            "strict_interpretation": "The axis Mobius hypothesis remains untested.",
        }
        close("C430", headline, {"route_accounted": True}, "C431_dynamic_h000")
        return
    states = np.load(OUTS["C429"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C429"] / "raw/hidden_index.jsonl")
    _, material_by_id = lookup()
    groups = {
        (family, axis): complete_groups(
            states, index, material_by_id, family, axis
        )
        for family, axis in itertools.product(FAMILIES, axes)
    }
    means_path = out / "analysis/axis_mobius_means.float16.npy"
    means_arr = np.lib.format.open_memmap(
        means_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(FAMILIES), len(axes), 16, 38, 6, 2560),
    )
    means = {}
    for fi, family in enumerate(FAMILIES):
        for ai, axis in enumerate(axes):
            train = [
                group
                for group in groups[(family, axis)]
                if group["construction"] in ("transcript", "memorandum")
                and group["unit"] < 3
            ]
            effect_mean = {
                mask: np.mean([group["effects"][mask] for group in train], axis=0)
                for mask in range(1, 16)
            }
            means[(family, axis)] = effect_mean
            for mask, value in effect_mean.items():
                means_arr[fi, ai, mask] = value.astype(np.float16)
            means_arr.flush()
    results = []
    for fi, family in enumerate(FAMILIES):
        wrong_family = FAMILIES[(fi + 1) % len(FAMILIES)]
        for axis in axes:
            test = [
                group
                for group in groups[(family, axis)]
                if group["construction"] == "casefile" and group["unit"] >= 3
            ]
            effect_mean = means[(family, axis)]
            wrong_mean = means[(wrong_family, axis)]
            panels = {}
            for panel_name, panel_groups in (
                ("all", test),
                ("correct_target", [group for group in test if group["target_correct"]]),
            ):
                if not panel_groups:
                    panels[panel_name] = {"ran": False}
                    continue
                scores = {
                    name: []
                    for name in (
                        "order1",
                        "order2",
                        "order3",
                        "order4",
                        "wrong_family",
                        "coordinate_roll",
                        "zero",
                    )
                }
                for group in panel_groups:
                    truth = group["H15"]
                    predictions = {
                        f"order{k}": predict(group["H0"], effect_mean, k)
                        for k in range(1, 5)
                    }
                    pair_response = predictions["order2"] - group["H0"]
                    controls = {
                        "wrong_family": predict(group["H0"], wrong_mean, 2),
                        "coordinate_roll": group["H0"]
                        + np.roll(pair_response, 1, axis=-1),
                        "zero": group["H0"],
                    }
                    for name, value in {**predictions, **controls}.items():
                        scores[name].append(nrmse(value, truth))
                values = {name: float(np.mean(v)) for name, v in scores.items()}
                gain = values["order1"] - values["order2"]
                panels[panel_name] = {
                    "ran": True,
                    "groups": len(panel_groups),
                    "nrmse": values,
                    "pair_gain": gain,
                    "passed": gain > 0.01
                    and values["order2"]
                    < min(
                        values["wrong_family"],
                        values["coordinate_roll"],
                        values["zero"],
                    ),
                }
            passed = (
                panels["all"].get("passed", False)
                and panels["correct_target"].get("passed", False)
            )
            results.append(
                {
                    "family": family,
                    "query_axis": axis,
                    "panels": panels,
                    "passed": passed,
                }
            )
    write_rows(out / "analysis/axis_mobius_predictions.jsonl", results)
    pass_counts = {
        family: sum(
            row["passed"] for row in results if row["family"] == family
        )
        for family in FAMILIES
    }
    candidates = [family for family, count in pass_counts.items() if count >= 1]
    headline = {
        "status": "axis_mobius_unseen_construction_prediction_closed",
        "prediction_ran": True,
        "cells": len(results),
        "pass_counts": pass_counts,
        "candidate_families": candidates,
        "strict_interpretation": (
            "All-row prediction is observational; correct-target replication is "
            "required before a family becomes a candidate."
        ),
    }
    close_memmap(means_arr)
    close_memmap(states)
    close(
        "C430",
        headline,
        {"cells": len(results) == len(FAMILIES) * len(axes), "finite": finite(headline)},
        "C431_dynamic_h000",
    )


def distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.mean(
            (np.asarray(left, np.float32) - np.asarray(right, np.float32)) ** 2
        )
    )


def c431() -> None:
    axes = final("C429")["headline"]["qualified_axes"]
    out = begin(
        "C431",
        {
            "status": "axis_dynamic_h000_donor_prediction_frozen",
            "source": "full 38x6x2560 H0000 baseline",
            "target": "H1111-H0000",
            "controls": ["q0 donor", "mean response", "coordinate roll"],
            "pass": "beats q0 and mean by >0.005 and beats roll",
        },
        {"parent": final("C430")["all_checks_passed"]},
    )
    if not final("C430")["headline"]["prediction_ran"]:
        headline = {
            "status": "axis_dynamic_h000_not_run",
            "prediction_ran": False,
            "candidate_families": [],
        }
        close("C431", headline, {"route_accounted": True}, "C432_writer")
        return
    states = np.load(OUTS["C429"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C429"] / "raw/hidden_index.jsonl")
    _, material_by_id = lookup()
    results = []
    for family, axis in itertools.product(FAMILIES, axes):
        groups = complete_groups(states, index, material_by_id, family, axis)
        train = [
            group
            for group in groups
            if group["construction"] in ("transcript", "memorandum")
            and group["unit"] < 3
        ]
        test = [
            group
            for group in groups
            if group["construction"] == "casefile" and group["unit"] >= 3
        ]
        mean_response = np.mean(
            [group["H15"] - group["H0"] for group in train], axis=0
        )
        scores = {
            name: []
            for name in ("dynamic", "q0", "mean", "coordinate_roll")
        }
        for group in test:
            dynamic_i = int(
                np.argmin([distance(group["H0"], donor["H0"]) for donor in train])
            )
            q0_i = int(
                np.argmin(
                    [distance(group["H0"][0], donor["H0"][0]) for donor in train]
                )
            )
            dynamic_response = train[dynamic_i]["H15"] - train[dynamic_i]["H0"]
            q0_response = train[q0_i]["H15"] - train[q0_i]["H0"]
            truth = group["H15"]
            predictions = {
                "dynamic": group["H0"] + dynamic_response,
                "q0": group["H0"] + q0_response,
                "mean": group["H0"] + mean_response,
                "coordinate_roll": group["H0"]
                + np.roll(dynamic_response, 1, axis=-1),
            }
            for name, value in predictions.items():
                scores[name].append(nrmse(value, truth))
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
                "nrmse": values,
                "gain_over_mean": gain_mean,
                "gain_over_q0": gain_q0,
                "passed": passed,
            }
        )
    write_rows(out / "analysis/axis_dynamic_h000_predictions.jsonl", results)
    pass_counts = {
        family: sum(
            row["passed"] for row in results if row["family"] == family
        )
        for family in FAMILIES
    }
    candidates = [family for family, count in pass_counts.items() if count >= 1]
    headline = {
        "status": "axis_dynamic_h000_donor_prediction_closed",
        "prediction_ran": True,
        "cells": len(results),
        "pass_counts": pass_counts,
        "candidate_families": candidates,
        "strict_interpretation": (
            "A passed full-H000 donor is an effective forecast rule, not a unique "
            "neural state or causal path."
        ),
    }
    close_memmap(states)
    close(
        "C431",
        headline,
        {"cells": len(results) == len(FAMILIES) * len(axes), "finite": finite(headline)},
        "C432_writer",
    )


@torch.inference_mode()
def writer_test(family: str, axis: str, limit: int = 12) -> dict:
    rows, _ = lookup()
    compiled = read_rows(OUTS["C428"] / "compiled/qwen3.jsonl")
    comp = {row["case_id"]: row for row in compiled}
    index = read_rows(OUTS["C429"] / "raw/hidden_index.jsonl")
    idx = {row["case_id"]: row for row in index}
    states = np.load(OUTS["C429"] / "raw/role_states.float16.npy", mmap_mode="r")
    pairs = []
    for unit, order in itertools.product((3, 4, 5), (1, -1)):
        base_id = f"c427-{family}-casefile-u{unit}-0000-{axis}-{order:+d}"
        target_id = f"c427-{family}-casefile-u{unit}-1111-{axis}-{order:+d}"
        if base_id in idx and target_id in idx:
            pairs.append((base_id, target_id))
    pairs = pairs[:limit]
    if not pairs:
        close_memmap(states)
        return {"ran": False, "reason": "no_pairs"}
    model = None
    results = []
    try:
        model, _tokenizer, device, placement = (
            base.parent.previous.common.model_base.load_bf16("qwen3")
        )
        layer = model.model.layers[23]
        for base_id, target_id in pairs:
            base_row = comp[base_id]
            target_row = comp[target_id]
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
                    target_position = target_row["gold_position"]
                    return scores[target_position] - scores[1 - target_position]
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
                        margins["wrong_role"] - margins["baseline"],
                        margins["zero"] - margins["baseline"],
                    ),
                }
            )
        correct = float(np.mean([row["correct_shift"] for row in results]))
        control = float(np.mean([row["control_shift"] for row in results]))
        return {
            "ran": True,
            "placement": placement,
            "family": family,
            "axis": axis,
            "pairs": len(results),
            "mean_correct_shift": correct,
            "mean_control_shift": control,
            "specificity_passed": correct > control + 0.05,
            "results": results,
        }
    finally:
        close_memmap(states)
        base.parent.previous.common.model_base.release(model)
        gc.collect()


def c432() -> None:
    joint = sorted(
        set(final("C430")["headline"]["candidate_families"])
        & set(final("C431")["headline"]["candidate_families"])
    )
    out = begin(
        "C432",
        {
            "status": "qualified_axis_state_writer_frozen",
            "candidate_rule": "C430 and C431 family intersection",
            "writer": "q24 relation-role H1111 donor into matched casefile H0000",
            "controls": ["wrong role", "zero"],
        },
        {
            "parent": final("C431")["all_checks_passed"],
            "known_truth": base.parent.final("C408")["headline"]["writer_calibrated"],
        },
    )
    if not joint:
        headline = {
            "status": "axis_writer_not_run_no_joint_candidate",
            "candidates": [],
            "writer_ran": False,
            "strict_interpretation": "No causal axis conclusion is available.",
        }
    else:
        axis = final("C429")["headline"]["qualified_axes"][0]
        result = writer_test(joint[0], axis)
        save(out / "analysis/writer_results.json", result)
        headline = {
            "status": "qualified_axis_state_writer_closed",
            "candidates": joint,
            "writer_ran": result["ran"],
            "result": result,
            "strict_interpretation": (
                "Whole-state writing tests narrow sufficiency, not necessity or "
                "minimal coordinate mechanism."
            ),
        }
    close(
        "C432",
        headline,
        {"branch_accounted": ("result" in headline) == bool(joint), "finite": finite(headline)},
        "C433_synthesis",
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


def c433() -> None:
    out = begin(
        "C433",
        {
            "status": "axis_campaign_synthesis_heatmap_cleanup_frozen",
            "visual": "q24 pair Mobius effects, all 2560 coordinates",
            "cleanup": "streaming SHA256 then remove bulk fields",
        },
        {"parent": final("C432")["all_checks_passed"]},
    )
    rows = []
    if final("C430")["headline"]["prediction_ran"]:
        means = np.load(
            OUTS["C430"] / "analysis/axis_mobius_means.float16.npy",
            mmap_mode="r",
        )
        axes = final("C429")["headline"]["qualified_axes"]
        for fi, family in enumerate(FAMILIES):
            for ai, axis in enumerate(axes):
                for mask in range(1, 16):
                    if mask.bit_count() != 2:
                        continue
                    for ri, role in enumerate(ROLES):
                        rows.append(
                            {
                                "id": f"axis:{family}:{axis}:m{mask}:q24:{role}",
                                "source": "axis_pair_mobius",
                                "family": family,
                                "query_axis": axis,
                                "mask": mask,
                                "checkpoint": 24,
                                "role": role,
                                "values": np.asarray(
                                    means[fi, ai, mask, 24, ri], np.float32
                                )
                                .round(6)
                                .tolist(),
                            }
                        )
        close_memmap(means)
    payload = {
        "schema": "c433.axis_lockbox_field.v1",
        "phase": 1967,
        "campaign": "C433",
        "model": "Qwen3-4B",
        "dimensions": list(range(2560)),
        "rows": rows,
        "summary": {
            "eligible_axes": final("C428")["headline"]["eligible_axes"],
            "mobius_candidates": final("C430")["headline"]["candidate_families"],
            "dynamic_candidates": final("C431")["headline"]["candidate_families"],
            "writer_ran": final("C432")["headline"]["writer_ran"],
        },
        "claim_boundary": (
            "Rows retain all 2560 activation coordinates and remain observational "
            "unless the separately reported writer branch passes."
        ),
    }
    save(VISUAL, payload)
    cleanup = hash_and_remove(
        [
            OUTS["C429"] / "raw/role_states.float16.npy",
            OUTS["C429"] / "raw/full_fields_holdout.float16.npy",
            OUTS["C430"] / "analysis/axis_mobius_means.float16.npy",
        ],
        out,
    )
    field_ran = final("C429")["headline"]["field_ran"]
    headline = {
        "status": "axis_lockbox_campaign_closed",
        "eligible_axes": final("C428")["headline"]["eligible_axes"],
        "mobius_candidates": final("C430")["headline"]["candidate_families"],
        "dynamic_candidates": final("C431")["headline"]["candidate_families"],
        "writer_ran": final("C432")["headline"]["writer_ran"],
        "causal": bool(
            final("C432")["headline"].get("result", {}).get(
                "specificity_passed", False
            )
        ),
        "visual_rows": len(rows),
        "visual_path": str(VISUAL.relative_to(ROOT)),
        "cleanup_files": len(cleanup),
        "cleanup_bytes": sum(row["bytes"] for row in cleanup),
        "new_math_gate_passed": False,
        "strict_interpretation": (
            "Axis-level qualification prevents weak event/object behavior from "
            "masking or legitimizing outer/attitude internal evidence."
        ),
    }
    close(
        "C433",
        headline,
        {
            "visual": (
                (not field_ran and not rows)
                or (
                    bool(rows)
                    and all(len(row["values"]) == 2560 for row in rows)
                )
            ),
            "cleanup": all(row["removed"] and row["sha256"] for row in cleanup),
            "finite": finite(headline),
        },
        "independent_audit_then_human_naturalness_or_new_language_family",
    )


RUNNERS = {name: globals()[name.lower()] for name in PHASES}


def parse_range(value: str) -> list[str]:
    if value in ("all", "C426-C433"):
        return list(PHASES)
    if "-" in value:
        left, right = value.split("-", 1)
        return [f"C{i}" for i in range(int(left[1:]), int(right[1:]) + 1)]
    return [value]


def validate_only() -> None:
    rows = material()
    checks = {
        "rows": len(rows) == 3456,
        "semantic_balance": sum(r["correct_answer"] == "Yes" for r in rows)
        == len(rows) // 2,
        "position_balance": sum(r["gold_position"] == 0 for r in rows)
        == len(rows) // 2,
        "axes": {r["query_axis"] for r in rows} == set(AXES),
        "roles": all(set(r["role_values"]) == set(ROLES) - {"boundary"} for r in rows),
        "phase_sequence": [PHASES[f"C{i}"][0] for i in range(426, 434)]
        == list(range(1960, 1968)),
    }
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    if not all(checks.values()):
        raise AssertionError(checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="C426-C433")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_only()
        return
    for name in parse_range(args.run):
        RUNNERS[name]()


if __name__ == "__main__":
    main()
