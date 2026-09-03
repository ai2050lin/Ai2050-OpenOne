#!/usr/bin/env python3
"""C471-C484 language-program guard and full-coordinate coupling campaign.

Neural observations are embeddings and HiddenState checkpoints only. Every
registered response keeps all 2560 physical activation coordinates. No PCA,
Top-K coordinate selection, Attention/MLP inspection, or weight inspection is
used. Full-coordinate ridge is evaluated as a predictive dependency object,
not as a unique causal circuit.
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
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c484_program_guard_hypergraph.json"
sys.path.insert(0, str(TESTS))

import phase1988_c454_c470_semantic_residual_campaign as prior

PHASES = {
    f"C{campaign}": (2005 + campaign - 471, slug)
    for campaign, slug in (
        (471, "evidence_adjudication_and_program_guard_master_contract"),
        (472, "eight_family_four_factor_language_program_material"),
        (473, "program_compiler_zero_model_and_naturalness_audit"),
        (474, "qwen_program_graph_behavior_qualification"),
        (475, "qualified_program_graph_full_coordinate_field"),
        (476, "complete_factorial_walsh_response_ledger"),
        (477, "observation_first_condition_and_coordinate_atlas"),
        (478, "shared_propagation_reconstruction"),
        (479, "family_blind_program_graph_guard"),
        (480, "coordinate_state_guarded_propagation"),
        (481, "cross_family_selector_lockbox_adjudication"),
        (482, "arbitrary_full_coordinate_coupling_tournament"),
        (483, "unseen_composition_and_conditional_writer"),
        (484, "campaign_synthesis_visual_cleanup_and_audit"),
    )
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}" for name, (phase, slug) in PHASES.items()}

DIM = 2560
CHECKPOINTS = 38
ROLES = prior.ROLES
FIELD_WIDTH = 192
CONSTRUCTIONS = ("ledger", "brief", "report")
FAMILIES = (
    "attitude_event", "agent_patient_relation", "predicate_negation", "comparison",
    "temporal_order", "causal_relation", "type_graph", "part_whole",
)
TRAIN_FAMILIES = FAMILIES[:6]
FAMILY_LOCKBOX = FAMILIES[6:]
FACTOR_NAMES = ("entity_binding", "relation_polarity", "context_binding", "query_match")
BITS = tuple(itertools.product((0, 1), repeat=4))
MASKS = tuple(range(1, 16))
Q_STARTS = (0, 8, 16, 24, 32)
QPOINTS = tuple(value for q in Q_STARTS for value in (q, q + 1))

UNITS = (
    {"p": "Aldren", "s": "Beral", "x": "cedar", "y": "dahlia", "noise": "sextant"},
    {"p": "Cyran", "s": "Delis", "x": "elm", "y": "freesia", "noise": "theodolite"},
    {"p": "Edrin", "s": "Fara", "x": "ginkgo", "y": "heather", "noise": "barometer"},
    {"p": "Galen", "s": "Heris", "x": "iris", "y": "juniper", "noise": "calorimeter"},
    {"p": "Ilyan", "s": "Jora", "x": "kelp", "y": "linden", "noise": "dynamometer"},
    {"p": "Korin", "s": "Lysa", "x": "moss", "y": "nettle", "noise": "electrometer"},
    {"p": "Miren", "s": "Neral", "x": "orchid", "y": "palm", "noise": "galvanometer"},
    {"p": "Oren", "s": "Pelis", "x": "quince", "y": "reed", "noise": "hygrometer"},
    {"p": "Quarin", "s": "Rela", "x": "spruce", "y": "thyme", "noise": "interferometer"},
    {"p": "Soren", "s": "Talia", "x": "ulmus", "y": "verbena", "noise": "manometer"},
)

RELATIONS = {
    "attitude_event": ("likes eating", "does not like eating"),
    "agent_patient_relation": ("guides", "does not guide"),
    "predicate_negation": ("describes", "does not describe"),
    "comparison": ("is taller than", "is not taller than"),
    "temporal_order": ("occurred before", "did not occur before"),
    "causal_relation": ("caused", "did not cause"),
    "type_graph": ("is a kind of", "is not a kind of"),
    "part_whole": ("is part of", "is not part of"),
}


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
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


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


def wrap(construction: str, fact: str, noise: str, question: str) -> str:
    if construction == "ledger":
        return f"A ledger marks one relevant fact: {fact} An unrelated margin note says {noise}. Using only the marked fact, {question}"
    if construction == "brief":
        return f"A brief contains a relevant statement, {fact} Separately, {noise}. From the relevant statement alone, {question}"
    if construction == "report":
        return f"A report records that {fact} Elsewhere it records that {noise}. Based only on the first record, {question}"
    raise KeyError(construction)


def render_clause(family: str, entity: str, relation_bit: int, context: str) -> tuple[str, str]:
    relation = RELATIONS[family][relation_bit]
    if family == "predicate_negation":
        if relation_bit == 0:
            return f"{entity} describes {context} as bright", relation
        return f"{entity} does not describe {context} as bright", relation
    if family in ("temporal_order", "causal_relation"):
        return f"the {entity} event {relation} the {context} event", relation
    return f"{entity} {relation} {context}", relation


def program_case(family: str, construction: str, unit: int, bits: tuple[int, int, int, int]) -> dict:
    u = UNITS[unit]
    entity = u["p"] if bits[0] == 0 else u["s"]
    secondary = u["s"] if bits[0] == 0 else u["p"]
    context = u["x"] if bits[2] == 0 else u["y"]
    fact_clause, relation = render_clause(family, entity, bits[1], context)
    query_clause, query_relation = render_clause(family, entity, bits[1] ^ bits[3], context)
    fact = f"{fact_clause}."
    question = f"Is it true that {query_clause}?"
    noise = f"{secondary} inspected the {u['noise']}"
    core = wrap(construction, fact, noise, question)
    truth = bits[3] == 0
    return {
        "prompt_core": core,
        "truth": truth,
        "roles": {"primary": entity, "secondary": secondary, "relation": relation, "context": context, "query": entity},
        "semantic_graph": {
            "family": family, "factor_names": FACTOR_NAMES, "bits": list(bits),
            "nodes": [entity, secondary, context], "relation": relation,
            "query_relation": query_relation, "truth": truth,
        },
    }


def material() -> list[dict]:
    rows = []
    for family, construction, unit, bits in itertools.product(FAMILIES, CONSTRUCTIONS, range(10), BITS):
        case = program_case(family, construction, unit, bits)
        prompt = f"{case['prompt_core']} (A) Yes (B) No. Reply with only A or B."
        bit_code = "".join(map(str, bits))
        rows.append({
            "case_id": f"c472-{family}-{construction}-u{unit}-x{bit_code}", "panel": "program_factorial",
            "family": family, "surface": construction, "construction": construction, "unit": unit,
            "bits": list(bits), "factor_a": bits[0], "factor_b": bits[1], "factor_c": bits[2], "factor_d": bits[3],
            "order": 1, "partition": partition(unit), "gold_position": 0 if case["truth"] else 1,
            "correct_answer": "Yes" if case["truth"] else "No", "wrong_answer": "No" if case["truth"] else "Yes",
            "prompt_core": case["prompt_core"], "prompt": prompt,
            "free_prompt": f"{case['prompt_core']} Answer only Yes or No.",
            "role_values": case["roles"], "semantic_graph": case["semantic_graph"],
        })
    return rows


def material_lookup() -> tuple[list[dict], dict[str, dict]]:
    rows = read_rows(OUTS["C472"] / "material/cases.jsonl")
    return rows, {row["case_id"]: row for row in rows}


def mask_order(mask: int) -> int:
    return int(mask).bit_count()


def walsh_weights(mask: int) -> np.ndarray:
    values = []
    for bits in BITS:
        sign = 1
        for bit in range(4):
            if mask & (1 << bit):
                sign *= 1 if bits[bit] else -1
        values.append(sign / 16.0)
    return np.asarray(values, dtype=np.float32)


def metric() -> dict:
    return {"se": 0.0, "sy": 0.0, "n": 0}


def add_metric(acc: dict, prediction: np.ndarray, truth: np.ndarray) -> None:
    acc["se"] += float(np.sum((prediction - truth) ** 2, dtype=np.float64))
    acc["sy"] += float(np.sum(truth ** 2, dtype=np.float64))
    acc["n"] += int(truth.shape[0])


def finish_metric(acc: dict) -> dict:
    return {"nrmse": math.sqrt(acc["se"] / (acc["sy"] + 1e-12)), "samples": acc["n"]}


def fit_diagonal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return prior.fit_diagonal(x, y)


def close_mmap(value) -> None:
    prior.close_mmap(value)


def effect_splits(records: list[dict]) -> dict[str, list[dict]]:
    low = lambda row: row["effect_order"] <= 2
    return {
        "train": [r for r in records if r["family"] in TRAIN_FAMILIES and r["unit"] < 5 and r["construction"] in CONSTRUCTIONS[:2] and low(r)],
        "within": [r for r in records if r["family"] in TRAIN_FAMILIES and r["unit"] >= 5 and r["construction"] in CONSTRUCTIONS[:2] and low(r)],
        "family": [r for r in records if r["family"] in FAMILY_LOCKBOX and r["unit"] >= 5 and low(r)],
        "report": [r for r in records if r["family"] in TRAIN_FAMILIES and r["unit"] >= 5 and r["construction"] == "report" and low(r)],
        "composition": [r for r in records if r["unit"] >= 5 and r["effect_order"] >= 3],
    }


def effect_arrays(states: np.ndarray, records: list[dict], q_index: int, role: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray([row["effect_index"] for row in records], dtype=np.int64)
    return np.asarray(states[indices, q_index, role], dtype=np.float32), np.asarray(states[indices, q_index + 1, role], dtype=np.float32)


def model_paths(name: str, stem: str) -> Path:
    return OUTS[name] / f"analysis/{stem}.float16.npy"


def c471() -> None:
    audit = load(prior.OUTS["C470"] / "audit/independent_audit.json")
    begin("C471", {
        "status": "program_guard_master_contract_frozen", "parent": "C454-C470 independent audit",
        "corrections": ["C461 is a shared response baseline, not a universal learning law",
                        "C466 M3-seen rollout is not cross-family composition",
                        "C468 rejects one affine depth model, not all nonlinear path mechanisms"],
        "routes": ["complete program factorial", "observation atlas", "shared propagation", "family-blind program guard",
                   "coordinate state guard", "arbitrary full-coordinate coupling", "unseen composition", "conditional writer"],
        "route_policy": "a failed route does not stop other registered routes",
        "measurement": "embedding and HiddenState only; all 2560 coordinates; no Attention/MLP/weights/PCA/Top-K",
    }, {"parent": audit["all_checks_passed"], "continuity": PHASES["C471"][0] == 2005})
    close("C471", {"status": "contract_closed", "strict_interpretation": "The campaign tests whether external program structure or current coordinate state predicts family-specific increments after a shared baseline."},
          {"families": len(FAMILIES) == 8, "factors": len(FACTOR_NAMES) == 4}, "C472_material")


def c472() -> None:
    out = begin("C472", {
        "status": "four_factor_program_material_frozen", "families": list(FAMILIES), "factors": list(FACTOR_NAMES),
        "constructions": list(CONSTRUCTIONS), "units": 10, "cells_per_program": 16,
        "family_split": {"train": list(TRAIN_FAMILIES), "lockbox": list(FAMILY_LOCKBOX)},
        "unit_split": {"discovery": [0,1,2,3,4], "confirmation": [5,6,7], "lockbox": [8,9]},
        "codebook": "A Yes; B No; fixed across every cell",
    }, {"parent": final("C471")["all_checks_passed"]})
    rows = material()
    write_rows(out / "material/cases.jsonl", rows)
    write_rows(out / "material/program_graphs.jsonl", [{"case_id": r["case_id"], **r["semantic_graph"]} for r in rows])
    headline = {"status": "material_closed", "rows": len(rows), "programs": len(rows) // 16,
                "truth_frequency": float(np.mean([r["gold_position"] == 0 for r in rows])),
                "strict_interpretation": "The graph is an operational controlled-language compiler, not a complete linguistic theory."}
    close("C472", headline, {"rows": len(rows) == 3840, "balance": headline["truth_frequency"] == 0.5}, "C473_audit")


def c473() -> None:
    out = begin("C473", {"status": "compiler_zero_model_audit_frozen", "zero_models": ["always A", "always B", "family majority", "construction majority", "unit majority"], "naturalness": "controlled English machine audit; no independent human panel"}, {"parent": final("C472")["all_checks_passed"]})
    rows, _ = material_lookup()
    zero = {
        "always_a": float(np.mean([r["gold_position"] == 0 for r in rows])),
        "always_b": float(np.mean([r["gold_position"] == 1 for r in rows])),
    }
    for key in ("family", "construction", "unit"):
        zero[f"{key}_majority"] = float(np.mean([
            max(np.mean([x["gold_position"] == 0 for x in rows if x[key] == value]),
                np.mean([x["gold_position"] == 1 for x in rows if x[key] == value]))
            for value in sorted({r[key] for r in rows})
        ]))
    roles = all(all(str(value) in row["prompt_core"] for value in row["role_values"].values()) for row in rows)
    cell_complete = all(len([r for r in rows if r["family"] == f and r["construction"] == c and r["unit"] == u]) == 16 for f, c, u in itertools.product(FAMILIES, CONSTRUCTIONS, range(10)))
    malformed = ("says is bright", "says is not bright", "occurred before cedar", "caused cedar")
    machine_naturalness = all(not any(fragment in row["prompt_core"] for fragment in malformed) for row in rows)
    eligible = all(abs(value - 0.5) < 1e-12 for value in zero.values()) and roles and cell_complete and machine_naturalness
    headline = {"status": "material_audit_closed", "zero_model_accuracies": zero, "role_occurrence": roles,
                "complete_factorial": cell_complete, "machine_naturalness": machine_naturalness,
                "material_eligible": eligible, "human_naturalness_review": False,
                "strict_interpretation": "Query-match is the intended semantic determinant; exact balance does not prove naturalness or semantic uniqueness."}
    close("C473", headline, {"eligible": eligible}, "C474_behavior")


def c474() -> None:
    out = begin("C474", {"status": "qwen_program_behavior_frozen", "model": "Qwen3-4B BF16 CUDA", "gates": {"heldout": 0.75, "family": 0.60, "construction": 0.60, "minimum_families": 6}, "policy": "only qualified families enter fields"}, {"parent": final("C473")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows, by_id = material_lookup()
    tokenizer = prior.graph_base.axis_old.base.parent.fresh.tokenizer_qwen()
    compiled = prior.compile_base.compile_qwen(tokenizer, rows)
    write_rows(out / "compiled/qwen3.jsonl", compiled)
    run = prior.graph_base.axis_old.base.parent.previous.qwen_behavior(rows, compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl")
    held = [r for r in behavior if by_id[r["case_id"]]["partition"] != "discovery"]
    by_family = {f: float(np.mean([r["correct"] for r in held if by_id[r["case_id"]]["family"] == f])) for f in FAMILIES}
    by_construction = {c: float(np.mean([r["correct"] for r in held if by_id[r["case_id"]]["construction"] == c])) for c in CONSTRUCTIONS}
    eligible = [f for f in FAMILIES if by_family[f] >= 0.60]
    heldout = float(np.mean([r["correct"] for r in held]))
    authorized = heldout >= 0.75 and min(by_construction.values()) >= 0.60 and len(eligible) >= 6 and all(f in eligible for f in FAMILY_LOCKBOX)
    headline = {"status": "behavior_closed", **run, "heldout_accuracy": heldout, "family_accuracy": by_family,
                "construction_accuracy": by_construction, "eligible_families": eligible, "field_authorized": authorized,
                "strict_interpretation": "Behavior qualifies this fixed-codebook program interface only."}
    close("C474", headline, {"rows": len(behavior) == len(rows), "finite": finite(headline)}, "C475_field")


def c475() -> None:
    out = begin("C475", {"status": "program_graph_full_field_frozen", "role_field": "all eligible cells x 38 checkpoints x six roles x all coordinates", "full_token": "24 deterministic lockbox report cells", "no_compression": True}, {"parent": final("C474")["all_checks_passed"]})
    if not final("C474")["headline"]["field_authorized"]:
        close("C475", {"status": "field_not_run_behavior_ineligible", "field_ran": False}, {"route_accounted": True}, "C476_walsh")
        return
    rows, _ = material_lookup()
    eligible = set(final("C474")["headline"]["eligible_families"])
    compiled = {r["case_id"]: r for r in read_rows(OUTS["C474"] / "compiled/qwen3.jsonl")}
    selected = [r for r in rows if r["family"] in eligible]
    selected_compiled = [compiled[r["case_id"]] for r in selected]
    full_ids = set(r["case_id"] for r in sorted([x for x in selected if x["partition"] == "lockbox" and x["construction"] == "report" and sum(x["bits"]) >= 2], key=lambda x: x["case_id"])[:24])
    run = prior.common.batch_capture_qwen(selected, selected_compiled, out, full_selector=lambda row: row["case_id"] in full_ids, batch_size=8, field_width=FIELD_WIDTH)
    headline = {"status": "full_field_closed", **run, "field_ran": True, "eligible_families": sorted(eligible),
                "strict_interpretation": "The archive contains activations, not weights or semantic neurons."}
    close("C475", headline, {"shape": run["role_shape"][1:] == [38,6,2560], "full": run["full_token_rows"] == 24}, "C476_walsh")


def c476() -> None:
    out = begin("C476", {"status": "complete_factorial_walsh_ledger_frozen", "effects": list(MASKS), "checkpoints": list(QPOINTS), "qualification": "all 16 cells behavior-correct within a program", "coefficient": "signed 1/16 over every factorial cell"}, {"parent": final("C475")["all_checks_passed"]})
    if not final("C475")["headline"].get("field_ran"):
        close("C476", {"status": "walsh_not_run_no_field", "ran": False}, {"route_accounted": True}, "C477_observation")
        return
    rows, by_id = material_lookup()
    hidden_index = read_rows(OUTS["C475"] / "raw/hidden_index.jsonl")
    by_case = {r["case_id"]: r for r in hidden_index}
    groups = []
    for family, construction, unit in itertools.product(final("C475")["headline"]["eligible_families"], CONSTRUCTIONS, range(10)):
        cells = []
        for bits in BITS:
            case_id = f"c472-{family}-{construction}-u{unit}-x{''.join(map(str,bits))}"
            if case_id not in by_case or not by_case[case_id]["correct"]:
                cells = []
                break
            cells.append(by_case[case_id]["hidden_index"])
        if cells:
            groups.append({"family": family, "construction": construction, "unit": unit, "partition": partition(unit), "indices": cells})
    source = np.load(OUTS["C475"] / "raw/role_states.float16.npy", mmap_mode="r")
    n_effects = len(groups) * 15
    effects = np.lib.format.open_memmap(out / "raw/walsh_effects.float16.npy", mode="w+", dtype=np.float16, shape=(n_effects, len(QPOINTS), 6, DIM))
    weight_matrix = torch.tensor(np.stack([walsh_weights(mask) for mask in MASKS]), dtype=torch.float32, device="cuda")
    records = []
    cursor = 0
    for group_i, group in enumerate(groups):
        block = torch.tensor(np.asarray(source[group["indices"]][:, QPOINTS], dtype=np.float32), device="cuda")
        transformed = torch.matmul(weight_matrix, block.reshape(16, -1)).reshape(15, len(QPOINTS), 6, DIM).cpu().numpy()
        effects[cursor:cursor + 15] = transformed.astype(np.float16)
        for local, mask in enumerate(MASKS):
            records.append({"effect_index": cursor + local, "group": group_i, "family": group["family"], "construction": group["construction"], "unit": group["unit"], "partition": group["partition"], "effect_mask": mask, "effect_order": mask_order(mask), "factor_names": [FACTOR_NAMES[i] for i in range(4) if mask & (1 << i)]})
        cursor += 15
        effects.flush()
        if group_i % 20 == 0 or group_i + 1 == len(groups):
            print(f"[C476 Walsh] {group_i + 1}/{len(groups)}", flush=True)
    close_mmap(source); close_mmap(effects)
    write_rows(out / "analysis/effect_index.jsonl", records)
    headline = {"status": "walsh_ledger_closed", "ran": True, "complete_programs": len(groups), "effect_records": len(records),
                "effect_shape": [n_effects, len(QPOINTS), 6, DIM], "family_program_counts": {f: sum(g["family"] == f for g in groups) for f in FAMILIES},
                "strict_interpretation": "Walsh effects are controlled contrasts, not latent variables stored explicitly by the model."}
    close("C476", headline, {"effects": len(records) == len(groups) * 15, "train_families": all(any(g["family"] == f for g in groups) for f in TRAIN_FAMILIES), "lockbox_families": all(any(g["family"] == f for g in groups) for f in FAMILY_LOCKBOX)}, "C477_observation")


def c477() -> None:
    out = begin("C477", {"status": "observation_first_coordinate_atlas_frozen", "scope": "family x effect mask x registered checkpoint x role x every coordinate", "statistics": ["RMS", "positive fraction", "nonzero fraction", "centroid sign agreement"], "no_predictive_gate": True}, {"parent": final("C476")["all_checks_passed"]})
    if not final("C476")["headline"].get("ran"):
        close("C477", {"status": "observation_not_run", "ran": False}, {"route_accounted": True}, "C478_shared")
        return
    states = np.load(OUTS["C476"] / "raw/walsh_effects.float16.npy", mmap_mode="r")
    records = read_rows(OUTS["C476"] / "analysis/effect_index.jsonl")
    centroids = np.lib.format.open_memmap(out / "analysis/family_effect_centroids.float16.npy", mode="w+", dtype=np.float16, shape=(len(FAMILIES),15,len(QPOINTS),6,DIM))
    atlas = []
    for fi, family in enumerate(FAMILIES):
        for mask in MASKS:
            subset = [r for r in records if r["family"] == family and r["effect_mask"] == mask and r["unit"] < 5 and r["construction"] in CONSTRUCTIONS[:2]]
            indices = [r["effect_index"] for r in subset]
            values = np.asarray(states[indices], dtype=np.float32)
            mean = values.mean(0)
            centroids[fi, mask - 1] = mean.astype(np.float16)
            for qi, checkpoint in enumerate(QPOINTS):
                for role in range(6):
                    block = values[:, qi, role]
                    centroid = mean[qi, role]
                    agreement = float(np.mean(np.sign(block) == np.sign(centroid[None, :])))
                    atlas.append({"family": family, "effect_mask": mask, "effect_order": mask_order(mask), "checkpoint": checkpoint, "role": ROLES[role], "samples": len(indices), "rms": float(np.sqrt(np.mean(block * block))), "positive_fraction": float(np.mean(block > 0)), "nonzero_fraction": float(np.mean(block != 0)), "centroid_sign_agreement": agreement})
        centroids.flush()
    close_mmap(states); close_mmap(centroids)
    write_rows(out / "analysis/condition_atlas.jsonl", atlas)
    headline = {"status": "observation_atlas_closed", "ran": True, "rows": len(atlas), "mean_sign_agreement": float(np.mean([r["centroid_sign_agreement"] for r in atlas])), "mean_nonzero_fraction": float(np.mean([r["nonzero_fraction"] for r in atlas])), "strict_interpretation": "The atlas describes distributed effects without clustering or selecting coordinates."}
    close("C477", headline, {"rows": len(atlas) == len(FAMILIES) * 15 * len(QPOINTS) * 6, "finite": finite(headline)}, "C478_shared")


def c478() -> None:
    out = begin("C478", {"status": "shared_propagation_reconstruction_frozen", "train": "six families, units0-4, ledger/brief, effect order1-2", "evaluation": ["within-family", "whole-family lockbox", "unseen report", "order3-4 composition"], "edges": [[q,q+1] for q in Q_STARTS], "controls": ["identity", "training mean"]}, {"parent": final("C477")["all_checks_passed"]})
    states = np.load(OUTS["C476"] / "raw/walsh_effects.float16.npy", mmap_mode="r")
    records = read_rows(OUTS["C476"] / "analysis/effect_index.jsonl"); splits = effect_splits(records)
    slope = np.lib.format.open_memmap(out / "analysis/slope.float16.npy", mode="w+", dtype=np.float16, shape=(len(Q_STARTS),6,DIM)); intercept = np.lib.format.open_memmap(out / "analysis/intercept.float16.npy", mode="w+", dtype=np.float16, shape=(len(Q_STARTS),6,DIM)); means = np.lib.format.open_memmap(out / "analysis/mean.float16.npy", mode="w+", dtype=np.float16, shape=(len(Q_STARTS),6,DIM))
    acc = {split: {name: metric() for name in ("shared","identity","mean")} for split in splits if split != "train"}
    for edge in range(len(Q_STARTS)):
        for role in range(6):
            xt, yt = effect_arrays(states, splits["train"], edge * 2, role)
            a, b = fit_diagonal(xt, yt); slope[edge,role] = a.astype(np.float16); intercept[edge,role] = b.astype(np.float16); means[edge,role] = yt.mean(0).astype(np.float16)
            for split, rows in splits.items():
                if split == "train" or not rows: continue
                x, y = effect_arrays(states, rows, edge * 2, role)
                add_metric(acc[split]["shared"], a * x + b, y); add_metric(acc[split]["identity"], x, y); add_metric(acc[split]["mean"], np.broadcast_to(yt.mean(0), y.shape), y)
        slope.flush(); intercept.flush(); means.flush(); print(f"[C478] edge={Q_STARTS[edge]}", flush=True)
    for value in (states,slope,intercept,means): close_mmap(value)
    metrics = {split: {name: finish_metric(value) for name,value in models.items()} for split,models in acc.items()}; save(out / "analysis/metrics.json", metrics)
    candidate = all(metrics[s]["shared"]["nrmse"] < min(metrics[s]["identity"]["nrmse"], metrics[s]["mean"]["nrmse"]) for s in ("within","family","report"))
    headline = {"status": "shared_reconstruction_closed", "metrics": metrics, "shared_candidate": candidate, "train_records": len(splits["train"]), "split_records": {k:len(v) for k,v in splits.items()}, "strict_interpretation": "This is a family-blind diagonal response baseline on registered edges."}
    close("C478", headline, {"finite": finite(headline), "train": len(splits["train"]) >= 200}, "C479_program_guard")


def c479() -> None:
    out = begin("C479", {"status": "family_blind_program_guard_frozen", "model": "C478 plus mean residual indexed only by four-bit effect mask", "forbidden_inputs": ["family label", "lexical identity"], "unseen_composition": "order3-4 masks receive zero here", "pass_scope": "whole-family and report lockboxes"}, {"parent": final("C478")["all_checks_passed"]})
    states=np.load(OUTS["C476"] / "raw/walsh_effects.float16.npy",mmap_mode="r");records=read_rows(OUTS["C476"] / "analysis/effect_index.jsonl");splits=effect_splits(records);slope=np.load(OUTS["C478"] / "analysis/slope.float16.npy",mmap_mode="r");intercept=np.load(OUTS["C478"] / "analysis/intercept.float16.npy",mmap_mode="r")
    residual=np.lib.format.open_memmap(out/"analysis/mask_residual.float16.npy",mode="w+",dtype=np.float16,shape=(15,len(Q_STARTS),6,DIM));acc={split:metric() for split in ("within","family","report","composition")};baseline={split:metric() for split in acc};mask_acc={str(mask):{"program":metric(),"shared":metric()} for mask in MASKS if mask_order(mask)<=2}
    for edge in range(len(Q_STARTS)):
        for role in range(6):
            xt,yt=effect_arrays(states,splits["train"],edge*2,role);base=np.asarray(slope[edge,role],np.float32)*xt+np.asarray(intercept[edge,role],np.float32)
            for mask in MASKS:
                selected=[i for i,r in enumerate(splits["train"]) if r["effect_mask"]==mask]
                residual[mask-1,edge,role]=((yt[selected]-base[selected]).mean(0) if selected else np.zeros(DIM,np.float32)).astype(np.float16)
            for split,rows in splits.items():
                if split=="train" or not rows:continue
                x,y=effect_arrays(states,rows,edge*2,role);shared=np.asarray(slope[edge,role],np.float32)*x+np.asarray(intercept[edge,role],np.float32);pred=shared.copy()
                for i,row in enumerate(rows):
                    if row["effect_order"]<=2:pred[i]+=np.asarray(residual[row["effect_mask"]-1,edge,role],np.float32)
                add_metric(acc[split],pred,y);add_metric(baseline[split],shared,y)
                if split=="family":
                    for mask in mask_acc:
                        m=int(mask);idx=[i for i,r in enumerate(rows) if r["effect_mask"]==m]
                        if idx:add_metric(mask_acc[mask]["program"],pred[idx],y[idx]);add_metric(mask_acc[mask]["shared"],shared[idx],y[idx])
        residual.flush();print(f"[C479] edge={Q_STARTS[edge]}",flush=True)
    for value in (states,slope,intercept,residual):close_mmap(value)
    metrics={split:{"program":finish_metric(acc[split]),"shared":finish_metric(baseline[split])} for split in acc};mask_metrics={mask:{name:finish_metric(v) for name,v in models.items()} for mask,models in mask_acc.items()};save(out/"analysis/metrics.json",metrics);save(out/"analysis/family_mask_metrics.json",mask_metrics)
    headline={"status":"program_guard_closed","metrics":metrics,"strict_interpretation":"The mask guard uses external program structure but no family identity; improvement would be predictive, not causal."};close("C479",headline,{"finite":finite(headline)},"C480_state_guard")


def fit_state_bins(x:np.ndarray,y:np.ndarray,fallback_a:np.ndarray,fallback_b:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    threshold=np.median(np.abs(x),axis=0).astype(np.float32);a=np.empty((4,DIM),np.float32);b=np.empty((4,DIM),np.float32)
    bins=((x<0)&(np.abs(x)<=threshold),(x<0)&(np.abs(x)>threshold),(x>=0)&(np.abs(x)<=threshold),(x>=0)&(np.abs(x)>threshold))
    for bi,mask in enumerate(bins):
        count=mask.sum(0).astype(np.float64);sx=np.sum(np.where(mask,x,0),axis=0,dtype=np.float64);sy=np.sum(np.where(mask,y,0),axis=0,dtype=np.float64);sxx=np.sum(np.where(mask,x*x,0),axis=0,dtype=np.float64);sxy=np.sum(np.where(mask,x*y,0),axis=0,dtype=np.float64);den=sxx-sx*sx/np.maximum(count,1);num=sxy-sx*sy/np.maximum(count,1);aa=(num/(den+1e-6)).astype(np.float32);bb=((sy-aa*sx)/np.maximum(count,1)).astype(np.float32);valid=count>=4;a[bi]=np.where(valid,aa,fallback_a);b[bi]=np.where(valid,bb,fallback_b)
    return threshold,a,b


def state_predict(x:np.ndarray,threshold:np.ndarray,a:np.ndarray,b:np.ndarray)->np.ndarray:
    result=np.empty_like(x,dtype=np.float32);bins=((x<0)&(np.abs(x)<=threshold),(x<0)&(np.abs(x)>threshold),(x>=0)&(np.abs(x)<=threshold),(x>=0)&(np.abs(x)>threshold))
    for bi,mask in enumerate(bins):result[mask]=(a[bi][None,:]*x+b[bi][None,:])[mask]
    return result


def c480() -> None:
    out=begin("C480",{"status":"coordinate_state_guard_frozen","state_bins":["negative-low","negative-high","nonnegative-low","nonnegative-high"],"threshold":"per-coordinate training median absolute response","model":"piecewise diagonal state propagation plus family-blind effect-mask residual","no_coordinate_selection":True},{"parent":final("C479")["all_checks_passed"]})
    states=np.load(OUTS["C476"] / "raw/walsh_effects.float16.npy",mmap_mode="r");records=read_rows(OUTS["C476"] / "analysis/effect_index.jsonl");splits=effect_splits(records);shared_a=np.load(OUTS["C478"] / "analysis/slope.float16.npy",mmap_mode="r");shared_b=np.load(OUTS["C478"] / "analysis/intercept.float16.npy",mmap_mode="r")
    threshold=np.lib.format.open_memmap(out/"analysis/state_threshold.float16.npy",mode="w+",dtype=np.float16,shape=(len(Q_STARTS),6,DIM));slope=np.lib.format.open_memmap(out/"analysis/state_slope.float16.npy",mode="w+",dtype=np.float16,shape=(len(Q_STARTS),6,4,DIM));intercept=np.lib.format.open_memmap(out/"analysis/state_intercept.float16.npy",mode="w+",dtype=np.float16,shape=(len(Q_STARTS),6,4,DIM));maskres=np.lib.format.open_memmap(out/"analysis/state_mask_residual.float16.npy",mode="w+",dtype=np.float16,shape=(15,len(Q_STARTS),6,DIM));acc={split:{"state":metric(),"program":metric()} for split in ("within","family","report","composition")}
    program_res=np.load(OUTS["C479"] / "analysis/mask_residual.float16.npy",mmap_mode="r")
    for edge in range(len(Q_STARTS)):
        for role in range(6):
            xt,yt=effect_arrays(states,splits["train"],edge*2,role);t,a,b=fit_state_bins(xt,yt,np.asarray(shared_a[edge,role],np.float32),np.asarray(shared_b[edge,role],np.float32));threshold[edge,role]=t.astype(np.float16);slope[edge,role]=a.astype(np.float16);intercept[edge,role]=b.astype(np.float16);base=state_predict(xt,t,a,b)
            for mask in MASKS:
                idx=[i for i,r in enumerate(splits["train"]) if r["effect_mask"]==mask];maskres[mask-1,edge,role]=((yt[idx]-base[idx]).mean(0) if idx else np.zeros(DIM,np.float32)).astype(np.float16)
            for split,rows in splits.items():
                if split=="train" or not rows:continue
                x,y=effect_arrays(states,rows,edge*2,role);state=state_predict(x,t,a,b);program=np.asarray(shared_a[edge,role],np.float32)*x+np.asarray(shared_b[edge,role],np.float32)
                for i,row in enumerate(rows):
                    if row["effect_order"]<=2:state[i]+=np.asarray(maskres[row["effect_mask"]-1,edge,role],np.float32);program[i]+=np.asarray(program_res[row["effect_mask"]-1,edge,role],np.float32)
                add_metric(acc[split]["state"],state,y);add_metric(acc[split]["program"],program,y)
        for value in (threshold,slope,intercept,maskres):value.flush()
        print(f"[C480] edge={Q_STARTS[edge]}",flush=True)
    for value in (states,shared_a,shared_b,program_res,threshold,slope,intercept,maskres):close_mmap(value)
    metrics={split:{name:finish_metric(v) for name,v in models.items()} for split,models in acc.items()};save(out/"analysis/metrics.json",metrics);headline={"status":"state_guard_closed","metrics":metrics,"strict_interpretation":"State bins are deterministic coordinate conditions, not discovered semantic categories."};close("C480",headline,{"finite":finite(headline)},"C481_adjudication")


def c481() -> None:
    out=begin("C481",{"status":"cross_family_selector_gate_frozen","program_pass":{"family_gain":0.01,"report_gain":0.005,"mask_wins":6},"state_pass":{"family_gain":0.01,"report_gain":0.005},"selection":"state if state gate passes, else program if program gate passes, else none"},{"parent":final("C480")["all_checks_passed"]})
    shared=load(OUTS["C478"] / "analysis/metrics.json");program=load(OUTS["C479"] / "analysis/metrics.json");state=load(OUTS["C480"] / "analysis/metrics.json");mask=load(OUTS["C479"] / "analysis/family_mask_metrics.json")
    pg={s:shared[s]["shared"]["nrmse"]-program[s]["program"]["nrmse"] for s in ("family","report")};sg={s:program[s]["program"]["nrmse"]-state[s]["state"]["nrmse"] for s in ("family","report")};wins=sum(row["shared"]["nrmse"]-row["program"]["nrmse"]>0.005 for row in mask.values());program_pass=pg["family"]>=.01 and pg["report"]>=.005 and wins>=6;state_pass=sg["family"]>=.01 and sg["report"]>=.005;selected="state" if state_pass else "program" if program_pass else "none"
    headline={"status":"selector_adjudication_closed","program_gains":pg,"program_mask_wins":wins,"program_guard_candidate":program_pass,"state_gains_over_program":sg,"state_guard_candidate":state_pass,"selected_model":selected,"selector_candidate":selected!="none","strict_interpretation":"A candidate predicts held-out families from registered graph/state conditions; it is not yet causal or a universal language algebra."};save(out/"analysis/gate.json",headline);close("C481",headline,{"finite":finite(headline)},"C482_coupling")


def predict_registered(model:str,x:np.ndarray,edge:int,role:int,masks:list[int],shared_a,shared_b,program_res,state_t=None,state_a=None,state_b=None,state_res=None)->np.ndarray:
    if model=="state":base=state_predict(x,np.asarray(state_t[edge,role],np.float32),np.asarray(state_a[edge,role],np.float32),np.asarray(state_b[edge,role],np.float32));res=state_res
    else:base=np.asarray(shared_a[edge,role],np.float32)*x+np.asarray(shared_b[edge,role],np.float32);res=program_res
    for i,mask in enumerate(masks):
        if mask_order(mask)<=2:base[i]+=np.asarray(res[mask-1,edge,role],np.float32)
    return base


def c482() -> None:
    out=begin("C482",{"status":"arbitrary_full_coordinate_ridge_frozen","nodes":{"checkpoints":[16,24],"roles":list(ROLES)},"operator":"all 2560 source coordinates to all 2560 target coordinates via dual ridge","lambda":"0.1 times mean training Gram diagonal plus 1e-6","controls":["selected diagonal guard","shuffled targets","source coordinate roll257"],"pass":"ridge improves every control by 0.02 on family and report lockboxes","matrix_archive":"full 2560x2560 operator for q24 boundary"},{"parent":final("C481")["all_checks_passed"],"cuda":torch.cuda.is_available()})
    states=np.load(OUTS["C476"] / "raw/walsh_effects.float16.npy",mmap_mode="r");records=read_rows(OUTS["C476"] / "analysis/effect_index.jsonl");splits=effect_splits(records);shared_a=np.load(OUTS["C478"] / "analysis/slope.float16.npy",mmap_mode="r");shared_b=np.load(OUTS["C478"] / "analysis/intercept.float16.npy",mmap_mode="r");program_res=np.load(OUTS["C479"] / "analysis/mask_residual.float16.npy",mmap_mode="r");state_t=np.load(OUTS["C480"] / "analysis/state_threshold.float16.npy",mmap_mode="r");state_a=np.load(OUTS["C480"] / "analysis/state_slope.float16.npy",mmap_mode="r");state_b=np.load(OUTS["C480"] / "analysis/state_intercept.float16.npy",mmap_mode="r");state_res=np.load(OUTS["C480"] / "analysis/state_mask_residual.float16.npy",mmap_mode="r");selected=final("C481")["headline"]["selected_model"] if final("C481")["headline"]["selected_model"]!="none" else "state"
    train=splits["train"];evals={k:splits[k] for k in ("family","report")};ntrain=len(train);alpha_store=np.lib.format.open_memmap(out/"analysis/ridge_alpha.float16.npy",mode="w+",dtype=np.float16,shape=(2,6,ntrain,DIM));xmean_store=np.lib.format.open_memmap(out/"analysis/xmean.float16.npy",mode="w+",dtype=np.float16,shape=(2,6,DIM));ymean_store=np.lib.format.open_memmap(out/"analysis/ymean.float16.npy",mode="w+",dtype=np.float16,shape=(2,6,DIM));acc={split:{name:metric() for name in ("ridge","registered","shuffled","roll")} for split in evals};operator_path=out/"analysis/q24_boundary_full_operator.float16.npy"
    for local_edge,edge in enumerate((2,3)):
        for role in range(6):
            xtr,ytr=effect_arrays(states,train,edge*2,role);xm=xtr.mean(0);ym=ytr.mean(0);xc=xtr-xm;yc=ytr-ym;xt=torch.tensor(xc,device="cuda");yt=torch.tensor(yc,device="cuda");gram=xt@xt.T/DIM;lam=.1*float(torch.diagonal(gram).mean())+1e-6;system=gram+torch.eye(ntrain,device="cuda")*lam;rhs=torch.cat([yt,torch.flip(yt,[0])],dim=1);solutions=torch.linalg.solve(system,rhs);alpha=solutions[:,:DIM];alpha_shuf=solutions[:,DIM:];alpha_store[local_edge,role]=alpha.cpu().numpy().astype(np.float16);xmean_store[local_edge,role]=xm.astype(np.float16);ymean_store[local_edge,role]=ym.astype(np.float16)
            if edge==3 and role==5:
                operator=(xt.T@alpha/DIM).cpu().numpy().astype(np.float16);np.save(operator_path,operator);del operator
            for split,rows in evals.items():
                x,y=effect_arrays(states,rows,edge*2,role);xg=torch.tensor(x-xm,device="cuda");ridge=(xg@xt.T/DIM@alpha).cpu().numpy()+ym;shuffled=(xg@xt.T/DIM@alpha_shuf).cpu().numpy()+ym;rolled=(xg@torch.roll(xt,257,dims=1).T/DIM@alpha).cpu().numpy()+ym;registered=predict_registered(selected,x,edge,role,[r["effect_mask"] for r in rows],shared_a,shared_b,program_res,state_t,state_a,state_b,state_res);add_metric(acc[split]["ridge"],ridge,y);add_metric(acc[split]["registered"],registered,y);add_metric(acc[split]["shuffled"],shuffled,y);add_metric(acc[split]["roll"],rolled,y)
            del xt,yt,gram,system,rhs,solutions,alpha,alpha_shuf;torch.cuda.empty_cache()
        alpha_store.flush();xmean_store.flush();ymean_store.flush();print(f"[C482] edge={Q_STARTS[edge]}",flush=True)
    for value in (states,shared_a,shared_b,program_res,state_t,state_a,state_b,state_res,alpha_store,xmean_store,ymean_store):close_mmap(value)
    metrics={split:{name:finish_metric(v) for name,v in models.items()} for split,models in acc.items()};save(out/"analysis/metrics.json",metrics);gains={split:{control:metrics[split][control]["nrmse"]-metrics[split]["ridge"]["nrmse"] for control in ("registered","shuffled","roll")} for split in metrics};candidate=all(min(gains[s].values())>=.02 for s in ("family","report"));matrix=np.load(operator_path,mmap_mode="r");passport={"shape":list(matrix.shape),"positive_fraction":float(np.mean(matrix>0)),"negative_fraction":float(np.mean(matrix<0)),"nonzero_fraction":float(np.mean(matrix!=0)),"rms":float(np.sqrt(np.mean(np.asarray(matrix,np.float32)**2))),"magnitude_bins":{"zero":float(np.mean(matrix==0)),"le_1e-4":float(np.mean((np.abs(matrix)>0)&(np.abs(matrix)<=1e-4))),"1e-4_to_1e-3":float(np.mean((np.abs(matrix)>1e-4)&(np.abs(matrix)<=1e-3))),"gt_1e-3":float(np.mean(np.abs(matrix)>1e-3))}};close_mmap(matrix);save(out/"analysis/operator_passport.json",passport);headline={"status":"full_coordinate_coupling_closed","metrics":metrics,"gains":gains,"full_coordinate_coupling_candidate":candidate,"operator_passport":passport,"strict_interpretation":"The full ridge operator is a predictive dependency map; underdetermination prevents unique-edge or causal interpretation."};close("C482",headline,{"matrix":passport["shape"]==[2560,2560],"finite":finite(headline)},"C483_composition_writer")


@torch.inference_mode()
def score_patch(model,device,row:dict,layer:int|None,positions:list[int],delta:np.ndarray|None)->list[float]:
    hook=None
    if layer is not None and delta is not None:
        value=torch.tensor(delta,dtype=torch.float32,device=device)
        def patch(_module,_args,output):
            state=output[0] if isinstance(output,tuple) else output;changed=state.clone();changed[0,positions]+=value.to(changed.dtype);return (changed,*output[1:]) if isinstance(output,tuple) else changed
        hook=model.model.layers[layer].register_forward_hook(patch)
    try:
        ids=torch.tensor([row["prompt_ids"]],dtype=torch.long,device=device);output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True);return [float(output.logits[0,-1,c[0]]) for c in row["candidate_ids"]]
    finally:
        if hook:hook.remove()


def c483() -> None:
    out=begin("C483",{"status":"unseen_composition_and_conditional_writer_frozen","composition":"order3-4 residual predicted as sum of all contained singleton and pair program residuals","composition_pass":"gain over state-only >=0.01 and at least3/5 masks positive","writer_qualification":"C481 selector candidate","writer_node":{"checkpoint":24,"role":"boundary","factor":"query_match"},"writer_controls":["actual","shared","wrong mask","coordinate roll257","wrong role","wrong checkpoint"]},{"parent":final("C482")["all_checks_passed"]})
    states=np.load(OUTS["C476"] / "raw/walsh_effects.float16.npy",mmap_mode="r");records=read_rows(OUTS["C476"] / "analysis/effect_index.jsonl");splits=effect_splits(records);shared_a=np.load(OUTS["C478"] / "analysis/slope.float16.npy",mmap_mode="r");shared_b=np.load(OUTS["C478"] / "analysis/intercept.float16.npy",mmap_mode="r");program_res=np.load(OUTS["C479"] / "analysis/mask_residual.float16.npy",mmap_mode="r");state_t=np.load(OUTS["C480"] / "analysis/state_threshold.float16.npy",mmap_mode="r");state_a=np.load(OUTS["C480"] / "analysis/state_slope.float16.npy",mmap_mode="r");state_b=np.load(OUTS["C480"] / "analysis/state_intercept.float16.npy",mmap_mode="r");state_res=np.load(OUTS["C480"] / "analysis/state_mask_residual.float16.npy",mmap_mode="r")
    acc={name:metric() for name in ("composition","state","shared")};mask_acc={str(mask):{name:metric() for name in ("composition","state")} for mask in MASKS if mask_order(mask)>=3};family_acc={family:{name:metric() for name in ("composition","state")} for family in ("attitude_event","type_graph")};rows=splits["composition"]
    for edge in range(len(Q_STARTS)):
        for role in range(6):
            x,y=effect_arrays(states,rows,edge*2,role);shared=np.asarray(shared_a[edge,role],np.float32)*x+np.asarray(shared_b[edge,role],np.float32);state=state_predict(x,np.asarray(state_t[edge,role],np.float32),np.asarray(state_a[edge,role],np.float32),np.asarray(state_b[edge,role],np.float32));composition=state.copy()
            for i,row in enumerate(rows):
                mask=row["effect_mask"];components=[sub for sub in MASKS if mask_order(sub)<=2 and (sub&mask)==sub]
                composition[i]+=sum(np.asarray(program_res[sub-1,edge,role],np.float32) for sub in components)
            add_metric(acc["composition"],composition,y);add_metric(acc["state"],state,y);add_metric(acc["shared"],shared,y)
            for mask in mask_acc:
                m=int(mask);idx=[i for i,r in enumerate(rows) if r["effect_mask"]==m];add_metric(mask_acc[mask]["composition"],composition[idx],y[idx]);add_metric(mask_acc[mask]["state"],state[idx],y[idx])
            for family in family_acc:
                idx=[i for i,r in enumerate(rows) if r["family"]==family]
                if idx:add_metric(family_acc[family]["composition"],composition[idx],y[idx]);add_metric(family_acc[family]["state"],state[idx],y[idx])
    metrics={name:finish_metric(v) for name,v in acc.items()};mask_metrics={mask:{name:finish_metric(v) for name,v in models.items()} for mask,models in mask_acc.items()};family_metrics={family:{name:finish_metric(v) for name,v in models.items()} for family,models in family_acc.items()};wins=sum(row["state"]["nrmse"]-row["composition"]["nrmse"]>0.005 for row in mask_metrics.values());gain=metrics["state"]["nrmse"]-metrics["composition"]["nrmse"];composition_candidate=gain>=.01 and wins>=3;save(out/"analysis/composition_metrics.json",{"metrics":metrics,"mask_metrics":mask_metrics,"family_metrics":family_metrics})
    writer_ran=False;specificity=False;writer={"status":"not_run_selector_ineligible"}
    if final("C481")["headline"]["selector_candidate"]:
        writer_ran=True;selected=final("C481")["headline"]["selected_model"];material_rows,by_id=material_lookup();hidden=read_rows(OUTS["C475"] / "raw/hidden_index.jsonl");hcase={r["case_id"]:r for r in hidden};compiled={r["case_id"]:r for r in read_rows(OUTS["C474"] / "compiled/qwen3.jsonl")};raw=np.load(OUTS["C475"] / "raw/role_states.float16.npy",mmap_mode="r");edge=3;role=5;trials=[];model=None
        try:
            model,_tok,device,_placement=prior.model_base.load_bf16("qwen3")
            for family,unit,b0,b1,b2 in itertools.product(FAMILY_LOCKBOX,(8,9),(0,1),(0,1),(0,1)):
                left_id=f"c472-{family}-report-u{unit}-x{b0}{b1}{b2}0";right_id=f"c472-{family}-report-u{unit}-x{b0}{b1}{b2}1"
                if left_id not in hcase or right_id not in hcase:continue
                left_i,right_i=hcase[left_id]["hidden_index"],hcase[right_id]["hidden_index"];x=np.asarray(raw[right_i,24,role],np.float32)-np.asarray(raw[left_i,24,role],np.float32);actual=np.asarray(raw[right_i,25,role],np.float32)-np.asarray(raw[left_i,25,role],np.float32);pred=predict_registered(selected,x[None,:],edge,role,[8],shared_a,shared_b,program_res,state_t,state_a,state_b,state_res)[0];shared=np.asarray(shared_a[edge,role],np.float32)*x+np.asarray(shared_b[edge,role],np.float32);wrong=shared+np.asarray(program_res[0,edge,role],np.float32);left,right=compiled[left_id],compiled[right_id];positions=left["role_positions"]["boundary"]
                conditions={"base":(None,None,positions),"target":(None,None,right["role_positions"]["boundary"]),"predicted":(24,pred,positions),"actual":(24,actual,positions),"shared":(24,shared,positions),"wrong_mask":(24,wrong,positions),"roll":(24,np.roll(pred,257),positions),"wrong_role":(24,pred,left["role_positions"]["primary"]),"wrong_checkpoint":(25,pred,positions)};scores={name:score_patch(model,device,right if name=="target" else left,layer,pos,delta) for name,(layer,delta,pos) in conditions.items()};base_margin=scores["base"][1]-scores["base"][0];shifts={name:(value[1]-value[0])-base_margin for name,value in scores.items() if name not in ("base","target")};trials.append({"family":family,"unit":unit,"bits":[b0,b1,b2],"shifts":shifts,"target_choice":int(np.argmax(scores["predicted"]))==right["gold_position"]})
        finally:prior.model_base.release_bf16(model)
        close_mmap(raw);write_rows(out/"raw/writer_trials.jsonl",trials);med={name:float(np.median([r["shifts"][name] for r in trials])) for name in ("predicted","actual","shared","wrong_mask","roll","wrong_role","wrong_checkpoint")};rate=float(np.mean([r["target_choice"] for r in trials])) if trials else 0.0;specificity=bool(trials) and med["predicted"]>0 and med["predicted"]>max(med[k] for k in ("shared","wrong_mask","roll","wrong_role","wrong_checkpoint")) and rate>=.60;writer={"status":"writer_closed","selected_model":selected,"trials":len(trials),"median_shifts":med,"target_choice_rate":rate,"specificity_passed":specificity}
    for value in (states,shared_a,shared_b,program_res,state_t,state_a,state_b,state_res):close_mmap(value)
    headline={"status":"composition_writer_closed","composition_metrics":metrics,"composition_gain":gain,"composition_mask_wins":wins,"composition_candidate":composition_candidate,"family_panels":family_metrics,"writer_ran":writer_ran,"writer":writer,"specificity_passed":specificity,"strict_interpretation":"Composition is a frozen residual formula; writer results, if run, test narrow sufficiency rather than necessity or uniqueness."};close("C483",headline,{"finite":finite(headline)},"C484_synthesis")


def hash_remove(paths:list[Path],out:Path)->list[dict]:
    rows=[]
    for path in paths:
        if not path.exists():continue
        h=hashlib.sha256();size=path.stat().st_size
        with path.open("rb") as handle:
            while chunk:=handle.read(8*1024*1024):h.update(chunk)
        rows.append({"path":str(path.relative_to(ROOT)),"sha256":h.hexdigest(),"bytes":size,"deleted":True});path.unlink()
    save(out/"audit/cleanup.json",rows);return rows


def c484() -> None:
    out=begin("C484",{"status":"campaign_synthesis_visual_cleanup_frozen","visual":"deterministic family/effect slices and deterministic full-operator target rows, each row preserves all 2560 coordinates","operator_rows":"every twentieth target coordinate from q24 boundary; never amplitude-ranked","cleanup":"hash and delete nonvisual raw fields and fitted arrays after all routes close","new_math_gate":"requires selector, arbitrary coupling, unseen composition, causal writer, and cross-model composition; cross-model clause absent here"},{"parent":final("C483")["all_checks_passed"]})
    visual=[];centroids=np.load(OUTS["C477"] / "analysis/family_effect_centroids.float16.npy",mmap_mode="r")
    for fi,family in enumerate(FAMILIES):
        for mask,checkpoint,role in itertools.product(MASKS,(0,16,24,32),(0,5)):
            qi=QPOINTS.index(checkpoint);visual.append({"id":f"effect:{family}:m{mask}:q{checkpoint}:{ROLES[role]}","source":"program_effect_centroid","family":family,"effect_mask":mask,"effect_order":mask_order(mask),"checkpoint":checkpoint,"role":ROLES[role],"values":np.asarray(centroids[fi,mask-1,qi,role],np.float32).round(6).tolist()})
    close_mmap(centroids);operator=np.load(OUTS["C482"] / "analysis/q24_boundary_full_operator.float16.npy",mmap_mode="r")
    for target in range(0,DIM,20):visual.append({"id":f"operator:q24:boundary:target{target}","source":"full_coordinate_operator_row","checkpoint":24,"role":"boundary","target_coordinate":target,"values":np.asarray(operator[:,target],np.float32).round(7).tolist()})
    close_mmap(operator);payload={"schema":"c484.program-guard-hypergraph.v1","phase":2018,"campaign":"C471-C484","dimensions":list(range(DIM)),"rows":visual,"summary":{"behavior":final("C474")["headline"],"observation":final("C477")["headline"],"shared":final("C478")["headline"],"selector":final("C481")["headline"],"coupling":final("C482")["headline"],"composition_writer":final("C483")["headline"]},"claim_boundary":"Rows are full-coordinate activation responses or predictive ridge coefficients, not model weights, semantic neurons, or unique causal edges."};save(VISUAL,payload)
    cleanup_paths=[OUTS["C475"]/"raw/role_states.float16.npy",OUTS["C475"]/"raw/full_fields_holdout.float16.npy",OUTS["C476"]/"raw/walsh_effects.float16.npy",OUTS["C477"]/"analysis/family_effect_centroids.float16.npy",OUTS["C478"]/"analysis/slope.float16.npy",OUTS["C478"]/"analysis/intercept.float16.npy",OUTS["C478"]/"analysis/mean.float16.npy",OUTS["C479"]/"analysis/mask_residual.float16.npy",OUTS["C480"]/"analysis/state_threshold.float16.npy",OUTS["C480"]/"analysis/state_slope.float16.npy",OUTS["C480"]/"analysis/state_intercept.float16.npy",OUTS["C480"]/"analysis/state_mask_residual.float16.npy",OUTS["C482"]/"analysis/ridge_alpha.float16.npy",OUTS["C482"]/"analysis/xmean.float16.npy",OUTS["C482"]/"analysis/ymean.float16.npy",OUTS["C482"]/"analysis/q24_boundary_full_operator.float16.npy"]
    cleanup=hash_remove(cleanup_paths,out);selector=final("C481")["headline"]["selector_candidate"];coupling=final("C482")["headline"]["full_coordinate_coupling_candidate"];composition=final("C483")["headline"]["composition_candidate"];causal=final("C483")["headline"]["specificity_passed"];cross_model=False;new_math=selector and coupling and composition and causal and cross_model;next_same=selector and not causal
    headline={"status":"program_guard_campaign_closed","gates":{"shared":final("C478")["headline"]["shared_candidate"],"program_guard":final("C481")["headline"]["program_guard_candidate"],"state_guard":final("C481")["headline"]["state_guard_candidate"],"full_coordinate_coupling":coupling,"unseen_composition":composition,"natural_writer":causal,"cross_model_composition":cross_model},"visual_rows":len(visual),"visual_path":str(VISUAL.relative_to(ROOT)),"cleanup_files":len(cleanup),"cleanup_bytes":sum(r["bytes"] for r in cleanup),"new_math_gate_passed":new_math,"next_stage_same_goal":next_same,"strict_interpretation":"The campaign tests program and state guards without claiming a universal algebra or unique circuit."};close("C484",headline,{"visual":len(visual)>0,"cleanup":all(r["deleted"] for r in cleanup),"finite":finite(headline)},"independent_audit_then_registered_next_stage")


RUNNERS={f"C{i}":globals()[f"c{i}"] for i in range(471,485)}


def validate_only()->None:
    rows=material();checks={"phase_sequence":[PHASES[f"C{i}"][0] for i in range(471,485)]==list(range(2005,2019)),"rows":len(rows)==3840,"families":len({r["family"] for r in rows})==8,"programs":len(rows)//16==240,"balance":sum(r["gold_position"]==0 for r in rows)==len(rows)//2,"roles":all(all(str(v) in r["prompt_core"] for v in r["role_values"].values()) for r in rows),"masks":all(abs(float(np.sum(walsh_weights(m))))<1e-8 for m in MASKS)};print(json.dumps(checks));assert all(checks.values())


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("campaign",nargs="?",choices=list(PHASES));parser.add_argument("--validate-only",action="store_true");args=parser.parse_args()
    if args.validate_only:validate_only();return
    for name in ([args.campaign] if args.campaign else list(PHASES)):RUNNERS[name]()


if __name__=="__main__":main()
