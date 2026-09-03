#!/usr/bin/env python3
"""Shared frozen objects for the C310-C335 dual-axis response campaign.

The campaign only observes embeddings and hidden states. It does not read
attention maps, MLP activations, or model weights. Every observational archive
keeps the complete physical activation axis of the model under test.
"""
from __future__ import annotations

import gc
import itertools
import json
import math
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase1827_c293_c309_conditional_hypergraph_common as old
import phase1830_c296_complete_three_state_transition as old_transition
import phase1836_c302_six_family_composition_forecast as old_composition
import phase1748_c214_cross_model_functional_isomorphism as interface_base
from model_utils import get_layers, get_model_info

core = old.core
model_base = old.model_base
graph_base = old.previous.graph_base

DIM = old.DIM
WIDTH = old.WIDTH
FAMILIES = old.FAMILIES
ROLES = old.ROLES
MODELS = old.MODELS
CANONICAL_INDICES = old.CANONICAL_NEW_INDICES
CANONICAL_CHECKPOINTS = old.CANONICAL_CHECKPOINTS

PHASES = {
    "C310": (1844, "dual_axis_master_contract"),
    "C311": (1845, "interaction_residual_specificity"),
    "C312": (1846, "atomic_operator_transport"),
    "C313": (1847, "dual_axis_square_prediction"),
    "C314": (1848, "full_coordinate_operator_atlas"),
    "C315": (1849, "interaction_residual_causal_test"),
    "C316": (1850, "amplitude_phase_coupling"),
    "C317": (1851, "multisource_response_grammar"),
    "C318": (1852, "distributed_intervention_contract"),
    "C319": (1853, "distributed_width_dose_causal_test"),
    "C320": (1854, "residual_causal_stage_audit"),
    "C321": (1855, "five_surface_semantic_contract"),
    "C322": (1856, "qwen_five_surface_behavior"),
    "C323": (1857, "qwen_five_surface_full_coordinate_capture"),
    "C324": (1858, "natural_surface_composition_lockbox"),
    "C325": (1859, "natural_surface_stage_audit"),
    "C326": (1860, "cross_model_natural_panel_contract"),
    "C327": (1861, "qwen_cross_model_panel"),
    "C328": (1862, "glm_cross_model_panel"),
    "C329": (1863, "deepseek_cross_model_panel"),
    "C330": (1864, "cross_model_response_bisimulation"),
    "C331": (1865, "four_level_graph_contract"),
    "C332": (1866, "qwen_four_level_graph_full_field"),
    "C333": (1867, "graph_depth_operator_atlas"),
    "C334": (1868, "renamed_graph_lockbox"),
    "C335": (1869, "dual_axis_campaign_synthesis"),
}

OUTS = {
    campaign: RESULT / f"phase{phase}_{campaign.lower()}_{slug}"
    for campaign, (phase, slug) in PHASES.items()
}

TRAIN_ROLE_SOURCES = (
    (
        old_transition.C265 / "raw/training_role_states.float16.npy",
        old_transition.C248 / "raw/hidden_index.jsonl",
        "third",
    ),
    (
        old_transition.C264 / "raw/role_states.float16.npy",
        old_transition.C264 / "raw/hidden_index.jsonl",
        "fourth",
    ),
    (
        old_transition.C278 / "raw/role_states.float16.npy",
        old_transition.C278 / "raw/hidden_index.jsonl",
        "fifth",
    ),
)
SIXTH_STATES = old.OUTS["C295"] / "raw/role_states.float16.npy"
SIXTH_FIELDS = old.OUTS["C295"] / "raw/full_fields.float16.npy"
SIXTH_INDEX = old.OUTS["C295"] / "raw/hidden_index.jsonl"
SIXTH_COMPILED = old.OUTS["C294"] / "compiled/qwen3.jsonl"

NATURAL_SURFACES = (
    "report",
    "briefing",
    "notes",
    "archive",
    "witness",
)

INTERFACES = ("strict_chat", "demonstrated_chat", "plain")
BATCH = {"qwen3": 8, "glm4": 1, "deepseek7b": 1}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare(campaign: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[campaign]
    if out.exists():
        raise RuntimeError(out)
    if not all(checks.values()):
        raise RuntimeError(checks)
    for sub in ("analysis", "audit", "compiled", "material", "protocol", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": PHASES[campaign][0],
        "campaign": campaign,
        "created_at_utc": utc_now(),
        "producer_sha256": core.sha(Path(sys.modules["__main__"].__file__)),
        **protocol,
    }
    core.save(out / "protocol/preregistration.json", payload)
    core.save(out / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    return out


def close(campaign: str, headline: dict, audit_checks: dict, next_authorization: str) -> dict:
    out = OUTS[campaign]
    core.save(out / "analysis/summary.json", headline)
    core.save(out / "audit/internal_analysis_audit.json", {
        "checks": audit_checks,
        "all_checks_passed": all(audit_checks.values()),
    })
    protocol = core.load(out / "protocol/preregistration.json")
    final_checks = {
        "contract": core.load(out / "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": all(audit_checks.values()),
        "producer_hash": core.sha(Path(sys.modules["__main__"].__file__)) == protocol["producer_sha256"],
    }
    final = {
        "phase": PHASES[campaign][0],
        "campaign": campaign,
        "status": "closed",
        "checks": final_checks,
        "all_checks_passed": all(final_checks.values()),
        "headline": headline,
        "next_authorization": next_authorization,
    }
    core.save(out / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return final


def canonical(states: np.ndarray) -> np.ndarray:
    return np.asarray(states[:, CANONICAL_INDICES], np.float32) if states.shape[1] == 38 else np.asarray(states, np.float32)


def grouped_cells(index: list[dict], family: str, require_correct: bool = True) -> list[dict]:
    panel = "nested_composition" if family == "nested_attitude" else "core"
    rows = [
        row for row in index
        if row["panel"] == panel
        and row["family"] == family
        and (not require_correct or row.get("correct", row.get("behavior_correct", True)))
    ]
    lookup = {
        (row["surface"], row["unit"], row["order"], row["factor_a"], row["factor_b"]): row["hidden_index"]
        for row in rows
    }
    result = []
    for surface, unit, order in itertools.product(
        sorted({row["surface"] for row in rows}),
        sorted({row["unit"] for row in rows}),
        (1, -1),
    ):
        ids = {(a, b): lookup.get((surface, unit, order, a, b)) for a, b in itertools.product((0, 1), repeat=2)}
        if all(value is not None for value in ids.values()):
            result.append({"surface": surface, "unit": unit, "order": order, "ids": ids})
    return result


def factorial_arrays(states: np.ndarray, index: list[dict], family: str) -> tuple[dict[str, np.ndarray], list[dict]]:
    groups = grouped_cells(index, family)
    cells = {name: [] for name in ("h00", "h10", "h01", "h11", "a0", "a1", "b0", "b1", "interaction")}
    for group in groups:
        h = {
            key: (
                np.asarray(states[value, CANONICAL_INDICES], np.float32)
                if states.shape[1] == 38
                else np.asarray(states[value], np.float32)
            )
            for key, value in group["ids"].items()
        }
        cells["h00"].append(h[(0, 0)])
        cells["h10"].append(h[(1, 0)])
        cells["h01"].append(h[(0, 1)])
        cells["h11"].append(h[(1, 1)])
        cells["a0"].append(h[(1, 0)] - h[(0, 0)])
        cells["a1"].append(h[(1, 1)] - h[(0, 1)])
        cells["b0"].append(h[(0, 1)] - h[(0, 0)])
        cells["b1"].append(h[(1, 1)] - h[(1, 0)])
        cells["interaction"].append(h[(1, 1)] - h[(1, 0)] - h[(0, 1)] + h[(0, 0)])
    return {name: np.asarray(rows, np.float32) for name, rows in cells.items()}, groups


def load_training_factorials() -> dict[str, list[tuple[dict[str, np.ndarray], list[dict], str]]]:
    result = {family: [] for family in FAMILIES}
    for state_path, index_path, material_name in TRAIN_ROLE_SOURCES:
        states = np.load(state_path, mmap_mode="r")
        index = core.rows(index_path)
        for family in FAMILIES:
            arrays, groups = factorial_arrays(states, index, family)
            result[family].append((arrays, groups, material_name))
    return result


def training_mean_interactions() -> dict[str, np.ndarray]:
    training = load_training_factorials()
    return {
        family: np.concatenate([entry[0]["interaction"] for entry in training[family]], axis=0).mean(axis=0)
        for family in FAMILIES
    }


def surface_slot(groups: list[dict]) -> np.ndarray:
    names = sorted({group["surface"] for group in groups})
    mapping = {name: i for i, name in enumerate(names)}
    return np.asarray([mapping[group["surface"]] for group in groups], np.int8)


def relative_gain(baseline: np.ndarray, candidate: np.ndarray) -> float:
    base = float(np.mean(np.abs(baseline)))
    error = float(np.mean(np.abs(candidate)))
    return float((base - error) / max(base, 1e-12))


def norm_match(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    cn = float(np.sqrt(np.mean(np.square(candidate, dtype=np.float64))))
    rn = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
    return candidate * (rn / max(cn, 1e-12))


def role_position_vectors(row: dict, role_vectors: np.ndarray) -> dict[int, np.ndarray]:
    by_position: dict[int, list[np.ndarray]] = {}
    for role_i, role in enumerate(ROLES):
        for position in row["role_positions"][role]:
            by_position.setdefault(int(position), []).append(np.asarray(role_vectors[role_i], np.float32))
    return {position: np.mean(vectors, axis=0) for position, vectors in by_position.items()}


def score_prompt_candidates(model, ids: list[int], candidates: list[list[int]], device, pad: int) -> np.ndarray:
    expanded = [(ci, ids + candidate, len(ids), candidate) for ci, candidate in enumerate(candidates)]
    width = max(len(item[1]) for item in expanded)
    input_ids = torch.full((len(expanded), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(input_ids)
    for i, (_ci, values, _prefix, _candidate) in enumerate(expanded):
        input_ids[i, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[i, :len(values)] = 1
    output = model(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True)
    logp = torch.log_softmax(output.logits.float(), dim=-1)
    scores = np.zeros(len(candidates), np.float32)
    for i, (ci, _values, prefix, candidate) in enumerate(expanded):
        scores[ci] = sum(float(logp[i, prefix + offset - 1, token]) for offset, token in enumerate(candidate))
    return scores


def render_interface(tokenizer, row: dict, interface: str) -> tuple[list[int], list[list[int]]]:
    return interface_base.compile_interface(tokenizer, row, interface)


def score_interface_batch(model, device, pad: int, compiled: list[tuple[dict, list[int], list[list[int]]]]) -> list[dict]:
    expanded = []
    for row, ids, candidates in compiled:
        for ci, candidate in enumerate(candidates):
            expanded.append((row, ci, ids + candidate, len(ids), candidate))
    width = max(len(item[2]) for item in expanded)
    input_ids = torch.full((len(expanded), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(input_ids)
    for i, (_row, _ci, values, _prefix, _candidate) in enumerate(expanded):
        input_ids[i, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[i, :len(values)] = 1
    output = model(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True)
    logp = torch.log_softmax(output.logits.float(), dim=-1)
    scores = np.zeros((len(compiled), 2), np.float32)
    for i, (_row, ci, _values, prefix, candidate) in enumerate(expanded):
        scores[i // 2, ci] = sum(float(logp[i, prefix + offset - 1, token]) for offset, token in enumerate(candidate))
    result = []
    for i, (row, _ids, _candidates) in enumerate(compiled):
        prediction = int(scores[i, 1] > scores[i, 0])
        result.append({
            "case_id": row["case_id"],
            "family": row.get("family", "type_graph"),
            "surface": row["surface"],
            "unit": row["unit"],
            "factor_a": row.get("factor_a"),
            "factor_b": row.get("factor_b"),
            "depth": row.get("depth"),
            "shortcut": row.get("shortcut"),
            "partition": row.get("partition"),
            "gold_position": row["gold_position"],
            "prediction": prediction,
            "correct": prediction == row["gold_position"],
            "score0": float(scores[i, 0]),
            "score1": float(scores[i, 1]),
        })
    return result


def extract_dossier_parts(prompt_core: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"A dossier states: (.*) A separate entry states: (.*) Question: (.*)", prompt_core)
    if not match:
        raise ValueError(prompt_core)
    return match.group(1), match.group(2), match.group(3)


def natural_wrap(surface: str, fact1: str, fact2: str, question: str) -> str:
    templates = {
        "report": f"A report records the following. {fact1} It also records that {fact2} Based only on the report, {question}",
        "briefing": f"In a briefing, the first statement was: {fact1} The second was: {fact2} Please decide: {question}",
        "notes": f"Notes: {fact1} Separately, {fact2} Answer this: {question}",
        "archive": f"An archive contains two entries. Entry one: {fact1} Entry two: {fact2} Query: {question}",
        "witness": f"A witness stated, '{fact1}' The same witness added, '{fact2}' Now answer: {question}",
    }
    return templates[surface]


def natural_material() -> list[dict]:
    base = old.previous.previous
    rows = []
    saved_units = base.UNITS
    try:
        base.UNITS = old.UNITS
        for family, surface, unit, a, b, order in itertools.product(FAMILIES, NATURAL_SURFACES, range(8), (0, 1), (0, 1), (1, -1)):
            if family == "nested_attitude":
                case = base.nested_case("dossier", unit, a, b)
                panel = "nested_composition"
            else:
                case = base.semantic_case(family, "dossier", unit, a, b)
                panel = "core"
            fact1, fact2, question = extract_dossier_parts(case["prompt_core"])
            prompt_core = natural_wrap(surface, fact1, fact2, question)
            choices, gold = base.options(case["correct"], case["wrong"], order)
            rows.append({
                "case_id": f"c321-{family}-{surface}-u{unit}-{a}{b}-{order:+d}",
                "panel": panel,
                "family": family,
                "surface": surface,
                "unit": unit,
                "factor_a": a,
                "factor_b": b,
                "order": order,
                "partition": "discovery" if unit < 4 else "confirmation",
                "gold_position": gold,
                "correct_answer": case["correct"],
                "wrong_answer": case["wrong"],
                "prompt_core": prompt_core,
                "prompt": f"{prompt_core} {choices}. Reply with only A or B.",
                "free_prompt": f"{prompt_core} Answer with only the answer word.",
                "role_values": case["roles"],
                "semantic_graph": {**case.get("semantic_graph", {}), "surface": surface, "material": "natural_five_surface"},
            })
    finally:
        base.UNITS = saved_units
    return rows


def compile_qwen(tokenizer, rows: list[dict]) -> list[dict]:
    base = old.previous.previous
    return base.compile_qwen(tokenizer, rows)


GRAPH_UNITS = tuple(
    {
        "root": f"zel{chr(97 + i)}",
        "mid1": f"miv{chr(97 + i)}",
        "mid2": f"tor{chr(97 + i)}",
        "mid3": f"pel{chr(97 + i)}",
        "final": f"class{chr(97 + i)}",
        "wrong": f"other{chr(97 + i)}",
        "noise": f"noise{chr(97 + i)}",
    }
    for i in range(12)
)

GRAPH_LOCKBOX_UNITS = tuple(
    {
        "root": f"vax{chr(97 + i)}",
        "mid1": f"qen{chr(97 + i)}",
        "mid2": f"rud{chr(97 + i)}",
        "mid3": f"sop{chr(97 + i)}",
        "final": f"group{chr(97 + i)}",
        "wrong": f"alien{chr(97 + i)}",
        "noise": f"drift{chr(97 + i)}",
    }
    for i in range(6)
)


def graph_facts(unit: dict, depth: int, shortcut: int, mode: str = "chain") -> tuple[list[str], str]:
    mids = [unit["mid1"], unit["mid2"], unit["mid3"]]
    if depth == 1:
        nodes = [unit["root"], unit["final"]]
    else:
        nodes = [unit["root"], *mids[:depth - 1], unit["final"]]
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    correct = unit["final"]
    if mode == "reversed":
        edges = [(right, left) for left, right in edges]
        correct = "unknown"
    elif mode == "broken" and edges:
        cut = len(edges) // 2
        edges[cut] = (edges[cut][0], unit["noise"])
        correct = "unknown"
    facts = [f'The code "{left}" belongs to "{right}".' for left, right in edges]
    if shortcut and mode == "chain":
        facts.append(f'A direct register also places "{unit["root"]}" within "{unit["final"]}".')
    if mode == "irrelevant":
        facts.append(f'The unrelated code "{unit["noise"]}" belongs to "{unit["wrong"]}".')
    if mode == "multipath" and depth > 1:
        facts.append(f'An independent path also places "{unit["root"]}" within "{unit["final"]}".')
    return facts, correct


def graph_material(units: tuple[dict, ...] | None = None, lockbox: bool = False) -> list[dict]:
    if units is None:
        units = GRAPH_LOCKBOX_UNITS if lockbox else GRAPH_UNITS
    surfaces = ("registry", "briefing")
    rows = []
    if lockbox:
        modes = ("chain", "reversed", "broken", "irrelevant", "multipath", "shortcut")
        iterator = itertools.product(range(6), range(1, 5), surfaces, modes)
        for unit_i, depth, surface, mode in iterator:
            unit = units[unit_i]
            shortcut = int(mode == "shortcut")
            effective_mode = "chain" if mode == "shortcut" else mode
            facts, correct = graph_facts(unit, depth, shortcut, effective_mode)
            wrong = "unknown" if correct != "unknown" else unit["final"]
            order = 1 if (unit_i + depth + surfaces.index(surface) + modes.index(mode)) % 2 == 0 else -1
            rows.append(_graph_row(unit, len(units), unit_i, depth, surface, shortcut, order, facts, correct, wrong, mode, "lockbox"))
    else:
        iterator = itertools.product(range(12), range(1, 5), surfaces, (0, 1), (1, -1))
        for unit_i, depth, surface, shortcut, order in iterator:
            unit = units[unit_i]
            facts, correct = graph_facts(unit, depth, shortcut)
            rows.append(_graph_row(unit, len(units), unit_i, depth, surface, shortcut, order, facts, correct, unit["wrong"], "chain", "main"))
    return rows


def _graph_row(unit: dict, unit_count: int, unit_i: int, depth: int, surface: str, shortcut: int, order: int, facts: list[str], correct: str, wrong: str, mode: str, material: str) -> dict:
    body = " ".join(facts)
    if surface == "registry":
        prompt_core = f"A registry contains these entries: {body} Which final class contains the code \"{unit['root']}\"?"
    else:
        prompt_core = f"During a briefing, the following links were stated. {body} Based only on those links, what is the final class of \"{unit['root']}\"?"
    choices, gold = old.previous.previous.options(correct, wrong, order)
    relation = "belongs to"
    return {
        "case_id": f"c33{'4' if material == 'lockbox' else '1'}-{surface}-u{unit_i}-d{depth}-h{shortcut}-{mode}-{order:+d}",
        "panel": "knowledge_graph",
        "family": "type_graph",
        "surface": surface,
        "unit": unit_i,
        "depth": depth,
        "shortcut": shortcut,
        "mode": mode,
        "order": order,
        "partition": "discovery" if unit_i < max(1, unit_count * 2 // 3) else "confirmation",
        "gold_position": gold,
        "correct_answer": correct,
        "wrong_answer": wrong,
        "prompt_core": prompt_core,
        "prompt": f"{prompt_core} {choices}. Reply with only A or B.",
        "free_prompt": f"{prompt_core} Answer with only the answer word.",
        "role_values": {
            "primary": unit["root"],
            "secondary": unit["mid1"] if depth > 1 else unit["final"],
            "relation": relation,
            "context": unit["final"],
            "query": unit["root"],
        },
        "semantic_graph": {"depth": depth, "shortcut": bool(shortcut), "mode": mode, "material": material},
    }


def compile_general(tokenizer, rows: list[dict], interface: str = "strict_chat") -> list[dict]:
    compiled = []
    for row in rows:
        ids, candidates = render_interface(tokenizer, row, interface)
        positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value, interface))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return compiled


@torch.inference_mode()
def batch_capture_qwen(rows: list[dict], compiled: list[dict], out: Path, full_selector=None, batch_size: int = 4, field_width: int = WIDTH) -> dict:
    """Capture all Qwen checkpoints and all coordinates; optionally all tokens."""
    model = None
    hooks = []
    captured = []
    n = len(compiled)
    role_states = np.lib.format.open_memmap(out / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(n, 38, len(ROLES), DIM))
    selected = [i for i, row in enumerate(rows) if full_selector is not None and full_selector(row)]
    full_fields = None
    full_lookup = {row_i: i for i, row_i in enumerate(selected)}
    if selected:
        full_fields = np.lib.format.open_memmap(out / "raw/full_fields_holdout.float16.npy", mode="w+", dtype=np.float16, shape=(len(selected), 38, field_width, DIM))
    behavior = []
    hidden_index = []
    try:
        model, tokenizer, device, placement = model_base.load_bf16("qwen3")
        quant = model_base.quantization_audit(model)
        base = model.model

        def capture(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        hooks.append(base.embed_tokens.register_forward_hook(capture))
        hooks.extend(layer.register_forward_hook(capture) for layer in base.layers)
        hooks.append(base.norm.register_forward_hook(capture))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, n, batch_size):
            batch = compiled[start:start + batch_size]
            ids = torch.full((len(batch), field_width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            positions = torch.zeros_like(ids)
            lengths = []
            for local, row in enumerate(batch):
                values = row["prompt_ids"]
                if len(values) > field_width:
                    raise RuntimeError((row["case_id"], len(values), field_width))
                lengths.append(len(values))
                ids[local, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
                mask[local, :len(values)] = 1
                positions[local, :len(values)] = torch.arange(len(values), device=device)
            captured.clear()
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            if len(captured) != 38:
                raise RuntimeError(("checkpoint_count", len(captured)))
            for local, row in enumerate(batch):
                i = start + local
                length = lengths[local]
                for q, state in enumerate(captured):
                    value = state[local, :length].float().cpu().numpy().astype(np.float16)
                    for role_i, role in enumerate(ROLES):
                        role_states[i, q, role_i] = value[row["role_positions"][role]].mean(axis=0).astype(np.float16)
                    if i in full_lookup:
                        full_fields[full_lookup[i], q, :length] = value
                logits = [float(output.logits[local, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(logits[1] > logits[0])
                behavior.append({"case_id": row["case_id"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "score0": logits[0], "score1": logits[1]})
                hidden_index.append({
                    "hidden_index": i,
                    "case_id": row["case_id"],
                    "panel": row["panel"],
                    "family": row["family"],
                    "surface": row["surface"],
                    "unit": row["unit"],
                    "factor_a": row.get("factor_a"),
                    "factor_b": row.get("factor_b"),
                    "depth": row.get("depth"),
                    "shortcut": row.get("shortcut"),
                    "mode": row.get("mode"),
                    "order": row["order"],
                    "partition": row.get("partition"),
                    "length": length,
                    "gold_position": row["gold_position"],
                    "prediction": prediction,
                    "correct": prediction == row["gold_position"],
                    "role_positions": row["role_positions"],
                })
            role_states.flush()
            if full_fields is not None:
                full_fields.flush()
            if start % 128 == 0 or start + len(batch) == n:
                print(f"[capture] {start + len(batch)}/{n}", flush=True)
        core.write_rows(out / "raw/behavior.jsonl", behavior)
        core.write_rows(out / "raw/hidden_index.jsonl", hidden_index)
        if selected:
            core.save(out / "raw/full_field_row_map.json", {"source_indices": selected})
        return {
            "placement": placement,
            "quantization": quant,
            "rows": n,
            "full_token_rows": len(selected),
            "accuracy": float(np.mean([row["correct"] for row in behavior])),
            "role_shape": list(role_states.shape),
            "full_shape": list(full_fields.shape) if full_fields is not None else None,
        }
    finally:
        for hook in hooks:
            hook.remove()
        model_base.release(model)
        gc.collect()


def finite_dict(value) -> bool:
    if isinstance(value, Mapping):
        return all(finite_dict(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_dict(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


@torch.inference_mode()
def run_cross_model_panel(campaign: str, model_name: str) -> None:
    """Run one model for C327-C329. Models are invoked by separate processes."""
    parent = core.load(OUTS["C326"] / "analysis/final.json")
    rows = core.rows(OUTS["C326"] / "material/cases.jsonl")
    checks = {
        "parent": parent["all_checks_passed"],
        "registered_model": model_name in MODELS,
        "rows": len(rows) == 480,
        "cuda": torch.cuda.is_available(),
        "model_sequential_process": True,
    }
    protocol = {
        "status": "single_model_cross_panel_frozen",
        "model": model_name,
        "interfaces": list(INTERFACES),
        "interface_calibration": "unit0 discovery rows only; select highest accuracy with frozen tie order",
        "behavior_panel": "all 480 cases; units4-5 are confirmation",
        "role_archive": "all model-native hidden checkpoints x six roles x every model-native activation coordinate",
        "causal_panel": "confirmation H11 cells on report and witness surfaces",
        "causal_checkpoint": "round(0.67 * number of model layers), frozen independent of outcomes",
        "causal_conditions": ["natural", "correct_delete", "wrong_family_delete", "coordinate_roll_delete", "role_roll_delete"],
        "behavior_gate": {"confirmation_global_min": 0.70, "family_min": 0.50, "surface_min": 0.50},
        "claim_boundary": "Each model keeps its native coordinate axis. A pass is model-specific residual response evidence, not cross-model coordinate identity or full functional equivalence.",
    }
    out = prepare(campaign, protocol, checks)
    model = None
    try:
        model, tokenizer, device, placement = model_base.load_bf16(model_name)
        quantization = model_base.quantization_audit(model)
        info = get_model_info(model, model_name)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        calibration_rows = [row for row in rows if row["unit"] == 0]
        interface_results = {}
        for interface in INTERFACES:
            compiled = [(row, *render_interface(tokenizer, row, interface)) for row in calibration_rows]
            results = []
            for start in range(0, len(compiled), BATCH[model_name]):
                results.extend(score_interface_batch(model, device, pad, compiled[start:start + BATCH[model_name]]))
            interface_results[interface] = results
            core.write_rows(out / f"raw/interface_{interface}.jsonl", results)
            print(f"[{campaign}] {model_name}/{interface}: {np.mean([row['correct'] for row in results]):.4f}", flush=True)
        interface_accuracy = {name: float(np.mean([row["correct"] for row in result])) for name, result in interface_results.items()}
        selected_interface = max(INTERFACES, key=lambda name: (interface_accuracy[name], -INTERFACES.index(name)))
        compiled = compile_general(tokenizer, rows, selected_interface)
        if any(any(len(candidate) != 1 for candidate in row["candidate_ids"]) for row in compiled):
            raise RuntimeError((model_name, "multi_token_candidate_not_supported_in_unified_capture"))
        core.write_rows(out / "compiled/model_rows.jsonl", compiled)
        nq = info.n_layers + 1
        states = np.lib.format.open_memmap(out / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(len(rows), nq, len(ROLES), info.d_model))
        behavior = []
        hidden_index = []
        for i, row in enumerate(compiled):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, output_hidden_states=True)
            if len(output.hidden_states) != nq:
                raise RuntimeError((model_name, len(output.hidden_states), nq))
            for q, state in enumerate(output.hidden_states):
                for role_i, role in enumerate(ROLES):
                    states[i, q, role_i] = state[0, row["role_positions"][role]].mean(0).float().cpu().numpy().astype(np.float16)
            scores = np.asarray([float(output.logits[0, ids.shape[1] - 1, candidate[0]]) for candidate in row["candidate_ids"]], np.float32)
            prediction = int(scores[1] > scores[0])
            correct = prediction == row["gold_position"]
            behavior.append({"case_id": row["case_id"], "prediction": prediction, "correct": correct, "score0": float(scores[0]), "score1": float(scores[1])})
            hidden_index.append({
                "hidden_index": i,
                "case_id": row["case_id"],
                "panel": row["panel"],
                "family": row["family"],
                "surface": row["surface"],
                "unit": row["unit"],
                "factor_a": row["factor_a"],
                "factor_b": row["factor_b"],
                "order": row["order"],
                "partition": row["partition"],
                "correct": correct,
                "role_positions": row["role_positions"],
            })
            states.flush()
            if i % 60 == 0 or i + 1 == len(rows):
                print(f"[{campaign}] {model_name} capture {i + 1}/{len(rows)}", flush=True)
        core.write_rows(out / "raw/behavior.jsonl", behavior)
        core.write_rows(out / "raw/hidden_index.jsonl", hidden_index)
        lookup = {row["case_id"]: source for row, source in zip(behavior, rows)}
        confirmation = [row for row in behavior if lookup[row["case_id"]]["partition"] == "confirmation"]
        confirmation_accuracy = float(np.mean([row["correct"] for row in confirmation]))
        by_family = {family: float(np.mean([row["correct"] for row in confirmation if lookup[row["case_id"]]["family"] == family])) for family in FAMILIES}
        by_surface = {surface: float(np.mean([row["correct"] for row in confirmation if lookup[row["case_id"]]["surface"] == surface])) for surface in NATURAL_SURFACES}
        gate = protocol["behavior_gate"]
        eligible = confirmation_accuracy >= gate["confirmation_global_min"] and min(by_family.values()) >= gate["family_min"] and min(by_surface.values()) >= gate["surface_min"]

        interaction_means = {}
        prediction_rows = []
        for family in FAMILIES:
            arrays, groups = factorial_arrays(states, hidden_index, family)
            discovery_mask = np.asarray([group["unit"] in (0, 1) for group in groups], bool)
            confirmation_mask = np.asarray([group["unit"] in (4, 5) for group in groups], bool)
            mean = arrays["interaction"][discovery_mask].mean(axis=0)
            interaction_means[family] = mean
            truth = arrays["interaction"][confirmation_mask]
            prediction_rows.append({"family": family, "groups": int(confirmation_mask.sum()), "relative_mae_gain": relative_gain(truth, truth - mean[None, ...])})
        interaction_archive = np.stack([interaction_means[family] for family in FAMILIES], axis=0)
        np.save(out / "analysis/family_interaction_means.float32.npy", interaction_archive)

        causal_rows = []
        if eligible:
            layers = get_layers(model)
            q = max(1, min(info.n_layers - 1, int(round(0.67 * info.n_layers))))
            selected_cases = [row for row in hidden_index if row["partition"] == "confirmation" and row["factor_a"] == 1 and row["factor_b"] == 1 and row["surface"] in ("report", "witness") and row["correct"]]
            compiled_lookup = {row["case_id"]: row for row in compiled}
            global_mean = interaction_archive.mean(axis=0)
            for source in selected_cases:
                family = source["family"]
                family_i = FAMILIES.index(family)
                row = compiled_lookup[source["case_id"]]
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                natural_next = np.asarray(states[source["hidden_index"], q + 1], np.float32)
                additive_next = natural_next - interaction_means[family][q + 1]
                denom = float(np.mean(np.abs(natural_next - additive_next)))
                correct_vector = interaction_means[family][q]
                wrong_vector = interaction_means[FAMILIES[(family_i + 1) % 6]][q]
                vectors_by_condition = {
                    "correct_delete": correct_vector,
                    "wrong_family_delete": norm_match(wrong_vector, correct_vector),
                    "coordinate_roll_delete": np.roll(correct_vector, 97, axis=-1),
                    "role_roll_delete": np.roll(correct_vector, 1, axis=0),
                }
                conditions = {}
                for condition in protocol["causal_conditions"]:
                    captured = []
                    vectors = None if condition == "natural" else role_position_vectors(row, vectors_by_condition[condition])

                    def patch_hook(_module, _args, output):
                        if vectors is None:
                            return output
                        value = output[0] if isinstance(output, tuple) else output
                        updated = value.clone()
                        for position, vector in vectors.items():
                            updated[0, position] = updated[0, position] - torch.tensor(vector, dtype=updated.dtype, device=updated.device)
                        return (updated, *output[1:]) if isinstance(output, tuple) else updated

                    def capture_hook(_module, _args, output):
                        value = output[0] if isinstance(output, tuple) else output
                        captured.append(np.asarray([value[0, row["role_positions"][role]].mean(0).float().cpu().numpy() for role in ROLES], np.float32))

                    patch = layers[q - 1].register_forward_hook(patch_hook)
                    capture = layers[q].register_forward_hook(capture_hook)
                    try:
                        output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                    finally:
                        patch.remove()
                        capture.remove()
                    scores = np.asarray([float(output.logits[0, ids.shape[1] - 1, candidate[0]]) for candidate in row["candidate_ids"]], np.float32)
                    gold = row["gold_position"]
                    conditions[condition] = {
                        "next_field_movement_toward_additive": float(1.0 - np.mean(np.abs(captured[0] - additive_next)) / max(denom, 1e-12)),
                        "gold_margin": float(scores[gold] - scores[1 - gold]),
                    }
                    del output
                correct_movement = conditions["correct_delete"]["next_field_movement_toward_additive"]
                best_wrong = max(conditions[name]["next_field_movement_toward_additive"] for name in ("wrong_family_delete", "coordinate_roll_delete", "role_roll_delete"))
                causal_rows.append({"case_id": source["case_id"], "family": family, "surface": source["surface"], "unit": source["unit"], "q": q, "conditions": conditions, "correct_movement": correct_movement, "correct_minus_best_wrong": correct_movement - best_wrong})
            core.write_rows(out / "raw/causal_results.jsonl", causal_rows)
        causal_families = []
        for family in FAMILIES:
            values = [row for row in causal_rows if row["family"] == family]
            causal_families.append({
                "family": family,
                "samples": len(values),
                "mean_correct_movement": float(np.mean([row["correct_movement"] for row in values])) if values else None,
                "mean_correct_minus_best_wrong": float(np.mean([row["correct_minus_best_wrong"] for row in values])) if values else None,
            })
        headline = {
            "status": "single_model_cross_panel_closed",
            "model": model_name,
            "interface_accuracy": interface_accuracy,
            "selected_interface": selected_interface,
            "confirmation_accuracy": confirmation_accuracy,
            "by_family_accuracy": by_family,
            "by_surface_accuracy": by_surface,
            "behavior_eligible": eligible,
            "model_info": {"layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "composition_prediction": prediction_rows,
            "causal_response": causal_families,
            "placement": placement,
            "quantization": quantization,
            "strict_interpretation": protocol["claim_boundary"],
        }
        close(campaign, headline, {
            "behavior_rows": len(behavior) == 480,
            "state_shape": list(states.shape) == [480, nq, 6, info.d_model],
            "six_composition_rows": len(prediction_rows) == 6,
            "causal_accounting": (len(causal_rows) > 0) == eligible,
            "finite": finite_dict(headline),
            "bf16": quantization["has_bf16_parameters"],
            "unquantized": not quantization["has_quantized_modules"],
        }, "C330_cross_model_synthesis_after_all_three")
    finally:
        model_base.release(model)
        gc.collect()
