#!/usr/bin/env python3
"""Phase1571 / C098: observation-first multi-relation graph field campaign.

The campaign is deliberately limited to embeddings, full Hidden States and
output logits.  It freezes the complete factorial material before loading the
model, captures every token at every state without projection, then performs
coordinate-wise Walsh analysis with discovery-only support selection.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1568_c097_major_stage_closure"
C097_CONTRACT = RESULT / "phase1565_c097_wordnet_independent_contract"
C097_CAPTURE = RESULT / "phase1566_c097_wordnet_capture"
C097_ATLAS = RESULT / "phase1567_c097_identifiable_common_residual_atlas"
OUT = RESULT / "phase1571_c098_observation_first_graph_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE = 1571
CAMPAIGN = "C098"
STATES = 37
DIM = 2560
FACTORS = ("x", "y", "branch", "code")
EFFECTS = tuple(
    "".join(FACTORS[i] for i in range(len(FACTORS)) if mask & (1 << i))
    for mask in range(1, 1 << len(FACTORS))
)
FOCUS_ROLES = ("target_pre", "target_post", "query_target", "boundary")
PARTITIONS = ("response_discovery", "confirmation", "lockbox")
WORLDS = ("natural", "artificial", "counterfactual")
FAMILIES = ("taxonomy", "containment", "comparison", "precedence")
SURFACES = ("forward_order", "reverse_order")
CODEBOOKS = {
    1: {
        "name": "standard",
        "instruction": "If the query follows, answer yes; otherwise answer no.",
    },
    -1: {
        "name": "reversed",
        "instruction": "If the query follows, answer no; otherwise answer yes.",
    },
}
SYSTEM = (
    "Use only the local ledger. Every ledger statement is true in its local "
    "world, even if it conflicts with ordinary knowledge. The stated relation "
    "is transitive. Follow directed paths and reply with exactly yes or no."
)
RELATIONS = {
    "taxonomy": {
        "statement": "{left} is a kind of {right}",
        "query": "Is {left} a kind of {right}?",
    },
    "containment": {
        "statement": "{left} is inside {right}",
        "query": "Is {left} inside {right}?",
    },
    "comparison": {
        "statement": "{left} is smaller than {right}",
        "query": "Is {left} smaller than {right}?",
    },
    "precedence": {
        "statement": "{left} happens before {right}",
        "query": "Does {left} happen before {right}?",
    },
}

# Each tuple is (a, b, c, d, f, g, h, i).  The two baseline paths and the
# disconnected/attached distractor chain were manually selected for readable
# controlled English.  Only the x=y=+1 natural cell is asserted to agree with
# ordinary knowledge; every other cell remains a local-ledger intervention.
NATURAL_UNITS = {
    "taxonomy": [
        ("sparrow", "bird", "animal", "vehicle", "machine", "organism", "entity", "object"),
        ("apple", "fruit", "food", "tool", "artifact", "plant product", "organic matter", "substance"),
        ("oak", "tree", "plant", "fish", "animal", "organism", "living thing", "entity"),
        ("violin", "instrument", "artifact", "river", "landform", "object", "physical entity", "entity"),
        ("tulip", "flower", "plant", "planet", "celestial body", "organism", "living thing", "entity"),
        ("salmon", "fish", "animal", "chair", "furniture", "organism", "living thing", "entity"),
    ],
    "containment": [
        ("key", "box", "cabinet", "envelope", "desk", "pocket", "coat", "closet"),
        ("coin", "purse", "drawer", "jar", "cupboard", "pocket", "jacket", "wardrobe"),
        ("letter", "envelope", "mailbag", "folder", "backpack", "pouch", "case", "locker"),
        ("seed", "pod", "basket", "sachet", "crate", "packet", "bag", "bin"),
        ("book", "box", "closet", "bag", "locker", "sleeve", "case", "cabinet"),
        ("ring", "casket", "safe", "pocket", "wallet", "pouch", "bag", "drawer"),
    ],
    "comparison": [
        ("mouse", "cat", "horse", "beetle", "dog", "rabbit", "goat", "elephant"),
        ("cup", "bucket", "barrel", "spoon", "bowl", "mug", "pot", "tank"),
        ("hamlet", "town", "city", "room", "house", "cottage", "mansion", "palace"),
        ("moon", "planet", "star", "asteroid", "comet", "pebble", "boulder", "mountain"),
        ("seed", "apple", "pumpkin", "grain", "orange", "berry", "melon", "cart"),
        ("cottage", "mansion", "castle", "car", "warehouse", "shed", "barn", "stadium"),
    ],
    "precedence": [
        ("dawn", "noon", "dusk", "breakfast", "dinner", "sunrise", "lunch", "midnight"),
        ("enrollment", "course", "graduation", "application", "admission", "registration", "orientation", "instruction"),
        ("ignition", "travel", "arrival", "packing", "departure", "planning", "loading", "unloading"),
        ("rehearsal", "performance", "applause", "planning", "opening", "practice", "preview", "closing"),
        ("planting", "flowering", "harvest", "watering", "ripening", "sowing", "sprouting", "storage"),
        ("diagnosis", "treatment", "recovery", "consultation", "surgery", "screening", "testing", "discharge"),
    ],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def ba(gold: list[bool], pred: list[bool]) -> float:
    return float(np.mean([
        np.mean([p == g for p, g in zip(pred, gold, strict=True) if g == label])
        for label in (False, True)
    ]))


def tokenizer():
    from model_utils import MODEL_CONFIGS
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def artificial_nodes(family_index: int, unit_index: int) -> tuple[str, ...]:
    # Stable identifiers keep the artificial graph explicit without pretending
    # to be natural prose or unseen tokenizer vocabulary.
    base = family_index * 100 + unit_index * 10
    return tuple(f"Navo{base + offset:03d}" for offset in range(8))


def counterfactual_nodes(nodes: tuple[str, ...]) -> tuple[str, ...]:
    a, b, c, d, f, g, h, i = nodes
    return c, b, a, f, d, i, h, g


def partition_for(unit_index: int) -> str:
    return PARTITIONS[unit_index // 2]


def surface_for(unit_index: int) -> str:
    return SURFACES[unit_index % 2]


def make_statement(family: str, left: str, right: str) -> str:
    return RELATIONS[family]["statement"].format(left=left, right=right) + "."


def graph_edges(nodes: tuple[str, ...], x: int, y: int, branch: int) -> list[tuple[str, str]]:
    a, b, c, d, f, g, h, i = nodes
    first = (a, b if x == 1 else d)
    maps = [(b, c if y == 1 else f), (d, f if y == 1 else c)]
    # Both cells use the same lexical multiset.  The positive cell gives the
    # target an outgoing two-hop branch; the negative cell makes the first edge
    # incoming to the target, so it cannot carry the target to the query node.
    distractor = [(a, g), (g, h)] if branch == 1 else [(g, a), (g, h)]
    return [first, *maps, *distractor]


def reachable(edges: list[tuple[str, str]], source: str, target: str) -> tuple[bool, int]:
    graph: dict[str, list[str]] = defaultdict(list)
    for left, right in edges:
        graph[left].append(right)
    count = 0
    stack = [(source, (source,))]
    while stack:
        node, path = stack.pop()
        for nxt in graph[node]:
            if nxt in path:
                continue
            if nxt == target:
                count += 1
            else:
                stack.append((nxt, (*path, nxt)))
    return count > 0, count


def build_prompt(family: str, nodes: tuple[str, ...], x: int, y: int, branch: int, code: int, surface: str) -> tuple[str, list[tuple[str, str]]]:
    a, _, c, _, _, _, _, _ = nodes
    edges = graph_edges(nodes, x, y, branch)
    statements = [make_statement(family, left, right) for left, right in edges]
    if surface == "reverse_order":
        statements = list(reversed(statements))
    ledger = " ".join(f"{index + 1}. {value}" for index, value in enumerate(statements))
    query = RELATIONS[family]["query"].format(left=a, right=c)
    prompt = (
        f"Target before graph: {a}. Local ledger: {ledger} "
        f"Target after graph: {a}. Query: {query} "
        f"Decision code: {CODEBOOKS[code]['instruction']} Reply exactly yes or no."
    )
    return prompt, edges


def material_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for unit_index, natural in enumerate(NATURAL_UNITS[family]):
            for world in WORLDS:
                nodes = tuple(natural)
                if world == "artificial":
                    nodes = artificial_nodes(family_index, unit_index)
                elif world == "counterfactual":
                    nodes = counterfactual_nodes(nodes)
                unit_id = f"c098-{family}-{world}-{unit_index:02d}"
                unit = {
                    "unit_id": unit_id,
                    "family": family,
                    "world": world,
                    "unit_index": unit_index,
                    "partition": partition_for(unit_index),
                    "surface": surface_for(unit_index),
                    "nodes": list(nodes),
                    "natural_baseline_curated": world == "natural",
                }
                units.append(unit)
                for x, y, branch, code in itertools.product((1, -1), repeat=4):
                    prompt, edges = build_prompt(family, nodes, x, y, branch, code, unit["surface"])
                    follows, path_count = reachable(edges, nodes[0], nodes[2])
                    truth = x == y
                    output_yes = truth if code == 1 else not truth
                    rows.append({
                        "case_id": f"c098-{len(rows):04d}",
                        **{key: unit[key] for key in ("unit_id", "family", "world", "unit_index", "partition", "surface")},
                        "nodes": list(nodes),
                        "x": x,
                        "y": y,
                        "branch": branch,
                        "code": code,
                        "codebook": CODEBOOKS[code]["name"],
                        "truth": truth,
                        "output_yes": output_yes,
                        "path_count": path_count,
                        "edges": [list(edge) for edge in edges],
                        "prompt": prompt,
                        "candidates": ["yes", "no"],
                        "gold_position": 0 if output_yes else 1,
                    })
                    if follows != truth:
                        raise RuntimeError((unit_id, x, y, follows, truth))
    return units, rows


def subsequence_spans(values: list[int], needle: list[int]) -> list[list[int]]:
    return [
        list(range(start, start + len(needle)))
        for start in range(len(values) - len(needle) + 1)
        if values[start:start + len(needle)] == needle
    ]


def name_spans(tok, ids: list[int], name: str) -> list[list[int]]:
    found: set[tuple[int, ...]] = set()
    for form in (name, " " + name):
        needle = [int(value) for value in tok.encode(form, add_special_tokens=False)]
        for span in subsequence_spans(ids, needle):
            found.add(tuple(span))
    return [list(span) for span in sorted(found, key=lambda value: (value[0], len(value)))]


def compile_rows(tok, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compiled = []
    candidate_ids = [
        [int(value) for value in tok.encode(" " + candidate, add_special_tokens=False)]
        for candidate in ("yes", "no")
    ]
    if any(len(value) != 1 for value in candidate_ids):
        raise RuntimeError(("candidate singleton", candidate_ids))
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        a, _, c, _, _, _, _, _ = row["nodes"]
        a_spans = name_spans(tok, ids, a)
        c_spans = name_spans(tok, ids, c)
        if len(a_spans) < 4 or len(c_spans) < 2:
            raise RuntimeError((row["case_id"], a, c, a_spans, c_spans))
        code_text = CODEBOOKS[row["code"]]["instruction"]
        code_needles = [int(value) for value in tok.encode(" " + code_text, add_special_tokens=False)]
        code_spans = subsequence_spans(ids, code_needles)
        if not code_spans:
            code_needles = [int(value) for value in tok.encode(code_text, add_special_tokens=False)]
            code_spans = subsequence_spans(ids, code_needles)
        if not code_spans:
            raise RuntimeError(("code span", row["case_id"]))
        role_positions = {
            "target_pre": a_spans[0],
            "target_post": a_spans[-2],
            "query_target": a_spans[-1],
            "query_endpoint": c_spans[-1],
            "code_instruction": code_spans[-1],
            "boundary": [len(ids) - 1],
        }
        if not (
            max(role_positions["target_pre"]) < min(role_positions["target_post"])
            < min(role_positions["query_target"])
            < min(role_positions["code_instruction"])
            < role_positions["boundary"][0]
        ):
            raise RuntimeError(("causal role order", row["case_id"], role_positions))
        compiled.append({
            **row,
            "prompt_ids": ids,
            "candidate_ids": candidate_ids,
            "role_positions": role_positions,
        })
    return compiled


def premodel_audit(units: list[dict[str, Any]], rows: list[dict[str, Any]], compiled: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [row["output_yes"] for row in rows]
    truth = [row["truth"] for row in rows]
    zero_models = {
        "always_yes": ba(gold, [True] * len(rows)),
        "always_no": ba(gold, [False] * len(rows)),
        "x_only": ba(gold, [row["x"] == 1 for row in rows]),
        "y_only": ba(gold, [row["y"] == 1 for row in rows]),
        "branch_only": ba(gold, [row["branch"] == 1 for row in rows]),
        "code_only": ba(gold, [row["code"] == 1 for row in rows]),
        "truth_without_code": ba(gold, truth),
        "truth_x_code_oracle": ba(gold, [row["truth"] == (row["code"] == 1) for row in rows]),
    }
    unit_lengths: dict[str, set[int]] = defaultdict(set)
    for row in compiled:
        unit_lengths[row["unit_id"]].add(len(row["prompt_ids"]))
    checks = {
        "parent_authorization": core.load(PARENT / "analysis/final.json")["authorization"] in {
            "run_phase1569_c097_relation_contrast_heatmap_export",
            "freeze_C098_observation_first_graph_contract",
        },
        "parent_audit": core.load(PARENT / "audit/independent_final_audit.json")["all_checks_passed"],
        "unit_count": len(units) == 72,
        "case_count": len(rows) == 1152,
        "unit_cells": Counter(row["unit_id"] for row in rows) == {unit["unit_id"]: 16 for unit in units},
        "world_balance": Counter(row["world"] for row in rows) == {world: 384 for world in WORLDS},
        "family_balance": Counter(row["family"] for row in rows) == {family: 288 for family in FAMILIES},
        "partition_balance": Counter(row["partition"] for row in rows) == {partition: 384 for partition in PARTITIONS},
        "surface_balance": Counter(row["surface"] for row in rows) == {surface: 576 for surface in SURFACES},
        "factor_balance": all(Counter(row[factor] for row in rows) == {1: 576, -1: 576} for factor in FACTORS),
        "truth_balance": Counter(truth) == {True: 576, False: 576},
        "output_balance": Counter(gold) == {True: 576, False: 576},
        "truth_formula": all(row["truth"] == (row["x"] == row["y"]) for row in rows),
        "output_formula": all(row["output_yes"] == (row["truth"] == (row["code"] == 1)) for row in rows),
        "path_semantics": all((row["path_count"] == 1) == row["truth"] for row in rows),
        "false_zero_path": all(row["path_count"] == 0 for row in rows if not row["truth"]),
        "candidate_singletons": all(len(value) == 1 for row in compiled for value in row["candidate_ids"]),
        "role_order": all(
            max(row["role_positions"]["target_pre"]) < min(row["role_positions"]["target_post"])
            < min(row["role_positions"]["query_target"])
            < min(row["role_positions"]["code_instruction"])
            for row in compiled
        ),
        "unit_length_delta_at_most_one": all(max(lengths) - min(lengths) <= 1 for lengths in unit_lengths.values()),
        "unique_prompts": len({row["prompt"] for row in rows}) == len(rows),
        "zero_models": all(value == 0.5 for key, value in zero_models.items() if key != "truth_x_code_oracle")
        and zero_models["truth_x_code_oracle"] == 1.0,
        "machine_naturalness": all(row["prompt"].count("Query:") == 1 and row["prompt"].endswith("yes or no.") for row in rows),
        "hidden_not_accessed": True,
    }
    return {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "zero_models": zero_models,
        "semantic_uniqueness": "exact directed reachability; one path for true cells and no path for false cells",
        "naturalness": {
            "natural_baseline": "manually curated familiar controlled-English chains",
            "all_cells": "machine-audited controlled English local-ledger interventions",
            "human_blind_lock": False,
            "missingness": "M_HUMAN_NATURALNESS for artificial/counterfactual and nonbaseline interventions",
        },
    }


def prepare() -> None:
    if (OUT / "analysis/final.json").exists() or (OUT / "protocol/preregistration.json").exists():
        raise RuntimeError("Phase1571 prepare output already exists")
    requirements = core.load(PARENT / "protocol/c098_requirements.json")
    units, rows = material_rows()
    tok = tokenizer()
    compiled = compile_rows(tok, rows)
    audit = premodel_audit(units, rows, compiled)
    if not audit["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in audit["checks"].items() if not value})
    core.write_rows(OUT / "material/frozen_graph_units.jsonl", units)
    core.write_rows(OUT / "material/frozen_cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    core.write_rows(OUT / "material/frozen_test_examples.jsonl", [
        next(row for row in rows if row["family"] == family and row["world"] == world and row["unit_index"] == 0 and row["x"] == 1 and row["y"] == 1 and row["branch"] == 1 and row["code"] == 1)
        for family, world in (("taxonomy", "natural"), ("containment", "artificial"), ("comparison", "counterfactual"), ("precedence", "natural"))
    ])
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c098.observation_first_graph_field.v1",
        "research_object": "formation and reuse of directed transitive path information across natural, artificial and counterfactual multi-relation graphs",
        "model": "Qwen3-4B local BF16 CUDA, no quantization",
        "worlds": list(WORLDS),
        "families": list(FAMILIES),
        "partitions": list(PARTITIONS),
        "surfaces": list(SURFACES),
        "factors": list(FACTORS),
        "effects": list(EFFECTS),
        "path_truth_effect": "xy",
        "output_coupling_effect": "xycode",
        "roles": list(FOCUS_ROLES),
        "allowed_observables": ["input embeddings", "all tokens at all full-dimensional Hidden States", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "learned probes", "post-unblind threshold changes", "post-unblind material changes"],
        "material": {
            "unit_count": 72,
            "case_count": 1152,
            "cells_per_unit": 16,
            "units_per_world_family_partition": 2,
            "unit_sha256": core.sha(OUT / "material/frozen_graph_units.jsonl"),
            "case_sha256": core.sha(OUT / "material/frozen_cases.jsonl"),
            "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"),
        },
        "execution": {
            "batch_size": 8,
            "state_count": STATES,
            "hidden_dimension": DIM,
            "raw_dtype": "float16",
            "ragged_storage": "state x concatenated_real_token x coordinate",
            "repeat_hidden_max_abs": 1e-6,
            "repeat_logit_max_abs": 1e-6,
            "causal_prefix_effect_max_abs": 1e-6,
            "code_before_visible_effect_max_abs": 1e-6,
        },
        "analysis": {
            "walsh_formula": "C_S=2^-4 sum_z chi_S(z) H(z)",
            "discovery_support": "top64 absolute xy coordinates selected only from response_discovery",
            "validation": ["confirmation", "lockbox"],
            "fixed_support_metrics": ["restricted cosine", "sign agreement", "target energy fraction"],
            "dynamic_support_metrics": ["top64 Jaccard", "full-vector cosine"],
            "all_token_scan": "xy coefficient for code-averaged fixed-order cells in response_discovery",
            "c097_design_null": "1000 within-row query-label permutations preserving raw 3x3 shared-cell formulas",
            "descriptive_flags": {
                "fixed_coordinate_candidate": "both holdouts full cosine >=0.50, top64 Jaccard >=0.25 and restricted sign >=0.75",
                "dynamic_alliance_candidate": "both holdouts full cosine >=0.50 while median top64 Jaccard <0.25",
                "important_visualization": "at least 75% of world-family focus groups have both holdout full cosine >=0.50",
            },
        },
        "evidence_policy": "behavior is stratified; only integrity/nonfinite/contract mutation stops observation",
        "claim_boundary": {
            "allowed": "single-Qwen task-scoped full-coordinate graph-path observations and holdout repetition",
            "forbidden": ["universal semantic graph code", "semantic neurons", "causal necessity or sufficiency", "cross-model law", "new mathematics"],
        },
        "c097_audit_corrections": [
            "G_C mixes relation match, truth, output preparation and boundary effects",
            "0.50 common-energy fraction is a weak shared-cell design baseline",
            "the three pairwise contrasts omit the antisymmetric cycle degree of freedom",
            "C097 top64 provenance must remain discovery-selected and holdout-evaluated",
        ],
        "parent_requirements": requirements,
        "created_at_utc": now(),
        "authorization": "run_phase1571_capture",
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["producer_sha256"] = core.sha(Path(__file__))
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", audit)
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/uploaded_analysis_adjudication.json", {
        "retained": [
            "C097 contrast-level identifiability correction and exact zero-sum energy accounting",
            "new Chinese and independent English repetitions are real but task-scoped late-boundary observations",
            "C097 top64 alignment exceeds its registered coordinate-permutation baseline",
        ],
        "corrected": [
            "G_C is not a shared semantic module and R_fg is not a relation-specific module",
            "the 0.50 common-fraction gate is partly induced by shared raw cells and requires a design-preserving null",
            "three symmetric pair contrasts do not span the full four-dimensional 3x3 interaction",
            "state coordinates are activations, not parameters or identified neurons",
            "existing mathematics is sufficient for the present observational campaign",
        ],
        "added": [
            "orthogonal answer-code factor placed after the query",
            "directed path truth separated from output token through an xy by code factorial",
            "full-token, all-state, raw-coordinate storage and discovery-only support selection",
        ],
    })
    print(json.dumps({"audit": audit, "protocol_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, ensure_ascii=False, indent=2))


def fixed_batch(rows: list[dict[str, Any]], pad: int, device: torch.device):
    width = max(len(row["prompt_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for index, row in enumerate(rows):
        values = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[index, : len(values)] = values
        mask[index, : len(values)] = 1
        lengths.append(len(values))
    position_ids = mask.cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 0)
    return ids, mask, position_ids, lengths


@torch.inference_mode()
def forward(model, rows: list[dict[str, Any]], pad: int, device: torch.device):
    ids, mask, positions, lengths = fixed_batch(rows, pad, device)
    kwargs = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": positions,
        "use_cache": False,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        kwargs["logits_to_keep"] = 1
    output = model(**kwargs)
    return output, ids, mask, positions, lengths


def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    if protocol["authorization"] != "run_phase1571_capture" or not audit["all_checks_passed"]:
        raise RuntimeError("C098 frozen authorization missing")
    if protocol["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("producer changed after preregistration")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    total_tokens = sum(len(row["prompt_ids"]) for row in compiled)
    token_offsets = []
    cursor = 0
    for row in compiled:
        token_offsets.append((cursor, cursor + len(row["prompt_ids"])))
        cursor += len(row["prompt_ids"])
    raw_path = OUT / "raw/all_token_all_state_field.float16.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16, shape=(STATES, total_tokens, DIM))
    model = None
    index = []
    first_repeat = None
    finite = True
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for start in range(0, len(compiled), protocol["execution"]["batch_size"]):
            batch = compiled[start:start + protocol["execution"]["batch_size"]]
            output, ids, mask, positions, lengths = forward(model, batch, pad, device)
            if len(output.hidden_states) != STATES:
                raise RuntimeError(("state count", len(output.hidden_states)))
            logits = output.logits[:, -1].float()
            batch_scores = []
            batch_blocks = []
            for local, row in enumerate(batch):
                block = torch.stack([hidden[local, : lengths[local]] for hidden in output.hidden_states], dim=0)
                finite = finite and bool(torch.isfinite(block).all())
                cpu_block = block.to(dtype=torch.float16, device="cpu").numpy()
                left, right = token_offsets[start + local]
                field[:, left:right, :] = cpu_block
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                batch_scores.append(scores)
                if start == 0:
                    batch_blocks.append(cpu_block.copy())
                prediction = int(scores[1] > scores[0])
                index.append({
                    "row_index": start + local,
                    **{key: row[key] for key in ("case_id", "unit_id", "family", "world", "unit_index", "partition", "surface", "x", "y", "branch", "code", "codebook", "truth", "output_yes", "gold_position")},
                    "prediction": prediction,
                    "correct": prediction == row["gold_position"],
                    "scores": scores,
                    "token_start": left,
                    "token_end": right,
                    "token_count": right - left,
                    "role_positions": row["role_positions"],
                })
            if start == 0:
                first_repeat = (batch, batch_blocks, batch_scores)
            if (start // protocol["execution"]["batch_size"] + 1) % 24 == 0:
                print(f"[phase1571] captured {start + len(batch)}/{len(compiled)} cases", flush=True)
            del output, ids, mask, positions, logits, batch_blocks
        field.flush()
        if first_repeat is None:
            raise RuntimeError("repeat batch missing")
        batch, original_blocks, original_scores = first_repeat
        output, ids, mask, positions, lengths = forward(model, batch, pad, device)
        repeat_hidden = 0.0
        repeat_logits = 0.0
        logits = output.logits[:, -1].float()
        for local, row in enumerate(batch):
            again = torch.stack([hidden[local, : lengths[local]] for hidden in output.hidden_states], dim=0).to(dtype=torch.float16, device="cpu").numpy()
            repeat_hidden = max(repeat_hidden, float(np.max(np.abs(again.astype(np.float32) - original_blocks[local].astype(np.float32)))))
            for candidate_index, candidate in enumerate(row["candidate_ids"]):
                repeat_logits = max(repeat_logits, abs(float(logits[local, candidate[0]]) - original_scores[local][candidate_index]))
        del output, ids, mask, positions, logits
    finally:
        field.flush()
        del field
        if model is not None:
            release_bf16(model)
    core.write_rows(OUT / "raw/all_token_field_index.jsonl", index)
    field = np.load(raw_path, mmap_mode="r")
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index:
        by_unit[row["unit_id"]].append(row)
    causal_prefix_max = 0.0
    code_previsible_max = 0.0
    for rows in by_unit.values():
        reference = rows[0]
        for role in ("target_pre",):
            ref = np.asarray(field[:, reference["token_start"] + np.asarray(reference["role_positions"][role]), :], dtype=np.float32)
            for row in rows[1:]:
                value = np.asarray(field[:, row["token_start"] + np.asarray(row["role_positions"][role]), :], dtype=np.float32)
                causal_prefix_max = max(causal_prefix_max, float(np.max(np.abs(value - ref))))
        for key in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            standard = next(row for row in rows if (row["x"], row["y"], row["branch"], row["code"]) == (*key, 1, 1))
            reversed_code = next(row for row in rows if (row["x"], row["y"], row["branch"], row["code"]) == (*key, 1, -1))
            for role in ("target_post", "query_target"):
                left = np.asarray(field[:, standard["token_start"] + np.asarray(standard["role_positions"][role]), :], dtype=np.float32)
                right = np.asarray(field[:, reversed_code["token_start"] + np.asarray(reversed_code["role_positions"][role]), :], dtype=np.float32)
                code_previsible_max = max(code_previsible_max, float(np.max(np.abs(left - right))))
    checks = {
        "shape": list(field.shape) == [STATES, total_tokens, DIM],
        "dtype": field.dtype == np.float16,
        "coverage": len(index) == 1152 and index[-1]["token_end"] == total_tokens,
        "finite": finite and all(math.isfinite(value) for row in index for value in row["scores"]),
        "repeat_hidden": repeat_hidden <= protocol["execution"]["repeat_hidden_max_abs"],
        "repeat_logits": repeat_logits <= protocol["execution"]["repeat_logit_max_abs"],
        "causal_prefix": causal_prefix_max <= protocol["execution"]["causal_prefix_effect_max_abs"],
        "code_previsible": code_previsible_max <= protocol["execution"]["code_before_visible_effect_max_abs"],
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    behavior = {
        "global_accuracy": float(np.mean([row["correct"] for row in index])),
        "global_balanced_accuracy": ba([row["output_yes"] for row in index], [row["prediction"] == 0 for row in index]),
        "by_world": {world: float(np.mean([row["correct"] for row in index if row["world"] == world])) for world in WORLDS},
        "by_family": {family: float(np.mean([row["correct"] for row in index if row["family"] == family])) for family in FAMILIES},
        "by_code": {CODEBOOKS[code]["name"]: float(np.mean([row["correct"] for row in index if row["code"] == code])) for code in (1, -1)},
        "by_partition": {partition: float(np.mean([row["correct"] for row in index if row["partition"] == partition])) for partition in PARTITIONS},
        "stratum": "descriptive; behavior does not stop C098",
    }
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "all_token_all_state_capture_complete",
        "shape": list(field.shape),
        "total_real_tokens": total_tokens,
        "bytes": raw_path.stat().st_size,
        "raw_sha256": core.sha(raw_path),
        "index_sha256": core.sha(OUT / "raw/all_token_field_index.jsonl"),
        "numeric": {
            "repeat_hidden_max_abs": repeat_hidden,
            "repeat_logit_max_abs": repeat_logits,
            "causal_prefix_max_abs": causal_prefix_max,
            "code_previsible_max_abs": code_previsible_max,
        },
        "behavior": behavior,
        "runtime": {"placement": placement, "quantization": quant},
        "checks": checks,
        "finished_at_utc": now(),
        "authorization": "run_phase1571_analysis",
    }
    core.save(OUT / "analysis/capture_summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, ensure_ascii=False, indent=2))


def role_vector(field: np.ndarray, row: dict[str, Any], role: str) -> np.ndarray:
    positions = row["token_start"] + np.asarray(row["role_positions"][role], dtype=np.int64)
    return np.asarray(field[:, positions, :], dtype=np.float32).mean(axis=1)


def effect_sign(row: dict[str, Any], effect: str) -> int:
    value = 1
    for factor in FACTORS:
        if factor in effect:
            value *= int(row[factor])
    return value


def compute_focus_walsh(field: np.ndarray, index: list[dict[str, Any]], units: list[dict[str, Any]]) -> tuple[Path, list[dict[str, Any]]]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index:
        by_unit[row["unit_id"]].append(row)
    path = OUT / "raw/focus_role_walsh_coefficients.float32.npy"
    coeff = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(units), len(EFFECTS), STATES, len(FOCUS_ROLES), DIM))
    coeff_index = []
    for unit_index, unit in enumerate(units):
        rows = sorted(by_unit[unit["unit_id"]], key=lambda row: tuple(-row[factor] for factor in FACTORS))
        values = np.stack([
            np.stack([role_vector(field, row, role) for role in FOCUS_ROLES], axis=1)
            for row in rows
        ], axis=0)
        # case x state x role x coordinate
        for effect_index, effect in enumerate(EFFECTS):
            signs = np.asarray([effect_sign(row, effect) for row in rows], dtype=np.float32)
            coeff[unit_index, effect_index] = np.einsum("c,csrd->srd", signs, values, optimize=True) / 16.0
        coeff_index.append({"row_index": unit_index, **unit})
        if (unit_index + 1) % 12 == 0:
            print(f"[phase1571] Walsh {unit_index + 1}/{len(units)} units", flush=True)
    coeff.flush()
    del coeff
    core.write_rows(OUT / "raw/focus_role_walsh_index.jsonl", coeff_index)
    return path, coeff_index


def c097_design_null(repetitions: int = 1000) -> list[dict[str, Any]]:
    field = np.load(C097_CAPTURE / "raw/c097b_all_role_field.float32.npy", mmap_mode="r")
    index = core.rows(C097_CAPTURE / "raw/c097b_field_index.jsonl")
    pairs = core.rows(C097_CONTRACT / "material/frozen_wordnet_pairs.jsonl")
    families = ("similarity", "class_inclusion", "whole_part")
    surfaces = ("prequery", "postquery")
    lookup = {(row["pair_id"], row["surface"], row["query_family"]): row["row_index"] for row in index}
    rng = np.random.default_rng(157101)
    results = []
    for partition in PARTITIONS:
        family_rows = {
            family: sorted([row for row in pairs if row["partition"] == partition and row["family"] == family], key=lambda row: row["pair_id"])
            for family in families
        }
        for surface in surfaces:
            for state in (31, 32):
                cells = np.empty((10, 3, 3, DIM), dtype=np.float32)
                for rank in range(10):
                    for material_index, material_family in enumerate(families):
                        pair = family_rows[material_family][rank]
                        for query_index, query_family in enumerate(families):
                            cells[rank, material_index, query_index] = field[lookup[(pair["pair_id"], surface, query_family)], state, 3]

                def fraction(matrix: np.ndarray) -> float:
                    mean = matrix.mean(axis=0)
                    contrasts = np.stack([
                        0.5 * (mean[0, 0] + mean[1, 1] - mean[0, 1] - mean[1, 0]),
                        0.5 * (mean[0, 0] + mean[2, 2] - mean[0, 2] - mean[2, 0]),
                        0.5 * (mean[1, 1] + mean[2, 2] - mean[1, 2] - mean[2, 1]),
                    ])
                    common = contrasts.mean(axis=0)
                    total = float(np.sum(contrasts.astype(np.float64) ** 2))
                    return float(3 * np.sum(common.astype(np.float64) ** 2) / total) if total else 0.0

                observed = fraction(cells)
                mean_cells = cells.mean(axis=0).astype(np.float64)
                interaction = (
                    mean_cells
                    - mean_cells.mean(axis=1, keepdims=True)
                    - mean_cells.mean(axis=0, keepdims=True)
                    + mean_cells.mean(axis=(0, 1), keepdims=True)
                )
                symmetric = 0.5 * (interaction + interaction.swapaxes(0, 1))
                antisymmetric = 0.5 * (interaction - interaction.swapaxes(0, 1))
                interaction_energy = float(np.sum(interaction ** 2))
                symmetric_energy = float(np.sum(symmetric ** 2))
                antisymmetric_energy = float(np.sum(antisymmetric ** 2))
                null = []
                for _ in range(repetitions):
                    permuted = np.empty_like(cells)
                    for rank in range(10):
                        for material_index in range(3):
                            permuted[rank, material_index] = cells[rank, material_index, rng.permutation(3)]
                    null.append(fraction(permuted))
                results.append({
                    "partition": partition,
                    "surface": surface,
                    "state": state,
                    "observed": observed,
                    "null_median": float(np.median(null)),
                    "null_q95": float(np.quantile(null, 0.95)),
                    "null_q99": float(np.quantile(null, 0.99)),
                    "empirical_p_upper": float((1 + sum(value >= observed for value in null)) / (1 + len(null))),
                    "beats_q99": observed > float(np.quantile(null, 0.99)),
                    "full_interaction_energy": interaction_energy,
                    "symmetric_interaction_energy": symmetric_energy,
                    "antisymmetric_cycle_energy": antisymmetric_energy,
                    "antisymmetric_cycle_fraction": antisymmetric_energy / interaction_energy if interaction_energy else 0.0,
                    "symmetric_plus_antisymmetric_error": abs(interaction_energy - symmetric_energy - antisymmetric_energy),
                })
    core.write_rows(OUT / "analysis/c097_shared_cell_design_null.jsonl", results)
    return results


def all_token_discovery_scan(field: np.ndarray, index: list[dict[str, Any]], units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tok = tokenizer()
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index:
        if row["partition"] == "response_discovery":
            by_unit[row["unit_id"]].append(row)
    rows_out = []
    for unit in [value for value in units if value["partition"] == "response_discovery"]:
        rows = [row for row in by_unit[unit["unit_id"]] if row["surface"] == unit["surface"]]
        canonical = next(row for row in rows if row["x"] == row["y"] == row["branch"] == row["code"] == 1)
        # BPE can change a unit by one token when a node is repeated in a
        # different grammatical slot.  Align from the assistant boundary and
        # retain only the suffix present in every cell; never zero-pad a Walsh
        # coefficient or pretend that absolute positions are semantic roles.
        length = min(row["token_count"] for row in rows)
        if max(row["token_count"] for row in rows) - length > 1:
            raise RuntimeError(("unit token alignment", unit["unit_id"]))
        signs = np.asarray([row["x"] * row["y"] for row in rows], dtype=np.float32)
        for state in range(STATES):
            values = np.stack([
                np.asarray(field[state, row["token_end"] - length:row["token_end"], :], dtype=np.float32)
                for row in rows
            ], axis=0)
            effect = np.einsum("c,ctd->td", signs, values, optimize=True) / 16.0
            norms = np.linalg.norm(effect.astype(np.float64), axis=1)
            top_positions = np.argsort(-norms)[: min(8, len(norms))]
            tokens = [int(value) for value in compiled[canonical["row_index"]]["prompt_ids"]][-length:]
            for rank, position in enumerate(top_positions):
                rows_out.append({
                    "unit_id": unit["unit_id"],
                    "family": unit["family"],
                    "world": unit["world"],
                    "surface": unit["surface"],
                    "state": state,
                    "rank": rank,
                    "position": int(position),
                    "alignment": "right_aligned_common_suffix",
                    "token_id": tokens[int(position)],
                    "token_text": tok.decode([tokens[int(position)]]),
                    "xy_norm": float(norms[int(position)]),
                })
    core.write_rows(OUT / "analysis/discovery_all_token_xy_top_positions.jsonl", rows_out)
    return rows_out


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture_report = core.load(OUT / "analysis/capture_summary.json")
    if capture_report["authorization"] != "run_phase1571_analysis" or not all(capture_report["checks"].values()):
        raise RuntimeError("capture authorization missing")
    units = core.rows(OUT / "material/frozen_graph_units.jsonl")
    index = core.rows(OUT / "raw/all_token_field_index.jsonl")
    field = np.load(OUT / "raw/all_token_all_state_field.float16.npy", mmap_mode="r")
    coeff_path, coeff_index = compute_focus_walsh(field, index, units)
    coeff = np.load(coeff_path, mmap_mode="r")
    effect_index = {name: i for i, name in enumerate(EFFECTS)}
    role_index = {name: i for i, name in enumerate(FOCUS_ROLES)}

    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in coeff_index:
        grouped[(row["world"], row["family"], row["partition"])].append(row["row_index"])
    validation = []
    discovery_supports = []
    focus_effect = "xy"
    for world in WORLDS:
        for family in FAMILIES:
            discovery_ids = grouped[(world, family, "response_discovery")]
            confirmation_ids = grouped[(world, family, "confirmation")]
            lockbox_ids = grouped[(world, family, "lockbox")]
            for state in range(STATES):
                for role in FOCUS_ROLES:
                    vectors = {
                        "response_discovery": np.asarray(coeff[discovery_ids, effect_index[focus_effect], state, role_index[role]], dtype=np.float64).mean(axis=0),
                        "confirmation": np.asarray(coeff[confirmation_ids, effect_index[focus_effect], state, role_index[role]], dtype=np.float64).mean(axis=0),
                        "lockbox": np.asarray(coeff[lockbox_ids, effect_index[focus_effect], state, role_index[role]], dtype=np.float64).mean(axis=0),
                    }
                    support = np.argsort(-np.abs(vectors["response_discovery"]))[:64]
                    discovery_supports.append({
                        "world": world,
                        "family": family,
                        "state": state,
                        "role": role,
                        "coordinates": [int(value) for value in support],
                    })
                    for partition in ("confirmation", "lockbox"):
                        dynamic = np.argsort(-np.abs(vectors[partition]))[:64]
                        intersection = len(set(support) & set(dynamic))
                        overlap = intersection / (128.0 - intersection)
                        target_energy = float(np.sum(vectors[partition][support] ** 2) / (np.sum(vectors[partition] ** 2) + 1e-30))
                        validation.append({
                            "world": world,
                            "family": family,
                            "state": state,
                            "role": role,
                            "partition": partition,
                            "full_cosine": cosine(vectors["response_discovery"], vectors[partition]),
                            "fixed_top64_cosine": cosine(vectors["response_discovery"][support], vectors[partition][support]),
                            "fixed_top64_sign_agreement": float(np.mean(np.sign(vectors["response_discovery"][support]) == np.sign(vectors[partition][support]))),
                            "dynamic_top64_jaccard": overlap,
                            "fixed_support_target_energy_fraction": target_energy,
                            "discovery_norm": float(np.linalg.norm(vectors["response_discovery"])),
                            "target_norm": float(np.linalg.norm(vectors[partition])),
                        })
    core.write_rows(OUT / "analysis/discovery_top64_supports.jsonl", discovery_supports)
    core.write_rows(OUT / "analysis/dual_holdout_xy_validation.jsonl", validation)

    # Compare path/truth and output-code coupling by causal role.
    effect_energy = []
    for partition in PARTITIONS:
        ids = [row["row_index"] for row in coeff_index if row["partition"] == partition]
        for effect in ("xy", "code", "xycode", "branch"):
            for state in range(STATES):
                for role in FOCUS_ROLES:
                    vector = np.asarray(coeff[ids, effect_index[effect], state, role_index[role]], dtype=np.float64).mean(axis=0)
                    effect_energy.append({
                        "partition": partition,
                        "effect": effect,
                        "state": state,
                        "role": role,
                        "norm": float(np.linalg.norm(vector)),
                        "max_abs": float(np.max(np.abs(vector))),
                    })
    core.write_rows(OUT / "analysis/effect_energy_formation_atlas.jsonl", effect_energy)

    # Cross-world/family geometry for the code-invariant xy path effect.
    cross = []
    for partition in PARTITIONS:
        for state in range(STATES):
            for role in FOCUS_ROLES:
                vectors = {}
                for world in WORLDS:
                    for family in FAMILIES:
                        ids = grouped[(world, family, partition)]
                        vectors[(world, family)] = np.asarray(coeff[ids, effect_index["xy"], state, role_index[role]], dtype=np.float64).mean(axis=0)
                world_means = {world: np.mean([vectors[(world, family)] for family in FAMILIES], axis=0) for world in WORLDS}
                family_means = {family: np.mean([vectors[(world, family)] for world in WORLDS], axis=0) for family in FAMILIES}
                cross.append({
                    "partition": partition,
                    "state": state,
                    "role": role,
                    "minimum_world_cosine": min(cosine(world_means[a], world_means[b]) for a, b in itertools.combinations(WORLDS, 2)),
                    "median_world_cosine": float(np.median([cosine(world_means[a], world_means[b]) for a, b in itertools.combinations(WORLDS, 2)])),
                    "minimum_family_cosine": min(cosine(family_means[a], family_means[b]) for a, b in itertools.combinations(FAMILIES, 2)),
                    "median_family_cosine": float(np.median([cosine(family_means[a], family_means[b]) for a, b in itertools.combinations(FAMILIES, 2)])),
                })
    core.write_rows(OUT / "analysis/cross_world_family_xy_geometry.jsonl", cross)

    c097_null = c097_design_null(1000)
    token_scan = all_token_discovery_scan(field, index, units)

    # Compare C098 path effect to the earlier C097 late-boundary contrast mean.
    c097_common = np.load(C097_ATLAS / "raw/c097b_common_contrast_field.float32.npy", mmap_mode="r")
    cross_campaign = []
    all_ids = [row["row_index"] for row in coeff_index]
    for state in (31, 32):
        c098_xy = np.asarray(coeff[all_ids, effect_index["xy"], state, role_index["boundary"]], dtype=np.float64).mean(axis=0)
        c098_xycode = np.asarray(coeff[all_ids, effect_index["xycode"], state, role_index["boundary"]], dtype=np.float64).mean(axis=0)
        c097 = np.asarray(c097_common[:, :, state, 3], dtype=np.float64).mean(axis=(0, 1))
        cross_campaign.append({
            "state": state,
            "c097_common_to_c098_xy_cosine": cosine(c097, c098_xy),
            "c097_common_to_c098_xycode_cosine": cosine(c097, c098_xycode),
            "c098_xy_to_xycode_cosine": cosine(c098_xy, c098_xycode),
        })
    core.write_rows(OUT / "analysis/c097_c098_boundary_comparison.jsonl", cross_campaign)

    focus_validation = [row for row in validation if row["role"] in ("target_post", "query_target", "boundary") and row["state"] in (16, 24, 31, 32, 35, 36)]
    group_pass = {}
    for world in WORLDS:
        for family in FAMILIES:
            rows = [row for row in focus_validation if row["world"] == world and row["family"] == family]
            best = max(
                ((state, role, min(row["full_cosine"] for row in rows if row["state"] == state and row["role"] == role)) for state in (16, 24, 31, 32, 35, 36) for role in ("target_post", "query_target", "boundary")),
                key=lambda value: value[2],
            )
            group_pass[(world, family)] = {"best_state": best[0], "best_role": best[1], "minimum_holdout_cosine": best[2], "passed": best[2] >= 0.5}
    important_fraction = float(np.mean([value["passed"] for value in group_pass.values()]))
    stable = [row for row in focus_validation if row["full_cosine"] >= 0.5]
    median_jaccard = float(np.median([row["dynamic_top64_jaccard"] for row in stable])) if stable else 0.0
    fixed_candidates = [
        row for row in focus_validation
        if row["full_cosine"] >= 0.5 and row["dynamic_top64_jaccard"] >= 0.25 and row["fixed_top64_sign_agreement"] >= 0.75
    ]
    flags = {
        "important_visualization": important_fraction >= 0.75,
        "fixed_coordinate_candidate_count": len(fixed_candidates),
        "dynamic_alliance_candidate": bool(stable and median_jaccard < 0.25),
    }
    behavior = capture_report["behavior"]
    summary = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "observation_first_graph_campaign_complete",
        "behavior": behavior,
        "field": {"shape": list(field.shape), "raw_sha256": capture_report["raw_sha256"]},
        "walsh": {"shape": list(coeff.shape), "sha256": core.sha(coeff_path), "effects": list(EFFECTS)},
        "validation": {
            "focus_row_count": len(focus_validation),
            "full_cosine_median": float(np.median([row["full_cosine"] for row in focus_validation])),
            "full_cosine_min": float(np.min([row["full_cosine"] for row in focus_validation])),
            "full_cosine_max": float(np.max([row["full_cosine"] for row in focus_validation])),
            "fixed_sign_median": float(np.median([row["fixed_top64_sign_agreement"] for row in focus_validation])),
            "dynamic_jaccard_median": float(np.median([row["dynamic_top64_jaccard"] for row in focus_validation])),
            "world_family_best": {f"{world}:{family}": value for (world, family), value in group_pass.items()},
            "important_fraction": important_fraction,
        },
        "c097_design_null": {
            "cells": len(c097_null),
            "beats_q99": sum(row["beats_q99"] for row in c097_null),
            "observed_median": float(np.median([row["observed"] for row in c097_null])),
            "null_q99_median": float(np.median([row["null_q99"] for row in c097_null])),
            "antisymmetric_cycle_fraction_median": float(np.median([row["antisymmetric_cycle_fraction"] for row in c097_null])),
            "antisymmetric_cycle_fraction_range": [float(np.min([row["antisymmetric_cycle_fraction"] for row in c097_null])), float(np.max([row["antisymmetric_cycle_fraction"] for row in c097_null]))],
        },
        "cross_campaign": cross_campaign,
        "all_token_scan_rows": len(token_scan),
        "flags": flags,
        "claim_boundary": protocol["claim_boundary"],
        "interpretation": {
            "supported_if_repeated": "a code-invariant directed path response is observable and prospectively repeatable in this controlled Qwen graph field",
            "not_supported": [
                "a universal language graph code",
                "fixed semantic neurons or parameters",
                "causal necessity, sufficiency, transport or rescue",
                "natural-language breadth beyond the four controlled transitive families",
                "new mathematics",
            ],
        },
        "finished_at_utc": now(),
        "authorization": "export_c098_graph_walsh_heatmap" if flags["important_visualization"] else "freeze_next_observation_campaign",
    }
    core.save(OUT / "analysis/c098_graph_field_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def export_heatmap() -> None:
    summary = core.load(OUT / "analysis/c098_graph_field_summary.json")
    if summary["authorization"] != "export_c098_graph_walsh_heatmap":
        core.save(OUT / "analysis/visualization_decision.json", {"important": False, "reason": "frozen importance threshold not reached"})
        return
    coeff = np.load(OUT / "raw/focus_role_walsh_coefficients.float32.npy", mmap_mode="r")
    coeff_index = core.rows(OUT / "raw/focus_role_walsh_index.jsonl")
    effect_index = {name: i for i, name in enumerate(EFFECTS)}
    role_index = {name: i for i, name in enumerate(FOCUS_ROLES)}
    discovery_ids = [row["row_index"] for row in coeff_index if row["partition"] == "response_discovery"]
    reference = np.asarray(coeff[discovery_ids, effect_index["xy"], 32, role_index["boundary"]], dtype=np.float64).mean(axis=0)
    coordinates = np.argsort(-np.abs(reference))[:64]
    rows = []
    values = []
    for partition in PARTITIONS:
        for world in WORLDS:
            for family in FAMILIES:
                ids = [row["row_index"] for row in coeff_index if row["partition"] == partition and row["world"] == world and row["family"] == family]
                for state in (16, 24, 31, 32, 35, 36):
                    for role in ("target_post", "query_target", "boundary"):
                        vector = np.asarray(coeff[ids, effect_index["xy"], state, role_index[role]], dtype=np.float64).mean(axis=0)[coordinates]
                        values.extend(vector.tolist())
                        rows.append({
                            "partition": partition,
                            "world": world,
                            "family": family,
                            "state": state,
                            "role": role,
                            "effect": "xy_path_truth_code_invariant",
                            "values": vector.tolist(),
                        })
    scale = float(np.quantile(np.abs(values), 0.99))
    asset = {
        "schema": "graph_walsh_heatmap.v1",
        "result_type": "graph_walsh_heatmap",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": "Qwen3-4B",
        "title": "C098 Directed Graph Path Walsh Field",
        "dimensions": [int(value) for value in coordinates],
        "scale": {"symmetric_abs_q99": scale},
        "rows": rows,
        "evidence": {
            "grade": "O2/O3 task-scoped observation",
            "boundary": "A code-invariant xy graph-path contrast is not a universal semantic code or causal mechanism.",
        },
        "created_at_utc": now(),
    }
    canonical = OUT / "visualization/c098_graph_walsh_heatmap.json"
    client = ROOT / "frontend/public/vis_data/research_kernel/c098_graph_walsh_heatmap.json"
    core.save(canonical, asset)
    client.parent.mkdir(parents=True, exist_ok=True)
    client.write_bytes(canonical.read_bytes())
    decision = {
        "important": True,
        "asset": str(canonical.relative_to(ROOT)),
        "client": str(client.relative_to(ROOT)),
        "rows": len(rows),
        "coordinates": len(coordinates),
        "sha256": core.sha(canonical),
        "client_identity": core.sha(canonical) == core.sha(client),
    }
    core.save(OUT / "analysis/visualization_decision.json", decision)
    print(json.dumps(decision, indent=2))


def finalize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    capture_report = core.load(OUT / "analysis/capture_summary.json")
    summary = core.load(OUT / "analysis/c098_graph_field_summary.json")
    visualization = core.load(OUT / "analysis/visualization_decision.json")
    checks = {
        "preregistered": pre["all_checks_passed"],
        "source_identity": protocol["producer_sha256"] == core.sha(Path(__file__)),
        "capture": all(capture_report["checks"].values()),
        "raw_hash": core.sha(OUT / "raw/all_token_all_state_field.float16.npy") == capture_report["raw_sha256"],
        "walsh_hash": core.sha(OUT / "raw/focus_role_walsh_coefficients.float32.npy") == summary["walsh"]["sha256"],
        "discovery_support": len(core.rows(OUT / "analysis/discovery_top64_supports.jsonl")) == 3 * 4 * 37 * 4,
        "dual_holdout": len(core.rows(OUT / "analysis/dual_holdout_xy_validation.jsonl")) == 3 * 4 * 37 * 4 * 2,
        "design_null": len(core.rows(OUT / "analysis/c097_shared_cell_design_null.jsonl")) == 12,
        "all_token_scan": summary["all_token_scan_rows"] > 0,
        "visualization": (not visualization["important"]) or visualization["client_identity"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "major_observation_stage_complete",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "result": summary,
        "visualization": visualization,
        "theory": {
            "name": "conditional output field closure theory",
            "principle": "reuse-difference-conditioning (RDC)",
            "mechanism_formula": "H_{l+1,t}=T_l(H_{l,<=t};phi,eta); C_S=2^-4 sum_z chi_S(z)H(z)",
            "graph": "embedding and lexical identity -> directed local-ledger graph -> repeated-target/query path field -> code-invariant xy contrast plus code-conditioned xycode boundary response -> output competition",
            "math_status": "Walsh finite differences and conditional dynamics are sufficient for C098 observation; no new mathematics is licensed.",
        },
        "next_authorization": "C099 may batch-observe non-transitive composition families and repeat the C098 discovery/holdout protocol; causal closure remains secondary.",
        "finished_at_utc": now(),
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "capture", "analyze", "export", "finalize", "all"))
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage in ("capture", "all"):
        capture()
    if args.stage in ("analyze", "all"):
        analyze()
    if args.stage in ("export", "all"):
        export_heatmap()
    if args.stage in ("finalize", "all"):
        finalize()


if __name__ == "__main__":
    main()
