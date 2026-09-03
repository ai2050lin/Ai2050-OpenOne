#!/usr/bin/env python3
"""C244: independent event-rule replication and scaffold-neutralized C242 diagnostics."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.RESULT / "phase1778_c244_independent_event_replication"
C235 = common.OUTS["C235"]
C236 = common.OUTS["C236"]
C237 = common.OUTS["C237"]
C242 = common.OUTS["C242"]
WIDTH = 128
BATCH = 2
SURFACES = ("field_memo", "public_hearing")

UNITS = (
    {"primary": "Nolan", "secondary": "Petra", "observer": "Ravi", "object": "quince", "other": "compass", "node": "quorin", "middle": "frutex", "parent": "produce", "wrong": "vehicle"},
    {"primary": "Selma", "secondary": "Tarin", "observer": "Uma", "object": "okra", "other": "goblet", "node": "okarin", "middle": "podlex", "parent": "vegetable", "wrong": "instrument"},
    {"primary": "Vanya", "secondary": "Wes", "observer": "Xara", "object": "lychee", "other": "hammer", "node": "lyrin", "middle": "fruval", "parent": "food", "wrong": "building"},
    {"primary": "Yuna", "secondary": "Zorin", "observer": "Alma", "object": "parsnip", "other": "mirror", "node": "parvik", "middle": "rootal", "parent": "plant", "wrong": "weather"},
    {"primary": "Belen", "secondary": "Cyrus", "observer": "Dara", "object": "fig", "other": "saddle", "node": "figrin", "middle": "orchal", "parent": "organism", "wrong": "metal"},
    {"primary": "Erena", "secondary": "Farid", "observer": "Gita", "object": "leek", "other": "tripod", "node": "leorin", "middle": "stalkal", "parent": "entity", "wrong": "motion"},
)


def wrap(surface: str, fact1: str, fact2: str, question: str) -> str:
    if surface == "field_memo":
        return f"A field memo states: {fact1} A separate line states: {fact2} Question: {question}"
    if surface == "public_hearing":
        return f'At a public hearing, the first testimony was: "{fact1}" The second was: "{fact2}" Decide: {question}'
    raise KeyError(surface)


def semantic_case(family: str, surface: str, unit: int, a: int, b: int) -> dict:
    u = UNITS[unit]
    p, s, o = u["primary"], u["secondary"], u["observer"]
    obj, other = u["object"], u["other"]
    node, middle, parent, wrong = u["node"], u["middle"], u["parent"], u["wrong"]
    if family == "attitude_event":
        relation = "endorses" if a == 0 else "questions"
        target = f"{o} {relation} the report that {p} inspected the {obj}." if b == 0 else f"{o} {relation} the report that the {obj} was inspected by {p}."
        noise = f"{s} catalogued the {other}."
        question, correct, distractor = f"Who inspected the {obj}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": obj}
    elif family == "type_graph":
        relation = "is classified as"
        if a == 0:
            target = f"The {node} is classified as {parent}."
            noise = f"The {middle} is classified as {wrong}."
        else:
            target = f"The {node} is classified as {middle}."
            noise = f"The {middle} is classified as {parent}."
        if b:
            target += f" A registry also lists the {node} directly as {parent}."
        question, correct, distractor = f"What final class contains the {node}?", parent, wrong
        roles = {"primary": node, "secondary": middle, "relation": relation, "context": parent, "query": node}
    elif family == "contrast":
        if a == 0:
            relation = "yet"
            target = f"{p} stayed calm, yet {s} became anxious." if b == 0 else f"{s} became anxious, yet {p} stayed calm."
        else:
            relation = "Even though" if b == 0 else "even though"
            target = f"Even though {s} became anxious, {p} stayed calm." if b == 0 else f"{p} stayed calm even though {s} became anxious."
        noise = f"The {obj} remained on the table."
        question, correct, distractor = "Who stayed calm?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": "calm"}
    elif family == "translation":
        relation = "stands for"
        if a == 0:
            target = f'In this lexicon, "{node}" stands for "{parent}".'
            noise = f'"{middle}" stands for "{wrong}".'
        else:
            target = f'In this lexicon, "{node}" stands for "{middle}".'
            noise = f'"{middle}" stands for "{parent}".'
        if b:
            target += f' A margin note translates "{node}" directly as "{parent}".'
        question, correct, distractor = f'What does "{node}" finally stand for?', parent, wrong
        roles = {"primary": node, "secondary": middle, "relation": relation, "context": parent, "query": node}
    elif family == "comparison":
        dimension = "faster" if a == 0 else "older"
        inverse = "slower" if a == 0 else "younger"
        relation = dimension if b == 0 else inverse
        target = f"{p} is {dimension} than {s}." if b == 0 else f"{s} is {inverse} than {p}."
        noise = f"The {obj} is near the {other}."
        question, correct, distractor = f"Who is {dimension}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": s, "query": dimension}
    else:
        raise KeyError(family)
    return {"prompt_core": wrap(surface, target, noise, question), "correct": correct, "wrong": distractor, "roles": roles}


def material() -> list[dict]:
    rows = []
    for surface, family, unit, a, b, order in itertools.product(SURFACES, common.FAMILIES, range(len(UNITS)), (0, 1), (0, 1), (1, -1)):
        if (surface == "field_memo") != (unit < 3):
            continue
        case = semantic_case(family, surface, unit, a, b)
        choices, gold = common.options(case["correct"], case["wrong"], order)
        prompt_core = case["prompt_core"]
        rows.append({
            "case_id": f"c244-{family}-{surface}-u{unit}-{a}{b}-{order:+d}",
            "family": family, "surface": surface, "partition": "independent", "unit": unit,
            "factor_a": a, "factor_b": b, "order": order, "gold_position": gold,
            "correct_answer": case["correct"], "wrong_answer": case["wrong"],
            "prompt_core": prompt_core,
            "prompt": f"{prompt_core} {choices}. Reply with only A or B.",
            "free_prompt": f"{prompt_core} Answer with only the answer word.",
            "role_values": case["roles"],
        })
    return rows


def compile_rows(rows: list[dict]) -> list[dict]:
    tokenizer = common.graph_base.tokenizer()
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(value) != 1 for value in candidates):
        raise RuntimeError(candidates)
    system = "Answer only from the supplied text. Do not use outside knowledge."
    compiled = []
    for row in rows:
        ids = core.chat_ids(tokenizer, system, row["prompt"])
        free_ids = core.chat_ids(tokenizer, system, row["free_prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = common.graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "free_prompt_ids": free_ids, "candidate_ids": candidates, "role_positions": positions})
    return compiled


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C243"] / "audit/independent_final_audit.json")
    rows = material()
    compiled = compile_rows(rows)
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C244"),
        "rows": len(rows) == 240,
        "candidate_balance": sum(row["gold_position"] == 0 for row in rows) == 120,
        "new_surfaces": not ({row["surface"] for row in rows} & set(common.SURFACES)),
        "new_units": not ({u["primary"] for u in UNITS} & {u["primary"] for u in common.UNITS}),
        "unique_prompts": len({row["prompt"] for row in rows}) == 240,
        "roles": all(set(row["role_positions"]) == set(common.ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) <= WIDTH,
        "human_blind_missing": True,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled)})
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": 1778, "campaign": "C244", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "independent_event_replication_frozen",
        "research_object": "prospective replication of C237 signed event rules on two new surfaces and six new lexical systems",
        "rows": 240, "families": list(common.FAMILIES), "surfaces": list(SURFACES),
        "field_shape": [240, 37, WIDTH, 2560],
        "behavior_gate": {"global_min": 0.85, "each_family_min": 0.70},
        "replication_gate": {"attitude_event_and_contrast_each_jaccard_min": 0.15, "each_all_control_margin_min": 0.02},
        "controls": ["best_wrong_family", "discovery_generic", "relation_only", "nearest_length_discovery_group", "zero"],
        "c242_diagnostic_transforms": ["no_embedding", "family_centered", "effect_centered", "double_centered", "depth_difference"],
        "diagnostic_gate": "at least three transforms retain all three model-pair cosines >=0.30 and role-permutation p<=0.05; retrospective diagnostic only",
        "naturalness": "controlled-English internal audit; independent human blind naturalness remains registered missingness",
        "forbidden": ["attention", "MLP", "weights", "PCA", "Top-K", "threshold refit", "rule refit", "project-level stop"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "capture_Qwen3_once_then_evaluate_frozen_rules_and_C242_controls",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "max_width": max(len(row["prompt_ids"]) for row in compiled)})
    print(json.dumps({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled)}, indent=2))


def batch_inputs(rows: list[dict], pad: int, device):
    ids = torch.full((len(rows), WIDTH), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    positions = torch.zeros_like(ids)
    lengths = []
    for i, row in enumerate(rows):
        values = row["prompt_ids"]
        lengths.append(len(values))
        ids[i, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[i, :len(values)] = 1
        positions[i, :len(values)] = torch.arange(len(values), device=device)
    return ids, mask, positions, lengths


@torch.inference_mode()
def capture() -> None:
    if (OUT / "raw/full_fields.float16.npy").exists():
        raise RuntimeError("already captured")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    fields = np.lib.format.open_memmap(OUT / "raw/full_fields.float16.npy", mode="w+", dtype=np.float16, shape=(240, 37, WIDTH, 2560))
    logits = np.zeros((240, 2), np.float32)
    index = []
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = common.previous.load_bf16("qwen3")
        quant = common.previous.quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            ids, mask, positions, lengths = batch_inputs(batch, pad, device)
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True, output_hidden_states=True)
            if len(output.hidden_states) != 37:
                raise RuntimeError(len(output.hidden_states))
            for local, row in enumerate(batch):
                i = start + local
                length = lengths[local]
                for q, state in enumerate(output.hidden_states):
                    fields[i, q, :length] = state[local, :length].float().cpu().numpy().astype(np.float16)
                logits[i] = [float(output.logits[local, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(logits[i, 1] > logits[i, 0])
                index.append({
                    "hidden_index": i, "case_id": row["case_id"], "family": row["family"], "surface": row["surface"], "unit": row["unit"],
                    "factor_a": row["factor_a"], "factor_b": row["factor_b"], "order": row["order"], "length": length,
                    "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"],
                    "role_positions": row["role_positions"],
                })
            del output, ids, mask, positions
            if start % 40 == 0 or start + len(batch) == len(rows):
                fields.flush()
                print(f"[C244] full fields {start + len(batch)}/{len(rows)}", flush=True)
        fields.flush()
        np.save(OUT / "raw/behavior_logits.float32.npy", logits)
        core.write_rows(OUT / "raw/hidden_index.jsonl", index)
        by_family = {family: float(np.mean([row["correct"] for row in index if row["family"] == family])) for family in common.FAMILIES}
        global_accuracy = float(np.mean([row["correct"] for row in index]))
        eligible = global_accuracy >= 0.85 and min(by_family.values()) >= 0.70
        report = {"global_accuracy": global_accuracy, "by_family_accuracy": by_family, "behavior_eligible": eligible, "placement": placement, "quantization": quant, "elapsed_seconds": time.time() - started, "field_bytes": int(fields.nbytes)}
        core.save(OUT / "analysis/behavior_capture.json", report)
        checks = {"rows": len(index) == 240, "shape": list(fields.shape) == [240, 37, 128, 2560], "finite": bool(np.isfinite(logits).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / "audit/internal_capture_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"report": report, "checks": checks}, indent=2))
    finally:
        common.previous.release(model)
        gc.collect()


def signed_jaccard(pred: np.ndarray, truth: np.ndarray) -> float:
    union = (pred != 0) | (truth != 0)
    return float(np.mean(pred[union] == truth[union])) if union.any() else 1.0


def evaluate_events() -> None:
    behavior = core.load(OUT / "analysis/behavior_capture.json")
    if not behavior["behavior_eligible"]:
        result = {"status": "behavior_ineligible", "event_rules_tested": False, "next": "continue_C242_diagnostics"}
        core.save(OUT / "analysis/event_replication.json", result)
        core.save(OUT / "audit/internal_event_audit.json", {"checks": {"typed_stop": True}, "all_checks_passed": True})
        print(json.dumps(result, indent=2)); return
    fields = np.load(OUT / "raw/full_fields.float16.npy", mmap_mode="r")
    index = core.rows(OUT / "raw/hidden_index.jsonl")
    key = {(row["family"], row["surface"], row["unit"], row["factor_a"], row["factor_b"], row["order"]): row for row in index}
    rules = np.load(C237 / "analysis/rule_codes.int8.npy", mmap_mode="r")
    thresholds = np.asarray(core.load(C236 / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    old_events = np.load(C237 / "raw/role_events.int8.npy", mmap_mode="r")
    old_groups = core.rows(C236 / "protocol/effect_groups.jsonl")
    old_hidden = core.rows(C235 / "raw/hidden_index.jsonl")
    old_lengths = {row["effect_index"]: float(np.mean([h["length"] for h in old_hidden if h["family"] == row["family"] and h["surface"] == row["surface"] and h["unit"] == row["unit"] and h["order"] == row["order"]])) for row in old_groups}
    discovery = [row["effect_index"] for row in old_groups if row["partition"] == "discovery"]
    disc = np.asarray(old_events[discovery])
    up, down = np.mean(disc == 1, axis=0), np.mean(disc == -1, axis=0)
    active = up + down
    generic = np.where((active >= 0.75) & (np.maximum(up, down) / np.maximum(active, 1e-9) >= 0.80), np.where(up >= down, 1, -1), 0).astype(np.int8)
    rows = []
    event_cache = {}
    for family, surface, unit, order in itertools.product(common.FAMILIES, SURFACES, range(len(UNITS)), (1, -1)):
        if (surface == "field_memo") != (unit < 3):
            continue
        cells = {}
        complete = True
        lengths = []
        for a, b in itertools.product((0, 1), repeat=2):
            row = key[(family, surface, unit, a, b, order)]
            complete &= row["correct"]
            lengths.append(row["length"])
            aligned = np.empty((37, 6, 2560), np.float32)
            state = np.asarray(fields[row["hidden_index"]], np.float32)
            for role_i, role in enumerate(common.ROLES):
                aligned[:, role_i] = state[:, row["role_positions"][role], :].mean(axis=1)
            cells[(a, b)] = aligned
        if not complete:
            continue
        values = common.factorial_effect(cells)
        events = np.where(values > thresholds[None, :, None, None], 1, np.where(values < -thresholds[None, :, None, None], -1, 0)).astype(np.int8)
        event_cache[(family, surface, unit, order)] = events
        nearest = min((idx for idx in discovery if old_groups[idx]["order"] == order), key=lambda idx: abs(old_lengths[idx] - float(np.mean(lengths))))
        family_i = common.FAMILIES.index(family)
        for effect_i, effect in enumerate(common.EFFECTS):
            truth = events[effect_i]
            correct = np.asarray(rules[family_i, effect_i])
            wrong = max(signed_jaccard(np.asarray(rules[i, effect_i]), truth) for i in range(5) if i != family_i)
            relation = np.zeros_like(correct); relation[:, common.ROLES.index("relation")] = correct[:, common.ROLES.index("relation")]
            controls = {
                "best_wrong_family": wrong,
                "discovery_generic": signed_jaccard(generic[effect_i], truth),
                "relation_only": signed_jaccard(relation, truth),
                "nearest_length": signed_jaccard(np.asarray(old_events[nearest, effect_i]), truth),
                "zero": signed_jaccard(np.zeros_like(correct), truth),
            }
            rows.append({"family": family, "surface": surface, "unit": unit, "order": order, "effect": effect, "correct_signed_jaccard": signed_jaccard(correct, truth), "controls": controls})
    core.write_rows(OUT / "analysis/event_rows.jsonl", rows)
    family_results = []
    for family in common.FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        correct = float(np.median([row["correct_signed_jaccard"] for row in selected])) if selected else 0.0
        control_values = {name: float(np.median([row["controls"][name] for row in selected])) for name in ("best_wrong_family", "discovery_generic", "relation_only", "nearest_length", "zero")} if selected else {}
        margin = correct - max(control_values.values()) if control_values else -1.0
        family_results.append({"family": family, "support": len(selected), "correct_signed_jaccard": correct, "controls": control_values, "all_control_margin": margin, "passed": correct >= 0.15 and margin >= 0.02})
    order_agreement = []
    for family, surface, unit in itertools.product(common.FAMILIES, SURFACES, range(len(UNITS))):
        if (surface == "field_memo") != (unit < 3) or (family, surface, unit, 1) not in event_cache or (family, surface, unit, -1) not in event_cache:
            continue
        left, right = event_cache[(family, surface, unit, 1)], event_cache[(family, surface, unit, -1)]
        union = (left != 0) | (right != 0)
        order_agreement.append({"family": family, "surface": surface, "unit": unit, "signed_agreement": float(np.mean(left[union] == right[union]))})
    target = {row["family"]: row for row in family_results}
    campaign_passed = all(target[name]["passed"] for name in ("attitude_event", "contrast"))
    report = {"status": "event_replication_adjudicated", "event_rules_tested": True, "family_results": family_results, "target_families_passed": campaign_passed, "candidate_order_signed_agreement_median": float(np.median([row["signed_agreement"] for row in order_agreement])), "complete_group_count": len(event_cache)}
    core.save(OUT / "analysis/event_replication.json", report)
    checks = {"rows": len(rows) == len(event_cache) * 3, "families": len(family_results) == 5, "finite": bool(np.isfinite([row["correct_signed_jaccard"] for row in rows]).all()), "no_refit": True}
    core.save(OUT / "audit/internal_event_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"report": report, "checks": checks}, indent=2))


def transform_graph(graph: np.ndarray, kind: str) -> np.ndarray:
    x = np.asarray(graph, np.float64)
    if kind == "no_embedding": x = x[:, :, 1:, :]
    elif kind == "family_centered": x = x - x.mean(axis=0, keepdims=True)
    elif kind == "effect_centered": x = x - x.mean(axis=1, keepdims=True)
    elif kind == "double_centered": x = x - x.mean(axis=0, keepdims=True) - x.mean(axis=1, keepdims=True) + x.mean(axis=(0, 1), keepdims=True)
    elif kind == "depth_difference": x = np.diff(x, axis=2)
    else: raise KeyError(kind)
    return x - x.mean(axis=-1, keepdims=True)


def c242_controls() -> None:
    source = core.load(C242 / "analysis/summary.json")
    transforms = core.load(OUT / "protocol/preregistration.json")["c242_diagnostic_transforms"]
    graphs = {name: np.asarray(value, np.float64) for name, value in source["graphs"].items()}
    permutations = list(itertools.permutations(range(6)))
    results = []
    for kind in transforms:
        transformed = {name: transform_graph(value, kind) for name, value in graphs.items()}
        pairs = []
        names = list(transformed)
        for i, left_name in enumerate(names):
            for right_name in names[i + 1:]:
                left, right = transformed[left_name], transformed[right_name]
                observed = common.cosine(left, right)
                null = np.asarray([common.cosine(left, right[..., permutation]) for permutation in permutations])
                pairs.append({"models": [left_name, right_name], "cosine": observed, "null_q95": float(np.quantile(null, 0.95)), "exact_upper_p": float((1 + np.sum(null >= observed)) / 721)})
        passed = min(row["cosine"] for row in pairs) >= 0.30 and max(row["exact_upper_p"] for row in pairs) <= 0.05
        results.append({"transform": kind, "pairs": pairs, "passed": passed})
    report = {"status": "retrospective_scaffold_control_diagnostic", "transforms": results, "transforms_passed": sum(row["passed"] for row in results), "diagnostic_gate_passed": sum(row["passed"] for row in results) >= 3, "claim_boundary": "These are frozen-transform reanalyses of revealed C242 data, not an independent cross-model replication."}
    core.save(OUT / "analysis/c242_scaffold_controls.json", report)
    checks = {"transforms": len(results) == 5, "pairs": all(len(row["pairs"]) == 3 for row in results), "finite": bool(np.isfinite([pair[key] for row in results for pair in row["pairs"] for key in ("cosine", "null_q95", "exact_upper_p")]).all())}
    core.save(OUT / "audit/internal_c242_controls_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"report": report, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior = core.load(OUT / "analysis/behavior_capture.json")
    events = core.load(OUT / "analysis/event_replication.json")
    controls = core.load(OUT / "analysis/c242_scaffold_controls.json")
    report = {
        "phase": 1778, "campaign": "C244", "status": "closed",
        "behavior": behavior, "event_replication": events, "cross_model_scaffold_diagnostic": controls,
        "strict_conclusion": "New-material event replication and retrospective cross-model controls are adjudicated separately; neither can by itself establish a causal or cross-model coordinate code.",
        "next_authorization": "C245 narrow rule revision only if replication diagnoses a prespecified stable target; otherwise independent natural-language observation expansion",
    }
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"], "events": core.load(OUT / "audit/internal_event_audit.json")["all_checks_passed"], "controls": core.load(OUT / "audit/internal_c242_controls_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1778, "campaign": "C244", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "capture", "events", "controls", "close"))
    args = parser.parse_args()
    {"contract": contract, "capture": capture, "events": evaluate_events, "controls": c242_controls, "close": close}[args.command]()


if __name__ == "__main__":
    main()
