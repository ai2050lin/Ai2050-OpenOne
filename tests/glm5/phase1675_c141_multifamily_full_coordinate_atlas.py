#!/usr/bin/env python3
"""C141: five-family, full-token, full-coordinate Qwen3 observation atlas."""
from __future__ import annotations

import gc
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1675_c141_multifamily_full_coordinate_atlas"
C140 = RESULT / "phase1674_c140_identifiability_and_master_contract"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127

PHASE, CAMPAIGN = 1675, "C141"
ARMS = ("event_composition", "type_graph", "discourse", "translation", "comparison")
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
CHECKPOINTS = c127.CHECKPOINTS
DIM, WIDTH, BATCH = 2560, 256, 2
SYL = ("zaf", "yud", "xir", "wep", "voq", "utn", "sim", "ral", "qek", "poj", "nuv", "mox", "lir", "keg", "jaf", "huz")
VERBS = (("carry", "carried"), ("inspect", "inspected"), ("move", "moved"), ("paint", "painted"), ("clean", "cleaned"), ("repair", "repaired"), ("store", "stored"), ("measure", "measured"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fresh(i: int, j: int) -> str:
    return f"N{SYL[(i * 3 + j) % len(SYL)]}{SYL[(i * 7 + j * 5 + 1) % len(SYL)]}{i}{j}"


def option_text(left: str, right: str, code: int) -> tuple[str, int, dict[str, str]]:
    if code == 1:
        return f"A = {left}; B = {right}.", 0, {"A": left, "B": right}
    return f"A = {right}; B = {left}.", 1, {"A": right, "B": left}


def make_case(arm: str, unit: int, f1: int, f2: int, f3: int, surface: int, code: int) -> dict:
    v = [fresh(ARMS.index(arm) * 8 + unit, j) for j in range(10)]
    partition = "discovery" if unit < 4 else "confirmation"
    answer_index = 0 if f1 == 1 else 1
    role_values: dict[str, str]
    if arm == "event_composition":
        experiencer, a0, a1, p0, p1 = v[:5]
        agent = a0 if f1 == 1 else a1
        patient = p0 if f3 == 1 else p1
        base, past = VERBS[(unit + (f2 == -1)) % len(VERBS)]
        statement = (f"{experiencer} enjoys watching {agent} {base} {patient}." if surface == 1 else f"According to {experiencer}'s journal, {patient} was {past} by {agent}, and the event was enjoyable to witness.")
        options, pos0, mapping = option_text(a0, a1, code)
        prompt = f"{statement} Who performed the described action? {options} Reply with only A or B."
        gold = pos0 if answer_index == 0 else 1 - pos0
        role_values = {"primary": agent, "secondary": patient, "relation": base if surface == 1 else past, "context": experiencer, "query": "Who"}
    elif arm == "type_graph":
        source, b1, b2, t0, t1 = v[:5]
        intended = t0 if f1 == 1 else t1
        other = t1 if f1 == 1 else t0
        if f2 == 1:
            main = [(source, "is a kind of", intended)]
        else:
            main = [(source, "is a kind of", b1), (b1, "is a kind of", b2), (b2, "is a kind of", intended)]
        if f3 == -1:
            main[-1] = (main[-1][0], "is related to", main[-1][2])
            main.append((source, "is a kind of", other))
            answer_index = 1 - answer_index
        entries = [f"{a} {r} {b}" for a, r, b in main]
        statement = ("; ".join(entries) + "." if surface == 1 else "The registry contains these links: " + " | ".join(reversed(entries)) + ".")
        options, pos0, mapping = option_text(t0, t1, code)
        prompt = f"{statement} Using only 'is a kind of' links, which target is reachable from {source}? {options} Reply with only A or B."
        gold = pos0 if answer_index == 0 else 1 - pos0
        role_values = {"primary": source, "secondary": intended if f3 == 1 else other, "relation": "is a kind of", "context": b1 if f2 == -1 else intended, "query": source}
    elif arm == "discourse":
        person, c0, c1, premise0, premise1 = v[:5]
        conclusion = c0 if f1 == 1 else c1
        premise = premise0 if f3 == 1 else premise1
        connective = "Although" if f2 == 1 else "After"
        statement = (f"{connective} {person} considered {premise}, {person} ultimately chose {conclusion}." if surface == 1 else f"The report says that {person} ultimately chose {conclusion}; this {'contrasted with' if f2 == 1 else 'followed'} consideration of {premise}.")
        options, pos0, mapping = option_text(c0, c1, code)
        prompt = f"{statement} What did {person} ultimately choose? {options} Reply with only A or B."
        gold = pos0 if answer_index == 0 else 1 - pos0
        role_values = {"primary": person, "secondary": conclusion, "relation": connective if surface == 1 else ("contrasted" if f2 == 1 else "followed"), "context": premise, "query": person}
    elif arm == "translation":
        s0, s1, t0, t1, language = v[:5]
        source = s0 if f3 == 1 else s1
        target = t0 if f1 == 1 else t1
        if f2 == 1:
            statement = f"In the {language} glossary, '{source}' means '{target}'."
            relation = "means"
        else:
            statement = f"The {language} glossary translates '{target}' back as '{source}'."
            relation = "translates"
        if surface == -1:
            statement = (f"A verified {language} dictionary entry pairs source '{source}' with target '{target}'." if f2 == 1 else f"A verified {language} reverse dictionary entry pairs target '{target}' with source '{source}'.")
            relation = "pairs"
        options, pos0, mapping = option_text(t0, t1, code)
        prompt = f"{statement} Which target word corresponds to '{source}'? {options} Reply with only A or B."
        gold = pos0 if answer_index == 0 else 1 - pos0
        role_values = {"primary": source, "secondary": target, "relation": relation, "context": language, "query": source}
    elif arm == "comparison":
        e0, e1v = v[:2]
        winner = e0 if f1 == 1 else e1v
        loser = e1v if f1 == 1 else e0
        dimension = "size" if f2 == 1 else "weight"
        adjective = "larger" if f2 == 1 else "heavier"
        if f3 == 1:
            statement = f"On the recorded {dimension} comparison, {winner} is {adjective} than {loser}."
        else:
            statement = f"Compared with {loser}, {winner} has the greater recorded {dimension}."
            adjective = "greater"
        if surface == -1:
            statement = (f"The measurement note ranks {winner} above {loser} for {dimension}." if f3 == 1 else f"For {dimension}, the measurement note places {loser} below {winner}.")
            adjective = "above" if f3 == 1 else "below"
        options, pos0, mapping = option_text(e0, e1v, code)
        prompt = f"{statement} Which item ranks higher for {dimension}? {options} Reply with only A or B."
        gold = pos0 if answer_index == 0 else 1 - pos0
        role_values = {"primary": winner, "secondary": loser, "relation": adjective, "context": dimension, "query": dimension}
    else:
        raise KeyError(arm)
    return {
        "case_id": "",
        "unit_id": f"c141-{arm}-{unit:02d}",
        "arm": arm,
        "partition": partition,
        "factors": {"f1": f1, "f2": f2, "f3": f3},
        "surface_factor": surface,
        "codebook_factor": code,
        "semantic_answer_index": answer_index,
        "gold_position": gold,
        "prompt": prompt,
        "role_values": role_values,
        "option_mapping": mapping,
    }


def material() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for arm in ARMS:
        for unit in range(8):
            units.append({"unit_id": f"c141-{arm}-{unit:02d}", "arm": arm, "partition": "discovery" if unit < 4 else "confirmation"})
            for f1, f2, f3, surface, code in itertools.product((1, -1), repeat=5):
                row = make_case(arm, unit, f1, f2, f3, surface, code)
                row["case_id"] = f"c141-{len(cases):05d}"
                cases.append(row)
    return units, cases


def compile_rows(tokenizer, cases: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(ids) != 1 for ids in candidates):
        raise RuntimeError(candidates)
    rows = []
    system = "Read the supplied statement literally, solve the multiple-choice question, and output exactly A or B."
    for row in cases:
        ids = core.chat_ids(tokenizer, system, row["prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value, row["prompt"]))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        rows.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return rows


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C140 / "audit/independent_contract_audit.json")
    units, cases = material()
    compiled = compile_rows(graph_base.tokenizer(), cases)
    cells = {}
    for row in cases:
        key = (row["arm"], row["partition"], *(row["factors"][f"f{i}"] for i in range(1, 4)), row["surface_factor"], row["codebook_factor"])
        cells[key] = cells.get(key, 0) + 1
    gold = [row["gold_position"] for row in cases]
    total_tokens = sum(len(row["prompt_ids"]) for row in compiled)
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "run_C141_contract_and_capture",
        "units": len(units) == 40,
        "cases": len(cases) == 1280,
        "arms": all(sum(row["arm"] == arm for row in cases) == 256 for arm in ARMS),
        "cells": len(cells) == 320 and set(cells.values()) == {4},
        "output_balance": sum(value == 0 for value in gold) == sum(value == 1 for value in gold) == 640,
        "unique": len({row["prompt"] for row in cases}) == 1280,
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "naturalness": all("Reply with only A or B" in row["prompt"] and "{" not in row["prompt"] for row in cases),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "five_arm_full_coordinate_contract_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "arms": list(ARMS),
        "roles": list(ROLES),
        "units": 40,
        "cases": 1280,
        "factor_design": "3 semantic factors x 2 surfaces x 2 output codebooks per unit",
        "total_actual_tokens": total_tokens,
        "max_width": max(len(row["prompt_ids"]) for row in compiled),
        "expected_full_field_shape": [38, total_tokens, DIM],
        "expected_role_field_shape": [1280, 6, 38, DIM],
        "behavior_observation_thresholds": {"global": 0.80, "arm": 0.70, "partition": 0.70},
        "behavior_policy": "descriptive qualification only; capture is authoritative and continues for incorrect trajectories",
        "capture": "same authoritative forward produces behavior logits, all-token fields, and aligned-role fields",
        "naturalness": "controlled grammatical English with machine uniqueness audit; no independent human blind rating",
        "forbidden": ["attention", "MLP", "weights", "PCA", "post-unblind threshold changes"],
        "source_paths": {"C140": str(C140 / "audit/independent_contract_audit.json")},
        "source_hashes": {"C140": core.sha(C140 / "audit/independent_contract_audit.json")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_authoritative_qwen3_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "total_tokens": total_tokens, "max_width": protocol["max_width"], "estimated_full_field_bytes": 38 * total_tokens * DIM * 2}, indent=2))


def tensor_output(value):
    return value[0] if isinstance(value, tuple) else value


def accuracy(rows: list[dict]) -> float:
    return float(np.mean([row["correct"] for row in rows])) if rows else 0.0


@torch.inference_mode()
def run() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if protocol["authorization"] != "run_authoritative_qwen3_capture":
        raise RuntimeError("unauthorized")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    total_tokens = protocol["total_actual_tokens"]
    full_path = OUT / "raw/qwen3_all_token_all_checkpoint.bf16.npy"
    role_path = OUT / "raw/qwen3_six_role_field.bf16.npy"
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.uint16, shape=(38, total_tokens, DIM))
    role = np.lib.format.open_memmap(role_path, mode="w+", dtype=np.uint16, shape=(1280, 6, 38, DIM))
    logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(1280, 2))
    model = None
    results, index = [], []
    offset = 0
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def forward(batch):
            cap = {}
            handles = [model.model.embed_tokens.register_forward_hook(lambda _m, _a, o: cap.__setitem__("e", tensor_output(o).detach()))]
            handles += [layer.register_forward_hook(lambda _m, _a, o, j=i: cap.__setitem__(f"b{j}", tensor_output(o).detach())) for i, layer in enumerate(model.model.layers)]
            handles.append(model.model.norm.register_forward_hook(lambda _m, _a, o: cap.__setitem__("n", tensor_output(o).detach())))
            try:
                ids, mask, pos, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                for handle in handles:
                    handle.remove()
            tensors = [cap["e"], *[cap[f"b{i}"] for i in range(36)], cap["n"]]
            scores = np.asarray([[float(output.logits[i, lengths[i] - 1, ids_[0]]) for ids_ in row["candidate_ids"]] for i, row in enumerate(batch)], np.float32)
            return tensors, scores, output, ids, mask, pos, lengths

        first_repeat = None
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            tensors, scores, output, ids, mask, pos, lengths = forward(batch)
            logits[start:start + len(batch)] = scores
            for i, row in enumerate(batch):
                n = int(lengths[i])
                begin, end = offset, offset + n
                for q, state in enumerate(tensors):
                    full[q, begin:end] = state[i, :n].contiguous().view(torch.uint16).cpu().numpy()
                    for ri, role_name in enumerate(ROLES):
                        role[start + i, ri, q] = state[i, row["role_positions"][role_name]].mean(0).contiguous().view(torch.uint16).cpu().numpy()
                pred = int(scores[i, 1] > scores[i, 0])
                results.append({
                    "row_index": start + i,
                    "case_id": row["case_id"],
                    "unit_id": row["unit_id"],
                    "arm": row["arm"],
                    "partition": row["partition"],
                    "factors": row["factors"],
                    "surface_factor": row["surface_factor"],
                    "codebook_factor": row["codebook_factor"],
                    "gold_position": row["gold_position"],
                    "prediction": pred,
                    "correct": pred == row["gold_position"],
                })
                index.append({"row_index": start + i, "case_id": row["case_id"], "token_offset_start": begin, "token_offset_end": end, "token_count": n})
                offset = end
            if start == 0:
                first_repeat = (batch, scores.copy())
            if (start // BATCH + 1) % 40 == 0:
                full.flush(); role.flush(); logits.flush()
                print(f"[C141] {start + len(batch)}/1280 tokens={offset}/{total_tokens}", flush=True)
            del tensors, output, ids, mask, pos
        full.flush(); role.flush(); logits.flush()
        repeat_tensors, repeat_scores, repeat_output, repeat_ids, repeat_mask, repeat_pos, repeat_lengths = forward(first_repeat[0])
        repeat = float(np.max(np.abs(repeat_scores - first_repeat[1])))
        del repeat_tensors, repeat_output, repeat_ids, repeat_mask, repeat_pos
    finally:
        full.flush(); role.flush(); logits.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()
        torch.cuda.empty_cache()
    if offset != total_tokens:
        raise RuntimeError((offset, total_tokens))
    core.write_rows(OUT / "raw/qwen3_behavior_index.jsonl", results)
    core.write_rows(OUT / "raw/all_token_field_index.jsonl", index)
    by_arm = {arm: accuracy([row for row in results if row["arm"] == arm]) for arm in ARMS}
    by_partition = {part: accuracy([row for row in results if row["partition"] == part]) for part in ("discovery", "confirmation")}
    by_code = {str(code): accuracy([row for row in results if row["codebook_factor"] == code]) for code in (1, -1)}
    by_surface = {str(surface): accuracy([row for row in results if row["surface_factor"] == surface]) for surface in (1, -1)}
    global_accuracy = accuracy(results)
    thresholds = protocol["behavior_observation_thresholds"]
    behavior_qualified = global_accuracy >= thresholds["global"] and min(by_arm.values()) >= thresholds["arm"] and min(by_partition.values()) >= thresholds["partition"]
    checks = {
        "rows": len(results) == 1280,
        "offset": offset == total_tokens,
        "full_shape": list(full.shape) == protocol["expected_full_field_shape"],
        "role_shape": list(role.shape) == protocol["expected_role_field_shape"],
        "finite": bool(np.isfinite(logits).all()),
        "repeat": repeat == 0.0,
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "authoritative_capture_complete",
        "behavior": {"global": global_accuracy, "arm": by_arm, "partition": by_partition, "surface": by_surface, "codebook": by_code, "observation_gate_passed": behavior_qualified},
        "capture": {"full_shape": list(full.shape), "role_shape": list(role.shape), "full_sha256": core.sha(full_path), "role_sha256": core.sha(role_path), "logits_sha256": core.sha(logits_path)},
        "checks": checks,
        "repeat_logits_max_abs": repeat,
        "runtime": placement,
        "authorization": "analyze_C142_mobius_regardless_of_behavior",
    }
    core.save(OUT / "analysis/authoritative_run.json", report)
    core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_behavior_gate_passed": behavior_qualified, "authorization": report["authorization"]})
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def close() -> None:
    report = core.load(OUT / "analysis/authoritative_run.json")
    behavior_rows = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    error_counts = {arm: sum(row["arm"] == arm and not row["correct"] for row in behavior_rows) for arm in ARMS}
    closure = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "five_arm_atlas_closed",
        "headline": report["behavior"],
        "error_counts": error_counts,
        "observation_book": "all correct and incorrect trajectories retained with typed labels",
        "functional_book": "only behavior-qualified cells may support functional claims",
        "claim_boundary": "large controlled five-family Qwen3 activation atlas, not a discovered transition law or semantic circuit",
        "next_authorization": "C142 coordinate Mobius and output-code separation",
    }
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"],
        "all_arms": set(error_counts) == set(ARMS),
    }
    core.save(OUT / "analysis/closure.json", closure)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_final_then_C142"})
    print(json.dumps(closure, indent=2))


def main() -> None:
    modes = {"contract": contract, "run": run, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit(f"usage: {Path(__file__).name} {'|'.join(modes)}")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
