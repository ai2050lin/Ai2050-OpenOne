#!/usr/bin/env python3
"""C162: broad linguistic-program HiddenState field with first/second-order contrasts."""
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
OUT = RESULT / "phase1696_c162_linguistic_program_field"
C161 = RESULT / "phase1695_c161_full_coordinate_local_transmission"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127

PHASE, CAMPAIGN = 1696, "C162"
DIM, WIDTH, BATCH = 2560, 224, 4
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
LATE = tuple(range(24, 35))
FACTOR_NAMES = ("agent_identity", "coreference_form", "attitude", "action", "patient", "negation_scope")
UNITS = (
    ("Mira", "Jon", "apple", "pear"),
    ("Lena", "Omar", "bread", "cheese"),
    ("Nora", "Pavel", "peach", "plum"),
    ("Iris", "Mateo", "carrot", "onion"),
    ("Asha", "Felix", "mango", "melon"),
    ("Rina", "Damon", "cookie", "cracker"),
    ("Tara", "Hugo", "berry", "grape"),
    ("Sofia", "Niko", "lemon", "lime"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def partition(unit):
    return "discovery" if unit < 4 else ("confirmation" if unit < 6 else "fresh")


def make_case(unit, factors):
    f1, f2, f3, f4, f5, f6, surface, code = factors
    p0, p1, o0, o1 = UNITS[unit]
    agent = p0 if f1 == 1 else p1
    patient = o0 if f5 == 1 else o1
    attitude = "likes" if f3 == 1 else "dislikes"
    active, passive = (("eat", "eaten") if f4 == 1 else ("inspect", "inspected"))
    if f2 == 1:
        embedded_agent = "I" if surface == 1 else "me"
        if surface == 1:
            clause = f'I will {"not " if f6 == -1 else ""}{active} the {patient}'
        else:
            clause = f'The {patient} will {"not " if f6 == -1 else ""}be {passive} by me'
        statement = f'{agent} {"does not " if f6 == 1 else ""}{attitude.rstrip("s")} the statement, "{clause}."'
        primary_occurrence = 0
    else:
        embedded_agent = agent
        if surface == 1:
            clause = f'{agent} will {"not " if f6 == -1 else ""}{active} the {patient}'
        else:
            clause = f'the {patient} will {"not " if f6 == -1 else ""}be {passive} by {agent}'
        statement = f'{agent} {"does not " if f6 == 1 else ""}{attitude.rstrip("s")} the statement that {clause}'
        primary_occurrence = 1
    if code == 1:
        options, p0pos = f"(A) {p0} (B) {p1}", 0
    else:
        options, p0pos = f"(A) {p1} (B) {p0}", 1
    gold = p0pos if f1 == 1 else 1 - p0pos
    prompt = f"{statement}. Who is the embedded agent of the described action? {options} Reply with only A or B."
    return {"case_id": "", "unit": unit, "unit_id": f"c162-{unit:02d}", "partition": partition(unit), "factors": dict(zip((*FACTOR_NAMES, "voice", "codebook"), factors)), "gold_position": gold, "prompt": prompt, "role_values": {"primary": embedded_agent, "secondary": patient, "relation": active if surface == 1 else passive, "context": agent, "query": "embedded agent"}, "primary_occurrence": primary_occurrence}


def material():
    cases = []
    for unit in range(len(UNITS)):
        for factors in itertools.product((1, -1), repeat=8):
            row = make_case(unit, factors)
            row["case_id"] = f"c162-{len(cases):05d}"
            cases.append(row)
    return cases


def compile_rows(tokenizer, cases):
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    system = "Read the sentence literally and identify the embedded agent. Answer exactly A or B."
    rows = []
    for row in cases:
        ids = core.chat_ids(tokenizer, system, row["prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            if role == "primary":
                positions[role] = spans[row["primary_occurrence"]]
            elif role == "context":
                positions[role] = spans[0]
            else:
                positions[role] = spans[0]
        positions["boundary"] = [len(ids) - 1]
        rows.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return rows


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C161 / "audit/independent_final_audit.json")
    cases = material()
    compiled = compile_rows(graph_base.tokenizer(), cases)
    checks = {"authorization": parent["all_checks_passed"], "cases": len(cases) == 2048, "unique": len({row["prompt"] for row in cases}) == 2048, "balance": sum(row["gold_position"] == 0 for row in cases) == 1024, "partitions": sum(row["partition"] == "discovery" for row in cases) == 1024 and sum(row["partition"] == "confirmation" for row in cases) == sum(row["partition"] == "fresh" for row in cases) == 512, "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled), "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "linguistic_program_contract_frozen", "model": "Qwen3-4B BF16 CUDA nonquantized", "cases": 2048, "semantic_factors": list(FACTOR_NAMES), "nuisance_factors": ["voice", "codebook"], "terms": "six first-order plus fifteen pair interactions", "capture": "six aligned roles x embedding+36 blocks+final norm x all 2560 coordinates", "behavior_thresholds": {"global_min": 0.75, "unit_min": 0.60}, "transfer_gates": {"first_order_term_count_min": 3, "first_order_cosine_min": 0.20, "pair_term_count_min": 3, "pair_cosine_min": 0.10}, "behavior_policy": "descriptive; all trajectories retained", "claim_boundary": "controlled English program-factor atlas; not a complete syntax or natural discourse mechanism", "forbidden": ["attention", "MLP", "weights", "PCA", "post-unblind term selection"], "source_hashes": {"C161": core.sha(C161 / "analysis/transmission.json")}, "producer_sha256": core.sha(Path(__file__)), "authorization": "run_C162_qwen"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True, "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled)}, indent=2))


@torch.inference_mode()
def run():
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(OUT / "raw/qwen3_six_role_all_checkpoint.bf16.npy", mode="w+", dtype=np.uint16, shape=(2048, 6, 38, DIM))
    logits = np.lib.format.open_memmap(OUT / "raw/qwen3_candidate_logits.float32.npy", mode="w+", dtype=np.float32, shape=(2048, 2))
    behavior, model = [], None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        base, pad = model.model, int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, 2048, BATCH):
            batch = rows[start:start + BATCH]
            cap = {}
            hooks = [base.embed_tokens.register_forward_hook(lambda _m, _a, o: cap.__setitem__(0, tensor(o).detach()))]
            hooks += [layer.register_forward_hook(lambda _m, _a, o, q=i + 1: cap.__setitem__(q, tensor(o).detach())) for i, layer in enumerate(base.layers)]
            hooks += [base.norm.register_forward_hook(lambda _m, _a, o: cap.__setitem__(37, tensor(o).detach()))]
            try:
                ids, mask, pos, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                for hook in hooks: hook.remove()
            scores = np.asarray([[float(output.logits[i, lengths[i] - 1, candidate[0]]) for candidate in row["candidate_ids"]] for i, row in enumerate(batch)], np.float32)
            logits[start:start + len(batch)] = scores
            for local, row in enumerate(batch):
                for q in range(38):
                    for role_i, role in enumerate(ROLES):
                        raw[start + local, role_i, q] = cap[q][local, row["role_positions"][role]].mean(0).contiguous().view(torch.uint16).cpu().numpy()
                pred = int(scores[local, 1] > scores[local, 0])
                behavior.append({"row_index": start + local, "case_id": row["case_id"], "unit": row["unit"], "partition": row["partition"], "factors": row["factors"], "gold_position": row["gold_position"], "prediction": pred, "correct": pred == row["gold_position"]})
            if (start // BATCH + 1) % 64 == 0:
                raw.flush(); logits.flush(); print(f"[C162] {start + len(batch)}/2048", flush=True)
            del cap, output
        raw.flush(); logits.flush()
    finally:
        raw.flush(); logits.flush()
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    core.write_rows(OUT / "raw/qwen3_behavior_index.jsonl", behavior)
    accuracy = lambda subset: float(np.mean([row["correct"] for row in subset]))
    by_unit = {str(unit): accuracy([row for row in behavior if row["unit"] == unit]) for unit in range(8)}
    by_partition = {part: accuracy([row for row in behavior if row["partition"] == part]) for part in ("discovery", "confirmation", "fresh")}
    protocol = core.load(OUT / "protocol/preregistration.json")
    gates = {"global": accuracy(behavior) >= protocol["behavior_thresholds"]["global_min"], "units": min(by_unit.values()) >= protocol["behavior_thresholds"]["unit_min"]}
    checks = {"rows": len(behavior) == 2048, "shape": list(raw.shape) == [2048, 6, 38, DIM], "finite": bool(np.isfinite(logits).all()), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "capture_complete", "behavior": {"global": accuracy(behavior), "unit": by_unit, "partition": by_partition, "gates": gates, "qualified": all(gates.values())}, "checks": checks, "runtime": placement, "authorization": "analyze_C162_regardless_of_behavior"}
    core.save(OUT / "analysis/run.json", report)
    core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_behavior_qualified": all(gates.values()), "authorization": report["authorization"]})
    print(json.dumps({"behavior": report["behavior"], "checks": checks}, indent=2))


def cosine(a, b):
    return float(np.sum(a * b, dtype=np.float64) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def analyze():
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    raw = np.load(OUT / "raw/qwen3_six_role_all_checkpoint.bf16.npy", mmap_mode="r")
    terms = [(name, (i,)) for i, name in enumerate(FACTOR_NAMES)] + [(f"{FACTOR_NAMES[i]}*{FACTOR_NAMES[j]}", (i, j)) for i in range(6) for j in range(i + 1, 6)]
    term_path = OUT / "analysis/unit_term_fields.float16.npy"
    fields = np.lib.format.open_memmap(term_path, mode="w+", dtype=np.float16, shape=(8, len(terms), 11, 6, DIM))
    term_index = [{"term_index": i, "name": name, "factor_indices": list(indices), "order": len(indices)} for i, (name, indices) in enumerate(terms)]
    core.write_rows(OUT / "analysis/term_index.jsonl", term_index)
    for unit in range(8):
        ids = [i for i, row in enumerate(rows) if row["unit"] == unit]
        signs = np.asarray([[rows[i]["factors"][name] for name in FACTOR_NAMES] for i in ids], np.float32)
        for qi, q in enumerate(LATE):
            h = np.asarray([c127.decode(raw[i, :, q]) for i in ids], np.float32)
            for ti, (_name, indices) in enumerate(terms):
                weight = np.prod(signs[:, indices], axis=1)
                fields[unit, ti, qi] = np.mean(h * weight[:, None, None], axis=0).astype(np.float16)
        fields.flush(); print(f"[C162-analysis] unit {unit + 1}/8", flush=True)
    term_rows = []
    for ti, (name, indices) in enumerate(terms):
        discovery = np.asarray(fields[:4, ti], np.float32).mean(0)
        for part, unit_ids in (("confirmation", (4, 5)), ("fresh", (6, 7))):
            cosines = [cosine(discovery[qi], np.asarray(fields[unit, ti, qi], np.float32)) for unit in unit_ids for qi in range(11)]
            ratios = [float(np.linalg.norm(np.asarray(fields[unit, ti, qi], np.float32)) / max(np.linalg.norm(discovery[qi]), 1e-12)) for unit in unit_ids for qi in range(11)]
            term_rows.append({"name": name, "order": len(indices), "partition": part, "median_cosine": float(np.median(cosines)), "median_norm_ratio": float(np.median(ratios))})
    first_fresh = [row for row in term_rows if row["partition"] == "fresh" and row["order"] == 1]
    pair_fresh = [row for row in term_rows if row["partition"] == "fresh" and row["order"] == 2]
    protocol = core.load(OUT / "protocol/preregistration.json")
    passing_first = [row["name"] for row in first_fresh if row["median_cosine"] >= protocol["transfer_gates"]["first_order_cosine_min"]]
    passing_pair = [row["name"] for row in pair_fresh if row["median_cosine"] >= protocol["transfer_gates"]["pair_cosine_min"]]
    gates = {"first": len(passing_first) >= protocol["transfer_gates"]["first_order_term_count_min"], "pair": len(passing_pair) >= protocol["transfer_gates"]["pair_term_count_min"]}
    coordinate_rows = []
    for ti, (name, indices) in enumerate(terms):
        if len(indices) == 1 or name in passing_pair:
            mean = np.asarray(fields[:, ti], np.float32).mean(0)
            for qi, q in enumerate(LATE):
                for role_i, role in enumerate(ROLES):
                    coordinate_rows.append({"dataset": "C162", "term": name, "order": len(indices), "checkpoint": q, "role": role, "values": mean[qi, role_i].tolist()})
    core.save(OUT / "analysis/coordinate_rows.json", coordinate_rows)
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "linguistic_program_field_adjudicated", "behavior": core.load(OUT / "analysis/run.json")["behavior"], "term_rows": term_rows, "passing_first_order": passing_first, "passing_pair_terms": passing_pair, "gates": gates, "transfer_passed": all(gates.values()), "coordinate_rows": len(coordinate_rows), "claim_boundary": protocol["claim_boundary"], "next_authorization": "C163 natural graph call domain regardless of C162 gate"}
    core.save(OUT / "analysis/program_field.json", report)
    checks = {"shape": list(fields.shape) == [8, 21, 11, 6, DIM], "terms": len(term_rows) == 42, "coordinates": len(coordinate_rows) >= 396, "finite": bool(np.isfinite(fields).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_transfer_passed": all(gates.values()), "authorization": report["next_authorization"]})
    print(json.dumps({"behavior": report["behavior"], "passing_first": passing_first, "passing_pairs": passing_pair, "gates": gates, "fresh_terms": first_fresh + pair_fresh}, indent=2))


def close():
    report = core.load(OUT / "analysis/program_field.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"behavior": report["behavior"], "passing_first": report["passing_first_order"], "passing_pairs": report["passing_pair_terms"], "gates": report["gates"]}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_audit_then_C163"})
    print(json.dumps(final, indent=2))


def main():
    modes = {"contract": contract, "run": run, "analyze": analyze, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("contract|run|analyze|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
