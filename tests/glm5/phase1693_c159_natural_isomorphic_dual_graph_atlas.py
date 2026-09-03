#!/usr/bin/env python3
"""C159: natural-lexical and isomorphic-nonce graph HiddenState atlas."""
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
OUT = RESULT / "phase1693_c159_natural_isomorphic_dual_graph_atlas"
C158 = RESULT / "phase1692_c158_increment_source_decomposition"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127

PHASE, CAMPAIGN = 1693, "C159"
DIM, WIDTH, BATCH = 2560, 256, 4
STATES = tuple(range(38))
LATE = tuple(range(24, 35))
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
PANELS = ("natural_lexical", "isomorphic_nonce")

UNITS = (
    ("is_a", "is a kind of", "apple", "fruit", "food", "consumable", "instrument", "spoon", "table"),
    ("is_a", "is a kind of", "robin", "bird", "animal", "organism", "vehicle", "wing", "garage"),
    ("is_a", "is a kind of", "oak", "tree", "plant", "organism", "machine", "leaf", "factory"),
    ("part_of", "is part of", "key", "keyboard", "laptop", "workstation", "forest", "screen", "garden"),
    ("part_of", "is part of", "page", "chapter", "book", "collection", "engine", "cover", "garage"),
    ("part_of", "is part of", "wheel", "bicycle", "fleet", "transport network", "kitchen", "pedal", "stove"),
    ("located_in", "is located in", "archive", "east wing", "museum", "city", "ocean", "ticket", "harbor"),
    ("located_in", "is located in", "village", "county", "state", "country", "desert", "bridge", "island"),
    ("located_in", "is located in", "cabin", "clearing", "forest", "region", "factory", "trail", "warehouse"),
    ("precedes", "precedes", "dawn", "morning", "noon", "evening", "winter", "clock", "snow"),
    ("precedes", "precedes", "spark", "ignition", "combustion", "motion", "silence", "wire", "echo"),
    ("precedes", "precedes", "seed", "sprout", "sapling", "tree", "stone", "soil", "wall"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def partition(unit):
    return ("discovery", "confirmation", "fresh")[unit % 3]


def nonce_terms(unit, count=7):
    syllables = ("nava", "torel", "miku", "saren", "velu", "prax", "dovin", "kela", "zumi", "raben", "faro", "lutin")
    return [f"{syllables[(unit * count + i) % len(syllables)]}{unit}{i}" for i in range(count)]


def option_text(t0, t1, code):
    if code == 1:
        return f"(A) {t0} (B) {t1}", 0
    return f"(A) {t1} (B) {t0}", 1


def edge_sentence(a, relation, b, direction_form):
    if direction_form == 1:
        return f"{a} {relation} {b}"
    return f"A directed registry link from {a} to {b} is labelled '{relation}'"


def make_case(panel, unit, f1, f2, f3, f4, surface, code):
    relation_key, relation, source, b1, b2, t0, t1, other0, other1 = UNITS[unit]
    natural_values = [source, b1, b2, t0, t1, other0, other1]
    values = natural_values if panel == "natural_lexical" else nonce_terms(unit)
    source, b1, b2, t0, t1, other0, other1 = values
    intended, alternative = (t0, t1) if f1 == 1 else (t1, t0)
    if f2 == 1:
        edges = [(source, intended)]
        context = intended
    else:
        edges = [(source, b1), (b1, b2), (b2, intended)]
        context = b1
    sentences = [edge_sentence(a, relation, b, f4) for a, b in edges]
    if f3 == -1:
        sentences.extend((f"{source} is associated with {other0}", f"{other0} is adjacent to {alternative}", f"{other1} is associated with {b2}"))
    if surface == -1:
        sentences = list(reversed(sentences))
        statement = "The registry has the following entries: " + " | ".join(sentences) + "."
    else:
        statement = "; ".join(sentences) + "."
    options, t0_pos = option_text(t0, t1, code)
    gold = t0_pos if f1 == 1 else 1 - t0_pos
    prompt = f"{statement} Following only directed '{relation}' links, which registered target is reachable from {source}? {options} Reply with only A or B."
    return {
        "case_id": "",
        "panel": panel,
        "unit": unit,
        "unit_id": f"c159-{panel}-{unit:02d}",
        "partition": partition(unit),
        "relation_family": relation_key,
        "relation_phrase": relation,
        "factors": {"target": f1, "path": f2, "interference": f3, "direction_form": f4},
        "surface_factor": surface,
        "codebook_factor": code,
        "gold_position": gold,
        "prompt": prompt,
        "role_values": {"primary": source, "secondary": intended, "relation": relation, "context": context, "query": source},
        "semantic_edges": edges,
        "intended": intended,
        "alternative": alternative,
    }


def material():
    units, cases = [], []
    for panel in PANELS:
        for unit in range(len(UNITS)):
            units.append({"panel": panel, "unit": unit, "unit_id": f"c159-{panel}-{unit:02d}", "partition": partition(unit), "relation_family": UNITS[unit][0]})
            for factors in itertools.product((1, -1), repeat=6):
                row = make_case(panel, unit, *factors)
                row["case_id"] = f"c159-{len(cases):05d}"
                cases.append(row)
    return units, cases


def compile_rows(tokenizer, cases):
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(ids) != 1 for ids in candidates):
        raise RuntimeError(candidates)
    system = "Use only the supplied directed registry links. Answer exactly A or B."
    compiled = []
    for row in cases:
        ids = core.chat_ids(tokenizer, system, row["prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return compiled


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C158 / "audit/independent_final_audit.json")
    units, cases = material()
    compiled = compile_rows(graph_base.tokenizer(), cases)
    representatives = [i for i, row in enumerate(compiled) if row["factors"] == {"target": 1, "path": 1, "interference": 1, "direction_form": 1} and row["surface_factor"] == 1 and row["codebook_factor"] == 1]
    representative_tokens = sum(len(compiled[i]["prompt_ids"]) for i in representatives)
    cells = {(row["panel"], row["unit"], *row["factors"].values(), row["surface_factor"], row["codebook_factor"]) for row in cases}
    checks = {
        "authorization": parent["all_checks_passed"],
        "units": len(units) == 24,
        "cases": len(cases) == 1536 and len(cells) == 1536,
        "panels": all(sum(row["panel"] == panel for row in cases) == 768 for panel in PANELS),
        "relations": all(sum(row["relation_family"] == relation for row in cases) == 384 for relation in {u[0] for u in UNITS}),
        "partitions": all(sum(row["partition"] == part for row in cases) == 512 for part in ("discovery", "confirmation", "fresh")),
        "balance": sum(row["gold_position"] == 0 for row in cases) == 768,
        "unique": len({row["prompt"] for row in cases}) == 1536,
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "representatives": len(representatives) == 24,
        "semantic_uniqueness": all(row["intended"] != row["alternative"] and row["semantic_edges"][-1][1] == row["intended"] for row in cases),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    core.save(OUT / "material/representatives.json", {"row_indices": representatives, "total_tokens": representative_tokens})
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "natural_isomorphic_dual_graph_contract_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "cases": 1536,
        "design": "12 lexical units x 2 panels x target/path/interference/direction-form/surface/codebook",
        "relations": sorted({u[0] for u in UNITS}),
        "partitions": {"discovery": 512, "confirmation": 512, "fresh": 512},
        "capture": {"all_cases": "6 roles x embedding+36 blocks+final norm x 2560", "representatives": "24 cases x all tokens x all checkpoints x 2560"},
        "behavior_thresholds": {"panel_min": 0.75, "relation_min": 0.65},
        "response_thresholds": {"natural_nonce_median_cosine_min": 0.30, "each_relation_median_cosine_min": 0.10},
        "behavior_policy": "descriptive strata; incorrect trajectories retained and do not stop observation",
        "naturalness": "hand-curated natural lexical items plus machine grammar/uniqueness audit; independent human blind rating missing",
        "claim_boundary": "external lexical and graph-isomorphism atlas; not natural world-knowledge or a unique circuit",
        "forbidden": ["attention", "MLP", "weights", "PCA", "post-unblind threshold changes"],
        "source_hashes": {"C158": core.sha(C158 / "analysis/final.json")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C159_qwen",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True, "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "representative_tokens": representative_tokens, "max_width": max(len(row["prompt_ids"]) for row in compiled)}, indent=2))


@torch.inference_mode()
def run():
    protocol = core.load(OUT / "protocol/preregistration.json")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    reps = core.load(OUT / "material/representatives.json")
    rep_set = set(reps["row_indices"])
    rep_offsets, offset = {}, 0
    for i in reps["row_indices"]:
        n = len(rows[i]["prompt_ids"])
        rep_offsets[i] = (offset, offset + n)
        offset += n
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    role_path = OUT / "raw/qwen3_six_role_all_checkpoint.bf16.npy"
    token_path = OUT / "raw/qwen3_representative_all_token_all_checkpoint.bf16.npy"
    role_raw = np.lib.format.open_memmap(role_path, mode="w+", dtype=np.uint16, shape=(1536, 6, 38, DIM))
    token_raw = np.lib.format.open_memmap(token_path, mode="w+", dtype=np.uint16, shape=(38, reps["total_tokens"], DIM))
    logits = np.lib.format.open_memmap(OUT / "raw/qwen3_candidate_logits.float32.npy", mode="w+", dtype=np.float32, shape=(1536, 2))
    behavior = []
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        base = model.model
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def forward(batch):
            cap = {}
            hooks = [base.embed_tokens.register_forward_hook(lambda _m, _a, o: cap.__setitem__(0, tensor(o).detach()))]
            hooks += [layer.register_forward_hook(lambda _m, _a, o, q=i + 1: cap.__setitem__(q, tensor(o).detach())) for i, layer in enumerate(base.layers)]
            hooks += [base.norm.register_forward_hook(lambda _m, _a, o: cap.__setitem__(37, tensor(o).detach()))]
            try:
                ids, mask, pos, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                for hook in hooks:
                    hook.remove()
            return cap, output, lengths

        for start in range(0, 1536, BATCH):
            batch = rows[start:start + BATCH]
            cap, output, lengths = forward(batch)
            scores = np.asarray([[float(output.logits[i, lengths[i] - 1, candidate[0]]) for candidate in row["candidate_ids"]] for i, row in enumerate(batch)], np.float32)
            logits[start:start + len(batch)] = scores
            for local, row in enumerate(batch):
                ri = start + local
                for q in STATES:
                    state = cap[q][local]
                    for role_index, role in enumerate(ROLES):
                        role_raw[ri, role_index, q] = state[row["role_positions"][role]].mean(0).contiguous().view(torch.uint16).cpu().numpy()
                    if ri in rep_set:
                        begin, end = rep_offsets[ri]
                        n = end - begin
                        token_raw[q, begin:end] = state[:n].contiguous().view(torch.uint16).cpu().numpy()
                prediction = int(scores[local, 1] > scores[local, 0])
                behavior.append({"row_index": ri, "case_id": row["case_id"], "panel": row["panel"], "unit": row["unit"], "partition": row["partition"], "relation_family": row["relation_family"], "factors": row["factors"], "surface_factor": row["surface_factor"], "codebook_factor": row["codebook_factor"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"]})
            if (start // BATCH + 1) % 48 == 0:
                role_raw.flush(); token_raw.flush(); logits.flush(); print(f"[C159] {start + len(batch)}/1536", flush=True)
            del cap, output
        role_raw.flush(); token_raw.flush(); logits.flush()
    finally:
        role_raw.flush(); token_raw.flush(); logits.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    core.write_rows(OUT / "raw/qwen3_behavior_index.jsonl", behavior)
    core.write_rows(OUT / "raw/representative_token_index.jsonl", [{"row_index": i, "case_id": rows[i]["case_id"], "token_offset_start": rep_offsets[i][0], "token_offset_end": rep_offsets[i][1], "token_count": rep_offsets[i][1] - rep_offsets[i][0]} for i in reps["row_indices"]])
    accuracy = lambda subset: float(np.mean([row["correct"] for row in subset]))
    by_panel = {panel: accuracy([row for row in behavior if row["panel"] == panel]) for panel in PANELS}
    by_relation = {rel: accuracy([row for row in behavior if row["relation_family"] == rel]) for rel in sorted({u[0] for u in UNITS})}
    by_partition = {part: accuracy([row for row in behavior if row["partition"] == part]) for part in ("discovery", "confirmation", "fresh")}
    gates = {"panel": min(by_panel.values()) >= protocol["behavior_thresholds"]["panel_min"], "relation": min(by_relation.values()) >= protocol["behavior_thresholds"]["relation_min"]}
    checks = {"rows": len(behavior) == 1536, "role_shape": list(role_raw.shape) == [1536, 6, 38, DIM], "token_shape": list(token_raw.shape) == [38, reps["total_tokens"], DIM], "finite": bool(np.isfinite(logits).all()), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "authoritative_capture_complete", "behavior": {"global": accuracy(behavior), "panel": by_panel, "relation": by_relation, "partition": by_partition, "gates": gates, "qualified": all(gates.values())}, "checks": checks, "runtime": placement, "raw_hashes": {"role": core.sha(role_path), "representative_token": core.sha(token_path)}, "authorization": "analyze_C159_regardless_of_behavior"}
    core.save(OUT / "analysis/run.json", report)
    core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_behavior_qualified": all(gates.values()), "authorization": report["authorization"]})
    print(json.dumps({"behavior": report["behavior"], "checks": checks}, indent=2))


def cosine(a, b):
    af, bf = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(np.dot(af, bf) / max(np.linalg.norm(af) * np.linalg.norm(bf), 1e-12))


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    raw = np.load(OUT / "raw/qwen3_six_role_all_checkpoint.bf16.npy", mmap_mode="r")
    lookup = {}
    for i, row in enumerate(rows):
        f = row["factors"]
        key = (row["panel"], row["unit"], f["path"], f["interference"], f["direction_form"], row["surface_factor"], row["codebook_factor"])
        lookup.setdefault(key, {})[f["target"]] = i
    keys = sorted(lookup)
    path = OUT / "analysis/late_half_difference.float16.npy"
    late = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(keys), len(LATE), 6, DIM))
    pair_rows = []
    for j, key in enumerate(keys):
        plus, minus = lookup[key][1], lookup[key][-1]
        for qi, q in enumerate(LATE):
            late[j, qi] = ((c127.decode(raw[plus, :, q]) - c127.decode(raw[minus, :, q])) / 2.0).astype(np.float16)
        panel, unit, f2, f3, f4, surface, code = key
        pair_rows.append({"pair_index": j, "panel": panel, "unit": unit, "partition": partition(unit), "relation_family": UNITS[unit][0], "path": f2, "interference": f3, "direction_form": f4, "surface": surface, "code": code, "plus_row": plus, "minus_row": minus})
    late.flush()
    core.write_rows(OUT / "analysis/late_half_difference_index.jsonl", pair_rows)
    pair_lookup = {(row["panel"], row["unit"], row["path"], row["interference"], row["direction_form"], row["surface"], row["code"]): row["pair_index"] for row in pair_rows}
    similarities = []
    relation_values = {rel: [] for rel in sorted({u[0] for u in UNITS})}
    partition_values = {part: [] for part in ("discovery", "confirmation", "fresh")}
    for unit in range(len(UNITS)):
        for f2, f3, f4, surface, code in itertools.product((1, -1), repeat=5):
            ni = pair_lookup[("natural_lexical", unit, f2, f3, f4, surface, code)]
            pi = pair_lookup[("isomorphic_nonce", unit, f2, f3, f4, surface, code)]
            for qi, q in enumerate(LATE):
                value = cosine(np.asarray(late[ni, qi], np.float32), np.asarray(late[pi, qi], np.float32))
                similarities.append({"unit": unit, "relation_family": UNITS[unit][0], "partition": partition(unit), "q": q, "cosine": value})
                relation_values[UNITS[unit][0]].append(value)
                partition_values[partition(unit)].append(value)
    checkpoint = {str(q): float(np.median([row["cosine"] for row in similarities if row["q"] == q])) for q in LATE}
    relation = {name: float(np.median(values)) for name, values in relation_values.items()}
    partitions = {name: float(np.median(values)) for name, values in partition_values.items()}
    overall = float(np.median([row["cosine"] for row in similarities]))
    g = protocol["response_thresholds"]
    response_gates = {"overall": overall >= g["natural_nonce_median_cosine_min"], "relations": all(value >= g["each_relation_median_cosine_min"] for value in relation.values())}
    coordinate_rows = []
    top_coordinate_rows = []
    for panel in PANELS:
        for relation_name in sorted(relation_values):
            ids = [row["pair_index"] for row in pair_rows if row["panel"] == panel and row["relation_family"] == relation_name]
            for qi, q in enumerate(LATE):
                mean = np.asarray(late[ids, qi], np.float32).mean(0)
                for role_i, role in enumerate(ROLES):
                    values = mean[role_i]
                    top = np.argsort(np.abs(values))[-256:][::-1]
                    coordinate_rows.append({"dataset": "C159", "panel": panel, "relation_family": relation_name, "checkpoint": q, "role": role, "values": values.tolist()})
                    top_coordinate_rows.append({"panel": panel, "relation_family": relation_name, "checkpoint": q, "role": role, "coordinate_ids": top.tolist(), "coordinate_values": values[top].tolist()})
    core.save(OUT / "analysis/coordinate_rows.json", coordinate_rows)
    core.save(OUT / "analysis/top_coordinate_rows.json", top_coordinate_rows)
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "dual_graph_atlas_adjudicated", "behavior": core.load(OUT / "analysis/run.json")["behavior"], "natural_nonce_response": {"overall_median_cosine": overall, "checkpoint_median_cosine": checkpoint, "relation_median_cosine": relation, "partition_median_cosine": partitions, "gates": response_gates, "passed": all(response_gates.values())}, "counts": {"pairs": len(keys), "matched_similarity_rows": len(similarities), "coordinate_rows": len(coordinate_rows)}, "claim_boundary": protocol["claim_boundary"], "next_authorization": "C160 recipient-only prediction regardless of atlas gate"}
    core.save(OUT / "analysis/atlas.json", report)
    checks = {"pairs": len(keys) == 768, "late_shape": list(late.shape) == [768, 11, 6, DIM], "similarities": len(similarities) == 4224, "coordinate_rows": len(coordinate_rows) == 528, "finite": bool(np.isfinite([row["cosine"] for row in similarities]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_response_passed": all(response_gates.values()), "authorization": report["next_authorization"]})
    print(json.dumps({"behavior": report["behavior"], "natural_nonce_response": report["natural_nonce_response"], "counts": report["counts"]}, indent=2))


def close():
    atlas = core.load(OUT / "analysis/atlas.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"behavior": atlas["behavior"], "natural_nonce_response": atlas["natural_nonce_response"]}, "next_authorization": atlas["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_audit_then_C160"})
    print(json.dumps(final, indent=2))


def main():
    modes = {"contract": contract, "run": run, "analyze": analyze, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("contract|run|analyze|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
