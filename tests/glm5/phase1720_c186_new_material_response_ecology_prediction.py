#!/usr/bin/env python3
"""C186: prospective new-vocabulary/paraphrase validation of relation-target response ecology."""
from __future__ import annotations

import argparse
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
OUT = RESULT / "phase1720_c186_new_material_response_ecology_prediction"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
C180 = RESULT / "phase1714_c180_reachable_target_choice_ecology"
C185 = RESULT / "phase1719_c185_family_conditioned_routing_grammar"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

PHASE, CAMPAIGN = 1720, "C186"
DIM, WIDTH, BATCH = 2560, 224, 8
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
SPLITS = ("new_confirmation", "new_fresh")
RELATIONS = {
    "is_a": (("is a kind of", "belongs to the category of"), [
        ("sparrow", "bird", "animal", "organism", "mallet", "tool"),
        ("salmon", "fish", "animal", "creature", "kettle", "appliance"),
        ("cactus", "plant", "organism", "entity", "trumpet", "instrument"),
        ("bicycle", "vehicle", "machine", "artifact", "fork", "utensil"),
        ("ruby", "mineral", "material", "substance", "falcon", "bird"),
        ("soprano", "singer", "artist", "person", "cedar", "tree"),
    ]),
    "part_of": (("is a component of", "forms part of"), [
        ("piston", "engine", "vehicle", "system", "knob", "cabinet"),
        ("petal", "flower", "bouquet", "display", "screw", "machine"),
        ("neuron", "cortex", "brain", "body", "tile", "roof"),
        ("cabin", "airplane", "fleet", "network", "key", "piano"),
        ("chapter", "textbook", "curriculum", "program", "pedal", "bicycle"),
        ("pixel", "photograph", "album", "archive", "cable", "bridge"),
    ]),
    "located_in": (("is located inside", "can be found within"), [
        ("memo", "folder", "archive", "vault", "coin", "purse"),
        ("specimen", "tray", "cabinet", "laboratory", "spoon", "drawer"),
        ("village", "province", "country", "continent", "boat", "harbor"),
        ("server", "rack", "datacenter", "campus", "book", "shelf"),
        ("satellite", "orbit", "system", "galaxy", "seed", "pod"),
        ("actor", "theater", "district", "city", "tool", "garage"),
    ]),
    "causes": (("directly causes", "brings about"), [
        ("virus", "infection", "fever", "fatigue", "music", "joy"),
        ("voltage", "surge", "outage", "shutdown", "rain", "puddle"),
        ("drought", "cropfailure", "famine", "migration", "breeze", "motion"),
        ("toxin", "illness", "weakness", "collapse", "exercise", "strength"),
        ("collision", "fracture", "bleeding", "shock", "light", "shadow"),
        ("stress", "error", "defect", "crash", "heat", "expansion"),
    ]),
    "depends_on": (("depends on", "requires"), [
        ("camera", "lens", "sensor", "power", "door", "hinge"),
        ("orchard", "irrigation", "reservoir", "river", "car", "fuel"),
        ("website", "server", "database", "storage", "lamp", "switch"),
        ("compass", "magnet", "field", "planet", "clock", "spring"),
        ("surgery", "diagnosis", "scan", "device", "house", "roof"),
        ("shipment", "truck", "road", "network", "book", "page"),
    ]),
    "reports_to": (("reports to", "is accountable to"), [
        ("associate", "manager", "director", "president", "musician", "audience"),
        ("sergeant", "lieutenant", "colonel", "general", "student", "teacher"),
        ("nurse", "supervisor", "administrator", "board", "actor", "director"),
        ("reporter", "editor", "publisher", "owner", "driver", "station"),
        ("apprentice", "technician", "engineer", "architect", "chef", "guest"),
        ("researcher", "chair", "dean", "provost", "pilot", "tower"),
    ]),
    "derives_from": (("derives from", "originates from"), [
        ("paper", "pulp", "timber", "tree", "shadow", "light"),
        ("ethanol", "sugar", "cane", "crop", "song", "guitar"),
        ("ceramic", "clay", "mineral", "rock", "bread", "grain"),
        ("vaccine", "antigen", "protein", "gene", "river", "lake"),
        ("estimate", "sample", "measurement", "sensor", "painting", "canvas"),
        ("translation", "document", "archive", "collection", "smoke", "fire"),
    ]),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def split_for(unit: int) -> str:
    return SPLITS[unit // 3]


def make_case(family, phrase, unit, nodes, phrase_variant, order):
    a, b, c, d, e, f = nodes
    edges = [(a, b), (b, c), (c, d), (e, f)]
    facts = ". ".join(f"{x} {phrase} {y}" for x, y in edges) + "."
    options, gold = ((f"(A) {d} (B) {f}", 0) if order == 1 else (f"(A) {f} (B) {d}", 1))
    if phrase_variant == 0:
        prompt = f"Facts: {facts} Following only the stated '{phrase}' links, which target is reachable from {a}? {options}. Reply with only A or B."
    else:
        prompt = f"Registry notes: {facts} Begin at {a} and follow only arrows meaning '{phrase}'. Which registered target can be reached? {options}. Reply with only A or B."
    return {
        "case_id": "",
        "family": family,
        "phrase": phrase,
        "phrase_variant": phrase_variant,
        "unit": unit,
        "partition": split_for(unit),
        "order": order,
        "gold_position": gold,
        "prompt": prompt,
        "nodes": list(nodes),
        "semantic_edges": edges,
        "role_values": {"primary": a, "secondary": d, "relation": phrase, "context": b, "query": a},
    }


def material():
    cases = []
    for family, (phrases, units) in RELATIONS.items():
        for unit, nodes in enumerate(units):
            for phrase_variant, order in itertools.product((0, 1), (1, -1)):
                row = make_case(family, phrases[phrase_variant], unit, nodes, phrase_variant, order)
                row["case_id"] = f"c186-{len(cases):04d}"
                cases.append(row)
    return cases


def compile_rows(tokenizer, cases):
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(candidate) != 1 for candidate in candidates):
        raise RuntimeError(candidates)
    compiled = []
    for row in cases:
        ids = core.chat_ids(tokenizer, "Use only the supplied directed links. Answer exactly A or B.", row["prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return compiled


def c180_discovery_profiles():
    response = np.load(C180 / "raw/anchor_role_response.float16.npy", mmap_mode="r")
    anchors = core.rows(C180 / "raw/anchor_index.jsonl")
    families = core.load(C180 / "protocol/behavior_eligibility_lock.json")["eligible_families"]
    lookup = {(row["partition"], row["family"]): row["anchor_index"] for row in anchors}
    profiles = []
    for family in families:
        values = np.asarray(response[2, lookup[("discovery", family)]], dtype=np.float32)
        energy = np.square(values, dtype=np.float64).sum(axis=(0, 1))
        profiles.append((energy / max(energy.sum(), 1e-30)).astype(np.float32))
    return families, np.stack(profiles)


def contract():
    if OUT.exists() and (OUT / "protocol/preregistration.json").exists():
        raise RuntimeError(OUT)
    parent = core.load(C185 / "audit/independent_final_audit.json")
    cases = material()
    compiled = compile_rows(graph_base.tokenizer(), cases)
    families, predictors = c180_discovery_profiles()
    checks = {
        "authorization": parent["all_checks_passed"] and "C186" in parent["authorization"],
        "cases": len(cases) == 168,
        "families": set(families) == set(RELATIONS),
        "split_balance": all(sum(row["partition"] == split for row in cases) == 84 for split in SPLITS),
        "candidate_balance": float(np.mean([row["gold_position"] == 0 for row in cases])) == 0.5,
        "paraphrase_balance": float(np.mean([row["phrase_variant"] == 1 for row in cases])) == 0.5,
        "unique_target": all(row["nodes"][3] != row["nodes"][5] for row in cases),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True, exist_ok=True)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    (OUT / "protocol").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "protocol/c180_discovery_relation_target_profiles.float32.npy", predictors)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "new_material_prediction_contract_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "cases": len(cases),
        "families": families,
        "material": "six new lexical graph units per family; canonical phrase plus unseen paraphrase; two option orders",
        "behavior_gates": {"global_min": 0.80, "family_split_min": 0.75, "both_frozen_anchors_correct": True},
        "hidden_policy": "relation q24 source only; no hidden run before behavior lock",
        "prediction": "C185 relation-target energy profile; same-family versus six wrong-family C180 discovery profiles",
        "prediction_label": {"median_wrong_advantage_min": 0.01, "positive_family_fraction_min": 0.70},
        "claim_boundary": "new vocabulary and paraphrase transfer in an explicit registry task; not spontaneous world knowledge",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cosine", "testing unsupported roles"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_behavior_then_lock",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled)}, indent=2))


@torch.inference_mode()
def behavior():
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    logits = np.lib.format.open_memmap(OUT / "raw/behavior_logits.float32.npy", mode="w+", dtype=np.float32, shape=(len(rows), 2))
    index = []
    model = None
    try:
        model, _tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(_tokenizer.pad_token_id if _tokenizer.pad_token_id is not None else _tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            ids, mask, pos, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            for local, row in enumerate(batch):
                scores = [float(output.logits[local, lengths[local] - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                logits[start + local] = scores
                index.append({"row_index": start + local, "case_id": row["case_id"], "family": row["family"], "partition": row["partition"], "phrase_variant": row["phrase_variant"], "unit": row["unit"], "order": row["order"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"]})
        logits.flush()
        core.write_rows(OUT / "raw/behavior_index.jsonl", index)
        checks = {"rows": len(index) == 168, "finite": bool(np.isfinite(logits).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/behavior_run.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_behavior_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def lock():
    rows = core.rows(OUT / "raw/behavior_index.jsonl")
    q = core.load(OUT / "protocol/preregistration.json")["behavior_gates"]
    accuracy = lambda selected: float(np.mean([row["correct"] for row in selected]))
    global_accuracy = accuracy(rows)
    by_family_split = {family: {split: accuracy([row for row in rows if row["family"] == family and row["partition"] == split]) for split in SPLITS} for family in RELATIONS}
    anchor_rows = []
    for family in RELATIONS:
        for split, unit, phrase_variant in (("new_confirmation", 0, 0), ("new_fresh", 3, 1)):
            matches = [row for row in rows if row["family"] == family and row["partition"] == split and row["unit"] == unit and row["phrase_variant"] == phrase_variant and row["order"] == 1]
            if len(matches) != 1:
                raise RuntimeError((family, split, matches))
            anchor_rows.append(matches[0]["row_index"])
    anchor_correct = {rows[index]["case_id"]: rows[index]["correct"] for index in anchor_rows}
    eligible = [family for family in RELATIONS if min(by_family_split[family].values()) >= q["family_split_min"] and all(rows[index]["correct"] for index in anchor_rows if rows[index]["family"] == family)]
    if global_accuracy < q["global_min"]:
        eligible = []
    result = {"phase": PHASE, "campaign": CAMPAIGN, "status": "behavior_locked", "global_accuracy": global_accuracy, "by_family_split": by_family_split, "anchor_rows": anchor_rows, "anchor_correct": anchor_correct, "eligible_families": eligible, "authorization": "run_relation_response" if eligible else "close_hidden_not_tested"}
    core.save(OUT / "protocol/behavior_eligibility_lock.json", result)
    checks = {"global": global_accuracy >= q["global_min"], "anchors": len(anchor_rows) == 14, "eligible_nonempty": bool(eligible)}
    core.save(OUT / "audit/internal_behavior_lock_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_eligible": bool(eligible)})
    print(json.dumps(result, indent=2))


@torch.inference_mode()
def hidden():
    eligibility = core.load(OUT / "protocol/behavior_eligibility_lock.json")
    families = eligibility["eligible_families"]
    if not families:
        raise RuntimeError("no eligible families")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    behavior_rows = core.rows(OUT / "raw/behavior_index.jsonl")
    anchors = [index for index in eligibility["anchor_rows"] if rows[index]["family"] in families]
    coordinates = core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:64]
    raw = np.lib.format.open_memmap(OUT / "raw/new_relation_role_response.float16.npy", mode="w+", dtype=np.float16, shape=(len(anchors), 64, 6, DIM))
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        base = model.model
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def perturb(row, selected, sign, epsilon):
            batch = [row] * len(selected)
            ids, mask, pos, _lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            captured = {}
            def patch(_module, _args, value):
                hidden_state = tensor(value)
                patched = hidden_state.clone()
                for local, coordinate in enumerate(selected):
                    for position in row["role_positions"]["relation"]:
                        patched[local, position, int(coordinate)] += sign * epsilon
                return (patched,) + value[1:] if isinstance(value, tuple) else patched
            first = base.layers[23].register_forward_hook(patch)
            second = base.layers[24].register_forward_hook(lambda _m, _a, value: captured.__setitem__("state", tensor(value).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                first.remove(); second.remove()
            field = np.zeros((len(selected), 6, DIM), np.float32)
            for local in range(len(selected)):
                for role_i, role in enumerate(ROLES):
                    field[local, role_i] = captured["state"][local, row["role_positions"][role]].mean(0).float().cpu().numpy()
            return field

        for anchor_i, row_index in enumerate(anchors):
            row = rows[row_index]
            ids, mask, pos, _lengths = fixed_base.fixed_batch([row], pad, device, WIDTH)
            captured = {}
            hook = base.layers[23].register_forward_hook(lambda _m, _a, value: captured.__setitem__("state", tensor(value).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                hook.remove()
            source = captured["state"][0, row["role_positions"]["relation"]].mean(0).float().cpu().numpy()
            epsilon = 0.5 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))
            for start in range(0, 64, 16):
                selected = coordinates[start:start + 16]
                plus = perturb(row, selected, 1.0, epsilon)
                minus = perturb(row, selected, -1.0, epsilon)
                raw[anchor_i, start:start + len(selected)] = ((plus - minus) / (2 * epsilon)).astype(np.float16)
            raw.flush()
            print(f"[C186-response] {anchor_i + 1}/{len(anchors)} {row['family']} {row['partition']}", flush=True)
        core.write_rows(OUT / "raw/response_anchor_index.jsonl", [{"anchor_index": i, "row_index": row_index, "case_id": behavior_rows[row_index]["case_id"], "family": rows[row_index]["family"], "partition": rows[row_index]["partition"], "phrase_variant": rows[row_index]["phrase_variant"]} for i, row_index in enumerate(anchors)])
        checks = {"shape": list(raw.shape) == [len(anchors), 64, 6, DIM], "finite": bool(np.isfinite(raw).all()), "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/hidden_run.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_hidden_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        raw.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def profile(values):
    energy = np.square(values, dtype=np.float64).sum(axis=(0, 1))
    return energy / max(energy.sum(), 1e-30)


def similarity(left, right):
    return float(1.0 - 0.5 * np.abs(left - right).sum())


def analyze():
    eligibility = core.load(OUT / "protocol/behavior_eligibility_lock.json")
    families = eligibility["eligible_families"]
    if not families:
        result = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed_hidden_not_tested", "behavior": eligibility, "next_authorization": "C187_repair_behavior_interface_or_end_route"}
        core.save(OUT / "analysis/prediction_atlas.json", result)
        core.save(OUT / "audit/internal_analysis_audit.json", {"checks": {"typed": True}, "all_checks_passed": True})
        print(json.dumps(result, indent=2)); return
    predictors = np.load(OUT / "protocol/c180_discovery_relation_target_profiles.float32.npy")
    predictor_families = core.load(OUT / "protocol/preregistration.json")["families"]
    predictor = {family: predictors[i].astype(np.float64) for i, family in enumerate(predictor_families)}
    raw = np.load(OUT / "raw/new_relation_role_response.float16.npy", mmap_mode="r")
    anchors = core.rows(OUT / "raw/response_anchor_index.jsonl")
    limits = core.load(OUT / "protocol/preregistration.json")["prediction_label"]
    rows = []
    summary = {}
    for split in SPLITS:
        selected_rows = []
        for anchor in anchors:
            if anchor["partition"] != split:
                continue
            family = anchor["family"]
            actual = profile(np.asarray(raw[anchor["anchor_index"]], dtype=np.float32))
            same = similarity(predictor[family], actual)
            wrong = [similarity(predictor[other], actual) for other in predictor_families if other != family]
            row = {"partition": split, "family": family, "phrase_variant": anchor["phrase_variant"], "same_similarity": same, "median_wrong_similarity": float(np.median(wrong)), "max_wrong_similarity": float(np.max(wrong)), "median_wrong_advantage": same - float(np.median(wrong)), "hard_wrong_advantage": same - float(np.max(wrong))}
            rows.append(row); selected_rows.append(row)
        median_advantage = float(np.median([row["median_wrong_advantage"] for row in selected_rows]))
        positive_count = int(sum(row["median_wrong_advantage"] > 0 for row in selected_rows))
        positive_fraction = positive_count / max(len(selected_rows), 1)
        summary[split] = {"families": len(selected_rows), "median_same_similarity": float(np.median([row["same_similarity"] for row in selected_rows])), "median_wrong_advantage": median_advantage, "median_hard_wrong_advantage": float(np.median([row["hard_wrong_advantage"] for row in selected_rows])), "positive_family_count": positive_count, "positive_family_fraction": positive_fraction, "prediction_label": median_advantage >= limits["median_wrong_advantage_min"] and positive_fraction >= limits["positive_family_fraction_min"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "new_material_prediction_adjudicated", "behavior": {"global_accuracy": eligibility["global_accuracy"], "eligible_families": families, "by_family_split": eligibility["by_family_split"]}, "summary": summary, "rows": rows, "all_splits_replicated": all(value["prediction_label"] for value in summary.values()), "claim_boundary": core.load(OUT / "protocol/preregistration.json")["claim_boundary"], "next_authorization": "run_C187_signed_target_profile_and_surface_invariance" if all(value["prediction_label"] for value in summary.values()) else "run_C187_failure_decomposition_without_global_stop"}
    core.save(OUT / "analysis/prediction_atlas.json", result)
    checks = {"rows": len(rows) == 2 * len(families), "finite": all(np.isfinite([value for value in row.values() if isinstance(value, float)]).all() for row in rows), "typed": isinstance(result["all_splits_replicated"], bool)}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(result, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json")
    result = core.load(OUT / "analysis/prediction_atlas.json")
    hidden_expected = bool(core.load(OUT / "protocol/behavior_eligibility_lock.json")["eligible_families"])
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "behavior": core.load(OUT / "audit/internal_behavior_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "hidden": (not hidden_expected) or core.load(OUT / "audit/internal_hidden_run_audit.json")["all_checks_passed"], "hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": result, "next_authorization": result["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "behavior", "lock", "hidden", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "behavior": behavior, "lock": lock, "hidden": hidden, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
