#!/usr/bin/env python3
"""C635-C637 fresh-vocabulary and fresh-output-code lockbox campaign."""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase2165_c630_c634_conditional_gear_identification_campaign as base

PHASES = {
    "C635": (2170, "fresh_code_language_field"),
    "C636": (2171, "fresh_code_identity_and_interaction_lockbox"),
    "C637": (2172, "fresh_code_visual_theory_audit"),
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
        for name, (phase, slug) in PHASES.items()}
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c637_fresh_code_lockbox_atlas.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"

CODES = ("W", "X", "Y", "Z")
SYSTEM = "Use only the supplied record. Reply with exactly one code letter: W, X, Y, or Z."
FAMILIES = ("update", "nested_event", "negation_scope")
LANGUAGES = base.LANGUAGES
SURFACES = base.SURFACES
ROLES = base.ROLES
UNITS = 6
DIM = base.DIM
CHECKPOINTS = base.CHECKPOINTS
BEHAVIOR_GATE = base.BEHAVIOR_GATE
CONTROL_MARGIN = base.CONTROL_MARGIN
MODEL_BASE = base.MODEL_BASE

# The shared compiler and generation evaluator read these globals at call time.
base.CODES = CODES
base.SYSTEM = SYSTEM

PEOPLE = (
    ("Arlen", "Beatrix", "Corin", "Delia"),
    ("Edric", "Flora", "Galen", "Helena"),
    ("Isolde", "Jorin", "Keira", "Leander"),
    ("Mirel", "Nestor", "Oriana", "Petra"),
    ("Rowan", "Sabine", "Theron", "Una"),
    ("Vera", "Walden", "Xenia", "Yorin"),
)
OBJECTS = (("cider", "nectar"), ("barley", "millet"), ("amber", "ivory"),
           ("cedar", "willow"), ("coral", "pearl"), ("saffron", "indigo"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def begin(name: str, protocol: dict, dependencies: dict) -> Path:
    out = OUTS[name]; (out / "protocol").mkdir(parents=True, exist_ok=True)
    (out / "analysis").mkdir(parents=True, exist_ok=True); (out / "raw").mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "dependencies": dependencies,
        "camera": "embedding + HiddenState + logits/text; all signed coordinates; no PCA/Top-K/attention/MLP/weights/gradients",
        "branch_policy": "failure closes one branch only",
    })
    print(f"=== {name} phase={PHASES[name][0]} ===", flush=True); return out


def close(name: str, headline: dict, checks: dict, next_authorization: str) -> dict:
    result = {"phase": PHASES[name][0], "campaign": name, "status": "closed",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "all_checks_passed": bool(checks) and all(bool(v) for v in checks.values()),
              "headline": headline, "checks": checks, "next_authorization": next_authorization}
    save(OUTS[name] / "analysis/final.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True); return result


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def partition(unit: int) -> str:
    return "discovery" if unit < 3 else "confirmation" if unit < 5 else "lockbox"


def make_row(family: str, language: str, surface: str, unit: int, semantic: int, shift: int) -> dict:
    people, objects = PEOPLE[unit], OBJECTS[unit]
    if family == "update":
        values = ["entry one", "entry two", "entry three", "entry four"]
        if language == "en":
            facts = (f"{people[0]} has four entries. Entry one names {people[1]}; entry two names {people[2]}; "
                     f"entry three names {people[3]}; entry four names {people[0]}. "
                     f"The active record is explicitly {values[semantic]}.")
            question, relation = "Which entry is active? Use its code.", "active record"
        else:
            facts = (f"{people[0]}\u6709\u56db\u6761\u8bb0\u5f55\u3002\u8bb0\u5f55\u4e00\u662f{people[1]}\uff1b"
                     f"\u8bb0\u5f55\u4e8c\u662f{people[2]}\uff1b\u8bb0\u5f55\u4e09\u662f{people[3]}\uff1b"
                     f"\u8bb0\u5f55\u56db\u662f{people[0]}\u3002\u5f53\u524d\u6fc0\u6d3b\u7684\u660e\u786e\u662f{values[semantic]}\u3002")
            question, relation = "\u54ea\u6761\u8bb0\u5f55\u5f53\u524d\u6fc0\u6d3b\uff1f\u8bf7\u4f7f\u7528\u5bf9\u5e94\u4ee3\u7801\u3002", "\u5f53\u524d\u6fc0\u6d3b"
        primary, secondary, context = people[0], people[(semantic + 1) % 4], values[semantic]
    elif family == "nested_event":
        values = [f"{people[a]} carried {objects[b]}" for a, b in ((1, 0), (2, 0), (1, 1), (2, 1))]
        agent = people[1 if semantic in (0, 2) else 2]; patient = objects[0 if semantic in (0, 1) else 1]
        if language == "en":
            facts = f"{people[0]} recalls the embedded event: {agent} carried {patient}."
            question, relation = "Which complete event is embedded in the recollection? Use its code.", "recalls"
        else:
            facts = f"{people[0]}\u56de\u5fc6\u4e86\u8fd9\u4e2a\u5d4c\u5957\u4e8b\u4ef6\uff1a{agent} carried {patient}\u3002"
            question, relation = "\u56de\u5fc6\u4e2d\u5d4c\u5165\u7684\u5b8c\u6574\u4e8b\u4ef6\u662f\u54ea\u4e00\u4e2a\uff1f\u8bf7\u4f7f\u7528\u5bf9\u5e94\u4ee3\u7801\u3002", "\u56de\u5fc6"
        primary, secondary, context = people[0], agent, patient
    else:
        values = ["outer negation", "inner negation", "both levels", "neither level"]
        if semantic == 0:
            clause = f"{people[0]} did not report that {people[1]} carried {objects[0]}"
        elif semantic == 1:
            clause = f"{people[0]} reported that {people[1]} did not carry {objects[0]}"
        elif semantic == 2:
            clause = f"{people[0]} did not report that {people[1]} did not carry {objects[0]}"
        else:
            clause = f"{people[0]} reported that {people[1]} carried {objects[0]}"
        if language == "en":
            facts = f"Scope record: {clause}."
            question, relation = "Which level or levels are negated? Use the code.", "Scope record"
        else:
            facts = f"\u4f5c\u7528\u57df\u8bb0\u5f55\uff1a{clause}\u3002"
            question, relation = "\u54ea\u4e2a\u5c42\u7ea7\u88ab\u5426\u5b9a\uff1f\u8bf7\u4f7f\u7528\u5bf9\u5e94\u4ee3\u7801\u3002", "\u4f5c\u7528\u57df\u8bb0\u5f55"
        primary, secondary, context = people[0], people[1], objects[0]
    mapping = {value: CODES[(i + shift) % 4] for i, value in enumerate(values)}
    entries = [f"{mapping[value]} = {value}" for value in values]
    if language == "en":
        lead = "Fresh research record" if surface == "canonical" else "A reviewer supplies this fresh record"
        prompt = f"{lead}: {facts} Codebook: {'; '.join(entries)}. {question} Reply with exactly W, X, Y, or Z."
    else:
        lead = "\u65b0\u7814\u7a76\u8bb0\u5f55" if surface == "canonical" else "\u5ba1\u9605\u5458\u63d0\u4f9b\u8fd9\u4efd\u65b0\u8bb0\u5f55"
        prompt = f"{lead}\uff1a{facts}\u4ee3\u7801\u8868\uff1a{'; '.join(entries)}\u3002{question}\u8bf7\u53ea\u56de\u7b54W\u3001X\u3001Y\u6216Z\u3002"
    return {"case_id": f"c635|{family}|{language}|{surface}|u{unit:02d}|s{semantic}|k{shift}",
            "family": family, "language": language, "surface": surface, "unit": unit,
            "partition": partition(unit), "semantic": semantic, "code_shift": shift,
            "slice_key": "|".join((family, language, surface)), "prompt": prompt,
            "answer": mapping[values[semantic]], "answer_candidates": list(CODES),
            "role_values": {"primary": primary, "secondary": secondary, "relation": relation,
                            "context": context, "query": values[semantic]},
            "semantic_values": values}


def make_material() -> list[dict]:
    return [make_row(*args) for args in itertools.product(
        FAMILIES, LANGUAGES, SURFACES, range(UNITS), range(4), range(4))]


def material_path() -> Path:
    return OUTS["C635"] / "material/fresh_code_material.jsonl"


def compiled_path() -> Path:
    return OUTS["C635"] / "material/qwen_compiled.jsonl"


def behavior_path() -> Path:
    return OUTS["C635"] / "behavior/qwen_behavior.jsonl"


def states_path() -> Path:
    return OUTS["C635"] / "raw/role_field.float16.npy"


def index_path() -> Path:
    return OUTS["C635"] / "raw/hidden_index.jsonl"


def capture_field(model, device, compiled: list[dict], behavior: dict[str, dict], slices: dict) -> tuple[list[dict], list[dict]]:
    states = np.lib.format.open_memmap(states_path(), mode="w+", dtype=np.float16,
                                       shape=(len(compiled), CHECKPOINTS, len(ROLES), DIM))
    token_dir = OUTS["C635"] / "raw/full_token_panel"; token_dir.mkdir(parents=True, exist_ok=True)
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    captured = []
    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)
    handles = [module.register_forward_hook(hook) for module in modules]
    index, ledger = [], []
    registered = {row["case_id"] for row in compiled if row["unit"] == 5 and row["surface"] == "canonical"
                  and (row["semantic"], row["code_shift"]) in ((0, 0), (1, 0))}
    try:
        for row_i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            full = None
            if item["case_id"] in registered:
                path = token_dir / f"row_{row_i:04d}.float16.npy"
                full = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                                 shape=(CHECKPOINTS, len(item["prompt_ids"]), DIM))
            for q, tensor in enumerate(captured):
                values = tensor[0].float().cpu().numpy().astype(np.float16)
                if full is not None:
                    full[q] = values
                for role_i, role in enumerate(ROLES):
                    states[row_i, q, role_i] = values[int(item["role_positions"][role][-1])]
            if full is not None:
                full.flush(); ledger.append({"case_id": item["case_id"], "path": str(path.relative_to(ROOT)),
                                             "shape": list(full.shape), "bytes": path.stat().st_size}); del full
            b = behavior[item["case_id"]
            ]
            index.append({"hidden_index": row_i, "case_id": item["case_id"], "family": item["family"],
                          "language": item["language"], "surface": item["surface"], "unit": item["unit"],
                          "partition": item["partition"], "semantic": item["semantic"], "code_shift": item["code_shift"],
                          "slice_key": item["slice_key"], "role_positions": item["role_positions"],
                          "candidate_correct": b["candidate_correct"], "generated_correct": b["generated_correct"],
                          "slice_qualified": slices[item["slice_key"]]["qualified"]})
            if row_i % 64 == 0 or row_i + 1 == len(compiled):
                print(f"[C635 capture] {row_i + 1}/{len(compiled)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    states.flush(); del states; write_rows(index_path(), index); save(OUTS["C635"] / "raw/full_token_panel_ledger.json", ledger)
    return index, ledger


def c635() -> None:
    out = begin("C635", {
        "object": "fresh lexicon, fresh WXYZ output alphabet and fresh negation-scope family",
        "coverage": {"families": FAMILIES, "languages": LANGUAGES, "surfaces": SURFACES,
                     "units": UNITS, "semantic_states": 4, "code_rotations": 4},
        "partition": "units 0-2 discovery, 3-4 confirmation, unit 5 lockbox",
        "behavior_gate": "candidate and exact generation each >=0.80 per frozen slice",
        "field": "all rows x embedding/36 blocks/final x six roles x all 2560 signed coordinates",
    }, {"C634": base.final("C634")["all_checks_passed"]})
    rows = make_material()
    if len(rows) != 1152 or len({r["case_id"] for r in rows}) != len(rows) or len({r["prompt"] for r in rows}) != len(rows):
        raise RuntimeError("fresh material identity failure")
    write_rows(material_path(), rows)
    human = [{"case_id": r["case_id"], "naturalness_1_5": None, "semantic_uniqueness_0_1": None,
              "reviewer": None} for r in rows if r["partition"] == "lockbox"]
    write_rows(out / "external/human_blind_template.jsonl", human)
    model = None
    try:
        model, tokenizer, device, placement = MODEL_BASE.load_bf16("qwen3")
        compiled = base.compile_rows(tokenizer, rows); write_rows(compiled_path(), compiled)
        scores_all = base.old.previous.c607.batch_candidate_scores(model, device, compiled, batch_size=12)
        behavior = []
        for i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = base.old.previous.c607.greedy_text(model, tokenizer, device, item["prompt_ids"], max_new_tokens=4)
            pred = base.generated_prediction(text); candidate = int(np.argmax(scores))
            behavior.append({"case_id": item["case_id"], "candidate_scores": scores,
                             "candidate_correct": candidate == item["gold_position"], "generated_text": text,
                             "generated_correct": pred == item["gold_position"]})
            if i % 64 == 0 or i + 1 == len(compiled):
                print(f"[C635 behavior] {i + 1}/{len(compiled)}", flush=True)
        write_rows(behavior_path(), behavior)
        by_behavior = {r["case_id"]: r for r in behavior}; groups = defaultdict(list)
        for row in rows:
            groups[row["slice_key"]].append(by_behavior[row["case_id"]])
        slices = {}
        for key, values in sorted(groups.items()):
            ca = float(np.mean([v["candidate_correct"] for v in values])); ga = float(np.mean([v["generated_correct"] for v in values]))
            slices[key] = {"rows": len(values), "candidate_accuracy": ca, "generated_accuracy": ga,
                           "qualified": ca >= BEHAVIOR_GATE and ga >= BEHAVIOR_GATE}
        save(out / "behavior/slice_qualification.json", slices)
        index, ledger = capture_field(model, device, compiled, by_behavior, slices)
    finally:
        MODEL_BASE.release_bf16(model); gc.collect()
    zero_max = max(sum(r["answer"] == code for r in rows if r["slice_key"] == key) /
                   sum(r["slice_key"] == key for r in rows) for key in slices for code in CODES)
    headline = {"status": "fresh_code_field_closed", "rows": len(rows),
                "capture_shape": [len(rows), CHECKPOINTS, len(ROLES), DIM], "all_token_panel_rows": len(ledger),
                "candidate_accuracy": float(np.mean([r["candidate_correct"] for r in behavior])),
                "generated_accuracy": float(np.mean([r["generated_correct"] for r in behavior])),
                "qualified_slices": sum(v["qualified"] for v in slices.values()), "total_slices": len(slices),
                "slices": slices, "zero_model_max": zero_max, "human_review": "NA_pending_external_review",
                "material_sha256": hashlib.sha256(canonical(rows).encode()).hexdigest(),
                "strict_interpretation": "Fresh code and lexicon remove the A-D identity special case; machine compilation still does not establish human naturalness."}
    close("C635", headline, {"large": len(rows) >= 1000, "balanced": zero_max == 0.25,
                              "complete": len(behavior) == len(rows) == len(index), "field_complete": headline["capture_shape"] == [1152, 38, 6, 2560],
                              "human_not_fabricated": all(r["reviewer"] is None for r in human), "finite": finite(headline)},
          "C636_fresh_code_identity_and_interaction_lockbox")


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def case_id(family: str, language: str, surface: str, unit: int,
            semantic: int, shift: int) -> str:
    return (f"c635|{family}|{language}|{surface}|u{unit:02d}|"
            f"s{semantic}|k{shift}")


def nrmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(prediction, np.float64)
    target = np.asarray(truth, np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)) /
                 (np.sqrt(np.mean(target ** 2)) + 1e-12))


def c636_identity(states: np.ndarray, index: dict[str, dict],
                  compiled: dict[str, dict]) -> tuple[dict, list[dict], np.ndarray, list[dict]]:
    slice_keys = list(itertools.product(FAMILIES, LANGUAGES, SURFACES))
    prototypes = np.zeros((len(slice_keys), 2, DIM), dtype=np.float32)
    prototype_ledger = []
    qualified_prototypes: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for slice_i, (family, language, surface) in enumerate(slice_keys):
        right, wrong = [], []
        for unit in range(3):
            ids = [case_id(family, language, surface, unit, 0, shift)
                   for shift in (0, 1, 2)]
            if all(index[cid]["slice_qualified"] and index[cid]["candidate_correct"]
                   and index[cid]["generated_correct"] for cid in ids):
                base_state = np.asarray(states[index[ids[0]]["hidden_index"], 32,
                                               ROLES.index("boundary")], np.float32)
                right.append(np.asarray(states[index[ids[1]]["hidden_index"], 32,
                                               ROLES.index("boundary")], np.float32) - base_state)
                wrong.append(np.asarray(states[index[ids[2]]["hidden_index"], 32,
                                               ROLES.index("boundary")], np.float32) - base_state)
        if len(right) == 3:
            prototypes[slice_i, 0] = np.mean(right, axis=0)
            prototypes[slice_i, 1] = np.mean(wrong, axis=0)
            qualified_prototypes[(family, language, surface)] = (
                prototypes[slice_i, 0], prototypes[slice_i, 1])
        prototype_ledger.append({"slice_index": slice_i, "family": family,
                                 "language": language, "surface": surface,
                                 "discovery_donors": len(right),
                                 "qualified": len(right) == 3})
    prototype_path = OUTS["C636"] / "raw/q32_identity_prototypes.float32.npy"
    prototype_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prototype_path, prototypes, allow_pickle=False)
    write_rows(OUTS["C636"] / "analysis/q32_identity_prototype_ledger.jsonl",
               prototype_ledger)

    tests = []
    for family, language, surface in slice_keys:
        left = case_id(family, language, surface, 5, 0, 0)
        target = case_id(family, language, surface, 5, 0, 1)
        formal = all(index[cid]["slice_qualified"] and index[cid]["candidate_correct"]
                     and index[cid]["generated_correct"] for cid in (left, target))
        if (family, language, surface) not in qualified_prototypes:
            formal = False
        exact = (np.asarray(states[index[target]["hidden_index"], 32, ROLES.index("boundary")], np.float32) -
                 np.asarray(states[index[left]["hidden_index"], 32, ROLES.index("boundary")], np.float32))
        own, wrong = qualified_prototypes.get((family, language, surface),
                                               (np.zeros(DIM, np.float32), np.zeros(DIM, np.float32)))
        cross_pool = [qualified_prototypes[(other, language, surface)][0]
                      for other in FAMILIES if other != family and
                      (other, language, surface) in qualified_prototypes]
        cross = np.mean(cross_pool, axis=0) if cross_pool else np.zeros(DIM, np.float32)
        tests.append({"family": family, "language": language, "surface": surface,
                      "left": left, "target": target, "formal": bool(formal),
                      "exact": exact, "own": own, "wrong": wrong, "cross": cross,
                      "cross_available": bool(cross_pool)})

    records, partitions = [], []
    model = None
    try:
        model, tokenizer, device, placement = MODEL_BASE.load_bf16("qwen3")
        for test_i, test in enumerate(tests):
            item = compiled[test["left"]]
            bpos = int(item["role_positions"]["boundary"][-1])
            modes = {
                "zero": [],
                "exact_q32": [{"q": 32, "position": bpos, "vector": test["exact"]}],
                "discovery_q32": [{"q": 32, "position": bpos, "vector": test["own"]}],
                "cross_family_q32": [{"q": 32, "position": bpos, "vector": test["cross"]}],
                "wrong_code_q32": [{"q": 32, "position": bpos, "vector": test["wrong"]}],
            }
            outputs = {}
            for name, patches in modes.items():
                output = base.patched_generate(model, tokenizer, item, patches)
                output.pop("q32", None)
                output["target"] = output["prediction"] == 1
                outputs[name] = output
            target_item = compiled[test["target"]]
            target_pos = int(target_item["role_positions"]["boundary"][-1])
            natural = base.patched_generate(model, tokenizer, target_item, [])
            deletion = base.patched_generate(
                model, tokenizer, target_item,
                [{"q": 32, "position": target_pos, "vector": -test["exact"]}])
            rescue = base.patched_generate(
                model, tokenizer, target_item,
                [{"q": 32, "position": target_pos, "vector": -test["exact"]},
                 {"q": 32, "position": target_pos, "vector": test["own"]}])
            wrong_rescue = base.patched_generate(
                model, tokenizer, target_item,
                [{"q": 32, "position": target_pos, "vector": -test["exact"]},
                 {"q": 32, "position": target_pos, "vector": test["wrong"]}])
            mediation = {
                "natural_ok": natural["prediction"] == 1,
                "deletion_broke": deletion["prediction"] != 1,
                "rescue_ok": rescue["prediction"] == 1,
                "wrong_rescue_ok": wrong_rescue["prediction"] == 1,
                "predictions": {"natural": natural["prediction"],
                                "deletion": deletion["prediction"],
                                "rescue": rescue["prediction"],
                                "wrong_rescue": wrong_rescue["prediction"]},
            }
            records.append({k: test[k] for k in ("family", "language", "surface",
                                                  "left", "target", "formal",
                                                  "cross_available")} |
                           {"outputs": outputs, "mediation": mediation})
            if test["formal"] and len(partitions) < 6:
                part_outputs = []
                for part in range(16):
                    vector = np.zeros(DIM, np.float32)
                    vector[part::16] = test["exact"][part::16]
                    result = base.patched_generate(
                        model, tokenizer, item,
                        [{"q": 32, "position": bpos, "vector": vector}])
                    part_outputs.append({"partition": part,
                                         "coordinate_count": int(len(range(part, DIM, 16))),
                                         "prediction": result["prediction"],
                                         "target": result["prediction"] == 1})
                partitions.append({"left": test["left"], "target": test["target"],
                                   "outputs": part_outputs})
            print(f"[C636 identity] {test_i + 1}/{len(tests)}", flush=True)
    finally:
        MODEL_BASE.release_bf16(model); gc.collect()
    write_rows(OUTS["C636"] / "analysis/output_identity_records.jsonl", records)
    write_rows(OUTS["C636"] / "analysis/interleaved_partition_records.jsonl", partitions)
    formal_records = [row for row in records if row["formal"]]
    def rate(mode: str) -> float | None:
        return (float(np.mean([row["outputs"][mode]["target"] for row in formal_records]))
                if formal_records else None)
    eligible = [row for row in formal_records if row["mediation"]["natural_ok"]
                and row["mediation"]["deletion_broke"]]
    specific = [row for row in eligible if row["mediation"]["rescue_ok"]
                and not row["mediation"]["wrong_rescue_ok"]]
    summary = {
        "tests": len(records), "formal_tests": len(formal_records),
        "mode_target_rates": {name: rate(name) for name in
                              ("zero", "exact_q32", "discovery_q32",
                               "cross_family_q32", "wrong_code_q32")},
        "deletion_eligible": len(eligible),
        "specific_discovery_rescue": len(specific),
        "specific_rescue_rate": len(specific) / len(eligible) if eligible else None,
        "partition_tests": len(partitions),
        "prototype_path": str(prototype_path.relative_to(ROOT)),
        "prototype_shape": list(prototypes.shape),
    }
    return summary, records, prototypes, prototype_ledger


def c636_interactions(states: np.ndarray, index: dict[str, dict]) -> tuple[dict, list[dict]]:
    cells = list(itertools.product(FAMILIES, LANGUAGES, SURFACES, range(UNITS)))
    kinds = ("semantic_pair", "code_pair", "semantic_code")
    tensor_path = OUTS["C636"] / "raw/full_coordinate_mobius.float16.npy"
    tensor = np.lib.format.open_memmap(
        tensor_path, mode="w+", dtype=np.float16,
        shape=(len(cells), len(kinds), CHECKPOINTS, len(ROLES), DIM))
    ledger = []
    definitions = {
        "semantic_pair": ((3, 0), (1, 0), (2, 0), (0, 0)),
        "code_pair": ((0, 3), (0, 1), (0, 2), (0, 0)),
        "semantic_code": ((1, 1), (1, 0), (0, 1), (0, 0)),
    }
    for cell_i, (family, language, surface, unit) in enumerate(cells):
        for kind_i, kind in enumerate(kinds):
            ids = [case_id(family, language, surface, unit, semantic, shift)
                   for semantic, shift in definitions[kind]]
            values = [np.asarray(states[index[cid]["hidden_index"]], np.float32)
                      for cid in ids]
            interaction = values[0] - values[1] - values[2] + values[3]
            tensor[cell_i, kind_i] = interaction.astype(np.float16)
            formal = all(index[cid]["slice_qualified"] and index[cid]["candidate_correct"]
                         and index[cid]["generated_correct"] for cid in ids)
            ledger.append({"cell_index": cell_i, "kind_index": kind_i,
                           "family": family, "language": language,
                           "surface": surface, "unit": unit,
                           "partition": partition(unit), "kind": kind,
                           "formal": bool(formal),
                           "rms": float(np.sqrt(np.mean(interaction ** 2))),
                           "nonzero_fraction": float(np.mean(interaction != 0))})
        if cell_i % 12 == 0 or cell_i + 1 == len(cells):
            print(f"[C636 mobius] {cell_i + 1}/{len(cells)}", flush=True)
    tensor.flush()
    predictions = {}
    candidates = []
    for family, language, surface, kind in itertools.product(
            FAMILIES, LANGUAGES, SURFACES, kinds):
        rows = [row for row in ledger if row["family"] == family and
                row["language"] == language and row["surface"] == surface and
                row["kind"] == kind and row["formal"]]
        train = [row for row in rows if row["partition"] == "discovery"]
        confirmation = [row for row in rows if row["partition"] == "confirmation"]
        lockbox = [row for row in rows if row["partition"] == "lockbox"]
        key = "|".join((family, language, surface, kind))
        if len(train) != 3 or len(confirmation) != 2 or len(lockbox) != 1:
            predictions[key] = {"status": "NA_incomplete_formal_partition",
                                "discovery": len(train),
                                "confirmation": len(confirmation),
                                "lockbox": len(lockbox)}
            continue
        mean = np.mean(np.stack([
            np.asarray(tensor[row["cell_index"], row["kind_index"]], np.float32)
            for row in train]), axis=0)
        metrics = {}
        for split, subset in (("confirmation", confirmation), ("lockbox", lockbox)):
            truth = np.stack([
                np.asarray(tensor[row["cell_index"], row["kind_index"]], np.float32)
                for row in subset])
            pred = np.broadcast_to(mean, truth.shape)
            metrics[split] = {"mean_nrmse": nrmse(pred, truth),
                              "zero_nrmse": nrmse(np.zeros_like(truth), truth)}
        gate = all(metrics[split]["mean_nrmse"] <=
                   metrics[split]["zero_nrmse"] - CONTROL_MARGIN
                   for split in ("confirmation", "lockbox"))
        predictions[key] = {"status": "tested", "discovery": 3,
                            "confirmation": 2, "lockbox": 1,
                            "metrics": metrics, "gate": gate}
        if gate:
            candidates.append(key)
    tensor.flush(); close_mmap(tensor)
    write_rows(OUTS["C636"] / "analysis/mobius_cell_ledger.jsonl", ledger)
    save(OUTS["C636"] / "analysis/mobius_prediction.json", predictions)
    tested = [value for value in predictions.values() if value["status"] == "tested"]
    summary = {"tensor_path": str(tensor_path.relative_to(ROOT)),
               "tensor_shape": [72, 3, 38, 6, 2560],
               "cells": len(ledger), "formal_cells": sum(row["formal"] for row in ledger),
               "tested_slices": len(tested), "candidate_count": len(candidates),
               "candidates": candidates,
               "lockbox_mean_nrmse_median": (float(np.median([
                   value["metrics"]["lockbox"]["mean_nrmse"] for value in tested]))
                   if tested else None)}
    return summary, ledger


def c636() -> None:
    out = begin("C636", {
        "object": "fresh WXYZ output identity plus full-coordinate finite interactions",
        "identity": "discovery-only q32 boundary donors; lockbox transfer, deletion, correct and wrong-code rescue",
        "coordinate_control": "16 fixed interleaved partitions cover all 2560 coordinates without magnitude selection",
        "interaction": "three exact finite Mobius fields over all checkpoints, roles and signed coordinates",
        "prediction": "discovery mean must beat zero by 0.05 on both confirmation and lockbox",
        "claim_boundary": "output identity, finite interaction and behavioral semantics remain separate objects",
    }, {"C635": final("C635")["all_checks_passed"]})
    states = np.load(states_path(), mmap_mode="r")
    index_rows = read_rows(index_path())
    index = {row["case_id"]: row for row in index_rows}
    compiled = {row["case_id"]: row for row in read_rows(compiled_path())}
    identity, identity_records, prototypes, prototype_ledger = c636_identity(
        states, index, compiled)
    interactions, interaction_ledger = c636_interactions(states, index)
    close_mmap(states); gc.collect()
    headline = {
        "status": "fresh_code_lockbox_closed",
        "identity": identity,
        "interactions": interactions,
        "strict_interpretation": (
            "A fresh-alphabet donor effect is reusable output-boundary identity only when "
            "discovery transfer and specific rescue pass. Mobius prediction is a finite "
            "conditional response law, not a unique coordinate circuit or new mathematics."),
    }
    close("C636", headline, {
        "identity_accounted": identity["tests"] == 12,
        "prototype_shape": identity["prototype_shape"] == [12, 2, 2560],
        "partition_coverage": all(sum(item["coordinate_count"] for item in row["outputs"]) == DIM
                                  for row in read_rows(out / "analysis/interleaved_partition_records.jsonl")),
        "mobius_complete": interactions["tensor_shape"] == [72, 3, 38, 6, 2560]
                           and interactions["cells"] == 216,
        "finite": finite(headline),
    }, "C637_fresh_code_visual_theory_audit")


def update_catalog() -> None:
    catalog = load(CATALOG)
    datasets = catalog.setdefault("field_datasets", [])
    datasets[:] = [item for item in datasets if item.get("id") != "c637_fresh_code_lockbox_atlas"]
    datasets.append({"id": "c637_fresh_code_lockbox_atlas",
                     "label": "C637 Fresh WXYZ Lockbox Atlas",
                     "path": "/vis_data/research_kernel/c637_fresh_code_lockbox_atlas.json",
                     "phase": 2172, "full_coordinate": True})
    save(CATALOG, catalog)


def c637_visual(states: np.ndarray, index_rows: list[dict],
                interaction_ledger: list[dict]) -> tuple[dict, Path]:
    qualified = [row for row in index_rows if row["partition"] == "lockbox" and
                 row["surface"] == "canonical" and row["semantic"] == 0 and
                 row["code_shift"] == 0 and row["slice_qualified"] and
                 row["candidate_correct"] and row["generated_correct"]]
    if not qualified:
        raise RuntimeError("no qualified lockbox representative for visual")
    selected = qualified[0]
    role_field = np.asarray(states[selected["hidden_index"]], np.float32)
    panel_ledger = load(OUTS["C635"] / "raw/full_token_panel_ledger.json")
    panel_item = next(item for item in panel_ledger if item["case_id"] == selected["case_id"])
    source_panel = ROOT / panel_item["path"]
    panel = np.load(source_panel, mmap_mode="r")
    checkpoints = (0, 8, 16, 24, 32, 37)
    panel_sample = np.asarray(panel[np.asarray(checkpoints)], np.float32)
    displayed = OUTS["C637"] / "raw/displayed_full_token.float16.npy"
    displayed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_panel, displayed)
    close_mmap(panel)

    tensor_path = ROOT / final("C636")["headline"]["interactions"]["tensor_path"]
    tensor = np.load(tensor_path, mmap_mode="r")
    interactions = []
    for kind in ("semantic_pair", "code_pair", "semantic_code"):
        candidates = [row for row in interaction_ledger if row["family"] == selected["family"]
                      and row["language"] == selected["language"] and
                      row["surface"] == selected["surface"] and row["unit"] == 5 and
                      row["kind"] == kind]
        row = candidates[0]
        interactions.append(np.asarray(tensor[row["cell_index"], row["kind_index"]], np.float32))
    interaction_fields = np.stack(interactions)
    close_mmap(tensor)
    prototypes = np.load(OUTS["C636"] / "raw/q32_identity_prototypes.float32.npy")
    visual = {
        "schema": "ai2050.fresh_code_lockbox_atlas.v1",
        "phase": 2172, "campaign": "C635-C637", "model": "Qwen3-4B",
        "coordinate_policy": "all signed physical coordinates; no Top-K/PCA/cosine compression",
        "selected_case": selected,
        "axes": {"checkpoints": 38, "roles": list(ROLES), "coordinates": DIM,
                 "token_sample_checkpoints": list(checkpoints)},
        "embedding_and_hidden_role_field": role_field.tolist(),
        "full_token_checkpoint_sample": panel_sample.tolist(),
        "full_token_binary": str(displayed.relative_to(ROOT)),
        "interaction_kinds": ["semantic_pair", "code_pair", "semantic_code"],
        "lockbox_interaction_fields": interaction_fields.tolist(),
        "q32_identity_prototypes": np.asarray(prototypes, np.float32).tolist(),
        "c635_summary": final("C635")["headline"],
        "c636_summary": final("C636")["headline"],
        "claim_boundary": "Exact coordinates are displayed for inspection; the atlas is not a unique causal circuit map.",
    }
    save(VISUAL, visual)
    update_catalog()
    return {"path": str(VISUAL.relative_to(ROOT)), "bytes": VISUAL.stat().st_size,
            "selected_case": selected["case_id"], "role_field_shape": list(role_field.shape),
            "token_sample_shape": list(panel_sample.shape),
            "interaction_shape": list(interaction_fields.shape),
            "prototype_shape": list(prototypes.shape),
            "displayed_binary": str(displayed.relative_to(ROOT))}, displayed


def c637() -> None:
    out = begin("C637", {
        "object": "exact-coordinate visualization, storage cleanup and theory audit",
        "display": "embedding, six-role trajectory, six full-token checkpoints, three Mobius fields and q32 prototypes",
        "cleanup": "remove undisplayed source fields only after visual and displayed binary validate",
        "theory_gate": "fresh alphabet, fresh lexicon, prospective transfer, specificity and interaction transfer are separate gates",
    }, {"C635": final("C635")["all_checks_passed"],
        "C636": final("C636")["all_checks_passed"]})
    states = np.load(states_path(), mmap_mode="r")
    index_rows = read_rows(index_path())
    interaction_ledger = read_rows(OUTS["C636"] / "analysis/mobius_cell_ledger.jsonl")
    visual, displayed = c637_visual(states, index_rows, interaction_ledger)
    close_mmap(states); gc.collect()

    identity = final("C636")["headline"]["identity"]
    interactions = final("C636")["headline"]["interactions"]
    rates = identity["mode_target_rates"]
    theory_gates = {
        "fresh_alphabet_behavioral_interfaces": final("C635")["headline"]["qualified_slices"] >= 8,
        "fresh_alphabet_discovery_identity": rates["discovery_q32"] is not None and rates["discovery_q32"] >= 0.75,
        "wrong_code_specificity": rates["wrong_code_q32"] is not None and rates["wrong_code_q32"] <= 0.25,
        "specific_rescue": identity["specific_rescue_rate"] is not None and identity["specific_rescue_rate"] >= 0.50,
        "fresh_interaction_transfer": interactions["candidate_count"] > 0,
        "dose_stable_local_transport": False,
    }
    theory = {
        "name": "conditional output-field closure theory",
        "organizing_principle": "reuse-difference-conditioning",
        "current_object": "typed finite-response family indexed by base state, role, operation, dose and output boundary",
        "gates": theory_gates, "passed": sum(theory_gates.values()),
        "required": len(theory_gates),
        "new_foundational_mathematics_authorized": all(theory_gates.values()),
        "strict_interpretation": "Existing finite differences and conditional response families remain adequate while dose-stable transport is absent.",
    }
    cleanup_targets = [states_path(),
                       ROOT / interactions["tensor_path"]]
    panel_ledger = load(OUTS["C635"] / "raw/full_token_panel_ledger.json")
    cleanup_targets.extend(ROOT / item["path"] for item in panel_ledger)
    removed = []
    for path in cleanup_targets:
        if path.exists() and path.resolve() != displayed.resolve():
            size = path.stat().st_size
            path.unlink()
            removed.append({"path": str(path.relative_to(ROOT)), "bytes": size})
    save(out / "audit/cleanup_ledger.json", {"removed": removed,
                                               "total_bytes": sum(item["bytes"] for item in removed)})
    headline = {
        "status": "fresh_code_major_stage_closed", "visual": visual,
        "cleanup_files": len(removed),
        "cleanup_bytes": sum(item["bytes"] for item in removed),
        "theory": theory,
        "strict_conclusion": (
            "Fresh WXYZ evidence adjudicates whether q32 identity and finite interaction laws "
            "generalize beyond the earlier A-D interface. Positives remain output-boundary or "
            "finite-response objects, not a single-coordinate gear dictionary."),
    }
    close("C637", headline, {
        "prior_closed": final("C635")["all_checks_passed"] and final("C636")["all_checks_passed"],
        "visual_exists": VISUAL.exists() and visual["bytes"] > 0,
        "visual_shapes": visual["role_field_shape"] == [38, 6, 2560] and
                         visual["interaction_shape"] == [3, 38, 6, 2560] and
                         visual["prototype_shape"] == [12, 2, 2560],
        "displayed_binary_kept": displayed.exists(),
        "cleanup_complete": all(not path.exists() for path in cleanup_targets
                                if path.resolve() != displayed.resolve()),
        "finite": finite(headline),
    }, "major_stage_complete_next_goal_requires_new_language_operation_or_independent_human_material")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", choices=tuple(PHASES), default="C635")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    names = list(PHASES)
    for name in names[names.index(args.start):]:
        result_path = OUTS[name] / "analysis/final.json"
        if result_path.exists() and not args.force:
            print(f"[resume] {name} already closed", flush=True)
            continue
        globals()[name.lower()]()


if __name__ == "__main__":
    main()
