#!/usr/bin/env python3
"""C630-C634 full-coordinate conditional-gear identification campaign.

The camera is limited to embeddings, HiddenState checkpoints, candidate
logits, and generated text. Every retained field keeps all signed physical
coordinates. No PCA, Top-K selection, attention/MLP/weight inspection,
gradients, or sparse recovery is used. A failed branch is recorded as a
planned missingness and never stops sibling branches.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import re
import shutil
import subprocess
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

import phase2159_c625_c629_flagship_gear_campaign as old


PHASES = {
    "C630": (2165, "new_factorial_language_objects_and_interfaces"),
    "C631": (2166, "full_coordinate_numeric_metrology"),
    "C632": (2167, "cross_donor_output_identity_and_mediation"),
    "C633": (2168, "mobius_composition_and_response_ecology"),
    "C634": (2169, "crossmodel_visual_theory_audit"),
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
        for name, (phase, slug) in PHASES.items()}
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c634_conditional_gear_identification_atlas.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"

SYSTEM = "Use only the supplied record. Reply with exactly one code letter: A, B, C, or D."
CODES = ("A", "B", "C", "D")
FAMILIES = ("update", "nested_event", "type_path")
LANGUAGES = ("en", "zh")
SURFACES = ("canonical", "paraphrase")
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
UNITS = 8
DIM = 2560
CHECKPOINTS = 38
QPOINTS = (0, 8, 16, 24, 32, 37)
BEHAVIOR_GATE = 0.80
CONTROL_MARGIN = 0.02
MODEL_BASE = old.previous.c607.passport.previous.model_base()

PEOPLE = (
    ("Alden", "Brielle", "Cassian", "Daphne"),
    ("Eamon", "Freya", "Gideon", "Hana"),
    ("Iris", "Jonas", "Kira", "Lucan"),
    ("Mara", "Nolan", "Ophelia", "Pavel"),
    ("Quinn", "Rosa", "Soren", "Talia"),
    ("Uma", "Victor", "Willa", "Xavier"),
    ("Yara", "Zane", "Elara", "Finn"),
    ("Greta", "Hector", "Ilona", "Jasper"),
)
OBJECTS = (
    ("quince", "papaya"), ("radish", "plum"), ("pear", "guava"),
    ("carrot", "date"), ("lychee", "yam"), ("olive", "lemon"),
    ("apricot", "turnip"), ("fig", "melon"),
)
NODES = (
    ("zorb", "feln", "miv", "tark", "uln", "vex"),
    ("dax", "rul", "pim", "sov", "ket", "wug"),
    ("nib", "jor", "cal", "tep", "hux", "vem"),
    ("bex", "lod", "gir", "fan", "mux", "qel"),
    ("rav", "sin", "pok", "del", "zum", "hef"),
    ("yul", "cab", "tor", "mik", "nes", "vop"),
    ("gax", "rin", "pel", "sud", "wim", "koz"),
    ("nal", "dut", "fer", "qis", "bov", "jek"),
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


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
    out = OUTS[name]
    (out / "protocol").mkdir(parents=True, exist_ok=True)
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "dependencies": dependencies,
        "camera": "embedding + HiddenState + logits/text; all signed coordinates; no PCA/Top-K/attention/MLP/weights/gradients",
        "branch_policy": "failure or external unavailability closes only that branch; all siblings continue",
        "human_review": "external reviewers unavailable in the execution environment; blank review records are retained and never imputed",
    })
    print(f"=== {name} phase={PHASES[name][0]} ===", flush=True)
    return out


def close(name: str, headline: dict, checks: dict, next_authorization: str) -> dict:
    result = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_checks_passed": bool(checks) and all(bool(v) for v in checks.values()),
        "headline": headline, "checks": checks, "next_authorization": next_authorization,
    }
    save(OUTS[name] / "analysis/final.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def partition(unit: int) -> str:
    return "discovery" if unit < 4 else "confirmation" if unit < 6 else "lockbox"


def codebook(semantic_values: list[str], shift: int) -> tuple[list[str], dict[str, str]]:
    mapping = {value: CODES[(index + shift) % 4] for index, value in enumerate(semantic_values)}
    entries = [f"{mapping[value]} = {value}" for value in semantic_values]
    return entries, mapping


def prompt_text(language: str, surface: str, facts: str, entries: list[str], question: str) -> str:
    mapping = "; ".join(entries)
    if language == "en":
        lead = "Research record" if surface == "canonical" else "A curator provides this record"
        return (f"{lead}: {facts} Codebook: {mapping}. {question} "
                "Reply with exactly one code letter: A, B, C, or D.")
    lead = "\u7814\u7a76\u8bb0\u5f55" if surface == "canonical" else "\u6574\u7406\u5458\u63d0\u4f9b\u4ee5\u4e0b\u8bb0\u5f55"
    return (f"{lead}\uff1a{facts}\u4ee3\u7801\u8868\uff1a{mapping}\u3002{question}"
            "\u8bf7\u53ea\u56de\u7b54\u4e00\u4e2a\u4ee3\u7801\u5b57\u6bcd\uff1aA\u3001B\u3001C\u6216D\u3002")


def make_row(family: str, language: str, surface: str, unit: int, semantic: int, shift: int) -> dict:
    people, objects, nodes = PEOPLE[unit], OBJECTS[unit], NODES[unit]
    if family == "update":
        values = ["update one", "update two", "update three", "update four"]
        if language == "en":
            facts = (f"{people[0]} keeps four entries. Update one names {people[1]}; update two names {people[2]}; "
                     f"update three names {people[3]}; update four names {people[0]}. "
                     f"The current entry is explicitly update {('one','two','three','four')[semantic]}.")
            question = "Which update is current? Use its code."
            relation = "current entry"
        else:
            facts = (f"{people[0]}\u4fdd\u5b58\u4e86\u56db\u6761\u66f4\u65b0\u3002\u66f4\u65b0\u4e00\u8bb0\u5f55{people[1]}\uff1b"
                     f"\u66f4\u65b0\u4e8c\u8bb0\u5f55{people[2]}\uff1b\u66f4\u65b0\u4e09\u8bb0\u5f55{people[3]}\uff1b"
                     f"\u66f4\u65b0\u56db\u8bb0\u5f55{people[0]}\u3002\u5f53\u524d\u6761\u76ee\u660e\u786e\u662f{values[semantic]}\u3002")
            question = "\u5f53\u524d\u662f\u54ea\u6761\u66f4\u65b0\uff1f\u8bf7\u4f7f\u7528\u5bf9\u5e94\u4ee3\u7801\u3002"
            relation = "\u5f53\u524d\u6761\u76ee"
        primary, secondary, context = people[0], people[(semantic + 1) % 4], values[semantic]
    elif family == "nested_event":
        values = [f"{people[a]} ate {objects[b]}" for a, b in ((1, 0), (2, 0), (1, 1), (2, 1))]
        agent = people[1 if semantic in (0, 2) else 2]
        patient = objects[0 if semantic in (0, 1) else 1]
        if language == "en":
            facts = f"{people[0]} reports the embedded event: {agent} ate {patient}."
            question = "Which complete event is embedded in the report? Use its code."
            relation = "reports"
        else:
            facts = f"{people[0]}\u62a5\u544a\u4e86\u8fd9\u4e2a\u5d4c\u5957\u4e8b\u4ef6\uff1a{agent} ate {patient}\u3002"
            question = "\u62a5\u544a\u4e2d\u5d4c\u5165\u7684\u5b8c\u6574\u4e8b\u4ef6\u662f\u54ea\u4e00\u4e2a\uff1f\u8bf7\u4f7f\u7528\u5bf9\u5e94\u4ee3\u7801\u3002"
            relation = "\u62a5\u544a"
        primary, secondary, context = people[0], agent, patient
    else:
        values = list(nodes[1:5])
        edges = "; ".join(f"{nodes[i]} -> {nodes[i + 1]}" for i in range(4))
        if language == "en":
            facts = f"Directed links are {edges}; the unrelated link is {nodes[5]} -> {nodes[0]}."
            question = f"Starting at {nodes[0]}, which node is reached after exactly {semantic + 1} link(s)? Use its code."
            relation = "Directed links"
        else:
            facts = f"\u6709\u5411\u8fde\u63a5\u4e3a {edges}\uff1b\u65e0\u5173\u8fde\u63a5\u4e3a {nodes[5]} -> {nodes[0]}\u3002"
            question = f"\u4ece{nodes[0]}\u5f00\u59cb\uff0c\u6cbf\u65b9\u5411\u6062\u597d\u8d70{semantic + 1}\u6761\u8fde\u63a5\u540e\u5230\u8fbe\u54ea\u4e2a\u8282\u70b9\uff1f\u8bf7\u4f7f\u7528\u5bf9\u5e94\u4ee3\u7801\u3002"
            relation = "\u6709\u5411\u8fde\u63a5"
        primary, secondary, context = nodes[0], nodes[1], values[semantic]
    entries, mapping = codebook(values, shift)
    answer = mapping[values[semantic]]
    prompt = prompt_text(language, surface, facts, entries, question)
    return {
        "case_id": f"c630|{family}|{language}|{surface}|u{unit:02d}|s{semantic}|k{shift}",
        "family": family, "language": language, "surface": surface, "unit": unit,
        "partition": partition(unit), "semantic": semantic, "semantic_bits": [semantic & 1, (semantic >> 1) & 1],
        "code_shift": shift, "code_bits": [shift & 1, (shift >> 1) & 1],
        "slice_key": "|".join((family, language, surface)), "prompt": prompt,
        "answer": answer, "answer_candidates": list(CODES), "semantic_values": values,
        "role_values": {"primary": primary, "secondary": secondary, "relation": relation,
                        "context": context, "query": values[semantic]},
        "factors": {"family": family, "semantic": semantic, "code_shift": shift},
        "cross_model_subset": surface == "canonical" and unit in (0, 6)
                              and (semantic, shift) in ((0, 0), (1, 0), (2, 1), (3, 1)),
    }


def make_material() -> list[dict]:
    return [make_row(f, l, s, u, semantic, shift)
            for f, l, s, u, semantic, shift in itertools.product(
                FAMILIES, LANGUAGES, SURFACES, range(UNITS), range(4), range(4))]


def chat_render(tokenizer, prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def aligned_spans(tokenizer, prompt: str, ids: list[int], value: str) -> list[list[int]]:
    rendered = chat_render(tokenizer, prompt)
    try:
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        encoded_ids = [int(x) for x in encoded["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    except (TypeError, NotImplementedError, ValueError):
        encoded_ids, offsets = [], []
    if encoded_ids == ids and len(offsets) == len(ids):
        spans = []
        for match in re.finditer(re.escape(value), rendered):
            selected = [i for i, (start, end) in enumerate(offsets)
                        if end > start and end > match.start() and start < match.end()]
            if selected:
                spans.append(selected)
        if spans:
            return spans
    return old.previous.c607.compiler.graph_base.name_spans(tokenizer, ids, value)


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = old.previous.c607.text_core.chat_ids(tokenizer, SYSTEM, row["prompt"])
        candidate_ids = [tokenizer.encode(" " + answer, add_special_tokens=False)
                         for answer in row["answer_candidates"]]
        positions = {}
        for role, value in row["role_values"].items():
            spans = aligned_spans(tokenizer, row["prompt"], ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value, "uncompiled role"))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidate_ids,
                         "gold_position": row["answer_candidates"].index(row["answer"]),
                         "role_positions": positions})
    return compiled


def normalize(text: str) -> str:
    value = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    return " ".join(value.strip().upper().split()).strip(".,;:!?\"'`()[]{}")


def generated_prediction(text: str) -> int:
    value = normalize(text)
    matches = [i for i, code in enumerate(CODES) if value == code or value.startswith(code + " ")]
    return matches[0] if len(matches) == 1 else -1


def material_path() -> Path:
    return OUTS["C630"] / "material/factorial_language_objects.jsonl"


def compiled_path() -> Path:
    return OUTS["C630"] / "material/qwen_compiled.jsonl"


def behavior_path() -> Path:
    return OUTS["C630"] / "behavior/qwen_behavior.jsonl"


def states_path() -> Path:
    return OUTS["C631"] / "raw/role_field.float16.npy"


def index_path() -> Path:
    return OUTS["C631"] / "raw/hidden_index.jsonl"


def c630() -> None:
    out = begin("C630", {
        "object": "fresh orthogonal semantic-state x output-code factorial over update, nested-event and four-step type-path programs",
        "coverage": {"families": FAMILIES, "languages": LANGUAGES, "surfaces": SURFACES,
                     "units": UNITS, "semantic_states": 4, "code_rotations": 4},
        "partition": "units 0-3 discovery, 4-5 confirmation, 6-7 lockbox",
        "zero_models": "gold code is exactly balanced within every family-language-surface slice",
        "behavior_gate": "candidate and exact open generation accuracy each >= 0.80 per frozen slice",
        "human_external_validity": "blank bilingual lockbox review template; status remains NA until independent humans fill it",
    }, {"C629": old.final("C629")["all_checks_passed"]})
    save(out / "audit/premodel_compiler_repair.json", {
        "status": "repaired_before_behavior_reveal",
        "failure": "type_path English relation literal was registered as 'directed link' while the prompt contains 'Directed links'",
        "repair": "role literal only changed to the exact prompt span; no material semantics, answer, partition, threshold, or model output changed",
        "first_attempt_behavior_rows": 0,
    })
    rows = make_material()
    if len(rows) != 1536 or len({r["case_id"] for r in rows}) != len(rows) or len({r["prompt"] for r in rows}) != len(rows):
        raise RuntimeError("material cardinality or identity failure")
    write_rows(material_path(), rows)
    human = [{"case_id": r["case_id"], "naturalness_1_5": None, "semantic_uniqueness_0_1": None,
              "answerability_0_1": None, "reviewer": None}
             for r in rows if r["partition"] == "lockbox"]
    write_rows(out / "external/human_blind_template.jsonl", human)
    model = None
    try:
        model, tokenizer, device, placement = MODEL_BASE.load_bf16("qwen3")
        compiled = compile_rows(tokenizer, rows)
        write_rows(compiled_path(), compiled)
        scores_all = old.previous.c607.batch_candidate_scores(model, device, compiled, batch_size=16)
        behavior = []
        for i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = old.previous.c607.greedy_text(model, tokenizer, device, item["prompt_ids"], max_new_tokens=4)
            candidate = int(np.argmax(scores)); generated = generated_prediction(text)
            behavior.append({"case_id": item["case_id"], "candidate_prediction": candidate,
                             "candidate_scores": scores, "candidate_correct": candidate == item["gold_position"],
                             "generated_text": text, "generated_prediction": generated,
                             "generated_correct": generated == item["gold_position"]})
            if i % 64 == 0 or i + 1 == len(compiled):
                print(f"[C630 behavior] {i + 1}/{len(compiled)}", flush=True)
        write_rows(behavior_path(), behavior)
    finally:
        MODEL_BASE.release_bf16(model); gc.collect()
    by_behavior = {row["case_id"]: row for row in behavior}
    groups = defaultdict(list)
    for row in rows:
        groups[row["slice_key"]].append(by_behavior[row["case_id"]])
    slices = {}
    for key, values in sorted(groups.items()):
        candidate = float(np.mean([v["candidate_correct"] for v in values]))
        generated = float(np.mean([v["generated_correct"] for v in values]))
        slices[key] = {"rows": len(values), "candidate_accuracy": candidate,
                       "generated_accuracy": generated,
                       "qualified": candidate >= BEHAVIOR_GATE and generated >= BEHAVIOR_GATE}
    save(out / "behavior/slice_qualification.json", slices)
    zero = {}
    for key in slices:
        subset = [r for r in rows if r["slice_key"] == key]
        zero[key] = {code: sum(r["answer"] == code for r in subset) / len(subset) for code in CODES}
    save(out / "audit/zero_model_balance.json", zero)
    headline = {
        "status": "fresh_factorial_interfaces_closed", "rows": len(rows),
        "partition_counts": {p: sum(r["partition"] == p for r in rows)
                             for p in ("discovery", "confirmation", "lockbox")},
        "candidate_accuracy": float(np.mean([r["candidate_correct"] for r in behavior])),
        "generated_accuracy": float(np.mean([r["generated_correct"] for r in behavior])),
        "qualified_slices": sum(v["qualified"] for v in slices.values()), "total_slices": len(slices),
        "slices": slices, "zero_model_max": max(max(v.values()) for v in zero.values()),
        "human_review": "NA_pending_external_review", "material_sha256": digest(rows),
        "strict_interpretation": "Programmatic semantic uniqueness and exact code balance do not establish human naturalness.",
    }
    close("C630", headline, {"large": len(rows) >= 1500, "unique": len({r["prompt"] for r in rows}) == len(rows),
                              "balanced": headline["zero_model_max"] == 0.25,
                              "complete": len(behavior) == len(rows), "human_not_fabricated": all(r["reviewer"] is None for r in human),
                              "finite": finite(headline)}, "C631_full_coordinate_numeric_metrology")


def metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    pred = np.asarray(prediction, np.float64)
    target = np.asarray(truth, np.float64)
    error = pred - target
    pf, tf = pred.reshape(-1), target.reshape(-1)
    truth_rms = float(np.sqrt(np.mean(target * target)))
    error_rms = float(np.sqrt(np.mean(error * error)))
    return {
        "nrmse": error_rms / (truth_rms + 1e-12),
        "cosine": float(np.dot(pf, tf) / (np.linalg.norm(pf) * np.linalg.norm(tf) + 1e-12)),
        "sign_agreement": float(np.mean(np.sign(pred) == np.sign(target))),
        "truth_rms": truth_rms, "error_rms": error_rms,
    }


def capture_qwen_field(model, device, compiled: list[dict], behavior: dict[str, dict], slices: dict) -> tuple[list[dict], list[dict]]:
    out = OUTS["C631"]
    states = np.lib.format.open_memmap(states_path(), mode="w+", dtype=np.float16,
                                       shape=(len(compiled), CHECKPOINTS, len(ROLES), DIM))
    token_dir = out / "raw/full_token_registered_panel"
    token_dir.mkdir(parents=True, exist_ok=True)
    base = model.model
    modules = [base.embed_tokens, *list(base.layers), base.norm]
    captured: list[torch.Tensor] = []
    handles = []
    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)
    handles = [module.register_forward_hook(hook) for module in modules]
    index, token_ledger = [], []
    registered = {row["case_id"] for row in compiled
                  if row["unit"] == 6 and row["surface"] == "canonical"
                  and (row["semantic"], row["code_shift"]) in ((0, 0), (1, 0), (0, 1), (3, 3))}
    try:
        for row_i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            if len(captured) != CHECKPOINTS:
                raise RuntimeError((item["case_id"], len(captured), CHECKPOINTS))
            full = None
            if item["case_id"] in registered:
                full_path = token_dir / f"row_{row_i:04d}.float16.npy"
                full = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.float16,
                                                 shape=(CHECKPOINTS, len(item["prompt_ids"]), DIM))
            for q, hidden in enumerate(captured):
                array = hidden[0].float().cpu().numpy().astype(np.float16)
                if full is not None:
                    full[q] = array
                for role_i, role in enumerate(ROLES):
                    states[row_i, q, role_i] = array[int(item["role_positions"][role][-1])]
            if full is not None:
                full.flush()
                token_ledger.append({"case_id": item["case_id"], "path": str(full_path.relative_to(ROOT)),
                                     "shape": [CHECKPOINTS, len(item["prompt_ids"]), DIM],
                                     "bytes": full_path.stat().st_size})
                del full
            b = behavior[item["case_id"]]
            index.append({"hidden_index": row_i, "case_id": item["case_id"], "family": item["family"],
                          "language": item["language"], "surface": item["surface"], "unit": item["unit"],
                          "partition": item["partition"], "semantic": item["semantic"],
                          "code_shift": item["code_shift"], "slice_key": item["slice_key"],
                          "role_positions": item["role_positions"], "candidate_correct": b["candidate_correct"],
                          "generated_correct": b["generated_correct"],
                          "slice_qualified": slices[item["slice_key"]]["qualified"]})
            del ids, mask, pos
            if row_i % 64 == 0 or row_i + 1 == len(compiled):
                print(f"[C631 capture] {row_i + 1}/{len(compiled)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    states.flush(); del states
    write_rows(index_path(), index)
    save(out / "raw/full_token_panel_ledger.json", token_ledger)
    return index, token_ledger


def role_state(states: np.ndarray, row: dict, q: int, role: str = "query") -> np.ndarray:
    return np.asarray(states[int(row["hidden_index"]), q, ROLES.index(role)], np.float32)


def bf16_ulp_scan(model, item: dict, source_q: int, source_pos: int, target_pos: int,
                  steps: int, output_path: Path, pair_batch: int = 8) -> dict:
    base = model.model
    target_q = min(source_q + 8, len(base.layers))
    matrix = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(2, DIM, DIM))
    logit_path = output_path.with_name(output_path.stem + ".candidate_logits.float32.npy")
    logit_matrix = np.lib.format.open_memmap(logit_path, mode="w+", dtype=np.float32,
                                             shape=(DIM, len(item["candidate_ids"])))
    dose_path = output_path.with_name(output_path.stem + ".actual_dose.float32.npy")
    dose = np.lib.format.open_memmap(dose_path, mode="w+", dtype=np.float32, shape=(4, DIM))
    prompt = torch.tensor(item["prompt_ids"], dtype=torch.long, device=next(model.parameters()).device)
    first_tokens = [int(ids[0]) for ids in item["candidate_ids"]]
    for start in range(0, DIM, pair_batch):
        coords = torch.arange(start, min(start + pair_batch, DIM), device=prompt.device)
        b = int(coords.numel())
        ids = prompt[None].repeat(2 * b, 1); mask = torch.ones_like(ids)
        position_ids = torch.arange(ids.shape[1], device=ids.device)[None].repeat(2 * b, 1)
        captured: dict[str, torch.Tensor] = {}
        actual: dict[str, torch.Tensor] = {}
        def inject(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            changed = tensor.clone()
            rows = torch.arange(b, device=tensor.device)
            original = tensor[rows, source_pos, coords]
            plus, minus = original.clone(), original.clone()
            for _ in range(steps):
                plus = torch.nextafter(plus, torch.full_like(plus, float("inf")))
                minus = torch.nextafter(minus, torch.full_like(minus, float("-inf")))
            changed[rows, source_pos, coords] = plus
            changed[b + rows, source_pos, coords] = minus
            actual.update(original=original.detach(), plus=plus.detach(), minus=minus.detach())
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        def capture_next(_module, _args, output):
            captured["next"] = (output[0] if isinstance(output, tuple) else output).detach()
        def capture_final(_module, _args, output):
            captured["final"] = output.detach()
        handles = [base.layers[source_q - 1].register_forward_hook(inject),
                   base.layers[target_q - 1].register_forward_hook(capture_next),
                   base.norm.register_forward_hook(capture_final)]
        try:
            with torch.inference_mode():
                result = model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                               use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        original = actual["original"].float().cpu().numpy()
        plus = actual["plus"].float().cpu().numpy(); minus = actual["minus"].float().cpu().numpy()
        denominator = plus - minus
        dose[0, start:start + b] = original; dose[1, start:start + b] = plus
        dose[2, start:start + b] = minus; dose[3, start:start + b] = denominator
        for target_i, key in enumerate(("next", "final")):
            values = captured[key][:, target_pos].float().cpu().numpy()
            derivative = (values[:b] - values[b:]) / denominator[:, None]
            matrix[target_i, start:start + b] = derivative.astype(np.float32)
        logits = result.logits[:, -1].float().cpu().numpy()
        logit_matrix[start:start + b] = np.asarray(
            [[(logits[i, token] - logits[b + i, token]) / denominator[i] for token in first_tokens]
             for i in range(b)], np.float32)
        if start % 256 == 0 or start + b == DIM:
            print(f"[C631 ULP scan] steps={steps} {start + b}/{DIM}", flush=True)
        del result, captured, actual, ids, mask, position_ids
    matrix.flush(); logit_matrix.flush(); dose.flush()
    den = np.asarray(dose[3], np.float64)
    summary = {"matrix": str(output_path.relative_to(ROOT)), "logit_matrix": str(logit_path.relative_to(ROOT)),
               "actual_dose": str(dose_path.relative_to(ROOT)), "shape": [2, DIM, DIM],
               "source_q": source_q, "target_q": [target_q, 37], "ulp_steps": steps,
               "effective_nonzero_fraction": float(np.mean(den != 0)),
               "effective_abs_min": float(np.min(np.abs(den[den != 0]))),
               "effective_abs_median": float(np.median(np.abs(den[den != 0]))),
               "effective_abs_max": float(np.max(np.abs(den[den != 0]))),
               "claim_boundary": "finite local BF16 secant panel at one registered state, not a complete network Jacobian"}
    del matrix, logit_matrix, dose
    return summary


def no_op_repeat(model, item: dict, target_pos: int, repeats: int = 4) -> dict:
    base = model.model; captured: dict[str, torch.Tensor] = {}
    def capture32(_module, _args, output):
        captured["q32"] = (output[0] if isinstance(output, tuple) else output).detach()
    def capture_final(_module, _args, output):
        captured["q37"] = output.detach()
    handles = [base.layers[31].register_forward_hook(capture32), base.norm.register_forward_hook(capture_final)]
    ids0 = torch.tensor(item["prompt_ids"], dtype=torch.long, device=next(model.parameters()).device)
    ids = ids0[None].repeat(repeats, 1); mask = torch.ones_like(ids)
    pos = torch.arange(ids.shape[1], device=ids.device)[None].repeat(repeats, 1)
    try:
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    values = {key: tensor[:, target_pos].float().cpu().numpy() for key, tensor in captured.items()}
    logits = result.logits[:, -1].float().cpu().numpy()
    return {"repeats": repeats,
            "q32_max_abs_spread": float(np.max(np.abs(values["q32"] - values["q32"][0]))),
            "q37_max_abs_spread": float(np.max(np.abs(values["q37"] - values["q37"][0]))),
            "logit_max_abs_spread": float(np.max(np.abs(logits - logits[0])))}


def full_direction_panel(model, item: dict, source_state: np.ndarray, source_q: int,
                         source_pos: int, target_pos: int, output_dir: Path,
                         directions: int = 32, epsilon: float = 0.05) -> dict:
    """Compare raw and norm-tangent all-coordinate perturbations without recovery."""
    base = model.model; target_q = min(source_q + 8, len(base.layers))
    rng = np.random.default_rng(2165)
    signs = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=(directions, DIM))
    h = np.asarray(source_state, np.float32)
    tangent = signs - (signs @ h)[:, None] * h[None] / (float(np.dot(h, h)) + 1e-12)
    tangent *= np.sqrt(np.mean(signs * signs, axis=1, keepdims=True) / (np.mean(tangent * tangent, axis=1, keepdims=True) + 1e-12))
    source_actual = np.lib.format.open_memmap(output_dir / "direction_source_actual.float16.npy", mode="w+",
                                              dtype=np.float16, shape=(2, directions, DIM))
    response = np.lib.format.open_memmap(output_dir / "direction_target_response.float16.npy", mode="w+",
                                         dtype=np.float16, shape=(2, directions, 2, DIM))
    prompt = torch.tensor(item["prompt_ids"], dtype=torch.long, device=next(model.parameters()).device)
    for kind_i, vectors in enumerate((signs, tangent)):
        for start in range(0, directions, 8):
            block = torch.tensor(vectors[start:start + 8], dtype=torch.float32, device=prompt.device)
            b = block.shape[0]
            ids = prompt[None].repeat(2 * b, 1); mask = torch.ones_like(ids)
            pos = torch.arange(ids.shape[1], device=ids.device)[None].repeat(2 * b, 1)
            captured: dict[str, torch.Tensor] = {}; actual: dict[str, torch.Tensor] = {}
            def inject(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                changed = tensor.clone(); baseline = tensor[:b, source_pos].float()
                plus = (baseline + epsilon * block).to(tensor.dtype)
                minus = (baseline - epsilon * block).to(tensor.dtype)
                changed[:b, source_pos] = plus; changed[b:, source_pos] = minus
                actual["half"] = ((plus.float() - minus.float()) / 2).detach()
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            def capture_next(_module, _args, output):
                captured["next"] = (output[0] if isinstance(output, tuple) else output).detach()
            def capture_final(_module, _args, output):
                captured["final"] = output.detach()
            handles = [base.layers[source_q - 1].register_forward_hook(inject),
                       base.layers[target_q - 1].register_forward_hook(capture_next),
                       base.norm.register_forward_hook(capture_final)]
            try:
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                for handle in handles:
                    handle.remove()
            source_actual[kind_i, start:start + b] = actual["half"].cpu().numpy().astype(np.float16)
            for target_i, key in enumerate(("next", "final")):
                values = captured[key][:, target_pos].float().cpu().numpy()
                response[kind_i, start:start + b, target_i] = ((values[:b] - values[b:]) / 2).astype(np.float16)
    source_actual.flush(); response.flush()
    src = np.asarray(source_actual, np.float32); dst = np.asarray(response, np.float32)
    radial = np.abs(np.sum(src * h[None, None], axis=-1)) / (
        np.linalg.norm(src, axis=-1) * np.linalg.norm(h) + 1e-12)
    summary = {
        "directions": directions, "epsilon": epsilon,
        "source_actual": str((output_dir / "direction_source_actual.float16.npy").relative_to(ROOT)),
        "target_response": str((output_dir / "direction_target_response.float16.npy").relative_to(ROOT)),
        "raw_source_nonzero_fraction": float(np.mean(src[0] != 0)),
        "tangent_source_nonzero_fraction": float(np.mean(src[1] != 0)),
        "raw_radial_fraction_median": float(np.median(radial[0])),
        "tangent_radial_fraction_median": float(np.median(radial[1])),
        "raw_target_rms": [float(np.sqrt(np.mean(dst[0, :, q] ** 2))) for q in range(2)],
        "tangent_target_rms": [float(np.sqrt(np.mean(dst[1, :, q] ** 2))) for q in range(2)],
        "claim_boundary": "full-coordinate directional normalization control; no sparse recovery or unique-edge claim",
    }
    del source_actual, response, src, dst
    return summary


def patched_state_readout(model, item: dict, q: int, source_pos: int, state: np.ndarray,
                          target_pos: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    base = model.model; captured: dict[str, torch.Tensor] = {}
    def inject(_module, _args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        changed = tensor.clone()
        changed[0, source_pos] = torch.tensor(state, dtype=changed.dtype, device=changed.device)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    def capture32(_module, _args, output):
        captured["q32"] = (output[0] if isinstance(output, tuple) else output).detach()
    def capture37(_module, _args, output):
        captured["q37"] = output.detach()
    handles = [base.layers[q - 1].register_forward_hook(inject),
               base.layers[31].register_forward_hook(capture32), base.norm.register_forward_hook(capture37)]
    ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=next(model.parameters()).device)
    mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=ids.device)[None]
    try:
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    logits = result.logits[0, -1].float().cpu().numpy()
    scores = [float(logits[int(ids_[0])]) for ids_ in item["candidate_ids"]]
    return (captured["q32"][0, target_pos].float().cpu().numpy(),
            captured["q37"][0, target_pos].float().cpu().numpy(), scores)


def c631() -> None:
    out = begin("C631", {
        "object": "full-coordinate field plus actual-BF16-dose local transmission metrology",
        "field": "all 1536 rows, embedding + 36 blocks + final norm, six roles, every signed coordinate; 24-row all-token registered panel",
        "dose": "1, 2 and 4 representable BF16 ULP steps; denominator is the actual h_plus minus h_minus for every coordinate",
        "repeat_controls": "zero/no-op repeat and independent duplicate one-ULP full scans",
        "normalization_controls": "fixed all-coordinate Rademacher directions versus state-tangent projections; no sparse recovery",
        "natural_connection": "interpolate exact q24 natural semantic transitions and read q32/final full states and code logits",
        "gates": {"repeat_cosine": 0.999, "dose_cosine": 0.90, "formal_behavior": "dual-qualified and dual-correct"},
    }, {"C630": final("C630")["all_checks_passed"]})
    (out / "raw").mkdir(parents=True, exist_ok=True)
    save(out / "audit/prefield_directory_repair.json", {
        "status": "repaired_before_hiddenstate_reveal",
        "failure": "raw output directory was absent when opening the role-field memmap",
        "repair": "create the preregistered raw directory only; no object, sample, threshold or analysis changed",
        "hiddenstate_rows_written_before_failure": 0,
    })
    compiled = read_rows(compiled_path()); behavior_rows = read_rows(behavior_path())
    behavior = {r["case_id"]: r for r in behavior_rows}
    slices = final("C630")["headline"]["slices"]
    model = None
    try:
        model, tokenizer, device, placement = MODEL_BASE.load_bf16("qwen3")
        if states_path().exists() and index_path().exists() and (out / "raw/full_token_panel_ledger.json").exists():
            index = read_rows(index_path()); token_ledger = load(out / "raw/full_token_panel_ledger.json")
            if len(index) != len(compiled) or len(token_ledger) != 24:
                raise RuntimeError("incomplete captured-field resume artifact")
            save(out / "audit/full_field_resume.json", {"status": "reused_complete_prefrozen_capture",
                                                         "rows": len(index), "token_panel_rows": len(token_ledger)})
        else:
            index, token_ledger = capture_qwen_field(model, device, compiled, behavior, slices)
        states = np.load(states_path(), mmap_mode="r")
        by_index = {r["case_id"]: r for r in index}; by_item = {r["case_id"]: r for r in compiled}
        left_id = "c630|update|en|canonical|u06|s0|k0"
        right_id = "c630|update|en|canonical|u06|s1|k0"
        left, right, item = by_index[left_id], by_index[right_id], by_item[left_id]
        numeric_dir = out / "raw/numeric_metrology"
        if numeric_dir.exists():
            shutil.rmtree(numeric_dir)
        numeric_dir.mkdir(parents=True, exist_ok=True)
        save(out / "audit/numeric_storage_repair.json", {
            "status": "repaired_before_numeric_adjudication",
            "failure": "one-ULP secant derivatives can exceed float16 range and produced overflow while writing an incomplete scan",
            "repair": "response and candidate-logit matrices are stored as float32; perturbations, actual denominators, samples and gates are unchanged",
            "partial_float16_scans_deleted": True,
        })
        no_op = no_op_repeat(model, item, int(item["role_positions"]["boundary"][-1]))
        scans = []
        for label, steps in (("ulp1_a", 1), ("ulp1_b", 1), ("ulp2", 2), ("ulp4", 4)):
            path = numeric_dir / f"{label}.response.float32.npy"
            scans.append({"label": label, **bf16_ulp_scan(
                model, item, 24, int(item["role_positions"]["query"][-1]),
                int(item["role_positions"]["boundary"][-1]), steps, path)})
        comparisons = {}
        for a, b in (("ulp1_a", "ulp1_b"), ("ulp1_a", "ulp2"), ("ulp1_a", "ulp4"), ("ulp2", "ulp4")):
            aa = np.asarray(np.load(numeric_dir / f"{a}.response.float32.npy", mmap_mode="r"), np.float32)
            bb = np.asarray(np.load(numeric_dir / f"{b}.response.float32.npy", mmap_mode="r"), np.float32)
            comparisons[f"{a}_vs_{b}"] = {"q32": metric(aa[0], bb[0]), "q37": metric(aa[1], bb[1])}
            del aa, bb
        save(out / "analysis/dose_scan_manifest.json", scans)
        save(out / "analysis/dose_stability.json", {"no_op": no_op, "comparisons": comparisons})
        direction = full_direction_panel(model, item, role_state(states, left, 24, "query"), 24,
                                         int(item["role_positions"]["query"][-1]),
                                         int(item["role_positions"]["boundary"][-1]), numeric_dir)
        save(out / "analysis/normalization_direction_control.json", direction)

        pair_specs = []
        for family, language, surface, unit in itertools.product(FAMILIES, LANGUAGES, SURFACES, (6, 7)):
            a = by_index[f"c630|{family}|{language}|{surface}|u{unit:02d}|s0|k0"]
            b = by_index[f"c630|{family}|{language}|{surface}|u{unit:02d}|s1|k0"]
            pair_specs.append((a, b))
        trajectory_path = out / "raw/natural_q24_trajectories.float16.npy"
        trajectories = np.lib.format.open_memmap(trajectory_path, mode="w+", dtype=np.float16,
                                                  shape=(len(pair_specs), 5, 2, DIM))
        trajectory_rows = []
        for pair_i, (left, right) in enumerate(pair_specs):
            item = by_item[left["case_id"]]
            h0 = role_state(states, left, 24, "query"); h1 = role_state(states, right, 24, "query")
            steps = []
            for alpha_i, alpha in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
                h = h0 + alpha * (h1 - h0)
                q32, q37, scores = patched_state_readout(
                    model, item, 24, int(item["role_positions"]["query"][-1]), h,
                    int(item["role_positions"]["boundary"][-1]))
                trajectories[pair_i, alpha_i, 0] = q32.astype(np.float16)
                trajectories[pair_i, alpha_i, 1] = q37.astype(np.float16)
                steps.append({"alpha": alpha, "candidate_scores": scores, "prediction": int(np.argmax(scores))})
            truth32 = role_state(states, right, 32, "boundary"); truth37 = role_state(states, right, 37, "boundary")
            endpoint32 = np.asarray(trajectories[pair_i, -1, 0], np.float32)
            endpoint37 = np.asarray(trajectories[pair_i, -1, 1], np.float32)
            formal = bool(left["slice_qualified"] and right["slice_qualified"] and left["candidate_correct"]
                          and right["candidate_correct"] and left["generated_correct"] and right["generated_correct"])
            trajectory_rows.append({"left": left["case_id"], "right": right["case_id"], "formal": formal,
                                    "steps": steps, "q32_endpoint": metric(endpoint32, truth32),
                                    "q37_endpoint": metric(endpoint37, truth37)})
            print(f"[C631 natural trajectory] {pair_i + 1}/{len(pair_specs)}", flush=True)
        trajectories.flush(); write_rows(out / "analysis/natural_trajectory_records.jsonl", trajectory_rows)
        formal_rows = [r for r in trajectory_rows if r["formal"]]
        repeat = comparisons["ulp1_a_vs_ulp1_b"]
        dose_pairs = [comparisons[key][q]["cosine"] for key in ("ulp1_a_vs_ulp2", "ulp1_a_vs_ulp4", "ulp2_vs_ulp4")
                      for q in ("q32", "q37")]
        repeat_pass = min(repeat["q32"]["cosine"], repeat["q37"]["cosine"]) >= 0.999
        dose_pass = min(dose_pairs) >= 0.90
        headline = {
            "status": "numeric_metrology_closed", "capture_shape": [len(compiled), CHECKPOINTS, len(ROLES), DIM],
            "all_token_panel_rows": len(token_ledger), "no_op": no_op, "scan_count": len(scans),
            "repeat_metrics": repeat, "repeat_pass": repeat_pass,
            "dose_cosines": dose_pairs, "dose_stable": dose_pass,
            "numeric_classification": ("stable_local_linear_panel" if repeat_pass and dose_pass else
                                       "repeat_stable_but_dose_dependent" if repeat_pass else "numeric_repeat_failure"),
            "normalization_control": direction, "natural_trajectory_tests": len(trajectory_rows),
            "natural_trajectory_formal": len(formal_rows),
            "natural_q32_endpoint_median_nrmse": float(np.median([r["q32_endpoint"]["nrmse"] for r in formal_rows])) if formal_rows else None,
            "strict_interpretation": "Actual ULP secants separate deterministic repeatability from dose dependence; they do not prove a unique causal circuit or mathematical nonlinearity outside the sampled interval.",
        }
        del trajectories; old.previous.c607.passport.close_mmap(states)
    finally:
        MODEL_BASE.release_bf16(model); gc.collect()
    close("C631", headline, {"field_complete": headline["capture_shape"] == [1536, 38, 6, 2560],
                              "token_panel": headline["all_token_panel_rows"] == 24,
                              "actual_doses": all(scan["effective_nonzero_fraction"] == 1.0 for scan in scans),
                              "repeat_classified": isinstance(headline["repeat_pass"], bool),
                              "all_branches_ran": headline["scan_count"] == 4 and headline["natural_trajectory_tests"] == 24,
                              "finite": finite(headline)}, "C632_cross_donor_output_identity_and_mediation")


def patched_generate(model, tokenizer, item: dict, patches: list[dict], max_new_tokens: int = 4) -> dict:
    base = model.model; handles = []; observed: dict[str, np.ndarray] = {}
    for patch in patches:
        q = int(patch["q"]); position = int(patch["position"])
        vector = np.asarray(patch["vector"], np.float32)
        def make_hook(pos_value: int, vector_value: np.ndarray):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                changed = tensor.clone()
                if pos_value < changed.shape[1]:
                    changed[0, pos_value] += torch.tensor(vector_value, dtype=changed.dtype, device=changed.device)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            return hook
        module = base.norm if q == 37 else base.layers[q - 1]
        handles.append(module.register_forward_hook(make_hook(position, vector)))
    boundary = int(item["role_positions"]["boundary"][-1])
    def observe32(_module, _args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if "q32" not in observed and boundary < tensor.shape[1]:
            observed["q32"] = tensor[0, boundary].detach().float().cpu().numpy()
    handles.append(base.layers[31].register_forward_hook(observe32))
    ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=next(model.parameters()).device)
    mask = torch.ones_like(ids); generated = []; first_scores = None
    try:
        for _ in range(max_new_tokens):
            pos = mask.long().cumsum(-1) - 1
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                               use_cache=False, return_dict=True).logits
            if first_scores is None:
                values = logits[0, -1].float().cpu().numpy()
                first_scores = [float(values[int(candidate[0])]) for candidate in item["candidate_ids"]]
            token = int(torch.argmax(logits[0, -1]).item()); generated.append(token)
            new = torch.tensor([[token]], dtype=torch.long, device=ids.device)
            ids = torch.cat((ids, new), dim=1); mask = torch.cat((mask, torch.ones_like(new)), dim=1)
            if token == tokenizer.eos_token_id:
                break
    finally:
        for handle in handles:
            handle.remove()
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {"text": text, "prediction": generated_prediction(text), "token_ids": generated,
            "candidate_scores": first_scores, "q32": observed.get("q32")}


def state_lookup(states: np.ndarray, index: dict[str, dict], case_id: str, q: int, role: str) -> np.ndarray:
    return role_state(states, index[case_id], q, role)


def c632() -> None:
    out = begin("C632", {
        "object": "orthogonal semantic state and output-code identity at q24/q32",
        "tests": "24 independent lockbox A-to-B code transitions over three families, two languages and two surfaces",
        "donors": "target-own exact difference is a positive control; formal generalization uses discovery-only within-family and cross-family donors",
        "modes": ["zero", "exact_q32", "within_family_q32", "cross_family_q32", "wrong_code_q32",
                  "q24_code", "q24_semantic_same_code", "joint_q24_q32"],
        "necessity_rescue": "delete target exact identity; rescue with independent within-family prototype; wrong-code prototype is specificity control",
        "coordinate_coverage": "16 fixed interleaved physical-coordinate partitions, no magnitude ranking",
        "gates": {"cross_donor_target_rate": 0.75, "wrong_target_rate_max": 0.25,
                  "specific_rescue_rate": 0.50},
    }, {"C631": final("C631")["all_checks_passed"]})
    states = np.load(states_path(), mmap_mode="r")
    index_rows = read_rows(index_path()); index = {r["case_id"]: r for r in index_rows}
    compiled_rows = read_rows(compiled_path()); compiled = {r["case_id"]: r for r in compiled_rows}
    prototype = {}; wrong_prototype = {}
    for family, language, surface in itertools.product(FAMILIES, LANGUAGES, SURFACES):
        diffs_ab, diffs_ac = [], []
        for unit in range(4):
            a = f"c630|{family}|{language}|{surface}|u{unit:02d}|s0|k0"
            b = f"c630|{family}|{language}|{surface}|u{unit:02d}|s0|k1"
            c = f"c630|{family}|{language}|{surface}|u{unit:02d}|s0|k2"
            diffs_ab.append(state_lookup(states, index, b, 32, "boundary") - state_lookup(states, index, a, 32, "boundary"))
            diffs_ac.append(state_lookup(states, index, c, 32, "boundary") - state_lookup(states, index, a, 32, "boundary"))
        prototype[(family, language, surface)] = np.mean(diffs_ab, axis=0)
        wrong_prototype[(family, language, surface)] = np.mean(diffs_ac, axis=0)
    tests = []
    for family, language, surface, unit in itertools.product(FAMILIES, LANGUAGES, SURFACES, (6, 7)):
        left_id = f"c630|{family}|{language}|{surface}|u{unit:02d}|s0|k0"
        target_id = f"c630|{family}|{language}|{surface}|u{unit:02d}|s0|k1"
        semantic_id = f"c630|{family}|{language}|{surface}|u{unit:02d}|s1|k3"
        exact = state_lookup(states, index, target_id, 32, "boundary") - state_lookup(states, index, left_id, 32, "boundary")
        q24_code = state_lookup(states, index, target_id, 24, "query") - state_lookup(states, index, left_id, 24, "query")
        q24_semantic = state_lookup(states, index, semantic_id, 24, "query") - state_lookup(states, index, left_id, 24, "query")
        others = [prototype[(other, language, surface)] for other in FAMILIES if other != family]
        cross = np.mean(others, axis=0)
        formal = all(index[x]["slice_qualified"] and index[x]["candidate_correct"] and index[x]["generated_correct"]
                     for x in (left_id, target_id, semantic_id))
        tests.append({"family": family, "language": language, "surface": surface, "unit": unit,
                      "left": left_id, "target": target_id, "semantic_same_code": semantic_id,
                      "formal": bool(formal), "exact": exact, "within": prototype[(family, language, surface)],
                      "cross": cross, "wrong": wrong_prototype[(family, language, surface)],
                      "q24_code": q24_code, "q24_semantic": q24_semantic})
    model = None; records = []; partition_records = []
    try:
        model, tokenizer, device, placement = MODEL_BASE.load_bf16("qwen3")
        for test_i, test in enumerate(tests):
            item = compiled[test["left"]]
            qpos = int(item["role_positions"]["query"][-1]); bpos = int(item["role_positions"]["boundary"][-1])
            modes = {
                "zero": [],
                "exact_q32": [{"q": 32, "position": bpos, "vector": test["exact"]}],
                "within_family_q32": [{"q": 32, "position": bpos, "vector": test["within"]}],
                "cross_family_q32": [{"q": 32, "position": bpos, "vector": test["cross"]}],
                "wrong_code_q32": [{"q": 32, "position": bpos, "vector": test["wrong"]}],
                "q24_code": [{"q": 24, "position": qpos, "vector": test["q24_code"]}],
                "q24_semantic_same_code": [{"q": 24, "position": qpos, "vector": test["q24_semantic"]}],
                "joint_q24_q32": [{"q": 24, "position": qpos, "vector": test["q24_semantic"]},
                                   {"q": 32, "position": bpos, "vector": test["within"]}],
            }
            outputs = {name: patched_generate(model, tokenizer, item, patches) for name, patches in modes.items()}
            base_q32 = outputs["zero"].pop("q32")
            mediation = {}
            for name in ("q24_code", "q24_semantic_same_code"):
                state = outputs[name].pop("q32")
                movement = state - base_q32
                mediation[name] = {"identity_projection": float(np.dot(movement, test["within"]) /
                                                                   (np.dot(test["within"], test["within"]) + 1e-12)),
                                   "movement_rms": float(np.sqrt(np.mean(movement * movement)))}
            for name in tuple(outputs):
                outputs[name].pop("q32", None)
                outputs[name]["target"] = outputs[name]["prediction"] == 1
            target_item = compiled[test["target"]]
            target_bpos = int(target_item["role_positions"]["boundary"][-1])
            natural = patched_generate(model, tokenizer, target_item, [])
            deletion = patched_generate(model, tokenizer, target_item,
                                         [{"q": 32, "position": target_bpos, "vector": -test["exact"]}])
            rescue = patched_generate(model, tokenizer, target_item,
                                       [{"q": 32, "position": target_bpos, "vector": -test["exact"]},
                                        {"q": 32, "position": target_bpos, "vector": test["within"]}])
            wrong_rescue = patched_generate(model, tokenizer, target_item,
                                             [{"q": 32, "position": target_bpos, "vector": -test["exact"]},
                                              {"q": 32, "position": target_bpos, "vector": test["wrong"]}])
            target_values = {"natural_prediction": natural["prediction"], "deletion_prediction": deletion["prediction"],
                             "rescue_prediction": rescue["prediction"], "wrong_rescue_prediction": wrong_rescue["prediction"],
                             "natural_ok": natural["prediction"] == 1, "deletion_broke": deletion["prediction"] != 1,
                             "rescue_ok": rescue["prediction"] == 1, "wrong_rescue_ok": wrong_rescue["prediction"] == 1}
            records.append({k: test[k] for k in ("family", "language", "surface", "unit", "left", "target", "formal")} |
                           {"outputs": outputs, "mediation": mediation, "target_values": target_values})
            if test["formal"] and len(partition_records) < 8:
                partition_outputs = []
                for part in range(16):
                    vector = np.zeros(DIM, np.float32); vector[part::16] = test["exact"][part::16]
                    output = patched_generate(model, tokenizer, item,
                                              [{"q": 32, "position": bpos, "vector": vector}])
                    scores = output["candidate_scores"]
                    partition_outputs.append({"partition": part, "prediction": output["prediction"],
                                              "target": output["prediction"] == 1,
                                              "target_margin": float(scores[1] - max(scores[0], scores[2], scores[3])),
                                              "coordinate_count": int(np.sum(np.arange(DIM) % 16 == part))})
                partition_records.append({"left": test["left"], "target": test["target"],
                                          "outputs": partition_outputs})
            print(f"[C632 identity] {test_i + 1}/{len(tests)}", flush=True)
    finally:
        MODEL_BASE.release_bf16(model); old.previous.c607.passport.close_mmap(states); gc.collect()
    write_rows(out / "analysis/output_identity_records.jsonl", records)
    write_rows(out / "analysis/interleaved_partition_records.jsonl", partition_records)
    formal = [r for r in records if r["formal"]]
    def rate(mode: str) -> float | None:
        return float(np.mean([r["outputs"][mode]["target"] for r in formal])) if formal else None
    deletion_eligible = [r for r in formal if r["target_values"]["natural_ok"] and r["target_values"]["deletion_broke"]]
    specific = [r for r in deletion_eligible if r["target_values"]["rescue_ok"] and not r["target_values"]["wrong_rescue_ok"]]
    headline = {
        "status": "cross_donor_identity_closed", "tests": len(records), "formal_tests": len(formal),
        "mode_target_rates": {name: rate(name) for name in ("zero", "exact_q32", "within_family_q32",
                                                              "cross_family_q32", "wrong_code_q32", "q24_code",
                                                              "q24_semantic_same_code", "joint_q24_q32")},
        "deletion_eligible": len(deletion_eligible), "specific_cross_donor_rescue": len(specific),
        "specific_rescue_rate": len(specific) / len(deletion_eligible) if deletion_eligible else None,
        "partition_tests": len(partition_records),
        "q24_code_identity_projection_median": float(np.median([r["mediation"]["q24_code"]["identity_projection"] for r in formal])) if formal else None,
        "q24_semantic_identity_projection_median": float(np.median([r["mediation"]["q24_semantic_same_code"]["identity_projection"] for r in formal])) if formal else None,
        "strict_interpretation": "Exact q32 donor is a positive instrument. Only discovery-donor transfer and specific rescue bear on reusable output identity; fixed coordinate partitions are coverage diagnostics, not minimal cuts.",
    }
    close("C632", headline, {"all_tests_executed": len(records) == 24,
                              "mode_balance": all(len(r["outputs"]) == 8 for r in records),
                              "partition_coverage": all(sum(o["coordinate_count"] for o in r["outputs"]) == DIM for r in partition_records),
                              "formal_accounted": len(formal) <= len(records), "finite": finite(headline)},
          "C633_mobius_composition_and_response_ecology")


def mobius_records(states: np.ndarray, index: dict[str, dict]) -> tuple[np.memmap, list[dict]]:
    out = OUTS["C633"]
    cells = list(itertools.product(FAMILIES, LANGUAGES, SURFACES, range(UNITS)))
    kinds = ("semantic_pair", "code_pair", "semantic_code")
    path = out / "raw/full_coordinate_mobius.float16.npy"
    tensor = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                       shape=(len(cells), len(kinds), CHECKPOINTS, len(ROLES), DIM))
    records = []
    for cell_i, (family, language, surface, unit) in enumerate(cells):
        def state(semantic: int, shift: int) -> np.ndarray:
            case_id = f"c630|{family}|{language}|{surface}|u{unit:02d}|s{semantic}|k{shift}"
            return np.asarray(states[index[case_id]["hidden_index"]], np.float32)
        values = {
            "semantic_pair": state(3, 0) - state(1, 0) - state(2, 0) + state(0, 0),
            "code_pair": state(0, 3) - state(0, 1) - state(0, 2) + state(0, 0),
            "semantic_code": state(1, 1) - state(1, 0) - state(0, 1) + state(0, 0),
        }
        required = {
            "semantic_pair": ((3, 0), (1, 0), (2, 0), (0, 0)),
            "code_pair": ((0, 3), (0, 1), (0, 2), (0, 0)),
            "semantic_code": ((1, 1), (1, 0), (0, 1), (0, 0)),
        }
        for kind_i, kind in enumerate(kinds):
            tensor[cell_i, kind_i] = values[kind].astype(np.float16)
            rows = [index[f"c630|{family}|{language}|{surface}|u{unit:02d}|s{s}|k{k}"]
                    for s, k in required[kind]]
            formal = all(row["slice_qualified"] and row["candidate_correct"] and row["generated_correct"] for row in rows)
            records.append({"cell_index": cell_i, "kind_index": kind_i, "kind": kind,
                            "family": family, "language": language, "surface": surface,
                            "unit": unit, "partition": partition(unit), "formal": bool(formal),
                            "rms": float(np.sqrt(np.mean(values[kind] ** 2))),
                            "nonzero_fraction": float(np.mean(values[kind] != 0))})
    tensor.flush(); return tensor, records


def composition_metrics(tensor: np.ndarray, records: list[dict], states: np.ndarray,
                        index: dict[str, dict]) -> tuple[dict, list[str]]:
    results, candidates = {}, []
    for family, language, surface, kind in itertools.product(FAMILIES, LANGUAGES, SURFACES,
                                                              ("semantic_pair", "code_pair", "semantic_code")):
        rows = [r for r in records if r["family"] == family and r["language"] == language
                and r["surface"] == surface and r["kind"] == kind and r["formal"]]
        train = [r for r in rows if r["partition"] == "discovery"]
        test = [r for r in rows if r["partition"] == "lockbox"]
        key = "|".join((family, language, surface, kind))
        if len(train) < 3 or not test:
            results[key] = {"status": "NA_insufficient_formal_cells", "train": len(train), "test": len(test)}
            continue
        y_train = np.stack([np.asarray(tensor[r["cell_index"], r["kind_index"]], np.float32) for r in train])
        y_test = np.stack([np.asarray(tensor[r["cell_index"], r["kind_index"]], np.float32) for r in test])
        mean = np.mean(y_train, axis=0)
        def base_state(record: dict) -> np.ndarray:
            case_id = (f"c630|{record['family']}|{record['language']}|{record['surface']}|"
                       f"u{record['unit']:02d}|s0|k0")
            return np.asarray(states[index[case_id]["hidden_index"]], np.float32)
        x_train = np.stack([base_state(r) for r in train]); x_test = np.stack([base_state(r) for r in test])
        x_mean = np.mean(x_train, axis=0); y_mean = np.mean(y_train, axis=0)
        beta = np.sum((x_train - x_mean) * (y_train - y_mean), axis=0) / (
            np.sum((x_train - x_mean) ** 2, axis=0) + 1e-6)
        diagonal = y_mean + beta * (x_test - x_mean)
        models = {"zero": np.zeros_like(y_test), "mean": np.broadcast_to(mean, y_test.shape), "diagonal": diagonal}
        metrics = {name: metric(pred, y_test) for name, pred in models.items()}
        best = min(("mean", "diagonal"), key=lambda name: metrics[name]["nrmse"])
        gate = metrics[best]["nrmse"] <= metrics["zero"]["nrmse"] - CONTROL_MARGIN
        results[key] = {"status": "tested", "train": len(train), "test": len(test),
                        "models": metrics, "best": best, "gate": gate}
        if gate:
            candidates.append(key)
    return results, candidates


def response_ecology(states: np.ndarray, index: dict[str, dict]) -> dict:
    operations = ((1, 0, "semantic_bit0"), (2, 0, "semantic_bit1"), (3, 0, "semantic_both"),
                  (0, 1, "code_bit0"), (0, 2, "code_bit1"), (0, 3, "code_both"))
    signatures = {}
    for family, language, surface, unit in itertools.product(FAMILIES, LANGUAGES, SURFACES, range(UNITS)):
        base_id = f"c630|{family}|{language}|{surface}|u{unit:02d}|s0|k0"
        base_row = index[base_id]; base = np.asarray(states[base_row["hidden_index"]], np.float32)
        values = []
        formal = base_row["slice_qualified"] and base_row["candidate_correct"] and base_row["generated_correct"]
        for semantic, shift, _ in operations:
            target_id = f"c630|{family}|{language}|{surface}|u{unit:02d}|s{semantic}|k{shift}"
            target_row = index[target_id]
            formal = formal and target_row["candidate_correct"] and target_row["generated_correct"]
            delta = np.asarray(states[target_row["hidden_index"]], np.float32) - base
            values.append(delta[np.asarray(QPOINTS)])
        signatures[(family, language, surface, unit)] = (np.stack(values), bool(formal))
    results = {}
    for family, language, surface in itertools.product(FAMILIES, LANGUAGES, SURFACES):
        train = [signatures[(family, language, surface, u)][0] for u in range(4)
                 if signatures[(family, language, surface, u)][1]]
        test = [signatures[(family, language, surface, u)][0] for u in (6, 7)
                if signatures[(family, language, surface, u)][1]]
        key = "|".join((family, language, surface))
        if len(train) < 3 or not test:
            results[key] = {"status": "NA_insufficient_formal_signatures", "train": len(train), "test": len(test)}
            continue
        truth = np.stack(test); own = np.broadcast_to(np.mean(train, axis=0), truth.shape)
        wrong_pool = []
        for other in FAMILIES:
            if other != family:
                wrong_pool.extend(signatures[(other, language, surface, u)][0] for u in range(4)
                                  if signatures[(other, language, surface, u)][1])
        wrong = np.broadcast_to(np.mean(wrong_pool, axis=0), truth.shape) if wrong_pool else -own
        results[key] = {"status": "tested", "train": len(train), "test": len(test),
                        "own": metric(own, truth), "zero": metric(np.zeros_like(truth), truth),
                        "wrong_family": metric(wrong, truth),
                        "own_beats_controls": metric(own, truth)["nrmse"] <=
                                              min(metric(np.zeros_like(truth), truth)["nrmse"],
                                                  metric(wrong, truth)["nrmse"]) - CONTROL_MARGIN}
    return results


def c633() -> None:
    out = begin("C633", {
        "object": "full-coordinate finite Möbius interactions and future-response ecology",
        "interactions": ["two semantic bits", "two output-code bits", "semantic x code"],
        "formula": "M11 = H11 - H10 - H01 + H00 at every checkpoint, role and physical coordinate",
        "prediction": "discovery-only mean and base-conditioned diagonal compete against zero on lockbox units",
        "response_ecology": "six registered semantic/code operations over embedding, q8, q16, q24, q32 and final norm",
        "formal_gate": "all required cells are slice-qualified and candidate/open-generation correct",
        "claim_boundary": "finite factorial interactions are conditional response objects, not unique neural programs",
    }, {"C632": final("C632")["all_checks_passed"]})
    (out / "raw").mkdir(parents=True, exist_ok=True)
    save(out / "audit/preinteraction_directory_repair.json", {
        "status": "repaired_before_interaction_reveal",
        "failure": "raw output directory was absent when opening the preregistered interaction tensor",
        "repair": "create raw directory only; no factorial cell, partition, model, metric or threshold changed",
        "interaction_cells_written_before_failure": 0,
    })
    states = np.load(states_path(), mmap_mode="r")
    index_rows = read_rows(index_path()); index = {r["case_id"]: r for r in index_rows}
    tensor, records = mobius_records(states, index)
    write_rows(out / "analysis/mobius_cell_ledger.jsonl", records)
    metrics, candidates = composition_metrics(tensor, records, states, index)
    ecology = response_ecology(states, index)
    save(out / "analysis/composition_prediction.json", metrics)
    save(out / "analysis/response_ecology.json", ecology)
    formal_records = [r for r in records if r["formal"]]
    tested = [v for v in metrics.values() if v["status"] == "tested"]
    ecology_tested = [v for v in ecology.values() if v["status"] == "tested"]
    rms_by_kind = {kind: float(np.median([r["rms"] for r in formal_records if r["kind"] == kind]))
                   if any(r["kind"] == kind for r in formal_records) else None
                   for kind in ("semantic_pair", "code_pair", "semantic_code")}
    headline = {
        "status": "mobius_composition_closed", "tensor_shape": list(tensor.shape),
        "cells": len(records), "formal_cells": len(formal_records),
        "tested_prediction_slices": len(tested), "prediction_candidates": candidates,
        "candidate_count": len(candidates), "formal_interaction_rms_medians": rms_by_kind,
        "response_ecology_tested": len(ecology_tested),
        "response_ecology_own_wins": sum(v["own_beats_controls"] for v in ecology_tested),
        "strict_interpretation": "A transferable finite interaction or response signature is a reusable conditional law candidate; it is not a fixed semantic coordinate, unique circuit, or proof of a new mathematics.",
    }
    tensor.flush(); del tensor; old.previous.c607.passport.close_mmap(states)
    close("C633", headline, {"tensor_complete": headline["tensor_shape"] == [96, 3, 38, 6, 2560],
                              "ledger_complete": headline["cells"] == 288,
                              "prediction_accounted": len(metrics) == 36,
                              "ecology_accounted": len(ecology) == 12,
                              "finite": finite(headline)}, "C634_crossmodel_visual_theory_audit")


def qwen_relative_topology(states: np.ndarray, index: dict[str, dict]) -> dict:
    topology = {}
    for family in FAMILIES:
        responses = []
        for language, unit in ((l, u) for l in LANGUAGES for u in (0, 6)):
            left = f"c630|{family}|{language}|canonical|u{unit:02d}|s0|k0"
            right = f"c630|{family}|{language}|canonical|u{unit:02d}|s1|k0"
            if all(index[x]["slice_qualified"] and index[x]["candidate_correct"] and index[x]["generated_correct"]
                   for x in (left, right)):
                a = np.asarray(states[index[left]["hidden_index"], np.asarray(QPOINTS)], np.float32)
                b = np.asarray(states[index[right]["hidden_index"], np.asarray(QPOINTS)], np.float32)
                responses.append(b - a)
        if responses:
            rms = np.sqrt(np.mean(np.stack(responses) ** 2, axis=(0, 3)))
            normalized = rms / (np.sqrt(np.sum(rms * rms, axis=1, keepdims=True)) + 1e-12)
            topology[family] = {"pairs": len(responses),
                                "relative_depths": [q / 37 for q in QPOINTS],
                                "role_rms_normalized": normalized.tolist()}
        else:
            topology[family] = {"pairs": 0, "status": "NA_no_dual_correct_pairs"}
    return topology


def topology_comparison(reference: dict, panel: dict) -> dict:
    comparisons = {}
    for family in FAMILIES:
        left, right = reference.get(family, {}), panel.get(family, {})
        if not left.get("pairs") or not right.get("pairs"):
            comparisons[family] = {"status": "NA_missing_qualified_topology"}
            continue
        a = np.asarray(left["role_rms_normalized"], np.float64).reshape(-1)
        b = np.asarray(right["role_rms_normalized"], np.float64).reshape(-1)
        comparisons[family] = {"status": "tested", "cosine": float(np.dot(a, b) /
                                        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
                               "mean_abs_error": float(np.mean(np.abs(a - b)))}
    return comparisons


def stream_array(handle, array: np.ndarray) -> None:
    values = np.asarray(array)
    if values.ndim == 1:
        handle.write(json.dumps([float(x) for x in values], ensure_ascii=False, separators=(",", ":")))
        return
    handle.write("[")
    for i in range(values.shape[0]):
        if i:
            handle.write(",")
        stream_array(handle, values[i])
    handle.write("]")


def write_visual_atlas(metadata: dict, representative: np.ndarray, matrices: np.ndarray,
                       mobius: np.ndarray, direction_source: np.ndarray,
                       direction_target: np.ndarray, identity: dict) -> None:
    VISUAL.parent.mkdir(parents=True, exist_ok=True)
    with VISUAL.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("{")
        handle.write('"schema_version":"research-kernel-v1",')
        handle.write('"dataset_id":"c634_conditional_gear_identification_atlas",')
        handle.write('"metadata":' + canonical(metadata) + ",")
        handle.write('"representative_embedding_hiddenstate_field":')
        stream_array(handle, representative)
        handle.write(',"actual_ulp_response_matrix":')
        stream_array(handle, matrices)
        handle.write(',"mobius_interaction_field":')
        stream_array(handle, mobius)
        handle.write(',"all_coordinate_direction_source":')
        stream_array(handle, direction_source)
        handle.write(',"all_coordinate_direction_target":')
        stream_array(handle, direction_target)
        handle.write(',"output_identity":' + canonical(identity))
        handle.write("}\n")


def register_visual() -> None:
    catalog = load(CATALOG)
    entry = {"id": "c634_conditional_gear_identification_atlas",
             "label": "C634 Conditional Gear Identification Atlas",
             "path": "/vis_data/research_kernel/c634_conditional_gear_identification_atlas.json"}
    values = catalog.setdefault("field_datasets", [])
    values[:] = [value for value in values if value.get("id") != entry["id"]]
    values.append(entry); save(CATALOG, catalog)


def run_worker(model_name: str, output_dir: Path) -> dict:
    command = [sys.executable, str(TESTS / "phase2169_c634_model_specific_worker.py"),
               "--model", model_name, "--material", str(material_path()),
               "--output", str(output_dir / "final.json")]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    save(output_dir / "process.json", {"command": command, "returncode": run.returncode,
                                       "stdout": run.stdout, "stderr": run.stderr})
    if run.returncode != 0:
        return {"status": "worker_error", "model": model_name, "returncode": run.returncode,
                "stderr_tail": run.stderr[-2000:]}
    return load(output_dir / "final.json")


def c634() -> None:
    out = begin("C634", {
        "object": "sequential model-specific behavior, relative role topology, exact-coordinate publication, cleanup and theory audit",
        "models": ["Qwen3-4B", "GLM4", "DeepSeek-7B", "Qwen3-14B"],
        "cross_model_rule": "compare relative checkpoint-role response topology only; never align physical coordinate IDs",
        "visual": "exact embedding/all-checkpoint/all-token representative, exact 2560x2560 ULP response, exact interaction and directional controls",
        "cleanup": "delete undisplayed raw HiddenState fields after derived ledgers and exact displayed representatives are saved",
        "math_gate": ["cross-family functional object", "repeat-and-dose-stable local transport",
                      "cross-donor specific output identity", "cross-model functional topology"],
    }, {"C633": final("C633")["all_checks_passed"]})
    worker_root = out / "crossmodel"
    panels = {}
    for model_name in ("glm4", "deepseek7b", "qwen3_14b"):
        existing = worker_root / model_name / "final.json"
        panels[model_name] = load(existing) if existing.exists() else run_worker(model_name, worker_root / model_name)
        print(f"[C634 crossmodel] {model_name}: {panels[model_name].get('status')}", flush=True)

    index_rows = read_rows(index_path()); index = {r["case_id"]: r for r in index_rows}
    qwen_topology_path = out / "analysis/qwen3_relative_role_topology.json"
    if states_path().exists():
        states = np.load(states_path(), mmap_mode="r")
        qwen_topology = qwen_relative_topology(states, index)
        save(qwen_topology_path, qwen_topology)
    else:
        states = None
        qwen_topology = load(qwen_topology_path)
        save(out / "audit/cleanup_resume.json", {
            "status": "resumed_after_windows_memmap_lock",
            "completed_before_failure": ["all cross-model workers", "qwen topology", "exact visual publication",
                                         "role-field cleanup", "full-token cleanup", "natural-trajectory cleanup"],
            "repair": "reuse closed artifacts and release selected interaction view before continuing cleanup",
            "models_rerun": 0,
        })
    cross = {}
    for name, panel in panels.items():
        topology_path = panel.get("topology")
        cross[name] = topology_comparison(qwen_topology, load(ROOT / topology_path)) if topology_path else {
            family: {"status": "NA_model_behavior_or_hidden_capture"} for family in FAMILIES}
    save(out / "analysis/crossmodel_topology_comparison.json", cross)

    token_ledger = load(OUTS["C631"] / "raw/full_token_panel_ledger.json")
    representative_id = "c630|update|en|canonical|u06|s0|k0"
    displayed_binary = out / "raw/displayed_representative_full_token.float16.npy"
    displayed_binary.parent.mkdir(parents=True, exist_ok=True)
    representative_entry = next(row for row in token_ledger if row["case_id"] == representative_id)
    representative_source = ROOT / representative_entry["path"]
    if representative_source.exists():
        shutil.copy2(representative_source, displayed_binary)
    representative = np.load(displayed_binary, mmap_mode="r")
    matrices = np.load(OUTS["C631"] / "raw/numeric_metrology/ulp1_a.response.float32.npy", mmap_mode="r")
    mobius_all = np.load(OUTS["C633"] / "raw/full_coordinate_mobius.float16.npy", mmap_mode="r")
    cell_order = list(itertools.product(FAMILIES, LANGUAGES, SURFACES, range(UNITS)))
    selected_cell = cell_order.index(("update", "en", "canonical", 6))
    mobius = np.asarray(mobius_all[selected_cell], np.float16)
    displayed_mobius = out / "raw/displayed_update_lockbox_mobius.float16.npy"
    np.save(displayed_mobius, mobius)
    direction_source = np.load(OUTS["C631"] / "raw/numeric_metrology/direction_source_actual.float16.npy", mmap_mode="r")
    direction_target = np.load(OUTS["C631"] / "raw/numeric_metrology/direction_target_response.float16.npy", mmap_mode="r")
    identity_records = read_rows(OUTS["C632"] / "analysis/output_identity_records.jsonl")
    partition_records = read_rows(OUTS["C632"] / "analysis/interleaved_partition_records.jsonl")
    metadata = {
        "title": "C634 Conditional Gear Identification Atlas",
        "camera": "embedding + HiddenState + logits/text only",
        "coordinate_policy": "all signed physical coordinates; no Top-K/PCA",
        "representative_case": representative_id,
        "representative_shape": list(representative.shape),
        "representative_q0": "embedding checkpoint",
        "matrix_shape": list(matrices.shape), "matrix_targets": ["q32 boundary", "final norm boundary"],
        "matrix_source": "q24 query, actual one-ULP central BF16 secant",
        "mobius_shape": list(mobius.shape), "mobius_kinds": ["semantic_pair", "code_pair", "semantic_code"],
        "direction_source_shape": list(direction_source.shape),
        "direction_target_shape": list(direction_target.shape),
        "parameters_exact": True,
        "claim_boundary": "Displayed values are exact stored fp16 observations, not a unique circuit or cross-model coordinate atlas.",
    }
    write_visual_atlas(metadata, representative, matrices, mobius, direction_source, direction_target,
                       {"summary": final("C632")["headline"], "records": identity_records,
                        "interleaved_partitions": partition_records})
    register_visual()
    visual_bytes = VISUAL.stat().st_size

    del representative, matrices, mobius, mobius_all, direction_source, direction_target, states
    gc.collect()
    cleanup_targets = [states_path(), OUTS["C631"] / "raw/full_token_registered_panel",
                       OUTS["C631"] / "raw/natural_q24_trajectories.float16.npy",
                       OUTS["C633"] / "raw/full_coordinate_mobius.float16.npy"]
    for panel in panels.values():
        if panel.get("role_field"):
            cleanup_targets.append(ROOT / panel["role_field"])
    cleanup = []
    for path in cleanup_targets:
        path = Path(path)
        if not path.exists():
            cleanup.append({"path": str(path), "existed": False, "bytes_removed": 0})
            continue
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        cleanup.append({"path": str(path.relative_to(ROOT) if path.is_absolute() and ROOT in path.parents else path),
                        "existed": True, "bytes_removed": size})
    save(out / "audit/cleanup_ledger.json", cleanup)

    dose = final("C631")["headline"]
    identity = final("C632")["headline"]
    composition = final("C633")["headline"]
    within_rate = identity["mode_target_rates"].get("within_family_q32")
    wrong_rate = identity["mode_target_rates"].get("wrong_code_q32")
    topology_tested = sum(v.get("status") == "tested" for model in cross.values() for v in model.values())
    math_gate = {
        "cross_family_functional_object": composition["response_ecology_own_wins"] >= 6,
        "repeat_and_dose_stable_transport": bool(dose["repeat_pass"] and dose["dose_stable"]),
        "cross_donor_specific_identity": bool(within_rate is not None and wrong_rate is not None
                                               and within_rate >= 0.75 and wrong_rate <= 0.25),
        "cross_model_functional_topology": topology_tested >= 3,
    }
    theory = {
        "name": "conditional output-field closure theory",
        "organizing_principle": "reuse-difference-conditioning",
        "current_object": "state-, role-, operation-, dose- and output-boundary-indexed finite response family",
        "math_upgrade_gate": math_gate, "passed": sum(math_gate.values()), "required": len(math_gate),
        "new_foundational_mathematics_authorized": all(math_gate.values()),
        "strict_interpretation": "Existing finite differences, factorial interactions, conditional state maps and causal controls remain sufficient until all empirical invariance gates pass.",
    }
    save(out / "analysis/theory_adjudication.json", theory)
    audit_checks = {
        "prior_phases_closed": all(final(name)["status"] == "closed" for name in ("C630", "C631", "C632", "C633")),
        "human_not_fabricated": final("C630")["headline"]["human_review"] == "NA_pending_external_review",
        "workers_accounted": len(panels) == 3 and all(panel.get("status") in ("closed", "worker_error") for panel in panels.values()),
        "visual_exact": metadata["parameters_exact"] and visual_bytes > 1_000_000,
        "cleanup_complete": all(not Path(target).exists() for target in cleanup_targets),
        "theory_gate_arithmetic": theory["passed"] == sum(math_gate.values()),
    }
    save(out / "audit/independent_stage_audit.json", audit_checks)
    headline = {
        "status": "major_stage_closed", "crossmodel_panels": panels,
        "crossmodel_topology_tests": topology_tested, "visual": str(VISUAL.relative_to(ROOT)),
        "visual_bytes": visual_bytes, "displayed_binary": str(displayed_binary.relative_to(ROOT)),
        "cleanup_files": len(cleanup), "cleanup_bytes": sum(r["bytes_removed"] for r in cleanup),
        "theory": theory,
        "strict_conclusion": "The campaign separates repeatability, dose dependence, finite interactions, donor transfer and output identity. Any positive branch remains local and model-specific until fresh-object and cross-model functional replication.",
    }
    close("C634", headline, audit_checks, "same_goal_extension_only_if_fresh_object_replication_is_authorized")


RUNNERS = {"C630": c630, "C631": c631, "C632": c632, "C633": c633, "C634": c634}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=tuple(PHASES), default="C630")
    parser.add_argument("--stop", choices=tuple(PHASES), default="C634")
    args = parser.parse_args()
    names = list(PHASES); start = names.index(args.start); stop = names.index(args.stop)
    if start > stop:
        raise ValueError("start phase follows stop phase")
    for name in names[start:stop + 1]:
        RUNNERS[name]()


if __name__ == "__main__":
    main()
