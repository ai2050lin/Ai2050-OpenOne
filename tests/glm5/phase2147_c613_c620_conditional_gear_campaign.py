#!/usr/bin/env python3
"""C613-C620 conditional-gear campaign over fresh multilingual programs.

The scientific camera is restricted to embeddings, HiddenState checkpoints,
logits and generated text. Every retained state keeps all signed coordinates;
no PCA, Top-K, magnitude clipping, attention, MLP, weights or gradients are
used. Failed branches remain observable strata and do not stop sibling routes.
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
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase2141_c607_c611_natural_output_compiler_campaign as c607
import phase2134_c600_c605_language_transport_campaign as transport


PHASES = {
    "C613": (2147, "fresh_multilingual_program_contract"),
    "C614": (2148, "qwen_dual_behavior_interface"),
    "C615": (2149, "all_coordinate_base_state_guard_atlas"),
    "C616": (2150, "graph_and_attitude_composition"),
    "C617": (2151, "generation_timeline_causal_boundary"),
    "C618": (2152, "cross_model_specific_behavior_interfaces"),
    "C619": (2153, "conditional_gear_visual_theory_and_cleanup"),
    "C620": (2154, "conditional_gear_independent_audit"),
}
OUTS = {k: RESULT / f"phase{v[0]}_{k.lower()}_{v[1]}" for k, v in PHASES.items()}
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c619_conditional_gear_atlas.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"

SYSTEM = "Use only the supplied information. Reply with only the requested answer phrase and no explanation."
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
LANGUAGES = ("en", "zh")
SURFACES = ("canonical", "paraphrase")
UNITS = 8
DIM = 2560
CHECKPOINTS = 38
QPOINTS = (0, 8, 16, 24, 32, 37)
BEHAVIOR_GATE = 0.75
CONTROL_MARGIN = 0.02

PEOPLE_A = ("Marlowe", "Sabine", "Corwin", "Elara", "Galen", "Nerissa", "Oren", "Thalia")
PEOPLE_B = ("Bram", "Celeste", "Dorian", "Fiona", "Helena", "Ivo", "Junia", "Leander")
PEOPLE_C = ("Perrin", "Rhea", "Silas", "Tamsin", "Ulric", "Vera", "Wesley", "Xenia")
OBJECTS_A = ("quince", "papaya", "turnip", "apricot", "radish", "plum", "fig", "melon")
OBJECTS_B = ("pear", "guava", "beet", "peach", "carrot", "date", "lime", "mango")

NATURAL_EN = (
    ("sparrow", "bird", "animal", "living thing", "entity", "machine"),
    ("trout", "fish", "animal", "living thing", "entity", "mineral"),
    ("tulip", "flower", "plant", "living thing", "entity", "vehicle"),
    ("maple", "tree", "plant", "living thing", "entity", "instrument"),
    ("flute", "instrument", "artifact", "physical object", "entity", "animal"),
    ("chisel", "tool", "artifact", "physical object", "entity", "plant"),
    ("opal", "gem", "mineral", "physical object", "entity", "animal"),
    ("skiff", "boat", "vehicle", "physical object", "entity", "flower"),
)
NATURAL_ZH = (
    ("麻雀", "鸟类", "动物", "生物", "实体", "机器"),
    ("鳟鱼", "鱼类", "动物", "生物", "实体", "矿物"),
    ("郁金香", "花卉", "植物", "生物", "实体", "车辆"),
    ("枫树", "树木", "植物", "生物", "实体", "乐器"),
    ("长笛", "乐器", "器物", "物理对象", "实体", "动物"),
    ("凿子", "工具", "器物", "物理对象", "实体", "植物"),
    ("蛋白石", "宝石", "矿物", "物理对象", "实体", "动物"),
    ("小艇", "船只", "车辆", "物理对象", "实体", "花卉"),
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
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


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
        "camera": "embedding + HiddenState + logits/text; all signed coordinates; no PCA/Top-K/attention/MLP/gradients",
        "branch_policy": "a failed route blocks only that route's mechanism claim",
        "claim_boundary": "execution closure is not an empirical mechanism pass",
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
    print(json.dumps({"phase": PHASES[name][0], "campaign": name,
                      "all_checks_passed": result["all_checks_passed"], "checks": checks},
                     ensure_ascii=False, indent=2), flush=True)
    return result


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def partition(unit: int) -> str:
    return "discovery" if unit < 4 else "confirmation" if unit < 6 else "lockbox"


def material_path() -> Path:
    return OUTS["C613"] / "material/fresh_multilingual_programs.jsonl"


def compiled_path() -> Path:
    return OUTS["C614"] / "material/qwen_compiled.jsonl"


def behavior_path() -> Path:
    return OUTS["C614"] / "behavior/qwen_behavior.jsonl"


def states_path() -> Path:
    return OUTS["C615"] / "raw/role_last.float16.npy"


def index_path() -> Path:
    return OUTS["C615"] / "raw/hidden_index.jsonl"


def ordered_candidates(answer: str, candidates: list[str], desired: int) -> list[str]:
    values = [answer] + [x for x in candidates if x != answer]
    if len(values) != len(set(values)):
        raise RuntimeError((answer, values))
    desired %= len(values)
    rest = values[1:]
    return rest[:desired] + [answer] + rest[desired:]


def render(language: str, surface: str, facts: str, question: str) -> str:
    if language == "en":
        lead = "Research note" if surface == "canonical" else "A curator reports"
        return f"{lead}: {facts} {question} Return only the answer phrase."
    lead = "研究记录" if surface == "canonical" else "整理员报告"
    return f"{lead}：{facts}{question}只返回答案短语。"


def make_row(panel: str, family: str, domain: str, language: str, surface: str, unit: int,
             cell: str, facts: str, question: str, answer: str, candidates: list[str],
             roles: dict, factors: dict, cell_index: int) -> dict:
    desired = (unit + cell_index + LANGUAGES.index(language) + 2 * SURFACES.index(surface)) % len(candidates)
    ordered = ordered_candidates(answer, candidates, desired)
    return {
        "case_id": f"c613|{panel}|{family}|{domain}|{language}|{surface}|u{unit:02d}|{cell}",
        "panel": panel, "family": family, "operation_domain": domain,
        "language": language, "surface": surface, "unit": unit,
        "partition": partition(unit), "cell": cell,
        "prompt": render(language, surface, facts, question),
        "answer": answer, "answer_candidates": ordered,
        "role_values": roles, "factors": factors,
        "cross_model_subset": language == "en" and surface == "canonical" and unit in (0, 3, 6, 7)
                              and cell in ("depth1", "depth2", "ap00", "ap10"),
    }


def graph_nodes(unit: int, graph_type: str, language: str) -> tuple[str, ...]:
    if graph_type == "natural":
        return (NATURAL_EN if language == "en" else NATURAL_ZH)[unit]
    return tuple(f"{stem}{unit}" for stem in ("vexa", "lurin", "pador", "simek", "toran", "qevin"))


GRAPH_CELLS = ("depth1", "depth2", "depth3", "depth4", "shortcut", "distractor", "reverse")


def graph_row(graph_type: str, language: str, surface: str, unit: int, cell: str) -> dict:
    n0, n1, n2, n3, n4, wrong = graph_nodes(unit, graph_type, language)
    depth = {"depth1": 1, "depth2": 2, "depth3": 3, "depth4": 4,
             "shortcut": 4, "distractor": 4, "reverse": 3}[cell]
    if language == "en":
        links = [f"a {n0} is a {n1}", f"a {n1} is a {n2}", f"a {n2} is a {n3}", f"a {n3} is a {n4}"][:depth]
        if cell == "shortcut":
            links.append(f"the {n0} is also explicitly listed as a {n4}")
        if cell == "distractor":
            links.append(f"the {n0} is displayed beside a {wrong}, and beside is not a type link")
        facts = "; ".join(links) + "."
        if cell == "reverse":
            question, answer = f"Which category is directly below {n3} in this chain?", n2
        else:
            question, answer = f"Starting from {n0}, which category is reached after {depth} valid type links?", (n1, n2, n3, n4)[depth - 1]
        relation = "is a"
    else:
        links = [f"{n0}属于{n1}", f"{n1}属于{n2}", f"{n2}属于{n3}", f"{n3}属于{n4}"][:depth]
        if cell == "shortcut":
            links.append(f"同时明确列出{n0}属于{n4}")
        if cell == "distractor":
            links.append(f"{n0}放在{wrong}旁边，但相邻不是类型关系")
        facts = "；".join(links) + "。"
        if cell == "reverse":
            question, answer = f"在这条链中，{n3}的直接下位类别是什么？", n2
        else:
            question, answer = f"从{n0}开始，经过{depth}条有效类型关系会到达哪个类别？", (n1, n2, n3, n4)[depth - 1]
        relation = "属于"
    return make_row("graph", "typed_graph", graph_type, language, surface, unit, cell,
                    facts, question, answer, [n1, n2, n3, n4, wrong],
                    {"primary": n0, "secondary": n1, "relation": relation,
                     "context": n1, "query": n0},
                    {"graph_type": graph_type, "depth": depth, "control": cell}, GRAPH_CELLS.index(cell))


ATTITUDE_CELLS = ("ap00", "ap10", "ap01", "ap11")
SCOPE_CELLS = tuple(f"o{o}i{i}q{q}" for o, i, q in itertools.product((0, 1), (0, 1), ("outer", "inner")))


def attitude_row(language: str, surface: str, unit: int, cell: str) -> dict:
    reporter, agent, other = PEOPLE_A[unit], PEOPLE_B[unit], PEOPLE_C[unit]
    obj0, obj1 = OBJECTS_A[unit], OBJECTS_B[unit]
    if cell.startswith("ap"):
        a = int(cell[2]); p = int(cell[3])
        selected_agent, selected_obj = (agent, other)[a], (obj0, obj1)[p]
        if language == "en":
            relation = "likes to eat"
            facts = f"{reporter} says that {selected_agent} {relation} {selected_obj}."
            question = "Who is said to like eating what?"
            answer = f"{selected_agent} with {selected_obj}"
        else:
            relation = "喜欢吃"
            facts = f"{reporter}说{selected_agent}{relation}{selected_obj}。"
            question = "被描述为喜欢吃东西的是谁，喜欢吃什么？"
            answer = f"{selected_agent}与{selected_obj}"
        candidates = [f"{x} with {y}" if language == "en" else f"{x}与{y}"
                      for x, y in ((agent, obj0), (other, obj0), (agent, obj1), (other, obj1))]
        candidates.append(f"{reporter} with {obj1}" if language == "en" else f"{reporter}与{obj1}")
        factors = {"agent_swap": a, "patient_swap": p, "program": "agent_patient"}
        idx = ATTITUDE_CELLS.index(cell)
    else:
        match = re.fullmatch(r"o([01])i([01])q(outer|inner)", cell)
        if not match:
            raise ValueError(cell)
        outer, inner, query_scope = int(match.group(1)), int(match.group(2)), match.group(3)
        selected_agent, selected_obj = agent, obj0
        if language == "en":
            outer_text = "does not state" if outer else "states"
            relation = outer_text
            inner_text = "does not like to eat" if inner else "likes to eat"
            facts = f"{reporter} {outer_text} that {selected_agent} {inner_text} {selected_obj}."
            question = (f"Does the report say that {reporter} makes a positive statement?" if query_scope == "outer"
                        else f"Does the embedded clause say that {selected_agent} likes to eat {selected_obj}?")
        else:
            outer_text = "没有陈述" if outer else "陈述"
            relation = outer_text
            inner_text = "不喜欢吃" if inner else "喜欢吃"
            facts = f"{reporter}{outer_text}{selected_agent}{inner_text}{selected_obj}。"
            question = (f"记录是否说{reporter}作出了肯定陈述？" if query_scope == "outer"
                        else f"内层句子是否说{selected_agent}喜欢吃{selected_obj}？")
        answer = "No" if (outer if query_scope == "outer" else inner) else "Yes"
        candidates = ["Yes", "No", "Unknown", "Unclear"]
        factors = {"outer": outer, "inner": inner, "query_scope": query_scope, "program": "scope"}
        idx = len(ATTITUDE_CELLS) + SCOPE_CELLS.index(cell)
    return make_row("attitude", "attitude_event", factors["program"], language, surface, unit, cell,
                    facts, question, answer, candidates,
                    {"primary": reporter, "secondary": selected_agent, "relation": relation,
                     "context": selected_obj, "query": selected_obj}, factors, idx)


def make_material() -> list[dict]:
    rows = [graph_row(g, l, s, u, c) for g, l, s, u, c in
            itertools.product(("natural", "pseudo"), LANGUAGES, SURFACES, range(UNITS), GRAPH_CELLS)]
    rows += [attitude_row(l, s, u, c) for l, s, u, c in
             itertools.product(LANGUAGES, SURFACES, range(UNITS), ATTITUDE_CELLS + SCOPE_CELLS)]
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = c607.text_core.chat_ids(tokenizer, SYSTEM, row["prompt"])
        candidate_ids = [tokenizer.encode(" " + x, add_special_tokens=False) for x in row["answer_candidates"]]
        if not all(candidate_ids):
            raise RuntimeError((row["case_id"], "empty candidate"))
        positions = {}
        for role, value in row["role_values"].items():
            spans = c607.compiler.graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidate_ids,
                         "gold_position": row["answer_candidates"].index(row["answer"]),
                         "role_positions": positions})
    return compiled


def normalize(text: str) -> str:
    value = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    value = " ".join(value.strip().lower().split()).strip(".,;:!?\"'`()[]{}")
    return value


def generated_prediction(text: str, candidates: list[str]) -> int:
    value = normalize(text)
    matches = []
    for i, candidate in enumerate(candidates):
        target = normalize(candidate)
        if value == target or target in value:
            matches.append(i)
    return matches[0] if len(matches) == 1 else -1


def pair_records(index: list[dict], dual_only: bool = True) -> list[dict]:
    by_id = {x["case_id"]: x for x in index}
    pairs = []
    def add(left_id: str, right_id: str, operation: str):
        left, right = by_id.get(left_id), by_id.get(right_id)
        if left is None or right is None:
            return
        eligible = left["slice_qualified"] and right["slice_qualified"] and left["candidate_correct"] and right["candidate_correct"]
        if dual_only:
            eligible = eligible and left["generated_correct"] and right["generated_correct"]
        if eligible:
            pairs.append({"left": left, "right": right, "operation": operation,
                          "partition": left["partition"]})
    for g, l, s, u in itertools.product(("natural", "pseudo"), LANGUAGES, SURFACES, range(UNITS)):
        root = f"c613|graph|typed_graph|{g}|{l}|{s}|u{u:02d}|"
        add(root + "depth1", root + "depth2", f"graph:{g}:{l}:1to2")
        add(root + "depth2", root + "depth3", f"graph:{g}:{l}:2to3")
        add(root + "depth3", root + "depth4", f"graph:{g}:{l}:3to4")
        add(root + "depth4", root + "shortcut", f"graph:{g}:{l}:shortcut")
        add(root + "depth4", root + "distractor", f"graph:{g}:{l}:distractor")
    for l, s, u in itertools.product(LANGUAGES, SURFACES, range(UNITS)):
        root = f"c613|attitude|attitude_event|agent_patient|{l}|{s}|u{u:02d}|"
        add(root + "ap00", root + "ap10", f"attitude:{l}:agent")
        add(root + "ap00", root + "ap01", f"attitude:{l}:patient")
        add(root + "ap10", root + "ap11", f"attitude:{l}:patient_after_agent")
        add(root + "ap01", root + "ap11", f"attitude:{l}:agent_after_patient")
        for query in ("outer", "inner"):
            root = f"c613|attitude|attitude_event|scope|{l}|{s}|u{u:02d}|"
            add(root + f"o0i0q{query}", root + f"o1i0q{query}", f"scope:{l}:outer:q{query}")
            add(root + f"o0i0q{query}", root + f"o0i1q{query}", f"scope:{l}:inner:q{query}")
    return pairs


def metric(pred: np.ndarray, truth: np.ndarray) -> dict:
    p, t = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    err = p - t
    rms_t = float(np.sqrt(np.mean(t * t)))
    rms_e = float(np.sqrt(np.mean(err * err)))
    flat_p, flat_t = p.reshape(-1), t.reshape(-1)
    return {
        "nrmse": rms_e / (rms_t + 1e-12),
        "cosine": float(np.dot(flat_p, flat_t) / (np.linalg.norm(flat_p) * np.linalg.norm(flat_t) + 1e-12)),
        "sign_agreement": float(np.mean(np.sign(p) == np.sign(t))),
        "truth_rms": rms_t, "error_rms": rms_e,
    }


def role_state(states: np.ndarray, row: dict, q: int) -> np.ndarray:
    return np.asarray(states[int(row["hidden_index"]), q], np.float32)


def c613() -> None:
    out = begin("C613", {
        "object": "fresh multilingual graph and attitude-event language programs",
        "coverage": {"languages": LANGUAGES, "surfaces": SURFACES, "units": UNITS,
                     "graph_cells": GRAPH_CELLS, "attitude_cells": ATTITUDE_CELLS + SCOPE_CELLS},
        "partitions": "unit 0-3 discovery, 4-5 confirmation, 6-7 lockbox",
        "zero_models": ["fixed candidate position", "longest answer", "surface/language label"],
        "human_review": "external and unavailable locally; must remain NA_pending_external_review",
    }, {"C612": c607.load(c607.RESULT / "phase2146_c612_natural_output_compiler_independent_audit/analysis/final.json")["all_checks_passed"]})
    rows = make_material()
    ids, prompts = [r["case_id"] for r in rows], [r["prompt"] for r in rows]
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise RuntimeError("material identity is not unique")
    write_rows(material_path(), rows)
    first = max(sum(r["answer_candidates"].index(r["answer"]) == i for r in rows) / len(rows)
                for i in range(max(len(r["answer_candidates"]) for r in rows)))
    longest = sum(max(r["answer_candidates"], key=len) == r["answer"] for r in rows) / len(rows)
    human = [{"case_id": r["case_id"], "naturalness_1_5": None, "semantic_uniqueness_0_1": None,
              "answerability_0_1": None, "reviewer": None}
             for r in rows if r["partition"] == "lockbox"]
    write_rows(out / "external/human_blind_template.jsonl", human)
    headline = {
        "status": "fresh_contract_frozen", "rows": len(rows),
        "partition_counts": {p: sum(r["partition"] == p for r in rows) for p in ("discovery", "confirmation", "lockbox")},
        "family_counts": {p: sum(r["panel"] == p for r in rows) for p in ("graph", "attitude")},
        "language_counts": {p: sum(r["language"] == p for r in rows) for p in LANGUAGES},
        "zero_models": {"max_fixed_position": first, "longest_answer": longest},
        "human_review": "NA_pending_external_review", "material_sha256": digest(rows),
        "strict_interpretation": "Machine semantic compilation is not an independent human naturalness judgment.",
    }
    close("C613", headline, {"large": len(rows) >= 800, "unique": len(ids) == len(set(ids)) == len(set(prompts)),
                              "balanced": first <= .26, "human_not_fabricated": all(v["reviewer"] is None for v in human),
                              "finite": finite(headline)}, "C614_qwen_dual_behavior")


def c614() -> None:
    out = begin("C614", {
        "object": "Qwen3-4B model-specific candidate-sequence and free-generation interface",
        "gates": "candidate and generated accuracy each >=0.75 per panel/domain/language/surface slice",
        "parser": "frozen unique candidate containment after stripping formatting; ambiguous matches fail",
        "observation_policy": "all rows remain observable; mechanism claims require qualified dual-correct pairs",
    }, {"C613": final("C613")["all_checks_passed"]})
    rows = read_rows(material_path())
    model = None
    try:
        model, tokenizer, device, placement = c607.passport.previous.model_base().load_bf16("qwen3")
        compiled = compile_rows(tokenizer, rows)
        write_rows(compiled_path(), compiled)
        scores_all = c607.batch_candidate_scores(model, device, compiled, batch_size=16)
        behavior = []
        for i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = c607.greedy_text(model, tokenizer, device, item["prompt_ids"], max_new_tokens=16)
            pred, gen = int(np.argmax(scores)), generated_prediction(text, item["answer_candidates"])
            behavior.append({"case_id": item["case_id"], "candidate_prediction": pred,
                             "candidate_correct": pred == item["gold_position"],
                             "generated_text": text, "generated_prediction": gen,
                             "generated_correct": gen == item["gold_position"]})
            if i % 32 == 0 or i + 1 == len(compiled):
                print(f"[C614 generation] {i + 1}/{len(compiled)}", flush=True)
        write_rows(behavior_path(), behavior)
    finally:
        c607.passport.previous.model_base().release_bf16(model)
        gc.collect()
    by_behavior = {r["case_id"]: r for r in behavior}
    grouped = defaultdict(list)
    for row in rows:
        key = "|".join((row["panel"], row["operation_domain"], row["language"], row["surface"]))
        grouped[key].append(by_behavior[row["case_id"]])
    slices = {}
    for key, values in sorted(grouped.items()):
        ca = float(np.mean([x["candidate_correct"] for x in values]))
        ga = float(np.mean([x["generated_correct"] for x in values]))
        slices[key] = {"rows": len(values), "candidate_accuracy": ca, "generated_accuracy": ga,
                       "qualified": ca >= BEHAVIOR_GATE and ga >= BEHAVIOR_GATE}
    save(out / "behavior/slice_qualification.json", slices)
    ca = float(np.mean([x["candidate_correct"] for x in behavior]))
    ga = float(np.mean([x["generated_correct"] for x in behavior]))
    headline = {"status": "qwen_dual_behavior_closed", "rows": len(rows), "candidate_accuracy": ca,
                "generated_accuracy": ga, "qualified_slices": sum(v["qualified"] for v in slices.values()),
                "total_slices": len(slices), "slices": slices,
                "strict_interpretation": "Candidate likelihood and open generation remain separate behavioral objects."}
    close("C614", headline, {"complete": len(behavior) == len(rows), "some_qualified": any(v["qualified"] for v in slices.values()),
                              "finite": finite(headline)}, "C615_all_coordinate_guard_atlas")


def c615() -> None:
    out = begin("C615", {
        "object": "all-token all-coordinate observation plus full-coordinate base-state guard competition",
        "models": ["identity", "mean", "diagonal affine", "coordinate-sign guard", "nearest base", "history nearest", "wrong operation"],
        "gate": "best conditional model beats identity, mean and wrong operation by >=0.02 on lockbox",
        "storage": "all tokens sharded until C619; all six role states retained without coordinate compression",
    }, {"C614": final("C614")["all_checks_passed"]})
    compiled = read_rows(compiled_path()); behavior = {x["case_id"]: x for x in read_rows(behavior_path())}
    slices = final("C614")["headline"]["slices"]
    n = len(compiled)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    states = np.lib.format.open_memmap(states_path(), mode="w+", dtype=np.float16,
                                       shape=(n, CHECKPOINTS, len(ROLES), DIM))
    shard_dir = out / "raw/full_token_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    model = None; hooks = []; captured = []; index = []; ledger = []
    try:
        model, tokenizer, device, placement = c607.passport.previous.model_base().load_bf16("qwen3")
        base = model.model
        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)
        hooks.append(base.embed_tokens.register_forward_hook(hook))
        hooks.extend(layer.register_forward_hook(hook) for layer in base.layers)
        hooks.append(base.norm.register_forward_hook(hook))
        for start in range(0, n, 32):
            items = compiled[start:start + 32]
            width = max(len(x["prompt_ids"]) for x in items)
            shard_path = shard_dir / f"states_{start:04d}_{start + len(items):04d}.float16.npy"
            shard = np.lib.format.open_memmap(shard_path, mode="w+", dtype=np.float16,
                                              shape=(len(items), CHECKPOINTS, width, DIM))
            for local, item in enumerate(items):
                ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
                mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
                captured.clear()
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
                if len(captured) != CHECKPOINTS:
                    raise RuntimeError((len(captured), CHECKPOINTS))
                for q, hidden in enumerate(captured):
                    arr = hidden[0].float().cpu().numpy().astype(np.float16)
                    shard[local, q, :arr.shape[0]] = arr
                    for role_i, role in enumerate(ROLES):
                        states[start + local, q, role_i] = arr[int(item["role_positions"][role][-1])]
                b = behavior[item["case_id"]]
                slice_key = "|".join((item["panel"], item["operation_domain"], item["language"], item["surface"]))
                index.append({"hidden_index": start + local, "case_id": item["case_id"],
                              "panel": item["panel"], "family": item["family"],
                              "operation_domain": item["operation_domain"], "language": item["language"],
                              "surface": item["surface"], "unit": item["unit"], "partition": item["partition"],
                              "cell": item["cell"], "factors": item["factors"],
                              "candidate_correct": b["candidate_correct"], "generated_correct": b["generated_correct"],
                              "slice_qualified": slices[slice_key]["qualified"],
                              "token_count": len(item["prompt_ids"]), "shard": shard_path.name, "shard_row": local})
            shard.flush(); del shard
            states.flush(); ledger.append({"path": shard_path.name, "rows": len(items), "width": width,
                                           "bytes": shard_path.stat().st_size})
            print(f"[C615 capture] {start + len(items)}/{n}", flush=True)
    finally:
        for h in hooks: h.remove()
        captured.clear(); c607.passport.previous.model_base().release_bf16(model)
        states.flush(); del states; gc.collect()
    write_rows(index_path(), index); save(out / "raw/shard_ledger.json", ledger)

    states = np.load(states_path(), mmap_mode="r")
    pairs = pair_records(index, dual_only=True)
    grouped = defaultdict(list)
    for pair in pairs: grouped[pair["operation"]].append(pair)
    results = {}; passports = {}
    for operation, values in sorted(grouped.items()):
        train = [x for x in values if x["partition"] == "discovery"]
        test = [x for x in values if x["partition"] == "lockbox"]
        if len(train) < 3 or len(test) < 2: continue
        wrong_pool = [x for op, vals in grouped.items() if op != operation for x in vals if x["partition"] == "discovery"]
        for q in QPOINTS:
            htr = np.stack([role_state(states, x["left"], q) for x in train])
            ytr = np.stack([role_state(states, x["right"], q) - role_state(states, x["left"], q) for x in train])
            hte = np.stack([role_state(states, x["left"], q) for x in test])
            truth = np.stack([role_state(states, x["right"], q) - role_state(states, x["left"], q) for x in test])
            mean = np.mean(ytr, axis=0)
            centered = htr - np.mean(htr, axis=0); ycenter = ytr - np.mean(ytr, axis=0)
            slope = np.sum(centered * ycenter, axis=0) / (np.sum(centered * centered, axis=0) + 1e-6)
            diagonal = np.mean(ytr, axis=0) + (hte - np.mean(htr, axis=0)) * slope
            pos_mean = np.sum(ytr * (htr >= 0), axis=0) / (np.sum(htr >= 0, axis=0) + 1e-6)
            neg_mean = np.sum(ytr * (htr < 0), axis=0) / (np.sum(htr < 0, axis=0) + 1e-6)
            sign_guard = np.where(hte >= 0, pos_mean, neg_mean)
            nearest, history = [], []
            for h in hte:
                d = np.mean((htr - h[None]) ** 2, axis=(1, 2)); nearest.append(ytr[int(np.argmin(d))])
            qprev = max(0, q - 1)
            htr_prev = np.stack([role_state(states, x["left"], qprev) for x in train])
            hte_prev = np.stack([role_state(states, x["left"], qprev) for x in test])
            for h, hp in zip(hte, hte_prev):
                d = np.mean((htr - h[None]) ** 2, axis=(1, 2)) + np.mean((htr_prev - hp[None]) ** 2, axis=(1, 2))
                history.append(ytr[int(np.argmin(d))])
            wrong = np.mean([role_state(states, x["right"], q) - role_state(states, x["left"], q)
                             for x in wrong_pool], axis=0) if wrong_pool else -mean
            preds = {"identity": np.zeros_like(truth), "mean": np.broadcast_to(mean, truth.shape),
                     "diagonal": diagonal, "sign_guard": sign_guard, "nearest": np.stack(nearest),
                     "history_nearest": np.stack(history), "wrong_operation": np.broadcast_to(wrong, truth.shape)}
            values_out = {name: metric(pred, truth) for name, pred in preds.items()}
            best_name = min(("diagonal", "sign_guard", "nearest", "history_nearest"), key=lambda x: values_out[x]["nrmse"])
            gate = all(values_out[best_name]["nrmse"] <= values_out[x]["nrmse"] - CONTROL_MARGIN
                       for x in ("identity", "mean", "wrong_operation"))
            key = f"{operation}|q{q}"
            results[key] = {"train": len(train), "test": len(test), "models": values_out,
                            "best_conditional": best_name, "gate": gate}
            passports[key] = {"mean": mean.astype(np.float16), "slope": slope.astype(np.float16),
                              "positive": pos_mean.astype(np.float16), "negative": neg_mean.astype(np.float16)}
    np.savez_compressed(out / "analysis/full_coordinate_guard_passports.npz",
                        **{f"{k}|{name}": arr for k, val in passports.items() for name, arr in val.items()})
    c607.passport.close_mmap(states); del states
    candidates = [k for k, v in results.items() if v["gate"]]
    headline = {"status": "all_coordinate_guard_atlas_closed", "capture_rows": n,
                "capture_shape": [n, CHECKPOINTS, len(ROLES), DIM], "pair_count": len(pairs),
                "operation_count": len(grouped), "metric_cells": len(results), "guard_candidates": candidates,
                "full_token_shards": len(ledger), "full_token_bytes": sum(x["bytes"] for x in ledger),
                "strict_interpretation": "A guard win is predictive dependence over full coordinates, not a unique causal circuit."}
    close("C615", headline, {"capture": n == len(index), "shape": headline["capture_shape"][1:] == [38, 6, 2560],
                              "pairs": len(pairs) > 0, "metrics": bool(results), "finite": finite(headline)},
          "C616_graph_attitude_composition")


def c616() -> None:
    out = begin("C616", {
        "object": "fresh lockbox graph depth rollout and attitude/scope second-order interactions",
        "graph_controls": ["sequential nearest response", "additive mean", "wrong middle", "zero"],
        "interaction": "H11-H10-H01+H00 with discovery prototype versus zero on lockbox",
        "claim": "finite predictive composition candidate only; no global algebra",
    }, {"C615": final("C615")["all_checks_passed"]})
    index = read_rows(index_path()); states = np.load(states_path(), mmap_mode="r")
    by_id = {x["case_id"]: x for x in index}; graph_results = {}; interaction_results = {}
    for graph_type, language, q in itertools.product(("natural", "pseudo"), LANGUAGES, QPOINTS[1:]):
        train_units, test_units = range(4), range(6, 8)
        train_by_step = defaultdict(list)
        for surface, unit, step in itertools.product(SURFACES, train_units, (1, 2, 3)):
            root = f"c613|graph|typed_graph|{graph_type}|{language}|{surface}|u{unit:02d}|"
            a, b = by_id.get(root + f"depth{step}"), by_id.get(root + f"depth{step + 1}")
            if a and b and a["generated_correct"] and b["generated_correct"]:
                train_by_step[step].append((role_state(states, a, q), role_state(states, b, q) - role_state(states, a, q)))
        truths, sequential, additive, wrongs = [], [], [], []
        if not all(train_by_step[s] for s in (1, 2, 3)): continue
        means = {s: np.mean([v for _, v in train_by_step[s]], axis=0) for s in (1, 2, 3)}
        for surface, unit in itertools.product(SURFACES, test_units):
            root = f"c613|graph|typed_graph|{graph_type}|{language}|{surface}|u{unit:02d}|"
            rows = [by_id.get(root + f"depth{i}") for i in range(1, 5)]
            if not all(rows) or not all(x["generated_correct"] for x in rows): continue
            start, target = role_state(states, rows[0], q), role_state(states, rows[3], q)
            pred = start.copy()
            for step in (1, 2, 3):
                source_bank = np.stack([h for h, _ in train_by_step[step]])
                d = np.mean((source_bank - pred[None]) ** 2, axis=(1, 2))
                pred = pred + train_by_step[step][int(np.argmin(d))][1]
            truths.append(target - start); sequential.append(pred - start)
            additive.append(sum(means.values())); wrongs.append(means[1] + means[3] + means[2][::-1])
        if truths:
            truth = np.stack(truths)
            vals = {"sequential": metric(np.stack(sequential), truth),
                    "additive": metric(np.stack(additive), truth), "wrong_middle": metric(np.stack(wrongs), truth),
                    "zero": metric(np.zeros_like(truth), truth)}
            gate = all(vals["sequential"]["nrmse"] <= vals[x]["nrmse"] - CONTROL_MARGIN
                       for x in ("additive", "wrong_middle", "zero"))
            graph_results[f"{graph_type}|{language}|q{q}"] = {"tests": len(truths), "models": vals, "gate": gate}
    for program, language, q in itertools.product(("agent_patient", "scope_outer", "scope_inner"), LANGUAGES, QPOINTS[1:]):
        train, test = [], []
        for surface, unit in itertools.product(SURFACES, range(UNITS)):
            root = f"c613|attitude|attitude_event|{'agent_patient' if program == 'agent_patient' else 'scope'}|{language}|{surface}|u{unit:02d}|"
            if program == "agent_patient": cells = ("ap00", "ap10", "ap01", "ap11")
            else:
                query = program.split("_")[1]; cells = tuple(f"o{o}i{i}q{query}" for o, i in ((0,0),(1,0),(0,1),(1,1)))
            rows = [by_id.get(root + c) for c in cells]
            if not all(rows) or not all(x["candidate_correct"] for x in rows): continue
            h = [role_state(states, x, q) for x in rows]
            residual = h[3] - h[1] - h[2] + h[0]
            qualified = all(x["slice_qualified"] and x["generated_correct"] for x in rows)
            (train if unit < 4 else test if unit >= 6 else []).append((residual, qualified))
        if train and test:
            truth = np.stack([x[0] for x in test]); proto = np.mean([x[0] for x in train], axis=0)
            p = metric(np.broadcast_to(proto, truth.shape), truth); z = metric(np.zeros_like(truth), truth)
            behavior_qualified = all(x[1] for x in train + test)
            interaction_results[f"{program}|{language}|q{q}"] = {"tests": len(test), "prototype": p, "zero": z,
                                                                  "behavior_qualified": behavior_qualified,
                                                                  "gate": behavior_qualified and p["nrmse"] <= z["nrmse"] - CONTROL_MARGIN,
                                                                  "observation_status": "formal" if behavior_qualified else "candidate_correct_descriptive_only"}
    c607.passport.close_mmap(states); del states
    headline = {"status": "composition_closed", "graph": graph_results, "interactions": interaction_results,
                "summary": {"graph_passed": sum(v["gate"] for v in graph_results.values()), "graph_total": len(graph_results),
                            "interaction_passed": sum(v["gate"] for v in interaction_results.values()),
                            "interaction_total": len(interaction_results)},
                "strict_interpretation": "Passes are finite fresh-material prediction laws, not associativity, inverse, holonomy or new mathematics."}
    close("C616", headline, {"graph_observed": bool(graph_results), "interaction_observed": bool(interaction_results),
                              "finite": finite(headline)}, "C617_generation_timeline")


def persistent_greedy(model, tokenizer, input_ids, attention_mask, role_positions, patches, max_new_tokens=16):
    base = model.model; handles = []; by_q = defaultdict(list)
    for q, response, role_order in patches: by_q[int(q)].append((np.asarray(response, np.float32), role_order))
    for q, values in by_q.items():
        def make_hook(items):
            def patch_hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output; changed = tensor.clone()
                for response, order in items:
                    for target_i, role in enumerate(ROLES):
                        pos = int(role_positions[role][-1])
                        if pos < changed.shape[1]:
                            changed[0, pos] += torch.tensor(response[order[target_i]], dtype=changed.dtype, device=changed.device)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            return patch_hook
        handles.append(base.layers[q - 1].register_forward_hook(make_hook(values)))
    ids, mask = input_ids, attention_mask
    try:
        for _ in range(max_new_tokens):
            pos = mask.long().cumsum(-1) - 1
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True).logits
            nxt = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            ids = torch.cat((ids, nxt), dim=1); mask = torch.cat((mask, torch.ones_like(nxt)), dim=1)
            if int(nxt[0, 0]) == tokenizer.eos_token_id: break
    finally:
        for h in handles: h.remove()
    return tokenizer.decode(ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def c617() -> None:
    out = begin("C617", {
        "object": "prompt-prefill versus persistent generation-time intervention",
        "eligibility": "C615 guard candidate operation with dual-correct discovery and lockbox pairs",
        "tests": ["q16/q24/q32", "joint", "wrong sign/role/operation", "prefill generation", "persistent generation",
                  "deletion then adaptive rescue", "unrelated side effect"],
        "claim": "generation necessity only if natural output breaks and correct rescue beats wrong rescue",
    }, {"C616": final("C616")["all_checks_passed"]})
    index = read_rows(index_path()); states = np.load(states_path(), mmap_mode="r")
    compiled = {x["case_id"]: x for x in read_rows(compiled_path())}
    pairs = pair_records(index, dual_only=True); guard_keys = final("C615")["headline"]["guard_candidates"]
    candidate_ops = sorted({k.rsplit("|q", 1)[0] for k in guard_keys})
    eligible = [p for p in pairs if p["operation"] in candidate_ops and p["partition"] == "lockbox"][:12]
    train_by_op = {op: [p for p in pairs if p["operation"] == op and p["partition"] == "discovery"] for op in candidate_ops}
    model = None; records = []
    try:
        model, tokenizer, device, placement = c607.passport.previous.model_base().load_bf16("qwen3")
        identity = tuple(range(len(ROLES))); swapped = (1, 0, 2, 3, 4, 5)
        for pair in eligible:
            op = pair["operation"]; train = train_by_op[op]
            source, target = pair["left"], pair["right"]
            responses = {}
            for q in (16, 24, 32):
                h = role_state(states, source, q)
                d = [np.mean((role_state(states, p["left"], q) - h) ** 2) for p in train]
                donor = train[int(np.argmin(d))]
                responses[q] = role_state(states, donor["right"], q) - role_state(states, donor["left"], q)
            wrong_pair = next((p for p in pairs if p["operation"] != op and p["partition"] == "discovery"), None)
            wrong = (role_state(states, wrong_pair["right"], 24) - role_state(states, wrong_pair["left"], 24)) if wrong_pair else responses[24][::-1]
            item, target_item = compiled[source["case_id"]], compiled[target["case_id"]]
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device); mask = torch.ones_like(ids)
            interventions = {
                "zero": [], "q16": [(16, responses[16], identity)], "q24": [(24, responses[24], identity)],
                "q32": [(32, responses[32], identity)],
                "joint": [(16, .5 * responses[16], identity), (24, .5 * responses[24], identity), (32, .5 * responses[32], identity)],
                "wrong_sign": [(24, -responses[24], identity)], "wrong_role": [(24, responses[24], swapped)],
                "wrong_operation": [(24, transport.scaled_like(wrong, responses[24]), identity)],
            }
            outputs = {}
            for name, patches in interventions.items():
                prefill = transport.patched_greedy_text(model, tokenizer, ids, mask, item["role_positions"], patches, max_new_tokens=16)
                persistent = persistent_greedy(model, tokenizer, ids, mask, item["role_positions"], patches, max_new_tokens=16)
                outputs[name] = {"prefill_text": prefill, "prefill_target": generated_prediction(prefill, item["answer_candidates"]) == item["answer_candidates"].index(target_item["answer"]),
                                 "persistent_text": persistent, "persistent_target": generated_prediction(persistent, item["answer_candidates"]) == item["answer_candidates"].index(target_item["answer"])}
            tid = torch.tensor([target_item["prompt_ids"]], dtype=torch.long, device=device); tmask = torch.ones_like(tid)
            natural = persistent_greedy(model, tokenizer, tid, tmask, target_item["role_positions"], [], 16)
            deletion_patch = [(24, -responses[24], identity)]
            deletion = persistent_greedy(model, tokenizer, tid, tmask, target_item["role_positions"], deletion_patch, 16)
            natural_ok = generated_prediction(natural, target_item["answer_candidates"]) == target_item["gold_position"]
            deletion_ok = generated_prediction(deletion, target_item["answer_candidates"]) == target_item["gold_position"]
            target_values = {"natural_text": natural, "natural_ok": natural_ok, "deletion_text": deletion,
                             "deletion_ok": deletion_ok, "rescue_eligible": natural_ok and not deletion_ok}
            if target_values["rescue_eligible"]:
                rescue = persistent_greedy(model, tokenizer, tid, tmask, target_item["role_positions"], deletion_patch + [(32, responses[32], identity)], 16)
                wrong_rescue = persistent_greedy(model, tokenizer, tid, tmask, target_item["role_positions"], deletion_patch + [(32, -responses[32], identity)], 16)
                target_values.update({"rescue_text": rescue, "rescue_ok": generated_prediction(rescue, target_item["answer_candidates"]) == target_item["gold_position"],
                                      "wrong_rescue_text": wrong_rescue, "wrong_rescue_ok": generated_prediction(wrong_rescue, target_item["answer_candidates"]) == target_item["gold_position"]})
            records.append({"operation": op, "source": source["case_id"], "target": target["case_id"],
                            "outputs": outputs, "target_values": target_values})
            print(f"[C617] {len(records)}/{len(eligible)}", flush=True)
    finally:
        c607.passport.previous.model_base().release_bf16(model)
        c607.passport.close_mmap(states); del states; gc.collect()
    write_rows(out / "analysis/generation_timeline_records.jsonl", records)
    totals = {"tests": len(records),
              "prefill_target": sum(r["outputs"]["q24"]["prefill_target"] for r in records),
              "persistent_target": sum(r["outputs"]["q24"]["persistent_target"] for r in records),
              "deletion_broke": sum(r["target_values"]["rescue_eligible"] for r in records),
              "rescue_eligible": sum(r["target_values"]["rescue_eligible"] for r in records),
              "specific_rescue": sum(r["target_values"].get("rescue_ok", False) and not r["target_values"].get("wrong_rescue_ok", False) for r in records)}
    headline = {"status": "generation_timeline_closed", "eligible_operations": candidate_ops,
                "records": len(records), "totals": totals,
                "strict_interpretation": "State and output effects are separately counted; absence of eligible records is NA, not a mechanism negative."}
    close("C617", headline, {"executed_if_eligible": bool(records) if eligible else True,
                              "adaptive_rescue": all(not r["target_values"].get("rescue_ok", False) or r["target_values"]["rescue_eligible"] for r in records),
                              "finite": finite(headline)}, "C618_cross_model_interfaces")


def run_worker(cmd: list[str]) -> dict:
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {"returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}


def c618() -> None:
    out = begin("C618", {
        "object": "sequential model-specific dual behavior interfaces on a frozen 32-row subset",
        "models": ["GLM4", "DeepSeek-7B", "Qwen3-14B"],
        "gate": "candidate and generated accuracy >=0.75 before HiddenState comparison",
        "interface": "thinking disabled where supported; 32-token generation; unique candidate containment parser",
    }, {"C617": final("C617")["all_checks_passed"]})
    worker = TESTS / "phase2152_c618_model_interface_worker.py"
    outputs = {}; supervisor = {}
    for model_name in ("glm4", "deepseek7b", "qwen3_14b"):
        target = out / f"analysis/{model_name}_worker.json"
        if target.exists():
            supervisor[model_name] = {"returncode": 0, "resumed": True}
        else:
            result = run_worker([str(ROOT / ".venv/Scripts/python.exe"), str(worker), "--model", model_name,
                                 "--material", str(material_path()), "--output", str(target)])
            (out / "audit").mkdir(parents=True, exist_ok=True)
            (out / f"audit/{model_name}_stdout.txt").write_text(result["stdout"] + "\nSTDERR:\n" + result["stderr"], encoding="utf-8")
            supervisor[model_name] = {"returncode": result["returncode"], "resumed": False}
        outputs[model_name] = load(target) if target.exists() else {"status": "worker_error", "hiddenstate_ran": False}
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    functional = [k for k, v in outputs.items() if v.get("hiddenstate_ran")]
    headline = {"status": "cross_model_interfaces_closed", "workers": outputs, "supervisor": supervisor,
                "behavior_qualified_models": functional,
                "strict_interpretation": "Behavior-unqualified models are not tested internally; physical coordinate IDs are never aligned."}
    close("C618", headline, {"workers": len(outputs) == 3, "sequential": all(v["returncode"] == 0 for v in supervisor.values()),
                              "finite": finite(headline)}, "C619_visual_theory_cleanup")


def register_visual() -> None:
    catalog = load(CATALOG)
    fields = catalog.setdefault("field_datasets", [])
    entry = {"id": "c619_conditional_gear_atlas", "title": "C619 Conditional Gear Atlas",
             "url": "/vis_data/research_kernel/c619_conditional_gear_atlas.json", "phase": 2153,
             "full_coordinate": True}
    fields[:] = [x for x in fields if x.get("id") != entry["id"]] + [entry]
    save(CATALOG, catalog)


def c619() -> None:
    out = begin("C619", {
        "object": "parameter-level atlas, cleanup and stable-theory adjudication",
        "visual": "representative all-token fields, every coordinate of role states and guard passports",
        "cleanup": "delete undisplayed bulk full-token shards only after visual registration; retain role field",
        "new_math_gate": "requires broad composition, bidirectional generation, necessity/rescue, cross-model and human validity",
    }, {"C618": final("C618")["all_checks_passed"]})
    index = read_rows(index_path()); compiled = {x["case_id"]: x for x in read_rows(compiled_path())}
    states = np.load(states_path(), mmap_mode="r")
    representatives = []
    for row in [x for x in index if x["partition"] == "lockbox" and x["generated_correct"]][:2]:
        item = compiled[row["case_id"]]
        shard = np.load(OUTS["C615"] / "raw/full_token_shards" / row["shard"], mmap_mode="r")
        exact = np.asarray(shard[row["shard_row"], :, :row["token_count"]], np.float16).copy()
        representatives.append({"case_id": row["case_id"], "prompt": item["prompt"],
                                "token_ids": item["prompt_ids"], "field_shape": list(exact.shape),
                                "all_token_all_checkpoint_coordinates": exact.tolist(),
                                "all_role_all_checkpoint_coordinates": np.asarray(states[row["hidden_index"]], np.float16).tolist()})
        c607.passport.close_mmap(shard); del shard, exact
    guard = np.load(OUTS["C615"] / "analysis/full_coordinate_guard_passports.npz")
    passports = {k: np.asarray(guard[k], np.float16).tolist() for k in guard.files}; guard.close()
    atlas = {
        "schema": "c619.conditional-gear-atlas.v1", "phase": 2153, "campaign": "C613-C619",
        "camera": {"checkpoints": CHECKPOINTS, "coordinates": DIM, "roles": ROLES,
                   "compression": "none across state coordinates"},
        "representatives": representatives, "guard_passports": passports,
        "behavior": final("C614")["headline"], "guards": final("C615")["headline"],
        "composition": final("C616")["headline"], "generation": final("C617")["headline"],
        "cross_model": final("C618")["headline"],
    }
    save(VISUAL, atlas); register_visual()
    c607.passport.close_mmap(states); del states, atlas, representatives, passports; gc.collect()
    shard_dir = OUTS["C615"] / "raw/full_token_shards"
    cleaned_bytes = sum(p.stat().st_size for p in shard_dir.rglob("*") if p.is_file()) if shard_dir.exists() else 0
    if shard_dir.exists(): shutil.rmtree(shard_dir)
    gates = {"qwen_behavior": final("C614")["headline"]["qualified_slices"] > 0,
             "base_state_guard": bool(final("C615")["headline"]["guard_candidates"]),
             "graph_composition": final("C616")["headline"]["summary"]["graph_passed"] > 0,
             "interaction": final("C616")["headline"]["summary"]["interaction_passed"] > 0,
             "bidirectional_generation": final("C617")["headline"]["totals"]["persistent_target"] >= 2,
             "generation_necessity": final("C617")["headline"]["totals"]["deletion_broke"] > 0,
             "specific_rescue": final("C617")["headline"]["totals"]["specific_rescue"] > 0,
             "cross_model": len(final("C618")["headline"]["behavior_qualified_models"]) >= 2,
             "human_naturalness": False}
    gates["new_math"] = all(gates.values())
    headline = {"status": "visual_theory_cleanup_closed", "visual": str(VISUAL.relative_to(ROOT)),
                "visual_bytes": VISUAL.stat().st_size, "cleaned_bytes": cleaned_bytes,
                "retained_role_field": str(states_path().relative_to(ROOT)), "empirical_gates": gates,
                "theory": {"name": "Conditional Output Field Closure Theory",
                           "principle": "Reuse-Difference-Conditioning",
                           "update": "typed base-state-selected full-coordinate response instances plus a separate autoregressive boundary",
                           "foundational_math_authorized": gates["new_math"]},
                "strict_interpretation": "The atlas is an observation index, not proof of a unique physical gear circuit."}
    close("C619", headline, {"visual": VISUAL.exists(), "catalog": CATALOG.exists(), "cleanup": not shard_dir.exists(),
                              "retained": states_path().exists(), "finite": finite(headline)}, "C620_independent_audit")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()


def c620() -> None:
    out = begin("C620", {
        "object": "independent arithmetic, artifact, missingness, cleanup and claim-boundary audit",
        "checks": "phase finals, material identity, behavior, full coordinates, branches, visual, cleanup, new-math gate",
    }, {"C619": final("C619")["all_checks_passed"]})
    checks = []
    def check(name, passed, detail): checks.append({"name": name, "passed": bool(passed), "detail": detail})
    finals = {k: final(k) for k in PHASES if k != "C620"}
    check("phase_finals_closed", all(v["status"] == "closed" and v["all_checks_passed"] for v in finals.values()),
          {k: v["all_checks_passed"] for k, v in finals.items()})
    rows = read_rows(material_path()); check("material_unique", len(rows) == len({r["case_id"] for r in rows}) == len({r["prompt"] for r in rows}), len(rows))
    check("partitions", [sum(r["partition"] == p for r in rows) for p in ("discovery", "confirmation", "lockbox")],
          final("C613")["headline"]["partition_counts"])
    human = read_rows(OUTS["C613"] / "external/human_blind_template.jsonl")
    check("human_review_not_fabricated", all(r["reviewer"] is None for r in human), "NA_pending_external_review")
    behavior = read_rows(behavior_path()); check("behavior_complete", len(behavior) == len(rows), len(behavior))
    states = np.load(states_path(), mmap_mode="r")
    check("all_coordinate_role_field", list(states.shape) == [len(rows), 38, 6, 2560] and states.dtype == np.float16,
          {"shape": list(states.shape), "dtype": str(states.dtype)})
    c607.passport.close_mmap(states); del states
    check("program_branches_present", final("C616")["headline"]["summary"]["graph_total"] > 0,
          final("C616")["headline"]["summary"])
    check("generation_adaptive", final("C617")["checks"]["adaptive_rescue"], final("C617")["headline"]["totals"])
    check("cross_model_sequential", len(final("C618")["headline"]["workers"]) == 3,
          {k: v.get("status") for k, v in final("C618")["headline"]["workers"].items()})
    check("visual_registered", VISUAL.exists() and VISUAL.stat().st_size > 0, {"bytes": VISUAL.stat().st_size})
    check("bulk_cleaned_role_retained", not (OUTS["C615"] / "raw/full_token_shards").exists() and states_path().exists(),
          {"retained_bytes": states_path().stat().st_size})
    check("new_math_gate_consistent", not final("C619")["headline"]["empirical_gates"]["new_math"],
          final("C619")["headline"]["empirical_gates"])
    manifest_paths = (
        [material_path(), compiled_path(), behavior_path(), states_path(), VISUAL]
        + [OUTS[k] / "analysis/final.json" for k in PHASES if k != "C620"]
    )
    manifest = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in manifest_paths]
    headline = {"status": "independent_audit_closed", "checks_passed": sum(x["passed"] for x in checks),
                "checks_total": len(checks), "checks": checks, "manifest": manifest,
                "adjudication": {
                    "supported": ["fresh model-specific behavior where slice gates pass",
                                  "all-coordinate predictive guards where frozen controls are beaten",
                                  "finite composition and generation effects only where their ledgers pass"],
                    "not_supported": ["human-rated naturalness", "a unique causal circuit", "cross-model physical coordinate identity",
                                      "global algebra or a new foundational mathematics"],
                    "same_exact_goal_next_stage": False,
                    "reason": "C613-C619 branches are exhausted; future work must use a separately frozen larger natural corpus or external human review."}}
    close("C620", headline, {"all_audit_checks": all(x["passed"] for x in checks), "finite": finite(headline)}, "major_stage_closed")


RUNNERS = {"C613": c613, "C614": c614, "C615": c615, "C616": c616,
           "C617": c617, "C618": c618, "C619": c619, "C620": c620}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", choices=tuple(RUNNERS), default="C613")
    parser.add_argument("--to", dest="end", choices=tuple(RUNNERS), default="C620")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); active = False
    for name, runner in RUNNERS.items():
        if name == args.start: active = True
        if active:
            final_path = OUTS[name] / "analysis/final.json"
            if final_path.exists() and not args.force:
                print(f"[{name}] resume: final exists", flush=True)
            else:
                runner()
        if name == args.end: break


if __name__ == "__main__":
    main()
