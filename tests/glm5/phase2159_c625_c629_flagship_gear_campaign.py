#!/usr/bin/env python3
"""C625-C629 flagship language-program and output-identity campaign.

The scientific camera is restricted to embeddings, HiddenState checkpoints,
logits and generated text. All retained state observations preserve every
signed physical coordinate. No PCA, Top-K, magnitude screening, attention,
MLP, weights or gradients are read. A failed branch closes only that branch.
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

import phase2147_c613_c620_conditional_gear_campaign as previous


PHASES = {
    "C625": (2159, "flagship_language_program_and_interface"),
    "C626": (2160, "full_coordinate_causal_transmission"),
    "C627": (2161, "unseen_flagship_composition"),
    "C628": (2162, "output_identity_generation_clock"),
    "C629": (2163, "crossmodel_visual_theory_audit"),
}
OUTS = {k: RESULT / f"phase{v[0]}_{k.lower()}_{v[1]}" for k, v in PHASES.items()}
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c629_flagship_gear_atlas.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"

SYSTEM = "Use only the supplied information. Reply with exactly one requested answer phrase and no explanation."
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
LANGUAGES = ("en", "zh")
SURFACES = ("canonical", "paraphrase")
WORLDS = ("memory", "consistent", "pseudo", "counterfactual")
GRAPH_CELLS = ("depth1", "depth2", "depth3", "depth4", "shortcut", "distractor", "reverse", "exception")
ATTITUDE_CELLS = ("ap00", "ap10", "ap01", "ap11", "active", "passive", "outer_neg", "inner_neg", "update_first", "update_last")
UNITS = 6
DIM = 2560
CHECKPOINTS = 38
QPOINTS = (8, 16, 24, 32, 37)
BEHAVIOR_GATE = 0.75
CONTROL_MARGIN = 0.02

PEOPLE = (
    ("Alden", "Brielle", "Cassian", "Daphne"),
    ("Eamon", "Freya", "Gideon", "Hana"),
    ("Iris", "Jonas", "Kira", "Lucan"),
    ("Mara", "Nolan", "Ophelia", "Pavel"),
    ("Quinn", "Rosa", "Soren", "Talia"),
    ("Uma", "Victor", "Willa", "Xavier"),
)
OBJECTS = (
    ("quince", "papaya", "turnip", "apricot"),
    ("radish", "plum", "fig", "melon"),
    ("pear", "guava", "beet", "peach"),
    ("carrot", "date", "lime", "mango"),
    ("lychee", "yam", "kiwi", "leek"),
    ("olive", "lemon", "onion", "grape"),
)
NATURAL_EN = (
    ("sparrow", "bird", "animal", "living thing", "entity", "mineral"),
    ("trout", "fish", "animal", "living thing", "entity", "machine"),
    ("tulip", "flower", "plant", "living thing", "entity", "vehicle"),
    ("maple", "tree", "plant", "living thing", "entity", "instrument"),
    ("flute", "instrument", "artifact", "physical object", "entity", "animal"),
    ("opal", "gem", "mineral", "physical object", "entity", "plant"),
)
NATURAL_ZH = (
    ("麻雀", "鸟类", "动物", "生物", "实体", "矿物"),
    ("鳟鱼", "鱼类", "动物", "生物", "实体", "机器"),
    ("郁金香", "花卉", "植物", "生物", "实体", "车辆"),
    ("枫树", "树木", "植物", "生物", "实体", "乐器"),
    ("长笛", "乐器", "器物", "物理对象", "实体", "动物"),
    ("蛋白石", "宝石", "矿物", "物理对象", "实体", "植物"),
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
        "branch_policy": "failure closes only that branch; descriptive strata remain visible",
        "human_review": "external reviewers unavailable locally; never impute or fabricate",
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
    return "discovery" if unit < 3 else "confirmation" if unit < 5 else "lockbox"


def ordered_candidates(answer: str, candidates: list[str], offset: int) -> list[str]:
    values = [answer] + [v for v in candidates if v != answer]
    if len(values) != len(set(values)):
        raise RuntimeError((answer, values))
    target = offset % len(values)
    others = values[1:]
    return others[:target] + [answer] + others[target:]


def render(language: str, surface: str, facts: str, question: str) -> str:
    if language == "en":
        lead = "Research record" if surface == "canonical" else "A curator gives this record"
        return f"{lead}: {facts} {question} Reply with exactly one answer phrase."
    lead = "研究记录" if surface == "canonical" else "整理员给出以下记录"
    return f"{lead}：{facts}{question}请只返回一个答案短语。"


def make_row(panel: str, domain: str, language: str, surface: str, unit: int, cell: str,
             facts: str, question: str, answer: str, candidates: list[str], roles: dict,
             factors: dict, cell_index: int) -> dict:
    ordered = ordered_candidates(answer, candidates, unit + cell_index + LANGUAGES.index(language) + 2 * SURFACES.index(surface))
    slice_key = "|".join((panel, domain, language, surface))
    return {
        "case_id": f"c625|{panel}|{domain}|{language}|{surface}|u{unit:02d}|{cell}",
        "panel": panel, "operation_domain": domain, "language": language, "surface": surface,
        "unit": unit, "partition": partition(unit), "cell": cell, "slice_key": slice_key,
        "prompt": render(language, surface, facts, question), "answer": answer,
        "answer_candidates": ordered, "role_values": roles, "factors": factors,
        "cross_model_subset": language == "en" and surface == "canonical" and unit in (0, 5)
                              and cell in ("depth1", "depth3", "ap00", "ap11", "outer_neg", "update_last"),
    }


def pseudo_nodes(unit: int, language: str) -> tuple[str, ...]:
    if language == "en":
        return tuple(f"{stem}{unit}" for stem in ("zorb", "feln", "miv", "tark", "uln", "vex"))
    return tuple(f"{stem}{unit}" for stem in ("佐布", "费伦", "米弗", "塔克", "乌伦", "维克"))


def graph_nodes(unit: int, world: str, language: str) -> tuple[str, ...]:
    natural = NATURAL_EN[unit] if language == "en" else NATURAL_ZH[unit]
    if world == "pseudo":
        return pseudo_nodes(unit, language)
    if world == "counterfactual":
        root, n1, n2, n3, n4, alt = natural
        pool = ("mineral", "artifact", "machine", "vehicle", "instrument", "substance", "entity") if language == "en" \
               else ("矿物", "器物", "机器", "车辆", "乐器", "物质", "实体")
        tail = [value for value in pool if value not in {root, n1, alt}][:3]
        return (root, alt, *tail, n1)
    return natural


def graph_row(world: str, language: str, surface: str, unit: int, cell: str) -> dict:
    root, n1, n2, n3, n4, alt = graph_nodes(unit, world, language)
    if language == "en":
        relation = "Category chain"
        if world == "memory":
            facts = f"Category chain: use ordinary taxonomy for {root}; no temporary redefinition is supplied."
        else:
            facts = f"Category chain: in this temporary registry, {root} is a {n1}; every {n1} is a {n2}; every {n2} is a {n3}; every {n3} is a {n4}."
        questions = {
            "depth1": f"What is the direct category of {root}?",
            "depth2": f"What category is two steps above {root}?",
            "depth3": f"What category is three steps above {root}?",
            "depth4": f"What category is four steps above {root}?",
            "shortcut": f"Also, {root} is directly listed as a {n3}. What category remains four steps above {root}?",
            "distractor": f"Separately, {alt} is unrelated to {root}. What category is four steps above {root}?",
            "reverse": f"Which item is at the bottom of this category chain that ends at {n4}?",
            "exception": f"Exception: for the direct-category query only, record {root} as a {alt}. What is its direct recorded category?",
        }
    else:
        relation = "类别链"
        if world == "memory":
            facts = f"类别链：请使用关于{root}的普通分类知识，不提供临时重定义。"
        else:
            facts = f"类别链：在这个临时登记中，{root}属于{n1}；每个{n1}都属于{n2}；每个{n2}都属于{n3}；每个{n3}都属于{n4}。"
        questions = {
            "depth1": f"{root}的直接类别是什么？",
            "depth2": f"{root}向上两步的类别是什么？",
            "depth3": f"{root}向上三步的类别是什么？",
            "depth4": f"{root}向上四步的类别是什么？",
            "shortcut": f"另外直接登记{root}属于{n3}。{root}向上四步的类别仍是什么？",
            "distractor": f"另外，{alt}与{root}无关。{root}向上四步的类别是什么？",
            "reverse": f"在终点为{n4}的类别链底部是哪一项？",
            "exception": f"例外：只对直接类别查询，把{root}登记为{alt}。它的直接登记类别是什么？",
        }
    answers = {"depth1": n1, "depth2": n2, "depth3": n3, "depth4": n4,
               "shortcut": n4, "distractor": n4, "reverse": root, "exception": alt}
    candidates = [root, n1, n2, n3, n4, alt]
    return make_row("graph", f"graph_{world}", language, surface, unit, cell, facts, questions[cell],
                    answers[cell], candidates,
                    {"primary": root, "secondary": root, "relation": relation,
                     "context": root, "query": root},
                    {"world": world, "program": "typed_graph", "cell": cell}, GRAPH_CELLS.index(cell))


def attitude_row(language: str, surface: str, unit: int, cell: str) -> dict:
    p0, p1, p2, p3 = PEOPLE[unit]
    o0, o1, o2, o3 = OBJECTS[unit]
    if language == "zh":
        relation = "喜欢"
        scope_candidates = ["报告", "进食", "两者", "都不是"]
        if cell in ("ap00", "ap10", "ap01", "ap11"):
            agent = p2 if cell in ("ap10", "ap11") else p1
            obj = o1 if cell in ("ap01", "ap11") else o0
            facts = f"{p0}报告说，{agent}喜欢吃{obj}。"
            question = f"谁执行吃这个动作？问题中的对象是{obj}。"
            answer, candidates, domain = agent, list((p0, p1, p2, p3)), "attitude_factorial"
            role_secondary, role_context = agent, obj
        elif cell in ("active", "passive"):
            relation = "进食"
            facts = f"{p0}的进食记录：" + (f"{p1}吃了{o0}。" if cell == "active" else f"{o0}被{p1}吃了。")
            question = f"谁是吃这个动作的施事？对象是{o0}。"
            answer, candidates, domain = p1, list((p0, p1, p2, p3)), "attitude_voice"
            role_secondary, role_context = p1, o0
        elif cell in ("outer_neg", "inner_neg"):
            relation = "报告"
            facts = f"{p0}没有报告{p1}吃{o0}。" if cell == "outer_neg" else f"{p0}报告{p1}没有吃{o0}。"
            question = "被否定的是哪一层？只回答报告、进食、两者或都不是。"
            answer, candidates, domain = ("报告" if cell == "outer_neg" else "进食"), scope_candidates, "attitude_scope"
            role_secondary, role_context = p1, o0
        else:
            current = "第一次更新" if cell == "update_first" else "第二次更新"
            facts = (f"{p0}记录更新：第一次更新说{p1}吃{o0}；第二次更新说{p2}吃{o1}。"
                     f"当前记录采用{current}。")
            question = "哪一次更新决定当前记录？只回答第一次或第二次。"
            answer = "第一次" if cell == "update_first" else "第二次"
            candidates, domain, relation = ["第一次", "第二次", "两次", "都不是"], "attitude_update", "更新"
            role_secondary, role_context = p1, o0
        roles = {"primary": p0, "secondary": role_secondary, "relation": relation,
                 "context": role_context, "query": role_context}
    else:
        relation = "likes"
        scope_candidates = ["reporting", "eating", "both", "neither"]
        if cell in ("ap00", "ap10", "ap01", "ap11"):
            agent = p2 if cell in ("ap10", "ap11") else p1
            obj = o1 if cell in ("ap01", "ap11") else o0
            facts = f"{p0} reports that {agent} likes eating {obj}."
            question = f"Who performs the eating action? The queried object is {obj}."
            answer, candidates, domain = agent, list((p0, p1, p2, p3)), "attitude_factorial"
            role_secondary, role_context = agent, obj
        elif cell in ("active", "passive"):
            relation = "eating"
            facts = f"{p0} gives an eating record: " + (f"{p1} ate {o0}." if cell == "active" else f"{o0} was eaten by {p1}.")
            question = f"Who is the agent of eating? The object is {o0}."
            answer, candidates, domain = p1, list((p0, p1, p2, p3)), "attitude_voice"
            role_secondary, role_context = p1, o0
        elif cell in ("outer_neg", "inner_neg"):
            relation = "Report"
            facts = "Report scope: " + (f"{p0} did not report that {p1} ate {o0}." if cell == "outer_neg" else f"{p0} reported that {p1} did not eat {o0}.")
            question = "Which level is negated? Answer reporting, eating, both, or neither."
            answer, candidates, domain = ("reporting" if cell == "outer_neg" else "eating"), scope_candidates, "attitude_scope"
            role_secondary, role_context = p1, o0
        else:
            current = "first" if cell == "update_first" else "second"
            facts = (f"{p0} records an update sequence: the first update says {p1} ate {o0}; "
                     f"the second update says {p2} ate {o1}. The current record follows the {current} update.")
            question = "Which update determines the current record? Answer first, second, both, or neither."
            answer = "first" if cell == "update_first" else "second"
            candidates, domain, relation = ["first", "second", "both", "neither"], "attitude_update", "update"
            role_secondary, role_context = p1, o0
        roles = {"primary": p0, "secondary": role_secondary, "relation": relation,
                 "context": role_context, "query": role_context}
    return make_row("attitude", domain, language, surface, unit, cell, facts, question, answer, candidates,
                    roles, {"program": "attitude_event", "cell": cell}, ATTITUDE_CELLS.index(cell))


def make_material() -> list[dict]:
    rows = [graph_row(w, l, s, u, c) for w, l, s, u, c in
            itertools.product(WORLDS, LANGUAGES, SURFACES, range(UNITS), GRAPH_CELLS)]
    rows += [attitude_row(l, s, u, c) for l, s, u, c in
             itertools.product(LANGUAGES, SURFACES, range(UNITS), ATTITUDE_CELLS)]
    return rows


def character_aligned_spans(tokenizer, prompt: str, ids: list[int], value: str) -> list[list[int]]:
    """Map a literal role mention to physical tokens without isolated-token assumptions."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        encoded_ids = [int(item) for item in encoded["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    except (TypeError, NotImplementedError, ValueError):
        return []
    if encoded_ids != ids or len(offsets) != len(ids):
        raise RuntimeError(("chat_offset_identity_mismatch", len(encoded_ids), len(ids)))
    spans = []
    for match in re.finditer(re.escape(value), rendered):
        selected = [i for i, (start, end) in enumerate(offsets)
                    if end > start and end > match.start() and start < match.end()]
        if selected:
            spans.append(selected)
    return spans


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = previous.c607.text_core.chat_ids(tokenizer, SYSTEM, row["prompt"])
        prefix = "" if row["language"] == "zh" else " "
        candidate_ids = [tokenizer.encode(prefix + answer, add_special_tokens=False) for answer in row["answer_candidates"]]
        if not all(candidate_ids):
            raise RuntimeError((row["case_id"], "empty candidate"))
        positions = {}
        for role, value in row["role_values"].items():
            spans = character_aligned_spans(tokenizer, row["prompt"], ids, value)
            if not spans:
                spans = previous.c607.compiler.graph_base.name_spans(tokenizer, ids, value)
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
    return " ".join(value.strip().lower().split()).strip(".,;:!?\"'`()[]{}，。；：！？")


def generated_prediction(text: str, candidates: list[str]) -> int:
    value = normalize(text)
    exact = [i for i, candidate in enumerate(candidates) if value == normalize(candidate)]
    if len(exact) == 1:
        return exact[0]
    starts = [i for i, candidate in enumerate(candidates)
              if value.startswith(normalize(candidate) + " ") or value.startswith(normalize(candidate) + "，")]
    return starts[0] if len(starts) == 1 else -1


def material_path() -> Path:
    return OUTS["C625"] / "material/flagship_programs.jsonl"


def compiled_path() -> Path:
    return OUTS["C625"] / "material/qwen_compiled.jsonl"


def behavior_path() -> Path:
    return OUTS["C625"] / "behavior/qwen_behavior.jsonl"


def states_path() -> Path:
    return OUTS["C626"] / "raw/role_last.float16.npy"


def index_path() -> Path:
    return OUTS["C626"] / "raw/hidden_index.jsonl"


def metric(pred: np.ndarray, truth: np.ndarray) -> dict:
    p, t = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    err = p - t
    rms_t = float(np.sqrt(np.mean(t * t)))
    rms_e = float(np.sqrt(np.mean(err * err)))
    pf, tf = p.reshape(-1), t.reshape(-1)
    return {"nrmse": rms_e / (rms_t + 1e-12),
            "cosine": float(np.dot(pf, tf) / (np.linalg.norm(pf) * np.linalg.norm(tf) + 1e-12)),
            "sign_agreement": float(np.mean(np.sign(p) == np.sign(t))),
            "truth_rms": rms_t, "error_rms": rms_e}


def role_state(states: np.ndarray, row: dict, q: int) -> np.ndarray:
    return np.asarray(states[int(row["hidden_index"]), int(q)], np.float32)


def c625() -> None:
    out = begin("C625", {
        "object": "two flagship programs: attitude-event binding and four-world typed graphs",
        "coverage": {"languages": LANGUAGES, "surfaces": SURFACES, "units": UNITS,
                     "worlds": WORLDS, "graph_cells": GRAPH_CELLS, "attitude_cells": ATTITUDE_CELLS},
        "partitions": "unit 0-2 discovery, 3-4 confirmation, 5 lockbox",
        "interface": "Chinese candidates have no artificial leading blank; English candidates use one leading blank",
        "qualification": "candidate and open generation accuracy each >=0.75 per frozen slice",
        "human_review": "blank external template; human validity remains NA",
    }, {"C624": previous.load(previous.RESULT / "phase2158_c624_guard_metric_recovery_audit/analysis/final.json")["all_checks_passed"]})
    rows = make_material()
    if len({r["case_id"] for r in rows}) != len(rows) or len({r["prompt"] for r in rows}) != len(rows):
        raise RuntimeError("material identity collision")
    write_rows(material_path(), rows)
    max_position = max(sum(r["answer_candidates"].index(r["answer"]) == i for r in rows) / len(rows)
                       for i in range(max(len(r["answer_candidates"]) for r in rows)))
    longest = sum(max(r["answer_candidates"], key=len) == r["answer"] for r in rows) / len(rows)
    human = [{"case_id": r["case_id"], "naturalness_1_5": None, "semantic_uniqueness_0_1": None,
              "answerability_0_1": None, "reviewer": None}
             for r in rows if r["partition"] == "lockbox" and r["operation_domain"] != "graph_pseudo"]
    write_rows(out / "external/human_blind_template.jsonl", human)

    model = None
    try:
        model, tokenizer, device, placement = previous.c607.passport.previous.model_base().load_bf16("qwen3")
        compiled = compile_rows(tokenizer, rows)
        write_rows(compiled_path(), compiled)
        scores_all = previous.c607.batch_candidate_scores(model, device, compiled, batch_size=12)
        behavior = []
        for i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = previous.c607.greedy_text(model, tokenizer, device, item["prompt_ids"], max_new_tokens=18)
            candidate_prediction = int(np.argmax(scores))
            generation_prediction = generated_prediction(text, item["answer_candidates"])
            behavior.append({"case_id": item["case_id"], "candidate_prediction": candidate_prediction,
                             "candidate_correct": candidate_prediction == item["gold_position"],
                             "candidate_scores": scores, "generated_text": text,
                             "generated_prediction": generation_prediction,
                             "generated_correct": generation_prediction == item["gold_position"]})
            if i % 32 == 0 or i + 1 == len(compiled):
                print(f"[C625 behavior] {i + 1}/{len(compiled)}", flush=True)
        write_rows(behavior_path(), behavior)
    finally:
        previous.c607.passport.previous.model_base().release_bf16(model)
        gc.collect()

    by_behavior = {r["case_id"]: r for r in behavior}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["slice_key"]].append(by_behavior[row["case_id"]])
    slices = {}
    for key, values in sorted(grouped.items()):
        ca = float(np.mean([v["candidate_correct"] for v in values]))
        ga = float(np.mean([v["generated_correct"] for v in values]))
        slices[key] = {"rows": len(values), "candidate_accuracy": ca, "generated_accuracy": ga,
                       "qualified": ca >= BEHAVIOR_GATE and ga >= BEHAVIOR_GATE}
    save(out / "behavior/slice_qualification.json", slices)
    headline = {"status": "flagship_interface_closed", "rows": len(rows),
                "partition_counts": {p: sum(r["partition"] == p for r in rows) for p in ("discovery", "confirmation", "lockbox")},
                "panel_counts": {p: sum(r["panel"] == p for r in rows) for p in ("graph", "attitude")},
                "candidate_accuracy": float(np.mean([v["candidate_correct"] for v in behavior])),
                "generated_accuracy": float(np.mean([v["generated_correct"] for v in behavior])),
                "qualified_slices": sum(v["qualified"] for v in slices.values()), "total_slices": len(slices),
                "slices": slices, "zero_models": {"max_fixed_position": max_position, "longest_answer": longest},
                "human_review": "NA_pending_external_review", "material_sha256": digest(rows),
                "contract_correction": "C616 natural English 0/5 is formal; natural Chinese 0/5 was descriptive with mechanism NA.",
                "strict_interpretation": "Machine behavior and compilation do not replace independent human naturalness review."}
    close("C625", headline, {"large": len(rows) >= 1000, "unique": len({r["prompt"] for r in rows}) == len(rows),
                              "complete": len(behavior) == len(rows), "balanced": max_position <= .25,
                              "human_not_fabricated": all(v["reviewer"] is None for v in human), "finite": finite(headline)},
          "C626_full_coordinate_transmission")


def pair_records(index: list[dict], formal_only: bool) -> list[dict]:
    by_id = {r["case_id"]: r for r in index}
    pairs = []
    def add(left_id: str, right_id: str, operation: str):
        left, right = by_id.get(left_id), by_id.get(right_id)
        if left is None or right is None:
            return
        eligible = left["candidate_correct"] and right["candidate_correct"]
        if formal_only:
            eligible = eligible and left["slice_qualified"] and right["slice_qualified"] and left["generated_correct"] and right["generated_correct"]
        if eligible:
            pairs.append({"left": left, "right": right, "operation": operation,
                          "partition": left["partition"], "formal": formal_only})
    for world, language, surface, unit in itertools.product(WORLDS, LANGUAGES, SURFACES, range(UNITS)):
        root = f"c625|graph|graph_{world}|{language}|{surface}|u{unit:02d}|"
        add(root + "depth1", root + "depth2", f"graph:{world}:{language}:1to2")
        add(root + "depth2", root + "depth3", f"graph:{world}:{language}:2to3")
        add(root + "depth3", root + "depth4", f"graph:{world}:{language}:3to4")
        add(root + "depth4", root + "shortcut", f"graph:{world}:{language}:shortcut")
        add(root + "depth4", root + "distractor", f"graph:{world}:{language}:distractor")
        add(root + "depth1", root + "exception", f"graph:{world}:{language}:exception")
    for language, surface, unit in itertools.product(LANGUAGES, SURFACES, range(UNITS)):
        root = f"c625|attitude|attitude_factorial|{language}|{surface}|u{unit:02d}|"
        add(root + "ap00", root + "ap10", f"attitude:{language}:agent")
        add(root + "ap00", root + "ap01", f"attitude:{language}:patient")
        add(root + "ap10", root + "ap11", f"attitude:{language}:patient_after_agent")
        add(root + "ap01", root + "ap11", f"attitude:{language}:agent_after_patient")
        add(f"c625|attitude|attitude_voice|{language}|{surface}|u{unit:02d}|active",
            f"c625|attitude|attitude_voice|{language}|{surface}|u{unit:02d}|passive", f"attitude:{language}:voice")
        add(f"c625|attitude|attitude_scope|{language}|{surface}|u{unit:02d}|outer_neg",
            f"c625|attitude|attitude_scope|{language}|{surface}|u{unit:02d}|inner_neg", f"attitude:{language}:scope")
        add(f"c625|attitude|attitude_update|{language}|{surface}|u{unit:02d}|update_first",
            f"c625|attitude|attitude_update|{language}|{surface}|u{unit:02d}|update_last", f"attitude:{language}:update")
    return pairs


def fit_guard_metrics(states: np.ndarray, pairs: list[dict]) -> tuple[dict, list[str]]:
    results, candidates = {}, []
    operations = sorted({p["operation"] for p in pairs})
    for operation in operations:
        train = [p for p in pairs if p["operation"] == operation and p["partition"] == "discovery"]
        test = [p for p in pairs if p["operation"] == operation and p["partition"] == "lockbox"]
        if len(train) < 2 or not test:
            continue
        wrong_pool = [p for p in pairs if p["operation"] != operation and p["partition"] == "discovery"]
        for q in QPOINTS:
            htr = np.stack([role_state(states, p["left"], q) for p in train])
            dtr = np.stack([role_state(states, p["right"], q) - role_state(states, p["left"], q) for p in train])
            hte = np.stack([role_state(states, p["left"], q) for p in test])
            truth = np.stack([role_state(states, p["right"], q) - role_state(states, p["left"], q) for p in test])
            mean = np.mean(dtr, axis=0)
            hx, dx = np.mean(htr, axis=0), np.mean(dtr, axis=0)
            beta = np.sum((htr - hx) * (dtr - dx), axis=0) / (np.sum((htr - hx) ** 2, axis=0) + 1e-6)
            diagonal = dx + beta * (hte - hx)
            plus = np.mean(np.where(htr >= 0, dtr, np.nan), axis=0)
            minus = np.mean(np.where(htr < 0, dtr, np.nan), axis=0)
            plus = np.nan_to_num(plus, nan=0.0); minus = np.nan_to_num(minus, nan=0.0)
            sign_guard = np.where(hte >= 0, plus, minus)
            nearest = []
            for h in hte:
                nearest.append(dtr[int(np.argmin(np.mean((htr - h) ** 2, axis=(1, 2))))])
            nearest = np.stack(nearest)
            if wrong_pool:
                wrong = np.mean(np.stack([role_state(states, p["right"], q) - role_state(states, p["left"], q)
                                          for p in wrong_pool[:len(train)]]), axis=0)
            else:
                wrong = -mean
            models = {"identity": np.zeros_like(truth), "mean": np.broadcast_to(mean, truth.shape),
                      "diagonal": diagonal, "sign_guard": sign_guard, "nearest": nearest,
                      "wrong_operation": np.broadcast_to(wrong, truth.shape)}
            metrics = {name: metric(pred, truth) for name, pred in models.items()}
            best = min(("diagonal", "sign_guard", "nearest"), key=lambda name: metrics[name]["nrmse"])
            control = min(metrics[name]["nrmse"] for name in ("identity", "mean", "wrong_operation"))
            gate = metrics[best]["nrmse"] <= control - CONTROL_MARGIN
            key = f"{operation}|q{q}"
            results[key] = {"train": len(train), "test": len(test), "models": metrics,
                            "best_conditional": best, "control_nrmse": control, "gate": gate}
            if gate:
                candidates.append(key)
    return results, candidates


def finite_difference_scan(model, item: dict, source_q: int, source_pos: int, target_pos: int,
                           eps: float, output_path: Path, pair_batch: int = 4) -> dict:
    base = model.model
    target_q = min(source_q + 8, len(base.layers))
    matrix = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=(2, DIM, DIM))
    logit_path = output_path.with_name(output_path.stem + ".candidate_logits.float16.npy")
    logit_j = np.lib.format.open_memmap(logit_path, mode="w+", dtype=np.float16,
                                        shape=(DIM, len(item["candidate_ids"])))
    base_ids = torch.tensor(item["prompt_ids"], dtype=torch.long, device=next(model.parameters()).device)
    candidate_first = [int(ids[0]) for ids in item["candidate_ids"]]
    for start in range(0, DIM, pair_batch):
        coords = list(range(start, min(start + pair_batch, DIM)))
        b = len(coords)
        ids = base_ids[None].repeat(2 * b, 1)
        mask = torch.ones_like(ids); pos_ids = torch.arange(ids.shape[1], device=ids.device)[None].repeat(2 * b, 1)
        captured: dict[str, torch.Tensor] = {}
        def inject(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            changed = tensor.clone()
            for i, coordinate in enumerate(coords):
                changed[i, source_pos, coordinate] += eps
                changed[b + i, source_pos, coordinate] -= eps
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        def capture_next(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured["next"] = tensor.detach()
        def capture_final(_module, _args, output):
            captured["final"] = output.detach()
        handles = [base.layers[source_q - 1].register_forward_hook(inject),
                   base.layers[target_q - 1].register_forward_hook(capture_next),
                   base.norm.register_forward_hook(capture_final)]
        try:
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos_ids,
                               use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        for target_i, key in enumerate(("next", "final")):
            values = captured[key][:, target_pos].float().cpu().numpy()
            derivative = (values[:b] - values[b:]) / (2.0 * eps)
            matrix[target_i, start:start + b] = derivative.astype(np.float16)
        logits = output.logits[:, -1].float().cpu().numpy()
        for i in range(b):
            logit_j[start + i] = np.asarray([(logits[i, token] - logits[b + i, token]) / (2.0 * eps)
                                             for token in candidate_first], np.float16)
        if start % 256 == 0 or start + b == DIM:
            print(f"[finite scan] q{source_q} eps={eps} {start + b}/{DIM}", flush=True)
        del output, captured, ids, mask, pos_ids
    matrix.flush(); logit_j.flush()
    summary = {"path": str(output_path.relative_to(ROOT)), "logit_path": str(logit_path.relative_to(ROOT)),
               "shape": [2, DIM, DIM], "source_q": source_q, "target_q": [target_q, 37],
               "source_pos": source_pos, "target_pos": target_pos, "epsilon": eps,
               "probe": "central finite difference over every physical source coordinate; no gradient or coordinate preselection"}
    del matrix, logit_j
    return summary


def c626() -> None:
    out = begin("C626", {
        "object": "full field observation, full-coordinate guard competition and registered dense coordinate transmission panels",
        "capture": "all rows, all tokens, embedding + 36 blocks + final norm, all 2560 signed coordinates",
        "guard": "formal claims require dual-qualified and dual-correct pairs; descriptive rows remain captured",
        "finite_difference": "central +/- doses for every source coordinate at registered samples; two downstream full-coordinate targets",
        "scope_boundary": "registered matrices are local model-specific panels, not the complete network Jacobian",
    }, {"C625": final("C625")["all_checks_passed"]})
    compiled = read_rows(compiled_path()); behavior = {r["case_id"]: r for r in read_rows(behavior_path())}
    slices = final("C625")["headline"]["slices"]
    n = len(compiled)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    states = np.lib.format.open_memmap(states_path(), mode="w+", dtype=np.float16,
                                       shape=(n, CHECKPOINTS, len(ROLES), DIM))
    shard_dir = out / "raw/full_token_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    model = None; hooks = []; captured = []; index = []; ledger = []
    try:
        model, tokenizer, device, placement = previous.c607.passport.previous.model_base().load_bf16("qwen3")
        base = model.model
        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)
        hooks.append(base.embed_tokens.register_forward_hook(hook))
        hooks.extend(layer.register_forward_hook(hook) for layer in base.layers)
        hooks.append(base.norm.register_forward_hook(hook))
        for start in range(0, n, 24):
            items = compiled[start:start + 24]
            width = max(len(item["prompt_ids"]) for item in items)
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
                index.append({"hidden_index": start + local, "case_id": item["case_id"], "panel": item["panel"],
                              "operation_domain": item["operation_domain"], "language": item["language"],
                              "surface": item["surface"], "unit": item["unit"], "partition": item["partition"],
                              "cell": item["cell"], "slice_key": item["slice_key"], "factors": item["factors"],
                              "role_positions": item["role_positions"], "candidate_correct": b["candidate_correct"],
                              "generated_correct": b["generated_correct"],
                              "slice_qualified": slices[item["slice_key"]]["qualified"],
                              "shard": shard_path.name, "shard_row": local, "token_count": len(item["prompt_ids"])})
                del ids, mask, pos
            shard.flush(); ledger.append({"file": shard_path.name, "rows": len(items), "width": width, "bytes": shard_path.stat().st_size})
            del shard
            print(f"[C626 capture] {min(start + len(items), n)}/{n}", flush=True)
        for handle in hooks:
            handle.remove()
        hooks.clear(); captured.clear(); states.flush(); write_rows(index_path(), index); save(out / "raw/shard_ledger.json", ledger)

        formal_pairs = pair_records(index, formal_only=True)
        metrics, guard_candidates = fit_guard_metrics(states, formal_pairs)
        save(out / "analysis/guard_metrics.json", metrics)

        eligible = [p for p in formal_pairs if p["partition"] == "lockbox" and any(k.startswith(p["operation"] + "|") for k in guard_candidates)]
        selected = []
        for wanted in ("graph:pseudo", "attitude:"):
            found = next((p for p in eligible if p["operation"].startswith(wanted)), None)
            if found and found not in selected:
                selected.append(found)
        for pair in eligible:
            if len(selected) >= 2:
                break
            if pair not in selected:
                selected.append(pair)
        if not selected:
            descriptive = [p for p in pair_records(index, formal_only=False) if p["partition"] == "lockbox"]
            selected = descriptive[:2]
        compiled_by_id = {r["case_id"]: r for r in compiled}
        scan_summaries = []
        for scan_i, pair in enumerate(selected):
            item = compiled_by_id[pair["left"]["case_id"]]
            for eps in (0.025, 0.05):
                path = out / f"raw/coordinate_scan_{scan_i:02d}_eps{str(eps).replace('.', 'p')}.float16.npy"
                scan_summaries.append({"operation": pair["operation"], "case_id": item["case_id"],
                                       "formal": pair.get("formal", False),
                                       **finite_difference_scan(model, item, 24,
                                            int(item["role_positions"]["query"][-1]),
                                            int(item["role_positions"]["boundary"][-1]), eps, path)})
        save(out / "analysis/coordinate_scan_manifest.json", scan_summaries)
    finally:
        for handle in hooks:
            handle.remove()
        previous.c607.passport.previous.model_base().release_bf16(model)
        states.flush(); del states; gc.collect()

    dose_consistency = []
    for scan_i in range(len(selected)):
        p1 = out / f"raw/coordinate_scan_{scan_i:02d}_eps0p025.float16.npy"
        p2 = out / f"raw/coordinate_scan_{scan_i:02d}_eps0p05.float16.npy"
        a = np.load(p1, mmap_mode="r"); b = np.load(p2, mmap_mode="r")
        dose_consistency.append({"scan": scan_i, "operation": selected[scan_i]["operation"],
                                 "matrix_cosine": metric(np.asarray(a, np.float32), np.asarray(b, np.float32))["cosine"],
                                 "relative_difference": metric(np.asarray(a, np.float32), np.asarray(b, np.float32))["nrmse"]})
        previous.c607.passport.close_mmap(a); previous.c607.passport.close_mmap(b)
    save(out / "analysis/dose_consistency.json", dose_consistency)
    headline = {"status": "full_coordinate_transmission_closed", "capture_rows": n,
                "capture_shape": [n, CHECKPOINTS, len(ROLES), DIM], "full_token_shards": len(ledger),
                "full_token_bytes": sum(v["bytes"] for v in ledger), "formal_pairs": len(formal_pairs),
                "metric_cells": len(metrics), "guard_candidates": guard_candidates,
                "coordinate_scans": len(scan_summaries), "dose_pairs": dose_consistency,
                "strict_interpretation": "Dense finite-difference panels are local causal response maps for registered states, not a complete Jacobian or unique circuit."}
    close("C626", headline, {"capture": n == len(index), "shape": headline["capture_shape"][1:] == [38, 6, 2560],
                              "all_coordinates": all(s["shape"] == [2, 2560, 2560] for s in scan_summaries),
                              "dose_registered": len(scan_summaries) == 2 * len(selected), "finite": finite(headline)},
          "C627_unseen_composition")


def c627() -> None:
    out = begin("C627", {
        "object": "unseen graph depth and attitude factorial composition using only discovery atomic transitions",
        "graph": "roll predictions through their own intermediate states; compare additive, wrong-middle and zero",
        "attitude": "predict lockbox second-order interaction from discovery interaction prototype",
        "formal": "only dual-qualified, dual-correct cells can yield a mechanism pass; other cells are descriptive NA",
    }, {"C626": final("C626")["all_checks_passed"]})
    states = np.load(states_path(), mmap_mode="r"); index = read_rows(index_path())
    by_id = {r["case_id"]: r for r in index}
    graph_results = {}
    for world, language, surface, q in itertools.product(WORLDS, LANGUAGES, SURFACES, QPOINTS):
        train_units = (0, 1, 2); test_unit = 5
        def get(unit, cell):
            return by_id.get(f"c625|graph|graph_{world}|{language}|{surface}|u{unit:02d}|{cell}")
        train = []
        for unit in train_units:
            rows = [get(unit, cell) for cell in ("depth1", "depth2", "depth3", "depth4")]
            if all(rows):
                train.append(rows)
        test = [get(test_unit, cell) for cell in ("depth1", "depth2", "depth3", "depth4")]
        if not train or not all(test):
            continue
        formal = all(r["slice_qualified"] and r["candidate_correct"] and r["generated_correct"] for rows in train for r in rows)
        formal = formal and all(r["slice_qualified"] and r["candidate_correct"] and r["generated_correct"] for r in test)
        deltas = [[role_state(states, rows[i + 1], q) - role_state(states, rows[i], q) for rows in train] for i in range(3)]
        current = role_state(states, test[0], q).copy(); additive = current.copy(); wrong = current.copy()
        for step in range(3):
            left_states = np.stack([role_state(states, rows[step], q) for rows in train])
            step_delta = np.stack(deltas[step])
            donor = int(np.argmin(np.mean((left_states - current) ** 2, axis=(1, 2))))
            far = int(np.argmax(np.mean((left_states - wrong) ** 2, axis=(1, 2))))
            current = current + step_delta[donor]
            additive = additive + np.mean(step_delta, axis=0)
            wrong = wrong + step_delta[far]
        truth = role_state(states, test[3], q)
        values = {"sequential": metric(current, truth), "additive": metric(additive, truth),
                  "wrong_middle": metric(wrong, truth), "zero": metric(role_state(states, test[0], q), truth)}
        seq = values["sequential"]
        controls = (values["additive"], values["wrong_middle"], values["zero"])
        gate = formal and seq["nrmse"] <= min(v["nrmse"] for v in controls) - CONTROL_MARGIN \
               and seq["cosine"] >= max(v["cosine"] for v in controls) + CONTROL_MARGIN
        graph_results[f"{world}|{language}|{surface}|q{q}"] = {"formal": formal, "models": values,
                                                                 "gate": gate, "status": "formal" if formal else "NA_behavior"}

    attitude_results = {}
    for language, surface, q in itertools.product(LANGUAGES, SURFACES, QPOINTS):
        interactions = []
        formal_train = True
        for unit in (0, 1, 2):
            rows = [by_id.get(f"c625|attitude|attitude_factorial|{language}|{surface}|u{unit:02d}|{cell}")
                    for cell in ("ap00", "ap10", "ap01", "ap11")]
            if not all(rows):
                continue
            formal_train = formal_train and all(r["slice_qualified"] and r["candidate_correct"] and r["generated_correct"] for r in rows)
            h = [role_state(states, r, q) for r in rows]
            interactions.append(h[3] - h[1] - h[2] + h[0])
        test_rows = [by_id.get(f"c625|attitude|attitude_factorial|{language}|{surface}|u05|{cell}")
                     for cell in ("ap00", "ap10", "ap01", "ap11")]
        if not interactions or not all(test_rows):
            continue
        formal = formal_train and all(r["slice_qualified"] and r["candidate_correct"] and r["generated_correct"] for r in test_rows)
        h = [role_state(states, r, q) for r in test_rows]
        truth = h[3] - h[1] - h[2] + h[0]
        pred = np.mean(np.stack(interactions), axis=0)
        m_pred, m_zero = metric(pred, truth), metric(np.zeros_like(truth), truth)
        gate = formal and m_pred["nrmse"] <= m_zero["nrmse"] - CONTROL_MARGIN
        attitude_results[f"{language}|{surface}|q{q}"] = {"formal": formal, "prototype": m_pred,
                                                            "zero": m_zero, "gate": gate,
                                                            "status": "formal" if formal else "NA_behavior"}
    previous.c607.passport.close_mmap(states); del states
    save(out / "analysis/graph_composition.json", graph_results)
    save(out / "analysis/attitude_interaction.json", attitude_results)
    headline = {"status": "unseen_composition_closed",
                "graph_formal": sum(v["formal"] for v in graph_results.values()),
                "graph_passed": sum(v["gate"] for v in graph_results.values()),
                "graph_total": len(graph_results),
                "attitude_formal": sum(v["formal"] for v in attitude_results.values()),
                "attitude_passed": sum(v["gate"] for v in attitude_results.values()),
                "attitude_total": len(attitude_results),
                "strict_interpretation": "A pass is a finite unseen-composition prediction; failure or NA in one family does not close sibling families."}
    close("C627", headline, {"graph": bool(graph_results), "attitude": bool(attitude_results), "finite": finite(headline)},
          "C628_output_identity")


def patched_generation(model, tokenizer, item: dict, patches: list[dict], max_new_tokens: int = 12) -> tuple[str, list[int]]:
    base = model.model; handles = []
    for patch in patches:
        q, kind, vector = int(patch["q"]), patch["kind"], np.asarray(patch["vector"], np.float32)
        def make_hook(kind_value, vector_value):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                changed = tensor.clone()
                if kind_value == "roles":
                    for role_i, role in enumerate(ROLES):
                        pos = int(item["role_positions"][role][-1])
                        if pos < changed.shape[1]:
                            changed[0, pos] += torch.tensor(vector_value[role_i], dtype=changed.dtype, device=changed.device)
                else:
                    pos = int(item["role_positions"]["boundary"][-1])
                    if pos < changed.shape[1]:
                        changed[0, pos] += torch.tensor(vector_value, dtype=changed.dtype, device=changed.device)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            return hook
        module = base.norm if q == 37 else base.layers[q - 1]
        handles.append(module.register_forward_hook(make_hook(kind, vector)))
    ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=next(model.parameters()).device)
    mask = torch.ones_like(ids); generated_ids = []
    try:
        for _ in range(max_new_tokens):
            pos = mask.long().cumsum(-1) - 1
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True).logits
            nxt = int(torch.argmax(logits[0, -1]).item())
            generated_ids.append(nxt)
            token = torch.tensor([[nxt]], dtype=torch.long, device=ids.device)
            ids = torch.cat((ids, token), dim=1); mask = torch.cat((mask, torch.ones_like(token)), dim=1)
            if nxt == tokenizer.eos_token_id:
                break
    finally:
        for handle in handles:
            handle.remove()
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), generated_ids


def c628() -> None:
    out = begin("C628", {
        "object": "2x2 semantic-state versus output-identity-state intervention at registered generation times",
        "semantic": "discovery nearest response at q24 over roles with boundary role zeroed",
        "identity": "exact target-minus-source q32 boundary state; explicit strong instrument, not a minimal mechanism",
        "positive_control": "exact q37 boundary donor difference for first-token logits only",
        "modes": ["zero", "semantic", "identity", "joint", "wrong_identity", "final_identity_control"],
        "necessity": "target q32 identity deletion; rescue only if deletion first breaks target generation",
    }, {"C627": final("C627")["all_checks_passed"]})
    states = np.load(states_path(), mmap_mode="r"); index = read_rows(index_path())
    compiled = {r["case_id"]: r for r in read_rows(compiled_path())}
    formal_pairs = pair_records(index, formal_only=True)
    operations = sorted({k.rsplit("|q", 1)[0] for k in final("C626")["headline"]["guard_candidates"]})
    eligible = [p for p in formal_pairs if p["partition"] == "lockbox" and p["operation"] in operations
                and compiled[p["left"]["case_id"]]["answer"] != compiled[p["right"]["case_id"]]["answer"]][:12]
    train_by_op = {op: [p for p in formal_pairs if p["partition"] == "discovery" and p["operation"] == op] for op in operations}
    model = None; records = []
    try:
        model, tokenizer, device, placement = previous.c607.passport.previous.model_base().load_bf16("qwen3")
        for pair in eligible:
            source, target = pair["left"], pair["right"]
            item, target_item = compiled[source["case_id"]], compiled[target["case_id"]]
            train = train_by_op[pair["operation"]]
            h = role_state(states, source, 24)
            donor = train[int(np.argmin([np.mean((role_state(states, p["left"], 24) - h) ** 2) for p in train]))]
            semantic = role_state(states, donor["right"], 24) - role_state(states, donor["left"], 24)
            semantic[ROLES.index("boundary")] = 0
            identity = role_state(states, target, 32)[ROLES.index("boundary")] - role_state(states, source, 32)[ROLES.index("boundary")]
            final_identity = role_state(states, target, 37)[ROLES.index("boundary")] - role_state(states, source, 37)[ROLES.index("boundary")]
            wrong_pair = next((p for p in formal_pairs if p["operation"] != pair["operation"] and p["partition"] == "discovery"), None)
            wrong = identity[::-1] if wrong_pair is None else role_state(states, wrong_pair["right"], 32)[5] - role_state(states, wrong_pair["left"], 32)[5]
            wrong = previous.transport.scaled_like(wrong, identity)
            modes = {
                "zero": [], "semantic": [{"q": 24, "kind": "roles", "vector": semantic}],
                "identity": [{"q": 32, "kind": "boundary", "vector": identity}],
                "joint": [{"q": 24, "kind": "roles", "vector": semantic}, {"q": 32, "kind": "boundary", "vector": identity}],
                "wrong_identity": [{"q": 32, "kind": "boundary", "vector": wrong}],
                "final_identity_control": [{"q": 37, "kind": "boundary", "vector": final_identity}],
            }
            target_pos = item["answer_candidates"].index(target_item["answer"])
            outputs = {}
            for name, patches in modes.items():
                text, token_ids = patched_generation(model, tokenizer, item, patches)
                outputs[name] = {"text": text, "target": generated_prediction(text, item["answer_candidates"]) == target_pos,
                                 "first_token": token_ids[0] if token_ids else None,
                                 "first_token_target": bool(token_ids) and token_ids[0] in item["candidate_ids"][target_pos][:1]}
            target_natural, _ = patched_generation(model, tokenizer, target_item, [])
            deletion, _ = patched_generation(model, tokenizer, target_item,
                                               [{"q": 32, "kind": "boundary", "vector": -identity}])
            natural_ok = generated_prediction(target_natural, target_item["answer_candidates"]) == target_item["gold_position"]
            deletion_ok = generated_prediction(deletion, target_item["answer_candidates"]) == target_item["gold_position"]
            target_values = {"natural": target_natural, "natural_ok": natural_ok, "deletion": deletion,
                             "deletion_ok": deletion_ok, "rescue_eligible": natural_ok and not deletion_ok}
            if target_values["rescue_eligible"]:
                rescue, _ = patched_generation(model, tokenizer, target_item,
                                                [{"q": 32, "kind": "boundary", "vector": -identity},
                                                 {"q": 32, "kind": "boundary", "vector": identity}])
                wrong_rescue, _ = patched_generation(model, tokenizer, target_item,
                                                      [{"q": 32, "kind": "boundary", "vector": -identity},
                                                       {"q": 32, "kind": "boundary", "vector": wrong}])
                target_values.update({"rescue": rescue,
                                      "rescue_ok": generated_prediction(rescue, target_item["answer_candidates"]) == target_item["gold_position"],
                                      "wrong_rescue": wrong_rescue,
                                      "wrong_rescue_ok": generated_prediction(wrong_rescue, target_item["answer_candidates"]) == target_item["gold_position"]})
            records.append({"operation": pair["operation"], "source": source["case_id"], "target": target["case_id"],
                            "target_answer": target_item["answer"], "outputs": outputs, "target_values": target_values})
            print(f"[C628] {len(records)}/{len(eligible)}", flush=True)
    finally:
        previous.c607.passport.previous.model_base().release_bf16(model)
        previous.c607.passport.close_mmap(states); del states; gc.collect()
    write_rows(out / "analysis/output_identity_records.jsonl", records)
    modes = ("zero", "semantic", "identity", "joint", "wrong_identity", "final_identity_control")
    summary = {name: {"tests": len(records), "target": sum(r["outputs"][name]["target"] for r in records),
                      "first_token_target": sum(r["outputs"][name]["first_token_target"] for r in records)} for name in modes}
    deletion_broke = sum(r["target_values"]["rescue_eligible"] for r in records)
    specific_rescue = sum(r["target_values"].get("rescue_ok", False) and not r["target_values"].get("wrong_rescue_ok", False)
                          for r in records)
    headline = {"status": "output_identity_clock_closed", "eligible": len(eligible), "records": len(records),
                "modes": summary, "deletion_broke": deletion_broke, "specific_rescue": specific_rescue,
                "strict_interpretation": "Exact boundary donor differences are instrument controls. They do not identify a minimal output-identity code or an upstream language mechanism."}
    close("C628", headline, {"executed_if_eligible": bool(records) if eligible else True,
                              "mode_balance": all(v["tests"] == len(records) for v in summary.values()),
                              "adaptive_rescue": specific_rescue <= deletion_broke, "finite": finite(headline)},
          "C629_crossmodel_visual_audit")


def run_worker(command: list[str]) -> dict:
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {"returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}


def register_visual() -> None:
    catalog = load(CATALOG)
    entry = {"id": "c629_flagship_gear_atlas", "label": "C629 Flagship Gear Atlas",
             "path": "/vis_data/research_kernel/c629_flagship_gear_atlas.json"}
    values = catalog.setdefault("field_datasets", [])
    values[:] = [v for v in values if v.get("id") != entry["id"]]
    values.append(entry)
    save(CATALOG, catalog)


def c629() -> None:
    out = begin("C629", {
        "object": "sequential model-specific interfaces, exact-coordinate visual publication, cleanup, theory and independent audit",
        "models": ["GLM4", "DeepSeek-7B", "Qwen3-14B"],
        "cross_model": "HiddenState only after each model passes its own dual behavior gate; compare relative layers and role topology, never coordinate IDs",
        "visual": "one exact all-token field and one exact 2560x2560 response matrix plus manifests for every retained matrix",
        "cleanup": "delete undisplayed full-token shards after visual materialization; retain role field and all coordinate-scan matrices",
    }, {"C628": final("C628")["all_checks_passed"]})
    worker = TESTS / "phase2163_c629_model_specific_worker.py"
    workers, supervisor = {}, {}
    for name in ("glm4", "deepseek7b", "qwen3_14b"):
        target = out / f"crossmodel/{name}/final.json"
        command = [str(ROOT / ".venv/Scripts/python.exe"), str(worker), "--model", name,
                   "--material", str(material_path()), "--output", str(target)]
        supervisor[name] = run_worker(command)
        if not target.exists():
            save(target, {"status": "supervisor_error", "model": name, "hiddenstate_ran": False, **supervisor[name]})
        workers[name] = load(target)
        gc.collect()

    ledger = load(OUTS["C626"] / "raw/shard_ledger.json")
    index = read_rows(index_path()); compiled = {r["case_id"]: r for r in read_rows(compiled_path())}
    representative = next((r for r in index if r["partition"] == "lockbox" and r["slice_qualified"] and r["generated_correct"]), index[0])
    shard = np.load(OUTS["C626"] / "raw/full_token_shards" / representative["shard"], mmap_mode="r")
    token_field = np.asarray(shard[representative["shard_row"], :, :representative["token_count"]], np.float16).copy()
    previous.c607.passport.close_mmap(shard); del shard
    scan_manifest = load(OUTS["C626"] / "analysis/coordinate_scan_manifest.json")
    exact_matrix = []
    if scan_manifest:
        matrix = np.load(ROOT / scan_manifest[0]["path"], mmap_mode="r")
        exact_matrix = np.asarray(matrix[0], np.float16).tolist()
        previous.c607.passport.close_mmap(matrix); del matrix
    visual = {
        "schema": "ai2050.flagship_gear_atlas.v1", "phase": 2163,
        "checkpoints": ["embedding"] + [f"block_{i:02d}_post" for i in range(36)] + ["final_norm"],
        "coordinates": list(range(DIM)), "roles": list(ROLES),
        "representative": {"case_id": representative["case_id"],
                           "tokens": compiled[representative["case_id"]]["prompt_ids"],
                           "field_shape": list(token_field.shape), "field": token_field.tolist()},
        "coordinate_scan": {"manifest": scan_manifest, "exact_first_target_shape": [DIM, DIM],
                            "exact_first_target": exact_matrix},
        "behavior": final("C625")["headline"], "guard": final("C626")["headline"],
        "composition": final("C627")["headline"], "output_identity": final("C628")["headline"],
        "crossmodel": workers,
        "claim_boundary": "Exact displayed coordinates are observations, not a unique semantic circuit.",
    }
    save(VISUAL, visual); register_visual(); visual_bytes = VISUAL.stat().st_size
    del token_field, exact_matrix, visual; gc.collect()

    shard_dir = OUTS["C626"] / "raw/full_token_shards"
    cleaned_bytes = sum(path.stat().st_size for path in shard_dir.glob("*.npy")) if shard_dir.exists() else 0
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    empirical = {
        "human_naturalness": False,
        "qwen_flagship_behavior": final("C625")["headline"]["qualified_slices"] > 0,
        "full_coordinate_guard": bool(final("C626")["headline"]["guard_candidates"]),
        "dense_coordinate_response": final("C626")["headline"]["coordinate_scans"] > 0,
        "natural_unseen_composition": any(v.get("gate") and v.get("formal") and v.get("status") == "formal"
                                           and key.startswith(("memory|", "consistent|", "counterfactual|"))
                                           for key, v in load(OUTS["C627"] / "analysis/graph_composition.json").items()),
        "output_identity_nontrivial": final("C628")["headline"]["modes"]["identity"]["target"] > 0,
        "generation_necessity": final("C628")["headline"]["deletion_broke"] > 0,
        "specific_rescue": final("C628")["headline"]["specific_rescue"] > 0,
        "cross_model_functional": sum(v.get("hiddenstate_ran", False) for v in workers.values()) >= 2,
    }
    empirical["new_math"] = all(empirical.values())
    theory = {"name": "Conditional Output Field Closure Theory", "principle": "Reuse-Difference-Conditioning",
              "update": "external language programs, base-conditioned full-coordinate response, local coordinate transmission panels, separate output-identity clock",
              "foundational_math_authorized": empirical["new_math"]}
    save(out / "analysis/empirical_gates.json", empirical); save(out / "analysis/theory.json", theory)

    checks = {
        "phase_finals": all(final(name)["status"] == "closed" for name in ("C625", "C626", "C627", "C628")),
        "material_unique": len({r["case_id"] for r in read_rows(material_path())}) == len(read_rows(material_path())),
        "human_not_fabricated": final("C625")["headline"]["human_review"] == "NA_pending_external_review",
        "role_field": states_path().exists(),
        "coordinate_matrices": all((ROOT / item["path"]).exists() for item in scan_manifest),
        "visual": VISUAL.exists() and visual_bytes > 0,
        "cleanup": not shard_dir.exists(),
        "crossmodel_sequential": all(supervisor[name]["returncode"] == 0 for name in supervisor),
        "new_math_consistent": empirical["new_math"] is False or all(empirical.values()),
    }
    temp_states = np.load(states_path(), mmap_mode="r")
    checks["role_field"] = (checks["role_field"] and temp_states.dtype == np.float16
                            and temp_states.shape[1:] == (CHECKPOINTS, len(ROLES), DIM))
    previous.c607.passport.close_mmap(temp_states); del temp_states
    headline = {"status": "crossmodel_visual_theory_audit_closed", "workers": workers,
                "visual": str(VISUAL.relative_to(ROOT)), "visual_bytes": visual_bytes,
                "cleaned_bytes": cleaned_bytes, "retained_role_field": str(states_path().relative_to(ROOT)),
                "empirical_gates": empirical, "theory": theory,
                "audit": {"passed": sum(checks.values()), "total": len(checks), "checks": checks},
                "strict_interpretation": "The major stage maps finite model-specific laws and missing branches; it does not establish a universal gear algebra."}
    close("C629", headline, checks, "major_stage_closed_new_object_or_external_human_review_required")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=tuple(PHASES), default="C625")
    parser.add_argument("--stop", choices=tuple(PHASES), default="C629")
    args = parser.parse_args()
    names = list(PHASES)
    for name in names[names.index(args.start):names.index(args.stop) + 1]:
        target = OUTS[name] / "analysis/final.json"
        if target.exists() and load(target).get("status") == "closed":
            print(f"[resume] {name} already closed", flush=True)
            continue
        globals()[name.lower()]()


if __name__ == "__main__":
    main()
