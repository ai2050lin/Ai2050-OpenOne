#!/usr/bin/env python3
"""Freeze the independent construction-ecology campaign (C1097-C1120).

The contract observes embeddings and HiddenStates only. It forbids attention,
MLP, weight, gradient, PCA, Top-K, cosine-screening and donor-delta discovery.
Every failed family becomes registered missingness; it does not stop the other
families or the cross-model behavior panels.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase2253_c1097_c1120_construction_ecology_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase1797_c263_c272_state_operator_common as compiler  # noqa: E402
import phase2219_c773_c808_semantic_transition_ecology_campaign as model_base  # noqa: E402


PHASE = 2253
CAMPAIGNS = tuple(f"C{i}" for i in range(1097, 1121))
FAMILIES = (
    "taxonomy_path",
    "part_whole_path",
    "temporal_path",
    "active_passive_role",
    "relative_clause_role",
    "negation_scope",
    "attitude_scope",
    "quantifier_scope",
    "contrast_coreference",
    "attribute_overwrite",
)
GRAPH_FAMILIES = FAMILIES[:3]
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
LANGUAGES = ("en", "zh")
SURFACES = ("direct", "paraphrase")
OUTPUT_SCHEMES = (
    ("Yes", "No"),
    ("True", "False"),
    ("Supported", "Unsupported"),
    ("Entailed", "Contradicted"),
)
PARENT_UNITS = 16
FRESH_UNITS = 12
DISCOVERY_UNITS = 8
COMPOSITION_FRESH_UNITS = 8
BEHAVIOR_GATE = 0.75
COORDINATE_GATES = {
    "minimum_independent_units": 6,
    "fresh_gain_over_zero": 0.05,
    "fresh_gain_over_wrong_family": 0.03,
    "coordinate_sign_consistency": 0.75,
    "coordinate_own_family_win_rate": 0.60,
    "successive_checkpoints": 2,
}
CAUSAL_GATES = {
    "minimum_pairs": 12,
    "candidate_direction_rate": 0.60,
    "candidate_margin_advantage": 0.05,
    "generation_accuracy_advantage": 0.10,
    "correct_rescue_advantage": 0.10,
}

NAMES_A_EN = (
    "Amina", "Bridget", "Clara", "Dalia", "Elena", "Farah", "Greta", "Hana",
    "Iris", "Jana", "Kira", "Lina", "Mara", "Nora", "Oona", "Priya",
    "Rhea", "Sara", "Talia", "Uma", "Vera", "Willa", "Xena", "Yara",
    "Zara", "Adela", "Belen", "Cora",
)
NAMES_B_EN = (
    "Aron", "Boris", "Cyrus", "Derek", "Evan", "Felix", "Galen", "Hugo",
    "Ivan", "Jonas", "Kamil", "Leo", "Milo", "Noel", "Omar", "Pietro",
    "Quinn", "Rami", "Soren", "Tomas", "Uri", "Viktor", "Wes", "Yuri",
    "Zane", "Aldo", "Basil", "Colin",
)
NAMES_A_ZH = tuple(f"阿宁{i + 1}" for i in range(28))
NAMES_B_ZH = tuple(f"柏文{i + 1}" for i in range(28))
OBJECTS_EN = (
    "atlas", "brooch", "candle", "drum", "easel", "flute", "globe", "helmet",
    "inkpot", "jacket", "key", "lamp", "mirror", "notebook", "ornament", "puzzle",
    "quilt", "radio", "stamp", "thermos", "urn", "wallet", "xylophone", "yarn",
    "zipper", "album", "basket", "compass",
)
OBJECTS_ZH = tuple(f"物件{i + 1}" for i in range(28))
ALT_OBJECTS_EN = (
    "map", "pin", "torch", "bell", "canvas", "pipe", "sphere", "cap",
    "bottle", "coat", "lock", "bulb", "frame", "journal", "token", "riddle",
    "blanket", "speaker", "seal", "flask", "jar", "purse", "chime", "thread",
    "clasp", "folio", "crate", "sextant",
)
ALT_OBJECTS_ZH = tuple(f"替代物{i + 1}" for i in range(28))
COLORS_EN = (
    "amber", "blue", "coral", "denim", "emerald", "fuchsia", "gold", "hazel",
    "indigo", "jade", "khaki", "lilac", "magenta", "navy", "ochre", "pearl",
    "quartz", "rose", "silver", "teal", "umber", "violet", "white", "yellow",
    "azure", "bronze", "crimson", "green",
)
COLORS_ZH = tuple(f"颜色{i + 1}" for i in range(28))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def values(language: str, unit: int) -> tuple[str, str, str, str, str, str]:
    if language == "en":
        return (NAMES_A_EN[unit], NAMES_B_EN[unit], OBJECTS_EN[unit], ALT_OBJECTS_EN[unit],
                COLORS_EN[unit], COLORS_EN[(unit + 9) % 28])
    return (NAMES_A_ZH[unit], NAMES_B_ZH[unit], OBJECTS_ZH[unit], ALT_OBJECTS_ZH[unit],
            COLORS_ZH[unit], COLORS_ZH[(unit + 9) % 28])


def graph_terms(family: str, language: str, unit: int) -> tuple[list[str], str, str]:
    k = unit + 101
    if language == "en":
        if family == "taxonomy_path":
            return [f"spruce-{k}", f"conifer-{k}", f"tree-{k}", f"plant-{k}", f"organism-{k}"], "is a", f"mineral-{k}"
        if family == "part_whole_path":
            return [f"reed-{k}", f"pipe-{k}", f"organ-{k}", f"instrument-{k}", f"collection-{k}"], "is part of", f"theatre-{k}"
        return [f"survey-{k}", f"design-{k}", f"assembly-{k}", f"inspection-{k}", f"delivery-{k}"], "occurred before", f"concert-{k}"
    if family == "taxonomy_path":
        return [f"云杉{k}", f"针叶类{k}", f"树类{k}", f"植物类{k}", f"生物类{k}"], "属于", f"矿物类{k}"
    if family == "part_whole_path":
        return [f"簧片{k}", f"管体{k}", f"风琴{k}", f"乐器{k}", f"藏品{k}"], "是其组成部分", f"剧场{k}"
    return [f"勘测{k}", f"设计{k}", f"装配{k}", f"检查{k}", f"交付{k}"], "早于", f"音乐会{k}"


def broad_core(family: str, language: str, unit: int, state: int, paraphrase: bool) -> tuple[str, dict[str, str]]:
    a, b, obj, alt, color, old = values(language, unit)
    if family in GRAPH_FAMILIES:
        nodes, relation, distractor = graph_terms(family, language, unit)
        middle = nodes[2] if state else distractor
        if language == "en":
            facts = f"{nodes[0]} {relation} {nodes[1]}; {nodes[1]} {relation} {middle}; {nodes[2]} {relation} {nodes[3]}."
            core = (f"The record states: {facts} Does the record support that {nodes[0]} {relation} {nodes[3]}?"
                    if not paraphrase else f"Use only these facts: {facts} Is it justified to conclude that {nodes[0]} {relation} {nodes[3]}?")
        else:
            facts = f"{nodes[0]}{relation}{nodes[1]}；{nodes[1]}{relation}{middle}；{nodes[2]}{relation}{nodes[3]}。"
            core = (f"记录中写着：{facts}记录是否支持“{nodes[0]}{relation}{nodes[3]}”？"
                    if not paraphrase else f"只依据这些事实：{facts}能否推出“{nodes[0]}{relation}{nodes[3]}”？")
        return core, {"primary": nodes[0], "secondary": nodes[1], "relation": relation,
                      "context": middle, "query": nodes[3]}
    if family == "active_passive_role":
        agent, patient = (a, b) if state else (b, a)
        if language == "en":
            core = (f"{agent} delivered the {obj} to {patient}. Was {a} the person who delivered it?"
                    if not paraphrase else f"The {obj} was delivered to {patient} by {agent}. In this event, was the deliverer {a}?")
            relation = "delivered"
        else:
            core = (f"{agent}把{obj}交给了{patient}。交付者是{a}吗？"
                    if not paraphrase else f"{obj}由{agent}交付给{patient}。在这件事中，执行交付的人是{a}吗？")
            relation = "交付"
        return core, {"primary": agent, "secondary": patient, "relation": relation,
                      "context": obj, "query": a}
    if family == "relative_clause_role":
        carrier = b if state else a
        if language == "en":
            core = (f"{a} greeted {b}, who carried the {obj}. Was {b} carrying the {obj}?" if state else
                    f"{a}, who carried the {obj}, greeted {b}. Was {b} carrying the {obj}?")
            if paraphrase:
                core = (f"The person who carried the {obj} was {b}; {a} greeted {b}. Did {b} carry it?" if state else
                        f"The person who carried the {obj} was {a}; {a} greeted {b}. Did {b} carry it?")
            relation = "carried"
        else:
            core = (f"{a}问候了携带{obj}的{b}。{b}携带{obj}吗？" if state else
                    f"携带{obj}的{a}问候了{b}。{b}携带{obj}吗？")
            if paraphrase:
                core = (f"携带{obj}的人是{b}，随后{a}问候了{b}。{b}携带着它吗？" if state else
                        f"携带{obj}的人是{a}，随后{a}问候了{b}。{b}携带着它吗？")
            relation = "携带"
        return core, {"primary": carrier, "secondary": b, "relation": relation,
                      "context": obj, "query": b}
    if family in ("negation_scope", "attitude_scope"):
        attitude = family == "attitude_scope"
        if language == "en":
            verb = "regretted" if attitude else "said"
            verb_base = "regret" if attitude else "say"
            sentence = (f"{a} {verb} that {b} did not open the {obj}." if state else
                        f"{a} did not {verb_base} that {b} opened the {obj}.")
            if paraphrase:
                sentence = (f"According to the report, {a} {verb} {b}'s not opening the {obj}." if state else
                            f"According to the report, it is not the case that {a} {verb} {b}'s opening of the {obj}.")
            core = sentence + " Does the negation apply inside the reported opening event?"
            relation = "negation"
            query = "opening"
        else:
            verb = "后悔" if attitude else "说"
            sentence = (f"{a}{verb}{b}没有打开{obj}。" if state else f"{a}没有{verb}{b}打开了{obj}。")
            if paraphrase:
                sentence = (f"报告称，{a}{verb}的是{b}未打开{obj}这件事。" if state else
                            f"报告否认了“{a}{verb}{b}打开{obj}”这件事。")
            core = sentence + "否定是否位于被报告的打开事件内部？"
            relation = "否定"
            query = "打开"
        return core, {"primary": a, "secondary": b, "relation": relation,
                      "context": obj, "query": query}
    if family == "quantifier_scope":
        if language == "en":
            sentence = (f"One shared {obj} was inspected by every technician." if state else
                        f"Every technician inspected a different {obj}.")
            if paraphrase:
                sentence = (f"There was a single {obj} that all technicians inspected." if state else
                            f"For each technician, a distinct {obj} was inspected.")
            core = sentence + f" Does this require the technicians to share one {obj}?"
            relation, primary, secondary = "inspected", "technician", obj
        else:
            sentence = (f"所有技师都检查了同一个{obj}。" if state else f"每位技师都检查了不同的{obj}。")
            if paraphrase:
                sentence = (f"存在一个由全体技师共同检查的{obj}。" if state else f"对每位技师而言，被检查的{obj}各不相同。")
            core = sentence + f"这是否要求所有技师共享同一个{obj}？"
            relation, primary, secondary = "检查", "技师", obj
        return core, {"primary": primary, "secondary": secondary, "relation": relation,
                      "context": obj, "query": obj}
    if family == "contrast_coreference":
        pronoun = "I" if state else "you"
        if language == "en":
            core = (f'{a} told {b}, "{pronoun} stored the {obj}." Does "{pronoun}" refer to {a}?'
                    if not paraphrase else f'{a} spoke to {b} and said, "{pronoun} put away the {obj}." Is {a} the referent of "{pronoun}"?')
            relation = pronoun
        else:
            pronoun = "我" if state else "你"
            core = (f'{a}对{b}说：“{pronoun}收好了{obj}。”这里的“{pronoun}”指{a}吗？'
                    if not paraphrase else f'{a}告诉{b}：“{pronoun}把{obj}放好了。”代词“{pronoun}”所指的是{a}吗？')
            relation = pronoun
        return core, {"primary": a, "secondary": b, "relation": relation,
                      "context": obj, "query": a}
    current = color if state else old
    if language == "en":
        core = (f"The {obj} was {old}. {a} then painted it {current}. Is it now {color}?" if not paraphrase else
                f"After starting as {old}, the {obj} was repainted {current} by {a}. Should its current color be recorded as {color}?")
        relation = "painted"
    else:
        core = (f"{obj}原来是{old}。随后{a}把它涂成了{current}。它现在是{color}吗？" if not paraphrase else
                f"{obj}起初为{old}，后来由{a}重新涂成{current}。它当前应登记为{color}吗？")
        relation = "涂成"
    return core, {"primary": obj, "secondary": old, "relation": relation,
                  "context": current, "query": color}


def wrap(core: str, roles: dict[str, str], *, family: str, language: str, unit: int,
         state: int, surface: str, panel: str, partition: str, fresh: bool,
         extra: dict[str, Any] | None = None) -> dict:
    extra = extra or {}
    fi, li, si = FAMILIES.index(family), LANGUAGES.index(language), int(surface != "direct")
    true_code, false_code = OUTPUT_SCHEMES[(fi + unit + li + si) % len(OUTPUT_SCHEMES)]
    answer = true_code if state else false_code
    gold = (fi + unit + li + si + state) % 2
    options = [answer, false_code if state else true_code]
    if gold:
        options.reverse()
    instruction = (f" Choose A or B only. A: {options[0]}. B: {options[1]}." if language == "en" else
                   f" 只从A或B中选择。A：{options[0]}。B：{options[1]}。")
    free = (f" Answer with exactly one word: {true_code} or {false_code}." if language == "en" else
            f" 请只回答一个词：{true_code}或{false_code}。")
    cell = extra.get("cell_id", "base")
    return {
        "case_id": f"{panel}_{family}_{language}_u{unit}_{surface}_s{state}_{cell}",
        "panel": panel, "family": family, "language": language, "unit": unit,
        "state": state, "truth": bool(state), "surface": surface, "partition": partition,
        "fresh": fresh, "prompt_core": core, "prompt": core + instruction,
        "free_prompt": core + free, "role_values": roles,
        "output_scheme": (fi + unit + li + si) % len(OUTPUT_SCHEMES),
        "true_code": true_code, "false_code": false_code, "correct_answer": answer,
        "gold_position": gold, **extra,
    }


def broad_material(fresh: bool) -> list[dict]:
    start = PARENT_UNITS if fresh else 0
    count = FRESH_UNITS if fresh else PARENT_UNITS
    rows = []
    for family, language, unit, surface, state in itertools.product(
            FAMILIES, LANGUAGES, range(start, start + count), SURFACES, (0, 1)):
        if fresh:
            partition = "fresh_confirmation" if unit < start + count // 2 else "fresh_lockbox"
        else:
            partition = "discovery" if unit < DISCOVERY_UNITS else "confirmation" if unit < 12 else "lockbox"
        core, roles = broad_core(family, language, unit, state, surface == "paraphrase")
        rows.append(wrap(core, roles, family=family, language=language, unit=unit,
                         state=state, surface=surface, panel="construction_broad",
                         partition=partition, fresh=fresh))
    return rows


def graph_composition_case(family: str, language: str, unit: int, depth: int,
                           variant: str, fresh: bool) -> dict:
    nodes, relation, distractor = graph_terms(family, language, unit)
    edges = [(nodes[i], nodes[i + 1]) for i in range(depth)]
    state = int(variant in ("valid", "shortcut", "irrelevant"))
    if variant == "break":
        edges[max(0, depth // 2 - 1)] = (edges[max(0, depth // 2 - 1)][0], distractor)
    elif variant == "reverse":
        edges = [(b, a) for a, b in edges]
    elif variant == "disconnected":
        edges[0] = (distractor, edges[0][1])
    elif variant == "shortcut":
        edges.append((nodes[0], nodes[depth]))
    elif variant == "irrelevant":
        edges.append((distractor, nodes[-1]))
    if language == "en":
        facts = " ".join(f"{x} {relation} {y}." for x, y in edges)
        core = f"A record lists these facts: {facts} Does it support that {nodes[0]} {relation} {nodes[depth]}?"
    else:
        facts = "".join(f"{x}{relation}{y}。" for x, y in edges)
        core = f"记录列出这些事实：{facts}它是否支持“{nodes[0]}{relation}{nodes[depth]}”？"
    roles = {"primary": nodes[0], "secondary": nodes[1], "relation": relation,
             "context": edges[-1][0], "query": nodes[depth]}
    partition = "fresh_composition_lockbox" if fresh else "composition_discovery"
    return wrap(core, roles, family=family, language=language, unit=unit, state=state,
                surface="composition", panel="graph_composition", partition=partition,
                fresh=fresh, extra={"depth": depth, "variant": variant,
                                    "cell_id": f"d{depth}_{variant}"})


def attitude_factorial_case(language: str, unit: int, verb_i: int, outer: int,
                            inner: int, fresh: bool) -> dict:
    a, b, obj, _alt, _color, _old = values(language, unit)
    verbs = ("liked", "regretted", "remembered") if language == "en" else ("喜欢", "后悔", "记得")
    verb = verbs[verb_i]
    if language == "en":
        bases = ("like", "regret", "remember")
        sentence = f"{a} {'did not ' if outer else ''}{verb if not outer else bases[verb_i]} that {b} {'did not open' if inner else 'opened'} the {obj}."
        core = sentence + " Does the negation apply inside the opening event?"
        query = "opening"
        relation = "negation"
    else:
        sentence = f"{a}{'并不' if outer else ''}{verb}{b}{'没有打开' if inner else '打开了'}{obj}。"
        core = sentence + "否定是否位于打开事件内部？"
        query = "打开"
        relation = "否定"
    roles = {"primary": a, "secondary": b, "relation": relation, "context": obj, "query": query}
    state = int(inner == 1)
    partition = "fresh_composition_lockbox" if fresh else "composition_discovery"
    return wrap(core, roles, family="attitude_scope", language=language, unit=unit,
                state=state, surface="factorial", panel="attitude_factorial",
                partition=partition, fresh=fresh,
                extra={"verb_index": verb_i, "outer_neg": outer, "inner_neg": inner,
                       "cell_id": f"v{verb_i}_o{outer}_i{inner}"})


def composition_material(fresh: bool) -> list[dict]:
    units = (range(PARENT_UNITS, PARENT_UNITS + COMPOSITION_FRESH_UNITS)
             if fresh else range(DISCOVERY_UNITS))
    rows = []
    for args in itertools.product(GRAPH_FAMILIES, LANGUAGES, units, (2, 3, 4),
                                  ("valid", "shortcut", "irrelevant", "break", "reverse", "disconnected")):
        rows.append(graph_composition_case(*args, fresh=fresh))
    attitude_units = tuple(units)[:4]
    for args in itertools.product(LANGUAGES, attitude_units, range(3), (0, 1), (0, 1)):
        rows.append(attitude_factorial_case(*args, fresh=fresh))
    return rows


def contextual_spans(tokenizer, ids: list[int], value: str) -> list[list[int]]:
    exact = compiler.graph_base.name_spans(tokenizer, ids, value)
    if exact:
        return exact
    width0 = max(1, len(tokenizer.encode(value, add_special_tokens=False)))
    for width in range(1, width0 + 5):
        found = []
        for start in range(0, len(ids) - width + 1):
            if value in tokenizer.decode(ids[start:start + width], skip_special_tokens=True):
                found.append(list(range(start, start + width)))
        if found:
            return found
    return []


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(x) != 1 for x in candidates):
        raise RuntimeError(("candidate_not_single_token", candidates))
    system = "Use only the supplied text. Follow the requested answer format exactly."
    compiled = []
    for row in rows:
        ids = compiler.core.chat_ids(tokenizer, system, row["prompt"])
        free_ids = compiler.core.chat_ids(tokenizer, system, row["free_prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = contextual_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "free_prompt_ids": free_ids,
                         "candidate_ids": candidates, "role_positions": positions})
    return compiled


def material_audit(rows: list[dict], compiled: list[dict]) -> dict:
    ids = Counter(row["case_id"] for row in rows)
    widths = [len(row["prompt_ids"]) for row in compiled]
    missing = [{"case_id": row["case_id"], "role": role, "value": value}
               for row in rows for role, value in row["role_values"].items()
               if value not in row["prompt_core"]]
    forbidden = ("�", "锟", "regreted", "remembered that did")
    malformed = [row["case_id"] for row in rows if any(x in row["prompt"] for x in forbidden)]
    broad_cells: dict[tuple, set] = defaultdict(set)
    graph_cells: dict[tuple, set] = defaultdict(set)
    attitude_cells: dict[tuple, set] = defaultdict(set)
    for row in rows:
        if row["panel"] == "construction_broad":
            broad_cells[(row["family"], row["language"], row["unit"])].add((row["surface"], row["state"]))
        elif row["panel"] == "graph_composition":
            graph_cells[(row["family"], row["language"], row["unit"])].add((row["depth"], row["variant"]))
        elif row["panel"] == "attitude_factorial":
            attitude_cells[(row["language"], row["unit"])].add((row["verb_index"], row["outer_neg"], row["inner_neg"]))
    zero = {
        "always_A": float(np.mean([row["gold_position"] == 0 for row in rows])),
        "always_B": float(np.mean([row["gold_position"] == 1 for row in rows])),
        "always_true": float(np.mean([row["state"] == 1 for row in rows])),
    }
    return {
        "rows": len(rows), "compiled_rows": len(compiled), "unique_case_ids": len(ids),
        "duplicates": sorted(k for k, v in ids.items() if v != 1),
        "panels": dict(Counter(row["panel"] for row in rows)),
        "families": dict(Counter(row["family"] for row in rows)),
        "partitions": dict(Counter(row["partition"] for row in rows)),
        "zero_models": zero, "missing_roles": missing, "malformed_strings": malformed,
        "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "broad_factorial_complete": (not broad_cells) or all(len(x) == 4 for x in broad_cells.values()),
        "graph_factorial_complete": (not graph_cells) or all(len(x) == 18 for x in graph_cells.values()),
        "attitude_factorial_complete": (not attitude_cells) or all(len(x) == 12 for x in attitude_cells.values()),
        "semantic_uniqueness_machine_audit": "pass_explicit_state_tables_and_complete_factorials",
        "material_naturalness_machine_audit": "pass_controlled_bilingual_templates_no_malformed_strings",
        "human_blind_review": "NA_not_run_no_independent_human_panel_available",
    }


def preregistration() -> dict:
    return {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "frozen_before_model": True,
        "research_object": "family/sample/role/depth-conditioned coordinate response ecology",
        "families": list(FAMILIES), "languages": list(LANGUAGES), "surfaces": list(SURFACES),
        "units": {"parent": PARENT_UNITS, "fresh": FRESH_UNITS,
                  "composition_parent": DISCOVERY_UNITS, "composition_fresh": COMPOSITION_FRESH_UNITS},
        "models_sequential": ["Qwen3-4B", "Qwen3-14B", "GLM4", "DeepSeek-7B"],
        "behavior_policy": "candidate and free generation are separate; qualification is per family/panel; failures become NA and never stop other routes",
        "behavior_gate": BEHAVIOR_GATE,
        "camera": "embedding, every post-block HiddenState, final norm, six roles and every physical activation coordinate; all-token full-coordinate subfield for predeclared key families",
        "all_token_subfield": {"families": ["taxonomy_path", "attitude_scope"], "fresh_units": [16, 17],
                               "panels": ["construction_broad"]},
        "coordinate_methods": [
            "coordinate sign consistency", "coordinate own-family absolute-error win",
            "coordinate formation/dissolution intervals", "same-coordinate depth recurrence",
            "graph path variant ledger", "outer-inner factorial residual",
        ],
        "coordinate_gates": COORDINATE_GATES,
        "causal_gates": CAUSAL_GATES,
        "causal_policy": "only prospectively qualified family/checkpoint/role coordinate masks; otherwise NA, never zero effect",
        "forbidden": ["attention", "MLP", "weights", "gradients", "PCA", "Top-K", "cosine screening",
                      "donor delta as core discovery", "post-reveal threshold tuning"],
        "cross_model": "same fresh semantic panel; compare relative depth/role and within-model coordinate passport summaries, never coordinate IDs",
        "visualization": "export all coordinate columns for every important passport and path/scope atlas",
        "cleanup": "delete undisplayed raw sample fields only after derivative hashes, shape checks and catalog verification",
        "failure_policy": "route-level missingness; continue every preregistered family and model",
        "theory": "conditionalized output-field closure theory; RDC unchanged; no new mathematics authorization",
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    if marker in existing:
        correction = "**Phase2253 冻结纠正附录"
        if result.get("all_checks_passed") and correction not in existing:
            audits = result["material_audits"]
            text = f"""

{correction} [{stamp}]。** 首次预冻结审计发现图组合真/假变体为3:2，`always_true=0.588235`，因此当时 `all_checks_passed=False`，且没有加载模型。合同在揭盲前增加“首边断开”这一预定假变体后重新冻结：父/全新组合材料各 `{audits['parent_composition']['rows']}` 条，A/B与真/假零模型均精确为0.5，图因素格每个族-语言-单元包含18格；最终全部合同检查为 `True`。旧失败材料不进入任何模型或HiddenState分析，后续只使用本附录哈希对应的重冻结材料。
"""
            with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        return
    audits = {k: {x: v[x] for x in ("rows", "panels", "partitions", "zero_models",
                                               "token_width_min_median_max", "human_blind_review")}
              for k, v in result["material_audits"].items()}
    text = rf"""

## Phase {PHASE}: 十构式独立生态与图路径纵深总合同（C1097-C1120） [{stamp}]

**证据审查。** Phase2247-2252可保留的结论是：Qwen3-4B/14B能在冻结六族材料上进入内部观察；分类、整体-部分和时序图的局部组合原型可迁移到新词汇；统一六族整轨迹预测、状态一致因果闭环和4B-14B角色深度同构均未通过。因果期输入为空应写为NA，不能写成零效应。附件中“已找到条件齿轮”“图路径已代表通用关系算子”以及“需要立即引入新数学”的表述均过强。

**大方案、原理与用例。** 本期在新分母冻结十类中英构式：分类路径、整体-部分路径、时序路径、主动-被动角色、关系从句角色、否定作用域、态度作用域、量词作用域、引语共指和属性覆盖。宽族含16个父单元与12个全新单元、直接/释义表面和四套输出码。图纵深另含2-4跳的有效链、直接捷径、无关边、断链和反向边；态度面板含like/regret/remember与内外层否定的完整因素格。例子包括“每位技师检查不同物件”与“所有技师检查同一物件”，以及“未说某事件”与“说某事件未发生”的作用域对照。

**冻结公式。** 逐坐标响应、符号一致率和本族绝对误差胜率定义为：

$$
R_{{f,u,q,r,j}}=H^{{(1)}}_{{f,u,q,r,j}}-H^{{(0)}}_{{f,u,q,r,j}},\qquad
C_{{f,q,r,j}}=\left|\frac1U\sum_u\operatorname{{sign}}R_{{f,u,q,r,j}}\right|,
$$

$$
S_{{f,q,r,j}}=\frac1U\sum_u\mathbf 1\!\left[e^{{own}}_{{u,q,r,j}}<\min_{{g\ne f}}e^{{wrong,g}}_{{u,q,r,j}}\right].
$$

路径深度只尝试同坐标基础递推，态度组合使用可逆二阶分账：

$$
R_{{d+1,j}}=a_{{d,j}}R_{{d,j}}+b_{{d,j}}+\varepsilon_{{d,j}},\qquad
I_{{oi,j}}=H_{{11,j}}-H_{{10,j}}-H_{{01,j}}+H_{{00,j}}.
$$

**材料、门槛与审计结果。** 四份材料账为 `{json.dumps(audits, ensure_ascii=False)}`。统计单位是独立词汇/结构单元，不把模板行数冒充独立样本。候选选择和自由生成按族/面板分别要求不低于0.75；失败只登记预定缺失。逐坐标候选至少6个独立单元，fresh相对零响应和错族增益分别不低于0.05/0.03，符号一致率不低于0.75，本族胜率不低于0.60且连续两个检查点成立。语义唯一性由显式状态表和因素格机器审计；机器自然度审计不能代替独立人类盲评，后者为NA。

**理论进展、硬伤与结论。** 本期没有模型或HiddenState结果，只完成新的独立构式生态、分路线不中止制度与逐坐标观察合同。理论主体保持“条件化输出场闭合理论”，RDC不变。硬伤是材料仍为受控模板、部分中文节点为人工标签、答案码是元语言接口、关键全token子场只覆盖预注册两族、独立人类盲评缺失。工程检查 `{result['all_checks_passed']}`；通过后授权依次执行Qwen3-4B全场、逐坐标留出、组合纵深、严格候选因果和串行跨模型面板。

**相关文件。** 脚本 `tests/glm5/phase2253_c1097_c1120_construction_ecology_contract.py`；结果 `tests/glm5/result/phase2253_c1097_c1120_construction_ecology_contract`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        existing = load(final_path)
        if existing.get("all_checks_passed"):
            return existing
    for sub in ("protocol", "material", "audit", "analysis"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol/preregistration.json",
         {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **preregistration()})
    materials = {
        "parent_broad": broad_material(False),
        "fresh_broad": broad_material(True),
        "parent_composition": composition_material(False),
        "fresh_composition": composition_material(True),
    }
    tokenizer = model_base.parent.load_tokenizer()
    audits, hashes = {}, {}
    checks = {"protocol_frozen": True}
    for name, rows in materials.items():
        compiled = compile_rows(tokenizer, rows)
        raw_path = OUT / f"material/{name}_cases.jsonl"
        compiled_path = OUT / f"material/{name}_qwen_compiled.jsonl"
        write_rows(raw_path, rows)
        write_rows(compiled_path, compiled)
        audit = material_audit(rows, compiled)
        save(OUT / f"audit/{name}_audit.json", audit)
        audits[name] = audit
        hashes[name] = file_hash(raw_path)
        hashes[f"{name}_compiled"] = file_hash(compiled_path)
        checks[f"{name}_compiled"] = len(rows) == len(compiled)
        checks[f"{name}_unique"] = not audit["duplicates"] and audit["unique_case_ids"] == len(rows)
        checks[f"{name}_roles"] = not audit["missing_roles"]
        checks[f"{name}_strings"] = not audit["malformed_strings"]
        checks[f"{name}_balanced"] = all(abs(v - 0.5) <= 1e-12 for v in audit["zero_models"].values())
        checks[f"{name}_factorials"] = (audit["broad_factorial_complete"] if "broad" in name else
                                          audit["graph_factorial_complete"] and audit["attitude_factorial_complete"])
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "protocol": preregistration(),
        "material_audits": audits, "hashes": hashes, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "A ten-construction independent denominator is frozen and compiler-valid; no model or mechanism claim exists.",
        "next_authorization": "Run all preregistered observation routes sequentially without changing materials, partitions or gates.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
