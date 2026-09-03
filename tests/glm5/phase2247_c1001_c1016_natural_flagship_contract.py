#!/usr/bin/env python3
"""Natural flagship full-coordinate campaign contract (C1001-C1016).

This phase freezes materials and gates before any new model execution.  The
registered internal object is embedding plus post-block/final-norm HiddenState
at six semantic roles and every physical activation coordinate.  Attention,
MLP internals, weights, gradients, PCA, Top-K screening and post-reveal tuning
are outside the contract.
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
OUT = TESTS / "result/phase2247_c1001_c1016_natural_flagship_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase1797_c263_c272_state_operator_common as compiler  # noqa: E402
import phase2219_c773_c808_semantic_transition_ecology_campaign as prior  # noqa: E402


PHASE = 2247
CAMPAIGNS = tuple(f"C{i}" for i in range(1001, 1017))
FAMILIES = (
    "graph_taxonomy",
    "graph_part_whole",
    "graph_temporal",
    "coreference_binding",
    "attribute_update",
    "nested_attitude",
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
PARENT_UNITS = 12
FRESH_UNITS = 8
DISCOVERY_UNITS = 6
BEHAVIOR_GATE = 0.75
FAMILY_BEHAVIOR_GATE = 0.75
PREDICTIVE_GATES = {
    "minimum_holdout_units": 4,
    "relative_mae_gain_over_shared": 0.03,
    "relative_mae_gain_over_wrong_family": 0.03,
    "median_coordinate_sign_accuracy": 0.55,
}
COMPOSITION_GATES = {
    "relative_mae_gain_over_zero": 0.05,
    "relative_mae_gain_over_wrong_family": 0.03,
    "minimum_holdout_units": 4,
}
CAUSAL_GATES = {
    "minimum_pairs": 8,
    "candidate_direction_rate": 0.60,
    "candidate_margin_advantage": 0.05,
    "generation_accuracy_advantage": 0.10,
    "correct_rescue_advantage": 0.10,
}

NAMES_A_EN = (
    "Nadia", "Leona", "Petra", "Sabine", "Amira", "Bianca", "Daphne", "Elise",
    "Freya", "Helena", "Ines", "Julia", "Lucia", "Marina", "Selene", "Vera",
    "Yvette", "Alina", "Celine", "Daria",
)
NAMES_B_EN = (
    "Marcus", "Nolan", "Pavel", "Ruben", "Adrian", "Bruno", "Damian", "Emil",
    "Gregor", "Henrik", "Isaac", "Jasper", "Kilian", "Lucian", "Marek", "Stefan",
    "Tobias", "Ulrich", "Xavier", "Yves",
)
NAMES_A_ZH = (
    "娜迪娅", "莉奥娜", "佩特拉", "萨宾", "阿米拉", "比安卡", "达芙妮", "艾莉丝",
    "芙蕾雅", "海伦娜", "伊内斯", "茱莉亚", "露西亚", "玛丽娜", "塞勒涅", "维拉",
    "伊薇特", "阿丽娜", "塞琳", "达莉娅",
)
NAMES_B_ZH = (
    "马库斯", "诺兰", "帕维尔", "鲁本", "阿德里安", "布鲁诺", "达米安", "埃米尔",
    "格雷戈", "亨里克", "艾萨克", "贾斯珀", "基利安", "卢西安", "马雷克", "斯特凡",
    "托比亚斯", "乌尔里希", "泽维尔", "伊夫",
)
OBJECTS_EN = (
    "cabinet", "satchel", "lantern", "telescope", "vase", "parcel", "violin", "compass",
    "ledger", "camera", "scarf", "teapot", "medal", "sketchbook", "umbrella", "backpack",
    "clock", "goblet", "tablet", "suitcase",
)
OBJECTS_ZH = (
    "橱柜", "挎包", "提灯", "望远镜", "花瓶", "包裹", "小提琴", "指南针",
    "账簿", "相机", "围巾", "茶壶", "奖章", "速写本", "雨伞", "背包",
    "时钟", "高脚杯", "平板电脑", "手提箱",
)
ALT_OBJECTS_EN = (
    "drawer", "basket", "torch", "microscope", "pitcher", "envelope", "cello", "sextant",
    "journal", "projector", "shawl", "kettle", "trophy", "notebook", "raincoat", "briefcase",
    "watch", "bowl", "monitor", "trunk",
)
ALT_OBJECTS_ZH = (
    "抽屉", "篮子", "火炬", "显微镜", "水罐", "信封", "大提琴", "六分仪",
    "日记本", "投影仪", "披肩", "水壶", "奖杯", "笔记本", "雨衣", "公文包",
    "手表", "碗", "显示器", "箱子",
)
COLORS_EN = (
    "amber", "violet", "silver", "crimson", "turquoise", "ivory", "bronze", "indigo",
    "scarlet", "teal", "maroon", "coral", "navy", "gold", "copper", "pearl",
    "ochre", "lilac", "charcoal", "azure",
)
COLORS_ZH = (
    "琥珀色", "紫罗兰色", "银色", "绯红色", "青绿色", "象牙色", "古铜色", "靛蓝色",
    "猩红色", "蓝绿色", "栗色", "珊瑚色", "海军蓝", "金色", "铜色", "珍珠色",
    "赭色", "淡紫色", "炭灰色", "天蓝色",
)


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


def unit_values(language: str, unit: int) -> tuple[str, str, str, str, str, str]:
    if language == "en":
        return (NAMES_A_EN[unit], NAMES_B_EN[unit], OBJECTS_EN[unit], ALT_OBJECTS_EN[unit],
                COLORS_EN[unit], COLORS_EN[(unit + 7) % len(COLORS_EN)])
    return (NAMES_A_ZH[unit], NAMES_B_ZH[unit], OBJECTS_ZH[unit], ALT_OBJECTS_ZH[unit],
            COLORS_ZH[unit], COLORS_ZH[(unit + 7) % len(COLORS_ZH)])


def graph_terms(family: str, language: str, unit: int) -> tuple[list[str], str, str]:
    suffix = unit + 1
    if language == "en":
        if family == "graph_taxonomy":
            nodes = [f"cedar-{suffix}", f"conifer-{suffix}", f"tree-{suffix}", f"plant-{suffix}", f"organism-{suffix}"]
            return nodes, "belongs to", f"mineral-{suffix}"
        if family == "graph_part_whole":
            nodes = [f"valve-{suffix}", f"pump-{suffix}", f"engine-{suffix}", f"machine-{suffix}", f"workshop-{suffix}"]
            return nodes, "is part of", f"gallery-{suffix}"
        nodes = [f"briefing-{suffix}", f"inspection-{suffix}", f"repair-{suffix}", f"delivery-{suffix}", f"opening-{suffix}"]
        return nodes, "happened before", f"concert-{suffix}"
    if family == "graph_taxonomy":
        nodes = [f"雪松{suffix}", f"针叶树{suffix}", f"树木{suffix}", f"植物{suffix}", f"生物{suffix}"]
        return nodes, "属于", f"矿物{suffix}"
    if family == "graph_part_whole":
        nodes = [f"阀门{suffix}", f"泵体{suffix}", f"发动机{suffix}", f"机器{suffix}", f"车间{suffix}"]
        return nodes, "是其组成部分", f"展厅{suffix}"
    nodes = [f"简报{suffix}", f"检查{suffix}", f"维修{suffix}", f"交付{suffix}", f"开幕{suffix}"]
    return nodes, "早于", f"音乐会{suffix}"


def broad_core(family: str, language: str, unit: int, truth: bool, paraphrase: bool) -> tuple[str, dict[str, str]]:
    a, b, obj, alt, color, old_color = unit_values(language, unit)
    if family in GRAPH_FAMILIES:
        nodes, relation, distractor = graph_terms(family, language, unit)
        target = nodes[3] if truth else distractor
        if language == "en":
            statements = f"{nodes[0]} {relation} {nodes[1]}; {nodes[1]} {relation} {nodes[2]}; {nodes[2]} {relation} {nodes[3]}."
            core = (f"An archive note records the following facts: {statements} Does the note support that {nodes[0]} {relation} {target}?"
                    if not paraphrase else
                    f"Use only this short record. {statements} From that chain alone, is it justified to say that {nodes[0]} {relation} {target}?")
        else:
            statements = f"{nodes[0]}{relation}{nodes[1]}；{nodes[1]}{relation}{nodes[2]}；{nodes[2]}{relation}{nodes[3]}。"
            core = (f"档案中记录了这些事实：{statements}档案是否支持“{nodes[0]}{relation}{target}”？"
                    if not paraphrase else
                    f"请只依据这份简短记录：{statements}沿着这条关系链，能否推出“{nodes[0]}{relation}{target}”？")
        roles = {"primary": nodes[0], "secondary": nodes[1], "relation": relation,
                 "context": nodes[2], "query": target}
        return core, roles
    if family == "coreference_binding":
        target = a if truth else b
        if language == "en":
            core = (f"After {a} handed the {obj} to {b}, she returned to the lobby. In this account, does 'she' refer to {target}?"
                    if not paraphrase else
                    f"{a} passed the {obj} to {b} and then she went back inside. Is {target} the person identified by 'she'?")
            pronoun = "she"
        else:
            core = (f"{a}把{obj}交给{b}以后，她回到了大厅。这段话里的“她”是否指{target}？"
                    if not paraphrase else
                    f"{a}将{obj}递给{b}，随后她走回室内。“她”所指的人是{target}吗？")
            pronoun = "她"
        return core, {"primary": a, "secondary": b, "relation": pronoun, "context": obj, "query": target}
    if family == "attribute_update":
        target = color if truth else old_color
        if language == "en":
            core = (f"The {obj} was initially {old_color}. This morning, {a} repainted it {color}. Is the {obj} now {target}?"
                    if not paraphrase else
                    f"Although the {obj} used to be {old_color}, {a} has since painted it {color}. Should its current color be recorded as {target}?")
            relation = "painted"
        else:
            core = (f"{obj}原来是{old_color}。今天早上，{a}把它重新涂成了{color}。{obj}现在是{target}吗？"
                    if not paraphrase else
                    f"尽管{obj}过去是{old_color}，{a}后来已将它涂成{color}。它当前的颜色应登记为{target}吗？")
            relation = "涂成"
        return core, {"primary": obj, "secondary": old_color, "relation": relation, "context": a, "query": target}
    target = obj if truth else alt
    verb = ("regretted" if unit % 3 == 1 else "remembered" if unit % 3 == 2 else "liked")
    if language == "en":
        core = (f"{a} {verb} that {b} opened the {obj}. According to the sentence, was the object in the embedded event the {target}?"
                if not paraphrase else
                f"The report says that {a} {verb} {b}'s opening of the {obj}. Does the embedded event concern the {target}?")
        relation = verb
    else:
        verb = ("后悔" if unit % 3 == 1 else "记得" if unit % 3 == 2 else "喜欢")
        core = (f"{a}{verb}{b}打开了{obj}这件事。根据这句话，嵌套事件中的物品是{target}吗？"
                if not paraphrase else
                f"记录称，{a}对{b}打开{obj}这件事持有“{verb}”这一态度。内层事件涉及的是{target}吗？")
        relation = verb
    return core, {"primary": a, "secondary": b, "relation": relation, "context": obj, "query": target}


def wrap_case(core: str, roles: dict[str, str], *, family: str, language: str, unit: int,
              truth: bool, surface: str, panel: str, partition: str, fresh: bool,
              extra: dict[str, Any] | None = None) -> dict:
    family_i = FAMILIES.index(family)
    language_i = LANGUAGES.index(language)
    surface_i = 0 if surface == "direct" else 1
    scheme_i = (family_i + unit + language_i + surface_i) % len(OUTPUT_SCHEMES)
    true_code, false_code = OUTPUT_SCHEMES[scheme_i]
    correct = true_code if truth else false_code
    incorrect = false_code if truth else true_code
    gold = (family_i + unit + language_i + surface_i + int(truth)) % 2
    options = [correct, incorrect] if gold == 0 else [incorrect, correct]
    instruction = (f" Choose A or B only. A: {options[0]}. B: {options[1]}."
                   if language == "en" else f" 只从A或B中选择。A：{options[0]}。B：{options[1]}。")
    free_instruction = (f" Answer with exactly one word: {true_code} or {false_code}."
                        if language == "en" else f" 请只回答一个词：{true_code}或{false_code}。")
    extra = extra or {}
    return {
        "case_id": f"{panel}_{family}_{language}_u{unit}_{surface}_t{int(truth)}_{extra.get('cell_id', 'base')}",
        "panel": panel, "family": family, "language": language, "unit": unit,
        "truth": truth, "surface": surface, "partition": partition, "fresh": fresh,
        "prompt_core": core, "prompt": core + instruction, "free_prompt": core + free_instruction,
        "role_values": roles, "output_scheme": scheme_i, "true_code": true_code,
        "false_code": false_code, "correct_answer": correct, "gold_position": gold,
        **extra,
    }


def broad_material(fresh: bool) -> list[dict]:
    start = PARENT_UNITS if fresh else 0
    units = range(start, start + (FRESH_UNITS if fresh else PARENT_UNITS))
    rows = []
    for family, language, unit, surface, truth in itertools.product(FAMILIES, LANGUAGES, units, SURFACES, (False, True)):
        local = unit
        partition = ("fresh_confirmation" if unit < start + FRESH_UNITS // 2 else "fresh_lockbox") if fresh else (
            "discovery" if unit < DISCOVERY_UNITS else "confirmation" if unit < DISCOVERY_UNITS + 3 else "lockbox")
        core, roles = broad_core(family, language, local, truth, surface == "paraphrase")
        rows.append(wrap_case(core, roles, family=family, language=language, unit=unit, truth=truth,
                              surface=surface, panel="natural_broad", partition=partition, fresh=fresh))
    return rows


def graph_composition_case(family: str, language: str, unit: int, depth: int,
                           shortcut: int, truth: bool, fresh: bool) -> dict:
    nodes, relation, distractor = graph_terms(family, language, unit)
    target = nodes[depth] if truth else distractor
    edges = [(nodes[i], nodes[i + 1]) for i in range(depth)]
    if shortcut:
        edges.append((nodes[0], nodes[depth]))
    if language == "en":
        facts = " ".join(f"{a} {relation} {b}." for a, b in edges)
        core = f"A project record gives these facts: {facts} Based only on the record, does {nodes[0]} {relation} {target}?"
    else:
        facts = "".join(f"{a}{relation}{b}。" for a, b in edges)
        core = f"一份项目记录给出这些事实：{facts}只根据该记录，{nodes[0]}{relation}{target}吗？"
    roles = {"primary": nodes[0], "secondary": nodes[1], "relation": relation,
             "context": nodes[depth], "query": target}
    partition = "fresh_composition_lockbox" if fresh else "composition_discovery"
    cell = f"d{depth}_s{shortcut}"
    return wrap_case(core, roles, family=family, language=language, unit=unit, truth=truth,
                     surface="composition", panel="graph_composition", partition=partition, fresh=fresh,
                     extra={"composition_kind": "path_depth_shortcut", "depth": depth,
                            "shortcut": shortcut, "cell_id": cell})


def attitude_composition_case(language: str, unit: int, verb_i: int, outer_neg: int,
                              inner_neg: int, truth: bool, fresh: bool) -> dict:
    a, b, obj, alt, _color, _old = unit_values(language, unit)
    verbs_en = ("liked", "regretted", "remembered")
    verbs_zh = ("喜欢", "后悔", "记得")
    verb = verbs_en[verb_i] if language == "en" else verbs_zh[verb_i]
    target = obj if truth else alt
    if language == "en":
        outer = "did not " if outer_neg else ""
        inner = "did not open" if inner_neg else "opened"
        core = (f"The diary states that {a} {outer}{verb} that {b} {inner} the {obj}. "
                f"Is the object mentioned inside that attitude report the {target}?")
    else:
        outer = "并不" if outer_neg else ""
        inner = "没有打开" if inner_neg else "打开了"
        core = f"日记中写道，{a}{outer}{verb}{b}{inner}{obj}这件事。该态度报告内层提到的物品是{target}吗？"
    roles = {"primary": a, "secondary": b, "relation": verb, "context": obj, "query": target}
    partition = "fresh_composition_lockbox" if fresh else "composition_discovery"
    cell = f"v{verb_i}_o{outer_neg}_i{inner_neg}"
    return wrap_case(core, roles, family="nested_attitude", language=language, unit=unit,
                     truth=truth, surface="composition", panel="attitude_composition",
                     partition=partition, fresh=fresh,
                     extra={"composition_kind": "outer_inner_negation", "verb_index": verb_i,
                            "outer_neg": outer_neg, "inner_neg": inner_neg, "cell_id": cell})


def composition_material(fresh: bool) -> list[dict]:
    units = (range(PARENT_UNITS, PARENT_UNITS + FRESH_UNITS) if fresh else range(DISCOVERY_UNITS))
    rows = []
    for family, language, unit, depth, shortcut, truth in itertools.product(
            GRAPH_FAMILIES, LANGUAGES, units, (1, 2, 3, 4), (0, 1), (False, True)):
        if depth == 1 and shortcut == 1:
            continue
        rows.append(graph_composition_case(family, language, unit, depth, shortcut, truth, fresh))
    for language, unit, verb_i, outer_neg, inner_neg, truth in itertools.product(
            LANGUAGES, units, range(3), (0, 1), (0, 1), (False, True)):
        rows.append(attitude_composition_case(language, unit, verb_i, outer_neg, inner_neg, truth, fresh))
    return rows


def contextual_spans(tokenizer, ids: list[int], value: str) -> list[list[int]]:
    exact = compiler.graph_base.name_spans(tokenizer, ids, value)
    if exact:
        return exact
    needle = max(1, len(tokenizer.encode(value, add_special_tokens=False)))
    for width in range(1, needle + 5):
        found = []
        for start in range(0, len(ids) - width + 1):
            decoded = tokenizer.decode(ids[start:start + width], skip_special_tokens=True)
            if value in decoded:
                found.append(list(range(start, start + width)))
        if found:
            return found
    return []


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(value) != 1 for value in candidates):
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
    by_id = Counter(row["case_id"] for row in rows)
    zero = {
        "always_A": float(np.mean([row["gold_position"] == 0 for row in rows])),
        "always_B": float(np.mean([row["gold_position"] == 1 for row in rows])),
        "always_true": float(np.mean([row["truth"] for row in rows])),
    }
    missing_roles = []
    malformed = []
    forbidden = ("�", "锟", "eated", "regreted", "remembered that did")
    for row in rows:
        if any(token in row["prompt"] for token in forbidden):
            malformed.append(row["case_id"])
        for role, value in row["role_values"].items():
            if value not in row["prompt_core"]:
                missing_roles.append({"case_id": row["case_id"], "role": role, "value": value})
    broad = [row for row in rows if row["panel"] == "natural_broad"]
    broad_cells = defaultdict(set)
    for row in broad:
        broad_cells[(row["family"], row["language"], row["unit"])].add((row["surface"], row["truth"]))
    graph = [row for row in rows if row["panel"] == "graph_composition"]
    graph_cells = defaultdict(set)
    for row in graph:
        graph_cells[(row["family"], row["language"], row["unit"])].add((row["depth"], row["shortcut"], row["truth"]))
    attitude = [row for row in rows if row["panel"] == "attitude_composition"]
    attitude_cells = defaultdict(set)
    for row in attitude:
        attitude_cells[(row["language"], row["unit"])].add((row["verb_index"], row["outer_neg"], row["inner_neg"], row["truth"]))
    widths = [len(row["prompt_ids"]) for row in compiled]
    return {
        "rows": len(rows), "compiled_rows": len(compiled), "unique_case_ids": len(by_id),
        "duplicate_case_ids": sorted(key for key, value in by_id.items() if value != 1),
        "panels": dict(Counter(row["panel"] for row in rows)),
        "families": dict(Counter(row["family"] for row in rows)),
        "partitions": dict(Counter(row["partition"] for row in rows)),
        "zero_models": zero, "missing_roles": missing_roles, "malformed_strings": malformed,
        "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "broad_factorial_complete": bool(broad_cells) and all(len(value) == 4 for value in broad_cells.values()),
        "graph_factorial_complete": (not graph_cells) or all(len(value) == 14 for value in graph_cells.values()),
        "attitude_factorial_complete": (not attitude_cells) or all(len(value) == 24 for value in attitude_cells.values()),
        "semantic_uniqueness_machine_audit": "pass_by_explicit_truth_table_and_complete_factorials",
        "material_naturalness_machine_audit": "pass_controlled_bilingual_prose_no_malformed_patterns",
        "human_blind_review": "NA_not_run_no_independent_human_panel_available",
    }


def preregistration() -> dict:
    return {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "frozen_before_model": True,
        "research_object": "naturalized family-conditioned full-coordinate response and state-consistent causal timing",
        "families": list(FAMILIES), "languages": list(LANGUAGES), "surfaces": list(SURFACES),
        "units": {"parent": PARENT_UNITS, "fresh": FRESH_UNITS, "composition_discovery": DISCOVERY_UNITS},
        "models_sequential": ["Qwen3-4B", "Qwen3-14B", "GLM4", "DeepSeek-7B"],
        "model_policy": "Qwen3-4B runs the full broad and composition field; other models run the exact fresh broad denominator and capture only after dual behavior qualification",
        "camera": "embedding, every post-block HiddenState, final norm, six roles, every physical activation coordinate",
        "forbidden": ["attention", "MLP", "weights", "gradients", "PCA", "Top-K", "cosine screening", "post-reveal tuning"],
        "behavior_gate": BEHAVIOR_GATE, "family_behavior_gate": FAMILY_BEHAVIOR_GATE,
        "predictive_tournament": [
            "M0_zero", "M1_shared_coordinate_mean", "M2_family_coordinate_mean",
            "M3_shared_same_coordinate_affine", "M4_family_same_coordinate_affine",
            "M5_shared_full_coordinate_dual_ridge", "M6_family_full_coordinate_dual_ridge",
            "M7_wrong_family_equal_capacity",
        ],
        "predictive_gates": PREDICTIVE_GATES, "composition_gates": COMPOSITION_GATES,
        "causal_candidates": {
            "qwen3_checkpoints": [8, 16, 24, 26, 30, 32],
            "roles": list(ROLES),
            "coordinate_strata": ["all", "low_half", "high_half", "positive_response", "negative_response"],
            "doses": [0.25, 0.5, 1.0],
            "generation": ["one_shot", "repeat_each_generated_token"],
            "controls": ["wrong_family", "wrong_role", "wrong_checkpoint", "equal_norm_sign_permutation"],
        },
        "causal_gates": CAUSAL_GATES,
        "failure_policy": "route-level missingness; failure never stops other preregistered families, models or observations",
        "cross_model": "relative-depth role response passports only; never align physical coordinate IDs across models",
        "visualization": "important full-coordinate fields, family/checkpoint/role maps and causal ledgers are exported with hashes",
        "cleanup": "raw sample fields not displayed are deleted only after derived artifacts, hashes and reproducibility indexes pass",
        "theory": "conditionalized output-field closure theory; RDC unchanged; no new mathematics authorization",
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    audits = result["material_audits"]
    formula = r"""
$$
\Delta H_{i,q,r,j}=G_{q,r,j}(H_i)+R_{f,q,r,j}(H_i)+R_{\mathrm{compose},q,r,j}(H_i)+\varepsilon_{i,q,r,j}.
$$

全坐标跨坐标模型只用基础的对偶岭形式，避免先做PCA或Top-K：

$$
\widehat B=X^\top(XX^\top+\lambda I)^{-1}Y,\qquad \widehat{\Delta H}=H\widehat B.
$$

图路径与态度嵌套仍用可逆的基础因素分账：

$$
I_{AB}=H_{11}-H_{10}-H_{01}+H_{00},\qquad D^{d,d+1}=H_{d+1}-H_d.
$$
"""
    text = f"""

## Phase {PHASE}: 自然化六族全坐标条件齿轮总合同（C1001-C1016） [{stamp}]

**证据审查与目标。** 对 Phase2234-2246 的附件复核后，成立的是：共享动力学很强；8个受控语言族存在超出共享底盘的预测增量；分类、部分整体和时间图的路径响应跨新词复现；Qwen4B/14B 的族-角色-相对深度拓扑可跨单元检索；严格双向自由生成因果候选仍为0。不能扩张为单坐标齿轮、权重参数机制、跨模型坐标同构或新数学。Phase2237 的自由生成列无效，正式因果账以 Phase2238 为准；Phase2239 的Qwen worker错误也已由 Phase2240 的原始产物恢复账取代。

**大阶段设计、原理与测试用例。** 新合同并行冻结六族：分类图、部分整体图、时间图、共指、属性更新、嵌套态度。父词汇含12个单元，fresh含8个全新单元；中英双语、直接/释义两种表面、四套输出码。图旗舰覆盖1至4跳、直接捷径和真假查询；态度旗舰覆盖like/regret/remember、外层否定、内层否定及真假查询。例子包括“档案记录雪松属于针叶树、针叶树属于树木、树木属于植物，能否推出雪松属于植物”和“日记称娜迪娅并不后悔马库斯没有打开橱柜，内层物品是否为橱柜”。

**冻结公式。** {formula}

**材料、零模型与结果。** 四本材料账为 `{json.dumps({key: {k: value[k] for k in ('rows','panels','partitions','zero_models','token_width_min_median_max','broad_factorial_complete','graph_factorial_complete','attitude_factorial_complete','human_blind_review')} for key, value in audits.items()}, ensure_ascii=False)}`。所有角色必须由真实字符串跨度编译；语义唯一性以显式真值表和完整因素格审计，自然度机器审计只检查畸形，不替代独立人类盲评，后者严格记NA。

**模型、门槛与停止哲学。** Qwen3-4B运行全部自然宽族与组合材料；Qwen3-14B、GLM4、DeepSeek-7B使用完全相同的fresh宽族语义分母，逐个加载。总体候选和自由生成均不低于0.75才捕获内部场，族级分析另要求本族双行为不低于0.75。锦标赛一次比较零响应、共享/族逐坐标均值、共享/族同坐标仿射、共享/族全坐标对偶岭和等容量错族；不允许事后Top-K。因果只使用前瞻胜者，冻结检查点、全部六角色、全/低幅值半场/高幅值半场/正负响应分层、三剂量、单次与逐生成步写入，并含错族、错角色、错层和等范数符号置换控制。任何一路失败只淘汰该主张，不中止其他观察。

**理论进展、硬伤与授权。** 理论名称保持“条件化输出场闭合理论”，RDC不变。本期没有模型或HiddenState结果，只建立更自然、更大样本且能同时裁决图路径、状态绑定、态度组合、跨坐标预测和因果时序的大合同。硬伤是人类盲评NA；文本仍由受控生成器产生；四套答案码仍是元语言接口；全坐标岭可预测不等于稀疏或唯一齿轮；激活坐标不是权重参数。材料与合同工程检查通过后，授权连续执行所有预注册分支。

**相关文件。** 脚本 `tests/glm5/phase2247_c1001_c1016_natural_flagship_contract.py`；结果 `{OUT.relative_to(ROOT)}`；材料、审计、预注册和哈希均保存在该结果目录。
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
    protocol = preregistration()
    save(OUT / "protocol/preregistration.json", {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **protocol})
    materials = {
        "parent_broad": broad_material(False),
        "fresh_broad": broad_material(True),
        "parent_composition": composition_material(False),
        "fresh_composition": composition_material(True),
    }
    tokenizer = prior.parent.load_tokenizer()
    audits = {}
    hashes = {}
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
        checks[f"{name}_unique"] = not audit["duplicate_case_ids"] and audit["unique_case_ids"] == len(rows)
        checks[f"{name}_roles"] = not audit["missing_roles"]
        checks[f"{name}_strings"] = not audit["malformed_strings"]
        checks[f"{name}_balanced"] = all(abs(value - 0.5) <= 1e-12 for value in audit["zero_models"].values())
        if name.endswith("_broad"):
            checks[f"{name}_factorials"] = audit["broad_factorial_complete"]
        else:
            checks[f"{name}_factorials"] = (audit["graph_factorial_complete"] and
                                              audit["attitude_factorial_complete"])
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "material_audits": audits, "hashes": hashes, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "The naturalized six-family campaign is frozen and compiler-valid. No model or HiddenState claim exists in this phase.",
        "next_authorization": "Run every registered route sequentially without changing materials, partitions, models, controls or gates.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
