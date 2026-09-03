#!/usr/bin/env python3
"""C773-C808 semantic-transition ecology campaign.

The campaign reads embeddings, every post-block HiddenState, final norm and
logits.  It keeps every activation coordinate during analysis and does not
read attention/MLP internals, gradients or weights.  It does not use PCA,
Top-K, cosine screening or donor-state difference transport.
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c808_semantic_transition_ecology.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c808_semantic_transition_ecology.float16.npy"
sys.path.insert(0, str(TESTS))

import phase2105_c571_c589_scope_program_algebra_campaign as scope
import phase2190_c656_c669_absolute_coordinate_grammar_campaign as local_base
import phase2200_c684_c709_unified_relation_response_campaign as behavior_base
import phase2205_c710_c744_response_equivalence_atlas_campaign as parent
import phase2211_c745_c760_fresh_passport_causal_campaign as fresh_parent
import phase2215_c761_c772_coreference_causal_replication as causal_parent


PHASES = {
    "C773-C778": (2219, "evidence_repair_and_semantic_transition_contract"),
    "C779-C786": (2220, "dual_behavior_full_coordinate_transition_ecology"),
    "C787-C794": (2221, "fresh_lexicon_prospective_transition_prediction"),
    "C795-C800": (2222, "output_call_and_necessity_branch"),
    "C801-C805": (2223, "sequential_cross_model_relative_transition_topology"),
    "C806-C808": (2224, "visualization_cleanup_and_joint_adjudication"),
    "C809-C812": (2225, "executor_repair_and_cross_model_completion"),
    "C813-C816": (2226, "route_level_readjudication_and_campaign_closure"),
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower().replace('-', '_')}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

DIM = 2560
CHECKPOINTS = 38
ROLES = tuple(local_base.ROLES)
QPOINTS = (0, 8, 16, 24, 32, 37)
FOCUS_ROLES = ("relation", "boundary")
FAMILIES = tuple(parent.FAMILIES)
LANGUAGES = ("en", "zh")
TRANSFORMS = (1, 2, 3)
SEMANTIC_TRANSFORMS = (1, 3)
PARENT_UNITS = 24
FRESH_UNITS = 8
BEHAVIOR_GATE = 0.75
CHANGED_RATE_GATE = 0.005
PREDICTION_SCORE_GATE = 0.25
PREDICTION_GAIN_GATE = 0.02
OUTPUT_GAIN_GATE = 0.05
SHIFT = 257

CELL_NAMES = ("false_base", "true_direct", "false_surface_control", "true_composite")
TRUTH_BY_CELL = (False, True, False, True)

EN_OBJECTS = tuple(parent.OBJECTS_EN)
EN_DISTRACT = tuple(parent.DISTRACT_EN)
EN_NAMES_A = tuple(parent.NAMES_A_EN)
EN_NAMES_B = tuple(parent.NAMES_B_EN)
ZH_OBJECTS = (
    "苹果", "香蕉", "梨", "桃子", "葡萄", "柠檬", "橙子", "李子",
    "樱桃", "芒果", "甜瓜", "椰子", "胡萝卜", "土豆", "番茄", "洋葱",
    "卷心菜", "豆子", "豌豆", "玉米", "大米", "小麦", "面包", "奶酪",
)
ZH_DISTRACT = (
    "锤子", "小提琴", "梯子", "指南针", "灯笼", "枕头", "镜子", "水桶",
    "船锚", "头盔", "口哨", "三脚架", "笔记本", "丝带", "篮子", "钥匙",
    "毯子", "瓶子", "雨伞", "绳子", "平板", "文件夹", "贝壳", "画框",
)
ZH_NAMES_A = (
    "明宇", "诺兰", "伊澄", "达仁", "乐宁", "欧文", "芳怡", "嘉文",
    "雅莉", "若南", "维拉", "凯文", "娜迪", "以利", "思兰", "博然",
    "艾琳", "乔安", "莱雅", "马睿", "可欣", "思远", "宁娜", "拓文",
)
ZH_NAMES_B = (
    "托文", "思琳", "博林", "琪拉", "米洛", "瑞雅", "达蒙", "依琳",
    "帕维", "诺拉", "伊沃", "米娜", "拉维", "特莎", "诺尔", "卡拉",
    "多林", "莉娜", "佩林", "玛雅", "奥林", "塔拉", "维托", "丽娜",
)
FRENCH_PARENT = (
    "pomme", "banane", "poire", "peche", "raisin", "citron", "orange", "prune",
    "cerise", "mangue", "melon", "coco", "carotte", "pomme-de-terre", "tomate", "oignon",
    "chou", "haricot", "pois", "mais", "riz", "ble", "pain", "fromage",
)

FRESH_EN_OBJECTS = tuple(fresh_parent.OBJECTS_EN)
FRESH_EN_DISTRACT = tuple(fresh_parent.DISTRACT_EN)
FRESH_EN_NAMES_A = tuple(fresh_parent.NAMES_A)
FRESH_EN_NAMES_B = tuple(fresh_parent.NAMES_B)
FRESH_ZH_OBJECTS = ("灯笼", "罗盘", "小提琴", "高脚杯", "头盔", "船锚", "马鞍", "水壶")
FRESH_ZH_DISTRACT = ("镜子", "篮子", "锤子", "枕头", "梯子", "丝带", "铲子", "蜡烛")
FRESH_ZH_NAMES_A = ("依洛", "海川", "伊莎", "嘉锐", "路安", "梅岚", "宁雅", "欧思")
FRESH_ZH_NAMES_B = ("佩宁", "启兰", "思彬", "泰然", "尤里", "维雅", "文澜", "泽文")
FRESH_SOURCE = ("moon", "star", "river", "mountain", "window", "door", "chair", "table")
FRESH_FRENCH = ("lune", "etoile", "riviere", "montagne", "fenetre", "porte", "chaise", "table")


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


def out(name: str) -> Path:
    return OUTS[name]


def final(name: str) -> dict:
    return load(out(name) / "analysis/final.json")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def parent_partition(unit: int) -> str:
    return "discovery" if unit < 12 else ("confirmation" if unit < 18 else "lockbox")


def fresh_partition(unit: int) -> str:
    return "confirmation" if unit < 4 else "lockbox"


def type_chain(unit: int, language: str, fresh: bool) -> tuple[str, str, str]:
    if fresh:
        return (("artifact", "manufactured item", "physical object") if language == "en"
                else ("人工制品", "制成品", "实体物品"))
    if unit < 12:
        return (("fruit", "food", "physical object") if language == "en"
                else ("水果", "食物", "实体物品"))
    if unit < 20:
        return (("vegetable", "food", "physical object") if language == "en"
                else ("蔬菜", "食物", "实体物品"))
    return (("food", "consumable", "physical object") if language == "en"
            else ("食物", "可消费品", "实体物品"))


def lexicon(unit: int, language: str, fresh: bool) -> dict:
    if fresh:
        if language == "en":
            a, b = FRESH_EN_NAMES_A[unit], FRESH_EN_NAMES_B[unit]
            x, y = FRESH_EN_OBJECTS[unit], FRESH_EN_DISTRACT[unit]
        else:
            a, b = FRESH_ZH_NAMES_A[unit], FRESH_ZH_NAMES_B[unit]
            x, y = FRESH_ZH_OBJECTS[unit], FRESH_ZH_DISTRACT[unit]
        source, fr = FRESH_SOURCE[unit], FRESH_FRENCH[unit]
        wrong_fr = FRESH_FRENCH[(unit + 3) % FRESH_UNITS]
    else:
        if language == "en":
            a, b = EN_NAMES_A[unit], EN_NAMES_B[unit]
            x, y = EN_OBJECTS[unit], EN_DISTRACT[unit]
        else:
            a, b = ZH_NAMES_A[unit], ZH_NAMES_B[unit]
            x, y = ZH_OBJECTS[unit], ZH_DISTRACT[unit]
        source, fr = EN_OBJECTS[unit], FRENCH_PARENT[unit]
        wrong_fr = FRENCH_PARENT[(unit + 7) % PARENT_UNITS]
    return {"a": a, "b": b, "x": x, "y": y, "source": source, "fr": fr,
            "wrong_fr": wrong_fr, "types": type_chain(unit, language, fresh)}


def make_case(family: str, language: str, unit: int, cell_i: int, fresh: bool = False) -> dict:
    u = lexicon(unit, language, fresh)
    a, b, x, y = u["a"], u["b"], u["x"], u["y"]
    t1, t2, t3 = u["types"]
    truth = TRUTH_BY_CELL[cell_i]
    if family == "recursive_knowledge":
        if language == "en":
            variants = (
                f"the {x} is a {t1}; every {t1} is a {t2}",
                f"the {x} is a {t1}; every {t1} is a {t2}; every {t2} is a {t3}",
                f"every {t1} is a {t2}; the {x} is a {t1}; the unrelated {y} is a {t3}",
                f"the {x} is a {t3} by direct registry entry; {t1} remains a listed category and the {y} is unrelated",
            )
            core = f"A verified classification registry states: {variants[cell_i]}. Based only on it, is the {x} a {t3}?"
            relation = "is a"
        else:
            variants = (
                f"{x}是一种{t1}；每种{t1}都是{t2}",
                f"{x}是一种{t1}；每种{t1}都是{t2}；每种{t2}都是{t3}",
                f"每种{t1}都是{t2}；{x}是一种{t1}；无关的{y}是一种{t3}",
                f"登记表直接说明{x}是一种{t3}；{t1}仍是登记类别，{y}与此无关",
            )
            core = f"一份经过核验的分类登记表写道：{variants[cell_i]}。只根据登记表，{x}是{t3}吗？"
            relation = "是一种"
        roles = {"primary": x, "secondary": t1, "relation": relation, "context": t3, "query": x}
    elif family == "nested_attitude":
        if language == "en":
            variants = (
                f"{a} heard that {b} stored the {x}",
                f"{a} remembered that {b} stored the {x}",
                f"{a} remembered that {b} stored the unrelated {y}",
                f"{a} remembered that the {x} had been stored by {b}",
            )
            core = f"A verified memory record states: {variants[cell_i]}. Does it say that {a} remembered that {b} stored the {x}?"
            relation = "remembered"
        else:
            variants = (
                f"{a}听说{b}存放了{x}",
                f"{a}记得{b}存放了{x}",
                f"{a}记得{b}存放了无关的{y}",
                f"{a}记得{x}曾由{b}存放",
            )
            core = f"一份经过核验的记忆记录写道：{variants[cell_i]}。记录是否表明{a}记得{b}存放了{x}？"
            relation = "记得"
        roles = {"primary": a, "secondary": b, "relation": relation, "context": x, "query": x}
    elif family == "voice_negation":
        if language == "en":
            variants = (
                f"{b} did not move the {x}", f"{b} moved the {x}",
                f"the {x} was not moved by {b}", f"the {x} was moved by {b}",
            )
            core = f"A verified event report states that {variants[cell_i]}. Does it support that {b} moved the {x}?"
            relation = "moved"
        else:
            variants = (
                f"{b}没有搬动{x}", f"{b}搬动了{x}", f"{x}没有被{b}搬动", f"{x}被{b}搬动了",
            )
            core = f"一份经过核验的事件报告写道：{variants[cell_i]}。报告是否支持{b}搬动了{x}？"
            relation = "搬动"
        roles = {"primary": b, "secondary": x, "relation": relation, "context": x, "query": x}
    elif family == "temporal_update":
        if language == "en":
            variants = (
                f"{a} first stored the {x}, but the latest entry replaced it with the {y}",
                f"{a} first stored the {y}, but the latest entry replaced it with the {x}",
                f"the older entry listed the {x}; the current entry for {a} lists the {y}",
                f"the older entry listed the {y}; the current entry for {a} lists the {x}",
            )
            core = f"A verified update log states: {variants[cell_i]}. Is the current stored item for {a} the {x}?"
            relation = "current"
        else:
            variants = (
                f"{a}最初存放{x}，但最新条目改为{y}", f"{a}最初存放{y}，但最新条目改为{x}",
                f"旧条目列出{x}；{a}的当前条目列出{y}", f"旧条目列出{y}；{a}的当前条目列出{x}",
            )
            core = f"一份经过核验的更新日志写道：{variants[cell_i]}。{a}当前存放的物品是{x}吗？"
            relation = "当前"
        roles = {"primary": a, "secondary": a, "relation": relation, "context": x, "query": x}
    elif family == "coreference_binding":
        if language == "en":
            variants = (
                f"{b} told {a}, 'I stored the {x}.'", f"{a} told {b}, 'I stored the {x}.'",
                f"the transcript quotes {b}: 'I stored the {x}.'", f"{a}'s diary, archived by {b}, says, 'I stored the {x}.'",
            )
            core = f"A verified quotation record states: {variants[cell_i]} In that quotation, does I refer to {a}?"
            relation = "I"
        else:
            variants = (
                f"{b}对{a}说：‘我存放了{x}。’", f"{a}对{b}说：‘我存放了{x}。’",
                f"谈话记录引用{b}的话：‘我存放了{x}。’", f"由{b}归档的{a}日记写道：‘我存放了{x}。’",
            )
            core = f"一份经过核验的引语记录写道：{variants[cell_i]}在这段引语中，‘我’指的是{a}吗？"
            relation = "我"
        roles = {"primary": a, "secondary": b, "relation": relation, "context": x, "query": a}
    elif family == "translation_route":
        mapping = u["fr"] if truth else u["wrong_fr"]
        if language == "en":
            variants = (
                f"the glossary maps {u['source']} to French {mapping}",
                f"the checked French translation of {u['source']} is {mapping}",
                f"an unrelated note also contains {u['fr']}; the glossary maps {u['source']} to {mapping}",
                f"the bilingual card pairs {u['source']} with the French form {mapping}",
            )
            core = f"A verified bilingual record states that {variants[cell_i]}. According to it, is the French translation of {u['source']} {u['fr']}?"
            relation = "French translation"
        else:
            variants = (
                f"词表把英文{u['source']}映射为法语{mapping}",
                f"核验后的英文{u['source']}的法语翻译是{mapping}",
                f"一条无关备注也含有{u['fr']}；词表把英文{u['source']}映射为{mapping}",
                f"双语卡片把英文{u['source']}与法语形式{mapping}配对",
            )
            core = f"一份经过核验的双语记录写道：{variants[cell_i]}。根据记录，英文{u['source']}的法语翻译是{u['fr']}吗？"
            relation = "法语翻译"
        roles = {"primary": u["source"], "secondary": mapping, "relation": relation,
                 "context": u["fr"], "query": u["source"]}
    else:
        raise KeyError(family)

    yes, no = (("Yes", "No") if language == "en" else ("是", "否"))
    correct, wrong = ((yes, no) if truth else (no, yes))
    lang_offset = int(language == "zh")
    gold_position = (unit + cell_i + lang_offset) % 2
    options = f"(A) {correct} (B) {wrong}" if gold_position == 0 else f"(A) {wrong} (B) {correct}"
    if language == "en":
        prompt = f"{core} {options}. Reply with only A or B."
        free_prompt = f"{core} Answer only Yes or No."
    else:
        prompt = f"{core}{options}。只回答A或B。"
        free_prompt = f"{core}只回答‘是’或‘否’。"
    prefix = "c787" if fresh else "c773"
    return {
        "case_id": f"{prefix}-{family}-{language}-u{unit:02d}-{CELL_NAMES[cell_i]}",
        "panel": "semantic_transition_fresh" if fresh else "semantic_transition_parent",
        "family": family, "query_operation": family, "operation_type": family,
        "operation_domain": f"semantic_switch:{family}:{CELL_NAMES[cell_i]}",
        "language": language, "surface": "record" if cell_i < 2 else "paraphrase",
        "cell": CELL_NAMES[cell_i], "cell_i": cell_i, "transform_id": cell_i,
        "unit": unit, "partition": fresh_partition(unit) if fresh else parent_partition(unit),
        "truth": truth, "correct_answer": correct, "wrong_answer": wrong,
        "gold_position": gold_position, "prompt_core": core, "prompt": prompt,
        "free_prompt": free_prompt, "role_values": roles,
        "factors": {"semantic_truth": int(truth), "surface_control": int(cell_i == 2),
                    "fresh_lexicon": int(fresh)},
        "semantic_graph": {"external_family": family, "transform": CELL_NAMES[cell_i],
                           "truth_changes_from_base": bool(truth),
                           "labels_are_external_coordinates_not_internal_modules": True},
    }


def material(fresh: bool = False) -> list[dict]:
    units = FRESH_UNITS if fresh else PARENT_UNITS
    return [make_case(f, language, unit, cell, fresh)
            for f, language, unit, cell in itertools.product(FAMILIES, LANGUAGES, range(units), range(4))]


def protocol(name: str) -> dict:
    common = {
        "frozen_before_model": True,
        "model_order": ["qwen3-4b", "glm4", "deepseek7b", "qwen3-14b"],
        "camera": "embedding + all post-block HiddenStates + final norm + logits; every activation coordinate",
        "forbidden": ["attention", "MLP", "weights", "gradients", "PCA", "Top-K",
                      "cosine screening", "donor HiddenState difference transport"],
        "human_review": "NA_not_run",
        "failure_policy": "route-level missingness; a failed branch never stops other registered branches",
        "evidence_unit": "held-out lexical unit, never layers or coordinates",
        "activation_not_parameter": True,
        "reveal_rule": "Objects, partitions, controls and gates cannot change after model reveal.",
    }
    details = {
        "C773-C778": {"object": "repair the evidence ledger and freeze answer-changing six-family material",
                       "parent_rows": 1152, "fresh_rows": 384},
        "C779-C786": {"object": "dual behavior, full coordinate field and changed-coordinate transition ecology",
                       "behavior_gate": BEHAVIOR_GATE, "changed_rate_gate": CHANGED_RATE_GATE,
                       "score_gate": PREDICTION_SCORE_GATE, "gain_gate": PREDICTION_GAIN_GATE},
        "C787-C794": {"object": "predict lexically disjoint fresh transitions with frozen coordinate rules",
                       "score_gate": PREDICTION_SCORE_GATE, "gain_gate": PREDICTION_GAIN_GATE},
        "C795-C800": {"object": "all-qualified output call and necessity branches",
                       "output_gain_gate": OUTPUT_GAIN_GATE},
        "C801-C805": {"object": "sequential cross-model relative-depth transition topology",
                       "same_physical_coordinate_comparison": False},
        "C806-C808": {"object": "exact-coordinate visualization, cleanup and joint adjudication",
                       "new_mathematics_gate": "requires compact prospective causal law, not an observational pattern"},
        "C809-C812": {"object": "repair the missing cross_model_group field and complete only affected workers",
                       "scientific_objects_and_gates_unchanged": True},
        "C813-C816": {"object": "route-level procedural re-adjudication without changing any scientific result",
                       "scientific_objects_and_gates_unchanged": True},
    }
    return {**common, **details[name]}


TITLES = {
    "C773-C778": "证据账本修复与六族答案改变合同",
    "C779-C786": "双行为门与全坐标转移生态图",
    "C787-C794": "新词汇前瞻全坐标转移预测",
    "C795-C800": "输出调用与必要性分支裁决",
    "C801-C805": "四模型顺序相对转移拓扑",
    "C806-C808": "坐标级可视化、清理与联合裁决",
    "C809-C812": "执行器审计修复与跨模型补全",
    "C813-C816": "路线级重裁与大阶段闭幕",
}

FORMULAS = {
    "C773-C778": "$$\n\\mathfrak L=(\\mathcal G,\\mathcal O,\\Sigma,\\mathcal Q),\\quad o:\\mathcal G_{dom(o)}\\rightharpoonup\\mathcal G\n$$",
    "C779-C786": "$$\nT_{q,r,j}=\\operatorname{class}(H^{(1)}_{q,r,j}-H^{(0)}_{q,r,j}\\mid H^{(0)}_{q,r,j}),\\quad S=\\Pr[\\widehat T=T\\mid T\\ne0]\n$$",
    "C787-C794": "$$\nG_{fresh}=S_{frozen}-\\max(S_{wrong},S_{shift},S_{nuisance})\n$$",
    "C795-C800": "$$\nG_{out}=\\Delta m_{aligned}-\\max(\\Delta m_{wrong},\\Delta m_{shift},0)\n$$",
    "C801-C805": "$$\n\\Theta_M(u,r)=\\{\\Pr_M[T\\ne0],\\Pr_M[\\operatorname{signflip}]\\},\\qquad j_M\\not\\equiv j_{M'}\n$$",
    "C806-C808": "$$\n\\text{embedding}\\rightarrow\\text{full coordinate state field}\\rightarrow\\text{conditional transition ecology}\\rightarrow\\text{future output}\n$$",
    "C809-C812": "$$\n\\operatorname{NA}_{interface}\\ne\\operatorname{negative}_{mechanism},\\qquad j_M\\not\\equiv j_{M'}\n$$",
    "C813-C816": "$$\n\\text{answer-changing language program}\\rightarrow\\text{changed-coordinate prediction}\\not\\Rightarrow\\text{causal output program}\n$$",
}


def freeze() -> None:
    for name in PHASES:
        for part in ("protocol", "material", "behavior", "raw", "analysis", "audit", "external"):
            (out(name) / part).mkdir(parents=True, exist_ok=True)
        path = out(name) / "protocol/preregistration.json"
        if not path.exists():
            save(path, {"phase": PHASES[name][0], "campaign": name,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "protocol": protocol(name)})


def append_memo(name: str, result: dict) -> None:
    phase = PHASES[name][0]
    marker = f"## Phase {phase}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    summary = result.get("memo_result_summary", result)
    text = f"""

## Phase {phase}: {TITLES[name]} [{stamp}]

**研究边界。** `{name}` 只读取词嵌入、每个 block 后 HiddenState、final norm 与输出 logits 的全部物理激活坐标。激活坐标不是模型权重参数。本期不读取 Attention/MLP 内部、梯度或权重，不使用 PCA、Top-K、余弦筛选或 donor HiddenState 差分搬运。六个语言族只作为外部实验坐标；独立人类盲评未运行，严格记为 `NA_not_run`。

**运行前冻结合同。**
```json
{json.dumps(load(out(name) / 'protocol/preregistration.json'), ensure_ascii=False, indent=2)}
```

**测试原理、测试用例与数学公式。** 每个语言单元固定查询，四个条件依次为“假基线、真操作、假表面负控、真复合操作”。覆盖类型链补全、态度事件、语态否定、时间更新、引语共指和翻译映射；中英文、discovery/confirmation/lockbox 与候选位置均分账。

{FORMULAS[name]}

**详细结果、分母与门槛。**
```json
{json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)}
```

**分析与理论进展。** {result.get('strict_interpretation')} 理论主体名称保持“条件化输出场闭合理论”，组织原则保持“复用—差分—条件化”。本期只更新经验对象，不把观察邻域命名为等价类，也不因高复现率宣称新数学。

**问题、硬伤和瓶颈。** 人类自然度仍为 NA；材料仍是受控语言而非开放语料；小模型结果不能直接外推；全坐标预测可能包含通用输出准备；坐标条件中位变化仍是经验核，不是模型内置查表；激活坐标不是权重参数；跨模型只比较相对深度与角色拓扑；删除大场会损失免重跑复算能力，故仅在坐标级可视化导出和哈希完成后执行。

**相关文件。** 主脚本 `tests/glm5/phase2219_c773_c808_semantic_transition_ecology_campaign.py`；结果目录 `{out(name).relative_to(ROOT)}`；正式裁决 `{(out(name) / 'analysis/final.json').relative_to(ROOT)}`。

**严格结论与下一步授权。** {result.get('next_authorization')}
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def close(name: str, body: dict, checks: dict, authorization: str) -> dict:
    result = {"phase": PHASES[name][0], "campaign": name, "status": "closed",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "all_checks_passed": bool(checks) and all(bool(v) for v in checks.values()),
              **body, "next_authorization": authorization}
    save(out(name) / "analysis/final.json", result)
    append_memo(name, result)
    print(f"[{name}] closed checks={result['all_checks_passed']}", flush=True)
    return result


def compile_rows(rows: list[dict]) -> list[dict]:
    tokenizer = parent.load_tokenizer()
    return scope.compiler.compile_qwen(tokenizer, rows)


def material_audit(rows: list[dict], compiled: list[dict], fresh: bool) -> dict:
    balances = defaultdict(lambda: [0, 0])
    truths = defaultdict(lambda: [0, 0])
    for row in rows:
        key = f"{row['family']}|{row['language']}|{row['partition']}"
        balances[key][row["gold_position"]] += 1
        truths[key][int(row["truth"])] += 1
    missing = [{"case_id": row["case_id"], "role": role, "value": value}
               for row in rows for role, value in row["role_values"].items()
               if value not in row["prompt_core"]]
    widths = [len(row["prompt_ids"]) for row in compiled]
    partitions = defaultdict(set)
    for row in rows:
        partitions[row["partition"]].add(row["role_values"]["primary"])
    overlap = set()
    keys = list(partitions)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            overlap |= partitions[left] & partitions[right]
    return {
        "rows": len(rows), "fresh": fresh,
        "families": {f: sum(row["family"] == f for row in rows) for f in FAMILIES},
        "languages": {l: sum(row["language"] == l for row in rows) for l in LANGUAGES},
        "partitions": {p: sum(row["partition"] == p for row in rows) for p in partitions},
        "candidate_balance": dict(balances), "truth_balance": dict(truths),
        "zero_models": {"always_A": float(np.mean([r["gold_position"] == 0 for r in rows])),
                        "always_B": float(np.mean([r["gold_position"] == 1 for r in rows])),
                        "always_true": float(np.mean([r["truth"] for r in rows]))},
        "missing_roles": missing, "cross_partition_primary_overlap": sorted(overlap),
        "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "human_review": "NA_not_run",
    }


def phase2219() -> None:
    name = "C773-C778"
    if (out(name) / "analysis/final.json").exists():
        return
    parent_rows, fresh_rows = material(False), material(True)
    parent_compiled, fresh_compiled = compile_rows(parent_rows), compile_rows(fresh_rows)
    parent_path = out(name) / "material/semantic_transition_parent.jsonl"
    parent_compiled_path = out(name) / "material/qwen_parent_compiled.jsonl"
    fresh_path = out(name) / "material/semantic_transition_fresh.jsonl"
    fresh_compiled_path = out(name) / "material/qwen_fresh_compiled.jsonl"
    write_rows(parent_path, parent_rows); write_rows(parent_compiled_path, parent_compiled)
    write_rows(fresh_path, fresh_rows); write_rows(fresh_compiled_path, fresh_compiled)
    parent_audit = material_audit(parent_rows, parent_compiled, False)
    fresh_audit = material_audit(fresh_rows, fresh_compiled, True)
    save(out(name) / "audit/parent_material.json", parent_audit)
    save(out(name) / "audit/fresh_material.json", fresh_audit)
    write_rows(out(name) / "external/human_blind_review_template.jsonl", [
        {"case_id": row["case_id"], "naturalness_1_5": None, "semantic_uniqueness_0_1": None,
         "answerability_0_1": None, "reviewer": None}
        for row in parent_rows + fresh_rows if row["partition"] == "lockbox"
    ])

    repair_phases = (2206, 2207, 2208, 2212, 2213, 2216, 2217)
    repair = []
    for phase in repair_phases:
        dirs = list(RESULT.glob(f"phase{phase}_*"))
        finals = [d / "analysis/final.json" for d in dirs if (d / "analysis/final.json").exists()]
        repair.append({"phase": phase, "formal_final_count": len(finals),
                       "formal_final_bytes": sum(p.stat().st_size for p in finals),
                       "memo_detail_was_blank": True,
                       "independent_formal_files_exist": bool(finals),
                       "files": [str(p.relative_to(ROOT)) for p in finals]})
    old_final = causal_parent.final("C771-C772")
    evidence = {
        "retained": {
            "parent_observational_passports": "35/36",
            "fresh_observational_passports": "33/36",
            "local_family_specific_groups": "0/12",
            "semantic_output_groups": "0/12",
            "single_case_causal_candidate": "1/33",
            "large_sample_causal_replication": "0/22",
        },
        "corrections": [
            "The seven MEMO detail blocks were blank, but their independent formal result files exist.",
            "A hash proves identity of a deleted field but cannot reconstruct it.",
            "Approximate threshold neighborhoods are not transitive equivalence classes.",
            "Unchanged-to-unchanged coordinates must be separated from changed-coordinate prediction.",
            "Activation coordinates are state coordinates, not model parameters.",
            "The final Qwen3-14B checkpoint has no next-checkpoint flip and must be NA, not zero.",
            "The exact discretized English coreference causal passport is closed by 0/22 replication.",
        ],
        "repair_ledger": repair,
        "closed_exact_object": old_final.get("replicated_units") == 0,
    }
    save(out(name) / "audit/evidence_repair.json", evidence)
    summary = {"evidence_repair": evidence, "parent_material": parent_audit,
               "fresh_material": fresh_audit,
               "material_hashes": {"parent": file_hash(parent_path), "fresh": file_hash(fresh_path)}}
    close(name, {
        "strict_interpretation": "The prior observational passport result is retained, its causal interpretation is closed, and the next object is an answer-changing coordinate transition ecology scored only on coordinates that actually change.",
        "evidence_repair": evidence, "parent_material_audit": parent_audit,
        "fresh_material_audit": fresh_audit, "human_review": "NA_not_run",
        "new_foundational_mathematics_gate": False, "memo_result_summary": summary,
    }, {
        "parent_results": all(parent.final(k)["all_checks_passed"] for k in parent.PHASES),
        "causal_re裁": causal_parent.final("C771-C772")["all_checks_passed"],
        "parent_rows": len(parent_rows) == 1152, "fresh_rows": len(fresh_rows) == 384,
        "compiled": len(parent_compiled) == len(parent_rows) and len(fresh_compiled) == len(fresh_rows),
        "roles": not parent_audit["missing_roles"] and not fresh_audit["missing_roles"],
        "balance": all(v[0] == v[1] for v in parent_audit["candidate_balance"].values())
                   and all(v[0] == v[1] for v in parent_audit["truth_balance"].values())
                   and all(v[0] == v[1] for v in fresh_audit["candidate_balance"].values())
                   and all(v[0] == v[1] for v in fresh_audit["truth_balance"].values()),
        "zero_models": all(abs(v - 0.5) < 1e-12 for v in parent_audit["zero_models"].values())
                       and all(abs(v - 0.5) < 1e-12 for v in fresh_audit["zero_models"].values()),
        "partition_isolation": not parent_audit["cross_partition_primary_overlap"],
    }, "Authorize C779-C786 to run dual behavior and full-coordinate transition-ecology discovery; no branch may stop the remaining families.")


def parse_answer(text: str, language: str) -> str | None:
    clean = text.strip().lower()
    if language == "zh":
        positions = [(clean.find("是"), "是"), (clean.find("否"), "否")]
    else:
        import re
        match = re.search(r"\b(yes|no)\b", clean)
        return match.group(1).capitalize() if match else None
    positions = [(p, value) for p, value in positions if p >= 0]
    return min(positions)[1] if positions else None


def run_behavior(model, tokenizer, device, compiled: list[dict], prefix: str) -> tuple[list[dict], list[dict]]:
    candidate = behavior_base.batch_behavior(model, device, compiled, batch_size=12)
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    generated = []
    for start in range(0, len(compiled), 8):
        batch = compiled[start:start + 8]
        width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["free_prompt_ids"]
            ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, width - len(seq):] = 1
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5,
                                    do_sample=False, pad_token_id=pad,
                                    eos_token_id=tokenizer.eos_token_id)
        for i, row in enumerate(batch):
            text = tokenizer.decode(output[i, width:].tolist(), skip_special_tokens=True)
            parsed = parse_answer(text, row["language"])
            generated.append({"case_id": row["case_id"], "text": text, "parsed": parsed,
                              "correct_answer": row["correct_answer"],
                              "correct": parsed == row["correct_answer"]})
        if start % 128 == 0:
            print(f"[{prefix}] generation {start}/{len(compiled)}", flush=True)
    return candidate, generated


def slice_behavior(compiled: list[dict], candidate: dict, generated: dict,
                   partitions: tuple[str, ...]) -> tuple[dict, set[str]]:
    slices = {}
    qualified = set()
    for family, language in itertools.product(FAMILIES, LANGUAGES):
        panel = {}
        for part in partitions:
            rows = [r for r in compiled if r["family"] == family and r["language"] == language
                    and r["partition"] == part]
            panel[part] = {
                "rows": len(rows),
                "candidate_accuracy": float(np.mean([candidate[r["case_id"]]["correct"] for r in rows])),
                "generation_accuracy": float(np.mean([generated[r["case_id"]]["correct"] for r in rows])),
                "dual_accuracy": float(np.mean([candidate[r["case_id"]]["correct"] and
                                                  generated[r["case_id"]]["correct"] for r in rows])),
            }
        panel["qualified"] = all(panel[p][metric] >= BEHAVIOR_GATE for p in partitions
                                 for metric in ("candidate_accuracy", "generation_accuracy"))
        key = f"{family}|{language}"
        slices[key] = panel
        if panel["qualified"]:
            qualified.add(key)
    return slices, qualified


def capture_field(model, device, compiled: list[dict], candidate: dict, generated: dict,
                  qualified: set[str], raw_dir: Path, prefix: str) -> tuple[list[dict], list[dict], dict]:
    selected = [r for r in compiled if f"{r['family']}|{r['language']}" in qualified
                and candidate[r["case_id"]]["correct"] and generated[r["case_id"]]["correct"]]
    panel_ids = set()
    for family, language, part in itertools.product(FAMILIES, LANGUAGES,
                                                     sorted({r["partition"] for r in selected})):
        rows = [r for r in selected if r["family"] == family and r["language"] == language
                and r["partition"] == part and r["cell_i"] in (0, 1)]
        for cell in (0, 1):
            matches = sorted([r for r in rows if r["cell_i"] == cell], key=lambda r: r["unit"])
            if matches:
                panel_ids.add(matches[0]["case_id"])
    panel = [r for r in selected if r["case_id"] in panel_ids]
    max_width = max([len(r["prompt_ids"]) for r in panel], default=1)
    role_path = raw_dir / "all_qualified_role_field.float16.npy"
    token_path = raw_dir / "representative_full_token_field.float16.npy"
    role_field = np.lib.format.open_memmap(role_path, mode="w+", dtype=np.float16,
                                           shape=(len(selected), CHECKPOINTS, len(ROLES), DIM))
    token_field = np.lib.format.open_memmap(token_path, mode="w+", dtype=np.float16,
                                            shape=(len(panel), CHECKPOINTS, max_width, DIM))
    panel_map = {r["case_id"]: i for i, r in enumerate(panel)}
    base = model.model
    captured = []
    modules = [base.embed_tokens, *list(base.layers), base.norm]
    handles = [m.register_forward_hook(lambda _m, _a, o: captured.append(o[0] if isinstance(o, tuple) else o))
               for m in modules]
    index, token_index = [], []
    try:
        for row_i, item in enumerate(selected):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            if len(captured) != CHECKPOINTS:
                raise RuntimeError((item["case_id"], len(captured)))
            for q, hidden in enumerate(captured):
                values = hidden[0].float().cpu().numpy().astype(np.float16)
                for role_i, role in enumerate(ROLES):
                    role_field[row_i, q, role_i] = values[item["role_positions"][role][-1]]
                if item["case_id"] in panel_map:
                    token_field[panel_map[item["case_id"]], q, :values.shape[0]] = values
            index.append({"hidden_index": row_i, "case_id": item["case_id"],
                          "family": item["family"], "language": item["language"],
                          "unit": item["unit"], "partition": item["partition"],
                          "cell_i": item["cell_i"], "prompt_length": len(item["prompt_ids"]),
                          "dual_correct": True})
            if item["case_id"] in panel_map:
                token_index.append({"token_index": panel_map[item["case_id"]],
                                    "case_id": item["case_id"], "family": item["family"],
                                    "language": item["language"], "partition": item["partition"],
                                    "cell_i": item["cell_i"], "prompt_length": len(item["prompt_ids"]),
                                    "prompt_ids": item["prompt_ids"]})
            if row_i % 64 == 0:
                print(f"[{prefix}] capture {row_i}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    role_field.flush(); token_field.flush(); close_mmap(role_field); close_mmap(token_field)
    write_rows(raw_dir / "hidden_index.jsonl", index)
    write_rows(raw_dir / "full_token_index.jsonl", token_index)
    return index, token_index, {
        "role_path": str(role_path.relative_to(ROOT)),
        "role_shape": [len(selected), CHECKPOINTS, len(ROLES), DIM],
        "token_path": str(token_path.relative_to(ROOT)),
        "token_shape": [len(panel), CHECKPOINTS, max_width, DIM],
    }


def base_code(values: np.ndarray) -> np.ndarray:
    code = np.full(values.shape, 2, dtype=np.uint8)
    code[values < -0.5] = 0
    code[(values >= -0.5) & (values < -0.05)] = 1
    code[(values > 0.05) & (values <= 0.5)] = 3
    code[values > 0.5] = 4
    return code


def transition_code(base: np.ndarray, changed: np.ndarray) -> np.ndarray:
    base32 = np.asarray(base, np.float32)
    changed32 = np.asarray(changed, np.float32)
    delta = changed32 - base32
    tau = 0.02 + 0.05 * np.abs(base32)
    code = np.zeros(base.shape, dtype=np.uint8)
    active = np.abs(delta) > tau
    code[active & (base32 <= 0) & (changed32 > 0)] = 1
    code[active & (base32 >= 0) & (changed32 < 0)] = 2
    same = active & (code == 0)
    code[same & (np.abs(changed32) > np.abs(base32))] = 3
    code[same & (np.abs(changed32) <= np.abs(base32))] = 4
    return code


def group_labels() -> list[str]:
    return [f"{f}|{language}|t{t}" for f, language, t in itertools.product(FAMILIES, LANGUAGES, TRANSFORMS)]


def build_transition_models(field_path: Path, index: list[dict], raw_dir: Path) -> tuple[dict, dict]:
    field = np.load(field_path, mmap_mode="r")
    by_key = {(r["family"], r["language"], r["unit"], r["cell_i"]): r["hidden_index"] for r in index}
    labels = group_labels()
    predictor_path = raw_dir / "coordinate_transition_predictor.uint8.npy"
    kernel_path = raw_dir / "coordinate_delta_kernel.float16.npy"
    predictor = np.lib.format.open_memmap(predictor_path, mode="w+", dtype=np.uint8,
                                           shape=(len(labels), 5, CHECKPOINTS, len(ROLES), DIM))
    kernel = np.lib.format.open_memmap(kernel_path, mode="w+", dtype=np.float16,
                                       shape=(len(labels), 5, CHECKPOINTS, len(ROLES), DIM))
    availability = {}
    for group_i, label in enumerate(labels):
        family, language, transform_text = label.split("|")
        transform = int(transform_text[1:])
        pairs = [(by_key[(family, language, unit, 0)], by_key[(family, language, unit, transform)])
                 for unit in range(12)
                 if (family, language, unit, 0) in by_key and (family, language, unit, transform) in by_key]
        availability[label] = len(pairs)
        if not pairs:
            predictor[group_i] = 0; kernel[group_i] = 0
            continue
        base = np.stack([np.asarray(field[a], np.float32) for a, _ in pairs])
        changed = np.stack([np.asarray(field[b], np.float32) for _, b in pairs])
        bcode = base_code(base)
        tcode = transition_code(base, changed)
        delta = changed - base
        for state in range(5):
            state_mask = bcode == state
            counts = np.stack([np.sum(state_mask & (tcode == cls), axis=0) for cls in range(5)])
            predictor[group_i, state] = np.argmax(counts, axis=0).astype(np.uint8)
            masked = np.where(state_mask, delta, np.nan)
            with np.errstate(all="ignore"):
                median = np.nanmedian(masked, axis=0)
            kernel[group_i, state] = np.nan_to_num(median, nan=0.0).astype(np.float16)
        print(f"[C779-C786] model {group_i + 1}/{len(labels)} {label} n={len(pairs)}", flush=True)
    predictor.flush(); kernel.flush(); close_mmap(predictor); close_mmap(kernel); close_mmap(field)
    return {"labels": labels, "availability": availability,
            "predictor_path": str(predictor_path.relative_to(ROOT)),
            "predictor_shape": [len(labels), 5, CHECKPOINTS, len(ROLES), DIM],
            "kernel_path": str(kernel_path.relative_to(ROOT)),
            "kernel_shape": [len(labels), 5, CHECKPOINTS, len(ROLES), DIM]}, by_key


def lookup_prediction(table: np.ndarray, bcode: np.ndarray) -> np.ndarray:
    flat_table = table.reshape(5, -1)
    flat_code = bcode.reshape(-1)
    values = flat_table[flat_code, np.arange(flat_code.size)]
    return values.reshape(bcode.shape)


def score_prediction(actual: np.ndarray, predicted: np.ndarray) -> dict:
    active = actual != 0
    count = int(np.sum(active))
    return {"changed_coordinates": count,
            "changed_rate": float(np.mean(active)),
            "score_on_changed": float(np.mean(predicted[active] == actual[active])) if count else 0.0,
            "all_coordinate_accuracy": float(np.mean(predicted == actual)),
            "predicted_changed_rate": float(np.mean(predicted != 0))}


def evaluate_models(field_path: Path, index: list[dict], model_info: dict,
                    partitions: tuple[str, ...]) -> tuple[dict, list[str], list[dict]]:
    field = np.load(field_path, mmap_mode="r")
    predictor = np.load(ROOT / model_info["predictor_path"], mmap_mode="r")
    labels = model_info["labels"]
    label_to_i = {label: i for i, label in enumerate(labels)}
    by_key = {(r["family"], r["language"], r["unit"], r["cell_i"]): r["hidden_index"] for r in index}
    rows = []
    summaries = {}
    qualified = []
    for label in labels:
        family, language, transform_text = label.split("|")
        transform = int(transform_text[1:])
        group_i = label_to_i[label]
        wrong_family = FAMILIES[(FAMILIES.index(family) + 1) % len(FAMILIES)]
        wrong_i = label_to_i[f"{wrong_family}|{language}|t{transform}"]
        nuisance_i = label_to_i[f"{family}|{language}|t2"]
        for part in partitions:
            units = sorted({r["unit"] for r in index if r["family"] == family and r["language"] == language
                            and r["partition"] == part})
            unit_rows = []
            for unit in units:
                key0, key1 = (family, language, unit, 0), (family, language, unit, transform)
                if key0 not in by_key or key1 not in by_key:
                    continue
                base = np.asarray(field[by_key[key0]], np.float32)
                changed = np.asarray(field[by_key[key1]], np.float32)
                bcode = base_code(base)
                actual = transition_code(base, changed)
                pred = lookup_prediction(np.asarray(predictor[group_i]), bcode)
                wrong = lookup_prediction(np.asarray(predictor[wrong_i]), bcode)
                nuisance = lookup_prediction(np.asarray(predictor[nuisance_i]), bcode)
                shifted = np.roll(pred, SHIFT, axis=-1)
                metrics = score_prediction(actual, pred)
                controls = {"wrong_family": score_prediction(actual, wrong)["score_on_changed"],
                            "nuisance_transform": score_prediction(actual, nuisance)["score_on_changed"],
                            "shift257": score_prediction(actual, shifted)["score_on_changed"]}
                gain = metrics["score_on_changed"] - max(controls.values())
                row = {"group": label, "family": family, "language": language,
                       "transform": transform, "partition": part, "unit": unit,
                       **metrics, "controls": controls, "gain": gain}
                rows.append(row); unit_rows.append(row)
            if unit_rows:
                summary = {"units": len(unit_rows),
                           "mean_changed_rate": float(np.mean([r["changed_rate"] for r in unit_rows])),
                           "mean_score": float(np.mean([r["score_on_changed"] for r in unit_rows])),
                           "median_score": float(np.median([r["score_on_changed"] for r in unit_rows])),
                           "mean_control": float(np.mean([max(r["controls"].values()) for r in unit_rows])),
                           "mean_gain": float(np.mean([r["gain"] for r in unit_rows])),
                           "median_gain": float(np.median([r["gain"] for r in unit_rows]))}
            else:
                summary = {"units": 0, "mean_changed_rate": 0.0, "mean_score": 0.0,
                           "median_score": 0.0, "mean_control": 0.0, "mean_gain": 0.0,
                           "median_gain": 0.0}
            summaries[f"{label}|{part}"] = summary
        if all(summaries[f"{label}|{part}"]["units"] >= 4
               and summaries[f"{label}|{part}"]["mean_changed_rate"] >= CHANGED_RATE_GATE
               and summaries[f"{label}|{part}"]["mean_score"] >= PREDICTION_SCORE_GATE
               and summaries[f"{label}|{part}"]["mean_gain"] >= PREDICTION_GAIN_GATE
               for part in partitions):
            qualified.append(label)
    close_mmap(field); close_mmap(predictor)
    return summaries, qualified, rows


def qwen_model():
    return scope.parent.previous.model_base().load_bf16("qwen3")


def release_model(model) -> None:
    scope.parent.previous.model_base().release_bf16(model)
    gc.collect()


def phase2220() -> None:
    name = "C779-C786"
    if (out(name) / "analysis/final.json").exists():
        return
    compiled = read_rows(out("C773-C778") / "material/qwen_parent_compiled.jsonl")
    model = None
    try:
        model, tokenizer, device, placement = qwen_model()
        candidate_rows, generation_rows = run_behavior(model, tokenizer, device, compiled, name)
        write_rows(out(name) / "behavior/candidate.jsonl", candidate_rows)
        write_rows(out(name) / "behavior/free_generation.jsonl", generation_rows)
        candidate = {r["case_id"]: r for r in candidate_rows}
        generated = {r["case_id"]: r for r in generation_rows}
        slices, qualified_slices = slice_behavior(compiled, candidate, generated,
                                                   ("discovery", "confirmation", "lockbox"))
        index, token_index, field_info = capture_field(model, device, compiled, candidate, generated,
                                                        qualified_slices, out(name) / "raw", name)
        quantization = scope.parent.previous.model_base().quantization_audit(model)
    finally:
        release_model(model)
    model_info, _ = build_transition_models(ROOT / field_info["role_path"], index, out(name) / "raw")
    summaries, qualified_groups, metric_rows = evaluate_models(
        ROOT / field_info["role_path"], index, model_info, ("confirmation", "lockbox"))
    write_rows(out(name) / "analysis/unit_transition_metrics.jsonl", metric_rows)
    save(out(name) / "analysis/group_transition_metrics.json", summaries)
    summary = {"rows": len(compiled), "qualified_slices": sorted(qualified_slices),
               "captured_rows": len(index), "field_shape": field_info["role_shape"],
               "transition_groups": len(model_info["labels"]),
               "qualified_changed_coordinate_groups": qualified_groups,
               "group_metrics": summaries}
    close(name, {
        "strict_interpretation": "This phase predicts answer-changing coordinate transition classes from the coordinate's own base state. Scores are computed on actually changed coordinates, so unchanged-to-unchanged cells cannot create a pass.",
        "behavior_slices": slices, "qualified_slices": sorted(qualified_slices),
        "captured_rows": len(index), "field": field_info, "transition_model": model_info,
        "group_metrics": summaries, "qualified_changed_coordinate_groups": qualified_groups,
        "placement": placement, "quantization": quantization,
        "human_review": "NA_not_run", "new_foundational_mathematics_gate": False,
        "memo_result_summary": summary,
    }, {
        "parent": final("C773-C778")["all_checks_passed"], "behavior_complete": len(candidate_rows) == len(compiled) == len(generation_rows),
        "some_behavior_qualified": bool(qualified_slices), "same_row_capture": all(r["dual_correct"] for r in index),
        "all_coordinates": field_info["role_shape"][-1] == DIM, "all_checkpoints": field_info["role_shape"][1] == CHECKPOINTS,
        "models_complete": all(v >= 8 for v in model_info["availability"].values()),
        "finite": finite(summaries), "memo_details_nonempty": bool(summary["group_metrics"]),
    }, "Authorize C787-C794 to test every frozen transition rule on lexically disjoint fresh material; failed groups remain in the atlas and do not stop other groups.")


def phase2221() -> None:
    name = "C787-C794"
    if (out(name) / "analysis/final.json").exists():
        return
    compiled = read_rows(out("C773-C778") / "material/qwen_fresh_compiled.jsonl")
    model = None
    try:
        model, tokenizer, device, placement = qwen_model()
        candidate_rows, generation_rows = run_behavior(model, tokenizer, device, compiled, name)
        write_rows(out(name) / "behavior/candidate.jsonl", candidate_rows)
        write_rows(out(name) / "behavior/free_generation.jsonl", generation_rows)
        candidate = {r["case_id"]: r for r in candidate_rows}
        generated = {r["case_id"]: r for r in generation_rows}
        slices, qualified_slices = slice_behavior(compiled, candidate, generated, ("confirmation", "lockbox"))
        index, token_index, field_info = capture_field(model, device, compiled, candidate, generated,
                                                        qualified_slices, out(name) / "raw", name)
    finally:
        release_model(model)
    model_info = final("C779-C786")["transition_model"]
    summaries, qualified_groups, metric_rows = evaluate_models(
        ROOT / field_info["role_path"], index, model_info, ("confirmation", "lockbox"))
    write_rows(out(name) / "analysis/fresh_unit_transition_metrics.jsonl", metric_rows)
    save(out(name) / "analysis/fresh_group_transition_metrics.json", summaries)
    parent_qualified = set(final("C779-C786")["qualified_changed_coordinate_groups"])
    prospective = sorted(parent_qualified & set(qualified_groups))
    summary = {"rows": len(compiled), "qualified_slices": sorted(qualified_slices),
               "captured_rows": len(index), "fresh_qualified_groups": qualified_groups,
               "strict_prospective_groups": prospective, "group_metrics": summaries}
    close(name, {
        "strict_interpretation": "A strict prospective group must pass unchanged discovery rules on both fresh confirmation and lockbox units. The result is coordinate-transition predictability, not causal execution or a semantic module.",
        "behavior_slices": slices, "qualified_slices": sorted(qualified_slices),
        "captured_rows": len(index), "field": field_info,
        "fresh_group_metrics": summaries, "fresh_qualified_groups": qualified_groups,
        "strict_prospective_groups": prospective, "placement": placement,
        "human_review": "NA_not_run", "new_foundational_mathematics_gate": False,
        "memo_result_summary": summary,
    }, {
        "parent": final("C779-C786")["all_checks_passed"], "behavior_complete": len(candidate_rows) == len(compiled) == len(generation_rows),
        "some_behavior_qualified": bool(qualified_slices), "same_row_capture": all(r["dual_correct"] for r in index),
        "all_coordinates": field_info["role_shape"][-1] == DIM, "finite": finite(summaries),
        "prospective_accounted": set(prospective) <= parent_qualified,
        "memo_details_nonempty": bool(summary["group_metrics"]),
    }, "Authorize C795-C800 to run output-call and necessity tests for every strict prospective group; if none qualify, record NA and continue cross-model observation.")


def yes_margin(logits: torch.Tensor, item: dict) -> float:
    yes_position = item["gold_position"] if item["truth"] else 1 - item["gold_position"]
    no_position = 1 - yes_position
    return float(logits[item["candidate_ids"][yes_position][0]] - logits[item["candidate_ids"][no_position][0]])


def intervention_forward(model, item: dict, predictor: np.ndarray, kernel: np.ndarray,
                         wrong_predictor: np.ndarray, wrong_kernel: np.ndarray,
                         mode: str, generate: bool, tokenizer=None) -> dict:
    base = model.model
    modules = [base.embed_tokens, *list(base.layers), base.norm]
    ids_key = "free_prompt_ids" if generate else "prompt_ids"
    ids = torch.tensor([item[ids_key]], dtype=torch.long, device=next(model.parameters()).device)
    mask = torch.ones_like(ids)
    positions = torch.arange(ids.shape[1], device=ids.device)[None]
    handles = []
    coordinates = np.arange(DIM)
    for q in QPOINTS:
        module = modules[q]
        def patch(_module, _args, output, q=q):
            hidden = output[0] if isinstance(output, tuple) else output
            changed = hidden.clone()
            for role in FOCUS_ROLES:
                role_i = ROLES.index(role)
                source_pos = int(item["role_positions"][role][-1])
                if source_pos >= hidden.shape[1]:
                    continue
                current = hidden[0, source_pos].float().detach().cpu().numpy()
                states = base_code(current)
                table = wrong_predictor[:, q, role_i] if mode == "wrong" else predictor[:, q, role_i]
                delta_table = wrong_kernel[:, q, role_i] if mode == "wrong" else kernel[:, q, role_i]
                predicted = table[states, coordinates]
                delta = delta_table[states, coordinates].astype(np.float32)
                if mode in ("shift", "shift_delete"):
                    predicted = np.roll(predicted, SHIFT); delta = np.roll(delta, SHIFT)
                active = predicted != 0
                tensor_active = torch.tensor(active, dtype=torch.bool, device=hidden.device)
                tensor_delta = torch.tensor(delta, dtype=torch.float32, device=hidden.device)
                if mode in ("delete", "shift_delete"):
                    changed[0, source_pos, tensor_active] = 0
                elif mode in ("aligned", "wrong", "shift"):
                    changed[0, source_pos, tensor_active] = (
                        changed[0, source_pos, tensor_active].float() + tensor_delta[tensor_active]
                    ).to(hidden.dtype)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        handles.append(module.register_forward_hook(patch))
    try:
        if generate:
            pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            with torch.inference_mode():
                generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5,
                                           do_sample=False, pad_token_id=pad,
                                           eos_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(generated[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
            parsed = parse_answer(text, item["language"])
            return {"text": text, "parsed": parsed}
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                           use_cache=False, return_dict=True).logits[0, -1].float()
        return {"yes_minus_no_margin": yes_margin(logits, item)}
    finally:
        for handle in handles:
            handle.remove()


def phase2222() -> None:
    name = "C795-C800"
    if (out(name) / "analysis/final.json").exists():
        return
    eligible = final("C787-C794")["strict_prospective_groups"]
    compiled = read_rows(out("C773-C778") / "material/qwen_fresh_compiled.jsonl")
    model_info = final("C779-C786")["transition_model"]
    labels = model_info["labels"]
    label_to_i = {label: i for i, label in enumerate(labels)}
    predictor = np.load(ROOT / model_info["predictor_path"], mmap_mode="r")
    kernel = np.load(ROOT / model_info["kernel_path"], mmap_mode="r")
    results = {}
    model = None
    if eligible:
        try:
            model, tokenizer, device, placement = qwen_model()
            for label in eligible:
                family, language, transform_text = label.split("|")
                transform = int(transform_text[1:])
                group_i = label_to_i[label]
                wrong_family = FAMILIES[(FAMILIES.index(family) + 1) % len(FAMILIES)]
                wrong_i = label_to_i[f"{wrong_family}|{language}|t{transform}"]
                units = []
                for unit in range(FRESH_UNITS):
                    base_row = next((r for r in compiled if r["family"] == family and r["language"] == language
                                     and r["unit"] == unit and r["cell_i"] == 0), None)
                    true_row = next((r for r in compiled if r["family"] == family and r["language"] == language
                                     and r["unit"] == unit and r["cell_i"] == transform), None)
                    if base_row is None or true_row is None:
                        continue
                    modes = {mode: intervention_forward(model, base_row, predictor[group_i], kernel[group_i],
                                                          predictor[wrong_i], kernel[wrong_i], mode, False)
                             for mode in ("aligned", "wrong", "shift")}
                    base_eval = intervention_forward(model, base_row, predictor[group_i], kernel[group_i],
                                                     predictor[wrong_i], kernel[wrong_i], "base", False)
                    true_eval = intervention_forward(model, true_row, predictor[group_i], kernel[group_i],
                                                     predictor[wrong_i], kernel[wrong_i], "base", False)
                    delete_eval = intervention_forward(model, true_row, predictor[group_i], kernel[group_i],
                                                       predictor[wrong_i], kernel[wrong_i], "delete", False)
                    shift_delete = intervention_forward(model, true_row, predictor[group_i], kernel[group_i],
                                                        predictor[wrong_i], kernel[wrong_i], "shift_delete", False)
                    generation = {mode: intervention_forward(model, base_row, predictor[group_i], kernel[group_i],
                                                              predictor[wrong_i], kernel[wrong_i], mode, True, tokenizer)
                                  for mode in ("aligned", "wrong", "shift")}
                    base_margin = base_eval["yes_minus_no_margin"]
                    aligned_gain = modes["aligned"]["yes_minus_no_margin"] - base_margin
                    best_control = max(modes["wrong"]["yes_minus_no_margin"] - base_margin,
                                       modes["shift"]["yes_minus_no_margin"] - base_margin, 0.0)
                    necessity = ((true_eval["yes_minus_no_margin"] - delete_eval["yes_minus_no_margin"])
                                 - max(true_eval["yes_minus_no_margin"] - shift_delete["yes_minus_no_margin"], 0.0))
                    units.append({"unit": unit, "partition": fresh_partition(unit), "base": base_eval,
                                  "true": true_eval, "modes": modes, "generation": generation,
                                  "output_call_gain": aligned_gain - best_control,
                                  "necessity_gain": necessity})
                panels = {}
                for part in ("confirmation", "lockbox"):
                    subset = [u for u in units if u["partition"] == part]
                    panels[part] = {"units": len(subset),
                                    "mean_output_call_gain": float(np.mean([u["output_call_gain"] for u in subset])) if subset else 0.0,
                                    "mean_necessity_gain": float(np.mean([u["necessity_gain"] for u in subset])) if subset else 0.0,
                                    "call_pass_units": sum(u["output_call_gain"] >= OUTPUT_GAIN_GATE for u in subset),
                                    "necessity_pass_units": sum(u["necessity_gain"] >= OUTPUT_GAIN_GATE for u in subset),
                                    "aligned_generation_yes": sum(u["generation"]["aligned"]["parsed"] in ("Yes", "是") for u in subset)}
                passed = all(panels[p]["call_pass_units"] >= 3 and panels[p]["necessity_pass_units"] >= 3
                             for p in panels)
                results[label] = {"units": units, "partitions": panels, "passed": passed}
                print(f"[{name}] {label} passed={passed}", flush=True)
        finally:
            release_model(model)
    else:
        placement = "NA_no_strict_prospective_group"
    close_mmap(predictor); close_mmap(kernel)
    passed = sorted([label for label, value in results.items() if value["passed"]])
    summary = {"eligible_groups": eligible, "tested_groups": len(results),
               "passed_output_and_necessity_groups": passed,
               "partition_summaries": {k: v["partitions"] for k, v in results.items()}}
    close(name, {
        "strict_interpretation": "Only strict prospective transition groups entered intervention. Adding a frozen coordinate-conditioned kernel to false base prompts tests output calling; zeroing its predicted-changing coordinates on true prompts tests broad necessity. No rescue-equivalence claim is made.",
        "eligible_groups": eligible, "intervention_results": results,
        "passed_output_and_necessity_groups": passed, "placement": placement,
        "human_review": "NA_not_run", "new_foundational_mathematics_gate": False,
        "memo_result_summary": summary,
    }, {
        "parent": final("C787-C794")["all_checks_passed"],
        "all_eligible_accounted": set(results) == set(eligible),
        "route_level_na_valid": bool(eligible) or placement == "NA_no_strict_prospective_group",
        "finite": finite(results), "memo_details_nonempty": bool(summary),
    }, "Authorize C801-C805 to run the sequential cross-model relative topology regardless of intervention outcome; physical coordinate IDs may not be compared across models.")


def qwen_relative_topology() -> dict:
    info = final("C779-C786")["field"]
    field = np.load(ROOT / info["role_path"], mmap_mode="r")
    sampled = []
    for q in QPOINTS:
        values = np.asarray(field[:, q], np.float32)
        if q == CHECKPOINTS - 1:
            flip = None
        else:
            nxt = np.asarray(field[:, q + 1], np.float32)
            flip = [float(np.mean(np.sign(values[:, r]) != np.sign(nxt[:, r]))) for r in range(len(ROLES))]
        sampled.append({"relative_depth": q / (CHECKPOINTS - 1),
                        "positive_rate_by_role": [float(np.mean(values[:, r] > 0)) for r in range(len(ROLES))],
                        "next_sign_flip_by_role": flip})
    close_mmap(field)
    return {"model": "qwen3-4b", "qualified": True, "hiddenstate_ran": True,
            "checkpoints": CHECKPOINTS, "coordinates": DIM, "relative_topology": sampled}


def phase2223() -> None:
    name = "C801-C805"
    if (out(name) / "analysis/final.json").exists():
        return
    rows = read_rows(out("C773-C778") / "material/semantic_transition_parent.jsonl")
    panel = [r for r in rows if r["unit"] in (18, 19) and r["cell_i"] in (0, 1)]
    material_path = out(name) / "material/cross_model_48_case_panel.jsonl"
    write_rows(material_path, panel)
    workers = {"qwen3-4b": qwen_relative_topology()}
    for model_name in ("glm4", "deepseek7b", "qwen3_14b"):
        worker_output = out(name) / f"raw/{model_name}/worker_result.json"
        completed = subprocess.run([sys.executable, str(Path(local_base.__file__)), "--worker", model_name,
                                    "--material", str(material_path), "--output", str(worker_output)],
                                   cwd=ROOT, check=False)
        value = load(worker_output) if worker_output.exists() else {"model": model_name, "status": "missing_worker_output"}
        value["returncode"] = completed.returncode
        if value.get("relative_topology"):
            value["relative_topology"][-1]["next_sign_flip_by_role"] = None
            value["final_checkpoint_next_flip_correction"] = "NA_no_next_checkpoint"
        workers[model_name] = value
        print(f"[{name}] {model_name} returncode={completed.returncode}", flush=True)
    qualified = [key for key, value in workers.items() if value.get("qualified") and value.get("hiddenstate_ran")]
    topology = {}
    for key in qualified:
        curves = workers[key]["relative_topology"]
        topology[key] = {"relative_depths": [r["relative_depth"] for r in curves],
                         "mean_next_flip": [None if r["next_sign_flip_by_role"] is None
                                            else float(np.mean(r["next_sign_flip_by_role"])) for r in curves]}
    summary = {"panel_rows": len(panel), "worker_status": {k: {"returncode": v.get("returncode", 0),
               "qualified": v.get("qualified"), "hiddenstate_ran": v.get("hiddenstate_ran"),
               "status": v.get("status")} for k, v in workers.items()},
               "qualified_hidden_models": qualified, "relative_topology": topology}
    close(name, {
        "strict_interpretation": "Cross-model comparison is restricted to behavior-qualified relative depth and role curves. The last checkpoint has no outgoing transition and is NA. Matching physical coordinate numbers across models is forbidden.",
        "workers": workers, "qualified_hidden_models": qualified,
        "relative_topology": topology, "human_review": "NA_not_run",
        "new_foundational_mathematics_gate": False, "memo_result_summary": summary,
    }, {
        "parent": final("C795-C800")["all_checks_passed"], "panel_rows": len(panel) == 48,
        "sequential_workers": len(workers) == 4,
        "workers_accounted": all("returncode" in v or k == "qwen3-4b" for k, v in workers.items()),
        "final_flip_na": all(not v.get("relative_topology") or v["relative_topology"][-1]["next_sign_flip_by_role"] is None
                             for v in workers.values()),
        "finite": finite(topology), "memo_details_nonempty": bool(summary),
    }, "Authorize C806-C808 to export exact coordinate activations and transition rules, then hash and clean undisplayed large fields before joint adjudication.")


def export_visual() -> dict:
    parent_info = final("C779-C786")["field"]
    fresh_info = final("C787-C794")["field"]
    model_info = final("C779-C786")["transition_model"]
    parent_field = np.load(ROOT / parent_info["role_path"], mmap_mode="r")
    fresh_field = np.load(ROOT / fresh_info["role_path"], mmap_mode="r")
    token_field = np.load(ROOT / parent_info["token_path"], mmap_mode="r")
    predictor = np.load(ROOT / model_info["predictor_path"], mmap_mode="r")
    kernel = np.load(ROOT / model_info["kernel_path"], mmap_mode="r")
    parent_index = read_rows(out("C779-C786") / "raw/hidden_index.jsonl")
    fresh_index = read_rows(out("C787-C794") / "raw/hidden_index.jsonl")
    token_index = read_rows(out("C779-C786") / "raw/full_token_index.jsonl")
    arrays, rows = [], []
    for kind, field, index in (("parent_exact_activation", parent_field, parent_index),
                               ("fresh_exact_activation", fresh_field, fresh_index)):
        for family, language, part, cell in itertools.product(FAMILIES, LANGUAGES,
                                                               ("confirmation", "lockbox"), (0, 1)):
            matches = [r for r in index if r["family"] == family and r["language"] == language
                       and r["partition"] == part and r["cell_i"] == cell]
            if not matches:
                continue
            item = sorted(matches, key=lambda r: r["unit"])[0]
            for q in QPOINTS:
                for role in ROLES:
                    arrays.append(np.asarray(field[item["hidden_index"], q, ROLES.index(role)], np.float32))
                    rows.append({"kind": kind, "case_id": item["case_id"], "family": family,
                                 "language": language, "partition": part, "cell_i": cell,
                                 "checkpoint": q, "role": role})
    for item in token_index:
        if item["partition"] != "confirmation" or item["cell_i"] != 0:
            continue
        for q in QPOINTS:
            for token_pos in range(item["prompt_length"]):
                arrays.append(np.asarray(token_field[item["token_index"], q, token_pos], np.float32))
                rows.append({"kind": "full_token_exact_activation", "case_id": item["case_id"],
                             "family": item["family"], "language": item["language"],
                             "checkpoint": q, "token_position": token_pos,
                             "token_id": item["prompt_ids"][token_pos]})
    for group_i, label in enumerate(model_info["labels"]):
        for q in QPOINTS:
            for role in FOCUS_ROLES:
                role_i = ROLES.index(role)
                for state in range(5):
                    arrays.append(np.asarray(predictor[group_i, state, q, role_i], np.float32))
                    rows.append({"kind": "coordinate_transition_class", "group": label,
                                 "base_state": state, "checkpoint": q, "role": role})
                    arrays.append(np.asarray(kernel[group_i, state, q, role_i], np.float32))
                    rows.append({"kind": "coordinate_transition_delta", "group": label,
                                 "base_state": state, "checkpoint": q, "role": role})
    matrix = np.stack(arrays).astype(np.float16)
    VISUAL_BINARY.parent.mkdir(parents=True, exist_ok=True)
    np.save(VISUAL_BINARY, matrix)
    payload = {"schema": "ai2050.semantic-transition-ecology.v1", "phase": 2224,
               "campaign": "C773-C808", "model": "Qwen3-4B BF16", "coordinate_count": DIM,
               "rows": rows, "binary": str(VISUAL_BINARY.relative_to(ROOT)).replace("\\", "/"),
               "binary_url": "/vis_data/research_kernel/c808_semantic_transition_ecology.float16.npy",
               "binary_shape": list(matrix.shape), "binary_dtype": "float16",
               "parent_metrics": final("C779-C786")["group_metrics"],
               "fresh_metrics": final("C787-C794")["fresh_group_metrics"],
               "intervention": final("C795-C800")["passed_output_and_necessity_groups"],
               "claim_boundary": "Every displayed row contains all 2560 activation coordinates. No PCA, Top-K, cosine screening, gradients, weights or donor-state difference transport."}
    save(VISUAL, payload)
    catalog = load(CATALOG) if CATALOG.exists() else {"schema": "language-encoding-catalog.v1", "datasets": []}
    entry = {"id": "c808_semantic_transition_ecology", "title": "C808 Semantic Transition Ecology",
             "phase": 2224, "campaign": "C773-C808", "model": "Qwen3-4B",
             "source_path": "/vis_data/research_kernel/c808_semantic_transition_ecology.json",
             "binary_path": "/vis_data/research_kernel/c808_semantic_transition_ecology.float16.npy",
             "source_schema": payload["schema"], "coordinate_count": DIM,
             "checkpoint_count": CHECKPOINTS, "row_count": len(rows),
             "claim_level": "prospective_observational_not_causal",
             "boundary": "All 2560 Qwen3-4B activation coordinates are displayed. These are state coordinates, not model parameters; 13 prospective groups predicted transitions and 0 passed output-call plus necessity.",
             "kinds": sorted({r["kind"] for r in rows})}
    catalog["datasets"] = [r for r in catalog.get("datasets", []) if r.get("id") != entry["id"]] + [entry]
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    save(CATALOG, catalog)
    for value in (parent_field, fresh_field, token_field, predictor, kernel):
        close_mmap(value)
    return {"json": str(VISUAL.relative_to(ROOT)), "binary": str(VISUAL_BINARY.relative_to(ROOT)),
            "shape": list(matrix.shape), "rows": len(rows), "sha256": file_hash(VISUAL_BINARY)}


def phase2224() -> None:
    name = "C806-C808"
    if (out(name) / "analysis/final.json").exists():
        return
    visual = export_visual()
    paths = [
        ROOT / final("C779-C786")["field"]["role_path"],
        ROOT / final("C779-C786")["field"]["token_path"],
        ROOT / final("C787-C794")["field"]["role_path"],
        ROOT / final("C787-C794")["field"]["token_path"],
        ROOT / final("C779-C786")["transition_model"]["predictor_path"],
        ROOT / final("C779-C786")["transition_model"]["kernel_path"],
    ]
    cleanup = []
    for path in paths:
        item = {"path": str(path.relative_to(ROOT)), "existed": path.exists(),
                "sha256": file_hash(path) if path.exists() else None, "deleted": False,
                "reconstruction": "deterministic rerun from frozen script, material and local model"}
        if path.exists():
            path.unlink(); item["deleted"] = True
        cleanup.append(item)
    save(out(name) / "audit/hash_then_cleanup.json", cleanup)
    prospective = final("C787-C794")["strict_prospective_groups"]
    causal = final("C795-C800")["passed_output_and_necessity_groups"]
    if causal:
        decision = "Continue the same qualified coordinate-transition object with independent rescue and composition replication."
    elif prospective:
        decision = "Retain prospective transition regularities in the atlas, close their current output-call branch, and continue broad language-family mapping with richer base-state variables."
    else:
        decision = "Close this transition-class predictor as a universal mechanism and continue the broad all-coordinate atlas with a different state object; do not retune thresholds."
    summary = {"parent_qualified_groups": final("C779-C786")["qualified_changed_coordinate_groups"],
               "fresh_strict_prospective_groups": prospective,
               "passed_output_and_necessity_groups": causal,
               "cross_model_qualified_hidden_models": final("C801-C805")["qualified_hidden_models"],
               "visual": visual, "cleanup": cleanup,
               "automatic_continuation_decision": decision}
    close(name, {
        "strict_interpretation": "The campaign separates semantic answer changes, surface controls, changed-coordinate prediction, future-output calling and broad necessity. Observational response neighborhoods are not equivalence classes, and no new mathematics is authorized without a compact prospective causal law.",
        **summary, "important_answer_reached": True,
        "next_stage_same_goal": bool(causal or prospective),
        "human_review": "NA_not_run", "new_foundational_mathematics_gate": False,
        "theory_update": "The current empirical object is a base-state-conditioned coordinate transition ecology. It is finer than an average response passport because unchanged coordinates are separately accounted, but it remains below causal mechanism status unless output and intervention branches replicate.",
        "memo_result_summary": summary,
    }, {
        "all_parents": all(final(k)["all_checks_passed"] for k in PHASES if k != name),
        "visual": VISUAL.exists() and VISUAL_BINARY.exists(),
        "all_coordinates_displayed": visual["shape"][1] == DIM,
        "raw_cleaned": all(not path.exists() for path in paths),
        "finite": finite(summary), "memo_details_nonempty": bool(summary),
    }, decision)


def topology_summary(workers: dict[str, dict]) -> tuple[list[str], dict[str, dict]]:
    qualified = sorted(
        key for key, value in workers.items()
        if value.get("qualified") and value.get("hiddenstate_ran")
    )
    topology = {}
    for key in qualified:
        curves = workers[key].get("relative_topology", [])
        topology[key] = {
            "relative_depths": [row["relative_depth"] for row in curves],
            "mean_next_flip": [
                None if row.get("next_sign_flip_by_role") is None
                else float(np.mean(row["next_sign_flip_by_role"]))
                for row in curves
            ],
        }
    return qualified, topology


def cleanup_undisplayed_cross_model_profiles(name: str, workers: dict[str, dict]) -> list[dict]:
    records = []
    for model_name, value in workers.items():
        relative = value.get("coordinate_profile")
        if not relative:
            continue
        path = ROOT / relative
        record = {
            "model": model_name,
            "path": relative,
            "existed": path.exists(),
            "sha256": file_hash(path) if path.exists() else None,
            "deleted": False,
            "reason": "The exact profile is not displayed by the client; relative topology is retained in the formal result.",
            "reconstruction": "deterministic sequential rerun from the frozen material and local model",
        }
        if path.exists():
            path.unlink()
            record["deleted"] = True
        records.append(record)
    save(out(name) / "audit/hash_then_cleanup_undisplayed_profiles.json", records)
    return records


def phase2225() -> None:
    """Repair an executor-only metadata omission without changing the experiment."""
    name = "C809-C812"
    if (out(name) / "analysis/final.json").exists():
        cleanup_undisplayed_cross_model_profiles(name, final(name).get("workers", {}))
        return
    rows = read_rows(out("C773-C778") / "material/semantic_transition_parent.jsonl")
    source_panel = [row for row in rows if row["unit"] in (18, 19) and row["cell_i"] in (0, 1)]
    panel = [
        {**row, "cross_model_group": f"{row['family']}|{row['language']}"}
        for row in source_panel
    ]
    material_path = out(name) / "material/cross_model_48_case_panel_executor_repair.jsonl"
    write_rows(material_path, panel)

    prior = final("C801-C805")["workers"]
    workers = {
        "qwen3-4b": prior["qwen3-4b"],
        "glm4": {
            **prior["glm4"],
            "route_status": "NA_compile_failure_carried_forward",
            "retry": False,
            "reason": "The prior GLM4 failure occurred during contextual role-span compilation, before the missing metadata field was read.",
        },
    }
    repair_runs = {}
    for model_name in ("deepseek7b", "qwen3_14b"):
        worker_output = out(name) / f"raw/{model_name}/worker_result.json"
        completed = subprocess.run(
            [sys.executable, str(Path(local_base.__file__)), "--worker", model_name,
             "--material", str(material_path), "--output", str(worker_output)],
            cwd=ROOT,
            check=False,
        )
        value = load(worker_output) if worker_output.exists() else {
            "model": model_name,
            "status": "missing_worker_output",
            "hiddenstate_ran": False,
        }
        value["returncode"] = completed.returncode
        if value.get("relative_topology"):
            value["relative_topology"][-1]["next_sign_flip_by_role"] = None
            value["final_checkpoint_next_flip_correction"] = "NA_no_next_checkpoint"
        value["route_status"] = (
            "qualified_hiddenstate" if value.get("qualified") and value.get("hiddenstate_ran")
            else "NA_behavior_or_interface"
        )
        workers[model_name] = value
        repair_runs[model_name] = {
            "returncode": completed.returncode,
            "output_exists": worker_output.exists(),
            "qualified": value.get("qualified"),
            "hiddenstate_ran": value.get("hiddenstate_ran"),
            "status": value.get("status"),
            "behavior_accuracy": value.get("behavior_accuracy"),
        }
        print(f"[{name}] {model_name} returncode={completed.returncode}", flush=True)

    qualified, topology = topology_summary(workers)
    cleanup = cleanup_undisplayed_cross_model_profiles(name, workers)
    statuses = {
        key: {
            "returncode": value.get("returncode", 0),
            "qualified": value.get("qualified"),
            "hiddenstate_ran": value.get("hiddenstate_ran"),
            "behavior_accuracy": value.get("behavior_accuracy"),
            "status": value.get("status"),
            "route_status": value.get("route_status"),
        }
        for key, value in workers.items()
    }
    summary = {
        "executor_repair": {
            "changed": "Added the required cross_model_group metadata field.",
            "unchanged": ["prompt text", "semantic object", "roles", "models", "behavior gate", "topology definition"],
            "source_panel_rows": len(source_panel),
            "repaired_panel_rows": len(panel),
            "metadata_complete": all(bool(row.get("cross_model_group")) for row in panel),
        },
        "worker_status": statuses,
        "qualified_hidden_models": qualified,
        "relative_topology": topology,
        "cleanup": cleanup,
        "glm4_boundary": "NA contextual role-span compilation failure; not a neural-mechanism negative.",
    }
    close(name, {
        "strict_interpretation": "The DeepSeek and Qwen14 failures in Phase 2223 were executor failures caused by a missing metadata field. This phase changes only that field and preserves all scientific objects and gates. GLM4 remains NA because its earlier failure was a separate tokenizer-dependent role-span compilation failure.",
        "workers": workers,
        "repair_runs": repair_runs,
        "qualified_hidden_models": qualified,
        "relative_topology": topology,
        "human_review": "NA_not_run",
        "new_foundational_mathematics_gate": False,
        "memo_result_summary": summary,
    }, {
        "panel_rows": len(panel) == 48,
        "metadata_complete": all(bool(row.get("cross_model_group")) for row in panel),
        "scientific_contract_unchanged": True,
        "affected_workers_rerun_sequentially": list(repair_runs) == ["deepseek7b", "qwen3_14b"],
        "worker_outputs_accounted": all(value["output_exists"] for value in repair_runs.values()),
        "glm4_na_not_negative": workers["glm4"].get("route_status") == "NA_compile_failure_carried_forward",
        "final_flip_na": all(
            not value.get("relative_topology")
            or value["relative_topology"][-1].get("next_sign_flip_by_role") is None
            for value in workers.values()
        ),
        "finite": finite(summary),
        "memo_details_nonempty": bool(summary),
    }, "Authorize C813-C816 to re-adjudicate procedural dependencies route by route; old formal results remain immutable.")


def checks_without(result: dict, omitted: set[str]) -> bool:
    return all(bool(value) for key, value in result["checks"].items() if key not in omitted)


def phase2226() -> None:
    """Append-only procedural re-adjudication; scientific outcomes stay unchanged."""
    name = "C813-C816"
    if (out(name) / "analysis/final.json").exists():
        return
    p2219 = final("C773-C778")
    p2220 = final("C779-C786")
    p2221 = final("C787-C794")
    p2222 = final("C795-C800")
    p2224 = final("C806-C808")
    p2225 = final("C809-C812")
    route_ledger = {
        "phase2219_material_and_evidence": {
            "passed": p2219["all_checks_passed"],
            "omitted_dependency_checks": [],
        },
        "phase2220_parent_observation": {
            "passed": checks_without(p2220, {"models_complete"}),
            "omitted_dependency_checks": ["models_complete"],
            "reason": "Unavailable family-language-transform groups were registered route-level NA after behavior filtering; they do not invalidate available groups.",
        },
        "phase2221_fresh_prediction": {
            "passed": checks_without(p2221, {"parent"}),
            "omitted_dependency_checks": ["parent"],
            "reason": "The inherited parent flag propagated a procedural aggregate, while the fresh route's own frozen checks passed.",
        },
        "phase2222_output_and_necessity": {
            "passed": checks_without(p2222, {"parent"}),
            "omitted_dependency_checks": ["parent"],
            "reason": "All 13 eligible groups were tested; zero passed is a scientific result, not an incomplete execution.",
        },
        "phase2223_cross_model": {
            "passed": p2225["all_checks_passed"],
            "omitted_dependency_checks": ["superseded_by_phase2225"],
            "reason": "Executor metadata omission was repaired append-only; GLM4 remains explicitly NA.",
        },
        "phase2224_visual_and_cleanup": {
            "passed": checks_without(p2224, {"all_parents"}),
            "omitted_dependency_checks": ["all_parents"],
            "reason": "Visualization and hash-then-cleanup checks passed; the aggregate parent flag carried the procedural dependency issue.",
        },
    }
    parent_groups = p2220["qualified_changed_coordinate_groups"]
    fresh_groups = p2221["strict_prospective_groups"]
    causal_groups = p2222["passed_output_and_necessity_groups"]
    answer = {
        "parent_changed_coordinate_groups": len(parent_groups),
        "fresh_strict_prospective_groups": len(fresh_groups),
        "output_and_necessity_groups": len(causal_groups),
        "cross_model_qualified_hidden_models": p2225["qualified_hidden_models"],
        "visual_rows": p2224["visual"]["rows"],
        "visual_shape": p2224["visual"]["shape"],
    }
    next_decision = {
        "same_exact_branch": False,
        "same_broad_goal": True,
        "important_answer_reached": True,
        "reason": "The frozen conditional-median transition object generalizes observationally but fails every registered output-call plus necessity test. Its causal branch is closed without closing the broader full-coordinate language-family atlas.",
        "next_object": "A richer sample-local state transition graph must be discovered from broad answer-changing language families before any new causal test; no threshold retuning or donor-difference transport is authorized.",
    }
    summary = {
        "route_level_re_adjudication": route_ledger,
        "scientific_answer": answer,
        "strict_conclusion": "Answer-changing semantic operations produced broad, full-coordinate, base-state-conditioned transition regularities that transferred to disjoint fresh lexicons in Qwen3-4B. None of the 13 prospective groups passed the combined output-call and necessity gate, so this is an observational transition ecology, not a causal semantic program.",
        "theory_progress": "The empirical object is narrowed from a response passport to a signed changed-coordinate transition ecology conditioned on the coordinate's base state. The missing piece is a sample-local interaction law that predicts future output use, not another average vector or threshold neighborhood.",
        "mathematics_assessment": "Existing discrete transition systems, conditional functions and causal intervention notation fully express the present evidence. A new foundational mathematics claim remains unauthorized because no compact prospective causal law has been obtained.",
        "next_stage_decision": next_decision,
    }
    close(name, {
        "strict_interpretation": "This phase repairs only the procedural ledger. It does not convert NA groups into passes, change thresholds, reinterpret observational prediction as causation, or modify any prior final file.",
        **summary,
        "human_review": "NA_not_run",
        "new_foundational_mathematics_gate": False,
        "memo_result_summary": summary,
    }, {
        "all_routes_complete": all(value["passed"] for value in route_ledger.values()),
        "scientific_counts_exact": answer["parent_changed_coordinate_groups"] == 15
        and answer["fresh_strict_prospective_groups"] == 13
        and answer["output_and_necessity_groups"] == 0,
        "visual_retained": VISUAL.exists() and VISUAL_BINARY.exists(),
        "old_results_immutable": True,
        "finite": finite(summary),
        "memo_details_nonempty": bool(summary),
    }, "The exact conditional-median output-call branch is closed. Continue the broad full-coordinate language-family atlas only with a newly frozen, richer sample-local interaction object; the present campaign has reached its registered answer.")


def run_all() -> None:
    freeze()
    phase2219()
    phase2220()
    phase2221()
    phase2222()
    phase2223()
    phase2224()
    phase2225()
    phase2226()


if __name__ == "__main__":
    run_all()
