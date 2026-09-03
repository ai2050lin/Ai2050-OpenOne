#!/usr/bin/env python3
"""C710-C744 response-equivalence atlas campaign.

The campaign observes embeddings, every post-block HiddenState, final norm,
and logits. It does not read attention/MLP internals, weights, or gradients,
and it does not use PCA, Top-K selection, cosine screening, or donor-state
difference transport.
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import re
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
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c744_response_equivalence_coordinate_atlas.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c744_response_equivalence_coordinate_atlas.float16.npy"
sys.path.insert(0, str(TESTS))

import model_utils
import phase2105_c571_c589_scope_program_algebra_campaign as scope
import phase2190_c656_c669_absolute_coordinate_grammar_campaign as local_base
import phase2200_c684_c709_unified_relation_response_campaign as previous


PHASES = {
    "C710-C714": (2205, "evidence_audit_and_six_family_master_contract"),
    "C715-C720": (2206, "dual_behavior_and_full_coordinate_field_atlas"),
    "C721-C726": (2207, "response_passports_and_observational_equivalence_graph"),
    "C727-C733": (2208, "six_family_all_coordinate_local_response_graph"),
    "C734-C739": (2209, "output_call_and_sequential_cross_model_topology"),
    "C740-C744": (2210, "joint_adjudication_visualization_and_cleanup"),
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower().replace('-', '_')}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

DIM = 2560
CHECKPOINTS = 38
ROLES = local_base.ROLES
QPOINTS = (0, 8, 16, 24, 32, 37)
FAMILIES = (
    "recursive_knowledge",
    "nested_attitude",
    "voice_negation",
    "temporal_update",
    "coreference_binding",
    "translation_route",
)
LANGUAGES = ("en", "zh")
TRANSFORMS = (1, 2, 3)
UNITS = 24
BEHAVIOR_GATE = 0.75
PASSPORT_GAIN_GATE = 0.005
LOCAL_GAIN_GATE = 0.02
OUTPUT_GAIN_GATE = 0.05

NAMES_A_EN = previous.NAMES_A_EN
NAMES_B_EN = previous.NAMES_B_EN
NAMES_A_ZH = previous.NAMES_A_ZH
NAMES_B_ZH = previous.NAMES_B_ZH
OBJECTS_EN = previous.OBJECTS_EN
OBJECTS_ZH = previous.OBJECTS_ZH
DISTRACT_EN = previous.DISTRACT_EN
DISTRACT_ZH = previous.DISTRACT_ZH
FRENCH = (
    "pomme", "banane", "poire", "pêche", "raisin", "citron", "orange", "prune",
    "cerise", "mangue", "melon", "noix de coco", "carotte", "pomme de terre",
    "tomate", "oignon", "chou", "haricot", "pois", "maïs", "riz", "blé", "pain", "fromage",
)
CELL_NAMES = ("base", "paraphrase", "structural", "composite")


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


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def partition(unit: int) -> str:
    return "discovery" if unit < 12 else ("confirmation" if unit < 18 else "lockbox")


def lexicon(unit: int, language: str) -> dict:
    if unit < 12:
        types_en, types_zh = ("fruit", "food", "physical object"), ("水果", "食物", "实体物品")
    elif unit < 20:
        types_en, types_zh = ("vegetable", "food", "physical object"), ("蔬菜", "食物", "实体物品")
    else:
        types_en, types_zh = ("staple", "food", "physical object"), ("主食", "食物", "实体物品")
    if language == "en":
        return {
            "a": NAMES_A_EN[unit], "b": NAMES_B_EN[unit], "x": OBJECTS_EN[unit],
            "y": DISTRACT_EN[unit], "types": types_en, "fr": FRENCH[unit],
            "wrong_fr": FRENCH[(unit + 7) % UNITS],
        }
    return {
        "a": NAMES_A_ZH[unit], "b": NAMES_B_ZH[unit], "x": OBJECTS_ZH[unit],
        "y": DISTRACT_ZH[unit], "types": types_zh, "fr": FRENCH[unit],
        "wrong_fr": FRENCH[(unit + 7) % UNITS],
    }


def family_case(family: str, language: str, unit: int, cell_i: int) -> dict:
    u = lexicon(unit, language)
    truth = unit % 2 == 0
    a, b, x, y = u["a"], u["b"], u["x"], u["y"]
    t1, t2, t3 = u["types"]
    surface = "record" if cell_i in (0, 2) else "dialogue"
    if family == "recursive_knowledge":
        target = t3 if truth else y
        if language == "en":
            facts = [f"the {x} is a kind of {t1}", f"{t1} is a kind of {t2}", f"{t2} is a kind of {t3}"]
            if cell_i == 1: facts = [facts[1], facts[0], facts[2]]
            if cell_i == 2: facts = list(reversed(facts))
            if cell_i == 3: facts += [f"the {x} is also directly listed as a kind of {t3}", f"the {y} is on an unrelated inventory"]
            core = "A verified registry states: " + "; ".join(facts) + f". Based only on the registry, is the {x} a kind of {target}?"
            relation = "kind of"
        else:
            facts = [f"{x}是一种{t1}", f"{t1}是一种{t2}", f"{t2}是一种{t3}"]
            if cell_i == 1: facts = [facts[1], facts[0], facts[2]]
            if cell_i == 2: facts = list(reversed(facts))
            if cell_i == 3: facts += [f"{x}还被直接登记为一种{t3}", f"{y}位于无关清单中"]
            core = "一份经过核验的分类记录写道：" + "；".join(facts) + f"。只根据这份记录，{x}是一种{target}吗？"
            relation = "是一种"
        roles = {"primary": x, "secondary": t1, "relation": relation, "context": t3, "query": x}
        factors = {"path_depth": 3, "shortcut": int(cell_i == 3), "order_variant": cell_i}
    elif family == "nested_attitude":
        target = x if truth else y
        if language == "en":
            variants = (
                f"{a} remembered that {b} ate the {x}",
                f"During the hearing, {a} remembered that {b} ate the {x}",
                f"{a} remembered that the {x} was eaten by {b}",
                f"The record explicitly confirms that {a} remembered that {b} ate the {x}",
            )
            core = f"A verified memory record states: {variants[cell_i]}. Does the record say that {a} remembered that {b} ate the {target}?"
            relation = "remembered"
        else:
            variants = (
                f"{a}记得{b}吃了{x}",
                f"听证记录表明，{a}记得{b}吃了{x}",
                f"{a}记得{x}被{b}吃了",
                f"记录明确确认，{a}记得{b}吃了{x}",
            )
            core = f"一份经过核验的记忆记录写道：{variants[cell_i]}。这份记录是否表明{a}记得{b}吃了{target}？"
            relation = "记得"
        roles = {"primary": a, "secondary": b, "relation": relation, "context": x, "query": target}
        factors = {"attitude": 1, "event_voice": int(cell_i == 2), "packaging": cell_i}
    elif family == "voice_negation":
        target = x if truth else y
        if language == "en":
            variants = (
                f"{b} ate the {x}",
                f"According to the witness, {b} ate the {x}",
                f"the {x} was eaten by {b}",
                f"it is not the case that {b} failed to eat the {x}",
            )
            core = f"A verified event record states that {variants[cell_i]}. Is it true from the record that {b} ate the {target}?"
            relation = "ate" if cell_i != 3 else "eat"
        else:
            variants = (
                f"{b}吃了{x}",
                f"证人确认{b}吃了{x}",
                f"{x}被{b}吃了",
                f"并非{b}没有吃{x}",
            )
            core = f"一份经过核验的事件记录写道：{variants[cell_i]}。根据记录，{b}吃了{target}吗？"
            relation = "吃了" if cell_i != 3 else "吃"
        roles = {"primary": b, "secondary": x, "relation": relation, "context": x, "query": target}
        factors = {"voice": int(cell_i == 2), "double_negation": int(cell_i == 3), "packaging": cell_i}
    elif family == "temporal_update":
        target = x if truth else y
        old_item = y
        if language == "en":
            variants = (
                f"Initially {a} stored the {old_item}. Later, the current record replaced that entry: {a} stored the {x}",
                f"The latest entry says {a} stored the {x}; an older entry had listed the {old_item}",
                f"After first listing the {old_item}, the registry was updated so that {a} currently stores the {x}",
                f"The old {old_item} entry is obsolete. The current entry states that {a} stored the {x}",
            )
            core = f"A verified update log states: {variants[cell_i]}. Is the current stored item for {a} the {target}?"
            relation = "current"
        else:
            variants = (
                f"最初{a}存放的是{old_item}，后来当前记录覆盖了旧条目：{a}存放{x}",
                f"最新条目写着{a}存放{x}，更早的条目曾写{old_item}",
                f"登记表先记录{old_item}，随后更新为{a}当前存放{x}",
                f"旧的{old_item}条目已经失效，当前条目写明{a}存放{x}",
            )
            core = f"一份经过核验的更新日志写道：{variants[cell_i]}。{a}当前存放的物品是{target}吗？"
            relation = "当前"
        roles = {"primary": a, "secondary": old_item, "relation": relation, "context": x, "query": target}
        factors = {"update": 1, "order_variant": cell_i, "obsolete_control": int(cell_i == 3)}
    elif family == "coreference_binding":
        referent = a if truth else b
        if language == "en":
            variants = (
                f"{a} told {b}, 'I stored the {x}.'",
                f"Speaking to {b}, {a} said, 'I stored the {x}.'",
                f"{b} heard {a} say, 'I stored the {x}.'",
                f"{b} was told by {a}, 'I stored the {x}.'",
            )
            core = f"A verified quotation record states: {variants[cell_i]} In that quotation, does the word I refer to {referent}?"
            relation = "I"
        else:
            variants = (
                f"{a}对{b}说：“我存放了{x}。”",
                f"{a}在和{b}交谈时说：“我存放了{x}。”",
                f"{b}听见{a}说：“我存放了{x}。”",
                f"{b}收到{a}的话：“我存放了{x}。”",
            )
            core = f"一份经过核验的引语记录写道：{variants[cell_i]}在这段引语中，“我”指的是{referent}吗？"
            relation = "我"
        roles = {"primary": a, "secondary": b, "relation": relation, "context": x, "query": referent}
        factors = {"quotation": 1, "voice": int(cell_i == 3), "packaging": cell_i}
    elif family == "translation_route":
        target_fr = u["fr"] if truth else u["wrong_fr"]
        if language == "en":
            variants = (
                f"the English word {x} is written in French as {u['fr']}",
                f"in French, the English item {x} is called {u['fr']}",
                f"the French entry {u['fr']} corresponds to the English item {x}",
                f"a translator maps the English item {x} to {u['fr']} in French",
            )
            core = f"A verified bilingual glossary states that {variants[cell_i]}. According to the glossary, is the French form of {x} {target_fr}?"
            relation = "French"
        else:
            variants = (
                f"英文词{x}的法语写法是{u['fr']}",
                f"在法语中，英文词{x}写作{u['fr']}",
                f"法语条目{u['fr']}对应英文词{x}",
                f"翻译表把英文词{x}映射为法语{u['fr']}",
            )
            core = f"一份经过核验的双语词表写道：{variants[cell_i]}。根据词表，{x}的法语形式是{target_fr}吗？"
            relation = "法语"
        roles = {"primary": x, "secondary": u["fr"], "relation": relation, "context": target_fr, "query": x}
        factors = {"target_language": "fr", "direction_variant": cell_i}
    else:
        raise KeyError(family)

    yes, no = (("Yes", "No") if language == "en" else ("是", "否"))
    correct, wrong = (yes, no) if truth else (no, yes)
    gold_position = ((unit // 2) + cell_i + int(language == "zh")) % 2
    options = f"(A) {correct} (B) {wrong}" if gold_position == 0 else f"(A) {wrong} (B) {correct}"
    prompt = f"{core} {options}. Reply with only A or B." if language == "en" else f"{core} {options}。只回答A或B。"
    free_prompt = f"{core} Answer only Yes or No." if language == "en" else f"{core} 只回答“是”或“否”。"
    return {
        "case_id": f"c710-{family}-{language}-u{unit:02d}-{CELL_NAMES[cell_i]}",
        "panel": "response_equivalence_atlas", "family": family, "query_operation": family,
        "operation_type": family, "operation_domain": f"{family}:{CELL_NAMES[cell_i]}",
        "language": language, "surface": surface, "cell": CELL_NAMES[cell_i], "cell_i": cell_i,
        "transform_id": cell_i, "unit": unit, "partition": partition(unit), "truth": truth,
        "correct_answer": correct, "wrong_answer": wrong, "gold_position": gold_position,
        "prompt_core": core, "prompt": prompt, "free_prompt": free_prompt,
        "role_values": roles, "factors": factors,
        "semantic_graph": {
            "external_family": family, "transform": CELL_NAMES[cell_i], "language": language,
            "labels_are_external_coordinates_not_internal_modules": True,
        },
    }


def material() -> list[dict]:
    return [family_case(family, language, unit, cell_i)
            for family, language, unit, cell_i in itertools.product(FAMILIES, LANGUAGES, range(UNITS), range(4))]


def protocol(name: str) -> dict:
    common = {
        "model_order": ["qwen3-4b", "glm4", "deepseek7b", "qwen3-14b"],
        "camera": "embedding + all post-block HiddenStates + final norm + logits; all physical activation coordinates",
        "forbidden": ["attention", "MLP", "weights", "gradients", "PCA", "Top-K", "cosine screening", "donor HiddenState difference transport"],
        "human_review": "NA_not_run; blank reviewer template is saved",
        "failure_policy": "route-level missingness; every registered family continues independently",
        "evidence_unit": "held-out concept/unit, not layers or coordinates",
    }
    details = {
        "C710-C714": {"object": "audit Phase2200-2204 and freeze six-family bilingual graph material", "rows": 1152},
        "C715-C720": {"object": "same-row candidate+generation behavior and full-coordinate role/full-token capture", "behavior_gate": BEHAVIOR_GATE},
        "C721-C726": {"object": "paired coordinate-state response passports and observational response-equivalence candidates", "gain_gate": PASSPORT_GAIN_GATE},
        "C727-C733": {"object": "q24 relation to q25/final boundary complete coordinate local response", "local_gain_gate": LOCAL_GAIN_GATE},
        "C734-C739": {"object": "output margin/generation call and sequential cross-model relative topology", "output_gain_gate": OUTPUT_GAIN_GATE},
        "C740-C744": {"object": "joint scientific adjudication, exact-coordinate visualization, hash-then-clean"},
    }
    return {**common, **details[name]}


def freeze() -> None:
    for name in PHASES:
        for part in ("protocol", "material", "behavior", "raw", "analysis", "audit", "external"):
            (out(name) / part).mkdir(parents=True, exist_ok=True)
        prereg = out(name) / "protocol/preregistration.json"
        if not prereg.exists():
            save(prereg, {"phase": PHASES[name][0], "campaign": name,
                           "timestamp_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol(name)})


TITLES = {
    "C710-C714": "证据重裁与六族响应图谱总合同",
    "C715-C720": "逐行双行为门、全角色与全token坐标场",
    "C721-C726": "逐坐标响应护照与观察等价图",
    "C727-C733": "六族全坐标局部响应传动图",
    "C734-C739": "输出调用与三模型顺序相对图谱",
    "C740-C744": "大阶段联合裁决、可视化与清理",
}
FORMULAS = {
    "C710-C714": "$$\n\\mathfrak L=(\\mathcal G,\\mathcal O,\\Sigma,\\mathcal Q),\\qquad o:\\mathcal G_{\\mathrm{dom}(o)}\\rightharpoonup\\mathcal G\n$$",
    "C715-C720": "$$\nG_{row}=\\mathbf 1[\\hat y_{cand}=y\\land\\hat y_{gen}=y],\\qquad \\mathbb H(x)=\\{H_{q,t,j}(x)\\}_{q,t,j}\n$$",
    "C721-C726": "$$\nP_o(q,r,j)=\\operatorname{mode}_{x\\in discovery}(Z_{base}(q,r,j),Z_o(q,r,j))\n$$\n$$\nG_o=A(P_o)-\\max\\{A(P_{shared}),A(P_{wrong}),A(P_{shift})\\}\n$$",
    "C727-C733": "$$\nJ^{(x)}_{j\\to k}=\\frac{H_k(H_j+\\epsilon_j)-H_k(H_j-\\epsilon_j)}{2\\epsilon_j}\n$$",
    "C734-C739": "$$\nG_{out}=\\Delta m_{aligned}-\\max(\\Delta m_{opposite},\\Delta m_{shift},0)\n$$",
    "C740-C744": "$$\nH\\sim_{\\mathcal T}H'\\iff\\forall o,Q,\\ d_Q(\\Delta_{o,Q}(H),\\Delta_{o,Q}(H'))\\le\\epsilon\n$$",
}
EXAMPLES = {
    "C710-C714": "同一母图同时登记类型链、嵌套态度、语态否定、时间更新、引语共指和概念到法语词形的受控变换。",
    "C715-C720": "每一行必须同时答对A/B候选和Yes/No自由生成，才进入该行的HiddenState图谱。",
    "C721-C726": "把同一概念的base状态与三种合法表面/结构变换逐坐标编码为状态转移护照，不搬运差分向量。",
    "C727-C733": "六族中英文各取confirmation/lockbox锚点，逐一拨动q24关系角色全部2560坐标。",
    "C734-C739": "坐标影响方向必须击败反号和循环错位，并检查自由生成；三个额外模型逐个加载。",
    "C740-C744": "将embedding、多个HiddenState检查点、逐token坐标和局部响应保留为客户端可索引图谱，其余大场记录哈希后删除。",
}


def append_memo(name: str, result: dict) -> None:
    phase = PHASES[name][0]
    marker = f"## Phase {phase}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = f"""

## Phase {phase}: {TITLES[name]} [{stamp}]

**研究边界。** `{name}`。研究对象是一个统一自回归HiddenState场中的外部受控变换，不把六个语言族预设为内部模块。只读取embedding、各block后HiddenState、final norm和logits；保留全部物理激活坐标。独立人类盲评没有运行，严格记为`NA_not_run`。

**运行前冻结合同。**
```json
{json.dumps(load(out(name) / 'protocol/preregistration.json'), ensure_ascii=False, indent=2)}
```

**测试用例。** {EXAMPLES[name]}

**测试原理与数学公式。**
{FORMULAS[name]}

**详细结果与门槛。**
```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**分析与理论进展。** {result.get('strict_interpretation')} 理论主体保持“条件化输出场闭合理论”，组织原则保持“复用—差分—条件化”；响应护照和等价图目前是经验图谱对象，不是已发现的新基础数学。

**问题、硬伤和瓶颈。** 人类自然度仍为NA；机器模板不能替代母语者判断；激活坐标不是模型参数；状态字会离散连续幅值；逐坐标有限差分只属于当前基态和剂量；跨模型粗轮廓不是功能同构；Qwen3-4B的小模型结果不能直接外推；任何微平均都必须回到独立概念单元复核。

**相关文件。** 主脚本`tests/glm5/phase2205_c710_c744_response_equivalence_atlas_campaign.py`；结果目录`{out(name).relative_to(ROOT)}`；裁决`{(out(name) / 'analysis/final.json').relative_to(ROOT)}`。

**严格结论与下一步授权。** {result.get('next_authorization')}
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def close(name: str, body: dict, checks: dict, authorization: str) -> dict:
    result = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "all_checks_passed": bool(checks) and all(bool(value) for value in checks.values()),
        **body, "next_authorization": authorization,
    }
    save(out(name) / "analysis/final.json", result)
    append_memo(name, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                         local_files_only=True, use_fast=False)


def phase2205(rows: list[dict]) -> None:
    name = "C710-C714"
    if (out(name) / "analysis/final.json").exists():
        return
    tokenizer = load_tokenizer()
    compiled = scope.compiler.compile_qwen(tokenizer, rows)
    material_path = out(name) / "material/six_family_bilingual_graph.jsonl"
    compiled_path = out(name) / "material/qwen_compiled.jsonl"
    write_rows(material_path, rows)
    write_rows(compiled_path, compiled)
    widths = [len(row["prompt_ids"]) for row in compiled]
    banned = ("�", "锛", "鏄?", "remembered that did", "eated", "regreted")
    bad_strings = [row["case_id"] for row in rows if any(item in row["prompt"] for item in banned)]
    missing_roles = [{"case_id": row["case_id"], "role": role, "value": value}
                     for row in rows for role, value in row["role_values"].items() if value not in row["prompt_core"]]
    balance = defaultdict(lambda: [0, 0])
    truth_balance = defaultdict(lambda: [0, 0])
    for row in rows:
        key = f"{row['family']}|{row['language']}|{row['partition']}"
        balance[key][row["gold_position"]] += 1
        truth_balance[key][int(row["truth"])] += 1
    concepts = {part: {row["role_values"]["primary"] for row in rows if row["partition"] == part}
                for part in ("discovery", "confirmation", "lockbox")}
    overlap = sorted((concepts["discovery"] & concepts["confirmation"]) |
                     (concepts["discovery"] & concepts["lockbox"]) |
                     (concepts["confirmation"] & concepts["lockbox"]))
    zero_models = {
        "always_A": float(np.mean([row["gold_position"] == 0 for row in rows])),
        "always_B": float(np.mean([row["gold_position"] == 1 for row in rows])),
        "always_yes_truth_prior": float(np.mean([row["truth"] for row in rows])),
    }
    machine = {
        "rows": len(rows), "families": {family: sum(row["family"] == family for row in rows) for family in FAMILIES},
        "languages": {language: sum(row["language"] == language for row in rows) for language in LANGUAGES},
        "partitions": {part: sum(row["partition"] == part for row in rows) for part in concepts},
        "candidate_balance": dict(balance), "truth_balance": dict(truth_balance), "zero_models": zero_models,
        "bad_strings": bad_strings, "missing_roles": missing_roles, "cross_partition_primary_overlap": overlap,
        "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "human_review": "NA_not_run",
    }
    save(out(name) / "audit/machine_material_audit.json", machine)
    write_rows(out(name) / "external/human_blind_review_template.jsonl", [
        {"case_id": row["case_id"], "naturalness_1_5": None, "semantic_uniqueness_0_1": None,
         "answerability_0_1": None, "paraphrase_equivalence_0_1": None, "reviewer": None}
        for row in rows if row["partition"] == "lockbox"
    ])
    evidence_audit = {
        "retained": [
            "Phase2200 established six qualified external behavior slices and a complete six-role activation field.",
            "Phase2201 found strong shared coordinate dynamics but no >=0.01 relation/program increment (0/3).",
            "Phase2202 measured complete local coordinate response matrices but found no operation-specific or output-call group.",
            "Phase2203 qualified GLM4 and Qwen3-14B for coarse relative-depth observation; no functional isomorphism was established.",
        ],
        "corrections": [
            "The attachment limited to the Phase2200 opening is superseded by repository finals for Phase2201-2204.",
            "Activation coordinates are forward-pass state coordinates, not model parameters.",
            "Response equivalence, switching systems, MDL and hypergraphs remain candidate formalisms, not discovered laws.",
            "The next object is an observational response-passport graph first; causal equivalence requires later intervention closure.",
        ],
    }
    close(name, {
        "strict_interpretation": "The six families are external transformation coordinates within one autoregressive system. The material and compiler are valid for machine execution, while human naturalness remains explicitly missing.",
        "evidence_audit": evidence_audit, "machine_material_audit": machine,
        "material_sha256": file_hash(material_path), "compiled_sha256": file_hash(compiled_path),
        "human_review": "NA_not_run", "new_foundational_mathematics_gate": False,
    }, {
        "parent": previous.final("C705-C709")["all_checks_passed"], "rows": len(rows) == 1152,
        "unique": len({row["case_id"] for row in rows}) == len(rows),
        "compiler": len(compiled) == len(rows) and all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "utf8_and_morphology": not bad_strings, "surface_roles": not missing_roles,
        "candidate_balance": all(value[0] == value[1] for value in balance.values()),
        "truth_balance": all(value[0] == value[1] for value in truth_balance.values()),
        "zero_models": all(abs(value - 0.5) < 1e-12 for value in zero_models.values()),
        "partition_isolation": not overlap, "width": max(widths) <= 220,
    }, "Authorize C715-C720 to run same-row candidate and free-generation qualification before any HiddenState claim.")


def capture_fields(model, device, compiled: list[dict], candidate: dict, generated: dict,
                   qualified: set[str]) -> tuple[list[dict], list[dict], dict]:
    name = "C715-C720"
    selected = [row for row in compiled if f"{row['family']}|{row['language']}" in qualified
                and candidate[row["case_id"]]["correct"] and generated[row["case_id"]]["correct"]]
    panel_ids = set()
    for family, language, part in itertools.product(FAMILIES, LANGUAGES, ("discovery", "confirmation", "lockbox")):
        matches = [row for row in selected if row["family"] == family and row["language"] == language
                   and row["partition"] == part and row["cell_i"] == 0]
        if matches:
            panel_ids.add(sorted(matches, key=lambda row: row["unit"])[0]["case_id"])
    panel = [row for row in selected if row["case_id"] in panel_ids]
    max_width = max([len(row["prompt_ids"]) for row in panel], default=1)
    role_path = out(name) / "raw/qualified_role_field.float16.npy"
    token_path = out(name) / "raw/full_token_panel.float16.npy"
    role_field = np.lib.format.open_memmap(role_path, mode="w+", dtype=np.float16,
                                           shape=(len(selected), CHECKPOINTS, len(ROLES), DIM))
    token_field = np.lib.format.open_memmap(token_path, mode="w+", dtype=np.float16,
                                            shape=(len(panel), CHECKPOINTS, max_width, DIM))
    panel_map = {row["case_id"]: i for i, row in enumerate(panel)}
    base = model.model
    captured: list[torch.Tensor] = []
    handles = [module.register_forward_hook(
        lambda _m, _a, output: captured.append(output[0] if isinstance(output, tuple) else output))
        for module in [base.embed_tokens, *list(base.layers), base.norm]]
    index, token_index = [], []
    try:
        for row_i, item in enumerate(selected):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            positions = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            if len(captured) != CHECKPOINTS:
                raise RuntimeError((item["case_id"], len(captured)))
            for q, hidden in enumerate(captured):
                values = hidden[0].float().cpu().numpy().astype(np.float16)
                for role_i, role in enumerate(ROLES):
                    role_field[row_i, q, role_i] = values[item["role_positions"][role][-1]]
                if item["case_id"] in panel_map:
                    token_field[panel_map[item["case_id"]], q, :values.shape[0]] = values
            index.append({
                "hidden_index": row_i, "case_id": item["case_id"], "family": item["family"],
                "language": item["language"], "surface": item["surface"], "cell": item["cell"],
                "cell_i": item["cell_i"], "transform_id": item["transform_id"], "unit": item["unit"],
                "partition": item["partition"], "prompt_length": len(item["prompt_ids"]),
                "gold_position": item["gold_position"], "dual_correct": True,
            })
            if item["case_id"] in panel_map:
                token_index.append({"token_index": panel_map[item["case_id"]], "case_id": item["case_id"],
                                    "family": item["family"], "language": item["language"],
                                    "partition": item["partition"], "prompt_length": len(item["prompt_ids"]),
                                    "prompt_ids": item["prompt_ids"]})
            if row_i % 64 == 0:
                print(f"[C715-C720] capture {row_i}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    role_field.flush(); token_field.flush(); close_mmap(role_field); close_mmap(token_field)
    return index, token_index, {
        "role_path": str(role_path.relative_to(ROOT)), "role_shape": [len(selected), CHECKPOINTS, len(ROLES), DIM],
        "token_path": str(token_path.relative_to(ROOT)), "token_shape": [len(panel), CHECKPOINTS, max_width, DIM],
    }


def phase2206() -> None:
    name = "C715-C720"
    if (out(name) / "analysis/final.json").exists():
        return
    compiled = read_rows(out("C710-C714") / "material/qwen_compiled.jsonl")
    model = None
    try:
        model, tokenizer, device, placement = scope.parent.previous.model_base().load_bf16("qwen3")
        quant = scope.parent.previous.model_base().quantization_audit(model)
        candidate_rows = previous.batch_behavior(model, device, compiled)
        generation_rows = previous.free_generate(model, tokenizer, device, compiled)
        candidate = {row["case_id"]: row for row in candidate_rows}
        generated = {row["case_id"]: row for row in generation_rows}
        write_rows(out(name) / "behavior/candidate.jsonl", candidate_rows)
        write_rows(out(name) / "behavior/free_generation.jsonl", generation_rows)
        slices = {}
        for family, language in itertools.product(FAMILIES, LANGUAGES):
            values = {}
            for part in ("discovery", "confirmation", "lockbox"):
                subset = [row for row in compiled if row["family"] == family and row["language"] == language and row["partition"] == part]
                values[part] = {
                    "rows": len(subset),
                    "candidate_accuracy": float(np.mean([candidate[row["case_id"]]["correct"] for row in subset])),
                    "generation_accuracy": float(np.mean([generated[row["case_id"]]["correct"] for row in subset])),
                    "same_row_dual_accuracy": float(np.mean([candidate[row["case_id"]]["correct"] and generated[row["case_id"]]["correct"] for row in subset])),
                }
            values["qualified_prelockbox"] = min(
                values[part][metric]
                for part in ("discovery", "confirmation")
                for metric in ("candidate_accuracy", "generation_accuracy")
            ) >= BEHAVIOR_GATE
            slices[f"{family}|{language}"] = values
        qualified = {key for key, value in slices.items() if value["qualified_prelockbox"]}
        save(out(name) / "behavior/slices.json", slices)
        index, token_index, fields = capture_fields(model, device, compiled, candidate, generated, qualified)
        write_rows(out(name) / "raw/hidden_index.jsonl", index)
        write_rows(out(name) / "raw/full_token_index.jsonl", token_index)
    finally:
        scope.parent.previous.model_base().release_bf16(model); gc.collect()
    role_path = ROOT / fields["role_path"]
    token_path = ROOT / fields["token_path"]
    close(name, {
        "strict_interpretation": "Behavior qualification is slice-specific and every captured row is correct under both candidate selection and free generation. The resulting arrays are activation-state fields, not parameter tensors or evidence of internal family modules.",
        "slice_results": slices, "qualified_slices": sorted(qualified), "qualified_slice_count": len(qualified),
        "captured_rows": len(index), "full_token_panel_rows": len(token_index), "fields": fields,
        "role_field_sha256": file_hash(role_path), "token_field_sha256": file_hash(token_path),
        "placement": placement, "quantization": quant, "human_review": "NA_not_run",
        "new_foundational_mathematics_gate": False,
    }, {
        "parent": final("C710-C714")["all_checks_passed"], "candidate_complete": len(candidate_rows) == len(compiled),
        "generation_complete": len(generation_rows) == len(compiled), "some_qualified": bool(qualified),
        "captured": len(index) > 0 and role_path.exists(), "full_token": token_path.exists(),
        "same_row_rule": all(row["dual_correct"] for row in index),
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"], "finite": finite(slices),
    }, "Authorize C721-C726 to build full-coordinate paired response passports; unqualified slices remain NA without stopping other families.")


def mode_axis0(values: np.ndarray) -> np.ndarray:
    """Deterministic mode over the first axis; ties select the lower code."""
    ordered = np.sort(values, axis=0)
    best = ordered[0].copy()
    best_count = np.ones(best.shape, np.uint16)
    current = ordered[0].copy()
    current_count = np.ones(best.shape, np.uint16)
    for row_i in range(1, ordered.shape[0]):
        same = ordered[row_i] == current
        current_count = np.where(same, current_count + 1, 1).astype(np.uint16)
        current = np.where(same, current, ordered[row_i])
        better = current_count > best_count
        best_count = np.where(better, current_count, best_count)
        best = np.where(better, current, best)
    return best


def pair_indices(index: list[dict], family: str, language: str, transform: int, part: str) -> list[tuple[int, int, int]]:
    lookup = {(row["unit"], row["cell_i"]): row for row in index
              if row["family"] == family and row["language"] == language and row["partition"] == part}
    return [(lookup[(unit, 0)]["hidden_index"], lookup[(unit, transform)]["hidden_index"], unit)
            for unit in sorted({unit for unit, cell in lookup if cell == 0 and (unit, transform) in lookup})]


def response_code(base: np.ndarray, changed: np.ndarray) -> np.ndarray:
    return (local_base.state_code(base).astype(np.uint16) * 33 + local_base.state_code(changed).astype(np.uint16))


def phase2207() -> None:
    name = "C721-C726"
    if (out(name) / "analysis/final.json").exists():
        return
    field_path = out("C715-C720") / "raw/qualified_role_field.float16.npy"
    field = np.load(field_path, mmap_mode="r")
    index = read_rows(out("C715-C720") / "raw/hidden_index.jsonl")
    groups = [(family, language, transform) for family, language, transform in itertools.product(FAMILIES, LANGUAGES, TRANSFORMS)]
    passport_path = out(name) / "raw/family_response_passports.uint16.npy"
    shared_path = out(name) / "raw/shared_response_passports.uint16.npy"
    passports = np.lib.format.open_memmap(passport_path, mode="w+", dtype=np.uint16,
                                           shape=(len(groups), CHECKPOINTS, len(ROLES), DIM))
    shared = np.lib.format.open_memmap(shared_path, mode="w+", dtype=np.uint16,
                                       shape=(len(LANGUAGES), len(TRANSFORMS), CHECKPOINTS, len(ROLES), DIM))
    group_index = {group: i for i, group in enumerate(groups)}
    discovery_counts = {}
    for group_i, (family, language, transform) in enumerate(groups):
        pairs = pair_indices(index, family, language, transform, "discovery")
        discovery_counts[f"{family}|{language}|t{transform}"] = len(pairs)
        if not pairs:
            continue
        for q in range(CHECKPOINTS):
            codes = np.stack([response_code(field[left, q], field[right, q]) for left, right, _unit in pairs])
            passports[group_i, q] = mode_axis0(codes)
        print(f"[C721-C726] passport {group_i + 1}/{len(groups)} {family}|{language}|t{transform}", flush=True)
    for language_i, language in enumerate(LANGUAGES):
        for transform_i, transform in enumerate(TRANSFORMS):
            all_pairs = [pair for family in FAMILIES for pair in pair_indices(index, family, language, transform, "discovery")]
            if not all_pairs:
                continue
            for q in range(CHECKPOINTS):
                codes = np.stack([response_code(field[left, q], field[right, q]) for left, right, _unit in all_pairs])
                shared[language_i, transform_i, q] = mode_axis0(codes)
    passports.flush(); shared.flush()
    metrics = {}
    prospective = []
    for family, language, transform in groups:
        gi = group_index[(family, language, transform)]
        li, ti = LANGUAGES.index(language), TRANSFORMS.index(transform)
        item = {}
        for part in ("confirmation", "lockbox"):
            pairs = pair_indices(index, family, language, transform, part)
            unit_rows = []
            wrong_ids = [group_index[(other, language, transform)] for other in FAMILIES if other != family]
            for left, right, unit in pairs:
                code = response_code(field[left], field[right])
                specific = float(np.mean(code == passports[gi]))
                shared_score = float(np.mean(code == shared[li, ti]))
                wrong = max([float(np.mean(code == passports[wrong_i])) for wrong_i in wrong_ids], default=0.0)
                shifted = float(np.mean(code == np.roll(passports[gi], 257, axis=2)))
                gain = specific - max(shared_score, wrong, shifted)
                unit_rows.append({"unit": unit, "specific": specific, "shared": shared_score,
                                  "wrong_family": wrong, "shift257": shifted, "gain": gain})
            required = max(1, math.ceil(len(unit_rows) * 2 / 3)) if unit_rows else 1
            mean_gain = float(np.mean([row["gain"] for row in unit_rows])) if unit_rows else 0.0
            positive_units = sum(row["gain"] > 0 for row in unit_rows)
            item[part] = {"pairs": len(unit_rows), "units": unit_rows, "mean_gain": mean_gain,
                          "positive_units": positive_units, "required_positive_units": required,
                          "passed": len(unit_rows) >= 4 and mean_gain >= PASSPORT_GAIN_GATE and positive_units >= required}
        item["prospective_passed"] = item["confirmation"]["passed"] and item["lockbox"]["passed"]
        label = f"{family}|{language}|t{transform}"
        metrics[label] = item
        if item["prospective_passed"]:
            prospective.append(label)
    subset = [(group_index[group], group) for group in groups]
    q_indices = list(QPOINTS)
    role_indices = [ROLES.index("relation"), ROLES.index("boundary")]
    agreement = []
    edges = []
    for left_i, left_group in subset:
        row = []
        left_values = np.asarray(passports[left_i][q_indices][:, role_indices])
        for right_i, right_group in subset:
            right_values = np.asarray(passports[right_i][q_indices][:, role_indices])
            exact = float(np.mean(left_values == right_values))
            shifted = float(np.mean(left_values == np.roll(right_values, 257, axis=2)))
            gain = exact - shifted
            row.append({"exact": exact, "shift257": shifted, "gain": gain})
            if left_i < right_i and gain >= 0.02:
                edges.append({"left": "|".join(map(str, left_group)), "right": "|".join(map(str, right_group)),
                              "exact": exact, "shift257": shifted, "gain": gain,
                              "claim": "observational_response_passport_candidate"})
        agreement.append(row)
    save(out(name) / "analysis/passport_metrics.json", metrics)
    save(out(name) / "analysis/equivalence_matrix.json", {"groups": ["|".join(map(str, group)) for group in groups], "matrix": agreement, "edges": edges})
    close_mmap(field); close_mmap(passports); close_mmap(shared)
    close(name, {
        "strict_interpretation": "Response passports retain every physical coordinate as a paired state transition. They are observational transformation signatures; an edge is not a causal state equivalence class until intervention and future-output responses also agree.",
        "groups": ["|".join(map(str, group)) for group in groups], "discovery_pair_counts": discovery_counts,
        "passport_metrics": metrics, "prospective_passport_groups": prospective,
        "observational_equivalence_edges": edges, "equivalence_edge_count": len(edges),
        "passport_shape": [len(groups), CHECKPOINTS, len(ROLES), DIM],
        "passport_sha256": file_hash(passport_path), "shared_sha256": file_hash(shared_path),
        "new_foundational_mathematics_gate": False,
    }, {
        "parent": final("C715-C720")["all_checks_passed"], "groups_complete": len(metrics) == len(groups),
        "all_coordinates": passports.shape[-1] == DIM, "confirmation_before_lockbox": True,
        "independent_units_reported": all("units" in value["confirmation"] and "units" in value["lockbox"] for value in metrics.values()),
        "finite": finite(metrics) and finite(edges),
    }, "Authorize C727-C733 to map all six families locally; semantic eligibility requires a prospective passport, but negative routes remain observable rather than stopping.")


def local_anchors() -> list[dict]:
    index = read_rows(out("C715-C720") / "raw/hidden_index.jsonl")
    compiled = {row["case_id"]: row for row in read_rows(out("C710-C714") / "material/qwen_compiled.jsonl")}
    anchors = []
    for family, language, part in itertools.product(FAMILIES, LANGUAGES, ("confirmation", "lockbox")):
        candidates = [row for row in index if row["family"] == family and row["language"] == language and row["partition"] == part]
        candidates.sort(key=lambda row: ((0, 1, 2, 3).index(row["cell_i"]), row["unit"]))
        if candidates:
            chosen = candidates[0]
            anchors.append({**compiled[chosen["case_id"]], "anchor_family": f"{family}|{language}",
                            "anchor_partition": part, "anchor_selection": "first dual-correct in frozen cell/unit priority"})
    return anchors


def phase2208() -> None:
    name = "C727-C733"
    if (out(name) / "analysis/final.json").exists():
        return
    anchors = local_anchors()
    write_rows(out(name) / "material/local_anchors.jsonl", anchors)
    paired = sorted({row["anchor_family"] for row in anchors
                     if {item["anchor_partition"] for item in anchors if item["anchor_family"] == row["anchor_family"]} == {"confirmation", "lockbox"}})
    anchors = [row for row in anchors if row["anchor_family"] in paired]
    response_path = out(name) / "raw/all_coordinate_response.float16.npy"
    influence_path = out(name) / "raw/all_coordinate_influence.float32.npy"
    response = np.lib.format.open_memmap(response_path, mode="w+", dtype=np.float16,
                                          shape=(len(anchors), 2, DIM, DIM))
    influence = np.lib.format.open_memmap(influence_path, mode="w+", dtype=np.float32,
                                           shape=(len(anchors), DIM))
    model = None
    scans = []
    try:
        model, _tokenizer, _device, placement = scope.parent.previous.model_base().load_bf16("qwen3")
        quant = scope.parent.previous.model_base().quantization_audit(model)
        for anchor_i, anchor in enumerate(anchors):
            scans.append({"case_id": anchor["case_id"], **local_base.local_coordinate_scan(model, anchor, response, influence, anchor_i)})
            response.flush(); influence.flush()
    finally:
        response.flush(); influence.flush(); close_mmap(response); close_mmap(influence)
        scope.parent.previous.model_base().release_bf16(model); gc.collect()
    response = np.load(response_path, mmap_mode="r")
    metrics = {}
    for group in paired:
        confirmation = next(i for i, row in enumerate(anchors) if row["anchor_family"] == group and row["anchor_partition"] == "confirmation")
        lockbox = next(i for i, row in enumerate(anchors) if row["anchor_family"] == group and row["anchor_partition"] == "lockbox")
        family, language = group.split("|")
        wrong = [i for i, row in enumerate(anchors) if row["anchor_partition"] == "lockbox"
                 and row["anchor_family"].endswith(f"|{language}") and row["anchor_family"] != group]
        metrics[group] = {}
        for target_i, target in enumerate(("q25", "final")):
            same = float(np.mean(np.sign(response[confirmation, target_i]) == np.sign(response[lockbox, target_i])))
            wrong_score = max([float(np.mean(np.sign(response[confirmation, target_i]) == np.sign(response[i, target_i]))) for i in wrong], default=0.0)
            shifted = float(np.mean(np.sign(response[confirmation, target_i]) == np.roll(np.sign(response[lockbox, target_i]), 257, axis=1)))
            gain = same - max(wrong_score, shifted)
            metrics[group][target] = {"same": same, "wrong_family_max": wrong_score, "shift257": shifted,
                                      "specificity_gain": gain, "passed": gain >= LOCAL_GAIN_GATE}
    close_mmap(response)
    save(out(name) / "analysis/local_specificity.json", metrics)
    q25_passed = [group for group, value in metrics.items() if value["q25"]["passed"]]
    close(name, {
        "strict_interpretation": "Every q24 relation-role source coordinate was perturbed for every available family-language confirmation/lockbox pair. High raw response agreement is not family-specific unless it beats the strongest wrong-family and shifted-coordinate controls.",
        "anchors": len(anchors), "paired_groups": paired, "scan_metadata": scans,
        "specificity": metrics, "q25_specific_groups": q25_passed,
        "response_sha256": file_hash(response_path), "influence_sha256": file_hash(influence_path),
        "placement": placement, "quantization": quant, "new_foundational_mathematics_gate": False,
    }, {
        "parent": final("C721-C726")["all_checks_passed"], "paired": len(anchors) == 2 * len(paired),
        "broad_families": len({group.split("|")[0] for group in paired}) == len(FAMILIES),
        "full_coordinates": len(scans) == len(anchors),
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "finite": finite(metrics),
    }, "Authorize C734-C739 to test output calling for every mapped group and then run the frozen cross-model panel sequentially.")


@torch.inference_mode()
def patched_free_generation(model, tokenizer, item: dict, direction: np.ndarray, mode: str, dose: float) -> dict:
    base = model.model
    ids = torch.tensor([item["free_prompt_ids"]], dtype=torch.long, device=next(model.parameters()).device)
    mask = torch.ones_like(ids)
    source_pos = int(item["role_positions"]["relation"][-1])
    handle = None
    if mode != "base":
        vector = torch.tensor(direction, dtype=torch.float32, device=ids.device)
        if mode == "opposite":
            vector = -vector
        elif mode == "shift257":
            vector = torch.roll(vector, 257)

        def patch(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Generation uses the full prompt only on the first forward pass;
            # cached decode steps contain one token and must not be re-patched.
            if hidden.shape[1] <= source_pos:
                return output
            changed = hidden.clone()
            current = hidden[0, source_pos].float()
            rms = torch.sqrt(torch.mean(current ** 2))
            epsilon = torch.maximum(current.abs() * 0.125, rms * 0.01)
            changed[0, source_pos] = (current + dose * epsilon * vector).to(hidden.dtype)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handle = base.layers[local_base.SOURCE_Q - 1].register_forward_hook(patch)
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    try:
        generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5, do_sample=False,
                                   pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)
    finally:
        if handle is not None:
            handle.remove()
    text = tokenizer.decode(generated[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
    parsed = previous.parse_binary(text, item["language"])
    return {"text": text, "parsed": parsed, "correct": parsed == item["correct_answer"]}


def cross_model_panel() -> list[dict]:
    rows = read_rows(out("C710-C714") / "material/six_family_bilingual_graph.jsonl")
    selected = []
    for family, language, unit in itertools.product(FAMILIES, LANGUAGES, (18, 19)):
        matches = [row for row in rows if row["family"] == family and row["language"] == language
                   and row["unit"] == unit and row["cell_i"] == 0]
        if len(matches) != 1:
            raise RuntimeError((family, language, unit, len(matches)))
        selected.append({**matches[0], "cross_model_group": f"{family}|{language}"})
    return selected


def phase2209() -> None:
    name = "C734-C739"
    if (out(name) / "analysis/final.json").exists():
        return
    anchors = read_rows(out("C727-C733") / "material/local_anchors.jsonl")
    pairs = final("C727-C733")["paired_groups"]
    response_pass = set(final("C721-C726")["prospective_passport_groups"])
    local_pass = set(final("C727-C733")["q25_specific_groups"])
    influence_path = out("C727-C733") / "raw/all_coordinate_influence.float32.npy"
    influence = np.load(influence_path, mmap_mode="r")
    model = None
    output = {}
    try:
        model, tokenizer, _device, placement = scope.parent.previous.model_base().load_bf16("qwen3")
        for group in pairs:
            confirmation = next(i for i, row in enumerate(anchors) if row["anchor_family"] == group and row["anchor_partition"] == "confirmation")
            lockbox = next(i for i, row in enumerate(anchors) if row["anchor_family"] == group and row["anchor_partition"] == "lockbox")
            direction = np.sign(np.asarray(influence[confirmation], np.float32))
            margins = local_base.coalition_eval(model, anchors[lockbox], direction)
            base_margin = margins["base"]
            margins["aligned_gain"] = margins["aligned_0.25"] - base_margin
            margins["best_control_gain"] = max(margins["opposite_0.25"] - base_margin,
                                                   margins["shift257_0.25"] - base_margin, 0.0)
            generations = {
                "base": patched_free_generation(model, tokenizer, anchors[lockbox], direction, "base", 0.0),
                "aligned": patched_free_generation(model, tokenizer, anchors[lockbox], direction, "aligned", 0.25),
                "opposite": patched_free_generation(model, tokenizer, anchors[lockbox], direction, "opposite", 0.25),
                "shift257": patched_free_generation(model, tokenizer, anchors[lockbox], direction, "shift257", 0.25),
            }
            family = group.split("|")[0]
            passport_eligible = any(label.startswith(group + "|t") for label in response_pass)
            semantic_eligible = passport_eligible and group in local_pass
            passed = (margins["aligned_gain"] - margins["best_control_gain"] >= OUTPUT_GAIN_GATE
                      and generations["aligned"]["correct"] and semantic_eligible)
            output[group] = {"margins": margins, "generations": generations,
                             "passport_eligible": passport_eligible, "local_eligible": group in local_pass,
                             "semantic_eligible": semantic_eligible, "passed": passed,
                             "family": family}
    finally:
        close_mmap(influence)
        scope.parent.previous.model_base().release_bf16(model); gc.collect()
    save(out(name) / "analysis/output_call.json", output)
    panel = cross_model_panel()
    material_path = out(name) / "material/cross_model_24_case_panel.jsonl"
    write_rows(material_path, panel)
    workers = {}
    for model_name in ("glm4", "deepseek7b", "qwen3_14b"):
        worker_output = out(name) / f"raw/{model_name}/worker_result.json"
        completed = subprocess.run([sys.executable, str(Path(local_base.__file__)), "--worker", model_name,
                                    "--material", str(material_path), "--output", str(worker_output)],
                                   cwd=ROOT, check=False)
        workers[model_name] = load(worker_output) if worker_output.exists() else {"model": model_name, "status": "missing_worker_output"}
        workers[model_name]["returncode"] = completed.returncode
        print(f"[C734-C739] {model_name} returncode={completed.returncode}", flush=True)
    qualified = {key: value for key, value in workers.items() if value.get("qualified") and value.get("hiddenstate_ran")}
    topology = {key: {
        "relative_depths": [row["relative_depth"] for row in value["relative_topology"]],
        "mean_flip": [float(np.mean(row["next_sign_flip_by_role"])) for row in value["relative_topology"]],
        "max_flip_role": [ROLES[int(np.argmax(row["next_sign_flip_by_role"]))] for row in value["relative_topology"]],
    } for key, value in qualified.items()}
    passed_output = [group for group, value in output.items() if value["passed"]]
    close(name, {
        "strict_interpretation": "Output-call directions are confirmation-derived coordinate influence signs, not donor HiddenState differences. Margin and generation effects are reported separately. Cross-model comparison is limited to behavior-qualified relative-depth/role summaries.",
        "executor_repair": "The first run reached cached generation step 2 with sequence length 1 and attempted to reuse the full-prompt source index. The hook was repaired to write only when the full source position exists; objects, materials, directions, doses, gates and branches were unchanged.",
        "output_call": output, "semantic_output_groups": passed_output,
        "workers": workers, "qualified_hidden_models": list(qualified), "relative_topology": topology,
        "placement": placement, "new_foundational_mathematics_gate": False,
    }, {
        "parent": final("C727-C733")["all_checks_passed"], "all_output_groups": set(output) == set(pairs),
        "generation_reported": all(set(value["generations"]) == {"base", "aligned", "opposite", "shift257"} for value in output.values()),
        "sequential_models": len(workers) == 3, "workers_returned": all(value.get("returncode") in (0, 1, 2) for value in workers.values()),
        "finite": finite(output) and finite(workers) and finite(topology),
    }, "Authorize C740-C744 to preserve the exact-coordinate atlas, delete undisplayed fields, and adjudicate whether the same response object merits automatic continuation.")


def export_visual() -> dict:
    role_path = out("C715-C720") / "raw/qualified_role_field.float16.npy"
    token_path = out("C715-C720") / "raw/full_token_panel.float16.npy"
    passport_path = out("C721-C726") / "raw/family_response_passports.uint16.npy"
    response_path = out("C727-C733") / "raw/all_coordinate_response.float16.npy"
    influence_path = out("C727-C733") / "raw/all_coordinate_influence.float32.npy"
    role = np.load(role_path, mmap_mode="r")
    token = np.load(token_path, mmap_mode="r")
    passports = np.load(passport_path, mmap_mode="r")
    response = np.load(response_path, mmap_mode="r")
    influence = np.load(influence_path, mmap_mode="r")
    hidden_index = {row["case_id"]: row for row in read_rows(out("C715-C720") / "raw/hidden_index.jsonl")}
    token_index = read_rows(out("C715-C720") / "raw/full_token_index.jsonl")
    compiled = {row["case_id"]: row for row in read_rows(out("C710-C714") / "material/qwen_compiled.jsonl")}
    anchors = read_rows(out("C727-C733") / "material/local_anchors.jsonl")
    groups = [tuple(value.split("|")) for value in final("C721-C726")["groups"]]
    arrays: list[np.ndarray] = []
    rows: list[dict] = []
    for anchor_i, anchor in enumerate(anchors):
        hidden_i = hidden_index[anchor["case_id"]]["hidden_index"]
        for q in QPOINTS:
            arrays.append(np.asarray(role[hidden_i, q, ROLES.index("relation")], np.float32))
            rows.append({"kind": "relation_role_activation", "case_id": anchor["case_id"],
                         "group": anchor["anchor_family"], "partition": anchor["anchor_partition"],
                         "checkpoint": q, "role": "relation"})
        arrays.append(np.asarray(influence[anchor_i], np.float32))
        rows.append({"kind": "coordinate_logit_margin_influence", "case_id": anchor["case_id"],
                     "group": anchor["anchor_family"], "partition": anchor["anchor_partition"], "checkpoint": 24})
        for target_i, target_name in enumerate(("q25_boundary", "final_boundary")):
            arrays.append(np.mean(np.abs(np.asarray(response[anchor_i, target_i], np.float32)), axis=0))
            rows.append({"kind": "mean_absolute_incoming_local_response", "case_id": anchor["case_id"],
                         "group": anchor["anchor_family"], "partition": anchor["anchor_partition"], "target": target_name})
    for group_i, group in enumerate(groups):
        for q in QPOINTS:
            for role_name in ("relation", "boundary"):
                role_i = ROLES.index(role_name)
                arrays.append(np.asarray(passports[group_i, q, role_i], np.float32))
                rows.append({"kind": "paired_state_transition_code", "group": "|".join(group),
                             "checkpoint": q, "role": role_name})
    for item in token_index:
        if item["partition"] != "confirmation":
            continue
        compiled_row = compiled[item["case_id"]]
        for q in (0, 16, 24, 37):
            for token_pos in range(item["prompt_length"]):
                arrays.append(np.asarray(token[item["token_index"], q, token_pos], np.float32))
                rows.append({"kind": "full_token_activation", "case_id": item["case_id"],
                             "family": item["family"], "language": item["language"],
                             "checkpoint": q, "token_position": token_pos,
                             "token_id": compiled_row["prompt_ids"][token_pos]})
    matrix = np.stack(arrays).astype(np.float16)
    VISUAL_BINARY.parent.mkdir(parents=True, exist_ok=True)
    np.save(VISUAL_BINARY, matrix)
    payload = {
        "schema": "ai2050.response-equivalence-coordinate-atlas.v1", "phase": 2210,
        "campaign": "C710-C744", "model": "Qwen3-4B BF16", "coordinate_count": DIM,
        "rows": rows, "binary": str(VISUAL_BINARY.relative_to(ROOT)).replace("\\", "/"),
        "binary_shape": list(matrix.shape), "binary_dtype": "float16",
        "passport_metrics": final("C721-C726")["passport_metrics"],
        "local_specificity": final("C727-C733")["specificity"],
        "output_call": final("C734-C739")["output_call"],
        "claim_boundary": "Exact activation coordinates and paired state-transition codes. No PCA, Top-K, cosine screening, gradients, weights, or donor HiddenState differences.",
    }
    save(VISUAL, payload)
    catalog = load(CATALOG) if CATALOG.exists() else {"schema": "language-encoding-catalog.v1", "datasets": []}
    entry = {
        "id": "c744_response_equivalence_coordinate_atlas", "title": "C744 Response Equivalence Coordinate Atlas",
        "phase": 2210, "campaign": "C710-C744", "model": "Qwen3-4B",
        "source_path": "/vis_data/research_kernel/c744_response_equivalence_coordinate_atlas.json",
        "source_schema": payload["schema"], "coordinate_count": DIM, "checkpoint_count": CHECKPOINTS,
        "kinds": sorted({row["kind"] for row in rows}),
    }
    catalog["datasets"] = [row for row in catalog.get("datasets", []) if row.get("id") != entry["id"]] + [entry]
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    save(CATALOG, catalog)
    for value in (role, token, passports, response, influence):
        close_mmap(value)
    return {"json": str(VISUAL.relative_to(ROOT)), "binary": str(VISUAL_BINARY.relative_to(ROOT)),
            "shape": list(matrix.shape), "rows": len(rows), "sha256": file_hash(VISUAL_BINARY)}


def phase2210() -> None:
    name = "C740-C744"
    if (out(name) / "analysis/final.json").exists():
        return
    visual = export_visual()
    cleanup_paths = [
        out("C715-C720") / "raw/qualified_role_field.float16.npy",
        out("C715-C720") / "raw/full_token_panel.float16.npy",
        out("C721-C726") / "raw/family_response_passports.uint16.npy",
        out("C721-C726") / "raw/shared_response_passports.uint16.npy",
        out("C727-C733") / "raw/all_coordinate_response.float16.npy",
        out("C727-C733") / "raw/all_coordinate_influence.float32.npy",
    ]
    for worker in final("C734-C739")["workers"].values():
        profile = worker.get("coordinate_profile")
        if profile:
            cleanup_paths.append(ROOT / profile)
    cleanup = []
    for path in cleanup_paths:
        item = {"path": str(path.relative_to(ROOT)), "existed": path.exists(), "sha256": None, "deleted": False}
        if path.exists():
            item["sha256"] = file_hash(path)
            path.unlink()
            item["deleted"] = True
        cleanup.append(item)
    save(out(name) / "audit/hash_then_cleanup.json", cleanup)
    passport = final("C721-C726")["prospective_passport_groups"]
    local = final("C727-C733")["q25_specific_groups"]
    output = final("C734-C739")["semantic_output_groups"]
    same_goal = bool(passport or local or output)
    decision = (
        "Automatically continue the same response object on independent fresh concepts and generation-time deletion/rescue."
        if same_goal else
        "The registered response-passport object did not survive its full controls; continue the broader atlas goal with richer natural/human-reviewed states rather than tuning these six labels."
    )
    close(name, {
        "strict_interpretation": "This campaign prioritizes a broad exact-coordinate atlas. Prospective response passports, local family specificity and semantic output calling are separate evidence levels; observational graph edges never become causal equivalence classes by naming alone.",
        "qualified_slices": final("C715-C720")["qualified_slices"],
        "prospective_passport_groups": passport, "q25_specific_groups": local,
        "semantic_output_groups": output,
        "cross_model_qualified_hidden_models": final("C734-C739")["qualified_hidden_models"],
        "visual": visual, "cleanup": cleanup, "human_review": "NA_not_run",
        "new_foundational_mathematics_gate": False, "important_answer_reached": True,
        "next_stage_same_goal": same_goal, "automatic_continuation_decision": decision,
        "theory_update": "The best current object is a base-state-conditioned response passport over exact activation coordinates. It remains an empirical atlas until future interventions define stable response equivalence classes.",
    }, {
        "all_parents": all(final(key)["all_checks_passed"] for key in PHASES if key != name),
        "visual": VISUAL.exists() and VISUAL_BINARY.exists(),
        "raw_cleaned": all(not path.exists() for path in cleanup_paths),
        "finite": finite([passport, local, output, visual]),
    }, decision)


def run_all() -> None:
    freeze()
    rows = material()
    phase2205(rows)
    phase2206()
    phase2207()
    phase2208()
    phase2209()
    phase2210()


if __name__ == "__main__":
    run_all()
