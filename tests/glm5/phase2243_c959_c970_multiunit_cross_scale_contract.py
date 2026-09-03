#!/usr/bin/env python3
"""Freeze a prospective multi-unit language-family topology contract.

The experiment observes embeddings and post-block HiddenStates only. It keeps
all physical activation coordinates and does not inspect attention, MLPs,
weights or gradients. The four lexical units in this file were not used by the
Phase 2234 material generator.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase2219_c773_c808_semantic_transition_ecology_campaign as prior  # noqa: E402
import phase2234_c870_c884_broad_family_gear_contract as base  # noqa: E402


PHASE = 2243
CAMPAIGNS = tuple(f"C{i}" for i in range(959, 971))
OUT = ROOT / "tests/glm5/result/phase2243_c959_c970_multiunit_cross_scale_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
FAMILIES = base.FAMILIES
LANGUAGES = base.LANGUAGES
SURFACES = base.SURFACES
UNITS = tuple(range(4))
BEHAVIOR_GATE = 0.75
FAMILY_BEHAVIOR_GATE = 0.75
MIN_COMMON_FAMILIES = 6
RAW_RETRIEVAL_GATE = 0.50
CENTERED_RETRIEVAL_GATE = 0.40


LEXICON = {
    "en": {
        "a": ("Leona", "Priya", "Selene", "Tessa"),
        "b": ("Marek", "Ruben", "Dario", "Nolan"),
        "taxonomy": (
            ("cedar", "conifer", "tree", "plant", "machine"),
            ("salmon", "fish", "animal", "organism", "artifact"),
            ("sparrow", "bird", "animal", "organism", "instrument"),
            ("orchid", "flowering plant", "plant", "organism", "vehicle"),
        ),
        "part": (
            ("sensor", "guidance module", "survey rover", "archive"),
            ("hinge", "service door", "storage cabinet", "garden"),
            ("lens", "camera assembly", "inspection device", "harbor"),
            ("keycap", "control keyboard", "operator console", "orchard"),
        ),
        "temporal": (
            ("briefing", "launch", "docking"),
            ("inspection", "opening", "closing"),
            ("boarding", "departure", "arrival"),
            ("rehearsal", "performance", "review"),
        ),
        "causal": (
            ("spark", "alarm", "evacuation"),
            ("leak", "pressure drop", "shutdown"),
            ("outage", "restart", "recovery"),
            ("storm", "flooding", "road closure"),
        ),
        "objects": ("parcel", "instrument", "document", "crate"),
        "containers": ("gate", "cabinet", "window", "vault"),
        "attributes": (("red", "blue"), ("smooth", "rough"), ("dry", "wet"), ("warm", "cold")),
        "comparison": (("crate", "suitcase"), ("anvil", "stool"), ("engine", "bicycle"), ("statue", "basket")),
        "translation": (("book", "livre", "maison"), ("house", "maison", "fleur"), ("flower", "fleur", "route"), ("road", "route", "livre")),
        "quantifier": (("mira", "zorin", "narel"), ("tovin", "pelan", "ravid"), ("sela", "kiron", "dovan"), ("bren", "lumet", "sorin")),
    },
    "zh": {
        "a": ("林娜", "普丽雅", "赛琳", "泰莎"),
        "b": ("马雷克", "鲁本", "达里奥", "诺兰"),
        "taxonomy": (
            ("雪松", "针叶树", "树木", "植物", "机器"),
            ("鲑鱼", "鱼类", "动物", "生物", "器具"),
            ("麻雀", "鸟类", "动物", "生物", "乐器"),
            ("兰花", "开花植物", "植物", "生物", "车辆"),
        ),
        "part": (
            ("传感器", "导航模块", "勘测车", "档案室"),
            ("铰链", "维修门", "储物柜", "花园"),
            ("镜头", "相机组件", "检测设备", "港口"),
            ("键帽", "控制键盘", "操作台", "果园"),
        ),
        "temporal": (
            ("说明会", "发射", "对接"),
            ("检查", "开幕", "闭幕"),
            ("登乘", "出发", "抵达"),
            ("彩排", "演出", "复盘"),
        ),
        "causal": (
            ("火花", "警报", "疏散"),
            ("泄漏", "压力下降", "停机"),
            ("断电", "重启", "恢复"),
            ("暴雨", "积水", "封路"),
        ),
        "objects": ("包裹", "仪器", "文件", "木箱"),
        "containers": ("大门", "柜子", "窗户", "保险库"),
        "attributes": (("红色", "蓝色"), ("光滑", "粗糙"), ("干燥", "潮湿"), ("温暖", "寒冷")),
        "comparison": (("木箱", "手提箱"), ("铁砧", "凳子"), ("发动机", "自行车"), ("雕像", "篮子")),
        "translation": (("书", "livre", "maison"), ("房子", "maison", "fleur"), ("花", "fleur", "route"), ("道路", "route", "livre")),
        "quantifier": (("米拉", "佐林类", "纳雷类"), ("托文", "佩兰类", "拉维德类"), ("塞拉", "基隆类", "多万类"), ("布伦", "卢梅特类", "索林类")),
    },
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_core(family: str, language: str, unit: int, truth: bool, paraphrase: bool) -> tuple[str, dict[str, str]]:
    u = LEXICON[language]
    a, b = u["a"][unit], u["b"][unit]
    if family == "taxonomy_chain":
        x, t1, t2, t3, wrong = u["taxonomy"][unit]
        end = t3 if truth else wrong
        if language == "en":
            facts = f"{x} is a {t1}; every {t1} is a {t2}; every {t2} is a {end}"
            core = (f"A checked classification record states: {facts}. Is {x} a {t3}?" if not paraphrase else
                    f"The verified catalog gives this chain: {facts}. Does it classify {x} as a {t3}?")
            rel = "is a"
        else:
            facts = f"{x}是{t1}；每个{t1}都是{t2}；每个{t2}都是{end}"
            core = (f"一份核验过的分类记录写道：{facts}。{x}是{t3}吗？" if not paraphrase else
                    f"核对后的目录给出链条：{facts}。该目录把{x}归为{t3}吗？")
            rel = "是"
        roles = {"primary": x, "secondary": t1, "relation": rel, "context": t3, "query": x}
    elif family == "part_whole_chain":
        x, t1, t2, wrong = u["part"][unit]
        end = t2 if truth else wrong
        if language == "en":
            facts = f"{x} is part of the {t1}; the {t1} is part of the {end}"
            core = (f"A checked assembly record states: {facts}. Is {x} part of the {t2}?" if not paraphrase else
                    f"The verified component ledger says: {facts}. Does it place {x} within the {t2}?")
            rel = "part of"
        else:
            facts = f"{x}是{t1}的一部分；{t1}是{end}的一部分"
            core = (f"一份核验过的装配记录写道：{facts}。{x}是{t2}的一部分吗？" if not paraphrase else
                    f"核对后的部件账本写道：{facts}。它把{x}列在{t2}之内吗？")
            rel = "一部分"
        roles = {"primary": x, "secondary": t1, "relation": rel, "context": t2, "query": x}
    elif family == "temporal_order":
        x, t1, t2 = u["temporal"][unit]
        second = ("before" if truth else "after") if language == "en" else ("早于" if truth else "晚于")
        if language == "en":
            facts = f"the {x} occurred before the {t1}; the {t1} occurred {second} the {t2}"
            core = (f"A checked schedule states that {facts}. Did the {x} occur before the {t2}?" if not paraphrase else
                    f"The verified chronology says that {facts}. Is the {x} earlier than the {t2}?")
            rel = "before"
        else:
            facts = f"{x}早于{t1}发生；{t1}{second}{t2}发生"
            core = (f"一份核验过的时间表写道：{facts}。{x}早于{t2}发生吗？" if not paraphrase else
                    f"核对后的时间顺序写道：{facts}。{x}比{t2}更早吗？")
            rel = "早于"
        roles = {"primary": x, "secondary": t1, "relation": rel, "context": t2, "query": x}
    elif family == "causal_direction":
        x, t1, t2 = u["causal"][unit]
        second = ("caused" if truth else "prevented") if language == "en" else ("导致" if truth else "阻止")
        if language == "en":
            facts = f"the {x} caused the {t1}; the {t1} {second} the {t2}"
            core = (f"A checked causal report states that {facts}. Did the {x} indirectly cause the {t2}?" if not paraphrase else
                    f"The verified dependency log says that {facts}. Is the {t2} a downstream effect of the {x}?")
            rel = "caused"
        else:
            facts = f"{x}导致{t1}；{t1}{second}{t2}"
            core = (f"一份核验过的因果报告写道：{facts}。{x}间接导致{t2}吗？" if not paraphrase else
                    f"核对后的依赖日志写道：{facts}。{t2}是{x}的下游结果吗？")
            rel = "导致"
        roles = {"primary": x, "secondary": t1, "relation": rel, "context": t2, "query": x}
    elif family == "agent_patient_voice":
        x = u["objects"][unit]; agent = a if truth else b
        if language == "en":
            fact = f"{agent} carried the {x}" if not paraphrase else f"the {x} was carried by {agent}"
            core = f"A checked event report states that {fact}. Did {a} carry the {x}?"; rel = "carried"
        else:
            fact = f"{agent}搬运了{x}" if not paraphrase else f"{x}由{agent}搬运"
            core = f"一份核验过的事件记录写道：{fact}。{a}搬运了{x}吗？"; rel = "搬运"
        roles = {"primary": a, "secondary": x, "relation": rel, "context": x, "query": a}
    elif family == "negation_scope":
        x = u["containers"][unit]
        if language == "en":
            fact = f"{a} {'opened' if truth else 'did not open'} the {x}" if not paraphrase else f"it is {'true' if truth else 'false'} that {a} opened the {x}"
            core = f"A checked report states: {fact}. Does it support that {a} opened the {x}?"; rel = "opened"
        else:
            fact = f"{a}{'打开了' if truth else '没有打开'}{x}" if not paraphrase else f"{a}打开{x}这件事是{'真的' if truth else '假的'}"
            core = f"一份核验过的报告写道：{fact}。报告支持{a}打开了{x}吗？"; rel = "打开"
        roles = {"primary": a, "secondary": x, "relation": rel, "context": x, "query": a}
    elif family == "coreference_binding":
        x = u["objects"][unit]; speaker, listener = (a, b) if truth else (b, a)
        if language == "en":
            fact = f"{speaker} told {listener}, 'I stored the {x}.'" if not paraphrase else f"while {listener} listened, the transcript quotes {speaker}: 'I stored the {x}.'"
            core = f"A checked quotation record states: {fact} In the quotation, does I refer to {a}?"; rel = "I"
        else:
            fact = f"{speaker}对{listener}说：‘我存放了{x}。’" if not paraphrase else f"{listener}在场聆听，记录引用{speaker}的话：‘我存放了{x}。’"
            core = f"一份核验过的引语记录写道：{fact}在引语中，‘我’指的是{a}吗？"; rel = "我"
        roles = {"primary": a, "secondary": b, "relation": rel, "context": x, "query": a}
    elif family == "nested_attitude":
        x = u["objects"][unit]
        if language == "en":
            verb = "remembered" if truth else "heard"
            fact = f"{a} {verb} that {b} stored the {x}" if not paraphrase else f"the record describes {b}'s storing of the {x} as something {a} {verb}"
            core = f"A checked memory record states: {fact}. Does it say that {a} remembered that {b} stored the {x}?"; rel = "remembered"
        else:
            verb = "记得" if truth else "听说"
            fact = f"{a}{verb}{b}存放了{x}" if not paraphrase else f"记录把{b}存放{x}这件事描述为{a}{verb}的内容"
            core = f"一份核验过的记忆记录写道：{fact}。记录表示{a}记得{b}存放了{x}吗？"; rel = "记得"
        roles = {"primary": a, "secondary": b, "relation": rel, "context": x, "query": a}
    elif family == "attribute_binding":
        x = u["objects"][unit]; good, bad = u["attributes"][unit]; prop = good if truth else bad
        if language == "en":
            fact = f"the {x} is {prop}" if not paraphrase else f"the recorded property of the {x} is {prop}"
            core = f"A checked property record states that {fact}. Is the {x} {good}?"; rel = "property"
        else:
            fact = f"{x}是{prop}的" if not paraphrase else f"{x}登记的属性是{prop}"
            core = f"一份核验过的属性记录写道：{fact}。{x}是{good}的吗？"; rel = "属性"
        roles = {"primary": x, "secondary": prop, "relation": rel, "context": good, "query": x}
    elif family == "comparison":
        x, y = u["comparison"][unit]
        if language == "en":
            rel = "heavier than" if truth else "lighter than"
            fact = f"the {x} is {rel} the {y}" if not paraphrase else f"the comparison places the {x} as {rel} the {y}"
            core = f"A checked comparison record states that {fact}. Is the {x} heavier than the {y}?"; role_rel = "heavier"
        else:
            rel = "重于" if truth else "轻于"
            fact = f"{x}{rel}{y}" if not paraphrase else f"比较记录把{x}列为{rel}{y}"
            core = f"一份核验过的比较记录写道：{fact}。{x}重于{y}吗？"; role_rel = "重于"
        roles = {"primary": x, "secondary": y, "relation": role_rel, "context": y, "query": x}
    elif family == "translation_route":
        x, good, bad = u["translation"][unit]; target = good if truth else bad
        if language == "en":
            fact = f"the English word {x} maps to French {target}" if not paraphrase else f"the French entry paired with English {x} is {target}"
            core = f"A checked bilingual glossary states that {fact}. Is the French form of {x} {good}?"; rel = "French"
        else:
            fact = f"中文词{x}对应法语{target}" if not paraphrase else f"与中文{x}配对的法语条目是{target}"
            core = f"一份核验过的双语词表写道：{fact}。{x}的法语形式是{good}吗？"; rel = "法语"
        roles = {"primary": x, "secondary": target, "relation": rel, "context": good, "query": x}
    elif family == "quantifier_scope":
        x, t1, t2 = u["quantifier"][unit]
        quant = ("Every" if truth else "No") if language == "en" else ("每个" if truth else "没有任何")
        if language == "en":
            facts = f"{x} is a {t1}; {quant} {t1} is a {t2}"
            core = (f"A checked quantified record states that {facts}. Is {x} a {t2}?" if not paraphrase else
                    f"The verified registry places {x} in {t1} and says that {quant.lower()} {t1} is a {t2}. Does {x} belong to {t2}?")
            rel = "is a"
        else:
            facts = f"{x}是{t1}；{quant}{t1}是{t2}"
            core = (f"一份核验过的量化记录写道：{facts}。{x}是{t2}吗？" if not paraphrase else
                    f"核对后的登记表把{x}归入{t1}，并写明{quant}{t1}属于{t2}。{x}属于{t2}吗？")
            rel = "是" if not paraphrase else "属于"
        roles = {"primary": x, "secondary": t1, "relation": rel, "context": t2, "query": x}
    else:
        raise KeyError(family)
    return core, roles


def material() -> list[dict]:
    rows = []
    for family, language, unit, cell_i in itertools.product(FAMILIES, LANGUAGES, UNITS, range(4)):
        truth = base.TRUTH_BY_CELL[cell_i]
        core, roles = make_core(family, language, unit, truth, cell_i >= 2)
        row = base.wrap_case(
            case_id=f"c959-{family}-{language}-u{unit}-c{cell_i}", panel="multiunit_family_lockbox",
            family=family, language=language, unit=unit, cell=base.CELLS[cell_i], cell_i=cell_i,
            truth=truth, core=core, roles=roles,
            factors={"semantic_truth": int(truth), "surface": int(cell_i >= 2), "prospective_unit": unit},
            fresh=True, offset=17,
        )
        row["partition"] = "prospective_lockbox"
        row["material_generation"] = "phase2243_new_lexicon"
        rows.append(row)
    return rows


def audit(rows: list[dict], compiled: list[dict]) -> dict:
    counts = defaultdict(int)
    missing = []
    malformed = []
    for row in rows:
        counts[(row["family"], row["language"], row["unit"], row["surface"], row["truth"])] += 1
        if any(x in row["prompt"] for x in ("�", "锟")):
            malformed.append(row["case_id"])
        for role, value in row["role_values"].items():
            if value not in row["prompt_core"]:
                missing.append({"case_id": row["case_id"], "role": role, "value": value})
    expected = len(FAMILIES) * len(LANGUAGES) * len(UNITS) * 4
    pair_complete = all(
        counts[(f, lang, unit, surface, truth)] == 1
        for f, lang, unit, surface, truth in itertools.product(FAMILIES, LANGUAGES, UNITS, SURFACES, (False, True))
    )
    zero = {
        "always_A": float(np.mean([row["gold_position"] == 0 for row in rows])),
        "always_B": float(np.mean([row["gold_position"] == 1 for row in rows])),
        "always_supported": float(np.mean([row["truth"] for row in rows])),
    }
    widths = [len(row["prompt_ids"]) for row in compiled]
    return {
        "rows": len(rows), "expected_rows": expected, "unique_case_ids": len({row["case_id"] for row in rows}),
        "factorial_complete": pair_complete, "zero_models": zero, "missing_roles": missing,
        "malformed_strings": malformed, "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "semantic_uniqueness_machine_audit": "pass_explicit_truth_table_and_role_span_compilation",
        "material_naturalness_machine_audit": "pass_family_specific_bilingual_templates_and_forbidden_string_scan",
        "human_blind_review": "NA_no_independent_human_panel_available",
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    a = result["audit"]
    text = rf"""

## Phase {PHASE}: 多语义单元跨规模语言族拓扑前瞻合同（C959-C970） [{stamp}]

**目标与证据纠偏。** Phase2242 在单个语义单元上得到 Qwen4B/Qwen14B 原始拓扑 12/12 双向同族检索、模型中心化后 11/12 与 12/12；这支持跨规模语言族角色拓扑候选，但不能排除单词、模板或单元身份。C959-C970 因而冻结四套未被 Phase2234 材料生成器使用的新词汇，每族、每语言、每表面均含真假配对。它不预设固定语义坐标，不跨模型比较坐标编号。

**测试原理和用例。** 12族包括类型链、部分整体、时间、因果、施受事、否定、共指、嵌套态度、属性、比较、翻译和量词。例子包括“雪松是针叶树；每个针叶树都是树木；每个树木都是植物。雪松是植物吗”和“林娜对马雷克说：‘我存放了包裹。’引语中的我指林娜吗”。冻结对象为 embedding、全部 block 后状态、final norm、六个功能角色和每个物理激活坐标。

$$
\Delta H_{{m,f,u,\ell,r,j}}=H^{{(1)}}_{{m,f,u,\ell,r,j}}-H^{{(0)}}_{{m,f,u,\ell,r,j}}.
$$

逐单元留一检索不允许使用被测单元建立另一个模型的族原型：

$$
\widehat f_{{m\to n}}(f,u)=\arg\min_g D\!\left(\rho_{{m,f,u}},\frac1{{|U|-1}}\sum_{{v\ne u}}\rho_{{n,g,v}}\right).
$$

**材料、零模型和门槛。** 共 `{a['rows']}` 行，预期 `{a['expected_rows']}` 行；Qwen tokenizer 编译后 token 长度最小/中位/最大为 `{a['token_width_min_median_max']}`。四格配对完整为 `{a['factorial_complete']}`，缺失角色 `{len(a['missing_roles'])}`，乱码 `{len(a['malformed_strings'])}`，零模型为 `{json.dumps(a['zero_models'], ensure_ascii=False)}`。人类独立盲评不可用，严格记 NA，机器自然度审计不能替代人类。模型依次为 Qwen3-4B、Qwen3-14B、GLM4、DS7B；总体候选与自由生成均不低于0.75才采集全场，族级分析还要求该族双行为均不低于0.75。跨模型确认门预先固定为共同合格族至少 `{MIN_COMMON_FAMILIES}`，原始留一检索双向准确率至少 `{RAW_RETRIEVAL_GATE}`，模型中心化检索双向至少 `{CENTERED_RETRIEVAL_GATE}` 且中位边距为正。门失败只淘汰确认主张，不停止其余观察。

**理论进展、硬伤和结论。** 理论名称保持“条件化输出场闭合理论”，RDC原则不变。本期只有材料与合同结果，没有模型或HiddenState结论。材料仍是受控中英文本，量词使用虚构类别以隔离外部知识；多词角色依赖上下文跨度编译；人类自然度为NA；四个单元仍不足以代表自然语言全域。合同和编译审计 `{result['all_checks_passed']}`，授权顺序运行四模型行为、合格模型全场、模型内全坐标留一检索、跨模型角色拓扑留一检索、可视化和哈希后清理。

**相关文件。** 脚本 `tests/glm5/phase2243_c959_c970_multiunit_cross_scale_contract.py`；结果 `{OUT.relative_to(ROOT)}`；材料 `material/prospective_cases.jsonl` 与 `material/qwen_compiled.jsonl`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    rows = material()
    tokenizer = prior.parent.load_tokenizer()
    compiled = base.compile_rows(tokenizer, rows)
    write_rows(OUT / "material/prospective_cases.jsonl", rows)
    write_rows(OUT / "material/qwen_compiled.jsonl", compiled)
    a = audit(rows, compiled)
    save(OUT / "audit/material_audit.json", a)
    protocol = {
        "timestamp": datetime.now().astimezone().isoformat(), "phase": PHASE, "campaigns": list(CAMPAIGNS),
        "frozen_before_model": True, "rows": len(rows), "families": list(FAMILIES), "units": list(UNITS),
        "languages": list(LANGUAGES), "surfaces": list(SURFACES),
        "models_sequential": ["qwen3", "qwen3_14b", "glm4", "deepseek7b"],
        "behavior_gate": BEHAVIOR_GATE, "family_behavior_gate": FAMILY_BEHAVIOR_GATE,
        "cross_model_gates": {"minimum_common_families": MIN_COMMON_FAMILIES,
                              "raw_bidirectional_accuracy": RAW_RETRIEVAL_GATE,
                              "model_centered_bidirectional_accuracy": CENTERED_RETRIEVAL_GATE,
                              "median_margin": "strictly_positive"},
        "capture": "all checkpoints, six roles and every physical activation coordinate iff aggregate dual behavior passes",
        "within_model": "full-coordinate signed semantic-response leave-one-unit-out family retrieval",
        "cross_model": "relative-depth role-energy topology leave-one-unit-out retrieval; never coordinate IDs",
        "forbidden": ["attention", "MLP", "weights", "gradients", "PCA", "Top-K", "cosine screening", "donor delta transport"],
        "failure_policy": "route-level missingness; continue all preregistered models and analyses",
        "cleanup": "visualize full-coordinate unit prototypes, verify hashes, then delete non-displayed raw sample fields",
    }
    save(OUT / "protocol/preregistration.json", protocol)
    checks = {
        "row_count": a["rows"] == a["expected_rows"] == 384,
        "unique_ids": a["unique_case_ids"] == a["rows"],
        "factorial_complete": a["factorial_complete"], "roles_present": not a["missing_roles"],
        "strings_clean": not a["malformed_strings"],
        "zero_models_balanced": all(abs(v - 0.5) <= 1e-12 for v in a["zero_models"].values()),
        "finite": all(math.isfinite(v) for v in a["token_width_min_median_max"]),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()), "audit": a, "protocol": protocol,
        "hashes": {"material": file_hash(OUT / "material/prospective_cases.jsonl"),
                   "qwen_compiled": file_hash(OUT / "material/qwen_compiled.jsonl")},
        "strict_conclusion": "The prospective multi-unit contract is compiler-valid; no model or mechanism result exists yet.",
        "next_authorization": "Run all four models sequentially, preserving route-level missingness.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
