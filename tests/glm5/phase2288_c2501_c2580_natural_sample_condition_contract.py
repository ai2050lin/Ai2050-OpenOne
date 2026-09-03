#!/usr/bin/env python3
"""Freeze an independent natural bilingual sample-conditioned campaign."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2288_c2501_c2580_natural_sample_condition_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import model_utils  # noqa: E402
import phase1797_c263_c272_state_operator_common as compiler  # noqa: E402


PHASE = 2288
CAMPAIGN = "C2501-C2580"
FAMILIES = (
    "agent_patient",
    "possession_query",
    "relative_binding",
    "location_binding",
    "temporal_order",
    "comparison_order",
    "attitude_event",
    "taxonomy_chain",
)
LANGUAGES = ("en", "zh")
SURFACES = ("narrative", "dialogue")
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
UNITS = 32
PARTITION_RANGES = {
    "discovery": range(0, 12),
    "confirmation": range(12, 20),
    "fresh_confirmation": range(20, 26),
    "fresh_lockbox": range(26, 32),
}
BEHAVIOR_GATE = 0.75
PREDICTION_GATES = {
    "minimum_pairs": 12,
    "gain_over_target_mean": 0.03,
    "gain_over_shared_model": 0.01,
    "coordinate_win_fraction": 0.52,
    "maximum_oracle_ratio": 1.35,
}
CAUSAL_DENSITIES = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)
CAUSAL_DOSES = (0.25, 0.5, 1.0)
CAUSAL_GATES = {
    "middle_checkpoint_min": 6,
    "middle_checkpoint_max": 30,
    "confirmation_direction_rate": 0.60,
    "control_advantage": 0.03,
    "generation_advantage": 0.05,
}

NAMES_EN = (
    "Amina", "Boris", "Clara", "Derek", "Elena", "Felix", "Greta", "Hugo",
    "Iris", "Jonas", "Kira", "Leo", "Mara", "Noel", "Oona", "Priya",
    "Quinn", "Rhea", "Soren", "Talia", "Uma", "Viktor", "Willa", "Xavier",
    "Yara", "Zane", "Adela", "Basil", "Cora", "Dario", "Esme", "Farid",
)
NAMES_ZH = (
    "安宁", "白川", "陈曦", "丁岚", "方晴", "高远", "何静", "江帆",
    "孔明", "林悦", "罗宁", "孟然", "宁夏", "欧阳", "彭宇", "秦川",
    "任安", "沈佳", "唐宁", "吴越", "许文", "杨帆", "张岚", "周宁",
    "艾青", "包晨", "曹悦", "戴安", "冯雪", "郭阳", "韩梅", "金澄",
)
OBJECTS_EN = (
    "atlas", "basket", "camera", "drum", "easel", "flute", "globe", "helmet",
    "inkpot", "jacket", "key", "lantern", "mirror", "notebook", "ornament", "puzzle",
    "quilt", "radio", "stamp", "thermos", "umbrella", "wallet", "xylophone", "yarn",
    "album", "brooch", "compass", "diary", "envelope", "folder", "guitar", "hourglass",
)
OBJECTS_ZH = (
    "地图册", "篮子", "相机", "鼓", "画架", "长笛", "地球仪", "头盔",
    "墨水瓶", "夹克", "钥匙", "灯笼", "镜子", "笔记本", "饰品", "拼图",
    "被子", "收音机", "邮票", "保温瓶", "雨伞", "钱包", "木琴", "毛线",
    "相册", "胸针", "指南针", "日记本", "信封", "文件夹", "吉他", "沙漏",
)
PLACES_EN = (
    "archive", "balcony", "cabinet", "depot", "exhibit", "foyer", "gallery", "hall",
    "island", "junction", "kitchen", "library", "museum", "nursery", "office", "pantry",
    "quay", "reading room", "studio", "workshop", "courtyard", "station", "theater", "garden",
    "laboratory", "warehouse", "classroom", "observatory", "lobby", "harbor", "clinic", "tower",
)
PLACES_ZH = (
    "档案室", "阳台", "橱柜", "仓库", "展厅", "门厅", "画廊", "大厅",
    "岛上", "路口", "厨房", "图书馆", "博物馆", "育苗室", "办公室", "储藏室",
    "码头", "阅览室", "工作室", "车间", "庭院", "车站", "剧院", "花园",
    "实验室", "货仓", "教室", "天文台", "大厅入口", "港口", "诊所", "塔楼",
)
ADJECTIVES_EN = (
    "calm", "bright", "careful", "swift", "patient", "quiet", "precise", "gentle",
    "bold", "steady", "curious", "alert", "warm", "formal", "open", "focused",
    "kind", "direct", "thoughtful", "ready", "nimble", "frank", "modest", "orderly",
    "cheerful", "serious", "helpful", "honest", "polite", "active", "relaxed", "vigilant",
)
ADJECTIVES_ZH = (
    "沉着", "开朗", "仔细", "敏捷", "耐心", "安静", "精确", "温和",
    "勇敢", "稳重", "好奇", "警觉", "热情", "正式", "坦率", "专注",
    "友善", "直接", "周到", "从容", "灵活", "诚恳", "谦逊", "有序",
    "愉快", "严肃", "乐于助人", "诚实", "礼貌", "活跃", "放松", "谨慎",
)
KINDS_EN = (
    "fruit", "tool", "instrument", "vehicle", "fabric", "vessel", "device", "flower",
    "mineral", "bird", "tree", "building", "document", "container", "garment", "machine",
    "beverage", "grain", "furniture", "artwork", "signal", "toy", "material", "animal",
    "plant", "book", "utensil", "medicine", "landmark", "appliance", "ornament", "resource",
)
KINDS_ZH = (
    "水果", "工具", "乐器", "车辆", "织物", "容器", "设备", "花卉",
    "矿物", "鸟类", "树木", "建筑", "文档", "收纳物", "服装", "机器",
    "饮料", "谷物", "家具", "艺术品", "信号", "玩具", "材料", "动物",
    "植物", "书籍", "器具", "药品", "地标", "电器", "装饰物", "资源",
)
SUPER_EN = ("food", "artifact", "object", "organism")
SUPER_ZH = ("食物", "人工制品", "物体", "生物")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def text_span(tokenizer, ids: list[int], value: str) -> tuple[list[int], str]:
    exact = compiler.graph_base.name_spans(tokenizer, ids, value)
    if exact:
        return exact[0], "exact_token_subsequence"
    target = "".join(value.split()).lower()
    width = max(1, len(tokenizer.encode(value, add_special_tokens=False)))
    candidates = []
    for span_width in range(1, min(width + 4, 12) + 1):
        for start in range(0, len(ids) - span_width + 1):
            decoded = "".join(tokenizer.decode(ids[start:start + span_width], skip_special_tokens=True).split()).lower()
            if target and target in decoded:
                candidates.append(list(range(start, start + span_width)))
        if candidates:
            break
    if not candidates:
        raise RuntimeError(("role_text_span", value))
    return candidates[0], "decoded_text_span"


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(value) != 1 for value in candidates):
        raise RuntimeError(("candidate_singleton", candidates))
    system = "Answer only from the supplied text. Do not use outside knowledge."
    output = []
    for row in rows:
        ids = compiler.core.chat_ids(tokenizer, system, row["prompt"])
        free_ids = compiler.core.chat_ids(tokenizer, system, row["free_prompt"])
        positions, methods = {}, {}
        for role, value in row["role_values"].items():
            span, method = text_span(tokenizer, ids, value)
            positions[role], methods[role] = span, method
        positions["boundary"] = [len(ids) - 1]
        methods["boundary"] = "assistant_generation_boundary"
        output.append({**row, "prompt_ids": ids, "free_prompt_ids": free_ids,
                       "candidate_ids": candidates, "role_positions": positions,
                       "role_position_methods": methods})
    return output


def partition(unit: int) -> str:
    for name, units in PARTITION_RANGES.items():
        if unit in units:
            return name
    raise ValueError(unit)


def vocabulary(language: str, unit: int) -> dict[str, str]:
    names = NAMES_EN if language == "en" else NAMES_ZH
    objects = OBJECTS_EN if language == "en" else OBJECTS_ZH
    places = PLACES_EN if language == "en" else PLACES_ZH
    adjectives = ADJECTIVES_EN if language == "en" else ADJECTIVES_ZH
    kinds = KINDS_EN if language == "en" else KINDS_ZH
    supers = SUPER_EN if language == "en" else SUPER_ZH
    return {
        "a": names[unit], "b": names[(unit + 11) % UNITS], "c": names[(unit + 21) % UNITS],
        "obj": objects[unit], "alt_obj": objects[(unit + 13) % UNITS],
        "place": places[unit], "alt_place": places[(unit + 9) % UNITS],
        "adj": adjectives[unit], "alt_adj": adjectives[(unit + 7) % UNITS],
        "kind": kinds[unit], "alt_kind": kinds[(unit + 5) % UNITS],
        "super": supers[unit % len(supers)], "alt_super": supers[(unit + 1) % len(supers)],
    }


def semantic_case(family: str, language: str, surface: str, unit: int, state: int) -> dict:
    v = vocabulary(language, unit)
    a, b, c, obj, alt = v["a"], v["b"], v["c"], v["obj"], v["alt_obj"]
    dialogue = surface == "dialogue"
    if family == "agent_patient":
        agent, patient = ((a, b) if state else (b, a))
        if language == "en":
            core = (f"{agent} handed the {obj} to {patient}. Who handed over the {obj}?" if not dialogue else
                    f"A witness said, '{agent} gave the {obj} to {patient}.' Who was the giver?")
            relation = "gave" if dialogue else "handed"
        else:
            core = (f"{agent}把{obj}交给了{patient}。是谁交出了{obj}？" if not dialogue else
                    f"目击者说：“{agent}把{obj}给了{patient}。”谁是交付者？")
            relation = "给了" if dialogue else "交给"
        return {"core": core, "correct": agent, "wrong": patient,
                "roles": {"primary": agent, "secondary": patient, "relation": relation,
                          "context": obj, "query": obj},
                "graph": [[agent, "giver", obj], [obj, "recipient", patient]]}
    if family == "possession_query":
        owned, other = ((obj, alt) if state else (alt, obj))
        if language == "en":
            core = (f"{a} owns the {owned}; {b} owns the {other}. What does {a} own?" if not dialogue else
                    f"'{owned} belongs to {a}, while {other} belongs to {b},' the clerk said. Which item belongs to {a}?")
            relation = "owns" if not dialogue else "belongs"
        else:
            core = (f"{a}拥有{owned}；{b}拥有{other}。{a}拥有什么？" if not dialogue else
                    f"管理员说：“{owned}属于{a}，而{other}属于{b}。”哪件物品属于{a}？")
            relation = "拥有" if not dialogue else "属于"
        return {"core": core, "correct": owned, "wrong": other,
                "roles": {"primary": a, "secondary": b, "relation": relation,
                          "context": owned, "query": a},
                "graph": [[a, "owns", owned], [b, "owns", other]]}
    if family == "relative_binding":
        thanked, other = ((a, b) if state else (b, a))
        if language == "en":
            core = (f"The editor who thanked {thanked} later interviewed {other}. Whom did the editor thank?" if not dialogue else
                    f"'{thanked} was thanked by the editor, and {other} was interviewed afterward.' Whom did the editor thank?")
            relation = "thanked"
        else:
            core = (f"那位感谢了{thanked}的编辑后来采访了{other}。编辑感谢了谁？" if not dialogue else
                    f"记录写道：“编辑感谢了{thanked}，随后采访了{other}。”编辑感谢了谁？")
            relation = "感谢"
        return {"core": core, "correct": thanked, "wrong": other,
                "roles": {"primary": thanked, "secondary": other, "relation": relation,
                          "context": "editor" if language == "en" else "编辑", "query": relation},
                "graph": [["editor", "thanked", thanked], ["editor", "interviewed", other]]}
    if family == "location_binding":
        place, other = ((v["place"], v["alt_place"]) if state else (v["alt_place"], v["place"]))
        if language == "en":
            core = (f"The {obj} is in the {place}, not in the {other}. Where is the {obj}?" if not dialogue else
                    f"'Look for the {obj} in the {place}; the {other} is empty.' Where should one look for the {obj}?")
            relation = "in"
        else:
            core = (f"{obj}在{place}里，不在{other}里。{obj}在哪里？" if not dialogue else
                    f"管理员说：“去{place}找{obj}，{other}是空的。”应该去哪里找{obj}？")
            relation = "找" if dialogue else "在"
        return {"core": core, "correct": place, "wrong": other,
                "roles": {"primary": obj, "secondary": place, "relation": relation,
                          "context": other, "query": obj},
                "graph": [[obj, "located_in", place], [obj, "not_in", other]]}
    if family == "temporal_order":
        first, second = ((a, b) if state else (b, a))
        if language == "en":
            core = (f"{first} arrived before {second}. Who arrived first?" if not dialogue else
                    f"The log says, '{second} arrived after {first}.' Who arrived first?")
            relation = "before" if not dialogue else "after"
        else:
            core = (f"{first}比{second}先到。谁先到？" if not dialogue else
                    f"日志写道：“{second}在{first}之后到达。”谁先到？")
            relation = "先到" if not dialogue else "之后"
        return {"core": core, "correct": first, "wrong": second,
                "roles": {"primary": first, "secondary": second, "relation": relation,
                          "context": "arrived" if language == "en" else ("到达" if dialogue else "先到"),
                          "query": "first" if language == "en" else "先"},
                "graph": [[first, "before", second]]}
    if family == "comparison_order":
        stronger, weaker = ((a, b) if state else (b, a))
        if language == "en":
            core = (f"{stronger} is more {v['adj']} than {weaker}. Who is more {v['adj']}?" if not dialogue else
                    f"A reviewer said, '{weaker} is less {v['adj']} than {stronger}.' Who ranks higher for being {v['adj']}?")
            relation = "more" if not dialogue else "less"
        else:
            core = (f"{stronger}比{weaker}更{v['adj']}。谁更{v['adj']}？" if not dialogue else
                    f"评审说：“{weaker}不如{stronger}{v['adj']}。”谁在{v['adj']}方面排名更高？")
            relation = "更" if not dialogue else "不如"
        return {"core": core, "correct": stronger, "wrong": weaker,
                "roles": {"primary": stronger, "secondary": weaker, "relation": relation,
                          "context": v["adj"], "query": v["adj"]},
                "graph": [[stronger, "more", v["adj"]], [weaker, "less", v["adj"]]]}
    if family == "attitude_event":
        eater, other = ((b, c) if state else (c, b))
        if language == "en":
            core = (f"{a} likes the fact that {eater} ate the {obj}; {other} ate the {alt}. Who ate the {obj}?" if not dialogue else
                    f"{a} said, 'I am glad that {eater} ate the {obj}, while {other} ate the {alt}.' Who ate the {obj}?")
            relation = "likes" if not dialogue else "glad"
        else:
            core = (f"{a}喜欢这样一个事实：{eater}吃了{obj}，而{other}吃了{alt}。谁吃了{obj}？" if not dialogue else
                    f"{a}说：“我很高兴{eater}吃了{obj}，而{other}吃了{alt}。”谁吃了{obj}？")
            relation = "喜欢" if not dialogue else "高兴"
        return {"core": core, "correct": eater, "wrong": other,
                "roles": {"primary": eater, "secondary": other, "relation": relation,
                          "context": a, "query": obj},
                "graph": [[a, "attitude", "like"], [eater, "ate", obj], [other, "ate", alt]]}
    if family == "taxonomy_chain":
        parent = v["kind"]
        upper = v["super"] if state else v["alt_super"]
        queried = v["super"]
        answer_yes = state == 1
        yes, no = (("Yes", "No") if language == "en" else ("是", "否"))
        if language == "en":
            core = (f"In this catalog, every {obj} is a {parent}, and every {parent} is {upper}. Is the {obj} {queried}?" if not dialogue else
                    f"The curator says, 'A {obj} counts as a {parent}; each {parent} counts as {upper}.' Does the {obj} count as {queried}?")
            relation = "counts as" if dialogue else "is"
        else:
            core = (f"在这份目录中，每个{obj}都是{parent}，每个{parent}都是{upper}。{obj}是{queried}吗？" if not dialogue else
                    f"管理员说：“{obj}算作{parent}，每个{parent}又算作{upper}。”{obj}算作{queried}吗？")
            relation = "算作" if dialogue else "是"
        return {"core": core, "correct": yes if answer_yes else no, "wrong": no if answer_yes else yes,
                "roles": {"primary": obj, "secondary": parent, "relation": relation,
                          "context": upper, "query": queried},
                "graph": [[obj, "is_a", parent], [parent, "is_a", upper], [obj, "query_is_a", queried]]}
    raise KeyError(family)


def material() -> list[dict]:
    rows = []
    for family_i, (family, language, surface, unit, state) in enumerate(
        itertools.product(FAMILIES, LANGUAGES, SURFACES, range(UNITS), (0, 1))
    ):
        case = semantic_case(family, language, surface, unit, state)
        gold = (unit + state) % 2
        choices = [case["wrong"], case["correct"]] if gold else [case["correct"], case["wrong"]]
        option_text = f"Options: A) {choices[0]} B) {choices[1]}" if language == "en" else f"选项：A）{choices[0]} B）{choices[1]}"
        instruction = "Reply with only A or B." if language == "en" else "只回答A或B。"
        free_instruction = "Answer with only the answer word." if language == "en" else "只回答答案词。"
        case_id = f"c2501-{family}-{language}-{surface}-u{unit:02d}-s{state}"
        rows.append({
            "case_id": case_id,
            "family": family,
            "language": language,
            "surface": surface,
            "unit": unit,
            "state": state,
            "partition": partition(unit),
            "gold_position": gold,
            "correct_answer": case["correct"],
            "wrong_answer": case["wrong"],
            "prompt_core": case["core"],
            "prompt": f"{case['core']} {option_text} {instruction}",
            "free_prompt": f"{case['core']} {free_instruction}",
            "role_values": case["roles"],
            "semantic_graph": case["graph"],
            "design_index": family_i,
        })
    return rows


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 独立自然双语样本条件坐标大合同（{CAMPAIGN}） [{stamp}]

**证据审查。** Phase2281-2287 的严格账本支持 Qwen3-4B 的 `3/10` 个 q0 跨语言与 `6/13` 个跨表面观察阳性、修正后 `0/23` 个非退化析因 confirmation、Qwen3-14B 行为 `3/3` 合格但内部 confirmation `0/3`，以及因果分支 `NA`。保留“输入与浅层存在模型本地逐坐标可预测结构”；拒绝把它命名为深层语义算子、跨规模同构、超边、流形曲率或新数学。持久同调、Fisher 度量、规范场和微分同胚目前没有独立识别条件，不进入主裁决。

**测试原理、用例与冻结对象。** 本期在模型运行前冻结八类自然程序：施事-受事、持有查询、关系从句、位置、时间、比较、态度-事件（“某人喜欢/高兴某人吃了某物”）和两跳分类链。每类使用中英两种语言、叙述/对话两种表面、32套词汇、真假两状态和 discovery/confirmation/fresh-confirmation/fresh-lockbox 四分区，共 `{result['material']['rows']}` 行。正确答案位置在每个分区严格配额平衡。机器语义编译、角色跨度、候选唯一性和词汇分区审计完成；独立人类盲评模板已保存，但本轮人评为 `NA_not_run`，因此不得声称自然度已经被人类确认。

**统一公式。** 保留每个样本、检查点、角色和物理坐标的状态与响应：

$$
\mathcal F_i=(H_{{i,q,r,j}})_{{q,r,j}},\qquad
R_{{i,q,r,j}}=H_{{i,q,r,j}}^{{(1)}}-H_{{i,q,r,j}}^{{(0)}}.
$$

后续只用 discovery 拟合样本条件逐坐标函数，并强制与目标均值、共享构式、错族、打乱标签和上一检查点控制比较：

$$
\widehat R_{{i,q,r,j}}=g_{{f,q,r,j}}\!\left(H_{{i,q,r,j}}^{{(0)}},H_{{i,q-1,r,j}}^{{(0)}}\right).
$$

因果掩码密度冻结为 `{list(CAUSAL_DENSITIES)}`，剂量冻结为 `{list(CAUSAL_DOSES)}`；任何构式只有在未揭盲 fresh lockbox 前获得中层前瞻资格才能进入 delete/call/rescue。Qwen3-14B 只复验 Qwen3-4B 的冻结功能拓扑，不对齐物理坐标编号。

**结果汇总与审计。** 材料审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；配置与材料哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；全部检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**理论进展、硬伤与结论。** 本期只建立独立自然双语观察合同，没有模型机制结果。主要硬伤是研究者编写材料、人类盲评缺失、二元选项/短答案接口和小模型限制。下一阶段授权：按冻结材料先运行 Qwen3-4B 双行为；只有行为合格族进入 embedding、36个 block 后状态、final norm、六角色和全部2560坐标采集，同时为代表病例保存全部真实 token。脚本 `tests/glm5/phase2288_c2501_c2580_natural_sample_condition_contract.py`；结果 `tests/glm5/result/phase2288_c2501_c2580_natural_sample_condition_contract`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    from transformers import AutoTokenizer

    rows = material()
    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    compiled = compile_rows(tokenizer, rows)
    material_path = OUT / "material/natural_bilingual_cases.jsonl"
    compiled_path = OUT / "material/qwen_compiled.jsonl"
    write_rows(material_path, rows)
    write_rows(compiled_path, compiled)
    review_rows = [{
        "case_id": row["case_id"], "naturalness_1_5": None,
        "semantic_unique_0_1": None, "cross_surface_equivalent_0_1": None,
        "reviewer": None,
    } for row in rows if row["partition"] == "fresh_lockbox"]
    write_rows(OUT / "external/human_blind_template.jsonl", review_rows)

    counts = Counter((row["family"], row["language"], row["surface"], row["partition"]) for row in rows)
    balance = defaultdict(lambda: [0, 0])
    for row in rows:
        balance[(row["family"], row["language"], row["surface"], row["partition"])][row["gold_position"]] += 1
    role_failures = [{"case_id": row["case_id"], "role": role}
                     for row in compiled for role in ROLES if role not in row["role_positions"]]
    position_methods = Counter(method for row in compiled for method in row["role_position_methods"].values())
    partition_vocab = defaultdict(set)
    for row in rows:
        partition_vocab[row["partition"]].add(row["role_values"]["primary"])
    overlaps = {f"{a}|{b}": sorted(partition_vocab[a] & partition_vocab[b])
                for a, b in itertools.combinations(PARTITION_RANGES, 2)}
    widths = [len(row["prompt_ids"]) for row in compiled]
    audit = {
        "rows": len(rows), "families": list(FAMILIES), "languages": list(LANGUAGES),
        "surfaces": list(SURFACES), "partitions": {p: sum(r["partition"] == p for r in rows) for p in PARTITION_RANGES},
        "cell_count_min_max": [min(counts.values()), max(counts.values())],
        "candidate_balance_exact": all(a == b for a, b in balance.values()),
        "role_position_failures": role_failures,
        "role_position_methods": dict(position_methods),
        "primary_vocab_cross_partition_overlap": overlaps,
        "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "human_review": "NA_not_run",
        "semantic_uniqueness": "compiler_deterministic_not_human_validated",
    }
    config = {
        "phase": PHASE, "campaign": CAMPAIGN, "families": list(FAMILIES),
        "languages": list(LANGUAGES), "surfaces": list(SURFACES), "units": UNITS,
        "partitions": {name: list(values) for name, values in PARTITION_RANGES.items()},
        "roles": list(ROLES), "behavior_gate": BEHAVIOR_GATE,
        "prediction_gates": PREDICTION_GATES,
        "causal_densities": CAUSAL_DENSITIES, "causal_doses": CAUSAL_DOSES,
        "causal_gates": CAUSAL_GATES,
        "excluded_primary_methods": ["PCA", "Top-K", "cosine screening", "mean donor-delta transport",
                                     "persistent homology", "Fisher geometry", "manifold alignment"],
        "model_order": ["Qwen3-4B", "Qwen3-14B_if_authorized"],
    }
    save(OUT / "config/frozen_contract.json", config)
    save(OUT / "audit/material_audit.json", audit)
    hashes = {"material": file_hash(material_path), "compiled": file_hash(compiled_path),
              "config": file_hash(OUT / "config/frozen_contract.json")}
    checks = {
        "row_count": len(rows) == len(FAMILIES) * len(LANGUAGES) * len(SURFACES) * UNITS * 2,
        "all_cells_present": len(counts) == len(FAMILIES) * len(LANGUAGES) * len(SURFACES) * len(PARTITION_RANGES),
        "candidate_balance": audit["candidate_balance_exact"],
        "all_roles_compiled": not role_failures,
        "partition_primary_vocab_disjoint": all(not values for values in overlaps.values()),
        "four_partitions": set(audit["partitions"]) == set(PARTITION_RANGES),
        "human_review_honest_na": audit["human_review"] == "NA_not_run",
        "no_advanced_math_assumed": all(name in config["excluded_primary_methods"] for name in
                                          ("persistent homology", "Fisher geometry", "manifold alignment")),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "material": {"rows": len(rows), "path": str(material_path.relative_to(ROOT)),
                                             "compiled": str(compiled_path.relative_to(ROOT))},
        "audit": audit, "config": config, "hashes": hashes, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "A new natural bilingual, multi-family, sample-conditioned full-coordinate campaign is frozen; no model or mechanism result exists yet.",
        "next_authorization": "Run Qwen3-4B dual behavior and capture only behavior-qualified full-coordinate fields.",
    }
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
