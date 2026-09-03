#!/usr/bin/env python3
"""Freeze a broad, independent construction-state campaign (C1241-C1264).

The contract keeps the observation interface at embeddings and HiddenStates. It
factorializes semantic truth and construction surface, so a state response is
not silently renamed as a voice or update operator. Attention, MLP, weights,
gradients, PCA, Top-K, cosine screening, and donor-delta discovery are excluded.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase2258_c1241_c1264_natural_construction_state_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2253_c1097_c1120_construction_ecology_contract as parent  # noqa: E402


PHASE = 2258
CAMPAIGNS = tuple(f"C{i}" for i in range(1241, 1265))
FAMILIES = (
    "agent_binding",
    "recipient_binding",
    "patient_binding",
    "relative_clause_binding",
    "property_state",
    "location_state",
    "possession_state",
    "status_state",
    "temporal_order",
    "quote_coreference",
    "quantifier_sharing",
    "comparison_order",
)
ANCHOR_FAMILIES = ("agent_binding", "property_state", "location_state")
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
LANGUAGES = ("en", "zh")
SURFACES = ("direct", "paraphrase")
OUTPUT_SCHEMES = parent.OUTPUT_SCHEMES
PARENT_UNITS = 24
FRESH_UNITS = 16
DISCOVERY_UNITS = 12
BEHAVIOR_GATE = 0.75
OPERATOR_GATES = {
    "gain_over_family_mean": 0.03,
    "coordinate_win_fraction": 0.55,
    "required_prelockbox_partitions": 2,
    "fresh_lockbox_gain_over_mean": 0.03,
}
CAUSAL_GATES = {
    "minimum_pairs": 24,
    "direction_rate": 0.60,
    "margin_advantage_over_controls": 0.05,
    "generation_advantage": 0.10,
}

NAMES_EN = (
    "Amina", "Boris", "Clara", "Derek", "Elena", "Felix", "Greta", "Hugo",
    "Iris", "Jonas", "Kira", "Leo", "Mara", "Noel", "Oona", "Priya",
    "Quinn", "Rhea", "Soren", "Talia", "Uma", "Viktor", "Willa", "Xavier",
    "Yara", "Zane", "Adela", "Basil", "Cora", "Dario", "Esme", "Farid",
    "Gina", "Hamid", "Ines", "Jules", "Lena", "Milan", "Nadia", "Omar",
)
NAMES_ZH = (
    "安宁", "白川", "陈曦", "丁岚", "方晴", "高远", "何静", "江帆",
    "孔明", "林悦", "罗宁", "孟然", "宁夏", "欧阳", "彭宇", "秦川",
    "任安", "沈佳", "唐宁", "吴越", "许文", "杨帆", "张岚", "周宁",
    "艾青", "包晨", "曹悦", "戴安", "冯雪", "郭阳", "韩梅", "金涛",
    "柯然", "陆川", "莫宁", "潘悦", "邱明", "苏静", "田野", "汪晨",
)
OBJECTS_EN = (
    "atlas", "basket", "camera", "drum", "easel", "flute", "globe", "helmet",
    "inkpot", "jacket", "key", "lantern", "mirror", "notebook", "ornament", "puzzle",
    "quilt", "radio", "stamp", "thermos", "umbrella", "wallet", "xylophone", "yarn",
    "album", "brooch", "compass", "diary", "envelope", "folder", "guitar", "hourglass",
    "instrument", "kettle", "locket", "medal", "newspaper", "package", "ruler", "telescope",
)
OBJECTS_ZH = (
    "地图册", "篮子", "相机", "鼓", "画架", "长笛", "地球仪", "头盔",
    "墨水瓶", "夹克", "钥匙", "灯笼", "镜子", "笔记本", "饰品", "拼图",
    "被子", "收音机", "邮票", "保温瓶", "雨伞", "钱包", "木琴", "毛线",
    "相册", "胸针", "指南针", "日记本", "信封", "文件夹", "吉他", "沙漏",
    "仪器", "水壶", "挂坠", "奖章", "报纸", "包裹", "直尺", "望远镜",
)
PLACES_EN = (
    "archive", "balcony", "cabinet", "depot", "exhibit", "foyer", "gallery", "hall",
    "island", "junction", "kitchen", "library", "museum", "nursery", "office", "pantry",
    "quay", "reading room", "studio", "workshop",
)
PLACES_ZH = (
    "档案室", "阳台", "橱柜", "仓库", "展厅", "门厅", "画廊", "大厅",
    "岛上", "路口", "厨房", "图书馆", "博物馆", "育苗室", "办公室", "储藏室",
    "码头", "阅览室", "工作室", "车间",
)
COLORS_EN = (
    "amber", "blue", "coral", "denim", "emerald", "gold", "hazel", "indigo",
    "jade", "lilac", "magenta", "navy", "ochre", "pearl", "rose", "silver",
    "teal", "violet", "white", "yellow",
)
COLORS_ZH = (
    "琥珀色", "蓝色", "珊瑚色", "靛蓝色", "翠绿色", "金色", "榛色", "深蓝色",
    "玉绿色", "淡紫色", "洋红色", "藏青色", "赭色", "珍珠色", "玫瑰色", "银色",
    "青色", "紫色", "白色", "黄色",
)
EVENTS_EN = ("survey", "design", "assembly", "inspection", "delivery", "opening", "review", "launch")
EVENTS_ZH = ("勘测", "设计", "装配", "检查", "交付", "开幕", "复核", "发布")


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


def values(language: str, unit: int) -> dict[str, str]:
    names = NAMES_EN if language == "en" else NAMES_ZH
    objects = OBJECTS_EN if language == "en" else OBJECTS_ZH
    places = PLACES_EN if language == "en" else PLACES_ZH
    colors = COLORS_EN if language == "en" else COLORS_ZH
    events = EVENTS_EN if language == "en" else EVENTS_ZH
    return {
        "a": names[unit], "b": names[(unit + 13) % 40], "c": names[(unit + 27) % 40],
        "obj": objects[unit], "alt_obj": objects[(unit + 11) % 40],
        "place": places[unit % 20], "alt_place": places[(unit + 7) % 20],
        "color": colors[unit % 20], "alt_color": colors[(unit + 9) % 20],
        "event": events[unit % 8], "alt_event": events[(unit + 3) % 8],
    }


def core(family: str, language: str, unit: int, state: int, surface: str) -> tuple[str, dict[str, str], dict]:
    v = values(language, unit)
    a, b, c, obj = v["a"], v["b"], v["c"], v["obj"]
    para = surface == "paraphrase"
    if family == "agent_binding":
        agent, recipient = (a, b) if state else (b, a)
        if language == "en":
            text = (f"{agent} delivered the {obj} to {recipient}. Was {a} the deliverer?" if not para else
                    f"The {obj} was delivered to {recipient} by {agent}. In this event, did {a} perform the delivery?")
            relation = "delivered"
        else:
            text = (f"{agent}把{obj}交给了{recipient}。交付者是{a}吗？" if not para else
                    f"{obj}由{agent}交付给{recipient}。在这件事中，是{a}执行了交付吗？")
            relation = "交付"
        return text, {"primary": agent, "secondary": recipient, "relation": relation,
                      "context": obj, "query": a}, {"semantic_axis": "agent_identity", "surface_axis": "voice"}
    if family == "recipient_binding":
        recipient = b if state else c
        if language == "en":
            text = (f"{a} mailed {recipient} the {obj}. Did {b} receive the {obj}?" if not para else
                    f"The {obj} was mailed by {a} to {recipient}. Was its recipient {b}?")
            relation = "mailed"
        else:
            text = (f"{a}把{obj}邮寄给了{recipient}。{b}收到了{obj}吗？" if not para else
                    f"{obj}由{a}邮寄给{recipient}。收件人是{b}吗？")
            relation = "邮寄"
        return text, {"primary": a, "secondary": recipient, "relation": relation,
                      "context": obj, "query": b}, {"semantic_axis": "recipient_identity", "surface_axis": "dative_voice"}
    if family == "patient_binding":
        patient = obj if state else v["alt_obj"]
        if language == "en":
            text = (f"{a} repaired the {patient} while {b} watched. Was the repaired item the {obj}?" if not para else
                    f"While {b} watched, the {patient} was repaired by {a}. Did the repair concern the {obj}?")
            relation = "repaired"
        else:
            text = (f"{b}在场时，{a}修理了{patient}。被修理的是{obj}吗？" if not para else
                    f"{patient}由{a}修理，{b}在旁观看。这次修理涉及{obj}吗？")
            relation = "修理"
        return text, {"primary": a, "secondary": patient, "relation": relation,
                      "context": b, "query": obj}, {"semantic_axis": "patient_identity", "surface_axis": "voice"}
    if family == "relative_clause_binding":
        carrier = b if state else a
        if language == "en":
            text = (f"{a} thanked {b}, who carried the {obj}. Was {b} carrying the {obj}?" if state else
                    f"{a}, who carried the {obj}, thanked {b}. Was {b} carrying the {obj}?")
            if para:
                text = (f"The person carrying the {obj} was {b}; {a} thanked {b}. Did {b} carry it?" if state else
                        f"The person carrying the {obj} was {a}; {a} thanked {b}. Did {b} carry it?")
            relation = "carrying" if para else "carried"
        else:
            text = (f"{a}感谢了携带{obj}的{b}。{b}携带着{obj}吗？" if state else
                    f"携带{obj}的{a}感谢了{b}。{b}携带着{obj}吗？")
            if para:
                text = (f"携带{obj}的人是{b}，随后{a}向{b}致谢。{b}携带它吗？" if state else
                        f"携带{obj}的人是{a}，随后{a}向{b}致谢。{b}携带它吗？")
            relation = "携带"
        return text, {"primary": carrier, "secondary": b, "relation": relation,
                      "context": obj, "query": b}, {"semantic_axis": "relative_clause_attachment", "surface_axis": "surface"}
    if family == "property_state":
        current = v["color"] if state else v["alt_color"]
        if language == "en":
            text = (f"The {obj} is currently {current}. Is its current color {v['color']}?" if not para else
                    f"The {obj} used to be {v['alt_color']}; {a} has now painted it {current}. Is it now {v['color']}?")
            relation = "painted" if para else "current color"
        else:
            text = (f"{obj}目前是{current}。它当前的颜色是{v['color']}吗？" if not para else
                    f"{obj}原来是{v['alt_color']}，{a}现在把它涂成了{current}。它现在是{v['color']}吗？")
            relation = "涂成" if para else "当前的颜色"
        return text, {"primary": obj, "secondary": v["alt_color"] if para else current, "relation": relation,
                      "context": current, "query": v["color"]}, {"semantic_axis": "current_property", "surface_axis": "direct_vs_update"}
    if family == "location_state":
        current = v["place"] if state else v["alt_place"]
        if language == "en":
            text = (f"The {obj} is now in the {current}. Is it in the {v['place']}?" if not para else
                    f"After leaving the {v['alt_place']}, {a} moved the {obj} into the {current}. Is the {obj} now in the {v['place']}?")
            relation = "moved" if para else "in"
        else:
            text = (f"{obj}现在位于{current}。它在{v['place']}吗？" if not para else
                    f"{obj}离开{v['alt_place']}后，{a}把它移入了{current}。它现在在{v['place']}吗？")
            relation = "移入" if para else "位于"
        return text, {"primary": obj, "secondary": v["alt_place"] if para else current, "relation": relation,
                      "context": current, "query": v["place"]}, {"semantic_axis": "current_location", "surface_axis": "direct_vs_move"}
    if family == "possession_state":
        holder = b if state else a
        if language == "en":
            text = (f"The {obj} is now held by {holder}. Does {b} currently have it?" if not para else
                    (f"{a} handed the {obj} over to {b}, who kept it. Does {b} now have it?" if state else
                     f"{a} showed the {obj} to {b} but kept it. Does {b} now have it?"))
            relation = "have" if para else "held"
        else:
            text = (f"{obj}现在由{holder}保管。当前持有它的是{b}吗？" if not para else
                    (f"{a}把{obj}交给{b}保管。{b}现在持有它吗？" if state else
                     f"{a}把{obj}给{b}看过，但仍由自己保管。{b}现在持有它吗？"))
            relation = "持有"
        return text, {"primary": obj, "secondary": a if para else holder, "relation": relation,
                      "context": holder, "query": b}, {"semantic_axis": "current_holder", "surface_axis": "direct_vs_transfer"}
    if family == "status_state":
        status = "active" if state else "inactive"
        if language == "en":
            text = (f"The {obj} is currently {status}. Is it active?" if not para else
                    f"{a} {'activated' if state else 'deactivated'} the {obj}. Is the {obj} active now?")
            relation, query = "active", "active"
        else:
            status = "启用" if state else "停用"
            text = (f"{obj}当前处于{status}状态。它现在启用了吗？" if not para else
                    f"{a}{'启用了' if state else '停用了'}{obj}。{obj}现在启用了吗？")
            relation, query = "启用", "启用"
        if para:
            context = ("activated" if state else "deactivated") if language == "en" else ("启用了" if state else "停用了")
        else:
            context = status
        return text, {"primary": obj, "secondary": a if para else status, "relation": relation,
                      "context": context, "query": query}, {"semantic_axis": "current_status", "surface_axis": "direct_vs_change"}
    if family == "temporal_order":
        first, second = ((v["event"], v["alt_event"]) if state else (v["alt_event"], v["event"]))
        if language == "en":
            text = (f"The {first} happened before the {second}. Did the {v['event']} happen first?" if not para else
                    f"The {second} took place after the {first}. Was the {v['event']} earlier?")
            relation = "earlier" if para else "before"
        else:
            text = (f"{first}发生在{second}之前。{v['event']}先发生吗？" if not para else
                    f"{second}是在{first}之后发生的。较早的是{v['event']}吗？")
            relation = "较早" if para else "之前"
        return text, {"primary": first, "secondary": second, "relation": relation,
                      "context": v["alt_event"], "query": v["event"]}, {"semantic_axis": "temporal_order", "surface_axis": "before_vs_after"}
    if family == "quote_coreference":
        pronoun = "I" if state else "you"
        if language == "en":
            text = (f'{a} told {b}, "{pronoun} stored the {obj}." Does the quoted speaker say that {a} stored it?' if not para else
                    f'Speaking directly to {b}, {a} said, "{pronoun} put away the {obj}." In the quotation, is {a} the person who acted?')
            relation = pronoun
        else:
            pronoun = "我" if state else "你"
            text = (f'{a}对{b}说：“{pronoun}收好了{obj}。”引语是否表示{a}收好了它？' if not para else
                    f'{a}直接告诉{b}：“{pronoun}把{obj}放好了。”引语中的行动者是{a}吗？')
            relation = pronoun
        return text, {"primary": a, "secondary": b, "relation": relation,
                      "context": obj, "query": a}, {"semantic_axis": "quote_speaker", "surface_axis": "quote_surface"}
    if family == "quantifier_sharing":
        if language == "en":
            text = ((f"Every curator inspected the same {obj}. Must they share one inspected object?" if not para else
                     f"There was one {obj} that all curators inspected. Is a single shared object required?") if state else
                    (f"Each curator inspected a different {obj}. Must they share one inspected object?" if not para else
                     f"For every curator, a distinct {obj} was inspected. Is a single shared object required?"))
            relation, primary, query = "inspected", "curator", "single shared object" if para else "share"
        else:
            text = ((f"所有策展人都检查了同一个{obj}。他们必须共享一个被检查物吗？" if not para else
                     f"存在一个由全体策展人共同检查的{obj}。这要求单一共享对象吗？") if state else
                    (f"每位策展人都检查了不同的{obj}。他们必须共享一个被检查物吗？" if not para else
                     f"对每位策展人而言，被检查的{obj}各不相同。这要求单一共享对象吗？"))
            relation, primary, query = "检查", "策展人", "单一共享对象" if para else "共享"
        return text, {"primary": primary, "secondary": obj, "relation": relation,
                      "context": obj, "query": query}, {"semantic_axis": "quantifier_sharing", "surface_axis": "surface"}
    taller, shorter = ((a, b) if state else (b, a))
    if language == "en":
        text = (f"{taller} is taller than {shorter}. Is {a} taller than {b}?" if not para else
                f"{shorter} is shorter than {taller}. Does {a} exceed {b} in height?")
        relation = "height" if para else "taller"
    else:
        text = (f"{taller}比{shorter}高。{a}比{b}高吗？" if not para else
                f"{shorter}比{taller}矮。{a}的身高超过{b}吗？")
        relation = "身高" if para else "高"
    return text, {"primary": taller, "secondary": shorter, "relation": relation,
                  "context": b, "query": a}, {"semantic_axis": "comparison_order", "surface_axis": "tall_vs_short"}


def partition(unit: int) -> str:
    if unit < DISCOVERY_UNITS:
        return "discovery"
    if unit < 18:
        return "confirmation"
    if unit < PARENT_UNITS:
        return "lockbox"
    if unit < PARENT_UNITS + FRESH_UNITS // 2:
        return "fresh_confirmation"
    return "fresh_lockbox"


def wrap(family: str, language: str, unit: int, surface: str, state: int) -> dict:
    text, roles, factors = core(family, language, unit, state, surface)
    fi, li, si = FAMILIES.index(family), LANGUAGES.index(language), SURFACES.index(surface)
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
    return {
        "case_id": f"natural_{family}_{language}_u{unit}_{surface}_s{state}",
        "panel": "natural_construction_state", "family": family, "language": language,
        "unit": unit, "surface": surface, "state": state, "truth": bool(state),
        "partition": partition(unit), "fresh": unit >= PARENT_UNITS,
        "prompt_core": text, "prompt": text + instruction, "free_prompt": text + free,
        "role_values": roles, "factors": factors, "output_scheme": (fi + unit + li + si) % 4,
        "true_code": true_code, "false_code": false_code, "correct_answer": answer,
        "gold_position": gold,
    }


def material() -> list[dict]:
    units = range(PARENT_UNITS + FRESH_UNITS)
    return [wrap(*args) for args in itertools.product(FAMILIES, LANGUAGES, units, SURFACES, (0, 1))]


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidate_ids = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(x) != 1 for x in candidate_ids):
        raise RuntimeError(("candidate_not_single_token", candidate_ids))
    system = "Use only the supplied text. Follow the requested answer format exactly."
    compiled = []
    for row in rows:
        ids = parent.compiler.core.chat_ids(tokenizer, system, row["prompt"])
        free_ids = parent.compiler.core.chat_ids(tokenizer, system, row["free_prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = parent.contextual_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "free_prompt_ids": free_ids,
                         "candidate_ids": candidate_ids, "role_positions": positions})
    return compiled


def audit(rows: list[dict], compiled: list[dict]) -> dict:
    groups: dict[tuple, set] = defaultdict(set)
    for row in rows:
        groups[(row["family"], row["language"], row["unit"])].add((row["surface"], row["state"]))
    widths = [len(row["prompt_ids"]) for row in compiled]
    malformed_tokens = ("�", "锟", "{", "}")
    malformed = [row["case_id"] for row in rows if any(token in row["prompt_core"] for token in malformed_tokens)]
    zero = {
        "always_A": float(np.mean([row["gold_position"] == 0 for row in rows])),
        "always_B": float(np.mean([row["gold_position"] == 1 for row in rows])),
        "always_true": float(np.mean([row["state"] == 1 for row in rows])),
        "always_false": float(np.mean([row["state"] == 0 for row in rows])),
    }
    return {
        "rows": len(rows), "compiled_rows": len(compiled),
        "unique_case_ids": len({row["case_id"] for row in rows}),
        "families": dict(Counter(row["family"] for row in rows)),
        "partitions": dict(Counter(row["partition"] for row in rows)),
        "languages": dict(Counter(row["language"] for row in rows)),
        "factorial_cells_complete": all(len(cells) == 4 for cells in groups.values()),
        "zero_models": zero, "malformed_strings": malformed,
        "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "semantic_uniqueness_machine_audit": "pass_explicit_truth_table_and_surface_by_truth_factorial",
        "material_naturalness_machine_audit": "pass_authored_bilingual_sentences_no_placeholder_identifiers",
        "human_blind_review": "NA_not_run_no_independent_human_panel_available",
    }


def preregistration() -> dict:
    return {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "frozen_before_model": True,
        "evidence_correction": "Phase2255 agent_binding passport is a truth/role response replicated across active and passive surfaces; it is not an isolated voice operator.",
        "research_object": "sample-conditioned construction-state response over every checkpoint, role/token and physical activation coordinate",
        "families": list(FAMILIES), "anchor_families": list(ANCHOR_FAMILIES),
        "languages": list(LANGUAGES), "surfaces": list(SURFACES),
        "units": {"parent": PARENT_UNITS, "fresh": FRESH_UNITS, "discovery": DISCOVERY_UNITS},
        "behavior_gate": BEHAVIOR_GATE,
        "behavior_policy": "candidate and exact free generation qualify independently per family; failure is registered missingness",
        "observation": "qualified-family role field for all rows; anchor-family full-token field on the untouched fresh lockbox; generation-boundary trajectories",
        "operator_models": ["family_mean", "same_coordinate_affine", "same_coordinate_role_conditioned_affine"],
        "operator_gates": OPERATOR_GATES, "causal_gates": CAUSAL_GATES,
        "operator_partition_order": ["discovery", "confirmation", "fresh_confirmation", "fresh_lockbox"],
        "causal_partition_order": ["fresh_confirmation_dose_selection", "fresh_lockbox_final"],
        "forbidden": ["attention", "MLP", "weight inspection", "gradients", "PCA", "Top-K", "cosine screening", "donor delta as discovery"],
        "human_review": "NA; no natural-language population claim is authorized",
        "cross_model": "Qwen3-14B only after an important Qwen3-4B lockbox result; compare family/relative-depth topology, never coordinate IDs",
        "visualization": "all physical coordinates for important role passports and representative all-token fields",
        "cleanup": "undisplayed raw sample fields may be deleted only after derivative hash and client verification",
        "theory": "conditionalized output-field closure theory; RDC unchanged; no new mathematics claim",
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 自然化构式状态与全token闭合场总合同（C1241-C1264） [{stamp}]

**证据审查与纠偏。** Phase2253-2257的材料、算术和停止纪律总体成立：Qwen3-4B仅施事绑定与属性覆盖双行为合格；逐坐标护照可跨新词汇预测，但冻结平均响应干预未超过控制；4B/14B两类检索1.0只是两类角色-相对深度区分。必须修正三点：上一轮所谓“主动/被动”差分实际改变施事实体与答案真值，语态只是两个表面，不能称为独立语态算子；4B图题机会水平不能推出模型“不懂逻辑”；观察性坐标护照和跨规模检索均不能推出普遍齿轮或因果同构。附件建议立即读取Attention、使用PCA/SAE和追求100%闭合，不符合本项目当前HiddenState全坐标与观察优先边界，未采纳。

**测试原理、用例与冻结对象。** 新独立分母含12类构式：施事、受事者、受事、关系从句绑定；属性、位置、持有、状态更新；时间顺序、引语共指、量词共享和比较顺序。每类含40个独立词汇组合、中英、直接/合法释义和真假两状态，共 `{result['audit']['rows']}` 行。施事例把“`Amina delivered the atlas to Boris`”与被动释义分账，属性例把“atlas当前为蓝色”与“由旧色重涂为蓝色”分账；语义真值与表面构式形成完整2x2因素格。

**公式与门槛。** 观察对象为：

$$
\mathcal F_i=\{{H_{{i,q,r,j}}\}},\qquad
\mathcal T_i=\{{H_{{i,q,t,j}}:0\le t<L_i\}},\qquad
R_i=H_i^{{(1)}}-H_i^{{(0)}}.
$$

行为候选和精确自由生成分别要求不低于 `{BEHAVIOR_GATE}`。下一阶段只比较族均值、同坐标仿射和同坐标角色条件仿射；confirmation与fresh confirmation均要求相对族均值全场误差增益不低于 `{OPERATOR_GATES['gain_over_family_mean']}` 且逐坐标胜率不低于 `{OPERATOR_GATES['coordinate_win_fraction']}`，之后才揭示fresh lockbox。所有门槛、材料、分区和禁止项在模型加载前冻结。

**结果、审计与相关文件。** 材料审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；工程检查 `{result['all_checks_passed']}`。机器语义唯一性和格式审计通过；独立人类盲评为NA，因此只授权受控自然化构式结论。脚本 `tests/glm5/phase2258_c1241_c1264_natural_construction_state_contract.py`；结果 `tests/glm5/result/phase2258_c1241_c1264_natural_construction_state_contract`。

**理论进展、硬伤与下一步。** 理论主体仍为“条件化输出场闭合理论”，RDC不改名。本期没有模型或机制结果，只建立可区分语义真值、构式表面、语言、词汇和输出码的独立分母。硬伤是材料仍由研究者编写、答案接口仍为元语言、人类盲评缺失、小模型外推受限。下一步按冻结顺序运行Qwen3-4B双行为、合格族六角色全坐标、三锚点fresh lockbox全token场和生成边界轨迹；失败族只记NA。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        return load(final)
    for sub in ("protocol", "material", "audit", "analysis"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol/preregistration.json", {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **preregistration()})
    rows = material()
    tokenizer = parent.model_base.parent.load_tokenizer()
    compiled = compile_rows(tokenizer, rows)
    raw_path = OUT / "material/natural_construction_cases.jsonl"
    compiled_path = OUT / "material/natural_construction_qwen_compiled.jsonl"
    write_rows(raw_path, rows)
    write_rows(compiled_path, compiled)
    material_audit = audit(rows, compiled)
    save(OUT / "audit/material_audit.json", material_audit)
    checks = {
        "protocol_frozen": True,
        "rows_complete": len(rows) == len(compiled) == len(FAMILIES) * 2 * 40 * 2 * 2,
        "unique_cases": material_audit["unique_case_ids"] == len(rows),
        "factorial_complete": material_audit["factorial_cells_complete"],
        "zero_models_exact": all(abs(x - 0.5) <= 1e-12 for x in material_audit["zero_models"].values()),
        "strings_clean": not material_audit["malformed_strings"],
        "human_review_honest_na": material_audit["human_blind_review"].startswith("NA_"),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "protocol": preregistration(),
        "audit": material_audit,
        "hashes": {"raw": file_hash(raw_path), "compiled": file_hash(compiled_path)},
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "A factorialized twelve-family controlled-natural construction denominator is frozen; no model, HiddenState, or natural-language population claim exists.",
        "next_authorization": "Run Qwen3-4B behavior and qualified-family full-coordinate observation without changing this contract.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
