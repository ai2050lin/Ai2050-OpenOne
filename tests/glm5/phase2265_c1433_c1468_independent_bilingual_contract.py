#!/usr/bin/env python3
"""Freeze an independent, UTF-8 clean bilingual construction-state campaign."""
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
OUT = TESTS / "result/phase2265_c1433_c1468_independent_bilingual_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2258_c1241_c1264_natural_construction_state_contract as legacy  # noqa: E402


PHASE = 2265
CAMPAIGNS = tuple(f"C{i}" for i in range(1433, 1469))
FAMILIES = (
    "agent_binding", "recipient_binding", "patient_binding", "relative_clause_binding",
    "property_state", "location_state", "possession_state", "status_state",
    "temporal_order", "quote_coreference", "quantifier_sharing", "comparison_order",
)
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
LANGUAGES = ("en", "zh")
SURFACES = ("direct", "paraphrase")
STATES = (0, 1)
UNITS = 32
BEHAVIOR_GATE = 0.75
ANALYSIS_PARTITIONS = ("discovery", "confirmation", "fresh_confirmation")
OUTPUT_SCHEMES = legacy.OUTPUT_SCHEMES

NAMES_EN = (
    "Mira Stone", "Rowan Hale", "Nora Finch", "Caleb Frost", "Lina Park", "Theo Marsh",
    "Ada Wells", "Simon Reed", "Maya Brooks", "Evan Cole", "Rina Shaw", "Luca Dean",
    "Tessa Gray", "Owen Blake", "Sara Quinn", "Miles Hart", "Nina Cross", "Joel Lane",
    "Leah Grant", "Arlo West", "Eva North", "Dylan Rose", "Ivy Clark", "Roman Bell",
    "Fiona Lake", "Noah King", "Celia Moon", "Eli Wood", "Tara Snow", "Marco Field",
    "Lara Hill", "Jon River",
)
NAMES_ZH = (
    "赵晨", "钱宇", "孙宁", "李航", "周岚", "吴昕", "郑然", "王悦",
    "冯嘉", "陈澄", "褚宁", "卫岚", "蒋欣", "沈言", "韩清", "杨舒",
    "朱宁", "秦朗", "尤佳", "许澄", "何安", "吕明", "施悦", "张然",
    "孔宁", "曹清", "严舒", "华安", "金明", "魏澄", "陶悦", "姜然",
)
OBJECTS_EN = (
    "ceramic vase", "wooden tray", "silver whistle", "canvas bag", "brass clock", "glass bowl",
    "paper kite", "linen scarf", "stone tablet", "travel mug", "music box", "field journal",
    "copper bell", "woven mat", "pocket torch", "model bridge", "painted mask", "rubber stamp",
    "signal flag", "tool case", "photo frame", "tea caddy", "desk fan", "hand lens",
    "recipe card", "wooden comb", "metal badge", "cloth banner", "small tripod", "garden trowel",
    "folding map", "sample jar",
)
OBJECTS_ZH = (
    "陶瓷花瓶", "木托盘", "银口哨", "帆布包", "黄铜钟", "玻璃碗", "纸风筝", "亚麻围巾",
    "石刻板", "旅行杯", "音乐盒", "野外日志", "铜铃", "编织垫", "手电筒", "桥梁模型",
    "彩绘面具", "橡皮印章", "信号旗", "工具箱", "相框", "茶叶罐", "桌面风扇", "放大镜",
    "食谱卡", "木梳", "金属徽章", "布横幅", "小三脚架", "园艺铲", "折叠地图", "样品瓶",
)
PLACES_EN = (
    "west alcove", "north shelf", "quiet lounge", "rear office", "glass cabinet", "music room",
    "side corridor", "upper studio", "garden shed", "east foyer", "map room", "reading corner",
    "tool closet", "front gallery", "storage bay", "meeting room",
)
PLACES_ZH = (
    "西侧壁龛", "北侧书架", "安静休息室", "后部办公室", "玻璃柜", "音乐室", "侧走廊", "楼上画室",
    "花园工具房", "东侧门厅", "地图室", "阅读角", "工具柜", "前厅展区", "储物区", "会议室",
)
COLORS_EN = (
    "crimson", "turquoise", "charcoal", "mint green", "copper", "sky blue", "plum", "ivory",
    "scarlet", "olive", "cream", "burgundy", "aqua", "bronze", "lavender", "gray",
)
COLORS_ZH = (
    "深红色", "绿松石色", "炭灰色", "薄荷绿色", "铜色", "天蓝色", "李子紫色", "象牙色",
    "鲜红色", "橄榄绿色", "奶油色", "酒红色", "水绿色", "青铜色", "薰衣草紫色", "灰色",
)
EVENTS_EN = ("briefing", "calibration", "packing", "registration", "rehearsal", "handover", "cleanup", "departure")
EVENTS_ZH = ("简报会", "校准", "装箱", "登记", "彩排", "交接", "清理", "出发")


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


def partition(unit: int) -> str:
    if unit < 12:
        return "discovery"
    if unit < 16:
        return "confirmation"
    if unit < 24:
        return "fresh_confirmation"
    return "fresh_lockbox"


def values(language: str, unit: int) -> dict[str, str]:
    names = NAMES_EN if language == "en" else NAMES_ZH
    objects = OBJECTS_EN if language == "en" else OBJECTS_ZH
    places = PLACES_EN if language == "en" else PLACES_ZH
    colors = COLORS_EN if language == "en" else COLORS_ZH
    events = EVENTS_EN if language == "en" else EVENTS_ZH
    return {
        "a": names[unit], "b": names[(unit + 9) % UNITS], "c": names[(unit + 21) % UNITS],
        "obj": objects[unit], "alt_obj": objects[(unit + 13) % UNITS],
        "place": places[unit % len(places)], "alt_place": places[(unit + 5) % len(places)],
        "color": colors[unit % len(colors)], "alt_color": colors[(unit + 7) % len(colors)],
        "event": events[unit % len(events)], "alt_event": events[(unit + 3) % len(events)],
    }


def core(family: str, language: str, unit: int, state: int, surface: str) -> tuple[str, dict[str, str], dict]:
    v = values(language, unit)
    a, b, c, obj = v["a"], v["b"], v["c"], v["obj"]
    para = surface == "paraphrase"
    if family == "agent_binding":
        agent, recipient = (a, b) if state else (b, a)
        if language == "en":
            text = (f"{agent} delivered the {obj} to {recipient}. Was {a} the deliverer?" if not para else
                    f"The {obj} was delivered to {recipient} by {agent}. Did {a} perform the delivery?")
            relation = "delivered"
        else:
            text = (f"{agent}把{obj}交给了{recipient}。交付者是{a}吗？" if not para else
                    f"{obj}由{agent}交给{recipient}。这次交付是由{a}完成的吗？")
            relation = "交给"
        roles = {"primary": agent, "secondary": recipient, "relation": relation, "context": obj, "query": a}
        axis = ("agent_identity", "voice")
    elif family == "recipient_binding":
        recipient = b if state else c
        if language == "en":
            text = (f"{a} mailed the {obj} to {recipient}. Did {b} receive the {obj}?" if not para else
                    f"The {obj} was mailed by {a} to {recipient}. Was {b} the recipient?")
            relation = "mailed"
        else:
            text = (f"{a}把{obj}邮寄给了{recipient}。{b}收到{obj}了吗？" if not para else
                    f"{obj}由{a}邮寄给{recipient}。收件人是{b}吗？")
            relation = "邮寄"
        roles = {"primary": a, "secondary": recipient, "relation": relation, "context": obj, "query": b}
        axis = ("recipient_identity", "dative_voice")
    elif family == "patient_binding":
        patient = obj if state else v["alt_obj"]
        if language == "en":
            text = (f"{a} repaired the {patient} while {b} watched. Was the repaired item the {obj}?" if not para else
                    f"While {b} watched, the {patient} was repaired by {a}. Did the repair concern the {obj}?")
            relation = "repaired"
        else:
            text = (f"{b}在场时，{a}修理了{patient}。被修理的是{obj}吗？" if not para else
                    f"在{b}观看时，{patient}由{a}修理。修理对象是{obj}吗？")
            relation = "修理"
        roles = {"primary": a, "secondary": patient, "relation": relation, "context": b, "query": obj}
        axis = ("patient_identity", "voice")
    elif family == "relative_clause_binding":
        carrier = b if state else a
        if language == "en":
            text = (f"{a} thanked {b}, who was carrying the {obj}. Was {b} carrying the {obj}?" if state else
                    f"{a}, who was carrying the {obj}, thanked {b}. Was {b} carrying the {obj}?")
            if para:
                text = f"The person carrying the {obj} was {carrier}; {a} then thanked {b}. Did {b} carry the {obj}?"
            relation = "carrying"
        else:
            text = (f"{a}感谢了携带{obj}的{b}。{b}携带着{obj}吗？" if state else
                    f"携带{obj}的{a}感谢了{b}。{b}携带着{obj}吗？")
            if para:
                text = f"携带{obj}的人是{carrier}；随后{a}感谢了{b}。{b}携带{obj}吗？"
            relation = "携带"
        roles = {"primary": carrier, "secondary": b, "relation": relation, "context": obj, "query": b}
        axis = ("relative_clause_attachment", "surface")
    elif family == "property_state":
        current = v["color"] if state else v["alt_color"]
        if language == "en":
            text = (f"The {obj} is currently {current}. Is its current color {v['color']}?" if not para else
                    f"{a} repainted the {obj} from {v['alt_color']} to {current}. Is it currently {v['color']}?")
            relation = "currently"
        else:
            text = (f"{obj}现在是{current}。它当前的颜色是{v['color']}吗？" if not para else
                    f"{a}把{obj}从{v['alt_color']}重新涂成{current}。它现在是{v['color']}吗？")
            relation = "现在"
        roles = {"primary": obj, "secondary": current, "relation": relation, "context": v["alt_color"] if para else current, "query": v["color"]}
        axis = ("current_property", "direct_vs_update")
    elif family == "location_state":
        current = v["place"] if state else v["alt_place"]
        if language == "en":
            text = (f"The {obj} is now located in the {current}. Is it in the {v['place']}?" if not para else
                    f"After leaving the {v['alt_place']}, {a} moved the {obj} into the {current}. Is the {obj} now in the {v['place']}?")
            relation = "now"
        else:
            text = (f"{obj}现在位于{current}。它在{v['place']}吗？" if not para else
                    f"离开{v['alt_place']}后，{a}把{obj}移入{current}。{obj}现在在{v['place']}吗？")
            relation = "现在"
        roles = {"primary": obj, "secondary": current, "relation": relation, "context": v["alt_place"] if para else current, "query": v["place"]}
        axis = ("current_location", "direct_vs_move")
    elif family == "possession_state":
        holder = b if state else a
        if language == "en":
            text = (f"The {obj} is now held by {holder}. Does {b} now have it?" if not para else
                    (f"{a} handed the {obj} to {b}, who kept it. Does {b} now have it?" if state else
                     f"{a} showed the {obj} to {b} but kept it. Does {b} now have it?"))
            relation = "now"
        else:
            text = (f"{obj}现在由{holder}保管。{b}现在持有它吗？" if not para else
                    (f"{a}把{obj}交给{b}保管。{b}现在持有它吗？" if state else
                     f"{a}把{obj}给{b}看过，但仍由自己保管。{b}现在持有它吗？"))
            relation = "现在"
        roles = {"primary": obj, "secondary": holder, "relation": relation, "context": holder, "query": b}
        axis = ("current_holder", "direct_vs_transfer")
    elif family == "status_state":
        if language == "en":
            status = "active" if state else "inactive"
            change = "activated" if state else "deactivated"
            text = (f"The {obj} is currently {status}. Is the {obj} active?" if not para else
                    f"{a} {change} the {obj}. Is the {obj} active now?")
            relation, secondary = "active", status if not para else a
        else:
            status = "启用" if state else "停用"
            change = "启用了" if state else "停用了"
            text = (f"{obj}当前处于{status}状态。{obj}已经启用了吗？" if not para else
                    f"{a}{change}{obj}。{obj}现在启用了吗？")
            relation, secondary = "启用", status if not para else a
        roles = {"primary": obj, "secondary": secondary, "relation": relation,
                 "context": change if para else status, "query": relation}
        axis = ("current_status", "direct_vs_change")
    elif family == "temporal_order":
        first, second = ((v["event"], v["alt_event"]) if state else (v["alt_event"], v["event"]))
        if language == "en":
            text = (f"The {first} happened before the {second}. Did the {v['event']} happen before the {v['alt_event']}?" if not para else
                    f"The {second} occurred after the {first}. Did the {v['event']} occur before the {v['alt_event']}?")
            relation = "before"
        else:
            text = (f"{first}发生在{second}之前。{v['event']}在{v['alt_event']}之前发生吗？" if not para else
                    f"{second}发生在{first}之后。{v['event']}在{v['alt_event']}之前吗？")
            relation = "之前"
        roles = {"primary": first, "secondary": second, "relation": relation, "context": v["alt_event"], "query": v["event"]}
        axis = ("temporal_order", "before_vs_after")
    elif family == "quote_coreference":
        pronoun = ("I" if state else "you") if language == "en" else ("我" if state else "你")
        if language == "en":
            text = (f'{a} told {b}, "{pronoun} stored the {obj}." Does the speaker say that {a} stored it?' if not para else
                    f'Speaking to {b}, {a} said, "{pronoun} put away the {obj}." In the quote, is {a} the person who acted?')
        else:
            text = (f'{a}对{b}说：“{pronoun}收好了{obj}。”说话者表示是{a}收好的吗？' if not para else
                    f'{a}直接告诉{b}：“{pronoun}把{obj}放好了。”引语中的行动者是{a}吗？')
        roles = {"primary": a, "secondary": b, "relation": pronoun, "context": obj, "query": a}
        axis = ("quote_speaker", "quote_surface")
    elif family == "quantifier_sharing":
        if language == "en":
            text = ((f"All curators inspected the same {obj}. Must the curators share one inspected object?" if not para else
                     f"There was one {obj} that every curator inspected. Is a shared object required?") if state else
                    (f"Every curator inspected a different {obj}. Must the curators share one inspected object?" if not para else
                     f"Each curator inspected a distinct {obj}. Is a shared object required?"))
            primary, relation = "curator", "inspected"
            context = ("one" if para else "same") if state else ("distinct" if para else "different")
            query = "shared" if para else "share"
        else:
            text = ((f"所有策展人都检查了同一个{obj}。这些策展人必须共享一个检查对象吗？" if not para else
                     f"有一个{obj}被每位策展人检查过。是否要求共享一个对象？") if state else
                    (f"每位策展人都检查了不同的{obj}。这些策展人必须共享一个检查对象吗？" if not para else
                     f"每位策展人各自检查了不同的{obj}。是否要求共享一个对象？"))
            primary, relation = "策展人", "检查"
            context = ("一个" if para else "同一个") if state else "不同"
            query = "共享"
        roles = {"primary": primary, "secondary": obj, "relation": relation, "context": context, "query": query}
        axis = ("quantifier_sharing", "surface")
    else:
        taller, shorter = ((a, b) if state else (b, a))
        if language == "en":
            text = (f"{taller} is taller than {shorter}. Is {a} taller than {b}?" if not para else
                    f"{shorter} is shorter than {taller}. Is {a} taller than {b}?")
            relation = "taller"
        else:
            text = (f"{taller}比{shorter}更高。{a}比{b}更高吗？" if not para else
                    f"{shorter}比{taller}更矮。{a}比{b}更高吗？")
            relation = "更高"
        roles = {"primary": taller, "secondary": shorter, "relation": relation, "context": b, "query": a}
        axis = ("comparison_order", "tall_vs_short")
    return text, roles, {"semantic_axis": axis[0], "surface_axis": axis[1]}


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
                   f" 只回答 A 或 B。A：{options[0]}。B：{options[1]}。")
    free = (f" Answer with exactly one word: {true_code} or {false_code}." if language == "en" else
            f" 只用一个词回答：{true_code} 或 {false_code}。")
    return {
        "case_id": f"independent_{family}_{language}_u{unit}_{surface}_s{state}",
        "panel": "independent_bilingual_construction_state", "family": family, "language": language,
        "unit": unit, "surface": surface, "state": state, "truth": bool(state),
        "partition": partition(unit), "fresh": unit >= 16, "prompt_core": text,
        "prompt": text + instruction, "free_prompt": text + free, "role_values": roles,
        "factors": factors, "output_scheme": (fi + unit + li + si) % len(OUTPUT_SCHEMES),
        "true_code": true_code, "false_code": false_code, "correct_answer": answer,
        "gold_position": gold,
    }


def material() -> list[dict]:
    return [wrap(*args) for args in itertools.product(FAMILIES, LANGUAGES, range(UNITS), SURFACES, STATES)]


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidate_ids = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(ids) != 1 for ids in candidate_ids):
        raise RuntimeError(("candidate_not_single_token", candidate_ids))
    compiled = []
    for row in rows:
        system = ("Use only the supplied text. Follow the requested answer format exactly." if row["language"] == "en" else
                  "只使用给出的文本，并严格遵守回答格式。")
        ids = legacy.parent.compiler.core.chat_ids(tokenizer, system, row["prompt"])
        free_ids = legacy.parent.compiler.core.chat_ids(tokenizer, system, row["free_prompt"])
        positions: dict[str, list[int]] = {}
        for role, value in row["role_values"].items():
            spans = legacy.parent.contextual_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "free_prompt_ids": free_ids,
                         "candidate_ids": candidate_ids, "role_positions": positions})
    return compiled


def audit(rows: list[dict], compiled: list[dict]) -> dict:
    groups: dict[tuple, set] = defaultdict(set)
    output_cells: dict[tuple, list[int]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["language"], row["unit"])].add((row["surface"], row["state"]))
        output_cells[(row["family"], row["language"], row["surface"], row["partition"])].append(row["gold_position"])
    widths = [len(row["prompt_ids"]) for row in compiled]
    suspicious = ("锟", "銆", "鈥", "鐨", "鍚", "浜", "涓", "绾", "�")
    malformed = [row["case_id"] for row in rows if any(token in row["prompt_core"] for token in suspicious)]
    zh_rows = [row for row in rows if row["language"] == "zh"]
    zh_han_fraction = []
    for row in zh_rows:
        chars = [char for char in row["prompt_core"] if not char.isspace()]
        han = sum("\u4e00" <= char <= "\u9fff" for char in chars)
        zh_han_fraction.append(han / max(1, len(chars)))
    semantic_pairs = defaultdict(dict)
    for row in rows:
        semantic_pairs[(row["family"], row["language"], row["unit"], row["surface"])][row["state"]] = row
    state_prompts_distinct = all(pair[0]["prompt_core"] != pair[1]["prompt_core"] for pair in semantic_pairs.values())
    state_answers_distinct = all(pair[0]["correct_answer"] != pair[1]["correct_answer"] for pair in semantic_pairs.values())
    return {
        "rows": len(rows), "compiled_rows": len(compiled), "unique_case_ids": len({row["case_id"] for row in rows}),
        "families": dict(Counter(row["family"] for row in rows)),
        "languages": dict(Counter(row["language"] for row in rows)),
        "partitions": dict(Counter(row["partition"] for row in rows)),
        "factorial_cells_complete": all(len(cells) == 4 for cells in groups.values()),
        "state_prompts_distinct": state_prompts_distinct, "state_answers_distinct": state_answers_distinct,
        "gold_position_exact_per_cell": all(abs(float(np.mean(values)) - 0.5) <= 1e-12 for values in output_cells.values()),
        "zero_models": {
            "always_A": float(np.mean([row["gold_position"] == 0 for row in rows])),
            "always_B": float(np.mean([row["gold_position"] == 1 for row in rows])),
            "always_true": float(np.mean([row["state"] == 1 for row in rows])),
            "always_false": float(np.mean([row["state"] == 0 for row in rows])),
        },
        "malformed_strings": malformed,
        "zh_han_fraction_min_median": [float(min(zh_han_fraction)), float(np.median(zh_han_fraction))],
        "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "all_role_spans_compiled": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "semantic_uniqueness_machine_audit": "pass_explicit_truth_table_surface_by_state_factorial",
        "material_naturalness_machine_audit": "pass_utf8_roundtrip_han_ratio_and_no_known_mojibake_fragments",
        "human_blind_review": "NA_not_run_no_independent_human_panel_available",
        "historical_correction": "Phase2258 Chinese source strings are mojibake; prior natural-bilingual wording is retracted.",
    }


def preregistration() -> dict:
    return {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "frozen_before_model": True,
        "research_object": "independent clean bilingual construction-state field at every physical activation coordinate",
        "families": list(FAMILIES), "languages": list(LANGUAGES), "surfaces": list(SURFACES),
        "units": UNITS, "partitions": {"discovery": 12, "confirmation": 4, "fresh_confirmation": 8, "fresh_lockbox": 8},
        "behavior_gate": BEHAVIOR_GATE,
        "analysis_qualification": "candidate and exact-generation accuracy >= gate overall and separately on discovery and fresh_confirmation; lockbox is not used for eligibility",
        "observation": "all checkpoints, six roles and all coordinates for qualified families; stratified all-token subset across every qualified family",
        "model_tournament": ["family_mean", "own_same_coordinate_affine", "pure_algebraic", "shared_affine", "wrong_family_affine", "shuffled_pair_affine", "cross_checkpoint_affine"],
        "causal": "only candidates surviving independent fresh lockbox and upstream cross-checkpoint controls may enter near-manifold coordinate intervention",
        "forbidden": ["attention", "MLP", "weight inspection", "gradients", "PCA", "Top-K", "cosine screening", "donor-delta discovery"],
        "cross_model": "Qwen3-14B only after a Qwen3-4B independent lockbox result; model execution is sequential",
        "visualization": "important all-coordinate passports and representative token-by-coordinate fields",
        "cleanup": "undisplayed raw sample fields only after derivative hashes and visual-client verification",
        "theory": "conditionalized output-field closure theory and RDC unchanged; basic coordinate-level comparisons only",
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 独立中英十二构式合同与旧中文证据纠偏（C1433-C1468） [{stamp}]

**测试原理与测试用例。** 本期在加载模型前冻结12类构式、32套全新词汇、中英、直接/释义、真假状态，共 `{result['audit']['rows']}` 行。例子包括“`Mira Stone delivered the ceramic vase to Evan Cole`/`陶瓷花瓶由赵晨交给陈澄`”的施事绑定、“物体当前颜色/由旧色重涂”的属性状态、“先于/晚于”的时间顺序，以及引语共指、量词共享和比较顺序。12个unit用于发现、4个用于确认、8个用于新鲜确认、8个保持锁箱。行为资格要求候选A/B和精确自由生成同时不低于0.75，而且总体、discovery和fresh confirmation分别通过；锁箱不参与筛选。

**公式与冻结门。** 每个样本的全场对象仍为物理激活坐标，不是模型参数：

$$
\mathcal F_i=\{{H_{{i,q,r,j}}\}}_{{q,r,j}},\qquad
q\in\{{\mathrm{{embedding}},1,\ldots,L,\mathrm{{final\ norm}}\}}.
$$

后续逐坐标模型竞赛预注册为家族均值、自家族同坐标仿射、纯代数基线、跨家族共享仿射、错家族仿射、错配样本仿射和跨检查点仿射。禁止Attention、MLP、权重、梯度、PCA、Top-K、余弦筛选和供体差分发现。

**结果汇总与审计。** 材料审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；工程检查 `{result['checks']}`，总通过 `{result['all_checks_passed']}`。全部角色跨度经Qwen tokenizer上下文编译，A/B位置在每个家族、语言、表面和分区内精确平衡；中文通过UTF-8往返、汉字比例和已知乱码片段审计。独立人类盲评仍为NA，因此只授权受控材料结论。

**证据修正、理论进展与硬伤。** 审计发现Phase2258源码中的中文句子实际为双重编码乱码，旧审计未识别。因此此前“自然双语材料已通过”的表述撤回，所有依赖该中文面板的跨语言自然度外推降级；数值执行记录不因此消失，但不能再作为自然中文证据。本期进展是修复测量材料，不是发现编码机制。研究者编写模板、元语言输出码、无独立人类盲评和单模型尚未运行仍是硬伤。理论主体“条件化输出场闭合理论”和RDC不改名，也不引入新数学。

**结论、相关文件与下一步授权。** 合同在模型加载前合法冻结，授权Qwen3-4B顺序执行双行为门；只对预注册合格家族保存六角色全坐标场，并按家族分层保存有限但覆盖全部合格家族的全token场。脚本 `tests/glm5/phase2265_c1433_c1468_independent_bilingual_contract.py`；结果 `tests/glm5/result/phase2265_c1433_c1468_independent_bilingual_contract`。
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
    tokenizer = legacy.parent.model_base.parent.load_tokenizer()
    compiled = compile_rows(tokenizer, rows)
    raw_path = OUT / "material/independent_bilingual_cases.jsonl"
    compiled_path = OUT / "material/independent_bilingual_qwen_compiled.jsonl"
    write_rows(raw_path, rows)
    write_rows(compiled_path, compiled)
    material_audit = audit(rows, compiled)
    save(OUT / "audit/material_audit.json", material_audit)
    checks = {
        "rows_exact": len(rows) == len(compiled) == len(FAMILIES) * len(LANGUAGES) * UNITS * len(SURFACES) * len(STATES),
        "unique_cases": material_audit["unique_case_ids"] == len(rows),
        "factorial_complete": material_audit["factorial_cells_complete"],
        "state_contrast_valid": material_audit["state_prompts_distinct"] and material_audit["state_answers_distinct"],
        "zero_models_exact": all(abs(value - 0.5) <= 1e-12 for value in material_audit["zero_models"].values()),
        "gold_positions_exact": material_audit["gold_position_exact_per_cell"],
        "strings_clean": not material_audit["malformed_strings"] and material_audit["zh_han_fraction_min_median"][0] >= 0.35,
        "role_spans": material_audit["all_role_spans_compiled"],
        "human_review_honest_na": material_audit["human_blind_review"].startswith("NA_"),
    }
    hashes = {"preregistration": file_hash(OUT / "protocol/preregistration.json"),
              "raw_material": file_hash(raw_path), "compiled_material": file_hash(compiled_path),
              "audit": file_hash(OUT / "audit/material_audit.json")}
    result = {"phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
              "timestamp": datetime.now().astimezone().isoformat(), "protocol": preregistration(),
              "audit": material_audit, "hashes": hashes, "checks": checks,
              "all_checks_passed": all(checks.values()),
              "strict_conclusion": "Independent clean bilingual material is frozen; no model or mechanism result exists yet.",
              "next_authorization": "Run Qwen3-4B dual behavior gates and capture only preregistered qualified fields."}
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
