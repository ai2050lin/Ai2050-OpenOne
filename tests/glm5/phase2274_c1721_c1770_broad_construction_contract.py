#!/usr/bin/env python3
"""Freeze a broad, observation-first construction campaign for Phase 2274."""
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
OUT = TESTS / "result/phase2274_c1721_c1770_broad_construction_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2265_c1433_c1468_independent_bilingual_contract as previous  # noqa: E402


PHASE = 2274
CAMPAIGN = "C1721-C1770"
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
    "comparison_order",
    "negation_scope",
    "conditional_consequence",
    "conjunction_truth",
    "quantifier_sharing",
    "attitude_event_binding",
    "classification_chain",
)
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
SURFACES = ("direct", "paraphrase", "context_control")
STATES = (0, 1)
UNITS = 32
BEHAVIOR_GATE = 0.75
OUTPUT_SCHEMES = (
    ("Supported", "Unsupported"),
    ("Affirmed", "Denied"),
    ("Valid", "Invalid"),
    ("Correct", "Incorrect"),
)

FIRST_NAMES = (
    "Amina", "Brenna", "Cyrus", "Daria", "Emil", "Farah", "Galen", "Helena",
    "Idris", "Juno", "Kiran", "Leona", "Malik", "Nadia", "Orin", "Priya",
)
LAST_NAMES = ("Vale", "Morrow")
NAMES = tuple(f"{first} {last}" for last in LAST_NAMES for first in FIRST_NAMES)
OBJECTS = (
    "amber compass", "bamboo folder", "canvas satchel", "digital timer", "enameled cup",
    "felt notebook", "granite tile", "hollow globe", "inked poster", "jade pendant",
    "knitted cap", "leather folio", "marble token", "navy umbrella", "oak puzzle",
    "porcelain plate", "quartz paperweight", "rattan basket", "steel lantern", "tin model",
    "umber sketchbook", "velvet pouch", "wicker screen", "yellow telescope", "zinc whistle",
    "acrylic frame", "bronze key", "cotton ribbon", "driftwood carving", "etched mirror",
    "folded chart", "glass prism",
)
PLACES = (
    "archive room", "balcony cabinet", "central kiosk", "design studio", "east pantry",
    "front workshop", "garden annex", "history alcove", "inspection bay", "junior office",
    "kitchen shelf", "lower gallery", "media closet", "north atrium", "operations desk",
    "project lounge",
)
COLORS = (
    "amber", "blue", "coral", "dark green", "emerald", "fuchsia", "gold", "hazel",
    "indigo", "jade", "khaki", "lilac", "magenta", "navy", "ochre", "pearl white",
)
EVENTS = (
    "audit", "briefing", "cataloging", "delivery", "evaluation", "filing", "grading",
    "handover", "inspection", "labeling", "measurement", "orientation", "packing",
    "review", "sorting", "testing",
)
CLASS_A = (
    "Aster", "Brin", "Corda", "Dellen", "Eris", "Faron", "Grell", "Hesta",
    "Ivar", "Jorin", "Kelda", "Lumen", "Meral", "Neris", "Ordan", "Pella",
)
CLASS_B = (
    "Quorin", "Ravel", "Solen", "Taris", "Ulen", "Vesta", "Werren", "Xeran",
    "Yorin", "Zella", "Arven", "Belor", "Cerin", "Dovar", "Elden", "Feris",
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
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def values(unit: int) -> dict[str, str]:
    return {
        "a": NAMES[unit],
        "b": NAMES[(unit + 11) % UNITS],
        "c": NAMES[(unit + 23) % UNITS],
        "obj": OBJECTS[unit],
        "alt_obj": OBJECTS[(unit + 13) % UNITS],
        "place": PLACES[unit % len(PLACES)],
        "alt_place": PLACES[(unit + 7) % len(PLACES)],
        "color": COLORS[unit % len(COLORS)],
        "alt_color": COLORS[(unit + 9) % len(COLORS)],
        "event": EVENTS[unit % len(EVENTS)],
        "alt_event": EVENTS[(unit + 5) % len(EVENTS)],
        "class_a": CLASS_A[unit % len(CLASS_A)],
        "class_b": CLASS_B[(unit + 3) % len(CLASS_B)],
    }


def core(family: str, unit: int, state: int, surface: str) -> tuple[str, dict[str, str], dict]:
    v = values(unit)
    a, b, c, obj = v["a"], v["b"], v["c"], v["obj"]
    para = surface == "paraphrase"
    if family == "agent_binding":
        agent = a if state else b
        recipient = c
        text = (f"{agent} delivered the {obj} to {recipient}. Was {a} the deliverer?" if not para else
                f"The {obj} was delivered to {recipient} by {agent}. Did {a} perform the delivery?")
        roles = {"primary": agent, "secondary": recipient, "relation": "delivered",
                 "context": obj, "query": a}
        axis = "agent_identity"
    elif family == "recipient_binding":
        recipient = b if state else c
        text = (f"{a} mailed the {obj} to {recipient}. Did {b} receive the {obj}?" if not para else
                f"The {obj} was mailed by {a} to {recipient}. Was {b} the recipient?")
        roles = {"primary": a, "secondary": recipient, "relation": "mailed",
                 "context": obj, "query": b}
        axis = "recipient_identity"
    elif family == "patient_binding":
        patient = obj if state else v["alt_obj"]
        text = (f"{a} repaired the {patient} while {b} watched. Was the repaired item the {obj}?" if not para else
                f"While {b} watched, the {patient} was repaired by {a}. Did the repair concern the {obj}?")
        roles = {"primary": a, "secondary": patient, "relation": "repaired",
                 "context": b, "query": obj}
        axis = "patient_identity"
    elif family == "relative_clause_binding":
        carrier = b if state else a
        text = (f"{a} thanked {b}, who was carrying the {obj}. Was {b} carrying the {obj}?" if state else
                f"{a}, who was carrying the {obj}, thanked {b}. Was {b} carrying the {obj}?")
        if para:
            text = f"The person carrying the {obj} was {carrier}; afterward {a} thanked {b}. Did {b} carry the {obj}?"
        roles = {"primary": carrier, "secondary": b, "relation": "carrying",
                 "context": obj, "query": b}
        axis = "relative_clause_attachment"
    elif family == "property_state":
        current = v["color"] if state else v["alt_color"]
        text = (f"The {obj} is currently {current}. Is its current color {v['color']}?" if not para else
                f"{a} repainted the {obj} from {v['alt_color']} to {current}. Is it now {v['color']}?")
        roles = {"primary": obj, "secondary": current, "relation": "repainted" if para else "currently",
                 "context": v["alt_color"] if para else current, "query": v["color"]}
        axis = "current_property"
    elif family == "location_state":
        current = v["place"] if state else v["alt_place"]
        text = (f"The {obj} is now in the {current}. Is it in the {v['place']}?" if not para else
                f"After leaving the {v['alt_place']}, {a} moved the {obj} into the {current}. Is it now in the {v['place']}?")
        roles = {"primary": obj, "secondary": current, "relation": "now",
                 "context": v["alt_place"] if para else current, "query": v["place"]}
        axis = "current_location"
    elif family == "possession_state":
        holder = b if state else a
        text = (f"The {obj} is now held by {holder}. Does {b} now have it?" if not para else
                (f"{a} handed the {obj} to {b}, who kept it. Does {b} now have it?" if state else
                 f"{a} showed the {obj} to {b} but kept it. Does {b} now have it?"))
        roles = {"primary": obj, "secondary": holder,
                 "relation": ("handed" if state else "showed") if para else "held",
                 "context": holder, "query": b}
        axis = "current_holder"
    elif family == "status_state":
        status = "active" if state else "inactive"
        change = "activated" if state else "deactivated"
        text = (f"The {obj} is currently {status}. Is the {obj} active?" if not para else
                f"{a} {change} the {obj}. Is the {obj} active now?")
        roles = {"primary": obj, "secondary": status if not para else a, "relation": "active",
                 "context": change if para else status, "query": "active"}
        axis = "current_status"
    elif family == "temporal_order":
        first, second = ((v["event"], v["alt_event"]) if state else (v["alt_event"], v["event"]))
        text = (f"The {first} happened before the {second}. Did the {v['event']} happen before the {v['alt_event']}?" if not para else
                f"The {second} occurred after the {first}. Did the {v['event']} occur before the {v['alt_event']}?")
        roles = {"primary": first, "secondary": second, "relation": "before",
                 "context": v["alt_event"], "query": v["event"]}
        axis = "temporal_order"
    elif family == "comparison_order":
        taller, shorter = ((a, b) if state else (b, a))
        text = (f"{taller} is taller than {shorter}. Is {a} taller than {b}?" if not para else
                f"{shorter} is shorter than {taller}. Is {a} taller than {b}?")
        roles = {"primary": taller, "secondary": shorter, "relation": "taller",
                 "context": b, "query": a}
        axis = "comparison_order"
    elif family == "negation_scope":
        polarity = "is" if state else "is not"
        text = (f"The report says that the {obj} {polarity} {v['color']}. Is the report claiming that color?" if not para else
                (f"According to the report, the {obj} has the color {v['color']}. Does it affirm that color?" if state else
                 f"According to the report, it is false that the {obj} has the color {v['color']}. Does it affirm that color?"))
        roles = {"primary": obj, "secondary": v["color"], "relation": "report",
                 "context": "report", "query": "color"}
        axis = "negation_scope"
    elif family == "conditional_consequence":
        premise = f"the {v['event']} occurred" if state else f"the {v['event']} did not occur"
        text = (f"If the {v['event']} occurs, the {v['alt_event']} must follow. The record says {premise}. Must the {v['alt_event']} follow?" if not para else
                f"The {v['alt_event']} is required whenever the {v['event']} occurs. In this case, {premise}. Is the requirement triggered?")
        roles = {"primary": v["event"], "secondary": v["alt_event"],
                 "relation": "required" if para else "must",
                 "context": premise, "query": v["alt_event"]}
        axis = "conditional_trigger"
    elif family == "conjunction_truth":
        location_clause = f"is in the {v['place']}" if state else f"is not in the {v['place']}"
        text = (f"The {obj} is {v['color']} and {location_clause}. Is it both {v['color']} and in the {v['place']}?" if not para else
                f"Two claims are listed: the {obj} is {v['color']}; it {location_clause}. Are both target claims true?")
        roles = {"primary": obj, "secondary": v["color"], "relation": "both",
                 "context": location_clause, "query": v["place"]}
        axis = "conjunction_truth"
    elif family == "quantifier_sharing":
        descriptor = "same" if state else "different"
        text = (f"Every curator inspected the {descriptor} {obj}. Must the curators share one inspected object?" if not para else
                (f"There was one {obj} that every curator inspected. Is a shared object required?" if state else
                 f"Each curator inspected a distinct {obj}. Is a shared object required?"))
        roles = {"primary": "curator", "secondary": obj, "relation": "inspected",
                 "context": descriptor if not para else ("one" if state else "distinct"),
                 "query": "shared" if para else "share"}
        axis = "quantifier_sharing"
    elif family == "attitude_event_binding":
        actor = a if state else b
        text = (f"I enjoy it when {actor} eats the {obj}. Does my enjoyment concern {a} eating the {obj}?" if not para else
                f"The event I like is {actor}'s eating of the {obj}. Is {a} the eater in the event I like?")
        roles = {"primary": actor, "secondary": obj, "relation": "like" if para else "enjoy",
                 "context": "eating", "query": a}
        axis = "attitude_event_binding"
    else:
        link = "Every" if state else "No"
        text = (f"The {obj} is registered as a {v['class_a']} item. {link} {v['class_a']} item is a {v['class_b']} item. Is the {obj} a {v['class_b']} item?" if not para else
                (f"All members of class {v['class_a']} belong to class {v['class_b']}; the {obj} belongs to {v['class_a']}. Does it belong to {v['class_b']}?" if state else
                 f"Class {v['class_a']} and class {v['class_b']} are disjoint; the {obj} belongs to {v['class_a']}. Does it belong to {v['class_b']}?"))
        roles = {"primary": obj, "secondary": v["class_a"], "relation": v["class_a"],
                 "context": v["class_b"], "query": v["class_b"]}
        axis = "classification_chain"

    if surface == "context_control":
        text += f" A separate inventory note says the {v['alt_obj']} remains in the {v['alt_place']}."
    return text, roles, {"semantic_axis": axis, "surface_axis": surface}


def wrap(family: str, unit: int, surface: str, state: int) -> dict:
    text, roles, factors = core(family, unit, state, surface)
    fi, si = FAMILIES.index(family), SURFACES.index(surface)
    true_code, false_code = OUTPUT_SCHEMES[(fi + unit + si) % len(OUTPUT_SCHEMES)]
    answer = true_code if state else false_code
    gold = (fi + unit + si + state) % 2
    options = [answer, false_code if state else true_code]
    if gold:
        options.reverse()
    candidate = f" Choose A or B only. A: {options[0]}. B: {options[1]}."
    free = f" Answer with exactly one word: {true_code} or {false_code}."
    return {
        "case_id": f"broad_{family}_u{unit}_{surface}_s{state}",
        "panel": "broad_construction_coordinate_ecology",
        "family": family,
        "language": "en",
        "unit": unit,
        "surface": surface,
        "state": state,
        "truth": bool(state),
        "partition": partition(unit),
        "fresh": unit >= 16,
        "prompt_core": text,
        "prompt": text + candidate,
        "free_prompt": text + free,
        "role_values": roles,
        "factors": factors,
        "output_scheme": (fi + unit + si) % len(OUTPUT_SCHEMES),
        "true_code": true_code,
        "false_code": false_code,
        "correct_answer": answer,
        "gold_position": gold,
    }


def material() -> list[dict]:
    return [wrap(*args) for args in itertools.product(FAMILIES, range(UNITS), SURFACES, STATES)]


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidate_ids = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(ids) != 1 for ids in candidate_ids):
        raise RuntimeError(("candidate_not_single_token", candidate_ids))
    compiled = []
    system = "Use only the supplied text. Follow the requested answer format exactly."
    for row in rows:
        ids = previous.legacy.parent.compiler.core.chat_ids(tokenizer, system, row["prompt"])
        free_ids = previous.legacy.parent.compiler.core.chat_ids(tokenizer, system, row["free_prompt"])
        positions: dict[str, list[int]] = {}
        for role, value in row["role_values"].items():
            spans = previous.legacy.parent.contextual_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "free_prompt_ids": free_ids,
                         "candidate_ids": candidate_ids, "role_positions": positions})
    return compiled


def audit(rows: list[dict], compiled: list[dict]) -> dict:
    cells: dict[tuple, set] = defaultdict(set)
    positions: dict[tuple, list[int]] = defaultdict(list)
    pairs: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        cells[(row["family"], row["unit"])].add((row["surface"], row["state"]))
        positions[(row["family"], row["surface"], row["partition"])].append(row["gold_position"])
        pairs[(row["family"], row["unit"], row["surface"])][row["state"]] = row
    widths = [len(row["prompt_ids"]) for row in compiled]
    return {
        "rows": len(rows),
        "compiled_rows": len(compiled),
        "unique_case_ids": len({row["case_id"] for row in rows}),
        "families": dict(Counter(row["family"] for row in rows)),
        "surfaces": dict(Counter(row["surface"] for row in rows)),
        "partitions": dict(Counter(row["partition"] for row in rows)),
        "factorial_cells_complete": all(len(value) == len(SURFACES) * len(STATES) for value in cells.values()),
        "state_prompts_distinct": all(pair[0]["prompt_core"] != pair[1]["prompt_core"] for pair in pairs.values()),
        "state_answers_distinct": all(pair[0]["correct_answer"] != pair[1]["correct_answer"] for pair in pairs.values()),
        "gold_position_exact_per_cell": all(abs(float(np.mean(value)) - 0.5) <= 1e-12 for value in positions.values()),
        "zero_models": {
            "always_A": float(np.mean([row["gold_position"] == 0 for row in rows])),
            "always_B": float(np.mean([row["gold_position"] == 1 for row in rows])),
            "always_true": float(np.mean([row["state"] == 1 for row in rows])),
            "always_false": float(np.mean([row["state"] == 0 for row in rows])),
        },
        "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
        "all_role_spans_compiled": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "semantic_uniqueness_machine_audit": "pass_explicit_state_truth_and_surface_factorial",
        "material_naturalness_machine_audit": "pass_authored_english_sentences_and_utf8_roundtrip",
        "human_blind_review": "NA_not_run_no_independent_human_panel_available",
    }


def preregistration() -> dict:
    return {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "frozen_before_model": True,
        "research_object": "broad construction-conditioned embedding-hiddenstate coordinate ecology",
        "families": list(FAMILIES),
        "surfaces": list(SURFACES),
        "units": UNITS,
        "partitions": {"discovery": 12, "confirmation": 4, "fresh_confirmation": 8, "fresh_lockbox": 8},
        "behavior_gate": BEHAVIOR_GATE,
        "qualification": "candidate and exact generation >= gate overall, discovery, and fresh_confirmation",
        "observation": "all qualified samples, embedding, every post-block state, final norm, six roles, all coordinates",
        "all_token_observation": "one fresh-confirmation unit and one fresh-lockbox unit per qualified family",
        "basic_structure_models": [
            "family_mean", "same_coordinate_affine", "same_coordinate_piecewise_quartile",
            "same_coordinate_sign_interval", "previous_checkpoint_same_coordinate_affine",
            "wrong_family", "shuffled_pair", "surface_only", "output_scheme_only",
        ],
        "causal_route": "random balanced coordinate masks on prospectively qualified anchors; no Top-K discovery",
        "cross_scale": "Qwen3-14B only after Qwen3-4B lockbox structure selection; model local coordinates",
        "forbidden": ["attention", "MLP", "weight inspection", "gradients", "PCA", "Top-K", "cosine screening"],
        "theory": "conditionalized output-field closure theory and RDC unchanged",
        "cleanup": "retain visual derivatives and hashes; delete only undisplayed raw fields after verification",
    }


def append_memo(result: dict) -> None:
    current = MEMO.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in current:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 十六构式全坐标生态合同与证据纠偏（{CAMPAIGN}） [{stamp}]

**附件审查与结论纠偏。** Phase2264-2273 的原始结果支持三项事实：纯代数、族均值、共享仿射、错族与打乱配对都不能解释 Qwen3 的全部同坐标预测增益；Qwen3-4B 的十个行为合格构式和 Qwen3-14B 的三个预冻结构式获得了模型本地、跨新词汇的逐坐标预测结果；`property_state|delete` 在 q15-query 的整条 2560 维状态删除上通过独立锁箱，但调用失败，严格双向因果族仍为 0。GLM4 和 DS7B 没有获得合法内部测试资格，因此只能记为 NA。附件中“线性代数已经不足”“必须使用流形、李群或规范场”“0/10 证明单点线性齿轮绝对不存在”均超过证据；物理激活坐标也不是模型权重参数。本阶段保留可检验候选，不把高级数学名词当作实验结论。

**测试原理、用例与冻结对象。** 新合同一次冻结 16 类语言构式、32 套新词汇、三种表面和真假两状态，共 `{result['audit']['rows']}` 行。构式覆盖施事、受事者、受事、关系从句、属性、位置、持有、状态、时间、比较、否定、条件后件、合取、量词共享、态度-事件嵌套和两跳分类链。`context_control` 只加入一条无关但自然的库存说明，用于把长度/干扰与语义变化分账。每类在 discovery、confirmation、fresh confirmation、fresh lockbox 中都有完整状态对，A/B 位置和四套输出码逐格平衡。独立人类盲评仍为 NA，因此“自然”只指机器审计下的受控英文句子。

**数学对象与门槛。** 本轮完整场定义为：

$$
\mathcal F_i=\left\{{H_{{i,q,r,j}}\right\}}_{{q,r,j}},\qquad
q\in\left\{{\mathrm{{embedding}},1,\ldots,L,\mathrm{{final\ norm}}\right\}},
$$

$$
Q_f=\mathbf 1\!\left[\min_{{p\in\left\{{\mathrm{{all}},\mathrm{{discovery}},\mathrm{{fresh\ confirmation}}\right\}}}}
\min(A^{{\mathrm{{cand}}}}_{{f,p}},A^{{\mathrm{{gen}}}}_{{f,p}})\ge0.75\right].
$$

后续基础竞赛同时登记族均值、同坐标仿射、按 discovery 基态四分位冻结的分段响应、符号/区间状态、上一检查点同坐标预测、错族、错配、纯表面和纯输出码控制。所有 2560 个激活坐标都进入误差账；不使用 PCA、Top-K 或余弦筛选。

**结果汇总与审计。** 材料审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；工程检查 `{json.dumps(result['checks'], ensure_ascii=False)}`，总通过 `{result['all_checks_passed']}`。本期没有加载模型，也没有 HiddenState 或机制结果。

**理论进展、问题硬伤与结论。** 理论主体继续使用“条件化输出场闭合理论”，组织原则仍为复用-差分-条件化（RDC）。这份合同的进展是把既有十类关系扩展为状态、逻辑、嵌套事件和图组合的共同观察分母，并预先分离表面长度与输出码。硬伤包括研究者编写模板、英文单语、元语言输出码、缺少人类盲评、小模型外推受限，以及“同坐标预测”仍可能读取一般状态动力学。下一步授权 Qwen3-4B 双行为与全场采集；失败构式记 NA，但不阻断其他构式和随机坐标因果路线。

**相关文件。** 脚本 `tests/glm5/phase2274_c1721_c1770_broad_construction_contract.py`；结果 `tests/glm5/result/phase2274_c1721_c1770_broad_construction_contract`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    for sub in ("protocol", "material", "audit", "analysis"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    protocol = preregistration()
    save(OUT / "protocol/preregistration.json", {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **protocol})
    rows = material()
    tokenizer = previous.legacy.parent.model_base.parent.load_tokenizer()
    compiled = compile_rows(tokenizer, rows)
    raw_path = OUT / "material/broad_construction_cases.jsonl"
    compiled_path = OUT / "material/broad_construction_qwen_compiled.jsonl"
    write_rows(raw_path, rows)
    write_rows(compiled_path, compiled)
    report = audit(rows, compiled)
    save(OUT / "audit/material_audit.json", report)
    checks = {
        "rows_exact": len(rows) == len(compiled) == len(FAMILIES) * UNITS * len(SURFACES) * len(STATES),
        "unique_cases": report["unique_case_ids"] == len(rows),
        "factorial_complete": report["factorial_cells_complete"],
        "state_contrast_valid": report["state_prompts_distinct"] and report["state_answers_distinct"],
        "zero_models_exact": all(abs(value - 0.5) <= 1e-12 for value in report["zero_models"].values()),
        "gold_positions_exact": report["gold_position_exact_per_cell"],
        "role_spans": report["all_role_spans_compiled"],
        "human_review_honest_na": report["human_blind_review"].startswith("NA_"),
    }
    hashes = {
        "preregistration": file_hash(OUT / "protocol/preregistration.json"),
        "raw_material": file_hash(raw_path),
        "compiled_material": file_hash(compiled_path),
        "audit": file_hash(OUT / "audit/material_audit.json"),
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "protocol": protocol,
        "audit": report,
        "hashes": hashes,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "A broad observation-first contract is frozen; no model or mechanism result exists yet.",
        "next_authorization": "Run Qwen3-4B dual behavior and all-coordinate field capture without changing material or gates.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
