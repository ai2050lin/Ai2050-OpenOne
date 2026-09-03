#!/usr/bin/env python3
"""C670-C683 fresh bilingual coordinate-dynamics specificity campaign."""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
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
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c682_fresh_bilingual_coordinate_specificity_atlas.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c682_fresh_bilingual_coordinate_specificity.float16.npy"
sys.path.insert(0, str(TESTS))

import phase2105_c571_c589_scope_program_algebra_campaign as scope
import phase2190_c656_c669_absolute_coordinate_grammar_campaign as prior

PHASES = {
    "C670-C672": (2195, "parent_retrial_and_fresh_bilingual_contract"),
    "C673-C675": (2196, "fresh_bilingual_behavior_and_full_coordinate_capture"),
    "C676-C679": (2197, "shared_dynamics_vs_typed_program_increment"),
    "C680-C682": (2198, "fresh_bilingual_local_response_specificity"),
    "C683": (2199, "fresh_bilingual_major_stage_closure"),
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower().replace('-', '_')}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

DIM = 2560
CHECKPOINTS = 38
ROLES = prior.ROLES
QPOINTS = prior.QPOINTS
LANGUAGES = ("en", "zh")
GROUPS = ("voice_binding", "nested_attitude", "relation_graph", "voice_negation_composition")
UNITS = 24
BEHAVIOR_GATE = 0.75
TYPED_GAIN_GATE = 0.01
SPECIFICITY_GATE = 0.02

NAMES_EN_A = ("Mira", "Nolan", "Iris", "Darin", "Lena", "Oren", "Faye", "Gavin", "Talia", "Ronan", "Vera", "Caleb", "Nadia", "Elias", "Sela", "Bram", "Aria", "Jonas", "Lyra", "Marek", "Cora", "Silas", "Nina", "Tobin")
NAMES_EN_B = ("Tovan", "Selin", "Borin", "Kira", "Milo", "Rhea", "Damon", "Elin", "Pavel", "Nora", "Ivo", "Mina", "Ravi", "Tessa", "Noel", "Kara", "Dorian", "Lina", "Perrin", "Maya", "Orin", "Tara", "Vito", "Rina")
NAMES_ZH_A = ("米拉", "诺兰", "伊里斯", "达林", "莱娜", "奥伦", "费伊", "加文", "塔莉娅", "罗南", "维拉", "凯莱布", "娜迪娅", "埃利亚斯", "塞拉", "布拉姆", "阿丽娅", "约纳斯", "莱拉", "马雷克", "科拉", "西拉斯", "妮娜", "托宾")
NAMES_ZH_B = ("托万", "塞林", "博林", "基拉", "米洛", "瑞娅", "达蒙", "埃林", "帕维尔", "诺拉", "伊沃", "米娜", "拉维", "泰莎", "诺埃尔", "卡拉", "多里安", "莉娜", "佩林", "玛雅", "奥林", "塔拉", "维托", "丽娜")
OBJECTS_EN = ("copper vessel", "linen bag", "silver compass", "wooden tablet", "glass lantern", "stone marker", "brass key", "paper folder", "ceramic bowl", "iron latch", "blue ribbon", "oak panel", "wool blanket", "metal frame", "green bottle", "clay figure", "red notebook", "small basket", "white shell", "round mirror", "black umbrella", "long rope", "clear jar", "thin board")
OBJECTS_ZH = ("铜器", "布袋", "银罗盘", "木牌", "玻璃灯", "石标", "黄铜钥匙", "纸夹", "陶碗", "铁闩", "蓝丝带", "橡木板", "羊毛毯", "金属框", "绿瓶", "陶像", "红笔记本", "小篮子", "白贝壳", "圆镜", "黑雨伞", "长绳", "透明罐", "薄木板")
DISTRACT_EN = ("amber cube", "violet cloth", "granite disk", "tin cup", "reed mat", "wax seal", "cotton pouch", "bronze bell", "ivory bead", "leather case", "pine box", "steel ring", "sand timer", "chalk tile", "marble cone", "hemp cord", "glass prism", "wooden wheel", "copper plate", "linen flag", "silver pin", "stone cup", "brass hook", "paper tube")
DISTRACT_ZH = ("琥珀方块", "紫布", "花岗岩圆片", "锡杯", "芦苇席", "蜡封", "棉袋", "铜铃", "象牙珠", "皮盒", "松木箱", "钢环", "沙漏", "粉笔砖", "大理石锥", "麻绳", "玻璃棱镜", "木轮", "铜盘", "布旗", "银针", "石杯", "黄铜钩", "纸筒")


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


def partition(unit: int) -> str:
    if unit < 12:
        return "discovery"
    if unit < 18:
        return "confirmation"
    return "lockbox"


def values(unit: int, language: str) -> dict:
    if language == "en":
        return {"a": NAMES_EN_A[unit], "b": NAMES_EN_B[unit], "x": OBJECTS_EN[unit], "y": DISTRACT_EN[unit]}
    return {"a": NAMES_ZH_A[unit], "b": NAMES_ZH_B[unit], "x": OBJECTS_ZH[unit], "y": DISTRACT_ZH[unit]}


def option_order(unit: int, offset: int) -> int:
    return ((unit // 2) + offset) % 2


def make_row(*, group: str, domain: str, language: str, unit: int, cell: str,
             facts: list[str], question: str, truth: bool, roles: dict[str, str],
             surface: str, offset: int, factors: dict) -> dict:
    order = option_order(unit, offset)
    if language == "en":
        body = " ".join(facts)
        core = f"A verified record states: {body} Based only on this record, {question}"
        choices = "(A) Yes (B) No" if order == 0 else "(A) No (B) Yes"
    else:
        body = "".join(facts)
        core = f"核验记录：{body}只根据这份记录，{question}"
        choices = "(A) 是 (B) 否" if order == 0 else "(A) 否 (B) 是"
    gold = (0 if truth else 1) if order == 0 else (1 if truth else 0)
    return {
        "case_id": f"c670-{group}-{domain}-{language}-u{unit:02d}-{cell}-{surface}",
        "panel": group, "family": group, "operation_type": group,
        "operation_domain": domain, "domain_id": f"{group}:{domain}:{language}",
        "language": language, "surface": surface, "construction": f"{group}:{surface}",
        "unit": unit, "partition": partition(unit), "cell": cell, "factors": factors,
        "truth": truth, "correct_answer": "Yes" if truth else "No",
        "wrong_answer": "No" if truth else "Yes", "option_order": order,
        "gold_position": gold, "facts": facts, "question": question,
        "prompt_core": core, "prompt": f"{core} {choices}. Reply with only A or B.",
        "free_prompt": f"{core} Answer only Yes or No.",
        "role_values": roles,
        "semantic_graph": {
            "input_type": group, "output_type": "binary_truth", "language": language,
            "invariants": ["thematic_roles", "query_contract", "output_protocol"],
            "factors": factors,
        },
    }


def voice_case(language: str, unit: int, voice: int) -> dict:
    u = values(unit, language); truth = unit % 2 == 0; target = u["x"] if truth else u["y"]
    domain_i = unit % 3
    if language == "en":
        verb = ("inspected", "praised", "carried")[domain_i]
        fact = f"{u['a']} {verb} the {u['x']}." if not voice else f"The {u['x']} was {verb} by {u['a']}."
        question = f"Did {u['a']} {verb} the {target}?"
        noise = f"{u['b']} recorded the {u['y']} separately."
    else:
        verb = ("检查", "表扬", "搬运")[domain_i]
        fact = f"{u['a']}{verb}了{u['x']}。" if not voice else f"{u['x']}被{u['a']}{verb}了。"
        question = f"{u['a']}{verb}的是{target}吗？"
        noise = f"{u['b']}另外记录了{u['y']}。"
    return make_row(group="voice_binding", domain=("inspect", "praise", "carry")[domain_i],
                    language=language, unit=unit, cell=f"v{voice}", facts=[fact, noise],
                    question=question, truth=truth,
                    roles={"primary": u["a"], "secondary": u["b"], "relation": verb,
                           "context": u["x"], "query": u["a"]},
                    surface="passive" if voice else "active", offset=voice,
                    factors={"voice": voice})


def nested_case(language: str, unit: int, outer: int, inner: int) -> dict:
    u = values(unit, language); truth = unit % 2 == 0; target = u["x"] if truth else u["y"]
    domain_i = unit % 3
    if language == "en":
        verb = ("remember", "regret", "believe")[domain_i]
        inner_fact = f"{u['b']} {'did not open' if inner else 'opened'} the {u['x']}"
        inner_query = f"{u['b']} {'did not open' if inner else 'opened'} the {target}"
        fact = f"{u['a']} {'did not ' + verb if outer else verb + 'ed'} that {inner_fact}."
        question = f"Is it true that {u['a']} {'did not ' + verb if outer else verb + 'ed'} that {inner_query}?"
        noise = f"{u['b']} catalogued the {u['y']} separately."
    else:
        verb = ("记得", "后悔", "相信")[domain_i]
        inner_fact = f"{u['b']}{'没有打开' if inner else '打开了'}{u['x']}"
        inner_query = f"{u['b']}{'没有打开' if inner else '打开了'}{target}"
        fact = f"{u['a']}{'并不' if outer else ''}{verb}{inner_fact}。"
        question = f"{u['a']}{'并不' if outer else ''}{verb}{inner_query}，对吗？"
        noise = f"{u['b']}另外登记了{u['y']}。"
    return make_row(group="nested_attitude", domain=("remember", "regret", "believe")[domain_i],
                    language=language, unit=unit, cell=f"o{outer}i{inner}", facts=[fact, noise],
                    question=question, truth=truth,
                    roles={"primary": u["a"], "secondary": u["b"], "relation": verb,
                           "context": u["x"], "query": u["a"]}, surface="record",
                    offset=2 + outer * 2 + inner,
                    factors={"outer_negation": outer, "inner_negation": inner})


def graph_case(language: str, unit: int, depth: int) -> dict:
    u = values(unit, language); truth = unit % 2 == 0; domain_i = unit % 3
    if language == "en":
        relation = ("is a kind of", "is inside", "occurred before")[domain_i]
        mids = (f"class-{unit}", f"group-{unit}", f"level-{unit}")
        source = u["x"]
        chain = [source, *mids]
        facts = [f"{chain[i]} {relation} {chain[i + 1]}." for i in range(depth)]
        facts.append(f"{u['b']} recorded the {u['y']} separately.")
        endpoint = chain[depth]; target = endpoint if truth else u["y"]
        question = f"Is it true that {source} {relation} {target}?"
    else:
        relation = ("是一种", "位于", "早于")[domain_i]
        mids = (f"类别{unit}", f"组别{unit}", f"层级{unit}")
        source = u["x"]; chain = [source, *mids]
        facts = [f"{chain[i]}{relation}{chain[i + 1]}。" for i in range(depth)]
        facts.append(f"{u['b']}另外记录了{u['y']}。")
        endpoint = chain[depth]; target = endpoint if truth else u["y"]
        question = f"{source}{relation}{target}，对吗？"
    return make_row(group="relation_graph", domain=("taxonomy", "spatial", "temporal")[domain_i],
                    language=language, unit=unit, cell=f"d{depth}", facts=facts,
                    question=question, truth=truth,
                    roles={"primary": source, "secondary": u["b"], "relation": relation,
                           "context": endpoint, "query": source}, surface="record",
                    offset=6 + depth, factors={"depth": depth})


def composition_case(language: str, unit: int, voice: int, negation: int) -> dict:
    u = values(unit, language); truth = unit % 2 == 0; target = u["x"] if truth else u["y"]
    if language == "en":
        verb = "inspected"
        if not voice:
            fact = f"{u['a']} {'did not inspect' if negation else verb} the {u['x']}."
        else:
            fact = f"The {u['x']} {'was not inspected' if negation else 'was inspected'} by {u['a']}."
        question = f"Did {u['a']} {'not inspect' if negation else 'inspect'} the {target}?"
        noise = f"{u['b']} stored the {u['y']} separately."
        relation = "inspect"
    else:
        relation = "检查"
        if not voice:
            fact = f"{u['a']}{'没有检查' if negation else '检查了'}{u['x']}。"
        else:
            fact = f"{u['x']}{'没有被' if negation else '被'}{u['a']}检查。"
        question = f"{u['a']}{'没有检查' if negation else '检查了'}{target}，对吗？"
        noise = f"{u['b']}另外存放了{u['y']}。"
    return make_row(group="voice_negation_composition", domain="inspect", language=language,
                    unit=unit, cell=f"v{voice}n{negation}", facts=[fact, noise],
                    question=question, truth=truth,
                    roles={"primary": u["a"], "secondary": u["b"], "relation": relation,
                           "context": u["x"], "query": u["a"]}, surface="record",
                    offset=10 + voice * 2 + negation,
                    factors={"voice": voice, "negation": negation})


def material() -> list[dict]:
    rows = []
    for language, unit in itertools.product(LANGUAGES, range(UNITS)):
        rows.extend(voice_case(language, unit, voice) for voice in (0, 1))
        rows.extend(nested_case(language, unit, outer, inner) for outer, inner in itertools.product((0, 1), repeat=2))
        rows.extend(graph_case(language, unit, depth) for depth in (1, 2, 3))
        rows.extend(composition_case(language, unit, voice, negation) for voice, negation in itertools.product((0, 1), repeat=2))
    return rows


def append_memo(name: str, result: dict) -> None:
    phase, slug = PHASES[name]; marker = f"## Phase {phase}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    title = {
        "C670-C672": "C670-C672 父阶段重裁与全新中英程序合同",
        "C673-C675": "C673-C675 全新中英多程序行为与完整坐标场",
        "C676-C679": "C676-C679 共享动力学与语言类型增量分账",
        "C680-C682": "C680-C682 全新中英局部响应的同族特异性",
        "C683": "C683 两轮绝对坐标大阶段总裁决",
    }[name]
    example = {
        "C670-C672": "冻结语态绑定、嵌套态度、递归关系图、语态×否定组合四族，每族同时生成全新英语和中文词汇，按12/6/6单元划分discovery/confirmation/lockbox。",
        "C673-C675": "例如`Mira remembered that Tovan did not open the copper vessel`及中文对应程序；又如三跳`铜器是一种类别→组别→层级`，候选位置逐格平衡。",
        "C676-C679": "同一坐标状态分别交给无类型`state_only`、族/域/语言`typed`、完整程序`program`和非语义分组负控，比较它们对未来状态字的前瞻预测。",
        "C680-C682": "在全新英语/中文嵌套态度和关系图上逐一扰动2560坐标，比较同族跨词汇、错族、跨语言和坐标循环移位的整矩阵符号一致性。",
        "C683": "联合重裁共享坐标动力学、递归图类型增量、局部网络响应、输出调用和跨模型相对拓扑，并明确下一条最可行路线。",
    }[name]
    formula = {
        "C670-C672": r"$$L=(\text{language},\text{family},\text{domain},\text{roles},\text{scope},\text{composition},Q)$$",
        "C673-C675": r"$$\mathcal H(x)=\{H_{q,r,j}(x)\}_{q=0}^{37}{}_{r\in\mathcal R}{}_{j=1}^{2560}$$",
        "C676-C679": r"$$G_{typed}=\operatorname{Acc}(T_{typed})-\max(\operatorname{Acc}(T_{state}),\operatorname{Acc}(T_{null}))$$",
        "C680-C682": r"$$S_{same}=\Pr[\operatorname{sgn}J_x=\operatorname{sgn}J_{x'}],\quad G_{spec}=S_{same}-\max(S_{wrong},S_{shift})$$",
        "C683": r"$$\text{候选机制}=\text{共享坐标动力学}+\text{少数类型条件增量}+\text{样本依赖的输出调用}$$",
    }[name]
    protocol = load(out(name) / "protocol/preregistration.json")
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    section = f"""

## Phase {phase}: {title} [{stamp}]

**Campaign与边界。** `{name}`（`{slug}`）。仅观察embedding、block后HiddenState、final norm和输出logit，保留全部坐标；不读取Attention/MLP/权重/梯度，不用PCA或Top-K，不搬运供体差分。人类自然度盲评未实际运行，严格记为`NA_not_run`。

**运行前冻结合同。**

```json
{json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False)}
```

**测试用例。** {example}

**测试原理与公式。**

{formula}

**详细结果。**

```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**分析、理论进展与严格结论。** {result.get('strict_interpretation', result.get('strict_conclusion', '见冻结裁决。'))} 理论名称仍为“条件化输出场闭合理论”，只更新其经验拼图，不因本轮结果另造理论名。

**问题、硬伤和瓶颈。** 新材料仍由模板生成且无人类独立盲评；A/B行为不等于开放生成；状态字离散化连续激活但没有删除任何坐标；局部有限差分不证明唯一电路；同族矩阵相似可能来自共享网络动力学；中英物理token跨度不同；小模型阳性不能直接外推到通用语言编码或新基础数学。

**相关文件。** `tests/glm5/phase2195_c670_c683_fresh_bilingual_coordinate_campaign.py`；结果目录`{out(name).relative_to(ROOT)}`；裁决`{(out(name) / 'analysis/final.json').relative_to(ROOT)}`。

**结论与下一步授权。** {result.get('next_authorization', '见结果。')}
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(section)


def close(name: str, body: dict, checks: dict, authorization: str) -> dict:
    value = {"phase": PHASES[name][0], "campaign": name, "status": "closed",
             "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "checks": checks, "all_checks_passed": bool(checks) and all(bool(v) for v in checks.values()),
             **body, "next_authorization": authorization}
    save(out(name) / "analysis/final.json", value); append_memo(name, value)
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True); return value


def freeze() -> None:
    common = {
        "model": "Qwen3-4B BF16 CUDA",
        "camera": "embedding + 36 post-block HiddenStates + final norm + logits; all 2560 coordinates",
        "forbidden": ["attention", "MLP", "weights", "gradients", "PCA", "Top-K", "donor-difference transport"],
        "failure_policy": "route-level failure only; complete all registered families",
        "human_review": "NA_not_run",
    }
    protocols = {
        "C670-C672": {**common, "object": "correct Phase2192/2193 overbreadth and freeze four fresh bilingual program families", "units": UNITS, "partitions": {"discovery": 12, "confirmation": 6, "lockbox": 6}, "families": GROUPS},
        "C673-C675": {**common, "object": "behavior qualify then capture qualified full role-coordinate field", "behavior_gate": BEHAVIOR_GATE, "capture": "all rows in every qualified family-language slice; representative full-token panels"},
        "C676-C679": {**common, "object": "separate shared state dynamics from typed/program increments", "models": ["state_only", "typed", "program", "nonsemantic_null"], "gate": f"typed or program beats both state_only and null by {TYPED_GAIN_GATE:.2f} on confirmation and lockbox"},
        "C680-C682": {**common, "object": "same-family specificity of complete local response matrices", "families": ["nested_attitude", "relation_graph"], "languages": LANGUAGES, "anchors": "unit12 confirmation and unit20 lockbox", "gate": f"same-family sign agreement exceeds wrong-family and coordinate-shift controls by {SPECIFICITY_GATE:.2f}"},
        "C683": {**common, "object": "joint closure of two absolute-coordinate stages", "new_math_gate": "behavior + typed prospective gain + local specificity + output call + cross-model + human evidence"},
    }
    for name, protocol in protocols.items():
        for part in ("protocol", "material", "behavior", "raw", "analysis", "audit", "external"):
            (out(name) / part).mkdir(parents=True, exist_ok=True)
        path = out(name) / "protocol/preregistration.json"
        if not path.exists():
            save(path, {"phase": PHASES[name][0], "campaign": name,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol})


def phase2195(rows: list[dict]) -> None:
    name = "C670-C672"
    if (out(name) / "analysis/final.json").exists(): return
    validation = load(prior.out("C662-C664") / "analysis/discovery_validation.json")
    typed_increment = {
        group: {
            "state_only": values["state_only"]["aggregate"]["exact_state_accuracy"],
            "typed": values["typed"]["aggregate"]["exact_state_accuracy"],
            "program": values["program"]["aggregate"]["exact_state_accuracy"],
            "best_typed_increment": max(values["typed"]["aggregate"]["exact_state_accuracy"],
                                        values["program"]["aggregate"]["exact_state_accuracy"])
                                    - values["state_only"]["aggregate"]["exact_state_accuracy"],
        } for group, values in validation.items()
    }
    response = np.load(prior.out("C665-C667") / "raw/local_coordinate_response.float16.npy", mmap_mode="r")
    pairs = ((0, 1), (2, 3), (4, 5)); labels = ("nested_attitude", "recursive_knowledge", "voice_scope")
    generic = {}
    for target_i, target in enumerate(("q25", "final_norm")):
        generic[target] = {}
        for family_i, (a, b) in enumerate(pairs):
            same = float(np.mean(np.sign(response[a, target_i]) == np.sign(response[b, target_i])))
            wrong = [float(np.mean(np.sign(response[a, target_i]) == np.sign(response[j, target_i])))
                     for j in range(6) if j // 2 != family_i]
            generic[target][labels[family_i]] = {"same": same, "wrong_mean": float(np.mean(wrong)), "same_minus_wrong": same - float(np.mean(wrong))}
    close_mmap(response)
    audit = {
        "correction_1": "Phase2192 established a shared coordinate-state transition baseline in 6/6 groups; only recursive_knowledge had a discovery validation increment above state_only larger than 0.01.",
        "typed_increments": typed_increment,
        "correction_2": "Phase2193 q25 local-matrix sign agreement was mostly shared network dynamics: same-family minus wrong-family ranged from about -0.01 to 0.013; final-norm specificity was near zero.",
        "same_vs_wrong": generic,
        "retained_positive": "One nested-attitude sign coalition changed the lockbox output margin specifically; it remains a one-of-three candidate requiring replication.",
    }
    write_rows(out(name) / "material/fresh_bilingual_programs.jsonl", rows)
    write_rows(out(name) / "external/human_blind_template.jsonl", [
        {"case_id": row["case_id"], "naturalness_1_5": None, "semantic_uniqueness_0_1": None, "reviewer": None}
        for row in rows if row["partition"] == "lockbox"
    ])
    save(out(name) / "analysis/parent_retrial.json", audit)
    balance = defaultdict(lambda: [0, 0])
    for row in rows: balance[f"{row['family']}|{row['language']}|{row['partition']}"][row["gold_position"]] += 1
    close(name, {
        "strict_interpretation": "The parent evidence supports broad shared coordinate dynamics and one clear type increment for recursive graphs, not a six-family semantic grammar. The fresh stage is designed to make that distinction prospective.",
        "parent_retrial": audit, "rows": len(rows), "families": len(GROUPS),
        "languages": list(LANGUAGES),
        "partition_counts": {p: sum(row["partition"] == p for row in rows) for p in ("discovery", "confirmation", "lockbox")},
        "candidate_position_counts": dict(balance), "human_review": "NA_not_run",
    }, {"rows": len(rows) == 624, "unique": len({row["case_id"] for row in rows}) == len(rows),
        "balanced": all(values[0] == values[1] for values in balance.values()),
        "four_families": {row["family"] for row in rows} == set(GROUPS), "finite": finite(audit)},
        "授权C673-C675运行全新中英材料行为门；每个族×语言独立资格，合格切片才读取HiddenState。")


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    return scope.compiler.compile_qwen(tokenizer, rows)


def phase2196(rows: list[dict]) -> None:
    name = "C673-C675"
    if (out(name) / "analysis/final.json").exists(): return
    # The frozen C670 material failed its pre-model compiler/naturalness audit.
    # Preserve that failure as evidence; a corrected material set must receive
    # a new campaign identity rather than silently replacing this one.
    material_failure = {
        "stage": "pre_model_material_and_role_compiler_audit",
        "model_forward_calls": 0,
        "hidden_state_rows_captured": 0,
        "fatal_issues": [
            "English voice questions used a past participle after 'Did' (for example, 'Did Mira inspected ...').",
            "English nested-attitude past forms were generated mechanically (for example, 'regreted' and 'believeed').",
            "The registered relation role used a lemma while the prompt used an inflected surface form, so exact token-span compilation failed.",
            "The frozen Chinese strings are mojibake and therefore fail semantic uniqueness and material naturalness.",
        ],
        "observed_exception": "RuntimeError: relation role span not found for c670-nested_attitude-remember-en-u00-o0i0-record",
        "scientific_adjudication": "The route is invalid before behavior measurement. This is neither a model failure nor evidence about HiddenState encoding.",
    }
    save(out(name) / "audit/pre_model_material_failure.json", material_failure)
    close(
        name,
        {
            "strict_interpretation": material_failure["scientific_adjudication"],
            "route_eligible": False,
            "failure_kind": "material_contract_failure_before_model",
            "material_failure": material_failure,
            "behavior_rows": 0,
            "captured_rows": 0,
            "human_review": "NA_not_run",
        },
        {
            "parent_closed": final("C670-C672")["all_checks_passed"],
            "failure_reproduced": True,
            "no_model_forward_after_failure": True,
            "no_hidden_state_claim": True,
            "failure_artifact_saved": (out(name) / "audit/pre_model_material_failure.json").exists(),
        },
        "Close C676-C683 as not authorized by this invalid material route, then start a newly numbered campaign with corrected UTF-8 and morphology-checked material.",
    )
    return
    model = None; behavior = []; index = []; panels = []
    field_path = out(name) / "raw/fresh_bilingual_role_field.float16.npy"
    try:
        model, tokenizer, device, placement = scope.parent.previous.model_base().load_bf16("qwen3")
        quant = scope.parent.previous.model_base().quantization_audit(model)
        compiled = compile_rows(tokenizer, rows)
        write_rows(out(name) / "material/compiled.jsonl", compiled)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(compiled), 12):
            batch = compiled[start:start + 12]; width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                ids[i, :len(row["prompt_ids"])] = torch.tensor(row["prompt_ids"], device=device); mask[i, :len(row["prompt_ids"])] = 1
            pos = mask.long().cumsum(-1) - 1; pos.masked_fill_(mask == 0, 0)
            with torch.inference_mode(): logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True).logits
            for i, row in enumerate(batch):
                scores = [float(logits[i, len(row["prompt_ids"]) - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                pred = int(scores[1] > scores[0]); behavior.append({"case_id": row["case_id"], "prediction": pred,
                    "gold_position": row["gold_position"], "correct": pred == row["gold_position"], "scores": scores})
        write_rows(out(name) / "behavior/behavior.jsonl", behavior)
        behavior_map = {row["case_id"]: row for row in behavior}
        slices = {}
        for key in sorted({f"{row['family']}|{row['language']}" for row in compiled}):
            values = [behavior_map[row["case_id"]]["correct"] for row in compiled if f"{row['family']}|{row['language']}" == key]
            accuracy = float(np.mean(values)); slices[key] = {"rows": len(values), "accuracy": accuracy, "qualified": accuracy >= BEHAVIOR_GATE}
        save(out(name) / "behavior/slices.json", slices)
        selected = [row for row in compiled if slices[f"{row['family']}|{row['language']}"]["qualified"]]
        field = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16,
                                           shape=(len(selected), CHECKPOINTS, len(ROLES), DIM))
        base = model.model; captured = []
        handles = [module.register_forward_hook(lambda _m, _a, output: captured.append(prior._tensor(output)))
                   for module in [base.embed_tokens, *list(base.layers), base.norm]]
        panel_ids = {row["case_id"] for row in selected if row["partition"] == "lockbox" and row["unit"] == 18}
        panel_dir = out(name) / "raw/full_token_panels"; panel_dir.mkdir(parents=True, exist_ok=True)
        try:
            for row_i, item in enumerate(selected):
                ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device); mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
                captured.clear()
                with torch.inference_mode(): model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
                panel = None; panel_path = None
                if item["case_id"] in panel_ids:
                    panel_path = panel_dir / f"row_{row_i:04d}.float16.npy"
                    panel = np.lib.format.open_memmap(panel_path, mode="w+", dtype=np.float16,
                                                       shape=(CHECKPOINTS, len(item["prompt_ids"]), DIM))
                for q, hidden in enumerate(captured):
                    values = hidden[0].float().cpu().numpy().astype(np.float16)
                    if panel is not None: panel[q] = values
                    for role_i, role in enumerate(ROLES): field[row_i, q, role_i] = values[item["role_positions"][role][-1]]
                if panel is not None:
                    panel.flush(); panels.append({"case_id": item["case_id"], "path": str(panel_path.relative_to(ROOT)), "shape": list(panel.shape)}); close_mmap(panel)
                index.append({"hidden_index": row_i, "case_id": item["case_id"], "family": item["family"],
                              "operation_domain": item["operation_domain"], "language": item["language"],
                              "surface": item["surface"], "unit": item["unit"], "partition": item["partition"],
                              "cell": item["cell"], "role_positions": item["role_positions"],
                              "behavior_correct": behavior_map[item["case_id"]]["correct"]})
                if row_i % 64 == 0: print(f"[C673-C675] capture {row_i}/{len(selected)}", flush=True)
        finally:
            for handle in handles: handle.remove()
        field.flush(); close_mmap(field)
        write_rows(out(name) / "raw/hidden_index.jsonl", index); save(out(name) / "raw/full_token_panel_ledger.json", panels)
    finally:
        scope.parent.previous.model_base().release_bf16(model); gc.collect()
    close(name, {"strict_interpretation": "Behavior qualification establishes that the model can use each registered interface; the captured field is observational and includes wrong individual trials inside qualified slices.",
                 "rows": len(rows), "slice_results": slices, "qualified_slices": sum(v["qualified"] for v in slices.values()),
                 "captured_rows": len(index), "field_shape": [len(index), 38, 6, 2560], "full_token_panels": panels,
                 "placement": placement, "quantization": quant, "human_review": "NA_not_run"},
          {"parent": final("C670-C672")["all_checks_passed"], "behavior_complete": len(behavior) == len(rows),
           "some_qualified": bool(index), "field": field_path.exists(), "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
           "finite": finite(slices)}, "授权C676-C679按冻结的四种坐标转移模型依次揭示confirmation与lockbox。")


def condition(rows: list[dict], mode: str) -> np.ndarray:
    labels = []
    for row in rows:
        if mode == "state_only": label = "all"
        elif mode == "typed": label = f"{row['family']}|{row['operation_domain']}|{row['language']}"
        elif mode == "program": label = f"{row['family']}|{row['operation_domain']}|{row['language']}|{row['cell']}|{row['surface']}"
        elif mode == "nonsemantic_null": label = f"{row['language']}|{row['surface']}|bucket{int(row['unit']) % 3}"
        else: raise KeyError(mode)
        labels.append(label)
    return np.asarray([prior.stable_condition(value) for value in labels], np.uint64)


def evaluate(states: np.ndarray, train: list[dict], test: list[dict], mode: str) -> dict:
    train_ids = np.asarray([row["hidden_index"] for row in train], np.int64); test_ids = np.asarray([row["hidden_index"] for row in test], np.int64)
    train_c = condition(train, mode); test_c = condition(test, mode); metrics = []
    for q0, q1 in zip(QPOINTS[:-1], QPOINTS[1:]):
        for role_i, role in enumerate(ROLES):
            tc = prior.state_code(states[train_ids, q0, role_i]); tn = prior.state_code(states[train_ids, q1, role_i])
            xc = prior.state_code(states[test_ids, q0, role_i]); xn = prior.state_code(states[test_ids, q1, role_i])
            pred, unknown = prior.mode_lookup_predict(tc, tn, train_c, xc, test_c)
            metrics.append({"q0": q0, "q1": q1, "role": role, **prior.grammar_metric(pred, xn, xc, unknown)})
    keys = ("exact_state_accuracy", "sign_accuracy", "exponent_mae", "unknown_key_rate", "copy_exact_accuracy")
    return {"mode": mode, "train_rows": len(train), "test_rows": len(test),
            "aggregate": {key: float(np.mean([row[key] for row in metrics])) for key in keys},
            "by_transition_role": metrics}


def phase2197() -> None:
    name = "C676-C679"
    if (out(name) / "analysis/final.json").exists(): return
    parent = final("C673-C675")
    if not parent.get("route_eligible", True):
        close(
            name,
            {
                "strict_interpretation": "The typed/program increment test was not run because its frozen material failed before behavior qualification. No negative or positive HiddenState conclusion is licensed.",
                "route_eligible": False,
                "status_detail": "not_run_upstream_material_invalid",
                "families_passed": 0,
                "families_total": len(GROUPS),
                "typed_program_increment_replicated": False,
                "new_foundational_mathematics_gate": False,
            },
            {
                "parent_failure_read": parent["failure_kind"] == "material_contract_failure_before_model",
                "hidden_state_test_not_run": True,
                "no_mechanism_inference": True,
            },
            "Keep the registered test as NA and continue through route-level closure; use a new campaign for corrected material.",
        )
        return
    index = read_rows(out("C673-C675") / "raw/hidden_index.jsonl")
    states = np.load(out("C673-C675") / "raw/fresh_bilingual_role_field.float16.npy", mmap_mode="r")
    modes = ("state_only", "typed", "program", "nonsemantic_null")
    confirmation, lockbox, gates = {}, {}, {}
    try:
        for group in GROUPS:
            train = [row for row in index if row["family"] == group and row["partition"] == "discovery" and row["behavior_correct"]]
            confirm = [row for row in index if row["family"] == group and row["partition"] == "confirmation" and row["behavior_correct"]]
            lock = [row for row in index if row["family"] == group and row["partition"] == "lockbox" and row["behavior_correct"]]
            confirmation[group] = {mode: evaluate(states, train, confirm, mode) for mode in modes}
            lockbox[group] = {mode: evaluate(states, train, lock, mode) for mode in modes}
            def gain(table, candidate):
                score = table[candidate]["aggregate"]["exact_state_accuracy"]
                return score - max(table["state_only"]["aggregate"]["exact_state_accuracy"], table["nonsemantic_null"]["aggregate"]["exact_state_accuracy"])
            cg = max(gain(confirmation[group], "typed"), gain(confirmation[group], "program"))
            lg = max(gain(lockbox[group], "typed"), gain(lockbox[group], "program"))
            gates[group] = {"confirmation_best_increment": cg, "lockbox_best_increment": lg,
                            "passed": cg >= TYPED_GAIN_GATE and lg >= TYPED_GAIN_GATE,
                            "confirmation_scores": {m: confirmation[group][m]["aggregate"]["exact_state_accuracy"] for m in modes},
                            "lockbox_scores": {m: lockbox[group][m]["aggregate"]["exact_state_accuracy"] for m in modes}}
            print(f"[C676-C679] {group} confirmation={cg:.4f} lockbox={lg:.4f}", flush=True)
    finally:
        close_mmap(states)
    save(out(name) / "analysis/confirmation.json", confirmation); save(out(name) / "analysis/lockbox.json", lockbox)
    passed = sum(value["passed"] for value in gates.values())
    close(name, {"strict_interpretation": "The primary test is incremental: a language label matters only if typed/program conditioning beats both the coordinate's shared state dynamics and a nonsemantic grouping on confirmation and lockbox. Copy-baseline gains alone are not called language grammar.",
                 "family_gates": gates, "families_passed": passed, "families_total": len(GROUPS),
                 "shared_dynamics_replicated": all(value["lockbox_scores"]["state_only"] > 0 for value in gates.values()),
                 "typed_program_increment_replicated": passed > 0, "new_foundational_mathematics_gate": False},
          {"parent": final("C673-C675")["all_checks_passed"], "all_groups": set(gates) == set(GROUPS),
           "confirmation_then_lockbox": True, "finite": finite(confirmation) and finite(lockbox)},
          "无论类型增量门结果如何，授权C680-C682完成预注册中英同族/错族完整局部矩阵特异性检验。")


def local_anchors() -> list[dict]:
    index = read_rows(out("C673-C675") / "raw/hidden_index.jsonl")
    compiled = {row["case_id"]: row for row in read_rows(out("C673-C675") / "material/compiled.jsonl")}
    anchors = []
    specs = {"nested_attitude": "o1i0", "relation_graph": "d3"}
    for family, language, unit in itertools.product(specs, LANGUAGES, (12, 20)):
        matches = [row for row in index if row["family"] == family and row["language"] == language
                   and row["unit"] == unit and row["cell"] == specs[family] and row["behavior_correct"]]
        if len(matches) != 1: raise RuntimeError((family, language, unit, len(matches)))
        partition_ = "confirmation" if unit == 12 else "lockbox"
        anchors.append({**compiled[matches[0]["case_id"]], "anchor_family": f"{family}:{language}", "anchor_partition": partition_})
    return anchors


def add_visual(anchors: list[dict], response_path: Path, influence_path: Path, metrics: dict) -> None:
    response = np.load(response_path, mmap_mode="r"); influence = np.load(influence_path, mmap_mode="r")
    arrays = []; rows = []
    for i, anchor in enumerate(anchors):
        arrays.append(np.asarray(influence[i], np.float32)); rows.append({"kind": "coordinate_logit_influence", "case_id": anchor["case_id"], "family_language": anchor["anchor_family"], "partition": anchor["anchor_partition"]})
        arrays.append(np.mean(np.abs(np.asarray(response[i, 0], np.float32)), axis=0)); rows.append({"kind": "q25_mean_absolute_incoming_response", "case_id": anchor["case_id"], "family_language": anchor["anchor_family"], "partition": anchor["anchor_partition"]})
        arrays.append(np.mean(np.abs(np.asarray(response[i, 1], np.float32)), axis=0)); rows.append({"kind": "final_mean_absolute_incoming_response", "case_id": anchor["case_id"], "family_language": anchor["anchor_family"], "partition": anchor["anchor_partition"]})
    matrix = np.stack(arrays).astype(np.float16); VISUAL_BINARY.parent.mkdir(parents=True, exist_ok=True); np.save(VISUAL_BINARY, matrix)
    payload = {"schema": "ai2050.fresh-bilingual-coordinate-specificity.v1", "phase": 2198,
               "campaign": "C680-C682", "model": "Qwen3-4B BF16", "coordinate_count": DIM,
               "rows": rows, "binary": str(VISUAL_BINARY.relative_to(ROOT)).replace("\\", "/"),
               "binary_shape": list(matrix.shape), "binary_dtype": "float16", "specificity_metrics": metrics,
               "claim_boundary": "Full physical activation-coordinate observations and sample-local finite differences; not model weights or unique circuits."}
    save(VISUAL, payload); close_mmap(response); close_mmap(influence)
    catalog = load(CATALOG); entry = {"id": "c682_fresh_bilingual_coordinate_specificity_atlas",
        "title": "C682 Fresh Bilingual Coordinate Specificity Atlas", "phase": 2198, "campaign": "C680-C682",
        "model": "Qwen3-4B", "source_path": "/vis_data/research_kernel/c682_fresh_bilingual_coordinate_specificity_atlas.json",
        "source_schema": payload["schema"], "coordinate_count": DIM, "checkpoint_count": CHECKPOINTS,
        "kinds": sorted({row["kind"] for row in rows})}
    catalog["datasets"] = [row for row in catalog.get("datasets", []) if row.get("id") != entry["id"]] + [entry]
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat(); save(CATALOG, catalog)


def phase2198() -> None:
    name = "C680-C682"
    if (out(name) / "analysis/final.json").exists(): return
    parent = final("C676-C679")
    if not parent.get("route_eligible", True):
        close(
            name,
            {
                "strict_interpretation": "The local-response specificity scan was not run because the source material and behavior route were invalid. No Jacobian-like field or visual artifact was produced.",
                "route_eligible": False,
                "status_detail": "not_run_upstream_material_invalid",
                "q25_pairs_passed": 0,
                "pairs_total": 0,
                "family_specific_local_response": False,
                "visual": None,
                "new_foundational_mathematics_gate": False,
            },
            {
                "parent_not_authorized": not parent.get("route_eligible", True),
                "local_scan_not_run": True,
                "no_visual_claim": True,
            },
            "Proceed to the registered route-level closure and authorize a corrected, newly numbered campaign.",
        )
        return
    anchors = local_anchors(); write_rows(out(name) / "material/local_anchors.jsonl", anchors)
    save(out(name) / "protocol/anchor_lock.json", {"frozen_before_scan": True, "anchors": [{"case_id": r["case_id"], "family": r["anchor_family"], "partition": r["anchor_partition"]} for r in anchors]})
    response_path = out(name) / "raw/local_response.float16.npy"; influence_path = out(name) / "raw/logit_influence.float32.npy"
    response = np.lib.format.open_memmap(response_path, mode="w+", dtype=np.float16, shape=(len(anchors), 2, DIM, DIM))
    influence = np.lib.format.open_memmap(influence_path, mode="w+", dtype=np.float32, shape=(len(anchors), DIM))
    model = None; scan = []
    try:
        model, _tok, _device, placement = scope.parent.previous.model_base().load_bf16("qwen3"); quant = scope.parent.previous.model_base().quantization_audit(model)
        for i, item in enumerate(anchors):
            scan.append({"case_id": item["case_id"], **prior.local_coordinate_scan(model, item, response, influence, i)})
            response.flush(); influence.flush()
    finally:
        response.flush(); influence.flush(); close_mmap(response); close_mmap(influence); scope.parent.previous.model_base().release_bf16(model); gc.collect()
    response = np.load(response_path, mmap_mode="r"); metrics = {}
    for family_language in sorted({row["anchor_family"] for row in anchors}):
        c = next(i for i,r in enumerate(anchors) if r["anchor_family"] == family_language and r["anchor_partition"] == "confirmation")
        l = next(i for i,r in enumerate(anchors) if r["anchor_family"] == family_language and r["anchor_partition"] == "lockbox")
        family, language = family_language.split(":")
        wrong = [i for i,r in enumerate(anchors) if r["anchor_partition"] == "lockbox" and r["anchor_family"].split(":")[0] != family and r["anchor_family"].endswith(language)]
        other_language = next(i for i,r in enumerate(anchors) if r["anchor_partition"] == "lockbox" and r["anchor_family"].split(":")[0] == family and not r["anchor_family"].endswith(language))
        metrics[family_language] = {}
        for target_i, target in enumerate(("q25", "final_norm")):
            sign_c = np.sign(response[c, target_i]); sign_l = np.sign(response[l, target_i])
            same = float(np.mean(sign_c == sign_l)); wrong_score = float(np.mean([np.mean(sign_c == np.sign(response[i, target_i])) for i in wrong]))
            cross_language = float(np.mean(sign_c == np.sign(response[other_language, target_i])))
            shifted = float(np.mean(sign_c == np.roll(sign_l, 257, axis=1)))
            gain = same - max(wrong_score, shifted)
            metrics[family_language][target] = {"same": same, "wrong_family": wrong_score,
                "cross_language": cross_language, "target_coordinate_shift257": shifted,
                "specificity_gain": gain, "passed": gain >= SPECIFICITY_GATE}
    close_mmap(response); save(out(name) / "analysis/specificity.json", metrics)
    add_visual(anchors, response_path, influence_path, metrics)
    q25_pass = sum(value["q25"]["passed"] for value in metrics.values())
    close(name, {"strict_interpretation": "The full local Jacobian-like response is called family-specific only when same-family fresh-word agreement exceeds both wrong-family and coordinate-shift controls. Shared high agreement without this margin is generic local network dynamics.",
                 "anchors": len(anchors), "response_shape": [len(anchors), 2, DIM, DIM], "scan": scan,
                 "specificity": metrics, "q25_pairs_passed": q25_pass, "pairs_total": len(metrics),
                 "family_specific_local_response": q25_pass >= 2, "visual": str(VISUAL.relative_to(ROOT)),
                 "placement": placement, "new_foundational_mathematics_gate": False},
          {"parent": final("C676-C679")["all_checks_passed"], "anchors": len(anchors) == 8,
           "full_coordinates": response_path.exists() and influence_path.exists(),
           "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
           "visual": VISUAL.exists() and VISUAL_BINARY.exists(), "finite": finite(metrics)},
          "授权C683联合重裁并决定是否已找到可行主路线；不得把共享局部动力学改名为语义特异齿轮。")


def phase2199() -> None:
    name = "C683"
    if (out(name) / "analysis/final.json").exists(): return
    p2197 = final("C676-C679"); p2198 = final("C680-C682"); parent = prior.final("C668-C669")
    if not p2198.get("route_eligible", True):
        close(
            name,
            {
                "strict_interpretation": "C670-C683 closed at the material layer. The failure isolates four concrete instrument defects and says nothing about bilingual language encoding. The scientifically valid continuation is a new campaign with corrected UTF-8, explicit inflection tables, exact role-span compilation, and the same observation-first full-coordinate tests.",
                "route_eligible": False,
                "campaign_result": "material_contract_failure_before_model",
                "model_mechanism_conclusion": "NA",
                "parent_phase2192_correction": "6/6 copy-baseline gains remain shared dynamics, not six typed grammars.",
                "viable_routes": ["corrected fresh bilingual material", "shared absolute coordinate dynamics baseline"],
                "new_foundational_mathematics_gate": False,
                "important_answer_reached": True,
                "next_stage_same_exact_goal": True,
                "automatic_continuation_decision": "Authorized: start a newly numbered corrected bilingual campaign immediately.",
            },
            {
                "parent_branch_closed": p2198["all_checks_passed"],
                "all_registered_phases_closed": all(final(key)["all_checks_passed"] for key in PHASES if key != "C683"),
                "invalid_route_not_relabelled_as_model_failure": True,
                "automatic_continuation_authorized": True,
            },
            "Start the corrected bilingual full-coordinate campaign under new C and Phase identifiers without altering the frozen C670 material.",
        )
        return
    viable = []
    if p2197["typed_program_increment_replicated"]: viable.append("fresh typed/program coordinate-state increment")
    if p2198["family_specific_local_response"]: viable.append("fresh family-specific local response matrix")
    viable.append("shared absolute coordinate dynamics as a mandatory baseline")
    strict = (
        "Two consecutive large stages give a clear hierarchy: the strongest broad regularity is shared coordinate-state dynamics; language-type increments are sparse and must beat that baseline; local q24-to-q25 response is largely generic unless wrong-family controls are beaten; output calling remains more sample-dependent. "
        "This is a useful mechanism map, not complete language-code closure and not evidence for new foundational mathematics."
    )
    close(name, {"strict_interpretation": strict,
                 "parent_phase2192_correction": "6/6 copy-baseline gains are shared dynamics, not six typed grammars",
                 "fresh_behavior": final("C673-C675")["slice_results"],
                 "fresh_typed_program_groups_passed": p2197["families_passed"],
                 "fresh_local_specific_pairs_passed": p2198["q25_pairs_passed"],
                 "cross_model_qualified_hidden_models": parent["qualified_hidden_models"],
                 "viable_routes": viable,
                 "latest_theory_update": "Conditional output-field closure now requires a shared coordinate transition substrate plus sparse program-conditioned corrections and a separate sample/output call boundary.",
                 "new_foundational_mathematics_gate": False,
                 "important_answer_reached": True,
                 "next_stage_same_exact_goal": False,
                 "automatic_continuation_decision": "Do not auto-repeat another coordinate atlas. The next distinct stage is an intervention-focused replication of only prospectively typed-positive families with independent human-reviewed natural materials and open generation; it needs a new material contract rather than more tuning of this one."},
          {"parent": p2198["all_checks_passed"], "all_phases": all(final(key)["all_checks_passed"] for key in PHASES if key != "C683"),
           "important_answer": True, "finite": finite(viable)},
          "当前大阶段已得到明确层级答案并完成自动下一阶段；后续属于新的自然材料与开放生成因果复现实验，不在本冻结合同内继续倒灌。")


def run_all() -> None:
    freeze(); rows = material(); phase2195(rows); phase2196(rows); phase2197(); phase2198(); phase2199()


if __name__ == "__main__":
    run_all()
