#!/usr/bin/env python3
"""C745-C760 fresh-concept response-passport and causal-call campaign.

Only embeddings, post-block HiddenStates, final norm and logits are read.
The campaign retains all 2560 physical activation coordinates and uses no
PCA, Top-K screening, cosine similarity, gradients, weights, attention/MLP
internals, or donor HiddenState differences.
"""
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
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c760_fresh_passport_causal_atlas.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c760_fresh_passport_causal_atlas.float16.npy"
PARENT_VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c744_response_equivalence_coordinate_atlas.json"
PARENT_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c744_response_equivalence_coordinate_atlas.float16.npy"
sys.path.insert(0, str(TESTS))

import model_utils
import phase2105_c571_c589_scope_program_algebra_campaign as scope
import phase2190_c656_c669_absolute_coordinate_grammar_campaign as local_base
import phase2200_c684_c709_unified_relation_response_campaign as behavior
import phase2205_c710_c744_response_equivalence_atlas_campaign as parent


PHASES = {
    "C745-C748": (2211, "fresh_concept_and_surface_contract"),
    "C749-C753": (2212, "frozen_passport_fresh_concept_replication"),
    "C754-C758": (2213, "generation_time_passport_deletion_and_rescue"),
    "C759-C760": (2214, "joint_adjudication_visualization_and_cleanup"),
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower().replace('-', '_')}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

DIM = 2560
QPOINTS = (0, 8, 16, 24, 32, 37)
ROLES = ("relation", "boundary")
FAMILIES = parent.FAMILIES
LANGUAGES = parent.LANGUAGES
TRANSFORMS = parent.TRANSFORMS
UNITS = 8
BEHAVIOR_GATE = 0.75
PASSPORT_GAIN_GATE = parent.PASSPORT_GAIN_GATE
CAUSAL_GAIN_GATE = 0.05
CELL_NAMES = parent.CELL_NAMES

NAMES_A = ("Elora", "Hadrian", "Isolde", "Jarek", "Lucan", "Maelis", "Nyra", "Osric")
NAMES_B = ("Petra", "Quillan", "Sabine", "Theron", "Ulric", "Veyra", "Wylan", "Zevan")
OBJECTS_EN = ("lantern", "compass", "violin", "goblet", "helmet", "anchor", "saddle", "kettle")
DISTRACT_EN = ("mirror", "basket", "hammer", "pillow", "ladder", "ribbon", "shovel", "candle")
OBJECTS_ZH = ("灯笼", "罗盘", "小提琴", "高脚杯", "头盔", "锚", "马鞍", "水壶")
DISTRACT_ZH = ("镜子", "篮子", "锤子", "枕头", "梯子", "丝带", "铲子", "蜡烛")
FRENCH = ("lune", "étoile", "rivière", "montagne", "fenêtre", "porte", "chaise", "table")
SOURCE_WORDS = ("moon", "star", "river", "mountain", "window", "door", "chair", "table")
TYPE_EN = ("artifact", "manufactured item", "physical object")
TYPE_ZH = ("人工制品", "制成品", "实体物品")

TITLES = {
    "C745-C748": "独立新概念与新表面六族合同",
    "C749-C753": "冻结响应护照的前瞻新概念复验",
    "C754-C758": "生成时多检查点全坐标删除与救援",
    "C759-C760": "续跑联合裁决、参数级可视化与清理",
}
FORMULAS = {
    "C745-C748": "$$\n\\mathcal D_{fresh}\\cap\\mathcal D_{parent}=\\varnothing,\\qquad G_{row}=\\mathbf 1[\\hat y_{cand}=y\\land\\hat y_{gen}=y]\n$$",
    "C749-C753": "$$\nP_o^{old}(q,r,j)=(Z_0,Z_o),\\quad A_x(P)=\\frac{1}{|Q||R|d}\\sum_{q,r,j}\\mathbf 1[(Z_0^x,Z_o^x)=P_o^{old}]\n$$\n$$\nG_x=A_x(P_o^{old})-\\max\\{A_x(P_{wrong}^{old}),A_x(P_{shift}^{old})\\}\n$$",
    "C754-C758": "$$\nM_o(q,r,j)=\\mathbf 1[Z_0^{old}(q,r,j)\\ne Z_o^{old}(q,r,j)]\n$$\n$$\nG_{rescue}=(m_{correct}-m_{delete})-\\max(m_{wrong}-m_{delete},m_{shift}-m_{delete},0)\n$$",
    "C759-C760": "$$\nH\\sim_{\\mathcal T}H'\\;\\text{only if observational passports and future intervention responses both agree.}\n$$",
}


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


def append_memo(name: str, result: dict) -> None:
    phase = PHASES[name][0]
    marker = f"## Phase {phase}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    prereg = load(out(name) / "protocol/preregistration.json")
    text = f"""

## Phase {phase}: {TITLES[name]} [{stamp}]

**研究对象与边界。** `{name}` 延续 Phase 2210 自动授权，只分析 embedding、各 block 后 HiddenState、final norm 与 logits 的 2560 个物理激活坐标。没有读取 Attention/MLP 内部、权重或梯度；没有使用 PCA、Top-K、余弦筛选或 donor HiddenState 差分搬运。六个语言族是外部实验坐标，不预设为内部模块。独立人类盲评未运行，严格记为 `NA_not_run`。

**运行前冻结合同。**
```json
{json.dumps(prereg, ensure_ascii=False, indent=2)}
```

**测试原理、测试用例与数学公式。** 新词汇包括 `lantern/灯笼`、`compass/罗盘` 等，覆盖递归分类、嵌套态度、语态否定、时间更新、共指绑定和翻译路线；每族同时改变词汇、事实顺序、语态或复合控制。公式：

{FORMULAS[name]}

**详细结果与门槛。**
```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**结果分析与理论进展。** {result.get('strict_interpretation')} 理论主体继续保持“条件化输出场闭合理论”，组织原则继续保持“复用—差分—条件化”。响应护照是经验图谱对象；只有跨新概念预测与未来干预响应同时稳定，才可能升级为响应等价类。

**问题、硬伤与瓶颈。** 材料仍是受控双语而非开放自然语料；人类自然度为 NA；状态码离散化损失连续幅值；全坐标掩码可能同时伤害通用计算与目标操作；小模型结果不能直接外推；候选 margin 与自由生成必须分账；物理激活坐标不是模型参数；任何平均增益都必须回到独立概念单元复核。

**相关文件。** 主脚本 `tests/glm5/phase2211_c745_c760_fresh_passport_causal_campaign.py`；结果目录 `{out(name).relative_to(ROOT)}`；正式裁决 `{(out(name) / 'analysis/final.json').relative_to(ROOT)}`。

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


def freeze() -> None:
    parent_final = parent.final("C740-C744")
    if not parent_final["all_checks_passed"] or not parent_final["next_stage_same_goal"]:
        raise RuntimeError("Phase2210 did not authorize same-object continuation")
    common = {
        "frozen_before_model": True, "parent_phase": 2210,
        "model": "Qwen3-4B local BF16 CUDA", "dimensions": DIM,
        "families": list(FAMILIES), "languages": list(LANGUAGES),
        "checkpoints": list(QPOINTS), "roles": list(ROLES),
        "forbidden": ["attention_internal", "mlp_internal", "weights", "gradients", "PCA", "Top-K", "cosine", "donor_hiddenstate_difference"],
        "human_review": "NA_not_run", "behavior_gate": BEHAVIOR_GATE,
        "passport_gain_gate": PASSPORT_GAIN_GATE, "causal_gain_gate": CAUSAL_GAIN_GATE,
        "reveal_rule": "Objects, materials, masks, controls, doses and gates remain frozen after reveal; executor bugs may be repaired only with an explicit audit entry.",
    }
    details = {
        "C745-C748": {"object": "fresh concepts and fresh surfaces disjoint from Phase2205 primary-role lexicon", "rows": 384},
        "C749-C753": {"object": "predict fresh paired state codes using only frozen c744 passports", "gate": PASSPORT_GAIN_GATE},
        "C754-C758": {"object": "multi-checkpoint, two-role complete-coordinate deletion and state-code rescue", "gate": CAUSAL_GAIN_GATE},
        "C759-C760": {"object": "joint adjudication, exact-coordinate client dataset and hash-then-clean"},
    }
    for name, detail in details.items():
        path = out(name) / "protocol/preregistration.json"
        if not path.exists():
            save(path, {**common, **detail, "campaign": name, "phase": PHASES[name][0]})


def partition(unit: int) -> str:
    return "confirmation" if unit < 4 else "lockbox"


def make_case(family: str, language: str, unit: int, cell_i: int) -> dict:
    a, b = NAMES_A[unit], NAMES_B[unit]
    x = OBJECTS_EN[unit] if language == "en" else OBJECTS_ZH[unit]
    y = DISTRACT_EN[unit] if language == "en" else DISTRACT_ZH[unit]
    truth = unit % 2 == 0
    surface = CELL_NAMES[cell_i]
    if family == "recursive_knowledge":
        t1, t2, t3 = TYPE_EN if language == "en" else TYPE_ZH
        target = t3 if truth else y
        if language == "en":
            facts = [f"the {x} belongs to the {t1} class", f"every {t1} belongs to the {t2} class", f"every {t2} belongs to the {t3} class"]
            orders = (facts, [facts[2], facts[0], facts[1]], [facts[1], facts[2], facts[0]], facts + [f"the {y} occurs only in an unrelated note"])
            core = "A newly audited classification sheet records: " + "; ".join(orders[cell_i]) + f". From that sheet alone, does the {x} belong to the {target} class?"
            relation = "belongs to"
        else:
            facts = [f"{x}属于{t1}类", f"每个{t1}都属于{t2}类", f"每个{t2}都属于{t3}类"]
            orders = (facts, [facts[2], facts[0], facts[1]], [facts[1], facts[2], facts[0]], facts + [f"{y}只出现在一条无关备注中"])
            core = "一份新核验的分类表记录：" + "；".join(orders[cell_i]) + f"。只根据该表，{x}属于{target}类吗？"
            relation = "属于"
        roles = {"primary": x, "secondary": t1, "relation": relation, "context": t3, "query": target}
    elif family == "nested_attitude":
        target = x if truth else y
        if language == "en":
            variants = (f"{a} recalled that {b} stored the {x}", f"During review, {a} recalled that {b} stored the {x}", f"{a} recalled that the {x} had been stored by {b}", f"The signed note confirms that {a} recalled {b}'s storing of the {x}")
            core = f"A newly audited memory note says: {variants[cell_i]}. Does it say that {a} recalled that {b} stored the {target}?"
            relation = "recalled"
        else:
            variants = (f"{a}记得{b}存放了{x}", f"复核时，{a}记得{b}存放了{x}", f"{a}记得{x}由{b}存放", f"签字记录确认{a}记得{b}存放{x}这件事")
            core = f"一份新核验的记忆记录写道：{variants[cell_i]}。它是否表明{a}记得{b}存放了{target}？"
            relation = "记得"
        roles = {"primary": a, "secondary": b, "relation": relation, "context": x, "query": target}
    elif family == "voice_negation":
        target = x if truth else y
        if language == "en":
            variants = (f"{b} carried the {x}", f"the witness saw {b} carry the {x}", f"the {x} was carried by {b}", f"it is not true that {b} failed to carry the {x}")
            core = f"A fresh event report checked by {a} states that {variants[cell_i]}. Does the report support that {b} carried the {target}?"
            relation = "carried"
        else:
            variants = (f"{b}搬运了{x}", f"证人看到{b}搬运{x}", f"{x}由{b}搬运", f"并非{b}没有搬运{x}")
            core = f"一份由{a}新核验的事件报告写道：{variants[cell_i]}。报告是否支持{b}搬运了{target}？"
            relation = "搬运"
        roles = {"primary": b, "secondary": a, "relation": relation, "context": x, "query": target}
    elif family == "temporal_update":
        target = x if truth else y
        if language == "en":
            variants = (f"{a} first held the {y}, but the current entry says {a} holds the {x}", f"the newest entry lists the {x} for {a}; the older one listed the {y}", f"after the {y} entry, the log was updated to the {x} for {a}", f"the obsolete {y} entry was replaced by the current {x} entry for {a}")
            core = f"A fresh update log maintained by {b} says: {variants[cell_i]}. Is the current item for {a} the {target}?"
            relation = "current"
        else:
            variants = (f"{a}起初持有{y}，但当前条目写着{a}持有{x}", f"最新条目为{a}列出{x}，旧条目列出{y}", f"在{y}条目之后，日志更新为{a}持有{x}", f"过时的{y}条目已被{a}当前持有{x}的条目替换")
            core = f"一份由{b}维护的新更新日志写道：{variants[cell_i]}。{a}当前的物品是{target}吗？"
            relation = "当前"
        roles = {"primary": a, "secondary": b, "relation": relation, "context": x, "query": target}
    elif family == "coreference_binding":
        target = a if truth else b
        if language == "en":
            variants = (f"{a} told {b}, 'I stored the {x}.'", f"While speaking to {b}, {a} said, 'I stored the {x}.'", f"{b} heard {a} say, 'I stored the {x}.'", f"The transcript records {a}'s words to {b}: 'I stored the {x}.'")
            core = f"A fresh transcript states: {variants[cell_i]} In the quoted sentence, does 'I' refer to {target}?"
            relation = "refer to"
        else:
            variants = (f"{a}对{b}说：‘我存放了{x}。’", f"{a}与{b}交谈时说：‘我存放了{x}。’", f"{b}听见{a}说：‘我存放了{x}。’", f"记录保留了{a}对{b}的原话：‘我存放了{x}。’")
            core = f"一份新谈话记录写道：{variants[cell_i]} 引号中的“我”指的是{target}吗？"
            relation = "指的是"
        roles = {"primary": a, "secondary": b, "relation": relation, "context": x, "query": target}
    else:
        source = SOURCE_WORDS[unit]
        right = FRENCH[unit]
        wrong = FRENCH[(unit + 3) % UNITS]
        target = right if truth else wrong
        if language == "en":
            variants = (f"the French translation of {source} is {right}", f"a checked glossary maps {source} to French {right}", f"for the English word {source}, the French entry is {right}", f"the bilingual card pairs {source} with {right}")
            core = f"A fresh bilingual record states that {variants[cell_i]}. According to it, is the French translation of {source} {target}?"
            relation = "French translation"
        else:
            variants = (f"英文{source}的法语翻译是{right}", f"核验词表把英文{source}映射为法语{right}", f"对于英文词{source}，法语条目是{right}", f"双语卡片把英文{source}与法语{right}配对")
            core = f"一份新双语记录写道：{variants[cell_i]}。根据记录，英文{source}的法语翻译是{target}吗？"
            relation = "法语翻译"
        roles = {"primary": source, "secondary": right, "relation": relation, "context": target, "query": target}
    yes, no = (("Yes", "No") if language == "en" else ("是", "否"))
    correct, wrong_answer = (yes, no) if truth else (no, yes)
    gold = (unit + cell_i + int(language == "zh")) % 2
    options = f"(A) {correct} (B) {wrong_answer}" if gold == 0 else f"(A) {wrong_answer} (B) {correct}"
    prompt = f"{core} {options}. Reply only A or B." if language == "en" else f"{core} {options}。只回答A或B。"
    free_prompt = f"{core} Answer only Yes or No." if language == "en" else f"{core} 只回答‘是’或‘否’。"
    return {
        "case_id": f"c745-{family}-{language}-u{unit:02d}-{CELL_NAMES[cell_i]}",
        "panel": "fresh_response_passport", "family": family, "language": language,
        "operation_type": family, "operation_domain": f"fresh:{family}:{CELL_NAMES[cell_i]}",
        "surface": surface, "cell": CELL_NAMES[cell_i], "cell_i": cell_i,
        "transform_id": cell_i, "unit": unit, "partition": partition(unit), "truth": truth,
        "correct_answer": correct, "wrong_answer": wrong_answer, "gold_position": gold,
        "prompt_core": core, "prompt": prompt, "free_prompt": free_prompt,
        "role_values": roles, "factors": {"fresh_lexicon": 1, "surface_transform": cell_i},
        "semantic_graph": {"external_family": family, "fresh_concept": True, "internal_module_assumption": False},
    }


def material() -> list[dict]:
    return [make_case(family, language, unit, cell_i)
            for family, language, unit, cell_i in itertools.product(FAMILIES, LANGUAGES, range(UNITS), range(4))]


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                         local_files_only=True, use_fast=False)


def phase2211(rows: list[dict]) -> None:
    name = "C745-C748"
    if (out(name) / "analysis/final.json").exists():
        return
    tokenizer = load_tokenizer()
    compiled = scope.compiler.compile_qwen(tokenizer, rows)
    material_path = out(name) / "material/fresh_six_family_bilingual.jsonl"
    compiled_path = out(name) / "material/qwen_compiled.jsonl"
    write_rows(material_path, rows); write_rows(compiled_path, compiled)
    balances = defaultdict(lambda: [0, 0]); truths = defaultdict(lambda: [0, 0])
    for row in rows:
        key = f"{row['family']}|{row['language']}|{row['partition']}"
        balances[key][row["gold_position"]] += 1; truths[key][int(row["truth"])] += 1
    parent_rows = read_rows(parent.out("C710-C714") / "material/six_family_bilingual_graph.jsonl")
    parent_primary = {row["role_values"]["primary"] for row in parent_rows}
    fresh_primary = {row["role_values"]["primary"] for row in rows}
    overlap = sorted(parent_primary & fresh_primary)
    missing_roles = [{"case_id": row["case_id"], "role": role, "value": value}
                     for row in rows for role, value in row["role_values"].items() if value not in row["prompt_core"]]
    widths = [len(row["prompt_ids"]) for row in compiled]
    zero = {"always_A": float(np.mean([row["gold_position"] == 0 for row in rows])),
            "always_B": float(np.mean([row["gold_position"] == 1 for row in rows])),
            "truth_prior": float(np.mean([row["truth"] for row in rows]))}
    review_path = out(name) / "audit/human_blind_review_template.jsonl"
    write_rows(review_path, [{"case_id": row["case_id"], "semantic_unique": None, "natural": None, "reviewer": None}
                             for row in rows])
    audit = {"rows": len(rows), "candidate_balance": balances, "truth_balance": truths,
             "zero_models": zero, "parent_primary_overlap": overlap, "missing_roles": missing_roles,
             "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
             "human_review": "NA_not_run"}
    save(out(name) / "audit/material_audit.json", audit)
    close(name, {
        "strict_interpretation": "The fresh material changes lexical identities and surface realizations while preserving the six external operation coordinates. Passing this phase validates the executable material only; it is not a neural claim.",
        "material_audit": audit, "material_sha256": file_hash(material_path),
        "compiled_sha256": file_hash(compiled_path), "new_foundational_mathematics_gate": False,
    }, {
        "parent": parent.final("C740-C744")["all_checks_passed"], "rows": len(rows) == 384,
        "compiler": len(compiled) == len(rows), "roles": not missing_roles,
        "candidate_balance": all(v[0] == v[1] for v in balances.values()),
        "truth_balance": all(v[0] == v[1] for v in truths.values()),
        "zero_models": all(abs(v - 0.5) < 1e-12 for v in zero.values()),
        "lexical_disjoint": not overlap, "width": max(widths) <= 220,
    }, "Authorize C749-C753 to run dual behavior and test frozen C721-C726 passports on fresh concepts without refitting.")


def load_parent_prototypes() -> dict[tuple[str, int, str], np.ndarray]:
    payload = load(PARENT_VISUAL)
    matrix = np.load(PARENT_BINARY, mmap_mode="r")
    result = {}
    for row_i, row in enumerate(payload["rows"]):
        if row["kind"] == "paired_state_transition_code":
            result[(row["group"], int(row["checkpoint"]), row["role"])] = np.rint(np.asarray(matrix[row_i], np.float32)).astype(np.uint16)
    close_mmap(matrix)
    expected = len(FAMILIES) * len(LANGUAGES) * len(TRANSFORMS) * len(QPOINTS) * len(ROLES)
    if len(result) != expected:
        raise RuntimeError((len(result), expected))
    return result


def capture_selected(model, device, compiled: list[dict], candidate: dict, generated: dict,
                     qualified: set[str]) -> tuple[list[dict], Path]:
    selected = [row for row in compiled if f"{row['family']}|{row['language']}" in qualified
                and candidate[row["case_id"]]["correct"] and generated[row["case_id"]]["correct"]]
    field_path = out("C749-C753") / "raw/fresh_qpoint_relation_boundary_field.float16.npy"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16,
                                      shape=(len(selected), len(QPOINTS), len(ROLES), DIM))
    base = model.model; captured = []
    modules = [base.embed_tokens, *list(base.layers), base.norm]
    handles = [module.register_forward_hook(lambda _m, _a, output: captured.append(output[0] if isinstance(output, tuple) else output))
               for module in modules]
    index = []
    try:
        for row_i, item in enumerate(selected):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); positions = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            if len(captured) != 38:
                raise RuntimeError((item["case_id"], len(captured)))
            for qi, q in enumerate(QPOINTS):
                values = captured[q][0].float().cpu().numpy()
                for ri, role in enumerate(ROLES):
                    field[row_i, qi, ri] = values[item["role_positions"][role][-1]].astype(np.float16)
            index.append({"field_index": row_i, "case_id": item["case_id"], "family": item["family"],
                          "language": item["language"], "partition": item["partition"],
                          "unit": item["unit"], "cell_i": item["cell_i"], "dual_correct": True})
            if row_i % 64 == 0:
                print(f"[C749-C753] capture {row_i}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    field.flush(); close_mmap(field)
    write_rows(out("C749-C753") / "raw/field_index.jsonl", index)
    return index, field_path


def response_code(base: np.ndarray, changed: np.ndarray) -> np.ndarray:
    return parent.response_code(base, changed)


def phase2212() -> None:
    name = "C749-C753"
    if (out(name) / "analysis/final.json").exists():
        return
    compiled = read_rows(out("C745-C748") / "material/qwen_compiled.jsonl")
    model = None
    try:
        model, tokenizer, device, placement = scope.parent.previous.model_base().load_bf16("qwen3")
        quant = scope.parent.previous.model_base().quantization_audit(model)
        candidate_rows = behavior.batch_behavior(model, device, compiled)
        generation_rows = behavior.free_generate(model, tokenizer, device, compiled)
        candidate = {row["case_id"]: row for row in candidate_rows}
        generated = {row["case_id"]: row for row in generation_rows}
        write_rows(out(name) / "behavior/candidate.jsonl", candidate_rows)
        write_rows(out(name) / "behavior/free_generation.jsonl", generation_rows)
        slices = {}
        for family, language in itertools.product(FAMILIES, LANGUAGES):
            values = {}
            for part in ("confirmation", "lockbox"):
                subset = [row for row in compiled if row["family"] == family and row["language"] == language and row["partition"] == part]
                values[part] = {"rows": len(subset),
                    "candidate_accuracy": float(np.mean([candidate[row["case_id"]]["correct"] for row in subset])),
                    "generation_accuracy": float(np.mean([generated[row["case_id"]]["correct"] for row in subset])),
                    "dual_accuracy": float(np.mean([candidate[row["case_id"]]["correct"] and generated[row["case_id"]]["correct"] for row in subset]))}
            values["qualified"] = min(values[p][m] for p in values for m in ("candidate_accuracy", "generation_accuracy")) >= BEHAVIOR_GATE
            slices[f"{family}|{language}"] = values
        qualified = {key for key, value in slices.items() if value["qualified"]}
        index, field_path = capture_selected(model, device, compiled, candidate, generated, qualified)
    finally:
        scope.parent.previous.model_base().release_bf16(model); gc.collect()
    prototypes = load_parent_prototypes()
    field = np.load(field_path, mmap_mode="r")
    index_map = {(row["family"], row["language"], row["partition"], row["unit"], row["cell_i"]): row["field_index"] for row in index}
    metrics = {}; prospective = []
    for family, language, transform in itertools.product(FAMILIES, LANGUAGES, TRANSFORMS):
        label = f"{family}|{language}|{transform}"
        proto = np.stack([[prototypes[(label, q, role)] for role in ROLES] for q in QPOINTS])
        wrong_family = FAMILIES[(FAMILIES.index(family) + 1) % len(FAMILIES)]
        wrong_label = f"{wrong_family}|{language}|{transform}"
        wrong_proto = np.stack([[prototypes[(wrong_label, q, role)] for role in ROLES] for q in QPOINTS])
        panels = {}
        for part in ("confirmation", "lockbox"):
            units = range(0, 4) if part == "confirmation" else range(4, 8)
            unit_rows = []
            for unit in units:
                left = index_map.get((family, language, part, unit, 0)); right = index_map.get((family, language, part, unit, transform))
                if left is None or right is None:
                    continue
                code = response_code(np.asarray(field[left]), np.asarray(field[right]))
                specific = float(np.mean(code == proto)); wrong = float(np.mean(code == wrong_proto))
                shifted = float(np.mean(code == np.roll(proto, 257, axis=2)))
                gain = specific - max(wrong, shifted)
                unit_rows.append({"unit": unit, "specific": specific, "wrong_family": wrong, "shift257": shifted, "gain": gain})
            required = max(1, math.ceil(len(unit_rows) * 2 / 3))
            panels[part] = {"pairs": len(unit_rows), "units": unit_rows,
                            "mean_gain": float(np.mean([row["gain"] for row in unit_rows])) if unit_rows else 0.0,
                            "positive_units": sum(row["gain"] >= PASSPORT_GAIN_GATE for row in unit_rows),
                            "required_positive_units": required}
            panels[part]["passed"] = (len(unit_rows) >= 3 and panels[part]["positive_units"] >= required
                                               and panels[part]["mean_gain"] >= PASSPORT_GAIN_GATE)
        panels["prospective_passed"] = panels["confirmation"]["passed"] and panels["lockbox"]["passed"]
        metrics[label] = panels
        if panels["prospective_passed"]:
            prospective.append(label)
    close_mmap(field)
    save(out(name) / "analysis/fresh_passport_metrics.json", metrics)
    close(name, {
        "strict_interpretation": "Frozen parent passports were evaluated on lexically disjoint concepts without refitting. A pass means discrete all-coordinate transition predictability at six checkpoints and two roles, not causal necessity or a semantic module.",
        "executor_repair": "The first run completed behavior calculation but open_memmap found that the raw parent directory did not exist. The directory creation was added before field allocation; data, model, gates and analysis were unchanged.",
        "slice_results": slices, "qualified_slices": sorted(qualified), "captured_rows": len(index),
        "passport_metrics": metrics, "fresh_prospective_passport_groups": prospective,
        "fresh_prospective_count": len(prospective), "field_shape": [len(index), len(QPOINTS), len(ROLES), DIM],
        "field_sha256": file_hash(field_path), "placement": placement, "quantization": quant,
        "new_foundational_mathematics_gate": False,
    }, {
        "parent": final("C745-C748")["all_checks_passed"], "behavior_complete": len(candidate_rows) == len(compiled) == len(generation_rows),
        "some_qualified": bool(qualified), "same_row_capture": all(row["dual_correct"] for row in index),
        "frozen_prototypes": len(prototypes) == 432, "all_groups_reported": len(metrics) == 36,
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"], "finite": finite(metrics),
    }, "Authorize C754-C758 to test every fresh-qualified passport as a complete multi-checkpoint deletion/rescue object; nonpassing groups remain reported without stopping others.")


def decode_response(code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    code = np.asarray(code, np.uint16)
    return (code // 33).astype(np.uint8), (code % 33).astype(np.uint8)


def state_center(code: np.ndarray) -> np.ndarray:
    code = np.asarray(code, np.uint8)
    exponent = ((code.astype(np.int16) - 1) % 16) - 8
    sign = np.where(code == 0, 0.0, np.where(code <= 16, -1.0, 1.0))
    return (sign * np.exp2(exponent.astype(np.float32) + 0.5)).astype(np.float32)


def passport_arrays(prototypes: dict, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = np.stack([[prototypes[(label, q, role)] for role in ROLES] for q in QPOINTS])
    base_codes, target_codes = decode_response(codes)
    return base_codes, target_codes, base_codes != target_codes


@torch.inference_mode()
def run_intervention(model, tokenizer, item: dict, arrays: dict, mode: str, free: bool) -> dict:
    ids_key = "free_prompt_ids" if free else "prompt_ids"
    ids = torch.tensor([item[ids_key]], dtype=torch.long, device=next(model.parameters()).device)
    mask = torch.ones_like(ids); positions = torch.arange(ids.shape[1], device=ids.device)[None]
    base = model.model; handles = []
    module_by_q = {0: base.embed_tokens, 37: base.norm}
    module_by_q.update({q: base.layers[q - 1] for q in QPOINTS if q not in (0, 37)})
    if mode != "base":
        for qi, q in enumerate(QPOINTS):
            module = module_by_q[q]
            for ri, role in enumerate(ROLES):
                source_pos = int(item["role_positions"][role][-1])
                correct_mask = arrays["mask"][qi, ri]
                target_codes = arrays["target"][qi, ri]
                wrong_codes = arrays["wrong_target"][qi, ri]

                def patch(_module, _args, output, source_pos=source_pos, correct_mask=correct_mask,
                          target_codes=target_codes, wrong_codes=wrong_codes):
                    hidden = output[0] if isinstance(output, tuple) else output
                    if hidden.shape[1] <= source_pos:
                        return output
                    changed = hidden.clone(); current = hidden[0, source_pos].float()
                    if mode == "delete":
                        active = torch.tensor(correct_mask, dtype=torch.bool, device=current.device)
                        current[active] = 0.0
                    elif mode == "shift_delete":
                        active = torch.tensor(np.roll(correct_mask, 257).copy(), dtype=torch.bool, device=current.device)
                        current[active] = 0.0
                    else:
                        active = torch.tensor(correct_mask, dtype=torch.bool, device=current.device)
                        if mode == "correct_rescue":
                            centers = state_center(target_codes)
                        elif mode == "wrong_rescue":
                            centers = state_center(wrong_codes)
                        else:
                            centers = state_center(np.roll(target_codes, 257).copy())
                        center_tensor = torch.tensor(centers, dtype=torch.float32, device=current.device)
                        current[active] = center_tensor[active]
                    changed[0, source_pos] = current.to(hidden.dtype)
                    return (changed, *output[1:]) if isinstance(output, tuple) else changed

                handles.append(module.register_forward_hook(patch))
    try:
        if free:
            pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5, do_sample=False,
                                       pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(generated[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
            parsed = behavior.parse_binary(text, item["language"])
            return {"text": text, "parsed": parsed, "correct": parsed == item["correct_answer"]}
        result = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
        first = [int(candidate[0]) for candidate in item["candidate_ids"]]
        scores = result.logits[0, -1, first].float().cpu().numpy(); gold = int(item["gold_position"])
        return {"margin": float(scores[gold] - scores[1 - gold]), "correct": bool(scores[gold] > scores[1 - gold])}
    finally:
        for handle in handles:
            handle.remove()


def phase2213() -> None:
    name = "C754-C758"
    if (out(name) / "analysis/final.json").exists():
        return
    prospective = final("C749-C753")["fresh_prospective_passport_groups"]
    compiled = read_rows(out("C745-C748") / "material/qwen_compiled.jsonl")
    candidate = {row["case_id"]: row for row in read_rows(out("C749-C753") / "behavior/candidate.jsonl")}
    generated = {row["case_id"]: row for row in read_rows(out("C749-C753") / "behavior/free_generation.jsonl")}
    prototypes = load_parent_prototypes(); cases = []
    for label in prospective:
        family, language, transform_text = label.split("|"); transform = int(transform_text)
        options = [row for row in compiled if row["family"] == family and row["language"] == language
                   and row["partition"] == "lockbox" and row["cell_i"] == transform
                   and candidate[row["case_id"]]["correct"] and generated[row["case_id"]]["correct"]]
        if options:
            cases.append((label, sorted(options, key=lambda row: row["unit"])[0]))
    model = None; results = {}
    try:
        model, tokenizer, _device, placement = scope.parent.previous.model_base().load_bf16("qwen3")
        quant = scope.parent.previous.model_base().quantization_audit(model)
        for case_i, (label, item) in enumerate(cases):
            family, language, transform_text = label.split("|"); transform = int(transform_text)
            wrong_family = FAMILIES[(FAMILIES.index(family) + 1) % len(FAMILIES)]
            _, target, active = passport_arrays(prototypes, label)
            _, wrong_target, _ = passport_arrays(prototypes, f"{wrong_family}|{language}|{transform}")
            arrays = {"target": target, "wrong_target": wrong_target, "mask": active}
            modes = ("base", "delete", "shift_delete", "correct_rescue", "wrong_rescue", "shift_rescue")
            candidate_modes = {mode: run_intervention(model, tokenizer, item, arrays, mode, False) for mode in modes}
            generation_modes = {mode: run_intervention(model, tokenizer, item, arrays, mode, True) for mode in modes}
            base_margin = candidate_modes["base"]["margin"]; deleted = candidate_modes["delete"]["margin"]
            necessity = (base_margin - deleted) - max(base_margin - candidate_modes["shift_delete"]["margin"], 0.0)
            rescue_gain = (candidate_modes["correct_rescue"]["margin"] - deleted) - max(
                candidate_modes["wrong_rescue"]["margin"] - deleted,
                candidate_modes["shift_rescue"]["margin"] - deleted, 0.0)
            passed = (necessity >= CAUSAL_GAIN_GATE and rescue_gain >= CAUSAL_GAIN_GATE
                      and generation_modes["base"]["correct"] and generation_modes["correct_rescue"]["correct"])
            results[label] = {"case_id": item["case_id"], "mask_coordinates": int(active.sum()),
                              "mask_fraction": float(active.mean()), "candidate": candidate_modes,
                              "generation": generation_modes, "necessity_specific_gain": necessity,
                              "rescue_specific_gain": rescue_gain, "passed": passed}
            print(f"[C754-C758] {case_i + 1}/{len(cases)} {label} pass={passed}", flush=True)
    finally:
        scope.parent.previous.model_base().release_bf16(model); gc.collect()
    passed_groups = [label for label, value in results.items() if value["passed"]]
    save(out(name) / "analysis/deletion_rescue.json", results)
    close(name, {
        "strict_interpretation": "The intervention treats the frozen six-checkpoint, two-role passport as one distributed object and edits every coordinate whose frozen state code changes. Correct-code rescue must beat shifted-mask and wrong-family controls in both candidate margin accounting and free generation. A failure rejects this passport as a sufficient causal call under the registered discretization; it does not reject distributed language encoding.",
        "executor_repair": "The first intervention attempt reached the first candidate forward but logits still required gradients before NumPy conversion. run_intervention was marked inference_mode; masks, state centers, cases, modes and gates were unchanged, and no partial result had been adjudicated.",
        "eligible_groups": prospective, "tested_groups": list(results), "deletion_rescue": results,
        "causal_passed_groups": passed_groups, "causal_passed_count": len(passed_groups),
        "placement": placement, "quantization": quant, "new_foundational_mathematics_gate": False,
    }, {
        "parent": final("C749-C753")["all_checks_passed"], "all_eligible_tested": len(results) == len(prospective),
        "modes_complete": all(set(value["candidate"]) == {"base", "delete", "shift_delete", "correct_rescue", "wrong_rescue", "shift_rescue"}
                              and set(value["generation"]) == {"base", "delete", "shift_delete", "correct_rescue", "wrong_rescue", "shift_rescue"}
                              for value in results.values()),
        "all_coordinates_not_topk": all(value["mask_coordinates"] > 0 for value in results.values()),
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"], "finite": finite(results),
    }, "Authorize C759-C760 to preserve the fresh exact-coordinate atlas, clean the duplicate raw field, and decide whether this exact passport object or only the broader atlas goal continues.")


def phase2214() -> None:
    name = "C759-C760"
    if (out(name) / "analysis/final.json").exists():
        return
    field_path = out("C749-C753") / "raw/fresh_qpoint_relation_boundary_field.float16.npy"
    field = np.load(field_path, mmap_mode="r")
    index = read_rows(out("C749-C753") / "raw/field_index.jsonl")
    matrix = np.asarray(field).reshape(-1, DIM).astype(np.float16)
    rows = []
    for item in index:
        for q in QPOINTS:
            for role in ROLES:
                rows.append({"kind": "fresh_exact_activation", "case_id": item["case_id"],
                             "family": item["family"], "language": item["language"],
                             "partition": item["partition"], "unit": item["unit"],
                             "cell_i": item["cell_i"], "checkpoint": q, "role": role})
    VISUAL_BINARY.parent.mkdir(parents=True, exist_ok=True); np.save(VISUAL_BINARY, matrix)
    close_mmap(field)
    payload = {"schema": "ai2050.fresh-response-passport-causal-atlas.v1", "phase": 2214,
               "campaign": "C745-C760", "model": "Qwen3-4B BF16", "coordinate_count": DIM,
               "rows": rows, "binary": str(VISUAL_BINARY.relative_to(ROOT)).replace("\\", "/"),
               "binary_shape": list(matrix.shape), "binary_dtype": "float16",
               "fresh_passport_metrics": final("C749-C753")["passport_metrics"],
               "deletion_rescue": final("C754-C758")["deletion_rescue"],
               "claim_boundary": "Exact activation-state coordinates plus frozen-passport prediction and complete-mask intervention results; no parameter tensor is displayed."}
    save(VISUAL, payload)
    catalog = load(CATALOG) if CATALOG.exists() else {"schema": "language-encoding-catalog.v1", "datasets": []}
    entry = {"id": "c760-fresh-response-passport-causal-atlas", "title": "Fresh response passport causal atlas",
             "phase": 2214, "type": "exact-coordinate-heatmap", "json": str(VISUAL.relative_to(ROOT)).replace("\\", "/"),
             "binary": str(VISUAL_BINARY.relative_to(ROOT)).replace("\\", "/"), "shape": list(matrix.shape)}
    catalog["datasets"] = [row for row in catalog.get("datasets", []) if row.get("id") != entry["id"]] + [entry]
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat(); save(CATALOG, catalog)
    cleanup = {"path": str(field_path.relative_to(ROOT)), "sha256": file_hash(field_path), "deleted": False}
    field_path.unlink(); cleanup["deleted"] = True
    save(out(name) / "audit/hash_then_cleanup.json", cleanup)
    passport = final("C749-C753")["fresh_prospective_passport_groups"]
    causal = final("C754-C758")["causal_passed_groups"]
    same_goal = bool(causal)
    decision = ("Automatically continue the same causal passport object on additional doses and open natural panels."
                if same_goal else
                "The frozen passport remains an observational atlas candidate, but this exact discretized multi-checkpoint object did not earn further automatic causal tuning; continue the broader language-family atlas with richer state-conditioned objects.")
    close(name, {
        "strict_interpretation": "Fresh lexical replication and causal intervention are separate evidence levels. A reproducible discrete passport can be a useful atlas coordinate even when complete-mask deletion/rescue does not identify a sufficient semantic mechanism.",
        "fresh_prospective_passport_groups": passport, "fresh_prospective_count": len(passport),
        "causal_passed_groups": causal, "causal_passed_count": len(causal),
        "visual": {"json": str(VISUAL.relative_to(ROOT)), "binary": str(VISUAL_BINARY.relative_to(ROOT)),
                   "shape": list(matrix.shape), "sha256": file_hash(VISUAL_BINARY)},
        "cleanup": cleanup, "important_answer_reached": True, "next_stage_same_goal": same_goal,
        "automatic_continuation_decision": decision, "human_review": "NA_not_run",
        "theory_update": "Base-state-conditioned response passports remain the best empirical coordinate object. Causal response equivalence is not granted unless deletion/rescue and future outputs survive matched controls.",
        "new_foundational_mathematics_gate": False,
    }, {
        "parents": final("C749-C753")["all_checks_passed"] and final("C754-C758")["all_checks_passed"],
        "visual": VISUAL.exists() and VISUAL_BINARY.exists(), "row_index": len(rows) == matrix.shape[0],
        "raw_cleaned": not field_path.exists(), "finite": finite([passport, causal, matrix.shape]),
    }, decision)


def run_all() -> None:
    freeze(); rows = material(); phase2211(rows); phase2212(); phase2213(); phase2214()


if __name__ == "__main__":
    run_all()
