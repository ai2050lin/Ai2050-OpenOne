#!/usr/bin/env python3
"""Audit Phase 2388-2395 claims and freeze an event-aligned conditional-operation contract."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2396_c21361_c21680_conditional_operation_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2396
CAMPAIGN = "C21361-C21680"
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\ae76a631-5f0f-441d-aba3-ca6eeb6206a0\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\9d919240-8984-4ccc-9c2d-5b8d70dcdb91\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\2d663e84-4bed-47a5-9474-4daf322bb8de\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\ee5a5696-8c9e-44aa-932d-f5f46a506c88\pasted-text.txt"),
)
FAMILIES = (
    "preference", "ownership", "spatial", "temporal",
    "causal", "comparison", "role_binding", "taxonomy",
)
COMPOSITION_FAMILIES = ("spatial", "temporal", "comparison", "taxonomy")
LANGUAGES = ("en", "zh")
PAIR_EN = (
    ("Mira", "Nolan"), ("cedar", "willow"), ("lantern", "compass"), ("falcon", "heron"),
    ("violin", "piano"), ("harbor", "bridge"), ("copper", "silver"), ("comet", "asteroid"),
)
PAIR_ZH = (
    ("米拉", "诺兰"), ("雪松", "柳树"), ("灯笼", "罗盘"), ("隼", "鹭"),
    ("小提琴", "钢琴"), ("港口", "桥梁"), ("铜", "银"), ("彗星", "小行星"),
)
TRIPLE_EN = (
    ("Mira", "Nolan", "Priya"), ("cedar", "willow", "birch"),
    ("lantern", "compass", "map"), ("falcon", "heron", "sparrow"),
    ("violin", "piano", "flute"), ("harbor", "bridge", "station"),
    ("copper", "silver", "gold"), ("comet", "asteroid", "planet"),
)
TRIPLE_ZH = (
    ("米拉", "诺兰", "普里娅"), ("雪松", "柳树", "桦树"),
    ("灯笼", "罗盘", "地图"), ("隼", "鹭", "麻雀"),
    ("小提琴", "钢琴", "长笛"), ("港口", "桥梁", "车站"),
    ("铜", "银", "金"), ("彗星", "小行星", "行星"),
)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def partition(unit: int) -> str:
    return "discovery" if unit < 4 else "confirmation" if unit < 6 else "fresh_unit_lockbox"


def relation_template(family: str, language: str, surface: str) -> tuple[str, str]:
    """Return a format string and the exact relation phrase whose span will be anchored."""
    if language == "en":
        canonical = {
            "preference": ("{s} prefers {t}.", "prefers"),
            "ownership": ("{s} owns {t}.", "owns"),
            "spatial": ("{s} is north of {t}.", "is north of"),
            "temporal": ("{s} happened before {t}.", "happened before"),
            "causal": ("{s} caused {t}.", "caused"),
            "comparison": ("{s} is taller than {t}.", "is taller than"),
            "role_binding": ("{s} thanked {t}.", "thanked"),
            "taxonomy": ("{s} is a type of {t}.", "is a type of"),
        }
        paraphrase = {
            "preference": ("{t} is preferred by {s}.", "is preferred by"),
            "ownership": ("{t} belongs to {s}.", "belongs to"),
            "spatial": ("{t} lies south of {s}.", "lies south of"),
            "temporal": ("{t} happened after {s}.", "happened after"),
            "causal": ("{t} resulted from {s}.", "resulted from"),
            "comparison": ("{t} is shorter than {s}.", "is shorter than"),
            "role_binding": ("{t} received thanks from {s}.", "received thanks from"),
            "taxonomy": ("{t} is the category containing {s}.", "is the category containing"),
        }
    else:
        canonical = {
            "preference": ("{s}更喜欢{t}。", "更喜欢"),
            "ownership": ("{s}拥有{t}。", "拥有"),
            "spatial": ("{s}位于{t}以北。", "位于"),
            "temporal": ("{s}发生在{t}之前。", "发生在"),
            "causal": ("{s}导致了{t}。", "导致了"),
            "comparison": ("{s}比{t}更高。", "比"),
            "role_binding": ("{s}感谢了{t}。", "感谢了"),
            "taxonomy": ("{s}是{t}的一种。", "是"),
        }
        paraphrase = {
            "preference": ("相比之下，{t}更受{s}偏爱。", "更受"),
            "ownership": ("{t}归{s}所有。", "归"),
            "spatial": ("{t}坐落在{s}以南。", "坐落在"),
            "temporal": ("{t}是在{s}之后发生的。", "之后发生"),
            "causal": ("{t}是由{s}引起的。", "是由"),
            "comparison": ("{t}没有{s}高。", "没有"),
            "role_binding": ("{t}收到了{s}的感谢。", "收到了"),
            "taxonomy": ("{t}这个类别包含{s}。", "这个类别包含"),
        }
    return (canonical if surface == "canonical" else paraphrase)[family]


def role_query(family: str, language: str, role: str) -> str:
    en = {
        "preference": ("Which entity has the preference?", "Which entity is preferred?"),
        "ownership": ("Which entity is the owner?", "Which entity is owned?"),
        "spatial": ("Which entity is farther north?", "Which entity is farther south?"),
        "temporal": ("Which entity happened earlier?", "Which entity happened later?"),
        "causal": ("Which entity is the cause?", "Which entity is the effect?"),
        "comparison": ("Which entity is taller?", "Which entity is shorter?"),
        "role_binding": ("Which entity gave thanks?", "Which entity received thanks?"),
        "taxonomy": ("Which entity is the subtype?", "Which entity is its category?"),
    }
    zh = {
        "preference": ("哪个实体具有偏好？", "哪个实体受到偏爱？"),
        "ownership": ("哪个实体是拥有者？", "哪个实体是被拥有者？"),
        "spatial": ("哪个实体更靠北？", "哪个实体更靠南？"),
        "temporal": ("哪个实体发生得更早？", "哪个实体发生得更晚？"),
        "causal": ("哪个实体是原因？", "哪个实体是结果？"),
        "comparison": ("哪个实体更高？", "哪个实体更矮？"),
        "role_binding": ("哪个实体表达了感谢？", "哪个实体收到了感谢？"),
        "taxonomy": ("哪个实体是子类？", "哪个实体是它的类别？"),
    }
    return (en if language == "en" else zh)[family][0 if role == "source" else 1]


def render_fact(family: str, language: str, surface: str, source: str, target: str) -> tuple[str, dict]:
    template, phrase = relation_template(family, language, surface)
    fact = template.format(s=source, t=target)
    spans = {}
    for key, text in (("source", source), ("target", target), ("relation", phrase)):
        start = fact.index(text)
        spans[key] = [start, start + len(text)]
    return fact, spans


def prompt_with_events(language: str, facts: list[tuple[str, dict]], query: str,
                       candidates: list[str], extra_events: list[dict] | None = None) -> tuple[str, list[dict]]:
    intro = "Read the local record and answer with exactly one candidate.\n" if language == "en" else "阅读局部记录，并且只用一个候选项作答。\n"
    fact_label = "Record: " if language == "en" else "记录："
    query_label = "Question: " if language == "en" else "问题："
    candidates_label = "Candidates:\n" if language == "en" else "候选项：\n"
    answer_label = "Answer:" if language == "en" else "答案："
    parts = [intro]
    events: list[dict] = []
    for index, (fact, spans) in enumerate(facts, start=1):
        parts.append(fact_label)
        base = sum(len(part) for part in parts)
        parts.append(fact)
        for key in ("source", "relation", "target"):
            start, end = spans[key]
            events.append({"event": f"fact{index}_{key}", "char_start": base + start, "char_end": base + end})
        events.append({"event": f"fact{index}_end", "char_start": base, "char_end": base + len(fact)})
        parts.append("\n")
    parts.extend((query_label, query, "\n", candidates_label))
    query_end = sum(len(part) for part in parts[:-2])
    events.append({"event": "query_end", "char_start": query_end - len(query), "char_end": query_end})
    for index, candidate in enumerate(candidates, start=1):
        line = f"- {candidate}\n"
        start = sum(len(part) for part in parts) + 2
        parts.append(line)
        events.append({"event": f"candidate{index}_end", "char_start": start, "char_end": start + len(candidate)})
    parts.append(answer_label)
    prompt = "".join(parts)
    events.append({"event": "answer_boundary", "char_start": len(prompt), "char_end": len(prompt)})
    if extra_events:
        events.extend(extra_events)
    assert all(0 <= event["char_start"] <= event["char_end"] <= len(prompt) for event in events)
    return prompt, events


def compile_selection() -> list[dict]:
    rows: list[dict] = []
    for family in FAMILIES:
        for unit in range(8):
            for language in LANGUAGES:
                entity_a, entity_b = (PAIR_EN if language == "en" else PAIR_ZH)[unit]
                for surface in ("canonical", "paraphrase"):
                    for direction in (0, 1):
                        source, target = (entity_a, entity_b) if direction == 0 else (entity_b, entity_a)
                        fact, spans = render_fact(family, language, surface, source, target)
                        for query_role in ("source", "target"):
                            answer = source if query_role == "source" else target
                            foil = target if query_role == "source" else source
                            query = role_query(family, language, query_role)
                            for candidate_order in (0, 1):
                                candidates = [answer, foil] if candidate_order == 0 else [foil, answer]
                                prompt, events = prompt_with_events(language, [(fact, spans)], query, candidates)
                                rows.append({
                                    "case_id": f"sel-{family}-u{unit}-{language}-{surface}-d{direction}-{query_role}-o{candidate_order}",
                                    "task": "selection", "family": family, "unit": unit, "language": language,
                                    "surface": surface, "direction": direction, "query_role": query_role,
                                    "candidate_order": candidate_order, "target_candidate_slot": candidates.index(answer),
                                    "partition": partition(unit), "source": source, "target": target,
                                    "fact": fact, "query": query, "candidates": candidates,
                                    "answer": answer, "foil": foil, "prompt": prompt, "events": events,
                                })
    return rows


def composition_query(family: str, language: str, source: str, steps: int) -> str:
    rel_en = {"spatial": "is north of", "temporal": "happened before", "comparison": "is taller than", "taxonomy": "is a type of"}
    rel_zh = {"spatial": "位于北侧", "temporal": "发生在前", "comparison": "比……更高", "taxonomy": "是……的一种"}
    if language == "en":
        return f"Starting from {source}, follow the relation '{rel_en[family]}' exactly {steps} time{'s' if steps > 1 else ''}. Which entity do you reach?"
    return f"从{source}开始，沿着“{rel_zh[family]}”关系恰好走{steps}步，会到达哪个实体？"


def compile_composition() -> list[dict]:
    rows: list[dict] = []
    for family in COMPOSITION_FAMILIES:
        for unit in range(8):
            for language in LANGUAGES:
                a, b, c = (TRIPLE_EN if language == "en" else TRIPLE_ZH)[unit]
                for surface in ("canonical", "paraphrase"):
                    for direction in (0, 1):
                        source, middle, endpoint = (a, b, c) if direction == 0 else (c, b, a)
                        fact1 = render_fact(family, language, surface, source, middle)
                        fact2 = render_fact(family, language, surface, middle, endpoint)
                        for steps in (1, 2):
                            answer = middle if steps == 1 else endpoint
                            foil = endpoint if steps == 1 else middle
                            query = composition_query(family, language, source, steps)
                            for candidate_order in (0, 1):
                                candidates = [answer, foil] if candidate_order == 0 else [foil, answer]
                                prompt, events = prompt_with_events(language, [fact1, fact2], query, candidates)
                                split = partition(unit)
                                if steps == 2:
                                    split = {"discovery": "exploratory_composition", "confirmation": "composition_confirmation",
                                             "fresh_unit_lockbox": "fresh_composition_lockbox"}[split]
                                rows.append({
                                    "case_id": f"cmp-{family}-u{unit}-{language}-{surface}-d{direction}-s{steps}-o{candidate_order}",
                                    "task": "composition", "family": family, "unit": unit, "language": language,
                                    "surface": surface, "direction": direction, "steps": steps,
                                    "candidate_order": candidate_order, "target_candidate_slot": candidates.index(answer),
                                    "partition": split, "source": source, "middle": middle, "endpoint": endpoint,
                                    "facts": [fact1[0], fact2[0]], "query": query, "candidates": candidates,
                                    "answer": answer, "foil": foil, "prompt": prompt, "events": events,
                                })
    return rows


def audit_material(selection: list[dict], composition: list[dict]) -> dict:
    all_rows = selection + composition
    return {
        "rows": len(all_rows), "selection_rows": len(selection), "composition_rows": len(composition),
        "selection_families": Counter(row["family"] for row in selection),
        "composition_families": Counter(row["family"] for row in composition),
        "languages": Counter(row["language"] for row in all_rows),
        "surfaces": Counter(row["surface"] for row in all_rows),
        "directions": Counter(row["direction"] for row in all_rows),
        "target_slots": Counter(row["target_candidate_slot"] for row in all_rows),
        "selection_query_roles": Counter(row["query_role"] for row in selection),
        "composition_steps": Counter(row["steps"] for row in composition),
        "partitions": Counter(row["partition"] for row in all_rows),
        "unique_case_ids": len({row["case_id"] for row in all_rows}) == len(all_rows),
        "unique_prompts": len({row["prompt"] for row in all_rows}) == len(all_rows),
        "event_bounds_valid": all(all(0 <= e["char_start"] <= e["char_end"] <= len(row["prompt"]) for e in row["events"]) for row in all_rows),
        "selection_event_count": sorted({len(row["events"]) for row in selection}),
        "composition_event_count": sorted({len(row["events"]) for row in composition}),
        "counterbalance_unit": "family x unit x language x surface x direction x query-role/steps x candidate-order",
    }


def evidence_audit() -> dict:
    return {
        "retained": [
            "The most useful next object is a sample-conditioned, event-aligned layer update rather than another last-prompt-token class readout.",
            "A proposed mechanism must connect external language operation, coordinate cooperation, cross-layer propagation, reuse/differentiation, composition, and next-token probability.",
            "Phase 2388-2395 rejected the tested global linear relation direction on those materials; it did not close a language mechanism.",
            "Full physical coordinates and low-magnitude coordinates remain primary; sparse subsets are diagnostics only.",
            "Behavior, counterbalancing, new-unit lockboxes, and cross-surface controls must be designed before inspecting HiddenState results.",
        ],
        "corrected": [
            "A condition-coordinate gear is a hypothesis/metaphor until it predicts local updates and output behavior on held-out units and expressions.",
            "The prior negative relation-vector result does not prove that no abstract, global, nonlinear, or component-level relation operator exists.",
            "A single prompt-boundary patch cannot establish that the observed field is epiphenomenal or unused elsewhere in the sequence.",
            "Attention key/value binding, MLP key-value retrieval, and sparse-autoencoder features were not measured in Phase 2388-2395 and cannot be called the true mechanism.",
            "SAE is neither the only key nor a primary method here; an arbitrary 30k dictionary may compress away distributed low-value structure.",
            "Logit lens localizes output-readout evolution but is not by itself causal attribution.",
            "Difference fields and sparse/group penalties may be bounded competitors, not the core representation or proof of coordinate gears.",
            "Old scripts and negative results must be preserved rather than destroyed.",
            "Phase 2388 counts must distinguish 768 independent sentence rows from 384 paired selection prompts.",
        ],
        "evidence_boundary": "Current evidence supports condition-dependent distributed readout fields, not a demonstrated internal gear, compiler, or new mathematics.",
    }


def mechanism_contract() -> dict:
    return {
        "primary_object": "U[row,event,q,j] = H[row,event,q+1,j] - H[row,event,q,j], with every physical coordinate retained",
        "events": ["fact source", "fact relation phrase", "fact target", "fact end", "query end", "each candidate end", "answer boundary"],
        "operator_competition": [
            "layer/event constant update", "same-coordinate shared affine update",
            "operation-conditioned coordinate offset", "operation-conditioned same-coordinate affine update",
            "sample-conditioned diagonal factor update", "bounded nonlinear upper bound only after simple competitors",
        ],
        "locks": ["new unit", "canonical-to-paraphrase", "English-to-Chinese", "counterfactual direction", "candidate slot", "one-step-to-two-step composition"],
        "claim_gates": {
            "task_supported": "teacher-forced target-over-foil and autonomous exact behavior reported by every family/condition; failure does not stop field mapping",
            "local_update": "held-out update prediction must beat a layer/event constant baseline and coordinate-permuted controls",
            "reuse": "a frozen coordinate passport must recur across new units or surfaces without selecting coordinates on lockbox",
            "composition": "one-step structure must predict two-step local update and output margin beyond matched controls",
            "compilation": "predicted local update must account for held-out first-divergence target-vs-foil logit contribution",
            "gear": "the term remains provisional unless update, reuse/composition, output link, and temporally aligned causal controls jointly pass",
        },
        "prohibited_shortcuts": ["Top-K as primary result", "single relation sentence", "same prompt boundary only", "SAE-only discovery", "causal failure as a stop rule", "destroying historical scripts"],
    }


def mega_plan() -> list[dict]:
    return [
        {"phase": 2396, "campaign": "C21361-C21680", "task": "audit, material, event anchors, mechanism contract"},
        {"phase": 2397, "campaign": "C21681-C22000", "task": "four-model behavior and token-anchor calibration; freeze discovery/replication roles"},
        {"phase": 2398, "campaign": "C22001-C22320", "task": "Qwen4B full-coordinate event x layer capture plus all-token references"},
        {"phase": 2399, "campaign": "C22321-C22640", "task": "local update/operator atlas with untouched lockboxes and negative controls"},
        {"phase": 2400, "campaign": "C22641-C22960", "task": "coordinate passports, cooperation, reuse, differentiation, and layer propagation"},
        {"phase": 2401, "campaign": "C22961-C23280", "task": "composition and local-update-to-next-token compilation bridge"},
        {"phase": 2402, "campaign": "C23281-C23600", "task": "Qwen14B frozen large-sample replication"},
        {"phase": 2403, "campaign": "C23601-C23920", "task": "GLM4 then DS7B frozen cross-architecture replication"},
        {"phase": 2404, "campaign": "C23921-C24240", "task": "publish parameter-level heatmaps, verify client, clean raw duplicates, audit successor"},
    ]


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 条件语言操作族证据审查、事件材料与机制合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 审查本轮四份分析及Phase2388–2395原始记录，先区分“实测、有限推断、机制假说”。不把条件坐标齿轮、SAE稀疏特征、Attention K/V绑定或MLP键值检索预设为事实；冻结新的外部语言操作→内部局部更新→坐标协同→跨层复用/分化→组合→下一token概率合同。材料包含8个操作族、8个新unit、中英双语、canonical/paraphrase、正反方向、source/target查询和候选顺序的2048条选择任务；另含空间/时序/比较/分类4族、1/2步组合的1024条任务。所有实体、答案槽、方向、表面与查询角色反平衡，unit 0–3用于发现、4–5用于确认、6–7只作fresh lockbox；每条prompt标注事实源、关系短语、事实目标、事实结束、查询结束、候选结束和答案边界的字符事件。

$$U_{{r,e,q,j}}=H_{{r,e,q+1,j}}-H_{{r,e,q,j}},\qquad j=1,\ldots,d,$$

$$\mathcal G\ \text{{只能在}}\quad
\mathrm{{Predict}}(U_{{heldout}})>\mathrm{{baseline}},\quad
\mathrm{{Reuse}},\quad\mathrm{{Compose}},\quad
\mathrm{{Explain}}(\Delta\log p)\quad\text{{共同成立后升级为机制候选。}}$$

**结果汇总。** 附件审查 `{json.dumps(result['evidence_audit'], ensure_ascii=False)}`；材料审计 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；机制合同 `{json.dumps(result['mechanism_contract'], ensure_ascii=False)}`；完整大方案 `{json.dumps(result['mega_plan'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2396_c21361_c21680_conditional_operation_contract.py`；全部3072条材料、事件锚点、合同和final位于 `tests/glm5/result/phase2396_c21361_c21680_conditional_operation_contract`。未修改其他Markdown。

**理论进展。** 将“齿轮”从命名性隐喻改造成可证伪对象：其最低要求不是某层能分类，而是相同物理坐标上的样本条件局部更新能跨新unit/表达预测，坐标护照能显示复用与分化，组合关系能从一步推广到两步，并且更新能解释正确候选相对foil的第一分歧token概率贡献。全坐标场是主对象，坐标子集和非线性模型只作诊断或上界。

**问题硬伤与结论。** 字符事件必须在各模型chat模板下重新标定为token事件；局部差分仍可能混入残差流常规变换，不能自动等同于语义操作。人工微世界包含反事实关系，模型先验可能降低行为；因此行为失败只限定任务有效性，不停止观察通过的族。Phase2388–2395只否决已测的全局线性关系方向和边界干预，不足以宣称不存在抽象关系算子，更不能断言局部非线性稀疏特征就是真机制。本Phase冻结Phase2397–2404一次性大方案，下一Phase先完成四模型行为与事件token锚点校准。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        cached = json.loads(final_path.read_text(encoding="utf-8"))
        if cached.get("all_checks_passed"):
            append_memo(cached)
            print(json.dumps(cached, ensure_ascii=False, indent=2))
            return
    selection = compile_selection()
    composition = compile_composition()
    write_rows(OUT / "material/selection_operation_rows.jsonl", selection)
    write_rows(OUT / "material/composition_operation_rows.jsonl", composition)
    write_rows(OUT / "material/all_operation_rows.jsonl", selection + composition)
    material = audit_material(selection, composition)
    checks = {
        "attachments_present": all(path.exists() for path in ATTACHMENTS),
        "phase_continuity": "## Phase 2395:" in MEMO.read_text(encoding="utf-8") and "## Phase 2396:" not in MEMO.read_text(encoding="utf-8"),
        "selection_rows": material["selection_rows"] == 2048,
        "composition_rows": material["composition_rows"] == 1024,
        "factor_balance": all(count == 1536 for count in material["languages"].values()) and all(count == 1536 for count in material["surfaces"].values()) and all(count == 1536 for count in material["directions"].values()) and all(count == 1536 for count in material["target_slots"].values()),
        "unique": material["unique_case_ids"] and material["unique_prompts"],
        "event_bounds": material["event_bounds_valid"],
        "full_coordinate_contract": "Top-K as primary result" in mechanism_contract()["prohibited_shortcuts"],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "attachments": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in ATTACHMENTS],
        "evidence_audit": evidence_audit(), "material_audit": material,
        "mechanism_contract": mechanism_contract(), "mega_plan": mega_plan(),
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/evidence_audit.json", result["evidence_audit"])
    save(OUT / "analysis/mechanism_contract.json", result["mechanism_contract"])
    save(OUT / "analysis/mega_plan.json", result["mega_plan"])
    save(final_path, result)
    append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
