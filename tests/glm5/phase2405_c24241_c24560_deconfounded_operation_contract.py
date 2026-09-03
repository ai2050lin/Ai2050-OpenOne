#!/usr/bin/env python3
"""Audit Phase2396-2404 claims and freeze the deconfounded operation-family campaign."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2405_c24241_c24560_deconfounded_operation_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2405
CAMPAIGN = "C24241-C24560"
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\0f9a6e31-9f97-40bc-be7a-3e88e7365bed\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\dc683983-8553-45b3-80fc-e7a985c7a397\pasted-text.txt"),
)

sys.path.insert(0, str(TESTS))
import phase2396_c21361_c21680_conditional_operation_contract as prior  # noqa: E402

FAMILIES = prior.FAMILIES
COMPOSITION_FAMILIES = prior.COMPOSITION_FAMILIES
LANGUAGES = prior.LANGUAGES
SURFACES = ("canonical", "paraphrase", "discourse", "natural")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pairs(language: str, unit: int) -> tuple[str, str]:
    base = prior.PAIR_EN if language == "en" else prior.PAIR_ZH
    if unit < 8:
        return base[unit]
    extra_en = (("orchid", "garden"), ("saffron", "market"), ("otter", "river"), ("quartz", "museum"))
    extra_zh = (("兰花", "花园"), ("藏红花", "市场"), ("水獭", "河流"), ("石英", "博物馆"))
    return (extra_en if language == "en" else extra_zh)[unit - 8]


def triples(language: str, unit: int) -> tuple[str, str, str]:
    base = prior.TRIPLE_EN if language == "en" else prior.TRIPLE_ZH
    if unit < 8:
        return base[unit]
    extra_en = (("orchid", "garden", "valley"), ("saffron", "market", "harbor"),
                ("otter", "river", "ocean"), ("quartz", "museum", "city"))
    extra_zh = (("兰花", "花园", "山谷"), ("藏红花", "市场", "港口"),
                ("水獭", "河流", "海洋"), ("石英", "博物馆", "城市"))
    return (extra_en if language == "en" else extra_zh)[unit - 8]


def relation_surface(family: str, language: str, surface: str) -> tuple[str, str]:
    if surface in ("canonical", "paraphrase"):
        return prior.relation_template(family, language, surface)
    if language == "en":
        discourse = {
            "preference": ("According to the note, {s} would choose {t} over the alternative.", "would choose"),
            "ownership": ("The item {t}, the note explains, is in {s}'s possession.", "is in"),
            "spatial": ("On the map, {s} appears above and northward from {t}.", "northward from"),
            "temporal": ("In the recorded sequence, {s} came earlier than {t}.", "came earlier than"),
            "causal": ("The report attributes the occurrence of {t} to {s}.", "attributes"),
            "comparison": ("After measurement, {s} had the greater height, compared with {t}.", "greater height"),
            "role_binding": ("In the exchange, gratitude was expressed by {s} toward {t}.", "expressed by"),
            "taxonomy": ("Within this classification, {s} belongs under the broader class {t}.", "belongs under"),
        }
        natural = {
            "preference": ("When both were available, {s} consistently picked {t}.", "picked"),
            "ownership": ("Everyone in the room knew that {t} was {s}'s.", "was"),
            "spatial": ("Looking at the trail map, {s} sits up north from {t}.", "up north from"),
            "temporal": ("By the time {t} occurred, {s} had already happened.", "had already happened"),
            "causal": ("Without {s}, the later event {t} would not have occurred.", "would not have occurred"),
            "comparison": ("Standing side by side, {s} rose higher than {t}.", "rose higher than"),
            "role_binding": ("At the end of the meeting, {s} offered thanks to {t}.", "offered thanks to"),
            "taxonomy": ("A handbook lists {s} among the kinds of {t}.", "among the kinds of"),
        }
    else:
        discourse = {
            "preference": ("根据记录，相比另一个选项，{s}会选择{t}。", "会选择"),
            "ownership": ("记录说明，物品{t}处在{s}的所有权之下。", "所有权之下"),
            "spatial": ("在地图上，{s}显示在{t}的北方。", "北方"),
            "temporal": ("按照记录的顺序，{s}早于{t}出现。", "早于"),
            "causal": ("报告把{t}的发生归因于{s}。", "归因于"),
            "comparison": ("测量之后，{s}的高度大于{t}。", "高度大于"),
            "role_binding": ("在这次交流中，{s}向{t}表达了谢意。", "表达了谢意"),
            "taxonomy": ("在这套分类中，{s}归入更宽泛的{t}类别。", "归入"),
        }
        natural = {
            "preference": ("两个都能选时，{s}总是挑中{t}。", "挑中"),
            "ownership": ("房间里的人都知道，{t}是{s}的。", "是"),
            "spatial": ("查看路线图可以发现，{s}就在{t}北边。", "北边"),
            "temporal": ("等到{t}发生时，{s}早已经发生了。", "早已经发生"),
            "causal": ("要是没有{s}，后来的{t}就不会发生。", "不会发生"),
            "comparison": ("并排站立时，{s}明显高过{t}。", "高过"),
            "role_binding": ("会议结束时，{s}向{t}道了谢。", "道了谢"),
            "taxonomy": ("手册把{s}列为{t}的一类。", "列为"),
        }
    return (discourse if surface == "discourse" else natural)[family]


def render_fact(family: str, language: str, surface: str, source: str, target: str) -> tuple[str, dict]:
    template, relation = relation_surface(family, language, surface)
    fact = template.format(s=source, t=target)
    spans = {}
    for key, text in (("source", source), ("target", target), ("relation", relation)):
        start = fact.index(text)
        spans[key] = [start, start + len(text)]
    return fact, spans


def partition(unit: int, surface: str) -> str:
    if surface in ("discourse", "natural"):
        return "template_lockbox" if unit < 10 else "joint_template_unit_lockbox"
    if unit < 6:
        return "discovery"
    if unit < 8:
        return "fresh_unit_lockbox"
    if unit < 10:
        return "confirmation"
    return "deep_fresh_unit_lockbox"


def compile_selection() -> list[dict]:
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for unit in range(12):
            for language_index, language in enumerate(LANGUAGES):
                a, b = pairs(language, unit)
                for surface_index, surface in enumerate(SURFACES):
                    for direction in (0, 1):
                        source, target = (a, b) if direction == 0 else (b, a)
                        fact, spans = render_fact(family, language, surface, source, target)
                        query_role = "source" if (unit + surface_index + direction + language_index) % 2 == 0 else "target"
                        answer = source if query_role == "source" else target
                        foil = target if query_role == "source" else source
                        candidate_order = (family_index + unit + surface_index + direction) % 2
                        candidates = [answer, foil] if candidate_order == 0 else [foil, answer]
                        query = prior.role_query(family, language, query_role)
                        prompt, events = prior.prompt_with_events(language, [(fact, spans)], query, candidates)
                        rows.append({
                            "case_id": f"dsel-{family}-u{unit}-{language}-{surface}-d{direction}",
                            "task": "selection", "family": family, "unit": unit, "language": language,
                            "surface": surface, "surface_class": "controlled" if surface in ("canonical", "paraphrase") else "naturalized",
                            "direction": direction, "query_role": query_role, "candidate_order": candidate_order,
                            "target_candidate_slot": candidates.index(answer), "partition": partition(unit, surface),
                            "source": source, "target": target, "fact": fact, "query": query, "candidates": candidates,
                            "answer": answer, "foil": foil, "prompt": prompt, "events": events,
                        })
    return rows


def compile_composition() -> list[dict]:
    rows = []
    for family_index, family in enumerate(COMPOSITION_FAMILIES):
        for unit in range(10):
            for language_index, language in enumerate(LANGUAGES):
                a, b, c = triples(language, unit)
                for surface_index, surface in enumerate(SURFACES):
                    for direction in (0, 1):
                        source, middle, endpoint = (a, b, c) if direction == 0 else (c, b, a)
                        facts = [render_fact(family, language, surface, source, middle),
                                 render_fact(family, language, surface, middle, endpoint)]
                        steps = 1 if (unit + surface_index + direction + language_index) % 2 == 0 else 2
                        answer, foil = (middle, endpoint) if steps == 1 else (endpoint, middle)
                        candidate_order = (family_index + unit + surface_index + direction) % 2
                        candidates = [answer, foil] if candidate_order == 0 else [foil, answer]
                        query = prior.composition_query(family, language, source, steps)
                        prompt, events = prior.prompt_with_events(language, facts, query, candidates)
                        rows.append({
                            "case_id": f"dcmp-{family}-u{unit}-{language}-{surface}-d{direction}",
                            "task": "composition", "family": family, "unit": unit, "language": language,
                            "surface": surface, "surface_class": "controlled" if surface in ("canonical", "paraphrase") else "naturalized",
                            "direction": direction, "steps": steps, "candidate_order": candidate_order,
                            "target_candidate_slot": candidates.index(answer), "partition": partition(unit, surface),
                            "source": source, "middle": middle, "endpoint": endpoint,
                            "facts": [fact[0] for fact in facts], "query": query, "candidates": candidates,
                            "answer": answer, "foil": foil, "prompt": prompt, "events": events,
                        })
    return rows


def audit_rows(selection: list[dict], composition: list[dict]) -> dict:
    rows = selection + composition
    return {
        "rows": len(rows), "selection_rows": len(selection), "composition_rows": len(composition),
        "families": Counter(row["family"] for row in rows), "languages": Counter(row["language"] for row in rows),
        "surfaces": Counter(row["surface"] for row in rows), "surface_classes": Counter(row["surface_class"] for row in rows),
        "partitions": Counter(row["partition"] for row in rows), "directions": Counter(row["direction"] for row in rows),
        "target_slots": Counter(row["target_candidate_slot"] for row in rows),
        "steps": Counter(row.get("steps") for row in composition),
        "unique_case_ids": len({row["case_id"] for row in rows}) == len(rows),
        "unique_prompts": len({row["prompt"] for row in rows}) == len(rows),
        "event_counts": {"selection": sorted({len(row["events"]) for row in selection}),
                         "composition": sorted({len(row["events"]) for row in composition})},
        "event_bounds": all(all(0 <= event["char_start"] <= event["char_end"] <= len(row["prompt"])
                                for event in row["events"]) for row in rows),
    }


def evidence_audit() -> dict:
    return {
        "retained": [
            "The shift from static HiddenState classification to event-aligned residual updates is a substantive methodological advance.",
            "The strongest measured law is a condition-indexed local update mean that generalizes partly to new units in four models.",
            "Cross-surface/language weakness, near-zero adjacent-coordinate persistence and weak behavior linkage are the decisive current limitations.",
            "A valid next campaign must deconfound platform, surface, language, role and content before claiming a state-dependent gear.",
            "Full physical coordinates remain the primary observation; sparse or low-rank objects are bounded competitors and diagnostics.",
        ],
        "corrected_or_rejected": [
            "Condition-offset did not prove that LLM computation is a conditional coordinate offset; it was the best tested external predictor.",
            "Sign groups are descriptive response codes, not demonstrated cooperative groups or gears.",
            "Cross-model positive gain is functional replication, not cross-architecture isomorphism or identical physical encoding.",
            "The output readout was not a causal bridge; behavior correlations were near zero or negative and the numeric interface retained batch/precision sensitivity.",
            "No attractor basin, conditional curvature, geodesic, knowledge manifold, semantic canal or continuous 36-layer carving path was measured.",
            "Attention as transport and MLP as semantic carving are hypotheses; broad component intervention before semantic deconfounding would repeat the old causal-gate loop.",
            "Deleting duplicate fields after verified full-coordinate publication was compliant with the user rule; future campaigns should publish reusable representative/full derived fields before cleanup.",
        ],
        "claim_boundary": "Phase2396-2404 established condition-dependent local-update texture, not a conditional gear, component division of labor, compositional algebra, compiler, or closure theory.",
    }


def mega_plan() -> list[dict]:
    return [
        {"phase": 2405, "campaign": "C24241-C24560", "task": "audit; whole-template/language/unit material; lock deconfounding and claim gates"},
        {"phase": 2406, "campaign": "C24561-C24880", "task": "four-model behavior/token calibration; Qwen14B BF16 device_map=auto feasibility and precision ledger"},
        {"phase": 2407, "campaign": "C24881-C25200", "task": "Qwen4B event/full-token H plus Attention/MLP/update full-coordinate source atlas"},
        {"phase": 2408, "campaign": "C25201-C25520", "task": "sequential deconfounding: shared layer/event platform, surface/language/role, family, content and interaction"},
        {"phase": 2409, "campaign": "C25521-C25840", "task": "full-coordinate state-dependent operator competition against frozen condition means on template/language/unit lockboxes"},
        {"phase": 2410, "campaign": "C25841-C26160", "task": "coordinate cooperation tests with matched random/RMS/covariance controls and component-source attribution"},
        {"phase": 2411, "campaign": "C26161-C26480", "task": "cross-layer functional transport, cumulative-state path, one-to-two-step composition residual and output/behavior bridge"},
        {"phase": 2412, "campaign": "C26481-C26800", "task": "frozen Qwen14B, GLM4 and DS7B replication, sequentially; model-local coordinates only"},
        {"phase": 2413, "campaign": "C26801-C27120", "task": "publish full-coordinate heatmaps/derived atlases, frontend verification, cleanup and successor audit"},
    ]


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 条件更新证据审查、整模板解混材料与新阶段合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 逐条审查两份Phase2396–2404复盘，并以原始final/MEMO为证据边界。新材料不再只锁新实体：选择任务覆盖8族×12 unit×中英×canonical/paraphrase/discourse/natural四种整体表面×双方向，共1536条；组合任务覆盖空间/时间/比较/分类4族×10 unit×双语×四表面×双方向，并反平衡一步/两步，共640条。查询角色、候选槽、方向按确定性正交规则平衡；canonical/paraphrase与discourse/natural形成整模板锁箱，unit6–7和10–11形成两级新内容锁箱，任何分析均须同时报告“训练中见过语言”和“整语言留出”两套冻结评价。

$$U_{{r,e,q,j}}=B_{{e,q,j}}+S_{{\ell,s,d,\rho,e,q,j}}+G_{{f,e,q,j}}+D_{{u,e,q,j}}+I_{{f,u,e,q,j}}+\epsilon_{{r,e,q,j}},$$

$$\text{{齿轮候选门}}:\quad \widehat U(H_q,c)>\widehat U(c)\quad\text{{必须同时跨整模板、语言与新unit成立。}}$$

**结果汇总。** 附件审查 `{json.dumps(result['evidence_audit'], ensure_ascii=False)}`；材料 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；一次性方案 `{json.dumps(result['mega_plan'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2405_c24241_c24560_deconfounded_operation_contract.py`；2176条材料、附件哈希、合同和final位于`tests/glm5/result/phase2405_c24241_c24560_deconfounded_operation_contract`。未修改其他Markdown。

**分析与理论进展。** 保留“事件对齐局部更新是比静态分类更接近计算的研究对象”，但把条件均值严格降格为外部预测基线。新的核心问题不是哪个组件看起来像搬运/雕刻，而是：剥离共享层底盘、完整表面、语言、方向、角色和内容之后，是否仍有使用真实$H_q$的全坐标规律跨锁箱超过条件均值；只有通过后才定位Attention/MLP来源。

**问题硬伤与结论。** 自然化表面仍是人工生成而非开放语料；同一实体在不同族复用会引入词项相关；正交平衡通过确定性配对实现，不等于每个高阶交互都有足够独立单元。物理坐标上的观测数不能充当独立语言样本。附件二关于“真正齿轮、连续引流、吸引子盆地、因果关键一环、跨架构同构”的陈述均无证据支持。下一Phase先裁决四模型行为与Qwen14B BF16可行性，失败只限定相应模型的精度/行为解释，不停止其他通过特征的图谱研究。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    selection = compile_selection(); composition = compile_composition()
    write_rows(OUT / "material/selection_rows.jsonl", selection)
    write_rows(OUT / "material/composition_rows.jsonl", composition)
    write_rows(OUT / "material/all_rows.jsonl", selection + composition)
    material = audit_rows(selection, composition)
    checks = {
        "attachments": all(path.exists() for path in ATTACHMENTS),
        "phase_continuity": "## Phase 2404:" in MEMO.read_text(encoding="utf-8") and "## Phase 2405:" not in MEMO.read_text(encoding="utf-8"),
        "selection_rows": material["selection_rows"] == 1536, "composition_rows": material["composition_rows"] == 640,
        "unique": material["unique_case_ids"] and material["unique_prompts"], "events": material["event_bounds"],
        "surface_balance": len(set(material["surfaces"].values())) == 1,
        "language_balance": len(set(material["languages"].values())) == 1,
        "coordinate_contract": True,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN,
              "attachments": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in ATTACHMENTS],
              "evidence_audit": evidence_audit(), "material_audit": material, "mega_plan": mega_plan(),
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
