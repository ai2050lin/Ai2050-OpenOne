#!/usr/bin/env python3
"""Audit Phase 2378-2387 interpretations and freeze a semantic/lexical factorial contract."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2388_c18801_c19120_semantic_lexical_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2388
CAMPAIGN = "C18801-C19120"
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\b5aac698-eb0b-49ed-be7e-6977d370400e\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\b4136af0-7444-4393-a344-2b0f9bd2158f\pasted-text.txt"),
)
FAMILIES = (
    "preference", "taxonomy", "temporal", "causal",
    "comparison", "spatial", "role_binding", "ownership_transfer",
)
LANGUAGES = ("en", "zh")


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
    if unit <= 5:
        return "discovery"
    if unit <= 8:
        return "confirmation"
    return "fresh_unit_lockbox"


NAMES_A = ("Mira", "Jonas", "Lena", "Omar", "Priya", "Tomas", "Nadia", "Felix", "Asha", "Ruben", "Inez", "Caleb")
NAMES_B = ("Noah", "Sara", "Arun", "Mei", "Dario", "Elena", "Hugo", "Zara", "Kiran", "Sofia", "Bruno", "Yuna")
ITEM_A = ("apples", "pears", "tea", "coffee", "maps", "charts", "cedar", "willow", "silver", "copper", "linen", "wool")
ITEM_B = ("oranges", "plums", "water", "cocoa", "notes", "tables", "birch", "maple", "bronze", "iron", "cotton", "silk")
SPECIFIC = ("robin", "salmon", "orchid", "violin", "sparrow", "trout", "tulip", "compass", "eagle", "carp", "rose", "thermometer")
BROAD = ("bird", "fish", "plant", "instrument", "bird", "fish", "plant", "instrument", "bird", "fish", "plant", "instrument")
PLACE_A = ("cafe", "clinic", "museum", "orchard", "school", "harbor", "bakery", "station", "library", "theater", "market", "garden")
PLACE_B = ("library", "station", "market", "bridge", "clinic", "museum", "school", "harbor", "cafe", "garden", "theater", "bakery")
REGION = ("Alder", "Birch", "Cedar", "Dover", "Elm", "Frost", "Grove", "Hazel", "Iris", "Juniper", "Kingfisher", "Laurel")
EVENT_A = ("the spark", "the leak", "the frost", "the alarm", "the blockage", "the storm", "the signal", "the spill", "the crack", "the outage", "the vibration", "the warning")
EVENT_B = ("the alarm", "the shutdown", "the wilt", "the evacuation", "the pressure drop", "the delay", "the launch", "the closure", "the collapse", "the restart", "the fracture", "the inspection")


def forms(family: str, unit: int, language: str, bit: int) -> tuple[str, str]:
    a, b = NAMES_A[unit], NAMES_B[unit]
    ia, ib = ITEM_A[unit], ITEM_B[unit]
    specific, broad = SPECIFIC[unit], BROAD[unit]
    pa, pb = f"{REGION[unit]} {PLACE_A[unit]}", f"{REGION[unit]} {PLACE_B[unit]}"
    ea, eb = EVENT_A[unit], EVENT_B[unit]
    if bit:
        a, b = b, a
        ia, ib = ib, ia
        specific, broad = broad, specific
        pa, pb = pb, pa
        ea, eb = eb, ea
    if language == "en":
        table = {
            "preference": (f"{a} prefers {ia} to {ib}.", f"Given a choice between {ib} and {ia}, {a} would choose {ia}."),
            "taxonomy": (f"A {specific} is a kind of {broad}.", f"The class called {broad} includes the {specific}."),
            "temporal": (f"{a} arrived before {b}.", f"{b} arrived only after {a} had come."),
            "causal": (f"{ea} caused {eb}.", f"{eb} occurred because of {ea}."),
            "comparison": (f"The {ia} sample is heavier than the {ib} sample.", f"The {ib} sample weighs less than the {ia} sample."),
            "spatial": (f"The {pa} is north of the {pb}.", f"The {pb} lies to the south of the {pa}."),
            "role_binding": (f"{a} thanked {b} after the review.", f"Following the review, {b} received thanks from {a}."),
            "ownership_transfer": (f"{a} lent the atlas to {b}.", f"{b} borrowed the atlas from {a}."),
        }
    else:
        table = {
            "preference": (f"{a}喜欢{ia}胜过{ib}。", f"如果在{ib}和{ia}之间选择，{a}会选{ia}。"),
            "taxonomy": (f"{specific}是一种{broad}。", f"名为{broad}的类别包含{specific}。"),
            "temporal": (f"{a}在{b}之前到达。", f"{b}直到{a}到达之后才来。"),
            "causal": (f"{ea}导致了{eb}。", f"{eb}是由于{ea}而发生的。"),
            "comparison": (f"{ia}样品比{ib}样品更重。", f"{ib}样品的重量小于{ia}样品。"),
            "spatial": (f"{pa}位于{pb}以北。", f"{pb}坐落在{pa}以南。"),
            "role_binding": (f"复核以后，{a}感谢了{b}。", f"复核结束后，{b}收到了{a}的感谢。"),
            "ownership_transfer": (f"{a}把地图册借给了{b}。", f"{b}从{a}那里借来了地图册。"),
        }
    return table[family]


def prompt(language: str, candidates: list[str], query: str) -> str:
    if language == "en":
        return (
            "Two statements use nearly the same words but express opposite relations:\n"
            + "\n".join(candidates)
            + f"\n\nRepeat exactly the statement equivalent in meaning to: {query}\n"
            + "Output only that statement.\n"
        )
    return (
        "下面两句话使用几乎相同的词，但表达相反的关系：\n"
        + "\n".join(candidates)
        + f"\n\n请逐字重复与下面含义相同的那句话：{query}\n"
        + "只输出选中的原句。\n"
    )


def compile_material() -> tuple[list[dict], list[dict], dict]:
    independent: list[dict] = []
    selection: list[dict] = []
    for family in FAMILIES:
        for unit in range(12):
            for language in LANGUAGES:
                pair = [forms(family, unit, language, bit) for bit in (0, 1)]
                canonical = [item[0] for item in pair]
                paraphrase = [item[1] for item in pair]
                group_id = f"{family}-u{unit:02d}-{language}"
                for bit in (0, 1):
                    independent.extend((
                        {"case_id": f"{group_id}-b{bit}-canonical", "group_id": group_id, "family": family,
                         "unit": unit, "language": language, "relation_bit": bit, "form": "canonical",
                         "partition": partition(unit), "text": canonical[bit], "paired_counterfactual": canonical[1 - bit]},
                        {"case_id": f"{group_id}-b{bit}-paraphrase", "group_id": group_id, "family": family,
                         "unit": unit, "language": language, "relation_bit": bit, "form": "paraphrase",
                         "partition": partition(unit), "text": paraphrase[bit], "paired_counterfactual": paraphrase[1 - bit]},
                    ))
                    order = (0, 1) if (unit + bit + FAMILIES.index(family)) % 2 == 0 else (1, 0)
                    candidates = [canonical[index] for index in order]
                    selection.append({
                        "case_id": f"{group_id}-target-b{bit}", "group_id": group_id, "family": family,
                        "unit": unit, "language": language, "relation_bit": bit, "partition": partition(unit),
                        "candidate_order": list(order), "candidates": candidates, "query_paraphrase": paraphrase[bit],
                        "prompt": prompt(language, candidates, paraphrase[bit]), "target": canonical[bit],
                        "foil": canonical[1 - bit], "target_candidate_slot": order.index(bit),
                    })
    audit = {
        "independent_rows": len(independent), "selection_rows": len(selection),
        "families": Counter(row["family"] for row in selection),
        "languages": Counter(row["language"] for row in selection),
        "partitions": Counter(row["partition"] for row in selection),
        "independent_unique": len({row["text"] for row in independent}) == len(independent),
        "selection_unique": len({row["prompt"] for row in selection}) == len(selection),
        "paired_exact_bag_control": "canonical bit pairs reverse roles/order while retaining the same principal lexical items",
        "semantic_surface_control": "canonical and paraphrase express the same directed relation with different syntax and relation wording",
    }
    return independent, selection, audit


def append_memo(result: dict) -> None:
    memo_text = MEMO.read_text(encoding="utf-8")
    if f"## Phase {PHASE}:" in memo_text:
        marker = "**执行修正（Phase2388唯一性门）**"
        if result.get("all_checks_passed") and marker not in memo_text:
            with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    "\n\n**执行修正（Phase2388唯一性门）**：首次生成发现两个空间unit的地点对互为交换，"
                    "导致独立句与选择prompt唯一性门失败。保留该失败记录后，为每个空间unit加入不同地区名并重新生成；"
                    f"修正后材料审计 `{json.dumps(result['material_audit'], ensure_ascii=False)}`，全部合同检查通过。\n"
                )
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 附件证据审查与语义—词汇正交总合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 审查两份Phase2378–2387分析，冻结“描述、推断、机制”边界；不预设组合纹理、Attention复制、仿射齿轮、曲率盆地或跨模型微分同胚。建立8个关系族×12个独立unit×中英双语×两个相反关系方向×canonical/paraphrase，共768个独立句；另建384个无标签二选一语义等价任务。相反方向canonical尽量保留同一组实体和主要词汇，只交换顺序/角色；paraphrase保留关系方向但更换句法和关系表达。unit级discovery/confirmation/fresh lockbox严格隔离。

$$x=(f,u,\ell,d,s),\qquad R_{{f,u,\ell}}=H(x_{{d=0}})-H(x_{{d=1}}),$$

$$\text{{semantic invariance}}\Rightarrow R^{{canonical}}\approx R^{{paraphrase}},\qquad
\text{{lexical fingerprint only}}\Rightarrow \text{{deep gain over embedding disappears}}.$$

**结果汇总。** 材料审计 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；附件哈希 `{json.dumps(result['attachments'], ensure_ascii=False)}`；保留结论 `{json.dumps(result['evidence_audit']['retained'], ensure_ascii=False)}`；修正结论 `{json.dumps(result['evidence_audit']['corrected'], ensure_ascii=False)}`；后续合同 `{json.dumps(result['mega_plan'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2388_c18801_c19120_semantic_lexical_contract.py`；材料与final位于 `tests/glm5/result/phase2388_c18801_c19120_semantic_lexical_contract`。未修改其他Markdown。

**理论进展、问题硬伤与结论。** 保留“独立句场可条件匹配输出准备场”和“同模型稳定坐标组织重要”；将“独立组合语义纹理已证明”降为候选，因为GLM/DS不超过embedding且旧材料词汇重叠高。0.9951属于sentence-end而非pre-sentence；pre-sentence锁箱0.9043属于Phase2381。Attention head25/10只是经多重搜索得到的观察候选，MLP质心失败只否决简单门控读法。下一步首先以同词相反关系和异表达同关系裁决语义增益，再构建坐标响应图谱；只有冻结候选才进入有限因果测试。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        cached = json.loads(final.read_text(encoding="utf-8"))
        if cached.get("all_checks_passed"):
            append_memo(cached); print(json.dumps(cached, ensure_ascii=False, indent=2)); return
    independent, selection, material_audit = compile_material()
    write_rows(OUT / "material/independent_relation_sentences.jsonl", independent)
    write_rows(OUT / "material/semantic_selection_rows.jsonl", selection)
    evidence_audit = {
        "retained": [
            "independent sentence fields conditionally match output-preparation states",
            "within-model stable coordinate organization matters while absolute coordinate labels are not universal",
            "Qwen4B and Qwen14B show a confirmation-selected deep gain over embedding means on the prior material",
            "Qwen4B layer25/head10 is an observational source-routing candidate",
            "native chat generation corrects the old truncation confound but does not close reverse/coreference behavior",
        ],
        "corrected": [
            "Phase2380 0.9951 is sentence-end matching, not pre-sentence matching",
            "deep mean-state gain does not yet prove semantic or compositional texture",
            "donor reduction is a content control, not a complete causal separation",
            "attention routing has not been shown necessary, sufficient, or responsible for copying",
            "diagonal regression coefficients are fitted readout parameters, not internal runtime gears",
            "no curvature basin, geodesic computation, diffeomorphism, or knowledge-manifold intervention has been measured",
        ],
    }
    mega_plan = {
        "A": "uniform autonomous capability audit for four local models; choose a discovery model by frozen behavior criteria",
        "B": "full-coordinate same-bag opposite-relation and cross-surface same-relation adjudication",
        "C": "coordinate response fingerprints, all-coordinate dynamics, group reuse and unseen-unit prediction",
        "D": "generation-success bridge plus limited frozen coordinate/head interventions with wrong controls and rescue",
        "publication": "publish important embedding/HiddenState coordinate fields and delete only verified duplicates",
    }
    checks = {
        "attachments_present": all(path.exists() for path in ATTACHMENTS),
        "independent_rows": material_audit["independent_rows"] == 768,
        "selection_rows": material_audit["selection_rows"] == 384,
        "eight_families": len(material_audit["families"]) == 8,
        "unit_lockbox": material_audit["partitions"]["fresh_unit_lockbox"] == 96,
        "unique": material_audit["independent_unique"] and material_audit["selection_unique"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN,
              "attachments": [{"path": str(path), "sha256": sha256(path)} for path in ATTACHMENTS],
              "evidence_audit": evidence_audit, "material_audit": material_audit,
              "mega_plan": mega_plan, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
