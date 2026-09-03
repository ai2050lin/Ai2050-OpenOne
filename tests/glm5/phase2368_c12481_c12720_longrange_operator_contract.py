#!/usr/bin/env python3
"""Audit Phase 2358-2367 and freeze a long-range language-operator contract."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase2368_c12481_c12720_longrange_operator_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
LONG_MATERIAL = OUT / "material/long_sentence_permutation.jsonl"
FLAGSHIP_MATERIAL = OUT / "material/flagship_factorial.jsonl"
PHASE = 2368
CAMPAIGN = "C12481-C12720"
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\03487f93-b931-4226-8537-c1f823c818a6\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\798ec120-122b-4cac-be99-ef3b9a5334dd\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\9504c992-d3bd-4b06-bead-e781b558311d\pasted-text.txt"),
)
FAMILIES = (
    "temporal_narrative", "causal_process", "taxonomy_explanation", "spatial_route",
    "procedure", "comparison", "dialogue_coreference", "scientific_description",
)
LANGUAGES = ("en", "zh")
SOURCE_PERMS = ((0, 1, 2, 3), (2, 0, 3, 1))
ALL_PERMS = tuple(itertools.permutations(range(4)))
ADJACENT = ((1, 0, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2))
COPY_PERMS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2),
    (1, 2, 0, 3), (0, 2, 3, 1), (2, 0, 3, 1), (3, 2, 1, 0),
)
PARAPHRASE_PERMS = COPY_PERMS[:4]
MARKERS = (
    "Alder", "Birch", "Cedar", "Dahlia", "Elm", "Fir", "Ginkgo", "Hazel",
    "Iris", "Juniper", "Kestrel", "Linden", "Maple", "Nectar", "Olive", "Pine",
    "Quartz", "Rowan", "Saffron", "Tulip", "Umber", "Violet", "Willow", "Xenia",
    "Yarrow", "Zephyr", "Amber", "Beryl", "Copper", "Delta", "Ember", "Flint",
)
FAMILY_TERMS = {
    "temporal_narrative": ("expedition", "recorded", "chronology"),
    "causal_process": ("irrigation system", "changed", "causal chain"),
    "taxonomy_explanation": ("botanical archive", "classified", "category ladder"),
    "spatial_route": ("mountain route", "crossed", "spatial path"),
    "procedure": ("laboratory protocol", "completed", "ordered procedure"),
    "comparison": ("design committee", "compared", "evidence ranking"),
    "dialogue_coreference": ("review meeting", "clarified", "speaker reference"),
    "scientific_description": ("field observatory", "measured", "physical sequence"),
}
RANK_PATTERNS = ((2, 0, 3, 1), (1, 3, 0, 2), (3, 1, 2, 0), (0, 2, 1, 3))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def unit_partition(unit: int) -> str:
    return "discovery" if unit < 3 else "confirmation" if unit == 3 else "fresh_unit_lockbox"


def perm_partition(perm: tuple[int, ...]) -> str:
    if perm == (0, 1, 2, 3) or perm in ADJACENT:
        return "generator_discovery"
    if perm in ((1, 2, 0, 3), (0, 2, 3, 1)):
        return "composition_confirmation"
    return "unseen_composition_lockbox"


def render_sentences(family: str, unit: int, language: str) -> tuple[list[str], list[str], list[int]]:
    topic, verb, order_word = FAMILY_TERMS[family]
    base = FAMILIES.index(family) * 4 + unit * 7
    ranks = list(RANK_PATTERNS[unit % len(RANK_PATTERNS)])
    sentences, paraphrases = [], []
    for sid in range(4):
        marker = MARKERS[(base + sid) % len(MARKERS)]
        rank = ranks[sid] + 1
        code = f"{family[:3].upper()}{unit}{sid}"
        if language == "en":
            sentence = (f"{marker} reports stage {rank}: the {topic} {verb} a distinct item under code {code}, "
                        f"while two observers preserved its wording and linked it only to this {order_word}.")
            para = (f"{marker} describes the item labeled {code} as stage {rank} in the {order_word}; "
                    f"both observers retained that specific fact when the {topic} was reviewed.")
        else:
            sentence = (f"{marker}记录第{rank}阶段：{topic}对编号{code}的独立项目完成了“{verb}”事件；"
                        f"两名观察者保留原句，并确认它只属于这条{order_word}。")
            para = (f"{marker}说明编号{code}的项目位于{order_word}的第{rank}阶段；"
                    f"两名观察者在复核{topic}时都保留了这一特定事实。")
        sentences.append(sentence)
        paraphrases.append(para)
    return sentences, paraphrases, ranks


def desired_indices(source_perm: tuple[int, ...], target_perm: tuple[int, ...]) -> list[int]:
    return [source_perm.index(sid) + 1 for sid in target_perm]


def compile_long_material(tokenizer) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for family in FAMILIES:
        for unit in range(6):
            for language in LANGUAGES:
                sentences, paraphrases, ranks = render_sentences(family, unit, language)
                markers = [s.split()[0].split("记录")[0] for s in sentences]
                semantic_order = tuple(sorted(range(4), key=lambda i: ranks[i]))
                for source_perm in SOURCE_PERMS:
                    source_lines = "\n".join(f"Sentence {slot + 1}: {sentences[sid]}"
                                             for slot, sid in enumerate(source_perm))
                    for task, perms in (
                        ("index_only", ALL_PERMS), ("exact_copy", COPY_PERMS),
                        ("paraphrase_reorder", PARAPHRASE_PERMS), ("semantic_sort", (semantic_order,)),
                    ):
                        for target_perm in perms:
                            marker_order = ", ".join(markers[i] for i in target_perm)
                            if language == "en":
                                if task == "index_only":
                                    instruction = ("Return only the comma-separated source sentence numbers whose leading markers "
                                                   f"appear in this requested order: {marker_order}. Answer:")
                                elif task == "exact_copy":
                                    instruction = ("Copy the four source sentences verbatim in this marker order, with one sentence "
                                                   f"per line: {marker_order}. Answer:")
                                elif task == "paraphrase_reorder":
                                    instruction = ("Paraphrase each sentence without changing its facts, in this marker order, one "
                                                   f"per line: {marker_order}. Answer:")
                                else:
                                    instruction = "Sort all sentences by the explicit stage number, earliest first, and copy them verbatim. Answer:"
                            else:
                                if task == "index_only":
                                    instruction = f"只返回首标记按以下顺序出现时对应的来源句编号，用逗号分隔：{marker_order}。答案："
                                elif task == "exact_copy":
                                    instruction = f"按照以下首标记顺序逐行原样复制四个来源句：{marker_order}。答案："
                                elif task == "paraphrase_reorder":
                                    instruction = f"不改变事实，按照以下首标记顺序逐行改写四句：{marker_order}。答案："
                                else:
                                    instruction = "按明确写出的阶段编号从早到晚排序，并原样复制全部句子。答案："
                            prompt = source_lines + "\n\n" + instruction
                            foil_perm = target_perm[1:] + target_perm[:1]
                            if task == "index_only":
                                # A leading ASCII space is a standalone Qwen token, which would make every
                                # target/foil first token identical and invalidate the frozen behavior gate.
                                target = ",".join(map(str, desired_indices(source_perm, target_perm)))
                                foil = ",".join(map(str, desired_indices(source_perm, foil_perm)))
                            elif task == "paraphrase_reorder":
                                target = " " + "\n".join(paraphrases[i] for i in target_perm)
                                foil = " " + "\n".join(paraphrases[i] for i in foil_perm)
                            else:
                                target = " " + "\n".join(sentences[i] for i in target_perm)
                                foil = " " + "\n".join(sentences[i] for i in foil_perm)
                            target_ids = tokenizer.encode(target, add_special_tokens=False)
                            foil_ids = tokenizer.encode(foil, add_special_tokens=False)
                            row = {
                                "case_id": f"c12481-{family}-u{unit}-{language}-s{SOURCE_PERMS.index(source_perm)}-{task}-p{ALL_PERMS.index(tuple(target_perm)):02d}",
                                "design_index": len(rows), "family": family, "unit": unit,
                                "unit_partition": unit_partition(unit), "language": language, "task": task,
                                "source_perm": list(source_perm), "target_perm": list(target_perm),
                                "permutation_index": ALL_PERMS.index(tuple(target_perm)),
                                "permutation_partition": perm_partition(tuple(target_perm)),
                                "semantic_order": list(semantic_order), "source_sentences": sentences,
                                "source_paraphrases": paraphrases, "markers": markers, "ranks": ranks,
                                "prompt": prompt, "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False),
                                "target": target, "foil": foil, "target_ids": target_ids, "foil_ids": foil_ids,
                                "target_first_id": target_ids[0], "foil_first_id": foil_ids[0],
                            }
                            rows.append(row)
    task_counts = Counter(r["task"] for r in rows)
    audit = {
        "rows": len(rows), "expected_rows": 8 * 6 * 2 * 2 * (24 + 8 + 4 + 1),
        "families": len(set(r["family"] for r in rows)), "languages": sorted(set(r["language"] for r in rows)),
        "units": len(set(r["unit"] for r in rows)), "source_permutations": len(set(tuple(r["source_perm"]) for r in rows)),
        "target_permutations_index_task": len(set(tuple(r["target_perm"]) for r in rows if r["task"] == "index_only")),
        "task_counts": dict(task_counts), "unique_case_ids": len(set(r["case_id"] for r in rows)) == len(rows),
        "first_answer_token_distinct": all(r["target_first_id"] != r["foil_first_id"] for r in rows),
        "prompt_token_range": [min(len(r["prompt_ids"]) for r in rows), max(len(r["prompt_ids"]) for r in rows)],
        "unit_partitions": dict(Counter(r["unit_partition"] for r in rows)),
        "permutation_partitions": dict(Counter(r["permutation_partition"] for r in rows if r["task"] == "index_only")),
    }
    return rows, audit


def localized(values: tuple[str, str], language: str) -> str:
    return values[0] if language == "en" else values[1]


def compile_flagships(tokenizer) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    roles = ("subject", "polarity", "attitude", "action", "object")
    for unit in range(12):
        for language in LANGUAGES:
            for surface in ("direct", "independent_prose"):
                for cell in range(32):
                    bits = [(cell >> i) & 1 for i in range(5)]
                    subject = localized((("I", "我"), ("she", "她"))[bits[0]], language)
                    polarity = bits[1]
                    attitude = localized((("like", "喜欢"), ("prefer", "偏爱"))[bits[2]], language)
                    action = localized((("eat", "吃"), ("buy", "购买"))[bits[3]], language)
                    obj = localized((("apples", "苹果"), ("pears", "梨"))[bits[4]], language)
                    values = {"subject": subject, "polarity": localized((("affirmative", "肯定"), ("negative", "否定"))[polarity], language),
                              "attitude": attitude, "action": action, "object": obj}
                    query = roles[unit % len(roles)]
                    neg = localized(("not ", "不"), language) if polarity else ""
                    if language == "en":
                        statement = f"{subject} {neg}{attitude} to {action} orchard {obj}."
                        prompt = (statement if surface == "direct" else
                                  f"During an unrelated morning report, the exact attitude sentence was: '{statement}'")
                        prompt += f"\nReturn only the {query} expressed in that sentence. Answer:"
                    else:
                        statement = f"{subject}{neg}{attitude}{action}果园里的{obj}。"
                        prompt = (statement if surface == "direct" else f"在一段无关的晨间报告中，原态度句是：“{statement}”")
                        prompt += f"\n只返回该句表达的{query}。答案："
                    target = " " + values[query]
                    foil_value = localized(("unknown", "未知"), language)
                    foil = " " + foil_value
                    target_ids, foil_ids = [tokenizer.encode(x, add_special_tokens=False) for x in (target, foil)]
                    rows.append({
                        "case_id": f"c13681-attitude-u{unit}-{language}-{surface}-c{cell:02d}", "design_index": len(rows),
                        "system": "attitude_role", "unit": unit, "partition": unit_partition(unit // 2),
                        "language": language, "surface": surface, "cell": cell, "bits": bits,
                        "factors": dict(zip(("subject", "polarity", "attitude", "action", "object"), bits)),
                        "query": query, "prompt": prompt, "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False),
                        "target": target, "foil": foil, "target_ids": target_ids, "foil_ids": foil_ids,
                        "target_first_id": target_ids[0], "foil_first_id": foil_ids[0],
                    })
    chains = (
        (("apple", "苹果"), ("fruit", "水果"), ("food", "食物"), ("object", "物体")),
        (("pear", "梨"), ("fruit", "水果"), ("food", "食物"), ("object", "物体")),
    )
    for unit in range(12):
        for language in LANGUAGES:
            for surface in ("facts", "independent_prose"):
                for cell in range(32):
                    bits = [(cell >> i) & 1 for i in range(5)]
                    nodes = [localized(v, language) for v in chains[bits[0]]]
                    rel = localized((("is a", "是一种"), ("belongs to", "属于"))[bits[1]], language)
                    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3])]
                    if bits[2]:
                        edges.pop(1)
                    if bits[3]:
                        edges.append((nodes[0], localized(("seed-bearing thing", "含种子的事物"), language)))
                    query_depth = 3 if bits[4] else 1
                    target_value = nodes[query_depth] if not (bits[2] and query_depth == 3) else localized(("unknown", "未知"), language)
                    foil_value = localized(("unknown", "未知"), language) if target_value != localized(("unknown", "未知"), language) else nodes[3]
                    if language == "en":
                        facts = "; ".join(f"{a} {rel} {b}" for a, b in edges) + "."
                        prompt = (facts if surface == "facts" else f"An archivist wrote these independent facts in prose: {facts}")
                        prompt += f"\nUsing only these facts, what is the depth-{query_depth} category of {nodes[0]}? Answer:"
                    else:
                        facts = "；".join(f"{a}{rel}{b}" for a, b in edges) + "。"
                        prompt = (facts if surface == "facts" else f"档案员用独立叙述记录了这些事实：{facts}")
                        prompt += f"\n只根据这些事实，{nodes[0]}的第{query_depth}层类别是什么？答案："
                    target, foil = " " + target_value, " " + foil_value
                    target_ids, foil_ids = [tokenizer.encode(x, add_special_tokens=False) for x in (target, foil)]
                    rows.append({
                        "case_id": f"c13681-chain-u{unit}-{language}-{surface}-c{cell:02d}", "design_index": len(rows),
                        "system": "taxonomy_chain", "unit": unit, "partition": unit_partition(unit // 2),
                        "language": language, "surface": surface, "cell": cell, "bits": bits,
                        "factors": dict(zip(("entity_lexicon", "relation_variant", "middle_edge_removed", "branch_edge", "query_depth"), bits)),
                        "query": f"depth_{query_depth}", "edges": edges, "prompt": prompt,
                        "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False), "target": target, "foil": foil,
                        "target_ids": target_ids, "foil_ids": foil_ids, "target_first_id": target_ids[0], "foil_first_id": foil_ids[0],
                    })
    audit = {
        "rows": len(rows), "expected_rows": 2 * 12 * 2 * 2 * 32,
        "systems": dict(Counter(r["system"] for r in rows)), "languages": sorted(set(r["language"] for r in rows)),
        "surfaces": sorted(set(r["surface"] for r in rows)), "cells": len(set(r["cell"] for r in rows)),
        "unique_case_ids": len(set(r["case_id"] for r in rows)) == len(rows),
        "first_answer_token_distinct": all(r["target_first_id"] != r["foil_first_id"] for r in rows),
        "prompt_token_range": [min(len(r["prompt_ids"]) for r in rows), max(len(r["prompt_ids"]) for r in rows)],
    }
    return rows, audit


def evidence_audit() -> dict:
    return {
        "attachment_sha256": [{"path": str(p), "sha256": sha256(p)} for p in ATTACHMENTS],
        "retained": [
            "Phase2358-2367 built a valid full-coordinate five-factor response atlas; exact Walsh algebra closes the frozen Boolean design.",
            "The matched lockbox result supports weak global predictive reuse (R2=0.2051) and rejects the selected structure-conditioned template as a superior general law.",
            "Long-sentence reordering behavior requires functional long-range boundary recognition, content binding, target-order control, source selection and reconstruction.",
            "State-response laws and operators are a better next object than another family-mean deletion direction.",
            "Adjacent transpositions provide a compact, falsifiable route from Boolean factors to S_n composition tests.",
        ],
        "corrected_overclaims": [
            "Target-over-one-foil=0.9572 is not equivalent to full-vocabulary correctness; Phase2359 top1 was only 0.263.",
            "A sorted-coordinate control proves address-sensitive prediction, not causal necessity, precision gears or an internal coordinate ontology.",
            "Five prompt factors are experimental variables, not demonstrated primitive generators of the model's language mechanism.",
            "Exact Walsh decomposition is an algebraic identity of the 32-cell design, not the ultimate mathematical answer or mechanism closure.",
            "Failure of a global third-order template does not prove that no global semantic gear exists; it rejects that estimator on that lockbox.",
            "Behavioral exact-copy reorder does not by itself prove an explicit permutation group, tensor-block transport, topological isomorphism or non-Euclidean manifold.",
            "Local Jacobian, singularity, diffeomorphism, optimal transport and persistent topology remain competing diagnostics until lockbox prediction and controls pass.",
            "HiddenState coordinates are activations; they must not be labeled as trained model parameters.",
        ],
        "claim_boundary": {
            "behavior": "functional cross-sentence binding/selection/reconstruction",
            "descriptive_state": "full physical-coordinate activation response",
            "predictive": "fresh content, source order and unseen permutation lockboxes",
            "mechanism": "requires composition/generalization and selective intervention; not presupposed",
        },
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 长距离语言算子证据纠错与统一冻结合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本Phase逐项审查三份附件及Phase2358–2367原始记录，严格区分“行为能无损重排”“HiddenState存在可读结构”和“内部实现显式置换群/拓扑搬运”。冻结四类任务：仅输出来源索引、逐句原样重排、保义改写重排、按内容语义排序；覆盖8类长文本、6个独立unit、中英、2种独立来源排列和$S_4$全部24种目标排列。另冻结“我喜欢吃苹果”的角色五因子立方与“苹果—水果—食物—物体”的关系链五因子立方。

$$
H=B+C(	ext{{sentence}})+O(pi,k)+A(sigma,	ext{{source}})+R(	ext{{offset}})+I(	ext{{sentence}},k)+\varepsilon,
$$

$$
\rho(	au_i)\rho(	au_i)=I,quad
\rho(	au_i)\rho(	au_{{i+1}})\rho(	au_i)=\rho(	au_{{i+1}})\rho(	au_i)\rho(	au_{{i+1}}).
$$

**结果汇总。** 证据审查 `{json.dumps(result['evidence_audit'], ensure_ascii=False)}`；长句材料 `{json.dumps(result['long_material_audit'], ensure_ascii=False)}`；双旗舰材料 `{json.dumps(result['flagship_material_audit'], ensure_ascii=False)}`；冻结竞赛 `{json.dumps(result['freeze'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2368_c12481_c12720_longrange_operator_contract.py`；材料和审计位于 `tests/glm5/result/phase2368_c12481_c12720_longrange_operator_contract`；未新增其他Markdown。

**理论进展、问题硬伤与结论。** 正确保留的是：长句重排证明模型具有功能性的跨句选择与输出规划能力，Phase2358–2367证明特定实验域存在地址敏感的分布式激活响应和弱全局预测复用。必须撤回的是：把Walsh恒等式称为终极机制、把排序坐标对照称为因果必要性、把重排行为直接解释成显式置换群/张量块搬运/拓扑同构。新阶段不预设哪种高等数学正确；所有方法在同一数据、同一目标、同一锁箱和MDL账本上竞争，并必须用自然语言、位置、长度、来源排列和语言对照排除捷径。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists() and LONG_MATERIAL.exists() and FLAGSHIP_MATERIAL.exists() and json.loads(final_path.read_text(encoding="utf-8")).get("all_checks_passed"):
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    sys.path.insert(0, str(TESTS))
    import model_utils
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                               local_files_only=True, use_fast=False)
    long_rows, long_audit = compile_long_material(tokenizer)
    flagship_rows, flagship_audit = compile_flagships(tokenizer)
    write_rows(LONG_MATERIAL, long_rows)
    write_rows(FLAGSHIP_MATERIAL, flagship_rows)
    freeze = {
        "frozen_before_forward": True,
        "primary_object": "sample-conditioned full-coordinate sentence binding, pointer state and permutation response laws",
        "four_tasks": ["index_only", "exact_copy", "paraphrase_reorder", "semantic_sort"],
        "behavior_metrics": ["target_over_frozen_foil", "full_vocab_first_token_top1", "index_sequence_exact", "sentence_identity", "order_exact", "content_preservation", "free_generation_first_divergence"],
        "primary_routes": ["raw_full_coordinate", "pointer_state", "group_generator_composition", "residual_coordinate_response"],
        "advanced_competitors": ["S4_group_Fourier", "sentence_tensor_HOSVD", "sentence_alignment_OT", "conditional_information", "persistent_H0_topology"],
        "controls": ["source_position_sigma", "target_permutation_pi", "length", "language", "task", "content_family", "unit", "sentence_label_shuffle", "coordinate_permutation", "random_geometry"],
        "lockboxes": ["fresh units4-5", "unseen permutation compositions", "different source order", "bilingual transport", "cross-model"],
        "selection_rule": "same target rows, same sample count, discovery selection only, confirmation then untouched lockbox, MDL penalty",
        "interpretation_rule": "advanced mathematics remains diagnostic unless it predicts held-out states/operations and beats matched simpler baselines",
        "coordinate_policy": "all embedding plus every block and final-norm activation coordinate; no Top-K/PCA primary analysis",
        "automatic_continuation": "after the planned audit, continue only if the next experiment tests the same unresolved operator hypothesis with a new lockbox; otherwise record a phase boundary",
    }
    checks = {
        "long_rows": long_audit["rows"] == long_audit["expected_rows"] == 7104,
        "four_tasks": set(long_audit["task_counts"]) == {"index_only", "exact_copy", "paraphrase_reorder", "semantic_sort"},
        "all_s4": long_audit["target_permutations_index_task"] == 24,
        "long_unique": long_audit["unique_case_ids"], "long_first_token": long_audit["first_answer_token_distinct"],
        "flagship_rows": flagship_audit["rows"] == flagship_audit["expected_rows"] == 3072,
        "flagship_cells": flagship_audit["cells"] == 32, "flagship_unique": flagship_audit["unique_case_ids"],
        "flagship_first_token": flagship_audit["first_answer_token_distinct"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "evidence_audit": evidence_audit(),
              "long_material_audit": long_audit, "flagship_material_audit": flagship_audit,
              "freeze": freeze, "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "config/frozen_contract.json", freeze)
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
