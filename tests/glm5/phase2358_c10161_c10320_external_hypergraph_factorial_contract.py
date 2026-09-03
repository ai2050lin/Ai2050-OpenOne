#!/usr/bin/env python3
"""Audit Phase 2351-2357 claims and freeze a bilingual typed-hypergraph factorial contract."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase2358_c10161_c10320_external_hypergraph_factorial_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = OUT / "material/bilingual_typed_hypergraph_factorial.jsonl"
PHASE = 2358
CAMPAIGN = "C10161-C10320"
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\f237d85f-1c3f-4c54-81a3-ac0fc7f28547\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\583fafc5-7b11-4b02-b0ae-31a9655de748\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\4dc69f72-0be1-4360-8fda-a0645a2e4afd\pasted-text.txt"),
)
FAMILIES = (
    "taxonomy", "attribute", "attitude", "grammar", "coreference", "translation",
    "causal", "temporal", "spatial", "possession", "partwhole", "negation",
)
FAMILY_PARTITIONS = {
    **{name: "family_discovery" for name in FAMILIES[:8]},
    **{name: "family_confirmation" for name in FAMILIES[8:10]},
    **{name: "whole_family_lockbox" for name in FAMILIES[10:]},
}
UNIT_PARTITIONS = {
    **{i: "unit_discovery" for i in range(4)},
    **{i: "unit_confirmation" for i in range(4, 6)},
    **{i: "fresh_unit_lockbox" for i in range(6, 8)},
}
FACTOR_NAMES = ("lexical_realization", "relation_variant", "branch_edge", "conflict_edge", "query_role")
LANGUAGES = ("en", "zh")
UNITS = 8
CELLS = 32

FAMILY_SPEC = {
    "taxonomy": ("concept", ("belongs to", "is classified as"), ("属于", "被归类为")),
    "attribute": ("attribute", ("has the property", "is described as"), ("具有属性", "被描述为")),
    "attitude": ("relation", ("likes", "prefers"), ("喜欢", "偏爱")),
    "grammar": ("grammar", ("modifies", "governs"), ("修饰", "支配")),
    "coreference": ("context", ("refers to", "identifies"), ("指代", "标识")),
    "translation": ("translation", ("translates to", "means"), ("翻译为", "意为")),
    "causal": ("relation", ("causes", "enables"), ("导致", "促成")),
    "temporal": ("context", ("precedes", "happens before"), ("先于", "发生在之前")),
    "spatial": ("relation", ("is north of", "is beyond"), ("位于北侧", "位于远侧")),
    "possession": ("relation", ("owns", "keeps"), ("拥有", "保管")),
    "partwhole": ("concept", ("is part of", "is a component of"), ("是其部分", "是其组件")),
    "negation": ("context", ("excludes", "blocks"), ("排除", "阻断")),
}

# Natural two-word names are deliberately broad, not a single sentence or one semantic domain.
WORDS_A = (
    "Orchard Apple", "Citrus Fruit", "Forest Plant", "Material Object",
    "Silver Harbor", "Quiet Meadow", "Amber Lantern", "Copper Bridge",
    "Winter Cedar", "Summer River", "Northern Vale", "Southern Field",
    "Patient Scholar", "Curious Artist", "Gentle Baker", "Skilled Weaver",
)
WORDS_B = (
    "Maple Crown", "Olive Stone", "Hazel Grove", "Indigo Lake",
    "Dawn Signal", "Evening Bell", "Rapid Current", "Calm Shelter",
    "Ancient Script", "Modern Phrase", "Subject Marker", "Object Marker",
    "Crimson Petal", "Golden Seed", "Granite Arch", "Willow Path",
)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def bit_vector(cell: int) -> list[int]:
    return [(cell >> bit) & 1 for bit in range(5)]


def node_names(family_index: int, unit: int, lexical: int) -> list[str]:
    pool = WORDS_A if lexical == 0 else WORDS_B
    offset = family_index * 5 + unit * 3
    return [pool[(offset + k * 3) % len(pool)] for k in range(8)]


def compile_material(tokenizer) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    graph_pair_hashes: dict[tuple, set[str]] = defaultdict(set)
    for family_index, family in enumerate(FAMILIES):
        category, rel_en, rel_zh = FAMILY_SPEC[family]
        for unit in range(UNITS):
            depth = 2 + unit % 3
            surface = "enumerated" if unit % 2 == 0 else "independent_prose"
            author = "template_author_a" if unit % 2 == 0 else "template_author_b"
            for language in LANGUAGES:
                rels = rel_en if language == "en" else rel_zh
                for cell in range(CELLS):
                    lexical, relation_variant, branch, conflict, query_role = bit_vector(cell)
                    nodes = node_names(family_index, unit, lexical)
                    relation = rels[relation_variant]
                    main_edges = [
                        {"source": nodes[k], "relation": relation, "target": nodes[k + 1],
                         "edge_type": "main_path", "truth": True, "order": k}
                        for k in range(depth)
                    ]
                    extra_edges: list[dict] = []
                    if branch:
                        extra_edges.append({"source": nodes[1], "relation": relation, "target": nodes[depth + 1],
                                            "edge_type": "branch", "truth": True})
                    if conflict:
                        extra_edges.append({"source": nodes[0], "relation": relation, "target": nodes[depth + 2],
                                            "edge_type": "explicit_false_conflict", "truth": False})
                    # Unit-specific topology broadens the atlas without changing the five controlled factors.
                    if not branch and not conflict and unit % 4 == 2:
                        extra_edges.append({"source": nodes[depth], "relation": relation, "target": nodes[0],
                                            "edge_type": "cycle_context", "truth": True})
                    elif not branch and not conflict and unit % 4 == 3:
                        extra_edges.append({"source": nodes[depth + 1], "relation": relation,
                                            "target": nodes[depth + 2], "edge_type": "irrelevant", "truth": True})
                    all_edges = main_edges + extra_edges
                    query = "first" if query_role == 0 else "terminal"
                    target = nodes[1] if query == "first" else nodes[depth]
                    foil = nodes[depth + 2]
                    if language == "en":
                        if surface == "enumerated":
                            facts = " ".join(
                                f"Main step {k + 1}: {edge['source']} {edge['relation']} {edge['target']}."
                                for k, edge in enumerate(main_edges)
                            )
                        else:
                            facts = "A field note traces the main route: " + "; then ".join(
                                f"{edge['source']} {edge['relation']} {edge['target']}" for edge in main_edges
                            ) + "."
                        if branch:
                            e = extra_edges[0]
                            facts += f" A side branch says {e['source']} {e['relation']} {e['target']}; it is not on the main route."
                        if conflict:
                            e = [x for x in extra_edges if not x["truth"]][0]
                            facts += f" The claim '{e['source']} {e['relation']} {e['target']}' is explicitly false and must be ignored."
                        if any(e["edge_type"] == "cycle_context" for e in extra_edges):
                            facts += " A return link closes a cycle after the main endpoint; do not continue past the endpoint."
                        if any(e["edge_type"] == "irrelevant" for e in extra_edges):
                            facts += " A separate unrelated link is also mentioned; ignore it."
                        question = ("Which exact name is reached by the first main step?" if query == "first"
                                    else "Which exact name is the endpoint of the stated main route?")
                        prompt = f"{facts}\n{question}\nReply with only the exact two-word name and stop.\nAnswer:"
                    else:
                        if surface == "enumerated":
                            facts = "".join(
                                f"主路径第{k + 1}步：{edge['source']}{edge['relation']}{edge['target']}。"
                                for k, edge in enumerate(main_edges)
                            )
                        else:
                            facts = "一份记录按主路径依次写道：" + "；随后".join(
                                f"{edge['source']}{edge['relation']}{edge['target']}" for edge in main_edges
                            ) + "。"
                        if branch:
                            e = extra_edges[0]
                            facts += f" 旁支称{e['source']}{e['relation']}{e['target']}，它不属于主路径。"
                        if conflict:
                            e = [x for x in extra_edges if not x["truth"]][0]
                            facts += f" “{e['source']}{e['relation']}{e['target']}”被明确标为假，必须忽略。"
                        if any(e["edge_type"] == "cycle_context" for e in extra_edges):
                            facts += " 主路径终点后另有回边形成环，但不要越过终点。"
                        if any(e["edge_type"] == "irrelevant" for e in extra_edges):
                            facts += " 另有一条无关连接，必须忽略。"
                        question = "主路径第一步到达的精确名称是什么？" if query == "first" else "给定主路径的终点精确名称是什么？"
                        prompt = f"{facts}\n{question}\n只回答精确的双词名称，然后停止。\n答案："

                    abstract = {
                        "family": family, "category": category, "depth": depth,
                        "factor_values": {name: value for name, value in zip(FACTOR_NAMES, bit_vector(cell))},
                        "typed_nodes": [
                            {"id": f"entity_{k}", "type": "entity"} for k in range(depth + 3)
                        ] + [
                            {"id": "relation", "type": "relation"},
                            {"id": query, "type": "query_role"},
                            {"id": "answer", "type": "output_role"},
                        ],
                        "typed_edges": [
                            {"source": f"entity_{e['order']}", "relation": "relation",
                             "target": f"entity_{e['order'] + 1}", "type": "main_path", "truth": True}
                            for e in main_edges
                        ] + [
                            {"source": "condition", "relation": e["edge_type"], "target": "main_path",
                             "type": "context_hyperedge", "truth": e["truth"]} for e in extra_edges
                        ],
                    }
                    abstract_hash = hashlib.sha256(json.dumps(abstract, sort_keys=True).encode()).hexdigest()
                    prompt_ids = [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]
                    target_ids = [int(x) for x in tokenizer.encode(" " + target, add_special_tokens=False)]
                    foil_ids = [int(x) for x in tokenizer.encode(" " + foil, add_special_tokens=False)]
                    if not target_ids or not foil_ids or target_ids[0] == foil_ids[0]:
                        raise RuntimeError(("non_distinct_first_token", family, unit, language, cell, target, foil,
                                            target_ids[:2], foil_ids[:2]))
                    row = {
                        "case_id": f"c10161-{family}-u{unit:02d}-{language}-c{cell:02d}",
                        "design_index": len(rows), "family": family, "family_index": family_index,
                        "category": category, "family_partition": FAMILY_PARTITIONS[family],
                        "unit": unit, "unit_partition": UNIT_PARTITIONS[unit], "language": language,
                        "surface": surface, "author": author, "cell": cell, "bits": bit_vector(cell),
                        "factors": {name: value for name, value in zip(FACTOR_NAMES, bit_vector(cell))},
                        "query": query, "depth": depth, "topology": sorted({e["edge_type"] for e in all_edges}),
                        "edge_count": len(all_edges), "graph": all_edges, "abstract_graph": abstract,
                        "abstract_graph_hash": abstract_hash, "prompt": prompt, "prompt_ids": prompt_ids,
                        "target": target, "foil": foil, "target_ids": target_ids, "foil_ids": foil_ids,
                        "target_first_id": target_ids[0], "foil_first_id": foil_ids[0],
                    }
                    rows.append(row)
                    graph_pair_hashes[(family, unit, cell)].add(abstract_hash)

    edge_counts = Counter(row["edge_count"] for row in rows)
    audit = {
        "rows": len(rows), "expected_rows": len(FAMILIES) * UNITS * len(LANGUAGES) * CELLS,
        "families": len(FAMILIES), "categories": sorted({row["category"] for row in rows}),
        "factor_names": list(FACTOR_NAMES), "full_factorial_cells": sorted({row["cell"] for row in rows}),
        "languages": list(LANGUAGES), "units": UNITS, "surfaces": sorted({row["surface"] for row in rows}),
        "authors": sorted({row["author"] for row in rows}), "queries": Counter(row["query"] for row in rows),
        "edge_count_distribution": dict(sorted(edge_counts.items())),
        "edge_range": [min(edge_counts), max(edge_counts)],
        "topologies": sorted({kind for row in rows for kind in row["topology"]}),
        "cross_language_abstract_graph_shared": all(len(values) == 1 for values in graph_pair_hashes.values()),
        "unique_case_ids": len({row["case_id"] for row in rows}) == len(rows),
        "first_answer_token_distinct": all(row["target_first_id"] != row["foil_first_id"] for row in rows),
        "prompt_token_range": [min(len(row["prompt_ids"]) for row in rows), max(len(row["prompt_ids"]) for row in rows)],
        "family_partitions": Counter(row["family_partition"] for row in rows),
        "unit_partitions": Counter(row["unit_partition"] for row in rows),
    }
    return rows, json.loads(json.dumps(audit, ensure_ascii=False, default=dict))


def evidence_audit() -> dict:
    return {
        "attachment_sha256": [{"path": str(path), "sha256": sha256(path)} for path in ATTACHMENTS],
        "retained": [
            "Phase2351 correctly decomposed raw exact=0: first-line identifier accuracy was 0.912109.",
            "Phase2352 established a 6144-row prompt field and a genuine model-generated token trajectory, while teacher-forced preference (1.0) exceeded autonomous first-line exactness (0.515625).",
            "Phase2353 supports a physical-address-dependent q24 signed prompt residual; its post-first-token minimum transfer gate did not pass.",
            "Phase2354 removed one norm-scale confound and found no selective family-mean causal candidate.",
            "Phase2355 strictly replicated a descriptive functional gate in Qwen14B and GLM4, with DeepSeek only partial.",
            "Phase2357 correctly withdrew the universal generation-success marker because BA_query-only=0.940559 exceeded BA_coordinate=0.725524.",
            "Failure of a deletion/rescue gate is evidence against that intervention/model pair, not evidence that no distributed encoding structure exists.",
        ],
        "corrected_overclaims": [
            "The third attachment misnumbers Phase2351-2355; conclusions are reconciled to the repository MEMO and final.json records.",
            "Target-versus-one-foil teacher preference does not prove that the model 'knows' or semantically understands the answer.",
            "Phase2354 measured teacher-forced target-versus-foil margins, not autonomous generation; it did not show that a small perturbation destroys generation.",
            "A failed family-mean norm-preserving direction does not refute static coordinates, strength texture, energy routing, or every possible gear hypothesis at once.",
            "Phase2356 confounded generation-success prediction; it did not invalidate Phase2353 family classification, which explicitly removed query main effects.",
            "One failed step>0 transfer route does not establish that post-token state is no longer a function of knowledge, nor that generation-time observation is useless.",
            "Cross-model coordinate numbers are not aligned, but functional cross-language/cross-model replication did not globally fail.",
            "Prompt activations are activation coordinates, not trained parameter coordinates; the visual client must label them accordingly.",
        ],
        "evidence_ladder": {
            "L0": "behavior/material qualification", "L1": "full-coordinate observation",
            "L2": "paired/factorial structure", "L3": "held-out prediction",
            "L4": "dynamic response", "L5": "selective causality", "L6": "cross-model equivalence",
            "rule": "Failure at L5 never deletes independently valid L1-L4 evidence.",
        },
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    audit = result["evidence_audit"]
    material = result["material_audit"]
    text = rf"""

## Phase {PHASE}: 证据纠错、外部有类型语言超图与五因子全析因合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本Phase先逐项核对三份附件与Phase2351–2357原始记录，再把“存在某种编码机制”降格为可检验工作假设，而非预设结论。外部对象冻结为有类型超图

$$
\mathcal G_L=(V,E,\tau,\rho,C,Q,Y),\qquad
H_{{i,t,q,j}}=\Psi_M(\mathcal G_L,\text{{language}},\text{{surface}})_{{t,q,j}}.
$$

材料覆盖12族、概念/属性/关系/语法/上下文/翻译六大类、8个词汇unit、中英、两位独立表述作者、2–6边、链/支路/环/冲突/无关边及5个严格二值因素，共{material['rows']}条。每个抽象图在中英文共享同一哈希；unit与whole-family锁箱在模型前冻结。

**结果汇总。** 附件哈希与保留结论 `{json.dumps(audit['retained'], ensure_ascii=False)}`；修正的过度结论 `{json.dumps(audit['corrected_overclaims'], ensure_ascii=False)}`；材料审计 `{json.dumps(material, ensure_ascii=False)}`；冻结合同 `{json.dumps(result['freeze'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2358_c10161_c10320_external_hypergraph_factorial_contract.py`；材料与审计 `tests/glm5/result/phase2358_c10161_c10320_external_hypergraph_factorial_contract`；未新增其他Markdown。

**理论进展、问题硬伤与结论。** 正确保留的是“行为合格域中存在依赖具体激活坐标地址的条件化prompt准备场”，而不是“模型已经知道答案”或“齿轮已闭合”。第三份附件的Phase编号、自由生成因果表述、step>0本体结论和query混淆外推均过度。新合同不把删除、救援或完整闭环设为观察结构的前置门；但外部超图仍只是实验坐标系，不能反向证明模型内部真的存储同构超图。下一Phase按冻结顺序采集全部embedding+HiddenState物理坐标，不用Top-K/PCA作为主分析。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists() and MATERIAL.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    sys.path.insert(0, str(TESTS))
    import model_utils
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                               local_files_only=True, use_fast=False)
    rows, material_audit = compile_material(tokenizer)
    write_rows(MATERIAL, rows)
    freeze = {
        "frozen_before_model_forward": True, "working_hypothesis": "finite parameters implement reusable language regularities",
        "primary_priority": "full-coordinate family atlas and predictive regularities",
        "secondary_priority": "causal closure after structural prediction",
        "coordinate_policy": "all physical activation coordinates; no Top-K/PCA primary representation",
        "factor_order": list(FACTOR_NAMES), "cell_index": "sum(bit_k * 2**k)",
        "unit_split": UNIT_PARTITIONS, "family_split": FAMILY_PARTITIONS,
        "behavior_gate": "target first-token logit exceeds a frozen foil; qualification only, not knowledge proof",
        "next_phases": [
            "2359 Qwen3-4B full field", "2360 factorial/full-coordinate route scan",
            "2361 token/layer dynamics and coordinate cooperation", "2362 held-out composition prediction",
            "2363 balanced autonomous realization trajectory", "2364 sequential cross-model functional replication",
            "2365 audit, visualization verification, cleanup and automatic next-stage adjudication",
        ],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "evidence_audit": evidence_audit(),
        "material_audit": material_audit, "freeze": freeze,
        "checks": {
            "rows": material_audit["rows"] == 6144,
            "five_factor_cube": material_audit["full_factorial_cells"] == list(range(32)),
            "six_or_more_categories": len(material_audit["categories"]) >= 6,
            "edge_range": material_audit["edge_range"] == [2, 6],
            "cross_language_graph": material_audit["cross_language_abstract_graph_shared"],
            "unique": material_audit["unique_case_ids"],
            "answer_tokens": material_audit["first_answer_token_distinct"],
        },
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(OUT / "config/frozen_contract.json", freeze)
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
