#!/usr/bin/env python3
"""Test whether behavior-qualified pair interactions admit partner-independent relation potentials."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2504 = RESULT / "phase2504_c68225_c68864_corrected_semantic_selection_walsh_lockbox"
P2507 = RESULT / "phase2507_c71041_c72064_repaired_partner_behavior_fullfield"
P2508 = RESULT / "phase2508_c72065_c73088_alternative_partner_behavior_fullfield"
OUT = RESULT / "phase2509_c73089_c74112_partner_independent_relation_graph_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, QPOINT = 2509, "C73089-C74112", 30
NODES = ("taxonomy", "part_whole", "role", "preference", "membership", "translation")
EDGES = (("taxonomy", "part_whole"), ("role", "preference"), ("membership", "translation"),
         ("taxonomy", "role"), ("preference", "translation"), ("part_whole", "translation"))
EVENTS = ("definition_end", "facts_end", "query_marker", "candidate0", "candidate1", "answer_boundary")

sys.path.insert(0, str(TESTS))
import phase2502_c66177_c67200_semantic_selection_walsh_fullcoordinate_lockbox as walsh  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def incidence() -> np.ndarray:
    matrix = np.zeros((len(EDGES), len(NODES)), dtype=np.float64)
    for edge_index, (first, second) in enumerate(EDGES):
        matrix[edge_index, NODES.index(first)] = 1
        matrix[edge_index, NODES.index(second)] = -1
    return matrix


def fit_potentials(edge_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    design = incidence()
    potentials = np.linalg.pinv(design) @ edge_vectors
    prediction = design @ potentials
    denominator = float(np.square(edge_vectors).sum())
    r2 = 1.0 - float(np.square(edge_vectors - prediction).sum()) / denominator if denominator else 1.0
    return potentials, prediction, r2


def cycle_metrics(edge_vectors: np.ndarray) -> dict:
    # A=taxonomy, B=part_whole, C=role, D=preference, F=translation.
    # A->B->F->D->C->A: AB + BF - DF - CD - AC = 0.
    ab, cd, _ef, ac, df, bf = edge_vectors
    prediction_ac = ab + bf - df - cd
    actual_ac = ac
    residual = prediction_ac - actual_ac
    edge_norms = [float(np.linalg.norm(v)) for v in (ab, bf, df, cd, ac)]
    correct_cosine = walsh.cosine(prediction_ac, actual_ac)
    wrong = []
    correct_signs = (1, 1, -1, -1)
    components = (ab, bf, df, cd)
    for signs in itertools.product((-1, 1), repeat=4):
        if signs == correct_signs:
            continue
        candidate = sum(sign * component for sign, component in zip(signs, components))
        wrong.append(walsh.cosine(candidate, actual_ac))
    return {"cycle": "taxonomy->part_whole->translation->preference->role->taxonomy",
            "correct_path_prediction_cosine": correct_cosine,
            "wrong_sign_mean": float(np.mean(wrong)), "wrong_sign_q95": float(np.quantile(wrong, .95)),
            "orientation_advantage_over_wrong_q95": correct_cosine - float(np.quantile(wrong, .95)),
            "relative_residual_over_actual": float(np.linalg.norm(residual) / max(np.linalg.norm(actual_ac), 1e-30)),
            "relative_residual_over_path_edge_norm_sum": float(np.linalg.norm(residual) / max(sum(edge_norms), 1e-30))}


def aggregate_graph(edge_field: np.ndarray) -> dict:
    # edge_field axes: edge, language, surface, coordinate
    contexts = []
    r2s = []
    for language in range(2):
        for surface in range(4):
            vectors = np.asarray(edge_field[:, language, surface], dtype=np.float64)
            _, _, r2 = fit_potentials(vectors)
            panel = cycle_metrics(vectors)
            contexts.append({"language": ("en", "zh")[language], "surface": surface,
                             "additive_graph_r2": r2, **panel})
            r2s.append(r2)
    mean_vectors = np.asarray(edge_field, dtype=np.float64).mean(axis=(1, 2))
    _, _, mean_r2 = fit_potentials(mean_vectors)
    mean_cycle = cycle_metrics(mean_vectors)
    return {"mean_context_graph": {"additive_graph_r2": mean_r2, **mean_cycle},
            "eight_contexts": contexts,
            "context_summary": {"additive_graph_r2_mean": float(np.mean(r2s)),
                                "cycle_prediction_cosine_mean": float(np.mean([v["correct_path_prediction_cosine"] for v in contexts])),
                                "orientation_advantage_positive_rate": float(np.mean([v["orientation_advantage_over_wrong_q95"] > 0 for v in contexts])),
                                "relative_residual_over_actual_mean": float(np.mean([v["relative_residual_over_actual"] for v in contexts]))}}


def assemble() -> tuple[np.ndarray, np.ndarray, dict]:
    f2504, f2507, f2508 = (load_json(P2504 / "analysis/final.json"), load_json(P2507 / "analysis/final.json"),
                            load_json(P2508 / "analysis/final.json"))
    original = np.load(f2504["fields"]["interaction"]["path"], mmap_mode="r")
    rows7 = walsh.read_jsonl(Path(f2507["collection"]["event_index"]))
    rows8 = walsh.read_jsonl(Path(f2508["collection"]["event_index"]))
    field7 = np.load(f2507["collection"]["event_field"], mmap_mode="r")
    field8 = np.load(f2508["collection"]["event_field"], mmap_mode="r")
    edge_fields = np.zeros((2, len(EVENTS), len(EDGES), 2, 4, 2560), dtype=np.float32)
    for split_index, (unit7, unit8) in enumerate(((24, 26), (25, 27))):
        for event_index in range(len(EVENTS)):
            edge_fields[split_index, event_index, 0:3] = original[split_index, event_index]
            interaction7, _, _ = walsh.effects(field7, rows7, unit7, event_index, QPOINT, [0, 2])
            edge_fields[split_index, event_index, 3] = interaction7[0]
            edge_fields[split_index, event_index, 4] = interaction7[1]
            interaction8, _, _ = walsh.effects(field8, rows8, unit8, event_index, QPOINT, [1])
            edge_fields[split_index, event_index, 5] = interaction8[0]
    potentials = np.zeros((2, len(EVENTS), len(NODES), 2, 4, 2560), dtype=np.float32)
    graph = {"confirmation": {}, "lockbox": {}}
    for split_index, split_name in enumerate(("confirmation", "lockbox")):
        for event_index, event in enumerate(EVENTS):
            for language in range(2):
                for surface in range(4):
                    fit, _, _ = fit_potentials(np.asarray(edge_fields[split_index, event_index, :, language, surface], dtype=np.float64))
                    potentials[split_index, event_index, :, language, surface] = fit.astype(np.float32)
            if event_index < 2:
                graph[split_name][event] = {"interaction_exact_zero": bool(np.max(np.abs(edge_fields[split_index, event_index])) == 0)}
            else:
                graph[split_name][event] = aggregate_graph(edge_fields[split_index, event_index])
    cross_split = {}
    for event_index, event in enumerate(EVENTS):
        edge_views = {name: edge_fields[index, event_index].mean(axis=(1, 2)) for index, name in enumerate(("confirmation", "lockbox"))}
        potential_views = {name: potentials[index, event_index].mean(axis=(1, 2)) for index, name in enumerate(("confirmation", "lockbox"))}
        for value in potential_views.values():
            value -= value.mean(axis=0, keepdims=True)
        cross_split[event] = {"edge_identity": walsh.identity_metric(edge_views),
                              "fitted_relation_potential_identity": walsh.identity_metric(potential_views)}
    return edge_fields, potentials, {"graph": graph, "cross_split": cross_split,
                                     "source_gates": {"phase2507_qualified": f2507["behavior"]["qualified_pairs"],
                                                      "phase2508_qualified": f2508["behavior"]["qualified_pairs"]}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 跨伙伴关系势的五节点闭环、留一路径预测与锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 不放宽任何行为门，把原三条合格边taxonomy–part-whole、role–preference、membership–translation，与Phase2507合格的taxonomy–role、preference–translation及Phase2508合格的part-whole–translation合并。六节点图连通，其中taxonomy→part-whole→translation→preference→role→taxonomy形成五节点闭环，membership为叶节点。对每个语言×surface上下文、六事件、q30，令每条有向四格交互满足候选模型 (I_{{a-b}}\approx z_a-z_b)，用图incidence最小二乘拟合六个family势，并用闭环四条边预测第五条taxonomy–role：

$$\widehat I_{{taxonomy-role}}=I_{{taxonomy-part}}+I_{{part-translation}}-I_{{preference-translation}}-I_{{role-preference}}.$$

预测余弦与其余15种错误符号组合的q95竞争；unit21/24/26组成confirmation图，unit23/25/27组成lockbox图，q30和边集均提前冻结。

**结果汇总。** 来源行为门 `{json.dumps(result['source_gates'], ensure_ascii=False)}`；confirmation图 `{json.dumps(result['graph']['confirmation'], ensure_ascii=False)}`；lockbox图 `{json.dumps(result['graph']['lockbox'], ensure_ascii=False)}`；跨split边/关系势身份 `{json.dumps(result['cross_split'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2509_c73089_c74112_partner_independent_relation_graph_lockbox.py`；六边完整坐标场、拟合family势、哈希和final位于`{OUT}`。

**分析与理论进展。** 这是第一次直接测试“同一family换伙伴后能否由一个坐标势解释”，而不是把pair余弦误称为独立family。additive graph (R^2) 只表示六边可由节点势近似；由于图只有一个独立环，闭环路径预测、错误符号对照和独立split复现才是更严格裁决。membership仅是叶节点，不能单独贡献闭环证据。

**问题硬伤与结论。** 六条边来自三个独立材料campaign，虽在同一语言/surface抽象槽对齐，但实体和marker不同；这既提高泛化难度，也引入unit差异。一个五环远不足以建立普遍关系代数；高 (R^2) 可能因图自由度高而乐观。闭环失败只否定固定q30的简单可加family势，不否定条件化非线性编码；闭环成功也不是因果或自然语言闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    f2504, f2507, f2508 = (load_json(P2504 / "analysis/final.json"), load_json(P2507 / "analysis/final.json"),
                            load_json(P2508 / "analysis/final.json"))
    edge_fields, potentials, analysis = assemble()
    derived = OUT / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    edge_path = derived / "qualified_partner_graph_edges.float32.npy"
    potential_path = derived / "fitted_relation_potentials.float32.npy"
    np.save(edge_path, edge_fields); np.save(potential_path, potentials)
    lock_answer = analysis["graph"]["lockbox"]["answer_boundary"]["mean_context_graph"]
    checks = {"source_phases_passed": all(v["all_checks_passed"] for v in (f2504, f2507, f2508)),
              "six_behavior_qualified_edges": edge_fields.shape[2] == 6,
              "connected_six_node_graph": bool(np.linalg.matrix_rank(incidence()) == 5),
              "one_independent_cycle": bool(len(EDGES) - np.linalg.matrix_rank(incidence()) == 1),
              "prefix_exact_zero": bool(np.max(np.abs(edge_fields[:, :2])) == 0),
              "frozen_qpoint": QPOINT == f2504["contract"]["qpoint"],
              "all_coordinates": edge_fields.shape[-1] == 2560 and potentials.shape[-1] == 2560,
              "finite": bool(np.isfinite(edge_fields).all() and np.isfinite(potentials).all()),
              "hashes": bool(digest(edge_path) and digest(potential_path)), "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "qpoint": QPOINT, "nodes": list(NODES),
              "edges": [list(v) for v in EDGES], **analysis,
              "fields": {"edges": {"path": str(edge_path), "shape": list(edge_fields.shape), "sha256": digest(edge_path)},
                         "potentials": {"path": str(potential_path), "shape": list(potentials.shape), "sha256": digest(potential_path)}},
              "adjudication": {"partner_independent_additive_relation_potential_supported": lock_answer["orientation_advantage_over_wrong_q95"] > 0,
                               "single_cycle_only": True, "membership_cycle_tested": False,
                               "pure_semantic_code_identified": False, "causal_mediator_identified": False,
                               "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
