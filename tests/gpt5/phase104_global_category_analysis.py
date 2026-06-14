#!/usr/bin/env python3
"""
Phase 104: global category analysis from GLM5 Phase 483/484 artifacts.

This script does not run model inference. It assembles existing confirmed
category-boundary results into a global map:
  - category-layer map
  - competition/release graph
  - MLP writer concentration and causal status
  - relation-slot invariance
  - anomaly explanations

Usage:
  python tests/gpt5/phase104_global_category_analysis.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GLM_RESULTS = ROOT / "results" / "glm5"
OUT_DIR = ROOT / "results" / "gpt5"

MODELS = ["qwen3", "glm4", "deepseek7b"]
CATEGORIES = ["fruit", "animal", "tool", "vehicle", "clothing", "furniture", "food", "plant"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rnd(x: Any, nd: int = 3) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    return x


def edge_strength_class(delta: float) -> str:
    if delta >= 5:
        return "strong"
    if delta >= 1:
        return "medium"
    if delta > 0:
        return "weak"
    return "none"


def writer_class(cos50: float | None, sig_count: int | None, causal_cos: float | None) -> str:
    if causal_cos is not None and causal_cos >= 0.85:
        return "mlp_causal_writer"
    if causal_cos is not None and causal_cos < 0:
        return "non_mlp_or_opposed"
    if cos50 is not None and cos50 >= 0.55 and sig_count is not None and sig_count <= 300:
        return "concentrated_candidate"
    if cos50 is not None and cos50 < 0.35:
        return "distributed_or_missing"
    return "mixed_or_unresolved"


def load_all() -> dict[str, dict[str, dict[str, Any]]]:
    data: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODELS:
        data[model] = {
            "p483_r1": load_json(GLM_RESULTS / f"phase483_{model}_r1.json"),
            "p483_r2": load_json(GLM_RESULTS / f"phase483_{model}_r2.json"),
            "p484_r1": load_json(GLM_RESULTS / f"phase484_{model}_r1.json"),
            "p484_r2": load_json(GLM_RESULTS / f"phase484_{model}_r2.json"),
        }
    return data


def build_layer_map(data: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    layer_map: dict[str, Any] = {}
    for model in MODELS:
        release_detail = data[model]["p483_r1"]["exp2_competition_release"]["release_detail"]
        model_map = {}
        for cat in CATEGORIES:
            detail = release_detail[cat]
            model_map[cat] = {
                "best_layer": detail["best_layer"],
                "target_delta": rnd(detail["target_delta"]),
                "selectivity": rnd(detail["selectivity"]),
                "spec_norm": rnd(detail["spec_norm"]),
            }
        layer_map[model] = model_map
    return layer_map


def build_competition_graph(data: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    graph_by_model: dict[str, Any] = {}
    edge_counts: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"models": [], "deltas": []})

    for model in MODELS:
        release_detail = data[model]["p483_r1"]["exp2_competition_release"]["release_detail"]
        model_edges = []
        for removed in CATEGORIES:
            detail = release_detail[removed]
            for released, delta in detail["dcf_delta"].items():
                if released == removed or delta <= 0:
                    continue
                item = {
                    "removed": removed,
                    "released": released,
                    "delta": rnd(delta),
                    "strength": edge_strength_class(delta),
                    "removed_target_delta": rnd(detail["target_delta"]),
                }
                model_edges.append(item)
                edge_counts[(removed, released)]["models"].append(model)
                edge_counts[(removed, released)]["deltas"].append(delta)
        model_edges.sort(key=lambda x: x["delta"], reverse=True)
        graph_by_model[model] = {
            "edges_positive": model_edges,
            "top_edges": model_edges[:12],
            "out_degree_positive": {
                cat: sum(1 for e in model_edges if e["removed"] == cat) for cat in CATEGORIES
            },
        }

    cross_model_edges = []
    for (removed, released), info in edge_counts.items():
        models = info["models"]
        deltas = info["deltas"]
        cross_model_edges.append({
            "removed": removed,
            "released": released,
            "model_count": len(set(models)),
            "models": sorted(set(models), key=MODELS.index),
            "avg_positive_delta": rnd(sum(deltas) / len(deltas)),
            "max_delta": rnd(max(deltas)),
        })
    cross_model_edges.sort(key=lambda x: (x["model_count"], x["avg_positive_delta"]), reverse=True)

    return {
        "by_model": graph_by_model,
        "cross_model_positive_edges": cross_model_edges,
        "cross_model_stable_edges": [e for e in cross_model_edges if e["model_count"] >= 2],
        "universal_positive_edges": [e for e in cross_model_edges if e["model_count"] == 3],
    }


def build_writer_map(data: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    writer_map: dict[str, Any] = {}
    for model in MODELS:
        recon = data[model]["p484_r1"]["exp1_writer_reconstruction"]
        causal = data[model]["p484_r1"]["exp2_writer_causal_test"]
        r2 = data[model]["p484_r2"]["confirm_results"]
        model_map = {}
        for cat in sorted(set(recon) | set(causal)):
            rec = recon.get(cat, {})
            cau = causal.get(cat, {})
            k5 = cau.get("ablation_results", {}).get("k=5", {})
            r2_cat = r2.get(cat, {})
            r2_k5 = r2_cat.get("ablation_results", {}).get("k=5", {}) if isinstance(r2_cat, dict) else {}
            causal_cos = r2_k5.get("cos_with_direction_remove", k5.get("cos_with_direction_remove"))
            sig = rec.get("neuron_contribution_stats", {}).get("n_significant")
            cos50 = rec.get("cos_at_k", {}).get("50")
            model_map[cat] = {
                "best_layer": rec.get("best_layer", cau.get("best_layer")),
                "cos_at_10": rnd(rec.get("cos_at_k", {}).get("10")),
                "cos_at_50": rnd(cos50),
                "cos_at_200": rnd(rec.get("cos_at_k", {}).get("200")),
                "energy_at_50": rnd(rec.get("energy_at_k", {}).get("50")),
                "n_significant": sig,
                "direction_remove_target": rnd(cau.get("direction_remove_target")),
                "k5_target_delta": rnd(r2_k5.get("target_delta", k5.get("target_delta"))),
                "k5_cos_with_direction_remove": rnd(causal_cos),
                "writer_class": writer_class(cos50, sig, causal_cos),
            }
        writer_map[model] = model_map
    return writer_map


def build_relation_map(data: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    relation_map: dict[str, Any] = {}
    for model in MODELS:
        r2 = data[model]["p484_r2"]["confirm_results"].get("relation_invariance", {})
        raw = r2.get("relation_raw", {})
        deltas = []
        for rel, item in raw.items():
            if "target_delta" in item:
                deltas.append(item["target_delta"])
            elif "injection_delta" in item:
                deltas.append(item["injection_delta"])
        relation_map[model] = {
            "category": r2.get("category"),
            "best_layer": r2.get("best_layer"),
            "relations": {k: {kk: rnd(vv) for kk, vv in v.items()} for k, v in raw.items()},
            "delta_range": rnd(max(deltas) - min(deltas)) if deltas else None,
            "status": "exact_or_near_invariant" if deltas and (max(deltas) - min(deltas)) < 0.01 else "needs_scale_audit",
        }
    return relation_map


def build_anomaly_map(data: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    anomaly_map: dict[str, Any] = {}
    for model in MODELS:
        anom = data[model]["p484_r1"]["exp4_anomalous_competition"]
        model_map = {}
        for edge, item in anom.items():
            attrs = []
            for attr, vals in item.get("shared_attribute_analysis", {}).items():
                if isinstance(vals, dict):
                    changes = [v.get("delta", 0.0) for v in vals.values() if isinstance(v, dict)]
                    if changes:
                        attrs.append({
                            "attribute": attr,
                            "avg_delta": rnd(sum(changes) / len(changes)),
                            "max_delta": rnd(max(changes)),
                        })
            attrs.sort(key=lambda x: x["avg_delta"], reverse=True)
            model_map[edge] = {
                "best_layer": item.get("best_layer"),
                "cos_between_boundaries": rnd(item.get("cos_between_boundaries")),
                "top_released_attributes": attrs[:4],
            }
        anomaly_map[model] = model_map
    return anomaly_map


def synthesize_findings(layer_map: dict[str, Any], graph: dict[str, Any], writer_map: dict[str, Any],
                        relation_map: dict[str, Any], anomaly_map: dict[str, Any]) -> list[str]:
    findings = []
    findings.append(
        "全局图谱支持'类别=共享语义流形+类别边界残差+竞争释放'，但写入机制不是统一的单一模块。"
    )
    universal = graph["universal_positive_edges"]
    findings.append(
        "跨三模型都为正的释放边包括: "
        + ", ".join(f"{e['removed']}->{e['released']}" for e in universal[:10])
        + "。这些边更像稳定竞争骨架。"
    )
    findings.append(
        "Qwen3 的强释放边幅度最大，GLM4 幅度整体很小，DS7B 存在方向不干净和抑制性神经元问题。"
    )
    findings.append(
        "MLP 因果写入器只在局部类别中清晰出现: Qwen3 clothing、GLM4 fruit 最强；fruit/animal 等类别常表现为非 MLP 主导或反向。"
    )
    findings.append(
        "类别最佳层位不是统一层: Qwen3 多在 L23-L34，GLM4 多在 L27-L39，DS7B 多在 L23-L27，说明边界有类别-模型特异发育时间。"
    )
    findings.append(
        "关系槽位测试显示 B_c 读出跨 kind_of/used_for/found_in 基本不变，但 scale=1.0 可能过强，必须做小尺度复核。"
    )
    findings.append(
        "food->vehicle、animal->clothing 不是简单错误边，更可能由属性共享/压制释放产生；但 DS7B 的异常边不能作为干净证据。"
    )
    return findings


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Phase 104 Global Category Analysis")
    lines.append("")
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("## Core Findings")
    for item in result["core_findings"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Universal Positive Competition Edges")
    for e in result["competition_graph"]["universal_positive_edges"]:
        lines.append(
            f"- {e['removed']} -> {e['released']}: models={','.join(e['models'])}, "
            f"avg_delta={e['avg_positive_delta']}, max_delta={e['max_delta']}"
        )
    lines.append("")
    lines.append("## Top Model Edges")
    for model in MODELS:
        lines.append(f"### {model}")
        for e in result["competition_graph"]["by_model"][model]["top_edges"][:8]:
            lines.append(f"- {e['removed']} -> {e['released']}: delta={e['delta']} ({e['strength']})")
    lines.append("")
    lines.append("## MLP Writer Map")
    for model in MODELS:
        lines.append(f"### {model}")
        for cat, item in result["writer_map"][model].items():
            lines.append(
                f"- {cat}: {item['writer_class']}, L{item['best_layer']}, "
                f"cos50={item['cos_at_50']}, sig={item['n_significant']}, "
                f"k5_cos={item['k5_cos_with_direction_remove']}"
            )
    lines.append("")
    lines.append("## Layer Map")
    for model in MODELS:
        vals = result["layer_map"][model]
        compact = ", ".join(f"{cat}:L{vals[cat]['best_layer']}" for cat in CATEGORIES)
        lines.append(f"- {model}: {compact}")
    lines.append("")
    lines.append("## Hard Limits")
    for item in result["hard_limits"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = load_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    layer_map = build_layer_map(data)
    graph = build_competition_graph(data)
    writer_map = build_writer_map(data)
    relation_map = build_relation_map(data)
    anomaly_map = build_anomaly_map(data)

    result = {
        "phase": 104,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_artifacts": [
            str(GLM_RESULTS / f"phase483_{m}_r1.json") for m in MODELS
        ] + [
            str(GLM_RESULTS / f"phase483_{m}_r2.json") for m in MODELS
        ] + [
            str(GLM_RESULTS / f"phase484_{m}_r1.json") for m in MODELS
        ] + [
            str(GLM_RESULTS / f"phase484_{m}_r2.json") for m in MODELS
        ],
        "layer_map": layer_map,
        "competition_graph": graph,
        "writer_map": writer_map,
        "relation_map": relation_map,
        "anomaly_map": anomaly_map,
        "core_findings": [],
        "hard_limits": [
            "本轮没有重新运行模型，只整合 Phase 483/484 既有结果；结论是全局拼图，不是新因果实验。",
            "类别只有 8 类，每类 8 个对象；足以看边界网络雏形，不足以证明完整语义大陆。",
            "DCF 词表仍可能造成候选集偏置，尤其 food->vehicle、animal->clothing 等异常边需要更宽属性词表复核。",
            "关系不变性使用 scale=1.0 注入，可能覆盖关系模板差异，必须用更小 scale 做下一轮。",
            "MLP writer 只覆盖 fruit/animal/clothing 三类的 Phase 484 重构，其他五类仍缺少写入器级证据。",
        ],
        "next_stage_tasks": [
            "扩展类别到至少 32 类，每类不少于 24 个对象，优先覆盖自然物、人造物、生物、身体、地点、抽象概念。",
            "为每个类别建立 Category-Layer Map: 形成层、选择性层、移除最大层、竞争释放最大层。",
            "对全类别做 scale sweep: 0.05/0.1/0.2/0.5/1.0，确认关系槽位不变性不是强注入假象。",
            "把 MLP、attention output、residual route 分开测试，寻找 fruit/animal 等非 MLP 主导边界的真正写入器。",
            "对异常边做属性级宽词表审计，区分真实共享属性释放和 DCF 候选集偏置。",
        ],
    }
    result["core_findings"] = synthesize_findings(layer_map, graph, writer_map, relation_map, anomaly_map)

    json_path = OUT_DIR / "phase104_global_category_analysis.json"
    md_path = OUT_DIR / "phase104_global_category_analysis.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, md_path)

    print(f"[ok] wrote {json_path}")
    print(f"[ok] wrote {md_path}")
    print("[summary]")
    for item in result["core_findings"]:
        print(f"- {item}")


if __name__ == "__main__":
    main()
