#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE = 941
ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "tests" / "result" / "phase941_color_feature_neuron_atlas"
OUT_DIR = RESULT_ROOT / "cross_model_atlas"
ANALYSIS_DOC = ROOT / "research" / "MainAnalysis" / "20260705_02_Phase941三模型颜色编码特征图谱.md"

ROUNDS = [
    ("qwen3", "color_feature_neuron_atlas_qwen3_full"),
    ("deepseek7b", "color_feature_neuron_atlas_deepseek7b_full"),
    ("glm4", "color_feature_neuron_atlas_glm4_full"),
]

STABLE_MIN_OBJECTS = 3
STABLE_MIN_TEMPLATES = 3
STABLE_MIN_COUNT = 8
FACTORS = ["0.0", "0.5", "1.5", "2.0"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def finite(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def stable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if int(row.get("coverage_objects") or 0) >= STABLE_MIN_OBJECTS
        and int(row.get("coverage_templates") or 0) >= STABLE_MIN_TEMPLATES
        and int(row.get("count") or 0) >= STABLE_MIN_COUNT
    ]


def best_row(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    stable = stable_rows(rows)
    if stable:
        return stable[0], True
    return rows[0], False


def load_interventions(model: str, round_name: str) -> dict[str, dict[str, float | None]]:
    path = RESULT_ROOT / round_name / f"phase941_{model}_intervention_rows.jsonl"
    raw: dict[tuple[str, str], list[float]] = defaultdict(list)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw[(str(row["target_label"]), str(row["factor"]))].append(float(row["margin_delta"]))

    out: dict[str, dict[str, float | None]] = defaultdict(dict)
    for (label, factor), values in raw.items():
        out[label][factor] = mean(values)
    return dict(out)


def causal_metrics(by_factor: dict[str, float | None]) -> dict[str, Any]:
    delta_0 = by_factor.get("0.0")
    delta_2 = by_factor.get("2.0")
    zero_damage = -delta_0 if delta_0 is not None else None
    boost_gain = delta_2 if delta_2 is not None else None
    direction_score = None
    if zero_damage is not None and boost_gain is not None:
        direction_score = zero_damage + boost_gain

    if direction_score is None:
        grade = "not_tested"
    elif zero_damage is not None and boost_gain is not None and zero_damage >= 0.25 and boost_gain >= 0.25:
        grade = "strong_directional"
    elif direction_score >= 0.25:
        grade = "directional"
    elif direction_score <= -0.15:
        grade = "inverse_or_mixed"
    else:
        grade = "weak_or_mixed"

    return {
        "margin_delta_0x": delta_0,
        "margin_delta_05x": by_factor.get("0.5"),
        "margin_delta_15x": by_factor.get("1.5"),
        "margin_delta_2x": delta_2,
        "zero_damage": zero_damage,
        "boost_gain": boost_gain,
        "direction_score": direction_score,
        "causal_grade": grade,
    }


def compact_candidate(row: dict[str, Any], stable: bool, causal: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer": int(row["layer"]),
        "channel": int(row["channel"]),
        "stable": bool(stable),
        "effective_score": float(row["effective_score"]),
        "mean_contribution": float(row["mean_contribution"]),
        "selectivity_delta": float(row["selectivity_delta"]),
        "coverage_objects": int(row["coverage_objects"]),
        "coverage_templates": int(row["coverage_templates"]),
        "count": int(row["count"]),
        "positive_rate": float(row["positive_rate"]),
        **causal,
    }


def build_atlas() -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    interventions: dict[str, dict[str, dict[str, float | None]]] = {}
    for model, round_name in ROUNDS:
        summary_path = RESULT_ROOT / round_name / f"phase941_{model}_summary.json"
        summaries[model] = read_json(summary_path)
        interventions[model] = load_interventions(model, round_name)

    labels = sorted(
        {
            label
            for summary in summaries.values()
            for label in summary["top_channels_by_label"].keys()
        }
    )

    model_overview: dict[str, Any] = {}
    colors: dict[str, dict[str, Any]] = {}
    model_reuse: dict[str, list[dict[str, Any]]] = {}
    model_layer_locus: dict[str, dict[str, int]] = {}

    for model, summary in summaries.items():
        model_overview[model] = {
            "sample_rows": int(summary["sample_rows"]),
            "channel_rows": int(summary["channel_rows"]),
            "intervention_rows": int(summary.get("intervention_rows") or 0),
            "target_rank_top1": int(summary.get("target_rank_top1") or 0),
            "target_rank_top10": int(summary.get("target_rank_top10") or 0),
            "target_rank_mean": float(summary.get("target_rank_mean") or 0.0),
            "color_margin_mean": float(summary.get("color_margin_mean") or 0.0),
            "selected_layers": [int(x) for x in summary.get("selected_layers", [])],
        }

        reuse: dict[tuple[int, int], list[str]] = defaultdict(list)
        layers = Counter()
        for label in labels:
            rows = summary["top_channels_by_label"].get(label) or []
            if not rows:
                continue
            row, stable = best_row(rows)
            reuse[(int(row["layer"]), int(row["channel"]))].append(label)
            layers[str(int(row["layer"]))] += 1
            by_factor = interventions[model].get(label, {})
            colors.setdefault(label, {})[model] = compact_candidate(row, stable, causal_metrics(by_factor))

        model_reuse[model] = [
            {"layer": layer, "channel": channel, "labels": sorted(labels_for_channel)}
            for (layer, channel), labels_for_channel in sorted(
                reuse.items(),
                key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
            )
            if len(labels_for_channel) >= 2
        ]
        model_layer_locus[model] = dict(layers)

    return {
        "phase": PHASE,
        "title": "Three-model color encoding feature atlas",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_rounds": {model: round_name for model, round_name in ROUNDS},
        "stability_rule": {
            "coverage_objects_min": STABLE_MIN_OBJECTS,
            "coverage_templates_min": STABLE_MIN_TEMPLATES,
            "count_min": STABLE_MIN_COUNT,
        },
        "model_overview": model_overview,
        "model_layer_locus": model_layer_locus,
        "model_reuse": model_reuse,
        "colors": colors,
    }


def candidate_text(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return "-"
    stable = "稳" if candidate["stable"] else "弱"
    grade = str(candidate.get("causal_grade") or "-")
    return (
        f"L{candidate['layer']} C{candidate['channel']} "
        f"{stable} score={finite(candidate['effective_score'])} "
        f"0x={finite(candidate.get('margin_delta_0x'))} "
        f"2x={finite(candidate.get('margin_delta_2x'))} "
        f"{grade}"
    )


def make_markdown(atlas: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 941 三模型颜色编码特征图谱")
    lines.append("")
    lines.append(f"生成时间：{atlas['created_at']}")
    lines.append("")
    lines.append("## 1. 本轮目标")
    lines.append("")
    lines.append(
        "本轮任务不是只验证模型能回答颜色，而是把颜色相关的内部编码特征压缩成可预测、可验证、可复用的图谱。"
        "测试对象是 qwen3、deepseek7b、glm4 三个本地模型。每个模型使用相同的颜色对象数据、相同的提示模板、相同的通道贡献公式和相同的 top 通道干预流程。"
    )
    lines.append("")
    lines.append("## 2. 核心机制公式")
    lines.append("")
    lines.append("对一个颜色标签 \\(c\\)，先构造输出读出方向：")
    lines.append("")
    lines.append("$$")
    lines.append(r"d_c = W_U[t_c] - \frac{1}{|C|-1}\sum_{c' \ne c} W_U[t_{c'}]")
    lines.append("$$")
    lines.append("")
    lines.append("其中 \\(W_U[t_c]\\) 是颜色词 token 的输出权重行。再把 MLP down projection 的第 \\(j\\) 个通道投影到这个方向上：")
    lines.append("")
    lines.append("$$")
    lines.append(r"r_{\ell,j,c} = d_c^\top W^{down}_{\ell,:,j}")
    lines.append("$$")
    lines.append("")
    lines.append("样本 \\(x\\) 在该通道上的颜色贡献为：")
    lines.append("")
    lines.append("$$")
    lines.append(r"K_{\ell,j,c}(x)=a_{\ell,j}(x)\cdot r_{\ell,j,c}")
    lines.append("$$")
    lines.append("")
    lines.append("最终通道分数为：")
    lines.append("")
    lines.append("$$")
    lines.append(r"S_{\ell,j,c}=\overline{K}_{c}+(\overline{K}_{c}-\overline{K}_{\neg c})+0.05\overline{|K|}_{c}+0.02N_{obj}+0.01N_{tpl}")
    lines.append("$$")
    lines.append("")
    lines.append("通俗说：一个通道要成为颜色通道，必须同时满足四件事：它对目标颜色有正贡献；它比其他颜色更偏向目标颜色；贡献不是偶然的小数值；它能覆盖多个对象和多个模板。")
    lines.append("")
    lines.append("## 3. 全量测试规模")
    lines.append("")
    lines.append("| 模型 | 样本数 | 通道统计 | 干预记录 | Top1 | Top10 | 平均 margin | 主要层位 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for model, row in atlas["model_overview"].items():
        locus = ", ".join(f"L{k}:{v}" for k, v in sorted(atlas["model_layer_locus"][model].items(), key=lambda x: int(x[0])))
        lines.append(
            f"| {model} | {row['sample_rows']} | {row['channel_rows']} | {row['intervention_rows']} | "
            f"{row['target_rank_top1']} | {row['target_rank_top10']} | {finite(row['color_margin_mean'])} | {locus} |"
        )
    lines.append("")
    lines.append("## 4. 三模型颜色图谱")
    lines.append("")
    lines.append("表中 `0x` 表示把 top 通道组置零后的 margin 变化，通常越负说明该通道组越必要；`2x` 表示放大到 2 倍后的 margin 变化，通常越正说明该通道组越有推动作用。")
    lines.append("")
    lines.append("| 颜色 | qwen3 | deepseek7b | glm4 |")
    lines.append("|---|---|---|---|")
    for label in sorted(atlas["colors"]):
        per_model = atlas["colors"][label]
        lines.append(
            f"| {label} | {candidate_text(per_model.get('qwen3'))} | "
            f"{candidate_text(per_model.get('deepseek7b'))} | {candidate_text(per_model.get('glm4'))} |"
        )
    lines.append("")
    lines.append("## 5. 共享通道和复用结构")
    lines.append("")
    for model, rows in atlas["model_reuse"].items():
        lines.append(f"### {model}")
        lines.append("")
        if not rows:
            lines.append("未发现作为多个颜色最佳候选的通道。")
        else:
            for row in rows:
                labels = ", ".join(row["labels"])
                lines.append(f"- L{row['layer']} C{row['channel']}：{labels}")
        lines.append("")
    lines.append("## 6. 关键结论")
    lines.append("")
    lines.append("1. qwen3 的颜色编码主要落在后段 L27/L35，红色、银色、黑色最清楚。红色 L35 C284 同时也是黑色强候选，说明单通道可能不是纯颜色名，而是承载一组颜色/材质/对象联动方向。")
    lines.append("2. deepseek7b 的候选分数很高，但颜色回答 baseline 较弱，很多颜色不是 top1。因此它的读出通道很强，因果干预却更混合，说明读出方向和实际生成决策之间还有竞争项没有被当前公式完全捕获。")
    lines.append("3. glm4 的图谱最像清晰的可控机制：置零 top 通道通常降低 margin，放大 top 通道通常提高 margin。green、red、yellow 的干预信号尤其明显。")
    lines.append("4. 三个模型不存在可直接比较的相同 channel id，因为架构和训练不同；真正可复用的是角色层级：颜色特征通常集中在中后层 MLP 通道，并通过输出词向量方向形成读出贡献。")
    lines.append("")
    lines.append("## 7. 当前缺口")
    lines.append("")
    lines.append("当前图谱已经能定位颜色编码候选，但还不是完整编码机制。还缺三块：")
    lines.append("")
    lines.append("- 反事实对象控制：例如 red apple、green apple、yellow apple，区分对象知识和颜色属性。")
    lines.append("- 跨 token 位置追踪：当前主要看最后 token，需要追踪颜色信息在前文对象 token、属性 token、答案 token 之间如何迁移。")
    lines.append("- 竞争项建模：特别是 deepseek7b，读出贡献强但生成结果弱，说明还有 blocker/suppressor 或其他候选词竞争。")
    lines.append("")
    lines.append("## 8. 下一步算法")
    lines.append("")
    lines.append("下一步应进入 Phase 942：颜色反事实闭环。做法是固定对象、替换颜色、固定模板，比较同一对象在不同颜色属性下的通道变化，并对当前 top 通道做正负向干预。")
    lines.append("")
    lines.append("目标公式：")
    lines.append("")
    lines.append("$$")
    lines.append(r"\Delta K_{\ell,j,c}(x_{object,color_1},x_{object,color_2}) = K_{\ell,j,c}(x_{object,color_1}) - K_{\ell,j,c}(x_{object,color_2})")
    lines.append("$$")
    lines.append("")
    lines.append("如果一个通道是真正的颜色编码通道，它应该随颜色改变而改变，而不是只随对象改变而改变。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atlas = build_atlas()
    json_path = OUT_DIR / "phase941_color_cross_model_atlas.json"
    md_path = OUT_DIR / "phase941_color_cross_model_atlas.md"
    json_path.write_text(json.dumps(atlas, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = make_markdown(atlas)
    md_path.write_text(markdown, encoding="utf-8")
    ANALYSIS_DOC.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "json": str(json_path),
        "markdown": str(md_path),
        "analysis_doc": str(ANALYSIS_DOC),
        "colors": len(atlas["colors"]),
        "models": len(atlas["model_overview"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
