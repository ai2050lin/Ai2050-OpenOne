#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE941_ROOT = ROOT / "tests" / "result" / "phase941_color_feature_neuron_atlas"
PHASE942_ROOT = ROOT / "tests" / "result" / "phase942_color_counterfactual_closure"
PHASE943_ROOT = ROOT / "tests" / "result" / "phase943_color_counterfactual_feature_discovery"
OUT_DIR = PHASE943_ROOT / "mechanism_report"
ANALYSIS_DOC = ROOT / "research" / "MainAnalysis" / "20260705_03_Phase941-943颜色编码机制闭环.md"

MODELS = ["qwen3", "deepseek7b", "glm4"]
ROUNDS_942 = {
    "qwen3": "color_counterfactual_closure_qwen3_full",
    "deepseek7b": "color_counterfactual_closure_deepseek7b_full",
    "glm4": "color_counterfactual_closure_glm4_full",
}
ROUNDS_943 = {
    "qwen3": "color_counterfactual_feature_discovery_qwen3_full",
    "deepseek7b": "color_counterfactual_feature_discovery_deepseek7b_full",
    "glm4": "color_counterfactual_feature_discovery_glm4_full",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        if int(row.get("coverage_objects") or 0) >= 3
        and int(row.get("coverage_templates") or 0) >= 3
        and int(row.get("count") or 0) >= 8
    ]


def best_stable(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    stable = stable_rows(rows)
    if stable:
        return stable[0], True
    return rows[0], False


def intervention_means(path: Path) -> dict[str, dict[str, float | None]]:
    raw: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in read_jsonl(path):
        if row.get("margin_delta") is not None:
            raw[(str(row["target_label"]), str(row["factor"]))].append(float(row["margin_delta"]))
    out: dict[str, dict[str, float | None]] = defaultdict(dict)
    for (label, factor), values in raw.items():
        out[label][factor] = mean(values)
    return dict(out)


def load_phase941_atlas() -> dict[str, Any]:
    return read_json(PHASE941_ROOT / "cross_model_atlas" / "phase941_color_cross_model_atlas.json")


def build_report_payload() -> dict[str, Any]:
    atlas941 = load_phase941_atlas()
    summaries942 = {
        model: read_json(PHASE942_ROOT / ROUNDS_942[model] / f"phase942_{model}_summary.json")
        for model in MODELS
    }
    summaries943 = {
        model: read_json(PHASE943_ROOT / ROUNDS_943[model] / f"phase943_{model}_summary.json")
        for model in MODELS
    }
    interventions943 = {
        model: intervention_means(PHASE943_ROOT / ROUNDS_943[model] / f"phase943_{model}_intervention_rows.jsonl")
        for model in MODELS
    }

    labels = sorted(atlas941["colors"].keys())
    overview: dict[str, Any] = {}
    phase943_channels: dict[str, dict[str, Any]] = {}
    channel_shift: dict[str, dict[str, Any]] = {}

    for model in MODELS:
        s942 = summaries942[model]
        s943 = summaries943[model]
        overview[model] = {
            "phase942_samples": int(s942["sample_rows"]),
            "phase942_answer_top1": int(s942["target_rank_top1"]),
            "phase942_answer_top1_rate": int(s942["target_rank_top1"]) / max(1, int(s942["sample_rows"])),
            "phase942_old_feature_top1": int(s942["feature_top1_norm"]),
            "phase942_old_feature_top1_rate": float(s942["feature_top1_norm_rate"]),
            "phase942_color_margin_mean": float(s942["color_margin_mean"]),
            "phase943_samples": int(s943["sample_rows"]),
            "phase943_channel_rows": int(s943["channel_rows"]),
            "phase943_intervention_rows": int(s943["intervention_rows"]),
            "phase943_answer_top1": int(s943["target_rank_top1"]),
            "phase943_answer_top1_rate": int(s943["target_rank_top1"]) / max(1, int(s943["sample_rows"])),
            "phase943_color_margin_mean": float(s943["color_margin_mean"]),
        }

        for label in labels:
            row943, stable = best_stable(s943["top_channels_by_label"][label])
            old = atlas941["colors"][label][model]
            same_channel = int(old["layer"]) == int(row943["layer"]) and int(old["channel"]) == int(row943["channel"])
            intv = interventions943[model].get(label, {})
            phase943_channels.setdefault(label, {})[model] = {
                "layer": int(row943["layer"]),
                "channel": int(row943["channel"]),
                "stable": stable,
                "effective_score": float(row943["effective_score"]),
                "coverage_objects": int(row943["coverage_objects"]),
                "coverage_templates": int(row943["coverage_templates"]),
                "count": int(row943["count"]),
                "phase941_layer": int(old["layer"]),
                "phase941_channel": int(old["channel"]),
                "same_as_phase941": same_channel,
                "margin_delta_0x": intv.get("0.0"),
                "margin_delta_2x": intv.get("2.0"),
            }
            channel_shift.setdefault(model, {"same": [], "changed": []})
            channel_shift[model]["same" if same_channel else "changed"].append(label)

    return {
        "title": "Phase 941-943 color encoding mechanism closure",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": MODELS,
        "labels": labels,
        "overview": overview,
        "phase943_channels": phase943_channels,
        "channel_shift": channel_shift,
    }


def channel_cell(row: dict[str, Any]) -> str:
    mark = "同" if row["same_as_phase941"] else "新"
    stable = "稳" if row["stable"] else "弱"
    return (
        f"L{row['layer']} C{row['channel']} {mark}/{stable} "
        f"score={finite(row['effective_score'])} "
        f"0x={finite(row.get('margin_delta_0x'))} "
        f"2x={finite(row.get('margin_delta_2x'))}"
    )


def make_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 941-943 颜色编码机制闭环")
    lines.append("")
    lines.append(f"生成时间：{payload['created_at']}")
    lines.append("")
    lines.append("## 1. 本轮完成了什么")
    lines.append("")
    lines.append("本轮把颜色编码测试从“典型对象颜色”推进到“反事实颜色变量”。核心问题是：apple 通常是 red，这种典型知识通道，是否等同于 `A blue cube` 这种显式颜色变量通道？")
    lines.append("")
    lines.append("结论很清楚：二者有关联，但不能完全等同。Phase 941 找到的是典型对象颜色的读出通道；Phase 942 证明这些旧通道不能直接完成显式颜色反事实分类；Phase 943 在同一套反事实样本上重新扫描，找到了更贴近显式颜色变量的一批稳定通道。")
    lines.append("")
    lines.append("## 2. 机制公式")
    lines.append("")
    lines.append("三轮使用同一个通道贡献公式：")
    lines.append("")
    lines.append("$$")
    lines.append(r"d_c = W_U[t_c]-\frac{1}{|C|-1}\sum_{c'\ne c}W_U[t_{c'}]")
    lines.append("$$")
    lines.append("")
    lines.append("$$")
    lines.append(r"K_{\ell,j,c}(x)=a_{\ell,j}(x)\cdot d_c^\top W^{down}_{\ell,:,j}")
    lines.append("$$")
    lines.append("")
    lines.append("Phase 942 的关键判据是旧图谱能否在反事实样本上完成分类：")
    lines.append("")
    lines.append("$$")
    lines.append(r"\hat c(x)=\arg\max_c \frac{K_{\ell_c,j_c,c}(x)}{|\overline K^{941}_{c}|+\epsilon}")
    lines.append("$$")
    lines.append("")
    lines.append("如果旧通道真是完整颜色变量，\\(\\hat c(x)\\) 应该随 prompt 中的颜色变化而变化。实际结果接近随机，因此需要 Phase 943 重新发现。")
    lines.append("")
    lines.append("## 3. 三模型总览")
    lines.append("")
    lines.append("| 模型 | 反事实回答 Top1 | 旧通道反事实 Top1 | Phase943 通道行 | Phase943 干预行 | 平均 margin |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model in payload["models"]:
        row = payload["overview"][model]
        lines.append(
            f"| {model} | {row['phase942_answer_top1']}/198 ({row['phase942_answer_top1_rate']:.1%}) | "
            f"{row['phase942_old_feature_top1']}/198 ({row['phase942_old_feature_top1_rate']:.1%}) | "
            f"{row['phase943_channel_rows']} | {row['phase943_intervention_rows']} | {finite(row['phase943_color_margin_mean'])} |"
        )
    lines.append("")
    lines.append("解读：模型本身能回答反事实颜色，尤其 qwen3 和 glm4；但 Phase 941 旧候选通道几乎不能作为反事实颜色分类器。这就是本轮发现的核心缺口。")
    lines.append("")
    lines.append("## 4. Phase943 反事实颜色变量通道")
    lines.append("")
    lines.append("表中 `同` 表示与 Phase 941 的典型颜色候选是同一 layer/channel，`新` 表示反事实数据上换成了新的最佳候选。`0x` 为置零 top 通道组后的 margin 变化，`2x` 为放大两倍后的 margin 变化。")
    lines.append("")
    lines.append("| 颜色 | qwen3 | deepseek7b | glm4 |")
    lines.append("|---|---|---|---|")
    for label in payload["labels"]:
        row = payload["phase943_channels"][label]
        lines.append(
            f"| {label} | {channel_cell(row['qwen3'])} | {channel_cell(row['deepseek7b'])} | {channel_cell(row['glm4'])} |"
        )
    lines.append("")
    lines.append("## 5. 通道是否发生迁移")
    lines.append("")
    for model in payload["models"]:
        shift = payload["channel_shift"][model]
        same = ", ".join(shift["same"]) if shift["same"] else "无"
        changed = ", ".join(shift["changed"]) if shift["changed"] else "无"
        lines.append(f"- {model}：同通道颜色 = {same}；迁移颜色 = {changed}。")
    lines.append("")
    lines.append("## 6. 关键结论")
    lines.append("")
    lines.append("1. 颜色编码不是单一图谱，而是至少分成两层：典型对象颜色图谱和显式颜色变量图谱。")
    lines.append("2. qwen3 在反事实重新发现后，red/brown/silver 等仍沿用强典型通道，但 blue/green/orange/yellow 等出现明显迁移，说明对象典型颜色和上下文指定颜色会走不同子路径。")
    lines.append("3. deepseek7b 的颜色读出集中在 L27，分数很高，但干预方向混合，说明竞争项仍然很强。它不像 qwen3/glm4 那样形成清晰可控的单通道闭环。")
    lines.append("4. glm4 的结构最规整：green/red/yellow/blue/orange 多集中在 L30，brown/gray/purple/silver 集中在 L39，说明它的颜色变量更像分层的读出结构。")
    lines.append("5. 当前机制公式可以发现颜色相关通道，但还不能单独完成全局编码图谱，因为它还缺少 token 位置传播、竞争抑制项和多通道联合解码。")
    lines.append("")
    lines.append("## 7. 下一步")
    lines.append("")
    lines.append("下一步应进入 Phase 944：颜色 token 轨迹追踪。要同时抓取颜色词位置、对象词位置、最后答案位置，比较同一个通道在三个位置的贡献变化：")
    lines.append("")
    lines.append("$$")
    lines.append(r"T_{\ell,j,c}(p)=a_{\ell,j}(x,p)\cdot d_c^\top W^{down}_{\ell,:,j}")
    lines.append("$$")
    lines.append("")
    lines.append("如果颜色变量真的闭合，应该能看到颜色信息从颜色词位置进入残差流，再经过中后层 MLP/attention 传到最后答案位置。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_report_payload()
    json_path = OUT_DIR / "phase941_943_color_mechanism_report.json"
    md_path = OUT_DIR / "phase941_943_color_mechanism_report.md"
    markdown = make_markdown(payload)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    ANALYSIS_DOC.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "json": str(json_path),
        "markdown": str(md_path),
        "analysis_doc": str(ANALYSIS_DOC),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
