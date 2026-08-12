#!/usr/bin/env python3
"""Independent artifact audit for Phase 998."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase998_minimal_causal_thread_protocol import (
    OUT_ROOT,
    PHASE,
    canonical,
    write_json,
)


EXPECTED = {
    "cases": 4096,
    "pairs": 2048,
    "worlds": 128,
    "behavior_rows": 4096,
    "trace_pairs": 528,
    "trace_events": 111,
    "causal_rows": 8352,
    "restoration_rows": 288,
    "natural_rows": 1056,
    "confirmation_pairs": 288,
    "confirmation_causal_rows": 16704,
    "confirmation_restoration_rows": 576,
    "confirmation_natural_rows": 3168,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return {
        condition: {
            "n": len(values),
            "mean_transfer": float(
                np.mean([row["normalized_transfer"] for row in values])
            ),
            "median_transfer": float(
                np.median([row["normalized_transfer"] for row in values])
            ),
            "flip_rate": float(
                np.mean([row["candidate_flipped_to_source"] for row in values])
            ),
            "toward_rate": float(
                np.mean([row["toward_source"] for row in values])
            ),
        }
        for condition, values in groups.items()
    }


def natural_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return {
        condition: {
            "n": len(values),
            "flip_rate": float(np.mean([row["flipped_to_source"] for row in values])),
            "retention_rate": float(np.mean([row["remained_target"] for row in values])),
            "eos_rate": float(np.mean([row["eos_seen"] for row in values])),
            "exact_short_rate": float(np.mean([row["exact_short"] for row in values])),
        }
        for condition, values in groups.items()
    }


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def audit() -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    protocol_audit = read_json(OUT_ROOT / "protocol" / "audit.json")
    cases = read_jsonl(OUT_ROOT / "protocol" / "cases.jsonl")
    behavior = read_jsonl(OUT_ROOT / "behavior" / "behavior_rows.jsonl")
    behavior_summary = read_json(OUT_ROOT / "behavior" / "summary.json")
    selected = read_jsonl(OUT_ROOT / "trace" / "selected_pairs.jsonl")
    trace_events = read_jsonl(OUT_ROOT / "trace" / "event_metrics.jsonl")
    trace_summary = read_json(OUT_ROOT / "trace" / "summary.json")
    causal = read_jsonl(OUT_ROOT / "causal" / "causal_rows.jsonl")
    restoration = read_jsonl(OUT_ROOT / "causal" / "restoration_rows.jsonl")
    natural = read_jsonl(OUT_ROOT / "causal" / "natural_rows.jsonl")
    causal_summary = read_json(OUT_ROOT / "causal" / "summary.json")
    confirmation_selected = read_jsonl(
        OUT_ROOT / "confirmation" / "selected_pairs.jsonl"
    )
    confirmation_causal = read_jsonl(
        OUT_ROOT / "confirmation" / "causal_rows.jsonl"
    )
    confirmation_restoration = read_jsonl(
        OUT_ROOT / "confirmation" / "restoration_rows.jsonl"
    )
    confirmation_natural = read_jsonl(
        OUT_ROOT / "confirmation" / "natural_rows.jsonl"
    )
    confirmation_summary = read_json(
        OUT_ROOT / "confirmation" / "summary.json"
    )

    failures = []
    counts = {
        "cases": len(cases),
        "pairs": len({row["pair_id"] for row in cases}),
        "worlds": len({row["world_id"] for row in cases}),
        "behavior_rows": len(behavior),
        "trace_pairs": len(selected),
        "trace_events": len(trace_events),
        "causal_rows": len(causal),
        "restoration_rows": len(restoration),
        "natural_rows": len(natural),
        "confirmation_pairs": len(confirmation_selected),
        "confirmation_causal_rows": len(confirmation_causal),
        "confirmation_restoration_rows": len(confirmation_restoration),
        "confirmation_natural_rows": len(confirmation_natural),
    }
    for key, expected in EXPECTED.items():
        if counts[key] != expected:
            failures.append(f"count/{key}:{counts[key]}!={expected}")

    if not protocol_audit["passed"] or not protocol["cpu_protocol_pass"]:
        failures.append("protocol_gate")
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        pair_groups[row["pair_id"]].append(row)
    for pair_id, values in pair_groups.items():
        if len(values) != 2:
            failures.append(f"pair_count/{pair_id}")
            continue
        a, b = sorted(values, key=lambda row: row["arm"])
        if Counter(a["input_ids"]) != Counter(b["input_ids"]):
            failures.append(f"token_multiset/{pair_id}")
        if a["gold"] != b["foil"] or a["foil"] != b["gold"]:
            failures.append(f"semantic_swap/{pair_id}")

    behavior_metrics = {
        "candidate_accuracy": float(
            np.mean([row["candidate_correct"] for row in behavior])
        ),
        "natural_accuracy": float(
            np.mean([row["natural_correct_both"] for row in behavior])
        ),
        "repeat_stability": float(
            np.mean([row["repeat_stable"] for row in behavior])
        ),
        "eos_rate": float(np.mean([row["eos_both"] for row in behavior])),
        "exact_short_rate": float(
            np.mean([row["exact_short_both"] for row in behavior])
        ),
    }
    for key, value in behavior_metrics.items():
        if not close(value, behavior_summary["metrics"][key]):
            failures.append(f"behavior_recompute/{key}")
    if not behavior_summary["behavior_gate_pass"]:
        failures.append("behavior_gate")

    selected_partitions = Counter(row["partition"] for row in selected)
    if dict(selected_partitions) != {
        "discovery": 288,
        "validation": 144,
        "holdout": 96,
    }:
        failures.append(f"trace_partitions/{dict(selected_partitions)}")
    if trace_summary["selection_uses_holdout"] or not trace_summary[
        "observation_gate_pass"
    ]:
        failures.append("trace_gate_or_holdout_leak")

    recomputed_candidate = condition_summary(causal)
    recomputed_natural = natural_summary(natural)
    for condition, values in recomputed_candidate.items():
        stored = causal_summary["candidate_condition_summary"][condition]
        if not close(values["mean_transfer"], stored["mean_normalized_transfer"]):
            failures.append(f"causal_mean/{condition}")
        if not close(values["flip_rate"], stored["candidate_flip_rate"]):
            failures.append(f"causal_flip/{condition}")
    for condition, values in recomputed_natural.items():
        stored = causal_summary["natural_condition_summary"][condition]
        if not close(values["flip_rate"], stored["natural_flip_rate"]):
            failures.append(f"natural_flip/{condition}")

    restoration_median = float(
        np.median([row["recovery_fraction"] for row in restoration])
    )
    if not close(
        restoration_median,
        causal_summary["restoration_summary"]["median_recovery_fraction"],
    ):
        failures.append("restoration_recompute")

    write_diff_64 = recomputed_candidate["write_difference_64"]["mean_transfer"]
    write_top_64 = recomputed_candidate["write_top_activation_64"]["mean_transfer"]
    write_diff_256 = recomputed_candidate["write_difference_256"]["mean_transfer"]
    write_top_256 = recomputed_candidate[
        "write_top_activation_256"
    ]["mean_transfer"]
    write_full = recomputed_candidate["write_full"]["mean_transfer"]
    read_full = recomputed_candidate["read_full"]["mean_transfer"]
    decision_full = recomputed_candidate["decision_full"]["mean_transfer"]
    audit_metrics = {
        "write_full_mean_transfer": write_full,
        "write_difference_64_mean_transfer": write_diff_64,
        "write_top_activation_64_mean_transfer": write_top_64,
        "write_difference_64_vs_top_ratio": write_diff_64
        / max(abs(write_top_64), 1e-8),
        "write_difference_256_mean_transfer": write_diff_256,
        "write_top_activation_256_mean_transfer": write_top_256,
        "write_difference_256_vs_top_ratio": write_diff_256
        / max(abs(write_top_256), 1e-8),
        "write_difference_256_fraction_of_full": write_diff_256
        / max(abs(write_full), 1e-8),
        "read_full_mean_transfer": read_full,
        "decision_full_mean_transfer": decision_full,
        "write_difference_256_candidate_flip_rate": recomputed_candidate[
            "write_difference_256"
        ]["flip_rate"],
        "write_difference_256_natural_flip_rate": recomputed_natural[
            "write_difference_256"
        ]["flip_rate"],
        "write_top_activation_256_natural_flip_rate": recomputed_natural[
            "write_top_activation_256"
        ]["flip_rate"],
        "write_random_256_natural_flip_rate": recomputed_natural[
            "write_random_256"
        ]["flip_rate"],
        "write_wrong_position_natural_flip_rate": recomputed_natural[
            "write_wrong_position_difference_256"
        ]["flip_rate"],
        "restoration_median_recovery": restoration_median,
        "write_to_read_median_mediation_fraction": causal_summary[
            "mediation_summary"
        ]["write_to_read_median_mediation_fraction"],
    }

    claims = {
        "behavior_denominator_closed": behavior_metrics["natural_accuracy"] == 1.0,
        "stable_observational_chain_found": trace_summary["observation_gate_pass"],
        "source_color_write_is_locally_causal": (
            audit_metrics["write_difference_256_candidate_flip_rate"] == 1.0
            and audit_metrics["write_difference_256_natural_flip_rate"] == 1.0
            and audit_metrics["write_random_256_natural_flip_rate"] == 0.0
            and audit_metrics["write_wrong_position_natural_flip_rate"] == 0.0
        ),
        "query_read_event_is_causal": abs(read_full) >= 0.10,
        "ordered_three_event_thread_closed": causal_summary[
            "causal_thread_gate_pass"
        ],
        "difference_selection_dominates_top_activation_at_64": (
            audit_metrics["write_difference_64_vs_top_ratio"] >= 2.0
        ),
        "difference_selection_dominates_top_activation_at_256": (
            audit_metrics["write_difference_256_vs_top_ratio"] >= 2.0
        ),
    }
    if causal_summary["causal_thread_gate_pass"]:
        failures.append("unexpected_causal_gate_pass")
    trace_pair_ids = {row["pair_id"] for row in selected}
    confirmation_pair_ids = {row["pair_id"] for row in confirmation_selected}
    if trace_pair_ids & confirmation_pair_ids:
        failures.append("confirmation_selection_overlap")
    if confirmation_summary["causal_thread_gate_pass"]:
        failures.append("unexpected_confirmation_gate_pass")
    confirmation_candidate = condition_summary(confirmation_causal)
    confirmation_natural_summary = natural_summary(confirmation_natural)
    confirmation_restoration_median = float(
        np.median(
            [row["recovery_fraction"] for row in confirmation_restoration]
        )
    )
    confirmation_metrics = {
        "write_difference_256_candidate_flip_rate": confirmation_candidate[
            "write_difference_256"
        ]["flip_rate"],
        "write_difference_256_natural_flip_rate": confirmation_natural_summary[
            "write_difference_256"
        ]["flip_rate"],
        "write_random_256_natural_flip_rate": confirmation_natural_summary[
            "write_random_256"
        ]["flip_rate"],
        "write_wrong_position_natural_flip_rate": confirmation_natural_summary[
            "write_wrong_position_difference_256"
        ]["flip_rate"],
        "restoration_median_recovery": confirmation_restoration_median,
        "difference_vs_top_activation_effect_ratio": confirmation_summary[
            "gate_metrics"
        ]["difference_vs_top_activation_effect_ratio"],
        "minimum_mediation_fraction": confirmation_summary["gate_metrics"][
            "minimum_mediation_fraction"
        ],
    }
    for condition, values in confirmation_candidate.items():
        stored = confirmation_summary["candidate_condition_summary"][condition]
        if not close(values["mean_transfer"], stored["mean_normalized_transfer"]):
            failures.append(f"confirmation_causal_mean/{condition}")
        if not close(values["flip_rate"], stored["candidate_flip_rate"]):
            failures.append(f"confirmation_causal_flip/{condition}")
    for condition, values in confirmation_natural_summary.items():
        stored = confirmation_summary["natural_condition_summary"][condition]
        if not close(values["flip_rate"], stored["natural_flip_rate"]):
            failures.append(f"confirmation_natural_flip/{condition}")

    result = {
        "schema_version": "phase998_independent_audit.v1",
        "phase": PHASE,
        "passed": not failures,
        "failures": failures,
        "artifact_counts": counts,
        "behavior_metrics": behavior_metrics,
        "selected_chain": trace_summary["selected_chain"],
        "selected_event_metrics": trace_summary["selected_event_metrics"],
        "causal_gate_pass": causal_summary["causal_thread_gate_pass"],
        "causal_gate_checks": causal_summary["gate_checks"],
        "audit_metrics": audit_metrics,
        "confirmation_gate_pass": confirmation_summary[
            "causal_thread_gate_pass"
        ],
        "confirmation_gate_checks": confirmation_summary["gate_checks"],
        "confirmation_metrics": confirmation_metrics,
        "claims": claims,
        "interpretation": {
            "strict_result": "partial_local_causality_without_thread_closure",
            "source_write_caveat": (
                "The selected write event is layer-1 at the literal color token; "
                "it may primarily reflect lexical token identity rather than an "
                "entity-attribute binding operation."
            ),
            "query_read_caveat": (
                "The layer-17 query event is highly decodable but its full-vector "
                "swap has near-zero output effect, so it is not a demonstrated "
                "causal mediator."
            ),
            "mediation_caveat": (
                "Read-to-decision mediation ratios are not interpretable when the "
                "read intervention itself has approximately zero effect."
            ),
        },
    }
    return result


def report_markdown(audit_result: dict[str, Any]) -> str:
    m = audit_result["audit_metrics"]
    c = audit_result["confirmation_metrics"]
    chain = audit_result["selected_chain"]
    checks = audit_result["causal_gate_checks"]
    return f"""# Phase 998 最小因果脉络可行性测试

## 严格结论

整体结果为：**局部因果写入成立，但三事件动态脉络没有闭合**。

行为分母在 4096 条记录、2048 个反事实对和 128 个世界上完全闭合。观测算法选择出
`{chain['write']} -> {chain['read']} -> {chain['decision']}`，三个事件在验证集和封存集上的
方向命中率均为 100%。但是因果测试表明，这三个相关事件不能连接成一条中介链。

## 关键数据

- Layer 1 颜色词写入，完整 residual 交换的平均归一化转移：{m['write_full_mean_transfer']:.6f}
- 256 个差分通道的平均转移：{m['write_difference_256_mean_transfer']:.6f}
- 256 个差分通道占完整交换效应：{m['write_difference_256_fraction_of_full']:.2%}
- 256 个差分通道候选翻转率：{m['write_difference_256_candidate_flip_rate']:.2%}
- 256 个差分通道自然答案翻转率：{m['write_difference_256_natural_flip_rate']:.2%}
- 同预算最高激活通道自然翻转率：{m['write_top_activation_256_natural_flip_rate']:.2%}
- 随机通道自然翻转率：{m['write_random_256_natural_flip_rate']:.2%}
- 错误位置自然翻转率：{m['write_wrong_position_natural_flip_rate']:.2%}
- Layer 17 查询位置完整 residual 交换平均转移：{m['read_full_mean_transfer']:.6f}
- Layer 33 答案边界完整 residual 交换平均转移：{m['decision_full_mean_transfer']:.6f}
- 写入到读取的中介比例中位数：{m['write_to_read_median_mediation_fraction']:.6f}
- 恢复比例中位数：{m['restoration_median_recovery']:.6f}
- 64 通道差分/最高激活效应比：{m['write_difference_64_vs_top_ratio']:.3f}
- 256 通道差分/最高激活效应比：{m['write_difference_256_vs_top_ratio']:.3f}

## 独立确认轮

冻结候选后，又使用 288 个与发现/验证样本不重叠的反事实对，执行 576 个双向候选干预和
288 个自然生成干预。没有重新选择层、通道或阈值。

- 256 个差分通道候选翻转率：{c['write_difference_256_candidate_flip_rate']:.2%}
- 256 个差分通道自然答案翻转率：{c['write_difference_256_natural_flip_rate']:.2%}
- 随机通道自然翻转率：{c['write_random_256_natural_flip_rate']:.2%}
- 错误位置自然翻转率：{c['write_wrong_position_natural_flip_rate']:.2%}
- 恢复比例中位数：{c['restoration_median_recovery']:.6f}
- 差分/最高激活效应比：{c['difference_vs_top_activation_effect_ratio']:.3f}
- 最小中介比例：{c['minimum_mediation_fraction']:.6f}

## 门槛

{json.dumps(checks, ensure_ascii=False, indent=2)}

## 解释

Layer 1 的结果是强而特异的局部因果效应：交换正确位置的差分通道能够稳定改写候选和自然
答案，随机通道、空操作和错误位置都不能做到。但该位置正是字面颜色 token，因此这首先证明
的是颜色词源表示具有因果充分性，尚不能证明已经提取出对象—属性绑定规则。

Layer 17 查询词状态虽然稳定携带答案差异，但完整交换几乎不改变输出。它是一个典型的
“可读但没有证明被使用”的状态。恢复和中介门因此失败，说明观测轨迹不能自动升级为因果
脉络。模型可能在后层直接重新读取事实颜色，也可能通过未被选中的注意力/KV路径传递信息。

差分选通在 64 通道的小预算下明显优于最高激活，但不足以稳定翻转答案；扩大到 256 通道后
两者都很强，差分方法只有小幅优势，没有达到预注册的 2 倍门。这否定了“差分选通已经显著
优于最高激活并找到了独立机制组”的强结论。
"""


def main() -> None:
    result = audit()
    output = OUT_ROOT / "audit"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit.json", result)
    report = report_markdown(result)
    temp = output / "report.md.tmp"
    temp.write_text(report, encoding="utf-8")
    temp.replace(output / "report.md")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "strict_result": result["interpretation"]["strict_result"],
                "causal_thread_gate_pass": result["causal_gate_pass"],
                "failures": result["failures"],
            },
            ensure_ascii=False,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
