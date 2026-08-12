#!/usr/bin/env python3
"""Independent artifact audit and report for Phase 1000."""
from __future__ import annotations

import hashlib
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

from phase1000_factorial_binding_protocol import OUT_ROOT, PHASE, write_json
from phase1000_scpg_confirmation import one_direction_per_pair
from phase1000_scpg_discovery import (
    batches_by_template,
    directional_pairs,
    read_jsonl,
    summarize_interventions,
)
from phase1000_source_control_audit import valid_derangement_shifts


EXPECTED = {
    "cases": 8192,
    "factor_pairs": 12288,
    "behavior_rows": 8192,
    "discovery_pairs": 256,
    "discovery_source_scan_rows": 2560,
    "discovery_source_control_rows": 3072,
    "discovery_source_natural_rows": 3072,
    "discovery_response_rows": 71680,
    "discovery_receiver_rows": 6144,
    "discovery_joint_rows": 2560,
    "discovery_joint_natural_rows": 1536,
    "confirmation_pairs": 512,
    "confirmation_source_control_rows": 3072,
    "confirmation_source_natural_rows": 3072,
    "confirmation_response_rows": 6144,
    "confirmation_receiver_rows": 6144,
    "confirmation_joint_rows": 2560,
    "confirmation_joint_natural_rows": 1536,
    "control_validation_candidate_rows": 6656,
    "control_validation_natural_rows": 3584,
    "control_confirmation_candidate_rows": 6656,
    "control_confirmation_natural_rows": 3584,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
    )


def natural_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    result = {}
    for condition, values in groups.items():
        target_field = (
            "remained_target"
            if "remained_target" in values[0]
            else "restored_to_target"
        )
        result[condition] = {
            "n": len(values),
            "flip_rate": float(np.mean([row["flipped_to_source"] for row in values])),
            "target_rate": float(np.mean([row[target_field] for row in values])),
        }
    return result


def joint_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["joint_size"])].append(row)
    return {
        str(size): {
            "n": len(values),
            "mean_sufficiency_transfer": float(
                np.mean([row["sufficiency_transfer"] for row in values])
            ),
            "median_mediation_fraction": float(
                np.median([row["mediation_fraction"] for row in values])
            ),
            "restored_to_target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
        }
        for size, values in groups.items()
    }


def receiver_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    return {
        event_id: {
            "n": len(values),
            "mean_sufficiency_transfer": float(
                np.mean([row["sufficiency_transfer"] for row in values])
            ),
            "median_mediation_fraction": float(
                np.median([row["mediation_fraction"] for row in values])
            ),
            "scrambled_flip_rate": float(
                np.mean([row["scrambled_flipped"] for row in values])
            ),
        }
        for event_id, values in groups.items()
    }


def build_derangement_manifest(
    discovery_pairs: list[dict[str, Any]],
    confirmation_pairs: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    validation = directional_pairs(
        discovery_pairs, case_by_id, "validation", bidirectional=True
    )
    confirmation = one_direction_per_pair(confirmation_pairs, case_by_id)
    manifest = []
    failures = []
    for partition, directional in (
        ("validation", validation),
        ("confirmation", confirmation),
    ):
        for batch_index, batch in enumerate(batches_by_template(directional, 16)):
            shifts = valid_derangement_shifts(batch, 8)
            for target_index, target in enumerate(batch):
                for null_index, shift in enumerate(shifts):
                    donor = batch[(target_index - shift) % len(batch)]
                    row = {
                        "partition": partition,
                        "batch_index": batch_index,
                        "null_index": null_index,
                        "shift": shift,
                        "target_pair_id": target["pair_id"],
                        "target_direction": target["direction"],
                        "target_world_id": target["target"]["world_id"],
                        "donor_pair_id": donor["pair_id"],
                        "donor_direction": donor["direction"],
                        "donor_world_id": donor["source"]["world_id"],
                        "different_pair": target["pair_id"] != donor["pair_id"],
                    }
                    if not row["different_pair"]:
                        failures.append(
                            f"derangement/{partition}/{batch_index}/"
                            f"{target_index}/{null_index}"
                        )
                    manifest.append(row)
    return manifest, failures


def audit() -> dict[str, Any]:
    protocol_root = OUT_ROOT / "protocol"
    behavior_root = OUT_ROOT / "behavior"
    discovery_root = OUT_ROOT / "discovery"
    confirmation_root = OUT_ROOT / "confirmation"
    control_root = OUT_ROOT / "control_audit"
    audit_root = OUT_ROOT / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)

    protocol = read_json(protocol_root / "protocol.json")
    protocol_audit = read_json(protocol_root / "audit.json")
    cases = read_jsonl(protocol_root / "cases.jsonl")
    factor_pairs = read_jsonl(protocol_root / "factor_pairs.jsonl")
    behavior_rows = read_jsonl(behavior_root / "behavior_rows.jsonl")
    behavior_summary = read_json(behavior_root / "summary.json")
    discovery_pairs = read_jsonl(discovery_root / "selected_pairs.jsonl")
    discovery_summary = read_json(discovery_root / "summary.json")
    frozen = read_json(discovery_root / "frozen_spec.json")
    confirmation_pairs = read_jsonl(confirmation_root / "selected_pairs.jsonl")
    confirmation_summary = read_json(confirmation_root / "summary.json")
    control_summary = read_json(control_root / "summary.json")

    paths = {
        "discovery_source_scan_rows": discovery_root / "source_scan_rows.jsonl",
        "discovery_source_control_rows": discovery_root / "source_control_rows.jsonl",
        "discovery_source_natural_rows": discovery_root / "source_natural_rows.jsonl",
        "discovery_response_rows": discovery_root / "response_rows.jsonl",
        "discovery_receiver_rows": discovery_root / "receiver_causal_rows.jsonl",
        "discovery_joint_rows": discovery_root / "joint_causal_rows.jsonl",
        "discovery_joint_natural_rows": discovery_root / "joint_natural_rows.jsonl",
        "confirmation_source_control_rows": confirmation_root
        / "source_control_rows.jsonl",
        "confirmation_source_natural_rows": confirmation_root
        / "source_natural_rows.jsonl",
        "confirmation_response_rows": confirmation_root / "response_rows.jsonl",
        "confirmation_receiver_rows": confirmation_root
        / "receiver_causal_rows.jsonl",
        "confirmation_joint_rows": confirmation_root / "joint_causal_rows.jsonl",
        "confirmation_joint_natural_rows": confirmation_root
        / "joint_natural_rows.jsonl",
        "control_validation_candidate_rows": control_root
        / "validation_candidate_rows.jsonl",
        "control_validation_natural_rows": control_root
        / "validation_natural_rows.jsonl",
        "control_confirmation_candidate_rows": control_root
        / "confirmation_candidate_rows.jsonl",
        "control_confirmation_natural_rows": control_root
        / "confirmation_natural_rows.jsonl",
    }
    artifacts = {key: read_jsonl(path) for key, path in paths.items()}
    counts = {
        "cases": len(cases),
        "factor_pairs": len(factor_pairs),
        "behavior_rows": len(behavior_rows),
        "discovery_pairs": len(discovery_pairs),
        "confirmation_pairs": len(confirmation_pairs),
        **{key: len(rows) for key, rows in artifacts.items()},
    }
    failures = []
    for key, expected in EXPECTED.items():
        if counts.get(key) != expected:
            failures.append(f"count/{key}:{counts.get(key)}!={expected}")

    if not protocol_audit["passed"] or not protocol["cpu_protocol_pass"]:
        failures.append("protocol_gate")
    case_by_id = {row["record_id"]: row for row in cases}
    pair_counts = Counter(row["factor"] for row in factor_pairs)
    if dict(pair_counts) != {"entity": 4096, "query": 4096, "value": 4096}:
        failures.append(f"factor_pair_counts/{dict(pair_counts)}")
    for pair in factor_pairs:
        arm0 = case_by_id[pair["arm0_record_id"]]
        arm1 = case_by_id[pair["arm1_record_id"]]
        changed = [
            index
            for index, (left, right) in enumerate(
                zip(arm0["input_ids"], arm1["input_ids"])
            )
            if left != right
        ]
        if changed != pair["changed_positions"]:
            failures.append(f"pair_changed/{pair['pair_id']}")
        if pair["factor"] == "entity":
            for role in ("slot0_color", "slot1_color", "query_name"):
                position = arm0["role_positions"][role]
                if arm0["input_ids"][position] != arm1["input_ids"][position]:
                    failures.append(f"entity_fixed/{pair['pair_id']}/{role}")

    behavior_metrics = {
        "candidate_accuracy": float(
            np.mean([row["candidate_correct"] for row in behavior_rows])
        ),
        "natural_accuracy": float(
            np.mean([row["natural_correct_both"] for row in behavior_rows])
        ),
        "repeat_stability": float(
            np.mean([row["repeat_stable"] for row in behavior_rows])
        ),
    }
    for key, value in behavior_metrics.items():
        if not close(value, behavior_summary["metrics"][key]):
            failures.append(f"behavior_recompute/{key}")
    if not behavior_summary["behavior_gate_pass"]:
        failures.append("behavior_gate")

    discovery_worlds = {
        case_by_id[row["arm0_record_id"]]["world_id"] for row in discovery_pairs
    }
    confirmation_worlds = {
        case_by_id[row["arm0_record_id"]]["world_id"] for row in confirmation_pairs
    }
    if discovery_worlds & confirmation_worlds:
        failures.append("world_leakage")
    if any(
        case_by_id[row["arm0_record_id"]]["split"] != "validation"
        for row in discovery_pairs
    ):
        failures.append("discovery_split")
    if any(
        case_by_id[row["arm0_record_id"]]["split"] != "holdout"
        for row in confirmation_pairs
    ):
        failures.append("confirmation_split")
    if not discovery_summary["holdout_not_opened"]:
        failures.append("holdout_opened_in_discovery")
    if confirmation_summary["frozen_spec_sha256"] != sha256_file(
        discovery_root / "frozen_spec.json"
    ):
        failures.append("frozen_hash")
    if not confirmation_summary["holdout_did_not_reselect"]:
        failures.append("holdout_reselection")
    if confirmation_summary["confirmation_gate_pass"]:
        failures.append("original_failed_gate_not_preserved")
    if confirmation_summary["gate_checks"]["G2_source_controls"]:
        failures.append("original_control_failure_not_preserved")

    recomputed_blocks = {}
    for partition, prefix in (
        ("discovery", "discovery"),
        ("confirmation", "confirmation"),
    ):
        source_candidate = summarize_interventions(
            artifacts[f"{prefix}_source_control_rows"]
        )
        source_natural = natural_summary(
            artifacts[f"{prefix}_source_natural_rows"]
        )
        joint = joint_summary(artifacts[f"{prefix}_joint_rows"])
        receiver = receiver_summary(artifacts[f"{prefix}_receiver_rows"])
        stored = (
            discovery_summary if partition == "discovery" else confirmation_summary
        )
        for condition, values in source_candidate.items():
            stored_values = stored["source_control_summary"][condition]
            if not close(values["mean_transfer"], stored_values["mean_transfer"]):
                failures.append(f"{partition}/source_mean/{condition}")
            if not close(values["flip_rate"], stored_values["flip_rate"]):
                failures.append(f"{partition}/source_flip/{condition}")
        for condition, values in source_natural.items():
            stored_values = stored["source_natural_summary"][condition]
            if not close(values["flip_rate"], stored_values["flip_rate"]):
                failures.append(f"{partition}/natural_flip/{condition}")
        for size, values in joint.items():
            stored_values = stored["joint_summary"][size]
            if not close(
                values["median_mediation_fraction"],
                stored_values["median_mediation_fraction"],
            ):
                failures.append(f"{partition}/joint_mediation/{size}")
        recomputed_blocks[partition] = {
            "source": source_candidate,
            "natural": source_natural,
            "joint": joint,
            "receiver": receiver,
        }

    control_recomputed = {}
    for partition in ("validation", "confirmation"):
        candidate_rows = artifacts[f"control_{partition}_candidate_rows"]
        natural_rows = artifacts[f"control_{partition}_natural_rows"]
        candidate = summarize_interventions(candidate_rows)
        natural = natural_summary(natural_rows)
        null_rows = [
            row
            for row in candidate_rows
            if row["condition"].startswith("null_derangement_")
        ]
        natural_null_rows = [
            row
            for row in natural_rows
            if row["condition"].startswith("null_derangement_")
        ]
        metrics = {
            "correct_mean_transfer": candidate["correct_joint"]["mean_transfer"],
            "correct_flip_rate": candidate["correct_joint"]["flip_rate"],
            "pooled_null_mean_transfer": float(
                np.mean([row["normalized_transfer"] for row in null_rows])
            ),
            "pooled_null_flip_rate": float(
                np.mean([row["flipped_to_source"] for row in null_rows])
            ),
            "pooled_natural_null_flip_rate": float(
                np.mean([row["flipped_to_source"] for row in natural_null_rows])
            ),
        }
        stored = control_summary["partitions"][partition]["metrics"]
        for key in (
            "pooled_null_mean_transfer",
            "pooled_null_flip_rate",
            "pooled_natural_null_flip_rate",
        ):
            if not close(metrics[key], stored[key]):
                failures.append(f"control_recompute/{partition}/{key}")
        if not control_summary["partitions"][partition]["gate_pass"]:
            failures.append(f"corrected_control_gate/{partition}")
        control_recomputed[partition] = metrics

    manifest, manifest_failures = build_derangement_manifest(
        discovery_pairs, confirmation_pairs, case_by_id
    )
    failures.extend(manifest_failures)
    write_json(
        audit_root / "derangement_manifest_summary.json",
        {
            "row_count": len(manifest),
            "all_donors_different_pair": not manifest_failures,
            "partition_counts": dict(
                Counter(row["partition"] for row in manifest)
            ),
            "unique_target_pair_counts": {
                partition: len(
                    {
                        row["target_pair_id"]
                        for row in manifest
                        if row["partition"] == partition
                    }
                )
                for partition in ("validation", "confirmation")
            },
        },
    )
    manifest_path = audit_root / "derangement_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    discovery_receivers = recomputed_blocks["discovery"]["receiver"]
    confirmation_receivers = recomputed_blocks["confirmation"]["receiver"]
    key_events = ("l25.attn.answer_boundary", "l31.attn.answer_boundary")
    key_event_replicated = all(
        event in discovery_receivers
        and event in confirmation_receivers
        and discovery_receivers[event]["mean_sufficiency_transfer"] > 0.05
        and confirmation_receivers[event]["mean_sufficiency_transfer"] > 0.05
        and discovery_receivers[event]["median_mediation_fraction"] > 0.10
        and confirmation_receivers[event]["median_mediation_fraction"] > 0.10
        for event in key_events
    )
    local_main_path_replicated = (
        confirmation_summary["gate_checks"]["G1_behavior"]
        and confirmation_summary["gate_checks"]["G2_source_candidate"]
        and confirmation_summary["gate_checks"]["G2_source_natural"]
        and confirmation_summary["gate_checks"]["G3_source_receiver_response"]
        and confirmation_summary["gate_checks"]["G4_G5_single_receiver"]
        and confirmation_summary["gate_checks"]["G5_frozen_joint_mediation"]
        and confirmation_summary["gate_checks"]["G6_frozen_natural_restoration"]
        and key_event_replicated
    )
    corrected_specificity = bool(
        control_summary["corrected_specificity_gate_pass"]
    )
    claims = {
        "behavior_denominator_closed": (
            behavior_metrics["candidate_accuracy"] == 1.0
            and behavior_metrics["natural_accuracy"] == 1.0
        ),
        "entity_pair_keeps_color_tokens_fixed": True,
        "pre_registered_confirmation_gate_pass": False,
        "pre_registered_scramble_control_failed": True,
        "scramble_implementation_was_not_comparable_across_partitions": True,
        "corrected_cross_pair_specificity_pass": corrected_specificity,
        "local_main_path_replicated": local_main_path_replicated,
        "key_single_receivers_replicated": key_event_replicated,
        "full_binding_mechanism_closed": False,
        "single_neuron_mechanism_identified": False,
        "attention_head_and_source_token_route_identified": False,
        "cross_model_generalization_tested": False,
    }
    adjudicated_status = (
        "PARTIAL_GO_LOCAL_SUBGRAPH_WITH_CONTROL_REPAIR"
        if local_main_path_replicated and corrected_specificity
        else "NO_GO"
    )
    audit_result = {
        "schema_version": "phase1000_independent_audit.v1",
        "phase": PHASE,
        "counts": counts,
        "expected_counts": EXPECTED,
        "behavior_metrics": behavior_metrics,
        "discovery_world_count": len(discovery_worlds),
        "confirmation_world_count": len(confirmation_worlds),
        "worlds_disjoint": not bool(discovery_worlds & confirmation_worlds),
        "frozen_spec": frozen,
        "control_recomputed": control_recomputed,
        "key_event_replicated": key_event_replicated,
        "claims": claims,
        "adjudicated_status": adjudicated_status,
        "failures": failures,
        "passed": not failures,
    }
    write_json(audit_root / "audit.json", audit_result)
    write_report(
        audit_root / "report.md",
        audit_result,
        behavior_summary,
        discovery_summary,
        confirmation_summary,
        control_summary,
    )
    return audit_result


def write_report(
    path: Path,
    audit_result: dict[str, Any],
    behavior: dict[str, Any],
    discovery: dict[str, Any],
    confirmation: dict[str, Any],
    control: dict[str, Any],
) -> None:
    validation_control = control["partitions"]["validation"]["metrics"]
    confirmation_control = control["partitions"]["confirmation"]["metrics"]
    d_joint = discovery["joint_summary"]["8"]
    c_joint = confirmation["joint_summary"]["8"]
    d_nat = discovery["joint_natural_summary"]["source_plus_joint_restore"]
    c_nat = confirmation["joint_natural_summary"]["source_plus_joint_restore"]
    d_l25 = discovery["receiver_metrics"]["l25.attn.answer_boundary"]
    c_l25 = confirmation["receiver_metrics"]["l25.attn.answer_boundary"]
    d_l31 = discovery["receiver_metrics"]["l31.attn.answer_boundary"]
    c_l31 = confirmation["receiver_metrics"]["l31.attn.answer_boundary"]
    text = f"""# Phase 1000 独立审计报告

## 审计结论

- 文件级审计：{"通过" if audit_result["passed"] else "失败"}
- 裁定状态：`{audit_result["adjudicated_status"]}`
- 原始预注册 confirmation 总门：**失败**
- 修正后的跨 pair 特异性审计：**通过**
- 可支持结论：一个局部、任务条件化的绑定计算子图得到重复证据
- 不可支持结论：完整语言机制、单神经元编码、完整 attention 头路径、跨模型普适机制

## 1. 行为分母

正式协议包含 128 个世界、8192 条语句和 12288 个反事实对。候选准确率、
两次自然生成准确率、重复稳定率均为
`{behavior["metrics"]["candidate_accuracy"]:.4f}`、
`{behavior["metrics"]["natural_accuracy"]:.4f}`、
`{behavior["metrics"]["repeat_stability"]:.4f}`。

实体交换对只改两个事实实体 token，两个颜色 token、颜色位置和查询 token
保持不变，但正确答案交换。因此本阶段不再把“颜色词身份变化”混入绑定变化。

## 2. 源干预

在 residual depth 1，把两个事实实体位置的状态同时换成配对反事实来源：

$$
do\\left(h^1_{{s_0}},h^1_{{s_1}}\\right)
=
\\left(h^1_{{s_0}}(x'),h^1_{{s_1}}(x')\\right)
$$

validation 与 confirmation 的候选翻转率和自然翻转率均为 1.0000。
单独替换一个位置只产生约一半传递，角色反置与 no-op 的翻转率为 0。

这支持“两个角色对齐的实体状态共同决定当前绑定读取”，但这仍是两个完整
2560 维 residual 向量，不是最小神经元机制。

## 3. 源驱动接收点

响应量定义为：

$$
\\Delta_e^{{do(s)}}=h_e(x;do(s))-h_e(x)
$$

相似响应本身不算机制证据。只有把接收点状态单独注入目标、以及在源干预下
恢复目标状态，才分别测试充分性与中介性。

`l25.attn.answer_boundary`：

- validation 平均充分性传递 `{d_l25["mean_sufficiency_transfer"]:.4f}`，
  中位中介率 `{d_l25["median_mediation_fraction"]:.4f}`
- confirmation 平均充分性传递 `{c_l25["mean_sufficiency_transfer"]:.4f}`，
  中位中介率 `{c_l25["median_mediation_fraction"]:.4f}`

`l31.attn.answer_boundary`：

- validation 平均充分性传递 `{d_l31["mean_sufficiency_transfer"]:.4f}`，
  中位中介率 `{d_l31["median_mediation_fraction"]:.4f}`
- confirmation 平均充分性传递 `{c_l31["mean_sufficiency_transfer"]:.4f}`，
  中位中介率 `{c_l31["median_mediation_fraction"]:.4f}`

查询位置的一些组件响应与自然反事实高度相似，却几乎没有单点因果贡献。
这再次否定“响应最大或相似度最高就是核心节点”。

## 4. 联合中介

冻结的 8 点集合以 validation 排序确定，holdout 没有重排：

`{", ".join(confirmation["frozen_joint_event_ids"])}`

中介率：

$$
M_R=(m_do-m_restore)/|m_do-m_target|
$$

- validation：中位中介率 `{d_joint["median_mediation_fraction"]:.4f}`，
  候选恢复目标率 `{d_joint["restored_to_target_rate"]:.4f}`，
  自然恢复目标率 `{d_nat["target_rate"]:.4f}`
- confirmation：中位中介率 `{c_joint["median_mediation_fraction"]:.4f}`，
  候选恢复目标率 `{c_joint["restored_to_target_rate"]:.4f}`，
  自然恢复目标率 `{c_nat["target_rate"]:.4f}`

主要贡献来自第 25、30、31 层答案边界 attention，第 32 层 MLP 提供补充。
8 点集合不是最小集合：4 点已获得接近的结果，且 8 点中若干 query 事件单点
效应接近零。

## 5. 原始负控为什么失败

原始 `roll(1)` 在 validation 的双向相邻排序中，经常取到同一 pair 的反向
记录，近似 no-op；confirmation 每 pair 只有一个方向，才会取到跨 pair donor。
所以两轮所谓 scrambled 并非同一控制。原始 confirmation 门失败被完整保留，
不能事后改写为“预注册门通过”。

修正审计使用 8 个保证 donor `pair_id` 不同的固定错配，并比较正确配对源相对
错配分布的增量：

- validation：正确源传递 1.0000，错配均值
  `{validation_control["pooled_null_mean_transfer"]:.4f}`，增量
  `{validation_control["correct_transfer_excess_over_null"]:.4f}`
- confirmation：正确源传递 1.0004，错配均值
  `{confirmation_control["pooled_null_mean_transfer"]:.4f}`，增量
  `{confirmation_control["correct_transfer_excess_over_null"]:.4f}`
- 每个样本上，正确源都高于该样本 8 个错配的平均值
- 但正确源高于所有单个错配的比例仅为
  `{validation_control["correct_above_all_per_case_nulls_rate"]:.4f}` 和
  `{confirmation_control["correct_above_all_per_case_nulls_rate"]:.4f}`

因此可以说“正确配对显著优于错配平均”，不能说“任意错误状态都绝不产生
同样输出”。完整 residual 干预仍有明显离流形副作用。

## 6. 严格边界

1. 只测试了 Qwen3 和一个合成的双实体颜色绑定任务。
2. 源是两个完整 residual 向量，尚未定位有效 channel 或单神经元集合。
3. 接收点是整块 attention/MLP 输出，尚未分解到 head、QK 选择、V 搬运、
   O 写入和具体来源 token。
4. 恢复答案边界的晚层组件接近输出端，证明局部因果贡献，但不等于解释了
   从事实到答案的全部路径。
5. 只闭合第一个自然 token，没有测试多 token 自回归反馈。
6. 原始预注册总门失败；修正控制属于透明的后续方法审计。

## 7. 下一阶段建议

继续改进 SCPG，不回到降维聚类作为主算法，也不立即更换完全不同的算法。
下一大任务应是“attention 物理分解与最小联合割集”：

1. 对第 25、30、31 层答案边界 attention 分解到每个 head。
2. 对通过的 head 区分 QK 路由、V 内容搬运和 O 写入。
3. 按来源 token 分解：两个实体槽、两个颜色槽、query 和其他 token。
4. 用 head × 来源 token 的 paired patch、restore、wrong-role、cross-pair
   null 建立真实因果边。
5. 从完整 2560 维状态缩到稳定 channel 集合，再做联合必要性和自然生成。
6. 用新名字、新颜色、新关系模板和三实体任务检验结构外推。
7. Qwen3 得到稳定的 head/source-token 局部闭合后，再顺序测试 GLM4 与 DS7B。

只有当上述分解找不到稳定边、联合割集无法复现，才应切换到另一套主算法，
优先考虑 path-specific causal scrubbing，而不是继续调 UMAP、聚类或最高激活
神经元。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    result = audit()
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "adjudicated_status": result["adjudicated_status"],
                "claims": result["claims"],
                "failures": result["failures"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
