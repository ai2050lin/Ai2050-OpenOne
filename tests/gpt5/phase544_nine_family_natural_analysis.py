#!/usr/bin/env python3
"""Aggregate Phase544 behavior rows into the frozen qualification matrix."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase544_nine_family_natural_behavior"
PROTOCOL_PATH = OUT_DIR / "phase544_frozen_protocol.json"
STATIC_AUDIT_PATH = OUT_DIR / "phase544_static_audit.json"
MATRIX_PATH = OUT_DIR / "phase544_model_mechanism_qualification.jsonl"
SHARED_PATH = OUT_DIR / "phase544_cross_model_entry_matrix.jsonl"
HISTORICAL_PATH = OUT_DIR / "phase544_historical_72_behavior_audit.jsonl"
SUMMARY_PATH = OUT_DIR / "phase544_global_summary.json"
REPORT_PATH = OUT_DIR / "phase544_report.md"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "independent_confirmation")
SURFACES = ("direct_natural", "reordered_natural")
VOCAB_SETS = ("vocab_a", "vocab_b")
Z = 1.96


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denominator = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denominator
    radius = Z * math.sqrt((p * (1 - p) + Z * Z / (4 * n)) / n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def rate(values: Iterable[bool]) -> dict[str, Any]:
    data = [bool(value) for value in values]
    n = len(data)
    count = sum(data)
    lower, upper = wilson(count, n)
    return {
        "n": n,
        "count": count,
        "rate": count / n if n else 0.0,
        "lcb95": lower,
        "ucb95": upper,
    }


def exact_groups(rows: list[dict[str, Any]], key: str, field: str, expected_size: int) -> tuple[dict[str, Any], dict[str, bool]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    values = {
        group_id: len(group) == expected_size and all(bool(row[field]) for row in group)
        for group_id, group in groups.items()
    }
    return rate(values.values()), values


def split_report(rows: list[dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    unit_rate, unit_values = exact_groups(rows, "semantic_unit_id", "semantic_correct", 4)
    pair_rate, _pair_values = exact_groups(rows, "surface_pair_id", "semantic_correct", 2)
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        units[row["semantic_unit_id"]].append(row)
    unrecoverable = rate(
        any(not bool(row["semantic_event_recoverable"]) for row in group)
        for group in units.values()
    )
    strict = rate(row["strict_sequence_correct"] for row in rows)
    protocol = rate(row["protocol_valid"] for row in rows)
    by_surface = {
        surface: rate(row["semantic_correct"] for row in rows if row["surface"] == surface)
        for surface in SURFACES
    }
    by_vocab = {}
    for vocab in VOCAB_SETS:
        local_units = {
            row["semantic_unit_id"]
            for row in rows if row["vocab_set"] == vocab
        }
        by_vocab[vocab] = rate(unit_values[unit_id] for unit_id in sorted(local_units))
    gate_pass = (
        unit_rate["lcb95"] >= gates["semantic_unit_exact_lcb95_min"]
        and pair_rate["lcb95"] >= gates["surface_pair_exact_lcb95_min"]
        and unrecoverable["ucb95"] <= gates["unrecoverable_unit_ucb95_max"]
        and all(item["lcb95"] >= gates["surface_row_lcb95_min"] for item in by_surface.values())
        and all(item["lcb95"] >= gates["vocabulary_unit_lcb95_min"] for item in by_vocab.values())
    )
    return {
        "semantic_unit_exact": unit_rate,
        "surface_pair_exact": pair_rate,
        "unrecoverable_unit": unrecoverable,
        "strict_sequence": strict,
        "protocol_valid": protocol,
        "by_surface": by_surface,
        "by_vocab_set": by_vocab,
        "gate_pass": gate_pass,
    }


def qualification_matrix(all_rows: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        groups[(row["model"], row["family_id"], row["mechanism_id"])].append(row)
    matrix = []
    for (model, family, mechanism), rows in sorted(groups.items()):
        split_reports = {
            split: split_report(
                [row for row in rows if row["split"] == split],
                protocol["behavior_gates"],
            )
            for split in SPLITS
        }
        eligible = all(report["gate_pass"] for report in split_reports.values())
        matrix.append({
            "schema_version": "phase544_model_mechanism_qualification.v1",
            "phase_id": "Phase544",
            "created_at": now(),
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "row_count": len(rows),
            "split_reports": split_reports,
            "behavior_entry_eligible": eligible,
            "physical_collection_authorized": False,
            "physical_stop_reason": (
                "new natural behavior gate failed"
                if not eligible else
                "behavior passed, but a new matching physical discovery/prediction split has not been collected"
            ),
            "historical_phase424_trace_is_new_contract_evidence": False,
            "causal": False,
            "compute_edge": False,
            "single_neuron": False,
            "sealed": False,
        })
    return matrix


def cross_model_rows(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in matrix:
        groups[(row["family_id"], row["mechanism_id"])].append(row)
    result = []
    for (family, mechanism), rows in sorted(groups.items()):
        eligible_models = [row["model"] for row in rows if row["behavior_entry_eligible"]]
        result.append({
            "schema_version": "phase544_cross_model_entry.v1",
            "phase_id": "Phase544",
            "created_at": now(),
            "family_id": family,
            "mechanism_id": mechanism,
            "models_measured": sorted(row["model"] for row in rows),
            "behavior_eligible_models": eligible_models,
            "model_specific_entry_count": len(eligible_models),
            "shared_behavior_entry": len(eligible_models) >= 2,
            "shared_physical_mechanism": False,
            "strict_closed": False,
        })
    return result


def historical_72_audit() -> list[dict[str, Any]]:
    root = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas"
    paths = sorted(root.glob("survey/*/*/behavior.parquet"))
    if len(paths) != 27:
        raise RuntimeError(f"Expected 27 Phase330 behavior partitions, found {len(paths)}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    rows = []
    for (model, family, mechanism), group in frame.groupby(["model", "family_id", "mechanism_id"]):
        heldout = group[group["split"] == "heldout"]
        rows.append({
            "schema_version": "phase544_historical_72_behavior_audit.v1",
            "phase_id": "Phase544",
            "source_phase": "Phase330",
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "prompt_rows": int(len(group)),
            "heldout_prompt_rows": int(len(heldout)),
            "heldout_independent_item_count": int(heldout["item_id"].nunique()),
            "behavior_success_rate": float(group["behavior_success"].mean()),
            "heldout_behavior_success_rate": float(heldout["behavior_success"].mean()),
            "target_match_rate": float(group["target_match"].mean()),
            "target_leak_rate": float((~group["target_absent_from_prompt"]).mean()),
            "phase544_strict_entry_eligible": False,
            "entry_blockers": [
                "only six heldout independent items",
                "heldout was historically exposed",
                "typed counterfactual gate absent",
            ],
        })
    return rows


def report_text(summary: dict[str, Any], matrix: list[dict[str, Any]], shared: list[dict[str, Any]]) -> str:
    eligible = [row for row in matrix if row["behavior_entry_eligible"]]
    shared_entries = [row for row in shared if row["shared_behavior_entry"]]
    old = summary["old_denominator_audit"]
    by_model = summary["behavior_entry_count_by_model"]
    return rf"""# Phase544 九族自然行为资格矩阵与物理入口重选

生成时间：{summary['created_at']}

## 一、旧分母审计

Phase330（阶段330）的九族72机制是有价值的冻结分类分母，但不能直接等同于72个独立自然机制合同。静态复核得到：

- 旧提示中目标直接可见：{old['phase330_target_leak_count']}/{old['phase330_prompt_case_count']}（{old['phase330_target_leak_rate']:.2%}）。目标可见并非对所有任务都错误，但必须按“来源事实、抽取字段、格式目标”分别记账，不能统一解释成知识形成。
- 旧独立留出每机制只有6个题项；即使6/6全对，固定置信下界也无法达到0.90。
- 有{old['phase330_identical_semantic_contract_group_count']}个跨名称完全相同的语义合同组；最大组把输出协议、读出竞争、状态漂移、闭合等多个名称落在同一基础问答上。
- Phase350（阶段350）每族只覆盖1个代表机制，且控制条件允许显式捷径或放松协议。
- Phase424（阶段424）没有严格双盲物理留出；Phase535（阶段535）旧密封已污染。

因此附件“转向九族统一资格矩阵”的方向正确，但不能直接复用旧72分母执行。附件还遗漏了一个硬约束：在零不可恢复率的95%上界不超过0.05时，至少需要73个独立单位；表面改写不能增加这个分母。

## 二、新合同与执行量

- 九族各2个结构不同的代表机制，共18个筛选机制；这只是72机制的入口筛选，不是72机制完成。
- 每机制在发现集和独立确认集各73个独立反事实世界对。
- 每个世界对包含2个相反世界条件和2种自然表面；表面是重复测量，不计入独立样本数。
- 三模型共31,536条冻结提示，执行顺序为 Qwen3（通义千问3）、GLM4（智谱清言4）、DS7B（深度求索7B）。
- 所有输出均为自然答案，没有A/B、0/1等任意代码标签；严格序列化与首个语义事件分账。

行为资格公式为：

$$
G^{{\mathrm{{beh}}}}_{{m,f,k}}=
G_{{\mathrm{{unit}}}}\land G_{{\mathrm{{pair}}}}\land
G_{{\mathrm{{surface}}}}\land G_{{\mathrm{{vocab}}}}\land
G_{{\mathrm{{confirmation}}}}\land G_{{\mathrm{{recoverable}}}}.
$$

## 三、客观结果

| 模型 | 行为合格入口/18 |
|---|---:|
| Qwen3（通义千问3） | {by_model['qwen3']}/18 |
| GLM4（智谱清言4） | {by_model['glm4']}/18 |
| DS7B（深度求索7B） | {by_model['deepseek7b']}/18 |

模型专属合格单元共{len(eligible)}个；至少两个模型共同合格的行为入口共{len(shared_entries)}个。

行为通过只说明模型能在冻结自然合同上稳定完成任务：

$$
G^{{\mathrm{{beh}}}}_{{m,f,k}}=1
\not\Rightarrow
G^{{\mathrm{{physical}}}}_{{m,f,k}}=1.
$$

旧 Phase424 轨迹来自不同且已暴露的合同，不能替代新合同的物理发现和独立预测。因此本阶段没有把任何行为入口升级为物理机制、计算边、因果边或单神经元机制。

## 四、停止与下一步

新物理采集授权数：{summary['physical_collection_authorized_count']}。严格闭合保持0/72。

下一步只允许对行为合格入口建立同合同的全层、多位置、提示处理与生成时间轨迹；若没有合格入口，则先按失败类型修复合同，不允许用旧答案接口或后验阈值补丁恢复候选。物理轨迹通过独立预测后，才进入粗路径必要性、充分性与中介性干预；头、通道和神经元扫描继续关闭。

## 五、证据边界

1. 两个内容知识代表合同主要检验显式来源事实的形成与读取，不代表预训练知识网络已经定位。
2. 翻译合同使用数字表达，覆盖跨语言变换但不代表开放词汇翻译系统。
3. 小模型的内部编码相对大型语言系统仍可能有30%-50%偏差；这个风险不用于降低门槛。
4. 九族与72机制仍是工作本体；重复合同审计证明名称计数不能直接作为机制完成度。

总体进度不因行为资格自动提高：严格机制闭合0/72，全局物理图谱31%，总体科学成熟度26%。
"""


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    audit = read_json(STATIC_AUDIT_PATH)
    if audit["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase544 static audit failed")
    all_rows = []
    execution = {}
    for model in MODELS:
        summary_path = OUT_DIR / f"phase544_{model}_execution_summary.json"
        rows_path = OUT_DIR / f"phase544_{model}_behavior_rows.jsonl"
        if not summary_path.exists() or not rows_path.exists():
            raise RuntimeError(f"Missing Phase544 model result: {model}")
        execution[model] = read_json(summary_path)
        if execution[model]["status"] != "complete":
            raise RuntimeError(f"Incomplete Phase544 result: {model}")
        all_rows.extend(read_jsonl(rows_path))
    matrix = qualification_matrix(all_rows, protocol)
    shared = cross_model_rows(matrix)
    historical = historical_72_audit()
    write_jsonl(MATRIX_PATH, matrix)
    write_jsonl(SHARED_PATH, shared)
    write_jsonl(HISTORICAL_PATH, historical)
    entry_by_model = {
        model: sum(row["behavior_entry_eligible"] for row in matrix if row["model"] == model)
        for model in MODELS
    }
    summary = {
        "schema_version": "phase544_global_summary.v1",
        "phase_id": "Phase544",
        "created_at": now(),
        "status": "behavior_matrix_complete_physical_gate_closed",
        "denominator": {
            "registered_families": 9,
            "registered_mechanisms": 72,
            "screened_representative_mechanisms": 18,
            "model_mechanism_cells": len(matrix),
            "registered_prompt_rows": len(all_rows),
            "independent_world_pairs_per_mechanism_split": 73,
        },
        "behavior_entry_count_by_model": entry_by_model,
        "model_specific_behavior_entry_cell_count": sum(entry_by_model.values()),
        "shared_behavior_entry_mechanism_count": sum(row["shared_behavior_entry"] for row in shared),
        "physical_collection_authorized_count": 0,
        "physical_authorization_reason": (
            "No behavior cell passed" if not any(entry_by_model.values()) else
            "Behavior entries require a new matching physical discovery and prediction protocol"
        ),
        "old_denominator_audit": audit["old_denominator_audit"],
        "evidence_boundary": {
            "behavior_matrix_complete": True,
            "72_mechanisms_behavior_complete": False,
            "new_hidden_state_collection_executed": False,
            "historical_phase424_trace_reused_as_new_evidence": False,
            "causal_intervention_executed": False,
            "single_neuron_scan_executed": False,
            "sealed_split_read": False,
            "strict_closed_mechanisms": 0,
            "strict_mechanism_denominator": 72,
        },
        "progress": {
            "strict_mechanism_closure_percent": 0.0,
            "global_physical_atlas_percent": 31.0,
            "scientific_maturity_percent": 26.0,
        },
        "execution_summaries": execution,
    }
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(report_text(summary, matrix, shared), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    analyze()


if __name__ == "__main__":
    main()
