#!/usr/bin/env python3
"""Final all-model analysis for the frozen Phase330 global atlas."""

from __future__ import annotations

import argparse
import json
import math
import string
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase330_nine_family_case_bank import FAMILY_MECHANISMS, MODELS  # noqa: E402


PHASE = "Phase330"
SCHEMA_VERSION = "8.0.0"
ROUND_DEFAULT = "nine_family_global_atlas"
OUT = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas"
LAYER_COUNTS = {"qwen3": 36, "glm4": 40, "deepseek7b": 28}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_parquet_parts(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"survey/*/*/{filename}"))
    if len(paths) != 27:
        raise RuntimeError(f"Expected 27 {filename} partitions, found {len(paths)}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def load_path_signatures(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("survey/*/*/path_signatures.jsonl"))
    if len(paths) != 27:
        raise RuntimeError(f"Expected 27 path signature partitions, found {len(paths)}")
    return pd.concat([pd.read_json(path, lines=True) for path in paths], ignore_index=True)


def mean_rows(frame: pd.DataFrame, keys: list[str], columns: list[str]) -> list[dict[str, Any]]:
    grouped = frame.groupby(keys, dropna=False)[columns].mean().reset_index()
    result = []
    for row in grouped.to_dict("records"):
        result.append({key: round(float(value), 7) if isinstance(value, float) else value for key, value in row.items()})
    return result


def classify_competitor(row: dict[str, Any]) -> str:
    if row["is_target_first_token"]:
        return "target"
    if row["is_distractor_first_token"]:
        return "registered_distractor"
    raw = str(row["token_text"])
    text = raw.strip().lower()
    if not text:
        return "whitespace_or_control"
    if all(char in string.punctuation or char in "，。！？：；（）【】《》“”‘’" for char in text):
        return "punctuation"
    if text in {
        "the", "a", "an", "answer", "final", "response", "is", "it", "this", "that",
        "yes", "no", "true", "false", "here", "sure", "of", "to", "and", "not",
    }:
        return "surface_protocol"
    if text.replace(".", "", 1).isdigit():
        return "numeric"
    return "open_lexical"


def competitor_summary(root: Path) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    for path in sorted(root.glob("survey/*/*/top50.parquet")):
        table = pq.read_table(path, columns=[
            "model", "family_id", "rank", "token_text", "is_target_first_token", "is_distractor_first_token",
        ]).to_pylist()
        for row in table:
            if int(row["rank"]) != 1:
                continue
            category = classify_competitor(row)
            counts[(row["model"], row["family_id"], category)] += 1
            totals[(row["model"], row["family_id"])] += 1
    return [{
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "model": model,
        "family_id": family,
        "competitor_type": category,
        "top1_count": count,
        "top1_rate": round(count / totals[(model, family)], 7),
    } for (model, family, category), count in sorted(counts.items())]


def heldout_predictions(root: Path, paths: pd.DataFrame) -> list[dict[str, Any]]:
    registry = read_jsonl(root / "path_registry.jsonl")
    selected = {
        (row["model"], row["family_id"], row["mechanism_id"], row["component_type"]): row
        for row in registry
    }
    result = []
    heldout = paths[(paths["split"] == "heldout") & (paths["template_id"] == "template_c")]
    for row in heldout.to_dict("records"):
        key = (row["model"], row["family_id"], row["mechanism_id"], row["component_type"])
        prediction = selected[key]
        if row["position_role"] != prediction["position_role"]:
            continue
        layer_count = int(row["layer_count"])
        difference = abs(int(row["peak_layer"]) - int(prediction["component_layer"]))
        tolerance = max(1, math.ceil(layer_count * 0.10))
        result.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": row["model"],
            "family_id": row["family_id"],
            "mechanism_id": row["mechanism_id"],
            "case_id": row["case_id"],
            "component_type": row["component_type"],
            "conditioned_position_role": row["position_role"],
            "predicted_peak_layer": int(prediction["component_layer"]),
            "observed_peak_layer": int(row["peak_layer"]),
            "absolute_layer_error": difference,
            "normalized_layer_error": round(difference / max(1, layer_count - 1), 7),
            "exact_layer_match": difference == 0,
            "within_10pct_depth": difference <= tolerance,
            "prediction_scope": "peak_layer_conditioned_on_frozen_role",
            "heldout_template": "template_c",
        })
    if len(result) != 3888:
        raise RuntimeError(f"Expected 3888 heldout predictions, got {len(result)}")
    return result


def matched_control_rows(causal: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    condition_means = causal.groupby(
        ["model", "family_id", "mechanism_id", "condition"], dropna=False
    )[["target_margin", "behavior_success", "target_match"]].mean().reset_index()
    pivot = condition_means.pivot_table(
        index=["model", "family_id", "mechanism_id"], columns="condition",
        values=["target_margin", "behavior_success", "target_match"], aggfunc="first",
    )
    rows = []
    for index, row in pivot.iterrows():
        model, family, mechanism = index
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "joint_minus_random_margin": round(float(row[("target_margin", "joint_set_zero")] - row[("target_margin", "matched_random_joint_zero")]), 7),
            "joint_minus_wrong_layer_margin": round(float(row[("target_margin", "joint_set_zero")] - row[("target_margin", "wrong_layer_joint_zero")]), 7),
            "natural_minus_wrong_donor_margin": round(float(row[("target_margin", "matched_natural_state")] - row[("target_margin", "wrong_donor_transplant")]), 7),
            "joint_behavior_change_vs_baseline": round(float(row[("behavior_success", "joint_set_zero")] - row[("behavior_success", "baseline")]), 7),
            "joint_readout_specific_vs_both_controls": bool(
                row[("target_margin", "joint_set_zero")] < row[("target_margin", "matched_random_joint_zero")]
                and row[("target_margin", "joint_set_zero")] < row[("target_margin", "wrong_layer_joint_zero")]
            ),
            "natural_identity_positive": bool(
                row[("target_margin", "matched_natural_state")] > row[("target_margin", "wrong_donor_transplant")]
            ),
            "joint_behavior_necessity_positive": bool(
                row[("behavior_success", "joint_set_zero")] < row[("behavior_success", "baseline")]
            ),
        })
    frame = pd.DataFrame(rows)
    cross_rows = []
    for (family, mechanism), group in frame.groupby(["family_id", "mechanism_id"]):
        cross_rows.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "family_id": family,
            "mechanism_id": mechanism,
            "model_count": int(group["model"].nunique()),
            "cross_model_joint_readout_specific": bool(group["joint_readout_specific_vs_both_controls"].all()),
            "cross_model_natural_identity_positive": bool(group["natural_identity_positive"].all()),
            "cross_model_behavior_necessity_positive": bool(group["joint_behavior_necessity_positive"].all()),
            "mean_joint_minus_random_margin": round(float(group["joint_minus_random_margin"].mean()), 7),
            "mean_joint_minus_wrong_layer_margin": round(float(group["joint_minus_wrong_layer_margin"].mean()), 7),
            "mean_natural_minus_wrong_donor_margin": round(float(group["natural_minus_wrong_donor_margin"].mean()), 7),
        })
    return rows, cross_rows


def claim_registry(
    cross_rows: list[dict[str, Any]], predictions: list[dict[str, Any]], path_summary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    robust_readout = sum(row["cross_model_joint_readout_specific"] for row in cross_rows)
    robust_identity = sum(row["cross_model_natural_identity_positive"] for row in cross_rows)
    robust_behavior = sum(row["cross_model_behavior_necessity_positive"] for row in cross_rows)
    prediction_frame = pd.DataFrame(predictions)
    tolerance_rate = float(prediction_frame["within_10pct_depth"].mean())
    residual_rows = [row for row in path_summary if row["component_type"] == "residual"]
    return [
        {
            "claim_id": "phase330_data_denominator_complete",
            "status": "supported",
            "evidence_level": "engineering_complete",
            "claim": "The frozen nine-family observational and registered causal denominator is complete.",
            "direct_count": "15552 prompt-model cases; 4852224 component events; 4320 balanced causal rows",
            "boundary": "Completeness is not mechanism closure.",
        },
        {
            "claim_id": "phase330_late_target_direction_paths",
            "status": "supported_observationally",
            "evidence_level": "L2",
            "claim": "Target-versus-distractor projections generally peak in late relative depth across all three models.",
            "direct_count": f"{len(path_summary)} model-family-component-role summaries",
            "boundary": "A target-aligned projection is a measurement coordinate, not the full nonlinear runtime mechanism.",
        },
        {
            "claim_id": "phase330_residual_persistent_backbone",
            "status": "supported_observationally",
            "evidence_level": "L2",
            "claim": "Residual target-direction paths are more persistent and less sign-oscillatory than attention/MLP outputs.",
            "direct_count": f"{len(residual_rows)} residual summaries",
            "boundary": "Persistence does not identify where the information was created.",
        },
        {
            "claim_id": "phase330_heldout_peak_prediction",
            "status": "partial_support",
            "evidence_level": "L3",
            "claim": "Discovery-selected peak layers predict heldout template-C peaks within 10% relative depth in many cases.",
            "direct_count": f"{len(predictions)} predictions; tolerance match {tolerance_rate:.4f}",
            "boundary": "Prediction is conditioned on the already selected position role; it does not predict the role itself.",
        },
        {
            "claim_id": "phase330_cross_model_set_readout_necessity",
            "status": "partial_support",
            "evidence_level": "L4",
            "claim": "A small subset of mechanisms shows carrier-set-specific target-margin loss against random and wrong-layer controls in all three models.",
            "direct_count": f"{robust_readout}/72 mechanisms",
            "boundary": "Set-level first-token readout necessity is not full behavior or single-neuron causality.",
        },
        {
            "claim_id": "phase330_cross_model_natural_identity",
            "status": "partial_support",
            "evidence_level": "L4",
            "claim": "Correct natural donor states outperform wrong donors in a minority of mechanisms across all three models.",
            "direct_count": f"{robust_identity}/72 mechanisms",
            "boundary": "The transfer patches selected carrier slices, not the entire natural state.",
        },
        {
            "claim_id": "phase330_cross_model_behavior_closure",
            "status": "not_supported",
            "evidence_level": "L4_negative",
            "claim": "No mechanism has cross-model carrier-set behavioral necessity under the registered two-case audit.",
            "direct_count": f"{robust_behavior}/72 mechanisms",
            "boundary": "Two heldout cases per mechanism are sufficient for screening, not for final behavior-effect estimation.",
        },
        {
            "claim_id": "phase330_single_neuron_or_full_language_closure",
            "status": "not_tested_not_supported",
            "evidence_level": "gate_closed",
            "claim": "Phase330 does not establish single-neuron causality, a complete natural chain, or a closed language-encoding formula.",
            "direct_count": "single-neuron CUDA audit 0; cross-model behavior-necessary mechanisms 0/72",
            "boundary": "The Phase288 single-neuron gate remains closed.",
        },
    ]


def report_text(summary: dict[str, Any], claims: list[dict[str, Any]]) -> str:
    b = summary["behavior_by_model"]
    bmap = {row["model"]: row for row in b}
    return rf"""# Phase330 九族全局物理图谱报告

生成时间：{summary['created_at']}

## 1. 直接完成量

- 9 个语言模式族，72 个机制。
- 1728 个机制内独立题项，3 个模板，5184 条提示。
- 三模型共 15552 个提示模型实例。
- 行为、读出、生成各 15552 行；前 50 竞争词元 777600 行。
- 注意力、MLP、残差 × 来源、查询、末位 × 全层：4852224 个事件。
- 路径签名 139968 行；细组件候选 487296 行；冻结载体成员 864 个。
- 平衡留出因果案例 432 个，10 条件共 4320 行。

这些数量全部完成，但“数据分母完成”不等于“语言机制完成”。

## 2. 行为与读出

| 模型 | 候选集合目标胜出 | 全词表前50 | 生成目标命中 | 协议成功 | 综合行为成功 |
|---|---:|---:|---:|---:|---:|
| Qwen3 | {bmap['qwen3']['candidate_winner_is_target']:.4f} | {bmap['qwen3']['target_in_top50']:.4f} | {bmap['qwen3']['target_match']:.4f} | {bmap['qwen3']['protocol_success']:.4f} | {bmap['qwen3']['behavior_success']:.4f} |
| GLM4 | {bmap['glm4']['candidate_winner_is_target']:.4f} | {bmap['glm4']['target_in_top50']:.4f} | {bmap['glm4']['target_match']:.4f} | {bmap['glm4']['protocol_success']:.4f} | {bmap['glm4']['behavior_success']:.4f} |
| DS7B | {bmap['deepseek7b']['candidate_winner_is_target']:.4f} | {bmap['deepseek7b']['target_in_top50']:.4f} | {bmap['deepseek7b']['target_match']:.4f} | {bmap['deepseek7b']['protocol_success']:.4f} | {bmap['deepseek7b']['behavior_success']:.4f} |

GLM4 原始补全文本对模板 A/C 高度不适配。统一无缓存后该现象仍存在，因此它不是缓存造成的全部问题；原始 completion（补全）接口与 chat（对话）训练接口仍是强混杂。跨模型生成率不能直接解释为内部能力高低。

## 3. 路径测量原理

第一词元目标方向仅作为统一读数坐标：

$$
\hat d = \frac{{W_U(t)-\frac1J\sum_j W_U(d_j)}}{{\left\|W_U(t)-\frac1J\sum_jW_U(d_j)\right\|}}
$$

组件、层和位置角色事件为：

$$
s_{{l,c,r}}=\left\langle \operatorname{{Pool}}_r(o_{{l,c}}),\hat d\right\rangle
$$

冻结路径选择分数为基础乘积，不是新理论公式：

$$
S_{{select}}=\operatorname{{support}}\times\operatorname{{amplitude}}\times\operatorname{{persistence}}
$$

三模型的目标方向峰值总体集中在相对后段；残差路径的后段持续性高、符号翻转少，注意力与 MLP 更振荡。这支持“残差是累积主干、注意力/MLP 是路由与写入事件”的工作图景，但没有证明残差自己存储完整知识。

## 4. 留出预测与因果

- 条件于冻结位置角色的峰层预测：3888 行。
- 10% 相对深度容差命中率：{summary['heldout_prediction']['within_10pct_depth_rate']:.4f}。
- 精确层命中率：{summary['heldout_prediction']['exact_layer_match_rate']:.4f}。
- 联合集合相对随机同规模和错层均更强、且三个模型一致：{summary['cross_model']['joint_readout_specific_mechanisms']}/72。
- 正确自然供体优于错误供体、且三个模型一致：{summary['cross_model']['natural_identity_positive_mechanisms']}/72。
- 三模型均出现可见行为必要性：{summary['cross_model']['behavior_necessity_mechanisms']}/72。

因此当前最强结果只是“少量机制的集合级第一词元读出必要性”。它没有升级成跨模型可见行为必要性。

## 5. 两次质量校准

1. 观测生成的缓存运行时与因果钩子运行时不一致，已对三个模型全部 15552 条生成统一为无缓存并重跑。
2. 初始因果索引 18/19 使包装族目标集中在 7/moon（七/月亮），旧 4320 行判为失效试跑；正式索引改为 18/23 并对三个模型全部重跑。

## 6. 硬伤

1. 所有模型仍是小模型，内部机制相对大型语言系统可能有 30% 到 50% 的结构偏差。
2. 九族是工作分类，不是由 open-set（开放集）检验证明的完备本体。
3. “1728 个独立题项”只在机制内部独立；包装族之间复用了 BASE_QA（基础问答）语义骨架，不能当成 1728 个完全独立语义对象。
4. 第一词元目标方向是线性读数坐标，不能模拟真实多词元、非线性、条件化运行机制。
5. 注意力头和 MLP 分组是组件块，不是单神经元；Phase288 全量单神经元 CUDA 干预仍未运行。
6. 因果筛选每机制只有 2 个留出项，只适合筛选；重要的 5 个正机制必须用其余 4 个留出项扩大确认。
7. 原始补全模板与 GLM4 的对话训练接口不匹配，行为数字不可直接用于模型能力排名。
8. 当前迁移只替换冻结成员切片，不是完整自然状态，也没有验证补偿路径和长程 rollout（生成展开）。
9. 跨模型行为必要机制为 0/72，完整自然闭合链仍为 0。

## 7. 理论约束

“语言是动态模式网络”仍保留为工作框架，且不改名：

$$
Y=\operatorname{{Closure}}\circ\operatorname{{Rollout}}\circ\operatorname{{Competition}}\circ
\operatorname{{Readout}}\circ\operatorname{{ComponentPath}}\circ\operatorname{{StateWrite}}\circ
\operatorname{{Route}}\circ\operatorname{{Trigger}}(X,T)
$$

Phase330 增加的是九族的物理路径坐标和少量集合级读出因果，不是对上述复合映射的完整求解。真实机制很可能是上下文条件化、跨层递归和非线性耦合的；继续给线性公式补参数已进入边际收益递减区。

## 8. 进度口径

- Phase330 工程合同：100%。
- 九族行为/读出/全层路径覆盖：9/9。
- 72 机制细组件候选覆盖：72/72 × 3 模型。
- 72 机制注册集合干预覆盖：72/72 × 3 模型。
- 跨三模型集合级读出特异正结果：{summary['cross_model']['joint_readout_specific_mechanisms']}/72。
- 跨三模型行为必要性：0/72。
- 单神经元因果：0。
- 完整语言编码机制和智能理论实验闭合：未完成。

## 9. 下一阶段

自动继续方向应为 Phase331：**五个跨模型集合读出正机制的扩大留出、通道细化与补偿路径审计**。

1. 冻结本阶段的 5 个正机制，不从其余结果后验增加候选。
2. 使用每机制剩余 4 个留出题项和新的对话模板/原始补全双接口，先验证现象是否复现。
3. 对集合做 leave-one-member-out（逐成员移除）和递归二分，将 MLP 组缩小到神经元候选，但仍保留随机同规模、错层、错供体控制。
4. 记录干预后相邻层注意力、MLP、残差的补偿流，判断行为不变是冗余、旁路还是候选选错。
5. 只有至少一个机制在三个模型上通过读出、完整生成、低副作用和单元级扩大确认，才打开全量单神经元 CUDA 门。
"""


def analyze(round_name: str) -> dict[str, Any]:
    root = OUT / round_name
    quality_files = sorted(root.glob("survey/*/*/quality.json"))
    if len(quality_files) != 27:
        raise RuntimeError(f"Expected 27 survey quality files, found {len(quality_files)}")
    quality_rows = [json.loads(path.read_text(encoding="utf-8")) for path in quality_files]
    if not all(row.get("valid") and row.get("generation_runtime") == "cache_free_uniform" for row in quality_rows):
        raise RuntimeError("Survey quality or uniform generation runtime check failed")
    behavior = load_parquet_parts(root, "behavior.parquet")
    readout = load_parquet_parts(root, "readout.parquet")
    rollout = load_parquet_parts(root, "rollout.parquet")
    paths = load_path_signatures(root)
    causal = pd.read_parquet(root / "causal_rows.parquet")
    carrier_rows = read_jsonl(root / "carrier_sets.jsonl")
    component_candidate_count = sum(
        pq.read_metadata(path).num_rows for path in root.glob("component_scan/*/component_candidates.parquet")
    )
    component_event_count = sum(
        pq.read_metadata(path).num_rows for path in root.glob("survey/*/*/component_events.parquet")
    )
    top50_count = sum(pq.read_metadata(path).num_rows for path in root.glob("survey/*/*/top50.parquet"))

    behavior_columns = [
        "candidate_winner_is_target", "target_in_top50", "target_match", "protocol_success", "behavior_success",
    ]
    behavior_by_model = mean_rows(behavior, ["model"], behavior_columns)
    family_summary = mean_rows(behavior, ["model", "family_id"], behavior_columns)
    mechanism_summary = mean_rows(behavior, ["model", "family_id", "mechanism_id"], behavior_columns)
    template_summary = mean_rows(behavior, ["model", "template_id"], behavior_columns)

    paths = paths.copy()
    paths["relative_onset_depth"] = paths["onset_layer"] / (paths["layer_count"] - 1).clip(lower=1)
    paths["relative_peak_depth"] = paths["peak_layer"] / (paths["layer_count"] - 1).clip(lower=1)
    path_columns = [
        "relative_onset_depth", "relative_peak_depth", "peak_absolute_projection",
        "post_onset_persistence", "sign_flip_count",
    ]
    path_summary = mean_rows(paths, ["model", "family_id", "component_type", "position_role"], path_columns)
    causal_summary = mean_rows(
        causal, ["model", "family_id", "mechanism_id", "condition"],
        ["delta_target_margin_vs_baseline", "target_rank_change_vs_baseline", "target_match", "behavior_success"],
    )
    matched_rows, cross_rows = matched_control_rows(causal)
    predictions = heldout_predictions(root, paths)
    competitor_rows = competitor_summary(root)
    claims = claim_registry(cross_rows, predictions, path_summary)
    prediction_frame = pd.DataFrame(predictions)
    cross_model = {
        "joint_readout_specific_mechanisms": sum(row["cross_model_joint_readout_specific"] for row in cross_rows),
        "natural_identity_positive_mechanisms": sum(row["cross_model_natural_identity_positive"] for row in cross_rows),
        "behavior_necessity_mechanisms": sum(row["cross_model_behavior_necessity_positive"] for row in cross_rows),
        "total_mechanisms": len(cross_rows),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "status": "complete_not_closed",
        "counts": {
            "families": 9,
            "mechanisms": 72,
            "independent_items_within_mechanism": 1728,
            "prompt_cases": 5184,
            "prompt_model_cases": len(behavior),
            "behavior_rows": len(behavior),
            "readout_rows": len(readout),
            "rollout_rows": len(rollout),
            "top50_rows": top50_count,
            "component_event_rows": component_event_count,
            "path_signature_rows": len(paths),
            "component_candidate_rows": component_candidate_count,
            "carrier_set_rows": len(carrier_rows),
            "registered_causal_cases": int(causal["causal_case_id"].nunique()),
            "causal_condition_rows": len(causal),
            "heldout_prediction_rows": len(predictions),
        },
        "behavior_by_model": behavior_by_model,
        "cross_model": cross_model,
        "heldout_prediction": {
            "exact_layer_match_rate": round(float(prediction_frame["exact_layer_match"].mean()), 7),
            "within_10pct_depth_rate": round(float(prediction_frame["within_10pct_depth"].mean()), 7),
            "mean_normalized_layer_error": round(float(prediction_frame["normalized_layer_error"].mean()), 7),
            "scope": "peak_layer_conditioned_on_frozen_role",
        },
        "quality_corrections": [
            "all rollout rows rerun cache-free after model-specific cache interface confound",
            "adjacent 18/19 causal registration invalidated; balanced 18/23 registration fully rerun",
        ],
        "coverage": {
            "family_behavior_readout_layer_path": "9/9",
            "mechanism_component_candidates": "72/72 x 3 models",
            "mechanism_registered_set_interventions": "72/72 x 3 models",
            "single_neuron_cuda_interventions": 0,
            "complete_natural_language_chains": 0,
        },
        "single_unit_intervention_gate_open": False,
        "language_encoding_mechanism_closed": False,
        "intelligence_theory_closed": False,
    }
    write_jsonl(root / "family_summary.jsonl", family_summary)
    write_jsonl(root / "mechanism_summary.jsonl", mechanism_summary)
    write_jsonl(root / "template_summary.jsonl", template_summary)
    write_jsonl(root / "path_shape_summary.jsonl", path_summary)
    write_jsonl(root / "causal_summary.jsonl", causal_summary)
    write_jsonl(root / "matched_control_summary.jsonl", matched_rows)
    write_jsonl(root / "cross_model_mechanism_summary.jsonl", cross_rows)
    write_jsonl(root / "heldout_predictions.jsonl", predictions)
    write_jsonl(root / "competitor_type_summary.jsonl", competitor_rows)
    write_jsonl(root / "claim_registry.jsonl", claims)
    write_json(root / "phase330_global_summary.json", summary)
    (root / "phase330_report.md").write_text(report_text(summary, claims), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
