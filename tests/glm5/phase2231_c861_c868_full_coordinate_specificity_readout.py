#!/usr/bin/env python3
"""C861-C868 full-coordinate coefficient specificity readout.

This append-only stage reads the complete coordinate coefficients retained in
the C860 visualization.  It uses no PCA, Top-K, cosine similarity, model
weights, attention/MLP internals, or new model execution.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
SOURCE_META = ROOT / "frontend/public/vis_data/research_kernel/c860_sample_local_interaction_atlas.json"
SOURCE_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c860_sample_local_interaction_atlas.float16.npy"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c868_coefficient_specificity_atlas.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c868_coefficient_specificity_atlas.float16.npy"
OUT = RESULT / "phase2231_c861_c868_full_coordinate_coefficient_specificity"
sys.path.insert(0, str(TESTS))

import phase2227_c817_c860_sample_local_interaction_campaign as campaign


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def freeze() -> dict:
    for part in ("protocol", "analysis", "audit", "raw"):
        (OUT / part).mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": 2231,
        "campaign": "C861-C868",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_readout": True,
        "source": str(SOURCE_META.relative_to(ROOT)),
        "object": "full-coordinate decomposition of frozen sample-local predictive coefficients",
        "questions": [
            "Are same-family coefficients closer across language than different-family coefficients within language?",
            "How much coordinate-wise variation is shared, family-associated, language-associated, or transform-associated?",
            "Can family identity be retrieved by nearest full-coordinate coefficient field without coordinate selection?",
        ],
        "allowed": ["coordinate-wise arithmetic", "mean absolute difference", "sign disagreement", "exact nearest full-field retrieval"],
        "forbidden": ["PCA", "Top-K", "cosine", "coordinate screening", "new model execution", "causal language"],
        "missingness": "Only groups with nonzero discovery availability are analyzed; zero-filled unavailable visual rows are excluded.",
        "human_review": "NA_not_applicable_no_new_material",
    }
    path = OUT / "protocol/preregistration.json"
    if not path.exists():
        save(path, protocol)
    return load(path)


def coefficient_groups() -> tuple[dict[str, np.ndarray], list[dict]]:
    meta = load(SOURCE_META)
    matrix = np.load(SOURCE_BINARY, mmap_mode="r")
    group_rows: dict[str, list[tuple[tuple, int]]] = {}
    for row_i, row in enumerate(meta["row_metadata"]):
        if row.get("kind") != "coordinate_coefficient":
            continue
        key = (row["feature"], int(row["checkpoint"]), row["role"])
        group_rows.setdefault(row["group"], []).append((key, row_i))
    groups = {}
    row_schema = None
    for group, entries in group_rows.items():
        ordered = sorted(entries)
        schema = [{"feature": key[0], "checkpoint": key[1], "role": key[2]} for key, _ in ordered]
        if row_schema is None:
            row_schema = schema
        elif row_schema != schema:
            raise RuntimeError("coefficient row schema mismatch")
        groups[group] = np.stack([np.asarray(matrix[row_i], dtype=np.float32) for _, row_i in ordered])
    return groups, row_schema or []


def parts(label: str) -> tuple[str, str, int]:
    family, language, transform = label.split("|")
    return family, language, int(transform[1:])


def pair_metric(left: np.ndarray, right: np.ndarray) -> dict:
    delta = left - right
    return {
        "mean_absolute_difference": float(np.mean(np.abs(delta))),
        "root_mean_square_difference": float(np.sqrt(np.mean(delta * delta))),
        "sign_disagreement": float(np.mean(np.sign(left) != np.sign(right))),
        "both_near_zero_rate": float(np.mean((np.abs(left) < 1e-4) & (np.abs(right) < 1e-4))),
    }


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def integrate_catalog(shape: list[int]) -> None:
    catalog = load(CATALOG) if CATALOG.exists() else {"datasets": []}
    datasets = catalog.setdefault("datasets", [])
    entry = {
        "id": "c868_coefficient_specificity_atlas",
        "title": "C868 共享-语言-语义族逐坐标系数图谱",
        "type": "full_coordinate_coefficient_decomposition",
        "metadata_url": "/vis_data/research_kernel/c868_coefficient_specificity_atlas.json",
        "binary_url": "/vis_data/research_kernel/c868_coefficient_specificity_atlas.float16.npy",
        "shape": shape,
        "coordinate_count": campaign.DIM,
    }
    found = next((i for i, value in enumerate(datasets) if value.get("id") == entry["id"]), None)
    if found is None:
        datasets.append(entry)
    else:
        datasets[found] = entry
    save(CATALOG, catalog)


def append_memo(result: dict) -> None:
    marker = "## Phase 2231:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase 2231: 共享、语言与语义族的全坐标系数分解 [${{STAMP}}]

**研究边界与冻结合同。** 本期对应 `C861-C868`，是 Phase 2230 自动授权的同目标观察阶段。它只读取 C860 已展示并保留的逐坐标预测系数，不重新运行模型，不读取权重、Attention/MLP 内部或梯度，不使用 PCA、Top-K、余弦或坐标筛选。所有 2560 个激活坐标进入同一计算；激活坐标及其预测系数均不等于模型权重或因果边。

**测试原理与用例。** 对相同变换，比较同一语言族跨中英文的完整系数场距离，以及同一语言内不同语义族的距离；再把每个物理坐标拆成共享均值、语义族关联残差和语言关联残差。最近邻检索使用完整系数场的平均绝对差，不丢弃低值坐标。

$$
\bar\beta_t=\frac{{1}}{{|F||L|}}\sum_{{f,l}}\beta_{{f,l,t}},\qquad
R^F_{{f,t}}=\frac{{1}}{{|L|}}\sum_l\beta_{{f,l,t}}-\bar\beta_t,
$$
$$
R^L_{{l,t}}=\frac{{1}}{{|F|}}\sum_f\beta_{{f,l,t}}-\bar\beta_t,\qquad
d_1(A,B)=\frac{{1}}{{N}}\sum_{{q,r,j,k}}|A_{{q,r,j,k}}-B_{{q,r,j,k}}|.
$$

**结果汇总。**
```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**分析、理论进展与严格裁决。** {result['strict_conclusion']} 理论主体名称继续保持“条件化输出场闭合理论”，组织原则继续保持“复用—差分—条件化”。本期提供的是预测仪器的全坐标结构审计，不是语言机制闭合，也不授权新基础数学。

**问题、硬伤与瓶颈。** 系数只由每组 8 个 discovery 单元拟合，可能有明显估计不稳定；只有三个语义族同时具备中英文及两个变换的完整系数，外推广度有限；系数结构可能主要反映模板、答案码或通用残差动力学；最近邻距离是基础描述量，不是机制；没有新的自然语言盲评、行为测试或因果干预。

**相关文件。** 脚本 `tests/glm5/phase2231_c861_c868_full_coordinate_specificity_readout.py`；结果 `{OUT.relative_to(ROOT)}`；坐标图谱 `{VISUAL.relative_to(ROOT)}` 与 `{VISUAL_BINARY.relative_to(ROOT)}`。

**结论与下一步授权。** {result['next_authorization']}
""".replace("${STAMP}", stamp)
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    if (OUT / "analysis/final.json").exists():
        return load(OUT / "analysis/final.json")
    protocol = freeze()
    all_groups, schema = coefficient_groups()
    availability = campaign.final("C827-C840")["availability"]
    groups = {key: value for key, value in all_groups.items() if availability.get(key, 0) > 0}
    complete_families = sorted({
        family for family in campaign.FAMILIES
        if all(f"{family}|{language}|t{transform}" in groups for language in campaign.LANGUAGES for transform in (1, 3))
    })
    if not complete_families:
        raise RuntimeError("no complete family-language-transform block")

    pair_rows = []
    for left, right in itertools.combinations(sorted(groups), 2):
        lf, ll, lt = parts(left); rf, rl, rt = parts(right)
        if lt != rt:
            pair_type = "same_family_transform_difference" if lf == rf and ll == rl else "other_transform_difference"
        elif lf == rf and ll != rl:
            pair_type = "same_family_cross_language"
        elif lf != rf and ll == rl:
            pair_type = "same_language_cross_family"
        else:
            pair_type = "cross_family_cross_language"
        pair_rows.append({"left": left, "right": right, "pair_type": pair_type, **pair_metric(groups[left], groups[right])})
    with (OUT / "analysis/full_field_pair_metrics.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in pair_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    summaries = {}
    for pair_type in sorted({row["pair_type"] for row in pair_rows}):
        rows = [row for row in pair_rows if row["pair_type"] == pair_type]
        summaries[pair_type] = {
            "pairs": len(rows),
            "mean_absolute_difference": mean([row["mean_absolute_difference"] for row in rows]),
            "sign_disagreement": mean([row["sign_disagreement"] for row in rows]),
        }

    retrieval = []
    for label, field in sorted(groups.items()):
        family, language, transform = parts(label)
        candidates = [other for other in groups if other != label and parts(other)[2] == transform]
        distances = [(pair_metric(field, groups[other])["mean_absolute_difference"], other) for other in candidates]
        distance, nearest = min(distances)
        nearest_family, nearest_language, _ = parts(nearest)
        retrieval.append({
            "query": label, "nearest": nearest, "distance": distance,
            "same_family": nearest_family == family, "cross_language": nearest_language != language,
        })
    retrieval_accuracy = float(np.mean([row["same_family"] for row in retrieval]))

    derived, derived_rows = [], []
    dominance = []
    for transform in (1, 3):
        block = np.stack([
            groups[f"{family}|{language}|t{transform}"]
            for family in complete_families for language in campaign.LANGUAGES
        ]).reshape(len(complete_families), len(campaign.LANGUAGES), len(schema), campaign.DIM)
        shared = block.mean(axis=(0, 1))
        family_residual = block.mean(axis=1) - shared[None]
        language_residual = block.mean(axis=0) - shared[None]
        family_amplitude = np.mean(np.abs(family_residual), axis=0)
        language_amplitude = np.mean(np.abs(language_residual), axis=0)
        dominance.append({
            "transform": transform,
            "family_mean_absolute_amplitude": float(np.mean(family_amplitude)),
            "language_mean_absolute_amplitude": float(np.mean(language_amplitude)),
            "family_dominant_coordinate_rate": float(np.mean(family_amplitude > language_amplitude)),
        })
        for schema_i, coordinate_schema in enumerate(schema):
            derived.append(shared[schema_i].astype(np.float16))
            derived_rows.append({"kind": "shared", "transform": transform, **coordinate_schema})
            for family_i, family in enumerate(complete_families):
                derived.append(family_residual[family_i, schema_i].astype(np.float16))
                derived_rows.append({"kind": "family_residual", "family": family, "transform": transform, **coordinate_schema})
            for language_i, language in enumerate(campaign.LANGUAGES):
                derived.append(language_residual[language_i, schema_i].astype(np.float16))
                derived_rows.append({"kind": "language_residual", "language": language, "transform": transform, **coordinate_schema})
    matrix = np.stack(derived).astype(np.float16)
    VISUAL.parent.mkdir(parents=True, exist_ok=True)
    np.save(VISUAL_BINARY, matrix, allow_pickle=False)
    visual_meta = {
        "id": "c868_coefficient_specificity_atlas",
        "title": "C868 共享-语言-语义族逐坐标系数图谱",
        "shape": list(matrix.shape), "dtype": "float16", "coordinate_count": campaign.DIM,
        "row_metadata": derived_rows,
        "binary_url": "/vis_data/research_kernel/c868_coefficient_specificity_atlas.float16.npy",
        "binary_sha256": file_hash(VISUAL_BINARY),
        "warning": "Predictive coefficient decomposition; not causal circuitry or weight parameters.",
    }
    save(VISUAL, visual_meta)
    integrate_catalog(list(matrix.shape))

    same_family = summaries.get("same_family_cross_language", {}).get("mean_absolute_difference")
    cross_family = summaries.get("same_language_cross_family", {}).get("mean_absolute_difference")
    family_signal = bool(same_family is not None and cross_family is not None and same_family < cross_family and retrieval_accuracy > 0.5)
    strict_conclusion = (
        "完整逐坐标系数中存在可观察的语义族关联结构，但它只是预测器结构，尚未证明模型内部语义机制。"
        if family_signal else
        "完整逐坐标系数没有表现出足以压过语言/共享动力学的稳定语义族身份；Phase2228 的高预测主要不能归因于已识别的语义族齿轮。"
    )
    next_authorization = (
        "The exact coefficient-readout object is complete. A different object may now freeze a family-residual full-support predictor on new material; no automatic causal patch is authorized."
        if family_signal else
        "The exact coefficient-readout object is complete. The next object must orthogonalize surface, output code, and operation on broader material before another full-support predictor; do not retune this distance rule."
    )
    checks = {
        "source_hash_verified": load(SOURCE_META)["binary_sha256"] == file_hash(SOURCE_BINARY),
        "all_available_groups_accounted": len(groups) == sum(v > 0 for v in availability.values()),
        "complete_block": len(complete_families) >= 3,
        "full_coordinates": matrix.shape[1] == campaign.DIM,
        "visual_exists": VISUAL.exists() and VISUAL_BINARY.exists(),
        "finite": finite(summaries) and finite(dominance) and finite(retrieval),
    }
    result = {
        "phase": 2231, "campaign": "C861-C868", "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "available_groups": sorted(groups), "complete_families": complete_families,
        "pair_summaries": summaries, "nearest_full_field_retrieval": retrieval,
        "same_family_retrieval_accuracy": retrieval_accuracy,
        "coordinate_dominance": dominance,
        "family_signal": family_signal, "visual": {"shape": list(matrix.shape), "sha256": visual_meta["binary_sha256"]},
        "strict_conclusion": strict_conclusion,
        "new_foundational_mathematics_gate": False,
        "same_exact_goal_next_stage": False,
        "same_broad_goal": True,
        "next_authorization": next_authorization,
    }
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({"checks": checks, "family_signal": family_signal, "retrieval_accuracy": retrieval_accuracy, "pair_summaries": summaries}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
