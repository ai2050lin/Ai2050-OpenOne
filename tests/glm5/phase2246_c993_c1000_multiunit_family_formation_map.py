#!/usr/bin/env python3
"""Observe checkpoint-role formation of multi-unit family specificity."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase2234_c870_c884_broad_family_gear_contract as base  # noqa: E402
import phase2243_c959_c970_multiunit_cross_scale_contract as contract  # noqa: E402
import phase2244_c971_c984_multiunit_cross_model_topology as stage_code  # noqa: E402


PHASE = 2246
CAMPAIGNS = tuple(f"C{i}" for i in range(993, 1001))
SOURCE = ROOT / "tests/glm5/result/phase2244_c971_c984_multiunit_cross_model_topology"
OUT = ROOT / "tests/glm5/result/phase2246_c993_c1000_multiunit_family_formation_map"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MODELS = ("qwen3", "qwen3_14b")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_model(model: str, stage: dict) -> tuple[list[dict], dict]:
    families = stage["qualified_families"][model]
    index = read_rows(SOURCE / model / "analysis/unit_semantic_response_prototype_index.jsonl")
    topology = json.loads((SOURCE / model / "analysis/unit_role_topology.json").read_text(encoding="utf-8"))
    field = np.load(ROOT / topology["prototype_path"], mmap_mode="r")
    by_key = {(row["family"], int(row["unit"])): int(row["prototype_index"]) for row in index}
    try:
        raw = np.stack([[np.asarray(field[by_key[(family, unit)]], dtype=np.float32)
                         for unit in contract.UNITS] for family in families])
    finally:
        mmap = getattr(field, "_mmap", None)
        if mmap is not None:
            mmap.close()
    centered = raw - np.mean(raw, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    rows = []
    for kind, values in (("raw", raw), ("unit_centered", centered)):
        for checkpoint in range(values.shape[2]):
            for role_i, role in enumerate(base.ROLES):
                correct = []
                margins = []
                for unit in contract.UNITS:
                    query = values[:, unit, checkpoint, role_i]
                    other_units = [u for u in contract.UNITS if u != unit]
                    candidates = np.mean(values[:, other_units, checkpoint, role_i], axis=1,
                                         dtype=np.float64).astype(np.float32)
                    distances = stage_code.normalized_euclidean_matrix(query, candidates)
                    for family_i in range(len(families)):
                        order = np.argsort(distances[family_i])
                        predicted_i = int(order[0])
                        same = float(distances[family_i, family_i])
                        wrong = float(min(distances[family_i, j] for j in range(len(families)) if j != family_i))
                        correct.append(predicted_i == family_i)
                        margins.append(wrong - same)
                rows.append({
                    "model": model, "kind": kind, "checkpoint": checkpoint,
                    "relative_depth": checkpoint / max(1, values.shape[2] - 1), "role": role,
                    "families": len(families), "queries": len(correct),
                    "accuracy": float(np.mean(correct)), "median_margin": float(np.median(margins)),
                    "positive_margin_fraction": float(np.mean([x > 0 for x in margins])),
                    "chance_accuracy": 1.0 / len(families),
                })
    summary = {}
    for kind in ("raw", "unit_centered"):
        subset = [row for row in rows if row["kind"] == kind]
        ranked = sorted(subset, key=lambda row: (row["accuracy"], row["median_margin"]), reverse=True)
        by_checkpoint = {}
        for checkpoint in range(raw.shape[2]):
            checkpoint_rows = [row for row in subset if row["checkpoint"] == checkpoint]
            by_checkpoint[str(checkpoint)] = {
                "mean_role_accuracy": float(np.mean([row["accuracy"] for row in checkpoint_rows])),
                "max_role_accuracy": float(np.max([row["accuracy"] for row in checkpoint_rows])),
                "mean_role_margin": float(np.mean([row["median_margin"] for row in checkpoint_rows])),
            }
        summary[kind] = {"best_checkpoint_role": ranked[0], "checkpoint_curve": by_checkpoint}
    return rows, summary


def failure_ledger(stage: dict) -> dict:
    within = {}
    for model, ledger in stage["within_model_retrieval"].items():
        if ledger.get("status") != "closed":
            within[model] = []
            continue
        within[model] = [row for kind in ("raw", "unit_centered")
                         for row in ledger[kind]["rows"] if not row["correct"]]
    cross = [
        {"kind": kind, "direction": direction, **row}
        for kind in ("raw", "unit_centered")
        for direction in ("forward", "reverse")
        for row in stage["cross_model_retrieval"].get(kind, {}).get(direction, {}).get("rows", [])
        if not row["correct"]
    ]
    return {"within_model": within, "cross_model": cross}


def export_visual(rows: list[dict]) -> dict:
    ordered = sorted(rows, key=lambda row: (row["model"], row["kind"], row["checkpoint"], base.ROLES.index(row["role"])))
    array = np.asarray([[row["accuracy"], row["median_margin"], row["positive_margin_fraction"]]
                        for row in ordered], dtype=np.float16)
    stem = "c1000_multiunit_family_formation_map"
    binary = VIS / f"{stem}.float16.npy"
    metadata = VIS / f"{stem}.json"
    np.save(binary, array, allow_pickle=False)
    payload = {
        "schema": "ai2050.multiunit-family-formation-map.v1", "phase": PHASE,
        "campaigns": list(CAMPAIGNS), "dtype": "float16", "shape": list(array.shape),
        "columns": ["accuracy", "median_margin", "positive_margin_fraction"],
        "rows": [{k: row[k] for k in ("model", "kind", "checkpoint", "relative_depth", "role", "families", "queries", "chance_accuracy")}
                 for row in ordered],
        "binary": binary.name, "sha256": sha256(binary),
        "boundary": "Each cell uses every physical coordinate at one checkpoint and role. This is predictive formation timing, not a causal circuit.",
    }
    save(metadata, payload)
    check = np.load(binary, mmap_mode="r")
    try:
        verified = list(check.shape) == payload["shape"] and check.dtype == np.float16 and np.isfinite(check).all()
    finally:
        mmap = getattr(check, "_mmap", None)
        if mmap is not None:
            mmap.close()
    return {"id": stem, "json": metadata.name, "binary": binary.name,
            "shape": list(array.shape), "rows": len(ordered), "sha256": payload["sha256"], "verified": bool(verified)}


def update_catalog(export: dict) -> bool:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8-sig"))
    datasets = catalog.setdefault("datasets", [])
    if export["id"] not in {item.get("id") for item in datasets}:
        datasets.append({
            "id": export["id"], "title": "C1000 Multi-unit Family Formation Map",
            "phase": PHASE, "campaign": "C993-C1000", "model": "Qwen3-4B / Qwen3-14B",
            "source_path": f"/vis_data/research_kernel/{export['json']}",
            "binary_path": f"/vis_data/research_kernel/{export['binary']}",
            "source_schema": "ai2050.multiunit-family-formation-map.v1",
            "coordinate_count": 3, "row_count": export["rows"],
            "claim_level": "exploratory_full_coordinate_formation_map",
            "boundary": "Each score consumes all activation coordinates at one checkpoint-role cell; not a causal or parameter map.",
            "kinds": ["raw", "unit_centered"],
        })
        CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return True
    return False


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    formula = r"""
$$
D_{\ell,r}(f,u;g)=\frac{\operatorname{RMS}\left(P_{f,u,\ell,r}-\bar P_{g,-u,\ell,r}\right)}
{\frac12\left[\operatorname{RMS}(P_{f,u,\ell,r})+\operatorname{RMS}(\bar P_{g,-u,\ell,r})\right]+\epsilon}.
$$
"""
    text = f"""

## Phase {PHASE}: 多语义单元语言族信息的全坐标形成位置图（C993-C1000） [{stamp}]

**目标与算法。** Phase2244 已确认多单元跨规模角色拓扑，但整场分数没有回答族身份在何时、哪个功能角色上变得可读。本期不加载模型、不提出固定齿轮假说；对每个检查点和角色单独使用全部物理激活坐标，执行四单元留一族检索。原始账和“同单元减全族平均”账并列，后者只削弱共享底盘，不筛坐标。

{formula}

**结果。** 每模型、每账的最佳检查点-角色与逐层曲线为 `{json.dumps(result['summaries'], ensure_ascii=False)}`。错误账为 `{json.dumps(result['failures'], ensure_ascii=False)}`。这些分数使用完整2560或5120坐标，不是Top-K、PCA或余弦；但“可读形成”不等于该处独占地执行语言操作。

**理论进展与硬伤。** 形成位置图把“族结构存在”细化为随层和角色变化的响应生态位，符合条件化输出场理论中的状态、角色和深度条件化。硬伤是：每格检索仍是距离读出；四个单元和受控模板有限；早层低分可能来自角色尚未路由，也可能来自测量位置不合适；晚层高分可能只是输出准备；没有删除或救援，因此不是因果时钟。理论名称和RDC原则不变，不授权新数学闭合。

**结论与下一步。** 工程验证 `{result['all_checks_passed']}`。形成图已加入客户端，并可与 C992 的具体激活坐标图联动观察。下一独立阶段需要更自然材料或第三个行为合格模型；继续在同一四单元数据上增加距离变体只会形成分析者自由度，因此本轮数据挖掘在这里结束。

**相关文件。** 脚本 `tests/glm5/phase2246_c993_c1000_multiunit_family_formation_map.py`；结果 `{OUT.relative_to(ROOT)}`；可视化 `frontend/public/vis_data/research_kernel/c1000_multiunit_family_formation_map.json`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    stage = json.loads((SOURCE / "analysis/final.json").read_text(encoding="utf-8"))
    rows = []
    summaries = {}
    for model in MODELS:
        model_rows, summary = analyze_model(model, stage)
        rows.extend(model_rows); summaries[model] = summary
    write_rows(OUT / "analysis/checkpoint_role_formation.jsonl", rows)
    failures = failure_ledger(stage)
    save(OUT / "analysis/error_ledger.json", failures)
    export = export_visual(rows)
    catalog_added = update_catalog(export)
    checks = {
        "source_confirmed": stage["family_specific_topology_confirmed"],
        "all_models_mapped": {row["model"] for row in rows} == set(MODELS),
        "all_checkpoint_role_cells": len(rows) == sum(
            2 * json.loads((SOURCE / model / "analysis/unit_role_topology.json").read_text(encoding="utf-8"))["prototype_shape"][1] * len(base.ROLES)
            for model in MODELS),
        "finite": all(np.isfinite(row["accuracy"]) and np.isfinite(row["median_margin"]) for row in rows),
        "visual_verified": export["verified"],
        "catalog_valid": bool(json.loads(CATALOG.read_text(encoding="utf-8"))),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()), "summaries": summaries,
        "failures": failures, "visual_export": export, "catalog_added": catalog_added,
        "strict_conclusion": "Family identity has checkpoint-role-specific full-coordinate readability; this is observational timing, not causal localization.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
