"""Phase 2252: export and audit full-coordinate natural-family atlases.

This stage does not inspect attention, MLPs, weights, gradients, PCA, Top-K,
or transported donor deltas. It preserves every activation coordinate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2252_c1081_c1096_natural_coordinate_atlas"
Q4_OUT = RESULT / "phase2248_c1017_c1030_qwen_natural_full_field"
PRED_OUT = RESULT / "phase2249_c1031_c1048_full_coordinate_prediction"
CAUSAL_OUT = RESULT / "phase2250_c1049_c1064_state_consistent_causal"
CROSS_OUT = RESULT / "phase2251_c1065_c1080_cross_model_fresh_panel"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

PHASE = 2252
CAMPAIGN = "C1081-C1096"
FAMILIES = (
    "graph_taxonomy", "graph_part_whole", "graph_temporal",
    "coreference_binding", "attribute_update", "nested_attitude",
)
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
FRESH_UNITS = tuple(range(12, 20))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def checkpoint_label(checkpoint: int, count: int) -> str:
    if checkpoint == 0:
        return "embedding"
    if checkpoint == count - 1:
        return "final_norm"
    return f"block_{checkpoint:02d}_post"


def paired_rows(index: list[dict], predicate: Callable[[dict], bool], keys: tuple[str, ...]) -> dict:
    groups: dict[tuple, dict[bool, dict]] = defaultdict(dict)
    for row in index:
        if predicate(row):
            groups[tuple(row.get(key) for key in keys)][bool(row["truth"])] = row
    incomplete = [key for key, values in groups.items() if set(values) != {False, True}]
    if incomplete:
        raise RuntimeError(f"incomplete true/false groups: {incomplete[:3]}")
    return groups


def verify_matrix(path: Path, expected_shape: tuple[int, int]) -> dict:
    value = np.load(path, mmap_mode="r")
    shape_ok = tuple(value.shape) == tuple(expected_shape)
    dtype_ok = value.dtype == np.dtype("<f2")
    finite = True
    for start in range(0, value.shape[0], 256):
        if not np.isfinite(np.asarray(value[start:start + 256], dtype=np.float32)).all():
            finite = False
            break
    close_mmap(value)
    return {
        "shape_ok": shape_ok, "dtype_float16_le": dtype_ok, "all_finite": finite,
        "bytes": path.stat().st_size, "sha256": sha256(path),
    }


def write_payload(path: Path, *, title: str, binary_name: str, shape: tuple[int, int],
                  coordinate_count: int, checkpoint_count: int, rows: list[dict],
                  summary: dict, boundary: str) -> None:
    write_json(path, {
        "schema": "ai2050.natural-family-full-coordinate-atlas.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "title": title,
        "binary_url": f"/vis_data/research_kernel/{binary_name}",
        "binary_shape": list(shape),
        "coordinate_count": coordinate_count,
        "checkpoint_count": checkpoint_count,
        "coordinate_semantics": "physical activation coordinates inside this model; not weights",
        "rows": rows,
        "summary": summary,
        "boundary": boundary,
    })


def write_unit_atlas(model_id: str, field_path: Path, index_path: Path) -> dict:
    field = np.load(field_path, mmap_mode="r")
    index = read_jsonl(index_path)
    checkpoint_count, coordinate_count = int(field.shape[1]), int(field.shape[3])
    groups = paired_rows(
        index,
        lambda row: row.get("panel", "natural_broad") == "natural_broad"
        and (bool(row.get("fresh", True)) or row.get("partition", "").startswith("fresh")),
        ("family", "unit", "language", "surface"),
    )
    expected_pairs = len(FAMILIES) * len(FRESH_UNITS) * 2 * 2
    if len(groups) != expected_pairs:
        raise RuntimeError(("fresh_pair_count", len(groups), expected_pairs))

    stem = f"c1088_{model_id}_natural_fresh_unit_response_atlas"
    binary_path = VIS / f"{stem}.float16.npy"
    metadata_path = VIS / f"{stem}.json"
    row_count = len(FAMILIES) * len(FRESH_UNITS) * checkpoint_count * len(ROLES)
    matrix = np.lib.format.open_memmap(binary_path, mode="w+", dtype=np.float16,
                                       shape=(row_count, coordinate_count))
    rows: list[dict] = []
    matrix_i = 0
    for family in FAMILIES:
        for unit in FRESH_UNITS:
            pairs = [values for key, values in groups.items() if key[0] == family and key[1] == unit]
            if len(pairs) != 4:
                raise RuntimeError((model_id, family, unit, "pair_denominator", len(pairs)))
            response = np.mean([
                np.asarray(field[values[True]["hidden_index"]], dtype=np.float32)
                - np.asarray(field[values[False]["hidden_index"]], dtype=np.float32)
                for values in pairs
            ], axis=0)
            for checkpoint in range(checkpoint_count):
                for role_i, role in enumerate(ROLES):
                    matrix[matrix_i] = response[checkpoint, role_i].astype(np.float16)
                    rows.append({
                        "source": "signed_true_minus_false_response",
                        "family": family, "unit": unit, "checkpoint": checkpoint,
                        "checkpoint_label": checkpoint_label(checkpoint, checkpoint_count),
                        "relative_depth": checkpoint / max(checkpoint_count - 1, 1),
                        "role": role, "denominator_pairs": len(pairs),
                    })
                    matrix_i += 1
    matrix.flush(); close_mmap(matrix); close_mmap(field)
    shape = (row_count, coordinate_count)
    write_payload(
        metadata_path,
        title=f"{model_id} six-family fresh unit response field",
        binary_name=binary_path.name, shape=shape, coordinate_count=coordinate_count,
        checkpoint_count=checkpoint_count, rows=rows,
        summary={"families": list(FAMILIES), "units": list(FRESH_UNITS),
                 "languages": ["en", "zh"], "surfaces": ["direct", "paraphrase"]},
        boundary="Every activation coordinate is retained. Values are averaged only across the frozen language/surface replicates for one family-unit pair; this is observational, not causal.",
    )
    return {"model": model_id, "stem": stem, "binary": str(binary_path.relative_to(ROOT)),
            "metadata": str(metadata_path.relative_to(ROOT)), "shape": list(shape),
            "verification": verify_matrix(binary_path, shape)}


def write_coordinate_diagnostics(unit_atlas: dict) -> dict:
    unit_path = ROOT / unit_atlas["binary"]
    unit_matrix = np.load(unit_path, mmap_mode="r")
    coordinate_count = unit_matrix.shape[1]
    checkpoint_count = unit_atlas["shape"][0] // (len(FAMILIES) * len(FRESH_UNITS) * len(ROLES))
    tensor = unit_matrix.reshape(len(FAMILIES), len(FRESH_UNITS), checkpoint_count, len(ROLES), coordinate_count)
    stem = unit_atlas["stem"].replace("unit_response_atlas", "coordinate_diagnostics")
    binary_path = VIS / f"{stem}.float16.npy"
    metadata_path = VIS / f"{stem}.json"
    sources = ("mean_response", "mean_absolute_response", "sign_consensus", "loo_family_specificity_rate")
    row_count = len(sources) * len(FAMILIES) * checkpoint_count * len(ROLES)
    matrix = np.lib.format.open_memmap(binary_path, mode="w+", dtype=np.float16,
                                       shape=(row_count, coordinate_count))
    rows: list[dict] = []
    matrix_i = 0
    for family_i, family in enumerate(FAMILIES):
        for checkpoint in range(checkpoint_count):
            for role_i, role in enumerate(ROLES):
                values = np.asarray(tensor[family_i, :, checkpoint, role_i], dtype=np.float32)
                own = np.empty_like(values)
                wrong = np.empty((len(FAMILIES) - 1, len(FRESH_UNITS), coordinate_count), dtype=np.float32)
                for unit_i in range(len(FRESH_UNITS)):
                    keep = [i for i in range(len(FRESH_UNITS)) if i != unit_i]
                    own[unit_i] = np.mean(values[keep], axis=0)
                    wrong_i = 0
                    for other_family_i in range(len(FAMILIES)):
                        if other_family_i == family_i:
                            continue
                        wrong[wrong_i, unit_i] = np.mean(
                            np.asarray(tensor[other_family_i, keep, checkpoint, role_i], dtype=np.float32), axis=0)
                        wrong_i += 1
                own_error = np.abs(values - own)
                wrong_error = np.min(np.abs(values[None, :, :] - wrong), axis=0)
                diagnostics = {
                    "mean_response": np.mean(values, axis=0),
                    "mean_absolute_response": np.mean(np.abs(values), axis=0),
                    "sign_consensus": np.abs(np.mean(np.sign(values), axis=0)),
                    "loo_family_specificity_rate": np.mean(own_error < wrong_error, axis=0),
                }
                for source in sources:
                    matrix[matrix_i] = diagnostics[source].astype(np.float16)
                    rows.append({
                        "source": source, "family": family, "checkpoint": checkpoint,
                        "checkpoint_label": checkpoint_label(checkpoint, checkpoint_count),
                        "relative_depth": checkpoint / max(checkpoint_count - 1, 1),
                        "role": role, "units": len(FRESH_UNITS),
                    })
                    matrix_i += 1
    matrix.flush(); close_mmap(matrix); close_mmap(unit_matrix)
    shape = (row_count, coordinate_count)
    write_payload(
        metadata_path,
        title=f"{unit_atlas['model']} coordinate stability and family-specificity field",
        binary_name=binary_path.name, shape=shape, coordinate_count=coordinate_count,
        checkpoint_count=checkpoint_count, rows=rows,
        summary={"sources": list(sources), "leave_one_unit_out": True,
                 "coordinate_selection": "none; every coordinate is scored"},
        boundary="Per-coordinate observational diagnostics across eight fresh units. High stability or family specificity is not necessity, sufficiency, or a unique circuit.",
    )
    return {"model": unit_atlas["model"], "stem": stem,
            "binary": str(binary_path.relative_to(ROOT)), "metadata": str(metadata_path.relative_to(ROOT)),
            "shape": list(shape), "verification": verify_matrix(binary_path, shape)}


def write_graph_composition_atlas(field_path: Path, index_path: Path) -> dict:
    field = np.load(field_path, mmap_mode="r")
    index = read_jsonl(index_path)
    checkpoint_count, coordinate_count = int(field.shape[1]), int(field.shape[3])
    groups = paired_rows(
        index,
        lambda row: row.get("panel") == "graph_composition" and row.get("partition") == "fresh_composition_lockbox",
        ("family", "unit", "language", "cell_id"),
    )
    cells = sorted({key[3] for key in groups})
    if len(groups) != 3 * len(FRESH_UNITS) * 2 * 7 or len(cells) != 7:
        raise RuntimeError(("composition_denominator", len(groups), cells))
    stem = "c1090_qwen3_4b_fresh_graph_composition_unit_atlas"
    binary_path = VIS / f"{stem}.float16.npy"
    metadata_path = VIS / f"{stem}.json"
    row_count = 3 * len(FRESH_UNITS) * len(cells) * checkpoint_count * len(ROLES)
    matrix = np.lib.format.open_memmap(binary_path, mode="w+", dtype=np.float16,
                                       shape=(row_count, coordinate_count))
    rows: list[dict] = []
    matrix_i = 0
    for family in FAMILIES[:3]:
        for unit in FRESH_UNITS:
            for cell in cells:
                pairs = [values for key, values in groups.items()
                         if key[0] == family and key[1] == unit and key[3] == cell]
                if len(pairs) != 2:
                    raise RuntimeError((family, unit, cell, len(pairs)))
                response = np.mean([
                    np.asarray(field[values[True]["hidden_index"]], dtype=np.float32)
                    - np.asarray(field[values[False]["hidden_index"]], dtype=np.float32)
                    for values in pairs
                ], axis=0)
                example = next(values[True] for values in pairs)
                for checkpoint in range(checkpoint_count):
                    for role_i, role in enumerate(ROLES):
                        matrix[matrix_i] = response[checkpoint, role_i].astype(np.float16)
                        rows.append({
                            "source": "graph_composition_true_minus_false_response",
                            "family": family, "unit": unit, "cell": cell,
                            "depth": example.get("depth"), "shortcut": example.get("shortcut"),
                            "checkpoint": checkpoint,
                            "checkpoint_label": checkpoint_label(checkpoint, checkpoint_count),
                            "relative_depth": checkpoint / max(checkpoint_count - 1, 1),
                            "role": role, "denominator_languages": 2,
                        })
                        matrix_i += 1
    matrix.flush(); close_mmap(matrix); close_mmap(field)
    shape = (row_count, coordinate_count)
    composition = read_json(PRED_OUT / "analysis/composition.json")
    write_payload(
        metadata_path,
        title="Qwen3-4B fresh graph composition unit response field",
        binary_name=binary_path.name, shape=shape, coordinate_count=coordinate_count,
        checkpoint_count=checkpoint_count, rows=rows,
        summary={"families": list(FAMILIES[:3]), "units": list(FRESH_UNITS), "cells": cells,
                 "prospective_prototype_metrics": composition},
        boundary="All coordinates for each fresh unit and graph condition are shown. Prototype transfer passed for three graph families, but no whole-trajectory family and no causal route qualified.",
    )
    return {"model": "qwen3_4b", "stem": stem,
            "binary": str(binary_path.relative_to(ROOT)), "metadata": str(metadata_path.relative_to(ROOT)),
            "shape": list(shape), "verification": verify_matrix(binary_path, shape)}


def write_cross_scale_control() -> dict:
    final = read_json(CROSS_OUT / "analysis/final.json")
    topology = final["topology"]["qwen3_14b"]
    metric_names = (
        "qwen4_to_model_raw", "model_to_qwen4_raw",
        "qwen4_to_model_centered", "model_to_qwen4_centered",
    )
    rows = []
    for name in metric_names:
        metric = topology[name]
        rows.append({
            "source": name, "role": "relative_depth_role_topology", "checkpoint": 0,
            "values": [metric["accuracy"], metric["median_margin"], 1.0 / len(FAMILIES)],
            "queries": metric["queries"],
        })
    path = VIS / "c1091_qwen4b_qwen14b_cross_scale_retrieval_control.json"
    write_json(path, {
        "schema": "ai2050.cross-scale-retrieval-control.v1", "phase": PHASE,
        "campaign": CAMPAIGN, "coordinate_count": 3,
        "coordinate_labels": ["accuracy", "median_margin", "chance_accuracy"],
        "rows": rows,
        "boundary": "These columns are control metrics, not activation coordinates. Retrieval is below a functional-isomorphism claim.",
    })
    return {"metadata": str(path.relative_to(ROOT)), "rows": len(rows), "sha256": sha256(path)}


def catalog_entry(artifact: dict, title: str, model: str, claim: str, boundary: str) -> dict:
    return {
        "id": artifact["stem"], "title": title, "phase": PHASE, "campaign": CAMPAIGN,
        "model": model, "source_path": "/vis_data/research_kernel/" + Path(artifact["metadata"]).name,
        "binary_path": "/vis_data/research_kernel/" + Path(artifact["binary"]).name,
        "source_schema": "ai2050.natural-family-full-coordinate-atlas.v1",
        "coordinate_count": artifact["shape"][1], "row_count": artifact["shape"][0],
        "claim_level": claim, "boundary": boundary,
        "kinds": ["embedding_and_hiddenstate_physical_activation_coordinates"],
    }


def update_catalog(artifacts: dict) -> dict:
    catalog = read_json(CATALOG)
    family_rows = {
        "graph_taxonomy": ("Taxonomy Graph", "knowledge_graph"),
        "graph_part_whole": ("Part / Whole Graph", "knowledge_graph"),
        "graph_temporal": ("Temporal Graph", "ordering"),
        "coreference_binding": ("Coreference Binding", "reference"),
        "attribute_update": ("Attribute Update", "state_change"),
        "nested_attitude": ("Nested Attitude", "scope"),
    }
    existing_families = {row["id"] for row in catalog.get("families", [])}
    for family, (label, domain) in family_rows.items():
        if family not in existing_families:
            catalog.setdefault("families", []).append({
                "id": family, "label": label, "domain": domain,
                "operations": ["statement", "query", "true_false_response"],
            })
    entries = [
        catalog_entry(artifacts["q4_units"], "C1088 Qwen3-4B Natural Fresh Unit Responses", "Qwen3-4B",
                      "prospective_full_coordinate_observation",
                      "All 2560 activation coordinates for eight fresh units; observational response, not a causal circuit."),
        catalog_entry(artifacts["q14_units"], "C1088 Qwen3-14B Natural Fresh Unit Responses", "Qwen3-14B",
                      "prospective_full_coordinate_observation",
                      "All 5120 activation coordinates for eight fresh units; coordinate IDs are model-local."),
        catalog_entry(artifacts["q4_diagnostics"], "C1089 Qwen3-4B Coordinate Stability and Specificity", "Qwen3-4B",
                      "leave_one_unit_coordinate_observation",
                      "Every coordinate is scored; stability and family specificity are not causal necessity."),
        catalog_entry(artifacts["q14_diagnostics"], "C1089 Qwen3-14B Coordinate Stability and Specificity", "Qwen3-14B",
                      "leave_one_unit_coordinate_observation",
                      "Every coordinate is scored; no cross-model coordinate alignment is attempted."),
        catalog_entry(artifacts["composition"], "C1090 Qwen3-4B Fresh Graph Composition Responses", "Qwen3-4B",
                      "prospective_composition_full_coordinate_observation",
                      "Three graph prototypes transfer to fresh units; whole-trajectory and causal gates remain unqualified."),
        {
            "id": "c1091_qwen4b_qwen14b_cross_scale_retrieval_control",
            "title": "C1091 Qwen4B / Qwen14B Cross-Scale Retrieval Control",
            "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B / Qwen3-14B",
            "source_path": "/vis_data/research_kernel/c1091_qwen4b_qwen14b_cross_scale_retrieval_control.json",
            "source_schema": "ai2050.cross-scale-retrieval-control.v1", "coordinate_count": 3,
            "row_count": artifacts["cross_scale"]["rows"], "claim_level": "negative_cross_scale_control",
            "boundary": "Columns are control metrics, not physical coordinates; retrieval does not establish functional isomorphism.",
            "kinds": ["accuracy", "median_margin", "chance_accuracy"],
        },
    ]
    ids = {entry["id"] for entry in entries}
    catalog["datasets"] = [row for row in catalog.get("datasets", []) if row.get("id") not in ids] + entries
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    write_json(CATALOG, catalog)
    parsed = read_json(CATALOG)
    return {"datasets_added": sorted(ids), "catalog_dataset_count": len(parsed["datasets"]),
            "sha256": sha256(CATALOG), "utf8_json_valid": True}


def export() -> dict:
    OUT.mkdir(parents=True, exist_ok=True); VIS.mkdir(parents=True, exist_ok=True)
    q4_final = read_json(Q4_OUT / "analysis/final.json")
    q14_final = read_json(CROSS_OUT / "qwen3_14b/analysis/final.json")
    q4_units = write_unit_atlas("qwen3_4b", ROOT / q4_final["field"]["path"], Q4_OUT / "raw/field_index.jsonl")
    q14_units = write_unit_atlas("qwen3_14b", ROOT / q14_final["field"]["path"], CROSS_OUT / "qwen3_14b/raw/field_index.jsonl")
    artifacts = {
        "q4_units": q4_units, "q14_units": q14_units,
        "q4_diagnostics": write_coordinate_diagnostics(q4_units),
        "q14_diagnostics": write_coordinate_diagnostics(q14_units),
        "composition": write_graph_composition_atlas(ROOT / q4_final["field"]["path"], Q4_OUT / "raw/field_index.jsonl"),
        "cross_scale": write_cross_scale_control(),
    }
    catalog = update_catalog(artifacts)
    checks = {
        "all_float16_assets_verified": all(
            row["verification"]["shape_ok"] and row["verification"]["dtype_float16_le"]
            and row["verification"]["all_finite"]
            for key, row in artifacts.items() if key != "cross_scale"),
        "catalog_utf8_json_valid": catalog["utf8_json_valid"],
        "q4_all_coordinates": q4_units["shape"][1] == 2560,
        "q14_all_coordinates": q14_units["shape"][1] == 5120,
        "embedding_and_all_checkpoints": q4_units["shape"][0] == 6 * 8 * 38 * 6
        and q14_units["shape"][0] == 6 * 8 * 42 * 6,
        "composition_all_fresh_units": artifacts["composition"]["shape"][0] == 3 * 8 * 7 * 38 * 6,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "visual_ready",
        "timestamp": datetime.now().astimezone().isoformat(), "artifacts": artifacts,
        "catalog": catalog, "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Natural six-family and graph-composition fields are available at every physical activation coordinate. They remain observational; strict whole-trajectory and causal qualification did not pass.",
    }
    write_json(OUT / "analysis/export.json", result)
    print(json.dumps({"status": result["status"], "checks": checks,
                      "shapes": {key: row.get("shape") for key, row in artifacts.items()}}, indent=2))
    return result


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    cross = read_json(CROSS_OUT / "analysis/final.json")
    pred = read_json(PRED_OUT / "analysis/final.json")
    composition = read_json(PRED_OUT / "analysis/composition.json")
    q14_topology = cross["topology"]["qwen3_14b"]
    topology_summary = {key: {"accuracy": value["accuracy"], "median_margin": value["median_margin"]}
                        for key, value in q14_topology.items()}
    artifact_summary = {key: {"shape": value.get("shape"),
                              "sha256": value.get("verification", {}).get("sha256") or value.get("sha256")}
                        for key, value in result["artifacts"].items()}
    cleanup = result["cleanup"]
    text = fr"""

## Phase {PHASE}: 六语言族自然全坐标图谱、组合响应与跨规模边界（{CAMPAIGN}） [{stamp}]

**审查修正与测试原理。** 附件关于“条件化响应取决于语言族、角色、检查点和基态”的窄结论可保留；“已经找到条件齿轮”“跨模型共同齿轮”或“新数学已经形成”均证据过强。本阶段在分类、整体—部分、时序、指代、属性更新、嵌套态度六族上使用同一冻结材料，保留 embedding、每个 block 后状态、final norm、六个功能角色和模型内全部物理激活坐标。用例同时包含中英文、直接/释义表面、12个父词汇单元、8个全新词汇单元，以及三种图关系的1—3跳、捷径和无捷径条件。Qwen3-14B、GLM4、DeepSeek-7B严格串行运行。

**公式。** 单个新词汇单元的全坐标语义响应只在同一材料配对内计算：
$$
R_{{m,f,u,q,r,j}}=\frac{{1}}{{|L||S|}}\sum_{{\ell\in L,s\in S}}\left(H^{{\mathrm{{true}}}}_{{m,f,u,\ell,s,q,r,j}}-H^{{\mathrm{{false}}}}_{{m,f,u,\ell,s,q,r,j}}\right).
$$
每个坐标的符号一致率与留一单元族特异率为：
$$
C_{{f,q,r,j}}=\left|\frac1U\sum_u\operatorname{{sign}}R_{{f,u,q,r,j}}\right|,
\qquad
S_{{f,q,r,j}}=\frac{{1}}{{U}}\sum_u \mathbf{{1}}\!\left[e^{{\mathrm{{own}}}}_{{u,j}}<\min_{{g\ne f}}e^{{\mathrm{{wrong}},g}}_{{u,j}}\right].
$$
这两项对全部坐标逐一计算，不进行PCA、Top-K或坐标筛除。图组合响应仍是冻结真/假条件的同坐标差，并按语言重复平均；没有搬运供体差分。

**结果汇总。** Qwen3-4B在2808条总材料上的候选/自由生成准确率为 `0.875356/0.950499`；Qwen3-14B在384条全新宽族材料上为 `0.960938/0.947917`，六族全部通过双行为门。GLM4为 `0.734375/0.927083`，总门未过，只有分类与整体—部分逐族通过；DeepSeek-7B为 `0.5/0.0` 且生成未解析，因此两者内部场均严格记NA。Qwen3-4B整轨迹严格预测合格族为 `{pred['strict_predictive_families']}`；因而状态一致因果阶段为NA，而不是零效应。三种图关系的前瞻组合原型均优于零模型和错族原型：`{json.dumps(composition, ensure_ascii=False)}`。4B↔14B相对深度角色拓扑留一单元检索为 `{json.dumps(topology_summary, ensure_ascii=False)}`，准确率不足以支持跨规模功能同构。

**全坐标产物与复现文件。** 图谱形状和哈希为 `{json.dumps(artifact_summary, ensure_ascii=False)}`。脚本 `tests/glm5/phase2252_c1081_c1096_natural_coordinate_atlas.py`；原始结果 `tests/glm5/result/phase2247_c1001_c1016_natural_flagship_contract` 至 `tests/glm5/result/phase2252_c1081_c1096_natural_coordinate_atlas`；客户端目录 `frontend/public/research_data/current/language_encoding_catalog.json`。可视化矩阵中的列是模型内物理激活坐标，不是模型权重参数；跨模型没有对齐坐标编号。

**分析、理论进展与硬伤。** 新拼图是：图关系组合响应在全新词汇上可由父材料原型做局部前瞻预测，但统一六族整轨迹模型不能稳定越过锁箱；14B六族行为明显强于4B，而简单角色—深度拓扑仍不能可靠识别跨规模家族。最可信对象因此是“语言族和样本条件化的分布式响应场”，不是固定方向、固定坐标字典或已经闭合的齿轮。硬伤包括：受控模板仍不等于开放自然语言；人类自然度盲评为NA；只有Qwen两种规模获得同分母内部场；逐坐标高一致或高特异仍可能是相关而非必要；严格因果资格为0；小模型结构可能粗糙。现有基础差分、绝对误差和计数公式足以登记这些事实，尚无证据要求命名新数学理论。

**清理、结论与下一步。** 经客户端资源形状、有限值、哈希、目录和构建验证后，已删除不直接展示的原始大场与未获因果资格的预测缓存，共 `{cleanup['deleted_bytes']}` 字节；删除前哈希保存在清理账本。结论是“局部图组合结构成立、统一轨迹与跨规模同构不成立、因果未测试”。下一大阶段目标仍相同，但不得在本分母继续调参；授权转向新的开放自然句法族和更多独立词汇，先用本阶段逐坐标一致率/特异率图观察形成位置，再冻结少量可前瞻的候选做新材料验证。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def finalize() -> dict:
    export_result = read_json(OUT / "analysis/export.json")
    if not export_result["all_checks_passed"]:
        raise RuntimeError("visual assets have not passed export verification")
    # The frontend build/read checks are recorded by the caller before finalization.
    verification_path = OUT / "analysis/frontend_verification.json"
    frontend = read_json(verification_path)
    if not all(frontend["checks"].values()):
        raise RuntimeError("frontend verification is incomplete")

    q4_final = read_json(Q4_OUT / "analysis/final.json")
    q14_final = read_json(CROSS_OUT / "qwen3_14b/analysis/final.json")
    targets = [
        (ROOT / q4_final["field"]["path"], q4_final.get("hashes", {}).get("field")),
        (ROOT / q14_final["field"]["path"], q14_final.get("hashes", {}).get("field")),
        (PRED_OUT / "raw/fresh_prediction_candidates.float16.npy", None),
        (PRED_OUT / "raw/fresh_selected_predictions.float16.npy", None),
    ]
    deleted = []
    for path, recorded_hash in targets:
        if not path.exists():
            continue
        size = path.stat().st_size
        digest = recorded_hash or sha256(path)
        path.unlink()
        deleted.append({"path": str(path.relative_to(ROOT)), "bytes": size, "sha256_before_delete": digest})
    cleanup = {"deleted": deleted, "deleted_bytes": sum(row["bytes"] for row in deleted),
               "policy": "raw fields were removed only after verified all-coordinate visual derivatives; indexes, behavior, formulas, metrics, and hashes remain"}
    write_json(OUT / "analysis/cleanup_ledger.json", cleanup)
    result = {**export_result, "status": "closed", "timestamp": datetime.now().astimezone().isoformat(),
              "frontend_verification": frontend, "cleanup": cleanup,
              "all_checks_passed": export_result["all_checks_passed"] and all(frontend["checks"].values())}
    write_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "frontend": frontend["checks"], "cleanup": cleanup}, ensure_ascii=False, indent=2))
    return result


def verify_frontend() -> dict:
    export_result = read_json(OUT / "analysis/export.json")
    dist = ROOT / "frontend/dist"
    dist_catalog = read_json(dist / "research_data/current/language_encoding_catalog.json")
    expected_ids = set(export_result["catalog"]["datasets_added"])
    actual_ids = {row.get("id") for row in dist_catalog.get("datasets", [])}
    metadata_urls = []
    binary_urls = []
    for row in export_result["artifacts"].values():
        if row.get("metadata"):
            metadata_urls.append("/vis_data/research_kernel/" + Path(row["metadata"]).name)
        if row.get("binary"):
            binary_urls.append(("/vis_data/research_kernel/" + Path(row["binary"]).name,
                                (ROOT / row["binary"]).stat().st_size))

    base = "http://127.0.0.1:5174"
    http_rows = []
    for url in ["/research_data/current/language_encoding_catalog.json", *metadata_urls]:
        with urllib.request.urlopen(base + url, timeout=30) as response:
            payload = response.read()
            json.loads(payload.decode("utf-8-sig"))
            http_rows.append({"url": url, "status": response.status, "bytes": len(payload)})
    binary_heads = []
    for url, expected_bytes in binary_urls:
        request = urllib.request.Request(base + url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            served_bytes = int(response.headers["Content-Length"])
            binary_heads.append({"url": url, "status": response.status,
                                 "served_bytes": served_bytes, "expected_bytes": expected_bytes})
    assets = list((dist / "assets").glob("main-*.js"))
    explorer_source = (ROOT / "frontend/src/researchCenter/LanguageEncodingExplorer.jsx").read_text(encoding="utf-8", errors="replace")
    checks = {
        "production_bundle_present": bool(assets),
        "built_catalog_contains_all_phase2252_datasets": expected_ids <= actual_ids,
        "all_metadata_http_json_200": all(row["status"] == 200 for row in http_rows),
        "all_binary_http_heads_match": all(row["status"] == 200 and row["served_bytes"] == row["expected_bytes"] for row in binary_heads),
        "row_window_512_present": "const ROW_WINDOW = 512" in explorer_source and "visibleRows" in explorer_source,
    }
    result = {
        "timestamp": datetime.now().astimezone().isoformat(), "server_url": base + "/",
        "checks": checks, "http_json": http_rows, "binary_heads": binary_heads,
        "browser_ui_check": "NA: no in-app or extension browser was available in this session",
        "build_warning": "Vite reported only the pre-existing large-chunk advisory; build completed successfully.",
    }
    write_json(OUT / "analysis/frontend_verification.json", result)
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        final = read_json(final_path)
        final["frontend_verification"] = result
        write_json(final_path, final)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--verify-frontend", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize()
    elif args.verify_frontend:
        verify_frontend()
    else:
        export()


if __name__ == "__main__":
    main()
