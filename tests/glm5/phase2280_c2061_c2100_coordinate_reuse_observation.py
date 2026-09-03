#!/usr/bin/env python3
"""Deterministic exact-coordinate reuse observation for Phase 2280."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
STRUCTURE_OUT = RESULT / "phase2276_c1821_c1890_full_coordinate_structure_tournament"
CAUSAL_OUT = RESULT / "phase2277_c1891_c1960_coordinate_causal_identification"
OUT = RESULT / "phase2280_c2061_c2100_coordinate_reuse_observation"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

PHASE = 2280
CAMPAIGN = "C2061-C2100"
PASSPORT = STRUCTURE_OUT / "atlas/qwen4b_selected_coordinate_passport.float32.npy"
PASSPORT_ROWS = STRUCTURE_OUT / "atlas/qwen4b_selected_coordinate_passport.rows.jsonl"
CAUSAL_MASKS = CAUSAL_OUT / "protocol/frozen_candidate_masks.uint8.npy"
FAMILIES = ("recipient_binding", "patient_binding", "relative_clause_binding",
            "location_state", "possession_state", "status_state", "temporal_order")
CAUSAL_ORDER = ("property_state", "patient_binding", "location_state")
OFFSETS = tuple(((k + 1) * 79) % 2560 for k in range(32))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlap(a: np.ndarray, b: np.ndarray) -> dict:
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return {"intersection": intersection, "union": union,
            "jaccard": intersection / max(union, 1),
            "fraction_of_left": intersection / max(int(a.sum()), 1),
            "fraction_of_right": intersection / max(int(b.sum()), 1)}


def shifted_control(a: np.ndarray, b: np.ndarray) -> dict:
    values = [overlap(a, np.roll(b, offset)) for offset in OFFSETS]
    output = {}
    for metric in ("intersection", "jaccard", "fraction_of_left", "fraction_of_right"):
        numbers = np.asarray([row[metric] for row in values], dtype=np.float64)
        output[metric] = {"minimum": float(numbers.min()), "median": float(np.median(numbers)),
                          "maximum": float(numbers.max())}
    return output


def frontend_build() -> dict:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm:
        command = [npm, "run", "build"]
    else:
        candidates = sorted(Path.home().glob(
            "AppData/Local/OpenAI/Codex/runtimes/cua_node/*/bin/node.exe"), reverse=True)
        if not candidates:
            raise FileNotFoundError("No Node runtime")
        command = [str(candidates[0]), str(ROOT / "frontend/node_modules/vite/bin/vite.js"), "build"]
    completed = subprocess.run(command, cwd=ROOT / "frontend", capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=600)
    return {"command": command, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:],
            "passed": completed.returncode == 0}


def update_catalog(dataset_id: str, title: str, metadata_path: Path, binary_path: Path,
                   row_count: int) -> dict:
    catalog = load_json(CATALOG)
    entry = {"id": dataset_id, "title": title, "phase": PHASE, "campaign": CAMPAIGN,
             "model": "Qwen3-4B", "source_path": "/vis_data/research_kernel/" + metadata_path.name,
             "binary_path": "/vis_data/research_kernel/" + binary_path.name,
             "source_schema": "ai2050.coordinate-reuse-overlap.v1", "coordinate_count": 2560,
             "row_count": row_count, "claim_level": "deterministic_exact_coordinate_observation",
             "boundary": "Exact within-model coordinate counts with cyclic-shift controls; no causal or universal-coordinate claim.",
             "kinds": ["embedding_hiddenstate_full_coordinate"]}
    catalog["datasets"] = [entry] + [row for row in catalog.get("datasets", [])
                                             if row.get("id") != dataset_id]
    field = {"id": dataset_id, "title": title,
             "url": "/vis_data/research_kernel/" + metadata_path.name,
             "phase": PHASE, "full_coordinate": True,
             "heatmap_type": "embedding_hiddenstate_full_coordinate"}
    catalog["field_datasets"] = [row for row in catalog.get("field_datasets", [])
                                 if row.get("id") != dataset_id] + [field]
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save_json(CATALOG, catalog)
    return {"dataset_count": len(catalog["datasets"]),
            "field_dataset_count": len(catalog["field_datasets"]), "added": dataset_id}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 七构式坐标复用与预测—因果交集观察（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 本期只读取 Phase2276 已在 fresh lockbox 通过的七个 `candidate_coordinate_pass` 完整布尔掩码，以及 Phase2277 的三个单坐标敏感候选掩码，不运行模型、不重新选择坐标。第一组观察逐坐标复用度 $D_j$：同一个 Qwen3-4B 残差坐标在多少个不同构式的预测结构中同时胜过全部冻结控制。第二组观察 `patient_binding` 和 `location_state` 的预测坐标与因果敏感候选坐标是否精确重合。控制不是随机抽样，而是把第二个完整掩码依次循环错位 32 个预先固定偏移，保留每个掩码的坐标数和稠密度。

**公式。** 七构式复用度与两掩码交集为：

$$
D_j=\sum_{{f=1}}^7 P_{{f,j}},\qquad
J(A,B)=\frac{{|A\cap B|}}{{|A\cup B|}}.
$$

确定性错位控制为：

$$
B^{{(s)}}_j=B_{{(j+s)\bmod 2560}},
\qquad s\in\{{79,158,\ldots,2528\}}.
$$

这只是坐标标签复用计数，不把坐标编号解释为语义原子，也不把错位范围解释为概率显著性。

**结果汇总。** 各族坐标数与复用度分布：`{json.dumps(result['reuse'], ensure_ascii=False)}`。两两交集及错位范围：`{json.dumps(result['pairwise'], ensure_ascii=False)}`。预测—因果交集：`{json.dumps(result['prediction_causal_overlap'], ensure_ascii=False)}`。图谱与检查：`{json.dumps(result['dataset'], ensure_ascii=False)}`、`{json.dumps(result['checks'], ensure_ascii=False)}`。前端构建：`{json.dumps(result['frontend_build'], ensure_ascii=False)}`。

**分析与理论进展。** {result['strict_conclusion']} 如果某个实际交集高于所有循环错位控制，它最多表明同一模型物理坐标标签在两个冻结条件结构中被重复利用；如果落在控制范围内，则没有证据支持特殊坐标复用。预测掩码与因果敏感掩码的重合不能挽救 Phase2277 已失败的联盟验证，只有在其交集被独立干预并优于等规模控制后才能升级为因果候选。

**问题、硬伤与瓶颈。** 七个预测掩码来自不同层和角色，残差维编号可以比较但功能上下文不同；掩码稠密度差异很大；循环错位不是交换性随机化，只是保持规模的确定性基线；`candidate_coordinate_pass` 是逐坐标误差胜出，不是必要性；因果候选掩码在独立验证中已经失败；只有两个构式可做预测—因果交集比较。当前结果仍是观察拼图，不足以提出新数学结构。

**结论与下一步。** 下一阶段不再自动重复同一种掩码计数。若继续同一目标，最有价值的新大阶段应冻结新的自然语言族和多模型材料，再以前述图谱格式积累全 token、全检查点、全坐标观察；只有出现跨材料稳定的复用模式，才为该模式设计独立因果检验。脚本 `tests/glm5/phase2280_c2061_c2100_coordinate_reuse_observation.py`；结果 `tests/glm5/result/phase2280_c2061_c2100_coordinate_reuse_observation`；图谱 `frontend/public/vis_data/research_kernel/c2061_qwen4b_coordinate_reuse_overlap.*`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load_json(final)
        append_memo(result)
        return result
    prereg = {"phase": PHASE, "campaign": CAMPAIGN, "frozen_before_analysis": True,
              "families": FAMILIES, "causal_order": CAUSAL_ORDER, "offsets": OFFSETS,
              "objects": ["candidate_coordinate_pass", "frozen_candidate_mask"],
              "forbidden": ["Top-K", "PCA", "cosine", "mask reselection", "probability claim"]}
    save_json(OUT / "protocol/preregistration.json", prereg)
    rows = read_jsonl(PASSPORT_ROWS)
    passport = np.load(PASSPORT, mmap_mode="r")
    predictive = np.stack([passport[next(row["row"] for row in rows
                                         if row["family"] == family and
                                         row["metric"] == "candidate_coordinate_pass")].astype(bool)
                           for family in FAMILIES])
    causal_all = np.load(CAUSAL_MASKS, mmap_mode="r").astype(bool)
    degree = predictive.sum(axis=0).astype(np.float32)
    degree_counts = {str(value): int(np.sum(degree == value)) for value in range(8)}
    null_degree_counts = []
    for offset_i, offset in enumerate(OFFSETS):
        shifted = np.stack([np.roll(mask, offset * (family_i + 1))
                            for family_i, mask in enumerate(predictive)])
        shifted_degree = shifted.sum(axis=0)
        null_degree_counts.append({str(value): int(np.sum(shifted_degree == value))
                                   for value in range(8)})
    null_ranges = {str(value): {"minimum": min(row[str(value)] for row in null_degree_counts),
                                "median": float(np.median([row[str(value)] for row in null_degree_counts])),
                                "maximum": max(row[str(value)] for row in null_degree_counts)}
                   for value in range(8)}
    pairwise = []
    for left_i, left in enumerate(FAMILIES):
        for right_i in range(left_i + 1, len(FAMILIES)):
            right = FAMILIES[right_i]
            pairwise.append({"left": left, "right": right,
                             "actual": overlap(predictive[left_i], predictive[right_i]),
                             "shifted_control": shifted_control(predictive[left_i], predictive[right_i])})
    causal_lookup = {family: causal_all[CAUSAL_ORDER.index(family)]
                     for family in ("patient_binding", "location_state")}
    prediction_causal = {}
    overlap_masks = []
    for family in ("patient_binding", "location_state"):
        p = predictive[FAMILIES.index(family)]
        c = causal_lookup[family]
        prediction_causal[family] = {"predictive_coordinates": int(p.sum()),
                                     "causal_candidate_coordinates": int(c.sum()),
                                     "actual": overlap(p, c),
                                     "shifted_control": shifted_control(p, c)}
        overlap_masks.append(np.logical_and(p, c).astype(np.float32))
    values = np.concatenate([predictive.astype(np.float32), degree[None],
                             np.stack([causal_lookup["patient_binding"],
                                       causal_lookup["location_state"]]).astype(np.float32),
                             np.stack(overlap_masks)], axis=0)
    binary = VIS / "c2061_qwen4b_coordinate_reuse_overlap.float32.npy"
    binary.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float32, shape=values.shape)
    output[:] = values
    output.flush()
    mmap = getattr(output, "_mmap", None)
    if mmap is not None:
        mmap.close()
    visual_rows = [{"row": i, "family": family, "metric": "predictive_coordinate_pass"}
                   for i, family in enumerate(FAMILIES)]
    visual_rows.append({"row": 7, "family": "all_seven", "metric": "predictive_reuse_degree"})
    visual_rows.extend([
        {"row": 8, "family": "patient_binding", "metric": "causal_candidate_mask"},
        {"row": 9, "family": "location_state", "metric": "causal_candidate_mask"},
        {"row": 10, "family": "patient_binding", "metric": "predictive_and_causal_intersection"},
        {"row": 11, "family": "location_state", "metric": "predictive_and_causal_intersection"},
    ])
    sha = file_hash(binary)
    dataset_id = "c2061_qwen4b_coordinate_reuse_overlap"
    title = "Qwen3-4B Coordinate Reuse and Prediction-Causal Overlap"
    metadata_path = VIS / f"{dataset_id}.json"
    save_json(metadata_path, {"schema": "ai2050.coordinate-reuse-overlap.v1",
              "generated_at": datetime.now().astimezone().isoformat(), "phase": PHASE,
              "campaign": CAMPAIGN, "id": dataset_id, "title": title, "model": "Qwen3-4B",
              "binary_url": "/vis_data/research_kernel/" + binary.name,
              "binary_shape": list(values.shape), "binary_sha256": sha,
              "coordinate_count": 2560, "rows": visual_rows,
              "reuse_degree_counts": degree_counts, "shifted_degree_ranges": null_ranges,
              "pairwise": pairwise, "prediction_causal_overlap": prediction_causal,
              "coordinate_semantics": "Qwen3-4B model-local residual activation coordinate labels",
              "boundary": "Deterministic exact-coordinate observation; shifted controls are not probability tests and causal candidate masks failed independent validation."})
    catalog = update_catalog(dataset_id, title, metadata_path, binary, len(visual_rows))
    build = frontend_build()
    check_value = np.load(binary, mmap_mode="r")
    checks = {"shape": list(check_value.shape) == [12, 2560],
              "finite": bool(np.isfinite(check_value).all()),
              "binary_hash": file_hash(binary) == sha,
              "row_metadata": len(visual_rows) == 12,
              "predictive_counts_preserved": all(int(predictive[i].sum()) ==
                                                int(values[i].sum()) for i in range(7)),
              "causal_counts_preserved": int(values[8].sum()) == int(causal_lookup["patient_binding"].sum())
                                         and int(values[9].sum()) == int(causal_lookup["location_state"].sum()),
              "frontend_build": build["passed"]}
    mmap = getattr(check_value, "_mmap", None)
    if mmap is not None:
        mmap.close()
    result = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
              "timestamp": datetime.now().astimezone().isoformat(),
              "reuse": {"family_coordinate_counts": {family: int(predictive[i].sum())
                                                       for i, family in enumerate(FAMILIES)},
                        "degree_counts": degree_counts, "shifted_control_ranges": null_ranges},
              "pairwise": pairwise, "prediction_causal_overlap": prediction_causal,
              "dataset": {"id": dataset_id, "shape": list(values.shape), "sha256": sha,
                          "catalog": catalog},
              "frontend_build": build, "checks": checks, "all_checks_passed": all(checks.values()),
              "strict_conclusion": "Exact coordinate-label reuse is measurable within Qwen3-4B, but its interpretation is conditional on layer and role; prediction-causal overlap does not restore the failed alliance causal gate.",
              "next_authorization": "A new broad natural-language and multi-model observation campaign, not another post hoc mask refinement."}
    save_json(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
