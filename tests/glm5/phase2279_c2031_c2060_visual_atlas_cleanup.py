#!/usr/bin/env python3
"""Publish Phase2274-2278 exact-coordinate atlases and clean undisplayed fields."""
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
FIELD_OUT = RESULT / "phase2275_c1771_c1820_qwen4b_broad_fullfield"
STRUCTURE_OUT = RESULT / "phase2276_c1821_c1890_full_coordinate_structure_tournament"
CAUSAL_OUT = RESULT / "phase2277_c1891_c1960_coordinate_causal_identification"
Q14_OUT = RESULT / "phase2278_c1961_c2030_qwen14_relative_depth_replication"
OUT = RESULT / "phase2279_c2031_c2060_visual_atlas_cleanup"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

PHASE = 2279
CAMPAIGN = "C2031-C2060"
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
FLAGSHIPS = ("patient_binding", "relative_clause_binding", "location_state")

TOKEN_FIELD = FIELD_OUT / "raw/qwen3_4b_broad_all_token_field.float16.npy"
TOKEN_INDEX = FIELD_OUT / "raw/all_token_field_index.jsonl"
ROLE_FIELD = FIELD_OUT / "raw/qwen3_4b_broad_role_field.float16.npy"
Q4_PASSPORT = STRUCTURE_OUT / "atlas/qwen4b_selected_coordinate_passport.float32.npy"
Q4_PASSPORT_ROWS = STRUCTURE_OUT / "atlas/qwen4b_selected_coordinate_passport.rows.jsonl"
SINGLE_EFFECT = CAUSAL_OUT / "atlas/qwen4b_single_coordinate_margin_effect.float32.npy"
SINGLE_EFFECT_ROWS = CAUSAL_OUT / "atlas/qwen4b_single_coordinate_margin_effect.rows.jsonl"
CANDIDATE_MASK = CAUSAL_OUT / "protocol/frozen_candidate_masks.uint8.npy"
Q14_PASSPORT = Q14_OUT / "atlas/qwen14_selected_coordinate_passport.float32.npy"


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
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(value) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def checkpoint_label(q: int, layers: int) -> str:
    if q == 0:
        return "embedding"
    if q == layers + 1:
        return "final_norm"
    return f"block_{q:02d}_post"


def metadata(dataset_id: str, title: str, binary: Path, shape: list[int], rows: list[dict],
             schema: str, model: str, boundary: str) -> dict:
    sha = file_hash(binary)
    value = {"schema": schema, "generated_at": datetime.now().astimezone().isoformat(),
             "phase": PHASE, "campaign": CAMPAIGN, "id": dataset_id, "title": title,
             "model": model, "binary_url": "/vis_data/research_kernel/" + binary.name,
             "binary_shape": shape, "binary_sha256": sha, "coordinate_count": shape[-1],
             "coordinate_semantics": "model-local runtime activation coordinates, not weight parameters",
             "rows": rows, "boundary": boundary}
    target = VIS / f"{dataset_id}.json"
    save_json(target, value)
    return {"id": dataset_id, "title": title, "metadata": target, "binary": binary,
            "shape": shape, "sha256": sha, "rows": len(rows), "model": model,
            "schema": schema, "boundary": boundary}


def publish_token_field() -> dict:
    index = read_jsonl(TOKEN_INDEX)
    chosen = []
    for family in FLAGSHIPS:
        for state in (0, 1):
            hits = [row for row in index if row["family"] == family and int(row["unit"]) == 16
                    and row["surface"] == "direct" and int(row["state"]) == state]
            if len(hits) != 1:
                raise RuntimeError(("token_case", family, state, len(hits)))
            chosen.append(hits[0])
    source = np.load(TOKEN_FIELD, mmap_mode="r")
    row_count = sum(38 * int(row["prompt_length"]) for row in chosen)
    binary = VIS / "c2031_qwen4b_broad_full_token_coordinates.float16.npy"
    binary.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float16,
                                       shape=(row_count, 2560))
    rows, cursor = [], 0
    try:
        for case in chosen:
            for q in range(38):
                for token in range(int(case["prompt_length"])):
                    output[cursor] = source[int(case["hidden_index"]), q, token]
                    role_hits = [role for role, positions in case["role_positions"].items()
                                 if token in positions]
                    rows.append({"row": cursor, "case_id": case["case_id"],
                                 "family": case["family"], "unit": case["unit"],
                                 "surface": case["surface"], "state": case["state"],
                                 "checkpoint": q, "checkpoint_label": checkpoint_label(q, 36),
                                 "token_position": token, "token_id": case["prompt_ids"][token],
                                 "role": "+".join(role_hits) if role_hits else "untyped_token",
                                 "metric": "physical_activation"})
                    cursor += 1
        output.flush()
    finally:
        close(output)
        close(source)
    return metadata(
        "c2031_qwen4b_broad_full_token_coordinates",
        "Qwen3-4B Broad Construction Full Token Coordinates", binary,
        [row_count, 2560], rows, "ai2050.broad-full-token-coordinate-field.v1", "Qwen3-4B",
        "Six frozen representative cases preserve every actual token, embedding, post-block state, final norm, and coordinate; case-local token positions are not semantic alignment.")


def publish_q4_passport() -> dict:
    source = np.load(Q4_PASSPORT, mmap_mode="r")
    binary = VIS / "c2032_qwen4b_predictive_coordinate_passport.float32.npy"
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Q4_PASSPORT, binary)
    rows = read_jsonl(Q4_PASSPORT_ROWS)
    shape = list(source.shape)
    close(source)
    if shape != [84, 2560] or len(rows) != 84:
        raise RuntimeError(("q4_passport", shape, len(rows)))
    return metadata(
        "c2032_qwen4b_predictive_coordinate_passport",
        "Qwen3-4B Predictive Coordinate Passport", binary, shape, rows,
        "ai2050.predictive-coordinate-passport.v1", "Qwen3-4B",
        "All physical coordinates for seven fresh-lockbox predictive structures; prediction is not causal necessity or a semantic-neuron dictionary.")


def publish_causal_scan() -> dict:
    effects = np.load(SINGLE_EFFECT, mmap_mode="r")
    masks = np.load(CANDIDATE_MASK, mmap_mode="r")
    binary = VIS / "c2033_qwen4b_single_coordinate_effect_and_masks.float32.npy"
    output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float32,
                                       shape=(effects.shape[0] + masks.shape[0], 2560))
    output[:effects.shape[0]] = effects
    output[effects.shape[0]:] = masks.astype(np.float32)
    output.flush()
    rows = read_jsonl(SINGLE_EFFECT_ROWS)
    families = ("property_state", "patient_binding", "location_state")
    for i, family in enumerate(families):
        rows.append({"row": len(rows), "family": family, "metric": "frozen_candidate_mask",
                     "checkpoint": {"property_state": 15, "patient_binding": 11,
                                    "location_state": 11}[family], "role": "query"})
    shape = list(output.shape)
    for value in (effects, masks, output):
        close(value)
    return metadata(
        "c2033_qwen4b_single_coordinate_effect_and_masks",
        "Qwen3-4B Single-Coordinate Effects and Candidate Masks", binary, shape, rows,
        "ai2050.single-coordinate-effect-mask.v1", "Qwen3-4B",
        "Identification effects and full masks are shown without Top-K. All three masks failed independent alliance controls and are not causal gears.")


def publish_q14_passport() -> dict:
    source = np.load(Q14_PASSPORT, mmap_mode="r")
    final = load_json(Q14_OUT / "analysis/final.json")
    selected = final["structure"]["selected"]
    family_order = ("patient_binding", "relative_clause_binding", "location_state")
    values, rows = [], []
    for family_i, family in enumerate(family_order):
        model = selected[family]["model"]
        if model == "own_affine":
            entries = ((0, "own_affine_slope"), (1, "own_affine_intercept"),
                       (7, "candidate_coordinate_error"))
        else:
            entries = tuple([(i, f"piecewise_threshold_{i}") for i in range(3)] +
                            [(i + 3, f"piecewise_bin_mean_{i}") for i in range(4)] +
                            [(7, "candidate_coordinate_error")])
        for slot, metric in entries:
            values.append(np.asarray(source[family_i, slot], np.float32))
            rows.append({"row": len(rows), "family": family, "metric": metric,
                         "checkpoint": selected[family]["checkpoint"],
                         "relative_depth": selected[family]["checkpoint"] / 40.0,
                         "role": selected[family]["role"], "model": model,
                         "confirmation_passed": selected[family]["confirmation_passed"]})
    array = np.stack(values).astype(np.float32)
    binary = VIS / "c2034_qwen14_relative_depth_coordinate_passport.float32.npy"
    output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float32, shape=array.shape)
    output[:] = array
    output.flush()
    shape = list(array.shape)
    close(output)
    close(source)
    return metadata(
        "c2034_qwen14_relative_depth_coordinate_passport",
        "Qwen3-14B Relative-Depth Coordinate Passport", binary, shape, rows,
        "ai2050.relative-depth-coordinate-passport.v1", "Qwen3-14B",
        "All fitted coordinates and errors for three mid-layer structures. Zero of three passed confirmation; Qwen3-4B and Qwen3-14B coordinate IDs are not aligned.")


def verify(dataset: dict) -> dict:
    info = load_json(dataset["metadata"])
    value = np.load(dataset["binary"], mmap_mode="r")
    try:
        sample = np.asarray(value[:min(64, len(value))], np.float32)
        checks = {"shape": list(value.shape) == info["binary_shape"] == dataset["shape"],
                  "rows": len(info["rows"]) == value.shape[0],
                  "coordinates": value.shape[-1] == info["coordinate_count"],
                  "finite": bool(np.isfinite(sample).all())}
    finally:
        close(value)
    checks["sha256"] = file_hash(dataset["binary"]) == info["binary_sha256"]
    return {"id": dataset["id"], **checks}


def update_catalog(datasets: list[dict]) -> dict:
    catalog = load_json(CATALOG)
    ids = {row["id"] for row in datasets}
    entries = [{"id": row["id"], "title": row["title"], "phase": PHASE,
                "campaign": CAMPAIGN, "model": row["model"],
                "source_path": "/vis_data/research_kernel/" + row["metadata"].name,
                "binary_path": "/vis_data/research_kernel/" + row["binary"].name,
                "source_schema": row["schema"], "coordinate_count": row["shape"][-1],
                "row_count": row["shape"][0], "claim_level": "full_coordinate_observation",
                "boundary": row["boundary"],
                "kinds": ["embedding_hiddenstate_full_coordinate"]} for row in datasets]
    catalog["datasets"] = entries + [row for row in catalog.get("datasets", [])
                                      if row.get("id") not in ids]
    fields = [{"id": row["id"], "title": row["title"],
               "url": "/vis_data/research_kernel/" + row["metadata"].name,
               "phase": PHASE, "full_coordinate": True,
               "heatmap_type": "embedding_hiddenstate_full_coordinate"} for row in datasets]
    catalog["field_datasets"] = [row for row in catalog.get("field_datasets", [])
                                 if row.get("id") not in ids] + fields
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save_json(CATALOG, catalog)
    return {"dataset_count": len(catalog["datasets"]),
            "field_dataset_count": len(catalog["field_datasets"]),
            "added": sorted(ids)}


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
            "stdout_tail": completed.stdout[-2500:], "stderr_tail": completed.stderr[-2500:],
            "passed": completed.returncode == 0}


def cleanup(paths: list[Path]) -> dict:
    result_root = RESULT.resolve()
    rows, total = [], 0
    for path in paths:
        resolved = path.resolve()
        if result_root not in resolved.parents:
            raise RuntimeError(("outside_result", str(resolved)))
        if not path.exists():
            rows.append({"path": str(path.relative_to(ROOT)), "status": "already_absent",
                         "bytes_deleted": 0})
            continue
        size, sha = path.stat().st_size, file_hash(path)
        path.unlink()
        rows.append({"path": str(path.relative_to(ROOT)),
                     "status": "deleted_after_verified_visual_derivative",
                     "sha256_before": sha, "bytes_deleted": size})
        total += size
    ledger = {"files": rows, "bytes_deleted": total}
    save_json(OUT / "cleanup/ledger.json", ledger)
    return ledger


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 广构式逐坐标图谱发布、清理与大阶段裁决（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 本期不再从热力图挑选机制，而是发布已经揭盲的四类完整坐标对象：Qwen3-4B 三个代表构式的六个 state0/state1 全 token 轨迹，包含 embedding、36 个 block 后状态和 final norm；Phase2276 七个预测结构的 84 行全坐标护照；Phase2277 的 36 行单坐标删除效应与三个完整候选掩码；Phase2278 三个 Qwen3-14B 中层结构的有效函数参数与逐坐标误差。每个二进制文件都保存逐行语义、形状和 SHA-256。这里的“参数级显示”指每个 HiddenState 激活坐标的具体值，不把它误称为权重参数。

**公式。** 可视化行与物理激活坐标保持一一对应：

$$
V_{{m,j}}=H_{{i,q,t,j}},\qquad j=1,\ldots,d,
$$

或对因果扫描：

$$
V_{{m,j}}=M_i(H_i^1-R_{{i,j}}e_j)-M_i(H_i^1).
$$

不做 PCA、Top-K、余弦压缩或坐标重排。跨模型图谱分别使用 2560 与 5120 列，绝不按列号对齐。

**结果汇总。** 数据集：`{json.dumps(result['datasets'], ensure_ascii=False)}`。逐项验证：`{json.dumps(result['verification'], ensure_ascii=False)}`。目录更新：`{json.dumps(result['catalog'], ensure_ascii=False)}`。前端构建：`{json.dumps(result['frontend_build'], ensure_ascii=False)}`。清理账本：`{json.dumps(result['cleanup'], ensure_ascii=False)}`。总检查：`{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析与理论进展。** 本大阶段得到三层证据。第一，Qwen3-4B 的 13/16 个受控构式具有可靠双行为资格，其中七个构式存在超过家族均值、共享仿射、错家族、打乱、表面和输出码控制的模型本地同坐标预测结构。第二，把整状态效应拆成单坐标后，三个候选联盟均未在独立等规模控制下复验，且随机掩码表现出强烈非加性，因此不能把单坐标局部效应简单相加成因果齿轮。第三，三个中层 4B 预测拓扑在行为合格的 14B 上均未通过相对层深确认，说明目前的基础函数不是简单跨规模不变量。理论主体仍是“条件化输出场闭合理论”，组织原则仍是“复用—差分—条件化”；更新仅是把候选对象收紧为模型本地、角色与深度条件化、可能高度非加性的响应装配。

**问题、硬伤与瓶颈。** 材料是受控英文且独立人类盲评为 NA；输出仍是元语言代码；4B 到 14B 不能识别纯参数规模效应；坐标函数只建模同坐标输入，跨坐标耦合仍未恢复；Phase2277 的干预可能离开自然分布；代表 token 图只覆盖三个构式；图谱能帮助观察却不能自动给出因果方向。当前没有证据要求发明新数学，也没有证据证明现有数学足够闭合；最诚实的瓶颈是缺少能从大量完整坐标图谱中稳定提取“条件复用与高阶非加性”的基础算法。

**结论与下一步。** {result['strict_conclusion']} 下一阶段目标仍相同，因此授权只利用已发布的完整坐标护照做组合计数：检查七个 4B 预测结构之间的坐标复用模式，以及预测坐标与 Phase2277 因果敏感坐标的精确交集；先做观察和确定性错位控制，不运行新模型、不预设流形或群结构。脚本 `tests/glm5/phase2279_c2031_c2060_visual_atlas_cleanup.py`；结果 `tests/glm5/result/phase2279_c2031_c2060_visual_atlas_cleanup`；图谱 `frontend/public/vis_data/research_kernel/c2031_*` 至 `c2034_*`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load_json(final)
        append_memo(result)
        return result
    VIS.mkdir(parents=True, exist_ok=True)
    datasets = [publish_token_field(), publish_q4_passport(), publish_causal_scan(),
                publish_q14_passport()]
    verification = [verify(row) for row in datasets]
    catalog = update_catalog(datasets)
    build = frontend_build()
    checks = {"datasets_verified": all(all(value for key, value in row.items() if key != "id")
                                        for row in verification),
              "catalog_added_all": len(catalog["added"]) == len(datasets),
              "frontend_build": build["passed"],
              "q4_predictive_seven": len(load_json(
                  STRUCTURE_OUT / "analysis/final.json")["lockbox_passed_families"]) == 7,
              "q4_causal_zero": load_json(
                  CAUSAL_OUT / "analysis/final.json")["strict_bidirectional_families"] == [],
              "q14_replication_zero": load_json(
                  Q14_OUT / "analysis/final.json")["structure"]["lockbox_passed_families"] == []}
    if not all(checks.values()):
        raise RuntimeError(("publication_check", checks))
    cleanup_result = cleanup([
        TOKEN_FIELD, ROLE_FIELD,
        Q14_OUT / "raw/qwen3_14b_midlayer_role_field.float16.npy",
        Q14_OUT / "raw/qwen3_14b_relative_role_field.float16.npy",
    ])
    result = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
              "timestamp": datetime.now().astimezone().isoformat(),
              "datasets": [{"id": row["id"], "shape": row["shape"],
                            "rows": row["rows"], "sha256": row["sha256"]} for row in datasets],
              "verification": verification, "catalog": catalog, "frontend_build": build,
              "cleanup": cleanup_result, "checks": checks, "all_checks_passed": all(checks.values()),
              "strict_conclusion": ("The campaign establishes broad model-local exact-coordinate prediction in Qwen3-4B, "
                                    "but neither exact-coordinate causal alliances nor simple Qwen3-14B relative-depth "
                                    "replication; no universal gear or new mathematics is established."),
              "next_authorization": "Deterministic coordinate-reuse and prediction-versus-causal-overlap observation from published atlases."}
    save_json(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
