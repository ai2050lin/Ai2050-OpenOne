#!/usr/bin/env python3
"""Publish Phase2290-2294 exact-coordinate assets and clean raw fields."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
Q4_OUT = RESULT / "phase2290_c2601_c2700_qwen4b_natural_dynamic_field"
PREDICT_OUT = RESULT / "phase2291_c2701_c2800_sample_conditioned_coordinate_tournament"
TRANSPORT_OUT = RESULT / "phase2292_c2801_c2900_full_coordinate_layer_transport"
Q14_OUT = RESULT / "phase2294_c2941_c3020_qwen14_frozen_coordinate_replication"
OUT = RESULT / "phase2295_c3021_c3100_visual_atlas_cleanup"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
Q4_MODEL = ROOT / "models/hf/Qwen3-4B"

PHASE = 2295
CAMPAIGN = "C3021-C3100"
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
REPRESENTATIVE_UNIT = 26
TOKEN_FAMILY = "attitude_event"

Q4_ROLE = Q4_OUT / "raw/qwen3_4b_natural_role_field.float16.npy"
Q4_ROLE_INDEX = Q4_OUT / "raw/role_field_index.jsonl"
Q4_TOKEN = Q4_OUT / "raw/qwen3_4b_representative_all_token_field.float16.npy"
Q4_TOKEN_INDEX = Q4_OUT / "raw/all_token_field_index.jsonl"
PREDICT_ATLAS = PREDICT_OUT / "atlas/selected_coordinate_error_passports.float32.npy"
PREDICT_ROWS = PREDICT_OUT / "atlas/selected_coordinate_error_rows.jsonl"
TRANSPORT_ATLAS = TRANSPORT_OUT / "atlas/layer_transport_error_field.float32.npy"
TRANSPORT_ROWS = TRANSPORT_OUT / "atlas/layer_transport_error_rows.jsonl"
Q14_ROLE = Q14_OUT / "raw/qwen3_14b_selected_role_field.float16.npy"
Q14_ROLE_INDEX = Q14_OUT / "raw/selected_role_field_index.jsonl"
Q14_ATLAS = Q14_OUT / "atlas/qwen3_14b_frozen_coordinate_errors.float32.npy"
Q14_ROWS = Q14_OUT / "atlas/qwen3_14b_frozen_coordinate_error_rows.jsonl"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def checkpoint_label(q: int, layers: int, final_norm: bool = True) -> str:
    if q == 0:
        return "embedding"
    if final_norm and q == layers + 1:
        return "final_norm"
    return f"block_{q:02d}_post"


def metadata(dataset_id: str, title: str, binary: Path, rows: list[dict], schema: str,
             model: str, claim_level: str, boundary: str) -> dict:
    value = np.load(binary, mmap_mode="r")
    shape = list(value.shape)
    finite = bool(np.isfinite(value).all())
    close(value)
    if len(rows) != shape[0] or not finite:
        raise RuntimeError(("invalid_visual_asset", dataset_id, shape, len(rows), finite))
    sha = file_hash(binary)
    info = {
        "schema": schema, "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE, "campaign": CAMPAIGN, "id": dataset_id, "title": title,
        "model": model, "binary_url": "/vis_data/research_kernel/" + binary.name,
        "binary_shape": shape, "binary_sha256": sha, "coordinate_count": shape[-1],
        "coordinate_semantics": "model-local runtime activation coordinates, not model weight parameters",
        "coordinate_order": "original physical activation coordinate order; no Top-K, PCA, or reordering",
        "claim_level": claim_level, "rows": rows, "boundary": boundary,
    }
    target = VIS / f"{dataset_id}.json"
    save_json(target, info)
    return {"id": dataset_id, "title": title, "metadata": target, "binary": binary,
            "shape": shape, "sha256": sha, "rows": len(rows), "model": model,
            "schema": schema, "claim_level": claim_level, "boundary": boundary}


def publish_q4_role_trajectory() -> dict:
    index = read_jsonl(Q4_ROLE_INDEX)
    chosen = [row for row in index if int(row["unit"]) == REPRESENTATIVE_UNIT]
    source = np.load(Q4_ROLE, mmap_mode="r")
    binary = VIS / "c3021_qwen4b_natural_role_trajectory.float16.npy"
    binary.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(chosen) * source.shape[1] * len(ROLES)
    output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float16,
                                       shape=(row_count, source.shape[-1]))
    rows, cursor = [], 0
    try:
        for case in chosen:
            for q in range(source.shape[1]):
                for role_i, role in enumerate(ROLES):
                    output[cursor] = source[int(case["hidden_index"]), q, role_i]
                    rows.append({"row": cursor, "case_id": case["case_id"], "family": case["family"],
                                 "language": case["language"], "surface": case["surface"],
                                 "state": case["state"], "unit": case["unit"], "role": role,
                                 "checkpoint": q, "checkpoint_label": checkpoint_label(q, 36),
                                 "metric": "physical_activation"})
                    cursor += 1
        output.flush()
    finally:
        close(output); close(source)
    return metadata(
        "c3021_qwen4b_natural_role_trajectory", "Qwen3-4B Natural Role Trajectory",
        binary, rows, "ai2050.natural-role-trajectory.v1", "Qwen3-4B",
        "full_coordinate_observation",
        "Unit26 for every behavior-qualified family, language, surface and state; embedding, all 36 post-block states, final norm, six roles and all 2560 coordinates.")


def publish_q4_all_token_trajectory() -> dict:
    index = [row for row in read_jsonl(Q4_TOKEN_INDEX)
             if row["family"] == TOKEN_FAMILY and row["surface"] == "narrative"]
    source = np.load(Q4_TOKEN, mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(str(Q4_MODEL), local_files_only=True,
                                               trust_remote_code=True, use_fast=False)
    row_count = sum(int(row["prompt_length"]) for row in index) * source.shape[1]
    binary = VIS / "c3022_qwen4b_attitude_all_token_trajectory.float16.npy"
    output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float16,
                                       shape=(row_count, source.shape[-1]))
    rows, cursor = [], 0
    try:
        for case in index:
            for q in range(source.shape[1]):
                for token_position, token_id in enumerate(case["prompt_ids"]):
                    output[cursor] = source[int(case["hidden_index"]), q, token_position]
                    rows.append({"row": cursor, "case_id": case["case_id"], "family": case["family"],
                                 "language": case["language"], "surface": case["surface"],
                                 "state": case["state"], "unit": case["unit"],
                                 "checkpoint": q, "checkpoint_label": checkpoint_label(q, 36),
                                 "token_position": token_position, "token_id": int(token_id),
                                 "token_text": tokenizer.decode([int(token_id)]),
                                 "metric": "physical_activation"})
                    cursor += 1
        output.flush()
    finally:
        close(output); close(source)
    return metadata(
        "c3022_qwen4b_attitude_all_token_trajectory", "Qwen3-4B Attitude All-Token Trajectory",
        binary, rows, "ai2050.natural-all-token-trajectory.v1", "Qwen3-4B",
        "full_coordinate_observation",
        "Attitude-event unit26 narrative cases in both languages and states; every real prompt token, embedding, all post-block states, final norm and every coordinate.")


def copy_atlas(source: Path, rows_path: Path, dataset_id: str, title: str,
               schema: str, model: str, claim_level: str, boundary: str) -> dict:
    binary = VIS / f"{dataset_id}.{source.name.split('.')[-2]}.npy"
    shutil.copy2(source, binary)
    return metadata(dataset_id, title, binary, read_jsonl(rows_path), schema,
                    model, claim_level, boundary)


def publish_q14_role_trajectory(final: dict) -> dict | None:
    if not Q14_ROLE.exists():
        return None
    index = [row for row in read_jsonl(Q14_ROLE_INDEX) if int(row["unit"]) == REPRESENTATIVE_UNIT]
    source = np.load(Q14_ROLE, mmap_mode="r")
    qpoints = list(final["qpoints"])
    row_count = len(index) * len(qpoints) * len(ROLES)
    binary = VIS / "c3025_qwen14_frozen_role_trajectory.float16.npy"
    output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float16,
                                       shape=(row_count, source.shape[-1]))
    rows, cursor = [], 0
    try:
        for case in index:
            for q_i, q in enumerate(qpoints):
                for role_i, role in enumerate(ROLES):
                    output[cursor] = source[int(case["hidden_index"]), q_i, role_i]
                    rows.append({"row": cursor, "case_id": case["case_id"], "family": case["family"],
                                 "language": case["language"], "surface": case["surface"],
                                 "state": case["state"], "unit": case["unit"], "role": role,
                                 "checkpoint": q, "checkpoint_label": checkpoint_label(q, 40, False),
                                 "metric": "physical_activation"})
                    cursor += 1
        output.flush()
    finally:
        close(output); close(source)
    return metadata(
        "c3025_qwen14_frozen_role_trajectory", "Qwen3-14B Frozen Role Trajectory",
        binary, rows, "ai2050.qwen14-frozen-role-trajectory.v1", "Qwen3-14B",
        "full_coordinate_observation",
        "Unit26 behavior-qualified cases at only the preregistered q0/q1/q5/q6 checkpoints; all six roles and all 5120 model-local coordinates. Coordinate IDs are not aligned with 4B.")


def verify(dataset: dict) -> dict:
    info = load_json(dataset["metadata"])
    value = np.load(dataset["binary"], mmap_mode="r")
    try:
        checks = {"shape": list(value.shape) == info["binary_shape"] == dataset["shape"],
                  "rows": len(info["rows"]) == value.shape[0],
                  "coordinates": value.shape[-1] == info["coordinate_count"],
                  "finite": bool(np.isfinite(value).all())}
    finally:
        close(value)
    checks["sha256"] = file_hash(dataset["binary"]) == info["binary_sha256"]
    return {"id": dataset["id"], **checks}


def published_dataset(dataset_id: str) -> dict:
    metadata_path = VIS / f"{dataset_id}.json"
    info = load_json(metadata_path)
    binary = VIS / Path(info["binary_url"]).name
    return {"id": dataset_id, "title": info["title"], "metadata": metadata_path,
            "binary": binary, "shape": info["binary_shape"],
            "sha256": info["binary_sha256"], "rows": len(info["rows"]),
            "model": info["model"], "schema": info["schema"],
            "claim_level": info["claim_level"], "boundary": info["boundary"]}


def serializable_dataset(dataset: dict) -> dict:
    return {**dataset,
            "metadata": str(Path(dataset["metadata"]).relative_to(ROOT)),
            "binary": str(Path(dataset["binary"]).relative_to(ROOT))}


def update_catalog(datasets: list[dict]) -> dict:
    catalog = load_json(CATALOG)
    ids = {row["id"] for row in datasets}
    entries = [{"id": row["id"], "title": row["title"], "phase": PHASE,
                "campaign": CAMPAIGN, "model": row["model"],
                "source_path": "/vis_data/research_kernel/" + row["metadata"].name,
                "binary_path": "/vis_data/research_kernel/" + row["binary"].name,
                "source_schema": row["schema"], "coordinate_count": row["shape"][-1],
                "row_count": row["shape"][0], "claim_level": row["claim_level"],
                "boundary": row["boundary"], "kinds": ["embedding_hiddenstate_full_coordinate"]}
               for row in datasets]
    catalog["datasets"] = entries + [row for row in catalog.get("datasets", [])
                                      if row.get("id") not in ids]
    fields = [{"id": row["id"], "title": row["title"],
               "url": "/vis_data/research_kernel/" + row["metadata"].name,
               "phase": PHASE, "full_coordinate": True,
               "heatmap_type": "embedding_hiddenstate_full_coordinate"}
              for row in datasets]
    catalog["field_datasets"] = [row for row in catalog.get("field_datasets", [])
                                 if row.get("id") not in ids] + fields
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save_json(CATALOG, catalog)
    return {"added": sorted(ids), "dataset_count": len(catalog["datasets"]),
            "field_dataset_count": len(catalog["field_datasets"])}


def frontend_build() -> dict:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm:
        command = [npm, "run", "build"]
    else:
        candidates = sorted(Path.home().glob(
            "AppData/Local/OpenAI/Codex/runtimes/cua_node/*/bin/node.exe"), reverse=True)
        if not candidates:
            raise FileNotFoundError("No npm or local Node runtime")
        command = [str(candidates[0]), str(ROOT / "frontend/node_modules/vite/bin/vite.js"), "build"]
    completed = subprocess.run(command, cwd=ROOT / "frontend", capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=900)
    return {"command": command, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:],
            "passed": completed.returncode == 0}


def cleanup(paths: list[Path]) -> dict:
    root = RESULT.resolve()
    rows, total = [], 0
    for path in paths:
        resolved = path.resolve()
        if root not in resolved.parents:
            raise RuntimeError(("cleanup_outside_result", str(resolved)))
        if not path.exists():
            rows.append({"path": str(path.relative_to(ROOT)), "status": "already_absent",
                         "bytes_deleted": 0})
            continue
        size, sha = path.stat().st_size, file_hash(path)
        path.unlink()
        total += size
        rows.append({"path": str(path.relative_to(ROOT)),
                     "status": "deleted_after_verified_visual_derivative",
                     "sha256_before": sha, "bytes_deleted": size})
    ledger = {"files": rows, "bytes_deleted": total}
    save_json(OUT / "cleanup/ledger.json", ledger)
    return ledger


def append_memo(result: dict) -> None:
    current = MEMO.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in current:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    datasets = [{key: row[key] for key in ("id", "shape", "sha256", "claim_level")}
                for row in result["datasets"]]
    text = rf"""

## Phase {PHASE}: 自然语言全坐标图谱发布、清理与大阶段裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本期不重新搜索机制，只把 Phase2290-2294 已揭盲对象转换为可复查图谱。4B面板包含：unit26全部合格构式的六角色全层轨迹；态度事件中英、真假状态的全部真实token全层轨迹；样本条件预测的候选及控制逐坐标误差；所有相邻检查点的候选及控制逐坐标传动误差。14B面板包含冻结检查点六角色轨迹和五个跨尺度单元的逐坐标误差。所有二进制行保留原物理激活坐标顺序；没有PCA、Top-K、余弦筛选、坐标重排或跨模型坐标对齐。

**公式。** 观察图逐行保存：

$$
V_{{m,j}}=H_{{i,q,t,j}},\qquad j=1,\ldots,d.
$$

预测图逐行保存候选和每个控制的坐标误差：

$$
E_{{m,j}}=\mathbb E_i\left|\widehat R_{{i,j}}-R_{{i,j}}\right|.
$$

**结果汇总。** 发布资产 `{json.dumps(datasets, ensure_ascii=False)}`；逐项验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；目录 `{json.dumps(result['catalog'], ensure_ascii=False)}`；构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。本大阶段最可靠的拼图是：自然双语材料中存在模型本地、样本基态条件化的逐坐标预测单元；相邻层还存在更广泛的家族特异预测传动，但大量可复验状态函数集中在embedding/浅层。14B结果只能支持或淘汰功能拓扑的跨尺度迁移，不能把相同坐标编号命名为共享神经元。图谱是运行时激活，不是模型权重参数；角色末token、人工自然度、受控任务、小模型粗糙性和逐坐标独立函数仍是硬伤。现有基础代数足以记录这些规律；目前没有证据证明需要新数学，也没有证据证明基础对象已经足够闭合。

**结论与后续边界。** 本 Campaign 的具体目标是“独立自然材料上的样本条件逐坐标规律及其14B跨尺度复验”，发布与清理后已经完成。下一阶段若继续扩展到新的语言族、全token坐标联动图或非仿射基础规律，研究对象与材料将改变，必须另立前瞻合同，不能沿本期锁箱自动续跑。脚本 `tests/glm5/phase2295_c3021_c3100_visual_atlas_cleanup.py`；结果 `tests/glm5/result/phase2295_c3021_c3100_visual_atlas_cleanup`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = load_json(final_path)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return
    cleanup_ledger = OUT / "cleanup/ledger.json"
    if cleanup_ledger.exists() and not Q4_ROLE.exists() and not Q4_TOKEN.exists() and not Q14_ROLE.exists():
        ids = [
            "c3021_qwen4b_natural_role_trajectory",
            "c3022_qwen4b_attitude_all_token_trajectory",
            "c3023_qwen4b_sample_conditioned_error_atlas",
            "c3024_qwen4b_layer_transport_error_atlas",
            "c3025_qwen14_frozen_role_trajectory",
            "c3026_qwen14_crossscale_error_atlas",
        ]
        datasets = [published_dataset(dataset_id) for dataset_id in ids]
        verification = [verify(row) for row in datasets]
        if not all(all(value for key, value in row.items() if key != "id") for row in verification):
            raise RuntimeError(("recovery_visual_verification_failed", verification))
        catalog_value = load_json(CATALOG)
        catalog = {"added": ids, "dataset_count": len(catalog_value.get("datasets", [])),
                   "field_dataset_count": len(catalog_value.get("field_datasets", [])),
                   "recovered_after_final_json_serialization_failure": True}
        build = frontend_build()
        if not build["passed"]:
            raise RuntimeError(("recovery_frontend_build_failed", build))
        cleanup_result = load_json(cleanup_ledger)
        checks = {"parents_passed": True, "datasets_verified": True,
                  "catalog_updated": all(dataset_id in {
                      row.get("id") for row in catalog_value.get("datasets", [])} for dataset_id in ids),
                  "frontend_built": build["passed"], "raw_fields_cleaned": True,
                  "only_exact_coordinates": all(row["shape"][-1] in (2560, 5120) for row in datasets),
                  "recovered_without_raw_recomputation": True}
        result = {"phase": PHASE, "campaign": CAMPAIGN,
                  "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
                  "datasets": [serializable_dataset(row) for row in datasets],
                  "verification": verification, "catalog": catalog,
                  "frontend_build": build, "cleanup": cleanup_result, "checks": checks,
                  "all_checks_passed": all(checks.values()),
                  "strict_conclusion": ("The natural sample-conditioned campaign is published as exact-coordinate "
                                        "observation/prediction assets with explicit claim levels; verified visual "
                                        "derivatives replace the undisplayed large raw fields, without claiming a "
                                        "shared neuron dictionary or causal semantic gear."),
                  "next_stage_same_specific_target": False,
                  "next_stage_reason": "New language families or coordinate-coupling rules require new materials and a new preregistration."}
        save_json(final_path, result)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return
    parents = [load_json(path / "analysis/final.json") for path in
               (Q4_OUT, PREDICT_OUT, TRANSPORT_OUT, Q14_OUT)]
    if not all(row["all_checks_passed"] for row in parents):
        raise RuntimeError("A source phase is not authorized")
    VIS.mkdir(parents=True, exist_ok=True)
    q14_final = parents[-1]
    datasets = [
        publish_q4_role_trajectory(),
        publish_q4_all_token_trajectory(),
        copy_atlas(PREDICT_ATLAS, PREDICT_ROWS,
                   "c3023_qwen4b_sample_conditioned_error_atlas",
                   "Qwen3-4B Sample-Conditioned Coordinate Errors",
                   "ai2050.sample-conditioned-errors.v1", "Qwen3-4B",
                   "prospective_prediction",
                   "Every selected family-route cell retains candidate and frozen-control MAE for all 2560 coordinates."),
        copy_atlas(TRANSPORT_ATLAS, TRANSPORT_ROWS,
                   "c3024_qwen4b_layer_transport_error_atlas",
                   "Qwen3-4B Layer Transport Coordinate Errors",
                   "ai2050.layer-transport-errors.v1", "Qwen3-4B",
                   "prospective_prediction",
                   "All family, adjacent-checkpoint and role cells retain candidate and control MAE in original coordinate order."),
    ]
    q14_role = publish_q14_role_trajectory(q14_final)
    if q14_role:
        datasets.append(q14_role)
    if Q14_ATLAS.exists() and np.load(Q14_ATLAS, mmap_mode="r").shape[0] > 0:
        datasets.append(copy_atlas(
            Q14_ATLAS, Q14_ROWS, "c3026_qwen14_crossscale_error_atlas",
            "Qwen3-14B Frozen Cross-Scale Coordinate Errors",
            "ai2050.qwen14-crossscale-errors.v1", "Qwen3-14B",
            "prospective_prediction",
            "Five frozen functional cells, ordered reveal outcomes and every 14B model-local coordinate; no 4B coordinate identity is claimed."))
    verification = [verify(row) for row in datasets]
    if not all(all(value for key, value in row.items() if key != "id") for row in verification):
        raise RuntimeError(("visual_verification_failed", verification))
    catalog = update_catalog(datasets)
    build = frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    cleanup_paths = [Q4_ROLE, Q4_TOKEN]
    if Q14_ROLE.exists():
        cleanup_paths.append(Q14_ROLE)
    cleanup_result = cleanup(cleanup_paths)
    checks = {"parents_passed": True, "datasets_verified": True,
              "catalog_updated": set(catalog["added"]) == {row["id"] for row in datasets},
              "frontend_built": build["passed"],
              "raw_fields_cleaned": all(not path.exists() for path in cleanup_paths),
              "only_exact_coordinates": all(row["shape"][-1] in (2560, 5120) for row in datasets)}
    result = {"phase": PHASE, "campaign": CAMPAIGN,
              "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
              "datasets": [serializable_dataset(row) for row in datasets],
              "verification": verification, "catalog": catalog,
              "frontend_build": build, "cleanup": cleanup_result, "checks": checks,
              "all_checks_passed": all(checks.values()),
              "strict_conclusion": ("The natural sample-conditioned campaign is published as exact-coordinate "
                                    "observation/prediction assets with explicit claim levels; verified visual "
                                    "derivatives replace the undisplayed large raw fields, without claiming a "
                                    "shared neuron dictionary or causal semantic gear."),
              "next_stage_same_specific_target": False,
              "next_stage_reason": "New language families or coordinate-coupling rules require new materials and a new preregistration."}
    save_json(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
