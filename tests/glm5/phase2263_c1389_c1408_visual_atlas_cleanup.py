#!/usr/bin/env python3
"""Publish exact-coordinate Phase2259-2262 fields and clean verified raw copies."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
Q4_FIELD_OUT = RESULT / "phase2259_c1265_c1296_qwen_natural_full_token_field"
Q4_OPERATOR_OUT = RESULT / "phase2260_c1297_c1336_coordinate_local_operator_tournament"
Q4_CAUSAL_OUT = RESULT / "phase2261_c1337_c1368_sample_conditioned_causal_adjudication"
Q14_OUT = RESULT / "phase2262_c1369_c1388_qwen14_coordinate_operator_replication"
OUT = RESULT / "phase2263_c1389_c1408_visual_atlas_cleanup"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

PHASE = 2263
CAMPAIGN = "C1389-C1408"
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")

Q4_TOKEN = Q4_FIELD_OUT / "raw/qwen3_4b_anchor_all_token_field.float16.npy"
Q4_TOKEN_INDEX = Q4_FIELD_OUT / "raw/all_token_field_index.jsonl"
Q4_GENERATION = Q4_FIELD_OUT / "raw/qwen3_4b_generation_boundary_field.float16.npy"
Q4_GENERATION_INDEX = Q4_FIELD_OUT / "raw/generation_boundary_index.jsonl"
Q4_ROLE = Q4_FIELD_OUT / "raw/qwen3_4b_qualified_role_field.float16.npy"
Q4_GAIN = Q4_OPERATOR_OUT / "atlas/full_coordinate_model_gain.float16.npy"
Q4_WIN = Q4_OPERATOR_OUT / "atlas/full_coordinate_role_win.uint8.npy"
Q14_FIELD = Q14_OUT / "raw/qwen3_14b_selected_checkpoint_field.float16.npy"
Q14_INDEX = Q14_OUT / "raw/field_index.jsonl"


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


def checkpoint_label(checkpoint: int, total: int = 36) -> str:
    if checkpoint == 0:
        return "embedding"
    if checkpoint == total + 1:
        return "final_norm"
    return f"block_{checkpoint:02d}_post"


def write_flat_float16(source: Path, target: Path) -> tuple[list[int], str]:
    value = np.load(source, mmap_mode="r")
    try:
        shape = list(value.shape)
        rows, coordinates = int(np.prod(shape[:-1])), shape[-1]
        target.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16,
                                           shape=(rows, coordinates))
        flat = value.reshape(rows, coordinates)
        for start in range(0, rows, 256):
            output[start:start + 256] = np.asarray(flat[start:start + 256], np.float16)
        output.flush()
        mmap = getattr(output, "_mmap", None)
        if mmap is not None:
            mmap.close()
    finally:
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()
    return [rows, coordinates], file_hash(target)


def publish_gain_atlas(q4: dict) -> dict:
    family_order = q4["analysis"]["family_order"]
    partition_order = q4["analysis"]["partition_order"]
    model_order = q4["analysis"]["model_order"]
    binary = VIS / "c1392_qwen4b_coordinate_operator_gain.float16.npy"
    shape, sha = write_flat_float16(Q4_GAIN, binary)
    rows = []
    for family in family_order:
        for partition in partition_order:
            for model in model_order[1:]:
                for checkpoint in range(38):
                    for role in ROLES:
                        rows.append({"family": family, "partition": partition, "source": model,
                                     "checkpoint": checkpoint,
                                     "checkpoint_label": checkpoint_label(checkpoint),
                                     "role": role, "metric": "coordinate_gain_over_family_mean"})
    assert len(rows) == shape[0]
    metadata = {
        "schema": "ai2050.coordinate-local-operator-atlas.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE, "campaign": CAMPAIGN,
        "title": "Qwen3-4B Coordinate-Local Prediction Gain",
        "binary_url": "/vis_data/research_kernel/c1392_qwen4b_coordinate_operator_gain.float16.npy",
        "binary_shape": shape, "binary_sha256": sha, "coordinate_count": 2560,
        "coordinate_semantics": "model-local physical activation coordinates; not weights",
        "rows": rows,
        "boundary": "Fresh-lockbox predictive gain over a family-mean response. Association, not causal necessity.",
    }
    target = VIS / "c1392_qwen4b_coordinate_operator_gain.json"
    save_json(target, metadata)
    return {"id": "c1392_qwen4b_coordinate_operator_gain", "metadata": target,
            "binary": binary, "shape": shape, "sha256": sha, "rows": len(rows)}


def publish_win_atlas(q4: dict) -> dict:
    family_order = q4["analysis"]["family_order"]
    partition_order = q4["analysis"]["partition_order"]
    source = np.load(Q4_WIN, mmap_mode="r")
    binary = VIS / "c1393_qwen4b_coordinate_operator_win.float16.npy"
    try:
        flat = source.reshape(-1, source.shape[-1])
        output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float16, shape=flat.shape)
        for start in range(0, flat.shape[0], 256):
            output[start:start + 256] = np.asarray(flat[start:start + 256], np.float16)
        output.flush()
        mmap = getattr(output, "_mmap", None)
        if mmap is not None:
            mmap.close()
    finally:
        mmap = getattr(source, "_mmap", None)
        if mmap is not None:
            mmap.close()
    rows = []
    for family in family_order:
        for partition in partition_order:
            for checkpoint in range(38):
                for role in ROLES:
                    rows.append({"family": family, "partition": partition,
                                 "source": "same_coordinate_affine_win",
                                 "checkpoint": checkpoint,
                                 "checkpoint_label": checkpoint_label(checkpoint),
                                 "role": role, "metric": "coordinate_error_win"})
    shape = [len(rows), 2560]
    assert np.load(binary, mmap_mode="r").shape == tuple(shape)
    sha = file_hash(binary)
    metadata = {
        "schema": "ai2050.coordinate-local-operator-atlas.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE, "campaign": CAMPAIGN,
        "title": "Qwen3-4B Coordinate-Local Error-Win Mask",
        "binary_url": "/vis_data/research_kernel/c1393_qwen4b_coordinate_operator_win.float16.npy",
        "binary_shape": shape, "binary_sha256": sha, "coordinate_count": 2560,
        "coordinate_semantics": "1 means the coordinate-local predictor beat the family mean for that physical activation coordinate",
        "rows": rows,
        "boundary": "A coordinate-wise predictive comparison. It is not a selected Top-K mask or a neuron dictionary.",
    }
    target = VIS / "c1393_qwen4b_coordinate_operator_win.json"
    save_json(target, metadata)
    return {"id": "c1393_qwen4b_coordinate_operator_win", "metadata": target,
            "binary": binary, "shape": shape, "sha256": sha, "rows": len(rows)}


def publish_token_examples() -> dict:
    index = read_jsonl(Q4_TOKEN_INDEX)
    keys = {}
    for row in index:
        key = (row["family"], row["language"], row["surface"], row["state"])
        if key not in keys or row["unit"] < keys[key]["unit"]:
            keys[key] = row
    selected = sorted(keys.values(), key=lambda row: (row["family"], row["language"],
                                                       row["surface"], row["state"]))
    if len(selected) != 16:
        raise RuntimeError(("expected_16_token_examples", len(selected)))
    source = np.load(Q4_TOKEN, mmap_mode="r")
    binary = VIS / "c1394_qwen4b_all_token_examples.float16.npy"
    output_shape = (len(selected) * 38 * 72, 2560)
    output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float16, shape=output_shape)
    rows = []
    cursor = 0
    try:
        for row in selected:
            hidden_index = row["hidden_index"]
            for checkpoint in range(38):
                for token_position in range(72):
                    output[cursor] = source[hidden_index, checkpoint, token_position]
                    roles = [role for role, positions in row["role_positions"].items()
                             if token_position in positions]
                    rows.append({"case_id": row["case_id"], "family": row["family"],
                                 "language": row["language"], "surface": row["surface"],
                                 "state": row["state"], "unit": row["unit"],
                                 "checkpoint": checkpoint,
                                 "checkpoint_label": checkpoint_label(checkpoint),
                                 "token_position": token_position,
                                 "token_id": row["prompt_ids"][token_position]
                                 if token_position < row["prompt_length"] else None,
                                 "role": "+".join(roles) if roles else "untyped_token",
                                 "source": "token_activation" if token_position < row["prompt_length"] else "padding",
                                 "metric": "physical_activation"})
                    cursor += 1
        output.flush()
    finally:
        for value in (output, source):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
    sha = file_hash(binary)
    metadata = {
        "schema": "ai2050.all-token-physical-field.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE, "campaign": CAMPAIGN,
        "title": "Qwen3-4B Full Token-by-Checkpoint Examples",
        "binary_url": "/vis_data/research_kernel/c1394_qwen4b_all_token_examples.float16.npy",
        "binary_shape": list(output_shape), "binary_sha256": sha, "coordinate_count": 2560,
        "coordinate_semantics": "all model-local physical activation coordinates at each saved token and checkpoint",
        "rows": rows, "selected_cases": [row["case_id"] for row in selected],
        "boundary": "Sixteen frozen natural examples. Physical token positions are not treated as cross-surface semantic alignment.",
    }
    target = VIS / "c1394_qwen4b_all_token_examples.json"
    save_json(target, metadata)
    return {"id": "c1394_qwen4b_all_token_examples", "metadata": target,
            "binary": binary, "shape": list(output_shape), "sha256": sha, "rows": len(rows)}


def publish_generation_field() -> dict:
    index = read_jsonl(Q4_GENERATION_INDEX)
    binary = VIS / "c1395_qwen4b_generation_boundary.float16.npy"
    shape, sha = write_flat_float16(Q4_GENERATION, binary)
    rows = []
    for row in index:
        for step in range(4):
            for checkpoint in range(38):
                rows.append({"case_id": row["case_id"], "family": row["family"],
                             "language": row["language"], "surface": row["surface"],
                             "state": row["state"], "unit": row["unit"],
                             "generation_step": step,
                             "generated_token_id": row["generated_ids"][step]
                             if step < len(row["generated_ids"]) else None,
                             "checkpoint": checkpoint,
                             "checkpoint_label": checkpoint_label(checkpoint),
                             "role": "generation_boundary", "source": "generated_boundary_activation",
                             "metric": "physical_activation"})
    assert len(rows) == shape[0]
    metadata = {
        "schema": "ai2050.generation-boundary-field.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE, "campaign": CAMPAIGN,
        "title": "Qwen3-4B Generation Boundary Field",
        "binary_url": "/vis_data/research_kernel/c1395_qwen4b_generation_boundary.float16.npy",
        "binary_shape": shape, "binary_sha256": sha, "coordinate_count": 2560,
        "coordinate_semantics": "all model-local physical activation coordinates during four autoregressive steps",
        "rows": rows,
        "boundary": "Generated boundary states are downstream observations and cannot be used as pre-generation predictors.",
    }
    target = VIS / "c1395_qwen4b_generation_boundary.json"
    save_json(target, metadata)
    return {"id": "c1395_qwen4b_generation_boundary", "metadata": target,
            "binary": binary, "shape": shape, "sha256": sha, "rows": len(rows)}


def publish_q14_field(q14: dict) -> dict | None:
    if not q14["field"].get("ran") or not Q14_FIELD.exists():
        return None
    index = read_jsonl(Q14_INDEX)
    binary = VIS / "c1396_qwen14_coordinate_operator_replication.float16.npy"
    shape, sha = write_flat_float16(Q14_FIELD, binary)
    rows = [{**row, "checkpoint": row["q14_checkpoint"],
             "checkpoint_label": checkpoint_label(row["q14_checkpoint"], 40),
             "source": "selected_checkpoint_activation", "metric": "physical_activation"}
            for row in index]
    assert len(rows) == shape[0]
    metadata = {
        "schema": "ai2050.cross-scale-coordinate-local-replication.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE, "campaign": CAMPAIGN,
        "title": "Qwen3-14B Relative-Depth Coordinate-Local Replication",
        "binary_url": "/vis_data/research_kernel/c1396_qwen14_coordinate_operator_replication.float16.npy",
        "binary_shape": shape, "binary_sha256": sha, "coordinate_count": 5120,
        "coordinate_semantics": "Qwen3-14B model-local physical activation coordinates; not aligned to Qwen3-4B coordinate IDs",
        "rows": rows, "replication": q14["replication"],
        "boundary": "Relative-depth, role-matched cross-scale replication. Physical coordinate identities are model-local.",
    }
    target = VIS / "c1396_qwen14_coordinate_operator_replication.json"
    save_json(target, metadata)
    return {"id": "c1396_qwen14_coordinate_operator_replication", "metadata": target,
            "binary": binary, "shape": shape, "sha256": sha, "rows": len(rows)}


def catalog_entry(dataset: dict, title: str, model: str, claim: str, boundary: str) -> dict:
    return {
        "id": dataset["id"], "title": title, "phase": PHASE, "campaign": CAMPAIGN,
        "model": model,
        "source_path": "/vis_data/research_kernel/" + dataset["metadata"].name,
        "binary_path": "/vis_data/research_kernel/" + dataset["binary"].name,
        "source_schema": load_json(dataset["metadata"])["schema"],
        "coordinate_count": dataset["shape"][-1], "row_count": dataset["shape"][0],
        "claim_level": claim, "boundary": boundary,
        "kinds": ["embedding_and_hiddenstate_physical_activation_coordinates"],
    }


def update_catalog(datasets: list[dict]) -> dict:
    catalog = load_json(CATALOG)
    entries = [
        catalog_entry(datasets[0], "Qwen3-4B Coordinate-Local Prediction Gain", "Qwen3-4B",
                      "fresh_lockbox_predictive_association",
                      "Exact-coordinate prediction gain; Phase2261 found no strict causal family."),
        catalog_entry(datasets[1], "Qwen3-4B Coordinate-Local Error-Win Mask", "Qwen3-4B",
                      "fresh_lockbox_coordinate_comparison",
                      "Complete coordinate outcomes, not Top-K and not a neuron dictionary."),
        catalog_entry(datasets[2], "Qwen3-4B Full Token-by-Checkpoint Examples", "Qwen3-4B",
                      "full_token_observation",
                      "Physical token positions are not cross-surface semantic roles."),
        catalog_entry(datasets[3], "Qwen3-4B Generation Boundary Field", "Qwen3-4B",
                      "downstream_generation_observation",
                      "Downstream observations cannot be used as pre-generation predictors."),
    ]
    if len(datasets) == 5:
        entries.append(catalog_entry(
            datasets[4], "Qwen3-14B Relative-Depth Coordinate-Local Replication", "Qwen3-14B",
            "cross_scale_model_local_replication",
            "Qwen3-4B and Qwen3-14B physical coordinate IDs are not aligned."))
    ids = {row["id"] for row in entries}
    catalog["datasets"] = entries + [row for row in catalog.get("datasets", []) if row.get("id") not in ids]
    fields = [{"id": row["id"], "title": row["title"], "url": row["source_path"],
               "phase": PHASE, "full_coordinate": True,
               "heatmap_type": "embedding_hiddenstate_full_coordinate"} for row in entries]
    catalog["field_datasets"] = [row for row in catalog.get("field_datasets", [])
                                 if row.get("id") not in ids] + fields
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save_json(CATALOG, catalog)
    return {"entries_added": [row["id"] for row in entries],
            "dataset_count": len(catalog["datasets"]),
            "field_dataset_count": len(catalog["field_datasets"])}


def verify_dataset(dataset: dict) -> dict:
    metadata = load_json(dataset["metadata"])
    value = np.load(dataset["binary"], mmap_mode="r")
    try:
        shape_ok = list(value.shape) == metadata["binary_shape"] == dataset["shape"]
        finite = bool(np.isfinite(np.asarray(value[:min(32, value.shape[0])], np.float32)).all())
        row_ok = len(metadata["rows"]) == value.shape[0]
        coordinate_ok = value.shape[1] == metadata["coordinate_count"]
    finally:
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()
    return {"id": dataset["id"], "shape_ok": shape_ok, "finite_sample": finite,
            "row_metadata_ok": row_ok, "coordinate_count_ok": coordinate_ok,
            "hash_ok": file_hash(dataset["binary"]) == metadata["binary_sha256"]}


def frontend_build() -> dict:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm:
        command = [npm, "run", "build"]
    else:
        candidates = sorted(Path.home().glob(
            "AppData/Local/OpenAI/Codex/runtimes/cua_node/*/bin/node.exe"), reverse=True)
        if not candidates:
            raise FileNotFoundError("No npm or local Node runtime for the Vite build")
        command = [str(candidates[0]), str(ROOT / "frontend/node_modules/vite/bin/vite.js"), "build"]
    completed = subprocess.run(command, cwd=ROOT / "frontend", text=True, encoding="utf-8",
                               errors="replace", capture_output=True, timeout=300)
    return {"command": command, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:],
            "passed": completed.returncode == 0}


def cleanup(paths: Iterable[Path]) -> dict:
    records = []
    total = 0
    for path in paths:
        if not path.exists():
            records.append({"path": str(path.relative_to(ROOT)), "status": "already_absent",
                            "bytes_deleted": 0})
            continue
        sha = file_hash(path)
        size = path.stat().st_size
        path.unlink()
        total += size
        records.append({"path": str(path.relative_to(ROOT)),
                        "status": "deleted_after_visual_copy_and_hash_verification",
                        "sha256_before": sha, "bytes_deleted": size})
    ledger = {"files": records, "bytes_deleted": total}
    save_json(OUT / "cleanup/ledger.json", ledger)
    return ledger


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    q14 = result["q14_replication"]
    text = rf"""

## Phase {PHASE}: 全坐标规律图谱发布、跨尺度复验与原始场清理（C1389-C1408） [{stamp}]

**测试原理与测试用例。** 本阶段不再创造新机制假说，而把 Phase2259-2262 已冻结的观测链做成可逐坐标检查的研究图谱。发布五类对象：九个行为合格语言族在四分区、38检查点、六角色上的同坐标预测增益；每个物理坐标是否优于族均值的完整胜负表；属性与位置两个锚点族的16个自然句逐token逐检查点场；128个样本四步自由生成边界场；以及14B三个预冻结代表族的相对层深、同角色、全5120坐标复验场。典型用例包括“属性状态改变”“收件人绑定改变”“量词共享改变”；跨表面只保存各自场，不把相同文本位置冒充语义同构。

**数学对象。** 对每个模型本地物理坐标分别拟合，不混合坐标：

$$
R_{{i,j}}=H^{{(1)}}_{{i,j}}-H^{{(0)}}_{{i,j}},\qquad
\widehat R_{{i,j}}=a_j+b_jH^{{(0)}}_{{i,j}},\qquad
G_j=1-\frac{{\operatorname{{MAE}}_j(\widehat R,R)}}{{\operatorname{{MAE}}_j(\bar R,R)+\epsilon}}.
$$

跨尺度只冻结相对深度与角色：

$$
q_{{14}}=\operatorname{{round}}\!\left(\frac{{q_4}}{{36}}40\right),
$$

不比较4B与14B的坐标编号。

**结果汇总。** 可视化数据集与形状为 `{json.dumps(result['datasets'], ensure_ascii=False)}`；数据完整性复验为 `{json.dumps(result['verification'], ensure_ascii=False)}`；14B正式复验摘要为 `{json.dumps(q14, ensure_ascii=False)}`。Phase2260的九族同坐标模型均在各自fresh lockbox优于族均值，但Phase2261严格调用/删除、错族、错符号、错角色、错检查点和自由生成合取门没有任何族通过。因此热力图显示的是“基态条件预测规律”，不是“已找到因果齿轮”。

**理论进展与严格边界。** 目前最可靠的拼图是：语言状态变化并非只有族平均响应，当前样本同一物理坐标的基态值对晚层响应含有大量可迁移信息；这种规律可在多族出现，并接受14B代表族的独立判决。它不推出固定语义坐标、跨模型同坐标、注意力/MLP电路、单神经元字典或完整语言数学结构。晚层boundary占多数也说明其中相当部分可能是答案准备，而非上游关系形成。

**问题、硬伤与结论。** 材料仍是受控自然句且人类盲评为NA；全token面板只显示16个代表样本；14B只复验三族；预测模型是一阶同坐标关系；严格因果结果为阴性。客户端构建结果为 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。在可视化副本完成形状、有限值、行元数据和哈希复验后，清理账本为 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。相关脚本：`tests/glm5/phase2263_c1389_c1408_visual_atlas_cleanup.py`；结果：`tests/glm5/result/phase2263_c1389_c1408_visual_atlas_cleanup`；图谱：`frontend/public/vis_data/research_kernel/c1392_*` 至 `c1396_*`。下一大阶段应在全新材料上前瞻冻结上游检查点和token角色，区分关系形成轨迹与末层答案准备，再进行坐标级必要性/充分性裁决；不得从本阶段热力图事后挑坐标宣称机制闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load_json(final)
        append_memo(result)
        return result
    q4 = load_json(Q4_OPERATOR_OUT / "analysis/final.json")
    causal = load_json(Q4_CAUSAL_OUT / "analysis/final.json")
    q14 = load_json(Q14_OUT / "analysis/final.json")
    VIS.mkdir(parents=True, exist_ok=True)
    datasets = [publish_gain_atlas(q4), publish_win_atlas(q4),
                publish_token_examples(), publish_generation_field()]
    q14_dataset = publish_q14_field(q14)
    if q14_dataset is not None:
        datasets.append(q14_dataset)
    catalog = update_catalog(datasets)
    verification = [verify_dataset(dataset) for dataset in datasets]
    build = frontend_build()
    checks = {
        "phase2260_nine_predictive_families": len(q4["analysis"]["selected"]) == 9,
        "phase2261_strict_causal_empty": causal["strict_causal_families"] == [],
        "all_visual_datasets_verified": all(all(value for key, value in row.items() if key != "id")
                                                for row in verification),
        "catalog_entries_complete": len(catalog["entries_added"]) == len(datasets),
        "frontend_build_passed": build["passed"],
    }
    if not all(checks.values()):
        raise RuntimeError(("publication_verification_failed", checks))
    raw_to_clean = [Q4_TOKEN, Q4_GENERATION, Q4_ROLE, Q4_GAIN, Q4_WIN]
    if q14_dataset is not None:
        raw_to_clean.append(Q14_FIELD)
    clean = cleanup(raw_to_clean)
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "datasets": [{"id": row["id"], "shape": row["shape"],
                      "sha256": row["sha256"], "rows": row["rows"]} for row in datasets],
        "verification": verification, "catalog": catalog, "frontend_build": build,
        "q14_replication": {"behavior": q14["behavior"],
                            "replication": q14["replication"],
                            "replicated_families": q14["replicated_families"]},
        "strict_causal_families": causal["strict_causal_families"],
        "cleanup": clean, "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Coordinate-local base-state prediction is observable and lockbox-predictive, but strict sample-conditioned causality remains unconfirmed.",
    }
    save_json(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
