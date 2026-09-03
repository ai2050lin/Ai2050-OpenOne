#!/usr/bin/env python3
"""Publish Phase2265-2269 exact-coordinate atlases and clean verified raw fields."""
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
Q4_FIELD_OUT = RESULT / "phase2266_c1469_c1504_qwen4b_independent_fullfield"
Q4_MODEL_OUT = RESULT / "phase2267_c1505_c1540_coordinate_model_tournament"
Q4_CAUSAL_OUT = RESULT / "phase2268_c1541_c1576_near_manifold_causal_adjudication"
Q14_OUT = RESULT / "phase2269_c1577_c1600_qwen14_relative_topology_replication"
OUT = RESULT / "phase2270_c1601_c1624_visual_atlas_cleanup"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

PHASE = 2270
CAMPAIGN = "C1601-C1624"
FLAGSHIP_FAMILIES = ("location_state", "property_state", "patient_binding")
LANGUAGES = ("en", "zh")
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")

Q4_TOKEN = Q4_FIELD_OUT / "raw/qwen3_4b_stratified_all_token_field.float16.npy"
Q4_TOKEN_INDEX = Q4_FIELD_OUT / "raw/all_token_field_index.jsonl"
Q4_ROLE = Q4_FIELD_OUT / "raw/qwen3_4b_qualified_role_field.float16.npy"
Q4_ATLAS = Q4_MODEL_OUT / "atlas/qwen4b_coordinate_model_errors.float16.npy"
Q4_ATLAS_ROWS = Q4_MODEL_OUT / "atlas/qwen4b_coordinate_model_errors.rows.jsonl"
Q14_FIELD = Q14_OUT / "raw/qwen3_14b_relative_window_field.float16.npy"
Q14_ATLAS = Q14_OUT / "atlas/qwen14_relative_topology_coordinates.float16.npy"
Q14_ATLAS_ROWS = Q14_OUT / "atlas/qwen14_relative_topology_coordinates.rows.jsonl"


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


def checkpoint_label(q: int, layers: int) -> str:
    if q == 0:
        return "embedding"
    if q == layers + 1:
        return "final_norm"
    return f"block_{q:02d}_post"


def copy_npy(source: Path, target: Path) -> tuple[list[int], str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    value = np.load(target, mmap_mode="r")
    try:
        shape = list(value.shape)
    finally:
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()
    return shape, file_hash(target)


def publish_flagship_token_fields() -> dict:
    index = read_jsonl(Q4_TOKEN_INDEX)
    chosen = []
    for family in FLAGSHIP_FAMILIES:
        for language in LANGUAGES:
            for state in (0, 1):
                matches = [row for row in index if row["family"] == family
                           and row["language"] == language and int(row["unit"]) == 16
                           and row["surface"] == "direct" and int(row["state"]) == state]
                if len(matches) != 1:
                    raise RuntimeError(("flagship_case_not_unique", family, language, state, len(matches)))
                chosen.append(matches[0])
    source = np.load(Q4_TOKEN, mmap_mode="r")
    if source.shape[1:] != (38, 74, 2560):
        raise RuntimeError(("unexpected_q4_token_shape", source.shape))
    binary = VIS / "c1601_qwen4b_clean_bilingual_flagship_token_field.float16.npy"
    output_shape = (len(chosen) * 38 * 74, 2560)
    output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float16, shape=output_shape)
    rows = []
    cursor = 0
    try:
        for case in chosen:
            hidden_index = int(case["hidden_index"])
            for q in range(38):
                for token_position in range(74):
                    output[cursor] = source[hidden_index, q, token_position]
                    role_hits = [role for role, positions in case["role_positions"].items()
                                 if token_position in positions]
                    rows.append({
                        "row": cursor, "case_id": case["case_id"], "family": case["family"],
                        "language": case["language"], "unit": case["unit"],
                        "surface": case["surface"], "state": case["state"],
                        "checkpoint": q, "checkpoint_label": checkpoint_label(q, 36),
                        "token_position": token_position,
                        "token_id": case["prompt_ids"][token_position]
                        if token_position < case["prompt_length"] else None,
                        "role": "+".join(role_hits) if role_hits else "untyped_token",
                        "metric": "physical_activation" if token_position < case["prompt_length"] else "padding",
                    })
                    cursor += 1
        output.flush()
    finally:
        for value in (output, source):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
    sha = file_hash(binary)
    metadata = {
        "schema": "ai2050.clean-bilingual-flagship-token-field.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE, "campaign": CAMPAIGN,
        "title": "Qwen3-4B Clean Bilingual Flagship Token Fields",
        "binary_url": "/vis_data/research_kernel/" + binary.name,
        "binary_shape": list(output_shape), "binary_sha256": sha,
        "coordinate_count": 2560, "checkpoint_count": 38, "token_slots": 74,
        "coordinate_semantics": "Qwen3-4B model-local physical activations; q0 is embedding, q1-q36 are post-block states, q37 is final norm",
        "selected_cases": [row["case_id"] for row in chosen], "rows": rows,
        "boundary": "Twelve frozen clean UTF-8 cases retain every coordinate. Token positions are case-local and are not cross-surface semantic alignment.",
    }
    target = VIS / "c1601_qwen4b_clean_bilingual_flagship_token_field.json"
    save_json(target, metadata)
    return {"id": "c1601_qwen4b_clean_bilingual_flagship_token_field", "metadata": target,
            "binary": binary, "shape": list(output_shape), "sha256": sha, "rows": len(rows)}


def publish_q4_prediction_atlas() -> dict:
    rows = read_jsonl(Q4_ATLAS_ROWS)
    binary = VIS / "c1602_qwen4b_clean_coordinate_prediction_errors.float16.npy"
    shape, sha = copy_npy(Q4_ATLAS, binary)
    if shape != [70, 2560] or len(rows) != shape[0]:
        raise RuntimeError(("q4_atlas_contract", shape, len(rows)))
    metadata = {
        "schema": "ai2050.clean-coordinate-prediction-errors.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE, "campaign": CAMPAIGN,
        "title": "Qwen3-4B Clean-Material Coordinate Prediction Errors",
        "binary_url": "/vis_data/research_kernel/" + binary.name,
        "binary_shape": shape, "binary_sha256": sha, "coordinate_count": 2560,
        "rows": rows,
        "coordinate_semantics": "Every column is one model-local physical activation coordinate",
        "boundary": "Fresh-lockbox coordinate prediction diagnostics. Predictive association is not causal necessity or a semantic-neuron dictionary.",
    }
    target = VIS / "c1602_qwen4b_clean_coordinate_prediction_errors.json"
    save_json(target, metadata)
    return {"id": "c1602_qwen4b_clean_coordinate_prediction_errors", "metadata": target,
            "binary": binary, "shape": shape, "sha256": sha, "rows": len(rows)}


def publish_q14_replication_atlas() -> dict:
    q14 = load_json(Q14_OUT / "analysis/final.json")
    rows = read_jsonl(Q14_ATLAS_ROWS)
    binary = VIS / "c1603_qwen14_relative_topology_replication.float16.npy"
    shape, sha = copy_npy(Q14_ATLAS, binary)
    if shape[-1] != 5120 or len(rows) != shape[0]:
        raise RuntimeError(("q14_atlas_contract", shape, len(rows)))
    metadata = {
        "schema": "ai2050.relative-depth-coordinate-prediction-replication.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE, "campaign": CAMPAIGN,
        "title": "Qwen3-14B Relative-Depth Coordinate Prediction Replication",
        "binary_url": "/vis_data/research_kernel/" + binary.name,
        "binary_shape": shape, "binary_sha256": sha, "coordinate_count": 5120,
        "rows": rows, "replicated_families": q14["replicated_families"],
        "decisions": q14["decisions"],
        "coordinate_semantics": "Qwen3-14B model-local physical activation coordinates; coordinate IDs are not aligned to Qwen3-4B",
        "boundary": "Relative-depth and role-topology prediction replication only; no cross-model coordinate identity and no causal claim.",
    }
    target = VIS / "c1603_qwen14_relative_topology_replication.json"
    save_json(target, metadata)
    return {"id": "c1603_qwen14_relative_topology_replication", "metadata": target,
            "binary": binary, "shape": shape, "sha256": sha, "rows": len(rows)}


def catalog_entry(dataset: dict, title: str, model: str, claim: str, boundary: str) -> dict:
    metadata = load_json(dataset["metadata"])
    return {
        "id": dataset["id"], "title": title, "phase": PHASE, "campaign": CAMPAIGN,
        "model": model, "source_path": "/vis_data/research_kernel/" + dataset["metadata"].name,
        "binary_path": "/vis_data/research_kernel/" + dataset["binary"].name,
        "source_schema": metadata["schema"], "coordinate_count": dataset["shape"][-1],
        "row_count": dataset["shape"][0], "claim_level": claim, "boundary": boundary,
        "kinds": ["embedding_and_hiddenstate_physical_activation_coordinates"],
    }


def update_catalog(datasets: list[dict]) -> dict:
    entries = [
        catalog_entry(datasets[0], "Qwen3-4B Clean Bilingual Flagship Token Fields", "Qwen3-4B",
                      "full_token_clean_material_observation",
                      "All coordinates for frozen representative cases; token positions remain case-local."),
        catalog_entry(datasets[1], "Qwen3-4B Clean Coordinate Prediction Errors", "Qwen3-4B",
                      "fresh_lockbox_predictive_association",
                      "All-coordinate errors; Phase2268 found no strict bidirectional causal family."),
        catalog_entry(datasets[2], "Qwen3-14B Relative-Depth Prediction Replication", "Qwen3-14B",
                      "cross_scale_model_local_predictive_replication",
                      "Qwen3-4B and Qwen3-14B physical coordinate IDs are not aligned."),
    ]
    catalog = load_json(CATALOG)
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
        sample = np.asarray(value[:min(32, value.shape[0])], np.float32)
        checks = {
            "shape_ok": list(value.shape) == metadata["binary_shape"] == dataset["shape"],
            "row_metadata_ok": len(metadata["rows"]) == value.shape[0],
            "coordinate_count_ok": value.shape[-1] == metadata["coordinate_count"],
            "finite_sample": bool(np.isfinite(sample).all()),
        }
    finally:
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()
    checks["hash_ok"] = file_hash(dataset["binary"]) == metadata["binary_sha256"]
    return {"id": dataset["id"], **checks}


def frontend_build() -> dict:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm:
        command = [npm, "run", "build"]
    else:
        candidates = sorted(Path.home().glob(
            "AppData/Local/OpenAI/Codex/runtimes/cua_node/*/bin/node.exe"), reverse=True)
        if not candidates:
            raise FileNotFoundError("No npm or local Node runtime for Vite build")
        command = [str(candidates[0]), str(ROOT / "frontend/node_modules/vite/bin/vite.js"), "build"]
    completed = subprocess.run(command, cwd=ROOT / "frontend", text=True, encoding="utf-8",
                               errors="replace", capture_output=True, timeout=600)
    return {"command": command, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:],
            "passed": completed.returncode == 0}


def cleanup_raw_fields(paths: list[Path]) -> dict:
    records, total = [], 0
    result_root = RESULT.resolve()
    for path in paths:
        resolved = path.resolve()
        if result_root not in resolved.parents:
            raise RuntimeError(("cleanup_outside_result_root", str(resolved)))
        if not path.exists():
            records.append({"path": str(path.relative_to(ROOT)), "status": "already_absent", "bytes_deleted": 0})
            continue
        size, sha = path.stat().st_size, file_hash(path)
        path.unlink()
        total += size
        records.append({"path": str(path.relative_to(ROOT)),
                        "status": "deleted_after_visual_derivative_verification",
                        "sha256_before": sha, "bytes_deleted": size})
    ledger = {"files": records, "bytes_deleted": total}
    save_json(OUT / "cleanup/ledger.json", ledger)
    return ledger


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    q4 = result["q4_predictive"]
    q14 = result["q14_replication"]
    causal = result["q4_causal"]
    text = rf"""

## Phase {PHASE}: 清洁双语全坐标图谱发布与跨尺度总审计（C1601-C1624） [{stamp}]

**测试原理与用例。** 本阶段不从热力图事后挑选机制，而是发布 Phase2265-2269 已冻结、已揭盲的三类证据。第一，选取位置状态、属性状态、受事绑定三个代表家族，在英语和中文中各固定 unit16、direct 表面、state0/state1，共十二个样本；逐 token 保存 embedding、36 个 block 后状态、final norm 和全部 2560 个 Qwen3-4B 物理激活坐标。第二，发布十个行为合格家族的七类逐坐标误差，包括自家族同坐标仿射、家族均值、纯代数、共享、错配和错家族控制。第三，发布 Qwen3-14B 三个预冻结代表家族在相对层深窗口中的全部 5120 坐标复现图。坐标均指运行时激活，不是模型权重参数；跨模型不对齐坐标编号。

**数学对象。** 对每个家族、检查点、角色和物理坐标分别估计：

$$
R_{{i,j}}=H^{{(1)}}_{{i,j}}-H^{{(0)}}_{{i,j}},\qquad
\widehat R_{{i,j}}=a_{{f,j}}H^{{(0)}}_{{i,j}}+b_{{f,j}}.
$$

控制增益和逐坐标胜率为：

$$
G_c=\frac{{\operatorname{{MAE}}(c)-\operatorname{{MAE}}(\widehat R)}}
{{\max(\operatorname{{MAE}}(c),\epsilon)}},\qquad
W_c=\frac1d\sum_{{j=1}}^d\mathbf 1[e_j(\widehat R)<e_j(c)].
$$

14B 仅冻结相对层深邻域：

$$
q_{{14}}=\operatorname{{round}}\!\left(\frac{{q_4}}{{36}}L_{{14}}\right)+\delta,
\qquad \delta\in\{{-2,-1,0,1,2\}}.
$$

**结果汇总。** 4B 预测结果为 `{json.dumps(q4, ensure_ascii=False)}`；4B 严格因果结果为 `{json.dumps(causal, ensure_ascii=False)}`；14B 前瞻复现为 `{json.dumps(q14, ensure_ascii=False)}`。已发布数据集为 `{json.dumps(result['datasets'], ensure_ascii=False)}`，逐项形状、行元数据、有限值、坐标数和 SHA-256 验证为 `{json.dumps(result['verification'], ensure_ascii=False)}`，客户端构建为 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。

**分析与理论进展。** 当前最稳固的新拼图是：在干净、独立、中英双语材料中，十个不同语言家族的响应都含有可由同一物理坐标基态预测的样本特异信息；这种“模型本地、角色条件化、相对层深”的预测拓扑可接受 14B 的独立前瞻裁决。它比家族平均方向更细，也没有压缩为 Top-K、PCA 或余弦。但 Phase2268 的严格调用与删除没有任何家族同时通过，所以证据只能命名为“条件化逐坐标预测规律”，不能命名为因果齿轮、语义神经元组、通用语言算子或新数学定理。晚层 query/boundary 占优还可能包含答案准备成分。

**问题、硬伤与瓶颈。** 材料虽通过程序化语义唯一性、平衡和自然度检查，但独立人类盲评仍为 NA；自由输出是受控代码，不是开放自然回答；4B 与 14B 同属 Qwen3，不能推出跨架构普遍性；14B 只复现三个代表家族；同坐标仿射仍可能部分读取通用状态转移，而非语言关系本身；float16 落盘和固定角色末 token 是测量近似；严格因果阴性可能来自干预分布外，也可能说明预测状态不是必要通道。

**结论与下一步。** `{result['strict_conclusion']}` 原始大场只在可视化导数完成哈希、形状和构建验证后清理，账本为 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。相关脚本：`tests/glm5/phase2270_c1601_c1624_visual_atlas_cleanup.py`；结果：`tests/glm5/result/phase2270_c1601_c1624_visual_atlas_cleanup`；图谱：`frontend/public/vis_data/research_kernel/c1601_*` 至 `c1603_*`。下一阶段与当前目标相同，但应扩大“观察优先”图谱到更多构式、自然输出和异架构模型，并使用已冻结的完整坐标图寻找可重复的层—角色—坐标条件结构；在出现跨材料稳定候选前，不再反复做同一种差分搬运因果门。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load_json(final)
        append_memo(result)
        return result
    q4 = load_json(Q4_MODEL_OUT / "analysis/final.json")
    causal = load_json(Q4_CAUSAL_OUT / "analysis/final.json")
    q14 = load_json(Q14_OUT / "analysis/final.json")
    VIS.mkdir(parents=True, exist_ok=True)
    datasets = [publish_flagship_token_fields(), publish_q4_prediction_atlas(),
                publish_q14_replication_atlas()]
    catalog = update_catalog(datasets)
    verification = [verify_dataset(dataset) for dataset in datasets]
    build = frontend_build()
    checks = {
        "q4_ten_lockbox_predictive_families": len(q4["lockbox_survivors"]) == 10,
        "q4_no_strict_causal_family": causal["strict_causal_families"] == [],
        "q14_decisions_complete": len(q14["decisions"]) == len(q14["behavior"]["qualified_families"]),
        "all_visual_datasets_verified": all(all(value for key, value in row.items() if key != "id")
                                                for row in verification),
        "catalog_complete": len(catalog["entries_added"]) == len(datasets),
        "frontend_build_passed": build["passed"],
    }
    if not all(checks.values()):
        raise RuntimeError(("publication_checks_failed", checks))
    cleanup = cleanup_raw_fields([Q4_TOKEN, Q4_ROLE, Q14_FIELD])
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "datasets": [{"id": row["id"], "shape": row["shape"], "rows": row["rows"],
                      "sha256": row["sha256"]} for row in datasets],
        "verification": verification, "catalog": catalog, "frontend_build": build,
        "q4_predictive": {"families": q4["families"],
                          "lockbox_survivors": q4["lockbox_survivors"],
                          "strict_conclusion": q4["strict_conclusion"]},
        "q4_causal": {"strict_causal_families": causal["strict_causal_families"],
                      "strict_conclusion": causal["strict_conclusion"]},
        "q14_replication": {"behavior": q14["behavior"],
                            "replicated_families": q14["replicated_families"],
                            "strict_conclusion": q14["strict_conclusion"]},
        "cleanup": cleanup, "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": ("Clean bilingual all-coordinate prediction structure is broad in Qwen3-4B "
                              "and receives a preregistered model-local Qwen3-14B topology test, while "
                              "strict bidirectional causality remains unconfirmed."),
    }
    save_json(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
