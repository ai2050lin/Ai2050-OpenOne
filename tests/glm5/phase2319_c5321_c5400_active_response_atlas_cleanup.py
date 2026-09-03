#!/usr/bin/env python3
"""Publish exact-coordinate active-response atlases and clean verified raw fields."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

import phase2315_c5041_c5100_active_response_contract as contract


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2315 = RESULT / "phase2315_c5041_c5100_active_response_contract"
P2316 = RESULT / "phase2316_c5101_c5160_qwen4b_active_baseline"
P2317 = RESULT / "phase2317_c5161_c5240_directional_response_identification"
P2318 = RESULT / "phase2318_c5241_c5320_crossmodel_directional_topology"
OUT = RESULT / "phase2319_c5321_c5400_active_response_atlas_cleanup"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

PHASE = 2319
CAMPAIGN = "C5321-C5400"
Q4_BOUNDARY = P2316 / "raw/qwen4b_boundary_all_checkpoints.float16.npy"
Q4_BOUNDARY_INDEX = P2316 / "index/boundary_rows.jsonl"
Q4_ALL_TOKEN = P2316 / "raw/qwen4b_representative_all_token_all_checkpoints.float16.npy"
Q4_ALL_TOKEN_INDEX = P2316 / "index/all_token_rows.jsonl"
Q4_DERIVATIVE = P2317 / "raw/directional_derivative.float16.npy"
Q4_EVEN = P2317 / "raw/even_response.float16.npy"
Q4_ACTIVE_INDEX = P2317 / "index/active_rows.jsonl"
Q4_PASSPORT = P2317 / "atlas/fresh_family_vs_global_coordinate_improvement.float32.npy"
PROBE_LEDGER = P2317 / "config/probe_ledger.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def checkpoint_label(q: int, block_count: int) -> str:
    if q == 0:
        return "embedding"
    if q == block_count + 1:
        return "final_norm"
    return f"block_{q:02d}_post"


def create_binary(name: str, rows: int, coordinates: int, dtype: np.dtype) -> np.memmap:
    VIS.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        VIS / name, mode="w+", dtype=dtype, shape=(rows, coordinates)
    )


def write_metadata(
    dataset_id: str,
    title: str,
    binary: Path,
    rows: list[dict],
    model: str,
    schema: str,
    claim_level: str,
    boundary: str,
    coordinate_semantics: str,
    extra: dict | None = None,
) -> dict:
    values = np.load(binary, mmap_mode="r")
    try:
        shape = list(values.shape)
        finite = bool(np.isfinite(values).all())
        dtype = str(values.dtype)
    finally:
        close_memmap(values)
    if len(shape) != 2 or len(rows) != shape[0] or not finite:
        raise RuntimeError(("invalid_asset", dataset_id, shape, len(rows), finite))
    info = {
        "schema": schema,
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "id": dataset_id,
        "title": title,
        "model": model,
        "binary_url": "/vis_data/research_kernel/" + binary.name,
        "binary_shape": shape,
        "binary_dtype": dtype,
        "binary_sha256": file_hash(binary),
        "coordinate_count": shape[-1],
        "coordinate_semantics": coordinate_semantics,
        "coordinate_order": (
            "original model-local physical order; no Top-K, PCA, averaging, coordinate "
            "reordering, or cross-model physical-coordinate alignment"
        ),
        "heatmap_type": "embedding_hiddenstate_full_coordinate",
        "claim_level": claim_level,
        "boundary": boundary,
        "rows": rows,
    }
    if extra:
        info.update(extra)
    metadata_path = VIS / f"{dataset_id}.json"
    save_json(metadata_path, info)
    return {
        "id": dataset_id,
        "title": title,
        "metadata": metadata_path,
        "binary": binary,
        "shape": shape,
        "model": model,
        "schema": schema,
        "claim_level": claim_level,
        "boundary": boundary,
        "heatmap_type": info["heatmap_type"],
    }


def case_fields(row: dict) -> dict:
    return {
        key: row[key] for key in
        ("case_id", "family", "language", "surface", "partition", "state", "unit")
        if key in row
    }


def publish_boundary() -> dict:
    index = read_jsonl(Q4_BOUNDARY_INDEX)
    selected = [(i, row) for i, row in enumerate(index) if row["partition"] == "fresh_lockbox"]
    source = np.load(Q4_BOUNDARY, mmap_mode="r")
    output = create_binary(
        "c5321_qwen4b_fresh_boundary_full_coordinate.float16.npy",
        len(selected) * source.shape[1], source.shape[-1], source.dtype,
    )
    rows = []
    cursor = 0
    for source_index, row in selected:
        for q in range(source.shape[1]):
            output[cursor] = source[source_index, q]
            rows.append({**case_fields(row), "source_index": source_index, "checkpoint_q": q,
                         "checkpoint": checkpoint_label(q, 36), "token_role": "output_boundary"})
            cursor += 1
    output.flush(); close_memmap(output); close_memmap(source)
    binary = VIS / "c5321_qwen4b_fresh_boundary_full_coordinate.float16.npy"
    return write_metadata(
        "c5321_qwen4b_fresh_boundary_full_coordinate",
        "Qwen3-4B fresh lockbox boundary trajectory, exact coordinates",
        binary, rows, "Qwen3-4B", "full_coordinate_boundary_trajectory_v1",
        "observational", "teacher-forced prefix output boundary",
        "raw hidden activation at the output-boundary token",
        {"sample_selection": "all 512 fresh_lockbox rows", "includes_embedding": True},
    )


def publish_all_token() -> dict:
    index = read_jsonl(Q4_ALL_TOKEN_INDEX)
    selected = [(i, row) for i, row in enumerate(index)
                if row["partition"] == "fresh_lockbox" and row["surface"] == "narrative"]
    source = np.load(Q4_ALL_TOKEN, mmap_mode="r")
    row_count = sum(source.shape[1] * int(row["token_count"]) for _, row in selected)
    output = create_binary(
        "c5322_qwen4b_fresh_all_token_full_coordinate.float16.npy",
        row_count, source.shape[-1], source.dtype,
    )
    rows = []
    cursor = 0
    for source_index, row in selected:
        token_count = int(row["token_count"])
        role_positions = row.get("role_positions", {})
        for q in range(source.shape[1]):
            for token_position in range(token_count):
                output[cursor] = source[source_index, q, token_position]
                roles = [name for name, positions in role_positions.items()
                         if token_position in positions]
                rows.append({
                    **case_fields(row), "source_index": source_index,
                    "checkpoint_q": q, "checkpoint": checkpoint_label(q, 36),
                    "token_position": token_position,
                    "token_id": int(row["token_ids"][token_position]),
                    "token_roles": roles or ["other"],
                    "boundary_position": int(row["boundary_position"]),
                })
                cursor += 1
    output.flush(); close_memmap(output); close_memmap(source)
    binary = VIS / "c5322_qwen4b_fresh_all_token_full_coordinate.float16.npy"
    return write_metadata(
        "c5322_qwen4b_fresh_all_token_full_coordinate",
        "Qwen3-4B fresh bilingual all-token field, exact coordinates",
        binary, rows, "Qwen3-4B", "full_coordinate_all_token_field_v1",
        "observational", "valid prompt tokens only",
        "raw embedding or hidden activation at a physical token position",
        {"sample_selection": "32 fresh_lockbox narrative rows; every family/language/state",
         "includes_embedding": True, "padded_tokens_removed": True},
    )


def flatten_directional(
    dataset_id: str,
    title: str,
    source_path: Path,
    index_rows: list[dict],
    probes: list[dict],
    sources: list[int],
    targets: dict[int, list[int]],
    model: str,
    semantics: str,
    boundary: str,
) -> dict:
    source = np.load(source_path, mmap_mode="r")
    if list(source.shape[:-1]) != [len(index_rows), len(sources), len(probes), 3]:
        raise RuntimeError(("directional_index_mismatch", source_path, source.shape))
    binary = VIS / f"{dataset_id}.{str(source.dtype)}.npy"
    output = create_binary(binary.name, int(np.prod(source.shape[:-1])), source.shape[-1], source.dtype)
    rows = []
    cursor = 0
    for row_index, row in enumerate(index_rows):
        for source_index, source_q in enumerate(sources):
            for probe_index, probe in enumerate(probes):
                for target_index, target_q in enumerate(targets[source_q]):
                    output[cursor] = source[row_index, source_index, probe_index, target_index]
                    rows.append({
                        **case_fields(row), "active_index": row_index,
                        "source_q": source_q, "source_relative_depth": source_q / max(sources),
                        "probe": probe_index, "probe_kind": probe["kind"],
                        "probe_members": probe["members"], "target_q": target_q,
                        "target_slot": ("q_plus_1", "q_plus_4", "final_norm")[target_index],
                    })
                    cursor += 1
    output.flush(); close_memmap(output); close_memmap(source)
    return write_metadata(
        dataset_id, title, binary, rows, model, "full_coordinate_directional_response_v1",
        "active local response", boundary, semantics,
        {"source_qpoints": sources, "target_qpoints": targets,
         "probe_ledger": probes, "exact_flattening_only": True},
    )


def publish_q4_directional() -> list[dict]:
    index = read_jsonl(Q4_ACTIVE_INDEX)
    probes = load_json(PROBE_LEDGER)
    sources = [10, 20, 30]
    targets = {q: [q + 1, min(q + 4, 36), 37] for q in sources}
    return [
        flatten_directional(
            "c5323_qwen4b_directional_derivative", "Qwen3-4B signed directional response",
            Q4_DERIVATIVE, index, probes, sources, targets, "Qwen3-4B",
            "central finite directional derivative in every target physical coordinate",
            "1% source-state-norm Rademacher perturbation",
        ),
        flatten_directional(
            "c5324_qwen4b_even_response", "Qwen3-4B finite-dose even response",
            Q4_EVEN, index, probes, sources, targets, "Qwen3-4B",
            "symmetric finite-dose residual in every target physical coordinate",
            "half-sum of positive and negative responses minus baseline",
        ),
    ]


def publish_passport() -> dict:
    source = np.load(Q4_PASSPORT, mmap_mode="r")
    probes = load_json(PROBE_LEDGER)[:8]
    sources = [10, 20, 30]
    targets = {q: [q + 1, min(q + 4, 36), 37] for q in sources}
    binary = VIS / "c5325_qwen4b_family_global_coordinate_passport.float32.npy"
    output = create_binary(binary.name, int(np.prod(source.shape[:-1])), source.shape[-1], source.dtype)
    output[:] = source.reshape(-1, source.shape[-1])
    rows = []
    for family in contract.FAMILIES:
        for source_q in sources:
            for probe in probes:
                for target_index, target_q in enumerate(targets[source_q]):
                    rows.append({"family": family, "source_q": source_q,
                                 "probe": int(probe["probe"]), "probe_kind": probe["kind"],
                                 "target_q": target_q,
                                 "target_slot": ("q_plus_1", "q_plus_4", "final_norm")[target_index]})
    output.flush(); close_memmap(output); close_memmap(source)
    return write_metadata(
        "c5325_qwen4b_family_global_coordinate_passport",
        "Qwen3-4B family-versus-global coordinate improvement passport", binary, rows,
        "Qwen3-4B", "full_coordinate_prediction_improvement_v1", "prospective derived",
        "fresh partitions predicted from frozen discovery means",
        "positive means family-conditioned squared error improved over global mean",
        {"warning": "a coordinate-wise prediction diagnostic, not a semantic-neuron score"},
    )


def model_probe_ledger() -> list[dict]:
    return [
        {"probe": 0, "kind": "base_rademacher", "members": [0]},
        {"probe": 1, "kind": "base_rademacher", "members": [1]},
        {"probe": 2, "kind": "base_rademacher", "members": [2]},
        {"probe": 3, "kind": "base_rademacher", "members": [3]},
        {"probe": 4, "kind": "pair_sum", "members": [0, 1]},
        {"probe": 5, "kind": "pair_sum", "members": [2, 3]},
    ]


def publish_crossmodel(final: dict) -> list[dict]:
    assets = []
    identifiers = {"qwen3_14b": 5326, "glm4": 5328, "deepseek7b": 5330}
    for model_key, number in identifiers.items():
        model = final["models"].get(model_key, {})
        if not model.get("all_checks_passed"):
            continue
        field = model["field"]
        worker = P2318 / model_key
        index = read_jsonl(worker / "index/active_rows.jsonl")
        sources = [int(q) for q in field["sources"]]
        targets = {int(q): [int(value) for value in values]
                   for q, values in field["targets"].items()}
        for offset, (kind, semantics) in enumerate((
            ("directional_derivative", "central finite directional derivative"),
            ("even_response", "symmetric finite-dose residual"),
        )):
            dataset_id = f"c{number + offset}_{model_key}_{kind}"
            assets.append(flatten_directional(
                dataset_id, f"{model['model']} model-local {kind.replace('_', ' ')}",
                worker / f"raw/{kind}.float16.npy", index, model_probe_ledger(),
                sources, targets, model["model"], semantics + " in each model-local coordinate",
                "1% source-state-norm perturbation; coordinates are not aligned across models",
            ))
    return assets


def verify(dataset: dict) -> dict:
    info = load_json(dataset["metadata"])
    values = np.load(dataset["binary"], mmap_mode="r")
    try:
        checks = {
            "shape": list(values.shape) == info["binary_shape"] == dataset["shape"],
            "rows": len(info["rows"]) == values.shape[0],
            "coordinate_count": values.shape[-1] == info["coordinate_count"],
            "finite": bool(np.isfinite(values).all()),
        }
    finally:
        close_memmap(values)
    checks["sha256"] = file_hash(dataset["binary"]) == info["binary_sha256"]
    return {"id": dataset["id"], **checks}


def update_catalog(datasets: list[dict]) -> dict:
    catalog = load_json(CATALOG)
    ids = {row["id"] for row in datasets}
    entries = [{
        "id": row["id"], "title": row["title"], "phase": PHASE, "campaign": CAMPAIGN,
        "model": row["model"],
        "source_path": "/vis_data/research_kernel/" + row["metadata"].name,
        "binary_path": "/vis_data/research_kernel/" + row["binary"].name,
        "source_schema": row["schema"], "coordinate_count": row["shape"][-1],
        "row_count": row["shape"][0], "claim_level": row["claim_level"],
        "boundary": row["boundary"], "kinds": [row["heatmap_type"]],
    } for row in datasets]
    catalog["datasets"] = entries + [row for row in catalog.get("datasets", [])
                                      if row.get("id") not in ids]
    fields = [{
        "id": row["id"], "title": row["title"],
        "url": "/vis_data/research_kernel/" + row["metadata"].name,
        "phase": PHASE, "full_coordinate": True, "heatmap_type": row["heatmap_type"],
    } for row in datasets]
    catalog["field_datasets"] = fields + [row for row in catalog.get("field_datasets", [])
                                           if row.get("id") not in ids]
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
            raise FileNotFoundError("No Node runtime is available")
        command = [str(candidates[0]), str(ROOT / "frontend/node_modules/vite/bin/vite.js"), "build"]
    completed = subprocess.run(
        command, cwd=ROOT / "frontend", capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900,
    )
    return {"command": command, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:],
            "passed": completed.returncode == 0, "browser_or_client_connection": False}


def raw_cleanup_paths(crossmodel: dict) -> list[Path]:
    paths = [Q4_BOUNDARY, Q4_ALL_TOKEN, Q4_DERIVATIVE, Q4_EVEN]
    for model_key in ("qwen3_14b", "glm4", "deepseek7b"):
        if crossmodel["models"].get(model_key, {}).get("all_checks_passed"):
            paths.extend((P2318 / model_key / "raw/directional_derivative.float16.npy",
                          P2318 / model_key / "raw/even_response.float16.npy"))
    paths.extend((P2318 / "qwen3_4b_reference/raw/directional_derivative.float16.npy",
                  P2318 / "qwen3_4b_reference/raw/even_response.float16.npy"))
    return paths


def cleanup(paths: list[Path]) -> dict:
    result_root = RESULT.resolve()
    rows = []
    total = 0
    for path in paths:
        resolved = path.resolve()
        if result_root not in resolved.parents:
            raise RuntimeError(("cleanup_outside_result", str(resolved)))
        if not path.exists():
            rows.append({"path": str(path.relative_to(ROOT)), "status": "already_absent",
                         "bytes_deleted": 0})
            continue
        size = path.stat().st_size
        sha = file_hash(path)
        path.unlink()
        total += size
        rows.append({"path": str(path.relative_to(ROOT)),
                     "status": "deleted_after_verified_coordinate_publication",
                     "sha256_before": sha, "bytes_deleted": size})
    ledger = {"files": rows, "bytes_deleted": total}
    save_json(OUT / "cleanup/ledger.json", ledger)
    return ledger


def serializable(dataset: dict) -> dict:
    return {**dataset, "metadata": str(dataset["metadata"].relative_to(ROOT)),
            "binary": str(dataset["binary"].relative_to(ROOT))}


def scientific_summary() -> dict:
    p2316 = load_json(P2316 / "analysis/final.json")
    p2317 = load_json(P2317 / "analysis/final.json")
    p2318 = load_json(P2318 / "analysis/final.json")
    return {
        "qwen4_behavior": {"qualified_families": p2316["qualified_families"],
                           "sequence_overall": p2316["sequence"]["overall"],
                           "free_overall": p2316["free"]["overall"]},
        "qwen4_prediction": p2317["prediction"]["summary"]["fresh_lockbox"],
        "qwen4_linearity": p2317["linearity"],
        "crossmodel_successful": p2318["successful_models"],
        "crossmodel_checks": p2318["checks"],
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    assets = [{"id": row["id"], "shape": row["shape"], "model": row["model"],
               "claim_level": row["claim_level"]} for row in result["datasets"]]
    gib = result["cleanup"]["bytes_deleted"] / (1024 ** 3)
    record = rf"""

## Phase {PHASE}: 主动响应全坐标图谱发布、证据总审计与清理（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 本阶段没有新增模型结论，而是把 Phase2316 的 8 语言族双语边界场与代表性全 token 场、Phase2317 的正负方向导数/偶响应/逐坐标预测护照，以及 Phase2318 成功模型的本地扰动响应，统一编译为二维“行事件 × 原始物理坐标”资产。边界资产覆盖全部 512 行 fresh_lockbox；全 token 资产覆盖 32 行 fresh_lockbox narrative 样本的每个有效 token、embedding、36 个 block 后状态和 final norm；方向资产没有删坐标、取 Top-K、PCA、均值压缩或跨模型坐标对齐。
$$
R_{{i,q,r,t,j}}=\frac{{H_{{t,j}}(h_q+\epsilon r)-H_{{t,j}}(h_q-\epsilon r)}}{{2\epsilon\lVert h_q\rVert_2}},\qquad
E_{{i,q,r,t,j}}=\frac{{H_{{t,j}}^++H_{{t,j}}^-}}{{2}}-H_{{t,j}}^0.
$$

**结果汇总、相关文件与门槛。** 科学汇总 `{json.dumps(result['scientific_summary'], ensure_ascii=False)}`。发布资产 `{json.dumps(assets, ensure_ascii=False)}`；形状、行数、坐标数、有限值与 SHA256 验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；目录更新 `{json.dumps(result['catalog'], ensure_ascii=False)}`；Vite 离线构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2319_c5321_c5400_active_response_atlas_cleanup.py`；结果目录 `tests/glm5/result/phase2319_c5321_c5400_active_response_atlas_cleanup`；可视化资产 `frontend/public/vis_data/research_kernel`。未启动浏览器、客户端或本地服务。

**证据审查、理论进展与硬伤。** 附件对 Phase2309-2314 的“teacher-forced 多步场、对角运输失败、局部输出梯度”概括可保留；“模型已做自由规划”“局部梯度就是语义齿轮”“已经出现流形/Grassmann/Koopman 或新数学结构”均过度。新结果进一步表明，冻结发现集上的全局平均局部响应比族平均更能预测 fresh 样本，且有限剂量成对叠加误差和偶响应不小；因此当前更像强基态条件化的共享传播响应，尚非稳定族算子。行为门没有任何族同时通过完整候选与自由生成，主动场只能作为观察和仪器校准，不能升级为语言理解机制。跨模型仅比较功能统计，物理坐标不可直接对应。数据仍缺独立人类盲评，发现样本小，BF16/有限剂量会污染微分线性区。

**结论、清理与下一阶段授权。** 理论主体保持“条件化输出场闭合理论”，组织原则保持“复用—差分—条件化”。本期没有数学闭合，只建立了可逐坐标回看的观察底座。资产全部验证且前端构建通过后，删除 `{len(result['cleanup']['files'])}` 个不再单独展示的大型原始副本，共 `{gib:.3f}` GiB；删除前 SHA256 写入 `tests/glm5/result/phase2319_c5321_c5400_active_response_atlas_cleanup/cleanup/ledger.json`。下一阶段目标仍相同，但当前双行为门为空且局部响应主要是共享传播，故不重复搬运差分；授权从图谱中观察跨族复用事件、符号翻转、层迁移和低幅坐标共同出现模式，再用全新材料前瞻验证，不把图谱模式直接命名为语义齿轮。"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = load_json(final_path)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parents = [load_json(path / "analysis/final.json") for path in (P2315, P2316, P2317, P2318)]
    if not all(row["all_checks_passed"] for row in parents):
        raise RuntimeError("A parent phase is not authorized")
    crossmodel = parents[-1]
    datasets = [publish_boundary(), publish_all_token(), *publish_q4_directional(),
                publish_passport(), *publish_crossmodel(crossmodel)]
    verification = [verify(row) for row in datasets]
    if not all(all(value for key, value in row.items() if key != "id") for row in verification):
        raise RuntimeError(("asset_verification_failed", verification))
    catalog = update_catalog(datasets)
    build = frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    cleanup_result = cleanup(raw_cleanup_paths(crossmodel))
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
        "datasets": [serializable(row) for row in datasets], "verification": verification,
        "catalog": catalog, "frontend_build": build, "cleanup": cleanup_result,
        "scientific_summary": scientific_summary(),
        "evidence_audit": {
            "retained": ["teacher forcing is not free planning",
                         "diagonal future transport failed on fresh samples",
                         "local output gradients are narrow sample-local output sensitivities"],
            "corrected": ["no semantic manifold, Grassmann operator, Koopman key, or new mathematics was established",
                          "a local output gradient is not a language-family gear",
                          "shared perturbation propagation is not semantic mechanism closure"],
        },
        "checks": {
            "parents_authorized": True,
            "all_assets_verified": all(all(value for key, value in row.items() if key != "id")
                                           for row in verification),
            "catalog_updated": set(catalog["added"]) == {row["id"] for row in datasets},
            "frontend_build_passed": build["passed"],
            "no_client_connection": not build["browser_or_client_connection"],
            "raw_fields_cleaned": all(not path.exists() for path in raw_cleanup_paths(crossmodel)),
            "no_coordinate_compression": True,
        },
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save_json(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
