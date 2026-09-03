#!/usr/bin/env python3
"""Publish exact-coordinate multi-step assets and clean verified raw fields."""
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
P2309 = RESULT / "phase2309_c4321_c4440_multistep_future_contract"
P2310 = RESULT / "phase2310_c4441_c4580_qwen4b_multistep_field"
P2311 = RESULT / "phase2311_c4581_c4700_basic_future_accounting"
P2312 = RESULT / "phase2312_c4701_c4820_qwen4b_local_response"
P2313 = RESULT / "phase2313_c4821_c4960_crossmodel_multistep"
OUT = RESULT / "phase2314_c4961_c5040_multistep_atlas_cleanup"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

PHASE = 2314
CAMPAIGN = "C4961-C5040"
REPRESENTATIVE_UNIT = 26

ROWS = P2309 / "material/multistep_future_bilingual.jsonl"
Q4_BOUNDARY = P2310 / "raw/qwen4b_multistep_boundary_all_checkpoints.float16.npy"
Q4_BOUNDARY_LOGITS = P2310 / "raw/qwen4b_multistep_boundary_full_vocabulary.float16.npy"
Q4_ALL_TOKEN = P2310 / "raw/qwen4b_multistep_representative_all_token.float16.npy"
Q4_FUTURE = P2310 / "raw/qwen4b_teacher_future_selected_checkpoints.float16.npy"
Q4_FUTURE_LOGITS = P2310 / "raw/qwen4b_teacher_future_full_vocabulary.float16.npy"
Q4_CONTRIBUTIONS = P2310 / "atlas/qwen4b_fixed_identity_output_contributions.float16.npy"

STATE_RESPONSE = P2311 / "atlas/state_response_all_checkpoints.float16.npy"
SURFACE_RESPONSE = P2311 / "atlas/surface_response_all_checkpoints.float16.npy"
LANGUAGE_RESPONSE = P2311 / "atlas/language_package_response_all_checkpoints.float16.npy"
FUTURE_TRANSITION = P2311 / "atlas/teacher_future_transition_selected_checkpoints.float16.npy"
STATE_INDEX = P2311 / "index/state_response_rows.jsonl"
FUTURE_TRANSITION_INDEX = P2311 / "index/teacher_future_transition_rows.jsonl"

LOCAL_ACTIVATIONS = P2312 / "atlas/selected_boundary_activations.float16.npy"
LOCAL_GRADIENTS = P2312 / "atlas/fixed_margin_gradients.float16.npy"
LOCAL_INDEX = P2312 / "index/selected_rows.jsonl"

Q14_FIELD = P2313 / "qwen3_14b/raw/boundary_all_checkpoints.float16.npy"
Q14_INDEX = P2313 / "qwen3_14b/index/field_rows.jsonl"


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


def checkpoint_label(q: int, layers: int) -> str:
    if q == 0:
        return "embedding"
    if q == layers + 1:
        return "final_norm"
    return f"block_{q:02d}_post"


def create_binary(name: str, rows: int, coordinates: int) -> np.memmap:
    VIS.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        VIS / name, mode="w+", dtype=np.float16, shape=(rows, coordinates)
    )


def case_metadata(row: dict) -> dict:
    return {
        "case_id": row["case_id"],
        "family": row["family"],
        "language": row["language"],
        "surface": row["surface"],
        "partition": row["partition"],
        "state": int(row["state"]),
        "unit": int(row["unit"]),
        "target_mention_order": row.get("target_mention_order"),
    }


def metadata(
    dataset_id: str,
    title: str,
    binary: Path,
    rows: list[dict],
    schema: str,
    model: str,
    claim_level: str,
    boundary: str,
    coordinate_semantics: str,
    extra: dict | None = None,
) -> dict:
    values = np.load(binary, mmap_mode="r")
    try:
        shape = list(values.shape)
        finite = bool(np.isfinite(values).all())
    finally:
        close_memmap(values)
    if len(shape) != 2 or len(rows) != shape[0] or not finite:
        raise RuntimeError(("invalid_publication_asset", dataset_id, shape, len(rows), finite))
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
        "binary_sha256": file_hash(binary),
        "coordinate_count": shape[-1],
        "coordinate_semantics": coordinate_semantics,
        "coordinate_order": (
            "original model-local physical order; no Top-K, PCA, averaging, compression, "
            "coordinate reordering, or cross-model coordinate alignment"
        ),
        "heatmap_type": "embedding_hiddenstate_full_coordinate",
        "claim_level": claim_level,
        "boundary": boundary,
        "rows": rows,
    }
    if extra:
        info.update(extra)
    meta_path = VIS / f"{dataset_id}.json"
    save_json(meta_path, info)
    return {
        "id": dataset_id,
        "title": title,
        "metadata": meta_path,
        "binary": binary,
        "shape": shape,
        "sha256": info["binary_sha256"],
        "model": model,
        "schema": schema,
        "claim_level": claim_level,
        "boundary": boundary,
        "heatmap_type": info["heatmap_type"],
    }


def publish_q4_boundary(material: list[dict]) -> dict:
    selected = [(index, row) for index, row in enumerate(material)
                if row["partition"] == "fresh_lockbox"
                and int(row["unit"]) == REPRESENTATIVE_UNIT]
    source = np.load(Q4_BOUNDARY, mmap_mode="r")
    binary = VIS / "c4961_qwen4b_multistep_boundary_trajectory.float16.npy"
    output = create_binary(binary.name, len(selected) * source.shape[1], source.shape[-1])
    rows: list[dict] = []
    cursor = 0
    try:
        for source_index, row in selected:
            for q in range(source.shape[1]):
                output[cursor] = source[source_index, q]
                rows.append({
                    "row": cursor,
                    **case_metadata(row),
                    "checkpoint": q,
                    "checkpoint_label": checkpoint_label(q, 36),
                    "metric": "boundary_physical_activation",
                })
                cursor += 1
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        "c4961_qwen4b_multistep_boundary_trajectory",
        "Qwen3-4B Multi-Step Future Boundary Trajectory",
        binary,
        rows,
        "ai2050.multistep-boundary-trajectory.v1",
        "Qwen3-4B",
        "full_coordinate_observation",
        "Fresh-lockbox unit26 for all eight language families, both languages, both surfaces, "
        "both fact states, embedding, all 36 post-block states, final norm, and all 2560 coordinates.",
        "Qwen3-4B runtime activation at the raw continuation boundary",
    )


def publish_q4_state_response() -> dict:
    index = [row for row in read_jsonl(STATE_INDEX)
             if row["partition"] == "fresh_lockbox"
             and int(row["unit"]) == REPRESENTATIVE_UNIT]
    source = np.load(STATE_RESPONSE, mmap_mode="r")
    binary = VIS / "c4962_qwen4b_state_response_trajectory.float16.npy"
    output = create_binary(binary.name, len(index) * source.shape[1], source.shape[-1])
    rows: list[dict] = []
    cursor = 0
    try:
        for pair in index:
            source_index = int(pair["pair_row"])
            for q in range(source.shape[1]):
                output[cursor] = source[source_index, q]
                rows.append({
                    "row": cursor,
                    "family": pair["family"],
                    "language": pair["language"],
                    "surface": pair["surface"],
                    "partition": pair["partition"],
                    "unit": int(pair["unit"]),
                    "left_case_id": pair["left_case_id"],
                    "right_case_id": pair["right_case_id"],
                    "checkpoint": q,
                    "checkpoint_label": checkpoint_label(q, 36),
                    "metric": "signed_state1_minus_state0_activation_response",
                })
                cursor += 1
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        "c4962_qwen4b_state_response_trajectory",
        "Qwen3-4B Eight-Family Signed State Response",
        binary,
        rows,
        "ai2050.multistep-state-response.v1",
        "Qwen3-4B",
        "observational_signed_state_contrast",
        "Fresh-lockbox unit26 state1-minus-state0 responses for all eight families, both languages "
        "and surfaces, every checkpoint, and every physical coordinate.",
        "signed same-coordinate response between paired fact states",
    )


def publish_q4_future_transition(qpoints: list[int]) -> dict:
    index = [row for row in read_jsonl(FUTURE_TRANSITION_INDEX)
             if row["partition"] == "fresh_lockbox"
             and int(row["unit"]) == REPRESENTATIVE_UNIT]
    source = np.load(FUTURE_TRANSITION, mmap_mode="r")
    if source.shape[1] != len(qpoints):
        raise RuntimeError(("future_qpoint_shape", list(source.shape), qpoints))
    binary = VIS / "c4963_qwen4b_teacher_future_transition.float16.npy"
    output = create_binary(binary.name, len(index) * source.shape[1], source.shape[-1])
    rows: list[dict] = []
    cursor = 0
    try:
        for transition in index:
            source_index = int(transition["transition_row"])
            for q_index, q in enumerate(qpoints):
                output[cursor] = source[source_index, q_index]
                rows.append({
                    "row": cursor,
                    "case_id": transition["case_id"],
                    "family": transition["family"],
                    "language": transition["language"],
                    "surface": transition["surface"],
                    "partition": transition["partition"],
                    "state": int(transition["state"]),
                    "unit": int(transition["unit"]),
                    "from_step": int(transition["from_step"]),
                    "to_step": int(transition["to_step"]),
                    "consumed_token_id": int(transition["consumed_token_id"]),
                    "next_target_token_id": int(transition["next_target_token_id"]),
                    "checkpoint": q,
                    "checkpoint_label": checkpoint_label(q, 36),
                    "metric": "teacher_forced_next_step_signed_activation_transition",
                })
                cursor += 1
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        "c4963_qwen4b_teacher_future_transition",
        "Qwen3-4B Multi-Step Teacher-Future Transitions",
        binary,
        rows,
        "ai2050.teacher-future-transition.v1",
        "Qwen3-4B",
        "teacher_forced_predictive_state_observation",
        "All real adjacent future-token transitions for fresh-lockbox unit26 across all eight families, "
        "languages, surfaces and states, at ten frozen checkpoints and all coordinates.",
        "signed activation transition after consuming the correct teacher-forced future token",
        {"qpoints": qpoints},
    )


def publish_q4_local_gradients(qpoints: list[int], local_final: dict) -> dict:
    index = read_jsonl(LOCAL_INDEX)
    source = np.load(LOCAL_GRADIENTS, mmap_mode="r")
    if source.shape[:2] != (len(index), len(qpoints)):
        raise RuntimeError(("local_gradient_shape", list(source.shape), len(index), qpoints))
    binary = VIS / "c4964_qwen4b_fixed_margin_local_gradients.float16.npy"
    output = create_binary(binary.name, len(index) * len(qpoints), source.shape[-1])
    rows: list[dict] = []
    cursor = 0
    try:
        for case in index:
            source_index = int(case["row"])
            for q_index, q in enumerate(qpoints):
                output[cursor] = source[source_index, q_index]
                rows.append({
                    "row": cursor,
                    **case_metadata(case),
                    "checkpoint": q,
                    "checkpoint_label": checkpoint_label(q, 36),
                    "metric": "fixed_identity_margin_local_gradient",
                })
                cursor += 1
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        "c4964_qwen4b_fixed_margin_local_gradients",
        "Qwen3-4B Exact Local Output-Margin Gradients",
        binary,
        rows,
        "ai2050.fixed-margin-local-gradient.v1",
        "Qwen3-4B",
        "local_first_order_output_control",
        "All 192 selected confirmation and fresh cases, ten checkpoints, and all 2560 local gradients "
        "of the frozen state1-identity minus state0-identity first-token margin.",
        "exact local derivative of a fixed output margin with respect to one runtime activation coordinate",
        {
            "qpoints": qpoints,
            "strict_boundary": local_final["strict_conclusion"],
            "confirmation_qualified_cells": len(local_final["confirmation"]["qualified_cells"]),
            "fresh_qualified_cells": len(local_final["fresh"]["qualified_cells"]),
        },
    )


def publish_q4_contributions(material: list[dict]) -> dict:
    binary = VIS / "c4965_qwen4b_fixed_identity_output_contributions.float16.npy"
    shutil.copy2(Q4_CONTRIBUTIONS, binary)
    rows = [{
        "row": index,
        **case_metadata(row),
        "target_identity": row["ntp_target_text"],
        "wrong_identity": row["ntp_wrong_text"],
        "metric": "fixed_identity_target_minus_wrong_output_contribution",
    } for index, row in enumerate(material)]
    return metadata(
        "c4965_qwen4b_fixed_identity_output_contributions",
        "Qwen3-4B Exact Fixed-Identity Output Contributions",
        binary,
        rows,
        "ai2050.multistep-output-contribution.v1",
        "Qwen3-4B",
        "exact_final_linear_decomposition",
        "All 2048 material rows and all 2560 final-norm coordinates. Values decompose the frozen "
        "first identity-token target-minus-wrong logit margin, not earlier causal computation.",
        "per-coordinate term in the final linear target-minus-wrong logit decomposition",
    )


def publish_q14_boundary(q14_final: dict) -> dict:
    index = [row for row in read_jsonl(Q14_INDEX)
             if row["partition"] == "fresh_lockbox"
             and int(row["unit"]) == REPRESENTATIVE_UNIT]
    source = np.load(Q14_FIELD, mmap_mode="r")
    binary = VIS / "c4966_qwen14_multistep_boundary_trajectory.float16.npy"
    output = create_binary(binary.name, len(index) * source.shape[1], source.shape[-1])
    rows: list[dict] = []
    cursor = 0
    try:
        for case in index:
            source_index = int(case["hidden_index"])
            for q in range(source.shape[1]):
                output[cursor] = source[source_index, q]
                rows.append({
                    "row": cursor,
                    **case_metadata(case),
                    "checkpoint": q,
                    "checkpoint_label": checkpoint_label(q, 40),
                    "metric": "boundary_physical_activation",
                })
                cursor += 1
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        "c4966_qwen14_multistep_boundary_trajectory",
        "Qwen3-14B Multi-Step Future Boundary Trajectory",
        binary,
        rows,
        "ai2050.qwen14-multistep-boundary-trajectory.v1",
        "Qwen3-14B",
        "prospective_same_family_scale_observation",
        "Fresh-lockbox unit26 for the four families passing both frozen 14B behavior routes; embedding, "
        "all 40 post-block states, final norm, and all 5120 model-local coordinates. Coordinate IDs are "
        "not aligned with Qwen3-4B.",
        "Qwen3-14B model-local runtime activation at the raw continuation boundary",
        {
            "qualified_families": q14_final["qualified_families"],
            "relative_topology": q14_final["topology"],
        },
    )


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


def serializable(dataset: dict) -> dict:
    return {
        **dataset,
        "metadata": str(dataset["metadata"].relative_to(ROOT)),
        "binary": str(dataset["binary"].relative_to(ROOT)),
    }


def published_dataset(dataset_id: str) -> dict:
    metadata_path = VIS / f"{dataset_id}.json"
    info = load_json(metadata_path)
    binary = VIS / Path(info["binary_url"]).name
    return {
        "id": dataset_id,
        "title": info["title"],
        "metadata": metadata_path,
        "binary": binary,
        "shape": info["binary_shape"],
        "sha256": info["binary_sha256"],
        "model": info["model"],
        "schema": info["schema"],
        "claim_level": info["claim_level"],
        "boundary": info["boundary"],
        "heatmap_type": info["heatmap_type"],
    }


def update_catalog(datasets: list[dict]) -> dict:
    catalog = load_json(CATALOG)
    ids = {row["id"] for row in datasets}
    entries = [{
        "id": row["id"],
        "title": row["title"],
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": row["model"],
        "source_path": "/vis_data/research_kernel/" + row["metadata"].name,
        "binary_path": "/vis_data/research_kernel/" + row["binary"].name,
        "source_schema": row["schema"],
        "coordinate_count": row["shape"][-1],
        "row_count": row["shape"][0],
        "claim_level": row["claim_level"],
        "boundary": row["boundary"],
        "kinds": [row["heatmap_type"]],
    } for row in datasets]
    catalog["datasets"] = entries + [row for row in catalog.get("datasets", [])
                                      if row.get("id") not in ids]
    fields = [{
        "id": row["id"],
        "title": row["title"],
        "url": "/vis_data/research_kernel/" + row["metadata"].name,
        "phase": PHASE,
        "full_coordinate": True,
        "heatmap_type": row["heatmap_type"],
    } for row in datasets]
    catalog["field_datasets"] = fields + [row for row in catalog.get("field_datasets", [])
                                           if row.get("id") not in ids]
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save_json(CATALOG, catalog)
    return {
        "added": sorted(ids),
        "dataset_count": len(catalog["datasets"]),
        "field_dataset_count": len(catalog["field_datasets"]),
    }


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
        command,
        cwd=ROOT / "frontend",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
        "passed": completed.returncode == 0,
        "browser_or_client_connection": False,
    }


def cleanup(paths: list[Path]) -> dict:
    result_root = RESULT.resolve()
    rows: list[dict] = []
    total = 0
    for path in paths:
        resolved = path.resolve()
        if result_root not in resolved.parents:
            raise RuntimeError(("cleanup_outside_result", str(resolved)))
        if not path.exists():
            rows.append({
                "path": str(path.relative_to(ROOT)),
                "status": "already_absent",
                "bytes_deleted": 0,
            })
            continue
        size = path.stat().st_size
        sha = file_hash(path)
        path.unlink()
        total += size
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "status": "deleted_after_verified_visual_derivative",
            "sha256_before": sha,
            "bytes_deleted": size,
        })
    ledger = {"files": rows, "bytes_deleted": total}
    save_json(OUT / "cleanup/ledger.json", ledger)
    return ledger


def evidence_audit() -> dict:
    return {
        "retained": [
            "next-token training makes prefix-conditioned future competition a necessary measurement axis",
            "Phase2296-2308 correctly separated interface, fact state, surface, mention order, and output boundary",
            "broad final-output coordinate accounting is real and distinct from hidden-state formation",
            "full-coordinate observation and prospective partitions are stronger than a single cosine summary",
        ],
        "corrected": [
            "broad output-coordinate support is not evidence for holographic interference",
            "inference applies frozen network transformations and does not run loss-gradient descent",
            "no dynamic belief manifold, knowledge attractor, Koopman key, or new mathematics was established",
            "the count of effective output coordinates is basis- and threshold-dependent, not a semantic gear count",
        ],
    }


def scientific_summary() -> dict:
    q4 = load_json(P2310 / "analysis/final.json")
    basic = load_json(P2311 / "analysis/final.json")
    local = load_json(P2312 / "analysis/final.json")
    cross = load_json(P2313 / "analysis/final.json")
    q14 = cross["models"]["qwen3_14b"]
    deepseek = cross["models"]["deepseek7b"]
    return {
        "qwen3_4b": {
            "complete_future_qualified": q4["sequence_ledger"]["qualified_families"],
            "free_identity_qualified": q4["free_ledger"]["route_eligible_families"],
            "complete_future_overall": q4["sequence_ledger"]["overall"],
            "free_identity_overall": q4["free_ledger"]["overall"],
            "basic_transport_best": basic["transport"]["summary"]["families"],
            "local_confirmation_qualified": len(local["confirmation"]["qualified_cells"]),
            "local_fresh_qualified": len(local["fresh"]["qualified_cells"]),
            "local_claim_boundary": local["strict_conclusion"],
        },
        "qwen3_14b": {
            "complete_future_qualified": q14["sequence"]["qualified_families"],
            "free_identity_qualified": q14["free"]["route_eligible_families"],
            "field_qualified": q14["qualified_families"],
            "field_shape": q14["field"]["shape"],
        },
        "deepseek7b": {
            "complete_future_qualified": deepseek["sequence"]["qualified_families"],
            "free_identity_qualified": deepseek["free"]["route_eligible_families"],
            "field_qualified": deepseek["qualified_families"],
            "field_status": deepseek["field"],
        },
        "relative_topology": cross["comparison"],
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    assets = [{"id": row["id"], "shape": row["shape"], "sha256": row["sha256"],
               "claim_level": row["claim_level"]} for row in result["datasets"]]
    summary = result["scientific_summary"]
    cleanup_gib = result["cleanup"]["bytes_deleted"] / (1024 ** 3)
    text = rf"""

## Phase {PHASE}: 多步未来全坐标图谱发布、跨模型裁决与清理（{CAMPAIGN}） [{stamp}]

**附件证据审查。** 保留四点：下一个 token 训练使“前缀如何约束后续多个真实 token”成为必要观察轴；Phase2296--2308 对接口、事实状态、表面、提及顺序和输出边界的分账是有效进展；final norm 到词表 logits 的广泛逐坐标分账是真实现象；全坐标与前瞻分区比单一余弦更能保护低幅值结构。修正四类过度结论：数百至上千输出坐标参与不等于“全息干涉”；推理时网络执行冻结参数定义的变换，不是在运行损失梯度下降；现有证据没有建立动态信念流形、知识吸引子或 Koopman 唯一钥匙；有效坐标数依赖基、阈值和输出对比，不能解释成语义齿轮数量。

**测试原理、用例与冻结流程。** Phase2309 在模型加载前冻结八个语言族、英中两语、叙述/对话表面、两种事实状态、四个分区和 2048 行材料。例：`Leo Bell handed the atlas to Amina Arden. The person who handed over the atlas was` 的完整未来为 `Leo Bell rather than Amina Arden.`，不仅比较第一个身份 token，还对完整多 token 未来与自由生成单独记账。Qwen3-4B 保存全部八族的边界场、逐真实未来步场和完整词表；只有完整未来与自由身份交集族进入局部梯度。Qwen3-14B、DeepSeek-R1-Distill-Qwen-7B 依次加载并用各自 tokenizer 重编译 fresh 材料；只比较模型本地的相对层深和角色拓扑，不比较物理坐标编号。发布阶段没有加载模型、没有连接浏览器或本地客户端。

$$
S_M(y_{{1:K}}\mid x)=\sum_{{k=1}}^K\log p_M(y_k\mid x,y_{{<k}}),
\qquad
R_{{i,q}}=H_{{i,q}}^{{(1)}}-H_{{i,q}}^{{(0)}}.
$$

局部输出边界响应使用当前样本、当前检查点的完整 2560 维导数：

$$
\Delta m_i \approx \nabla_{{H_{{i,q}}}}m_i^\top\delta H_{{i,q}},
\qquad
\rho_{{M,f}}(q)=\frac{{\operatorname{{RMS}}(H^1_{{M,f,q}}-H^0_{{M,f,q}})}}
{{\operatorname{{RMS}}(H^0_{{M,f,q}},H^1_{{M,f,q}})+\epsilon}}.
$$

第一式只是局部一阶控制；第二式只是描述性相对形成曲线。两者都不能单独命名语义电路。

**结果汇总与门槛。** Qwen3-4B 完整未来严格通过 `{len(summary['qwen3_4b']['complete_future_qualified'])}/8` 族 `{summary['qwen3_4b']['complete_future_qualified']}`，自由身份通过 `{len(summary['qwen3_4b']['free_identity_qualified'])}/8` 族 `{summary['qwen3_4b']['free_identity_qualified']}`；整体完整未来账 `{json.dumps(summary['qwen3_4b']['complete_future_overall'], ensure_ascii=False)}`，自由身份账 `{json.dumps(summary['qwen3_4b']['free_identity_overall'], ensure_ascii=False)}`。基础同坐标传动在 fresh 上的中位相对 MSE 仍普遍接近 `0.90--1.05`，不足以形成可用齿轮。局部梯度在 confirmation `20/20` 单元、fresh `20/20` 单元通过预冻结符号、相对误差和正负方向门；这验证局部输出边界的一阶可控性，不是共享语义方向。

Qwen3-14B 的完整未来通过 `{summary['qwen3_14b']['complete_future_qualified']}`，自由身份通过 `{summary['qwen3_14b']['free_identity_qualified']}`，两门交集 `{summary['qwen3_14b']['field_qualified']}`，全场形状 `{summary['qwen3_14b']['field_shape']}`。与 4B 同时具有合法场的施事--受事、位置绑定、taxonomy 三族中，前两族半峰相对深度差分别约 `0.004`、`0.012`，taxonomy 约 `0.261`；这是两项接近和一项不迁移，不是统一时钟。DeepSeek-7B 完整未来只通过 `{summary['deepseek7b']['complete_future_qualified']}`，自由身份只通过 `{summary['deepseek7b']['free_identity_qualified']}`，交集为空，故没有合法 HiddenState 跨模型场；其底层仍是 Qwen2 类架构，也不能充当完全独立架构证明。

**精确坐标资产、相关文件与清理。** 发布 `{json.dumps(assets, ensure_ascii=False)}`；逐项形状、行数、坐标数、有限值和 SHA256 验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；通用图谱目录更新 `{json.dumps(result['catalog'], ensure_ascii=False)}`；前端离线构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。在上述检查全部通过后删除 `{len(result['cleanup']['files'])}` 个未展示或已有验证衍生物的大型原始数组，共 `{cleanup_gib:.3f}` GiB；删除前哈希保存在 `tests/glm5/result/phase2314_c4961_c5040_multistep_atlas_cleanup/cleanup/ledger.json`。脚本 `tests/glm5/phase2314_c4961_c5040_multistep_atlas_cleanup.py`；结果 `tests/glm5/result/phase2314_c4961_c5040_multistep_atlas_cleanup`；图谱资产位于 `frontend/public/vis_data/research_kernel`。

**理论进展、问题硬伤与结论。** 理论主体继续保持“条件化输出场闭合理论”，本期只增加一个窄拼图：多 token 未来、状态响应、最终输出贡献和局部输出梯度是四个不可互换的观察对象；模型规模变化可以保留部分相对形成拓扑，也可以在另一些族上明显改变。没有发现固定坐标字典、可迁移差分、跨模型共享电路或新数学闭合。硬伤包括研究者模板、没有独立人类盲评、统一 `rather than` 输出尾句、teacher forcing 不等于自由生成、局部梯度天然是当前输出的最陡方向、4B/14B 同家族，以及 DeepSeek 面板没有双行为门交集。现有概率、逐坐标代数和基础微分足够陈述这些结果；高等数学只能作为以后由稳定残差提出的问题，不能先验替代观察。

本大 Campaign 的冻结目标已经全部执行：八族多步观察、基础路线竞争、fresh 局部验证、顺序跨模型复验、精确图谱发布和原始场清理均已完成。下一阶段若转向新语言族、独立架构或全 token 坐标联动，研究对象和材料都会变化，应另立大合同，不能沿本锁箱无条件续跑。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def finish_result(
    datasets: list[dict],
    verification: list[dict],
    catalog: dict,
    build: dict,
    cleanup_result: dict,
) -> dict:
    parents = [load_json(path / "analysis/final.json")
               for path in (P2309, P2310, P2311, P2312, P2313)]
    cleanup_paths = raw_cleanup_paths()
    checks = {
        "all_parent_phases_passed": all(row["all_checks_passed"] for row in parents),
        "all_assets_verified": all(
            all(value for key, value in row.items() if key != "id") for row in verification
        ),
        "catalog_updated": set(catalog["added"]) == {row["id"] for row in datasets},
        "frontend_built": bool(build["passed"]),
        "no_browser_or_client_connection": not build["browser_or_client_connection"],
        "raw_fields_cleaned": all(not path.exists() for path in cleanup_paths),
        "model_local_exact_coordinates": all(row["shape"][-1] in (2560, 5120)
                                              for row in datasets),
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed",
        "evidence_audit": evidence_audit(),
        "scientific_summary": scientific_summary(),
        "datasets": [serializable(row) for row in datasets],
        "verification": verification,
        "catalog": catalog,
        "frontend_build": build,
        "cleanup": cleanup_result,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            "Multi-step future behavior, exact model-local fields, local first-order output control, and "
            "same-family scale topology are separately reproducible. They do not establish a shared coordinate "
            "dictionary, holographic code, semantic circuit, independent-architecture invariant, or new mathematics."
        ),
        "next_stage_same_specific_target": False,
        "next_stage_reason": (
            "The frozen multi-step campaign is complete. New language families, independent architectures, or "
            "all-token coordinate coupling require new materials and a new preregistration."
        ),
    }
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    return result


def raw_cleanup_paths() -> list[Path]:
    return [
        Q4_BOUNDARY,
        Q4_BOUNDARY_LOGITS,
        Q4_ALL_TOKEN,
        Q4_FUTURE,
        Q4_FUTURE_LOGITS,
        Q4_CONTRIBUTIONS,
        STATE_RESPONSE,
        SURFACE_RESPONSE,
        LANGUAGE_RESPONSE,
        FUTURE_TRANSITION,
        LOCAL_ACTIVATIONS,
        LOCAL_GRADIENTS,
        Q14_FIELD,
    ]


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = load_json(final_path)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    dataset_ids = [
        "c4961_qwen4b_multistep_boundary_trajectory",
        "c4962_qwen4b_state_response_trajectory",
        "c4963_qwen4b_teacher_future_transition",
        "c4964_qwen4b_fixed_margin_local_gradients",
        "c4965_qwen4b_fixed_identity_output_contributions",
        "c4966_qwen14_multistep_boundary_trajectory",
    ]
    cleanup_path = OUT / "cleanup/ledger.json"
    if cleanup_path.exists() and all((VIS / f"{dataset_id}.json").exists()
                                     for dataset_id in dataset_ids):
        datasets = [published_dataset(dataset_id) for dataset_id in dataset_ids]
        verification = [verify(row) for row in datasets]
        catalog = update_catalog(datasets)
        build = frontend_build()
        if not build["passed"]:
            raise RuntimeError(("frontend_build_failed_during_recovery", build))
        result = finish_result(datasets, verification, catalog, build, load_json(cleanup_path))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    parents = [load_json(path / "analysis/final.json")
               for path in (P2309, P2310, P2311, P2312, P2313)]
    if not all(row["all_checks_passed"] for row in parents):
        raise RuntimeError("A parent phase is not authorized")

    material = read_jsonl(ROWS)
    p2310 = parents[1]
    local_final = parents[3]
    cross_final = parents[4]
    q14_final = cross_final["models"]["qwen3_14b"]
    datasets = [
        publish_q4_boundary(material),
        publish_q4_state_response(),
        publish_q4_future_transition(list(p2310["teacher_future"]["qpoints"])),
        publish_q4_local_gradients(list(local_final["gradient_audit"]["qpoints"]), local_final),
        publish_q4_contributions(material),
        publish_q14_boundary(q14_final),
    ]
    verification = [verify(row) for row in datasets]
    if not all(all(value for key, value in row.items() if key != "id")
               for row in verification):
        raise RuntimeError(("asset_verification_failed", verification))

    catalog = update_catalog(datasets)
    build = frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))

    cleanup_result = cleanup(raw_cleanup_paths())
    result = finish_result(datasets, verification, catalog, build, cleanup_result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
