#!/usr/bin/env python3
"""Publish the residual operator field, verify the client, clean raw fields, and append Phase 2413."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2407 = RESULT / "phase2407_c24881_c25200_qwen4b_component_fullfield"
P2408 = RESULT / "phase2408_c25201_c25520_fullcoordinate_deconfounding"
P2409 = RESULT / "phase2409_c25521_c25840_state_dependent_coordinate_operator"
P2411 = RESULT / "phase2411_c26161_c26480_crosslayer_composition_output_bridge"
P2412 = RESULT / "phase2412_c26481_c26800_frozen_crossmodel_operator_replication"
OUT = RESULT / "phase2413_c26801_c27120_publish_cleanup_audit"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2413
CAMPAIGN = "C26801-C27120"
SCHEMA = "c26801.residual_state_operator_field.v1"
JSON_PATH = PUBLIC / "c26801_residual_state_operator_field.json"
NPY_PATH = PUBLIC / "c26801_residual_state_operator_field.float32.npy"
BUNDLED_NODE = Path(r"C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
VITE = ROOT / "frontend/node_modules/vite/bin/vite.js"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def publish() -> dict:
    state = np.load(P2407 / "raw/selection_state_event.float16.npy", mmap_mode="r")
    passports = {component: np.load(P2408 / f"derived/selection_{component}_family_residual_passport.float32.npy", mmap_mode="r")
                 for component in ("total", "attention", "mlp")}
    slopes = np.load(P2409 / "derived/selection_matched_diagonal_slope.float32.npy", mmap_mode="r")
    gains = np.load(P2409 / "derived/selection_coordinate_gain.float32.npy", mmap_mode="r")
    families = ["causal", "comparison", "ownership", "preference", "role_binding", "spatial", "taxonomy", "temporal"]
    events = {"fact1_relation": 1, "query_end": 4, "answer_boundary": 7}
    rows: list[dict] = []; vectors: list[np.ndarray] = []

    def add(metadata: dict, vector: np.ndarray) -> None:
        index = len(vectors); value = np.asarray(vector, dtype=np.float32)
        if value.shape != (2560,) or not np.isfinite(value).all(): raise RuntimeError((metadata, value.shape))
        vectors.append(value)
        rows.append({**metadata, "row_index": index, "values": [float(v) for v in value]})

    # One exact sample: relation-token embedding plus query/answer HiddenState checkpoints.
    add({"source": "sample_state", "coordinate_kind": "embedding", "component": "state", "family": "preference",
         "layer": 0, "event": "fact1_relation", "case_id": "dsel-preference-u0-en-canonical-d0"}, state[0, 0, events["fact1_relation"]])
    for event_name in ("query_end", "answer_boundary"):
        for qpoint in (12, 24, 35):
            add({"source": "sample_state", "coordinate_kind": "hidden_state", "component": "state", "family": "preference",
                 "layer": qpoint, "event": event_name, "case_id": "dsel-preference-u0-en-canonical-d0"},
                state[0, qpoint, events[event_name]])
    # Discovery family residual passports for total/A/M at fixed physical coordinates.
    for component in ("total", "attention", "mlp"):
        for qpoint in (11, 23, 35):
            for event_name in ("query_end", "answer_boundary"):
                for fi, family in enumerate(families):
                    add({"source": "family_passport", "coordinate_kind": "component_update", "component": component,
                         "family": family, "layer": qpoint, "event": event_name},
                        passports[component][qpoint, events[event_name], fi])
    # State-conditioned diagonal law and its held-out matched-vs-mismatch coordinate advantage.
    for qpoint in (11, 23, 35):
        for event_name in ("query_end", "answer_boundary"):
            add({"source": "state_slope", "coordinate_kind": "operator_parameter", "component": "total", "family": "all",
                 "layer": qpoint, "event": event_name}, slopes[0, qpoint, events[event_name]])
            add({"source": "physical_gain", "coordinate_kind": "heldout_coordinate_evidence", "component": "total", "family": "all",
                 "layer": qpoint, "event": event_name, "split": "template_lockbox"},
                gains[0, 1, 1, qpoint, events[event_name]])
    matrix = np.stack(vectors).astype(np.float32)
    NPY_PATH.parent.mkdir(parents=True, exist_ok=True); np.save(NPY_PATH, matrix)
    phase2411 = json.loads((P2411 / "analysis/final.json").read_text(encoding="utf-8"))
    phase2412 = json.loads((P2412 / "analysis/final.json").read_text(encoding="utf-8"))
    payload = {
        "schema": SCHEMA, "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B-BF16",
        "dimensions": list(range(2560)), "coordinate_count": 2560, "rows": rows,
        "binary": {"path": "/vis_data/research_kernel/c26801_residual_state_operator_field.float32.npy",
                   "dtype": "float32", "shape": list(matrix.shape), "sha256": sha256(NPY_PATH)},
        "crosslayer": phase2411["crosslayer"],
        "crossmodel_geometry": phase2412["cross_architecture"],
        "summary": {"selection_state_increment_template": 0.048873597978894345,
                    "selection_physical_advantage_template": 0.05360633427061601,
                    "selection_adjacent_relation_geometry": 0.8770437836647034,
                    "selection_adjacent_coordinate_cosine": -0.044141944497823715},
        "coordinate_semantics": "q0 is the exact event-token embedding; q12/q24/q35 are HiddenState inputs to those blocks; component rows are residual-stream Attention/MLP/total updates; slope rows are fitted per-coordinate operator parameters.",
        "claim_boundary": "All 2560 values are Qwen3-4B physical activation coordinates or fitted parameters on that fixed basis, not model weights, independent neurons, a universal semantic code, or a causal language mechanism.",
    }
    save(JSON_PATH, payload)
    for value in passports.values(): close(value)
    close(state); close(slopes); close(gains)
    result = {"json": str(JSON_PATH), "json_bytes": JSON_PATH.stat().st_size, "json_sha256": sha256(JSON_PATH),
              "binary": str(NPY_PATH), "binary_bytes": NPY_PATH.stat().st_size, "binary_sha256": sha256(NPY_PATH),
              "rows": len(rows), "dimensions": 2560, "shape": list(matrix.shape)}
    save(OUT / "analysis/publish.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2)); return result


def cleanup_raw() -> dict:
    targets = list((P2407 / "raw").glob("*.npy"))
    for key in ("qwen14b", "glm4", "deepseek7b"):
        targets.extend((P2412 / key / "raw").glob("*.npy"))
    workspace = ROOT.resolve()
    resolved = []
    for path in targets:
        target = path.resolve()
        if workspace not in target.parents: raise RuntimeError(("outside workspace", target))
        if not target.is_file(): continue
        resolved.append((target, target.stat().st_size))
    removed = []
    for target, size in resolved:
        target.unlink(); removed.append({"path": str(target), "bytes": size})
    result = {"files": len(removed), "bytes": sum(item["bytes"] for item in removed),
              "gib": sum(item["bytes"] for item in removed) / 2**30, "removed": removed, "recoverable": False}
    save(OUT / "analysis/cleanup.json", result); return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 残差状态算子全坐标发布、客户端验证与原始场清理（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 从Phase2407–2412冻结结果发布一个可审计坐标资产：同一Qwen4B样本的关系token词嵌入、query/answer事件q12/q24/q35 HiddenState；8个选择族在total/Attention/MLP、q11/q23/q35、query/answer的残差护照；同坐标状态斜率及整模板matched-vs-mismatch逐坐标收益。每行保留全部2560坐标，JSON供客户端，float32 NPY作参数级伴随资产。客户端新增独立热力图类型并支持默认坐标筛选和“全部2560坐标”。

$$a_j=\frac{{\sum_i\widetilde H_{{ij}}\widetilde U_{{ij}}}}{{\sum_i\widetilde H_{{ij}}^2+\epsilon}},\qquad
g_j^{{phys}}=\sum_i\left[(U_{{ij}}-\widehat U_{{mismatch,ij}})^2-(U_{{ij}}-\widehat U_{{matched,ij}})^2\right].$$

**结果汇总。** 发布 `{json.dumps(result['publish'], ensure_ascii=False)}`；前端构建 `{json.dumps(result['build'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`；审计 `{json.dumps(result['audit'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2413_c26801_c27120_publish_cleanup_audit.py`；资产`frontend/public/vis_data/research_kernel/c26801_residual_state_operator_field.json`及`.float32.npy`；路由`frontend/src/researchKernel/heatmapResearchRoute.js`、加载`frontend/src/researchKernel/useResearchKernel.js`、渲染`frontend/src/components/app/ResearchHeatmapRoute.jsx`、入口`frontend/src/App.jsx`；final与清理台账位于`tests/glm5/result/phase2413_c26801_c27120_publish_cleanup_audit`。

**分析与理论进展。** 客户端把四类对象明确分开：词嵌入/HiddenState观测、Attention/MLP真实组件更新、族残差护照、拟合算子参数与锁箱证据。这样可直接检查低值坐标和正负纹理，不把Top-K当主要分析。跨层关系几何与同坐标余弦只作为摘要，避免把0.85级关系图稳定误读成坐标向量搬运。

**问题硬伤与结论。** 发布的是代表性Qwen4B全坐标切片，不是所有2176样本的浏览器内复制；三模型不同宽度只发布关系几何摘要。构建通过不验证科学结论。客户端/派生资产确认后，删除未直接展示的Phase2407与Phase2412原始npy以回收空间；derived护照、逐坐标收益、索引、行为结果和发布资产保留。当前最强结论仍是“模型内固定坐标耦合与跨层/跨模型族关系几何并存”，尚未破解语言编译机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def finalize() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        print(json.dumps(json.loads(final_path.read_text(encoding="utf-8")), ensure_ascii=True, indent=2))
        return
    publish_result = json.loads((OUT / "analysis/publish.json").read_text(encoding="utf-8")) if (OUT / "analysis/publish.json").exists() else publish()
    completed = subprocess.run([str(BUNDLED_NODE), str(VITE), "build"], cwd=ROOT / "frontend", capture_output=True, text=True, encoding="utf-8", errors="replace")
    build = {"command": f"{BUNDLED_NODE} {VITE} build", "exit_code": completed.returncode,
             "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]}
    save(OUT / "analysis/frontend_build.json", build)
    if completed.returncode != 0: raise RuntimeError(build)
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8")); binary = np.load(NPY_PATH, mmap_mode="r")
    audit = {"schema": payload.get("schema") == SCHEMA, "rows": len(payload.get("rows", [])) == binary.shape[0] == publish_result["rows"],
             "dimensions": binary.shape[1] == 2560 and len(payload.get("dimensions", [])) == 2560,
             "finite": bool(np.isfinite(binary).all()), "binary_hash": sha256(NPY_PATH) == payload["binary"]["sha256"],
             "embedding_present": any(row["coordinate_kind"] == "embedding" for row in payload["rows"]),
             "hidden_state_present": any(row["coordinate_kind"] == "hidden_state" for row in payload["rows"]),
             "attention_mlp_present": {row["component"] for row in payload["rows"]} >= {"attention", "mlp"},
             "operator_and_evidence_present": {row["source"] for row in payload["rows"]} >= {"state_slope", "physical_gain"}}
    close(binary)
    if not all(audit.values()): raise RuntimeError(audit)
    cleanup = cleanup_raw()
    result = {"phase": PHASE, "campaign": CAMPAIGN, "publish": publish_result, "build": build,
              "cleanup": cleanup, "audit": audit, "all_checks_passed": completed.returncode == 0 and all(audit.values())}
    save(final_path, result); append_memo(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--publish", action="store_true"); parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.publish: publish()
    elif args.finalize: finalize()
    else: parser.error("pass --publish or --finalize")


if __name__ == "__main__": main()
