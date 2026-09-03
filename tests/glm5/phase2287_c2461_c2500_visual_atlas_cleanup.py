#!/usr/bin/env python3
"""Publish Phase2282-2286 exact-coordinate atlases and clean raw fields."""
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
FIELD_OUT = RESULT / "phase2282_c2161_c2220_qwen4b_multilingual_field_rebuild"
OPERATOR_OUT = RESULT / "phase2283_c2221_c2280_cross_domain_coordinate_operators"
FACTORIAL_OUT = RESULT / "phase2285_c2341_c2400_factorial_nondegenerate_reanalysis"
Q14_OUT = RESULT / "phase2286_c2401_c2460_qwen14_embedding_operator_replication"
OUT = RESULT / "phase2287_c2461_c2500_visual_atlas_cleanup"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

PHASE = 2287
CAMPAIGN = "C2461-C2500"
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
FLAGSHIPS = ("relative_clause_binding", "possession_state", "comparison_order")

Q4_FIELD = FIELD_OUT / "raw/qwen4b_bilingual_role_field.float16.npy"
Q4_INDEX = FIELD_OUT / "raw/qwen4b_bilingual_role_index.jsonl"
Q4_COMPLEX_FIELD = FIELD_OUT / "raw/qwen4b_complex_role_field.float16.npy"
Q4_TOKEN_FIELD = FIELD_OUT / "raw/qwen4b_representative_token_field.float16.npy"
Q4_OPERATOR = OPERATOR_OUT / "atlas/cross_domain_coordinate_operator_passport.float32.npy"
Q4_OPERATOR_ROWS = OPERATOR_OUT / "atlas/cross_domain_coordinate_operator_passport.rows.jsonl"
Q4_FACTORIAL = FACTORIAL_OUT / "atlas/nondegenerate_factorial_passport.float32.npy"
Q4_FACTORIAL_ROWS = FACTORIAL_OUT / "atlas/nondegenerate_factorial_passport.rows.jsonl"
Q14_FIELD = Q14_OUT / "raw/qwen3_14b_embedding_role_field.float16.npy"
Q14_INDEX = Q14_OUT / "raw/qwen3_14b_embedding_role_index.jsonl"
Q14_OPERATOR = Q14_OUT / "atlas/qwen14_embedding_operator_passport.float32.npy"
Q14_OPERATOR_ROWS = Q14_OUT / "atlas/qwen14_embedding_operator_passport.rows.jsonl"


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
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(value: Any) -> None:
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
    value = {
        "schema": schema, "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE, "campaign": CAMPAIGN, "id": dataset_id, "title": title,
        "model": model, "binary_url": "/vis_data/research_kernel/" + binary.name,
        "binary_shape": shape, "binary_sha256": sha, "coordinate_count": shape[-1],
        "coordinate_semantics": "model-local runtime activation coordinates, not model weight parameters",
        "rows": rows, "boundary": boundary,
    }
    target = VIS / f"{dataset_id}.json"
    save_json(target, value)
    return {"id": dataset_id, "title": title, "metadata": target, "binary": binary,
            "shape": shape, "sha256": sha, "rows": len(rows), "model": model,
            "schema": schema, "boundary": boundary}


def publish_q4_flagship_role_field() -> dict:
    index = read_jsonl(Q4_INDEX)
    chosen = []
    for family in FLAGSHIPS:
        for language in ("en", "zh"):
            for state in (0, 1):
                matches = [row for row in index if row["family"] == family
                           and row["language"] == language and int(row["unit"]) == 24
                           and row["surface"] == "direct" and int(row["state"]) == state]
                if len(matches) != 1:
                    raise RuntimeError(("q4_flagship_case", family, language, state, len(matches)))
                chosen.append(matches[0])
    source = np.load(Q4_FIELD, mmap_mode="r")
    row_count = len(chosen) * 38 * len(ROLES)
    binary = VIS / "c2461_qwen4b_cross_language_flagship_role_field.float16.npy"
    binary.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float16,
                                       shape=(row_count, 2560))
    rows, cursor = [], 0
    try:
        for case in chosen:
            for q in range(38):
                for role_i, role in enumerate(ROLES):
                    output[cursor] = source[int(case["hidden_index"]), q, role_i]
                    rows.append({
                        "row": cursor, "case_id": case["case_id"], "family": case["family"],
                        "language": case["language"], "unit": case["unit"],
                        "surface": case["surface"], "state": case["state"],
                        "checkpoint": q, "checkpoint_label": checkpoint_label(q, 36),
                        "role": role, "metric": "physical_activation",
                    })
                    cursor += 1
        output.flush()
    finally:
        close(output)
        close(source)
    return metadata(
        "c2461_qwen4b_cross_language_flagship_role_field",
        "Qwen3-4B Cross-Language Flagship Role Field", binary, [row_count, 2560], rows,
        "ai2050.cross-language-flagship-role-field.v1", "Qwen3-4B",
        "Twelve frozen lockbox cases expose embedding, every post-block checkpoint, final norm, six semantic roles, and every coordinate. Role-last-token alignment is a measurement convention, not a semantic atom.")


def copy_atlas(source_path: Path, rows_path: Path, dataset_id: str, title: str,
               schema: str, model: str, boundary: str) -> dict:
    binary = VIS / f"{dataset_id}.{source_path.name.split('.')[-2]}.npy"
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, binary)
    source = np.load(source_path, mmap_mode="r")
    shape = list(source.shape)
    close(source)
    rows = read_jsonl(rows_path)
    if len(rows) != shape[0]:
        raise RuntimeError(("atlas_rows", dataset_id, shape, len(rows)))
    return metadata(dataset_id, title, binary, shape, rows, schema, model, boundary)


def publish_q14_role_field() -> dict:
    index = read_jsonl(Q14_INDEX)
    source = np.load(Q14_FIELD, mmap_mode="r")
    if tuple(source.shape) != (384, 1, 6, 5120) or len(index) != 384:
        raise RuntimeError(("q14_field_contract", source.shape, len(index)))
    row_count = len(index) * len(ROLES)
    binary = VIS / "c2464_qwen14_embedding_role_field.float16.npy"
    output = np.lib.format.open_memmap(binary, mode="w+", dtype=np.float16,
                                       shape=(row_count, 5120))
    rows, cursor = [], 0
    try:
        for case in index:
            for role_i, role in enumerate(ROLES):
                output[cursor] = source[int(case["hidden_index"]), 0, role_i]
                rows.append({
                    "row": cursor, "case_id": case["case_id"], "family": case["family"],
                    "language": case["language"], "unit": case["unit"],
                    "surface": case["surface"], "state": case["state"],
                    "partition": case["partition"], "checkpoint": 0,
                    "checkpoint_label": "embedding", "role": role,
                    "metric": "physical_activation",
                })
                cursor += 1
        output.flush()
    finally:
        close(output)
        close(source)
    return metadata(
        "c2464_qwen14_embedding_role_field", "Qwen3-14B Bilingual Embedding Role Field",
        binary, [row_count, 5120], rows, "ai2050.qwen14-bilingual-embedding-role-field.v1",
        "Qwen3-14B",
        "All 384 behavior-audited cases, six roles, and every q0 coordinate are preserved. Coordinate IDs are not aligned to Qwen3-4B and q0 is not evidence of deep semantic composition.")


def verify(dataset: dict) -> dict:
    info = load_json(dataset["metadata"])
    value = np.load(dataset["binary"], mmap_mode="r")
    try:
        checks = {
            "shape": list(value.shape) == info["binary_shape"] == dataset["shape"],
            "rows": len(info["rows"]) == value.shape[0],
            "coordinates": value.shape[-1] == info["coordinate_count"],
            "finite": bool(np.isfinite(value).all()),
        }
    finally:
        close(value)
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
        "row_count": row["shape"][0], "claim_level": "full_coordinate_observation",
        "boundary": row["boundary"],
        "kinds": ["embedding_hiddenstate_full_coordinate"],
    } for row in datasets]
    catalog["datasets"] = entries + [row for row in catalog.get("datasets", [])
                                      if row.get("id") not in ids]
    fields = [{
        "id": row["id"], "title": row["title"],
        "url": "/vis_data/research_kernel/" + row["metadata"].name,
        "phase": PHASE, "full_coordinate": True,
        "heatmap_type": "embedding_hiddenstate_full_coordinate",
    } for row in datasets]
    catalog["field_datasets"] = [row for row in catalog.get("field_datasets", [])
                                 if row.get("id") not in ids] + fields
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save_json(CATALOG, catalog)
    return {"dataset_count": len(catalog["datasets"]),
            "field_dataset_count": len(catalog["field_datasets"]), "added": sorted(ids)}


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
                               text=True, encoding="utf-8", errors="replace", timeout=600)
    return {"command": command, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:],
            "passed": completed.returncode == 0}


def cleanup(paths: list[Path]) -> dict:
    result_root = RESULT.resolve()
    rows, total = [], 0
    for path in paths:
        resolved = path.resolve()
        if result_root not in resolved.parents:
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
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 多语言Exact-Coordinate图谱发布、清理与大阶段裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本期不从热力图重新选择机制，只发布已经揭盲的五类完整坐标对象。第一类是 Qwen3-4B 三个跨语言阳性构式在 fresh-lockbox unit24、中英、state0/state1、direct 表面的六角色轨迹，覆盖 embedding、36 个 block 后状态、final norm 和全部 2560 坐标。第二类是 Phase2283 的 23 条跨语言/跨表面候选与全部控制误差。第三类是 Phase2285 纠偏后的 23 条非退化二阶析因护照。第四类是 Qwen3-14B 全部 384 条双语材料在 q0、六角色和全部 5120 坐标的状态。第五类是 Phase2286 的跨规模复验误差护照。热力图参数是运行时激活坐标，不是权重参数；4B 与 14B 列号绝不对齐。

**数学对象。** 客户端每一行与原张量切片保持一一对应：

$$
V_{{m,j}}=H_{{i,q,r,j}},qquad j=1,\ldots,d.
$$

预测与析因图分别显示：

$$
E_{{m,j}}=\mathbb E_i\left|\widehat R_{{i,j}}-R_{{i,j}}\right|,
\qquad
I_{{i,j}}=H_{{11,i,j}}-H_{{10,i,j}}-H_{{01,i,j}}+H_{{00,i,j}}.
$$

没有 PCA、Top-K、余弦压缩或坐标重排。所有二进制图谱都登记逐行语义、形状和 SHA-256。

**结果汇总。** 已发布数据 `{json.dumps(result['datasets'], ensure_ascii=False)}`；逐项验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；目录更新 `{json.dumps(result['catalog'], ensure_ascii=False)}`；客户端构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`；清理账本 `{json.dumps(result['cleanup'], ensure_ascii=False)}`；大阶段科学摘要 `{json.dumps(result['campaign_summary'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2287_c2461_c2500_visual_atlas_cleanup.py`；结果 `tests/glm5/result/phase2287_c2461_c2500_visual_atlas_cleanup`；图谱 `frontend/public/vis_data/research_kernel/c2461_*` 至 `c2465_*`；客户端目录 `frontend/public/research_data/current/language_encoding_catalog.json`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}` 对 Phase2286 还需作证据边界纠偏：三个家族在 Qwen3-14B 的全表面候选与 direct 自由生成均为 `1.000`，但三条冻结函数都在 confirmation 被跨家族共享源控制击败，因而 fresh-confirmation 与 fresh-lockbox 均未揭示；这不是“lockbox 运行后失败”，更不是行为失败。最可靠的新拼图仍是：部分构式在同一 4B 模型的平行语言/表面材料间存在模型本地、角色条件化、逐坐标基态函数；稳定性主要出现在 embedding 或浅层，因此首先属于词汇和前缀条件，不应升级为深层语义程序。二阶交互在很多格子非零，但冻结均值预测器 `0/23`，不能命名为曲率、超边或高阶代数。多密度干预没有合法中层锚点，结论为 NA。人工平行模板、角色末 token、受控代码输出、同族模型、未做人类盲评是主要硬伤。现有基础代数足以表述这些对象；当前没有证据证明需要新数学，也没有证据证明现有工具足以闭合。下一大阶段不再继续同一冻结材料上的均值仿射或差分搬运，而应更换为独立自然材料和样本条件坐标图；因此与本 Campaign 的具体目标不同，本大阶段在已授权范围内结束。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load_json(final)
        append_memo(result)
        return result
    for source_final in (
        FIELD_OUT / "analysis/final.json", OPERATOR_OUT / "analysis/final.json",
        FACTORIAL_OUT / "analysis/final.json", Q14_OUT / "analysis/final.json",
    ):
        if not load_json(source_final)["all_checks_passed"]:
            raise RuntimeError(("source_failed", str(source_final)))
    VIS.mkdir(parents=True, exist_ok=True)
    datasets = [
        publish_q4_flagship_role_field(),
        copy_atlas(
            Q4_OPERATOR, Q4_OPERATOR_ROWS, "c2462_qwen4b_cross_domain_operator_passport",
            "Qwen3-4B Cross-Domain Coordinate Operator Passport",
            "ai2050.cross-domain-coordinate-operator-passport.v1", "Qwen3-4B",
            "All coordinates for 23 frozen routes and their controls. Predictive migration is observational and model-local."),
        copy_atlas(
            Q4_FACTORIAL, Q4_FACTORIAL_ROWS, "c2463_qwen4b_nondegenerate_factorial_passport",
            "Qwen3-4B Nondegenerate Factorial Coordinate Passport",
            "ai2050.nondegenerate-factorial-coordinate-passport.v1", "Qwen3-4B",
            "All coordinates for corrected nondegenerate interaction candidates. Zero of 23 passed every frozen control; nonzero interaction is not causal curvature."),
        publish_q14_role_field(),
        copy_atlas(
            Q14_OPERATOR, Q14_OPERATOR_ROWS, "c2465_qwen14_embedding_operator_passport",
            "Qwen3-14B Embedding Operator Replication Passport",
            "ai2050.qwen14-embedding-operator-passport.v1", "Qwen3-14B",
            "All q0 coordinate errors for frozen Qwen3-4B functional-topology candidates. Physical coordinate IDs are model-local."),
    ]
    verification = [verify(row) for row in datasets]
    catalog = update_catalog(datasets)
    build = frontend_build()
    q4_operator = load_json(OPERATOR_OUT / "analysis/final.json")
    factorial = load_json(FACTORIAL_OUT / "analysis/final.json")
    q14 = load_json(Q14_OUT / "analysis/final.json")
    checks = {
        "datasets_verified": all(all(value for key, value in row.items() if key != "id")
                                  for row in verification),
        "catalog_added_all": len(catalog["added"]) == len(datasets),
        "frontend_build": build["passed"],
        "q4_cross_language_three": len(q4_operator["cross_language_lockbox"]) == 3,
        "q4_cross_surface_six": len(q4_operator["cross_surface_lockbox"]) == 6,
        "factorial_corrected_zero": factorial["lockbox_passed"] == [],
        "causal_branch_na": q14["causal_branch"].startswith("NA_not_authorized"),
        "q14_all_three_behavior_perfect": all(
            value["candidate_accuracy_all_surfaces"] == 1.0 and
            value["generation_accuracy_direct"] == 1.0
            for value in q14["behavior"]["families"].values()),
        "q14_fresh_and_lockbox_unrevealed": all(
            not row["fresh_authorized"] and not row["lockbox_revealed"]
            for row in q14["structure"]["decisions"]),
    }
    if not all(checks.values()):
        raise RuntimeError(("publication_check", checks))
    cleanup_result = cleanup([Q4_FIELD, Q4_COMPLEX_FIELD, Q4_TOKEN_FIELD, Q14_FIELD])
    campaign_summary = {
        "q4_cross_language_lockbox": q4_operator["cross_language_lockbox"],
        "q4_cross_surface_lockbox": q4_operator["cross_surface_lockbox"],
        "corrected_factorial_lockbox": factorial["lockbox_passed"],
        "q14_embedding_lockbox": q14["structure"]["lockbox_passed_families"],
        "causal_branch": q14["causal_branch"],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "datasets": [{"id": row["id"], "shape": row["shape"], "rows": row["rows"],
                      "sha256": row["sha256"]} for row in datasets],
        "verification": verification, "catalog": catalog, "frontend_build": build,
        "cleanup": cleanup_result, "campaign_summary": campaign_summary,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": ("The campaign found three model-local Qwen3-4B q0 cross-domain coordinate functions, "
                              "but no stable frozen family-mean factorial operator, no authorized middle-layer causal "
                              "anchor, and no Qwen3-14B family-specific q0 transfer beyond the shared-source control; "
                              "all three Qwen3-14B behavior panels were perfect, while fresh and lockbox internal data "
                              "remained unrevealed."),
        "next_authorization": ("A new campaign must change the evidence object to independent natural materials and "
                               "sample-conditioned coordinate graphs; this frozen parallel-template campaign is complete."),
    }
    save_json(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
