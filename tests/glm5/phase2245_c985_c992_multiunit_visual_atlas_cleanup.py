#!/usr/bin/env python3
"""Export Phase 2244 full-coordinate unit prototypes and clean sample fields."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase2234_c870_c884_broad_family_gear_contract as base  # noqa: E402
import phase2243_c959_c970_multiunit_cross_scale_contract as contract  # noqa: E402


PHASE = 2245
CAMPAIGNS = tuple(f"C{i}" for i in range(985, 993))
SOURCE = ROOT / "tests/glm5/result/phase2244_c971_c984_multiunit_cross_model_topology"
OUT = ROOT / "tests/glm5/result/phase2245_c985_c992_multiunit_visual_atlas_cleanup"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export_model(model: str, stage: dict) -> dict:
    topology_path = SOURCE / model / "analysis/unit_role_topology.json"
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    source_path = ROOT / topology["prototype_path"]
    source_index = read_rows(SOURCE / model / "analysis/unit_semantic_response_prototype_index.jsonl")
    source = np.load(source_path, mmap_mode="r")
    coordinate_count = int(source.shape[-1])
    row_count = int(np.prod(source.shape[:-1]))
    stem = f"c992_{model}_multiunit_family_response_atlas"
    binary_path = VIS / f"{stem}.float16.npy"
    metadata_path = VIS / f"{stem}.json"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(binary_path, mode="w+", dtype=np.float16,
                                       shape=(row_count, coordinate_count))
    rows = []
    cursor = 0
    try:
        for proto in source_index:
            proto_i = int(proto["prototype_index"])
            for checkpoint in range(source.shape[1]):
                for role_i, role in enumerate(base.ROLES):
                    target[cursor] = source[proto_i, checkpoint, role_i]
                    rows.append({
                        "row_index": cursor, "model": model, "family": proto["family"],
                        "unit": int(proto["unit"]), "checkpoint": checkpoint,
                        "relative_depth": checkpoint / max(1, source.shape[1] - 1),
                        "role": role, "signal": "signed_semantic_response_prototype",
                    })
                    cursor += 1
    finally:
        target.flush()
        mmap = getattr(target, "_mmap", None)
        if mmap is not None:
            mmap.close()
        mmap = getattr(source, "_mmap", None)
        if mmap is not None:
            mmap.close()
    binary_hash = sha256(binary_path)
    metadata = {
        "schema": "ai2050.multiunit-full-coordinate-family-response.v1", "phase": PHASE,
        "campaigns": list(CAMPAIGNS), "model": model,
        "description": "Every physical activation coordinate of four prospective unit-level family response prototypes.",
        "coordinate_semantics": "activation coordinates inside one model; checkpoint 0 is embedding; no cross-model coordinate alignment",
        "dtype": "float16", "shape": [row_count, coordinate_count], "rows": rows,
        "binary": binary_path.name, "sha256": binary_hash,
        "behavior": {"candidate_accuracy": stage["model_results"][model]["candidate_accuracy"],
                     "generation_accuracy": stage["model_results"][model]["generation_accuracy"],
                     "qualified_families": stage["qualified_families"][model]},
        "boundary": "Signed prototypes average four language/surface truth contrasts. They are full-coordinate observations, not parameters or causal circuits.",
    }
    save(metadata_path, metadata)
    check = np.load(binary_path, mmap_mode="r")
    try:
        valid = list(check.shape) == metadata["shape"] and check.dtype == np.float16 and np.isfinite(check).all()
    finally:
        mmap = getattr(check, "_mmap", None)
        if mmap is not None:
            mmap.close()
    return {
        "id": stem, "model": model, "json": metadata_path.name, "binary": binary_path.name,
        "shape": metadata["shape"], "rows": row_count, "coordinates": coordinate_count,
        "sha256": binary_hash, "verified": bool(valid and sha256(binary_path) == binary_hash),
    }


def export_cross_model(stage: dict) -> dict:
    cross = stage["cross_model_retrieval"]
    rows = []
    values = []
    for kind in ("raw", "unit_centered"):
        for direction in ("forward", "reverse"):
            ledger = cross.get(kind, {}).get(direction, {})
            for item in ledger.get("rows", []):
                rows.append({"kind": kind, "direction": direction, "unit": item["unit"],
                             "family": item["family"], "predicted": item["predicted"]})
                values.append([item["same_family_distance"], item["nearest_wrong_distance"], item["margin"]])
    stem = "c992_multiunit_cross_model_family_retrieval"
    binary = VIS / f"{stem}.float16.npy"
    metadata_path = VIS / f"{stem}.json"
    array = np.asarray(values, dtype=np.float16)
    np.save(binary, array, allow_pickle=False)
    metadata = {
        "schema": "ai2050.multiunit-cross-model-family-retrieval.v1", "phase": PHASE,
        "campaigns": list(CAMPAIGNS), "dtype": "float16", "shape": list(array.shape),
        "columns": ["same_family_distance", "nearest_wrong_distance", "margin"],
        "rows": rows, "binary": binary.name, "sha256": sha256(binary),
        "hypothesis_gates": stage["hypothesis_gates"],
        "boundary": "Distances compare relative-depth role topology only; columns are not physical model coordinates.",
    }
    save(metadata_path, metadata)
    return {"id": stem, "json": metadata_path.name, "binary": binary.name,
            "shape": list(array.shape), "rows": len(rows), "coordinates": 3,
            "sha256": metadata["sha256"], "verified": bool(np.isfinite(array).all())}


def update_catalog(exports: list[dict]) -> list[str]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8-sig"))
    datasets = catalog.setdefault("datasets", [])
    ids = {item.get("id") for item in datasets}
    added = []
    for item in exports:
        if item["id"] in ids:
            continue
        is_matrix = item["id"].endswith("family_retrieval")
        datasets.append({
            "id": item["id"],
            "title": ("C992 Multi-unit Cross-model Family Retrieval" if is_matrix else
                      f"C992 {item.get('model')} Multi-unit Full-coordinate Family Responses"),
            "phase": PHASE, "campaign": "C985-C992",
            "model": item.get("model", "Qwen3-4B / Qwen3-14B"),
            "source_path": f"/vis_data/research_kernel/{item['json']}",
            "binary_path": f"/vis_data/research_kernel/{item['binary']}",
            "source_schema": ("ai2050.multiunit-cross-model-family-retrieval.v1" if is_matrix else
                              "ai2050.multiunit-full-coordinate-family-response.v1"),
            "coordinate_count": item["coordinates"], "row_count": item["rows"],
            "claim_level": "prospective_multiunit_observation",
            "boundary": ("Cross-model role-topology retrieval; no physical coordinate alignment." if is_matrix else
                         "All activation coordinates of unit response prototypes; not weights or causal circuits."),
            "kinds": (["same_family_distance", "nearest_wrong_distance", "margin"] if is_matrix else
                      ["signed_semantic_response_prototype"]),
        })
        ids.add(item["id"]); added.append(item["id"])
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    json.loads(CATALOG.read_text(encoding="utf-8"))
    return added


def cleanup_sample_fields(stage: dict) -> list[dict]:
    ledger = []
    for model in stage["qualified_field_models"]:
        field = ROOT / stage["model_results"][model]["field"]["path"]
        if not field.exists():
            ledger.append({"path": str(field.relative_to(ROOT)), "status": "already_absent"})
            continue
        item = {"path": str(field.relative_to(ROOT)), "bytes": field.stat().st_size,
                "sha256_before_delete": sha256(field)}
        field.unlink()
        item["status"] = "deleted_after_prototype_visual_verification"
        ledger.append(item)
    return ledger


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 多语义单元全坐标响应图谱与逐样本大场清理（C985-C992） [{stamp}]

**目标与测试原理。** 本期不运行模型，只把 Phase2244 合格模型的每个“语言族×新词单元×检查点×功能角色”的有符号语义响应原型完整导出。每一行保留模型内全部物理激活坐标，checkpoint 0 是词嵌入，随后是全部 block 后 HiddenState 与 final norm；没有Top-K、PCA、余弦筛选或跨模型坐标对齐。

$$
\mathcal A_{{m,f,u,\ell,r}}=\left(P_{{m,f,u,\ell,r,1}},\ldots,P_{{m,f,u,\ell,r,d_m}}\right).
$$

**结果与文件。** 导出 `{json.dumps(result['exports'], ensure_ascii=False)}`。每个二进制均复核 dtype、shape、有限性和SHA256；客户端 catalog 通过标准JSON重新解析。跨模型图只显示同族距离、最近错族距离和边距，不伪装成物理坐标图。

**清理。** 清理账为 `{json.dumps(result['cleanup'], ensure_ascii=False)}`，共释放 `{result['bytes_deleted']}` 字节。删除对象仅是已被完整单元原型和行为/索引账替代、且不直接展示的逐样本大场；脚本、材料、行为结果、全坐标原型、索引、拓扑、矩阵和哈希全部保留。

**理论进展、问题与结论。** 可视化允许逐层、逐角色、逐坐标观察四个新单元的共同结构和残差，但平均四个“语言×表面”真假差分会丢失单条样本波动，因此原型不能冒充原始样本全集。激活坐标不是模型权重参数；4B与14B坐标编号不可比较；颜色缩放不构成统计证据。工程验证 `{result['all_checks_passed']}`。本大阶段到此获得多单元行为、全坐标模型内检索、跨规模角色拓扑检索和可视化四本账；下一阶段只有在引入更自然材料或第三个行为合格模型时才提供新的独立分母。

**相关文件。** 脚本 `tests/glm5/phase2245_c985_c992_multiunit_visual_atlas_cleanup.py`；结果 `{OUT.relative_to(ROOT)}`；图谱目录 `frontend/public/vis_data/research_kernel`；catalog `frontend/public/research_data/current/language_encoding_catalog.json`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    stage = json.loads((SOURCE / "analysis/final.json").read_text(encoding="utf-8"))
    exports = [export_model(model, stage) for model in stage["qualified_field_models"]]
    if stage["cross_model_retrieval"].get("status") == "closed":
        exports.append(export_cross_model(stage))
    added = update_catalog(exports)
    checks_before_cleanup = {
        "source_complete": stage["all_checks_passed"],
        "exports_exist": all((VIS / item["json"]).exists() and (VIS / item["binary"]).exists() for item in exports),
        "exports_verified": all(item["verified"] for item in exports),
        "catalog_entries_present": all(item["id"] in {x.get("id") for x in json.loads(CATALOG.read_text(encoding="utf-8"))["datasets"]}
                                       for item in exports),
    }
    if not all(checks_before_cleanup.values()):
        raise RuntimeError(("visual_verification_failed", checks_before_cleanup))
    cleanup = cleanup_sample_fields(stage)
    bytes_deleted = sum(item.get("bytes", 0) for item in cleanup if item["status"].startswith("deleted"))
    checks = {**checks_before_cleanup,
              "raw_sample_fields_absent": all(not (ROOT / item["path"]).exists() for item in cleanup),
              "catalog_valid_json": bool(json.loads(CATALOG.read_text(encoding="utf-8"))),
              "only_owned_raw_fields_cleaned": all("phase2244_c971_c984" in item["path"] for item in cleanup)}
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()), "exports": exports,
        "catalog_entries_added": added, "cleanup": cleanup, "bytes_deleted": bytes_deleted,
        "strict_conclusion": "The client exposes every activation coordinate of multi-unit response prototypes; visualization is not causal closure.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
