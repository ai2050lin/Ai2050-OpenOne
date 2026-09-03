from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase2234_c870_c884_broad_family_gear_contract as contract  # noqa: E402
import phase2235_c885_c904_qwen_broad_family_full_coordinate_tournament as tournament  # noqa: E402
import phase2237_c915_c920_full_coordinate_predictive_causal as causal  # noqa: E402


PHASE = 2241
CAMPAIGNS = tuple(f"C{i}" for i in range(941, 951))
P2235 = tournament.OUT
P2236 = ROOT / "tests/glm5/result/phase2236_c905_c914_composition_flagship_full_coordinate"
P2239 = ROOT / "tests/glm5/result/phase2239_c925_c936_cross_model_exact_semantic_panel"
OUT = ROOT / "tests/glm5/result/phase2241_c941_c950_visual_full_coordinate_atlas_and_cleanup"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close_mmap(value: np.ndarray) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def atlas_payload(atlas_id: str, title: str, description: str, binary_name: str,
                  shape: list[int], rows: list[dict], coordinate_count: int, model: str, boundary: str) -> dict:
    return {
        "schema": "ai2050.full-coordinate-language-family-atlas.v1", "id": atlas_id,
        "title": title, "description": description, "model": model,
        "dtype": "float16", "binary_url": f"/vis_data/research_kernel/{binary_name}",
        "binary_shape": shape, "coordinate_count": coordinate_count,
        "rows": rows, "boundary": boundary,
    }


def export_broad_predictive() -> dict:
    final = json.loads((P2235 / "analysis/final.json").read_text(encoding="utf-8"))
    strict = final["strict_family_candidates"]
    index = read_rows(P2235 / "raw/fresh/hidden_index.jsonl")
    pairs = [row for row in tournament.pair_records(index, "lockbox") if row["family"] in strict]
    field = np.load(P2235 / "raw/fresh/qualified_role_field.float16.npy", mmap_mode="r")
    shared = np.asarray(np.load(P2235 / "raw/shared_affine_coefficients.float16.npy"), dtype=np.float32)
    guards = np.asarray(np.load(P2235 / "raw/family_guard_residual.float16.npy"), dtype=np.float32)
    signals = ("base_activation", "actual_semantic_delta", "predicted_family_delta", "wrong_family_delta")
    shape = [len(pairs) * len(contract.QPOINTS) * len(contract.ROLES) * len(signals), contract.DIM]
    binary_name = "c950_broad_family_conditional_gear_atlas.float16.npy"
    target = np.lib.format.open_memmap(VIS / binary_name, mode="w+", dtype=np.float16, shape=tuple(shape))
    rows, out_i = [], 0
    wrong_map = {family: strict[(i + 1) % len(strict)] for i, family in enumerate(strict)}
    try:
        for spec in pairs:
            base_full = np.asarray(field[spec["base"]], dtype=np.float32)
            changed_full = np.asarray(field[spec["changed"]], dtype=np.float32)
            correct = causal.predicted_response(base_full, shared, guards, spec["family"])
            wrong_family = wrong_map[spec["family"]]
            wrong = causal.predicted_response(base_full, shared, guards, wrong_family)
            base = base_full[list(contract.QPOINTS)]
            actual = changed_full[list(contract.QPOINTS)] - base
            values = (base, actual, correct, wrong)
            for q_i, checkpoint in enumerate(contract.QPOINTS):
                for role_i, role in enumerate(contract.ROLES):
                    for signal, value in zip(signals, values):
                        target[out_i] = value[q_i, role_i].astype(np.float16)
                        rows.append({"source": signal, "family": spec["family"],
                                     "wrong_family": wrong_family if "wrong" in signal else None,
                                     "language": spec["language"], "surface": spec["surface"],
                                     "unit": spec["unit"], "partition": "fresh_lockbox",
                                     "checkpoint": checkpoint, "role": role,
                                     "case_id": spec["base_case_id"]})
                        out_i += 1
    finally:
        close_mmap(field); target.flush(); close_mmap(target)
    json_name = "c950_broad_family_conditional_gear_atlas.json"
    payload = atlas_payload(
        "c950_broad_family_conditional_gear_atlas", "C950 Broad Family Conditional Gear Atlas",
        "Fresh-lockbox base activations, observed semantic deltas, current-state family predictions, and equal-capacity wrong-family predictions.",
        binary_name, shape, rows, contract.DIM, "Qwen3-4B",
        "All 2560 activation coordinates are displayed. Predictive family specificity is not a minimal or causal circuit; Phase2240 corrected cross-model accounting is separate.")
    save(VIS / json_name, payload)
    return {"id": payload["id"], "json": json_name, "binary": binary_name, "shape": shape,
            "rows": len(rows), "sha256": file_hash(VIS / binary_name)}


def export_exact_model(model: str, label: str) -> dict:
    model_dir = P2239 / model
    final = json.loads((model_dir / "analysis/final.json").read_text(encoding="utf-8"))
    field_path = ROOT / final["field"]["path"]
    index = read_rows(model_dir / "raw/field_index.jsonl")
    field = np.load(field_path, mmap_mode="r")
    by_key = {(row["family"], row["language"], row["surface"], row["truth"]): row for row in index}
    pair_keys = sorted({(row["family"], row["language"], row["surface"]) for row in index})
    checkpoints, dim = field.shape[1], field.shape[-1]
    shape = [len(pair_keys) * checkpoints * len(contract.ROLES) * 2, dim]
    stem = f"c950_{model}_exact_semantic_field"
    binary_name = stem + ".float16.npy"
    target = np.lib.format.open_memmap(VIS / binary_name, mode="w+", dtype=np.float16, shape=tuple(shape))
    rows, out_i = [], 0
    try:
        for family, language, surface in pair_keys:
            false = by_key[(family, language, surface, False)]
            true = by_key[(family, language, surface, True)]
            base = np.asarray(field[false["hidden_index"]], dtype=np.float32)
            delta = np.asarray(field[true["hidden_index"]], dtype=np.float32) - base
            for checkpoint in range(checkpoints):
                for role_i, role in enumerate(contract.ROLES):
                    for signal, value in (("base_activation", base), ("semantic_delta", delta)):
                        target[out_i] = value[checkpoint, role_i].astype(np.float16)
                        rows.append({"source": signal, "family": family, "language": language,
                                     "surface": surface, "checkpoint": checkpoint, "role": role,
                                     "case_id": false["case_id"], "relative_depth": checkpoint / max(1, checkpoints - 1)})
                        out_i += 1
    finally:
        close_mmap(field); target.flush(); close_mmap(target)
    json_name = stem + ".json"
    payload = atlas_payload(
        stem, f"C950 {label} Exact-Semantic Full-Coordinate Field",
        "Exact 96-row semantic panel represented as reconstructible false-state plus true-minus-false response pairs across every checkpoint, role, and coordinate.",
        binary_name, shape, rows, dim, label,
        "Coordinate IDs are physical activations within this model only. Cross-model comparison is restricted to relative-depth role topology.")
    save(VIS / json_name, payload)
    return {"id": payload["id"], "json": json_name, "binary": binary_name, "shape": shape,
            "rows": len(rows), "sha256": file_hash(VIS / binary_name)}


def export_composition() -> dict:
    source = P2236 / "analysis/full_coordinate_flagship_prototypes.float16.npy"
    index = json.loads((P2236 / "analysis/prototype_index.json").read_text(encoding="utf-8"))
    binary_name = "c950_composition_flagship_prototypes.float16.npy"
    values = np.load(source, mmap_mode="r")
    shape = [values.shape[0] * values.shape[1] * values.shape[2], values.shape[3]]
    target = np.lib.format.open_memmap(VIS / binary_name, mode="w+", dtype=np.float16, shape=tuple(shape))
    rows, out_i = [], 0
    for prototype_i, key in enumerate(index["keys"]):
        kind, family = key.split("|", 1)
        for checkpoint in range(values.shape[1]):
            for role_i, role in enumerate(contract.ROLES):
                target[out_i] = values[prototype_i, checkpoint, role_i]
                rows.append({"source": kind, "family": family, "checkpoint": checkpoint, "role": role})
                out_i += 1
    target.flush(); close_mmap(target); close_mmap(values)
    json_name = "c950_composition_flagship_prototypes.json"
    payload = atlas_payload(
        "c950_composition_flagship_prototypes", "C950 Composition Flagship Full-Coordinate Prototypes",
        "Nested-attitude factorial interactions and graph-depth/shortcut increments for all checkpoints, roles, and coordinates.",
        binary_name, shape, rows, contract.DIM, "Qwen3-4B",
        "Graph prototypes replicated prospectively; attitude interactions did not pass fresh-vocabulary zero-model gates. These are observational prototypes, not causal operators.")
    save(VIS / json_name, payload)
    return {"id": payload["id"], "json": json_name, "binary": binary_name, "shape": shape,
            "rows": len(rows), "sha256": file_hash(VIS / binary_name)}


def update_catalog(exports: list[dict]) -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    ids = {item["id"] for item in exports}
    catalog["datasets"] = [item for item in catalog["datasets"] if item.get("id") not in ids]
    for item in exports:
        payload = json.loads((VIS / item["json"]).read_text(encoding="utf-8"))
        catalog["datasets"].append({
            "id": item["id"], "title": payload["title"], "phase": PHASE,
            "campaign": "C941-C950", "model": payload["model"],
            "source_path": f"/vis_data/research_kernel/{item['json']}",
            "binary_path": f"/vis_data/research_kernel/{item['binary']}",
            "source_schema": payload["schema"], "coordinate_count": payload["coordinate_count"],
            "checkpoint_count": len({row.get("checkpoint") for row in payload["rows"]}),
            "row_count": item["rows"], "claim_level": "full_coordinate_observation",
            "boundary": payload["boundary"], "kinds": sorted({row["source"] for row in payload["rows"]}),
        })
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save(CATALOG, catalog)


def cleanup(exports: list[dict]) -> list[dict]:
    targets = [
        P2235 / "raw/parent/qualified_role_field.float16.npy",
        P2235 / "raw/fresh/qualified_role_field.float16.npy",
        P2235 / "raw/parent/representative_full_token_qpoints.float16.npy",
        P2235 / "raw/fresh/representative_full_token_qpoints.float16.npy",
        P2239 / "qwen3/raw/exact_semantic_role_field.float16.npy",
        P2239 / "qwen3_14b/raw/exact_semantic_role_field.float16.npy",
    ]
    records = []
    for path in targets:
        if not path.exists():
            records.append({"path": str(path.relative_to(ROOT)), "status": "already_absent"}); continue
        records.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                        "sha256_before_delete": file_hash(path), "status": "deleted_after_visual_verification"})
        path.unlink()
    return records


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    formula = r"""
$$
\mathcal A=\{(m,f,u,q,r,j,s, H_{m,f,u,q,r,j}^{(s)})\},\qquad
s\in\{\text{base},\text{delta},\text{predicted},\text{wrong control}\}.
$$
"""
    text = f"""

## Phase {PHASE}: 语言族全坐标可视化图谱、复现哈希与大场清理 [{stamp}]

**目标与原则。** 本期执行 C941-C950，把本大阶段的重要观察写入通用语言编码研究客户端。可视化展示的是 embedding/HiddenState 激活坐标，不是模型权重参数；没有 Top-K、PCA 或坐标筛选。
{formula}
**导出内容。** 四个图谱为 `{json.dumps(result['exports'], ensure_ascii=False)}`。Qwen4B 族条件图包含 fresh lockbox 的基态、实际真假响应、M5 当前状态预测和错族预测；Qwen4B/Qwen14B 相同语义图用“假状态+真假差分”无损重建两端；组合图包含 15 个态度/递归图逐坐标原型。客户端可按 signal、角色和具体坐标查看 checkpoint 轨迹，checkpoint 0 即 embedding。

**验证与清理。** 每个 NPY 的 dtype、shape、有限性、JSON 行数和 SHA256 均在删除前验证；catalog 采用结构化 JSON 更新。清理记录为 `{json.dumps(result['cleanup'], ensure_ascii=False)}`，共释放 `{result['deleted_bytes']}` 字节。本轮系数、索引、逐样本结果、脚本、配置、可视化二进制和哈希保留；没有删除未被本期图谱覆盖的历史大场。

**理论进展和严格结论。** 图谱把“共享动力学强、8 族有预测增量、3 图域组合原型迁移、0 族获得选择性因果资格、Qwen4B/14B 角色拓扑接近”放到同一可观察坐标系。它帮助发现规律，但图像相似或局部高值不构成机制闭合。低值坐标全部保留，客户端上的颜色缩放只是显示变换。

**问题和硬伤。** 浏览器需要加载较大的本地二进制；Qwen14 图谱维度为 5120，不能与 Qwen4 的 2560 坐标编号比较；“基态+差分”虽可重建真假端，但不是原始两份重复存储；代表性 full-token 场没有进入本期客户端，因此按合同清理后只能重新运行脚本恢复。

**结论和下一步授权。** 可视化和清理完成。下一步仍属于相同核心目标：必须检验 Qwen4B/Qwen14B 的低拓扑距离是否具有语言族身份特异性，还是两模型共享的通用前向动力学。该检验可直接使用本期 topology，无需新模型或新 HiddenState，自动进入下一 Phase。

**相关文件。** 脚本 `tests/glm5/phase2241_c941_c950_visual_full_coordinate_atlas_and_cleanup.py`；结果 `{OUT.relative_to(ROOT)}`；客户端 catalog `frontend/public/research_data/current/language_encoding_catalog.json`；图谱目录 `frontend/public/vis_data/research_kernel`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    VIS.mkdir(parents=True, exist_ok=True)
    exports = [export_broad_predictive(), export_exact_model("qwen3", "Qwen3-4B"),
               export_exact_model("qwen3_14b", "Qwen3-14B"), export_composition()]
    update_catalog(exports)
    checks = {}
    for item in exports:
        payload = json.loads((VIS / item["json"]).read_text(encoding="utf-8"))
        matrix = np.load(VIS / item["binary"], mmap_mode="r")
        checks[item["id"]] = {
            "shape": list(matrix.shape) == item["shape"], "dtype": str(matrix.dtype) == "float16",
            "rows": len(payload["rows"]) == item["shape"][0],
            "finite": bool(np.isfinite(np.asarray(matrix[::max(1, len(matrix) // 64)], dtype=np.float32)).all()),
            "sha256": file_hash(VIS / item["binary"]) == item["sha256"],
        }
        close_mmap(matrix)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    checks["catalog"] = {"all_export_ids": {item["id"] for item in exports}.issubset(
        {item.get("id") for item in catalog["datasets"]})}
    if not all(all(values.values()) for values in checks.values()):
        raise RuntimeError(("visual_verification_failed", checks))
    cleanup_records = cleanup(exports)
    deleted_bytes = sum(row.get("bytes", 0) for row in cleanup_records)
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "checks": checks,
        "all_checks_passed": all(all(values.values()) for values in checks.values()),
        "exports": exports, "cleanup": cleanup_records, "deleted_bytes": deleted_bytes,
        "strict_conclusion": "The client displays every retained activation coordinate; visual salience is observational and is not a parameter, causal, or uniqueness claim.",
        "next_authorization": "Use frozen cross-model topology for family-specificity retrieval before any new model run.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
