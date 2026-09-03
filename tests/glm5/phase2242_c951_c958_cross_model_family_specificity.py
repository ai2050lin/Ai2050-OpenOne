from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase2234_c870_c884_broad_family_gear_contract as contract  # noqa: E402


PHASE = 2242
CAMPAIGNS = tuple(f"C{i}" for i in range(951, 959))
SOURCE = ROOT / "tests/glm5/result/phase2239_c925_c936_cross_model_exact_semantic_panel"
OUT = ROOT / "tests/glm5/result/phase2242_c951_c958_cross_model_family_specificity"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
FAMILIES = contract.FAMILIES


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def curves(path: Path) -> dict[str, np.ndarray]:
    topology = json.loads(path.read_text(encoding="utf-8"))
    grid = np.linspace(0.0, 1.0, 21)
    output = {}
    for family in FAMILIES:
        item = topology["families"][family]
        x = np.asarray(item["relative_depth"], dtype=np.float64)
        y = np.asarray(item["role_energy_share"], dtype=np.float64)
        output[family] = np.stack([np.interp(grid, x, y[:, role_i])
                                   for role_i in range(y.shape[1])], axis=1)
    return output


def centered(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean = np.mean(np.stack([values[family] for family in FAMILIES]), axis=0)
    return {family: values[family] - mean for family in FAMILIES}


def distance_matrix(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([[np.mean(np.abs(left[a] - right[b])) for b in FAMILIES] for a in FAMILIES], dtype=np.float64)


def retrieval(matrix: np.ndarray) -> dict:
    forward = np.argmin(matrix, axis=1)
    reverse = np.argmin(matrix, axis=0)
    diagonal = np.diag(matrix)
    nearest_wrong_forward = np.asarray([np.min(np.delete(matrix[i], i)) for i in range(len(FAMILIES))])
    nearest_wrong_reverse = np.asarray([np.min(np.delete(matrix[:, i], i)) for i in range(len(FAMILIES))])
    offdiag = matrix[~np.eye(len(FAMILIES), dtype=bool)]
    return {
        "forward_predictions": {family: FAMILIES[int(forward[i])] for i, family in enumerate(FAMILIES)},
        "reverse_predictions": {family: FAMILIES[int(reverse[i])] for i, family in enumerate(FAMILIES)},
        "forward_accuracy": float(np.mean(forward == np.arange(len(FAMILIES)))),
        "reverse_accuracy": float(np.mean(reverse == np.arange(len(FAMILIES)))),
        "diagonal_mean_distance": float(np.mean(diagonal)),
        "off_diagonal_mean_distance": float(np.mean(offdiag)),
        "diagonal_to_off_diagonal_ratio": float(np.mean(diagonal) / max(np.mean(offdiag), 1e-12)),
        "forward_margins": {family: float(nearest_wrong_forward[i] - diagonal[i]) for i, family in enumerate(FAMILIES)},
        "reverse_margins": {family: float(nearest_wrong_reverse[i] - diagonal[i]) for i, family in enumerate(FAMILIES)},
        "positive_forward_margins": int(np.sum(nearest_wrong_forward > diagonal)),
        "positive_reverse_margins": int(np.sum(nearest_wrong_reverse > diagonal)),
    }


def export_visual(raw: np.ndarray, residual: np.ndarray) -> dict:
    matrix = np.concatenate([raw, residual], axis=0).astype(np.float16)
    binary_name = "c958_cross_model_family_specificity.float16.npy"
    np.save(VIS / binary_name, matrix)
    rows = []
    for kind in ("raw_role_topology_distance", "model_centered_family_residual_distance"):
        for family in FAMILIES:
            rows.append({"source": kind, "family": family, "role": "all_roles",
                         "checkpoint": 0, "column_families": list(FAMILIES)})
    json_name = "c958_cross_model_family_specificity.json"
    payload = {
        "schema": "ai2050.cross-model-family-specificity.v1", "id": "c958_cross_model_family_specificity",
        "title": "C958 Qwen4B/Qwen14B Family Specificity Matrix",
        "description": "Exhaustive 12x12 L1 distance matrices for raw relative-depth role topology and model-centered family residual topology.",
        "dtype": "float16", "binary_url": f"/vis_data/research_kernel/{binary_name}",
        "binary_shape": list(matrix.shape), "coordinate_count": len(FAMILIES), "rows": rows,
        "coordinate_labels": list(FAMILIES),
        "boundary": "Columns are language families, not physical activation coordinates. This control tests family identity after removing generic model topology.",
    }
    save(VIS / json_name, payload)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog["datasets"] = [item for item in catalog["datasets"] if item.get("id") != payload["id"]]
    catalog["datasets"].append({
        "id": payload["id"], "title": payload["title"], "phase": PHASE,
        "campaign": "C951-C958", "model": "Qwen3-4B / Qwen3-14B",
        "source_path": f"/vis_data/research_kernel/{json_name}",
        "binary_path": f"/vis_data/research_kernel/{binary_name}",
        "source_schema": payload["schema"], "coordinate_count": len(FAMILIES),
        "checkpoint_count": 1, "row_count": len(rows), "claim_level": "cross_model_control",
        "boundary": payload["boundary"], "kinds": sorted({row["source"] for row in rows}),
    })
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save(CATALOG, catalog)
    return {"json": json_name, "binary": binary_name, "shape": list(matrix.shape)}


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    formula = r"""
$$
\bar\rho_m(t,r)=\frac{1}{|F|}\sum_f\rho_{m,f}(t,r),\qquad
\rho'_{m,f}=\rho_{m,f}-\bar\rho_m.
$$
$$
D'_{fg}=\frac{1}{21|R|}\sum_{t,r}|\rho'_{4B,f}(t,r)-\rho'_{14B,g}(t,r)|,
\qquad \widehat g(f)=\arg\min_g D'_{fg}.
$$
"""
    text = f"""

## Phase {PHASE}: 跨模型角色拓扑的语言族身份特异性控制 [{stamp}]

**为什么自动继续。** Phase 2240 的 Qwen4B/Qwen14B 平均角色拓扑距离 0.0189 很小，但这可能只是共享前向动力学。C951-C958 使用已经冻结的 12 族拓扑穷举 12×12 配对，不加载模型、不筛坐标、不训练探针，检验同族是否真的比错族更近。

**算法。** 第一账直接比较原始相对层深角色份额；第二账在每个模型内部减去 12 族平均拓扑，只保留族相对残差。两个方向都做最近邻检索，并记录同族距离与最近错族距离之差。
{formula}
**结果。** 原始拓扑：`{json.dumps(result['raw_retrieval'], ensure_ascii=False)}`。模型中心化族残差：`{json.dumps(result['residual_retrieval'], ensure_ascii=False)}`。完整 12×12 距离矩阵写入结果和客户端热力图，未只报告命中族。

**理论进展与严格裁决。** 若原始距离低而中心化检索接近偶然水平，则 Phase 2240 的相似性主要属于通用前向/角色组织，不能称语言族同构；若中心化后仍能双向检索同族，则得到跨规模族特异拓扑候选。无论哪种结果，这仍是角色能量拓扑，不是单坐标齿轮、因果电路或参数同构。

**问题与硬伤。** 只有 12 族、每族一个跨模型语义单元；L1 距离等权处理六角色和 21 个相对深度点；减全族均值可能同时去掉真实共享语义结构；最近邻没有独立第三模型确认；GLM4/DS7B 因行为门未过无法参与内部比较。

**结论与下一阶段。** 本期完成对“低跨模型距离”的必要负控。下一大阶段虽然仍服务于破解语言族齿轮，但需要全新词汇、多单元自然材料和第三个行为合格模型来前瞻确认，已超出本轮旧数据大阶段；本轮在证据、可视化和清理完成后正式结束，不继续对同一 96 条数据反复挖掘。

**相关文件。** 脚本 `tests/glm5/phase2242_c951_c958_cross_model_family_specificity.py`；结果 `{OUT.relative_to(ROOT)}`；可视化 `frontend/public/vis_data/research_kernel/c958_cross_model_family_specificity.json`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol/preregistration.json", {
        "timestamp": datetime.now().astimezone().isoformat(), "phase": PHASE,
        "source": "Phase2240 recovered exact-semantic response topologies",
        "families": list(FAMILIES), "grid": 21, "distance": "mean absolute L1",
        "views": ["raw role-share topology", "within-model family-centered residual topology"],
        "readout": "bidirectional exhaustive nearest-family retrieval and same-vs-nearest-wrong margin",
        "status": "descriptive control; no post hoc pass threshold",
    })
    q4 = curves(SOURCE / "qwen3/analysis/response_topology.json")
    q14 = curves(SOURCE / "qwen3_14b/analysis/response_topology.json")
    raw_matrix = distance_matrix(q4, q14)
    residual_matrix = distance_matrix(centered(q4), centered(q14))
    raw_result = retrieval(raw_matrix)
    residual_result = retrieval(residual_matrix)
    np.save(OUT / "analysis/raw_distance_matrix.float64.npy", raw_matrix)
    np.save(OUT / "analysis/residual_distance_matrix.float64.npy", residual_matrix)
    visual = export_visual(raw_matrix, residual_matrix)
    checks = {
        "matrix_shape": raw_matrix.shape == residual_matrix.shape == (12, 12),
        "finite": bool(np.isfinite(raw_matrix).all() and np.isfinite(residual_matrix).all()),
        "all_families_accounted": len(raw_result["forward_predictions"]) == len(residual_result["reverse_predictions"]) == 12,
        "visual_shape": visual["shape"] == [24, 12],
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()), "raw_retrieval": raw_result,
        "residual_retrieval": residual_result, "visual": visual,
        "strict_conclusion": "Family-specific cross-model topology is judged from centered exhaustive retrieval, not the low raw same-family distance alone.",
        "next_authorization": "A new prospective stage requires multiple unseen lexical units and another behavior-qualified model; repeated mining of this 96-row panel is closed.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
