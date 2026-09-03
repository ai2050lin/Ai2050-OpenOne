from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase2234_c870_c884_broad_family_gear_contract as contract  # noqa: E402


PHASE = 2239
CAMPAIGNS = tuple(f"C{i}" for i in range(925, 937))
SOURCE = ROOT / "tests/glm5/result/phase2234_c870_c884_broad_family_conditional_gear_contract/material/fresh_qwen_compiled.jsonl"
OUT = ROOT / "tests/glm5/result/phase2239_c925_c936_cross_model_exact_semantic_panel"
MATERIAL = OUT / "material/exact_semantic_panel.jsonl"
WORKER = TESTS / "phase2239_c925_c936_cross_model_full_coordinate_worker.py"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MODELS = ("qwen3", "qwen3_14b", "glm4", "deepseek7b")
ROLE_NAMES = contract.ROLES


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_material() -> list[dict]:
    rows = [row for row in read_rows(SOURCE)
            if row["panel"] == "broad_family" and row["partition"] == "confirmation" and row["unit"] == 0]
    rows.sort(key=lambda row: (row["family"], row["language"], row["surface"], row["truth"]))
    if len(rows) != len(contract.FAMILIES) * len(contract.LANGUAGES) * len(contract.SURFACES) * 2:
        raise RuntimeError(("exact_panel_size", len(rows)))
    write_rows(MATERIAL, rows)
    return rows


def run_workers() -> dict:
    status = {}
    for model in MODELS:
        model_out = OUT / model
        final_path = model_out / "analysis/final.json"
        previous = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else None
        if previous is None or previous.get("status") == "worker_error":
            command = [sys.executable, str(WORKER), "--model", model,
                       "--material", str(MATERIAL), "--output", str(model_out)]
            print(f"[supervisor] starting {model}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            status[model] = {"returncode": completed.returncode}
        else:
            status[model] = {"returncode": 0, "resumed": True}
        if final_path.exists():
            status[model]["result"] = json.loads(final_path.read_text(encoding="utf-8"))
        else:
            status[model]["result"] = {"status": "missing_worker_result", "all_checks_passed": False}
    return status


def response_topology(model: str, result: dict) -> dict:
    field_path = ROOT / result["field"]["path"]
    index_path = OUT / model / "raw/field_index.jsonl"
    index = read_rows(index_path)
    field = np.load(field_path, mmap_mode="r")
    by_key = {(row["family"], row["language"], row["surface"], row["truth"]): row for row in index}
    family_curves = {}
    try:
        for family in contract.FAMILIES:
            deltas = []
            for language in contract.LANGUAGES:
                for surface in contract.SURFACES:
                    false = by_key[(family, language, surface, False)]
                    true = by_key[(family, language, surface, True)]
                    deltas.append(np.asarray(field[true["hidden_index"]], dtype=np.float32)
                                  - np.asarray(field[false["hidden_index"]], dtype=np.float32))
            stacked = np.stack(deltas)
            rms = np.sqrt(np.mean(stacked * stacked, axis=(0, 3), dtype=np.float64))
            role_share = rms / np.maximum(np.sum(rms, axis=1, keepdims=True), 1e-12)
            family_curves[family] = {
                "relative_depth": np.linspace(0.0, 1.0, role_share.shape[0]).tolist(),
                "role_energy_share": role_share.tolist(),
                "peak_relative_depth_by_role": {
                    role: float(np.argmax(role_share[:, role_i]) / max(1, role_share.shape[0] - 1))
                    for role_i, role in enumerate(ROLE_NAMES)},
            }
    finally:
        mmap = getattr(field, "_mmap", None)
        if mmap is not None:
            mmap.close()
    topology = {"model": model, "rows": len(index), "field_shape": result["field"]["shape"],
                "roles": list(ROLE_NAMES), "families": family_curves}
    save(OUT / model / "analysis/response_topology.json", topology)
    return topology


def interpolate_curve(topology: dict, family: str, grid: np.ndarray) -> np.ndarray:
    source_x = np.asarray(topology["families"][family]["relative_depth"], dtype=np.float64)
    source_y = np.asarray(topology["families"][family]["role_energy_share"], dtype=np.float64)
    return np.stack([np.interp(grid, source_x, source_y[:, role_i])
                     for role_i in range(len(ROLE_NAMES))], axis=1)


def compare_topologies(topologies: dict[str, dict]) -> list[dict]:
    rows = []
    grid = np.linspace(0.0, 1.0, 21)
    names = sorted(topologies)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for family in contract.FAMILIES:
                a = interpolate_curve(topologies[left], family, grid)
                b = interpolate_curve(topologies[right], family, grid)
                rows.append({
                    "left": left, "right": right, "family": family,
                    "relative_depth_grid": grid.tolist(),
                    "mean_absolute_role_share_distance": float(np.mean(np.abs(a - b))),
                    "max_absolute_role_share_distance": float(np.max(np.abs(a - b))),
                })
    return rows


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    behavior = {model: {"status": value["status"], "candidate": value.get("candidate_accuracy"),
                        "generation": value.get("generation_accuracy"),
                        "hidden": value.get("field", {}).get("ran", False)}
                for model, value in result["model_results"].items()}
    formula = r"""
$$
E_{m,f,q,r}=\sqrt{\frac{1}{|\mathcal P_f|d_m}
\sum_{u\in\mathcal P_f}\sum_{j=1}^{d_m}
\left(H^{(m)}_{1,u,q,r,j}-H^{(m)}_{0,u,q,r,j}\right)^2},\qquad
\rho_{m,f,q,r}=\frac{E_{m,f,q,r}}{\sum_{r'}E_{m,f,q,r'}}.
$$
$$
D(m,n;f)=\frac{1}{21|R|}\sum_{t\in\{0,.05,\ldots,1\}}
\sum_r\left|\widetilde\rho_{m,f,r}(t)-\widetilde\rho_{n,f,r}(t)\right|.
$$
"""
    text = f"""

## Phase {PHASE}: 四模型相同语义分母的全坐标语言族响应拓扑 [{stamp}]

**合同与证据范围。** 本期执行 C925-C936。冻结 96 条完全相同的语义记录：12 语言族、中英文、直接/释义、真/假各一条。Qwen3-4B、Qwen3-14B、GLM4、DeepSeek-7B 按顺序单独加载和释放，各模型自行分词；候选和自由生成整体均达到 0.75 才采集 embedding、全部 block 后状态、final norm、六角色和全部坐标。

**测试原理与用例。** 用例包括类型、整体部分、时间、因果、施受事、否定、共指、嵌套态度、属性、比较、翻译路线和量词。跨模型不比较物理坐标编号；在各模型内部先对同一真假语义对做全坐标响应，再把每层六角色的 RMS 能量归一化为角色份额，以相对层深插值到固定 21 点后比较拓扑。
{formula}
**行为与全场结果。** `{json.dumps(behavior, ensure_ascii=False)}`。行为未合格模型的 HiddenState 分支记为 NA，不解释为内部机制阴性。获得全场的模型为 `{result['qualified_models']}`；相同语义跨模型拓扑比较条数为 `{len(result['topology_comparisons'])}`，平均绝对角色份额距离汇总为 `{json.dumps(result['comparison_summary'], ensure_ascii=False)}`。

**理论进展与严格分析。** 该面板只检验不同模型是否在相对深度和功能角色上出现近似响应组织，不寻找相同坐标编号。较小距离可称功能拓扑相似，不能称同一内部数学结构；较大距离可能来自层数、维度、训练、分词和行为策略差异。全维场已保存，RMS 仅用于跨维数摘要，不能替代模型内逐坐标图谱。

**问题、硬伤和瓶颈。** 96 条是受控语义面板，每族仅一个词汇单元；总体行为门可能掩盖局部族失败；Qwen14 使用磁盘卸载、GLM/DS 可能使用量化，数值身份不同；角色跨度的 BPE 兜底仍可能包含邻接字符；跨模型归一化丢失绝对能量；这不是参数或因果比较。

**结论和下一步。** 本期只保留模型内完整物理坐标和跨模型功能拓扑两本账。下一 Phase 导出可视化客户端可直接读取的 embedding/HiddenState 参数级图谱，并在哈希和形状复核后删除未展示的大型临时场。若跨模型只有一个合格模型，则功能比较正式记 NA，但 Qwen 主模型图谱仍继续。

**相关文件。** 主脚本 `tests/glm5/phase2239_c925_c936_cross_model_exact_semantic_panel.py`；worker `tests/glm5/phase2239_c925_c936_cross_model_full_coordinate_worker.py`；结果 `{OUT.relative_to(ROOT)}`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    rows = prepare_material()
    save(OUT / "protocol/preregistration.json", {
        "timestamp": datetime.now().astimezone().isoformat(), "phase": PHASE,
        "semantic_rows": len(rows), "families": list(contract.FAMILIES),
        "languages": list(contract.LANGUAGES), "surfaces": list(contract.SURFACES),
        "models_in_order": list(MODELS), "behavior_gate": contract.BEHAVIOR_GATE,
        "capture": "all 96 rows, all checkpoints, six roles, every model coordinate iff aggregate dual behavior passes",
        "comparison": "relative-depth role-energy topology only; no physical coordinate alignment",
    })
    worker_status = run_workers()
    model_results = {name: value["result"] for name, value in worker_status.items()}
    topologies = {}
    for name, model_result in model_results.items():
        if model_result.get("all_checks_passed") and model_result.get("behavior_qualified"):
            topologies[name] = response_topology(name, model_result)
    comparisons = compare_topologies(topologies)
    write_rows(OUT / "analysis/topology_comparisons.jsonl", comparisons)
    summary = {}
    for pair in sorted({(row["left"], row["right"]) for row in comparisons}):
        subset = [row for row in comparisons if (row["left"], row["right"]) == pair]
        summary[f"{pair[0]}|{pair[1]}"] = {
            "families": len(subset),
            "mean_absolute_role_share_distance": float(np.mean([row["mean_absolute_role_share_distance"] for row in subset])),
            "max_absolute_role_share_distance": float(np.max([row["max_absolute_role_share_distance"] for row in subset])),
        }
    checks = {
        "exact_96_semantic_rows": len(rows) == 96,
        "all_workers_returned_results": all("result" in value for value in worker_status.values()),
        "worker_accounting_complete": all(result.get("status") in ("closed", "behavior_unqualified", "worker_error")
                                          for result in model_results.values()),
        "qualified_fields_complete": all(topology["rows"] == 96 for topology in topologies.values()),
        "comparison_finite": all(np.isfinite(row["mean_absolute_role_share_distance"]) for row in comparisons),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()), "worker_status": worker_status,
        "model_results": model_results, "qualified_models": sorted(topologies),
        "topology_comparisons": comparisons, "comparison_summary": summary,
        "strict_conclusion": "Cross-model evidence is restricted to exact-semantic relative-depth role topology; coordinate identity and causal isomorphism are not tested.",
        "next_authorization": "Export complete coordinate atlases for the visualization client, hash outputs, and clean only superseded raw fields.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
