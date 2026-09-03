#!/usr/bin/env python3
"""Run the frozen multi-unit family topology campaign across four local models."""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase2234_c870_c884_broad_family_gear_contract as base  # noqa: E402
import phase2243_c959_c970_multiunit_cross_scale_contract as contract  # noqa: E402


PHASE = 2244
CAMPAIGNS = tuple(f"C{i}" for i in range(971, 985))
OUT = ROOT / "tests/glm5/result/phase2244_c971_c984_multiunit_cross_model_topology"
SOURCE = ROOT / "tests/glm5/result/phase2243_c959_c970_multiunit_cross_scale_contract/material/prospective_cases.jsonl"
WORKER = TESTS / "phase2239_c925_c936_cross_model_full_coordinate_worker.py"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MODELS = ("qwen3", "qwen3_14b", "glm4", "deepseek7b")
ROLES = base.ROLES


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def run_workers() -> dict[str, dict]:
    status = {}
    for model in MODELS:
        model_out = OUT / model
        final_path = model_out / "analysis/final.json"
        previous = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else None
        if previous is None or previous.get("status") == "worker_error":
            command = [sys.executable, str(WORKER), "--model", model,
                       "--material", str(SOURCE), "--output", str(model_out)]
            print(f"[phase2244] starting {model}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            status[model] = {"returncode": completed.returncode}
        else:
            status[model] = {"returncode": 0, "resumed": True}
        if final_path.exists():
            status[model]["result"] = json.loads(final_path.read_text(encoding="utf-8"))
        else:
            status[model]["result"] = {"status": "missing_worker_result", "all_checks_passed": False}
    return status


def behavior_by_family(model: str, material: list[dict]) -> dict[str, dict]:
    by_id = {row["case_id"]: row for row in material}
    ledgers = {}
    for kind in ("candidate", "generation"):
        path = OUT / model / "behavior" / f"{kind}.jsonl"
        grouped = defaultdict(list)
        if path.exists():
            for item in read_rows(path):
                grouped[by_id[item["case_id"]]["family"]].append(bool(item["correct"]))
        ledgers[kind] = {family: float(np.mean(grouped[family])) if grouped[family] else None
                         for family in contract.FAMILIES}
    result = {}
    for family in contract.FAMILIES:
        candidate = ledgers["candidate"][family]
        generation = ledgers["generation"][family]
        result[family] = {
            "candidate_accuracy": candidate, "generation_accuracy": generation,
            "qualified": candidate is not None and generation is not None
                         and candidate >= contract.FAMILY_BEHAVIOR_GATE
                         and generation >= contract.FAMILY_BEHAVIOR_GATE,
        }
    return result


def semantic_prototypes(model: str, result: dict, material: list[dict]) -> dict:
    field_path = ROOT / result["field"]["path"]
    index = read_rows(OUT / model / "raw/field_index.jsonl")
    field = np.load(field_path, mmap_mode="r")
    material_by_id = {row["case_id"]: row for row in material}
    field_by_key = {}
    for row in index:
        source = material_by_id[row["case_id"]]
        field_by_key[(source["family"], int(source["unit"]), source["language"], source["surface"], bool(source["truth"]))] = int(row["hidden_index"])
    shape = (len(contract.FAMILIES) * len(contract.UNITS), field.shape[1], field.shape[2], field.shape[3])
    proto_path = OUT / model / "analysis/unit_semantic_response_prototypes.float16.npy"
    prototypes = np.lib.format.open_memmap(proto_path, mode="w+", dtype=np.float16, shape=shape)
    topology = {}
    proto_index = []
    slot = 0
    try:
        for family in contract.FAMILIES:
            topology[family] = {}
            for unit in contract.UNITS:
                deltas = []
                for language in contract.LANGUAGES:
                    for surface in contract.SURFACES:
                        false_i = field_by_key[(family, unit, language, surface, False)]
                        true_i = field_by_key[(family, unit, language, surface, True)]
                        deltas.append(np.asarray(field[true_i], dtype=np.float32)
                                      - np.asarray(field[false_i], dtype=np.float32))
                stacked = np.stack(deltas)
                prototype = np.mean(stacked, axis=0, dtype=np.float64).astype(np.float32)
                prototypes[slot] = prototype.astype(np.float16)
                rms = np.sqrt(np.mean(stacked * stacked, axis=(0, 3), dtype=np.float64))
                share = rms / np.maximum(np.sum(rms, axis=1, keepdims=True), 1e-12)
                topology[family][str(unit)] = {
                    "relative_depth": np.linspace(0.0, 1.0, share.shape[0]).tolist(),
                    "role_energy_share": share.tolist(),
                }
                proto_index.append({"prototype_index": slot, "family": family, "unit": unit})
                slot += 1
    finally:
        prototypes.flush()
        mmap = getattr(prototypes, "_mmap", None)
        if mmap is not None:
            mmap.close()
        mmap = getattr(field, "_mmap", None)
        if mmap is not None:
            mmap.close()
    write_rows(OUT / model / "analysis/unit_semantic_response_prototype_index.jsonl", proto_index)
    output = {
        "model": model, "field_shape": list(field.shape), "prototype_shape": list(shape),
        "prototype_path": str(proto_path.relative_to(ROOT)), "roles": list(ROLES), "families": topology,
    }
    save(OUT / model / "analysis/unit_role_topology.json", output)
    return output


def normalized_euclidean_matrix(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    n = query.shape[1]
    q2 = np.sum(query * query, axis=1, dtype=np.float64)
    c2 = np.sum(candidates * candidates, axis=1, dtype=np.float64)
    cross = np.asarray(query @ candidates.T, dtype=np.float64)
    mse = np.maximum((q2[:, None] + c2[None, :] - 2.0 * cross) / n, 0.0)
    denom = (np.sqrt(q2 / n)[:, None] + np.sqrt(c2 / n)[None, :]) * 0.5 + 1e-12
    return np.sqrt(mse) / denom


def retrieval_summary(distance_blocks: list[tuple[int, np.ndarray]], families: list[str]) -> dict:
    rows = []
    for unit, distances in distance_blocks:
        for i, family in enumerate(families):
            order = np.argsort(distances[i])
            predicted = families[int(order[0])]
            correct_distance = float(distances[i, i])
            wrong_distance = float(min(distances[i, j] for j in range(len(families)) if j != i)) if len(families) > 1 else None
            rows.append({
                "unit": unit, "family": family, "predicted": predicted, "correct": predicted == family,
                "same_family_distance": correct_distance, "nearest_wrong_distance": wrong_distance,
                "margin": None if wrong_distance is None else wrong_distance - correct_distance,
            })
    margins = [row["margin"] for row in rows if row["margin"] is not None]
    return {
        "queries": len(rows), "accuracy": float(np.mean([row["correct"] for row in rows])) if rows else None,
        "median_margin": float(np.median(margins)) if margins else None,
        "positive_margin_fraction": float(np.mean([x > 0 for x in margins])) if margins else None,
        "rows": rows,
    }


def within_model_full_coordinate_retrieval(model: str, topology: dict, qualified_families: list[str]) -> dict:
    if len(qualified_families) < 2:
        return {"status": "NA_fewer_than_two_qualified_families"}
    index = read_rows(OUT / model / "analysis/unit_semantic_response_prototype_index.jsonl")
    by_key = {(row["family"], int(row["unit"])): int(row["prototype_index"]) for row in index}
    field = np.load(ROOT / topology["prototype_path"], mmap_mode="r")
    try:
        raw = np.stack([[np.asarray(field[by_key[(family, unit)]], dtype=np.float32).reshape(-1)
                         for unit in contract.UNITS] for family in qualified_families])
    finally:
        mmap = getattr(field, "_mmap", None)
        if mmap is not None:
            mmap.close()
    centered = raw - np.mean(raw, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    results = {}
    for name, values in (("raw", raw), ("unit_centered", centered)):
        blocks = []
        for unit in contract.UNITS:
            query = values[:, unit]
            candidates = np.mean(values[:, [u for u in contract.UNITS if u != unit]], axis=1, dtype=np.float64).astype(np.float32)
            blocks.append((unit, normalized_euclidean_matrix(query, candidates)))
        results[name] = retrieval_summary(blocks, qualified_families)
    results["status"] = "closed"
    results["families"] = qualified_families
    results["metric"] = "all-coordinate symmetric RMS-normalized Euclidean distance"
    return results


def interp_curve(topology: dict, family: str, unit: int, grid: np.ndarray) -> np.ndarray:
    row = topology["families"][family][str(unit)]
    x = np.asarray(row["relative_depth"], dtype=np.float64)
    y = np.asarray(row["role_energy_share"], dtype=np.float64)
    return np.stack([np.interp(grid, x, y[:, r]) for r in range(len(ROLES))], axis=1)


def cross_model_topology_retrieval(left: dict, right: dict, families: list[str]) -> dict:
    if len(families) < 2:
        return {"status": "NA_fewer_than_two_common_families"}
    grid = np.linspace(0.0, 1.0, 21)
    arrays = {}
    for name, topology in (("left", left), ("right", right)):
        values = np.stack([[interp_curve(topology, family, unit, grid) for unit in contract.UNITS]
                           for family in families])
        arrays[name] = {"raw": values,
                        "unit_centered": values - np.mean(values, axis=0, keepdims=True)}
    output = {"status": "closed", "families": families, "relative_depth_grid": grid.tolist()}
    for kind in ("raw", "unit_centered"):
        directional = {}
        for direction, source_name, target_name in (("forward", "left", "right"), ("reverse", "right", "left")):
            source = arrays[source_name][kind]
            target = arrays[target_name][kind]
            blocks = []
            for unit in contract.UNITS:
                query = source[:, unit].reshape(len(families), -1)
                candidates = np.mean(target[:, [u for u in contract.UNITS if u != unit]], axis=1).reshape(len(families), -1)
                distances = np.mean(np.abs(query[:, None, :] - candidates[None, :, :]), axis=2)
                blocks.append((unit, distances))
            directional[direction] = retrieval_summary(blocks, families)
        output[kind] = directional
    return output


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    behavior = {m: {"status": r["status"], "candidate": r.get("candidate_accuracy"),
                    "generation": r.get("generation_accuracy"), "field": r.get("field", {}).get("ran", False),
                    "qualified_families": result["qualified_families"].get(m, [])}
                for m, r in result["model_results"].items()}
    cross = result["cross_model_retrieval"]
    within = {m: {k: v.get(k, {}).get("accuracy") for k in ("raw", "unit_centered")}
              for m, v in result["within_model_retrieval"].items() if v.get("status") == "closed"}
    text = rf"""

## Phase {PHASE}: 四模型多语义单元全坐标与跨规模族拓扑确认（C971-C984） [{stamp}]

**测试原理和执行身份。** 本期严格使用 Phase2243 冻结的384行、四套全新词汇材料，依次单独加载 Qwen3-4B、Qwen3-14B、GLM4 和 DS7B。候选选择与自由生成总体双门均过0.75才采集 embedding、全部 block 后状态、final norm、六角色和全部物理激活坐标；族级机制分析另要求该族双行为均过0.75。失败模型内部结果记 NA，未阻断后续模型。

模型内使用每族每单元的四个“语言×表面”真假响应形成全坐标签名，不筛坐标：

$$
P_{{m,f,u}}=\frac14\sum_{{\lambda,s}}\left(H^1_{{m,f,u,\lambda,s}}-H^0_{{m,f,u,\lambda,s}}\right).
$$

被测单元 $u$ 不进入候选原型；距离用全部坐标的对称RMS归一化欧氏误差。跨模型只比较21点相对深度上的六角色能量份额，不比较4B和14B的坐标编号。

**行为和场结果。** `{json.dumps(behavior, ensure_ascii=False)}`。逐族行为账保存于 `analysis/family_behavior.json`。模型内全坐标留一检索准确率为 `{json.dumps(within, ensure_ascii=False)}`；这是当前样本上的身份可预测性，不等于坐标因果齿轮。

**跨模型结果与门槛。** 共同族为 `{result['common_qualified_families']}`。跨模型留一检索结果为 `{json.dumps(cross, ensure_ascii=False)}`。冻结确认门裁决为 `{json.dumps(result['hypothesis_gates'], ensure_ascii=False)}`，总确认 `{result['family_specific_topology_confirmed']}`。模型中心化按同一词汇单元减去全族平均，以削弱共享前向和词汇单元底盘；它也可能减掉真实共享语义，因此与原始账并列而不互相替代。

**理论进展、问题和硬伤。** 若多单元留一仍命中同族，Phase2242 的结果不再能仅由单词身份解释，得到跨规模“语言族角色响应拓扑”候选；它仍不是相同坐标、参数同构或因果电路。材料是受控中英文本；只有四个单元；Qwen14层数和维数不同；GLM/DS若未过行为门只能形成接口边界；全坐标欧氏距离会受能量尺度影响；角色RMS用于跨模型摘要会丢符号和低值坐标细节。理论名称和RDC原则不变，没有新数学闭合。

**结论与下一步。** 工程自检 `{result['all_checks_passed']}`。下一期把每个合格模型的48个“族×单元”全坐标响应原型（包含embedding和每层HiddenState）写入客户端，保存跨模型矩阵和哈希，再删除未展示的逐样本大场。若确认门通过，下一研究应扩展自然材料与第三个行为合格模型；若失败，则保留模型内结构并降低跨规模普遍性主张。

**相关文件。** 脚本 `tests/glm5/phase2244_c971_c984_multiunit_cross_model_topology.py`；worker `tests/glm5/phase2239_c925_c936_cross_model_full_coordinate_worker.py`；结果 `{OUT.relative_to(ROOT)}`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    material = read_rows(SOURCE)
    save(OUT / "protocol/execution_identity.json", {
        "timestamp": datetime.now().astimezone().isoformat(), "phase": PHASE,
        "source": str(SOURCE.relative_to(ROOT)), "source_contract_hash": contract.file_hash(SOURCE),
        "models_in_order": list(MODELS), "worker": str(WORKER.relative_to(ROOT)),
        "no_post_reveal_changes": True,
    })
    worker_status = run_workers()
    model_results = {model: value["result"] for model, value in worker_status.items()}
    family_behavior = {model: behavior_by_family(model, material) for model in MODELS}
    save(OUT / "analysis/family_behavior.json", family_behavior)
    qualified_families = {
        model: [family for family in contract.FAMILIES if ledger[family]["qualified"]]
        for model, ledger in family_behavior.items()
    }
    topologies = {}
    within = {}
    for model, model_result in model_results.items():
        if model_result.get("all_checks_passed") and model_result.get("behavior_qualified"):
            topologies[model] = semantic_prototypes(model, model_result, material)
            within[model] = within_model_full_coordinate_retrieval(model, topologies[model], qualified_families[model])
        else:
            within[model] = {"status": "NA_model_behavior_unqualified_or_worker_error"}
    save(OUT / "analysis/within_model_full_coordinate_retrieval.json", within)
    common = []
    cross = {"status": "NA_qwen_pair_unavailable"}
    if "qwen3" in topologies and "qwen3_14b" in topologies:
        common = [family for family in contract.FAMILIES
                  if family in qualified_families["qwen3"] and family in qualified_families["qwen3_14b"]]
        cross = cross_model_topology_retrieval(topologies["qwen3"], topologies["qwen3_14b"], common)
    save(OUT / "analysis/cross_model_leave_one_unit_out.json", cross)
    gates = {
        "minimum_common_families": len(common) >= contract.MIN_COMMON_FAMILIES,
        "raw_forward_accuracy": cross.get("raw", {}).get("forward", {}).get("accuracy", -1) >= contract.RAW_RETRIEVAL_GATE,
        "raw_reverse_accuracy": cross.get("raw", {}).get("reverse", {}).get("accuracy", -1) >= contract.RAW_RETRIEVAL_GATE,
        "centered_forward_accuracy": cross.get("unit_centered", {}).get("forward", {}).get("accuracy", -1) >= contract.CENTERED_RETRIEVAL_GATE,
        "centered_reverse_accuracy": cross.get("unit_centered", {}).get("reverse", {}).get("accuracy", -1) >= contract.CENTERED_RETRIEVAL_GATE,
        "all_median_margins_positive": all(
            cross.get(kind, {}).get(direction, {}).get("median_margin") is not None
            and cross[kind][direction]["median_margin"] > 0
            for kind in ("raw", "unit_centered") for direction in ("forward", "reverse")
        ),
    }
    checks = {
        "material_384": len(material) == 384,
        "all_workers_accounted": all("result" in value for value in worker_status.values()),
        "worker_status_known": all(result.get("status") in ("closed", "behavior_unqualified", "worker_error", "missing_worker_result")
                                   for result in model_results.values()),
        "qualified_fields_have_prototypes": all(model in topologies for model, result in model_results.items()
                                                if result.get("behavior_qualified")),
        "analysis_finite": all(
            value.get("status") != "closed" or all(
                np.isfinite(value[k]["accuracy"]) and np.isfinite(value[k]["median_margin"])
                for k in ("raw", "unit_centered")
            ) for value in within.values()
        ),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()), "worker_status": worker_status,
        "model_results": model_results, "family_behavior": family_behavior,
        "qualified_families": qualified_families, "qualified_field_models": sorted(topologies),
        "within_model_retrieval": within, "common_qualified_families": common,
        "cross_model_retrieval": cross, "hypothesis_gates": gates,
        "family_specific_topology_confirmed": all(gates.values()),
        "strict_conclusion": "Multi-unit family specificity is a predictive activation-topology claim only; coordinate and causal isomorphism are not tested.",
        "next_authorization": "Export every coordinate of unit response prototypes and matrices, verify, then clean only replaced sample fields.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps({
        "phase": PHASE, "all_checks_passed": result["all_checks_passed"],
        "qualified_field_models": result["qualified_field_models"],
        "common_qualified_families": common, "hypothesis_gates": gates,
        "family_specific_topology_confirmed": result["family_specific_topology_confirmed"],
    }, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
