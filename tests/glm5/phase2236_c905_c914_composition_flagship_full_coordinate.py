from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase2234_c870_c884_broad_family_gear_contract as contract  # noqa: E402


PHASE = 2236
CAMPAIGNS = tuple(f"C{i}" for i in range(905, 915))
SOURCE = ROOT / "tests/glm5/result/phase2235_c885_c904_qwen_broad_family_full_coordinate_tournament"
OUT = ROOT / "tests/glm5/result/phase2236_c905_c914_composition_flagship_full_coordinate"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PANELS = ("parent_confirmation", "parent_lockbox", "fresh_confirmation", "fresh_lockbox")


def save(path: Path, value: Any) -> None:
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


def cell_key(row: dict) -> tuple:
    return row["family"], row["language"], row["surface"], row["unit"]


def attitude_observations(field: np.ndarray, index: list[dict]) -> list[dict]:
    rows = [row for row in index if row["panel"] == "nested_attitude_flagship"]
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(cell_key(row), {})[row["cell"]] = row
    output = []
    for key, cells in sorted(grouped.items()):
        if not all(cell in cells for cell in ("o0i0", "o0i1", "o1i0", "o1i1")):
            continue
        values = {cell: np.asarray(field[cells[cell]["hidden_index"]], dtype=np.float32) for cell in cells}
        interaction = values["o1i1"] - values["o1i0"] - values["o0i1"] + values["o0i0"]
        row = cells["o0i0"]
        output.append({"kind": "attitude_interaction", "family": row["family"],
                       "language": row["language"], "surface": row["surface"],
                       "unit": row["unit"], "partition": row["partition"], "value": interaction})
    return output


def graph_observations(field: np.ndarray, index: list[dict]) -> list[dict]:
    rows = [row for row in index if row["panel"] == "recursive_graph_flagship"]
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(cell_key(row), {})[row["cell"]] = row
    definitions = {
        "depth_1_to_2": ("d2k0", "d1k0"),
        "depth_2_to_3": ("d3k0", "d2k0"),
        "shortcut_depth_2": ("d2k1", "d2k0"),
        "shortcut_depth_3": ("d3k1", "d3k0"),
    }
    output = []
    for key, cells in sorted(grouped.items()):
        for kind, (target, base) in definitions.items():
            if target not in cells or base not in cells:
                continue
            value = (np.asarray(field[cells[target]["hidden_index"]], dtype=np.float32)
                     - np.asarray(field[cells[base]["hidden_index"]], dtype=np.float32))
            row = cells[base]
            output.append({"kind": kind, "family": row["family"], "language": row["language"],
                           "surface": row["surface"], "unit": row["unit"],
                           "partition": row["partition"], "value": value})
    return output


def fit_prototypes(observations: list[dict]) -> dict[tuple, np.ndarray]:
    prototypes = {}
    keys = sorted({(row["kind"], row["family"]) for row in observations if row["partition"] == "discovery"})
    for key in keys:
        values = [row["value"] for row in observations
                  if row["partition"] == "discovery" and (row["kind"], row["family"]) == key]
        prototypes[key] = np.mean(values, axis=0, dtype=np.float32)
    return prototypes


def panel_name(dataset: str, partition: str) -> str:
    return f"{dataset}_{partition}"


def evaluate(observations: list[dict], prototypes: dict[tuple, np.ndarray], dataset: str) -> list[dict]:
    families_by_kind: dict[str, list[str]] = {}
    for kind, family in prototypes:
        families_by_kind.setdefault(kind, []).append(family)
    rows = []
    for row in observations:
        if row["partition"] == "discovery" or (row["kind"], row["family"]) not in prototypes:
            continue
        family_list = sorted(families_by_kind[row["kind"]])
        if len(family_list) < 2:
            continue
        family_i = family_list.index(row["family"])
        wrong_family = family_list[(family_i + 1) % len(family_list)]
        actual = row["value"]
        prediction = prototypes[(row["kind"], row["family"])]
        wrong = prototypes[(row["kind"], wrong_family)]
        mae = float(np.mean(np.abs(actual - prediction), dtype=np.float64))
        zero_mae = float(np.mean(np.abs(actual), dtype=np.float64))
        wrong_mae = float(np.mean(np.abs(actual - wrong), dtype=np.float64))
        rows.append({
            "dataset": dataset, "panel": panel_name(dataset, row["partition"]),
            "kind": row["kind"], "family": row["family"], "language": row["language"],
            "surface": row["surface"], "unit": row["unit"], "mae": mae,
            "zero_mae": zero_mae, "wrong_domain_mae": wrong_mae, "wrong_family": wrong_family,
            "relative_mae_gain_over_zero": 1.0 - mae / max(zero_mae, 1e-12),
            "relative_mae_gain_over_wrong_domain": 1.0 - mae / max(wrong_mae, 1e-12),
        })
    return rows


def summarize(rows: list[dict]) -> tuple[dict, list[str]]:
    summary = {}
    candidates = []
    families = sorted({row["family"] for row in rows})
    for family in families:
        family_panels = {}
        for panel in PANELS:
            subset = [row for row in rows if row["family"] == family and row["panel"] == panel]
            if not subset:
                family_panels[panel] = {"units": 0, "passed": False}
                continue
            units = len({(row["kind"], row["language"], row["surface"], row["unit"]) for row in subset})
            gain_zero = float(np.mean([row["relative_mae_gain_over_zero"] for row in subset]))
            gain_wrong = float(np.mean([row["relative_mae_gain_over_wrong_domain"] for row in subset]))
            family_panels[panel] = {
                "units": units, "rows": len(subset),
                "relative_mae_gain_over_zero": gain_zero,
                "relative_mae_gain_over_wrong_domain": gain_wrong,
                "passed": units >= contract.FLAGSHIP_GATES["minimum_units"]
                          and gain_zero >= contract.FLAGSHIP_GATES["relative_mae_gain_over_zero"]
                          and gain_wrong >= contract.FLAGSHIP_GATES["relative_mae_gain_over_wrong_domain"],
            }
        family_panels["strict_pass"] = all(family_panels[p]["passed"] for p in PANELS)
        if family_panels["strict_pass"]:
            candidates.append(family)
        summary[family] = family_panels
    return summary, candidates


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {
        family: {panel: {
            "n": values.get("units", 0),
            "gain0": values.get("relative_mae_gain_over_zero"),
            "gain_wrong": values.get("relative_mae_gain_over_wrong_domain"),
            "pass": values.get("passed"),
        } for panel, values in panels.items() if panel in PANELS}
        for family, panels in result["family_summary"].items()
    }
    formula = r"""
$$
I_u=H_{11,u}-H_{10,u}-H_{01,u}+H_{00,u},\qquad
D^{12}_u=H_{2,0,u}-H_{1,0,u},\qquad
S^d_u=H_{d,1,u}-H_{d,0,u}.
$$
$$
\widehat R_{k,f}=\frac{1}{|\mathcal D_f|}\sum_{u\in\mathcal D_f}R_{k,f,u},\qquad
G_0=1-\frac{\operatorname{MAE}(R,\widehat R_f)}{\operatorname{MAE}(R,0)},\qquad
G_w=1-\frac{\operatorname{MAE}(R,\widehat R_f)}{\operatorname{MAE}(R,\widehat R_{f'})}.
$$
"""
    text = f"""

## Phase {PHASE}: 组合语言旗舰的全坐标二阶交互与路径增量审计 [{stamp}]

**范围与证据审查。** 本期执行 C905-C914，只复用 Phase 2235 已通过双行为切片的 Qwen3-4B 全坐标场，不重新选择材料或门槛。它检验的是组合响应能否从发现集逐坐标预测到确认、锁箱和新词，不检验 Attention、MLP、权重，也不把高预测分数称作因果电路。

**测试原理与用例。** 嵌套态度用“外层否定/内层否定”的四格二阶交互；递归图用一跳到两跳、两跳到三跳以及有无直接捷径的逐坐标增量。例子覆盖 like/regret/remember 三种态度域，以及 taxonomy/part-whole/temporal 三种图域，中英文、直接/释义表面、父词汇/全新词汇。全部 38 检查点、6 角色、2560 坐标参与误差，未做 Top-K、PCA 或跨坐标压缩。
{formula}
**冻结门槛与结果汇总。** 四面板均要求至少 4 个单位、相对零模型 MAE 增益不低于 0.05、相对等容量错领域原型增益不低于 0.03。逐族汇总为 `{json.dumps(compact, ensure_ascii=False)}`；严格组合候选为 `{result['strict_candidates']}`。详细逐单位结果保存在 JSONL，不以均值遮蔽缺失面板。

**理论进展与严格分析。** 本期若通过，只说明某类组合的完整逐坐标响应具有跨单位可重复性，并超过零响应和错领域响应；它比整向量余弦更具体，但发现集原型仍是样本均值，不是从单个初态推导出的唯一变换律。若图域或态度域不通过，不影响 Phase 2235 的一阶族条件候选，也不授权宣称模型缺少组合能力。

**问题、硬伤和瓶颈。** 受控模板与机器自然度不能替代人类盲评；双行为筛选改变了可用样本；错领域循环只是冻结的等容量负控之一；二阶差分会把四个输入各自的词汇与长度变化带入交互；全场 MAE 会让高能坐标贡献更大；预测正确仍不等于因果必要或充分。

**结论和授权。** 组合旗舰继续按逐坐标规律积累，不因单一路线阴性停止。Phase 2235 的严格语言族候选进入预注册的全坐标调用/删除/错族干预；跨模型相同语义分母、可视化和清理继续执行。理论名称保持“条件化输出场闭合理论”，本期不授权新数学闭合。

**相关文件。** 脚本 `tests/glm5/phase2236_c905_c914_composition_flagship_full_coordinate.py`；结果目录 `{OUT.relative_to(ROOT)}`；源全场目录 `{SOURCE.relative_to(ROOT)}`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    source_final = json.loads((SOURCE / "analysis/final.json").read_text(encoding="utf-8"))
    if not source_final["all_checks_passed"]:
        raise RuntimeError("Phase 2235 is incomplete")
    save(OUT / "protocol/execution_identity.json", {
        "timestamp": datetime.now().astimezone().isoformat(),
        "source_final_sha256": file_hash(SOURCE / "analysis/final.json"),
        "source_parent_field_sha256": file_hash(SOURCE / "raw/parent/qualified_role_field.float16.npy"),
        "source_fresh_field_sha256": file_hash(SOURCE / "raw/fresh/qualified_role_field.float16.npy"),
        "gates": contract.FLAGSHIP_GATES,
        "post_reveal_changes": "none",
    })
    all_observations = []
    counts = {}
    prototype_arrays = {}
    for dataset in ("parent", "fresh"):
        field_path = SOURCE / f"raw/{dataset}/qualified_role_field.float16.npy"
        index = read_rows(SOURCE / f"raw/{dataset}/hidden_index.jsonl")
        field = np.load(field_path, mmap_mode="r")
        try:
            observations = attitude_observations(field, index) + graph_observations(field, index)
        finally:
            close_mmap(field)
        counts[dataset] = len(observations)
        if dataset == "parent":
            prototypes = fit_prototypes(observations)
            for (kind, family), value in prototypes.items():
                prototype_arrays[f"{kind}|{family}"] = value.astype(np.float16)
        all_observations.extend((dataset, row) for row in observations)
    keys = sorted(prototype_arrays)
    prototype_stack = np.stack([prototype_arrays[key] for key in keys], axis=0)
    np.save(OUT / "analysis/full_coordinate_flagship_prototypes.float16.npy", prototype_stack)
    save(OUT / "analysis/prototype_index.json", {"keys": keys, "shape": list(prototype_stack.shape)})
    parent_observations = [row for dataset, row in all_observations if dataset == "parent"]
    prototypes = fit_prototypes(parent_observations)
    metric_rows = []
    for dataset in ("parent", "fresh"):
        metric_rows.extend(evaluate([row for name, row in all_observations if name == dataset], prototypes, dataset))
    write_rows(OUT / "analysis/unit_metrics.jsonl", metric_rows)
    summary, candidates = summarize(metric_rows)
    save(OUT / "analysis/family_summary.json", summary)
    checks = {
        "source_passed": True,
        "both_flagships_observed": bool([r for r in metric_rows if r["kind"] == "attitude_interaction"])
                                   and bool([r for r in metric_rows if r["kind"] != "attitude_interaction"]),
        "all_coordinates_retained": prototype_stack.shape[-1] == contract.DIM,
        "all_checkpoints_retained": prototype_stack.shape[1] == contract.CHECKPOINTS,
        "four_panels_scored": set(row["panel"] for row in metric_rows) == set(PANELS),
        "finite": all(np.isfinite(row[k]) for row in metric_rows for k in
                      ("mae", "zero_mae", "wrong_domain_mae", "relative_mae_gain_over_zero",
                       "relative_mae_gain_over_wrong_domain")),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()), "observation_counts": counts,
        "prototype_shape": list(prototype_stack.shape), "gates": contract.FLAGSHIP_GATES,
        "family_summary": summary, "strict_candidates": candidates,
        "strict_conclusion": "A strict pass is prospective full-coordinate composition-response repeatability beyond zero and wrong-domain prototypes, not a causal circuit or a state-derived law.",
        "next_authorization": "Run registered causal branch for Phase2235 strict family candidates; continue cross-model and visualization branches regardless of flagship outcome.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
