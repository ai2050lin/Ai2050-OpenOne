#!/usr/bin/env python3
"""Semantic-validity interaction atlas versus lexical control across six lockboxes."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2423 = RESULT / "phase2423_c30001_c30320_semantic_validity_behavior_contract"
P2424 = RESULT / "phase2424_c30321_c30640_semantic_validity_multievent_fullfield"
OUT = RESULT / "phase2425_c30641_c30960_semantic_specific_interaction_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2425
CAMPAIGN = "C30641-C30960"
INTERACTIONS = ("semantic_validity", "lexical_control")
COMPONENTS = ("total", "attention", "mlp")
SPLITS = ("confirmation", "fresh_unit", "template", "joint", "language", "family")
STAGES = ("global", "family", "state", "mismatch")
EVENTS = ("fact1_relation", "query_end", "answer_boundary")
SHIFT = 791
ANALYSIS_VERSION = "v2_zero_variance_guard"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def configuration_index(rows: list[dict]) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    configs = sorted({row["config_id"] for row in rows})
    mapping = {(row["config_id"], row["variant"], row["query_role"]): index for index, row in enumerate(rows)}
    meta, indices = [], {variant: {role: [] for role in ("source", "target")} for variant in ("valid", "broken_a", "broken_b")}
    for config in configs:
        source = rows[mapping[(config, "valid", "source")]]
        meta.append({key: source[key] for key in ("config_id", "family", "unit", "language", "surface", "surface_class", "direction", "partition")})
        for variant in indices:
            for role in indices[variant]:
                indices[variant][role].append(mapping[(config, variant, role)])
    for variant in indices:
        for role in indices[variant]:
            indices[variant][role] = np.asarray(indices[variant][role], dtype=np.int64)
    return meta, indices


def interactions(field: np.ndarray, layer: int, event: int, indices: dict) -> tuple[np.ndarray, np.ndarray]:
    differences = {}
    for variant in ("valid", "broken_a", "broken_b"):
        target = np.asarray(field[indices[variant]["target"], layer, event], dtype=np.float32)
        source = np.asarray(field[indices[variant]["source"], layer, event], dtype=np.float32)
        differences[variant] = target - source
    return differences["valid"] - differences["broken_a"], differences["broken_a"] - differences["broken_b"]


def fit(train: np.ndarray, families: np.ndarray, h: np.ndarray, y: np.ndarray, family_conditioned: bool = True) -> dict:
    global_h, global_y = h[train].mean(0), y[train].mean(0)
    family_h, family_y = {}, {}
    centered_h, centered_y = [], []
    for index in train:
        family = families[index]
        if family_conditioned and family not in family_h:
            chosen = train[families[train] == family]
            family_h[family], family_y[family] = h[chosen].mean(0), y[chosen].mean(0)
        base_h = family_h.get(family, global_h); base_y = family_y.get(family, global_y)
        centered_h.append(h[index] - base_h); centered_y.append(y[index] - base_y)
    x, target = np.asarray(centered_h), np.asarray(centered_y)
    slope = np.sum(x * target, axis=0) / (np.sum(x * x, axis=0) + 1e-8)
    return {"global_h": global_h, "global_y": global_y, "family_h": family_h, "family_y": family_y, "slope": slope}


def predict(test: np.ndarray, families: np.ndarray, h: np.ndarray, fitted: dict, mismatch: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    family_h = np.stack([fitted["family_h"].get(families[i], fitted["global_h"]) for i in test])
    family_y = np.stack([fitted["family_y"].get(families[i], fitted["global_y"]) for i in test])
    slope = np.roll(fitted["slope"], SHIFT) if mismatch else fitted["slope"]
    return np.broadcast_to(fitted["global_y"], family_y.shape), family_y, family_y + (h[test] - family_h) * slope


def gains(truth: np.ndarray, global_pred: np.ndarray, family_pred: np.ndarray, state_pred: np.ndarray,
          mismatch_pred: np.ndarray) -> list[float]:
    denominator = float(np.sum((truth - global_pred) ** 2))
    scale = float(np.sum(truth * truth))
    if denominator <= max(1e-20, scale * 1e-12):
        return [0.0, 0.0, 0.0, 0.0]
    return [0.0, 1 - float(np.sum((truth - family_pred) ** 2)) / denominator,
            1 - float(np.sum((truth - state_pred) ** 2)) / denominator,
            1 - float(np.sum((truth - mismatch_pred) ** 2)) / denominator]


def split_specs(meta: list[dict], families: np.ndarray) -> dict:
    controlled = np.asarray([row["surface_class"] == "controlled" for row in meta])
    unit = np.asarray([int(row["unit"]) for row in meta])
    language = np.asarray([row["language"] for row in meta], dtype=object)
    natural = ~controlled
    full_train = np.flatnonzero(controlled & (unit < 6))
    return {
        "confirmation": (np.flatnonzero(controlled & (unit < 4)), np.flatnonzero(controlled & (unit >= 4) & (unit < 6)), True),
        "fresh_unit": (full_train, np.flatnonzero(controlled & (unit >= 6)), True),
        "template": (full_train, np.flatnonzero(natural & (unit < 6)), True),
        "joint": (full_train, np.flatnonzero(natural & (unit >= 6)), True),
        "language": (np.flatnonzero(controlled & (unit < 6) & (language == "en")),
                     np.flatnonzero(controlled & (unit < 6) & (language == "zh")), True),
        "family": (full_train, np.flatnonzero(controlled & (unit >= 6)), False),
    }


def family_holdout(meta: list[dict], families: np.ndarray, train_pool: np.ndarray, test_pool: np.ndarray,
                   h: np.ndarray, y: np.ndarray) -> list[float]:
    truth_all, global_all, family_all, state_all, mismatch_all = [], [], [], [], []
    for family in sorted(set(families)):
        train = train_pool[families[train_pool] != family]
        test = test_pool[families[test_pool] == family]
        fitted = fit(train, families, h, y, family_conditioned=False)
        global_pred, family_pred, state_pred = predict(test, families, h, fitted)
        _, _, mismatch_pred = predict(test, families, h, fitted, mismatch=True)
        truth_all.append(y[test]); global_all.append(global_pred); family_all.append(family_pred)
        state_all.append(state_pred); mismatch_all.append(mismatch_pred)
    return gains(np.concatenate(truth_all), np.concatenate(global_all), np.concatenate(family_all),
                 np.concatenate(state_all), np.concatenate(mismatch_all))


def analyze(rows: list[dict], collection: dict) -> dict:
    meta, index = configuration_index(rows)
    families = np.asarray([row["family"] for row in meta], dtype=object)
    family_names = sorted(set(families))
    specs = split_specs(meta, families)
    full_train = specs["fresh_unit"][0]
    state = np.load(collection["state"]["path"], mmap_mode="r")
    attention = np.load(collection["attention"]["path"], mmap_mode="r")
    mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    layers, events, dim = attention.shape[1:]
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    metrics = np.zeros((2, 3, len(SPLITS), len(STAGES) + 1, layers, events), dtype=np.float32)
    passports = np.lib.format.open_memmap(derived / "interaction_update_family.float32.npy", mode="w+", dtype=np.float32,
                                           shape=(2, 3, layers, events, len(family_names), dim))
    state_passports = np.lib.format.open_memmap(derived / "interaction_state_family.float32.npy", mode="w+", dtype=np.float32,
                                                 shape=(2, layers, events, len(family_names), dim))
    slopes = np.lib.format.open_memmap(derived / "interaction_diagonal_slope.float32.npy", mode="w+", dtype=np.float32,
                                       shape=(2, 3, layers, events, dim))
    coordinate_rms = np.lib.format.open_memmap(derived / "interaction_coordinate_rms.float32.npy", mode="w+", dtype=np.float32,
                                                shape=(2, 4, layers, events, dim))
    for layer in range(layers):
        for event in range(events):
            h_sem, h_lex = interactions(state, layer, event, index)
            a_sem, a_lex = interactions(attention, layer, event, index)
            m_sem, m_lex = interactions(mlp, layer, event, index)
            for ii, (h, a, m) in enumerate(((h_sem, a_sem, m_sem), (h_lex, a_lex, m_lex))):
                coordinate_rms[ii, 0, layer, event] = np.sqrt(np.mean(h * h, axis=0))
                for fi, family in enumerate(family_names):
                    state_passports[ii, layer, event, fi] = h[full_train[families[full_train] == family]].mean(0)
                for ci, y in enumerate((a + m, a, m)):
                    coordinate_rms[ii, ci + 1, layer, event] = np.sqrt(np.mean(y * y, axis=0))
                    fitted_full = fit(full_train, families, h, y)
                    slopes[ii, ci, layer, event] = fitted_full["slope"]
                    for fi, family in enumerate(family_names):
                        passports[ii, ci, layer, event, fi] = fitted_full["family_y"][family]
                    for si, split in enumerate(SPLITS):
                        train, test, conditioned = specs[split]
                        if split == "family":
                            values = family_holdout(meta, families, train, test, h, y)
                        else:
                            fitted = fit(train, families, h, y, family_conditioned=conditioned)
                            global_pred, family_pred, state_pred = predict(test, families, h, fitted)
                            _, _, mismatch_pred = predict(test, families, h, fitted, mismatch=True)
                            values = gains(y[test], global_pred, family_pred, state_pred, mismatch_pred)
                        metrics[ii, ci, si, :4, layer, event] = values
                        metrics[ii, ci, si, 4, layer, event] = float(np.mean(y[test] * y[test]))
            passports.flush(); state_passports.flush(); slopes.flush(); coordinate_rms.flush()
        print(f"[phase2425] layer {layer + 1}/{layers}", flush=True)
    np.save(derived / "interaction_atlas_metrics.float32.npy", metrics)
    summary = {}
    for ii, interaction in enumerate(INTERACTIONS):
        summary[interaction] = {}
        for ci, component in enumerate(COMPONENTS):
            summary[interaction][component] = {}
            for si, split in enumerate(SPLITS):
                values = metrics[ii, ci, si]
                summary[interaction][component][split] = {
                    "family_gain": float(values[1].mean()), "state_gain": float(values[2].mean()),
                    "mismatch_gain": float(values[3].mean()),
                    "state_increment": float((values[2] - values[1]).mean()),
                    "physical_advantage": float((values[2] - values[3]).mean()),
                    "interaction_energy": float(values[4].mean()),
                    "best_state_cell": [int(x) for x in np.unravel_index(np.argmax(values[2]), values[2].shape)],
                }
    specificity = {component: {split: {
        "state_gain_margin": summary["semantic_validity"][component][split]["state_gain"] - summary["lexical_control"][component][split]["state_gain"],
        "physical_advantage_margin": summary["semantic_validity"][component][split]["physical_advantage"] - summary["lexical_control"][component][split]["physical_advantage"],
        "energy_ratio": summary["semantic_validity"][component][split]["interaction_energy"] /
                        max(summary["lexical_control"][component][split]["interaction_energy"], 1e-30)}
        for split in SPLITS} for component in COMPONENTS}
    files = {"metrics": str(derived / "interaction_atlas_metrics.float32.npy"),
             "update_passports": str(derived / "interaction_update_family.float32.npy"),
             "state_passports": str(derived / "interaction_state_family.float32.npy"),
             "slopes": str(derived / "interaction_diagonal_slope.float32.npy"),
             "coordinate_rms": str(derived / "interaction_coordinate_rms.float32.npy")}
    for value in (passports, state_passports, slopes, coordinate_rms, state, attention, mlp):
        close(value)
    return {"analysis_version": ANALYSIS_VERSION, "configurations": len(meta), "families": family_names,
            "split_sizes": {name: {"train": len(spec[0]), "test": len(spec[1])} for name, spec in specs.items()},
            "summary": summary, "specificity": specificity, "files": files}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 谓词语义有效性—词项对照多锁箱全坐标交互图谱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将Phase2424三个事件、36层、全部2560坐标按同配置双角色和三有效性组合成$I_{{sem}}$与$I_{{lex}}$。对总更新$A+M$、Attention、MLP分别只在训练材料拟合家族均值和逐物理坐标斜率；循环错配791位为第一坐标零假设。用unit0–3拟合/unit4–5确认、unit0–5拟合/unit6–7 fresh、自然模板、unit×模板联合、英文到中文、留一关系族六种锁箱。留一家族不借用被留家族均值，只测试其他族拟合的全局同坐标律。

$$\hat U_{{f,i}}=\bar U_f+\beta_i(H_i-\bar H_{{f,i}}),\qquad
\beta_i=\frac{{\sum(H_i-\bar H_{{f,i}})(U_i-\bar U_{{f,i}})}}{{\sum(H_i-\bar H_{{f,i}})^2+10^{{-8}}}},$$

$$G=1-\frac{{\sum\|U-\hat U\|^2}}{{\sum\|U-\bar U_{{train}}\|^2}},\qquad
\Delta_{{sem}}=G(I_{{sem}})-G(I_{{lex}}).$$

**结果汇总。** 配置/切分 `{json.dumps({'configurations': result['analysis']['configurations'], 'split_sizes': result['analysis']['split_sizes']}, ensure_ascii=False)}`；语义与词项图谱摘要 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；语义特异性 `{json.dumps(result['analysis']['specificity'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2425_c30641_c30960_semantic_specific_interaction_atlas.py`；逐interaction×组件×split×stage×layer×event指标，以及保留全部2560坐标的家族更新护照、状态护照、逐坐标斜率和RMS位于`tests/glm5/result/phase2425_c30641_c30960_semantic_specific_interaction_atlas/derived`，final位于`analysis`。未修改其他Markdown。

**分析与理论进展。** 这一步把“能量更大”与“存在可复用的语义专属坐标律”分开。能量比只说明有效谓词对双角色场影响更强；只有语义交互在六锁箱中同时超过词项交互、家族均值和错位坐标，才可提升为候选条件齿轮。家族留出尤其检验有限参数是否复用同一规律到未拟合关系族。

**问题硬伤与结论。** 对角斜率仍是最基础的同坐标近似，不覆盖非线性协同；循环错配只是单一坐标零假设，Phase2426将加入随机、等方差bin与样本置乱。不同谓词token长度和难度仍可能影响交互能量。所有拟合预测均是离线关联，不是因果。无论能量是否阳性，本Phase都不宣称纯逻辑张量或条件齿轮。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8"))
        if result.get("analysis", {}).get("analysis_version") == ANALYSIS_VERSION:
            append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2424 / "index/semantic_validity_rows.jsonl")
    collection = json.loads((P2424 / "analysis/final.json").read_text(encoding="utf-8"))["collection"]
    analysis = analyze(rows, collection)
    specificity = analysis["specificity"]
    reusable = all(specificity[component][split]["state_gain_margin"] > 0 and
                   specificity[component][split]["physical_advantage_margin"] > 0
                   for component in COMPONENTS for split in SPLITS)
    adjudication = {"semantic_energy_exceeds_lexical_all_cells_averaged":
                    all(specificity[component][split]["energy_ratio"] > 1 for component in COMPONENTS for split in SPLITS),
                    "semantic_state_law_exceeds_lexical_all_components_splits": reusable,
                    "semantic_specific_coordinate_operator_detected": reusable,
                    "conditional_coordinate_gear_proven": False}
    checks = {"configurations_1024": analysis["configurations"] == 1024,
              "six_splits": set(analysis["split_sizes"]) == set(SPLITS),
              "three_components": set(analysis["summary"]["semantic_validity"]) == set(COMPONENTS),
              "full_coordinate_derived": all(Path(path).exists() for path in analysis["files"].values()),
              "finite": all(math.isfinite(value) for component in specificity.values() for split in component.values() for value in split.values()),
              "raw_retained": all(Path(item["path"]).exists() for item in collection.values()),
              "claim_boundary": not adjudication["conditional_coordinate_gear_proven"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
