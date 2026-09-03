#!/usr/bin/env python3
"""Cross-layer semantic interaction transport and two-hop path consistency."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import phase2411_c26161_c26480_crosslayer_composition_output_bridge as geometry
import phase2425_c30641_c30960_semantic_specific_interaction_atlas as atlas


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2424 = RESULT / "phase2424_c30321_c30640_semantic_validity_multievent_fullfield"
OUT = RESULT / "phase2428_c31601_c31920_crosslayer_path_consistency"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2428
CAMPAIGN = "C31601-C31920"
INTERACTIONS = atlas.INTERACTIONS
SPLITS = ("confirmation", "fresh_unit", "template", "joint", "language", "family")
STAGES = ("identity", "scalar", "diagonal", "mismatch")
EVENT = 1


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def build_path(state_path: str, index: dict) -> dict:
    output = OUT / "derived/semantic_lexical_state_path.float32.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    state = np.load(state_path, mmap_mode="r")
    shape = (2, 37, len(index["valid"]["source"]), state.shape[-1])
    path = np.lib.format.open_memmap(output, mode="r+" if output.exists() else "w+", dtype=np.float32, shape=shape)
    progress = OUT / "derived/state_path_progress.json"
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed_qpoints"]) if progress.exists() else 0
    for qpoint in range(completed, 37):
        semantic, lexical = atlas.interactions(state, qpoint, EVENT, index)
        path[0, qpoint] = semantic; path[1, qpoint] = lexical; path.flush()
        save(progress, {"completed_qpoints": qpoint + 1})
        print(f"[phase2428 path] {qpoint + 1}/37", flush=True)
    path.flush(); close(path); close(state)
    return {"path": str(output), "shape": list(shape), "bytes": output.stat().st_size}


def partitions(split: str, specs: dict, families: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    train, test, conditioned = specs[split]
    if split != "family":
        return [(train, test, conditioned)]
    return [(train[families[train] != family], test[families[test] == family], False) for family in sorted(set(families))]


def gain(truth: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> float:
    denominator = float(np.sum((truth - baseline) ** 2))
    scale = float(np.sum(truth * truth))
    if denominator <= max(1e-20, scale * 1e-12):
        return 0.0
    return 1 - float(np.sum((truth - prediction) ** 2)) / denominator


def evaluate_adjacent(x: np.ndarray, y: np.ndarray, split: str, specs: dict, families: np.ndarray) -> tuple[list[float], list[np.ndarray]]:
    truths, bases = [], []
    predictions = {name: [] for name in STAGES}
    slopes = []
    for train, test, conditioned in partitions(split, specs, families):
        fitted = atlas.fit(train, families, x, y, family_conditioned=conditioned)
        base_x = np.stack([fitted["family_h"].get(families[i], fitted["global_h"]) for i in test])
        base_y = np.stack([fitted["family_y"].get(families[i], fitted["global_y"]) for i in test])
        centered_train_x = np.stack([x[i] - fitted["family_h"].get(families[i], fitted["global_h"]) for i in train])
        centered_train_y = np.stack([y[i] - fitted["family_y"].get(families[i], fitted["global_y"]) for i in train])
        scalar = float(np.sum(centered_train_x * centered_train_y) / (np.sum(centered_train_x * centered_train_x) + 1e-20))
        delta = x[test] - base_x
        truths.append(y[test]); bases.append(np.broadcast_to(fitted["global_y"], y[test].shape))
        predictions["identity"].append(base_y + delta)
        predictions["scalar"].append(base_y + delta * scalar)
        predictions["diagonal"].append(base_y + delta * fitted["slope"])
        predictions["mismatch"].append(base_y + delta * np.roll(fitted["slope"], atlas.SHIFT))
        slopes.append(fitted["slope"])
    truth, base = np.concatenate(truths), np.concatenate(bases)
    return [gain(truth, np.concatenate(predictions[name]), base) for name in STAGES], slopes


def evaluate_path(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, split: str, specs: dict,
                  families: np.ndarray) -> dict:
    truths, bases, direct_predictions, composed_predictions = [], [], [], []
    for train, test, conditioned in partitions(split, specs, families):
        first = atlas.fit(train, families, x0, x1, family_conditioned=conditioned)
        second = atlas.fit(train, families, x1, x2, family_conditioned=conditioned)
        direct = atlas.fit(train, families, x0, x2, family_conditioned=conditioned)
        base0 = np.stack([first["family_h"].get(families[i], first["global_h"]) for i in test])
        base2 = np.stack([second["family_y"].get(families[i], second["global_y"]) for i in test])
        direct_base0 = np.stack([direct["family_h"].get(families[i], direct["global_h"]) for i in test])
        direct_base2 = np.stack([direct["family_y"].get(families[i], direct["global_y"]) for i in test])
        composed = base2 + (x0[test] - base0) * first["slope"] * second["slope"]
        direct_p = direct_base2 + (x0[test] - direct_base0) * direct["slope"]
        truths.append(x2[test]); bases.append(np.broadcast_to(direct["global_y"], x2[test].shape))
        direct_predictions.append(direct_p); composed_predictions.append(composed)
    truth, base = np.concatenate(truths), np.concatenate(bases)
    direct_p, composed_p = np.concatenate(direct_predictions), np.concatenate(composed_predictions)
    return {"direct_gain": gain(truth, direct_p, base), "composed_gain": gain(truth, composed_p, base),
            "composed_minus_direct_gain": gain(truth, composed_p, base) - gain(truth, direct_p, base),
            "prediction_relative_rmse": float(np.sqrt(np.sum((composed_p - direct_p) ** 2) /
                                                           max(np.sum((truth - base) ** 2), 1e-30)))}


def relation_geometry(path: np.ndarray, train: np.ndarray, families: np.ndarray) -> dict:
    family_names = sorted(set(families))
    passports = np.stack([[path[q, train[families[train] == family]].mean(0) for family in family_names]
                          for q in range(path.shape[0])])
    relation, coordinates = [], []
    for qpoint in range(path.shape[0] - 1):
        relation.append(geometry.correlation(geometry.geometry_vector(passports[qpoint]),
                                             geometry.geometry_vector(passports[qpoint + 1])))
        coordinates.append(float(np.mean([geometry.cosine(passports[qpoint, fi], passports[qpoint + 1, fi])
                                          for fi in range(len(family_names))])))
    return {"adjacent_relation_geometry_mean": float(np.mean(relation)),
            "adjacent_relation_geometry_min": float(np.min(relation)),
            "adjacent_coordinate_cosine_mean": float(np.mean(coordinates)),
            "adjacent_coordinate_cosine_min": float(np.min(coordinates)),
            "layers": len(relation)}


def analyze(meta: list[dict], state_path: dict) -> dict:
    families = np.asarray([row["family"] for row in meta], dtype=object)
    specs = atlas.split_specs(meta, families)
    paths = np.load(state_path["path"], mmap_mode="r")
    metrics = np.zeros((2, len(SPLITS), len(STAGES), 36), dtype=np.float32)
    path_metrics = np.zeros((2, len(SPLITS), 4, 35), dtype=np.float32)
    slopes = np.lib.format.open_memmap(OUT / "derived/adjacent_diagonal_slope.float32.npy", mode="w+", dtype=np.float32,
                                       shape=(2, 36, paths.shape[-1]))
    full_train = specs["fresh_unit"][0]
    geometries = {}
    for ii, interaction in enumerate(INTERACTIONS):
        geometries[interaction] = relation_geometry(paths[ii], full_train, families)
        for layer in range(36):
            fitted = atlas.fit(full_train, families, paths[ii, layer], paths[ii, layer + 1])
            slopes[ii, layer] = fitted["slope"]
            for si, split in enumerate(SPLITS):
                values, _ = evaluate_adjacent(paths[ii, layer], paths[ii, layer + 1], split, specs, families)
                metrics[ii, si, :, layer] = values
                if layer < 35:
                    item = evaluate_path(paths[ii, layer], paths[ii, layer + 1], paths[ii, layer + 2], split, specs, families)
                    path_metrics[ii, si, :, layer] = [item[key] for key in
                                                       ("direct_gain", "composed_gain", "composed_minus_direct_gain", "prediction_relative_rmse")]
            print(f"[phase2428 analyze] {interaction} {layer + 1}/36", flush=True)
    slopes.flush(); close(slopes)
    np.save(OUT / "derived/crosslayer_stage_metrics.float32.npy", metrics)
    np.save(OUT / "derived/two_hop_path_metrics.float32.npy", path_metrics)
    summary = {interaction: {split: {
        "identity_gain": float(metrics[ii, si, 0].mean()), "scalar_gain": float(metrics[ii, si, 1].mean()),
        "diagonal_gain": float(metrics[ii, si, 2].mean()), "mismatch_gain": float(metrics[ii, si, 3].mean()),
        "diagonal_physical_advantage": float((metrics[ii, si, 2] - metrics[ii, si, 3]).mean()),
        "two_hop_direct_gain": float(path_metrics[ii, si, 0].mean()),
        "two_hop_composed_gain": float(path_metrics[ii, si, 1].mean()),
        "composed_minus_direct_gain": float(path_metrics[ii, si, 2].mean()),
        "two_hop_prediction_relative_rmse": float(path_metrics[ii, si, 3].mean())}
        for si, split in enumerate(SPLITS)} for ii, interaction in enumerate(INTERACTIONS)}
    specificity = {split: {
        "semantic_minus_lexical_diagonal_gain": summary["semantic_validity"][split]["diagonal_gain"] - summary["lexical_control"][split]["diagonal_gain"],
        "semantic_minus_lexical_physical_advantage": summary["semantic_validity"][split]["diagonal_physical_advantage"] - summary["lexical_control"][split]["diagonal_physical_advantage"]}
        for split in SPLITS}
    files = {"state_path": state_path["path"], "slopes": str(OUT / "derived/adjacent_diagonal_slope.float32.npy"),
             "metrics": str(OUT / "derived/crosslayer_stage_metrics.float32.npy"),
             "path_metrics": str(OUT / "derived/two_hop_path_metrics.float32.npy")}
    close(paths)
    return {"summary": summary, "specificity": specificity, "geometry": geometries, "files": files}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 语义有效性交互的跨层传递与两跳路径一致性（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 固定query-end事件，从embedding到36个block输出构造每个配置的$I_{{sem}}^H$与$I_{{lex}}^H$完整2560坐标路径。逐相邻层只在训练集拟合家族均值与四个基础候选：单位传递、全局标量、逐坐标对角、坐标错位；在六锁箱评价。对每个两跳$q\to q+2$同时拟合直接对角图和相邻图复合，检验局部律能否沿路径组合。另比较八关系族均值Gram图的相邻保持与同编号坐标余弦。

$$Z_{{q+1}}-\bar Z_{{f,q+1}}\approx\beta_q\odot(Z_q-\bar Z_{{f,q}}),$$

$$\widehat Z_{{q+2}}^{{compose}}=\bar Z_{{f,q+2}}+(Z_q-\bar Z_{{f,q}})\odot\beta_q\odot\beta_{{q+1}},$$

$$E_{{path}}=\frac{{\|\widehat Z_{{q+2}}^{{compose}}-\widehat Z_{{q+2}}^{{direct}}\|}}{{\|Z_{{q+2}}-\bar Z_{{train,q+2}}\|}}.$$

**结果汇总。** 跨层摘要 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；语义—词项特异性 `{json.dumps(result['analysis']['specificity'], ensure_ascii=False)}`；关系图与同坐标保持 `{json.dumps(result['analysis']['geometry'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2428_c31601_c31920_crosslayer_path_consistency.py`；两类交互×37状态点×1024配置×2560完整坐标路径、逐相邻层全坐标斜率、stage指标、两跳路径指标及final位于`tests/glm5/result/phase2428_c31601_c31920_crosslayer_path_consistency`。未修改其他Markdown。

**分析与理论进展。** 这一步区分“关系族几何看起来连续”和“同一个可拟合坐标律沿层复用”。Gram保持可以在坐标纹理完全重写时仍为高；对角物理优势要求编号身份；两跳复合接近直接图才进一步支持路径代数。语义交互必须比词项对照更强，才能从通用残差传播提升为语言机制候选。

**问题硬伤与结论。** 对角传递忽略坐标群协同；Phase2427内生组不稳定，故不把失败分组强行带入。本Phase是离线状态映射，不说明Transformer实际以该拟合图执行。两跳复合误差会累积float16采集噪声和基线误差；单位/标量/错位仅为基础零假设。任何阳性仍需Phase2429在直接关系与组合关系间检验代数内容。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2424 / "index/semantic_validity_rows.jsonl")
    meta, index = atlas.configuration_index(rows)
    collection = json.loads((P2424 / "analysis/final.json").read_text(encoding="utf-8"))["collection"]
    state_path = build_path(collection["state"]["path"], index)
    analysis = analyze(meta, state_path)
    semantic_specific = all(value["semantic_minus_lexical_diagonal_gain"] > 0 and
                            value["semantic_minus_lexical_physical_advantage"] > 0
                            for value in analysis["specificity"].values())
    path_closed = all(value["composed_minus_direct_gain"] >= -0.01 and value["two_hop_prediction_relative_rmse"] < .25
                      for value in analysis["summary"]["semantic_validity"].values())
    adjudication = {"semantic_crosslayer_law_exceeds_lexical_all_splits": semantic_specific,
                    "semantic_two_hop_path_consistent_all_splits": path_closed,
                    "reusable_semantic_crosslayer_operator_detected": semantic_specific and path_closed,
                    "conditional_coordinate_gear_proven": False}
    checks = {"full_path_shape": state_path["shape"] == [2, 37, 1024, 2560],
              "six_splits": set(analysis["specificity"]) == set(SPLITS),
              "thirty_six_adjacent_maps": all(value["layers"] == 36 for value in analysis["geometry"].values()),
              "full_coordinate_files": all(Path(path).exists() for path in analysis["files"].values()),
              "finite": all(math.isfinite(number) for value in analysis["specificity"].values() for number in value.values()),
              "raw_retained": all(Path(item["path"]).exists() for item in collection.values()),
              "claim_boundary": not adjudication["conditional_coordinate_gear_proven"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "event": "query_end", "state_path": state_path,
              "analysis": analysis, "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
