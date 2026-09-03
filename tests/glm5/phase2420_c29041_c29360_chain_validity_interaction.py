#!/usr/bin/env python3
"""Second-order chain-validity interaction versus two broken-chain lexical controls."""
from __future__ import annotations

import gc
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2419 = RESULT / "phase2419_c28721_c29040_semantic_specificity_controls"
OUT = RESULT / "phase2420_c29041_c29360_chain_validity_interaction"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2420
CAMPAIGN = "C29041-C29360"
VARIANTS = ("valid_composition", "broken_chain", "broken_chain_alt")
INTERACTIONS = ("valid_minus_broken", "broken_minus_broken_alt")
COMPONENTS = ("total", "attention", "mlp")
KEYS = ("family_gain", "state_gain", "mismatch_gain", "shuffle_mean_gain", "shuffle_q95_gain",
        "state_over_shuffle_q95", "interaction_energy", "layer_win_rate")
BRIDGE_LAYER = 14

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2397_c21681_c22000_operation_behavior_token_calibration as behavior  # noqa: E402
import phase2405_c24241_c24560_deconfounded_operation_contract as contract  # noqa: E402
import phase2415_c27441_c27760_exact_paired_composition as paired  # noqa: E402
import phase2416_c27761_c28080_crossmodel_exact_pair_replication as capture_utils  # noqa: E402
import phase2418_c28401_c28720_heteroscedasticity_residual_control as controls  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def alt_broken_prompt(row: dict) -> tuple[str, list[dict]]:
    first = contract.render_fact(row["family"], row["language"], row["surface"], row["source"], row["middle"])
    other = contract.triples(row["language"], (int(row["unit"]) + 2) % 8)
    distractor = next(value for value in other if value not in (row["source"], row["middle"], row["endpoint"]))
    second = contract.render_fact(row["family"], row["language"], row["surface"], distractor, row["endpoint"])
    return contract.prior.prompt_with_events(row["language"], [first, second], row["query"], row["candidates"])


def compile_material() -> list[dict]:
    old = read_rows(P2419 / "material/semantic_specificity_controls.jsonl")
    rows = [{**row} for row in old if row["variant"] in ("valid_composition", "broken_chain")]
    for row in old:
        if row["variant"] != "valid_composition":
            continue
        item = {**row}
        item["variant"] = "broken_chain_alt"
        item["pair_id"] = f"broken_chain_alt-{row['original_pair_id']}"
        item["case_id"] = f"{item['pair_id']}-s{row['steps']}"
        item["prompt"], item["events"] = alt_broken_prompt(item)
        rows.append(item)
    order = {name: index for index, name in enumerate(VARIANTS)}
    rows.sort(key=lambda row: (row["original_pair_id"], order[row["variant"]], row["steps"]))
    return rows


def material_audit(rows: list[dict]) -> dict:
    return {"rows": len(rows), "configurations": len({row["original_pair_id"] for row in rows}),
            "variants": dict(Counter(row["variant"] for row in rows)),
            "families": dict(Counter(row["family"] for row in rows)),
            "languages": dict(Counter(row["language"] for row in rows)),
            "surfaces": dict(Counter(row["surface"] for row in rows)),
            "directions": dict(Counter(row["direction"] for row in rows)),
            "unique_cases": len({row["case_id"] for row in rows}) == len(rows)}


def configuration_index(rows: list[dict]) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    configurations = sorted({row["original_pair_id"] for row in rows})
    mapping = {(row["original_pair_id"], row["variant"], int(row["steps"])): index for index, row in enumerate(rows)}
    config_rows, indices = [], {variant: {"step1": [], "step2": []} for variant in VARIANTS}
    for config in configurations:
        source = rows[mapping[(config, "valid_composition", 1)]]
        config_rows.append({key: source[key] for key in ("original_pair_id", "family", "unit", "language", "surface", "direction", "partition")})
        for variant in VARIANTS:
            indices[variant]["step1"].append(mapping[(config, variant, 1)])
            indices[variant]["step2"].append(mapping[(config, variant, 2)])
    for variant in VARIANTS:
        for step in ("step1", "step2"):
            indices[variant][step] = np.asarray(indices[variant][step], dtype=np.int64)
    return config_rows, indices


def behavior_summary(teacher: list[dict], rows: list[dict]) -> tuple[dict, dict[str, np.ndarray]]:
    meta = {row["case_id"]: row for row in rows}
    by_config: dict[str, dict[str, list[bool]]] = {}
    for record in teacher:
        row = meta[record["case_id"]]
        by_config.setdefault(row["original_pair_id"], {}).setdefault(row["variant"], []).append(record["mean_logprob_margin"] > 0)
    configs = sorted(by_config)
    arrays = {variant: np.asarray([all(by_config[config][variant]) for config in configs], dtype=np.float64) for variant in VARIANTS}
    summary = {variant: {"pairs": len(configs), "both_target_over_foil": float(values.mean()),
                         "both_correct_pairs": int(values.sum())} for variant, values in arrays.items()}
    summary["valid_minus_broken_behavior"] = float((arrays["valid_composition"] - arrays["broken_chain"]).mean())
    summary["broken_minus_alt_behavior"] = float((arrays["broken_chain"] - arrays["broken_chain_alt"]).mean())
    return summary, arrays


def analyze(rows: list[dict], collection: dict, behavior_arrays: dict[str, np.ndarray]) -> dict:
    config_rows, indices = configuration_index(rows)
    families = np.asarray([row["family"] for row in config_rows], dtype=object)
    train = np.asarray([i for i, row in enumerate(config_rows) if int(row["unit"]) < 4], dtype=np.int64)
    test = np.asarray([i for i, row in enumerate(config_rows) if int(row["unit"]) >= 4], dtype=np.int64)
    state = np.load(collection["state"]["path"], mmap_mode="r")
    attention = np.load(collection["attention"]["path"], mmap_mode="r")
    mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    layers = state.shape[1]
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    metrics = np.zeros((len(INTERACTIONS), len(COMPONENTS), len(KEYS), layers), dtype=np.float32)
    bridge_cache = None
    for layer in range(layers):
        fields = {"state": state, "attention": attention, "mlp": mlp}
        deltas = {}
        for variant in VARIANTS:
            s1, s2 = indices[variant]["step1"], indices[variant]["step2"]
            deltas[variant] = {
                name: np.asarray(field[s2, layer], dtype=np.float32) - np.asarray(field[s1, layer], dtype=np.float32)
                for name, field in fields.items()}
        interaction_fields = {
            "valid_minus_broken": {name: deltas["valid_composition"][name] - deltas["broken_chain"][name]
                                   for name in fields},
            "broken_minus_broken_alt": {name: deltas["broken_chain"][name] - deltas["broken_chain_alt"][name]
                                        for name in fields},
        }
        for ii, interaction in enumerate(INTERACTIONS):
            h = interaction_fields[interaction]["state"]
            a = interaction_fields[interaction]["attention"]
            m = interaction_fields[interaction]["mlp"]
            for ci, y in enumerate((a + m, a, m)):
                values, _ = controls.predict_controls(config_rows, train, test, families, h, y,
                                                     PHASE * 10000 + ii * 1000 + ci * 100 + layer)
                metrics[ii, ci, :, layer] = [values["family"], values["state"], values["coordinate_mismatch"],
                                             values["sample_shuffle_mean"], values["sample_shuffle_q95"],
                                             values["state"] - values["sample_shuffle_q95"],
                                             float(np.mean(y[test] * y[test])),
                                             float(values["state"] > values["sample_shuffle_q95"])]
            if interaction == "valid_minus_broken" and layer == BRIDGE_LAYER:
                bridge_cache = (h.copy(), (a + m).copy())
        print(f"[phase2420 analysis] layer {layer + 1}/{layers}", flush=True)
    np.save(derived / "chain_validity_interaction_metrics.float32.npy", metrics)
    summary = {interaction: {component: {key: float(metrics[ii, ci, ki].mean()) for ki, key in enumerate(KEYS)}
                             for ci, component in enumerate(COMPONENTS)}
               for ii, interaction in enumerate(INTERACTIONS)}
    specificity = {component: {"state_over_shuffle_margin":
                               summary["valid_minus_broken"][component]["state_over_shuffle_q95"] -
                               summary["broken_minus_broken_alt"][component]["state_over_shuffle_q95"],
                               "energy_ratio": summary["valid_minus_broken"][component]["interaction_energy"] /
                               max(summary["broken_minus_broken_alt"][component]["interaction_energy"], 1e-30)}
                   for component in COMPONENTS}
    if bridge_cache is None:
        raise RuntimeError("bridge cache missing")
    h, y = bridge_cache
    fitted = controls.family_bases(train, families, h, y)
    base_y, base_h = controls.base_for(test, families, fitted)
    truth = y[test]; prediction = base_y + (h[test] - base_h) * fitted["slope"]
    improvement = ((np.mean((truth - base_y) ** 2, axis=1) - np.mean((truth - prediction) ** 2, axis=1)) /
                   (np.mean((truth - fitted["global_y"]) ** 2, axis=1) + 1e-30))
    behavior_delta = behavior_arrays["valid_composition"][test] - behavior_arrays["broken_chain"][test]
    bridge = {"layer": BRIDGE_LAYER, "test_configurations": len(test),
              "behavior_delta_nonzero": int(np.sum(behavior_delta != 0)),
              "improvement_behavior_delta_correlation": controls.pearson(improvement, behavior_delta),
              "mean_improvement": float(improvement.mean()),
              "mean_behavior_delta": float(behavior_delta.mean())}
    np.save(derived / "validity_interaction_pair_improvement.float32.npy", improvement.astype(np.float32))
    close(state); close(attention); close(mlp)
    return {"configurations": len(config_rows), "train": len(train), "test": len(test), "summary": summary,
            "semantic_specificity": specificity, "behavior_bridge": bridge,
            "metrics": str(derived / "chain_validity_interaction_metrics.float32.npy")}


def cleanup(collection: dict) -> dict:
    paths, total = [], 0
    for value in collection.values():
        path = Path(value["path"])
        if path.exists():
            total += path.stat().st_size; paths.append(str(path)); path.unlink()
    return {"removed_files": len(paths), "removed_bytes": total, "removed_gib": total / 2**30,
            "recoverable": False, "paths": paths}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 事实链有效性的二阶全坐标交互场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2419显示有效组合与断链任务的matched-over-shuffle几乎相同，但通用查询场可能掩盖小的语义分量。本Phase使用同256个配置，建立有效链、断链A（干扰实体来自unit+1）、断链B（来自unit+2）三个严格一步/两步面板，共1536条。在Qwen3-4B完成目标—foil评分，并于answer boundary采集36层状态、Attention、MLP全部2560坐标。先取每面板的两步减一步，再计算二阶交互：语义候选$I_{{sem}}=D_{{valid}}-D_{{brokenA}}$；等规模词项对照$I_{{lex}}=D_{{brokenA}}-D_{{brokenB}}$。查询、候选、答案角色在二阶差分中相消。unit0–3拟合，unit4–7评价，仍用16次同条件样本置乱。

$$I^X_{{sem}}=(X_{{v,2}}-X_{{v,1}})-(X_{{a,2}}-X_{{a,1}}),\qquad
I^X_{{lex}}=(X_{{a,2}}-X_{{a,1}})-(X_{{b,2}}-X_{{b,1}}),$$

$$\Delta_{{validity}}=S(I_{{sem}})-S(I_{{lex}}),\qquad
R_E=\frac{{\mathbb E\|I^U_{{sem}}\|^2}}{{\mathbb E\|I^U_{{lex}}\|^2}}.$$

**结果汇总。** 材料 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；交互全坐标结果 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；有效性特异性 `{json.dumps(result['analysis']['semantic_specificity'], ensure_ascii=False)}`；行为桥 `{json.dumps(result['analysis']['behavior_bridge'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2420_c29041_c29360_chain_validity_interaction.py`；1536条材料、token索引、行为输出、逐interaction×组件×层全坐标指标、pair级桥和final位于`tests/glm5/result/phase2420_c29041_c29360_chain_validity_interaction`。未修改其他Markdown。

**分析与理论进展。** 二阶差分把“一步/两步查询+答案交换”的公共主效应去掉，再用两个不同断链作词项基线。$I_{{sem}}$若在能量、跨unit家族结构、当前样本状态增量上系统超过$I_{{lex}}$，才是事实链有效性编码的候选拼图；否则上一阶段的微小差异可由更换第二事实实体解释。

**问题硬伤与结论。** 两种断链实体并未逐token长度匹配；有效链和断链的自然语言可接受度不同。二阶差分会放大量化噪声，且线性预测只能检测可复用的同坐标局部律。行为仍是教师强制。即使$I_{{sem}}$阳性，也只定位“事实链有效性影响步数对比”的场，不证明模型执行了递归关系组合或形成输出闭合。原始float16场派生后删除且不可恢复。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    source_path = OUT / "material/chain_validity_interaction.jsonl"
    source = read_rows(source_path) if source_path.exists() else compile_material()
    if not source_path.exists():
        write_rows(source_path, source)
    audit = material_audit(source)
    model, tokenizer, label = capability.load_model("qwen4b")
    behavior.OUT = OUT; capture_utils.OUT = OUT
    try:
        index = OUT / "index/composition_rows.jsonl"
        if index.exists():
            rows = read_rows(index)
            calibration = json.loads((OUT / "analysis/token_calibration.json").read_text(encoding="utf-8"))
        else:
            rows, calibration = behavior.compile_rows(tokenizer, source)
            write_rows(index, rows); save(OUT / "analysis/token_calibration.json", calibration)
        teacher, teacher_all = behavior.score_rows("qwen4b", model, rows, 16)
        collection = capture_utils.collect("qwen4b", model, rows, 4)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    behavior_result, behavior_arrays = behavior_summary(teacher, rows)
    analysis = analyze(rows, collection, behavior_arrays)
    raw_cleanup = cleanup(collection)
    specificity = analysis["semantic_specificity"]
    adjudication = {"semantic_interaction_exceeds_lexical_all_components":
                    all(value["state_over_shuffle_margin"] > 0 for value in specificity.values()),
                    "semantic_interaction_energy_exceeds_lexical_all_components":
                    all(value["energy_ratio"] > 1 for value in specificity.values()),
                    "behavior_bridge_positive": analysis["behavior_bridge"]["improvement_behavior_delta_correlation"] > 0,
                    "fact_chain_validity_field_detected": False,
                    "recursive_composition_mechanism_proven": False}
    checks = {"three_panels_1536_rows": audit["rows"] == 1536 and audit["configurations"] == 256,
              "token_calibration": calibration["rows"] == 1536 and calibration["event_monotonic_rate"] == 1.0,
              "teacher_complete": len(teacher) == 1536,
              "full_coordinates": collection["state"]["shape"] == [1536, 36, 2560],
              "two_second_order_interactions": set(analysis["summary"]) == set(INTERACTIONS),
              "finite": all(math.isfinite(value) for interaction in analysis["summary"].values()
                            for component in interaction.values() for value in component.values()),
              "raw_cleaned": raw_cleanup["removed_files"] == 3, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": label, "material_audit": audit,
              "token_calibration": calibration, "teacher_all": teacher_all, "behavior": behavior_result,
              "collection": collection, "analysis": analysis, "cleanup": raw_cleanup,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "material": audit, "behavior": behavior_result,
                      "summary": analysis["summary"], "specificity": specificity,
                      "bridge": analysis["behavior_bridge"], "adjudication": adjudication,
                      "cleanup": raw_cleanup, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
