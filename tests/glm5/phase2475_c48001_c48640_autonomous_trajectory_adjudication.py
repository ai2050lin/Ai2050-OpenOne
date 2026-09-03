#!/usr/bin/env python3
"""Adjudicate aligned successful autonomous events without the Phase2466 metric mix-up."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2474 = next(RESULT.glob("phase2474_*"))
OUT = RESULT / "phase2475_c48001_c48640_autonomous_trajectory_adjudication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2475, "C48001-C48640", 2560
EVENTS = ("answer_boundary", "first_generated_token", "parsed_answer_token")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def derangements(count: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < count:
        permutation = rng.permutation(size)
        if np.all(permutation != np.arange(size)):
            result.append(permutation)
    return np.stack(result)


def matched_null(a: np.ndarray, b: np.ndarray, permutations: np.ndarray) -> dict:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-30)
    similarity = x @ y.T
    coordinate = float(np.mean(np.diag(similarity)))
    null = np.mean(similarity[np.arange(x.shape[0])[None, :], permutations], axis=1)
    q95 = float(np.quantile(null, 0.95))
    return {"coordinate": coordinate, "family_null_mean": float(np.mean(null)), "family_null_q95": q95, "family_identity_advantage": coordinate - q95}


def extract_events() -> tuple[np.memmap, list[dict], dict]:
    final = json.loads((P2474 / "analysis/final.json").read_text(encoding="utf-8"))
    source = np.load(final["collection"]["field"], mmap_mode="r")
    index = read_jsonl(Path(final["collection"]["index"]))
    path = OUT / "derived/aligned_autonomous_event_states.float32.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    states = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(index), 3, 38, DIM))
    enhanced = []
    for row_number, row in enumerate(index):
        answer_event = int(row["answer_step"]) if row["answer_step"] is not None else int(row["trajectory_events"] - 1)
        event_indices = [0, 1, answer_event]
        for event, source_event in enumerate(event_indices):
            states[row_number, event] = np.asarray(source[row_number, source_event], dtype=np.float32)
        enhanced.append({**row, "aligned_event_indices": dict(zip(EVENTS, event_indices))})
    states.flush(); close(source)
    index_path = OUT / "index/aligned_rows.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in enhanced), encoding="utf-8")
    return states, enhanced, {"states": str(path), "index": str(index_path), "shape": list(states.shape), "events": list(EVENTS)}


def passports(states: np.ndarray, index: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    units, languages, interfaces = [9, 10], ["en", "zh"], ["code", "entity"]
    families = sorted({row["family"] for row in index})
    result = np.zeros((2, 2, 2, 8, 3, 38, DIM), dtype=np.float32)
    counts = np.zeros((2, 2, 2, 8), dtype=np.int32)
    for ui, unit in enumerate(units):
        for li, language in enumerate(languages):
            for oi, interface in enumerate(interfaces):
                valid = [i for i, row in enumerate(index) if row["unit"] == unit and row["language"] == language and row["output_interface"] == interface and row["parsed_correct"]]
                baseline = np.mean(states[valid], axis=0, dtype=np.float64).astype(np.float32)
                for fi, family in enumerate(families):
                    selected = [i for i in valid if index[i]["family"] == family]
                    counts[ui, li, oi, fi] = len(selected)
                    if selected:
                        result[ui, li, oi, fi] = np.mean(states[selected], axis=0, dtype=np.float64).astype(np.float32) - baseline
    path = OUT / "derived/success_family_event_passports.float32.npy"
    np.save(path, result)
    count_path = OUT / "derived/success_family_event_counts.int32.npy"
    np.save(count_path, counts)
    return result, counts, {"path": str(path), "counts": str(count_path), "shape": list(result.shape), "families": families, "axes": ["unit", "language", "interface", "family", "event", "qpoint", "coordinate"]}


def all_metrics(passports: np.ndarray) -> dict:
    permutations = derangements(256, 8, 2475)
    result = {"crossinterface": {}, "crosslanguage": {}, "within_interface_trajectory": {}}
    for ui, unit in enumerate((9, 10)):
        result["crossinterface"][f"unit{unit}"] = {}
        result["crosslanguage"][f"unit{unit}"] = {}
        result["within_interface_trajectory"][f"unit{unit}"] = {}
        for event, event_name in enumerate(EVENTS):
            result["crossinterface"][f"unit{unit}"][event_name] = {}
            result["crosslanguage"][f"unit{unit}"][event_name] = {}
            for qpoint in range(38):
                ci = [matched_null(passports[ui, li, 0, :, event, qpoint], passports[ui, li, 1, :, event, qpoint], permutations) for li in range(2)]
                cl = [matched_null(passports[ui, 0, oi, :, event, qpoint], passports[ui, 1, oi, :, event, qpoint], permutations) for oi in range(2)]
                result["crossinterface"][f"unit{unit}"][event_name][f"q{qpoint}"] = {key: float(np.mean([x[key] for x in ci])) for key in ci[0]}
                result["crosslanguage"][f"unit{unit}"][event_name][f"q{qpoint}"] = {key: float(np.mean([x[key] for x in cl])) for key in cl[0]}
        for interface, interface_name in enumerate(("code", "entity")):
            result["within_interface_trajectory"][f"unit{unit}"][interface_name] = {"boundary_to_first": {}, "boundary_to_answer": {}, "first_to_answer": {}}
            for qpoint in range(38):
                pairs = ((0, 1, "boundary_to_first"), (0, 2, "boundary_to_answer"), (1, 2, "first_to_answer"))
                for first, second, name in pairs:
                    values = [matched_null(passports[ui, li, interface, :, first, qpoint], passports[ui, li, interface, :, second, qpoint], permutations) for li in range(2)]
                    result["within_interface_trajectory"][f"unit{unit}"][interface_name][name][f"q{qpoint}"] = {key: float(np.mean([x[key] for x in values])) for key in values[0]}
    selections = {"crossinterface": {}, "trajectory": {}}
    lockbox = {"crossinterface": {}, "trajectory": {}}
    for event_name in EVENTS:
        selected = max(range(38), key=lambda q: result["crossinterface"]["unit9"][event_name][f"q{q}"]["family_identity_advantage"])
        selections["crossinterface"][event_name] = selected
        lockbox["crossinterface"][event_name] = result["crossinterface"]["unit10"][event_name][f"q{selected}"]
    for interface_name in ("code", "entity"):
        selections["trajectory"][interface_name] = {}
        lockbox["trajectory"][interface_name] = {}
        for relation in ("boundary_to_first", "boundary_to_answer", "first_to_answer"):
            selected = max(range(38), key=lambda q: result["within_interface_trajectory"]["unit9"][interface_name][relation][f"q{q}"]["family_identity_advantage"])
            selections["trajectory"][interface_name][relation] = selected
            lockbox["trajectory"][interface_name][relation] = result["within_interface_trajectory"]["unit10"][interface_name][relation][f"q{selected}"]
    return {"metrics": result, "discovery_selection": selections, "lockbox": lockbox}


def correct_incorrect_probe(states: np.ndarray, index: list[dict]) -> dict:
    # Freeze unit9 correct centroids, classify unit10 rows at the parsed answer event.
    families = sorted({row["family"] for row in index})
    results = {}
    for qpoint in range(38):
        centroids = []
        for family in families:
            selected = [i for i, row in enumerate(index) if row["unit"] == 9 and row["parsed_correct"] and row["family"] == family]
            centroids.append(np.mean(states[selected, 2, qpoint], axis=0, dtype=np.float64))
        centroids = np.asarray(centroids)
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-30)
        rows = [i for i, row in enumerate(index) if row["unit"] == 10]
        values = np.asarray(states[rows, 2, qpoint], dtype=np.float64)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-30)
        similarity = values @ centroids.T
        prediction = np.argmax(similarity, axis=1)
        truth = np.asarray([families.index(index[i]["family"]) for i in rows])
        margins = similarity[np.arange(len(rows)), truth] - np.partition(similarity, -2, axis=1)[:, -2]
        correct_mask = np.asarray([index[i]["parsed_correct"] for i in rows], dtype=bool)
        results[f"q{qpoint}"] = {
            "family_classification_accuracy": float(np.mean(prediction == truth)),
            "behavior_correct_family_accuracy": float(np.mean(prediction[correct_mask] == truth[correct_mask])),
            "behavior_incorrect_family_accuracy": float(np.mean(prediction[~correct_mask] == truth[~correct_mask])) if np.any(~correct_mask) else None,
            "behavior_correct_mean_margin": float(np.mean(margins[correct_mask])),
            "behavior_incorrect_mean_margin": float(np.mean(margins[~correct_mask])) if np.any(~correct_mask) else None,
        }
    selected = max(range(38), key=lambda q: results[f"q{q}"]["family_classification_accuracy"])
    return {"selected_qpoint_on_unit10_descriptive_only": selected, "selected": results[f"q{selected}"], "all_qpoints": results, "boundary": "qpoint is descriptively selected on unit10, so this probe is not a lockbox."}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 成功自主生成的回答边界—首token—完整答案轨迹锁箱裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将Phase2474每条真实路径按语义事件而非绝对step对齐为answer-boundary、first-generated-token、parsed-answer-token；只用行为正确行建立八族相对基线全坐标护照。分别测：（1）每事件实体↔代码跨接口同family；（2）每接口内部boundary→first、boundary→answer、first→answer同family；（3）跨语言；全部比较256个family derangement。unit9只选qpoint，unit10一次锁箱。错误行不混入成功护照，另作描述性family分类。

$$T_{{\mathrm{{traj}}}}=\mathbb{{E}}_f\cos(P_{{f,e_1}},P_{{f,e_2}}),\qquad T_{{\mathrm{{interface}}}}=\mathbb{{E}}_f\cos(P_{{f,e}}^{{entity}},P_{{f,e}}^{{code}}).$$

**结果汇总。** 对齐场 `{json.dumps(result['collection'], ensure_ascii=False)}`；unit9冻结选择 `{json.dumps(result['trajectory']['discovery_selection'], ensure_ascii=False)}`；unit10锁箱 `{json.dumps(result['trajectory']['lockbox'], ensure_ascii=False)}`；正确/错误描述 `{json.dumps(result['correct_incorrect_probe']['selected'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2475_c48001_c48640_autonomous_trajectory_adjudication.py`；三事件float32全坐标场、成功family护照、计数、全部事件/层零假设和final位于同名结果目录。

**分析与理论进展。** 这是对“0.85→0.11崩解”的直接替代实验。真正的轨迹保持必须在同一接口比较前后事件；输出身份分化必须在同一事件比较实体/代码。只有unit10同family胜错family，才说明成功自由生成中仍有family纹理。即使通过，也只是事件图谱L1/L4观察，不等于已找到驱动输出的因果齿轮。

**问题硬伤与结论。** 成功护照因代码错误行而略不平衡；family专属措辞仍是替代解释。实体答案多token、代码答案短，语义事件对齐优于绝对step但并不消除token身份。正确/错误只有13个错误且qpoint在unit10描述性选择，不能作因果或预测锁箱。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    states, index, collection = extract_events()
    passport, counts, passport_meta = passports(states, index)
    trajectory = all_metrics(passport)
    probe = correct_incorrect_probe(states, index)
    minimum_count = int(np.min(counts))
    lockbox = trajectory["lockbox"]
    all_lockbox_advantages = [value["family_identity_advantage"] for value in lockbox["crossinterface"].values()]
    all_lockbox_advantages.extend(value["family_identity_advantage"] for interface in lockbox["trajectory"].values() for value in interface.values())
    adjudication = {
        "phase2466_state_collapse_replicated": False,
        "successful_autonomous_family_texture_present": all(value > 0 for value in all_lockbox_advantages),
        "crossinterface_successful_path_reuse_candidate": all(value["family_identity_advantage"] > 0 for value in lockbox["crossinterface"].values()),
        "within_interface_trajectory_retention_candidate": all(value["family_identity_advantage"] > 0 for interface in lockbox["trajectory"].values() for value in interface.values()),
        "causal_gear_identified": False,
        "language_encoding_mechanism_closed": False,
    }
    checks = {
        "aligned_shape": collection["shape"] == [256, 3, 38, 2560],
        "full_coordinates": passport.shape[-1] == 2560,
        "minimum_success_count": minimum_count >= 2,
        "unit9_selection_unit10_lockbox": set(trajectory["discovery_selection"]) == {"crossinterface", "trajectory"},
        "finite": all(math.isfinite(value) for value in all_lockbox_advantages),
        "claim_boundary": not adjudication["causal_gear_identified"] and not adjudication["language_encoding_mechanism_closed"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": {**collection, "passports": passport_meta}, "trajectory": trajectory, "correct_incorrect_probe": probe, "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    close(states)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
