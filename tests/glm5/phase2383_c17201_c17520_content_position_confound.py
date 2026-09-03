#!/usr/bin/env python3
"""Adjudicate content identity versus source-position/distribution confounds in four models."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2378 = RESULT / "phase2378_c15601_c15920_label_free_binding_contract"
P2379 = RESULT / "phase2379_c15921_c16240_qwen_label_free_full_field"
P2381 = RESULT / "phase2381_c16561_c16880_residual_component_routing"
P2382 = RESULT / "phase2382_c16881_c17200_crossmodel_label_free_binding"
OUT = RESULT / "phase2383_c17201_c17520_content_position_confound"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2378 / "material/label_free_natural_binding.jsonl"
PHASE = 2383
CAMPAIGN = "C17201-C17520"
MODELS = ("qwen4b", "qwen14b", "glm4", "deepseek7b")

sys.path.insert(0, str(TESTS))
import phase2380_c16241_c16560_object_slot_progress_adjudication as adjudicate  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def panel_source_indices() -> list[int]:
    return [int(row["source_index"]) for row in read_rows(P2381 / "index/component_panel_rows.jsonl")]


def model_data(key: str, all_rows: list[dict], panel: list[int]):
    if key == "qwen4b":
        source_map = np.load(P2379 / "raw/qwen4b_source_sentence_end.float16.npy", mmap_mode="r")
        output_map = np.load(P2379 / "raw/qwen4b_output_progress_anchors.float16.npy", mmap_mode="r")
        qpoint = 11
        source = np.asarray(source_map[panel, :, qpoint], dtype=np.float32)
        output = np.asarray(output_map[panel, :, 0, qpoint], dtype=np.float32)
        close(source_map); close(output_map)
        return [all_rows[i] for i in panel], source, output, qpoint
    base = P2382 / key
    rows = read_rows(base / "material/rows.jsonl")
    final = json.loads((base / "analysis/final.json").read_text(encoding="utf-8")); qpoint = int(final["object_matching"]["qpoint"])
    source_map = np.load(base / "raw/source_sentence_end.float16.npy", mmap_mode="r")
    output_map = np.load(base / "raw/output_pre_sentence.float16.npy", mmap_mode="r")
    source = np.asarray(source_map[:, :, qpoint], dtype=np.float32); output = np.asarray(output_map[:, :, qpoint], dtype=np.float32)
    close(source_map); close(output_map); return rows, source, output, qpoint


def split_indices(rows: list[dict]) -> dict[str, np.ndarray]:
    return {part: np.asarray([i for i, row in enumerate(rows) if row["partition"] == part])
            for part in ("discovery", "confirmation", "fresh_joint_lockbox")}


def slot_templates(source: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.stack([source[train, slot].mean(0) for slot in range(4)])
    std = np.stack([source[train, slot].std(0) + 1e-4 for slot in range(4)])
    return mean, std


def transform_source(source: np.ndarray, kind: str, templates: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = templates
    if kind == "original": return source.copy()
    if kind == "position_template": return np.broadcast_to(mean[None], source.shape).copy()
    if kind == "slot_centered": return source - mean[None]
    if kind == "slot_zscore": return (source - mean[None]) / std[None]
    if kind == "coordinate_sorted": return np.sort(source, axis=-1)
    if kind == "row_specific_coordinate_permutation":
        result = np.empty_like(source)
        for row in range(len(source)):
            for slot in range(4):
                perm = np.random.default_rng(23830000 + row * 4 + slot).permutation(source.shape[-1])
                result[row, slot] = source[row, slot, perm]
        return result
    raise ValueError(kind)


def fit_params(source: np.ndarray, output: np.ndarray, rows: list[dict], train: np.ndarray, labels: np.ndarray):
    result = {}
    for target_slot in range(4):
        for reverse in (False, True):
            use = [i for i in train if bool(rows[int(i)]["reverse"]) == reverse]
            x = np.stack([source[int(i), labels[int(i), target_slot]] for i in use])
            y = np.stack([output[int(i), target_slot] for i in use])
            result[(target_slot, reverse)] = adjudicate.fit_diagonal(x, y)
    return result


def accuracy(source: np.ndarray, output: np.ndarray, rows: list[dict], indices: np.ndarray,
             labels: np.ndarray, params: dict) -> float:
    correct = total = 0
    for i in indices:
        row = rows[int(i)]
        for target_slot in range(4):
            a, b = params[(target_slot, bool(row["reverse"]))]
            distances = np.square(source[int(i)] * a + b - output[int(i), target_slot]).mean(1)
            correct += int(int(distances.argmin()) == labels[int(i), target_slot]); total += 1
    return correct / total


def donor_source(source: np.ndarray, rows: list[dict], lock: np.ndarray) -> tuple[np.ndarray, dict]:
    result = source.copy(); swaps = []
    groups: dict[tuple, list[int]] = {}
    for i in lock:
        row = rows[int(i)]; groups.setdefault((row["language"], row["surface"], row["source_index"]), []).append(int(i))
    for members in groups.values():
        ordered = sorted(members, key=lambda i: (rows[i]["family"], rows[i]["unit"], rows[i]["reverse"]))
        donors = ordered[1:] + ordered[:1]
        for target, donor in zip(ordered, donors):
            if rows[target]["family"] == rows[donor]["family"]:
                donor = donors[(donors.index(donor) + 7) % len(donors)]
            result[target] = source[donor]
            swaps.append({"target": rows[target]["case_id"], "donor": rows[donor]["case_id"],
                          "same_source_perm": rows[target]["source_perm"] == rows[donor]["source_perm"],
                          "different_content": (rows[target]["family"], rows[target]["unit"]) != (rows[donor]["family"], rows[donor]["unit"])})
    return result, {"swaps": len(swaps), "same_source_perm": all(x["same_source_perm"] for x in swaps),
                    "different_content": all(x["different_content"] for x in swaps)}


def analyze_model(key: str, rows: list[dict], source: np.ndarray, output: np.ndarray, qpoint: int) -> dict:
    splits = split_indices(rows); train, confirm, lock = splits["discovery"], splits["confirmation"], splits["fresh_joint_lockbox"]
    labels = adjudicate.slot_labels(rows); templates = slot_templates(source, train); controls = {}
    for kind in ("original", "position_template", "slot_centered", "slot_zscore", "coordinate_sorted", "row_specific_coordinate_permutation"):
        transformed = transform_source(source, kind, templates); params = fit_params(transformed, output, rows, train, labels)
        controls[kind] = {"confirmation_accuracy": accuracy(transformed, output, rows, confirm, labels, params),
                          "lockbox_accuracy": accuracy(transformed, output, rows, lock, labels, params)}
    original_params = fit_params(source, output, rows, train, labels); donor, donor_audit = donor_source(source, rows, lock)
    controls["cross_row_same_slot_donor"] = {"lockbox_accuracy": accuracy(donor, output, rows, lock, labels, original_params), **donor_audit}
    original = controls["original"]["lockbox_accuracy"]; donor_score = controls["cross_row_same_slot_donor"]["lockbox_accuracy"]
    position_score = controls["position_template"]["lockbox_accuracy"]; shuffled = controls["row_specific_coordinate_permutation"]["lockbox_accuracy"]
    verdict = {"content_specific_binding_supported": original - max(position_score, donor_score) >= 0.10,
               "fixed_coordinate_correspondence_supported": original - shuffled >= 0.10,
               "distributed_position_or_distribution_confound": max(position_score, donor_score, shuffled) >= original - 0.03}
    return {"model": key, "qpoint": qpoint, "dimension": source.shape[-1], "controls": controls, "verdict": verdict}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 四模型句内容－来源位置－坐标分布混淆总裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2382各模型confirmation选定的句前层，在相同锁箱中把来源候选依次替换为：原场、每个来源槽的训练均值模板、槽位中心化场、槽位z-score场、坐标值排序场、每个样本独立随机坐标排列场，以及保持来源槽位但换成另一内容unit/family的跨样本donor。所有变换均用discovery拟合相同的逐坐标条件仿射，confirmation不参与参数估计，fresh联合锁箱裁决。

$$H^{{src}}=\mu_{{slot}}+R_{{content}},\qquad
H^{{template}}_s=\mu_s,\qquad
H^{{donor}}_{{r,s}}=H^{{src}}_{{r',s}},\quad r'\ne r.$$

**结果汇总。** 模型裁决 `{json.dumps({key: value['controls'] for key, value in result['models'].items()}, ensure_ascii=False)}`；结论矩阵 `{json.dumps(result['verdict'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2383_c17201_c17520_content_position_confound.py`；逐模型完整控制结果位于 `tests/glm5/result/phase2383_c17201_c17520_content_position_confound`。本Phase只读取已有全场，未新增压缩HiddenState。

**理论进展、问题硬伤与结论。** 这个Phase专门防止把“正确来源槽可从候选中选出”误写成“句内容对象被搬运”。position-template或same-slot donor若接近原分数，说明来源句末端的因果位置/长度纹理足以解释匹配；slot-centered若保留，才支持内容残差。样本独立坐标置乱若不降，固定坐标解释被直接否决。下一Phase不再使用上下文化来源句末端，而把每个句子放进相同中性局部包装独立编码，并与纯词嵌入均值基线竞争，继续自动分离内容与位置。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    all_rows = [row for row in read_rows(MATERIAL) if row["task"] == "exact_copy"]; panel = panel_source_indices(); models = {}
    for key in MODELS:
        rows, source, output, qpoint = model_data(key, all_rows, panel)
        models[key] = analyze_model(key, rows, source, output, qpoint)
        print(f"[phase2383] {key} complete", flush=True)
    verdict = {"content_specific_models": [key for key, value in models.items() if value["verdict"]["content_specific_binding_supported"]],
               "fixed_coordinate_models": [key for key, value in models.items() if value["verdict"]["fixed_coordinate_correspondence_supported"]],
               "confounded_models": [key for key, value in models.items() if value["verdict"]["distributed_position_or_distribution_confound"]]}
    checks = {"all_models": set(models) == set(MODELS), "all_controls": all(len(value["controls"]) == 7 for value in models.values()),
              "all_donors_valid": all(value["controls"]["cross_row_same_slot_donor"]["same_source_perm"] and
                                      value["controls"]["cross_row_same_slot_donor"]["different_content"] for value in models.values()),
              "finite": all(math.isfinite(control["lockbox_accuracy"]) for value in models.values() for control in value["controls"].values())}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "verdict": verdict,
              "checks": checks, "all_checks_passed": all(checks.values()),
              "next_stage": {"same_overall_goal": True, "same_immediate_target": True,
                             "action": "automatically continue with position-normalized isolated-sentence content fields"}}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
