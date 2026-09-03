#!/usr/bin/env python3
"""Test causal-chain texture across canonical versus interleaved fact surfaces."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2481 = RESULT / "phase2481_c51841_c52480_chain_node_edge_basic_atlas"
OUT = RESULT / "phase2484_c53441_c54080_causal_surface_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2484, "C53441-C54080", 2560


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def cos(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.asarray(first, dtype=np.float64); b = np.asarray(second, dtype=np.float64)
    return np.sum(a * b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-30)


def compare(first: np.ndarray, second: np.ndarray) -> dict:
    # Arrays [language,node,coordinate]. Wrong-node cyclic shifts preserve language
    # and surface but break the relative chain position.
    coordinate = float(np.mean(cos(first, second)))
    null = [float(np.mean(cos(first, np.roll(second, shift, axis=1)))) for shift in (1, 2, 3)]
    return {"coordinate": coordinate, "wrong_node_mean": float(np.mean(null)), "wrong_node_max": float(np.max(null)), "same_node_advantage": coordinate - float(np.max(null))}


def build_surface_fields() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    final = json.loads((P2481 / "analysis/final.json").read_text(encoding="utf-8"))
    rows = read_jsonl(P2481 / "index/node_event_rows.jsonl")
    prompt = np.load(final["collection"]["prompt_nodes"]["path"], mmap_mode="r")
    generated = np.load(final["collection"]["generated_nodes"]["path"], mmap_mode="r")
    fields = np.zeros((3, 2, 2, 4, 38, DIM), dtype=np.float32)
    paired = np.zeros_like(fields)
    generated_fields = np.full_like(fields, np.nan)
    counts = np.zeros((3, 2, 2), dtype=np.int32)
    try:
        for ui, unit in enumerate((11, 12, 13)):
            for li, language in enumerate(("en", "zh")):
                for si, surface in enumerate((0, 2)):
                    selected = [i for i, row in enumerate(rows) if row["unit"] == unit and row["language"] == language and row["family"] == "causal" and row["surface"] == surface]
                    counts[ui, li, si] = len(selected)
                    fields[ui, li, si] = np.mean(prompt[selected, 0], axis=0, dtype=np.float64)
                    paired[ui, li, si] = np.mean(prompt[selected, 0] - prompt[selected, 1], axis=0, dtype=np.float64)
                    for node in range(4):
                        valid = [i for i in selected if np.isfinite(generated[i, node]).all()]
                        if valid: generated_fields[ui, li, si, node] = np.mean(generated[valid, node], axis=0, dtype=np.float64)
    finally:
        close(prompt); close(generated)
    paths = {}
    for name, value in (("prompt", fields), ("main_minus_distractor", paired), ("generated", generated_fields)):
        path = OUT / f"derived/causal_surface_{name}.float32.npy"; path.parent.mkdir(parents=True, exist_ok=True); np.save(path, value); paths[name] = str(path)
    count_path = OUT / "derived/causal_surface_counts.int32.npy"; np.save(count_path, counts)
    return {"fields": paths, "counts": str(count_path), "shape": list(fields.shape), "axes": ["unit11/12/13", "language", "surface0/2", "node0-3", "qpoint", "coordinate"]}, fields, paired, generated_fields


def analyze(fields: np.ndarray, paired: np.ndarray, generated: np.ndarray) -> dict:
    metrics = {"prompt_main": {}, "main_minus_distractor": {}, "generated_main": {}}
    arrays = {"prompt_main": fields, "main_minus_distractor": paired, "generated_main": generated}
    for name, values in arrays.items():
        for ui, unit in enumerate((11, 12, 13)):
            metrics[name][f"unit{unit}"] = {}
            for qpoint in range(38):
                first, second = values[ui, :, 0, :, qpoint], values[ui, :, 1, :, qpoint]
                if np.isfinite(first).all() and np.isfinite(second).all(): metrics[name][f"unit{unit}"][f"q{qpoint}"] = compare(first, second)
                else: metrics[name][f"unit{unit}"][f"q{qpoint}"] = None
    selection = {}; lockbox = {}
    for name in ("prompt_main", "main_minus_distractor"):
        candidates = [q for q in range(38) if metrics[name]["unit12"][f"q{q}"] is not None]
        selected = max(candidates, key=lambda q: metrics[name]["unit12"][f"q{q}"]["same_node_advantage"])
        selection[name] = selected; lockbox[name] = metrics[name]["unit13"][f"q{selected}"]
    # One unit12 generated path missed nodes under the hooked BF16 run.  Do not
    # select on incomplete generated data: reuse the prompt-main qpoint and test
    # the complete unit13 generated trajectories once.
    selection["generated_main"] = selection["prompt_main"]
    lockbox["generated_main"] = metrics["generated_main"]["unit13"][f"q{selection['generated_main']}"]
    qpoint = selection["main_minus_distractor"]
    energy0 = np.sum(np.square(paired[1, :, 0, :, qpoint].astype(np.float64)), axis=(0, 1))
    energy2 = np.sum(np.square(paired[1, :, 1, :, qpoint].astype(np.float64)), axis=(0, 1))
    order = np.argsort(-energy0); np.save(OUT / "derived/surface0_coordinate_order.int32.npy", order.astype(np.int32))
    energy_corr_discovery = float(np.corrcoef(energy0, energy2)[0, 1])
    lock0 = np.sum(np.square(paired[2, :, 0, :, qpoint].astype(np.float64)), axis=(0, 1))
    lock2 = np.sum(np.square(paired[2, :, 1, :, qpoint].astype(np.float64)), axis=(0, 1))
    energy_corr_lockbox = float(np.corrcoef(lock0, lock2)[0, 1])
    return {"all_qpoints": metrics, "unit12_selection": selection, "unit13_lockbox": lockbox, "coordinate_energy": {"selected_qpoint": qpoint, "unit12_surface0_surface2_correlation": energy_corr_discovery, "unit13_surface0_surface2_correlation": energy_corr_lockbox}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: causal三跳链规范顺序—交织顺序的节点与查询路径全坐标锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 使用Phase2480中唯一同时通过两个表面的causal链，固定相同relation family与三跳结构，比较surface0（主链后接干扰链）和surface2（两链事实交织）。分别对prompt主链节点、main-minus-distractor查询路径差、实际generated节点，在每个qpoint比较相同相对node位置；将surface2节点循环错位1/2/3格作为基本零对照。prompt与主/干扰差由unit12选择qpoint、unit13锁箱；因unit12有一条hook生成缺失节点，generated不在不完整数据上选层，而继承prompt选层后只在完整unit13测试一次。另比较两个surface逐坐标能量图的Pearson相关。全程保留2560坐标。

$$S_q=\mathbb E_{{\lambda,k}}\cos(F_{{s0,\lambda,k}}^q,F_{{s2,\lambda,k}}^q),\qquad S_q^{{null}}=\max_{{r=1,2,3}}\mathbb E\cos(F_{{s0,k}}^q,F_{{s2,k+r}}^q).$$

**结果汇总。** 原场 `{json.dumps(result['collection'], ensure_ascii=False)}`；unit12选择 `{json.dumps(result['analysis']['unit12_selection'], ensure_ascii=False)}`；unit13锁箱 `{json.dumps(result['analysis']['unit13_lockbox'], ensure_ascii=False)}`；能量 `{json.dumps(result['analysis']['coordinate_energy'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2484_c53441_c54080_causal_surface_lockbox.py`；causal双表面prompt、main-minus-distractor、generated全坐标场、计数、冻结surface0排序及全部qpoint结果位于同名目录。

**分析与理论进展。** 该Phase不是再按结果挑一种数学工具，而是直接攻击当前最严重的surface混杂。如果unit13相同node显著胜错node，说明部分相对位置纹理能跨事实交织保存；若主/干扰差也通过，证据更接近“被查询路径条件化”，但仍不是因果齿轮。

**问题硬伤与结论。** 只有causal一个family，不能外推到普遍知识链；surface0/2同时改变节点排列与事实交织，尚非单因素；相同英文节点跨中英文重复，token身份仍存在。坐标能量相关只描述纹理，不说明方向或功能。下一步应构造更多family均拥有同一组正交surface的材料，而不是在单family上闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    collection, fields, paired, generated = build_surface_fields(); analysis = analyze(fields, paired, generated)
    lockbox = analysis["unit13_lockbox"]
    checks = {
        "shape": collection["shape"] == [3, 2, 2, 4, 38, 2560],
        "all_coordinates": collection["shape"][-1] == 2560,
        "three_metrics": set(lockbox) == {"prompt_main", "main_minus_distractor", "generated_main"},
        "finite": all(value is not None and all(math.isfinite(number) for number in value.values()) for value in lockbox.values()),
        "unit12_selection_unit13_lockbox": len(analysis["unit12_selection"]) == 3,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "analysis": analysis,
        "adjudication": {
            "causal_surface_same_node_reuse_candidate": all(value["same_node_advantage"] > 0 for value in lockbox.values()),
            "universal_crossfamily_chain_rule": False, "query_path_causal_gear_identified": False,
            "language_encoding_mechanism_closed": False,
        },
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps({"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "analysis_summary": {"unit12_selection": analysis["unit12_selection"], "unit13_lockbox": lockbox, "coordinate_energy": analysis["coordinate_energy"]}, "adjudication": result["adjudication"], "checks": checks, "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
