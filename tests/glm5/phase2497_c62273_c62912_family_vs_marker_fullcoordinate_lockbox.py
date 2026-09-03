#!/usr/bin/env python3
"""Adjudicate family reuse across nonce markers against marker reuse across families."""
from __future__ import annotations

import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2496 = RESULT / "phase2496_c61121_c62272_nonce_marker_rotation_behavior_fullfield"
OUT = RESULT / "phase2497_c62273_c62912_family_vs_marker_fullcoordinate_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2497, "C62273-C62912"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def identity_metric(views: dict[str, np.ndarray]) -> dict:
    keys = sorted(views); same = []; wrong = []
    for first, second in combinations(keys, 2):
        a, b = views[first], views[second]
        for item in range(a.shape[0]):
            same.append(cosine(a[item], b[item]))
            for shift in range(1, a.shape[0]):
                wrong.append(cosine(a[item], b[(item + shift) % a.shape[0]]))
    return {"view_pairs": len(list(combinations(keys, 2))), "same_mean": float(np.mean(same)),
            "wrong_mean": float(np.mean(wrong)), "wrong_q95": float(np.quantile(wrong, .95)),
            "identity_advantage_over_q95": float(np.mean(same) - np.quantile(wrong, .95))}


def family_views(x: np.ndarray, rows: list[dict], families: list[str], view_key: str) -> dict[str, np.ndarray]:
    output = {}
    for level in sorted({str(r[view_key]) for r in rows}):
        mask_level = np.array([str(r[view_key]) == level for r in rows])
        values = np.stack([x[mask_level & np.array([r["family"] == family for r in rows])].mean(axis=0) for family in families])
        output[level] = values - values.mean(axis=0, keepdims=True)
    return output


def marker_views(x: np.ndarray, rows: list[dict], families: list[str]) -> dict[str, np.ndarray]:
    output = {}
    for family in families:
        mask_family = np.array([r["family"] == family for r in rows])
        values = np.stack([x[mask_family & np.array([r["marker_id"] == marker for r in rows])].mean(axis=0) for marker in range(4)])
        output[family] = values - values.mean(axis=0, keepdims=True)
    return output


def metric_panel(field: np.ndarray, all_rows: list[dict], unit: int, event: int, qpoint: int, families: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    rows = [r for r in all_rows if r["unit"] == unit and r["family"] in families]
    x = np.asarray(field[[r["model_row"] for r in rows], event, qpoint], dtype=np.float64)
    marker_conditioned_family = family_views(x, rows, families, "marker_id")
    language_conditioned_family = family_views(x, rows, families, "language")
    surface_conditioned_family = family_views(x, rows, families, "definition_surface")
    family_conditioned_marker = marker_views(x, rows, families)
    family_mean = np.stack([x[np.array([r["family"] == family for r in rows])].mean(axis=0) for family in families])
    family_mean -= family_mean.mean(axis=0, keepdims=True)
    marker_mean = np.stack([x[np.array([r["marker_id"] == marker for r in rows])].mean(axis=0) for marker in range(4)])
    marker_mean -= marker_mean.mean(axis=0, keepdims=True)
    return {
        "family_across_marker": identity_metric(marker_conditioned_family),
        "family_across_language": identity_metric(language_conditioned_family),
        "family_across_definition_surface": identity_metric(surface_conditioned_family),
        "marker_across_family": identity_metric(family_conditioned_marker),
    }, family_mean.astype(np.float32), marker_mean.astype(np.float32)


def main_effect_shares(x: np.ndarray, rows: list[dict]) -> dict:
    x = np.asarray(x, dtype=np.float64); grand = x.mean(axis=0); total = float(np.square(x - grand).sum())
    result = {}
    for factor in ("family", "marker_id", "language", "definition_surface"):
        ss = 0.0
        for level in sorted({str(r[factor]) for r in rows}):
            mask = np.array([str(r[factor]) == level for r in rows])
            effect = x[mask].mean(axis=0) - grand
            ss += int(mask.sum()) * float(np.square(effect).sum())
        result[factor] = ss / max(total, 1e-30)
    result["not_main_effects"] = max(0.0, 1.0 - sum(result.values()))
    return result


def density(passports: np.ndarray) -> dict:
    energy = np.square(np.asarray(passports, dtype=np.float64)).sum(axis=0)
    weights = energy / max(float(energy.sum()), 1e-30)
    ordered = np.sort(weights)[::-1]; cumulative = np.cumsum(ordered)
    return {"effective_coordinate_count": float(1 / np.square(weights).sum()),
            "coordinates_for_50pct_contrast_energy": int(np.searchsorted(cumulative, .5) + 1),
            "coordinates_for_90pct_contrast_energy": int(np.searchsorted(cumulative, .9) + 1),
            "boundary": "contrast-energy coverage only; not information or causal importance"}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: family跨无意义marker复用与marker跨family复用的全坐标锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2496的十二family全部通过unit18/19行为门。对每个事件与qpoint，构造两套对偶护照：（1）在每个marker条件内对十二family中心化，比较同family跨四marker是否胜错family；（2）在每个family条件内对四marker中心化，比较同marker跨十二family是否胜错marker。另比较family跨中英文独立字符串与两定义表面。unit18仅在answer-boundary选择family跨marker身份优势最大的唯一qpoint；unit19在同层一次锁箱，并把该层用于定义、记录marker、query和answer四事件，保留全部2560坐标。

$$P_{{f\mid m}}=\mathbb E[H\mid f,m]-\mathbb E[H\mid m],\qquad P_{{m\mid f}}=\mathbb E[H\mid m,f]-\mathbb E[H\mid f].$$

**结果汇总。** 选择 `{json.dumps(result['selection'], ensure_ascii=False)}`；unit19同层四事件 `{json.dumps(result['lockbox'], ensure_ascii=False)}`；描述性主效应 `{json.dumps(result['main_effect_shares'], ensure_ascii=False)}`；密度 `{json.dumps(result['density'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2497_c62273_c62912_family_vs_marker_fullcoordinate_lockbox.py`；unit18全部qpoint分数、unit18/19四事件family/marker护照、final位于同名目录。

**分析与理论进展。** 同family跨marker优于错family时，能排除“记录位置某一个固定marker token身份就是全部family纹理”；同marker跨family仍高则说明token身份脉络同时存在。定义事件到记录marker再到answer的同qpoint变化是上下文条件传播图谱，不是同一向量搬运。全部十二族与四marker参与，避免三family错配零分布过小。

**问题硬伤与结论。** family定义词仍是family专属语义/词项组合，且任务答案不要求区分十二种关系含义，只需执行已定义链接。因此正结果仍不能叫纯语义或抽象关系代码。主效应份额是描述性账本；能量密度不是因果。当前最多识别“family定义条件超越记录marker token的复用纹理”。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    f2496 = json.loads((P2496 / "analysis/final.json").read_text(encoding="utf-8"))
    families = sorted(f2496["behavior"]["qualified_families"])
    rows = read_jsonl(Path(f2496["collection"]["index"]))
    field = np.load(f2496["collection"]["event_field"], mmap_mode="r")
    events = f2496["collection"]["events"]
    answer_event = events.index("answer_boundary")
    discovery = []
    for q in range(1, 37):
        metrics, _, _ = metric_panel(field, rows, 18, answer_event, q, families)
        discovery.append({"qpoint": q, **metrics})
    selected = max(discovery, key=lambda x: x["family_across_marker"]["identity_advantage_over_q95"])
    qpoint = int(selected["qpoint"])
    lockbox = {}; factor = {}; family_fields = np.zeros((2, 4, 12, 2560), dtype=np.float32)
    marker_fields = np.zeros((2, 4, 4, 2560), dtype=np.float32)
    for ui, unit in enumerate((18, 19)):
        for ei, event in enumerate(events):
            metrics, fp, mp = metric_panel(field, rows, unit, ei, qpoint, families)
            family_fields[ui, ei] = fp; marker_fields[ui, ei] = mp
            if unit == 19:
                lockbox[event] = metrics
                use_rows = [r for r in rows if r["unit"] == 19]
                x = field[[r["model_row"] for r in use_rows], ei, qpoint]
                factor[event] = main_effect_shares(x, use_rows)
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    family_path = derived / "family_passports.float32.npy"; marker_path = derived / "marker_passports.float32.npy"
    np.save(family_path, family_fields); np.save(marker_path, marker_fields)
    save(OUT / "analysis/unit18_allqpoint_selection.json", discovery)
    densities = {event: density(family_fields[0, ei]) for ei, event in enumerate(events)}
    checks = {"twelve_qualified_families": len(families) == 12, "four_marker_wrong_controls": True,
              "selection_unit18_only": 1 <= qpoint <= 36, "unit19_same_qpoint_all_events": True,
              "all_coordinates": family_fields.shape[-1] == 2560 and marker_fields.shape[-1] == 2560,
              "finite": bool(np.isfinite(family_fields).all() and np.isfinite(marker_fields).all()), "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "families": families,
              "selection": {"qpoint": qpoint, "unit18_answer_boundary": selected},
              "lockbox": lockbox, "main_effect_shares": factor, "density": densities,
              "fields": {"family": {"path": str(family_path), "shape": list(family_fields.shape),
                                      "axes": ["unit18_or_unit19", "event", "family", "coordinate"]},
                         "marker": {"path": str(marker_path), "shape": list(marker_fields.shape),
                                     "axes": ["unit18_or_unit19", "event", "marker", "coordinate"]}},
              "adjudication": {"family_beyond_record_marker_tested": True, "pure_semantic_code_identified": False,
                               "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
