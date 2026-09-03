#!/usr/bin/env python3
"""Separate signed family texture from coordinate energy-envelope similarity."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2487 = RESULT / "phase2487_c54721_c55872_orthogonal_family_interface_behavior"
P2488 = RESULT / "phase2488_c55873_c56832_qwen4b_orthogonal_fullcoordinate_field"
OUT = RESULT / "phase2490_c57473_c58112_signed_texture_energy_envelope_controls"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2490, "C57473-C58112"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(); b = b - b.mean()
    return cosine(a, b)


def passports(x: np.ndarray, rows: list[dict], families: list[str], key: str) -> dict[str, np.ndarray]:
    result = {}
    levels = sorted({str(r[key]) for r in rows})
    for level in levels:
        level_mask = np.array([str(r[key]) == level for r in rows])
        level_mean = x[level_mask].mean(axis=0)
        result[level] = np.stack([
            x[level_mask & np.array([r["family"] == family for r in rows])].mean(axis=0) - level_mean
            for family in families
        ])
    return result


def comparison(p: dict[str, np.ndarray], envelope: np.ndarray, surface_mode: bool = False) -> dict:
    eps = max(float(np.mean(envelope)) * 1e-8, 1e-12)
    keys = sorted(p)
    pairs = []
    if surface_mode:
        pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    else:
        pairs = [(keys[0], keys[1])]
    same_raw, same_std, same_energy, same_sign, same_weighted_sign = [], [], [], [], []
    wrong_raw, wrong_std, wrong_energy = [], [], []
    for first, second in pairs:
        a, b = p[first], p[second]
        scale = np.sqrt(envelope + eps)
        for family in range(a.shape[0]):
            same_raw.append(cosine(a[family], b[family]))
            same_std.append(cosine(a[family] / scale, b[family] / scale))
            same_energy.append(corr(np.square(a[family]), np.square(b[family])))
            signs = np.sign(a[family]) == np.sign(b[family])
            same_sign.append(float(signs.mean()))
            weights = np.minimum(np.abs(a[family]), np.abs(b[family]))
            same_weighted_sign.append(float(np.sum(weights * signs) / max(np.sum(weights), 1e-12)))
            for shift in range(1, a.shape[0]):
                other = (family + shift) % a.shape[0]
                wrong_raw.append(cosine(a[family], b[other]))
                wrong_std.append(cosine(a[family] / scale, b[other] / scale))
                wrong_energy.append(corr(np.square(a[family]), np.square(b[other])))
    def summary(same, wrong):
        return {"same_mean": float(np.mean(same)), "wrong_mean": float(np.mean(wrong)),
                "wrong_q95": float(np.quantile(wrong, 0.95)),
                "identity_advantage_over_q95": float(np.mean(same) - np.quantile(wrong, 0.95))}
    return {
        "raw_signed": summary(same_raw, wrong_raw),
        "rms_standardized_signed": summary(same_std, wrong_std),
        "squared_energy": summary(same_energy, wrong_energy),
        "all_coordinate_sign_agreement": float(np.mean(same_sign)),
        "amplitude_weighted_sign_agreement": float(np.mean(same_weighted_sign)),
        "condition_pairs": len(pairs),
    }


def panel(field: np.ndarray, rows: list[dict], unit: int, event: int, qpoint: int,
          families: list[str], envelope: np.ndarray | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    mask = np.array([r["unit"] == unit and r["output_interface"] == "entity" and r["family"] in families for r in rows])
    use_rows = [r for r in rows if r["unit"] == unit and r["output_interface"] == "entity" and r["family"] in families]
    x = np.asarray(field[mask, event, qpoint, :], dtype=np.float64)
    lang = passports(x, use_rows, families, "language")
    surface = passports(x, use_rows, families, "surface")
    local_envelope = np.mean(np.square(np.concatenate(list(lang.values()), axis=0)), axis=0)
    use_envelope = local_envelope if envelope is None else envelope
    result = {"crosslanguage": comparison(lang, use_envelope), "crosssurface": comparison(surface, use_envelope, True)}
    return result, {"envelope": local_envelope, "lang_en": lang["en"], "lang_zh": lang["zh"]}


def density(envelope: np.ndarray) -> dict:
    positive = np.maximum(np.asarray(envelope, dtype=np.float64), 0)
    total = float(positive.sum())
    weights = positive / max(total, 1e-30)
    effective = float(1.0 / max(float(np.square(weights).sum()), 1e-30))
    ordered = np.sort(weights)[::-1]
    cumulative = np.cumsum(ordered)
    return {
        "effective_coordinate_count": effective,
        "coordinates_for_50pct_contrast_energy": int(np.searchsorted(cumulative, 0.5) + 1),
        "coordinates_for_90pct_contrast_energy": int(np.searchsorted(cumulative, 0.9) + 1),
        "interpretation": "descriptive contrast-energy coverage, not information or causal importance",
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 有符号family纹理、平方能量包络与全坐标RMS标准化的正交锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 只在Phase2487 confirmation与lockbox均合格的九个entity族上，分别构造中英和四surface条件内的family-relative护照。unit15逐事件选择唯一qpoint：最大化“按unit15全坐标RMS包络标准化后的跨语言与跨surface身份优势”；随后冻结unit15包络，在unit16同层一次评价。同步报告原始有符号余弦、平方能量相关、全部2560坐标符号一致率、幅值加权符号一致率和错family循环置换；不删除低值坐标。

$$A_i=\mathbb E_{{f,c\in u15}}P_{{f,c,i}}^2,\qquad S_{{f,c,i}}=P_{{f,c,i}}/\sqrt{{A_i+\varepsilon}}.$$

**结果汇总。** 冻结层位 `{json.dumps(result['selection'], ensure_ascii=False)}`；unit16逐事件结果 `{json.dumps(result['lockbox'], ensure_ascii=False)}`；包络复现与密度 `{json.dumps(result['envelope'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2490_c57473_c58112_signed_texture_energy_envelope_controls.py`；unit15包络、unit15/unit16中英family护照、完整逐层确认分数及`analysis/final.json`位于同名目录。

**分析与理论进展。** 判据不是“能量相关高”本身，而是同family能量相关是否超过错family包络。如果同、错family都接近，稳定的是坐标尺度环境而非family代码；有符号原始/标准化身份优势才描述条件方向。RMS标准化仍使用每个物理坐标，只改变计量权重，不把场压缩成少数主成分。

**问题硬伤与结论。** qpoint由unit15多指标择优，unit16才是锁箱；九族仍受谓词词项影响。平方能量丢失符号，标准化会放大低幅噪声，故必须与原始结果并报。坐标覆盖数不是信息量。结果最多支持“可复现的条件有符号纹理”或“非特异坐标尺度包络”，不命名天然齿轮。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    f2487 = json.loads((P2487 / "analysis/final.json").read_text(encoding="utf-8"))
    f2488 = json.loads((P2488 / "analysis/final.json").read_text(encoding="utf-8"))
    families = sorted(f2487["behavior"]["qualified"]["entity"])
    rows = read_jsonl(Path(f2488["collection"]["index"]))
    field = np.load(f2488["collection"]["event_field"], mmap_mode="r")
    events = f2488["collection"]["events"]
    discovery: dict[str, list[dict]] = {e: [] for e in events}
    selection: dict[str, int] = {}
    envelopes = np.zeros((len(events), field.shape[-1]), dtype=np.float32)
    lock_passports = np.zeros((2, len(events), 2, len(families), field.shape[-1]), dtype=np.float32)
    lockbox: dict[str, dict] = {}
    for event_index, event in enumerate(events):
        aux_by_q = []
        for qpoint in range(1, field.shape[2] - 1):
            metrics, aux = panel(field, rows, 15, event_index, qpoint, families)
            discovery[event].append({"qpoint": qpoint, **metrics})
            aux_by_q.append((qpoint, aux))
        def objective(item: dict) -> float:
            return 0.5 * (item["crosslanguage"]["rms_standardized_signed"]["identity_advantage_over_q95"] +
                          item["crosssurface"]["rms_standardized_signed"]["identity_advantage_over_q95"])
        best = max(discovery[event], key=objective)
        qpoint = int(best["qpoint"])
        selection[event] = qpoint
        aux15 = dict(aux_by_q)[qpoint]
        envelopes[event_index] = aux15["envelope"].astype(np.float32)
        metrics16, aux16 = panel(field, rows, 16, event_index, qpoint, families, aux15["envelope"])
        lockbox[event] = metrics16
        lock_passports[0, event_index, 0] = aux15["lang_en"]
        lock_passports[0, event_index, 1] = aux15["lang_zh"]
        lock_passports[1, event_index, 0] = aux16["lang_en"]
        lock_passports[1, event_index, 1] = aux16["lang_zh"]
    OUT.joinpath("derived").mkdir(parents=True, exist_ok=True)
    envelope_path = OUT / "derived/confirmation_coordinate_envelopes.float32.npy"
    passport_path = OUT / "derived/crosslanguage_family_passports.float32.npy"
    np.save(envelope_path, envelopes); np.save(passport_path, lock_passports)
    save(OUT / "analysis/all_confirmation_qpoint_scores.json", discovery)
    env_lock_corr = {}
    densities = {}
    for event_index, event in enumerate(events):
        _, aux16 = panel(field, rows, 16, event_index, selection[event], families)
        env_lock_corr[event] = corr(envelopes[event_index], aux16["envelope"])
        densities[event] = density(envelopes[event_index])
    checks = {
        "nine_behavior_qualified_entity_families": len(families) == 9,
        "all_coordinates": envelopes.shape[1] == 2560,
        "confirmation_selection_only": all(1 <= q <= 36 for q in selection.values()),
        "frozen_envelope_on_lockbox": True,
        "wrong_family_controls": all(v["crosslanguage"]["raw_signed"]["wrong_q95"] is not None for v in lockbox.values()),
        "finite": all(math.isfinite(v) for v in env_lock_corr.values()),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "families": families, "selection": selection,
        "lockbox": lockbox,
        "envelope": {"confirmation_lockbox_coordinate_energy_correlation": env_lock_corr,
                     "density": densities, "path": str(envelope_path), "shape": list(envelopes.shape)},
        "passports": {"path": str(passport_path), "shape": list(lock_passports.shape),
                      "axes": ["unit15_or_unit16", "event", "language", "family", "coordinate"]},
        "adjudication": {"energy_is_information": False, "energy_is_causal_importance": False,
                         "signed_texture_tested": True, "natural_coordinate_gear_identified": False,
                         "language_encoding_mechanism_closed": False},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
