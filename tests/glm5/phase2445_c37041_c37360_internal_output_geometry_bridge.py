#!/usr/bin/env python3
"""Test whether the replicated semantic coordinate geometry survives into output contributions."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2437 = RESULT / "phase2437_c34481_c34800_signed_trajectory_atlas"
P2439 = RESULT / "phase2439_c35121_c35440_output_autonomous_bridge"
P2444 = RESULT / "phase2444_c36721_c37040_semantic_specific_multiunit_multinull"
OUT = RESULT / "phase2445_c37041_c37360_internal_output_geometry_bridge"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2445
CAMPAIGN = "C37041-C37360"
INTERACTIONS = ("semantic_validity", "lexical_control")
COMPONENTS = ("signed_state", "block_update")
SPLITS = ("discovery", "confirmation", "fresh")
LANGUAGES = ("en", "zh")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-30 else 0.0


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    if float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def analyze(meta: list[dict]) -> dict:
    internal = np.load(P2444 / "derived/selected_semantic_lexical_split_passports.float32.npy", mmap_mode="r")
    output = np.load(P2439 / "derived/signed_output_contribution_interaction.float32.npy", mmap_mode="r")
    permutations = np.load(P2444 / "derived/family_null_permutations.int16.npy")
    final_output = np.asarray(output[:, 37], dtype=np.float32)
    families = json.loads((P2444 / "analysis/final.json").read_text(encoding="utf-8"))["analysis"]["families"]
    family_array = np.asarray([row["family"] for row in meta], dtype=object)
    unit = np.asarray([int(row["unit"]) for row in meta])
    language = np.asarray([row["language"] for row in meta], dtype=object)
    split_masks = (unit < 4, unit == 4, unit == 5)
    output_passports = np.zeros((2, 3, 2, 8, 2560), dtype=np.float32)
    for ii in range(2):
        for si, chosen_units in enumerate(split_masks):
            for li, lang in enumerate(LANGUAGES):
                for fi, family in enumerate(families):
                    chosen = chosen_units & (language == lang) & (family_array == family)
                    output_passports[ii, si, li, fi] = final_output[ii, chosen].mean(axis=0)
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    np.save(derived / "split_language_family_output_passports.float32.npy", output_passports)
    summary = {}
    upper = np.triu_indices(8, 1)
    for ii, interaction in enumerate(INTERACTIONS):
        summary[interaction] = {}
        for ci, component in enumerate(COMPONENTS):
            summary[interaction][component] = {}
            for si, split in enumerate(SPLITS):
                summary[interaction][component][split] = {}
                for li, lang in enumerate(LANGUAGES):
                    h = np.asarray(internal[ii, ci, si, li], dtype=np.float64)
                    c = np.asarray(output_passports[ii, si, li], dtype=np.float64)
                    h_n = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-30)
                    c_n = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-30)
                    observed = correlation((h_n @ h_n.T)[upper], (c_n @ c_n.T)[upper])
                    null = np.asarray([correlation((h_n @ h_n.T)[upper],
                                                   (c_n @ c_n.T)[np.ix_(perm, perm)][upper])
                                       for perm in permutations], dtype=np.float64)
                    summary[interaction][component][split][lang] = {
                        "same_coordinate_cosine": float(np.mean([cosine(h[fi], c[fi]) for fi in range(8)])),
                        "family_geometry_correlation": observed,
                        "geometry_null_mean": float(null.mean()), "geometry_null_q95": float(np.quantile(null, .95)),
                        "geometry_q95_advantage": observed - float(np.quantile(null, .95))}
    phase2439 = json.loads((P2439 / "analysis/final.json").read_text(encoding="utf-8"))
    gains = np.load(phase2439["internal_to_output"]["metrics"])
    p2444 = json.loads((P2444 / "analysis/final.json").read_text(encoding="utf-8"))
    split_names = ("fresh_unit", "surface", "language", "direction", "family_holdout")
    targeted = {}
    for ii, interaction in enumerate(INTERACTIONS):
        layer = int(p2444["analysis"]["selections"][f"{interaction}:signed_state"]) + 1
        targeted[interaction] = {"qpoint": layer, "event": "query_end"}
        for si, split in enumerate(split_names):
            targeted[interaction][split] = {"diagonal_gain": float(gains[ii, si, 2, layer, 4]),
                                             "coordinate_mismatch_gain": float(gains[ii, si, 3, layer, 4]),
                                             "physical_advantage": float(gains[ii, si, 2, layer, 4] - gains[ii, si, 3, layer, 4])}
    close(internal); close(output)
    return {"families": families, "summary": summary, "targeted_absolute_output_gains": targeted,
            "files": {"output_passports": str(derived / "split_language_family_output_passports.float32.npy")}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 自动续研——语义条件坐标纹理到最终token贡献的绝对与关系几何桥（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 固定使用Phase2444仅由unit0–3选出的semantic/lexical signed-state与block-update层，不重新挑输出最优层。对discovery、unit4、unit5及中英分别建立八family×2560内部护照；最终输出侧用Phase2439真实$H_i\Delta W_i$贡献组成相同护照。先报告同物理坐标余弦，再比较内部family Gram与输出family Gram的相关；64个family排列给出几何q95。同时读取Phase2439目标qpoint/query-end的绝对逐坐标输出预测收益，避免用几何阳性替代绝对闭合。

$$G^H=\widehat P^H(\widehat P^H)^\top,\quad G^C=\widehat P^C(\widehat P^C)^\top,\quad
r_{{geom}}=\operatorname{{corr}}(\operatorname{{vech}}G^H,\operatorname{{vech}}G^C).$$

**结果汇总。** 内部—输出几何 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；冻结候选绝对输出收益 `{json.dumps(result['analysis']['targeted_absolute_output_gains'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2445_c37041_c37360_internal_output_geometry_bridge.py`；split×language×family×2560最终输出贡献护照与final位于同名结果目录；内部护照、64置乱及绝对输出指标沿用Phase2444/2439。

**分析与理论进展。** 这一步严格区分两种闭合：逐坐标绝对预测回答“能否重建最终贡献”，关系几何回答“八族相对关系是否保留到输出”。只有二者在unit4/5及中英都通过，才有编译器候选；几何单独通过只说明关系排序保留，绝对收益失败仍是硬伤。

**问题硬伤与结论。** 内部状态与$H_i\Delta W_i$量纲不同，同坐标余弦不是架构恒等。Gram会忽略共同旋转和尺度，可能放大弱关系；64置乱样本有限。输出贡献interaction本身混合行特异输出权重，几何对应不等于固定齿轮，更不等于因果闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    meta = read_rows(P2437 / "index/configurations.jsonl")
    analysis = analyze(meta)
    sem = analysis["summary"]["semantic_validity"]
    geometry_all = all(sem[component][split][lang]["geometry_q95_advantage"] > 0
                       for component in COMPONENTS for split in ("confirmation", "fresh") for lang in LANGUAGES)
    target = analysis["targeted_absolute_output_gains"]["semantic_validity"]
    absolute_all = all(target[split]["diagonal_gain"] > 0 and target[split]["physical_advantage"] > 0
                       for split in ("fresh_unit", "surface", "language", "direction", "family_holdout"))
    adjudication = {"semantic_internal_output_geometry_above_q95_all_lockboxes_languages": geometry_all,
                    "semantic_targeted_absolute_output_positive_all_splits": absolute_all,
                    "semantic_output_bridge_closed": geometry_all and absolute_all,
                    "language_encoding_mechanism_closed": False}
    checks = {"eight_families": len(analysis["families"]) == 8,
              "all_splits": all(set(component) == set(SPLITS) for interaction in analysis["summary"].values()
                                for component in interaction.values()),
              "all_files": all(Path(path).exists() for path in analysis["files"].values()),
              "finite": all(math.isfinite(value) for interaction in analysis["summary"].values()
                            for component in interaction.values() for split in component.values()
                            for language in split.values() for value in language.values()),
              "sources_retained": (P2444 / "derived/selected_semantic_lexical_split_passports.float32.npy").exists() and
                                  (P2439 / "derived/signed_output_contribution_interaction.float32.npy").exists(),
              "claim_boundary": not adjudication["language_encoding_mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
