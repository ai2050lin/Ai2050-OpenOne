#!/usr/bin/env python3
"""Multi-unit, multi-null semantic-specific cross-language interaction audit."""
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
OUT = RESULT / "phase2444_c36721_c37040_semantic_specific_multiunit_multinull"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2444
CAMPAIGN = "C36721-C37040"
INTERACTIONS = ("semantic_validity", "lexical_control")
COMPONENTS = ("signed_state", "block_update")
SHIFT = 791
EVENT = 4
N_NULL = 64


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


def corr(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    if float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def null_permutations(count: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(2444)
    output = []
    while len(output) < count:
        perm = rng.permutation(n)
        if np.any(perm != np.arange(n)):
            output.append(perm)
    return np.asarray(output, dtype=np.int64)


def cell(state: np.ndarray, meta: list[dict], permutations: np.ndarray) -> dict:
    families = sorted({row["family"] for row in meta})
    units = sorted({int(row["unit"]) for row in meta})
    surfaces = sorted({row["surface"] for row in meta})
    lookup = {(row["family"], int(row["unit"]), row["surface"], int(row["direction"]), row["language"]): index
              for index, row in enumerate(meta)}
    per_unit = {unit: {"matched": [], "shift": [], "null": [[] for _ in range(len(permutations))],
                       "gram": [], "gram_null": [[] for _ in range(len(permutations))]}
                for unit in units}
    surface_scores, direction_scores = [], []
    upper = np.triu_indices(len(families), 1)
    for unit in units:
        for surface in surfaces:
            for direction in (0, 1):
                en = np.stack([state[lookup[(family, unit, surface, direction, "en")]] for family in families])
                zh = np.stack([state[lookup[(family, unit, surface, direction, "zh")]] for family in families])
                en_n = en / np.maximum(np.linalg.norm(en, axis=1, keepdims=True), 1e-30)
                zh_n = zh / np.maximum(np.linalg.norm(zh, axis=1, keepdims=True), 1e-30)
                cosine_matrix = en_n @ zh_n.T
                gram_en, gram_zh = en_n @ en_n.T, zh_n @ zh_n.T
                per_unit[unit]["matched"].extend(np.diag(cosine_matrix).tolist())
                per_unit[unit]["shift"].extend([cosine(en[i], np.roll(zh[i], SHIFT)) for i in range(len(families))])
                per_unit[unit]["gram"].append(corr(gram_en[upper], gram_zh[upper]))
                for ni, perm in enumerate(permutations):
                    per_unit[unit]["null"][ni].append(float(np.mean(cosine_matrix[np.arange(len(families)), perm])))
                    per_unit[unit]["gram_null"][ni].append(corr(gram_en[upper], gram_zh[np.ix_(perm, perm)][upper]))
    for family in families:
        for unit in units:
            for language in ("en", "zh"):
                for direction in (0, 1):
                    surface_scores.append(cosine(state[lookup[(family, unit, "canonical", direction, language)]],
                                                 state[lookup[(family, unit, "natural", direction, language)]]))
                for surface in surfaces:
                    direction_scores.append(cosine(state[lookup[(family, unit, surface, 0, language)]],
                                                   state[lookup[(family, unit, surface, 1, language)]]))
    unit_summary = {}
    for unit, values in per_unit.items():
        null_distribution = np.asarray([np.mean(item) for item in values["null"]], dtype=np.float64)
        gram_null_distribution = np.asarray([np.mean(item) for item in values["gram_null"]], dtype=np.float64)
        matched = float(np.mean(values["matched"])); shifted = float(np.mean(values["shift"])); gram = float(np.mean(values["gram"]))
        unit_summary[str(unit)] = {"matched": matched, "shift791": shifted,
                                   "family_null_mean": float(null_distribution.mean()),
                                   "family_null_q95": float(np.quantile(null_distribution, .95)),
                                   "gram": gram, "gram_null_mean": float(gram_null_distribution.mean()),
                                   "gram_null_q95": float(np.quantile(gram_null_distribution, .95)),
                                   "physical_advantage": matched - shifted,
                                   "family_q95_advantage": matched - float(np.quantile(null_distribution, .95)),
                                   "gram_q95_advantage": gram - float(np.quantile(gram_null_distribution, .95))}
    return {"units": unit_summary, "surface": float(np.mean(surface_scores)), "direction": float(np.mean(direction_scores))}


def analyze(meta: list[dict]) -> dict:
    path = P2437 / "derived/signed_interaction_state.float16.npy"
    values = np.load(path, mmap_mode="r")
    permutations = null_permutations(N_NULL, 8)
    all_cells: dict[str, dict[str, dict]] = {interaction: {component: {} for component in COMPONENTS}
                                            for interaction in INTERACTIONS}
    selections, lockbox = {}, {}
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    np.save(derived / "family_null_permutations.int16.npy", permutations.astype(np.int16))
    passports = np.zeros((2, 2, 3, 2, 8, 2560), dtype=np.float32)
    families = sorted({row["family"] for row in meta})
    family_array = np.asarray([row["family"] for row in meta], dtype=object)
    unit_array = np.asarray([int(row["unit"]) for row in meta])
    language = np.asarray([row["language"] for row in meta], dtype=object)
    for ii, interaction in enumerate(INTERACTIONS):
        for ci, component in enumerate(COMPONENTS):
            discovery_scores = []
            for layer in range(36):
                state = (np.asarray(values[ii, layer + 1, EVENT], dtype=np.float32) if component == "signed_state" else
                         np.asarray(values[ii, layer + 1, EVENT], dtype=np.float32) - np.asarray(values[ii, layer, EVENT], dtype=np.float32))
                result = cell(state, meta, permutations)
                all_cells[interaction][component][str(layer)] = result
                discovery_scores.append(float(np.mean([result["units"][str(unit)]["physical_advantage"] +
                                                       result["units"][str(unit)]["family_q95_advantage"]
                                                       for unit in range(4)])))
            selected = int(np.argmax(discovery_scores)); selections[f"{interaction}:{component}"] = selected
            chosen = all_cells[interaction][component][str(selected)]
            lockbox.setdefault(interaction, {})[component] = {
                "selected_layer": selected, "discovery_score": float(discovery_scores[selected]),
                "confirmation_unit4": chosen["units"]["4"], "fresh_unit5": chosen["units"]["5"],
                "surface": chosen["surface"], "direction": chosen["direction"]}
            state = (np.asarray(values[ii, selected + 1, EVENT], dtype=np.float32) if component == "signed_state" else
                     np.asarray(values[ii, selected + 1, EVENT], dtype=np.float32) - np.asarray(values[ii, selected, EVENT], dtype=np.float32))
            for si, units in enumerate((unit_array < 4, unit_array == 4, unit_array == 5)):
                for li, lang in enumerate(("en", "zh")):
                    for fi, family in enumerate(families):
                        chosen_mask = units & (language == lang) & (family_array == family)
                        passports[ii, ci, si, li, fi] = state[chosen_mask].mean(axis=0)
        print(f"[phase2444] interaction={interaction} complete", flush=True)
    np.save(derived / "selected_semantic_lexical_split_passports.float32.npy", passports)
    save(OUT / "analysis/all_layer_unit_metrics.json", all_cells)
    close(values)
    return {"interactions": INTERACTIONS, "components": COMPONENTS, "null_permutations": N_NULL,
            "families": families, "selections": selections, "lockbox": lockbox,
            "files": {"nulls": str(derived / "family_null_permutations.int16.npy"),
                      "passports": str(derived / "selected_semantic_lexical_split_passports.float32.npy"),
                      "all_metrics": str(OUT / "analysis/all_layer_unit_metrics.json")}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 自动续研——语义有效性交互的六unit跨语言64置乱特异性审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2443的valid角色差可能只是family查询模板。本Phase回到二阶交互：$I_{{sem}}=(D_{{valid}}-D_{{brokenA}})$消去共享查询文字，$I_{{lex}}=(D_{{brokenA}}-D_{{brokenB}})$作为词项/结构对照；覆盖384配置、六unit、双表述、双方向、query-end全部2560坐标。分别检查signed state与36个block update，同坐标对照+791错配；family身份零假设由64个冻结随机排列给出均值和q95。层仍只用unit0–3选择，unit4/5锁箱。

$$\Delta_{{family,q95}}=c_{{matched}}-Q_{{.95}}\{{c_{{\pi(f)}}\}}_{{64}},\qquad
\Delta_{{phys}}=c_{{matched}}-c_{{shift791}}.$$

**结果汇总。** 冻结选择 `{json.dumps(result['analysis']['selections'], ensure_ascii=False)}`；语义/词项锁箱 `{json.dumps(result['analysis']['lockbox'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2444_c36721_c37040_semantic_specific_multiunit_multinull.py`；64置乱、semantic/lexical×state/update×split×language×family×2560护照、全部层/每unit指标位于同名结果目录。

**分析与理论进展。** 这是本轮第一次把多unit跨语言坐标复用与语义有效性二阶对照合并。若$I_{{sem}}$在unit4/5同时超过坐标错配、family置乱q95及$I_{{lex}}$，才可把Phase2443从查询模板动力学提升为语义条件纹理候选；否则只能保留一般语言操作/模板坐标律。

**问题硬伤与结论。** broken-A/B难度与词项并非严格等价；64置乱只检验family标签，不检验所有模板/长度混淆。余弦高表示方向复用，不代表可加、可搬运或因果必要。输出桥在Phase2439仍未闭合，因此即使语义特异性通过也只是拼图，不是完整机制。
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
    adjudication = {}
    for interaction in INTERACTIONS:
        for component in COMPONENTS:
            for split in ("confirmation_unit4", "fresh_unit5"):
                value = analysis["lockbox"][interaction][component][split]
                adjudication[f"{interaction}_{component}_{split}_physical_family_q95"] = (
                    value["physical_advantage"] > 0 and value["family_q95_advantage"] > 0)
    semantic_specific = True
    for component in COMPONENTS:
        for split in ("confirmation_unit4", "fresh_unit5"):
            sem = analysis["lockbox"]["semantic_validity"][component][split]
            lex = analysis["lockbox"]["lexical_control"][component][split]
            semantic_specific &= (sem["matched"] > lex["matched"] and sem["physical_advantage"] > 0 and
                                  sem["family_q95_advantage"] > 0)
    adjudication["semantic_specific_crosslanguage_candidate"] = bool(semantic_specific)
    adjudication["conditional_coordinate_gear_proven"] = False
    checks = {"two_interactions": set(analysis["interactions"]) == set(INTERACTIONS),
              "two_components": set(analysis["components"]) == set(COMPONENTS),
              "nulls_64": analysis["null_permutations"] == 64,
              "all_files": all(Path(path).exists() for path in analysis["files"].values()),
              "finite": all(math.isfinite(value) for interaction in analysis["lockbox"].values()
                            for component in interaction.values() for split in ("confirmation_unit4", "fresh_unit5")
                            for value in component[split].values()),
              "source_retained": (P2437 / "derived/signed_interaction_state.float16.npy").exists(),
              "claim_boundary": not adjudication["conditional_coordinate_gear_proven"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
