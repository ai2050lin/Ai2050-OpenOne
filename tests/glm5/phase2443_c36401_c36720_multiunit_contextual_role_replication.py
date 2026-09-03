#!/usr/bin/env python3
"""Replicate the contextual cross-language role field across six units and two surfaces."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2435 = RESULT / "phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior/qwen4b"
P2436 = RESULT / "phase2436_c34161_c34480_qwen4b_hypergraph_fullfield"
OUT = RESULT / "phase2443_c36401_c36720_multiunit_contextual_role_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2443
CAMPAIGN = "C36401-C36720"
COMPONENTS = ("raw_state", "embedding_subtracted", "block_update")
MEASURES = ("language_coordinate", "language_shift791", "language_family_permuted",
            "language_family_gram", "language_family_gram_permuted", "surface_coordinate", "direction_coordinate")
SHIFT = 791
EVENT = 4  # query_end


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
    if len(a) < 2 or float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def config_index(rows: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    configs = sorted({row["config_id"] for row in rows})
    lookup = {(row["config_id"], row["variant"], row["query_role"]): index for index, row in enumerate(rows)}
    meta, source, target = [], [], []
    for config in configs:
        row = rows[lookup[(config, "valid", "source")]]
        meta.append({key: row[key] for key in ("config_id", "family", "unit", "language", "surface", "direction", "partition")})
        source.append(lookup[(config, "valid", "source")]); target.append(lookup[(config, "valid", "target")])
    return meta, np.asarray(source, dtype=np.int64), np.asarray(target, dtype=np.int64)


def role_difference(field: np.ndarray, source: np.ndarray, target: np.ndarray, qpoint: int) -> np.ndarray:
    return np.asarray(field[target, qpoint, EVENT], dtype=np.float32) - np.asarray(field[source, qpoint, EVENT], dtype=np.float32)


def metric_cell(state: np.ndarray, meta: list[dict], family_permutation: dict[str, str]) -> tuple[np.ndarray, dict]:
    lookup = {(row["family"], int(row["unit"]), row["surface"], int(row["direction"]), row["language"]): index
              for index, row in enumerate(meta)}
    families = sorted({row["family"] for row in meta})
    units = sorted({int(row["unit"]) for row in meta})
    surfaces = sorted({row["surface"] for row in meta})
    language_coord, language_shift, language_perm = [], [], []
    per_unit = {unit: {"coord": [], "shift": [], "perm": []} for unit in units}
    grams, gram_perm = [], []
    for unit in units:
        for surface in surfaces:
            for direction in (0, 1):
                en = np.stack([state[lookup[(family, unit, surface, direction, "en")]] for family in families])
                zh = np.stack([state[lookup[(family, unit, surface, direction, "zh")]] for family in families])
                en_n = en / np.maximum(np.linalg.norm(en, axis=1, keepdims=True), 1e-30)
                zh_n = zh / np.maximum(np.linalg.norm(zh, axis=1, keepdims=True), 1e-30)
                upper = np.triu_indices(len(families), 1)
                permutation = np.asarray([families.index(family_permutation[family]) for family in families])
                grams.append(correlation((en_n @ en_n.T)[upper], (zh_n @ zh_n.T)[upper]))
                gram_perm.append(correlation((en_n @ en_n.T)[upper], (zh_n @ zh_n.T)[np.ix_(permutation, permutation)][upper]))
                for fi, family in enumerate(families):
                    c = cosine(en[fi], zh[fi]); s = cosine(en[fi], np.roll(zh[fi], SHIFT)); p = cosine(en[fi], zh[permutation[fi]])
                    language_coord.append(c); language_shift.append(s); language_perm.append(p)
                    per_unit[unit]["coord"].append(c); per_unit[unit]["shift"].append(s); per_unit[unit]["perm"].append(p)
    surface_cos = []
    for family in families:
        for unit in units:
            for language in ("en", "zh"):
                for direction in (0, 1):
                    surface_cos.append(cosine(state[lookup[(family, unit, "canonical", direction, language)]],
                                              state[lookup[(family, unit, "natural", direction, language)]]))
    direction_cos = []
    for family in families:
        for unit in units:
            for language in ("en", "zh"):
                for surface in surfaces:
                    direction_cos.append(cosine(state[lookup[(family, unit, surface, 0, language)]],
                                                state[lookup[(family, unit, surface, 1, language)]]))
    measures = np.asarray((np.mean(language_coord), np.mean(language_shift), np.mean(language_perm),
                           np.mean(grams), np.mean(gram_perm), np.mean(surface_cos), np.mean(direction_cos)), dtype=np.float32)
    unit_summary = {str(unit): {"coordinate": float(np.mean(value["coord"])),
                                "shift791": float(np.mean(value["shift"])),
                                "family_permuted": float(np.mean(value["perm"])),
                                "physical_advantage": float(np.mean(value["coord"]) - np.mean(value["shift"])),
                                "family_identity_advantage": float(np.mean(value["coord"]) - np.mean(value["perm"]))}
                    for unit, value in per_unit.items()}
    return measures, unit_summary


def analyze(rows: list[dict]) -> dict:
    meta, source, target = config_index(rows)
    field = np.load(P2436 / "raw/hypergraph_event_field.float16.npy", mmap_mode="r")
    families = sorted({row["family"] for row in meta})
    perm_values = np.random.default_rng(2443).permutation(families)
    family_permutation = dict(zip(families, perm_values))
    metrics = np.zeros((len(COMPONENTS), 36, len(MEASURES)), dtype=np.float32)
    unit_metrics: dict[str, dict[str, dict]] = {component: {} for component in COMPONENTS}
    embedding = role_difference(field, source, target, 0)
    for update in range(36):
        current = role_difference(field, source, target, update + 1)
        previous = role_difference(field, source, target, update)
        states = (current, current - embedding, current - previous)
        for ci, component in enumerate(COMPONENTS):
            cell, units = metric_cell(states[ci], meta, family_permutation)
            metrics[ci, update] = cell; unit_metrics[component][str(update)] = units
        if (update + 1) % 6 == 0 or update + 1 == 36:
            print(f"[phase2443] block_output={update + 1}/36", flush=True)
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    np.save(derived / "multiunit_contextual_role_metrics.float32.npy", metrics)
    selections, lockbox, passports = {}, {}, np.zeros((len(COMPONENTS), 3, 2, len(families), 2560), dtype=np.float32)
    family_array = np.asarray([row["family"] for row in meta], dtype=object)
    unit_array = np.asarray([int(row["unit"]) for row in meta])
    language_array = np.asarray([row["language"] for row in meta], dtype=object)
    for ci, component in enumerate(COMPONENTS):
        # Layer is selected solely from unit0-3 via exact per-unit score.
        discovery_scores = []
        for update in range(36):
            units = unit_metrics[component][str(update)]
            discovery_scores.append(float(np.mean([units[str(unit)]["physical_advantage"] + units[str(unit)]["family_identity_advantage"]
                                                   for unit in range(4)])))
        selected = int(np.argmax(discovery_scores)); selections[component] = selected
        lockbox[component] = {"discovery_score": float(discovery_scores[selected]),
                              "confirmation_unit4": unit_metrics[component][str(selected)]["4"],
                              "fresh_unit5": unit_metrics[component][str(selected)]["5"],
                              "all_measure_values": {MEASURES[mi]: float(metrics[ci, selected, mi]) for mi in range(len(MEASURES))}}
        current = role_difference(field, source, target, selected + 1)
        previous = role_difference(field, source, target, selected)
        state = (current, current - embedding, current - previous)[ci]
        for si, units_chosen in enumerate((unit_array < 4, unit_array == 4, unit_array == 5)):
            for li, lang in enumerate(("en", "zh")):
                for fi, family in enumerate(families):
                    chosen = units_chosen & (language_array == lang) & (family_array == family)
                    passports[ci, si, li, fi] = state[chosen].mean(axis=0)
    np.save(derived / "selected_split_language_family_passports.float32.npy", passports)
    close(field)
    return {"configurations": len(meta), "families": families, "components": COMPONENTS, "measures": MEASURES,
            "selections": selections, "lockbox": lockbox,
            "files": {"metrics": str(derived / "multiunit_contextual_role_metrics.float32.npy"),
                      "passports": str(derived / "selected_split_language_family_passports.float32.npy")}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 自动续研——六unit双表述双方向的跨语言上下文坐标冻结复制（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 用Phase2436同batch事件场的全部384个valid配置（8 family×6 unit×中英×canonical/natural×双方向）构造query-end source/target角色差。分别分析raw block输出、减q0 embedding的上下文残差和36个真实block update。每个英文配置与相同family/unit/surface/direction中文配置逐一比较，不先平均方向；零假设为+791物理坐标错配及同unit/surface/direction下family标签置乱。层选择只用unit0–3的物理优势+family身份优势，unit4和unit5冻结确认。

$$D_{{q,c}}=H_{{target,q,c}}-H_{{source,q,c}},\quad
D^{{ctx}}_q=D_q-D_0,\quad U_q=D_{{q+1}}-D_q,$$
$$S_q=\frac14\sum_{{u=0}}^3[(c_{{same}}-c_{{shift}})+(c_{{same}}-c_{{perm-family}})].$$

**结果汇总。** 冻结选择与锁箱 `{json.dumps(result['analysis']['lockbox'], ensure_ascii=False)}`；层选择 `{json.dumps(result['analysis']['selections'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2443_c36401_c36720_multiunit_contextual_role_replication.py`；三component×36block全指标及选中层的discovery/confirmation/fresh×中英×八族×2560坐标护照位于同名结果目录。原始事件场保留。

**分析与理论进展。** 该测试从单unit升级为六组新实体，并把相同family标签、固定物理坐标、表述和方向逐项拆开。若unit4/5中上下文残差与block update都同时高于坐标错配和family置乱，才说明Phase2442的query候选不是共享名称偶然；若只raw state通过，则更可能是残差携带或词项身份。

**问题硬伤与结论。** 中英模板仍是人工平行材料，实体名称均为拉丁字符串；family查询角色词也具有稳定模板。family置乱是单个冻结排列。跨unit复制证明的是受控材料中的条件坐标复用，不等于开放语言普遍机制；还需不同实体脚本、非翻译改写与输出桥。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2435 / "index/trajectory_rows.jsonl")
    analysis = analyze(rows)
    adjudication = {}
    for component in COMPONENTS:
        confirm = analysis["lockbox"][component]["confirmation_unit4"]
        fresh = analysis["lockbox"][component]["fresh_unit5"]
        adjudication[f"{component}_confirmation_physical_and_family_identity"] = confirm["physical_advantage"] > 0 and confirm["family_identity_advantage"] > 0
        adjudication[f"{component}_fresh_physical_and_family_identity"] = fresh["physical_advantage"] > 0 and fresh["family_identity_advantage"] > 0
    adjudication["contextual_role_candidate_replicated"] = all(adjudication[key] for key in (
        "embedding_subtracted_confirmation_physical_and_family_identity",
        "embedding_subtracted_fresh_physical_and_family_identity",
        "block_update_confirmation_physical_and_family_identity",
        "block_update_fresh_physical_and_family_identity"))
    adjudication["universal_language_coordinate_mechanism_closed"] = False
    checks = {"configs_384": analysis["configurations"] == 384, "eight_families": len(analysis["families"]) == 8,
              "three_components": set(analysis["components"]) == set(COMPONENTS),
              "selection_in_discovery": all(0 <= value < 36 for value in analysis["selections"].values()),
              "all_files": all(Path(path).exists() for path in analysis["files"].values()),
              "finite": all(math.isfinite(value) for component in analysis["lockbox"].values()
                            for split in ("confirmation_unit4", "fresh_unit5") for value in component[split].values()),
              "raw_retained": (P2436 / "raw/hypergraph_event_field.float16.npy").exists(),
              "claim_boundary": not adjudication["universal_language_coordinate_mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
