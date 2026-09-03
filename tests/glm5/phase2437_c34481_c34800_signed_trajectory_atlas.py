#!/usr/bin/env python3
"""Build a signed, full-coordinate emergence/reuse atlas for eight language families."""
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
OUT = RESULT / "phase2437_c34481_c34800_signed_trajectory_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2437
CAMPAIGN = "C34481-C34800"
INTERACTIONS = ("semantic_validity", "lexical_control")
SPLITS = ("discovery", "confirmation", "fresh_lockbox")
FACTORS = ("language", "surface", "direction")
ANALYSIS_VERSION = "v2_exclude_final_norm_from_residual_updates"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-30 else 0.0


def configuration_index(rows: list[dict]) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    configs = sorted({row["config_id"] for row in rows})
    lookup = {(row["config_id"], row["variant"], row["query_role"]): index for index, row in enumerate(rows)}
    meta = []
    indices = {variant: {role: [] for role in ("source", "target")}
               for variant in ("valid", "broken_a", "broken_b")}
    for config in configs:
        base = rows[lookup[(config, "valid", "source")]]
        meta.append({key: base[key] for key in ("config_id", "family", "unit", "language", "surface",
                                                "surface_class", "direction", "partition")})
        for variant in indices:
            for role in indices[variant]:
                indices[variant][role].append(lookup[(config, variant, role)])
    for variant in indices:
        for role in indices[variant]:
            indices[variant][role] = np.asarray(indices[variant][role], dtype=np.int64)
    return meta, indices


def interaction_at(field: np.ndarray, qpoint: int, event: int, index: dict) -> tuple[np.ndarray, np.ndarray]:
    differences = {}
    for variant in ("valid", "broken_a", "broken_b"):
        target = np.asarray(field[index[variant]["target"], qpoint, event], dtype=np.float32)
        source = np.asarray(field[index[variant]["source"], qpoint, event], dtype=np.float32)
        differences[variant] = target - source
    return differences["valid"] - differences["broken_a"], differences["broken_a"] - differences["broken_b"]


def build_interaction_path(field: np.ndarray, index: dict, configs: int) -> tuple[Path, tuple[int, ...]]:
    path = OUT / "derived/signed_interaction_state.float16.npy"
    progress = OUT / "derived/signed_interaction_progress.json"
    shape = (2, field.shape[1], field.shape[2], configs, field.shape[3])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and progress.exists():
        output = np.lib.format.open_memmap(path, mode="r+")
        completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed_cell"])
        if tuple(output.shape) != shape:
            raise RuntimeError(("stale_interaction_shape", output.shape, shape))
    else:
        output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        completed = 0
    total = field.shape[1] * field.shape[2]
    for cell in range(completed, total):
        qpoint, event = divmod(cell, field.shape[2])
        semantic, lexical = interaction_at(field, qpoint, event, index)
        output[0, qpoint, event] = semantic.astype(np.float16)
        output[1, qpoint, event] = lexical.astype(np.float16)
        output.flush(); save(progress, {"completed_cell": cell + 1, "shape": shape})
        if (cell + 1) % 32 == 0 or cell + 1 == total:
            print(f"[phase2437 interaction] {cell + 1}/{total}", flush=True)
    output.flush(); close(output)
    return path, shape


def split_masks(meta: list[dict]) -> dict[str, np.ndarray]:
    unit = np.asarray([int(row["unit"]) for row in meta])
    return {"discovery": unit < 4, "confirmation": unit == 4, "fresh_lockbox": unit == 5}


def analyze(path: Path, meta: list[dict]) -> dict:
    values = np.load(path, mmap_mode="r")
    interactions, qpoints, events, configs, dim = values.shape
    # q0=embedding, q1..q36=block outputs, q37=final norm.  The q36->q37
    # interface jump is measured separately and must never be called a block update.
    updates = qpoints - 2
    families = np.asarray([row["family"] for row in meta], dtype=object)
    family_names = sorted(set(families))
    masks = split_masks(meta)
    derived = OUT / "derived"
    passports = np.lib.format.open_memmap(derived / "signed_update_family_passports.float32.npy", mode="w+",
                                           dtype=np.float32,
                                           shape=(interactions, len(SPLITS), updates, events, len(family_names), dim))
    energy = np.zeros((interactions, updates, events, len(family_names)), dtype=np.float64)
    coord_sumsq = np.zeros((interactions, dim), dtype=np.float64)
    coord_count = np.zeros(interactions, dtype=np.int64)
    for ii in range(interactions):
        for qpoint in range(updates):
            for event in range(events):
                update = np.asarray(values[ii, qpoint + 1, event], dtype=np.float32) - np.asarray(values[ii, qpoint, event], dtype=np.float32)
                coord_sumsq[ii] += np.sum(update.astype(np.float64) ** 2, axis=0)
                coord_count[ii] += update.shape[0]
                for fi, family in enumerate(family_names):
                    fam = families == family
                    energy[ii, qpoint, event, fi] = float(np.mean(update[fam].astype(np.float64) ** 2))
                    for si, split in enumerate(SPLITS):
                        chosen = fam & masks[split]
                        passports[ii, si, qpoint, event, fi] = update[chosen].mean(axis=0)
            passports.flush()
            if (qpoint + 1) % 6 == 0 or qpoint + 1 == updates:
                print(f"[phase2437 atlas] interaction={INTERACTIONS[ii]} update={qpoint + 1}/{updates}", flush=True)
    np.save(derived / "signed_update_energy.float64.npy", energy)
    coord_rms = np.sqrt(coord_sumsq / np.maximum(coord_count[:, None], 1))
    np.save(derived / "signed_update_coordinate_rms.float64.npy", coord_rms)

    norm_interface_energy = np.zeros((interactions, events, len(family_names)), dtype=np.float64)
    for ii in range(interactions):
        for event in range(events):
            jump = np.asarray(values[ii, qpoints - 1, event], dtype=np.float32) - np.asarray(values[ii, qpoints - 2, event], dtype=np.float32)
            for fi, family in enumerate(family_names):
                norm_interface_energy[ii, event, fi] = float(np.mean(jump[families == family].astype(np.float64) ** 2))
    np.save(derived / "final_norm_interface_energy.float64.npy", norm_interface_energy)

    # Split transfer retains all coordinates; quartiles only audit whether low-RMS coordinates carry stable pattern.
    split_cos = {name: [] for name in INTERACTIONS}
    split_sign = {name: [] for name in INTERACTIONS}
    quartile_cos: dict[str, dict[str, float]] = {}
    quartile_ids = np.zeros((interactions, dim), dtype=np.uint8)
    for ii, interaction in enumerate(INTERACTIONS):
        discovery = np.asarray(passports[ii, 0], dtype=np.float32)
        confirmation = np.asarray(passports[ii, 1], dtype=np.float32)
        lockbox = np.asarray(passports[ii, 2], dtype=np.float32)
        for qpoint in range(updates):
            for event in range(events):
                for fi in range(len(family_names)):
                    split_cos[interaction].append(cosine(discovery[qpoint, event, fi], lockbox[qpoint, event, fi]))
                    split_sign[interaction].append(float(np.mean(np.sign(confirmation[qpoint, event, fi]) ==
                                                                 np.sign(lockbox[qpoint, event, fi]))))
        order = np.argsort(coord_rms[ii], kind="stable")
        bins = np.array_split(order, 4)
        quartile_cos[interaction] = {}
        for qi, coordinates in enumerate(bins):
            quartile_ids[ii, coordinates] = qi
            quartile_cos[interaction][f"q{qi + 1}_low_to_high"] = cosine(discovery[..., coordinates], lockbox[..., coordinates])
    np.save(derived / "signed_update_coordinate_rms_quartile.uint8.npy", quartile_ids)

    # Cross-surface/language/direction reuse on the untouched unit-5 lockbox.
    factor_reuse = np.zeros((interactions, len(FACTORS), updates, events, len(family_names)), dtype=np.float32)
    lock = masks["fresh_lockbox"]
    for ii in range(interactions):
        for qpoint in range(updates):
            for event in range(events):
                update = np.asarray(values[ii, qpoint + 1, event], dtype=np.float32) - np.asarray(values[ii, qpoint, event], dtype=np.float32)
                for fi, family in enumerate(family_names):
                    base = lock & (families == family)
                    for ai, factor in enumerate(FACTORS):
                        levels = sorted({meta[index][factor] for index in np.flatnonzero(base)}, key=str)
                        left = update[base & np.asarray([row[factor] == levels[0] for row in meta])].mean(0)
                        right = update[base & np.asarray([row[factor] == levels[1] for row in meta])].mean(0)
                        factor_reuse[ii, ai, qpoint, event, fi] = cosine(left, right)
    np.save(derived / "signed_update_lockbox_factor_reuse.float32.npy", factor_reuse)

    # Each physical coordinate gets a persistence score from its whole family x event passport.
    persistence = np.zeros((interactions, updates - 1, dim), dtype=np.float32)
    lock_pass = passports[:, 2]
    for ii in range(interactions):
        for qpoint in range(updates - 1):
            left = np.asarray(lock_pass[ii, qpoint], dtype=np.float64).transpose(2, 0, 1).reshape(dim, -1)
            right = np.asarray(lock_pass[ii, qpoint + 1], dtype=np.float64).transpose(2, 0, 1).reshape(dim, -1)
            numerator = np.sum(left * right, axis=1)
            persistence[ii, qpoint] = numerator / np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-30)
    np.save(derived / "signed_coordinate_adjacent_update_persistence.float32.npy", persistence)

    peak = {}
    for ii, interaction in enumerate(INTERACTIONS):
        peak[interaction] = {}
        for fi, family in enumerate(family_names):
            matrix = energy[ii, :, :, fi]
            flat = int(np.argmax(matrix)); qpoint, event = np.unravel_index(flat, matrix.shape)
            family_max = float(matrix[qpoint, event])
            above = np.flatnonzero(matrix.max(axis=1) >= .5 * family_max)
            peak[interaction][family] = {"peak_update": int(qpoint), "peak_event": int(event),
                                         "peak_energy": family_max,
                                         "half_peak_first_update": int(above[0]) if len(above) else None}
    summary = {
        "split_discovery_to_fresh_cosine": {key: float(np.mean(value)) for key, value in split_cos.items()},
        "confirmation_to_fresh_sign_agreement": {key: float(np.mean(value)) for key, value in split_sign.items()},
        "lockbox_factor_reuse": {INTERACTIONS[ii]: {factor: float(factor_reuse[ii, ai].mean())
                                                     for ai, factor in enumerate(FACTORS)}
                                 for ii in range(interactions)},
        "coordinate_adjacent_update_persistence": {INTERACTIONS[ii]: float(persistence[ii].mean()) for ii in range(interactions)},
        "coordinate_rms_quartile_discovery_to_fresh_cosine": quartile_cos,
        "energy_mean": {INTERACTIONS[ii]: float(energy[ii].mean()) for ii in range(interactions)},
        "semantic_to_lexical_energy_ratio": float(energy[0].mean() / max(energy[1].mean(), 1e-30)),
        "final_norm_interface_energy_mean_separate": {INTERACTIONS[ii]: float(norm_interface_energy[ii].mean())
                                                       for ii in range(interactions)},
        "peak_map": peak,
    }
    files = {"interaction_state": str(path),
             "update_passports": str(derived / "signed_update_family_passports.float32.npy"),
             "energy": str(derived / "signed_update_energy.float64.npy"),
             "coordinate_rms": str(derived / "signed_update_coordinate_rms.float64.npy"),
             "rms_quartile": str(derived / "signed_update_coordinate_rms_quartile.uint8.npy"),
             "factor_reuse": str(derived / "signed_update_lockbox_factor_reuse.float32.npy"),
             "persistence": str(derived / "signed_coordinate_adjacent_update_persistence.float32.npy"),
             "final_norm_interface_energy": str(derived / "final_norm_interface_energy.float64.npy")}
    passports.flush(); close(passports); close(values)
    return {"analysis_version": ANALYSIS_VERSION, "configurations": configs, "families": family_names, "qpoints": qpoints, "updates": updates,
            "events": events, "dimension": dim, "summary": summary, "files": files}


def append_memo(result: dict) -> None:
    memo_text = MEMO.read_text(encoding="utf-8")
    if f"## Phase {PHASE}:" in memo_text and "Phase 2437 final-norm质量门修正" in memo_text:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    if f"## Phase {PHASE}:" in memo_text:
        text = rf"""

### Phase 2437 final-norm质量门修正 [{stamp}]

首次汇总将$q_{{36}}\to q_{{37}}$计入了37个“更新”，而$q_{{37}}$是final norm，不是Transformer block。这导致八族能量峰伪集中于update36。修正后动力学只包含$q_0\to q_1,\ldots,q_{{35}}\to q_{{36}}$ 36个真实block更新，final-norm跳变单独记录为输出接口量。修正后结果 `{json.dumps(result, ensure_ascii=False)}`。后续Phase一律使用此v2结果。
"""
    else:
        text = rf"""

## Phase {PHASE}: 有符号条件纹理的出现—延续—复用—分化全坐标图谱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 每个family×unit×language×surface×direction配置有valid/broken-A/broken-B和source/target双查询。先在完全相同的物理坐标上取查询角色差，再构造语义有效性交互与词项/结构对照交互；保留384配置×38 checkpoint×8事件×2560坐标的完整有符号场。按unit0–3 discovery、unit4 confirmation、unit5 fresh lockbox建立更新护照，逐坐标测出现层、相邻更新延续、中英/表述/方向复用、跨split余弦与符号一致率。

$$D_v(q,e)=H_{{v,target}}(q,e)-H_{{v,source}}(q,e),$$
$$I_{{sem}}=D_{{valid}}-D_{{brokenA}},\qquad I_{{lex}}=D_{{brokenA}}-D_{{brokenB}},$$
$$P_{{f,s,q,e,j}}=\mathbb E[\,I(q+1,e,j)-I(q,e,j)\mid f,s\,].$$

**结果汇总。** 维度 `{json.dumps({k: result['analysis'][k] for k in ('configurations','qpoints','updates','events','dimension')}, ensure_ascii=False)}`；摘要 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2437_c34481_c34800_signed_trajectory_atlas.py`；完整interaction状态、三个split的全2560坐标更新护照、能量、坐标RMS四分位、因素复用和逐坐标相邻层持续度位于`tests/glm5/result/phase2437_c34481_c34800_signed_trajectory_atlas/derived`。

**分析与理论进展。** 本Phase把“某层能区分类别”改写为一张可检索的坐标生命周期图：同一个外部操作形成的角色条件纹理在何事件/层出现，能否跨新实体、语言、表述和方向保留，以及每个坐标是否沿层延续。RMS四分位仅用于审计低值坐标是否仍有跨split结构，绝不用于删坐标或构成主表示。

**问题硬伤与结论。** $I_{{sem}}$仍可能含谓词token、句法和任务难度差，$I_{{lex}}$不是完美等难零假设；跨层持续也可能主要来自残差连接。出现层的“半峰值”只是描述阈值，不是机制边界。只有跨split、跨因素、优于词项对照且能预测下一层和输出的有符号纹理，才可升级为条件坐标齿轮候选；本Phase不声称已破解编码。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8"))
        if result.get("analysis", {}).get("analysis_version") == ANALYSIS_VERSION:
            append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2435 / "index/trajectory_rows.jsonl")
    meta, index = configuration_index(rows)
    write_rows(OUT / "index/configurations.jsonl", meta)
    field_path = P2436 / "raw/hypergraph_event_field.float16.npy"
    field = np.load(field_path, mmap_mode="r")
    interaction_path, shape = build_interaction_path(field, index, len(meta))
    close(field)
    analysis = analyze(interaction_path, meta)
    summary = analysis["summary"]
    adjudication = {
        "semantic_energy_exceeds_lexical": summary["semantic_to_lexical_energy_ratio"] > 1,
        "semantic_new_unit_cosine_positive": summary["split_discovery_to_fresh_cosine"]["semantic_validity"] > 0,
        "semantic_all_factor_reuse_positive": all(value > 0 for value in summary["lockbox_factor_reuse"]["semantic_validity"].values()),
        "low_rms_quartile_structure_positive": summary["coordinate_rms_quartile_discovery_to_fresh_cosine"]["semantic_validity"]["q1_low_to_high"] > 0,
        "conditional_coordinate_gear_proven": False,
    }
    files = analysis["files"]
    checks = {"configurations_384": analysis["configurations"] == 384,
              "interaction_shape": list(shape) == [2, 38, 8, 384, 2560],
              "residual_updates_36": analysis["updates"] == 36,
              "all_families": len(analysis["families"]) == 8,
              "all_files": all(Path(path).exists() for path in files.values()),
              "finite": all(math.isfinite(value) for value in (
                  *summary["split_discovery_to_fresh_cosine"].values(),
                  *summary["confirmation_to_fresh_sign_agreement"].values(),
                  summary["semantic_to_lexical_energy_ratio"])),
              "raw_retained": field_path.exists(), "claim_boundary": not adjudication["conditional_coordinate_gear_proven"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "analysis": analysis, "adjudication": adjudication,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
