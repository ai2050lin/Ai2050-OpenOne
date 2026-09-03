#!/usr/bin/env python3
"""Finite symmetric intervention of frozen semantic VJP directions on held units."""
from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2435 = RESULT / "phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior/qwen4b"
P2448 = RESULT / "phase2448_c38001_c38480_vjp_semantic_multiunit_replication"
OUT = RESULT / "phase2451_c39121_c39440_vjp_finite_intervention"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2451
CAMPAIGN = "C39121-C39440"
QPOINT = 18
DOSE = 0.01
SHIFT = 791
CONTROLS = ("matched_family_coordinate", "shift791_coordinate", "permuted_family_coordinate")
VARIANTS = ("valid", "broken_a", "broken_b")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def selected_rows() -> list[dict]:
    rows = read_rows(P2435 / "index/trajectory_rows.jsonl")
    selected = [row for row in rows if int(row["unit"]) in (4, 5) and row["surface"] == "natural"]
    selected.sort(key=lambda row: row["case_id"])
    return selected


def load_directions(families: list[str]) -> tuple[np.ndarray, np.ndarray]:
    final = json.loads((P2448 / "analysis/final.json").read_text(encoding="utf-8"))
    if int(final["analysis"]["selection"]["semantic_validity"]["gradient"]) != QPOINT:
        raise RuntimeError("frozen qpoint mismatch")
    passports = np.load(final["analysis"]["passports"], mmap_mode="r")
    # Copy before closing the memmap; retaining a view would dereference a closed Windows mapping.
    raw = np.asarray(passports[0, 0, 0, :, QPOINT], dtype=np.float32).copy()  # [language, family, coordinate]
    close(passports)
    norms = np.sqrt(np.mean(raw.astype(np.float64) ** 2, axis=-1, keepdims=True))
    directions = raw / np.maximum(norms, 1e-30)
    rng = np.random.default_rng(2451)
    while True:
        permutation = rng.permutation(len(families))
        if np.all(permutation != np.arange(len(families))):
            break
    return directions.astype(np.float32), permutation


def capture(rows: list[dict], families: list[str], directions: np.ndarray, permutation: np.ndarray) -> dict:
    raw_dir = OUT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    actual_path = raw_dir / "central_margin_effect.float32.npy"
    predicted_path = raw_dir / "vjp_predicted_effect.float32.npy"
    rms_path = raw_dir / "intervention_state_rms.float32.npy"
    actual = np.lib.format.open_memmap(actual_path, mode="r+" if actual_path.exists() else "w+", dtype=np.float32, shape=(len(rows), 3))
    predicted = np.lib.format.open_memmap(predicted_path, mode="r+" if predicted_path.exists() else "w+", dtype=np.float32, shape=(len(rows), 3))
    state_rms = np.lib.format.open_memmap(rms_path, mode="r+" if rms_path.exists() else "w+", dtype=np.float32, shape=(len(rows), 3))
    progress_path = raw_dir / "progress.json"
    completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"]) if progress_path.exists() else 0
    p2448_rows = read_rows(P2448 / "index/vjp_rows.jsonl")
    p2448_lookup = {row["case_id"]: index for index, row in enumerate(p2448_rows)}
    gradient = np.load(P2448 / "raw/query_margin_vjp.float32.npy", mmap_mode="r")
    model = tokenizer = None
    if completed < len(rows):
        model, tokenizer, _ = capability.load_model("qwen4b")
        model.eval()
        modules = field_utils.modules(model)
        module = modules[QPOINT]
        device = model.get_input_embeddings().weight.device
    else:
        module = device = None
    try:
        family_lookup = {family: index for index, family in enumerate(families)}
        language_lookup = {language: index for index, language in enumerate(("en", "zh"))}
        for row_index in range(completed, len(rows)):
            row = rows[row_index]
            family_index = family_lookup[row["family"]]
            language_index = language_lookup[row["language"]]
            base_direction = directions[language_index, family_index]
            control_directions = (base_direction, np.roll(base_direction, SHIFT), directions[language_index, permutation[family_index]])
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            positions = torch.arange(ids.shape[1], device=device)[None]
            token_index = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}["query_end"]
            target, foil = int(row["target_ids"][0]), int(row["foil_ids"][0])
            row_gradient = np.asarray(gradient[p2448_lookup[row["case_id"]], QPOINT], dtype=np.float32)
            for control_index, direction in enumerate(control_directions):
                margins = []
                observed_rms = []
                for sign in (-1.0, 1.0):
                    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

                    def intervention(_module, _inputs, result, sign=sign, direction_tensor=direction_tensor):
                        tensor = result[0] if isinstance(result, tuple) else result
                        altered = tensor.clone()
                        rms = tensor[0, token_index].detach().float().square().mean().sqrt()
                        delta = (sign * DOSE * rms * direction_tensor).to(dtype=altered.dtype)
                        altered[0, token_index] = altered[0, token_index] + delta
                        observed_rms.append(float(rms.cpu()))
                        return (altered,) + tuple(result[1:]) if isinstance(result, tuple) else altered

                    handle = module.register_forward_hook(intervention)
                    try:
                        with torch.inference_mode():
                            output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                            margins.append(float((output.logits[0, -1, target] - output.logits[0, -1, foil]).float().cpu()))
                    finally:
                        handle.remove()
                rms_value = float(np.mean(observed_rms))
                actual[row_index, control_index] = (margins[1] - margins[0]) / 2.0
                predicted[row_index, control_index] = DOSE * rms_value * float(np.dot(row_gradient.astype(np.float64), direction.astype(np.float64)))
                state_rms[row_index, control_index] = rms_value
            actual.flush(); predicted.flush(); state_rms.flush()
            save(progress_path, {"completed": row_index + 1, "rows": len(rows), "qpoint": QPOINT, "dose_rms_fraction": DOSE})
            if (row_index + 1) % 32 == 0 or row_index + 1 == len(rows):
                print(f"[phase2451 finite] {row_index + 1}/{len(rows)}", flush=True)
            del ids, mask, positions
    finally:
        del model, tokenizer
        close(gradient)
        for value in (actual, predicted, state_rms):
            value.flush(); close(value)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    index_path = OUT / "index/intervention_rows.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("".join(json.dumps({key: row[key] for key in ("case_id", "config_id", "family", "unit", "language", "direction", "variant", "query_role")}, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"actual": str(actual_path), "predicted": str(predicted_path), "state_rms": str(rms_path),
            "rows": len(rows), "controls": list(CONTROLS), "qpoint": QPOINT, "dose_rms_fraction": DOSE,
            "forward_passes": len(rows) * len(CONTROLS) * 2, "inference": "Qwen3-4B BF16 CUDA finite symmetric intervention"}


def interactions(rows: list[dict], values: np.ndarray) -> dict:
    lookup = {(int(row["unit"]), row["family"], row["language"], int(row["direction"]), row["variant"], row["query_role"]): index for index, row in enumerate(rows)}
    families = sorted({row["family"] for row in rows})
    output = {}
    for unit in (4, 5):
        semantic_cells = []
        lexical_cells = []
        for family in families:
            for language in ("en", "zh"):
                for direction in (0, 1):
                    role_differences = {}
                    for variant in VARIANTS:
                        source = lookup[(unit, family, language, direction, variant, "source")]
                        target = lookup[(unit, family, language, direction, variant, "target")]
                        role_differences[variant] = values[target] - values[source]
                    semantic_cells.append(role_differences["valid"] - role_differences["broken_a"])
                    lexical_cells.append(role_differences["broken_a"] - role_differences["broken_b"])
        output[f"unit{unit}"] = {"semantic": np.stack(semantic_cells), "lexical": np.stack(lexical_cells)}
    return output


def summarize_matrix(matrix: np.ndarray) -> dict:
    means = np.mean(matrix, axis=0)
    return {"matched": float(means[0]), "shift791": float(means[1]), "family_permuted": float(means[2]),
            "matched_minus_shift": float(means[0] - means[1]), "matched_minus_family": float(means[0] - means[2]),
            "matched_positive_fraction": float(np.mean(matrix[:, 0] > 0)),
            "matched_median": float(np.median(matrix[:, 0]))}


def analyze(rows: list[dict], collection: dict, families: list[str], permutation: np.ndarray) -> dict:
    actual = np.load(collection["actual"], mmap_mode="r")
    predicted = np.load(collection["predicted"], mmap_mode="r")
    actual64 = np.asarray(actual, dtype=np.float64)
    predicted64 = np.asarray(predicted, dtype=np.float64)
    residual = actual64 - predicted64
    linearity = {"correlation": float(np.corrcoef(actual64.reshape(-1), predicted64.reshape(-1))[0, 1]),
                 "relative_rmse": float(np.sqrt(np.mean(residual ** 2)) / max(np.sqrt(np.mean(actual64 ** 2)), 1e-30)),
                 "sign_agreement": float(np.mean(np.sign(actual64) == np.sign(predicted64)))}
    actual_interactions = interactions(rows, actual64)
    predicted_interactions = interactions(rows, predicted64)
    summary = {}
    for unit in (4, 5):
        summary[f"unit{unit}"] = {
            "actual_semantic": summarize_matrix(actual_interactions[f"unit{unit}"]["semantic"]),
            "actual_lexical": summarize_matrix(actual_interactions[f"unit{unit}"]["lexical"]),
            "predicted_semantic": summarize_matrix(predicted_interactions[f"unit{unit}"]["semantic"]),
            "predicted_lexical": summarize_matrix(predicted_interactions[f"unit{unit}"]["lexical"]),
        }
    close(actual); close(predicted)
    held_lockbox = all(summary[f"unit{unit}"]["actual_semantic"][key] > 0 for unit in (4, 5) for key in ("matched_minus_shift", "matched_minus_family"))
    positive_direction = all(summary[f"unit{unit}"]["actual_semantic"]["matched"] > 0 for unit in (4, 5))
    return {"families": families, "family_permutation": permutation.tolist(), "linearity": linearity,
            "summary": summary, "finite_semantic_direction_lockbox": held_lockbox and positive_direction}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 自动续研——冻结语义VJP方向的双unit有限对称扰动（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 从Phase2448 natural-unit3在冻结q18得到的每语言×family语义gradient护照构造RMS归一化方向。对unit4/5共384条natural、三variant、双角色、中英双方向八族样本，在query-end实际注入$\pm1\%$当前HiddenState RMS的方向；比较同family同坐标、+791坐标移位和固定family错配，共2304次前向。以中心差分测真实margin效应，并用存档VJP预测同一效应；随后构造valid−brokenA语义交互与brokenA−brokenB词项控制。

$$\delta=0.01\,\operatorname{{RMS}}(H)d,\qquad C(d)=\frac{{m(H+\delta)-m(H-\delta)}}2,$$
$$I_{{sem}}^C=(C_t-C_s)_{{valid}}-(C_t-C_s)_{{brokenA}}.$$

**结果汇总。** 设置 `{json.dumps(result['collection'], ensure_ascii=False)}`；局部线性质量门 `{json.dumps(result['analysis']['linearity'], ensure_ascii=False)}`；双unit双交互 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2451_c39121_c39440_vjp_finite_intervention.py`；实际/预测中心效应、HiddenState RMS、384行索引和final位于同名结果目录；冻结方向来自Phase2448全坐标护照。

**分析与理论进展。** 该测试不删除坐标，而是在模型原轨迹附近沿候选方向做可逆小扰动。若两个held unit的真实语义交互都呈现匹配方向大于物理移位和family错配，并与VJP预测一致，则输出条件纹理不仅可观察，而且在局部有限尺度进入真实margin计算。

**问题硬伤与结论。** 1%单剂量仍只覆盖局部邻域；方向由unit3数据估计，不能证明唯一性或必要性；改变第一分歧token margin不等于完整自然生成。失败不会否定分布式纹理，成功也只构成局部充分性候选，不是语言编码机制闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = selected_rows()
    families = sorted({row["family"] for row in rows})
    directions, permutation = load_directions(families)
    collection = capture(rows, families, directions, permutation)
    analysis = analyze(rows, collection, families, permutation)
    adjudication = {"finite_semantic_direction_lockbox": analysis["finite_semantic_direction_lockbox"],
                    "local_vjp_prediction_supported": analysis["linearity"]["correlation"] > .9 and analysis["linearity"]["sign_agreement"] > .9,
                    "finite_intervention_semantic_candidate": analysis["finite_semantic_direction_lockbox"] and analysis["linearity"]["correlation"] > .9,
                    "language_encoding_mechanism_closed": False}
    checks = {"rows_384": collection["rows"] == 384, "forward_passes_2304": collection["forward_passes"] == 2304,
              "three_controls": len(collection["controls"]) == 3, "eight_families": len(families) == 8,
              "all_files": all(Path(path).exists() for path in (collection["actual"], collection["predicted"], collection["state_rms"])),
              "finite": all(math.isfinite(value) for value in analysis["linearity"].values()) and all(math.isfinite(value) for unit in analysis["summary"].values() for interaction in unit.values() for value in interaction.values()),
              "claim_boundary": not adjudication["language_encoding_mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
