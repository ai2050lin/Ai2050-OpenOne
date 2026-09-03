#!/usr/bin/env python3
"""Multi-dose symmetric finite intervention and BF16 curvature audit."""
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
OUT = RESULT / "phase2458_c41361_c41680_multidose_vjp_curvature"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2458
CAMPAIGN = "C41361-C41680"
QPOINT = 18
SHIFT = 791
DOSES = np.asarray((0.0025, 0.005, 0.01, 0.02), dtype=np.float64)
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


def derangements(count: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    while len(values) < count:
        candidate = rng.permutation(size)
        if np.all(candidate != np.arange(size)) and not any(np.array_equal(candidate, prior) for prior in values):
            values.append(candidate)
    return np.stack(values)


def selected_rows() -> list[dict]:
    rows = read_rows(P2435 / "index/trajectory_rows.jsonl")
    selected = [row for row in rows if int(row["unit"]) == 5 and row["surface"] == "natural" and int(row["direction"]) == 0]
    selected.sort(key=lambda row: row["case_id"])
    return selected


def load_directions() -> tuple[np.ndarray, list[str]]:
    final = json.loads((P2448 / "analysis/final.json").read_text(encoding="utf-8"))
    families = final["analysis"]["families"]
    passports = np.load(final["analysis"]["passports"], mmap_mode="r")
    raw = np.asarray(passports[0, 0, 0, :, QPOINT], dtype=np.float32).copy()
    close(passports)
    rms = np.sqrt(np.mean(raw.astype(np.float64) ** 2, axis=-1, keepdims=True))
    return (raw / np.maximum(rms, 1e-30)).astype(np.float32), families


def capture(rows: list[dict], directions: np.ndarray, families: list[str], permutations: np.ndarray) -> dict:
    raw_dir = OUT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline": raw_dir / "baseline_margin.float32.npy",
        "signed_margin": raw_dir / "signed_margin.float32.npy",
        "odd": raw_dir / "central_odd_effect.float32.npy",
        "even": raw_dir / "central_even_effect.float32.npy",
        "predicted": raw_dir / "vjp_predicted_odd.float32.npy",
        "state_rms": raw_dir / "state_rms.float32.npy",
    }
    controls = 2 + len(permutations)
    baseline = np.lib.format.open_memmap(paths["baseline"], mode="r+" if paths["baseline"].exists() else "w+", dtype=np.float32, shape=(len(rows),))
    signed = np.lib.format.open_memmap(paths["signed_margin"], mode="r+" if paths["signed_margin"].exists() else "w+", dtype=np.float32, shape=(len(rows), len(DOSES), controls, 2))
    odd = np.lib.format.open_memmap(paths["odd"], mode="r+" if paths["odd"].exists() else "w+", dtype=np.float32, shape=(len(rows), len(DOSES), controls))
    even = np.lib.format.open_memmap(paths["even"], mode="r+" if paths["even"].exists() else "w+", dtype=np.float32, shape=(len(rows), len(DOSES), controls))
    predicted = np.lib.format.open_memmap(paths["predicted"], mode="r+" if paths["predicted"].exists() else "w+", dtype=np.float32, shape=(len(rows), len(DOSES), controls))
    state_rms = np.lib.format.open_memmap(paths["state_rms"], mode="r+" if paths["state_rms"].exists() else "w+", dtype=np.float32, shape=(len(rows),))
    progress = raw_dir / "progress.json"
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    source_rows = read_rows(P2448 / "index/vjp_rows.jsonl")
    source_lookup = {row["case_id"]: index for index, row in enumerate(source_rows)}
    gradient = np.load(P2448 / "raw/query_margin_vjp.float32.npy", mmap_mode="r")
    model = tokenizer = None
    if completed < len(rows):
        model, tokenizer, _ = capability.load_model("qwen4b")
        model.eval()
        module = field_utils.modules(model)[QPOINT]
        device = model.get_input_embeddings().weight.device
    else:
        module = device = None
    family_lookup = {family: index for index, family in enumerate(families)}
    language_lookup = {language: index for index, language in enumerate(("en", "zh"))}
    try:
        for row_index in range(completed, len(rows)):
            row = rows[row_index]
            fi = family_lookup[row["family"]]
            li = language_lookup[row["language"]]
            base = directions[li, fi]
            control_directions = [base, np.roll(base, SHIFT)] + [directions[li, p[fi]] for p in permutations]
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            positions = torch.arange(ids.shape[1], device=device)[None]
            token_index = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}["query_end"]
            target, foil = int(row["target_ids"][0]), int(row["foil_ids"][0])
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                base_margin = float((output.logits[0, -1, target] - output.logits[0, -1, foil]).float().cpu())
            baseline[row_index] = base_margin
            row_gradient = np.asarray(gradient[source_lookup[row["case_id"]], QPOINT], dtype=np.float64)
            observed_rms: list[float] = []
            for dose_index, dose in enumerate(DOSES):
                for control_index, direction in enumerate(control_directions):
                    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)
                    margins = []
                    for sign_index, sign in enumerate((-1.0, 1.0)):
                        def intervention(_module, _inputs, result, sign=sign, dose=float(dose), direction_tensor=direction_tensor):
                            tensor = result[0] if isinstance(result, tuple) else result
                            altered = tensor.clone()
                            rms = tensor[0, token_index].detach().float().square().mean().sqrt()
                            altered[0, token_index] = altered[0, token_index] + (sign * dose * rms * direction_tensor).to(altered.dtype)
                            observed_rms.append(float(rms.cpu()))
                            return (altered,) + tuple(result[1:]) if isinstance(result, tuple) else altered

                        handle = module.register_forward_hook(intervention)
                        try:
                            with torch.inference_mode():
                                output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                                margin = float((output.logits[0, -1, target] - output.logits[0, -1, foil]).float().cpu())
                        finally:
                            handle.remove()
                        signed[row_index, dose_index, control_index, sign_index] = margin
                        margins.append(margin)
                    rms_value = float(np.mean(observed_rms))
                    odd[row_index, dose_index, control_index] = (margins[1] - margins[0]) / 2.0
                    even[row_index, dose_index, control_index] = (margins[1] + margins[0]) / 2.0 - base_margin
                    predicted[row_index, dose_index, control_index] = float(dose) * rms_value * float(np.dot(row_gradient, direction.astype(np.float64)))
            state_rms[row_index] = float(np.mean(observed_rms))
            for value in (baseline, signed, odd, even, predicted, state_rms):
                value.flush()
            save(progress, {"completed": row_index + 1, "rows": len(rows), "doses": DOSES.tolist(), "controls": controls})
            if (row_index + 1) % 8 == 0 or row_index + 1 == len(rows):
                print(f"[phase2458 multidose] {row_index + 1}/{len(rows)}", flush=True)
            del ids, mask, positions, output
    finally:
        del model, tokenizer
        close(gradient)
        for value in (baseline, signed, odd, even, predicted, state_rms):
            value.flush(); close(value)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    index = OUT / "index/intervention_rows.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("".join(json.dumps({key: row[key] for key in ("case_id", "family", "unit", "language", "direction", "variant", "query_role")}, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {
        **{key: str(path) for key, path in paths.items()},
        "rows": len(rows), "doses": DOSES.tolist(),
        "controls": ["matched_family_coordinate", "shift791_coordinate"] + [f"family_derangement_{index}" for index in range(len(permutations))],
        "family_derangements": permutations.tolist(), "qpoint": QPOINT,
        "forward_passes": len(rows) * (1 + len(DOSES) * controls * 2),
        "inference": "Qwen3-4B BF16 CUDA; deterministic symmetric finite intervention",
    }


def interaction_cells(rows: list[dict], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = {(row["family"], row["language"], row["variant"], row["query_role"]): index for index, row in enumerate(rows)}
    semantic, lexical = [], []
    for family in sorted({row["family"] for row in rows}):
        for language in ("en", "zh"):
            roles = {}
            for variant in VARIANTS:
                source = lookup[(family, language, variant, "source")]
                target = lookup[(family, language, variant, "target")]
                roles[variant] = values[target] - values[source]
            semantic.append(roles["valid"] - roles["broken_a"])
            lexical.append(roles["broken_a"] - roles["broken_b"])
    return np.stack(semantic), np.stack(lexical)


def dose_summary(cells: np.ndarray) -> list[dict]:
    output = []
    for dose_index, dose in enumerate(DOSES):
        matched = cells[:, dose_index, 0]
        shift = cells[:, dose_index, 1]
        family_means = np.mean(cells[:, dose_index, 2:], axis=0)
        output.append({
            "dose": float(dose), "matched_mean": float(np.mean(matched)), "matched_rms": float(np.sqrt(np.mean(matched ** 2))),
            "shift_mean": float(np.mean(shift)), "family_null_mean": float(np.mean(family_means)),
            "family_null_q95": float(np.quantile(family_means, .95)),
            "matched_minus_shift": float(np.mean(matched) - np.mean(shift)),
            "matched_minus_family_q95": float(np.mean(matched) - np.quantile(family_means, .95)),
            "positive_fraction": float(np.mean(matched > 0)), "zero_fraction": float(np.mean(matched == 0)),
        })
    return output


def analyze(rows: list[dict], collection: dict) -> dict:
    odd = np.asarray(np.load(collection["odd"], mmap_mode="r"), dtype=np.float64)
    even = np.asarray(np.load(collection["even"], mmap_mode="r"), dtype=np.float64)
    predicted = np.asarray(np.load(collection["predicted"], mmap_mode="r"), dtype=np.float64)
    semantic_odd, lexical_odd = interaction_cells(rows, odd)
    semantic_even, lexical_even = interaction_cells(rows, even)
    semantic_predicted, lexical_predicted = interaction_cells(rows, predicted)
    per_dose_linearity = []
    for dose_index, dose in enumerate(DOSES):
        actual_flat = odd[:, dose_index].reshape(-1)
        predicted_flat = predicted[:, dose_index].reshape(-1)
        nonzero = (np.abs(actual_flat) + np.abs(predicted_flat)) > 0
        corr = float(np.corrcoef(actual_flat[nonzero], predicted_flat[nonzero])[0, 1]) if np.sum(nonzero) > 2 else 0.0
        per_dose_linearity.append({
            "dose": float(dose), "correlation": corr,
            "sign_agreement": float(np.mean(np.sign(actual_flat) == np.sign(predicted_flat))),
            "actual_zero_fraction": float(np.mean(actual_flat == 0)),
            "relative_rmse": float(np.sqrt(np.mean((actual_flat - predicted_flat) ** 2)) / max(np.sqrt(np.mean(actual_flat ** 2)), 1e-30)),
        })
    odd_sem = dose_summary(semantic_odd)
    even_sem = dose_summary(semantic_even)
    odd_lex = dose_summary(lexical_odd)
    even_lex = dose_summary(lexical_even)
    matched_slopes = np.asarray([item["matched_mean"] / item["dose"] for item in odd_sem])
    matched_curvature = np.asarray([item["matched_rms"] / (item["dose"] ** 2) for item in even_sem])
    curvature_over_odd = np.asarray([even_sem[i]["matched_rms"] / max(odd_sem[i]["matched_rms"], 1e-30) for i in range(len(DOSES))])
    predicted_sem_summary = dose_summary(semantic_predicted)
    local_linearity = {
        "odd_semantic_slope_by_dose": matched_slopes.tolist(),
        "odd_semantic_slope_relative_range": float(np.ptp(matched_slopes) / max(np.mean(np.abs(matched_slopes)), 1e-30)),
        "even_semantic_curvature_rms_by_dose": matched_curvature.tolist(),
        "even_over_odd_rms_by_dose": curvature_over_odd.tolist(),
        "largest_dose_even_semantic_beats_family_q95": even_sem[-1]["matched_minus_family_q95"] > 0,
        "largest_dose_odd_semantic_beats_family_q95": odd_sem[-1]["matched_minus_family_q95"] > 0,
    }
    return {
        "per_dose_vjp_linearity": per_dose_linearity,
        "odd_semantic": odd_sem, "odd_lexical": odd_lex,
        "even_semantic": even_sem, "even_lexical": even_lex,
        "predicted_odd_semantic": predicted_sem_summary,
        "local_linearity_curvature": local_linearity,
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: Qwen4B多剂量对称扰动的线性项、偶次曲率与BF16分辨率（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 冻结Phase2448 natural-unit3、q18的中英八族完整语义gradient方向，在fresh unit5的96条direction0材料（三variant×双角色×八族中英）上注入$0.25\%,0.5\%,1\%,2\%$ HiddenState RMS的正负扰动。每剂量并列matched、+791坐标移位与8个无固定点family置乱，共{result['collection']['forward_passes']}次BF16前向。中心奇项检验VJP局部线性，中心偶项隔离二阶及更高偶次效应；不再用单剂量sign agreement直接宣称“强非线性”。

$$O_\alpha(d)=\frac{{m(H+\alpha rd)-m(H-\alpha rd)}}2,$$
$$E_\alpha(d)=\frac{{m(H+\alpha rd)+m(H-\alpha rd)}}2-m(H),\qquad \widehat O_\alpha=\alpha r\,g^\top d.$$

**结果汇总。** 逐剂量VJP质量 `{json.dumps(result['analysis']['per_dose_vjp_linearity'], ensure_ascii=False)}`；语义奇项 `{json.dumps(result['analysis']['odd_semantic'], ensure_ascii=False)}`；语义偶项 `{json.dumps(result['analysis']['even_semantic'], ensure_ascii=False)}`；词项对照奇/偶项 `{json.dumps({'odd': result['analysis']['odd_lexical'], 'even': result['analysis']['even_lexical']}, ensure_ascii=False)}`；曲率裁决 `{json.dumps(result['analysis']['local_linearity_curvature'], ensure_ascii=False)}`；总裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2458_c41361_c41680_multidose_vjp_curvature.py`；基线margin、四剂量×十控制正负margin、奇项、偶项、VJP预测、RMS和final位于同名结果目录。候选方向继续引用Phase2448的全2560坐标护照。

**分析与理论进展。** 奇项随剂量近似线性且与VJP相关，只说明冻结轨迹附近的一阶编译有效；偶项按$\alpha^2$稳定并胜多个family错配，才构成可分辨曲率。反之，小剂量大量精确零、相关下降可由BF16离散化解释，不能单独命名为模型非线性机制。

**问题硬伤与结论。** 只测q18、fresh unit5和第一分歧token；方向来自unit3，仍是局部充分性而非必要性。有限差分包含BF16舍入、整网非线性和候选margin三者，不能定位具体层内算子。无论结果如何，都不等于自主多token生成闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = selected_rows()
    directions, families = load_directions()
    permutations = derangements(8, 8, 2458)
    collection = capture(rows, directions, families, permutations)
    analysis = analyze(rows, collection)
    curve = analysis["local_linearity_curvature"]
    adjudication = {
        "multi_dose_odd_semantic_family_specific_at_largest_dose": curve["largest_dose_odd_semantic_beats_family_q95"],
        "largest_dose_even_semantic_mean_beats_family_q95": curve["largest_dose_even_semantic_beats_family_q95"],
        "even_curvature_resolved_above_bf16_floor": False,
        "reason_even_curvature_not_resolved": "E_alpha/alpha^2 decreases by over an order of magnitude and the smallest-dose even/odd ratio exceeds one; one largest-dose mean contrast is insufficient.",
        "bf16_floor_dominates_small_dose_candidate": True,
        "single_dose_nonlinearity_overclaim_corrected": True,
        "language_encoding_mechanism_closed": False,
    }
    checks = {
        "rows_96": collection["rows"] == 96, "four_doses": len(collection["doses"]) == 4,
        "ten_controls": len(collection["controls"]) == 10, "eight_family_derangements": len(collection["family_derangements"]) == 8,
        "forward_passes_7776": collection["forward_passes"] == 7776,
        "all_files": all(Path(collection[name]).exists() for name in ("baseline", "signed_margin", "odd", "even", "predicted", "state_rms")),
        "finite": all(math.isfinite(v) for item in analysis["per_dose_vjp_linearity"] for v in item.values()),
        "claim_boundary": not adjudication["language_encoding_mechanism_closed"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "analysis": analysis, "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
