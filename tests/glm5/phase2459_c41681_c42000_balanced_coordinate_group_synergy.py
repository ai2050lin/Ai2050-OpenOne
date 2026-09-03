#!/usr/bin/env python3
"""Balanced all-coordinate group decomposition of the semantic VJP direction."""
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
P2458 = RESULT / "phase2458_c41361_c41680_multidose_vjp_curvature"
OUT = RESULT / "phase2459_c41681_c42000_balanced_coordinate_group_synergy"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, QPOINT, DOSE, DIM = 2459, "C41681-C42000", 18, 0.02, 2560
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


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-30 else 0.0


def derangements(count: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = []
    while len(values) < count:
        candidate = rng.permutation(size)
        if np.all(candidate != np.arange(size)):
            values.append(candidate)
    return np.stack(values)


def selected_rows() -> list[dict]:
    rows = read_rows(P2435 / "index/trajectory_rows.jsonl")
    rows = [row for row in rows if int(row["unit"]) == 5 and row["surface"] == "natural" and int(row["direction"]) == 0]
    rows.sort(key=lambda row: row["case_id"])
    return rows


def load_directions() -> tuple[np.ndarray, list[str]]:
    final = json.loads((P2448 / "analysis/final.json").read_text(encoding="utf-8"))
    passports = np.load(final["analysis"]["passports"], mmap_mode="r")
    raw = np.asarray(passports[0, 0, 0, :, QPOINT], dtype=np.float32).copy()
    close(passports)
    rms = np.sqrt(np.mean(raw.astype(np.float64) ** 2, axis=-1, keepdims=True))
    return (raw / np.maximum(rms, 1e-30)).astype(np.float32), final["analysis"]["families"]


def masks() -> np.ndarray:
    coordinate = np.arange(DIM, dtype=np.uint16)
    values = []
    for bit in range(8):
        left = (((coordinate >> bit) & 1) == 0).astype(np.float32)
        values.extend((left, 1.0 - left))
    return np.stack(values)


def capture(rows: list[dict], directions: np.ndarray, families: list[str], group_masks: np.ndarray) -> dict:
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = {
        "odd": raw / "balanced_group_odd_effect.float32.npy",
        "even": raw / "balanced_group_even_effect.float32.npy",
        "predicted": raw / "balanced_group_vjp_prediction.float32.npy",
        "signed_margin": raw / "balanced_group_signed_margin.float32.npy",
    }
    odd = np.lib.format.open_memmap(paths["odd"], mode="r+" if paths["odd"].exists() else "w+", dtype=np.float32, shape=(len(rows), 16))
    even = np.lib.format.open_memmap(paths["even"], mode="r+" if paths["even"].exists() else "w+", dtype=np.float32, shape=(len(rows), 16))
    predicted = np.lib.format.open_memmap(paths["predicted"], mode="r+" if paths["predicted"].exists() else "w+", dtype=np.float32, shape=(len(rows), 16))
    signed = np.lib.format.open_memmap(paths["signed_margin"], mode="r+" if paths["signed_margin"].exists() else "w+", dtype=np.float32, shape=(len(rows), 16, 2))
    progress = raw / "progress.json"
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    baseline = np.load(P2458 / "raw/baseline_margin.float32.npy", mmap_mode="r")
    state_rms = np.load(P2458 / "raw/state_rms.float32.npy", mmap_mode="r")
    source_rows = read_rows(P2448 / "index/vjp_rows.jsonl")
    source_lookup = {row["case_id"]: index for index, row in enumerate(source_rows)}
    gradient = np.load(P2448 / "raw/query_margin_vjp.float32.npy", mmap_mode="r")
    model = tokenizer = None
    if completed < len(rows):
        model, tokenizer, _ = capability.load_model("qwen4b")
        model.eval(); module = field_utils.modules(model)[QPOINT]
        device = model.get_input_embeddings().weight.device
    else:
        module = device = None
    family_lookup = {family: i for i, family in enumerate(families)}
    language_lookup = {language: i for i, language in enumerate(("en", "zh"))}
    try:
        for row_index in range(completed, len(rows)):
            row = rows[row_index]
            base_direction = directions[language_lookup[row["language"]], family_lookup[row["family"]]]
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            attention = torch.ones_like(ids)
            positions = torch.arange(ids.shape[1], device=device)[None]
            token_index = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}["query_end"]
            target, foil = int(row["target_ids"][0]), int(row["foil_ids"][0])
            row_gradient = np.asarray(gradient[source_lookup[row["case_id"]], QPOINT], dtype=np.float64)
            for group_index, mask in enumerate(group_masks):
                group_direction = base_direction * mask
                direction_tensor = torch.tensor(group_direction, dtype=torch.float32, device=device)
                margins = []
                for sign_index, sign in enumerate((-1.0, 1.0)):
                    def intervention(_module, _inputs, result, sign=sign, direction_tensor=direction_tensor):
                        tensor = result[0] if isinstance(result, tuple) else result
                        altered = tensor.clone()
                        delta = sign * DOSE * float(state_rms[row_index]) * direction_tensor
                        altered[0, token_index] = altered[0, token_index] + delta.to(altered.dtype)
                        return (altered,) + tuple(result[1:]) if isinstance(result, tuple) else altered

                    handle = module.register_forward_hook(intervention)
                    try:
                        with torch.inference_mode():
                            output = model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False, return_dict=True)
                            margin = float((output.logits[0, -1, target] - output.logits[0, -1, foil]).float().cpu())
                    finally:
                        handle.remove()
                    signed[row_index, group_index, sign_index] = margin
                    margins.append(margin)
                odd[row_index, group_index] = (margins[1] - margins[0]) / 2.0
                even[row_index, group_index] = (margins[1] + margins[0]) / 2.0 - float(baseline[row_index])
                predicted[row_index, group_index] = DOSE * float(state_rms[row_index]) * float(np.dot(row_gradient, group_direction.astype(np.float64)))
            for value in (odd, even, predicted, signed): value.flush()
            save(progress, {"completed": row_index + 1, "rows": len(rows), "groups": 16})
            if (row_index + 1) % 8 == 0 or row_index + 1 == len(rows):
                print(f"[phase2459 groups] {row_index + 1}/{len(rows)}", flush=True)
            del ids, attention, positions, output
    finally:
        del model, tokenizer
        for value in (odd, even, predicted, signed): value.flush(); close(value)
        for value in (baseline, state_rms, gradient): close(value)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    index = OUT / "index/intervention_rows.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("".join(json.dumps({key: row[key] for key in ("case_id", "family", "unit", "language", "variant", "query_role")}, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    mask_path = OUT / "derived/balanced_coordinate_masks.uint8.npy"
    mask_path.parent.mkdir(parents=True, exist_ok=True); np.save(mask_path, group_masks.astype(np.uint8))
    return {**{key: str(path) for key, path in paths.items()}, "masks": str(mask_path), "rows": len(rows), "groups": 16,
            "partitions": 8, "dimension": DIM, "qpoint": QPOINT, "dose": DOSE, "forward_passes": len(rows) * 16 * 2,
            "group_rule": "For bit b=0..7 of the physical coordinate index, A_b and B_b are complementary half-coordinate masks; no renormalization and every coordinate is retained."}


def interaction_cells(rows: list[dict], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = {(row["family"], row["language"], row["variant"], row["query_role"]): i for i, row in enumerate(rows)}
    semantic, lexical = [], []
    for family in sorted({row["family"] for row in rows}):
        for language in ("en", "zh"):
            role = {}
            for variant in VARIANTS:
                source, target = lookup[(family, language, variant, "source")], lookup[(family, language, variant, "target")]
                role[variant] = values[target] - values[source]
            semantic.append(role["valid"] - role["broken_a"])
            lexical.append(role["broken_a"] - role["broken_b"])
    return np.stack(semantic), np.stack(lexical)


def analyze(rows: list[dict], collection: dict) -> dict:
    odd = np.asarray(np.load(collection["odd"], mmap_mode="r"), dtype=np.float64)
    even = np.asarray(np.load(collection["even"], mmap_mode="r"), dtype=np.float64)
    predicted = np.asarray(np.load(collection["predicted"], mmap_mode="r"), dtype=np.float64)
    full_odd = np.asarray(np.load(P2458 / "raw/central_odd_effect.float32.npy", mmap_mode="r"), dtype=np.float64)[:, 3, 0]
    full_even = np.asarray(np.load(P2458 / "raw/central_even_effect.float32.npy", mmap_mode="r"), dtype=np.float64)[:, 3, 0]
    full_predicted = np.asarray(np.load(P2458 / "raw/vjp_predicted_odd.float32.npy", mmap_mode="r"), dtype=np.float64)[:, 3, 0]
    sem_odd, lex_odd = interaction_cells(rows, odd)
    sem_even, lex_even = interaction_cells(rows, even)
    sem_pred, lex_pred = interaction_cells(rows, predicted)
    sem_full, lex_full = interaction_cells(rows, full_odd)
    sem_full_even, lex_full_even = interaction_cells(rows, full_even)
    sem_full_pred, lex_full_pred = interaction_cells(rows, full_predicted)
    derived = OUT / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    np.save(derived / "semantic_group_interaction.float32.npy", sem_odd.astype(np.float32))
    np.save(derived / "lexical_group_interaction.float32.npy", lex_odd.astype(np.float32))
    partition = {}
    for name, groups, full in (("semantic", sem_odd, sem_full), ("lexical", lex_odd, lex_full)):
        values = []
        for bit in range(8):
            summed = groups[:, 2 * bit] + groups[:, 2 * bit + 1]
            synergy = full - summed
            values.append({"partition_bit": bit, "full_mean": float(np.mean(full)), "halves_sum_mean": float(np.mean(summed)),
                           "synergy_mean": float(np.mean(synergy)), "synergy_rms": float(np.sqrt(np.mean(synergy ** 2))),
                           "relative_synergy_rms": float(np.sqrt(np.mean(synergy ** 2)) / max(np.sqrt(np.mean(full ** 2)), 1e-30))})
        partition[name] = values
    prediction_additivity = {
        "semantic_max_abs_partition_residual": float(max(np.max(np.abs(sem_full_pred - (sem_pred[:, 2*b] + sem_pred[:, 2*b+1]))) for b in range(8))),
        "lexical_max_abs_partition_residual": float(max(np.max(np.abs(lex_full_pred - (lex_pred[:, 2*b] + lex_pred[:, 2*b+1]))) for b in range(8))),
    }
    # Cell order is family-major then language.  Turn it into language,family,group.
    semantic_signature = sem_odd.reshape(8, 2, 16).transpose(1, 0, 2)
    lexical_signature = lex_odd.reshape(8, 2, 16).transpose(1, 0, 2)
    permutations = derangements(4096, 8, 2459)
    signature = {}
    for name, values in (("semantic", semantic_signature), ("lexical", lexical_signature)):
        observed = float(np.mean([cosine(values[0, family], values[1, family]) for family in range(8)]))
        null = np.asarray([np.mean([cosine(values[0, family], values[1, p[family]]) for family in range(8)]) for p in permutations])
        q95 = float(np.quantile(null, .95))
        signature[name] = {"same_family_crosslanguage": observed, "family_null_mean": float(np.mean(null)), "family_null_q95": q95,
                           "family_identity_advantage": observed - q95, "permutations": len(permutations)}
    curvature_partition = {
        "semantic_mean_relative_even_synergy": float(np.mean([v["relative_synergy_rms"] for v in partition["semantic"]])),
        "lexical_mean_relative_even_synergy": float(np.mean([v["relative_synergy_rms"] for v in partition["lexical"]])),
        "note": "odd central effects are used for partition synergy; even interaction arrays are retained separately",
        "semantic_even_group_rms": float(np.sqrt(np.mean(sem_even ** 2))),
        "semantic_full_even_rms": float(np.sqrt(np.mean(sem_full_even ** 2))),
        "lexical_even_group_rms": float(np.sqrt(np.mean(lex_even ** 2))),
        "lexical_full_even_rms": float(np.sqrt(np.mean(lex_full_even ** 2))),
    }
    return {"partition_odd_synergy": partition, "prediction_additivity": prediction_additivity,
            "group_signature_crosslanguage": signature, "curvature_partition": curvature_partition,
            "derived": {"semantic": str(derived / "semantic_group_interaction.float32.npy"), "lexical": str(derived / "lexical_group_interaction.float32.npy")}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 全2560坐标平衡二分的条件齿轮组协同与跨语言签名（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 不做Top-K、不丢低值坐标。对物理坐标编号的低8位逐位构造8个互补二分$A_b\cup B_b=\{{1,\ldots,2560\}}$，得到16个各含1280坐标的原幅值子方向；每个坐标恰在每个partition的一侧。对Phase2458同一fresh unit5、96条八族中英材料，在q18和2% RMS剂量分别注入每个子方向正负，共3072次前向。用完整方向效应与两半效应之和计算协同，并将16维组效应作为中英family签名，与4096个family错配比较。

$$d=d_{{A_b}}+d_{{B_b}},\qquad S_b=O(d)-O(d_{{A_b}})-O(d_{{B_b}}).$$

**结果汇总。** 组方案 `{json.dumps(result['collection'], ensure_ascii=False)}`；八个partition的语义/词项奇项协同 `{json.dumps(result['analysis']['partition_odd_synergy'], ensure_ascii=False)}`；VJP精确可加核对 `{json.dumps(result['analysis']['prediction_additivity'], ensure_ascii=False)}`；16维组签名跨语言裁决 `{json.dumps(result['analysis']['group_signature_crosslanguage'], ensure_ascii=False)}`；偶项留存与相对量 `{json.dumps(result['analysis']['curvature_partition'], ensure_ascii=False)}`；总裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2459_c41681_c42000_balanced_coordinate_group_synergy.py`；16×2560二值掩码、每行16组正负margin/奇偶项/VJP预测、语义及词项组interaction和final位于同名结果目录。Phase2448的完整2560坐标方向继续保留；本Phase分组不是原场替代品。

**分析与理论进展。** 这一步第一次直接问“完整候选方向由哪些互补坐标群协同实现”，而不是删除Top-K坐标。若VJP子效应严格相加而真实有限效应出现稳定$S_b$，说明非加性来自候选方向进入后续网络的有限动力学；若16维中英同family签名胜family错配，则平衡坐标群携带可复用的条件纹理。

**问题硬伤与结论。** 坐标编号bit分组是无监督测量基，不是模型天然模块；8种二分远未穷尽$2^{{2560}}$组合。由于BF16，有限协同含舍入台阶，且只覆盖q18、2%和第一token。特别更正Phase2458自动标签：2%偶项均值虽胜family q95，但$E_\alpha/\alpha^2$跨剂量不稳定，因此“稳定曲率已解析”不成立，只保留BF16地板以上可能存在有限非加性的候选。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = selected_rows(); directions, families = load_directions(); group_masks = masks()
    collection = capture(rows, directions, families, group_masks); analysis = analyze(rows, collection)
    semantic_synergy = [v["relative_synergy_rms"] for v in analysis["partition_odd_synergy"]["semantic"]]
    adjudication = {
        "balanced_group_signature_family_specific": analysis["group_signature_crosslanguage"]["semantic"]["family_identity_advantage"] > 0,
        "finite_partition_nonadditivity_observed": float(np.median(semantic_synergy)) > 0.1,
        "natural_coordinate_modules_identified": False,
        "phase2458_stable_curvature_claim_rejected": True,
        "language_encoding_mechanism_closed": False,
    }
    checks = {"rows_96": collection["rows"] == 96, "all_coordinates": bool(np.all(masks().sum(axis=0) == 8)),
              "eight_complementary_partitions": all(np.allclose(group_masks[2*b] + group_masks[2*b+1], 1) for b in range(8)),
              "forward_passes_3072": collection["forward_passes"] == 3072,
              "files": all(Path(collection[k]).exists() for k in ("odd", "even", "predicted", "signed_margin", "masks")),
              "finite": all(math.isfinite(v) for section in analysis["partition_odd_synergy"].values() for item in section for v in item.values()),
              "claim_boundary": not adjudication["language_encoding_mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
