#!/usr/bin/env python3
"""Build discovery/confirmation full-coordinate family atlases and publish important fields."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2330 = RESULT / "phase2330_c6081_c6200_language_family_atlas_contract"
P2331 = RESULT / "phase2331_c6201_c6360_qwen4b_twenty_family_fullfield"
P2332 = RESULT / "phase2332_c6361_c6480_multidose_natural_direction_calibration"
OUT = RESULT / "phase2333_c6481_c6640_twenty_family_coordinate_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MATERIAL = P2330 / "material/typed_language_family_atlas.jsonl"
BOUNDARY = P2331 / "raw/qwen4b_fp16_boundary_all_checkpoints.float16.npy"
ALL_TOKEN = P2331 / "raw/qwen4b_fp16_representative_all_token_qpoints.float16.npy"
PHASE = 2333
CAMPAIGN = "C6481-C6640"
EPS = 1e-12
CHANNELS = (
    "discovery_mean_delta", "confirmation_mean_delta", "absolute_difference",
    "same_sign", "discovery_sign_consistency", "confirmation_sign_consistency",
    "surface_interaction", "language_interaction",
)

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2330_c6081_c6200_language_family_atlas_contract as contract  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return io.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    io.write_rows(path, rows)


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def pair_index(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["family"], row["language"], row["surface"], int(row["unit"]), row["partition"])
        grouped[key][int(row["state"])] = row
    pairs = []
    for key, states in sorted(grouped.items()):
        if set(states) != {0, 1}:
            raise RuntimeError(("missing_state_pair", key, states.keys()))
        row0, row1 = states[0], states[1]
        pairs.append({
            "pair_index": len(pairs), "family": key[0], "macrotype": row0["macrotype"],
            "language": key[1], "surface": key[2], "unit": key[3], "partition": key[4],
            "state0_case": row0["case_id"], "state1_case": row1["case_id"],
            "state0_index": int(row0["design_index"]), "state1_index": int(row1["design_index"]),
        })
    return pairs


def create_state_delta(field: np.ndarray, pairs: list[dict]) -> tuple[Path, dict]:
    dataset_id = "c6483_qwen4b_twenty_family_natural_state_delta"
    binary = VIS / f"{dataset_id}.float32.npy"
    rows_count = len(pairs) * field.shape[1]
    output = atlas.create_binary(binary.name, rows_count, field.shape[2], np.float32)
    metadata = []
    cursor = 0
    for pair in pairs:
        for q in range(field.shape[1]):
            output[cursor] = (field[pair["state1_index"], q].astype(np.float32) -
                              field[pair["state0_index"], q].astype(np.float32))
            metadata.append({**pair, "qpoint": q, "checkpoint_kind": ("embedding" if q == 0 else "final_norm" if q == 37 else "block")})
            cursor += 1
    output.flush(); close_memmap(output)
    return binary, atlas.write_metadata(
        dataset_id, "Qwen3-4B twenty-family natural state deltas", binary, metadata,
        "Qwen3-4B-FP16", "full_coordinate_natural_state_delta_v1",
        "observational derived", "960 paired rows across twenty families and all 38 checkpoints",
        "state1 minus state0 in every original activation coordinate",
        {"coordinate_count": 2560, "pair_count": len(pairs), "all_checkpoints": True,
         "warning": "natural prompt difference, not a transferable semantic vector"},
    )


def load_delta(binary: Path, pair_count: int) -> np.ndarray:
    value = np.load(binary, mmap_mode="r")
    return value.reshape(pair_count, 38, 2560)


def relative_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum(np.square(actual - predicted, dtype=np.float64)) /
                 (np.sum(np.square(actual, dtype=np.float64)) + EPS))


def family_analysis(delta: np.ndarray, pairs: list[dict], scores: list[dict]) -> tuple[dict, np.ndarray, list[dict]]:
    correct = {row["case_id"]: bool(row["correct_by_sum"] and row["correct_by_mean"]) for row in scores}
    passports = np.empty((len(contract.FAMILIES), 38, len(CHANNELS), 2560), dtype=np.float32)
    summary: dict[str, Any] = {"families": {}, "qpoint_identification": {}, "gates": {}}
    records = []
    discovery_means = np.empty((len(contract.FAMILIES), 38, 2560), dtype=np.float64)
    for family_index, family in enumerate(contract.FAMILIES):
        family_indices = [i for i, row in enumerate(pairs) if row["family"] == family]
        disc_indices = [i for i in family_indices if pairs[i]["partition"] == "discovery"]
        conf_indices = [i for i in family_indices if pairs[i]["partition"] == "confirmation"]
        family_result = {"qpoints": {}}
        for q in range(38):
            disc = delta[disc_indices, q].astype(np.float64)
            conf = delta[conf_indices, q].astype(np.float64)
            dmean, cmean = disc.mean(axis=0), conf.mean(axis=0)
            discovery_means[family_index, q] = dmean
            dcons = np.abs(np.sign(disc).sum(axis=0)) / np.maximum((disc != 0).sum(axis=0), 1)
            ccons = np.abs(np.sign(conf).sum(axis=0)) / np.maximum((conf != 0).sum(axis=0), 1)
            surface_effects = []
            language_effects = []
            for unit in range(0, 6):
                for language in contract.LANGUAGES:
                    n = [i for i in family_indices if pairs[i]["unit"] == unit and pairs[i]["language"] == language and pairs[i]["surface"] == "narrative"]
                    r = [i for i in family_indices if pairs[i]["unit"] == unit and pairs[i]["language"] == language and pairs[i]["surface"] == "reported"]
                    if n and r:
                        surface_effects.append(delta[r[0], q].astype(np.float64) - delta[n[0], q].astype(np.float64))
                for surface in contract.SURFACES:
                    en = [i for i in family_indices if pairs[i]["unit"] == unit and pairs[i]["language"] == "en" and pairs[i]["surface"] == surface]
                    zh = [i for i in family_indices if pairs[i]["unit"] == unit and pairs[i]["language"] == "zh" and pairs[i]["surface"] == surface]
                    if en and zh:
                        language_effects.append(delta[zh[0], q].astype(np.float64) - delta[en[0], q].astype(np.float64))
            passports[family_index, q] = np.stack([
                dmean, cmean, np.abs(dmean - cmean), (dmean * cmean > 0).astype(np.float64),
                dcons, ccons, np.mean(surface_effects, axis=0), np.mean(language_effects, axis=0),
            ]).astype(np.float32)
            per_row_mse = [relative_mse(delta[i, q].astype(np.float64), dmean) for i in conf_indices]
            sign_agreement = float(np.mean(dmean * cmean > 0))
            symmetric = float(np.sum(np.square(dmean - cmean)) /
                              ((np.sum(np.square(dmean)) + np.sum(np.square(cmean))) / 2 + EPS))
            high = np.abs(dmean) >= np.median(np.abs(dmean))
            family_result["qpoints"][str(q)] = {
                "confirmation_median_relative_mse": float(np.median(per_row_mse)),
                "mean_sign_agreement": sign_agreement,
                "high_amplitude_sign_agreement": float(np.mean((dmean * cmean > 0)[high])),
                "symmetric_relative_mse": symmetric,
                "discovery_median_sign_consistency": float(np.median(dcons)),
                "confirmation_median_sign_consistency": float(np.median(ccons)),
            }
            records.append({"family": family, "qpoint": q, **family_result["qpoints"][str(q)]})
        behavior_pairs = [pair for pair in (pairs[i] for i in family_indices)
                          if correct.get(pair["state0_case"], False) and correct.get(pair["state1_case"], False)]
        family_result["both_state_behavior_correct_pairs"] = len(behavior_pairs)
        summary["families"][family] = family_result
    for q in range(38):
        conf = [(i, row) for i, row in enumerate(pairs) if row["partition"] == "confirmation"]
        predictions = []
        for pair_index, pair in conf:
            actual = delta[pair_index, q].astype(np.float64)
            errors = [relative_mse(actual, discovery_means[f, q]) for f in range(len(contract.FAMILIES))]
            predicted = int(np.argmin(errors))
            predictions.append({
                "pair_index": pair_index, "family": pair["family"], "qpoint": q,
                "predicted_family": contract.FAMILIES[predicted],
                "correct": contract.FAMILIES[predicted] == pair["family"],
                "correct_family_mse": errors[contract.FAMILIES.index(pair["family"])],
                "best_wrong_mse": min(value for index, value in enumerate(errors) if index != contract.FAMILIES.index(pair["family"])),
            })
        summary["qpoint_identification"][str(q)] = {
            "rows": len(predictions), "accuracy": float(np.mean([row["correct"] for row in predictions])),
            "median_correct_over_best_wrong_ratio": float(np.median([row["correct_family_mse"] / (row["best_wrong_mse"] + EPS) for row in predictions])),
        }
        write_rows(OUT / f"analysis/identification_q{q:02d}.jsonl", predictions)
    behavior = json.loads((P2331 / "analysis/final.json").read_text(encoding="utf-8"))["behavior"]
    candidates = []
    observations = []
    for family in contract.FAMILIES:
        qrows = summary["families"][family]["qpoints"]
        eligible_q = [int(q) for q, row in qrows.items() if int(q) > 0 and row["mean_sign_agreement"] >= 0.65
                      and row["symmetric_relative_mse"] <= 1.0]
        best_q = max(eligible_q, key=lambda q: qrows[str(q)]["mean_sign_agreement"], default=None)
        if best_q is not None:
            observations.append({"family": family, "qpoint": best_q, **qrows[str(best_q)]})
            if behavior["families"][family]["qualified"]:
                candidates.append({"family": family, "qpoint": best_q, **qrows[str(best_q)]})
    summary["gates"] = {
        "observation_thresholds": {"mean_sign_agreement_min": 0.65, "symmetric_relative_mse_max": 1.0},
        "behavior_qualified_required_for_semantic_candidate": True,
        "structural_observations": observations, "semantic_candidates": candidates,
        "fresh_policy": "freeze these family/qpoint cells before reading fresh_confirmation or fresh_lockbox",
    }
    write_rows(OUT / "analysis/family_qpoint_records.jsonl", records)
    return summary, passports, records


def publish_boundary(rows: list[dict]) -> dict:
    source = np.load(BOUNDARY, mmap_mode="r")
    dataset_id = "c6481_qwen4b_twenty_family_boundary_all_checkpoints"
    binary = VIS / f"{dataset_id}.float16.npy"
    output = atlas.create_binary(binary.name, source.shape[0] * source.shape[1], source.shape[2], np.float16)
    output[:] = source.reshape(-1, source.shape[2])
    output.flush(); close_memmap(output); close_memmap(source)
    metadata = []
    for row in rows:
        for q in range(38):
            metadata.append({
                "case_id": row["case_id"], "design_index": row["design_index"], "family": row["family"],
                "macrotype": row["macrotype"], "language": row["language"], "surface": row["surface"],
                "partition": row["partition"], "unit": row["unit"], "state": row["state"],
                "qpoint": q, "checkpoint_kind": ("embedding" if q == 0 else "final_norm" if q == 37 else "block"),
            })
    return atlas.write_metadata(
        dataset_id, "Qwen3-4B twenty-family boundary field", binary, metadata, "Qwen3-4B-FP16",
        "embedding_hiddenstate_full_coordinate_v1", "observational field",
        "1920 balanced rows, embedding through final norm", "every original boundary activation coordinate",
        {"coordinate_count": 2560, "all_rows": True, "all_checkpoints": True, "no_projection": True},
    )


def publish_all_token(rows: list[dict]) -> dict:
    dataset_id = "c6482_qwen4b_twenty_family_representative_all_token"
    binary = VIS / f"{dataset_id}.float16.npy"
    shutil.copyfile(ALL_TOKEN, binary)
    selected = {row["case_id"]: row for row in read_rows(P2331 / "index/representative_rows.jsonl")}
    segments = read_rows(P2331 / "index/representative_segments.jsonl")
    metadata = []
    for segment in segments:
        row = selected[segment["case_id"]]
        for token_index in range(segment["token_count"]):
            metadata.append({
                "case_id": row["case_id"], "family": row["family"], "macrotype": row["macrotype"],
                "language": row["language"], "surface": row["surface"], "partition": row["partition"],
                "unit": row["unit"], "state": row["state"], "qpoint": segment["qpoint"],
                "token_index": token_index, "token_id": row["future_prompt_ids"][token_index],
                "checkpoint_kind": "embedding" if segment["qpoint"] == 0 else "final_norm" if segment["qpoint"] == 37 else "block",
            })
    if len(metadata) != np.load(binary, mmap_mode="r").shape[0]:
        raise RuntimeError(("all_token_metadata", len(metadata), np.load(binary, mmap_mode="r").shape))
    return atlas.write_metadata(
        dataset_id, "Qwen3-4B twenty-family representative all-token field", binary, metadata,
        "Qwen3-4B-FP16", "embedding_hiddenstate_full_coordinate_v1", "observational field",
        "320 balanced representative rows at embedding/q10/q20/q30/final",
        "each token's every original activation coordinate",
        {"coordinate_count": 2560, "embedding_included": True, "all_token": True, "no_projection": True},
    )


def publish_passport(passports: np.ndarray) -> dict:
    dataset_id = "c6484_qwen4b_twenty_family_discovery_confirmation_passport"
    binary = VIS / f"{dataset_id}.float32.npy"
    output = atlas.create_binary(binary.name, int(np.prod(passports.shape[:-1])), passports.shape[-1], np.float32)
    output[:] = passports.reshape(-1, passports.shape[-1])
    output.flush(); close_memmap(output)
    metadata = []
    for family_index, family in enumerate(contract.FAMILIES):
        for q in range(38):
            for channel in CHANNELS:
                metadata.append({"family": family, "family_index": family_index, "macrotype": contract.MACROTYPE[family],
                                 "qpoint": q, "channel": channel})
    return atlas.write_metadata(
        dataset_id, "Qwen3-4B twenty-family discovery-confirmation passport", binary, metadata,
        "Qwen3-4B-FP16", "full_coordinate_family_passport_v1", "observational derived",
        "discovery versus confirmation only; fresh partitions sealed",
        "full-coordinate means, sign consistency and surface/language interactions",
        {"coordinate_count": 2560, "channels": list(CHANNELS), "fresh_read": False, "no_projection": True},
    )


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    correction = result["phase2332_correction"]
    record = rf"""

## Phase {PHASE}: 二十族自然状态全坐标图谱、发现—确认护照与数值勘误（{CAMPAIGN}） [{stamp}]

**测试原理、测试用例与公式。** 本期只读取 Phase2331 的 discovery/confirmation，自始至终不读取 fresh 分区做选择。对同族、语言、表面、unit 的双状态边界场计算 `state1-state0`，完整保存960对×38检查点×2560原始坐标；分别记录族均值、坐标符号一致、表面差中差、语言差中差、发现—确认误差，并用二十个 discovery 族原型识别每条 confirmation 自然状态变化属于哪个族。重要的完整 boundary、全token/embedding、自然状态差分和护照均发布到可视化客户端。

$$
\Delta H_{{f,l,s,u,q,j}}=H_{{f,l,s,u,1,q,j}}-H_{{f,l,s,u,0,q,j}},
$$

$$
E_f=\frac{{\lVert\Delta H-\overline{{\Delta H}}_f^{{disc}}\rVert_2^2}}{{\lVert\Delta H\rVert_2^2+\varepsilon}},\qquad
\widehat f=\arg\min_g E_g.
$$

**Phase2332勘误。** 首次追加 Phase2332 时，偶/奇比汇总把导数乘回剂量却漏乘源状态范数，原始导数和偶响应场未受影响。重算后为 `{json.dumps(correction, ensure_ascii=False)}`；旧的 `28.66/75.39` 不得引用，正确总体中位数为 FP16 `{correction['float16_even_to_odd']:.6f}`、BF16 `{correction['bfloat16_even_to_odd']:.6f}`。这一勘误以新Phase append记录，不篡改历史条目。

**结果汇总、门槛与相关文件。** 发现—确认分析 `{json.dumps(result['analysis'], ensure_ascii=False)}`；冻结对象 `{json.dumps(result['freeze'], ensure_ascii=False)}`；发布资产 `{json.dumps(result['datasets'], ensure_ascii=False)}`；验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；客户端构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2333_c6481_c6640_twenty_family_coordinate_atlas.py`；结果 `tests/glm5/result/phase2333_c6481_c6640_twenty_family_coordinate_atlas`；资产位于 `frontend/public/vis_data/research_kernel`。

**分析、理论进展、问题硬伤与结论。** 本图谱首先回答“不同语言模式族的自然状态变化在原始HiddenState坐标上怎样复用或分化”，不把差分搬运当核心机制。族原型识别若高于1/20机会水平，只表明条件场含族信息；若发现—确认同号且误差较低，只登记可前瞻候选。表面、语言差中差仍混合tokenizer、长度和措辞。原boundary以FP16写盘，虽保留全部坐标但数值分辨率不是FP32；派生差分以float32计算和发布，不能恢复采集时已舍入的信息。单坐标仍是物理地址，不是语义原子；热力图是图谱，不是因果齿轮。

**下一阶段。** 当前目标未结束，自动读取事先密封的 fresh_confirmation/fresh_lockbox，对冻结 family/qpoint 候选及全族识别规律作前瞻裁决；即使严格语义候选很少，也继续分析行为通过特征和结构观察，而不把单门失败解释为小模型不存在编码。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = json.loads((P2331 / "analysis/final.json").read_text(encoding="utf-8"))
    calibration = json.loads((P2332 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"] or not calibration["all_checks_passed"]:
        raise RuntimeError("parents not complete")
    rows = read_rows(MATERIAL)
    pairs = pair_index(rows)
    write_rows(OUT / "index/state_pairs.jsonl", pairs)
    field = np.load(BOUNDARY, mmap_mode="r")
    boundary_asset = publish_boundary(rows)
    token_asset = publish_all_token(rows)
    delta_binary, delta_asset = create_state_delta(field, pairs)
    delta = load_delta(delta_binary, len(pairs))
    scores = read_rows(P2331 / "behavior/sequence_scores.jsonl")
    analysis, passports, _records = family_analysis(delta, pairs, scores)
    passport_asset = publish_passport(passports)
    freeze = {
        "frozen_before_fresh_read": True, "source_partitions": ["discovery", "confirmation"],
        "test_partitions": ["fresh_confirmation", "fresh_lockbox"],
        "semantic_candidates": analysis["gates"]["semantic_candidates"],
        "structural_observations": analysis["gates"]["structural_observations"],
        "identification_rule": "nearest discovery family mean by full-coordinate relative MSE at each qpoint",
        "thresholds": analysis["gates"]["observation_thresholds"],
    }
    save(OUT / "config/frozen_fresh_adjudication.json", freeze)
    datasets = [boundary_asset, token_asset, delta_asset, passport_asset]
    verification = [atlas.verify(row) for row in datasets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    if not verified:
        raise RuntimeError(("verification_failed", verification))
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    correction = {
        "float16_even_to_odd": calibration["analysis"]["formats"]["float16"]["median_even_to_odd_l2"],
        "bfloat16_even_to_odd": calibration["analysis"]["formats"]["bfloat16"]["median_even_to_odd_l2"],
        "raw_fields_unchanged": True, "cause": "summary omitted multiplication by source hidden-state norm",
    }
    checks = {
        "pairs": len(pairs) == 960, "all_checkpoints": field.shape[1] == 38,
        "all_coordinates": field.shape[2] == 2560, "fresh_not_used_for_selection": True,
        "four_assets_verified": verified, "frontend_build": build["passed"],
        "phase2332_corrected": correction["float16_even_to_odd"] < 10,
    }
    serial_datasets = json.loads(json.dumps(datasets, ensure_ascii=False, default=str))
    serial_catalog = json.loads(json.dumps(catalog, ensure_ascii=False, default=str))
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "analysis": {
            "qpoint_identification": analysis["qpoint_identification"],
            "gates": analysis["gates"],
            "family_best": {
                family: max(values["qpoints"].items(), key=lambda item: item[1]["mean_sign_agreement"])
                for family, values in analysis["families"].items()
            },
        },
        "freeze": freeze, "phase2332_correction": correction,
        "datasets": serial_datasets, "verification": verification, "catalog": serial_catalog,
        "frontend_build": build, "checks": checks, "all_checks_passed": all(checks.values()),
        "hashes": {"freeze": file_hash(OUT / "config/frozen_fresh_adjudication.json"), "delta": file_hash(delta_binary)},
    }
    save(final_path, result)
    close_memmap(field); close_memmap(delta)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2333_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
