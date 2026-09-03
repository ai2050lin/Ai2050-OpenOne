#!/usr/bin/env python3
"""Exact-coordinate and random-mask causal identification for Phase 2277."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT_OUT = RESULT / "phase2274_c1721_c1770_broad_construction_contract"
FIELD_OUT = RESULT / "phase2275_c1771_c1820_qwen4b_broad_fullfield"
STRUCTURE_OUT = RESULT / "phase2276_c1821_c1890_full_coordinate_structure_tournament"
OUT = RESULT / "phase2277_c1891_c1960_coordinate_causal_identification"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2250_c1049_c1064_state_consistent_causal as causal_base  # noqa: E402
import phase2274_c1721_c1770_broad_construction_contract as contract  # noqa: E402


PHASE = 2277
CAMPAIGN = "C1891-C1960"
FIELD = FIELD_OUT / "raw/qwen3_4b_broad_role_field.float16.npy"
INDEX = FIELD_OUT / "raw/role_field_index.jsonl"
COMPILED = CONTRACT_OUT / "material/broad_construction_qwen_compiled.jsonl"
FAMILY_SETTINGS = {
    "property_state": {"checkpoint": 15, "role": "query", "role_index": 4,
                       "provenance": "Phase2268 historical full-vector deletion anchor"},
    "patient_binding": {"checkpoint": 11, "role": "query", "role_index": 4,
                        "provenance": "Phase2276 fresh-lockbox predictive structure"},
    "location_state": {"checkpoint": 11, "role": "query", "role_index": 4,
                       "provenance": "Phase2276 fresh-lockbox predictive structure"},
}
BATCH = 64
RANDOM_MASKS = 128
SINGLE_GATES = {"mean_margin_gain": 0.001, "direction_rate": 2 / 3, "minimum_coordinates": 4}
CAUSAL_GATES = {"minimum_pairs": 12, "direction_rate": 2 / 3,
                "margin_advantage_over_controls": 0.05, "generation_advantage": 0.10}
CONTROL_NAMES = ("wrong_coordinates", "opposite_sign", "random_0", "random_1", "random_2")


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def pair_rows(index: list[dict]) -> list[dict]:
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in index:
        if row["family"] not in FAMILY_SETTINGS:
            continue
        key = (row["family"], int(row["unit"]), row["surface"], row["partition"])
        groups[key][int(row["state"])] = row
    output = []
    for key, states in sorted(groups.items()):
        if set(states) != {0, 1}:
            raise RuntimeError(("incomplete_pair", key))
        output.append({
            "family": key[0], "unit": key[1], "surface": key[2], "partition": key[3],
            "state0_case": states[0]["case_id"], "state1_case": states[1]["case_id"],
            "state0_index": int(states[0]["hidden_index"]), "state1_index": int(states[1]["hidden_index"]),
        })
    return output


def candidate_margin_batch(logits: torch.Tensor, row: dict, target_position: int) -> np.ndarray:
    ids = [int(value[0]) for value in row["candidate_ids"]]
    other = 1 - target_position
    return (logits[:, ids[target_position]] - logits[:, ids[other]]).float().cpu().numpy()


def batch_margins(model, row: dict, device, q: int, role: str,
                  vectors: np.ndarray, target_position: int) -> np.ndarray:
    base = model.model
    seq = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
    ids = seq[None, :].repeat(len(vectors), 1)
    mask = torch.ones_like(ids)
    pos_ids = mask.long().cumsum(-1) - 1
    patch = torch.tensor(vectors, dtype=next(model.parameters()).dtype, device=device)
    token_position = int(row["role_positions"][role][-1])

    def hook(_module, _args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        changed = tensor.clone()
        changed[:, token_position, :] = changed[:, token_position, :] + patch
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = base.layers[q - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=mask, position_ids=pos_ids,
                           use_cache=False, return_dict=True)
        logits = result.logits[:, -1, :]
    finally:
        handle.remove()
        del ids, mask, pos_ids, patch
    return candidate_margin_batch(logits, row, target_position)


def no_patch_margin(model, row: dict, device, target_position: int) -> float:
    values = batch_margins(model, row, device, 1, "query",
                           np.zeros((1, 2560), dtype=np.float32), target_position)
    return float(values[0])


def single_coordinate_effects(model, row: dict, device, q: int, role: str,
                              response: np.ndarray, target_position: int, sign: float) -> tuple[np.ndarray, float]:
    natural = no_patch_margin(model, row, device, target_position)
    output = np.empty(response.shape[0], dtype=np.float32)
    for start in range(0, response.shape[0], BATCH):
        coordinates = np.arange(start, min(start + BATCH, response.shape[0]))
        vectors = np.zeros((len(coordinates), response.shape[0]), dtype=np.float32)
        vectors[np.arange(len(coordinates)), coordinates] = sign * response[coordinates]
        output[coordinates] = batch_margins(model, row, device, q, role, vectors, target_position) - natural
    return output, natural


def balanced_masks(seed: int, rows: int, coordinates: int = 2560) -> np.ndarray:
    rng = np.random.default_rng(seed)
    masks = np.zeros((rows, coordinates), dtype=np.uint8)
    for i in range(rows):
        masks[i, rng.permutation(coordinates)[:coordinates // 2]] = 1
    return masks


def masks_with_exact_size(seed: int, rows: int, size: int, coordinates: int = 2560) -> np.ndarray:
    if not 0 <= size <= coordinates:
        raise ValueError((size, coordinates))
    rng = np.random.default_rng(seed)
    masks = np.zeros((rows, coordinates), dtype=np.uint8)
    for i in range(rows):
        masks[i, rng.permutation(coordinates)[:size]] = 1
    return masks


def random_mask_effects(model, row: dict, device, q: int, role: str, response: np.ndarray,
                        target_position: int, masks: np.ndarray, natural: float) -> np.ndarray:
    values = np.empty(len(masks), dtype=np.float32)
    for start in range(0, len(masks), BATCH):
        subset = masks[start:start + BATCH].astype(np.float32)
        vectors = -subset * response[None, :]
        values[start:start + len(subset)] = batch_margins(
            model, row, device, q, role, vectors, target_position) - natural
    return values


def vector_conditions(response: np.ndarray, mask: np.ndarray, random_masks: list[np.ndarray],
                      direction_sign: float) -> dict[str, np.ndarray]:
    count = int(mask.sum())
    wrong = np.roll(mask, 257)
    if int(wrong.sum()) != count:
        raise RuntimeError("wrong mask changed size")
    output = {
        "candidate": direction_sign * response * mask,
        "wrong_coordinates": direction_sign * response * wrong,
        "opposite_sign": -direction_sign * response * mask,
        "full": direction_sign * response,
    }
    for i, random in enumerate(random_masks):
        output[f"random_{i}"] = direction_sign * response * random
    return output


def free_positions(tokenizer, row: dict) -> dict[str, list[int]]:
    ids = row["free_prompt_ids"]
    positions = {}
    for role, value in row["role_values"].items():
        spans = contract.previous.legacy.parent.contextual_spans(tokenizer, ids, value)
        if not spans:
            raise RuntimeError((row["case_id"], role, value, "free_prompt"))
        positions[role] = spans[-1] if role == "query" else spans[0]
    positions["boundary"] = [len(ids) - 1]
    return positions


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def generated_text(model, tokenizer, row: dict, device, q: int, role: str, vector: np.ndarray) -> str:
    positions = free_positions(tokenizer, row)
    bundle = np.zeros((len(contract.ROLES), vector.shape[0]), dtype=np.float32)
    bundle[contract.ROLES.index(role)] = vector
    return causal_base.generated_text(model, tokenizer, row, device, positions,
                                      [(q, bundle, tuple(range(len(contract.ROLES))))], "one_shot")


def summarize_condition(rows: list[dict], direction: str) -> dict:
    subset = [row for row in rows if row["direction"] == direction]
    if not subset:
        return {"pairs": 0, "passes": False, "reason": "no_rows"}
    gains = np.asarray([row["candidate_margin"] - row["natural_margin"] for row in subset], dtype=np.float32)
    best_control = np.asarray([max(row[f"{name}_margin"] for name in CONTROL_NAMES) for row in subset], dtype=np.float32)
    candidate_generation = float(np.mean([row["candidate_generation"] for row in subset]))
    best_control_generation = float(max(np.mean([row[f"{name}_generation"] for row in subset])
                                        for name in CONTROL_NAMES))
    summary = {
        "pairs": len(subset),
        "direction_rate": float(np.mean(gains > 0)),
        "mean_margin_gain": float(gains.mean()),
        "margin_advantage_over_controls": float(np.mean(
            np.asarray([row["candidate_margin"] for row in subset]) - best_control)),
        "candidate_generation_accuracy": candidate_generation,
        "best_control_generation_accuracy": best_control_generation,
        "generation_advantage": candidate_generation - best_control_generation,
        "full_direction_rate": float(np.mean([
            row["full_margin"] - row["natural_margin"] > 0 for row in subset])),
    }
    summary["passes"] = bool(
        summary["pairs"] >= CAUSAL_GATES["minimum_pairs"] and
        summary["direction_rate"] >= CAUSAL_GATES["direction_rate"] and
        summary["margin_advantage_over_controls"] >= CAUSAL_GATES["margin_advantage_over_controls"] and
        summary["generation_advantage"] >= CAUSAL_GATES["generation_advantage"]
    )
    return summary


def append_clean_memo(result: dict) -> None:
    current = MEMO.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in current:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 单坐标扫描、随机掩码与最小联盟因果裁决（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 本期把 Phase2268 的整条 2560 维状态删除拆成单坐标效应与坐标联盟。冻结对象为历史锚点 `property_state q15-query`，以及 Phase2276 前瞻锁箱通过的 `patient_binding q11-query`、`location_state q11-query`。识别集逐样本扫描全部 2560 个坐标，不按幅值排序，也不使用 Top-K；验证集只检验冻结联盟，并以错坐标、反号、完整响应和三个严格等规模随机坐标联盟为控制。一次初始运行在候选联盟形成前被实现审计中止：随机控制错误地固定为半密度。该运行没有揭示候选、验证或锁箱结果，不进入证据；修复为等规模控制后从头完整重跑。

**公式。** 单坐标删除效应为：

$$
e_{{i,j}}=M_i\left(H_i^1-R_{{i,j}}e_j\right)-M_i(H_i^1).
$$

候选坐标不用 Top-K，而使用冻结的绝对门槛：

$$
S_f=\operatorname{{Select}}_j\left(
\operatorname{{mean}}_i e_{{i,j}}\ge 0.001,\quad
\operatorname{{mean}}_i\mathbf 1[e_{{i,j}}>0]\ge\frac{{2}}{{3}}\right).
$$

对 128 个精确半密度随机掩码，仅登记真实联合效应与单坐标加和的偏差：

$$
I_{{i,m}}=M_i(H_i^1-z_m\odot R_i)-M_i(H_i^1)-\sum_j z_{{m,j}}e_{{i,j}}.
$$

**结果汇总。** 单坐标识别：`{json.dumps(result['single_coordinate_identification'], ensure_ascii=False)}`。随机掩码非加性：`{json.dumps(result['random_mask_nonadditivity'], ensure_ascii=False)}`。冻结候选联盟：`{json.dumps(result['candidate_masks'], ensure_ascii=False)}`。验证：`{json.dumps(result['validation'], ensure_ascii=False)}`。锁箱授权族：`{json.dumps(result['lockbox_authorized_families'], ensure_ascii=False)}`。锁箱调用/删除：`{json.dumps(result['lockbox_summaries'], ensure_ascii=False)}`。严格双向因果族：`{json.dumps(result['strict_bidirectional_families'], ensure_ascii=False)}`。逐坐标矩阵：`{json.dumps(result['coordinate_atlas'], ensure_ascii=False)}`。哈希与检查：`{json.dumps(result['hashes'], ensure_ascii=False)}`、`{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析与理论进展。** {result['strict_conclusion']} 单坐标效应只表示特定位置、层、角色和样本状态下的局部激活敏感性；联盟通过验证才表示这些坐标共同具有可复验的行为效力。即使双向通过，也不能称为唯一电路，因为其他 token、角色和检查点尚未穷举。随机掩码的加和误差只检验局部可加近似，不预设流形、李群或新守恒律。

**问题、硬伤与瓶颈。** 干预仍以自然状态差为局部测量探针；一次只写入一个角色位置；修改后的激活可能离开自然分布；单坐标效应受 bfloat16 前向精度限制；固定 `0.001` 门槛可能漏掉大量微弱协同坐标；随机掩码只能显示非加性，不能唯一恢复高阶超边；自由生成仍是一次前缀写入，不能代表持续多 token 因果通道。

**结论与下一步。** 下一阶段只把获得锁箱资格的模型本地结构带到 Qwen3-14B，在相对层深窗口内重做行为、全坐标预测和候选联盟效力；4B 与 14B 不对齐物理坐标编号。脚本 `tests/glm5/phase2277_c1891_c1960_coordinate_causal_identification.py`；结果 `tests/glm5/result/phase2277_c1891_c1960_coordinate_causal_identification`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def append_memo(result: dict) -> None:
    return append_clean_memo(result)
    current = MEMO.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in current:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 单坐标扫描、随机掩码与最小联盟因果裁决（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 本期把 Phase2268 的“整条 2560 维状态删除”拆成单个激活坐标与坐标联盟。三个预冻结对象为：历史属性锚点 `property_state q15-query`，以及 Phase2276 锁箱通过的 `patient_binding q11-query`、`location_state q11-query`。对 fresh-confirmation 的前四个 unit，逐样本、逐坐标只删除自然状态响应 $R_j=H^1_j-H^0_j$ 的一个坐标；候选坐标不按幅值排序，而按固定绝对门和跨样本方向率形成完整布尔掩码。后四个 unit 只验证整个候选联盟；fresh lockbox 最后同时测试调用、删除、错坐标、反号、三组等大小随机坐标、完整响应和自由生成。

**公式。** 单坐标删除效应为：

$$
e_{{i,j}}=M_i\!\left(H^1_i-R_{{i,j}}e_j\right)-M_i(H^1_i).
$$

候选坐标不使用 Top-K：

$$
S_f=\left\{{j:\operatorname{{mean}}_i e_{{i,j}}\ge0.001,
\quad \operatorname{{mean}}_i\mathbf 1[e_{{i,j}}>0]\ge\frac23\right\}}.
$$

同时使用 128 个精确半密度随机掩码 $z_m$，比较真实组合效应与单坐标加和：

$$
I_{{i,m}}=M_i(H^1_i-z_m\odot R_i)-M_i(H^1_i)-\sum_jz_{{m,j}}e_{{i,j}}.
$$

该量只登记非加性偏差，不预设流形、李群或新守恒律。

**结果汇总。** 单坐标识别 `{json.dumps(result['single_coordinate_identification'], ensure_ascii=False)}`；随机掩码非加性 `{json.dumps(result['random_mask_nonadditivity'], ensure_ascii=False)}`；冻结候选联盟 `{json.dumps(result['candidate_masks'], ensure_ascii=False)}`；验证 `{json.dumps(result['validation'], ensure_ascii=False)}`；锁箱授权族 `{json.dumps(result['lockbox_authorized_families'], ensure_ascii=False)}`；锁箱调用/删除 `{json.dumps(result['lockbox_summaries'], ensure_ascii=False)}`；严格双向因果族 `{json.dumps(result['strict_bidirectional_families'], ensure_ascii=False)}`；逐坐标矩阵 `{json.dumps(result['coordinate_atlas'], ensure_ascii=False)}`；哈希与检查 `{json.dumps(result['hashes'], ensure_ascii=False)}`、`{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析与理论进展。** `{result['strict_conclusion']}` 单坐标效应成立只表示该位置、层、角色和样本状态下的局部激活敏感性；候选联盟通过才表示这些坐标共同具有可复验行为效力；即使双向通过，也不是唯一电路，因为未扫描其他 token、角色与检查点。随机掩码的加和误差用于判断“齿轮”是否近似独立：误差大时，单坐标账不能简单相加，需要保留坐标联合条件。

**问题、硬伤与瓶颈。** 干预仍使用自然状态差作为局部测量探针；一次只写一个角色位置；激活修改可能离开自然分布；单坐标效应受 bfloat16 前向精度限制；固定 0.001 门槛可能漏掉大量微弱协同坐标；随机掩码只能揭示存在非加性，不能唯一恢复高阶超边；自由生成是一次前缀写入，不能代表持续多 token 因果通道。

**结论与下一步。** 下一阶段只把获得锁箱资格的模型本地结构带到 Qwen3-14B，在相对层深窗口内重做行为、逐坐标预测和候选掩码效应；4B 与 14B 不对齐物理坐标编号。脚本 `tests/glm5/phase2277_c1891_c1960_coordinate_causal_identification.py`；结果 `tests/glm5/result/phase2277_c1891_c1960_coordinate_causal_identification`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    for sub in ("protocol", "analysis", "raw", "atlas"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    structure = load(STRUCTURE_OUT / "analysis/final.json")
    if not structure["all_checks_passed"]:
        raise RuntimeError("Phase2276 is invalid")
    index = read_rows(INDEX)
    pairs = pair_rows(index)
    compiled = {row["case_id"]: row for row in read_rows(COMPILED)}
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in pairs:
        by_family[row["family"]].append(row)
    random_masks = balanced_masks(2277001, RANDOM_MASKS)
    random_mask_path = OUT / "protocol/balanced_random_masks.uint8.npy"
    np.save(random_mask_path, random_masks)
    prereg = {
        "phase": PHASE, "campaign": CAMPAIGN, "frozen_before_model": True,
        "family_settings": FAMILY_SETTINGS, "batch": BATCH, "random_masks": RANDOM_MASKS,
        "identification_units": [16, 17, 18, 19], "validation_units": [20, 21, 22, 23],
        "lockbox_units": list(range(24, 32)), "single_gates": SINGLE_GATES,
        "causal_gates": CAUSAL_GATES, "controls": list(CONTROL_NAMES),
        "candidate_policy": "all coordinates crossing absolute gain and direction gates; no Top-K",
        "validation_random_control_policy": "three independently seeded masks with exactly the candidate-mask coordinate count",
        "forbidden": ["attention", "MLP", "weight", "gradient", "PCA", "Top-K", "lockbox reselection"],
    }
    save(OUT / "protocol/preregistration.json", prereg)
    save(OUT / "protocol/aborted_run_control_construction_audit.json", {
        "status": "aborted_before_candidate_mask_freeze_and_before_validation_or_lockbox_reveal",
        "reason": "initial implementation generated fixed half-density validation controls instead of candidate-size-matched controls",
        "action": "discarded the incomplete scan and restarted the full phase after implementing exact-size random masks",
        "observed_before_abort": "progress counters only; no candidate, validation, lockbox, or generation result",
        "scientific_use": "none",
    })
    field = np.load(FIELD, mmap_mode="r")
    model = None
    identification_values = []
    identification_rows = []
    random_rows = []
    candidate_masks = {}
    validation_rows = []
    validation_summaries = {}
    lockbox_rows = []
    try:
        model, tokenizer, device, placement = contract.previous.legacy.parent.model_base.qwen_model()
        for family, setting in FAMILY_SETTINGS.items():
            q, role = int(setting["checkpoint"]), setting["role"]
            role_i = int(setting["role_index"])
            identify = [row for row in by_family[family]
                        if row["partition"] == "fresh_confirmation" and row["unit"] in (16, 17, 18, 19)]
            family_effects = []
            for pair_i, pair in enumerate(identify):
                source = compiled[pair["state1_case"]]
                response = (np.asarray(field[pair["state1_index"], q, role_i], np.float32) -
                            np.asarray(field[pair["state0_index"], q, role_i], np.float32))
                target_position = 1 - int(source["gold_position"])
                effects, natural = single_coordinate_effects(
                    model, source, device, q, role, response, target_position, -1.0)
                family_effects.append(effects)
                identification_rows.append({"row": len(identification_rows), "family": family,
                                            "unit": pair["unit"], "surface": pair["surface"],
                                            "source_case": pair["state1_case"], "natural_margin": natural})
                identification_values.append(effects)
                actual = random_mask_effects(model, source, device, q, role, response,
                                             target_position, random_masks, natural)
                predicted = random_masks.astype(np.float32) @ effects
                random_rows.append({
                    "family": family, "unit": pair["unit"], "surface": pair["surface"],
                    "masks": RANDOM_MASKS,
                    "actual_mean": float(actual.mean()), "predicted_mean": float(predicted.mean()),
                    "additivity_mae": float(np.mean(np.abs(actual - predicted))),
                    "sign_agreement": float(np.mean((actual > 0) == (predicted > 0))),
                    "actual_direction_rate": float(np.mean(actual > 0)),
                })
                print(f"[identify] {family} {pair_i + 1}/{len(identify)}", flush=True)
            matrix = np.stack(family_effects)
            mean_gain = matrix.mean(axis=0)
            direction = np.mean(matrix > 0, axis=0)
            mask = np.logical_and(mean_gain >= SINGLE_GATES["mean_margin_gain"],
                                  direction >= SINGLE_GATES["direction_rate"])
            candidate_masks[family] = mask
        identification_matrix = np.stack(identification_values).astype(np.float32)
        identification_path = OUT / "atlas/qwen4b_single_coordinate_margin_effect.float32.npy"
        np.save(identification_path, identification_matrix)
        write_rows(OUT / "atlas/qwen4b_single_coordinate_margin_effect.rows.jsonl", identification_rows)
        candidate_path = OUT / "protocol/frozen_candidate_masks.uint8.npy"
        candidate_matrix = np.stack([candidate_masks[family] for family in FAMILY_SETTINGS]).astype(np.uint8)
        np.save(candidate_path, candidate_matrix)
        candidate_meta = {
            family: {"row": i, "coordinates": int(candidate_masks[family].sum()),
                     "qualified": int(candidate_masks[family].sum()) >= SINGLE_GATES["minimum_coordinates"]}
            for i, family in enumerate(FAMILY_SETTINGS)
        }
        save(OUT / "protocol/frozen_candidate_masks.json", candidate_meta)

        validation_random = {
            family: list(masks_with_exact_size(
                2277100 + 10 * fi, 3, int(candidate_masks[family].sum())).astype(bool))
            for fi, family in enumerate(FAMILY_SETTINGS)
        }
        for family, setting in FAMILY_SETTINGS.items():
            mask = candidate_masks[family]
            validate = [row for row in by_family[family]
                        if row["partition"] == "fresh_confirmation" and row["unit"] in (20, 21, 22, 23)]
            if int(mask.sum()) < SINGLE_GATES["minimum_coordinates"]:
                validation_summaries[family] = {"pairs": len(validate), "passes": False,
                                                "reason": "candidate_mask_too_small",
                                                "coordinates": int(mask.sum())}
                continue
            for pair in validate:
                source = compiled[pair["state1_case"]]
                q, role_i = int(setting["checkpoint"]), int(setting["role_index"])
                response = (np.asarray(field[pair["state1_index"], q, role_i], np.float32) -
                            np.asarray(field[pair["state0_index"], q, role_i], np.float32))
                target = 1 - int(source["gold_position"])
                natural = no_patch_margin(model, source, device, target)
                conditions = vector_conditions(response, mask, validation_random[family], -1.0)
                margins = {name: float(batch_margins(model, source, device, q, setting["role"],
                                                     vector[None, :], target)[0])
                           for name, vector in conditions.items()}
                validation_rows.append({"family": family, "unit": pair["unit"],
                                        "surface": pair["surface"], "natural_margin": natural,
                                        **{f"{name}_margin": value for name, value in margins.items()}})
            subset = [row for row in validation_rows if row["family"] == family]
            gains = np.asarray([row["candidate_margin"] - row["natural_margin"] for row in subset])
            controls = np.asarray([max(row[f"{name}_margin"] for name in CONTROL_NAMES) for row in subset])
            summary = {"pairs": len(subset), "coordinates": int(mask.sum()),
                       "direction_rate": float(np.mean(gains > 0)), "mean_margin_gain": float(gains.mean()),
                       "margin_advantage_over_controls": float(np.mean(
                           np.asarray([row["candidate_margin"] for row in subset]) - controls))}
            summary["passes"] = bool(summary["pairs"] >= CAUSAL_GATES["minimum_pairs"] and
                                     summary["direction_rate"] >= CAUSAL_GATES["direction_rate"] and
                                     summary["margin_advantage_over_controls"] >=
                                     CAUSAL_GATES["margin_advantage_over_controls"])
            validation_summaries[family] = summary
        write_rows(OUT / "analysis/validation_trials.jsonl", validation_rows)
        save(OUT / "analysis/validation_summaries.json", validation_summaries)
        authorized = [family for family, row in validation_summaries.items() if row.get("passes")]
        save(OUT / "protocol/frozen_lockbox_authorization.json",
             {"families": authorized, "candidate_masks_sha256": file_hash(candidate_path)})

        for family in authorized:
            setting = FAMILY_SETTINGS[family]
            mask = candidate_masks[family]
            lock = [row for row in by_family[family] if row["partition"] == "fresh_lockbox"]
            for pair in lock:
                q, role_i = int(setting["checkpoint"]), int(setting["role_index"])
                response = (np.asarray(field[pair["state1_index"], q, role_i], np.float32) -
                            np.asarray(field[pair["state0_index"], q, role_i], np.float32))
                for direction, case_id, sign, target_code in (
                    ("call", pair["state0_case"], 1.0, compiled[pair["state0_case"]]["true_code"]),
                    ("delete", pair["state1_case"], -1.0, compiled[pair["state1_case"]]["false_code"]),
                ):
                    source = compiled[case_id]
                    target = 1 - int(source["gold_position"])
                    natural = no_patch_margin(model, source, device, target)
                    conditions = vector_conditions(response, mask, validation_random[family], sign)
                    margins = {name: float(batch_margins(model, source, device, q, setting["role"],
                                                         vector[None, :], target)[0])
                               for name, vector in conditions.items()}
                    texts = {name: generated_text(model, tokenizer, source, device, q, setting["role"], vector)
                             for name, vector in conditions.items()}
                    lockbox_rows.append({
                        "family": family, "unit": pair["unit"], "surface": pair["surface"],
                        "direction": direction, "source_case": case_id, "target_code": target_code,
                        "natural_margin": natural,
                        **{f"{name}_margin": value for name, value in margins.items()},
                        **{f"{name}_generation": parse_code(text, source) == target_code for name, text in texts.items()},
                    })
                print(f"[lockbox] {family} unit={pair['unit']} surface={pair['surface']}", flush=True)
        write_rows(OUT / "raw/lockbox_trials.jsonl", lockbox_rows)
    finally:
        close_mmap(field)
        if model is not None:
            contract.previous.legacy.parent.model_base.scope.parent.previous.model_base().release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    random_summary = {}
    for family in FAMILY_SETTINGS:
        subset = [row for row in random_rows if row["family"] == family]
        random_summary[family] = {
            "pairs": len(subset),
            "median_additivity_mae": float(np.median([row["additivity_mae"] for row in subset])),
            "median_sign_agreement": float(np.median([row["sign_agreement"] for row in subset])),
            "median_actual_direction_rate": float(np.median([row["actual_direction_rate"] for row in subset])),
        }
    single_summary = {}
    offset = 0
    for family in FAMILY_SETTINGS:
        subset = identification_matrix[offset:offset + 12]
        offset += 12
        single_summary[family] = {
            "pairs": len(subset), "coordinates": 2560,
            "positive_mean_coordinates": int(np.sum(subset.mean(axis=0) > 0)),
            "candidate_coordinates": int(candidate_masks[family].sum()),
            "median_absolute_mean_effect": float(np.median(np.abs(subset.mean(axis=0)))),
            "maximum_mean_effect": float(np.max(subset.mean(axis=0))),
        }
    lockbox_summaries = {family: {direction: summarize_condition(lockbox_rows, direction)
                                  for direction in ("call", "delete")}
                         for family in validation_summaries if validation_summaries[family].get("passes")}
    strict = [family for family, value in lockbox_summaries.items()
              if value["call"].get("passes") and value["delete"].get("passes")]
    coordinate_meta = {
        "effect_path": str(identification_path.relative_to(ROOT)),
        "effect_shape": list(identification_matrix.shape),
        "candidate_mask_path": str(candidate_path.relative_to(ROOT)),
        "candidate_mask_shape": list(candidate_matrix.shape),
    }
    checks = {
        "settings_frozen": set(FAMILY_SETTINGS) == {"property_state", "patient_binding", "location_state"},
        "single_scan_complete": identification_matrix.shape == (36, 2560),
        "random_masks_balanced": bool(np.all(random_masks.sum(axis=1) == 1280)),
        "candidate_masks_full_coordinate": candidate_matrix.shape == (3, 2560),
        "validation_complete": set(validation_summaries) == set(FAMILY_SETTINGS),
        "lockbox_only_authorized": all(row["family"] in [f for f, v in validation_summaries.items() if v.get("passes")]
                                       for row in lockbox_rows),
        "finite_effects": bool(np.isfinite(identification_matrix).all()),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "placement": placement,
        "settings": FAMILY_SETTINGS, "single_gates": SINGLE_GATES, "causal_gates": CAUSAL_GATES,
        "single_coordinate_identification": single_summary,
        "random_mask_nonadditivity": random_summary,
        "candidate_masks": candidate_meta,
        "validation": validation_summaries,
        "lockbox_authorized_families": [family for family, row in validation_summaries.items() if row.get("passes")],
        "lockbox_summaries": lockbox_summaries,
        "strict_bidirectional_families": strict,
        "coordinate_atlas": coordinate_meta,
        "hashes": {
            "preregistration": file_hash(OUT / "protocol/preregistration.json"),
            "random_masks": file_hash(random_mask_path),
            "single_effects": file_hash(identification_path),
            "candidate_masks": file_hash(candidate_path),
            "lockbox_authorization": file_hash(OUT / "protocol/frozen_lockbox_authorization.json"),
        },
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (f"{len(strict)}/{len(FAMILY_SETTINGS)} families passed exact-coordinate "
                              "identification, independent alliance validation, and bidirectional lockbox controls; "
                              "no result establishes a unique circuit."),
        "next_authorization": "Replicate qualified model-local topology on Qwen3-14B at relative depth.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
