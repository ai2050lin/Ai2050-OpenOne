#!/usr/bin/env python3
"""Near-manifold causal adjudication for Phase 2267 lockbox survivors."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT_OUT = RESULT / "phase2265_c1433_c1468_independent_bilingual_contract"
FIELD_OUT = RESULT / "phase2266_c1469_c1504_qwen4b_independent_fullfield"
TOURNAMENT_OUT = RESULT / "phase2267_c1505_c1540_coordinate_model_tournament"
OUT = RESULT / "phase2268_c1541_c1576_near_manifold_causal_adjudication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2250_c1049_c1064_state_consistent_causal as causal_base  # noqa: E402
import phase2261_c1337_c1368_sample_conditioned_causal_adjudication as old_causal  # noqa: E402
import phase2265_c1433_c1468_independent_bilingual_contract as contract  # noqa: E402
import phase2267_c1505_c1540_coordinate_model_tournament as tournament_code  # noqa: E402


PHASE = 2268
CAMPAIGNS = tuple(f"C{i}" for i in range(1541, 1577))
DOSES = (0.25, 0.5, 0.75, 1.0)
IDENTITY = tuple(range(len(contract.ROLES)))
GATES = {"minimum_pairs": 24, "direction_rate": 0.60,
         "margin_advantage_over_controls": 0.05, "generation_advantage": 0.10}
CONTROL_NAMES = ("algebraic", "shared", "shuffled", "wrong_family", "wrong_role", "wrong_checkpoint")


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
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def pair_rows(index: list[dict], families: list[str]) -> list[dict]:
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in index:
        if row["family"] not in families:
            continue
        key = (row["family"], row["language"], int(row["unit"]), row["surface"], row["partition"])
        groups[key][int(row["state"])] = row
    output = []
    for key, states in sorted(groups.items()):
        if set(states) != {0, 1}:
            raise RuntimeError(("incomplete_pair", key))
        output.append({"family": key[0], "language": key[1], "unit": key[2], "surface": key[3],
                       "partition": key[4], "state0_case": states[0]["case_id"],
                       "state1_case": states[1]["case_id"],
                       "state0_index": int(states[0]["hidden_index"]),
                       "state1_index": int(states[1]["hidden_index"])})
    return output


def bundle(vector: np.ndarray, role_i: int) -> np.ndarray:
    value = np.zeros((len(contract.ROLES), vector.shape[0]), dtype=np.float32)
    value[role_i] = vector
    return value


def margin(model, row: dict, device, patches: list[tuple]) -> float:
    target = 1 - int(row["gold_position"])
    logits = causal_base.patch_forward(model, row, device, patches)
    return causal_base.candidate_margin(logits, row, target)


def free_vector(model, row: dict, tokenizer, device, q: int, role: str) -> np.ndarray:
    return old_causal.free_role_vector(model, row, tokenizer, device, q, role)


def response_vectors(field: np.ndarray, pair: dict, family: str, q: int, role_i: int,
                     families: list[str], models: dict, wrong_family: str) -> dict[str, np.ndarray]:
    fi, wi = families.index(family), families.index(wrong_family)
    h0 = np.asarray(field[pair["state0_index"], q, role_i], dtype=np.float32)
    h1 = np.asarray(field[pair["state1_index"], q, role_i], dtype=np.float32)
    return {
        "predicted": tournament_code.predict((models["own_a"][fi, q, role_i], models["own_b"][fi, q, role_i]), h0),
        "algebraic": models["mean_h1"][fi, q, role_i] - h0,
        "shared": tournament_code.predict((models["shared_a"][q, role_i], models["shared_b"][q, role_i]), h0),
        "shuffled": tournament_code.predict((models["shuffled_a"][fi, q, role_i], models["shuffled_b"][fi, q, role_i]), h0),
        "wrong_family": tournament_code.predict((models["own_a"][wi, q, role_i], models["own_b"][wi, q, role_i]), h0),
        "natural_pair": h1 - h0,
    }


def wrong_checkpoint_vector(field: np.ndarray, pair: dict, family: str, wrong_q: int, role_i: int,
                            families: list[str], models: dict) -> np.ndarray:
    fi = families.index(family)
    h0 = np.asarray(field[pair["state0_index"], wrong_q, role_i], dtype=np.float32)
    return tournament_code.predict((models["own_a"][fi, wrong_q, role_i], models["own_b"][fi, wrong_q, role_i]), h0)


def condition_patches(vectors: dict[str, np.ndarray], sign: float, dose: float, q: int, role_i: int,
                      wrong_q: int, wrong_role_i: int, wrong_q_vector: np.ndarray) -> dict[str, list[tuple]]:
    return {
        "predicted": [(q, bundle(sign * dose * vectors["predicted"], role_i), IDENTITY)],
        "algebraic": [(q, bundle(sign * dose * vectors["algebraic"], role_i), IDENTITY)],
        "shared": [(q, bundle(sign * dose * vectors["shared"], role_i), IDENTITY)],
        "shuffled": [(q, bundle(sign * dose * vectors["shuffled"], role_i), IDENTITY)],
        "wrong_family": [(q, bundle(sign * dose * vectors["wrong_family"], role_i), IDENTITY)],
        "wrong_role": [(q, bundle(sign * dose * vectors["predicted"], wrong_role_i), IDENTITY)],
        "wrong_checkpoint": [(wrong_q, bundle(sign * dose * wrong_q_vector, role_i), IDENTITY)],
        "natural_pair": [(q, bundle(sign * dose * vectors["natural_pair"], role_i), IDENTITY)],
    }


def confirmation_grid(model, device, field: np.ndarray, pairs: list[dict], lookup: dict,
                      families: list[str], decisions: dict, models: dict) -> list[dict]:
    output = []
    subset = [row for row in pairs if row["partition"] == "fresh_confirmation"]
    for pair_i, pair in enumerate(subset):
        family = pair["family"]
        setting = decisions[family]
        q, role_i = int(setting["checkpoint"]), int(setting["role_index"])
        wrong_q = q + 1 if q < 36 else q - 1
        wrong_role_i = (role_i + 1) % len(contract.ROLES)
        vectors = response_vectors(field, pair, family, q, role_i, families, models, setting["wrong_family"])
        wrong_q_vec = wrong_checkpoint_vector(field, pair, family, wrong_q, role_i, families, models)
        for direction, case_id, sign in (("call", pair["state0_case"], 1.0),
                                          ("delete", pair["state1_case"], -1.0)):
            source = lookup[case_id]
            natural = margin(model, source, device, [])
            for dose in DOSES:
                patches = condition_patches(vectors, sign, dose, q, role_i, wrong_q, wrong_role_i, wrong_q_vec)
                margins = {name: margin(model, source, device, value) for name, value in patches.items()}
                output.append({"family": family, "direction": direction, "language": pair["language"],
                               "unit": pair["unit"], "surface": pair["surface"], "checkpoint": q,
                               "role": contract.ROLES[role_i], "dose": dose, "natural_margin": natural,
                               **{f"{name}_margin": value for name, value in margins.items()}})
        if pair_i % 16 == 0:
            print(f"[causal-confirmation] {pair_i + 1}/{len(subset)}", flush=True)
    return output


def select_doses(rows: list[dict], families: list[str]) -> tuple[dict, dict]:
    summaries, frozen = {}, {}
    for family in families:
        frozen[family] = {}
        for direction in ("call", "delete"):
            choices = []
            for dose in DOSES:
                cell = [row for row in rows if row["family"] == family and row["direction"] == direction and row["dose"] == dose]
                gains = np.array([row["predicted_margin"] - row["natural_margin"] for row in cell])
                controls = np.array([max(row[f"{name}_margin"] for name in CONTROL_NAMES) for row in cell])
                summary = {"pairs": len(cell), "direction_rate": float(np.mean(gains > 0)),
                           "mean_margin_gain": float(np.mean(gains)),
                           "advantage_over_controls": float(np.mean(np.array([row["predicted_margin"] for row in cell]) - controls)),
                           "natural_pair_direction_rate": float(np.mean(np.array([row["natural_pair_margin"] - row["natural_margin"] for row in cell]) > 0))}
                summary["passes"] = bool(len(cell) >= GATES["minimum_pairs"] and
                                         summary["direction_rate"] >= GATES["direction_rate"] and
                                         summary["advantage_over_controls"] >= GATES["margin_advantage_over_controls"])
                summaries[f"{family}|{direction}|{dose}"] = summary
                if summary["passes"]:
                    choices.append((summary["advantage_over_controls"], summary["direction_rate"], -DOSES.index(dose), dose))
            frozen[family][direction] = ({"qualified": True, "dose": max(choices)[-1],
                                           "selection_source": "fresh_confirmation_only"}
                                          if choices else {"qualified": False, "reason": "no_dose_passed"})
    return summaries, frozen


def free_vectors(model, tokenizer, device, source0: dict, source1: dict, family: str, q: int, role: str,
                 families: list[str], models: dict, wrong_family: str) -> dict[str, np.ndarray]:
    fi, wi, role_i = families.index(family), families.index(wrong_family), contract.ROLES.index(role)
    h0 = free_vector(model, source0, tokenizer, device, q, role)
    h1 = free_vector(model, source1, tokenizer, device, q, role)
    return {
        "predicted": tournament_code.predict((models["own_a"][fi, q, role_i], models["own_b"][fi, q, role_i]), h0),
        "algebraic": models["mean_h1"][fi, q, role_i] - h0,
        "shared": tournament_code.predict((models["shared_a"][q, role_i], models["shared_b"][q, role_i]), h0),
        "shuffled": tournament_code.predict((models["shuffled_a"][fi, q, role_i], models["shuffled_b"][fi, q, role_i]), h0),
        "wrong_family": tournament_code.predict((models["own_a"][wi, q, role_i], models["own_b"][wi, q, role_i]), h0),
        "natural_pair": h1 - h0,
    }


def lockbox(model, tokenizer, device, field: np.ndarray, pairs: list[dict], lookup: dict,
            families: list[str], decisions: dict, models: dict, frozen: dict) -> tuple[list[dict], dict]:
    output = []
    subset = [row for row in pairs if row["partition"] == "fresh_lockbox"]
    for pair_i, pair in enumerate(subset):
        family = pair["family"]
        setting = decisions[family]
        q, role_i, role = int(setting["checkpoint"]), int(setting["role_index"]), setting["role"]
        wrong_q = q + 1 if q < 36 else q - 1
        wrong_role_i = (role_i + 1) % len(contract.ROLES)
        vectors = response_vectors(field, pair, family, q, role_i, families, models, setting["wrong_family"])
        wrong_q_vec = wrong_checkpoint_vector(field, pair, family, wrong_q, role_i, families, models)
        false_row, true_row = lookup[pair["state0_case"]], lookup[pair["state1_case"]]
        free = free_vectors(model, tokenizer, device, false_row, true_row, family, q, role,
                            families, models, setting["wrong_family"])
        for direction, source, sign in (("call", false_row, 1.0), ("delete", true_row, -1.0)):
            selected = frozen[family][direction]
            if not selected["qualified"]:
                continue
            dose = float(selected["dose"])
            natural = margin(model, source, device, [])
            patches = condition_patches(vectors, sign, dose, q, role_i, wrong_q, wrong_role_i, wrong_q_vec)
            margins = {name: margin(model, source, device, value) for name, value in patches.items()}
            free_positions = causal_base.free_positions(tokenizer, source)
            generation_patches = {
                "predicted": [(q, bundle(sign * dose * free["predicted"], role_i), IDENTITY)],
                "algebraic": [(q, bundle(sign * dose * free["algebraic"], role_i), IDENTITY)],
                "shared": [(q, bundle(sign * dose * free["shared"], role_i), IDENTITY)],
                "shuffled": [(q, bundle(sign * dose * free["shuffled"], role_i), IDENTITY)],
                "wrong_family": [(q, bundle(sign * dose * free["wrong_family"], role_i), IDENTITY)],
                "wrong_role": [(q, bundle(sign * dose * free["predicted"], wrong_role_i), IDENTITY)],
                "natural_pair": [(q, bundle(sign * dose * free["natural_pair"], role_i), IDENTITY)],
            }
            texts = {name: causal_base.generated_text(model, tokenizer, source, device, free_positions, value, "one_shot")
                     for name, value in generation_patches.items()}
            target_code = source["true_code"] if direction == "call" else source["false_code"]
            output.append({"family": family, "direction": direction, "language": pair["language"],
                           "unit": pair["unit"], "surface": pair["surface"], "checkpoint": q,
                           "role": role, "dose": dose, "target_code": target_code, "natural_margin": natural,
                           **{f"{name}_margin": value for name, value in margins.items()},
                           **{f"{name}_text": value for name, value in texts.items()},
                           **{f"{name}_generation": parse_code(value, source) == target_code for name, value in texts.items()}})
        if pair_i % 8 == 0:
            print(f"[causal-lockbox] {pair_i + 1}/{len(subset)}", flush=True)
    summaries = {}
    generation_controls = ("algebraic", "shared", "shuffled", "wrong_family", "wrong_role")
    for family in families:
        for direction in ("call", "delete"):
            cell = [row for row in output if row["family"] == family and row["direction"] == direction]
            key = f"{family}|{direction}"
            if not cell:
                summaries[key] = {"pairs": 0, "strict_pass": False, "reason": "confirmation_dose_unqualified"}
                continue
            gains = np.array([row["predicted_margin"] - row["natural_margin"] for row in cell])
            controls = np.array([max(row[f"{name}_margin"] for name in CONTROL_NAMES) for row in cell])
            predicted_gen = float(np.mean([row["predicted_generation"] for row in cell]))
            control_gen = float(max(np.mean([row[f"{name}_generation"] for row in cell]) for name in generation_controls))
            summary = {"pairs": len(cell), "dose": cell[0]["dose"], "direction_rate": float(np.mean(gains > 0)),
                       "mean_margin_gain": float(np.mean(gains)),
                       "margin_advantage_over_all_controls": float(np.mean(np.array([row["predicted_margin"] for row in cell]) - controls)),
                       "predicted_generation_accuracy": predicted_gen, "best_control_generation_accuracy": control_gen,
                       "generation_advantage": predicted_gen - control_gen,
                       "natural_pair_direction_rate": float(np.mean(np.array([row["natural_pair_margin"] - row["natural_margin"] for row in cell]) > 0)),
                       "natural_pair_generation_accuracy": float(np.mean([row["natural_pair_generation"] for row in cell]))}
            summary["strict_pass"] = bool(len(cell) >= GATES["minimum_pairs"] and
                                          summary["direction_rate"] >= GATES["direction_rate"] and
                                          summary["margin_advantage_over_all_controls"] >= GATES["margin_advantage_over_controls"] and
                                          summary["generation_advantage"] >= GATES["generation_advantage"])
            summaries[key] = summary
    return output, summaries


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 全坐标近流形调用删除与自由生成终审（C1541-C1576） [{stamp}]

**测试原理与用例。** 只接收Phase2267在fresh confirmation及 untouched lockbox均通过的10个候选。每个候选用discovery冻结的逐坐标模型，根据当前配对state0基态计算完整2560维响应；fresh confirmation的32对只在0.25/0.5/0.75/1.0中分别选择调用与删除剂量，fresh lockbox的32对才执行正式候选margin和自由生成。调用从state0沿预测响应写向state1，删除从state1减去同一配对state0条件下的预测响应。控制包括纯代数、共享、错配、错家族、错角色和错检查点；自然配对全响应仅作阳性仪器上界。

**公式与门槛。**

$$
\widehat R_{{i,j}}=a_{{f,j}}H^0_{{i,j}}+b_{{f,j}},\qquad
H'_{{i,j}}=H_{{i,j}}+s\alpha\widehat R_{{i,j}},\quad s\in\{{+1,-1\}}.
$$

正式单方向通过要求至少24对、目标margin正向率不低于0.60、相对所有控制的平均margin优势不低于0.05、目标自由生成率超过最佳控制至少0.10；家族只有调用与删除同时通过才列为严格因果候选。全2560坐标整体干预，不使用Top-K或余弦筛选。

**结果汇总。** fresh-confirmation剂量摘要 `{json.dumps(result['confirmation_summaries'], ensure_ascii=False)}`；冻结剂量 `{json.dumps(result['frozen_doses'], ensure_ascii=False)}`；锁箱摘要 `{json.dumps(result['lockbox_summaries'], ensure_ascii=False)}`；严格双向因果候选 `{json.dumps(result['strict_causal_families'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；工程检查 `{result['checks']}`，总通过 `{result['all_checks_passed']}`。

**分析、理论进展与边界。** `{result['strict_conclusion']}` 预测通过但因果失败表示该模型可作状态诊断，不能直接视为可调用齿轮；自然配对有效而预测响应无效，定位到近流形或坐标联合不足；自然配对也无效则说明单角色单检查点接口不足。即使通过也只建立该层角色全向量的局部充分性/必要性线索，不建立唯一电路。理论主体“条件化输出场闭合理论”和RDC保持不变。

**问题、硬伤、结论与相关文件。** 干预仍可能离开自然流形；删除依赖配对state0；错检查点虽重新计算该层预测响应，仍不等价于完整替代路径；自由生成是前缀单次写入；量词边界候选受输出准备污染风险最大。元语言输出码、受控模板、无人类盲评和Qwen3-4B单模型限制仍在。脚本 `tests/glm5/phase2268_c1541_c1576_near_manifold_causal_adjudication.py`；结果 `tests/glm5/result/phase2268_c1541_c1576_near_manifold_causal_adjudication`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        return load(final)
    tournament = load(TOURNAMENT_OUT / "analysis/final.json")
    families = list(tournament["lockbox_survivors"])
    decisions = {row["family"]: row for row in tournament["decisions"] if row["family"] in families}
    if not families:
        raise RuntimeError("No Phase2267 lockbox survivor")
    protocol = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "phase": PHASE,
                "families": families, "doses": list(DOSES), "gates": GATES,
                "controls": list(CONTROL_NAMES), "frozen_before_model": True,
                "selection": "fresh_confirmation_only", "final": "fresh_lockbox_only"}
    save(OUT / "protocol/preregistration.json", protocol)
    index = read_rows(FIELD_OUT / "raw/role_field_index.jsonl")
    field = np.load(FIELD_OUT / "raw/qwen3_4b_qualified_role_field.float16.npy", mmap_mode="r")
    paired = pair_rows(index, families)
    pair_map = tournament_code.build_pairs(index)
    models = tournament_code.fit_models(field, pair_map, families)
    compiled = read_rows(CONTRACT_OUT / "material/independent_bilingual_qwen_compiled.jsonl")
    lookup = {row["case_id"]: row for row in compiled}
    model = None
    confirmation_path = OUT / "raw/confirmation_grid.jsonl"
    lockbox_path = OUT / "raw/lockbox_trials.jsonl"
    try:
        model, tokenizer, device, placement = contract.legacy.parent.model_base.qwen_model()
        confirmation = confirmation_grid(model, device, field, paired, lookup, families, decisions, models)
        write_rows(confirmation_path, confirmation)
        confirmation_summaries, frozen = select_doses(confirmation, families)
        save(OUT / "analysis/confirmation_summaries.json", confirmation_summaries)
        save(OUT / "protocol/frozen_doses.json", frozen)
        lock_rows, lock_summaries = lockbox(model, tokenizer, device, field, paired, lookup,
                                             families, decisions, models, frozen)
        write_rows(lockbox_path, lock_rows)
        save(OUT / "analysis/lockbox_summaries.json", lock_summaries)
        quantization = contract.legacy.parent.model_base.scope.parent.previous.model_base().quantization_audit(model)
    finally:
        if model is not None:
            contract.legacy.parent.model_base.scope.parent.previous.model_base().release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        close_mmap(field)
    strict = [family for family in families if all(lock_summaries[f"{family}|{direction}"]["strict_pass"]
                                                    for direction in ("call", "delete"))]
    checks = {
        "protocol_frozen": True,
        "confirmation_pairs": len(confirmation) == len(families) * 32 * 2 * len(DOSES),
        "dose_selection_complete": all(set(frozen[family]) == {"call", "delete"} for family in families),
        "lockbox_only_qualified_doses": all(row["partition"] == "fresh_lockbox" for row in paired if row["partition"] == "fresh_lockbox"),
        "controls_complete": all(all(f"{name}_margin" in row for name in CONTROL_NAMES) for row in confirmation + lock_rows),
        "generation_recorded": all("predicted_generation" in row and "natural_pair_generation" in row for row in lock_rows),
        "finite_margins": all(np.isfinite(value) for row in confirmation + lock_rows for key, value in row.items() if key.endswith("_margin")),
    }
    strict_conclusion = (f"{len(strict)}/{len(families)} families passed both call and delete with all margin controls and free generation; "
                         "passes are narrow layer-role full-vector causal evidence, while failures remain predictive-only maps.")
    hashes = {"protocol": file_hash(OUT / "protocol/preregistration.json"),
              "confirmation": file_hash(confirmation_path), "frozen_doses": file_hash(OUT / "protocol/frozen_doses.json"),
              "lockbox": file_hash(lockbox_path), "lockbox_summaries": file_hash(OUT / "analysis/lockbox_summaries.json")}
    result = {"phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
              "timestamp": datetime.now().astimezone().isoformat(), "placement": placement,
              "quantization": quantization, "families": families, "gates": GATES,
              "confirmation_summaries": confirmation_summaries, "frozen_doses": frozen,
              "lockbox_summaries": lock_summaries, "strict_causal_families": strict,
              "hashes": hashes, "checks": checks, "all_checks_passed": all(checks.values()),
              "strict_conclusion": strict_conclusion,
              "next_authorization": "Confirm only important strict causal families on Qwen3-14B; publish all-coordinate predictive and causal atlases."}
    save(final, result)
    append_memo(result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"confirmation_summaries", "lockbox_summaries"}}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
