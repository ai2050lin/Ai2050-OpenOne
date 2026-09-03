#!/usr/bin/env python3
"""Causally adjudicate prospectively qualified coordinate-local operators."""
from __future__ import annotations

import gc
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
CONTRACT_OUT = RESULT / "phase2258_c1241_c1264_natural_construction_state_contract"
FIELD_OUT = RESULT / "phase2259_c1265_c1296_qwen_natural_full_token_field"
OPERATOR_OUT = RESULT / "phase2260_c1297_c1336_coordinate_local_operator_tournament"
OUT = RESULT / "phase2261_c1337_c1368_sample_conditioned_causal_adjudication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2250_c1049_c1064_state_consistent_causal as causal_base  # noqa: E402
import phase2258_c1241_c1264_natural_construction_state_contract as contract  # noqa: E402


PHASE = 2261
CAMPAIGNS = tuple(f"C{i}" for i in range(1337, 1369))
DOSES = (0.25, 0.5, 1.0)
IDENTITY = tuple(range(len(contract.ROLES)))


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


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


def pairs(index: list[dict], families: list[str]) -> list[dict]:
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in index:
        if row["family"] not in families:
            continue
        key = (row["family"], row["language"], int(row["unit"]), row["surface"], row["partition"])
        groups[key][int(row["state"])] = row
    output = []
    for key, states in sorted(groups.items()):
        if set(states) != {0, 1}:
            raise RuntimeError(("incomplete_pair", key, sorted(states)))
        output.append({"family": key[0], "language": key[1], "unit": key[2],
                       "surface": key[3], "partition": key[4],
                       "state0_case": states[0]["case_id"], "state1_case": states[1]["case_id"],
                       "state0_index": int(states[0]["hidden_index"]),
                       "state1_index": int(states[1]["hidden_index"])})
    return output


def compiled_lookup() -> dict[str, dict]:
    rows = read_rows(CONTRACT_OUT / "material/natural_construction_qwen_compiled.jsonl")
    return {row["case_id"]: row for row in rows}


def fit_diagonal(field: np.ndarray, pair_rows: list[dict], family: str, q: int, role_i: int) -> tuple[np.ndarray, np.ndarray]:
    subset = [row for row in pair_rows if row["family"] == family and row["partition"] == "discovery"]
    h0 = np.asarray(field[[row["state0_index"] for row in subset], q, role_i], np.float32)
    h1 = np.asarray(field[[row["state1_index"] for row in subset], q, role_i], np.float32)
    y = h1 - h0
    xm, ym = h0.mean(axis=0), y.mean(axis=0)
    xc, yc = h0 - xm, y - ym
    b = np.sum(xc * yc, axis=0) / (np.sum(xc * xc, axis=0) + 0.05)
    return (ym - b * xm).astype(np.float32), b.astype(np.float32)


def prediction(a: np.ndarray, b: np.ndarray, state: np.ndarray) -> np.ndarray:
    return (a + b * np.asarray(state, np.float32)).astype(np.float32)


def bundle(vector: np.ndarray, role_i: int) -> np.ndarray:
    value = np.zeros((len(contract.ROLES), vector.shape[0]), dtype=np.float32)
    value[role_i] = vector
    return value


def sign_control(vector: np.ndarray) -> np.ndarray:
    signs = np.where(np.arange(vector.shape[0]) % 2 == 0, 1.0, -1.0).astype(np.float32)
    return np.asarray(vector, np.float32) * signs


def margin(model, row: dict, device, patches: list[tuple]) -> float:
    target = 1 - int(row["gold_position"])
    logits = causal_base.patch_forward(model, row, device, patches)
    return causal_base.candidate_margin(logits, row, target)


def free_role_vector(model, row: dict, tokenizer, device, q: int, role: str) -> np.ndarray:
    positions = causal_base.free_positions(tokenizer, row)
    captured = []

    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handle = model.model.layers[q - 1].register_forward_hook(hook)
    ids = torch.tensor([row["free_prompt_ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    pos = mask.long().cumsum(-1) - 1
    try:
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    return captured[0][0, positions[role][-1]].float().cpu().numpy().astype(np.float32)


def summarize_grid(rows: list[dict]) -> tuple[dict, dict]:
    summaries = {}
    frozen = {}
    families = sorted({row["family"] for row in rows})
    for family in families:
        frozen[family] = {}
        for direction in ("call", "delete"):
            choices = []
            for dose in DOSES:
                cell = [row for row in rows if row["family"] == family
                        and row["direction"] == direction and row["dose"] == dose]
                gains = np.array([row["predicted_margin"] - row["natural_margin"] for row in cell])
                controls = np.array([max(row["wrong_family_margin"], row["sign_control_margin"])
                                     for row in cell])
                summary = {
                    "pairs": len(cell), "direction_rate": float(np.mean(gains > 0)),
                    "mean_margin_gain": float(np.mean(gains)),
                    "advantage_over_controls": float(np.mean(
                        np.array([row["predicted_margin"] for row in cell]) - controls)),
                    "natural_pair_direction_rate": float(np.mean(
                        np.array([row["natural_pair_margin"] - row["natural_margin"] for row in cell]) > 0)),
                }
                summaries[f"{family}|{direction}|{dose}"] = summary
                gates = contract.CAUSAL_GATES
                if (len(cell) >= gates["minimum_pairs"]
                        and summary["direction_rate"] >= gates["direction_rate"]
                        and summary["advantage_over_controls"] >= gates["margin_advantage_over_controls"]):
                    choices.append((summary["advantage_over_controls"], summary["direction_rate"],
                                    -DOSES.index(dose), dose))
            frozen[family][direction] = ({"qualified": True, "dose": max(choices)[-1],
                                           "selection_source": "fresh_confirmation_only"}
                                          if choices else {"qualified": False,
                                                           "reason": "no_confirmation_dose_passed"})
    return summaries, frozen


def confirmation_grid(model, device, field: np.ndarray, pair_rows: list[dict], lookup: dict,
                      selected: dict, coefficients: dict) -> list[dict]:
    output = []
    subset = [row for row in pair_rows if row["partition"] == "fresh_confirmation"]
    families = sorted(selected)
    for pair_i, pair in enumerate(subset):
        family = pair["family"]
        setting = selected[family]
        q, role = int(setting["checkpoint"]), setting["role"]
        role_i = contract.ROLES.index(role)
        wrong_family = families[(families.index(family) + 1) % len(families)]
        h0 = np.asarray(field[pair["state0_index"], q, role_i], np.float32)
        h1 = np.asarray(field[pair["state1_index"], q, role_i], np.float32)
        actual = h1 - h0
        a, b = coefficients[(family, q, role_i)]
        wa, wb = coefficients[(wrong_family, q, role_i)]
        predicted = prediction(a, b, h0)
        wrong = prediction(wa, wb, h0)
        signed = sign_control(predicted)
        for direction, case_id, sign in (("call", pair["state0_case"], 1.0),
                                          ("delete", pair["state1_case"], -1.0)):
            source = lookup[case_id]
            natural = margin(model, source, device, [])
            for dose in DOSES:
                output.append({
                    "family": family, "wrong_family": wrong_family, "direction": direction,
                    "language": pair["language"], "unit": pair["unit"], "surface": pair["surface"],
                    "checkpoint": q, "role": role, "dose": dose, "natural_margin": natural,
                    "predicted_margin": margin(model, source, device,
                                               [(q, bundle(sign * dose * predicted, role_i), IDENTITY)]),
                    "wrong_family_margin": margin(model, source, device,
                                                  [(q, bundle(sign * dose * wrong, role_i), IDENTITY)]),
                    "sign_control_margin": margin(model, source, device,
                                                  [(q, bundle(sign * dose * signed, role_i), IDENTITY)]),
                    "natural_pair_margin": margin(model, source, device,
                                                  [(q, bundle(sign * dose * actual, role_i), IDENTITY)]),
                })
        if pair_i % 16 == 0:
            print(f"[causal-confirmation] {pair_i + 1}/{len(subset)}", flush=True)
    return output


def lockbox(model, tokenizer, device, field: np.ndarray, pair_rows: list[dict], lookup: dict,
            selected: dict, coefficients: dict, frozen: dict) -> tuple[list[dict], dict]:
    output = []
    subset = [row for row in pair_rows if row["partition"] == "fresh_lockbox"]
    families = sorted(selected)
    for pair_i, pair in enumerate(subset):
        family = pair["family"]
        setting = selected[family]
        q, role = int(setting["checkpoint"]), setting["role"]
        role_i = contract.ROLES.index(role)
        wrong_role_i = (role_i + 1) % len(contract.ROLES)
        wrong_q = q + 1 if q < 36 else q - 1
        wrong_family = families[(families.index(family) + 1) % len(families)]
        h0 = np.asarray(field[pair["state0_index"], q, role_i], np.float32)
        h1 = np.asarray(field[pair["state1_index"], q, role_i], np.float32)
        actual = h1 - h0
        a, b = coefficients[(family, q, role_i)]
        wa, wb = coefficients[(wrong_family, q, role_i)]
        predicted = prediction(a, b, h0)
        wrong = prediction(wa, wb, h0)
        signed = sign_control(predicted)
        false_row, true_row = lookup[pair["state0_case"]], lookup[pair["state1_case"]]
        free_h0 = free_role_vector(model, false_row, tokenizer, device, q, role)
        free_h1 = free_role_vector(model, true_row, tokenizer, device, q, role)
        free_actual = free_h1 - free_h0
        free_predicted = prediction(a, b, free_h0)
        free_wrong = prediction(wa, wb, free_h0)
        free_signed = sign_control(free_predicted)
        for direction, source, sign in (("call", false_row, 1.0), ("delete", true_row, -1.0)):
            dose_setting = frozen[family][direction]
            if not dose_setting["qualified"]:
                continue
            dose = float(dose_setting["dose"])
            natural = margin(model, source, device, [])
            correct_bundle = bundle(sign * dose * predicted, role_i)
            wrong_bundle = bundle(sign * dose * wrong, role_i)
            sign_bundle = bundle(sign * dose * signed, role_i)
            natural_bundle = bundle(sign * dose * actual, role_i)
            wrong_role_bundle = bundle(sign * dose * predicted, wrong_role_i)
            free_positions = causal_base.free_positions(tokenizer, source)
            free_correct_bundle = bundle(sign * dose * free_predicted, role_i)
            free_wrong_bundle = bundle(sign * dose * free_wrong, role_i)
            free_sign_bundle = bundle(sign * dose * free_signed, role_i)
            free_natural_bundle = bundle(sign * dose * free_actual, role_i)
            target_code = source["true_code"] if direction == "call" else source["false_code"]
            predicted_text = causal_base.generated_text(
                model, tokenizer, source, device, free_positions, [(q, free_correct_bundle, IDENTITY)], "one_shot")
            wrong_text = causal_base.generated_text(
                model, tokenizer, source, device, free_positions, [(q, free_wrong_bundle, IDENTITY)], "one_shot")
            sign_text = causal_base.generated_text(
                model, tokenizer, source, device, free_positions, [(q, free_sign_bundle, IDENTITY)], "one_shot")
            natural_text = causal_base.generated_text(
                model, tokenizer, source, device, free_positions, [(q, free_natural_bundle, IDENTITY)], "one_shot")
            output.append({
                "family": family, "wrong_family": wrong_family, "direction": direction,
                "language": pair["language"], "unit": pair["unit"], "surface": pair["surface"],
                "checkpoint": q, "role": role, "dose": dose, "target_code": target_code,
                "natural_margin": natural,
                "predicted_margin": margin(model, source, device, [(q, correct_bundle, IDENTITY)]),
                "wrong_family_margin": margin(model, source, device, [(q, wrong_bundle, IDENTITY)]),
                "sign_control_margin": margin(model, source, device, [(q, sign_bundle, IDENTITY)]),
                "wrong_role_margin": margin(model, source, device, [(q, wrong_role_bundle, IDENTITY)]),
                "wrong_checkpoint_margin": margin(model, source, device, [(wrong_q, correct_bundle, IDENTITY)]),
                "natural_pair_margin": margin(model, source, device, [(q, natural_bundle, IDENTITY)]),
                "predicted_text": predicted_text, "wrong_text": wrong_text,
                "sign_text": sign_text, "natural_pair_text": natural_text,
                "predicted_generation": parse_code(predicted_text, source) == target_code,
                "wrong_generation": parse_code(wrong_text, source) == target_code,
                "sign_generation": parse_code(sign_text, source) == target_code,
                "natural_pair_generation": parse_code(natural_text, source) == target_code,
            })
        if pair_i % 8 == 0:
            print(f"[causal-lockbox] {pair_i + 1}/{len(subset)}", flush=True)

    summaries = {}
    for family in families:
        for direction in ("call", "delete"):
            cell = [row for row in output if row["family"] == family and row["direction"] == direction]
            key = f"{family}|{direction}"
            if not cell:
                summaries[key] = {"pairs": 0, "strict_pass": False, "reason": "confirmation_setting_unqualified"}
                continue
            gains = np.array([row["predicted_margin"] - row["natural_margin"] for row in cell])
            controls = np.array([max(row["wrong_family_margin"], row["sign_control_margin"],
                                     row["wrong_role_margin"], row["wrong_checkpoint_margin"]) for row in cell])
            predicted_gen = float(np.mean([row["predicted_generation"] for row in cell]))
            control_gen = float(np.mean([max(row["wrong_generation"], row["sign_generation"]) for row in cell]))
            summary = {
                "pairs": len(cell), "direction_rate": float(np.mean(gains > 0)),
                "mean_margin_gain": float(np.mean(gains)),
                "margin_advantage_over_all_controls": float(np.mean(
                    np.array([row["predicted_margin"] for row in cell]) - controls)),
                "predicted_generation_accuracy": predicted_gen,
                "control_generation_accuracy": control_gen,
                "generation_advantage": predicted_gen - control_gen,
                "natural_pair_direction_rate": float(np.mean(
                    np.array([row["natural_pair_margin"] - row["natural_margin"] for row in cell]) > 0)),
                "natural_pair_generation_accuracy": float(np.mean([row["natural_pair_generation"] for row in cell])),
            }
            gates = contract.CAUSAL_GATES
            summary["strict_pass"] = bool(
                len(cell) >= gates["minimum_pairs"]
                and summary["direction_rate"] >= gates["direction_rate"]
                and summary["margin_advantage_over_all_controls"] >= gates["margin_advantage_over_controls"]
                and summary["generation_advantage"] >= gates["generation_advantage"])
            summaries[key] = summary
    return output, summaries


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 样本条件同坐标调用、删除与生成因果终审（C1337-C1368） [{stamp}]

**测试原理与时序。** 仅接收Phase2260在fresh lockbox前已冻结、并于fresh lockbox保持预测增益的构式。每个族使用discovery拟合的同坐标仿射式，根据当前配对基态逐样本计算完整2560维响应；fresh confirmation只选择0.25/0.5/1.0剂量，fresh lockbox才检验调用、删除、候选边际和自由生成。错族使用另一族在同一检查点-角色拟合的系数，另设等范数交替符号、错角色、错检查点；自然配对状态差只作仪器阳性控制，不参与算子发现。

**公式。** 对冻结检查点和角色：

$$
\widehat R_{{i,j}}=a_j+b_jH^{{(0)}}_{{i,j}},\qquad
H'_{{i,j}}=H_{{i,j}}+s\alpha\widehat R_{{i,j}},\quad s\in\{{+1,-1\}}.
$$

删除方向仍使用该配对的state0基态预测响应后从state1减去，因此是“配对样本条件”而非只读删除源状态的自主逆算子。自由生成在自由回答前缀上重新读取同一检查点-角色基态后计算响应，避免直接搬用候选提示向量。

**结果汇总。** confirmation摘要 `{json.dumps(result['confirmation_summaries'], ensure_ascii=False)}`；冻结剂量 `{json.dumps(result['frozen'], ensure_ascii=False)}`；fresh lockbox `{json.dumps(result['lockbox_summaries'], ensure_ascii=False)}`；同时通过双向候选、全部控制和自由生成的严格因果族为 `{json.dumps(result['strict_causal_families'], ensure_ascii=False)}`。

**分析、理论进展与边界。** 预测通过而因果失败，表示同坐标仿射是状态诊断规律，不是可调用齿轮；自然配对阳性控制若有效而预测注入无效，定位到模型近似或坐标联合不足；自然配对也无效则说明单角色单检查点接口本身不足。任何通过也只建立局部充分性/必要性证据，不建立唯一电路。理论主体与RDC不改名。

**问题、硬伤、结论与相关文件。** 干预离开自然状态流形，删除依赖配对state0，生成只在前缀单次写入，错族控制只有当前九族，答案边界候选可能主要是输出准备而非上游构式计算。工程检查 `{result['all_checks_passed']}`。脚本 `tests/glm5/phase2261_c1337_c1368_sample_conditioned_causal_adjudication.py`；结果 `tests/glm5/result/phase2261_c1337_c1368_sample_conditioned_causal_adjudication`。下一步不因因果阴性删除观察结果，而是生成全坐标可视化，并只对重要前瞻结果启动独立大模型确认。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        return load(final)
    tournament = load(OPERATOR_OUT / "analysis/final.json")
    families = tournament["analysis"]["operator_qualified_families"]
    selected = {family: tournament["analysis"]["selected"][family] for family in families}
    field = np.load(FIELD_OUT / "raw/qwen3_4b_qualified_role_field.float16.npy", mmap_mode="r")
    index = read_rows(FIELD_OUT / "raw/role_field_index.jsonl")
    pair_rows = pairs(index, families)
    lookup = compiled_lookup()
    coefficients = {}
    for family in families:
        for target_family, setting in selected.items():
            q, role_i = int(setting["checkpoint"]), contract.ROLES.index(setting["role"])
            coefficients[(family, q, role_i)] = fit_diagonal(field, pair_rows, family, q, role_i)
    model = None
    try:
        model, tokenizer, device, placement = contract.parent.model_base.qwen_model()
        grid = confirmation_grid(model, device, field, pair_rows, lookup, selected, coefficients)
        confirmation_summaries, frozen = summarize_grid(grid)
        lockbox_rows, lockbox_summaries = lockbox(
            model, tokenizer, device, field, pair_rows, lookup, selected, coefficients, frozen)
    finally:
        if model is not None:
            contract.parent.model_base.scope.parent.previous.model_base().release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        close_mmap(field)
    write_rows(OUT / "causal/confirmation_grid.jsonl", grid)
    write_rows(OUT / "causal/fresh_lockbox_rows.jsonl", lockbox_rows)
    strict = [family for family in families
              if all(lockbox_summaries[f"{family}|{direction}"]["strict_pass"]
                     for direction in ("call", "delete"))]
    checks = {
        "only_operator_qualified_families": set(families) == set(selected),
        "confirmation_complete": len(grid) == len(families) * 32 * 2 * len(DOSES),
        "all_controls_present": all(set(row) >= {"wrong_family_margin", "sign_control_margin",
                                                 "wrong_role_margin", "wrong_checkpoint_margin",
                                                 "natural_pair_margin"} for row in lockbox_rows),
        "generation_recorded": all("predicted_generation" in row for row in lockbox_rows),
        "summary_complete": set(lockbox_summaries) == {
            f"{family}|{direction}" for family in families for direction in ("call", "delete")},
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "placement": placement,
        "families": families, "selected": selected,
        "confirmation_summaries": confirmation_summaries, "frozen": frozen,
        "lockbox_summaries": lockbox_summaries, "strict_causal_families": strict,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Only strict_causal_families pass bidirectional margin, wrong-family/sign/role/checkpoint controls, and free-generation advantage; prediction alone is not causality.",
        "next_authorization": "Export full-coordinate visual atlases, verify the client, clean only undisplayed raw fields, and run cross-scale confirmation for an important predictive result.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
