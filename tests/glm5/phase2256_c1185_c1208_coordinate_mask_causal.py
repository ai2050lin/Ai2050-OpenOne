#!/usr/bin/env python3
"""Prospective causal adjudication of frozen coordinate-passport masks."""
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
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CONTRACT_OUT = RESULT / "phase2253_c1097_c1120_construction_ecology_contract"
FIELD_OUT = RESULT / "phase2254_c1121_c1152_qwen_construction_full_field"
PASSPORT_OUT = RESULT / "phase2255_c1153_c1184_coordinate_passport_ecology"
OUT = RESULT / "phase2256_c1185_c1208_coordinate_mask_causal"
sys.path.insert(0, str(TESTS))

import phase2250_c1049_c1064_state_consistent_causal as causal_base  # noqa: E402
import phase2253_c1097_c1120_construction_ecology_contract as contract  # noqa: E402
import phase2255_c1153_c1184_coordinate_passport_ecology as passports  # noqa: E402


PHASE = 2256
CAMPAIGNS = tuple(f"C{i}" for i in range(1185, 1209))
DOSES = (0.25, 0.5, 1.0)
IDENTITY = tuple(range(len(contract.ROLES)))
SHIFTED = tuple((i + 1) % len(contract.ROLES) for i in range(len(contract.ROLES)))


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def compiled_lookup() -> dict[str, dict]:
    rows = []
    for name in ("parent_broad", "fresh_broad"):
        rows.extend(read_rows(CONTRACT_OUT / f"material/{name}_qwen_compiled.jsonl"))
    return {row["case_id"]: row for row in rows}


def pairs(index: list[dict], families: list[str]) -> list[dict]:
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in index:
        if row["panel"] != "construction_broad" or row["family"] not in families:
            continue
        key = (row["family"], row["language"], int(row["unit"]), row["surface"], row["partition"])
        groups[key][int(row["state"])] = row
    output = []
    for key, states in sorted(groups.items()):
        if set(states) != {0, 1}:
            continue
        output.append({"family": key[0], "language": key[1], "unit": key[2],
                       "surface": key[3], "partition": key[4],
                       "false_case_id": states[0]["case_id"], "true_case_id": states[1]["case_id"]})
    return output


def anchor_plan(passport_result: dict) -> dict:
    output = {}
    for family in passport_result["analysis"]["causal_qualified_families"]:
        formation = passport_result["analysis"]["families"][family]["prelockbox_formation"]
        anchors = []
        for role in ("query", "boundary"):
            row = next(x for x in formation if x["role"] == role)
            if row["formation"] is not None:
                anchors.append({"role": role, "checkpoint": int(row["formation"])})
        output[family] = anchors
    return output


def prototypes(field: np.ndarray, index: list[dict], families: list[str]) -> dict[str, np.ndarray]:
    responses = passports.paired_unit_responses(field, index)
    return {family: np.mean([responses[family][u] for u in range(contract.DISCOVERY_UNITS)],
                            axis=0, dtype=np.float32) for family in families}


def response_bundle(vector: np.ndarray, role: str) -> np.ndarray:
    bundle = np.zeros((len(contract.ROLES), vector.shape[0]), dtype=np.float32)
    bundle[contract.ROLES.index(role)] = vector
    return bundle


def alternating_sign(vector: np.ndarray) -> np.ndarray:
    signs = np.where(np.arange(vector.shape[0]) % 2 == 0, 1.0, -1.0).astype(np.float32)
    return np.asarray(vector, np.float32) * signs


def logits(model, row: dict, device, patches: list[tuple]) -> np.ndarray:
    return causal_base.patch_forward(model, row, device, patches)


def margin(model, row: dict, device, patches: list[tuple]) -> float:
    target = 1 - int(row["gold_position"])
    return causal_base.candidate_margin(logits(model, row, device, patches), row, target)


def vector_for(family: str, anchor: dict, protos: dict, masks: np.ndarray,
               family_order: list[str]) -> np.ndarray:
    fi = family_order.index(family)
    q = anchor["checkpoint"]
    role_i = contract.ROLES.index(anchor["role"])
    return np.where(masks[fi, q, role_i].astype(bool), protos[family][q, role_i], 0.0).astype(np.float32)


def confirmation_grid(model, device, lookup: dict, pair_rows: list[dict], anchors: dict,
                      protos: dict, masks: np.ndarray, family_order: list[str]) -> tuple[list[dict], dict]:
    rows = []
    subset = [row for row in pair_rows if row["partition"] == "fresh_confirmation"]
    for pair_i, pair in enumerate(subset):
        family = pair["family"]
        wrong_family = family_order[(family_order.index(family) + 1) % len(family_order)]
        for direction, case_id, sign in (("call", pair["false_case_id"], 1.0),
                                         ("delete", pair["true_case_id"], -1.0)):
            source = lookup[case_id]
            natural = margin(model, source, device, [])
            for anchor_i, anchor in enumerate(anchors[family]):
                q = anchor["checkpoint"]
                correct_v = vector_for(family, anchor, protos, masks, family_order)
                wrong_anchor = {"checkpoint": q, "role": anchor["role"]}
                wrong_v = vector_for(wrong_family, wrong_anchor, protos, masks, family_order)
                sign_v = alternating_sign(correct_v)
                for dose in DOSES:
                    correct = response_bundle(sign * dose * correct_v, anchor["role"])
                    wrong = response_bundle(sign * dose * wrong_v, anchor["role"])
                    permuted = response_bundle(sign * dose * sign_v, anchor["role"])
                    rows.append({
                        "family": family, "direction": direction, "pair": pair,
                        "anchor_index": anchor_i, "checkpoint": q, "role": anchor["role"],
                        "dose": dose, "natural_margin": natural,
                        "correct_margin": margin(model, source, device, [(q, correct, IDENTITY)]),
                        "wrong_family_margin": margin(model, source, device, [(q, wrong, IDENTITY)]),
                        "sign_control_margin": margin(model, source, device, [(q, permuted, IDENTITY)]),
                    })
        if pair_i % 8 == 0:
            print(f"[causal-confirmation] {pair_i}/{len(subset)}", flush=True)
    summaries = {}
    frozen = {}
    for family in family_order:
        frozen[family] = {}
        for direction in ("call", "delete"):
            choices = []
            for anchor_i, _anchor in enumerate(anchors[family]):
                for dose in DOSES:
                    cell = [x for x in rows if x["family"] == family and x["direction"] == direction
                            and x["anchor_index"] == anchor_i and x["dose"] == dose]
                    gains = np.array([x["correct_margin"] - x["natural_margin"] for x in cell])
                    controls = np.array([max(x["wrong_family_margin"], x["sign_control_margin"]) for x in cell])
                    advantage = float(np.mean(np.array([x["correct_margin"] for x in cell]) - controls))
                    summary = {"pairs": len(cell), "direction_rate": float(np.mean(gains > 0)),
                               "mean_margin_gain": float(np.mean(gains)),
                               "advantage_over_controls": advantage}
                    key = f"{family}|{direction}|a{anchor_i}|d{dose}"
                    summaries[key] = summary
                    gates = contract.CAUSAL_GATES
                    if (len(cell) >= gates["minimum_pairs"]
                            and summary["direction_rate"] >= gates["candidate_direction_rate"]
                            and advantage >= gates["candidate_margin_advantage"]):
                        choices.append((advantage, summary["direction_rate"], -anchor_i,
                                        -DOSES.index(dose), anchor_i, dose))
            if choices:
                best = max(choices)
                anchor_i, dose = best[-2], best[-1]
                frozen[family][direction] = {"qualified": True, "anchor_index": anchor_i,
                                              "anchor": anchors[family][anchor_i], "dose": dose,
                                              "selection_source": "fresh_confirmation_only"}
            else:
                frozen[family][direction] = {"qualified": False,
                                              "reason": "no_confirmation_cell_passed"}
    return rows, {"summaries": summaries, "frozen": frozen}


def confirmation_controls(model, device, lookup: dict, pair_rows: list[dict], frozen: dict,
                          protos: dict, masks: np.ndarray, family_order: list[str]) -> dict:
    output = {}
    subset = [row for row in pair_rows if row["partition"] == "fresh_confirmation"]
    for family in family_order:
        for direction in ("call", "delete"):
            setting = frozen[family][direction]
            key = f"{family}|{direction}"
            if not setting["qualified"]:
                output[key] = {"pairs": 0, "passed": False, "reason": setting["reason"]}
                continue
            anchor, dose = setting["anchor"], float(setting["dose"])
            q, role = anchor["checkpoint"], anchor["role"]
            vector = vector_for(family, anchor, protos, masks, family_order)
            rows = []
            for pair in subset:
                if pair["family"] != family:
                    continue
                case_id, sign = ((pair["false_case_id"], 1.0) if direction == "call" else
                                 (pair["true_case_id"], -1.0))
                source = lookup[case_id]
                bundle = response_bundle(sign * dose * vector, role)
                correct = margin(model, source, device, [(q, bundle, IDENTITY)])
                wrong_role = margin(model, source, device, [(q, bundle, SHIFTED)])
                wrong_checkpoint = margin(model, source, device, [(min(q + 1, 36), bundle, IDENTITY)])
                rows.append({"correct": correct, "wrong_role": wrong_role,
                             "wrong_checkpoint": wrong_checkpoint})
            advantage = float(np.mean([x["correct"] - max(x["wrong_role"], x["wrong_checkpoint"])
                                       for x in rows]))
            output[key] = {"pairs": len(rows), "advantage": advantage,
                           "passed": advantage >= contract.CAUSAL_GATES["candidate_margin_advantage"]}
    return output


def lockbox(model, tokenizer, device, lookup: dict, pair_rows: list[dict], frozen: dict,
            protos: dict, masks: np.ndarray, family_order: list[str]) -> tuple[list[dict], dict]:
    rows = []
    subset = [row for row in pair_rows if row["partition"] == "fresh_lockbox"]
    for pair_i, pair in enumerate(subset):
        family = pair["family"]
        wrong_family = family_order[(family_order.index(family) + 1) % len(family_order)]
        for direction, case_id, sign in (("call", pair["false_case_id"], 1.0),
                                         ("delete", pair["true_case_id"], -1.0)):
            setting = frozen[family][direction]
            if not setting["qualified"]:
                continue
            source = lookup[case_id]
            anchor, dose = setting["anchor"], float(setting["dose"])
            q, role = anchor["checkpoint"], anchor["role"]
            vector = vector_for(family, anchor, protos, masks, family_order)
            wrong_vector = vector_for(wrong_family, anchor, protos, masks, family_order)
            correct_bundle = response_bundle(sign * dose * vector, role)
            wrong_bundle = response_bundle(sign * dose * wrong_vector, role)
            sign_bundle = response_bundle(sign * dose * alternating_sign(vector), role)
            natural = margin(model, source, device, [])
            correct = margin(model, source, device, [(q, correct_bundle, IDENTITY)])
            wrong = margin(model, source, device, [(q, wrong_bundle, IDENTITY)])
            wrong_role = margin(model, source, device, [(q, correct_bundle, SHIFTED)])
            wrong_q = margin(model, source, device, [(min(q + 1, 36), correct_bundle, IDENTITY)])
            sign_control = margin(model, source, device, [(q, sign_bundle, IDENTITY)])
            free_positions = causal_base.free_positions(tokenizer, source)
            correct_text = causal_base.generated_text(
                model, tokenizer, source, device, free_positions,
                [(q, correct_bundle, IDENTITY)], "one_shot")
            wrong_text = causal_base.generated_text(
                model, tokenizer, source, device, free_positions,
                [(q, wrong_bundle, IDENTITY)], "one_shot")
            target_code = source["true_code"] if direction == "call" else source["false_code"]
            rows.append({
                "family": family, "direction": direction, "pair": pair,
                "checkpoint": q, "role": role, "dose": dose,
                "natural_margin": natural, "correct_margin": correct,
                "wrong_family_margin": wrong, "wrong_role_margin": wrong_role,
                "wrong_checkpoint_margin": wrong_q, "sign_control_margin": sign_control,
                "correct_text": correct_text, "wrong_text": wrong_text,
                "target_code": target_code,
                "correct_generation": parse_code(correct_text, source) == target_code,
                "wrong_generation": parse_code(wrong_text, source) == target_code,
            })
        if pair_i % 8 == 0:
            print(f"[causal-lockbox] {pair_i}/{len(subset)}", flush=True)
    summaries = {}
    for family in family_order:
        for direction in ("call", "delete"):
            cell = [x for x in rows if x["family"] == family and x["direction"] == direction]
            key = f"{family}|{direction}"
            if not cell:
                summaries[key] = {"pairs": 0, "strict_pass": False, "reason": "setting_unqualified"}
                continue
            gains = np.array([x["correct_margin"] - x["natural_margin"] for x in cell])
            controls = np.array([max(x["wrong_family_margin"], x["wrong_role_margin"],
                                     x["wrong_checkpoint_margin"], x["sign_control_margin"]) for x in cell])
            correct_gen = float(np.mean([x["correct_generation"] for x in cell]))
            wrong_gen = float(np.mean([x["wrong_generation"] for x in cell]))
            summary = {
                "pairs": len(cell), "direction_rate": float(np.mean(gains > 0)),
                "mean_margin_gain": float(np.mean(gains)),
                "margin_advantage_over_all_controls": float(np.mean(
                    np.array([x["correct_margin"] for x in cell]) - controls)),
                "correct_generation_accuracy": correct_gen,
                "wrong_generation_accuracy": wrong_gen,
                "generation_advantage": correct_gen - wrong_gen,
            }
            gates = contract.CAUSAL_GATES
            summary["strict_pass"] = bool(
                len(cell) >= gates["minimum_pairs"]
                and summary["direction_rate"] >= gates["candidate_direction_rate"]
                and summary["margin_advantage_over_all_controls"] >= gates["candidate_margin_advantage"]
                and summary["generation_advantage"] >= gates["generation_accuracy_advantage"])
            summaries[key] = summary
    return rows, summaries


def rescue(model, device, lookup: dict, pair_rows: list[dict], frozen: dict,
           protos: dict, masks: np.ndarray, family_order: list[str]) -> dict:
    output = {}
    subset = [row for row in pair_rows if row["partition"] == "fresh_lockbox"]
    for family in family_order:
        setting = frozen[family]["delete"]
        if not setting["qualified"]:
            output[family] = {"pairs": 0, "strict_pass": False, "reason": "delete_setting_unqualified"}
            continue
        wrong_family = family_order[(family_order.index(family) + 1) % len(family_order)]
        anchor, dose = setting["anchor"], float(setting["dose"])
        q, role = anchor["checkpoint"], anchor["role"]
        next_anchor = {"checkpoint": min(q + 1, 36), "role": role}
        delete_v = vector_for(family, anchor, protos, masks, family_order)
        rescue_v = vector_for(family, next_anchor, protos, masks, family_order)
        wrong_v = vector_for(wrong_family, next_anchor, protos, masks, family_order)
        rows = []
        for pair in subset:
            if pair["family"] != family:
                continue
            source = lookup[pair["true_case_id"]]
            delete_bundle = response_bundle(-dose * delete_v, role)
            rescue_bundle = response_bundle(dose * rescue_v, role)
            wrong_bundle = response_bundle(dose * wrong_v, role)
            deleted = margin(model, source, device, [(q, delete_bundle, IDENTITY)])
            rescued = margin(model, source, device, [(q, delete_bundle, IDENTITY),
                                                     (next_anchor["checkpoint"], rescue_bundle, IDENTITY)])
            wrong = margin(model, source, device, [(q, delete_bundle, IDENTITY),
                                                   (next_anchor["checkpoint"], wrong_bundle, IDENTITY)])
            rows.append({"deleted": deleted, "rescued": rescued, "wrong_rescue": wrong})
        advantage = float(np.mean([x["rescued"] - x["wrong_rescue"] for x in rows]))
        recovery = float(np.mean([x["rescued"] - x["deleted"] for x in rows]))
        output[family] = {"pairs": len(rows), "correct_over_wrong_advantage": advantage,
                          "recovery_over_deleted": recovery,
                          "strict_pass": advantage >= contract.CAUSAL_GATES["correct_rescue_advantage"]}
    return output


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 冻结逐坐标掩码的调用、删除、生成与救援裁决（C1185-C1208） [{stamp}]

**测试原理与时序纪律。** 仅接收Phase2255时序重裁后的主动/被动角色和属性覆盖。坐标掩码由8个discovery单元留一冻结；查询/答案边界锚点由父holdout与fresh confirmation确定，fresh lockbox不参与锚点或剂量选择。fresh confirmation比较0.25/0.5/1.0剂量以及错族和等范数交替符号控制，再检查错角色/错检查点；冻结胜者原封不动进入fresh lockbox，分别测试false到true调用、true到false删除、候选边际、自由生成和删除后下一检查点救援。

**公式。** 对冻结坐标掩码 $M$：

$$
H'_{{q,r,j}}=H_{{q,r,j}}+s\alpha M_{{f,q,r,j}}\bar R_{{f,q,r,j}},\qquad s\in\{{+1,-1\}}.
$$

两事件救援为：

$$
H'=H-\alpha M_q\bar R_q+\alpha M_{{q+1}}\bar R_{{q+1}}.
$$

**结果汇总。** 预冻结锚点 `{json.dumps(result['anchors'], ensure_ascii=False)}`；confirmation冻结设置 `{json.dumps(result['confirmation']['frozen'], ensure_ascii=False)}`；错角色/错层控制 `{json.dumps(result['confirmation_controls'], ensure_ascii=False)}`；fresh lockbox候选与自由生成 `{json.dumps(result['lockbox_summary'], ensure_ascii=False)}`；救援 `{json.dumps(result['rescue'], ensure_ascii=False)}`。最终同时满足调用、删除、全控制、生成和救援的严格族为 `{json.dumps(result['strict_causal_families'], ensure_ascii=False)}`。

**分析与理论进展。** 正结果只授权“该冻结坐标联合在该检查点/角色/剂量下具有局部、可控的行为效力”；它不等于唯一电路、最小齿轮或坐标语义字典。候选边际通过而自由生成失败表示只移动局部答案竞争；调用与删除不对称表示充分性和必要性不同；救援失败表示响应不能作为可组合下游状态。理论主体与RDC不改名。

**问题、硬伤与结论。** 干预是激活级强操作；只有两个族可提供错族控制；生成使用无cache的一次前缀写入与正常cache路径并不完全同构；锚点只覆盖查询和边界；同一fresh lockbox同时承担候选、生成和救援终审。工程检查 `{result['all_checks_passed']}`。任何失败只关闭本轮因果主张，不中止跨模型行为与全坐标图谱。

**相关文件。** 脚本 `tests/glm5/phase2256_c1185_c1208_coordinate_mask_causal.py`；结果 `tests/glm5/result/phase2256_c1185_c1208_coordinate_mask_causal`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    passport_result = load(PASSPORT_OUT / "analysis/final.json")
    families = passport_result["analysis"]["causal_qualified_families"]
    anchors = anchor_plan(passport_result)
    source = load(FIELD_OUT / "analysis/final.json")
    field = np.load(FIELD_OUT / "raw/qwen3_4b_qualified_role_field.float16.npy", mmap_mode="r")
    index = read_rows(FIELD_OUT / "raw/role_field_index.jsonl")
    masks = np.load(PASSPORT_OUT / "atlas/discovery_loo_candidate_masks.uint8.npy", mmap_mode="r")
    protos = prototypes(field, index, families)
    field_order = passport_result["analysis"]["family_order"]
    pair_rows = pairs(index, families)
    lookup = compiled_lookup()
    model = None
    try:
        model, tokenizer, device, placement = contract.model_base.qwen_model()
        grid_rows, confirmation = confirmation_grid(model, device, lookup, pair_rows, anchors,
                                                    protos, masks, field_order)
        controls = confirmation_controls(model, device, lookup, pair_rows,
                                         confirmation["frozen"], protos, masks, field_order)
        lockbox_rows, lockbox_summary = lockbox(model, tokenizer, device, lookup, pair_rows,
                                               confirmation["frozen"], protos, masks, field_order)
        rescue_summary = rescue(model, device, lookup, pair_rows, confirmation["frozen"],
                                protos, masks, field_order)
    finally:
        if model is not None:
            contract.model_base.scope.parent.previous.model_base().release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        passports.close_mmap(field)
        passports.close_mmap(masks)
    write_rows(OUT / "causal/confirmation_grid.jsonl", grid_rows)
    write_rows(OUT / "causal/lockbox_rows.jsonl", lockbox_rows)
    save(OUT / "causal/confirmation.json", confirmation)
    save(OUT / "causal/confirmation_controls.json", controls)
    save(OUT / "causal/lockbox_summary.json", lockbox_summary)
    save(OUT / "causal/rescue.json", rescue_summary)
    strict = []
    for family in families:
        if (all(confirmation["frozen"][family][d]["qualified"] for d in ("call", "delete"))
                and all(controls[f"{family}|{d}"]["passed"] for d in ("call", "delete"))
                and all(lockbox_summary[f"{family}|{d}"]["strict_pass"] for d in ("call", "delete"))
                and rescue_summary[family]["strict_pass"]):
            strict.append(family)
    checks = {
        "prospective_anchors": all(anchors.get(family) for family in families),
        "confirmation_rows": bool(grid_rows), "all_lockbox_controls": all(
            set(row) >= {"wrong_family_margin", "wrong_role_margin", "wrong_checkpoint_margin", "sign_control_margin"}
            for row in lockbox_rows),
        "generation_recorded": all("correct_generation" in row and "wrong_generation" in row for row in lockbox_rows),
        "rescue_each_family": set(rescue_summary) == set(families),
        "no_unqualified_family_intervention": set(families) == set(passport_result["analysis"]["causal_qualified_families"]),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "placement": placement,
        "families": families, "anchors": anchors, "confirmation": confirmation,
        "confirmation_controls": controls, "lockbox_summary": lockbox_summary,
        "rescue": rescue_summary, "strict_causal_families": strict,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Only families in strict_causal_families have passed bidirectional candidate, generation, control and rescue gates; all others are route-level failures.",
        "next_authorization": "Run every cross-model fresh behavior panel sequentially, export important full-coordinate atlases and audit cleanup.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps({"anchors": anchors, "confirmation": confirmation,
                      "controls": controls, "lockbox": lockbox_summary,
                      "rescue": rescue_summary, "strict": strict, "checks": checks},
                     ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
