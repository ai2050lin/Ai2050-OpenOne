"""C1049-C1064 state-consistent call, delete and rescue causal adjudication."""
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
CONTRACT_OUT = RESULT / "phase2247_c1001_c1016_natural_flagship_contract"
FIELD_OUT = RESULT / "phase2248_c1017_c1030_qwen_natural_full_field"
PREDICT_OUT = RESULT / "phase2249_c1031_c1048_full_coordinate_prediction"
OUT = RESULT / "phase2250_c1049_c1064_state_consistent_causal"
sys.path.insert(0, str(TESTS))

import phase2134_c600_c605_language_transport_campaign as patcher
import phase2247_c1001_c1016_natural_flagship_contract as contract
import phase2249_c1031_c1048_full_coordinate_prediction as prediction


PHASE = 2250
CAMPAIGNS = tuple(f"C{i}" for i in range(1049, 1065))
STRATA = ("all", "low_half", "high_half", "positive_response", "negative_response")
DOSES = (0.25, 0.5, 1.0)
MODES = ("one_shot", "repeat_each_generated_token")
IDENTITY = tuple(range(len(contract.ROLES)))
SHIFTED = tuple((i + 1) % len(contract.ROLES) for i in range(len(contract.ROLES)))


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def contextual_spans(tokenizer, ids: list[int], value: str) -> list[list[int]]:
    exact = contract.compiler.graph_base.name_spans(tokenizer, ids, value)
    if exact:
        return exact
    needle = max(1, len(tokenizer.encode(value, add_special_tokens=False)))
    for width in range(1, needle + 5):
        found = []
        for start in range(0, len(ids) - width + 1):
            if value in tokenizer.decode(ids[start:start + width], skip_special_tokens=True):
                found.append(list(range(start, start + width)))
        if found:
            return found
    return []


def free_positions(tokenizer, row: dict) -> dict[str, list[int]]:
    ids = row["free_prompt_ids"]
    positions = {}
    for role, value in row["role_values"].items():
        spans = contextual_spans(tokenizer, ids, value)
        if not spans:
            raise RuntimeError((row["case_id"], role, value, "free_prompt"))
        positions[role] = spans[-1] if role == "query" else spans[0]
    positions["boundary"] = [len(ids) - 1]
    return positions


def inputs(row: dict, device, free: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq = row["free_prompt_ids"] if free else row["prompt_ids"]
    ids = torch.tensor([seq], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    pos = mask.long().cumsum(-1) - 1
    return ids, mask, pos


def candidate_margin(logits: np.ndarray, row: dict, target_position: int) -> float:
    candidate_ids = [int(value[0]) for value in row["candidate_ids"]]
    current = 1 - target_position
    return float(logits[candidate_ids[target_position]] - logits[candidate_ids[current]])


def response_stratum(response: np.ndarray, name: str) -> np.ndarray:
    response = np.asarray(response, np.float32)
    if name == "all":
        return response.copy()
    magnitude = np.abs(response)
    median = float(np.median(magnitude))
    if name == "low_half":
        mask = magnitude <= median
    elif name == "high_half":
        mask = magnitude > median
    elif name == "positive_response":
        mask = response > 0
    elif name == "negative_response":
        mask = response < 0
    else:
        raise ValueError(name)
    return np.where(mask, response, 0.0).astype(np.float32)


def sign_permutation(response: np.ndarray) -> np.ndarray:
    return np.asarray(response, np.float32)[:, ::-1].copy()


def generated_text(model, tokenizer, row: dict, device, role_positions: dict,
                   patches: list[tuple], mode: str) -> str:
    base = model.model
    by_q = defaultdict(list)
    for q, response, role_order in patches:
        by_q[int(q)].append((np.asarray(response, np.float32), role_order))
    handles = []
    for q, values in by_q.items():
        def make_hook(items):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                changed = tensor.clone()
                for response, role_order in items:
                    for target_i, role in enumerate(contract.ROLES):
                        source_i = role_order[target_i]
                        position = int(role_positions[role][-1])
                        if position < changed.shape[1]:
                            changed[0, position] = changed[0, position] + torch.tensor(
                                response[source_i], dtype=changed.dtype, device=changed.device)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            return hook
        handles.append(base.layers[q - 1].register_forward_hook(make_hook(values)))
    ids, mask, _ = inputs(row, device, free=True)
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False,
                use_cache=(mode == "one_shot"),
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()
    return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()


def prediction_lookup(pair_index: list[dict], selected) -> tuple[dict, dict]:
    by_pair = {row["pair_id"]: np.asarray(selected[row["prediction_index"]], np.float32) for row in pair_index}
    by_semantic = {(row["family"], row["language"], row["unit"], row["surface"]): row["pair_id"] for row in pair_index}
    return by_pair, by_semantic


def wrong_family_for(family: str, strict: list[str]) -> str:
    return contract.FAMILIES[(contract.FAMILIES.index(family) + 1) % len(contract.FAMILIES)]


def patch_forward(model, row: dict, device, patches: list[tuple]) -> np.ndarray:
    ids, mask, pos = inputs(row, device)
    _state, logits = patcher.patched_forward_multi(
        model, ids, mask, pos, row["role_positions"], patches)
    return logits


def setting_summaries(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["direction"], row["stratum"], row["dose"])].append(row)
    output = {}
    for key, subset in grouped.items():
        correct_gain = np.array([row["correct_margin"] - row["natural_margin"] for row in subset])
        control = np.array([max(row["wrong_family_margin"], row["sign_permutation_margin"]) for row in subset])
        output["|".join(map(str, key))] = {
            "pairs": len(subset), "candidate_direction_rate": float(np.mean(correct_gain > 0)),
            "mean_margin_gain": float(np.mean(correct_gain)),
            "margin_advantage_over_wrong": float(np.mean(np.array([row["correct_margin"] for row in subset]) - control)),
        }
    return output


def select_settings(summaries: dict, strict: list[str]) -> dict:
    selected = {}
    for family in strict:
        selected[family] = {}
        for direction in ("call", "delete"):
            choices = []
            for stratum in STRATA:
                for dose in DOSES:
                    row = summaries[f"{family}|{direction}|{stratum}|{dose}"]
                    passes = (row["pairs"] >= contract.CAUSAL_GATES["minimum_pairs"]
                              and row["candidate_direction_rate"] >= contract.CAUSAL_GATES["candidate_direction_rate"]
                              and row["margin_advantage_over_wrong"] >= contract.CAUSAL_GATES["candidate_margin_advantage"])
                    if passes:
                        choices.append((row["margin_advantage_over_wrong"], row["candidate_direction_rate"],
                                        -STRATA.index(stratum), -DOSES.index(dose), stratum, dose))
            if choices:
                best = max(choices)
                selected[family][direction] = {"qualified": True, "stratum": best[-2], "dose": best[-1],
                                                "selection_source": "fresh_confirmation_only"}
            else:
                selected[family][direction] = {"qualified": False, "reason": "no_frozen_grid_cell_passed"}
    return selected


def run_candidate_grid(model, device, compiled: dict, pair_index: list[dict], predictions: dict,
                       semantic: dict, selection: dict, strict: list[str]) -> tuple[list[dict], dict, dict]:
    rows = []
    qindex = {q: i for i, q in enumerate(prediction.QPOINTS)}
    confirmation = [row for row in pair_index if row["partition"] == "fresh_confirmation" and row["family"] in strict]
    for pair_i, pair in enumerate(confirmation):
        family = pair["family"]
        wrong_family = wrong_family_for(family, strict)
        wrong_pair = semantic[(wrong_family, pair["language"], pair["unit"], pair["surface"])]
        q = int(selection[family]["causal_checkpoint"])
        qi = qindex[q]
        correct_full = predictions[pair["pair_id"]][qi]
        wrong_full = predictions[wrong_pair][qi]
        false_row = compiled[pair["pair_id"] + "_t0_base"]
        true_row = compiled[pair["pair_id"] + "_t1_base"]
        for direction, source, sign in (("call", false_row, 1.0), ("delete", true_row, -1.0)):
            target_position = 1 - int(source["gold_position"])
            natural = candidate_margin(patch_forward(model, source, device, []), source, target_position)
            for stratum in STRATA:
                correct = response_stratum(correct_full, stratum)
                wrong = response_stratum(wrong_full, stratum)
                permuted = sign_permutation(correct)
                for dose in DOSES:
                    correct_margin = candidate_margin(
                        patch_forward(model, source, device, [(q, sign * dose * correct, IDENTITY)]), source, target_position)
                    wrong_margin = candidate_margin(
                        patch_forward(model, source, device, [(q, sign * dose * wrong, IDENTITY)]), source, target_position)
                    sign_margin = candidate_margin(
                        patch_forward(model, source, device, [(q, sign * dose * permuted, IDENTITY)]), source, target_position)
                    rows.append({
                        "family": family, "wrong_family": wrong_family, "direction": direction,
                        "pair_id": pair["pair_id"], "source_case_id": source["case_id"],
                        "language": pair["language"], "unit": pair["unit"], "surface": pair["surface"],
                        "checkpoint": q, "stratum": stratum, "dose": dose,
                        "natural_margin": natural, "correct_margin": correct_margin,
                        "wrong_family_margin": wrong_margin, "sign_permutation_margin": sign_margin,
                    })
        if pair_i % 8 == 0:
            print(f"[causal-grid] {pair_i}/{len(confirmation)}", flush=True)
    summaries = setting_summaries(rows)
    frozen = select_settings(summaries, strict)
    control_rows = []
    for pair in confirmation:
        family = pair["family"]
        q = int(selection[family]["causal_checkpoint"])
        qi = qindex[q]
        q_wrong = prediction.QPOINTS[(prediction.QPOINTS.index(q) + 1) % len(prediction.QPOINTS)]
        for direction, source_id, sign in (("call", pair["pair_id"] + "_t0_base", 1.0),
                                           ("delete", pair["pair_id"] + "_t1_base", -1.0)):
            setting = frozen[family][direction]
            if not setting["qualified"]:
                continue
            source = compiled[source_id]
            target_position = 1 - int(source["gold_position"])
            response = response_stratum(predictions[pair["pair_id"]][qi], setting["stratum"])
            dose = float(setting["dose"])
            correct = candidate_margin(patch_forward(model, source, device, [(q, sign * dose * response, IDENTITY)]), source, target_position)
            wrong_role = candidate_margin(patch_forward(model, source, device, [(q, sign * dose * response, SHIFTED)]), source, target_position)
            wrong_checkpoint = candidate_margin(patch_forward(model, source, device, [(q_wrong, sign * dose * response, IDENTITY)]), source, target_position)
            control_rows.append({"family": family, "direction": direction, "pair_id": pair["pair_id"],
                                 "correct_margin": correct, "wrong_role_margin": wrong_role,
                                 "wrong_checkpoint_margin": wrong_checkpoint})
    control_summary = {}
    for family in strict:
        for direction in ("call", "delete"):
            subset = [row for row in control_rows if row["family"] == family and row["direction"] == direction]
            if subset:
                advantage = float(np.mean([row["correct_margin"] - max(row["wrong_role_margin"], row["wrong_checkpoint_margin"]) for row in subset]))
                control_summary[f"{family}|{direction}"] = {"pairs": len(subset), "advantage": advantage,
                                                             "passed": advantage >= contract.CAUSAL_GATES["candidate_margin_advantage"]}
            else:
                control_summary[f"{family}|{direction}"] = {"pairs": 0, "advantage": None, "passed": False}
    return rows, frozen, {"setting_summaries": summaries, "control_rows": control_rows,
                          "control_summary": control_summary}


def run_lockbox_generation(model, tokenizer, device, compiled: dict, pair_index: list[dict], predictions: dict,
                           semantic: dict, selection: dict, strict: list[str], frozen: dict) -> tuple[list[dict], dict]:
    qindex = {q: i for i, q in enumerate(prediction.QPOINTS)}
    lockbox = [row for row in pair_index if row["partition"] == "fresh_lockbox" and row["family"] in strict]
    rows = []
    for pair_i, pair in enumerate(lockbox):
        family = pair["family"]
        wrong_family = wrong_family_for(family, strict)
        wrong_pair = semantic[(wrong_family, pair["language"], pair["unit"], pair["surface"])]
        q = int(selection[family]["causal_checkpoint"]); qi = qindex[q]
        q_next = prediction.QPOINTS[(prediction.QPOINTS.index(q) + 1) % len(prediction.QPOINTS)]
        q_next_i = qindex[q_next]
        for direction, source_id, sign in (("call", pair["pair_id"] + "_t0_base", 1.0),
                                           ("delete", pair["pair_id"] + "_t1_base", -1.0)):
            setting = frozen[family][direction]
            if not setting["qualified"]:
                continue
            source = compiled[source_id]
            role_positions = free_positions(tokenizer, source)
            correct = response_stratum(predictions[pair["pair_id"]][qi], setting["stratum"])
            wrong = response_stratum(predictions[wrong_pair][qi], setting["stratum"])
            dose = float(setting["dose"])
            target_code = source["true_code"] if direction == "call" else source["false_code"]
            for mode in MODES:
                correct_text = generated_text(model, tokenizer, source, device, role_positions,
                                              [(q, sign * dose * correct, IDENTITY)], mode)
                wrong_text = generated_text(model, tokenizer, source, device, role_positions,
                                            [(q, sign * dose * wrong, IDENTITY)], mode)
                rows.append({
                    "kind": "direction", "family": family, "direction": direction, "mode": mode,
                    "pair_id": pair["pair_id"], "target_code": target_code,
                    "correct_text": correct_text, "wrong_text": wrong_text,
                    "correct_target": parse_code(correct_text, source) == target_code,
                    "wrong_target": parse_code(wrong_text, source) == target_code,
                })
        delete_setting = frozen[family]["delete"]
        if delete_setting["qualified"]:
            source = compiled[pair["pair_id"] + "_t1_base"]
            role_positions = free_positions(tokenizer, source)
            q_response = response_stratum(predictions[pair["pair_id"]][qi], delete_setting["stratum"])
            next_response = response_stratum(predictions[pair["pair_id"]][q_next_i], delete_setting["stratum"])
            wrong_next = response_stratum(predictions[wrong_pair][q_next_i], delete_setting["stratum"])
            dose = float(delete_setting["dose"])
            for mode in MODES:
                deleted = generated_text(model, tokenizer, source, device, role_positions,
                                         [(q, -dose * q_response, IDENTITY)], mode)
                rescued = generated_text(model, tokenizer, source, device, role_positions,
                                         [(q, -dose * q_response, IDENTITY), (q_next, dose * next_response, IDENTITY)], mode)
                wrong_rescue = generated_text(model, tokenizer, source, device, role_positions,
                                              [(q, -dose * q_response, IDENTITY), (q_next, dose * wrong_next, IDENTITY)], mode)
                rows.append({
                    "kind": "rescue", "family": family, "direction": "delete_rescue", "mode": mode,
                    "pair_id": pair["pair_id"], "target_code": source["true_code"],
                    "deleted_text": deleted, "rescued_text": rescued, "wrong_rescue_text": wrong_rescue,
                    "deleted_target": parse_code(deleted, source) == source["true_code"],
                    "rescued_target": parse_code(rescued, source) == source["true_code"],
                    "wrong_rescue_target": parse_code(wrong_rescue, source) == source["true_code"],
                })
        if pair_i % 8 == 0:
            print(f"[causal-generation] {pair_i}/{len(lockbox)}", flush=True)
    summary = {}
    passed = []
    for family in strict:
        family_pass = True
        summary[family] = {}
        for direction in ("call", "delete"):
            summary[family][direction] = {}
            for mode in MODES:
                subset = [row for row in rows if row["kind"] == "direction" and row["family"] == family and row["direction"] == direction and row["mode"] == mode]
                correct = float(np.mean([row["correct_target"] for row in subset])) if subset else 0.0
                wrong = float(np.mean([row["wrong_target"] for row in subset])) if subset else 0.0
                mode_pass = bool(subset) and correct - wrong >= contract.CAUSAL_GATES["generation_accuracy_advantage"]
                summary[family][direction][mode] = {"pairs": len(subset), "correct": correct,
                                                     "wrong_family": wrong, "advantage": correct - wrong,
                                                     "passed": mode_pass}
                family_pass &= mode_pass
        summary[family]["rescue"] = {}
        for mode in MODES:
            subset = [row for row in rows if row["kind"] == "rescue" and row["family"] == family and row["mode"] == mode]
            rescued = float(np.mean([row["rescued_target"] for row in subset])) if subset else 0.0
            control = max(float(np.mean([row["deleted_target"] for row in subset])) if subset else 0.0,
                          float(np.mean([row["wrong_rescue_target"] for row in subset])) if subset else 0.0)
            mode_pass = bool(subset) and rescued - control >= contract.CAUSAL_GATES["correct_rescue_advantage"]
            summary[family]["rescue"][mode] = {"pairs": len(subset), "correct_rescue": rescued,
                                                "best_control": control, "advantage": rescued - control,
                                                "passed": mode_pass}
            family_pass &= mode_pass
        if family_pass:
            passed.append(family)
    return rows, {"families": summary, "strict_generation_and_rescue": passed}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = f"""

## Phase {PHASE}: 状态一致的调用、删除与救援因果裁决（C1049-C1064） [{stamp}]

**测试原理与用例。** 本期只接收 Phase2249 的严格全轨迹预测族。对fresh confirmation的真假配对，冻结预测检查点后扫描全场、低幅值半场、高幅值半场、正响应和负响应五个完备坐标分层，以及0.25/0.5/1.0三剂量；与错族、错角色、错层和等范数坐标符号置换比较。冻结设置随后原封不动地进入fresh lockbox，分别测试false到true调用、true到false删除、单次写入、每生成步重算写入，以及删除后在下一检查点的正确/错族救援。

**公式。** 对预测响应束按检查点写入：

$$
H'_{{q,r,j}}=H_{{q,r,j}}+s\\alpha M_{{r,j}}\\widehat{{\\Delta H}}_{{q,r,j}},\\qquad s\\in\\{{+1,-1\\}}.
$$

救援使用两个冻结事件：

$$
H' = H-\\alpha\\widehat{{\\Delta H}}_q+\\alpha\\widehat{{\\Delta H}}_{{q'}}.
$$

**结果汇总。** 输入严格预测族为 `{json.dumps(result['input_predictive_families'], ensure_ascii=False)}`；confirmation冻结设置为 `{json.dumps(result['frozen_settings'], ensure_ascii=False)}`；锁箱生成与救援账为 `{json.dumps(result['generation_summary'], ensure_ascii=False)}`。最终同时通过候选控制、双向生成和正确救援的族为 `{json.dumps(result['strict_causal_families'], ensure_ascii=False)}`。

**分析与理论进展。** 只有最终列表中的族可称为“这套预测响应在该检查点和角色束上具有状态一致的局部因果效力”；仍不能称为唯一、最小或参数级齿轮。候选边界通过但自由生成失败，说明只能移动局部分数；调用通过但删除失败，说明充分性与必要性不对称；救援失败则说明预测响应不是可组合的下游状态。

**问题、硬伤与结论。** 这是强幅值激活干预；坐标半场按预测幅值确定但没有删去任何坐标类别；逐生成步模式通过关闭cache重算前缀实现，计算路径与正常cache解码不同；错族只有本轮严格族内循环；小模型和受控材料限制外推。失败按路线记缺失，不停止跨模型和图谱观察。工程检查 `{result['all_checks_passed']}`。

**相关文件。** 脚本 `tests/glm5/phase2250_c1049_c1064_state_consistent_causal.py`；结果 `tests/glm5/result/phase2250_c1049_c1064_state_consistent_causal`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    source = load(PREDICT_OUT / "analysis/final.json")
    strict = list(source["strict_predictive_families"])
    protocol = {"phase": PHASE, "campaigns": list(CAMPAIGNS), "input_families": strict,
                "coordinate_strata": list(STRATA), "doses": list(DOSES), "modes": list(MODES),
                "selection": "fresh_confirmation grid then untouched fresh_lockbox",
                "controls": ["wrong_family", "wrong_role", "wrong_checkpoint", "equal_norm_sign_permutation"],
                "failure_policy": "route-level missingness"}
    save(OUT / "protocol/preregistration.json", protocol)
    if not strict:
        checks = {"source_complete": source["all_checks_passed"], "no_ineligible_model_run": True}
        result = {"phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed_NA",
                  "timestamp": datetime.now().astimezone().isoformat(),
                  "input_predictive_families": [], "frozen_settings": {},
                  "generation_summary": {"families": {}, "strict_generation_and_rescue": []},
                  "strict_causal_families": [], "checks": checks, "all_checks_passed": all(checks.values()),
                  "strict_conclusion": "Causal testing is NA because no family passed the preregistered predictive gate."}
        save(final_path, result); append_memo(result); return result
    compiled_rows = read_rows(CONTRACT_OUT / "material/fresh_broad_qwen_compiled.jsonl")
    compiled = {row["case_id"]: row for row in compiled_rows}
    pair_index = read_rows(PREDICT_OUT / "raw/fresh_pair_index.jsonl")
    selected = np.load(prediction.PREDICTIONS, mmap_mode="r")
    predictions, semantic = prediction_lookup(pair_index, selected)
    selection = source["selection"]
    model = None
    try:
        model, tokenizer, device, placement = contract.prior.qwen_model()
        grid_rows, frozen, controls = run_candidate_grid(
            model, device, compiled, pair_index, predictions, semantic, selection, strict)
        generation_rows, generation_summary = run_lockbox_generation(
            model, tokenizer, device, compiled, pair_index, predictions, semantic,
            selection, strict, frozen)
    finally:
        close = getattr(selected, "_mmap", None)
        if close is not None:
            close.close()
        if model is not None:
            contract.prior.scope.parent.previous.model_base().release_bf16(model)
        gc.collect()
    write_rows(OUT / "analysis/confirmation_grid_rows.jsonl", grid_rows)
    write_rows(OUT / "analysis/confirmation_control_rows.jsonl", controls["control_rows"])
    write_rows(OUT / "analysis/lockbox_generation_rows.jsonl", generation_rows)
    save(OUT / "analysis/confirmation_setting_summaries.json", controls["setting_summaries"])
    save(OUT / "analysis/confirmation_control_summary.json", controls["control_summary"])
    save(OUT / "analysis/frozen_settings.json", frozen)
    save(OUT / "analysis/generation_summary.json", generation_summary)
    strict_causal = []
    for family in generation_summary["strict_generation_and_rescue"]:
        candidate_ok = all(frozen[family][direction]["qualified"] and controls["control_summary"][f"{family}|{direction}"]["passed"]
                           for direction in ("call", "delete"))
        if candidate_ok:
            strict_causal.append(family)
    checks = {
        "source_complete": source["all_checks_passed"],
        "grid_complete": bool(grid_rows) and set(row["stratum"] for row in grid_rows) == set(STRATA)
                         and set(row["dose"] for row in grid_rows) == set(DOSES),
        "both_directions": set(row["direction"] for row in grid_rows) == {"call", "delete"},
        "all_controls": all(set(row) >= {"correct_margin", "wrong_role_margin", "wrong_checkpoint_margin"}
                            for row in controls["control_rows"]),
        "lockbox_only_generation": all(next(x for x in pair_index if x["pair_id"] == row["pair_id"])["partition"] == "fresh_lockbox" for row in generation_rows),
        "both_generation_modes": set(row["mode"] for row in generation_rows) == set(MODES) if generation_rows else True,
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "input_predictive_families": strict, "frozen_settings": frozen,
        "confirmation_controls": controls["control_summary"],
        "generation_summary": generation_summary, "strict_causal_families": strict_causal,
        "placement": placement, "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Only strict_causal_families have state-consistent local evidence under all frozen controls; no uniqueness or parameter-mechanism claim is licensed.",
        "next_authorization": "Continue exact-semantic cross-model observation and full-coordinate visualization regardless of causal outcome.",
    }
    save(final_path, result); append_memo(result)
    print(json.dumps({"strict_causal": strict_causal, "checks": checks}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
