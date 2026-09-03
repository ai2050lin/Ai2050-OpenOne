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
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase2134_c600_c605_language_transport_campaign as patcher  # noqa: E402
import phase2234_c870_c884_broad_family_gear_contract as contract  # noqa: E402
import phase2235_c885_c904_qwen_broad_family_full_coordinate_tournament as tournament  # noqa: E402


PHASE = 2237
CAMPAIGNS = tuple(f"C{i}" for i in range(915, 921))
SOURCE = ROOT / "tests/glm5/result/phase2235_c885_c904_qwen_broad_family_full_coordinate_tournament"
MATERIAL = ROOT / "tests/glm5/result/phase2234_c870_c884_broad_family_conditional_gear_contract/material/fresh_qwen_compiled.jsonl"
OUT = ROOT / "tests/glm5/result/phase2237_c915_c920_full_coordinate_predictive_causal"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
LAYERS = (8, 16, 24, 32)
PRIMARY_LAYER = 24
DOSE = 1.0
GATES = {
    "minimum_pairs_per_direction": 8,
    "candidate_directional_rate": 0.60,
    "mean_margin_advantage_over_wrong": 0.05,
    "generation_accuracy_advantage_over_wrong": 0.10,
}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def make_inputs(row: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    pos = mask.long().cumsum(-1) - 1
    return ids, mask, pos


def predicted_response(base_full: np.ndarray, shared_beta: np.ndarray,
                       family_guard: np.ndarray, family: str) -> np.ndarray:
    base = base_full[list(contract.QPOINTS)]
    previous = base_full[[max(0, q - 1) for q in contract.QPOINTS]]
    shared = tournament.predict(shared_beta, tournament.shared_features(base[None], previous[None]))[0]
    family_i = contract.FAMILIES.index(family)
    residual = tournament.guard_predict(family_guard[family_i], base[None])[0]
    return shared + residual


def candidate_margin(logits: np.ndarray, row: dict, target_position: int) -> float:
    target_ids = row["candidate_ids"][target_position]
    other_ids = row["candidate_ids"][1 - target_position]
    if len(target_ids) != 1 or len(other_ids) != 1:
        raise RuntimeError("registered candidate codes must be one token")
    return float(logits[target_ids[0]] - logits[other_ids[0]])


def summarize(rows: list[dict], strict: list[str]) -> tuple[dict, list[str]]:
    summary = {}
    passed = []
    for family in strict:
        family_result = {}
        for direction in ("call", "delete"):
            subset = [row for row in rows if row["family"] == family and row["direction"] == direction
                      and row["layer"] == PRIMARY_LAYER]
            n = len(subset)
            directional = float(np.mean([row["correct_margin_effect"] > 0 for row in subset])) if subset else 0.0
            correct_effect = float(np.mean([row["correct_margin_effect"] for row in subset])) if subset else 0.0
            wrong_effect = float(np.mean([row["wrong_margin_effect"] for row in subset])) if subset else 0.0
            correct_gen = float(np.mean([row["correct_generation_target"] for row in subset])) if subset else 0.0
            wrong_gen = float(np.mean([row["wrong_generation_target"] for row in subset])) if subset else 0.0
            gate = (n >= GATES["minimum_pairs_per_direction"]
                    and directional >= GATES["candidate_directional_rate"]
                    and correct_effect - wrong_effect >= GATES["mean_margin_advantage_over_wrong"]
                    and correct_gen - wrong_gen >= GATES["generation_accuracy_advantage_over_wrong"])
            family_result[direction] = {
                "pairs": n, "candidate_directional_rate": directional,
                "correct_mean_margin_effect": correct_effect, "wrong_mean_margin_effect": wrong_effect,
                "margin_advantage_over_wrong": correct_effect - wrong_effect,
                "correct_generation_target_accuracy": correct_gen,
                "wrong_generation_target_accuracy": wrong_gen,
                "generation_advantage_over_wrong": correct_gen - wrong_gen, "passed": gate,
            }
        family_result["strict_bidirectional_pass"] = all(family_result[d]["passed"] for d in ("call", "delete"))
        if family_result["strict_bidirectional_pass"]:
            passed.append(family)
        summary[family] = family_result
    return summary, passed


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {family: {
        direction: {
            "n": values["pairs"], "dir": values["candidate_directional_rate"],
            "margin_adv": values["margin_advantage_over_wrong"],
            "gen_adv": values["generation_advantage_over_wrong"], "pass": values["passed"],
        } for direction, values in panel.items() if direction in ("call", "delete")
    } for family, panel in result["family_summary"].items()}
    formula = r"""
$$
\widehat{\Delta H}_{f}(H)=M_{\mathrm{shared}}(H)+G_f\!\left(\operatorname{sgn}H,\operatorname{sgn}H_{\mathrm{relation}}\right).
$$
$$
H'_{24,r,:}=H_{24,r,:}+s\widehat{\Delta H}_{f,24,r,:},\qquad
s=+1\ \text{(call)},\quad s=-1\ \text{(delete)}.
$$
"""
    text = f"""

## Phase {PHASE}: 新词锁箱的全坐标预测响应调用、删除与错族控制 [{stamp}]

**范围与冻结。** 本期执行 C915-C920。输入候选仅来自 Phase 2235 四面板严格通过的 `{result['authorized_families']}`。运行前冻结正式层 q24、描述层 q8/q16/q32、剂量 1、六角色全部 2560 坐标、错族循环控制以及双向通过门。没有按因果结果选择层、坐标、剂量或样本。

**测试原理与用例。** 对每个新词 lockbox 的真假配对，M5 只读取当前假命题初态，生成完整族条件响应。调用在假命题上加响应，删除在真命题上减同一响应，错族分支使用下一个严格候选族的同容量响应。测试覆盖类型链、施受事语态、否定、共指、嵌套态度、属性绑定、比较、量词中实际存在行为合格 lockbox 配对的全部中英文与表面。
{formula}
**结果和门槛。** 每方向至少 8 对；q24 正确响应的候选边际正向率至少 0.60；平均候选边际效应领先错族至少 0.05；自由生成目标准确率领先错族至少 0.10；调用与删除必须同时通过。结果为 `{json.dumps(compact, ensure_ascii=False)}`。双向严格通过族为 `{result['strict_causal_families']}`。逐样本、逐层、自然/正确/错族候选边际和自由生成均保存为 JSONL。

**理论进展与严格分析。** 这里搬运的不是 donor 差分，而是当前样本初态经过冻结逐坐标模型计算出的响应；这比固定方向干预更接近状态条件动力学。即便通过，也只说明该编译器在 q24 对这些提示具有局部调用与删除效力，不证明唯一性、最小性、参数级电路或自然语言普适性。若不通过，则 Phase 2235 保留为预测候选，因果资格关闭但其他路线继续。

**问题、硬伤和瓶颈。** 该预测器仍由族发现样本监督拟合；q24 是历史工作点而非数学必然；六角色同时写入不能定位最小角色子集；持续生成期间 hook 会在可用前缀位置重复施加响应；自由生成受输出码和解码路径影响；小模型、受控模板和行为筛选限制外推。

**结论和下一步。** 本期只授予 `{result['strict_causal_families']}` 局部双向因果候选资格。跨模型相同语义分母、全坐标图谱可视化和大场哈希清理继续执行，不因任何族失败停止。理论主体和 RDC 组织原则不改名，新数学授权仍关闭。

**相关文件。** 脚本 `tests/glm5/phase2237_c915_c920_full_coordinate_predictive_causal.py`；结果目录 `{OUT.relative_to(ROOT)}`；源系数和全场目录 `{SOURCE.relative_to(ROOT)}`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    source_final = json.loads((SOURCE / "analysis/final.json").read_text(encoding="utf-8"))
    strict = source_final["strict_family_candidates"]
    prereg = {
        "timestamp": datetime.now().astimezone().isoformat(), "phase": PHASE,
        "authorized_families": strict, "dataset": "fresh lockbox pairs only",
        "layers": list(LAYERS), "primary_layer": PRIMARY_LAYER, "dose": DOSE,
        "patch": "all six roles and all 2560 coordinates from M5 current-state prediction",
        "directions": ["call_false_to_true", "delete_true_to_false"],
        "control": "next authorized family, same predictor capacity, no norm rescaling",
        "gates": GATES, "no_post_reveal_tuning": True,
    }
    save(OUT / "protocol/preregistration.json", prereg)
    if not strict:
        result = {"phase": PHASE, "status": "closed_NA_no_authorized_candidate",
                  "all_checks_passed": True, "authorized_families": [], "strict_causal_families": []}
        save(final_path, result); append_memo(result); return result

    compiled = {row["case_id"]: row for row in read_rows(MATERIAL)}
    index = read_rows(SOURCE / "raw/fresh/hidden_index.jsonl")
    pairs = [row for row in tournament.pair_records(index, "lockbox") if row["family"] in strict]
    field = np.load(SOURCE / "raw/fresh/qualified_role_field.float16.npy", mmap_mode="r")
    shared_beta = np.asarray(np.load(SOURCE / "raw/shared_affine_coefficients.float16.npy"), dtype=np.float32)
    family_guard = np.asarray(np.load(SOURCE / "raw/family_guard_residual.float16.npy"), dtype=np.float32)
    predictions = {}
    try:
        for spec in pairs:
            base = np.asarray(field[spec["base"]], dtype=np.float32)
            predictions[spec["base_case_id"]] = {
                family: predicted_response(base, shared_beta, family_guard, family)
                for family in strict
            }
    finally:
        tournament.close_mmap(field)
    save(OUT / "protocol/execution_identity.json", {
        "source_final_sha256": file_hash(SOURCE / "analysis/final.json"),
        "shared_beta_sha256": file_hash(SOURCE / "raw/shared_affine_coefficients.float16.npy"),
        "family_guard_sha256": file_hash(SOURCE / "raw/family_guard_residual.float16.npy"),
        "pairs": len(pairs), "revealed_changes": "none",
    })

    model = None
    rows = []
    try:
        model, tokenizer, device, placement = contract.prior.qwen_model()
        identity = tuple(range(len(contract.ROLES)))
        strict_wrong = {family: strict[(i + 1) % len(strict)] for i, family in enumerate(strict)}
        for pair_i, spec in enumerate(pairs):
            false_row = compiled[spec["base_case_id"]]
            true_row = compiled[spec["changed_case_id"]]
            correct_field = predictions[spec["base_case_id"]][spec["family"]]
            wrong_family = strict_wrong[spec["family"]]
            wrong_field = predictions[spec["base_case_id"]][wrong_family]
            for direction, source_row, sign in (("call", false_row, 1.0), ("delete", true_row, -1.0)):
                ids, mask, pos = make_inputs(source_row, device)
                target_position = 1 - int(source_row["gold_position"])
                _, natural_logits = patcher.patched_forward_multi(model, ids, mask, pos, source_row["role_positions"], [])
                natural_margin = candidate_margin(natural_logits, source_row, target_position)
                target_answer = source_row["wrong_answer"]
                for layer in LAYERS:
                    q_i = contract.QPOINTS.index(layer)
                    correct_patch = [(layer, sign * DOSE * correct_field[q_i], identity)]
                    wrong_patch = [(layer, sign * DOSE * wrong_field[q_i], identity)]
                    _, correct_logits = patcher.patched_forward_multi(
                        model, ids, mask, pos, source_row["role_positions"], correct_patch)
                    _, wrong_logits = patcher.patched_forward_multi(
                        model, ids, mask, pos, source_row["role_positions"], wrong_patch)
                    correct_text = patcher.patched_greedy_text(
                        model, tokenizer, ids, mask, source_row["role_positions"], correct_patch, max_new_tokens=6)
                    wrong_text = patcher.patched_greedy_text(
                        model, tokenizer, ids, mask, source_row["role_positions"], wrong_patch, max_new_tokens=6)
                    correct_margin = candidate_margin(correct_logits, source_row, target_position)
                    wrong_margin = candidate_margin(wrong_logits, source_row, target_position)
                    rows.append({
                        "family": spec["family"], "wrong_family": wrong_family,
                        "language": spec["language"], "surface": spec["surface"], "unit": spec["unit"],
                        "direction": direction, "layer": layer, "dose": DOSE,
                        "source_case_id": source_row["case_id"], "target_answer": target_answer,
                        "natural_target_margin": natural_margin,
                        "correct_target_margin": correct_margin, "wrong_target_margin": wrong_margin,
                        "correct_margin_effect": correct_margin - natural_margin,
                        "wrong_margin_effect": wrong_margin - natural_margin,
                        "correct_generation": correct_text, "wrong_generation": wrong_text,
                        "correct_generation_target": parse_code(correct_text, source_row) == target_answer,
                        "wrong_generation_target": parse_code(wrong_text, source_row) == target_answer,
                    })
            if pair_i % 8 == 0:
                print(f"[causal] {pair_i}/{len(pairs)}", flush=True)
    finally:
        tournament.release_model(model)
        gc.collect()
    write_rows(OUT / "analysis/intervention_rows.jsonl", rows)
    summary, passed = summarize(rows, strict)
    save(OUT / "analysis/family_summary.json", summary)
    checks = {
        "source_passed": source_final["all_checks_passed"], "authorized_nonempty": bool(strict),
        "all_pairs_ran": len(rows) == len(pairs) * 2 * len(LAYERS),
        "primary_layer_present": all(any(row["family"] == family and row["layer"] == PRIMARY_LAYER for row in rows)
                                     for family in strict),
        "both_directions": set(row["direction"] for row in rows) == {"call", "delete"},
        "all_coordinates": shared_beta.shape[-1] == contract.DIM and family_guard.shape[-1] == contract.DIM,
        "finite": all(np.isfinite(row[k]) for row in rows for k in
                      ("natural_target_margin", "correct_target_margin", "wrong_target_margin")),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()), "authorized_families": strict,
        "pairs": len(pairs), "intervention_rows": len(rows), "layers": list(LAYERS),
        "primary_layer": PRIMARY_LAYER, "gates": GATES, "family_summary": summary,
        "strict_causal_families": passed, "placement": placement,
        "strict_conclusion": "Passing is bidirectional local sufficiency/necessity evidence for this full-coordinate current-state predictor at q24; it is not uniqueness, minimality, or a parameter circuit.",
        "next_authorization": "Continue exact-denominator cross-model topology and visualization for every observable family regardless of causal outcome.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
