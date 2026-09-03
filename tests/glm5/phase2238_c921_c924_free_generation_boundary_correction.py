from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase2134_c600_c605_language_transport_campaign as patcher  # noqa: E402
import phase2234_c870_c884_broad_family_gear_contract as contract  # noqa: E402
import phase2235_c885_c904_qwen_broad_family_full_coordinate_tournament as tournament  # noqa: E402
import phase2237_c915_c920_full_coordinate_predictive_causal as causal  # noqa: E402


PHASE = 2238
CAMPAIGNS = tuple(f"C{i}" for i in range(921, 925))
SOURCE = causal.OUT
FIELD_SOURCE = causal.SOURCE
MATERIAL = causal.MATERIAL
OUT = ROOT / "tests/glm5/result/phase2238_c921_c924_free_generation_boundary_correction"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def contextual_spans(tokenizer, ids: list[int], value: str) -> list[list[int]]:
    exact = contract.compiler.graph_base.name_spans(tokenizer, ids, value)
    if exact:
        return exact
    needle_len = max(1, len(tokenizer.encode(value, add_special_tokens=False)))
    for width in range(1, needle_len + 4):
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


def inputs(row: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([row["free_prompt_ids"]], dtype=torch.long, device=device)
    return ids, torch.ones_like(ids)


def summarize(candidate_rows: list[dict], generation_rows: list[dict], strict: list[str]) -> tuple[dict, list[str]]:
    by_key = {(row["family"], row["direction"], row["source_case_id"]): row for row in generation_rows}
    merged = []
    for row in candidate_rows:
        if row["layer"] != causal.PRIMARY_LAYER:
            continue
        generation = by_key[(row["family"], row["direction"], row["source_case_id"])]
        merged.append({**row, **{
            "correct_generation": generation["correct_generation"],
            "wrong_generation": generation["wrong_generation"],
            "correct_generation_target": generation["correct_generation_target"],
            "wrong_generation_target": generation["wrong_generation_target"],
        }})
    summary, passed = causal.summarize(merged, strict)
    return summary, passed


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {family: {
        direction: {"n": value["pairs"], "dir": value["candidate_directional_rate"],
                    "margin_adv": value["margin_advantage_over_wrong"],
                    "gen_adv": value["generation_advantage_over_wrong"], "pass": value["passed"]}
        for direction, value in panel.items() if direction in ("call", "delete")
    } for family, panel in result["family_summary"].items()}
    formula = r"""
$$
\phi_{\mathrm{free}}(r)=\operatorname{span}_{\mathrm{token}}
\left(x_{\mathrm{free}},\operatorname{text}(r)\right),\qquad
\phi_{\mathrm{free}}(\mathrm{boundary})=|x_{\mathrm{free}}|-1.
$$
"""
    text = f"""

## Phase {PHASE}: 自由生成边界与角色坐标修复后的因果重裁 [{stamp}]

**错误审计。** Phase 2237 的候选 A/B 边际读取有效，但自由生成在 A/B 候选提示上运行、再用自然答案码解析，导致生成目标命中率被机械记为 0。该值不构成模型或因果阴性证据。本期不覆盖旧结果，以 C921-C924 新 Phase 修复测量合同。

**冻结修复与测试原理。** q24、剂量 1、六角色全 2560 坐标、正确/错族预测器、96 个 fresh lockbox 配对和原通过门全部不变。唯一修复是对每条 `free_prompt_ids` 独立编译五个语义角色和真实 assistant 生成边界，再用 Yes/No、True/False、Supported/Unsupported、Entailed/Contradicted 自然码判分。
{formula}
**结果汇总。** 修复后的正式结果为 `{json.dumps(compact, ensure_ascii=False)}`；双向严格因果候选为 `{result['strict_causal_families']}`。Phase 2237 候选边际与本期自由生成按 `family + direction + case_id` 一一合并，未删除失败样本。

**理论进展与限制。** 本期把“候选边际可推动”与“完整生成可改写”重新放在合法的同一语义任务上。通过仍只表示 q24 局部编译器的双向效力，不证明唯一、最小或参数级机制；失败则说明大幅候选边际移动没有形成族特异的完整生成优势。自由生成 hook 在解码步持续生效，依旧是强干预。

**问题、硬伤和结论。** 这是揭盲后的仪器修复，因此只能修复预先存在的同一读出，不能新增层或调门槛。中英文 BPE 的角色跨度仍采用“精确跨度优先、最小包含区间兜底”。正式结论只采用本期重裁，Phase 2237 的全零生成列标记为 invalid，不再用于机制判断。跨模型、可视化和清理继续。

**相关文件。** 脚本 `tests/glm5/phase2238_c921_c924_free_generation_boundary_correction.py`；结果 `{OUT.relative_to(ROOT)}`；被纠正结果 `{SOURCE.relative_to(ROOT)}`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    source_final = json.loads((SOURCE / "analysis/final.json").read_text(encoding="utf-8"))
    strict = source_final["authorized_families"]
    prereg = {
        "timestamp": datetime.now().astimezone().isoformat(), "phase": PHASE,
        "reason": "Phase2237 generation used candidate prompt but natural-code parser",
        "unchanged": {"pairs": source_final["pairs"], "layer": causal.PRIMARY_LAYER,
                      "dose": causal.DOSE, "families": strict, "gates": causal.GATES},
        "only_change": "compile role spans and boundary on free_prompt_ids; score natural answer codes",
        "forbidden": "new layers, doses, thresholds, samples, or family selection",
    }
    save(OUT / "protocol/preregistration.json", prereg)
    compiled = {row["case_id"]: row for row in causal.read_rows(MATERIAL)}
    index = causal.read_rows(FIELD_SOURCE / "raw/fresh/hidden_index.jsonl")
    pairs = [row for row in tournament.pair_records(index, "lockbox") if row["family"] in strict]
    field = np.load(FIELD_SOURCE / "raw/fresh/qualified_role_field.float16.npy", mmap_mode="r")
    shared_beta = np.asarray(np.load(FIELD_SOURCE / "raw/shared_affine_coefficients.float16.npy"), dtype=np.float32)
    family_guard = np.asarray(np.load(FIELD_SOURCE / "raw/family_guard_residual.float16.npy"), dtype=np.float32)
    predictions = {}
    try:
        for spec in pairs:
            base = np.asarray(field[spec["base"]], dtype=np.float32)
            predictions[spec["base_case_id"]] = {
                family: causal.predicted_response(base, shared_beta, family_guard, family) for family in strict}
    finally:
        tournament.close_mmap(field)

    rows = []
    model = None
    try:
        model, tokenizer, device, placement = contract.prior.qwen_model()
        identity = tuple(range(len(contract.ROLES)))
        wrong_map = {family: strict[(i + 1) % len(strict)] for i, family in enumerate(strict)}
        for pair_i, spec in enumerate(pairs):
            false_row = compiled[spec["base_case_id"]]
            true_row = compiled[spec["changed_case_id"]]
            q_i = contract.QPOINTS.index(causal.PRIMARY_LAYER)
            correct = predictions[spec["base_case_id"]][spec["family"]][q_i]
            wrong_family = wrong_map[spec["family"]]
            wrong = predictions[spec["base_case_id"]][wrong_family][q_i]
            for direction, row, sign in (("call", false_row, 1.0), ("delete", true_row, -1.0)):
                role_positions = free_positions(tokenizer, row)
                ids, mask = inputs(row, device)
                correct_patch = [(causal.PRIMARY_LAYER, sign * correct, identity)]
                wrong_patch = [(causal.PRIMARY_LAYER, sign * wrong, identity)]
                correct_text = patcher.patched_greedy_text(
                    model, tokenizer, ids, mask, role_positions, correct_patch, max_new_tokens=6)
                wrong_text = patcher.patched_greedy_text(
                    model, tokenizer, ids, mask, role_positions, wrong_patch, max_new_tokens=6)
                target_answer = row["wrong_answer"]
                rows.append({
                    "family": spec["family"], "wrong_family": wrong_family,
                    "direction": direction, "source_case_id": row["case_id"],
                    "language": spec["language"], "surface": spec["surface"], "unit": spec["unit"],
                    "target_answer": target_answer, "correct_generation": correct_text,
                    "wrong_generation": wrong_text,
                    "correct_generation_target": causal.parse_code(correct_text, row) == target_answer,
                    "wrong_generation_target": causal.parse_code(wrong_text, row) == target_answer,
                    "free_role_positions": role_positions,
                })
            if pair_i % 12 == 0:
                print(f"[free-correction] {pair_i}/{len(pairs)}", flush=True)
    finally:
        tournament.release_model(model)
        gc.collect()
    write_rows(OUT / "analysis/free_generation_rows.jsonl", rows)
    candidate_rows = causal.read_rows(SOURCE / "analysis/intervention_rows.jsonl")
    summary, passed = summarize(candidate_rows, rows, strict)
    save(OUT / "analysis/family_summary.json", summary)
    checks = {
        "source_complete": source_final["all_checks_passed"],
        "same_pair_count": len(rows) == len(pairs) * 2,
        "all_free_boundaries": all(row["free_role_positions"]["boundary"] for row in rows),
        "both_directions": set(row["direction"] for row in rows) == {"call", "delete"},
        "no_gate_change": prereg["unchanged"]["gates"] == source_final["gates"],
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed_corrected",
        "timestamp": datetime.now().astimezone().isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()), "invalidated_measurement": "Phase2237 generation columns only",
        "retained_measurement": "Phase2237 candidate margin columns",
        "pairs": len(pairs), "generation_rows": len(rows), "gates": causal.GATES,
        "family_summary": summary, "strict_causal_families": passed, "placement": placement,
        "strict_conclusion": "This is the formal q24 causal re-adjudication using independently compiled free-generation coordinates.",
        "next_authorization": "Continue cross-model exact semantic panel and full-coordinate visualization; no further causal tuning is authorized.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
