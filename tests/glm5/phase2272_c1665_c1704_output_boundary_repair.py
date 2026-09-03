#!/usr/bin/env python3
"""Repair only model-specific assistant boundaries, then rerun cross-architecture gates."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT_OUT = RESULT / "phase2265_c1433_c1468_independent_bilingual_contract"
Q4_OUT = RESULT / "phase2267_c1505_c1540_coordinate_model_tournament"
PRIOR_OUT = RESULT / "phase2271_c1625_c1664_cross_architecture_topology"
OUT = RESULT / "phase2272_c1665_c1704_output_boundary_repair"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2163_c629_model_specific_worker as model_worker  # noqa: E402
import phase2265_c1433_c1468_independent_bilingual_contract as contract  # noqa: E402
import phase2269_c1577_c1600_qwen14_relative_topology_replication as topology  # noqa: E402
import phase2271_c1625_c1664_cross_architecture_topology as cross  # noqa: E402


PHASE = 2272
CAMPAIGN = "C1665-C1704"
MODEL_NAMES = ("glm4", "deepseek7b")
FAMILIES = ("location_state", "property_state", "patient_binding",
            "temporal_order", "comparison_order")
OFFSETS = (-2, -1, 0, 1, 2)
MAX_NEW_TOKENS = 8


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def boundary_suffix(model_name: str, tokenizer) -> tuple[list[int], str]:
    literal = "\n" if model_name == "glm4" else "</think>\n"
    ids = tokenizer.encode(literal, add_special_tokens=False)
    if not ids:
        raise RuntimeError(("empty_boundary_suffix", model_name, literal))
    return ids, tokenizer.decode(ids, skip_special_tokens=False)


def repair_compiled(model_name: str, tokenizer, rows: list[dict]) -> tuple[list[dict], dict]:
    suffix, decoded = boundary_suffix(model_name, tokenizer)
    repaired = []
    before_tail = None
    for row in rows:
        item = dict(row)
        if before_tail is None:
            before_tail = tokenizer.decode(item["free_prompt_ids"][-12:], skip_special_tokens=False)
        item["prompt_ids"] = [*item["prompt_ids"], *suffix]
        item["free_prompt_ids"] = [*item["free_prompt_ids"], *suffix]
        repaired.append(item)
    audit = {
        "model": model_name, "suffix_ids": suffix, "suffix_decoded": decoded,
        "before_tail": before_tail,
        "after_tail": tokenizer.decode(repaired[0]["free_prompt_ids"][-20:], skip_special_tokens=False),
        "role_positions_unchanged": True,
        "repair_scope": "assistant_output_boundary_only",
    }
    return repaired, audit


def configure_paths(model_dir: Path) -> None:
    topology.FAMILIES = FAMILIES
    topology.OUT = model_dir
    topology.FIELD = model_dir / "raw/relative_window_field.float16.npy"
    topology.INDEX = model_dir / "raw/field_index.jsonl"
    topology.PROGRESS = model_dir / "raw/capture_progress.json"
    cross.FAMILIES = FAMILIES
    cross.MAX_NEW_TOKENS = MAX_NEW_TOKENS


def run_model(model_name: str, raw: list[dict], q4_decisions: dict[str, dict]) -> dict:
    model_dir = OUT / model_name
    final = model_dir / "analysis/final.json"
    if final.exists():
        return load(final)
    configure_paths(model_dir)
    model = None
    hidden_dim = 0
    try:
        model, tokenizer, device, placement, loader = model_worker.load_model(model_name)
        compiled = contract.compile_rows(tokenizer, raw)
        compiled, boundary = repair_compiled(model_name, tokenizer, compiled)
        write_rows(model_dir / "material/compiled_boundary_repaired.jsonl", compiled)
        save(model_dir / "protocol/output_boundary.json", boundary)
        candidate_path = model_dir / "behavior/candidate.jsonl"
        generation_path = model_dir / "behavior/generation.jsonl"
        if candidate_path.exists():
            candidates = read_rows(candidate_path)
        else:
            candidates = contract.legacy.parent.model_base.behavior_base.batch_behavior(
                model, device, compiled, batch_size=8)
            write_rows(candidate_path, candidates)
        generated = cross.generation_resumable(
            model, tokenizer, device, compiled, generation_path, model_name)
        topology.FAMILIES = FAMILIES
        ledger = topology.behavior(compiled, candidates, generated)
        layers = len(model.model.layers)
        hidden_dim = int(model.model.embed_tokens.weight.shape[1])
        settings = {}
        for family in FAMILIES:
            q4_checkpoint = int(q4_decisions[family]["checkpoint"])
            center = topology.relative_center(q4_checkpoint, layers)
            settings[family] = {
                "q4_checkpoint": q4_checkpoint, "q4_role": q4_decisions[family]["role"],
                "q14_layers": layers, "q14_center": center,
                "q14_window": [max(1, min(layers, center + offset)) for offset in OFFSETS],
            }
        save(model_dir / "protocol/frozen_relative_windows.json", settings)
        observed = [row for row in compiled if row["family"] in ledger["qualified_families"]]
        if len(ledger["qualified_families"]) >= 2:
            field_info = topology.capture(model, device, observed, settings)
        else:
            field_info = {"ran": False, "reason": "fewer_than_two_dual_behavior_qualified_families",
                          "qualified_families": ledger["qualified_families"]}
    finally:
        model_worker.release_model(model_name, model)
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    decisions = []
    atlas = np.empty((0, hidden_dim), np.float16)
    if field_info.get("ran"):
        field = np.load(topology.FIELD, mmap_mode="r")
        index = read_rows(topology.INDEX)
        try:
            decisions, atlas_path, labels_path, atlas = topology.replicate(
                field, index, ledger["qualified_families"], settings)
        finally:
            topology.close_mmap(field)
        atlas_info = {"path": str(atlas_path.relative_to(ROOT)),
                      "rows": str(labels_path.relative_to(ROOT)),
                      "shape": list(atlas.shape), "all_coordinates": True}
    else:
        atlas_info = {"path": None, "rows": None, "shape": [0, hidden_dim], "all_coordinates": False}
    replicated = [row["family"] for row in decisions if row["replicated"]]
    checks = {
        "material_rows": len(raw) == 1280,
        "behavior_complete": ledger["rows"] == len(raw),
        "boundary_repair_frozen": boundary["repair_scope"] == "assistant_output_boundary_only",
        "field_follows_behavior": (not field_info.get("ran") or
                                   field_info["shape"][0] == len(observed)),
        "decisions_follow_behavior": (not field_info.get("ran") or
                                      len(decisions) == len(ledger["qualified_families"])),
        "all_coordinates_if_ran": (not field_info.get("ran") or
                                   field_info["shape"][-1] == hidden_dim == atlas.shape[-1]),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "model": model_name,
        "status": "closed" if all(checks.values()) else "audit_failed",
        "timestamp": datetime.now().astimezone().isoformat(), "placement": placement,
        "loader": loader, "boundary": boundary, "behavior": ledger,
        "settings": settings, "field": field_info, "decisions": decisions,
        "replicated_families": replicated, "atlas": atlas_info,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (f"{model_name}: output-boundary repair yielded "
                              f"{len(ledger['qualified_families'])}/5 dual-behavior-qualified families "
                              f"and {len(replicated)}/5 lockbox topology replications."),
    }
    save(final, result)
    print(json.dumps({key: value for key, value in result.items() if key != "decisions"},
                     ensure_ascii=True, indent=2), flush=True)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 跨架构助手输出边界修复与全坐标重裁（C1665-C1704） [{stamp}]

**测试原理与用例。** Phase2271 揭示两个模型特异接口问题：GLM4 自由生成总体 0.946875，但答案前固定先生成换行，直接候选拼接只有 0.68125；DS7B 的 chat template 固定以 `<think>` 开始，24-token 输出大多尚未抵达代码答案。新阶段在任何模型运行前冻结唯一修复：GLM4 在助手边界预填一个换行，DS7B 预填 `</think>` 关闭模板强制打开的推理段。材料、五个家族、1280 行、分区、0.75 双行为门、相对层深窗口、全部控制和锁箱门槛均不改变。模型继续严格顺序运行；边界修复只追加 token，不移动原有六角色位置。

**数学公式。** 设原助手边界 token 序列为 $P_m$，冻结后为：

$$
P'_{{\mathrm{{GLM}}}}=P_{{\mathrm{{GLM}}}}\mathbin\Vert \texttt{{\textbackslash n}},
\qquad
P'_{{\mathrm{{DS}}}}=P_{{\mathrm{{DS}}}}\mathbin\Vert \texttt{{</think>\textbackslash n}}.
$$

其余逐坐标预测保持不变：

$$
R_{{i,j}}=H^1_{{i,j}}-H^0_{{i,j}},\qquad
\widehat R_{{i,j}}=a_{{f,j}}H^0_{{i,j}}+b_{{f,j}}.
$$

**结果汇总。** GLM4 为 `{json.dumps(result['models']['glm4'], ensure_ascii=False)}`；DS7B 为 `{json.dumps(result['models']['deepseek7b'], ensure_ascii=False)}`；相对 Phase2271 的行为变化为 `{json.dumps(result['behavior_changes'], ensure_ascii=False)}`；工程检查为 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析与理论进展。** `{result['strict_conclusion']}` 本阶段回答的是测量接口是否阻断跨架构资格，而不是通过提示工程提高模型能力。只有边界修复后双行为通过的家族才读取 HiddenState；仍不合格的模型—家族没有内部机制结论。若拓扑通过，它只表明模型本地逐坐标基态预测结构可以跨架构出现，不代表相同坐标、相同权重或因果齿轮；若失败，则是该冻结模型—构式—边界合同中的不迁移。

**问题、硬伤与瓶颈。** 边界修复由 Phase2271 输出揭盲后提出，因此它是新的前瞻合同，不是对 Phase2271 的重算；两种模型使用不同物理后缀，不能据此比较绝对行为难度；预填 `</think>` 改变 DS7B 的默认生成制度；代码答案仍是受控接口；五家族、人类盲评缺失、CPU/GPU 分层和 float16 落盘仍限制外推。

**结论与下一步。** 脚本：`tests/glm5/phase2272_c1665_c1704_output_boundary_repair.py`；结果：`tests/glm5/result/phase2272_c1665_c1704_output_boundary_repair`。下一步发布行为边界和所有获得锁箱资格的模型本地全坐标图；无资格或不显示的原始场按哈希账本清理。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    q4 = load(Q4_OUT / "analysis/final.json")
    prior = load(PRIOR_OUT / "analysis/final.json")
    q4_decisions = {row["family"]: row for row in q4["decisions"]}
    raw = [row for row in read_rows(CONTRACT_OUT / "material/independent_bilingual_cases.jsonl")
           if row["family"] in FAMILIES]
    prereg = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "phase": PHASE,
        "models_in_order": list(MODEL_NAMES), "families": list(FAMILIES), "rows": len(raw),
        "repairs": {"glm4": "append newline after assistant boundary",
                    "deepseek7b": "append </think> newline after forced think opener"},
        "unchanged": ["materials", "partitions", "roles", "behavior_gate", "prediction_gates", "lockbox_order"],
        "max_new_tokens": MAX_NEW_TOKENS, "frozen_before_models": True,
    }
    save(OUT / "protocol/preregistration.json", prereg)
    models, sequence = {}, []
    for model_name in MODEL_NAMES:
        started = datetime.now().astimezone().isoformat()
        models[model_name] = run_model(model_name, raw, q4_decisions)
        sequence.append({"model": model_name, "started": started,
                         "finished": datetime.now().astimezone().isoformat(),
                         "status": models[model_name]["status"]})
    behavior_changes = {}
    for model_name in MODEL_NAMES:
        old, new = prior["models"][model_name]["behavior"], models[model_name]["behavior"]
        behavior_changes[model_name] = {
            "candidate_accuracy_before": old["candidate_accuracy"],
            "candidate_accuracy_after": new["candidate_accuracy"],
            "generation_accuracy_before": old["generation_accuracy"],
            "generation_accuracy_after": new["generation_accuracy"],
            "parsed_fraction_before": old["parsed_generation_fraction"],
            "parsed_fraction_after": new["parsed_generation_fraction"],
            "qualified_families_before": old["qualified_families"],
            "qualified_families_after": new["qualified_families"],
        }
    checks = {
        "material_fixed": len(raw) == 1280,
        "models_sequential": [row["model"] for row in sequence] == list(MODEL_NAMES),
        "both_models_complete": all(row["all_checks_passed"] for row in models.values()),
        "repairs_only_at_boundary": all(row["boundary"]["repair_scope"] == "assistant_output_boundary_only"
                                        for row in models.values()),
    }
    replicated = {name: row["replicated_families"] for name, row in models.items()}
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "status": "closed" if all(checks.values()) else "audit_failed",
        "timestamp": datetime.now().astimezone().isoformat(), "preregistration": prereg,
        "models": models, "sequence": sequence, "behavior_changes": behavior_changes,
        "replicated_families": replicated, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": ("Boundary-calibrated cross-architecture result: "
                              + "; ".join(f"{name} behavior {len(models[name]['behavior']['qualified_families'])}/5, "
                                                  f"topology {len(rows)}/5"
                                          for name, rows in replicated.items())
                              + ". Interpretation remains model-local and non-causal."),
        "next_authorization": "Publish behavior and qualifying coordinate atlases; clean undisplayed raw fields.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps({key: value for key, value in result.items() if key != "models"},
                     ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
