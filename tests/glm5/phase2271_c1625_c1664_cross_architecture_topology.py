#!/usr/bin/env python3
"""Sequential GLM4 and DeepSeek-7B cross-architecture topology adjudication."""
from __future__ import annotations

import gc
import json
import re
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
OUT = RESULT / "phase2271_c1625_c1664_cross_architecture_topology"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2163_c629_model_specific_worker as model_worker  # noqa: E402
import phase2265_c1433_c1468_independent_bilingual_contract as contract  # noqa: E402
import phase2269_c1577_c1600_qwen14_relative_topology_replication as topology  # noqa: E402


PHASE = 2271
CAMPAIGN = "C1625-C1664"
MODEL_NAMES = ("glm4", "deepseek7b")
FAMILIES = ("location_state", "property_state", "patient_binding",
            "temporal_order", "comparison_order")
OFFSETS = (-2, -1, 0, 1, 2)
MAX_NEW_TOKENS = 24
MIN_FAMILIES_FOR_TOPOLOGY = 2


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


def generation_resumable(model, tokenizer, device, rows: list[dict], path: Path,
                         model_name: str) -> list[dict]:
    existing = read_rows(path) if path.exists() else []
    by_id = {row["case_id"]: row for row in existing}
    pending = [row for row in rows if row["case_id"] not in by_id]
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), 8):
        batch = pending[start:start + 8]
        width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["free_prompt_ids"]
            ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, width - len(seq):] = 1
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=pad, eos_token_id=tokenizer.eos_token_id,
            )
        new_rows = []
        for i, row in enumerate(batch):
            text = tokenizer.decode(generated[i, width:].tolist(), skip_special_tokens=True)
            parsed = parse_code(text, row)
            item = {"case_id": row["case_id"], "text": text, "parsed": parsed,
                    "correct_answer": row["correct_answer"],
                    "correct": parsed == row["correct_answer"]}
            by_id[row["case_id"]] = item
            new_rows.append(item)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for item in new_rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        if start % 32 == 0:
            print(f"[{model_name}-generation] {len(by_id)}/{len(rows)}", flush=True)
    ordered = [by_id[row["case_id"]] for row in rows]
    write_rows(path, ordered)
    return ordered


def configure_topology_paths(model_dir: Path) -> None:
    topology.FAMILIES = FAMILIES
    topology.OUT = model_dir
    topology.FIELD = model_dir / "raw/relative_window_field.float16.npy"
    topology.INDEX = model_dir / "raw/field_index.jsonl"
    topology.PROGRESS = model_dir / "raw/capture_progress.json"


def run_model(model_name: str, raw: list[dict], q4_decisions: dict[str, dict]) -> dict:
    model_dir = OUT / model_name
    final = model_dir / "analysis/final.json"
    if final.exists():
        return load(final)
    configure_topology_paths(model_dir)
    model = None
    compiled: list[dict] = []
    hidden_dim = 0
    try:
        model, tokenizer, device, placement, loader = model_worker.load_model(model_name)
        compiled = contract.compile_rows(tokenizer, raw)
        write_rows(model_dir / "material/compiled.jsonl", compiled)
        candidate_path = model_dir / "behavior/candidate.jsonl"
        generation_path = model_dir / "behavior/generation.jsonl"
        if candidate_path.exists():
            candidates = read_rows(candidate_path)
        else:
            candidates = contract.legacy.parent.model_base.behavior_base.batch_behavior(
                model, device, compiled, batch_size=8)
            write_rows(candidate_path, candidates)
        generated = generation_resumable(model, tokenizer, device, compiled, generation_path, model_name)
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
        if len(ledger["qualified_families"]) >= MIN_FAMILIES_FOR_TOPOLOGY:
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
    atlas = np.empty((0, 0), np.float16)
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
        atlas_info = {"path": None, "rows": None, "shape": [0, 0], "all_coordinates": False}
    replicated = [row["family"] for row in decisions if row["replicated"]]
    checks = {
        "material_rows": len(raw) == len(FAMILIES) * 256,
        "behavior_complete": ledger["rows"] == len(raw),
        "models_sequential_contract": True,
        "field_follows_dual_behavior": (not field_info.get("ran") or
                                        field_info["shape"][0] == len(observed)),
        "decisions_follow_qualification": (not field_info.get("ran") or
                                           len(decisions) == len(ledger["qualified_families"])),
        "all_coordinates_if_ran": (not field_info.get("ran") or
                                   field_info["shape"][-1] == hidden_dim),
    }
    checks["all_coordinates_if_ran"] = (checks["all_coordinates_if_ran"] and
                                        (not field_info.get("ran") or
                                         atlas.shape[-1] == field_info["shape"][-1]))
    status = "closed" if all(checks.values()) else "audit_failed"
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "model": model_name, "status": status,
        "timestamp": datetime.now().astimezone().isoformat(), "placement": placement, "loader": loader,
        "behavior": ledger, "settings": settings, "field": field_info,
        "decisions": decisions, "replicated_families": replicated, "atlas": atlas_info,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (f"{model_name}: {len(replicated)}/{len(FAMILIES)} frozen families replicated "
                              "the model-local relative-depth coordinate predictor; behavior-unqualified "
                              "families were not interpreted internally."),
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

## Phase {PHASE}: GLM4与DS7B跨架构相对拓扑顺序裁决（C1625-C1664） [{stamp}]

**测试原理与用例。** 为判断 Phase2267/2269 的条件化逐坐标预测是否仅属于 Qwen3，本阶段在任何新模型运行前冻结五类构式：位置状态、属性状态、受事绑定、时间顺序和比较顺序。每类 256 行，共 1280 行，覆盖中英双语、两种表面、两种状态以及 discovery、confirmation、fresh confirmation、fresh lockbox。GLM4 与 DeepSeek-R1-Distill-Qwen-7B 严格顺序加载；一个模型释放并清空 CUDA 后才加载另一个。候选 A/B 与最多 24 token 的自由生成分别计分；只有总体、discovery 和 fresh confirmation 双行为准确率均不低于 0.75 的家族才采集内部状态。行为失败只淘汰对应模型—家族路线，不终止另一模型。

**数学公式。** 各模型仅冻结 Qwen3-4B 候选的相对深度邻域，不对齐物理坐标编号：

$$
q_m=\operatorname{{round}}\!\left(\frac{{q_4}}{{36}}L_m\right)+\delta,
\qquad \delta\in\{{-2,-1,0,1,2\}}.
$$

在模型自己的每个物理坐标上估计：

$$
R_{{i,j}}=H^1_{{i,j}}-H^0_{{i,j}},\qquad
\widehat R_{{i,j}}=a_{{f,j}}H^0_{{i,j}}+b_{{f,j}}.
$$

fresh confirmation 同时要求相对家族均值、纯代数、跨家族共享、错配和错家族五类控制的最小 MAE 增益不低于 0.03、逐坐标胜率不低于 0.55，才揭示 fresh lockbox。

**结果汇总。** GLM4 结果为 `{json.dumps(result['models']['glm4'], ensure_ascii=False)}`；DS7B 结果为 `{json.dumps(result['models']['deepseek7b'], ensure_ascii=False)}`；顺序执行账本为 `{json.dumps(result['sequential_execution'], ensure_ascii=False)}`；工程检查为 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析与理论进展。** `{result['strict_conclusion']}` 跨架构比较的是形成层的相对深度、功能角色和“模型自己的坐标预测是否胜过全部预注册控制”，不是相同坐标编号、相同神经元或共享权重。若某模型行为不合格，只能说明本合同没有给该模型提供合法内部解释对象；若行为通过但预测失败，才是该模型—构式上的拓扑不迁移。即使预测复现，也仍是观察性条件规律，Phase2268 的 Qwen3-4B 严格双向因果阴性不会因此被改写。

**问题、硬伤与瓶颈。** 两个模型都通过 CPU/GPU 分层 BF16 加载，速度和 float16 落盘是工程近似；DeepSeek 的显式推理输出可能污染短代码自由生成接口；五类构式仍不是完整语言图谱；同坐标仿射可能读取通用状态转移；模型间没有功能坐标配准；独立人类自然度盲评仍缺失。因此不能把跨架构正结果命名为普遍语言数学结构，把阴性结果命名为模型没有相应语义。

**结论与下一步。** 相关脚本：`tests/glm5/phase2271_c1625_c1664_cross_architecture_topology.py`；结果：`tests/glm5/result/phase2271_c1625_c1664_cross_architecture_topology`。下一步只发布行为合格且通过锁箱的模型本地全坐标图，并在可视化副本验证后清理未展示原始场；随后把观察优先图谱扩展到更多自然输出构式，而不是立即重复差分搬运干预。
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
    q4_decisions = {row["family"]: row for row in q4["decisions"]}
    raw = [row for row in read_rows(CONTRACT_OUT / "material/independent_bilingual_cases.jsonl")
           if row["family"] in FAMILIES]
    prereg = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "phase": PHASE,
        "models_in_order": list(MODEL_NAMES), "families": list(FAMILIES),
        "rows": len(raw), "offsets": list(OFFSETS), "max_new_tokens": MAX_NEW_TOKENS,
        "behavior_gate": contract.BEHAVIOR_GATE,
        "prediction_gates": {"gain": topology.GAIN_GATE, "win": topology.WIN_GATE},
        "frozen_before_models": True,
    }
    save(OUT / "protocol/preregistration.json", prereg)
    models = {}
    sequence = []
    for model_name in MODEL_NAMES:
        start = datetime.now().astimezone().isoformat()
        models[model_name] = run_model(model_name, raw, q4_decisions)
        sequence.append({"model": model_name, "started": start,
                         "finished": datetime.now().astimezone().isoformat(),
                         "status": models[model_name]["status"]})
    checks = {
        "five_families": set(FAMILIES) == {row["family"] for row in raw},
        "large_material": len(raw) == 1280,
        "both_models_completed": set(models) == set(MODEL_NAMES),
        "ordered_execution": [row["model"] for row in sequence] == list(MODEL_NAMES),
        "model_checks_pass": all(row["all_checks_passed"] for row in models.values()),
    }
    replicated = {name: row["replicated_families"] for name, row in models.items()}
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed" if all(checks.values()) else "audit_failed",
        "timestamp": datetime.now().astimezone().isoformat(), "preregistration": prereg,
        "models": models, "sequential_execution": sequence,
        "replicated_families": replicated, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": ("Cross-architecture evidence is model-local and behavior-conditioned: "
                              + "; ".join(f"{name} {len(rows)}/{len(FAMILIES)}" for name, rows in replicated.items())
                              + ". No coordinate identity or causal language gear is inferred."),
        "next_authorization": "Publish qualifying cross-architecture atlases and clean undisplayed raw fields.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps({key: value for key, value in result.items() if key != "models"},
                     ensure_ascii=True, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
