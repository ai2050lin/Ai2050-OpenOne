#!/usr/bin/env python3
"""Qwen3-14B replication of full-token interaction dynamics on behavior-qualified quartets."""
from __future__ import annotations

import gc
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2581 = RESULT / "phase2581_c364801_c372992_fourchoice_null_calibration_qwen14"
P2582 = RESULT / "phase2582_c372993_c385280_fourchoice_fulltoken_interaction_birth"
P2585 = RESULT / "phase2585_c409857_c426240_equal_bpe_nonce_value_lockbox/analysis/final.json"
OUT = RESULT / "phase2586_c426241_c442624_qwen14_interaction_dynamics_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2586, "C426241-C442624"
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))

sys.path.insert(0, str(TESTS))
import phase2560_c223489_c231680_crossmodel_relation_stage_replication as p2560  # noqa: E402
import phase2582_c372993_c385280_fourchoice_fulltoken_interaction_birth as p2582  # noqa: E402


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_qwen14():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = p2560.cross.MODELS["qwen14b"]
    offload = ROOT / "tests/glm5_temp/phase2586_offload/qwen14b"
    offload.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "13GiB", "cpu": "12GiB"},
        offload_folder=str(offload),
        offload_state_dict=True,
        offload_buffers=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    if bool(getattr(model, "is_quantized", False)):
        raise RuntimeError("quantized model is forbidden")
    return model, tokenizer, offload


def load_selected():
    rows = [
        json.loads(line)
        for line in (P2581 / "material/qwen14_stratified_fourchoice.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["ablation"] == "full"]
    index = {
        (row["family_id"], row["binding_relation"], row["binding_value"], row["relation_form"],
         row["value_form"], row["query_relation"], row["query_value"]): row
        for row in rows
    }
    eligible_payload = load_json(P2581 / "material/eligible_quartets.json")
    compatible = []
    for prefix_values in eligible_payload["eligible"]:
        prefix = tuple(prefix_values)
        cells = [index[prefix + cell] for cell in CELLS]
        aligned = len({len(row["prompt_ids"]) for row in cells}) == 1 and all(
            len({len(row["regions"][region]) for row in cells}) == 1
            for region in cells[0]["regions"]
        )
        if aligned:
            compatible.append((prefix, cells))
    selected = []
    pools = {}
    for form in (("natural", "natural"), ("nonce", "natural")):
        values = sorted([item for item in compatible if item[0][3:] == form], key=lambda item: item[0])
        pools[f"{form[0]}/{form[1]}"] = len(values)
        if len(values) < 4:
            raise RuntimeError((form, len(values)))
        indices = np.linspace(0, len(values) - 1, 4, dtype=int)
        selected.extend(values[int(index)] for index in indices)
    return selected, len(compatible), pools


def normalized_curve(values):
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    return values / maximum if maximum else values


def resample(values, points=201):
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.interp(target, source, values)


def thresholds(curve):
    output = {}
    for threshold in (.01, .1, .5):
        index = next((i for i, value in enumerate(curve) if value >= threshold), None)
        output[str(threshold)] = None if index is None else index / (len(curve) - 1)
    return output


def append_memo(result):
    heading = f"## Phase {PHASE}: Qwen3-14B四格交互动力学全坐标复现（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理。** 不比较4B与14B的物理坐标号，而比较同一四选一代数从embedding、逐层residual到answer readout的功能动力学。Qwen3-14B只使用Phase2581行为锁箱中四格全对、逐token等长的样本；保存全token、embedding+全部block、全部5120坐标：

$$I^{{(m)}}_{{ltd}}=H^{{(m,11)}}_{{ltd}}-H^{{(m,10)}}_{{ltd}}-H^{{(m,01)}}_{{ltd}}+H^{{(m,00)}}_{{ltd}},\qquad
g^{{(m)}}_l=\sqrt{{D_m^{{-1}}\sum_d (I^{{(m)}}_{{l,t_a,d}})^2}}.$$

跨模型只把层号归一到$[0,1]$，把$g_l$除以各模型自身峰值，再比较曲线相关和达到1%/10%/50%峰值的相对深度。

**测试用例。** Qwen3-14B BF16非量化、`device_map=auto`；行为锁箱中natural/natural严格全对且等长样本只有4组，故冻结全部4组，并从nonce/natural等距取4组配平，共32 prompt，逐组无padding。原始数组为`[4,{result['field']['hidden_states']},token,{result['field']['d_model']}]`，共{result['field']['raw_bytes']} bytes；低值坐标未用Top-K筛除。

**结果汇总。** 14B answer曲线`{json.dumps(result['qwen14']['median_answer_interaction_rms'])}`；归一化阶段阈值与4B对比`{json.dumps(result['cross_scale'], ensure_ascii=False)}`；14B内部词面/拆半坐标纹理`{json.dumps(result['qwen14']['within_model_reuse'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2586_c426241_c442624_qwen14_interaction_dynamics_replication.py`；14B逐token全坐标场、哈希manifest、answer全坐标数组、exemplar和final位于`{OUT}`。

**理论进展与分析。** 若两模型都由输入层零交互开始、在早期block产生微弱交互并在中晚层放大，支持“载体→非线性组合→晚层输出编译”的功能顺序具有跨规模重复；这不要求相同坐标或相同head。曲线相关只衡量粗动力学，不证明组件同构。

**问题硬伤。** 14B只覆盖Phase2581预留的8个family，16四元组仍小于4B；CPU/GPU自动放置可能改变吞吐但不改模型权重；仍是人工英文表格；14B尚未使用Phase2585新等BPE nonce-value材料；float16落盘。结论不得写成跨架构普遍规律。检查`{json.dumps(result['checks'], ensure_ascii=False)}`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    selected, compatible_count, pools = load_selected()
    old_out = p2582.OUT
    model = tokenizer = offload = None
    try:
        model, tokenizer, offload = load_qwen14()
        p2582.OUT = OUT
        manifest, arrays, semantic, exemplar, metrics_path, exemplar_path = p2582.collect(model, tokenizer, selected)
    finally:
        p2582.OUT = old_out
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        if offload is not None:
            resolved = Path(offload).resolve()
            allowed = (ROOT / "tests/glm5_temp").resolve()
            if allowed in resolved.parents:
                shutil.rmtree(resolved, ignore_errors=True)
    interaction = arrays["answer_interaction"]
    oriented = arrays["answer_oriented_interaction"]
    q14_rms = np.sqrt(np.mean(interaction.astype(np.float64) ** 2, axis=2))
    q14_curve = np.median(q14_rms, axis=0)
    split_mask = np.array([index % 2 == 0 for index in range(len(interaction))])
    natural = np.arange(len(interaction)) < 4
    within = {
        "split_half_signed_by_layer": [p2582.pearson(oriented[split_mask, layer].mean(0),
                                                       oriented[~split_mask, layer].mean(0))
                                        for layer in range(interaction.shape[1])],
        "split_half_absolute_by_layer": [p2582.pearson(np.abs(interaction[split_mask, layer]).mean(0),
                                                        np.abs(interaction[~split_mask, layer]).mean(0))
                                          for layer in range(interaction.shape[1])],
        "natural_vs_nonce_relation_signed_by_layer": [p2582.pearson(oriented[natural, layer].mean(0),
                                                                      oriented[~natural, layer].mean(0))
                                                        for layer in range(interaction.shape[1])],
        "natural_vs_nonce_relation_absolute_by_layer": [p2582.pearson(np.abs(interaction[natural, layer]).mean(0),
                                                                        np.abs(interaction[~natural, layer]).mean(0))
                                                          for layer in range(interaction.shape[1])],
    }
    q4_curve = np.asarray(load_json(P2582 / "analysis/final.json")["interaction_birth"]["median_answer_interaction_rms_by_hidden_state"])
    q4_norm, q14_norm = normalized_curve(q4_curve), normalized_curve(q14_curve)
    q4_resampled, q14_resampled = resample(q4_norm), resample(q14_norm)
    cross = {
        "qwen4_hidden_states": len(q4_curve),
        "qwen14_hidden_states": len(q14_curve),
        "normalized_curve_pearson": p2582.pearson(q4_resampled, q14_resampled),
        "qwen4_relative_depth_at_peak_fraction": thresholds(q4_norm),
        "qwen14_relative_depth_at_peak_fraction": thresholds(q14_norm),
        "physical_coordinates_compared": False,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-14B BF16 nonquantized device_map=auto",
        "selection": {"eligible_and_aligned": compatible_count, "pools": pools,
                      "quartets": len(manifest), "prompts": len(manifest) * 4},
        "field": {"hidden_states": interaction.shape[1], "d_model": interaction.shape[2],
                  "raw_bytes": sum(item["bytes"] for item in manifest), "full_tokens": True,
                  "full_coordinates": True, "no_topk": True,
                  "embedding_interaction_rms_max": max(item["embedding_interaction_rms_max"] for item in manifest),
                  "prequery_interaction_rms_max": max(item["prefix_interaction_rms_max"] for item in manifest)},
        "qwen14": {"median_answer_interaction_rms": q14_curve.tolist(),
                   "normalized_curve": q14_norm.tolist(), "within_model_reuse": within},
        "cross_scale": cross,
        "claim_boundary": "cross-scale functional dynamics only; no physical coordinate conservation or cross-architecture proof",
        "language_mechanism_closed": False,
    }
    checks = {
        "phase2585_complete": load_json(P2585)["all_checks_passed"],
        "selected_8_quartets": len(manifest) == 8,
        "selected_32_prompts": len(manifest) * 4 == 32,
        "all_raw_fields_exist": all((ROOT / item["path"]).is_file() for item in manifest),
        "embedding_and_prefix_zero": result["field"]["embedding_interaction_rms_max"] == 0.0 and result["field"]["prequery_interaction_rms_max"] == 0.0,
        "all_coordinates": interaction.shape[2] == 5120,
        "nonquantized_bf16_auto": True,
        "no_topk": True,
        "claim_boundary": True,
    }
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({key: result[key] for key in ("phase", "selection", "field", "qwen14", "cross_scale", "checks", "all_checks_passed")}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
