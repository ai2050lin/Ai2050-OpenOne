#!/usr/bin/env python3
"""Twenty-family fixed A/B full-coordinate atlas with paired option-swap controls."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2330 = RESULT / "phase2330_c6081_c6200_language_family_atlas_contract"
OUT = RESULT / "phase2340_c7601_c7840_twenty_family_fixed_interface_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
SOURCE = P2330 / "material/typed_language_family_atlas.jsonl"
MATERIAL = OUT / "material/twenty_family_fixed_ab.jsonl"
STATES = OUT / "raw/boundary_all_checkpoints.float16.npy"
FINAL_CONTRIB = OUT / "raw/final_coordinate_contribution.float32.npy"
DECISIONS = OUT / "raw/decisions.float32.npy"
GAMMA = OUT / "raw/final_norm_gamma.float32.npy"
AB_WEIGHT = OUT / "raw/a_minus_b_weight.float32.npy"
PROGRESS = OUT / "raw/progress.json"
TRAJECTORY = OUT / "derived/layerwise_coordinate_contribution.float32.npy"
TRAJECTORY_PROGRESS = OUT / "derived/trajectory_progress.json"
PHASE = 2340
CAMPAIGN = "C7601-C7840"
PARTITIONS = ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox")
TRAIN_PARTITIONS = ("discovery", "confirmation")
TEST_PARTITIONS = ("fresh_confirmation", "fresh_lockbox")
EPS = 1e-12

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2330_c6081_c6200_language_family_atlas_contract as contract  # noqa: E402
import phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls as layer_control  # noqa: E402

FAMILIES = tuple(contract.FAMILIES)
MACROTYPES = tuple(sorted(set(contract.MACROTYPE.values())))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return io.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    io.write_rows(path, rows)


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def prompt_text(source: dict, choice_a: str, choice_b: str) -> str:
    if source["language"] == "en":
        return (f'{source["future_prompt"]}\nOption A: {choice_a}\nOption B: {choice_b}\n'
                "Select the factually correct continuation. Reply with exactly A or B.\nAnswer:")
    return (f'{source["future_prompt"]}\n选项A：{choice_a}\n选项B：{choice_b}\n'
            "选择符合事实的续写，只回答A或B。\n答案：")


def compile_material(tokenizer) -> tuple[list[dict], dict]:
    code_ids = {code: tokenizer.encode(" " + code, add_special_tokens=False) for code in ("A", "B")}
    if any(len(ids) != 1 for ids in code_ids.values()):
        raise RuntimeError(("codes_not_single_token", code_ids))
    sources = read_rows(SOURCE)
    output = []
    for condition in ("original", "option_swapped"):
        for source in sources:
            language_index = 0 if source["language"] == "en" else 1
            surface_index = 0 if source["surface"] == "narrative" else 1
            target_is_a_original = (source["unit"] + source["state"] + language_index + surface_index) % 2 == 0
            target_is_a = target_is_a_original if condition == "original" else not target_is_a_original
            target_code, wrong_code = (("A", "B") if target_is_a else ("B", "A"))
            choice_a = source["future_target_text"] if target_is_a else source["future_wrong_text"]
            choice_b = source["future_wrong_text"] if target_is_a else source["future_target_text"]
            prompt = prompt_text(source, choice_a, choice_b)
            prompt_ids = [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]
            output.append({
                **source,
                "case_id": source["case_id"].replace("c6081-", "c7601-") + ("-swap" if condition == "option_swapped" else "-orig"),
                "source_case_id": source["case_id"], "condition": condition,
                "model_design_index": len(output),
                "natural_target_text": source["future_target_text"], "natural_wrong_text": source["future_wrong_text"],
                "choice_a_text": choice_a, "choice_b_text": choice_b,
                "target_code": target_code, "wrong_code": wrong_code,
                "target_code_id": int(code_ids[target_code][0]), "wrong_code_id": int(code_ids[wrong_code][0]),
                "future_prompt": prompt, "future_prompt_ids": prompt_ids,
                "future_target_text": " " + target_code, "future_wrong_text": " " + wrong_code,
                "future_target_ids": [int(code_ids[target_code][0])], "future_wrong_ids": [int(code_ids[wrong_code][0])],
                "identity_target": target_code, "identity_wrong": wrong_code,
                "boundary_position": len(prompt_ids) - 1, "cue_id": "twenty_family_fixed_ab_" + condition,
            })
    balance = {}
    for condition in ("original", "option_swapped"):
        balance[condition] = {}
        for family in FAMILIES:
            balance[condition][family] = {}
            for partition in PARTITIONS:
                cell = [r for r in output if r["condition"] == condition and r["family"] == family and r["partition"] == partition]
                balance[condition][family][partition] = {code: sum(r["target_code"] == code for r in cell) for code in ("A", "B")}
    audit = {
        "rows": len(output), "source_rows": len(sources), "families": len(FAMILIES), "macrotypes": len(MACROTYPES),
        "code_ids": code_ids, "balance": balance,
        "balanced_every_condition_family_partition": all(
            cell["A"] == cell["B"] for condition in balance.values() for family in condition.values() for cell in family.values()
        ),
        "paired_semantics_preserved": all(
            output[i]["natural_target_text"] == output[i + len(sources)]["natural_target_text"] and
            output[i]["target_code"] != output[i + len(sources)]["target_code"]
            for i in range(len(sources))
        ),
    }
    return output, audit


def collect(model, device, rows: list[dict], batch_size: int = 12) -> dict:
    module_list = modules(model)
    dimension = int(model.config.hidden_size)
    shape = (len(rows), len(module_list), dimension)
    if all(path.exists() for path in (STATES, FINAL_CONTRIB, DECISIONS, PROGRESS)):
        completed = int(json.loads(PROGRESS.read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(STATES, mode="r+")
        final = np.lib.format.open_memmap(FINAL_CONTRIB, mode="r+")
        decisions = np.lib.format.open_memmap(DECISIONS, mode="r+")
    else:
        completed = 0
        STATES.parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(STATES, mode="w+", dtype=np.float16, shape=shape)
        final = np.lib.format.open_memmap(FINAL_CONTRIB, mode="w+", dtype=np.float32, shape=(len(rows), dimension))
        decisions = np.lib.format.open_memmap(DECISIONS, mode="w+", dtype=np.float32, shape=(len(rows), 5))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(module_list):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = baseline.pad_right([row["future_prompt_ids"] for row in batch], device, pad)
                captures.clear()
                output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for q in range(len(module_list)):
                    selected = torch.stack([captures[q][local, ends[local]] for local in range(len(batch))])
                    states[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                for local, row in enumerate(batch):
                    target, wrong = row["target_code_id"], row["wrong_code_id"]
                    h = captures[len(module_list) - 1][local, ends[local]].float()
                    w = model.lm_head.weight[target].float() - model.lm_head.weight[wrong].float()
                    contribution = h * w
                    logits = output.logits[local, ends[local]].float()
                    margin = logits[target] - logits[wrong]
                    reconstructed = contribution.sum()
                    final[start + local] = contribution.cpu().numpy().astype(np.float32)
                    decisions[start + local] = [
                        float(margin.item()), float(reconstructed.item()), float(abs(margin.item() - reconstructed.item())),
                        float(margin.item() > 0), float(int(torch.argmax(logits).item()) == target),
                    ]
                for value in (states, final, decisions):
                    value.flush()
                save(PROGRESS, {"completed": start + len(batch), "shape": list(shape)})
                if (start + len(batch)) % 96 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2340 collect] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        for value in (states, final, decisions):
            value.flush(); close_memmap(value)
    np.save(GAMMA, model.model.norm.weight.detach().float().cpu().numpy().astype(np.float32))
    a_id, b_id = 362, 425
    np.save(AB_WEIGHT, (model.lm_head.weight[a_id].float() - model.lm_head.weight[b_id].float()).detach().cpu().numpy().astype(np.float32))
    return {"shape": list(shape), "layers": len(module_list) - 2, "dimension": dimension,
            "a_token_id": a_id, "b_token_id": b_id, "rms_norm_eps": float(model.config.rms_norm_eps)}


def derive(rows: list[dict], collection: dict) -> dict:
    states = np.load(STATES, mmap_mode="r")
    final = np.load(FINAL_CONTRIB, mmap_mode="r")
    gamma = np.load(GAMMA)
    ab_weight = np.load(AB_WEIGHT)
    if TRAJECTORY.exists() and TRAJECTORY_PROGRESS.exists():
        completed_q = int(json.loads(TRAJECTORY_PROGRESS.read_text(encoding="utf-8"))["completed_q"])
        trajectory = np.lib.format.open_memmap(TRAJECTORY, mode="r+")
    else:
        completed_q = 0
        TRAJECTORY.parent.mkdir(parents=True, exist_ok=True)
        trajectory = np.lib.format.open_memmap(TRAJECTORY, mode="w+", dtype=np.float32, shape=states.shape)
    signs = np.asarray([1.0 if row["target_code"] == "A" else -1.0 for row in rows], dtype=np.float32)[:, None]
    for q in range(completed_q, states.shape[1] - 1):
        trajectory[:, q] = layer_control.rms_norm(states[:, q], gamma, collection["rms_norm_eps"]) * ab_weight[None, :] * signs
        trajectory.flush(); save(TRAJECTORY_PROGRESS, {"completed_q": q + 1, "shape": list(states.shape)})
        print(f"[phase2340 derive] q={q}/{states.shape[1]-1}", flush=True)
    trajectory[:, -1] = final
    trajectory.flush()
    error = float(np.max(np.abs(trajectory[:, -1] - final)))
    shape = list(states.shape)
    for value in (states, final, trajectory):
        close_memmap(value)
    return {"shape": shape, "final_copy_max_abs_error": error}


def behavior(rows: list[dict], decisions: np.ndarray) -> dict:
    result = {"families": {}, "qualified": [], "qualified_macrotypes": [],
              "overall_forced_ab_accuracy": float(np.mean(decisions[:, 3])),
              "overall_unconstrained_exact_next_code_accuracy": float(np.mean(decisions[:, 4])),
              "max_accounting_abs_error": float(np.max(decisions[:, 2]))}
    for family in FAMILIES:
        cells = {}
        passed = True
        for condition in ("original", "option_swapped"):
            for partition in PARTITIONS:
                idx = [i for i, row in enumerate(rows) if row["condition"] == condition and row["family"] == family and row["partition"] == partition]
                cell = {"rows": len(idx), "forced_ab_accuracy": float(np.mean(decisions[idx, 3])),
                        "unconstrained_exact_next_code_accuracy": float(np.mean(decisions[idx, 4]))}
                cell["passed"] = cell["forced_ab_accuracy"] >= 0.70
                passed = passed and cell["passed"]
                cells[f"{condition}:{partition}"] = cell
        result["families"][family] = {"qualified": passed, "macrotype": contract.MACROTYPE[family], "cells": cells}
        if passed:
            result["qualified"].append(family)
    result["qualified_macrotypes"] = sorted({contract.MACROTYPE[family] for family in result["qualified"]})
    return result


def vector_classify(train: np.ndarray, test: np.ndarray, train_rows: list[dict], test_rows: list[dict],
                    labels: tuple[str, ...], label_key: str, train_partitions: tuple[str, ...], test_partition: str) -> dict:
    prototypes = np.stack([
        train[[i for i, row in enumerate(train_rows) if row["partition"] in train_partitions and row[label_key] == label]]
        .astype(np.float64).mean(axis=0)
        for label in labels
    ])
    indices = [i for i, row in enumerate(test_rows) if row["partition"] == test_partition and row[label_key] in labels]
    actual = test[indices].astype(np.float64)
    distances = (np.sum(np.square(actual), axis=1, keepdims=True) + np.sum(np.square(prototypes), axis=1)[None, :]
                 - 2 * actual @ prototypes.T)
    distances = np.maximum(distances, 0)
    correct = np.asarray([labels.index(test_rows[i][label_key]) for i in indices], dtype=np.int64)
    predicted = np.argmin(distances, axis=1)
    correct_distance = distances[np.arange(len(indices)), correct]
    masked = distances.copy(); masked[np.arange(len(indices)), correct] = np.inf
    best_wrong = np.min(masked, axis=1)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for c, p in zip(correct, predicted):
        confusion[c, p] += 1
    return {"rows": len(indices), "labels": len(labels), "chance": 1 / len(labels),
            "accuracy": float(np.mean(predicted == correct)),
            "median_correct_over_best_wrong_ratio": float(np.median(correct_distance / (best_wrong + EPS))),
            "confusion": confusion.tolist()}


def analyze(rows: list[dict], behavior_result: dict) -> dict:
    states = np.load(STATES, mmap_mode="r")
    trajectory = np.load(TRAJECTORY, mmap_mode="r")
    source_count = len(rows) // 2
    original_rows, swapped_rows = rows[:source_count], rows[source_count:]
    qualified = tuple(behavior_result["qualified"])
    trajectory_records = []
    for q in range(states.shape[1]):
        os, ss = states[:source_count, q].astype(np.float32), states[source_count:, q].astype(np.float32)
        oa, sa = np.abs(trajectory[:source_count, q].astype(np.float32)), np.abs(trajectory[source_count:, q].astype(np.float32))
        record = {"qpoint": q, "relative_block_depth": None if q in (0, states.shape[1] - 1) else float((q - 1) / 35),
                  "checkpoint_kind": "embedding" if q == 0 else "final_norm" if q == states.shape[1] - 1 else "block",
                  "paired_absolute": layer_control.pair_metrics(oa, sa), "partitions": {}}
        for partition in TEST_PARTITIONS:
            record["partitions"][partition] = {
                "all20_original_to_swap_abs": vector_classify(oa, sa, original_rows, swapped_rows, FAMILIES, "family", TRAIN_PARTITIONS, partition),
                "all20_swap_to_original_abs": vector_classify(sa, oa, swapped_rows, original_rows, FAMILIES, "family", TRAIN_PARTITIONS, partition),
                "qualified_original_to_swap_abs": vector_classify(oa, sa, original_rows, swapped_rows, qualified, "family", TRAIN_PARTITIONS, partition) if len(qualified) >= 2 else None,
                "qualified_swap_to_original_abs": vector_classify(sa, oa, swapped_rows, original_rows, qualified, "family", TRAIN_PARTITIONS, partition) if len(qualified) >= 2 else None,
                "all20_original_hidden": vector_classify(os, os, original_rows, original_rows, FAMILIES, "family", TRAIN_PARTITIONS, partition),
                "all20_swapped_hidden": vector_classify(ss, ss, swapped_rows, swapped_rows, FAMILIES, "family", TRAIN_PARTITIONS, partition),
                "macrotype_original_to_swap_abs": vector_classify(oa, sa, original_rows, swapped_rows, MACROTYPES, "macrotype", TRAIN_PARTITIONS, partition),
            }
        trajectory_records.append(record)
        print(f"[phase2340 analyze] q={q}/{states.shape[1]-1}", flush=True)
    write_rows(OUT / "analysis/checkpoint_trajectory.jsonl", trajectory_records)
    final = trajectory_records[-1]
    lock = final["partitions"]["fresh_lockbox"]
    k = len(qualified)
    if k >= 2:
        qa, qb = lock["qualified_original_to_swap_abs"], lock["qualified_swap_to_original_abs"]
        accuracy_threshold = max(0.30, 3 / k)
        family_pass = min(qa["accuracy"], qb["accuracy"]) >= accuracy_threshold and max(
            qa["median_correct_over_best_wrong_ratio"], qb["median_correct_over_best_wrong_ratio"]
        ) < 1.0
    else:
        qa = qb = None; accuracy_threshold = None; family_pass = False
    gate = {"qualified_family_min": 10, "qualified_macrotype_min": 6,
            "actual_qualified_families": k, "actual_qualified_macrotypes": len(behavior_result["qualified_macrotypes"]),
            "family_accuracy_threshold": accuracy_threshold,
            "lockbox_family_graph_pass": family_pass,
            "paired_final_mse_pass": final["paired_absolute"]["median_symmetric_relative_mse"] <= 0.50}
    gate["passed"] = all((k >= 10, len(behavior_result["qualified_macrotypes"]) >= 6,
                          gate["lockbox_family_graph_pass"], gate["paired_final_mse_pass"]))
    peaks = {}
    for name in ("all20_original_to_swap_abs", "all20_swap_to_original_abs"):
        best = max(trajectory_records, key=lambda row: row["partitions"]["fresh_lockbox"][name]["accuracy"])
        peaks[name] = {"qpoint": best["qpoint"], "relative_block_depth": best["relative_block_depth"],
                       **best["partitions"]["fresh_lockbox"][name]}
    for value in (states, trajectory):
        close_memmap(value)
    return {"qualified_families": list(qualified), "qualified_macrotypes": behavior_result["qualified_macrotypes"],
            "checkpoint_count": len(trajectory_records), "peaks_fresh_lockbox": peaks,
            "fresh_confirmation_final": final["partitions"]["fresh_confirmation"],
            "fresh_lockbox_final": lock, "paired_final": final["paired_absolute"], "gate": gate}


def publish(rows: list[dict]) -> list[dict]:
    specs = (
        ("c7601_qwen4b_twenty_family_fixed_ab_boundary", STATES, np.float16,
         "Qwen3-4B twenty-family fixed-A/B HiddenState full field", "twenty_family_fixed_ab_hiddenstate_v1"),
        ("c7602_qwen4b_twenty_family_fixed_ab_layerwise_contribution", TRAJECTORY, np.float32,
         "Qwen3-4B twenty-family layerwise coordinate-use field", "twenty_family_fixed_ab_coordinate_contribution_v1"),
    )
    assets = []
    for dataset_id, path, dtype, title, schema in specs:
        source = np.load(path, mmap_mode="r")
        flat = source.reshape(-1, source.shape[-1])
        metadata = [
            {"case_id": row["case_id"], "condition": row["condition"], "family": row["family"],
             "macrotype": row["macrotype"], "language": row["language"], "surface": row["surface"],
             "unit": row["unit"], "state": row["state"], "partition": row["partition"],
             "target_code": row["target_code"], "qpoint": q}
            for row in rows for q in range(source.shape[1])
        ]
        binary = VIS / f"{dataset_id}.{np.dtype(dtype).name}.npy"
        out = atlas.create_binary(binary.name, flat.shape[0], flat.shape[1], dtype)
        out[:] = flat
        out.flush(); close_memmap(out); close_memmap(source)
        assets.append(atlas.write_metadata(
            dataset_id, title, binary, metadata, "Qwen3-4B-FP16", schema,
            "twenty-family breadth atlas", "20 families, 9 macrotypes, 3840 prompts, all 38 checkpoints",
            title, {"coordinate_count": 2560, "no_projection": True, "fixed_output_codes": ["A", "B"]},
        ))
    return assets


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 二十语言模式族的固定接口全坐标广度图谱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 自动续阶段从三族跨模型复验回到普遍性优先。将Phase2330的20族、9宏类、1920条自然候选材料全部重编译为公共单token A/B选择；再逐题交换A/B候选，得到 `{result['material_audit']['rows']}` 条。每个condition×family×partition内A/B严格平衡；discovery+confirmation只建原型，fresh_confirmation复核，fresh_lockbox裁决。Qwen3-4B FP16非量化、CUDA；保存3840×38×2560全部HiddenState及最终RMSNorm读出的逐层逐坐标贡献，不做Top-K、PCA或坐标压缩。

$$
c_{{i,q,j}}=\operatorname{{RMSNorm}}(h_{{i,q}})_j(W_{{y_i^+,j}}-W_{{y_i^-,j}}),
\qquad a_{{i,q,j}}=|c_{{i,q,j}}|.
$$

$$
\operatorname{{AtlasPass}}=[K_{{beh}}ge10]\land[M_{{beh}}ge6]\land
[A^{{cross}}_{{lock}}\ge\max(0.30,3/K_{{beh}})]\land[\operatorname{{median}}(E_c/E_w)<1]\land[E^{{pair}}\le0.5].
$$

**结果汇总与相关文件。** 材料 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；分析 `{json.dumps(result['analysis'], ensure_ascii=False)}`；发布 `{json.dumps(result['datasets'], ensure_ascii=False)}`；验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；客户端构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2340_c7601_c7840_twenty_family_fixed_interface_atlas.py`；结果 `tests/glm5/result/phase2340_c7601_c7840_twenty_family_fixed_interface_atlas`。

**理论进展、问题硬伤与结论。** 这一Phase回答“固定输出接口的坐标强度纹理能覆盖多少语言族”，不回答“族的抽象定义是否已被找到”。族分类可能直接利用家族特有词汇、提示长度、标点和模板；即使选项交换、跨分区通过，也只能建立族条件全坐标图谱。行为未通过的族不应用内部纹理支持语义结论；行为通过也只是模型能完成二选一。absolute场丢失促进/抑制方向，RMS logit lens在q0–q36不是原生中层logit；程序生成材料、同一模型与FP16低值误差仍是硬伤。

**下一阶段路线判断。** 若广度门通过，目标相同，下一阶段自动把合格族按“同词汇跨任务、同任务跨词汇、同句跨标签”三轴正交重编译，寻找能在排除族词汇身份后仍复用的坐标图谱；若门失败，则只对行为合格族扩充自然构式，不做因果闭合。无论结果如何都不回退到单方向搬运。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    rows, audit = compile_material(tokenizer)
    write_rows(MATERIAL, rows)
    freeze = {"frozen_before_model_load": True, "families": list(FAMILIES), "macrotypes": list(MACROTYPES),
              "partitions": list(PARTITIONS), "train_partitions": list(TRAIN_PARTITIONS), "test_partitions": list(TEST_PARTITIONS),
              "behavior_each_condition_partition": 0.70, "qualified_family_min": 10, "qualified_macrotype_min": 6,
              "family_accuracy": "max(0.30,3/K)", "paired_symmetric_mse_max": 0.50}
    save(OUT / "config/frozen_contract.json", freeze)
    model = None
    try:
        model, _tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        collection = collect(model, device, rows)
    finally:
        if model is not None:
            model_utils.release_model(model)
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    decisions = np.load(DECISIONS, mmap_mode="r")
    behavior_result = behavior(rows, decisions)
    close_memmap(decisions)
    derived = derive(rows, collection)
    analysis = analyze(rows, behavior_result)
    datasets = publish(rows)
    verification = [atlas.verify(row) for row in datasets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    if not verified: raise RuntimeError(("verification_failed", verification))
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not build["passed"]: raise RuntimeError(("frontend_build_failed", build))
    checks = {"rows": len(rows) == 3840, "balanced": audit["balanced_every_condition_family_partition"],
              "all_coordinates": collection["shape"] == [3840, 38, 2560] and derived["shape"] == [3840, 38, 2560],
              "final_copy_exact": derived["final_copy_max_abs_error"] == 0, "assets_verified": verified, "frontend_build": build["passed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "material_audit": audit,
              "collection": collection, "behavior": behavior_result, "derived": derived, "analysis": analysis,
              "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)), "verification": verification,
              "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)), "frontend_build": build,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]: raise RuntimeError(("phase2340_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
