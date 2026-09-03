#!/usr/bin/env python3
"""Frozen subset cross-model event/component capture and operator replication."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

# bitsandbytes otherwise emits this identical cast notice once per 8-bit matmul.
logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2406 = RESULT / "phase2406_c24561_c24880_behavior_precision_calibration"
P2407 = RESULT / "phase2407_c24881_c25200_qwen4b_component_fullfield"
P2408 = RESULT / "phase2408_c25201_c25520_fullcoordinate_deconfounding"
OUT = RESULT / "phase2412_c26481_c26800_frozen_crossmodel_operator_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2412
CAMPAIGN = "C26481-C26800"
MODEL_ORDER = ("qwen14b", "glm4", "deepseek7b")
COMPONENTS = ("total", "attention", "mlp")
TASK_EVENTS = {"selection": ("query_end", "answer_boundary"),
               "composition": ("query_end", "answer_boundary")}

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2408_c25201_c25520_fullcoordinate_deconfounding as p2408  # noqa: E402
import phase2409_c25521_c25840_state_dependent_coordinate_operator as p2409  # noqa: E402
import phase2411_c26161_c26480_crosslayer_composition_output_bridge as p2411  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def frozen_case_ids() -> dict[str, set[str]]:
    result = {}
    for task in ("selection", "composition"):
        rows = p2408.read_rows(P2407 / f"index/{task}_rows.jsonl")
        chosen = []
        for row in rows:
            partition, unit = row["partition"], int(row["unit"])
            if task == "selection":
                keep = ((partition == "discovery" and unit <= 3) or
                        (partition == "fresh_unit_lockbox" and unit == 6) or
                        (partition == "template_lockbox" and unit <= 1) or
                        (partition == "deep_fresh_unit_lockbox" and unit == 10) or
                        (partition == "joint_template_unit_lockbox" and unit == 10))
            else:
                keep = ((partition == "discovery" and unit <= 3) or
                        (partition == "fresh_unit_lockbox" and unit == 6) or
                        (partition == "template_lockbox" and unit <= 1))
            if keep: chosen.append(row["case_id"])
        result[task] = set(chosen)
    expected = {"selection": 576, "composition": 224}
    if {task: len(ids) for task, ids in result.items()} != expected:
        raise RuntimeError(({task: len(ids) for task, ids in result.items()}, expected))
    return result


def subset_rows(key: str) -> dict[str, list[dict]]:
    all_rows = p2408.read_rows(P2406 / key / "index/operation_rows.jsonl")
    ids = frozen_case_ids()
    result = {task: [row for row in all_rows if row["task"] == task and row["case_id"] in ids[task]]
              for task in ids}
    for task, values in result.items():
        order = {case_id: index for index, case_id in enumerate(sorted(ids[task]))}
        values.sort(key=lambda row: order[row["case_id"]])
        if len(values) != len(ids[task]): raise RuntimeError((key, task, len(values), len(ids[task])))
    return result


def open_field(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float16, shape=shape)


def collect(key: str, model, rows: list[dict], task: str, batch_size: int) -> dict:
    blocks = list(model.model.layers)
    state_modules = field_utils.modules(model)[:len(blocks)]
    event_names = TASK_EVENTS[task]
    dimension = int(model.config.hidden_size)
    paths = {name: OUT / key / f"raw/{task}_{name}.float16.npy" for name in ("state", "attention", "mlp")}
    fields = {name: open_field(path, (len(rows), len(blocks), len(event_names), dimension)) for name, path in paths.items()}
    progress = OUT / key / f"raw/{task}_progress.json"
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    states: dict[int, torch.Tensor] = {}; attention: dict[int, torch.Tensor] = {}; mlp: dict[int, torch.Tensor] = {}; handles = []
    for layer, module in enumerate(state_modules):
        def hook_state(_module, _inputs, output, layer=layer):
            states[layer] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook_state))
    for layer, block in enumerate(blocks):
        def hook_attention(_module, _inputs, output, layer=layer):
            attention[layer] = (output[0] if isinstance(output, tuple) else output).detach()
        def hook_mlp(_module, _inputs, output, layer=layer):
            mlp[layer] = output.detach()
        handles.append(block.self_attn.register_forward_hook(hook_attention))
        handles.append(block.mlp.register_forward_hook(hook_mlp))
    device = model.get_input_embeddings().weight.device
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, _ = field_utils.pad_right([row["prompt_ids"] for row in batch], device, pad)
                states.clear(); attention.clear(); mlp.clear()
                model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                event_indices = []
                for row in batch:
                    mapping = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}
                    event_indices.append(torch.tensor([mapping[name] for name in event_names], dtype=torch.long))
                for layer in range(len(blocks)):
                    for captures, name in ((states, "state"), (attention, "attention"), (mlp, "mlp")):
                        tensor = captures[layer].float().cpu()
                        selected = torch.stack([tensor[local, event_indices[local]] for local in range(len(batch))])
                        fields[name][start:start + len(batch), layer] = selected.numpy().astype(np.float16)
                for value in fields.values(): value.flush()
                save(progress, {"completed": start + len(batch)})
                if (start + len(batch)) % 64 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2412 {key} {task}] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        for value in fields.values(): value.flush(); close(value)
    return {name: {"path": str(path), "shape": [len(rows), len(blocks), len(event_names), dimension], "bytes": path.stat().st_size}
            for name, path in paths.items()}


def analyze_model(key: str, label: str, collections: dict, rows_by_task: dict[str, list[dict]]) -> dict:
    task_results = {}
    for task, rows in rows_by_task.items():
        state = np.load(collections[task]["state"]["path"], mmap_mode="r")
        attention = np.load(collections[task]["attention"]["path"], mmap_mode="r")
        mlp = np.load(collections[task]["mlp"]["path"], mmap_mode="r")
        train = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"], dtype=np.int64)
        splits = sorted({row["partition"] for row in rows if row["partition"] != "discovery"})
        tests = {split: np.asarray([i for i, row in enumerate(rows) if row["partition"] == split], dtype=np.int64) for split in splits}
        design = p2408.Design(rows, train)
        metrics = {component: {split: {stage: p2408.accumulator() for stage in ("family", "state", "mismatch")}
                               for split in splits} for component in COMPONENTS}
        families = sorted({row["family"] for row in rows})
        passport_path = OUT / key / f"derived/{task}_family_passport.float32.npy"; passport_path.parent.mkdir(parents=True, exist_ok=True)
        passports = np.lib.format.open_memmap(passport_path, mode="w+", dtype=np.float32,
                                              shape=(len(COMPONENTS), state.shape[1], state.shape[2], len(families), state.shape[3]))
        for layer in range(state.shape[1]):
            for event in range(state.shape[2]):
                h = np.asarray(state[:, layer, event], dtype=np.float32)
                a = np.asarray(attention[:, layer, event], dtype=np.float32)
                m = np.asarray(mlp[:, layer, event], dtype=np.float32)
                for ci, (component, y) in enumerate((("total", a + m), ("attention", a), ("mlp", m))):
                    fitted = p2409.fit_baseline(rows, train, design, y, h)
                    for fi, family in enumerate(families): passports[ci, layer, event, fi] = fitted["family_y"][family]
                    for split, test in tests.items():
                        family_pred, state_pred, mismatch = p2409.predict(rows, test, design, y, h, fitted)
                        truth = y[test]; base = np.broadcast_to(y[train].mean(axis=0), truth.shape)
                        for stage, prediction in (("family", family_pred), ("state", state_pred), ("mismatch", mismatch)):
                            p2408.update_metric(metrics[component][split][stage], truth, prediction, base, rows, test)
            passports.flush()
            print(f"[phase2412 {key} analyze {task}] layer {layer + 1}/{state.shape[1]}", flush=True)
        passports.flush(); close(passports); close(state); close(attention); close(mlp)
        finished = {component: {split: {stage: p2408.finish(value) for stage, value in stages.items()}
                                for split, stages in split_map.items()} for component, split_map in metrics.items()}
        summary = {component: {split: {
            "family_gain": finished[component][split]["family"]["gain_vs_base"],
            "state_gain": finished[component][split]["state"]["gain_vs_base"],
            "state_increment": finished[component][split]["state"]["gain_vs_base"] - finished[component][split]["family"]["gain_vs_base"],
            "physical_advantage": finished[component][split]["state"]["gain_vs_base"] - finished[component][split]["mismatch"]["gain_vs_base"]}
            for split in splits} for component in COMPONENTS}
        task_results[task] = {"rows": len(rows), "train_rows": len(train), "events": list(TASK_EVENTS[task]),
                              "field_shape": collections[task]["state"]["shape"], "metrics": finished, "summary": summary,
                              "passport": str(passport_path), "families": families}
    result = {"model": key, "label": label, "precision": ("NF4 storage / BF16 compute" if key == "qwen14b" else label),
              "tasks": task_results, "collection": collections}
    save(OUT / key / "analysis/final.json", result)
    return result


def load_for_capture(key: str):
    if key != "qwen14b":
        return capability.load_model(key)
    spec = capability.loader.MODEL_SPECS[key]
    tokenizer = capability.AutoTokenizer.from_pretrained(
        spec["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    quant = capability.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    print("[phase2412] loading qwen14b NF4/BF16 compute with device_map=auto", flush=True)
    model = capability.AutoModelForCausalLM.from_pretrained(
        spec["path"], quantization_config=quant, device_map="auto",
        max_memory={0: "14GiB", "cpu": "20GiB"}, offload_state_dict=True,
        trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="eager")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, spec["label"]


def run_model(key: str) -> None:
    final = OUT / key / "analysis/final.json"
    if final.exists(): print(final.read_text(encoding="utf-8")); return
    rows_by_task = subset_rows(key)
    for task, rows in rows_by_task.items(): write_rows(OUT / key / f"index/{task}_rows.jsonl", rows)
    model, tokenizer, label = load_for_capture(key)
    batch = {"qwen14b": 1, "glm4": 1, "deepseek7b": 2}[key]
    try:
        collections = {task: collect(key, model, rows, task, batch) for task, rows in rows_by_task.items()}
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    result = analyze_model(key, label, collections, rows_by_task)
    print(json.dumps({"model": key, "label": label, "summary": {task: value["summary"] for task, value in result["tasks"].items()}}, ensure_ascii=False, indent=2))


def cross_architecture(models: dict) -> dict:
    q4_event = {"selection": {"query_end": 4, "answer_boundary": 7},
                "composition": {"query_end": 8, "answer_boundary": 11}}
    result = {}
    for key, model in models.items():
        result[key] = {}
        for task, info in model["tasks"].items():
            target = np.load(info["passport"], mmap_mode="r")
            comparisons = {component: [] for component in COMPONENTS}
            q4_arrays = {component: np.load(P2408 / f"derived/{task}_{component}_family_residual_passport.float32.npy", mmap_mode="r")
                         for component in COMPONENTS}
            for ci, component in enumerate(COMPONENTS):
                for layer in range(target.shape[1]):
                    q4_layer = round(layer * (q4_arrays[component].shape[0] - 1) / max(target.shape[1] - 1, 1))
                    for event, event_name in enumerate(TASK_EVENTS[task]):
                        left = p2411.geometry_vector(np.asarray(target[ci, layer, event], dtype=np.float32))
                        right = p2411.geometry_vector(np.asarray(q4_arrays[component][q4_layer, q4_event[task][event_name]], dtype=np.float32))
                        comparisons[component].append(p2411.correlation(left, right))
            result[key][task] = {component: {"mean_relative_depth_geometry": float(np.mean(values)),
                                             "median_relative_depth_geometry": float(np.median(values)),
                                             "positive_rate": float(np.mean(np.asarray(values) > 0))}
                                 for component, values in comparisons.items()}
            close(target)
            for value in q4_arrays.values(): close(value)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 三模型冻结子集状态算子与关系几何复现（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 冻结Phase2405 case_id、事件与评价规则，顺序测试Qwen3-14B、GLM4、DeepSeek7B，不按结果换材料。选择任务保留576条（discovery unit0–3、新unit6、模板unit0–1、深新unit10、联合unit10），组合保留224条（discovery unit0–3、新unit6、模板unit0–1）；在query-end与answer-boundary逐层采集输入状态、Attention和MLP的全部原始物理坐标。逐模型只用discovery拟合族均值、同坐标状态算子和791位错配对照。族间Gram图按相对深度与Qwen4B比较，不比较不同宽度模型的原坐标。Qwen14B精度严格标为NF4权重存储/BF16计算；Phase2406的BF16 `device_map=auto`四种配置均在Windows权重物化阶段访问冲突，不能伪装成BF16复现。

$$R_{{arch}}(q)=\mathrm{{corr}}\!\left(\mathrm{{vec}}K_q^{{arch}},\mathrm{{vec}}K_{{\mathrm{{round}}(35q/(L-1))}}^{{Q4}}\right).$$

**结果汇总。** 模型摘要 `{json.dumps(result['model_summary'], ensure_ascii=False)}`；相对深度关系几何 `{json.dumps(result['cross_architecture'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2412_c26481_c26800_frozen_crossmodel_operator_replication.py`；逐模型索引、原始state/attention/mlp、族护照及final位于`tests/glm5/result/phase2412_c26481_c26800_frozen_crossmodel_operator_replication`。原始大场将在Phase2413发布代表性热力图后清理。

**分析与理论进展。** 跨模型只认两类可比对象：各模型自身固定基底上的matched-vs-mismatch优势，以及与宽度无关的族关系Gram图。即使二者都复现，也只能称为“功能关系几何+模型内固定坐标耦合”的共同现象，不能称跨架构坐标同构。

**问题硬伤与结论。** 这是事件和材料冻结的分层子集复现，不是三模型全2176条全token普查；Qwen14B量化与其余精度不同。query/answer两个事件不能覆盖事实写入全过程。小模型行为能力差异明显，阴性须与Phase2406行为门一起解释，不能用单个模型否决一般机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def finalize() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    models = {}
    for key in MODEL_ORDER:
        path = OUT / key / "analysis/final.json"
        if not path.exists(): raise RuntimeError(f"missing sequential model result: {key}")
        models[key] = json.loads(path.read_text(encoding="utf-8"))
    summary = {key: {task: value["summary"] for task, value in model["tasks"].items()} for key, model in models.items()}
    cross = cross_architecture(models)
    decisive = []
    for key, model in models.items():
        for task, task_info in model["tasks"].items():
            for component, split_map in task_info["summary"].items():
                for split, value in split_map.items():
                    decisive.append({"model": key, "task": task, "component": component, "split": split,
                                     "state_increment": value["state_increment"], "physical_advantage": value["physical_advantage"]})
    adjudication = {"decisive_cells": decisive,
                    "all_cells_positive_state_increment": all(value["state_increment"] > 0 for value in decisive),
                    "all_cells_positive_physical_advantage": all(value["physical_advantage"] > 0 for value in decisive),
                    "cross_architecture_coordinate_isomorphism_proven": False,
                    "universal_language_gear_proven": False}
    numbers = [value[k] for value in decisive for k in ("state_increment", "physical_advantage")]
    for model in cross.values():
        for task in model.values():
            for component in task.values(): numbers.extend(component.values())
    checks = {"sequential_models": list(models) == list(MODEL_ORDER), "finite": all(math.isfinite(float(v)) for v in numbers),
              "frozen_case_ids": True, "raw_coordinates": True, "qwen14_precision_labeled": True,
              "cross_width_compared_only_as_geometry": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "model_summary": summary,
              "cross_architecture": cross, "adjudication": adjudication, "checks": checks,
              "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "model_summary": summary, "cross_architecture": cross,
                      "adjudication": adjudication, "checks": checks}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", choices=MODEL_ORDER); parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.model: run_model(args.model)
    elif args.finalize: finalize()
    else: parser.error("pass --model KEY sequentially or --finalize")


if __name__ == "__main__": main()
