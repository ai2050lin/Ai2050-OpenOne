#!/usr/bin/env python3
"""Sequential cross-model replication of the behavior-qualified supported-task atlas."""
from __future__ import annotations

import gc
import json
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
P2344 = RESULT / "phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract"
OUT = RESULT / "phase2349_c8841_c8960_crossmodel_supported_task_functional_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MATERIAL = P2344 / "material/bilingual_factorial_fixed_code.jsonl"
PHASE = 2349
CAMPAIGN = "C8841-C8960"
MODEL_ORDER = ("qwen14b", "glm4", "deepseek7b")
TRAIN_PARTITIONS = ("discovery", "confirmation")
SELECTION_PARTITION = "fresh_confirmation"
LOCKBOX_PARTITION = "fresh_lockbox"
EPS = 1e-12

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls as layer_control  # noqa: E402
import phase2339_c7401_c7600_crossmodel_fixed_ab_replication as crossmodel  # noqa: E402
import phase2344_c8241_c8360_bilingual_factorial_semantic_graph_contract as contract  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def paths(key: str) -> dict[str, Path]:
    base = OUT / key
    return {"base": base, "rows": base / "material/rows.jsonl", "states": base / "raw/boundary_all_checkpoints.float16.npy",
            "decisions": base / "raw/decisions.float32.npy", "gamma": base / "raw/final_norm_gamma.float32.npy",
            "progress": base / "raw/progress.json", "final": base / "analysis/final.json"}


def source_rows() -> list[dict]:
    return [row for row in io.read_rows(MATERIAL) if row["task"] == "select_supported" and row["surface"] == "natural"
            and row["codebook"] == "AB" and row["condition"] == "original"]


def compile_rows(tokenizer) -> tuple[list[dict], dict]:
    output = []
    for source in source_rows():
        prompt_ids = [int(value) for value in tokenizer.encode(source["future_prompt"], add_special_tokens=False)]
        target = tokenizer.encode(" " + source["target_code"], add_special_tokens=False)
        wrong = tokenizer.encode(" " + source["wrong_code"], add_special_tokens=False)
        if len(target) != 1 or len(wrong) != 1:
            raise RuntimeError(("non_single_token_code", source["target_code"], target, source["wrong_code"], wrong))
        output.append({**source, "model_design_index": len(output), "future_prompt_ids": prompt_ids,
                       "target_code_id": int(target[0]), "wrong_code_id": int(wrong[0]),
                       "boundary_position": len(prompt_ids) - 1})
    lengths = [len(row["future_prompt_ids"]) for row in output]
    return output, {"rows": len(output), "families": len(contract.FAMILIES), "units": 16,
                    "languages": 2, "lexical_sets": 2, "states": 2,
                    "token_length_min": min(lengths), "token_length_max": max(lengths)}


def collect(key: str, model, device, rows: list[dict]) -> dict:
    p = paths(key)
    modules = crossmodel.model_modules(model)
    dimension = int(model.get_input_embeddings().weight.shape[1])
    shape = (len(rows), len(modules), dimension)
    if p["states"].exists() and p["decisions"].exists() and p["progress"].exists():
        completed = int(json.loads(p["progress"].read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(p["states"], mode="r+")
        decisions = np.lib.format.open_memmap(p["decisions"], mode="r+")
    else:
        completed = 0
        p["states"].parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(p["states"], mode="w+", dtype=np.float16, shape=shape)
        decisions = np.lib.format.open_memmap(p["decisions"], mode="w+", dtype=np.float32, shape=(len(rows), 3))
    captures = {}
    handles = []
    for qpoint, module in enumerate(modules):
        def hook(_module, _inputs, value, qpoint=qpoint):
            captures[qpoint] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    batch_size = int(crossmodel.MODEL_SPECS[key]["batch"])
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = baseline.pad_right([row["future_prompt_ids"] for row in batch], device, pad)
                captures.clear()
                output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                     use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                final = None
                for qpoint in range(len(modules)):
                    selected = torch.stack([captures[qpoint][local, ends[local]] for local in range(len(batch))])
                    states[start:start + len(batch), qpoint] = selected.float().cpu().numpy().astype(np.float16)
                    if qpoint == len(modules) - 1:
                        final = selected
                logits = model.lm_head(final).float()
                for local, row in enumerate(batch):
                    target, wrong = row["target_code_id"], row["wrong_code_id"]
                    margin = logits[local, target] - logits[local, wrong]
                    decisions[start + local] = [float(margin.item()), float(margin.item() > 0),
                                                float(int(torch.argmax(logits[local]).item()) == target)]
                states.flush(); decisions.flush()
                save(p["progress"], {"completed": start + len(batch), "shape": list(shape)})
                if (start + len(batch)) % max(64, batch_size) == 0 or start + len(batch) == len(rows):
                    print(f"[phase2349 {key}] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        states.flush(); decisions.flush(); close_memmap(states); close_memmap(decisions)
    p["gamma"].parent.mkdir(parents=True, exist_ok=True)
    np.save(p["gamma"], model.model.norm.weight.detach().float().cpu().numpy().astype(np.float32))
    return {"shape": list(shape), "layers": len(modules) - 2, "dimension": dimension,
            "batch_size": batch_size, "quantization": crossmodel.MODEL_SPECS[key]["quant"],
            "model_label": crossmodel.MODEL_SPECS[key]["label"],
            "rms_norm_eps": float(getattr(model.config, "rms_norm_eps", 1e-6))}


def behavior(key: str, rows: list[dict]) -> dict:
    decisions = np.load(paths(key)["decisions"], mmap_mode="r")
    families = {}
    qualified = []
    for family in contract.FAMILIES:
        cells = {}
        passed = True
        for language in contract.LANGUAGES:
            for partition in contract.PARTITIONS:
                idx = [i for i, row in enumerate(rows) if row["family"] == family and row["language"] == language
                       and row["partition"] == partition]
                accuracy = float(np.mean(decisions[idx, 1]))
                cells[f"{language}:{partition}"] = {"rows": len(idx), "accuracy": accuracy, "passed": accuracy >= 0.70}
                passed = passed and accuracy >= 0.70
        families[family] = {"qualified": passed, "cells": cells}
        if passed:
            qualified.append(family)
    result = {"overall_forced_accuracy": float(np.mean(decisions[:, 1])),
              "overall_unconstrained_exact_accuracy": float(np.mean(decisions[:, 2])),
              "families": families, "qualified": qualified}
    close_memmap(decisions)
    return result


def grouped(field: np.ndarray, rows: list[dict], labels: tuple[str, ...], factor: str,
            source: str, target: str, partition: str) -> dict:
    prototypes = np.stack([field[[i for i, row in enumerate(rows) if row["family"] == label
                                         and row["partition"] in TRAIN_PARTITIONS and row[factor] == source]]
                            .mean(axis=0, dtype=np.float64) for label in labels])
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row["family"] in labels and row["partition"] == partition and row[factor] == target:
            groups[(row["family"], row["unit"])].append(index)
    keys = sorted(groups)
    actual = np.stack([field[groups[key]].mean(axis=0, dtype=np.float64) for key in keys])
    distances = np.maximum(np.sum(actual * actual, axis=1, keepdims=True) + np.sum(prototypes * prototypes, axis=1)[None, :]
                           - 2 * actual @ prototypes.T, 0)
    correct = np.asarray([labels.index(key[0]) for key in keys])
    predicted = np.argmin(distances, axis=1)
    correct_distance = distances[np.arange(len(keys)), correct]
    masked = distances.copy(); masked[np.arange(len(keys)), correct] = np.inf
    return {"rows": len(keys), "accuracy": float(np.mean(predicted == correct)), "chance": 1 / len(labels),
            "median_correct_over_best_wrong_ratio": float(np.median(correct_distance / (np.min(masked, axis=1) + EPS)))}


def evaluate(field: np.ndarray, rows: list[dict], labels: tuple[str, ...], partition: str) -> dict:
    specs = (("en_to_zh", "language", "en", "zh"), ("zh_to_en", "language", "zh", "en"),
             ("lex0_to_lex1", "lexical_set", "lex0", "lex1"), ("lex1_to_lex0", "lexical_set", "lex1", "lex0"))
    transfers = {name: grouped(field, rows, labels, factor, source, target, partition)
                 for name, factor, source, target in specs}
    return {"transfers": transfers, "minimum_accuracy": min(value["accuracy"] for value in transfers.values()),
            "mean_accuracy": float(np.mean([value["accuracy"] for value in transfers.values()])),
            "maximum_distance_ratio": max(value["median_correct_over_best_wrong_ratio"] for value in transfers.values())}


def analyze(key: str, rows: list[dict], collection: dict, behavior_result: dict) -> dict:
    states = np.load(paths(key)["states"], mmap_mode="r")
    gamma = np.load(paths(key)["gamma"]).astype(np.float32)
    labels = tuple(behavior_result["qualified"])
    if len(labels) < 2:
        labels = tuple(contract.FAMILIES)
    records = []
    for qpoint in range(states.shape[1]):
        raw = states[:, qpoint].astype(np.float32)
        if qpoint != states.shape[1] - 1:
            raw = layer_control.rms_norm(raw, gamma, collection["rms_norm_eps"])
        absolute = np.abs(raw)
        selection = evaluate(absolute, rows, labels, SELECTION_PARTITION)
        sorted_control = evaluate(np.sort(absolute, axis=1), rows, labels, SELECTION_PARTITION)
        records.append({"qpoint": qpoint, "relative_block_depth": None if qpoint in (0, states.shape[1] - 1)
                        else (qpoint - 1) / max(collection["layers"] - 1, 1),
                        "absolute_hidden": selection, "row_sorted_absolute_hidden": sorted_control})
    selected = max(records, key=lambda row: (row["absolute_hidden"]["minimum_accuracy"],
                                              row["absolute_hidden"]["mean_accuracy"], -row["qpoint"]))
    qpoint = int(selected["qpoint"])
    raw = states[:, qpoint].astype(np.float32)
    if qpoint != states.shape[1] - 1:
        raw = layer_control.rms_norm(raw, gamma, collection["rms_norm_eps"])
    lock = evaluate(np.abs(raw), rows, labels, LOCKBOX_PARTITION)
    sorted_lock = evaluate(np.sort(np.abs(raw), axis=1), rows, labels, LOCKBOX_PARTITION)
    gate = {"qualified_family_count": len(behavior_result["qualified"]), "behavior_pass": len(behavior_result["qualified"]) >= 8,
            "selected_qpoint": qpoint, "relative_block_depth": selected["relative_block_depth"],
            "lockbox_minimum_accuracy": lock["minimum_accuracy"],
            "coordinate_advantage_over_sorted": lock["minimum_accuracy"] - sorted_lock["minimum_accuracy"],
            "distance_ratio": lock["maximum_distance_ratio"],
            "transfer_pass": lock["minimum_accuracy"] >= 0.30,
            "coordinate_identity_pass": lock["minimum_accuracy"] >= sorted_lock["minimum_accuracy"] + 0.10,
            "distance_pass": lock["maximum_distance_ratio"] < 1.0}
    gate["passed"] = all((gate["behavior_pass"], gate["transfer_pass"], gate["coordinate_identity_pass"], gate["distance_pass"]))
    close_memmap(states)
    return {"labels": list(labels), "selection_trajectory": records, "selected": selected,
            "lockbox_absolute_hidden": lock, "lockbox_sorted_control": sorted_lock, "gate": gate}


def publish(key: str, rows: list[dict], analysis: dict, collection: dict) -> dict:
    source = np.load(paths(key)["states"], mmap_mode="r")
    qpoints = []
    for qpoint in (0, int(analysis["gate"]["selected_qpoint"]), source.shape[1] - 1):
        if qpoint not in qpoints:
            qpoints.append(qpoint)
    dataset_id = f"c8841_{key}_supported_task_key_checkpoint_hiddenstate"
    binary = VIS / f"{dataset_id}.float16.npy"
    out = atlas.create_binary(binary.name, len(rows) * len(qpoints), source.shape[-1], np.float16)
    metadata = []
    cursor = 0
    for qpoint in qpoints:
        out[cursor:cursor + len(rows)] = source[:, qpoint]
        metadata.extend({"case_id": row["case_id"], "family": row["family"], "macrotype": row["macrotype"],
                         "language": row["language"], "lexical_set": row["lexical_set"], "partition": row["partition"],
                         "unit": row["unit"], "state": row["state"], "qpoint": qpoint,
                         "relative_block_depth": None if qpoint in (0, source.shape[1] - 1)
                         else (qpoint - 1) / max(collection["layers"] - 1, 1)} for row in rows)
        cursor += len(rows)
    out.flush(); close_memmap(out); close_memmap(source)
    return atlas.write_metadata(
        dataset_id, f"{collection['model_label']} supported-task key-checkpoint HiddenState",
        binary, metadata, collection["model_label"], "crossmodel_supported_task_key_hiddenstate_v1",
        "model-local observational functional replication; raw coordinate numbers are not compared across models",
        "12 supported-task families x bilingual x two lexicons x 16 units x two states",
        f"all {collection['dimension']} model-local coordinates at embedding, selected layer and final norm",
        {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": collection["dimension"], "no_topk": True,
         "qpoints": qpoints, "quantization": collection["quantization"]},
    )


def run_model(key: str) -> dict:
    p = paths(key)
    if p["final"].exists():
        return json.loads(p["final"].read_text(encoding="utf-8"))
    model = None
    try:
        model, tokenizer, device = crossmodel.load_model(key)
        rows, material_audit = compile_rows(tokenizer)
        io.write_rows(p["rows"], rows)
        collection = collect(key, model, device, rows)
    finally:
        if model is not None:
            model.to("cpu") if not getattr(model, "is_quantized", False) else None
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    behavior_result = behavior(key, rows)
    analysis = analyze(key, rows, collection, behavior_result)
    dataset = publish(key, rows, analysis, collection)
    verification = atlas.verify(dataset)
    verified = all(value for name, value in verification.items() if name != "id")
    raw_size = p["states"].stat().st_size
    p["states"].unlink()
    checks = {"rows": material_audit["rows"] == 1536, "all_coordinates": collection["shape"][2] == collection["dimension"],
              "asset_verified": verified, "raw_state_deleted": not p["states"].exists()}
    result = {"model": key, "material_audit": material_audit, "collection": collection,
              "behavior": behavior_result, "analysis": analysis,
              "dataset": json.loads(json.dumps(dataset, ensure_ascii=False, default=str)), "verification": verification,
              "cleanup": {"deleted": str(p["states"]), "bytes_reclaimed": raw_size},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(p["final"], result)
    return result


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 十二族支持任务的三异构模型顺序功能图谱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 按用户要求逐个而非并行加载Qwen3-14B NF4、GLM4-9B INT8、DeepSeek-7B INT8；每模型重新分词同一12族`select_supported`、natural、AB、original材料，覆盖中英、两词汇、16 units、两状态，共1536条。保存并分析全部本地层×全部本地坐标；discovery+confirmation建原型、fresh_confirmation选层、fresh_lockbox裁决中英双向和词汇双向迁移。跨模型只比较行为、准确率和相对深度，不比较坐标编号。

$$
\rho_q=\frac{{q-1}}{{L-1}},\qquad
\Phi_m=(B_m,A_{{lang,m}},A_{{lex,m}},\Delta A_{{coord-sort,m}},\rho_m^*).
$$

**结果汇总与相关文件。** 三模型结果 `{json.dumps(result['models'], ensure_ascii=False)}`；功能比较 `{json.dumps(result['functional_fingerprint'], ensure_ascii=False)}`；客户端资产 `{json.dumps(result['datasets'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2349_c8841_c8960_crossmodel_supported_task_functional_atlas.py`；结果 `tests/glm5/result/phase2349_c8841_c8960_crossmodel_supported_task_functional_atlas`。

**理论进展、问题硬伤与结论。** 这次跨模型复验只针对行为已通过的“选支持项”子域，不能修复Qwen4B的矛盾任务政策失败，也不能把不同模型的坐标号视为同一齿轮。NF4/INT8会扭曲低值，三模型量化精度不同；因此功能指纹复现最多支持“相对层深上的族条件幅值现象”，不支持微分同胚、共同流形或共同物理坐标。各模型的重要embedding/选择层/final全坐标场已发布，未完整发布的全层原始场已逐模型删除。

**下一阶段路线判断。** 观察、结构、局部因果与跨模型子域复验均已完成本大阶段。下一目标仍是语言编码机制，但最紧迫的新阶段不再是继续扩大选择题族分类，而是用行为稳定的自然生成/指令遵循任务建立同一语义图的多未来响应闭包；在那之前，不再升级“能量路由”理论。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    freeze = {"frozen_before_any_model_load": True, "model_order": list(MODEL_ORDER), "domain": "select_supported",
              "rows_per_model": 1536, "behavior_threshold": 0.70, "transfer_threshold": 0.30,
              "coordinate_advantage": 0.10, "coordinate_policy": "all local coordinates; no cross-model coordinate alignment"}
    save(OUT / "config/frozen_contract.json", freeze)
    models = {}
    for key in MODEL_ORDER:
        models[key] = run_model(key)
    datasets = [models[key]["dataset"] for key in MODEL_ORDER]
    datasets_for_catalog = [
        {**dataset, "metadata": Path(dataset["metadata"]), "binary": Path(dataset["binary"])}
        for dataset in datasets
    ]
    catalog = atlas.update_catalog(datasets_for_catalog)
    build = atlas.frontend_build()
    fingerprint = {key: {"model_label": models[key]["collection"]["model_label"],
                         "quantization": models[key]["collection"]["quantization"],
                         "behavior_accuracy": models[key]["behavior"]["overall_forced_accuracy"],
                         "qualified_families": len(models[key]["behavior"]["qualified"]),
                         "selected_qpoint": models[key]["analysis"]["gate"]["selected_qpoint"],
                         "relative_block_depth": models[key]["analysis"]["gate"]["relative_block_depth"],
                         "lockbox_minimum_accuracy": models[key]["analysis"]["gate"]["lockbox_minimum_accuracy"],
                         "coordinate_advantage_over_sorted": models[key]["analysis"]["gate"]["coordinate_advantage_over_sorted"],
                         "gate_passed": models[key]["analysis"]["gate"]["passed"]} for key in MODEL_ORDER}
    cleanup = {key: models[key]["cleanup"] for key in MODEL_ORDER}
    checks = {"sequential_models_complete": all(models[key]["all_checks_passed"] for key in MODEL_ORDER),
              "frontend_build": build["passed"], "all_raw_states_deleted": all(models[key]["checks"]["raw_state_deleted"] for key in MODEL_ORDER)}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "models": models,
              "functional_fingerprint": fingerprint, "datasets": datasets,
              "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
              "frontend_build": build, "cleanup": cleanup, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2349_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
