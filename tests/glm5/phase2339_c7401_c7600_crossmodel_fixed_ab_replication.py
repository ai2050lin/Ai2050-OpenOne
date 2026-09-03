#!/usr/bin/env python3
"""Sequential Qwen3-14B, GLM4-9B and DS7B replication of fixed-A/B coordinate-use maps."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore", message=r"MatMul8bitLt: inputs will be cast.*")


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2337 = RESULT / "phase2337_c7081_c7240_fixed_ab_output_interface"
P2338 = RESULT / "phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls"
OUT = RESULT / "phase2339_c7401_c7600_crossmodel_fixed_ab_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
ORIGINAL_ROWS = P2337 / "material/fixed_ab_rows.jsonl"
SWAPPED_ROWS = P2338 / "material/option_swapped_rows.jsonl"
PHASE = 2339
CAMPAIGN = "C7401-C7600"
FAMILIES = ("coreference_anaphora", "translation_equivalence", "attribute_binding")
PARTITIONS = ("independent_development", "independent_lockbox")
MODEL_ORDER = ("qwen14b", "glm4", "deepseek7b")
MODEL_SPECS = {
    "qwen14b": {"path": ROOT / "models/hf/Qwen3-14B", "quant": "4bit", "batch": 2, "label": "Qwen3-14B-NF4-BF16"},
    "glm4": {"path": ROOT / "models/hf/glm4-9b-chat-hf", "quant": "8bit", "batch": 5, "label": "GLM4-9B-INT8"},
    "deepseek7b": {"path": ROOT / "models/hf/deepseek-r1-distill-qwen-7b", "quant": "8bit", "batch": 6, "label": "DeepSeek-R1-Distill-Qwen-7B-INT8"},
}
EPS = 1e-12

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls as layer_control  # noqa: E402

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


def paths(key: str) -> dict[str, Path]:
    base = OUT / key
    return {
        "base": base,
        "rows": base / "material/rows.jsonl",
        "states": base / "raw/boundary_all_checkpoints.float16.npy",
        "final": base / "raw/final_coordinate_contribution.float32.npy",
        "decisions": base / "raw/decisions.float32.npy",
        "gamma": base / "raw/final_norm_gamma.float32.npy",
        "ab_weight": base / "raw/a_minus_b_weight.float32.npy",
        "progress": base / "raw/progress.json",
        "trajectory": base / "derived/layerwise_coordinate_contribution.float32.npy",
        "trajectory_progress": base / "derived/trajectory_progress.json",
        "metrics": base / "analysis/checkpoint_trajectory.jsonl",
        "final_json": base / "analysis/final.json",
    }


def model_modules(model) -> list[Any]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(("unsupported_architecture", type(model).__name__))
    embed = model.model.embed_tokens if hasattr(model.model, "embed_tokens") else model.get_input_embeddings()
    norm = model.model.norm
    return [embed, *list(model.model.layers), norm]


def compile_rows(tokenizer) -> tuple[list[dict], dict]:
    rows = []
    for condition, path in (("original", ORIGINAL_ROWS), ("option_swapped", SWAPPED_ROWS)):
        for source in read_rows(path):
            target = tokenizer.encode(" " + source["target_code"], add_special_tokens=False)
            wrong = tokenizer.encode(" " + source["wrong_code"], add_special_tokens=False)
            if len(target) != 1 or len(wrong) != 1:
                raise RuntimeError(("code_not_single_token", condition, target, wrong))
            prompt_ids = [int(x) for x in tokenizer.encode(source["future_prompt"], add_special_tokens=False)]
            rows.append({
                **source, "condition": condition,
                "model_design_index": len(rows),
                "future_prompt_ids": prompt_ids,
                "target_code_id": int(target[0]), "wrong_code_id": int(wrong[0]),
                "boundary_position": len(prompt_ids) - 1,
            })
    audit = {
        "rows": len(rows), "original_rows": sum(r["condition"] == "original" for r in rows),
        "swapped_rows": sum(r["condition"] == "option_swapped" for r in rows),
        "paired_semantic_case_count": len({r["source_case_id"] for r in rows}),
        "single_token_codes": True,
        "family_counts": {family: sum(r["family"] == family for r in rows) for family in FAMILIES},
    }
    return rows, audit


def load_model(key: str):
    spec = MODEL_SPECS[key]
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if spec["quant"] == "4bit":
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    print(f"[phase2339] loading {key} {spec['quant']}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        spec["path"], quantization_config=quant, device_map="auto", trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation="eager",
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    print(f"[phase2339] loaded {key}, class={type(model).__name__}, device={device}, gpu={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    return model, tokenizer, device


def collect(key: str, model, device, rows: list[dict]) -> dict:
    p = paths(key)
    modules = model_modules(model)
    dimension = int(model.get_input_embeddings().weight.shape[1])
    shape = (len(rows), len(modules), dimension)
    if all(p[name].exists() for name in ("states", "final", "decisions", "progress")):
        completed = int(json.loads(p["progress"].read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(p["states"], mode="r+")
        final = np.lib.format.open_memmap(p["final"], mode="r+")
        decisions = np.lib.format.open_memmap(p["decisions"], mode="r+")
    else:
        completed = 0
        p["states"].parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(p["states"], mode="w+", dtype=np.float16, shape=shape)
        final = np.lib.format.open_memmap(p["final"], mode="w+", dtype=np.float32, shape=(len(rows), dimension))
        decisions = np.lib.format.open_memmap(p["decisions"], mode="w+", dtype=np.float32, shape=(len(rows), 5))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(modules):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    batch_size = int(MODEL_SPECS[key]["batch"])
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = baseline.pad_right([row["future_prompt_ids"] for row in batch], device, pad)
                captures.clear()
                output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for q in range(len(modules)):
                    selected = torch.stack([captures[q][local, ends[local]] for local in range(len(batch))])
                    states[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                for local, row in enumerate(batch):
                    target, wrong = row["target_code_id"], row["wrong_code_id"]
                    h = captures[len(modules) - 1][local, ends[local]].float()
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
                save(p["progress"], {"completed": start + len(batch), "shape": list(shape)})
                if (start + len(batch)) % max(24, batch_size) == 0 or start + len(batch) == len(rows):
                    print(f"[phase2339 {key}] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        for value in (states, final, decisions):
            value.flush(); close_memmap(value)
    gamma = model.model.norm.weight.detach().float().cpu().numpy().astype(np.float32)
    a_id = int(rows[0]["target_code_id"] if rows[0]["target_code"] == "A" else rows[0]["wrong_code_id"])
    b_id = int(rows[0]["target_code_id"] if rows[0]["target_code"] == "B" else rows[0]["wrong_code_id"])
    ab_weight = (model.lm_head.weight[a_id].float() - model.lm_head.weight[b_id].float()).detach().cpu().numpy().astype(np.float32)
    p["gamma"].parent.mkdir(parents=True, exist_ok=True)
    np.save(p["gamma"], gamma)
    np.save(p["ab_weight"], ab_weight)
    return {
        "shape": list(shape), "layers": len(modules) - 2, "dimension": dimension,
        "a_token_id": a_id, "b_token_id": b_id,
        "rms_norm_eps": float(getattr(model.config, "rms_norm_eps", 1e-6)),
        "quantization": MODEL_SPECS[key]["quant"],
    }


def derive(key: str, rows: list[dict], collection: dict) -> dict:
    p = paths(key)
    states = np.load(p["states"], mmap_mode="r")
    final = np.load(p["final"], mmap_mode="r")
    gamma = np.load(p["gamma"])
    ab_weight = np.load(p["ab_weight"])
    shape = states.shape
    if p["trajectory"].exists() and p["trajectory_progress"].exists():
        completed_q = int(json.loads(p["trajectory_progress"].read_text(encoding="utf-8"))["completed_q"])
        trajectory = np.lib.format.open_memmap(p["trajectory"], mode="r+")
    else:
        completed_q = 0
        p["trajectory"].parent.mkdir(parents=True, exist_ok=True)
        trajectory = np.lib.format.open_memmap(p["trajectory"], mode="w+", dtype=np.float32, shape=shape)
    signs = np.asarray([1.0 if row["target_code"] == "A" else -1.0 for row in rows], dtype=np.float32)[:, None]
    for q in range(completed_q, shape[1] - 1):
        normalized = layer_control.rms_norm(states[:, q], gamma, collection["rms_norm_eps"])
        trajectory[:, q] = normalized * ab_weight[None, :] * signs
        trajectory.flush()
        save(p["trajectory_progress"], {"completed_q": q + 1, "shape": list(shape)})
        print(f"[phase2339 {key} derive] q={q}/{shape[1]-1}", flush=True)
    trajectory[:, -1] = final
    trajectory.flush()
    error = float(np.max(np.abs(trajectory[:, -1] - final)))
    for value in (states, final, trajectory):
        close_memmap(value)
    return {"shape": list(shape), "final_copy_max_abs_error": error}


def behavior(rows: list[dict], decisions: np.ndarray) -> dict:
    output = {
        "overall_forced_ab_accuracy": float(np.mean(decisions[:, 3])),
        "overall_unconstrained_exact_next_code_accuracy": float(np.mean(decisions[:, 4])),
        "max_accounting_abs_error": float(np.max(decisions[:, 2])),
        "families": {}, "all_forced_cells_pass": True,
    }
    for condition in ("original", "option_swapped"):
        output["families"][condition] = {}
        for family in FAMILIES:
            output["families"][condition][family] = {}
            for partition in PARTITIONS:
                idx = [i for i, row in enumerate(rows) if row["condition"] == condition and row["family"] == family and row["partition"] == partition]
                cell = {"rows": len(idx), "forced_ab_accuracy": float(np.mean(decisions[idx, 3])),
                        "unconstrained_exact_next_code_accuracy": float(np.mean(decisions[idx, 4]))}
                cell["passed"] = cell["forced_ab_accuracy"] >= 0.70
                output["all_forced_cells_pass"] = output["all_forced_cells_pass"] and cell["passed"]
                output["families"][condition][family][partition] = cell
    return output


def analyze(key: str, rows: list[dict], collection: dict) -> dict:
    p = paths(key)
    states = np.load(p["states"], mmap_mode="r")
    trajectory = np.load(p["trajectory"], mmap_mode="r")
    decisions = np.load(p["decisions"], mmap_mode="r")
    split = len(rows) // 2
    original_rows, swapped_rows = rows[:split], rows[split:]
    behavior_result = behavior(rows, decisions)
    metrics = []
    for q in range(states.shape[1]):
        os = states[:split, q].astype(np.float32)
        ss = states[split:, q].astype(np.float32)
        oc = trajectory[:split, q].astype(np.float32)
        sc = trajectory[split:, q].astype(np.float32)
        oa, sa = np.abs(oc), np.abs(sc)
        metrics.append({
            "qpoint": q,
            "relative_block_depth": None if q in (0, states.shape[1] - 1) else float((q - 1) / max(collection["layers"] - 1, 1)),
            "checkpoint_kind": "embedding" if q == 0 else "final_norm" if q == states.shape[1] - 1 else "block",
            "original_hidden_family": layer_control.classify(os, os, original_rows, original_rows, "family", FAMILIES),
            "swapped_hidden_family": layer_control.classify(ss, ss, swapped_rows, swapped_rows, "family", FAMILIES),
            "original_to_swapped_abs_family": layer_control.classify(oa, sa, original_rows, swapped_rows, "family", FAMILIES),
            "swapped_to_original_abs_family": layer_control.classify(sa, oa, swapped_rows, original_rows, "family", FAMILIES),
            "original_signed_target_code": layer_control.classify(oc, oc, original_rows, original_rows, "target_code", ("A", "B")),
            "swapped_signed_target_code": layer_control.classify(sc, sc, swapped_rows, swapped_rows, "target_code", ("A", "B")),
            "paired_absolute_contribution": layer_control.pair_metrics(oa, sa),
        })
        print(f"[phase2339 {key} analyze] q={q}/{states.shape[1]-1}", flush=True)
    write_rows(p["metrics"], metrics)
    qualifying = [row["qpoint"] for row in metrics if min(
        row["original_to_swapped_abs_family"]["accuracy"], row["swapped_to_original_abs_family"]["accuracy"]
    ) >= 2 / 3]
    width = max(3, int(np.ceil(collection["layers"] * 0.08)))
    onset = None
    for q in qualifying:
        if all(q + offset in qualifying for offset in range(width)):
            onset = q
            break
    final = metrics[-1]
    gate = {
        "all_forced_behavior_cells": behavior_result["all_forced_cells_pass"],
        "both_final_cross_swap_family": min(final["original_to_swapped_abs_family"]["accuracy"], final["swapped_to_original_abs_family"]["accuracy"]) >= 2 / 3,
        "final_pair_geometry": final["paired_absolute_contribution"]["median_symmetric_relative_mse"] <= 0.50,
        "sustained_width": width, "sustained_onset": onset,
    }
    gate["passed"] = all((gate["all_forced_behavior_cells"], gate["both_final_cross_swap_family"], gate["final_pair_geometry"], onset is not None))
    result = {
        "behavior": behavior_result, "checkpoint_count": len(metrics), "qualifying_qpoints": qualifying,
        "first_sustained_onset": onset,
        "first_sustained_relative_depth": None if onset is None or onset == 0 else float((onset - 1) / max(collection["layers"] - 1, 1)),
        "peak_original_to_swapped": max(metrics, key=lambda r: r["original_to_swapped_abs_family"]["accuracy"]),
        "peak_swapped_to_original": max(metrics, key=lambda r: r["swapped_to_original_abs_family"]["accuracy"]),
        "final_checkpoint": final, "gate": gate,
    }
    for value in (states, trajectory, decisions):
        close_memmap(value)
    return result


def publish(key: str, rows: list[dict]) -> list[dict]:
    p = paths(key)
    spec = MODEL_SPECS[key]
    fields = (
        (f"c7401_{key}_fixed_ab_boundary", p["states"], np.float16, "cross-model fixed-A/B HiddenState full field", "crossmodel_fixed_ab_hiddenstate_v1"),
        (f"c7402_{key}_fixed_ab_layerwise_contribution", p["trajectory"], np.float32, "cross-model RMS-normalized layerwise coordinate contribution", "crossmodel_fixed_ab_coordinate_contribution_v1"),
    )
    assets = []
    for dataset_id, path, dtype, title, schema in fields:
        source = np.load(path, mmap_mode="r")
        flat = source.reshape(-1, source.shape[-1])
        metadata = [
            {"case_id": row["case_id"], "condition": row["condition"], "family": row["family"],
             "language": row["language"], "surface": row["surface"], "unit": row["unit"],
             "state": row["state"], "partition": row["partition"], "target_code": row["target_code"], "qpoint": q}
            for row in rows for q in range(source.shape[1])
        ]
        binary = VIS / f"{dataset_id}.{np.dtype(dtype).name}.npy"
        out = atlas.create_binary(binary.name, flat.shape[0], flat.shape[1], dtype)
        out[:] = flat
        out.flush(); close_memmap(out); close_memmap(source)
        assets.append(atlas.write_metadata(
            dataset_id, f"{spec['label']} {title}", binary, metadata, spec["label"], schema,
            "cross-model functional replication", "paired original/option-swapped, three language families",
            title, {"coordinate_count": int(flat.shape[1]), "no_projection": True, "quantization": spec["quant"]},
        ))
    return assets


def run_model(key: str) -> dict:
    p = paths(key)
    if p["final_json"].exists():
        return json.loads(p["final_json"].read_text(encoding="utf-8"))
    model = tokenizer = None
    try:
        model, tokenizer, device = load_model(key)
        rows, audit = compile_rows(tokenizer)
        write_rows(p["rows"], rows)
        freeze = {
            "frozen_before_forward": True, "model": key, "quantization": MODEL_SPECS[key]["quant"],
            "families": list(FAMILIES), "cross_swap_accuracy": 2 / 3,
            "paired_symmetric_mse_max": 0.50, "forced_behavior_each_cell": 0.70,
            "sustained_width_fraction": 0.08,
        }
        save(p["base"] / "config/frozen_contract.json", freeze)
        collection = collect(key, model, device, rows)
    finally:
        if model is not None:
            del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[phase2339] released {key}, gpu={torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0:.2f}GB", flush=True)
    derived = derive(key, rows, collection)
    analysis = analyze(key, rows, collection)
    datasets = publish(key, rows)
    verification = [atlas.verify(row) for row in datasets]
    verified = all(all(value for name, value in row.items() if name != "id") for row in verification)
    checks = {
        "rows": audit["rows"] == 768, "all_coordinates": collection["shape"][0] == 768,
        "final_copy_exact": derived["final_copy_max_abs_error"] == 0, "assets_verified": verified,
    }
    result = {
        "model": key, "model_label": MODEL_SPECS[key]["label"], "freeze": freeze,
        "material_audit": audit, "collection": collection, "derived": derived, "analysis": analysis,
        "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)),
        "verification": verification, "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(p["final_json"], result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("model_replication_failed", key, checks))
    return result


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 三个异构大模型的固定A/B全坐标功能复验（{CAMPAIGN}） [{stamp}]

**测试原理、测试用例与公式。** 完整复用Phase2338冻结的三族×开发/lockbox×原始/逐题交换选项设计，但由每个模型自己的tokenizer重新编译；Qwen3-14B以NF4权重/BF16计算，GLM4-9B与DeepSeek-R1-Distill-Qwen-7B以INT8加载，全部在CUDA上严格一次只驻留一个模型。每模型768题，保存embedding、全部block、final norm的全部HiddenState坐标，并在真实RMSNorm和各模型自身A/B输出权重下计算所有检查点逐坐标贡献。跨模型不比较坐标编号，只比较族×相对深度×行为×坐标强度图谱的功能关系。

$$
r_q=\frac{{q-1}}{{L-1}},\qquad
c^{{(M)}}_{{i,q,j}}=\operatorname{{RMSNorm}}_M(h^{{(M)}}_{{i,q}})_j
\cdot(W^{{(M)}}_{{y_i^+,j}}-W^{{(M)}}_{{y_i^-,j}}).
$$

$$
\operatorname{{Replicate}}_M=B_M\land A^{{o\to s}}_{{M,final}}\ge2/3\land
A^{{s\to o}}_{{M,final}}\ge2/3\land E^{{pair}}_{{M,final}}\le0.5\land\exists\text{{连续层段}}.
$$

**结果汇总、相关文件与可视化。** 聚合结果 `{json.dumps(result, ensure_ascii=False)}`。脚本 `tests/glm5/phase2339_c7401_c7600_crossmodel_fixed_ab_replication.py`；逐模型结果和全场位于 `tests/glm5/result/phase2339_c7401_c7600_crossmodel_fixed_ab_replication/<model>`；重要HiddenState与逐坐标贡献均已发布到可视化客户端并通过哈希/shape/finite核验，客户端构建结果见聚合JSON。

**分析、理论进展、问题硬伤与结论。** 这是功能图谱复验，不是参数坐标同构：模型维度、层数、tokenizer、训练数据、量化格式不同，坐标j没有跨模型共同含义。只有行为先通过，绝对贡献图谱才可解释；signed场主要编码正确A/B符号，absolute场描述坐标使用强度。量化会改变低值坐标，特别是INT8/NF4不能用于声称精确微小值不变量；程序材料与共同选择题格式仍可能产生共享任务模板。若多模型通过，最谨慎结论是“族条件的坐标强度分配”是一类跨架构功能现象，而非发现固定语义方向、黎曼流形或语言数学理论。

**下一阶段路线判断。** 无论三模型是否全过，目标仍相同，下一阶段自动回到广度优先：把固定公共输出接口扩展到原20语言族，先做行为分层，再为行为合格族建立族×相对层×坐标强度图谱；跨模型通过的相对层段只作为冻结观察窗，不以三族结果替代普遍性。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def aggregate() -> dict:
    finals = {}
    for key in MODEL_ORDER:
        final_path = paths(key)["final_json"]
        if not final_path.exists():
            raise RuntimeError(("missing_model_run", key, str(final_path)))
        finals[key] = json.loads(final_path.read_text(encoding="utf-8"))
    datasets = [dataset for key in MODEL_ORDER for dataset in finals[key]["datasets"]]
    catalog_rows = [{**dataset, "metadata": Path(dataset["metadata"]), "binary": Path(dataset["binary"])} for dataset in datasets]
    catalog = atlas.update_catalog(catalog_rows)
    build = atlas.frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    summary = {}
    for key, record in finals.items():
        final = record["analysis"]["final_checkpoint"]
        summary[key] = {
            "model_label": record["model_label"], "layers": record["collection"]["layers"],
            "dimension": record["collection"]["dimension"],
            "behavior_forced_all_cells": record["analysis"]["behavior"]["all_forced_cells_pass"],
            "final_original_to_swap_accuracy": final["original_to_swapped_abs_family"]["accuracy"],
            "final_swap_to_original_accuracy": final["swapped_to_original_abs_family"]["accuracy"],
            "final_pair_symmetric_mse": final["paired_absolute_contribution"]["median_symmetric_relative_mse"],
            "onset_qpoint": record["analysis"]["first_sustained_onset"],
            "onset_relative_depth": record["analysis"]["first_sustained_relative_depth"],
            "scientific_gate_passed": record["analysis"]["gate"]["passed"],
        }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "model_order": list(MODEL_ORDER),
        "sequential_gpu_loading": True, "summary": summary,
        "replicated_models": [key for key in MODEL_ORDER if finals[key]["analysis"]["gate"]["passed"]],
        "replication_count": sum(finals[key]["analysis"]["gate"]["passed"] for key in MODEL_ORDER),
        "important_caveat": "coordinates are model-specific; quantized low-value magnitudes are not precision invariants",
        "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
        "frontend_build": build,
        "checks": {"all_model_runs_valid": all(finals[k]["all_checks_passed"] for k in MODEL_ORDER), "frontend_build": build["passed"]},
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_ORDER)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.model:
        print(json.dumps(run_model(args.model), ensure_ascii=False, indent=2))
    elif args.aggregate:
        print(json.dumps(aggregate(), ensure_ascii=False, indent=2))
    else:
        raise SystemExit("use --model qwen14b|glm4|deepseek7b or --aggregate")


if __name__ == "__main__":
    main()
