#!/usr/bin/env python3
"""Collect Qwen3-4B full-coordinate fields for the bilingual factorial atlas."""
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
OUT = RESULT / "phase2345_c8361_c8480_qwen4b_bilingual_factorial_full_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MATERIAL = P2344 / "material/bilingual_factorial_fixed_code.jsonl"
STATES = OUT / "raw/boundary_all_checkpoints.float16.npy"
DECISIONS = OUT / "raw/decisions.float32.npy"
GAMMA = OUT / "raw/final_norm_gamma.float32.npy"
CODE_WEIGHTS = OUT / "raw/codebook_weight_vectors.float32.npz"
PROGRESS = OUT / "raw/boundary_progress.json"
TRAJECTORY = OUT / "derived/layerwise_coordinate_contribution.float32.npy"
TRAJECTORY_PROGRESS = OUT / "derived/trajectory_progress.json"
TOKEN_STATES = OUT / "raw/reference_all_token_all_checkpoints.float16.npy"
TOKEN_INDEX = OUT / "index/reference_all_token_rows.jsonl"
TOKEN_PROGRESS = OUT / "raw/reference_token_progress.json"
PHASE = 2345
CAMPAIGN = "C8361-C8480"
QPOINTS_TO_PUBLISH = (0, 23, 37)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2338_c7241_c7400_fixed_ab_layer_trajectory_controls as layer_control  # noqa: E402
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


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def collect_boundary(model, device, rows: list[dict], batch_size: int = 16) -> dict:
    module_list = modules(model)
    dimension = int(model.config.hidden_size)
    shape = (len(rows), len(module_list), dimension)
    if STATES.exists() and DECISIONS.exists() and PROGRESS.exists():
        completed = int(json.loads(PROGRESS.read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(STATES, mode="r+")
        decisions = np.lib.format.open_memmap(DECISIONS, mode="r+")
    else:
        completed = 0
        STATES.parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(STATES, mode="w+", dtype=np.float16, shape=shape)
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
                model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                selected_final = None
                for q in range(len(module_list)):
                    selected = torch.stack([captures[q][local, ends[local]] for local in range(len(batch))])
                    states[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                    if q == len(module_list) - 1:
                        selected_final = selected
                logits = model.lm_head(selected_final).float()
                for local, row in enumerate(batch):
                    target, wrong = row["target_code_id"], row["wrong_code_id"]
                    margin = logits[local, target] - logits[local, wrong]
                    weight = model.lm_head.weight[target].float() - model.lm_head.weight[wrong].float()
                    reconstructed = (selected_final[local].float() * weight).sum()
                    decisions[start + local] = [
                        float(margin.item()), float(reconstructed.item()),
                        float(abs(margin.item() - reconstructed.item())), float(margin.item() > 0),
                        float(int(torch.argmax(logits[local]).item()) == target),
                    ]
                states.flush(); decisions.flush()
                save(PROGRESS, {"completed": start + len(batch), "shape": list(shape)})
                if (start + len(batch)) % 256 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2345 boundary] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        states.flush(); decisions.flush()
        close_memmap(states); close_memmap(decisions)
    np.save(GAMMA, model.model.norm.weight.detach().float().cpu().numpy().astype(np.float32))
    weights = {}
    for codebook, (left, right) in contract.CODEBOOKS.items():
        left_id = next(row["target_code_id"] if row["target_code"] == left else row["wrong_code_id"]
                       for row in rows if row["codebook"] == codebook and left in (row["target_code"], row["wrong_code"]))
        right_id = next(row["target_code_id"] if row["target_code"] == right else row["wrong_code_id"]
                        for row in rows if row["codebook"] == codebook and right in (row["target_code"], row["wrong_code"]))
        weights[codebook] = (model.lm_head.weight[left_id].float() - model.lm_head.weight[right_id].float()).detach().cpu().numpy().astype(np.float32)
    CODE_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CODE_WEIGHTS, **weights)
    return {"shape": list(shape), "layers": len(module_list) - 2, "dimension": dimension,
            "rms_norm_eps": float(model.config.rms_norm_eps), "batch_size": batch_size}


def collect_reference_tokens(model, device, rows: list[dict]) -> dict:
    references = [row for row in rows if row["all_token_reference"]]
    checkpoint_count = len(modules(model))
    total_rows = sum(len(row["future_prompt_ids"]) * checkpoint_count for row in references)
    if TOKEN_STATES.exists() and TOKEN_PROGRESS.exists() and TOKEN_INDEX.exists():
        progress = json.loads(TOKEN_PROGRESS.read_text(encoding="utf-8"))
        if progress.get("completed_references") == len(references):
            return {"reference_prompts": len(references), "flat_rows": total_rows,
                    "shape": list(np.load(TOKEN_STATES, mmap_mode="r").shape), "resumed_complete": True}
    TOKEN_STATES.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_INDEX.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(TOKEN_STATES, mode="w+", dtype=np.float16,
                                      shape=(total_rows, int(model.config.hidden_size)))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(modules(model)):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    metadata = []
    cursor = 0
    try:
        with torch.inference_mode():
            for ref_index, row in enumerate(references):
                ids = torch.tensor([row["future_prompt_ids"]], dtype=torch.long, device=device)
                positions = torch.arange(ids.shape[1], device=device)[None, :]
                captures.clear()
                model.model(input_ids=ids, position_ids=positions, use_cache=False, return_dict=True)
                for token_index in range(ids.shape[1]):
                    for q in range(checkpoint_count):
                        field[cursor] = captures[q][0, token_index].float().cpu().numpy().astype(np.float16)
                        metadata.append({
                            "case_id": row["case_id"], "reference_index": ref_index, "token_index": token_index,
                            "token_id": int(row["future_prompt_ids"][token_index]), "qpoint": q,
                            "family": row["family"], "macrotype": row["macrotype"], "language": row["language"],
                            "task": row["task"], "surface": row["surface"], "lexical_set": row["lexical_set"],
                            "codebook": row["codebook"], "partition": row["partition"], "token_role": "all_prompt_tokens",
                        })
                        cursor += 1
                field.flush()
                save(TOKEN_PROGRESS, {"completed_references": ref_index + 1, "cursor": cursor, "shape": list(field.shape)})
                print(f"[phase2345 all-token] {ref_index + 1}/{len(references)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush(); close_memmap(field)
    io.write_rows(TOKEN_INDEX, metadata)
    return {"reference_prompts": len(references), "flat_rows": total_rows,
            "shape": [total_rows, int(model.config.hidden_size)], "resumed_complete": False}


def derive(rows: list[dict], collection: dict) -> dict:
    states = np.load(STATES, mmap_mode="r")
    gamma = np.load(GAMMA).astype(np.float32)
    code_weights = np.load(CODE_WEIGHTS)
    if TRAJECTORY.exists() and TRAJECTORY_PROGRESS.exists():
        completed_q = int(json.loads(TRAJECTORY_PROGRESS.read_text(encoding="utf-8"))["completed_q"])
        trajectory = np.lib.format.open_memmap(TRAJECTORY, mode="r+")
    else:
        completed_q = 0
        TRAJECTORY.parent.mkdir(parents=True, exist_ok=True)
        trajectory = np.lib.format.open_memmap(TRAJECTORY, mode="w+", dtype=np.float32, shape=states.shape)
    row_weights = np.stack([
        code_weights[row["codebook"]] * (1.0 if row["target_code"] == row["code_first"] else -1.0)
        for row in rows
    ]).astype(np.float32)
    chunk = 512
    for q in range(completed_q, states.shape[1]):
        for start in range(0, len(rows), chunk):
            value = states[start:start + chunk, q].astype(np.float32)
            if q != states.shape[1] - 1:
                value = layer_control.rms_norm(value, gamma, collection["rms_norm_eps"])
            trajectory[start:start + len(value), q] = value * row_weights[start:start + len(value)]
        trajectory.flush()
        save(TRAJECTORY_PROGRESS, {"completed_q": q + 1, "shape": list(states.shape)})
        print(f"[phase2345 derive] {q}/{states.shape[1] - 1}", flush=True)
    decisions = np.load(DECISIONS, mmap_mode="r")
    final_sums = np.zeros(len(rows), dtype=np.float64)
    for start in range(0, len(rows), chunk):
        final_sums[start:start + chunk] = trajectory[start:start + chunk, -1].astype(np.float64).sum(axis=1)
    residual = float(np.max(np.abs(final_sums - decisions[:, 1].astype(np.float64))))
    shape = list(states.shape)
    code_weights.close()
    close_memmap(states); close_memmap(trajectory); close_memmap(decisions)
    return {"shape": shape, "max_final_reconstruction_copy_residual": residual}


def behavior(rows: list[dict]) -> dict:
    decisions = np.load(DECISIONS, mmap_mode="r")
    result = {
        "overall_forced_code_accuracy": float(np.mean(decisions[:, 3])),
        "overall_unconstrained_exact_next_code_accuracy": float(np.mean(decisions[:, 4])),
        "max_logit_accounting_abs_error": float(np.max(decisions[:, 2])),
        "families": {}, "qualified": [], "qualified_macrotypes": [],
    }
    for family in contract.FAMILIES:
        core_cells = {}
        passed = True
        for language in contract.LANGUAGES:
            for codebook in contract.CODEBOOKS:
                for partition in contract.PARTITIONS:
                    idx = [i for i, row in enumerate(rows) if row["family"] == family and row["language"] == language
                           and row["codebook"] == codebook and row["partition"] == partition]
                    accuracy = float(np.mean(decisions[idx, 3]))
                    core_cells[f"{language}:{codebook}:{partition}"] = {"rows": len(idx), "accuracy": accuracy,
                                                                       "passed": accuracy >= 0.70}
                    passed = passed and accuracy >= 0.70
        factor_cells = {}
        for key in ("language", "lexical_set", "task", "surface", "codebook", "condition", "partition"):
            factor_cells[key] = {}
            for value in sorted({row[key] for row in rows}):
                idx = [i for i, row in enumerate(rows) if row["family"] == family and row[key] == value]
                factor_cells[key][value] = {"rows": len(idx), "accuracy": float(np.mean(decisions[idx, 3]))}
        result["families"][family] = {"qualified": passed, "macrotype": contract.MACROTYPE[family],
                                       "core_cells": core_cells, "factor_cells": factor_cells}
        if passed:
            result["qualified"].append(family)
    result["qualified_macrotypes"] = sorted({contract.MACROTYPE[family] for family in result["qualified"]})
    close_memmap(decisions)
    return result


def base_metadata(row: dict, qpoint: int | None = None) -> dict:
    value = {key: row[key] for key in (
        "case_id", "family", "macrotype", "language", "lexical_set", "task", "surface", "codebook",
        "condition", "partition", "unit", "state", "target_code", "selected_truth_value"
    )}
    if qpoint is not None:
        value["qpoint"] = qpoint
        value["checkpoint_kind"] = "embedding" if qpoint == 0 else "final_norm" if qpoint == 37 else "block"
    return value


def publish(rows: list[dict]) -> list[dict]:
    datasets = []
    specs = (
        ("c8361_qwen4b_bilingual_factorial_key_checkpoint_hiddenstate", STATES, np.float16,
         "Bilingual factorial embedding/q23/final HiddenState fields", "bilingual_factorial_key_hiddenstate_v1"),
        ("c8362_qwen4b_bilingual_factorial_key_checkpoint_contribution", TRAJECTORY, np.float32,
         "Bilingual factorial embedding/q23/final output-coordinate contribution fields", "bilingual_factorial_key_contribution_v1"),
    )
    for dataset_id, source_path, dtype, title, schema in specs:
        source = np.load(source_path, mmap_mode="r")
        binary = VIS / f"{dataset_id}.{np.dtype(dtype).name}.npy"
        out = atlas.create_binary(binary.name, len(rows) * len(QPOINTS_TO_PUBLISH), source.shape[-1], dtype)
        metadata = []
        cursor = 0
        for qpoint in QPOINTS_TO_PUBLISH:
            out[cursor:cursor + len(rows)] = source[:, qpoint]
            metadata.extend(base_metadata(row, qpoint) for row in rows)
            cursor += len(rows)
        out.flush(); close_memmap(out); close_memmap(source)
        datasets.append(atlas.write_metadata(
            dataset_id, title, binary, metadata, "Qwen3-4B-FP16", schema,
            "observational factorial full-coordinate field; not a causal mechanism",
            "12 families x bilingual x lexical x task x surface x codebook x semantic state x option swap",
            "all 2560 model-local coordinates at embedding, frozen q23 and final norm",
            {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": 2560, "no_topk": True,
             "qpoints": list(QPOINTS_TO_PUBLISH)},
        ))
    token_source = np.load(TOKEN_STATES, mmap_mode="r")
    token_metadata = io.read_rows(TOKEN_INDEX)
    token_id = "c8363_qwen4b_bilingual_factorial_reference_all_token_all_checkpoint"
    token_binary = VIS / f"{token_id}.float16.npy"
    token_out = atlas.create_binary(token_binary.name, token_source.shape[0], token_source.shape[1], np.float16)
    token_out[:] = token_source
    token_out.flush(); close_memmap(token_out); close_memmap(token_source)
    datasets.append(atlas.write_metadata(
        token_id, "Bilingual factorial reference prompts: every token, checkpoint and coordinate",
        token_binary, token_metadata, "Qwen3-4B-FP16", "bilingual_factorial_reference_all_token_v1",
        "observational token-resolved reference field; not a family mechanism",
        "48 frozen reference prompts: every family x language x task at unit12/lex0/natural/AB/state0/original",
        "raw embedding and HiddenState activations for every prompt token, all 38 checkpoints and all 2560 coordinates",
        {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": 2560, "no_topk": True,
         "includes_embedding": True, "all_tokens": True, "all_checkpoints": True},
    ))
    return datasets


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: Qwen3-4B十二族双语正交全坐标场与全token参考场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 按Phase2344冻结合同，在CUDA上以Qwen3-4B FP16顺序处理全部 `{result['collection']['shape'][0]}` 条材料；每条保存输出边界的embedding、36层block和final norm，共38×2560坐标。对48条冻结关键材料另存每个prompt token×38检查点×2560坐标。模型主体前向后只在边界状态上计算词表logit，避免把无关token的全词表logit当研究对象。AB/CD输出码逐行使用各自真实权重，保存完整逐层逐坐标贡献，不做Top-K、PCA或投影。

$$
h_{{i,q}}=H_q(x_i,t_{{boundary}}),\qquad
c_{{i,q,j}}=\operatorname{{RMSNorm}}(h_{{i,q}})_j
(W_{{y_i^+,j}}-W_{{y_i^-,j}}).
$$

$$
B_f=\bigwedge_{{l\in L,o\in O,p\in P}}
[\operatorname{{Acc}}_{{f,l,o,p}}\ge0.70].
$$

**结果汇总与相关文件。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；全token `{json.dumps(result['all_token_collection'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；派生 `{json.dumps(result['derived'], ensure_ascii=False)}`；客户端全参数热力图 `{json.dumps(result['datasets'], ensure_ascii=False)}`；验证 `{json.dumps(result['verification'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2345_c8361_c8480_qwen4b_bilingual_factorial_full_field.py`；结果 `tests/glm5/result/phase2345_c8361_c8480_qwen4b_bilingual_factorial_full_field`。

**理论进展、问题硬伤与结论。** 本Phase只建立新的大分母与行为授权，不用族分类反推机制。行为资格要求每族在语言×输出码×四分区的聚合单元全部达到0.70；未合格族仍保留为形式/失败对照。全token场只覆盖48条冻结参考材料，不代表36,864条全部token；大规模主场覆盖的是边界token。FP16低值受Phase2332数值边界约束；严格占位符提高跨语言可比性但降低生态自然度。重要embedding、q23、final和参考全token场已进入客户端；未发布的全层原始边界场暂留给下一Phase的全层路线竞争，竞赛结束后必须清理或发布。

**下一阶段路线判断。** 目标相同，自动在同一冻结数据上进行基础路线竞争：raw H、|H|、逐行标准化、真实signed/absolute contribution、坐标置乱输出权重、排序分布、析因残差与语言配对差异；必须在fresh_confirmation选层、fresh_lockbox裁决，并同时跨词汇、任务、表述、语言、输出码和选项顺序。
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
    rows = io.read_rows(MATERIAL)
    freeze = json.loads((P2344 / "config/frozen_contract.json").read_text(encoding="utf-8"))
    save(OUT / "config/frozen_contract.json", freeze)
    model = None
    try:
        model, _tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        collection = collect_boundary(model, device, rows)
        all_token = collect_reference_tokens(model, device, rows)
    finally:
        if model is not None:
            model_utils.release_model(model)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    behavior_result = behavior(rows)
    derived = derive(rows, collection)
    datasets = publish(rows)
    verification = [atlas.verify(dataset) for dataset in datasets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    checks = {
        "rows": collection["shape"] == [36864, 38, 2560],
        "full_coordinates": derived["shape"] == [36864, 38, 2560],
        "reference_prompts": all_token["reference_prompts"] == 48,
        "reference_all_coordinates": all_token["shape"][1] == 2560,
        "assets_verified": verified,
        "frontend_build": build["passed"],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "collection": collection,
        "all_token_collection": all_token, "behavior": behavior_result, "derived": derived,
        "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)),
        "verification": verification, "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
        "frontend_build": build, "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2345_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
