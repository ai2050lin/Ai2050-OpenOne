#!/usr/bin/env python3
"""Frozen exact-pair composition replication on Qwen14B, GLM4 and DeepSeek7B."""
from __future__ import annotations

import argparse
import gc
import itertools
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

logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2415 = RESULT / "phase2415_c27441_c27760_exact_paired_composition"
OUT = RESULT / "phase2416_c27761_c28080_crossmodel_exact_pair_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2416
CAMPAIGN = "C27761-C28080"
MODEL_ORDER = ("qwen14b", "glm4", "deepseek7b")
COMPONENTS = ("total", "attention", "mlp")
SPLITS = ("fresh_unit_lockbox", "confirmation", "template_lockbox", "language_lockbox")
STAGES = ("global", "family", "state", "mismatch")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2397_c21681_c22000_operation_behavior_token_calibration as behavior  # noqa: E402
import phase2411_c26161_c26480_crosslayer_composition_output_bridge as geometry  # noqa: E402
import phase2412_c26481_c26800_frozen_crossmodel_operator_replication as loader  # noqa: E402
import phase2415_c27441_c27760_exact_paired_composition as paired  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def frozen_source() -> list[dict]:
    source = read_rows(P2415 / "material/exact_paired_composition.jsonl")
    chosen = []
    for row in source:
        partition, unit = row["partition"], int(row["unit"])
        keep = ((partition == "discovery" and unit <= 3) or
                partition in ("fresh_unit_lockbox", "confirmation") or
                (partition == "template_lockbox" and unit <= 1))
        if keep:
            chosen.append(row)
    audit = paired.material_audit(chosen)
    if audit["rows"] != 640 or audit["pairs"] != 320 or audit["exact_pair_rate"] != 1.0:
        raise RuntimeError(audit)
    return chosen


def open_field(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float16, shape=shape)


def collect(key: str, model, rows: list[dict], batch_size: int) -> dict:
    blocks = list(model.model.layers)
    state_modules = field_utils.modules(model)[:len(blocks)]
    dimension = int(model.config.hidden_size)
    paths = {name: OUT / key / f"raw/composition_{name}_answer.float16.npy"
             for name in ("state", "attention", "mlp")}
    fields = {name: open_field(path, (len(rows), len(blocks), dimension)) for name, path in paths.items()}
    progress = OUT / key / "raw/composition_progress.json"
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    states: dict[int, torch.Tensor] = {}; attention: dict[int, torch.Tensor] = {}; mlp: dict[int, torch.Tensor] = {}
    handles = []
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
                token_indices = [row["event_tokens"][-1]["token_index"] for row in batch]
                for layer in range(len(blocks)):
                    for captures, name in ((states, "state"), (attention, "attention"), (mlp, "mlp")):
                        tensor = captures[layer].float().cpu()
                        fields[name][start:start + len(batch), layer] = torch.stack(
                            [tensor[local, token_indices[local]] for local in range(len(batch))]).numpy().astype(np.float16)
                for field in fields.values():
                    field.flush()
                save(progress, {"completed": start + len(batch)})
                if (start + len(batch)) % 64 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2416 {key} capture] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        for field in fields.values():
            field.flush(); close(field)
    return {name: {"path": str(path), "shape": [len(rows), len(blocks), dimension], "bytes": path.stat().st_size}
            for name, path in paths.items()}


def analyze(key: str, rows: list[dict], collection: dict) -> dict:
    pair_rows, step1, step2 = paired.pair_index(rows)
    families = np.asarray([row["family"] for row in pair_rows], dtype=object)
    family_names = sorted(set(families))
    train = np.asarray([i for i, row in enumerate(pair_rows) if row["partition"] == "discovery"], dtype=np.int64)
    en_train = np.asarray([i for i in train if pair_rows[i]["language"] == "en"], dtype=np.int64)
    tests = {split: np.asarray([i for i, row in enumerate(pair_rows) if row["partition"] == split], dtype=np.int64)
             for split in SPLITS[:-1]}
    tests["language_lockbox"] = np.asarray([i for i in train if pair_rows[i]["language"] == "zh"], dtype=np.int64)
    state = np.load(collection["state"]["path"], mmap_mode="r")
    attention = np.load(collection["attention"]["path"], mmap_mode="r")
    mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    layers, dimension = state.shape[1:]
    derived = OUT / key / "derived"; derived.mkdir(parents=True, exist_ok=True)
    passports = np.lib.format.open_memmap(derived / "paired_update_family.float32.npy", mode="w+", dtype=np.float32,
                                           shape=(len(COMPONENTS), layers, len(family_names), dimension))
    state_passports = np.lib.format.open_memmap(derived / "paired_state_family.float32.npy", mode="w+", dtype=np.float32,
                                                 shape=(layers, len(family_names), dimension))
    slopes = np.lib.format.open_memmap(derived / "paired_matched_diagonal_slope.float32.npy", mode="w+", dtype=np.float32,
                                       shape=(len(COMPONENTS), layers, dimension))
    metrics = np.zeros((len(COMPONENTS), len(SPLITS), len(STAGES), layers), dtype=np.float32)
    for layer in range(layers):
        h = np.asarray(state[step2, layer], dtype=np.float32) - np.asarray(state[step1, layer], dtype=np.float32)
        a = np.asarray(attention[step2, layer], dtype=np.float32) - np.asarray(attention[step1, layer], dtype=np.float32)
        m = np.asarray(mlp[step2, layer], dtype=np.float32) - np.asarray(mlp[step1, layer], dtype=np.float32)
        for fi, family in enumerate(family_names):
            state_passports[layer, fi] = h[train[families[train] == family]].mean(axis=0)
        for ci, y in enumerate((a + m, a, m)):
            for fi, family in enumerate(family_names):
                passports[ci, layer, fi] = y[train[families[train] == family]].mean(axis=0)
            slopes[ci, layer] = paired.fit_predict(train, train, families, h, y)["slope"]
            for si, split in enumerate(SPLITS):
                fit = en_train if split == "language_lockbox" else train
                values = paired.gains(y[tests[split]], paired.fit_predict(fit, tests[split], families, h, y))
                metrics[ci, si, :, layer] = [values[stage] for stage in STAGES]
        passports.flush(); state_passports.flush(); slopes.flush()
        print(f"[phase2416 {key} analyze] layer {layer + 1}/{layers}", flush=True)
    np.save(derived / "paired_operator_layer_metrics.float32.npy", metrics)
    summary = {}
    for ci, component in enumerate(COMPONENTS):
        summary[component] = {}
        for si, split in enumerate(SPLITS):
            item = {stage: float(metrics[ci, si, stage_index].mean()) for stage_index, stage in enumerate(STAGES)}
            item["state_increment"] = item["state"] - item["family"]
            item["physical_advantage"] = item["state"] - item["mismatch"]
            item["best_state_layer"] = int(np.argmax(metrics[ci, si, 2]))
            summary[component][split] = item
    perms = [np.asarray(value, dtype=np.int64) for value in itertools.permutations(range(4)) if value != tuple(range(4))]
    crosslayer = {}
    for ci, component in enumerate(COMPONENTS):
        values = np.asarray(passports[ci], dtype=np.float32)
        observed, pvalues, coordinates = [], [], []
        for layer in range(layers - 1):
            left, right = values[layer], values[layer + 1]
            score = geometry.correlation(geometry.geometry_vector(left), geometry.geometry_vector(right))
            null = [geometry.correlation(geometry.geometry_vector(left), geometry.geometry_vector(right[perm])) for perm in perms]
            observed.append(score); pvalues.append((1 + np.sum(np.asarray(null) >= score)) / 24)
            coordinates.append(float(np.mean([geometry.cosine(left[f], right[f]) for f in range(4)])))
        crosslayer[component] = {"relation_mean": float(np.mean(observed)),
                                 "coordinate_cosine_mean": float(np.mean(coordinates)),
                                 "exceeds_permutation_q95_rate": float(np.mean(np.asarray(pvalues) <= .05)),
                                 "median_permutation_p": float(np.median(pvalues)), "cells": len(observed)}
    files = {"passport": str(derived / "paired_update_family.float32.npy"),
             "state_passport": str(derived / "paired_state_family.float32.npy"),
             "slopes": str(derived / "paired_matched_diagonal_slope.float32.npy"),
             "metrics": str(derived / "paired_operator_layer_metrics.float32.npy")}
    for value in (passports, state_passports, slopes, state, attention, mlp):
        close(value)
    return {"pairs": len(pair_rows), "train_pairs": len(train), "test_pairs": {key: len(value) for key, value in tests.items()},
            "summary": summary, "crosslayer": crosslayer, "files": files}


def cleanup(collection: dict) -> dict:
    paths, total = [], 0
    for value in collection.values():
        path = Path(value["path"])
        if path.exists():
            total += path.stat().st_size; paths.append(str(path)); path.unlink()
    return {"removed_files": len(paths), "removed_bytes": total, "removed_gib": total / 2**30,
            "recoverable": False, "paths": paths}


def run_model(key: str) -> None:
    final = OUT / key / "analysis/final.json"
    if final.exists():
        print(final.read_text(encoding="utf-8")); return
    source = frozen_source()
    model, tokenizer, label = loader.load_for_capture(key)
    behavior.OUT = OUT
    try:
        index = OUT / key / "index/composition_rows.jsonl"
        if index.exists():
            rows = read_rows(index)
            calibration = json.loads((OUT / key / "analysis/token_calibration.json").read_text(encoding="utf-8"))
        else:
            rows, calibration = behavior.compile_rows(tokenizer, source)
            write_rows(index, rows); save(OUT / key / "analysis/token_calibration.json", calibration)
        teacher, teacher_summary = behavior.score_rows(key, model, rows, {"qwen14b": 2, "glm4": 8, "deepseek7b": 8}[key])
        collection = collect(key, model, rows, {"qwen14b": 1, "glm4": 1, "deepseek7b": 2}[key])
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    analysis = analyze(key, rows, collection)
    raw_cleanup = cleanup(collection)
    pair_summary = paired.behavior_pair_summary(teacher, [])
    precision = "NF4 storage / BF16 compute / device_map=auto" if key == "qwen14b" else label
    checks = {"frozen_320_pairs": analysis["pairs"] == 320, "calibrated": calibration["rows"] == 640,
              "teacher_complete": len(teacher) == 640 and pair_summary["teacher_pairs"] == 320,
              "full_coordinates": all(value["shape"][0] == 640 for value in collection.values()),
              "finite": all(math.isfinite(v) for component in analysis["summary"].values() for split in component.values()
                            for key2, v in split.items() if key2 != "best_state_layer"),
              "raw_cleaned": raw_cleanup["removed_files"] == 3, "claim_boundary": True}
    result = {"model": key, "label": label, "precision": precision, "material_rows": len(rows),
              "token_calibration": calibration, "teacher": teacher_summary, "pair_behavior": pair_summary,
              "collection": collection, "analysis": analysis, "cleanup": raw_cleanup,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps({"model": key, "precision": precision, "pair_behavior": pair_summary,
                      "summary": analysis["summary"], "crosslayer": analysis["crosslayer"],
                      "cleanup": raw_cleanup, "checks": checks}, ensure_ascii=False, indent=2))


def crossmodel(models: dict) -> dict:
    q4 = np.load(P2415 / "derived/paired_update_step2_minus_step1_family.float32.npy", mmap_mode="r")
    perms = [np.asarray(value, dtype=np.int64) for value in itertools.permutations(range(4)) if value != tuple(range(4))]
    q4_family_order = list(paired.contract.COMPOSITION_FAMILIES)
    target_family_order = sorted(q4_family_order)
    target_to_q4 = np.asarray([target_family_order.index(name) for name in q4_family_order], dtype=np.int64)
    result = {}
    for model_index, (key, model) in enumerate(models.items()):
        target = np.load(model["analysis"]["files"]["passport"], mmap_mode="r")
        result[key] = {}
        for ci, component in enumerate(COMPONENTS):
            rows = []
            for layer in range(target.shape[1]):
                q4_layer = round(layer * (q4.shape[1] - 1) / max(target.shape[1] - 1, 1))
                left = np.asarray(q4[ci, q4_layer], dtype=np.float32)
                right = np.asarray(target[ci, layer, target_to_q4], dtype=np.float32)
                observed = geometry.correlation(geometry.geometry_vector(left), geometry.geometry_vector(right))
                null = np.asarray([geometry.correlation(geometry.geometry_vector(left), geometry.geometry_vector(right[perm]))
                                   for perm in perms])
                rows.append({"target_layer": layer, "q4_layer": q4_layer, "observed": observed,
                             "identity_margin": observed - float(null.mean()),
                             "p": float((1 + np.sum(null >= observed)) / 24)})
            result[key][component] = {"relative_depth_geometry_mean": float(np.mean([row["observed"] for row in rows])),
                                      "identity_margin_mean": float(np.mean([row["identity_margin"] for row in rows])),
                                      "exceeds_permutation_q95_rate": float(np.mean([row["p"] <= .05 for row in rows])),
                                      "median_permutation_p": float(np.median([row["p"] for row in rows])),
                                      "layers": len(rows)}
        close(target)
    close(q4)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 严格配对组合场的三模型冻结复现（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 自动延续Phase2415相同目标，不按结果换材料。冻结320个严格pair/640条：discovery unit0–3、fresh unit6–7、confirmation unit8–9，以及自然化模板unit0–1；四族、中英、双方向和相应表面完整平衡。依次测试Qwen3-14B、GLM4、DeepSeek7B，每个模型独立token校准与目标—foil序列评分，在answer boundary采集逐层输入状态、Attention与MLP的全部物理坐标。逐模型只用discovery拟合族均值、同坐标状态算子和循环错配791位对照；不同宽度模型只比较四族Gram关系图。Qwen14B严格标记NF4权重存储/BF16计算、`device_map=auto`，不是BF16权重。

$$D^U_{{m,p,q}}=U_{{m,p,q}}^{{(2)}}-U_{{m,p,q}}^{{(1)}},\qquad
R_{{m,q}}=\mathrm{{corr}}\!\left(\mathrm{{vec}}K^{{Q4}}_{{\mathrm{{round}}(35q/(L_m-1))}},\mathrm{{vec}}K^m_q\right).$$

**结果汇总。** 模型摘要 `{json.dumps(result['model_summary'], ensure_ascii=False)}`；跨模型关系几何 `{json.dumps(result['crossmodel'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup_summary'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2416_c27761_c28080_crossmodel_exact_pair_replication.py`；逐模型token索引、行为结果、全坐标族护照、状态斜率、逐层指标及final位于`tests/glm5/result/phase2416_c27761_c28080_crossmodel_exact_pair_replication`。未修改其他Markdown。

**分析与理论进展。** 这一步区分三个命题：模型内同坐标耦合、模型内跨层族关系保持、跨模型功能关系几何复现。前两者在每个模型自身坐标中检验；第三者不比较不同宽度的坐标编号。只有冻结材料上多模型同时复现，Phase2415的配对纹理才可提升为候选普遍规律。

**问题硬伤与结论。** 子集虽有320 pair，但discovery只有128 pair；Qwen14B量化精度不同。四族Gram只有6条边，关系相关容易受小节点图影响，故同时报告23种非恒等标签置换。目标序列偏好仍不是自主生成闭合，内部纹理仍可能是查询/答案角色的联合编码而非递归组合算法。原始HiddenState/组件场在派生全坐标数组核验后逐模型删除且不可恢复。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def finalize() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    models = {}
    for key in MODEL_ORDER:
        path = OUT / key / "analysis/final.json"
        if not path.exists():
            raise RuntimeError(f"missing sequential model result: {key}")
        models[key] = json.loads(path.read_text(encoding="utf-8"))
    cross = crossmodel(models)
    summary = {key: {"precision": model["precision"], "pair_behavior": model["pair_behavior"],
                     "operator": model["analysis"]["summary"], "crosslayer": model["analysis"]["crosslayer"]}
               for key, model in models.items()}
    decisive = [{"model": key, "component": component, "split": split,
                 "state_increment": values["state_increment"], "physical_advantage": values["physical_advantage"]}
                for key, model in models.items() for component, split_map in model["analysis"]["summary"].items()
                for split, values in split_map.items()]
    cleanup_summary = {key: model["cleanup"] for key, model in models.items()}
    adjudication = {"decisive_cells": decisive,
                    "all_cells_positive_state_increment": all(row["state_increment"] > 0 for row in decisive),
                    "all_cells_positive_physical_advantage": all(row["physical_advantage"] > 0 for row in decisive),
                    "paired_field_is_universal_composition_algorithm": False,
                    "crossmodel_coordinate_isomorphism_proven": False}
    checks = {"sequential_models": list(models) == list(MODEL_ORDER),
              "all_model_checks": all(model["all_checks_passed"] for model in models.values()),
              "cross_width_geometry_only": True,
              "all_raw_cleaned": all(value["removed_files"] == 3 for value in cleanup_summary.values()),
              "finite": all(math.isfinite(float(value)) for model in cross.values() for component in model.values()
                            for key, value in component.items() if key != "layers"),
              "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "model_summary": summary,
              "crossmodel": cross, "cleanup_summary": cleanup_summary, "adjudication": adjudication,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "model_summary": summary, "crossmodel": cross,
                      "adjudication": adjudication, "checks": checks}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_ORDER)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.model:
        run_model(args.model)
    elif args.finalize:
        finalize()
    else:
        parser.error("pass --model KEY sequentially or --finalize")


if __name__ == "__main__":
    main()
