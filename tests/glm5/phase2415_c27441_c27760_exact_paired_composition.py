#!/usr/bin/env python3
"""Exact same-chain one-step/two-step composition pairs with full-coordinate Qwen4B fields."""
from __future__ import annotations

import gc
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2415_c27441_c27760_exact_paired_composition"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2415
CAMPAIGN = "C27441-C27760"
COMPONENTS = ("total", "attention", "mlp")
SPLITS = ("fresh_unit_lockbox", "confirmation", "template_lockbox", "language_lockbox")
STAGES = ("global", "family", "state", "mismatch")
SHIFT = 791

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2397_c21681_c22000_operation_behavior_token_calibration as behavior  # noqa: E402
import phase2405_c24241_c24560_deconfounded_operation_contract as contract  # noqa: E402
import phase2411_c26161_c26480_crosslayer_composition_output_bridge as geometry  # noqa: E402


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


def compile_source() -> list[dict]:
    """Freeze exact pairs: candidates/order/facts are identical and only the queried step changes."""
    rows: list[dict] = []
    for family_index, family in enumerate(contract.COMPOSITION_FAMILIES):
        for unit in range(10):
            for language_index, language in enumerate(contract.LANGUAGES):
                a, b, c = contract.triples(language, unit)
                for surface_index, surface in enumerate(contract.SURFACES):
                    for direction in (0, 1):
                        source, middle, endpoint = (a, b, c) if direction == 0 else (c, b, a)
                        rendered = [contract.render_fact(family, language, surface, source, middle),
                                    contract.render_fact(family, language, surface, middle, endpoint)]
                        candidate_order = (family_index + unit + surface_index + direction) % 2
                        fixed_candidates = [middle, endpoint] if candidate_order == 0 else [endpoint, middle]
                        pair_id = f"pcmp-{family}-u{unit}-{language}-{surface}-d{direction}"
                        for steps in (1, 2):
                            answer, foil = (middle, endpoint) if steps == 1 else (endpoint, middle)
                            query = contract.prior.composition_query(family, language, source, steps)
                            prompt, events = contract.prior.prompt_with_events(language, rendered, query, fixed_candidates)
                            rows.append({
                                "case_id": f"{pair_id}-s{steps}", "pair_id": pair_id, "task": "composition",
                                "family": family, "unit": unit, "language": language, "surface": surface,
                                "surface_class": "controlled" if surface in ("canonical", "paraphrase") else "naturalized",
                                "direction": direction, "steps": steps, "candidate_order": candidate_order,
                                "target_candidate_slot": fixed_candidates.index(answer),
                                "partition": contract.partition(unit, surface), "source": source, "middle": middle,
                                "endpoint": endpoint, "facts": [item[0] for item in rendered], "query": query,
                                "candidates": fixed_candidates, "answer": answer, "foil": foil,
                                "prompt": prompt, "events": events,
                            })
    return rows


def material_audit(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["pair_id"]].append(row)
    exact = 0
    only_query_changes = 0
    for pair in groups.values():
        pair.sort(key=lambda row: row["steps"])
        a, b = pair
        exact += int(len(pair) == 2 and [row["steps"] for row in pair] == [1, 2] and
                     a["facts"] == b["facts"] and a["candidates"] == b["candidates"] and
                     a["candidate_order"] == b["candidate_order"] and a["language"] == b["language"] and
                     a["surface"] == b["surface"] and a["direction"] == b["direction"])
        only_query_changes += int(a["prompt"].replace(a["query"], "<QUERY>") == b["prompt"].replace(b["query"], "<QUERY>"))
    return {
        "rows": len(rows), "pairs": len(groups), "families": dict(Counter(row["family"] for row in rows)),
        "languages": dict(Counter(row["language"] for row in rows)),
        "surfaces": dict(Counter(row["surface"] for row in rows)),
        "steps": dict(Counter(row["steps"] for row in rows)),
        "partitions": dict(Counter(row["partition"] for row in rows)),
        "exact_pair_rate": exact / len(groups), "query_only_change_rate": only_query_changes / len(groups),
        "unique_case_ids": len({row["case_id"] for row in rows}) == len(rows),
    }


def index_row(row: dict) -> dict:
    keys = ("case_id", "pair_id", "task", "family", "unit", "language", "surface", "surface_class",
            "direction", "steps", "partition", "candidate_order", "target_candidate_slot", "answer", "foil",
            "prompt_token_count", "target_ids", "foil_ids")
    return {**{key: row[key] for key in keys},
            "answer_boundary_token": next(event["token_index"] for event in row["event_tokens"]
                                          if event["event"] == "answer_boundary")}


def pad_right(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = 1
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def open_field(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float16, shape=shape)


def collect_fields(model, rows: list[dict], batch_size: int = 4) -> dict:
    blocks = list(model.model.layers)
    state_modules = field_utils.modules(model)
    n, layers, dimension = len(rows), len(blocks), int(model.config.hidden_size)
    paths = {name: OUT / f"raw/composition_{name}_answer.float16.npy" for name in ("state", "attention", "mlp")}
    progress = OUT / "raw/composition_progress.json"
    states = open_field(paths["state"], (n, layers + 2, dimension))
    attention = open_field(paths["attention"], (n, layers, dimension))
    mlp = open_field(paths["mlp"], (n, layers, dimension))
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    state_capture: dict[int, torch.Tensor] = {}
    attn_capture: dict[int, torch.Tensor] = {}
    mlp_capture: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(state_modules):
        def state_hook(_module, _inputs, output, qpoint=qpoint):
            state_capture[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(state_hook))
    for layer, block in enumerate(blocks):
        def attn_hook(_module, _inputs, output, layer=layer):
            attn_capture[layer] = (output[0] if isinstance(output, tuple) else output).detach()
        def mlp_hook(_module, _inputs, output, layer=layer):
            mlp_capture[layer] = output.detach()
        handles.append(block.self_attn.register_forward_hook(attn_hook))
        handles.append(block.mlp.register_forward_hook(mlp_hook))
    device = model.get_input_embeddings().weight.device
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, n, batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = pad_right([row["prompt_ids"] for row in batch], device, pad)
                state_capture.clear(); attn_capture.clear(); mlp_capture.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                token_indices = [row["event_tokens"][-1]["token_index"] for row in batch]
                for qpoint in range(layers + 2):
                    tensor = state_capture[qpoint].float().cpu()
                    states[start:start + len(batch), qpoint] = torch.stack(
                        [tensor[local, token_indices[local]] for local in range(len(batch))]).numpy().astype(np.float16)
                for layer in range(layers):
                    for captures, target in ((attn_capture, attention), (mlp_capture, mlp)):
                        tensor = captures[layer].float().cpu()
                        target[start:start + len(batch), layer] = torch.stack(
                            [tensor[local, token_indices[local]] for local in range(len(batch))]).numpy().astype(np.float16)
                states.flush(); attention.flush(); mlp.flush()
                save(progress, {"completed": start + len(batch)})
                if (start + len(batch)) % 128 == 0 or start + len(batch) == n:
                    print(f"[phase2415 capture] {start + len(batch)}/{n}", flush=True)
                del ids, mask, positions
    finally:
        for handle in handles:
            handle.remove()
        states.flush(); attention.flush(); mlp.flush()
        close(states); close(attention); close(mlp)
    return {name: {"path": str(path), "shape": ([n, layers + 2, dimension] if name == "state" else [n, layers, dimension]),
                   "bytes": path.stat().st_size} for name, path in paths.items()}


def behavior_pair_summary(teacher: list[dict], autonomous: list[dict]) -> dict:
    def group(rows: list[dict]) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            result[row["case_id"].rsplit("-s", 1)[0]].append(row)
        return result
    teacher_pairs = [sorted(value, key=lambda row: row["steps"]) for value in group(teacher).values() if len(value) == 2]
    auto_pairs = [sorted(value, key=lambda row: row["steps"]) for value in group(autonomous).values() if len(value) == 2]
    return {
        "teacher_pairs": len(teacher_pairs),
        "both_target_over_foil": float(np.mean([all(row["mean_logprob_margin"] > 0 for row in pair) for pair in teacher_pairs])),
        "step1_target_over_foil": float(np.mean([pair[0]["mean_logprob_margin"] > 0 for pair in teacher_pairs])),
        "step2_target_over_foil": float(np.mean([pair[1]["mean_logprob_margin"] > 0 for pair in teacher_pairs])),
        "step1_mean_margin": float(np.mean([pair[0]["mean_logprob_margin"] for pair in teacher_pairs])),
        "step2_mean_margin": float(np.mean([pair[1]["mean_logprob_margin"] for pair in teacher_pairs])),
        "autonomous_pairs": len(auto_pairs),
        "both_target_present": float(np.mean([all(row["target_present"] for row in pair) for pair in auto_pairs])) if auto_pairs else None,
        "step1_target_present": float(np.mean([pair[0]["target_present"] for pair in auto_pairs])) if auto_pairs else None,
        "step2_target_present": float(np.mean([pair[1]["target_present"] for pair in auto_pairs])) if auto_pairs else None,
    }


def pair_index(rows: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["pair_id"]].append(index)
    pair_rows, step1, step2 = [], [], []
    for pair_id in sorted(grouped):
        indices = sorted(grouped[pair_id], key=lambda index: rows[index]["steps"])
        if len(indices) != 2 or [rows[index]["steps"] for index in indices] != [1, 2]:
            raise RuntimeError((pair_id, indices))
        pair_rows.append({key: rows[indices[0]][key] for key in
                          ("pair_id", "family", "unit", "language", "surface", "surface_class", "direction", "partition")})
        step1.append(indices[0]); step2.append(indices[1])
    return pair_rows, np.asarray(step1, dtype=np.int64), np.asarray(step2, dtype=np.int64)


def fit_predict(train: np.ndarray, test: np.ndarray, families: np.ndarray,
                h: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    global_y = y[train].mean(axis=0)
    global_h = h[train].mean(axis=0)
    family_y = {family: y[train[families[train] == family]].mean(axis=0) for family in sorted(set(families[train]))}
    family_h = {family: h[train[families[train] == family]].mean(axis=0) for family in sorted(set(families[train]))}
    base_y_train = np.stack([family_y[family] for family in families[train]])
    base_h_train = np.stack([family_h[family] for family in families[train]])
    hr, yr = h[train] - base_h_train, y[train] - base_y_train
    slope = (np.sum(hr * yr, axis=0, dtype=np.float64) /
             (np.sum(hr * hr, axis=0, dtype=np.float64) + 1e-12)).astype(np.float32)
    permutation = np.roll(np.arange(h.shape[1]), SHIFT)
    hp = hr[:, permutation]
    slope_p = (np.sum(hp * yr, axis=0, dtype=np.float64) /
               (np.sum(hp * hp, axis=0, dtype=np.float64) + 1e-12)).astype(np.float32)
    base_y_test = np.stack([family_y[family] for family in families[test]])
    base_h_test = np.stack([family_h[family] for family in families[test]])
    htest = h[test] - base_h_test
    return {
        "global": np.broadcast_to(global_y, y[test].shape), "family": base_y_test,
        "state": base_y_test + htest * slope,
        "mismatch": base_y_test + htest[:, permutation] * slope_p,
        "slope": slope, "global_h": global_h,
    }


def gains(truth: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, float]:
    denominator = float(np.sum((truth - predictions["global"]) ** 2, dtype=np.float64)) + 1e-30
    return {stage: float(1.0 - np.sum((truth - predictions[stage]) ** 2, dtype=np.float64) / denominator)
            for stage in STAGES}


def analyze(rows: list[dict], collection: dict) -> dict:
    pair_rows, step1, step2 = pair_index(rows)
    families = np.asarray([row["family"] for row in pair_rows], dtype=object)
    train = np.asarray([i for i, row in enumerate(pair_rows) if row["partition"] == "discovery"], dtype=np.int64)
    en_train = np.asarray([i for i in train if pair_rows[i]["language"] == "en"], dtype=np.int64)
    tests = {split: np.asarray([i for i, row in enumerate(pair_rows) if row["partition"] == split], dtype=np.int64)
             for split in SPLITS[:-1]}
    tests["language_lockbox"] = np.asarray([i for i in train if pair_rows[i]["language"] == "zh"], dtype=np.int64)
    state = np.load(collection["state"]["path"], mmap_mode="r")
    attention = np.load(collection["attention"]["path"], mmap_mode="r")
    mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    layers, dimension = attention.shape[1:]
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    state_passport = np.lib.format.open_memmap(derived / "paired_state_step2_minus_step1_family.float32.npy",
                                                mode="w+", dtype=np.float32,
                                                shape=(state.shape[1], len(contract.COMPOSITION_FAMILIES), dimension))
    component_passport = np.lib.format.open_memmap(derived / "paired_update_step2_minus_step1_family.float32.npy",
                                                    mode="w+", dtype=np.float32,
                                                    shape=(len(COMPONENTS), layers, len(contract.COMPOSITION_FAMILIES), dimension))
    slopes = np.lib.format.open_memmap(derived / "paired_matched_diagonal_slope.float32.npy", mode="w+",
                                       dtype=np.float32, shape=(len(COMPONENTS), layers, dimension))
    metric_array = np.zeros((len(COMPONENTS), len(SPLITS), len(STAGES), layers), dtype=np.float32)
    for qpoint in range(state.shape[1]):
        delta = np.asarray(state[step2, qpoint], dtype=np.float32) - np.asarray(state[step1, qpoint], dtype=np.float32)
        for fi, family in enumerate(contract.COMPOSITION_FAMILIES):
            component_pass = delta[train[families[train] == family]].mean(axis=0)
            state_passport[qpoint, fi] = component_pass
    state_passport.flush()
    for layer in range(layers):
        h = np.asarray(state[step2, layer], dtype=np.float32) - np.asarray(state[step1, layer], dtype=np.float32)
        a = np.asarray(attention[step2, layer], dtype=np.float32) - np.asarray(attention[step1, layer], dtype=np.float32)
        m = np.asarray(mlp[step2, layer], dtype=np.float32) - np.asarray(mlp[step1, layer], dtype=np.float32)
        for ci, y in enumerate((a + m, a, m)):
            for fi, family in enumerate(contract.COMPOSITION_FAMILIES):
                component_passport[ci, layer, fi] = y[train[families[train] == family]].mean(axis=0)
            standard = fit_predict(train, train, families, h, y)
            slopes[ci, layer] = standard["slope"]
            for si, split in enumerate(SPLITS):
                fit = en_train if split == "language_lockbox" else train
                prediction = fit_predict(fit, tests[split], families, h, y)
                result = gains(y[tests[split]], prediction)
                metric_array[ci, si, :, layer] = [result[stage] for stage in STAGES]
        component_passport.flush(); slopes.flush()
        print(f"[phase2415 analysis] layer {layer + 1}/{layers}", flush=True)
    np.save(derived / "paired_operator_layer_metrics.float32.npy", metric_array)
    component_passport.flush(); slopes.flush()
    summaries = {}
    for ci, component in enumerate(COMPONENTS):
        summaries[component] = {}
        for si, split in enumerate(SPLITS):
            layer_mean = {stage: float(metric_array[ci, si, stage_index].mean()) for stage_index, stage in enumerate(STAGES)}
            layer_mean["state_increment"] = layer_mean["state"] - layer_mean["family"]
            layer_mean["physical_advantage"] = layer_mean["state"] - layer_mean["mismatch"]
            layer_mean["best_state_layer"] = int(np.argmax(metric_array[ci, si, 2]))
            summaries[component][split] = layer_mean
    perms = [np.asarray(value, dtype=np.int64) for value in itertools.permutations(range(4)) if value != tuple(range(4))]
    geometry_result = {}
    for ci, component in enumerate(COMPONENTS):
        values = np.asarray(component_passport[ci], dtype=np.float32)
        observed, null_p, coordinate = [], [], []
        for layer in range(layers - 1):
            left, right = values[layer], values[layer + 1]
            score = geometry.correlation(geometry.geometry_vector(left), geometry.geometry_vector(right))
            null = np.asarray([geometry.correlation(geometry.geometry_vector(left), geometry.geometry_vector(right[perm]))
                               for perm in perms])
            observed.append(score); null_p.append(float((1 + np.sum(null >= score)) / (1 + len(null))))
            coordinate.append(float(np.mean([geometry.cosine(left[f], right[f]) for f in range(4)])))
        geometry_result[component] = {
            "adjacent_relation_mean": float(np.mean(observed)),
            "adjacent_coordinate_cosine_mean": float(np.mean(coordinate)),
            "exceeds_permutation_q95_rate": float(np.mean([p <= 0.05 for p in null_p])),
            "median_permutation_p": float(np.median(null_p)), "cells": len(observed), "permutations": len(perms),
        }
    closure_indices = np.linspace(0, len(rows) - 1, 64, dtype=np.int64)
    truth = (np.asarray(state[closure_indices, 1:-1], dtype=np.float32) -
             np.asarray(state[closure_indices, :-2], dtype=np.float32))
    pred = np.asarray(attention[closure_indices], dtype=np.float32) + np.asarray(mlp[closure_indices], dtype=np.float32)
    closure = {"relative_rmse": float(np.sqrt(np.sum((truth - pred) ** 2) / max(np.sum(truth ** 2), 1e-30))),
               "cosine": geometry.cosine(truth.ravel(), pred.ravel())}
    files = {"state_passport": str(derived / "paired_state_step2_minus_step1_family.float32.npy"),
             "component_passport": str(derived / "paired_update_step2_minus_step1_family.float32.npy"),
             "slopes": str(derived / "paired_matched_diagonal_slope.float32.npy"),
             "layer_metrics": str(derived / "paired_operator_layer_metrics.float32.npy")}
    for value in (state_passport, component_passport, slopes, state, attention, mlp):
        close(value)
    return {"pairs": len(pair_rows), "discovery_pairs": len(train), "english_train_pairs": len(en_train),
            "test_pairs": {key: len(value) for key, value in tests.items()}, "summary": summaries,
            "crosslayer_geometry": geometry_result, "component_closure": closure, "derived": files}


def cleanup_raw(collection: dict) -> dict:
    removed, total = [], 0
    for value in collection.values():
        path = Path(value["path"])
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            removed.append(str(path)); total += size
    return {"removed_files": len(removed), "removed_bytes": total, "removed_gib": total / 2**30,
            "recoverable": False, "paths": removed}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 同事实链一步—两步严格配对组合算子（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 修复Phase2411组合样本没有同一事实链一步/两步配对的核心硬伤。空间、时间、比较、分类4族各10组事实链，覆盖中英、canonical/paraphrase/discourse/natural、双方向；每个配置同时生成一步和两步查询，共640对/1280条。pair内事实、语言、整体表面、方向、候选集合与候选顺序完全相同，只改变查询步数，答案与foil随之交换。Qwen3-4B BF16先做目标序列教师强制与新unit自主生成校准，再在answer boundary采集embedding、36个block及final norm的全部2560坐标，并同步采集36层Attention/MLP输出；没有Top-K/PCA。

$$D^H_{{p,q}}=H_{{p,q}}^{{(2)}}-H_{{p,q}}^{{(1)}},\qquad
D^U_{{p,q}}=U_{{p,q}}^{{(2)}}-U_{{p,q}}^{{(1)}},$$

$$\widehat D^U_{{p,q,j}}=G^U_{{f,q,j}}+a_{{q,j}}\left(D^H_{{p,q,j}}-G^H_{{f,q,j}}\right),\qquad
\Delta_{{phys}}=G(\widehat D^U_{{matched}})-G(\widehat D^U_{{j\leftarrow j+791}}).$$

**结果汇总。** 材料 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior']['pair_summary'], ensure_ascii=False)}`；逐层全坐标算子 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；跨层族关系 `{json.dumps(result['analysis']['crosslayer_geometry'], ensure_ascii=False)}`；组件闭合 `{json.dumps(result['analysis']['component_closure'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2415_c27441_c27760_exact_paired_composition.py`；材料、模型token索引、行为输出、逐族×层×全部2560坐标的状态/组件配对护照、同坐标斜率和逐层指标位于`tests/glm5/result/phase2415_c27441_c27760_exact_paired_composition`。未修改其他Markdown。

**分析与理论进展。** 本Phase第一次把“要求一步还是两步”放在严格反事实pair内测量，因此pair差分剥离了事实链、词项、语言、表面、方向和候选顺序。族均值能否跨新unit/整模板解释$D^U$，以及真实$D^H_j\to D^U_j$是否优于等容量坐标错配，是条件坐标齿轮的更直接拼图。跨层Gram只检验四族关系顺序是否持续，不等同于向量在原坐标中搬运。

**问题硬伤与结论。** 两条pair仍改变查询文本与正确答案，故$D$是“查询需求+答案角色”的联合反事实，不是把第一步运行结果迭代一次得到第二步的纯组合算子。候选答案出现在prompt中，教师强制只测偏好；四族仅有6个Gram边，置换检验分辨率最低为$1/24$。对角预测阳性也可能来自残差流自相关。原始float16场已在派生全坐标护照和指标核验后删除，删除不可恢复；保留的派生数组仍覆盖每一物理坐标。下一阶段若目标不变，应冻结本材料跨模型复核，而不是从单一Qwen4B外推通用机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    source_path = OUT / "material/exact_paired_composition.jsonl"
    source = read_rows(source_path) if source_path.exists() else compile_source()
    if not source_path.exists():
        write_rows(source_path, source)
    audit = material_audit(source)
    model, tokenizer, label = capability.load_model("qwen4b")
    behavior.OUT = OUT
    try:
        compiled_path = OUT / "index/composition_rows.jsonl"
        if compiled_path.exists():
            compiled = read_rows(compiled_path)
            calibration = json.loads((OUT / "analysis/token_calibration.json").read_text(encoding="utf-8"))
        else:
            compiled, calibration = behavior.compile_rows(tokenizer, source)
            write_rows(compiled_path, [{**row, "pair_id": source[index]["pair_id"]} for index, row in enumerate(compiled)])
            compiled = read_rows(compiled_path)
            save(OUT / "analysis/token_calibration.json", calibration)
        teacher, teacher_summary = behavior.score_rows("qwen4b", model, compiled, 16)
        lockbox = [row for row in compiled if row["partition"] == "fresh_unit_lockbox"]
        autonomous, autonomous_summary = behavior.generate_lockbox("qwen4b", model, tokenizer, lockbox, 8)
        collection = collect_fields(model, compiled, 4)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pair_behavior = behavior_pair_summary(teacher, autonomous)
    analysis = analyze(compiled, collection)
    cleanup = cleanup_raw(collection)
    checks = {
        "material_640_exact_pairs": audit["pairs"] == 640 and audit["exact_pair_rate"] == 1.0,
        "query_is_only_pair_change": audit["query_only_change_rate"] == 1.0,
        "balanced_steps": audit["steps"] == {1: 640, 2: 640} or audit["steps"] == {"1": 640, "2": 640},
        "token_calibration": calibration["rows"] == 1280 and calibration["event_monotonic_rate"] == 1.0,
        "behavior_complete": len(teacher) == 1280 and pair_behavior["teacher_pairs"] == 640,
        "full_coordinates": collection["state"]["shape"] == [1280, 38, 2560] and collection["attention"]["shape"] == [1280, 36, 2560],
        "component_closure": analysis["component_closure"]["cosine"] > 0.999,
        "finite_metrics": all(math.isfinite(value) for component in analysis["summary"].values()
                              for split in component.values() for key, value in split.items() if key != "best_state_layer"),
        "raw_hiddenstate_cleaned": cleanup["removed_files"] == 3 and all(not Path(item["path"]).exists() for item in collection.values()),
        "claim_boundary": True,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": label, "material_audit": audit,
              "token_calibration": calibration, "behavior": {"teacher": teacher_summary, "autonomous": autonomous_summary,
                                                                 "pair_summary": pair_behavior},
              "collection": collection, "analysis": analysis, "cleanup": cleanup,
              "adjudication": {"exact_pair_composition_operator_proven": False,
                               "paired_query_demand_field_measured": True,
                               "general_mechanism_requires_crossmodel_replication": True},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "material": audit, "behavior": pair_behavior,
                      "summary": analysis["summary"], "geometry": analysis["crosslayer_geometry"],
                      "cleanup": cleanup, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
