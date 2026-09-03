#!/usr/bin/env python3
"""Compositional semantic-graph atlas with multiple natural future responses."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2348 = RESULT / "phase2348_c8721_c8840_supported_task_continuous_coalition_causality"
OUT = RESULT / "phase2350_c8961_c9120_compositional_graph_natural_response_closure"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MATERIAL = OUT / "material/compositional_natural_response.jsonl"
STATES = OUT / "raw/prompt_boundary_all_checkpoints.float16.npy"
DECISIONS = OUT / "raw/response_decisions.float32.npy"
PROGRESS = OUT / "raw/progress.json"
GENERATION = OUT / "raw/lockbox_generation.jsonl"
PHASE = 2350
CAMPAIGN = "C8961-C9120"
FAMILIES = (
    "attitude_taxonomy", "event_taxonomy", "causal_location", "possession_partwhole",
    "negation_coreference", "quantifier_temporal", "translation_taxonomy", "conjunction_attribute",
)
MACROTYPE = {
    "attitude_taxonomy": "attitude_concept", "event_taxonomy": "event_concept",
    "causal_location": "cause_relation", "possession_partwhole": "possession_structure",
    "negation_coreference": "scope_reference", "quantifier_temporal": "scope_time",
    "translation_taxonomy": "cross_language_concept", "conjunction_attribute": "grammar_attribute",
}
RELATIONS = {
    "attitude_taxonomy": ("likes", "instance_of", "subtype_of"),
    "event_taxonomy": ("acts_on", "instance_of", "subtype_of"),
    "causal_location": ("caused_change_in", "located_in", "inside"),
    "possession_partwhole": ("owns", "part_of", "located_in"),
    "negation_coreference": ("former_carried", "located_in", "instance_of"),
    "quantifier_temporal": ("only_inspected", "before_event", "occurs_in"),
    "translation_taxonomy": ("translates_to", "instance_of", "subtype_of"),
    "conjunction_attribute": ("repaired", "has_property", "property_group"),
}
LANGUAGES = ("en", "zh")
SURFACES = ("direct", "natural")
QUERIES = ("source", "object", "intermediate", "terminal")
PARTITIONS = ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox")
UNITS = 16
PARTITION_BY_UNIT = {unit: PARTITIONS[unit // 4] for unit in range(UNITS)}
TRAIN_PARTITIONS = ("discovery", "confirmation")
EPS = 1e-12

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2346_c8481_c8600_factorial_coordinate_route_competition as route  # noqa: E402

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


def identifiers(unit: int, state: int) -> tuple[list[str], list[str]]:
    offset = unit + state * 200
    nodes = [f"actor-{offset:03d}", f"item-{offset:03d}", f"class-{offset:03d}", f"domain-{offset:03d}"]
    foils = [f"actor-{offset + 100:03d}", f"item-{offset + 100:03d}",
             f"class-{offset + 100:03d}", f"domain-{offset + 100:03d}"]
    return nodes, foils


def graph_record(family: str, language: str, surface: str, unit: int, state: int) -> dict:
    nodes, foils = identifiers(unit, state)
    relations = RELATIONS[family]
    graph = [[nodes[0], relations[0], nodes[1]], [nodes[1], relations[1], nodes[2]],
             [nodes[2], relations[2], nodes[3]], [foils[0], "unrelated_to", foils[1]]]
    if language == "en":
        direct = (f"{nodes[0]} {relations[0]} {nodes[1]}. {nodes[1]} {relations[1]} {nodes[2]}. "
                  f"{nodes[2]} {relations[2]} {nodes[3]}. {foils[0]} is unrelated.")
        natural = (f"A record links {nodes[0]} to {nodes[1]} by {relations[0]}; it then places {nodes[1]} under "
                   f"{nodes[2]} through {relations[1]}, and connects {nodes[2]} onward to {nodes[3]} by {relations[2]}. "
                   f"The label {foils[0]} is only a distractor.")
        questions = {
            "source": "Which exact actor identifier begins the linked path?",
            "object": "Which exact item identifier is directly linked from the actor?",
            "intermediate": "Which exact class identifier is reached after two edges?",
            "terminal": "Which exact domain identifier is reached at the end of the three-edge path?",
        }
        suffix = "Answer with only that exact identifier, without explanation.\nAnswer:"
    else:
        direct = (f"{nodes[0]}通过{relations[0]}关联{nodes[1]}。{nodes[1]}通过{relations[1]}关联{nodes[2]}。"
                  f"{nodes[2]}通过{relations[2]}关联{nodes[3]}。{foils[0]}与此无关。")
        natural = (f"一份记录先用{relations[0]}把{nodes[0]}和{nodes[1]}连接起来，再用{relations[1]}把{nodes[1]}归到"
                   f"{nodes[2]}，最后用{relations[2]}把{nodes[2]}连接到{nodes[3]}。{foils[0]}只是干扰标签。")
        questions = {
            "source": "这条关联路径起点的精确actor标识符是什么？",
            "object": "从起点直接连接到的精确item标识符是什么？",
            "intermediate": "经过两条边到达的精确class标识符是什么？",
            "terminal": "三条边路径终点的精确domain标识符是什么？",
        }
        suffix = "只回答该精确标识符，不要解释。\n答案："
    facts = direct if surface == "direct" else natural
    return {"graph": graph, "graph_hash": hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest(),
            "facts": facts, "questions": questions, "answers": dict(zip(QUERIES, nodes)),
            "wrongs": dict(zip(QUERIES, foils)), "suffix": suffix}


def compile_material(tokenizer) -> tuple[list[dict], dict]:
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for language in LANGUAGES:
            for surface in SURFACES:
                for unit in range(UNITS):
                    for state in (0, 1):
                        record = graph_record(family, language, surface, unit, state)
                        for query in QUERIES:
                            prompt = f"{record['facts']}\n{record['questions'][query]}\n{record['suffix']}"
                            prompt_ids = [int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)]
                            target_ids = [int(value) for value in tokenizer.encode(" " + record["answers"][query], add_special_tokens=False)]
                            wrong_ids = [int(value) for value in tokenizer.encode(" " + record["wrongs"][query], add_special_tokens=False)]
                            rows.append({"case_id": f"c8961-{family}-{language}-{surface}-u{unit:02d}-s{state}-{query}",
                                         "design_index": len(rows), "family": family, "family_index": family_index,
                                         "macrotype": MACROTYPE[family], "language": language, "surface": surface,
                                         "unit": unit, "state": state, "partition": PARTITION_BY_UNIT[unit], "query": query,
                                         "semantic_graph": record["graph"], "semantic_graph_hash": record["graph_hash"],
                                         "future_prompt": prompt, "future_prompt_ids": prompt_ids,
                                         "natural_target": record["answers"][query], "natural_wrong": record["wrongs"][query],
                                         "future_target_ids": target_ids, "future_wrong_ids": wrong_ids,
                                         "boundary_position": len(prompt_ids) - 1})
    graph_groups = defaultdict(set)
    for row in rows:
        graph_groups[(row["family"], row["surface"], row["unit"], row["state"])].add(row["semantic_graph_hash"])
    lengths = [len(row["future_prompt_ids"]) for row in rows]
    audit = {"rows": len(rows), "families": len(FAMILIES), "macrotypes": len(set(MACROTYPE.values())),
             "units": UNITS, "queries_per_graph": len(QUERIES), "languages": list(LANGUAGES), "surfaces": list(SURFACES),
             "parallel_graph_hash_across_languages_and_queries": all(len(values) == 1 for values in graph_groups.values()),
             "unique_case_ids": len({row["case_id"] for row in rows}) == len(rows),
             "token_length_min": min(lengths), "token_length_max": max(lengths),
             "target_wrong_distinct": all(row["natural_target"] != row["natural_wrong"] for row in rows)}
    return rows, audit


def candidate_score(model, device, batch: list[dict], answer_key: str, capture: dict[int, torch.Tensor], pad: int) -> np.ndarray:
    combined = [row["future_prompt_ids"] + row[answer_key] for row in batch]
    ids, mask, positions = baseline.pad_right(combined, device, pad)
    capture.clear()
    output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
    scores = []
    for local, row in enumerate(batch):
        answer = row[answer_key]
        start = len(row["future_prompt_ids"])
        prediction_positions = torch.arange(start - 1, start + len(answer) - 1, device=device)
        hidden = output.last_hidden_state[local, prediction_positions]
        logits = model.lm_head(hidden).float()
        token_ids = torch.tensor(answer, dtype=torch.long, device=device)
        scores.append(float(F.log_softmax(logits, dim=-1)[torch.arange(len(answer), device=device), token_ids].mean().item()))
    return np.asarray(scores, dtype=np.float32)


def collect(model, tokenizer, device, rows: list[dict], batch_size: int = 12) -> dict:
    module_list = modules(model)
    shape = (len(rows), len(module_list), int(model.config.hidden_size))
    if STATES.exists() and DECISIONS.exists() and PROGRESS.exists():
        completed = int(json.loads(PROGRESS.read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(STATES, mode="r+")
        decisions = np.lib.format.open_memmap(DECISIONS, mode="r+")
    else:
        completed = 0
        STATES.parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(STATES, mode="w+", dtype=np.float16, shape=shape)
        decisions = np.lib.format.open_memmap(DECISIONS, mode="w+", dtype=np.float32, shape=(len(rows), 4))
    captures = {}
    handles = []
    for qpoint, module in enumerate(module_list):
        def hook(_module, _inputs, value, qpoint=qpoint):
            captures[qpoint] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                target_scores = candidate_score(model, device, batch, "future_target_ids", captures, pad)
                for qpoint in range(len(module_list)):
                    selected = torch.stack([captures[qpoint][local, len(row["future_prompt_ids"]) - 1]
                                            for local, row in enumerate(batch)])
                    states[start:start + len(batch), qpoint] = selected.float().cpu().numpy().astype(np.float16)
                wrong_scores = candidate_score(model, device, batch, "future_wrong_ids", captures, pad)
                margins = target_scores - wrong_scores
                decisions[start:start + len(batch), 0] = target_scores
                decisions[start:start + len(batch), 1] = wrong_scores
                decisions[start:start + len(batch), 2] = margins
                decisions[start:start + len(batch), 3] = margins > 0
                states.flush(); decisions.flush(); save(PROGRESS, {"completed": start + len(batch), "shape": list(shape)})
                if (start + len(batch)) % 192 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2350 score] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        states.flush(); decisions.flush(); close_memmap(states); close_memmap(decisions)
    generation_rows = []
    with torch.inference_mode():
        lockbox = [row for row in rows if row["partition"] == "fresh_lockbox" and row["surface"] == "natural"]
        for index, row in enumerate(lockbox):
            ids = torch.tensor([row["future_prompt_ids"]], dtype=torch.long, device=device)
            generated = model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=pad, eos_token_id=model.config.eos_token_id)
            text = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
            generation_rows.append({"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                                    "surface": row["surface"], "query": row["query"], "unit": row["unit"],
                                    "target": row["natural_target"], "generated": text,
                                    "exact": text == row["natural_target"], "prefix": text.startswith(row["natural_target"])})
            if (index + 1) % 64 == 0:
                print(f"[phase2350 generate] {index + 1}/{len(lockbox)}", flush=True)
    io.write_rows(GENERATION, generation_rows)
    return {"shape": list(shape), "batch_size": batch_size, "generation_rows": len(generation_rows)}


def behavior(rows: list[dict]) -> dict:
    decisions = np.load(DECISIONS, mmap_mode="r")
    result = {"overall_teacher_forced_preference": float(np.mean(decisions[:, 3])), "families": {}, "qualified": []}
    for family in FAMILIES:
        cells = {}
        passed = True
        for language in LANGUAGES:
            for surface in SURFACES:
                for query in QUERIES:
                    for partition in PARTITIONS:
                        idx = [i for i, row in enumerate(rows) if row["family"] == family and row["language"] == language
                               and row["surface"] == surface and row["query"] == query and row["partition"] == partition]
                        accuracy = float(np.mean(decisions[idx, 3]))
                        cells[f"{language}:{surface}:{query}:{partition}"] = accuracy
                        passed = passed and accuracy >= 0.70
        result["families"][family] = {"qualified": passed, "minimum_cell_accuracy": min(cells.values()), "cells": cells}
        if passed:
            result["qualified"].append(family)
    generated = io.read_rows(GENERATION)
    result["lockbox_generation"] = {"rows": len(generated), "exact": float(np.mean([row["exact"] for row in generated])),
                                     "target_prefix": float(np.mean([row["prefix"] for row in generated]))}
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
            "median_distance_ratio": float(np.median(correct_distance / (np.min(masked, axis=1) + EPS)))}


def evaluate(field: np.ndarray, rows: list[dict], labels: tuple[str, ...], partition: str) -> dict:
    specs = (("en_to_zh", "language", "en", "zh"), ("zh_to_en", "language", "zh", "en"),
             ("direct_to_natural", "surface", "direct", "natural"), ("natural_to_direct", "surface", "natural", "direct"),
             ("source_to_terminal", "query", "source", "terminal"), ("terminal_to_source", "query", "terminal", "source"),
             ("object_to_intermediate", "query", "object", "intermediate"),
             ("intermediate_to_object", "query", "intermediate", "object"))
    transfers = {name: grouped(field, rows, labels, factor, source, target, partition)
                 for name, factor, source, target in specs}
    return {"transfers": transfers, "minimum_accuracy": min(value["accuracy"] for value in transfers.values()),
            "mean_accuracy": float(np.mean([value["accuracy"] for value in transfers.values()])),
            "maximum_distance_ratio": max(value["median_distance_ratio"] for value in transfers.values())}


def residualize(field: np.ndarray, rows: list[dict]) -> np.ndarray:
    train = np.asarray([row["partition"] in TRAIN_PARTITIONS for row in rows])
    grand = field[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    prediction = np.broadcast_to(grand, field.shape).copy()
    for factor in ("language", "surface", "query", "state"):
        for level in sorted({row[factor] for row in rows}, key=str):
            level_train = train & np.asarray([row[factor] == level for row in rows])
            effect = field[level_train].mean(axis=0, dtype=np.float64).astype(np.float32) - grand
            prediction[np.asarray([row[factor] == level for row in rows])] += effect
    return field - prediction


def analyze(rows: list[dict], behavior_result: dict) -> tuple[dict, dict[str, np.ndarray]]:
    states = np.load(STATES, mmap_mode="r")
    labels = tuple(behavior_result["qualified"] if len(behavior_result["qualified"]) >= 2 else FAMILIES)
    records = []
    for qpoint in range(states.shape[1]):
        absolute = np.abs(states[:, qpoint].astype(np.float32))
        residual = residualize(absolute, rows)
        full = evaluate(residual, rows, labels, "fresh_confirmation")
        sorted_control = evaluate(np.sort(residual, axis=1), rows, labels, "fresh_confirmation")
        records.append({"qpoint": qpoint, "factorial_residual_absolute_hidden": full,
                        "row_sorted_residual_control": sorted_control})
    selected = max(records, key=lambda row: (row["factorial_residual_absolute_hidden"]["minimum_accuracy"],
                                              row["factorial_residual_absolute_hidden"]["mean_accuracy"], -row["qpoint"]))
    qpoint = int(selected["qpoint"])
    absolute = np.abs(states[:, qpoint].astype(np.float32))
    residual = residualize(absolute, rows)
    sorted_residual = np.sort(residual, axis=1)
    lock = evaluate(residual, rows, labels, "fresh_lockbox")
    sorted_lock = evaluate(sorted_residual, rows, labels, "fresh_lockbox")
    causal = json.loads((P2348 / "analysis/final.json").read_text(encoding="utf-8"))["analysis"]["gate"]["scoped_causal_candidate_passed"]
    gate = {"qualified_family_count": len(behavior_result["qualified"]), "selected_qpoint": qpoint,
            "lockbox_minimum_accuracy": lock["minimum_accuracy"],
            "coordinate_advantage_over_sorted": lock["minimum_accuracy"] - sorted_lock["minimum_accuracy"],
            "distance_ratio": lock["maximum_distance_ratio"],
            "generation_prefix_accuracy": behavior_result["lockbox_generation"]["target_prefix"],
            "behavior_pass": len(behavior_result["qualified"]) >= 6,
            "graph_field_pass": lock["minimum_accuracy"] >= 0.30,
            "coordinate_identity_pass": lock["minimum_accuracy"] >= sorted_lock["minimum_accuracy"] + 0.10,
            "distance_pass": lock["maximum_distance_ratio"] < 1.0,
            "natural_generation_pass": behavior_result["lockbox_generation"]["target_prefix"] >= 0.50,
            "prior_causal_bridge_pass": causal}
    gate["full_compositional_closure_passed"] = all(gate[name] for name in
                                                     ("behavior_pass", "graph_field_pass", "coordinate_identity_pass",
                                                      "distance_pass", "natural_generation_pass", "prior_causal_bridge_pass"))
    publish_fields = {"factorial_residual_absolute_hidden": residual,
                      "row_sorted_residual_control": sorted_residual,
                      "absolute_hidden": absolute}
    close_memmap(states)
    return {"labels": list(labels), "selection_trajectory": records, "selected": selected,
            "lockbox": lock, "sorted_lockbox": sorted_lock, "gate": gate}, publish_fields


def publish(rows: list[dict], analysis: dict, fields: dict[str, np.ndarray]) -> list[dict]:
    datasets = []
    source = np.load(STATES, mmap_mode="r")
    qpoints = []
    for qpoint in (0, int(analysis["selected"]["qpoint"]), source.shape[1] - 1):
        if qpoint not in qpoints:
            qpoints.append(qpoint)
    dataset_id = "c8961_qwen4b_compositional_natural_key_checkpoint_hiddenstate"
    binary = VIS / f"{dataset_id}.float16.npy"
    out = atlas.create_binary(binary.name, len(rows) * len(qpoints), 2560, np.float16)
    metadata = []
    cursor = 0
    for qpoint in qpoints:
        out[cursor:cursor + len(rows)] = source[:, qpoint]
        metadata.extend({"case_id": row["case_id"], "family": row["family"], "macrotype": row["macrotype"],
                         "language": row["language"], "surface": row["surface"], "query": row["query"],
                         "partition": row["partition"], "unit": row["unit"], "state": row["state"], "qpoint": qpoint}
                        for row in rows)
        cursor += len(rows)
    out.flush(); close_memmap(out); close_memmap(source)
    datasets.append(atlas.write_metadata(
        dataset_id, "Qwen3-4B compositional natural-response key-checkpoint HiddenState",
        binary, metadata, "Qwen3-4B-FP16", "compositional_natural_key_hiddenstate_v1",
        "observational compositional graph and natural-response field; causal bridge separately failed",
        "8 compositional graph families x bilingual x two surfaces x four future queries x 16 units x two states",
        "all 2560 coordinates at embedding, selected checkpoint and final norm",
        {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": 2560, "no_topk": True, "qpoints": qpoints},
    ))
    passport_id = "c8962_qwen4b_compositional_response_route_passport"
    passport_binary = VIS / f"{passport_id}.float32.npy"
    passport = np.concatenate([value.astype(np.float32) for value in fields.values()], axis=0)
    out = atlas.create_binary(passport_binary.name, passport.shape[0], passport.shape[1], np.float32)
    out[:] = passport; out.flush(); close_memmap(out)
    passport_metadata = []
    for view in fields:
        passport_metadata.extend({"case_id": row["case_id"], "view": view, "family": row["family"],
                                  "language": row["language"], "surface": row["surface"], "query": row["query"],
                                  "partition": row["partition"], "unit": row["unit"], "state": row["state"],
                                  "qpoint": int(analysis["selected"]["qpoint"])} for row in rows)
    datasets.append(atlas.write_metadata(
        passport_id, "Qwen3-4B compositional natural-response route passport", passport_binary, passport_metadata,
        "Qwen3-4B-FP16", "compositional_natural_route_passport_v1",
        "factorial-residual composition atlas with sorted and unresidualized controls",
        "selected checkpoint on fresh_confirmation; untouched fresh_lockbox adjudication",
        "all 2560 coordinates retained in residual, sorted-control and absolute-H views",
        {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": 2560, "no_topk": True,
         "qpoint": int(analysis["selected"]["qpoint"]), "views": list(fields)},
    ))
    return datasets


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 八类组合语义图的多未来响应与自然生成闭合裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 为补齐大方案的组合与自然输出关卡，建立8种三边组合图（态度×类别、事件×类别、因果×位置、占有×部件、否定×指代、量词×时序、翻译×类别、连接×属性），每类16 units、两个状态、中英、direct/natural表述；对同一图分别查询起点、直接对象、两跳中间节点和三跳终点，共4096条。输出不再是A/B，而是自然多token标识符。用目标/错误序列的平均teacher-forced log概率判断行为，并在512条fresh_lockbox上贪心生成；同时采集prompt边界38×2560场。

$$
\ell(y|x)=\frac1{{|y|}}\sum_{{k=1}}^{{|y|}}\log p(y_k|x,y_{{<k}}),
\qquad B=[\ell(y^+|x)>\ell(y^-|x)].
$$

$$
\operatorname{{Closure}}=B\land G_{{field}}\land G_{{coord}}\land G_{{distance}}
\land G_{{generation}}\land G_{{causal}}.
$$

**结果汇总与相关文件。** 材料 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；内部图谱 `{json.dumps(result['analysis'], ensure_ascii=False)}`；客户端全坐标场 `{json.dumps(result['datasets'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2350_c8961_c9120_compositional_graph_natural_response_closure.py`；结果 `tests/glm5/result/phase2350_c8961_c9120_compositional_graph_natural_response_closure`。

**理论进展、问题硬伤与结论。** 该测试把“同一外部图能否支持多个未来查询”放到自然多token输出中，较A/B族分类更接近响应等价；但标识符生成仍是受控任务，组合图共享三边模板，teacher forcing偏好不等于自由生成，且内部族分类仍可能利用关系词。Phase2348的因果桥冻结为失败，因此即使行为、内部图和自然生成都通过，也不能宣布完整闭合。只有外部图—内部具体坐标—多未来输出—选择性干预五项同时通过，才可称组合编码机制。

**下一阶段路线判断。** 本大阶段A（双语正交）、B（路线竞赛）、C（局部因果和跨模型）与D（组合/自然输出）均已执行。若完整闭合失败，下一阶段仍同一总目标，但应从“继续分类族”转向行为稳定的多未来响应等价与自然指令图谱，先解决任务政策和选择性干预，再考虑新数学。
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
    tokenizer = model_utils.load_model.__globals__.get("AutoTokenizer")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                               local_files_only=True, use_fast=False)
    rows, material_audit = compile_material(tokenizer)
    io.write_rows(MATERIAL, rows)
    freeze = {"frozen_before_model_load": True, "families": list(FAMILIES), "units": UNITS,
              "languages": list(LANGUAGES), "surfaces": list(SURFACES), "queries": list(QUERIES),
              "behavior_cell_threshold": 0.70, "graph_transfer_threshold": 0.30,
              "coordinate_advantage": 0.10, "natural_generation_prefix_threshold": 0.50,
              "coordinate_policy": "all 2560; no Top-K/PCA/projection"}
    save(OUT / "config/frozen_contract.json", freeze)
    model = None
    try:
        model, tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        collection = collect(model, tokenizer, device, rows)
    finally:
        if model is not None:
            model_utils.release_model(model)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    behavior_result = behavior(rows)
    analysis, fields = analyze(rows, behavior_result)
    datasets = publish(rows, analysis, fields)
    verification = [atlas.verify(dataset) for dataset in datasets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    raw_size = STATES.stat().st_size
    if not (verified and build["passed"]):
        raise RuntimeError(("publication_failed_before_cleanup", verification, build))
    STATES.unlink()
    cleanup = {"deleted": str(STATES), "bytes_reclaimed": raw_size, "deleted_ok": not STATES.exists()}
    checks = {"rows": material_audit["rows"] == 4096, "parallel_graphs": material_audit["parallel_graph_hash_across_languages_and_queries"],
              "collection": collection["shape"] == [4096, 38, 2560], "assets_verified": verified,
              "frontend_build": build["passed"], "raw_state_deleted": cleanup["deleted_ok"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "material_audit": material_audit,
              "collection": collection, "behavior": behavior_result, "analysis": analysis,
              "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)), "verification": verification,
              "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)), "frontend_build": build,
              "cleanup": cleanup, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2350_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
