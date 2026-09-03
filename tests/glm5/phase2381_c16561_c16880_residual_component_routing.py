#!/usr/bin/env python3
"""Trace label-free pre-sentence binding through attention, MLP and residual components."""
from __future__ import annotations

import gc
import json
import math
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
P2378 = RESULT / "phase2378_c15601_c15920_label_free_binding_contract"
P2379 = RESULT / "phase2379_c15921_c16240_qwen_label_free_full_field"
OUT = RESULT / "phase2381_c16561_c16880_residual_component_routing"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2378 / "material/label_free_natural_binding.jsonl"
OUTPUT_STATE = P2379 / "raw/qwen4b_output_progress_anchors.float16.npy"
COMPONENT = OUT / "raw/qwen4b_pre_sentence_attention_mlp.float16.npy"
GATE = OUT / "raw/qwen4b_pre_sentence_mlp_intermediate.float16.npy"
ROUTING = OUT / "raw/qwen4b_pre_sentence_source_attention_mass.float16.npy"
PHASE = 2381
CAMPAIGN = "C16561-C16880"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2380_c16241_c16560_object_slot_progress_adjudication as adjudicate  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def panel_indices(rows: list[dict]) -> list[int]:
    discovery_groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row["partition"] == "discovery":
            discovery_groups[(row["family"], row["language"], row["surface"], row["reverse"], row["source_index"])].append(i)
    discovery = []
    for key in sorted(discovery_groups, key=str): discovery.extend(discovery_groups[key][:2])
    confirmation = [i for i, row in enumerate(rows) if row["partition"] == "confirmation"]
    lockbox = [i for i, row in enumerate(rows) if row["partition"] == "fresh_joint_lockbox"]
    result = sorted(discovery) + confirmation + lockbox
    if len(discovery) != 384 or len(confirmation) != 128 or len(lockbox) != 256: raise RuntimeError((len(discovery), len(confirmation), len(lockbox)))
    return result


def route_indices(rows: list[dict], panel: list[int]) -> list[int]:
    discovery = [i for i in panel if rows[i]["partition"] == "discovery"][::6][:64]
    confirmation = [i for i in panel if rows[i]["partition"] == "confirmation"]
    lockbox = [i for i in panel if rows[i]["partition"] == "fresh_joint_lockbox"][::2]
    result = discovery + confirmation + lockbox
    if len(result) != 320: raise RuntimeError(len(result))
    return result


def right_pad(sequences: list[list[int]], device: torch.device, pad: int):
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences): ids[i, :len(sequence)] = torch.tensor(sequence, device=device); mask[i, :len(sequence)] = 1
    return ids, mask, (mask.cumsum(1) - 1).clamp_min(0)


def pre_positions(row: dict) -> list[int]:
    return [len(row["prompt_ids"]) + span[0] - 1 for span in row["target_spans"]]


def collect_components(model, rows: list[dict], panel: list[int], batch_size: int = 2) -> dict:
    layers = list(model.model.layers); dim = int(model.config.hidden_size); intermediate = int(model.config.intermediate_size)
    component_shape = (len(panel), 4, len(layers), 2, dim); gate_shape = (len(panel), 4, len(layers), intermediate)
    progress_path = OUT / "raw/component_progress.json"
    if COMPONENT.exists() and GATE.exists() and progress_path.exists():
        components = np.lib.format.open_memmap(COMPONENT, mode="r+"); gates = np.lib.format.open_memmap(GATE, mode="r+")
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
    else:
        COMPONENT.parent.mkdir(parents=True, exist_ok=True)
        components = np.lib.format.open_memmap(COMPONENT, mode="w+", dtype=np.float16, shape=component_shape)
        gates = np.lib.format.open_memmap(GATE, mode="w+", dtype=np.float16, shape=gate_shape); completed = 0
    captures: dict[tuple, torch.Tensor] = {}; context: dict[str, Any] = {"positions": []}; handles = []
    for layer_index, layer in enumerate(layers):
        def attn_hook(_module, _inputs, output, layer_index=layer_index):
            value = output[0] if isinstance(output, tuple) else output
            captures[(layer_index, "attention")] = torch.stack([value[i, torch.tensor(pos, device=value.device)]
                                                                  for i, pos in enumerate(context["positions"])]).detach().float().cpu()
        def mlp_hook(_module, _inputs, output, layer_index=layer_index):
            captures[(layer_index, "mlp")] = torch.stack([output[i, torch.tensor(pos, device=output.device)]
                                                            for i, pos in enumerate(context["positions"])]).detach().float().cpu()
        def gate_hook(_module, inputs, layer_index=layer_index):
            value = inputs[0]
            captures[(layer_index, "gate")] = torch.stack([value[i, torch.tensor(pos, device=value.device)]
                                                             for i, pos in enumerate(context["positions"])]).detach().float().cpu()
        handles.append(layer.self_attn.register_forward_hook(attn_hook)); handles.append(layer.mlp.register_forward_hook(mlp_hook))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(gate_hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(panel), batch_size):
                indices = panel[start:start + batch_size]; batch = [rows[i] for i in indices]
                sequences = [row["prompt_ids"] + row["target_ids"] for row in batch]
                context["positions"] = [pre_positions(row) for row in batch]
                ids, mask, positions = right_pad(sequences, device, pad); captures.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for layer_index in range(len(layers)):
                    components[start:start + len(batch), :, layer_index, 0] = captures[(layer_index, "attention")].numpy().astype(np.float16)
                    components[start:start + len(batch), :, layer_index, 1] = captures[(layer_index, "mlp")].numpy().astype(np.float16)
                    gates[start:start + len(batch), :, layer_index] = captures[(layer_index, "gate")].numpy().astype(np.float16)
                components.flush(); gates.flush(); save(progress_path, {"completed": start + len(batch), "component_shape": component_shape, "gate_shape": gate_shape})
                if (start + len(batch)) % 64 == 0 or start + len(batch) == len(panel):
                    print(f"[phase2381 components] {start + len(batch)}/{len(panel)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        components.flush(); gates.flush(); close(components); close(gates)
    write_rows(OUT / "index/component_panel_rows.jsonl", [{"panel_index": p, "source_index": i, "case_id": rows[i]["case_id"],
               "partition": rows[i]["partition"], "family": rows[i]["family"], "language": rows[i]["language"],
               "surface": rows[i]["surface"], "reverse": rows[i]["reverse"], "source_perm": rows[i]["source_perm"]}
              for p, i in enumerate(panel)])
    return {"component_shape": list(component_shape), "gate_shape": list(gate_shape), "rows": len(panel)}


def collect_attention_routing(model, rows: list[dict], indices: list[int]) -> dict:
    layers = list(model.model.layers); heads = int(model.config.num_attention_heads)
    shape = (len(indices), len(layers), 4, heads, 4); progress_path = OUT / "raw/routing_progress.json"
    if ROUTING.exists() and progress_path.exists():
        mass = np.lib.format.open_memmap(ROUTING, mode="r+"); completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
    else:
        mass = np.lib.format.open_memmap(ROUTING, mode="w+", dtype=np.float16, shape=shape); completed = 0
    device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            for local in range(completed, len(indices)):
                row = rows[indices[local]]; sequence = row["prompt_ids"] + row["target_ids"]
                ids = torch.tensor([sequence], dtype=torch.long, device=device)
                result = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=torch.arange(len(sequence), device=device)[None],
                               output_attentions=True, use_cache=False, return_dict=True)
                anchors = pre_positions(row)
                for layer_index, attention in enumerate(result.attentions):
                    weights = attention[0, :, anchors].float()
                    for source_slot, (start, end) in enumerate(row["source_spans"]):
                        mass[local, layer_index, :, :, source_slot] = weights[:, :, start:end].sum(-1).T.cpu().numpy().astype(np.float16)
                mass.flush(); save(progress_path, {"completed": local + 1, "shape": shape})
                if (local + 1) % 32 == 0 or local + 1 == len(indices): print(f"[phase2381 routing] {local + 1}/{len(indices)}", flush=True)
    finally:
        mass.flush(); close(mass)
    write_rows(OUT / "index/routing_rows.jsonl", [{"routing_index": p, "source_index": i, "case_id": rows[i]["case_id"],
               "partition": rows[i]["partition"], "family": rows[i]["family"], "language": rows[i]["language"],
               "surface": rows[i]["surface"], "reverse": rows[i]["reverse"], "source_perm": rows[i]["source_perm"]}
              for p, i in enumerate(indices)])
    return {"shape": list(shape), "rows": len(indices), "heads": heads, "stored": "all heads x all layers x all four source-sentence masses"}


def panel_labels(rows: list[dict], indices: list[int]) -> np.ndarray:
    full = adjudicate.slot_labels(rows); return full[indices]


def eta_squared(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1]).astype(np.float32); y = labels.reshape(-1); mean = flat.mean(0)
    total = np.square(flat - mean).sum(0); between = np.zeros_like(total)
    for label in range(4):
        group = flat[y == label]; between += len(group) * np.square(group.mean(0) - mean)
    return between / np.maximum(total, 1e-8)


def decode_component(panel_rows: list[dict], values: np.ndarray, labels: np.ndarray, name: str) -> dict:
    splits = {part: np.asarray([i for i, row in enumerate(panel_rows) if row["partition"] == part])
              for part in ("discovery", "confirmation", "fresh_joint_lockbox")}
    train, confirm, lock = splits["discovery"], splits["confirmation"], splits["fresh_joint_lockbox"]
    layers = []; eta_path = OUT / f"derived/{name}_coordinate_slot_eta2.float32.npy"
    eta_path.parent.mkdir(parents=True, exist_ok=True)
    eta = np.lib.format.open_memmap(eta_path, mode="w+", dtype=np.float32, shape=(values.shape[2], values.shape[-1]))
    for layer in range(values.shape[2]):
        tx = np.asarray(values[train, :, layer], dtype=np.float32).reshape(-1, values.shape[-1]); ty = labels[train].reshape(-1)
        eta[layer] = eta_squared(np.asarray(values[train, :, layer]), labels[train])
        entry = {"layer": layer, "methods": {}}
        for scaled, method in ((False, "raw_centroid"), (True, "zscore_centroid")):
            item = {}
            for part, indices in (("confirmation", confirm), ("lockbox", lock)):
                vx = np.asarray(values[indices, :, layer], dtype=np.float32).reshape(-1, values.shape[-1]); vy = labels[indices].reshape(-1)
                pred = adjudicate.nearest_centroid(tx, ty, vx, scaled)
                item[part] = {"n": int(vy.size), "accuracy": float(np.mean(pred == vy))}
            entry["methods"][method] = item
        layers.append(entry)
    eta.flush(); close(eta)
    candidates = [(entry["methods"][method]["confirmation"]["accuracy"], entry, method) for entry in layers for method in entry["methods"]]
    _, selected, method = max(candidates, key=lambda item: (item[0], -item[1]["layer"], item[2]))
    return {"layers": layers, "selected": {"layer": selected["layer"], "method": method,
            **selected["methods"][method]["lockbox"]}, "chance": 0.25, "eta2_path": str(eta_path),
            "eta2_shape": [values.shape[2], values.shape[-1]]}


def residual_closure(panel: list[int], components: np.ndarray, state: np.ndarray) -> dict:
    entries = []
    for layer in range(components.shape[2]):
        before = np.asarray(state[panel, :, 0, layer], dtype=np.float32)
        after = np.asarray(state[panel, :, 0, layer + 1], dtype=np.float32)
        update = np.asarray(components[:, :, layer, 0], dtype=np.float32) + np.asarray(components[:, :, layer, 1], dtype=np.float32)
        residual = after - before; error = residual - update
        entries.append({"layer": layer, "relative_rmse": float(np.sqrt(np.square(error).mean()) /
                                                                   max(float(np.sqrt(np.square(residual).mean())), 1e-8)),
                        "cosine": float(np.sum(residual * update) /
                                        max(float(np.sqrt(np.sum(residual * residual) * np.sum(update * update))), 1e-8))})
    return {"layers": entries, "median_relative_rmse": float(np.median([x["relative_rmse"] for x in entries])),
            "median_cosine": float(np.median([x["cosine"] for x in entries])),
            "meaning": "numerical architecture closure, not evidence that either component copies an object"}


def analyze_routing(rows: list[dict], indices: list[int], mass: np.ndarray) -> dict:
    labels = adjudicate.slot_labels(rows)[indices]
    parts = {part: np.asarray([i for i, source in enumerate(indices) if rows[source]["partition"] == part])
             for part in ("discovery", "confirmation", "fresh_joint_lockbox")}
    confirmation, lock = parts["confirmation"], parts["fresh_joint_lockbox"]
    candidates = []
    for layer in range(mass.shape[1]):
        aggregate_pred = np.asarray(mass[confirmation, layer], dtype=np.float32).mean(2).argmax(-1)
        candidates.append((float(np.mean(aggregate_pred == labels[confirmation])), layer, "all_heads_mean", -1))
        for head in range(mass.shape[3]):
            pred = np.asarray(mass[confirmation, layer, :, head], dtype=np.float32).argmax(-1)
            candidates.append((float(np.mean(pred == labels[confirmation])), layer, "single_head", head))
    confirmation_accuracy, layer, method, head = max(candidates, key=lambda item: (item[0], -item[1], -item[3]))
    selected_mass = np.asarray(mass[lock, layer], dtype=np.float32).mean(2) if method == "all_heads_mean" else np.asarray(mass[lock, layer, :, head], dtype=np.float32)
    pred = selected_mass.argmax(-1); truth = labels[lock]
    rng = np.random.default_rng(2381); shuffled = truth.reshape(-1).copy(); rng.shuffle(shuffled)
    return {"selection_rule": "confirmation chooses layer/head; independent lockbox evaluated once",
            "selected": {"layer": layer, "method": method, "head": None if head < 0 else head,
                         "confirmation_accuracy": confirmation_accuracy, "lockbox_accuracy": float(np.mean(pred == truth)),
                         "lockbox_shuffled_label_accuracy": float(np.mean(pred.reshape(-1) == shuffled)), "n": int(truth.size)},
            "chance": 0.25, "boundary": "attention mass is observational routing evidence, not content-copy causality"}


def pre_sentence_coordinate_controls(rows: list[dict]) -> dict:
    source = np.load(P2379 / "raw/qwen4b_source_sentence_end.float16.npy", mmap_mode="r")
    output = np.load(OUTPUT_STATE, mmap_mode="r"); splits = adjudicate.split_indices(rows); labels = adjudicate.slot_labels(rows)
    train, confirm, lock = splits["discovery"], splits["confirmation"], splits["fresh_joint_lockbox"]
    candidates = []
    for qpoint in range(output.shape[3]):
        params = adjudicate.fit_params(source, output, rows, train, labels, 0, qpoint, True)
        score = adjudicate.match_accuracy(source, output, confirm, labels, 0, qpoint, params, rows)
        candidates.append((score, qpoint, params))
    confirmation, qpoint, params = max(candidates, key=lambda item: (item[0], -item[1]))
    baseline = adjudicate.match_accuracy(source, output, lock, labels, 0, qpoint, params, rows)
    perm = np.random.default_rng(2381).permutation(output.shape[-1])
    perm_params = adjudicate.fit_params(source, output, rows, train, labels, 0, qpoint, True, coordinate_perm=perm)
    wrong_params = adjudicate.fit_params(source, output, rows, train, labels, 0, qpoint, True, wrong_pair=True)
    result = {"qpoint": qpoint, "confirmation_accuracy": confirmation, "lockbox_accuracy": baseline,
              "coordinate_permuted_lockbox_accuracy": adjudicate.match_accuracy(source, output, lock, labels, 0, qpoint, perm_params, rows, perm),
              "wrong_source_fit_lockbox_accuracy": adjudicate.match_accuracy(source, output, lock, labels, 0, qpoint, wrong_params, rows)}
    close(source); close(output); return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 句前绑定的Attention－MLP－残差全坐标路由候选追踪（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 因Phase2380无标签pre-sentence对象匹配锁箱通过，本Phase在768条分层panel上采集36层每个目标句开始前的Attention输出、MLP输出全部2560坐标，以及SwiGLU乘积送入down-proj之前的全部中间神经元；另对320条分层样本保留所有层×所有head到四个来源句token跨度的完整注意力质量。比较各分量的fresh槽位解码、全坐标$\eta^2$、残差数值闭合和来源句注意力命中；不使用Top-K。

$$H_{{l+1}}-H_l=A_l+M_l,\qquad
g_l=\operatorname{{SiLU}}(W_{{gate}}\tilde H_l)\odot(W_{{up}}\tilde H_l),\qquad
R_{{l,h,t,s}}=\sum_{{p\in S_s}}\alpha_{{l,h}}(a_t,p).$$

**结果汇总。** 组件采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；残差闭合 `{json.dumps(result['residual_closure'], ensure_ascii=False)}`；Attention输出 `{json.dumps(result['attention_component']['selected'], ensure_ascii=False)}`；MLP输出 `{json.dumps(result['mlp_component']['selected'], ensure_ascii=False)}`；MLP中间神经元 `{json.dumps(result['mlp_intermediate']['selected'], ensure_ascii=False)}`；全头路由 `{json.dumps(result['attention_routing']['selected'], ensure_ascii=False)}`；句前坐标负控 `{json.dumps(result['pre_sentence_controls'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2381_c16561_c16880_residual_component_routing.py`；完整component/gate/routing场、逐坐标$\eta^2$和索引位于 `tests/glm5/result/phase2381_c16561_c16880_residual_component_routing`。

**理论进展、问题硬伤与结论。** 残差等式通过只验证实现分解。组件可解码说明该增量与来源槽位相关；注意力质量命中说明某些head更关注正确来源句；两者都不自动证明“复制”。MLP中间神经元不是附件所谓单一$\beta$门控系数，逐坐标对角响应也不能与它直接等同。若坐标置乱控制仍高，保留的是分布式对象相似性而非固定坐标齿轮。下一Phase在四模型上只复验行为合格的无标签边界/句前结构，不跨模型对齐坐标编号。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = [row for row in read_rows(MATERIAL) if row["task"] == "exact_copy"]
    panel = panel_indices(rows); route = route_indices(rows, panel)
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        collection = collect_components(model, rows, panel)
        routing_collection = collect_attention_routing(model, rows, route)
    finally:
        model_utils.release_model(model); del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    components = np.load(COMPONENT, mmap_mode="r"); gates = np.load(GATE, mmap_mode="r")
    states = np.load(OUTPUT_STATE, mmap_mode="r"); panel_rows = [rows[i] for i in panel]; labels = panel_labels(rows, panel)
    residual = residual_closure(panel, components, states)
    attention_component = decode_component(panel_rows, components[:, :, :, 0], labels, "attention_output")
    mlp_component = decode_component(panel_rows, components[:, :, :, 1], labels, "mlp_output")
    mlp_intermediate = decode_component(panel_rows, gates, labels, "mlp_intermediate")
    close(components); close(gates); close(states)
    mass = np.load(ROUTING, mmap_mode="r"); attention_routing = analyze_routing(rows, route, mass); close(mass)
    pre_controls = pre_sentence_coordinate_controls(rows)
    checks = {
        "component_shape": collection["component_shape"] == [768, 4, 36, 2, 2560],
        "gate_shape": collection["gate_shape"][0:3] == [768, 4, 36],
        "routing_shape": routing_collection["shape"][0:3] == [320, 36, 4],
        "residual_numerical_closure": residual["median_cosine"] > 0.999,
        "decoders_complete": all(len(item["layers"]) == 36 for item in (attention_component, mlp_component, mlp_intermediate)),
        "finite": all(math.isfinite(x) for x in (attention_component["selected"]["accuracy"], mlp_component["selected"]["accuracy"],
                                                  mlp_intermediate["selected"]["accuracy"], attention_routing["selected"]["lockbox_accuracy"])),
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "routing_collection": routing_collection,
              "residual_closure": residual, "attention_component": attention_component, "mlp_component": mlp_component,
              "mlp_intermediate": mlp_intermediate, "attention_routing": attention_routing,
              "pre_sentence_controls": pre_controls, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
