#!/usr/bin/env python3
"""Sequential Qwen14B/GLM4/DS7B long-range pointer and S4 operator replication."""
from __future__ import annotations

import gc
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2368 = RESULT / "phase2368_c12481_c12720_longrange_operator_contract"
OUT = RESULT / "phase2374_c14321_c14640_crossmodel_longrange_operator"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2368 / "material/long_sentence_permutation.jsonl"
PHASE = 2374
CAMPAIGN = "C14321-C14640"
MODELS = ("qwen14b", "glm4", "deepseek7b")
FAMILIES = ("temporal_narrative", "causal_process", "taxonomy_explanation", "spatial_route",
            "procedure", "comparison", "dialogue_coreference", "scientific_description")

warnings.filterwarnings("ignore", message=r"MatMul8bitLt: inputs will be cast.*")
sys.path.insert(0, str(TESTS))
import phase2339_c7401_c7600_crossmodel_fixed_ab_replication as loader  # noqa: E402
import phase2370_c13041_c13360_pointer_group_operator as basic  # noqa: E402
import phase2371_c13361_c13680_advanced_math_tournament as advanced  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def paths(key: str) -> dict[str, Path]:
    base = OUT / key
    return {"base": base, "rows": base / "material/rows.jsonl", "states": base / "raw/boundary.float16.npy",
            "decisions": base / "raw/decisions.float32.npy", "progress": base / "raw/progress.json",
            "analysis": base / "analysis/final.json"}


def modules(model):
    return [model.model.embed_tokens if hasattr(model.model, "embed_tokens") else model.get_input_embeddings(),
            *list(model.model.layers), model.model.norm]


def left_pad(sequences: list[list[int]], device: torch.device, pad: int):
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences): ids[i, -len(seq):] = torch.tensor(seq, device=device); mask[i, -len(seq):] = 1
    return ids, mask, (mask.cumsum(1) - 1).clamp_min(0)


def compile_rows(tokenizer) -> tuple[list[dict], dict]:
    source = [r for r in read_rows(MATERIAL) if r["task"] == "index_only" and r["unit"] == 5]
    rows, collisions = [], 0
    for row in source:
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        target_ids = tokenizer.encode(row["target"], add_special_tokens=False)
        foil_ids = tokenizer.encode(row["foil"], add_special_tokens=False)
        collisions += int(target_ids[0] == foil_ids[0])
        rows.append({**row, "model_index": len(rows), "prompt_ids": prompt_ids, "target_ids": target_ids, "foil_ids": foil_ids,
                     "target_first_id": target_ids[0], "foil_first_id": foil_ids[0], "first_token_collision": target_ids[0] == foil_ids[0]})
    return rows, {"rows": len(rows), "expected": 768, "first_token_collisions": collisions,
                  "token_range": [min(len(r["prompt_ids"]) for r in rows), max(len(r["prompt_ids"]) for r in rows)]}


def collect_model(key: str, model, rows: list[dict]) -> dict:
    p = paths(key); qmods = modules(model); dim = int(model.get_input_embeddings().weight.shape[1]); shape = (len(rows), len(qmods), dim)
    if p["states"].exists() and p["decisions"].exists() and p["progress"].exists():
        completed = int(json.loads(p["progress"].read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(p["states"], mode="r+"); decisions = np.lib.format.open_memmap(p["decisions"], mode="r+")
    else:
        completed = 0; p["states"].parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(p["states"], mode="w+", dtype=np.float16, shape=shape)
        decisions = np.lib.format.open_memmap(p["decisions"], mode="w+", dtype=np.float32, shape=(len(rows), 5))
    captures = {}; handles = []
    for qi, module in enumerate(qmods):
        def hook(_module, _inputs, output, qi=qi): captures[qi] = (output[0] if isinstance(output, tuple) else output)[:, -1].detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    batch_size = int(loader.MODEL_SPECS[key]["batch"])
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]; ids, mask, pos = left_pad([r["prompt_ids"] for r in batch], device, pad); captures.clear()
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
                for qi in range(len(qmods)): states[start:start + len(batch), qi] = captures[qi].float().cpu().numpy().astype(np.float16)
                logits = output.logits[:, -1].float()
                for local, row in enumerate(batch):
                    t, f = row["target_first_id"], row["foil_first_id"]; tl, fl = float(logits[local, t]), float(logits[local, f])
                    decisions[start + local] = [tl, fl, tl - fl, float(tl > fl), float(int(logits[local].argmax()) == t)]
                states.flush(); decisions.flush(); save(p["progress"], {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 96 == 0 or start + len(batch) == len(rows): print(f"[phase2374 {key}] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
    return {"shape": list(shape), "batch_size": batch_size, "quantization": loader.MODEL_SPECS[key]["quant"]}


def build_field(rows: list[dict], states: np.ndarray, qpoint: int):
    groups = {}
    for i, r in enumerate(rows):
        key = (r["family"], r["language"], tuple(r["source_perm"]))
        groups.setdefault(key, {})[r["permutation_index"]] = i
    keys = sorted(groups, key=str); row_index = np.asarray([[groups[k][pi] for pi in range(24)] for k in keys])
    return keys, row_index, np.asarray(states[row_index, qpoint], dtype=np.float32)


def group_splits(keys: list[tuple]) -> dict[str, np.ndarray]:
    return {"train": np.asarray([i for i, k in enumerate(keys) if FAMILIES.index(k[0]) <= 3 and k[2] == (0, 1, 2, 3)]),
            "confirmation": np.asarray([i for i, k in enumerate(keys) if 4 <= FAMILIES.index(k[0]) <= 5 and k[2] == (2, 0, 3, 1)]),
            "lockbox": np.asarray([i for i, k in enumerate(keys) if FAMILIES.index(k[0]) >= 6 and k[2] == (2, 0, 3, 1)])}


def pointer_data(keys: list[tuple], row_index: np.ndarray, groups: np.ndarray):
    indices, labels = [], []
    for gi in groups:
        source = keys[gi][2]
        for pi, row in enumerate(row_index[gi]): indices.append(row); labels.append(source.index(basic.PERMS[pi][0]))
    return np.asarray(indices), np.asarray(labels)


def irrep_spectrum(field: np.ndarray, groups: np.ndarray, projectors: dict[str, np.ndarray]) -> dict:
    centered = field[groups] - field[groups].mean(1, keepdims=True); energies = {}
    for name, projector in projectors.items():
        projected = np.einsum("ab,gbd->gad", projector, centered, optimize=True); energies[name] = float(np.square(projected.astype(np.float64)).sum())
    total = sum(energies.values()); return {name: value / max(total, 1e-12) for name, value in energies.items()}


def analyze_model(key: str, rows: list[dict], material_audit: dict, collection: dict) -> dict:
    p = paths(key); states = np.load(p["states"], mmap_mode="r"); decisions = np.load(p["decisions"], mmap_mode="r")
    valid = np.asarray([not r["first_token_collision"] for r in rows]); behavior = {"valid_rows": int(valid.sum()),
        "target_over_foil": float(np.asarray(decisions[valid, 3]).mean()), "full_vocab_argmax": float(np.asarray(decisions[valid, 4]).mean())}
    keys, row_index, _ = build_field(rows, states, 0); splits = group_splits(keys)
    pointer_layers, operator_layers = [], []
    methods = ("identity", "translation", "diagonal_affine", "direct_template", "coordinate_permuted_affine")
    for qpoint in range(states.shape[1]):
        _, _, field = build_field(rows, states, qpoint)
        train_i, train_y = pointer_data(keys, row_index, splits["train"]); confirm_i, confirm_y = pointer_data(keys, row_index, splits["confirmation"]); lock_i, lock_y = pointer_data(keys, row_index, splits["lockbox"])
        pointer = {"qpoint": qpoint, "methods": {}}
        tx = np.asarray(states[train_i, qpoint], dtype=np.float32)
        for scaled in (False, True):
            name = "zscore_centroid" if scaled else "raw_centroid"
            pc = basic.nearest_centroid(tx, train_y, np.asarray(states[confirm_i, qpoint], dtype=np.float32), scaled)
            pl = basic.nearest_centroid(tx, train_y, np.asarray(states[lock_i, qpoint], dtype=np.float32), scaled)
            pointer["methods"][name] = {"confirmation_accuracy": float((pc == confirm_y).mean()), "lockbox_accuracy": float((pl == lock_y).mean())}
        pointer_layers.append(pointer)
        translations, slopes, intercepts, direct = basic.fit_operators(field, splits["train"])
        metric = {"qpoint": qpoint, "methods": {}}
        for method in methods:
            ac, pc = basic.predict_responses(field, splits["confirmation"], translations, slopes, intercepts, direct, method)
            al, pl = basic.predict_responses(field, splits["lockbox"], translations, slopes, intercepts, direct, method)
            rc, _ = basic.response_r2(ac, pc); rl, _ = basic.response_r2(al, pl)
            metric["methods"][method] = {"confirmation_response_r2": rc, "lockbox_response_r2": rl}
        operator_layers.append(metric)
    pointer_choices = [(m["confirmation_accuracy"], layer["qpoint"], name, m) for layer in pointer_layers for name, m in layer["methods"].items()]
    _, pq, pm, pv = max(pointer_choices)
    operator_choices = [(m["confirmation_response_r2"], layer["qpoint"], name, m) for layer in operator_layers for name, m in layer["methods"].items()
                        if name != "coordinate_permuted_affine" and np.isfinite(m["confirmation_response_r2"])]
    _, oq, om, ov = max(operator_choices)
    _, _, selected_field = build_field(rows, states, oq); projectors, _ = advanced.projectors(); spectrum = irrep_spectrum(selected_field, splits["lockbox"], projectors)
    same = operator_layers[oq]["methods"]
    result = {"model": key, "model_label": loader.MODEL_SPECS[key]["label"], "material": material_audit, "collection": collection,
              "behavior": behavior, "splits": {k: len(v) for k, v in splits.items()},
              "pointer": {"selected_qpoint": pq, "relative_depth": pq / (states.shape[1] - 1), "method": pm,
                          "confirmation_accuracy": pv["confirmation_accuracy"], "lockbox_accuracy": pv["lockbox_accuracy"], "chance": 0.25},
              "operator": {"selected_qpoint": oq, "relative_depth": oq / (states.shape[1] - 1), "method": om,
                           "confirmation_response_r2": ov["confirmation_response_r2"], "lockbox_response_r2": ov["lockbox_response_r2"],
                           "same_layer": same, "passed": om in ("translation", "diagonal_affine") and ov["lockbox_response_r2"] > 0
                           and ov["lockbox_response_r2"] > same["direct_template"]["lockbox_response_r2"]
                           and ov["lockbox_response_r2"] > same["coordinate_permuted_affine"]["lockbox_response_r2"]},
              "selected_lockbox_irrep_spectrum": spectrum,
              "replication_flags": {"behavior_target_over_foil_ge_0_7": behavior["target_over_foil"] >= .7,
                                    "pointer_lockbox_gt_0_5": pv["lockbox_accuracy"] > .5}}
    save(p["analysis"], result); return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen14B、GLM4、DS7B顺序式长距离算子复现（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 按显存安全顺序一次只加载一个模型：Qwen3-14B-NF4、GLM4-9B-INT8、DeepSeek-R1-Distill-Qwen-7B-INT8。每模型重新分词同一冻结unit5锁箱：8内容族×中英×2来源顺序×$S_4$全部24排列=768条，采集该架构的embedding、全部block和final norm全坐标。前4族/$\sigma=e$训练，5–6族/新$\sigma$确认，最后2族/新$\sigma$锁箱；同场比较来源槽位解码、逐坐标相邻交换、直接模板和坐标置乱。

$$
T_i(h)_j=a_{{i,j}}h_j+b_{{i,j}},\qquad
\mathrm{{depth}}_{{rel}}=q/(Q-1).
$$

**结果汇总。** 模型汇总 `{json.dumps(result['models'], ensure_ascii=False)}`；跨模型判定 `{json.dumps(result['crossmodel'], ensure_ascii=False)}`。量化精度、分词、层数和维数均不同，因此只比较行为、相对深度、锁箱预测与$S_4$不可约能量比例，不比较坐标编号。

**相关文件。** 脚本 `tests/glm5/phase2374_c14321_c14640_crossmodel_longrange_operator.py`；逐模型材料、全坐标场和结果位于 `tests/glm5/result/phase2374_c14321_c14640_crossmodel_longrange_operator`。

**理论进展、问题硬伤与结论。** 跨模型复现若存在，只能支持功能/响应形式复用，不能支持坐标同一或模型间微分同胚。4/8bit量化、chat模型未套聊天模板、英文标记和中英分词差异都是硬伤；行为不合格模型的结构只作描述，不进入普遍机制结论。下一Phase自动修复fresh知识链重复与首token合并，再决定相邻交换规律是否值得扩展到$S_5$/不同句长。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    source = read_rows(MATERIAL); models = {}
    for key in MODELS:
        p = paths(key)
        if p["analysis"].exists(): models[key] = json.loads(p["analysis"].read_text(encoding="utf-8")); continue
        model, tokenizer, _ = loader.load_model(key)
        try:
            rows, audit = compile_rows(tokenizer); write_rows(p["rows"], rows)
            collection = collect_model(key, model, rows)
        finally:
            del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
            print(f"[phase2374] released {key}; gpu={torch.cuda.memory_allocated()/1e9:.3f}GB", flush=True)
        models[key] = analyze_model(key, rows, audit, collection)
    crossmodel = {"behavior_qualified_models": [k for k, v in models.items() if v["replication_flags"]["behavior_target_over_foil_ge_0_7"]],
                  "pointer_replicated_models": [k for k, v in models.items() if v["replication_flags"]["pointer_lockbox_gt_0_5"]],
                  "operator_replicated_models": [k for k, v in models.items() if v["operator"]["passed"]],
                  "universal_operator_claim_passed": all(v["operator"]["passed"] and v["replication_flags"]["behavior_target_over_foil_ge_0_7"] for v in models.values())}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "crossmodel": crossmodel}
    save(final_path, result); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
