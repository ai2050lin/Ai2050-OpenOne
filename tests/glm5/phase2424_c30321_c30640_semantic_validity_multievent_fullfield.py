#!/usr/bin/env python3
"""Qwen4B full-coordinate multi-event semantic-validity field and stable component closure."""
from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
P2423 = TESTS / "result/phase2423_c30001_c30320_semantic_validity_behavior_contract"
OUT = TESTS / "result/phase2424_c30321_c30640_semantic_validity_multievent_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2424
CAMPAIGN = "C30321-C30640"
EVENTS = ("fact1_relation", "query_end", "answer_boundary")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


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


def open_field(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float16, shape=shape)


def pad_right(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = 1
    positions = (mask.cumsum(1) - 1).clamp_min(0)
    return ids, mask, positions


def reduced_index(rows: list[dict]) -> list[dict]:
    result = []
    for source_index, row in enumerate(rows):
        mapping = {event["event"]: int(event["token_index"]) for event in row["event_tokens"]}
        if not all(event in mapping for event in EVENTS):
            raise RuntimeError((row["case_id"], sorted(mapping)))
        result.append({key: row[key] for key in (
            "case_id", "config_id", "task", "family", "fact_family", "unit", "language", "surface",
            "surface_class", "direction", "variant", "query_role", "candidate_order", "target_candidate_slot",
            "partition", "source", "target", "answer", "foil", "prompt_token_count", "prompt_ids", "target_ids", "foil_ids")})
        result[-1].update({"source_index": source_index, "event_names": list(EVENTS),
                           "event_token_indices": [mapping[event] for event in EVENTS]})
    return result


def collect_events(model, rows: list[dict], batch_size: int = 4) -> dict:
    blocks = list(model.model.layers)
    qmods = field_utils.modules(model)
    n, layers, qcount, dim = len(rows), len(blocks), len(qmods), int(model.config.hidden_size)
    paths = {name: OUT / f"raw/semantic_validity_{name}_event.float16.npy" for name in ("state", "attention", "mlp")}
    fields = {
        "state": open_field(paths["state"], (n, qcount, len(EVENTS), dim)),
        "attention": open_field(paths["attention"], (n, layers, len(EVENTS), dim)),
        "mlp": open_field(paths["mlp"], (n, layers, len(EVENTS), dim)),
    }
    progress = OUT / "raw/event_progress.json"
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    state_capture: dict[int, torch.Tensor] = {}
    attn_capture: dict[int, torch.Tensor] = {}
    mlp_capture: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(qmods):
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
                indices = [torch.tensor(row["event_token_indices"], dtype=torch.long) for row in batch]
                for qpoint in range(qcount):
                    tensor = state_capture[qpoint].float().cpu()
                    fields["state"][start:start + len(batch), qpoint] = torch.stack(
                        [tensor[i, indices[i]] for i in range(len(batch))]).numpy().astype(np.float16)
                for layer in range(layers):
                    for capture, name in ((attn_capture, "attention"), (mlp_capture, "mlp")):
                        tensor = capture[layer].float().cpu()
                        fields[name][start:start + len(batch), layer] = torch.stack(
                            [tensor[i, indices[i]] for i in range(len(batch))]).numpy().astype(np.float16)
                for field in fields.values():
                    field.flush()
                save(progress, {"completed": start + len(batch)})
                if (start + len(batch)) % 128 == 0 or start + len(batch) == n:
                    print(f"[phase2424 events] {start + len(batch)}/{n}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        for field in fields.values():
            field.flush(); close(field)
    shapes = {"state": [n, qcount, len(EVENTS), dim], "attention": [n, layers, len(EVENTS), dim],
              "mlp": [n, layers, len(EVENTS), dim]}
    return {name: {"path": str(path), "shape": shapes[name], "bytes": path.stat().st_size} for name, path in paths.items()}


def reference_rows(rows: list[dict]) -> list[dict]:
    chosen = {}
    for row in rows:
        if int(row["unit"]) != 6 or row["language"] != "zh" or row["surface"] != "natural" or int(row["direction"]) != 0:
            continue
        key = (row["family"], row["variant"], row["query_role"])
        chosen.setdefault(key, row)
    result = [chosen[key] for key in sorted(chosen)]
    if len(result) != 48:
        raise RuntimeError(("reference_count", len(result)))
    return result


def collect_all_token_state(model, rows: list[dict], batch_size: int = 2) -> dict:
    selected = reference_rows(rows)
    modules = field_utils.modules(model)
    qcount, dim = len(modules), int(model.config.hidden_size)
    total = sum(len(row["prompt_ids"]) * qcount for row in selected)
    path = OUT / "raw/reference_all_token_state.float16.npy"
    values = open_field(path, (total, dim))
    progress = OUT / "raw/reference_all_token_progress.json"
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed_cases"]) if progress.exists() else 0
    offsets, cursor = [], 0
    for row in selected[:completed]:
        count = len(row["prompt_ids"]); begin = cursor; cursor += count * qcount
        offsets.append({"case_id": row["case_id"], "offset": begin, "tokens": count, "qpoints": qcount,
                        "prompt_ids": row["prompt_ids"]})
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(modules):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(selected), batch_size):
                batch = selected[start:start + batch_size]
                ids, mask, positions = pad_right([row["prompt_ids"] for row in batch], device, pad)
                captures.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for local, row in enumerate(batch):
                    count = len(row["prompt_ids"]); begin = cursor
                    for qpoint in range(qcount):
                        array = captures[qpoint][local, :count].float().cpu().numpy().astype(np.float16)
                        values[cursor:cursor + count] = array; cursor += count
                    offsets.append({"case_id": row["case_id"], "offset": begin, "tokens": count, "qpoints": qcount,
                                    "prompt_ids": row["prompt_ids"]})
                values.flush(); save(progress, {"completed_cases": start + len(batch), "cursor": cursor})
                if (start + len(batch)) % 12 == 0 or start + len(batch) == len(selected):
                    print(f"[phase2424 all-token] {start + len(batch)}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        values.flush(); close(values)
    if cursor != total:
        raise RuntimeError(("all_token_cursor", cursor, total))
    write_rows(OUT / "index/reference_all_token_offsets.jsonl", offsets)
    return {"path": str(path), "shape": [total, dim], "bytes": path.stat().st_size,
            "cases": len(selected), "qpoints": qcount, "total_token_qpoints": total}


def stable_closure(collection: dict) -> dict:
    state = np.load(collection["state"]["path"], mmap_mode="r")
    attention = np.load(collection["attention"]["path"], mmap_mode="r")
    mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    indices = np.linspace(0, state.shape[0] - 1, 64, dtype=np.int64)
    dot = norm_truth = norm_pred = residual_ss = 0.0
    values = 0
    finite = True
    for index in indices:
        truth = np.asarray(state[index, 1:-1], dtype=np.float64) - np.asarray(state[index, :-2], dtype=np.float64)
        pred = np.asarray(attention[index], dtype=np.float64) + np.asarray(mlp[index], dtype=np.float64)
        residual = truth - pred
        t, p, r = truth.ravel(), pred.ravel(), residual.ravel()
        finite = finite and bool(np.isfinite(t).all() and np.isfinite(p).all())
        dot += float(np.dot(t, p)); norm_truth += float(np.dot(t, t)); norm_pred += float(np.dot(p, p))
        residual_ss += float(np.dot(r, r)); values += t.size
    denominator = math.sqrt(norm_truth * norm_pred)
    raw_cosine = dot / denominator if denominator else 0.0
    cosine = min(1.0, max(-1.0, raw_cosine))
    result = {"sample_rows": len(indices), "values": values, "accumulator": "float64",
              "mse": residual_ss / max(values, 1), "relative_rmse": math.sqrt(residual_ss / max(norm_truth, 1e-300)),
              "raw_cosine": raw_cosine, "cosine": cosine, "within_mathematical_bounds": -1 <= raw_cosine <= 1,
              "finite": finite, "phase2415_out_of_range_reproduced": raw_cosine > 1}
    for value in (state, attention, mlp):
        close(value)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 语义有效性多事件H/A/M全坐标场与稳定组件闭合（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2423全部6144条在Qwen3-4B BF16权重上，于事实谓词结束、查询结束和答案边界三个事件采集embedding、36个block输出、final norm的全部2560坐标，并同步采集36层Attention输出与MLP输出。另固定unit6/中文/natural/方向0，覆盖八族×三有效性×双角色48条，保留prompt每个token×38 qpoint×全部坐标的状态路径。所有源数组保持物理坐标次序，不作Top-K/PCA/压缩；组件闭合从64条均匀样本以float64逐块累加，专门修复Phase2415大数组float32余弦越界。

$$H_{{q+1,e}}-H_{{q,e}}=A_{{q,e}}+M_{{q,e}}+\varepsilon_{{q,e}},$$

$$\cos_{{64}}=\frac{{\sum_i\langle\Delta H_i,A_i+M_i\rangle_{{64}}}}{{\sqrt{{\sum_i\|\Delta H_i\|_{{64}}^2\sum_i\|A_i+M_i\|_{{64}}^2}}}}.$$

**结果汇总。** 事件场 `{json.dumps(result['collection'], ensure_ascii=False)}`；全token参考 `{json.dumps(result['all_token'], ensure_ascii=False)}`；稳定组件闭合 `{json.dumps(result['component_closure'], ensure_ascii=False)}`；总量`{result['total_gib']:.3f}` GiB；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2424_c30321_c30640_semantic_validity_multievent_fullfield.py`；6144条精简索引、三个事件的state/attention/mlp全坐标原始场、48条全token状态及offset、closure和final位于`tests/glm5/result/phase2424_c30321_c30640_semantic_validity_multievent_fullfield`。未修改其他Markdown。

**分析与理论进展。** 本Phase提供后续统一的“外部谓词操作—事件位置—层—物理坐标—组件来源”观测底座。组件物理可加只说明残差流来源，不给Attention或MLP预先命名语言功能。三个事件可以检验语义特异性何时出现、是否在查询后改变以及是否只是答案槽主效应；全token参考防止事件终点遗漏中间形成过程。

**问题硬伤与结论。** float16落盘量化了BF16源；全token只保留48条代表而非6144条全部token，主样本则在三个预注册事件全覆盖。hook得到的是所有head混合后的Attention输出和完成门控/down projection后的MLP输出，不是内部微单元。组件闭合是实现/采集一致性检查，不是语义机制闭合。原始场暂留到Phase2431统一派生、可视化核验后再清理。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    source = read_rows(P2423 / "qwen4b/index/semantic_validity_rows.jsonl")
    rows = reduced_index(source)
    write_rows(OUT / "index/semantic_validity_rows.jsonl", rows)
    model, tokenizer, label = capability.load_model("qwen4b")
    try:
        collection = collect_events(model, rows)
        all_token = collect_all_token_state(model, rows)
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    closure = stable_closure(collection)
    total_bytes = sum(item["bytes"] for item in collection.values()) + all_token["bytes"]
    checks = {
        "state_shape": collection["state"]["shape"] == [6144, 38, 3, 2560],
        "component_shapes": collection["attention"]["shape"] == [6144, 36, 3, 2560] and collection["mlp"]["shape"] == [6144, 36, 3, 2560],
        "reference_48": all_token["cases"] == 48,
        "float64_closure_bounded": closure["accumulator"] == "float64" and closure["within_mathematical_bounds"],
        "closure_high": closure["cosine"] > 0.999,
        "full_coordinates_no_topk": True,
        "raw_retained_for_campaign": all(Path(item["path"]).exists() for item in collection.values()),
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": label, "precision": "BF16 weights / float16 field storage",
              "events": list(EVENTS), "collection": collection, "all_token": all_token,
              "component_closure": closure, "total_bytes": total_bytes, "total_gib": total_bytes / 2**30,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/component_closure.json", closure)
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
