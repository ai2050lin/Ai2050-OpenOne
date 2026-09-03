#!/usr/bin/env python3
"""Sequential Qwen14B, GLM4 and DS7B replication of label-free pre-sentence binding."""
from __future__ import annotations

import argparse
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
P2378 = RESULT / "phase2378_c15601_c15920_label_free_binding_contract"
P2380 = RESULT / "phase2380_c16241_c16560_object_slot_progress_adjudication"
P2381 = RESULT / "phase2381_c16561_c16880_residual_component_routing"
OUT = RESULT / "phase2382_c16881_c17200_crossmodel_label_free_binding"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2378 / "material/label_free_natural_binding.jsonl"
PHASE = 2382
CAMPAIGN = "C16881-C17200"
MODELS = ("qwen14b", "glm4", "deepseek7b")

warnings.filterwarnings("ignore", message=r"MatMul8bitLt: inputs will be cast.*")
sys.path.insert(0, str(TESTS))
import phase2339_c7401_c7600_crossmodel_fixed_ab_replication as loader  # noqa: E402
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


def paths(key: str) -> dict[str, Path]:
    base = OUT / key
    return {"base": base, "rows": base / "material/rows.jsonl", "source": base / "raw/source_sentence_end.float16.npy",
            "output": base / "raw/output_pre_sentence.float16.npy", "scores": base / "raw/sequence_scores.float32.npy",
            "progress": base / "raw/teacher_progress.json", "foil_progress": base / "raw/foil_progress.json",
            "final": base / "analysis/final.json"}


def panel_source_indices() -> list[int]:
    index = read_rows(P2381 / "index/component_panel_rows.jsonl")
    result = [int(row["source_index"]) for row in index]
    if len(result) != 768: raise RuntimeError(len(result))
    return result


def encode_segments(tokenizer, segments: list[tuple[str, bool]]) -> tuple[list[int], list[list[int]]]:
    ids: list[int] = []; spans = []
    for text, marked in segments:
        start = len(ids); ids += [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        if marked: spans.append([start, len(ids)])
    return ids, spans


def retokenize(tokenizer, source_rows: list[dict], source_indices: list[int]) -> tuple[list[dict], dict]:
    rows = []
    for source_index in source_indices:
        source = source_rows[source_index]; cursor = 0; segments = []
        for sentence in source["source_lines"]:
            start = source["prompt"].find(sentence, cursor)
            if start < 0: raise RuntimeError((source["case_id"], sentence))
            segments.append((source["prompt"][cursor:start], False)); segments.append((sentence, True)); cursor = start + len(sentence)
        segments.append((source["prompt"][cursor:], False))
        prompt_ids, source_spans = encode_segments(tokenizer, segments)
        target_segments, foil_segments = [], []
        for slot, sentence in enumerate(source["target_sentences"]):
            if slot: target_segments.append(("\n", False))
            target_segments.append((sentence, True))
        for slot, sentence_id in enumerate(source["foil_order"]):
            if slot: foil_segments.append(("\n", False))
            foil_segments.append((source["sentences"][sentence_id], True))
        target_ids, target_spans = encode_segments(tokenizer, target_segments); foil_ids, _ = encode_segments(tokenizer, foil_segments)
        rows.append({**{key: source[key] for key in ("case_id", "family", "unit", "language", "surface", "reverse",
                                                       "source_index", "source_perm", "target_order", "foil_order", "partition")},
                     "model_index": len(rows), "source_design_index": source_index, "prompt_ids": prompt_ids,
                     "source_spans": source_spans, "target_ids": target_ids, "target_spans": target_spans, "foil_ids": foil_ids})
    audit = {"rows": len(rows), "partitions": {part: sum(row["partition"] == part for row in rows)
                                                for part in ("discovery", "confirmation", "fresh_joint_lockbox")},
             "prompt_token_range": [min(len(row["prompt_ids"]) for row in rows), max(len(row["prompt_ids"]) for row in rows)],
             "target_token_range": [min(len(row["target_ids"]) for row in rows), max(len(row["target_ids"]) for row in rows)],
             "four_source_spans": all(len(row["source_spans"]) == 4 for row in rows),
             "four_target_spans": all(len(row["target_spans"]) == 4 for row in rows)}
    return rows, audit


def modules(model) -> list[Any]:
    embed = model.model.embed_tokens if hasattr(model.model, "embed_tokens") else model.get_input_embeddings()
    return [embed, *list(model.model.layers), model.model.norm]


def right_pad(sequences: list[list[int]], device: torch.device, pad: int):
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences): ids[i, :len(sequence)] = torch.tensor(sequence, device=device); mask[i, :len(sequence)] = 1
    return ids, mask, (mask.cumsum(1) - 1).clamp_min(0)


def token_logprob(logits: torch.Tensor, sequence: list[int], prompt_length: int) -> tuple[float, float, int]:
    target = torch.tensor(sequence[prompt_length:], device=logits.device); pred = logits[prompt_length - 1:len(sequence) - 1].float()
    selected = torch.log_softmax(pred, dim=-1).gather(1, target[:, None]).squeeze(1)
    return float(selected.mean()), float(selected.sum()), int(selected.numel())


def collect_model(key: str, model, rows: list[dict]) -> dict:
    p = paths(key); qmods = modules(model); dim = int(model.get_input_embeddings().weight.shape[1]); qcount = len(qmods)
    shape = (len(rows), 4, qcount, dim)
    if p["source"].exists() and p["output"].exists() and p["scores"].exists() and p["progress"].exists():
        source = np.lib.format.open_memmap(p["source"], mode="r+"); output = np.lib.format.open_memmap(p["output"], mode="r+")
        scores = np.lib.format.open_memmap(p["scores"], mode="r+"); completed = int(json.loads(p["progress"].read_text(encoding="utf-8"))["completed"])
    else:
        p["source"].parent.mkdir(parents=True, exist_ok=True)
        source = np.lib.format.open_memmap(p["source"], mode="w+", dtype=np.float16, shape=shape)
        output = np.lib.format.open_memmap(p["output"], mode="w+", dtype=np.float16, shape=shape)
        scores = np.lib.format.open_memmap(p["scores"], mode="w+", dtype=np.float32, shape=(len(rows), 7)); scores[:] = np.nan; completed = 0
    captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}; context: dict[str, Any] = {"source": [], "output": []}; handles = []
    for qpoint, module in enumerate(qmods):
        def hook(_module, _inputs, result, qpoint=qpoint):
            value = result[0] if isinstance(result, tuple) else result
            src = torch.stack([value[i, torch.tensor(pos, device=value.device)] for i, pos in enumerate(context["source"])])
            out = torch.stack([value[i, torch.tensor(pos, device=value.device)] for i, pos in enumerate(context["output"])])
            captures[qpoint] = (src.detach().float().cpu(), out.detach().float().cpu())
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    batch_size = int(loader.MODEL_SPECS[key]["batch"])
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]; sequences = [row["prompt_ids"] + row["target_ids"] for row in batch]
                context["source"] = [[span[1] - 1 for span in row["source_spans"]] for row in batch]
                context["output"] = [[len(row["prompt_ids"]) + span[0] - 1 for span in row["target_spans"]] for row in batch]
                ids, mask, positions = right_pad(sequences, device, pad); captures.clear()
                result = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for qpoint in range(qcount):
                    src, out = captures[qpoint]; source[start:start + len(batch), :, qpoint] = src.numpy().astype(np.float16)
                    output[start:start + len(batch), :, qpoint] = out.numpy().astype(np.float16)
                for local, (row, sequence) in enumerate(zip(batch, sequences)):
                    mean, total, count = token_logprob(result.logits[local], sequence, len(row["prompt_ids"]))
                    scores[start + local, 0] = mean; scores[start + local, 3] = total; scores[start + local, 5] = count
                source.flush(); output.flush(); scores.flush(); save(p["progress"], {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 96 == 0 or start + len(batch) == len(rows): print(f"[phase2382 {key} teacher] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        source.flush(); output.flush(); scores.flush(); close(source); close(output); close(scores)
    return {"shape": list(shape), "qpoints": qcount, "dimension": dim, "batch_size": batch_size,
            "quantization": loader.MODEL_SPECS[key]["quant"]}


def collect_foil(key: str, model, rows: list[dict]) -> dict:
    p = paths(key); scores = np.lib.format.open_memmap(p["scores"], mode="r+")
    completed = int(json.loads(p["foil_progress"].read_text(encoding="utf-8"))["completed"]) if p["foil_progress"].exists() else 0
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    batch_size = int(loader.MODEL_SPECS[key]["batch"])
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]; sequences = [row["prompt_ids"] + row["foil_ids"] for row in batch]
                ids, mask, positions = right_pad(sequences, device, pad)
                result = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for local, (row, sequence) in enumerate(zip(batch, sequences)):
                    mean, total, count = token_logprob(result.logits[local], sequence, len(row["prompt_ids"])); index = start + local
                    scores[index, 1] = mean; scores[index, 2] = float(scores[index, 0]) - mean; scores[index, 4] = total; scores[index, 6] = count
                scores.flush(); save(p["foil_progress"], {"completed": start + len(batch)})
                if (start + len(batch)) % 192 == 0 or start + len(batch) == len(rows): print(f"[phase2382 {key} foil] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        scores.flush(); close(scores)
    values = np.load(p["scores"], mmap_mode="r")
    result = {"target_over_foil": float(np.mean(values[:, 2] > 0)), "mean_margin": float(np.mean(values[:, 2])),
              "finite": bool(np.isfinite(values).all())}; close(values); return result


def fit_params(source: np.ndarray, output: np.ndarray, rows: list[dict], indices: np.ndarray, labels: np.ndarray,
               qpoint: int, coordinate_perm: np.ndarray | None = None, wrong: bool = False):
    params = {}
    for target_slot in range(4):
        for reverse in (False, True):
            use = [i for i in indices if bool(rows[int(i)]["reverse"]) == reverse]
            xs, ys = [], []
            for row_index in use:
                slot = int(labels[int(row_index), target_slot]); slot = (slot + 1) % 4 if wrong else slot
                x = np.asarray(source[int(row_index), slot, qpoint], dtype=np.float32)
                if coordinate_perm is not None: x = x[coordinate_perm]
                xs.append(x); ys.append(np.asarray(output[int(row_index), target_slot, qpoint], dtype=np.float32))
            params[(target_slot, reverse)] = adjudicate.fit_diagonal(np.asarray(xs), np.asarray(ys))
    return params


def match_accuracy(source: np.ndarray, output: np.ndarray, rows: list[dict], indices: np.ndarray, labels: np.ndarray,
                   qpoint: int, params: dict, coordinate_perm: np.ndarray | None = None) -> float:
    correct = total = 0
    for row_index in indices:
        row = rows[int(row_index)]; candidates = np.asarray(source[int(row_index), :, qpoint], dtype=np.float32)
        if coordinate_perm is not None: candidates = candidates[:, coordinate_perm]
        for target_slot in range(4):
            a, b = params[(target_slot, bool(row["reverse"]))]; predicted = candidates * a + b
            y = np.asarray(output[int(row_index), target_slot, qpoint], dtype=np.float32)
            choice = int(np.square(predicted - y).mean(1).argmin())
            correct += int(choice == labels[int(row_index), target_slot]); total += 1
    return correct / total


def analyze_model(key: str, rows: list[dict], material_audit: dict, collection: dict, behavior: dict) -> dict:
    p = paths(key); source = np.load(p["source"], mmap_mode="r"); output = np.load(p["output"], mmap_mode="r")
    labels = adjudicate.slot_labels(rows); splits = adjudicate.split_indices(rows)
    train, confirm, lock = splits["discovery"], splits["confirmation"], splits["fresh_joint_lockbox"]
    decoder_layers, match_layers = [], []
    for qpoint in range(output.shape[2]):
        tx = np.asarray(output[train, :, qpoint], dtype=np.float32).reshape(-1, output.shape[-1]); ty = labels[train].reshape(-1)
        decoder = {"qpoint": qpoint, "methods": {}}
        for scaled, method in ((False, "raw_centroid"), (True, "zscore_centroid")):
            item = {}
            for part, indices in (("confirmation", confirm), ("lockbox", lock)):
                vx = np.asarray(output[indices, :, qpoint], dtype=np.float32).reshape(-1, output.shape[-1]); vy = labels[indices].reshape(-1)
                item[part] = float(np.mean(adjudicate.nearest_centroid(tx, ty, vx, scaled) == vy))
            decoder["methods"][method] = item
        decoder_layers.append(decoder)
        params = fit_params(source, output, rows, train, labels, qpoint)
        match_layers.append({"qpoint": qpoint, "confirmation_accuracy": match_accuracy(source, output, rows, confirm, labels, qpoint, params),
                             "lockbox_accuracy": match_accuracy(source, output, rows, lock, labels, qpoint, params)})
    dec_candidates = [(entry["methods"][method]["confirmation"], entry, method) for entry in decoder_layers for method in entry["methods"]]
    _, dec_entry, dec_method = max(dec_candidates, key=lambda item: (item[0], -item[1]["qpoint"], item[2]))
    match_entry = max(match_layers, key=lambda entry: (entry["confirmation_accuracy"], -entry["qpoint"]))
    qpoint = match_entry["qpoint"]; rng = np.random.default_rng(2382); perm = rng.permutation(output.shape[-1])
    perm_params = fit_params(source, output, rows, train, labels, qpoint, coordinate_perm=perm)
    wrong_params = fit_params(source, output, rows, train, labels, qpoint, wrong=True)
    result = {"model": key, "model_label": loader.MODEL_SPECS[key]["label"], "material": material_audit,
              "collection": collection, "behavior": behavior,
              "slot_decoder": {"selected_qpoint": dec_entry["qpoint"], "method": dec_method,
                               "confirmation_accuracy": dec_entry["methods"][dec_method]["confirmation"],
                               "lockbox_accuracy": dec_entry["methods"][dec_method]["lockbox"], "layers": decoder_layers},
              "object_matching": {**match_entry,
                  "coordinate_permuted_lockbox_accuracy": match_accuracy(source, output, rows, lock, labels, qpoint, perm_params, perm),
                  "wrong_source_fit_lockbox_accuracy": match_accuracy(source, output, rows, lock, labels, qpoint, wrong_params),
                  "layers": match_layers},
              "behavior_qualified": behavior["target_over_foil"] >= 0.60}
    close(source); close(output); save(p["final"], result); return result


def run_model(key: str, source_rows: list[dict], source_indices: list[int]) -> dict:
    p = paths(key)
    if p["final"].exists(): return json.loads(p["final"].read_text(encoding="utf-8"))
    model, tokenizer, _ = loader.load_model(key)
    try:
        rows, audit = retokenize(tokenizer, source_rows, source_indices); write_rows(p["rows"], rows)
        collection = collect_model(key, model, rows); behavior = collect_foil(key, model, rows)
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return analyze_model(key, rows, audit, collection, behavior)


def aggregate(models: dict[str, dict]) -> dict:
    qwen4 = json.loads((P2381 / "analysis/final.json").read_text(encoding="utf-8"))
    comparison = {"qwen4b": {"behavior_target_over_foil": 0.841796875,
                               "pre_object_match_lockbox": qwen4["pre_sentence_controls"]["lockbox_accuracy"],
                               "coordinate_permuted": qwen4["pre_sentence_controls"]["coordinate_permuted_lockbox_accuracy"]}}
    for key, result in models.items():
        comparison[key] = {"behavior_target_over_foil": result["behavior"]["target_over_foil"],
                           "behavior_qualified": result["behavior_qualified"],
                           "slot_decoder_lockbox": result["slot_decoder"]["lockbox_accuracy"],
                           "pre_object_match_lockbox": result["object_matching"]["lockbox_accuracy"],
                           "coordinate_permuted": result["object_matching"]["coordinate_permuted_lockbox_accuracy"],
                           "wrong_source": result["object_matching"]["wrong_source_fit_lockbox_accuracy"]}
    qualified = [key for key, value in comparison.items() if key == "qwen4b" or value.get("behavior_qualified")]
    return {"comparison": comparison, "behavior_qualified_models": qualified,
            "universal_fixed_coordinate_gear": False,
            "reason": "Every positive object match must beat coordinate permutation to support fixed physical-coordinate correspondence; Qwen4B does not."}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 四模型无显式标签句前绑定顺序复验（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 一次只加载Qwen3-14B-NF4、GLM4-9B-INT8、DS-R1-Distill-Qwen-7B-INT8；按各自tokenizer重建同一768条分层panel，采集四个来源句末端和四个teacher-forced输出句前的embedding、全部block、final norm所有模型本地物理坐标。完整目标序列与循环错序foil构成行为门；confirmation选层，fresh unit+来源排列联合锁箱裁决槽位质心和条件逐坐标对象匹配，并测试坐标置乱和错误来源拟合。模型之间不比较坐标编号。

$$\mathcal H^m_{{src/out}}\in\mathbb R^{{N\times4\times Q_m\times d_m}},\qquad
\widehat s_m=\arg\min_s\|H^m_{{out}}-(a^m\odot H^m_{{src,s}}+b^m)\|^2.$$

**结果汇总。** 四模型比较 `{json.dumps(result['aggregate']['comparison'], ensure_ascii=False)}`；普遍性裁决 `{json.dumps(result['aggregate'], ensure_ascii=False)}`；执行检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2382_c16881_c17200_crossmodel_label_free_binding.py`；三模型各自材料、完整本地坐标场和逐层分析位于 `tests/glm5/result/phase2382_c16881_c17200_crossmodel_label_free_binding`。

**理论进展、问题硬伤与结论。** 跨模型只保留行为合格模型的描述性结果。量化精度、聊天训练和架构谱系仍是混淆，DS-R1-Distill-Qwen不是真正独立于Qwen的架构证据。若对象匹配不胜坐标置乱，就只能称分布式样本内相似性，不能称固定坐标编码。即使多模型通过，也仍没有证明自主指针；下一Phase自动续研将移除自然名字和直接时间/顺序词，检验当前信号是否依赖词汇锚点。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", choices=MODELS); args = parser.parse_args()
    source_rows = [row for row in read_rows(MATERIAL) if row["task"] == "exact_copy"]; source_indices = panel_source_indices()
    if args.model:
        result = run_model(args.model, source_rows, source_indices); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    models = {}
    for key in MODELS: models[key] = run_model(key, source_rows, source_indices)
    aggregate_result = aggregate(models)
    checks = {"all_models_complete": set(models) == set(MODELS),
              "all_rows": all(result["material"]["rows"] == 768 for result in models.values()),
              "all_coordinate_shapes": all(result["collection"]["shape"][0:2] == [768, 4] for result in models.values()),
              "all_finite": all(math.isfinite(result["object_matching"]["lockbox_accuracy"]) for result in models.values()),
              "sequential_order": list(models) == list(MODELS)}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "aggregate": aggregate_result,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
