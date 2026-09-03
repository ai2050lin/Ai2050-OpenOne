#!/usr/bin/env python3
"""Adjudicate label-free object/slot/output-progress structure and autonomous copy behavior."""
from __future__ import annotations

import gc
import json
import math
import re
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
P2378 = RESULT / "phase2378_c15601_c15920_label_free_binding_contract"
P2379 = RESULT / "phase2379_c15921_c16240_qwen_label_free_full_field"
OUT = RESULT / "phase2380_c16241_c16560_object_slot_progress_adjudication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2378 / "material/label_free_natural_binding.jsonl"
SOURCE_PATH = P2379 / "raw/qwen4b_source_sentence_end.float16.npy"
OUTPUT_PATH = P2379 / "raw/qwen4b_output_progress_anchors.float16.npy"
SCORES_PATH = P2379 / "raw/qwen4b_sequence_scores.float32.npy"
COORD_R2 = OUT / "derived/diagonal_object_match_coordinate_r2.float32.npy"
PHASE = 2380
CAMPAIGN = "C16241-C16560"
OFFSETS = ("pre_sentence", "early_token_2", "sentence_end")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


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


def split_indices(rows: list[dict]) -> dict[str, np.ndarray]:
    return {name: np.asarray([i for i, row in enumerate(rows) if row["partition"] == name], dtype=np.int64)
            for name in ("discovery", "confirmation", "fresh_joint_lockbox")}


def slot_labels(rows: list[dict]) -> np.ndarray:
    labels = np.empty((len(rows), 4), dtype=np.int64)
    for i, row in enumerate(rows):
        for target_slot, sentence_id in enumerate(row["target_order"]):
            labels[i, target_slot] = row["source_perm"].index(sentence_id)
    return labels


def nearest_centroid(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, scaled: bool) -> np.ndarray:
    if scaled:
        mean = train_x.mean(0); scale = train_x.std(0) + 1e-4
        train_x, test_x = (train_x - mean) / scale, (test_x - mean) / scale
    centroids = np.stack([train_x[train_y == label].mean(0) for label in range(4)])
    return np.square(test_x[:, None, :] - centroids[None, :, :]).mean(2).argmin(1)


def stratified_shuffle(rows: list[dict], indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(2380); shuffled = labels[indices].copy()
    groups: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for local, row_index in enumerate(indices):
        row = rows[int(row_index)]
        for target_slot in range(4):
            groups[(row["family"], row["language"], row["surface"], row["reverse"], target_slot)].append((local, target_slot))
    for members in groups.values():
        values = np.asarray([shuffled[i, t] for i, t in members]); rng.shuffle(values)
        for (i, t), value in zip(members, values): shuffled[i, t] = value
    return shuffled


def decode_slots(rows: list[dict], output: np.ndarray, splits: dict[str, np.ndarray], labels: np.ndarray) -> dict:
    train, confirm, lock = splits["discovery"], splits["confirmation"], splits["fresh_joint_lockbox"]
    shuffled = stratified_shuffle(rows, train, labels)
    layers = []
    for offset in range(output.shape[2]):
        for qpoint in range(output.shape[3]):
            tx = np.asarray(output[train, :, offset, qpoint], dtype=np.float32).reshape(-1, output.shape[-1])
            ty = labels[train].reshape(-1); sy = shuffled.reshape(-1)
            row = {"offset": offset, "offset_name": OFFSETS[offset], "qpoint": qpoint, "methods": {}}
            for scaled, name in ((False, "raw_centroid"), (True, "zscore_centroid")):
                method = {}
                for split_name, indices in (("confirmation", confirm), ("lockbox", lock)):
                    vx = np.asarray(output[indices, :, offset, qpoint], dtype=np.float32).reshape(-1, output.shape[-1])
                    vy = labels[indices].reshape(-1)
                    pred = nearest_centroid(tx, ty, vx, scaled); shuffled_pred = nearest_centroid(tx, sy, vx, scaled)
                    method[split_name] = {"n": int(vy.size), "accuracy": float(np.mean(pred == vy)),
                                          "stratified_shuffle_accuracy": float(np.mean(shuffled_pred == vy))}
                row["methods"][name] = method
            layers.append(row)
    candidates = [(entry["methods"][method]["confirmation"]["accuracy"], entry, method)
                  for entry in layers for method in entry["methods"]]
    _, selected, method = max(candidates, key=lambda item: (item[0], -item[1]["qpoint"], item[2]))
    return {"layers": layers, "selection_rule": "maximize confirmation accuracy; lockbox unopened",
            "selected": {"qpoint": selected["qpoint"], "offset": selected["offset"], "offset_name": selected["offset_name"],
                         "method": method, **selected["methods"][method]["lockbox"]}, "chance": 0.25}


def fit_diagonal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mx, my = x.mean(0), y.mean(0); xc = x - mx; yc = y - my
    a = np.sum(xc * yc, axis=0) / (np.sum(xc * xc, axis=0) + 1e-6)
    return a.astype(np.float32), (my - a * mx).astype(np.float32)


def pair_data(source: np.ndarray, output: np.ndarray, row_indices: np.ndarray, labels: np.ndarray,
              offset: int, qpoint: int) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(source[row_indices, :, qpoint], dtype=np.float32)
    out = np.asarray(output[row_indices, :, offset, qpoint], dtype=np.float32)
    lab = labels[row_indices]
    matched = np.stack([src[i, lab[i]] for i in range(len(row_indices))])
    return matched.reshape(-1, src.shape[-1]), out.reshape(-1, src.shape[-1])


def match_accuracy(source: np.ndarray, output: np.ndarray, indices: np.ndarray, labels: np.ndarray,
                   offset: int, qpoint: int, params: dict[tuple, tuple[np.ndarray, np.ndarray]],
                   rows: list[dict], coordinate_perm: np.ndarray | None = None) -> float:
    correct, total = 0, 0
    src = np.asarray(source[indices, :, qpoint], dtype=np.float32)
    out = np.asarray(output[indices, :, offset, qpoint], dtype=np.float32)
    for local, row_index in enumerate(indices):
        row = rows[int(row_index)]
        for target_slot in range(4):
            key = next(iter(params)) if len(params) == 1 else (target_slot, bool(row["reverse"]))
            a, b = params[key]; candidates = src[local]
            if coordinate_perm is not None: candidates = candidates[:, coordinate_perm]
            predicted = candidates * a + b
            distance = np.square(predicted - out[local, target_slot]).mean(1)
            correct += int(int(distance.argmin()) == labels[int(row_index), target_slot]); total += 1
    return correct / total


def fit_params(source: np.ndarray, output: np.ndarray, rows: list[dict], indices: np.ndarray, labels: np.ndarray,
               offset: int, qpoint: int, conditional: bool, wrong_pair: bool = False,
               coordinate_perm: np.ndarray | None = None) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    result = {}; keys = [(t, r) for t in range(4) for r in (False, True)] if conditional else [(0, False)]
    for key in keys:
        use = indices if not conditional else np.asarray([i for i in indices if bool(rows[int(i)]["reverse"]) == key[1]])
        src = np.asarray(source[use, :, qpoint], dtype=np.float32); out = np.asarray(output[use, :, offset, qpoint], dtype=np.float32)
        target_slots = range(4) if not conditional else (key[0],)
        xs, ys = [], []
        for local, row_index in enumerate(use):
            for target_slot in target_slots:
                slot = int(labels[int(row_index), target_slot]); slot = (slot + 1) % 4 if wrong_pair else slot
                candidate = src[local, slot]
                if coordinate_perm is not None: candidate = candidate[coordinate_perm]
                xs.append(candidate); ys.append(out[local, target_slot])
        result[key] = fit_diagonal(np.asarray(xs), np.asarray(ys))
    return result


def object_matching(rows: list[dict], source: np.ndarray, output: np.ndarray,
                    splits: dict[str, np.ndarray], labels: np.ndarray) -> dict:
    train, confirm, lock = splits["discovery"], splits["confirmation"], splits["fresh_joint_lockbox"]
    coordinate_r2 = np.zeros((3, output.shape[3], 2, output.shape[-1]), dtype=np.float32)
    layers = []
    for offset in range(3):
        for qpoint in range(output.shape[3]):
            entry = {"offset": offset, "offset_name": OFFSETS[offset], "qpoint": qpoint, "methods": {}}
            for conditional, name in ((False, "global_diagonal"), (True, "output_direction_conditioned_diagonal")):
                params = fit_params(source, output, rows, train, labels, offset, qpoint, conditional)
                entry["methods"][name] = {
                    "confirmation_accuracy": match_accuracy(source, output, confirm, labels, offset, qpoint, params, rows),
                    "lockbox_accuracy": match_accuracy(source, output, lock, labels, offset, qpoint, params, rows),
                }
                x_lock, y_lock = pair_data(source, output, lock, labels, offset, qpoint)
                if conditional:
                    predictions = np.empty_like(y_lock); cursor = 0
                    src_lock = np.asarray(source[lock, :, qpoint], dtype=np.float32)
                    for local, row_index in enumerate(lock):
                        for target_slot in range(4):
                            a, b = params[(target_slot, bool(rows[int(row_index)]["reverse"]))]
                            predictions[cursor] = src_lock[local, labels[int(row_index), target_slot]] * a + b; cursor += 1
                else:
                    a, b = params[(0, False)]; predictions = x_lock * a + b
                sse = np.square(y_lock - predictions).sum(0); sst = np.square(y_lock - y_lock.mean(0)).sum(0)
                coordinate_r2[offset, qpoint, int(conditional)] = 1.0 - sse / np.maximum(sst, 1e-8)
                entry["methods"][name]["lockbox_global_r2"] = float(1.0 - np.square(y_lock - predictions).sum() /
                                                                        max(float(np.square(y_lock - y_lock.mean(0)).sum()), 1e-8))
            layers.append(entry)
            if qpoint % 8 == 0: print(f"[phase2380 matching] offset={offset} q={qpoint}", flush=True)
    COORD_R2.parent.mkdir(parents=True, exist_ok=True); np.save(COORD_R2, coordinate_r2)
    candidates = [(entry["methods"][method]["confirmation_accuracy"], entry, method)
                  for entry in layers for method in entry["methods"]]
    _, selected, method = max(candidates, key=lambda item: (item[0], -item[1]["qpoint"], item[2]))
    offset, qpoint = selected["offset"], selected["qpoint"]
    conditional = method.startswith("output_direction")
    rng = np.random.default_rng(2380); perm = rng.permutation(output.shape[-1])
    perm_params = fit_params(source, output, rows, train, labels, offset, qpoint, conditional, coordinate_perm=perm)
    wrong_params = fit_params(source, output, rows, train, labels, offset, qpoint, conditional, wrong_pair=True)
    controls = {
        "coordinate_permuted_lockbox_accuracy": match_accuracy(source, output, lock, labels, offset, qpoint, perm_params, rows, perm),
        "wrong_source_fit_lockbox_accuracy": match_accuracy(source, output, lock, labels, offset, qpoint, wrong_params, rows),
    }
    return {"layers": layers, "selection_rule": "maximize confirmation object-match accuracy; fresh joint lockbox unopened",
            "selected": {"qpoint": qpoint, "offset": offset, "offset_name": OFFSETS[offset], "method": method,
                         **selected["methods"][method], **controls}, "chance": 0.25,
            "coordinate_r2_path": str(COORD_R2), "coordinate_r2_shape": list(coordinate_r2.shape)}


def grouped_sequence_behavior(rows: list[dict], scores: np.ndarray) -> dict:
    result = {"overall": {"n": len(rows), "target_over_foil": float(np.mean(scores[:, 2] > 0)),
                          "mean_margin": float(np.mean(scores[:, 2]))}}
    for dimension in ("family", "language", "surface", "partition", "reverse"):
        groups = {}
        for value in sorted({str(row[dimension]) for row in rows}):
            indices = [i for i, row in enumerate(rows) if str(row[dimension]) == value]
            groups[value] = {"n": len(indices), "target_over_foil": float(np.mean(scores[indices, 2] > 0)),
                             "mean_margin": float(np.mean(scores[indices, 2]))}
        result[f"by_{dimension}"] = groups
    return result


def clean_generation(text: str) -> str:
    text = re.sub(r"(?s)^.*?</think>\s*", "", text).strip()
    return text


def score_generation(row: dict, generated: str, generated_tokens: int, maximum: int) -> dict:
    text = clean_generation(generated); positions = [text.find(sentence) for sentence in row["sentences"]]
    target_positions = [positions[sid] for sid in row["target_order"]]
    found = [position >= 0 for position in positions]
    order_exact = all(position >= 0 for position in target_positions) and target_positions == sorted(target_positions)
    normalized = lambda value: re.sub(r"\s+", " ", value).strip()
    return {"generated": text, "generated_tokens": generated_tokens, "hit_max_new_tokens": generated_tokens >= maximum,
            "sentence_recall": float(np.mean(found)), "all_sentences_present": all(found),
            "identity_order_exact": order_exact, "verbatim_full_exact": normalized(text) == normalized(row["target"])}


def autonomous_generation(rows: list[dict], batch_size: int = 8) -> dict:
    output_path = OUT / "generation/qwen4b_lockbox_generations.jsonl"
    selected = [row for row in rows if row["partition"] == "fresh_joint_lockbox"]
    if output_path.exists():
        existing = read_rows(output_path)
        if len(existing) == len(selected): return summarize_generation(existing)
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    generated_rows = []
    try:
        with torch.inference_mode():
            for start in range(0, len(selected), batch_size):
                batch = selected[start:start + batch_size]; maximum = max(len(row["target_ids"]) for row in batch) + 48
                width = max(len(row["prompt_ids"]) for row in batch)
                ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
                for local, row in enumerate(batch):
                    sequence = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
                    ids[local, -len(sequence):] = sequence; mask[local, -len(sequence):] = 1
                output = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=maximum,
                                        pad_token_id=pad, eos_token_id=model.config.eos_token_id, use_cache=True)
                for local, row in enumerate(batch):
                    new_ids = output[local, width:].tolist(); text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    generated_rows.append({"case_id": row["case_id"], "family": row["family"], "unit": row["unit"],
                                           "language": row["language"], "surface": row["surface"], "reverse": row["reverse"],
                                           "source_index": row["source_index"], **score_generation(row, text, len(new_ids), maximum)})
                write_rows(output_path, generated_rows)
                if (start + len(batch)) % 64 == 0 or start + len(batch) == len(selected):
                    print(f"[phase2380 generation] {start + len(batch)}/{len(selected)}", flush=True)
    finally:
        model_utils.release_model(model); del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summarize_generation(generated_rows)


def summarize_generation(rows: list[dict]) -> dict:
    metrics = ("sentence_recall", "all_sentences_present", "identity_order_exact", "verbatim_full_exact", "hit_max_new_tokens")
    result = {"rows": len(rows), **{metric: float(np.mean([row[metric] for row in rows])) for metric in metrics}}
    for dimension in ("family", "language", "surface", "reverse"):
        result[f"by_{dimension}"] = {value: {metric: float(np.mean([row[metric] for row in rows if str(row[dimension]) == value]))
                                                      for metric in metrics[:-1]}
                                          for value in sorted({str(row[dimension]) for row in rows})}
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 无标签句对象－来源槽位－输出进度锁箱裁决与自主生成（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将每个teacher-forced输出句在输入中的真实来源位置定义为$z_{{r,t}}=\sigma_r^{{-1}}(\pi^*_{{r,t}})$。只用discovery训练，confirmation选层/时点/方法，fresh unit+未见source permutation联合锁箱一次裁决。比较全坐标质心槽位解码、来源句末端到输出锚点的逐坐标对角仿射匹配，并设置分层标签置乱、错误源句拟合、固定坐标置乱；另对全部256条联合锁箱以足够长度自主生成，逐句核对身份、目标顺序和逐字保持。

$$\hat z=\arg\min_c\|H^{{out}}-\mu_c\|^2,\qquad
\widehat H^{{out}}_j=a_jH^{{src}}_j+b_j,\qquad
\widehat s=\arg\min_{{s=1,\ldots,4}}\|H^{{out}}-(a\odot H^{{src}}_s+b)\|^2.$$

**结果汇总。** 完整序列分层行为 `{json.dumps(result['sequence_behavior'], ensure_ascii=False)}`；槽位解码选定结果 `{json.dumps(result['slot_decoder']['selected'], ensure_ascii=False)}`；句对象匹配选定结果 `{json.dumps(result['object_matching']['selected'], ensure_ascii=False)}`；自主生成 `{json.dumps(result['generation'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2380_c16241_c16560_object_slot_progress_adjudication.py`；逐层全方法、逐坐标$R^2$、生成明细位于 `tests/glm5/result/phase2380_c16241_c16560_object_slot_progress_adjudication`。

**理论进展、问题硬伤与结论。** 只有锁箱显著高于0.25且三类负控不能解释的信号，才称为“无显式标签下可读/可匹配的来源绑定”，仍不称自主指针或对象封装。pre-sentence若通过，证据强于句末，因为它尚未看见目标句词汇；early/end的升高则主要说明输出内容逐步进入状态。自主生成是内容闭合的独立行为门；teacher-forced正结果不能救援生成失败。下一Phase只沿本Phase实际通过的时点和层追踪Attention/MLP贡献。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = [row for row in read_rows(MATERIAL) if row["task"] == "exact_copy"]
    source = np.load(SOURCE_PATH, mmap_mode="r"); output = np.load(OUTPUT_PATH, mmap_mode="r"); scores = np.load(SCORES_PATH, mmap_mode="r")
    splits = split_indices(rows); labels = slot_labels(rows)
    sequence_behavior = grouped_sequence_behavior(rows, scores)
    slot_decoder = decode_slots(rows, output, splits, labels)
    object_result = object_matching(rows, source, output, splits, labels)
    close(source); close(output); close(scores)
    generation = autonomous_generation(rows)
    checks = {
        "split_sizes": {name: len(value) for name, value in splits.items()} == {"discovery": 768, "confirmation": 128, "fresh_joint_lockbox": 256},
        "decoder_complete": len(slot_decoder["layers"]) == 3 * 38,
        "matching_complete": len(object_result["layers"]) == 3 * 38,
        "coordinate_r2_complete": object_result["coordinate_r2_shape"] == [3, 38, 2, 2560],
        "generation_complete": generation["rows"] == 256,
        "finite": all(math.isfinite(x) for x in (slot_decoder["selected"]["accuracy"],
                                                  object_result["selected"]["lockbox_accuracy"], generation["sentence_recall"])),
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "sequence_behavior": sequence_behavior,
              "slot_decoder": slot_decoder, "object_matching": object_result, "generation": generation,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
