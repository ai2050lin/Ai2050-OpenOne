#!/usr/bin/env python3
"""Position-isolated Qwen3-4B sentence-content fields versus embedding baselines."""
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
RESULT = TESTS / "result"
P2378 = RESULT / "phase2378_c15601_c15920_label_free_binding_contract"
P2379 = RESULT / "phase2379_c15921_c16240_qwen_label_free_full_field"
P2381 = RESULT / "phase2381_c16561_c16880_residual_component_routing"
OUT = RESULT / "phase2384_c17521_c17840_isolated_sentence_content_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2378 / "material/label_free_natural_binding.jsonl"
END_FIELD = OUT / "raw/qwen4b_isolated_sentence_end.float16.npy"
MEAN_FIELD = OUT / "raw/qwen4b_isolated_sentence_token_mean.float16.npy"
OUTPUT_PRE = P2379 / "raw/qwen4b_output_progress_anchors.float16.npy"
PHASE = 2384
CAMPAIGN = "C17521-C17840"
FROZEN_OUTPUT_QPOINT = 11

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


def panel_indices() -> list[int]:
    return [int(row["source_index"]) for row in read_rows(P2381 / "index/component_panel_rows.jsonl")]


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def right_pad(sequences: list[list[int]], device: torch.device, pad: int):
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences): ids[i, :len(sequence)] = torch.tensor(sequence, device=device); mask[i, :len(sequence)] = 1
    return ids, mask, (mask.cumsum(1) - 1).clamp_min(0)


def compile_sentence_rows(tokenizer, rows: list[dict], panel: list[int]) -> list[dict]:
    result = []
    for panel_index, source_index in enumerate(panel):
        row = rows[source_index]
        for sentence_id, sentence in enumerate(row["sentences"]):
            ids = [int(x) for x in tokenizer.encode(sentence, add_special_tokens=False)]
            result.append({"flat_index": len(result), "panel_index": panel_index, "source_design_index": source_index,
                           "case_id": row["case_id"], "sentence_id": sentence_id, "sentence": sentence,
                           "token_ids": ids, "token_count": len(ids), "family": row["family"], "unit": row["unit"],
                           "language": row["language"], "surface": row["surface"], "partition": row["partition"]})
    return result


def collect(model, sentence_rows: list[dict], panel_count: int, batch_size: int = 16) -> dict:
    qmods = modules(model); dim = int(model.config.hidden_size); shape = (panel_count, 4, len(qmods), dim)
    progress_path = OUT / "raw/progress.json"
    if END_FIELD.exists() and MEAN_FIELD.exists() and progress_path.exists():
        ends = np.lib.format.open_memmap(END_FIELD, mode="r+"); means = np.lib.format.open_memmap(MEAN_FIELD, mode="r+")
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
    else:
        END_FIELD.parent.mkdir(parents=True, exist_ok=True)
        ends = np.lib.format.open_memmap(END_FIELD, mode="w+", dtype=np.float16, shape=shape)
        means = np.lib.format.open_memmap(MEAN_FIELD, mode="w+", dtype=np.float16, shape=shape); completed = 0
    captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}; context: dict[str, Any] = {"lengths": []}; handles = []
    for qpoint, module in enumerate(qmods):
        def hook(_module, _inputs, result, qpoint=qpoint):
            value = result[0] if isinstance(result, tuple) else result
            end = torch.stack([value[i, length - 1] for i, length in enumerate(context["lengths"])])
            mean = torch.stack([value[i, :length].mean(0) for i, length in enumerate(context["lengths"])])
            captures[qpoint] = (end.detach().float().cpu(), mean.detach().float().cpu())
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(sentence_rows), batch_size):
                batch = sentence_rows[start:start + batch_size]; sequences = [row["token_ids"] for row in batch]
                context["lengths"] = list(map(len, sequences)); ids, mask, positions = right_pad(sequences, device, pad); captures.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for local, row in enumerate(batch):
                    for qpoint in range(len(qmods)):
                        end, mean = captures[qpoint]
                        ends[row["panel_index"], row["sentence_id"], qpoint] = end[local].numpy().astype(np.float16)
                        means[row["panel_index"], row["sentence_id"], qpoint] = mean[local].numpy().astype(np.float16)
                ends.flush(); means.flush(); save(progress_path, {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 256 == 0 or start + len(batch) == len(sentence_rows):
                    print(f"[phase2384 isolated] {start + len(batch)}/{len(sentence_rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        ends.flush(); means.flush(); close(ends); close(means)
    write_rows(OUT / "index/isolated_sentence_rows.jsonl", sentence_rows)
    return {"shape": list(shape), "flat_sentence_rows": len(sentence_rows),
            "token_range": [min(row["token_count"] for row in sentence_rows), max(row["token_count"] for row in sentence_rows)],
            "fields": ["last_sentence_token", "all_sentence_token_mean"], "no_source_slot_context": True}


def fit_params(source: np.ndarray, output: np.ndarray, rows: list[dict], train: np.ndarray, labels: np.ndarray):
    params = {}
    for target_slot in range(4):
        for reverse in (False, True):
            use = [i for i in train if bool(rows[int(i)]["reverse"]) == reverse]
            x = np.stack([source[int(i), rows[int(i)]["target_order"][target_slot]] for i in use])
            y = np.stack([output[int(i), target_slot] for i in use])
            params[(target_slot, reverse)] = adjudicate.fit_diagonal(x, y)
    return params


def accuracy(source: np.ndarray, output: np.ndarray, rows: list[dict], indices: np.ndarray, params: dict) -> float:
    correct = total = 0
    for i in indices:
        row = rows[int(i)]
        for target_slot in range(4):
            a, b = params[(target_slot, bool(row["reverse"]))]
            distances = np.square(source[int(i)] * a + b - output[int(i), target_slot]).mean(1)
            predicted_sentence_id = int(distances.argmin()); truth = int(row["target_order"][target_slot])
            correct += int(predicted_sentence_id == truth); total += 1
    return correct / total


def donor_control(source: np.ndarray, rows: list[dict], lock: np.ndarray) -> np.ndarray:
    result = source.copy(); groups: dict[tuple, list[int]] = {}
    for i in lock:
        row = rows[int(i)]; groups.setdefault((row["language"], row["surface"], row["source_index"]), []).append(int(i))
    for members in groups.values():
        ordered = sorted(members, key=lambda i: (rows[i]["family"], rows[i]["unit"], rows[i]["reverse"]))
        donors = ordered[7:] + ordered[:7]
        for target, donor in zip(ordered, donors): result[target] = source[donor]
    return result


def analyze(rows: list[dict], panel: list[int]) -> dict:
    panel_rows = [rows[i] for i in panel]; splits = adjudicate.split_indices(panel_rows)
    train, confirm, lock = splits["discovery"], splits["confirmation"], splits["fresh_joint_lockbox"]
    output_map = np.load(OUTPUT_PRE, mmap_mode="r")
    output = np.asarray(output_map[panel, :, 0, FROZEN_OUTPUT_QPOINT], dtype=np.float32); close(output_map)
    fields = {"isolated_end": np.load(END_FIELD, mmap_mode="r"), "isolated_token_mean": np.load(MEAN_FIELD, mmap_mode="r")}
    layers = []
    for field_name, field in fields.items():
        for qpoint in range(field.shape[2]):
            source = np.asarray(field[:, :, qpoint], dtype=np.float32); params = fit_params(source, output, panel_rows, train, None)
            layers.append({"field": field_name, "source_qpoint": qpoint,
                           "confirmation_accuracy": accuracy(source, output, panel_rows, confirm, params),
                           "lockbox_accuracy": accuracy(source, output, panel_rows, lock, params)})
    selected = max(layers, key=lambda item: (item["confirmation_accuracy"], -item["source_qpoint"], item["field"]))
    selected_field = fields[selected["field"]]; source = np.asarray(selected_field[:, :, selected["source_qpoint"]], dtype=np.float32)
    params = fit_params(source, output, panel_rows, train, None)
    donor = donor_control(source, panel_rows, lock); rng = np.random.default_rng(2384); shuffled = np.empty_like(source)
    for row in range(len(source)):
        for sid in range(4): shuffled[row, sid] = source[row, sid, rng.permutation(source.shape[-1])]
    shuffle_params = fit_params(shuffled, output, panel_rows, train, None)
    controls = {"cross_row_content_donor_lockbox": accuracy(donor, output, panel_rows, lock, params),
                "row_specific_coordinate_permutation_lockbox": accuracy(shuffled, output, panel_rows, lock, shuffle_params),
                "embedding_mean_q0_confirmation": next(x["confirmation_accuracy"] for x in layers if x["field"] == "isolated_token_mean" and x["source_qpoint"] == 0),
                "embedding_mean_q0_lockbox": next(x["lockbox_accuracy"] for x in layers if x["field"] == "isolated_token_mean" and x["source_qpoint"] == 0)}
    for field in fields.values(): close(field)
    return {"output_qpoint_frozen": FROZEN_OUTPUT_QPOINT, "layers": layers, "selected": {**selected, **controls},
            "chance": 0.25, "selection_rule": "source layer/summary selected on confirmation; joint lockbox unopened",
            "interpretation_gate": {"position_independent_content": selected["lockbox_accuracy"] - controls["cross_row_content_donor_lockbox"] >= 0.10,
                                    "beyond_embedding_mean": selected["lockbox_accuracy"] - controls["embedding_mean_q0_lockbox"] >= 0.05,
                                    "stable_coordinate_texture": selected["lockbox_accuracy"] - controls["row_specific_coordinate_permutation_lockbox"] >= 0.10}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-4B来源位置归一的独立句内容全坐标场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 把768条分层panel中的四个自然句各自从长prompt抽离，作为独立短序列编码；每句采集embedding、36个block、final norm的最后token全部2560坐标和全句token均值全部2560坐标。它们没有来源槽位上下文，再映射到Phase2380冻结的$q=11$输出句前状态。纯embedding均值$q=0$是词汇基线；confirmation选择来源层/汇总方式，fresh unit+来源排列联合锁箱裁决，并比较跨内容donor和每样本坐标置乱。

$$C^{{end}}_{{r,s,q}}=H_q(S_{{r,s}})[-1],\qquad
C^{{mean}}_{{r,s,q}}={{1\over |S_{{r,s}}|}}\sum_pH_q(S_{{r,s}})[p].$$

**结果汇总。** 采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；选定及基线 `{json.dumps(result['analysis']['selected'], ensure_ascii=False)}`；解释门 `{json.dumps(result['analysis']['interpretation_gate'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2384_c17521_c17840_isolated_sentence_content_field.py`；独立句两套全层全坐标场、索引、逐层锁箱位于 `tests/glm5/result/phase2384_c17521_c17840_isolated_sentence_content_field`。

**理论进展、问题硬伤与结论。** 独立句正结果可排除来源槽位和前文累积，但仍可能由词汇重叠完成；只有明显超过embedding均值且跨内容donor下降，才支持层内组合内容纹理。token均值是本Phase的对照汇总，不替代此前all-token原场。即使通过，逐坐标仿射仍只是可预测映射，不是对象搬运因果机制；下一Phase只在正结果成立时做跨模型相同功能复验。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = [row for row in read_rows(MATERIAL) if row["task"] == "exact_copy"]; panel = panel_indices()
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        sentence_rows = compile_sentence_rows(tokenizer, rows, panel); collection = collect(model, sentence_rows, len(panel))
    finally:
        model_utils.release_model(model); del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    analysis = analyze(rows, panel)
    checks = {"end_shape": collection["shape"] == [768, 4, 38, 2560], "mean_shape": collection["shape"] == [768, 4, 38, 2560],
              "all_sentences": collection["flat_sentence_rows"] == 3072, "analysis_complete": len(analysis["layers"]) == 76,
              "finite": all(math.isfinite(analysis["selected"][key]) for key in ("confirmation_accuracy", "lockbox_accuracy",
                                                                                   "embedding_mean_q0_lockbox"))}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B-BF16", "collection": collection,
              "analysis": analysis, "checks": checks, "all_checks_passed": all(checks.values()),
              "next_stage": {"same_target": bool(analysis["interpretation_gate"]["position_independent_content"]),
                             "action": "cross-model isolated-content replication if positive"}}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
