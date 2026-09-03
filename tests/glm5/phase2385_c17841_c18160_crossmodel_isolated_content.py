#!/usr/bin/env python3
"""Sequential cross-model replication of position-isolated sentence-content fields."""
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
P2378 = RESULT / "phase2378_c15601_c15920_label_free_binding_contract"
P2381 = RESULT / "phase2381_c16561_c16880_residual_component_routing"
P2382 = RESULT / "phase2382_c16881_c17200_crossmodel_label_free_binding"
P2384 = RESULT / "phase2384_c17521_c17840_isolated_sentence_content_field"
OUT = RESULT / "phase2385_c17841_c18160_crossmodel_isolated_content"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2378 / "material/label_free_natural_binding.jsonl"
PHASE = 2385
CAMPAIGN = "C17841-C18160"
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
    return {"base": base, "end": base / "raw/isolated_end.float16.npy", "mean": base / "raw/isolated_mean.float16.npy",
            "progress": base / "raw/progress.json", "rows": base / "material/sentence_rows.jsonl", "final": base / "analysis/final.json"}


def panel_indices() -> list[int]:
    return [int(row["source_index"]) for row in read_rows(P2381 / "index/component_panel_rows.jsonl")]


def modules(model) -> list[Any]:
    embed = model.model.embed_tokens if hasattr(model.model, "embed_tokens") else model.get_input_embeddings()
    return [embed, *list(model.model.layers), model.model.norm]


def right_pad(sequences: list[list[int]], device: torch.device, pad: int):
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences): ids[i, :len(sequence)] = torch.tensor(sequence, device=device); mask[i, :len(sequence)] = 1
    return ids, mask, (mask.cumsum(1) - 1).clamp_min(0)


def compile_rows(tokenizer, source_rows: list[dict], panel: list[int]) -> list[dict]:
    result = []
    for panel_index, source_index in enumerate(panel):
        row = source_rows[source_index]
        for sentence_id, sentence in enumerate(row["sentences"]):
            ids = [int(x) for x in tokenizer.encode(sentence, add_special_tokens=False)]
            result.append({"flat_index": len(result), "panel_index": panel_index, "source_index": source_index,
                           "case_id": row["case_id"], "sentence_id": sentence_id, "token_ids": ids,
                           "token_count": len(ids), "partition": row["partition"], "family": row["family"], "language": row["language"]})
    return result


def collect(key: str, model, sentence_rows: list[dict], batch_size: int) -> dict:
    p = paths(key); qmods = modules(model); dim = int(model.get_input_embeddings().weight.shape[1]); shape = (768, 4, len(qmods), dim)
    if p["end"].exists() and p["mean"].exists() and p["progress"].exists():
        ends = np.lib.format.open_memmap(p["end"], mode="r+"); means = np.lib.format.open_memmap(p["mean"], mode="r+")
        completed = int(json.loads(p["progress"].read_text(encoding="utf-8"))["completed"])
    else:
        p["end"].parent.mkdir(parents=True, exist_ok=True)
        ends = np.lib.format.open_memmap(p["end"], mode="w+", dtype=np.float16, shape=shape)
        means = np.lib.format.open_memmap(p["mean"], mode="w+", dtype=np.float16, shape=shape); completed = 0
    captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}; context = {"lengths": []}; handles = []
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
                        end, mean = captures[qpoint]; ends[row["panel_index"], row["sentence_id"], qpoint] = end[local].numpy().astype(np.float16)
                        means[row["panel_index"], row["sentence_id"], qpoint] = mean[local].numpy().astype(np.float16)
                ends.flush(); means.flush(); save(p["progress"], {"completed": start + len(batch), "shape": shape})
                if (start + len(batch)) % 384 == 0 or start + len(batch) == len(sentence_rows): print(f"[phase2385 {key}] {start + len(batch)}/{len(sentence_rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        ends.flush(); means.flush(); close(ends); close(means)
    write_rows(p["rows"], sentence_rows)
    return {"shape": list(shape), "batch_size": batch_size, "token_range": [min(x["token_count"] for x in sentence_rows), max(x["token_count"] for x in sentence_rows)]}


def fit_params(source: np.ndarray, output: np.ndarray, rows: list[dict], train: np.ndarray):
    params = {}
    for target_slot in range(4):
        for reverse in (False, True):
            use = [i for i in train if bool(rows[int(i)]["reverse"]) == reverse]
            x = np.stack([source[int(i), rows[int(i)]["target_order"][target_slot]] for i in use]); y = output[use, target_slot]
            params[(target_slot, reverse)] = adjudicate.fit_diagonal(x, y)
    return params


def accuracy(source: np.ndarray, output: np.ndarray, rows: list[dict], indices: np.ndarray, params: dict) -> float:
    correct = total = 0
    for i in indices:
        row = rows[int(i)]
        for target_slot in range(4):
            a, b = params[(target_slot, bool(row["reverse"]))]
            choice = int(np.square(source[int(i)] * a + b - output[int(i), target_slot]).mean(1).argmin())
            correct += int(choice == row["target_order"][target_slot]); total += 1
    return correct / total


def analyze(key: str, rows: list[dict], collection: dict) -> dict:
    p = paths(key); cross = json.loads((P2382 / key / "analysis/final.json").read_text(encoding="utf-8"))
    output_q = int(cross["object_matching"]["qpoint"]); output_map = np.load(P2382 / key / "raw/output_pre_sentence.float16.npy", mmap_mode="r")
    output = np.asarray(output_map[:, :, output_q], dtype=np.float32); close(output_map)
    fields = {"isolated_end": np.load(p["end"], mmap_mode="r"), "isolated_mean": np.load(p["mean"], mmap_mode="r")}
    splits = adjudicate.split_indices(rows); train, confirm, lock = splits["discovery"], splits["confirmation"], splits["fresh_joint_lockbox"]
    layers = []
    for name, field in fields.items():
        for qpoint in range(field.shape[2]):
            source = np.asarray(field[:, :, qpoint], dtype=np.float32); params = fit_params(source, output, rows, train)
            layers.append({"field": name, "qpoint": qpoint, "confirmation_accuracy": accuracy(source, output, rows, confirm, params),
                           "lockbox_accuracy": accuracy(source, output, rows, lock, params)})
    selected = max(layers, key=lambda x: (x["confirmation_accuracy"], -x["qpoint"], x["field"]))
    source = np.asarray(fields[selected["field"]][:, :, selected["qpoint"]], dtype=np.float32); params = fit_params(source, output, rows, train)
    donor = source.copy(); groups: dict[tuple, list[int]] = {}
    for i in lock:
        row = rows[int(i)]; groups.setdefault((row["language"], row["surface"], row["source_index"]), []).append(int(i))
    for members in groups.values():
        ordered = sorted(members); donors = ordered[7:] + ordered[:7]
        for target, source_i in zip(ordered, donors): donor[target] = source[source_i]
    rng = np.random.default_rng(2385); shuffled = np.empty_like(source)
    for i in range(len(source)):
        for sid in range(4): shuffled[i, sid] = source[i, sid, rng.permutation(source.shape[-1])]
    shuffle_params = fit_params(shuffled, output, rows, train)
    baseline = next(x for x in layers if x["field"] == "isolated_mean" and x["qpoint"] == 0)
    selected.update({"donor_lockbox": accuracy(donor, output, rows, lock, params),
                     "row_specific_coordinate_permutation_lockbox": accuracy(shuffled, output, rows, lock, shuffle_params),
                     "embedding_mean_lockbox": baseline["lockbox_accuracy"]})
    for field in fields.values(): close(field)
    return {"model": key, "model_label": loader.MODEL_SPECS[key]["label"], "collection": collection,
            "behavior": cross["behavior"], "output_qpoint": output_q, "layers": layers, "selected": selected,
            "qualified": cross["behavior_qualified"] and selected["lockbox_accuracy"] - selected["donor_lockbox"] >= 0.10}


def run_model(key: str, source_rows: list[dict], panel: list[int]) -> dict:
    p = paths(key)
    if p["final"].exists(): return json.loads(p["final"].read_text(encoding="utf-8"))
    model, tokenizer, _ = loader.load_model(key)
    try:
        sentence_rows = compile_rows(tokenizer, source_rows, panel); batch_size = 8 if key == "qwen14b" else 16
        collection = collect(key, model, sentence_rows, batch_size)
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    panel_rows = [source_rows[i] for i in panel]; result = analyze(key, panel_rows, collection); save(p["final"], result); return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 四模型独立句内容场的顺序复验（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 按显存安全顺序一次加载Qwen14B、GLM4、DS7B；各模型用自己的tokenizer独立编码同一768×4个自然句，采集embedding、全部block和final norm的句末及全token均值全部本地坐标。输出端使用Phase2382各模型confirmation冻结的句前层；来源层/汇总在confirmation选择，fresh unit+来源排列联合锁箱裁决。比较纯embedding均值、跨内容donor和每样本坐标置乱。

$$\widehat s_m=\arg\min_s\|H^m_{{out}}-(a^m\odot C^m_{{isolated,s}}+b^m)\|^2.$$

**结果汇总。** Qwen4B和三模型比较 `{json.dumps(result['comparison'], ensure_ascii=False)}`；通过模型 `{json.dumps(result['qualified_models'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2385_c17841_c18160_crossmodel_isolated_content.py`；三模型完整独立句本地坐标场、逐层锁箱和索引位于 `tests/glm5/result/phase2385_c17841_c18160_crossmodel_isolated_content`。

**理论进展、问题硬伤与结论。** 多模型复验的是功能关系而不是坐标编号。纯embedding基线揭示词汇内容已经提供大量身份信息；深层提升才是组合编码候选。量化差异和DS的Qwen谱系仍限制架构普遍性。这个观察闭合仍不等于生成闭合或因果闭合；下一Phase将修复自主生成的停止/思考混淆，用聊天模板关闭thinking并以四句停止条件复测完整内容。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    source_rows = [row for row in read_rows(MATERIAL) if row["task"] == "exact_copy"]; panel = panel_indices(); models = {}
    for key in MODELS: models[key] = run_model(key, source_rows, panel)
    qwen4 = json.loads((P2384 / "analysis/final.json").read_text(encoding="utf-8"))["analysis"]["selected"]
    comparison = {"qwen4b": {"lockbox": qwen4["lockbox_accuracy"], "embedding_mean": qwen4["embedding_mean_q0_lockbox"],
                               "donor": qwen4["cross_row_content_donor_lockbox"], "coordinate_shuffle": qwen4["row_specific_coordinate_permutation_lockbox"]}}
    for key, value in models.items(): comparison[key] = {"lockbox": value["selected"]["lockbox_accuracy"],
        "embedding_mean": value["selected"]["embedding_mean_lockbox"], "donor": value["selected"]["donor_lockbox"],
        "coordinate_shuffle": value["selected"]["row_specific_coordinate_permutation_lockbox"], "behavior": value["behavior"]["target_over_foil"]}
    qualified = ["qwen4b", *[key for key, value in models.items() if value["qualified"]]]
    checks = {"all_models": set(models) == set(MODELS), "all_shapes": all(value["collection"]["shape"][0:2] == [768, 4] for value in models.values()),
              "sequential_order": list(models) == list(MODELS), "finite": all(math.isfinite(value["selected"]["lockbox_accuracy"]) for value in models.values())}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "comparison": comparison,
              "qualified_models": qualified, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
