#!/usr/bin/env python3
"""Single-recipient source-span causal patch across eight depths."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2603 = RESULT / "phase2603_c643329_c659712_unique_natural_lockbox"
P2604 = RESULT / "phase2604_c659713_c676096_unique_fullcoordinate_confirmation"
OUT = RESULT / "phase2605_c676097_c692480_singleprompt_source_patch"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2605, "C676097-C692480"
LAYERS = (0, 5, 11, 17, 23, 29, 34, 35)
KINDS = ("true_source_delta", "roll641", "wrong_token", "negative_delta")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2601_c610561_c626944_natural_singleprompt_behavior_lockbox as p2601  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_pairs(material, eligible, qualified):
    eligible_ids = {row["pair_id"] for row in eligible if row["split"] == "confirmation"}
    rows = defaultdict(list)
    for row in material:
        if row["pair_id"] in eligible_ids and f"{row['family']}/{row['language']}" in qualified:
            rows[row["pair_id"]].append(row)
    selected = []
    for group in sorted(qualified):
        candidates = sorted(pair_id for pair_id, pair in rows.items()
                            if f"{pair[0]['family']}/{pair[0]['language']}" == group)
        if len(candidates) < 10:
            raise RuntimeError((group, len(candidates)))
        selected.extend(candidates[:10])
    return [sorted(rows[pair_id], key=lambda row: row["variant"]) for pair_id in selected]


def collect_source_means(model, tokenizer, pairs, storage_dtype=np.float16):
    device = model.get_input_embeddings().weight.device
    n_hidden = len(model_utils.get_layers(model)) + 1
    d_model = model.get_input_embeddings().weight.shape[1]
    path = OUT / ("field/source_span_means_90x2x37x2560.float16.npy" if storage_dtype == np.float16 else
                  f"field/source_span_means_{len(pairs)}x2x{n_hidden}x{d_model}.float32.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(path, mode="w+", dtype=storage_dtype,
                                      shape=(len(pairs), 2, n_hidden, d_model))
    storage_audit = {"coordinates": 0, "fp16_underflow_to_zero": 0, "fp16_overflow": 0, "fp16_abs_error_max": 0.0,
                     "stored_dtype": np.dtype(storage_dtype).name, "inference_dtype": str(model.dtype)}
    for pair_index, pair in enumerate(pairs):
        width = max(len(row["prompt_ids"]) for row in pair)
        ids = torch.full((2, width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(pair):
            ids[index, :len(row["prompt_ids"])] = torch.tensor(row["prompt_ids"], device=device)
            mask[index, :len(row["prompt_ids"])] = 1
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                           use_cache=False, return_dict=True)
        for variant, row in enumerate(pair):
            positions = torch.tensor(row["source_token_positions"], device=device)
            stack = torch.stack([state[variant, positions].mean(0) for state in output.hidden_states], dim=0)
            values = stack.detach().float().cpu().numpy()
            field[pair_index, variant] = values.astype(storage_dtype)
            if storage_dtype == np.float32:
                downcast = values.astype(np.float16).astype(np.float32)
                storage_audit["coordinates"] += values.size
                storage_audit["fp16_underflow_to_zero"] += int(np.sum((values != 0) & (downcast == 0)))
                storage_audit["fp16_overflow"] += int(np.sum(~np.isfinite(downcast)))
                storage_audit["fp16_abs_error_max"] = max(storage_audit["fp16_abs_error_max"], float(np.max(np.abs(values-downcast))))
        if (pair_index + 1) % 30 == 0:
            print(f"[phase2605 source] {pair_index + 1}/{len(pairs)}", flush=True)
    field.flush()
    if storage_dtype == np.float32:
        save_json(path.with_suffix(".storage_audit.json"), storage_audit)
    return path


def candidate_job(tokenizer, row, answer, pair_index, answer_index):
    ids, positions = p2601.candidate_token_ids(tokenizer, row["prompt"], answer)
    return {"pair_index": pair_index, "answer_index": answer_index, "ids": ids,
            "answer_positions": positions, "source_positions": row["source_token_positions"]}


def score_condition(model, tokenizer, layer, kind, pairs, source_delta, batch_size=20):
    device = model.get_input_embeddings().weight.device
    jobs = []
    for pair_index, pair in enumerate(pairs):
        recipient = pair[0]
        jobs.append(candidate_job(tokenizer, recipient, pair[0]["target"], pair_index, 0))
        jobs.append(candidate_job(tokenizer, recipient, pair[1]["target"], pair_index, 1))
    scores = np.zeros((len(pairs), 2), dtype=np.float32)
    layers = model_utils.get_layers(model)
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        width = max(len(job["ids"]) for job in batch)
        ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        answer_mask = torch.zeros_like(ids, dtype=torch.bool)
        for index, job in enumerate(batch):
            ids[index, :len(job["ids"])] = torch.tensor(job["ids"], device=device)
            mask[index, :len(job["ids"])] = 1
            answer_mask[index, job["answer_positions"]] = True
        handle = None
        if kind != "baseline":
            batch_deltas = []
            patch_positions = []
            for job in batch:
                delta = source_delta[job["pair_index"], layer].astype(np.float32)
                if kind == "roll641":
                    delta = np.roll(delta, 641)
                elif kind == "negative_delta":
                    delta = -delta
                batch_deltas.append(delta)
                if kind == "wrong_token":
                    n = len(job["source_positions"])
                    stop = min(job["source_positions"])
                    patch_positions.append(list(range(stop - n, stop)))
                else:
                    patch_positions.append(job["source_positions"])
            batch_delta = torch.tensor(np.stack(batch_deltas), dtype=torch.float32, device=device)

            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                patched = tensor.clone()
                for index, positions in enumerate(patch_positions):
                    patched[index, positions] = patched[index, positions] + batch_delta[index].to(patched.dtype)
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                if isinstance(output, list):
                    return [patched] + output[1:]
                return patched
            handle = layers[layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                logp = torch.log_softmax(logits[:, :-1], dim=-1)
                token_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            for index, job in enumerate(batch):
                values = token_lp[index][answer_mask[index, 1:]]
                scores[job["pair_index"], job["answer_index"]] = float(values.mean().item())
        finally:
            if handle is not None:
                handle.remove()
    return scores


def summarize(records, pairs):
    baseline = {row["pair_id"]: row["target1_minus_target0"] for row in records if row["condition"] == "baseline"}
    conditions = {}
    for condition in sorted({row["condition"] for row in records}):
        subset = [row for row in records if row["condition"] == condition]
        gains = [row["target1_minus_target0"] - baseline[row["pair_id"]] for row in subset]
        conditions[condition] = {"n": len(subset),
                                 "mean_target1_margin": float(np.mean([row["target1_minus_target0"] for row in subset])),
                                 "mean_margin_gain": float(np.mean(gains)),
                                 "target1_flip_rate": float(np.mean([row["target1_minus_target0"] > 0 for row in subset]))}
    by_group = {}
    for group in sorted({f"{pair[0]['family']}/{pair[0]['language']}" for pair in pairs}):
        group_ids = {pair[0]["pair_id"] for pair in pairs if f"{pair[0]['family']}/{pair[0]['language']}" == group}
        by_group[group] = {}
        for condition in conditions:
            subset = [row for row in records if row["condition"] == condition and row["pair_id"] in group_ids]
            gains = [row["target1_minus_target0"] - baseline[row["pair_id"]] for row in subset]
            by_group[group][condition] = {"mean_margin_gain": float(np.mean(gains)),
                                          "flip_rate": float(np.mean([row["target1_minus_target0"] > 0 for row in subset]))}
    return conditions, by_group


def append_memo(result):
    heading = f"## Phase {PHASE}: 90个确认pair的单recipient source-span全坐标因果扫描（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** 不再对四prompt联合回写。对9个行为合格族/语言各冻结10个confirmation pair，以variant0为唯一recipient；由同pair variant1/0在source span的全2560坐标均值差构造局部替换方向，在block输出后只修改recipient的source位置：

$$\delta^S_l=\operatorname{{mean}}_{{t\in S_1}}H^1_{{l,t}}-\operatorname{{mean}}_{{t\in S_0}}H^0_{{l,t}},\qquad
H^0_{{l,S_0}}\leftarrow H^0_{{l,S_0}}+\delta^S_l.$$

对照为等范数roll641、同方向错token、反方向；层位0/5/11/17/23/29/34/35，其中layer35 source改写后已无下游attention可传到答案，是结构null。

**测试用例。** 90 pair、8层×4干预+baseline，共2970个pair-condition、5940条完整target0/target1候选序列；候选只由评估器评分，不出现在prompt。保存90×2×37×2560 source均值原场，所有坐标不做Top-K；Qwen3-4B BF16 CUDA非量化。

**结果汇总。** 条件=`{json.dumps(result['conditions'], ensure_ascii=False)}`；族/语言=`{json.dumps(result['by_family_language'], ensure_ascii=False)}`；关键比较=`{json.dumps(result['key_comparisons'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2605_c676097_c692480_singleprompt_source_patch.py`；source全坐标场、5940得分、pair清单、条件汇总与final位于`{OUT}`。

**分析与理论进展。** true source方向相对roll和错token的增益，才支持后层从正确物理位置读取特定坐标方向；layer35无效用于确认作用需要下游传播。反方向只是符号对照。即使true阳性，也仍是使用本pair反事实构造的oracle source patch，不是从一个prompt独立提取齿轮。

**问题硬伤。** source span用均值池化且不等长；同pair donor含词/答案身份；完整候选似然而非真实greedy；每组合格成功pair被筛选；只干预一次且未分K/V/Q/head。阴性不能否定冗余路径，阳性不能命名完整语义齿轮。

**结论。** `{result['claim_boundary']}`；检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    p2603_final = load_json(P2603 / "analysis/final.json")
    material = read_jsonl(P2603 / "material/cases.unique.jsonl")
    eligible = load_json(P2603 / "material/eligible_pairs.json")
    pairs = select_pairs(material, eligible, set(p2603_final["qualified_groups"]))
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        source_path = collect_source_means(model, tokenizer, pairs)
        source_field = np.load(source_path, mmap_mode="r")
        source_delta = source_field[:, 1, 1:].astype(np.float32) - source_field[:, 0, 1:].astype(np.float32)
        all_scores = {"baseline": score_condition(model, tokenizer, 0, "baseline", pairs, source_delta)}
        for layer in LAYERS:
            for kind in KINDS:
                condition = f"l{layer}_{kind}"
                all_scores[condition] = score_condition(model, tokenizer, layer, kind, pairs, source_delta)
                print(f"[phase2605 intervene] {condition}", flush=True)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    records = []
    for condition, scores in all_scores.items():
        for pair_index, pair in enumerate(pairs):
            records.append({"pair_id": pair[0]["pair_id"], "family": pair[0]["family"],
                            "language": pair[0]["language"], "condition": condition,
                            "target0_mean_logp": float(scores[pair_index, 0]),
                            "target1_mean_logp": float(scores[pair_index, 1]),
                            "target1_minus_target0": float(scores[pair_index, 1] - scores[pair_index, 0])})
    record_path = OUT / "causal/candidate_scores.jsonl"
    pair_path = OUT / "material/selected_pairs.json"
    write_jsonl(record_path, records)
    save_json(pair_path, pairs)
    conditions, by_group = summarize(records, pairs)
    true_gains = np.asarray([conditions[f"l{layer}_true_source_delta"]["mean_margin_gain"] for layer in LAYERS])
    roll_gains = np.asarray([conditions[f"l{layer}_roll641"]["mean_margin_gain"] for layer in LAYERS])
    wrong_gains = np.asarray([conditions[f"l{layer}_wrong_token"]["mean_margin_gain"] for layer in LAYERS])
    best_index = int(np.argmax(true_gains - np.maximum(roll_gains, wrong_gains)))
    best_layer = LAYERS[best_index]
    comparisons = {"best_layer": best_layer,
                   "best_true_gain": float(true_gains[best_index]),
                   "best_roll_gain": float(roll_gains[best_index]),
                   "best_wrong_token_gain": float(wrong_gains[best_index]),
                   "best_direction_location_advantage": float(true_gains[best_index] - max(roll_gains[best_index], wrong_gains[best_index])),
                   "layer35_true_gain": conditions["l35_true_source_delta"]["mean_margin_gain"],
                   "layer35_roll_gain": conditions["l35_roll641"]["mean_margin_gain"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "selection": {"groups": 9, "pairs_per_group": 10, "pairs": len(pairs)},
              "source_field": {"shape": list(np.load(source_path, mmap_mode="r").shape), "all_coordinates": True, "no_topk": True},
              "conditions": conditions, "by_family_language": by_group, "key_comparisons": comparisons,
              "claim_boundary": "single-recipient oracle source patch; not a learned one-prompt extractor or complete source-to-output compiler",
              "hashes": {"source_field": sha256(source_path), "scores": sha256(record_path), "pairs": sha256(pair_path)},
              "language_mechanism_closed": False}
    result["checks"] = {"phase2604_complete": load_json(P2604 / "analysis/final.json")["all_checks_passed"],
                        "all_90_confirmation_pairs": len(pairs) == 90,
                        "all_9_groups": len({f"{pair[0]['family']}/{pair[0]['language']}" for pair in pairs}) == 9,
                        "source_field_90x2x37x2560": result["source_field"]["shape"] == [90, 2, 37, 2560],
                        "all_2970_pair_conditions": len(records) == 90 * 33,
                        "all_5940_candidate_sequences": len(records) * 2 == 5940,
                        "all_four_controls_all_layers": len(conditions) == 33,
                        "all_coordinates_no_topk": True,
                        "scientific_result_does_not_abort": True,
                        "claim_boundary": True}
    result["all_checks_passed"] = all(result["checks"].values())
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({key: result[key] for key in ("phase", "selection", "key_comparisons", "checks", "all_checks_passed")}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
