#!/usr/bin/env python3
"""Repair natural-state patching with exact-shape paired source and target forwards."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2513 = RESULT / "phase2513_c76673_c78624_fresh_context_factorial_behavior_fullfield"
P2515 = RESULT / "phase2515_c79777_c80800_output_margin_fullcoordinate_readout"
P2516 = RESULT / "phase2516_c80801_c81952_fullcoordinate_partition_natural_patch"
OUT = RESULT / "phase2517_c81953_c82976_exact_shape_paired_natural_patch"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, QPOINT, DIM = 2517, "C81953-C82976", 28, 2560
CONTEXTS = (0, 3, 5, 6, 9, 10, 12, 15)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pad(sequences: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences): ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device); mask[i, :len(seq)] = 1
    return ids, mask


def score_logits(logits: torch.Tensor, jobs: list[dict]) -> list[float]:
    totals = []
    for i, job in enumerate(jobs):
        values = []
        for j, token_id in enumerate(job["continuation"]):
            vector = logits[i, job["prompt_length"] - 1 + j].float()
            values.append(float(vector[token_id] - torch.logsumexp(vector, dim=-1)))
        totals.append(float(sum(values)))
    return totals


def run_exact(model, tokenizer, interventions: list[dict], masks: dict[str, np.ndarray], batch_size: int = 8) -> list[dict]:
    module = field_utils.modules(model)[QPOINT]
    device, active, captured = model.get_input_embeddings().weight.device, {"vectors": None, "masks": None, "positions": None}, {}
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["states"] = hidden.detach().clone()
        if active["vectors"] is None:
            return None
        changed = hidden.clone()
        for i in range(hidden.shape[0]):
            mask = active["masks"][i]
            changed[i, active["positions"][i], mask] = active["vectors"][i, mask].to(changed.dtype)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    handle = module.register_forward_hook(hook)
    sequence_jobs = []
    for item in interventions:
        for relation_index in (0, 1):
            candidate = item["base"]["relation_targets"][relation_index]
            cont = [int(v) for v in tokenizer.encode((" " if item["language"] == "en" else "") + candidate, add_special_tokens=False)]
            sequence_jobs.append({"intervention_id": item["intervention_id"], "relation_index": relation_index,
                                  "continuation": cont, "prompt_length": len(item["base"]["prompt_ids"]),
                                  "base_sequence": item["base"]["prompt_ids"] + cont,
                                  "donor_sequence": item["donor"]["prompt_ids"] + cont,
                                  "query_position": item["query_position"]})
    conditions = ["self_full", "donor_full", "shuffled_donor_full"] + [f"physical_{g}" for g in range(8)] + [f"random_{g}" for g in range(8)]
    results = []
    try:
        for start in range(0, len(sequence_jobs), batch_size):
            jobs = sequence_jobs[start:start + batch_size]
            base_sequences, donor_sequences = [j["base_sequence"] for j in jobs], [j["donor_sequence"] for j in jobs]
            assert [len(s) for s in base_sequences] == [len(s) for s in donor_sequences]
            base_ids, base_mask = pad(base_sequences, tokenizer.pad_token_id, device)
            donor_ids, donor_mask = pad(donor_sequences, tokenizer.pad_token_id, device)
            assert base_ids.shape == donor_ids.shape and torch.equal(base_mask, donor_mask)
            active.update(vectors=None, masks=None, positions=None); captured.clear()
            with torch.inference_mode(): base_logits = model(input_ids=base_ids, attention_mask=base_mask, use_cache=False).logits
            base_states = captured["states"][torch.arange(len(jobs), device=device), torch.tensor([j["query_position"] for j in jobs], device=device)].clone()
            base_scores = score_logits(base_logits, jobs)
            for job, value in zip(jobs, base_scores):
                results.append({"intervention_id": job["intervention_id"], "condition": "no_patch",
                                "relation_index": job["relation_index"], "sum_logprob": value})
            active.update(vectors=None, masks=None, positions=None); captured.clear()
            with torch.inference_mode(): model(input_ids=donor_ids, attention_mask=donor_mask, use_cache=False)
            donor_states = captured["states"][torch.arange(len(jobs), device=device), torch.tensor([j["query_position"] for j in jobs], device=device)].clone()
            shuffled_states = donor_states.roll(shifts=2 if len(jobs) > 2 else 1, dims=0)
            positions = [j["query_position"] for j in jobs]
            for condition in conditions:
                if condition == "self_full": vectors, mask_name = base_states, "all"
                elif condition == "donor_full": vectors, mask_name = donor_states, "all"
                elif condition == "shuffled_donor_full": vectors, mask_name = shuffled_states, "all"
                else: vectors, mask_name = donor_states, condition
                active.update(vectors=vectors, masks=[masks[mask_name]] * len(jobs), positions=positions); captured.clear()
                with torch.inference_mode(): logits = model(input_ids=base_ids, attention_mask=base_mask, use_cache=False).logits
                for job, value in zip(jobs, score_logits(logits, jobs)):
                    results.append({"intervention_id": job["intervention_id"], "condition": condition,
                                    "relation_index": job["relation_index"], "sum_logprob": value})
            if start % 32 == 0: print(f"[phase2517 exact batches] {min(start + len(jobs), len(sequence_jobs))}/{len(sequence_jobs)}", flush=True)
    finally:
        handle.remove()
    return results


def compile_interventions(rows: list[dict], pairs: list[int]) -> list[dict]:
    lookup = {(r["unit"], r["pair_id"], r["language"], r["context_id"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    output = []
    for pair_id in pairs:
        for language in ("en", "zh"):
            for context in CONTEXTS:
                for query in (0, 1):
                    base, donor = lookup[(29, pair_id, language, context, 0, query)], lookup[(29, pair_id, language, context, 1, query)]
                    assert len(base["prompt_ids"]) == len(donor["prompt_ids"]) and base["event_positions"][2] == donor["event_positions"][2]
                    output.append({"intervention_id": f"p{pair_id}-{language}-x{context}-q{query}", "pair_id": pair_id,
                                   "edge": base["families"], "language": language, "context_id": context, "query_marker": query,
                                   "query_position": base["event_positions"][2], "base": base, "donor": donor})
    return output


def analyze(interventions: list[dict], scores: list[dict]) -> tuple[dict, list[dict]]:
    lookup = {(r["intervention_id"], r["condition"], r["relation_index"]): r["sum_logprob"] for r in scores}
    conditions = sorted({r["condition"] for r in scores}); records = []
    for item in interventions:
        key, sign = item["intervention_id"], (1.0 if item["query_marker"] == 0 else -1.0)
        base_d = lookup[(key, "no_patch", 0)] - lookup[(key, "no_patch", 1)]
        for condition in conditions:
            value = lookup[(key, condition, 0)] - lookup[(key, condition, 1)]
            records.append({"intervention_id": key, "condition": condition, "base_difference": base_d,
                            "patched_difference": value, "shift_toward_donor": -sign * (value - base_d),
                            "donor_target_margin": -sign * value, "flipped_to_donor": -sign * value > 0})
    panels = {}
    for condition in conditions:
        subset = [r for r in records if r["condition"] == condition]
        panels[condition] = {"n": len(subset), "mean_shift_toward_donor": float(np.mean([r["shift_toward_donor"] for r in subset])),
                             "positive_shift_rate": float(np.mean([r["shift_toward_donor"] > 0 for r in subset])),
                             "donor_flip_rate": float(np.mean([r["flipped_to_donor"] for r in subset])),
                             "mean_donor_target_margin": float(np.mean([r["donor_target_margin"] for r in subset]))}
    by = {(r["intervention_id"], r["condition"]): r for r in records}
    additivity = {}
    for partition in ("physical", "random"):
        full, summed = [], []
        for item in interventions:
            key = item["intervention_id"]; full.append(by[(key, "donor_full")]["shift_toward_donor"])
            summed.append(sum(by[(key, f"{partition}_{g}")]["shift_toward_donor"] for g in range(8)))
        full, summed = np.asarray(full), np.asarray(summed); residual = full - summed
        additivity[partition] = {"mean_full": float(full.mean()), "mean_sum_groups": float(summed.mean()),
                                 "relative_residual_rms": float(np.sqrt(np.mean(residual ** 2)) / max(np.sqrt(np.mean(full ** 2)), 1e-30)),
                                 "full_vs_sum_correlation": float(np.corrcoef(full, summed)[0, 1])}
    self_error = [abs(r["patched_difference"] - r["base_difference"]) for r in records if r["condition"] == "self_full"]
    return {"panels": panels, "partition_additivity": additivity,
            "self_patch_max_abs_difference": float(max(self_error)), "self_patch_mean_abs_difference": float(np.mean(self_error))}, records


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: exact-shape成对前向修复后的全坐标自然patch裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 修复Phase2516的prompt-only/完整序列shape不一致。对每个unit29干预和每个候选序列，在完全相同batch布局、矩阵shape、attention mask与candidate后缀长度下依次运行base和meaning-swap donor，现场提取q28 query-token状态；随后在同一base张量上运行self、matched donor、batch内错配donor、物理mod8与固定随机八分区。128干预×2候选，每个条件5120条分数记录。只有self全状态与no-patch在机器精度一致时才查看donor。

$$h^{{patch}}_{{28,t_q,S}}\leftarrow h^{{donor}}_{{28,t_q,S}},\qquad \operatorname{{shape}}(X_{{base}})=\operatorname{{shape}}(X_{{donor}}).$$

**结果汇总。** exact-shape面板 `{json.dumps(result['analysis']['panels'], ensure_ascii=False)}`；分区加和 `{json.dumps(result['analysis']['partition_additivity'], ensure_ascii=False)}`；self对照 `{json.dumps({k: result['analysis'][k] for k in ('self_patch_max_abs_difference','self_patch_mean_abs_difference')}, ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2517_c81953_c82976_exact_shape_paired_natural_patch.py`；exact-shape干预合同、5120条候选分数、逐干预记录、分区哈希与final位于`{OUT}`。

**分析与理论进展。** 若self通过而matched donor明显强于错配donor，才能说明该单点状态具有关系匹配的因果作用；若matched donor仍不能稳定推动输出，则q28单一query token不是足够中介，不能靠缩小坐标集补救。八分区只检验全场效应是否近似按外部分区相加，不声称存在天然八模块。

**问题硬伤与结论。** exact-shape控制只消除已识别的数值核混杂；donor仍同时包含definition绑定、检索和关系选择；自然patch可能产生跨prompt不协调状态；只测q28单层单位置。无论正负都不足以闭合语言机制。下一阶段若单点中介失败，应把对象扩展为多位置/多事件计算路径及自然语言行为，而非继续优化同一patch。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    f13, f15, f16 = (load_json(P2513 / "analysis/final.json"), load_json(P2515 / "analysis/final.json"), load_json(P2516 / "analysis/final.json"))
    rows = read_jsonl(Path(f13["collection"]["event_index"])); material = {r["case_id"]: r for r in read_jsonl(P2513 / "material/factorial_rows.jsonl")}
    for row in rows: row["relation_targets"] = material[row["case_id"]]["relation_targets"]
    interventions = compile_interventions(rows, f13["behavior"]["qualified_pair_ids"])
    rng = np.random.default_rng(2517); perm = rng.permutation(DIM); masks = {"all": np.ones(DIM, dtype=bool)}
    for g in range(8):
        masks[f"physical_{g}"] = np.arange(DIM) % 8 == g
        m = np.zeros(DIM, dtype=bool); m[perm[g * 320:(g + 1) * 320]] = True; masks[f"random_{g}"] = m
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try: scores = run_exact(model, tokenizer, interventions, masks)
    finally: model_utils.release_model(model); gc.collect()
    score_path = OUT / "output/exact_shape_scores.jsonl"; write_jsonl(score_path, scores)
    analysis, records = analyze(interventions, scores)
    record_path = OUT / "analysis/intervention_records.jsonl"; write_jsonl(record_path, records)
    contract_path = OUT / "material/interventions.jsonl"; write_jsonl(contract_path, [{k: v for k, v in i.items() if k not in ("base", "donor")} for i in interventions])
    part_path = OUT / "material/partitions.npz"; part_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(part_path, **masks)
    self_valid = analysis["self_patch_max_abs_difference"] < 1e-6
    matched = analysis["panels"]["donor_full"]; shuffled = analysis["panels"]["shuffled_donor_full"]
    checks = {"sources_passed": f13["all_checks_passed"] and f15["all_checks_passed"] and f16["all_checks_passed"],
              "interventions_128": len(interventions) == 128, "scores_5120": len(scores) == 5120,
              "exact_shape_asserted": True, "self_control_valid": self_valid,
              "partitions_complete": all(np.all(np.stack([masks[f"{kind}_{g}"] for g in range(8)]).sum(axis=0) == 1) for kind in ("physical", "random")),
              "hash": len(digest(part_path)) == 64, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B nonquantized BF16 CUDA", "qpoint": QPOINT,
              "analysis": analysis, "files": {"scores": str(score_path), "records": str(record_path), "contract": str(contract_path),
                                                "partitions": str(part_path), "partition_sha256": digest(part_path)},
              "adjudication": {"causal_lockbox_valid": self_valid,
                               "matched_donor_relation_specific_advantage": matched["mean_shift_toward_donor"] - shuffled["mean_shift_toward_donor"],
                               "single_query_token_sufficient_mediator": bool(self_valid and matched["positive_shift_rate"] > .75 and matched["donor_flip_rate"] > .5),
                               "natural_coordinate_gear_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps({"phase": PHASE, "analysis": analysis, "adjudication": result["adjudication"],
                      "checks": checks, "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
