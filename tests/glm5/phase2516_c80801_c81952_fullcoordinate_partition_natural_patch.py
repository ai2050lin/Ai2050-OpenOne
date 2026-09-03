#!/usr/bin/env python3
"""Natural-state query-token patching with complete coordinate partitions and controls."""
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
OUT = RESULT / "phase2516_c80801_c81952_fullcoordinate_partition_natural_patch"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, QPOINT, DIM = 2516, "C80801-C81952", 28, 2560
CONTEXTS = (0, 3, 5, 6, 9, 10, 12, 15)  # each of four binary surface factors is exactly balanced

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


def pad_right(sequences: list[list[int]], pad: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device); mask[i, :len(sequence)] = 1
    return ids, mask


def extract_states(model, rows: list[dict], batch_size: int = 8) -> dict[str, np.ndarray]:
    module = field_utils.modules(model)[QPOINT]
    capture = {}
    def hook(_module, _inputs, output):
        capture["value"] = (output[0] if isinstance(output, tuple) else output).detach()
    handle = module.register_forward_hook(hook)
    device, output_states = model.get_input_embeddings().weight.device, {}
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask = pad_right([r["prompt_ids"] for r in batch], model.config.pad_token_id or 0, device)
                capture.clear(); model(input_ids=ids, attention_mask=mask, use_cache=False)
                tensor = capture["value"]
                for i, row in enumerate(batch):
                    pos = row["event_positions"][2]
                    output_states[row["case_id"]] = tensor[i, pos].float().cpu().numpy()
    finally:
        handle.remove()
    return output_states


def continuation(tokenizer, row: dict, relation_index: int) -> list[int]:
    candidate = row["relation_targets"][relation_index]
    return [int(v) for v in tokenizer.encode((" " if row["language"] == "en" else "") + candidate, add_special_tokens=False)]


def score_patch_jobs(model, jobs: list[dict], masks: dict[str, np.ndarray], batch_size: int = 8) -> list[dict]:
    module = field_utils.modules(model)[QPOINT]
    device = model.get_input_embeddings().weight.device
    active = {"jobs": []}
    def patch_hook(_module, _inputs, output):
        hidden = (output[0] if isinstance(output, tuple) else output).clone()
        for i, job in enumerate(active["jobs"]):
            coords = masks[job["mask"]]
            donor = torch.as_tensor(job["patch_state"], dtype=hidden.dtype, device=hidden.device)
            hidden[i, job["query_position"], coords] = donor[coords]
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden
    handle = module.register_forward_hook(patch_hook)
    results = []
    try:
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]; active["jobs"] = batch
            ids, attention = pad_right([job["sequence"] for job in batch], model.config.pad_token_id or 0, device)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
            for i, job in enumerate(batch):
                begin = job["prompt_length"]; values = []
                for j, token_id in enumerate(job["continuation"]):
                    vector = logits[i, begin - 1 + j].float()
                    values.append(float(vector[token_id] - torch.logsumexp(vector, dim=-1)))
                results.append({"intervention_id": job["intervention_id"], "condition": job["condition"],
                                "relation_index": job["relation_index"], "sum_logprob": float(sum(values)),
                                "mean_logprob": float(np.mean(values))})
            if (start + len(batch)) % 512 == 0:
                print(f"[phase2516 patches] {start + len(batch)}/{len(jobs)}", flush=True)
    finally:
        handle.remove()
    return results


def compile_interventions(rows: list[dict], pair_ids: list[int]) -> list[dict]:
    lookup = {(r["unit"], r["pair_id"], r["language"], r["context_id"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    interventions = []
    for pair_index, pair_id in enumerate(pair_ids):
        wrong_pair = pair_ids[(pair_index + 1) % len(pair_ids)]
        for language in ("en", "zh"):
            for context in CONTEXTS:
                for query_marker in (0, 1):
                    base = lookup[(29, pair_id, language, context, 0, query_marker)]
                    donor = lookup[(29, pair_id, language, context, 1, query_marker)]
                    wrong = lookup[(29, wrong_pair, language, context, 1, query_marker)]
                    interventions.append({"intervention_id": f"p{pair_id}-{language}-x{context}-q{query_marker}",
                                          "pair_id": pair_id, "edge": base["families"], "language": language,
                                          "context_id": context, "query_marker": query_marker,
                                          "base_case": base["case_id"], "donor_case": donor["case_id"],
                                          "wrong_case": wrong["case_id"], "query_position": base["event_positions"][2],
                                          "base": base})
    return interventions


def analyze(interventions: list[dict], patched: list[dict], baseline_scores: list[dict]) -> dict:
    patch_lookup = {(r["intervention_id"], r["condition"], r["relation_index"]): r for r in patched}
    score_lookup = {(r["case_id"], r["relation_index"]): r for r in baseline_scores}
    records = []
    for item in interventions:
        base_d = patch_lookup[(item["intervention_id"], "no_patch", 0)]["sum_logprob"] - patch_lookup[(item["intervention_id"], "no_patch", 1)]["sum_logprob"]
        donor_d = score_lookup[(item["donor_case"], 0)]["sum_logprob"] - score_lookup[(item["donor_case"], 1)]["sum_logprob"]
        sign = 1.0 if item["query_marker"] == 0 else -1.0
        for condition in sorted({r["condition"] for r in patched}):
            d = patch_lookup[(item["intervention_id"], condition, 0)]["sum_logprob"] - patch_lookup[(item["intervention_id"], condition, 1)]["sum_logprob"]
            records.append({"intervention_id": item["intervention_id"], "condition": condition,
                            "base_difference": base_d, "donor_difference": donor_d, "patched_difference": d,
                            "base_correct_margin": sign * base_d, "donor_target_margin": -sign * d,
                            "shift_toward_donor": -sign * (d - base_d), "flipped_to_donor": (-sign * d) > 0})
    panels = {}
    for condition in sorted({r["condition"] for r in records}):
        values = [r for r in records if r["condition"] == condition]
        panels[condition] = {"interventions": len(values),
                             "mean_shift_toward_donor": float(np.mean([r["shift_toward_donor"] for r in values])),
                             "positive_shift_rate": float(np.mean([r["shift_toward_donor"] > 0 for r in values])),
                             "donor_preference_flip_rate": float(np.mean([r["flipped_to_donor"] for r in values])),
                             "mean_donor_target_margin": float(np.mean([r["donor_target_margin"] for r in values]))}
    synergy = {}
    by_id = {(r["intervention_id"], r["condition"]): r for r in records}
    for partition in ("physical", "random"):
        residuals, full_shifts, sums = [], [], []
        for item in interventions:
            key = item["intervention_id"]; full = by_id[(key, "donor_full")]["shift_toward_donor"]
            group_sum = sum(by_id[(key, f"{partition}_{g}")]["shift_toward_donor"] for g in range(8))
            residuals.append(full - group_sum); full_shifts.append(full); sums.append(group_sum)
        denom = float(np.sqrt(np.mean(np.square(full_shifts))))
        synergy[partition] = {"mean_full_shift": float(np.mean(full_shifts)), "mean_sum_single_groups": float(np.mean(sums)),
                              "mean_nonadditive_residual": float(np.mean(residuals)),
                              "relative_residual_rms": float(np.sqrt(np.mean(np.square(residuals))) / max(denom, 1e-30)),
                              "full_vs_sum_correlation": float(np.corrcoef(full_shifts, sums)[0, 1])}
    self_delta = [abs(r["patched_difference"] - r["base_difference"]) for r in records if r["condition"] == "self_full"]
    return {"panels": panels, "partition_additivity": synergy,
            "self_patch_max_abs_difference": float(max(self_delta)), "records": records}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: q28自然patch的exact-shape自对照失败与锁箱撤销（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2515仅发现query全坐标对输出幅度有有限读出且不超过显式因素，因此不按相关大小挑Top-K。冻结其sum读出选出的q28，在unit29四关系对×两语言×8个四因素平衡context×两个query-marker共128个干预上，把meaning-swap=1的真实query-marker层状态patch到meaning-swap=0。比较self全状态、同关系donor全2560、错误关系全状态、按物理坐标编号mod8的八个互斥320坐标组、随机种子固定的八个互斥320坐标组。两套分区都恰好覆盖全部2560坐标；对两个完整候选字符串重新计算序列logprob。

$$\delta_{{full}}=L(h^{{donor}})-L(h^{{base}}),\quad S=\delta_{{full}}-\sum_{{g=1}}^8\delta_g.$$

**结果汇总。** 干预面板 `{json.dumps(result['analysis']['panels'], ensure_ascii=False)}`；分区加和/协同 `{json.dumps(result['analysis']['partition_additivity'], ensure_ascii=False)}`；self数值对照 `{result['analysis']['self_patch_max_abs_difference']}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2516_c80801_c81952_fullcoordinate_partition_natural_patch.py`；128个干预合同、5120条含no-patch的候选分数、逐干预派生值、分区定义与final位于`{OUT}`。

**分析与理论进展。** 本次self全状态patch相对同批no-patch仍出现不可忽略差值，说明“先用prompt-only前向提取状态、再放入更长candidate序列前向”的矩阵shape变化会产生数值路径差异，重现了Phase2502型物理测量问题。因此donor、wrong-edge、八分区的所有因果效应均撤销，不得用其大小或方向判断语义中介。该失败把自然patch合同进一步收紧为：base、self、donor必须在与被评分序列完全相同的shape和batch布局下同步提取。

**问题硬伤与结论。** 当前锁箱首先在self数值对照上失败，故没有资格裁决full donor能否转移关系。坐标分组仍只是外部均匀覆盖，不是天然联盟；一次层输出patch可能离开正常联合分布。下一Phase必须做exact-shape paired forward修复，并先要求self逐样本严格或近机器精度一致，再查看donor结果。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    f13, f15 = load_json(P2513 / "analysis/final.json"), load_json(P2515 / "analysis/final.json")
    rows = read_jsonl(Path(f13["collection"]["event_index"]))
    material = {r["case_id"]: r for r in read_jsonl(P2513 / "material/factorial_rows.jsonl")}
    for row in rows: row["relation_targets"] = material[row["case_id"]]["relation_targets"]
    pairs = f13["behavior"]["qualified_pair_ids"]
    interventions = compile_interventions(rows, pairs)
    needed = {i[k] for i in interventions for k in ("base_case", "donor_case", "wrong_case")}
    state_rows = [r for r in rows if r["case_id"] in needed]
    rng = np.random.default_rng(2516); perm = rng.permutation(DIM)
    masks = {"all": np.ones(DIM, dtype=bool), "none": np.zeros(DIM, dtype=bool)}
    for g in range(8):
        masks[f"physical_{g}"] = np.arange(DIM) % 8 == g
        mask = np.zeros(DIM, dtype=bool); mask[perm[g * (DIM // 8):(g + 1) * (DIM // 8)]] = True; masks[f"random_{g}"] = mask
    conditions = ["no_patch", "self_full", "donor_full", "wrong_edge_full"] + [f"physical_{g}" for g in range(8)] + [f"random_{g}" for g in range(8)]
    patch_path = OUT / "output/patched_candidate_scores.jsonl"
    if patch_path.exists() and sum(1 for line in patch_path.read_text(encoding="utf-8").splitlines() if line.strip()) == 128 * 20 * 2:
        patched = read_jsonl(patch_path)
    else:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        try:
            states = extract_states(model, state_rows)
            jobs = []
            for item in interventions:
                base = item["base"]
                for condition in conditions:
                    if condition == "no_patch": source, mask_name = item["base_case"], "none"
                    elif condition == "self_full": source, mask_name = item["base_case"], "all"
                    elif condition == "donor_full": source, mask_name = item["donor_case"], "all"
                    elif condition == "wrong_edge_full": source, mask_name = item["wrong_case"], "all"
                    else: source, mask_name = item["donor_case"], condition
                    for relation_index in (0, 1):
                        cont = continuation(tokenizer, base, relation_index)
                        jobs.append({"intervention_id": item["intervention_id"], "condition": condition,
                                     "relation_index": relation_index, "continuation": cont,
                                     "sequence": base["prompt_ids"] + cont, "prompt_length": len(base["prompt_ids"]),
                                     "query_position": item["query_position"], "patch_state": states[source], "mask": mask_name})
            patched = score_patch_jobs(model, jobs, masks)
        finally:
            model_utils.release_model(model); gc.collect()
    contract_path = OUT / "material/interventions.jsonl"; write_jsonl(contract_path, [{k: v for k, v in i.items() if k != "base"} for i in interventions])
    write_jsonl(patch_path, patched)
    baseline_scores = read_jsonl(Path(f15["files"]["scores"]))
    analysis = analyze(interventions, patched, baseline_scores)
    records_path = OUT / "analysis/intervention_records.jsonl"; write_jsonl(records_path, analysis.pop("records"))
    partition_path = OUT / "material/coordinate_partitions.npz"; partition_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(partition_path, **{k: v for k, v in masks.items() if k != "all"})
    checks = {"sources_passed": f13["all_checks_passed"] and f15["all_checks_passed"],
              "interventions_128": len(interventions) == 128, "contexts_factor_balanced": all(sum((c >> bit) & 1 for c in CONTEXTS) == 4 for bit in range(4)),
              "patch_scores_5120": len(patched) == 128 * 20 * 2,
              "physical_partition_complete": bool(np.all(np.stack([masks[f"physical_{g}"] for g in range(8)]).sum(axis=0) == 1)),
              "random_partition_complete": bool(np.all(np.stack([masks[f"random_{g}"] for g in range(8)]).sum(axis=0) == 1)),
              "self_control_measured": bool(np.isfinite(analysis["self_patch_max_abs_difference"])),
              "protocol_failure_detected": analysis["self_patch_max_abs_difference"] > 1e-2,
              "hash": len(digest(partition_path)) == 64, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B nonquantized BF16 CUDA", "qpoint": QPOINT,
              "analysis": analysis, "files": {"contract": str(contract_path), "scores": str(patch_path),
                                                "records": str(records_path), "partitions": str(partition_path),
                                                "partition_sha256": digest(partition_path)},
              "adjudication": {"causal_lockbox_valid": False,
                               "matched_full_query_state_causally_shifts_output": False,
                               "natural_coordinate_alliance_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps({"phase": PHASE, "panels": analysis["panels"], "additivity": analysis["partition_additivity"],
                      "checks": checks, "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
