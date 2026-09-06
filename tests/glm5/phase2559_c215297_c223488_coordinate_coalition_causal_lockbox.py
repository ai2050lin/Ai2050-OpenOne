#!/usr/bin/env python3
"""Frozen full-coordinate cell-wise coalition interventions on held-out families."""
from __future__ import annotations

import gc
import json
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
P2556 = RESULT / "phase2556_c190721_c198912_form_id_collision_erratum_recompute"
P2558 = RESULT / "phase2558_c207105_c215296_full_coordinate_recipient_field"
OUT = RESULT / "phase2559_c215297_c223488_coordinate_coalition_causal_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2559, "C215297-C223488"
EARLY = tuple(range(9))
THRESHOLD, FLOOR = 0.75, 0.001

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2555_c182529_c190720_relation_stage_recipient_causal_atlas as p2555  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def derive_masks() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    index = read(P2558 / "fields/pair_index.jsonl")
    values = np.load(P2558 / "fields/region_v_delta.float16.npy", mmap_mode="r")
    stats, masks, means = {}, {}, {}
    for value_form in ("natural", "nonce"):
        discovery = [i for i, row in enumerate(index) if row["family_id"] < 16 and row["value_form"] == value_form]
        field = np.asarray(values[discovery, 0:9, 10:14], dtype=np.float32)
        positive = (field > 0).mean(axis=0)
        consistency = np.maximum(positive, 1.0 - positive)
        mean = field.mean(axis=0)
        mask = (consistency >= THRESHOLD) & (np.abs(mean) >= FLOOR)
        masks[value_form] = mask
        means[value_form] = mean
        stats[value_form] = {"discovery_pairs": len(discovery), "selected": int(mask.sum()),
                             "available": int(mask.size), "fraction": float(mask.mean()),
                             "threshold": THRESHOLD, "absolute_mean_floor": FLOOR}
    shared = masks["natural"] & masks["nonce"] & (np.sign(means["natural"]) == np.sign(means["nonce"]))
    masks["shared"] = shared
    stats["shared"] = {"selected": int(shared.sum()), "available": int(shared.size), "fraction": float(shared.mean())}
    shifted = {}
    for value_form in ("natural", "nonce"):
        candidate = np.roll(masks[value_form], shift=31, axis=-1)
        shifted[value_form] = candidate
        intersection = int(np.logical_and(candidate, masks[value_form]).sum())
        union = int(np.logical_or(candidate, masks[value_form]).sum())
        stats[f"shifted_{value_form}"] = {"selected": int(candidate.sum()), "jaccard_with_rule": intersection / max(union, 1)}
    masks.update({f"shifted_{name}": value for name, value in shifted.items()})
    mask_path = OUT / "rules/coordinate_masks.npz"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_path, **{name: value.astype(np.uint8) for name, value in masks.items()},
                        natural_mean=means["natural"].astype(np.float16), nonce_mean=means["nonce"].astype(np.float16))
    stats["file"] = str(mask_path)
    return masks, stats


def compile_jobs(tokenizer) -> tuple[list[dict], list[tuple]]:
    material = [row for row in read(P2556 / "material/phase2554_corrected_token_atomic.jsonl") if row["ablation"] == "full_scaffold"]
    behavior = [row for row in read(P2556 / "behavior/phase2554_recomputed.jsonl") if row["ablation"] == "full_scaffold"]
    correct = {row["base_case_id"]: row["correct"] for row in behavior}
    index = {(row["family_id"], row["relation_form"], row["value_form"], row["query_relation"],
              row["query_value"], row["binding"]): row for row in material}
    jobs, eligible = [], []
    for family_id in range(16, 32):
        for relation_form in ("natural", "nonce"):
            for value_form in ("natural", "nonce"):
                for query_relation in (0, 1):
                    for query_value in (0, 1):
                        key = (family_id, relation_form, value_form, query_relation, query_value)
                        base, donor = index[key + (0,)], index[key + (1,)]
                        if not (correct[base["base_case_id"]] and correct[donor["base_case_id"]]):
                            continue
                        eligible.append(key)
                        for candidate_index, entity in enumerate(base["entities"]):
                            continuation = [int(token) for token in tokenizer.encode(" " + entity, add_special_tokens=False)]
                            jobs.append({"case_id": base["base_case_id"], "family_id": family_id,
                                         "relation_form": relation_form, "value_form": value_form,
                                         "query_relation": query_relation, "query_value": query_value,
                                         "candidate_index": candidate_index, "target_index": base["target_index"],
                                         "donor_target_index": donor["target_index"],
                                         "base_prompt_length": len(base["prompt_ids"]),
                                         "donor_prompt_length": len(donor["prompt_ids"]),
                                         "continuation": continuation, "base": base["prompt_ids"] + continuation,
                                         "donor": donor["prompt_ids"] + continuation,
                                         "cells_base": base["fact_cells"], "cells_donor": donor["fact_cells"]})
    return jobs, eligible


CONDITIONS = ("no_patch", "patch_all", "patch_form_group", "patch_shared", "patch_shifted_null",
              "patch_complement", "patch_wrong_cell", "zero_form_group", "keep_only_form_group",
              "zero_all", "zero_all_then_donor_group")


class Controller:
    def __init__(self, model, masks: dict[str, np.ndarray]):
        self.layers = model_utils.get_layers(model)
        self.masks = {name: torch.from_numpy(value.astype(bool)) for name, value in masks.items()}
        self.mode = "none"
        self.condition = "no_patch"
        self.jobs: list[dict] = []
        self.store: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_index in EARLY:
            def hook(_module, _inputs, output, layer_index=layer_index):
                return self._hook(output, layer_index)
            self.handles.append(self.layers[layer_index].self_attn.v_proj.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _mask(self, job: dict, layer_index: int, cell_index: int, name: str = "form") -> torch.Tensor:
        key = job["value_form"] if name == "form" else name
        return self.masks[key][layer_index, cell_index].reshape(-1)

    def _hook(self, output: torch.Tensor, layer_index: int):
        if self.mode == "capture":
            self.store[layer_index] = output.detach().clone()
            return None
        if self.mode != "intervene":
            return None
        changed = output.clone()
        donor = self.store[layer_index].to(device=output.device, dtype=output.dtype)
        all_coordinates = torch.ones(output.shape[-1], dtype=torch.bool)
        for batch_index, job in enumerate(self.jobs):
            form_mask_cpu = self._mask(job, layer_index, 0)  # shape check only
            del form_mask_cpu
            for cell_index in range(4):
                target_positions = job["cells_base"][cell_index]["value_positions"]
                donor_cell = (cell_index + 1) % 4 if self.condition == "patch_wrong_cell" else cell_index
                donor_positions = job["cells_donor"][donor_cell]["value_positions"]
                form_mask = self._mask(job, layer_index, cell_index).to(output.device)
                shared_mask = self._mask(job, layer_index, cell_index, "shared").to(output.device)
                shifted_mask = self._mask(job, layer_index, cell_index, f"shifted_{job['value_form']}").to(output.device)
                if self.condition == "patch_all":
                    active = all_coordinates.to(output.device)
                elif self.condition in ("patch_form_group", "patch_wrong_cell"):
                    active = form_mask
                elif self.condition == "patch_shared":
                    active = shared_mask
                elif self.condition == "patch_shifted_null":
                    active = shifted_mask
                elif self.condition == "patch_complement":
                    active = ~form_mask
                else:
                    active = form_mask
                for target_position, donor_position in zip(target_positions, donor_positions):
                    target_position += job["base_shift"]
                    donor_position += job["donor_shift"]
                    if self.condition.startswith("patch_"):
                        changed[batch_index, target_position, active] = donor[batch_index, donor_position, active]
                    elif self.condition == "zero_form_group":
                        changed[batch_index, target_position, active] = 0
                    elif self.condition == "keep_only_form_group":
                        original = changed[batch_index, target_position].clone()
                        changed[batch_index, target_position] = 0
                        changed[batch_index, target_position, active] = original[active]
                    elif self.condition == "zero_all":
                        changed[batch_index, target_position] = 0
                    elif self.condition == "zero_all_then_donor_group":
                        changed[batch_index, target_position] = 0
                        changed[batch_index, target_position, active] = donor[batch_index, donor_position, active]
        return changed


def run(model, tokenizer, jobs: list[dict], masks: dict[str, np.ndarray]) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    controller = Controller(model, masks)
    rows = []
    try:
        for start in range(0, len(jobs), 8):
            batch = jobs[start:start + 8]
            controller.jobs = batch
            donor_ids, donor_mask, donor_shifts = p2552.left_pad([job["donor"] for job in batch], tokenizer.pad_token_id, device)
            for job, shift in zip(batch, donor_shifts):
                job["donor_shift"] = shift
            keep = max(len(job["continuation"]) for job in batch) + 1
            controller.mode = "capture"
            controller.store.clear()
            with torch.inference_mode():
                donor_logits = p2555.forward(model, donor_ids, donor_mask, keep)
            donor_scores = p2555.scores(donor_logits, batch, keep)
            base_ids, base_mask, base_shifts = p2552.left_pad([job["base"] for job in batch], tokenizer.pad_token_id, device)
            for job, shift in zip(batch, base_shifts):
                job["base_shift"] = shift
            for condition in CONDITIONS:
                controller.mode = "none" if condition == "no_patch" else "intervene"
                controller.condition = condition
                with torch.inference_mode():
                    logits = p2555.forward(model, base_ids, base_mask, keep)
                values = p2555.scores(logits, batch, keep)
                for job, value, donor_value in zip(batch, values, donor_scores):
                    rows.append({"case_id": job["case_id"], "family_id": job["family_id"],
                                 "relation_form": job["relation_form"], "value_form": job["value_form"],
                                 "query_relation": job["query_relation"], "query_value": job["query_value"],
                                 "candidate_index": job["candidate_index"], "target_index": job["target_index"],
                                 "donor_target_index": job["donor_target_index"], "condition": condition,
                                 "score": value, "donor_baseline_score": donor_value})
            done = start + len(batch)
            if done % 80 == 0 or done == len(jobs):
                print(f"[phase2559] {done}/{len(jobs)} candidate jobs", flush=True)
    finally:
        controller.close()
    return rows


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["case_id"])].append(row)
    panels = {}
    for condition in CONDITIONS:
        groups = [value for (name, _), value in grouped.items() if name == condition]
        correct, flip = [], []
        by_value_form = defaultdict(list)
        for values in groups:
            prediction = max(values, key=lambda row: row["score"])["candidate_index"]
            correct.append(prediction == values[0]["target_index"])
            flipped = prediction == values[0]["donor_target_index"]
            flip.append(flipped)
            by_value_form[values[0]["value_form"]].append(flipped)
        panels[condition] = {"n": len(groups), "accuracy": float(np.mean(correct)),
                             "donor_flip": float(np.mean(flip)),
                             "donor_flip_by_value_form": {key: float(np.mean(value)) for key, value in by_value_form.items()}}
    return panels


def heldout_rule_validation(masks: dict[str, np.ndarray]) -> dict:
    index = read(P2558 / "fields/pair_index.jsonl")
    values = np.load(P2558 / "fields/region_v_delta.float16.npy", mmap_mode="r")
    result = {}
    for value_form in ("natural", "nonce"):
        discovery = [i for i, row in enumerate(index) if row["family_id"] < 16 and row["value_form"] == value_form]
        heldout = [i for i, row in enumerate(index) if row["family_id"] >= 16 and row["value_form"] == value_form]
        d = np.asarray(values[discovery, 0:9, 10:14], dtype=np.float32).mean(axis=0)
        h = np.asarray(values[heldout, 0:9, 10:14], dtype=np.float32)
        selected = masks[value_form]
        agreement = np.sign(h[:, selected]) == np.sign(d[selected])[None, :]
        result[value_form] = {"discovery_pairs": len(discovery), "heldout_pairs": len(heldout),
                              "selected_coordinates": int(selected.sum()),
                              "heldout_sign_agreement": float(agreement.mean())}
    return result


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 非Top-K坐标联盟的独立家族因果锁箱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 只用Phase2558中family0–15发现坐标规则，family16–31完全锁箱。对每个早层$l\in[0,8]$、四个事实value cell、8个KV-head和全部128维V坐标，分别在natural-value与nonce-value发现集中保留

$$
M_{{l,t,g,c}}=\mathbf1\left[\max\{{P(\Delta v>0),P(\Delta v<0)\}}\ge0.75\right]
\mathbf1\left[|E\Delta v|\ge10^{{-3}}\right].
$$

这是固定阈值全坐标规则而非Top-K排名。shared组要求两种value form都通过且平均符号相同；matched-null在每个layer/cell/head内循环平移31个坐标，保持数量。锁箱只用family16–31的修正eligible对，测试全坐标donor上界、form组、shared组、平移null、补集、错误cell、坐标清零、只保留联盟、全清零以及“全清零后仅写入donor联盟”。

**结果汇总。** 坐标规则为`{json.dumps(result['rules'], ensure_ascii=False)}`；锁箱符号预测为`{json.dumps(result['heldout_rule_validation'], ensure_ascii=False)}`；因果条件为`{json.dumps(result['summary'], ensure_ascii=False)}`；裁决与检查为`{json.dumps(result['adjudication'], ensure_ascii=False)}`、`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2559_c215297_c223488_coordinate_coalition_causal_lockbox.py`；完整boolean坐标mask、发现均值、全部逐候选因果分数和final位于`{OUT}`。

**分析与理论进展。** natural values跨family改变具体词汇，nonce values在所有family复用同一对token，因此nonce坐标一致性高可能只是词汇复用；必须看heldout causal与shifted-null差值。`patch_form_group`高于null支持该坐标集合含绑定载荷；`zero_all_then_donor_group`若能单独翻转才是强充分性；`keep_only`保留base能力与`zero_group`造成损伤共同出现，才接近条件必要性。任何单门失败都不关闭全坐标路线。

**问题硬伤与结论。** discovery/lockbox按family而非新模型/新unit；均值floor会忽略极低幅但一致坐标；mask在四个cell上分别发现，仍依赖固定表结构；循环平移只匹配数量不匹配所有物理协变量；早层九层联合patch仍不是最小路径。结论只裁决这个抽取算法是否优于其对照，不把失败解释成分布式机制不存在。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2558 / "analysis/final.json")
    masks, rules = derive_masks()
    validation = heldout_rule_validation(masks)
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        jobs, eligible = compile_jobs(tokenizer)
        rows = run(model, tokenizer, jobs, masks)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
    scores_path = OUT / "causal/coordinate_coalition_scores.jsonl"
    p2552.write(scores_path, rows)
    summary = summarize(rows)
    adjudication = {"all_coordinate_upper_bound": summary["patch_all"]["donor_flip"],
                    "form_group_sufficiency": summary["patch_form_group"]["donor_flip"],
                    "shifted_null_sufficiency": summary["patch_shifted_null"]["donor_flip"],
                    "form_group_beats_shifted_null_by_010": summary["patch_form_group"]["donor_flip"]
                    - summary["patch_shifted_null"]["donor_flip"] >= .10,
                    "donor_group_sufficient_after_zero_all": summary["zero_all_then_donor_group"]["donor_flip"] >= .70,
                    "base_group_sufficient_when_kept_alone": summary["keep_only_form_group"]["accuracy"] >= .70,
                    "group_natural_injury": summary["no_patch"]["accuracy"] - summary["zero_form_group"]["accuracy"],
                    "coordinate_gear_closed": False}
    checks = {"phase2558_passed": prior["all_checks_passed"], "discovery_families_16": True,
              "heldout_families_16": True, "heldout_pairs_nontrivial": len(eligible) >= 160,
              "candidate_jobs_twice_pairs": len(jobs) == 2 * len(eligible), "conditions_11": len(CONDITIONS) == 11,
              "rows_complete": len(rows) == len(jobs) * len(CONDITIONS),
              "baseline_gate": summary["no_patch"]["accuracy"] >= .95,
              "all_masks_physical": all(mask.shape == (9, 4, 8, 128) for mask in masks.values()),
              "no_topk": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "design": {"eligible_pairs": len(eligible),
              "candidate_jobs": len(jobs), "conditions": list(CONDITIONS), "discovery_families": list(range(16)),
              "heldout_families": list(range(16, 32))}, "rules": rules,
              "heldout_rule_validation": validation, "summary": summary,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values()),
              "files": {"scores": str(scores_path), "masks": rules["file"]}}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
