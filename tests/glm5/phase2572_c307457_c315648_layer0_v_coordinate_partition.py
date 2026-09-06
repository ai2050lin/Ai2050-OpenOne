#!/usr/bin/env python3
"""Exhaustive disjoint-coordinate partitions of the layer-0 V XOR intervention."""
from __future__ import annotations

import gc
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
P2571 = RESULT / "phase2571_c299265_c307456_new_entity_layer0_v_lockbox"
OUT = RESULT / "phase2572_c307457_c315648_layer0_v_coordinate_partition"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2572, "C307457-C315648"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2569_c284929_c291072_relation_value_xor_causal_interaction as p2569  # noqa: E402
import phase2571_c299265_c307456_new_entity_layer0_v_lockbox as p2571  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def specs(width: int, heads: int) -> dict[str, dict]:
    head_width = width // heads
    output: dict[str, dict] = {"no_patch": {"expected": "base"}}

    def add_triplet(prefix: str, coordinates: list[int]) -> None:
        output[f"{prefix}_relation"] = {"donor": "relation", "regions": ("query_relation",),
                                         "coordinates": coordinates, "expected": "flip"}
        output[f"{prefix}_value"] = {"donor": "value", "regions": ("query_value",),
                                      "coordinates": coordinates, "expected": "flip"}
        output[f"{prefix}_double"] = {"donor": "double", "regions": ("query_relation", "query_value"),
                                       "coordinates": coordinates, "expected": "base"}

    add_triplet("full", list(range(width)))
    output["full_null_relation_to_value"] = {"donor": "relation", "regions": ("query_value",),
        "coordinates": list(range(width)), "expected": "base"}
    output["full_null_value_to_relation"] = {"donor": "value", "regions": ("query_relation",),
        "coordinates": list(range(width)), "expected": "base"}
    for head in range(heads):
        own = list(range(head * head_width, (head + 1) * head_width))
        add_triplet(f"head{head}", own)
        add_triplet(f"leave_head{head}", [coordinate for coordinate in range(width) if coordinate not in set(own)])
    block_width = 32
    for block, start in enumerate(range(0, width, block_width)):
        add_triplet(f"block{block:02d}", list(range(start, min(start + block_width, width))))
    return output


class CoordinateController:
    def __init__(self, model):
        layer0 = model_utils.get_layers(model)[0]
        self.mode, self.label, self.spec, self.jobs = "none", "", {}, []
        self.store: dict[str, torch.Tensor] = {}
        self.handle = layer0.self_attn.v_proj.register_forward_hook(self._hook)

    def close(self) -> None:
        self.handle.remove()

    def _hook(self, _module, _inputs, output: torch.Tensor):
        if self.mode == "capture":
            self.store[self.label] = output.detach().clone()
            return None
        if self.mode != "patch":
            return None
        donor_label = self.spec["donor"]
        donor = self.store[donor_label].to(output.device)
        coordinates = torch.tensor(self.spec["coordinates"], dtype=torch.long, device=output.device)
        changed = output.clone()
        for batch_index, job in enumerate(self.jobs):
            for region in self.spec["regions"]:
                base_positions = job["regions"]["base"][region]
                donor_positions = job["regions"][donor_label][region]
                if len(base_positions) != len(donor_positions):
                    raise RuntimeError((region, len(base_positions), len(donor_positions)))
                for base_position, donor_position in zip(base_positions, donor_positions):
                    changed[batch_index, job["base_shift"] + base_position, coordinates] = donor[
                        batch_index, job[f"{donor_label}_shift"] + donor_position, coordinates]
        return changed


def run(model, tokenizer, jobs: list[dict], conditions: dict[str, dict]) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    controller, output = CoordinateController(model), []
    buckets: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for job in jobs:
        buckets[tuple(len(job[label]) for label in ("base", "relation", "value", "double"))].append(job)
    batches = [values[start:start + 4] for _, values in sorted(buckets.items())
               for start in range(0, len(values), 4)]
    done = 0
    try:
        for batch in batches:
            controller.jobs, controller.store = batch, {}
            for label in ("relation", "value", "double"):
                ids, mask, shifts = p2552.left_pad([job[label] for job in batch], tokenizer.pad_token_id, device)
                for job, shift in zip(batch, shifts):
                    job[f"{label}_shift"] = shift
                controller.mode, controller.label = "capture", label
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
            base_ids, base_mask, shifts = p2552.left_pad([job["base"] for job in batch], tokenizer.pad_token_id, device)
            for job, shift in zip(batch, shifts):
                job["base_shift"] = shift
            keep = max(len(job["continuation"]) for job in batch) + 1
            for condition, spec in conditions.items():
                controller.mode = "none" if condition == "no_patch" else "patch"
                controller.spec = spec
                with torch.inference_mode():
                    logits = model(input_ids=base_ids, attention_mask=base_mask,
                                   use_cache=False, logits_to_keep=keep).logits
                scores = p2569.continuation_scores(logits, batch, int(base_ids.shape[1]), "base")
                for job, score in zip(batch, scores):
                    row = {key: job[key] for key in ("case_id", "family_id", "depth", "relation_form",
                                                      "value_form", "binding", "candidate_index",
                                                      "target_index", "flip_target_index")}
                    row.update({"condition": condition, "expected": spec["expected"], "score": score})
                    output.append(row)
            done += len(batch)
            if done % 16 == 0 or done == len(jobs):
                print(f"[phase2572 coordinates] {done}/{len(jobs)}", flush=True)
    finally:
        controller.close()
    return output


def triple(summary: dict, prefix: str) -> dict:
    relation = summary[f"{prefix}_relation"]["flip_rate"]
    value = summary[f"{prefix}_value"]["flip_rate"]
    double = summary[f"{prefix}_double"]["base_accuracy"]
    core = min(relation, value, double)
    return {"relation_flip": relation, "value_flip": value, "double_base_preserve": double,
            "xor_core": core, "strong_gate": core >= .70}


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: layer0-V全1024坐标的穷举分区联盟图谱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Phase2571的新实体总体行为门未过（0.763672），但在28个四查询全对、token兼容四元组上冻结的layer0-V XOR局部充分性仍通过。本Phase不以Top-K寻找大值坐标，而把layer0 V投影的全部1024物理坐标作两个无遗漏分区：8个KV-head块（每块128坐标）和32个连续块（每块32坐标）。对每个head测single-head与leave-one-head-out；对每个32维块测single-block；每个分区都测试relation、value和double预言。另保留全1024坐标及错region null，共`{result['conditions']}`条件。

$$\mathcal V=\bigsqcup_{{h=0}}^7 H_h=\bigsqcup_{{b=0}}^{{31}}B_b,\quad
X(S)=\min(F_R(S),F_V(S),B_{{RV}}(S)).$$

**结果汇总。** 全坐标基准`{json.dumps(result['full'], ensure_ascii=False)}`；8个单head`{json.dumps(result['single_heads'], ensure_ascii=False)}`；8个leave-one-out`{json.dumps(result['leave_one_head_out'], ensure_ascii=False)}`；32块`{json.dumps(result['single_blocks'], ensure_ascii=False)}`；联盟摘要`{json.dumps(result['coalition_summary'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2572_c307457_c315648_layer0_v_coordinate_partition.py`；全部逐候选分区干预、坐标分区索引与final位于`{OUT}`。

**分析与理论进展。** single-head通过表示一个128维物理块已足够携带两个条件并按XOR合成；全部single失败而多数leave-one-out仍通过则说明冗余分布联盟；删除某head显著降低$X$只说明其条件必要性，不等于唯一语义head。32维无遗漏分区用于观察单head内部是否仍可缩小；所有坐标都进入某个块，低值坐标没有被预先丢弃。

**问题硬伤与结论。** 分区边界来自架构物理布局与连续坐标，不保证对应学习到的自然基元；同一28组用于确定分区效应，没有再留独立样本；只测试充分性替换；总体新实体行为未过80%门，因此结论严格限于行为合格子集。后续应冻结有效联盟，在全新关系词表/模板而非仅换实体上复验。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2571 / "analysis/final.json")
    material = read(P2571 / "material/rows.jsonl")
    behavior = read(P2571 / "behavior/scores.jsonl")
    selected = p2571.eligible_quartets(material, behavior)
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        jobs, compatible, excluded = p2569.prepare(material, selected, tokenizer, limit=96)
        width = int(model.config.num_key_value_heads * model.config.head_dim)
        conditions = specs(width, int(model.config.num_key_value_heads))
        rows = run(model, tokenizer, jobs, conditions)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    p2569.write(OUT / "causal/coordinate_partition_scores.jsonl", rows)
    summary = p2569.summarize(rows, conditions)
    full = triple(summary, "full")
    full_null = max(summary["full_null_relation_to_value"]["flip_rate"],
                    summary["full_null_value_to_relation"]["flip_rate"])
    full["matched_null_flip"] = full_null
    full["xor_margin"] = full["xor_core"] - full_null
    single_heads = {str(head): triple(summary, f"head{head}") for head in range(8)}
    leave_heads = {str(head): triple(summary, f"leave_head{head}") for head in range(8)}
    single_blocks = {str(block): triple(summary, f"block{block:02d}") for block in range(32)}
    coordinate_index = {"heads": {str(h): list(range(h * 128, (h + 1) * 128)) for h in range(8)},
                        "blocks": {str(b): list(range(b * 32, (b + 1) * 32)) for b in range(32)}}
    save(OUT / "material/coordinate_partitions.json", coordinate_index)
    coalition = {"single_head_strong": [h for h, value in single_heads.items() if value["strong_gate"]],
                 "single_block_strong": [b for b, value in single_blocks.items() if value["strong_gate"]],
                 "max_single_head_core": max(value["xor_core"] for value in single_heads.values()),
                 "max_single_block_core": max(value["xor_core"] for value in single_blocks.values()),
                 "leave_one_head_core_min": min(value["xor_core"] for value in leave_heads.values()),
                 "leave_one_head_core_max": max(value["xor_core"] for value in leave_heads.values()),
                 "most_conditionally_necessary_head": min(leave_heads,
                    key=lambda h: leave_heads[h]["xor_core"])}
    checks = {"prior_pipeline_complete": prior["all_checks_passed"],
              "prior_overall_behavior_gate_recorded_false": not prior["adjudication"]["overall_behavior_gate"],
              "compatible_28": len(compatible) == 28, "no_patch_at_least_95": summary["no_patch"]["base_accuracy"] >= .95,
              "all_1024_in_head_partition_once": sorted(sum(coordinate_index["heads"].values(), [])) == list(range(1024)),
              "all_1024_in_block_partition_once": sorted(sum(coordinate_index["blocks"].values(), [])) == list(range(1024)),
              "two_candidates_each": len(rows) == len(compatible) * 2 * len(conditions),
              "no_topk_primary": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "compatible_quartets": len(compatible),
              "excluded_token_mismatch": excluded, "conditions": len(conditions), "full": full,
              "single_heads": single_heads, "leave_one_head_out": leave_heads,
              "single_blocks": single_blocks, "coalition_summary": coalition,
              "checks": checks, "all_checks_passed": all(checks.values()),
              "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({"phase": PHASE, "full": full, "single_heads": single_heads,
                      "leave_one_head_out": leave_heads, "coalition_summary": coalition,
                      "checks": checks, "all_checks_passed": result["all_checks_passed"]},
                     ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
