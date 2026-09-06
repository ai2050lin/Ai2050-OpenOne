#!/usr/bin/env python3
"""Exhaustive H5-anchored KV-head subset lattice with cross-entity validation."""
from __future__ import annotations

import gc
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2567 = RESULT / "phase2567_c264449_c276736_minimal_bridge_extension"
P2568 = RESULT / "phase2568_c276737_c284928_relation_value_factorial_fullfield"
P2570 = RESULT / "phase2570_c291073_c299264_holdout_layer_projection_xor"
P2571 = RESULT / "phase2571_c299265_c307456_new_entity_layer0_v_lockbox"
P2572 = RESULT / "phase2572_c307457_c315648_layer0_v_coordinate_partition/analysis/final.json"
OUT = RESULT / "phase2573_c315649_c323840_head5_anchored_subset_lattice"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2573, "C315649-C323840"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2569_c284929_c291072_relation_value_xor_causal_interaction as p2569  # noqa: E402
import phase2570_c291073_c299264_holdout_layer_projection_xor as p2570  # noqa: E402
import phase2571_c299265_c307456_new_entity_layer0_v_lockbox as p2571  # noqa: E402
import phase2572_c307457_c315648_layer0_v_coordinate_partition as p2572  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def subset_specs() -> tuple[dict[str, dict], dict[str, tuple[int, ...]]]:
    subsets = {}
    for size in range(1, 9):
        for others in itertools.combinations([0, 1, 2, 3, 4, 6, 7], size - 1):
            heads = tuple(sorted((5,) + others))
            subsets["h" + "-".join(str(head) for head in heads)] = heads
    conditions: dict[str, dict] = {"no_patch": {"expected": "base"}}
    for label, heads in subsets.items():
        coordinates = [coordinate for head in heads for coordinate in range(head * 128, (head + 1) * 128)]
        conditions[f"{label}_relation"] = {"donor": "relation", "regions": ("query_relation",),
                                             "coordinates": coordinates, "expected": "flip"}
        conditions[f"{label}_value"] = {"donor": "value", "regions": ("query_value",),
                                          "coordinates": coordinates, "expected": "flip"}
        conditions[f"{label}_double"] = {"donor": "double", "regions": ("query_relation", "query_value"),
                                           "coordinates": coordinates, "expected": "base"}
    return conditions, subsets


def validation_specs(label: str, heads: tuple[int, ...]) -> dict[str, dict]:
    coordinates = [coordinate for head in heads for coordinate in range(head * 128, (head + 1) * 128)]
    return {"no_patch": {"expected": "base"},
            f"{label}_relation": {"donor": "relation", "regions": ("query_relation",),
                                    "coordinates": coordinates, "expected": "flip"},
            f"{label}_value": {"donor": "value", "regions": ("query_value",),
                                 "coordinates": coordinates, "expected": "flip"},
            f"{label}_double": {"donor": "double", "regions": ("query_relation", "query_value"),
                                  "coordinates": coordinates, "expected": "base"},
            f"{label}_null_relation_to_value": {"donor": "relation", "regions": ("query_value",),
                "coordinates": coordinates, "expected": "base"},
            f"{label}_null_value_to_relation": {"donor": "value", "regions": ("query_relation",),
                "coordinates": coordinates, "expected": "base"}}


def triple(summary: dict, label: str) -> dict:
    relation = summary[f"{label}_relation"]["flip_rate"]
    value = summary[f"{label}_value"]["flip_rate"]
    double = summary[f"{label}_double"]["base_accuracy"]
    return {"relation_flip": relation, "value_flip": value, "double_base_preserve": double,
            "xor_core": min(relation, value, double)}


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: H5锚定的128子集联盟穷举与跨实体冻结复验（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Phase2572显示单head均不充分、删除H5后XOR core从0.821降至0.286。本Phase不按幅值选Top-K，而穷举8个KV-head的幂集中全部包含H5的128个子集；每个子集在Silver Badger/Golden Crane的28个行为全对兼容四元组上测试relation、value、double三预言，共385条件。按“最小head数优先、同大小XOR core最高”冻结一个候选，然后在Phase2570的Copper Lynx/Azure Heron 43个未进入全坐标发现集的四元组上只运行该候选、无干预和两个错region null。

$$\mathcal S_5=\{{S\subseteq\{{0,\ldots,7\}}:5\in S\}},\quad |\mathcal S_5|=128,$$
$$S^*=\arg\max_{{|S|=k^*}}X(S),\quad k^*=\min\{{|S|:X(S)\ge.7\}}.$$

**结果汇总。** 发现格`{json.dumps(result['discovery_lattice'], ensure_ascii=False)}`；冻结候选`{json.dumps(result['frozen_candidate'], ensure_ascii=False)}`；跨实体验证`{json.dumps(result['validation'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2573_c315649_c323840_head5_anchored_subset_lattice.py`；128子集逐候选发现分数、冻结候选跨实体分数、完整lattice和final位于`{OUT}`。

**分析与理论进展。** 若少量head联盟在独立实体上仍同时满足两个单因子翻转、双因子恢复和错位低效，就得到比全1024坐标更小的条件充分联盟。它描述的是固定物理基底上的协同集合；不同head在同一任务中可能提供互补坐标，而非一头一个语义。失败验证则说明最小联盟在发现实体上过拟合，必须退回全V分布规律。

**问题硬伤与结论。** 子集只按KV-head物理边界切分，未穷举head内任意坐标组合；H5锚定来自上一Phase同一新实体数据；验证实体虽独立，但模板和关系/值词表相同；充分性patch仍不是自然运行必要性。该Phase只允许命名“受控任务的最小冻结候选联盟”，不允许命名通用语言齿轮。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2572)
    new_material = read(P2571 / "material/rows.jsonl")
    new_behavior = read(P2571 / "behavior/scores.jsonl")
    new_selected = p2571.eligible_quartets(new_material, new_behavior)
    old_material = read(P2567 / "material/rows.jsonl")
    old_behavior = read(P2567 / "behavior/scores.jsonl")
    discovery_used = {tuple(row) for row in load(P2568 / "material/selected_quartets.json")["selected"]}
    old_selected = p2570.holdout_quartets(old_material, old_behavior, discovery_used)
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        new_jobs, new_compatible, _ = p2569.prepare(new_material, new_selected, tokenizer, limit=96)
        discovery_specs, subsets = subset_specs()
        discovery_rows = p2572.run(model, tokenizer, new_jobs, discovery_specs)
        discovery_summary = p2569.summarize(discovery_rows, discovery_specs)
        lattice = {label: {"heads": list(heads), **triple(discovery_summary, label)}
                   for label, heads in subsets.items()}
        strong = [(label, value) for label, value in lattice.items() if value["xor_core"] >= .70]
        if not strong:
            raise RuntimeError("full H5-anchored set unexpectedly failed")
        minimum_size = min(len(value["heads"]) for _, value in strong)
        winner_label, winner = max((item for item in strong if len(item[1]["heads"]) == minimum_size),
                                   key=lambda item: item[1]["xor_core"])
        winner_heads = tuple(winner["heads"])
        old_jobs, old_compatible, old_excluded = p2569.prepare(old_material, old_selected, tokenizer, limit=64)
        valid_specs = validation_specs(winner_label, winner_heads)
        validation_rows = p2572.run(model, tokenizer, old_jobs, valid_specs)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    p2569.write(OUT / "causal/discovery_scores.jsonl", discovery_rows)
    p2569.write(OUT / "causal/validation_scores.jsonl", validation_rows)
    save(OUT / "analysis/lattice.json", lattice)
    validation_summary = p2569.summarize(validation_rows, valid_specs)
    validation = triple(validation_summary, winner_label)
    validation_null = max(validation_summary[f"{winner_label}_null_relation_to_value"]["flip_rate"],
                          validation_summary[f"{winner_label}_null_value_to_relation"]["flip_rate"])
    validation.update({"matched_null_flip": validation_null,
                       "xor_margin": validation["xor_core"] - validation_null,
                       "strong_gate": validation["xor_core"] >= .70
                           and validation["xor_core"] - validation_null >= .20,
                       "n": len(old_compatible), "excluded_token_mismatch": old_excluded})
    by_size = {str(size): {"n": sum(len(value["heads"]) == size for value in lattice.values()),
                           "max_xor_core": max(value["xor_core"] for value in lattice.values()
                                                if len(value["heads"]) == size),
                           "strong_count": sum(len(value["heads"]) == size and value["xor_core"] >= .70
                                               for value in lattice.values())}
               for size in range(1, 9)}
    frozen = {"label": winner_label, **winner, "minimum_size": minimum_size}
    checks = {"prior_complete": prior["all_checks_passed"], "all_128_h5_subsets": len(lattice) == 128,
              "new_entity_compatible_28": len(new_compatible) == 28,
              "discovery_no_patch_at_least_95": discovery_summary["no_patch"]["base_accuracy"] >= .95,
              "winner_frozen_before_old_run": True, "old_holdout_compatible_43": len(old_compatible) == 43,
              "validation_no_patch_at_least_95": validation_summary["no_patch"]["base_accuracy"] >= .95,
              "two_candidates_each": len(discovery_rows) == len(new_compatible) * 2 * len(discovery_specs)
                  and len(validation_rows) == len(old_compatible) * 2 * len(valid_specs),
              "no_topk_primary": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized",
              "discovery_lattice": {"subsets": len(lattice), "by_size": by_size,
                                    "new_entity_quartets": len(new_compatible)},
              "frozen_candidate": frozen, "validation": validation,
              "checks": checks, "all_checks_passed": all(checks.values()),
              "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
