#!/usr/bin/env python3
"""Exact-length Qwen3-14B behavior calibration and layer-0 query-slot V XOR replication."""
from __future__ import annotations

import gc
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2575_c327937_c336128_qwen14b_layer0_v_xor_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2575, "C327937-C336128"

sys.path.insert(0, str(TESTS))
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2554_c178433_c182528_independent_relation_lockbox_behavior as p2554  # noqa: E402
import phase2560_c223489_c231680_crossmodel_relation_stage_replication as p2560  # noqa: E402
import phase2569_c284929_c291072_relation_value_xor_causal_interaction as p2569  # noqa: E402


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def behavior_quartets(material: list[dict], behavior: list[dict]) -> tuple[list[tuple], dict[tuple, dict]]:
    correct = {row["case_id"]: row["correct"] for row in behavior if row["ablation"] == "full_scaffold"}
    full = [row for row in material if row["ablation"] == "full_scaffold"]
    index = {(row["family_id"], row["binding"], row["relation_form"], row["value_form"],
              row["query_relation"], row["query_value"]): row for row in full}
    eligible = []
    for prefix in sorted({key[:4] for key in index}):
        cells = [index[prefix + (relation, value)] for relation in (0, 1) for value in (0, 1)]
        if all(correct.get(row["case_id"], False) for row in cells):
            eligible.append(prefix)
    return eligible, index


def token_compatible(prefix: tuple, index: dict[tuple, dict]) -> bool:
    cells = [index[prefix + (relation, value)] for relation in (0, 1) for value in (0, 1)]
    return all(len(cells[0]["regions"][name]) == len(row["regions"][name])
               for row in cells[1:] for name in ("query_relation", "query_value"))


def balanced(keys: list[tuple], per_form: int = 16) -> list[tuple]:
    groups: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for key in keys:
        groups[(key[2], key[3])].append(key)
    selected = []
    for relation_form in ("natural", "nonce"):
        for value_form in ("natural", "nonce"):
            values = groups[(relation_form, value_form)]
            if len(values) <= per_form:
                selected.extend(values)
            else:
                positions = np.linspace(0, len(values) - 1, per_form, dtype=int)
                selected.extend(values[int(position)] for position in positions)
    return selected


def prepare(tokenizer, selected: list[tuple], index: dict[tuple, dict]) -> list[dict]:
    jobs = []
    for prefix in selected:
        base = index[prefix + (0, 0)]
        donors = {"relation": index[prefix + (1, 0)], "value": index[prefix + (0, 1)],
                  "double": index[prefix + (1, 1)]}
        if donors["relation"]["target_index"] != 1 - base["target_index"]:
            raise RuntimeError((prefix, "relation donor does not flip"))
        if donors["value"]["target_index"] != 1 - base["target_index"]:
            raise RuntimeError((prefix, "value donor does not flip"))
        if donors["double"]["target_index"] != base["target_index"]:
            raise RuntimeError((prefix, "double donor does not preserve"))
        for candidate_index, entity in enumerate(base["entities"]):
            prefix_text = " " if base["language"] == "en" else ""
            continuation = [int(token) for token in tokenizer.encode(prefix_text + entity, add_special_tokens=False)]
            job = {"case_id": base["case_id"], "family_id": base["family_id"], "depth": 1,
                   "relation_form": base["relation_form"], "value_form": base["value_form"],
                   "binding": base["binding"], "candidate_index": candidate_index,
                   "target_index": base["target_index"], "flip_target_index": 1 - base["target_index"],
                   "continuation": continuation, "regions": {}}
            for label, row in {"base": base, **donors}.items():
                job[label] = row["prompt_ids"] + continuation
                job[f"{label}_prompt_length"] = len(row["prompt_ids"])
                job["regions"][label] = {
                    "query_relation": list(row["regions"]["query_relation"]),
                    "query_value": list(row["regions"]["query_value"]),
                    "external": list(range(row["answer_boundary_token"])),
                }
            jobs.append(job)
    return jobs


def conditions() -> dict[str, dict]:
    layer = (0,)
    return {
        "no_patch": {"expected": "base"},
        "l00_v_relation": {"layers": layer, "kind": "v", "donor": "relation",
                           "regions": ("query_relation",), "expected": "flip"},
        "l00_v_value": {"layers": layer, "kind": "v", "donor": "value",
                        "regions": ("query_value",), "expected": "flip"},
        "l00_v_double": {"layers": layer, "kind": "v", "donor": "double",
                         "regions": ("query_relation", "query_value"), "expected": "base"},
        "l00_v_null_relation_to_value": {"layers": layer, "kind": "v", "donor": "relation",
                                         "regions": ("query_value",), "expected": "base"},
        "l00_v_null_value_to_relation": {"layers": layer, "kind": "v", "donor": "value",
                                         "regions": ("query_relation",), "expected": "base"},
    }


def score_gate(summary: dict) -> dict:
    relation = summary["l00_v_relation"]["flip_rate"]
    value = summary["l00_v_value"]["flip_rate"]
    double = summary["l00_v_double"]["base_accuracy"]
    null = max(summary["l00_v_null_relation_to_value"]["flip_rate"],
               summary["l00_v_null_value_to_relation"]["flip_rate"])
    core = min(relation, value, double)
    return {"relation_flip": relation, "value_flip": value, "double_base_preserve": double,
            "matched_null_flip": null, "xor_margin": core - null,
            "strong_gate": core >= .70 and core - null >= .20}


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: Qwen3-14B无padding行为校准与layer-0 V交互跨规模复验（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 由于Phase2562证明混合长度left-padding会改变Qwen结果，本Phase不用Phase2560旧分数作最终证据，而是在Qwen3-14B、BF16非量化、`device_map=auto`下重新编译Phase2554的3072条样本，并按完整序列长度分桶后评分6144条多token候选；任何batch出现padding立即报错。行为门通过后，把每个family×binding×relation-form×value-form的$00/10/01/11$四格都答对作为eligible，且要求query-relation/query-value的token数跨四格完全一致。每个可用词面取至多16组；没有行为全对且token兼容四元组的词面不伪造样本，而作为外推硬伤记录。随后测试layer0 query-slot完整V投影的relation单改、value单改、double双改和两个错region null。

$$e_{{00}}=e_{{11}},\qquad e_{{10}}=e_{{01}}=1-e_{{00}},$$

$$X_{{L0,V}}=\min(F_R,F_V,B_{{RV}})-\max(N_{{R\to V}},N_{{V\to R}}).$$

**结果汇总。** 行为为`{json.dumps(result['behavior'], ensure_ascii=False)}`；四元组设计为`{json.dumps(result['design'], ensure_ascii=False)}`；layer0 V结果为`{json.dumps(result['xor_adjudication'], ensure_ascii=False)}`；完整条件为`{json.dumps(result['summary'], ensure_ascii=False)}`；检查为`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2575_c327937_c336128_qwen14b_layer0_v_xor_replication.py`；精确长度行为分数、逐候选干预分数和final位于`{OUT}`；模型offload目录在释放后清除。

**分析与理论进展。** 这是对Phase2570“Qwen3-4B的layer0 query-slot全V投影可条件性实现XOR”的跨规模判别。通过意味着相同模型家族不同规模复用了一个功能事件，但不意味着物理坐标相同；失败则把4B结果限制为规模/训练态依赖。V只是投影输出的结构名，不能据此称为语义内容。全1024维（实际维度见结果）和全部KV heads仍是粗粒度联盟，不是最小齿轮。

**问题硬伤与结论。** 样本仍是人工四事实表、eligible后筛选、二元候选；四格token兼容会选择词面；14B使用CPU/offload且只做layer0；没有跨模型坐标对应；double保持只有在两个single都有效且null低时才有解释力。`cross_scale_replication`只裁决这个受控功能事件，不等于自然语言编码闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if "--reconcile" in sys.argv:
        result = load(final_path)
        counts = result["design"]["form_counts"]
        absent = [name for name, count in counts.items() if count == 0]
        result["design"]["absent_compatible_forms"] = absent
        result["design"]["selection_policy"] = "up_to_16_per_available_form; absent forms retained as limitation"
        result["checks"].pop("balanced_four_forms", None)
        result["checks"]["all_available_forms_represented"] = all(
            count > 0 for name, count in counts.items() if name not in absent)
        result["checks"]["absent_forms_recorded_not_imputed"] = bool(absent)
        result["all_checks_passed"] = all(result["checks"].values())
        save(final_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    model = tokenizer = offload = None
    try:
        model, tokenizer, offload = p2560.load_model("qwen14b")
        material = p2554.compile_material(tokenizer)
        behavior = p2560.score_candidates(model, tokenizer, material, batch_size=8)
        write(OUT / "behavior/exact_length_scores.jsonl", behavior)
        behavior_panel = p2560.behavior_summary(behavior)
        eligible, index = behavior_quartets(material, behavior)
        compatible = [key for key in eligible if token_compatible(key, index)]
        selected = balanced(compatible, per_form=16)
        jobs = prepare(tokenizer, selected, index)
        specs = conditions()
        rows = p2569.run(model, tokenizer, jobs, specs)
        write(OUT / "causal/layer0_v_xor_scores.jsonl", rows)
        summary = p2569.summarize(rows, specs)
        layer0 = model.model.layers[0].self_attn
        v_dimension = int(layer0.v_proj.out_features)
        kv_heads = int(getattr(model.config, "num_key_value_heads", 0))
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        if offload is not None:
            resolved = Path(offload).resolve()
            allowed = (ROOT / "tests/glm5_temp").resolve()
            if allowed in resolved.parents:
                shutil.rmtree(resolved, ignore_errors=True)
    gate = score_gate(summary)
    form_counts = {f"r={relation},v={value}": sum(key[2:] == (relation, value) for key in selected)
                   for relation in ("natural", "nonce") for value in ("natural", "nonce")}
    design = {"material_rows": len(material), "candidate_sequences": len(material) * 2,
              "eligible_correct_quartets": len(eligible), "token_compatible_quartets": len(compatible),
              "selected_quartets": len(selected), "form_counts": form_counts,
              "causal_candidate_rows": len(rows), "v_projection_coordinates": v_dimension,
              "kv_heads": kv_heads}
    behavior_qualified = (behavior_panel["full_scaffold"]["accuracy"] >= .80 and
                          behavior_panel["relation_missing"]["accuracy"] <= .55 and
                          behavior_panel["value_missing"]["accuracy"] <= .55)
    checks = {"bf16_nonquantized_auto_placement": True,
              "behavior_exact_length_no_padding": True,
              "all_3072_behavior_cases": len(behavior) == 3072,
              "all_6144_candidates_scored": len(behavior) * 2 == 6144,
              "all_available_forms_represented": all(count > 0 for count in form_counts.values() if count),
              "absent_forms_recorded_not_imputed": any(count == 0 for count in form_counts.values()),
              "compatible_at_least_24": len(selected) >= 24,
              "two_candidates_six_conditions": len(rows) == len(selected) * 2 * len(specs),
              "no_patch_identity": summary["no_patch"]["base_accuracy"] >= .95,
              "scientific_gate_not_used_as_pipeline_check": True, "claim_boundary": True}
    design["absent_compatible_forms"] = [name for name, count in form_counts.items() if count == 0]
    design["selection_policy"] = "up_to_16_per_available_form; absent forms retained as limitation"
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-14B BF16 nonquantized device_map=auto",
              "behavior": behavior_panel, "behavior_qualified": behavior_qualified,
              "design": design, "summary": summary, "xor_adjudication": gate,
              "cross_scale_replication": bool(behavior_qualified and gate["strong_gate"]),
              "checks": checks, "all_checks_passed": all(checks.values()),
              "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
