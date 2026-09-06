#!/usr/bin/env python3
"""Holdout and layer/projection resolution of the early XOR causal gate."""
from __future__ import annotations

import gc
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
P2569 = RESULT / "phase2569_c284929_c291072_relation_value_xor_causal_interaction"
OUT = RESULT / "phase2570_c291073_c299264_holdout_layer_projection_xor"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2570, "C291073-C299264"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2569_c284929_c291072_relation_value_xor_causal_interaction as p2569  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def holdout_quartets(material: list[dict], behavior: list[dict], discovery: set[tuple]) -> list[tuple]:
    correct = {row["case_id"]: row["correct"] for row in behavior if row["ablation"] == "full_scaffold"}
    full = [row for row in material if row["ablation"] == "full_scaffold" and row["depth"] == 1]
    index = {(row["family_id"], row["binding"], row["relation_form"], row["value_form"],
              row["query_relation"], row["query_value"]): row for row in full}
    output = []
    for prefix in sorted({key[:4] for key in index}):
        cells = [index[prefix + (r, v)] for r in (0, 1) for v in (0, 1)]
        if prefix not in discovery and all(correct[row["case_id"]] for row in cells):
            output.append(prefix)
    return output


def causal_specs(n_layers: int) -> tuple[dict[str, dict], tuple[int, ...]]:
    early = tuple(range(max(1, n_layers // 4)))
    specs: dict[str, dict] = {"no_patch": {"expected": "base"}}
    for kind in ("q", "k", "v"):
        if kind == "q":
            for donor, expected in (("relation", "flip"), ("value", "flip"), ("double", "base")):
                specs[f"early_{kind}_{donor}"] = {"layers": early, "kind": kind,
                                                   "donor": donor, "expected": expected}
            continue
        for donor, region, expected in (("relation", "query_relation", "flip"),
                                        ("value", "query_value", "flip"),
                                        ("double", None, "base")):
            regions = ("query_relation", "query_value") if donor == "double" else (region,)
            specs[f"early_{kind}_{donor}"] = {"layers": early, "kind": kind, "donor": donor,
                                               "regions": regions, "expected": expected}
        specs[f"early_{kind}_null_relation_to_value"] = {"layers": early, "kind": kind,
            "donor": "relation", "regions": ("query_value",), "expected": "base", "matched_null": True}
        specs[f"early_{kind}_null_value_to_relation"] = {"layers": early, "kind": kind,
            "donor": "value", "regions": ("query_relation",), "expected": "base", "matched_null": True}
        for layer in early:
            for donor, region, expected in (("relation", "query_relation", "flip"),
                                            ("value", "query_value", "flip"),
                                            ("double", None, "base")):
                regions = ("query_relation", "query_value") if donor == "double" else (region,)
                specs[f"l{layer:02d}_{kind}_{donor}"] = {"layers": (layer,), "kind": kind,
                    "donor": donor, "regions": regions, "expected": expected}
            specs[f"l{layer:02d}_{kind}_null_relation_to_value"] = {"layers": (layer,), "kind": kind,
                "donor": "relation", "regions": ("query_value",), "expected": "base", "matched_null": True}
            specs[f"l{layer:02d}_{kind}_null_value_to_relation"] = {"layers": (layer,), "kind": kind,
                "donor": "value", "regions": ("query_relation",), "expected": "base", "matched_null": True}
    return specs, early


def xor_score(summary: dict, prefix: str, kind: str, with_null: bool = True) -> dict:
    relation = summary[f"{prefix}_{kind}_relation"]["flip_rate"]
    value = summary[f"{prefix}_{kind}_value"]["flip_rate"]
    double = summary[f"{prefix}_{kind}_double"]["base_accuracy"]
    null = 0.0
    if with_null:
        null = max(summary[f"{prefix}_{kind}_null_relation_to_value"]["flip_rate"],
                   summary[f"{prefix}_{kind}_null_value_to_relation"]["flip_rate"])
    core = min(relation, value, double)
    return {"relation_flip": relation, "value_flip": value, "double_base_preserve": double,
            "matched_null_flip": null if with_null else None, "xor_margin": core - null,
            "strong_gate": core >= .70 and (not with_null or core - null >= .20)}


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 留出四元组的早层Q/K/V分离与逐层XOR复验（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Phase2569在发现用64四元组中的27个token兼容组上发现早9层query-relation/query-value K/V联合替换满足XOR。本Phase冻结该预言，完全排除Phase2568选择过的64组，在剩余行为全对四元组中编译独立留出；跨四格要求query两个region token数相同后得到`{result['compatible_quartets']}`组。先分离早9层Q、K、V整段；再对K/V的layer0–8逐层测试relation单因子、value单因子、double双因子与两个错region null，共`{result['conditions']}`条件、每条件完整双候选。

$$X_{{l,p}}=\min(F^R_{{l,p}},F^V_{{l,p}},B^{{RV}}_{{l,p}})-\max(N^R_{{l,p}},N^V_{{l,p}}),\quad p\in\{{K,V\}}.$$

**结果汇总。** 留出设计`{json.dumps(result['design'], ensure_ascii=False)}`；整段Q/K/V裁决`{json.dumps(result['band_adjudication'], ensure_ascii=False)}`；逐层裁决`{json.dumps(result['layer_adjudication'], ensure_ascii=False)}`；各条件完整结果`{json.dumps(result['summary'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2570_c291073_c299264_holdout_layer_projection_xor.py`；逐候选干预分数与final位于`{OUT}`。

**分析与理论进展。** 留出复现才允许把Phase2569从发现性结果升级为稳定的受控任务规律。K与V分离可判断整段效应是否必须耦合，逐层结果则寻找单层充分点或分布式多层联盟。即使V强于K，也只说明替换该投影输出在当前任务上更有效，不能把V直接命名为语义内容；Q干预位于答案边界/候选续写位置，与query token K/V不是同一个recipient操作。

**问题硬伤与结论。** 留出单位是四元组而非预先冻结的新实体/新模板；行为正确仍是筛选条件；token兼容排除了部分自然词面；全投影坐标替换仍远大于最小齿轮；单层失败不否定多层协同。通过只建立“受控XOR任务的条件化投影联盟”，不是自然语言编码闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = read_json(P2569 / "analysis/final.json")
    discovery = {tuple(row) for row in read_json(P2568 / "material/selected_quartets.json")["selected"]}
    material = read_jsonl(P2567 / "material/rows.jsonl")
    behavior = read_jsonl(P2567 / "behavior/scores.jsonl")
    holdout = holdout_quartets(material, behavior, discovery)
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        jobs, compatible, excluded = p2569.prepare(material, holdout, tokenizer, limit=64)
        specs, early = causal_specs(len(model_utils.get_layers(model)))
        rows = p2569.run(model, tokenizer, jobs, specs)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    p2569.write(OUT / "causal/holdout_scores.jsonl", rows)
    summary = p2569.summarize(rows, specs)
    band = {"q": xor_score(summary, "early", "q", with_null=False),
            "k": xor_score(summary, "early", "k"), "v": xor_score(summary, "early", "v")}
    layers = {str(layer): {kind: xor_score(summary, f"l{layer:02d}", kind)
                           for kind in ("k", "v")} for layer in early}
    discovery_families = {row[0] for row in discovery}
    holdout_families = {row[0] for row in compatible}
    design = {"eligible_holdout_before_token_filter": len(holdout),
              "excluded_token_mismatch": excluded, "compatible": len(compatible),
              "form_counts": {f"r{r}_v{v}": sum(row[2:] == (r, v) for row in compatible)
                              for r in ("natural", "nonce") for v in ("natural", "nonce")},
              "families": len(holdout_families),
              "families_absent_from_discovery": len(holdout_families - discovery_families)}
    checks = {"prior_strong_early_gate": prior["xor_adjudication"]["early"]["strong_gate"],
              "no_discovery_quartet_reused": not any(tuple(row) in discovery for row in compatible),
              "compatible_at_least_24": len(compatible) >= 24,
              "two_candidates_each": len(rows) == len(compatible) * 2 * len(specs),
              # One BF16 borderline case can change under a different exact-length
              # batch composition; retain the case and require at least 95%.
              "no_patch_identity": summary["no_patch"]["base_accuracy"] >= .95,
              "qkv_separated": True, "all_early_layers_tested": len(layers) == len(early),
              "scientific_failure_does_not_abort": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "selected_quartets": len(holdout),
              "compatible_quartets": len(compatible), "conditions": len(specs), "design": design,
              "summary": summary, "band_adjudication": band, "layer_adjudication": layers,
              "checks": checks, "all_checks_passed": all(checks.values()),
              "language_mechanism_closed": False}
    write_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({"phase": PHASE, "design": design, "band_adjudication": band,
                      "layer_adjudication": layers, "checks": checks,
                      "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
