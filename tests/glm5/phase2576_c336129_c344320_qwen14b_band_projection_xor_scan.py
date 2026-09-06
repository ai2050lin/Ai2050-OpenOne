#!/usr/bin/env python3
"""Scan Qwen3-14B relative layer bands for query-slot V/KV XOR sufficiency."""
from __future__ import annotations

import gc
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2575 = RESULT / "phase2575_c327937_c336128_qwen14b_layer0_v_xor_replication"
OUT = RESULT / "phase2576_c336129_c344320_qwen14b_band_projection_xor_scan"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2576, "C336129-C344320"

sys.path.insert(0, str(TESTS))
import phase2554_c178433_c182528_independent_relation_lockbox_behavior as p2554  # noqa: E402
import phase2560_c223489_c231680_crossmodel_relation_stage_replication as p2560  # noqa: E402
import phase2569_c284929_c291072_relation_value_xor_causal_interaction as p2569  # noqa: E402
import phase2575_c327937_c336128_qwen14b_layer0_v_xor_replication as p2575  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def conditions(n_layers: int) -> tuple[dict[str, dict], dict[str, tuple[int, ...]]]:
    names = ("early", "middle", "middlelate", "late")
    band_map = dict(zip(names, p2560.bands(n_layers)))
    output: dict[str, dict] = {"no_patch": {"expected": "base"}}
    for band, layers in band_map.items():
        for projection in ("v", "kv"):
            stem = f"{band}_{projection}"
            output[f"{stem}_relation"] = {"layers": layers, "kind": projection, "donor": "relation",
                                                    "regions": ("query_relation",), "expected": "flip"}
            output[f"{stem}_value"] = {"layers": layers, "kind": projection, "donor": "value",
                                                 "regions": ("query_value",), "expected": "flip"}
            output[f"{stem}_double"] = {"layers": layers, "kind": projection, "donor": "double",
                                                  "regions": ("query_relation", "query_value"), "expected": "base"}
            output[f"{stem}_null_relation_to_value"] = {
                "layers": layers, "kind": projection, "donor": "relation",
                "regions": ("query_value",), "expected": "base"}
            output[f"{stem}_null_value_to_relation"] = {
                "layers": layers, "kind": projection, "donor": "value",
                "regions": ("query_relation",), "expected": "base"}
    return output, band_map


def gate(summary: dict, stem: str) -> dict:
    relation = summary[f"{stem}_relation"]["flip_rate"]
    value = summary[f"{stem}_value"]["flip_rate"]
    double = summary[f"{stem}_double"]["base_accuracy"]
    null = max(summary[f"{stem}_null_relation_to_value"]["flip_rate"],
               summary[f"{stem}_null_value_to_relation"]["flip_rate"])
    core = min(relation, value, double)
    return {"relation_flip": relation, "value_flip": value, "double_base_preserve": double,
            "matched_null_flip": null, "xor_core": core, "xor_margin": core - null,
            "strong_gate": core >= .70 and core - null >= .20}


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: Qwen3-14B相对层段V/KV关系×值交互扫描（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Phase2575证明Qwen3-14B行为门通过，但Qwen3-4B发现的layer0全V XOR没有跨规模复现。本Phase不把单层失败解释为路线消失，而冻结Phase2575同一批`{result['selected_quartets']}`个行为全对、token兼容四元组，在14B的40层内只按相对四分位定义early/middle/middlelate/late；每段分别替换query-relation与query-value位置的全1024维V或K+V。每个投影测试relation单改、value单改、double双改、两个错region null，完整双候选评分。

$$B_k=\left[\operatorname{{round}}\frac{{kL}}4,\operatorname{{round}}\frac{{(k+1)L}}4\right),$$

$$X_{{B,p}}=\min(F^R_{{B,p}},F^V_{{B,p}},B^{{RV}}_{{B,p}})-\max(N^R_{{B,p}},N^V_{{B,p}}),\ p\in\{{V,KV\}}.$$

**结果汇总。** 层段为`{json.dumps(result['bands'], ensure_ascii=False)}`；V/KV裁决为`{json.dumps(result['adjudication'], ensure_ascii=False)}`；全部条件为`{json.dumps(result['summary'], ensure_ascii=False)}`；检查为`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2576_c336129_c344320_qwen14b_band_projection_xor_scan.py`；逐候选分数和final位于`{OUT}`；复用Phase2575精确长度行为锁箱，不重复选择样本。

**分析与理论进展。** 若某个较晚层段满足两个single翻转、double恢复且null低，说明4B的layer0局部充分性在14B上迁移为分布式/后移阶段；若V失败而KV通过，只能说K与V投影联合替换充分，不能把K命名为寻址或V命名为内容。若所有层段失败，则受控XOR的完整query-slot投影事件也不具跨规模稳定性。

**问题硬伤与结论。** 四分位边界不是学习得到的阶段；整段同时替换10层可能有过量和抵消；32组缺失nonce×nonce词面；同一批样本用于所有段比较；只测Qwen家族和人工二元表格。层段通过也不是最小坐标齿轮，失败也不排除attention/MLP/跨层混合路径。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2575 / "analysis/final.json")
    behavior = read(P2575 / "behavior/exact_length_scores.jsonl")
    model = tokenizer = offload = None
    try:
        model, tokenizer, offload = p2560.load_model("qwen14b")
        material = p2554.compile_material(tokenizer)
        eligible, index = p2575.behavior_quartets(material, behavior)
        compatible = [key for key in eligible if p2575.token_compatible(key, index)]
        selected = p2575.balanced(compatible, per_form=16)
        jobs = p2575.prepare(tokenizer, selected, index)
        specs, band_map = conditions(len(model.model.layers))
        rows = p2569.run(model, tokenizer, jobs, specs)
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
    write(OUT / "causal/band_projection_scores.jsonl", rows)
    summary = p2569.summarize(rows, specs)
    adjudication = {band: {projection: gate(summary, f"{band}_{projection}")
                            for projection in ("v", "kv")} for band in band_map}
    strong = [f"{band}:{projection}" for band, values in adjudication.items()
              for projection, outcome in values.items() if outcome["strong_gate"]]
    checks = {"phase2575_complete": prior["all_checks_passed"],
              "phase2575_behavior_qualified": prior["behavior_qualified"],
              "same_32_quartets": len(selected) == prior["design"]["selected_quartets"],
              "four_relative_bands_cover_40_layers": sorted(layer for band in band_map.values() for layer in band) == list(range(40)),
              "v_and_kv_five_conditions_each": len(specs) == 41,
              "two_candidates_each": len(rows) == len(selected) * 2 * len(specs),
              "no_patch_identity": summary["no_patch"]["base_accuracy"] >= .95,
              "scientific_failure_does_not_abort": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-14B BF16 nonquantized device_map=auto",
              "selected_quartets": len(selected), "conditions": len(specs),
              "bands": {name: list(layers) for name, layers in band_map.items()},
              "summary": summary, "adjudication": adjudication,
              "strong_events": strong, "cross_scale_stage_migration_found": bool(strong),
              "checks": checks, "all_checks_passed": all(checks.values()),
              "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({"phase": PHASE, "selected_quartets": len(selected), "adjudication": adjudication,
                      "strong_events": strong, "checks": checks,
                      "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
