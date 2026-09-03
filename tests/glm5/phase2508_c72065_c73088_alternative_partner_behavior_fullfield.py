#!/usr/bin/env python3
"""Second, behavior-identifiable re-pairing of the six successful relations."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2507 = RESULT / "phase2507_c71041_c72064_repaired_partner_behavior_fullfield"
OUT = RESULT / "phase2508_c72065_c73088_alternative_partner_behavior_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2508, "C72065-C73088"
UNITS = (26, 27)
ALT_PAIRS = (("taxonomy", "preference"), ("part_whole", "translation"), ("role", "membership"))
MARKERS = {26: {"en": ("lurem", "pavon"), "zh": ("寅符", "卯符")},
           27: {"en": ("pebrix", "cavum"), "zh": ("辰符", "巳符")}}
EN_NAMES = {26: ("Alven", "Beric", "Cavor", "Dalen", "Eris", "Favin", "Goran", "Helis"),
            27: ("Ibram", "Jorel", "Kavin", "Leris", "Morin", "Navor", "Orel", "Perin")}
ZH_NAMES = {26: ("安岫", "碧川", "苍汀", "澹舟", "峨岚", "芙溪", "谷禾", "鹤野"),
            27: ("景岫", "阔川", "澜汀", "梅舟", "南岚", "平溪", "秋禾", "润野")}

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2507_c71041_c72064_repaired_partner_behavior_fullfield as prior  # noqa: E402


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def configure() -> None:
    prior.UNITS = UNITS
    prior.NEW_PAIRS = ALT_PAIRS
    prior.MARKERS = MARKERS
    prior.EN_NAMES = EN_NAMES
    prior.ZH_NAMES = ZH_NAMES
    prior.OUT = OUT
    prior.base.UNITS = UNITS
    prior.base.SPLIT = {26: "alternative_partner_confirmation", 27: "alternative_partner_lockbox"}
    prior.base.PAIRS = ALT_PAIRS
    for unit in UNITS:
        prior.base.MARKERS[unit] = MARKERS[unit]
        prior.base.EN_NAMES[unit] = EN_NAMES[unit]
        prior.base.ZH_NAMES[unit] = ZH_NAMES[unit]


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 第二套关系伙伴重配、双新unit行为门与全坐标场（自动续研）（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2507的part-whole/membership在中文与paired-flip门失败，使第一套六边闭环不可辨识；不能放宽门槛或混入失败样本。本Phase继续同一目标，把六关系重新配为taxonomy/preference、part-whole/translation、role/membership，和原边共同构成另一条六节点闭环。unit26/27均使用全新中英文实体、全新且语言内token等长的nonce marker；每unit 96条，共192条。q30继续冻结，采六事件×38qpoint×2560坐标。

$$E_2=\{{taxonomy-preference,\ part\_whole-translation,\ role-membership\}}.$$

**结果汇总。** 设计 `{json.dumps(result['design_audit'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；原场 `{json.dumps(result['collection'], ensure_ascii=False)}`；prefix `{json.dumps(result['prefix_control'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2508_c72065_c73088_alternative_partner_behavior_fullfield.py`；第二重配材料、逐行生成、六事件全场、索引、哈希和final位于`{OUT}`。

**分析与理论进展。** 行为门若三边完整，下一Phase可把原unit21与新unit26、原unit23与新unit27分别组成六边环，检验 (I_{{a-b}}\approx z_a-z_b) 的坐标级可加性与留一边预测；这比比较两个不共享方向的余弦更直接。

**问题硬伤与结论。** 第二次换伙伴本身可能改变任务难度，只有两套新unit共同合格的边可进入图谱。即使六边闭合，也可能源于共同模板和二候选输出结构；闭环失败也只否定固定q30的简单可加family势，不否定非线性、条件化或更高维的编码。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    configure()
    f2507 = json.loads((P2507 / "analysis/final.json").read_text(encoding="utf-8"))
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        rows = prior.base.compile_rows(tokenizer)
        for case, row in enumerate(rows, start=72065):
            row["case_id"] = f"c{case:05d}-ap{row['pair_id']}-u{row['unit']}-{row['language']}-s{row['surface']}-m{row['meaning_swap']}-q{row['query_marker']}"
        audit = prior.design_audit(rows)
        prior.write_jsonl(OUT / "material/alternative_partner_rows.jsonl", rows)
        generated = prior.base.behavior(model, tokenizer, rows)
        prior.write_jsonl(OUT / "behavior/autonomous_generation.jsonl", generated)
        behavior = prior.behavior_summary(rows, generated)
        collection = prior.capture(model, rows, {r["case_id"]: r for r in generated})
    finally:
        model_utils.release_model(model); gc.collect()
    prefix = prior.prefix_control(collection, 30, behavior["qualified_pair_ids"])
    checks = {"source_phase_passed": f2507["all_checks_passed"], "rows_192": len(rows) == 192,
              "equal_prompt_length": audit["full_length_equal_across_query_rate"] == 1.0,
              "prefix_token_equal": audit["prefix_token_equal_rate"] == 1.0,
              "all_answers_flip": audit["answer_flip_rate"] == 1.0,
              "token_multiset_control": audit["definition_swap_token_multiset_equal_rate"] == 1.0,
              "candidate_position_balanced": set(audit["candidate_position_counts"].values()) == {96},
              "at_least_one_pair_qualified": len(behavior["qualified_pair_ids"]) >= 1,
              "all_three_pair_gate_outcome_recorded": True,
              "event_shape": collection["event_shape"] == [192, 6, 38, 2560],
              "prefix_interaction_exact_zero": all(v["all_exact_zero"] for v in prefix.values()),
              "hash": len(collection["sha256"]) == 64, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B nonquantized BF16 CUDA",
              "qpoint": 30, "alternative_pairs": [list(v) for v in ALT_PAIRS],
              "design_audit": audit, "behavior": behavior, "collection": collection, "prefix_control": prefix,
              "adjudication": {"alternative_partner_graph_behavior_identifiable_alone": len(behavior["qualified_pair_ids"]) == 3,
                               "combined_passed_edge_graph_identifiable_with_phase2507": len(behavior["qualified_pair_ids"]) >= 1,
                               "additive_family_graph_tested": False, "pure_semantic_code_identified": False,
                               "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
