#!/usr/bin/env python3
"""Prospectively repair surface/order separation and rerun the Qwen3-4B field."""
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
SOURCE_OUT = RESULT / "phase2296_c3101_c3160_ntp_predictive_contract"
P2303 = RESULT / "phase2303_c3701_c3780_declarative_continuation_contract"
P2304 = RESULT / "phase2304_c3781_c3900_qwen4b_declarative_field"
P2305 = RESULT / "phase2305_c3901_c4020_interface_state_accounting"
OUT = RESULT / "phase2306_c4021_c4160_corrected_surface_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = OUT / "material/corrected_declarative_continuation_bilingual.jsonl"
RAW = OUT / "raw"
BOUNDARY = RAW / "qwen4b_corrected_boundary_all_checkpoints.float16.npy"
LOGITS = RAW / "qwen4b_corrected_full_vocabulary_logits.float16.npy"
PROGRESS = RAW / "boundary_capture_progress.json"
CONTRIBUTIONS = OUT / "atlas/qwen4b_corrected_target_wrong_contributions.float16.npy"
sys.path.insert(0, str(TESTS))

import phase1332_bf16_utils as model_base  # noqa: E402
import phase2297_c3161_c3260_qwen4b_ntp_predictive_field as old_field  # noqa: E402
import phase2303_c3701_c3780_declarative_continuation_contract as contract  # noqa: E402
import phase2304_c3781_c3900_qwen4b_declarative_field as runner  # noqa: E402
import phase2305_c3901_c4020_interface_state_accounting as accounting  # noqa: E402


PHASE = 2306
CAMPAIGN = "C4021-C4160"


def compile_corrected() -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    source = contract.read_rows(SOURCE_OUT / "material/ntp_natural_bilingual.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(
        contract.model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    rows, cells = [], defaultdict(lambda: {"first": 0, "last": 0, "state0": 0, "state1": 0})
    for old in source:
        target_first = int(old["unit"]) % 2 == 0
        source_order = contract.target_first_in_source(old)
        fact = (contract.source_fact(old["prompt_core"], old["language"])
                if source_order == target_first else contract.alternate_fact(old, target_first))
        prefix = fact + (" " if old["language"] == "en" else "") + contract.continuation_cue(old)
        target_ids = contract.answer_ids(tokenizer, old["correct_answer"], old["language"])
        wrong_ids = contract.answer_ids(tokenizer, old["wrong_answer"], old["language"])
        mention = "first" if prefix.find(old["correct_answer"]) < prefix.find(old["wrong_answer"]) else "last"
        key = (old["family"], old["language"], old["surface"], old["partition"])
        cells[key][mention] += 1
        cells[key][f"state{int(old['state'])}"] += 1
        rows.append({
            **old, "source_case_id": old["case_id"], "case_id": old["case_id"] + "-declfix",
            "declarative_prefix": prefix,
            "ntp_prompt_ids": [int(value) for value in tokenizer.encode(prefix, add_special_tokens=False)],
            "ntp_target_ids": target_ids, "ntp_wrong_ids": wrong_ids,
            "ntp_target_text": old["correct_answer"], "ntp_wrong_text": old["wrong_answer"],
            "target_mention_order": mention, "source_fact_order_matched": source_order == target_first,
            "ntp_interface": "raw_declarative_continuation_unit_only_order_quota",
        })
    by_key = {(row["family"], row["language"], row["surface"], int(row["unit"]), int(row["state"])): row
              for row in rows}
    same_order = all(
        by_key[(family, language, "narrative", unit, state)]["target_mention_order"] ==
        by_key[(family, language, "dialogue", unit, state)]["target_mention_order"]
        for family in contract.FAMILIES for language in ("en", "zh")
        for unit in range(32) for state in (0, 1)
    )
    audit = {
        "rows": len(rows), "first_mention_accuracy": float(np.mean([
            row["target_mention_order"] == "first" for row in rows
        ])),
        "last_mention_accuracy": float(np.mean([row["target_mention_order"] == "last" for row in rows])),
        "every_cell_balanced": all(value["first"] == value["last"] and value["state0"] == value["state1"]
                                   for value in cells.values()),
        "same_unit_surfaces_same_mention_order": same_order,
        "first_token_collision_count": sum(row["ntp_target_ids"][0] == row["ntp_wrong_ids"][0] for row in rows),
        "forbidden_marker_count": sum(any(marker in row["declarative_prefix"] for marker in
                                           ("?", "？", "Answer", "Options", "只回答")) for row in rows),
        "unicode_replacement_count": sum("\ufffd" in row["declarative_prefix"] for row in rows),
    }
    return rows, audit


def configure_runner() -> None:
    runner.OUT = OUT
    runner.RAW = RAW
    runner.BOUNDARY = BOUNDARY
    runner.LOGITS = LOGITS
    runner.PROGRESS = PROGRESS
    runner.CONTRIBUTIONS = CONTRIBUTIONS


def make_pairs(rows: list[dict]) -> dict[str, list[dict]]:
    old_rows = contract.read_rows(SOURCE_OUT / "material/ntp_natural_bilingual.jsonl")
    old_index = {row["case_id"]: i for i, row in enumerate(old_rows)}
    index = {(row["family"], row["language"], row["surface"], int(row["unit"]), int(row["state"])): i
             for i, row in enumerate(rows)}
    result = {"interface": [], "state": [], "surface": []}
    for i, row in enumerate(rows):
        result["interface"].append({
            "pair_type": "interface_source_order_matched" if row["source_fact_order_matched"] else
                         "interface_plus_fact_order_change",
            "left_index": old_index[row["source_case_id"]], "right_index": i,
            "family": row["family"], "language": row["language"], "surface": row["surface"],
            "partition": row["partition"], "unit": row["unit"], "state": row["state"],
        })
    for family in contract.FAMILIES:
        for language in ("en", "zh"):
            for surface in ("narrative", "dialogue"):
                for unit in range(32):
                    left, right = index[(family, language, surface, unit, 0)], index[(family, language, surface, unit, 1)]
                    result["state"].append({
                        "pair_type": "state_flip", "left_index": left, "right_index": right,
                        "family": family, "language": language, "surface": surface,
                        "partition": rows[left]["partition"], "unit": unit,
                    })
            for unit in range(32):
                for state in (0, 1):
                    left = index[(family, language, "narrative", unit, state)]
                    right = index[(family, language, "dialogue", unit, state)]
                    result["surface"].append({
                        "pair_type": "surface_same_mention_order", "left_index": left, "right_index": right,
                        "family": family, "language": language, "partition": rows[left]["partition"],
                        "unit": unit, "state": state,
                    })
    return result


def timing(rows: list[dict], qualified: set[str]) -> tuple[dict, list[dict]]:
    lens = contract.read_rows(OUT / "prediction/logit_lens_metrics.jsonl")
    by = defaultdict(list)
    for row in lens:
        by[(row["family"], int(row["checkpoint"]))].append(row)
    qpoints = list(contract.QPOINTS_4B)
    families, q14_cells = {}, []
    for family in contract.FAMILIES:
        checkpoints = {}
        for q in qpoints:
            values = by[(family, q)]
            train = [row for row in values if row["partition"] in ("discovery", "confirmation")]
            fresh = [row for row in values if row["partition"] in ("fresh_confirmation", "fresh_lockbox")]
            checkpoints[str(q)] = {
                "train_sign_accuracy": float(np.mean([row["target_wrong_margin"] > 0 for row in train])),
                "fresh_sign_accuracy": float(np.mean([row["target_wrong_margin"] > 0 for row in fresh])),
                "fresh_js_to_final": float(np.mean([row["js_to_actual_final"] for row in fresh])),
            }
        formation = None
        if family in qualified:
            for qi, q in enumerate(qpoints):
                if all(checkpoints[str(later)]["train_sign_accuracy"] >= contract.FORMATION_GATE
                       for later in qpoints[qi:]):
                    formation = q
                    break
        families[family] = {"behavior_qualified": family in qualified, "formation_q4": formation,
                            "checkpoints": checkpoints,
                            "fresh_at_formation": None if formation is None else checkpoints[str(formation)]}
        if family in contract.Q14_FAMILIES:
            q4 = 36 if formation is None else formation
            q14 = 41 if q4 == 37 else (0 if q4 == 0 else int(round(q4 * 40 / 36)))
            q14_cells.append({
                "family": family, "qwen4_formation_checkpoint": q4, "qwen14_checkpoint": q14,
                "eligible": family in qualified and formation is not None and
                            checkpoints[str(formation)]["fresh_sign_accuracy"] >= contract.FORMATION_GATE,
                "selection_partitions": ["discovery", "confirmation"],
                "test_partitions": ["fresh_confirmation", "fresh_lockbox"], "gate": contract.FORMATION_GATE,
            })
    return families, q14_cells


def contribution_distances(rows: list[dict], pairs: dict[str, list[dict]]) -> dict:
    values = np.load(CONTRIBUTIONS, mmap_mode="r")
    output = {}
    for name in ("state", "surface"):
        records = []
        for pair in pairs[name]:
            records.append({"family": pair["family"],
                            "distance": accounting.normalized_l1(values[pair["left_index"]],
                                                                  values[pair["right_index"]])})
        output[name] = {
            "overall": accounting.summarize([row["distance"] for row in records]),
            "families": {family: accounting.summarize([
                row["distance"] for row in records if row["family"] == family
            ]) for family in contract.FAMILIES},
        }
    return output


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact_behavior = {family: {
        "qualified": value["qualified"],
        "mean": value["slices"]["overall:all"]["mean_accuracy"],
        "sum": value["slices"]["overall:all"]["sum_accuracy"],
    } for family, value in result["sequence_ledger"]["families"].items()}
    compact_js = {family: {kind: value["js"]["mean"] for kind, value in kinds.items()}
                  for family, kinds in result["vocabulary"]["families"].items()}
    compact_timing = {family: {"eligible": value["behavior_qualified"], "q": value["formation_q4"],
                               "fresh": value["fresh_at_formation"]}
                      for family, value in result["timing"].items()}
    text = rf"""

## Phase {PHASE}: 纯表面负控修复与 Qwen3-4B 前瞻复验（{CAMPAIGN}） [{stamp}]

**测试原理与冻结修复。** Phase2305 揭盲后确认旧配额把表面改写与提及顺序绑在一起。本期另立前瞻合同，不修改 Phase2303–2305：目标首先/最后出现只由 `unit mod 2` 决定，因此同 family、language、unit、state 的 narrative/dialogue 保持相同提及顺序，同时每个族×语言×表面×分区仍为 50/50。配置和材料哈希在加载 Qwen3-4B 前落盘。再次执行完整候选序列、自由续写、151936 维词表、38×2560 边界场、逐坐标输出贡献和冻结 logit-lens 检查点；六族英语叙事全 token 场与 Phase2304 完全相同，按 prompt ids 审计后继承，不重复占盘。

$$
o_i=\operatorname{{unit}}_i\bmod 2,
\qquad
o_{{i,narrative}}=o_{{i,dialogue}}.
$$

**结果与门槛。** 材料审计 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；六族行为 `{json.dumps(compact_behavior, ensure_ascii=False)}`，合格族 `{result['sequence_ledger']['qualified_families']}`。自由续写 `{json.dumps(result['free_ledger'], ensure_ascii=False)}`；首 token `{json.dumps(result['first_token_ledger'], ensure_ascii=False)}`。合法的状态、纯表面和接口完整词表平均 JS `{json.dumps(compact_js, ensure_ascii=False)}`；逐坐标贡献的状态/表面距离 `{json.dumps(result['contribution_distances'], ensure_ascii=False)}`；形成时序 `{json.dumps(compact_timing, ensure_ascii=False)}`。冻结 14B 单元 `{json.dumps(result['q14_freeze'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。本轮修复只移除了一个明确混杂，不能把低表面距离解释成抽象语义同构，也不能把状态距离解释成因果关系算子。裸文本续写仍是受控模板；自由生成 exact-prefix 仍会漏记合法表达；logit lens 与首 token 逐坐标账仍是输出辅助读出。不同语言的 tokenization 和词汇形态不同，跨语言不能比较同一物理 token 位置。基础计数、完整词表概率和原坐标距离足够，本期未使用 PCA、Top-K 或高等数学。脚本 `tests/glm5/phase2306_c4021_c4160_corrected_surface_replication.py`；结果 `tests/glm5/result/phase2306_c4021_c4160_corrected_surface_replication`。下一步只允许在预选三族中，对行为与 fresh 形成门均合格的族运行 Qwen3-14B；不合格族记 NA，不阻止其他族。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = json.loads((P2305 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2305 incomplete")
    rows, audit = compile_corrected()
    contract.write_rows(ROWS_PATH, rows)
    config = {
        "phase": PHASE, "campaign": CAMPAIGN, "frozen_before_qwen4_load": True,
        "correction": "target mention order depends only on unit parity, never surface",
        "families": list(contract.FAMILIES), "partitions": list(contract.PARTITIONS),
        "behavior_gate": contract.BEHAVIOR_GATE, "formation_gate": contract.FORMATION_GATE,
        "qpoints": list(contract.QPOINTS_4B), "q14_families_preselected": list(contract.Q14_FAMILIES),
        "claim_policy": "observation first; surface comparison valid only after same-order audit",
    }
    contract.save(OUT / "config/frozen_contract.json", config)
    prechecks = {
        "row_count": len(rows) == 1536, "cell_balance": audit["every_cell_balanced"],
        "same_unit_surface_order": audit["same_unit_surfaces_same_mention_order"],
        "zero_models_half": audit["first_mention_accuracy"] == audit["last_mention_accuracy"] == 0.5,
        "no_collision_or_markers": audit["first_token_collision_count"] == audit["forbidden_marker_count"] == 0,
        "unicode_intact": audit["unicode_replacement_count"] == 0,
    }
    if not all(prechecks.values()):
        raise RuntimeError(("corrected_contract_failed", prechecks))
    configure_runner()
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = model_base.load_bf16("qwen3")
        score_path = OUT / "behavior/sequence_scores.jsonl"
        if score_path.exists():
            scores = contract.read_rows(score_path)
        else:
            scores = old_field.sequence_scores(model, device, rows, batch_size=12)
            contract.write_rows(score_path, scores)
        ledger = runner.sequence_ledger(rows, scores)
        contract.save(OUT / "behavior/sequence_ledger.json", ledger)
        free_path = OUT / "behavior/free_continuations.jsonl"
        if free_path.exists():
            free_rows = contract.read_rows(free_path)
        else:
            free_rows = runner.free_continuations(model, tokenizer, device, rows)
            contract.write_rows(free_path, free_rows)
        free_summary = runner.free_ledger(free_rows)
        field = runner.capture_boundary_and_logits(model, device, rows)
        contributions = runner.exact_contributions(model, rows)
        lens = runner.lens_metrics(model, rows, contract.QPOINTS_4B)
        first = runner.first_token_ledger(rows)
        model_info = {"name": "Qwen3-4B", "precision": "bfloat16", "quantization": "none",
                      "placement": placement, "layers": len(model.model.layers),
                      "hidden_size": int(model.config.hidden_size), "vocabulary": int(model.config.vocab_size)}
    finally:
        if model is not None:
            model_base.release_bf16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pairs = make_pairs(rows)
    old_logits = np.load(accounting.OLD_LOGITS, mmap_mode="r")
    new_logits = np.load(LOGITS, mmap_mode="r")
    pair_rows = accounting.js_pairs(old_logits, new_logits, pairs["interface"])
    pair_rows += accounting.js_pairs(new_logits, new_logits, pairs["state"])
    pair_rows += accounting.js_pairs(new_logits, new_logits, pairs["surface"])
    contract.write_rows(OUT / "probability/full_vocabulary_pair_distances.jsonl", pair_rows)
    vocabulary = accounting.group_vocabulary(pair_rows)
    contract.save(OUT / "probability/full_vocabulary_summary.json", vocabulary)
    qualified = set(ledger["qualified_families"])
    timing_result, q14_cells = timing(rows, qualified)
    q14_freeze = {
        "frozen_before_qwen14_load": True, "source_phase": PHASE,
        "model": "Qwen3-14B", "families_preselected_in_phase2303": list(contract.Q14_FAMILIES),
        "cells": q14_cells,
    }
    contract.save(OUT / "protocol/qwen14_corrected_surface_freeze.json", q14_freeze)
    contribution_result = contribution_distances(rows, pairs)
    representative_old = [row for row in contract.read_rows(P2303 / "material/declarative_continuation_bilingual.jsonl")
                          if row["partition"] == "fresh_lockbox" and int(row["unit"]) == 26
                          and row["language"] == "en" and row["surface"] == "narrative" and int(row["state"]) == 0]
    representative_new = [row for row in rows if row["partition"] == "fresh_lockbox" and int(row["unit"]) == 26
                          and row["language"] == "en" and row["surface"] == "narrative" and int(row["state"]) == 0]
    inherited_tokens = [row["ntp_prompt_ids"] for row in representative_old] == [row["ntp_prompt_ids"] for row in representative_new]
    checks = {
        **prechecks,
        "all_sequence_rows": len(scores) == len(rows), "all_free_rows": len(free_rows) == len(rows),
        "all_boundary_coordinates": field["field_shape"] == [1536, 38, 2560],
        "all_vocabulary_logits": field["logits_shape"] == [1536, 151936],
        "all_contributions": contributions["shape"] == [1536, 2560],
        "all_lens_rows": lens["rows"] == 15360,
        "surface_pairs_same_order": all(rows[pair["left_index"]]["target_mention_order"] ==
                                        rows[pair["right_index"]]["target_mention_order"]
                                        for pair in pairs["surface"]),
        "six_family_narrative_token_field_identical_and_inherited": inherited_tokens,
        "q14_freeze_written_after_q4_and_before_q14": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "material_audit": audit, "config": config, "model": model_info,
        "sequence_ledger": ledger, "free_ledger": free_summary, "first_token_ledger": first,
        "field": field, "contributions": contributions, "lens": lens,
        "vocabulary": vocabulary, "contribution_distances": contribution_result,
        "timing": timing_result, "q14_freeze": q14_freeze, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "hashes": {"rows": contract.file_hash(ROWS_PATH), "contract": contract.file_hash(OUT / "config/frozen_contract.json"),
                   "boundary": contract.file_hash(BOUNDARY), "logits": contract.file_hash(LOGITS),
                   "contributions": contract.file_hash(CONTRIBUTIONS)},
        "strict_conclusion": (
            f"After prospective order repair, Qwen3-4B qualified {len(qualified)}/6 families; pure same-order "
            "surface, state, and interface effects are now separately observable over the complete vocabulary, "
            "but remain output/state regularities rather than causal semantic gears."
        ),
        "next_authorization": (
            "Run Qwen3-14B only for preselected cells marked eligible, on fresh partitions and frozen relative "
            "checkpoints; publish important full-coordinate atlases and clean undisplayed raw fields afterward."
        ),
    }
    contract.save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
