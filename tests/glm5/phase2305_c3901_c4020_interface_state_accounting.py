#!/usr/bin/env python3
"""Separate valid state/interface effects and expose the frozen surface-order confound."""
from __future__ import annotations

import json
import math
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
P2296 = RESULT / "phase2296_c3101_c3160_ntp_predictive_contract"
P2297 = RESULT / "phase2297_c3161_c3260_qwen4b_ntp_predictive_field"
P2299 = RESULT / "phase2299_c3341_c3440_predictive_timing_coordinate_structure"
P2303 = RESULT / "phase2303_c3701_c3780_declarative_continuation_contract"
P2304 = RESULT / "phase2304_c3781_c3900_qwen4b_declarative_field"
OUT = RESULT / "phase2305_c3901_c4020_interface_state_accounting"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
OLD_ROWS = P2296 / "material/ntp_natural_bilingual.jsonl"
NEW_ROWS = P2303 / "material/declarative_continuation_bilingual.jsonl"
OLD_LOGITS = P2297 / "raw/qwen4b_ntp_full_vocabulary_logits.float16.npy"
NEW_LOGITS = P2304 / "raw/qwen4b_declarative_full_vocabulary_logits.float16.npy"
NEW_FIELD = P2304 / "raw/qwen4b_declarative_boundary_all_checkpoints.float16.npy"
OLD_CONTRIBUTIONS = P2297 / "atlas/qwen4b_target_wrong_coordinate_contributions.float16.npy"
NEW_CONTRIBUTIONS = P2304 / "atlas/qwen4b_declarative_target_wrong_contributions.float16.npy"
OLD_VIS_META = VIS / "c3601_qwen4b_ntp_boundary_trajectory.json"
OLD_VIS_FIELD = VIS / "c3601_qwen4b_ntp_boundary_trajectory.float16.npy"
INTERFACE_DELTA = OUT / "atlas/qwen4b_qa_to_declarative_unit26_signed_delta.float16.npy"
sys.path.insert(0, str(TESTS))

import phase2303_c3701_c3780_declarative_continuation_contract as contract  # noqa: E402


PHASE = 2305
CAMPAIGN = "C3901-C4020"
EPS = 1e-12


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max())}


def normalized_l1(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    scale = 0.5 * (np.abs(a).sum() + np.abs(b).sum())
    return float(np.abs(a - b).sum() / max(float(scale), EPS))


def effective_count(values: np.ndarray) -> float:
    vector = np.abs(np.asarray(values, dtype=np.float64))
    return float(vector.sum() ** 2 / max(float(np.sum(vector * vector)), EPS))


def js_pairs(left: np.ndarray, right: np.ndarray, pairs: list[dict], batch_size: int = 16) -> list[dict]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output: list[dict] = []
    log_two = math.log(2.0)
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        left_values = torch.tensor(
            np.asarray(left[[row["left_index"] for row in batch]], dtype=np.float32), device=device,
        )
        right_values = torch.tensor(
            np.asarray(right[[row["right_index"] for row in batch]], dtype=np.float32), device=device,
        )
        logp, logq = torch.log_softmax(left_values, dim=-1), torch.log_softmax(right_values, dim=-1)
        logm = torch.logaddexp(logp, logq) - log_two
        p, q = torch.exp(logp), torch.exp(logq)
        js = 0.5 * torch.sum(p * (logp - logm), dim=-1) + 0.5 * torch.sum(q * (logq - logm), dim=-1)
        tv = 0.5 * torch.sum(torch.abs(p - q), dim=-1)
        for meta, js_value, tv_value in zip(batch, js.detach().cpu().tolist(), tv.detach().cpu().tolist()):
            output.append({**meta, "js": float(js_value), "total_variation": float(tv_value)})
        print(f"[phase2305 vocabulary] {start + len(batch)}/{len(pairs)}", flush=True)
    return output


def make_pairs(old_rows: list[dict], new_rows: list[dict]) -> dict[str, list[dict]]:
    old_index = {row["case_id"]: i for i, row in enumerate(old_rows)}
    new_key = {(row["family"], row["language"], row["surface"], int(row["unit"]), int(row["state"])): i
               for i, row in enumerate(new_rows)}
    interface = []
    for i, row in enumerate(new_rows):
        interface.append({
            "pair_type": "interface_source_order_matched" if row["source_fact_order_matched"] else
                         "interface_plus_fact_order_change",
            "left_index": old_index[row["source_case_id"]], "right_index": i,
            "family": row["family"], "language": row["language"], "surface": row["surface"],
            "partition": row["partition"], "unit": row["unit"], "state": row["state"],
            "source_case_id": row["source_case_id"], "new_case_id": row["case_id"],
        })
    state = []
    surface_order = []
    for family in contract.FAMILIES:
        for language in ("en", "zh"):
            for surface in ("narrative", "dialogue"):
                for unit in range(32):
                    left = new_key[(family, language, surface, unit, 0)]
                    right = new_key[(family, language, surface, unit, 1)]
                    row = new_rows[left]
                    state.append({"pair_type": "state_flip", "left_index": left, "right_index": right,
                                  "family": family, "language": language, "surface": surface,
                                  "partition": row["partition"], "unit": unit})
            for unit in range(32):
                for state_value in (0, 1):
                    left = new_key[(family, language, "narrative", unit, state_value)]
                    right = new_key[(family, language, "dialogue", unit, state_value)]
                    row = new_rows[left]
                    surface_order.append({
                        "pair_type": "surface_plus_opposite_mention_order", "left_index": left, "right_index": right,
                        "family": family, "language": language, "partition": row["partition"],
                        "unit": unit, "state": state_value,
                    })
    return {"interface": interface, "state": state, "surface_order_bundle": surface_order}


def group_vocabulary(rows: list[dict]) -> dict:
    output = {"overall": {}}
    for pair_type in sorted(set(row["pair_type"] for row in rows)):
        values = [row for row in rows if row["pair_type"] == pair_type]
        output["overall"][pair_type] = {
            "js": summarize([row["js"] for row in values]),
            "total_variation": summarize([row["total_variation"] for row in values]),
        }
    output["families"] = {}
    for family in contract.FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        output["families"][family] = {}
        for pair_type in sorted(set(row["pair_type"] for row in family_rows)):
            values = [row for row in family_rows if row["pair_type"] == pair_type]
            output["families"][family][pair_type] = {
                "js": summarize([row["js"] for row in values]),
                "total_variation": summarize([row["total_variation"] for row in values]),
            }
    return output


def sequence_interface(old_rows: list[dict], new_rows: list[dict]) -> dict:
    old_scores = {row["case_id"]: row for row in contract.read_rows(P2297 / "behavior/lexical_sequence_scores.jsonl")}
    new_scores = {row["case_id"]: row for row in contract.read_rows(P2304 / "behavior/sequence_scores.jsonl")}
    records = []
    for row in new_rows:
        old = old_scores[row["source_case_id"]]
        new = new_scores[row["case_id"]]
        records.append({
            "family": row["family"], "source_order_matched": row["source_fact_order_matched"],
            "qa_correct_mean": old["correct_by_mean"], "declarative_correct_mean": new["correct_by_mean"],
            "qa_mean_margin": old["mean_margin"], "declarative_mean_margin": new["mean_margin"],
        })
    def summary(values: list[dict]) -> dict:
        return {
            "rows": len(values),
            "qa_accuracy": float(np.mean([row["qa_correct_mean"] for row in values])),
            "declarative_accuracy": float(np.mean([row["declarative_correct_mean"] for row in values])),
            "both_correct": float(np.mean([row["qa_correct_mean"] and row["declarative_correct_mean"] for row in values])),
            "margin_sign_agreement": float(np.mean([
                (row["qa_mean_margin"] > 0) == (row["declarative_mean_margin"] > 0) for row in values
            ])),
            "mean_margin_change": float(np.mean([
                row["declarative_mean_margin"] - row["qa_mean_margin"] for row in values
            ])),
        }
    return {
        "overall": summary(records),
        "source_order_matched": summary([row for row in records if row["source_order_matched"]]),
        "fact_order_changed": summary([row for row in records if not row["source_order_matched"]]),
        "families": {family: summary([row for row in records if row["family"] == family])
                     for family in contract.FAMILIES},
    }


def contribution_accounting(old_rows: list[dict], new_rows: list[dict], pairs: dict[str, list[dict]]) -> dict:
    old = np.load(OLD_CONTRIBUTIONS, mmap_mode="r")
    new = np.load(NEW_CONTRIBUTIONS, mmap_mode="r")
    records = []
    for collection in pairs.values():
        for row in collection:
            left = old[row["left_index"]] if row["pair_type"].startswith("interface") else new[row["left_index"]]
            right = new[row["right_index"]]
            records.append({
                "pair_type": row["pair_type"], "family": row["family"],
                "normalized_l1": normalized_l1(left, right),
            })
    per_sample = []
    for i, row in enumerate(new_rows):
        per_sample.append({"case_id": row["case_id"], "family": row["family"],
                           "effective_coordinates": effective_count(new[i])})
    return {
        "pairs": {
            pair_type: summarize([row["normalized_l1"] for row in records if row["pair_type"] == pair_type])
            for pair_type in sorted(set(row["pair_type"] for row in records))
        },
        "effective_coordinates": {
            family: summarize([row["effective_coordinates"] for row in per_sample if row["family"] == family])
            for family in contract.FAMILIES
        },
    }


def interface_coordinate_delta(new_rows: list[dict]) -> dict:
    meta = json.loads(OLD_VIS_META.read_text(encoding="utf-8"))["rows"]
    old_field = np.load(OLD_VIS_FIELD, mmap_mode="r")
    new_field = np.load(NEW_FIELD, mmap_mode="r")
    new_index = {row["source_case_id"]: i for i, row in enumerate(new_rows)}
    INTERFACE_DELTA.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(INTERFACE_DELTA, mode="w+", dtype=np.float16,
                                       shape=old_field.shape)
    rows_out, distances = [], []
    for i, row in enumerate(meta):
        index = new_index[row["case_id"]]
        q = int(row["checkpoint"])
        delta = np.asarray(new_field[index, q], dtype=np.float32) - np.asarray(old_field[i], dtype=np.float32)
        output[i] = delta.astype(np.float16)
        source = new_rows[index]
        distance = normalized_l1(new_field[index, q], old_field[i])
        distances.append({"family": row["family"], "checkpoint": q,
                          "source_order_matched": source["source_fact_order_matched"], "distance": distance})
        rows_out.append({**row, "row": i, "metric": "declarative_minus_qa_signed_activation",
                         "source_order_matched": source["source_fact_order_matched"]})
    output.flush()
    contract.write_rows(OUT / "index/interface_delta_rows.jsonl", rows_out)
    by_q = {}
    for q in range(38):
        values = [row["distance"] for row in distances if row["checkpoint"] == q]
        matched = [row["distance"] for row in distances if row["checkpoint"] == q and row["source_order_matched"]]
        by_q[str(q)] = {"all": summarize(values), "source_order_matched": summarize(matched)}
    return {"path": str(INTERFACE_DELTA.relative_to(ROOT)), "shape": list(output.shape),
            "row_index": str((OUT / "index/interface_delta_rows.jsonl").relative_to(ROOT)),
            "normalized_l1_by_checkpoint": by_q}


def timing(new_rows: list[dict]) -> dict:
    lens = contract.read_rows(P2304 / "prediction/logit_lens_metrics.jsonl")
    qualified = set(json.loads((P2304 / "analysis/final.json").read_text(encoding="utf-8"))
                    ["sequence_ledger"]["qualified_families"])
    by = defaultdict(list)
    for row in lens:
        by[(row["family"], int(row["checkpoint"]))].append(row)
    families = {}
    qpoints = list(contract.QPOINTS_4B)
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
            for index, q in enumerate(qpoints):
                if all(checkpoints[str(later)]["train_sign_accuracy"] >= contract.FORMATION_GATE
                       for later in qpoints[index:]):
                    formation = q
                    break
        families[family] = {
            "behavior_qualified": family in qualified, "formation_q4": formation,
            "checkpoints": checkpoints,
            "fresh_at_formation": None if formation is None else checkpoints[str(formation)],
        }
    old = json.loads((P2299 / "analysis/final.json").read_text(encoding="utf-8"))["timing"]
    return {"declarative": families,
            "qa_formation": {family: old[family]["formation_q4"] for family in contract.FAMILIES}}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact_vocab = {
        family: {kind: value["js"]["mean"] for kind, value in kinds.items()}
        for family, kinds in result["vocabulary"]["families"].items()
    }
    formation = {family: {"qualified": value["behavior_qualified"], "q": value["formation_q4"],
                          "fresh": value["fresh_at_formation"]}
                 for family, value in result["timing"]["declarative"].items()}
    text = rf"""

## Phase {PHASE}: 问答—陈述接口、状态与顺序混杂全量分账（{CAMPAIGN}） [{stamp}]

**测试原理与合同审计。** 本期不运行模型，逐行配对 Phase2297 的问答接口与 Phase2304 的自然陈述续写接口，并对 151936 维完整下一 token 分布计算 Jensen–Shannon 距离和总变差。状态翻转在同一族、语言、表面、unit 内比较 state0/state1；接口比较只把 `source_fact_order_matched=True` 的 768 行视为较纯接口改写，另外 768 行明确标为“接口+事实顺序改写”。揭盲后发现 Phase2303 的 `(unit+surfaceIndex)%2` 配额使同 unit 的 narrative/dialogue 总是拥有相反提及顺序，因此该项只能称为“表面+顺序捆绑”，不能充当纯表面零模型。

$$
JS(P,Q)=\frac12KL(P\|M)+\frac12KL(Q\|M),\qquad M=\frac12(P+Q),
$$

$$
d_1(a,b)=\frac{{\|a-b\|_1}}{{(\|a\|_1+\|b\|_1)/2}},
\qquad
N_{{eff}}(c)=\frac{{(\sum_j|c_j|)^2}}{{\sum_jc_j^2}}.
$$

**结果汇总。** 完整词表各族平均 JS 为 `{json.dumps(compact_vocab, ensure_ascii=False)}`；总体账 `{json.dumps(result['vocabulary']['overall'], ensure_ascii=False)}`。问答—陈述候选行为逐行一致性 `{json.dumps(result['sequence_interface'], ensure_ascii=False)}`。最终输出逐坐标账 `{json.dumps(result['contributions'], ensure_ascii=False)}`；unit26 的 38 检查点逐坐标签名接口差分保存为 `{json.dumps({key: value for key, value in result['interface_coordinate_delta'].items() if key != 'normalized_l1_by_checkpoint'}, ensure_ascii=False)}`。陈述接口形成时序 `{json.dumps(formation, ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。本期最重要的科学纪律是没有把混杂项继续写成“表面等价”。接口 JS 大只说明输出分布对读出边界敏感；状态 JS 大只说明事实翻转改变预测竞争；二者都不是语义模块或因果齿轮。逐坐标 signed delta 是可观察差分，不赋予坐标固定语义。硬伤包括：纯表面负控缺失；问答与原始续写不仅改变指令，也改变边界 token 和长度；旧问答采用 chat template，新续写采用裸文本；因此“接口效应”是整个输入接口包的效应。基础全词表概率和逐坐标距离足够表达结果。脚本 `tests/glm5/phase2305_c3901_c4020_interface_state_accounting.py`；结果 `tests/glm5/result/phase2305_c3901_c4020_interface_state_accounting`。下一步必须另立前瞻修复：让事实顺序只由 unit 决定，使同 unit 两表面保持相同提及顺序，再复跑，不回改本期数字。
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
    parent = json.loads((P2304 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2304 incomplete")
    old_rows, new_rows = contract.read_rows(OLD_ROWS), contract.read_rows(NEW_ROWS)
    old_logits, new_logits = np.load(OLD_LOGITS, mmap_mode="r"), np.load(NEW_LOGITS, mmap_mode="r")
    pairs = make_pairs(old_rows, new_rows)
    interface = js_pairs(old_logits, new_logits, pairs["interface"])
    state = js_pairs(new_logits, new_logits, pairs["state"])
    surface_order = js_pairs(new_logits, new_logits, pairs["surface_order_bundle"])
    pair_rows = interface + state + surface_order
    contract.write_rows(OUT / "probability/full_vocabulary_pair_distances.jsonl", pair_rows)
    vocabulary = group_vocabulary(pair_rows)
    contract.save(OUT / "probability/full_vocabulary_summary.json", vocabulary)
    sequence = sequence_interface(old_rows, new_rows)
    contributions = contribution_accounting(old_rows, new_rows, pairs)
    delta = interface_coordinate_delta(new_rows)
    timing_result = timing(new_rows)
    confound = all(
        new_rows[row["left_index"]]["target_mention_order"] !=
        new_rows[row["right_index"]]["target_mention_order"]
        for row in pairs["surface_order_bundle"]
    )
    checks = {
        "old_new_row_identity": len(old_rows) == len(new_rows) == 1536,
        "interface_pairs_all_rows": len(interface) == 1536,
        "interface_source_order_matched_half": sum(
            row["pair_type"] == "interface_source_order_matched" for row in interface
        ) == 768,
        "state_pairs_complete": len(state) == 768,
        "surface_pairs_complete_but_confound_flagged": len(surface_order) == 768 and confound,
        "full_vocabulary_used": old_logits.shape[1] == new_logits.shape[1] == 151936,
        "unit26_all_checkpoint_coordinate_delta": delta["shape"] == [1824, 2560],
        "no_surface_only_claim": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "vocabulary": vocabulary, "sequence_interface": sequence,
        "contributions": contributions, "interface_coordinate_delta": delta,
        "timing": timing_result, "surface_order_confound": {
            "detected_after_unblinding": True,
            "all_same_unit_surface_pairs_reverse_target_mention_order": confound,
            "valid_name": "surface_plus_opposite_mention_order",
            "invalid_name": "matched_surface_equivalence",
        },
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            "The raw continuation interface changes complete-vocabulary competition and exact-coordinate states, "
            "while state flips remain measurable; however, the frozen surface comparison is inseparably confounded "
            "with mention order and provides no pure surface-equivalence evidence."
        ),
        "next_authorization": (
            "Freeze a corrected unit-only mention-order quota before another model load, rerun Qwen3-4B, and require "
            "same-unit surfaces to share mention order before any surface/state/interface comparison or Qwen3-14B test."
        ),
    }
    contract.save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
