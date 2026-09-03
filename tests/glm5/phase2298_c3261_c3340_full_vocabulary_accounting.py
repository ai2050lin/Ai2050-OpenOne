#!/usr/bin/env python3
"""Basic full-vocabulary accounting for the NTP predictive-state field."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT_OUT = RESULT / "phase2296_c3101_c3160_ntp_predictive_contract"
PARENT = RESULT / "phase2297_c3161_c3260_qwen4b_ntp_predictive_field"
OUT = RESULT / "phase2298_c3261_c3340_full_vocabulary_accounting"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
LOGITS = PARENT / "raw/qwen4b_ntp_full_vocabulary_logits.float16.npy"
sys.path.insert(0, str(TESTS))

import phase2296_c3101_c3160_ntp_predictive_contract as contract  # noqa: E402


PHASE = 2298
CAMPAIGN = "C3261-C3340"
EPS = 1e-12


def probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    values -= np.max(values)
    output = np.exp(values)
    output /= max(float(output.sum()), EPS)
    return output


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)
    return float(0.5 * np.sum(left * (np.log(left + EPS) - np.log(midpoint + EPS))) +
                 0.5 * np.sum(right * (np.log(right + EPS) - np.log(midpoint + EPS))))


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max())}


def row_metrics(rows: list[dict], logits: np.ndarray) -> tuple[list[dict], list[np.ndarray]]:
    metrics = []
    cached = []
    for i, row in enumerate(rows):
        values = np.asarray(logits[i], dtype=np.float32)
        p = probabilities(values)
        cached.append(p)
        target = int(row["ntp_target_ids"][0])
        wrong = int(row["ntp_wrong_ids"][0])
        target_logit = float(values[target])
        wrong_logit = float(values[wrong])
        metrics.append({
            "row": i, "case_id": row["case_id"], "family": row["family"],
            "language": row["language"], "surface": row["surface"],
            "partition": row["partition"], "unit": row["unit"], "state": row["state"],
            "entropy": float(-np.sum(p * np.log(p + EPS))),
            "target_probability": float(p[target]), "wrong_probability": float(p[wrong]),
            "target_wrong_logit_margin": target_logit - wrong_logit,
            "first_token_correct": target_logit > wrong_logit,
            "target_rank": int(1 + np.count_nonzero(values > target_logit)),
            "wrong_rank": int(1 + np.count_nonzero(values > wrong_logit)),
            "top_token_id": int(np.argmax(values)), "top_probability": float(np.max(p)),
        })
    return metrics, cached


def pair_metrics(rows: list[dict], probabilities_by_row: list[np.ndarray]) -> list[dict]:
    by_key = {(row["family"], row["language"], row["surface"], int(row["unit"]), int(row["state"])): i
              for i, row in enumerate(rows)}
    output = []
    for family in contract.FAMILIES:
        for language in ("en", "zh"):
            for surface in ("narrative", "dialogue"):
                for unit in range(32):
                    left = by_key[(family, language, surface, unit, 0)]
                    right = by_key[(family, language, surface, unit, 1)]
                    output.append({
                        "comparison": "state_flip", "family": family, "language": language,
                        "surface": surface, "unit": unit, "partition": rows[left]["partition"],
                        "left_case": rows[left]["case_id"], "right_case": rows[right]["case_id"],
                        "js": js_divergence(probabilities_by_row[left], probabilities_by_row[right]),
                    })
            for state in (0, 1):
                for unit in range(32):
                    left = by_key[(family, language, "narrative", unit, state)]
                    right = by_key[(family, language, "dialogue", unit, state)]
                    output.append({
                        "comparison": "surface_equivalent", "family": family, "language": language,
                        "state": state, "unit": unit, "partition": rows[left]["partition"],
                        "left_case": rows[left]["case_id"], "right_case": rows[right]["case_id"],
                        "js": js_divergence(probabilities_by_row[left], probabilities_by_row[right]),
                    })
    return output


def aggregate(rows: list[dict], metrics: list[dict], pairs: list[dict]) -> dict:
    output = {}
    for family in contract.FAMILIES:
        family_metrics = [row for row in metrics if row["family"] == family]
        state = [row["js"] for row in pairs if row["family"] == family and row["comparison"] == "state_flip"]
        surface = [row["js"] for row in pairs if row["family"] == family and row["comparison"] == "surface_equivalent"]
        partitions = {}
        for partition in contract.PARTITIONS:
            subset = [row for row in family_metrics if row["partition"] == partition]
            partitions[partition] = {
                "rows": len(subset),
                "first_token_accuracy": float(np.mean([row["first_token_correct"] for row in subset])),
                "mean_target_rank": float(np.mean([row["target_rank"] for row in subset])),
                "mean_entropy": float(np.mean([row["entropy"] for row in subset])),
            }
        output[family] = {
            "first_token_accuracy": float(np.mean([row["first_token_correct"] for row in family_metrics])),
            "mean_target_rank": float(np.mean([row["target_rank"] for row in family_metrics])),
            "mean_wrong_rank": float(np.mean([row["wrong_rank"] for row in family_metrics])),
            "target_probability": summarize([row["target_probability"] for row in family_metrics]),
            "wrong_probability": summarize([row["wrong_probability"] for row in family_metrics]),
            "entropy": summarize([row["entropy"] for row in family_metrics]),
            "state_flip_js": summarize(state),
            "surface_equivalent_js": summarize(surface),
            "surface_advantage": float(np.mean(state) - np.mean(surface)),
            "partitions": partitions,
        }
    return output


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {family: {
        "first_token_accuracy": value["first_token_accuracy"],
        "mean_target_rank": value["mean_target_rank"],
        "state_flip_js": value["state_flip_js"]["mean"],
        "surface_equivalent_js": value["surface_equivalent_js"]["mean"],
        "surface_advantage": value["surface_advantage"],
    } for family, value in result["families"].items()}
    text = rf"""

## Phase {PHASE}: 完整词表竞争、状态变化与表面等价分账（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 本期不拟合新模型，只读取 Phase2297 冻结的 `1536×151936` 实际 logits。每行计算完整词表熵、目标/错误第一token概率与排名；再对同一语言、同一构式、同一单元做两种严格匹配：保持表面不变而翻转事实状态，以及保持事实状态不变而改写 narrative/dialogue。跨语言词表分布没有直接比较，因为目标语言改变会合法地改变输出token支持集。

**公式。** 完整词表状态距离使用 Jensen-Shannon 散度：

$$
D_{{JS}}(P,Q)=\frac12D_{{KL}}(P\|M)+\frac12D_{{KL}}(Q\|M),
\qquad M=\frac12(P+Q).
$$

本期基础判别量为：

$$
A_f=\mathbb E[D_{{JS}}(P_{{s=0}},P_{{s=1}})]
-\mathbb E[D_{{JS}}(P_{{narr}},P_{{dial}})].
$$

**结果汇总。** 六构式摘要 `{json.dumps(compact, ensure_ascii=False)}`；总体 `{json.dumps(result['overall'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。逐样本概率账和全部匹配对保存在结果目录。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。状态翻转与表面改写都会改变完整词表分布，只有当 `A_f>0` 时，才能说该接口下状态改变平均大于合法表面变化；这仍不等于发现语义不变量。第一token只是完整答案的开头，长度均值与总序列分数仍以Phase2297为准。模型回答提示的风格、语言和答案长度都是概率分布的一部分，不能事后剥离后再宣称纯语义。脚本 `tests/glm5/phase2298_c3261_c3340_full_vocabulary_accounting.py`；结果 `tests/glm5/result/phase2298_c3261_c3340_full_vocabulary_accounting`。
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
    parent = json.loads((PARENT / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2297 did not pass execution checks")
    rows = contract.read_rows(CONTRACT_OUT / "material/ntp_natural_bilingual.jsonl")
    logits = np.load(LOGITS, mmap_mode="r")
    metrics, cached = row_metrics(rows, logits)
    pairs = pair_metrics(rows, cached)
    families = aggregate(rows, metrics, pairs)
    contract.write_rows(OUT / "probability/row_metrics.jsonl", metrics)
    contract.write_rows(OUT / "probability/pair_js.jsonl", pairs)
    overall = {
        "rows": len(metrics), "state_pairs": sum(row["comparison"] == "state_flip" for row in pairs),
        "surface_pairs": sum(row["comparison"] == "surface_equivalent" for row in pairs),
        "first_token_accuracy": float(np.mean([row["first_token_correct"] for row in metrics])),
        "families_state_above_surface": sum(value["surface_advantage"] > 0 for value in families.values()),
    }
    checks = {
        "all_logits_accounted": len(metrics) == len(rows) == logits.shape[0],
        "full_vocabulary_used": logits.shape[1] == parent["model"]["vocabulary"],
        "all_state_pairs": overall["state_pairs"] == len(contract.FAMILIES) * 2 * 2 * 32,
        "all_surface_pairs": overall["surface_pairs"] == len(contract.FAMILIES) * 2 * 2 * 32,
        "finite_metrics": all(np.isfinite(row["entropy"]) and np.isfinite(row["target_probability"]) for row in metrics),
        "no_cross_language_distribution_claim": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "families": families, "overall": overall, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            f"Complete-vocabulary accounting found state-flip JS larger than matched surface JS in "
            f"{overall['families_state_above_surface']}/6 families; this measures output-interface separation, "
            "not a language-independent semantic manifold."
        ),
        "files": {"rows": "probability/row_metrics.jsonl", "pairs": "probability/pair_js.jsonl"},
        "next_authorization": "Locate when target competition becomes readable and audit all-coordinate output contributions without coordinate selection.",
    }
    contract.save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
