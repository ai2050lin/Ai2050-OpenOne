#!/usr/bin/env python3
"""Decompose all-token cross-language reuse into embedding, contextual residual, and block update."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2441 = RESULT / "phase2441_c35761_c36080_alltoken_crosslanguage_trajectory"
OUT = RESULT / "phase2442_c36081_c36400_embedding_context_update_decomposition"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2442
CAMPAIGN = "C36081-C36400"
COMPONENTS = ("raw_state", "embedding_subtracted", "block_update")
MEASURES = ("language_coordinate", "language_shift791", "language_permuted",
            "family_gram", "family_gram_permuted", "direction_coordinate", "energy")
SHIFT = 791
ZONE_BOUNDS = {"context": (0, 12), "query": (12, 20), "candidates": (20, 28), "answer": (28, 32)}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-30 else 0.0


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    if len(a) < 2 or float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cell_metrics(state: np.ndarray, family_array: np.ndarray, language: np.ndarray,
                 direction: np.ndarray, families: list[str], permutation: np.ndarray,
                 upper: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    en = np.stack([state[(family_array == family) & (language == "en")].mean(0) for family in families])
    zh = np.stack([state[(family_array == family) & (language == "zh")].mean(0) for family in families])
    d0 = np.stack([state[(family_array == family) & (direction == 0)].mean(0) for family in families])
    d1 = np.stack([state[(family_array == family) & (direction == 1)].mean(0) for family in families])
    en_n = en / np.maximum(np.linalg.norm(en, axis=1, keepdims=True), 1e-30)
    zh_n = zh / np.maximum(np.linalg.norm(zh, axis=1, keepdims=True), 1e-30)
    gram_en, gram_zh = en_n @ en_n.T, zh_n @ zh_n.T
    return np.asarray((
        np.mean([cosine(en[i], zh[i]) for i in range(len(families))]),
        np.mean([cosine(en[i], np.roll(zh[i], SHIFT)) for i in range(len(families))]),
        np.mean([cosine(en[i], zh[permutation[i]]) for i in range(len(families))]),
        correlation(gram_en[upper], gram_zh[upper]),
        correlation(gram_en[upper], gram_zh[np.ix_(permutation, permutation)][upper]),
        np.mean([cosine(d0[i], d1[i]) for i in range(len(families))]),
        np.mean(state.astype(np.float64) ** 2),
    ), dtype=np.float32)


def analyze(meta: list[dict]) -> dict:
    source_path = P2441 / "derived/normalized_token_role_difference.float16.npy"
    values = np.load(source_path, mmap_mode="r")
    pairs, qpoints, bins, dim = values.shape
    families = sorted({row["family"] for row in meta})
    family_array = np.asarray([row["family"] for row in meta], dtype=object)
    language = np.asarray([row["language"] for row in meta], dtype=object)
    direction = np.asarray([int(row["direction"]) for row in meta])
    permutation = np.random.default_rng(2442).permutation(len(families))
    upper = np.triu_indices(len(families), 1)
    # q axis is padded to 38. block_update only uses q0->q1 ... q35->q36;
    # q36->q37 final norm is excluded.
    metrics = np.full((len(COMPONENTS), qpoints, bins, len(MEASURES)), np.nan, dtype=np.float32)
    for qpoint in range(qpoints):
        for token_bin in range(bins):
            raw = np.asarray(values[:, qpoint, token_bin], dtype=np.float32)
            metrics[0, qpoint, token_bin] = cell_metrics(raw, family_array, language, direction, families, permutation, upper)
            if qpoint >= 1:
                embedding_subtracted = raw - np.asarray(values[:, 0, token_bin], dtype=np.float32)
                metrics[1, qpoint, token_bin] = cell_metrics(embedding_subtracted, family_array, language, direction,
                                                             families, permutation, upper)
            if qpoint < 36:
                update = np.asarray(values[:, qpoint + 1, token_bin], dtype=np.float32) - raw
                metrics[2, qpoint, token_bin] = cell_metrics(update, family_array, language, direction,
                                                             families, permutation, upper)
        if (qpoint + 1) % 6 == 0 or qpoint + 1 == qpoints:
            print(f"[phase2442] qpoint={qpoint + 1}/{qpoints}", flush=True)
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    np.save(derived / "embedding_context_update_metrics.float32.npy", metrics)
    summary = {}
    for ci, component in enumerate(COMPONENTS):
        qslice = slice(1, 37) if component != "block_update" else slice(0, 36)
        summary[component] = {}
        for zone, (start, end) in ZONE_BOUNDS.items():
            block = metrics[ci, qslice, start:end]
            summary[component][zone] = {measure: float(np.nanmean(block[..., mi]))
                                        for mi, measure in enumerate(MEASURES)}
            summary[component][zone]["physical_advantage"] = float(
                summary[component][zone]["language_coordinate"] - summary[component][zone]["language_shift791"])
            summary[component][zone]["family_identity_advantage"] = float(
                summary[component][zone]["language_coordinate"] - summary[component][zone]["language_permuted"])
            summary[component][zone]["gram_identity_advantage"] = float(
                summary[component][zone]["family_gram"] - summary[component][zone]["family_gram_permuted"])
    # Explicit embedding-only token identity baseline (q0).
    embedding_baseline = {zone: {measure: float(np.nanmean(metrics[0, 0, start:end, mi]))
                                 for mi, measure in enumerate(MEASURES)}
                          for zone, (start, end) in ZONE_BOUNDS.items()}
    # Find the strongest contextual query cell, excluding q0 and final norm.
    start, end = ZONE_BOUNDS["query"]
    residual = metrics[1, 1:37, start:end]
    physical = residual[..., MEASURES.index("language_coordinate")] - residual[..., MEASURES.index("language_shift791")]
    best_q0, best_b0 = np.unravel_index(int(np.nanargmax(physical)), physical.shape)
    best_qpoint, best_bin = best_q0 + 1, best_b0 + start
    best_state = np.asarray(values[:, best_qpoint, best_bin], dtype=np.float32) - np.asarray(values[:, 0, best_bin], dtype=np.float32)
    rms = np.sqrt(np.mean(best_state.astype(np.float64) ** 2, axis=0))
    order = np.argsort(rms, kind="stable"); quartiles = np.array_split(order, 4)
    en = np.stack([best_state[(family_array == family) & (language == "en")].mean(0) for family in families])
    zh = np.stack([best_state[(family_array == family) & (language == "zh")].mean(0) for family in families])
    quartile = {f"q{qi + 1}_low_to_high": float(np.mean([cosine(en[fi, coordinates], zh[fi, coordinates])
                                                          for fi in range(len(families))]))
                for qi, coordinates in enumerate(quartiles)}
    np.save(derived / "best_contextual_query_coordinate_rms.float64.npy", rms)
    best = {"qpoint": int(best_qpoint), "token_bin": int(best_bin),
            "language_coordinate": float(metrics[1, best_qpoint, best_bin, 0]),
            "language_shift791": float(metrics[1, best_qpoint, best_bin, 1]),
            "language_permuted": float(metrics[1, best_qpoint, best_bin, 2]),
            "family_gram": float(metrics[1, best_qpoint, best_bin, 3]),
            "family_gram_permuted": float(metrics[1, best_qpoint, best_bin, 4]),
            "rms_quartile_language_cosine": quartile}
    close(values)
    return {"pairs": pairs, "qpoints": qpoints, "bins": bins, "dimension": dim,
            "components": COMPONENTS, "measures": MEASURES, "zone_summary": summary,
            "embedding_token_identity_baseline": embedding_baseline,
            "best_contextual_query_cell": best,
            "files": {"metrics": str(derived / "embedding_context_update_metrics.float32.npy"),
                      "best_rms": str(derived / "best_contextual_query_coordinate_rms.float64.npy")}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 自动续研——跨语言复用的embedding基线—上下文残差—block更新分解（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2441最强中英对应落在共享`Pira/Rovan`答案token，故不能直接命名语义共享。本Phase对32对×38 checkpoint×32 token位置×2560坐标的角色差分成三栏：原始状态$R_q$、去embedding基线$R_q-R_0$、真实block更新$R_{{q+1}}-R_q$；最后一栏只含36个block，不含final norm。每栏分别报告query/candidates/answer区的同family中英坐标余弦、+791错配、family置乱、Gram/置乱Gram和方向复用。

$$R^{{ctx}}_{{q,b,j}}=R_{{q,b,j}}-R_{{0,b,j}},\qquad
U_{{q,b,j}}=R_{{q+1,b,j}}-R_{{q,b,j}},\quad q=0,\ldots,35.$$

$$\Delta_{{id}}=c(f_{{en}},f_{{zh}})-c(f_{{en}},\pi(f)_{{zh}}),\qquad
\Delta_{{phys}}=c_{{same\ coord}}-c_{{shift791}}.$$

**结果汇总。** embedding-only基线 `{json.dumps(result['analysis']['embedding_token_identity_baseline'], ensure_ascii=False)}`；分区分解 `{json.dumps(result['analysis']['zone_summary'], ensure_ascii=False)}`；最佳上下文query单元 `{json.dumps(result['analysis']['best_contextual_query_cell'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2442_c36081_c36400_embedding_context_update_decomposition.py`；全component×checkpoint×token-bin指标和最佳单元全2560坐标RMS位于同名结果目录；源全坐标角色差继续保留。

**分析与理论进展。** 这个分解把“同一个拉丁名称在中英文prompt里使用相同embedding”与“上下文计算后仍保留family条件纹理”分开。只有query区去embedding残差和block update同时在同family标签、真实坐标、跨方向上优于置乱，才是共享上下文编码候选；candidate/answer区的高原始余弦主要是token身份上界。

**问题硬伤与结论。** 减去$q0$不是严格分离embedding与上下文，因为残差网络会非线性混合；block update仍含通用层变换。family置乱只有一个冻结随机种子、每族样本少，本Phase是基础排雷，不是最终显著性证明。任何阳性仍需多unit、不同实体词和输出桥复验。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    meta = read_rows(P2441 / "index/role_pair_configurations.jsonl")
    analysis = analyze(meta)
    query_ctx = analysis["zone_summary"]["embedding_subtracted"]["query"]
    query_update = analysis["zone_summary"]["block_update"]["query"]
    answer_raw = analysis["zone_summary"]["raw_state"]["answer"]
    answer_embedding = analysis["embedding_token_identity_baseline"]["answer"]
    adjudication = {"answer_raw_reuse_has_large_embedding_baseline": answer_embedding["language_coordinate"] > .5 and
                                                                 answer_raw["language_coordinate"] > .5,
                    "contextual_query_same_coordinate_above_shift": query_ctx["physical_advantage"] > 0,
                    "contextual_query_same_family_above_permuted": query_ctx["family_identity_advantage"] > 0,
                    "query_block_update_same_coordinate_above_shift": query_update["physical_advantage"] > 0,
                    "query_block_update_same_family_above_permuted": query_update["family_identity_advantage"] > 0,
                    "shared_contextual_coordinate_candidate": query_ctx["physical_advantage"] > 0 and
                                                                query_ctx["family_identity_advantage"] > 0 and
                                                                query_update["physical_advantage"] > 0 and
                                                                query_update["family_identity_advantage"] > 0,
                    "universal_language_encoding_mechanism_closed": False}
    checks = {"pairs_32": analysis["pairs"] == 32, "dimension_2560": analysis["dimension"] == 2560,
              "three_components": set(analysis["components"]) == set(COMPONENTS),
              "four_zones": all(set(component) == set(ZONE_BOUNDS) for component in analysis["zone_summary"].values()),
              "all_files": all(Path(path).exists() for path in analysis["files"].values()),
              "finite": all(math.isfinite(value) for component in analysis["zone_summary"].values()
                            for zone in component.values() for value in zone.values()),
              "source_retained": (P2441 / "derived/normalized_token_role_difference.float16.npy").exists(),
              "claim_boundary": not adjudication["universal_language_encoding_mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
