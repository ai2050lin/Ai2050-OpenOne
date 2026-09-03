#!/usr/bin/env python3
"""Automatic successor: test full-token role trajectories across language and direction."""
from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2436 = RESULT / "phase2436_c34161_c34480_qwen4b_hypergraph_fullfield"
OUT = RESULT / "phase2441_c35761_c36080_alltoken_crosslanguage_trajectory"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2441
CAMPAIGN = "C35761-C36080"
ZONE_BINS = {"context": 12, "query": 8, "candidates": 8, "answer": 4}
MEASURES = ("role_energy", "language_coordinate_cosine", "language_shift791_cosine",
            "language_permuted_cosine", "language_family_gram_correlation",
            "language_permuted_gram_correlation", "direction_coordinate_cosine")
SHIFT = 791
ANALYSIS_VERSION = "v2_paired_equal_shape_bf16_recapture"

sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


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


def normalized_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-30)


def token_bins(row: dict, valid_length: int) -> tuple[list[int], list[dict]]:
    event = dict(zip(row["event_names"], row["event_token_indices"]))
    context_end = int(event["context_end"])
    query_end = int(event["query_end"])
    answer_boundary = int(event["answer_boundary"])
    ranges = {"context": (0, context_end),
              "query": (context_end + 1, query_end),
              "candidates": (query_end + 1, answer_boundary),
              "answer": (answer_boundary + 1, valid_length - 1)}
    indices, metadata = [], []
    cursor = 0
    for zone, count in ZONE_BINS.items():
        start, end = ranges[zone]
        if end < start:
            start = end = max(0, min(valid_length - 1, end))
        chosen = np.rint(np.linspace(start, end, count)).astype(np.int64)
        chosen = np.clip(chosen, 0, valid_length - 1)
        for local, token_index in enumerate(chosen):
            indices.append(int(token_index))
            metadata.append({"bin": cursor, "zone": zone, "zone_bin": local,
                             "normalized_position": local / max(count - 1, 1), "token_index": int(token_index)})
            cursor += 1
    return indices, metadata


def pair_index(rows: list[dict]) -> tuple[list[dict], list[tuple[int, int]]]:
    configs = sorted({row["config_id"] for row in rows})
    lookup = {(row["config_id"], row["query_role"]): index for index, row in enumerate(rows)}
    meta, pairs = [], []
    for config in configs:
        source, target = lookup[(config, "source")], lookup[(config, "target")]
        row = rows[source]
        meta.append({key: row[key] for key in ("config_id", "family", "unit", "language", "surface", "direction", "partition")})
        pairs.append((source, target))
    return meta, pairs


def build_role_difference(rows: list[dict], meta: list[dict], pairs: list[tuple[int, int]]) -> tuple[Path, dict]:
    bins = sum(ZONE_BINS.values())
    shape = (len(pairs), 38, bins, 2560)
    path = OUT / "derived/normalized_token_role_difference.float16.npy"
    progress = OUT / "derived/normalized_token_role_difference_progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    progress_payload = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else {}
    if path.exists() and progress_payload.get("analysis_version") == ANALYSIS_VERSION:
        output = np.lib.format.open_memmap(path, mode="r+")
        completed = int(progress_payload["completed"])
        if tuple(output.shape) != shape:
            raise RuntimeError(("stale_shape", output.shape, shape))
    else:
        output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        completed = 0
    model = tokenizer = None
    modules = captures = handles = None
    if completed < len(pairs):
        model, tokenizer, _ = capability.load_model("qwen4b")
        modules = field_utils.modules(model)
        captures = {}
        handles = []
        for qpoint, module in enumerate(modules):
            def hook(_module, _inputs, result, qpoint=qpoint):
                captures[qpoint] = (result[0] if isinstance(result, tuple) else result).detach()
            handles.append(module.register_forward_hook(hook))
        device = model.get_input_embeddings().weight.device
        pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    token_meta = []
    try:
        with torch.inference_mode():
            for pair_id, (source, target) in enumerate(pairs):
                source_sequence = rows[source]["prompt_ids"] + rows[source]["target_ids"]
                target_sequence = rows[target]["prompt_ids"] + rows[target]["target_ids"]
                source_indices, source_meta = token_bins(rows[source], len(source_sequence))
                target_indices, target_meta = token_bins(rows[target], len(target_sequence))
                token_meta.append({"config_id": meta[pair_id]["config_id"], "source_bins": source_meta,
                                   "target_bins": target_meta})
                if pair_id < completed:
                    continue
                width = max(len(source_sequence), len(target_sequence))
                ids = torch.full((2, width), pad, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                ids[0, :len(source_sequence)] = torch.tensor(source_sequence, dtype=torch.long, device=device)
                ids[1, :len(target_sequence)] = torch.tensor(target_sequence, dtype=torch.long, device=device)
                mask[0, :len(source_sequence)] = 1; mask[1, :len(target_sequence)] = 1
                positions = mask.long().cumsum(-1) - 1; positions.masked_fill_(mask == 0, 0)
                captures.clear()
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                for qpoint in range(len(modules)):
                    source_state = captures[qpoint][0, torch.tensor(source_indices, dtype=torch.long, device=device)]
                    target_state = captures[qpoint][1, torch.tensor(target_indices, dtype=torch.long, device=device)]
                    output[pair_id, qpoint] = (target_state - source_state).float().cpu().numpy().astype(np.float16)
                output.flush(); save(progress, {"analysis_version": ANALYSIS_VERSION,
                                                "completed": pair_id + 1, "shape": shape})
                if (pair_id + 1) % 8 == 0 or pair_id + 1 == len(pairs):
                    print(f"[phase2441 paired recapture] {pair_id + 1}/{len(pairs)}", flush=True)
    finally:
        if handles:
            for handle in handles: handle.remove()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    write_rows(OUT / "index/normalized_token_pairs.jsonl", token_meta)
    output.flush(); close(output)
    return path, {"shape": list(shape), "bytes": path.stat().st_size, "bins": bins,
                  "zones": ZONE_BINS, "analysis_version": ANALYSIS_VERSION,
                  "capture": "source/target paired in one equal-width right-padded BF16 CUDA forward",
                  "storage": "float16; every physical coordinate retained"}


def analyze(path: Path, meta: list[dict]) -> dict:
    values = np.load(path, mmap_mode="r")
    pairs, qpoints, bins, dim = values.shape
    families = sorted({row["family"] for row in meta})
    family_array = np.asarray([row["family"] for row in meta], dtype=object)
    language = np.asarray([row["language"] for row in meta], dtype=object)
    direction = np.asarray([int(row["direction"]) for row in meta])
    permutation = np.random.default_rng(2441).permutation(len(families))
    upper = np.triu_indices(len(families), 1)
    metrics = np.zeros((qpoints, bins, len(MEASURES)), dtype=np.float32)
    for qpoint in range(qpoints):
        for token_bin in range(bins):
            state = np.asarray(values[:, qpoint, token_bin], dtype=np.float32)
            en = np.stack([state[(family_array == family) & (language == "en")].mean(0) for family in families])
            zh = np.stack([state[(family_array == family) & (language == "zh")].mean(0) for family in families])
            d0 = np.stack([state[(family_array == family) & (direction == 0)].mean(0) for family in families])
            d1 = np.stack([state[(family_array == family) & (direction == 1)].mean(0) for family in families])
            en_n, zh_n = normalized_rows(en), normalized_rows(zh)
            gram_en, gram_zh = en_n @ en_n.T, zh_n @ zh_n.T
            metrics[qpoint, token_bin] = (
                float(np.mean(state.astype(np.float64) ** 2)),
                float(np.mean([cosine(en[index], zh[index]) for index in range(len(families))])),
                float(np.mean([cosine(en[index], np.roll(zh[index], SHIFT)) for index in range(len(families))])),
                float(np.mean([cosine(en[index], zh[permutation[index]]) for index in range(len(families))])),
                correlation(gram_en[upper], gram_zh[upper]),
                correlation(gram_en[upper], gram_zh[np.ix_(permutation, permutation)][upper]),
                float(np.mean([cosine(d0[index], d1[index]) for index in range(len(families))])),
            )
        if (qpoint + 1) % 6 == 0 or qpoint + 1 == qpoints:
            print(f"[phase2441 atlas] qpoint={qpoint + 1}/{qpoints}", flush=True)
    derived = OUT / "derived"
    np.save(derived / "alltoken_crosslanguage_metrics.float32.npy", metrics)
    query_start = ZONE_BINS["context"]
    post = metrics[:, query_start:]
    physical = post[:, :, MEASURES.index("language_coordinate_cosine")] - post[:, :, MEASURES.index("language_shift791_cosine")]
    best_q, best_bin = np.unravel_index(int(np.argmax(physical)), physical.shape)
    best_bin += query_start
    best_state = np.asarray(values[:, best_q, best_bin], dtype=np.float32)
    coord_rms = np.sqrt(np.mean(best_state.astype(np.float64) ** 2, axis=0))
    order = np.argsort(coord_rms, kind="stable")
    quartiles = np.array_split(order, 4)
    en_best = np.stack([best_state[(family_array == family) & (language == "en")].mean(0) for family in families])
    zh_best = np.stack([best_state[(family_array == family) & (language == "zh")].mean(0) for family in families])
    quartile_cos = {f"q{index + 1}_low_to_high": float(np.mean([cosine(en_best[fi, coordinates], zh_best[fi, coordinates])
                                                                  for fi in range(len(families))]))
                    for index, coordinates in enumerate(quartiles)}
    np.save(derived / "best_cell_coordinate_rms.float64.npy", coord_rms)
    # Adjacent block persistence excludes q36->q37 final-norm interface.
    layer_persistence = []
    for qpoint in range(36):
        layer_persistence.append(float(np.mean([cosine(values[pair, qpoint, best_bin], values[pair, qpoint + 1, best_bin])
                                                for pair in range(pairs)])))
    token_persistence = []
    for token_bin in range(query_start, bins - 1):
        token_persistence.append(float(np.mean([cosine(values[pair, best_q, token_bin], values[pair, best_q, token_bin + 1])
                                                for pair in range(pairs)])))
    pre_energy = metrics[:, :query_start, MEASURES.index("role_energy")]
    post_energy = metrics[:, query_start:, MEASURES.index("role_energy")]
    summary = {
        "prequery_role_energy_max": float(pre_energy.max()), "postquery_role_energy_mean": float(post_energy.mean()),
        "causal_prequery_to_postquery_energy_ratio": float(pre_energy.max() / max(post_energy.mean(), 1e-30)),
        "postquery_language_coordinate_cosine": float(post[:, :, MEASURES.index("language_coordinate_cosine")].mean()),
        "postquery_language_shift791_cosine": float(post[:, :, MEASURES.index("language_shift791_cosine")].mean()),
        "postquery_language_physical_advantage": float(physical.mean()),
        "postquery_language_permuted_cosine": float(post[:, :, MEASURES.index("language_permuted_cosine")].mean()),
        "postquery_family_gram_correlation": float(post[:, :, MEASURES.index("language_family_gram_correlation")].mean()),
        "postquery_permuted_gram_correlation": float(post[:, :, MEASURES.index("language_permuted_gram_correlation")].mean()),
        "postquery_direction_coordinate_cosine": float(post[:, :, MEASURES.index("direction_coordinate_cosine")].mean()),
        "best_physical_cell": {"qpoint": int(best_q), "token_bin": int(best_bin),
                               "zone": next(zone for zone, (start, end) in zone_bounds().items() if start <= best_bin < end),
                               "coordinate_cosine": float(metrics[best_q, best_bin, 1]),
                               "shift791_cosine": float(metrics[best_q, best_bin, 2]),
                               "physical_advantage": float(metrics[best_q, best_bin, 1] - metrics[best_q, best_bin, 2]),
                               "gram_correlation": float(metrics[best_q, best_bin, 4])},
        "best_cell_rms_quartile_language_cosine": quartile_cos,
        "best_bin_adjacent_block_persistence_mean": float(np.mean(layer_persistence)),
        "best_qpoint_adjacent_token_persistence_mean": float(np.mean(token_persistence)),
    }
    close(values)
    return {"analysis_version": ANALYSIS_VERSION, "pairs": pairs, "qpoints": qpoints, "bins": bins, "dimension": dim, "families": families,
            "measures": MEASURES, "summary": summary,
            "files": {"role_difference": str(path), "metrics": str(derived / "alltoken_crosslanguage_metrics.float32.npy"),
                      "best_coordinate_rms": str(derived / "best_cell_coordinate_rms.float64.npy")}}


def zone_bounds() -> dict[str, tuple[int, int]]:
    result, cursor = {}, 0
    for zone, count in ZONE_BINS.items():
        result[zone] = (cursor, cursor + count); cursor += count
    return result


def append_memo(result: dict) -> None:
    memo_text = MEMO.read_text(encoding="utf-8")
    if f"## Phase {PHASE}:" in memo_text and "Phase 2441 等形状成对前向质量修正" in memo_text:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    if f"## Phase {PHASE}:" in memo_text:
        text = rf"""

### Phase 2441 等形状成对前向质量修正 [{stamp}]

首次分析使用Phase2436逐行、变长序列前向的全token归档，query之前角色差能量不为零。原始same-token对查表明：某些长度不同的前向在CUDA attention内核中从q1出现微小数值差，深层被放大，这是内核形状混淆，不是模型提前读到query。v2将每个source/target对放在同一右填充BF16 batch内重新前向，保证相同内核形状；后续只使用v2。修正结果 `{json.dumps(result, ensure_ascii=False)}`。
"""
    else:
        text = rf"""

## Phase {PHASE}: 自动续研——全token条件轨迹的跨语言坐标与关系几何裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 大阶段完成后目标仍是破解八种语言模式族的条件坐标轨迹，因此自动续研。使用Phase2436保留的64条fresh-unit/valid/natural prompt+answer全token场；按context/query/candidates/answer四因果区间分别以12/8/8/4个归一化位置取真实最近token，不做平均压缩，每点保留38 checkpoint×2560坐标。对32个同配置source/target查询对构造角色差，比较同family中英同坐标余弦、+791坐标错配、family标签置乱、family×family Gram几何相关及方向复用。

$$R_{{f,l,d,q,b,j}}=H_{{target}}(q,t_b,j)-H_{{source}}(q,t_b,j),$$
$$c_{{coord}}=\frac18\sum_f\cos(\bar R_{{f,en}},\bar R_{{f,zh}}),\quad
c_{{shift}}=\frac18\sum_f\cos(\bar R_{{f,en}},\Pi_{{791}}\bar R_{{f,zh}}),$$
$$G_l=\widehat R_l\widehat R_l^\top,\qquad g=\operatorname{{corr}}(\operatorname{{vech}}G_{{en}},\operatorname{{vech}}G_{{zh}}).$$

**结果汇总。** 全坐标场 `{json.dumps(result['field'], ensure_ascii=False)}`；摘要 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2441_c35761_c36080_alltoken_crosslanguage_trajectory.py`；32×38×32×2560完整角色差、token对齐索引、逐层逐token跨语言/方向指标与坐标RMS位于同名结果目录。

**分析与理论进展。** 这一步直接裁决Phase2438的跨语言失败是否只是八个事件过稀：若全token同坐标仍弱而family关系Gram明显高于置乱，则更像语言相关坐标重参数化下保留关系几何；若两者都弱，则当前材料没有提供共享机制。context区角色差应因自回归因果性接近零，它同时是对齐与泄漏检查。

**问题硬伤与结论。** 只有unit5且每个family×语言×方向一个配置，统计自由度有限；归一化位置对不同token长度是最近邻对齐，不是精确词义对齐。Gram相关会忽略共同正交旋转，属于关系层证据而非物理坐标机制。final norm与block状态分栏；任何几何阳性都不能单独闭合输出或因果机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8"))
        if result.get("analysis", {}).get("analysis_version") == ANALYSIS_VERSION:
            append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2436 / "index/fresh_valid_all_token_rows.jsonl")
    meta, pairs = pair_index(rows)
    write_rows(OUT / "index/role_pair_configurations.jsonl", meta)
    path, field = build_role_difference(rows, meta, pairs)
    analysis = analyze(path, meta)
    summary = analysis["summary"]
    adjudication = {"causal_alignment_passed": summary["causal_prequery_to_postquery_energy_ratio"] < 1e-6,
                    "same_coordinate_crosslanguage_above_shift": summary["postquery_language_physical_advantage"] > 0,
                    "same_family_crosslanguage_above_permuted": summary["postquery_language_coordinate_cosine"] > summary["postquery_language_permuted_cosine"],
                    "family_geometry_above_permuted": summary["postquery_family_gram_correlation"] > summary["postquery_permuted_gram_correlation"],
                    "language_reparameterized_shared_geometry_detected": summary["postquery_family_gram_correlation"] > .2 and
                                                                         summary["postquery_family_gram_correlation"] > summary["postquery_permuted_gram_correlation"],
                    "universal_language_coordinate_mechanism_closed": False}
    checks = {"rows_64": len(rows) == 64, "pairs_32": analysis["pairs"] == 32,
              "shape": field["shape"] == [32, 38, 32, 2560], "full_coordinates": analysis["dimension"] == 2560,
              "four_zones": set(field["zones"]) == set(ZONE_BINS),
              "all_files": all(Path(path).exists() for path in analysis["files"].values()),
              "finite": all(math.isfinite(value) for value in summary.values() if isinstance(value, float)),
              "source_raw_retained": (P2436 / "raw/fresh_valid_prompt_answer_all_token.float16.npy").exists(),
              "claim_boundary": not adjudication["universal_language_coordinate_mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "field": field, "analysis": analysis,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
