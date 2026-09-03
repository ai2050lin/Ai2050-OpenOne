#!/usr/bin/env python3
"""Compare valid composition pairs with slot, broken-chain and same-answer query controls."""
from __future__ import annotations

import gc
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2415 = RESULT / "phase2415_c27441_c27760_exact_paired_composition"
OUT = RESULT / "phase2419_c28721_c29040_semantic_specificity_controls"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2419
CAMPAIGN = "C28721-C29040"
VARIANTS = ("valid_composition", "candidate_slot", "broken_chain", "same_answer_paraphrase")
COMPONENTS = ("total", "attention", "mlp")
SHUFFLES = 16

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2397_c21681_c22000_operation_behavior_token_calibration as behavior  # noqa: E402
import phase2405_c24241_c24560_deconfounded_operation_contract as contract  # noqa: E402
import phase2415_c27441_c27760_exact_paired_composition as paired  # noqa: E402
import phase2416_c27761_c28080_crossmodel_exact_pair_replication as capture_utils  # noqa: E402
import phase2418_c28401_c28720_heteroscedasticity_residual_control as controls  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def slot_query(language: str, index: int) -> str:
    if language == "en":
        return f"Ignore the relations and return candidate number {'one' if index == 1 else 'two'} only."
    return f"忽略上述关系，只返回第{'一' if index == 1 else '二'}个候选项。"


def paraphrase_query(language: str, source: str) -> str:
    if language == "en":
        return f"Begin at {source}, traverse exactly two stated links, and output only the candidate reached."
    return f"从{source}开始，沿陈述的关系恰好走两步，只输出到达的候选项。"


def build_prompt(row: dict, query: str, broken: bool = False) -> tuple[str, list[dict]]:
    first = contract.render_fact(row["family"], row["language"], row["surface"], row["source"], row["middle"])
    if broken:
        other = contract.triples(row["language"], (int(row["unit"]) + 1) % 8)
        distractor = other[0] if other[0] not in (row["source"], row["middle"], row["endpoint"]) else other[2]
        second = contract.render_fact(row["family"], row["language"], row["surface"], distractor, row["endpoint"])
    else:
        second = contract.render_fact(row["family"], row["language"], row["surface"], row["middle"], row["endpoint"])
    return contract.prior.prompt_with_events(row["language"], [first, second], query, row["candidates"])


def compile_material() -> list[dict]:
    source = read_rows(P2415 / "material/exact_paired_composition.jsonl")
    selected = [row for row in source if int(row["unit"]) < 8 and row["surface"] in ("canonical", "paraphrase")]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[row["pair_id"]].append(row)
    rows = []
    for original_pair_id in sorted(grouped):
        pair = sorted(grouped[original_pair_id], key=lambda row: row["steps"])
        if len(pair) != 2:
            raise RuntimeError((original_pair_id, len(pair)))
        step1, step2 = pair
        for variant in VARIANTS:
            variant_pair_id = f"{variant}-{original_pair_id}"
            for pair_step in (1, 2):
                base = step1 if pair_step == 1 else step2
                if variant == "valid_composition":
                    query, answer, foil, broken = base["query"], base["answer"], base["foil"], False
                elif variant == "candidate_slot":
                    query = slot_query(base["language"], pair_step)
                    answer, foil = base["candidates"][pair_step - 1], base["candidates"][2 - pair_step]
                    broken = False
                elif variant == "broken_chain":
                    query, answer, foil, broken = base["query"], base["answer"], base["foil"], True
                else:
                    query = step2["query"] if pair_step == 1 else paraphrase_query(base["language"], base["source"])
                    answer, foil, broken = step2["answer"], step2["foil"], False
                prompt, events = build_prompt(base, query, broken)
                item = {key: base[key] for key in ("task", "family", "unit", "language", "surface", "surface_class",
                                                            "direction", "candidate_order", "source", "middle", "endpoint", "candidates")}
                item.update({"case_id": f"{variant_pair_id}-s{pair_step}", "pair_id": variant_pair_id,
                             "original_pair_id": original_pair_id, "variant": variant, "steps": pair_step,
                             "target_candidate_slot": base["candidates"].index(answer),
                             "partition": "discovery" if int(base["unit"]) < 4 else "fresh_unit_lockbox",
                             "query": query, "answer": answer, "foil": foil, "prompt": prompt, "events": events})
                rows.append(item)
    return rows


def material_audit(rows: list[dict]) -> dict:
    result = {"rows": len(rows), "pairs": len({row["pair_id"] for row in rows}),
              "variants": dict(Counter(row["variant"] for row in rows)),
              "families": dict(Counter(row["family"] for row in rows)),
              "languages": dict(Counter(row["language"] for row in rows)),
              "surfaces": dict(Counter(row["surface"] for row in rows)),
              "partitions": dict(Counter(row["partition"] for row in rows)),
              "unique_cases": len({row["case_id"] for row in rows}) == len(rows)}
    exact = 0
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in subset:
            groups[row["pair_id"]].append(row)
        valid = all(len(pair) == 2 and sorted(row["steps"] for row in pair) == [1, 2] and
                    pair[0]["candidates"] == pair[1]["candidates"] and
                    pair[0]["language"] == pair[1]["language"] and
                    pair[0]["surface"] == pair[1]["surface"] and
                    pair[0]["direction"] == pair[1]["direction"]
                    for pair in groups.values())
        exact += int(valid and len(groups) == 256)
    result["four_exact_pair_panels"] = exact == 4
    return result


def behavior_summary(teacher: list[dict], compiled: list[dict]) -> dict:
    metadata = {row["case_id"]: row for row in compiled}
    result = {}
    for variant in VARIANTS:
        rows = [{**row, "variant": variant} for row in teacher if metadata[row["case_id"]]["variant"] == variant]
        pair_groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            pair_groups[row["case_id"].rsplit("-s", 1)[0]].append(row)
        pairs = list(pair_groups.values())
        result[variant] = {"rows": len(rows), "pairs": len(pairs),
                           "target_over_foil": float(np.mean([row["mean_logprob_margin"] > 0 for row in rows])),
                           "both_target_over_foil": float(np.mean([all(row["mean_logprob_margin"] > 0 for row in pair) for pair in pairs])),
                           "mean_margin": float(np.mean([row["mean_logprob_margin"] for row in rows]))}
    return result


def analyze(rows: list[dict], collection: dict) -> dict:
    state = np.load(collection["state"]["path"], mmap_mode="r")
    attention = np.load(collection["attention"]["path"], mmap_mode="r")
    mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    layers = state.shape[1]
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    metrics = np.zeros((len(VARIANTS), len(COMPONENTS), 8, layers), dtype=np.float32)
    # family, state, mismatch, shuffle_mean, shuffle_q95, state-over-shuffle, pair-energy, state layer win
    variant_results = {}
    for vi, variant in enumerate(VARIANTS):
        source_indices = np.asarray([i for i, row in enumerate(rows) if row["variant"] == variant], dtype=np.int64)
        local_rows = [rows[i] for i in source_indices]
        pair_rows, local_step1, local_step2 = paired.pair_index(local_rows)
        step1, step2 = source_indices[local_step1], source_indices[local_step2]
        families = np.asarray([row["family"] for row in pair_rows], dtype=object)
        train = np.asarray([i for i, row in enumerate(pair_rows) if row["partition"] == "discovery"], dtype=np.int64)
        test = np.asarray([i for i, row in enumerate(pair_rows) if row["partition"] == "fresh_unit_lockbox"], dtype=np.int64)
        for layer in range(layers):
            h = np.asarray(state[step2, layer], dtype=np.float32) - np.asarray(state[step1, layer], dtype=np.float32)
            a = np.asarray(attention[step2, layer], dtype=np.float32) - np.asarray(attention[step1, layer], dtype=np.float32)
            m = np.asarray(mlp[step2, layer], dtype=np.float32) - np.asarray(mlp[step1, layer], dtype=np.float32)
            for ci, y in enumerate((a + m, a, m)):
                values, _ = controls.predict_controls(pair_rows, train, test, families, h, y,
                                                     PHASE * 10000 + vi * 1000 + ci * 100 + layer)
                energy = float(np.mean(y[test] * y[test]))
                metrics[vi, ci, :, layer] = [values["family"], values["state"], values["coordinate_mismatch"],
                                             values["sample_shuffle_mean"], values["sample_shuffle_q95"],
                                             values["state"] - values["sample_shuffle_q95"], energy,
                                             float(values["state"] > values["sample_shuffle_q95"])]
        print(f"[phase2419 analysis] {variant} {layers}/{layers}", flush=True)
        variant_results[variant] = {"pairs": len(pair_rows), "train_pairs": len(train), "test_pairs": len(test)}
    np.save(derived / "semantic_specificity_layer_metrics.float32.npy", metrics)
    summary = {}
    keys = ("family_gain", "state_gain", "mismatch_gain", "shuffle_mean_gain", "shuffle_q95_gain",
            "state_over_shuffle_q95", "pair_update_energy", "layer_win_rate")
    for vi, variant in enumerate(VARIANTS):
        summary[variant] = {component: {key: float(metrics[vi, ci, ki].mean()) for ki, key in enumerate(keys)}
                            for ci, component in enumerate(COMPONENTS)}
    valid = summary["valid_composition"]
    specificity = {component: {control: valid[component]["state_over_shuffle_q95"] -
                                          summary[control][component]["state_over_shuffle_q95"]
                               for control in VARIANTS[1:]}
                   for component in COMPONENTS}
    close(state); close(attention); close(mlp)
    return {"variants": variant_results, "summary": summary, "valid_specificity_margin": specificity,
            "metrics": str(derived / "semantic_specificity_layer_metrics.float32.npy"),
            "shuffles_per_cell": SHUFFLES}


def cleanup(collection: dict) -> dict:
    paths, total = [], 0
    for value in collection.values():
        path = Path(value["path"])
        if path.exists():
            total += path.stat().st_size; paths.append(str(path)); path.unlink()
    return {"removed_files": len(paths), "removed_bytes": total, "removed_gib": total / 2**30,
            "recoverable": False, "paths": paths}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 有效组合对非语义查询控制的全坐标特异性竞赛（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2418的样本置乱阳性证明当前pair状态有预测力，但这可能是任何查询变化都会产生的通用残差动力学。本Phase冻结4族×unit0–7×中英×canonical/paraphrase×双方向，共256个配置，并为每个配置建立四个严格pair面板（每面板512条/256对，总2048条）：①有效一步/两步组合；②忽略关系、按候选槽1/2作答；③第二条事实断链但仍询问一步/两步；④两个不同查询表述都要求同一个两步答案。每个pair内部事实、候选、语言、表面、方向固定。在Qwen3-4B先做全部目标—foil评分，再于answer boundary采集36层状态、Attention、MLP全部2560坐标；unit0–3拟合，unit4–7冻结评价，每个单元以同族×语言×表面×方向16次样本置乱为对照。

$$S_v=\frac1L\sum_q\left[G_v(\widehat U_q(H_p))-Q_{{.95}}G_v(\widehat U_q(H_{{\pi(p)}}))\right],\qquad
\Delta_{{semantic,c}}=S_{{valid}}-S_c.$$

**结果汇总。** 材料 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；四面板全坐标结果 `{json.dumps(result['analysis']['summary'], ensure_ascii=False)}`；有效组合特异性 `{json.dumps(result['analysis']['valid_specificity_margin'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2419_c28721_c29040_semantic_specificity_controls.py`；2048条材料、模型token索引、行为输出、逐variant×组件×层指标和final位于`tests/glm5/result/phase2419_c28721_c29040_semantic_specificity_controls`。未修改其他Markdown。

**分析与理论进展。** 这不是问“状态是否预测更新”，而是问这种可预测性是否对有效关系组合有增量特异性。候选槽控制保留答案交换和输出角色，断链控制保留一步/两步查询词，same-answer控制保留查询改写但取消答案交换。若有效组合同时超过三者，才支持语义组合贡献；若相当或更弱，Phase2418的阳性应降格为一般查询—答案角色—残差耦合。

**问题硬伤与结论。** 四面板的查询长度/token并非严格逐token匹配；断链终点仍出现在事实和候选中；候选槽控制比关系推理更容易。能量与预测增益受任务难度影响，故主要比较matched超过同面板样本置乱的增量，而不是裸场幅度。Phase2418的z-score matched与mismatch完全相同是对角OLS的代数重参数化，不构成独立异方差证据，本Phase正式限定该结论。原始float16场派生后删除且不可恢复。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    source_path = OUT / "material/semantic_specificity_controls.jsonl"
    source = read_rows(source_path) if source_path.exists() else compile_material()
    if not source_path.exists():
        write_rows(source_path, source)
    audit = material_audit(source)
    model, tokenizer, label = capability.load_model("qwen4b")
    behavior.OUT = OUT; capture_utils.OUT = OUT
    try:
        index = OUT / "index/composition_rows.jsonl"
        if index.exists():
            rows = read_rows(index)
            calibration = json.loads((OUT / "analysis/token_calibration.json").read_text(encoding="utf-8"))
        else:
            rows, calibration = behavior.compile_rows(tokenizer, source)
            write_rows(index, rows); save(OUT / "analysis/token_calibration.json", calibration)
        teacher, teacher_all = behavior.score_rows("qwen4b", model, rows, 16)
        collection = capture_utils.collect("qwen4b", model, rows, 4)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    teacher_summary = behavior_summary(teacher, rows)
    analysis = analyze(rows, collection)
    raw_cleanup = cleanup(collection)
    margins = analysis["valid_specificity_margin"]
    adjudication = {"valid_exceeds_all_controls_total": all(value > 0 for value in margins["total"].values()),
                    "valid_exceeds_all_controls_all_components": all(value > 0 for component in margins.values() for value in component.values()),
                    "phase2418_zscore_is_independent_control": False,
                    "state_texture_is_semantic_composition_specific": False,
                    "composition_gear_proven": False}
    checks = {"four_panels_2048_rows": audit["rows"] == 2048 and audit["four_exact_pair_panels"],
              "token_calibration": calibration["rows"] == 2048 and calibration["event_monotonic_rate"] == 1.0,
              "teacher_complete": len(teacher) == 2048,
              "full_coordinates": collection["state"]["shape"] == [2048, 36, 2560],
              "sixteen_shuffles": analysis["shuffles_per_cell"] == 16,
              "finite": all(math.isfinite(value) for variant in analysis["summary"].values()
                            for component in variant.values() for value in component.values()),
              "raw_cleaned": raw_cleanup["removed_files"] == 3, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": label, "material_audit": audit,
              "token_calibration": calibration, "teacher_all": teacher_all, "behavior": teacher_summary,
              "collection": collection, "analysis": analysis, "cleanup": raw_cleanup,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps({"phase": PHASE, "material": audit, "behavior": teacher_summary,
                      "summary": analysis["summary"], "specificity": margins,
                      "adjudication": adjudication, "cleanup": raw_cleanup, "checks": checks},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
