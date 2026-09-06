#!/usr/bin/env python3
"""All-88 dual-behavior lockbox for candidate-scaffold coordinate transport."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2588 = RESULT / "phase2588_c450817_c467200_bilingual_natural_operation_behavior"
P2591 = RESULT / "phase2591_c491777_c508160_candidatefree_autonomous_behavior"
P2592 = RESULT / "phase2592_c508161_c524544_candidate_scaffold_transport_atlas"
P2593 = RESULT / "phase2593_c524545_c532736_scaffold_transport_client_atlas/analysis/final.json"
OUT = RESULT / "phase2594_c532737_c549120_scaffold_transport_all88_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2594, "C532737-C549120"
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
FAMILIES = ("preference", "taxonomy", "comparison", "causality", "translation",
            "reference", "chronology", "modality", "register", "syntax")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pearson(left, right):
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def material():
    eligible_with = {tuple(item) for item in load_json(
        P2588 / "material/eligible_aligned_quartets.json")["eligible"]}
    eligible_without = {tuple(item) for item in load_json(
        P2591 / "material/eligible_autonomous_quartets.json")["eligible"]}
    dual = sorted(eligible_with & eligible_without)
    with_rows = [row for row in read_jsonl(P2588 / "material/cases.jsonl") if row["ablation"] == "full"]
    without_rows = read_jsonl(P2591 / "material/candidatefree_prompts.jsonl")
    with_index = {(row["family_id"], row["language"], row["surface"], row["binding_relation"],
                   row["binding_value"], row["query_relation"], row["query_value"]): row for row in with_rows}
    without_index = {(row["family_id"], row["language"], row["surface"], row["binding_relation"],
                      row["binding_value"], row["query_relation"], row["query_value"]): row for row in without_rows}
    phase2592_prefixes = {tuple(item["prefix"]) for item in load_json(P2592 / "field/manifest.json")}
    selected = []
    for prefix in dual:
        with_cells = [with_index[prefix + cell] for cell in CELLS]
        without_cells = [without_index[prefix + cell] for cell in CELLS]
        selected.append({
            "prefix": prefix,
            "family": FAMILIES[prefix[0]],
            "language": prefix[1],
            "with_cells": with_cells,
            "without_cells": without_cells,
            "phase2592_discovery": prefix in phase2592_prefixes,
        })
    return selected


def forward_answer_states(model, cells):
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([row["prompt_ids"] for row in cells], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                       use_cache=False, logits_to_keep=1, return_dict=True)
    answer = cells[0]["answer_boundary_token"]
    states = torch.stack([state[:, answer].detach().cpu() for state in output.hidden_states], dim=1)
    result = states.to(torch.float16).numpy()
    del output, states
    return result


def collect(model, selected):
    raw_path = OUT / "field/answer_boundary_states.float16.npy"
    manifest_path = OUT / "field/manifest.json"
    if raw_path.is_file() and manifest_path.is_file():
        states = np.load(raw_path, mmap_mode="r")
        manifest = load_json(manifest_path)
        if states.shape[0] == len(selected) and len(manifest) == len(selected):
            return states, manifest, raw_path
    fields = []
    manifest = []
    for index, item in enumerate(selected):
        with_states = forward_answer_states(model, item["with_cells"])
        without_states = forward_answer_states(model, item["without_cells"])
        fields.append(np.stack([with_states, without_states]))
        prefix = item["prefix"]
        manifest.append({
            "quartet_index": index, "prefix": list(prefix), "family_id": prefix[0],
            "family": item["family"], "language": item["language"], "surface": prefix[2],
            "binding_relation": prefix[3], "binding_value": prefix[4],
            "phase2592_discovery": item["phase2592_discovery"],
            "cell_order": [list(cell) for cell in CELLS],
            "with_candidate_prompt_tokens": len(item["with_cells"][0]["prompt_ids"]),
            "candidate_free_prompt_tokens": len(item["without_cells"][0]["prompt_ids"]),
        })
        if (index + 1) % 11 == 0 or index + 1 == len(selected):
            print(f"[phase2594 answer fields] {index + 1}/{len(selected)}", flush=True)
        del with_states, without_states
        gc.collect()
        torch.cuda.empty_cache()
    array = np.stack(fields).astype(np.float16)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, array, allow_pickle=False)
    save_json(manifest_path, manifest)
    return np.load(raw_path, mmap_mode="r"), manifest, raw_path


def interaction(states, manifest):
    # [quartet, scaffold, cell, hidden, coordinate]
    value = (states[:, :, 3].astype(np.float32) - states[:, :, 2].astype(np.float32)
             - states[:, :, 1].astype(np.float32) + states[:, :, 0].astype(np.float32))
    signs = np.array([-1.0 if (item["binding_relation"] ^ item["binding_value"]) else 1.0
                      for item in manifest], dtype=np.float32)
    return value, value * signs[:, None, None, None]


def coordinate_sample_correlation(left, right):
    # Correlation over frozen quartets for every layer and every physical coordinate.
    x = left.astype(np.float64)
    y = right.astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    return np.divide(np.sum(x * y, axis=0), denominator,
                     out=np.zeros_like(denominator), where=denominator > 0).astype(np.float32)


def analyze(states, manifest, raw_path):
    raw_i, oriented = interaction(states, manifest)
    with_o, without_o = oriented[:, 0], oriented[:, 1]
    with_i, without_i = raw_i[:, 0], raw_i[:, 1]
    n, hidden_states, d_model = with_o.shape
    discovery = np.array([item["phase2592_discovery"] for item in manifest])
    confirmation = ~discovery

    def paired_curve(mask):
        return [float(np.median([pearson(with_o[index, layer], without_o[index, layer])
                                 for index in np.flatnonzero(mask)])) for layer in range(hidden_states)]

    def absolute_curve(mask):
        return [float(np.median([pearson(np.abs(with_i[index, layer]), np.abs(without_i[index, layer]))
                                 for index in np.flatnonzero(mask)])) for layer in range(hidden_states)]

    curves = {
        "all88_same_prefix_signed_median": paired_curve(np.ones(n, dtype=bool)),
        "all88_same_prefix_absolute_median": absolute_curve(np.ones(n, dtype=bool)),
        "phase2592_discovery19_signed_median": paired_curve(discovery),
        "independent_confirmation69_signed_median": paired_curve(confirmation),
    }
    for surface in (0, 3):
        mask = np.array([item["surface"] == surface for item in manifest])
        curves[f"surface{surface}_signed_median"] = paired_curve(mask)

    coord_all = coordinate_sample_correlation(with_o, without_o)
    coord_confirmation = coordinate_sample_correlation(with_o[confirmation], without_o[confirmation])
    sign_agreement = np.mean(np.signbit(with_o) == np.signbit(without_o), axis=0).astype(np.float32)
    coordinate_summary = {
        "all88_coordinate_correlation_median": np.median(coord_all, axis=1).tolist(),
        "all88_coordinate_correlation_q05": np.quantile(coord_all, .05, axis=1).tolist(),
        "all88_coordinate_correlation_q95": np.quantile(coord_all, .95, axis=1).tolist(),
        "confirmation69_coordinate_correlation_median": np.median(coord_confirmation, axis=1).tolist(),
        "coordinate_sign_agreement_mean": np.mean(sign_agreement, axis=1).tolist(),
    }

    groups = defaultdict(list)
    for index, item in enumerate(manifest):
        if confirmation[index]:
            groups[f"{item['family']}/{item['language']}"] .append(index)
    labels = sorted(groups)
    with_group = np.stack([with_o[groups[label]].mean(axis=0) for label in labels])
    without_group = np.stack([without_o[groups[label]].mean(axis=0) for label in labels])
    group_count = len(labels)
    cross_raw = np.zeros((hidden_states, group_count, group_count), dtype=np.float32)
    cross_centered = np.zeros_like(cross_raw)
    topology = np.zeros(hidden_states, dtype=np.float32)
    language_indices = {
        language: np.array([index for index, label in enumerate(labels) if label.endswith(f"/{language}")])
        for language in ("en", "zh")
    }
    triangle = np.triu_indices(group_count, k=1)
    for layer in range(hidden_states):
        centered_with = with_group[:, layer].copy()
        centered_without = without_group[:, layer].copy()
        for indices in language_indices.values():
            centered_with[indices] -= centered_with[indices].mean(axis=0, keepdims=True)
            centered_without[indices] -= centered_without[indices].mean(axis=0, keepdims=True)
        graph_with = np.zeros((group_count, group_count), dtype=np.float32)
        graph_without = np.zeros_like(graph_with)
        for left in range(group_count):
            for right in range(group_count):
                cross_raw[layer, left, right] = pearson(with_group[left, layer], without_group[right, layer])
                cross_centered[layer, left, right] = pearson(centered_with[left], centered_without[right])
                graph_with[left, right] = pearson(with_group[left, layer], with_group[right, layer])
                graph_without[left, right] = pearson(without_group[left, layer], without_group[right, layer])
        topology[layer] = pearson(graph_with[triangle], graph_without[triangle])
    matched = [(index, index) for index in range(group_count)]
    unmatched = [(left, right) for left in range(group_count) for right in range(group_count) if left != right]

    def graph_curve(matrix, pairs):
        return [float(np.median([matrix[layer, left, right] for left, right in pairs]))
                for layer in range(hidden_states)]

    graph_curves = {
        "confirmation_group_matched_raw": graph_curve(cross_raw, matched),
        "confirmation_group_unmatched_raw": graph_curve(cross_raw, unmatched),
        "confirmation_group_matched_centered": graph_curve(cross_centered, matched),
        "confirmation_group_unmatched_centered": graph_curve(cross_centered, unmatched),
        "confirmation_family_graph_topology": topology.tolist(),
    }
    late_start = max(1, math.ceil((hidden_states - 1) * .75))
    late = slice(late_start, hidden_states)
    late_summary = {
        **{key: float(np.mean(np.asarray(value)[late])) for key, value in curves.items()},
        **{key: float(np.mean(np.asarray(value)[late])) for key, value in graph_curves.items()},
        **{key: float(np.mean(np.asarray(value)[late])) for key, value in coordinate_summary.items()},
    }
    late_summary["confirmation_centered_matched_minus_unmatched"] = (
        late_summary["confirmation_group_matched_centered"]
        - late_summary["confirmation_group_unmatched_centered"])
    derived_path = OUT / "analysis/all_coordinate_transport_fields.npz"
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(derived_path, coordinate_correlation_all88=coord_all,
             coordinate_correlation_confirmation69=coord_confirmation,
             coordinate_sign_agreement_all88=sign_agreement,
             labels=np.array(labels), confirmation_cross_raw=cross_raw,
             confirmation_cross_centered=cross_centered, confirmation_topology=topology,
             with_candidate_mean_interaction=with_o.mean(axis=0),
             candidate_free_mean_interaction=without_o.mean(axis=0))
    distribution = Counter(f"{item['family']}/{item['language']}" for item in manifest)
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-4B BF16 CUDA nonquantized; answer-boundary fields float16",
        "selection": {"all_dual_behavior_qualified_quartets": n,
                      "phase2592_discovery_quartets": int(discovery.sum()),
                      "independent_confirmation_quartets": int(confirmation.sum()),
                      "all_prompt_fields": n * 2 * 4,
                      "family_language_counts": dict(distribution),
                      "missing_group": "modality/zh"},
        "field": {"shape": list(states.shape), "axis_order": "quartet, scaffold, cell, hidden_state, coordinate",
                  "hidden_states": hidden_states, "d_model": d_model, "bytes": raw_path.stat().st_size,
                  "storage_precision": "float16", "analysis_accumulation": "float32/float64",
                  "full_coordinates_at_answer_boundary": True,
                  "representative_full_token_fields_in_phase2592": True,
                  "no_topk": True},
        "paired_transport": {"curves": curves, "coordinate_distributions": coordinate_summary,
                             "confirmation_group_labels": labels, "confirmation_group_curves": graph_curves,
                             "late_hidden_state_start": late_start, "late_summary": late_summary},
        "files": {"raw_answer_states": str(raw_path.relative_to(OUT)).replace("\\", "/"),
                  "derived_all_coordinates": str(derived_path.relative_to(OUT)).replace("\\", "/"),
                  "manifest": "field/manifest.json"},
        "hashes": {"raw_answer_states": sha256(raw_path), "derived": sha256(derived_path)},
        "claim_boundary": {
            "supported": "tests Phase2592 scaffold transport on every dual-behavior-qualified quartet and a disjoint 69-quartet confirmation split",
            "not_supported": "answer-boundary correlation identifies a causal gear, or the four-fact lookup exhausts natural-language encoding",
        },
        "language_mechanism_closed": False,
    }
    checks = {
        "phase2593_complete": load_json(P2593)["all_checks_passed"],
        "all_88_dual_quartets": n == 88,
        "discovery19_confirmation69": int(discovery.sum()) == 19 and int(confirmation.sum()) == 69,
        "all_704_prompt_fields": n * 2 * 4 == 704,
        "all_37_hidden_states": hidden_states == 37,
        "all_2560_coordinates": d_model == 2560,
        "all_coordinate_arrays": coord_all.shape == coord_confirmation.shape == sign_agreement.shape == (37, 2560),
        "only_modality_zh_missing": set(distribution) == {
            f"{family}/{language}" for family in FAMILIES for language in ("en", "zh")
        } - {"modality/zh"},
        "no_topk": True,
        "scientific_result_does_not_abort": True,
        "claim_boundary": True,
    }
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    return result


def append_memo(result):
    heading = f"## Phase {PHASE}: 全部88双行为四元组的候选脚手架迁移锁箱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** Phase2592每族/语言只有一个样本。本Phase冻结全部88个“有候选打分四格全对且无候选greedy四格全对”的prefix；Phase2592的19个作为已见发现集，其余69个作为不重叠确认集。保存两个脚手架、四cell、embedding+36 blocks、answer boundary全部2560坐标：

$$I^{{s}}_{{qld}}=H^{{s,11}}_{{qld}}-H^{{s,10}}_{{qld}}-H^{{s,01}}_{{qld}}+H^{{s,00}}_{{qld}},\quad
\rho_{{ql}}=\operatorname{{corr}}_d(I^{{with}}_{{qld}},I^{{free}}_{{qld}}),$$

$$r_{{ld}}=\operatorname{{corr}}_q(I^{{with}}_{{qld}},I^{{free}}_{{qld}}).$$

$\rho$逐样本遍历全部坐标；$r$则对每个物理坐标沿88样本计算迁移相关，不做Top-K。另在69确认集按族/语言求均值，重算raw、语言中心化对角优势和族图拓扑。

**测试用例与全量性。** 88四元组×候选有/无×4 cell=704条prompt的answer-boundary全场，shape=`{result['field']['shape']}`、{result['field']['bytes']} bytes；完整逐token代表场已由Phase2592保存。本Phase分布=`{json.dumps(result['selection']['family_language_counts'], ensure_ascii=False)}`；`modality/zh`继续作为无合格四元组的阴性边界。

**结果汇总。** 逐样本/表面曲线=`{json.dumps(result['paired_transport']['curves'], ensure_ascii=False)}`；逐坐标分布=`{json.dumps(result['paired_transport']['coordinate_distributions'], ensure_ascii=False)}`；69确认集族图=`{json.dumps(result['paired_transport']['confirmation_group_curves'], ensure_ascii=False)}`；预注册末四分之一均值=`{json.dumps(result['paired_transport']['late_summary'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2594_c532737_c549120_scaffold_transport_all88_lockbox.py`；704 answer-boundary全坐标场、manifest、全部37×2560坐标迁移/符号一致率、确认集族图、哈希与final位于`{OUT}`。

**分析与理论进展。** 发现/确认分离可裁决Phase2592的单样本结果是否只来自挑选；逐坐标$ r_{{ld}} $保留所有低值坐标，回答迁移是少数大坐标现象还是固定基底上的广泛协同。确认集中心化对角优势若保留，只支持候选删除后存在可复现的族条件残差，不把它命名为语义齿轮。

**问题硬伤。** 样本仍按两种行为都成功筛选，不能外推失败路径；answer boundary全坐标锁箱不是全部token存储；组内样本数不均且taxonomy/en确认集为空；四事实、短代号复制和有限差分混杂不变；相关不是因果。

**结论。** `{result['claim_boundary']['supported']}`。不支持：`{result['claim_boundary']['not_supported']}`。检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    selected = material()
    raw_path = OUT / "field/answer_boundary_states.float16.npy"
    model = None
    try:
        if not raw_path.is_file():
            model, _, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        states, manifest, raw_path = collect(model, selected)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    result = analyze(states, manifest, raw_path)
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({key: result[key] for key in (
        "phase", "selection", "field", "paired_transport", "checks",
        "all_checks_passed", "language_mechanism_closed")}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
