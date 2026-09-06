#!/usr/bin/env python3
"""Paired full-coordinate atlas with versus without a candidate list."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2588 = RESULT / "phase2588_c450817_c467200_bilingual_natural_operation_behavior"
P2589 = RESULT / "phase2589_c467201_c483584_bilingual_operation_fullcoordinate_atlas"
P2591 = RESULT / "phase2591_c491777_c508160_candidatefree_autonomous_behavior"
OUT = RESULT / "phase2592_c508161_c524544_candidate_scaffold_transport_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2592, "C508161-C524544"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64, copy=False).ravel()
    y = right.astype(np.float64, copy=False).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_pairs():
    eligible_with = {tuple(item) for item in load_json(
        P2588 / "material/eligible_aligned_quartets.json")["eligible"]}
    eligible_without = {tuple(item) for item in load_json(
        P2591 / "material/eligible_autonomous_quartets.json")["eligible"]}
    dual = eligible_with & eligible_without
    with_rows = [row for row in read_jsonl(P2588 / "material/cases.jsonl") if row["ablation"] == "full"]
    without_rows = read_jsonl(P2591 / "material/candidatefree_prompts.jsonl")
    with_index = {(row["family_id"], row["language"], row["surface"], row["binding_relation"],
                   row["binding_value"], row["query_relation"], row["query_value"]): row for row in with_rows}
    without_index = {(row["family_id"], row["language"], row["surface"], row["binding_relation"],
                      row["binding_value"], row["query_relation"], row["query_value"]): row for row in without_rows}
    prior_manifest = load_json(P2589 / "field/manifest.json")
    prior = {tuple(item["prefix"]): item for item in prior_manifest}
    selected = []
    missing_groups = []
    for family_id, family in enumerate(FAMILIES):
        for language in ("en", "zh"):
            candidates = sorted(prefix for prefix in dual if prefix[:2] == (family_id, language))
            if not candidates:
                missing_groups.append(f"{family}/{language}")
                continue
            prefix = min(candidates, key=lambda item: (item not in prior, item[2], item[3], item[4]))
            with_cells = [with_index[prefix + cell] for cell in CELLS]
            without_cells = [without_index[prefix + cell] for cell in CELLS]
            if len({len(row["prompt_ids"]) for row in with_cells}) != 1 or len(
                    {len(row["prompt_ids"]) for row in without_cells}) != 1:
                raise RuntimeError(f"unaligned paired quartet: {prefix}")
            selected.append((prefix, with_cells, without_cells, prior.get(prefix)))
    return selected, missing_groups, len(dual)


def forward_field(model, cells):
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([row["prompt_ids"] for row in cells], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                       use_cache=False, logits_to_keep=1, return_dict=True)
    field = torch.stack([state.detach().cpu() for state in output.hidden_states], dim=1)
    result = field.to(torch.float16).numpy()
    del output, field
    return result


def collect(model, tokenizer, selected):
    field_dir = OUT / "field"
    field_dir.mkdir(parents=True, exist_ok=True)
    old_path = field_dir / "manifest.json"
    old = load_json(old_path) if old_path.is_file() else []
    manifests = []
    with_interactions = []
    without_interactions = []
    with_oriented = []
    without_oriented = []
    for group_index, (prefix, with_cells, without_cells, prior) in enumerate(selected):
        without_path = field_dir / f"group_{group_index:02d}_candidatefree.npy"
        extra_with_path = field_dir / f"group_{group_index:02d}_withcandidate.npy"
        if without_path.is_file() and group_index < len(old):
            without_field = np.load(without_path, mmap_mode="r")
        else:
            if model is None:
                raise RuntimeError(f"missing candidate-free field {without_path}")
            without_field = forward_field(model, without_cells)
            np.save(without_path, without_field, allow_pickle=False)
        if prior is not None:
            with_path = ROOT / prior["path"]
            with_field = np.load(with_path, mmap_mode="r")
            with_reused = True
        elif extra_with_path.is_file() and group_index < len(old):
            with_path = extra_with_path
            with_field = np.load(with_path, mmap_mode="r")
            with_reused = False
        else:
            if model is None:
                raise RuntimeError(f"missing with-candidate field {extra_with_path}")
            with_field = forward_field(model, with_cells)
            np.save(extra_with_path, with_field, allow_pickle=False)
            with_path = extra_with_path
            with_reused = False
        wi = (with_field[3].astype(np.float32) - with_field[2].astype(np.float32)
              - with_field[1].astype(np.float32) + with_field[0].astype(np.float32))
        wo = (without_field[3].astype(np.float32) - without_field[2].astype(np.float32)
              - without_field[1].astype(np.float32) + without_field[0].astype(np.float32))
        with_answer = with_cells[0]["answer_boundary_token"]
        without_answer = without_cells[0]["answer_boundary_token"]
        sign = -1.0 if (prefix[3] ^ prefix[4]) else 1.0
        with_interactions.append(wi[:, with_answer])
        without_interactions.append(wo[:, without_answer])
        with_oriented.append(sign * wi[:, with_answer])
        without_oriented.append(sign * wo[:, without_answer])
        with_query_start = min(with_cells[0]["regions"]["query_relation"] + with_cells[0]["regions"]["query_value"])
        # Candidate-free prompts share an exact causal prefix; locate its first divergence directly.
        columns = list(zip(*(row["prompt_ids"] for row in without_cells)))
        without_query_start = next((index for index, values in enumerate(columns) if len(set(values)) > 1), without_answer)
        record = {
            "group_index": group_index, "prefix": list(prefix), "family_id": prefix[0],
            "family": FAMILIES[prefix[0]], "language": prefix[1], "surface": prefix[2],
            "binding_relation": prefix[3], "binding_value": prefix[4],
            "with_candidate": {
                "path": str(with_path.relative_to(ROOT)).replace("\\", "/"),
                "reused_phase2589": with_reused, "bytes": with_path.stat().st_size,
                "sha256": sha256(with_path), "prompt_tokens": int(with_field.shape[2]),
                "answer_boundary_token": with_answer, "query_start_token": with_query_start,
                "embedding_interaction_maxabs": float(np.max(np.abs(wi[0]))),
                "causal_prefix_interaction_maxabs": float(np.max(np.abs(wi[:, :with_query_start]))),
            },
            "candidate_free": {
                "path": str(without_path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": without_path.stat().st_size, "sha256": sha256(without_path),
                "prompt_tokens": int(without_field.shape[2]), "answer_boundary_token": without_answer,
                "query_start_token": without_query_start,
                "embedding_interaction_maxabs": float(np.max(np.abs(wo[0]))),
                "causal_prefix_interaction_maxabs": float(np.max(np.abs(wo[:, :without_query_start]))),
                "tokens": [tokenizer.decode([int(token)]) for token in without_cells[0]["prompt_ids"]]
                          if tokenizer is not None else old[group_index]["candidate_free"]["tokens"],
            },
            "hidden_states": int(with_field.shape[1]), "d_model": int(with_field.shape[3]),
            "cell_order": [list(cell) for cell in CELLS],
            "dual_behavior_qualified": True,
        }
        manifests.append(record)
        del with_field, without_field, wi, wo
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[phase2592 field] {group_index + 1}/{len(selected)}", flush=True)
    arrays = {
        "with_candidate_interaction": np.stack(with_interactions),
        "candidate_free_interaction": np.stack(without_interactions),
        "with_candidate_oriented": np.stack(with_oriented),
        "candidate_free_oriented": np.stack(without_oriented),
    }
    answer_path = OUT / "analysis/paired_answer_coordinate_fields.npz"
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(answer_path, **arrays)
    save_json(field_dir / "manifest.json", manifests)
    return manifests, arrays, answer_path


def analyze(manifests, arrays, answer_path, dual_count, missing_groups):
    with_i = arrays["with_candidate_interaction"]
    without_i = arrays["candidate_free_interaction"]
    with_o = arrays["with_candidate_oriented"]
    without_o = arrays["candidate_free_oriented"]
    groups, hidden_states, d_model = with_o.shape
    labels = [f"{item['family']}/{item['language']}" for item in manifests]
    scaffold_signed = np.zeros((hidden_states, groups), dtype=np.float32)
    scaffold_absolute = np.zeros_like(scaffold_signed)
    cross_raw = np.zeros((hidden_states, groups, groups), dtype=np.float32)
    cross_centered = np.zeros_like(cross_raw)
    topology = np.zeros(hidden_states, dtype=np.float32)
    lang_indices = {
        language: np.array([index for index, item in enumerate(manifests) if item["language"] == language])
        for language in ("en", "zh")
    }
    triangle = np.triu_indices(groups, k=1)
    for layer in range(hidden_states):
        centered_with = with_o[:, layer].copy()
        centered_without = without_o[:, layer].copy()
        for indices in lang_indices.values():
            centered_with[indices] -= centered_with[indices].mean(axis=0, keepdims=True)
            centered_without[indices] -= centered_without[indices].mean(axis=0, keepdims=True)
        graph_with = np.zeros((groups, groups), dtype=np.float32)
        graph_without = np.zeros_like(graph_with)
        for left in range(groups):
            scaffold_signed[layer, left] = pearson(with_o[left, layer], without_o[left, layer])
            scaffold_absolute[layer, left] = pearson(np.abs(with_i[left, layer]), np.abs(without_i[left, layer]))
            for right in range(groups):
                cross_raw[layer, left, right] = pearson(with_o[left, layer], without_o[right, layer])
                cross_centered[layer, left, right] = pearson(centered_with[left], centered_without[right])
                graph_with[left, right] = pearson(with_o[left, layer], with_o[right, layer])
                graph_without[left, right] = pearson(without_o[left, layer], without_o[right, layer])
        topology[layer] = pearson(graph_with[triangle], graph_without[triangle])
    matched = [(index, index) for index in range(groups)]
    unmatched = [(left, right) for left in range(groups) for right in range(groups) if left != right]

    def curve(matrix, pairs):
        return [float(np.median([matrix[layer, left, right] for left, right in pairs]))
                for layer in range(hidden_states)]

    curves = {
        "same_group_scaffold_signed_median": np.median(scaffold_signed, axis=1).tolist(),
        "same_group_scaffold_absolute_median": np.median(scaffold_absolute, axis=1).tolist(),
        "matched_cross_scaffold_raw_median": curve(cross_raw, matched),
        "unmatched_cross_scaffold_raw_median": curve(cross_raw, unmatched),
        "matched_cross_scaffold_language_centered_median": curve(cross_centered, matched),
        "unmatched_cross_scaffold_language_centered_median": curve(cross_centered, unmatched),
        "within_scaffold_family_graph_topology_correlation": topology.tolist(),
    }
    late_start = max(1, math.ceil((hidden_states - 1) * .75))
    late = slice(late_start, hidden_states)
    late_summary = {key: float(np.mean(np.asarray(value)[late])) for key, value in curves.items()}
    late_summary["raw_matched_minus_unmatched"] = (
        late_summary["matched_cross_scaffold_raw_median"] - late_summary["unmatched_cross_scaffold_raw_median"])
    late_summary["centered_matched_minus_unmatched"] = (
        late_summary["matched_cross_scaffold_language_centered_median"]
        - late_summary["unmatched_cross_scaffold_language_centered_median"])
    matrices_path = OUT / "analysis/scaffold_transport_coordinate_graphs.npz"
    np.savez(matrices_path, labels=np.array(labels), scaffold_signed=scaffold_signed,
             scaffold_absolute=scaffold_absolute, cross_raw=cross_raw,
             cross_language_centered=cross_centered, topology=topology)
    with_rms = np.sqrt(np.mean(with_i.astype(np.float64) ** 2, axis=2))
    without_rms = np.sqrt(np.mean(without_i.astype(np.float64) ** 2, axis=2))
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-4B BF16 CUDA nonquantized; full fields stored float16",
        "selection": {"dual_qualified_quartets_available": dual_count, "paired_groups": groups,
                      "one_quartet_per_group": True, "cell_prompts_per_scaffold": groups * 4,
                      "missing_groups": missing_groups,
                      "reused_phase2589_with_candidate_fields": sum(
                          item["with_candidate"]["reused_phase2589"] for item in manifests)},
        "field": {"hidden_states": hidden_states, "d_model": d_model,
                  "new_candidate_free_raw_bytes": sum(item["candidate_free"]["bytes"] for item in manifests),
                  "new_extra_with_candidate_raw_bytes": sum(item["with_candidate"]["bytes"] for item in manifests
                      if not item["with_candidate"]["reused_phase2589"]),
                  "candidate_prompt_token_range": [min(item["with_candidate"]["prompt_tokens"] for item in manifests),
                                                     max(item["with_candidate"]["prompt_tokens"] for item in manifests)],
                  "candidate_free_prompt_token_range": [min(item["candidate_free"]["prompt_tokens"] for item in manifests),
                                                          max(item["candidate_free"]["prompt_tokens"] for item in manifests)],
                  "full_tokens_no_padding": True, "full_coordinates_no_topk": True,
                  "storage_precision": "float16", "analysis_accumulation": "float32/float64"},
        "interaction": {"formula": "I=H11-H10-H01+H00",
                        "with_candidate_median_rms": np.median(with_rms, axis=0).tolist(),
                        "candidate_free_median_rms": np.median(without_rms, axis=0).tolist(),
                        "embedding_maxabs": max(max(item[variant]["embedding_interaction_maxabs"]
                                                    for variant in ("with_candidate", "candidate_free"))
                                                for item in manifests),
                        "causal_prefix_maxabs": max(max(item[variant]["causal_prefix_interaction_maxabs"]
                                                        for variant in ("with_candidate", "candidate_free"))
                                                    for item in manifests)},
        "scaffold_transport": {"labels": labels, "curves": curves,
                               "late_hidden_state_start": late_start, "late_summary": late_summary,
                               "warning": "paired coordinate correlation is descriptive and one quartet per group is not a stability estimate"},
        "files": {"manifest": "field/manifest.json",
                  "answer_fields": str(answer_path.relative_to(OUT)).replace("\\", "/"),
                  "coordinate_graphs": str(matrices_path.relative_to(OUT)).replace("\\", "/")},
        "claim_boundary": {
            "supported": "measures how the same behavior-qualified natural-operation interaction texture changes when the candidate list is removed",
            "not_supported": "a high paired correlation would prove scaffold-independent semantic gears or autonomous-generation causality",
        },
        "language_mechanism_closed": False,
    }
    checks = {
        "phase2591_complete": load_json(P2591 / "analysis/final.json")["all_checks_passed"],
        "nineteen_paired_groups": groups == 19,
        "only_modality_zh_missing": missing_groups == ["modality/zh"],
        "all_152_prompt_fields": groups * 2 * 4 == 152,
        "all_raw_fields_exist": all((ROOT / item[variant]["path"]).is_file()
                                    for item in manifests for variant in ("with_candidate", "candidate_free")),
        "all_2560_coordinates": d_model == 2560,
        "embedding_interaction_zero": result["interaction"]["embedding_maxabs"] <= 1e-7,
        "causal_prefix_interaction_zero": result["interaction"]["causal_prefix_maxabs"] <= 1e-7,
        "no_topk_primary_analysis": True,
        "scientific_result_does_not_abort": True,
        "claim_boundary": True,
    }
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    return result


def append_memo(result):
    heading = f"## Phase {PHASE}: 候选有/无的双语族全坐标脚手架迁移图（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** 对Phase2588候选行为与Phase2591无候选greedy四格同时全对的同一prefix，成对采集候选有/无时的embedding、全部block、全部token、全部2560物理坐标：

$$I^{{(s)}}_{{ltd}}=H^{{(s),11}}_{{ltd}}-H^{{(s),10}}_{{ltd}}-H^{{(s),01}}_{{ltd}}+H^{{(s),00}}_{{ltd}},\quad
T_l(g,h)=\operatorname{{corr}}_d(I^{{with}}_{{gld}},I^{{free}}_{{hld}}).$$

同组对角与异组非对角先在raw场比较，再分别减去英文/中文的公共均值。另比较两个脚手架内部19×19族图上所有非对角边的相关，检查“族关系拓扑”是否随候选删除保留。

**测试用例与全量性。** 双行为合格四元组共{result['selection']['dual_qualified_quartets_available']}个；19个可用族/语言单元各冻结1个，共候选有/无×19×4=152条prompt全场。`modality/zh`没有自主四格全对样本，作为阴性边界保留而不关闭路线。复用Phase2589候选原场{result['selection']['reused_phase2589_with_candidate_fields']}组；其余与全部无候选场新采集。每场含{result['field']['hidden_states']}检查点×全部token×{result['field']['d_model']}坐标，不作PCA、压缩或Top-K。

**结果汇总。** 逐层交互RMS与迁移曲线=`{json.dumps({'interaction': result['interaction'], 'curves': result['scaffold_transport']['curves']}, ensure_ascii=False)}`。预注册末四分之一从q{result['scaffold_transport']['late_hidden_state_start']}开始，均值=`{json.dumps(result['scaffold_transport']['late_summary'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2592_c508161_c524544_candidate_scaffold_transport_atlas.py`；逐token原场/跨Phase引用、哈希manifest、answer全坐标数组、逐层迁移矩阵、族图与final位于`{OUT}`。

**分析与理论进展。** 同组迁移回答相同自然词与事实在候选删除后是否复用坐标纹理；对角减非对角回答复用是否超过公共任务场；语言中心化版本进一步扣除语言模板共同量。族拓扑相关只说明19节点相对关系保留程度，不证明某节点对应一个语义功能。

**问题硬伤。** 每组只有一个冻结四元组，不能估计组内方差；`modality/zh`缺失使图不完整；无候选输出仍复制事实中的短代号，不是开放世界生成；answer boundary位置和前缀长度随脚手架变化；有限差分仍混合一般非线性与条件计算。

**结论。** `{result['claim_boundary']['supported']}`。不支持：`{result['claim_boundary']['not_supported']}`。检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    selected, missing_groups, dual_count = select_pairs()
    old_path = OUT / "field/manifest.json"
    old = load_json(old_path) if old_path.is_file() else []
    reusable = len(old) == len(selected) and all((ROOT / item["candidate_free"]["path"]).is_file() for item in old)
    model = tokenizer = None
    try:
        if not reusable:
            model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        manifests, arrays, answer_path = collect(model, tokenizer, selected)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    result = analyze(manifests, arrays, answer_path, dual_count, missing_groups)
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({key: result[key] for key in (
        "phase", "selection", "field", "interaction", "scaffold_transport", "checks",
        "all_checks_passed", "language_mechanism_closed")}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
