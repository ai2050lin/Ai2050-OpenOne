#!/usr/bin/env python3
"""Bilingual natural-operation family atlas over every token and physical coordinate."""
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
OUT = RESULT / "phase2589_c467201_c483584_bilingual_operation_fullcoordinate_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2589, "C467201-C483584"
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
FAMILY_NAMES = (
    "preference", "taxonomy", "comparison", "causality", "translation",
    "reference", "chronology", "modality", "register", "syntax",
)

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


def load_and_select():
    final = load_json(P2588 / "analysis/final.json")
    if not final["all_checks_passed"] or not final["adjudication"]["behavior_bridge_qualified"]:
        raise RuntimeError("Phase2588 bridge is not qualified")
    rows = [json.loads(line) for line in (P2588 / "material/cases.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    full = [row for row in rows if row["ablation"] == "full"]
    index = {(row["family_id"], row["language"], row["surface"], row["binding_relation"],
              row["binding_value"], row["query_relation"], row["query_value"]): row for row in full}
    payload = load_json(P2588 / "material/eligible_aligned_quartets.json")
    eligible = [tuple(item) for item in payload["eligible"]]
    buckets = defaultdict(list)
    for prefix in eligible:
        buckets[(prefix[0], prefix[1], prefix[2])].append(prefix)
    selected = []
    for family_id in range(10):
        for language in ("en", "zh"):
            # Two predeclared, maximally separated surface frames per group.
            for surface, preferred_binding in ((0, (0, 0)), (3, (1, 1))):
                candidates = sorted(buckets[(family_id, language, surface)])
                if not candidates:
                    raise RuntimeError(f"missing eligible surface: {(family_id, language, surface)}")
                prefix = min(candidates, key=lambda item: (item[3:5] != preferred_binding, item[3], item[4]))
                cells = [index[prefix + cell] for cell in CELLS]
                if len({len(row["prompt_ids"]) for row in cells}) != 1:
                    raise RuntimeError(f"unaligned selected quartet: {prefix}")
                selected.append((prefix, cells))
    return final, selected


def region_lookup(row: dict) -> list[tuple[str, int]]:
    lookup = [None] * len(row["prompt_ids"])
    for region, positions in row["regions"].items():
        for ordinal, position in enumerate(positions):
            if lookup[position] is not None:
                raise RuntimeError(f"overlapping token region at {position}")
            lookup[position] = (region, ordinal)
    if any(item is None for item in lookup):
        raise RuntimeError("semantic regions are not a token partition")
    return lookup


def collect(model, tokenizer, selected):
    raw_dir = OUT / "field"
    raw_dir.mkdir(parents=True, exist_ok=True)
    old_manifest_path = raw_dir / "manifest.json"
    old_manifest = load_json(old_manifest_path) if old_manifest_path.is_file() else []
    device = model.get_input_embeddings().weight.device if model is not None else None
    manifest = []
    interactions = []
    oriented = []
    region_rms = defaultdict(list)
    exemplar = None
    for quartet_index, (prefix, cells) in enumerate(selected):
        raw_path = raw_dir / f"quartet_{quartet_index:03d}.npy"
        if raw_path.is_file() and quartet_index < len(old_manifest):
            field = np.load(raw_path, mmap_mode="r")
        else:
            if model is None:
                raise RuntimeError(f"raw field missing: {raw_path}")
            ids = torch.tensor([row["prompt_ids"] for row in cells], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                               use_cache=False, logits_to_keep=1, return_dict=True)
            field = torch.stack([state.detach().cpu() for state in output.hidden_states], dim=1)
            field = field.to(torch.float16).numpy()
            del output
            np.save(raw_path, field, allow_pickle=False)
        h00, h01, h10, h11 = (field[index].astype(np.float32) for index in range(4))
        interaction = h11 - h10 - h01 + h00
        answer = cells[0]["answer_boundary_token"]
        answer_interaction = interaction[:, answer, :]
        orientation = -1.0 if (prefix[3] ^ prefix[4]) else 1.0
        interactions.append(answer_interaction)
        oriented.append(orientation * answer_interaction)
        lookup = region_lookup(cells[0])
        token_rms = np.sqrt(np.mean(interaction.astype(np.float64) ** 2, axis=-1))
        for region in cells[0]["regions"]:
            positions = [position for position, (name, _) in enumerate(lookup) if name == region]
            region_rms[region].append(token_rms[:, positions].mean(axis=1))
        query_start = min(cells[0]["regions"]["query_relation"] + cells[0]["regions"]["query_value"])
        tokens = ([tokenizer.decode([int(token)]) for token in cells[0]["prompt_ids"]]
                  if tokenizer is not None else old_manifest[quartet_index]["tokens"])
        record = {
            "quartet_index": quartet_index,
            "prefix": list(prefix),
            "family_id": prefix[0],
            "family": FAMILY_NAMES[prefix[0]],
            "language": prefix[1],
            "surface": prefix[2],
            "binding_relation": prefix[3],
            "binding_value": prefix[4],
            "cell_order": [list(cell) for cell in CELLS],
            "case_ids": [row["case_id"] for row in cells],
            "target_indices": [row["target_index"] for row in cells],
            "prompt_tokens": int(field.shape[2]),
            "hidden_states": int(field.shape[1]),
            "d_model": int(field.shape[3]),
            "dtype": "float16",
            "path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
            "answer_boundary_token": answer,
            "query_start_token": query_start,
            "regions": cells[0]["regions"],
            "tokens": tokens,
            "embedding_interaction_maxabs": float(np.max(np.abs(interaction[0]))),
            "causal_prefix_interaction_maxabs": float(np.max(np.abs(interaction[:, :query_start]))),
        }
        manifest.append(record)
        if exemplar is None:
            exemplar = {
                "field": np.asarray(field).copy(),
                "interaction": interaction.copy(),
                "metadata": record,
            }
        del field, h00, h01, h10, h11, interaction, answer_interaction, token_rms
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[phase2589 field] {quartet_index + 1}/{len(selected)}", flush=True)
    arrays = {
        "answer_interaction": np.stack(interactions),
        "answer_oriented_interaction": np.stack(oriented),
    }
    answer_path = OUT / "analysis/full_coordinate_answer_fields.npz"
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(answer_path, **arrays)
    exemplar_path = OUT / "analysis/exemplar_parameter_fields.npz"
    np.savez(
        exemplar_path,
        embedding_parameter=exemplar["field"][:, 0],
        answer_hidden_parameter=exemplar["field"][:, :, exemplar["metadata"]["answer_boundary_token"], :],
        answer_interaction_parameter=exemplar["interaction"][:, exemplar["metadata"]["answer_boundary_token"], :],
    )
    save_json(raw_dir / "manifest.json", manifest)
    return manifest, arrays, {key: np.stack(value) for key, value in region_rms.items()}, answer_path, exemplar_path


def off_diagonal(values: np.ndarray) -> list[float]:
    return [float(values[i, j]) for i in range(values.shape[0]) for j in range(values.shape[1]) if i != j]


def analyze(manifest, arrays, region_rms, answer_path, exemplar_path):
    interaction = arrays["answer_interaction"]
    oriented = arrays["answer_oriented_interaction"]
    hidden_states, d_model = interaction.shape[1:]
    labels = [f"{family}/{language}" for family in FAMILY_NAMES for language in ("en", "zh")]
    grouped = defaultdict(list)
    for index, item in enumerate(manifest):
        grouped[f"{item['family']}/{item['language']}"] .append(index)
    if list(grouped) != labels or any(len(grouped[label]) != 2 for label in labels):
        raise RuntimeError("expected exactly two surface replicates in every family/language group")
    group_profiles = np.stack([oriented[grouped[label]].mean(axis=0) for label in labels])
    group_abs_profiles = np.stack([np.abs(interaction[grouped[label]]).mean(axis=0) for label in labels])
    replicate_signed = np.zeros((hidden_states, len(labels)), dtype=np.float32)
    replicate_absolute = np.zeros_like(replicate_signed)
    raw_signed = np.zeros((hidden_states, len(labels), len(labels)), dtype=np.float32)
    raw_absolute = np.zeros_like(raw_signed)
    centered_signed = np.zeros_like(raw_signed)
    centered_absolute = np.zeros_like(raw_signed)
    language_index = {
        "en": np.array([index for index, label in enumerate(labels) if label.endswith("/en")]),
        "zh": np.array([index for index, label in enumerate(labels) if label.endswith("/zh")]),
    }
    for layer in range(hidden_states):
        for group_index, label in enumerate(labels):
            left, right = grouped[label]
            replicate_signed[layer, group_index] = pearson(oriented[left, layer], oriented[right, layer])
            replicate_absolute[layer, group_index] = pearson(
                np.abs(interaction[left, layer]), np.abs(interaction[right, layer]))
        centered = group_profiles[:, layer].copy()
        centered_abs = group_abs_profiles[:, layer].copy()
        for indices in language_index.values():
            centered[indices] -= centered[indices].mean(axis=0, keepdims=True)
            centered_abs[indices] -= centered_abs[indices].mean(axis=0, keepdims=True)
        for left in range(len(labels)):
            for right in range(len(labels)):
                raw_signed[layer, left, right] = pearson(group_profiles[left, layer], group_profiles[right, layer])
                raw_absolute[layer, left, right] = pearson(
                    group_abs_profiles[left, layer], group_abs_profiles[right, layer])
                centered_signed[layer, left, right] = pearson(centered[left], centered[right])
                centered_absolute[layer, left, right] = pearson(centered_abs[left], centered_abs[right])
    matrices_path = OUT / "analysis/family_coordinate_graphs.npz"
    np.savez(
        matrices_path,
        labels=np.array(labels),
        group_oriented_profiles=group_profiles,
        group_absolute_profiles=group_abs_profiles,
        replicate_signed=replicate_signed,
        replicate_absolute=replicate_absolute,
        raw_signed=raw_signed,
        raw_absolute=raw_absolute,
        language_centered_signed=centered_signed,
        language_centered_absolute=centered_absolute,
    )
    en_indices = language_index["en"]
    zh_indices = language_index["zh"]
    matched = [(2 * family, 2 * family + 1) for family in range(10)]
    unmatched = [(left, right) for left in en_indices for right in zh_indices if left // 2 != right // 2]
    within_en = [(left, right) for left in en_indices for right in en_indices if left < right]
    within_zh = [(left, right) for left in zh_indices for right in zh_indices if left < right]

    def curve(matrix, pairs):
        return [float(np.median([matrix[layer, left, right] for left, right in pairs]))
                for layer in range(hidden_states)]

    curves = {
        "surface_replicate_signed_median": np.median(replicate_signed, axis=1).tolist(),
        "surface_replicate_absolute_median": np.median(replicate_absolute, axis=1).tolist(),
        "matched_bilingual_raw_signed_median": curve(raw_signed, matched),
        "unmatched_bilingual_raw_signed_median": curve(raw_signed, unmatched),
        "matched_bilingual_centered_signed_median": curve(centered_signed, matched),
        "unmatched_bilingual_centered_signed_median": curve(centered_signed, unmatched),
        "matched_bilingual_raw_absolute_median": curve(raw_absolute, matched),
        "unmatched_bilingual_raw_absolute_median": curve(raw_absolute, unmatched),
        "within_english_raw_signed_median": curve(raw_signed, within_en),
        "within_chinese_raw_signed_median": curve(raw_signed, within_zh),
    }
    late_start = max(1, math.ceil((hidden_states - 1) * 0.75))
    late = slice(late_start, hidden_states)
    late_summary = {key: float(np.mean(np.asarray(value)[late])) for key, value in curves.items()}
    late_summary["matched_minus_unmatched_raw_signed"] = (
        late_summary["matched_bilingual_raw_signed_median"]
        - late_summary["unmatched_bilingual_raw_signed_median"])
    late_summary["matched_minus_unmatched_centered_signed"] = (
        late_summary["matched_bilingual_centered_signed_median"]
        - late_summary["unmatched_bilingual_centered_signed_median"])
    answer_rms = np.sqrt(np.mean(interaction.astype(np.float64) ** 2, axis=2))
    region_summary = {
        region: np.median(values, axis=0).tolist() for region, values in region_rms.items()
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-4B BF16 CUDA nonquantized; stored full fields float16",
        "selection": {
            "families": 10,
            "languages": 2,
            "groups": len(labels),
            "surface_replicates_per_group": 2,
            "selected_quartets": len(manifest),
            "cell_prompts": len(manifest) * 4,
            "selected_surfaces": [0, 3],
            "all_behavior_correct_and_token_aligned": True,
        },
        "field": {
            "cell_order": [list(cell) for cell in CELLS],
            "hidden_state_shape_per_quartet": "[4, embedding+blocks, token, d_model]",
            "hidden_states": hidden_states,
            "d_model": d_model,
            "raw_bytes": sum(item["bytes"] for item in manifest),
            "prompt_token_range": [min(item["prompt_tokens"] for item in manifest),
                                   max(item["prompt_tokens"] for item in manifest)],
            "full_tokens_no_padding": True,
            "full_coordinates_no_topk": True,
            "storage_precision": "float16",
            "analysis_accumulation": "float32/float64",
        },
        "interaction": {
            "formula": "I=H11-H10-H01+H00",
            "embedding_maxabs": max(item["embedding_interaction_maxabs"] for item in manifest),
            "causal_prefix_maxabs": max(item["causal_prefix_interaction_maxabs"] for item in manifest),
            "median_answer_rms_by_hidden_state": np.median(answer_rms, axis=0).tolist(),
            "median_region_rms_by_hidden_state": region_summary,
        },
        "coordinate_graph": {
            "labels": labels,
            "curves": curves,
            "late_hidden_state_start": late_start,
            "late_summary": late_summary,
            "interpretation_rule": (
                "raw correlations measure common task-plus-family texture; language-centered correlations remove "
                "each language's common mean and are the stricter family-specific comparison"
            ),
            "warning": "coordinate correlation is descriptive reuse, not a semantic decoder or causal proof",
        },
        "files": {
            "manifest": "field/manifest.json",
            "answer_fields": str(answer_path.relative_to(OUT)).replace("\\", "/"),
            "exemplar_fields": str(exemplar_path.relative_to(OUT)).replace("\\", "/"),
            "coordinate_graphs": str(matrices_path.relative_to(OUT)).replace("\\", "/"),
        },
        "claim_boundary": {
            "supported": "maps reuse and differentiation of full-coordinate interaction texture across 20 natural lexical family/language groups",
            "not_supported": "the task-reduced family texture is the general mechanism of preference, taxonomy, translation, reference, or syntax",
        },
        "language_mechanism_closed": False,
    }
    checks = {
        "phase2588_qualified": load_json(P2588 / "analysis/final.json")["adjudication"]["behavior_bridge_qualified"],
        "selected_40_quartets": len(manifest) == 40,
        "selected_160_prompts": len(manifest) * 4 == 160,
        "all_20_groups": len(grouped) == 20,
        "two_surface_replicates_each": all(len(grouped[label]) == 2 for label in labels),
        "all_raw_fields_exist": all((ROOT / item["path"]).is_file() for item in manifest),
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
    heading = f"## Phase {PHASE}: 双语十语言操作族逐token全坐标复用—分化图谱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    field = result["field"]
    graph = result["coordinate_graph"]
    inter = result["interaction"]
    text = rf"""


{heading} [{stamp}]

**测试原理。** 从Phase2588的20个`语言操作族/语言`单元各冻结surface 0与3两个行为全对、逐token等长四元组。对embedding、全部block、全部token、全部2560物理坐标保存四格场，并在answer boundary计算：

$$I_{{ltd}}=H^{{11}}_{{ltd}}-H^{{10}}_{{ltd}}-H^{{01}}_{{ltd}}+H^{{00}}_{{ltd}},\qquad
G_{{gld}}=\frac12\sum_{{s\in\{{0,3\}}}}\sigma_s I_{{sld}},$$

$$C^{{raw}}_l(g,h)=\operatorname{{corr}}_d(G_{{gld}},G_{{hld}}),\qquad
\widetilde G_{{gld}}=G_{{gld}}-\frac1{{10}}\sum_{{h:\,lang(h)=lang(g)}}G_{{hld}}.$$

其中$\sigma_s$只校正绑定奇偶朝向。raw相关包含公共四选一任务纹理；同语言均值中心化后的相关才是更严格的族特异比较。相关只描述同一固定物理基底的坐标纹理复用，不命名齿轮、也不证明因果。

**测试用例与全量性。** 十族×中英×两个相距最远表面=40四元组、160条prompt，全部来自Phase2588的292个合格四元组。每组原场为`[4,{field['hidden_states']},token,{field['d_model']}]`，token长度{field['prompt_token_range'][0]}—{field['prompt_token_range'][1]}，零padding；共{field['raw_bytes']} bytes。原场float16，有限差分、RMS与相关用float32/float64；主分析不做PCA、压缩或Top-K。

**结果汇总。** embedding交互最大绝对值={inter['embedding_maxabs']:.8g}，query前因果前缀最大绝对值={inter['causal_prefix_maxabs']:.8g}；answer交互逐层RMS=`{json.dumps(inter['median_answer_rms_by_hidden_state'])}`。两个表面复现、匹配/不匹配双语族、同语言跨族的全层曲线=`{json.dumps(graph['curves'], ensure_ascii=False)}`。预注册末四分之一HiddenState从{graph['late_hidden_state_start']}开始，均值=`{json.dumps(graph['late_summary'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2589_c467201_c483584_bilingual_operation_fullcoordinate_atlas.py`；40份逐token全坐标原场、SHA-256 manifest、answer全坐标数组、exemplar具体参数、20×20逐层坐标图与final位于`{OUT}`。

**分析与理论进展。** 这是第一次把“语言模式族图谱”落实为同一模型固定坐标上的20节点逐层图：表面复现回答同族换模板是否复用；raw双语匹配与不匹配回答公共任务纹理；语言中心化后匹配族相对不匹配族的差值才检验跨语言族特异残差。任何高raw相关都先按公共查表/候选输出解释，只有中心化对角优势才可作为较弱的族特异拼图。

**问题硬伤。** 每个节点只有两个表面复现；所有十族仍共享四事实—四候选骨架；英文与中文句长、token化和自然度不同；四格有限差分混合注意力竞争、一般非线性与语义条件计算；全坐标相关不能给单坐标命名，也没有证明必要性或充分性。下一步应先把本图加入客户端，然后把相同操作族扩展到无候选开放生成、长句重排和跨上下文指代，再检验这里的坐标图是否保留。

**结论。** `{result['claim_boundary']['supported']}`。不支持：`{result['claim_boundary']['not_supported']}`。检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    _, selected = load_and_select()
    manifest_path = OUT / "field/manifest.json"
    old = load_json(manifest_path) if manifest_path.is_file() else []
    reusable = len(old) == len(selected) and all((ROOT / item["path"]).is_file() for item in old)
    model = tokenizer = None
    try:
        if not reusable:
            model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        manifest, arrays, region_rms, answer_path, exemplar_path = collect(model, tokenizer, selected)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    result = analyze(manifest, arrays, region_rms, answer_path, exemplar_path)
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({key: result[key] for key in (
        "phase", "selection", "field", "interaction", "coordinate_graph", "checks",
        "all_checks_passed", "language_mechanism_closed")}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
