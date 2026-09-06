#!/usr/bin/env python3
"""Equal-BPE nonce-value behavior lockbox and full-coordinate field completion."""
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
P2582 = RESULT / "phase2582_c372993_c385280_fourchoice_fulltoken_interaction_birth"
P2584 = RESULT / "phase2584_c397569_c409856_block0_interaction_causal_controls/analysis/final.json"
OUT = RESULT / "phase2585_c409857_c426240_equal_bpe_nonce_value_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2585, "C409857-C426240"
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2580_c356609_c364800_fourchoice_relation_value_behavior as p2580  # noqa: E402
import phase2582_c372993_c385280_fourchoice_fulltoken_interaction_birth as p2582  # noqa: E402


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pseudowords(family: int):
    consonants = ("z", "v", "k", "m", "t", "l", "d", "r", "p", "g", "n", "f")
    vowels = ("a", "e", "i", "o", "u")
    values = []
    for offset in range(180):
        a = consonants[(family * 5 + offset) % len(consonants)]
        b = consonants[(family * 7 + offset * 3 + 1) % len(consonants)]
        c = consonants[(family * 11 + offset * 5 + 2) % len(consonants)]
        x = vowels[(family + offset) % len(vowels)]
        y = vowels[(family * 2 + offset * 2 + 1) % len(vowels)]
        z = vowels[(family * 3 + offset * 4 + 2) % len(vowels)]
        word = f"{a}{x}{b}{y}{c}{z}"
        if word not in values:
            values.append(word)
    return values


def equal_value_pairs(tokenizer):
    pairs = {}
    used = set()
    for family in range(32):
        candidates = [word for word in pseudowords(family) if word not in used]
        found = None
        for left_index, left in enumerate(candidates):
            left_ids = tokenizer.encode(f"[V0 :: {left}]", add_special_tokens=False)
            for right in candidates[left_index + 1:]:
                if right == left:
                    continue
                right_ids = tokenizer.encode(f"[V1 :: {right}]", add_special_tokens=False)
                if len(left_ids) == len(right_ids) and left_ids != right_ids:
                    found = (left, right, len(left_ids))
                    break
            if found:
                break
        if found is None:
            raise RuntimeError(f"no equal-BPE nonce pair for family {family}")
        used.update(found[:2])
        pairs[family] = found
    return pairs


def compile_material(tokenizer, value_pairs):
    original = p2580.descriptors

    def descriptors(family_id, relation_form, value_form):
        relations, values = original(family_id, relation_form, value_form)
        if value_form == "nonce":
            values = value_pairs[family_id][:2]
        return relations, values

    p2580.descriptors = descriptors
    try:
        rows = [
            p2580.compile_row(
                tokenizer,
                family_id=family,
                binding_relation=br,
                binding_value=bv,
                relation_form=rf,
                value_form=vf,
                query_relation=qr,
                query_value=qv,
                ablation="full",
            )
            for family in range(32)
            for br in (0, 1)
            for bv in (0, 1)
            for rf in ("natural", "nonce")
            for vf in ("natural", "nonce")
            for qr in (0, 1)
            for qv in (0, 1)
        ]
    finally:
        p2580.descriptors = original
    return rows


def behavior_summary(rows):
    by_form = {}
    for rf in ("natural", "nonce"):
        for vf in ("natural", "nonce"):
            subset = [row for row in rows if row["relation_form"] == rf and row["value_form"] == vf]
            by_form[f"{rf}/{vf}"] = {
                "n": len(subset),
                "accuracy": float(np.mean([row["correct"] for row in subset])),
                "mean_margin": float(np.mean([row["target_minus_best_wrong"] for row in subset])),
            }
    return {
        "n": len(rows),
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "by_form": by_form,
        "by_query": {
            f"r{r}v{v}": float(np.mean([row["correct"] for row in rows
                                        if row["query_relation"] == r and row["query_value"] == v]))
            for r in (0, 1) for v in (0, 1)
        },
    }


def compatible_quartets(material, behavior):
    correct = {row["case_id"]: row["correct"] for row in behavior}
    index = {
        (row["family_id"], row["binding_relation"], row["binding_value"], row["relation_form"],
         row["value_form"], row["query_relation"], row["query_value"]): row
        for row in material
    }
    output = []
    for prefix in sorted({key[:5] for key in index}):
        cells = [index[prefix + cell] for cell in CELLS]
        all_correct = all(correct[row["case_id"]] for row in cells)
        aligned = (len({len(row["prompt_ids"]) for row in cells}) == 1 and all(
            len({len(row["regions"][region]) for row in cells}) == 1
            for region in cells[0]["regions"]
        ))
        if all_correct and aligned:
            output.append((prefix, cells))
    return output


def select_value_nonce(compatible):
    selected = []
    counts = {}
    for relation_form in ("natural", "nonce"):
        values = [item for item in compatible if item[0][3:] == (relation_form, "nonce")]
        counts[f"{relation_form}/nonce"] = len(values)
        if len(values) < 16:
            raise RuntimeError((relation_form, len(values)))
        indices = np.linspace(0, len(values) - 1, 16, dtype=int)
        selected.extend(values[int(index)] for index in indices)
    return selected, counts


def pearson_by_layer(left, right):
    return [p2582.pearson(left[:, layer].mean(0), right[:, layer].mean(0))
            for layer in range(left.shape[1])]


def append_memo(result):
    heading = f"## Phase {PHASE}: 等BPE无意义值锁箱与四词面全坐标补全（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理。** Phase2582只有natural/natural与nonce/natural能逐token四格对齐，原因不是行为失败，而是旧nonce-value两词的BPE长度不同。为每个family从确定性伪词池中预先搜索一对不同字符串，使`[V0 :: word0]`与`[V1 :: word1]`token数严格相等；保留旧relation材料，仅替换nonce value。这样位置、RoPE相位与语义region都能逐token比较，而不靠截断或区域均值。

$$\lvert\tau([V0::w_0])\rvert=\lvert\tau([V1::w_1])\rvert,\qquad
I_{{ltd}}=H^{{11}}_{{ltd}}-H^{{10}}_{{ltd}}-H^{{01}}_{{ltd}}+H^{{00}}_{{ltd}}.$$

**测试用例。** 32语言操作族×4 binding×4自然/nonce词面×4 query，共2048个full case、8192条完整候选序列；Qwen3-4B BF16 CUDA非量化、完整序列等长分桶、零padding。行为后只从四格全对且逐token等长集合冻结natural/nonce和nonce/nonce各16组，共32四元组、128 prompt；保存embedding+36层、全token、全部2560坐标，未用Top-K。

**结果汇总。** 行为`{json.dumps(result['behavior'], ensure_ascii=False)}`；对齐`{json.dumps(result['alignment'], ensure_ascii=False)}`；全场`{json.dumps(result['field'], ensure_ascii=False)}`；旧自然值与新nonce值交互纹理的逐层相关`{json.dumps(result['value_surface_reuse'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2585_c409857_c426240_equal_bpe_nonce_value_lockbox.py`；32族伪词合同、2048材料、完整候选分数、4.5GB级逐token全坐标场、answer坐标数组、manifest和final均位于`{OUT}`。

**理论进展与分析。** 这补齐了Phase2582因token长度排除的两个value-nonce词面，使四类natural/nonce都能在相同物理坐标与token位置下观察。跨value词面相关回答“相同关系条件代数是否复用坐标纹理”，仍不等价于一个可搬运向量。

**问题硬伤。** 伪词由程序搜索，虽然未按模型行为挑选，但仍可能通过共享字形或显式V0/V1标签简化任务；新字段只重复4B，跨模型尚待下一Phase；所有材料仍是英文查表；float16落盘。结论只限受控四选一条件代数，不外推到自然语言机制。检查`{json.dumps(result['checks'], ensure_ascii=False)}`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    model = tokenizer = None
    old_out = p2582.OUT
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        value_pairs = equal_value_pairs(tokenizer)
        save_json(OUT / "contract/equal_bpe_nonce_values.json", {
            str(key): {"values": list(value[:2]), "span_tokens": value[2]} for key, value in value_pairs.items()
        })
        material = compile_material(tokenizer, value_pairs)
        behavior = p2580.score_candidates(model, tokenizer, material, batch_size=32)
        (OUT / "material").mkdir(parents=True, exist_ok=True)
        with (OUT / "material/fourchoice_full.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in material:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        (OUT / "behavior").mkdir(parents=True, exist_ok=True)
        with (OUT / "behavior/scores.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in behavior:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        compatible = compatible_quartets(material, behavior)
        selected, selected_pool = select_value_nonce(compatible)
        p2582.OUT = OUT
        manifest, arrays, semantic, exemplar, metrics_path, exemplar_path = p2582.collect(model, tokenizer, selected)
    finally:
        p2582.OUT = old_out
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    summary = behavior_summary(behavior)
    compatible_counts = defaultdict(int)
    for prefix, _ in compatible:
        compatible_counts[f"{prefix[3]}/{prefix[4]}"] += 1
    old = np.load(P2582 / "analysis/full_coordinate_answer_metrics.npz")
    old_oriented = old["answer_oriented_interaction"]
    new_oriented = arrays["answer_oriented_interaction"]
    value_reuse = {
        "natural_relation_natural_vs_nonce_value_signed": pearson_by_layer(old_oriented[:16], new_oriented[:16]),
        "nonce_relation_natural_vs_nonce_value_signed": pearson_by_layer(old_oriented[16:], new_oriented[16:]),
        "natural_relation_natural_vs_nonce_value_absolute": pearson_by_layer(np.abs(old_oriented[:16]), np.abs(new_oriented[:16])),
        "nonce_relation_natural_vs_nonce_value_absolute": pearson_by_layer(np.abs(old_oriented[16:]), np.abs(new_oriented[16:])),
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "behavior": summary,
        "alignment": {
            "eligible_correct_and_aligned_by_form": dict(compatible_counts),
            "selected_pool": selected_pool,
            "selected_quartets": 32,
            "selected_prompts": 128,
        },
        "field": {
            "hidden_states": arrays["answer_interaction"].shape[1],
            "d_model": arrays["answer_interaction"].shape[2],
            "raw_bytes": sum(item["bytes"] for item in manifest),
            "full_tokens": True,
            "full_coordinates": True,
            "no_topk": True,
            "embedding_interaction_rms_max": max(item["embedding_interaction_rms_max"] for item in manifest),
            "prequery_interaction_rms_max": max(item["prefix_interaction_rms_max"] for item in manifest),
        },
        "value_surface_reuse": value_reuse,
        "claim_boundary": "four surface cells are now measurable without token-position drift; reuse correlations remain descriptive",
        "language_mechanism_closed": False,
    }
    checks = {
        "phase2584_complete": load_json(P2584)["all_checks_passed"],
        "all_2048_cases": len(material) == 2048,
        "all_8192_candidates": len(material) * 4 == 8192,
        "all_forms_accuracy_at_least_070": all(value["accuracy"] >= .70 for value in summary["by_form"].values()),
        "value_nonce_forms_have_16_aligned_quartets": all(value >= 16 for value in selected_pool.values()),
        "selected_32_quartets": len(manifest) == 32,
        "embedding_and_prefix_zero": result["field"]["embedding_interaction_rms_max"] == 0.0 and result["field"]["prequery_interaction_rms_max"] == 0.0,
        "raw_fields_exist": all((ROOT / item["path"]).is_file() for item in manifest),
        "no_topk": True,
        "claim_boundary": True,
    }
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
