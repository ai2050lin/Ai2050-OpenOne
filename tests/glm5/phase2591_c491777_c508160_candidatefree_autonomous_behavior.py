#!/usr/bin/env python3
"""Candidate-free greedy-generation behavior gate for 20 bilingual operation groups."""
from __future__ import annotations

import gc
import hashlib
import json
import re
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
P2590 = RESULT / "phase2590_c483585_c491776_bilingual_family_client_atlas/analysis/final.json"
OUT = RESULT / "phase2591_c491777_c508160_candidatefree_autonomous_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2591, "C491777-C508160"
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2580_c356609_c364800_fourchoice_relation_value_behavior as p2580  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())


def remove_candidate_list(row: dict, tokenizer) -> dict:
    if row["language"] == "en":
        if "\nOptions:" not in row["prompt"]:
            raise RuntimeError("English option marker missing")
        prompt = row["prompt"].split("\nOptions:", 1)[0] + "\nReturn only the exact complete codename. Answer:"
    else:
        if "\n选项：" not in row["prompt"]:
            raise RuntimeError("Chinese option marker missing")
        prompt = row["prompt"].split("\n选项：", 1)[0] + "\n只返回完整且精确的代号。答案:"
    result = {key: row[key] for key in (
        "family_id", "family", "language", "surface", "binding_relation", "binding_value",
        "query_relation", "query_value", "entities", "target_index", "target",
    )}
    result.update({
        "case_id": row["case_id"] + "_candidatefree",
        "base_case_id": row["case_id"] + "_candidatefree",
        "ablation": "full",
        "binding_id": row["binding_id"],
        "relation_form": "natural",
        "value_form": "natural",
        "donor_indices": row["donor_indices"],
        "prompt": prompt,
        "prompt_ids": [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)],
        "answer_boundary_token": len(tokenizer.encode(prompt, add_special_tokens=False)) - 1,
        "candidate_list_in_prompt": False,
    })
    return result


def compile_material(tokenizer):
    rows = [json.loads(line) for line in (P2588 / "material/cases.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row["ablation"] == "full" and row["surface"] in (0, 3)]
    material = [remove_candidate_list(row, tokenizer) for row in selected]
    if len(material) != 640:
        raise RuntimeError(f"expected 640 candidate-free prompts, got {len(material)}")
    return material


def left_pad(sequences, pad_id, device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, width - len(sequence):] = 1
    return ids, mask


def generate(model, tokenizer, material, batch_size=8):
    device = model.get_input_embeddings().weight.device
    output = []
    for start in range(0, len(material), batch_size):
        batch = material[start:start + batch_size]
        ids, mask = left_pad([row["prompt_ids"] for row in batch], tokenizer.pad_token_id, device)
        width = ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=12,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row, sequence in zip(batch, generated):
            generated_ids = sequence[width:].detach().cpu().tolist()
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            hits = [index for index, entity in enumerate(row["entities"]) if norm(entity) in norm(text)]
            parsed = hits[0] if len(set(hits)) == 1 else None
            first_line = text.strip().splitlines()[0] if text.strip() else ""
            strict = norm(first_line) == norm(row["target"])
            output.append({
                "case_id": row["case_id"], "family_id": row["family_id"], "family": row["family"],
                "language": row["language"], "surface": row["surface"],
                "binding_relation": row["binding_relation"], "binding_value": row["binding_value"],
                "query_relation": row["query_relation"], "query_value": row["query_value"],
                "target_index": row["target_index"], "prediction_index": parsed,
                "generated_token_ids": generated_ids, "generated": text,
                "strict_exact": strict, "parsed_correct": parsed == row["target_index"],
            })
        if (start + len(batch)) % 160 == 0 or start + len(batch) == len(material):
            print(f"[phase2591 generate] {start + len(batch)}/{len(material)}", flush=True)
    return output


def summarize(material, candidate, generated):
    candidate_meta = {row["case_id"]: row for row in candidate}
    generated_meta = {row["case_id"]: row for row in generated}

    def metrics(case_ids):
        c = [candidate_meta[case_id] for case_id in case_ids]
        g = [generated_meta[case_id] for case_id in case_ids]
        return {
            "n": len(case_ids),
            "candidate_likelihood_accuracy": float(np.mean([row["correct"] for row in c])),
            "greedy_parsed_accuracy": float(np.mean([row["parsed_correct"] for row in g])),
            "greedy_strict_exact_accuracy": float(np.mean([row["strict_exact"] for row in g])),
            "mean_candidate_margin": float(np.mean([row["target_minus_best_wrong"] for row in c])),
        }

    overall = metrics([row["case_id"] for row in material])
    by_group = {}
    for family in sorted({row["family"] for row in material}):
        for language in ("en", "zh"):
            subset = [row["case_id"] for row in material if row["family"] == family and row["language"] == language]
            by_group[f"{family}/{language}"] = metrics(subset)
    by_surface = {str(surface): metrics([row["case_id"] for row in material if row["surface"] == surface])
                  for surface in (0, 3)}
    index = {(row["family_id"], row["language"], row["surface"], row["binding_relation"],
              row["binding_value"], row["query_relation"], row["query_value"]): row for row in material}
    eligible = []
    for prefix in sorted({key[:5] for key in index}):
        cells = [index[prefix + cell] for cell in CELLS]
        if len({len(row["prompt_ids"]) for row in cells}) != 1:
            continue
        if all(generated_meta[row["case_id"]]["parsed_correct"] for row in cells):
            eligible.append(prefix)
    return overall, by_group, by_surface, eligible


def append_memo(result):
    heading = f"## Phase {PHASE}: 双语十语言操作族无候选自主生成行为门（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** Phase2589晚层raw族图被公共四选一骨架主导。本Phase在不改变四事实与查询的前提下，从prompt中彻底删除四个候选代号列表，让Qwen3-4B用真实greedy token路径自主生成代号；同时仅由评估器计算四个完整候选序列似然，区分“内部可判别”与“自主编译为输出”：

$$\hat e_{{score}}=\arg\max_e\frac1{{|y_e|}}\sum_t\log p(y_{{e,t}}\mid x,y_{{e,<t}}),\qquad
\hat y_{{\mathrm{{greedy}},t}}=\arg\max_v p(v\mid x,\hat y_{{<t}}).$$

**测试用例。** 十族×中英×surface 0/3×四binding×四query=640条无候选prompt；评估器比较2560条完整代号序列，模型另对全部640条greedy生成最多12 token。报告严格首行精确率与唯一代号解析率；解析只允许事实中四代号恰有一个命中，不以候选列表提示模型。

**结果汇总。** overall=`{json.dumps(result['overall'], ensure_ascii=False)}`；20族/语言=`{json.dumps(result['by_family_language'], ensure_ascii=False)}`；双surface=`{json.dumps(result['by_surface'], ensure_ascii=False)}`；自主生成四格全对且等长四元组{result['eligible_autonomous_quartets']}个，分布=`{json.dumps(result['eligible_by_family_language'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2591_c491777_c508160_candidatefree_autonomous_behavior.py`；640条token材料、2560候选分数、640条真实生成token、eligible清单、哈希与final位于`{OUT}`。

**分析与理论进展。** candidate likelihood高而greedy低，表示prompt边界已有可读排序但输出动力学不能稳定编译；二者都高才允许在真实自主生成边界复测Phase2589族图。按族/语言报告可防止总体均值掩盖某类接口失败。

**问题硬伤。** 去掉候选不等于开放世界；目标仍是事实中出现过的短代号，可由复制实现；四事实交叉格未变；唯一字符串解析比严格格式宽松；greedy的12-token预算可能截断解释性输出。该Phase仍是行为门，不解释内部齿轮。

**结论。** `{result['claim_boundary']}`。检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        material = compile_material(tokenizer)
        candidate = p2580.score_candidates(model, tokenizer, material, batch_size=32)
        # Restore Phase2591 axes omitted by the reused scorer.
        axes = {row["case_id"]: (row["language"], row["surface"]) for row in material}
        for row in candidate:
            row["language"], row["surface"] = axes[row["case_id"]]
        generated = generate(model, tokenizer, material)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    material_path = OUT / "material/candidatefree_prompts.jsonl"
    candidate_path = OUT / "behavior/candidate_likelihood.jsonl"
    generated_path = OUT / "behavior/greedy_generation.jsonl"
    write_jsonl(material_path, material)
    write_jsonl(candidate_path, candidate)
    write_jsonl(generated_path, generated)
    overall, by_group, by_surface, eligible = summarize(material, candidate, generated)
    eligible_path = OUT / "material/eligible_autonomous_quartets.json"
    save_json(eligible_path, {"prefix_fields": ["family_id", "language", "surface", "binding_relation", "binding_value"],
                              "eligible": eligible})
    counts = defaultdict(int)
    for family_id, language, *_ in eligible:
        family = next(row["family"] for row in material if row["family_id"] == family_id)
        counts[f"{family}/{language}"] += 1
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "design": {"families": 10, "languages": 2, "surfaces": 2, "prompts": len(material),
                   "candidate_sequences": len(material) * 4, "greedy_generations": len(generated),
                   "max_new_tokens": 12, "candidate_list_in_prompt": False},
        "overall": overall, "by_family_language": by_group, "by_surface": by_surface,
        "eligible_autonomous_quartets": len(eligible), "eligible_by_family_language": dict(counts),
        "claim_boundary": (
            "measures candidate-free copying/selection and autonomous output compilation inside the structured four-fact task; "
            "it is not open-world generation or a direct test of ten general language abilities"
        ),
        "files": {"material": str(material_path.relative_to(OUT)).replace("\\", "/"),
                  "candidate": str(candidate_path.relative_to(OUT)).replace("\\", "/"),
                  "generated": str(generated_path.relative_to(OUT)).replace("\\", "/"),
                  "eligible": str(eligible_path.relative_to(OUT)).replace("\\", "/")},
        "hashes": {"material": sha256(material_path), "candidate": sha256(candidate_path),
                   "generated": sha256(generated_path)},
        "language_mechanism_closed": False,
    }
    result["checks"] = {
        "phase2590_complete": load_json(P2590)["all_checks_passed"],
        "all_640_prompts": len(material) == 640,
        "all_2560_candidate_sequences": len(candidate) * 4 == 2560,
        "all_640_greedy_generations": len(generated) == 640,
        "no_candidate_list_in_prompts": all("Options:" not in row["prompt"] and "选项：" not in row["prompt"] for row in material),
        "target_balanced": len({sum(row["target_index"] == index for row in material) for index in range(4)}) == 1,
        "all_20_family_language_groups": len(by_group) == 20,
        "scientific_result_does_not_abort": True,
        "claim_boundary": True,
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
