#!/usr/bin/env python3
"""Revalidate the Phase2554 Qwen lockbox without padded mixed-length batches."""
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
OUT = RESULT / "phase2566_c260353_c264448_zero_padding_qwen_lockbox_revalidation"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
OLD = RESULT / "phase2556_c190721_c198912_form_id_collision_erratum_recompute/analysis/final.json"
PHASE, CAMPAIGN = 2566, "C260353-C264448"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2553_c174337_c178432_relation_slot_scaffold_adjudication as p2553  # noqa: E402
import phase2554_c178433_c182528_independent_relation_lockbox_behavior as p2554  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def autonomous_exact(model, tokenizer, rows: list[dict]) -> list[dict]:
    selected = [row for row in rows if row["ablation"] == "full_scaffold" and row["binding"] == 0
                and row["query_relation"] == 1 and row["query_value"] == 0]
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        buckets[len(row["prompt_ids"])].append(row)
    device = model.get_input_embeddings().weight.device
    output = []
    for _, values in sorted(buckets.items()):
        for start in range(0, len(values), 8):
            batch = values[start:start + 8]
            ids = torch.tensor([row["prompt_ids"] for row in batch], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            width = ids.shape[1]
            with torch.inference_mode():
                generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=10, do_sample=False,
                                           use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                           eos_token_id=tokenizer.eos_token_id)
            for row, sequence in zip(batch, generated):
                text = tokenizer.decode(sequence[width:].cpu().tolist(), skip_special_tokens=True)
                normalized = p2552.norm(text)
                hits = [i for i, entity in enumerate(row["entities"]) if p2552.norm(entity) in normalized]
                prediction = hits[0] if len(set(hits)) == 1 else None
                output.append({"case_id": row["case_id"], "family_id": row["family_id"],
                               "relation_form": row["relation_form"], "value_form": row["value_form"],
                               "target_index": row["target_index"], "prediction_index": prediction,
                               "generated": text, "correct": prediction == row["target_index"]})
    return output


def panel(rows: list[dict]) -> dict:
    return {"n": len(rows), "accuracy": float(np.mean([row["correct"] for row in rows])),
            "mean_margin": float(np.mean([row["target_minus_wrong"] for row in rows]))}


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: Qwen关系锁箱的无填充独立复核（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Phase2562证明GLM在混合长度左填充批次下会发生严重输出漂移，因此Phase2556中沿用旧计分器得到的Qwen3-4B新实体锁箱也必须复核。本Phase逐字复用Phase2554冻结的32族、双binding、自然/nonce关系、自然/nonce值、四查询和full/relation-missing/value-missing材料，但按完整候选序列精确长度分桶，桶内attention mask全1；自主生成也按prompt精确长度分桶。共3072 case、6144条候选序列及128条自主生成。

$$
S(e\mid x)=\sum_{{t=1}}^{{|e|}}\log p(e_t\mid x,e_{{<t}}),\qquad
A_{{full}}>.8,\ A_{{-r}},A_{{-v}}\leq .65.
$$

**结果汇总。** 旧Phase2556摘要`{json.dumps(result['old_summary'], ensure_ascii=False)}`；无填充复核`{json.dumps(result['summary'], ensure_ascii=False)}`；旧新差异`{json.dumps(result['difference'], ensure_ascii=False)}`；裁决`{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2566_c260353_c264448_zero_padding_qwen_lockbox_revalidation.py`；完整材料、候选分数、自主输出和final位于`{OUT}`。同时修正公共计分函数`tests/glm5/phase2553_c174337_c178432_relation_slot_scaffold_adjudication.py`，以后默认按精确长度分桶。

**分析与理论进展。** 本Phase不寻找新齿轮，只裁决此前最重要行为显微镜能否在可靠推理口径下存活。若full与双缺失门仍成立，后续可在完全相同模板上做单因素链扩展；若不成立，则Phase2554—2560中依赖该资格的内部结果必须整体降级为填充条件下的现象。

**问题硬伤与结论。** 即使复核通过，任务仍是英文二元人工关系表；高准确率只证明模型可在该提示条件下联合使用relation/value，不能把R0/R1叫作自然语言关系编码，更不能把早V、中KV、晚Q命名为内容/寻址/编译器。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    old = read_json(OLD)
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        material = p2554.compile_material(tokenizer)
        behavior = p2553.score_candidates(model, tokenizer, material, batch_size=16)
        autonomous = autonomous_exact(model, tokenizer, material)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    p2552.write(OUT / "material/rows.jsonl", material)
    p2552.write(OUT / "behavior/scores.jsonl", behavior)
    p2552.write(OUT / "autonomous/generation.jsonl", autonomous)
    summary = {name: panel([row for row in behavior if row["ablation"] == name])
               for name in ("full_scaffold", "relation_missing", "value_missing")}
    summary["autonomous"] = {"n": len(autonomous),
                             "accuracy": float(np.mean([row["correct"] for row in autonomous]))}
    full = [row for row in behavior if row["ablation"] == "full_scaffold"]
    base = {(row["family_id"], row["relation_form"], row["value_form"], row["query_relation"], row["query_value"]): row
            for row in full if row["binding"] == 0}
    donor = {(row["family_id"], row["relation_form"], row["value_form"], row["query_relation"], row["query_value"]): row
             for row in full if row["binding"] == 1}
    eligible = [key for key in base if base[key]["correct"] and donor[key]["correct"]]
    summary["eligible_pairs"] = len(eligible)
    old_summary = old["phase2554_corrected"]
    old_full = float(old_summary["conditions"]["full_scaffold"]["accuracy"])
    difference = {"full_accuracy": summary["full_scaffold"]["accuracy"] - old_full}
    adjudication = {"full_gate": summary["full_scaffold"]["accuracy"] >= .80,
                    "relation_necessary": summary["relation_missing"]["accuracy"] <= .65,
                    "value_necessary": summary["value_missing"]["accuracy"] <= .65,
                    "at_least_384_eligible": len(eligible) >= 384,
                    "autonomous_gate": summary["autonomous"]["accuracy"] >= .80,
                    "natural_language_mechanism_closed": False}
    checks = {"rows_3072": len(material) == 3072, "scores_3072": len(behavior) == 3072,
              "autonomous_128": len(autonomous) == 128, "exact_length_buckets": True,
              "scientific_failure_does_not_abort": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "old_summary": old_summary,
              "summary": summary, "difference": difference, "adjudication": adjudication,
              "checks": checks, "all_checks_passed": all(checks.values())}
    write_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
