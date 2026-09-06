#!/usr/bin/env python3
"""Independent entity lockbox for the behavior-qualified English reverse-table stratum."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2554_c178433_c182528_independent_relation_lockbox_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
P2553 = RESULT / "phase2553_c174337_c178432_relation_slot_scaffold_adjudication/analysis/final.json"
PHASE, CAMPAIGN = 2554, "C178433-C182528"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2538_c117505_c121600_token_atomic_hypergraph_behavior as old_atlas  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402
import phase2553_c174337_c178432_relation_slot_scaffold_adjudication as p2553  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def compile_material(tokenizer) -> list[dict]:
    """Reuse the frozen compiler but substitute unseen entity names before tokenization."""
    original = old_atlas.NAMES[35]
    old_atlas.NAMES[35] = {"en": ("Copper Lynx", "Azure Heron"), "zh": ("铜色猞猁", "蔚蓝苍鹭")}
    rows = []
    try:
        for family_id in range(32):
            for binding in (0, 1):
                for relation_form in ("natural", "nonce"):
                    for value_form in ("natural", "nonce"):
                        for query_relation in (0, 1):
                            for query_value in (0, 1):
                                for condition in ("full_scaffold", "relation_missing", "value_missing"):
                                    row = p2553.compile_row(tokenizer, family_id=family_id, language="en", surface=1,
                                        binding=binding, relation_form=relation_form, value_form=value_form,
                                        query_relation=query_relation, query_value=query_value, condition=condition)
                                    row["unit"] = 36
                                    row["case_id"] = "u36_" + row["case_id"]
                                    row["base_case_id"] = "u36_" + row["base_case_id"]
                                    rows.append(row)
    finally:
        old_atlas.NAMES[35] = original
    return rows


def autonomous(model, tokenizer, rows: list[dict]) -> list[dict]:
    selected = [row for row in rows if row["ablation"] == "full_scaffold" and row["binding"] == 0
                and row["query_relation"] == 1 and row["query_value"] == 0]
    device = model.get_input_embeddings().weight.device
    output = []
    for start in range(0, len(selected), 8):
        batch = selected[start:start + 8]
        ids, mask, _ = p2552.left_pad([row["prompt_ids"] for row in batch], tokenizer.pad_token_id, device)
        width = ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=10, do_sample=False,
                                       use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
        for row, sequence in zip(batch, generated):
            text = tokenizer.decode(sequence[width:].cpu().tolist(), skip_special_tokens=True)
            normalized = p2552.norm(text)
            hits = [index for index, entity in enumerate(row["entities"]) if p2552.norm(entity) in normalized]
            prediction = hits[0] if len(set(hits)) == 1 else None
            output.append({"case_id": row["case_id"], "family_id": row["family_id"],
                           "relation_form": row["relation_form"], "value_form": row["value_form"],
                           "target_index": row["target_index"], "prediction_index": prediction,
                           "generated": text, "correct": prediction == row["target_index"]})
    return output


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 新实体独立锁箱的关系和值联合行为复验（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 不从Phase2553事后挑case，而是冻结其行为合格的英文、surface1、full-scaffold材料层，换成未参与发现的实体`Copper Lynx/Azure Heron`。覆盖32族、双binding、自然/nonce关系、自然/nonce值和四种query，共1024个full case；同格另做relation缺失与value缺失，共3072 case、6144条完整多token候选评分。自主锁箱固定binding0、query$(r=1,v=0)$，覆盖32族和四种词面组合共128条。

$$
e^*=r_q\oplus v_q\oplus b,\qquad
\operatorname{{Eligible}}(x)=\mathbf 1[\hat e_{{base}}=e^*_{{base}}\land \hat e_{{donor}}=e^*_{{donor}}].
$$

**结果汇总。** `{json.dumps(result['summary'], ensure_ascii=False)}`。裁决为`{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查为`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2554_c178433_c182528_independent_relation_lockbox_behavior.py`；新实体token材料、候选结果、自主输出和final位于`{OUT}`。

**分析与理论进展。** 只有新实体full准确率继续通过且relation/value缺失都回到机会水平，后续才使用该层做因果锁箱。自然/nonce四格被完整保留，以判断内部阶段规律是词义依赖还是结构复用。自主失败不删除candidate合格case，但会限制自主递归结论。

**问题硬伤与结论。** surface1来自Phase2553中的合格层，故本Phase只锁箱实体身份而不是全新格式；R0/R1脚手架仍可能主导；英文通过不能外推中文。它给出干净的关系条件检索显微镜，不等于自然语言关系机制已闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    previous = load(P2553)
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        material = compile_material(tokenizer)
        behavior = p2553.score_candidates(model, tokenizer, material, batch_size=32)
        generated = autonomous(model, tokenizer, material)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
    material_path = OUT / "material/u36_lockbox_token_atomic.jsonl"
    behavior_path = OUT / "behavior/u36_candidate_scores.jsonl"
    autonomous_path = OUT / "behavior/u36_autonomous.jsonl"
    p2552.write(material_path, material)
    p2552.write(behavior_path, behavior)
    p2552.write(autonomous_path, generated)
    summary = {}
    for condition in ("full_scaffold", "relation_missing", "value_missing"):
        subset = [row for row in behavior if row["ablation"] == condition]
        summary[condition] = {"n": len(subset), "accuracy": float(np.mean([row["correct"] for row in subset])),
                              "mean_margin": float(np.mean([row["target_minus_wrong"] for row in subset]))}
    full = [row for row in behavior if row["ablation"] == "full_scaffold"]
    summary["full_by_form"] = {f"r={rf},v={vf}": float(np.mean([row["correct"] for row in full
        if row["relation_form"] == rf and row["value_form"] == vf])) for rf in ("natural", "nonce") for vf in ("natural", "nonce")}
    summary["full_by_query"] = {f"r{r}v{v}": float(np.mean([row["correct"] for row in full
        if row["query_relation"] == r and row["query_value"] == v])) for r in (0, 1) for v in (0, 1)}
    summary["autonomous_n"] = len(generated)
    summary["autonomous_accuracy"] = float(np.mean([row["correct"] for row in generated]))
    base = {(row["family_id"], row["relation_form"], row["value_form"], row["query_relation"], row["query_value"]): row
            for row in full if row["binding"] == 0}
    donor = {(row["family_id"], row["relation_form"], row["value_form"], row["query_relation"], row["query_value"]): row
             for row in full if row["binding"] == 1}
    eligible = [key for key in base if base[key]["correct"] and donor[key]["correct"]]
    summary["paired_base_donor_total"] = len(base)
    summary["paired_base_donor_eligible"] = len(eligible)
    adjudication = {"independent_full_gate": summary["full_scaffold"]["accuracy"] >= .80,
                    "relation_necessary": summary["relation_missing"]["accuracy"] <= .65,
                    "value_necessary": summary["value_missing"]["accuracy"] <= .65,
                    "at_least_384_eligible_pairs": len(eligible) >= 384,
                    "autonomous_gate": summary["autonomous_accuracy"] >= .80,
                    "natural_language_mechanism_closed": False}
    checks = {"phase2553_complete": previous["all_checks_passed"], "full_rows_1024": len(full) == 1024,
              "all_rows_3072": len(material) == 3072, "candidate_complete": len(behavior) == len(material),
              "autonomous_128": len(generated) == 128, "all_forms_queries": len(summary["full_by_form"]) == 4
              and len(summary["full_by_query"]) == 4, "scientific_outcome_does_not_abort": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "summary": summary,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values()),
              "files": {"material": str(material_path), "behavior": str(behavior_path), "autonomous": str(autonomous_path)}}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
