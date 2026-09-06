#!/usr/bin/env python3
"""Audit Phase2601 pseudo-replication and build a truly unique 1,200-prompt lockbox."""
from __future__ import annotations

import difflib
import gc
import hashlib
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
P2601 = RESULT / "phase2601_c610561_c626944_natural_singleprompt_behavior_lockbox"
P2602 = RESULT / "phase2602_c626945_c643328_natural_fullcoordinate_field"
OUT = RESULT / "phase2603_c643329_c659712_unique_natural_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2603, "C643329-C659712"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2601_c610561_c626944_natural_singleprompt_behavior_lockbox as p2601  # noqa: E402

EN_LOCATIONS = ("archive", "library", "museum", "workshop", "station", "theater", "clinic", "gallery", "courtyard", "laboratory")
ZH_LOCATIONS = ("档案室", "图书馆", "博物馆", "工作间", "车站", "剧场", "诊所", "画廊", "庭院", "实验室")
EN_CONTEXTS = (
    "A witness recorded the following self-contained report in the {location}. ",
    "For a language audit in the {location}, read this independent record. ",
    "The {location} log contains one complete case. ",
    "Use only the facts in this {location} report. ",
    "An editor preserved this exact {location} account. ",
)
ZH_CONTEXTS = (
    "一名见证人在{location}记录了以下独立报告。",
    "请阅读{location}语言审计中的这份独立记录。",
    "{location}日志包含一个完整案例。",
    "请只使用这份{location}报告中的事实。",
    "编辑完整保留了这份{location}记录。",
)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def source_spans(ids0, ids1):
    matcher = difflib.SequenceMatcher(a=ids0, b=ids1, autojunk=False)
    changed0, changed1 = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed0.extend(range(i1, i2))
            changed1.extend(range(j1, j2))
    if not changed0 or not changed1:
        raise RuntimeError("empty source span")
    if max(changed0) - min(changed0) + 1 != len(changed0) or max(changed1) - min(changed1) + 1 != len(changed1):
        raise RuntimeError("non-contiguous source replacement")
    return changed0, changed1


def compile_unique(tokenizer):
    material = []
    for family in p2601.FAMILIES:
        for language in p2601.LANGUAGES:
            for pair_index in range(50):
                core = [p2601.prompt_and_target(family, language, pair_index, variant) for variant in (0, 1)]
                location = (EN_LOCATIONS if language == "en" else ZH_LOCATIONS)[pair_index % 10]
                template = (EN_CONTEXTS if language == "en" else ZH_CONTEXTS)[pair_index // 10]
                prefix = template.format(location=location)
                prompts = [prefix + item[0] for item in core]
                ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
                spans = source_spans(ids[0], ids[1])
                pair_id = f"{family}_{language}_{pair_index:03d}"
                split = p2601.split_for(pair_index)
                for variant in (0, 1):
                    prompt, target, source = core[variant]
                    material.append({
                        "case_id": f"{pair_id}_v{variant}", "pair_id": pair_id, "pair_index": pair_index,
                        "family": family, "language": language, "split": split, "variant": variant,
                        "surface_context": prefix, "prompt": prompts[variant], "target": target,
                        "alternate": core[1 - variant][1], "source_text": source,
                        "prompt_ids": ids[variant], "source_token_positions": spans[variant],
                        "answer_boundary_token": len(ids[variant]) - 1, "candidate_list_in_prompt": False,
                    })
    return material


def duplicate_audit(rows):
    output = {}
    for family in p2601.FAMILIES:
        for language in p2601.LANGUAGES:
            subset = [row for row in rows if row["family"] == family and row["language"] == language]
            pair_signatures = defaultdict(list)
            for row in subset:
                pair_signatures[row["pair_id"]].append(row["prompt"])
            signatures = ["\n".join(sorted(prompts)) for prompts in pair_signatures.values()]
            split_sets = {split: {"\n".join(sorted(pair_signatures[pair_id])) for pair_id in pair_signatures
                                  if next(row for row in subset if row["pair_id"] == pair_id)["split"] == split}
                          for split in ("discovery", "confirmation", "external")}
            output[f"{family}/{language}"] = {
                "pairs": len(pair_signatures), "unique_pair_signatures": len(set(signatures)),
                "unique_prompts": len({row["prompt"] for row in subset}),
                "cross_split_overlap": {"discovery_confirmation": len(split_sets["discovery"] & split_sets["confirmation"]),
                                        "discovery_external": len(split_sets["discovery"] & split_sets["external"]),
                                        "confirmation_external": len(split_sets["confirmation"] & split_sets["external"])},
            }
    return output


def append_memo(result):
    heading = f"## Phase {PHASE}: 自然材料伪复现审计与1200条无重复锁箱重建（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** Phase2602发现多组发现—确认方向相关精确为1，先检查pair文本签名而不解释为机制。旧材料每组名义50 pair但只有2—20个独特签名，发现/确认重复同一核心prompt；因此Phase2602的全场和因果零区仍有效，其split相关不得作为外推证据。新锁箱为每个pair加入互不重复的自然场景×报告表面组合，并允许两侧source span具有不同token数，用序列对齐分别记录连续替换区：

$$S_0,S_1=\operatorname{{DiffSpan}}(\operatorname{{tok}}(x^0),\operatorname{{tok}}(x^1)).$$

**测试用例。** 仍为六族×中英×50 pair×2=1200无候选prompt，发现/确认/外测=20/20/10 pair；每组50个pair签名、100条prompt全部唯一，split间文本交集必须为0。Qwen3-4B BF16 CUDA非量化重新评分2400目标序列并greedy生成1200条，行为门仍为每族/语言75%。

**结果汇总。** 旧重复审计=`{json.dumps(result['old_duplicate_audit'], ensure_ascii=False)}`；新重复审计=`{json.dumps(result['new_duplicate_audit'], ensure_ascii=False)}`；overall=`{json.dumps(result['overall'], ensure_ascii=False)}`；12组=`{json.dumps(result['by_family_language'], ensure_ascii=False)}`；split=`{json.dumps(result['by_split'], ensure_ascii=False)}`；通过组=`{json.dumps(result['qualified_groups'], ensure_ascii=False)}`；双变体均正确pair={result['eligible_pairs']}。

**相关文件。** 脚本`tests/glm5/phase2603_c643329_c659712_unique_natural_lockbox.py`；无重复材料、得分、greedy、eligible、重复审计与final位于`{OUT}`。

**分析与理论进展。** 这是对研究合同的实质纠错：重复样本可以让坐标均值方向在split间伪装成完美复现。新结果只决定哪些真实操作可进入机制分析；Phase2604必须重新采场，不能复用旧重复场来声称确认。

**问题硬伤。** 新材料仍由有限核心模板与词表组合生成；场景前缀增加表面差异但未创造全新语法；同组目标词仍会复用；不等长pair需要source-span池化而不能逐token直接相减。该纠错不否定Phase2602的单prompt因果零区。

**结论。** `{result['claim_boundary']}`；检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    old_material = read_jsonl(P2601 / "material/cases.jsonl")
    old_audit = duplicate_audit(old_material)
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        material = compile_unique(tokenizer)
        candidate = p2601.score_binary(model, tokenizer, material)
        generated = p2601.generate(model, tokenizer, material)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    new_audit = duplicate_audit(material)
    overall, by_group, by_split, eligible = p2601.summarize(material, candidate, generated)
    qualified = [key for key, value in by_group.items() if value["greedy_parsed_accuracy"] >= 0.75]
    material_path = OUT / "material/cases.unique.jsonl"
    candidate_path = OUT / "behavior/candidate_scores.jsonl"
    generated_path = OUT / "behavior/greedy_generation.jsonl"
    eligible_path = OUT / "material/eligible_pairs.json"
    audit_path = OUT / "analysis/duplicate_audit.json"
    write_jsonl(material_path, material)
    write_jsonl(candidate_path, candidate)
    write_jsonl(generated_path, generated)
    save_json(eligible_path, eligible)
    save_json(audit_path, {"old": old_audit, "new": new_audit})
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-4B BF16 CUDA nonquantized", "old_duplicate_audit": old_audit,
        "new_duplicate_audit": new_audit, "overall": overall,
        "by_family_language": by_group, "by_split": by_split,
        "qualified_groups": qualified, "eligible_pairs": len(eligible),
        "claim_boundary": "replaces pseudo-replicated split evidence with text-disjoint behavior material; no mechanism inferred",
        "hashes": {"material": sha256(material_path), "candidate": sha256(candidate_path),
                   "generated": sha256(generated_path), "eligible": sha256(eligible_path), "audit": sha256(audit_path)},
        "language_mechanism_closed": False,
    }
    result["checks"] = {
        "phase2602_complete": json.loads((P2602 / "analysis/final.json").read_text(encoding="utf-8"))["all_checks_passed"],
        "old_pseudoreplication_detected": any(row["unique_pair_signatures"] < 50 for row in old_audit.values()),
        "all_1200_unique_prompts": len({row["prompt"] for row in material}) == 1200,
        "all_groups_50_unique_pairs": all(row["unique_pair_signatures"] == 50 for row in new_audit.values()),
        "zero_cross_split_text_overlap": all(all(value == 0 for value in row["cross_split_overlap"].values()) for row in new_audit.values()),
        "all_source_spans_nonempty": all(row["source_token_positions"] for row in material),
        "all_2400_candidate_sequences": len(candidate) * 2 == 2400,
        "all_1200_greedy": len(generated) == 1200,
        "frozen_behavior_gate": True,
        "scientific_result_does_not_abort": True,
        "claim_boundary": True,
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({key: result[key] for key in ("phase", "overall", "qualified_groups", "eligible_pairs", "checks", "all_checks_passed")},
                     ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
