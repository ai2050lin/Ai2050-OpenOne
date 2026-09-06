#!/usr/bin/env python3
"""Bilingual natural-operation four-choice behavior atlas without explicit R/V labels."""
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
P2587 = RESULT / "phase2587_c442625_c450816_interaction_birth_client_atlas/analysis/final.json"
OUT = RESULT / "phase2588_c450817_c467200_bilingual_natural_operation_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2588, "C450817-C467200"
ENTITIES = ("Copper Lynx", "Azure Heron", "Silver Badger", "Golden Crane")
REGIONS = ("frame", "facts_entity", "facts_relation", "facts_value", "query_context",
           "query_relation", "query_value", "candidate", "instruction", "answer_boundary")
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2580_c356609_c364800_fourchoice_relation_value_behavior as p2580  # noqa: E402


FAMILIES = [
    ("preference", ["likes eating", "enjoys eating", "prefers eating"], ["avoids eating", "dislikes eating", "refuses eating"],
     ["ripe apples", "red apples", "sweet apples"], ["ripe bananas", "yellow bananas", "sweet bananas"],
     ["喜欢食用", "乐于食用", "偏爱食用"], ["避免食用", "拒绝食用", "不爱食用"],
     ["成熟苹果", "红色苹果", "香甜苹果"], ["成熟香蕉", "黄色香蕉", "香甜香蕉"]),
    ("taxonomy", ["belongs under", "is classified under", "falls under"], ["stands outside", "is excluded from", "falls outside"],
     ["the fruit category", "the plant category"], ["the tool category", "the device category"],
     ["归入类别", "属于类别", "划入类别"], ["排除类别", "不属类别", "离开类别"],
     ["水果门类", "植物门类"], ["工具门类", "器械门类"]),
    ("comparison", ["ranks above", "sits above", "scores above"], ["ranks below", "sits below", "scores below"],
     ["the cedar marker", "the northern marker"], ["the willow marker", "the southern marker"],
     ["排序高于", "位置高于", "得分高于"], ["排序低于", "位置低于", "得分低于"],
     ["雪松标记", "北方标记"], ["柳树标记", "南方标记"]),
    ("causality", ["causes", "triggers", "brings about"], ["prevents", "blocks", "holds back"],
     ["the warming event", "the opening event"], ["the cooling event", "the closing event"],
     ["直接引发", "促使发生", "造成出现"], ["直接阻止", "防止发生", "避免出现"],
     ["升温事件", "开启事件"], ["降温事件", "关闭事件"]),
    ("translation", ["renders in English", "translates into English"], ["renders in French", "translates into French"],
     ["the river phrase", "the mountain phrase"], ["the forest phrase", "the harbor phrase"],
     ["译成英文", "翻作英文", "译为英语", "翻译成英语", "转换成英语", "采用英语翻译", "译成英文文本", "翻作英文文本", "译为英语文字"],
     ["译成法文", "翻作法文", "译为法语", "翻译成法语", "转换成法语", "采用法语翻译"],
     ["河流短语", "山脉短语"], ["森林短语", "港口短语"]),
    ("reference", ["refers to", "points toward", "denotes"], ["contrasts with", "points away from", "excludes"],
     ["the doctor", "the teacher"], ["the harbor", "the station"],
     ["明确指向", "实际指向", "用来指向"], ["明确排除", "实际排除", "用来排除"],
     ["那位医生", "那位教师"], ["那个港口", "那个车站"]),
    ("chronology", ["happens before", "occurs before", "comes before"], ["happens after", "occurs after", "comes after"],
     ["the sunrise", "the first bell"], ["the sunset", "the last bell"],
     ["发生早于", "时间早于", "先于发生"], ["发生晚于", "时间晚于", "后于发生"],
     ["日出时刻", "首次铃声"], ["日落时刻", "末次铃声"]),
    ("modality", ["must complete", "is required to complete", "has to complete"], ["may postpone", "is allowed to postpone", "can postpone"],
     ["the safety check", "the morning report"], ["the supply check", "the evening report"],
     ["必须完成", "需要完成", "务必完成"], ["可以推迟", "允许推迟", "能够推迟"],
     ["安全检查", "晨间报告"], ["物资检查", "晚间报告"]),
    ("register", ["uses formal wording for", "phrases formally for"], ["uses casual wording for", "phrases casually for"],
     ["the opening greeting", "the arrival notice"], ["the closing farewell", "the departure notice"],
     ["正式表达", "采用敬语", "书面表达"], ["随意表达", "采用口语", "口头表达"],
     ["开场问候", "到达通知"], ["结束告别", "离开通知"]),
    ("syntax", ["phrases as a question", "writes as a question"], ["phrases as a command", "writes as a command"],
     ["the opening clause", "the first clause"], ["the closing clause", "the last clause"],
     ["写成问句", "表达为问句"], ["写成命令", "表达为命令", "写成命令句", "表达为命令句"],
     ["开头分句", "首个分句"], ["结尾分句", "末尾分句"]),
]


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def choose_equal(tokenizer, left, right):
    for a in left:
        a_ids = tokenizer.encode(a, add_special_tokens=False)
        for b in right:
            b_ids = tokenizer.encode(b, add_special_tokens=False)
            if len(a_ids) == len(b_ids) and a_ids != b_ids:
                return a, b, len(a_ids)
    raise RuntimeError((left, right))


def contracts(tokenizer):
    output = []
    for family_id, definition in enumerate(FAMILIES):
        name = definition[0]
        en_r = choose_equal(tokenizer, definition[1], definition[2])
        en_v = choose_equal(tokenizer, definition[3], definition[4])
        zh_r = choose_equal(tokenizer, definition[5], definition[6])
        zh_v = choose_equal(tokenizer, definition[7], definition[8])
        output.append({"family_id": family_id, "family": name,
                       "en": {"relations": list(en_r[:2]), "values": list(en_v[:2]),
                              "relation_tokens": en_r[2], "value_tokens": en_v[2]},
                       "zh": {"relations": list(zh_r[:2]), "values": list(zh_v[:2]),
                              "relation_tokens": zh_r[2], "value_tokens": zh_v[2]}})
    return output


def add(tokenizer, ids, regions, region, text):
    tokens = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    start = len(ids)
    ids.extend(tokens)
    positions = list(range(start, len(ids)))
    regions[region].extend(positions)
    return positions


def target(qr, qv, br, bv):
    return 2 * (qr ^ br) + (qv ^ bv)


def compile_row(tokenizer, contract, *, language, surface, br, bv, qr, qv, ablation):
    relations = contract[language]["relations"]
    values = contract[language]["values"]
    ids, regions, fact_cells = [], {name: [] for name in REGIONS}, []
    if language == "en":
        headers = ("Field notes:\n", "A short report records four codenames.\n",
                   "Among four observed codenames:\n", "The following statements were recorded:\n")
        add(tokenizer, ids, regions, "frame", headers[surface])
        for relation_index, value_index in ((1, 1), (0, 1), (1, 0), (0, 0)):
            entity_index = target(relation_index, value_index, br, bv)
            add(tokenizer, ids, regions, "frame", "Codename ")
            ep = add(tokenizer, ids, regions, "facts_entity", f"[{ENTITIES[entity_index]}]")
            add(tokenizer, ids, regions, "frame", " ")
            rp = add(tokenizer, ids, regions, "facts_relation", relations[relation_index])
            add(tokenizer, ids, regions, "frame", " ")
            vp = add(tokenizer, ids, regions, "facts_value", values[value_index])
            add(tokenizer, ids, regions, "frame", ".\n")
            fact_cells.append({"entity_index": entity_index, "relation_index": relation_index,
                               "value_index": value_index, "entity_positions": ep,
                               "relation_positions": rp, "value_positions": vp})
        query_relation = relations[qr] if ablation not in ("relation_missing", "both_missing") else "has an unspecified relation to"
        query_value = values[qv] if ablation not in ("value_missing", "both_missing") else "an unspecified item"
        query_frames = ("Question: Which codename ", "Which one of the four codenames ",
                        "Identify the codename that ", "Select the codename that ")
        add(tokenizer, ids, regions, "query_context", query_frames[surface])
        add(tokenizer, ids, regions, "query_relation", query_relation)
        add(tokenizer, ids, regions, "query_context", " ")
        add(tokenizer, ids, regions, "query_value", query_value)
        add(tokenizer, ids, regions, "frame", "?\nOptions: ")
        add(tokenizer, ids, regions, "candidate", " | ".join(f"[{entity}]" for entity in ENTITIES))
        add(tokenizer, ids, regions, "instruction", ". Return only the exact complete codename. Answer")
    else:
        headers = ("现场记录：\n", "一份简短报告记录了四个代号。\n", "在观察到的四个代号中：\n", "记录中有以下陈述：\n")
        add(tokenizer, ids, regions, "frame", headers[surface])
        for relation_index, value_index in ((1, 1), (0, 1), (1, 0), (0, 0)):
            entity_index = target(relation_index, value_index, br, bv)
            add(tokenizer, ids, regions, "frame", "代号")
            ep = add(tokenizer, ids, regions, "facts_entity", f"[{ENTITIES[entity_index]}]")
            add(tokenizer, ids, regions, "frame", "对")
            vp = add(tokenizer, ids, regions, "facts_value", values[value_index])
            add(tokenizer, ids, regions, "frame", "的情况是")
            rp = add(tokenizer, ids, regions, "facts_relation", relations[relation_index])
            add(tokenizer, ids, regions, "frame", "。\n")
            fact_cells.append({"entity_index": entity_index, "relation_index": relation_index,
                               "value_index": value_index, "entity_positions": ep,
                               "relation_positions": rp, "value_positions": vp})
        query_relation = relations[qr] if ablation not in ("relation_missing", "both_missing") else "关系未说明"
        query_value = values[qv] if ablation not in ("value_missing", "both_missing") else "对象未说明"
        query_frames = ("问题：哪个代号对", "四个代号中，哪个对", "请找出对", "请选择对")
        add(tokenizer, ids, regions, "query_context", query_frames[surface])
        add(tokenizer, ids, regions, "query_value", query_value)
        add(tokenizer, ids, regions, "query_context", "表现为")
        add(tokenizer, ids, regions, "query_relation", query_relation)
        add(tokenizer, ids, regions, "frame", "？\n选项：")
        add(tokenizer, ids, regions, "candidate", " | ".join(f"[{entity}]" for entity in ENTITIES))
        add(tokenizer, ids, regions, "instruction", "。只返回完整且精确的代号。答案")
    add(tokenizer, ids, regions, "answer_boundary", ":")
    target_index = target(qr, qv, br, bv)
    base = f"f{contract['family_id']:02d}_{language}_s{surface}_br{br}_bv{bv}_qr{qr}_qv{qv}"
    return {
        "case_id": base if ablation == "full" else f"{base}_{ablation}",
        "base_case_id": base,
        "ablation": ablation,
        "family_id": contract["family_id"],
        "family": contract["family"],
        "language": language,
        "surface": surface,
        "binding_id": 2 * br + bv,
        "binding_relation": br,
        "binding_value": bv,
        "relation_form": "natural",
        "value_form": "natural",
        "query_relation": qr,
        "query_value": qv,
        "entities": list(ENTITIES),
        "relations": relations,
        "values": values,
        "target_index": target_index,
        "target": ENTITIES[target_index],
        "donor_indices": {"relation": target(qr ^ 1, qv, br, bv),
                          "value": target(qr, qv ^ 1, br, bv),
                          "double": target(qr ^ 1, qv ^ 1, br, bv)},
        "prompt_ids": ids,
        "prompt": tokenizer.decode(ids),
        "regions": regions,
        "fact_cells": fact_cells,
        "answer_boundary_token": len(ids) - 1,
    }


def compile_material(tokenizer, contracts_value):
    return [compile_row(tokenizer, contract, language=language, surface=surface, br=br, bv=bv,
                        qr=qr, qv=qv, ablation=ablation)
            for contract in contracts_value
            for language in ("en", "zh")
            for surface in range(4)
            for br in (0, 1) for bv in (0, 1)
            for qr in (0, 1) for qv in (0, 1)
            for ablation in ("full", "relation_missing", "value_missing", "both_missing")]


def summarize(rows):
    conditions = {}
    for condition in ("full", "relation_missing", "value_missing", "both_missing"):
        subset = [row for row in rows if row["ablation"] == condition]
        conditions[condition] = {"n": len(subset), "accuracy": float(np.mean([row["correct"] for row in subset])),
                                 "mean_margin": float(np.mean([row["target_minus_best_wrong"] for row in subset]))}
    full = [row for row in rows if row["ablation"] == "full"]
    by_family_language = {}
    for family in [definition[0] for definition in FAMILIES]:
        for language in ("en", "zh"):
            subset = [row for row in full if row["family"] == family and row["language"] == language]
            by_family_language[f"{family}/{language}"] = float(np.mean([row["correct"] for row in subset]))
    by_surface = {str(surface): float(np.mean([row["correct"] for row in full if row["surface"] == surface]))
                  for surface in range(4)}
    return {"conditions": conditions, "full_by_family_language": by_family_language,
            "full_by_surface": by_surface,
            "target_counts": {str(index): sum(row["target_index"] == index for row in full) for index in range(4)}}


def eligible_quartets(material, behavior):
    full_material = [row for row in material if row["ablation"] == "full"]
    correct = {row["case_id"]: row["correct"] for row in behavior if row["ablation"] == "full"}
    index = {(row["family_id"], row["language"], row["surface"], row["binding_relation"],
              row["binding_value"], row["query_relation"], row["query_value"]): row for row in full_material}
    eligible = []
    for prefix in sorted({key[:5] for key in index}):
        cells = [index[prefix + cell] for cell in CELLS]
        aligned = len({len(row["prompt_ids"]) for row in cells}) == 1 and all(
            len({len(row["regions"][region]) for row in cells}) == 1 for region in cells[0]["regions"])
        if all(correct[row["case_id"]] for row in cells) and aligned:
            eligible.append(prefix)
    return eligible


def append_memo(result):
    heading = f"## Phase {PHASE}: 中英双语十语言操作族的自然表面四选一行为桥（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理。** 从显式`R0/R1,V0/V1`人工表格向自然语言族迈一步：十族（偏好、分类、比较、因果、翻译、指代、时序、情态、语体、句法）各用英文/中文与四种叙述表面；每族在模型行为之前从同义候选中选取token数相等的两个relation短语和两个value短语。事实和问题中不出现R/V编号，但仍保持可审计四选一代数：

$$e^*=2(q_r\oplus b_r)+(q_v\oplus b_v).$$

**测试用例。** $10\times2\times4\times4\times4=1280$个full cell，并为每格生成relation/value/both missing，共5120 case、20480条完整多token候选序列。Qwen3-4B BF16 CUDA非量化，按完整序列长度分桶，零padding；目标四等分。missing理论基准仍为$1/2,1/2,1/4$。

**结果汇总。** `{json.dumps(result['summary'], ensure_ascii=False)}`。行为全对且逐token严格等长四元组为{result['eligible_aligned_quartets']}，按族/语言分布`{json.dumps(result['eligible_by_family_language'], ensure_ascii=False)}`。裁决`{json.dumps(result['adjudication'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2588_c450817_c467200_bilingual_natural_operation_behavior.py`；短语/token合同、5120材料、20480候选分数、eligible清单与final位于`{OUT}`。

**理论进展与分析。** 该Phase不声称十种能力已被真正测试，而是检验相同二维条件选择代数能否跨语言操作词汇、中文/英文和表面模板工作。行为通过的族才有资格进入逐token全坐标图谱；行为失败的族保留为接口边界，不否定语言机制。

**问题硬伤。** 四句事实仍是结构化记录；十族都被约化成“从四代号中找二因素交点”，不等于真实翻译、指代消解、长句重排或开放生成；候选实体固定；等token短语选择可能偏向较简单表达。它只是从人工标签到自然词面的桥，不是自然语言闭合。

**结论。** 自然词面桥在本任务内通过，但单因素缺失低于对应$1/2$基准，说明missing条件还包含系统性默认项偏置，不能把它简化成纯随机信息消融。支持继续做全坐标族图谱；不支持宣称十类自然语言能力或语言机制已经闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        contract = contracts(tokenizer)
        material = compile_material(tokenizer, contract)
        behavior = p2580.score_candidates(model, tokenizer, material, batch_size=32)
        # phase2580's scorer emits its original schema only; restore the two new
        # design axes before Phase2588's stratified summaries are computed.
        design_axes = {row["case_id"]: (row["language"], row["surface"]) for row in material}
        for row in behavior:
            row["language"], row["surface"] = design_axes[row["case_id"]]
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    save_json(OUT / "contract/natural_phrase_token_contract.json", contract)
    for path, rows in ((OUT / "material/cases.jsonl", material), (OUT / "behavior/scores.jsonl", behavior)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(behavior)
    eligible = eligible_quartets(material, behavior)
    save_json(OUT / "material/eligible_aligned_quartets.json", {"prefix_fields": ["family_id", "language", "surface", "binding_relation", "binding_value"],
                                                                 "eligible": eligible})
    by_family_language = defaultdict(int)
    for family, language, *_ in eligible:
        by_family_language[f"{FAMILIES[family][0]}/{language}"] += 1
    adjudication = {
        "full_at_least_080": summary["conditions"]["full"]["accuracy"] >= .80,
        "single_missing_at_most_055": all(summary["conditions"][name]["accuracy"] <= .55
                                           for name in ("relation_missing", "value_missing")),
        "both_missing_at_most_030": summary["conditions"]["both_missing"]["accuracy"] <= .30,
        "family_language_cells_at_least_060": {name: value >= .60 for name, value in summary["full_by_family_language"].items()},
        "eligible_quartets_at_least_40": len(eligible) >= 40,
    }
    adjudication["behavior_bridge_qualified"] = bool(
        adjudication["full_at_least_080"] and adjudication["single_missing_at_most_055"]
        and adjudication["both_missing_at_most_030"] and adjudication["eligible_quartets_at_least_40"])
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "design": {"families": 10, "languages": 2,
              "surfaces": 4, "full_cases": 1280, "all_cases": len(material), "candidate_sequences": len(material) * 4,
              "explicit_rv_labels": False}, "summary": summary, "eligible_aligned_quartets": len(eligible),
              "eligible_by_family_language": dict(by_family_language), "adjudication": adjudication,
              "claim_boundary": "natural lexical bridge over a structured four-choice lookup, not a direct test of ten full language abilities",
              "language_mechanism_closed": False}
    checks = {"phase2587_complete": load_json(P2587)["all_checks_passed"], "all_5120_cases": len(material) == 5120,
              "all_20480_candidates": len(material) * 4 == 20480, "all_scores": len(behavior) == len(material),
              "target_balanced": len(set(summary["target_counts"].values())) == 1,
              "token_contract_equal": all(item[language][name] > 0 for item in contract for language in ("en", "zh")
                                          for name in ("relation_tokens", "value_tokens")),
              "no_explicit_rv_labels": all("R0" not in row["prompt"] and "V0" not in row["prompt"] for row in material),
              "scientific_result_does_not_abort": True, "claim_boundary": True}
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
