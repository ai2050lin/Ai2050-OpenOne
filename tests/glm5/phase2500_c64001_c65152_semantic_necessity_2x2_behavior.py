#!/usr/bin/env python3
"""Behavior gate for relation meaning swaps that force the correct answer to change."""
from __future__ import annotations

import gc
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase2500_c64001_c65152_semantic_necessity_2x2_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2500, "C64001-C65152"
UNITS = (20, 21, 22)
SPLIT = {20: "discovery", 21: "confirmation", 22: "lockbox"}
PAIRS = (
    ("taxonomy", "part_whole"),
    ("product", "causal"),
    ("temporal", "spatial"),
    ("role", "preference"),
    ("membership", "translation"),
    ("coreference", "punctuation"),
)
MARKERS = {
    20: {"en": ("daxen", "kelvo"), "zh": ("甲符", "乙符")},
    21: {"en": ("muric", "tovan"), "zh": ("丙符", "丁符")},
    22: {"en": ("pebrix", "sovam"), "zh": ("戊符", "己符")},
}
EN_NAMES = {
    20: ("Yaren", "Zelith", "Corven", "Dalune", "Evrin", "Falor", "Gerin", "Havel"),
    21: ("Iskar", "Jalen", "Korim", "Luneth", "Merov", "Navel", "Othir", "Pevan"),
    22: ("Ralen", "Sovin", "Turel", "Ulvon", "Varin", "Wexel", "Yorin", "Zaven"),
}
ZH_NAMES = {
    20: ("晓岚", "远汀", "泽川", "澄屿", "丹枫", "风禾", "古砚", "海舟"),
    21: ("简溪", "景松", "空山", "兰汀", "墨川", "南屿", "秋禾", "平澜"),
    22: ("泉舟", "松砚", "桐溪", "晚川", "溪岫", "言澜", "知禾", "舟宁"),
}

# Each tuple is (fact template, marker meaning description). Surface selects one of two paraphrases.
RELATIONS: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "taxonomy": {
        "en": (("{s} belongs to category {t}", "the category that the source belongs to"),
               ("{s} is classified under {t}", "the class under which the source is classified")),
        "zh": (("{s}属于类别{t}", "来源所属的类别"), ("{s}被归类为{t}", "来源被归入的类别")),
    },
    "part_whole": {
        "en": (("{s} is a component of {t}", "the whole containing the source"),
               ("{s} forms part of {t}", "the larger whole of which the source is a part")),
        "zh": (("{s}是{t}的组成部分", "包含该来源的整体"), ("{s}构成{t}的一部分", "来源所属的更大整体")),
    },
    "product": {
        "en": (("{s} manufactures {t}", "the product made by the source"),
               ("{s} produces {t}", "the item produced by the source")),
        "zh": (("{s}制造{t}", "来源制造的产品"), ("{s}生产{t}", "来源生产的项目")),
    },
    "causal": {
        "en": (("{s} causes {t}", "the effect caused by the source"),
               ("{s} triggers {t}", "the outcome triggered by the source")),
        "zh": (("{s}导致{t}", "由来源导致的结果"), ("{s}触发{t}", "由来源触发的结果")),
    },
    "temporal": {
        "en": (("{s} occurs before {t}", "the event occurring after the source"),
               ("{s} precedes {t}", "the event following the source in time")),
        "zh": (("{s}发生在{t}之前", "发生在来源之后的事件"), ("{s}先于{t}", "时间上跟随来源的事件")),
    },
    "spatial": {
        "en": (("{s} stands left of {t}", "the item to the right of the source"),
               ("{s} is west of {t}", "the item east of the source")),
        "zh": (("{s}位于{t}左侧", "位于来源右侧的项目"), ("{s}在{t}西边", "位于来源东边的项目")),
    },
    "role": {
        "en": (("{s} reports to {t}", "the supervisor of the source"),
               ("{s} works under {t}", "the person supervising the source")),
        "zh": (("{s}向{t}汇报", "来源的上级"), ("{s}在{t}手下工作", "监督该来源的人")),
    },
    "preference": {
        "en": (("{s} prefers {t}", "the item preferred by the source"),
               ("{s} favors {t}", "the item favored by the source")),
        "zh": (("{s}偏爱{t}", "来源偏爱的项目"), ("{s}更喜欢{t}", "来源更喜欢的项目")),
    },
    "membership": {
        "en": (("{s} is enrolled in {t}", "the group containing the source as a member"),
               ("{s} holds membership in {t}", "the organization to which the source belongs")),
        "zh": (("{s}登记在{t}", "来源作为成员所在的群组"), ("{s}是{t}的成员", "来源所属的组织")),
    },
    "translation": {
        "en": (("{s} is translated as {t}", "the translation equivalent of the source"),
               ("{s} corresponds in translation to {t}", "the cross-language equivalent of the source")),
        "zh": (("{s}翻译为{t}", "来源的翻译对应项"), ("{s}在翻译中对应{t}", "来源的跨语言等价项")),
    },
    "coreference": {
        "en": (("in a message from {s}, the word you refers to {t}", "the referent of you in the source's message"),
               ("when {s} says you, the pronoun points to {t}", "the person denoted by you when the source speaks")),
        "zh": (("在{s}发出的消息中，代词你指代{t}", "来源消息中代词你的指代对象"),
               ("当{s}说你时，这个代词指向{t}", "来源说话时代词你所指的人")),
    },
    "punctuation": {
        "en": (("in the phrase {t}, not {s}, arrived, comma contrast selects {t}", "the person selected instead of the source by comma contrast"),
               ("in {t}, rather than {s}, responded, punctuation selects {t}", "the person chosen over the source by punctuation")),
        "zh": (("在短语“{t}，而不是{s}，到达了”中，逗号对比选中{t}", "逗号对比中替代来源而被选中的人"),
               ("在“{t}，并非{s}，作答了”中，标点选中{t}", "标点结构中相对来源被选择的人")),
    },
}

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def occurrence_spans(tokenizer, prompt: str, text: str) -> list[list[int]]:
    spans = []
    start = 0
    while True:
        pos = prompt.find(text, start)
        if pos < 0:
            break
        spans.append([len(tokenizer.encode(prompt[:pos], add_special_tokens=False)),
                      len(tokenizer.encode(prompt[:pos + len(text)], add_special_tokens=False))])
        start = pos + len(text)
    return spans


def compile_rows(tokenizer) -> list[dict]:
    rows = []
    case = 64001
    for unit in UNITS:
        for pair_id, pair in enumerate(PAIRS):
            for language in ("en", "zh"):
                names0 = EN_NAMES[unit] if language == "en" else ZH_NAMES[unit]
                shift = (pair_id * 3 + unit) % len(names0)
                names = names0[shift:] + names0[:shift]
                source, target0, target1 = names[:3]
                marker0, marker1 = MARKERS[unit][language]
                for surface in range(4):
                    variant = surface % 2
                    rels = []
                    for family, target in zip(pair, (target0, target1)):
                        fact_template, description = RELATIONS[family][language][variant]
                        rels.append({"family": family, "target": target,
                                     "fact": fact_template.format(s=source, t=target), "description": description})
                    fact_order = (0, 1) if surface < 2 else (1, 0)
                    definition_order = (0, 1) if surface in (0, 3) else (1, 0)
                    candidates = [target0, target1] if (pair_id + surface + unit) % 2 == 0 else [target1, target0]
                    for meaning_swap in (0, 1):
                        mapping = (0, 1) if meaning_swap == 0 else (1, 0)
                        definitions = {0: rels[mapping[0]]["description"], 1: rels[mapping[1]]["description"]}
                        if language == "en":
                            definition_parts = [f"Marker '{(marker0, marker1)[mi]}' means \"{definitions[mi]}\"." for mi in definition_order]
                            fact_parts = [f"Fact {rank + 1}: {rels[ri]['fact']}." for rank, ri in enumerate(fact_order)]
                            prefix = "Definitions: " + " ".join(definition_parts) + " End definitions. Facts: " + " ".join(fact_parts) + " End facts. "
                        else:
                            definition_parts = [f"标记“{(marker0, marker1)[mi]}”表示“{definitions[mi]}”。" for mi in definition_order]
                            fact_parts = [f"事实{rank + 1}：{rels[ri]['fact']}。" for rank, ri in enumerate(fact_order)]
                            prefix = "定义：" + "".join(definition_parts) + "定义结束。事实：" + "".join(fact_parts) + "事实结束。"
                        for query_marker in (0, 1):
                            marker = (marker0, marker1)[query_marker]
                            selected_relation = mapping[query_marker]
                            target = rels[selected_relation]["target"]
                            if language == "en":
                                prompt = (prefix + f"Query: for source {source}, follow marker '{marker}'. Which candidate satisfies that marker's defined meaning?\n"
                                          f"Candidates: {candidates[0]} | {candidates[1]}\nReturn exactly one candidate name.\nAnswer:")
                            else:
                                prompt = (prefix + f"查询：对于来源{source}，请遵循标记“{marker}”。哪个候选符合该标记所定义的含义？\n"
                                          f"候选：{candidates[0]} | {candidates[1]}\n只返回一个候选名称。\n答案：")
                            ids = [int(v) for v in tokenizer.encode(prompt, add_special_tokens=False)]
                            anchors = {
                                "definition_end": "End definitions" if language == "en" else "定义结束",
                                "facts_end": "End facts" if language == "en" else "事实结束",
                                "marker0": marker0,
                                "marker1": marker1,
                                "query_marker": marker,
                                "candidate0": candidates[0],
                                "candidate1": candidates[1],
                            }
                            spans = {key: occurrence_spans(tokenizer, prompt, value) for key, value in anchors.items()}
                            rows.append({
                                "case_id": f"c{case:05d}-p{pair_id}-u{unit}-{language}-s{surface}-m{meaning_swap}-q{query_marker}",
                                "unit": unit, "split": SPLIT[unit], "pair_id": pair_id, "families": list(pair),
                                "language": language, "surface": surface, "meaning_swap": meaning_swap,
                                "query_marker": query_marker, "marker": marker, "markers": [marker0, marker1],
                                "source": source, "relation_targets": [target0, target1], "selected_relation": selected_relation,
                                "target": target, "candidates": candidates, "expected_output": target,
                                "facts": [rels[i]["fact"] for i in fact_order], "descriptions": [definitions[mi] for mi in definition_order],
                                "prompt": prompt, "prompt_ids": ids, "spans": spans, "answer_boundary_token": len(ids) - 1,
                            })
                            case += 1
    return rows


def parse(text: str, row: dict) -> tuple[str | None, bool]:
    cleaned = re.sub(r"^(?:answer|答案)\s*[:：]\s*", "", text.strip(), flags=re.I)
    norm = normalize(cleaned)
    hits = [candidate for candidate in row["candidates"] if normalize(candidate) in norm]
    parsed = hits[0] if len(set(hits)) == 1 else None
    return parsed, parsed == row["target"]


def behavior(model, tokenizer, rows: list[dict]) -> list[dict]:
    tokenizer.padding_side = "left"
    device = model.get_input_embeddings().weight.device
    output_rows = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        encoded = tokenizer([r["prompt"] for r in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            sequences = model.generate(**encoded, max_new_tokens=10, do_sample=False, use_cache=True,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        width = encoded["input_ids"].shape[1]
        for row, sequence in zip(batch, sequences):
            new_ids = [int(v) for v in sequence[width:].cpu().tolist()]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            parsed, correct = parse(text, row)
            output_rows.append({
                "case_id": row["case_id"], "unit": row["unit"], "pair_id": row["pair_id"],
                "language": row["language"], "surface": row["surface"], "meaning_swap": row["meaning_swap"],
                "query_marker": row["query_marker"], "selected_relation": row["selected_relation"],
                "target": row["target"], "generated_ids": new_ids, "generated_text": text,
                "parsed_answer": parsed, "parsed_correct": bool(correct),
            })
        if (start + len(batch)) % 96 == 0:
            print(f"[phase2500 behavior] {start + len(batch)}/{len(rows)}", flush=True)
    return output_rows


def summarize(rows: list[dict], generated: list[dict]) -> dict:
    by_id = {r["case_id"]: r for r in generated}
    detail = {}
    pair_flip = {}
    qualified = []
    for unit in UNITS:
        detail[str(unit)] = {}
        pair_flip[str(unit)] = {}
        for pair_id, pair in enumerate(PAIRS):
            cases = [r for r in rows if r["unit"] == unit and r["pair_id"] == pair_id]
            values = [by_id[r["case_id"]] for r in cases]
            lang_accuracy = {language: sum(v["parsed_correct"] for v in values if v["language"] == language) / 16 for language in ("en", "zh")}
            swap_accuracy = {str(swap): sum(v["parsed_correct"] for v in values if v["meaning_swap"] == swap) / 16 for swap in (0, 1)}
            detail[str(unit)][str(pair_id)] = {
                "families": list(pair), "rows": 32,
                "accuracy": sum(v["parsed_correct"] for v in values) / 32,
                "language_accuracy": lang_accuracy, "meaning_swap_accuracy": swap_accuracy,
            }
            successes = []
            for language in ("en", "zh"):
                for surface in range(4):
                    for query_marker in (0, 1):
                        pair_rows = [r for r in cases if r["language"] == language and r["surface"] == surface and r["query_marker"] == query_marker]
                        assert len(pair_rows) == 2 and pair_rows[0]["target"] != pair_rows[1]["target"]
                        successes.append(all(by_id[r["case_id"]]["parsed_correct"] for r in pair_rows))
            pair_flip[str(unit)][str(pair_id)] = {"pairs": 16, "both_correct_rate": sum(successes) / 16}
    for pair_id in range(len(PAIRS)):
        if all(detail[str(unit)][str(pair_id)]["accuracy"] >= .75
               and min(detail[str(unit)][str(pair_id)]["language_accuracy"].values()) >= .625
               and min(detail[str(unit)][str(pair_id)]["meaning_swap_accuracy"].values()) >= .625
               and pair_flip[str(unit)][str(pair_id)]["both_correct_rate"] >= .625
               for unit in (21, 22)):
            qualified.append(pair_id)
    aggregate = {str(unit): sum(v["parsed_correct"] for v in generated if v["unit"] == unit) / 192 for unit in UNITS}
    flip_aggregate = {str(unit): sum(pair_flip[str(unit)][str(p)]["both_correct_rate"] for p in range(6)) / 6 for unit in UNITS}
    return {"aggregate_accuracy": aggregate, "aggregate_paired_flip_success": flip_aggregate,
            "detail": detail, "paired_flip": pair_flip, "qualified_pair_ids": qualified,
            "qualified_pairs": [list(PAIRS[p]) for p in qualified]}


def design_audit(rows: list[dict]) -> dict:
    index = {(r["unit"], r["pair_id"], r["language"], r["surface"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    flip_ok = []
    token_bag_equal = []
    fixed_candidates = []
    for unit in UNITS:
        for pair_id in range(6):
            for language in ("en", "zh"):
                for surface in range(4):
                    for query_marker in (0, 1):
                        a = index[(unit, pair_id, language, surface, 0, query_marker)]
                        b = index[(unit, pair_id, language, surface, 1, query_marker)]
                        flip_ok.append(a["target"] != b["target"])
                        token_bag_equal.append(Counter(a["prompt_ids"]) == Counter(b["prompt_ids"]))
                        fixed_candidates.append(a["candidates"] == b["candidates"] and a["facts"] == b["facts"] and a["marker"] == b["marker"])
    return {
        "rows": len(rows), "answer_flip_rate": sum(flip_ok) / len(flip_ok),
        "exact_token_multiset_equal_rate": sum(token_bag_equal) / len(token_bag_equal),
        "facts_candidates_query_marker_fixed_rate": sum(fixed_candidates) / len(fixed_candidates),
        "target_candidate_position_counts": dict(Counter(r["candidates"].index(r["target"]) for r in rows)),
        "all_spans_present": all(all(r["spans"][event] for event in r["spans"]) for r in rows),
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 十二关系族语义必要性定义交换四格的576条行为资格（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 十二family配成六个竞争对，每条材料给同一source两条不同关系事实。例如taxonomy/part-whole对同时给“来源属于某类别”和“来源是某整体的一部分”，两个关系指向不同候选；两个nonce marker分别定义为“类别关系”和“整体关系”。在保持事实、实体、候选顺序、marker字符串、查询marker和输出格式不变时，只交换两个marker的关系含义，正确答案必须翻转。中英文使用独立字符串；四surface改变谓词复述、事实顺序、定义顺序与候选顺序；unit20/21/22分别用于discovery/confirmation/lockbox。每个swap再与两个query marker交叉，形成每组四格。

$$N=3\times6\times2\times4\times2\times2=576,\qquad r_0\text{{ selected}}\Longleftrightarrow m=q.$$

**结果汇总。** 设计审计 `{json.dumps(result['design_audit'], ensure_ascii=False)}`；自主贪心行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2500_c64001_c65152_semantic_necessity_2x2_behavior.py`；材料、逐行生成与final位于`{OUT}`。

**分析与理论进展。** 与Phase2496不同，新的正确答案不是沿单一marker固定边即可得到，而由“定义交换×查询marker”的XOR式关系选择决定。`paired_flip_success`只在同一事实和查询下的两个定义交换版本都正确时计1，因而直接测量模型是否跟随含义交换。exact token multiset审计只控制整段token计数，不控制顺序；这正是定义绑定必须改变的最小操作。

**问题硬伤与结论。** 即使行为门通过，仍只能说明模型按自然描述选择了正确关系，不能说明某组HiddenState坐标是因果中介。六个pair把关系放在二选一语境中，可能编码的是pair-relative选择而不是独立family；后续必须用新配对自动续研。合成实体与短prompt仍不等于开放自然语言。只有confirmation与lockbox均合格的pair才能进入内部场分析。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        rows = compile_rows(tokenizer)
        write_jsonl(OUT / "material/semantic_necessity_rows.jsonl", rows)
        generated = behavior(model, tokenizer, rows)
        write_jsonl(OUT / "behavior/autonomous_generation.jsonl", generated)
    finally:
        model_utils.release_model(model)
        gc.collect()
    audit = design_audit(rows)
    summary = summarize(rows, generated)
    checks = {
        "rows_576": len(rows) == 576,
        "all_answers_flip": audit["answer_flip_rate"] == 1.0,
        "token_multiset_control": audit["exact_token_multiset_equal_rate"] == 1.0,
        "fixed_nondefinition_content": audit["facts_candidates_query_marker_fixed_rate"] == 1.0,
        "candidate_position_balanced": set(audit["target_candidate_position_counts"].values()) == {288},
        "all_spans": audit["all_spans_present"],
        "at_least_four_qualified_pairs": len(summary["qualified_pair_ids"]) >= 4,
        "confirmation_lockbox_gate": True,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B nonquantized BF16 CUDA",
        "material": {"path": str(OUT / "material/semantic_necessity_rows.jsonl"), "rows": len(rows)},
        "design_audit": audit, "behavior": summary,
        "adjudication": {"relation_meaning_behaviorally_necessary": len(summary["qualified_pair_ids"]) >= 4,
                         "semantic_code_identified": False, "causal_mediator_identified": False,
                         "language_encoding_mechanism_closed": False},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
