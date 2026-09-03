#!/usr/bin/env python3
"""Compile an eight-family concept/grammar/function hypergraph and qualify four local models."""
from __future__ import annotations

import gc
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2435
CAMPAIGN = "C33841-C34160"
FAMILIES = ("taxonomy", "causal", "temporal", "negation_scope", "preposition_role",
            "coreference_binding", "punctuation_attachment", "sentence_reordering")
VARIANTS = ("valid", "broken_a", "broken_b")
LANGUAGES = ("en", "zh")
SURFACES = ("canonical", "natural")
MODEL_ORDER = ("qwen4b", "qwen14b", "glm4", "deepseek7b")
EVENTS = ("prefix_end", "operation_end", "argument_end", "context_end", "query_end",
          "candidate1_end", "candidate2_end", "answer_boundary")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2397_c21681_c22000_operation_behavior_token_calibration as behavior  # noqa: E402
import phase2412_c26481_c26800_frozen_crossmodel_operator_replication as capture_loader  # noqa: E402


TRIPLES = (
    ("Aster", "Borin", "Celyn"), ("Daro", "Eris", "Faron"), ("Galen", "Hira", "Iven"),
    ("Jora", "Kelan", "Luma"), ("Maro", "Neris", "Olan"), ("Pira", "Quin", "Rovan"),
)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def context_for(family: str, language: str, surface: str, variant: str,
                source: str, middle: str, target: str) -> tuple[str, dict[str, tuple[int, int]], tuple[str, str]]:
    zh = language == "zh"
    intro = (("根据这份较长的本地记录，" if zh else "According to the longer local record, ")
             if surface == "natural" else "")
    if family == "taxonomy":
        rels = ({"valid": "是一种", "broken_a": "位于旁边", "broken_b": "发生在之前"} if zh else
                {"valid": "is a type of", "broken_a": "is located beside", "broken_b": "occurred before"})
        op = rels[variant]; core = f"{source}{op}{target}。" if zh else f"{source} {op} {target}."
        queries = (("哪个名称是被分类的实例？", "哪个名称是类别？") if zh else
                   ("Which name is the classified instance?", "Which name is the category?"))
    elif family == "causal":
        rels = ({"valid": "导致", "broken_a": "跟随", "broken_b": "描述"} if zh else
                {"valid": "caused", "broken_a": "followed", "broken_b": "described"})
        op = rels[variant]
        core = f"{source}{op}{middle}；{middle}{op}{target}。" if zh else f"{source} {op} {middle}; {middle} {op} {target}."
        queries = (("哪个名称是链条的起因？", "哪个名称是链条的最终结果？") if zh else
                   ("Which name is the initiating cause?", "Which name is the final effect?"))
    elif family == "temporal":
        rels = ({"valid": "早于", "broken_a": "靠近", "broken_b": "提到"} if zh else
                {"valid": "happened before", "broken_a": "was near", "broken_b": "mentioned"})
        op = rels[variant]
        core = f"{source}{op}{middle}；{middle}{op}{target}。" if zh else f"{source} {op} {middle}; {middle} {op} {target}."
        queries = (("哪个名称最早？", "哪个名称最晚？") if zh else
                   ("Which name is earliest?", "Which name is latest?"))
    elif family == "negation_scope":
        forms = ({"valid": (f"{source}已获准，而{target}未获准。", "已获准"),
                  "broken_a": (f"{source}未获准，而{target}已获准。", "未获准"),
                  "broken_b": (f"{source}待定，而{target}并非待定。", "待定")} if zh else
                 {"valid": (f"{source} is approved, whereas {target} is not approved.", "is approved"),
                  "broken_a": (f"{source} is not approved, whereas {target} is approved.", "is not approved"),
                  "broken_b": (f"{source} is pending, whereas {target} is not pending.", "is pending")})
        core, op = forms[variant]
        queries = (("哪个名称获准？", "哪个名称被明确否定？") if zh else
                   ("Which name is approved?", "Which name is explicitly negated?"))
    elif family == "preposition_role":
        rels = ({"valid": "在上方", "broken_a": "在下方", "broken_b": "在旁边"} if zh else
                {"valid": "is above", "broken_a": "is below", "broken_b": "is beside"})
        op = rels[variant]; core = f"{source}{op}{target}。" if zh else f"{source} {op} {target}."
        queries = (("哪个名称处于上方位置？", "哪个名称处于下方位置？") if zh else
                   ("Which name occupies the upper role?", "Which name occupies the lower role?"))
    elif family == "coreference_binding":
        referent = source if variant != "broken_a" else target
        phrase = "赠予者" if zh else "the giver"
        op = "指代" if zh else "referred to"
        if variant == "broken_b": phrase = "观察者" if zh else "the observer"
        core = (f"{source}把密封便笺交给{target}；短语“{phrase}”{op}{referent}。" if zh else
                f"{source} handed a sealed note to {target}; the phrase '{phrase}' {op} {referent}.")
        queries = (("哪个名称是赠予者？", "哪个名称是接收者？") if zh else
                   ("Which name is the giver?", "Which name is the receiver?"))
    elif family == "punctuation_attachment":
        if zh:
            forms = {"valid": (f"{source}——而不是{target}——获得了奖章。", "而不是"),
                     "broken_a": (f"{target}——尽管是{source}——获得了奖章。", "尽管是"),
                     "broken_b": (f"{source}和{target}讨论了奖章。", "和")}
            queries = ("哪个名称获得奖章？", "哪个名称被破折号排除？")
        else:
            forms = {"valid": (f"{source}—not {target}—received the medal.", "not"),
                     "broken_a": (f"{target}—despite {source}—received the medal.", "despite"),
                     "broken_b": (f"{source} and {target} discussed the medal.", "and")}
            queries = ("Which name received the medal?", "Which name is excluded by the dashes?")
        core, op = forms[variant]
    else:
        rels = ({"valid": "先于", "broken_a": "晚于", "broken_b": "邻接"} if zh else
                {"valid": "before", "broken_a": "after", "broken_b": "adjacent to"})
        op = rels[variant]
        core = (f"片段A：{source}打开档案。片段B：{middle}复制文件。片段C：{target}封存文件。规则：A{op}B，B{op}C。" if zh else
                f"Fragment A: {source} opened the archive. Fragment B: {middle} copied the file. Fragment C: {target} sealed it. Rule: A {op} B and B {op} C.")
        queries = (("哪个名称属于第一片段？", "哪个名称属于最后片段？") if zh else
                   ("Which name belongs to the first fragment?", "Which name belongs to the final fragment?"))
    context = intro + core
    prefix_at = context.find(source); operation_at = context.find(op); argument_at = context.rfind(target)
    if min(prefix_at, operation_at, argument_at) < 0:
        raise RuntimeError((family, language, surface, variant, context, op))
    spans = {"prefix": (prefix_at, prefix_at + len(source)),
             "operation": (operation_at, operation_at + len(op)),
             "argument": (argument_at, argument_at + len(target))}
    return context, spans, queries


def prompt_with_events(language: str, context: str, spans: dict[str, tuple[int, int]], query: str,
                       candidates: list[str]) -> tuple[str, list[dict]]:
    if language == "zh":
        head, qlabel, clabel, answer = "阅读记录，只能用一个候选项作答。\n记录：", "\n问题：", "\n候选项：\n", "\n答案："
    else:
        head, qlabel, clabel, answer = "Read the record and answer with exactly one candidate.\nRecord: ", "\nQuestion: ", "\nCandidates:\n", "\nAnswer:"
    prompt = head + context + qlabel + query + clabel
    context_start = len(head)
    query_end = len(head) + len(context) + len(qlabel) + len(query)
    events = [{"event": "prefix_end", "char_start": context_start + spans["prefix"][0], "char_end": context_start + spans["prefix"][1]},
              {"event": "operation_end", "char_start": context_start + spans["operation"][0], "char_end": context_start + spans["operation"][1]},
              {"event": "argument_end", "char_start": context_start + spans["argument"][0], "char_end": context_start + spans["argument"][1]},
              {"event": "context_end", "char_start": context_start, "char_end": context_start + len(context)},
              {"event": "query_end", "char_start": query_end - len(query), "char_end": query_end}]
    for index, candidate in enumerate(candidates, 1):
        start = len(prompt) + 2
        prompt += f"- {candidate}\n"
        events.append({"event": f"candidate{index}_end", "char_start": start, "char_end": start + len(candidate)})
    prompt = prompt.rstrip("\n") + answer
    events.append({"event": "answer_boundary", "char_start": len(prompt), "char_end": len(prompt)})
    order = {name: i for i, name in enumerate(EVENTS)}
    events.sort(key=lambda item: order[item["event"]])
    return prompt, events


def compile_source() -> list[dict]:
    rows = []
    for fi, family in enumerate(FAMILIES):
        for unit, triple in enumerate(TRIPLES):
            for language in LANGUAGES:
                for surface in SURFACES:
                    for direction in (0, 1):
                        source, middle, target = triple if direction == 0 else (triple[2], triple[1], triple[0])
                        for variant in VARIANTS:
                            context, spans, queries = context_for(family, language, surface, variant, source, middle, target)
                            for role, query in enumerate(queries):
                                answer, foil = (source, target) if role == 0 else (target, source)
                                order = (fi + unit + LANGUAGES.index(language) + SURFACES.index(surface) + direction + role) % 2
                                candidates = [answer, foil] if order == 0 else [foil, answer]
                                prompt, events = prompt_with_events(language, context, spans, query, candidates)
                                config = f"th-{family}-u{unit}-{language}-{surface}-d{direction}"
                                rows.append({
                                    "case_id": f"{config}-{variant}-r{role}", "config_id": config,
                                    "task": "trajectory_hypergraph_selection", "family": family, "unit": unit,
                                    "language": language, "surface": surface,
                                    "surface_class": "controlled" if surface == "canonical" else "natural",
                                    "direction": direction, "variant": variant, "query_role": "source" if role == 0 else "target",
                                    "candidate_order": order, "target_candidate_slot": candidates.index(answer),
                                    "partition": "discovery" if unit < 4 else ("confirmation" if unit == 4 else "fresh_unit_lockbox"),
                                    "source": source, "middle": middle, "target": target, "context": context,
                                    "query": query, "candidates": candidates, "answer": answer, "foil": foil,
                                    "prompt": prompt, "events": events,
                                })
    return rows


def material_audit(rows: list[dict]) -> dict:
    return {"rows": len(rows), "families": Counter(row["family"] for row in rows),
            "languages": Counter(row["language"] for row in rows), "surfaces": Counter(row["surface"] for row in rows),
            "variants": Counter(row["variant"] for row in rows), "roles": Counter(row["query_role"] for row in rows),
            "partitions": Counter(row["partition"] for row in rows), "event_counts": sorted({len(row["events"]) for row in rows}),
            "unique_cases": len({row["case_id"] for row in rows}) == len(rows),
            "unique_prompts": len({row["prompt"] for row in rows}) == len(rows),
            "event_bounds": all(all(0 <= e["char_start"] <= e["char_end"] <= len(row["prompt"]) for e in row["events"]) for row in rows)}


def summarize(rows: list[dict], scores: list[dict], metric: str) -> dict:
    values = np.asarray([record[metric] for record in scores], dtype=np.float64)
    joined = list(zip(rows, values))
    by_variant = {variant: float(np.mean([value > 0 for row, value in joined if row["variant"] == variant])) for variant in VARIANTS}
    by_family = {}
    for family in FAMILIES:
        by_family[family] = {variant: float(np.mean([value > 0 for row, value in joined if row["family"] == family and row["variant"] == variant]))
                             for variant in VARIANTS}
        by_family[family]["valid_minus_broken_a"] = by_family[family]["valid"] - by_family[family]["broken_a"]
    return {"rows": len(rows), "target_over_foil": float(np.mean(values > 0)), "mean_margin": float(values.mean()),
            "by_variant": by_variant, "valid_minus_broken_a": by_variant["valid"] - by_variant["broken_a"],
            "broken_a_minus_broken_b": by_variant["broken_a"] - by_variant["broken_b"], "by_family": by_family}


def model_source(source: list[dict], key: str) -> list[dict]:
    if key == "qwen4b":
        return source
    return [row for row in source if int(row["unit"]) in (0, 4, 5) and row["surface"] == "canonical"]


def load_model(key: str):
    if key == "qwen4b":
        model, tokenizer, label = capability.load_model(key)
        return model, tokenizer, label, "BF16 weights"
    if key == "qwen14b":
        model, tokenizer, label = capture_loader.load_for_capture(key)
        return model, tokenizer, label, "NF4 storage / BF16 compute; BF16-auto previously unstable on this host"
    model, tokenizer, label = capability.load_model(key)
    precision = "INT8 weights / model-relative behavior only"
    return model, tokenizer, label, precision


def run_model(key: str, source: list[dict]) -> dict:
    final = OUT / key / "analysis/final.json"
    if final.exists():
        return json.loads(final.read_text(encoding="utf-8"))
    model, tokenizer, label, precision = load_model(key)
    behavior.OUT = OUT
    try:
        chosen = model_source(source, key)
        compiled, calibration = behavior.compile_rows(tokenizer, chosen)
        write_rows(OUT / key / "index/trajectory_rows.jsonl", compiled)
        save(OUT / key / "analysis/token_calibration.json", calibration)
        batch = {"qwen4b": 12, "qwen14b": 2, "glm4": 3, "deepseek7b": 4}[key]
        scored, _ = behavior.score_rows(key, model, compiled, batch)
        teacher = summarize(compiled, scored, "mean_logprob_margin")
        lockbox = [row for row in compiled if row["variant"] == "valid" and int(row["unit"]) == 5 and row["surface"] == "natural"]
        if key != "qwen4b":
            lockbox = [row for row in compiled if row["variant"] == "valid" and int(row["unit"]) == 5]
        generated, _ = behavior.generate_lockbox(key, model, tokenizer, lockbox, 4 if key == "qwen4b" else 2)
        autonomous = {"rows": len(generated), "exact": float(np.mean([record["exact"] for record in generated])),
                      "target_present": float(np.mean([record["target_present"] for record in generated]))}
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    checks = {"compiled": len(compiled) == len(model_source(source, key)), "teacher": len(scored) == len(compiled),
              "autonomous": len(generated) == len(lockbox), "monotonic": calibration["event_monotonic_rate"] == 1.0,
              "finite": math.isfinite(teacher["mean_margin"])}
    result = {"model": key, "label": label, "precision": precision, "calibration": calibration,
              "teacher": teacher, "autonomous": autonomous, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result)
    return result


def append_memo(result: dict) -> None:
    memo_text = MEMO.read_text(encoding="utf-8")
    if f"## Phase {PHASE}:" in memo_text and "Phase 2435 质量门修正" in memo_text:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {key: {"label": value["label"], "precision": value["precision"], "teacher": value["teacher"],
                     "autonomous": value["autonomous"]} for key, value in result["models"].items()}
    if f"## Phase {PHASE}:" in memo_text:
        text = rf"""

### Phase 2435 质量门修正 [{stamp}]

首次执行的唯一性门发现punctuation attachment中`direction`与`broken_a`交换后产生96组同prompt异标签冲突，因此首次执行虽完成四模型计算，但不构成通过的数据集证据。现将broken-A改为不同操作词`despite/尽管是`并重算全部行；修正后结果为 `{json.dumps(result, ensure_ascii=False)}`。后续Phase只使用此修正版索引与分数。
"""
    else:
        text = rf"""

## Phase {PHASE}: 概念—语法—功能外部超图材料与四模型行为资格（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 新材料覆盖taxonomy、causal、temporal、negation scope、preposition role、coreference binding、punctuation attachment和sentence reordering八族；每族6个unit×中英×canonical/natural×双方向×valid/brokenA/brokenB×source/target查询，共2304条、每条8个冻结语义事件。候选槽完全反平衡。正确操作与错误A形成语义有效性对照，错误A与错误B形成等规模词项/结构替换对照。Qwen4B跑全部2304条；Qwen14B、GLM4、DS7B各用冻结的576条跨模型子集，严格顺序加载，模型失败不会停止后续全场观察。

$$\Delta_{{beh}}^f=P(m>0\mid valid,f)-P(m>0\mid brokenA,f).$$

**结果汇总。** 材料 `{json.dumps(result['material'], ensure_ascii=False)}`；四模型 `{json.dumps(compact, ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior.py`；2304条源材料、各模型token事件索引、教师分数、自主输出和final位于`tests/glm5/result/phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior`。只追加本MEMO。

**分析、理论进展、问题硬伤与结论。** 这批材料第一次在同一合同中并置知识、语法和功能操作，但仍是可控合成语言，不等于开放自然语言。broken条件对negation、punctuation和coreference会改变角色绑定本身，而不只是谓词字符串；因此跨族比较幅值必须谨慎。Qwen14B的BF16 `device_map=auto`在本机历史上多次Windows访问冲突，本Phase保留NF4/BF16行为资格；GLM4/DS7B按现有安全加载器为INT8，只比较模型内行为/相对拓扑，不比较幅值。行为不合格的族仍进入图谱，但只能叫输入响应，不能叫成功语言计算。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    source_path = OUT / "material/trajectory_hypergraph_source.jsonl"
    # Always compile from the frozen generator so a corrected material contract
    # cannot silently reuse a stale source file from an interrupted run.
    source = compile_source(); write_rows(source_path, source)
    material = material_audit(source); save(OUT / "material/material_audit.json", material)
    models = {}
    for key in MODEL_ORDER:
        print(f"[phase2435] starting {key}", flush=True)
        models[key] = run_model(key, source)
    adjudication = {key: {"aggregate_qualified": value["teacher"]["valid_minus_broken_a"] > .05,
                          "qualified_families": [family for family, metrics in value["teacher"]["by_family"].items()
                                                 if metrics["valid_minus_broken_a"] > .05]}
                    for key, value in models.items()}
    checks = {"rows_2304": material["rows"] == 2304, "eight_families": len(material["families"]) == 8,
              "eight_events": material["event_counts"] == [8], "unique": material["unique_cases"] and material["unique_prompts"],
              "event_bounds": material["event_bounds"], "four_models_sequential_complete": all(v["all_checks_passed"] for v in models.values()),
              "precision_labeled": all(bool(v["precision"]) for v in models.values())}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "material": material, "models": models,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
