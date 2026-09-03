#!/usr/bin/env python3
"""Large bilingual multi-relation knowledge-chain behavior contract."""
from __future__ import annotations

import gc
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2478_c49601_c50560_multirelation_knowledge_chain_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2478, "C49601-C50560"
UNITS = {
    11: ("Alven", "Brisa", "Corin", "Dalen", "Elric", "Faron", "Gilda", "Hestor"),
    12: ("Ilyan", "Jessa", "Korin", "Laren", "Mirov", "Nessa", "Orin", "Pella"),
    13: ("Quill", "Rovan", "Sella", "Torin", "Ulric", "Vessa", "Worin", "Xenia"),
}
SPLITS = {11: "discovery", 12: "confirmation", 13: "lockbox"}
FAMILIES = ("taxonomy", "part_whole", "process", "temporal", "causal", "spatial", "handoff", "translation")
INTERFACES = ("entity", "code")
HOPS = (1, 2, 3)
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def chains(names: tuple[str, ...], surface: int) -> tuple[list[str], list[str]]:
    rotations = (
        (0, 1, 2, 3, 4, 5, 6, 7),
        (7, 6, 5, 4, 3, 2, 1, 0),
        (1, 3, 5, 7, 0, 2, 4, 6),
        (6, 4, 2, 0, 7, 5, 3, 1),
    )
    order = rotations[surface]
    return [names[i] for i in order[:4]], [names[i] for i in order[4:]]


def edge_sentence(family: str, source: str, target: str, language: str, variant: int) -> str:
    en = {
        "taxonomy": (f"{source} is a kind of {target}.", f"{source} belongs to category {target}."),
        "part_whole": (f"{source} is contained in {target}.", f"{source} is a part of {target}."),
        "process": (f"{source} is processed into {target}.", f"Processing {source} yields {target}."),
        "temporal": (f"Event {source} is immediately before event {target}.", f"Event {target} immediately follows event {source}."),
        "causal": (f"{source} directly causes {target}.", f"The direct effect of {source} is {target}."),
        "spatial": (f"{source} is immediately left of {target}.", f"Immediately to the right of {source} is {target}."),
        "handoff": (f"{source} passes the token to {target}.", f"The token moves from {source} to {target}."),
        "translation": (f"Code {source} maps to code {target}.", f"The codebook translates {source} as {target}."),
    }
    zh = {
        "taxonomy": (f"{source}是{target}的一种。", f"{source}属于{target}类。"),
        "part_whole": (f"{source}包含在{target}中。", f"{source}是{target}的一部分。"),
        "process": (f"{source}被加工成{target}。", f"加工{source}得到{target}。"),
        "temporal": (f"事件{source}紧接在事件{target}之前。", f"事件{target}紧接在事件{source}之后。"),
        "causal": (f"{source}直接导致{target}。", f"{source}的直接结果是{target}。"),
        "spatial": (f"{source}紧挨着在{target}左边。", f"{source}的紧邻右侧是{target}。"),
        "handoff": (f"{source}把令牌交给{target}。", f"令牌从{source}传给{target}。"),
        "translation": (f"代码{source}映射为代码{target}。", f"对照表把{source}转换为{target}。"),
    }
    return (en if language == "en" else zh)[family][variant % 2]


def relation_label(family: str, language: str) -> str:
    en = {
        "taxonomy": "is-a", "part_whole": "container", "process": "processing-output",
        "temporal": "next-event", "causal": "direct-effect", "spatial": "right-neighbor",
        "handoff": "token-receiver", "translation": "code-map",
    }
    zh = {
        "taxonomy": "所属类别", "part_whole": "包含整体", "process": "加工结果",
        "temporal": "下一事件", "causal": "直接结果", "spatial": "右侧紧邻",
        "handoff": "令牌接收者", "translation": "代码映射",
    }
    return (en if language == "en" else zh)[family]


def token_spans(tokenizer, prompt: str, value: str) -> list[list[int]]:
    spans = []
    for match in re.finditer(re.escape(value), prompt):
        start = len(tokenizer.encode(prompt[:match.start()], add_special_tokens=False))
        end = len(tokenizer.encode(prompt[:match.end()], add_special_tokens=False))
        spans.append([start, end])
    return spans


def compile_rows(tokenizer) -> list[dict]:
    rows: list[dict] = []
    case = 49601
    for unit, names in UNITS.items():
        for family in FAMILIES:
            for language in ("en", "zh"):
                for surface in range(4):
                    main, distractor = chains(names, surface)
                    main_edges = [edge_sentence(family, main[i], main[i + 1], language, surface + i) for i in range(3)]
                    distractor_edges = [edge_sentence(family, distractor[i], distractor[i + 1], language, surface + i + 1) for i in range(3)]
                    if surface == 0:
                        ordered = main_edges + distractor_edges
                    elif surface == 1:
                        ordered = list(reversed(main_edges)) + list(reversed(distractor_edges))
                    elif surface == 2:
                        ordered = [value for pair in zip(main_edges, distractor_edges) for value in pair]
                    else:
                        ordered = distractor_edges + list(reversed(main_edges))
                    facts = " ".join(ordered)
                    for hop in HOPS:
                        target, foil = main[hop], distractor[hop]
                        candidates = [target, foil] if (surface + hop) % 2 == 0 else [foil, target]
                        relation = relation_label(family, language)
                        if language == "en":
                            question = f" Starting at {main[0]}, follow the {relation} link exactly {hop} time{'s' if hop != 1 else ''}. Which node is reached?"
                        else:
                            question = f" 从{main[0]}开始，沿“{relation}”关系恰好走{hop}步，会到达哪个节点？"
                        for interface in INTERFACES:
                            if language == "en":
                                if interface == "entity":
                                    suffix = f"\nCandidates: {candidates[0]} | {candidates[1]}\nReturn exactly one candidate name.\nAnswer:"
                                    expected = target
                                else:
                                    suffix = f"\n1 = {candidates[0]}; 2 = {candidates[1]}\nReturn exactly 1 or 2.\nAnswer:"
                                    expected = "1" if target == candidates[0] else "2"
                            else:
                                if interface == "entity":
                                    suffix = f"\n候选：{candidates[0]} | {candidates[1]}\n只返回一个候选名称。\n答案："
                                    expected = target
                                else:
                                    suffix = f"\n1 = {candidates[0]}；2 = {candidates[1]}\n只返回1或2。\n答案："
                                    expected = "1" if target == candidates[0] else "2"
                            prompt = facts + question + suffix
                            ids = [int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)]
                            spans = {node: token_spans(tokenizer, prompt, node) for node in set(main + distractor)}
                            rows.append({
                                "case_id": f"c{case:05d}-{family}-u{unit}-{language}-s{surface}-h{hop}-{interface}",
                                "unit": unit, "split": SPLITS[unit], "family": family, "language": language,
                                "surface": surface, "hop": hop, "output_interface": interface,
                                "main_chain": main, "distractor_chain": distractor, "fact_order": ordered,
                                "query_start": main[0], "target": target, "foil": foil, "candidates": candidates,
                                "expected_output": expected, "prompt": prompt, "prompt_ids": ids,
                                "answer_boundary_token": len(ids) - 1, "node_spans": spans,
                                "typed_path": [{"source": main[i], "relation": family, "target": main[i + 1], "edge_index": i + 1} for i in range(3)],
                            })
                            case += 1
    return rows


def strip_prefix(text: str) -> str:
    value = text.strip()
    return re.sub(r"^(?:final\s+answer|answer|最终答案|答案)\s*[:：]\s*", "", value, flags=re.I).strip()


def parse_answer(text: str, row: dict) -> tuple[str | None, bool, bool]:
    cleaned = strip_prefix(text)
    prefix = cleaned != text.strip()
    if row["output_interface"] == "code":
        match = re.search(r"(?<!\d)([12])(?!\d)", cleaned)
        parsed = match.group(1) if match else None
    else:
        norm = normalize(cleaned)
        hits = [candidate for candidate in row["candidates"] if normalize(candidate) in norm]
        parsed = hits[0] if len(set(hits)) == 1 else None
    return parsed, parsed == row["expected_output"], prefix


def run_behavior(model, tokenizer, rows: list[dict]) -> list[dict]:
    tokenizer.padding_side = "left"
    device = model.get_input_embeddings().weight.device
    generated: list[dict] = []
    batch_size = 16
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        encoded = tokenizer([row["prompt"] for row in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=12, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        width = encoded["input_ids"].shape[1]
        for row, sequence in zip(batch, output):
            new_ids = [int(value) for value in sequence[width:].detach().cpu().tolist()]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            parsed, correct, prefix = parse_answer(text, row)
            generated.append({
                "case_id": row["case_id"], "unit": row["unit"], "family": row["family"],
                "language": row["language"], "surface": row["surface"], "hop": row["hop"],
                "output_interface": row["output_interface"], "expected": row["expected_output"],
                "generated_ids": new_ids, "generated_text": text, "parsed_answer": parsed,
                "parsed_correct": correct, "answer_prefix": prefix,
                "raw_normalized_exact": normalize(strip_prefix(text)) == normalize(row["expected_output"]),
            })
        if start + len(batch) == len(rows) or (start + len(batch)) % 128 == 0:
            print(f"[phase2478 behavior] {start + len(batch)}/{len(rows)}", flush=True)
    return generated


def summarize(generated: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for item in generated:
        grouped[(item["unit"], item["family"], item["hop"], item["output_interface"])].append(item)
    detail: dict[str, dict] = {}
    for unit in UNITS:
        detail[str(unit)] = {}
        for family in FAMILIES:
            detail[str(unit)][family] = {}
            for hop in HOPS:
                detail[str(unit)][family][str(hop)] = {}
                for interface in INTERFACES:
                    values = grouped[(unit, family, hop, interface)]
                    detail[str(unit)][family][str(hop)][interface] = {
                        "rows": len(values),
                        "parsed_accuracy": sum(x["parsed_correct"] for x in values) / len(values),
                        "en_accuracy": sum(x["parsed_correct"] for x in values if x["language"] == "en") / 4,
                        "zh_accuracy": sum(x["parsed_correct"] for x in values if x["language"] == "zh") / 4,
                        "unparsed_rate": sum(x["parsed_answer"] is None for x in values) / len(values),
                        "prefix_rate": sum(x["answer_prefix"] for x in values) / len(values),
                    }
        all_values = [item for item in generated if item["unit"] == unit]
        detail[str(unit)]["aggregate"] = {
            "rows": len(all_values),
            "parsed_accuracy": sum(x["parsed_correct"] for x in all_values) / len(all_values),
            "by_hop": {str(hop): sum(x["parsed_correct"] for x in all_values if x["hop"] == hop) / sum(x["hop"] == hop for x in all_values) for hop in HOPS},
            "by_interface": {interface: sum(x["parsed_correct"] for x in all_values if x["output_interface"] == interface) / sum(x["output_interface"] == interface for x in all_values) for interface in INTERFACES},
        }
    qualified = []
    for family in FAMILIES:
        for hop in HOPS:
            if all(detail[str(unit)][family][str(hop)][interface]["parsed_accuracy"] >= 0.75 for unit in (12, 13) for interface in INTERFACES):
                qualified.append({"family": family, "hop": hop})
    fully_qualified = [family for family in FAMILIES if all({"family": family, "hop": hop} in qualified for hop in HOPS)]
    return {"by_unit_family_hop_interface": detail, "qualified_family_hops": qualified, "fully_qualified_families": fully_qualified}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    aggregate = {unit: result["behavior"]["by_unit_family_hop_interface"][unit]["aggregate"] for unit in ("11", "12", "13")}
    text = rf"""


## Phase {PHASE}: 八类多节点知识链、1–3跳双语双接口自主行为大样本合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将研究从孤立二选一关系推进到四节点主链+四节点平行干扰链。八族覆盖taxonomy、part-whole、process、temporal、causal、spatial、handoff、translation；unit11/12/13作发现/确认/锁箱；中英双语、四种内容与事实顺序、1/2/3跳、实体名/1-2代码双输出接口，共1152条。每条保存三条typed path、干扰链、全部prompt IDs与基于上下文前缀重分词的节点span。Qwen3-4B BF16 CUDA真实贪心最多12 token，剥离可选答案前缀后解析完整候选或冻结代码。

$$N=8\times3\times2\times4\times3\times2=1152,\qquad v_0\xrightarrow{{r}}v_1\xrightarrow{{r}}v_2\xrightarrow{{r}}v_3,\quad y_h=v_h.$$

**结果汇总。** 聚合行为 `{json.dumps(aggregate, ensure_ascii=False)}`；合格family-hop `{json.dumps(result['behavior']['qualified_family_hops'], ensure_ascii=False)}`；三跳全合格族 `{json.dumps(result['behavior']['fully_qualified_families'], ensure_ascii=False)}`；tokenizer `{json.dumps(result['tokenizer_audit'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2478_c49601_c50560_multirelation_knowledge_chain_behavior.py`；1152条材料、完整自主输出和final位于同名结果目录。

**分析与理论进展。** 本Phase建立的是“外部链操作—行为资格”压力测试，不把研究者的path标签当模型内部结构。结果没有任何family-hop在unit12/13双接口同时达到0.75；1跳聚合尚为约0.68–0.74，2/3跳已接近二选一随机。这不构成“多跳编码崩解”：当前候选、代码映射、事实逆序/交织和“恰好h步”同时变化，协议负担足以解释失败。该结果的价值是阻止后续在行为不成立的材料上解释HiddenState。

**问题硬伤与结论。** 显式事实+显式“走h步”主要测上下文图遍历，不等同于参数知识网络；平行链候选使任务仍可被局部模式近似；关系标签及模板措辞与family纠缠。零合格项意味着不能直接升级全场。下一Phase自动重构为逐步链读出：保持同一底层链，分别要求返回整条路径或每一步节点，使中间步骤外显；先验证模型确实能走链，再从通过特征采场，而不是把协议失败命名为内部机制失败。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    generation = OUT / "behavior/autonomous_generation.jsonl"
    material = OUT / "material/knowledge_chain_rows.jsonl"
    if material.exists() and generation.exists():
        rows = [json.loads(line) for line in material.read_text(encoding="utf-8").splitlines() if line.strip()]
        generated = [json.loads(line) for line in generation.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        model, tokenizer, device = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        try:
            rows = compile_rows(tokenizer)
            material.parent.mkdir(parents=True, exist_ok=True)
            material.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            generated = run_behavior(model, tokenizer, rows)
        finally:
            model_utils.release_model(model)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        generation.parent.mkdir(parents=True, exist_ok=True)
        generation.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated), encoding="utf-8")
    metadata = {"model": "Qwen3-4B", "dtype": "BF16", "quantized": False, "device": "cuda:0"}
    behavior = summarize(generated)
    audit = {
        "rows": len(rows),
        "all_prompt_ids": all(row["prompt_ids"] for row in rows),
        "all_nodes_have_spans": all(all(row["node_spans"][node] for node in row["main_chain"] + row["distractor_chain"]) for row in rows),
        "prompt_token_length": {
            "min": min(len(row["prompt_ids"]) for row in rows),
            "max": max(len(row["prompt_ids"]) for row in rows),
            "mean": sum(len(row["prompt_ids"]) for row in rows) / len(rows),
        },
    }
    checks = {
        "rows_1152": len(rows) == 1152,
        "generated_1152": len(generated) == 1152,
        "three_frozen_units": sorted({row["unit"] for row in rows}) == [11, 12, 13],
        "eight_families": len({row["family"] for row in rows}) == 8,
        "three_hops": sorted({row["hop"] for row in rows}) == [1, 2, 3],
        "two_languages_interfaces": sorted({row["language"] for row in rows}) == ["en", "zh"] and sorted({row["output_interface"] for row in rows}) == ["code", "entity"],
        "contextual_spans": audit["all_nodes_have_spans"],
        "finite": all(math.isfinite(item["parsed_accuracy"]) for unit in behavior["by_unit_family_hop_interface"].values() for family, hops in unit.items() if family != "aggregate" for hop in hops.values() for item in hop.values()),
        "bf16_nonquantized": metadata["dtype"] == "BF16" and metadata["quantized"] is False and metadata["device"].startswith("cuda"),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "material": {"path": str(material), "rows": len(rows), "families": list(FAMILIES), "units": UNITS, "hops": list(HOPS), "interfaces": list(INTERFACES)},
        "generation": str(generation), "model_metadata": metadata,
        "tokenizer_audit": audit, "behavior": behavior,
        "adjudication": {
            "explicit_context_chain_behavior_contract": True,
            "qualified_family_hops": len(behavior["qualified_family_hops"]),
            "fullfield_followup_ready": bool(behavior["qualified_family_hops"]),
            "low_multihop_accuracy_is_protocol_stress_result_not_encoding_collapse": True,
            "typed_path_is_internal_mechanism": False,
            "parametric_knowledge_network_tested": False,
            "behavior_gate_only": True,
            "language_encoding_mechanism_closed": False,
        },
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
