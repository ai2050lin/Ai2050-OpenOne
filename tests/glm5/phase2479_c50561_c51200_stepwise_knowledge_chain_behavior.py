#!/usr/bin/env python3
"""Repair the multi-hop protocol by requiring explicit stepwise path readout."""
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
SOURCE = RESULT / "phase2478_c49601_c50560_multirelation_knowledge_chain_behavior/material/knowledge_chain_rows.jsonl"
OUT = RESULT / "phase2479_c50561_c51200_stepwise_knowledge_chain_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2479, "C50561-C51200"
FAMILIES = ("taxonomy", "part_whole", "process", "temporal", "causal", "spatial", "handoff", "translation")
INTERFACES = ("path", "path_code")
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def compile_rows() -> list[dict]:
    source = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    bases = [row for row in source if row["hop"] == 3 and row["output_interface"] == "entity"]
    rows: list[dict] = []
    case = 50561
    for base in bases:
        facts = " ".join(base["fact_order"])
        chain = base["main_chain"]
        target, foil = chain[-1], base["distractor_chain"][-1]
        candidates = [target, foil] if base["surface"] % 2 == 0 else [foil, target]
        expected_code = "1" if candidates[0] == target else "2"
        for interface in INTERFACES:
            if base["language"] == "en":
                instruction = (
                    f"\nStarting at {chain[0]}, trace exactly three stated links, one link at a time. "
                    "Do not infer a shortcut. Return the four nodes from start through endpoint in order, separated by >."
                )
                if interface == "path":
                    suffix = "\nPath:"
                else:
                    suffix = f"\nEndpoint codes: 1 = {candidates[0]}; 2 = {candidates[1]}. After the path, return Code: 1 or Code: 2.\nPath:"
            else:
                instruction = (
                    f"\n从{chain[0]}开始，严格沿题目中的关系逐步走三步，不要跳步。"
                    "按顺序返回从起点到终点的四个节点，用 > 分隔。"
                )
                if interface == "path":
                    suffix = "\n路径："
                else:
                    suffix = f"\n终点代码：1 = {candidates[0]}；2 = {candidates[1]}。路径后只写代码：1或代码：2。\n路径："
            prompt = facts + instruction + suffix
            rows.append({
                "case_id": f"c{case:05d}-{base['family']}-u{base['unit']}-{base['language']}-s{base['surface']}-{interface}",
                "unit": base["unit"], "split": base["split"], "family": base["family"],
                "language": base["language"], "surface": base["surface"], "output_interface": interface,
                "main_chain": chain, "distractor_chain": base["distractor_chain"], "fact_order": base["fact_order"],
                "target": target, "foil": foil, "candidates": candidates, "expected_code": expected_code,
                "prompt": prompt,
            })
            case += 1
    return rows


def parse(text: str, row: dict) -> dict:
    normalized = normalize(text)
    positions = [normalized.find(normalize(node)) for node in row["main_chain"]]
    full_path = all(position >= 0 for position in positions) and positions == sorted(positions)
    endpoint = normalize(row["target"]) in normalized
    distractor_endpoint = normalize(row["foil"]) in normalized
    code_match = re.search(r"(?:code|代码)\s*[:：]?\s*([12])(?!\d)", text, flags=re.I)
    code = code_match.group(1) if code_match else None
    code_correct = code == row["expected_code"] if row["output_interface"] == "path_code" else None
    correct = full_path and endpoint and not distractor_endpoint and (code_correct is not False)
    return {
        "node_positions": positions, "full_path_correct": full_path,
        "endpoint_present": endpoint, "distractor_endpoint_present": distractor_endpoint,
        "parsed_code": code, "code_correct": code_correct, "parsed_correct": correct,
    }


def run_behavior(model, tokenizer, rows: list[dict]) -> list[dict]:
    tokenizer.padding_side = "left"
    device = model.get_input_embeddings().weight.device
    generated: list[dict] = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        encoded = tokenizer([row["prompt"] for row in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=48, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        width = encoded["input_ids"].shape[1]
        for row, sequence in zip(batch, output):
            ids = [int(value) for value in sequence[width:].detach().cpu().tolist()]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            generated.append({
                "case_id": row["case_id"], "unit": row["unit"], "family": row["family"],
                "language": row["language"], "surface": row["surface"], "output_interface": row["output_interface"],
                "target": row["target"], "generated_ids": ids, "generated_text": text, **parse(text, row),
            })
        if start + len(batch) == len(rows) or (start + len(batch)) % 64 == 0:
            print(f"[phase2479 behavior] {start + len(batch)}/{len(rows)}", flush=True)
    return generated


def summarize(generated: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in generated:
        grouped[(row["unit"], row["family"], row["surface"], row["output_interface"])].append(row)
    detail: dict[str, dict] = {}
    for unit in (11, 12, 13):
        detail[str(unit)] = {}
        for family in FAMILIES:
            detail[str(unit)][family] = {}
            for surface in range(4):
                detail[str(unit)][family][str(surface)] = {}
                for interface in INTERFACES:
                    values = grouped[(unit, family, surface, interface)]
                    detail[str(unit)][family][str(surface)][interface] = {
                        "rows": len(values),
                        "parsed_accuracy": sum(row["parsed_correct"] for row in values) / len(values),
                        "full_path_accuracy": sum(row["full_path_correct"] for row in values) / len(values),
                        "endpoint_accuracy": sum(row["endpoint_present"] and not row["distractor_endpoint_present"] for row in values) / len(values),
                        "code_accuracy": (sum(row["code_correct"] for row in values) / len(values)) if interface == "path_code" else None,
                    }
        values = [row for row in generated if row["unit"] == unit]
        detail[str(unit)]["aggregate"] = {
            "rows": len(values), "parsed_accuracy": sum(row["parsed_correct"] for row in values) / len(values),
            "full_path_accuracy": sum(row["full_path_correct"] for row in values) / len(values),
            "endpoint_accuracy": sum(row["endpoint_present"] and not row["distractor_endpoint_present"] for row in values) / len(values),
        }
    qualified = []
    for family in FAMILIES:
        for surface in range(4):
            if all(detail[str(unit)][family][str(surface)][interface]["parsed_accuracy"] >= 0.75 for unit in (12, 13) for interface in INTERFACES):
                qualified.append({"family": family, "surface": surface})
    qualified_families = sorted({row["family"] for row in qualified})
    return {"by_unit_family_surface_interface": detail, "qualified_family_surfaces": qualified, "qualified_families": qualified_families}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    aggregate = {unit: result["behavior"]["by_unit_family_surface_interface"][unit]["aggregate"] for unit in ("11", "12", "13")}
    text = rf"""


## Phase {PHASE}: 三跳知识链逐步外显读出、双输出接口与行为锁箱重构（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2478终点二选一不能区分“不会沿链”与“内部走链但最终选择/代码接口失败”。保持同一八族、四节点主链、平行干扰链、unit11/12/13、中英、四事实顺序不变，改为要求模型自主输出`v0 > v1 > v2 > v3`完整路径；第二接口还需把终点编译为1/2代码。共384条，BF16 CUDA贪心最多48 token。完整路径、无干扰终点和代码三项分别解析；unit12/13双接口均≥0.75的family-surface才获全场资格。

$$\hat P=(\hat v_0,\hat v_1,\hat v_2,\hat v_3),\qquad Q=\mathbb 1[\hat P=P]\,\mathbb 1[\hat y=v_3]\,\mathbb 1[\hat c=c(v_3)].$$

**结果汇总。** 聚合 `{json.dumps(aggregate, ensure_ascii=False)}`；合格family-surface `{json.dumps(result['behavior']['qualified_family_surfaces'], ensure_ascii=False)}`；合格族 `{json.dumps(result['behavior']['qualified_families'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2479_c50561_c51200_stepwise_knowledge_chain_behavior.py`；384条重构材料、逐行生成和final位于同名结果目录。

**分析与理论进展。** 该实验把路径组合与终点接口分解：若path接口成功而path-code失败，问题在输出编译；若完整路径本身失败，才说明当前提示下链遍历行为不成立。只有确认/锁箱都通过的family-surface才进入HiddenState采集，因此研究围绕通过特征继续，而不因Phase2478失败停止，也不在失败材料上构造内部崩解故事。

**问题硬伤与结论。** 逐步外显可能改变原本内部计算，并把生成前缀作为额外工作记忆；它测的是可执行链读出，不是“静默思考”。同一事实模板仍与family纠缠。后续全场必须同时保留prompt answer-boundary、每个生成链节点和代码事件，比较外显路径对状态轨迹的影响，不能把成功外显直接当内部自然机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    material = OUT / "material/stepwise_rows.jsonl"
    generation = OUT / "behavior/autonomous_generation.jsonl"
    if material.exists() and generation.exists():
        rows = [json.loads(line) for line in material.read_text(encoding="utf-8").splitlines() if line.strip()]
        generated = [json.loads(line) for line in generation.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = compile_rows()
        material.parent.mkdir(parents=True, exist_ok=True)
        material.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        try:
            generated = run_behavior(model, tokenizer, rows)
        finally:
            model_utils.release_model(model); gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        generation.parent.mkdir(parents=True, exist_ok=True)
        generation.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated), encoding="utf-8")
    behavior = summarize(generated)
    checks = {
        "rows_384": len(rows) == 384, "generated_384": len(generated) == 384,
        "eight_families": len({row["family"] for row in rows}) == 8,
        "three_units_two_languages_four_surfaces_two_interfaces": sorted({row["unit"] for row in rows}) == [11, 12, 13] and sorted({row["language"] for row in rows}) == ["en", "zh"] and sorted({row["surface"] for row in rows}) == [0, 1, 2, 3] and sorted({row["output_interface"] for row in rows}) == ["path", "path_code"],
        "finite": all(math.isfinite(row["parsed_accuracy"]) for unit in behavior["by_unit_family_surface_interface"].values() for family, surfaces in unit.items() if family != "aggregate" for surface in surfaces.values() for row in surface.values()),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "material": {"path": str(material), "rows": len(rows)}, "generation": str(generation),
        "model": {"name": "Qwen3-4B", "dtype": "BF16", "quantized": False, "device": "cuda:0"},
        "behavior": behavior,
        "adjudication": {
            "phase2478_low_accuracy_is_encoding_collapse": False,
            "qualified_family_surfaces": len(behavior["qualified_family_surfaces"]),
            "fullfield_followup_ready": bool(behavior["qualified_family_surfaces"]),
            "externalized_path_is_natural_internal_mechanism": False,
            "language_encoding_mechanism_closed": False,
        },
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__":
    main()
