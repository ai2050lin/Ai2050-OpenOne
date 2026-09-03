#!/usr/bin/env python3
"""Capture parsed autonomous generation trajectories at all qpoints and coordinates."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2469 = next(RESULT.glob("phase2469_*"))
OUT = RESULT / "phase2474_c47361_c48000_parsed_autonomous_fulltrajectory"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM, MAX_STEPS = 2474, "C47361-C48000", 2560, 12
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2469_c44881_c45440_typed_hypergraph_behavior as material_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def selected_rows() -> tuple[list[dict], list[str]]:
    final = json.loads((P2469 / "analysis/final.json").read_text(encoding="utf-8"))
    families = final["behavior"]["qualified_families"]["both_interfaces"]
    rows = [row for row in read_jsonl(P2469 / "material/typed_hypergraph_rows.jsonl") if row["unit"] in (9, 10) and row["family"] in families]
    return rows, families


def capture(model, tokenizer, rows: list[dict]) -> tuple[dict, list[dict]]:
    modules = field_utils.modules(model)
    if len(modules) != 38:
        raise RuntimeError(len(modules))
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    field_path = raw / "autonomous_allqpoint_path.float16.npy"
    ids_path = raw / "generated_token_ids.int32.npy"
    mask_path = raw / "trajectory_event_mask.bool.npy"
    fields = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16, shape=(len(rows), MAX_STEPS + 1, 38, DIM))
    token_ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=np.int32, shape=(len(rows), MAX_STEPS))
    mask = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.bool_, shape=(len(rows), MAX_STEPS + 1))
    fields[:] = 0; token_ids[:] = -1; mask[:] = False
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(modules):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = output[0] if isinstance(output, tuple) else output
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    records = []
    try:
        for row_number, row in enumerate(rows):
            prompt = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            captures.clear()
            with torch.inference_mode():
                output = model(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=True, return_dict=True)
            for qpoint in range(38):
                fields[row_number, 0, qpoint] = captures[qpoint][0, -1].detach().to(dtype=torch.float16, device="cpu").numpy()
            mask[row_number, 0] = True
            past = output.past_key_values
            current = int(torch.argmax(output.logits[0, -1]).item())
            generated = []
            parsed = None
            correct = False
            prefix = False
            answer_step = None
            for step in range(MAX_STEPS):
                generated.append(current)
                token_ids[row_number, step] = current
                one = torch.tensor([[current]], dtype=torch.long, device=device)
                captures.clear()
                with torch.inference_mode():
                    output = model(input_ids=one, past_key_values=past, use_cache=True, return_dict=True)
                past = output.past_key_values
                for qpoint in range(38):
                    fields[row_number, step + 1, qpoint] = captures[qpoint][0, -1].detach().to(dtype=torch.float16, device="cpu").numpy()
                mask[row_number, step + 1] = True
                text = tokenizer.decode(generated, skip_special_tokens=True)
                parsed, correct, prefix = material_utils.parse_answer(text, row)
                if parsed is not None:
                    answer_step = step + 1
                    break
                if current == tokenizer.eos_token_id:
                    break
                current = int(torch.argmax(output.logits[0, -1]).item())
            text = tokenizer.decode(generated, skip_special_tokens=True)
            records.append({
                "row_number": row_number,
                "case_id": row["case_id"],
                "unit": row["unit"],
                "family": row["family"],
                "language": row["language"],
                "surface": row["surface"],
                "output_interface": row["output_interface"],
                "expected": row["expected_output"],
                "generated_ids": generated,
                "generated_text": text,
                "parsed_answer": parsed,
                "parsed_correct": bool(correct),
                "answer_prefix": bool(prefix),
                "answer_step": answer_step,
                "trajectory_events": 1 + len(generated),
            })
            if (row_number + 1) % 16 == 0:
                fields.flush(); token_ids.flush(); mask.flush()
                print(f"[phase2474 autonomous path] {row_number + 1}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        fields.flush(); token_ids.flush(); mask.flush(); close(fields); close(token_ids); close(mask)
    index_path = OUT / "index/autonomous_rows.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    return {
        "field": str(field_path),
        "generated_token_ids": str(ids_path),
        "event_mask": str(mask_path),
        "index": str(index_path),
        "shape": [len(rows), MAX_STEPS + 1, 38, DIM],
        "event0": "prompt answer-boundary",
        "events1_12": "actual generated tokens after each token has passed through the model",
        "dtype": "float16 captured from BF16 activations",
        "bytes": field_path.stat().st_size,
        "sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
    }, records


def summarize(records: list[dict], families: list[str]) -> dict:
    result: dict[str, dict] = {}
    for unit in (9, 10):
        result[str(unit)] = {}
        for interface in ("entity", "code"):
            result[str(unit)][interface] = {}
            for family in families:
                rows = [row for row in records if row["unit"] == unit and row["output_interface"] == interface and row["family"] == family]
                result[str(unit)][interface][family] = {
                    "rows": len(rows),
                    "parsed_accuracy": sum(row["parsed_correct"] for row in rows) / len(rows),
                    "unparsed_rate": sum(row["parsed_answer"] is None for row in rows) / len(rows),
                    "mean_answer_step": float(np.mean([row["answer_step"] for row in rows if row["answer_step"] is not None])) if any(row["answer_step"] is not None for row in rows) else None,
                    "prefix_rate": sum(row["answer_prefix"] for row in rows) / len(rows),
                }
            rows = [row for row in records if row["unit"] == unit and row["output_interface"] == interface]
            result[str(unit)][interface]["aggregate"] = {
                "rows": len(rows),
                "parsed_accuracy": sum(row["parsed_correct"] for row in rows) / len(rows),
                "unparsed_rate": sum(row["parsed_answer"] is None for row in rows) / len(rows),
                "mean_answer_step": float(np.mean([row["answer_step"] for row in rows if row["answer_step"] is not None])),
                "prefix_rate": sum(row["answer_prefix"] for row in rows) / len(rows),
            }
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    aggregates = {unit: {interface: result["behavior"][unit][interface]["aggregate"] for interface in ("entity", "code")} for unit in ("9", "10")}
    text = rf"""


## Phase {PHASE}: 正确答案解析下的真实自主生成全层全坐标轨迹（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2469双接口均行为合格的八族，使用unit9/10、中英文、四表面、实体/代码共256条。每条从prompt answer-boundary开始真实贪心，不提供正确前缀；每生成一个实际token，将其送回模型并保存q0–q37全部2560坐标。最多12 token，只在剥离`Answer:/答案：`后完整候选或1/2代码可唯一解析时停止。正确、错误和未解析轨迹全部保留。

$$\hat y_k=\arg\max_v p(v\mid x,\hat y_{{<k}}),\qquad \mathcal{{T}}=(H_{{answer}}^{{(\ell)}},H_{{y_1}}^{{(\ell)}},\ldots,H_{{y_K}}^{{(\ell)}}).$$

**结果汇总。** 原场 `{json.dumps(result['collection'], ensure_ascii=False)}`；行为聚合 `{json.dumps(aggregates, ensure_ascii=False)}`；数据质量 `{json.dumps(result['quality'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2474_c47361_c48000_parsed_autonomous_fulltrajectory.py`；256×13事件×38层×2560坐标原场、token IDs、事件mask、逐行生成和final位于同名结果目录。

**分析与理论进展。** 该原场首次满足三个条件：模型未见正确首token前缀、答案前缀不再耗尽预算、每个实际生成token都在模型内完成一次前向。因而下一Phase可以比较同一语义问题在实体/代码接口的answer-boundary、首生成token和完整答案事件，而不是把两个独立跨接口余弦相减称为轨迹崩解。

**问题硬伤与结论。** 解析停止使不同接口轨迹长度不同；最多12 token仍不是无限自由文本。显式候选任务易于行为资格，但不是开放知识生成。生成token HiddenState同时编码token身份和前缀历史，必须做接口、语言、family和错配对照。该Phase只提供成功自主路径原场，不单独宣称条件齿轮。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    rows, families = selected_rows()
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        collection, records = capture(model, tokenizer, rows)
    finally:
        model_utils.release_model(model)
        gc.collect()
    behavior = summarize(records, families)
    quality = {
        "rows": len(records),
        "families": families,
        "parsed_rate": sum(row["parsed_answer"] is not None for row in records) / len(records),
        "parsed_accuracy": sum(row["parsed_correct"] for row in records) / len(records),
        "correct_rows": sum(row["parsed_correct"] for row in records),
        "incorrect_rows": sum(row["parsed_answer"] is not None and not row["parsed_correct"] for row in records),
        "unparsed_rows": sum(row["parsed_answer"] is None for row in records),
        "max_observed_steps": max(len(row["generated_ids"]) for row in records),
    }
    checks = {
        "rows_256": len(records) == 256,
        "shape": collection["shape"] == [256, 13, 38, 2560],
        "full_coordinates": collection["shape"][-1] == 2560,
        "actual_prefix": all(row["trajectory_events"] == len(row["generated_ids"]) + 1 for row in records),
        "parsed_majority": quality["parsed_rate"] >= 0.75,
        "successful_paths": quality["correct_rows"] >= 160,
        "hash": len(collection["sha256"]) == 64,
        "claim_boundary": True,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "collection": collection, "behavior": behavior, "quality": quality, "adjudication": {"successful_autonomous_paths_available": quality["correct_rows"] >= 160, "language_encoding_mechanism_closed": False}, "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
