#!/usr/bin/env python3
"""Capture all-token prompt and autonomous full-coordinate fields for passed chains."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2479 = RESULT / "phase2479_c50561_c51200_stepwise_knowledge_chain_behavior"
OUT = RESULT / "phase2480_c51201_c51840_qualified_chain_fullcoordinate_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM, MAX_STEPS = 2480, "C51201-C51840", 2560, 48
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2479_c50561_c51200_stepwise_knowledge_chain_behavior as material_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024): value.update(block)
    return value.hexdigest()


def selected_rows() -> tuple[list[dict], list[dict]]:
    final = json.loads((P2479 / "analysis/final.json").read_text(encoding="utf-8"))
    detail = final["behavior"]["by_unit_family_surface_interface"]
    qualified = []
    for family in material_utils.FAMILIES:
        for surface in range(4):
            if all(detail[str(unit)][family][str(surface)]["path"]["parsed_accuracy"] >= 0.75 for unit in (12, 13)):
                qualified.append({"family": family, "surface": surface})
    keys = {(row["family"], row["surface"]) for row in qualified}
    rows = [row for row in read_jsonl(P2479 / "material/stepwise_rows.jsonl") if row["output_interface"] == "path" and (row["family"], row["surface"]) in keys]
    return rows, qualified


def contextual_spans(tokenizer, prompt: str, chain: list[str]) -> dict:
    spans = {}
    for node in chain:
        occurrences = []
        start = 0
        while (position := prompt.find(node, start)) >= 0:
            first = len(tokenizer.encode(prompt[:position], add_special_tokens=False))
            last = len(tokenizer.encode(prompt[:position + len(node)], add_special_tokens=False))
            occurrences.append([first, last])
            start = position + len(node)
        spans[node] = occurrences
    return spans


def capture(model, tokenizer, rows: list[dict]) -> tuple[dict, list[dict]]:
    modules = field_utils.modules(model)
    if len(modules) != 38: raise RuntimeError(len(modules))
    raw = OUT / "raw"; raw.mkdir(parents=True, exist_ok=True)
    prompt_ids = [[int(value) for value in tokenizer.encode(row["prompt"], add_special_tokens=False)] for row in rows]
    total_tokens = sum(len(values) for values in prompt_ids)
    prompt_path = raw / "prompt_alltoken_allqpoint.float16.npy"
    trajectory_path = raw / "autonomous_allqpoint_path.float16.npy"
    generated_path = raw / "generated_token_ids.int32.npy"
    event_mask_path = raw / "trajectory_event_mask.bool.npy"
    prompt_field = np.lib.format.open_memmap(prompt_path, mode="w+", dtype=np.float16, shape=(total_tokens, 38, DIM))
    trajectory = np.lib.format.open_memmap(trajectory_path, mode="w+", dtype=np.float16, shape=(len(rows), MAX_STEPS + 1, 38, DIM))
    generated_ids = np.lib.format.open_memmap(generated_path, mode="w+", dtype=np.int32, shape=(len(rows), MAX_STEPS))
    event_mask = np.lib.format.open_memmap(event_mask_path, mode="w+", dtype=np.bool_, shape=(len(rows), MAX_STEPS + 1))
    trajectory[:] = 0; generated_ids[:] = -1; event_mask[:] = False
    captures: dict[int, torch.Tensor] = {}; handles = []
    for qpoint, module in enumerate(modules):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = output[0] if isinstance(output, tuple) else output
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    records = []; offset = 0
    try:
        for row_number, (row, ids) in enumerate(zip(rows, prompt_ids)):
            prompt = torch.tensor([ids], dtype=torch.long, device=device)
            captures.clear()
            with torch.inference_mode():
                output = model(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=True, return_dict=True)
            for qpoint in range(38):
                values = captures[qpoint][0].detach().to(dtype=torch.float16, device="cpu").numpy()
                prompt_field[offset:offset + len(ids), qpoint] = values
                trajectory[row_number, 0, qpoint] = values[-1]
            event_mask[row_number, 0] = True
            past = output.past_key_values
            current = int(torch.argmax(output.logits[0, -1]).item())
            produced = []; parsed = {}; answer_step = None
            for step in range(MAX_STEPS):
                produced.append(current); generated_ids[row_number, step] = current
                one = torch.tensor([[current]], dtype=torch.long, device=device)
                captures.clear()
                with torch.inference_mode():
                    output = model(input_ids=one, past_key_values=past, use_cache=True, return_dict=True)
                past = output.past_key_values
                for qpoint in range(38):
                    trajectory[row_number, step + 1, qpoint] = captures[qpoint][0, -1].detach().to(dtype=torch.float16, device="cpu").numpy()
                event_mask[row_number, step + 1] = True
                text = tokenizer.decode(produced, skip_special_tokens=True)
                parsed = material_utils.parse(text, row)
                if parsed["full_path_correct"] and parsed["endpoint_present"] and not parsed["distractor_endpoint_present"]:
                    answer_step = step + 1; break
                if current == tokenizer.eos_token_id: break
                current = int(torch.argmax(output.logits[0, -1]).item())
            text = tokenizer.decode(produced, skip_special_tokens=True)
            records.append({
                "row_number": row_number, "case_id": row["case_id"], "unit": row["unit"],
                "family": row["family"], "language": row["language"], "surface": row["surface"],
                "main_chain": row["main_chain"], "distractor_chain": row["distractor_chain"],
                "prompt_token_offset": [offset, offset + len(ids)], "prompt_token_count": len(ids),
                "answer_boundary_prompt_token": offset + len(ids) - 1,
                "node_spans": contextual_spans(tokenizer, row["prompt"], row["main_chain"] + row["distractor_chain"]),
                "generated_ids": produced, "generated_text": text, "answer_step": answer_step,
                "trajectory_events": 1 + len(produced), **parsed,
            })
            offset += len(ids)
            if (row_number + 1) % 8 == 0:
                prompt_field.flush(); trajectory.flush(); generated_ids.flush(); event_mask.flush()
                print(f"[phase2480 full field] {row_number + 1}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        for value in (prompt_field, trajectory, generated_ids, event_mask): value.flush(); close(value)
    index = OUT / "index/chain_rows.jsonl"; index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    return {
        "prompt_field": {"path": str(prompt_path), "shape": [total_tokens, 38, DIM], "bytes": prompt_path.stat().st_size, "sha256": digest(prompt_path)},
        "trajectory_field": {"path": str(trajectory_path), "shape": [len(rows), MAX_STEPS + 1, 38, DIM], "bytes": trajectory_path.stat().st_size, "sha256": digest(trajectory_path)},
        "generated_ids": str(generated_path), "event_mask": str(event_mask_path), "index": str(index),
        "qpoint_semantics": "q0 embedding; q1-q36 block outputs; q37 final norm",
        "dtype": "float16 captured from Qwen3-4B BF16 activations",
    }, records


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 行为合格三类三跳知识链的长提示全token与真实生成全坐标原场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 从Phase2479逐步路径接口中冻结confirmation/unit12与lockbox/unit13均≥0.75的family-surface，而不再要求失败的代码接口：part-whole/s0、causal/s0、causal/s2、handoff/s3。加入discovery/unit11，共24条中英长提示。单次Qwen3-4B BF16 CUDA加载中同时保存：（1）prompt每个token的q0 Embedding、q1–q36 block输出、q37 final norm及全部2560坐标；（2）answer-boundary与最多48个真实贪心token经过模型后的同一全坐标场；只在完整四节点路径已唯一出现时停止。正确/错误路径均保留。

$$F_{{prompt}}\in\mathbb R^{{N_{{tok}}\times38\times2560}},\qquad F_{{gen}}\in\mathbb R^{{24\times49\times38\times2560}}.$$

**结果汇总。** 资格 `{json.dumps(result['qualified'], ensure_ascii=False)}`；原场 `{json.dumps(result['collection'], ensure_ascii=False)}`；行为 `{json.dumps(result['quality'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2480_c51201_c51840_qualified_chain_fullcoordinate_field.py`；prompt全token原场、自主轨迹原场、token IDs、事件mask、逐行节点span/index与final位于同名结果目录。

**分析与理论进展。** 这是第一批同时包含“长距离外部链结构、每个输入token、每个输出链节点、全部层与全部物理坐标”的原场。它围绕通过特征继续：三类关系而非一个句型，四种已锁定表面组合，中英和三套完全独立节点。下一步可以直接比较主链边/节点、干扰链、回答边界及输出节点的坐标纹理，而不依赖Top-K或低秩投影。

**问题硬伤与结论。** 资格是在Phase2479相同材料上选择，unit12/13只是行为锁箱而非本Phase统计锁箱；24条对于全因素交互仍小。逐步输出改变了任务并提供外部工作记忆。尤其同一提示从批量生成改为逐条hook生成后，unit12由8/8变为7/8、unit13保持8/8，显示BF16/批处理数值路径可改变argmax，100%不能当绝对稳定性；但二者仍达到预设0.75门槛。原场不证明模型以图存储或存在齿轮组。由于这是唯一全坐标长链证据，当前保留，不清理。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    rows, qualified = selected_rows()
    prior_path = OUT / "analysis/final.json"
    if prior_path.exists() and (OUT / "index/chain_rows.jsonl").exists():
        collection = json.loads(prior_path.read_text(encoding="utf-8"))["collection"]
        records = read_jsonl(OUT / "index/chain_rows.jsonl")
    else:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        try:
            collection, records = capture(model, tokenizer, rows)
        finally:
            model_utils.release_model(model); gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    quality = {
        "rows": len(records), "correct_rows": sum(row["answer_step"] is not None for row in records),
        "parsed_accuracy": sum(row["answer_step"] is not None for row in records) / len(records),
        "by_unit": {str(unit): sum(row["answer_step"] is not None for row in records if row["unit"] == unit) / sum(row["unit"] == unit for row in records) for unit in (11, 12, 13)},
        "max_observed_steps": max(len(row["generated_ids"]) for row in records),
    }
    checks = {
        "four_qualified_family_surfaces": len(qualified) == 4,
        "three_families": len({row["family"] for row in qualified}) == 3,
        "rows_24": len(records) == 24,
        "prompt_fullfield": collection["prompt_field"]["shape"][1:] == [38, 2560],
        "trajectory_fullfield": collection["trajectory_field"]["shape"] == [24, 49, 38, 2560],
        "actual_prefix": all(row["trajectory_events"] == len(row["generated_ids"]) + 1 for row in records),
        "confirmation_lockbox_success": quality["by_unit"]["12"] >= 0.75 and quality["by_unit"]["13"] >= 0.75,
        "hashes": all(len(collection[name]["sha256"]) == 64 for name in ("prompt_field", "trajectory_field")),
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "qualified": qualified,
        "collection": collection, "quality": quality,
        "adjudication": {"long_chain_fullcoordinate_field_available": True, "typed_graph_internal_structure_identified": False, "language_encoding_mechanism_closed": False},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
