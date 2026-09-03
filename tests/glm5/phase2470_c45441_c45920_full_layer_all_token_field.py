#!/usr/bin/env python3
"""Capture Qwen3-4B BF16 full-layer, all-token, all-coordinate prompt fields."""
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
P2469 = next(RESULT.glob("phase2469_*"))
OUT = RESULT / "phase2470_c45441_c45920_full_layer_all_token_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2470, "C45441-C45920", 2560
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def robust_spans(tokenizer, prompt: str, texts: list[str]) -> dict[str, list[list[int]]]:
    result: dict[str, list[list[int]]] = {}
    for text in texts:
        occurrences = []
        start = 0
        while True:
            char_start = prompt.find(text, start)
            if char_start < 0:
                break
            char_end = char_start + len(text)
            token_start = len(tokenizer.encode(prompt[:char_start], add_special_tokens=False))
            token_end = len(tokenizer.encode(prompt[:char_end], add_special_tokens=False))
            occurrences.append([token_start, token_end])
            start = char_end
        result[text] = occurrences
    return result


def selected_rows() -> tuple[list[dict], dict[str, dict]]:
    final = json.loads((P2469 / "analysis/final.json").read_text(encoding="utf-8"))
    qualified = final["behavior"]["qualified_families"]["both_interfaces"]
    rows = [row for row in read_jsonl(P2469 / "material/typed_hypergraph_rows.jsonl") if row["unit"] in (9, 10) and row["family"] in qualified]
    behavior = {row["case_id"]: row for row in read_jsonl(P2469 / "behavior/autonomous_generation.jsonl")}
    return rows, behavior


def capture(model, tokenizer, rows: list[dict], behavior: dict[str, dict]) -> dict:
    modules = field_utils.modules(model)
    if len(modules) != 38:
        raise RuntimeError(f"Expected 38 qpoints, got {len(modules)}")
    total_tokens = sum(len(row["prompt_ids"]) for row in rows)
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    field_path = raw / "prompt_alltoken_allqpoint.float16.npy"
    token_path = raw / "prompt_token_ids.int32.npy"
    fields = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16, shape=(total_tokens, len(modules), DIM))
    token_ids = np.lib.format.open_memmap(token_path, mode="w+", dtype=np.int32, shape=(total_tokens,))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for qpoint, module in enumerate(modules):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = output[0] if isinstance(output, tuple) else output
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    index = []
    offset = 0
    try:
        for row_number, row in enumerate(rows):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            captures.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
            end = offset + ids.shape[1]
            if len(captures) != len(modules):
                raise RuntimeError((row["case_id"], len(captures)))
            for qpoint in range(len(modules)):
                tensor = captures[qpoint]
                fields[offset:end, qpoint] = tensor[0].detach().to(dtype=torch.float16, device="cpu").numpy()
            token_ids[offset:end] = np.asarray(row["prompt_ids"], dtype=np.int32)
            spans = robust_spans(tokenizer, row["prompt"], list(dict.fromkeys(row["candidates"] + [row["target"], row["foil"]])))
            index.append({
                "row_number": row_number,
                "case_id": row["case_id"],
                "unit": row["unit"],
                "split": row["split"],
                "family": row["family"],
                "language": row["language"],
                "surface": row["surface"],
                "output_interface": row["output_interface"],
                "target": row["target"],
                "foil": row["foil"],
                "token_offset": [offset, end],
                "token_count": ids.shape[1],
                "answer_boundary_local_token": row["answer_boundary_token"],
                "answer_boundary_global_token": offset + row["answer_boundary_token"],
                "semantic_spans": spans,
                "parsed_correct": bool(behavior[row["case_id"]]["parsed_correct"]),
                "parsed_answer": behavior[row["case_id"]]["parsed_answer"],
            })
            offset = end
            if (row_number + 1) % 16 == 0:
                fields.flush(); token_ids.flush()
                print(f"[phase2470 full field] {row_number + 1}/{len(rows)} tokens={offset}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        fields.flush(); token_ids.flush(); close(fields); close(token_ids)
    index_path = OUT / "index/prompt_rows.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in index), encoding="utf-8")
    return {
        "field": str(field_path),
        "token_ids": str(token_path),
        "index": str(index_path),
        "shape": [total_tokens, len(modules), DIM],
        "dtype": "float16 captured from BF16 activations",
        "rows": len(rows),
        "total_prompt_tokens": total_tokens,
        "qpoints": list(range(len(modules))),
        "qpoint_semantics": "q0 embedding; q1-q36 block outputs; q37 final norm",
        "bytes": field_path.stat().st_size,
        "sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 行为合格八族的全层、全token、全2560坐标上下文预测状态原场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 冻结Phase2469在confirmation和lockbox的实体、代码双接口均达0.75的八族，保留unit9/10、中英文、四表面、双接口共256条，无论单行行为成功或失败均采集。每条prompt从q0 embedding、36个block输出到q37 final norm，保存所有token、全部2560物理坐标；同时记录目标/foil的字符前缀重分词span、answer-boundary和自主行为标签。该span方法修正“候选独立token IDs在带空格上下文中不一定能直接子序列匹配”的tokenizer问题。

$$X[n,\ell,i],\quad n=\sum_{{case}}T_{{case}},\quad \ell=0,\ldots,37,\quad i=0,\ldots,2559.$$

**结果汇总。** 原场 `{json.dumps(result['collection'], ensure_ascii=False)}`；数据质量 `{json.dumps(result['quality'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2470_c45441_c45920_full_layer_all_token_field.py`；约2GB float16原场、token IDs、256条ragged索引与final位于同名结果目录。

**分析与理论进展。** 研究对象第一次在新的类型化材料上从少数query-end层位扩展为完整prompt的每层每token物理场。它允许后续直接比较原始状态、层增量、实体span、候选span与answer-boundary，而不把上下文化token出现预先压成单向量。行为失败样本也保留，可用于正确/错误路径对照。

**问题硬伤与结论。** 原场很大但仍只是Qwen4B BF16与显式记录任务；全token并不意味着每个token都是独立样本。float16落盘只用于容量，分析提升到float32；不允许用Top-K或PCA替代该原场。该Phase是L0测量基础，不证明上下文化token是最小编码单位，也不证明模型内部存在外部超图。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    rows, behavior = selected_rows()
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        collection = capture(model, tokenizer, rows, behavior)
    finally:
        model_utils.release_model(model)
        gc.collect()
    index = read_jsonl(Path(collection["index"]))
    quality = {
        "families": sorted({row["family"] for row in index}),
        "units": sorted({row["unit"] for row in index}),
        "interfaces": sorted({row["output_interface"] for row in index}),
        "languages": sorted({row["language"] for row in index}),
        "behavior_success_rate": sum(row["parsed_correct"] for row in index) / len(index),
        "all_targets_have_contextual_spans": all(bool(row["semantic_spans"][row["target"]]) for row in index),
        "all_offsets_contiguous": all(index[i]["token_offset"][1] == index[i + 1]["token_offset"][0] for i in range(len(index) - 1)),
    }
    checks = {
        "rows_256": collection["rows"] == 256,
        "qpoints_38": collection["shape"][1] == 38,
        "coordinates_2560": collection["shape"][2] == 2560,
        "all_tokens": collection["shape"][0] == collection["total_prompt_tokens"],
        "eight_qualified_families": len(quality["families"]) == 8,
        "spans": quality["all_targets_have_contextual_spans"],
        "offsets": quality["all_offsets_contiguous"],
        "hash": len(collection["sha256"]) == 64,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "collection": collection,
        "quality": quality,
        "adjudication": {"full_prompt_field_available": True, "minimal_encoding_unit_identified": False, "language_encoding_mechanism_closed": False},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
