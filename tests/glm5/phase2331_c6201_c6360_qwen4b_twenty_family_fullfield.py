#!/usr/bin/env python3
"""Run Qwen3-4B FP16 behavior and broad full-coordinate language-family fields."""
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
P2330 = RESULT / "phase2330_c6081_c6200_language_family_atlas_contract"
OUT = RESULT / "phase2331_c6201_c6360_qwen4b_twenty_family_fullfield"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = P2330 / "material/typed_language_family_atlas.jsonl"
BOUNDARY = OUT / "raw/qwen4b_fp16_boundary_all_checkpoints.float16.npy"
BOUNDARY_PROGRESS = OUT / "raw/qwen4b_fp16_boundary_progress.json"
LOGITS = OUT / "raw/qwen4b_fp16_representative_full_vocabulary_logits.float16.npy"
ALL_TOKEN = OUT / "raw/qwen4b_fp16_representative_all_token_qpoints.float16.npy"
ALL_TOKEN_PROGRESS = OUT / "raw/qwen4b_fp16_all_token_progress.json"
PHASE = 2331
CAMPAIGN = "C6201-C6360"
QPOINTS = (0, 10, 20, 30, 37)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2330_c6081_c6200_language_family_atlas_contract as contract  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return io.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    io.write_rows(path, rows)


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def summarize(values: list[dict], free: bool = False) -> dict:
    if not values:
        return {"rows": 0}
    if free:
        return {
            "rows": len(values),
            "identity_accuracy": float(np.mean([row["first_identity_correct"] for row in values])),
            "target_found": float(np.mean([row["target_found"] for row in values])),
            "wrong_found": float(np.mean([row["wrong_found"] for row in values])),
            "future_prefix_exact": float(np.mean([row["future_prefix_exact"] for row in values])),
        }
    return {
        "rows": len(values),
        "sum_accuracy": float(np.mean([row["correct_by_sum"] for row in values])),
        "mean_accuracy": float(np.mean([row["correct_by_mean"] for row in values])),
        "first_accuracy": float(np.mean([row["correct_first"] for row in values])),
        "sum_margin": float(np.mean([row["sum_margin"] for row in values])),
        "mean_margin": float(np.mean([row["mean_margin"] for row in values])),
    }


def behavior_ledger(scores: list[dict], free: list[dict]) -> dict:
    families = {}
    qualified = []
    for family in contract.FAMILIES:
        family_scores = [row for row in scores if row["family"] == family]
        family_free = [row for row in free if row["family"] == family]
        slices = {}
        passed = True
        for partition in contract.PARTITIONS:
            s = summarize([row for row in family_scores if row["partition"] == partition])
            f = summarize([row for row in family_free if row["partition"] == partition], True)
            slices[partition] = {"candidate": s, "free": f}
            passed = passed and min(s["sum_accuracy"], s["mean_accuracy"]) >= 0.70 and f["identity_accuracy"] >= 0.40
        families[family] = {
            "qualified": passed, "candidate": summarize(family_scores),
            "free": summarize(family_free, True), "partitions": slices,
        }
        if passed:
            qualified.append(family)
    return {
        "candidate_gate_each_partition": 0.70, "free_identity_gate_each_partition": 0.40,
        "qualified_families": qualified, "families": families,
        "overall": {"candidate": summarize(scores), "free": summarize(free, True)},
        "claim_boundary": "behavior qualification for this generated atlas only; no human blind review",
    }


def capture_boundary(model, device, rows: list[dict], batch_size: int = 12) -> dict:
    module_list = modules(model)
    shape = (len(rows), len(module_list), int(model.config.hidden_size))
    BOUNDARY.parent.mkdir(parents=True, exist_ok=True)
    if BOUNDARY.exists() and BOUNDARY_PROGRESS.exists():
        progress = json.loads(BOUNDARY_PROGRESS.read_text(encoding="utf-8"))
        if progress["shape"] != list(shape):
            raise RuntimeError(("boundary_resume_shape", progress, shape))
        completed = int(progress["completed"])
        field = np.lib.format.open_memmap(BOUNDARY, mode="r+")
    else:
        completed = 0
        field = np.lib.format.open_memmap(BOUNDARY, mode="w+", dtype=np.float16, shape=shape)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(module_list):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = baseline.pad_right([row["future_prompt_ids"] for row in batch], device, pad)
                captures.clear()
                model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for q in range(len(module_list)):
                    selected = torch.stack([captures[q][local, ends[local]] for local in range(len(batch))])
                    field[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                field.flush()
                save(BOUNDARY_PROGRESS, {"completed": start + len(batch), "shape": list(shape)})
                print(f"[phase2331 boundary] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush(); close_memmap(field)
    return {"path": str(BOUNDARY.relative_to(ROOT)), "shape": list(shape), "dtype": "float16"}


def representative_rows(rows: list[dict]) -> list[dict]:
    values = [row for row in rows if int(row["unit"]) in (0, 9)]
    values.sort(key=lambda row: row["design_index"])
    return values


def capture_representative(model, device, rows: list[dict], batch_size: int = 8) -> dict:
    selected = representative_rows(rows)
    module_list = modules(model)
    if max(QPOINTS) >= len(module_list):
        raise RuntimeError(("qpoints", QPOINTS, len(module_list)))
    total_rows = sum(len(row["future_prompt_ids"]) * len(QPOINTS) for row in selected)
    shape = (total_rows, int(model.config.hidden_size))
    logits_shape = (len(selected), int(model.config.vocab_size))
    ALL_TOKEN.parent.mkdir(parents=True, exist_ok=True)
    if ALL_TOKEN.exists() and LOGITS.exists() and ALL_TOKEN_PROGRESS.exists():
        progress = json.loads(ALL_TOKEN_PROGRESS.read_text(encoding="utf-8"))
        if progress["shape"] != list(shape) or progress["logits_shape"] != list(logits_shape):
            raise RuntimeError(("representative_resume_shape", progress, shape, logits_shape))
        completed = int(progress["completed_samples"])
        cursor = int(progress["coordinate_rows_written"])
        field = np.lib.format.open_memmap(ALL_TOKEN, mode="r+")
        logits = np.lib.format.open_memmap(LOGITS, mode="r+")
        segments = read_rows(OUT / "index/representative_segments.jsonl")
    else:
        completed, cursor, segments = 0, 0, []
        field = np.lib.format.open_memmap(ALL_TOKEN, mode="w+", dtype=np.float16, shape=shape)
        logits = np.lib.format.open_memmap(LOGITS, mode="w+", dtype=np.float16, shape=logits_shape)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q in QPOINTS:
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module_list[q].register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(selected), batch_size):
                batch = selected[start:start + batch_size]
                ids, mask, positions = baseline.pad_right([row["future_prompt_ids"] for row in batch], device, pad)
                captures.clear()
                output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                boundary_logits = torch.stack([output.logits[local, ends[local]] for local in range(len(batch))])
                logits[start:start + len(batch)] = boundary_logits.float().cpu().numpy().astype(np.float16)
                for local, row in enumerate(batch):
                    length = len(row["future_prompt_ids"])
                    for q in QPOINTS:
                        begin, end = cursor, cursor + length
                        field[begin:end] = captures[q][local, :length].float().cpu().numpy().astype(np.float16)
                        segments.append({
                            "case_id": row["case_id"], "design_index": row["design_index"],
                            "family": row["family"], "macrotype": row["macrotype"],
                            "language": row["language"], "surface": row["surface"],
                            "partition": row["partition"], "unit": row["unit"], "state": row["state"],
                            "qpoint": q, "start": begin, "stop": end, "token_count": length,
                        })
                        cursor = end
                field.flush(); logits.flush()
                write_rows(OUT / "index/representative_segments.jsonl", segments)
                save(ALL_TOKEN_PROGRESS, {
                    "completed_samples": start + len(batch), "coordinate_rows_written": cursor,
                    "shape": list(shape), "logits_shape": list(logits_shape), "qpoints": list(QPOINTS),
                })
                print(f"[phase2331 all-token] {start + len(batch)}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush(); logits.flush(); close_memmap(field); close_memmap(logits)
    if cursor != total_rows:
        raise RuntimeError(("all_token_cursor", cursor, total_rows))
    write_rows(OUT / "index/representative_rows.jsonl", selected)
    return {
        "selected_rows": len(selected), "qpoints": list(QPOINTS),
        "all_token_path": str(ALL_TOKEN.relative_to(ROOT)), "all_token_shape": list(shape),
        "logits_path": str(LOGITS.relative_to(ROOT)), "logits_shape": list(logits_shape),
        "segments": len(segments), "embedding_included": 0 in QPOINTS,
        "claim_boundary": "balanced representative token field; boundary field covers every row and checkpoint",
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: Qwen3-4B 二十语言模式族行为与全坐标基线场（{CAMPAIGN}） [{stamp}]

**测试原理、测试用例与公式。** 严格读取 Phase2330 在模型加载前冻结的 1920 行材料，单独以本地非量化 Qwen3-4B FP16 运行二十个语言模式族。每行同时计算正确/错误完整候选序列总对数概率、长度归一化概率和自由续写；全部材料保存 embedding、36个 block 后状态及 final norm 的边界位置全部2560坐标。另对 unit0 与从未参与发现的 unit9 平衡代表行保存 embedding/q10/q20/q30/final 的全 token×全坐标场和完整151936词表边界 logits。典型用例横跨属性、part-whole、因果、否定、量词、介词、指代、标点、引语、翻译、风格、句法绑定及原八族，不把任一族的失败推广为其他族失败。

$$
S(y\mid x)=\sum_{{r=1}}^{{|y|}}\log p(y_r\mid x,y_{{<r}}),\qquad
\bar S(y\mid x)=S(y\mid x)/|y|.
$$

$$
\Psi(x)=\{{H_{{q,t,j}}(x):q=0,\ldots,37;\ j=1,\ldots,2560\}}.
$$

**结果汇总与相关文件。** 行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`。边界场 `{json.dumps(result['boundary'], ensure_ascii=False)}`；代表性全 token/embedding/完整词表场 `{json.dumps(result['representative'], ensure_ascii=False)}`；数值身份 `{json.dumps(result['model'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2331_c6201_c6360_qwen4b_twenty_family_fullfield.py`；结果 `tests/glm5/result/phase2331_c6201_c6360_qwen4b_twenty_family_fullfield`。

**分析、理论进展、问题硬伤与结论。** 本期建立广谱语言族观察相机和行为分母，不命名语义齿轮。完整候选依赖 teacher forcing，自由生成的字符串身份判分会漏掉合法释义；材料由程序生成且无独立人类盲评；翻译和风格是受控微型任务；同一模型物理坐标可比较，但单坐标仍不是换基不变量。只有同时跨四分区通过候选0.70和自由身份0.40的族具备严格语义解释资格；其他族的内部场仍可用于发现编码候选，但只能称无行为授权观察。全坐标场不做Top-K、PCA或预先压缩，低值坐标完整保留。

**下一阶段。** 目标仍是同一“广泛语言模式族编码图谱”，因此自动继续：先用FP16/BF16与四剂量审计数值微分区，再在 discovery/confirmation 上建立全坐标复用—差异护照，冻结后读取 fresh 分区。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = json.loads((P2330 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2330 contract is not valid")
    rows = read_rows(MATERIAL)
    model = tokenizer = None
    try:
        model, tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        sequence_path = OUT / "behavior/sequence_scores.jsonl"
        free_path = OUT / "behavior/free_generation.jsonl"
        scores = baseline.sequence_scores(model, device, rows, sequence_path, batch_size=20)
        free = baseline.free_generation(model, tokenizer, device, rows, free_path, batch_size=24)
        behavior = behavior_ledger(scores, free)
        boundary = capture_boundary(model, device, rows)
        representative = capture_representative(model, device, rows)
        dtypes = defaultdict(int)
        for parameter in model.parameters():
            dtypes[str(parameter.dtype).replace("torch.", "")] += int(parameter.numel())
        model_record = {
            "name": "Qwen3-4B", "precision": "float16_nonquantized", "device": str(device),
            "parameter_dtypes": dict(dtypes), "hidden_size": int(model.config.hidden_size),
            "layers": len(model.model.layers), "vocabulary": int(model.config.vocab_size),
        }
    finally:
        if model is not None:
            model_utils.release_model(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {
        "all_behavior_rows": behavior["overall"]["candidate"]["rows"] == len(rows),
        "all_free_rows": behavior["overall"]["free"]["rows"] == len(rows),
        "boundary_all_rows_checkpoints_coordinates": boundary["shape"] == [len(rows), 38, 2560],
        "representative_embedding_included": representative["embedding_included"],
        "representative_full_vocabulary": representative["logits_shape"][1] == 151936,
        "fp16_nonquantized": model_record["parameter_dtypes"].get("float16", 0) > 0,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "behavior": behavior,
        "boundary": boundary, "representative": representative, "model": model_record,
        "hashes": {
            "boundary": file_hash(BOUNDARY), "all_token": file_hash(ALL_TOKEN),
            "logits": file_hash(LOGITS), "material": file_hash(MATERIAL),
        },
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("phase2331_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
