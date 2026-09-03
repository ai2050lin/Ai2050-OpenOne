#!/usr/bin/env python3
"""Replicate frozen corrected declarative timing in Qwen3-14B fresh partitions."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase2306_c4021_c4160_corrected_surface_replication"
OUT = RESULT / "phase2307_c4161_c4240_qwen14_corrected_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
FIELD = OUT / "raw/qwen14_corrected_selected_checkpoints.float16.npy"
LOGITS = OUT / "raw/qwen14_corrected_full_vocabulary_logits.float16.npy"
PROGRESS = OUT / "raw/capture_progress.json"
sys.path.insert(0, str(TESTS))

import phase2278_c1961_c2030_qwen14_relative_depth_replication as q14_loader  # noqa: E402
import phase2297_c3161_c3260_qwen4b_ntp_predictive_field as q4field  # noqa: E402
import phase2301_c3501_c3600_qwen14_ntp_timing_replication as prior_q14  # noqa: E402
import phase2303_c3701_c3780_declarative_continuation_contract as contract  # noqa: E402


PHASE = 2307
CAMPAIGN = "C4161-C4240"
EPS = 1e-12


def compile_rows(tokenizer, families: set[str]) -> list[dict]:
    source = contract.read_rows(PARENT / "material/corrected_declarative_continuation_bilingual.jsonl")
    rows = []
    for row in source:
        if row["family"] not in families or row["partition"] not in ("fresh_confirmation", "fresh_lockbox"):
            continue
        prefix = row["declarative_prefix"]
        rows.append({
            **row,
            "ntp_prompt_ids": [int(value) for value in tokenizer.encode(prefix, add_special_tokens=False)],
            "ntp_target_ids": contract.answer_ids(tokenizer, row["correct_answer"], row["language"]),
            "ntp_wrong_ids": contract.answer_ids(tokenizer, row["wrong_answer"], row["language"]),
            "tokenizer_model": "Qwen3-14B",
        })
    contract.write_rows(OUT / "material/qwen14_corrected_fresh_rows.jsonl", rows)
    return rows


def modules(model, qpoints: list[int]) -> dict[int, object]:
    result = {}
    for q in qpoints:
        if q == 0:
            result[q] = model.model.embed_tokens
        elif q == len(model.model.layers) + 1:
            result[q] = model.model.norm
        else:
            result[q] = model.model.layers[q - 1]
    return result


def capture(model, device, rows: list[dict], qpoints: list[int], batch_size: int = 48) -> dict:
    dimension, vocabulary = int(model.config.hidden_size), int(model.config.vocab_size)
    shape, logit_shape = (len(rows), len(qpoints), dimension), (len(rows), vocabulary)
    FIELD.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if FIELD.exists() and LOGITS.exists() and PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if progress["field_shape"] != list(shape) or progress["logit_shape"] != list(logit_shape):
            raise RuntimeError(("resume_shape", progress))
        completed = int(progress["completed"])
        field = np.lib.format.open_memmap(FIELD, mode="r+")
        logits_file = np.lib.format.open_memmap(LOGITS, mode="r+")
    else:
        field = np.lib.format.open_memmap(FIELD, mode="w+", dtype=np.float16, shape=shape)
        logits_file = np.lib.format.open_memmap(LOGITS, mode="w+", dtype=np.float16, shape=logit_shape)
    captures, handles = {}, []
    for q, module in modules(model, qpoints).items():
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, position_ids = q4field.pad_batch([row["ntp_prompt_ids"] for row in batch], device, pad)
                output = model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                               use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for qi, q in enumerate(qpoints):
                    selected = torch.stack([captures[q][i, ends[i]] for i in range(len(batch))])
                    field[start:start + len(batch), qi] = selected.float().cpu().numpy().astype(np.float16)
                selected_logits = torch.stack([output.logits[i, ends[i]] for i in range(len(batch))])
                logits_file[start:start + len(batch)] = selected_logits.float().cpu().numpy().astype(np.float16)
                field.flush()
                logits_file.flush()
                contract.save(PROGRESS, {"completed": start + len(batch), "field_shape": list(shape),
                                         "logit_shape": list(logit_shape), "qpoints": qpoints})
                print(f"[phase2307 capture] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return {"field_path": str(FIELD.relative_to(ROOT)), "field_shape": list(shape),
            "logits_path": str(LOGITS.relative_to(ROOT)), "logits_shape": list(logit_shape),
            "qpoints": qpoints, "dimension": dimension, "vocabulary": vocabulary}


def lens_replication(model, rows: list[dict], qpoints: list[int], cells: list[dict], batch_size: int = 48) -> dict:
    field, actual_file = np.load(FIELD, mmap_mode="r"), np.load(LOGITS, mmap_mode="r")
    qindex = {q: i for i, q in enumerate(qpoints)}
    output = []
    # Accelerate leaves disk-offloaded parameter placeholders on meta. Its hooks
    # materialize norm/lm_head on cuda:0 when the input is a real CUDA tensor.
    device = torch.device("cuda:0")
    for cell in cells:
        if not cell["eligible"]:
            continue
        family_rows = [(i, row) for i, row in enumerate(rows) if row["family"] == cell["family"]]
        q = int(cell["qwen14_checkpoint"])
        for start in range(0, len(family_rows), batch_size):
            batch = family_rows[start:start + batch_size]
            indices = [item[0] for item in batch]
            h = torch.tensor(np.asarray(field[indices, qindex[q]], dtype=np.float32),
                             device=device, dtype=torch.float16)
            with torch.inference_mode():
                lens = model.lm_head(model.model.norm(h)).float()
            actual = torch.tensor(np.asarray(actual_file[indices], dtype=np.float32), device=device)
            p, final_p = torch.softmax(lens, dim=-1), torch.softmax(actual, dim=-1)
            midpoint = 0.5 * (p + final_p)
            js = 0.5 * torch.sum(p * (torch.log(p + EPS) - torch.log(midpoint + EPS)), dim=-1)
            js += 0.5 * torch.sum(final_p * (torch.log(final_p + EPS) - torch.log(midpoint + EPS)), dim=-1)
            for local, (_index, row) in enumerate(batch):
                target, wrong = int(row["ntp_target_ids"][0]), int(row["ntp_wrong_ids"][0])
                output.append({
                    "case_id": row["case_id"], "family": row["family"], "language": row["language"],
                    "surface": row["surface"], "partition": row["partition"], "state": row["state"],
                    "checkpoint": q, "lens_margin": float((lens[local, target] - lens[local, wrong]).item()),
                    "actual_margin": float((actual[local, target] - actual[local, wrong]).item()),
                    "lens_sign_correct": bool(lens[local, target] > lens[local, wrong]),
                    "actual_sign_correct": bool(actual[local, target] > actual[local, wrong]),
                    "js_to_actual_final": float(js[local].item()),
                })
        print(f"[phase2307 lens] {cell['family']} q={q}", flush=True)
    path = OUT / "prediction/qwen14_frozen_timing.jsonl"
    contract.write_rows(path, output)
    families = {}
    for cell in cells:
        if not cell["eligible"]:
            families[cell["family"]] = {"eligible": False, "passed": False, "status": "NA_q4_ineligible"}
            continue
        values = [row for row in output if row["family"] == cell["family"]]
        partitions = {}
        for partition in ("fresh_confirmation", "fresh_lockbox"):
            subset = [row for row in values if row["partition"] == partition]
            partitions[partition] = {
                "rows": len(subset),
                "lens_sign_accuracy": float(np.mean([row["lens_sign_correct"] for row in subset])),
                "actual_sign_accuracy": float(np.mean([row["actual_sign_correct"] for row in subset])),
                "mean_js_to_actual": float(np.mean([row["js_to_actual_final"] for row in subset])),
            }
        passed = all(value["lens_sign_accuracy"] >= cell["gate"] for value in partitions.values())
        families[cell["family"]] = {"eligible": True, "checkpoint": cell["qwen14_checkpoint"],
                                     "partitions": partitions, "passed": passed}
    return {"families": families,
            "passed_families": [family for family, value in families.items() if value.get("passed")],
            "rows": len(output)}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact_behavior = {family: {"qualified": value["qualified"],
                                 "mean": value["slices"]["overall:all"]["mean_accuracy"],
                                 "sum": value["slices"]["overall:all"]["sum_accuracy"]}
                        for family, value in result["sequence_ledger"]["families"].items()}
    text = rf"""

## Phase {PHASE}: Qwen3-14B 修复后续写形成时点前瞻复验（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 严格读取 Phase2306 在加载 14B 前冻结的三个语义多样族及模型相对时点：施事—受事、态度—事件、位置绑定都从 4B `q30` 映射到 14B `q33`。只使用 fresh-confirmation 与 fresh-lockbox 共 `{result['material_rows']}` 行；按 Qwen3-14B 自身 tokenizer 重新编译同一裸文本前缀，先比较正确/错误完整候选序列，再保存 `q33`、final norm 的全部 5120 坐标和完整 151936 词表。三个族顺序运行于同一个 14B 实例，不加载其他模型，也不读取 Attention/MLP 内部量。

$$
q_{{14}}=\operatorname{{round}}(40q_4/36)=33,
\qquad
\operatorname{{Acc}}_{{sign}}(q_{{14}},p)\ge0.75
\quad\forall p\in\{{fresh\ confirmation,fresh\ lockbox\}}.
$$

**结果与门槛。** 14B 完整候选行为 `{json.dumps(compact_behavior, ensure_ascii=False)}`；行为合格族 `{result['sequence_ledger']['qualified_families']}`。冻结时点复验 `{json.dumps(result['replication'], ensure_ascii=False)}`；全场 `{json.dumps(result['field'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。通过表示“在同一家族更大模型中，冻结相对深度的统一辅助读出可区分正确/错误首 token”，不表示相同物理坐标、相同参数电路、跨架构不变量或因果闭合。Qwen3-4B 与 14B 训练相关，不能当独立架构复现；材料仍是受控模板，词汇也沿用同一生成规则；logit lens 是外加尺。脚本 `tests/glm5/phase2307_c4161_c4240_qwen14_corrected_replication.py`；结果 `tests/glm5/result/phase2307_c4161_c4240_qwen14_corrected_replication`。下一步只发布重要的 4B/14B 原坐标图谱、验证前端构建并清理未展示 raw HiddenState，不再加载模型。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parent = json.loads((PARENT / "analysis/final.json").read_text(encoding="utf-8"))
    freeze = parent["q14_freeze"]
    if not parent["all_checks_passed"] or not freeze["frozen_before_qwen14_load"]:
        raise RuntimeError("Phase2306 Qwen3-14B freeze incomplete")
    cells = freeze["cells"]
    families = {cell["family"] for cell in cells if cell["eligible"]}
    qpoints = sorted({int(cell["qwen14_checkpoint"]) for cell in cells if cell["eligible"]} | {41})
    model = tokenizer = None
    try:
        def reduced_map() -> dict:
            value = {"model.embed_tokens": 0}
            value.update({f"model.layers.{i}": 0 if i < 12 else "disk" for i in range(40)})
            value.update({"model.norm": "disk", "model.rotary_emb": "cpu", "lm_head": "disk"})
            return value
        q14_loader.device_map = reduced_map
        model, tokenizer, device = q14_loader.load_model()
        rows = compile_rows(tokenizer, families)
        score_path = OUT / "behavior/sequence_scores.jsonl"
        if score_path.exists():
            scores = contract.read_rows(score_path)
        else:
            scores = q4field.sequence_scores(model, device, rows, batch_size=48)
            contract.write_rows(score_path, scores)
        ledger = prior_q14.fresh_sequence_ledger(rows, scores)
        contract.save(OUT / "behavior/sequence_ledger.json", ledger)
        field = capture(model, device, rows, qpoints)
        replication = lens_replication(model, rows, qpoints, cells)
        model_info = {"name": "Qwen3-14B", "precision": "float16", "quantization": "none",
                      "placement": "cuda_disk_offload", "layers": len(model.model.layers),
                      "hidden_size": int(model.config.hidden_size), "vocabulary": int(model.config.vocab_size)}
    finally:
        if model is not None:
            del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {
        "freeze_precedes_model_load": freeze["frozen_before_qwen14_load"],
        "only_frozen_families": set(row["family"] for row in rows) == families,
        "fresh_partitions_only": set(row["partition"] for row in rows) == {"fresh_confirmation", "fresh_lockbox"},
        "all_sequence_rows": len(scores) == len(rows),
        "selected_full_coordinates": field["field_shape"] == [len(rows), len(qpoints), 5120],
        "full_vocabulary": field["logits_shape"] == [len(rows), 151936],
        "replication_rows": replication["rows"] == len(rows),
        "no_coordinate_identity_transfer": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "material_rows": len(rows), "model": model_info,
        "sequence_ledger": ledger, "field": field, "replication": replication,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "hashes": {"rows": contract.file_hash(OUT / "material/qwen14_corrected_fresh_rows.jsonl"),
                   "scores": contract.file_hash(score_path), "field": contract.file_hash(FIELD),
                   "logits": contract.file_hash(LOGITS),
                   "replication": contract.file_hash(OUT / "prediction/qwen14_frozen_timing.jsonl")},
        "strict_conclusion": (
            f"Qwen3-14B passed {len(replication['passed_families'])}/{len(families)} prospectively frozen "
            "corrected declarative timing cells; this is model-family readout replication, not shared-coordinate "
            "or causal-mechanism closure."
        ),
        "next_authorization": "Publish verified exact-coordinate atlases, build the client without browser control, clean undisplayed raw fields, and close the campaign.",
    }
    contract.save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
