#!/usr/bin/env python3
"""Prospective Qwen3-14B replication of three frozen NTP formation checkpoints."""
from __future__ import annotations

import gc
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
CONTRACT_OUT = RESULT / "phase2296_c3101_c3160_ntp_predictive_contract"
FREEZE_OUT = RESULT / "phase2300_c3441_c3500_fisher_audit_q14_contract"
OUT = RESULT / "phase2301_c3501_c3600_qwen14_ntp_timing_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
FIELD = OUT / "raw/qwen14_ntp_selected_checkpoints.float16.npy"
LOGITS = OUT / "raw/qwen14_ntp_full_vocabulary_logits.float16.npy"
PROGRESS = OUT / "raw/capture_progress.json"
sys.path.insert(0, str(TESTS))

import phase2278_c1961_c2030_qwen14_relative_depth_replication as q14  # noqa: E402
import phase2296_c3101_c3160_ntp_predictive_contract as contract  # noqa: E402
import phase2297_c3161_c3260_qwen4b_ntp_predictive_field as q4field  # noqa: E402


PHASE = 2301
CAMPAIGN = "C3501-C3600"
EPS = 1e-12


def fresh_sequence_ledger(rows: list[dict], scores: list[dict]) -> dict:
    by_id = {row["case_id"]: row for row in scores}
    families = {}
    qualified = []
    for family in contract.Q14_FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        slices = {}
        keys = [("overall", "all")]
        keys += [("language", value) for value in ("en", "zh")]
        keys += [("surface", value) for value in ("narrative", "dialogue")]
        keys += [("partition", value) for value in ("fresh_confirmation", "fresh_lockbox")]
        for kind, value in keys:
            subset = family_rows if kind == "overall" else [row for row in family_rows if row[kind] == value]
            scored = [by_id[row["case_id"]] for row in subset]
            slices[f"{kind}:{value}"] = {
                "rows": len(scored),
                "mean_accuracy": float(np.mean([row["correct_by_mean"] for row in scored])),
                "sum_accuracy": float(np.mean([row["correct_by_sum"] for row in scored])),
                "mean_margin": float(np.mean([row["mean_margin"] for row in scored])),
            }
        passed = all(min(value["mean_accuracy"], value["sum_accuracy"]) >= contract.BEHAVIOR_GATE
                     for value in slices.values())
        families[family] = {"qualified": passed, "slices": slices}
        if passed:
            qualified.append(family)
    return {"families": families, "qualified_families": qualified, "gate": contract.BEHAVIOR_GATE}


def compile_rows(tokenizer, families: set[str]) -> list[dict]:
    source = contract.read_rows(CONTRACT_OUT / "material/ntp_natural_bilingual.jsonl")
    system = "Answer only from the supplied text. Do not use outside knowledge."
    rows = []
    for row in source:
        if row["family"] not in families or row["partition"] not in ("fresh_confirmation", "fresh_lockbox"):
            continue
        prompt_ids = contract.compiler.core.chat_ids(tokenizer, system, row["free_prompt"])
        rows.append({
            **row,
            "ntp_prompt_ids": [int(value) for value in prompt_ids],
            "ntp_target_ids": contract.answer_ids(tokenizer, row["correct_answer"], row["language"]),
            "ntp_wrong_ids": contract.answer_ids(tokenizer, row["wrong_answer"], row["language"]),
            "tokenizer_model": "Qwen3-14B",
        })
    contract.write_rows(OUT / "material/qwen14_ntp_fresh_rows.jsonl", rows)
    return rows


def modules(model, qpoints: list[int]) -> dict[int, Any]:
    output = {}
    for q in qpoints:
        if q == 0:
            output[q] = model.model.embed_tokens
        elif q == len(model.model.layers) + 1:
            output[q] = model.model.norm
        else:
            output[q] = model.model.layers[q - 1]
    return output


def capture(model, device, rows: list[dict], qpoints: list[int], batch_size: int = 48) -> dict:
    dimension = int(model.config.hidden_size)
    vocab = int(model.config.vocab_size)
    FIELD.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(rows), len(qpoints), dimension)
    logit_shape = (len(rows), vocab)
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
    captures = {}
    handles = []
    for q, module in modules(model, qpoints).items():
        def hook(_module, _inputs, output, q=q):
            captures[q] = output[0] if isinstance(output, tuple) else output
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
                    tensor = captures[q]
                    selected = torch.stack([tensor[i, ends[i]] for i in range(len(batch))])
                    field[start:start + len(batch), qi] = selected.detach().float().cpu().numpy().astype(np.float16)
                selected_logits = torch.stack([output.logits[i, ends[i]] for i in range(len(batch))])
                logits_file[start:start + len(batch)] = selected_logits.detach().float().cpu().numpy().astype(np.float16)
                field.flush()
                logits_file.flush()
                contract.save(PROGRESS, {"completed": start + len(batch), "field_shape": list(shape),
                                         "logit_shape": list(logit_shape), "qpoints": qpoints})
                print(f"[phase2301 capture] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return {"field_path": str(FIELD.relative_to(ROOT)), "field_shape": list(shape),
            "logits_path": str(LOGITS.relative_to(ROOT)), "logits_shape": list(logit_shape),
            "qpoints": qpoints, "dimension": dimension, "vocabulary": vocab}


def lens_replication(model, rows: list[dict], qpoints: list[int], cells: list[dict], batch_size: int = 48) -> dict:
    field = np.load(FIELD, mmap_mode="r")
    actual_file = np.load(LOGITS, mmap_mode="r")
    qindex = {q: i for i, q in enumerate(qpoints)}
    output = []
    device = torch.device("cuda:0")
    for cell in cells:
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
            p = torch.softmax(lens, dim=-1)
            final_p = torch.softmax(actual, dim=-1)
            midpoint = 0.5 * (p + final_p)
            js = 0.5 * torch.sum(p * (torch.log(p + EPS) - torch.log(midpoint + EPS)), dim=-1)
            js += 0.5 * torch.sum(final_p * (torch.log(final_p + EPS) - torch.log(midpoint + EPS)), dim=-1)
            for local, (index, row) in enumerate(batch):
                target = int(row["ntp_target_ids"][0])
                wrong = int(row["ntp_wrong_ids"][0])
                output.append({
                    "case_id": row["case_id"], "family": row["family"], "language": row["language"],
                    "surface": row["surface"], "partition": row["partition"], "state": row["state"],
                    "checkpoint": q,
                    "lens_margin": float((lens[local, target] - lens[local, wrong]).item()),
                    "actual_margin": float((actual[local, target] - actual[local, wrong]).item()),
                    "lens_sign_correct": bool(lens[local, target] > lens[local, wrong]),
                    "actual_sign_correct": bool(actual[local, target] > actual[local, wrong]),
                    "js_to_actual_final": float(js[local].item()),
                })
        print(f"[phase2301 lens] {cell['family']} q={q}", flush=True)
    contract.write_rows(OUT / "prediction/qwen14_frozen_timing.jsonl", output)
    families = {}
    for cell in cells:
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
        families[cell["family"]] = {"checkpoint": cell["qwen14_checkpoint"], "partitions": partitions,
                                     "passed": passed}
    return {"families": families, "passed_families": [key for key, value in families.items() if value["passed"]],
            "rows": len(output)}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-14B预测形成层前瞻复验（{CAMPAIGN}） [{stamp}]

**测试原理与用例。** 在加载14B前，Phase2296已固定施事—受事、态度—事件、位置绑定三构式；Phase2299只用4B discovery+confirmation冻结形成点，并映射为14B相对block深度：`agent_patient@q33`、`attitude_event@q28`、`location_binding@q28`。本期只运行 fresh-confirmation 与 fresh-lockbox 共 `{result['material_rows']}` 行。先比较正确/错误完整词汇答案的teacher-forced序列概率，再保存冻结检查点与final norm全部5120坐标及实际完整词表logits。4B物理坐标没有搬到14B。

**公式。** 相对深度映射和通过门为：

$$
q_{{14}}=\operatorname{{round}}(40q_4/36),
\qquad
\operatorname{{Acc}}_{{sign}}(q_{{14}},p)\ge0.75
\quad\forall p\in\{{fresh\ confirmation,fresh\ lockbox\}}.
$$

**结果汇总。** 序列行为 `{json.dumps(result['sequence_ledger'], ensure_ascii=False)}`；形成层复验 `{json.dumps(result['replication'], ensure_ascii=False)}`；全坐标与词表 `{json.dumps(result['field'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展、问题硬伤与结论。** `{result['strict_conclusion']}`。通过只表示“构式、相对深度、目标竞争读出”在同模型家族更大规模中复现，不表示相同坐标、相同参数电路或跨架构普遍规律。Qwen3-14B与4B训练数据相关；材料答案词并非全分区独立；logit lens仍是辅助尺。未运行Fisher、Koopman、TDA或因果干预。脚本 `tests/glm5/phase2301_c3501_c3600_qwen14_ntp_timing_replication.py`；结果 `tests/glm5/result/phase2301_c3501_c3600_qwen14_ntp_timing_replication`。
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
    freeze = json.loads((FREEZE_OUT / "analysis/final.json").read_text(encoding="utf-8"))
    if not freeze["all_checks_passed"]:
        raise RuntimeError("Phase2300 Qwen3-14B contract is not authorized")
    cells = freeze["q14_contract"]["cells"]
    qpoints = sorted({int(cell["qwen14_checkpoint"]) for cell in cells} | {41})
    model = tokenizer = None
    try:
        def reduced_map() -> dict:
            value = {"model.embed_tokens": 0}
            value.update({f"model.layers.{i}": 0 if i < 12 else "disk" for i in range(40)})
            value.update({"model.norm": "disk", "model.rotary_emb": "cpu", "lm_head": "disk"})
            return value
        q14.device_map = reduced_map
        model, tokenizer, device = q14.load_model()
        rows = compile_rows(tokenizer, set(contract.Q14_FAMILIES))
        score_path = OUT / "behavior/lexical_sequence_scores.jsonl"
        if score_path.exists():
            scores = contract.read_rows(score_path)
        else:
            scores = q4field.sequence_scores(model, device, rows, batch_size=48)
            contract.write_rows(score_path, scores)
        ledger = fresh_sequence_ledger(rows, scores)
        contract.save(OUT / "behavior/lexical_sequence_ledger.json", ledger)
        field = capture(model, device, rows, qpoints)
        replication = lens_replication(model, rows, qpoints, cells)
        model_info = {"name": "Qwen3-14B", "precision": "float16", "placement": "cuda_disk_offload",
                      "hidden_size": int(model.config.hidden_size), "layers": len(model.model.layers),
                      "vocabulary": int(model.config.vocab_size)}
    finally:
        if model is not None:
            del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {
        "contract_frozen_before_load": freeze["q14_contract"]["frozen_before_qwen14_load"],
        "material_rows": len(rows) == freeze["q14_contract"]["rows"],
        "all_rows_sequence_scored": len(scores) == len(rows),
        "all_coordinates_saved": field["dimension"] == 5120,
        "full_vocabulary_saved": field["logits_shape"] == [len(rows), model_info["vocabulary"]],
        "only_frozen_qpoints": field["qpoints"] == qpoints,
        "replication_rows": replication["rows"] == len(rows),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "material_rows": len(rows), "sequence_ledger": ledger,
        "field": field, "replication": replication, "model": model_info, "checks": checks,
        "hashes": {"field": contract.file_hash(FIELD), "logits": contract.file_hash(LOGITS),
                   "scores": contract.file_hash(OUT / "behavior/lexical_sequence_scores.jsonl"),
                   "replication": contract.file_hash(OUT / "prediction/qwen14_frozen_timing.jsonl")},
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            f"Qwen3-14B passed {len(replication['passed_families'])}/3 frozen model-relative formation checkpoints "
            "on both fresh partitions; this is cross-scale predictive timing evidence, not shared-coordinate or causal closure."
        ),
        "next_authorization": "Publish the exact-coordinate NTP atlases, verify the client, clean undisplayed HiddenState raw fields, and close the campaign.",
    }
    contract.save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
