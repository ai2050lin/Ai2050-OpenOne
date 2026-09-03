#!/usr/bin/env python3
"""Run Qwen3-4B multi-step behavior and full-coordinate future observations."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase2309_c4321_c4440_multistep_future_contract"
OUT = RESULT / "phase2310_c4441_c4580_qwen4b_multistep_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = PARENT / "material/multistep_future_bilingual.jsonl"
RAW = OUT / "raw"
BOUNDARY = RAW / "qwen4b_multistep_boundary_all_checkpoints.float16.npy"
BOUNDARY_LOGITS = RAW / "qwen4b_multistep_boundary_full_vocabulary.float16.npy"
BOUNDARY_PROGRESS = RAW / "boundary_progress.json"
FUTURE_FIELD = RAW / "qwen4b_teacher_future_selected_checkpoints.float16.npy"
FUTURE_LOGITS = RAW / "qwen4b_teacher_future_full_vocabulary.float16.npy"
FUTURE_PROGRESS = RAW / "future_progress.json"
FUTURE_INDEX = OUT / "index/teacher_future_rows.jsonl"
TOKEN_FIELD = RAW / "qwen4b_multistep_representative_all_token.float16.npy"
TOKEN_INDEX = OUT / "index/representative_all_token_rows.jsonl"
CONTRIBUTIONS = OUT / "atlas/qwen4b_fixed_identity_output_contributions.float16.npy"
sys.path.insert(0, str(TESTS))

import phase1332_bf16_utils as model_base  # noqa: E402
import phase2297_c3161_c3260_qwen4b_ntp_predictive_field as field_base  # noqa: E402
import phase2309_c4321_c4440_multistep_future_contract as contract  # noqa: E402


PHASE = 2310
CAMPAIGN = "C4441-C4580"
EPS = 1e-12
REPRESENTATIVE_UNIT = 26


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    contract.write_rows(path, rows)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def checkpoint_modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def score_rows(model, device, rows: list[dict]) -> list[dict]:
    adapted = [{
        **row,
        "ntp_prompt_ids": row["future_prompt_ids"],
        "ntp_target_ids": row["future_target_ids"],
        "ntp_wrong_ids": row["future_wrong_ids"],
    } for row in rows]
    return field_base.sequence_scores(model, device, adapted, batch_size=12)


def slice_accuracy(rows: list[dict], by_id: dict[str, dict]) -> dict:
    values = [by_id[row["case_id"]] for row in rows]
    return {
        "rows": len(values),
        "mean_accuracy": float(np.mean([value["correct_by_mean"] for value in values])),
        "sum_accuracy": float(np.mean([value["correct_by_sum"] for value in values])),
        "mean_margin": float(np.mean([value["mean_margin"] for value in values])),
        "sum_margin": float(np.mean([value["sum_margin"] for value in values])),
    }


def sequence_ledger(rows: list[dict], scores: list[dict]) -> dict:
    by_id = {row["case_id"]: row for row in scores}
    families, qualified = {}, []
    for family in contract.FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        slices = {"overall:all": slice_accuracy(family_rows, by_id)}
        for kind, values in (
            ("language", contract.LANGUAGES),
            ("surface", contract.SURFACES),
            ("partition", contract.PARTITIONS),
            ("target_mention_order", ("first", "last")),
        ):
            for value in values:
                subset = [row for row in family_rows if row[kind] == value]
                slices[f"{kind}:{value}"] = slice_accuracy(subset, by_id)
        passed = all(
            min(value["mean_accuracy"], value["sum_accuracy"]) >= contract.BEHAVIOR_GATE
            for value in slices.values()
        )
        families[family] = {"qualified": passed, "slices": slices}
        if passed:
            qualified.append(family)
    return {
        "overall": slice_accuracy(rows, by_id),
        "families": families,
        "qualified_families": qualified,
        "gate": contract.BEHAVIOR_GATE,
    }


def left_pad_batch(sequences: list[list[int]], device, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(sequences):
        ids[i, width - len(row):] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, width - len(row):] = 1
    return ids, mask


def starts_with(values: list[int], prefix: list[int]) -> bool:
    return len(values) >= len(prefix) and values[:len(prefix)] == prefix


def normalized(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum())


def free_continuations(model, tokenizer, device, rows: list[dict], batch_size: int = 24) -> list[dict]:
    output: list[dict] = []
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    max_new = max(12, max(len(row["future_target_ids"]) for row in rows))
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        ids, mask = left_pad_batch([row["future_prompt_ids"] for row in batch], device, pad)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=max_new,
                use_cache=True,
                pad_token_id=pad,
                eos_token_id=model.config.eos_token_id,
            )
        continuations = generated[:, ids.shape[1]:].detach().cpu().tolist()
        for row, token_ids in zip(batch, continuations):
            target_ids = [int(value) for value in row["future_identity_target_ids"]]
            wrong_ids = [int(value) for value in row["future_identity_wrong_ids"]]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            clean = normalized(text)
            target_text, wrong_text = normalized(row["ntp_target_text"]), normalized(row["ntp_wrong_text"])
            target_pos, wrong_pos = clean.find(target_text), clean.find(wrong_text)
            first_identity_correct = target_pos >= 0 and (wrong_pos < 0 or target_pos < wrong_pos)
            output.append({
                "case_id": row["case_id"],
                "family": row["family"],
                "language": row["language"],
                "surface": row["surface"],
                "partition": row["partition"],
                "state": int(row["state"]),
                "target_mention_order": row["target_mention_order"],
                "generated_ids": [int(value) for value in token_ids],
                "generated_text": text,
                "identity_target_prefix_exact": starts_with(token_ids, target_ids),
                "identity_wrong_prefix_exact": starts_with(token_ids, wrong_ids),
                "future_target_prefix_exact": starts_with(token_ids, row["future_target_ids"]),
                "first_identity_correct": first_identity_correct,
            })
        print(f"[phase2310 free] {start + len(batch)}/{len(rows)}", flush=True)
    return output


def free_ledger(rows: list[dict]) -> dict:
    def summary(values: list[dict]) -> dict:
        return {
            "rows": len(values),
            "identity_target_prefix_exact": float(np.mean([
                row["identity_target_prefix_exact"] for row in values
            ])),
            "future_target_prefix_exact": float(np.mean([
                row["future_target_prefix_exact"] for row in values
            ])),
            "first_identity_accuracy": float(np.mean([
                row["first_identity_correct"] for row in values
            ])),
            "wrong_prefix_exact": float(np.mean([
                row["identity_wrong_prefix_exact"] for row in values
            ])),
        }
    families = {family: summary([row for row in rows if row["family"] == family])
                for family in contract.FAMILIES}
    return {
        "overall": summary(rows),
        "families": families,
        "route_eligible_families": [family for family, value in families.items()
                                    if value["first_identity_accuracy"] >= contract.FREE_IDENTITY_GATE],
        "gate": contract.FREE_IDENTITY_GATE,
        "future_exact_is_descriptive": True,
    }


def capture_boundary(model, device, rows: list[dict], batch_size: int = 10) -> dict:
    modules = checkpoint_modules(model)
    dimension, vocabulary = int(model.config.hidden_size), int(model.config.vocab_size)
    field_shape = (len(rows), len(modules), dimension)
    logit_shape = (len(rows), vocabulary)
    RAW.mkdir(parents=True, exist_ok=True)
    completed = 0
    if BOUNDARY.exists() and BOUNDARY_LOGITS.exists() and BOUNDARY_PROGRESS.exists():
        progress = json.loads(BOUNDARY_PROGRESS.read_text(encoding="utf-8"))
        if progress["field_shape"] != list(field_shape) or progress["logit_shape"] != list(logit_shape):
            raise RuntimeError(("boundary_resume_shape", progress, field_shape, logit_shape))
        completed = int(progress["completed"])
        field = np.lib.format.open_memmap(BOUNDARY, mode="r+")
        logits_file = np.lib.format.open_memmap(BOUNDARY_LOGITS, mode="r+")
    else:
        field = np.lib.format.open_memmap(BOUNDARY, mode="w+", dtype=np.float16, shape=field_shape)
        logits_file = np.lib.format.open_memmap(
            BOUNDARY_LOGITS, mode="w+", dtype=np.float16, shape=logit_shape
        )
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(modules):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, position_ids = field_base.pad_batch(
                    [row["future_prompt_ids"] for row in batch], device, pad
                )
                result = model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                               use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for q in range(len(modules)):
                    selected = torch.stack([captures[q][i, ends[i]] for i in range(len(batch))])
                    field[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                selected_logits = torch.stack([result.logits[i, ends[i]] for i in range(len(batch))])
                logits_file[start:start + len(batch)] = selected_logits.float().cpu().numpy().astype(np.float16)
                field.flush()
                logits_file.flush()
                save(BOUNDARY_PROGRESS, {
                    "completed": start + len(batch),
                    "field_shape": list(field_shape),
                    "logit_shape": list(logit_shape),
                })
                print(f"[phase2310 boundary] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        close_memmap(field)
        close_memmap(logits_file)
    return {
        "field_path": str(BOUNDARY.relative_to(ROOT)),
        "field_shape": list(field_shape),
        "logits_path": str(BOUNDARY_LOGITS.relative_to(ROOT)),
        "logits_shape": list(logit_shape),
        "all_checkpoints": True,
        "all_coordinates": True,
        "complete_vocabulary": True,
    }


def teacher_items(rows: list[dict]) -> list[dict]:
    items: list[dict] = []
    for source_index, row in enumerate(rows):
        steps = min(contract.FUTURE_STEPS, len(row["future_target_ids"]))
        for step in range(steps):
            items.append({
                "row": len(items),
                "source_index": source_index,
                "case_id": row["case_id"],
                "family": row["family"],
                "language": row["language"],
                "surface": row["surface"],
                "partition": row["partition"],
                "state": int(row["state"]),
                "unit": int(row["unit"]),
                "future_step": step,
                "next_target_token_id": int(row["future_target_ids"][step]),
                "input_ids": row["future_prompt_ids"] + row["future_target_ids"][:step],
            })
    return items


def capture_teacher_future(model, device, rows: list[dict], batch_size: int = 8) -> dict:
    items = teacher_items(rows)
    qpoints = tuple(contract.QPOINTS_4B)
    modules = checkpoint_modules(model)
    dimension, vocabulary = int(model.config.hidden_size), int(model.config.vocab_size)
    field_shape = (len(items), len(qpoints), dimension)
    logit_shape = (len(items), vocabulary)
    completed = 0
    if FUTURE_FIELD.exists() and FUTURE_LOGITS.exists() and FUTURE_PROGRESS.exists() and FUTURE_INDEX.exists():
        progress = json.loads(FUTURE_PROGRESS.read_text(encoding="utf-8"))
        if progress["field_shape"] != list(field_shape) or progress["logit_shape"] != list(logit_shape):
            raise RuntimeError(("future_resume_shape", progress, field_shape, logit_shape))
        completed = int(progress["completed"])
        field = np.lib.format.open_memmap(FUTURE_FIELD, mode="r+")
        logits_file = np.lib.format.open_memmap(FUTURE_LOGITS, mode="r+")
    else:
        field = np.lib.format.open_memmap(FUTURE_FIELD, mode="w+", dtype=np.float16,
                                          shape=field_shape)
        logits_file = np.lib.format.open_memmap(FUTURE_LOGITS, mode="w+", dtype=np.float16,
                                                shape=logit_shape)
        write_rows(FUTURE_INDEX, [{key: value for key, value in item.items() if key != "input_ids"}
                                  for item in items])
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(modules):
        if q not in qpoints:
            continue
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(completed, len(items), batch_size):
                batch = items[start:start + batch_size]
                ids, mask, position_ids = field_base.pad_batch(
                    [item["input_ids"] for item in batch], device, pad
                )
                result = model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                               use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for q_i, q in enumerate(qpoints):
                    selected = torch.stack([captures[q][i, ends[i]] for i in range(len(batch))])
                    field[start:start + len(batch), q_i] = selected.float().cpu().numpy().astype(np.float16)
                selected_logits = torch.stack([result.logits[i, ends[i]] for i in range(len(batch))])
                logits_file[start:start + len(batch)] = selected_logits.float().cpu().numpy().astype(np.float16)
                field.flush()
                logits_file.flush()
                save(FUTURE_PROGRESS, {
                    "completed": start + len(batch),
                    "field_shape": list(field_shape),
                    "logit_shape": list(logit_shape),
                })
                print(f"[phase2310 future] {start + len(batch)}/{len(items)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        close_memmap(field)
        close_memmap(logits_file)
    return {
        "field_path": str(FUTURE_FIELD.relative_to(ROOT)),
        "field_shape": list(field_shape),
        "logits_path": str(FUTURE_LOGITS.relative_to(ROOT)),
        "logits_shape": list(logit_shape),
        "index_path": str(FUTURE_INDEX.relative_to(ROOT)),
        "rows": len(items),
        "qpoints": list(qpoints),
        "all_coordinates": True,
        "complete_vocabulary_each_real_step": True,
    }


def future_probability_metrics(items: list[dict]) -> dict:
    path = OUT / "prediction/teacher_future_probability_metrics.jsonl"
    if path.exists():
        rows = read_rows(path)
        return {"path": str(path.relative_to(ROOT)), "rows": len(rows), "resumed": True}
    logits = np.load(FUTURE_LOGITS, mmap_mode="r")
    output: list[dict] = []
    try:
        for start in range(0, len(items), 32):
            batch = items[start:start + 32]
            values = torch.tensor(np.asarray(logits[start:start + len(batch)], dtype=np.float32))
            log_z = torch.logsumexp(values, dim=-1)
            probabilities = torch.softmax(values, dim=-1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + EPS), dim=-1)
            top = torch.argmax(values, dim=-1)
            for i, item in enumerate(batch):
                target = int(item["next_target_token_id"])
                target_logit = values[i, target]
                rank = int(torch.sum(values[i] > target_logit).item()) + 1
                output.append({
                    **{key: value for key, value in item.items() if key != "input_ids"},
                    "target_logprob": float((target_logit - log_z[i]).item()),
                    "target_probability": float(probabilities[i, target].item()),
                    "target_rank": rank,
                    "top_token_id": int(top[i].item()),
                    "target_is_top1": int(top[i].item()) == target,
                    "entropy_nats": float(entropy[i].item()),
                })
    finally:
        close_memmap(logits)
    write_rows(path, output)
    return {"path": str(path.relative_to(ROOT)), "rows": len(output), "resumed": False}


def representative_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["partition"] == "fresh_lockbox"
            and int(row["unit"]) == REPRESENTATIVE_UNIT]


def capture_representative_tokens(model, device, rows: list[dict]) -> dict:
    reps = representative_rows(rows)
    expected = len(contract.FAMILIES) * len(contract.LANGUAGES) * len(contract.SURFACES) * 2
    if len(reps) != expected:
        raise RuntimeError(("representative_count", len(reps), expected))
    if TOKEN_FIELD.exists() and TOKEN_INDEX.exists():
        values = np.load(TOKEN_FIELD, mmap_mode="r")
        shape = list(values.shape)
        close_memmap(values)
        return {"path": str(TOKEN_FIELD.relative_to(ROOT)), "shape": shape,
                "cases": len(reps), "row_index": str(TOKEN_INDEX.relative_to(ROOT)), "resumed": True}
    modules = checkpoint_modules(model)
    dimension = int(model.config.hidden_size)
    total = sum(len(row["future_prompt_ids"]) * len(modules) for row in reps)
    TOKEN_FIELD.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(TOKEN_FIELD, mode="w+", dtype=np.float16,
                                      shape=(total, dimension))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(modules):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    meta: list[dict] = []
    cursor = 0
    try:
        model.eval()
        with torch.inference_mode():
            for n, row in enumerate(reps):
                ids = torch.tensor([row["future_prompt_ids"]], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                position_ids = torch.arange(ids.shape[1], device=device).unsqueeze(0)
                model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                      use_cache=False, return_dict=True)
                for q in range(len(modules)):
                    values = captures[q][0, :ids.shape[1]].float().cpu().numpy().astype(np.float16)
                    field[cursor:cursor + len(values)] = values
                    for token in range(len(values)):
                        meta.append({
                            "row": cursor + token,
                            "case_id": row["case_id"],
                            "family": row["family"],
                            "language": row["language"],
                            "surface": row["surface"],
                            "state": int(row["state"]),
                            "unit": int(row["unit"]),
                            "checkpoint": q,
                            "token": token,
                            "token_id": int(row["future_prompt_ids"][token]),
                        })
                    cursor += len(values)
                print(f"[phase2310 all-token] {n + 1}/{len(reps)}", flush=True)
        field.flush()
    finally:
        for handle in handles:
            handle.remove()
        close_memmap(field)
    write_rows(TOKEN_INDEX, meta)
    return {"path": str(TOKEN_FIELD.relative_to(ROOT)), "shape": [total, dimension],
            "cases": len(reps), "row_index": str(TOKEN_INDEX.relative_to(ROOT)), "resumed": False}


def exact_fixed_contributions(model, rows: list[dict]) -> dict:
    if CONTRIBUTIONS.exists():
        values = np.load(CONTRIBUTIONS, mmap_mode="r")
        shape = list(values.shape)
        close_memmap(values)
        return {"path": str(CONTRIBUTIONS.relative_to(ROOT)), "shape": shape, "resumed": True}
    field = np.load(BOUNDARY, mmap_mode="r")
    logits = np.load(BOUNDARY_LOGITS, mmap_mode="r")
    dimension = int(model.config.hidden_size)
    CONTRIBUTIONS.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(CONTRIBUTIONS, mode="w+", dtype=np.float16,
                                       shape=(len(rows), dimension))
    weight = model.lm_head.weight
    max_error = 0.0
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), 32):
                batch = rows[start:start + 32]
                h = torch.tensor(np.asarray(field[start:start + len(batch), -1], dtype=np.float32),
                                 device=weight.device, dtype=weight.dtype)
                positive = torch.tensor([
                    row["future_identity_target_ids"][0] if int(row["state"]) == 1
                    else row["future_identity_wrong_ids"][0] for row in batch
                ], device=weight.device)
                negative = torch.tensor([
                    row["future_identity_wrong_ids"][0] if int(row["state"]) == 1
                    else row["future_identity_target_ids"][0] for row in batch
                ], device=weight.device)
                delta_w = weight.index_select(0, positive) - weight.index_select(0, negative)
                values = (h * delta_w).float()
                output[start:start + len(batch)] = values.cpu().numpy().astype(np.float16)
                direct = np.asarray(logits[start:start + len(batch)], dtype=np.float32)
                expected = (direct[np.arange(len(batch)), positive.cpu().numpy()]
                            - direct[np.arange(len(batch)), negative.cpu().numpy()])
                max_error = max(max_error, float(np.max(np.abs(
                    values.sum(-1).cpu().numpy() - expected
                ))))
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(field)
        close_memmap(logits)
    return {"path": str(CONTRIBUTIONS.relative_to(ROOT)), "shape": [len(rows), dimension],
            "decomposition_max_abs_error_float16": max_error, "resumed": False}


def lens_metrics(model, rows: list[dict]) -> dict:
    path = OUT / "prediction/fixed_identity_logit_lens.jsonl"
    if path.exists():
        values = read_rows(path)
        return {"path": str(path.relative_to(ROOT)), "rows": len(values), "resumed": True}
    field = np.load(BOUNDARY, mmap_mode="r")
    actual_file = np.load(BOUNDARY_LOGITS, mmap_mode="r")
    device = model.lm_head.weight.device
    output: list[dict] = []
    try:
        with torch.inference_mode():
            for q in contract.QPOINTS_4B:
                for start in range(0, len(rows), 8):
                    batch = rows[start:start + 8]
                    h = torch.tensor(np.asarray(field[start:start + len(batch), q], dtype=np.float32),
                                     device=device, dtype=model.lm_head.weight.dtype)
                    normalized = h if q == field.shape[1] - 1 else model.model.norm(h)
                    lens = model.lm_head(normalized).float()
                    actual = torch.tensor(
                        np.asarray(actual_file[start:start + len(batch)], dtype=np.float32), device=device
                    )
                    p, final_p = torch.softmax(lens, dim=-1), torch.softmax(actual, dim=-1)
                    midpoint = 0.5 * (p + final_p)
                    js = 0.5 * torch.sum(p * (torch.log(p + EPS) - torch.log(midpoint + EPS)), dim=-1)
                    js += 0.5 * torch.sum(final_p * (
                        torch.log(final_p + EPS) - torch.log(midpoint + EPS)
                    ), dim=-1)
                    for i, row in enumerate(batch):
                        positive = (row["future_identity_target_ids"][0] if int(row["state"]) == 1
                                    else row["future_identity_wrong_ids"][0])
                        negative = (row["future_identity_wrong_ids"][0] if int(row["state"]) == 1
                                    else row["future_identity_target_ids"][0])
                        output.append({
                            "case_id": row["case_id"],
                            "family": row["family"],
                            "language": row["language"],
                            "surface": row["surface"],
                            "partition": row["partition"],
                            "state": int(row["state"]),
                            "checkpoint": int(q),
                            "fixed_margin": float((lens[i, positive] - lens[i, negative]).item()),
                            "js_to_actual_final": float(js[i].item()),
                        })
                print(f"[phase2310 lens] q={q}", flush=True)
    finally:
        close_memmap(field)
        close_memmap(actual_file)
    write_rows(path, output)
    return {"path": str(path.relative_to(ROOT)), "rows": len(output), "resumed": False}


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact = {family: {
        "qualified": value["qualified"],
        "mean_accuracy": value["slices"]["overall:all"]["mean_accuracy"],
        "sum_accuracy": value["slices"]["overall:all"]["sum_accuracy"],
        "free_identity": result["free_ledger"]["families"][family]["first_identity_accuracy"],
    } for family, value in result["sequence_ledger"]["families"].items()}
    text = rf"""

## Phase {PHASE}: Qwen3-4B 八构式多步未来全坐标观察（{CAMPAIGN}） [{stamp}]

**测试原理与执行。** 严格读取 Phase2309 在模型加载前冻结的 2048 行材料，使用本地 Qwen3-4B、BF16、非量化 CUDA。每行完成三套互不混称的账本：完整正确/控制未来的 teacher-forced 总分与长度均值；最多 12 token 贪心自由续写及首个身份命中；给定正确未来前缀时每个真实未来步的下一 token 完整词表。所有构式无论行为是否合格都保存回答边界的 embedding、36 个 block 后状态、final norm 和全部 2560 坐标；每个 teacher-forced 未来步保存 10 个冻结检查点、全部坐标和完整 151936 词表；fresh-lockbox unit26 的 64 个代表条件保存每个真实输入 token、每个检查点、每个坐标。未读取 Attention 或 MLP 内部量。

$$
S(y_{{1:K}}\mid x)=\sum_{{k=1}}^K\log p(y_k\mid x,y_{{<k}}),
\qquad
H^{{(k)}}_{{i,q}}=H_q(x_i,y_{{i,<k}}).
$$

final norm 的固定有向身份对比逐坐标分账为：
$$
m_i^{{fix}}=z_{{a_i}}-z_{{b_i}}
=\sum_{{j=1}}^{{2560}}h_{{i,j}}(W_{{a_i,j}}-W_{{b_i,j}}),
$$
其中同一 unit 的两种事实状态共享同一正负 token 方向，避免正确答案交换造成读尺翻转。

**结果与门槛。** 八族汇总 `{json.dumps(compact, ensure_ascii=False)}`；完整序列严格合格族 `{result['sequence_ledger']['qualified_families']}`，自由身份路线合格族 `{result['free_ledger']['route_eligible_families']}`。整体完整序列 `{json.dumps(result['sequence_ledger']['overall'], ensure_ascii=False)}`；自由续写 `{json.dumps(result['free_ledger']['overall'], ensure_ascii=False)}`。边界场 `{json.dumps(result['boundary'], ensure_ascii=False)}`；多步未来场 `{json.dumps(result['teacher_future'], ensure_ascii=False)}`；未来概率账 `{json.dumps(result['future_probability'], ensure_ascii=False)}`；全 token 场 `{json.dumps(result['token_field'], ensure_ascii=False)}`；固定输出贡献 `{json.dumps(result['contributions'], ensure_ascii=False)}`；辅助 lens `{json.dumps(result['lens'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**分析、理论进展与严格边界。** `{result['strict_conclusion']}`。本阶段的科学进展是把“首 token 候选次序”扩展为逐真实未来步的完整概率轨迹，并让行为失败族也保留为负面图谱。它仍不能证明 HiddenState 是完整未来的充分统计量，不能把统一输出尾句的模式命名为语义齿轮，也不能由宽坐标输出贡献推出全息、流形或因果结构。teacher forcing 给出了条件正确前缀后的概率，不等于自由生成会走到该前缀；固定 identity margin 只覆盖首身份 token。材料没有独立人类盲评，taxonomy 是封闭微世界，Qwen3-4B 是单一小模型。脚本 `tests/glm5/phase2310_c4441_c4580_qwen4b_multistep_field.py`；结果 `tests/glm5/result/phase2310_c4441_c4580_qwen4b_multistep_field`。下一步只用 discovery/confirmation 建立基础状态、表面、语言和逐层传动账，再在 fresh 分区裁决；不运行高级数学路线。
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
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2309 contract is not authorized")
    rows = read_rows(ROWS_PATH)
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = model_base.load_bf16("qwen3")
        score_path = OUT / "behavior/sequence_scores.jsonl"
        scores = read_rows(score_path) if score_path.exists() else score_rows(model, device, rows)
        if not score_path.exists():
            write_rows(score_path, scores)
        sequence = sequence_ledger(rows, scores)
        save(OUT / "behavior/sequence_ledger.json", sequence)

        free_path = OUT / "behavior/free_continuations.jsonl"
        free_rows = (read_rows(free_path) if free_path.exists()
                     else free_continuations(model, tokenizer, device, rows))
        if not free_path.exists():
            write_rows(free_path, free_rows)
        free = free_ledger(free_rows)
        save(OUT / "behavior/free_ledger.json", free)

        boundary = capture_boundary(model, device, rows)
        teacher = capture_teacher_future(model, device, rows)
        items = teacher_items(rows)
        future_probability = future_probability_metrics(items)
        token_field = capture_representative_tokens(model, device, rows)
        contributions = exact_fixed_contributions(model, rows)
        lens = lens_metrics(model, rows)
        model_info = {
            "name": "Qwen3-4B",
            "precision": "bfloat16",
            "quantization": "none",
            "placement": placement,
            "layers": len(model.model.layers),
            "hidden_size": int(model.config.hidden_size),
            "vocabulary": int(model.config.vocab_size),
        }
    finally:
        if model is not None:
            model_base.release_bf16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    checks = {
        "contract_precedes_model_load": parent["config"]["frozen_before_model_load"],
        "all_rows_sequence_scored": len(scores) == len(rows),
        "all_rows_free_generated": len(free_rows) == len(rows),
        "all_families_observed": set(sequence["families"]) == set(contract.FAMILIES),
        "boundary_all_rows_checkpoints_coordinates": boundary["field_shape"] == [len(rows), 38, 2560],
        "boundary_complete_vocabulary": boundary["logits_shape"] == [len(rows), model_info["vocabulary"]],
        "future_all_real_steps": teacher["rows"] == sum(
            min(contract.FUTURE_STEPS, len(row["future_target_ids"])) for row in rows
        ),
        "future_all_coordinates": teacher["field_shape"][-2:] == [len(contract.QPOINTS_4B), 2560],
        "future_complete_vocabulary": teacher["logits_shape"][-1] == model_info["vocabulary"],
        "future_metrics_complete": future_probability["rows"] == teacher["rows"],
        "representative_all_tokens": token_field["cases"] == 64,
        "all_fixed_output_coordinates": contributions["shape"] == [len(rows), 2560],
        "all_lens_rows": lens["rows"] == len(rows) * len(contract.QPOINTS_4B),
        "no_attention_or_mlp_internal_read": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed",
        "material_rows": len(rows),
        "model": model_info,
        "sequence_ledger": sequence,
        "free_ledger": free,
        "boundary": boundary,
        "teacher_future": teacher,
        "future_probability": future_probability,
        "token_field": token_field,
        "contributions": contributions,
        "lens": lens,
        "hashes": {
            "rows": file_hash(ROWS_PATH),
            "scores": file_hash(score_path),
            "free": file_hash(free_path),
            "boundary": file_hash(BOUNDARY),
            "boundary_logits": file_hash(BOUNDARY_LOGITS),
            "future_field": file_hash(FUTURE_FIELD),
            "future_logits": file_hash(FUTURE_LOGITS),
            "token_field": file_hash(TOKEN_FIELD),
            "contributions": file_hash(CONTRIBUTIONS),
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            f"Qwen3-4B strictly qualified {len(sequence['qualified_families'])}/8 complete-future families "
            f"and {len(free['route_eligible_families'])}/8 free-identity families. Every family, including "
            "failures, has complete boundary and teacher-forced future observations; these are predictive "
            "state records, not sufficient-state, holographic, manifold, or causal-mechanism claims."
        ),
        "next_authorization": (
            "Run the frozen basic full-coordinate accounting and route tournament; authorize local perturbation "
            "only for prospectively qualified family/checkpoint cells."
        ),
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
