#!/usr/bin/env python3
"""Phase 977: content-conditioned legal-mode transfer across decode time.

The donor and receiver are both official Qwen3 prefixes.  ``no_think`` uses
``enable_thinking=False`` and ``thinking`` uses ``enable_thinking=True``.
At a layer frozen by ``phase977_span_causal_decomposition.py``, the donor's
post-block last-position residual is copied into the receiver either once at
prefill, for the first four forwards, or at every forward.  Both branches are
then advanced with the *same receiver-generated continuation* and separate KV
caches.  Thus every-step transfer is an online two-forward diagnosis, not a
fixed controller.

The post-block patch cannot rewrite the selected layer's already-created KV;
it only changes later layers.  Consequently even a successful result is called
post-layer state transfer, never full cache repair or original write-in.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log
from phase973_conditional_trajectory import get_eos_ids, summarize_logits
from phase976_qwen_mode_external import build_external_dataset
from phase977_dev_dataset import audit_dataset as audit_dev, build_dataset as build_dev
from phase977_legal_mode_trajectories import semantic_match as legal_semantic_match
from phase977_span_causal_decomposition import build_mode_manifest


PHASE = 977
MODEL = "qwen3"
OUT = Path("tests/glm5/result/phase977_cross_time_maintenance")
CAUSAL_RESULT = Path("tests/glm5/result/phase977_span_causal_decomposition/qwen3_result.json")
LEGAL_DISCOVERY_REAUDIT = Path(
    "tests/glm5/result/phase977_legal_mode_trajectories/reaudit_discovery.json")
LEGAL_TRAJECTORY_SCRIPT = Path(
    "tests/glm5/phase977_legal_mode_trajectories.py")
LEGAL_DEV_SUMMARY = Path(
    "tests/glm5/result/phase977_legal_mode_trajectories/summary_development.json")
def semantic_match(item: dict, text: str) -> bool:
    groups = item.get("alias_groups") or [[item["answer"]]]
    return legal_semantic_match(groups, text, bool(item.get("exact", False)))


def score_generated(tok, eos_ids, item, generated: list[int],
                    think_open_id: int, think_close_id: int) -> dict:
    eos_at = next((i for i, x in enumerate(generated) if int(x) in eos_ids), None)
    before_eos = generated if eos_at is None else generated[:eos_at]
    opens = [i for i, value in enumerate(before_eos) if int(value) == think_open_id]
    closes = [i for i, value in enumerate(before_eos) if int(value) == think_close_id]
    well_formed = len(opens) == 1 and len(closes) == 1 and opens[0] < closes[0]
    open_at = opens[0] if opens else None
    close_at = closes[0] if well_formed else None
    malformed = bool(opens or closes) and not well_formed
    if close_at is not None:
        final_ids = before_eos[close_at + 1:]
        thought_ids = before_eos[open_at + 1:close_at] if open_at is not None else []
    elif open_at is None:
        final_ids = before_eos
        thought_ids = []
    else:
        final_ids = []
        thought_ids = before_eos[open_at + 1:]
    final_text = tok.decode(final_ids, skip_special_tokens=True).strip()
    thought_text = tok.decode(thought_ids, skip_special_tokens=True).strip()
    semantic = semantic_match(item, final_text)
    empty_think = bool(open_at is not None and close_at is not None and not thought_text)
    behavioral_no_think = bool(open_at is None or empty_think)
    behavioral_thinking = bool(open_at is not None and close_at is not None and thought_text)
    return {
        "generated_ids": [int(x) for x in generated],
        "generated": tok.decode(generated, skip_special_tokens=False),
        "before_eos_text": tok.decode(before_eos, skip_special_tokens=False),
        "final_text": final_text,
        "thought_text": thought_text,
        "think_open_step": open_at,
        "think_close_step": close_at,
        "think_open_positions": opens,
        "think_close_positions": closes,
        "empty_think": empty_think,
        "malformed_mode": malformed,
        "behavioral_no_think": behavioral_no_think and not malformed,
        "behavioral_thinking": behavioral_thinking and not malformed,
        "semantic_match": semantic,
        "has_eos": eos_at is not None,
        "first_eos_step": eos_at,
        "early_eos": eos_at is not None and not semantic,
        "n_tokens": len(generated),
    }


def top_k_top_p_sample(logits: torch.Tensor, temperature: float, top_p: float,
                       top_k: int, generator: torch.Generator) -> int:
    scores = logits.float() / temperature
    if top_k > 0:
        cutoff = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(0, sorted_indices, sorted_scores)
    probs = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


def model_forward_capture(model, layer_module, input_ids, attention_mask, past=None):
    box = []

    def hook(module, args, output):
        y = output[0] if isinstance(output, tuple) else output
        box.append(y[:, -1, :].detach().clone())

    h = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        past_key_values=past, use_cache=True, return_dict=True)
    finally:
        h.remove()
    if len(box) != 1:
        raise RuntimeError(f"capture count={len(box)}")
    return out, box[0]


def model_forward_patch(model, layer_module, input_ids, attention_mask, past,
                        vector=None, add=None):
    calls = [0]

    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        y = output[0] if is_tuple else output
        z = y.clone()
        if vector is not None:
            z[:, -1, :] = vector.to(device=z.device, dtype=z.dtype)
        if add is not None:
            z[:, -1, :] += add.to(device=z.device, dtype=z.dtype)
        calls[0] += 1
        return (z,) + output[1:] if is_tuple else z

    h = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        past_key_values=past, use_cache=True, return_dict=True)
    finally:
        h.remove()
    if calls[0] != 1:
        raise RuntimeError(f"patch count={calls[0]}")
    return out


def model_forward_plain(model, input_ids, attention_mask, past=None):
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask,
                     past_key_values=past, use_cache=True, return_dict=True)


def sampling_params(target_mode: str):
    # Local Qwen3 README recommendations.  Thinking mode must not use greedy.
    if target_mode == "no_think":
        return 0.7, 0.8, 20
    return 0.6, 0.95, 20


def prefix_for(tok, item, mode: str):
    m = build_mode_manifest(tok, item["prompt"])
    return m["no_think_ids"] if mode == "no_think" else m["thinking_ids"]


def trajectory_seed(split: str, item_id: str, target_mode: str) -> int:
    """Common random stream for every condition with the same target mode."""
    raw = f"977|{split}|{item_id}|target={target_mode}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def control_seed(split: str, item_id: str, condition: str) -> int:
    raw = f"977|{split}|{item_id}|control={condition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def patch_at_step(schedule: str, step: int) -> bool:
    """Return whether the receiver is patched at this zero-based forward."""
    if schedule == "prefill_only":
        return step == 0
    if schedule == "first_decode_only":
        return step == 1
    if schedule == "step5_only":
        return step == 5
    if schedule == "first_four":
        return step < 4
    if schedule == "every_step":
        return True
    raise ValueError(f"unknown schedule: {schedule}")


def donor_needed_at_step(schedule: str, step: int) -> bool:
    """Advance the donor until the last scheduled receiver patch."""
    if step <= 0:
        return True
    if schedule == "prefill_only":
        return False
    if schedule == "first_decode_only":
        return step <= 1
    if schedule == "step5_only":
        return step <= 5
    if schedule == "first_four":
        return step < 4
    if schedule == "every_step":
        return True
    raise ValueError(f"unknown schedule: {schedule}")


def run_single(model, tok, layers, device, eos_ids, think_open_id, think_close_id,
               item, mode, condition, split,
               budget=512):
    prefix = prefix_for(tok, item, mode)
    x = torch.tensor([prefix], dtype=torch.long, device=device)
    mask = torch.ones_like(x)
    past = None
    generated = []
    landmarks = []
    temp, top_p, top_k = sampling_params(mode)
    gen = torch.Generator(device=device.type).manual_seed(
        trajectory_seed(split, item["id"], mode))
    next_input = x
    for step in range(budget):
        out = model_forward_plain(model, next_input, mask, past)
        logits = out.logits[0, -1]
        if step in (0, 1, 4, 255):
            sm = summarize_logits(out.logits, eos_ids)
            landmarks.append({"step": step, "gap": float(sm["gap"][0]),
                              "eos_rank": int(sm["eos_rank"][0])})
        nxt = top_k_top_p_sample(logits, temp, top_p, top_k, gen)
        generated.append(nxt)
        past = out.past_key_values
        if nxt in eos_ids:
            break
        next_input = torch.tensor([[nxt]], dtype=torch.long, device=device)
        mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=device)], 1)
    row = score_generated(tok, eos_ids, item, generated,
                          think_open_id, think_close_id)
    row.update({"condition": condition, "target_mode": mode,
                "prompt_len": len(prefix),
                "sampling_seed": trajectory_seed(split, item["id"], mode),
                "hit_256_without_eos": len(generated) >= 256 and not any(x in eos_ids for x in generated[:256]),
                "hit_512_without_eos": len(generated) >= 512 and not row["has_eos"],
                "landmarks": landmarks, "patch_calls": 0, "donor_calls": 0})
    requested = "behavioral_no_think" if mode == "no_think" else "behavioral_thinking"
    row["valid_eos"] = bool(row["semantic_match"] and row["has_eos"] and row[requested])
    return row


def run_transfer(model, tok, layers, device, eos_ids, think_open_id, think_close_id,
                 item, layer, *,
                 donor_mode, target_mode, schedule, condition, split,
                 donor_item=None, random_control=False, budget=512):
    donor_item = item if donor_item is None else donor_item
    donor_prefix = prefix_for(tok, donor_item, donor_mode)
    target_prefix = prefix_for(tok, item, target_mode)
    donor_x = torch.tensor([donor_prefix], dtype=torch.long, device=device)
    target_x = torch.tensor([target_prefix], dtype=torch.long, device=device)
    donor_mask = torch.ones_like(donor_x)
    target_mask = torch.ones_like(target_x)
    donor_past = None
    target_past = None
    generated = []
    patch_calls = 0
    donor_calls = 0
    landmarks = []
    layer_module = layers[layer]
    temp, top_p, top_k = sampling_params(target_mode)
    gen = torch.Generator(device=device.type).manual_seed(
        trajectory_seed(split, item["id"], target_mode))

    donor_out, donor_vec = model_forward_capture(
        model, layer_module, donor_x, donor_mask, donor_past)
    donor_calls += 1
    donor_past = donor_out.past_key_values
    if random_control:
        if not patch_at_step(schedule, 0):
            raise ValueError("random control is defined only for a prefill patch")
        target_base, target_vec = model_forward_capture(
            model, layer_module, target_x, target_mask, target_past)
        delta = donor_vec - target_vec
        rng = torch.Generator(device=device.type).manual_seed(
            control_seed(split, item["id"], condition + "|random"))
        random = torch.randn(delta.shape, generator=rng, device=device, dtype=torch.float32)
        random *= torch.linalg.vector_norm(delta.float()) / max(
            float(torch.linalg.vector_norm(random)), 1e-12)
        target_out = model_forward_patch(
            model, layer_module, target_x, target_mask, None,
            add=random.to(dtype=target_vec.dtype))
        patch_calls += 1
        del target_base
    elif patch_at_step(schedule, 0):
        target_out = model_forward_patch(
            model, layer_module, target_x, target_mask, None, vector=donor_vec)
        patch_calls += 1
    else:
        target_out = model_forward_plain(model, target_x, target_mask, None)
    if schedule == "prefill_only":
        donor_past = None
        del donor_out
    target_past = target_out.past_key_values
    logits = target_out.logits[0, -1]

    for step in range(budget):
        if step > 0:
            token = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
            target_mask = torch.cat(
                [target_mask, torch.ones((1, 1), dtype=target_mask.dtype, device=device)], 1)
            should_patch = patch_at_step(schedule, step)
            if donor_needed_at_step(schedule, step):
                donor_mask = torch.cat(
                    [donor_mask, torch.ones((1, 1), dtype=donor_mask.dtype, device=device)], 1)
                donor_out, donor_vec = model_forward_capture(
                    model, layer_module, token, donor_mask, donor_past)
                donor_calls += 1
                donor_past = donor_out.past_key_values
            if should_patch:
                target_out = model_forward_patch(
                    model, layer_module, token, target_mask, target_past,
                    vector=donor_vec)
                patch_calls += 1
            else:
                target_out = model_forward_plain(model, token, target_mask, target_past)
            target_past = target_out.past_key_values
            logits = target_out.logits[0, -1]
        if step in (0, 1, 4, 5, 255):
            sm = summarize_logits(target_out.logits, eos_ids)
            landmarks.append({"step": step, "gap": float(sm["gap"][0]),
                              "eos_rank": int(sm["eos_rank"][0])})
        nxt = top_k_top_p_sample(logits, temp, top_p, top_k, gen)
        generated.append(nxt)
        if nxt in eos_ids:
            break

    expected_patch_calls = sum(patch_at_step(schedule, step)
                               for step in range(len(generated)))
    if patch_calls != expected_patch_calls:
        raise RuntimeError(f"patch schedule mismatch {patch_calls}!={expected_patch_calls}")
    expected_donor_calls = 1 + sum(
        donor_needed_at_step(schedule, step) for step in range(1, len(generated))
    )
    if donor_calls != expected_donor_calls:
        raise RuntimeError(f"donor schedule mismatch {donor_calls}!={expected_donor_calls}")
    row = score_generated(tok, eos_ids, item, generated,
                          think_open_id, think_close_id)
    row.update({
        "condition": condition, "donor_mode": donor_mode, "target_mode": target_mode,
        "donor_id": donor_item["id"], "schedule": schedule, "layer": layer,
        "prompt_len": len(target_prefix), "donor_prompt_len": len(donor_prefix),
        "sampling_seed": trajectory_seed(split, item["id"], target_mode),
        "control_seed": (control_seed(split, item["id"], condition + "|random")
                         if random_control else None),
        "hit_256_without_eos": len(generated) >= 256 and not any(x in eos_ids for x in generated[:256]),
        "hit_512_without_eos": len(generated) >= 512 and not row["has_eos"],
        "landmarks": landmarks, "patch_calls": patch_calls, "donor_calls": donor_calls,
        "post_block_cache_warning": "selected layer KV is not rewritten; only later layers see patch",
    })
    requested = "behavioral_no_think" if donor_mode == "no_think" else "behavioral_thinking"
    row["transfer_mode_success"] = bool(row[requested])
    row["valid_eos"] = bool(row["semantic_match"] and row["has_eos"]
                            and row["transfer_mode_success"] and not row["malformed_mode"])
    return row


def summarize(rows):
    if not rows:
        return {"n": 0}
    out = {
        "n": len(rows),
        "semantic_rate": float(np.mean([r["semantic_match"] for r in rows])),
        "eos_rate": float(np.mean([r["has_eos"] for r in rows])),
        "valid_eos_rate": float(np.mean([r["valid_eos"] for r in rows])),
        "early_eos_rate": float(np.mean([r["early_eos"] for r in rows])),
        "malformed_rate": float(np.mean([r["malformed_mode"] for r in rows])),
        "behavioral_no_think_rate": float(np.mean([r["behavioral_no_think"] for r in rows])),
        "behavioral_thinking_rate": float(np.mean([r["behavioral_thinking"] for r in rows])),
        "hit_256_rate": float(np.mean([r["hit_256_without_eos"] for r in rows])),
        "hit_512_rate": float(np.mean([r["hit_512_without_eos"] for r in rows])),
        "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])),
    }
    by_task = {}
    for task in sorted({r["task"] for r in rows}):
        vals = [r for r in rows if r["task"] == task]
        by_task[task] = {"n": len(vals), "valid_eos_n": sum(r["valid_eos"] for r in vals),
                         "semantic_n": sum(r["semantic_match"] for r in vals)}
    out["by_task"] = by_task
    return out


def conditions_for_split(split):
    common = [
        "hard_no_think_clean",
        "hard_thinking_clean",
        "paired_prefill_only",
        "shuffled_prefill_only",
        "random_prefill_only",
        "reverse_prefill_only",
    ]
    return common + ([
        "paired_first_decode_only",
        "paired_step5_only",
        "paired_first_four",
        "paired_every_step",
    ] if split == "dev" else [])


def run_split(split: str):
    if split not in ("dev", "holdout"):
        raise ValueError(split)
    if not LEGAL_DISCOVERY_REAUDIT.exists():
        raise RuntimeError("strict-v2 discovery legal re-audit is absent; cross-time scan stays closed")
    legal_reaudit_bytes = LEGAL_DISCOVERY_REAUDIT.read_bytes()
    legal_reaudit = json.loads(legal_reaudit_bytes.decode("utf-8"))
    legal_reaudit_sha256 = hashlib.sha256(legal_reaudit_bytes).hexdigest()
    discovery_gate = legal_reaudit.get("downstream_gate", {})
    if not (
        legal_reaudit.get("phase") == PHASE
        and legal_reaudit.get("split") == "discovery"
        and legal_reaudit.get("migration", {}).get("target_schema_version") == 2
        and legal_reaudit.get("migration", {}).get("target_parser_version")
        == "strict_final_region_v2"
        and legal_reaudit.get("target_identity", {}).get("legal_script_sha256")
        == hashlib.sha256(LEGAL_TRAJECTORY_SCRIPT.read_bytes()).hexdigest()
        and legal_reaudit.get("strict_v2_summary_recomputed", {}).get("complete") is True
        and legal_reaudit.get("strict_v2_summary_recomputed", {}).get(
            "decision_gate", {}).get("passed") is True
        and discovery_gate.get("no_go") is False
        and discovery_gate.get("status") == "GO"
        and discovery_gate.get("strict_v2_gate_passed") is True
    ):
        raise RuntimeError(
            "strict-v2 discovery legal gate is NO-GO; cross-time scan remains closed")
    ensure_dir(OUT)
    legal_summary = json.loads(LEGAL_DEV_SUMMARY.read_text(encoding="utf-8"))
    legal_gate_passed = bool(legal_summary.get("decision_gate", {}).get("passed", False))
    causal = json.loads(CAUSAL_RESULT.read_text(encoding="utf-8"))
    frozen = causal["frozen_candidates"]
    legal_scan_passed = bool(causal.get("legal_mode_layer_scan", {}).get(
        "selection_passed", False))
    dev_gates = causal.get("development", {}).get("gates", {})
    upstream_passed = bool(
        legal_scan_passed
        and legal_gate_passed
        and causal.get("legal_discovery_reaudit_sha256") == legal_reaudit_sha256
        and dev_gates.get("legal_current_step", False)
        and frozen.get("all_dev_gates_passed", False)
    )
    if not upstream_passed:
        result = {
            "phase": PHASE,
            "split": split,
            "not_run": True,
            "reason": "frozen current-step causal prerequisites failed",
            "upstream": {
                "legal_scan_passed": legal_scan_passed,
                "legal_trajectory_gate_passed": legal_gate_passed,
                "development_gates": dev_gates,
                "all_dev_gates_passed": frozen.get("all_dev_gates_passed", False),
            },
        }
        (OUT / f"{split}_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    layer = int(frozen["legal_mode_arrival_layer"])
    dev_prompts = [x["prompt"] for x in build_dev()]
    discovery_prompts = [x["prompt"] for x in build_external_dataset()]
    if split == "dev":
        items = build_dev()
        audit = audit_dev(previous_prompts=discovery_prompts)
    else:
        dev_path = OUT / "dev_result.json"
        if not dev_path.exists():
            raise RuntimeError("dev result absent; holdout remains closed")
        dev_result = json.loads(dev_path.read_text(encoding="utf-8"))
        if not dev_result.get("expansion_gate", {}).get("prefill_only_passed", False):
            result = {"phase": PHASE, "split": "holdout", "not_run": True,
                      "reason": "prefill-only development gate failed"}
            (OUT / "holdout_result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        holdout_module = importlib.import_module("phase977_holdout_dataset")
        items = holdout_module.build_dataset()
        audit = holdout_module.audit_dataset(
            previous_prompts=discovery_prompts + dev_prompts)
    audit_passed = bool(audit.get("passed", audit.get("ok", False)))
    if not audit_passed:
        raise RuntimeError(f"{split} dataset audit failed: {audit}")

    t0 = time.time()
    model, tok, device = load_model(MODEL)
    layers = get_layers(model)
    eos_ids = get_eos_ids(model, tok)
    token_probe = build_mode_manifest(tok, items[0]["prompt"])["suffix_ids"]
    think_open_id, think_close_id = int(token_probe[0]), int(token_probe[2])
    if layer >= len(layers):
        raise RuntimeError("frozen legal-mode layer absent")
    conditions = conditions_for_split(split)
    rows = []
    path = OUT / f"{split}_result.json"
    result = {
        "phase": PHASE, "model": MODEL, "split": split, "n_items": len(items),
        "legal_discovery_reaudit_sha256": legal_reaudit_sha256,
        "frozen_layer": layer, "layer_region": "early" if layer <= 11 else
            "middle" if layer <= 23 else "late",
        "conditions": conditions, "dataset_audit": audit,
        "upstream": {
            "causal_result_sha256": hashlib.sha256(CAUSAL_RESULT.read_bytes()).hexdigest(),
            "legal_development_summary_sha256": hashlib.sha256(
                LEGAL_DEV_SUMMARY.read_bytes()).hexdigest(),
            "legal_scan_passed": legal_scan_passed,
            "legal_trajectory_gate_passed": legal_gate_passed,
            "development_gates": dev_gates,
            "all_dev_gates_passed": frozen.get("all_dev_gates_passed", False),
        },
        "method_warning": "paired schedules are content-conditioned legal-state transfers; every-step uses a live donor",
        "rows": [], "summary": {},
    }
    for ci, condition in enumerate(conditions):
        for i, item in enumerate(items):
            other = items[(i + 1) % len(items)]
            if condition == "hard_no_think_clean":
                row = run_single(model, tok, layers, device, eos_ids,
                                 think_open_id, think_close_id, item,
                                 "no_think", condition, split)
            elif condition == "hard_thinking_clean":
                row = run_single(model, tok, layers, device, eos_ids,
                                 think_open_id, think_close_id, item,
                                 "thinking", condition, split)
            elif condition == "paired_prefill_only":
                row = run_transfer(model, tok, layers, device, eos_ids,
                                   think_open_id, think_close_id, item, layer,
                                   donor_mode="no_think", target_mode="thinking",
                                   schedule="prefill_only", condition=condition, split=split)
            elif condition == "paired_first_four":
                row = run_transfer(model, tok, layers, device, eos_ids,
                                   think_open_id, think_close_id, item, layer,
                                   donor_mode="no_think", target_mode="thinking",
                                   schedule="first_four", condition=condition, split=split)
            elif condition == "paired_first_decode_only":
                row = run_transfer(model, tok, layers, device, eos_ids,
                                   think_open_id, think_close_id, item, layer,
                                   donor_mode="no_think", target_mode="thinking",
                                   schedule="first_decode_only", condition=condition, split=split)
            elif condition == "paired_step5_only":
                row = run_transfer(model, tok, layers, device, eos_ids,
                                   think_open_id, think_close_id, item, layer,
                                   donor_mode="no_think", target_mode="thinking",
                                   schedule="step5_only", condition=condition, split=split)
            elif condition == "paired_every_step":
                row = run_transfer(model, tok, layers, device, eos_ids,
                                   think_open_id, think_close_id, item, layer,
                                   donor_mode="no_think", target_mode="thinking",
                                   schedule="every_step", condition=condition, split=split)
            elif condition == "shuffled_prefill_only":
                row = run_transfer(model, tok, layers, device, eos_ids,
                                   think_open_id, think_close_id, item, layer,
                                   donor_mode="no_think", target_mode="thinking",
                                   schedule="prefill_only", condition=condition, split=split,
                                   donor_item=other)
            elif condition == "random_prefill_only":
                row = run_transfer(model, tok, layers, device, eos_ids,
                                   think_open_id, think_close_id, item, layer,
                                   donor_mode="no_think", target_mode="thinking",
                                   schedule="prefill_only", condition=condition, split=split,
                                   random_control=True)
            elif condition == "reverse_prefill_only":
                row = run_transfer(model, tok, layers, device, eos_ids,
                                   think_open_id, think_close_id, item, layer,
                                   donor_mode="thinking", target_mode="no_think",
                                   schedule="prefill_only", condition=condition, split=split)
            else:
                raise ValueError(condition)
            rows.append({"id": item["id"], "task": item["task"], **row})
            if (i + 1) % 8 == 0:
                log(f"  Phase977 cross-time {split}/{condition} {i+1}/{len(items)}")
        result["rows"] = rows
        for c in conditions[:ci + 1]:
            result["summary"][c] = summarize([r for r in rows if r["condition"] == c])
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    nt = result["summary"]["hard_no_think_clean"]
    th = result["summary"]["hard_thinking_clean"]
    paired = result["summary"]["paired_prefill_only"]
    random = result["summary"]["random_prefill_only"]
    shuffled = result["summary"]["shuffled_prefill_only"]
    reverse = result["summary"]["reverse_prefill_only"]
    if split == "dev":
        per_task_ok = all(v["valid_eos_n"] >= 5 for v in paired["by_task"].values())
        reverse_ok = bool(
            reverse["valid_eos_rate"] >= max(29 / 64, th["valid_eos_rate"] - 3 / 64)
            and reverse["semantic_rate"] >= th["semantic_rate"] - 3 / 64
            and reverse["early_eos_rate"] <= 3 / 64
            and reverse["malformed_rate"] <= 3 / 64)
        shuffled_specificity = bool(
            paired["valid_eos_rate"] - shuffled["valid_eos_rate"] >= 13 / 64)
        gate = {
            "prefill_only_passed": bool(
                nt["valid_eos_rate"] >= 48 / 64
                and th["valid_eos_rate"] >= 32 / 64
                and paired["valid_eos_rate"] >= max(45 / 64, nt["valid_eos_rate"] - 3 / 64)
                and paired["semantic_rate"] >= nt["semantic_rate"] - 3 / 64
                and paired["early_eos_rate"] <= 3 / 64
                and paired["malformed_rate"] <= 3 / 64
                and random["valid_eos_rate"] <= 13 / 64
                and paired["valid_eos_rate"] - random["valid_eos_rate"] >= 0.50
                and reverse_ok
                and shuffled_specificity
                and per_task_ok),
            "every_step_only": False,
            "first_decode_only_passed": False,
            "per_task_prefill_ok": per_task_ok,
            "reverse_passed": reverse_ok,
            "paired_over_shuffled_passed": shuffled_specificity,
            "official_endpoint_measurable": bool(nt["valid_eos_rate"] >= 48 / 64
                                                 and th["valid_eos_rate"] >= 32 / 64),
        }
        every = result["summary"]["paired_every_step"]
        first_decode = result["summary"]["paired_first_decode_only"]
        gate["first_decode_only_passed"] = bool(
            first_decode["valid_eos_rate"] >= 45 / 64
            and first_decode["semantic_rate"] >= nt["semantic_rate"] - 3 / 64
            and first_decode["early_eos_rate"] <= 3 / 64
            and first_decode["malformed_rate"] <= 3 / 64)
        gate["every_step_only"] = bool(not gate["prefill_only_passed"]
                                       and every["valid_eos_rate"] >= 45 / 64
                                       and every["semantic_rate"] >= nt["semantic_rate"] - 3 / 64)
        result["expansion_gate"] = gate
    else:
        paired_success = int(round(paired["valid_eos_rate"] * len(items)))
        random_success = int(round(random["valid_eos_rate"] * len(items)))
        shuffled_success = int(round(shuffled["valid_eos_rate"] * len(items)))
        reverse_success = int(round(reverse["valid_eos_rate"] * len(items)))
        per_task_ok = all(v["valid_eos_n"] >= 10 for v in paired["by_task"].values())
        result["holdout_gate"] = {
            "passed": bool(
                nt["valid_eos_rate"] >= 115 / 128
                and th["valid_eos_rate"] >= 96 / 128
                and th["hit_512_rate"] <= 26 / 128
                and paired_success >= 90
                and paired["semantic_rate"] >= nt["semantic_rate"] - 6 / 128
                and paired["early_eos_rate"] <= 6 / 128
                and paired["malformed_rate"] <= 6 / 128
                and random_success <= 26
                and paired_success - random_success >= 64
                and paired_success - shuffled_success >= 26
                and reverse_success >= 58
                and per_task_ok),
            "paired_success_n": paired_success,
            "random_success_n": random_success,
            "shuffled_success_n": shuffled_success,
            "reverse_success_n": reverse_success,
            "per_task_prefill_ok": per_task_ok,
            "eligible_for_phase978": False,
        }
        result["holdout_gate"]["eligible_for_phase978"] = bool(
            result["holdout_gate"]["passed"] and layer <= 23)
    result["elapsed_seconds"] = time.time() - t0
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Saved {path}; elapsed={result['elapsed_seconds']/60:.1f} min")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "holdout"], required=True)
    args = p.parse_args()
    run_split(args.split)


if __name__ == "__main__":
    main()
