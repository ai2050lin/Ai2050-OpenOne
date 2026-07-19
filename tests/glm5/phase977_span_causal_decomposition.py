#!/usr/bin/env python3
"""Phase 977: Qwen3 mode-span decomposition and legal-mode layer diagnosis.

This script keeps two kinds of evidence separate:

1. ``enable_thinking=True/False`` prefixes are the only official hard-mode
   templates.  A bidirectional last-position residual scan asks when their
   already-formed states are sufficient to change the *next* token.  It does
   not identify where the mode was originally written.
2. The four-token hard no-think suffix is decomposed by fixed-position
   zero/newline embedding interventions.  The token sequence remains the
   official template, but the embeddings are out-of-distribution causal
   interventions; the subspans are not called independent legal templates.

Discovery reuses the 80 Phase-976 items only for selection.  All selected
groups and layers are then checked once on the new 64-item Phase-977 dev set.
No holdout item is imported or read by this script.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
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
from phase975_protocol_causal_transfer import neutral_id, token_ids
from phase976_qwen_mode_external import build_external_dataset
from phase977_dev_dataset import audit_dataset, build_dataset as build_dev_dataset


PHASE = 977
MODEL = "qwen3"
OUT = Path("tests/glm5/result/phase977_span_causal_decomposition")
LEGAL_DISCOVERY_REAUDIT = Path(
    "tests/glm5/result/phase977_legal_mode_trajectories/reaudit_discovery.json")
LEGAL_TRAJECTORY_SCRIPT = Path(
    "tests/glm5/phase977_legal_mode_trajectories.py")
LEGAL_DEV_SUMMARY = Path(
    "tests/glm5/result/phase977_legal_mode_trajectories/summary_development.json")
NEUTRAL_TEXT = "\n"

STATE_NAMES = [
    "answer_incomplete",
    "answer_complete",
    "answer_period",
    "answer_comma_requires_continue",
    "continuation_incomplete",
    "continuation_complete",
    "wrong_answer",
    "wrong_answer_period",
]

ATOMIC_GROUPS = ["open_tag", "inner_blank", "close_tag", "answer_separator"]
COMPOSITE_GROUPS = ["open_half", "close_half", "full_mode_block"]


def official_prefix(tok, prompt: str, thinking: bool) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking,
    )


def build_mode_manifest(tok, prompt: str) -> dict:
    thinking_text = official_prefix(tok, prompt, True)
    no_think_text = official_prefix(tok, prompt, False)
    thinking_ids = token_ids(tok, thinking_text, add_special_tokens=False)
    no_think_ids = token_ids(tok, no_think_text, add_special_tokens=False)
    suffix_ids = token_ids(tok, "<think>\n\n</think>\n\n", add_special_tokens=False)
    if len(suffix_ids) != 4:
        raise RuntimeError(f"expected four no-think suffix tokens, got {suffix_ids}")
    if no_think_ids != thinking_ids + suffix_ids:
        raise RuntimeError("official Qwen templates do not differ by the audited suffix")
    start = len(thinking_ids)
    groups = {
        "open_tag": [start],
        "inner_blank": [start + 1],
        "close_tag": [start + 2],
        "answer_separator": [start + 3],
        "open_half": [start, start + 1],
        "close_half": [start + 2, start + 3],
        "full_mode_block": list(range(start, start + 4)),
    }
    if sorted(groups["open_half"] + groups["close_half"]) != groups["full_mode_block"]:
        raise RuntimeError("mode groups do not cover the official suffix")
    user_header = token_ids(tok, "<|im_start|>user\n", add_special_tokens=False)
    im_end = token_ids(tok, "<|im_end|>", add_special_tokens=False)
    if not user_header or len(im_end) != 1:
        raise RuntimeError("Qwen user-boundary tokens are not auditable")
    header_at = next((i for i in range(len(thinking_ids) - len(user_header) + 1)
                      if thinking_ids[i:i + len(user_header)] == user_header), None)
    if header_at is None:
        raise RuntimeError("user header absent from official template")
    content_start = header_at + len(user_header)
    content_end = next((i for i in range(content_start, len(thinking_ids))
                        if thinking_ids[i] == im_end[0]), None)
    if content_end is None or content_end <= content_start:
        raise RuntimeError("user content span absent from official template")
    user_content_positions = list(range(content_start, content_end))
    return {
        "thinking_text": thinking_text,
        "no_think_text": no_think_text,
        "thinking_ids": thinking_ids,
        "no_think_ids": no_think_ids,
        "suffix_ids": suffix_ids,
        "suffix_tokens": tok.convert_ids_to_tokens(suffix_ids),
        "mode_start": start,
        "groups": groups,
        "user_content_positions": user_content_positions,
    }


def state_texts(item: dict) -> dict[str, str]:
    answer = item["answer"].strip().rstrip(".!?;:")
    words = answer.split()
    partial = " ".join(words[: max(1, len(words) // 2)]) if len(words) > 1 else ""
    return {
        "answer_incomplete": partial,
        "answer_complete": answer,
        "answer_period": answer + ".",
        "answer_comma_requires_continue": answer + ",",
        "continuation_incomplete": answer + ". Because",
        "continuation_complete": answer + ". This completes the response.",
        "wrong_answer": "incorrect",
        "wrong_answer_period": "incorrect.",
    }


def state_sequences(tok, manifest: dict, item: dict) -> dict[str, list[int] | None]:
    """Build exact teacher-forced states; never substitute an empty answer prefix."""
    prefix = manifest["no_think_ids"]
    texts = state_texts(item)
    out: dict[str, list[int] | None] = {}
    answer = item["answer"].strip().rstrip(".!?;:")
    complete_ids = token_ids(
        tok, manifest["no_think_text"] + answer, add_special_tokens=False)
    if complete_ids[:len(prefix)] != prefix:
        raise RuntimeError(f"prefix retokenized for {item['id']}/answer")
    # A strict token prefix exists only when the answer contributes >=2 tokens.
    out["answer_incomplete"] = (
        complete_ids[:-1] if len(complete_ids) - len(prefix) >= 2 else None)
    for state, content in texts.items():
        if state == "answer_incomplete":
            continue
        ids = token_ids(tok, manifest["no_think_text"] + content,
                        add_special_tokens=False)
        if ids[:len(prefix)] != prefix:
            raise RuntimeError(f"prefix retokenized for {item['id']}/{state}")
        out[state] = ids
    return out


def content_control_positions(manifest: dict, width: int) -> list[int]:
    """Frozen same-width ordinary user-content span, centered in the prompt."""
    positions = manifest["user_content_positions"]
    if width <= 0 or len(positions) < width:
        raise RuntimeError("prompt is too short for the same-width content control")
    start = (len(positions) - width) // 2
    return positions[start:start + width]


def _replace_positions(output, positions, vector=None, add=None, counter=None):
    is_tuple = isinstance(output, tuple)
    y = output[0] if is_tuple else output
    z = y.clone()
    pos = [p if p >= 0 else z.shape[1] + p for p in positions]
    if vector is not None:
        v = vector.to(device=z.device, dtype=z.dtype)
        if v.ndim == 1:
            if len(pos) != 1:
                raise RuntimeError("one vector supplied for multiple positions")
            z[:, pos[0], :] = v
        else:
            if v.shape[0] != len(pos):
                raise RuntimeError("patch vector/position count mismatch")
            z[:, pos, :] = v
    if add is not None:
        v = add.to(device=z.device, dtype=z.dtype)
        if v.ndim == 1:
            z[:, pos[0], :] += v
        else:
            z[:, pos, :] += v
    if counter is not None:
        counter[0] += 1
    return (z,) + output[1:] if is_tuple else z


def forward_snapshot(model, layers, device, eos_ids, ids, *,
                     capture_depths=None, capture_positions=None,
                     embed_positions=None, embed_mode=None, neutral_token=None,
                     patch_layer=None, patch_positions=None, patch_vector=None,
                     add_vector=None, mode_token_id=None) -> dict:
    """Batch-one forward at exact post-block tensor boundaries."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(x)
    handles = []
    captured = {}
    capture_counts = defaultdict(int)
    embed_count = [0]
    patch_count = [0]
    wanted = [] if capture_depths is None else sorted(set(int(d) for d in capture_depths))
    cap_pos = [-1] if capture_positions is None else list(capture_positions)
    try:
        emb = model.get_input_embeddings()
        if embed_positions:
            neutral = None if embed_mode == "zero" else emb.weight[int(neutral_token)].detach()

            def corrupt_hook(module, args, output):
                z = output.clone()
                valid = [p for p in embed_positions if 0 <= p < z.shape[1]]
                if len(valid) != len(embed_positions):
                    raise RuntimeError("embedding position outside prefix")
                if embed_mode == "zero":
                    z[:, valid, :] = 0
                elif embed_mode == "neutral":
                    z[:, valid, :] = neutral.to(device=z.device, dtype=z.dtype)
                else:
                    raise ValueError(embed_mode)
                embed_count[0] += 1
                return z

            handles.append(emb.register_forward_hook(corrupt_hook))

        def capture_tensor(depth, output):
            y = output[0] if isinstance(output, tuple) else output
            pos = [p if p >= 0 else y.shape[1] + p for p in cap_pos]
            value = y[0, pos, :].detach().to("cpu", torch.float32)
            captured[depth] = value[0] if len(pos) == 1 else value
            capture_counts[depth] += 1

        if 0 in wanted:
            handles.append(emb.register_forward_hook(lambda m, a, o: capture_tensor(0, o)))
        for depth in [d for d in wanted if d > 0]:
            if depth > len(layers):
                raise ValueError(depth)

            def make_capture(d):
                return lambda m, a, o: capture_tensor(d, o)

            handles.append(layers[depth - 1].register_forward_hook(make_capture(depth)))

        if patch_layer is not None:
            positions = [-1] if patch_positions is None else list(patch_positions)

            def patch_hook(module, args, output):
                return _replace_positions(output, positions, vector=patch_vector,
                                          add=add_vector, counter=patch_count)

            handles.append(layers[int(patch_layer)].register_forward_hook(patch_hook))

        with torch.no_grad():
            out = model(input_ids=x, attention_mask=mask, use_cache=False,
                        output_hidden_states=False, return_dict=True)
        sm = summarize_logits(out.logits, eos_ids)
        result = {
            "gap": float(sm["gap"][0]),
            "eos_rank": int(sm["eos_rank"][0]),
            "eos_won": bool(sm["gap"][0] < 0),
            "greedy_id": int(out.logits[0, -1].argmax().item()),
        }
        if mode_token_id is not None:
            logits = out.logits[0, -1].float()
            mode_id = int(mode_token_id)
            competitors = logits.clone()
            competitors[mode_id] = -torch.inf
            competitor_id = int(competitors.argmax().item())
            result.update({
                "mode_token_id": mode_id,
                "mode_margin": float((logits[mode_id] - competitors[competitor_id]).item()),
                "mode_competitor_id": competitor_id,
            })
        if wanted:
            missing = [d for d in wanted if capture_counts[d] != 1]
            if missing:
                raise RuntimeError(f"capture hook failure: {missing}")
            result["vectors"] = captured
        if embed_positions and embed_count[0] != 1:
            raise RuntimeError(f"embedding hook count={embed_count[0]}")
        if patch_layer is not None and patch_count[0] != 1:
            raise RuntimeError(f"patch hook count={patch_count[0]}")
        return result
    finally:
        for h in reversed(handles):
            h.remove()


def movement(source: dict, target: dict, patched: dict) -> dict:
    before = abs(target["gap"] - source["gap"])
    after = abs(patched["gap"] - source["gap"])
    return {
        "source_gap": source["gap"],
        "target_gap": target["gap"],
        "patched_gap": patched["gap"],
        "delta_gap": patched["gap"] - target["gap"],
        "eos_won": patched["eos_won"],
        "toward_source": after < before,
        "distance_before": before,
        "distance_after": after,
        "recovery_fraction": 1 - after / max(before, 1e-8),
    }


def mode_movement(source: dict, target: dict, patched: dict) -> dict:
    before = abs(target["mode_margin"] - source["mode_margin"])
    after = abs(patched["mode_margin"] - source["mode_margin"])
    return {
        "source_mode_margin": source["mode_margin"],
        "target_mode_margin": target["mode_margin"],
        "patched_mode_margin": patched["mode_margin"],
        "delta_mode_margin": patched["mode_margin"] - target["mode_margin"],
        "mode_toward_source": after < before,
        "mode_distance_before": before,
        "mode_distance_after": after,
        "mode_recovery_fraction": 1 - after / max(before, 1e-8),
    }


def summarize_delta(rows: list[dict], condition: str, state: str) -> dict:
    vals = [r for r in rows if r["condition"] == condition and r["state"] == state]
    if not vals:
        return {"n": 0, "mean_delta_gap": None,
                "positive_rate": None, "eos_win_rate": None}
    return {
        "n": len(vals),
        "mean_delta_gap": float(np.mean([r["delta_gap"] for r in vals])),
        "positive_rate": float(np.mean([r["delta_gap"] > 0 for r in vals])),
        "eos_win_rate": float(np.mean([r["eos_won"] for r in vals])),
    }


def teacher_span_screen(model, tok, layers, device, eos_ids, items) -> tuple[list, dict, dict]:
    rows = []
    neutral = neutral_id(tok)
    for idx, item in enumerate(items):
        manifest = build_mode_manifest(tok, item["prompt"])
        for state, ids in state_sequences(tok, manifest, item).items():
            if ids is None:
                continue
            clean = forward_snapshot(model, layers, device, eos_ids, ids)
            for group in ATOMIC_GROUPS + COMPOSITE_GROUPS:
                positions = manifest["groups"][group]
                for method in ("zero", "neutral"):
                    corrupt = forward_snapshot(
                        model, layers, device, eos_ids, ids,
                        embed_positions=positions, embed_mode=method, neutral_token=neutral,
                    )
                    rows.append({
                        "id": item["id"], "task": item["task"], "state": state,
                        "group": group, "method": method,
                        "condition": f"{group}/{method}",
                        "clean_gap": clean["gap"], "patched_gap": corrupt["gap"],
                        "delta_gap": corrupt["gap"] - clean["gap"],
                        "clean_eos_won": clean["eos_won"],
                        "eos_won": corrupt["eos_won"],
                    })
        if (idx + 1) % 10 == 0:
            log(f"  Phase977 span screen {idx+1}/{len(items)}")

    summary = {}
    for group in ATOMIC_GROUPS + COMPOSITE_GROUPS:
        summary[group] = {}
        for method in ("zero", "neutral"):
            cond = f"{group}/{method}"
            summary[group][method] = {
                state: summarize_delta(rows, cond, state) for state in STATE_NAMES
            }

    def candidate(group):
        z = summary[group]["zero"]["answer_period"]
        n = summary[group]["neutral"]["answer_period"]
        return {
            "group": group,
            "min_mean_effect": min(z["mean_delta_gap"], n["mean_delta_gap"]),
            "min_positive_rate": min(z["positive_rate"], n["positive_rate"]),
            "passes": bool(min(z["mean_delta_gap"], n["mean_delta_gap"]) >= 5
                           and min(z["positive_rate"], n["positive_rate"]) >= 0.80),
        }

    atomic = [candidate(g) for g in ATOMIC_GROUPS]
    composite = [candidate(g) for g in COMPOSITE_GROUPS]
    pool = [x for x in atomic if x["passes"]]
    selection_level = "atomic"
    if not pool:
        pool = [x for x in composite if x["passes"]]
        selection_level = "composite"
    if not pool:
        pool = [x for x in composite if x["group"] == "full_mode_block"]
        selection_level = "fallback_full"
    selected = max(pool, key=lambda x: (x["min_mean_effect"], x["min_positive_rate"]))
    selected["selection_level"] = selection_level
    selected["rule"] = "atomic first; both zero/neutral AP mean>=5 and positive>=80%; otherwise composite then full fallback"
    return rows, summary, selected


def aggregate_movement(rows: list[dict]) -> dict:
    before = sum(r["distance_before"] for r in rows)
    after = sum(r["distance_after"] for r in rows)
    return {
        "n": len(rows),
        "mean_delta_gap": float(np.mean([r["delta_gap"] for r in rows])),
        "toward_source_rate": float(np.mean([r["toward_source"] for r in rows])),
        "aggregate_recovery": float(1 - after / before) if before > 1e-8 else None,
        "eos_win_rate": float(np.mean([r["eos_won"] for r in rows])),
    }


def aggregate_mode_movement(rows: list[dict]) -> dict:
    before = sum(r["mode_distance_before"] for r in rows)
    after = sum(r["mode_distance_after"] for r in rows)
    return {
        "n": len(rows),
        "mean_delta_mode_margin": float(np.mean(
            [r["delta_mode_margin"] for r in rows])),
        "mode_toward_source_rate": float(np.mean(
            [r["mode_toward_source"] for r in rows])),
        "aggregate_mode_recovery": (
            float(1 - after / before) if before > 1e-8 else None),
    }


def marker_layer_scan(model, tok, layers, device, eos_ids, items, group: str) -> dict:
    rows = []
    depths = list(range(1, len(layers) + 1))
    neutral = neutral_id(tok)
    for idx, item in enumerate(items):
        manifest = build_mode_manifest(tok, item["prompt"])
        positions = manifest["groups"][group]
        content = state_texts(item)["answer_period"]
        ids = token_ids(tok, manifest["no_think_text"] + content, add_special_tokens=False)
        clean_last = forward_snapshot(model, layers, device, eos_ids, ids,
                                      capture_depths=depths, capture_positions=[-1])
        clean_span = forward_snapshot(model, layers, device, eos_ids, ids,
                                      capture_depths=depths, capture_positions=positions)
        for method in ("zero", "neutral"):
            corrupt_last = forward_snapshot(
                model, layers, device, eos_ids, ids, capture_depths=depths,
                capture_positions=[-1], embed_positions=positions,
                embed_mode=method, neutral_token=neutral,
            )
            corrupt_span = forward_snapshot(
                model, layers, device, eos_ids, ids, capture_depths=depths,
                capture_positions=positions, embed_positions=positions,
                embed_mode=method, neutral_token=neutral,
            )
            for layer in range(len(layers)):
                depth = layer + 1
                for scope, clean_cap, corrupt_cap, patch_positions in (
                    ("last_position", clean_last, corrupt_last, [-1]),
                    ("marker_span", clean_span, corrupt_span, positions),
                ):
                    rescued = forward_snapshot(
                        model, layers, device, eos_ids, ids,
                        embed_positions=positions, embed_mode=method,
                        neutral_token=neutral, patch_layer=layer,
                        patch_positions=patch_positions,
                        patch_vector=clean_cap["vectors"][depth],
                    )
                    damaged = forward_snapshot(
                        model, layers, device, eos_ids, ids,
                        patch_layer=layer, patch_positions=patch_positions,
                        patch_vector=corrupt_cap["vectors"][depth],
                    )
                    rows.append({"id": item["id"], "task": item["task"],
                                 "layer": layer, "scope": scope, "method": method,
                                 "direction": "clean_to_corrupt",
                                 **movement(clean_last, corrupt_last, rescued)})
                    rows.append({"id": item["id"], "task": item["task"],
                                 "layer": layer, "scope": scope, "method": method,
                                 "direction": "corrupt_to_clean",
                                 **movement(corrupt_last, clean_last, damaged)})
        log(f"  Phase977 marker layer scan {idx+1}/{len(items)}")

    summary = []
    for layer in range(len(layers)):
        rec = {"layer": layer}
        for scope in ("last_position", "marker_span"):
            for method in ("zero", "neutral"):
                for direction in ("clean_to_corrupt", "corrupt_to_clean"):
                    vals = [r for r in rows if r["layer"] == layer
                            and r["scope"] == scope and r["method"] == method
                            and r["direction"] == direction]
                    rec[f"{scope}/{method}/{direction}"] = aggregate_movement(vals)
        summary.append(rec)

    def bidirectional_pass(rec, scope, method):
        a = rec[f"{scope}/{method}/clean_to_corrupt"]
        b = rec[f"{scope}/{method}/corrupt_to_clean"]
        return (a["aggregate_recovery"] is not None and b["aggregate_recovery"] is not None
                and a["aggregate_recovery"] >= 0.80 and b["aggregate_recovery"] >= 0.80
                and a["toward_source_rate"] >= 0.75 and b["toward_source_rate"] >= 0.75)

    last_candidates = [r["layer"] for r in summary
                       if all(bidirectional_pass(r, "last_position", method)
                              for method in ("zero", "neutral"))]
    marker_candidates = [r["layer"] for r in summary
                         if r["layer"] <= 23
                         and all(bidirectional_pass(r, "marker_span", method)
                                 for method in ("zero", "neutral"))]
    return {
        "rows": rows,
        "summary": summary,
        "selected_last_arrival_layer": min(last_candidates) if last_candidates else len(layers) - 1,
        "last_arrival_passed": bool(last_candidates),
        "selected_marker_retention_layer": max(marker_candidates) if marker_candidates else None,
        "marker_retention_passed": bool(marker_candidates),
        "rule": "both zero and neutral must pass bidirectional recovery/toward-source thresholds",
        "warning": "arrival/retention are causal readout boundaries, not original write-in layers",
    }


def legal_mode_layer_scan(model, tok, layers, device, eos_ids, items) -> dict:
    """Swap official hard-mode states and score the explicit <think> margin."""
    rows = []
    eligibility = []
    depths = list(range(1, len(layers) + 1))
    think_open = token_ids(tok, "<think>", add_special_tokens=False)
    if len(think_open) != 1:
        raise RuntimeError("think-open is not one token")
    think_open_id = int(think_open[0])
    for idx, item in enumerate(items):
        manifest = build_mode_manifest(tok, item["prompt"])
        nt = forward_snapshot(model, layers, device, eos_ids, manifest["no_think_ids"],
                              capture_depths=depths, capture_positions=[-1],
                              mode_token_id=think_open_id)
        th = forward_snapshot(model, layers, device, eos_ids, manifest["thinking_ids"],
                              capture_depths=depths, capture_positions=[-1],
                              mode_token_id=think_open_id)
        eligible = bool(th["greedy_id"] == think_open_id
                        and nt["greedy_id"] != think_open_id
                        and th["mode_margin"] > 0 and nt["mode_margin"] < 0)
        eligibility.append({
            "id": item["id"], "task": item["task"], "eligible": eligible,
            "thinking_greedy_id": th["greedy_id"],
            "no_think_greedy_id": nt["greedy_id"],
            "thinking_mode_margin": th["mode_margin"],
            "no_think_mode_margin": nt["mode_margin"],
        })
        if not eligible:
            continue
        for layer in range(len(layers)):
            depth = layer + 1
            nt_to_th = forward_snapshot(
                model, layers, device, eos_ids, manifest["thinking_ids"],
                patch_layer=layer, patch_positions=[-1], patch_vector=nt["vectors"][depth],
                mode_token_id=think_open_id,
            )
            th_to_nt = forward_snapshot(
                model, layers, device, eos_ids, manifest["no_think_ids"],
                patch_layer=layer, patch_positions=[-1], patch_vector=th["vectors"][depth],
                mode_token_id=think_open_id,
            )
            rows.append({
                "id": item["id"], "task": item["task"], "layer": layer,
                "direction": "no_think_to_thinking", "source_id": nt["greedy_id"],
                "target_id": th["greedy_id"], "patched_id": nt_to_th["greedy_id"],
                "exact_source": nt_to_th["greedy_id"] == nt["greedy_id"],
                "mode_flip": nt_to_th["mode_margin"] < 0,
                "patched_eos": nt_to_th["eos_won"],
                **movement(nt, th, nt_to_th), **mode_movement(nt, th, nt_to_th),
            })
            rows.append({
                "id": item["id"], "task": item["task"], "layer": layer,
                "direction": "thinking_to_no_think", "source_id": th["greedy_id"],
                "target_id": nt["greedy_id"], "patched_id": th_to_nt["greedy_id"],
                "exact_source": th_to_nt["greedy_id"] == th["greedy_id"],
                "mode_flip": th_to_nt["mode_margin"] > 0,
                "patched_eos": th_to_nt["eos_won"],
                **movement(th, nt, th_to_nt), **mode_movement(th, nt, th_to_nt),
            })
        if (idx + 1) % 8 == 0:
            log(f"  Phase977 legal-mode layer scan {idx+1}/{len(items)}")

    summary = []
    for layer in range(len(layers)):
        rec = {"layer": layer}
        for direction in ("no_think_to_thinking", "thinking_to_no_think"):
            vals = [r for r in rows if r["layer"] == layer and r["direction"] == direction]
            rec[direction] = ({
                **aggregate_movement(vals), **aggregate_mode_movement(vals),
                "exact_source_rate": float(np.mean([r["exact_source"] for r in vals])),
                "mode_flip_rate": float(np.mean([r["mode_flip"] for r in vals])),
                "patched_eos_rate": float(np.mean([r["patched_eos"] for r in vals])),
            } if vals else {"n": 0, "mode_flip_rate": 0.0,
                            "mode_toward_source_rate": 0.0,
                            "aggregate_mode_recovery": None,
                            "exact_source_rate": 0.0, "patched_eos_rate": 0.0})
        summary.append(rec)
    candidates = []
    for rec in summary:
        a, b = rec["no_think_to_thinking"], rec["thinking_to_no_think"]
        if (a["mode_flip_rate"] >= 0.75 and b["mode_flip_rate"] >= 0.75
                and a["mode_toward_source_rate"] >= 0.75
                and b["mode_toward_source_rate"] >= 0.75
                and a["aggregate_mode_recovery"] is not None
                and b["aggregate_mode_recovery"] is not None
                and a["aggregate_mode_recovery"] >= 0.50
                and b["aggregate_mode_recovery"] >= 0.50):
            candidates.append(rec["layer"])
    eligibility_rate = float(np.mean([x["eligible"] for x in eligibility]))
    if eligibility_rate < 0.90:
        candidates = []
    return {
        "think_open_id": think_open_id,
        "eligibility": eligibility,
        "eligible_n": sum(x["eligible"] for x in eligibility),
        "eligibility_rate": eligibility_rate,
        "rows": rows,
        "summary": summary,
        "selected_arrival_layer": min(candidates) if candidates else len(layers) - 1,
        "selection_passed": bool(candidates),
        "rule": "eligible legal baselines>=90%; earliest bidirectional <think>-margin sign flip>=75%, toward-source>=75%, aggregate margin recovery>=50%; final-layer fallback remains failed exploration",
        "warning": "official prefixes differ in length and final token; this is legal-state sufficiency, not an equal-length token attribution",
    }


def dev_validate(model, tok, layers, device, eos_ids, items, selected_group,
                 marker_layer, last_layer, legal_layer, think_open_id) -> dict:
    rows = []
    span_rows = []
    neutral = neutral_id(tok)
    gen = torch.Generator(device="cpu").manual_seed(977)
    bases = []
    legal_eligibility = []
    for item in items:
        manifest = build_mode_manifest(tok, item["prompt"])
        positions = manifest["groups"][selected_group]
        # One-shot confirmation of the discovery-selected span on every named
        # state and with both corruption controls.  This is kept separate from
        # the layer-rescue rows because it is a necessity, not a movement,
        # measurement.
        for state, state_ids in state_sequences(tok, manifest, item).items():
            if state_ids is None:
                continue
            state_clean = forward_snapshot(model, layers, device, eos_ids, state_ids)
            for method in ("zero", "neutral"):
                state_corrupt = forward_snapshot(
                    model, layers, device, eos_ids, state_ids,
                    embed_positions=positions, embed_mode=method, neutral_token=neutral,
                )
                span_rows.append({
                    "id": item["id"], "task": item["task"], "state": state,
                    "method": method, "scope": "selected_span",
                    "clean_gap": state_clean["gap"],
                    "corrupt_gap": state_corrupt["gap"],
                    "delta_gap": state_corrupt["gap"] - state_clean["gap"],
                    "clean_eos_won": state_clean["eos_won"],
                    "corrupt_eos_won": state_corrupt["eos_won"],
                })
        # AP marker validation and selected layer rescues.
        ids = token_ids(tok, manifest["no_think_text"] + state_texts(item)["answer_period"],
                        add_special_tokens=False)
        ordinary_positions = content_control_positions(manifest, len(positions))
        ordinary_clean = forward_snapshot(model, layers, device, eos_ids, ids)
        for method in ("zero", "neutral"):
            ordinary_corrupt = forward_snapshot(
                model, layers, device, eos_ids, ids,
                embed_positions=ordinary_positions, embed_mode=method,
                neutral_token=neutral)
            span_rows.append({
                "id": item["id"], "task": item["task"],
                "state": "answer_period", "method": method,
                "scope": "ordinary_content_control",
                "clean_gap": ordinary_clean["gap"],
                "corrupt_gap": ordinary_corrupt["gap"],
                "delta_gap": ordinary_corrupt["gap"] - ordinary_clean["gap"],
                "clean_eos_won": ordinary_clean["eos_won"],
                "corrupt_eos_won": ordinary_corrupt["eos_won"],
            })
        marker_depths = sorted(set(x + 1 for x in [marker_layer, last_layer]
                                   if x is not None))
        clean_last = forward_snapshot(model, layers, device, eos_ids, ids,
                                      capture_depths=marker_depths, capture_positions=[-1])
        clean_span = (forward_snapshot(
            model, layers, device, eos_ids, ids,
            capture_depths=[marker_layer + 1], capture_positions=positions)
            if marker_layer is not None else None)
        for method in ("zero", "neutral"):
            corrupt_last = forward_snapshot(
                model, layers, device, eos_ids, ids, capture_depths=marker_depths,
                capture_positions=[-1], embed_positions=positions,
                embed_mode=method, neutral_token=neutral,
            )
            if last_layer is not None:
                rescue = forward_snapshot(
                    model, layers, device, eos_ids, ids, embed_positions=positions,
                    embed_mode=method, neutral_token=neutral, patch_layer=last_layer,
                    patch_positions=[-1],
                    patch_vector=clean_last["vectors"][last_layer + 1],
                )
                rows.append({"id": item["id"], "task": item["task"],
                             "family": "marker",
                             "condition": f"last_position_rescue/{method}",
                             "layer": last_layer,
                             **movement(clean_last, corrupt_last, rescue)})
            if marker_layer is not None:
                rescue = forward_snapshot(
                    model, layers, device, eos_ids, ids, embed_positions=positions,
                    embed_mode=method, neutral_token=neutral, patch_layer=marker_layer,
                    patch_positions=positions,
                    patch_vector=clean_span["vectors"][marker_layer + 1],
                )
                rows.append({"id": item["id"], "task": item["task"],
                             "family": "marker",
                             "condition": f"marker_span_rescue/{method}",
                             "layer": marker_layer,
                             **movement(clean_last, corrupt_last, rescue)})

        # Official legal-mode current-step transfer and controls.
        nt = forward_snapshot(model, layers, device, eos_ids, manifest["no_think_ids"],
                              capture_depths=[legal_layer + 1], capture_positions=[-1],
                              mode_token_id=think_open_id)
        th = forward_snapshot(model, layers, device, eos_ids, manifest["thinking_ids"],
                              capture_depths=[legal_layer + 1], capture_positions=[-1],
                              mode_token_id=think_open_id)
        eligible = bool(th["greedy_id"] == think_open_id
                        and nt["greedy_id"] != think_open_id
                        and th["mode_margin"] > 0 and nt["mode_margin"] < 0)
        legal_eligibility.append({"id": item["id"], "task": item["task"],
                                  "eligible": eligible})
        if eligible:
            paired = forward_snapshot(
                model, layers, device, eos_ids, manifest["thinking_ids"],
                patch_layer=legal_layer, patch_positions=[-1],
                patch_vector=nt["vectors"][legal_layer + 1],
                mode_token_id=think_open_id,
            )
            reverse = forward_snapshot(
                model, layers, device, eos_ids, manifest["no_think_ids"],
                patch_layer=legal_layer, patch_positions=[-1],
                patch_vector=th["vectors"][legal_layer + 1],
                mode_token_id=think_open_id,
            )
            delta = nt["vectors"][legal_layer + 1] - th["vectors"][legal_layer + 1]
            random = torch.randn(delta.shape, generator=gen)
            random *= torch.linalg.vector_norm(delta) / max(
                float(torch.linalg.vector_norm(random)), 1e-12)
            random_run = forward_snapshot(
                model, layers, device, eos_ids, manifest["thinking_ids"],
                patch_layer=legal_layer, patch_positions=[-1], add_vector=random,
                mode_token_id=think_open_id,
            )
            self_run = forward_snapshot(
                model, layers, device, eos_ids, manifest["thinking_ids"],
                patch_layer=legal_layer, patch_positions=[-1],
                patch_vector=th["vectors"][legal_layer + 1],
                mode_token_id=think_open_id,
            )
            for condition, run in (("paired_no_think_to_thinking", paired),
                                   ("random_norm_control", random_run),
                                   ("self_patch", self_run)):
                rows.append({
                    "id": item["id"], "task": item["task"], "family": "legal_mode",
                    "condition": condition, "layer": legal_layer,
                    "source_id": nt["greedy_id"], "target_id": th["greedy_id"],
                    "patched_id": run["greedy_id"],
                    "exact_source": run["greedy_id"] == nt["greedy_id"],
                    "exact_target": run["greedy_id"] == th["greedy_id"],
                    "mode_flip": run["mode_margin"] < 0,
                    **movement(nt, th, run), **mode_movement(nt, th, run),
                })
            rows.append({
                "id": item["id"], "task": item["task"], "family": "legal_mode",
                "condition": "paired_thinking_to_no_think", "layer": legal_layer,
                "source_id": th["greedy_id"], "target_id": nt["greedy_id"],
                "patched_id": reverse["greedy_id"],
                "exact_source": reverse["greedy_id"] == th["greedy_id"],
                "exact_target": reverse["greedy_id"] == nt["greedy_id"],
                "mode_flip": reverse["mode_margin"] > 0,
                **movement(th, nt, reverse), **mode_movement(th, nt, reverse),
            })
            bases.append({"id": item["id"], "task": item["task"],
                          "manifest": manifest, "nt": nt, "th": th})
        if len(legal_eligibility) % 8 == 0:
            log(f"  Phase977 dev validation bases {len(legal_eligibility)}/{len(items)}")

    # Same-mode shuffled donor: content-independence control, not a random control.
    for i, base in enumerate(bases):
        donor = bases[(i + 1) % len(bases)]["nt"]
        run = forward_snapshot(
            model, layers, device, eos_ids, base["manifest"]["thinking_ids"],
            patch_layer=legal_layer, patch_positions=[-1],
            patch_vector=donor["vectors"][legal_layer + 1],
            mode_token_id=think_open_id,
        )
        rows.append({
            "id": base["id"], "task": base["task"], "family": "legal_mode",
            "condition": "shuffled_no_think_to_thinking", "layer": legal_layer,
            "source_id": donor["greedy_id"], "target_id": base["th"]["greedy_id"],
            "patched_id": run["greedy_id"],
            "exact_source": run["greedy_id"] == donor["greedy_id"],
            "exact_target": run["greedy_id"] == base["th"]["greedy_id"],
            "mode_flip": run["mode_margin"] < 0,
            **movement(donor, base["th"], run),
            **mode_movement(donor, base["th"], run),
        })

    summary = {}
    for condition in sorted({r["condition"] for r in rows}):
        vals = [r for r in rows if r["condition"] == condition]
        entry = aggregate_movement(vals)
        if "mode_flip" in vals[0]:
            entry.update({**aggregate_mode_movement(vals),
                "mode_flip_rate": float(np.mean([r["mode_flip"] for r in vals])),
                "exact_source_rate": float(np.mean([r["exact_source"] for r in vals])),
                "exact_target_rate": float(np.mean([r["exact_target"] for r in vals])),
            })
        summary[condition] = entry
    span_summary = {}
    for method in ("zero", "neutral"):
        span_summary[method] = {}
        for state in STATE_NAMES:
            vals = [r for r in span_rows if r["method"] == method
                    and r["state"] == state and r["scope"] == "selected_span"]
            span_summary[method][state] = ({
                "n": len(vals),
                "mean_delta_gap": float(np.mean([r["delta_gap"] for r in vals])),
                "positive_rate": float(np.mean([r["delta_gap"] > 0 for r in vals])),
                "clean_eos_win_rate": float(np.mean([r["clean_eos_won"] for r in vals])),
                "corrupt_eos_win_rate": float(np.mean([r["corrupt_eos_won"] for r in vals])),
            } if vals else {"n": 0, "mean_delta_gap": None,
                            "positive_rate": None, "clean_eos_win_rate": None,
                            "corrupt_eos_win_rate": None})
    legal = summary["paired_no_think_to_thinking"]
    reverse = summary["paired_thinking_to_no_think"]
    random = summary["random_norm_control"]
    self_patch = summary["self_patch"]
    legal_eligible_rate = float(np.mean([x["eligible"] for x in legal_eligibility]))
    z_ap = span_summary["zero"]["answer_period"]
    n_ap = span_summary["neutral"]["answer_period"]
    content_control = {}
    span_by_task = {}
    for method in ("zero", "neutral"):
        vals = [r for r in span_rows if r["method"] == method
                and r["scope"] == "ordinary_content_control"]
        content_control[method] = {
            "n": len(vals),
            "mean_delta_gap": float(np.mean([r["delta_gap"] for r in vals])),
            "positive_rate": float(np.mean([r["delta_gap"] > 0 for r in vals])),
        }
        span_by_task[method] = {}
        for task in sorted({r["task"] for r in span_rows}):
            task_vals = [r for r in span_rows if r["method"] == method
                         and r["scope"] == "selected_span"
                         and r["state"] == "answer_period" and r["task"] == task]
            span_by_task[method][task] = float(np.mean(
                [r["delta_gap"] for r in task_vals]))
    content_specific = all(
        abs(content_control[method]["mean_delta_gap"])
        <= abs(span_summary[method]["answer_period"]["mean_delta_gap"]) / 3
        for method in ("zero", "neutral"))
    invalid_states = ("answer_incomplete", "answer_comma_requires_continue",
                      "continuation_incomplete", "wrong_answer",
                      "wrong_answer_period")
    invalid_state_safe = all(
        span_summary[method][state]["n"] > 0
        and (span_summary[method][state]["corrupt_eos_win_rate"]
             - span_summary[method][state]["clean_eos_win_rate"] <= 0.05)
        for method in ("zero", "neutral") for state in invalid_states)
    task_consistent = sum(
        span_by_task["zero"][task] > 0 and span_by_task["neutral"][task] > 0
        for task in span_by_task["zero"]) >= 6
    gates = {
        "span_replicated": bool(min(z_ap["mean_delta_gap"], n_ap["mean_delta_gap"]) >= 5
                                and min(z_ap["positive_rate"], n_ap["positive_rate"]) >= 0.80
                                and content_specific and invalid_state_safe
                                and task_consistent),
        "legal_current_step": bool(legal["mode_flip_rate"] >= 0.75
                                   and reverse["mode_flip_rate"] >= 0.75
                                   and legal["mode_toward_source_rate"] >= 0.75
                                   and reverse["mode_toward_source_rate"] >= 0.75
                                   and legal["aggregate_mode_recovery"] >= 0.50
                                   and reverse["aggregate_mode_recovery"] >= 0.50
                                   and random["mode_flip_rate"] <= 0.25
                                   and self_patch["exact_target_rate"] == 1.0
                                   and legal_eligible_rate >= 0.90),
        "last_rescue": bool(all(
            summary.get(f"last_position_rescue/{method}", {}).get(
                "aggregate_recovery", -math.inf) >= 0.80
            for method in ("zero", "neutral"))),
        "marker_span_rescue": bool(marker_layer is not None and all(
            summary.get(f"marker_span_rescue/{method}", {}).get(
                "aggregate_recovery", -math.inf) >= 0.80
            for method in ("zero", "neutral"))),
    }
    return {"span_summary": span_summary,
            "ordinary_content_control": content_control,
            "span_by_task": span_by_task,
            "span_rows": span_rows,
            "legal_eligibility": legal_eligibility,
            "legal_eligibility_rate": legal_eligible_rate,
            "summary": summary, "gates": gates, "rows": rows}


def run():
    if not LEGAL_DISCOVERY_REAUDIT.exists():
        raise RuntimeError("strict-v2 discovery legal re-audit is absent; mechanism scan stays closed")
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
            "strict-v2 discovery legal gate is NO-GO; mechanism scan remains closed")
    ensure_dir(OUT)
    t0 = time.time()
    if not LEGAL_DEV_SUMMARY.exists():
        raise RuntimeError("legal development trajectory summary is absent")
    legal_summary = json.loads(LEGAL_DEV_SUMMARY.read_text(encoding="utf-8"))
    if not legal_summary.get("decision_gate", {}).get("passed", False):
        raise RuntimeError("legal development trajectory gate failed; mechanism scan stays closed")
    discovery = build_external_dataset()
    dev = build_dev_dataset()
    audit = audit_dataset(previous_prompts=[x["prompt"] for x in discovery])
    if (audit["n_items"] != 64 or not audit["passed"] or audit["errors"]
            or audit["schema_issues"] or audit["cross_set_overlap"]):
        raise RuntimeError(f"Phase977 dev audit failed: {audit}")
    # Discovery scan uses 4 per task for stable first-token rates; marker
    # propagation scan uses one per task because it is a 4-way 36-layer sweep.
    discovery32 = []
    scan8 = []
    for task in sorted({x["task"] for x in discovery}):
        rows = [x for x in discovery if x["task"] == task]
        discovery32.extend(rows[:4])
        scan8.append(rows[0])

    model, tok, device = load_model(MODEL)
    layers = get_layers(model)
    eos_ids = get_eos_ids(model, tok)
    example = build_mode_manifest(tok, discovery[0]["prompt"])
    result = {
        "phase": PHASE, "schema_version": 2, "model": MODEL,
        "n_layers": len(layers),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "discovery_sha256": hashlib.sha256(json.dumps(
            discovery, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "development_sha256": hashlib.sha256(json.dumps(
            dev, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "legal_development_summary_sha256": hashlib.sha256(
            LEGAL_DEV_SUMMARY.read_bytes()).hexdigest(),
        "legal_discovery_reaudit_sha256": legal_reaudit_sha256,
        "eos_token_ids": eos_ids,
        "evidence_partition": {
            "official_legal_modes": "enable_thinking True/False only",
            "span_interventions": "fixed-position embedding OOD diagnostics, not legal templates",
        },
        "state_manifest": {
            "answer_incomplete": "strict canonical answer token prefix; unavailable for one-token answers",
            "answer_complete": "correct answer without punctuation",
            "answer_period": "correct answer plus period",
            "answer_comma_requires_continue": "correct answer plus comma; syntax requires continuation",
            "continuation_incomplete": "correct answer followed by an incomplete new clause",
            "continuation_complete": "correct answer followed by a completed new sentence",
            "wrong_answer": "semantic error without punctuation",
            "wrong_answer_period": "semantic error plus period; boundary may close but clean must remain false",
        },
        "memo_correction": "Phase975 code X/XC meant continuation_incomplete/complete; the Phase975 memo's wrong-answer relabeling was incorrect",
        "template_manifest_example": {
            "thinking_ids": example["thinking_ids"],
            "no_think_ids": example["no_think_ids"],
            "suffix_ids": example["suffix_ids"],
            "suffix_tokens": example["suffix_tokens"],
            "groups": example["groups"],
        },
        "split": {"discovery_n": len(discovery), "legal_scan_n": len(discovery32),
                  "marker_scan_n": len(scan8), "dev_n": len(dev)},
        "dev_audit": audit,
    }
    path = OUT / "qwen3_result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    log("Phase977: 80-item fixed-position span decomposition")
    rows, summary, selected = teacher_span_screen(
        model, tok, layers, device, eos_ids, discovery)
    result["span_discovery"] = {"summary": summary, "selected": selected, "rows": rows}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Phase977: marker-position/last-position layer scan; group={selected['group']}")
    marker_scan = marker_layer_scan(
        model, tok, layers, device, eos_ids, scan8, selected["group"])
    result["marker_layer_scan"] = marker_scan
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    log("Phase977: official hard-mode bidirectional current-step scan")
    legal_scan = legal_mode_layer_scan(model, tok, layers, device, eos_ids, discovery32)
    result["legal_mode_layer_scan"] = legal_scan
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    log("Phase977: frozen 64-item dev validation")
    dev_result = dev_validate(
        model, tok, layers, device, eos_ids, dev, selected["group"],
        marker_scan["selected_marker_retention_layer"],
        marker_scan["selected_last_arrival_layer"],
        legal_scan["selected_arrival_layer"], legal_scan["think_open_id"],
    )
    result["development"] = dev_result
    result["frozen_candidates"] = {
        "span_group": selected["group"],
        "marker_retention_layer": marker_scan["selected_marker_retention_layer"],
        "last_position_arrival_layer": marker_scan["selected_last_arrival_layer"],
        "legal_mode_arrival_layer": legal_scan["selected_arrival_layer"],
        "all_dev_gates_passed": bool(
            marker_scan["last_arrival_passed"]
            and marker_scan["marker_retention_passed"]
            and legal_scan["selection_passed"]
            and all(dev_result["gates"].values())),
        "warning": "none of these labels means original write-in layer",
    }
    result["elapsed_seconds"] = time.time() - t0
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Saved {path}; elapsed={result['elapsed_seconds']/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 977 fixed-position span and official-mode causal diagnosis."
    )
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
