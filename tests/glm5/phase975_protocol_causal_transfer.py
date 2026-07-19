#!/usr/bin/env python3
"""Phase 975: protocol-token necessity, layerwise rescue, and causal transfer.

This experiment deliberately separates three questions which Phase 974 mixed:

1. Does the identity of a native-chat protocol span matter when sequence length,
   attention mask, and positions are held fixed?
2. At which post-block residual can a clean state rescue an embedding-corrupted
   state, and can the corrupted state damage the clean run in the reverse direction?
3. Can a chat last-position residual, or a frozen content-independent mean
   direction, transfer to a position-matched plain sequence?

All decisive forwards use batch size one.  The primary state is a completed
answer followed by a period because its final token ID is matched between plain
and chat for all 160 Phase-973 items.  U/C/X are retained as safety/interaction
checks, but cross-protocol patch rows are accepted only when the final token IDs
are exactly equal.

The 160-item corpus has already appeared in Phase 973/974.  The split below is
therefore a strict holdout for *new interventions*, not a globally unseen data
set.  A strong positive result must later be confirmed on new items.
"""
from __future__ import annotations

import gc
import json
import math
import re
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
from phase973_conditional_trajectory import build_dataset, get_eos_ids, summarize_logits


PHASE = 975
OUT = Path("tests/glm5/result/phase975_protocol_causal_transfer")
STATES = ["U", "C", "P", "K", "X", "XC"]
PRIMARY_STATE = "P"
NEUTRAL_TEXT = "\n"


def split_items(items):
    """32 discovery, 32 development, 96 untouched intervention holdout."""
    discovery, development, holdout = [], [], []
    for task in sorted({x["task"] for x in items}):
        rows = [x for x in items if x["task"] == task]
        discovery.extend(rows[:4])
        development.extend(rows[4:8])
        holdout.extend(rows[8:])
    return discovery, development, holdout


def scan_items(items):
    """One item per task for the expensive exhaustive layer sweep."""
    out = []
    for task in sorted({x["task"] for x in items}):
        out.append([x for x in items if x["task"] == task][0])
    return out


def state_contents(item):
    answer = re.sub(r"[\s.!?;:]+$", "", item["answer"])
    words = answer.split()
    partial = " ".join(words[:max(1, len(words) // 2)]) if len(words) > 1 else ""

    def remainder(name):
        text = item["states"][name]
        assert text.startswith(item["prompt"])
        return text[len(item["prompt"]):].lstrip()

    return {
        "U": partial,
        "C": answer,
        "P": answer + ".",
        "K": answer + ",",
        "X": remainder("continuation_incomplete"),
        "XC": remainder("continuation_complete"),
    }


def plain_text(item, state):
    if state == "K":
        return item["prompt"] + " " + re.sub(r"[\s.!?;:]+$", "", item["answer"]) + ","
    mapping = {"U": "unfinished", "C": "just_complete", "P": "punctuation_complete",
               "X": "continuation_incomplete", "XC": "continuation_complete"}
    return item["states"][mapping[state]]


def chat_template_text(tok, model_name, prompt, add_generation_prompt, teacher_final):
    kwargs = {"enable_thinking": False} if model_name == "qwen3" else {}
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=add_generation_prompt, **kwargs)
    if model_name == "deepseek7b" and add_generation_prompt and teacher_final:
        # Synthetic final-answer condition.  It is never interpreted as a natural R1 trajectory.
        text += "</think>\n\n"
    return text


def token_ids(tok, text, add_special_tokens=False):
    return list(tok(text, add_special_tokens=add_special_tokens,
                    return_attention_mask=False).input_ids)


def find_first(tokens, value, stop=None):
    end = len(tokens) if stop is None else stop
    for i in range(end):
        if tokens[i] == value:
            return i
    return None


def protocol_manifest(tok, model_name, prompt, teacher_final=True):
    """Return actual token-position groups for this tokenizer/template."""
    prefix_text = chat_template_text(tok, model_name, prompt, True, teacher_final)
    no_assistant_text = chat_template_text(tok, model_name, prompt, False, False)
    ids = token_ids(tok, prefix_text, add_special_tokens=False)
    no_ids = token_ids(tok, no_assistant_text, add_special_tokens=False)
    if ids[:len(no_ids)] != no_ids:
        raise RuntimeError(f"{model_name}: non-prefix chat template decomposition")
    suffix_start = len(no_ids)
    pieces = tok.convert_ids_to_tokens(ids)

    groups = {"start_scaffold": [], "turn_frame": [], "user_role": [],
              "assistant_role": [], "mode_marker": [],
              "synthetic_final_transition": []}
    if model_name == "glm4":
        groups["start_scaffold"] = list(range(min(2, len(ids))))
        p = find_first(pieces, "<|user|>", suffix_start)
        if p is not None:
            groups["user_role"] = [p]
        groups["assistant_role"] = list(range(suffix_start, len(ids)))
    elif model_name == "qwen3":
        p = find_first(pieces, "user", suffix_start)
        if p is not None:
            groups["user_role"] = [p]
        # The first im_start and the user-turn im_end are framing, not role labels.
        for i in range(suffix_start):
            if pieces[i] in ("<|im_start|>", "<|im_end|>"):
                groups["turn_frame"].append(i)
        groups["assistant_role"] = list(range(suffix_start, min(suffix_start + 3, len(ids))))
        groups["mode_marker"] = list(range(min(suffix_start + 3, len(ids)), len(ids)))
    elif model_name == "deepseek7b":
        if ids and ids[0] == tok.bos_token_id:
            groups["start_scaffold"] = [0]
        p = find_first(pieces, "<｜User｜>", suffix_start)
        if p is not None:
            groups["user_role"] = [p]
        groups["assistant_role"] = [suffix_start] if suffix_start < len(ids) else []
        # Keep the naturally shared open-think span separate from the artificial
        # teacher-forced </think> transition appended by Phase 974/975.
        groups["mode_marker"] = list(range(suffix_start + 1,
                                            min(suffix_start + 3, len(ids))))
        groups["synthetic_final_transition"] = list(range(
            min(suffix_start + 3, len(ids)), len(ids)))
    else:
        raise ValueError(model_name)

    for key in list(groups):
        groups[key] = sorted(set(i for i in groups[key] if 0 <= i < len(ids)))
    # "all_protocol" contains only spans shared by teacher-forced and natural
    # prefixes.  The DS synthetic final transition remains an explicit group.
    groups["all_protocol"] = sorted(set(
        i for k, v in groups.items() if k != "synthetic_final_transition" for i in v))
    return {"prefix_text": prefix_text, "prefix_ids": ids, "no_assistant_ids": no_ids,
            "tokens": pieces, "groups": groups, "suffix_start": suffix_start}


def make_ids(tok, model_name, item, state):
    content = state_contents(item)[state]
    plain = token_ids(tok, plain_text(item, state), add_special_tokens=True)
    manifest = protocol_manifest(tok, model_name, item["prompt"], teacher_final=True)
    full_chat_text = manifest["prefix_text"] + content
    chat = token_ids(tok, full_chat_text, add_special_tokens=False)
    if chat[:len(manifest["prefix_ids"])] != manifest["prefix_ids"]:
        raise RuntimeError(f"{model_name}/{item['id']}/{state}: prefix tokenization changed")
    return plain, chat, manifest


def neutral_id(tok):
    ids = token_ids(tok, NEUTRAL_TEXT, add_special_tokens=False)
    if not ids:
        raise RuntimeError("neutral text produced no token")
    return int(ids[0])


def position_matched_plain(tok, plain_ids, chat_ids):
    """Match final absolute position using neutral context tokens, never padding."""
    if len(plain_ids) > len(chat_ids):
        return None
    n = len(chat_ids) - len(plain_ids)
    if n == 0:
        return list(plain_ids)
    specials = set(int(x) for x in tok.all_special_ids)
    insert_at = 0
    while insert_at < len(plain_ids) and int(plain_ids[insert_at]) in specials:
        insert_at += 1
    return list(plain_ids[:insert_at]) + [neutral_id(tok)] * n + list(plain_ids[insert_at:])


def _replace_output(output, position, vector=None, add=None, counter=None):
    is_tuple = isinstance(output, tuple)
    y = output[0] if is_tuple else output
    z = y.clone()
    if vector is not None:
        z[:, position, :] = vector.to(device=z.device, dtype=z.dtype)
    if add is not None:
        z[:, position, :] += add.to(device=z.device, dtype=z.dtype)
    if counter is not None:
        counter[0] += 1
    return (z,) + output[1:] if is_tuple else z


def run_snapshot(model, layers, device, eos_ids, ids, capture_layers=None,
                 embed_positions=None, embed_mode=None, neutral_token_id=None,
                 patch_layer=None, patch_vector=None, add_vector=None):
    """One batch-1 forward with optional length-preserving embedding corruption."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(x)
    handles = []
    embed_count = [0]
    patch_count = [0]
    captured = {}
    capture_counts = defaultdict(int)
    try:
        if embed_positions:
            emb = model.get_input_embeddings()
            if embed_mode == "neutral":
                neutral = emb.weight[int(neutral_token_id)].detach()
            elif embed_mode == "zero":
                neutral = None
            else:
                raise ValueError(embed_mode)

            def embedding_hook(module, args, output):
                z = output.clone()
                valid = [p for p in embed_positions if p < z.shape[1]]
                if valid:
                    if neutral is None:
                        z[:, valid, :] = 0
                    else:
                        z[:, valid, :] = neutral.to(device=z.device, dtype=z.dtype)
                    embed_count[0] += 1
                return z
            handles.append(emb.register_forward_hook(embedding_hook))

        # Capture the exact tensor boundary used by the corresponding patch.
        # transformers may expose hidden_states[-1] *after* final norm, whereas
        # the last block hook is before final norm; direct block hooks avoid that
        # otherwise-fatal last-layer space mismatch.
        if capture_layers is not None:
            wanted = list(range(len(layers) + 1)) if capture_layers == "all" \
                else sorted(set(int(x) for x in capture_layers))
            if 0 in wanted:
                def capture_embedding(module, args, output):
                    captured[0] = output[0, -1].detach().to("cpu", torch.float32)
                    capture_counts[0] += 1
                handles.append(model.get_input_embeddings().register_forward_hook(capture_embedding))
            for depth in [d for d in wanted if d > 0]:
                if depth > len(layers):
                    raise ValueError(f"capture depth {depth} > n_layers {len(layers)}")

                def make_capture(d):
                    def capture_block(module, args, output):
                        y = output[0] if isinstance(output, tuple) else output
                        captured[d] = y[0, -1].detach().to("cpu", torch.float32)
                        capture_counts[d] += 1
                    return capture_block
                handles.append(layers[depth - 1].register_forward_hook(make_capture(depth)))

        if patch_layer is not None:
            def layer_hook(module, args, output):
                return _replace_output(output, -1, vector=patch_vector, add=add_vector,
                                       counter=patch_count)
            handles.append(layers[int(patch_layer)].register_forward_hook(layer_hook))

        with torch.no_grad():
            out = model(input_ids=x, attention_mask=mask, use_cache=False,
                        output_hidden_states=False, return_dict=True)
        sm = summarize_logits(out.logits, eos_ids)
        result = {"gap": float(sm["gap"][0]), "eos_logit": float(sm["eos_logit"][0]),
                  "eos_rank": int(sm["eos_rank"][0]), "eos_id": int(sm["eos_id"][0]),
                  "top_id": int(sm["top_id"][0]), "eos_won": bool(sm["gap"][0] < 0)}
        result["greedy_id"] = int(out.logits[0, -1].argmax().item())
        if capture_layers is not None:
            missing = [d for d in wanted if capture_counts[d] != 1 or d not in captured]
            if missing:
                raise RuntimeError(f"capture hook failure at depths={missing}")
            result["vectors"] = captured
        if embed_positions and embed_count[0] != 1:
            raise RuntimeError(f"embedding intervention count={embed_count[0]}")
        if patch_layer is not None and patch_count[0] != 1:
            raise RuntimeError(f"layer patch count={patch_count[0]}")
        return result
    finally:
        for h in reversed(handles):
            h.remove()


def public_snapshot(s):
    return {k: v for k, v in s.items() if k != "vectors"}


def movement_record(item, state, condition, source, target, patched, layer):
    before = abs(target["gap"] - source["gap"])
    after = abs(patched["gap"] - source["gap"])
    return {"id": item["id"], "task": item["task"], "prompt_template": item["prompt_template"],
            "state": state, "condition": condition, "layer": int(layer),
            "source_gap": source["gap"], "target_gap": target["gap"],
            "patched_gap": patched["gap"], "delta_gap": patched["gap"] - target["gap"],
            "distance_before": before, "distance_after": after,
            "distance_reduction": before - after,
            "toward_source": after < before,
            "source_top_id": source["top_id"], "target_top_id": target["top_id"],
            "patched_top_id": patched["top_id"],
            "competitor_changed": patched["top_id"] != target["top_id"],
            "patched_eos_won": patched["eos_won"]}


def summarize_values(rows, field):
    vals = [float(r[field]) for r in rows]
    return {"n": len(vals), "mean": float(np.mean(vals)) if vals else None,
            "negative_rate": float(np.mean([x < 0 for x in vals])) if vals else None,
            "positive_rate": float(np.mean([x > 0 for x in vals])) if vals else None}


def summarize_movements(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["condition"], r["state"])].append(r)
    out = {}
    for (cond, state), vals in by.items():
        before = sum(r["distance_before"] for r in vals)
        after = sum(r["distance_after"] for r in vals)
        out[f"{cond}/{state}"] = {
            "n": len(vals), "mean_delta_gap": float(np.mean([r["delta_gap"] for r in vals])),
            "toward_source_rate": float(np.mean([r["toward_source"] for r in vals])),
            "aggregate_recovery": float(1 - after / before) if before > 1e-8 else None,
            "eos_win_rate": float(np.mean([r["patched_eos_won"] for r in vals])),
            "competitor_changed_rate": float(np.mean([r["competitor_changed"] for r in vals])),
        }
    return out


def screen_protocol_groups(model_name, model, tok, layers, device, eos_ids, items):
    rows = []
    for n, item in enumerate(items):
        for state in ["U", "P", "K"]:
            _, chat, manifest = make_ids(tok, model_name, item, state)
            clean = run_snapshot(model, layers, device, eos_ids, chat)
            for group, positions in manifest["groups"].items():
                if not positions:
                    continue
                for method in ["zero", "neutral"]:
                    corrupt = run_snapshot(model, layers, device, eos_ids, chat,
                                           embed_positions=positions, embed_mode=method,
                                           neutral_token_id=neutral_id(tok))
                    rows.append({"id": item["id"], "task": item["task"], "state": state,
                                 "group": group, "method": method,
                                 "clean_gap": clean["gap"], "corrupt_gap": corrupt["gap"],
                                 "delta_gap": corrupt["gap"] - clean["gap"],
                                 "clean_eos_won": clean["eos_won"],
                                 "corrupt_eos_won": corrupt["eos_won"]})
        if (n + 1) % 8 == 0:
            log(f"  {model_name} protocol screen {n+1}/{len(items)}")
    summary = {}
    for group in sorted({r["group"] for r in rows}):
        summary[group] = {}
        for method in ["zero", "neutral"]:
            summary[group][method] = {}
            for state in ["U", "P", "K"]:
                vals = [r for r in rows if r["group"] == group and r["method"] == method
                        and r["state"] == state]
                summary[group][method][state] = {
                    **summarize_values(vals, "delta_gap"),
                    "corrupt_eos_win_rate": float(np.mean([r["corrupt_eos_won"] for r in vals]))
                }
    candidates = []
    for group, value in summary.items():
        z = value["zero"]["P"]["mean"]
        n = value["neutral"]["P"]["mean"]
        candidates.append({"group": group, "consistent_necessity_score": min(z, n),
                           "zero_P": z, "neutral_P": n})
    # DS's synthetic close marker is measured, but cannot be selected for the
    # natural-prefix intervention because it does not exist there.
    selectable = [x for x in candidates if x["group"] != "synthetic_final_transition"]
    # BOS/start scaffold is still measured as a general-input necessity control,
    # but it is shared by GLM/DS plain and chat tokenization and therefore cannot
    # explain their plain-vs-chat difference.  The migration scan must freeze a
    # chat-specific role/turn/mode span.
    individual = [x for x in selectable if x["group"] not in
                  ("all_protocol", "start_scaffold")]
    pool = [x for x in individual if x["consistent_necessity_score"] > 0]
    if not pool:
        pool = selectable
    selected = max(pool, key=lambda x: x["consistent_necessity_score"])
    selected["passes_chat_specific_effect_ge_2"] = bool(
        selected["consistent_necessity_score"] >= 2.0)
    return rows, summary, selected, sorted(candidates,
                                           key=lambda x: x["consistent_necessity_score"], reverse=True)


def residual_metrics(a, b):
    delta = a - b
    rms = float(torch.sqrt(torch.mean(delta * delta)))
    denom = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    cosine = float(torch.dot(a, b) / denom) if denom > 1e-12 else 0.0
    return rms, cosine


def layer_scan(model_name, model, tok, layers, device, eos_ids, items, selected_group):
    n_layers = len(layers)
    cached = []
    metric_rows = []
    for item in items:
        plain, chat, manifest = make_ids(tok, model_name, item, PRIMARY_STATE)
        plain_pos = position_matched_plain(tok, plain, chat)
        if plain_pos is None or plain_pos[-1] != chat[-1]:
            raise RuntimeError(f"Primary P alignment failed: {model_name}/{item['id']}")
        positions = manifest["groups"][selected_group]
        clean = run_snapshot(model, layers, device, eos_ids, chat, capture_layers="all")
        corrupt = run_snapshot(model, layers, device, eos_ids, chat, capture_layers="all",
                               embed_positions=positions, embed_mode="neutral",
                               neutral_token_id=neutral_id(tok))
        posplain = run_snapshot(model, layers, device, eos_ids, plain_pos, capture_layers="all")
        cached.append((item, chat, plain_pos, positions, clean, corrupt, posplain))
        for depth in range(n_layers + 1):
            r1, c1 = residual_metrics(clean["vectors"][depth], corrupt["vectors"][depth])
            r2, c2 = residual_metrics(clean["vectors"][depth], posplain["vectors"][depth])
            metric_rows.append({"id": item["id"], "task": item["task"], "depth": depth,
                                "clean_corrupt_rms": r1, "clean_corrupt_cosine": c1,
                                "chat_plainpos_rms": r2, "chat_plainpos_cosine": c2})

    rows = []
    self_patch = []
    for idx, (item, chat, plain_pos, positions, clean, corrupt, posplain) in enumerate(cached):
        for layer in range(n_layers):
            depth = layer + 1
            rescued = run_snapshot(model, layers, device, eos_ids, chat,
                                   embed_positions=positions, embed_mode="neutral",
                                   neutral_token_id=neutral_id(tok), patch_layer=layer,
                                   patch_vector=clean["vectors"][depth])
            rows.append(movement_record(item, "P", "clean_to_corrupt_rescue",
                                        clean, corrupt, rescued, layer))
            damaged = run_snapshot(model, layers, device, eos_ids, chat, patch_layer=layer,
                                   patch_vector=corrupt["vectors"][depth])
            rows.append(movement_record(item, "P", "corrupt_to_clean_damage",
                                        corrupt, clean, damaged, layer))
            to_plain = run_snapshot(model, layers, device, eos_ids, plain_pos, patch_layer=layer,
                                    patch_vector=clean["vectors"][depth])
            rows.append(movement_record(item, "P", "chat_to_plainpos",
                                        clean, posplain, to_plain, layer))
            to_chat = run_snapshot(model, layers, device, eos_ids, chat, patch_layer=layer,
                                   patch_vector=posplain["vectors"][depth])
            rows.append(movement_record(item, "P", "plainpos_to_chat",
                                        posplain, clean, to_chat, layer))
            if idx == 0:
                same = run_snapshot(model, layers, device, eos_ids, chat, patch_layer=layer,
                                    patch_vector=clean["vectors"][depth])
                self_patch.append({"layer": layer, "base_gap": clean["gap"],
                                   "self_gap": same["gap"],
                                   "abs_diff": abs(same["gap"] - clean["gap"])})
        log(f"  {model_name} exhaustive layer scan {idx+1}/{len(cached)}")

    layer_summary = []
    for layer in range(n_layers):
        rec = {"layer": layer}
        for cond in ["clean_to_corrupt_rescue", "corrupt_to_clean_damage",
                     "chat_to_plainpos", "plainpos_to_chat"]:
            vals = [r for r in rows if r["layer"] == layer and r["condition"] == cond]
            before = sum(r["distance_before"] for r in vals)
            after = sum(r["distance_after"] for r in vals)
            rec[cond] = {"n": len(vals),
                         "mean_delta_gap": float(np.mean([r["delta_gap"] for r in vals])),
                         "toward_source_rate": float(np.mean([r["toward_source"] for r in vals])),
                         "aggregate_recovery": float(1 - after / before) if before > 1e-8 else None}
        layer_summary.append(rec)

    def earliest_pair(a, b):
        for rec in layer_summary:
            x, y = rec[a], rec[b]
            if (x["aggregate_recovery"] is not None and y["aggregate_recovery"] is not None
                    and x["aggregate_recovery"] >= 0.80 and y["aggregate_recovery"] >= 0.80
                    and x["toward_source_rate"] >= 0.75 and y["toward_source_rate"] >= 0.75):
                return rec["layer"]
        return n_layers - 1

    selected_rescue = earliest_pair("clean_to_corrupt_rescue", "corrupt_to_clean_damage")
    selected_transfer = earliest_pair("chat_to_plainpos", "plainpos_to_chat")
    metric_summary = []
    for depth in range(n_layers + 1):
        vals = [r for r in metric_rows if r["depth"] == depth]
        metric_summary.append({"depth": depth,
            "mean_clean_corrupt_rms": float(np.mean([r["clean_corrupt_rms"] for r in vals])),
            "mean_clean_corrupt_cosine": float(np.mean([r["clean_corrupt_cosine"] for r in vals])),
            "mean_chat_plainpos_rms": float(np.mean([r["chat_plainpos_rms"] for r in vals])),
            "mean_chat_plainpos_cosine": float(np.mean([r["chat_plainpos_cosine"] for r in vals]))})
    return {"rows": rows, "layer_summary": layer_summary,
            "residual_metric_rows": metric_rows, "residual_metric_summary": metric_summary,
            "self_patch": self_patch, "selected_rescue_layer": selected_rescue,
            "selected_transfer_layer": selected_transfer}


def mean_direction(model_name, model, tok, layers, device, eos_ids, items, layer):
    vectors = []
    for item in items:
        plain, chat, _ = make_ids(tok, model_name, item, PRIMARY_STATE)
        posplain = position_matched_plain(tok, plain, chat)
        if posplain is None or posplain[-1] != chat[-1]:
            continue
        c = run_snapshot(model, layers, device, eos_ids, chat, capture_layers=[layer + 1])
        p = run_snapshot(model, layers, device, eos_ids, posplain, capture_layers=[layer + 1])
        vectors.append(c["vectors"][layer + 1] - p["vectors"][layer + 1])
    if not vectors:
        raise RuntimeError("no matched vectors for frozen direction")
    return torch.stack(vectors).mean(0), len(vectors)


def evaluate_frozen(model_name, model, tok, layers, device, eos_ids, items,
                    group, rescue_layer, transfer_layer, frozen_direction, split_name):
    capture = sorted(set([rescue_layer + 1, transfer_layer + 1]))
    bases = []
    for item in items:
        for state in STATES:
            plain, chat, manifest = make_ids(tok, model_name, item, state)
            posplain = position_matched_plain(tok, plain, chat)
            clean = run_snapshot(model, layers, device, eos_ids, chat, capture_layers=capture)
            corrupt = run_snapshot(model, layers, device, eos_ids, chat, capture_layers=capture,
                                   embed_positions=manifest["groups"][group], embed_mode="neutral",
                                   neutral_token_id=neutral_id(tok))
            pp = None if posplain is None else run_snapshot(model, layers, device, eos_ids,
                                                            posplain, capture_layers=capture)
            bases.append({"item": item, "state": state, "chat_ids": chat,
                          "plainpos_ids": posplain, "positions": manifest["groups"][group],
                          "last_match": bool(posplain is not None and posplain[-1] == chat[-1]),
                          "clean": clean, "corrupt": corrupt, "plainpos": pp})
        if len(bases) % 48 == 0:
            log(f"  {model_name} {split_name} bases {len(bases)}/{len(items)*len(STATES)}")

    gen = torch.Generator(device="cpu").manual_seed(975)
    random_direction = torch.randn(frozen_direction.shape, generator=gen)
    random_direction *= torch.linalg.vector_norm(frozen_direction) / max(
        float(torch.linalg.vector_norm(random_direction)), 1e-12)
    rows = []
    for idx, b in enumerate(bases):
        item, state = b["item"], b["state"]
        clean, corrupt, pp = b["clean"], b["corrupt"], b["plainpos"]
        common = {"id": item["id"], "task": item["task"],
                  "prompt_template": item["prompt_template"], "state": state,
                  "last_token_match": b["last_match"]}
        rows.append({**common, "condition": "marker_neutral_corruption",
                     "base_gap": clean["gap"], "patched_gap": corrupt["gap"],
                     "delta_gap": corrupt["gap"] - clean["gap"],
                     "eos_won": corrupt["eos_won"]})
        rescued = run_snapshot(model, layers, device, eos_ids, b["chat_ids"],
                               embed_positions=b["positions"], embed_mode="neutral",
                               neutral_token_id=neutral_id(tok), patch_layer=rescue_layer,
                               patch_vector=clean["vectors"][rescue_layer + 1])
        m = movement_record(item, state, "clean_to_corrupt_rescue", clean, corrupt,
                            rescued, rescue_layer)
        rows.append({**common, **{k: v for k, v in m.items() if k not in common}})
        if pp is not None:
            mean_add = run_snapshot(model, layers, device, eos_ids, b["plainpos_ids"],
                                    patch_layer=transfer_layer, add_vector=frozen_direction)
            rows.append({**common, "condition": "frozen_mean_to_plainpos",
                         "base_gap": pp["gap"], "patched_gap": mean_add["gap"],
                         "delta_gap": mean_add["gap"] - pp["gap"],
                         "eos_won": mean_add["eos_won"]})
            random_add = run_snapshot(model, layers, device, eos_ids, b["plainpos_ids"],
                                      patch_layer=transfer_layer, add_vector=random_direction)
            rows.append({**common, "condition": "random_norm_control_to_plainpos",
                         "base_gap": pp["gap"], "patched_gap": random_add["gap"],
                         "delta_gap": random_add["gap"] - pp["gap"],
                         "eos_won": random_add["eos_won"]})
            if b["last_match"]:
                paired = run_snapshot(model, layers, device, eos_ids, b["plainpos_ids"],
                                      patch_layer=transfer_layer,
                                      patch_vector=clean["vectors"][transfer_layer + 1])
                m = movement_record(item, state, "paired_chat_to_plainpos", clean, pp,
                                    paired, transfer_layer)
                rows.append({**common, **{k: v for k, v in m.items() if k not in common}})

        # Reverse necessity and shuffled-content controls are primary-P only.
        if state == PRIMARY_STATE and pp is not None and b["last_match"]:
            damaged = run_snapshot(model, layers, device, eos_ids, b["chat_ids"],
                                   patch_layer=rescue_layer,
                                   patch_vector=corrupt["vectors"][rescue_layer + 1])
            m = movement_record(item, state, "corrupt_to_clean_damage", corrupt, clean,
                                damaged, rescue_layer)
            rows.append({**common, **{k: v for k, v in m.items() if k not in common}})
            reverse = run_snapshot(model, layers, device, eos_ids, b["chat_ids"],
                                   patch_layer=transfer_layer,
                                   patch_vector=pp["vectors"][transfer_layer + 1])
            m = movement_record(item, state, "paired_plainpos_to_chat", pp, clean,
                                reverse, transfer_layer)
            rows.append({**common, **{k: v for k, v in m.items() if k not in common}})
            mean_sub = run_snapshot(model, layers, device, eos_ids, b["chat_ids"],
                                    patch_layer=transfer_layer, add_vector=-frozen_direction)
            rows.append({**common, "condition": "frozen_mean_from_chat",
                         "base_gap": clean["gap"], "patched_gap": mean_sub["gap"],
                         "delta_gap": mean_sub["gap"] - clean["gap"],
                         "eos_won": mean_sub["eos_won"]})
            other = bases[((idx // len(STATES) + 1) % len(items)) * len(STATES) + STATES.index(state)]
            shuffled = run_snapshot(model, layers, device, eos_ids, b["plainpos_ids"],
                                    patch_layer=transfer_layer,
                                    patch_vector=other["clean"]["vectors"][transfer_layer + 1])
            m = movement_record(item, state, "shuffled_chat_to_plainpos", other["clean"], pp,
                                shuffled, transfer_layer)
            rows.append({**common, **{k: v for k, v in m.items() if k not in common}})
        if (idx + 1) % 48 == 0:
            log(f"  {model_name} {split_name} interventions {idx+1}/{len(bases)}")

    summary = {}
    for cond in sorted({r["condition"] for r in rows}):
        summary[cond] = {}
        for state in STATES:
            vals = [r for r in rows if r["condition"] == cond and r["state"] == state]
            if not vals:
                continue
            entry = {"n": len(vals),
                     "mean_delta_gap": float(np.mean([r["delta_gap"] for r in vals])),
                     "negative_delta_rate": float(np.mean([r["delta_gap"] < 0 for r in vals])),
                     "eos_win_rate": float(np.mean([r["eos_won"] if "eos_won" in r
                                                       else r.get("patched_eos_won", False) for r in vals]))}
            if "toward_source" in vals[0]:
                before = sum(r["distance_before"] for r in vals)
                after = sum(r["distance_after"] for r in vals)
                entry.update({"toward_source_rate": float(np.mean([r["toward_source"] for r in vals])),
                              "aggregate_recovery": float(1-after/before) if before > 1e-8 else None})
            summary[cond][state] = entry
    by_template = {}
    for template in ["shared_A", "unseen_B", "unseen_C"]:
        vals = [r for r in rows if r["condition"] == "frozen_mean_to_plainpos"
                and r["state"] == "P" and r["prompt_template"] == template]
        by_template[template] = summarize_values(vals, "delta_gap") if vals else {"n": 0}
    return rows, summary, by_template


def natural_corruption(model_name, model, tok, layers, device, eos_ids, items, group, method):
    """Native-chat generation with prompt-only, fixed-length embedding corruption."""
    rows = []
    max_new = 64 if model_name == "deepseek7b" else 32
    emb = model.get_input_embeddings()
    neutral = emb.weight[neutral_id(tok)].detach()
    for idx, item in enumerate(items):
        manifest = protocol_manifest(tok, model_name, item["prompt"], teacher_final=False)
        ids = manifest["prefix_ids"]
        positions = manifest["groups"].get(group, [])
        x = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.ones_like(x)
        count = [0]
        prefill_done = [False]

        def hook(module, args, output):
            z = output.clone()
            # Intervene exactly once on the full prompt prefill.  Cached decode
            # often has sequence length one, so checking absolute position alone
            # would accidentally corrupt every generated token when pos0 is in
            # the selected group.
            valid = [p for p in positions if p < z.shape[1]]
            if not prefill_done[0] and z.shape[1] == len(ids) and valid:
                if method == "zero":
                    z[:, valid, :] = 0
                else:
                    z[:, valid, :] = neutral.to(device=z.device, dtype=z.dtype)
                count[0] += 1
                prefill_done[0] = True
            return z

        h = None if method == "clean" else emb.register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = model.generate(input_ids=x, attention_mask=mask, max_new_tokens=max_new,
                                     do_sample=False, pad_token_id=tok.pad_token_id,
                                     eos_token_id=eos_ids, return_dict_in_generate=True)
        finally:
            if h is not None:
                h.remove()
        expected_count = 0 if method == "clean" else 1
        if count[0] != expected_count:
            raise RuntimeError(f"natural prefill intervention count={count[0]}, expected={expected_count}")
        generated = out.sequences[0, len(ids):]
        id_list = generated.tolist()
        decoded = tok.decode(generated, skip_special_tokens=False)
        plain = tok.decode(generated, skip_special_tokens=True)
        has_eos = any(int(t) in eos_ids for t in id_list)
        rows.append({"id": item["id"], "task": item["task"],
                     "prompt_template": item["prompt_template"], "group": group,
                     "method": method, "generated": decoded, "plain": plain,
                     "has_expected": item["answer"].lower() in plain.lower(),
                     "has_eos": has_eos, "n_tokens": len(id_list),
                     "prefill_intervention_count": count[0]})
        if (idx + 1) % 32 == 0:
            log(f"  {model_name} natural {group}/{method} {idx+1}/{len(items)}")
    return {"n": len(rows), "group": group, "method": method,
            "expected_rate": float(np.mean([r["has_expected"] for r in rows])),
            "eos_rate": float(np.mean([r["has_eos"] for r in rows])),
            "expected_and_eos_rate": float(np.mean([r["has_expected"] and r["has_eos"] for r in rows])),
            "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])), "rows": rows}


def run(model_name):
    ensure_dir(OUT)
    t0 = time.time()
    items = build_dataset()
    discovery, development, holdout = split_items(items)
    exhaustive = scan_items(discovery)
    model, tok, device = load_model(model_name)
    layers = get_layers(model)
    eos_ids = get_eos_ids(model, tok)
    out_path = OUT / f"{model_name}_result.json"
    result = {
        "phase": PHASE, "model": model_name, "n_layers": len(layers),
        "eos_token_ids": eos_ids, "batch_size": 1,
        "primary_state": PRIMARY_STATE,
        "state_warning": "U/C/X cross-protocol patches are filtered by exact final-token ID",
        "corpus_warning": "strict intervention holdout, but all texts were observed in Phase973/974",
        "split": {"discovery_n": len(discovery), "development_n": len(development),
                  "holdout_n": len(holdout), "exhaustive_scan_n": len(exhaustive),
                  "discovery_ids": [x["id"] for x in discovery],
                  "development_ids": [x["id"] for x in development],
                  "holdout_ids": [x["id"] for x in holdout]},
        "intervention": "same-length input-embedding zero/neutral replacement",
        "patch_point": "post-transformer-block residual, final valid position only",
    }
    if set(result["split"]["discovery_ids"]) & set(result["split"]["holdout_ids"]):
        raise RuntimeError("split overlap")

    # Save the actual tokenizer-specific manifest before any causal claims.
    result["protocol_manifest_example"] = protocol_manifest(tok, model_name,
                                                              discovery[0]["prompt"], True)
    log(f"Phase975 {model_name}: same-length protocol-group screen on 32 discovery items")
    raw, summary, selected, ranking = screen_protocol_groups(
        model_name, model, tok, layers, device, eos_ids, discovery)
    result["protocol_screen_rows"] = raw
    result["protocol_screen_summary"] = summary
    result["protocol_group_ranking"] = ranking
    result["selected_protocol_group"] = selected
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    group = selected["group"]
    log(f"Phase975 {model_name}: exhaustive bidirectional scan; frozen group={group}")
    scan = layer_scan(model_name, model, tok, layers, device, eos_ids, exhaustive, group)
    result["layer_scan"] = scan
    rescue_layer = scan["selected_rescue_layer"]
    transfer_layer = scan["selected_transfer_layer"]
    result["selected_layers"] = {"same_length_marker_rescue": rescue_layer,
                                 "plain_chat_transfer": transfer_layer,
                                 "selection_rule": "earliest layer with >=80% aggregate recovery and >=75% toward-source in both directions; final layer fallback"}
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Phase975 {model_name}: frozen dev evaluation Lrescue={rescue_layer}, Ltransfer={transfer_layer}")
    direction_d, n_d = mean_direction(model_name, model, tok, layers, device, eos_ids,
                                      discovery, transfer_layer)
    dev_rows, dev_summary, dev_templates = evaluate_frozen(
        model_name, model, tok, layers, device, eos_ids, development, group,
        rescue_layer, transfer_layer, direction_d, "development")
    result["frozen_direction_discovery_n"] = n_d
    result["development"] = {"summary": dev_summary, "by_template": dev_templates,
                             "rows": dev_rows}
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # The holdout is evaluated once with the already-frozen group/layers/rules.
    # D+Dev may estimate the single mean vector; no holdout activation contributes.
    direction_train, n_train = mean_direction(model_name, model, tok, layers, device,
                                              eos_ids, discovery + development, transfer_layer)
    torch.save({"model": model_name, "layer": transfer_layer, "n_train": n_train,
                "direction": direction_train.to(torch.float32)},
               OUT / f"{model_name}_frozen_direction.pt")
    log(f"Phase975 {model_name}: one-shot 96-item holdout")
    h_rows, h_summary, h_templates = evaluate_frozen(
        model_name, model, tok, layers, device, eos_ids, holdout, group,
        rescue_layer, transfer_layer, direction_train, "holdout")
    result["holdout"] = {"frozen_direction_train_n": n_train, "summary": h_summary,
                         "by_template": h_templates, "rows": h_rows}
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Natural necessity is run for the selected marker with both corruption controls.
    # DS is retained but interpreted only as an open-thinking trajectory.
    result["natural_marker_corruption"] = {}
    for method in ["clean", "zero", "neutral"]:
        result["natural_marker_corruption"][method] = natural_corruption(
            model_name, model, tok, layers, device, eos_ids, holdout, group, method)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    result["elapsed_seconds"] = time.time() - t0
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Saved {out_path}; elapsed={result['elapsed_seconds']/60:.1f} min")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "glm4")
