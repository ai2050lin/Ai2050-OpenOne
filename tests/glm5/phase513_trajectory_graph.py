#!/usr/bin/env python3
"""
Phase 513: Trajectory Graph and Path-Conditioned Semantic Readout
=================================================================
Phase 512 proved: Oracle logit patch (suppress all competitors) REDUCES hit.
This means language generation is a PATH problem, not a single-step selection.

Phase 513 investigates:
  Exp1: Multi-step trajectory graph (5-8 steps, top-k expansion)
  Exp2: Path-conditioned probability — P_hit(c | y_1) for each first-step token
  Exp3: Path hub token identification — "a"/"the"/"of" as bridges to category
  Exp4: Beam search vs greedy decoding comparison
  Exp5: Action category with natural templates

All models use BF16 + device_map="auto" with flash attention (sdpa).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase507_orthogonal_field import (  # noqa: E402
    ALL_CLASS,
    CATEGORIES,
    get_norm_g,
    get_token_ids,
    load_bf16_auto,
)
from phase508_orthogonal_field_basis_decomposition import (  # noqa: E402
    NEUTRAL_TEMPLATES,
    RICH_TEMPLATES,
    batched_hidden,
    cos,
    orthonormal_rows,
    project_remove_deltas,
    random_basis,
    score_logits,
    svd_basis,
    build_cat_meta,
    build_examples,
)
from phase509_rotation_stable_orthogonal_field import FOCUS_CATEGORIES  # noqa: E402


OUT_ROOT = Path("results/glm5_phase513_trajectory_graph")

# Generation templates
GEN_TEMPLATES = [
    ("category_of", "The {obj} belongs to the category of"),
    ("taxonomy_as", "In taxonomy, {obj} is classified as"),
    ("classify_colon", "Classify {obj}:"),
]

# More natural action templates (Exp5)
ACTION_TEMPLATES = [
    ("doing_what", "The person is doing"),
    ("activity_type", "This is an example of"),
    ("action_type", "The action of running is a type of"),
]

# Path hub tokens for Exp3 — reduced set for memory efficiency
HUB_TOKENS = [" a", " the", " of", " type", " kind", " category"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def token_ids(tokenizer: Any, words: list[str]) -> list[int]:
    ids: list[int] = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks:
            ids.append(int(toks[0]))
    return sorted(set(ids))


def build_generation_examples(
    cat: str, train_n: int, test_n: int, templates: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if templates is None:
        templates = GEN_TEMPLATES
    objs = CATEGORIES[cat]["objects"][train_n: train_n + test_n]
    rows = []
    for obj in objs:
        for tid, (name, tpl) in enumerate(templates):
            rows.append({
                "cat": cat,
                "obj": obj,
                "template_id": tid,
                "template_name": name,
                "prompt": tpl.format(obj=obj),
            })
    return rows


def build_token_groups(
    tokenizer: Any, cat: str, objects: list[str], competitor_cats: list[str]
) -> dict[str, list[int]]:
    """Build token ID groups for each competitor type."""
    return {
        "category": token_ids(tokenizer, [cat, " " + cat]),
        "competitor_category": token_ids(
            tokenizer, competitor_cats + [" " + c for c in competitor_cats]
        ),
        "punctuation": token_ids(tokenizer, [".", ",", ":", ";", "!", "?", "\n"]),
        "generic": token_ids(tokenizer, [" thing", " item", " type", " kind", " object",
                                         " entity", " one", " it", " that", " something",
                                         " of", " which", " and"]),
        "object_copy": token_ids(
            tokenizer, objects + [" " + o for o in objects]
        ),
        "hub": token_ids(tokenizer, HUB_TOKENS),
        "structure": token_ids(tokenizer, [" which", " what", " that", " of", " as", " to", " in",
                                           " is", " are", " was", " were"]),
    }


def contains_category(text: str, cat: str) -> float:
    return 1.0 if cat.lower() in text.lower() else 0.0


def classify_token(token_id: int, token_groups: dict[str, list[int]],
                   tokenizer: Any) -> str:
    """Classify a token into a group."""
    for gname, gids in token_groups.items():
        if token_id in gids:
            return gname
    # Check if it's a category-like word
    tok_text = tokenizer.decode([token_id]).strip().lower()
    return f"other({tok_text})" if tok_text else f"other(id={token_id})"


# ============================================================================
# Exp1: Multi-step trajectory graph with top-k expansion
# ============================================================================
def trajectory_graph_trace(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    examples: list[dict[str, Any]],
    token_groups: dict[str, list[int]],
    steps: int,
    top_k: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """
    Build a trajectory graph by tracing greedy + top-k alternatives at each step.

    For each prompt, we trace:
    - The greedy path (argmax at each step)
    - At step 1, expand to top-k alternatives, then trace greedy from each
    - Record all paths and whether they reach a category token
    """
    prompts = [x["prompt"] for x in examples]
    n = len(prompts)
    cat = examples[0]["cat"]
    cat_ids = set(token_groups.get("category", []))

    # === Step 1: Get top-k tokens at step 1 for each prompt ===
    log(f"    Getting step 1 top-{top_k} candidates...")
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length
    ).to(device)
    pos = enc["attention_mask"].sum(dim=1) - 1

    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            return_dict=True, use_cache=False
        )

    logits_s1 = out.logits[
        torch.arange(out.logits.shape[0], device=out.logits.device),
        pos.to(out.logits.device)
    ].clone().float()

    # Get top-k tokens at step 1 for each prompt
    topk_vals, topk_ids = logits_s1.topk(top_k, dim=1)  # [n, top_k]
    topk_probs = torch.softmax(topk_vals, dim=1)

    # Record step 1 info
    step1_info = []
    for i in range(n):
        row = []
        for j in range(top_k):
            tid = int(topk_ids[i, j].item())
            prob = float(topk_probs[i, j].item())
            group = classify_token(tid, token_groups, tokenizer)
            tok_text = tokenizer.decode([tid])
            row.append({
                "token_id": tid,
                "token_text": tok_text,
                "prob": prob,
                "group": group,
            })
        step1_info.append(row)

    # === Step 2+: For each top-k step-1 token, trace greedy path ===
    # We create n * top_k prompts by extending each with each top-k token
    # Then trace greedy for remaining steps
    all_extended_prompts_encoded = []
    # For each prompt, expand with each top-k candidate
    extended_input_ids_list = []
    extended_attn_list = []

    for i in range(n):
        for j in range(top_k):
            # Extend prompt i with top-k token j
            orig_ids = enc["input_ids"][i]  # [seq_len]
            orig_attn = enc["attention_mask"][i]  # [seq_len]
            new_id = topk_ids[i, j].item()
            ext_ids = torch.cat([orig_ids, torch.tensor([new_id], device=device)])
            ext_attn = torch.cat([orig_attn, torch.tensor([1], device=device)])
            extended_input_ids_list.append(ext_ids)
            extended_attn_list.append(ext_attn)

    # Pad all extended sequences
    max_ext_len = max(x.shape[0] for x in extended_input_ids_list)
    padded_ids = torch.full((n * top_k, max_ext_len), tokenizer.pad_token_id or 0,
                           device=device, dtype=torch.long)
    padded_attn = torch.zeros((n * top_k, max_ext_len), device=device, dtype=torch.long)
    for i, (ids, attn) in enumerate(zip(extended_input_ids_list, extended_attn_list)):
        offset = max_ext_len - ids.shape[0]
        padded_ids[i, offset:] = ids
        padded_attn[i, offset:] = attn

    # Trace greedy for steps 2..K
    cur_ids = padded_ids
    cur_attn = padded_attn
    path_tokens = [[int(topk_ids[i // top_k, i % top_k].item())]
                   for i in range(n * top_k)]  # Already has step 1 token

    step_metrics = []

    for step in range(1, steps):  # steps 2..K
        all_logits = []
        all_next = []

        for start in range(0, cur_ids.shape[0], batch_size):
            sub_ids = cur_ids[start:start + batch_size]
            sub_attn = cur_attn[start:start + batch_size]
            pos = sub_attn.sum(dim=1) - 1

            with torch.no_grad():
                out = model(
                    input_ids=sub_ids, attention_mask=sub_attn,
                    return_dict=True, use_cache=False
                )

            logits = out.logits[
                torch.arange(out.logits.shape[0], device=out.logits.device),
                pos.to(out.logits.device)
            ].clone().float()
            next_ids = logits.argmax(dim=1)
            all_logits.append(logits.detach())
            all_next.append(next_ids.detach())
            del out

        logits_step = torch.cat(all_logits, dim=0)
        next_step = torch.cat(all_next, dim=0)

        for i, tid in enumerate(next_step.detach().cpu().tolist()):
            path_tokens[i].append(int(tid))

        # Aggregate metrics
        step_metrics.append({
            "step": step + 1,
            "category_top1_rate": float(
                sum(1 for tid in next_step.detach().cpu().tolist()
                    if tid in cat_ids) / len(next_step)
            ),
        })

        # Extend for next step
        cur_ids = torch.cat([cur_ids, next_step[:, None].to(cur_ids.device)], dim=1)
        cur_attn = torch.cat(
            [cur_attn, torch.ones((cur_attn.shape[0], 1), device=cur_attn.device, dtype=cur_attn.dtype)],
            dim=1
        )
        del logits_step
        torch.cuda.empty_cache()

    # === Analyze paths ===
    # For each original prompt, check which paths reach category
    path_results = []
    for i in range(n):
        prompt_paths = []
        for j in range(top_k):
            idx = i * top_k + j
            tokens = path_tokens[idx]
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            hit = contains_category(decoded, cat)
            prompt_paths.append({
                "step1_token": step1_info[i][j],
                "path_tokens": tokens,
                "path_text": decoded,
                "hit": hit,
            })
        path_results.append(prompt_paths)

    # Compute P_hit(c | y_1) for each step-1 token type
    p_hit_by_group = defaultdict(list)
    for i in range(n):
        for j in range(top_k):
            group = step1_info[i][j]["group"]
            hit = path_results[i][j]["hit"]
            p_hit_by_group[group].append(hit)

    group_hit_rates = {}
    for group, hits in p_hit_by_group.items():
        group_hit_rates[group] = float(np.mean(hits))

    # Greedy path hit rate (j=0 for each prompt, since top-1 is argmax)
    greedy_hits = [path_results[i][0]["hit"] for i in range(n)]
    greedy_hit_rate = float(np.mean(greedy_hits))

    # Best-of-top-k hit rate (any of top-k paths reaches category)
    best_k_hits = [max(path_results[i][j]["hit"] for j in range(top_k))
                   for i in range(n)]
    best_k_hit_rate = float(np.mean(best_k_hits))

    del logits_s1, topk_vals, topk_ids, topk_probs, padded_ids, padded_attn
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "greedy_hit_rate": greedy_hit_rate,
        f"best_top{top_k}_hit_rate": best_k_hit_rate,
        "group_hit_rates": group_hit_rates,
        "step1_examples": step1_info[:5],  # Just first 5 examples
        "path_examples": path_results[:3],  # First 3 prompts' paths
        "step_metrics": step_metrics,
    }


# ============================================================================
# Exp2: Path-conditioned probability — P_hit(c | y_1)
# ============================================================================
def path_conditioned_probability(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    examples: list[dict[str, Any]],
    token_groups: dict[str, list[int]],
    first_step_candidates: list[str],
    steps: int,
    batch_size: int,
    max_length: int,
    n_sample: int,
) -> dict[str, Any]:
    """
    For each candidate first-step token, force it and trace what happens.

    P_hit(c | y_1) = probability of reaching category token within K steps
    when the first generated token is forced to y_1.
    """
    prompts = [x["prompt"] for x in examples]
    n = len(prompts)
    cat = examples[0]["cat"]
    cat_ids = set(token_groups.get("category", []))

    # Get token IDs for candidates
    candidate_ids = {}
    for cand in first_step_candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if ids:
            candidate_ids[cand] = ids[0]

    # Also add category token as candidate
    for cid in token_groups.get("category", []):
        text = tokenizer.decode([cid])
        if text.strip() not in candidate_ids:
            candidate_ids[text.strip()] = cid

    log(f"    Testing {len(candidate_ids)} first-step candidates")

    results = {}
    for cand_name, cand_id in candidate_ids.items():
        # Create forced-first-step prompts
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length
        ).to(device)

        # Force first step token
        forced_ids = torch.cat([enc["input_ids"],
                               torch.full((n, 1), cand_id, device=device, dtype=torch.long)], dim=1)
        forced_attn = torch.cat([enc["attention_mask"],
                                torch.ones((n, 1), device=device, dtype=torch.long)], dim=1)

        # Greedy trace for remaining steps
        cur_ids = forced_ids
        cur_attn = forced_attn
        path_tokens = [[cand_id] for _ in range(n)]

        for step in range(1, steps):
            all_next = []
            for start in range(0, cur_ids.shape[0], batch_size):
                sub_ids = cur_ids[start:start + batch_size]
                sub_attn = cur_attn[start:start + batch_size]

                with torch.no_grad():
                    out = model(
                        input_ids=sub_ids, attention_mask=sub_attn,
                        return_dict=True, use_cache=False
                    )

                pos = sub_attn.sum(dim=1) - 1
                logits = out.logits[
                    torch.arange(out.logits.shape[0], device=out.logits.device),
                    pos.to(out.logits.device)
                ]
                next_ids = logits.argmax(dim=1)
                all_next.append(next_ids.detach())
                del out

            next_step = torch.cat(all_next, dim=0)
            for i, tid in enumerate(next_step.detach().cpu().tolist()):
                path_tokens[i].append(int(tid))

            cur_ids = torch.cat([cur_ids, next_step[:, None].to(cur_ids.device)], dim=1)
            cur_attn = torch.cat(
                [cur_attn, torch.ones((cur_attn.shape[0], 1), device=cur_attn.device, dtype=cur_attn.dtype)],
                dim=1
            )
            torch.cuda.empty_cache()

        # Check hit
        decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in path_tokens]
        hits = [contains_category(d, cat) for d in decoded]
        hit_rate = float(np.mean(hits))

        # Also check step-by-step category appearance
        cat_at_step = [0] * steps
        for i in range(n):
            for s, tid in enumerate(path_tokens[i]):
                if tid in cat_ids:
                    cat_at_step[s] += 1
                    break

        results[cand_name] = {
            "token_id": cand_id,
            "hit_rate": hit_rate,
            "n_hit": sum(hits),
            "n_total": n,
            "cat_first_appear_step": [c / max(n, 1) for c in cat_at_step],
            "example_outputs": decoded[:5],
        }
        log(f"      {cand_name}: P_hit={hit_rate:.3f} ({sum(hits)}/{n})")

    # Sort by hit rate
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["hit_rate"])

    return {
        "forced_first_step_results": dict(sorted_results),
        "best_forced_token": sorted_results[0][0] if sorted_results else None,
        "best_forced_hit_rate": sorted_results[0][1]["hit_rate"] if sorted_results else 0,
    }


# ============================================================================
# Exp3: Path hub token analysis — remove/force hub tokens
# ============================================================================
def hub_token_analysis(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    examples: list[dict[str, Any]],
    token_groups: dict[str, list[int]],
    hub_ids: list[int],
    steps: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """
    Analyze the role of hub tokens (a, the, of, etc.)

    For each hub token, test:
    1. Force the hub token at step 1, then trace greedy
    2. Suppress the hub token (logit patch -20) at step 1, then trace greedy
    3. Compare hit rates
    """
    prompts = [x["prompt"] for x in examples]
    n = len(prompts)
    cat = examples[0]["cat"]
    cat_ids = set(token_groups.get("category", []))

    # Baseline: clean greedy trace
    clean_trace = _greedy_trace(
        model, tokenizer, device, prompts, steps, batch_size, max_length,
        token_groups, cat_name=cat
    )

    results = {
        "clean": {
            "hit_rate": clean_trace["hit_rate"],
            "step1_top_tokens": clean_trace["step1_top_tokens"],
            "step1_cat_top1_rate": clean_trace["step1_cat_top1_rate"],
        }
    }

    # For each hub token, force it at step 1
    for hub_id in hub_ids:
        hub_text = tokenizer.decode([hub_id])
        log(f"    Testing hub: '{hub_text}' (id={hub_id})")

        # Force hub at step 1
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length
        ).to(device)

        forced_ids = torch.cat([enc["input_ids"],
                               torch.full((n, 1), hub_id, device=device, dtype=torch.long)], dim=1)
        forced_attn = torch.cat([enc["attention_mask"],
                                torch.ones((n, 1), device=device, dtype=torch.long)], dim=1)

        forced_trace = _greedy_trace_from_state(
            model, tokenizer, device, forced_ids, forced_attn, steps - 1,
            batch_size, token_groups, prefix_tokens=[[hub_id] for _ in range(n)],
            cat_name=cat
        )

        results[f"force_{hub_text.strip()}"] = {
            "hit_rate": forced_trace["hit_rate"],
            "step1_token": hub_text,
            "example_outputs": forced_trace["decoded"][:5],
        }

    # Suppress hub tokens at step 1 (logit patch)
    suppress_trace = _greedy_trace_with_logit_patch(
        model, tokenizer, device, prompts, steps, batch_size, max_length,
        token_groups, {tid: 20.0 for tid in hub_ids}, cat_name=cat
    )

    results["suppress_all_hubs"] = {
        "hit_rate": suppress_trace["hit_rate"],
        "step1_top_tokens": suppress_trace["step1_top_tokens"],
        "step1_cat_top1_rate": suppress_trace["step1_cat_top1_rate"],
    }

    del clean_trace
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================================
# Exp4: Beam search vs greedy comparison
# ============================================================================
def beam_vs_greedy(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    examples: list[dict[str, Any]],
    token_groups: dict[str, list[int]],
    steps: int,
    beam_width: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """
    Compare beam search vs greedy decoding for category hit rate.

    Uses model.generate() for beam search and custom greedy loop.
    """
    prompts = [x["prompt"] for x in examples]
    n = len(prompts)
    cat = examples[0]["cat"]

    # === Greedy ===
    greedy_results = _greedy_trace(
        model, tokenizer, device, prompts, steps, batch_size, max_length,
        token_groups, cat_name=cat
    )

    # === Beam search ===
    input_device = next(model.parameters()).device
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length
    ).to(input_device)

    with torch.no_grad():
        gen_ids = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=steps,
            num_beams=beam_width,
            num_return_sequences=1,
            do_sample=False,
            early_stopping=False,
            use_cache=True,
        )

    beam_decoded = [tokenizer.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                   for g in gen_ids]
    beam_hits = [contains_category(d, cat) for d in beam_decoded]
    beam_hit_rate = float(np.mean(beam_hits))

    log(f"    Greedy hit: {greedy_results['hit_rate']:.3f}, "
        f"Beam-{beam_width} hit: {beam_hit_rate:.3f}")

    return {
        "greedy": {
            "hit_rate": greedy_results["hit_rate"],
            "example_outputs": greedy_results["decoded"][:5],
        },
        f"beam_{beam_width}": {
            "hit_rate": beam_hit_rate,
            "example_outputs": beam_decoded[:5],
        },
        "beam_improvement": beam_hit_rate - greedy_results["hit_rate"],
    }


# ============================================================================
# Helper functions
# ============================================================================
def _greedy_trace(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    steps: int,
    batch_size: int,
    max_length: int,
    token_groups: dict[str, list[int]],
    cat_name: str = "",
) -> dict[str, Any]:
    """Standard greedy decoding trace."""
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length
    ).to(device)

    cur_ids = enc["input_ids"]
    cur_attn = enc["attention_mask"]
    cat_ids = set(token_groups.get("category", []))
    n = len(prompts)
    path_tokens = [[] for _ in range(n)]
    step1_top_tokens = Counter()
    step1_cat_top1 = 0

    for step in range(steps):
        all_next = []
        for start in range(0, cur_ids.shape[0], batch_size):
            sub_ids = cur_ids[start:start + batch_size]
            sub_attn = cur_attn[start:start + batch_size]
            pos = sub_attn.sum(dim=1) - 1

            with torch.no_grad():
                out = model(
                    input_ids=sub_ids, attention_mask=sub_attn,
                    return_dict=True, use_cache=False
                )

            logits = out.logits[
                torch.arange(out.logits.shape[0], device=out.logits.device),
                pos.to(out.logits.device)
            ]
            next_ids = logits.argmax(dim=1)
            all_next.append(next_ids.detach())
            del out

        next_step = torch.cat(all_next, dim=0)
        for i, tid in enumerate(next_step.detach().cpu().tolist()):
            path_tokens[i].append(int(tid))
            if step == 0:
                tok_text = tokenizer.decode([tid])
                step1_top_tokens[tok_text] += 1
                if tid in cat_ids:
                    step1_cat_top1 += 1

        cur_ids = torch.cat([cur_ids, next_step[:, None].to(cur_ids.device)], dim=1)
        cur_attn = torch.cat(
            [cur_attn, torch.ones((cur_attn.shape[0], 1), device=cur_attn.device, dtype=cur_attn.dtype)],
            dim=1
        )
        torch.cuda.empty_cache()

    decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in path_tokens]
    if not cat_name:
        cat_ids_list = list(token_groups.get("category", []))
        cat_name = tokenizer.decode(cat_ids_list).strip().split()[0] if cat_ids_list else ""
    hits = [contains_category(d, cat_name) for d in decoded]

    return {
        "hit_rate": float(np.mean(hits)),
        "decoded": decoded,
        "path_tokens": path_tokens,
        "step1_top_tokens": dict(step1_top_tokens.most_common(10)),
        "step1_cat_top1_rate": step1_cat_top1 / max(n, 1),
    }


def _greedy_trace_from_state(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    steps: int,
    batch_size: int,
    token_groups: dict[str, list[int]],
    prefix_tokens: list[list[int]] | None = None,
    cat_name: str = "",
) -> dict[str, Any]:
    """Greedy trace starting from a given state."""
    n = input_ids.shape[0]
    cat_ids = set(token_groups.get("category", []))
    path_tokens = list(prefix_tokens) if prefix_tokens else [[] for _ in range(n)]

    cur_ids = input_ids
    cur_attn = attention_mask

    for step in range(steps):
        all_next = []
        for start in range(0, cur_ids.shape[0], batch_size):
            sub_ids = cur_ids[start:start + batch_size]
            sub_attn = cur_attn[start:start + batch_size]
            pos = sub_attn.sum(dim=1) - 1

            with torch.no_grad():
                out = model(
                    input_ids=sub_ids, attention_mask=sub_attn,
                    return_dict=True, use_cache=False
                )

            logits = out.logits[
                torch.arange(out.logits.shape[0], device=out.logits.device),
                pos.to(out.logits.device)
            ]
            next_ids = logits.argmax(dim=1)
            all_next.append(next_ids.detach())
            del out

        next_step = torch.cat(all_next, dim=0)
        for i, tid in enumerate(next_step.detach().cpu().tolist()):
            path_tokens[i].append(int(tid))

        cur_ids = torch.cat([cur_ids, next_step[:, None].to(cur_ids.device)], dim=1)
        cur_attn = torch.cat(
            [cur_attn, torch.ones((cur_attn.shape[0], 1), device=cur_attn.device, dtype=cur_attn.dtype)],
            dim=1
        )
        torch.cuda.empty_cache()

    decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in path_tokens]
    if not cat_name:
        cat_ids_list = list(token_groups.get("category", []))
        cat_name = tokenizer.decode(cat_ids_list).strip().split()[0] if cat_ids_list else ""
    hits = [contains_category(d, cat_name) for d in decoded]

    return {
        "hit_rate": float(np.mean(hits)),
        "decoded": decoded,
        "path_tokens": path_tokens,
    }


def _greedy_trace_with_logit_patch(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    steps: int,
    batch_size: int,
    max_length: int,
    token_groups: dict[str, list[int]],
    suppress_ids: dict[int, float],  # token_id -> suppression value
    cat_name: str = "",
) -> dict[str, Any]:
    """Greedy trace with logit suppression."""
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length
    ).to(device)

    cur_ids = enc["input_ids"]
    cur_attn = enc["attention_mask"]
    cat_ids = set(token_groups.get("category", []))
    n = len(prompts)
    path_tokens = [[] for _ in range(n)]
    step1_top_tokens = Counter()
    step1_cat_top1 = 0

    for step in range(steps):
        all_logits = []
        all_next = []
        for start in range(0, cur_ids.shape[0], batch_size):
            sub_ids = cur_ids[start:start + batch_size]
            sub_attn = cur_attn[start:start + batch_size]
            pos = sub_attn.sum(dim=1) - 1

            with torch.no_grad():
                out = model(
                    input_ids=sub_ids, attention_mask=sub_attn,
                    return_dict=True, use_cache=False
                )

            logits = out.logits[
                torch.arange(out.logits.shape[0], device=out.logits.device),
                pos.to(out.logits.device)
            ].clone().float()

            # Apply logit patch
            for tid, val in suppress_ids.items():
                if 0 <= tid < logits.shape[-1]:
                    logits[:, tid] -= val

            next_ids = logits.argmax(dim=1)
            all_logits.append(logits.detach())
            all_next.append(next_ids.detach())
            del out

        logits_step = torch.cat(all_logits, dim=0)
        next_step = torch.cat(all_next, dim=0)

        for i, tid in enumerate(next_step.detach().cpu().tolist()):
            path_tokens[i].append(int(tid))
            if step == 0:
                tok_text = tokenizer.decode([tid])
                step1_top_tokens[tok_text] += 1
                if tid in cat_ids:
                    step1_cat_top1 += 1

        cur_ids = torch.cat([cur_ids, next_step[:, None].to(cur_ids.device)], dim=1)
        cur_attn = torch.cat(
            [cur_attn, torch.ones((cur_attn.shape[0], 1), device=cur_attn.device, dtype=cur_attn.dtype)],
            dim=1
        )
        del logits_step
        torch.cuda.empty_cache()

    decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in path_tokens]
    if not cat_name:
        cat_ids_list = list(token_groups.get("category", []))
        cat_name = tokenizer.decode(cat_ids_list).strip().split()[0] if cat_ids_list else ""
    hits = [contains_category(d, cat_name) for d in decoded]

    return {
        "hit_rate": float(np.mean(hits)),
        "decoded": decoded,
        "path_tokens": path_tokens,
        "step1_top_tokens": dict(step1_top_tokens.most_common(10)),
        "step1_cat_top1_rate": step1_cat_top1 / max(n, 1),
    }


# ============================================================================
# Exp5: Action category with natural templates
# ============================================================================
def action_natural_templates(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    examples: list[dict[str, Any]],
    token_groups: dict[str, list[int]],
    steps: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """Test action category with natural action-oriented templates."""
    prompts = [x["prompt"] for x in examples]
    n = len(prompts)
    cat = examples[0]["cat"]

    # Standard templates
    standard_trace = _greedy_trace(
        model, tokenizer, device, prompts, steps, batch_size, max_length,
        token_groups, cat_name=cat
    )

    # Natural action templates
    action_prompts = []
    for ex in examples:
        obj = ex["obj"]
        action_prompts.extend([
            f"The person is doing {obj}",
            f"This activity is called {obj}",
            f"The action of {obj} is a type of",
        ])

    # Build token groups for action context
    action_token_groups = dict(token_groups)
    # Add verb tokens that might be natural action completions
    action_token_groups["action_verbs"] = token_ids(
        tokenizer, [" running", " walking", " jumping", " swimming", " eating",
                     " singing", " dancing", " writing", " reading", " playing"]
    )

    action_trace = _greedy_trace(
        model, tokenizer, device, action_prompts, steps, batch_size, max_length,
        action_token_groups, cat_name=cat
    )

    # Check verb hit rate for action prompts
    verb_ids = set(action_token_groups.get("action_verbs", []))
    verb_hits = sum(1 for tokens in action_trace["path_tokens"]
                   if any(t in verb_ids for t in tokens))
    verb_hit_rate = verb_hits / max(len(action_trace["path_tokens"]), 1)

    return {
        "standard_templates": {
            "hit_rate": standard_trace["hit_rate"],
            "example_outputs": standard_trace["decoded"][:5],
        },
        "natural_action_templates": {
            "hit_rate": action_trace["hit_rate"],
            "verb_hit_rate": verb_hit_rate,
            "example_outputs": action_trace["decoded"][:5],
        },
    }


# ============================================================================
# Main runner
# ============================================================================
def run_category(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    cat: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run all Phase 513 experiments for a single category."""
    is_action = (cat == "action")
    templates = ACTION_TEMPLATES if is_action else GEN_TEMPLATES

    log(f"  Building examples for {cat} (action={is_action})...")
    gen_examples = build_generation_examples(
        cat, args.train_objects, args.test_objects, templates
    )

    # Token groups
    objects = [x["obj"] for x in gen_examples]
    competitor_cats = [c for c in CATEGORIES if c != cat]
    token_groups = build_token_groups(tokenizer, cat, objects, competitor_cats)

    # Hub token IDs
    hub_id_list = token_ids(tokenizer, HUB_TOKENS)

    results = {}

    # ===== Exp1: Trajectory graph =====
    log(f"  Exp1: Trajectory graph for {cat} (top-{args.top_k}, {args.steps} steps)...")
    try:
        traj_result = trajectory_graph_trace(
            model, tokenizer, device, gen_examples, token_groups,
            args.steps, args.top_k, args.batch_size, args.max_length
        )
        results["trajectory_graph"] = traj_result
        log(f"    Greedy hit: {traj_result['greedy_hit_rate']:.3f}, "
            f"Best-top{k_name(args.top_k)}: {traj_result[f'best_top{args.top_k}_hit_rate']:.3f}")
        log(f"    Group hit rates: {traj_result['group_hit_rates']}")
    except Exception as e:
        log(f"    Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["trajectory_graph"] = {"error": str(e)}

    # ===== Exp2: Path-conditioned probability =====
    log(f"  Exp2: Path-conditioned probability for {cat}...")
    first_step_candidates = [" a", " the", " of", " type", " kind", " category"] + [f" {cat}"]
    # Remove duplicates
    first_step_candidates = list(dict.fromkeys(first_step_candidates))

    try:
        pcp_result = path_conditioned_probability(
            model, tokenizer, device, gen_examples, token_groups,
            first_step_candidates, args.steps, args.batch_size, args.max_length,
            args.test_objects * len(templates)
        )
        results["path_conditioned_prob"] = pcp_result
        log(f"    Best forced first-step: '{pcp_result['best_forced_token']}' "
            f"(P_hit={pcp_result['best_forced_hit_rate']:.3f})")
    except Exception as e:
        log(f"    Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["path_conditioned_prob"] = {"error": str(e)}

    # ===== Exp3: Hub token analysis =====
    log(f"  Exp3: Hub token analysis for {cat}...")
    try:
        hub_result = hub_token_analysis(
            model, tokenizer, device, gen_examples, token_groups,
            hub_id_list, args.steps, args.batch_size, args.max_length
        )
        results["hub_analysis"] = hub_result
        clean_hit = hub_result["clean"]["hit_rate"]
        suppress_hit = hub_result["suppress_all_hubs"]["hit_rate"]
        log(f"    Clean hit: {clean_hit:.3f}, Suppress hubs hit: {suppress_hit:.3f}")
        # Log forced hub results
        for key, val in hub_result.items():
            if key.startswith("force_"):
                log(f"    Force '{key}': hit={val['hit_rate']:.3f}")
    except Exception as e:
        log(f"    Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["hub_analysis"] = {"error": str(e)}

    # ===== Exp4: Beam vs greedy =====
    log(f"  Exp4: Beam search vs greedy for {cat}...")
    try:
        beam_result = beam_vs_greedy(
            model, tokenizer, device, gen_examples, token_groups,
            args.steps, args.beam_width, args.batch_size, args.max_length
        )
        results["beam_vs_greedy"] = beam_result
        log(f"    Greedy: {beam_result['greedy']['hit_rate']:.3f}, "
            f"Beam-{args.beam_width}: {beam_result[f'beam_{args.beam_width}']['hit_rate']:.3f}, "
            f"Improvement: {beam_result['beam_improvement']:+.3f}")
    except Exception as e:
        log(f"    Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["beam_vs_greedy"] = {"error": str(e)}

    # ===== Exp5: Action natural templates =====
    if is_action:
        log(f"  Exp5: Action natural templates for {cat}...")
        try:
            action_result = action_natural_templates(
                model, tokenizer, device, gen_examples, token_groups,
                args.steps, args.batch_size, args.max_length
            )
            results["action_templates"] = action_result
        except Exception as e:
            log(f"    Exp5 FAILED: {e}")
            import traceback; traceback.print_exc()
            results["action_templates"] = {"error": str(e)}

    return {
        "n_generation_prompts": len(gen_examples),
        "is_action": is_action,
        "templates": [t[0] for t in templates],
        "token_group_sizes": {k: len(v) for k, v in token_groups.items()},
        **results,
    }


def k_name(k: int) -> str:
    return str(k)


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_bf16_auto(args.model)
    try:
        info = get_model_info(model, args.model)
        L, d = info.n_layers, info.d_model

        categories = (
            args.categories.split(",")
            if args.categories
            else FOCUS_CATEGORIES[args.model]
        )
        log(f"{args.model}: L={L}, d={d}, categories={categories}")

        result = {
            "phase": 513,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "L": L,
            "d_model": d,
            "categories": categories,
            "train_objects": args.train_objects,
            "test_objects": args.test_objects,
            "steps": args.steps,
            "top_k": args.top_k,
            "beam_width": args.beam_width,
            "category_results": {},
        }

        for ci, cat in enumerate(categories, 1):
            log(f"{args.model}: category {ci}/{len(categories)} {cat}")
            cat_out = run_category(model, tokenizer, device, cat, args)
            result["category_results"][cat] = cat_out

            # Print summary
            traj = cat_out.get("trajectory_graph", {})
            pcp = cat_out.get("path_conditioned_prob", {})
            hub = cat_out.get("hub_analysis", {})
            beam = cat_out.get("beam_vs_greedy", {})
            log(f"  {cat} summary: "
                f"greedy_hit={traj.get('greedy_hit_rate', 'N/A')}, "
                f"best_topk_hit={traj.get(f'best_top{args.top_k}_hit_rate', 'N/A')}, "
                f"beam_improvement={beam.get('beam_improvement', 'N/A')}, "
                f"best_forced='{pcp.get('best_forced_token', 'N/A')}'")

        return result
    finally:
        release_model(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=20)
    parser.add_argument("--test-objects", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    path = out_dir / f"phase513_{args.model}_trajectory_graph.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
