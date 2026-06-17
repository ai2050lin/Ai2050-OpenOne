#!/usr/bin/env python3
"""
Phase 514: Path Value Function and Hidden-State Trajectory Control
====================================================================
Phase 513 proved: hub tokens are bridges, not obstacles; greedy doesn't
find the best path. Now we build the path value function and test
hidden-state trajectory control.

Core experiments:
  Exp1: Path value function V_c(y_1) + logit vs value rank comparison
  Exp2: Hub hidden state analysis — how hubs affect category logit
  Exp3: Path-value guided intervention — add_support + force best hub
  Exp4: Action natural templates with real verb targets

Memory-efficient: limited candidates, smaller batches for GLM4/DS7B.
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


OUT_ROOT = Path("results/glm5_phase514_path_value")

# Generation templates
GEN_TEMPLATES = [
    ("category_of", "The {obj} belongs to the category of"),
    ("taxonomy_as", "In taxonomy, {obj} is classified as"),
    ("classify_colon", "Classify {obj}:"),
]

# Natural action templates
ACTION_TEMPLATES = [
    ("doing_what", "The person is doing"),
    ("activity_type", "This is an example of"),
    ("action_type", "The action of running is a type of"),
]

# Hub tokens — keep small for memory efficiency
HUB_TOKENS = [" a", " the", " of", " type", " kind"]

# First-step candidates for path value function — minimal set for GLM4/DS7B
FIRST_STEP_CANDIDATES = [" a", " the", " kind", " type"]  # + category token added dynamically

# Action verb targets for Exp4
ACTION_VERB_TARGETS = [" running", " walking", " jumping", " swimming", " eating",
                       " singing", " dancing", " writing", " reading", " playing"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tok_ids(tokenizer: Any, words: list[str]) -> list[int]:
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
    return {
        "category": tok_ids(tokenizer, [cat, " " + cat]),
        "competitor_category": tok_ids(
            tokenizer, competitor_cats + [" " + c for c in competitor_cats]
        ),
        "punctuation": tok_ids(tokenizer, [".", ",", ":", ";", "!", "?", "\n"]),
        "generic": tok_ids(tokenizer, [" thing", " item", " type", " kind", " object",
                                        " entity", " one", " it", " that", " something"]),
        "object_copy": tok_ids(tokenizer, objects + [" " + o for o in objects]),
        "hub": tok_ids(tokenizer, HUB_TOKENS),
        "structure": tok_ids(tokenizer, [" which", " what", " that", " of", " as", " to",
                                          " is", " are", " was", " were"]),
    }


def contains_category(text: str, cat: str) -> float:
    return 1.0 if cat.lower() in text.lower() else 0.0


def classify_hit_quality(decoded: str, cat: str, path_tokens: list[int],
                         tokenizer: Any, cat_ids: set[int]) -> str:
    """Classify hit quality: miss/lexical/natural_phrase/semantic_answer."""
    if cat.lower() not in decoded.lower():
        return "miss"

    cat_positions = [i for i, t in enumerate(path_tokens) if t in cat_ids]
    if not cat_positions:
        return "lexical"

    cat_pos = cat_positions[0]
    if cat_pos > 0:
        prev_tok = tokenizer.decode([path_tokens[cat_pos - 1]]).strip().lower()
        if prev_tok in ["a", "an", "the", "type", "kind", "of"]:
            cat_count = sum(1 for t in path_tokens if t in cat_ids)
            return "semantic_answer" if cat_count <= 2 else "natural_phrase"

    return "lexical"


def classify_token(token_id: int, token_groups: dict[str, list[int]],
                   tokenizer: Any) -> str:
    for gname, gids in token_groups.items():
        if token_id in gids:
            return gname
    tok_text = tokenizer.decode([token_id]).strip().lower()
    return f"other({tok_text})" if tok_text else f"other(id={token_id})"


# ============================================================================
# Exp1: Path Value Function + Logit vs Value Rank
# ============================================================================
def path_value_and_rank(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    examples: list[dict[str, Any]],
    token_groups: dict[str, list[int]],
    steps: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """
    Estimate V_c(y_1) for key candidates and compare logit rank vs value rank.
    """
    prompts = [x["prompt"] for x in examples]
    n = len(prompts)
    cat = examples[0]["cat"]
    cat_ids = set(token_groups.get("category", []))

    # Step 1: Get logits at last position
    log(f"    Getting step-1 logits for {n} prompts...")
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

    # Get top-10 candidates per prompt for logit rank comparison
    top_k = 10
    topk_vals, topk_ids = logits_s1.topk(top_k, dim=1)
    topk_probs = torch.softmax(topk_vals, dim=1)

    del out
    torch.cuda.empty_cache()

    # Collect candidate token IDs: hub tokens + category tokens only
    candidate_ids = set()
    for cid in tok_ids(tokenizer, FIRST_STEP_CANDIDATES):
        candidate_ids.add(cid)
    for cid in cat_ids:
        candidate_ids.add(cid)

    candidate_ids = sorted(candidate_ids)
    log(f"    Testing {len(candidate_ids)} first-step candidates")

    # For each candidate, force it at step 1 and trace
    path_values = {}
    for ci, cand_id in enumerate(candidate_ids):
        cand_text = tokenizer.decode([cand_id])
        cand_group = classify_token(cand_id, token_groups, tokenizer)

        # Force this token and trace
        enc2 = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length
        ).to(device)
        forced_ids = torch.cat([enc2["input_ids"],
                                torch.full((n, 1), cand_id, device=device, dtype=torch.long)], dim=1)
        forced_attn = torch.cat([enc2["attention_mask"],
                                 torch.ones((n, 1), device=device, dtype=torch.long)], dim=1)

        cur_ids = forced_ids
        cur_attn = forced_attn
        path_tokens_list = [[cand_id] for _ in range(n)]

        for step in range(1, steps):
            all_next = []
            for start in range(0, cur_ids.shape[0], batch_size):
                sub_ids = cur_ids[start:start + batch_size]
                sub_attn = cur_attn[start:start + batch_size]

                with torch.no_grad():
                    o = model(
                        input_ids=sub_ids, attention_mask=sub_attn,
                        return_dict=True, use_cache=False
                    )

                p = sub_attn.sum(dim=1) - 1
                lg = o.logits[
                    torch.arange(o.logits.shape[0], device=o.logits.device),
                    p.to(o.logits.device)
                ]
                nid = lg.argmax(dim=1)
                all_next.append(nid.detach())
                del o

            next_step = torch.cat(all_next, dim=0)
            for i, tid in enumerate(next_step.detach().cpu().tolist()):
                path_tokens_list[i].append(int(tid))

            cur_ids = torch.cat([cur_ids, next_step[:, None].to(cur_ids.device)], dim=1)
            cur_attn = torch.cat(
                [cur_attn, torch.ones((cur_attn.shape[0], 1), device=cur_attn.device, dtype=cur_attn.dtype)],
                dim=1
            )
            torch.cuda.empty_cache()

        # Evaluate with quality classification
        qualities = []
        for i in range(n):
            decoded = tokenizer.decode(path_tokens_list[i], skip_special_tokens=True)
            quality = classify_hit_quality(decoded, cat, path_tokens_list[i], tokenizer, cat_ids)
            qualities.append(quality)

        lexical_rate = sum(1 for q in qualities if q != "miss") / n
        natural_rate = sum(1 for q in qualities if q in ("natural_phrase", "semantic_answer")) / n
        semantic_rate = sum(1 for q in qualities if q == "semantic_answer") / n

        # Average logit rank for this candidate
        ranks = []
        for i in range(n):
            sorted_ids = logits_s1[i].argsort(descending=True)
            rank = (sorted_ids == cand_id).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                ranks.append(int(rank[0].item()))
        avg_rank = float(np.mean(ranks)) if ranks else 999.0

        # Average logit value
        avg_logit = 0.0
        avg_prob = 0.0
        count = 0
        for i in range(n):
            if cand_id in topk_ids[i].tolist():
                idx = topk_ids[i].tolist().index(cand_id)
                avg_logit += float(topk_vals[i, idx].item())
                avg_prob += float(topk_probs[i, idx].item())
                count += 1
        if count > 0:
            avg_logit /= count
            avg_prob /= count

        path_values[cand_text.strip() or f"id={cand_id}"] = {
            "token_id": cand_id,
            "group": cand_group,
            "V_c_lexical": round(lexical_rate, 4),
            "V_c_natural": round(natural_rate, 4),
            "V_c_semantic": round(semantic_rate, 4),
            "avg_logit_rank": round(avg_rank, 1),
            "avg_logit": round(avg_logit, 3),
            "avg_prob": round(avg_prob, 5),
            "quality_dist": dict(Counter(qualities)),
        }

        log(f"      {cand_text.strip() or f'id={cand_id}'}: V_sem={semantic_rate:.3f}, "
            f"V_lex={lexical_rate:.3f}, rank={avg_rank:.0f}")

        del enc2, forced_ids, forced_attn, cur_ids, cur_attn
        gc.collect()
        torch.cuda.empty_cache()

    # Sort by semantic hit rate
    sorted_pv = sorted(path_values.items(), key=lambda x: -x[1]["V_c_semantic"])

    # Group-level summary
    group_values = defaultdict(list)
    for name, pv in path_values.items():
        group_values[pv["group"]].append(pv["V_c_semantic"])
    group_summary = {g: round(float(np.mean(v)), 4) for g, v in group_values.items()}

    # Logit rank vs value rank comparison
    # For each prompt, find the greedy choice and the best-value choice
    logit_vs_value = []
    for i in range(n):
        greedy_id = int(topk_ids[i, 0].item())
        greedy_group = classify_token(greedy_id, token_groups, tokenizer)
        greedy_text = tokenizer.decode([greedy_id])

        # Find best value among candidates for this prompt
        best_value_id = None
        best_value_score = -1
        for cand_id in candidate_ids:
            pv_name = tokenizer.decode([cand_id]).strip() or f"id={cand_id}"
            if pv_name in path_values:
                pv = path_values[pv_name]
                if pv["V_c_semantic"] > best_value_score:
                    best_value_score = pv["V_c_semantic"]
                    best_value_id = cand_id

        logit_vs_value.append({
            "greedy_id": greedy_id,
            "greedy_text": greedy_text.strip(),
            "greedy_group": greedy_group,
            "best_value_id": best_value_id,
            "best_value_score": best_value_score,
        })

    # Count how often greedy matches best value
    match_count = sum(1 for lv in logit_vs_value
                      if lv["greedy_id"] == lv["best_value_id"])

    del logits_s1, topk_vals, topk_ids, topk_probs
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "path_values": dict(sorted_pv),
        "group_summary": group_summary,
        "logit_vs_value": {
            "match_rate": round(match_count / max(n, 1), 4),
            "per_prompt_summary": logit_vs_value[:10],
        },
        "steps": steps,
        "n_prompts": n,
    }


# ============================================================================
# Exp2: Hub Hidden State Analysis
# ============================================================================
def hub_hidden_state_analysis(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    examples: list[dict[str, Any]],
    token_groups: dict[str, list[int]],
    hub_ids: list[int],
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """
    Analyze how hub tokens affect category logit at the next position.
    """
    cat = examples[0]["cat"]
    prompts = [x["prompt"] for x in examples]
    n = len(prompts)
    cat_ids_list = list(token_groups.get("category", []))
    info = get_model_info(model, "qwen3")  # approximate

    # Get category logit WITHOUT hub (baseline)
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            return_dict=True, use_cache=False
        )

    pos = enc["attention_mask"].sum(dim=1) - 1
    logits_base = out.logits[
        torch.arange(out.logits.shape[0], device=out.logits.device),
        pos.to(out.logits.device)
    ].clone().float()

    cat_logits_base = torch.stack([logits_base[:, cid] for cid in cat_ids_list]).max(dim=0)[0]

    del out, logits_base
    torch.cuda.empty_cache()

    # For each hub token, force it and get the category logit at the NEXT position
    hub_effects = {}
    for hub_id in hub_ids:
        hub_text = tokenizer.decode([hub_id])
        log(f"      Hub: '{hub_text}' (id={hub_id})")

        # Create prompts with hub appended
        enc2 = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length
        ).to(device)
        forced_ids = torch.cat([enc2["input_ids"],
                                torch.full((n, 1), hub_id, device=device, dtype=torch.long)], dim=1)
        forced_attn = torch.cat([enc2["attention_mask"],
                                 torch.ones((n, 1), device=device, dtype=torch.long)], dim=1)

        with torch.no_grad():
            out2 = model(
                input_ids=forced_ids, attention_mask=forced_attn,
                return_dict=True, use_cache=False
            )

        pos2 = forced_attn.sum(dim=1) - 1
        logits_hub = out2.logits[
            torch.arange(out2.logits.shape[0], device=out2.logits.device),
            pos2.to(out2.logits.device)
        ].clone().float()

        cat_logits_hub = torch.stack([logits_hub[:, cid] for cid in cat_ids_list]).max(dim=0)[0]

        delta = (cat_logits_hub - cat_logits_base).mean().item()

        hub_effects[hub_text.strip()] = {
            "cat_logit_base": round(float(cat_logits_base.mean()), 3),
            "cat_logit_after_hub": round(float(cat_logits_hub.mean()), 3),
            "cat_logit_delta": round(delta, 3),
            "cat_logit_delta_std": round(float((cat_logits_hub - cat_logits_base).std()), 3),
        }

        log(f"        cat_logit: {cat_logits_base.mean():.2f} → {cat_logits_hub.mean():.2f} (Δ={delta:+.3f})")

        del enc2, forced_ids, forced_attn, out2, logits_hub, cat_logits_hub
        gc.collect()
        torch.cuda.empty_cache()

    # Clean up
    del enc, cat_logits_base
    gc.collect()
    torch.cuda.empty_cache()

    return {"hub_effects": hub_effects}


# ============================================================================
# Exp3: Path-Value Guided Intervention
# ============================================================================
def path_value_guided_intervention(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    examples: list[dict[str, Any]],
    token_groups: dict[str, list[int]],
    steps: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """
    Test different interventions:
    A. Clean (baseline)
    B. Force best hub token at step 1
    C. Boost category logit by +5 at step 1
    D. Force best hub + boost category logit
    """
    cat = examples[0]["cat"]
    prompts = [x["prompt"] for x in examples]
    n = len(prompts)
    cat_ids = set(token_groups.get("category", []))
    cat_ids_list = list(cat_ids)
    hub_id_list = tok_ids(tokenizer, HUB_TOKENS[:3])  # Only test top 3 hubs for memory

    results = {}

    # A. Clean baseline
    log(f"    A. Clean baseline...")
    clean_result = _greedy_trace_q(
        model, tokenizer, device, prompts, steps, batch_size, max_length,
        token_groups, cat
    )
    results["clean"] = clean_result

    # B. Force best hub at step 1
    log(f"    B. Testing each hub at step 1...")
    best_hub_name = ""
    best_hub_sem = -1
    best_hub_result = None
    best_hub_id = None

    for hub_id in hub_id_list:
        hub_text = tokenizer.decode([hub_id]).strip()
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length
        ).to(device)
        forced_ids = torch.cat([enc["input_ids"],
                                torch.full((n, 1), hub_id, device=device, dtype=torch.long)], dim=1)
        forced_attn = torch.cat([enc["attention_mask"],
                                 torch.ones((n, 1), device=device, dtype=torch.long)], dim=1)

        result = _greedy_trace_q_from_state(
            model, tokenizer, device, forced_ids, forced_attn, steps - 1,
            batch_size, token_groups, cat, [[hub_id]] * n
        )

        results[f"force_{hub_text}"] = result
        if result["semantic_hit_rate"] > best_hub_sem:
            best_hub_sem = result["semantic_hit_rate"]
            best_hub_result = result
            best_hub_name = hub_text
            best_hub_id = hub_id

        log(f"      force '{hub_text}': lex={result['lexical_hit_rate']:.3f}, "
            f"sem={result['semantic_hit_rate']:.3f}")

        del enc, forced_ids, forced_attn
        gc.collect()
        torch.cuda.empty_cache()

    results["best_hub"] = {
        "name": best_hub_name,
        "lexical_hit_rate": best_hub_result["lexical_hit_rate"] if best_hub_result else 0,
        "semantic_hit_rate": best_hub_result["semantic_hit_rate"] if best_hub_result else 0,
    }

    # C. Boost category logit by +5 at step 1
    log(f"    C. Boost category logit at step 1...")
    boost_result = _greedy_trace_q_with_logit_boost(
        model, tokenizer, device, prompts, steps, batch_size, max_length,
        token_groups, cat, {cid: 5.0 for cid in cat_ids_list}
    )
    results["boost_cat_logit"] = boost_result

    # D. Force best hub + boost category logit
    if best_hub_id is not None:
        log(f"    D. Force '{best_hub_name}' + boost category logit...")
        enc4 = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length
        ).to(device)
        forced_ids4 = torch.cat([enc4["input_ids"],
                                  torch.full((n, 1), best_hub_id, device=device, dtype=torch.long)], dim=1)
        forced_attn4 = torch.cat([enc4["attention_mask"],
                                   torch.ones((n, 1), device=device, dtype=torch.long)], dim=1)

        combo_result = _greedy_trace_q_with_logit_boost_from_state(
            model, tokenizer, device, forced_ids4, forced_attn4, steps - 1,
            batch_size, token_groups, cat, {cid: 5.0 for cid in cat_ids_list},
            prefix_tokens=[[best_hub_id]] * n
        )
        results["best_hub_plus_boost"] = combo_result

        del enc4, forced_ids4, forced_attn4
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ============================================================================
# Exp4: Action Natural Templates
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
    """Test action category with verb-targeted templates."""
    cat = examples[0]["cat"]
    prompts = [x["prompt"] for x in examples]
    n = len(prompts)

    # Natural verb templates
    verb_templates = [
        "The person is {obj}. This is a type of",
        "When someone is {obj}, they are performing a",
        "{obj} is an example of a physical",
    ]

    verb_prompts = []
    for ex in examples:
        obj = ex["obj"]
        for tpl in verb_templates:
            verb_prompts.append(tpl.format(obj=obj))

    # Standard templates
    log(f"    Standard templates...")
    standard_result = _greedy_trace_q(
        model, tokenizer, device, prompts, steps, batch_size, max_length,
        token_groups, cat
    )

    # Verb templates
    log(f"    Verb-targeted templates ({len(verb_prompts)} prompts)...")
    verb_target_ids = set(tok_ids(tokenizer, ACTION_VERB_TARGETS))
    action_word_ids = set(tok_ids(tokenizer, [" action", " activity", " movement", " verb"]))

    verb_token_groups = dict(token_groups)
    verb_token_groups["verb_targets"] = list(verb_target_ids)
    verb_token_groups["action_words"] = list(action_word_ids)

    verb_result = _greedy_trace_q(
        model, tokenizer, device, verb_prompts, steps, batch_size, max_length,
        verb_token_groups, cat
    )

    # Check verb/action word hit rate
    verb_hits = 0
    action_hits = 0
    for tokens in verb_result.get("path_tokens_list", []):
        if any(t in verb_target_ids for t in tokens):
            verb_hits += 1
        if any(t in action_word_ids for t in tokens):
            action_hits += 1

    return {
        "standard": {
            "lexical_hit_rate": standard_result["lexical_hit_rate"],
            "semantic_hit_rate": standard_result["semantic_hit_rate"],
            "examples": standard_result.get("examples", [])[:5],
        },
        "verb_templates": {
            "lexical_hit_rate": verb_result["lexical_hit_rate"],
            "semantic_hit_rate": verb_result["semantic_hit_rate"],
            "verb_hit_rate": round(verb_hits / max(len(verb_prompts), 1), 4),
            "action_word_hit_rate": round(action_hits / max(len(verb_prompts), 1), 4),
            "examples": verb_result.get("examples", [])[:5],
        },
    }


# ============================================================================
# Helper: Greedy trace with quality classification
# ============================================================================
def _greedy_trace_q(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    steps: int,
    batch_size: int,
    max_length: int,
    token_groups: dict[str, list[int]],
    cat: str,
) -> dict[str, Any]:
    """Greedy trace with hit quality classification."""
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length
    ).to(device)

    cur_ids = enc["input_ids"]
    cur_attn = enc["attention_mask"]
    cat_ids = set(token_groups.get("category", []))
    n = len(prompts)
    path_tokens_list = [[] for _ in range(n)]
    step1_top_tokens = Counter()

    for step in range(steps):
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
            path_tokens_list[i].append(int(tid))
            if step == 0:
                step1_top_tokens[tokenizer.decode([tid])] += 1

        cur_ids = torch.cat([cur_ids, next_step[:, None].to(cur_ids.device)], dim=1)
        cur_attn = torch.cat(
            [cur_attn, torch.ones((cur_attn.shape[0], 1), device=cur_attn.device, dtype=cur_attn.dtype)],
            dim=1
        )
        torch.cuda.empty_cache()

    # Classify
    qualities = []
    examples = []
    for i in range(n):
        decoded = tokenizer.decode(path_tokens_list[i], skip_special_tokens=True)
        quality = classify_hit_quality(decoded, cat, path_tokens_list[i], tokenizer, cat_ids)
        qualities.append(quality)
        if i < 5:
            examples.append({"path": decoded, "quality": quality})

    lexical_rate = sum(1 for q in qualities if q != "miss") / n
    natural_rate = sum(1 for q in qualities if q in ("natural_phrase", "semantic_answer")) / n
    semantic_rate = sum(1 for q in qualities if q == "semantic_answer") / n

    return {
        "lexical_hit_rate": round(lexical_rate, 4),
        "natural_hit_rate": round(natural_rate, 4),
        "semantic_hit_rate": round(semantic_rate, 4),
        "quality_dist": dict(Counter(qualities)),
        "step1_top_tokens": dict(step1_top_tokens.most_common(10)),
        "examples": examples,
        "path_tokens_list": path_tokens_list,
    }


def _greedy_trace_q_from_state(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    steps: int,
    batch_size: int,
    token_groups: dict[str, list[int]],
    cat: str,
    prefix_tokens: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Greedy trace from a given state with quality classification."""
    n = input_ids.shape[0]
    cat_ids = set(token_groups.get("category", []))
    path_tokens_list = list(prefix_tokens) if prefix_tokens else [[] for _ in range(n)]

    cur_ids = input_ids
    cur_attn = attention_mask

    for step in range(steps):
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
            path_tokens_list[i].append(int(tid))

        cur_ids = torch.cat([cur_ids, next_step[:, None].to(cur_ids.device)], dim=1)
        cur_attn = torch.cat(
            [cur_attn, torch.ones((cur_attn.shape[0], 1), device=cur_attn.device, dtype=cur_attn.dtype)],
            dim=1
        )
        torch.cuda.empty_cache()

    # Classify
    qualities = []
    for i in range(n):
        decoded = tokenizer.decode(path_tokens_list[i], skip_special_tokens=True)
        quality = classify_hit_quality(decoded, cat, path_tokens_list[i], tokenizer, cat_ids)
        qualities.append(quality)

    lexical_rate = sum(1 for q in qualities if q != "miss") / n
    natural_rate = sum(1 for q in qualities if q in ("natural_phrase", "semantic_answer")) / n
    semantic_rate = sum(1 for q in qualities if q == "semantic_answer") / n

    return {
        "lexical_hit_rate": round(lexical_rate, 4),
        "natural_hit_rate": round(natural_rate, 4),
        "semantic_hit_rate": round(semantic_rate, 4),
        "quality_dist": dict(Counter(qualities)),
    }


def _greedy_trace_q_with_logit_boost(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    steps: int,
    batch_size: int,
    max_length: int,
    token_groups: dict[str, list[int]],
    cat: str,
    boost_ids: dict[int, float],
) -> dict[str, Any]:
    """Greedy trace with logit boost at step 1 only."""
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length
    ).to(device)

    cur_ids = enc["input_ids"]
    cur_attn = enc["attention_mask"]
    cat_ids = set(token_groups.get("category", []))
    n = len(prompts)
    path_tokens_list = [[] for _ in range(n)]

    for step in range(steps):
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
            ].clone().float()

            if step == 0:
                for tid, val in boost_ids.items():
                    if 0 <= tid < logits.shape[-1]:
                        logits[:, tid] += val

            next_ids = logits.argmax(dim=1)
            all_next.append(next_ids.detach())
            del out, logits

        next_step = torch.cat(all_next, dim=0)
        for i, tid in enumerate(next_step.detach().cpu().tolist()):
            path_tokens_list[i].append(int(tid))

        cur_ids = torch.cat([cur_ids, next_step[:, None].to(cur_ids.device)], dim=1)
        cur_attn = torch.cat(
            [cur_attn, torch.ones((cur_attn.shape[0], 1), device=cur_attn.device, dtype=cur_attn.dtype)],
            dim=1
        )
        torch.cuda.empty_cache()

    # Classify
    qualities = []
    for i in range(n):
        decoded = tokenizer.decode(path_tokens_list[i], skip_special_tokens=True)
        quality = classify_hit_quality(decoded, cat, path_tokens_list[i], tokenizer, cat_ids)
        qualities.append(quality)

    lexical_rate = sum(1 for q in qualities if q != "miss") / n
    semantic_rate = sum(1 for q in qualities if q == "semantic_answer") / n

    return {
        "lexical_hit_rate": round(lexical_rate, 4),
        "semantic_hit_rate": round(semantic_rate, 4),
        "quality_dist": dict(Counter(qualities)),
    }


def _greedy_trace_q_with_logit_boost_from_state(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    steps: int,
    batch_size: int,
    token_groups: dict[str, list[int]],
    cat: str,
    boost_ids: dict[int, float],
    prefix_tokens: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Greedy trace from a given state with logit boost at step 1 only."""
    n = input_ids.shape[0]
    cat_ids = set(token_groups.get("category", []))
    path_tokens_list = list(prefix_tokens) if prefix_tokens else [[] for _ in range(n)]

    cur_ids = input_ids
    cur_attn = attention_mask

    for step in range(steps):
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
            ].clone().float()

            if step == 0:
                for tid, val in boost_ids.items():
                    if 0 <= tid < logits.shape[-1]:
                        logits[:, tid] += val

            next_ids = logits.argmax(dim=1)
            all_next.append(next_ids.detach())
            del out, logits

        next_step = torch.cat(all_next, dim=0)
        for i, tid in enumerate(next_step.detach().cpu().tolist()):
            path_tokens_list[i].append(int(tid))

        cur_ids = torch.cat([cur_ids, next_step[:, None].to(cur_ids.device)], dim=1)
        cur_attn = torch.cat(
            [cur_attn, torch.ones((cur_attn.shape[0], 1), device=cur_attn.device, dtype=cur_attn.dtype)],
            dim=1
        )
        torch.cuda.empty_cache()

    # Classify
    qualities = []
    for i in range(n):
        decoded = tokenizer.decode(path_tokens_list[i], skip_special_tokens=True)
        quality = classify_hit_quality(decoded, cat, path_tokens_list[i], tokenizer, cat_ids)
        qualities.append(quality)

    lexical_rate = sum(1 for q in qualities if q != "miss") / n
    semantic_rate = sum(1 for q in qualities if q == "semantic_answer") / n

    return {
        "lexical_hit_rate": round(lexical_rate, 4),
        "semantic_hit_rate": round(semantic_rate, 4),
        "quality_dist": dict(Counter(qualities)),
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
    """Run all Phase 514 experiments for a single category."""
    is_action = (cat == "action")
    templates = ACTION_TEMPLATES if is_action else GEN_TEMPLATES

    log(f"  Building examples for {cat} (action={is_action})...")
    gen_examples = build_generation_examples(
        cat, args.train_objects, args.test_objects, templates
    )

    objects = [x["obj"] for x in gen_examples]
    competitor_cats = [c for c in CATEGORIES if c != cat]
    token_groups = build_token_groups(tokenizer, cat, objects, competitor_cats)
    hub_id_list = tok_ids(tokenizer, HUB_TOKENS[:3])  # Only test top 3 hubs for memory

    results = {}

    # ===== Exp1: Path Value Function + Logit vs Value Rank =====
    log(f"  Exp1: Path value function V_c(y_1) for {cat}...")
    try:
        pv_result = path_value_and_rank(
            model, tokenizer, device, gen_examples, token_groups,
            args.steps, args.batch_size, args.max_length
        )
        results["path_value"] = pv_result
        log(f"    Group V_c(semantic): {pv_result['group_summary']}")
        log(f"    Logit-vs-Value match rate: {pv_result['logit_vs_value']['match_rate']:.3f}")
    except Exception as e:
        log(f"    Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["path_value"] = {"error": str(e)}

    # ===== Exp2: Hub Hidden State Analysis =====
    log(f"  Exp2: Hub hidden state analysis for {cat}...")
    try:
        hub_result = hub_hidden_state_analysis(
            model, tokenizer, device, gen_examples, token_groups,
            hub_id_list, args.batch_size, args.max_length
        )
        results["hub_hidden_state"] = hub_result
    except Exception as e:
        log(f"    Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["hub_hidden_state"] = {"error": str(e)}

    # ===== Exp3: Path-Value Guided Intervention =====
    log(f"  Exp3: Path-value guided intervention for {cat}...")
    try:
        pvi_result = path_value_guided_intervention(
            model, tokenizer, device, gen_examples, token_groups,
            args.steps, args.batch_size, args.max_length
        )
        results["intervention"] = pvi_result
    except Exception as e:
        log(f"    Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["intervention"] = {"error": str(e)}

    # ===== Exp4: Action Natural Templates =====
    if is_action:
        log(f"  Exp4: Action natural templates for {cat}...")
        try:
            at_result = action_natural_templates(
                model, tokenizer, device, gen_examples, token_groups,
                args.steps, args.batch_size, args.max_length
            )
            results["action_templates"] = at_result
        except Exception as e:
            log(f"    Exp4 FAILED: {e}")
            import traceback; traceback.print_exc()
            results["action_templates"] = {"error": str(e)}

    return {
        "n_generation_prompts": len(gen_examples),
        "is_action": is_action,
        "templates": [t[0] for t in templates],
        "token_group_sizes": {k: len(v) for k, v in token_groups.items()},
        **results,
    }


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
            "phase": 514,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "L": L,
            "d_model": d,
            "categories": categories,
            "train_objects": args.train_objects,
            "test_objects": args.test_objects,
            "steps": args.steps,
            "category_results": {},
        }

        for ci, cat in enumerate(categories, 1):
            log(f"{args.model}: category {ci}/{len(categories)} {cat}")
            cat_out = run_category(model, tokenizer, device, cat, args)
            result["category_results"][cat] = cat_out

        return result
    finally:
        release_model(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=20)
    parser.add_argument("--test-objects", type=int, default=10)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    path = out_dir / f"phase514_{args.model}_path_value.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
