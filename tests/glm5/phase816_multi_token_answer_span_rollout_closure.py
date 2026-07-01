#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase796_global_competitor_token_identity_audit as p796  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase776_readout_bridge_competition_audit import normalize_token_text  # noqa: E402


PHASE = 816
RESULT_ROOT = Path("tests/result/phase816_multi_token_answer_span_rollout_closure")


CASES: list[dict[str, Any]] = [
    {
        "case_id": "p816_cat_living_thing",
        "object": "cat",
        "question": "Which category best describes a cat?",
        "answer": "living thing",
        "contrast_answer": "household object",
        "distractors": ["hand tool", "public transport", "musical instrument", "red color"],
    },
    {
        "case_id": "p816_hammer_hand_tool",
        "object": "hammer",
        "question": "Which category best describes a hammer?",
        "answer": "hand tool",
        "contrast_answer": "living thing",
        "distractors": ["public transport", "musical instrument", "warm color", "body organ"],
    },
    {
        "case_id": "p816_bus_public_transport",
        "object": "bus",
        "question": "Which category best describes a bus?",
        "answer": "public transport",
        "contrast_answer": "musical instrument",
        "distractors": ["living thing", "hand tool", "household furniture", "warm color"],
    },
    {
        "case_id": "p816_guitar_musical_instrument",
        "object": "guitar",
        "question": "Which category best describes a guitar?",
        "answer": "musical instrument",
        "contrast_answer": "public transport",
        "distractors": ["living thing", "hand tool", "edible fruit", "cold season"],
    },
    {
        "case_id": "p816_rose_flowering_plant",
        "object": "rose",
        "question": "Which category best describes a rose?",
        "answer": "flowering plant",
        "contrast_answer": "hand tool",
        "distractors": ["public transport", "musical instrument", "body organ", "warm color"],
    },
    {
        "case_id": "p816_chair_household_furniture",
        "object": "chair",
        "question": "Which category best describes a chair?",
        "answer": "household furniture",
        "contrast_answer": "living thing",
        "distractors": ["public transport", "musical instrument", "edible fruit", "body organ"],
    },
    {
        "case_id": "p816_apple_edible_fruit",
        "object": "apple",
        "question": "Which category best describes an apple?",
        "answer": "edible fruit",
        "contrast_answer": "hand tool",
        "distractors": ["public transport", "musical instrument", "warm color", "body organ"],
    },
    {
        "case_id": "p816_heart_body_organ",
        "object": "heart",
        "question": "Which category best describes a heart?",
        "answer": "body organ",
        "contrast_answer": "public transport",
        "distractors": ["hand tool", "musical instrument", "edible fruit", "warm color"],
    },
    {
        "case_id": "p816_red_warm_color",
        "object": "red",
        "question": "Which category best describes red?",
        "answer": "warm color",
        "contrast_answer": "cold season",
        "distractors": ["hand tool", "public transport", "living thing", "body organ"],
    },
    {
        "case_id": "p816_salmon_aquatic_animal",
        "object": "salmon",
        "question": "Which category best describes a salmon?",
        "answer": "aquatic animal",
        "contrast_answer": "hand tool",
        "distractors": ["public transport", "musical instrument", "warm color", "household object"],
    },
    {
        "case_id": "p816_oak_tall_tree",
        "object": "oak",
        "question": "Which category best describes an oak?",
        "answer": "tall tree",
        "contrast_answer": "musical instrument",
        "distractors": ["hand tool", "public transport", "warm color", "body organ"],
    },
    {
        "case_id": "p816_carrot_root_vegetable",
        "object": "carrot",
        "question": "Which category best describes a carrot?",
        "answer": "root vegetable",
        "contrast_answer": "public transport",
        "distractors": ["hand tool", "musical instrument", "warm color", "body organ"],
    },
    {
        "case_id": "p816_laptop_electronic_device",
        "object": "laptop",
        "question": "Which category best describes a laptop?",
        "answer": "electronic device",
        "contrast_answer": "living thing",
        "distractors": ["hand tool", "public transport", "edible fruit", "body organ"],
    },
    {
        "case_id": "p816_spoon_eating_utensil",
        "object": "spoon",
        "question": "Which category best describes a spoon?",
        "answer": "eating utensil",
        "contrast_answer": "living thing",
        "distractors": ["public transport", "musical instrument", "warm color", "body organ"],
    },
    {
        "case_id": "p816_triangle_geometric_shape",
        "object": "triangle",
        "question": "Which category best describes a triangle?",
        "answer": "geometric shape",
        "contrast_answer": "living thing",
        "distractors": ["hand tool", "public transport", "musical instrument", "warm color"],
    },
    {
        "case_id": "p816_winter_cold_season",
        "object": "winter",
        "question": "Which category best describes winter?",
        "answer": "cold season",
        "contrast_answer": "warm color",
        "distractors": ["hand tool", "public transport", "musical instrument", "body organ"],
    },
    {
        "case_id": "p816_gold_precious_metal",
        "object": "gold",
        "question": "Which category best describes gold?",
        "answer": "precious metal",
        "contrast_answer": "living thing",
        "distractors": ["hand tool", "public transport", "edible fruit", "body organ"],
    },
    {
        "case_id": "p816_oxygen_chemical_element",
        "object": "oxygen",
        "question": "Which category best describes oxygen?",
        "answer": "chemical element",
        "contrast_answer": "living thing",
        "distractors": ["hand tool", "public transport", "musical instrument", "warm color"],
    },
    {
        "case_id": "p816_cactus_desert_plant",
        "object": "cactus",
        "question": "Which category best describes a cactus?",
        "answer": "desert plant",
        "contrast_answer": "hand tool",
        "distractors": ["public transport", "musical instrument", "warm color", "body organ"],
    },
    {
        "case_id": "p816_doctor_medical_worker",
        "object": "doctor",
        "question": "Which category best describes a doctor?",
        "answer": "medical worker",
        "contrast_answer": "public transport",
        "distractors": ["hand tool", "musical instrument", "warm color", "body organ"],
    },
]

GENERIC_BLOCKERS = [
    "yes",
    "no",
    "the answer",
    "I don't know",
    "not sure",
    "none of these",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_text(value: Any) -> str:
    return normalize_token_text("" if value is None else str(value)).strip().lower()


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def select_evenly(n: int, k: int) -> list[int]:
    if k <= 0 or k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def build_prompt(case: dict[str, Any], variant: str) -> str:
    choices = [case["answer"], case["contrast_answer"], *case.get("distractors", [])]
    # Keep deterministic but avoid putting the target first in every case.
    shift = sum(ord(ch) for ch in case["case_id"]) % len(choices)
    choices = choices[shift:] + choices[:shift]
    if variant == "exact_choices":
        return (
            "Choose exactly one phrase from the choices.\n"
            "Write only the phrase, with no punctuation or explanation.\n"
            f"Question: {case['question']}\n"
            f"Choices: {'; '.join(choices)}\n"
            "Answer:"
        )
    if variant == "no_choices":
        return (
            "Answer with a short category phrase.\n"
            "Write only the phrase, with no punctuation or explanation.\n"
            f"Question: {case['question']}\n"
            "Answer:"
        )
    raise ValueError(f"unknown prompt variant: {variant}")


def phrase_variants(phrase: str) -> list[str]:
    raw = str(phrase).strip()
    variants = {
        raw,
        raw.lower(),
        raw.title(),
        f" {raw}",
        f" {raw.lower()}",
        f" {raw.title()}",
    }
    return sorted(v for v in variants if v)


def span_candidates(tokenizer, case: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[tuple[str, str]] = [("target", case["answer"]), ("contrast", case["contrast_answer"])]
    specs.extend(("distractor", value) for value in case.get("distractors", []))
    specs.extend(("generic_blocker", value) for value in GENERIC_BLOCKERS)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for cls, phrase in specs:
        for text in phrase_variants(phrase):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            key = (cls, tuple(int(x) for x in ids))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "candidate_class": cls,
                    "phrase": phrase,
                    "variant_text": text,
                    "token_ids": [int(x) for x in ids],
                    "span_len": len(ids),
                    "normalized_text": norm_text(text),
                }
            )
    return out[: int(args.max_span_candidates)]


def pad_batch(seqs: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in seqs)
    input_ids = torch.full((len(seqs), max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, : len(seq)] = 1
    return input_ids, attention_mask


def token_top_rows(tokenizer, logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
    vals, ids = torch.topk(logits.detach().float().cpu(), min(int(k), int(logits.numel())))
    rows = []
    for val, tid in zip(vals.tolist(), ids.tolist()):
        rows.append(
            {
                "token_id": int(tid),
                "token_text": tokenizer.decode([int(tid)], skip_special_tokens=False),
                "logit": float(val),
            }
        )
    return rows


def score_candidates(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    candidates: list[dict[str, Any]],
    batch_size: int,
    top_k: int,
) -> list[dict[str, Any]]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out: list[dict[str, Any]] = []
    for start in range(0, len(candidates), int(batch_size)):
        batch = candidates[start : start + int(batch_size)]
        seqs = [prompt_ids + cand["token_ids"][:-1] for cand in batch]
        input_ids, attention_mask = pad_batch(seqs, int(pad_id), device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        for row_idx, cand in enumerate(batch):
            token_logs = []
            sum_logprob = 0.0
            sum_logit = 0.0
            max_rank_above = 0
            all_top1 = True
            for step, token_id in enumerate(cand["token_ids"]):
                pos = len(prompt_ids) - 1 + step
                step_logits = logits[row_idx, pos].detach().float().cpu()
                token_logit = float(step_logits[int(token_id)].item())
                log_probs = torch.log_softmax(step_logits, dim=-1)
                token_logprob = float(log_probs[int(token_id)].item())
                rank_above = int((step_logits > token_logit).sum().item())
                max_rank_above = max(max_rank_above, rank_above)
                all_top1 = all_top1 and rank_above == 0
                token_logs.append(
                    {
                        "step": step,
                        "token_id": int(token_id),
                        "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                        "logit": token_logit,
                        "logprob": token_logprob,
                        "rank_above": rank_above,
                        "top_tokens": token_top_rows(tokenizer, step_logits, top_k) if step < 3 else [],
                    }
                )
                sum_logprob += token_logprob
                sum_logit += token_logit
            item = dict(cand)
            item.update(
                {
                    "score_sum_logprob": sum_logprob,
                    "score_mean_logprob": sum_logprob / max(len(cand["token_ids"]), 1),
                    "score_sum_logit": sum_logit,
                    "score_mean_logit": sum_logit / max(len(cand["token_ids"]), 1),
                    "step_all_top1": all_top1,
                    "max_rank_above": max_rank_above,
                    "token_logs": token_logs,
                }
            )
            out.append(item)
    out.sort(key=lambda x: (float(x["score_mean_logprob"]), float(x["score_sum_logprob"])), reverse=True)
    return out


def clean_generated(text: str) -> str:
    text = text.strip()
    for sep in ["\n", ".", ",", ";", ":"]:
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip().strip('"').strip("'").strip()


def greedy_generate(model, tokenizer, device: torch.device, prompt_ids: list[int], max_new_tokens: int) -> tuple[str, list[int]]:
    # Manual greedy rollout avoids model.generate backend differences across local models.
    current = [int(x) for x in prompt_ids]
    new_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    for _step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1].detach().float()
        next_id = int(torch.argmax(logits).item())
        new_ids.append(next_id)
        current.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    return tokenizer.decode(new_ids, skip_special_tokens=True), new_ids


def generation_match(generated: str, case: dict[str, Any]) -> dict[str, Any]:
    cleaned = clean_generated(generated)
    cleaned_norm = norm_text(cleaned)
    target_norm = norm_text(case["answer"])
    contrast_norm = norm_text(case["contrast_answer"])
    distractor_norms = {norm_text(x): x for x in case.get("distractors", [])}
    if cleaned_norm.startswith(target_norm):
        match_class = "target"
    elif cleaned_norm.startswith(contrast_norm):
        match_class = "contrast"
    elif any(cleaned_norm.startswith(key) for key in distractor_norms):
        match_class = "distractor"
    elif not cleaned_norm:
        match_class = "empty_or_format"
    else:
        match_class = "other"
    return {
        "generated_text": generated,
        "generated_clean": cleaned,
        "generated_norm": cleaned_norm,
        "generation_match_class": match_class,
        "generation_starts_target": match_class == "target",
        "generation_starts_contrast": match_class == "contrast",
    }


def best_by_class(scored: list[dict[str, Any]], cls: str) -> dict[str, Any] | None:
    vals = [x for x in scored if x.get("candidate_class") == cls]
    return vals[0] if vals else None


def finite(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) else default


def compact_span(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    keep = {
        "candidate_class",
        "phrase",
        "variant_text",
        "token_ids",
        "span_len",
        "score_sum_logprob",
        "score_mean_logprob",
        "score_sum_logit",
        "score_mean_logit",
        "step_all_top1",
        "max_rank_above",
    }
    out = {k: row.get(k) for k in keep}
    out["token_logs"] = row.get("token_logs", [])[:3]
    return out


def audit_case(model, tokenizer, device: torch.device, case: dict[str, Any], prompt_variant: str, args: argparse.Namespace) -> dict[str, Any]:
    prompt = build_prompt(case, prompt_variant)
    prompt_ids = [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]
    candidates = span_candidates(tokenizer, case, args)
    scored = score_candidates(model, tokenizer, device, prompt_ids, candidates, args.batch_size, args.top_k)
    best_target = best_by_class(scored, "target")
    best_contrast = best_by_class(scored, "contrast")
    non_target = [x for x in scored if x.get("candidate_class") != "target"]
    best_non_target = non_target[0] if non_target else None
    best_generic = best_by_class(scored, "generic_blocker")
    generated, generated_ids = greedy_generate(model, tokenizer, device, prompt_ids, args.max_new_tokens)
    gen = generation_match(generated, case)

    target_margin_vs_non_target = (
        finite(best_target.get("score_mean_logprob")) - finite(best_non_target.get("score_mean_logprob"))
        if best_target and best_non_target
        else None
    )
    target_margin_vs_contrast = (
        finite(best_target.get("score_mean_logprob")) - finite(best_contrast.get("score_mean_logprob"))
        if best_target and best_contrast
        else None
    )
    target_margin_vs_generic = (
        finite(best_target.get("score_mean_logprob")) - finite(best_generic.get("score_mean_logprob"))
        if best_target and best_generic
        else None
    )
    span_score_closure = bool(best_target and best_non_target and finite(target_margin_vs_non_target) > 0)
    contrast_cleared = bool(best_target and best_contrast and finite(target_margin_vs_contrast) > 0)
    generic_cleared = bool(best_target and best_generic and finite(target_margin_vs_generic) > 0)
    strict_step_top1 = bool(best_target and best_target.get("step_all_top1"))
    rollout_closure = bool(gen["generation_starts_target"])
    target_span_len = int(best_target.get("span_len")) if best_target else None
    target_requires_multi_token = bool(target_span_len and target_span_len > 1)
    full_span_closure = span_score_closure and contrast_cleared and generic_cleared and rollout_closure

    if full_span_closure:
        label = "span_score_and_rollout_closed"
    elif span_score_closure and not rollout_closure:
        label = "span_score_closed_rollout_not_closed"
    elif rollout_closure and not span_score_closure:
        label = "rollout_closed_span_score_not_closed"
    elif contrast_cleared and not span_score_closure:
        label = "contrast_cleared_but_other_span_wins"
    elif best_non_target and best_non_target.get("candidate_class") == "contrast":
        label = "contrast_span_wins"
    elif best_non_target and best_non_target.get("candidate_class") == "generic_blocker":
        label = "generic_blocker_span_wins"
    else:
        label = "distractor_or_other_span_wins"

    return {
        "row_kind": "phase816_multi_token_answer_span_rollout",
        "phase": PHASE,
        "model": args.model,
        "round": args.round_name,
        "case_id": case["case_id"],
        "object": case["object"],
        "prompt_variant": prompt_variant,
        "prompt": prompt,
        "target_answer": case["answer"],
        "contrast_answer": case["contrast_answer"],
        "distractors": case.get("distractors", []),
        "n_candidates": len(candidates),
        "best_target": compact_span(best_target),
        "best_contrast": compact_span(best_contrast),
        "best_non_target": compact_span(best_non_target),
        "best_generic_blocker": compact_span(best_generic),
        "top_scored_spans": [compact_span(x) for x in scored[: int(args.saved_top_spans)]],
        "target_margin_vs_non_target_mean_logprob": target_margin_vs_non_target,
        "target_margin_vs_contrast_mean_logprob": target_margin_vs_contrast,
        "target_margin_vs_generic_mean_logprob": target_margin_vs_generic,
        "span_score_closure": span_score_closure,
        "contrast_span_cleared": contrast_cleared,
        "generic_blocker_cleared": generic_cleared,
        "strict_step_top1": strict_step_top1,
        "target_requires_multi_token": target_requires_multi_token,
        "target_best_span_len": target_span_len,
        "generated_token_ids": generated_ids,
        **gen,
        "rollout_closure": rollout_closure,
        "full_span_rollout_closure": full_span_closure,
        "phase816_label": label,
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str) -> dict[str, Any]:
    labels = Counter(row.get("phase816_label") for row in rows)
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["prompt_variant"]].append(row)
    variant_summary = {}
    for variant, vals in by_variant.items():
        variant_summary[variant] = {
            "n": len(vals),
            "span_score_closure_rows": sum(1 for row in vals if row.get("span_score_closure")),
            "rollout_closure_rows": sum(1 for row in vals if row.get("rollout_closure")),
            "full_span_rollout_closure_rows": sum(1 for row in vals if row.get("full_span_rollout_closure")),
            "strict_step_top1_rows": sum(1 for row in vals if row.get("strict_step_top1")),
            "labels": dict(Counter(row.get("phase816_label") for row in vals)),
        }
    best = sorted(
        rows,
        key=lambda row: (
            not bool(row.get("full_span_rollout_closure")),
            not bool(row.get("span_score_closure")),
            -finite(row.get("target_margin_vs_non_target_mean_logprob"), -9999.0),
        ),
    )[:60]
    return {
        "phase": PHASE,
        "title": "Multi Token Answer Span Rollout Closure",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({row["case_id"] for row in rows}),
        "prompt_variants": parse_csv(args.prompt_variants),
        "target_requires_multi_token_rows": sum(1 for row in rows if row.get("target_requires_multi_token")),
        "span_score_closure_rows": sum(1 for row in rows if row.get("span_score_closure")),
        "contrast_span_cleared_rows": sum(1 for row in rows if row.get("contrast_span_cleared")),
        "generic_blocker_cleared_rows": sum(1 for row in rows if row.get("generic_blocker_cleared")),
        "rollout_closure_rows": sum(1 for row in rows if row.get("rollout_closure")),
        "full_span_rollout_closure_rows": sum(1 for row in rows if row.get("full_span_rollout_closure")),
        "strict_step_top1_rows": sum(1 for row in rows if row.get("strict_step_top1")),
        "by_label": dict(labels),
        "by_prompt_variant": variant_summary,
        "best_rows": best,
        "boundary": (
            "This phase builds genuinely multi-token answer-span tasks and checks teacher-forced span score plus greedy rollout closure."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 816 Multi Token Answer Span Rollout Closure ({payload['round']})",
        "",
        "- Boundary: target phrase must be multi-token; closure is tested by teacher-forced span score and greedy rollout.",
        "",
        "## Model Summary",
        "",
        "| model | rows | cases | multi-token rows | span-score | rollout | full | contrast cleared | generic cleared | strict step top1 | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | {data.get('n_rows')} | {data.get('n_cases')} | {data.get('target_requires_multi_token_rows')} | "
            f"{data.get('span_score_closure_rows')} | {data.get('rollout_closure_rows')} | "
            f"{data.get('full_span_rollout_closure_rows')} | {data.get('contrast_span_cleared_rows')} | "
            f"{data.get('generic_blocker_cleared_rows')} | {data.get('strict_step_top1_rows')} | "
            f"`{json.dumps(data.get('by_label') or {}, ensure_ascii=False)}` |"
        )
    lines += [
        "",
        "## Best Rows",
        "",
        "| model | variant | case | target | best target | best non-target | span-score | rollout | full | margin | generated | label |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("best_rows", [])[:24]:
            bt = row.get("best_target") or {}
            bn = row.get("best_non_target") or {}
            lines.append(
                f"| {model_name} | {row.get('prompt_variant')} | {row.get('case_id')} | `{row.get('target_answer')}` | "
                f"`{bt.get('variant_text')}` | `{bn.get('variant_text')}`/{bn.get('candidate_class')} | "
                f"{int(bool(row.get('span_score_closure')))} | {int(bool(row.get('rollout_closure')))} | "
                f"{int(bool(row.get('full_span_rollout_closure')))} | "
                f"{finite(row.get('target_margin_vs_non_target_mean_logprob')):.3f} | "
                f"`{row.get('generated_clean')}` | `{row.get('phase816_label')}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [CASES[i] for i in select_evenly(len(CASES), int(args.max_cases))]
    prompt_variants = parse_csv(args.prompt_variants)
    log(
        f"{args.model}/{args.round_name}: cases={len(selected)} prompt_variants={prompt_variants} "
        f"batch={args.batch_size} max_new={args.max_new_tokens}"
    )
    if args.dry_run:
        return {"model": args.model, "selected_cases": [case["case_id"] for case in selected]}
    model, tokenizer, device, attn_impl = p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows: list[dict[str, Any]] = []
    try:
        for ci, case in enumerate(selected, 1):
            for prompt_variant in prompt_variants:
                rows.append(audit_case(model, tokenizer, device, case, prompt_variant, args))
            if ci % int(args.log_every) == 0 or ci == len(selected):
                log(f"{args.model}: span rollout {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(rows, args, attn_impl)
    write_jsonl(out_dir / f"phase816_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase816_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_rows": summary["n_rows"],
                "n_cases": summary["n_cases"],
                "span_score_closure_rows": summary["span_score_closure_rows"],
                "rollout_closure_rows": summary["rollout_closure_rows"],
                "full_span_rollout_closure_rows": summary["full_span_rollout_closure_rows"],
                "by_label": summary["by_label"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "status": "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_summaries": {},
        "models": [],
    }
    for model_name in MODELS:
        path = out_dir / f"phase816_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase816_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase816_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--prompt-variants", default="exact_choices")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-span-candidates", type=int, default=96)
    parser.add_argument("--saved-top-spans", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        payload = summarize_round(args.round_name)
        print(json.dumps({"round": args.round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only")
    result = run_model(args)
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
