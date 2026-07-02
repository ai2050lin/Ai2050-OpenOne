#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase844_geometry_route_natural_gear_set_search as p844  # noqa: E402
import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase854_full_vocab_blocker_min_cut_validation as p854  # noqa: E402


PHASE = 856
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase856_identity_class_overlap_cross_domain_rollout_audit")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def clean_text(text: str) -> str:
    return p844.p828.p825.clean_generated(text)


def normalize(text: str) -> str:
    text = clean_text(str(text)).strip()
    text = re.sub(r"^[\s\"'`:\-–—,.;()\[\]{}]+", "", text)
    text = re.sub(r"[\s\"'`:\-–—,.;()\[\]{}]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def first_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    return int(ids[0]) if ids else None


def token_variant_ids(tokenizer, variants: list[str]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for variant in variants:
        token_id = first_token_id(tokenizer, variant)
        if token_id is not None and token_id not in seen:
            out.append(int(token_id))
            seen.add(int(token_id))
    return out


def decode_token(tokenizer, token_id: int | None) -> str | None:
    if token_id is None:
        return None
    return tokenizer.decode([int(token_id)])


def variants(texts: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        bases = [text]
        if not text.endswith("s"):
            bases.append(f"{text}s")
        if text.endswith("y"):
            bases.append(f"{text[:-1]}ies")
        forms = []
        for base in bases:
            forms.extend([base, base.capitalize(), base.title(), f" {base}", f" {base.capitalize()}", f" {base.title()}"])
        for form in forms:
            if form not in seen:
                out.append(form)
                seen.add(form)
    return out


def base_cases() -> list[dict[str, Any]]:
    raw = [
        ("geometry", "triangle", ["geometric shape", "geometric figure", "shape", "figure", "polygon"], "member_not_alias"),
        ("geometry", "square", ["geometric shape", "geometric figure", "shape", "figure", "polygon", "quadrilateral"], "member_not_alias"),
        ("geometry", "rectangle", ["geometric shape", "geometric figure", "shape", "figure", "polygon", "quadrilateral"], "member_not_alias"),
        ("geometry", "circle", ["geometric shape", "geometric figure", "shape", "figure", "round shape", "closed curve"], "member_not_alias"),
        ("geometry", "polygon", ["geometric shape", "geometric figure", "shape", "figure", "polygon"], "object_is_answer_alias"),
        ("animal", "dog", ["animal", "living thing", "living creature", "creature", "mammal"], "member_not_alias"),
        ("animal", "cat", ["animal", "living thing", "living creature", "creature", "mammal"], "member_not_alias"),
        ("animal", "bird", ["animal", "living thing", "living creature", "creature"], "member_not_alias"),
        ("animal", "fish", ["animal", "living thing", "living creature", "creature"], "member_not_alias"),
        ("animal", "mammal", ["animal", "living thing", "living creature", "creature", "mammal"], "object_is_answer_alias"),
        ("tool", "hammer", ["tool", "hand tool", "instrument", "implement", "object"], "member_not_alias"),
        ("tool", "screwdriver", ["tool", "hand tool", "instrument", "implement", "object"], "member_not_alias"),
        ("tool", "wrench", ["tool", "hand tool", "instrument", "implement", "object"], "member_not_alias"),
        ("tool", "saw", ["tool", "hand tool", "instrument", "implement", "object"], "member_not_alias"),
        ("tool", "tool", ["tool", "hand tool", "instrument", "implement", "object"], "object_is_answer_alias"),
        ("color", "red", ["color", "hue", "colour"], "attribute_value_not_alias"),
        ("color", "blue", ["color", "hue", "colour"], "attribute_value_not_alias"),
        ("color", "green", ["color", "hue", "colour"], "attribute_value_not_alias"),
        ("color", "yellow", ["color", "hue", "colour"], "attribute_value_not_alias"),
        ("color", "color", ["color", "property"], "object_is_answer_alias"),
        ("material", "wood", ["material", "substance", "matter"], "member_not_alias"),
        ("material", "iron", ["material", "substance", "matter", "metal"], "member_not_alias"),
        ("material", "plastic", ["material", "substance", "matter"], "member_not_alias"),
        ("material", "glass", ["material", "substance", "matter"], "member_not_alias"),
        ("material", "metal", ["material", "substance", "matter", "metal"], "object_is_answer_alias"),
        ("abstract", "freedom", ["abstract concept", "concept", "idea", "abstraction"], "member_not_alias"),
        ("abstract", "justice", ["abstract concept", "concept", "idea", "abstraction"], "member_not_alias"),
        ("abstract", "time", ["abstract concept", "concept", "idea", "abstraction"], "member_not_alias"),
        ("abstract", "truth", ["abstract concept", "concept", "idea", "abstraction"], "member_not_alias"),
        ("abstract", "concept", ["abstract concept", "concept", "idea", "abstraction"], "object_is_answer_alias"),
        ("plant", "oak", ["plant", "living thing", "organism", "tree"], "member_not_alias"),
        ("plant", "rose", ["plant", "living thing", "organism", "flower"], "member_not_alias"),
        ("plant", "grass", ["plant", "living thing", "organism"], "member_not_alias"),
        ("plant", "tree", ["plant", "living thing", "organism", "tree"], "object_is_answer_alias"),
        ("object", "chair", ["physical object", "object", "item"], "member_not_alias"),
        ("object", "car", ["physical object", "object", "item", "vehicle"], "member_not_alias"),
        ("object", "book", ["physical object", "object", "item"], "member_not_alias"),
        ("object", "object", ["physical object", "object"], "object_is_answer_alias"),
    ]
    cases: list[dict[str, Any]] = []
    for idx, (domain, obj, aliases, overlap_kind) in enumerate(raw, 1):
        cases.append(
            {
                "case_id": f"p856_{idx:03d}_{domain}_{obj}",
                "domain": domain,
                "object": obj,
                "answer_aliases": aliases,
                "canonical_answer": aliases[0],
                "overlap_kind": overlap_kind,
            }
        )
    return cases


def selected_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    domains = set(parse_csv(args.domains))
    rows = [case for case in base_cases() if not domains or case["domain"] in domains]
    if int(args.max_cases_per_domain) > 0:
        out: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for case in rows:
            if counts[case["domain"]] >= int(args.max_cases_per_domain):
                continue
            out.append(case)
            counts[case["domain"]] += 1
        rows = out
    if int(args.max_cases) > 0:
        rows = rows[: int(args.max_cases)]
    return rows


def prompt_for_case(case: dict[str, Any], variant: str) -> str:
    obj = case["object"]
    if variant == "natural_question":
        return f"Answer with only a short category phrase.\nWhich category best describes a {obj}?\nCategory:"
    if variant == "natural_category":
        return f"Give a concise category phrase.\nDo not explain.\nItem: {obj}\nCategory:"
    if variant == "object_only":
        return f"Write the best short category phrase for the item.\nItem: {obj}\nPhrase:"
    if variant == "classification":
        return f"Classify the item using a short noun phrase.\nItem: {obj}\nClass:"
    raise ValueError(f"unknown prompt variant: {variant}")


def token_sets(tokenizer, case: dict[str, Any]) -> dict[str, Any]:
    class_aliases = variants(list(case["answer_aliases"]))
    strict_aliases = variants([str(case["canonical_answer"])])
    object_aliases = variants([str(case["object"])])
    class_ids = token_variant_ids(tokenizer, class_aliases)
    strict_ids = token_variant_ids(tokenizer, strict_aliases)
    object_ids = token_variant_ids(tokenizer, object_aliases)
    overlap_ids = sorted(set(class_ids) & set(object_ids))
    return {
        "class_aliases": class_aliases,
        "strict_aliases": strict_aliases,
        "object_aliases": object_aliases,
        "class_target_ids": class_ids,
        "class_target_tokens": [decode_token(tokenizer, token_id) for token_id in class_ids],
        "strict_target_ids": strict_ids,
        "strict_target_tokens": [decode_token(tokenizer, token_id) for token_id in strict_ids],
        "object_ids": object_ids,
        "object_tokens": [decode_token(tokenizer, token_id) for token_id in object_ids],
        "identity_class_overlap_token_ids": overlap_ids,
        "identity_class_overlap_tokens": [decode_token(tokenizer, token_id) for token_id in overlap_ids],
    }


def best_score(logits: torch.Tensor, ids: list[int]) -> tuple[float | None, int | None]:
    score: float | None = None
    best_id: int | None = None
    for token_id in ids:
        if 0 <= int(token_id) < int(logits.numel()):
            value = float(logits[int(token_id)].item())
            if score is None or value > score:
                score = value
                best_id = int(token_id)
    return score, best_id


def rank_of_score(logits: torch.Tensor, score: float | None) -> int | None:
    if score is None:
        return None
    return int((logits > float(score)).sum().item()) + 1


def role_for(token_id: int, text: str, sets: dict[str, Any]) -> str:
    class_ids = set(int(x) for x in sets["class_target_ids"])
    object_ids = set(int(x) for x in sets["object_ids"])
    strict_ids = set(int(x) for x in sets["strict_target_ids"])
    if token_id in class_ids and token_id in object_ids:
        return "identity_class_overlap"
    if token_id in strict_ids:
        return "strict_target"
    if token_id in class_ids:
        return "answer_class"
    if token_id in object_ids:
        return "object_echo"
    stripped = text.strip()
    if not stripped:
        return "format_space"
    if re.fullmatch(r"[\W_]+", stripped, flags=re.UNICODE):
        return "format_punct"
    if stripped[:1].isdigit():
        return "number"
    return "other"


def first_token_metrics(tokenizer, logits: torch.Tensor, sets: dict[str, Any], topk: int) -> dict[str, Any]:
    class_score, class_id = best_score(logits, sets["class_target_ids"])
    strict_score, strict_id = best_score(logits, sets["strict_target_ids"])
    object_score, object_id = best_score(logits, sets["object_ids"])
    class_ids = set(int(x) for x in sets["class_target_ids"])
    clear_class_ids = class_ids - set(int(x) for x in sets["object_ids"])
    clear_score, clear_id = best_score(logits, sorted(clear_class_ids))

    def blocker_count(score: float | None, excluded: set[int]) -> int | None:
        if score is None:
            return None
        mask = logits > float(score)
        if excluded:
            idx = torch.tensor([x for x in excluded if 0 <= x < int(logits.numel())], dtype=torch.long, device=logits.device)
            if idx.numel() > 0:
                mask[idx] = False
        return int(mask.sum().item())

    top = torch.topk(logits, k=min(int(topk), int(logits.numel())))
    top_tokens = []
    for token_id, value in zip(top.indices.tolist(), top.values.tolist(), strict=False):
        text = decode_token(tokenizer, int(token_id)) or ""
        top_tokens.append(
            {
                "token_id": int(token_id),
                "token": text,
                "logit": float(value),
                "role": role_for(int(token_id), text, sets),
            }
        )
    return {
        "class_best_logit": class_score,
        "class_best_id": class_id,
        "class_best_token": decode_token(tokenizer, class_id),
        "class_best_rank": rank_of_score(logits, class_score),
        "class_blocker_count": blocker_count(class_score, class_ids) if class_score is not None else None,
        "first_token_answer_class": blocker_count(class_score, class_ids) == 0 if class_score is not None else False,
        "strict_best_logit": strict_score,
        "strict_best_id": strict_id,
        "strict_best_token": decode_token(tokenizer, strict_id),
        "strict_best_rank": rank_of_score(logits, strict_score),
        "strict_blocker_count": blocker_count(strict_score, set(int(x) for x in sets["strict_target_ids"]))
        if strict_score is not None
        else None,
        "first_token_strict": blocker_count(strict_score, set(int(x) for x in sets["strict_target_ids"])) == 0
        if strict_score is not None
        else False,
        "clear_class_best_logit": clear_score,
        "clear_class_best_id": clear_id,
        "clear_class_best_token": decode_token(tokenizer, clear_id),
        "clear_class_best_rank": rank_of_score(logits, clear_score),
        "clear_class_blocker_count": blocker_count(clear_score, clear_class_ids) if clear_score is not None else None,
        "first_token_clear_answer_class": blocker_count(clear_score, clear_class_ids) == 0 if clear_score is not None else False,
        "object_best_logit": object_score,
        "object_best_id": object_id,
        "object_best_token": decode_token(tokenizer, object_id),
        "object_rank": rank_of_score(logits, object_score),
        "class_minus_object_logit": None if class_score is None or object_score is None else float(class_score - object_score),
        "clear_class_minus_object_logit": None
        if clear_score is None or object_score is None
        else float(clear_score - object_score),
        "first_token_identity_class_overlap": bool(class_id is not None and class_id in set(sets["object_ids"])),
        "top_tokens": top_tokens,
    }


def classify_rollout(generated: str, case: dict[str, Any]) -> dict[str, Any]:
    cleaned = clean_text(generated)
    norm = normalize(cleaned)
    object_norm = normalize(str(case["object"]))
    alias_norms = [normalize(alias) for alias in variants(list(case["answer_aliases"]))]
    strict_norm = normalize(str(case["canonical_answer"]))
    starts_object = bool(object_norm and (norm == object_norm or norm.startswith(object_norm + " ")))
    starts_alias = any(alias and (norm == alias or norm.startswith(alias + " ")) for alias in alias_norms)
    starts_strict = bool(strict_norm and (norm == strict_norm or norm.startswith(strict_norm + " ")))
    if not norm:
        label = "format_or_empty"
    elif starts_object and starts_alias:
        label = "identity_class_overlap"
    elif starts_object:
        label = "object_echo"
    elif starts_strict:
        label = "strict_canonical"
    elif starts_alias:
        label = "answer_alias"
    elif re.fullmatch(r"[\W_]+", norm):
        label = "format_or_empty"
    else:
        label = "other"
    return {
        "generated_clean": cleaned,
        "rollout_label": label,
        "rollout_answer_class": label in {"strict_canonical", "answer_alias", "identity_class_overlap"},
        "rollout_clear_answer_class": label in {"strict_canonical", "answer_alias"},
        "rollout_identity_class_overlap": label == "identity_class_overlap",
        "rollout_strict_canonical": label == "strict_canonical",
        "rollout_object_echo": label == "object_echo",
        "rollout_other_or_format": label in {"other", "format_or_empty"},
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = selected_cases(args)
    prompt_variants = parse_csv(args.prompt_variants)
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "cases": cases,
            "prompt_variants": prompt_variants,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p844.p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for case_idx, case in enumerate(cases, 1):
            sets = token_sets(tokenizer, case)
            for variant in prompt_variants:
                prompt = prompt_for_case(case, variant)
                prompt_ids = p844.encode_prompt(tokenizer, prompt)
                logits = p844.first_logits_with_gears(model, device, prompt_ids, [], "original")
                metrics = first_token_metrics(tokenizer, logits, sets, int(args.topk_tokens))
                generated, token_ids = p844.greedy_with_gears(
                    model, tokenizer, device, prompt_ids, [], "original", int(args.max_new_tokens)
                )
                rollout = classify_rollout(generated, case)
                rows.append(
                    {
                        "row_kind": "phase856_identity_class_overlap_cross_domain_rollout_audit",
                        "phase": PHASE,
                        "model": args.model,
                        "round": args.round_name,
                        "case_id": case["case_id"],
                        "domain": case["domain"],
                        "object": case["object"],
                        "canonical_answer": case["canonical_answer"],
                        "answer_aliases": case["answer_aliases"],
                        "overlap_kind": case["overlap_kind"],
                        "prompt_variant": variant,
                        "prompt": prompt,
                        "token_ids": token_ids,
                        **sets,
                        **metrics,
                        **rollout,
                    }
                )
            if case_idx % max(1, int(args.log_every)) == 0 or case_idx == len(cases):
                log(f"{args.model}/{args.round_name}: cross-domain cases {case_idx}/{len(cases)} rows={len(rows)}")
    finally:
        if model is not None:
            p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(args, attn_impl, cases, rows)
    p846.write_jsonl(out_dir / f"phase856_{args.model}_rows.jsonl", rows)
    p846.write_json(out_dir / f"phase856_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "rows": len(rows),
                "first_token_answer_class": summary["overall"]["first_token_answer_class"],
                "rollout_answer_class": summary["overall"]["rollout_answer_class"],
                "rollout_clear_answer_class": summary["overall"]["rollout_clear_answer_class"],
                "object_echo": summary["overall"]["rollout_object_echo"],
                "first_to_rollout_f1": summary["overall"]["first_token_predicts_rollout"]["f1"],
                "clear_first_to_clear_rollout_f1": summary["overall"]["clear_first_predicts_clear_rollout"]["f1"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def binary_stats(actual: list[bool], predicted: list[bool]) -> dict[str, Any]:
    tp = sum(1 for a, p in zip(actual, predicted) if a and p)
    tn = sum(1 for a, p in zip(actual, predicted) if (not a) and (not p))
    fp = sum(1 for a, p in zip(actual, predicted) if (not a) and p)
    fn = sum(1 for a, p in zip(actual, predicted) if a and (not p))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    n = len(actual)
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = [bool(row.get("first_token_answer_class")) for row in rows]
    clear_first = [bool(row.get("first_token_clear_answer_class")) for row in rows]
    rollout = [bool(row.get("rollout_answer_class")) for row in rows]
    clear_rollout = [bool(row.get("rollout_clear_answer_class")) for row in rows]
    return {
        "n": len(rows),
        "first_token_answer_class": sum(first),
        "first_token_clear_answer_class": sum(clear_first),
        "first_token_strict": sum(1 for row in rows if row.get("first_token_strict")),
        "first_token_identity_class_overlap": sum(1 for row in rows if row.get("first_token_identity_class_overlap")),
        "rollout_answer_class": sum(rollout),
        "rollout_clear_answer_class": sum(clear_rollout),
        "rollout_identity_class_overlap": sum(1 for row in rows if row.get("rollout_identity_class_overlap")),
        "rollout_strict_canonical": sum(1 for row in rows if row.get("rollout_strict_canonical")),
        "rollout_object_echo": sum(1 for row in rows if row.get("rollout_object_echo")),
        "rollout_other_or_format": sum(1 for row in rows if row.get("rollout_other_or_format")),
        "rollout_labels": dict(Counter(str(row.get("rollout_label")) for row in rows)),
        "mean_class_blocker_count": mean([finite(row.get("class_blocker_count")) for row in rows if row.get("class_blocker_count") is not None]),
        "mean_class_rank": mean([finite(row.get("class_best_rank")) for row in rows if row.get("class_best_rank") is not None]),
        "mean_class_minus_object_logit": mean(
            [finite(row.get("class_minus_object_logit")) for row in rows if row.get("class_minus_object_logit") is not None]
        ),
        "first_token_predicts_rollout": binary_stats(rollout, first),
        "clear_first_predicts_clear_rollout": binary_stats(clear_rollout, clear_first),
    }


def summarize(args: argparse.Namespace, attn_impl: str | None, cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_overlap: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain"))].append(row)
        by_overlap[str(row.get("overlap_kind"))].append(row)
        by_prompt[str(row.get("prompt_variant"))].append(row)
    return {
        "phase": PHASE,
        "title": "Identity-Class Overlap and Cross-Domain Rollout Audit",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_cases": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "prompt_variants": parse_csv(args.prompt_variants),
        "overall": compact(rows),
        "domain_summary": {domain: compact(group) for domain, group in sorted(by_domain.items())},
        "overlap_summary": {kind: compact(group) for kind, group in sorted(by_overlap.items())},
        "prompt_summary": {prompt: compact(group) for prompt, group in sorted(by_prompt.items())},
        "top_mismatch_rows": [
            {
                "domain": row.get("domain"),
                "object": row.get("object"),
                "overlap_kind": row.get("overlap_kind"),
                "prompt_variant": row.get("prompt_variant"),
                "class_best_token": row.get("class_best_token"),
                "object_best_token": row.get("object_best_token"),
                "class_blocker_count": row.get("class_blocker_count"),
                "generated_clean": row.get("generated_clean"),
                "rollout_label": row.get("rollout_label"),
                "top_tokens": row.get("top_tokens"),
            }
            for row in sorted(
                rows,
                key=lambda r: (
                    int(bool(r.get("first_token_answer_class")) and not bool(r.get("rollout_clear_answer_class"))),
                    int(bool(r.get("rollout_object_echo"))),
                    finite(r.get("class_blocker_count")),
                ),
                reverse=True,
            )[:80]
        ],
        "boundary": (
            "This phase audits natural cross-domain first-token answer-class closure and short rollout closure. "
            "It is not a gear intervention phase and does not prove global causal closure."
        ),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 856 Identity-Class Overlap and Cross-Domain Rollout Audit ({payload['round']})",
        "",
        "- Source: natural prompts across domains; no gear intervention.",
        "- Boundary: first-token field vs short rollout field, not full causal closure.",
        "",
        "## Cross-Model Summary",
        "",
        "| model | rows | first class | clear first | rollout class | clear rollout | object echo | first->rollout F1 | clear F1 | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        overall = data.get("overall") or {}
        pred = overall.get("first_token_predicts_rollout") or {}
        clear_pred = overall.get("clear_first_predicts_clear_rollout") or {}
        lines.append(
            f"| {model_name} | {overall.get('n', 0)} | {overall.get('first_token_answer_class', 0)} | "
            f"{overall.get('first_token_clear_answer_class', 0)} | {overall.get('rollout_answer_class', 0)} | "
            f"{overall.get('rollout_clear_answer_class', 0)} | {overall.get('rollout_object_echo', 0)} | "
            f"{fmt(pred.get('f1'))} | {fmt(clear_pred.get('f1'))} | "
            f"`{json.dumps(overall.get('rollout_labels') or {}, ensure_ascii=False)}` |"
        )
    lines += [
        "",
        "## Domain Summary",
        "",
        "| model | domain | rows | first class | rollout class | clear rollout | object echo | clear F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for domain, stats in sorted((data.get("domain_summary") or {}).items()):
            clear_pred = stats.get("clear_first_predicts_clear_rollout") or {}
            lines.append(
                f"| {model_name} | `{domain}` | {stats.get('n', 0)} | {stats.get('first_token_answer_class', 0)} | "
                f"{stats.get('rollout_answer_class', 0)} | {stats.get('rollout_clear_answer_class', 0)} | "
                f"{stats.get('rollout_object_echo', 0)} | {fmt(clear_pred.get('f1'))} |"
            )
    lines += [
        "",
        "## Overlap Summary",
        "",
        "| model | overlap kind | rows | first overlap | rollout overlap | object echo | clear rollout | clear F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for kind, stats in sorted((data.get("overlap_summary") or {}).items()):
            clear_pred = stats.get("clear_first_predicts_clear_rollout") or {}
            lines.append(
                f"| {model_name} | `{kind}` | {stats.get('n', 0)} | {stats.get('first_token_identity_class_overlap', 0)} | "
                f"{stats.get('rollout_identity_class_overlap', 0)} | {stats.get('rollout_object_echo', 0)} | "
                f"{stats.get('rollout_clear_answer_class', 0)} | {fmt(clear_pred.get('f1'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "status": "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [],
        "model_summaries": {},
    }
    for model_name in MODELS:
        path = out_dir / f"phase856_{model_name}_summary.json"
        if path.exists():
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = p846.read_json(path)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase856_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase856_cross_model_summary.md", payload)
    print(json.dumps({"status": payload["status"], "round": round_name, "models": payload["models"]}, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--domains", default="geometry,animal,tool")
    parser.add_argument("--max-cases-per-domain", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--prompt-variants", default="natural_question")
    parser.add_argument("--topk-tokens", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        summarize_round(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    eval_model(args)


if __name__ == "__main__":
    main()
