#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase856_identity_class_overlap_cross_domain_rollout_audit as p856  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402


PHASE = 901
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase901_stop_token_competitiveness_audit")
PHASE899_ROOT = Path("tests/result/phase899_domain_axis_rollout_protocol_audit")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_gears(subset_key: str) -> list[dict[str, Any]]:
    gears = []
    for part in str(subset_key or "").split("+"):
        if part.startswith("L") and "C" in part:
            gear = p862.parse_gear_key(part)
            if gear is not None:
                gears.append(gear)
    return gears


def selected_phase899_rows(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE899_ROOT / args.phase899_round / f"phase899_{model_name}_rollout_rows.jsonl"
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("is_source_candidate") and row.get("rollout_answer_class") and row.get("protocol_drift")
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("eval_domain")),
            str(row.get("source_subset_key")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
            str(row.get("edit_mode")),
        )
    )
    max_rows = int(args.max_rows_per_model)
    return rows[:max_rows] if max_rows > 0 else rows


def first_token_ids(tokenizer, phrases: list[str]) -> list[int]:
    out: list[int] = []
    for phrase in phrases:
        try:
            ids = tokenizer.encode(phrase, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(phrase)
        if ids:
            out.append(int(ids[0]))
    seen = []
    for item in out:
        if item not in seen and item >= 0:
            seen.append(item)
    return seen


def token_groups(tokenizer) -> dict[str, list[int]]:
    eos = []
    if tokenizer.eos_token_id is not None:
        eos.append(int(tokenizer.eos_token_id))
    return {
        "eos": eos,
        "period": first_token_ids(tokenizer, [".", " .", ".\n"]),
        "newline": first_token_ids(tokenizer, ["\n", "\n\n"]),
        "field": first_token_ids(
            tokenizer,
            [
                "Category",
                " Category",
                "\nCategory",
                "Item",
                " Item",
                "\nItem",
                "Class",
                " Class",
                "Subclass",
                " Subclass",
                "Answer",
                " Answer",
            ],
        ),
        "explanation": first_token_ids(
            tokenizer,
            ["The", " The", "I", " I", "Okay", " Okay", "Please", " Please", "This", " This"],
        ),
        "list": first_token_ids(tokenizer, [",", " ,", " or", "or", "1", " 1", "\n1", "2", " 2"]),
    }


def best_for_ids(logits: torch.Tensor, ids: list[int]) -> dict[str, Any]:
    valid = [int(i) for i in ids if 0 <= int(i) < int(logits.numel())]
    if not valid:
        return {"best_id": None, "best_logit": None, "rank": None}
    scores = [(int(i), float(logits[int(i)].item())) for i in valid]
    best_id, best_score = max(scores, key=lambda item: item[1])
    rank = int((logits > best_score).sum().item()) + 1
    return {"best_id": best_id, "best_logit": best_score, "rank": rank}


def decode_token(tokenizer, token_id: int | None) -> str | None:
    if token_id is None:
        return None
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return None


def logits_after_answer_prefix(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    gears: list[dict[str, Any]],
    source_mode: str,
    case: dict[str, Any],
    max_prefix_tokens: int,
    scale_up_factor: float,
) -> tuple[torch.Tensor, list[int], str, bool]:
    current = [int(x) for x in prompt_ids]
    generated: list[int] = []
    answer_seen = False
    logits_next = None
    for step in range(int(max_prefix_tokens) + 1):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handles = []
        try:
            if step == 0 and gears and source_mode != "original":
                handles.extend(p862.install_scaled_gear_edit(model, gears, source_mode, scale_up_factor))
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
        finally:
            for handle in handles:
                handle.remove()
        logits_next = logits
        if generated:
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if p856.classify_rollout(text, case).get("rollout_answer_class"):
                answer_seen = True
                return logits_next, generated, text, answer_seen
        if step >= int(max_prefix_tokens):
            break
        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)
        current.append(next_id)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return logits_next, generated, text, answer_seen


def make_row(
    tokenizer,
    source_row: dict[str, Any],
    case: dict[str, Any],
    logits: torch.Tensor,
    prefix_ids: list[int],
    prefix_text: str,
    answer_seen: bool,
    groups: dict[str, list[int]],
) -> dict[str, Any]:
    top_id = int(torch.argmax(logits).item())
    top_logit = float(logits[top_id].item())
    payload = {
        "phase": PHASE,
        "row_kind": "phase901_stop_token_competitiveness_row",
        "model": source_row.get("model"),
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "case_split": source_row.get("case_split"),
        "object": source_row.get("object"),
        "prompt_variant": source_row.get("prompt_variant"),
        "edit_mode": source_row.get("edit_mode"),
        "prefix_ids": prefix_ids,
        "prefix_text": prefix_text,
        "answer_prefix_seen": answer_seen,
        "next_top_id": top_id,
        "next_top_token": decode_token(tokenizer, top_id),
        "next_top_logit": top_logit,
    }
    stop_ids = list(dict.fromkeys((groups.get("eos") or []) + (groups.get("period") or [])))
    protocol_ids = list(
        dict.fromkeys(
            (groups.get("field") or []) + (groups.get("explanation") or []) + (groups.get("list") or [])
        )
    )
    for group_name, ids in {**groups, "stop": stop_ids, "protocol": protocol_ids}.items():
        best = best_for_ids(logits, ids)
        payload[f"{group_name}_best_id"] = best.get("best_id")
        payload[f"{group_name}_best_token"] = decode_token(tokenizer, best.get("best_id"))
        payload[f"{group_name}_best_logit"] = best.get("best_logit")
        payload[f"{group_name}_rank"] = best.get("rank")
        payload[f"{group_name}_margin_vs_top"] = None if best.get("best_logit") is None else float(best["best_logit"] - top_logit)
    payload["stop_top10"] = bool(payload.get("stop_rank") is not None and int(payload["stop_rank"]) <= 10)
    payload["stop_top50"] = bool(payload.get("stop_rank") is not None and int(payload["stop_rank"]) <= 50)
    payload["stop_top100"] = bool(payload.get("stop_rank") is not None and int(payload["stop_rank"]) <= 100)
    payload["eos_top100"] = bool(payload.get("eos_rank") is not None and int(payload["eos_rank"]) <= 100)
    payload["period_top50"] = bool(payload.get("period_rank") is not None and int(payload["period_rank"]) <= 50)
    return payload


def mean(values: list[float]) -> float | None:
    return p846.mean(values)


def median(values: list[float]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = {
        name: [row.get(f"{name}_rank") for row in rows if row.get(f"{name}_rank") is not None]
        for name in ["eos", "period", "stop", "protocol", "field", "explanation", "list"]
    }
    return {
        "rows": len(rows),
        "answer_prefix_seen": sum(1 for row in rows if row.get("answer_prefix_seen")),
        "stop_top10": sum(1 for row in rows if row.get("stop_top10")),
        "stop_top50": sum(1 for row in rows if row.get("stop_top50")),
        "stop_top100": sum(1 for row in rows if row.get("stop_top100")),
        "eos_top100": sum(1 for row in rows if row.get("eos_top100")),
        "period_top50": sum(1 for row in rows if row.get("period_top50")),
        "mean_stop_rank": mean([float(x) for x in ranks["stop"]]),
        "median_stop_rank": median([float(x) for x in ranks["stop"]]),
        "mean_eos_rank": mean([float(x) for x in ranks["eos"]]),
        "median_eos_rank": median([float(x) for x in ranks["eos"]]),
        "mean_period_rank": mean([float(x) for x in ranks["period"]]),
        "median_period_rank": median([float(x) for x in ranks["period"]]),
        "mean_protocol_rank": mean([float(x) for x in ranks["protocol"]]),
        "median_protocol_rank": median([float(x) for x in ranks["protocol"]]),
        "next_top_tokens": dict(sorted(Counter(str(row.get("next_top_token")) for row in rows).items())),
    }


def summarize_model(model_name: str, rows: list[dict[str, Any]], attn_impl: str | None) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("eval_domain"))].append(row)
    overall = summarize_rows(rows)
    if overall["stop_top10"]:
        evidence_label = "stop_token_competitive_in_some_rows"
    elif overall["stop_top100"]:
        evidence_label = "stop_token_near_but_not_decisive"
    else:
        evidence_label = "stop_token_not_competitive"
    return {
        "phase": PHASE,
        "title": "Stop Token Competitiveness Audit",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "overall": overall,
        "domain_summaries": {domain: summarize_rows(vals) for domain, vals in sorted(by_domain.items())},
        "evidence_label": evidence_label,
        "boundary": (
            "Phase901 measures whether EOS/period stop tokens are competitive after answer-class prefix. "
            "It is a logit audit, not a causal stop-gate intervention."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = selected_phase899_rows(args.model, args)
    if args.dry_run or not selected_rows:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected_rows else "no_rows",
            "selected_rows": selected_rows,
        }
        p846.write_json(out_dir / f"phase901_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase901_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload
    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = token_groups(tokenizer)
        prompt_cache: dict[tuple[str, str], list[int]] = {}
        for idx, source_row in enumerate(selected_rows, 1):
            case = case_map.get(str(source_row.get("case_id")))
            if not case:
                continue
            prompt_key = (str(source_row.get("case_id")), str(source_row.get("prompt_variant")))
            if prompt_key not in prompt_cache:
                prompt = p885.prompt_for_case(case, str(source_row.get("prompt_variant")))
                prompt_cache[prompt_key] = p862.p844.encode_prompt(tokenizer, prompt)
            logits, prefix_ids, prefix_text, answer_seen = logits_after_answer_prefix(
                model,
                tokenizer,
                device,
                prompt_cache[prompt_key],
                parse_gears(str(source_row.get("source_subset_key"))),
                str(source_row.get("edit_mode")),
                case,
                int(args.max_prefix_tokens),
                float(args.scale_up_factor),
            )
            rows.append(make_row(tokenizer, source_row, case, logits, prefix_ids, prefix_text, answer_seen, groups))
            if idx % max(1, int(args.log_every)) == 0 or idx == len(selected_rows):
                log(f"{args.model}/{args.round_name}: row={idx}/{len(selected_rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = summarize_model(args.model, rows, attn_impl)
    p846.write_json(out_dir / f"phase901_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase901_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 901 stop token competitiveness audit",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model summaries", ""])
    lines.append(
        "| model | rows | stop top10 | stop top50 | stop top100 | eos top100 | period top50 | median stop rank | median eos rank | median protocol rank | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("model_summaries") or []:
        overall = row.get("overall") or {}
        lines.append(
            "| {model} | {rows} | {stop_top10} | {stop_top50} | {stop_top100} | {eos_top100} | {period_top50} | "
            "{median_stop_rank} | {median_eos_rank} | {median_protocol_rank} | {evidence} |".format(
                model=row.get("model"),
                evidence=row.get("evidence_label"),
                **overall,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    rows = []
    for model_name in MODELS:
        summary_path = out_dir / f"phase901_{model_name}_summary.json"
        if summary_path.exists():
            summaries.append(read_json(summary_path))
        rows.extend(read_jsonl(out_dir / f"phase901_{model_name}_rows.jsonl"))
    overall = summarize_rows(rows)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": overall,
        "model_summaries": summaries,
        "evidence_label_counts": dict(sorted(Counter(str(summary.get("evidence_label")) for summary in summaries).items())),
    }
    p846.write_json(out_dir / "phase901_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase901_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="stop_token_competitiveness_audit")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=16)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall_scalar"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
