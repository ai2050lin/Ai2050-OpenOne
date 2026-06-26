#!/usr/bin/env python3
"""
Phase 672: Graph Atlas Counterfactual Natural Trajectory Audit.

Runs natural forward/generation tests on the Phase 670 counterfactual control
set. No patching is used. Metrics include first-token readout, multi-competitor
margin, exact generation, normalized exact generation, and continuation prefix
matching.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase584_gate_repair import load_model_flash  # noqa: E402
from model_utils import release_model  # noqa: E402


CONTROL_PATH = Path(
    "results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_counterfactual_control_set.json"
)
OUT_ROOT = Path("results/glm5_phase672_graph_atlas_counterfactual_natural_trajectory_audit")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def compact_text(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def expected_variants(text: str) -> list[str]:
    variants = [text, " " + text, "\n" + text]
    if text.startswith(("{", "[", "-", "Value:")):
        variants.extend([" " + text, "\n" + text])
    out = []
    seen = set()
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def encode_variants(tokenizer: Any, text: str) -> list[list[int]]:
    seqs = []
    seen = set()
    for variant in expected_variants(text):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        key = tuple(ids)
        if ids and key not in seen:
            seen.add(key)
            seqs.append(ids)
    return seqs


def rank_of_ids(logits: torch.Tensor, ids: set[int]) -> tuple[int | None, float | None]:
    if not ids:
        return None, None
    scores = logits[list(ids)]
    best = float(scores.max().item())
    rank = int((logits > best).sum().item()) + 1
    return rank, best


def token_category(text: str, expected_ids: set[int], tid: int) -> str:
    if tid in expected_ids:
        return "expected"
    if text == " " or text.isspace() and "\n" not in text:
        return "space"
    if "\n" in text:
        return "newline"
    if text.strip().startswith(("{", "[", '"')):
        return "json_or_quote"
    if text.strip().startswith("-"):
        return "list_marker"
    if text.strip().startswith(("The", "the", "Record", "Value")):
        return "word_or_explanation"
    if text.strip() in {":", ".", ",", ";"}:
        return "punctuation"
    if not text.strip():
        return "blank"
    return "other"


def topk_metric(tokenizer: Any, logits: torch.Tensor, expected_ids: set[int], top_k: int) -> dict:
    vals, idx = torch.topk(logits, k=min(top_k, logits.numel()))
    top = []
    best_competitor = None
    for score, tid_t in zip(vals.tolist(), idx.tolist()):
        tid = int(tid_t)
        text = tokenizer.decode([tid])
        cat = token_category(text, expected_ids, tid)
        item = {"id": tid, "text": text, "score": float(score), "category": cat}
        top.append(item)
        if cat != "expected" and best_competitor is None:
            best_competitor = item
    rank, expected_score = rank_of_ids(logits, expected_ids)
    competitor_score = best_competitor["score"] if best_competitor else None
    return {
        "expected_rank": rank,
        "expected_score": expected_score,
        "top1_id": top[0]["id"] if top else None,
        "top1_text": top[0]["text"] if top else "",
        "top1_category": top[0]["category"] if top else "none",
        "best_competitor": best_competitor,
        "expected_minus_best_competitor": (
            expected_score - competitor_score
            if expected_score is not None and competitor_score is not None
            else None
        ),
        "top": top[: min(10, len(top))],
    }


def match_expected(generated: str, expected: str) -> dict:
    gen_norm = normalize_text(generated)
    exp_norm = normalize_text(expected)
    gen_compact = compact_text(generated)
    exp_compact = compact_text(expected)
    return {
        "strict_exact": generated.startswith(expected),
        "normalized_exact": gen_norm.startswith(exp_norm),
        "compact_exact": gen_compact.startswith(exp_compact),
        "contains_value": exp_norm in gen_norm or exp_compact in gen_compact,
    }


def continuation_metric(new_ids: list[int], expected_seqs: list[list[int]]) -> dict:
    best = {
        "matched_tokens": 0,
        "expected_len": 0,
        "first_token_match": False,
        "token1_match": False,
        "token2_match": False,
    }
    for seq in expected_seqs:
        m = 0
        for got, want in zip(new_ids, seq):
            if got != want:
                break
            m += 1
        if m > best["matched_tokens"]:
            best = {
                "matched_tokens": m,
                "expected_len": len(seq),
                "first_token_match": bool(m >= 1),
                "token1_match": bool(m >= 2),
                "token2_match": bool(m >= 3),
            }
    return best


def classify_generated(text: str) -> str:
    s = text.lstrip()
    if s.startswith("{"):
        return "json"
    if s.startswith("-"):
        return "list"
    if s.startswith("Value:"):
        return "label"
    if s.startswith("\n"):
        return "newline"
    if s.lower().startswith(("the ", "record ", "it ", "this ")):
        return "sentence_or_explanation"
    if "\n" in text[:4]:
        return "newline"
    return "short_or_other"


def select_cases(cases: list[dict], args) -> list[dict]:
    if args.max_cases <= 0 or args.max_cases >= len(cases):
        return cases
    by_family: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_family[c["family"]].append(c)
    families = sorted(by_family)
    selected = []
    per_family = max(1, args.max_cases // len(families))
    for family in families:
        selected.extend(by_family[family][:per_family])
    i = 0
    while len(selected) < args.max_cases and i < len(cases):
        if cases[i] not in selected:
            selected.append(cases[i])
        i += 1
    return selected[: args.max_cases]


def run_batch(model, tokenizer, device, batch: list[dict], args) -> list[dict]:
    prompts = [c["prompt"] for c in batch]
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
    with torch.inference_mode():
        out = model(**enc, return_dict=True)
        logits_all = out.logits.float()
        generated = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    rows = []
    for i, c in enumerate(batch):
        final_pos = int(prompt_lens[i]) - 1
        # Left padding means the final prompt token is at sequence end.
        logits = logits_all[i, -1, :]
        expected_seqs = encode_variants(tokenizer, c["expected_output"])
        expected_first_ids = {seq[0] for seq in expected_seqs if seq}
        metric = topk_metric(tokenizer, logits, expected_first_ids, args.top_k)
        full_ids = generated[i].tolist()
        new_ids = full_ids[enc["input_ids"].shape[1] :]
        gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        row = {
            "case_id": c["case_id"],
            "family": c["family"],
            "axis": c["axis"],
            "object_name": c["object_name"],
            "relation": c["relation"],
            "value": c["value"],
            "format_name": c["format_name"],
            "target_nodes": c["target_nodes"],
            "expected_output": c["expected_output"],
            "expected_variant_token_lens": [len(seq) for seq in expected_seqs],
            "prompt_tokens": int(prompt_lens[i]),
            "first_token_metric": metric,
            "generated_text": gen_text,
            "generated_token_ids": new_ids[: args.max_new_tokens],
            "generated_class": classify_generated(gen_text),
            "match": match_expected(gen_text, c["expected_output"]),
            "continuation": continuation_metric(new_ids, expected_seqs),
        }
        rows.append(row)
    del out, logits_all, generated, enc
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    def blank() -> dict:
        return {
            "n": 0,
            "normalized_exact": 0,
            "compact_exact": 0,
            "contains_value": 0,
            "first_expected_top1": 0,
            "token1_match": 0,
            "token2_match": 0,
            "mean_expected_rank_sum": 0.0,
            "mean_multi_margin_sum": 0.0,
            "multi_margin_count": 0,
            "top1_category": {},
            "generated_class": {},
        }

    groups: dict[tuple[str, str], dict] = defaultdict(blank)
    node_groups: dict[str, dict] = defaultdict(blank)
    for r in rows:
        keys = [("family", r["family"]), ("format", r["format_name"])]
        for key in keys:
            g = groups[key]
            add_summary(g, r)
        for node in r["target_nodes"]:
            add_summary(node_groups[node], r)
    return {
        "overall": finalize_summary(sum_group(rows)),
        "by_family": {k[1]: finalize_summary(v) for k, v in sorted(groups.items()) if k[0] == "family"},
        "by_format": {k[1]: finalize_summary(v) for k, v in sorted(groups.items()) if k[0] == "format"},
        "by_target_node": {k: finalize_summary(v) for k, v in sorted(node_groups.items())},
    }


def add_summary(g: dict, r: dict) -> None:
    g["n"] += 1
    g["normalized_exact"] += int(r["match"]["normalized_exact"])
    g["compact_exact"] += int(r["match"]["compact_exact"])
    g["contains_value"] += int(r["match"]["contains_value"])
    g["first_expected_top1"] += int(r["first_token_metric"]["expected_rank"] == 1)
    g["token1_match"] += int(r["continuation"]["token1_match"])
    g["token2_match"] += int(r["continuation"]["token2_match"])
    rank = r["first_token_metric"]["expected_rank"]
    if rank is not None:
        g["mean_expected_rank_sum"] += float(rank)
    margin = r["first_token_metric"]["expected_minus_best_competitor"]
    if margin is not None:
        g["mean_multi_margin_sum"] += float(margin)
        g["multi_margin_count"] += 1
    top_cat = r["first_token_metric"]["top1_category"]
    g["top1_category"][top_cat] = g["top1_category"].get(top_cat, 0) + 1
    gen_cls = r["generated_class"]
    g["generated_class"][gen_cls] = g["generated_class"].get(gen_cls, 0) + 1


def sum_group(rows: list[dict]) -> dict:
    g = {
        "n": 0,
        "normalized_exact": 0,
        "compact_exact": 0,
        "contains_value": 0,
        "first_expected_top1": 0,
        "token1_match": 0,
        "token2_match": 0,
        "mean_expected_rank_sum": 0.0,
        "mean_multi_margin_sum": 0.0,
        "multi_margin_count": 0,
        "top1_category": {},
        "generated_class": {},
    }
    for r in rows:
        add_summary(g, r)
    return g


def finalize_summary(g: dict) -> dict:
    n = max(1, g["n"])
    mc = max(1, g["multi_margin_count"])
    return {
        "n": g["n"],
        "normalized_exact_rate": g["normalized_exact"] / n,
        "compact_exact_rate": g["compact_exact"] / n,
        "contains_value_rate": g["contains_value"] / n,
        "first_expected_top1_rate": g["first_expected_top1"] / n,
        "token1_match_rate": g["token1_match"] / n,
        "token2_match_rate": g["token2_match"] / n,
        "mean_expected_rank": g["mean_expected_rank_sum"] / n,
        "mean_multi_margin": g["mean_multi_margin_sum"] / mc,
        "top1_category": dict(sorted(g["top1_category"].items(), key=lambda kv: kv[1], reverse=True)),
        "generated_class": dict(sorted(g["generated_class"].items(), key=lambda kv: kv[1], reverse=True)),
    }


def run_model(args) -> dict:
    control = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    cases = select_cases(control["cases"], args)
    model, tokenizer, device = load_model_flash(args.model)
    rows = []
    try:
        for start in range(0, len(cases), args.batch_size):
            batch = cases[start : start + args.batch_size]
            rows.extend(run_batch(model, tokenizer, device, batch, args))
            log(f"{args.model}: {len(rows)}/{len(cases)} cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    examples = []
    for r in rows:
        if len(examples) >= args.example_limit:
            break
        if not r["match"]["normalized_exact"] or r["first_token_metric"]["expected_rank"] != 1:
            examples.append(r)
    payload = {
        "phase": 672,
        "title": "Graph Atlas Counterfactual Natural Trajectory Audit",
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len(rows),
        "source_control": str(CONTROL_PATH),
        "config": {
            "max_cases": args.max_cases,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k,
        },
        "summary": summarize_rows(rows),
        "examples": examples,
        "rows": rows,
    }
    return payload


def write_model_outputs(payload: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    model = payload["model"]
    full_path = OUT_ROOT / f"phase672_{model}_natural_trajectory_confirm.json"
    rows_path = OUT_ROOT / f"phase672_{model}_natural_trajectory_rows.jsonl"
    full_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as f:
        for row in payload["rows"]:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    slim = {k: v for k, v in payload.items() if k != "rows"}
    (OUT_ROOT / f"phase672_{model}_natural_trajectory_summary.json").write_text(
        json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"Wrote {full_path}")
    log(f"Wrote {rows_path}")


def write_cross_model_summary() -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase672_*_natural_trajectory_summary.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        models.append({
            "model": d["model"],
            "n_cases": d["n_cases"],
            "overall": d["summary"]["overall"],
            "by_family": d["summary"]["by_family"],
        })
    payload = {
        "phase": 672,
        "title": "Graph Atlas Counterfactual Natural Trajectory Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    lines = [
        "# Phase 672 Graph Atlas Counterfactual Natural Trajectory Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | cases | norm_exact | compact_exact | contains_value | first_top1 | token1 | token2 | mean_rank | mean_margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in models:
        o = m["overall"]
        lines.append(
            f"| {m['model']} | {m['n_cases']} | {o['normalized_exact_rate']:.3f} | "
            f"{o['compact_exact_rate']:.3f} | {o['contains_value_rate']:.3f} | "
            f"{o['first_expected_top1_rate']:.3f} | {o['token1_match_rate']:.3f} | "
            f"{o['token2_match_rate']:.3f} | {o['mean_expected_rank']:.2f} | {o['mean_multi_margin']:.3f} |"
        )
    lines += ["", "## Family Details", ""]
    for m in models:
        lines += [f"### {m['model']}", "", "| family | n | norm_exact | first_top1 | token1 | mean_rank |", "|---|---:|---:|---:|---:|---:|"]
        for family, s in m["by_family"].items():
            lines.append(
                f"| {family} | {s['n']} | {s['normalized_exact_rate']:.3f} | "
                f"{s['first_expected_top1_rate']:.3f} | {s['token1_match_rate']:.3f} | {s['mean_expected_rank']:.2f} |"
            )
        lines.append("")
    (OUT_ROOT / "phase672_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_ROOT / "phase672_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--max-cases", type=int, default=630)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--example-limit", type=int, default=60)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.summarize_only:
        payload = write_cross_model_summary()
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is set")
    payload = run_model(args)
    write_model_outputs(payload)
    print(json.dumps({k: payload[k] for k in ["model", "n_cases", "summary"]}, ensure_ascii=False, indent=2), flush=True)
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
