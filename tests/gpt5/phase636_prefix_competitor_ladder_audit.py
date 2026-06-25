#!/usr/bin/env python3
"""
Phase 636: Prefix Competitor Ladder and Readout Vector Builder Audit.

Phase 635 showed that natural final states move the correct prefix up the
ranking but do not create enough readout-aligned vector. This phase measures
the full token0 competitor ladder instead of a single top competitor.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase631_token0_prefix_readout_competition import get_unembed  # noqa: E402
from phase635_final_readout_projection_bridge_audit import (  # noqa: E402
    final_state_probe,
    make_all6_source_patch,
)
from phase634_multi_position_format_source_field_closure import group_layer_defaults  # noqa: E402


OUT_ROOT = Path("results/glm5_phase636_prefix_competitor_ladder_audit")
MODES = [
    "base",
    "repair_prompt",
    "source_all6",
    "final_output_repair",
    "final_output_source",
    "readout_delta",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def clean_token(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def token_category(text: str, token_id: int, prefix_id: int, old_wrong_prefix_id: int, value_prefix_ids: set[int]) -> str:
    if token_id == prefix_id:
        return "correct_prefix"
    if token_id == old_wrong_prefix_id:
        return "old_wrong_prefix"
    if token_id in value_prefix_ids:
        return "value_prefix"
    raw = text
    s = text.strip()
    low = s.lower()
    if "\n" in raw or "\r" in raw:
        return "newline"
    if low in {"to", "the", "answer", "solve", "we", "let", "there", "therefore", "because", "yes", "no"}:
        return "explanation"
    if s in {"?", ":", ".", ",", ";", "!", "-", "(", ")", "[", "]", "{", "}", "\"", "'", "`", "/"}:
        return "punctuation"
    if s == "":
        return "space"
    if re.fullmatch(r"[A-Za-z]+", s):
        return "word"
    if re.fullmatch(r"\d+", s):
        return "number"
    if not any(ch.isalnum() for ch in s):
        return "symbol"
    return "other"


def ladder_for_logits(
    tokenizer,
    logits: torch.Tensor,
    prefix_id: int,
    old_wrong_prefix_id: int,
    value_prefix_ids: set[int],
    top_k: int,
) -> Dict:
    logits = logits.float().cpu()
    prefix_logit = float(logits[prefix_id].item())
    topv, topi = torch.topk(logits, k=top_k)
    entries = []
    group_stats: Dict[str, Dict] = {}
    for rank, (value, idx) in enumerate(zip(topv, topi), start=1):
        tid = int(idx.item())
        text = tokenizer.decode([tid])
        category = token_category(text, tid, prefix_id, old_wrong_prefix_id, value_prefix_ids)
        logit = float(value.item())
        entries.append({
            "rank": rank,
            "id": tid,
            "text": text,
            "text_clean": clean_token(text),
            "category": category,
            "logit": logit,
            "prefix_minus_token": prefix_logit - logit,
            "is_prefix": tid == prefix_id,
        })
        item = group_stats.setdefault(category, {
            "category": category,
            "top_count": 0,
            "max_logit": None,
            "max_token_id": None,
            "max_token_text": "",
            "best_rank": None,
        })
        item["top_count"] += 1
        if item["max_logit"] is None or logit > item["max_logit"]:
            item["max_logit"] = logit
            item["max_token_id"] = tid
            item["max_token_text"] = text
            item["best_rank"] = rank
    for item in group_stats.values():
        item["prefix_minus_group_max"] = prefix_logit - item["max_logit"]
        item["max_token_text_clean"] = clean_token(item["max_token_text"])
    prefix_rank = int((logits > logits[prefix_id]).sum().item()) + 1
    return {
        "prefix_rank": prefix_rank,
        "prefix_logit": prefix_logit,
        "top0_id": int(topi[0].item()),
        "top0_text": tokenizer.decode([int(topi[0].item())]),
        "top0_category": entries[0]["category"],
        "top": entries,
        "groups": group_stats,
    }


def summarize(rows: List[Dict]) -> Dict:
    by_mode = {}
    by_mode_category = {}
    for row in rows:
        mode = row["mode"]
        item = by_mode.setdefault(mode, {
            "mode": mode,
            "n": 0,
            "tok0_hit": 0,
            "sum_prefix_rank": 0.0,
            "sum_prefix_margin_vs_top": 0.0,
            "top0_category": {},
            "top0_text": {},
        })
        item["n"] += 1
        item["tok0_hit"] += int(row["top0_id"] == row["prefix_id"])
        item["sum_prefix_rank"] += row["prefix_rank"]
        item["sum_prefix_margin_vs_top"] += row["prefix_margin_vs_top"]
        item["top0_category"].setdefault(row["top0_category"], 0)
        item["top0_category"][row["top0_category"]] += 1
        item["top0_text"].setdefault(row["top0_text_clean"], 0)
        item["top0_text"][row["top0_text_clean"]] += 1
        for category, group in row["groups"].items():
            key = (mode, category)
            g = by_mode_category.setdefault(key, {
                "mode": mode,
                "category": category,
                "n": 0,
                "seen_topk": 0,
                "sum_margin": 0.0,
                "sum_best_rank": 0.0,
                "winner_count": 0,
                "max_token_text": {},
            })
            g["n"] += 1
            g["seen_topk"] += int(group["top_count"] > 0)
            g["sum_margin"] += group["prefix_minus_group_max"]
            g["sum_best_rank"] += group["best_rank"]
            g["winner_count"] += int(group["best_rank"] == 1)
            txt = group["max_token_text_clean"]
            g["max_token_text"].setdefault(txt, 0)
            g["max_token_text"][txt] += 1

    mode_rows = []
    for item in by_mode.values():
        n = max(1, item["n"])
        row = dict(item)
        row["tok0_rate"] = item["tok0_hit"] / n
        row["mean_prefix_rank"] = item["sum_prefix_rank"] / n
        row["mean_prefix_margin_vs_top"] = item["sum_prefix_margin_vs_top"] / n
        row["top0_category"] = dict(sorted(row["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        row["top0_text"] = dict(sorted(row["top0_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        mode_rows.append(row)
    mode_rows.sort(key=lambda x: (x["tok0_hit"], -x["mean_prefix_rank"], x["mean_prefix_margin_vs_top"]), reverse=True)

    category_rows = []
    for item in by_mode_category.values():
        n = max(1, item["n"])
        row = dict(item)
        row["seen_rate"] = item["seen_topk"] / n
        row["mean_prefix_minus_group_max"] = item["sum_margin"] / n
        row["mean_best_rank"] = item["sum_best_rank"] / n
        row["winner_rate"] = item["winner_count"] / n
        row["max_token_text"] = dict(sorted(row["max_token_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        category_rows.append(row)
    category_rows.sort(key=lambda x: (x["mode"], x["mean_prefix_minus_group_max"]))
    return {"by_mode": mode_rows, "by_mode_category": category_rows}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        W = get_unembed(model).detach().float().cpu()
        layer_map = group_layer_defaults(args.model)
        layers_needed = sorted(set(layer_map.values()))
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        rows = []
        examples = []
        filtered = {"token_len_mismatch": 0, "not_target": 0, "source_missing": 0}
        target_seen = 0
        log(f"{args.model}: top_k={args.top_k}, source_layers={layers_needed}, raw_cases={len(raw_cases)}")

        for si, case0 in enumerate(raw_cases):
            case = dict(case0)
            case["model_name"] = args.model
            if answer_prefix_pos(tokenizer, case["base_prompt"]) != answer_prefix_pos(tokenizer, case["repair_prompt"]):
                filtered["token_len_mismatch"] += 1
                continue
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, case["correct"])
            repair = winner_stats(repair_scores, case["correct"])
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                filtered["not_target"] += 1
                continue
            target_seen += int(target_case)

            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong_ids = answer_ids(tokenizer, base["top_wrong"])
            prefix_id = correct_ids[0]
            old_wrong_prefix_id = old_wrong_ids[0]
            base_state = final_state_probe(model, tokenizer, device, case["base_prompt"])
            repair_state = final_state_probe(model, tokenizer, device, case["repair_prompt"])
            base_logits = base_state["logits"]
            top_id = int(torch.argmax(base_logits).item())
            competitor_id = top_id if top_id != prefix_id else int(torch.topk(base_logits, k=2).indices[1].item())
            direction = W[prefix_id] - W[competitor_id]
            readout_unit = direction / max(float(direction.norm().item()), 1e-8)
            hidden_norm = float(base_state["final_norm_output"].norm().item()) if base_state["final_norm_output"] is not None else 1.0
            source_patches, missing = make_all6_source_patch(model, tokenizer, device, case, layers_needed, si * 1009 + 41)
            filtered["source_missing"] += missing
            source_state = final_state_probe(model, tokenizer, device, case["base_prompt"], source_patches=source_patches)
            readout_delta = readout_unit * (hidden_norm * args.readout_scale)

            mode_specs = {
                "base": {"prompt": case["base_prompt"], "source": [], "final": None, "state": base_state},
                "repair_prompt": {"prompt": case["repair_prompt"], "source": [], "final": None, "state": repair_state},
                "source_all6": {"prompt": case["base_prompt"], "source": source_patches, "final": None, "state": source_state},
                "final_output_repair": {
                    "prompt": case["base_prompt"],
                    "source": [],
                    "final": {"kind": "output", "target": repair_state["final_norm_output"]},
                    "state": None,
                },
                "final_output_source": {
                    "prompt": case["base_prompt"],
                    "source": [],
                    "final": {"kind": "output", "target": source_state["final_norm_output"]},
                    "state": None,
                },
                "readout_delta": {
                    "prompt": case["base_prompt"],
                    "source": [],
                    "final": {"kind": "delta", "delta": readout_delta},
                    "state": None,
                },
            }
            for mode, spec in mode_specs.items():
                probe = spec["state"] or final_state_probe(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    source_patches=spec["source"],
                    final_patch=spec["final"],
                )
                ladder = ladder_for_logits(
                    tokenizer,
                    probe["logits"],
                    prefix_id,
                    old_wrong_prefix_id,
                    value_prefix_ids,
                    args.top_k,
                )
                row = {
                    "sample_idx": si,
                    "mode": mode,
                    "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                    "old_top_wrong": base["top_wrong"],
                    "prefix_id": prefix_id,
                    "prefix_text": tokenizer.decode([prefix_id]),
                    "old_wrong_prefix_id": old_wrong_prefix_id,
                    "old_wrong_prefix_text": tokenizer.decode([old_wrong_prefix_id]),
                    "prefix_rank": ladder["prefix_rank"],
                    "prefix_logit": ladder["prefix_logit"],
                    "top0_id": ladder["top0_id"],
                    "top0_text": ladder["top0_text"],
                    "top0_text_clean": clean_token(ladder["top0_text"]),
                    "top0_category": ladder["top0_category"],
                    "prefix_margin_vs_top": ladder["prefix_logit"] - probe["logits"][ladder["top0_id"]].item(),
                    "top": ladder["top"],
                    "groups": ladder["groups"],
                }
                rows.append(row)
                if len(examples) < args.example_limit:
                    examples.append(row)

        summary = summarize(rows)
        log("Best ladder modes:")
        for item in summary["by_mode"]:
            log(
                f"  {item['mode']}: tok0={item['tok0_hit']}/{item['n']} "
                f"rank={item['mean_prefix_rank']:.1f} margin={item['mean_prefix_margin_vs_top']:.3f} "
                f"topcat={item['top0_category']}"
            )
        return {
            "phase": 636,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "top_k": args.top_k,
            "readout_scale": args.readout_scale,
            "source_layer_map": layer_map,
            "value_prefix_ids": sorted(value_prefix_ids),
            "n_raw_cases": len(raw_cases),
            "n_rows": len({r["sample_idx"] for r in rows}),
            "n_mode_rows": len(rows),
            "n_target_cases_seen": target_seen,
            "filtered": filtered,
            "target_only": args.target_only,
            "summary": summary,
            "examples": examples,
            "rows": rows if args.save_rows else examples,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--readout-scale", type=float, default=0.25)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=96)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        args.top_k = min(args.top_k, 12)
        args.example_limit = 24
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 32)
        args.max_samples = max(args.max_samples, 256)
        args.top_k = max(args.top_k, 20)
        args.example_limit = max(args.example_limit, 120)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase636_{args.model}_prefix_competitor_ladder_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
