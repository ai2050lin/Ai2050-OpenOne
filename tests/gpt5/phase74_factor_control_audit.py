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
from statistics import mean
from typing import Any

import torch


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, release_loaded  # noqa: E402
from phase68_object_attribute_natural_exchange import get_positions, load_model, parse_csv  # noqa: E402
from phase70_object_relation_value_closure import parse_layer_pairs, pick_control  # noqa: E402
from phase72_object_relation_value_fullseq_closure import (  # noqa: E402
    capture_state,
    candidate_ids,
    fullseq_logprob,
    stats_from_scores,
)
from phase73_multitoken_value_closure import build_multitoken_items  # noqa: E402


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def find_control(items: list[dict[str, Any]], idx: int, control_type: str) -> dict[str, Any] | None:
    item = items[idx]
    if control_type == "wrong_target_same_relation_frame":
        return pick_control(items, idx)
    if control_type == "same_target_same_relation_frame":
        pool = [
            x for x in items
            if x is not item
            and x["relation"] == item["relation"]
            and x["frame_key"] == item["frame_key"]
            and x["target"] == item["target"]
            and x["object"] != item["object"]
        ]
    elif control_type == "same_object_same_relation_other_frame":
        pool = [
            x for x in items
            if x is not item
            and x["relation"] == item["relation"]
            and x["object"] == item["object"]
            and x["target"] == item["target"]
            and x["frame_key"] != item["frame_key"]
        ]
    elif control_type == "same_object_different_relation":
        pool = [
            x for x in items
            if x is not item
            and x["object"] == item["object"]
            and x["relation"] != item["relation"]
        ]
    else:
        raise ValueError(f"unknown control_type={control_type}")
    if not pool:
        return None
    return pool[(idx * 7 + len(control_type)) % len(pool)]


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [v for v in vals if v["clean_target_rank"] == 1]
    return {
        "n": len(vals),
        "eligible_n": len(eligible),
        "destroy_drop": avg([float(v["destroy_drop"]) for v in vals]),
        "restore_gain": avg([float(v["restore_gain"]) for v in vals]),
        "restore_to_clean_gap": avg([float(v["restore_to_clean_gap"]) for v in vals]),
        "eligible_destroy_drop": avg([float(v["destroy_drop"]) for v in eligible]),
        "eligible_restore_gain": avg([float(v["restore_gain"]) for v in eligible]),
        "eligible_restore_to_clean_gap": avg([float(v["restore_to_clean_gap"]) for v in eligible]),
        "clean_top1": avg([1.0 if v["clean_target_rank"] == 1 else 0.0 for v in vals]),
        "destroy_top1": avg([1.0 if v["destroy_target_rank"] == 1 else 0.0 for v in vals]),
        "restore_top1": avg([1.0 if v["restore_target_rank"] == 1 else 0.0 for v in vals]),
        "eligible_destroy_top1": avg([1.0 if v["destroy_target_rank"] == 1 else 0.0 for v in eligible]),
        "eligible_restore_top1": avg([1.0 if v["restore_target_rank"] == 1 else 0.0 for v in eligible]),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_control_path: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    by_control_relation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_control_relation_path: dict[tuple[str, str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ct = str(row["control_type"])
        dl, rl, pos, rel = int(row["destroy_layer"]), int(row["restore_layer"]), str(row["position"]), str(row["relation"])
        by_control[ct].append(row)
        by_control_path[(ct, dl, rl, pos)].append(row)
        by_control_relation[(ct, rel)].append(row)
        by_control_relation_path[(ct, rel, dl, rl, pos)].append(row)
    return {
        "by_control": {k: group_summary(v) for k, v in by_control.items()},
        "by_control_path": {f"{ct}:L{dl}->L{rl}:{pos}": group_summary(v) for (ct, dl, rl, pos), v in by_control_path.items()},
        "by_control_relation": {f"{ct}:{rel}": group_summary(v) for (ct, rel), v in by_control_relation.items()},
        "by_control_relation_path": {
            f"{ct}:{rel}:L{dl}->L{rl}:{pos}": group_summary(v)
            for (ct, rel, dl, rl, pos), v in by_control_relation_path.items()
        },
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE74_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    positions = parse_csv(args.positions)
    control_types = parse_csv(args.control_types)
    items = build_multitoken_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    log(f"Phase74 model={args.model} items={len(items)} layer_pairs={layer_pairs} positions={positions} controls={control_types}")

    results: dict[str, Any] = {
        "phase": 74,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "factor_control_audit_multitoken_fullseq",
        "layer_pairs": layer_pairs,
        "module": args.module,
        "positions": positions,
        "control_types": control_types,
        "relations": sorted({x["relation"] for x in items}),
        "num_items": len(items),
        "rows": [],
        "summary": {},
    }
    t0 = time.time()

    for destroy_layer, restore_layer in layer_pairs:
        for idx, item in enumerate(items):
            clean_pos = get_positions(tokenizer, item["clean_prompt"], item["object"])
            h_clean_r = capture_state(model, tokenizer, device, layers[restore_layer], args.module, item["clean_prompt"], args.max_length)
            values = [item["target"]] + item["distractors"]
            clean_scores = {v: fullseq_logprob(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length) for v in values}
            clean_stats = stats_from_scores(clean_scores, item["target"], item["distractors"])
            target_token_len = len(candidate_ids(tokenizer, item["target"]))

            for control_type in control_types:
                control = find_control(items, idx, control_type)
                if control is None:
                    continue
                control_pos = get_positions(tokenizer, control["clean_prompt"], control["object"])
                h_control_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, control["clean_prompt"], args.max_length)
                for pos_name in positions:
                    sp = clean_pos.get(pos_name)
                    cp = control_pos.get(pos_name)
                    if sp is None or cp is None:
                        continue
                    destroy_scores = {
                        v: fullseq_logprob(
                            model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length,
                            args.module, destroy_layer, None, int(sp), h_control_d[int(cp)], None
                        )
                        for v in values
                    }
                    restore_scores = {
                        v: fullseq_logprob(
                            model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length,
                            args.module, destroy_layer, restore_layer, int(sp), h_control_d[int(cp)], h_clean_r[int(sp)]
                        )
                        for v in values
                    }
                    destroy_stats = stats_from_scores(destroy_scores, item["target"], item["distractors"])
                    restore_stats = stats_from_scores(restore_scores, item["target"], item["distractors"])
                    results["rows"].append(
                        {
                            "destroy_layer": destroy_layer,
                            "restore_layer": restore_layer,
                            "module": args.module,
                            "position": pos_name,
                            "control_type": control_type,
                            "relation": item["relation"],
                            "frame_key": item["frame_key"],
                            "object": item["object"],
                            "target": item["target"],
                            "target_token_len": target_token_len,
                            "control_relation": control["relation"],
                            "control_frame_key": control["frame_key"],
                            "control_object": control["object"],
                            "control_target": control["target"],
                            "clean_margin": clean_stats["margin"],
                            "destroy_margin": destroy_stats["margin"],
                            "restore_margin": restore_stats["margin"],
                            "destroy_drop": clean_stats["margin"] - destroy_stats["margin"],
                            "restore_gain": restore_stats["margin"] - destroy_stats["margin"],
                            "restore_to_clean_gap": clean_stats["margin"] - restore_stats["margin"],
                            "clean_target_rank": clean_stats["rank"],
                            "destroy_target_rank": destroy_stats["rank"],
                            "restore_target_rank": restore_stats["rank"],
                            "clean_top": clean_stats["top"],
                            "destroy_top": destroy_stats["top"],
                            "restore_top": restore_stats["top"],
                        }
                    )
            if (idx + 1) % args.progress_every == 0:
                log(f"pair={destroy_layer}->{restore_layer} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = out_dir / f"{args.model}_phase74_factor_control_audit.partial.json"
        partial.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize_rows(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase74_factor_control_audit.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layer-pairs", required=True)
    parser.add_argument("--module", default="resid_out")
    parser.add_argument("--positions", default="object_last")
    parser.add_argument("--control-types", default="wrong_target_same_relation_frame,same_target_same_relation_frame,same_object_same_relation_other_frame,same_object_different_relation")
    parser.add_argument("--relations", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=112)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=24)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        run_model(args)
    finally:
        release_loaded(None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    if args.hard_exit_after_model:
        log("Hard exit after model requested.")
        os._exit(0)


if __name__ == "__main__":
    main()
