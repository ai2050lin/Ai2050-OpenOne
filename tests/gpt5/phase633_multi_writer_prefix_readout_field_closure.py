#!/usr/bin/env python3
"""
Phase 633: Multi-Writer Prefix Readout Field Closure
多写入器前缀读出场闭合

Phase 632 found natural prefix readout writer candidates, but single-writer
restore did not close DS7B. This phase tests cumulative writer sets.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
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
from phase586_distributed_value_path_patch import random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase615_residual_state_builder_scan import collect_components as collect_answer_components  # noqa: E402
from phase624_result_state_downstream_propagation_atlas import default_downstream_layers  # noqa: E402
from phase628_prefix_format_semantic_integration import generation_eval, make_cumulative_patches, token_strings  # noqa: E402
from phase632_natural_prefix_readout_writer_backtrace import (  # noqa: E402
    collect_prompt_last_components,
    greedy_generate,
    logit_metrics,
    parse_node,
    token0_logits,
)


OUT_ROOT = Path("results/glm5_phase633_multi_writer_prefix_readout_field_closure")
PHASE632_ROOT = Path("results/glm5_phase632_natural_prefix_readout_writer_backtrace")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def load_phase632_rank(model_name: str, phase632_dir: Path) -> Dict:
    path = phase632_dir / f"phase632_{model_name}_natural_prefix_readout_writer_backtrace_confirm.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase632 rank source: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def equiv_key(node: str) -> Tuple[str, int, str]:
    li, component = parse_node(node)
    if component == "layer_out":
        return ("residual_after_layer", li, "resid")
    if component == "layer_input":
        return ("residual_after_layer", li - 1, "resid")
    return ("component", li, component)


def dedup_rank(scan_rank: List[Dict], max_nodes: int) -> List[str]:
    chosen = []
    seen = set()
    for item in scan_rank:
        node = item["node"]
        if item.get("mean_margin_delta", 0.0) <= 0:
            continue
        key = equiv_key(node)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(node)
        if len(chosen) >= max_nodes:
            break
    return chosen


def cumulative_sets(nodes: List[str], sizes: List[int]) -> Dict[str, List[str]]:
    out = {}
    for size in sizes:
        use = nodes[: min(size, len(nodes))]
        if use:
            out[f"top{len(use)}"] = use
    return out


def summarize_modes(rows: List[Dict]) -> Dict:
    stats = {}
    for row in rows:
        key = row["mode"]
        item = stats.setdefault(key, {
            "mode": key,
            "n": 0,
            "tok0_hit": 0,
            "exact": 0,
            "wrong_exact": 0,
            "sum_margin": 0.0,
            "top0_text": {},
        })
        item["n"] += 1
        item["tok0_hit"] += int(row["tok0_id"] == row["prefix_id"])
        item["exact"] += int(row["eval"]["exact_correct"])
        item["wrong_exact"] += int(row["eval"]["exact_wrong"])
        item["sum_margin"] += row["prefix_margin"]
        item["top0_text"].setdefault(row["tok0_text"], 0)
        item["top0_text"][row["tok0_text"]] += 1
    out = []
    for item in stats.values():
        n = max(1, item["n"])
        row = dict(item)
        row["tok0_rate"] = item["tok0_hit"] / n
        row["exact_rate"] = item["exact"] / n
        row["wrong_exact_rate"] = item["wrong_exact"] / n
        row["mean_prefix_margin"] = item["sum_margin"] / n
        row["top0_text"] = dict(sorted(row["top0_text"].items(), key=lambda kv: kv[1], reverse=True)[:6])
        out.append(row)
    out.sort(key=lambda x: (x["exact"], x["tok0_hit"], x["mean_prefix_margin"]), reverse=True)
    return {"by_mode": out}


def make_patch_targets(
    nodes: List[str],
    base_cache: Dict[int, Dict[str, torch.Tensor]],
    repair_cache: Dict[int, Dict[str, torch.Tensor]],
    mode: str,
    seed: int,
) -> List[Tuple[int, str, torch.Tensor]]:
    patches = []
    for ni, node in enumerate(nodes):
        li, component = parse_node(node)
        base = base_cache.get(li, {}).get(component)
        repair = repair_cache.get(li, {}).get(component)
        if base is None or repair is None:
            continue
        delta = repair - base
        if mode == "restore":
            target = repair
        elif mode == "random":
            target = base + random_same_norm(delta, seed + ni * 997)
        elif mode == "reverse":
            target = base - delta
        elif mode == "remove":
            target = base
        else:
            raise ValueError(mode)
        patches.append((li, component, target))
    return patches


def run_model(args) -> Dict:
    rank_source = load_phase632_rank(args.model, Path(args.phase632_dir))
    candidate_nodes = dedup_rank(rank_source["scan_rank"], args.max_nodes)
    set_defs = cumulative_sets(candidate_nodes, args.set_sizes)
    needed_layers = sorted({parse_node(node)[0] for node in candidate_nodes})
    components = sorted({parse_node(node)[1] for node in candidate_nodes})

    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        tokenization = {v: {"ids": answer_ids(tokenizer, v), "tokens": token_strings(tokenizer, answer_ids(tokenizer, v))} for v in values}
        max_new_tokens = max(len(v["ids"]) for v in tokenization.values())
        downstream_layers = rank_source.get("downstream_layers") or default_downstream_layers(args.model, info.n_layers)
        downstream_layers = [li for li in downstream_layers if 0 <= li < info.n_layers]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        filtered = {"token_len_mismatch": 0, "not_target": 0}
        rows = []
        target_seen = 0
        log(
            f"{args.model}: candidate_nodes={candidate_nodes}, set_defs={set_defs}, "
            f"needed_layers={needed_layers}, components={components}, raw_cases={len(raw_cases)}"
        )

        for si, case in enumerate(raw_cases):
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
            base_logits = token0_logits(model, tokenizer, device, case["base_prompt"])["logits"]
            top_id = int(torch.argmax(base_logits).item())
            competitor_id = top_id if top_id != prefix_id else int(torch.topk(base_logits, k=2).indices[1].item())
            base_cache = collect_prompt_last_components(model, tokenizer, device, case["base_prompt"], needed_layers, components)
            repair_cache = collect_prompt_last_components(model, tokenizer, device, case["repair_prompt"], needed_layers, components)
            answer_cache = {
                "base": collect_answer_components(model, tokenizer, device, case["base_prompt"], case["correct"], downstream_layers),
                "repair": collect_answer_components(model, tokenizer, device, case["repair_prompt"], case["correct"], downstream_layers),
            }
            semantic_cumulative = make_cumulative_patches(answer_cache, downstream_layers, "layer_out", False, si * 1009 + 23)

            mode_specs = {
                "base": {"prompt": case["base_prompt"], "prompt_patches": [], "answer_patches": []},
                "repair_prompt": {"prompt": case["repair_prompt"], "prompt_patches": [], "answer_patches": []},
                "semantic_cumulative": {
                    "prompt": case["base_prompt"],
                    "prompt_patches": [],
                    "answer_patches": semantic_cumulative,
                },
            }
            for set_name, nodes in set_defs.items():
                restore_patches = make_patch_targets(nodes, base_cache, repair_cache, "restore", si * 1009 + len(nodes))
                random_patches = make_patch_targets(nodes, base_cache, repair_cache, "random", si * 1009 + len(nodes) * 7)
                reverse_patches = make_patch_targets(nodes, base_cache, repair_cache, "reverse", si * 1009 + len(nodes) * 11)
                remove_patches = make_patch_targets(nodes, base_cache, repair_cache, "remove", si * 1009 + len(nodes) * 13)
                mode_specs[f"{set_name}_restore"] = {
                    "prompt": case["base_prompt"], "prompt_patches": restore_patches, "answer_patches": []
                }
                mode_specs[f"{set_name}_restore_semantic"] = {
                    "prompt": case["base_prompt"], "prompt_patches": restore_patches, "answer_patches": semantic_cumulative
                }
                mode_specs[f"{set_name}_random_semantic"] = {
                    "prompt": case["base_prompt"], "prompt_patches": random_patches, "answer_patches": semantic_cumulative
                }
                mode_specs[f"{set_name}_reverse_semantic"] = {
                    "prompt": case["base_prompt"], "prompt_patches": reverse_patches, "answer_patches": semantic_cumulative
                }
                mode_specs[f"{set_name}_remove_from_repair"] = {
                    "prompt": case["repair_prompt"], "prompt_patches": remove_patches, "answer_patches": []
                }

            for mode, spec in mode_specs.items():
                logits = token0_logits(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    prompt_patches=spec["prompt_patches"],
                )["logits"]
                metrics = logit_metrics(logits, prefix_id, competitor_id)
                gen = greedy_generate(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    max_new_tokens,
                    prompt_patches=spec["prompt_patches"],
                    answer_patches=spec["answer_patches"],
                )
                ev = generation_eval(gen, correct_ids, old_wrong_ids)
                rows.append({
                    "sample_idx": si,
                    "mode": mode,
                    "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                    "prefix_id": prefix_id,
                    "prefix_text": tokenizer.decode([prefix_id]),
                    "competitor_id": competitor_id,
                    "competitor_text": tokenizer.decode([competitor_id]),
                    "tok0_id": metrics["tok0_id"],
                    "tok0_text": tokenizer.decode([metrics["tok0_id"]]),
                    "prefix_margin": metrics["prefix_margin"],
                    "eval": ev,
                    "generation_text": gen["text"] if len(rows) < 180 else "",
                })

        summary = summarize_modes(rows)
        log("Best cumulative modes:")
        for item in summary["by_mode"][:20]:
            log(
                f"  {item['mode']}: tok0={item['tok0_hit']}/{item['n']} "
                f"exact={item['exact']}/{item['n']} margin={item['mean_prefix_margin']:.3f}"
            )
        return {
            "phase": 633,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "phase632_source": str(Path(args.phase632_dir) / f"phase632_{args.model}_natural_prefix_readout_writer_backtrace_confirm.json"),
            "candidate_nodes": candidate_nodes,
            "set_defs": set_defs,
            "downstream_layers": downstream_layers,
            "tokenization": tokenization,
            "max_new_tokens": max_new_tokens,
            "n_raw_cases": len(raw_cases),
            "n_rows": len({r["sample_idx"] for r in rows}),
            "n_mode_rows": len(rows),
            "n_target_cases_seen": target_seen,
            "filtered": filtered,
            "target_only": args.target_only,
            "summary": summary,
            "rows": rows,
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
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--set-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--phase632-dir", default=str(PHASE632_ROOT))
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        args.max_nodes = 4
        args.set_sizes = [1, 2, 4]
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 32)
        args.max_samples = max(args.max_samples, 256)
        args.max_nodes = max(args.max_nodes, 12)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase633_{args.model}_multi_writer_prefix_readout_field_closure_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
