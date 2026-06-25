#!/usr/bin/env python3
"""
Phase 634: Multi-Position Format Source Field Closure
多位置格式源场闭合

Phase 633 ruled out prompt_last multi-writer closure. This phase tests whether
multi-position source/format groups can close the token0 prefix gate.
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
from phase630_distributed_format_route_multisource import (  # noqa: E402
    GROUP_ORDER,
    collect_positions_components,
    greedy_generate_ids,
    source_groups,
)
from phase631_token0_prefix_readout_competition import token0_logits  # noqa: E402
from phase632_natural_prefix_readout_writer_backtrace import logit_metrics  # noqa: E402


OUT_ROOT = Path("results/glm5_phase634_multi_position_format_source_field_closure")
COMPONENT = "layer_out"


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def group_layer_defaults(model_name: str) -> Dict[str, int]:
    if model_name == "qwen3":
        return {
            "prompt_last": 27,
            "answer_label": 27,
            "question_mark_answer": 27,
            "relation_tail": 27,
            "question_subject": 27,
            "question_all": 27,
        }
    if model_name == "glm4":
        return {
            "prompt_last": 32,
            "answer_label": 32,
            "question_mark_answer": 32,
            "relation_tail": 32,
            "question_subject": 32,
            "question_all": 32,
        }
    if model_name == "deepseek7b":
        return {
            "prompt_last": 25,
            "answer_label": 21,
            "question_mark_answer": 21,
            "relation_tail": 23,
            "question_subject": 21,
            "question_all": 20,
        }
    return {g: -1 for g in GROUP_ORDER}


def group_set_defaults() -> Dict[str, List[str]]:
    return {
        "single_prompt_last": ["prompt_last"],
        "single_answer_label": ["answer_label"],
        "single_question_mark_answer": ["question_mark_answer"],
        "single_relation_tail": ["relation_tail"],
        "single_question_all": ["question_all"],
        "answer_prompt": ["answer_label", "prompt_last"],
        "qma_prompt": ["question_mark_answer", "prompt_last"],
        "relation_answer_prompt": ["relation_tail", "answer_label", "prompt_last"],
        "question_all_answer_prompt": ["question_all", "answer_label", "prompt_last"],
        "answer_qma_relation_prompt": ["answer_label", "question_mark_answer", "relation_tail", "prompt_last"],
        "all6": list(GROUP_ORDER),
    }


def make_group_patch(
    base_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    repair_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    positions: List[int],
    layer_idx: int,
    component: str,
    mode: str,
    seed: int,
) -> List[Tuple[int, str, List[int], List[torch.Tensor]]]:
    base_vals = base_cache.get(layer_idx, {}).get(component, [])
    repair_vals = repair_cache.get(layer_idx, {}).get(component, [])
    if len(base_vals) != len(repair_vals) or len(base_vals) != len(positions) or not base_vals:
        return []
    targets = []
    for pi, (base, repair) in enumerate(zip(base_vals, repair_vals)):
        delta = repair.float().cpu() - base.float().cpu()
        if mode == "restore":
            target = repair.float().cpu()
        elif mode == "random":
            target = base.float().cpu() + random_same_norm(delta, seed + layer_idx * 997 + pi * 101)
        elif mode == "reverse":
            target = base.float().cpu() - delta
        elif mode == "remove":
            target = base.float().cpu()
        else:
            raise ValueError(mode)
        targets.append(target)
    return [(layer_idx, component, positions, targets)]


def merge_patches(patches: List[Tuple[int, str, List[int], List[torch.Tensor]]]):
    merged: Dict[Tuple[int, str], Dict[int, torch.Tensor]] = {}
    for li, comp, positions, targets in patches:
        slot = merged.setdefault((li, comp), {})
        for p, t in zip(positions, targets):
            slot[int(p)] = t.float().cpu()
    out = []
    for (li, comp), mapping in sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        positions = sorted(mapping)
        out.append((li, comp, positions, [mapping[p] for p in positions]))
    return out


def summarize_modes(rows: List[Dict]) -> Dict:
    stats = {}
    for row in rows:
        item = stats.setdefault(row["mode"], {
            "mode": row["mode"],
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


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layer_map = group_layer_defaults(args.model)
        if args.layer_overrides:
            for spec in args.layer_overrides.split(","):
                if not spec.strip():
                    continue
                group, layer_s = spec.split(":")
                layer_map[group.strip()] = int(layer_s)
        set_defs = group_set_defaults()
        if args.sets:
            allowed = {s.strip() for s in args.sets.split(",") if s.strip()}
            set_defs = {k: v for k, v in set_defs.items() if k in allowed}
        groups_needed = sorted({g for groups in set_defs.values() for g in groups})
        layers_needed = sorted({layer_map[g] for g in groups_needed if layer_map[g] >= 0})
        downstream_layers = default_downstream_layers(args.model, info.n_layers)
        downstream_layers = [li for li in downstream_layers if 0 <= li < info.n_layers]
        values = CANDIDATE_VALUES[:4]
        tokenization = {v: {"ids": answer_ids(tokenizer, v), "tokens": token_strings(tokenizer, answer_ids(tokenizer, v))} for v in values}
        max_new_tokens = max(len(v["ids"]) for v in tokenization.values())
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        filtered = {"token_len_mismatch": 0, "not_target": 0, "group_mismatch": 0}
        rows = []
        target_seen = 0
        log(
            f"{args.model}: layer_map={layer_map}, set_defs={set_defs}, groups_needed={groups_needed}, "
            f"layers_needed={layers_needed}, raw_cases={len(raw_cases)}"
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

            answer_cache = {
                "base": collect_answer_components(model, tokenizer, device, case["base_prompt"], case["correct"], downstream_layers),
                "repair": collect_answer_components(model, tokenizer, device, case["repair_prompt"], case["correct"], downstream_layers),
            }
            semantic_cumulative = make_cumulative_patches(answer_cache, downstream_layers, "layer_out", False, si * 1009 + 23)

            base_groups = source_groups(tokenizer, case, case["base_prompt"], use_repair_subject=False)
            repair_groups = source_groups(tokenizer, case, case["repair_prompt"], use_repair_subject=True)
            group_payload = {}
            for group in groups_needed:
                base_pos = base_groups.get(group, [])
                repair_pos = repair_groups.get(group, [])
                if not base_pos or len(base_pos) != len(repair_pos):
                    filtered["group_mismatch"] += 1
                    continue
                li = layer_map[group]
                base_cache = collect_positions_components(
                    model, tokenizer, device, case["base_prompt"], base_pos, [li], [COMPONENT]
                )
                repair_cache = collect_positions_components(
                    model, tokenizer, device, case["repair_prompt"], repair_pos, [li], [COMPONENT]
                )
                group_payload[group] = {
                    "layer": li,
                    "base_pos": base_pos,
                    "repair_pos": repair_pos,
                    "base_cache": base_cache,
                    "repair_cache": repair_cache,
                }

            mode_specs = {
                "base": {"prompt": case["base_prompt"], "source": [], "answer": []},
                "repair_prompt": {"prompt": case["repair_prompt"], "source": [], "answer": []},
                "semantic_cumulative": {"prompt": case["base_prompt"], "source": [], "answer": semantic_cumulative},
            }

            for set_name, groups in set_defs.items():
                if any(g not in group_payload for g in groups):
                    continue
                for mode in ["restore", "random", "reverse"]:
                    patches = []
                    for gi, group in enumerate(groups):
                        payload = group_payload[group]
                        patches.extend(make_group_patch(
                            payload["base_cache"],
                            payload["repair_cache"],
                            payload["base_pos"],
                            payload["layer"],
                            COMPONENT,
                            mode,
                            si * 1009 + gi * 173 + len(set_name),
                        ))
                    patches = merge_patches(patches)
                    mode_specs[f"{set_name}_{mode}"] = {
                        "prompt": case["base_prompt"], "source": patches, "answer": []
                    }
                    mode_specs[f"{set_name}_{mode}_semantic"] = {
                        "prompt": case["base_prompt"], "source": patches, "answer": semantic_cumulative
                    }
                remove_patches = []
                for group in groups:
                    payload = group_payload[group]
                    remove_patches.extend(make_group_patch(
                        payload["base_cache"],
                        payload["repair_cache"],
                        payload["repair_pos"],
                        payload["layer"],
                        COMPONENT,
                        "remove",
                        si * 1009 + len(set_name),
                    ))
                mode_specs[f"{set_name}_remove_from_repair"] = {
                    "prompt": case["repair_prompt"], "source": merge_patches(remove_patches), "answer": []
                }

            for mode, spec in mode_specs.items():
                logits = token0_logits(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    source_patches=spec["source"],
                )["logits"]
                metrics = logit_metrics(logits, prefix_id, competitor_id)
                gen = greedy_generate_ids(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    max_new_tokens,
                    source_patches=spec["source"],
                    answer_patches=spec["answer"],
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
                    "generation_text": gen["text"] if len(rows) < 220 else "",
                })

        summary = summarize_modes(rows)
        log("Best multi-position modes:")
        for item in summary["by_mode"][:24]:
            log(
                f"  {item['mode']}: tok0={item['tok0_hit']}/{item['n']} "
                f"exact={item['exact']}/{item['n']} margin={item['mean_prefix_margin']:.3f}"
            )
        return {
            "phase": 634,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layer_map": layer_map,
            "set_defs": set_defs,
            "groups_needed": groups_needed,
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
    parser.add_argument("--layer-overrides", default="")
    parser.add_argument("--sets", default="")
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        args.sets = "single_prompt_last,question_all_answer_prompt,all6"
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 32)
        args.max_samples = max(args.max_samples, 256)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase634_{args.model}_multi_position_format_source_field_closure_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
