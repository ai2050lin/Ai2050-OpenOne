#!/usr/bin/env python3
"""
Phase 631: Token0 Prefix Readout Competition Audit
词元0前缀读出竞争审计

Phase 630 showed that DS7B's format/prefix bottleneck is not explained by the
scanned source layer_out carriers. This phase audits token0 logits directly and
tests whether an unembedding readout-direction injection can force the desired
prefix token into the autoregressive track.
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
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids, parse_layers  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase615_residual_state_builder_scan import collect_components  # noqa: E402
from phase624_result_state_downstream_propagation_atlas import default_downstream_layers  # noqa: E402
from phase628_prefix_format_semantic_integration import generation_eval, make_cumulative_patches, token_strings  # noqa: E402
from phase630_distributed_format_route_multisource import (  # noqa: E402
    collect_positions_components,
    greedy_generate_ids as greedy_generate_with_source,
    install_source_patch_hooks,
    source_groups,
    source_patch_from_cache,
)


OUT_ROOT = Path("results/glm5_phase631_token0_prefix_readout_competition")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def get_unembed(model) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    out = model.get_output_embeddings()
    return out.weight


def default_best_source(model_name: str) -> Tuple[str, int, str]:
    if model_name == "qwen3":
        return ("question_all", 27, "layer_out")
    if model_name == "glm4":
        return ("answer_label", 32, "layer_out")
    if model_name == "deepseek7b":
        return ("answer_label", 21, "layer_out")
    return ("prompt_last", -1, "layer_out")


def token0_logits(
    model,
    tokenizer,
    device,
    prompt: str,
    final_patch: Dict | None = None,
    source_patches: List[Tuple[int, str, List[int], List[torch.Tensor]]] | None = None,
) -> Dict:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    final_norm = get_final_norm(model)
    handle = None
    source_handles = []
    captured = {}
    if source_patches:
        source_handles = install_source_patch_hooks(model, tokenizer, prompt, source_patches)

    if final_norm is not None:
        def capture_hook(_module, _inputs, output):
            y = extract_tensor(output)
            captured["final_norm_out"] = y[0, -1].detach().float().cpu()
            if final_patch is None:
                return output
            y_new = y.clone()
            delta = final_patch["delta"].to(device=y_new.device, dtype=y_new.dtype)
            y_new[0, -1, :] = y_new[0, -1, :] + delta
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        handle = final_norm.register_forward_hook(capture_hook)

    try:
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
        return {
            "logits": logits.detach().cpu(),
            "final_norm_out": captured.get("final_norm_out"),
        }
    finally:
        if handle is not None:
            handle.remove()
        for h in source_handles:
            h.remove()


def install_readout_direction_hook(model, delta: torch.Tensor):
    final_norm = get_final_norm(model)
    if final_norm is None:
        return None

    def hook(_module, _inputs, output):
        y = extract_tensor(output)
        y_new = y.clone()
        y_new[0, -1, :] = y_new[0, -1, :] + delta.to(device=y_new.device, dtype=y_new.dtype)
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new
    return final_norm.register_forward_hook(hook)


def greedy_generate_readout(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int,
    readout_delta: torch.Tensor | None = None,
    answer_patches: List[Tuple[int, str, torch.Tensor]] | None = None,
) -> Dict:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = list(prompt_ids)
    gen = []
    top5 = []
    answer_handles = []
    if answer_patches:
        from phase629_format_prefix_gate_localization import install_answer_pos_layer_patch_hooks  # noqa: E402
        answer_handles = install_answer_pos_layer_patch_hooks(model, tokenizer, prompt, answer_patches)
    try:
        with torch.inference_mode():
            for step in range(max_new_tokens):
                h = None
                if readout_delta is not None and step == 0:
                    h = install_readout_direction_hook(model, readout_delta)
                try:
                    logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
                finally:
                    if h is not None:
                        h.remove()
                topv, topi = torch.topk(torch.log_softmax(logits, dim=-1), k=5)
                top5.append([
                    {"id": int(i), "text": tokenizer.decode([int(i)]), "logprob": float(v)}
                    for v, i in zip(topv.cpu(), topi.cpu())
                ])
                tid = int(torch.argmax(logits).item())
                gen.append(tid)
                ids.append(tid)
    finally:
        for h in answer_handles:
            h.remove()
    return {
        "ids": gen,
        "tokens": token_strings(tokenizer, gen),
        "text": tokenizer.decode(gen),
        "top5": top5,
    }


def logit_summary(tokenizer, logits: torch.Tensor, prefix_id: int, competitor_id: int, watch_ids: List[int]) -> Dict:
    probs = torch.log_softmax(logits.float(), dim=-1)
    topv, topi = torch.topk(probs, k=8)
    watch = {}
    for tid in watch_ids:
        watch[str(tid)] = {
            "id": int(tid),
            "text": tokenizer.decode([int(tid)]),
            "logit": float(logits[tid].item()),
            "logprob": float(probs[tid].item()),
        }
    return {
        "top": [
            {"id": int(i), "text": tokenizer.decode([int(i)]), "logit": float(logits[int(i)].item()), "logprob": float(v)}
            for v, i in zip(topv, topi)
        ],
        "prefix_id": int(prefix_id),
        "prefix_text": tokenizer.decode([int(prefix_id)]),
        "competitor_id": int(competitor_id),
        "competitor_text": tokenizer.decode([int(competitor_id)]),
        "prefix_logit": float(logits[prefix_id].item()),
        "competitor_logit": float(logits[competitor_id].item()),
        "prefix_minus_competitor": float((logits[prefix_id] - logits[competitor_id]).item()),
        "watch": watch,
    }


def summarize(rows: List[Dict]) -> Dict:
    modes = sorted({m for r in rows for m in r["modes"]})
    by_mode = {}
    for mode in modes:
        items = [r["modes"][mode] for r in rows if mode in r["modes"]]
        entry = {
            "mode": mode,
            "n": len(items),
            "tok0_hit": 0,
            "exact": 0,
            "wrong_exact": 0,
            "mean_prefix_margin": 0.0,
            "mean_prefix_logit": 0.0,
            "mean_competitor_logit": 0.0,
        }
        for item in items:
            ls = item["logit_summary"]
            entry["tok0_hit"] += int(item["tok0_id"] == item["prefix_id"])
            entry["mean_prefix_margin"] += ls["prefix_minus_competitor"]
            entry["mean_prefix_logit"] += ls["prefix_logit"]
            entry["mean_competitor_logit"] += ls["competitor_logit"]
            if "eval" in item:
                entry["exact"] += int(item["eval"]["exact_correct"])
                entry["wrong_exact"] += int(item["eval"]["exact_wrong"])
        n = max(1, len(items))
        for key in ["mean_prefix_margin", "mean_prefix_logit", "mean_competitor_logit"]:
            entry[key] /= n
        entry["tok0_hit_rate"] = entry["tok0_hit"] / n
        entry["exact_rate"] = entry["exact"] / n
        by_mode[mode] = entry
    best_tok0 = sorted(by_mode.values(), key=lambda x: (x["tok0_hit"], x["exact"], x["mean_prefix_margin"]), reverse=True)
    best_margin = sorted(by_mode.values(), key=lambda x: x["mean_prefix_margin"], reverse=True)
    log("Best tok0 modes:")
    for item in best_tok0[:16]:
        log(
            f"  {item['mode']}: tok0={item['tok0_hit']}/{item['n']} exact={item['exact']}/{item['n']} "
            f"margin={item['mean_prefix_margin']:.3f}"
        )
    return {"by_mode": by_mode, "best_tok0": best_tok0, "best_margin": best_margin}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        tokenization = {v: {"ids": answer_ids(tokenizer, v), "tokens": token_strings(tokenizer, answer_ids(tokenizer, v))} for v in values}
        max_new_tokens = max(len(v["ids"]) for v in tokenization.values())
        downstream_layers = parse_layers(args.downstream_layers) if args.downstream_layers else default_downstream_layers(args.model, info.n_layers)
        downstream_layers = [li for li in downstream_layers if 0 <= li < info.n_layers]
        source_group, source_layer, source_component = default_best_source(args.model)
        if args.source_group:
            source_group = args.source_group
        if args.source_layer >= 0:
            source_layer = args.source_layer
        if args.source_component:
            source_component = args.source_component
        source_layers = [source_layer]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        W = get_unembed(model).detach().float().cpu()
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0, "source_mismatch": 0}
        target_seen = 0
        log(
            f"{args.model}: source={source_group}/L{source_layer}/{source_component}, "
            f"downstream={downstream_layers}, raw_cases={len(raw_cases)}, tokenization={tokenization}"
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
            base_probe = token0_logits(model, tokenizer, device, case["base_prompt"])
            base_logits = base_probe["logits"]
            top_id = int(torch.argmax(base_logits).item())
            competitor_id = top_id if top_id != prefix_id else int(torch.topk(base_logits, k=2).indices[1].item())
            direction = W[prefix_id] - W[competitor_id]
            unit = direction / max(float(direction.norm().item()), 1e-8)
            hidden_norm = float(base_probe["final_norm_out"].norm().item()) if base_probe["final_norm_out"] is not None else 1.0

            comp_cache = {
                "base": collect_components(model, tokenizer, device, case["base_prompt"], case["correct"], downstream_layers),
                "repair": collect_components(model, tokenizer, device, case["repair_prompt"], case["correct"], downstream_layers),
            }
            semantic_cumulative = make_cumulative_patches(comp_cache, downstream_layers, "layer_out", False, si * 1009 + 23)

            base_groups = source_groups(tokenizer, case, case["base_prompt"], use_repair_subject=False)
            repair_groups = source_groups(tokenizer, case, case["repair_prompt"], use_repair_subject=True)
            source_patch = []
            base_pos = base_groups.get(source_group, [])
            repair_pos = repair_groups.get(source_group, [])
            if base_pos and len(base_pos) == len(repair_pos):
                base_cache = collect_positions_components(
                    model, tokenizer, device, case["base_prompt"], base_pos, source_layers, [source_component]
                )
                repair_cache = collect_positions_components(
                    model, tokenizer, device, case["repair_prompt"], repair_pos, source_layers, [source_component]
                )
                source_patch = source_patch_from_cache(
                    base_cache, repair_cache, base_pos, source_layer, source_component, False, si * 1009 + 41
                )
            else:
                filtered["source_mismatch"] += 1

            watch_ids = sorted({prefix_id, competitor_id, top_id, *correct_ids[:2], *old_wrong_ids[:2]})
            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "base_winner": base,
                "repair_winner": repair,
                "old_top_wrong": base["top_wrong"],
                "correct_ids": correct_ids,
                "correct_tokens": token_strings(tokenizer, correct_ids),
                "old_wrong_ids": old_wrong_ids,
                "old_wrong_tokens": token_strings(tokenizer, old_wrong_ids),
                "prefix_id": prefix_id,
                "prefix_text": tokenizer.decode([prefix_id]),
                "base_top_id": top_id,
                "base_top_text": tokenizer.decode([top_id]),
                "competitor_id": competitor_id,
                "competitor_text": tokenizer.decode([competitor_id]),
                "readout_direction_norm": float(direction.norm().item()),
                "base_hidden_norm": hidden_norm,
                "modes": {},
            }

            mode_specs = {
                "base": {"prompt": case["base_prompt"], "source": [], "answer": [], "delta": None},
                "repair_prompt": {"prompt": case["repair_prompt"], "source": [], "answer": [], "delta": None},
                "semantic_cumulative": {"prompt": case["base_prompt"], "source": [], "answer": semantic_cumulative, "delta": None},
                "best_source": {"prompt": case["base_prompt"], "source": source_patch, "answer": [], "delta": None},
                "best_source_semantic": {"prompt": case["base_prompt"], "source": source_patch, "answer": semantic_cumulative, "delta": None},
            }
            for scale in args.scales:
                delta = unit * (hidden_norm * scale)
                mode_specs[f"readout_scale{scale:g}"] = {
                    "prompt": case["base_prompt"], "source": [], "answer": [], "delta": delta,
                }
                mode_specs[f"readout_scale{scale:g}_semantic"] = {
                    "prompt": case["base_prompt"], "source": [], "answer": semantic_cumulative, "delta": delta,
                }

            for mode, spec in mode_specs.items():
                if spec["source"]:
                    probe = token0_logits(
                        model,
                        tokenizer,
                        device,
                        spec["prompt"],
                        source_patches=spec["source"],
                    )
                    logits = probe["logits"]
                    ls = logit_summary(tokenizer, logits, prefix_id, competitor_id, watch_ids)
                    gen = greedy_generate_with_source(
                        model, tokenizer, device, spec["prompt"], max_new_tokens,
                        source_patches=spec["source"],
                        answer_patches=spec["answer"],
                    )
                    tok0_id = int(torch.argmax(logits).item())
                else:
                    probe = token0_logits(
                        model,
                        tokenizer,
                        device,
                        spec["prompt"],
                        final_patch={"delta": spec["delta"]} if spec["delta"] is not None else None,
                    )
                    logits = probe["logits"]
                    ls = logit_summary(tokenizer, logits, prefix_id, competitor_id, watch_ids)
                    gen = greedy_generate_readout(
                        model, tokenizer, device, spec["prompt"], max_new_tokens,
                        readout_delta=spec["delta"],
                        answer_patches=spec["answer"],
                    )
                    tok0_id = int(torch.argmax(logits).item())
                row["modes"][mode] = {
                    "mode": mode,
                    "tok0_id": int(tok0_id),
                    "tok0_text": tokenizer.decode([int(tok0_id)]),
                    "prefix_id": int(prefix_id),
                    "competitor_id": int(competitor_id),
                    "logit_summary": ls,
                    "generation": gen,
                    "eval": generation_eval(gen, correct_ids, old_wrong_ids),
                }
            rows.append(row)

        return {
            "phase": 631,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "downstream_layers": downstream_layers,
            "source": {"group": source_group, "layer": source_layer, "component": source_component},
            "scales": args.scales,
            "tokenization": tokenization,
            "max_new_tokens": max_new_tokens,
            "n_raw_cases": len(raw_cases),
            "n_rows": len(rows),
            "n_target_cases_seen": target_seen,
            "filtered": filtered,
            "target_only": args.target_only,
            "summary": summarize(rows),
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
    parser.add_argument("--downstream-layers", default="")
    parser.add_argument("--source-group", default="")
    parser.add_argument("--source-layer", type=int, default=-1)
    parser.add_argument("--source-component", default="")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0])
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
        args.scales = [0.25]
        if not args.downstream_layers:
            if args.model == "qwen3":
                args.downstream_layers = "29,30"
            elif args.model == "glm4":
                args.downstream_layers = "34,35"
            else:
                args.downstream_layers = "22,23"
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
    out_path = out_dir / f"phase631_{args.model}_token0_prefix_readout_competition_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
