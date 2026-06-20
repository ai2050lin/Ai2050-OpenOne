#!/usr/bin/env python3
"""
Phase 561: Gated Continuous Injection and Low-Strength Prototype Control

Phase560 showed that continuous_static restore is destructive. This phase tests
whether low-strength gated blending can maintain donor/prototype information
without overwriting the autoregressive trajectory.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
import phase548_paraphrase_candidate_robustness as p548  # noqa: E402
import phase558_prototype_object_binding_audit as p558  # noqa: E402
import phase559_prototype_generation_closure as p559  # noqa: E402
from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import load_model_bf16_flash  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows  # noqa: E402


OUT_ROOT = Path("results/glm5_phase561_gated_continuous_injection")
DEFAULT_ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
DEFAULT_CONDITIONS = [
    "baseline",
    "resid_remove_perp",
    "resid_donor_vehicle_repeat2_add",
    "resid_donor_vehicle_repeat4_add",
    "resid_donor_vehicle_mean_cache_add",
    "resid_donor_vehicle_random_cache_add",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def install_surgery_hooks(
    layers: list[Any],
    layer_ids: list[int],
    components_by_layer: dict[str, dict[str, np.ndarray]],
    condition: str,
    batch_size: int,
    pos_value: int,
    restore_cache: dict[int, torch.Tensor],
    remove_scale: float,
    add_alpha: float,
    restore_beta: float,
) -> list[Any]:
    plan = p558.condition_plan(condition)
    handles: list[Any] = []
    if plan["site"] == "none":
        return handles
    pos_cpu = torch.full((batch_size,), int(pos_value), dtype=torch.long)
    for layer_id in layer_ids:
        layer = layers[layer_id]
        site = p559.module_for_site(layer, plan["site"])
        direction_np = components_by_layer[str(layer_id)][plan["component"]]
        direction_cpu = torch.tensor(p559.normalize_vec(direction_np), dtype=torch.float32)
        cached = restore_cache.get(layer_id)
        should_remove = bool(plan["remove"])
        should_add = bool(plan["add"] and plan["site"] == "resid")
        should_restore = bool(plan["restore"] is not None and cached is not None)

        def make_hook(d_vec_cpu, p_vec_cpu, cached_vec, rm, ad, rs, site_name, beta):
            def hook(_module, _inp, output):
                hs = p559.tensor_from_output(output)
                out = hs
                p_dev = p_vec_cpu.to(out.device)
                if rm:
                    out = p559.project_remove(out, p_dev, d_vec_cpu, remove_scale)
                if ad and site_name == "resid":
                    out = p559.add_direction(out, p_dev, d_vec_cpu, add_alpha)
                if rs and cached_vec is not None:
                    bidx = torch.arange(out.shape[0], device=out.device)
                    out = out.clone()
                    donor = cached_vec.to(out.device, dtype=out.dtype)
                    if beta >= 0.999:
                        out[bidx, p_dev, :] = donor
                    else:
                        cur = out[bidx, p_dev, :]
                        out[bidx, p_dev, :] = ((1.0 - float(beta)) * cur + float(beta) * donor).to(out.dtype)
                return p559.replace_output(output, out)
            return hook

        handles.append(site.register_forward_hook(
            make_hook(direction_cpu, pos_cpu, cached, should_remove, should_add, should_restore, plan["site"], restore_beta)
        ))
    return handles


def generate_batch(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    donor_prompts: list[str] | None,
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    condition: str,
    groups: dict[str, list[int]],
    mode: str,
    max_new_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    remove_scale: float,
    add_alpha: float,
    max_length: int,
    surgery_mode: str,
    restore_beta: float,
) -> tuple[list[list[int]], list[str], list[str], list[float], list[float]]:
    plan = p558.condition_plan(condition)
    rng = np.random.default_rng(seed)
    batch_size = len(prompts)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = original_padding_side

    restore_cache: dict[int, torch.Tensor] = {}
    if plan["restore"] is not None and donor_prompts is not None:
        donor_enc = tokenizer(donor_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        donor_batch = {k: v.to(device) for k, v in donor_enc.items()}
        donor_pos = donor_batch["attention_mask"].sum(dim=1) - 1
        raw_cache = p559.collect_donor_cache(
            model, layers, donor_batch, donor_pos, components_by_layer, layer_ids,
            plan["site"], plan["restore"], add_alpha,
        )
        restore_cache = p559.transform_restore_cache(raw_cache, plan.get("donor_variant"), 0)

    handles = install_surgery_hooks(
        layers, layer_ids, components_by_layer, condition, batch_size, answer_pos,
        restore_cache, remove_scale, add_alpha, restore_beta
    )
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits_step0 = out.logits[:, answer_pos, :].float().cpu().numpy()
    for h in handles:
        h.remove()

    toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits_step0]
    first_types = [p544.token_type(int(t), groups) for t in toks]
    first_target_ranks = [float(p544.best_rank(row, groups["target"])) for row in logits_step0]
    first_competitor_ranks = [float(p544.best_rank(row, groups["competitor"])) for row in logits_step0]
    generated: list[list[int]] = [[int(t)] for t in toks]

    full_attn_mask = attention_mask
    for _step in range(1, max_new_tokens):
        new_ids = torch.tensor([[t] for t in toks], dtype=torch.long, device=device)
        new_mask_col = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
        full_attn_mask = torch.cat([full_attn_mask, new_mask_col], dim=1)
        handles = []
        if surgery_mode == "continuous_blend":
            handles = install_surgery_hooks(
                layers, layer_ids, components_by_layer, condition, batch_size, 0,
                restore_cache, remove_scale, add_alpha, restore_beta
            )
        with torch.inference_mode():
            out = model(input_ids=new_ids, attention_mask=full_attn_mask,
                        past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :].float().cpu().numpy()
        for h in handles:
            h.remove()
        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        for i, t in enumerate(toks):
            generated[i].append(int(t))

    suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated, suffixes, first_types, first_target_ranks, first_competitor_ranks


def run_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt_rows: list[dict[str, str]],
    donor_rows: list[dict[str, str]] | None,
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    condition: str,
    groups: dict[str, list[int]],
    pair: str,
    mode: str,
    max_new_tokens: int,
    sample_seeds: list[int],
    temperature: float,
    top_p: float,
    remove_scale: float,
    add_alpha: float,
    max_length: int,
    batch_size: int,
    surgery_mode: str,
    restore_beta: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pos_label, neg_label = PAIR_SPECS[pair]
    all_records: list[dict[str, Any]] = []
    seed_aggs: list[dict[str, Any]] = []
    for seed in sample_seeds:
        batch_generated: list[list[int]] = []
        batch_suffixes: list[str] = []
        batch_first_types: list[str] = []
        batch_first_target_ranks: list[float] = []
        batch_first_competitor_ranks: list[float] = []
        for start in range(0, len(prompt_rows), batch_size):
            b_prompts = [r["prompt"] for r in prompt_rows[start:start + batch_size]]
            b_donor = None
            if donor_rows is not None:
                b_donor = [r["prompt"] for r in donor_rows[start:start + batch_size]]
            gen, suf, ft, ftr, fcr = generate_batch(
                model, tokenizer, device, layers, b_prompts, b_donor,
                components_by_layer, layer_ids, condition, groups, mode, max_new_tokens,
                seed, temperature, top_p, remove_scale, add_alpha, max_length, surgery_mode,
                restore_beta,
            )
            batch_generated.extend(gen)
            batch_suffixes.extend(suf)
            batch_first_types.extend(ft)
            batch_first_target_ranks.extend(ftr)
            batch_first_competitor_ranks.extend(fcr)

        records = []
        for i, (row, suffix, ids, ft, ftr, fcr) in enumerate(zip(
            prompt_rows, batch_suffixes, batch_generated,
            batch_first_types, batch_first_target_ranks, batch_first_competitor_ranks,
        )):
            cls = p548.classify_suffix(suffix, row["object"], pos_label, neg_label)
            rec = {
                "prompt_index": i,
                "object": row["object"],
                "prompt": row["prompt"],
                "donor_object": donor_rows[i]["object"] if donor_rows is not None else "",
                "donor_prompt": donor_rows[i]["prompt"] if donor_rows is not None else "",
                "condition": condition,
                "restore_beta": restore_beta,
                "seed": seed,
                "generated_suffix": suffix,
                "generated_ids": ids,
                "first_type": ft,
                "first_target_rank": float(ftr),
                "first_competitor_rank": float(fcr),
                **cls,
                **format_audit(suffix, ids),
            }
            records.append(rec)
            all_records.append(rec)
        seed_agg = p548.aggregate(records)
        add_format_aggregates(seed_agg, records)
        seed_agg["seed"] = seed
        seed_aggs.append(seed_agg)
    agg = p548.aggregate(all_records)
    add_format_aggregates(agg, all_records)
    agg["seed_aggregates"] = seed_aggs
    return agg, all_records


def format_audit(suffix: str, ids: list[int]) -> dict[str, Any]:
    text = suffix.strip()
    lower = text.lower()
    words = [w for w in lower.replace("\n", " ").split(" ") if w]
    repeated_word = any(words[i] == words[i - 1] for i in range(1, len(words)))
    repeated_token = any(ids[i] == ids[i - 1] == ids[i - 2] for i in range(2, len(ids)))
    punct_count = sum(1 for ch in text if ch in ".,;:!?")
    alpha_count = sum(1 for ch in text if ch.isalpha())
    short_output = len(text) < 3 or alpha_count < 2
    punctuation_break = punct_count >= 4 or "..." in text or "??" in text or "!!" in text
    return {
        "generated_char_len": len(text),
        "generated_token_len": len(ids),
        "short_output": short_output,
        "repetition_break": repeated_word or repeated_token,
        "punctuation_break": punctuation_break,
        "format_break": short_output or repeated_word or repeated_token or punctuation_break,
    }


def add_format_aggregates(agg: dict[str, Any], records: list[dict[str, Any]]) -> None:
    n = max(1, len(records))
    agg["avg_generated_char_len"] = float(sum(r["generated_char_len"] for r in records) / n)
    agg["avg_generated_token_len"] = float(sum(r["generated_token_len"] for r in records) / n)
    agg["short_output_rate"] = float(sum(1 for r in records if r["short_output"]) / n)
    agg["repetition_break_rate"] = float(sum(1 for r in records if r["repetition_break"]) / n)
    agg["punctuation_break_rate"] = float(sum(1 for r in records if r["punctuation_break"]) / n)
    agg["format_break_rate"] = float(sum(1 for r in records if r["format_break"]) / n)


def compact_row(
    combo_name: str,
    layer_ids: list[int],
    route: dict[str, str],
    condition: str,
    source_condition: str,
    restore_beta: float | None,
    row: dict[str, Any],
    base: dict[str, Any],
    remove_ref: dict[str, Any],
) -> dict[str, Any]:
    plan = p558.condition_plan(source_condition)
    clean_delta = row["clean_non_object_rate"] - base["clean_non_object_rate"]
    score_delta = row["clean_non_object_score"] - base["clean_non_object_score"]
    label_delta = row["any_label_violation_rate"] - base["any_label_violation_rate"]
    remove_delta = remove_ref["clean_non_object_rate"] - base["clean_non_object_rate"]
    steering_gain = row["clean_non_object_rate"] - base["clean_non_object_rate"]
    if condition == "resid_remove_perp":
        cls = "generation_drop" if clean_delta <= -0.06 else "no_generation_drop"
    elif "_donor_" in condition:
        cls = "positive_steer" if steering_gain >= 0.08 and label_delta <= 0.08 else "weak_or_fail"
    elif clean_delta >= 0.08:
        cls = "positive_add"
    else:
        cls = "flat"
    return {
        "combo": combo_name,
        "layers": layer_ids,
        "route": route["name"],
        "recipient_scaffold": route["recipient_scaffold"],
        "donor_scaffold": route["donor_scaffold"],
        "mode": route["mode"],
        "condition": condition,
        "source_condition": source_condition,
        "restore_beta": restore_beta,
        "donor_category": plan.get("donor_category") or "",
        "donor_variant": plan.get("donor_variant") or "",
        "base_clean_non_object_rate": base["clean_non_object_rate"],
        "clean_non_object_rate": row["clean_non_object_rate"],
        "label_violation_rate": row["any_label_violation_rate"],
        "object_echo_rate": row["object_echo_rate"],
        "prompt_echo_rate": row["prompt_echo_rate"],
        "clean_non_object_score": row["clean_non_object_score"],
        "format_break_rate": row.get("format_break_rate", 0.0),
        "short_output_rate": row.get("short_output_rate", 0.0),
        "repetition_break_rate": row.get("repetition_break_rate", 0.0),
        "punctuation_break_rate": row.get("punctuation_break_rate", 0.0),
        "avg_generated_char_len": row.get("avg_generated_char_len", 0.0),
        "avg_generated_token_len": row.get("avg_generated_token_len", 0.0),
        "clean_delta": float(clean_delta),
        "score_delta": float(score_delta),
        "label_delta": float(label_delta),
        "remove_delta": float(remove_delta),
        "steering_gain": float(steering_gain),
        "class": cls,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    routes = p558.parse_routes(args.routes)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    conditions = p558.parse_csv(args.conditions)
    sample_seeds = p558.parse_int_csv(args.sample_seeds)
    blend_betas = [float(x) for x in p558.parse_csv(args.blend_betas)]

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        if len(windows) != 1:
            raise ValueError(f"Phase561 expects one window, got {windows}")
        _, window = next(iter(windows.items()))
        combos = p558.combo_layers(window, args.layer_sets)
        all_layers = sorted(set(itertools.chain.from_iterable(combos.values())))
        W_U = get_W_U(model, args.model).astype(np.float32)
        groups = p544.token_groups(tokenizer, pair)
        prompt_sets = p548.build_prompts(pair, args.test_n, scaffolds)
        object_audit = p559.object_name_audit(pair, args.test_n, tokenizer)
        components_by_layer = p558.build_components_by_layer(
            model, tokenizer, device, pair, all_layers, args.train_n, args.batch_size, args.max_length, W_U
        )
        run_specs: list[dict[str, Any]] = []
        for condition in conditions:
            if "_donor_" in condition and args.surgery_mode == "continuous_blend":
                for beta in blend_betas:
                    run_specs.append({
                        "key": f"{condition}__b{str(beta).replace('.', 'p')}",
                        "condition": condition,
                        "restore_beta": beta,
                    })
            else:
                run_specs.append({"key": condition, "condition": condition, "restore_beta": 1.0})
        log(
            f"{args.model}: phase561 surgery={args.surgery_mode}, combos={combos}, "
            f"routes={[r['name'] for r in routes]}, units={len(run_specs)}"
        )

        audit: dict[str, Any] = {}
        compact: list[dict[str, Any]] = []
        saved_samples: list[dict[str, Any]] = []
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"phase561_{args.model}_{args.surgery_mode}_checkpoint.json"
        total_units = len(combos) * len(routes) * len(run_specs)
        done_units = 0
        t_start = time.time()
        for combo_name, layer_ids in combos.items():
            audit[combo_name] = {"layers": layer_ids, "rows": {}}
            for route in routes:
                audit[combo_name]["rows"][route["name"]] = {}
                prompt_rows = prompt_sets[route["recipient_scaffold"]]
                for spec in run_specs:
                    run_key = spec["key"]
                    condition = spec["condition"]
                    restore_beta = float(spec["restore_beta"])
                    plan = p558.condition_plan(condition)
                    donor_rows = p558.donor_rows_for(
                        pair, route["donor_scaffold"], plan.get("donor_category"),
                        plan.get("donor_variant"), args.test_n,
                    )
                    t_cond = time.time()
                    agg, records = run_condition(
                        model, tokenizer, device, layers, prompt_rows, donor_rows,
                        components_by_layer, layer_ids, condition, groups, pair, route["mode"],
                        args.max_new_tokens, sample_seeds, args.temperature, args.top_p,
                        args.remove_scale, args.add_alpha, args.max_length, args.batch_size,
                        args.surgery_mode, restore_beta,
                    )
                    audit[combo_name]["rows"][route["name"]][run_key] = agg
                    saved_samples.extend(records[: args.samples_per_row])
                    done_units += 1
                    elapsed = time.time() - t_start
                    eta = (elapsed / done_units) * (total_units - done_units)
                    log(
                        f"  [{done_units}/{total_units}] {combo_name} {route['name']} {run_key}: "
                        f"clean_no={agg['clean_non_object_rate']:.2f}, "
                        f"label={agg['any_label_violation_rate']:.2f}, "
                        f"echo={agg['object_echo_rate']:.2f}, "
                        f"fmt={agg.get('format_break_rate', 0.0):.2f}, "
                        f"len={agg.get('avg_generated_char_len', 0.0):.1f}, "
                        f"score={agg['clean_non_object_score']:.2f} "
                        f"({time.time()-t_cond:.1f}s, ETA {eta/60:.1f}min)"
                    )
                    checkpoint_path.write_text(json.dumps({
                        "phase": 561,
                        "model": args.model,
                        "surgery_mode": args.surgery_mode,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "pair": pair,
                        "window": window,
                        "combos": combos,
                        "conditions": conditions,
                        "blend_betas": blend_betas,
                        "run_specs": run_specs,
                        "routes": routes,
                        "done_units": done_units,
                        "total_units": total_units,
                        "audit": audit,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")

                rows = audit[combo_name]["rows"][route["name"]]
                base = rows["baseline"]
                remove_ref = rows.get("resid_remove_perp", base)
                spec_by_key = {spec["key"]: spec for spec in run_specs}
                for run_key, row in rows.items():
                    if run_key == "baseline":
                        continue
                    spec = spec_by_key[run_key]
                    compact.append(compact_row(
                        combo_name, layer_ids, route, run_key, spec["condition"],
                        None if "_donor_" not in spec["condition"] else float(spec["restore_beta"]),
                        row, base, remove_ref,
                    ))
                best = max((v["clean_non_object_rate"] for k, v in rows.items() if k != "baseline"), default=base["clean_non_object_rate"])
                log(
                    f"  SUMMARY {combo_name} {route['name']}: "
                    f"base={base['clean_non_object_rate']:.2f}; "
                    f"rm={rows.get('resid_remove_perp', base)['clean_non_object_rate']:.2f}; "
                    f"best={best:.2f}"
                )
        return {
            "phase": 561,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pair": pair,
            "window": window,
            "combos": combos,
            "conditions": conditions,
            "blend_betas": blend_betas,
            "run_specs": run_specs,
            "routes": routes,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "sample_seeds": sample_seeds,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "remove_scale": args.remove_scale,
            "add_alpha": args.add_alpha,
            "surgery_mode": args.surgery_mode,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "object_audit": object_audit,
            "audit": audit,
            "compact_rows": compact,
            "sample_records": saved_samples[: args.max_saved_samples],
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--windows", default=None)
    parser.add_argument("--pair", default="vehicle_tool")
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=12)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127")
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--layer-sets", default="all")
    parser.add_argument("--surgery-mode", choices=["one_shot", "continuous_blend"], default="continuous_blend")
    parser.add_argument("--blend-betas", default="0.05,0.10,0.25,0.50")
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--add-alpha", type=float, default=6.0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--samples-per-row", type=int, default=2)
    parser.add_argument("--max-saved-samples", type=int, default=1200)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase561_{args.model}_{args.surgery_mode}_gated_injection.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
