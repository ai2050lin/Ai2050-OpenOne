#!/usr/bin/env python3
"""
Phase 566: Multi-Step Prefix Fork and Logit Field Audit.

Phase565 showed that the first token alone does not transfer most GLM4 repeat4
semantic recovery. This phase tests whether the early prefix length 1/2/3 is
the causal carrier, while recording step0-step5 target/competitor margins.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import load_model_bf16_flash  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
import phase548_paraphrase_candidate_robustness as p548  # noqa: E402
import phase558_prototype_object_binding_audit as p558  # noqa: E402
import phase562_trajectory_response_audit as p562  # noqa: E402
import phase563_hidden_trajectory_distance as p563  # noqa: E402
import phase565_early_gate_token_fork as p565  # noqa: E402


OUT_ROOT = Path("results/glm5_phase566_prefix_fork_logit_field")
DEFAULT_ROUTES = ["forbidden_sentence_completion:temperature<-forbidden_definition"]
DEFAULT_INTERVENTIONS = [
    "one_shot_repeat2",
    "one_shot_repeat4",
    "one_shot_random",
    "add_normal_repeat2",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def generate_with_forced_prefix(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    layer_ids: list[int],
    prompts: list[str],
    condition: str,
    donor_cache: dict[int, torch.Tensor] | None,
    normal_dirs: dict[int, torch.Tensor] | None,
    components_by_layer: dict[str, dict[str, np.ndarray]],
    groups: dict[str, list[int]],
    mode: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    remove_scale: float,
    add_alpha: float,
    max_length: int,
    forced_prefix_ids: list[list[int]] | None = None,
    logit_steps: int = 6,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    batch_size = len(prompts)
    forced_len = 0 if forced_prefix_ids is None else min(len(x) for x in forced_prefix_ids)

    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = old_padding

    handles = p565.install_condition_hooks(
        layers, layer_ids, components_by_layer, condition, batch_size, answer_pos,
        donor_cache, normal_dirs, remove_scale, add_alpha
    )

    generated: list[list[int]] = [[] for _ in prompts]
    step_summaries: list[list[dict[str, Any]]] = [[] for _ in prompts]
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits0 = out.logits[:, answer_pos, :].float().cpu().numpy()
    for h in handles:
        h.remove()

    if forced_prefix_ids is not None and forced_len >= 1:
        toks = [int(ids[0]) for ids in forced_prefix_ids]
    else:
        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits0]
    for i, tok in enumerate(toks):
        generated[i].append(int(tok))
        if logit_steps > 0:
            step_summaries[i].append(p565.logits_summary(logits0[i], groups, int(tok)))

    full_mask = attention_mask
    for step in range(1, max_new_tokens):
        new_ids = torch.tensor([[int(t)] for t in toks], dtype=torch.long, device=device)
        full_mask = torch.cat(
            [full_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)], dim=1
        )
        with torch.inference_mode():
            out = model(
                input_ids=new_ids,
                attention_mask=full_mask,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :].float().cpu().numpy()
        if forced_prefix_ids is not None and step < forced_len:
            toks = [int(ids[step]) for ids in forced_prefix_ids]
        else:
            toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        for i, tok in enumerate(toks):
            generated[i].append(int(tok))
            if step < logit_steps:
                step_summaries[i].append(p565.logits_summary(logits[i], groups, int(tok)))

    suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"generated_ids": generated, "suffixes": suffixes, "step_summaries": step_summaries}


def aggregate_records(records: list[dict[str, Any]], pos_label: str, neg_label: str, logit_steps: int) -> dict[str, Any]:
    classified = []
    degenerations = []
    for rec in records:
        cls = p548.classify_suffix(rec["generated_suffix"], rec["object"], pos_label, neg_label)
        deg = p562.classify_degeneration(rec["generated_suffix"])
        classified.append({**rec, **cls, "degeneration": deg})
        degenerations.append(deg)
    agg = p548.aggregate(classified)
    counts = Counter(degenerations)
    agg["degeneration_distribution"] = {k: v / max(1, len(degenerations)) for k, v in sorted(counts.items())}
    agg["sample_records"] = classified[:8]
    for step in range(logit_steps):
        vals = [r["step_summaries"][step] for r in classified if len(r.get("step_summaries", [])) > step]
        if vals:
            agg[f"step{step}_avg_target_minus_competitor"] = float(np.mean([
                v["target_minus_competitor"] for v in vals
            ]))
            agg[f"step{step}_avg_target_minus_best_non_target"] = float(np.mean([
                v["target_minus_best_non_target"] for v in vals
            ]))
            agg[f"step{step}_avg_target_rank"] = float(np.mean([v["target_rank"] for v in vals]))
            type_counts = Counter(v["selected_type"] for v in vals)
            agg[f"step{step}_selected_type_distribution"] = {
                k: v / max(1, len(vals)) for k, v in sorted(type_counts.items())
            }
    return agg


def prefix_rows(
    source_ids: list[list[int]],
    prefix_len: int,
) -> list[list[int]]:
    return [ids[:prefix_len] for ids in source_ids]


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    routes = p558.parse_routes(args.routes)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    interventions = parse_csv(args.interventions)
    prefix_lengths = parse_int_csv(args.prefix_lengths)
    seeds = parse_int_csv(args.sample_seeds)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        if len(windows) != 1:
            raise ValueError(f"Phase566 expects one window, got {windows}")
        _, window = next(iter(windows.items()))
        combos = p558.combo_layers(window, args.layer_sets)
        all_layers = sorted(set(itertools.chain.from_iterable(combos.values())))
        W_U = get_W_U(model, args.model).astype(np.float32)
        groups = p544.token_groups(tokenizer, pair)
        prompt_sets = p548.build_prompts(pair, args.test_n, scaffolds)
        components_by_layer = p558.build_components_by_layer(
            model, tokenizer, device, pair, all_layers, args.train_n, args.batch_size, args.max_length, W_U
        )
        pos_label, neg_label = PAIR_SPECS[pair]
        objects = CATEGORY_BANK[pos_label][-args.test_n:]

        log(f"{args.model}: phase566 pair={pair}, combos={combos}, routes={[r['name'] for r in routes]}")
        log(f"  interventions={interventions}, prefix_lengths={prefix_lengths}, seeds={seeds}")

        audit: dict[str, Any] = {}
        compact: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        total_units = len(combos) * len(routes) * len(seeds) * (1 + len(interventions) * (1 + 2 * len(prefix_lengths)))
        done_units = 0
        t_start = time.time()

        for combo_name, layer_ids in combos.items():
            audit[combo_name] = {"layers": layer_ids, "rows": {}}
            for route in routes:
                route_name = route["name"]
                audit[combo_name]["rows"][route_name] = {}
                prompt_rows = prompt_sets[route["recipient_scaffold"]]
                prompts = [r["prompt"] for r in prompt_rows]
                donor_caches = p563.collect_donor_caches(
                    model, tokenizer, device, layers, components_by_layer, layer_ids, pair,
                    route["donor_scaffold"], args.test_n, args.max_length, args.add_alpha
                )

                records: dict[str, list[dict[str, Any]]] = {"baseline_free": []}
                for intervention in interventions:
                    records[f"{intervention}_free"] = []
                    for plen in prefix_lengths:
                        records[f"baseline_force_{intervention}_prefix{plen}"] = []
                        records[f"{intervention}_force_baseline_prefix{plen}"] = []

                for seed in seeds:
                    baseline = generate_with_forced_prefix(
                        model, tokenizer, device, layers, layer_ids, prompts, "baseline",
                        None, None, components_by_layer, groups, route["mode"], seed,
                        args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                        args.add_alpha, args.max_length, None, args.logit_steps,
                    )
                    done_units += 1
                    for i, row in enumerate(prompt_rows):
                        records["baseline_free"].append({
                            "prompt_index": i, "object": row["object"], "seed": seed,
                            "condition": "baseline_free", "generated_suffix": baseline["suffixes"][i],
                            "generated_ids": baseline["generated_ids"][i],
                            "step_summaries": baseline["step_summaries"][i],
                        })

                    base_for_dirs = p563.generate_trajectory(
                        model, tokenizer, device, layers, layer_ids, prompts, "baseline",
                        None, None, None, components_by_layer, groups, route["mode"], seed,
                        min(2, args.max_new_tokens), args.temperature, args.top_p, args.remove_scale,
                        args.add_alpha, args.max_length,
                    )
                    _, normal_dirs = p562.compute_tangent_normal(
                        donor_caches["repeat2"],
                        {lid: base_for_dirs["h_steps"][lid][0] for lid in layer_ids},
                        {lid: base_for_dirs["h_steps"][lid][1] for lid in layer_ids},
                        layer_ids,
                    )

                    for intervention in interventions:
                        variant = p563.condition_variant(intervention)
                        donor_cache = donor_caches.get(variant) if variant else None
                        result = generate_with_forced_prefix(
                            model, tokenizer, device, layers, layer_ids, prompts, intervention,
                            donor_cache, normal_dirs, components_by_layer, groups, route["mode"], seed,
                            args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                            args.add_alpha, args.max_length, None, args.logit_steps,
                        )
                        done_units += 1
                        for i, row in enumerate(prompt_rows):
                            records[f"{intervention}_free"].append({
                                "prompt_index": i, "object": row["object"], "seed": seed,
                                "condition": f"{intervention}_free",
                                "generated_suffix": result["suffixes"][i],
                                "generated_ids": result["generated_ids"][i],
                                "step_summaries": result["step_summaries"][i],
                            })

                        for plen in prefix_lengths:
                            bfi = generate_with_forced_prefix(
                                model, tokenizer, device, layers, layer_ids, prompts, "baseline",
                                None, None, components_by_layer, groups, route["mode"], seed,
                                args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                                args.add_alpha, args.max_length, prefix_rows(result["generated_ids"], plen),
                                args.logit_steps,
                            )
                            done_units += 1
                            ifb = generate_with_forced_prefix(
                                model, tokenizer, device, layers, layer_ids, prompts, intervention,
                                donor_cache, normal_dirs, components_by_layer, groups, route["mode"], seed,
                                args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                                args.add_alpha, args.max_length, prefix_rows(baseline["generated_ids"], plen),
                                args.logit_steps,
                            )
                            done_units += 1
                            for i, row in enumerate(prompt_rows):
                                records[f"baseline_force_{intervention}_prefix{plen}"].append({
                                    "prompt_index": i, "object": row["object"], "seed": seed,
                                    "condition": f"baseline_force_{intervention}_prefix{plen}",
                                    "generated_suffix": bfi["suffixes"][i],
                                    "generated_ids": bfi["generated_ids"][i],
                                    "step_summaries": bfi["step_summaries"][i],
                                    "prefix_len": plen,
                                })
                                records[f"{intervention}_force_baseline_prefix{plen}"].append({
                                    "prompt_index": i, "object": row["object"], "seed": seed,
                                    "condition": f"{intervention}_force_baseline_prefix{plen}",
                                    "generated_suffix": ifb["suffixes"][i],
                                    "generated_ids": ifb["generated_ids"][i],
                                    "step_summaries": ifb["step_summaries"][i],
                                    "prefix_len": plen,
                                })

                        elapsed = time.time() - t_start
                        eta = elapsed / max(1, done_units) * (total_units - done_units)
                        log(f"  [{done_units}/{total_units}] {combo_name} {route_name[:24]} seed={seed} "
                            f"{intervention} ({eta/60:.1f}min ETA)")

                base_agg = aggregate_records(records["baseline_free"], pos_label, neg_label, args.logit_steps)
                audit[combo_name]["rows"][route_name]["baseline_free"] = base_agg
                base_clean = base_agg["clean_non_object_rate"]
                samples.extend(base_agg["sample_records"][:args.samples_per_row])
                for intervention in interventions:
                    free_name = f"{intervention}_free"
                    free_agg = aggregate_records(records[free_name], pos_label, neg_label, args.logit_steps)
                    free_clean = free_agg["clean_non_object_rate"]
                    free_agg["prefix_transfer"] = {}
                    audit[combo_name]["rows"][route_name][free_name] = free_agg
                    samples.extend(free_agg["sample_records"][:args.samples_per_row])
                    for plen in prefix_lengths:
                        bfi_name = f"baseline_force_{intervention}_prefix{plen}"
                        ifb_name = f"{intervention}_force_baseline_prefix{plen}"
                        bfi_agg = aggregate_records(records[bfi_name], pos_label, neg_label, args.logit_steps)
                        ifb_agg = aggregate_records(records[ifb_name], pos_label, neg_label, args.logit_steps)
                        free_agg["prefix_transfer"][str(plen)] = {
                            "baseline_to_intervention_prefix_clean": bfi_agg["clean_non_object_rate"],
                            "intervention_to_baseline_prefix_clean": ifb_agg["clean_non_object_rate"],
                            "baseline_to_intervention_transfer_ratio": p565.transfer_ratio(
                                base_clean, free_clean, bfi_agg["clean_non_object_rate"]
                            ),
                            "intervention_to_baseline_transfer_ratio": p565.transfer_ratio(
                                free_clean, base_clean, ifb_agg["clean_non_object_rate"]
                            ),
                        }
                        audit[combo_name]["rows"][route_name][bfi_name] = bfi_agg
                        audit[combo_name]["rows"][route_name][ifb_name] = ifb_agg
                        for name, agg in [(bfi_name, bfi_agg), (ifb_name, ifb_agg)]:
                            compact.append({
                                "combo": combo_name, "route": route_name, "condition": name,
                                "clean_non_object_rate": agg["clean_non_object_rate"],
                                "clean_delta_vs_baseline": agg["clean_non_object_rate"] - base_clean,
                                "step0_margin": agg.get("step0_avg_target_minus_competitor", 0.0),
                                "step1_margin": agg.get("step1_avg_target_minus_competitor", 0.0),
                                "step2_margin": agg.get("step2_avg_target_minus_competitor", 0.0),
                                "step3_margin": agg.get("step3_avg_target_minus_competitor", 0.0),
                                "step4_margin": agg.get("step4_avg_target_minus_competitor", 0.0),
                                "step5_margin": agg.get("step5_avg_target_minus_competitor", 0.0),
                            })
                            samples.extend(agg["sample_records"][:args.samples_per_row])
                    compact.append({
                        "combo": combo_name, "route": route_name, "condition": free_name,
                        "clean_non_object_rate": free_clean,
                        "clean_delta_vs_baseline": free_clean - base_clean,
                        "step0_margin": free_agg.get("step0_avg_target_minus_competitor", 0.0),
                        "step1_margin": free_agg.get("step1_avg_target_minus_competitor", 0.0),
                        "step2_margin": free_agg.get("step2_avg_target_minus_competitor", 0.0),
                        "step3_margin": free_agg.get("step3_avg_target_minus_competitor", 0.0),
                        "step4_margin": free_agg.get("step4_avg_target_minus_competitor", 0.0),
                        "step5_margin": free_agg.get("step5_avg_target_minus_competitor", 0.0),
                    })
                best = max(r["clean_non_object_rate"] for r in audit[combo_name]["rows"][route_name].values())
                log(f"  SUMMARY {combo_name} {route_name}: base={base_clean:.2f}; best={best:.2f}")

        return {
            "phase": 566,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pair": pair,
            "window": window,
            "combos": combos,
            "interventions": interventions,
            "prefix_lengths": prefix_lengths,
            "routes": routes,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "sample_seeds": seeds,
            "max_new_tokens": args.max_new_tokens,
            "logit_steps": args.logit_steps,
            "remove_scale": args.remove_scale,
            "add_alpha": args.add_alpha,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "object_audit": [{"repeat_index": i, "object": o,
                              "token_length": len(tokenizer.encode(o, add_special_tokens=False))}
                             for i, o in enumerate(objects)],
            "audit": audit,
            "compact_rows": compact,
            "sample_records": samples[:args.max_saved_samples],
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
    parser.add_argument("--interventions", default=",".join(DEFAULT_INTERVENTIONS))
    parser.add_argument("--prefix-lengths", default="1,2,3")
    parser.add_argument("--layer-sets", default="all")
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--add-alpha", type=float, default=6.0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--logit-steps", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--max-saved-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase566_{args.model}_prefix_fork_logit_field.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
