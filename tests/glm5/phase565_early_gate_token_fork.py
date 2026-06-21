#!/usr/bin/env python3
"""
Phase 565: Early Gate and Token Fork Causality Audit.

Phase564 showed that long free-generation hidden distances mostly come from
token-path divergence. This phase tests whether the first generated token is a
causal carrier for the semantic recovery.

For each baseline/intervention pair:
  - free baseline;
  - free intervention;
  - baseline with intervention first token forced, then free generation;
  - intervention with baseline first token forced, then free generation.

It also records step0/step1/step2 logits margins for target vs competitor.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import sys
import time
from collections import Counter, defaultdict
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
import phase559_prototype_generation_closure as p559  # noqa: E402
import phase562_trajectory_response_audit as p562  # noqa: E402
import phase563_hidden_trajectory_distance as p563  # noqa: E402


OUT_ROOT = Path("results/glm5_phase565_early_gate_token_fork")
DEFAULT_ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
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


def group_max(row: np.ndarray, ids: list[int]) -> float:
    valid = [i for i in ids if 0 <= i < row.shape[0]]
    if not valid:
        return float("-inf")
    return float(np.max(row[valid]))


def logits_summary(row: np.ndarray, groups: dict[str, list[int]], selected: int) -> dict[str, Any]:
    target = group_max(row, groups["target"])
    competitor = group_max(row, groups["competitor"])
    cluster_other = group_max(row, groups.get("cluster_other", []))
    off_cluster = group_max(row, groups.get("off_cluster", []))
    non_target = max(competitor, cluster_other, off_cluster)
    return {
        "selected_token": int(selected),
        "selected_type": p544.token_type(int(selected), groups),
        "target_best_logit": target,
        "competitor_best_logit": competitor,
        "cluster_other_best_logit": cluster_other,
        "off_cluster_best_logit": off_cluster,
        "target_minus_competitor": float(target - competitor),
        "target_minus_best_non_target": float(target - non_target),
        "target_rank": float(p544.best_rank(row, groups["target"])),
        "competitor_rank": float(p544.best_rank(row, groups["competitor"])),
    }


def install_condition_hooks(
    layers: list[Any],
    layer_ids: list[int],
    components_by_layer: dict[str, dict[str, np.ndarray]],
    condition: str,
    batch_size: int,
    answer_pos: int,
    donor_cache: dict[int, torch.Tensor] | None,
    normal_dirs: dict[int, torch.Tensor] | None,
    remove_scale: float,
    add_alpha: float,
) -> list[Any]:
    plan = p562.condition_plan_562(condition)
    if plan["site"] == "none":
        return []

    handles: list[Any] = []
    pos_cpu = torch.full((batch_size,), answer_pos, dtype=torch.long)
    for lid in layer_ids:
        layer = layers[lid]
        site = p559.module_for_site(layer, plan["site"])
        direction_np = components_by_layer[str(lid)][plan["component"]]
        direction_cpu = torch.tensor(p559.normalize_vec(direction_np), dtype=torch.float32)
        cached = None if donor_cache is None else donor_cache.get(lid)
        nor = None if normal_dirs is None else normal_dirs.get(lid)

        def make_hook(d_vec_cpu, pos_vec_cpu, cached_vec, nor_vec, cond_type, do_remove, do_restore):
            def hook(_module, _inp, output):
                hs = p559.tensor_from_output(output)
                out = hs
                pos_dev = pos_vec_cpu.to(out.device)
                if do_remove:
                    out = p559.project_remove(out, pos_dev, d_vec_cpu, remove_scale)
                if do_restore and cached_vec is not None:
                    bidx = torch.arange(out.shape[0], device=out.device)
                    out = out.clone()
                    out[bidx, pos_dev, :] = cached_vec.to(out.device, dtype=out.dtype)
                if cond_type == "add_normal" and nor_vec is not None:
                    out = p562.add_direction_batch(out, pos_dev, nor_vec, add_alpha)
                return p559.replace_output(output, out)
            return hook

        handles.append(site.register_forward_hook(make_hook(
            direction_cpu, pos_cpu, cached, nor,
            plan["type"], bool(plan["remove"]), bool(plan["restore"] is not None and cached is not None),
        )))
    return handles


def generate_with_optional_forced_first(
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
    forced_first_ids: list[int] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    batch_size = len(prompts)

    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = old_padding

    handles = install_condition_hooks(
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

    if forced_first_ids is None:
        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits0]
    else:
        toks = [int(x) for x in forced_first_ids]
    for i, tok in enumerate(toks):
        generated[i].append(int(tok))
        step_summaries[i].append(logits_summary(logits0[i], groups, int(tok)))

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
        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        for i, tok in enumerate(toks):
            generated[i].append(int(tok))
            if step < 3:
                step_summaries[i].append(logits_summary(logits[i], groups, int(tok)))

    suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "generated_ids": generated,
        "suffixes": suffixes,
        "step_summaries": step_summaries,
    }


def first_divergence(base_ids: list[int], cond_ids: list[int]) -> int:
    n = min(len(base_ids), len(cond_ids))
    for i in range(n):
        if int(base_ids[i]) != int(cond_ids[i]):
            return i
    return n


def fork_bucket(step: int) -> str:
    if step <= 1:
        return "early_0_1"
    if step <= 5:
        return "middle_2_5"
    return "late_or_none_6p"


def aggregate_records(records: list[dict[str, Any]], pos_label: str, neg_label: str) -> dict[str, Any]:
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
    agg["sample_records"] = classified[:10]
    if classified:
        for step in range(3):
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


def bucket_clean(records: list[dict[str, Any]], pos_label: str, neg_label: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        buckets[rec.get("fork_bucket", "unknown")].append(rec)
    out = {}
    for name, rows in buckets.items():
        out[name] = aggregate_records(rows, pos_label, neg_label)
    return out


def transfer_ratio(source: float, target: float, forced: float) -> float:
    denom = target - source
    if abs(denom) < 1e-8:
        return 0.0
    return float((forced - source) / denom)


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    routes = p558.parse_routes(args.routes)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    interventions = parse_csv(args.interventions)
    seeds = parse_int_csv(args.sample_seeds)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        if len(windows) != 1:
            raise ValueError(f"Phase565 expects one window, got {windows}")
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

        log(f"{args.model}: phase565 pair={pair}, combos={combos}, routes={[r['name'] for r in routes]}")
        log(f"  interventions={interventions}, seeds={seeds}, max_tokens={args.max_new_tokens}")

        audit: dict[str, Any] = {}
        compact: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        total_units = len(combos) * len(routes) * len(seeds) * (1 + 3 * len(interventions))
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

                condition_records: dict[str, list[dict[str, Any]]] = {"baseline_free": []}
                forced_pairs: dict[str, dict[str, list[dict[str, Any]]]] = {}
                fork_records: dict[str, list[dict[str, Any]]] = {c: [] for c in interventions}
                for cond in interventions:
                    condition_records[f"{cond}_free"] = []
                    condition_records[f"baseline_force_{cond}_first"] = []
                    condition_records[f"{cond}_force_baseline_first"] = []
                    forced_pairs[cond] = {"b_to_i": [], "i_to_b": []}

                for seed in seeds:
                    baseline = generate_with_optional_forced_first(
                        model, tokenizer, device, layers, layer_ids, prompts, "baseline",
                        None, None, components_by_layer, groups, route["mode"], seed,
                        args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                        args.add_alpha, args.max_length,
                    )
                    done_units += 1
                    for i, row in enumerate(prompt_rows):
                        rec = {
                            "prompt_index": i, "object": row["object"], "condition": "baseline_free", "seed": seed,
                            "generated_suffix": baseline["suffixes"][i],
                            "generated_ids": baseline["generated_ids"][i],
                            "step_summaries": baseline["step_summaries"][i],
                            "forced_first": False,
                        }
                        condition_records["baseline_free"].append(rec)

                    # Baseline hidden steps for normal decomposition.
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
                        result = generate_with_optional_forced_first(
                            model, tokenizer, device, layers, layer_ids, prompts, intervention,
                            donor_cache, normal_dirs, components_by_layer, groups, route["mode"], seed,
                            args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                            args.add_alpha, args.max_length,
                        )
                        done_units += 1

                        base_first = [ids[0] for ids in baseline["generated_ids"]]
                        int_first = [ids[0] for ids in result["generated_ids"]]
                        base_force_int = generate_with_optional_forced_first(
                            model, tokenizer, device, layers, layer_ids, prompts, "baseline",
                            None, None, components_by_layer, groups, route["mode"], seed,
                            args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                            args.add_alpha, args.max_length, forced_first_ids=int_first,
                        )
                        done_units += 1
                        int_force_base = generate_with_optional_forced_first(
                            model, tokenizer, device, layers, layer_ids, prompts, intervention,
                            donor_cache, normal_dirs, components_by_layer, groups, route["mode"], seed,
                            args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                            args.add_alpha, args.max_length, forced_first_ids=base_first,
                        )
                        done_units += 1

                        for i, row in enumerate(prompt_rows):
                            div = first_divergence(baseline["generated_ids"][i], result["generated_ids"][i])
                            bucket = fork_bucket(div)
                            common = {
                                "prompt_index": i, "object": row["object"], "seed": seed,
                                "first_divergence_step": div,
                                "fork_bucket": bucket,
                            }
                            rec_int = {
                                **common, "condition": f"{intervention}_free",
                                "generated_suffix": result["suffixes"][i],
                                "generated_ids": result["generated_ids"][i],
                                "step_summaries": result["step_summaries"][i],
                                "forced_first": False,
                            }
                            rec_bfi = {
                                **common, "condition": f"baseline_force_{intervention}_first",
                                "generated_suffix": base_force_int["suffixes"][i],
                                "generated_ids": base_force_int["generated_ids"][i],
                                "step_summaries": base_force_int["step_summaries"][i],
                                "forced_first": True,
                                "forced_token_source": "intervention",
                            }
                            rec_ifb = {
                                **common, "condition": f"{intervention}_force_baseline_first",
                                "generated_suffix": int_force_base["suffixes"][i],
                                "generated_ids": int_force_base["generated_ids"][i],
                                "step_summaries": int_force_base["step_summaries"][i],
                                "forced_first": True,
                                "forced_token_source": "baseline",
                            }
                            condition_records[f"{intervention}_free"].append(rec_int)
                            condition_records[f"baseline_force_{intervention}_first"].append(rec_bfi)
                            condition_records[f"{intervention}_force_baseline_first"].append(rec_ifb)
                            fork_records[intervention].append(rec_int)
                            forced_pairs[intervention]["b_to_i"].append(rec_bfi)
                            forced_pairs[intervention]["i_to_b"].append(rec_ifb)

                        elapsed = time.time() - t_start
                        eta = elapsed / max(1, done_units) * (total_units - done_units)
                        log(f"  [{done_units}/{total_units}] {combo_name} {route_name[:24]} seed={seed} "
                            f"{intervention} ({eta/60:.1f}min ETA)")

                base_agg = aggregate_records(condition_records["baseline_free"], pos_label, neg_label)
                audit[combo_name]["rows"][route_name]["baseline_free"] = base_agg
                samples.extend(base_agg["sample_records"][:args.samples_per_row])
                for intervention in interventions:
                    free_name = f"{intervention}_free"
                    bfi_name = f"baseline_force_{intervention}_first"
                    ifb_name = f"{intervention}_force_baseline_first"
                    free_agg = aggregate_records(condition_records[free_name], pos_label, neg_label)
                    bfi_agg = aggregate_records(condition_records[bfi_name], pos_label, neg_label)
                    ifb_agg = aggregate_records(condition_records[ifb_name], pos_label, neg_label)
                    free_agg["fork_bucket_metrics"] = bucket_clean(fork_records[intervention], pos_label, neg_label)
                    free_agg["avg_first_divergence_step"] = float(np.mean([
                        r["first_divergence_step"] for r in condition_records[free_name]
                    ]))
                    counts = Counter(r["fork_bucket"] for r in condition_records[free_name])
                    free_agg["fork_bucket_distribution"] = {
                        k: v / max(1, len(condition_records[free_name])) for k, v in sorted(counts.items())
                    }
                    base_clean = base_agg["clean_non_object_rate"]
                    int_clean = free_agg["clean_non_object_rate"]
                    bfi_clean = bfi_agg["clean_non_object_rate"]
                    ifb_clean = ifb_agg["clean_non_object_rate"]
                    free_agg["forced_first_transfer"] = {
                        "baseline_to_intervention_first_clean": bfi_clean,
                        "intervention_to_baseline_first_clean": ifb_clean,
                        "baseline_to_intervention_transfer_ratio": transfer_ratio(base_clean, int_clean, bfi_clean),
                        "intervention_to_baseline_transfer_ratio": transfer_ratio(int_clean, base_clean, ifb_clean),
                    }
                    audit[combo_name]["rows"][route_name][free_name] = free_agg
                    audit[combo_name]["rows"][route_name][bfi_name] = bfi_agg
                    audit[combo_name]["rows"][route_name][ifb_name] = ifb_agg
                    for name, agg in [(free_name, free_agg), (bfi_name, bfi_agg), (ifb_name, ifb_agg)]:
                        compact.append({
                            "combo": combo_name,
                            "route": route_name,
                            "condition": name,
                            "clean_non_object_rate": agg["clean_non_object_rate"],
                            "clean_delta_vs_baseline": agg["clean_non_object_rate"] - base_clean,
                            "object_echo_rate": agg["object_echo_rate"],
                            "label_violation_rate": agg["any_label_violation_rate"],
                            "step0_margin": agg.get("step0_avg_target_minus_competitor", 0.0),
                            "step1_margin": agg.get("step1_avg_target_minus_competitor", 0.0),
                            "step2_margin": agg.get("step2_avg_target_minus_competitor", 0.0),
                            "avg_first_divergence_step": free_agg.get("avg_first_divergence_step", 0.0),
                        })
                        samples.extend(agg["sample_records"][:args.samples_per_row])
                best = max(r["clean_non_object_rate"] for r in audit[combo_name]["rows"][route_name].values())
                log(f"  SUMMARY {combo_name} {route_name}: base={base_agg['clean_non_object_rate']:.2f}; best={best:.2f}")

        return {
            "phase": 565,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pair": pair,
            "window": window,
            "combos": combos,
            "interventions": interventions,
            "routes": routes,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "sample_seeds": seeds,
            "max_new_tokens": args.max_new_tokens,
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
    parser.add_argument("--layer-sets", default="all")
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--add-alpha", type=float, default=6.0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
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
    out_path = out_dir / f"phase565_{args.model}_early_gate_token_fork.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
