#!/usr/bin/env python3
"""
Phase 563: Hidden Trajectory Distance and Finite-Time Response Audit.

Phase562 measured token-level relaxation and tangent/normal response. This
phase measures hidden-state trajectory distance directly:
  - For each seed, run baseline first and record hidden states at each generation step.
  - Run intervention conditions with the same seed.
  - Compare per-layer hidden deltas against the baseline trajectory.
  - Report hidden relaxation step, trajectory distance, and finite-time growth.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
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
import phase559_prototype_generation_closure as p559  # noqa: E402
import phase562_trajectory_response_audit as p562  # noqa: E402


OUT_ROOT = Path("results/glm5_phase563_hidden_trajectory_distance")
DEFAULT_ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
DEFAULT_CONDITIONS = [
    "baseline",
    "one_shot_repeat2",
    "one_shot_repeat4",
    "one_shot_mean",
    "one_shot_random",
    "add_tangent_repeat2",
    "add_normal_repeat2",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


class StepRecorder:
    def __init__(self, layers: list[Any], layer_ids: list[int]):
        self.layers = layers
        self.layer_ids = layer_ids
        self.step = 0
        self.values: dict[int, list[torch.Tensor]] = {lid: [] for lid in layer_ids}
        self.handles: list[Any] = []

    def __enter__(self):
        for lid in self.layer_ids:
            self.handles.append(self.layers[lid].register_forward_hook(self._make_hook(lid)))
        return self

    def __exit__(self, *_exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, lid: int):
        def hook(_module, _inp, output):
            hs = p559.tensor_from_output(output)
            last = hs.shape[1] - 1
            self.values[lid].append(hs[:, last, :].detach().float().cpu())
        return hook


def condition_variant(condition: str) -> str | None:
    if condition.startswith("one_shot_"):
        return condition[len("one_shot_"):]
    if condition in {"add_tangent_repeat2", "add_normal_repeat2"}:
        return "repeat2"
    return None


def p558_condition_for_variant(variant: str) -> str:
    variant_map = {"mean": "mean_cache", "random": "random_cache"}
    return f"resid_donor_vehicle_{variant_map.get(variant, variant)}_add"


def collect_donor_caches(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    pair: str,
    donor_scaffold: str,
    test_n: int,
    max_length: int,
    add_alpha: float,
) -> dict[str, dict[int, torch.Tensor]]:
    caches: dict[str, dict[int, torch.Tensor]] = {}
    for variant in ["repeat2", "repeat4", "mean", "random"]:
        cond = p558_condition_for_variant(variant)
        plan = p558.condition_plan(cond)
        donor_rows = p558.donor_rows_for(
            pair, donor_scaffold, plan.get("donor_category"), plan.get("donor_variant"), test_n
        )
        prompts = [r["prompt"] for r in donor_rows]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in enc.items()}
        pos = batch["attention_mask"].sum(dim=1) - 1
        raw = p559.collect_donor_cache(
            model, layers, batch, pos, components_by_layer, layer_ids, "resid", "add_perp", add_alpha
        )
        caches[variant] = p559.transform_restore_cache(raw, plan.get("donor_variant"), 0)
    return caches


def install_step0_hooks(
    layers: list[Any],
    layer_ids: list[int],
    components_by_layer: dict[str, dict[str, np.ndarray]],
    condition: str,
    batch_size: int,
    answer_pos: int,
    donor_cache: dict[int, torch.Tensor] | None,
    tangent_dirs: dict[int, torch.Tensor] | None,
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
        tan = None if tangent_dirs is None else tangent_dirs.get(lid)
        nor = None if normal_dirs is None else normal_dirs.get(lid)

        def make_hook(d_vec_cpu, pos_vec_cpu, cached_vec, tan_vec, nor_vec, cond_type, do_remove, do_restore):
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
                if cond_type == "add_tangent" and tan_vec is not None:
                    out = p562.add_direction_batch(out, pos_dev, tan_vec, add_alpha)
                if cond_type == "add_normal" and nor_vec is not None:
                    out = p562.add_direction_batch(out, pos_dev, nor_vec, add_alpha)
                return p559.replace_output(output, out)
            return hook

        handles.append(site.register_forward_hook(make_hook(
            direction_cpu, pos_cpu, cached, tan, nor,
            plan["type"], bool(plan["remove"]), bool(plan["restore"] is not None and cached is not None),
        )))
    return handles


def generate_trajectory(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    layer_ids: list[int],
    prompts: list[str],
    condition: str,
    donor_cache: dict[int, torch.Tensor] | None,
    tangent_dirs: dict[int, torch.Tensor] | None,
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

    handles = install_step0_hooks(
        layers, layer_ids, components_by_layer, condition, batch_size, answer_pos,
        donor_cache, tangent_dirs, normal_dirs, remove_scale, add_alpha
    )
    with StepRecorder(layers, layer_ids) as recorder:
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits0 = out.logits[:, answer_pos, :].float().cpu().numpy()
        for h in handles:
            h.remove()

        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits0]
        generated: list[list[int]] = [[int(t)] for t in toks]
        per_step_types = [[p544.token_type(int(t), groups) for t in toks]]
        first_target_ranks = [float(p544.best_rank(row, groups["target"])) for row in logits0]
        first_comp_ranks = [float(p544.best_rank(row, groups["competitor"])) for row in logits0]

        full_mask = attention_mask
        for _step in range(1, max_new_tokens):
            new_ids = torch.tensor([[t] for t in toks], dtype=torch.long, device=device)
            full_mask = torch.cat(
                [full_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)], dim=1
            )
            with torch.inference_mode():
                out = model(input_ids=new_ids, attention_mask=full_mask, past_key_values=past_kv,
                            use_cache=True, return_dict=True)
                past_kv = out.past_key_values
                logits = out.logits[:, -1, :].float().cpu().numpy()
            toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
            per_step_types.append([p544.token_type(int(t), groups) for t in toks])
            for i, t in enumerate(toks):
                generated[i].append(int(t))

        h_steps = recorder.values

    suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "generated_ids": generated,
        "suffixes": suffixes,
        "per_step_types": per_step_types,
        "first_target_ranks": first_target_ranks,
        "first_competitor_ranks": first_comp_ranks,
        "h_steps": h_steps,
    }


def compare_hidden_trajectories(
    base_steps: dict[int, list[torch.Tensor]],
    cond_steps: dict[int, list[torch.Tensor]],
    epsilon_ratio: float,
) -> dict[str, Any]:
    layer_metrics = {}
    all_delta_by_step: list[list[float]] = []
    eps = 1e-8
    for lid, base_list in base_steps.items():
        cond_list = cond_steps[lid]
        n_steps = min(len(base_list), len(cond_list))
        delta_means: list[float] = []
        base_norms: list[float] = []
        for s in range(n_steps):
            delta = (cond_list[s].float() - base_list[s].float()).norm(dim=-1)
            base_norm = base_list[s].float().norm(dim=-1)
            delta_means.append(float(delta.mean().item()))
            base_norms.append(float(base_norm.mean().item()))
        delta0 = delta_means[0] if delta_means else 0.0
        ratios = [float(d / (delta0 + eps)) for d in delta_means]
        relax_step = n_steps
        if delta0 < 1e-6:
            relax_step = 0
        else:
            for idx, ratio in enumerate(ratios[1:], start=1):
                if ratio <= epsilon_ratio:
                    relax_step = idx
                    break
        lyap = 0.0
        if n_steps > 1 and delta0 > 1e-6:
            lyap = math.log((delta_means[-1] + eps) / (delta0 + eps)) / float(n_steps - 1)
        traj_distance = float(sum(delta_means))
        rel_traj_distance = float(sum(d / (bn + eps) for d, bn in zip(delta_means, base_norms)))
        layer_metrics[str(lid)] = {
            "delta_means": delta_means,
            "delta_ratios": ratios,
            "delta0": float(delta0),
            "delta_last": float(delta_means[-1] if delta_means else 0.0),
            "hidden_relax_step": int(relax_step),
            "finite_time_log_growth": float(lyap),
            "trajectory_distance": traj_distance,
            "relative_trajectory_distance": rel_traj_distance,
        }
        all_delta_by_step.append(delta_means)

    if all_delta_by_step:
        min_steps = min(len(x) for x in all_delta_by_step)
        avg_delta = [float(np.mean([x[s] for x in all_delta_by_step])) for s in range(min_steps)]
    else:
        avg_delta = []
    delta0 = avg_delta[0] if avg_delta else 0.0
    avg_ratios = [float(x / (delta0 + eps)) for x in avg_delta]
    avg_relax = 0 if delta0 < 1e-6 else len(avg_delta)
    if delta0 >= 1e-6:
        for idx, ratio in enumerate(avg_ratios[1:], start=1):
            if ratio <= epsilon_ratio:
                avg_relax = idx
                break
    avg_growth = 0.0
    if len(avg_delta) > 1 and delta0 > 1e-6:
        avg_growth = math.log((avg_delta[-1] + eps) / (delta0 + eps)) / float(len(avg_delta) - 1)
    return {
        "layers": layer_metrics,
        "avg_delta_by_step": avg_delta,
        "avg_delta_ratio_by_step": avg_ratios,
        "avg_hidden_relax_step": int(avg_relax),
        "avg_finite_time_log_growth": float(avg_growth),
        "avg_trajectory_distance": float(sum(avg_delta)),
    }


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
    return agg


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    routes = p558.parse_routes(args.routes)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    conditions = parse_csv(args.conditions)
    seeds = parse_int_csv(args.sample_seeds)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        if len(windows) != 1:
            raise ValueError(f"Phase563 expects one window, got {windows}")
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
        object_audit = [{"repeat_index": i, "object": o,
                         "token_length": len(tokenizer.encode(o, add_special_tokens=False))}
                        for i, o in enumerate(objects)]

        log(f"{args.model}: phase563 pair={pair}, combos={combos}, routes={[r['name'] for r in routes]}")
        log(f"  conditions={conditions}, seeds={seeds}, max_tokens={args.max_new_tokens}")

        audit: dict[str, Any] = {}
        compact: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total_units = len(combos) * len(routes) * len(conditions) * len(seeds)
        done_units = 0
        t_start = time.time()

        for combo_name, layer_ids in combos.items():
            audit[combo_name] = {"layers": layer_ids, "rows": {}}
            for route in routes:
                route_name = route["name"]
                audit[combo_name]["rows"][route_name] = {}
                prompt_rows = prompt_sets[route["recipient_scaffold"]]
                prompts = [r["prompt"] for r in prompt_rows]
                donor_caches = collect_donor_caches(
                    model, tokenizer, device, layers, components_by_layer, layer_ids, pair,
                    route["donor_scaffold"], args.test_n, args.max_length, args.add_alpha
                )

                per_condition_records: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
                per_condition_hidden: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}

                for seed in seeds:
                    baseline = generate_trajectory(
                        model, tokenizer, device, layers, layer_ids, prompts, "baseline",
                        None, None, None, components_by_layer, groups, route["mode"], seed,
                        args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                        args.add_alpha, args.max_length,
                    )
                    # Add baseline records and hidden self-distance.
                    for i, row in enumerate(prompt_rows):
                        per_condition_records["baseline"].append({
                            "prompt_index": i, "object": row["object"], "condition": "baseline", "seed": seed,
                            "generated_suffix": baseline["suffixes"][i],
                            "generated_ids": baseline["generated_ids"][i],
                        })
                    per_condition_hidden["baseline"].append(compare_hidden_trajectories(
                        baseline["h_steps"], baseline["h_steps"], args.epsilon_ratio
                    ))
                    done_units += 1

                    tan_dirs = nor_dirs = None
                    if "add_tangent_repeat2" in conditions or "add_normal_repeat2" in conditions:
                        tan_dirs, nor_dirs = p562.compute_tangent_normal(
                            donor_caches["repeat2"],
                            {lid: baseline["h_steps"][lid][0] for lid in layer_ids},
                            {lid: baseline["h_steps"][lid][1] for lid in layer_ids},
                            layer_ids,
                        )

                    for condition in conditions:
                        if condition == "baseline":
                            continue
                        variant = condition_variant(condition)
                        donor_cache = donor_caches.get(variant) if variant else None
                        result = generate_trajectory(
                            model, tokenizer, device, layers, layer_ids, prompts, condition,
                            donor_cache, tan_dirs, nor_dirs, components_by_layer, groups, route["mode"],
                            seed, args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                            args.add_alpha, args.max_length,
                        )
                        for i, row in enumerate(prompt_rows):
                            per_condition_records[condition].append({
                                "prompt_index": i, "object": row["object"], "condition": condition, "seed": seed,
                                "generated_suffix": result["suffixes"][i],
                                "generated_ids": result["generated_ids"][i],
                            })
                        per_condition_hidden[condition].append(compare_hidden_trajectories(
                            baseline["h_steps"], result["h_steps"], args.epsilon_ratio
                        ))
                        done_units += 1
                        elapsed = time.time() - t_start
                        eta = elapsed / max(1, done_units) * (total_units - done_units)
                        log(f"  [{done_units}/{total_units}] {combo_name} {route_name[:24]} seed={seed} {condition} "
                            f"({eta/60:.1f}min ETA)")

                for condition in conditions:
                    agg = aggregate_records(per_condition_records[condition], pos_label, neg_label)
                    hidden_list = per_condition_hidden[condition]
                    agg["hidden_metrics"] = average_hidden_metrics(hidden_list)
                    audit[combo_name]["rows"][route_name][condition] = agg
                    samples.extend(agg["sample_records"][:args.samples_per_row])

                base = audit[combo_name]["rows"][route_name]["baseline"]
                for condition, row in audit[combo_name]["rows"][route_name].items():
                    hm = row["hidden_metrics"]
                    compact.append({
                        "combo": combo_name,
                        "route": route_name,
                        "condition": condition,
                        "clean_non_object_rate": row["clean_non_object_rate"],
                        "clean_delta": row["clean_non_object_rate"] - base["clean_non_object_rate"],
                        "object_echo_rate": row["object_echo_rate"],
                        "label_violation_rate": row["any_label_violation_rate"],
                        "avg_hidden_relax_step": hm["avg_hidden_relax_step"],
                        "avg_finite_time_log_growth": hm["avg_finite_time_log_growth"],
                        "avg_trajectory_distance": hm["avg_trajectory_distance"],
                        "avg_delta_ratio_last": hm["avg_delta_ratio_last"],
                    })
                best = max(r["clean_non_object_rate"] for r in audit[combo_name]["rows"][route_name].values())
                log(f"  SUMMARY {combo_name} {route_name}: base={base['clean_non_object_rate']:.2f}; best={best:.2f}")

        return {
            "phase": 563,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pair": pair,
            "window": window,
            "combos": combos,
            "conditions": conditions,
            "routes": routes,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "sample_seeds": seeds,
            "max_new_tokens": args.max_new_tokens,
            "epsilon_ratio": args.epsilon_ratio,
            "remove_scale": args.remove_scale,
            "add_alpha": args.add_alpha,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "object_audit": object_audit,
            "audit": audit,
            "compact_rows": compact,
            "sample_records": samples[:args.max_saved_samples],
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def average_hidden_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    max_steps = max(len(x["avg_delta_by_step"]) for x in items)
    avg_delta = []
    avg_ratio = []
    for s in range(max_steps):
        vals = [x["avg_delta_by_step"][s] for x in items if s < len(x["avg_delta_by_step"])]
        rats = [x["avg_delta_ratio_by_step"][s] for x in items if s < len(x["avg_delta_ratio_by_step"])]
        avg_delta.append(float(np.mean(vals)) if vals else 0.0)
        avg_ratio.append(float(np.mean(rats)) if rats else 0.0)
    return {
        "avg_delta_by_step": avg_delta,
        "avg_delta_ratio_by_step": avg_ratio,
        "avg_delta_ratio_last": float(avg_ratio[-1]) if avg_ratio else 0.0,
        "avg_hidden_relax_step": float(np.mean([x["avg_hidden_relax_step"] for x in items])),
        "avg_finite_time_log_growth": float(np.mean([x["avg_finite_time_log_growth"] for x in items])),
        "avg_trajectory_distance": float(np.mean([x["avg_trajectory_distance"] for x in items])),
    }


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
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--add-alpha", type=float, default=6.0)
    parser.add_argument("--epsilon-ratio", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--samples-per-row", type=int, default=2)
    parser.add_argument("--max-saved-samples", type=int, default=800)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase563_{args.model}_hidden_trajectory_distance.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
