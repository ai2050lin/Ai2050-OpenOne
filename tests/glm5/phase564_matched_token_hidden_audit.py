#!/usr/bin/env python3
"""
Phase 564: Matched-Token Hidden Trajectory Audit.

Phase563 showed long hidden trajectory distances after one-shot intervention,
but later hidden distance can mix two effects:
  1. persistent internal perturbation;
  2. generated token divergence.

This phase compares three distances:
  - free: baseline free generation vs intervention free generation.
  - matched_base: both runs are forced through baseline generated tokens.
  - matched_condition: both runs are forced through intervention generated tokens.
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
import phase559_prototype_generation_closure as p559  # noqa: E402
import phase562_trajectory_response_audit as p562  # noqa: E402
import phase563_hidden_trajectory_distance as p563  # noqa: E402


OUT_ROOT = Path("results/glm5_phase564_matched_token_hidden_audit")
DEFAULT_ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
DEFAULT_CONDITIONS = [
    "baseline",
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


def generate_prescribed_trajectory(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    layer_ids: list[int],
    prompts: list[str],
    prescribed_ids: list[list[int]],
    condition: str,
    donor_cache: dict[int, torch.Tensor] | None,
    tangent_dirs: dict[int, torch.Tensor] | None,
    normal_dirs: dict[int, torch.Tensor] | None,
    components_by_layer: dict[str, dict[str, np.ndarray]],
    remove_scale: float,
    add_alpha: float,
    max_length: int,
) -> dict[str, Any]:
    """Run the model through a fixed generated-token sequence.

    Surgery is applied only to the prompt-answer position at step 0, matching
    Phase563 one-shot semantics. Later steps are teacher-forced with
    prescribed_ids.
    """
    batch_size = len(prompts)
    max_steps = min(len(x) for x in prescribed_ids)

    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = old_padding

    handles = p563.install_step0_hooks(
        layers, layer_ids, components_by_layer, condition, batch_size, answer_pos,
        donor_cache, tangent_dirs, normal_dirs, remove_scale, add_alpha
    )

    with p563.StepRecorder(layers, layer_ids) as recorder:
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
        for h in handles:
            h.remove()

        full_mask = attention_mask
        for step in range(1, max_steps):
            prev_ids = [ids[step - 1] for ids in prescribed_ids]
            new_ids = torch.tensor([[int(t)] for t in prev_ids], dtype=torch.long, device=device)
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

        h_steps = recorder.values

    suffixes = [tokenizer.decode(ids[:max_steps], skip_special_tokens=True) for ids in prescribed_ids]
    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"h_steps": h_steps, "suffixes": suffixes, "prescribed_ids": [x[:max_steps] for x in prescribed_ids]}


def token_divergence_stats(base_ids: list[list[int]], cond_ids: list[list[int]]) -> dict[str, Any]:
    first_steps: list[int] = []
    exact = 0
    for b, c in zip(base_ids, cond_ids):
        n = min(len(b), len(c))
        div = n
        for i in range(n):
            if int(b[i]) != int(c[i]):
                div = i
                break
        if n == len(b) == len(c) and all(int(x) == int(y) for x, y in zip(b, c)):
            exact += 1
        first_steps.append(int(div))
    return {
        "exact_sequence_match_rate": exact / max(1, len(base_ids)),
        "avg_first_divergence_step": float(np.mean(first_steps)) if first_steps else 0.0,
        "first_divergence_steps": first_steps,
    }


def average_hidden(items: list[dict[str, Any]]) -> dict[str, Any]:
    return p563.average_hidden_metrics(items)


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


def compact_row(
    combo: str,
    route: str,
    condition: str,
    row: dict[str, Any],
    base_clean: float,
) -> dict[str, Any]:
    free_h = row.get("hidden_free", {})
    mb_h = row.get("hidden_matched_base", {})
    mc_h = row.get("hidden_matched_condition", {})
    div = row.get("token_divergence", {})
    return {
        "combo": combo,
        "route": route,
        "condition": condition,
        "clean_non_object_rate": row.get("clean_non_object_rate", 0.0),
        "clean_delta": row.get("clean_non_object_rate", 0.0) - base_clean,
        "object_echo_rate": row.get("object_echo_rate", 0.0),
        "label_violation_rate": row.get("any_label_violation_rate", 0.0),
        "free_traj": free_h.get("avg_trajectory_distance", 0.0),
        "matched_base_traj": mb_h.get("avg_trajectory_distance", 0.0),
        "matched_condition_traj": mc_h.get("avg_trajectory_distance", 0.0),
        "free_relax": free_h.get("avg_hidden_relax_step", 0.0),
        "matched_base_relax": mb_h.get("avg_hidden_relax_step", 0.0),
        "matched_condition_relax": mc_h.get("avg_hidden_relax_step", 0.0),
        "free_growth": free_h.get("avg_finite_time_log_growth", 0.0),
        "matched_base_growth": mb_h.get("avg_finite_time_log_growth", 0.0),
        "matched_condition_growth": mc_h.get("avg_finite_time_log_growth", 0.0),
        "free_last_ratio": free_h.get("avg_delta_ratio_last", 0.0),
        "matched_base_last_ratio": mb_h.get("avg_delta_ratio_last", 0.0),
        "matched_condition_last_ratio": mc_h.get("avg_delta_ratio_last", 0.0),
        "exact_sequence_match_rate": div.get("exact_sequence_match_rate", 1.0 if condition == "baseline" else 0.0),
        "avg_first_divergence_step": div.get("avg_first_divergence_step", 0.0),
    }


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
            raise ValueError(f"Phase564 expects one window, got {windows}")
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

        log(f"{args.model}: phase564 pair={pair}, combos={combos}, routes={[r['name'] for r in routes]}")
        log(f"  conditions={conditions}, seeds={seeds}, max_tokens={args.max_new_tokens}")

        audit: dict[str, Any] = {}
        compact: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        total_units = len(combos) * len(routes) * len(seeds) * (1 + 3 * (len(conditions) - 1))
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

                per_records: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
                free_hidden: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
                matched_base_hidden: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
                matched_condition_hidden: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
                divergence: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}

                for seed in seeds:
                    baseline = p563.generate_trajectory(
                        model, tokenizer, device, layers, layer_ids, prompts, "baseline",
                        None, None, None, components_by_layer, groups, route["mode"], seed,
                        args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                        args.add_alpha, args.max_length,
                    )
                    done_units += 1

                    for i, row in enumerate(prompt_rows):
                        per_records["baseline"].append({
                            "prompt_index": i, "object": row["object"], "condition": "baseline", "seed": seed,
                            "generated_suffix": baseline["suffixes"][i],
                            "generated_ids": baseline["generated_ids"][i],
                        })
                    self_dist = p563.compare_hidden_trajectories(
                        baseline["h_steps"], baseline["h_steps"], args.epsilon_ratio
                    )
                    free_hidden["baseline"].append(self_dist)
                    matched_base_hidden["baseline"].append(self_dist)
                    matched_condition_hidden["baseline"].append(self_dist)
                    divergence["baseline"].append({
                        "exact_sequence_match_rate": 1.0,
                        "avg_first_divergence_step": float(args.max_new_tokens),
                        "first_divergence_steps": [args.max_new_tokens for _ in prompts],
                    })

                    tan_dirs = nor_dirs = None
                    if any(c in conditions for c in ["add_tangent_repeat2", "add_normal_repeat2"]):
                        tan_dirs, nor_dirs = p562.compute_tangent_normal(
                            donor_caches["repeat2"],
                            {lid: baseline["h_steps"][lid][0] for lid in layer_ids},
                            {lid: baseline["h_steps"][lid][1] for lid in layer_ids},
                            layer_ids,
                        )

                    for condition in conditions:
                        if condition == "baseline":
                            continue
                        variant = p563.condition_variant(condition)
                        donor_cache = donor_caches.get(variant) if variant else None
                        result = p563.generate_trajectory(
                            model, tokenizer, device, layers, layer_ids, prompts, condition,
                            donor_cache, tan_dirs, nor_dirs, components_by_layer, groups, route["mode"],
                            seed, args.max_new_tokens, args.temperature, args.top_p, args.remove_scale,
                            args.add_alpha, args.max_length,
                        )
                        done_units += 1
                        free_hidden[condition].append(p563.compare_hidden_trajectories(
                            baseline["h_steps"], result["h_steps"], args.epsilon_ratio
                        ))
                        divergence[condition].append(token_divergence_stats(
                            baseline["generated_ids"], result["generated_ids"]
                        ))

                        cond_base_tokens = generate_prescribed_trajectory(
                            model, tokenizer, device, layers, layer_ids, prompts, baseline["generated_ids"],
                            condition, donor_cache, tan_dirs, nor_dirs, components_by_layer,
                            args.remove_scale, args.add_alpha, args.max_length
                        )
                        done_units += 1
                        matched_base_hidden[condition].append(p563.compare_hidden_trajectories(
                            baseline["h_steps"], cond_base_tokens["h_steps"], args.epsilon_ratio
                        ))

                        base_cond_tokens = generate_prescribed_trajectory(
                            model, tokenizer, device, layers, layer_ids, prompts, result["generated_ids"],
                            "baseline", None, None, None, components_by_layer,
                            args.remove_scale, args.add_alpha, args.max_length
                        )
                        done_units += 1
                        matched_condition_hidden[condition].append(p563.compare_hidden_trajectories(
                            base_cond_tokens["h_steps"], result["h_steps"], args.epsilon_ratio
                        ))

                        for i, row in enumerate(prompt_rows):
                            per_records[condition].append({
                                "prompt_index": i, "object": row["object"], "condition": condition, "seed": seed,
                                "generated_suffix": result["suffixes"][i],
                                "generated_ids": result["generated_ids"][i],
                            })
                        elapsed = time.time() - t_start
                        eta = elapsed / max(1, done_units) * (total_units - done_units)
                        log(f"  [{done_units}/{total_units}] {combo_name} {route_name[:24]} seed={seed} "
                            f"{condition} ({eta/60:.1f}min ETA)")

                for condition in conditions:
                    agg = aggregate_records(per_records[condition], pos_label, neg_label)
                    agg["hidden_free"] = average_hidden(free_hidden[condition])
                    agg["hidden_matched_base"] = average_hidden(matched_base_hidden[condition])
                    agg["hidden_matched_condition"] = average_hidden(matched_condition_hidden[condition])
                    if divergence[condition]:
                        agg["token_divergence"] = {
                            "exact_sequence_match_rate": float(np.mean([
                                x["exact_sequence_match_rate"] for x in divergence[condition]
                            ])),
                            "avg_first_divergence_step": float(np.mean([
                                x["avg_first_divergence_step"] for x in divergence[condition]
                            ])),
                        }
                    audit[combo_name]["rows"][route_name][condition] = agg
                    samples.extend(agg["sample_records"][:args.samples_per_row])

                base = audit[combo_name]["rows"][route_name]["baseline"]
                base_clean = base["clean_non_object_rate"]
                for condition, row in audit[combo_name]["rows"][route_name].items():
                    compact.append(compact_row(combo_name, route_name, condition, row, base_clean))
                best = max(r["clean_non_object_rate"] for r in audit[combo_name]["rows"][route_name].values())
                log(f"  SUMMARY {combo_name} {route_name}: base={base_clean:.2f}; best={best:.2f}")

        return {
            "phase": 564,
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
    out_path = out_dir / f"phase564_{args.model}_matched_token_hidden_audit.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
