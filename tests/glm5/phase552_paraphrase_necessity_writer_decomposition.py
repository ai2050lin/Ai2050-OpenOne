#!/usr/bin/env python3
"""
Phase 552: necessity and writer decomposition for the clean paraphrase gate.

Phase551 showed that GLM4 vehicle_tool has strong clean paraphrase effects when
directions are added to L24/L28 or L24/L26/L28.  This phase asks a stricter
question: if the native component along that direction is removed from residual,
attention output, or MLP output, does clean paraphrase generation degrade?
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

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import hidden_at_layer, load_model_bf16_flash, mean_dir  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
import phase548_paraphrase_candidate_robustness as p548  # noqa: E402


OUT_ROOT = Path("results/glm5_phase552_paraphrase_necessity_writer_decomposition")
DEFAULT_PAIR = "vehicle_tool"
DEFAULT_SCAFFOLD_MODES = [
    "forbidden_sentence_completion:temperature",
    "forbidden_natural_qa:top_p",
    "forbidden_definition:top_p",
]
DEFAULT_CONDITIONS = [
    "baseline",
    "add_perp",
    "resid_remove_perp",
    "resid_remove_full",
    "resid_remove_random_perp",
    "resid_remove_perp_add_perp",
    "attn_remove_perp",
    "mlp_remove_perp",
    "attn_remove_full",
    "mlp_remove_full",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_scaffold_modes(text: str) -> list[tuple[str, str]]:
    pairs = []
    for item in parse_csv(text):
        if ":" not in item:
            raise ValueError(f"scaffold mode must be scaffold:mode, got {item}")
        a, b = item.split(":", 1)
        pairs.append((a.strip(), b.strip()))
    return pairs


def combo_layers(window: list[int], spec: str) -> dict[str, list[int]]:
    if spec:
        out = {}
        for item in parse_csv(spec):
            if item == "all":
                out["all"] = list(window)
            elif "+" in item:
                vals = [int(x.strip().lstrip("L")) for x in item.split("+")]
                out[item] = vals
            else:
                val = int(item.strip().lstrip("L"))
                out[f"L{val}"] = [val]
        return out
    first, mid, last = window[0], window[len(window) // 2], window[-1]
    return {
        f"L{first}": [first],
        f"L{last}": [last],
        f"L{first}+L{last}": [first, last],
        "all": [first, mid, last],
    }


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    arr = vec.astype(np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return arr
    return arr / norm


def build_components_by_layer(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    pair: str,
    layers_to_collect: list[int],
    train_n: int,
    batch_size: int,
    max_length: int,
    W_U: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    candidates = p548.build_candidates(pair, train_n)
    components_by_layer: dict[str, dict[str, np.ndarray]] = {}
    for layer_id in layers_to_collect:
        log(f"  collect L{layer_id}")
        dirs = {}
        for name, meta in candidates.items():
            pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, batch_size, max_length)
            neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, batch_size, max_length)
            dirs[name] = mean_dir(pos_h, neg_h)
        components_by_layer[str(layer_id)] = p548.build_components(pair, dirs, W_U, tokenizer, layer_id)
    return components_by_layer


def tensor_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_output(output: Any, new_tensor: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_tensor,) + output[1:]
    return new_tensor


def project_remove(x: torch.Tensor, pos: torch.Tensor, direction: torch.Tensor, scale: float) -> torch.Tensor:
    out = x.clone()
    bidx = torch.arange(out.shape[0], device=out.device)
    vecs = out[bidx, pos, :].float()
    d = direction.to(out.device).float()
    d = d / (d.norm() + 1e-8)
    coeff = (vecs * d).sum(dim=-1, keepdim=True)
    proj = coeff * d.unsqueeze(0)
    out[bidx, pos, :] = out[bidx, pos, :] - float(scale) * proj.to(out.dtype)
    return out


def add_direction(x: torch.Tensor, pos: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    out = x.clone()
    bidx = torch.arange(out.shape[0], device=out.device)
    d = direction.to(out.device).float()
    d = d / (d.norm() + 1e-8)
    out[bidx, pos, :] = out[bidx, pos, :] + (float(alpha) * d).to(out.dtype)
    return out


def module_for_site(layer: Any, site: str) -> Any:
    if site == "resid":
        return layer
    if site == "attn":
        return layer.self_attn
    if site == "mlp":
        return layer.mlp
    raise ValueError(f"unknown site: {site}")


def condition_plan(condition: str) -> dict[str, Any]:
    if condition == "baseline":
        return {"site": "none", "component": None, "remove": False, "add": False}
    if condition == "add_perp":
        return {"site": "resid", "component": "residual_perp", "remove": False, "add": True}
    if condition == "resid_remove_perp":
        return {"site": "resid", "component": "residual_perp", "remove": True, "add": False}
    if condition == "resid_remove_full":
        return {"site": "resid", "component": "residual_full", "remove": True, "add": False}
    if condition == "resid_remove_random_perp":
        return {"site": "resid", "component": "random_perp", "remove": True, "add": False}
    if condition == "resid_remove_perp_add_perp":
        return {"site": "resid", "component": "residual_perp", "remove": True, "add": True}
    if condition == "attn_remove_perp":
        return {"site": "attn", "component": "residual_perp", "remove": True, "add": False}
    if condition == "mlp_remove_perp":
        return {"site": "mlp", "component": "residual_perp", "remove": True, "add": False}
    if condition == "attn_remove_full":
        return {"site": "attn", "component": "residual_full", "remove": True, "add": False}
    if condition == "mlp_remove_full":
        return {"site": "mlp", "component": "residual_full", "remove": True, "add": False}
    raise ValueError(f"unknown condition: {condition}")


def batched_next_logits_surgery(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    texts: list[str],
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    condition: str,
    batch_size: int,
    max_length: int,
    remove_scale: float,
    add_alpha: float,
) -> np.ndarray:
    plan = condition_plan(condition)
    outs = []
    for start in range(0, len(texts), batch_size):
        batch = p544.encode_batch(tokenizer, texts[start:start + batch_size], device, max_length)
        pos = batch["attention_mask"].sum(dim=1) - 1
        handles = []
        if plan["site"] != "none":
            for layer_id in layer_ids:
                layer = layers[layer_id]
                site = module_for_site(layer, plan["site"])
                layer_device = next(site.parameters()).device
                pos_local = pos.to(layer_device)
                direction_np = components_by_layer[str(layer_id)][plan["component"]]
                direction = torch.tensor(normalize_vec(direction_np), dtype=torch.float32, device=layer_device)

                def make_hook(site_name: str, d_vec: torch.Tensor, p_vec: torch.Tensor):
                    def hook(_module: Any, _inp: Any, output: Any):
                        hs = tensor_from_output(output)
                        out = hs
                        if plan["remove"]:
                            out = project_remove(out, p_vec.to(out.device), d_vec.to(out.device), remove_scale)
                        if plan["add"] and site_name == "resid":
                            out = add_direction(out, p_vec.to(out.device), d_vec.to(out.device), add_alpha)
                        return replace_output(output, out)
                    return hook

                handles.append(site.register_forward_hook(make_hook(plan["site"], direction, pos_local)))
        with torch.inference_mode():
            out = model(**batch, return_dict=True, use_cache=False)
            idx = pos.to(out.logits.device)
            rows = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), idx]
            outs.append(rows.float().cpu().numpy().astype(np.float32))
        for handle in handles:
            handle.remove()
        del out, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


def run_linear_decode_surgery(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    condition: str,
    groups: dict[str, list[int]],
    mode: str,
    max_new_tokens: int,
    batch_size: int,
    max_length: int,
    seed: int,
    temperature: float,
    top_p: float,
    remove_scale: float,
    add_alpha: float,
) -> tuple[list[list[int]], list[str], list[str], list[float], list[float]]:
    texts = list(prompts)
    generated: list[list[int]] = [[] for _ in prompts]
    first_types, target_ranks, competitor_ranks = [], [], []
    rng = np.random.default_rng(seed)
    for step in range(max_new_tokens):
        logits = batched_next_logits_surgery(
            model, tokenizer, device, layers, texts, components_by_layer, layer_ids, condition,
            batch_size, max_length, remove_scale, add_alpha
        )
        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        if step == 0:
            for row, tok in zip(logits, toks):
                first_types.append(p544.token_type(tok, groups))
                target_ranks.append(p544.best_rank(row, groups["target"]))
                competitor_ranks.append(p544.best_rank(row, groups["competitor"]))
        for i, tok in enumerate(toks):
            generated[i].append(tok)
            texts[i] += tokenizer.decode([tok], skip_special_tokens=False)
    suffixes = [texts[i][len(prompts[i]):] for i in range(len(prompts))]
    return generated, suffixes, first_types, target_ranks, competitor_ranks


def decode_and_classify_surgery(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt_rows: list[dict[str, str]],
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    condition: str,
    groups: dict[str, list[int]],
    pair: str,
    mode: str,
    max_new_tokens: int,
    batch_size: int,
    max_length: int,
    seed: int,
    temperature: float,
    top_p: float,
    remove_scale: float,
    add_alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = [r["prompt"] for r in prompt_rows]
    generated, suffixes, first_types, target_ranks, competitor_ranks = run_linear_decode_surgery(
        model, tokenizer, device, layers, prompts, components_by_layer, layer_ids, condition,
        groups, mode, max_new_tokens, batch_size, max_length, seed, temperature, top_p,
        remove_scale, add_alpha
    )
    pos_label, neg_label = PAIR_SPECS[pair]
    records = []
    for i, (row, suffix, ids, first_type, target_rank, competitor_rank) in enumerate(
        zip(prompt_rows, suffixes, generated, first_types, target_ranks, competitor_ranks)
    ):
        cls = p548.classify_suffix(suffix, row["object"], pos_label, neg_label)
        records.append({
            "prompt_index": i,
            "object": row["object"],
            "prompt": row["prompt"],
            "generated_suffix": suffix,
            "generated_ids": ids,
            "first_type": first_type,
            "first_target_rank": float(target_rank),
            "first_competitor_rank": float(competitor_rank),
            **cls,
        })
    return p548.aggregate(records), records


def compact_metrics(row: dict[str, Any], base: dict[str, Any], add_ref: dict[str, Any], random_ref: dict[str, Any]) -> dict[str, float | str]:
    clean_delta = row["clean_non_object_rate"] - base["clean_non_object_rate"]
    score_delta = row["clean_non_object_score"] - base["clean_non_object_score"]
    label_delta = row["any_label_violation_rate"] - base["any_label_violation_rate"]
    add_gain = add_ref["clean_non_object_rate"] - base["clean_non_object_rate"]
    random_drop = random_ref["clean_non_object_rate"] - base["clean_non_object_rate"]
    if clean_delta <= -0.10 and score_delta <= -0.08 and label_delta <= 0.05:
        cls = "necessity_drop"
    elif clean_delta <= -0.06 and score_delta <= -0.04:
        cls = "weak_drop"
    elif label_delta >= 0.12:
        cls = "label_leak_or_noise"
    elif clean_delta >= 0.08:
        cls = "positive_add_or_release"
    else:
        cls = "flat"
    return {
        "clean_delta": float(clean_delta),
        "score_delta": float(score_delta),
        "label_delta": float(label_delta),
        "add_gain": float(add_gain),
        "random_delta": float(random_drop),
        "drop_vs_random": float(clean_delta - random_drop),
        "class": cls,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    scaffold_modes = parse_scaffold_modes(args.scaffold_modes)
    scaffolds = sorted(set(x for x, _ in scaffold_modes))
    conditions = parse_csv(args.conditions)
    sample_seeds = parse_int_csv(args.sample_seeds)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        if len(windows) != 1:
            raise ValueError(f"Phase552 expects one window, got {windows}")
        _, window = next(iter(windows.items()))
        combos = combo_layers(window, args.layer_sets)
        all_layers = sorted(set(itertools.chain.from_iterable(combos.values())))
        W_U = get_W_U(model, args.model).astype(np.float32)
        groups = p544.token_groups(tokenizer, pair)
        prompt_sets = p548.build_prompts(pair, args.test_n, scaffolds)
        components_by_layer = build_components_by_layer(
            model, tokenizer, device, pair, all_layers, args.train_n, args.batch_size, args.max_length, W_U
        )
        log(f"{args.model}: phase552 pair={pair}, combos={combos}, scaffold_modes={scaffold_modes}")

        audit: dict[str, Any] = {}
        compact = []
        saved_samples: list[dict[str, Any]] = []
        all_tsv: list[dict[str, Any]] = []
        for combo_name, layer_ids in combos.items():
            audit[combo_name] = {"layers": layer_ids, "rows": {}}
            for scaffold, mode in scaffold_modes:
                key = f"{scaffold}:{mode}"
                audit[combo_name]["rows"][key] = {}
                prompt_rows = prompt_sets[scaffold]
                for condition in conditions:
                    all_records = []
                    seed_rows = []
                    for seed in sample_seeds:
                        agg, records = decode_and_classify_surgery(
                            model, tokenizer, device, layers, prompt_rows, components_by_layer,
                            layer_ids, condition, groups, pair, mode, args.max_new_tokens,
                            args.batch_size, args.max_length, seed, args.temperature, args.top_p,
                            args.remove_scale, args.add_alpha,
                        )
                        seed_rows.append({"seed": seed, **agg})
                        for rec in records:
                            rec2 = {
                                "combo": combo_name,
                                "layers": layer_ids,
                                "pair": pair,
                                "scaffold": scaffold,
                                "mode": mode,
                                "condition": condition,
                                "seed": seed,
                                **rec,
                            }
                            all_records.append(rec2)
                    row = p548.aggregate(all_records)
                    row["seed_aggregates"] = seed_rows
                    audit[combo_name]["rows"][key][condition] = row
                    saved_samples.extend(all_records[: args.samples_per_row])
                    all_tsv.extend(all_records)
                rows = audit[combo_name]["rows"][key]
                base = rows["baseline"]
                add_ref = rows.get("add_perp", base)
                random_ref = rows.get("resid_remove_random_perp", base)
                for condition, row in rows.items():
                    if condition == "baseline":
                        continue
                    compact.append({
                        "combo": combo_name,
                        "layers": layer_ids,
                        "scaffold": scaffold,
                        "mode": mode,
                        "condition": condition,
                        "base_clean_non_object_rate": base["clean_non_object_rate"],
                        "clean_non_object_rate": row["clean_non_object_rate"],
                        "base_label_violation_rate": base["any_label_violation_rate"],
                        "label_violation_rate": row["any_label_violation_rate"],
                        "object_echo_rate": row["object_echo_rate"],
                        "prompt_echo_rate": row["prompt_echo_rate"],
                        "clean_non_object_score": row["clean_non_object_score"],
                        **compact_metrics(row, base, add_ref, random_ref),
                    })
                rp = rows.get("resid_remove_perp", base)
                ap = rows.get("add_perp", base)
                attn = rows.get("attn_remove_perp", base)
                mlp = rows.get("mlp_remove_perp", base)
                log(
                    f"    {combo_name} {key}: base={base['clean_non_object_rate']:.2f}; "
                    f"add={ap['clean_non_object_rate']:.2f}; "
                    f"resid_rm={rp['clean_non_object_rate']:.2f}; "
                    f"attn_rm={attn['clean_non_object_rate']:.2f}; "
                    f"mlp_rm={mlp['clean_non_object_rate']:.2f}"
                )

        return {
            "phase": 552,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pair": pair,
            "window": window,
            "combos": combos,
            "conditions": conditions,
            "scaffold_modes": [f"{a}:{b}" for a, b in scaffold_modes],
            "train_n": args.train_n,
            "test_n": args.test_n,
            "sample_seeds": sample_seeds,
            "remove_scale": args.remove_scale,
            "add_alpha": args.add_alpha,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "audit": audit,
            "compact_rows": compact,
            "sample_records": saved_samples[: args.max_saved_samples],
            "all_records_for_tsv": all_tsv[: args.max_tsv_records],
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_tsv(result: dict[str, Any], out_dir: Path, model_name: str) -> None:
    fields = [
        "combo", "layers", "pair", "scaffold", "mode", "condition", "seed", "object", "quality",
        "clean_non_object", "any_label_violation", "object_echo", "prompt_echo",
        "target_non_object_matches", "target_label_matches", "competitor_synonym_matches",
        "generated_suffix",
    ]
    lines = ["\t".join(fields)]
    for rec in result.get("all_records_for_tsv", []):
        vals = []
        for field in fields:
            val = rec.get(field, "")
            if isinstance(val, list):
                val = ",".join(str(x) for x in val)
            vals.append(str(val).replace("\t", " ").replace("\n", " "))
        lines.append("\t".join(vals))
    path = out_dir / f"phase552_{model_name}_readable_samples.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--windows", default=None)
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=12)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127,131,137")
    parser.add_argument("--scaffold-modes", default=",".join(DEFAULT_SCAFFOLD_MODES))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--layer-sets", default="")
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--add-alpha", type=float, default=6.0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--samples-per-row", type=int, default=2)
    parser.add_argument("--max-saved-samples", type=int, default=1200)
    parser.add_argument("--max-tsv-records", type=int, default=8000)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase552_{args.model}_paraphrase_necessity_writer_decomposition.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tsv(result, out_dir, args.model)
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
