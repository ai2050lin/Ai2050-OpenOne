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

import numpy as np
import torch


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from model_registry import get_model_spec  # noqa: E402


FRAMES = [
    "The {obj} is {attr}.",
    "An {obj} is {attr}.",
    "This {obj} is {attr}.",
    "That {obj} is {attr}.",
]

CORRUPT_FRAMES = [
    "The item is {attr}.",
    "An item is {attr}.",
    "This item is {attr}.",
    "That item is {attr}.",
]

NEUTRAL_FRAMES = [
    "The {obj} is",
    "An {obj} is",
    "This {obj} is",
    "That {obj} is",
]

NEUTRAL_CORRUPT_FRAMES = [
    "The item is",
    "An item is",
    "This item is",
    "That item is",
]

RICH_DATA = {
    "moisture": {
        "objects": {
            "ocean": [("wet", "dry"), ("wet", "arid")],
            "rain": [("wet", "dry"), ("wet", "arid")],
            "river": [("wet", "dry"), ("wet", "arid")],
            "desert": [("dry", "wet"), ("dry", "moist")],
            "sand": [("dry", "wet"), ("dry", "moist")],
            "dust": [("dry", "wet"), ("dry", "moist")],
        },
    },
    "color": {
        "objects": {
            "apple": [("red", "blue"), ("red", "green")],
            "cherry": [("red", "blue"), ("red", "green")],
            "sky": [("blue", "red"), ("blue", "green")],
            "ocean_c": [("blue", "red"), ("blue", "green")],
            "snow": [("white", "black"), ("white", "gray")],
            "grass": [("green", "blue"), ("green", "red")],
        },
    },
    "size": {
        "objects": {
            "elephant": [("big", "small"), ("large", "tiny")],
            "mountain": [("big", "small"), ("large", "tiny")],
            "whale": [("big", "small"), ("large", "tiny")],
            "ant": [("small", "big"), ("tiny", "large")],
            "grain": [("small", "big"), ("tiny", "large")],
            "pin": [("small", "big"), ("tiny", "large")],
        },
    },
}

DISPLAY_TO_PROMPT = {"ocean_c": "ocean"}


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_model(model_name: str, attn_impls: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_model_spec(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.local_dir, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    errors: list[str] = []
    for impl in [x.strip() for x in attn_impls.split(",") if x.strip()]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                spec.local_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=impl,
            )
            log(f"Loaded {model_name} with attn_impl={impl}")
            break
        except Exception as exc:
            errors.append(f"{impl}: {exc}")
            log(f"Failed with {impl}: {exc}")
    if model is None:
        raise RuntimeError("failed to load model: " + " | ".join(errors))
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError(f"cannot find layers in {type(model).__name__}")


def get_d_model(model) -> int:
    return int(model.get_input_embeddings().weight.shape[1])


def build_pairs(max_pairs: int | None = None) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for cat, cat_data in RICH_DATA.items():
        for obj_name, value_combos in cat_data["objects"].items():
            prompt_obj = DISPLAY_TO_PROMPT.get(obj_name, obj_name)
            for value_idx, (target, comp) in enumerate(value_combos):
                for frame_idx in range(len(FRAMES)):
                    pairs.append(
                        {
                            "obj": obj_name,
                            "prompt_obj": prompt_obj,
                            "target": target,
                            "comp": comp,
                            "cat": cat,
                            "frame_idx": frame_idx,
                            "value_idx": value_idx,
                        }
                    )
    return pairs[:max_pairs] if max_pairs else pairs


def token_id(tokenizer, token: str) -> int | None:
    ids = tokenizer.encode(token, add_special_tokens=False)
    return int(ids[0]) if ids else None


def get_logit_stats(logits_tensor, target_id: int | None, comp_id: int | None) -> dict[str, float]:
    logits = logits_tensor.float().detach().cpu().numpy()
    t_logit = float(logits[target_id]) if target_id is not None else 0.0
    c_logit = float(logits[comp_id]) if comp_id is not None else 0.0
    return {"t_logit": t_logit, "c_logit": c_logit, "gap": t_logit - c_logit}


def classify_mechanism(td: float, cd: float) -> str:
    if td > 0 and cd < 0:
        return "IDEAL"
    if td > 0 and cd > 0:
        return "DOM_BOOST" if td > cd else "BOOST_C"
    if td < 0 and cd > 0:
        return "REVERSED"
    if td < 0 and cd < 0:
        return "SUPP_T" if abs(td) > abs(cd) else "SUPP_C"
    return "MIXED"


def make_add_hook(delta: torch.Tensor):
    def hook_fn(_module, _input, output):
        hs = output[0].clone() if isinstance(output, tuple) else output.clone()
        hs[0, -1, :] += delta
        return (hs,) + output[1:] if isinstance(output, tuple) else hs

    return hook_fn


def forward_capture(model, tokenizer, device, prompt: str, layer, max_length: int) -> tuple[np.ndarray, Any]:
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _input, output):
        captured["h"] = output[0].detach().float().cpu() if isinstance(output, tuple) else output.detach().float().cpu()

    handle = layer.register_forward_hook(hook_fn)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
        h = captured["h"][0, -1].numpy()
    finally:
        handle.remove()
    return h, out


def forward_with_delta(
    model,
    tokenizer,
    device,
    prompt: str,
    layer,
    delta_np: np.ndarray,
    max_length: int,
    target_id: int | None,
    comp_id: int | None,
) -> dict[str, float]:
    delta = torch.tensor(delta_np, dtype=torch.bfloat16, device=device)
    handle = layer.register_forward_hook(make_add_hook(delta))
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
        return get_logit_stats(out.logits[0, -1], target_id, comp_id)
    finally:
        handle.remove()


def layer_summary(obj_results: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    neutral_ideal = 0
    rows = list(obj_results.values())
    for row in rows:
        key = f"{row['version']}:{row['symmetric']}"
        counts[key] += 1
        by_cat[row["cat"]][key] += 1
        if row["neutral_mech"] == "IDEAL":
            neutral_ideal += 1
    return {
        "num_rows": len(rows),
        "counts": dict(counts),
        "by_cat": {k: dict(v) for k, v in by_cat.items()},
        "neutral_ideal_count": neutral_ideal,
        "full_symmetric_count": sum(1 for r in rows if r["symmetric"] == "FULL"),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE65_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers_list = get_layers(model)
    d_model = get_d_model(model)
    pairs = build_pairs(args.max_pairs)
    log(f"Phase65 model={args.model} pairs={len(pairs)} layers={args.layers}")

    value_tokens = sorted({p["target"] for p in pairs} | {p["comp"] for p in pairs})
    token_ids = {tok: token_id(tokenizer, tok) for tok in value_tokens}
    missing = [tok for tok, tid in token_ids.items() if tid is None]
    if missing:
        log(f"missing token ids: {missing}")

    results: dict[str, Any] = {
        "phase": 65,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "layers": args.layers,
        "num_pairs": len(pairs),
        "per_layer": {},
        "layer_summaries": {},
    }

    for li in args.layers:
        t0 = time.time()
        layer = layers_list[li]
        log(f"Layer {li}: collect activations")
        n = len(pairs)
        h_correct = np.zeros((n, d_model), dtype=np.float32)
        h_correct_corrupt = np.zeros((n, d_model), dtype=np.float32)
        h_incorrect = np.zeros((n, d_model), dtype=np.float32)
        h_incorrect_corrupt = np.zeros((n, d_model), dtype=np.float32)
        h_neutral = np.zeros((n, d_model), dtype=np.float32)
        h_neutral_corrupt = np.zeros((n, d_model), dtype=np.float32)
        baseline_correct: list[dict[str, float]] = []
        baseline_incorrect: list[dict[str, float]] = []
        baseline_neutral: list[dict[str, float]] = []

        for i, p in enumerate(pairs):
            tid = token_ids.get(p["target"])
            cid = token_ids.get(p["comp"])
            tpl = FRAMES[p["frame_idx"]]
            ctpl = CORRUPT_FRAMES[p["frame_idx"]]
            ntpl = NEUTRAL_FRAMES[p["frame_idx"]]
            nctpl = NEUTRAL_CORRUPT_FRAMES[p["frame_idx"]]

            correct_clean = tpl.format(obj=p["prompt_obj"], attr=p["target"])
            correct_corrupt = ctpl.format(attr=p["target"])
            incorrect_clean = tpl.format(obj=p["prompt_obj"], attr=p["comp"])
            incorrect_corrupt = ctpl.format(attr=p["comp"])
            neutral_clean = ntpl.format(obj=p["prompt_obj"])
            neutral_corrupt = nctpl

            h_correct[i], _ = forward_capture(model, tokenizer, device, correct_clean, layer, args.max_length)
            h_correct_corrupt[i], out = forward_capture(model, tokenizer, device, correct_corrupt, layer, args.max_length)
            baseline_correct.append(get_logit_stats(out.logits[0, -1], tid, cid))

            h_incorrect[i], _ = forward_capture(model, tokenizer, device, incorrect_clean, layer, args.max_length)
            h_incorrect_corrupt[i], out = forward_capture(model, tokenizer, device, incorrect_corrupt, layer, args.max_length)
            baseline_incorrect.append(get_logit_stats(out.logits[0, -1], tid, cid))

            h_neutral[i], out = forward_capture(model, tokenizer, device, neutral_clean, layer, args.max_length)
            baseline_neutral.append(get_logit_stats(out.logits[0, -1], tid, cid))
            h_neutral_corrupt[i], _ = forward_capture(model, tokenizer, device, neutral_corrupt, layer, args.max_length)

            if (i + 1) % args.progress_every == 0:
                log(f"Layer {li}: activation {i + 1}/{n} elapsed={time.time() - t0:.0f}s")

        dh_correct = h_correct - h_correct_corrupt
        _dh_incorrect = h_incorrect - h_incorrect_corrupt
        _dh_neutral = h_neutral - h_neutral_corrupt

        cat_labels = [p["cat"] for p in pairs]
        obj_labels = [p["obj"] for p in pairs]
        mu = dh_correct.mean(axis=0)
        unique_cats = sorted(set(cat_labels))
        cat_centroids = {cat: dh_correct[[i for i, c in enumerate(cat_labels) if c == cat]].mean(axis=0) for cat in unique_cats}
        a_cat = np.zeros_like(dh_correct)
        for i, cat in enumerate(cat_labels):
            a_cat[i] = cat_centroids[cat] - mu

        dh_resid_cat = dh_correct - mu - a_cat
        oc_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, p in enumerate(pairs):
            oc_groups[(p["obj"], p["cat"])].append(i)
        a_obj_cat_cf = np.zeros_like(dh_correct)
        for oc_key, group_indices in oc_groups.items():
            for test_i in group_indices:
                train = [j for j in group_indices if j != test_i]
                a_obj_cat_cf[test_i] = np.mean([dh_resid_cat[j] for j in train], axis=0) if train else np.zeros(d_model)

        obj_results: dict[str, Any] = {}
        for obj in sorted(set(obj_labels)):
            obj_indices = [i for i, x in enumerate(obj_labels) if x == obj]
            if not obj_indices:
                continue
            cat = cat_labels[obj_indices[0]]
            p0 = pairs[obj_indices[0]]
            tid = token_ids.get(p0["target"])
            cid = token_ids.get(p0["comp"])
            directions = {
                "L1": mu + a_cat[obj_indices[0]],
                "L2_cf": mu + a_cat[obj_indices[0]] + a_obj_cat_cf[obj_indices[0]],
                "OBJ_cf": a_obj_cat_cf[obj_indices[0]],
            }
            for version, direction in directions.items():
                td_corr: list[float] = []
                cd_corr: list[float] = []
                td_inc: list[float] = []
                cd_inc: list[float] = []
                td_neu: list[float] = []
                cd_neu: list[float] = []
                for idx in obj_indices:
                    p = pairs[idx]
                    ctpl = CORRUPT_FRAMES[p["frame_idx"]]
                    nctpl = NEUTRAL_CORRUPT_FRAMES[p["frame_idx"]]

                    stats = forward_with_delta(
                        model,
                        tokenizer,
                        device,
                        ctpl.format(attr=p["target"]),
                        layer,
                        direction,
                        args.max_length,
                        tid,
                        cid,
                    )
                    bs = baseline_correct[idx]
                    td_corr.append(stats["t_logit"] - bs["t_logit"])
                    cd_corr.append(stats["c_logit"] - bs["c_logit"])

                    stats = forward_with_delta(
                        model,
                        tokenizer,
                        device,
                        ctpl.format(attr=p["comp"]),
                        layer,
                        direction,
                        args.max_length,
                        tid,
                        cid,
                    )
                    bs = baseline_incorrect[idx]
                    td_inc.append(stats["t_logit"] - bs["t_logit"])
                    cd_inc.append(stats["c_logit"] - bs["c_logit"])

                    stats = forward_with_delta(
                        model,
                        tokenizer,
                        device,
                        nctpl,
                        layer,
                        direction,
                        args.max_length,
                        tid,
                        cid,
                    )
                    bs = baseline_neutral[idx]
                    td_neu.append(stats["t_logit"] - bs["t_logit"])
                    cd_neu.append(stats["c_logit"] - bs["c_logit"])

                corr_td = float(mean(td_corr))
                corr_cd = float(mean(cd_corr))
                inc_td = float(mean(td_inc))
                inc_cd = float(mean(cd_inc))
                neu_td = float(mean(td_neu))
                neu_cd = float(mean(cd_neu))
                corr_mech = classify_mechanism(corr_td, corr_cd)
                inc_mech = classify_mechanism(inc_td, inc_cd)
                neu_mech = classify_mechanism(neu_td, neu_cd)
                corr_ideal = corr_td > 0 and corr_cd < 0
                inc_ideal = inc_td > 0 and inc_cd < 0
                symmetric = "FULL" if corr_ideal and inc_ideal else "HALF" if corr_ideal else "NO"
                key = f"{obj}_{version}"
                obj_results[key] = {
                    "obj": obj,
                    "cat": cat,
                    "version": version,
                    "correct_td": corr_td,
                    "correct_cd": corr_cd,
                    "correct_mech": corr_mech,
                    "incorrect_td": inc_td,
                    "incorrect_cd": inc_cd,
                    "incorrect_mech": inc_mech,
                    "neutral_td": neu_td,
                    "neutral_cd": neu_cd,
                    "neutral_mech": neu_mech,
                    "symmetric": symmetric,
                }
        summary = layer_summary(obj_results)
        results["per_layer"][str(li)] = obj_results
        results["layer_summaries"][str(li)] = summary
        log(
            f"Layer {li}: full={summary['full_symmetric_count']} "
            f"neutral_ideal={summary['neutral_ideal_count']} done in {time.time() - t0:.0f}s"
        )

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = out_dir / f"{args.model}_phase65_object_attribute_compat_decomposition.partial.json"
        partial.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_path = out_dir / f"{args.model}_phase65_object_attribute_compat_decomposition{suffix}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    return results


def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layers", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=24)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    args.layers = parse_layers(args.layers)
    try:
        run_model(args)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    if args.hard_exit_after_model:
        log("Hard exit after model requested.")
        os._exit(0)


if __name__ == "__main__":
    main()
