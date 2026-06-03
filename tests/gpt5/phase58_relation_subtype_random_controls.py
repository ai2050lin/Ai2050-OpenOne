from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "glm5"))

from phase344_345_multi_relation import (  # noqa: E402
    MODEL_CONFIGS,
    capture_mlp_internals,
    channel_decomposition,
    get_W_U,
    get_layers,
    get_mlp_weights,
    get_token_id,
    interaction_decomposition,
    load_model_bf16,
)


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def safe_mean(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def safe_std(xs: list[float]) -> float:
    return float(pstdev(xs)) if len(xs) > 1 else 0.0


def cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def contextualize(rows: list[tuple[str, str, str, str, str, str]], max_pairs: int) -> list[tuple[str, str, str, str, str, str]]:
    prefixes = ["", "Usually, ", "In general, ", "People know ", "In this sentence, ", "For this example, "]
    out: list[tuple[str, str, str, str, str, str]] = []
    seen = set()
    for prefix in prefixes:
        for rel, subtype, clean, corrupt, target, competitor in rows:
            item = (rel, subtype, f"{prefix}{clean}", f"{prefix}{corrupt}", target, competitor)
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
            if len(out) >= max_pairs:
                return out
    return out


def build_subtype_pairs(max_pairs_per_subtype: int) -> dict[str, dict[str, list[tuple[str, str, str, str, str, str]]]]:
    data: dict[str, dict[str, list[tuple[str, str, str, str, str, str]]]] = {}

    def add(rel: str, subtype: str, rows: list[tuple[str, str, str, str, str, str]]) -> None:
        data.setdefault(rel, {})[subtype] = contextualize(rows, max_pairs_per_subtype)

    add("binding", "color", [
        ("binding", "color", f"The {obj}", "The item", good, bad)
        for obj, good, bad in [
            ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
            ("sky", "blue", "green"), ("grass", "green", "red"), ("coal", "black", "white"),
            ("lemon", "yellow", "blue"), ("rose", "red", "green"), ("milk", "white", "black"),
        ]
    ])
    add("binding", "temperature", [
        ("binding", "temperature", f"The {obj}", "The item", good, bad)
        for obj, good, bad in [
            ("fire", "hot", "cold"), ("ice", "cold", "hot"), ("coffee", "hot", "cold"),
            ("snow", "cold", "hot"), ("sun", "hot", "cold"), ("steam", "hot", "cold"),
        ]
    ])
    add("binding", "texture", [
        ("binding", "texture", f"The {obj}", "The item", good, bad)
        for obj, good, bad in [
            ("stone", "hard", "soft"), ("feather", "soft", "hard"), ("sandpaper", "rough", "smooth"),
            ("glass", "smooth", "rough"), ("pillow", "soft", "hard"), ("steel", "hard", "soft"),
        ]
    ])
    add("binding", "taste", [
        ("binding", "taste", f"The {obj}", "The item", good, bad)
        for obj, good, bad in [
            ("honey", "sweet", "bitter"), ("salt", "salty", "sweet"), ("pepper", "spicy", "plain"),
            ("lemon", "sour", "sweet"), ("sugar", "sweet", "salty"), ("vinegar", "sour", "sweet"),
        ]
    ])

    add("negation", "syntactic_not", [
        ("negation", "syntactic_not", f"The {obj} is {good}", f"The {obj} is not", good, "not")
        for obj, good in [("apple", "red"), ("sky", "blue"), ("fire", "hot"), ("snow", "white"), ("grass", "green"), ("door", "open"), ("room", "dark")]
    ])
    add("negation", "quantifier_no", [
        ("negation", "quantifier_no", f"No {noun}s are here", f"Some {noun}s are here", "no", "some")
        for noun in ["cat", "dog", "bird", "student", "teacher", "apple", "car", "book"]
    ])

    add("role", "active_swap", [
        ("role", "active_swap", f"The {a} {verb} the {b}", f"The {b} {verb} the {a}", a, b)
        for a, b, verb in [
            ("cat", "dog", "chased"), ("boy", "girl", "helped"), ("teacher", "student", "questioned"),
            ("doctor", "nurse", "called"), ("chef", "guest", "served"), ("artist", "critic", "admired"),
            ("lawyer", "judge", "met"), ("pilot", "driver", "saw"),
        ]
    ])
    add("same_class", "category_peer", [
        ("same_class", "category_peer", f"The {a}", f"The {b}", a, b)
        for a, b in [
            ("apple", "banana"), ("cat", "dog"), ("rose", "lily"), ("car", "bus"), ("oak", "pine"),
            ("teacher", "doctor"), ("chair", "table"), ("river", "lake"), ("piano", "guitar"),
        ]
    ])
    add("quantifier", "all_some", [
        ("quantifier", "all_some", f"All {noun}s are here", f"Some {noun}s are here", "all", "some")
        for noun in ["cat", "dog", "bird", "student", "teacher", "apple", "car", "book"]
    ])
    add("temporal_order", "before_after", [
        ("temporal_order", "before_after", f"{a} happened before {b}", f"{a} happened after {b}", "before", "after")
        for a, b in [("breakfast", "lunch"), ("morning", "night"), ("spring", "summer"), ("start", "finish"), ("birth", "death"), ("cause", "effect")]
    ])
    add("condition", "if_unless", [
        ("condition", "if_unless", f"If {event}, then it changes", f"Unless {event}, it changes", "if", "unless")
        for event in ["it rains", "he works", "she calls", "they agree", "the door opens", "the alarm rings"]
    ])
    add("contrast", "but_and", [
        ("contrast", "but_and", f"{a}, but {b}", f"{a}, and {b}", "but", "and")
        for a, b in [
            ("It rained", "the ground stayed dry"), ("He studied", "he failed"), ("She smiled", "she was sad"),
            ("The fire burned", "the room stayed cold"), ("The team played well", "they lost"),
        ]
    ])
    add("comparison", "greater_less", [
        ("comparison", "greater_less", f"The {a} is {x} than the {b}", f"The {a} is {y} than the {b}", x, y)
        for a, b, x, y in [
            ("elephant", "mouse", "larger", "smaller"), ("tower", "house", "taller", "shorter"),
            ("stone", "feather", "heavier", "lighter"), ("car", "bicycle", "faster", "slower"),
            ("adult", "child", "older", "younger"),
        ]
    ])
    return data


def aggregate(rows: list[dict[str, float]], target_layers: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_layer": {}}
    for li in target_layers:
        lr = [r for r in rows if int(r["layer"]) == li]
        out["per_layer"][str(li)] = {
            "net_gross_mean": safe_mean([r["net_gross"] for r in lr]),
            "balance_mean": safe_mean([r["balance"] for r in lr]),
            "interaction_frac": safe_mean([r["interaction_frac"] for r in lr]),
            "n": len(lr),
        }
    out.update({
        "net_gross_mean": safe_mean([r["net_gross"] for r in rows]),
        "net_gross_std": safe_std([r["net_gross"] for r in rows]),
        "balance_mean": safe_mean([r["balance"] for r in rows]),
        "interaction_frac": safe_mean([r["interaction_frac"] for r in rows]),
        "n": len(rows),
    })
    return out


def signature(summary: dict[str, Any], target_layers: list[int]) -> list[float]:
    sig = [summary["net_gross_mean"], summary["balance_mean"], summary["interaction_frac"]]
    for li in target_layers:
        p = summary.get("per_layer", {}).get(str(li), {})
        sig.extend([p.get("net_gross_mean", 0.0), p.get("balance_mean", 0.0), p.get("interaction_frac", 0.0)])
    return [float(x) for x in sig]


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["PHASE344_ATTN_IMPLEMENTATIONS"] = args.attn_implementations
    model_name = args.model
    log(f"Phase58 Relation Subtype Random Controls — {model_name}")
    log("=" * 72)
    t0 = time.time()
    np.random.seed(args.seed)
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    target_layers = cfg["binding_layers"]
    W_U = get_W_U(model, model_name)
    layers = get_layers(model)
    d_model = W_U.shape[1]
    log(f"W_U shape={W_U.shape}, target_layers={target_layers}")

    mlp_weights = {}
    for li in target_layers:
        _, _, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_down": W_down, "d_ff": d_ff}

    # W_U-subspace random control basis.
    subset = W_U[: min(args.svd_vocab, W_U.shape[0])]
    _u, _s, vt = np.linalg.svd(subset, full_matrices=False)
    wu_basis = vt[:d_model]

    datasets = build_subtype_pairs(args.max_pairs_per_subtype)
    capture_cache: dict[str, dict[str, np.ndarray]] = {}
    subtype_results: dict[str, dict[str, Any]] = {}

    for rel, subtypes in datasets.items():
        for subtype, rows in subtypes.items():
            key = f"{rel}/{subtype}"
            log(f"\nsubtype={key}, pairs={len(rows)}")
            real_rows: list[dict[str, float]] = []
            controls = {
                "norm_matched": [],
                "wu_subspace": [],
                "relation_orthogonal": [],
                "pure_random": [],
            }
            for pidx, (_rel, _sub, clean, corrupt, target, competitor) in enumerate(rows):
                tid_t = get_token_id(tokenizer, target)
                tid_c = get_token_id(tokenizer, competitor)
                if tid_t is None or tid_c is None:
                    continue
                direction = W_U[tid_t] - W_U[tid_c]
                norm = float(np.linalg.norm(direction))
                if norm < 1e-10:
                    continue
                direction = direction / norm
                if clean not in capture_cache:
                    capture_cache[clean] = capture_mlp_internals(model, tokenizer, device, clean, target_layers, cfg["n_layers"])
                if corrupt not in capture_cache:
                    capture_cache[corrupt] = capture_mlp_internals(model, tokenizer, device, corrupt, target_layers, cfg["n_layers"])
                clean_caps = capture_cache[clean]
                corrupt_caps = capture_cache[corrupt]

                random_dirs: dict[str, list[np.ndarray]] = {k: [] for k in controls}
                for _ in range(args.random_samples_per_pair):
                    rnd = np.random.randn(d_model)
                    rnd = rnd / max(np.linalg.norm(rnd), 1e-10)
                    random_dirs["pure_random"].append(rnd)
                    norm_rnd = np.random.randn(d_model) * norm
                    norm_rnd = norm_rnd / max(np.linalg.norm(norm_rnd), 1e-10)
                    random_dirs["norm_matched"].append(norm_rnd)
                    coeffs = np.random.randn(wu_basis.shape[0])
                    wu_rnd = wu_basis.T @ coeffs
                    wu_rnd = wu_rnd / max(np.linalg.norm(wu_rnd), 1e-10)
                    random_dirs["wu_subspace"].append(wu_rnd)
                    orth = np.random.randn(d_model)
                    orth = orth - float(orth @ direction) * direction
                    orth = orth / max(np.linalg.norm(orth), 1e-10)
                    random_dirs["relation_orthogonal"].append(orth)

                for li in target_layers:
                    W_down = mlp_weights[li]["W_down"]
                    d_ff = mlp_weights[li]["d_ff"]
                    if W_down is None:
                        continue
                    gk = f"gate_{li}"
                    uk = f"up_{li}"
                    if gk not in clean_caps or gk not in corrupt_caps:
                        continue
                    cg = clean_caps[gk][:d_ff]
                    crg = corrupt_caps[gk][:d_ff]
                    cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
                    cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
                    cd = channel_decomposition(W_down, direction, cg, cu, crg, cru)
                    it = interaction_decomposition(W_down, direction, cg, cu, crg, cru)
                    real_rows.append({
                        "layer": float(li),
                        "net_gross": float(cd["net_gross_ratio"]),
                        "balance": float(cd["balance"]),
                        "interaction_frac": float(abs(it["interaction"]) / max(abs(it["gate_main"]) + abs(it["up_main"]) + abs(it["interaction"]), 1e-10)),
                    })
                    for cname, dirs in random_dirs.items():
                        for d in dirs:
                            rcd = channel_decomposition(W_down, d, cg, cu, crg, cru)
                            controls[cname].append({
                                "layer": float(li),
                                "net_gross": float(rcd["net_gross_ratio"]),
                                "balance": float(rcd["balance"]),
                                "interaction_frac": 0.0,
                            })
                if (pidx + 1) % args.progress_every == 0 or pidx == len(rows) - 1:
                    log(f"  {key}: {pidx+1}/{len(rows)} elapsed={time.time()-t0:.0f}s")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            real = aggregate(real_rows, target_layers)
            control_summary = {name: aggregate(crows, target_layers) for name, crows in controls.items()}
            random_mean = safe_mean([control_summary[name]["net_gross_mean"] for name in control_summary])
            real["random_mean_net_gross"] = random_mean
            real["random_advantage"] = real["net_gross_mean"] - random_mean
            subtype_results[key] = {
                "relation": rel,
                "subtype": subtype,
                "real": real,
                "controls": control_summary,
            }
            log(
                f"  {key}: real={real['net_gross_mean']:.4f}, "
                f"random={random_mean:.4f}, advantage={real['random_advantage']:.4f}, "
                f"n={real['n']}"
            )

    sigs = {k: signature(v["real"], target_layers) for k, v in subtype_results.items()}
    sim = {a: {b: cosine(sigs[a], sigs[b]) for b in sigs} for a in sigs}
    result = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementations": args.attn_implementations,
        "target_layers": target_layers,
        "max_pairs_per_subtype": args.max_pairs_per_subtype,
        "random_samples_per_pair": args.random_samples_per_pair,
        "subtypes": subtype_results,
        "subtype_similarity_matrix": sim,
        "elapsed_sec": time.time() - t0,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase58_relation_subtype_random_controls.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Saved {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--output-dir", default=os.environ.get("PHASE58_OUTPUT_DIR", "results/gpt5_phase58_relation_subtype_random_controls_full"))
    parser.add_argument("--max-pairs-per-subtype", type=int, default=int(os.environ.get("PHASE58_MAX_PAIRS_PER_SUBTYPE", "30")))
    parser.add_argument("--random-samples-per-pair", type=int, default=int(os.environ.get("PHASE58_RANDOM_SAMPLES_PER_PAIR", "2")))
    parser.add_argument("--svd-vocab", type=int, default=int(os.environ.get("PHASE58_SVD_VOCAB", "5000")))
    parser.add_argument("--attn-implementations", default=os.environ.get("PHASE58_ATTN_IMPLEMENTATIONS", "flash_attention_2,sdpa,eager"))
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        run_model(args)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
