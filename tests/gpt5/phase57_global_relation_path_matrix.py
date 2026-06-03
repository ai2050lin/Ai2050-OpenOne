from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from itertools import islice, product
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "glm5"))

from model_registry import get_model_spec  # noqa: E402
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


DEFAULT_RELATIONS = [
    "binding",
    "negation",
    "antonym",
    "role",
    "tense",
    "same_class",
    "coreference",
    "quantifier",
    "causal",
    "comparison",
    "spatial",
    "temporal_order",
    "condition",
    "contrast",
]


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def safe_mean(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def safe_std(xs: list[float]) -> float:
    return float(pstdev(xs)) if len(xs) > 1 else 0.0


def cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def take_pairs(rows: list[tuple[str, str, str, str, str]], n: int) -> list[tuple[str, str, str, str, str]]:
    if n <= 0 or len(rows) <= n:
        return rows
    return list(islice(rows, n))


def contextualize(rows: list[tuple[str, str, str, str, str]], max_pairs: int) -> list[tuple[str, str, str, str, str]]:
    prefixes = [
        "",
        "Usually, ",
        "In general, ",
        "People know ",
        "In this sentence, ",
        "For this example, ",
    ]
    out: list[tuple[str, str, str, str, str]] = []
    seen = set()
    for prefix in prefixes:
        for rel, clean, corrupt, target, competitor in rows:
            clean_v = f"{prefix}{clean}"
            corrupt_v = f"{prefix}{corrupt}"
            key = (rel, clean_v, corrupt_v, target, competitor)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= max_pairs:
                return out
    return out


def build_relation_pairs(max_pairs_per_relation: int) -> dict[str, list[tuple[str, str, str, str, str]]]:
    colors = [
        ("apple", "red", "blue"),
        ("banana", "yellow", "purple"),
        ("snow", "white", "black"),
        ("sky", "blue", "green"),
        ("fire", "hot", "cold"),
        ("grass", "green", "red"),
        ("sun", "yellow", "purple"),
        ("ocean", "blue", "yellow"),
        ("blood", "red", "green"),
        ("ice", "cold", "hot"),
        ("coal", "black", "white"),
        ("lemon", "yellow", "blue"),
        ("rose", "red", "green"),
        ("milk", "white", "black"),
        ("night", "dark", "bright"),
        ("stone", "hard", "soft"),
        ("feather", "soft", "hard"),
        ("honey", "sweet", "bitter"),
        ("salt", "salty", "sweet"),
        ("pepper", "spicy", "plain"),
    ]
    antonyms = [
        ("temperature", "hot", "cold"),
        ("size", "big", "small"),
        ("room", "dark", "bright"),
        ("surface", "rough", "smooth"),
        ("animal", "alive", "dead"),
        ("door", "open", "closed"),
        ("bag", "full", "empty"),
        ("road", "wide", "narrow"),
        ("sound", "loud", "quiet"),
        ("line", "straight", "curved"),
        ("water", "deep", "shallow"),
        ("child", "young", "old"),
    ]
    entities = [
        ("cat", "dog", "chased"),
        ("boy", "girl", "helped"),
        ("teacher", "student", "questioned"),
        ("king", "guard", "ordered"),
        ("dog", "owner", "followed"),
        ("doctor", "nurse", "called"),
        ("chef", "guest", "served"),
        ("father", "mother", "called"),
        ("actor", "writer", "praised"),
        ("pilot", "driver", "saw"),
        ("artist", "critic", "admired"),
        ("lawyer", "judge", "met"),
    ]
    same_class = [
        ("apple", "banana"),
        ("cat", "dog"),
        ("rose", "lily"),
        ("car", "bus"),
        ("oak", "pine"),
        ("teacher", "doctor"),
        ("chair", "table"),
        ("river", "lake"),
        ("silver", "gold"),
        ("bread", "rice"),
        ("piano", "guitar"),
        ("shirt", "coat"),
    ]
    tense = [
        ("he", "ran", "runs"),
        ("she", "walked", "walks"),
        ("they", "ate", "eat"),
        ("he", "sang", "sings"),
        ("bird", "flew", "flies"),
        ("child", "played", "plays"),
        ("teacher", "taught", "teaches"),
        ("dog", "slept", "sleeps"),
        ("cat", "jumped", "jumps"),
        ("worker", "built", "builds"),
    ]
    comparisons = [
        ("elephant", "mouse", "larger", "smaller"),
        ("tower", "house", "taller", "shorter"),
        ("river", "stream", "wider", "narrower"),
        ("stone", "feather", "heavier", "lighter"),
        ("fire", "ice", "hotter", "colder"),
        ("car", "bicycle", "faster", "slower"),
        ("adult", "child", "older", "younger"),
        ("mountain", "hill", "higher", "lower"),
    ]
    locations = [
        ("cup", "table", "above", "below"),
        ("bird", "tree", "above", "below"),
        ("book", "shelf", "on", "under"),
        ("cat", "bed", "under", "on"),
        ("lamp", "desk", "beside", "behind"),
        ("chair", "door", "near", "far"),
    ]
    temporal = [
        ("breakfast", "lunch", "before", "after"),
        ("morning", "night", "before", "after"),
        ("spring", "summer", "before", "after"),
        ("cause", "effect", "before", "after"),
        ("birth", "death", "before", "after"),
        ("start", "finish", "before", "after"),
    ]

    pairs: dict[str, list[tuple[str, str, str, str, str]]] = {}
    pairs["binding"] = [
        ("binding", f"The {obj}", "The item", good, bad)
        for obj, good, bad in colors
    ]
    pairs["negation"] = [
        ("negation", f"The {obj} is {good}", f"The {obj} is not", good, "not")
        for obj, good, _bad in colors[:14]
    ]
    pairs["antonym"] = [
        ("antonym", f"The {noun} is {a}", f"The {noun} is", a, b)
        for noun, a, b in antonyms
    ]
    pairs["role"] = [
        ("role", f"The {a} {verb} the {b}", f"The {b} {verb} the {a}", a, b)
        for a, b, verb in entities
    ]
    pairs["tense"] = [
        ("tense", f"Yesterday {subj} {past}", f"Today {subj} {present}", past, present)
        for subj, past, present in tense
    ]
    pairs["same_class"] = [
        ("same_class", f"The {a}", f"The {b}", a, b)
        for a, b in same_class
    ]
    pairs["coreference"] = [
        ("coreference", f"{a} thanked {b} because {pron_a}", f"{a} thanked {b} because {pron_b}", a, b)
        for (a, b, _verb), (pron_a, pron_b) in zip(entities, [("he", "she"), ("she", "he")] * 8)
    ]
    pairs["quantifier"] = [
        ("quantifier", f"All {noun}s are here", f"Some {noun}s are here", "all", "some")
        for noun in ["cat", "dog", "bird", "student", "teacher", "apple", "car", "book", "flower", "tree"]
    ] + [
        ("quantifier", f"No {noun}s are here", f"Some {noun}s are here", "no", "some")
        for noun in ["cat", "dog", "bird", "student", "teacher", "apple"]
    ]
    pairs["causal"] = [
        ("causal", f"{event_a}, because {event_b}", f"{event_a}, although {event_b}", "because", "although")
        for event_a, event_b in [
            ("The ground was wet", "it rained"),
            ("The glass broke", "it fell"),
            ("The plant grew", "it got water"),
            ("The road flooded", "the river rose"),
            ("The child smiled", "she won"),
            ("The alarm rang", "smoke appeared"),
            ("The team celebrated", "they won"),
            ("The room warmed", "the fire burned"),
        ]
    ]
    pairs["comparison"] = [
        ("comparison", f"The {a} is {x} than the {b}", f"The {a} is {y} than the {b}", x, y)
        for a, b, x, y in comparisons
    ]
    pairs["spatial"] = [
        ("spatial", f"The {a} is {x} the {b}", f"The {a} is {y} the {b}", x, y)
        for a, b, x, y in locations
    ]
    pairs["temporal_order"] = [
        ("temporal_order", f"{a} happened {x} {b}", f"{a} happened {y} {b}", x, y)
        for a, b, x, y in temporal
    ]
    pairs["condition"] = [
        ("condition", f"If {event}, then it changes", f"Unless {event}, it changes", "if", "unless")
        for event in ["it rains", "he works", "she calls", "they agree", "the door opens", "the alarm rings", "the train arrives", "the light turns"]
    ]
    pairs["contrast"] = [
        ("contrast", f"{a}, but {b}", f"{a}, and {b}", "but", "and")
        for a, b in [
            ("It rained", "the ground stayed dry"),
            ("He studied", "he failed"),
            ("She smiled", "she was sad"),
            ("The fire burned", "the room stayed cold"),
            ("The team played well", "they lost"),
            ("The door was open", "nobody entered"),
            ("The road was clear", "traffic stopped"),
            ("The meal was small", "everyone was full"),
        ]
    ]

    expanded: dict[str, list[tuple[str, str, str, str, str]]] = {}
    for rel, rows in pairs.items():
        expanded[rel] = contextualize(rows, max_pairs_per_relation)
    return expanded


def aggregate_relation(layer_rows: dict[int, dict[str, list[float]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_layer": {}}
    all_balance: list[float] = []
    all_ng: list[float] = []
    all_gm: list[float] = []
    all_um: list[float] = []
    all_ia: list[float] = []
    all_net: list[float] = []
    all_gross: list[float] = []
    for li, rows in layer_rows.items():
        if not rows["balance"]:
            continue
        out["per_layer"][str(li)] = {
            "balance_mean": safe_mean(rows["balance"]),
            "balance_std": safe_std(rows["balance"]),
            "net_gross_mean": safe_mean(rows["net_gross"]),
            "net_gross_std": safe_std(rows["net_gross"]),
            "gate_main_abs_mean": safe_mean([abs(x) for x in rows["gate_main"]]),
            "up_main_abs_mean": safe_mean([abs(x) for x in rows["up_main"]]),
            "interaction_abs_mean": safe_mean([abs(x) for x in rows["interaction"]]),
            "net_mean": safe_mean(rows["net"]),
            "gross_mean": safe_mean(rows["gross"]),
            "n": len(rows["balance"]),
        }
        all_balance.extend(rows["balance"])
        all_ng.extend(rows["net_gross"])
        all_gm.extend(rows["gate_main"])
        all_um.extend(rows["up_main"])
        all_ia.extend(rows["interaction"])
        all_net.extend(rows["net"])
        all_gross.extend(rows["gross"])
    total_abs = safe_mean([abs(x) for x in all_gm]) + safe_mean([abs(x) for x in all_um]) + safe_mean([abs(x) for x in all_ia])
    out.update(
        {
            "balance_mean": safe_mean(all_balance),
            "balance_std": safe_std(all_balance),
            "net_gross_mean": safe_mean(all_ng),
            "net_gross_std": safe_std(all_ng),
            "net_mean": safe_mean(all_net),
            "gross_mean": safe_mean(all_gross),
            "gate_main_frac": safe_mean([abs(x) for x in all_gm]) / max(total_abs, 1e-12),
            "up_main_frac": safe_mean([abs(x) for x in all_um]) / max(total_abs, 1e-12),
            "interaction_frac": safe_mean([abs(x) for x in all_ia]) / max(total_abs, 1e-12),
            "n_observations": len(all_ng),
        }
    )
    return out


def relation_signature(rel: dict[str, Any], layers: list[int]) -> list[float]:
    sig = [
        float(rel.get("balance_mean", 0.0)),
        float(rel.get("net_gross_mean", 0.0)),
        float(rel.get("gate_main_frac", 0.0)),
        float(rel.get("up_main_frac", 0.0)),
        float(rel.get("interaction_frac", 0.0)),
    ]
    for li in layers:
        p = rel.get("per_layer", {}).get(str(li), {})
        sig.extend(
            [
                float(p.get("balance_mean", 0.0)),
                float(p.get("net_gross_mean", 0.0)),
                float(p.get("gate_main_abs_mean", 0.0)),
                float(p.get("up_main_abs_mean", 0.0)),
                float(p.get("interaction_abs_mean", 0.0)),
            ]
        )
    return sig


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["PHASE344_ATTN_IMPLEMENTATIONS"] = args.attn_implementations
    model_name = args.model
    log(f"Phase57 Global Relation Path Matrix — {model_name}")
    log("=" * 72)
    t0 = time.time()

    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    target_layers = cfg["binding_layers"]
    W_U = get_W_U(model, model_name)
    layers = get_layers(model)
    log(f"W_U shape={W_U.shape}, target_layers={target_layers}")

    mlp_weights: dict[int, dict[str, Any]] = {}
    for li in target_layers:
        _, _, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_down": W_down, "d_ff": d_ff}
        log(f"Layer {li}: W_down={None if W_down is None else W_down.shape}, d_ff={d_ff}")

    all_pairs = build_relation_pairs(args.max_pairs_per_relation)
    selected_relations = [r for r in args.relations.split(",") if r.strip()]
    selected_relations = [r.strip() for r in selected_relations if r.strip() in all_pairs]
    capture_cache: dict[str, dict[str, np.ndarray]] = {}

    relation_results: dict[str, Any] = {}
    skipped: dict[str, int] = {}
    for rel_idx, rel_name in enumerate(selected_relations, 1):
        rows = all_pairs[rel_name]
        log(f"\n[{rel_idx}/{len(selected_relations)}] relation={rel_name}, pairs={len(rows)}")
        layer_rows = {
            li: {
                "balance": [],
                "net_gross": [],
                "gate_main": [],
                "up_main": [],
                "interaction": [],
                "net": [],
                "gross": [],
            }
            for li in target_layers
        }
        skipped[rel_name] = 0
        valid = 0
        for pidx, (_rtype, clean_prompt, corrupt_prompt, target_word, competitor_word) in enumerate(rows):
            tid_t = get_token_id(tokenizer, target_word)
            tid_c = get_token_id(tokenizer, competitor_word)
            if tid_t is None or tid_c is None:
                skipped[rel_name] += 1
                continue
            direction = W_U[tid_t] - W_U[tid_c]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-10:
                skipped[rel_name] += 1
                continue
            direction = direction / norm

            if clean_prompt not in capture_cache:
                capture_cache[clean_prompt] = capture_mlp_internals(
                    model, tokenizer, device, clean_prompt, target_layers, cfg["n_layers"]
                )
            if corrupt_prompt not in capture_cache:
                capture_cache[corrupt_prompt] = capture_mlp_internals(
                    model, tokenizer, device, corrupt_prompt, target_layers, cfg["n_layers"]
                )
            clean_caps = capture_cache[clean_prompt]
            corrupt_caps = capture_cache[corrupt_prompt]

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
                layer_rows[li]["balance"].append(cd["balance"])
                layer_rows[li]["net_gross"].append(cd["net_gross_ratio"])
                layer_rows[li]["net"].append(cd["net"])
                layer_rows[li]["gross"].append(cd["total_gross"])
                layer_rows[li]["gate_main"].append(it["gate_main"])
                layer_rows[li]["up_main"].append(it["up_main"])
                layer_rows[li]["interaction"].append(it["interaction"])
            valid += 1
            if (pidx + 1) % args.progress_every == 0 or pidx == len(rows) - 1:
                log(f"  {rel_name}: {pidx+1}/{len(rows)} valid={valid} elapsed={time.time()-t0:.0f}s")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        relation_results[rel_name] = aggregate_relation(layer_rows)
        rr = relation_results[rel_name]
        log(
            f"  {rel_name}: balance={rr['balance_mean']:.4f}, "
            f"net/gross={rr['net_gross_mean']:.4f}, interaction={rr['interaction_frac']:.4f}, "
            f"n={rr['n_observations']}"
        )

    signatures = {rel: relation_signature(data, target_layers) for rel, data in relation_results.items()}
    similarity = {
        a: {b: cosine(signatures[a], signatures[b]) for b in relation_results}
        for a in relation_results
    }
    net_gross_rank = sorted(
        ((rel, data.get("net_gross_mean", 0.0)) for rel, data in relation_results.items()),
        key=lambda x: x[1],
        reverse=True,
    )

    result = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementations": args.attn_implementations,
        "target_layers": target_layers,
        "max_pairs_per_relation": args.max_pairs_per_relation,
        "relations": relation_results,
        "skipped": skipped,
        "signatures": signatures,
        "relation_similarity_matrix": similarity,
        "net_gross_rank": [{"relation": rel, "net_gross": val} for rel, val in net_gross_rank],
        "elapsed_sec": time.time() - t0,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase57_global_relation_path_matrix.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Saved {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--output-dir", default=os.environ.get("PHASE57_OUTPUT_DIR", "results/gpt5_phase57_global_relation_path_matrix_full"))
    parser.add_argument("--max-pairs-per-relation", type=int, default=int(os.environ.get("PHASE57_MAX_PAIRS_PER_RELATION", "40")))
    parser.add_argument("--relations", default=",".join(DEFAULT_RELATIONS))
    parser.add_argument("--attn-implementations", default=os.environ.get("PHASE57_ATTN_IMPLEMENTATIONS", "flash_attention_2,sdpa,eager"))
    parser.add_argument("--progress-every", type=int, default=10)
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
