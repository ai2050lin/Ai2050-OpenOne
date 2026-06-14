#!/usr/bin/env python3
"""
Phase 105: CUDA Global Category Atlas.

Builds a full-layer category atlas for broad semantic types. The analysis is
intentionally basic: centroids, layer-wise margins, cohesion, nearest neighbors,
boundary norms, and local logit-lens release maps.

Usage examples:
  python tests/gpt5/phase105_global_category_atlas_cuda.py qwen3 --hard-exit-after-model
  python tests/gpt5/phase105_global_category_atlas_cuda.py glm4 --hard-exit-after-model
  python tests/gpt5/phase105_global_category_atlas_cuda.py deepseek7b --hard-exit-after-model
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results" / "gpt5_phase105_global_category_atlas"


CATEGORY_OBJECTS: dict[str, list[str]] = {
    "fruit": ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "plum", "lemon", "lime", "melon", "cherry", "apricot", "fig", "kiwi", "papaya", "guava", "coconut", "date", "berry", "nectarine", "tangerine", "pomegranate", "persimmon"],
    "animal": ["dog", "cat", "horse", "lion", "bear", "rabbit", "eagle", "fish", "wolf", "fox", "deer", "cow", "sheep", "goat", "pig", "duck", "goose", "shark", "whale", "tiger", "zebra", "monkey", "frog", "snake"],
    "tool": ["hammer", "knife", "wrench", "saw", "drill", "axe", "chisel", "pliers", "screwdriver", "rake", "shovel", "tongs", "clamp", "file", "level", "mallet", "scissors", "needle", "spade", "hoe", "anvil", "vise", "crowbar", "trowel"],
    "vehicle": ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "motorcycle", "tram", "subway", "scooter", "van", "taxi", "ship", "ferry", "helicopter", "tractor", "wagon", "sled", "rocket", "canoe", "kayak", "ambulance", "crane"],
    "clothing": ["shirt", "dress", "hat", "coat", "sock", "glove", "jacket", "scarf", "pants", "skirt", "shoe", "boot", "belt", "tie", "sweater", "hoodie", "robe", "shorts", "uniform", "vest", "cap", "helmet", "apron", "poncho"],
    "furniture": ["chair", "table", "desk", "sofa", "bed", "shelf", "cabinet", "stool", "bench", "dresser", "wardrobe", "couch", "bookcase", "nightstand", "cupboard", "ottoman", "recliner", "crib", "counter", "drawer", "armchair", "locker", "vanity", "sideboard"],
    "food": ["bread", "rice", "cheese", "pasta", "soup", "steak", "salad", "cake", "pizza", "sandwich", "noodle", "egg", "butter", "yogurt", "sausage", "bacon", "cereal", "cookie", "pie", "curry", "dumpling", "tofu", "porridge", "taco"],
    "plant": ["tree", "flower", "grass", "bush", "fern", "cactus", "vine", "shrub", "moss", "algae", "bamboo", "rose", "tulip", "oak", "pine", "maple", "ivy", "weed", "herb", "lily", "orchid", "clover", "palm", "sapling"],
    "body": ["hand", "foot", "arm", "leg", "head", "eye", "ear", "nose", "mouth", "finger", "toe", "knee", "elbow", "shoulder", "back", "neck", "wrist", "ankle", "tooth", "tongue", "heart", "lung", "stomach", "skin"],
    "place": ["city", "village", "forest", "desert", "beach", "mountain", "river", "lake", "school", "hospital", "market", "park", "office", "kitchen", "farm", "airport", "station", "harbor", "museum", "library", "church", "temple", "hotel", "garden"],
    "building": ["house", "apartment", "castle", "tower", "bridge", "barn", "factory", "warehouse", "stadium", "theater", "garage", "mall", "skyscraper", "cabin", "hut", "palace", "shed", "greenhouse", "chapel", "fortress", "lighthouse", "mosque", "synagogue", "courthouse"],
    "material": ["wood", "metal", "stone", "glass", "plastic", "paper", "cloth", "leather", "rubber", "steel", "iron", "copper", "silver", "gold", "clay", "ceramic", "cotton", "wool", "silk", "concrete", "sand", "wax", "carbon", "fiber"],
    "color": ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "brown", "gray", "cyan", "magenta", "violet", "indigo", "beige", "maroon", "navy", "teal", "lime", "golden", "silver", "scarlet", "turquoise"],
    "emotion": ["joy", "sadness", "anger", "fear", "surprise", "disgust", "love", "hate", "shame", "pride", "envy", "grief", "hope", "anxiety", "calm", "delight", "boredom", "trust", "panic", "relief", "guilt", "awe", "loneliness", "curiosity"],
    "role": ["teacher", "doctor", "soldier", "farmer", "driver", "chef", "lawyer", "artist", "student", "nurse", "engineer", "manager", "judge", "pilot", "clerk", "guard", "worker", "parent", "child", "leader", "coach", "priest", "scientist", "merchant"],
    "profession": ["carpenter", "plumber", "electrician", "mechanic", "programmer", "designer", "accountant", "architect", "dentist", "surgeon", "baker", "butcher", "barber", "tailor", "journalist", "photographer", "musician", "actor", "dancer", "writer", "researcher", "analyst", "translator", "librarian"],
    "abstract": ["truth", "justice", "freedom", "beauty", "power", "knowledge", "wisdom", "logic", "meaning", "value", "identity", "order", "chaos", "time", "space", "change", "cause", "effect", "chance", "necessity", "possibility", "difference", "similarity", "structure"],
    "action": ["run", "walk", "jump", "swim", "write", "read", "eat", "drink", "sleep", "think", "build", "break", "carry", "throw", "catch", "open", "close", "push", "pull", "cut", "wash", "cook", "drive", "sing"],
    "event": ["wedding", "funeral", "meeting", "party", "festival", "concert", "game", "war", "election", "trial", "lesson", "exam", "race", "ceremony", "conference", "accident", "birth", "death", "storm", "fire", "flood", "journey", "competition", "debate"],
    "time": ["morning", "noon", "evening", "night", "today", "tomorrow", "yesterday", "week", "month", "year", "spring", "summer", "autumn", "winter", "minute", "hour", "second", "century", "future", "past", "present", "dawn", "dusk", "midnight"],
    "number": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "dozen", "hundred", "thousand", "million", "half", "quarter", "pair", "triple", "zero", "first", "second", "third", "many", "few"],
    "shape": ["circle", "square", "triangle", "rectangle", "sphere", "cube", "cone", "cylinder", "line", "curve", "angle", "point", "edge", "corner", "oval", "ring", "spiral", "arc", "plane", "surface", "pyramid", "prism", "disk", "loop"],
    "sound": ["voice", "music", "noise", "song", "whisper", "shout", "cry", "laugh", "bell", "horn", "drum", "thunder", "echo", "speech", "rhythm", "melody", "silence", "buzz", "hum", "siren", "clap", "crash", "bang", "tone"],
    "light": ["light", "shadow", "sun", "moon", "star", "lamp", "flame", "glow", "flash", "beam", "ray", "spark", "shine", "darkness", "brightness", "color", "reflection", "glare", "laser", "lantern", "candle", "sunrise", "sunset", "radiance"],
    "weather": ["rain", "snow", "wind", "cloud", "storm", "fog", "hail", "thunder", "lightning", "sunshine", "mist", "drizzle", "frost", "heat", "cold", "humidity", "breeze", "tornado", "hurricane", "blizzard", "drought", "monsoon", "climate", "temperature"],
    "container": ["box", "bag", "bottle", "cup", "jar", "basket", "bucket", "can", "barrel", "case", "chest", "drawer", "folder", "envelope", "wallet", "purse", "carton", "crate", "vase", "pot", "pan", "bowl", "plate", "tray"],
    "instrument": ["guitar", "piano", "violin", "drum", "flute", "trumpet", "saxophone", "cello", "harp", "clarinet", "trombone", "accordion", "banjo", "organ", "keyboard", "horn", "recorder", "mandolin", "ukulele", "xylophone", "cymbal", "tambourine", "bassoon", "oboe"],
    "machine": ["engine", "motor", "computer", "robot", "printer", "camera", "phone", "television", "radio", "pump", "turbine", "generator", "compressor", "fan", "oven", "washer", "dryer", "refrigerator", "elevator", "scanner", "router", "server", "drone", "sensor"],
    "communication": ["word", "sentence", "letter", "message", "email", "signal", "sign", "symbol", "code", "language", "speech", "question", "answer", "story", "news", "report", "promise", "command", "request", "warning", "joke", "argument", "rumor", "dialogue"],
    "relation": ["friend", "enemy", "neighbor", "partner", "family", "parent", "sibling", "spouse", "teacher", "student", "owner", "guest", "host", "buyer", "seller", "leader", "follower", "ally", "rival", "colleague", "member", "citizen", "stranger", "companion"],
    "property": ["size", "weight", "speed", "height", "width", "depth", "length", "strength", "weakness", "temperature", "density", "texture", "quality", "quantity", "price", "age", "shape", "color", "smell", "taste", "sound", "motion", "position", "direction"],
    "substance": ["water", "air", "fire", "earth", "oil", "gas", "salt", "sugar", "acid", "alkali", "protein", "fat", "blood", "milk", "honey", "steam", "smoke", "dust", "mud", "ice", "ink", "paint", "soap", "alcohol"],
}


CATEGORY_READOUT_WORDS: dict[str, list[str]] = {
    "fruit": ["fruit", "produce", "berry", "crop"],
    "animal": ["animal", "creature", "beast", "pet"],
    "tool": ["tool", "implement", "device", "instrument"],
    "vehicle": ["vehicle", "transport", "automobile", "car"],
    "clothing": ["clothing", "garment", "attire", "apparel"],
    "furniture": ["furniture", "furnishing", "fixture", "seat"],
    "food": ["food", "meal", "dish", "snack"],
    "plant": ["plant", "tree", "vegetation", "flora"],
    "body": ["body", "limb", "organ", "flesh"],
    "place": ["place", "location", "site", "area"],
    "building": ["building", "structure", "house", "facility"],
    "material": ["material", "substance", "matter", "fabric"],
    "color": ["color", "hue", "shade", "tone"],
    "emotion": ["emotion", "feeling", "mood", "affect"],
    "role": ["role", "person", "title", "status"],
    "profession": ["profession", "occupation", "career", "job"],
    "abstract": ["concept", "idea", "principle", "abstraction"],
    "action": ["action", "act", "verb", "movement"],
    "event": ["event", "occasion", "happening", "incident"],
    "time": ["time", "date", "period", "moment"],
    "number": ["number", "amount", "quantity", "count"],
    "shape": ["shape", "form", "geometry", "figure"],
    "sound": ["sound", "noise", "voice", "audio"],
    "light": ["light", "brightness", "glow", "illumination"],
    "weather": ["weather", "climate", "storm", "rain"],
    "container": ["container", "vessel", "box", "holder"],
    "instrument": ["instrument", "music", "device", "tool"],
    "machine": ["machine", "engine", "mechanism", "device"],
    "communication": ["communication", "language", "message", "speech"],
    "relation": ["relation", "relationship", "connection", "association"],
    "property": ["property", "attribute", "feature", "quality"],
    "substance": ["substance", "matter", "chemical", "material"],
}


PROMPT_TEMPLATES = [
    "The {obj} is a kind of",
    "A {obj} belongs to the category of",
    "The word {obj} refers to a type of",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def token_candidates(word: str) -> list[str]:
    return [word, " " + word, word.lower(), " " + word.lower()]


def find_token_id(tokenizer: Any, word: str) -> int | None:
    vocab = tokenizer.get_vocab()
    for cand in token_candidates(word):
        if cand in vocab:
            return int(vocab[cand])
    ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    if ids:
        return int(ids[0])
    return None


def collect_readout_rows(model: Any, tokenizer: Any, categories: list[str]) -> tuple[dict[str, list[int]], np.ndarray, list[str]]:
    emb = model.get_output_embeddings()
    if emb is None and hasattr(model, "lm_head"):
        emb = model.lm_head
    if emb is None:
        raise RuntimeError("Cannot locate output embedding/lm_head")
    weight = emb.weight.detach().float().cpu()

    cat_token_ids: dict[str, list[int]] = {}
    unique_ids: list[int] = []
    unique_seen = set()
    for cat in categories:
        ids = []
        for word in CATEGORY_READOUT_WORDS[cat]:
            tid = find_token_id(tokenizer, word)
            if tid is None or tid >= weight.shape[0]:
                continue
            ids.append(tid)
            if tid not in unique_seen:
                unique_seen.add(tid)
                unique_ids.append(tid)
        cat_token_ids[cat] = ids

    rows = weight[unique_ids].numpy()
    token_labels = [tokenizer.decode([tid]) for tid in unique_ids]
    id_to_local = {tid: i for i, tid in enumerate(unique_ids)}
    cat_local_ids = {cat: [id_to_local[tid] for tid in ids if tid in id_to_local] for cat, ids in cat_token_ids.items()}
    return cat_local_ids, rows, token_labels


def category_scores(vecs: np.ndarray, readout_rows: np.ndarray, cat_local_ids: dict[str, list[int]], categories: list[str]) -> np.ndarray:
    logits = vecs @ readout_rows.T
    scores = np.zeros((vecs.shape[0], len(categories)), dtype=np.float32)
    for ci, cat in enumerate(categories):
        ids = cat_local_ids.get(cat, [])
        if ids:
            scores[:, ci] = logits[:, ids].mean(axis=1)
    return scores


def summarize_curve(xs: list[float]) -> dict[str, Any]:
    arr = np.array(xs, dtype=np.float64)
    if arr.size == 0:
        return {"max_layer": None, "max": None, "mean": None}
    return {
        "max_layer": int(np.argmax(arr)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "final": float(arr[-1]),
    }


def classify_category(best_margin: float, best_cohesion: float, best_boundary_norm: float, release_count: int) -> str:
    if best_margin > 8 and best_cohesion > 0.65:
        return "sharp_readout_cohesive"
    if best_margin > 4:
        return "readout_clear"
    if best_cohesion > 0.65 and best_boundary_norm > 5:
        return "cohesive_boundary_unclear_readout"
    if release_count >= 5:
        return "competitive_broad"
    return "diffuse_or_contextual"


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this phase.")

    categories = list(CATEGORY_OBJECTS.keys())
    if args.max_categories:
        categories = categories[: args.max_categories]
    n_cat = len(categories)
    n_obj = min(args.objects_per_category, min(len(CATEGORY_OBJECTS[c]) for c in categories))
    templates = PROMPT_TEMPLATES[: args.templates]
    log(f"Loading {args.model} on CUDA; categories={n_cat}, objects/category={n_obj}, templates={len(templates)}")
    loaded = load_probe_model(args.model)
    model = loaded.model
    tokenizer = loaded.tokenizer
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = int(model.get_input_embeddings().weight.shape[1])
    alloc, reserved = vram_gb()
    log(f"Loaded {args.model}: layers={n_layers}, d_model={d_model}, vram={alloc:.2f}/{reserved:.2f}GB")

    cat_local_ids, readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
    readout_rows = readout_rows.astype(np.float32)
    log(f"Readout rows: {readout_rows.shape[0]} token rows for {n_cat} categories")

    sums = np.zeros((n_layers + 1, n_cat, d_model), dtype=np.float64)
    counts = np.zeros((n_cat,), dtype=np.int64)
    object_states: list[list[list[np.ndarray]]] = [
        [[] for _ in range(n_cat)] for _ in range(n_layers + 1)
    ]

    all_items: list[tuple[int, str, str]] = []
    for ci, cat in enumerate(categories):
        for obj in CATEGORY_OBJECTS[cat][:n_obj]:
            for template in templates:
                all_items.append((ci, cat, template.format(obj=obj)))

    batch_size = args.batch_size
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(all_items), batch_size):
            batch_items = all_items[start : start + batch_size]
            prompts = [x[2] for x in batch_items]
            batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
            batch = {k: v.to(loaded.input_device) for k, v in batch.items()}
            out = model(**batch, output_hidden_states=True, use_cache=False)
            pos = batch["attention_mask"].sum(dim=1) - 1
            for li in range(n_layers + 1):
                hs = out.hidden_states[li]
                picked = hs[torch.arange(hs.shape[0], device=hs.device), pos].detach().float().cpu().numpy()
                for bi, (ci, _cat, _prompt) in enumerate(batch_items):
                    vec = picked[bi].astype(np.float32)
                    sums[li, ci] += vec
                    object_states[li][ci].append(vec)
            for ci, _cat, _prompt in batch_items:
                counts[ci] += 1
            if (start // batch_size) % args.progress_every == 0:
                alloc, reserved = vram_gb()
                log(f"  batch {start + len(batch_items)}/{len(all_items)} vram={alloc:.2f}/{reserved:.2f}GB")
            del out, batch
            torch.cuda.empty_cache()

    centers = (sums / counts.reshape(1, n_cat, 1)).astype(np.float32)
    log(f"Captured all layers in {(time.time() - t0) / 60:.1f} min")

    layer_metrics: dict[str, Any] = {}
    category_summary: dict[str, Any] = {}
    layer_global: list[dict[str, Any]] = []

    for li in range(n_layers + 1):
        C = centers[li]
        norms = np.linalg.norm(C, axis=1) + 1e-8
        cos_mat = (C @ C.T) / (norms[:, None] * norms[None, :])
        scores = category_scores(C, readout_rows, cat_local_ids, categories)
        target_scores = np.diag(scores)
        other_scores = scores.copy()
        np.fill_diagonal(other_scores, -1e9)
        margins = target_scores - other_scores.max(axis=1)
        ranks = []
        cohesion = []
        boundary_norms = []
        for ci in range(n_cat):
            ranks.append(int(1 + np.sum(scores[ci] > scores[ci, ci])))
            objs = object_states[li][ci]
            c = C[ci]
            c_norm = np.linalg.norm(c) + 1e-8
            if objs:
                co = [float(np.dot(v, c) / ((np.linalg.norm(v) + 1e-8) * c_norm)) for v in objs]
                cohesion.append(float(np.mean(co)))
            else:
                cohesion.append(0.0)
            other_mean = np.mean(np.delete(C, ci, axis=0), axis=0)
            boundary_norms.append(float(np.linalg.norm(c - other_mean)))
        layer_metrics[str(li)] = {
            "mean_target_margin": float(np.mean(margins)),
            "median_target_margin": float(np.median(margins)),
            "top1_count": int(sum(1 for r in ranks if r == 1)),
            "mean_cohesion": float(np.mean(cohesion)),
            "mean_boundary_norm": float(np.mean(boundary_norms)),
        }
        layer_global.append({"layer": li, **layer_metrics[str(li)]})

    for ci, cat in enumerate(categories):
        margin_curve = []
        rank_curve = []
        cohesion_curve = []
        boundary_curve = []
        top_neighbors_by_best_layer: list[dict[str, Any]] = []
        best_layer_by_margin = 0

        for li in range(n_layers + 1):
            C = centers[li]
            scores = category_scores(C, readout_rows, cat_local_ids, categories)
            own = float(scores[ci, ci])
            others = np.delete(scores[ci], ci)
            margin = own - float(np.max(others))
            rank = int(1 + np.sum(scores[ci] > scores[ci, ci]))
            objs = object_states[li][ci]
            c = C[ci]
            c_norm = np.linalg.norm(c) + 1e-8
            co = [float(np.dot(v, c) / ((np.linalg.norm(v) + 1e-8) * c_norm)) for v in objs]
            other_mean = np.mean(np.delete(C, ci, axis=0), axis=0)
            margin_curve.append(margin)
            rank_curve.append(rank)
            cohesion_curve.append(float(np.mean(co)) if co else 0.0)
            boundary_curve.append(float(np.linalg.norm(c - other_mean)))

        best_layer_by_margin = int(np.argmax(np.array(margin_curve)))
        best_layer_by_boundary = int(np.argmax(np.array(boundary_curve)))
        best_layer_by_cohesion = int(np.argmax(np.array(cohesion_curve)))

        C_best = centers[best_layer_by_margin]
        norms = np.linalg.norm(C_best, axis=1) + 1e-8
        cos_row = (C_best[ci] @ C_best.T) / ((np.linalg.norm(C_best[ci]) + 1e-8) * norms)
        neighbor_ids = [j for j in np.argsort(cos_row)[::-1] if j != ci][:6]
        top_neighbors_by_best_layer = [
            {"category": categories[j], "cos": float(cos_row[j])} for j in neighbor_ids
        ]

        other_mean = np.mean(np.delete(C_best, ci, axis=0), axis=0)
        boundary = C_best[ci] - other_mean
        bhat = boundary / (np.linalg.norm(boundary) + 1e-8)
        proj = float(np.dot(C_best[ci], bhat))
        removed = C_best[ci] - proj * bhat
        before = category_scores(C_best[ci : ci + 1], readout_rows, cat_local_ids, categories)[0]
        after = category_scores(removed.reshape(1, -1), readout_rows, cat_local_ids, categories)[0]
        delta = after - before
        releases = []
        for j, other in enumerate(categories):
            if j != ci and delta[j] > 0:
                releases.append({"category": other, "delta": float(delta[j])})
        releases.sort(key=lambda x: x["delta"], reverse=True)

        category_summary[cat] = {
            "n_samples": int(counts[ci]),
            "best_layer_by_margin": best_layer_by_margin,
            "best_layer_by_boundary_norm": best_layer_by_boundary,
            "best_layer_by_cohesion": best_layer_by_cohesion,
            "margin_curve_summary": summarize_curve(margin_curve),
            "cohesion_curve_summary": summarize_curve(cohesion_curve),
            "boundary_norm_curve_summary": summarize_curve(boundary_curve),
            "best_margin": float(max(margin_curve)),
            "final_margin": float(margin_curve[-1]),
            "best_rank": int(min(rank_curve)),
            "final_rank": int(rank_curve[-1]),
            "best_cohesion": float(max(cohesion_curve)),
            "best_boundary_norm": float(max(boundary_curve)),
            "nearest_neighbors_at_best_margin_layer": top_neighbors_by_best_layer,
            "local_boundary_remove_target_delta": float(delta[ci]),
            "local_boundary_release_top": releases[:8],
            "type_class": classify_category(max(margin_curve), max(cohesion_curve), max(boundary_curve), len(releases)),
        }

    atlas = {
        "phase": 105,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cuda": {
            "device_name": torch.cuda.get_device_name(0),
            "torch_dtype_env": os.environ.get("PROBE_TORCH_DTYPE", ""),
            "attn_impl_env": os.environ.get("PROBE_ATTN_IMPLEMENTATION", ""),
            "batch_size": batch_size,
        },
        "n_layers": n_layers,
        "d_model": d_model,
        "categories": categories,
        "objects_per_category": n_obj,
        "templates": templates,
        "readout_token_labels": token_labels,
        "layer_global": layer_global,
        "category_summary": category_summary,
        "notes": [
            "Layer 0 is embedding output; layer k>0 is hidden_states[k], i.e. after transformer block k-1.",
            "Boundary removal here is local logit-lens removal at the best margin layer, not downstream causal patching.",
            "Metrics are basic centroid/readout/neighbor measurements, not statistical proof.",
        ],
    }
    return atlas


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = []
    lines.append(f"# Phase 105 Global Category Atlas: {result['model']}")
    lines.append("")
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    best_global = max(result["layer_global"], key=lambda x: x["top1_count"])
    lines.append("## Global Layer Shape")
    lines.append(
        f"- Best top1 layer: L{best_global['layer']} with {best_global['top1_count']} / "
        f"{len(result['categories'])} categories top1."
    )
    margin_best = max(result["layer_global"], key=lambda x: x["mean_target_margin"])
    boundary_best = max(result["layer_global"], key=lambda x: x["mean_boundary_norm"])
    lines.append(f"- Best mean margin layer: L{margin_best['layer']} margin={margin_best['mean_target_margin']:.3f}.")
    lines.append(f"- Best mean boundary layer: L{boundary_best['layer']} norm={boundary_best['mean_boundary_norm']:.3f}.")
    lines.append("")
    lines.append("## Category Layer Map")
    for cat, item in result["category_summary"].items():
        neigh = ", ".join(f"{x['category']}({x['cos']:.2f})" for x in item["nearest_neighbors_at_best_margin_layer"][:3])
        rel = ", ".join(f"{x['category']}+{x['delta']:.2f}" for x in item["local_boundary_release_top"][:3]) or "none"
        lines.append(
            f"- {cat}: class={item['type_class']}, marginL={item['best_layer_by_margin']}, "
            f"boundaryL={item['best_layer_by_boundary_norm']}, cohesionL={item['best_layer_by_cohesion']}, "
            f"best_margin={item['best_margin']:.2f}, best_rank={item['best_rank']}, "
            f"neighbors={neigh}, local_release={rel}"
        )
    lines.append("")
    lines.append("## Caution")
    for note in result["notes"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--objects-per-category", type=int, default=24)
    parser.add_argument("--templates", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-categories", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = None
    try:
        result = run_model(args)
        json_path = out_dir / f"phase105_{args.model}_atlas.json"
        md_path = out_dir / f"phase105_{args.model}_atlas.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(result, md_path)
        log(f"Wrote {json_path}")
        log(f"Wrote {md_path}")
    finally:
        release_loaded(loaded)
        if args.hard_exit_after_model:
            os._exit(0)


if __name__ == "__main__":
    main()
