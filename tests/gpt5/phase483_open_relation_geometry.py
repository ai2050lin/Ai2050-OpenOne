#!/usr/bin/env python3
"""Phase483 open relation geometry collection and online distance ledger.

Runs only open splits from Phase482. It never reads sealed_physical_holdout.
Two-pass design:
  1. Welford standardization on geometry_window_freeze only.
  2. Project open splits with frozen Rademacher matrix and compute paired
     distances plus frozen-window summaries.

No causal intervention, no head/channel/neuron scan, no raw hidden-state save.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase451_glm4_v2_pilot_behavior import prompt_for, write_jsonl  # noqa: E402


PHASE482_DIR = ROOT / "tests" / "gpt5" / "result" / "phase482_relation_geometry_protocol"
SAMPLES_PATH = PHASE482_DIR / "phase482_relation_geometry_samples.jsonl"
PROTOCOL_PATH = PHASE482_DIR / "phase482_relation_geometry_protocol.json"
AUDIT_PATH = PHASE482_DIR / "phase482_relation_geometry_static_audit.json"

OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase483_open_relation_geometry"
PROTOCOL_OUT = OUT_DIR / "phase483_open_relation_geometry_run_protocol.json"
NORM_PATH = OUT_DIR / "phase483_open_relation_geometry_norm_stats.npz"
DIST_ROWS_PATH = OUT_DIR / "phase483_open_relation_geometry_distance_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase483_open_relation_geometry_summary.json"

OPEN_SPLITS = ("geometry_window_freeze", "physical_prediction_holdout")
STD_SPLIT = "geometry_window_freeze"
HOLDOUT_SPLIT = "physical_prediction_holdout"
SEALED_SPLIT = "sealed_physical_holdout"
ROLES = ("evidence_block_end", "claim_end", "label_instruction_end", "terminal_token")
LAYER_FAMILIES = {
    "early": range(0, 9),
    "mid_front": range(9, 21),
    "mid_back": range(21, 33),
    "late": range(33, 40),
    "final": range(40, 41),
}
EPS = 1e-6


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    if not needle:
        return None
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start:start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def token_span(tokenizer: Any, full_ids: list[int], text: str) -> tuple[int, int]:
    for candidate in (text, " " + text, text + " ", text + "\n", " " + text + "\n"):
        found = find_subsequence(full_ids, tokenizer(candidate, add_special_tokens=False)["input_ids"])
        if found is not None:
            return found
    raise RuntimeError(f"Could not locate text span: {text[:80]}")


def locate_positions(tokenizer: Any, prompt: str, sample: dict[str, Any], variant: dict[str, Any]) -> dict[str, int]:
    full_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    fact_text = " ".join(sample["facts"])
    instruction = "Map: true=A; false=B." if sample["label_mapping"] == "mu_ab" else "Map: true=B; false=A."
    spans = {
        "evidence_block_end": token_span(tokenizer, full_ids, fact_text),
        "claim_end": token_span(tokenizer, full_ids, sample["claim"]),
        "label_instruction_end": token_span(tokenizer, full_ids, instruction),
    }
    out = {role: end - 1 for role, (_start, end) in spans.items()}
    out["terminal_token"] = len(full_ids) - 1
    return out


def iter_open_variants(samples: list[dict[str, Any]], split_filter: set[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if sample["sealed"] or sample["split"] == SEALED_SPLIT:
            continue
        if split_filter is not None and sample["split"] not in split_filter:
            continue
        for variant in sample["surface_variants"]:
            rows.append({"sample": sample, "variant": variant})
    return rows


class Welford:
    def __init__(self, n_layers: int, n_roles: int, d_model: int) -> None:
        self.n = np.zeros((n_layers, n_roles), dtype=np.int64)
        self.mean = np.zeros((n_layers, n_roles, d_model), dtype=np.float64)
        self.m2 = np.zeros((n_layers, n_roles, d_model), dtype=np.float64)

    def update(self, layer: int, role: int, vec: np.ndarray) -> None:
        self.n[layer, role] += 1
        n = self.n[layer, role]
        delta = vec - self.mean[layer, role]
        self.mean[layer, role] += delta / n
        delta2 = vec - self.mean[layer, role]
        self.m2[layer, role] += delta * delta2

    def std(self) -> np.ndarray:
        denom = np.maximum(self.n[:, :, None] - 1, 1)
        var = self.m2 / denom
        return np.sqrt(np.maximum(var, EPS)).astype(np.float32)


def frozen_projection(protocol: dict[str, Any], d_model: int) -> np.ndarray:
    cfg = protocol["projection"]
    rng = np.random.default_rng(int(cfg["seed"]))
    k = int(cfg["dimension_k"])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(k, d_model))
    return signs / np.sqrt(k)


def family_for(layer: int) -> str:
    for name, layers in LAYER_FAMILIES.items():
        if layer in layers:
            return name
    raise ValueError(layer)


def first_pass_norm(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]], n_layers: int, d_model: int) -> Welford:
    stats = Welford(n_layers, len(ROLES), d_model)
    for idx, row in enumerate(rows, start=1):
        sample = row["sample"]
        variant = row["variant"]
        prompt = prompt_for(variant["text"])
        positions = locate_positions(tokenizer, prompt, sample, variant)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        for layer_index, hidden in enumerate(outputs.hidden_states):
            h = hidden[0]
            for role_index, role in enumerate(ROLES):
                stats.update(layer_index, role_index, h[positions[role]].detach().float().cpu().numpy())
        if idx % 96 == 0:
            print(f"[phase483:norm] {idx}/{len(rows)}", flush=True)
    return stats


def second_pass_project(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    mean_arr: np.ndarray,
    std_arr: np.ndarray,
    proj: np.ndarray,
) -> dict[tuple[Any, ...], np.ndarray]:
    vectors: dict[tuple[Any, ...], np.ndarray] = {}
    for idx, row in enumerate(rows, start=1):
        sample = row["sample"]
        variant = row["variant"]
        prompt = prompt_for(variant["text"])
        positions = locate_positions(tokenizer, prompt, sample, variant)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        for layer_index, hidden in enumerate(outputs.hidden_states):
            h = hidden[0]
            for role_index, role in enumerate(ROLES):
                raw = h[positions[role]].detach().float().cpu().numpy()
                z = (raw - mean_arr[layer_index, role_index]) / (std_arr[layer_index, role_index] + EPS)
                vec = proj @ z.astype(np.float32)
                key = (
                    sample["split"],
                    sample["source_pair_id"],
                    sample["pair_role"],
                    bool(sample["truth_value"]),
                    sample["label_mapping"],
                    sample["subprotocol"],
                    variant["template"],
                    role,
                    layer_index,
                )
                vectors[key] = vec.astype(np.float32)
        if idx % 96 == 0:
            print(f"[phase483:project] {idx}/{len(rows)}", flush=True)
    return vectors


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    return float(1.0 - float(np.dot(a, b)) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def add_distances(vectors: dict[tuple[Any, ...], np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    splits = sorted({key[0] for key in vectors})
    pair_ids = sorted({key[1] for key in vectors})
    mappings = ("mu_ab", "mu_ba")
    subprotocols = ("label_post_relation_geometry", "label_pre_mapping_visible_control")
    templates = ("records_claim", "claim_records")
    roles = ROLES
    layers = sorted({key[8] for key in vectors})
    distance_fns = {"cosine": cosine_distance, "euclidean": euclidean_distance}
    for split in splits:
        split_pair_ids = sorted({key[1] for key in vectors if key[0] == split})
        for pair_id in split_pair_ids:
            for mapping in mappings:
                for subprotocol in subprotocols:
                    for role in roles:
                        for layer in layers:
                            for distance_name, distance_fn in distance_fns.items():
                                for template in templates:
                                    true_key = (split, pair_id, "base_true", True, mapping, subprotocol, template, role, layer)
                                    false_key = (split, pair_id, "counterfactual_false", False, mapping, subprotocol, template, role, layer)
                                    if true_key in vectors and false_key in vectors:
                                        rows.append({
                                            "split": split,
                                            "source_pair_id": pair_id,
                                            "label_mapping": mapping,
                                            "subprotocol": subprotocol,
                                            "template": template,
                                            "role": role,
                                            "layer_index": layer,
                                            "layer_family": family_for(layer),
                                            "distance": distance_name,
                                            "metric": "d_cf",
                                            "value": distance_fn(vectors[true_key], vectors[false_key]),
                                        })
                                for pair_role, truth in (("base_true", True), ("counterfactual_false", False)):
                                    left = (split, pair_id, pair_role, truth, mapping, subprotocol, "records_claim", role, layer)
                                    right = (split, pair_id, pair_role, truth, mapping, subprotocol, "claim_records", role, layer)
                                    if left in vectors and right in vectors:
                                        rows.append({
                                            "split": split,
                                            "source_pair_id": pair_id,
                                            "label_mapping": mapping,
                                            "subprotocol": subprotocol,
                                            "template": "template_pair",
                                            "role": role,
                                            "layer_index": layer,
                                            "layer_family": family_for(layer),
                                            "distance": distance_name,
                                            "metric": "d_surface",
                                            "value": distance_fn(vectors[left], vectors[right]),
                                        })
                                if subprotocol == "label_pre_mapping_visible_control":
                                    for pair_role, truth in (("base_true", True), ("counterfactual_false", False)):
                                        for template in templates:
                                            ab = (split, pair_id, pair_role, truth, "mu_ab", subprotocol, template, role, layer)
                                            ba = (split, pair_id, pair_role, truth, "mu_ba", subprotocol, template, role, layer)
                                            if ab in vectors and ba in vectors:
                                                rows.append({
                                                    "split": split,
                                                    "source_pair_id": pair_id,
                                                    "label_mapping": "mapping_pair",
                                                    "subprotocol": subprotocol,
                                                    "template": template,
                                                    "role": role,
                                                    "layer_index": layer,
                                                    "layer_family": family_for(layer),
                                                    "distance": distance_name,
                                                    "metric": "d_mu",
                                                    "value": distance_fn(vectors[ab], vectors[ba]),
                                                })
    return rows


def summarize_distance_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["split"], row["subprotocol"], row["role"], row["layer_family"], row["distance"], row["metric"])].append(row)
    grouped = []
    for key, items in sorted(buckets.items()):
        vals = [float(item["value"]) for item in items]
        grouped.append({
            "split": key[0],
            "subprotocol": key[1],
            "role": key[2],
            "layer_family": key[3],
            "distance": key[4],
            "metric": key[5],
            "n": len(vals),
            "mean": mean(vals),
            "median": float(np.median(vals)),
        })
    quality = []
    index = {(g["split"], g["subprotocol"], g["role"], g["layer_family"], g["distance"], g["metric"]): g for g in grouped}
    for key, dcf in index.items():
        if key[-1] != "d_cf":
            continue
        ds = index.get((*key[:-1], "d_surface"))
        if ds is None:
            continue
        q = (dcf["mean"] - ds["mean"]) / (dcf["mean"] + ds["mean"] + EPS)
        quality.append({
            "split": key[0],
            "subprotocol": key[1],
            "role": key[2],
            "layer_family": key[3],
            "distance": key[4],
            "mean_d_cf": dcf["mean"],
            "mean_d_surface": ds["mean"],
            "q_r": q,
            "positive": q > 0,
        })
    return {
        "grouped_metrics": grouped,
        "quality": quality,
        "quality_extrema": [
            {
                "split": split,
                "raw_max_q": max(q["q_r"] for q in split_items),
                "raw_min_q": min(q["q_r"] for q in split_items),
                "positive_q_ceiling": max(0.0, max(q["q_r"] for q in split_items)),
                "positive_window_count": sum(1 for q in split_items if q["q_r"] > 0),
                "window_count": len(split_items),
            }
            for split in sorted({q["split"] for q in quality})
            for split_items in [[q for q in quality if q["split"] == split]]
            if split_items
        ],
        "top_quality_geometry_freeze": sorted(
            [q for q in quality if q["split"] == STD_SPLIT],
            key=lambda x: x["q_r"],
            reverse=True,
        )[:12],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--max-open-variants", type=int, default=0, help="Debug only; 0 means full frozen open splits.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit["status"] != "static_pass_no_model_run" or not audit["count_contract"]["pass"]:
        raise RuntimeError("Phase482 static/count contract is not closed; refusing model run.")
    if audit["static_baselines"]["reports"]["truth_plus_mapping_oracle"]["accuracy"] != 1.0:
        raise RuntimeError("truth_plus_mapping_oracle is not 1.0; refusing model run.")
    samples = load_jsonl(SAMPLES_PATH)
    if any(row["sealed"] and row["split"] == SEALED_SPLIT for row in samples if row["split"] in OPEN_SPLITS):
        raise RuntimeError("Sealed/open split contamination detected.")
    norm_rows = iter_open_variants(samples, {STD_SPLIT})
    open_rows = iter_open_variants(samples, set(OPEN_SPLITS))
    if args.max_open_variants:
        norm_rows = norm_rows[: min(len(norm_rows), args.max_open_variants)]
        open_rows = open_rows[: min(len(open_rows), args.max_open_variants)]
    run_protocol = {
        "schema_version": "phase483_open_relation_geometry_run_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running_open_splits_only",
        "phase482_protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "phase482_static_audit": str(AUDIT_PATH.relative_to(ROOT)),
        "sealed_split_read": False,
        "norm_split": STD_SPLIT,
        "open_splits": list(OPEN_SPLITS),
        "norm_variant_records": len(norm_rows),
        "open_variant_records": len(open_rows),
        "projection": protocol["projection"],
        "roles": list(ROLES),
        "raw_hidden_state_saved": False,
        "debug_max_open_variants": args.max_open_variants,
    }
    PROTOCOL_OUT.write_text(json.dumps(run_protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model, tokenizer, device = load_model("glm4", use_8bit=args.use_8bit)
    try:
        d_model = int(model.get_input_embeddings().weight.shape[1])
        n_layers = int(model.config.num_hidden_layers) + 1
        proj = frozen_projection(protocol, d_model)
        stats = first_pass_norm(model, tokenizer, device, norm_rows, n_layers, d_model)
        mean_arr = stats.mean.astype(np.float32)
        std_arr = stats.std()
        np.savez_compressed(
            NORM_PATH,
            mean=mean_arr,
            std=std_arr,
            n=stats.n,
            projection_seed=np.array([protocol["projection"]["seed"]], dtype=np.int64),
            projection_dim=np.array([protocol["projection"]["dimension_k"]], dtype=np.int64),
        )
        vectors = second_pass_project(model, tokenizer, device, open_rows, mean_arr, std_arr, proj)
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    dist_rows = add_distances(vectors)
    write_jsonl(DIST_ROWS_PATH, dist_rows)
    summary = {
        "schema_version": "phase483_open_relation_geometry_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "open_relation_geometry_complete",
        "sealed_split_read": False,
        "norm_variant_records": len(norm_rows),
        "open_variant_records": len(open_rows),
        "projected_vector_count_in_memory": len(vectors),
        "distance_row_count": len(dist_rows),
        "raw_hidden_state_saved": False,
        **summarize_distance_rows(dist_rows),
        "interpretation": {
            "allowed_claim": "Open-split projected relation geometry distances can be compared by role/layer family.",
            "forbidden_claim": "No sealed validation, causal effect, head/channel/neuron attribution, or mechanism closure is authorized.",
            "next_step": "Freeze windows on geometry_window_freeze only, then evaluate physical_prediction_holdout without reselecting.",
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(DIST_ROWS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
