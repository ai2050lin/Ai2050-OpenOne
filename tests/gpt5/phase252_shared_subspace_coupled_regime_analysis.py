#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase228_module_tree_gateup_causal_validation as p228  # noqa: E402
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402
import phase248_regime_level_direction_bank as p248  # noqa: E402
import phase251_orthogonalized_natural_direction_causal_validation as p251  # noqa: E402


PHASE = 252
SOURCE_PHASE = 251
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
PHASE246_DIR = Path("tests/result/phase246_focused_causal_validation/focused_causal_validation")
PHASE250_DIR = Path("tests/result/phase250_natural_regime_direction_extraction/natural_regime_direction_extraction")
PHASE251_DIR = Path("tests/result/phase251_orthogonalized_natural_direction_causal_validation/orthogonalized_natural_direction_causal_validation")
RESULT_ROOT = Path("tests/result/phase252_shared_subspace_coupled_regime_analysis")
ROUND_DEFAULT = "shared_subspace_coupled_regime_analysis"

SPECS = {
    "qwen3": {"final_observe_layer": 33},
    "glm4": {"final_observe_layer": 32},
    "deepseek7b": {"final_observe_layer": 27},
}

TOKENBANK_CORE = ["continuation_regime", "protocol_short_regime", "answer_boundary_regime", "period_stop_regime", "because_reason_regime"]
NATURAL_CORE = [
    "natural_protocol_short",
    "natural_continuation_explain",
    "natural_answer_boundary",
    "natural_target_seed",
    "natural_concise_answer",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_unique_jsonl(path: Path, rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def norm(vec: torch.Tensor | None) -> float:
    if vec is None or not torch.is_tensor(vec):
        return 0.0
    return float(torch.linalg.vector_norm(vec.float()).item())


def unit(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    n = torch.linalg.vector_norm(vec).item()
    if n <= 1e-8:
        return torch.zeros_like(vec)
    return vec / n


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel() or norm(a) <= 1e-8 or norm(b) <= 1e-8:
        return 0.0
    return float(F.cosine_similarity(a.float(), b.float(), dim=0).item())


def normalize(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    n = torch.linalg.vector_norm(vec).item()
    if n <= 1e-8:
        return torch.zeros_like(vec)
    return vec / n * float(target_norm)


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("model")), str(row.get("case_id")), str(row.get("variant_id")))


def load_behavior_by_key() -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl")
    return {(str(x["model"]), str(x["case_id"]), str(x["variant_id"])): x for x in rows}


def load_raw_vectors(model_name: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    manifest = read_json(PHASE246_DIR / "phase246_raw_delta_vector_manifest.json").get("rows", [])
    out = {}
    for item in manifest:
        if str(item.get("model")) != model_name:
            continue
        path = Path(str(item.get("path")))
        if path.exists():
            out[key_for(item)] = {**item, "payload": torch.load(path, map_location="cpu")}
    return out


def load_natural_directions(model_name: str) -> dict[str, torch.Tensor]:
    rows = read_jsonl(PHASE250_DIR / "phase250_natural_direction_rows.jsonl")
    out = {}
    for row in rows:
        if str(row.get("model")) != model_name or str(row.get("scope")) != "global":
            continue
        path = Path(str(row.get("path")))
        if path.exists():
            out[str(row["contrast_id"])] = unit(torch.load(path, map_location="cpu")["direction"])
    return out


def build_tokenbank_directions(model: Any, tokenizer: Any) -> dict[str, torch.Tensor]:
    output_weight = model.get_output_embeddings().weight.detach().float().cpu()
    out = {}
    for regime, texts in p248.REGIME_TEXTS.items():
        token_ids = p248.token_ids_for_texts(tokenizer, texts)
        if regime == "period_stop_regime" and tokenizer.eos_token_id is not None:
            token_ids = sorted(set(token_ids + [int(tokenizer.eos_token_id)]))
        vectors = [unit(output_weight[token_id]) for token_id in token_ids if 0 <= token_id < output_weight.shape[0]]
        out[regime] = unit(torch.stack(vectors).mean(dim=0)) if vectors else torch.zeros(output_weight.shape[1])
    return out


def qr_basis(vectors: list[torch.Tensor]) -> torch.Tensor:
    vectors = [unit(v) for v in vectors if torch.is_tensor(v) and norm(v) > 1e-8]
    if not vectors:
        return torch.zeros(0, 0)
    mat = torch.stack(vectors, dim=1).float()
    q, _r = torch.linalg.qr(mat, mode="reduced")
    return q.cpu()


def subspace_overlap(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    qa = qr_basis(a)
    qb = qr_basis(b)
    if qa.numel() == 0 or qb.numel() == 0:
        return 0.0
    m = qa.T @ qb
    denom = max(1, min(qa.shape[1], qb.shape[1]))
    return float(torch.linalg.matrix_norm(m, ord="fro").item() ** 2 / denom)


def pca_rows(model_name: str, directions: dict[str, torch.Tensor], raw_vectors: dict[tuple[str, str, str], dict[str, Any]], comparisons: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    names = [name for name in directions if norm(directions[name]) > 1e-8]
    if len(names) < 2:
        return [], []
    mat = torch.stack([unit(directions[name]) for name in names], dim=0)
    centered = mat - mat.mean(dim=0, keepdim=True)
    _u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    total = float((s * s).sum().item()) or 1.0
    now = utc_now()
    subspace_rows = []
    for idx in range(min(5, vh.shape[0])):
        subspace_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase252",
                "created_at": now,
                "subspace_id": f"phase252:shared:{model_name}:pc{idx + 1}",
                "model": model_name,
                "component_index": idx + 1,
                "explained_variance_ratio": round(float((s[idx] * s[idx]).item()) / total, 6),
                "source_direction_count": len(names),
                "source_directions": names,
            }
        )
    comp_by_key = {key_for(x): x for x in comparisons}
    projection_rows = []
    for key, item in raw_vectors.items():
        comp = comp_by_key.get(key, {})
        vec = item["payload"].get("delta_residual")
        if not torch.is_tensor(vec):
            continue
        for idx in range(min(3, vh.shape[0])):
            projection_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase252",
                    "created_at": now,
                    "projection_id": f"phase252:shared_projection:{model_name}:{key[1]}:{key[2]}:pc{idx + 1}",
                    "model": model_name,
                    "case_id": key[1],
                    "variant_id": key[2],
                    "component_index": idx + 1,
                    "projection_cosine": round(cosine(vec, vh[idx]), 6),
                    "best_suppression_source": comp.get("best_suppression_source"),
                    "best_suppression_delta": comp.get("best_suppression_delta"),
                    "tokenbank_suppression_delta": comp.get("tokenbank_suppression_delta"),
                    "natural_raw_suppression_delta": comp.get("natural_raw_suppression_delta"),
                }
            )
    return subspace_rows, projection_rows


def token_ids(tokenizer: Any, texts: list[str]) -> list[int]:
    out = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            out.append(int(ids[0]))
    return sorted(set(out))


def group_score(logits: torch.Tensor, ids: list[int]) -> float:
    vals = [float(logits[i].item()) for i in ids if 0 <= i < logits.numel()]
    return max(vals) if vals else -1e30


def closure_scores(tokenizer: Any, logits: torch.Tensor) -> dict[str, float]:
    eos_ids = [int(tokenizer.eos_token_id)] if tokenizer.eos_token_id is not None else []
    period_ids = token_ids(tokenizer, [".", " .", ".\n", ". ", "。"])
    newline_ids = token_ids(tokenizer, ["\n", "\n\n", " \n"])
    continuation_ids = token_ids(tokenizer, [" the", " The", " is", " are", " and", " which", " because"])
    closure = max(group_score(logits, eos_ids), group_score(logits, period_ids), group_score(logits, newline_ids))
    continuation = group_score(logits, continuation_ids)
    return {
        "eos_logit": group_score(logits, eos_ids),
        "period_logit": group_score(logits, period_ids),
        "newline_logit": group_score(logits, newline_ids),
        "continuation_logit": continuation,
        "closure_proxy_margin": closure - continuation,
    }


def replace_last_token(output: Any, delta: torch.Tensor, sign: float = 1.0) -> Any:
    tensor = p228.extract_tensor(output)
    if tensor is None:
        return output
    changed = tensor.clone()
    changed[:, -1, :] = changed[:, -1, :] + float(sign) * delta.to(device=changed.device, dtype=changed.dtype)
    if torch.is_tensor(output):
        return changed
    if isinstance(output, tuple):
        return (changed, *output[1:])
    return output


@contextmanager
def no_hook() -> Iterator[None]:
    yield


@contextmanager
def residual_hook(model: Any, layer_idx: int, direction: torch.Tensor, sign: float) -> Iterator[None]:
    layers = p228.get_layers(model)

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return replace_last_token(output, direction, sign=sign)

    handle = layers[int(layer_idx)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def stepwise_trace(model: Any, tokenizer: Any, device: torch.device, prompt: str, steps: int, hook_factory: Any) -> tuple[str, list[dict[str, Any]]]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = int(encoded["input_ids"].shape[-1])
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    rows = []
    for step in range(1, int(steps) + 1):
        with hook_factory():
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[0, -1, :].detach().float().cpu()
        next_id = int(torch.argmax(logits).item())
        scores = closure_scores(tokenizer, logits)
        rows.append(
            {
                "step": step,
                "token_id": next_id,
                "token_text": tokenizer.decode([next_id]),
                **{k: round(v, 6) for k, v in scores.items()},
            }
        )
        next_tensor = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_tensor)], dim=1)
        if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
            break
    text = tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True).strip()
    return text, rows


def direction_for_source(source: str, tokenbank_id: str, natural_id: str, tokenbank: dict[str, torch.Tensor], natural: dict[str, torch.Tensor], raw: dict[str, Any], perturb_norm: float) -> torch.Tensor:
    if source == "tokenbank":
        return normalize(tokenbank[tokenbank_id], perturb_norm)
    if source == "natural_raw":
        return normalize(natural[natural_id], perturb_norm)
    if source == "target":
        return normalize(raw["target_direction"], perturb_norm)
    return normalize(raw["competitor_direction"], perturb_norm)


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_by_key = load_behavior_by_key()
    raw_vectors = load_raw_vectors(args.model)
    comparisons = read_jsonl(PHASE251_DIR / f"phase251_{args.model}_tokenbank_vs_natural_direction_rows.jsonl")
    high_conf = read_jsonl(PHASE251_DIR / f"phase251_{args.model}_high_confidence_rollout_candidates.jsonl")
    final_layer = int(SPECS[args.model]["final_observe_layer"])
    model = None
    tokenizer = None
    direction_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    subspace_projection_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenbank = build_tokenbank_directions(model, tokenizer)
        natural = load_natural_directions(args.model)
        orth, _orth_meta = p251.orthogonalize(natural)
        all_dirs: dict[str, torch.Tensor] = {}
        for name in TOKENBANK_CORE:
            all_dirs[f"tokenbank:{name}"] = tokenbank[name]
        for name in NATURAL_CORE:
            if name in natural:
                all_dirs[f"natural_raw:{name}"] = natural[name]
            if name in orth:
                all_dirs[f"natural_orth:{name}"] = orth[name]
        for name, vec in all_dirs.items():
            source, direction_id = name.split(":", 1)
            direction_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase252",
                    "created_at": utc_now(),
                    "direction_id": f"phase252:direction:{args.model}:{name}",
                    "model": args.model,
                    "direction_source": source,
                    "direction_name": direction_id,
                    "direction_norm": round(norm(vec), 6),
                }
            )
        names = sorted(all_dirs)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                cosine_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase252",
                        "created_at": utc_now(),
                        "cosine_id": f"phase252:cosine:{args.model}:{safe_slug(a)}:{safe_slug(b)}",
                        "model": args.model,
                        "direction_a": a,
                        "direction_b": b,
                        "cosine": round(cosine(all_dirs[a], all_dirs[b]), 6),
                        "abs_cosine": round(abs(cosine(all_dirs[a], all_dirs[b])), 6),
                    }
                )
        groups = {
            "tokenbank_core": [tokenbank[x] for x in TOKENBANK_CORE],
            "natural_raw_core": [natural[x] for x in NATURAL_CORE if x in natural],
            "natural_orth_core": [orth[x] for x in NATURAL_CORE if x in orth],
        }
        for a_name, a_vecs in groups.items():
            for b_name, b_vecs in groups.items():
                if a_name >= b_name:
                    continue
                overlap_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase252",
                        "created_at": utc_now(),
                        "overlap_id": f"phase252:overlap:{args.model}:{a_name}:{b_name}",
                        "model": args.model,
                        "subspace_a": a_name,
                        "subspace_b": b_name,
                        "overlap_score": round(subspace_overlap(a_vecs, b_vecs), 6),
                        "dim_a": len(a_vecs),
                        "dim_b": len(b_vecs),
                    }
                )
        subspace_rows, subspace_projection_rows = pca_rows(args.model, all_dirs, raw_vectors, comparisons)
        for item in high_conf[: int(args.max_rollout_candidates)]:
            key = key_for(item)
            behavior = behavior_by_key.get(key)
            raw_item = raw_vectors.get(key)
            if behavior is None or raw_item is None:
                missing_rows.append({"model": args.model, "case_id": key[1], "variant_id": key[2], "reason": "missing_behavior_or_raw"})
                continue
            raw_payload = raw_item["payload"]
            route = str(item.get("recommended_next_test"))
            tokenbank_id, natural_id = p251.ROUTE_TO_DIRECTIONS.get(route, ("continuation_regime", "natural_continuation_explain"))
            perturb_norm = max(norm(raw_payload.get("target_direction")), norm(raw_payload.get("competitor_direction")), norm(raw_payload.get("delta_residual")) * float(args.perturb_scale), 1e-6)
            best_source = str(item.get("best_suppression_source") or "natural_raw")
            best_direction = direction_for_source(best_source, tokenbank_id, natural_id, tokenbank, natural, raw_payload, perturb_norm)
            prompt = str(behavior["prompt_variant"])
            for steps in [8, 16]:
                for condition, factory in [
                    ("no_intervention", no_hook),
                    (f"{best_source}_suppression", lambda d=best_direction: residual_hook(model, final_layer, d, sign=-1.0)),
                ]:
                    text, trace = stepwise_trace(model, tokenizer, device, prompt, steps, factory)
                    for step_row in trace:
                        rollout_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "phase_id": "Phase252",
                                "created_at": utc_now(),
                                "trace_id": f"phase252:rollout:{args.model}:{key[1]}:{key[2]}:{condition}:{steps}:step{step_row['step']}",
                                "model": args.model,
                                "case_id": key[1],
                                "variant_id": key[2],
                                "family_id": behavior.get("family_id"),
                                "mode_id": behavior.get("mode_id"),
                                "condition": condition,
                                "max_steps": steps,
                                "best_suppression_source": best_source,
                                "tokenbank_regime": tokenbank_id,
                                "natural_contrast_id": natural_id,
                                "generated_text": text,
                                **step_row,
                            }
                        )
            log(f"{args.model}: rollout candidate {key[1]} {key[2]} rows={len(rollout_rows)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    observations = observation_rows(cosine_rows, overlap_rows, rollout_rows)
    metrics = metric_rows(args.model, cosine_rows, overlap_rows, subspace_rows, rollout_rows)
    edges = graph_edges(args.model, cosine_rows, overlap_rows, subspace_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Shared subspace and coupled regime field analysis",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "direction_rows": len(direction_rows),
        "direction_cosine_rows": len(cosine_rows),
        "subspace_overlap_rows": len(overlap_rows),
        "shared_effective_subspace_rows": len(subspace_rows),
        "shared_subspace_projection_rows": len(subspace_projection_rows),
        "rollout_closure_trace_rows": len(rollout_rows),
        "missing_rows": len(missing_rows),
        "top_abs_cosines": sorted(cosine_rows, key=lambda x: safe_float(x.get("abs_cosine")), reverse=True)[:10],
        "subspace_overlaps": overlap_rows,
        "high_confidence_candidates_traced": len({(x["case_id"], x["variant_id"]) for x in rollout_rows}),
    }
    write_json(out_dir / f"phase252_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase252_{args.model}_direction_rows.jsonl", direction_rows)
    write_jsonl(out_dir / f"phase252_{args.model}_direction_cosine_rows.jsonl", cosine_rows)
    write_jsonl(out_dir / f"phase252_{args.model}_subspace_overlap_rows.jsonl", overlap_rows)
    write_jsonl(out_dir / f"phase252_{args.model}_shared_effective_subspace_rows.jsonl", subspace_rows)
    write_jsonl(out_dir / f"phase252_{args.model}_shared_subspace_projection_rows.jsonl", subspace_projection_rows)
    write_jsonl(out_dir / f"phase252_{args.model}_rollout_closure_trace_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase252_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase252_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase252_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase252_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def observation_rows(cosines: list[dict[str, Any]], overlaps: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in cosines:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase252",
                "created_at": now,
                "observation_id": row["cosine_id"],
                "model": row["model"],
                "level": "direction_overlap",
                "component": "direction_cosine",
                "metric_name": "abs_cosine",
                "metric_value": row["abs_cosine"],
                "metric_unit": "cosine",
            }
        )
    for row in overlaps:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase252",
                "created_at": now,
                "observation_id": row["overlap_id"],
                "model": row["model"],
                "level": "subspace_overlap",
                "component": f"{row['subspace_a']}:{row['subspace_b']}",
                "metric_name": "overlap_score",
                "metric_value": row["overlap_score"],
                "metric_unit": "normalized_frobenius",
            }
        )
    for row in rollouts:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase252",
                "created_at": now,
                "observation_id": row["trace_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "rollout_closure_trace",
                "component": row["condition"],
                "metric_name": "closure_proxy_margin",
                "metric_value": row["closure_proxy_margin"],
                "metric_unit": "logit",
            }
        )
    return rows


def metric_rows(model_name: str, cosines: list[dict[str, Any]], overlaps: list[dict[str, Any]], subspaces: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    if cosines:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase252",
                "created_at": now,
                "metric_id": f"phase252:{model_name}:mean_abs_direction_cosine",
                "scope": "direction_overlap",
                "model": model_name,
                "metric_name": "mean_abs_direction_cosine",
                "metric_value": round(mean(safe_float(x.get("abs_cosine")) for x in cosines), 6),
                "rows": len(cosines),
            }
        )
    for row in overlaps:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase252",
                "created_at": now,
                "metric_id": row["overlap_id"],
                "scope": "subspace_overlap",
                "model": model_name,
                "metric_name": "overlap_score",
                "metric_value": row["overlap_score"],
                "rows": 1,
            }
        )
    for row in subspaces:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase252",
                "created_at": now,
                "metric_id": row["subspace_id"],
                "scope": "shared_effective_subspace",
                "model": model_name,
                "metric_name": "explained_variance_ratio",
                "metric_value": row["explained_variance_ratio"],
                "rows": row["source_direction_count"],
            }
        )
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        by_condition[str(row.get("condition"))].append(row)
    for condition, items in by_condition.items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase252",
                "created_at": now,
                "metric_id": f"phase252:{model_name}:{condition}:mean_closure_proxy",
                "scope": "rollout_closure_trace",
                "model": model_name,
                "condition": condition,
                "metric_name": "mean_closure_proxy_margin",
                "metric_value": round(mean(safe_float(x.get("closure_proxy_margin")) for x in items), 6),
                "rows": len(items),
            }
        )
    return rows


def graph_edges(model_name: str, cosines: list[dict[str, Any]], overlaps: list[dict[str, Any]], subspaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in sorted(cosines, key=lambda x: safe_float(x.get("abs_cosine")), reverse=True)[:15]:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase252",
                "created_at": now,
                "edge_id": f"phase252:cosine_edge:{row['cosine_id']}",
                "source": row["direction_a"],
                "target": row["direction_b"],
                "edge_type": "direction_overlap",
                "model": model_name,
                "evidence_type": "direction_cosine",
                "effect_direction": "aligned" if safe_float(row.get("cosine")) >= 0 else "opposed",
                "effect_size": row["cosine"],
                "confidence": 0.42,
                "supporting_phases": ["Phase248", "Phase250", "Phase251", "Phase252"],
                "status": "shared_subspace_candidate_not_closure",
            }
        )
    for row in overlaps:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase252",
                "created_at": now,
                "edge_id": f"phase252:overlap_edge:{row['overlap_id']}",
                "source": f"subspace:{row['subspace_a']}",
                "target": f"subspace:{row['subspace_b']}",
                "edge_type": "subspace_overlap",
                "model": model_name,
                "evidence_type": "normalized_frobenius_overlap",
                "effect_direction": "shared" if safe_float(row.get("overlap_score")) >= 0.25 else "weak_shared",
                "effect_size": row["overlap_score"],
                "confidence": 0.45,
                "supporting_phases": ["Phase250", "Phase251", "Phase252"],
                "status": "coupled_regime_field_candidate",
            }
        )
    return rows


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase252_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    direction_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for model in MODELS:
        direction_rows.extend(read_jsonl(out_dir / f"phase252_{model}_direction_rows.jsonl"))
        cosine_rows.extend(read_jsonl(out_dir / f"phase252_{model}_direction_cosine_rows.jsonl"))
        overlap_rows.extend(read_jsonl(out_dir / f"phase252_{model}_subspace_overlap_rows.jsonl"))
        subspace_rows.extend(read_jsonl(out_dir / f"phase252_{model}_shared_effective_subspace_rows.jsonl"))
        projection_rows.extend(read_jsonl(out_dir / f"phase252_{model}_shared_subspace_projection_rows.jsonl"))
        rollout_rows.extend(read_jsonl(out_dir / f"phase252_{model}_rollout_closure_trace_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase252_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase252_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase252_{model}_graph_edges.jsonl"))
        missing_rows.extend(read_jsonl(out_dir / f"phase252_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.80,
        "candidate_clustering": 0.43,
        "case_bank_calibration": 0.40,
        "high_value_trace_selection": 0.65,
        "trace_signature_validation": 0.37,
        "focused_causal_validation": 0.24,
        "raw_delta_vector_archive": 0.26,
        "raw_vector_factor_decomposition": 0.25,
        "regime_field_direction_bank": 0.34,
        "natural_regime_direction_bank": 0.29,
        "regime_level_causal_validation": 0.24,
        "orthogonalized_direction_validation": 0.17,
        "shared_subspace_analysis": 0.18,
        "coupled_regime_field_analysis": 0.16,
        "residual_state_signature": 0.47,
        "readout_competition_trace": 0.70,
        "stepwise_rollout_trace": 0.30,
        "causal_closure": 0.13,
        "general_language_mechanism_confidence": 0.61,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model shared subspace and coupled regime field analysis",
        "status": "complete" if len(summaries) == 3 else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "model_summaries": {x.get("model"): x for x in summaries},
        "direction_rows": len(direction_rows),
        "direction_cosine_rows": len(cosine_rows),
        "subspace_overlap_rows": len(overlap_rows),
        "shared_effective_subspace_rows": len(subspace_rows),
        "shared_subspace_projection_rows": len(projection_rows),
        "rollout_closure_trace_rows": len(rollout_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing_rows),
        "top_abs_cosines": sorted(cosine_rows, key=lambda x: safe_float(x.get("abs_cosine")), reverse=True)[:12],
        "subspace_overlaps": overlap_rows,
        "mean_closure_proxy_by_condition": mean_closure_by_condition(rollout_rows),
        "pattern_atlas_progress": progress,
        "judgement": "shared_subspace_and_rollout_trace_candidate_not_closure",
    }
    write_json(out_dir / "phase252_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase252_direction_rows.jsonl", direction_rows)
    write_jsonl(out_dir / "phase252_direction_cosine_rows.jsonl", cosine_rows)
    write_jsonl(out_dir / "phase252_subspace_overlap_rows.jsonl", overlap_rows)
    write_jsonl(out_dir / "phase252_shared_effective_subspace_rows.jsonl", subspace_rows)
    write_jsonl(out_dir / "phase252_shared_subspace_projection_rows.jsonl", projection_rows)
    write_jsonl(out_dir / "phase252_rollout_closure_trace_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / "phase252_observations.jsonl", observations)
    write_jsonl(out_dir / "phase252_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase252_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase252_missing_rows.jsonl", missing_rows)
    write_report(out_dir / "phase252_coupled_regime_field_report.md", payload)
    update_pattern_atlas(payload, observations, metrics, edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def mean_closure_by_condition(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("condition"))].append(safe_float(row.get("closure_proxy_margin")))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase252 shared subspace and coupled regime field analysis",
        "",
        "Phase252 analyzes direction overlap, shared subspace structure, and rollout closure traces for high-confidence candidates.",
        "It is not closure validation.",
        "",
        "## Counts",
        "",
        f"- direction_cosine_rows: {summary['direction_cosine_rows']}",
        f"- subspace_overlap_rows: {summary['subspace_overlap_rows']}",
        f"- rollout_closure_trace_rows: {summary['rollout_closure_trace_rows']}",
        "",
        "## Closure Proxy Means",
        "",
        "```json",
        json.dumps(summary["mean_closure_proxy_by_condition"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Top Direction Cosines",
        "",
        "```json",
        json.dumps(summary["top_abs_cosines"][:8], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Progress",
        "",
        "```json",
        json.dumps(summary["pattern_atlas_progress"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_pattern_atlas(summary: dict[str, Any], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    ATLAS_ROOT.mkdir(parents=True, exist_ok=True)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    progress = read_json(ATLAS_ROOT / "progress.json")
    progress.update(
        {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now(),
            "latest_phase": "Phase252",
            "latest_round": summary.get("title"),
            "progress": summary.get("pattern_atlas_progress", {}),
            "current_priority": "separate shared effective subspace from rollout and closure effects",
            "small_model_bias_warning": "Phase252 uses qwen3/glm4/deepseek7b only; shared subspace evidence is candidate-level, not closure.",
        }
    )
    write_json(ATLAS_ROOT / "progress.json", progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase252 shared subspace and coupled regime field analysis")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-rollout-candidates", type=int, default=5)
    parser.add_argument("--perturb-scale", type=float, default=0.35)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is set")
    evaluate_model(args)


if __name__ == "__main__":
    main()
