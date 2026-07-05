#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase901_stop_token_competitiveness_audit as p901  # noqa: E402
import phase937_semantic_reuse_difference_state_atlas as p937  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 938
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase938_semantic_factor_causal_transfer_audit")
PHASE937_ROOT = Path("tests/result/phase937_semantic_reuse_difference_state_atlas")


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


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return None if not vals else float(sum(vals) / len(vals))


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def phase937_best_hidden(model_name: str, phase937_round: str) -> dict[str, int]:
    path = PHASE937_ROOT / phase937_round / f"phase937_{model_name}_summary.json"
    summary = read_json(path)
    best = ((summary.get("summary") or {}).get("best_by_relation") or {})
    out = {}
    for relation, row in best.items():
        if row and row.get("hidden_idx") is not None:
            out[str(relation)] = int(row["hidden_idx"])
    return out


def first_token_candidates(tokenizer, label: str) -> list[int]:
    return p937.first_token_candidates(tokenizer, label)


def label_score(logits: torch.Tensor, token_ids: list[int]) -> tuple[float | None, int | None, str | None]:
    if not token_ids:
        return None, None, None
    best_id = max(token_ids, key=lambda token_id: float(logits[int(token_id)].item()))
    return float(logits[int(best_id)].item()), int(best_id), None


def rank_of(logits: torch.Tensor, token_id: int | None) -> int | None:
    return p937.rank_of(logits, token_id)


def relation_labels(samples: list[dict[str, Any]], relation: str) -> list[str]:
    return sorted({str(row["target_label"]) for row in samples if str(row["relation"]) == relation})


def label_token_map(tokenizer, labels: list[str]) -> dict[str, list[int]]:
    return {label: first_token_candidates(tokenizer, label) for label in labels}


def sample_key(sample: dict[str, Any]) -> str:
    return str(sample["sample_id"])


def encode_batch(tokenizer, device: torch.device, samples: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [str(sample["prompt"]) for sample in samples],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = attention_mask.sum(dim=1).long() - 1
    return input_ids, attention_mask, last_pos


def forward_logits(
    model,
    tokenizer,
    device: torch.device,
    samples: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for start in range(0, len(samples), max(1, int(batch_size))):
        batch = samples[start : start + max(1, int(batch_size))]
        input_ids, attention_mask, last_pos = encode_batch(tokenizer, device, batch)
        batch_idx = torch.arange(input_ids.shape[0], device=device)
        with torch.inference_mode():
            result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits = result.logits[batch_idx, last_pos].detach().float().cpu()
        for idx, sample in enumerate(batch):
            out[sample_key(sample)] = logits[idx]
        del result, logits, input_ids, attention_mask
    return out


def forward_vectors(
    model,
    tokenizer,
    device: torch.device,
    samples: list[dict[str, Any]],
    hidden_indices: list[int],
    batch_size: int,
) -> dict[tuple[str, int], torch.Tensor]:
    vectors: dict[tuple[str, int], torch.Tensor] = {}
    for start in range(0, len(samples), max(1, int(batch_size))):
        batch = samples[start : start + max(1, int(batch_size))]
        input_ids, attention_mask, last_pos = encode_batch(tokenizer, device, batch)
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        for idx, sample in enumerate(batch):
            pos = int(last_pos[idx].item())
            for hidden_idx in hidden_indices:
                vectors[(sample_key(sample), int(hidden_idx))] = (
                    result.hidden_states[int(hidden_idx)][idx, pos].detach().float().cpu()
                )
        del result, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return vectors


def target_margin(
    logits: torch.Tensor,
    target_label: str,
    label_tokens: dict[str, list[int]],
) -> dict[str, Any]:
    target_score, target_id, _ = label_score(logits, label_tokens.get(target_label) or [])
    scores = {}
    ids = {}
    for label, token_ids in label_tokens.items():
        score, token_id, _ = label_score(logits, token_ids)
        if score is not None:
            scores[label] = float(score)
            ids[label] = token_id
    if not scores or target_score is None:
        return {
            "target_label_logit": target_score,
            "target_label_rank": None,
            "relation_winner": None,
            "relation_winner_logit": None,
            "target_margin_vs_relation_best_other": None,
            "target_is_relation_winner": False,
            "target_token_id": target_id,
        }
    other_scores = {label: score for label, score in scores.items() if label != target_label}
    winner = max(scores, key=scores.get)
    other_best = None if not other_scores else max(other_scores.values())
    return {
        "target_label_logit": float(target_score),
        "target_label_rank": rank_of(logits, target_id),
        "relation_winner": winner,
        "relation_winner_logit": float(scores[winner]),
        "target_margin_vs_relation_best_other": None if other_best is None else float(target_score - other_best),
        "target_is_relation_winner": bool(winner == target_label),
        "target_token_id": target_id,
    }


def direction_for_label(
    train_samples: list[dict[str, Any]],
    vectors: dict[tuple[str, int], torch.Tensor],
    hidden_idx: int,
    label: str,
) -> torch.Tensor | None:
    positives = [vectors[(sample_key(row), hidden_idx)] for row in train_samples if str(row["target_label"]) == label]
    negatives = [vectors[(sample_key(row), hidden_idx)] for row in train_samples if str(row["target_label"]) != label]
    if not positives or not negatives:
        return None
    return torch.stack(positives).mean(dim=0) - torch.stack(negatives).mean(dim=0)


def deterministic_random_same_norm(direction: torch.Tensor, salt: str) -> torch.Tensor:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False) % (2**31 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_vec = torch.randn(direction.shape, generator=generator, dtype=torch.float32)
    norm = torch.linalg.vector_norm(random_vec)
    target_norm = torch.linalg.vector_norm(direction)
    if float(norm.item()) == 0.0:
        return random_vec
    return random_vec / norm * target_norm


def patch_module_for_hidden_idx(model, hidden_idx: int):
    if int(hidden_idx) == 0:
        return model.get_input_embeddings()
    layers = get_layers(model)
    layer_idx = int(hidden_idx) - 1
    if not (0 <= layer_idx < len(layers)):
        return None
    return layers[layer_idx]


def patched_logits_batch(
    model,
    tokenizer,
    device: torch.device,
    samples: list[dict[str, Any]],
    hidden_idx: int,
    direction: torch.Tensor,
    alpha: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    module = patch_module_for_hidden_idx(model, hidden_idx)
    if module is None:
        return out
    for start in range(0, len(samples), max(1, int(batch_size))):
        batch = samples[start : start + max(1, int(batch_size))]
        input_ids, attention_mask, last_pos = encode_batch(tokenizer, device, batch)
        batch_idx = torch.arange(input_ids.shape[0], device=device)
        delta = (direction.to(device=device, dtype=torch.float32) * float(alpha)).to(device=device)

        def patch_tensor(hidden: torch.Tensor) -> torch.Tensor:
            patched = hidden.clone()
            pos = last_pos.to(device=patched.device)
            idx = torch.arange(patched.shape[0], device=patched.device)
            patched[idx, pos, :] = patched[idx, pos, :] + delta.to(dtype=patched.dtype)
            return patched

        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                return patch_tensor(output)
            if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                return (patch_tensor(output[0]), *output[1:])
            return output

        handle = module.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            logits = result.logits[batch_idx, last_pos].detach().float().cpu()
            for idx, sample in enumerate(batch):
                out[sample_key(sample)] = logits[idx]
            del result, logits
        finally:
            handle.remove()
        del input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def build_direction_specs(
    samples: list[dict[str, Any]],
    vectors: dict[tuple[str, int], torch.Tensor],
    hidden_by_relation: dict[str, int],
    min_train_per_label: int,
) -> list[dict[str, Any]]:
    specs = []
    for relation, hidden_idx in sorted(hidden_by_relation.items()):
        rel_samples = [row for row in samples if str(row["relation"]) == relation]
        labels = sorted({str(row["target_label"]) for row in rel_samples})
        templates = sorted({str(row["prompt_template"]) for row in rel_samples})
        if len(templates) < 2:
            continue
        for train_template in templates:
            test_templates = [template for template in templates if template != train_template]
            train = [row for row in rel_samples if str(row["prompt_template"]) == train_template]
            label_counts = Counter(str(row["target_label"]) for row in train)
            directions: dict[str, torch.Tensor] = {}
            for label in labels:
                if label_counts[label] < int(min_train_per_label):
                    continue
                direction = direction_for_label(train, vectors, int(hidden_idx), label)
                if direction is not None and float(torch.linalg.vector_norm(direction).item()) > 0:
                    directions[label] = direction.float().cpu()
            for label, direction in directions.items():
                wrong_labels = [item for item in sorted(directions) if item != label]
                wrong_label = wrong_labels[0] if wrong_labels else None
                for test_template in test_templates:
                    test = [
                        row
                        for row in rel_samples
                        if str(row["prompt_template"]) == test_template and str(row["target_label"]) == label
                    ]
                    if not test:
                        continue
                    specs.append(
                        {
                            "relation": relation,
                            "hidden_idx": int(hidden_idx),
                            "target_label": label,
                            "train_template": train_template,
                            "test_template": test_template,
                            "test_samples": test,
                            "target_direction": direction,
                            "wrong_label": wrong_label,
                            "wrong_direction": None if wrong_label is None else directions[wrong_label],
                            "random_direction": deterministic_random_same_norm(
                                direction, f"{relation}|{hidden_idx}|{label}|{train_template}"
                            ),
                            "direction_norm": float(torch.linalg.vector_norm(direction).item()),
                            "train_label_count": int(label_counts[label]),
                            "train_other_count": int(len(train) - label_counts[label]),
                        }
                    )
    return specs


def make_row(
    model_name: str,
    sample: dict[str, Any],
    spec: dict[str, Any],
    condition: str,
    alpha: float | None,
    base_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
) -> dict[str, Any]:
    base_logit = base_metrics.get("target_label_logit")
    patched_logit = patched_metrics.get("target_label_logit")
    base_margin = base_metrics.get("target_margin_vs_relation_best_other")
    patched_margin = patched_metrics.get("target_margin_vs_relation_best_other")
    base_rank = base_metrics.get("target_label_rank")
    patched_rank = patched_metrics.get("target_label_rank")
    return {
        "phase": PHASE,
        "row_kind": "phase938_semantic_factor_causal_transfer_row",
        "model": model_name,
        "sample_id": sample.get("sample_id"),
        "domain": sample.get("domain"),
        "object": sample.get("object"),
        "relation": spec.get("relation"),
        "target_label": spec.get("target_label"),
        "prompt_template": sample.get("prompt_template"),
        "train_template": spec.get("train_template"),
        "test_template": spec.get("test_template"),
        "hidden_idx": int(spec.get("hidden_idx")),
        "condition": condition,
        "alpha": alpha,
        "wrong_label": spec.get("wrong_label"),
        "direction_norm": spec.get("direction_norm"),
        "train_label_count": spec.get("train_label_count"),
        "train_other_count": spec.get("train_other_count"),
        "base_target_label_logit": base_logit,
        "patched_target_label_logit": patched_logit,
        "target_logit_delta": None if base_logit is None or patched_logit is None else float(patched_logit - base_logit),
        "base_target_margin": base_margin,
        "patched_target_margin": patched_margin,
        "target_margin_delta": None if base_margin is None or patched_margin is None else float(patched_margin - base_margin),
        "base_target_label_rank": base_rank,
        "patched_target_label_rank": patched_rank,
        "target_rank_delta": None if base_rank is None or patched_rank is None else int(patched_rank) - int(base_rank),
        "rank_improved": bool(base_rank is not None and patched_rank is not None and int(patched_rank) < int(base_rank)),
        "base_relation_winner": base_metrics.get("relation_winner"),
        "patched_relation_winner": patched_metrics.get("relation_winner"),
        "base_target_is_relation_winner": base_metrics.get("target_is_relation_winner"),
        "patched_target_is_relation_winner": patched_metrics.get("target_is_relation_winner"),
        "new_relation_winner_target": bool(
            (not base_metrics.get("target_is_relation_winner")) and patched_metrics.get("target_is_relation_winner")
        ),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "target_logit_delta_mean": mean([row.get("target_logit_delta") for row in rows]),
        "target_margin_delta_mean": mean([row.get("target_margin_delta") for row in rows]),
        "rank_improved": sum(1 for row in rows if row.get("rank_improved")),
        "new_relation_winner_target": sum(1 for row in rows if row.get("new_relation_winner_target")),
        "patched_relation_winner_target": sum(1 for row in rows if row.get("patched_target_is_relation_winner")),
        "base_relation_winner_target": sum(1 for row in rows if row.get("base_target_is_relation_winner")),
    }


def summarize_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        item = {key: value for key, value in zip(keys, key_tuple)}
        item.update(summarize_rows(items))
        out.append(item)
    out.sort(key=lambda row: (finite(row.get("target_margin_delta_mean"), -999), finite(row.get("target_logit_delta_mean"), -999)), reverse=True)
    return out


def causal_evidence(by_relation_condition: list[dict[str, Any]]) -> str:
    relation_positive = 0
    for relation in sorted({row.get("relation") for row in by_relation_condition}):
        rel = [row for row in by_relation_condition if row.get("relation") == relation]
        target = [row for row in rel if row.get("condition") == "target_direction"]
        controls = [row for row in rel if row.get("condition") in {"wrong_label_direction", "random_same_norm"}]
        if not target or not controls:
            continue
        target_best = max(finite(row.get("target_margin_delta_mean"), -999) for row in target)
        control_best = max(finite(row.get("target_margin_delta_mean"), -999) for row in controls)
        if target_best > 0 and target_best > control_best + 0.02:
            relation_positive += 1
    if relation_positive >= 2:
        return "semantic_factor_causal_transfer_positive"
    if relation_positive == 1:
        return "partial_semantic_factor_causal_transfer"
    return "semantic_factor_causal_transfer_weak_or_negative"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = p937.build_samples(args)
    hidden_by_relation = phase937_best_hidden(args.model, args.phase937_round)
    if not hidden_by_relation:
        hidden_by_relation = {relation: -1 for relation in parse_csv(args.relations)}
    dry_payload = {
        "phase": PHASE,
        "title": "Semantic Factor Causal Transfer Audit",
        "model": args.model,
        "sample_count": len(samples),
        "relations": sorted({str(row["relation"]) for row in samples}),
        "hidden_by_relation": hidden_by_relation,
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase938_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase938_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    attn_impl = None
    rows: list[dict[str, Any]] = []
    direction_specs: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        auto_indices = p937.auto_hidden_indices(model)
        hidden_by_relation = {
            rel: (auto_indices[len(auto_indices) // 2] if int(idx) < 0 else int(idx))
            for rel, idx in hidden_by_relation.items()
        }
        hidden_indices = sorted(set(hidden_by_relation.values()))
        vectors = forward_vectors(model, tokenizer, device, samples, hidden_indices, int(args.batch_size))
        baseline_logits = forward_logits(model, tokenizer, device, samples, int(args.batch_size))
        direction_specs = build_direction_specs(samples, vectors, hidden_by_relation, int(args.min_train_per_label))
        alphas = [float(x) for x in parse_csv(args.alphas)]
        labels_by_relation = {relation: relation_labels(samples, relation) for relation in hidden_by_relation}
        token_maps = {relation: label_token_map(tokenizer, labels) for relation, labels in labels_by_relation.items()}
        baseline_metrics: dict[str, dict[str, Any]] = {}
        for sample in samples:
            relation = str(sample["relation"])
            baseline_metrics[sample_key(sample)] = target_margin(
                baseline_logits[sample_key(sample)],
                str(sample["target_label"]),
                token_maps[relation],
            )

        for spec_idx, spec in enumerate(direction_specs, 1):
            relation = str(spec["relation"])
            label_tokens = token_maps[relation]
            for sample in spec["test_samples"]:
                base = baseline_metrics[sample_key(sample)]
                rows.append(make_row(args.model, sample, spec, "baseline", None, base, base))
            conditions = [
                ("target_direction", spec["target_direction"]),
                ("wrong_label_direction", spec["wrong_direction"]),
                ("random_same_norm", spec["random_direction"]),
                ("negative_target_direction", -spec["target_direction"]),
            ]
            for condition, direction in conditions:
                if direction is None:
                    continue
                for alpha in alphas:
                    patched = patched_logits_batch(
                        model,
                        tokenizer,
                        device,
                        spec["test_samples"],
                        int(spec["hidden_idx"]),
                        direction,
                        float(alpha),
                        int(args.batch_size),
                    )
                    for sample in spec["test_samples"]:
                        base = baseline_metrics[sample_key(sample)]
                        patched_metrics = target_margin(
                            patched[sample_key(sample)],
                            str(sample["target_label"]),
                            label_tokens,
                        )
                        rows.append(make_row(args.model, sample, spec, condition, float(alpha), base, patched_metrics))
            if spec_idx % max(1, int(args.log_every)) == 0 or spec_idx == len(direction_specs):
                log(f"{args.model}/{args.round_name}: direction_spec={spec_idx}/{len(direction_specs)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    by_condition = summarize_by(rows, ["condition", "alpha"])
    by_relation_condition = summarize_by(rows, ["relation", "condition", "alpha"])
    by_model_relation_label = summarize_by(rows, ["relation", "target_label", "condition", "alpha"])
    evidence = causal_evidence(by_relation_condition)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "hidden_by_relation": hidden_by_relation,
        "direction_specs": len(direction_specs),
        "rows": len(rows),
        "overall": summarize_rows(rows),
        "by_condition": by_condition,
        "by_relation_condition": by_relation_condition,
        "by_relation_label": by_model_relation_label[:240],
        "evidence_label": evidence,
        "boundary": "direction injection causal audit on first-token label logits; not full semantic factor closure",
    }
    write_json(out_dir / f"phase938_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase938_{args.model}_rows.jsonl", rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "evidence": evidence,
                "by_condition": by_condition,
                "top_relation_condition": by_relation_condition[:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase938_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts = Counter(str(summary.get("evidence_label") or summary.get("status")) for summary in summaries)
    relation_rows = []
    condition_rows = []
    for summary in summaries:
        model = summary.get("model")
        for row in summary.get("by_condition") or []:
            item = dict(row)
            item["model"] = model
            condition_rows.append(item)
        for row in summary.get("by_relation_condition") or []:
            item = dict(row)
            item["model"] = model
            relation_rows.append(item)
    relation_rows.sort(key=lambda row: (finite(row.get("target_margin_delta_mean"), -999), finite(row.get("target_logit_delta_mean"), -999)), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "evidence_counts": dict(evidence_counts),
        "model_summaries": summaries,
        "condition_rows": condition_rows,
        "top_relation_condition_rows": relation_rows[:160],
    }
    write_json(out_dir / "phase938_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase938_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 938 semantic factor causal transfer audit", ""]
    lines += ["## Evidence", ""]
    for key, value in sorted((payload.get("evidence_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Condition Rows", ""]
    lines.append("| model | condition | alpha | rows | mean logit delta | mean margin delta | rank improved | new winner |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("condition_rows") or []:
        lines.append(
            "| {model} | {condition} | {alpha} | {rows} | {target_logit_delta_mean} | {target_margin_delta_mean} | {rank_improved} | {new_relation_winner_target} |".format(
                **row
            )
        )
    lines += ["", "## Top Relation Conditions", ""]
    lines.append("| model | relation | condition | alpha | rows | mean logit delta | mean margin delta | rank improved | new winner |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_relation_condition_rows") or []:
        lines.append(
            "| {model} | {relation} | {condition} | {alpha} | {rows} | {target_logit_delta_mean} | {target_margin_delta_mean} | {rank_improved} | {new_relation_winner_target} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="semantic_factor_causal_transfer_audit")
    parser.add_argument("--phase937-round", default="semantic_reuse_difference_state_atlas")
    parser.add_argument("--domains", default="")
    parser.add_argument("--relations", default="category,color,function")
    parser.add_argument("--max-objects-per-domain", type=int, default=6)
    parser.add_argument("--templates-per-relation", type=int, default=2)
    parser.add_argument("--min-train-per-label", type=int, default=2)
    parser.add_argument("--alphas", default="0.5,1.0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "evidence": payload["evidence_counts"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
