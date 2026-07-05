#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_demo_bf16 import get_device_for_input, load_model_bf16  # noqa: E402
from model_utils import get_layers, release_model  # noqa: E402


PHASE = 941
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase941_color_feature_neuron_atlas")


COLOR_CASES: list[dict[str, str]] = [
    {"domain": "fruit", "object": "apple", "color": "red"},
    {"domain": "fruit", "object": "cherry", "color": "red"},
    {"domain": "vehicle", "object": "sports car", "color": "red"},
    {"domain": "animal", "object": "cardinal", "color": "red"},
    {"domain": "plant", "object": "rose", "color": "red"},
    {"domain": "fruit", "object": "banana", "color": "yellow"},
    {"domain": "fruit", "object": "lemon", "color": "yellow"},
    {"domain": "vehicle", "object": "taxi", "color": "yellow"},
    {"domain": "animal", "object": "canary", "color": "yellow"},
    {"domain": "object", "object": "school bus", "color": "yellow"},
    {"domain": "fruit", "object": "orange", "color": "orange"},
    {"domain": "vegetable", "object": "carrot", "color": "orange"},
    {"domain": "object", "object": "pumpkin", "color": "orange"},
    {"domain": "plant", "object": "marigold", "color": "orange"},
    {"domain": "fruit", "object": "grape", "color": "purple"},
    {"domain": "vegetable", "object": "eggplant", "color": "purple"},
    {"domain": "plant", "object": "lavender", "color": "purple"},
    {"domain": "object", "object": "amethyst", "color": "purple"},
    {"domain": "material", "object": "metal", "color": "silver"},
    {"domain": "tool", "object": "knife", "color": "silver"},
    {"domain": "tool", "object": "spoon", "color": "silver"},
    {"domain": "vehicle", "object": "train", "color": "silver"},
    {"domain": "animal", "object": "shark", "color": "gray"},
    {"domain": "animal", "object": "whale", "color": "gray"},
    {"domain": "object", "object": "stone", "color": "gray"},
    {"domain": "tool", "object": "hammer", "color": "gray"},
    {"domain": "material", "object": "wood", "color": "brown"},
    {"domain": "animal", "object": "horse", "color": "brown"},
    {"domain": "animal", "object": "dog", "color": "brown"},
    {"domain": "food", "object": "chocolate", "color": "brown"},
    {"domain": "material", "object": "rubber", "color": "black"},
    {"domain": "vehicle", "object": "bicycle", "color": "black"},
    {"domain": "animal", "object": "crow", "color": "black"},
    {"domain": "object", "object": "coal", "color": "black"},
    {"domain": "tool", "object": "cup", "color": "white"},
    {"domain": "vehicle", "object": "boat", "color": "white"},
    {"domain": "animal", "object": "swan", "color": "white"},
    {"domain": "food", "object": "milk", "color": "white"},
    {"domain": "plant", "object": "grass", "color": "green"},
    {"domain": "vegetable", "object": "lettuce", "color": "green"},
    {"domain": "fruit", "object": "lime", "color": "green"},
    {"domain": "tree", "object": "leaf", "color": "green"},
    {"domain": "object", "object": "sky", "color": "blue"},
    {"domain": "object", "object": "ocean", "color": "blue"},
    {"domain": "plant", "object": "blueberry", "color": "blue"},
    {"domain": "object", "object": "sapphire", "color": "blue"},
]


COLOR_TEMPLATES = [
    "The typical color of {article} {object} is",
    "A common color for {article} {object} is",
    "In one word, the color of {article} {object} is",
    "When asked about color, {article} {object} is usually",
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return None if not vals else float(sum(vals) / len(vals))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def article_for(text: str) -> str:
    first = str(text).strip()[:1].lower()
    return "an" if first in {"a", "e", "i", "o", "u"} else "a"


def build_color_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    colors = set(parse_csv(args.colors)) if args.colors else set()
    domains = set(parse_csv(args.domains)) if args.domains else set()
    per_color: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    templates = COLOR_TEMPLATES[: max(1, int(args.templates_per_object))]
    for case in COLOR_CASES:
        if colors and case["color"] not in colors:
            continue
        if domains and case["domain"] not in domains:
            continue
        if int(args.max_objects_per_color) > 0 and per_color[case["color"]] >= int(args.max_objects_per_color):
            continue
        per_color[case["color"]] += 1
        fmt = {**case, "article": article_for(case["object"])}
        for template_idx, template in enumerate(templates):
            sample_id = f"color:{case['color']}:{case['domain']}:{case['object'].replace(' ', '_')}:t{template_idx}"
            samples.append(
                {
                    "phase": PHASE,
                    "sample_id": sample_id,
                    "relation": "color",
                    "domain": case["domain"],
                    "object": case["object"],
                    "target_label": case["color"],
                    "prompt_template": f"color_{template_idx}",
                    "prompt": template.format(**fmt),
                }
            )
    return samples


def color_labels(samples: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["target_label"]) for row in samples})


def first_token_candidates(tokenizer, label: str) -> list[int]:
    ids: list[int] = []
    for text in [label, " " + label]:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            encoded = []
        if encoded:
            token_id = int(encoded[0])
            if token_id not in ids:
                ids.append(token_id)
    return ids


def decode_token(tokenizer, token_id: int | None) -> str | None:
    if token_id is None:
        return None
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return f"<tok_{token_id}>"


def rank_of(logits: torch.Tensor, token_id: int | None) -> int | None:
    if token_id is None:
        return None
    token_id = int(token_id)
    if token_id < 0 or token_id >= int(logits.numel()):
        return None
    value = logits[token_id]
    return int((logits > value).sum().item() + 1)


def label_score(tokenizer, logits: torch.Tensor, label: str) -> dict[str, Any]:
    candidates = first_token_candidates(tokenizer, label)
    if not candidates:
        return {"token_id": None, "token": None, "logit": None, "rank": None}
    best_id = max(candidates, key=lambda token_id: float(logits[int(token_id)].item()))
    return {
        "token_id": int(best_id),
        "token": decode_token(tokenizer, int(best_id)),
        "logit": float(logits[int(best_id)].item()),
        "rank": rank_of(logits, int(best_id)),
    }


def label_token_ids(tokenizer, labels: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for label in labels:
        candidates = first_token_candidates(tokenizer, label)
        if candidates:
            out[label] = int(candidates[0])
    return out


def model_checkpoint_dir(model) -> Path | None:
    candidates = [
        getattr(model, "name_or_path", None),
        getattr(getattr(model, "config", None), "_name_or_path", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists():
            return path
    return None


def checkpoint_weight_map(model) -> tuple[Path, dict[str, str]] | None:
    ckpt_dir = model_checkpoint_dir(model)
    if ckpt_dir is None:
        return None
    for index_path in sorted(ckpt_dir.glob("*.safetensors.index.json")):
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict):
            return ckpt_dir, {str(k): str(v) for k, v in weight_map.items()}
    direct = {path.name: path.name for path in ckpt_dir.glob("*.safetensors")}
    if direct:
        return ckpt_dir, direct
    return None


def checkpoint_tensor_rows(model, weight_names: list[str], row_ids: list[int]) -> dict[int, torch.Tensor] | None:
    resolved = checkpoint_weight_map(model)
    if resolved is None:
        return None
    ckpt_dir, weight_map = resolved
    for weight_name in weight_names:
        shard_name = weight_map.get(weight_name)
        if shard_name is None:
            continue
        rows: dict[int, torch.Tensor] = {}
        with safe_open(str(ckpt_dir / shard_name), framework="pt", device="cpu") as handle:
            weight_slice = handle.get_slice(weight_name)
            for row_id in row_ids:
                rows[int(row_id)] = weight_slice[int(row_id)].detach().float().cpu()
        return rows
    return None


def checkpoint_tensor(model, weight_name: str) -> torch.Tensor | None:
    resolved = checkpoint_weight_map(model)
    if resolved is None:
        return None
    ckpt_dir, weight_map = resolved
    shard_name = weight_map.get(weight_name)
    if shard_name is None:
        return None
    with safe_open(str(ckpt_dir / shard_name), framework="pt", device="cpu") as handle:
        return handle.get_tensor(weight_name).detach().float().cpu()


def get_lm_head_rows(model, token_ids: list[int]) -> dict[int, torch.Tensor]:
    if not hasattr(model, "lm_head"):
        raise ValueError(f"Cannot find lm_head in {type(model).__name__}")
    weight = model.lm_head.weight
    if not getattr(weight, "is_meta", False):
        rows = {}
        with torch.no_grad():
            for token_id in token_ids:
                rows[int(token_id)] = weight[int(token_id)].detach().float().cpu()
        return rows

    rows = checkpoint_tensor_rows(
        model,
        ["lm_head.weight", "model.embed_tokens.weight"],
        token_ids,
    )
    if rows is not None:
        return rows
    raise ValueError("lm_head is on meta device and checkpoint rows could not be resolved")


def select_layers(model, layer_arg: str) -> list[int]:
    layers = get_layers(model)
    n_layers = len(layers)
    if str(layer_arg).lower() == "all":
        return list(range(n_layers))
    if str(layer_arg).lower() == "auto":
        raw = [0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1]
        return sorted(set(int(x) for x in raw if 0 <= int(x) < n_layers))
    out = []
    for item in parse_csv(layer_arg):
        idx = int(item)
        if 0 <= idx < n_layers:
            out.append(idx)
    return sorted(set(out))


def get_down_proj(layer):
    if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
        raise ValueError(f"Layer {type(layer).__name__} has no mlp.down_proj")
    return layer.mlp.down_proj


def get_down_proj_weight(model, layer_idx: int, layer) -> torch.Tensor:
    down_proj = get_down_proj(layer)
    weight = down_proj.weight
    if not getattr(weight, "is_meta", False):
        return weight.detach().float().cpu()
    weight_name = f"model.layers.{int(layer_idx)}.mlp.down_proj.weight"
    checkpoint_weight = checkpoint_tensor(model, weight_name)
    if checkpoint_weight is not None:
        return checkpoint_weight
    raise ValueError(f"{weight_name} is on meta device and checkpoint tensor could not be resolved")


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


def build_support_vectors(
    model,
    tokenizer,
    labels: list[str],
    selected_layers: list[int],
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[str, int]]:
    token_by_label = label_token_ids(tokenizer, labels)
    token_rows = get_lm_head_rows(model, sorted(set(token_by_label.values())))
    layers = get_layers(model)
    support: dict[tuple[int, str], torch.Tensor] = {}
    for layer_idx in selected_layers:
        weight = get_down_proj_weight(model, layer_idx, layers[layer_idx])  # [d_model, intermediate]
        for label, token_id in token_by_label.items():
            others = [tok for other_label, tok in token_by_label.items() if other_label != label]
            if not others:
                continue
            target_row = token_rows[token_id]
            other_mean = torch.stack([token_rows[tok] for tok in others]).mean(dim=0)
            direction = target_row - other_mean
            support[(int(layer_idx), label)] = torch.matmul(direction, weight)
        del weight
    return support, token_by_label


def init_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "object_set": set(),
        "template_set": set(),
        "activation_sum": 0.0,
        "activation_abs_sum": 0.0,
        "contribution_sum": 0.0,
        "contribution_abs_sum": 0.0,
        "contribution_sq_sum": 0.0,
        "positive_count": 0,
        "max_abs_contribution": 0.0,
        "max_sample_id": None,
    }


def update_stats(stats: dict[str, Any], sample: dict[str, Any], activation: float, contribution: float) -> None:
    stats["count"] += 1
    stats["object_set"].add(str(sample["object"]))
    stats["template_set"].add(str(sample["prompt_template"]))
    stats["activation_sum"] += float(activation)
    stats["activation_abs_sum"] += abs(float(activation))
    stats["contribution_sum"] += float(contribution)
    stats["contribution_abs_sum"] += abs(float(contribution))
    stats["contribution_sq_sum"] += float(contribution) * float(contribution)
    if contribution > 0:
        stats["positive_count"] += 1
    if abs(float(contribution)) > float(stats["max_abs_contribution"]):
        stats["max_abs_contribution"] = abs(float(contribution))
        stats["max_sample_id"] = str(sample["sample_id"])


def stats_to_row(
    model_name: str,
    layer_idx: int,
    channel_idx: int,
    label: str,
    stats: dict[str, Any],
    all_stats: dict[str, Any],
) -> dict[str, Any]:
    count = max(1, int(stats["count"]))
    all_count = max(1, int(all_stats["count"]))
    other_count = max(0, all_count - int(stats["count"]))
    mean_contribution = float(stats["contribution_sum"] / count)
    mean_abs_contribution = float(stats["contribution_abs_sum"] / count)
    mean_activation = float(stats["activation_sum"] / count)
    mean_abs_activation = float(stats["activation_abs_sum"] / count)
    if other_count > 0:
        other_mean_contribution = float((all_stats["contribution_sum"] - stats["contribution_sum"]) / other_count)
        other_mean_abs_activation = float((all_stats["activation_abs_sum"] - stats["activation_abs_sum"]) / other_count)
    else:
        other_mean_contribution = 0.0
        other_mean_abs_activation = 0.0
    selectivity_delta = mean_contribution - other_mean_contribution
    activation_selectivity = mean_abs_activation - other_mean_abs_activation
    positive_rate = float(stats["positive_count"] / count)
    coverage_objects = len(stats["object_set"])
    coverage_templates = len(stats["template_set"])
    effective_score = (
        mean_contribution
        + selectivity_delta
        + 0.05 * mean_abs_contribution
        + 0.02 * coverage_objects
        + 0.01 * coverage_templates
    )
    return {
        "phase": PHASE,
        "model": model_name,
        "relation": "color",
        "target_label": label,
        "layer": int(layer_idx),
        "channel": int(channel_idx),
        "count": int(stats["count"]),
        "coverage_objects": int(coverage_objects),
        "coverage_templates": int(coverage_templates),
        "mean_activation": mean_activation,
        "mean_abs_activation": mean_abs_activation,
        "activation_selectivity": activation_selectivity,
        "mean_contribution": mean_contribution,
        "mean_abs_contribution": mean_abs_contribution,
        "other_mean_contribution": other_mean_contribution,
        "selectivity_delta": selectivity_delta,
        "positive_rate": positive_rate,
        "max_abs_contribution": float(stats["max_abs_contribution"]),
        "max_sample_id": stats["max_sample_id"],
        "effective_score": float(effective_score),
        "evidence_level": "readout_channel_proxy_not_causal",
    }


def scan_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = build_color_samples(args)
    labels = color_labels(samples)
    dry_payload = {
        "phase": PHASE,
        "title": "Color Feature Neuron Atlas",
        "model": args.model,
        "sample_count": len(samples),
        "labels": labels,
        "objects": len({row["object"] for row in samples}),
        "domains": sorted({row["domain"] for row in samples}),
        "templates_per_object": int(args.templates_per_object),
        "schema": {
            "dataset_jsonl": "prompt-level generated samples",
            "sample_rows_jsonl": "one row per prompt with color logits and blockers",
            "channel_rows_jsonl": "one row per model/layer/channel/color with activation-readout statistics",
            "intervention_rows_jsonl": "optional top-channel-group causal scale validation",
        },
    }
    if args.generate_only or args.dry_run:
        payload = {**dry_payload, "status": "generate_only" if args.generate_only else "dry_run", "samples_preview": samples[:20]}
        write_json(out_dir / f"phase941_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase941_{args.model}_dataset.jsonl", samples)
        write_jsonl(out_dir / f"phase941_{args.model}_sample_rows.jsonl", [])
        write_jsonl(out_dir / f"phase941_{args.model}_channel_rows.jsonl", [])
        write_jsonl(out_dir / f"phase941_{args.model}_intervention_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    sample_rows: list[dict[str, Any]] = []
    channel_sample_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, _device = load_model_bf16(args.model)
        device = get_device_for_input(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        selected_layers = select_layers(model, args.layers)
        support_vectors, token_by_label = build_support_vectors(model, tokenizer, labels, selected_layers)
        layers = get_layers(model)
        stats_by_key: dict[tuple[int, int, str], dict[str, Any]] = defaultdict(init_stats)
        all_stats_by_key: dict[tuple[int, int], dict[str, Any]] = defaultdict(init_stats)

        for start in range(0, len(samples), max(1, int(args.batch_size))):
            batch = samples[start : start + max(1, int(args.batch_size))]
            input_ids, attention_mask, last_pos = encode_batch(tokenizer, device, batch)
            batch_idx = torch.arange(input_ids.shape[0], device=device)
            captured: dict[int, torch.Tensor] = {}
            hooks = []

            def make_capture_hook(layer_idx: int):
                def hook(_module, module_input):
                    hidden = module_input[0]
                    idx = batch_idx.to(hidden.device)
                    pos = last_pos.to(hidden.device)
                    if hidden.dim() == 3:
                        captured[layer_idx] = hidden[idx, pos].detach().float().cpu()
                    elif hidden.dim() == 2:
                        captured[layer_idx] = hidden.detach().float().cpu()
                    return None

                return hook

            for layer_idx in selected_layers:
                hooks.append(get_down_proj(layers[layer_idx]).register_forward_pre_hook(make_capture_hook(layer_idx)))

            with torch.inference_mode():
                result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            logits = result.logits[batch_idx, last_pos].detach().float().cpu()
            for hook in hooks:
                hook.remove()

            for row_idx, sample in enumerate(batch):
                target_label = str(sample["target_label"])
                label_scores = {label: label_score(tokenizer, logits[row_idx], label) for label in labels}
                target_info = label_scores.get(target_label) or {}
                competitor_items = [
                    (label, score)
                    for label, score in label_scores.items()
                    if label != target_label and score.get("logit") is not None
                ]
                best_competitor_label = None
                best_competitor_logit = None
                if competitor_items:
                    best_competitor_label, best_comp = max(competitor_items, key=lambda item: float(item[1]["logit"]))
                    best_competitor_logit = float(best_comp["logit"])
                target_logit = target_info.get("logit")
                color_margin = None
                if target_logit is not None and best_competitor_logit is not None:
                    color_margin = float(target_logit) - float(best_competitor_logit)
                top_values, top_ids = torch.topk(logits[row_idx], k=min(int(args.topk_blockers), logits.shape[-1]))
                sample_rows.append(
                    {
                        **sample,
                        "model": args.model,
                        "target_token_id": target_info.get("token_id"),
                        "target_token": target_info.get("token"),
                        "target_logit": target_logit,
                        "target_rank": target_info.get("rank"),
                        "best_competitor_label": best_competitor_label,
                        "best_competitor_logit": best_competitor_logit,
                        "color_margin": color_margin,
                        "top_tokens": [
                            {
                                "rank": int(i + 1),
                                "token_id": int(token_id),
                                "token": decode_token(tokenizer, int(token_id)),
                                "logit": float(value),
                            }
                            for i, (token_id, value) in enumerate(zip(top_ids.tolist(), top_values.tolist()))
                        ],
                    }
                )

                for layer_idx, activations in captured.items():
                    act_vec = activations[row_idx]
                    support = support_vectors.get((int(layer_idx), target_label))
                    if support is None:
                        continue
                    contributions = act_vec * support
                    topn = min(int(args.keep_top_channels_per_sample), int(contributions.numel()))
                    _, top_channel_ids = torch.topk(torch.abs(contributions), k=topn)
                    for channel_id_tensor in top_channel_ids:
                        channel_id = int(channel_id_tensor.item())
                        activation = float(act_vec[channel_id].item())
                        contribution = float(contributions[channel_id].item())
                        key = (int(layer_idx), channel_id, target_label)
                        all_key = (int(layer_idx), channel_id)
                        update_stats(stats_by_key[key], sample, activation, contribution)
                        update_stats(all_stats_by_key[all_key], sample, activation, contribution)
                        channel_sample_rows.append(
                            {
                                "phase": PHASE,
                                "model": args.model,
                                "sample_id": sample["sample_id"],
                                "target_label": target_label,
                                "domain": sample["domain"],
                                "object": sample["object"],
                                "layer": int(layer_idx),
                                "channel": channel_id,
                                "activation": activation,
                                "readout_support": float(support[channel_id].item()),
                                "contribution": contribution,
                            }
                        )

            del result, logits, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if int(args.log_every) > 0 and (start // max(1, int(args.batch_size))) % int(args.log_every) == 0:
                log(f"{args.model}: processed {min(start + len(batch), len(samples))}/{len(samples)} samples")

        channel_rows = []
        for (layer_idx, channel_idx, label), stats in stats_by_key.items():
            channel_rows.append(
                stats_to_row(args.model, layer_idx, channel_idx, label, stats, all_stats_by_key[(layer_idx, channel_idx)])
            )
        channel_rows.sort(key=lambda row: float(row["effective_score"]), reverse=True)
        keep_rows = int(args.keep_channel_rows)
        if keep_rows > 0:
            channel_rows = channel_rows[:keep_rows]
        intervention_rows = run_interventions(args, model, tokenizer, device, samples, labels, channel_rows)
    finally:
        if model is not None:
            release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target_ranks = [row.get("target_rank") for row in sample_rows if row.get("target_rank") is not None]
    margins = [row.get("color_margin") for row in sample_rows if row.get("color_margin") is not None]
    top_by_label: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        top_by_label[label] = [row for row in channel_rows if row["target_label"] == label][: int(args.summary_top_channels)]
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selected_layers": sorted({row["layer"] for row in channel_rows}),
        "sample_rows": len(sample_rows),
        "channel_sample_rows_total": len(channel_sample_rows),
        "channel_sample_rows_saved": min(len(channel_sample_rows), int(args.keep_channel_sample_rows)),
        "channel_rows": len(channel_rows),
        "intervention_rows": len(intervention_rows),
        "target_rank_mean": mean([float(x) for x in target_ranks]),
        "target_rank_top1": sum(1 for x in target_ranks if int(x) == 1),
        "target_rank_top10": sum(1 for x in target_ranks if int(x) <= 10),
        "color_margin_mean": mean([float(x) for x in margins]),
        "top_channels_by_label": top_by_label,
        "boundary": "feature-neuron atlas uses activation-readout proxy; causal rows are optional validation, not natural-gate closure",
    }
    write_json(out_dir / f"phase941_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase941_{args.model}_dataset.jsonl", samples)
    write_jsonl(out_dir / f"phase941_{args.model}_sample_rows.jsonl", sample_rows)
    write_jsonl(out_dir / f"phase941_{args.model}_channel_sample_rows.jsonl", channel_sample_rows[: int(args.keep_channel_sample_rows)])
    write_jsonl(out_dir / f"phase941_{args.model}_channel_rows.jsonl", channel_rows)
    write_jsonl(out_dir / f"phase941_{args.model}_intervention_rows.jsonl", intervention_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "sample_rows": len(sample_rows),
                "channel_rows": len(channel_rows),
                "top_labels": {label: top_by_label[label][:3] for label in labels},
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def run_interventions(
    args: argparse.Namespace,
    model,
    tokenizer,
    device: torch.device,
    samples: list[dict[str, Any]],
    labels: list[str],
    channel_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not args.run_interventions:
        return []
    factors = [float(x) for x in parse_csv(args.intervention_factors)]
    if not factors:
        return []
    layers = get_layers(model)
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in channel_rows:
        key = (str(row["target_label"]), int(row["layer"]))
        if len(grouped[key]) < int(args.intervention_top_channels):
            grouped[key].append(int(row["channel"]))
    selected_specs = [
        {"target_label": label, "layer": layer_idx, "channels": channels}
        for (label, layer_idx), channels in grouped.items()
        if channels
    ][: int(args.max_intervention_specs)]
    if not selected_specs:
        return []

    for spec in selected_specs:
        target_samples = [row for row in samples if str(row["target_label"]) == spec["target_label"]]
        target_samples = target_samples[: int(args.max_intervention_samples_per_spec)]
        if not target_samples:
            continue
        for factor in factors:
            patched_logits = forward_with_channel_scale(
                model,
                tokenizer,
                device,
                target_samples,
                int(spec["layer"]),
                spec["channels"],
                factor,
                int(args.batch_size),
            )
            base_logits = forward_with_channel_scale(
                model,
                tokenizer,
                device,
                target_samples,
                int(spec["layer"]),
                spec["channels"],
                1.0,
                int(args.batch_size),
            )
            for sample in target_samples:
                sample_id = str(sample["sample_id"])
                base = color_margin_for_logits(tokenizer, base_logits[sample_id], labels, str(sample["target_label"]))
                patched = color_margin_for_logits(tokenizer, patched_logits[sample_id], labels, str(sample["target_label"]))
                rows.append(
                    {
                        "phase": PHASE,
                        "model": args.model,
                        "sample_id": sample_id,
                        "target_label": sample["target_label"],
                        "layer": int(spec["layer"]),
                        "channels": spec["channels"],
                        "factor": float(factor),
                        "base_margin": base.get("margin"),
                        "patched_margin": patched.get("margin"),
                        "margin_delta": None
                        if base.get("margin") is None or patched.get("margin") is None
                        else float(patched["margin"]) - float(base["margin"]),
                        "base_target_rank": base.get("target_rank"),
                        "patched_target_rank": patched.get("target_rank"),
                        "target_rank_improved": (
                            base.get("target_rank") is not None
                            and patched.get("target_rank") is not None
                            and int(patched["target_rank"]) < int(base["target_rank"])
                        ),
                        "evidence_level": "causal_channel_group_scale",
                    }
                )
    return rows


def forward_with_channel_scale(
    model,
    tokenizer,
    device: torch.device,
    samples: list[dict[str, Any]],
    layer_idx: int,
    channels: list[int],
    factor: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    layers = get_layers(model)
    target_down = get_down_proj(layers[layer_idx])
    for start in range(0, len(samples), max(1, int(batch_size))):
        batch = samples[start : start + max(1, int(batch_size))]
        input_ids, attention_mask, last_pos = encode_batch(tokenizer, device, batch)
        batch_idx = torch.arange(input_ids.shape[0], device=device)

        def scale_hook(_module, module_input):
            hidden = module_input[0]
            patched = hidden.clone()
            idx = batch_idx.to(hidden.device)
            pos = last_pos.to(hidden.device)
            channel_tensor = torch.tensor(channels, dtype=torch.long, device=hidden.device)
            if patched.dim() == 3:
                patched[idx[:, None], pos[:, None], channel_tensor[None, :]] = (
                    patched[idx[:, None], pos[:, None], channel_tensor[None, :]] * float(factor)
                )
            elif patched.dim() == 2:
                patched[:, channel_tensor] = patched[:, channel_tensor] * float(factor)
            return (patched,)

        hook = target_down.register_forward_pre_hook(scale_hook)
        with torch.inference_mode():
            result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        hook.remove()
        logits = result.logits[batch_idx, last_pos].detach().float().cpu()
        for idx, sample in enumerate(batch):
            out[str(sample["sample_id"])] = logits[idx]
        del result, logits, input_ids, attention_mask
    return out


def color_margin_for_logits(tokenizer, logits: torch.Tensor, labels: list[str], target_label: str) -> dict[str, Any]:
    scores = {label: label_score(tokenizer, logits, label) for label in labels}
    target = scores.get(target_label) or {}
    competitors = [(label, row) for label, row in scores.items() if label != target_label and row.get("logit") is not None]
    best_label = None
    best_logit = None
    if competitors:
        best_label, best_row = max(competitors, key=lambda item: float(item[1]["logit"]))
        best_logit = float(best_row["logit"])
    margin = None
    if target.get("logit") is not None and best_logit is not None:
        margin = float(target["logit"]) - best_logit
    return {
        "target_rank": target.get("rank"),
        "target_logit": target.get("logit"),
        "best_competitor_label": best_label,
        "best_competitor_logit": best_logit,
        "margin": margin,
    }


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase941_{model}_summary.json") for model in MODELS]
    summaries = [row for row in summaries if row]
    all_channel_rows: list[dict[str, Any]] = []
    for model in MODELS:
        all_channel_rows.extend(read_jsonl(out_dir / f"phase941_{model}_channel_rows.jsonl"))
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_channel_rows:
        by_label[str(row.get("target_label"))].append(row)
    top_by_label = {}
    for label, rows in by_label.items():
        rows.sort(key=lambda row: float(row.get("effective_score") or 0.0), reverse=True)
        top_by_label[label] = rows[:50]
    payload = {
        "phase": PHASE,
        "title": "Color Feature Neuron Atlas Cross-Model Summary",
        "status": "complete" if summaries else "missing",
        "round_name": round_name,
        "models": [row.get("model") for row in summaries],
        "evidence_counts": Counter(str(row.get("status")) for row in summaries),
        "top_channels_by_label": top_by_label,
        "model_summaries": summaries,
        "boundary": "cross-model channel ids are not expected to match exactly; compare layer bands, signs, and response patterns",
    }
    write_json(out_dir / "phase941_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase941_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase941 Color Feature Neuron Atlas",
        "",
        f"status: {payload.get('status')}",
        f"round: {payload.get('round_name')}",
        "",
        "## Evidence Counts",
    ]
    for key, value in sorted((payload.get("evidence_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Channels By Color"])
    for label, rows in (payload.get("top_channels_by_label") or {}).items():
        lines.append(f"### {label}")
        for row in rows[:10]:
            lines.append(
                "- "
                f"{row.get('model')} L{row.get('layer')}C{row.get('channel')} "
                f"score={finite(row.get('effective_score')):.4f} "
                f"mean_contrib={finite(row.get('mean_contribution')):.4f} "
                f"selectivity={finite(row.get('selectivity_delta')):.4f} "
                f"coverage={row.get('coverage_objects')}/{row.get('coverage_templates')}"
            )
    lines.extend(["", "## Boundary", str(payload.get("boundary") or "")])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="qwen3")
    parser.add_argument("--round-name", default="color_feature_neuron_atlas")
    parser.add_argument("--colors", default="")
    parser.add_argument("--domains", default="")
    parser.add_argument("--templates-per-object", type=int, default=4)
    parser.add_argument("--max-objects-per-color", type=int, default=0)
    parser.add_argument("--layers", default="auto", help="auto, all, or comma-separated layer indices")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--topk-blockers", type=int, default=16)
    parser.add_argument("--keep-top-channels-per-sample", type=int, default=128)
    parser.add_argument("--keep-channel-rows", type=int, default=20000)
    parser.add_argument("--keep-channel-sample-rows", type=int, default=50000)
    parser.add_argument("--summary-top-channels", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-interventions", action="store_true")
    parser.add_argument("--intervention-top-channels", type=int, default=32)
    parser.add_argument("--max-intervention-specs", type=int, default=30)
    parser.add_argument("--max-intervention-samples-per-spec", type=int, default=12)
    parser.add_argument("--intervention-factors", default="0.0,0.5,1.5,2.0")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    scan_model(args)


if __name__ == "__main__":
    main()
