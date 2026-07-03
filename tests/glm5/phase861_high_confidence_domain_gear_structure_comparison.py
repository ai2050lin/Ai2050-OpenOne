#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS  # noqa: E402


PHASE = 861
MODELS = ("qwen3", "glm4", "deepseek7b")
SOURCE_ROOT = Path("tests/result/phase860_replicated_domain_gear_evidence_ladder")
RESULT_ROOT = Path("tests/result/phase861_high_confidence_domain_gear_structure_comparison")
GEAR_RE = re.compile(r"^L(?P<layer>\d+)C(?P<channel>\d+)$")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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
    if math.isfinite(out):
        return out
    return default


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def cosine(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def model_layer_count(model_name: str) -> int | None:
    cfg_path = Path(MODEL_CONFIGS[model_name]["path"]) / "config.json"
    if not cfg_path.exists():
        return None
    cfg = read_json(cfg_path)
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if key in cfg:
            return int(cfg[key])
    return None


def parse_gear_key(key: str) -> dict[str, Any]:
    match = GEAR_RE.match(key)
    if not match:
        return {"gear_key": key, "layer": None, "channel": None}
    return {
        "gear_key": key,
        "layer": int(match.group("layer")),
        "channel": int(match.group("channel")),
    }


def depth_band(norm_layer: float | None) -> str:
    if norm_layer is None:
        return "unknown"
    if norm_layer < 0.33:
        return "early"
    if norm_layer < 0.66:
        return "middle"
    return "late"


def vector_from_map(data: dict[str, Any], keys: list[str]) -> list[float]:
    return [finite(data.get(key)) for key in keys]


def find_effect(summary: dict[str, Any], section: str, domain: str) -> dict[str, Any] | None:
    for row in summary.get(section) or []:
        if str(row.get("domain")) == domain:
            return row
    return None


def gear_signature(
    model_name: str,
    domain: str,
    summary: dict[str, Any],
    n_layers: int | None,
    min_level: int,
) -> dict[str, Any] | None:
    ladder = (summary.get("evidence_ladder") or {}).get(domain)
    if not ladder or int(ladder.get("level") or 0) < min_level:
        return None
    best = find_effect(summary, "best_effects", domain)
    if not best:
        return None
    alternate = find_effect(summary, "alternate_effects", domain) or {}
    control = find_effect(summary, "same_layer_control_effects", domain) or {}
    parsed = [parse_gear_key(str(key)) for key in best.get("gear_keys") or []]
    layers = [int(item["layer"]) for item in parsed if item.get("layer") is not None]
    channels = [int(item["channel"]) for item in parsed if item.get("channel") is not None]
    norm_layers = [
        layer / max(1, int(n_layers) - 1)
        for layer in layers
        if n_layers is not None and int(n_layers) > 1
    ]
    mean_norm_layer = mean(norm_layers)
    split_keys = ["phase858_seen", "phase859_holdout_seen", "new_replication"]
    prompt_keys = ["natural_question", "natural_category", "classification"]
    split_vector = vector_from_map(best.get("split_clear_gain") or {}, split_keys)
    prompt_vector = vector_from_map(best.get("prompt_clear_gain") or {}, prompt_keys)
    best_clear = int(best.get("clear_rollout_gain") or 0)
    alternate_clear = int(alternate.get("clear_rollout_gain") or 0)
    control_clear = int(control.get("clear_rollout_gain") or 0)
    return {
        "model": model_name,
        "domain": domain,
        "level": int(ladder.get("level") or 0),
        "label": ladder.get("label"),
        "gear_keys": best.get("gear_keys") or [],
        "parsed_gears": parsed,
        "gear_count": len(parsed),
        "layers": layers,
        "channels": channels,
        "n_layers": n_layers,
        "layer_min": min(layers) if layers else None,
        "layer_max": max(layers) if layers else None,
        "layer_span": (max(layers) - min(layers)) if len(layers) >= 2 else 0,
        "mean_norm_layer": mean_norm_layer,
        "depth_band": depth_band(mean_norm_layer),
        "candidate_role": best.get("candidate_role"),
        "best_mode": best.get("mode"),
        "alternate_mode": alternate.get("mode"),
        "best_clear_gain": best_clear,
        "best_clear_loss": int(best.get("clear_rollout_loss") or 0),
        "alternate_clear_gain": alternate_clear,
        "alternate_clear_loss": int(alternate.get("clear_rollout_loss") or 0),
        "same_layer_control_clear_gain": control_clear,
        "same_layer_control_clear_loss": int(control.get("clear_rollout_loss") or 0),
        "first_gain": int(best.get("first_gain") or 0),
        "rollout_gain": int(best.get("rollout_gain") or 0),
        "mean_blocker_reduction": finite(best.get("mean_blocker_reduction")),
        "mean_class_minus_object_gain": finite(best.get("mean_class_minus_object_gain")),
        "split_keys": split_keys,
        "split_vector": split_vector,
        "split_hits": sum(1 for x in split_vector if x > 0),
        "prompt_keys": prompt_keys,
        "prompt_vector": prompt_vector,
        "prompt_hits": sum(1 for x in prompt_vector if x > 0),
        "new_replication_clear_gain": int((best.get("split_clear_gain") or {}).get("new_replication") or 0),
        "sign_ambiguity": alternate_clear > 0,
        "alternate_to_best_ratio": alternate_clear / best_clear if best_clear > 0 else None,
        "control_separation": best_clear - control_clear,
        "reasons": ladder.get("reasons") or [],
    }


def compare_signatures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            comparisons.append(
                {
                    "left": f"{left['model']}:{left['domain']}",
                    "right": f"{right['model']}:{right['domain']}",
                    "same_model": left["model"] == right["model"],
                    "same_domain": left["domain"] == right["domain"],
                    "same_depth_band": left["depth_band"] == right["depth_band"],
                    "mean_norm_layer_distance": (
                        abs(finite(left.get("mean_norm_layer")) - finite(right.get("mean_norm_layer")))
                        if left.get("mean_norm_layer") is not None and right.get("mean_norm_layer") is not None
                        else None
                    ),
                    "same_gear_count": left["gear_count"] == right["gear_count"],
                    "same_candidate_role": left["candidate_role"] == right["candidate_role"],
                    "same_best_mode": left["best_mode"] == right["best_mode"],
                    "both_sign_ambiguous": bool(left["sign_ambiguity"] and right["sign_ambiguity"]),
                    "split_cosine": cosine(left["split_vector"], right["split_vector"]),
                    "prompt_cosine": cosine(left["prompt_vector"], right["prompt_vector"]),
                    "both_control_zero": (
                        int(left["same_layer_control_clear_gain"]) == 0
                        and int(right["same_layer_control_clear_gain"]) == 0
                    ),
                }
            )
    return comparisons


def summarize(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]], min_level: int) -> dict[str, Any]:
    if not rows:
        return {
            "min_level": min_level,
            "n_signatures": 0,
            "finding": "no high-confidence signatures",
        }
    return {
        "min_level": min_level,
        "n_signatures": len(rows),
        "models": sorted({row["model"] for row in rows}),
        "domains": sorted({row["domain"] for row in rows}),
        "depth_band_counts": dict(Counter(row["depth_band"] for row in rows)),
        "gear_count_counts": dict(Counter(str(row["gear_count"]) for row in rows)),
        "candidate_role_counts": dict(Counter(str(row["candidate_role"]) for row in rows)),
        "best_mode_counts": dict(Counter(str(row["best_mode"]) for row in rows)),
        "sign_ambiguous_count": sum(1 for row in rows if row["sign_ambiguity"]),
        "control_zero_count": sum(1 for row in rows if int(row["same_layer_control_clear_gain"]) == 0),
        "new_replication_supported_count": sum(1 for row in rows if int(row["new_replication_clear_gain"]) > 0),
        "avg_mean_norm_layer": mean([finite(row.get("mean_norm_layer")) for row in rows if row.get("mean_norm_layer") is not None]),
        "avg_best_clear_gain": mean([finite(row.get("best_clear_gain")) for row in rows]),
        "avg_alternate_to_best_ratio": mean(
            [finite(row.get("alternate_to_best_ratio")) for row in rows if row.get("alternate_to_best_ratio") is not None]
        ),
        "pairwise_same_depth_band": sum(1 for row in comparisons if row["same_depth_band"]),
        "pairwise_same_role": sum(1 for row in comparisons if row["same_candidate_role"]),
        "pairwise_same_mode": sum(1 for row in comparisons if row["same_best_mode"]),
        "pairwise_both_control_zero": sum(1 for row in comparisons if row["both_control_zero"]),
        "pairwise_both_sign_ambiguous": sum(1 for row in comparisons if row["both_sign_ambiguous"]),
    }


def markdown(summary: dict[str, Any], rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Phase 861 High-Confidence Domain Gear Structure Comparison\n")
    lines.append(
        "- Source: Phase 860 replicated evidence ladder.\n"
        "- Boundary: offline structure comparison, not a new model intervention and not closure.\n"
    )
    lines.append("## Summary\n")
    for key in (
        "n_signatures",
        "models",
        "domains",
        "depth_band_counts",
        "gear_count_counts",
        "candidate_role_counts",
        "best_mode_counts",
        "sign_ambiguous_count",
        "control_zero_count",
        "new_replication_supported_count",
        "avg_mean_norm_layer",
        "avg_best_clear_gain",
        "avg_alternate_to_best_ratio",
    ):
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.append("\n## Signatures\n")
    lines.append(
        "| model | domain | level | gears | depth | role | mode | clear gain/loss | alt gain | control gain | split | prompt | new repl |"
    )
    lines.append("|---|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|")
    for row in rows:
        lines.append(
            "| {model} | {domain} | {level} | `{gears}` | {depth} | {role} | {mode} | {gain}/{loss} | {alt} | {control} | `{split}` | `{prompt}` | {new} |".format(
                model=row["model"],
                domain=row["domain"],
                level=row["level"],
                gears="+".join(row["gear_keys"]),
                depth=f"{row['depth_band']}:{row.get('mean_norm_layer'):.3f}" if row.get("mean_norm_layer") is not None else row["depth_band"],
                role=row["candidate_role"],
                mode=row["best_mode"],
                gain=row["best_clear_gain"],
                loss=row["best_clear_loss"],
                alt=row["alternate_clear_gain"],
                control=row["same_layer_control_clear_gain"],
                split=row["split_vector"],
                prompt=row["prompt_vector"],
                new=row["new_replication_clear_gain"],
            )
        )
    lines.append("\n## Pairwise Fingerprint Comparison\n")
    lines.append("| left | right | same depth | norm dist | same role | same mode | split cos | prompt cos | both control zero | both sign ambiguous |")
    lines.append("|---|---|---|---:|---|---|---:|---:|---|---|")
    for row in comparisons:
        lines.append(
            "| {left} | {right} | {depth} | {dist} | {role} | {mode} | {split} | {prompt} | {control} | {sign} |".format(
                left=row["left"],
                right=row["right"],
                depth=row["same_depth_band"],
                dist=f"{row['mean_norm_layer_distance']:.3f}" if row.get("mean_norm_layer_distance") is not None else "",
                role=row["same_candidate_role"],
                mode=row["same_best_mode"],
                split=f"{row['split_cosine']:.3f}" if row.get("split_cosine") is not None else "",
                prompt=f"{row['prompt_cosine']:.3f}" if row.get("prompt_cosine") is not None else "",
                control=row["both_control_zero"],
                sign=row["both_sign_ambiguous"],
            )
        )
    lines.append("\n## Conservative Reading\n")
    lines.append(
        "- The strongest common fingerprint is late-layer, two-channel, negative-blocker, flip-mode gear sets with zero same-layer control gain.\n"
        "- The same fingerprint appears in qwen3 material and DS7B animal/color, but this is structural similarity, not channel or semantic universality.\n"
        "- Alternate zero mode remains effective, so sign calibration is still open.\n"
        "- DS7B color lacks new-replication split gain in this round, so it is weaker than DS7B animal and qwen3 material on fresh-case replication.\n"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-round", default="replicate")
    parser.add_argument("--min-level", type=int, default=6)
    parser.add_argument("--output-dir", default=str(RESULT_ROOT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    signatures: list[dict[str, Any]] = []
    all_ladder: dict[str, Any] = {}

    for model_name in MODELS:
        source = SOURCE_ROOT / args.source_round / f"phase860_{model_name}_summary.json"
        if not source.exists():
            log(f"skip missing {source}")
            continue
        n_layers = model_layer_count(model_name)
        summary = read_json(source)
        all_ladder[model_name] = summary.get("evidence_ladder") or {}
        for domain in sorted((summary.get("evidence_ladder") or {}).keys()):
            sig = gear_signature(model_name, domain, summary, n_layers, args.min_level)
            if sig:
                signatures.append(sig)

    signatures.sort(key=lambda row: (row["model"], row["domain"]))
    comparisons = compare_signatures(signatures)
    summary = summarize(signatures, comparisons, args.min_level)
    payload = {
        "phase": PHASE,
        "title": "High-Confidence Domain Gear Structure Comparison",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_round": args.source_round,
        "source": str(SOURCE_ROOT / args.source_round),
        "min_level": args.min_level,
        "summary": summary,
        "signatures": signatures,
        "pairwise_comparisons": comparisons,
        "all_phase860_ladder": all_ladder,
        "boundary": "offline fingerprint comparison; no new model intervention; not language closure",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "phase861_summary.json", payload)
    write_jsonl(out_dir / "phase861_signatures.jsonl", signatures)
    write_jsonl(out_dir / "phase861_pairwise_comparisons.jsonl", comparisons)
    (out_dir / "phase861_summary.md").write_text(markdown(summary, signatures, comparisons), encoding="utf-8")
    log(f"wrote {out_dir / 'phase861_summary.md'}")


if __name__ == "__main__":
    main()
