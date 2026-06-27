#!/usr/bin/env python3
"""Merge Phase 712 QK/V factor diagnostics back into Phase 711 atlas v0."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ATLAS_ROOT = Path("results/glm5_phase711_global_mechanism_atlas_v0")
QKV_ROOT = Path("results/glm5_phase712_qkv_factor_atlas_audit")
OUT_ROOT = QKV_ROOT


def classify_factor(qk: float, v: float, interaction: float) -> str:
    mags = {"qk_addressing": abs(qk), "v_content": abs(v), "qk_v_interaction": abs(interaction)}
    best, best_val = max(mags.items(), key=lambda x: x[1])
    second = sorted(mags.values(), reverse=True)[1]
    if best_val < 1e-9:
        return "weak_or_zero"
    if best_val >= 1.35 * max(second, 1e-12):
        return best
    return "mixed_coupled"


def load_channel_scores() -> dict[tuple[str, int, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for path in QKV_ROOT.glob("phase712_*_qkv_factor_channel_scores.json"):
        model = path.name.split("_")[1]
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = (model, int(row["layer"]), int(row["head"]), int(row["channel"]))
            out[key] = row
    return out


def head_summaries(channel_scores: dict[tuple[str, int, int, int], dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for (model, li, head, _ch), row in channel_scores.items():
        grouped[(model, li, head)].append(row)
    out = {}
    for key, rows in grouped.items():
        top = sorted(rows, key=lambda r: float(r.get("mean_direct_effect", 0.0)), reverse=True)[:8]
        qk = sum(float(r.get("mean_qk_direct", 0.0) or 0.0) for r in top)
        v = sum(float(r.get("mean_v_direct", 0.0) or 0.0) for r in top)
        interaction = sum(float(r.get("mean_interaction_direct", 0.0) or 0.0) for r in top)
        total = sum(float(r.get("mean_direct_effect", 0.0) or 0.0) for r in top)
        denom = sum(abs(float(r.get(name, 0.0) or 0.0)) for r in top for name in ["mean_qk_direct", "mean_v_direct", "mean_interaction_direct"])
        out[key] = {
            "qkv_scope": "head_top8_channels",
            "qkv_dominant_factor": classify_factor(qk, v, interaction),
            "qkv_sum_total_direct": total,
            "qkv_sum_qk_direct": qk,
            "qkv_sum_v_direct": v,
            "qkv_sum_interaction_direct": interaction,
            "qkv_abs_qk_share": 0.0 if denom == 0 else sum(abs(float(r.get("mean_qk_direct", 0.0) or 0.0)) for r in top) / denom,
            "qkv_abs_v_share": 0.0 if denom == 0 else sum(abs(float(r.get("mean_v_direct", 0.0) or 0.0)) for r in top) / denom,
            "qkv_abs_interaction_share": 0.0 if denom == 0 else sum(abs(float(r.get("mean_interaction_direct", 0.0) or 0.0)) for r in top) / denom,
        }
    return out


def channel_factor(row: dict[str, Any]) -> dict[str, Any]:
    qk = float(row.get("mean_qk_direct", 0.0) or 0.0)
    v = float(row.get("mean_v_direct", 0.0) or 0.0)
    interaction = float(row.get("mean_interaction_direct", 0.0) or 0.0)
    denom = abs(qk) + abs(v) + abs(interaction)
    return {
        "qkv_scope": "channel",
        "qkv_dominant_factor": classify_factor(qk, v, interaction),
        "qkv_sum_total_direct": float(row.get("mean_direct_effect", 0.0) or 0.0),
        "qkv_sum_qk_direct": qk,
        "qkv_sum_v_direct": v,
        "qkv_sum_interaction_direct": interaction,
        "qkv_abs_qk_share": 0.0 if denom == 0 else abs(qk) / denom,
        "qkv_abs_v_share": 0.0 if denom == 0 else abs(v) / denom,
        "qkv_abs_interaction_share": 0.0 if denom == 0 else abs(interaction) / denom,
    }


def main() -> None:
    atlas_path = ATLAS_ROOT / "phase711_atlas_units.jsonl"
    if not atlas_path.exists():
        raise FileNotFoundError(atlas_path)
    channel_scores = load_channel_scores()
    heads = head_summaries(channel_scores)
    rows = []
    for line in atlas_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        model = row["model"]
        li = int(row["layer"])
        head = int(row["head"]) if row.get("head") is not None else None
        ch = row.get("channel")
        qkv = None
        if row.get("unit_type") == "attention_channel" and head is not None and ch is not None:
            score = channel_scores.get((model, li, head, int(ch)))
            if score is not None:
                qkv = channel_factor(score)
        elif row.get("unit_type") == "attention_head" and head is not None:
            qkv = heads.get((model, li, head))
        row["phase712_qkv_factor"] = qkv
        rows.append(row)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / "phase712_atlas_units_with_qkv.jsonl"
    out_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    factor_counts = Counter(
        row["phase712_qkv_factor"]["qkv_dominant_factor"]
        for row in rows
        if row.get("phase712_qkv_factor")
    )
    by_model = defaultdict(Counter)
    for row in rows:
        qkv = row.get("phase712_qkv_factor")
        if qkv:
            by_model[row["model"]][qkv["qkv_dominant_factor"]] += 1
    payload = {
        "phase": 712,
        "title": "Atlas v0 with QK-V Factor Backfill",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_units": len(rows),
        "n_units_with_qkv": sum(1 for row in rows if row.get("phase712_qkv_factor")),
        "factor_counts": dict(factor_counts.most_common()),
        "factor_counts_by_model": {k: dict(v.most_common()) for k, v in sorted(by_model.items())},
        "output": str(out_path),
    }
    (OUT_ROOT / "phase712_atlas_qkv_backfill_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 712 Atlas QK-V Backfill",
        "",
        f"- generated: `{payload['timestamp']}`",
        f"- units: `{payload['n_units']}`",
        f"- units_with_qkv: `{payload['n_units_with_qkv']}`",
        "",
        "## Factor Counts",
        "",
    ]
    for key, val in payload["factor_counts"].items():
        lines.append(f"- {key}: `{val}`")
    lines.extend(["", "## By Model", ""])
    for model, counts in payload["factor_counts_by_model"].items():
        lines.append(f"- {model}: `{counts}`")
    (OUT_ROOT / "phase712_atlas_qkv_backfill_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
