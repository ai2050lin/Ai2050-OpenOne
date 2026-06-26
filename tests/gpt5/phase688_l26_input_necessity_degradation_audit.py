#!/usr/bin/env python3
"""
Phase 688: L26 Input Necessity and Degradation Audit.

Phase 687 showed that the repair-effective value-support state is already
present at L26 layer_input. This phase runs the reverse intervention:

target = terse_no_explain success
donor  = same-case short_only failure

If replacing a successful target state with the failed state breaks the
correct value top1, the site is not only sufficient in restore tests but also
closer to necessary in the natural successful path.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase683_prose_route_bias_source_decomposition import (  # noqa: E402
    expected_first_ids,
    expected_for,
    prompt_for,
    route_id_sets,
    select_base_cases,
    value_phrase,
)
from phase685_natural_value_readout_writer_localization import (  # noqa: E402
    SHORT_VARIANT,
    TERSE_VARIANT,
    select_paired_cases,
)
from phase687_l26_l27_value_support_state_decomposition import (  # noqa: E402
    COMPONENTS,
    capture_states,
    choose_donors,
    classify,
    model_layers,
    paired_case_metadata,
    random_same_norm,
    run_patched,
)


OUT_ROOT = Path("results/glm5_phase688_l26_input_necessity_degradation_audit")
CROSS_COMPONENTS = {"layer_input", "layer_out"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    meta = paired_case_metadata(case_map, paired_ids)
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    try:
        layers = model_layers(args.model, len(get_layers(model)))
        sites = [(li, comp) for li in layers for comp in COMPONENTS]
        cache: dict[str, dict[str, Any]] = {}

        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            short_prompt = prompt_for(case, SHORT_VARIANT)
            terse_prompt = prompt_for(case, TERSE_VARIANT)
            short_logits, short_states = capture_states(model, tokenizer, device, short_prompt, sites)
            terse_logits, terse_states = capture_states(model, tokenizer, device, terse_prompt, sites)
            cache[case_id] = {
                "short_states": short_states,
                "terse_states": terse_states,
                "short_diag": classify(short_logits, routes, expected_ids),
                "terse_diag": classify(terse_logits, routes, expected_ids),
                "routes": routes,
                "expected_ids": expected_ids,
                "terse_prompt": terse_prompt,
            }
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: cached {idx}/{len(paired_ids)} paired cases")

        for idx, case_id in enumerate(paired_ids, 1):
            cur = cache[case_id]
            short_states = cur["short_states"]
            terse_states = cur["terse_states"]
            short_diag = cur["short_diag"]
            terse_diag = cur["terse_diag"]

            for li in layers:
                for component in COMPONENTS:
                    site = (li, component)
                    if site not in short_states or site not in terse_states:
                        continue
                    delta = terse_states[site] - short_states[site]
                    conditions = [
                        ("same_case_replace_short", short_states[site]),
                        ("same_case_remove_delta", terse_states[site] - delta),
                        ("random_same_norm_add", terse_states[site] + random_same_norm(delta, seed=idx * 2017 + li * 31)),
                    ]
                    for mode, new_vec in conditions:
                        patched = run_patched(
                            model,
                            tokenizer,
                            device,
                            cur["terse_prompt"],
                            [{"layer": li, "component": component, "new_vec": new_vec}],
                            cur["routes"],
                            cur["expected_ids"],
                        )
                        rows.append(make_row(meta, case_id, "same_case", mode, f"L{li}_{component}", None, short_diag, terse_diag, patched))

            donors = choose_donors(meta, case_id)
            for donor_kind in ["same_value", "same_relation_diff_value", "same_family_diff_value", "unrelated"]:
                donor_id = donors.get(donor_kind)
                if donor_id is None or donor_id not in cache:
                    continue
                donor = cache[donor_id]
                for li in layers:
                    for component in CROSS_COMPONENTS:
                        site = (li, component)
                        if site not in terse_states or site not in donor["short_states"] or site not in donor["terse_states"]:
                            continue
                        donor_delta = donor["terse_states"][site] - donor["short_states"][site]
                        conditions = [
                            (f"{donor_kind}_short_replace", donor["short_states"][site]),
                            (f"{donor_kind}_remove_donor_delta", terse_states[site] - donor_delta),
                        ]
                        for mode, new_vec in conditions:
                            patched = run_patched(
                                model,
                                tokenizer,
                                device,
                                cur["terse_prompt"],
                                [{"layer": li, "component": component, "new_vec": new_vec}],
                                cur["routes"],
                                cur["expected_ids"],
                            )
                            rows.append(make_row(meta, case_id, "cross_donor", mode, f"L{li}_{component}", donor_id, short_diag, terse_diag, patched))

            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: patched {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, paired_ids, rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase688_{args.model}_necessity_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 688,
        "title": "L26 Input Necessity and Degradation Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "layers": layers,
        "n_paired_cases": len(paired_ids),
        "n_rows": len(rows),
        "summary": summary,
    }
    (OUT_ROOT / f"phase688_{args.model}_necessity_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def make_row(meta, case_id, kind, mode, site, donor_id, short_diag, terse_diag, patched):
    terse_top1 = bool(terse_diag["expected_top1"])
    patched_top1 = bool(patched["expected_top1"])
    return {
        "case_id": case_id,
        "family": meta[case_id]["family"],
        "relation": meta[case_id]["relation"],
        "value": meta[case_id]["value"],
        "kind": kind,
        "mode": mode,
        "site": site,
        "donor_id": donor_id,
        "donor_family": meta[donor_id]["family"] if donor_id else None,
        "donor_relation": meta[donor_id]["relation"] if donor_id else None,
        "donor_value": meta[donor_id]["value"] if donor_id else None,
        "short_rank": short_diag["expected_rank"],
        "terse_rank": terse_diag["expected_rank"],
        "patched_rank": patched["expected_rank"],
        "short_top1": short_diag["expected_top1"],
        "terse_top1": terse_top1,
        "patched_top1": patched_top1,
        "drop_from_success": terse_top1 and not patched_top1,
        "rank_increase_from_terse": patched["expected_rank"] - terse_diag["expected_rank"],
        "pmv_increase_from_terse": patched["prose_minus_value"] - terse_diag["prose_minus_value"],
        "short_pmv": short_diag["prose_minus_value"],
        "terse_pmv": terse_diag["prose_minus_value"],
        "patched_pmv": patched["prose_minus_value"],
        "patched_best_other_route": patched["best_other_route"],
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": n,
        "baseline_terse_top1_rate": sum(1 for r in rows if r["terse_top1"]) / n,
        "patched_top1_rate": sum(1 for r in rows if r["patched_top1"]) / n,
        "drop_rate": sum(1 for r in rows if r["drop_from_success"]) / n,
        "mean_short_rank": sum(r["short_rank"] for r in rows) / n,
        "mean_terse_rank": sum(r["terse_rank"] for r in rows) / n,
        "mean_patched_rank": sum(r["patched_rank"] for r in rows) / n,
        "mean_rank_increase_from_terse": sum(r["rank_increase_from_terse"] for r in rows) / n,
        "mean_short_pmv": sum(r["short_pmv"] for r in rows) / n,
        "mean_terse_pmv": sum(r["terse_pmv"] for r in rows) / n,
        "mean_patched_pmv": sum(r["patched_pmv"] for r in rows) / n,
        "mean_pmv_increase_from_terse": sum(r["pmv_increase_from_terse"] for r in rows) / n,
        "patched_best_other_route": dict(Counter(r["patched_best_other_route"] for r in rows).most_common()),
    }


def summarize_model(model_name: str, paired_ids: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["kind"], r["mode"], r["site"])].append(r)
    by_condition = {f"{k}|{m}|{s}": summarize_group(v) for (k, m, s), v in grouped.items()}
    same_rows = [r for r in rows if r["kind"] == "same_case"]
    cross_rows = [r for r in rows if r["kind"] == "cross_donor"]
    strongest_drops = sorted(
        by_condition.items(),
        key=lambda kv: (kv[1].get("drop_rate", 0.0), kv[1].get("mean_rank_increase_from_terse", 0.0), kv[1].get("mean_pmv_increase_from_terse", 0.0)),
        reverse=True,
    )[:24]
    random_controls = {
        k: v for k, v in by_condition.items()
        if "|random_same_norm_add|" in k
    }
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "by_kind": {
            "same_case": summarize_group(same_rows),
            "cross_donor": summarize_group(cross_rows),
        },
        "by_condition": by_condition,
        "strongest_drop_conditions": [{"condition": k, **v} for k, v in strongest_drops],
        "random_controls": random_controls,
    }


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase688_*_necessity_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 688,
        "title": "L26 Input Necessity and Degradation Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase688_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 688 L26 Input Necessity and Degradation Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | layers | strongest_drop | drop | patched_rank | rank_inc | pmv_inc | best_other |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for item in models:
        strongest = item["summary"]["strongest_drop_conditions"][0] if item["summary"]["strongest_drop_conditions"] else {}
        best_other = strongest.get("patched_best_other_route", {})
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | {item['layers']} | "
            f"{strongest.get('condition', '')} | {strongest.get('drop_rate', 0.0):.3f} | "
            f"{strongest.get('mean_patched_rank', 0.0):.2f} | {strongest.get('mean_rank_increase_from_terse', 0.0):.2f} | "
            f"{strongest.get('mean_pmv_increase_from_terse', 0.0):.3f} | {best_other} |"
        )
    lines.extend(["", "## Strongest Drop Conditions", ""])
    for item in models:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append("| condition | drop | patched_top1 | patched_rank | rank_inc | patched_pmv | pmv_inc | best_other |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for row in item["summary"]["strongest_drop_conditions"][:18]:
            lines.append(
                f"| {row['condition']} | {row['drop_rate']:.3f} | {row['patched_top1_rate']:.3f} | "
                f"{row['mean_patched_rank']:.2f} | {row['mean_rank_increase_from_terse']:.2f} | "
                f"{row['mean_patched_pmv']:.3f} | {row['mean_pmv_increase_from_terse']:.3f} | "
                f"{row['patched_best_other_route']} |"
            )
        lines.append("")
    (OUT_ROOT / "phase688_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=12)
    args = parser.parse_args()
    if args.summarize_only:
        write_cross_summary()
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
