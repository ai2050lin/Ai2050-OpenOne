#!/usr/bin/env python3
"""
Phase 655: Final Token Policy Decomposition and Format Prior Gate Audit.

Offline analysis over Phase 654 rows. It decomposes bridge failures into
competing final-token policy groups: space, newline, explanation, word,
punctuation, symbol, and correct_prefix.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


IN_ROOT = Path("results/glm5_phase654_support_generation_bridge_policy_gate_audit")
OUT_ROOT = Path("results/glm5_phase655_final_token_policy_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]
POLICY_GROUPS = ["space", "newline", "explanation", "word", "punctuation", "symbol", "correct_prefix"]


def load_model(model: str) -> Dict:
    path = IN_ROOT / f"phase654_{model}_support_generation_bridge_policy_gate_audit_confirm.json"
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: List[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def row_group_margin(row: Dict, category: str) -> float | None:
    group = row.get("groups", {}).get(category)
    if not group:
        return None
    return group.get("prefix_minus_group_max")


def summarize_rows(rows: List[Dict]) -> Dict:
    by_mode: Dict[Tuple, Dict] = {}
    failure_rows = []
    for row in rows:
        key = (row["pair_task"], row["eval_task"], row["mode"], row.get("direction"), row.get("site"))
        item = by_mode.setdefault(key, {
            "pair_task": row["pair_task"],
            "eval_task": row["eval_task"],
            "mode": row["mode"],
            "kind": row["kind"],
            "direction": row.get("direction"),
            "site": row.get("site"),
            "n": 0,
            "exact": 0,
            "tok0": 0,
            "support_no_gen": 0,
            "sum_rank": 0.0,
            "top0_category": {},
            "group_margins": {g: [] for g in POLICY_GROUPS},
            "winner_vs_prefix": {},
        })
        item["n"] += 1
        exact = bool(row["eval"]["exact_correct"])
        item["exact"] += int(exact)
        item["tok0"] += int(row["top0_id"] == row["prefix_id"])
        item["sum_rank"] += row["prefix_rank"]
        top0 = row["top0_category"]
        item["top0_category"][top0] = item["top0_category"].get(top0, 0) + 1
        if row["kind"] == "patch" and row["prefix_rank"] <= 15 and not exact:
            item["support_no_gen"] += 1
            failure_rows.append(row)
            item["winner_vs_prefix"][top0] = item["winner_vs_prefix"].get(top0, 0) + 1
        for group in POLICY_GROUPS:
            margin = row_group_margin(row, group)
            if margin is not None:
                item["group_margins"][group].append(float(margin))

    out = []
    for item in by_mode.values():
        n = max(1, item["n"])
        r = dict(item)
        r["mean_rank"] = item["sum_rank"] / n
        r["exact_rate"] = item["exact"] / n
        r["tok0_rate"] = item["tok0"] / n
        r["support_no_gen_rate"] = item["support_no_gen"] / n
        r["top0_category"] = dict(sorted(item["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        r["winner_vs_prefix"] = dict(sorted(item["winner_vs_prefix"].items(), key=lambda kv: kv[1], reverse=True))
        r["mean_prefix_minus_group"] = {
            g: mean(vals) for g, vals in item["group_margins"].items()
        }
        del r["group_margins"]
        out.append(r)

    failure_by_group: Dict[str, Dict] = {}
    for row in failure_rows:
        top0 = row["top0_category"]
        item = failure_by_group.setdefault(top0, {
            "top0_category": top0,
            "n": 0,
            "sum_rank": 0.0,
            "sum_margin_vs_top": 0.0,
            "tasks": {},
            "models": {},
            "sites": {},
            "examples": [],
        })
        item["n"] += 1
        item["sum_rank"] += row["prefix_rank"]
        item["sum_margin_vs_top"] += row["prefix_margin_vs_top"]
        item["tasks"][row["pair_task"]] = item["tasks"].get(row["pair_task"], 0) + 1
        item["sites"][row.get("site") or ""] = item["sites"].get(row.get("site") or "", 0) + 1
        if len(item["examples"]) < 8:
            item["examples"].append({
                "pair_task": row["pair_task"],
                "eval_task": row["eval_task"],
                "direction": row.get("direction"),
                "site": row.get("site"),
                "prefix_rank": row["prefix_rank"],
                "prefix_margin_vs_top": row["prefix_margin_vs_top"],
                "top0_text": row["top0_text_clean"],
                "gen_first_text": row.get("gen_first_text", ""),
                "generation_text": row.get("generation_text", "").replace("\n", "\\n")[:120],
            })
    failure_summary = []
    for item in failure_by_group.values():
        n = max(1, item["n"])
        r = dict(item)
        r["mean_rank"] = item["sum_rank"] / n
        r["mean_margin_vs_top"] = item["sum_margin_vs_top"] / n
        r["tasks"] = dict(sorted(item["tasks"].items(), key=lambda kv: kv[1], reverse=True))
        r["sites"] = dict(sorted(item["sites"].items(), key=lambda kv: kv[1], reverse=True))
        failure_summary.append(r)
    failure_summary.sort(key=lambda r: r["n"], reverse=True)
    out.sort(key=lambda r: (
        r["pair_task"],
        r["eval_task"],
        0 if r["kind"] == "baseline" else 1,
        r.get("direction") or "",
        r.get("site") or "",
    ))
    return {
        "by_mode": out,
        "failure_by_top0_category": failure_summary,
        "n_bridge_failures": len(failure_rows),
    }


def table_line(row: Dict) -> str:
    margins = row["mean_prefix_minus_group"]
    margin_text = ", ".join(
        f"{g}:{margins[g]:.2f}" for g in POLICY_GROUPS if margins.get(g) is not None
    )
    return (
        f"| {row['pair_task']} | {row['eval_task']} | {row.get('direction') or ''} | "
        f"{row.get('site') or ''} | {row['n']} | {row['mean_rank']:.2f} | "
        f"{row['exact']}/{row['n']} | {row['tok0']}/{row['n']} | "
        f"{row['support_no_gen']}/{row['n']} | {row['top0_category']} | "
        f"{row['winner_vs_prefix']} | {margin_text} |"
    )


def write_markdown(results: Dict[str, Dict]) -> None:
    lines = ["# Phase 655 Final Token Policy Decomposition\n"]
    lines.append(
        "离线分解 Phase 654 中 rank<=15 但 exact=false 的 bridge failures，"
        "查看 correct_prefix 被哪些 final-token policy groups 压过。\n"
    )
    for model, data in results.items():
        lines.append(f"## {model}\n")
        lines.append(f"- bridge_failures: {data['summary']['n_bridge_failures']}\n")
        lines.append("### By Mode\n")
        lines.append("| pair_task | eval_task | direction | site | n | mean_rank | exact | tok0 | support_no_gen | top0_category | winner_vs_prefix | mean prefix-minus-group |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|")
        for row in data["summary"]["by_mode"]:
            lines.append(table_line(row))
        lines.append("")
        lines.append("### Failure Categories\n")
        lines.append("| top0_category | n | mean_rank | mean_margin_vs_top | tasks | sites |")
        lines.append("|---|---:|---:|---:|---|---|")
        for row in data["summary"]["failure_by_top0_category"]:
            lines.append(
                f"| {row['top0_category']} | {row['n']} | {row['mean_rank']:.2f} | "
                f"{row['mean_margin_vs_top']:.3f} | {row['tasks']} | {row['sites']} |"
            )
        lines.append("")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "phase655_final_token_policy_decomposition_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results = {}
    for model in MODELS:
        source = load_model(model)
        summary = summarize_rows(source["rows"])
        results[model] = {
            "source_phase": 654,
            "model": model,
            "source_file": str(IN_ROOT / f"phase654_{model}_support_generation_bridge_policy_gate_audit_confirm.json"),
            "summary": summary,
        }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase655_final_token_policy_decomposition.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(results)
    print(f"Wrote {out}")
    print(f"Wrote {OUT_ROOT / 'phase655_final_token_policy_decomposition_summary.md'}")
    for model, data in results.items():
        print(model, "bridge_failures", data["summary"]["n_bridge_failures"])


if __name__ == "__main__":
    main()
