from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def summarize_model(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary", {})
    probe_best = summary.get("probe_best", {})
    best_by_variable = summary.get("best_by_variable", {})
    return {
        "complete": data.get("complete"),
        "bases": data.get("num_bases"),
        "train_bases": data.get("num_train_bases"),
        "test_bases": data.get("num_test_bases"),
        "rows": data.get("num_results"),
        "nonfinite_rows": summary.get("nonfinite_rows"),
        "probe_best": probe_best,
        "best_by_variable": best_by_variable,
    }


def merge_model_shards(model: str, input_dir: Path) -> dict[str, Any] | None:
    files = sorted(input_dir.glob(f"{model}_phase302_passive_token_role_closure_test*.json"))
    if not files:
        direct = input_dir / f"{model}_phase302_passive_token_role_closure.json"
        return load_json(direct)
    merged: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    test_bases: list[str] = []
    shards = []
    for path in files:
        data = load_json(path)
        if not data:
            continue
        if merged is None:
            merged = {k: v for k, v in data.items() if k not in {"results", "summary", "test_bases"}}
        results.extend(data.get("results", []))
        test_bases.extend(data.get("test_bases", []))
        shards.append({
            "file": str(path),
            "shard_label": data.get("shard_label"),
            "test_start": data.get("test_start"),
            "test_end": data.get("test_end"),
            "rows": data.get("num_results"),
            "complete": data.get("complete"),
        })
    if merged is None:
        return None
    from phase302_passive_token_role_closure import summarize

    merged["complete"] = all(bool(item.get("complete")) for item in shards)
    merged["test_bases"] = test_bases
    merged["num_test_bases"] = len(test_bases)
    merged["num_results"] = len(results)
    merged["results"] = results
    merged["shards"] = shards
    merged["summary"] = summarize(results, merged.get("probe_rows", []))
    return merged


def render_markdown(results: dict[str, dict[str, Any]]) -> str:
    lines = ["# Phase 302 Passive Token Role Closure Summary", ""]
    for model, item in results.items():
        lines.append(f"## {model}")
        if not item:
            lines.extend(["- missing", ""])
            continue
        lines.append(f"- complete: {item.get('complete')}")
        lines.append(f"- bases/train/test: {item.get('bases')} / {item.get('train_bases')} / {item.get('test_bases')}")
        lines.append(f"- rows: {item.get('rows')}")
        lines.append(f"- nonfinite_rows: {item.get('nonfinite_rows')}")
        if item.get("shards"):
            lines.append(f"- shards: {len(item.get('shards', []))}")
        lines.append("- probe_best:")
        for variable, row in sorted(item.get("probe_best", {}).items()):
            lines.append(
                "  - {variable}: L{layer} {module} {token} acc={acc} margin={margin}".format(
                    variable=variable,
                    layer=row.get("layer"),
                    module=row.get("module"),
                    token=row.get("token_position"),
                    acc=fmt(row.get("test_accuracy")),
                    margin=fmt(row.get("mean_signed_margin")),
                )
            )
        lines.append("- best_by_variable:")
        for variable, row in sorted(item.get("best_by_variable", {}).items()):
            lines.append(
                "  - {variable}: {mode}/{pos} L{layer} {module} progress={progress} kl={kl} delta={delta}".format(
                    variable=variable,
                    mode=row.get("patch_mode"),
                    pos=row.get("source_position"),
                    layer=row.get("layer"),
                    module=row.get("module"),
                    progress=fmt(row.get("mean_progress")),
                    kl=fmt(row.get("mean_kl_ratio")),
                    delta=fmt(row.get("mean_logit_delta_ratio")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    merged_full: dict[str, Any] = {}
    for model in MODELS:
        data = merge_model_shards(model, input_dir)
        if data:
            merged_full[model] = data
        results[model] = summarize_model(data) if data else {}

    out_json = output_dir / "passive_token_role_summary.json"
    out_md = output_dir / "PASSIVE_TOKEN_ROLE_SUMMARY.md"
    out_merged = output_dir / "passive_token_role_merged.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_merged.write_text(json.dumps(merged_full, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(results), encoding="utf-8")
    print(f"saved {out_json}")
    print(f"saved {out_merged}")
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
