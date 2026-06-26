#!/usr/bin/env python3
"""
Phase 694: Boundary Head Source-Token Attention Audit.

Phase 693 found boundary attention head candidates but did not identify their
source-token path. This phase is observational: for the top candidate heads, it
measures answer-last attention mass to simple token groups:

record line, question line, instruction line, answer label/self, target value,
object name, and relation token.

This is not causal source-token patching. It is a source-path map used to decide
where the next causal graph audit should focus.
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

from model_utils import release_model  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase683_prose_route_bias_source_decomposition import (  # noqa: E402
    prompt_for,
    select_base_cases,
    value_phrase,
)
from phase685_natural_value_readout_writer_localization import (  # noqa: E402
    SHORT_VARIANT,
    TERSE_VARIANT,
    select_paired_cases,
)


OUT_ROOT = Path("results/glm5_phase694_boundary_head_source_token_attention_audit")
PHASE693_ROOT = Path("results/glm5_phase693_boundary_attention_head_candidate_audit")
GROUPS = [
    "record_line",
    "question_line",
    "instruction_line",
    "answer_line",
    "self_last",
    "target_value",
    "object_name",
    "relation",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_top_heads(model_name: str, top_k: int) -> list[dict[str, int]]:
    path = PHASE693_ROOT / f"phase693_{model_name}_candidate_scores.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Phase693 candidate scores: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [{"layer": int(r["layer"]), "head": int(r["head"]), "head_key": r["head_key"]} for r in rows[:top_k]]


def line_spans(prompt: str) -> dict[str, tuple[int, int]]:
    q = prompt.index("Question:")
    ins = prompt.index("Instruction:")
    ans = prompt.index("Answer:")
    return {
        "record_line": (0, q),
        "question_line": (q, ins),
        "instruction_line": (ins, ans),
        "answer_line": (ans, len(prompt)),
    }


def all_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    out = []
    start = 0
    while needle:
        idx = text.find(needle, start)
        if idx < 0:
            break
        out.append((idx, idx + len(needle)))
        start = idx + max(1, len(needle))
    return out


def positions_for_char_spans(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    pos = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        for s, e in spans:
            if b > s and a < e:
                pos.append(i)
                break
    return pos


def token_groups(tokenizer, prompt: str, case: dict[str, Any], ids: list[int]) -> dict[str, list[int]]:
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    groups: dict[str, list[int]] = {}
    for name, span in line_spans(prompt).items():
        groups[name] = positions_for_char_spans(offsets, [span])
    groups["self_last"] = [len(ids) - 1]
    groups["target_value"] = positions_for_char_spans(offsets, all_occurrences(prompt, value_phrase(case)))
    groups["object_name"] = positions_for_char_spans(offsets, all_occurrences(prompt, case["object_name"]))
    groups["relation"] = positions_for_char_spans(offsets, all_occurrences(prompt, case["relation"]))
    return groups


def run_attention(model, tokenizer, device, prompt: str):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([ids], device=device),
            return_dict=True,
            use_cache=False,
            output_attentions=True,
        )
    if out.attentions is None:
        raise RuntimeError("model returned no attentions")
    return ids, out.attentions


def group_mass(row: torch.Tensor, groups: dict[str, list[int]]) -> dict[str, float]:
    n = row.numel()
    result = {}
    for name in GROUPS:
        idxs = [i for i in groups.get(name, []) if 0 <= i < n]
        result[f"mass_{name}"] = float(row[idxs].sum().detach().float().cpu().item()) if idxs else 0.0
    line_union = set(groups["record_line"]) | set(groups["question_line"]) | set(groups["instruction_line"]) | set(groups["answer_line"])
    line_mass = sum(result[f"mass_{g}"] for g in ["record_line", "question_line", "instruction_line", "answer_line"])
    result["mass_line_partition"] = line_mass
    result["mass_outside_lines"] = max(0.0, 1.0 - line_mass)
    value_set = set(groups.get("target_value", []))
    result["target_value_in_record_mass"] = float(row[[i for i in value_set & set(groups["record_line"]) if 0 <= i < n]].sum().detach().float().cpu().item()) if value_set else 0.0
    return result


def make_rows_for_prompt(model, tokenizer, device, case, variant_name: str, prompt: str, heads: list[dict[str, int]]) -> list[dict[str, Any]]:
    ids, attentions = run_attention(model, tokenizer, device, prompt)
    groups = token_groups(tokenizer, prompt, case, ids)
    rows = []
    answer_pos = len(ids) - 1
    for h in heads:
        li = h["layer"]
        head = h["head"]
        if li >= len(attentions) or head >= attentions[li].shape[1]:
            continue
        row = attentions[li][0, head, answer_pos, :].detach()
        top_pos = int(torch.argmax(row).detach().cpu().item())
        top_text = tokenizer.decode([ids[top_pos]])
        masses = group_mass(row, groups)
        rows.append({
            "case_id": case["case_id"],
            "family": case["family"],
            "relation": case["relation"],
            "value": value_phrase(case),
            "variant": variant_name,
            "layer": li,
            "head": head,
            "head_key": h["head_key"],
            "seq_len": len(ids),
            "answer_pos": answer_pos,
            "top_attn_pos": top_pos,
            "top_attn_token": top_text,
            "top_attn_mass": float(row[top_pos].detach().float().cpu().item()),
            **masses,
        })
    del attentions
    return rows


def summarize_rows(model_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["variant"], r["head_key"])].append(r)
    by_head_variant = {}
    for (variant, head_key), vals in grouped.items():
        rec = {"n": len(vals)}
        for key in [k for k in vals[0] if k.startswith("mass_") or k == "target_value_in_record_mass"]:
            rec[f"mean_{key}"] = sum(v[key] for v in vals) / len(vals)
        rec["top_attn_token_counts"] = dict(Counter(v["top_attn_token"] for v in vals).most_common(8))
        by_head_variant[f"{variant}|{head_key}"] = rec

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)
    variant_summary = {}
    for variant, vals in by_variant.items():
        rec = {"n": len(vals)}
        for key in [k for k in vals[0] if k.startswith("mass_") or k == "target_value_in_record_mass"]:
            rec[f"mean_{key}"] = sum(v[key] for v in vals) / len(vals)
        rec["top_attn_token_counts"] = dict(Counter(v["top_attn_token"] for v in vals).most_common(12))
        variant_summary[variant] = rec

    return {
        "model": model_name,
        "n_rows": len(rows),
        "by_variant": variant_summary,
        "by_head_variant": by_head_variant,
        "heads_high_value_mass": sorted(
            [
                {"condition": k, **v}
                for k, v in by_head_variant.items()
            ],
            key=lambda x: x.get("mean_target_value_in_record_mass", 0.0),
            reverse=True,
        )[:24],
        "heads_high_instruction_mass": sorted(
            [{"condition": k, **v} for k, v in by_head_variant.items()],
            key=lambda x: x.get("mean_mass_instruction_line", 0.0),
            reverse=True,
        )[:24],
    }


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    heads = load_top_heads(args.model, args.top_heads)
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    try:
        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            for variant_name, variant in [("short_only", SHORT_VARIANT), ("terse_no_explain", TERSE_VARIANT)]:
                prompt = prompt_for(case, variant)
                rows.extend(make_rows_for_prompt(model, tokenizer, device, case, variant_name, prompt, heads))
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: source-attn audited {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(args.model, rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase694_{args.model}_source_attention_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 694,
        "title": "Boundary Head Source-Token Attention Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "top_heads": heads,
        "n_paired_cases": len(paired_ids),
        "summary": summary,
    }
    (OUT_ROOT / f"phase694_{args.model}_source_attention_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase694_*_source_attention_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 694,
        "title": "Boundary Head Source-Token Attention Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase694_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 694 Boundary Head Source-Token Attention Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | rows | short_value | terse_value | short_record | terse_record | short_instruction | terse_instruction | short_answer | terse_answer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        s = item["summary"]["by_variant"].get("short_only", {})
        t = item["summary"]["by_variant"].get("terse_no_explain", {})
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | {item['summary']['n_rows']} | "
            f"{s.get('mean_target_value_in_record_mass', 0.0):.3f} | {t.get('mean_target_value_in_record_mass', 0.0):.3f} | "
            f"{s.get('mean_mass_record_line', 0.0):.3f} | {t.get('mean_mass_record_line', 0.0):.3f} | "
            f"{s.get('mean_mass_instruction_line', 0.0):.3f} | {t.get('mean_mass_instruction_line', 0.0):.3f} | "
            f"{s.get('mean_mass_answer_line', 0.0):.3f} | {t.get('mean_mass_answer_line', 0.0):.3f} |"
        )
    for section, key in [("High Value-Mass Heads", "heads_high_value_mass"), ("High Instruction-Mass Heads", "heads_high_instruction_mass")]:
        lines.extend(["", f"## {section}", ""])
        for item in models:
            lines.append(f"### {item['model']}")
            lines.append("")
            lines.append("| condition | value | record | question | instruction | answer | self | object | relation | top_tokens |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for row in item["summary"][key][:16]:
                lines.append(
                    f"| {row['condition']} | {row.get('mean_target_value_in_record_mass', 0.0):.3f} | "
                    f"{row.get('mean_mass_record_line', 0.0):.3f} | {row.get('mean_mass_question_line', 0.0):.3f} | "
                    f"{row.get('mean_mass_instruction_line', 0.0):.3f} | {row.get('mean_mass_answer_line', 0.0):.3f} | "
                    f"{row.get('mean_mass_self_last', 0.0):.3f} | {row.get('mean_mass_object_name', 0.0):.3f} | "
                    f"{row.get('mean_mass_relation', 0.0):.3f} | {row.get('top_attn_token_counts', {})} |"
                )
            lines.append("")
    (OUT_ROOT / "phase694_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-heads", type=int, default=16)
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
