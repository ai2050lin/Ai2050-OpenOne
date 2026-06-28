#!/usr/bin/env python3
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

from model_utils import load_model, release_model  # noqa: E402


OUT_ROOT = Path("results/glm5_phase721_global_functional_head_atlas_expansion")
MODELS = ["qwen3", "glm4", "deepseek7b"]
GROUP_NAMES = [
    "record_line",
    "question_line",
    "instruction_line",
    "answer_line",
    "self_last",
    "object_name",
    "relation_name",
    "target_value",
    "source_value",
    "target_language",
    "grammar_marker",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def all_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not needle:
        return out
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        out.append((idx, idx + len(needle)))
        start = idx + max(1, len(needle))
    return out


def positions_for_char_spans(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    pos: list[int] = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        for s, e in spans:
            if b > s and a < e:
                pos.append(i)
                break
    return pos


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


def prompt_for(case: dict[str, Any]) -> str:
    return (
        f"Record:\n{case['record']}\n"
        f"Question: {case['question']}\n"
        f"Instruction: {case['instruction']}\n"
        "Answer:"
    )


def line_texts(prompt: str) -> dict[str, str]:
    spans = line_spans(prompt)
    return {name: prompt[s:e] for name, (s, e) in spans.items()}


def build_cases() -> list[dict[str, Any]]:
    fruits = [
        ("apple", "fruit", "red", "sweet"),
        ("banana", "fruit", "yellow", "sweet"),
        ("orange", "fruit", "orange", "sweet"),
        ("grape", "fruit", "purple", "sweet"),
        ("pear", "fruit", "green", "sweet"),
        ("lemon", "fruit", "yellow", "sour"),
        ("peach", "fruit", "pink", "sweet"),
        ("plum", "fruit", "purple", "sweet"),
        ("mango", "fruit", "orange", "sweet"),
        ("kiwi", "fruit", "green", "tart"),
        ("melon", "fruit", "green", "sweet"),
        ("cherry", "fruit", "red", "sweet"),
    ]
    tools = [
        ("hammer", "tool", "gray", "hard"),
        ("saw", "tool", "silver", "sharp"),
        ("knife", "tool", "silver", "sharp"),
        ("wrench", "tool", "gray", "metal"),
        ("drill", "tool", "black", "electric"),
        ("pliers", "tool", "gray", "metal"),
    ]
    animals = [
        ("tiger", "animal", "orange", "wild"),
        ("cat", "animal", "black", "pet"),
        ("frog", "animal", "green", "small"),
        ("bird", "animal", "blue", "flying"),
        ("horse", "animal", "brown", "large"),
        ("fish", "animal", "silver", "swimming"),
    ]
    color_objects = [
        ("strawberry", "red"),
        ("tomato", "red"),
        ("rose", "red"),
        ("firetruck", "red"),
        ("sky", "blue"),
        ("ocean", "blue"),
        ("sapphire", "blue"),
        ("blueberry", "blue"),
        ("grass", "green"),
        ("leaf", "green"),
        ("emerald", "green"),
        ("cucumber", "green"),
        ("sunflower", "yellow"),
        ("corn", "yellow"),
        ("lemon", "yellow"),
        ("banana", "yellow"),
        ("snow", "white"),
        ("milk", "white"),
        ("coal", "black"),
        ("ink", "black"),
        ("carrot", "orange"),
        ("pumpkin", "orange"),
        ("violet", "purple"),
        ("grape", "purple"),
    ]
    translations = [
        ("apple", "pomme", "manzana", "苹果"),
        ("banana", "banane", "plátano", "香蕉"),
        ("red", "rouge", "rojo", "红色"),
        ("blue", "bleu", "azul", "蓝色"),
        ("water", "eau", "agua", "水"),
        ("book", "livre", "libro", "书"),
        ("cat", "chat", "gato", "猫"),
        ("sun", "soleil", "sol", "太阳"),
    ]
    grammar = [
        ("walk", "past tense", "walked", "tense=past"),
        ("play", "past tense", "played", "tense=past"),
        ("jump", "past tense", "jumped", "tense=past"),
        ("look", "past tense", "looked", "tense=past"),
        ("call", "past tense", "called", "tense=past"),
        ("open", "past tense", "opened", "tense=past"),
        ("apple", "plural", "apples", "number=plural"),
        ("car", "plural", "cars", "number=plural"),
        ("box", "plural", "boxes", "number=plural"),
        ("book", "plural", "books", "number=plural"),
        ("dish", "plural", "dishes", "number=plural"),
        ("city", "plural", "cities", "number=plural"),
        ("fast", "comparative", "faster", "degree=comparative"),
        ("small", "comparative", "smaller", "degree=comparative"),
        ("bright", "comparative", "brighter", "degree=comparative"),
        ("cold", "comparative", "colder", "degree=comparative"),
        ("happy", "comparative", "happier", "degree=comparative"),
        ("easy", "comparative", "easier", "degree=comparative"),
        ("fast", "superlative", "fastest", "degree=superlative"),
        ("small", "superlative", "smallest", "degree=superlative"),
        ("bright", "superlative", "brightest", "degree=superlative"),
        ("cold", "superlative", "coldest", "degree=superlative"),
        ("happy", "superlative", "happiest", "degree=superlative"),
        ("easy", "superlative", "easiest", "degree=superlative"),
    ]

    cases: list[dict[str, Any]] = []

    for obj, cat, color, taste in fruits + tools + animals:
        cases.append(
            {
                "case_id": f"fruit_category_{obj}",
                "function_family": "fruit_identity_reuse_difference",
                "record": f"object = {obj}; category = {cat}; color = {color}; property = {taste}.",
                "question": f"What category is {obj}?",
                "instruction": "Answer with only the value.",
                "answer": cat,
                "object_name": obj,
                "relation_name": "category",
                "target_value": cat,
                "source_value": cat,
                "target_language": "",
                "grammar_marker": "",
            }
        )

    for obj, color in color_objects:
        cases.append(
            {
                "case_id": f"color_value_{obj}",
                "function_family": "color_value_reuse_difference",
                "record": f"object = {obj}; color = {color}; relation = visual_color.",
                "question": f"What color is {obj}?",
                "instruction": "Answer with only the value.",
                "answer": color,
                "object_name": obj,
                "relation_name": "color",
                "target_value": color,
                "source_value": color,
                "target_language": "",
                "grammar_marker": "",
            }
        )

    for english, french, spanish, chinese in translations:
        for lang, value in [("French", french), ("Spanish", spanish), ("Chinese", chinese)]:
            cases.append(
                {
                    "case_id": f"translation_{lang.lower()}_{english}",
                    "function_family": "translation_language_route",
                    "record": f"English = {english}; French = {french}; Spanish = {spanish}; Chinese = {chinese}.",
                    "question": f"Translate {english} into {lang}.",
                    "instruction": "Answer with only the translated word.",
                    "answer": value,
                    "object_name": english,
                    "relation_name": "translation",
                    "target_value": value,
                    "source_value": english,
                    "target_language": lang,
                    "grammar_marker": "",
                }
            )

    for base, relation, answer, marker in grammar:
        cases.append(
            {
                "case_id": f"grammar_{relation.replace(' ', '_')}_{base}",
                "function_family": "simple_grammar_protocol_route",
                "record": f"base = {base}; rule = {relation}; marker = {marker}; result = {answer}.",
                "question": f"Apply {relation} to {base}.",
                "instruction": "Answer with only the result.",
                "answer": answer,
                "object_name": base,
                "relation_name": relation,
                "target_value": answer,
                "source_value": base,
                "target_language": "",
                "grammar_marker": marker,
            }
        )
    return cases


def select_cases(max_per_family: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in build_cases():
        grouped[case["function_family"]].append(case)
    selected: list[dict[str, Any]] = []
    for family in sorted(grouped):
        selected.extend(grouped[family][:max_per_family])
    return selected


def find_subseq_positions(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    out: list[int] = []
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            out.extend(range(i, i + n))
    return sorted(set(out))


def token_positions_for_text(tokenizer, ids: list[int], text: str) -> list[int]:
    if not text:
        return []
    variants = [text, " " + text, "\n" + text, text + "\n"]
    out: set[int] = set()
    for variant in variants:
        toks = tokenizer.encode(variant, add_special_tokens=False)
        out.update(find_subseq_positions(ids, toks))
    return sorted(out)


def token_groups(tokenizer, prompt: str, case: dict[str, Any], ids: list[int]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    try:
        enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        for name, span in line_spans(prompt).items():
            groups[name] = positions_for_char_spans(offsets, [span])
        groups["object_name"] = positions_for_char_spans(offsets, all_occurrences(prompt, case["object_name"]))
        groups["relation_name"] = positions_for_char_spans(offsets, all_occurrences(prompt, case["relation_name"]))
        groups["target_value"] = positions_for_char_spans(offsets, all_occurrences(prompt, case["target_value"]))
        groups["source_value"] = positions_for_char_spans(offsets, all_occurrences(prompt, case["source_value"]))
        groups["target_language"] = positions_for_char_spans(offsets, all_occurrences(prompt, case["target_language"]))
        groups["grammar_marker"] = positions_for_char_spans(offsets, all_occurrences(prompt, case["grammar_marker"]))
    except Exception:
        for name, text in line_texts(prompt).items():
            groups[name] = token_positions_for_text(tokenizer, ids, text)
        groups["object_name"] = token_positions_for_text(tokenizer, ids, case["object_name"])
        groups["relation_name"] = token_positions_for_text(tokenizer, ids, case["relation_name"])
        groups["target_value"] = token_positions_for_text(tokenizer, ids, case["target_value"])
        groups["source_value"] = token_positions_for_text(tokenizer, ids, case["source_value"])
        groups["target_language"] = token_positions_for_text(tokenizer, ids, case["target_language"])
        groups["grammar_marker"] = token_positions_for_text(tokenizer, ids, case["grammar_marker"])
    groups["self_last"] = [len(ids) - 1]
    return groups


def group_mass(head_row: torch.Tensor, groups: dict[str, list[int]]) -> dict[str, float]:
    n = head_row.numel()
    out: dict[str, float] = {}
    for name in GROUP_NAMES:
        idxs = [i for i in groups.get(name, []) if 0 <= i < n]
        out[f"mass_{name}"] = float(head_row[idxs].sum().detach().float().cpu().item()) if idxs else 0.0
    line_total = sum(out[f"mass_{name}"] for name in ["record_line", "question_line", "instruction_line", "answer_line"])
    out["mass_line_partition"] = line_total
    out["mass_outside_lines"] = max(0.0, 1.0 - line_total)
    return out


def source_focus_score(row: dict[str, Any]) -> float:
    return (
        row.get("mean_mass_target_value", 0.0)
        + 0.5 * row.get("mean_mass_object_name", 0.0)
        + 0.5 * row.get("mean_mass_relation_name", 0.0)
        + 0.5 * row.get("mean_mass_target_language", 0.0)
        + 0.5 * row.get("mean_mass_grammar_marker", 0.0)
        - 0.5 * row.get("mean_mass_instruction_line", 0.0)
        - 0.25 * row.get("mean_mass_answer_line", 0.0)
    )


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


def audit_prompt(model, tokenizer, device, case: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = prompt_for(case)
    ids, attentions = run_attention(model, tokenizer, device, prompt)
    groups = token_groups(tokenizer, prompt, case, ids)
    answer_pos = len(ids) - 1
    rows: list[dict[str, Any]] = []
    for layer, att in enumerate(attentions):
        # [batch, heads, query, key]
        head_rows = att[0, :, answer_pos, :].detach()
        for head in range(head_rows.shape[0]):
            row = head_rows[head]
            top_pos = int(torch.argmax(row).detach().cpu().item())
            rows.append(
                {
                    "case_id": case["case_id"],
                    "function_family": case["function_family"],
                    "answer": case["answer"],
                    "layer": layer,
                    "head": head,
                    "head_key": f"L{layer}H{head}",
                    "seq_len": len(ids),
                    "answer_pos": answer_pos,
                    "top_attn_pos": top_pos,
                    "top_attn_token": tokenizer.decode([ids[top_pos]]),
                    "top_attn_mass": float(row[top_pos].detach().float().cpu().item()),
                    **group_mass(row, groups),
                }
            )
    del attentions
    return rows


def summarize_rows(model_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["function_family"], r["head_key"])].append(r)

    head_scores: list[dict[str, Any]] = []
    for (family, head_key), vals in grouped.items():
        rec: dict[str, Any] = {
            "model": model_name,
            "function_family": family,
            "head_key": head_key,
            "layer": vals[0]["layer"],
            "head": vals[0]["head"],
            "n": len(vals),
        }
        for key in [k for k in vals[0] if k.startswith("mass_") or k == "top_attn_mass"]:
            rec[f"mean_{key}"] = sum(float(v[key]) for v in vals) / len(vals)
        rec["top_attn_token_counts"] = dict(Counter(str(v["top_attn_token"]) for v in vals).most_common(10))
        rec["source_focus_score"] = source_focus_score(rec)
        head_scores.append(rec)

    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({r["function_family"] for r in rows}):
        family_scores = [r for r in head_scores if r["function_family"] == family]
        by_family[family] = {
            "n_prompt_head_rows": sum(r["n"] for r in family_scores),
            "n_heads": len(family_scores),
            "top_source_focus_heads": sorted(family_scores, key=lambda r: r["source_focus_score"], reverse=True)[:24],
            "top_target_value_heads": sorted(family_scores, key=lambda r: r.get("mean_mass_target_value", 0.0), reverse=True)[:24],
            "top_instruction_heads": sorted(family_scores, key=lambda r: r.get("mean_mass_instruction_line", 0.0), reverse=True)[:12],
        }

    return {
        "phase": 721,
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_prompt_head_rows": len(rows),
        "n_function_families": len(by_family),
        "by_family": by_family,
        "top_global_source_focus_heads": sorted(head_scores, key=lambda r: r["source_focus_score"], reverse=True)[:48],
        "top_global_target_value_heads": sorted(head_scores, key=lambda r: r.get("mean_mass_target_value", 0.0), reverse=True)[:48],
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n", encoding="utf-8")


def run_model(args) -> dict[str, Any]:
    cases = select_cases(args.max_cases_per_family)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUT_ROOT / "phase721_case_set.json", {"cases": cases, "n": len(cases)})
    log(f"{args.model}: selected {len(cases)} cases, max_per_family={args.max_cases_per_family}")

    model, tokenizer, device = load_model(args.model)
    rows: list[dict[str, Any]] = []
    try:
        for idx, case in enumerate(cases, 1):
            rows.extend(audit_prompt(model, tokenizer, device, case))
            if idx % args.log_every == 0 or idx == len(cases):
                log(f"{args.model}: audited {idx}/{len(cases)} prompts; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(args.model, rows)
    write_jsonl(OUT_ROOT / f"phase721_{args.model}_prompt_head_rows.jsonl", rows)
    write_json(OUT_ROOT / f"phase721_{args.model}_head_scores.json", summary)
    log(f"{args.model}: wrote {len(rows)} prompt-head rows")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return summary


def write_cross_summary() -> dict[str, Any]:
    models = []
    for model in MODELS:
        path = OUT_ROOT / f"phase721_{model}_head_scores.json"
        if path.exists():
            models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 721,
        "title": "Global Functional Head Atlas Data Expansion",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_models": len(models),
        "models": [m["model"] for m in models],
        "same_stage_as_phase720": True,
        "status": "complete" if len(models) == len(MODELS) else "partial",
        "small_model_caution": "attention mass is observational and may be model-scale specific; causal patch is required before claiming mechanism",
        "by_model": {
            m["model"]: {
                "n_prompt_head_rows": m["n_prompt_head_rows"],
                "n_function_families": m["n_function_families"],
                "top_global_source_focus_heads": m["top_global_source_focus_heads"][:12],
            }
            for m in models
        },
    }
    write_json(OUT_ROOT / "phase721_cross_model_summary.json", payload)
    lines = [
        "# Phase 721 Global Functional Head Atlas Data Expansion",
        "",
        "## Status",
        "",
        f"- Models complete: `{payload['models']}`",
        f"- Status: `{payload['status']}`",
        "- Evidence type: observational answer-last attention mass, not causal patch.",
        "",
        "## Top Source-Focus Heads By Model",
        "",
    ]
    for model, item in payload["by_model"].items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| family | head | score | target_value | object | relation | instruction | top_tokens |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for r in item["top_global_source_focus_heads"]:
            lines.append(
                f"| {r['function_family']} | {r['head_key']} | {r.get('source_focus_score', 0.0):.4f} | "
                f"{r.get('mean_mass_target_value', 0.0):.4f} | {r.get('mean_mass_object_name', 0.0):.4f} | "
                f"{r.get('mean_mass_relation_name', 0.0):.4f} | {r.get('mean_mass_instruction_line', 0.0):.4f} | "
                f"{r.get('top_attn_token_counts', {})} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Strict Interpretation",
            "",
            "- This phase identifies candidate head routes for functional atlas expansion.",
            "- It does not prove semantic identity, necessity, or sufficiency.",
            "- The next causal phase should patch only the repeated top heads per function family.",
            "",
        ]
    )
    (OUT_ROOT / "phase721_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def dry_run(max_cases_per_family: int) -> None:
    cases = select_cases(max_cases_per_family)
    counts = Counter(c["function_family"] for c in cases)
    print(json.dumps({"n": len(cases), "by_family": dict(counts), "sample_cases": cases[:8]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases-per-family", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.max_cases_per_family)
        return
    if args.summarize_only:
        write_cross_summary()
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only or --dry-run is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
