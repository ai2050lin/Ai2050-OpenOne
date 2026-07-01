#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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

import phase796_global_competitor_token_identity_audit as p796  # noqa: E402
import phase816_multi_token_answer_span_rollout_closure as p816  # noqa: E402
import phase818_alias_span_candidate_scoring_benchmark as p818  # noqa: E402
import phase820_answer_boundary_standard_v1 as p820  # noqa: E402
from model_utils import get_layers, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402


PHASE = 821
SOURCE_820 = Path("tests/result/phase820_answer_boundary_standard_v1")
RESULT_ROOT = Path("tests/result/phase821_boundary_standard_guided_causal_localization")

BOUNDARY_RANK = {
    "target_equivalent": 5,
    "close_near_miss": 4,
    "broad_near_miss": 3,
    "format_with_target": 2,
    "generic_blocker": 1,
    "format_echo": 0,
    "object_echo": 0,
    "wrong": 0,
    "unknown_other": 0,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) else default


def clean_generated(text: str) -> str:
    return p816.clean_generated(text)


def boundary_for(lookup: dict[tuple[str, str], dict[str, Any]], case_id: str, phrase: Any) -> dict[str, Any]:
    std = p820.class_for_phrase(lookup, case_id, clean_generated(str(phrase or "")))
    cls = str(std.get("final_boundary_class") or "unknown_other")
    out = dict(std)
    out["boundary_rank"] = int(BOUNDARY_RANK.get(cls, 0))
    return out


def case_map() -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in p816.CASES}


def select_failure_rows(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(SOURCE_820 / "phase820_reanalysis_rows.jsonl")
        if row.get("model") == model_name
        and row.get("round") == args.source_round
        and row.get("prompt_variant") == args.recipient_prompt
    ]
    allowed = set(parse_csv(args.source_boundary_classes))
    if allowed:
        rows = [row for row in rows if row.get("final_boundary_class") in allowed]
    if args.only_unclosed:
        rows = [row for row in rows if not row.get("strict_full_v1")]

    # Prefer diverse boundary classes, then fill by worst rank.
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row.get("final_boundary_class"))].append(row)
    selected: list[dict[str, Any]] = []
    for cls in sorted(by_class, key=lambda c: (BOUNDARY_RANK.get(c, 0), c)):
        if by_class[cls]:
            selected.append(by_class[cls][0])
        if len(selected) >= int(args.max_cases):
            break
    if len(selected) < int(args.max_cases):
        seen = {row.get("case_id") for row in selected}
        for row in sorted(rows, key=lambda r: (BOUNDARY_RANK.get(str(r.get("final_boundary_class")), 0), str(r.get("case_id")))):
            if row.get("case_id") in seen:
                continue
            selected.append(row)
            seen.add(row.get("case_id"))
            if len(selected) >= int(args.max_cases):
                break
    return selected[: int(args.max_cases)]


def layer_indices(model, args: argparse.Namespace) -> list[int]:
    n_layers = len(get_layers(model))
    spec = str(args.layers).strip()
    if spec.startswith("last"):
        n = int(spec.replace("last", "") or args.max_layers)
        return list(range(max(0, n_layers - n), n_layers))
    if spec == "spread":
        k = max(1, int(args.max_layers))
        if k >= n_layers:
            return list(range(n_layers))
        if k == 1:
            return [n_layers - 1]
        return sorted({round(i * (n_layers - 1) / (k - 1)) for i in range(k)})
    vals = []
    for part in parse_csv(spec):
        idx = int(part)
        if idx < 0:
            idx = n_layers + idx
        if 0 <= idx < n_layers:
            vals.append(idx)
    return sorted(dict.fromkeys(vals))[: int(args.max_layers)]


def tensor_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]


def donor_hidden_vectors(model, tokenizer, device: torch.device, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
    ids = encode_prompt(tokenizer, prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = out.hidden_states
    vectors: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        hs_idx = int(layer_idx) + 1
        if 0 <= hs_idx < len(hidden_states):
            vectors[int(layer_idx)] = hidden_states[hs_idx][0, -1].detach().float().cpu()
    del out
    return vectors


def make_patch_hook(donor_vec: torch.Tensor, alpha: float):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        patched = hidden.clone()
        vec = donor_vec.to(device=patched.device, dtype=patched.dtype)
        patched[:, -1, :] = patched[:, -1, :] + float(alpha) * (vec - patched[:, -1, :])
        if rest is not None:
            return (patched, *rest)
        return patched

    return hook


def greedy_generate_first_step_patch(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    max_new_tokens: int,
    layer_idx: int | None,
    donor_vec: torch.Tensor | None,
    alpha: float,
) -> tuple[str, list[int]]:
    current = [int(x) for x in prompt_ids]
    new_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    layers = get_layers(model)
    for step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handle = None
        if step == 0 and layer_idx is not None and donor_vec is not None:
            handle = layers[int(layer_idx)].register_forward_hook(make_patch_hook(donor_vec, alpha))
        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1].detach().float()
        finally:
            if handle is not None:
                handle.remove()
        next_id = int(torch.argmax(logits).item())
        new_ids.append(next_id)
        current.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    return tokenizer.decode(new_ids, skip_special_tokens=True), new_ids


def audit_case(
    model,
    tokenizer,
    device: torch.device,
    case: dict[str, Any],
    source_row: dict[str, Any],
    standards: list[dict[str, Any]],
    layers: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    lookup = p820.standard_lookup(standards)
    recipient_prompt = p816.build_prompt(case, args.recipient_prompt)
    donor_prompt = p816.build_prompt(case, args.donor_prompt)
    recipient_ids = encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = greedy_generate_first_step_patch(
        model, tokenizer, device, recipient_ids, args.max_new_tokens, None, None, args.patch_alpha
    )
    baseline_boundary = boundary_for(lookup, case["case_id"], baseline_text)
    donor_vectors = donor_hidden_vectors(model, tokenizer, device, donor_prompt, layers)
    out: list[dict[str, Any]] = []
    for layer_idx in layers:
        donor_vec = donor_vectors.get(int(layer_idx))
        if donor_vec is None:
            continue
        patched_text, patched_ids = greedy_generate_first_step_patch(
            model,
            tokenizer,
            device,
            recipient_ids,
            args.max_new_tokens,
            int(layer_idx),
            donor_vec,
            args.patch_alpha,
        )
        patched_boundary = boundary_for(lookup, case["case_id"], patched_text)
        delta_rank = int(patched_boundary["boundary_rank"]) - int(baseline_boundary["boundary_rank"])
        out.append(
            {
                "row_kind": "phase821_boundary_standard_guided_causal_localization",
                "phase": PHASE,
                "model": args.model,
                "round": args.round_name,
                "case_id": case["case_id"],
                "object": case["object"],
                "target_answer": case["answer"],
                "recipient_prompt_variant": args.recipient_prompt,
                "donor_prompt_variant": args.donor_prompt,
                "source_phase820_boundary_class": source_row.get("final_boundary_class"),
                "source_phase820_strict_full_v1": bool(source_row.get("strict_full_v1")),
                "layer_idx": int(layer_idx),
                "patch_kind": "donor_exact_choices_residual_to_no_choices_first_step",
                "patch_alpha": float(args.patch_alpha),
                "baseline_generated": clean_generated(baseline_text),
                "baseline_token_ids": baseline_ids,
                "baseline_boundary_class": baseline_boundary.get("final_boundary_class"),
                "baseline_boundary_rank": int(baseline_boundary["boundary_rank"]),
                "baseline_strict_accept": bool(baseline_boundary.get("strict_accept")),
                "baseline_protocol_valid": bool(baseline_boundary.get("protocol_valid")),
                "patched_generated": clean_generated(patched_text),
                "patched_token_ids": patched_ids,
                "patched_boundary_class": patched_boundary.get("final_boundary_class"),
                "patched_boundary_rank": int(patched_boundary["boundary_rank"]),
                "patched_strict_accept": bool(patched_boundary.get("strict_accept")),
                "patched_protocol_valid": bool(patched_boundary.get("protocol_valid")),
                "delta_boundary_rank": delta_rank,
                "improved_boundary": delta_rank > 0,
                "degraded_boundary": delta_rank < 0,
                "target_transition": patched_boundary.get("final_boundary_class") == "target_equivalent",
                "protocol_repaired": (
                    not bool(baseline_boundary.get("protocol_valid")) and bool(patched_boundary.get("protocol_valid"))
                ),
            }
        )
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str | None = None) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case_id"))].append(row)
    case_summaries = {}
    for case_id, vals in by_case.items():
        best = max(vals, key=lambda r: (int(r.get("delta_boundary_rank", 0)), int(r.get("patched_boundary_rank", 0))))
        case_summaries[case_id] = {
            "baseline_class": vals[0].get("baseline_boundary_class"),
            "baseline_rank": vals[0].get("baseline_boundary_rank"),
            "best_layer": best.get("layer_idx"),
            "best_delta_boundary_rank": best.get("delta_boundary_rank"),
            "best_patched_class": best.get("patched_boundary_class"),
            "best_patched_generated": best.get("patched_generated"),
            "any_improved": any(row.get("improved_boundary") for row in vals),
            "any_target_transition": any(row.get("target_transition") for row in vals),
            "any_protocol_repaired": any(row.get("protocol_repaired") for row in vals),
        }
    return {
        "phase": PHASE,
        "title": "Boundary-Standard-Guided Causal Localization",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "patch_kind": "exact_choices donor residual -> no_choices recipient, first generated token",
        "n_rows": len(rows),
        "n_cases": len(case_summaries),
        "layers": sorted({int(row["layer_idx"]) for row in rows}) if rows else [],
        "baseline_classes": dict(Counter(row.get("baseline_boundary_class") for row in rows)),
        "patched_classes": dict(Counter(row.get("patched_boundary_class") for row in rows)),
        "improved_rows": sum(1 for row in rows if row.get("improved_boundary")),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "target_transition_rows": sum(1 for row in rows if row.get("target_transition")),
        "protocol_repaired_rows": sum(1 for row in rows if row.get("protocol_repaired")),
        "improved_cases": sum(1 for row in case_summaries.values() if row.get("any_improved")),
        "target_transition_cases": sum(1 for row in case_summaries.values() if row.get("any_target_transition")),
        "protocol_repaired_cases": sum(1 for row in case_summaries.values() if row.get("any_protocol_repaired")),
        "mean_delta_boundary_rank": (
            sum(finite(row.get("delta_boundary_rank")) for row in rows) / len(rows) if rows else None
        ),
        "by_case": case_summaries,
        "boundary": (
            "This phase tests whether an exact-choices donor residual state can causally move a no-choices generation across Phase 820 boundary classes."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    standards = p820.standard_rows()
    selected_rows = select_failure_rows(args.model, args)
    cmap = case_map()
    log(f"{args.model}/{args.round_name}: selected failure cases={len(selected_rows)}")
    if args.dry_run:
        payload = {"model": args.model, "selected_cases": [row.get("case_id") for row in selected_rows]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    model, tokenizer, device, attn_impl = p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    layers = layer_indices(model, args)
    log(f"{args.model}: layers={layers} max_new_tokens={args.max_new_tokens} alpha={args.patch_alpha}")
    rows: list[dict[str, Any]] = []
    try:
        for idx, source_row in enumerate(selected_rows, 1):
            case = cmap.get(str(source_row.get("case_id")))
            if not case:
                continue
            rows.extend(audit_case(model, tokenizer, device, case, source_row, standards, layers, args))
            if idx % int(args.log_every) == 0 or idx == len(selected_rows):
                log(f"{args.model}: boundary localization {idx}/{len(selected_rows)} cases rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, attn_impl)
    write_jsonl(out_dir / f"phase821_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase821_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "rows": summary["n_rows"],
                "cases": summary["n_cases"],
                "improved_cases": summary["improved_cases"],
                "target_transition_cases": summary["target_transition_cases"],
                "protocol_repaired_cases": summary["protocol_repaired_cases"],
                "mean_delta_boundary_rank": summary["mean_delta_boundary_rank"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 821 Boundary-Standard-Guided Causal Localization ({payload['round']})",
        "",
        "- Boundary: Phase 820 answer-boundary standard v1.",
        "- Intervention: exact_choices donor residual state patched into no_choices recipient at the first generated token.",
        "",
        "## Model Summary",
        "",
        "| model | cases | rows | improved cases | target transitions | protocol repairs | improved rows | degraded rows | mean delta rank | baseline classes | patched classes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | {data.get('n_cases')} | {data.get('n_rows')} | {data.get('improved_cases')} | "
            f"{data.get('target_transition_cases')} | {data.get('protocol_repaired_cases')} | "
            f"{data.get('improved_rows')} | {data.get('degraded_rows')} | "
            f"{finite(data.get('mean_delta_boundary_rank')):.3f} | "
            f"`{json.dumps(data.get('baseline_classes') or {}, ensure_ascii=False)}` | "
            f"`{json.dumps(data.get('patched_classes') or {}, ensure_ascii=False)}` |"
        )
    lines += ["", "## Best Case Transitions", ""]
    lines += [
        "| model | case | baseline | best patched | best layer | delta | generated |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for case_id, row in sorted((data.get("by_case") or {}).items()):
            lines.append(
                f"| {model_name} | {case_id} | `{row.get('baseline_class')}` | `{row.get('best_patched_class')}` | "
                f"{row.get('best_layer')} | {row.get('best_delta_boundary_rank')} | `{row.get('best_patched_generated')}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "status": "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_summaries": {},
        "models": [],
    }
    for model_name in MODELS:
        path = out_dir / f"phase821_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase821_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase821_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--recipient-prompt", default="no_choices")
    parser.add_argument("--donor-prompt", default="exact_choices")
    parser.add_argument("--source-boundary-classes", default="close_near_miss,broad_near_miss,format_echo,format_with_target,object_echo,unknown_other,wrong")
    parser.add_argument("--only-unclosed", action="store_true", default=True)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--layers", default="last2")
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--patch-alpha", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        payload = summarize_round(args.round_name)
        print(json.dumps({"round": args.round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only")
    run_model(args)


if __name__ == "__main__":
    main()
