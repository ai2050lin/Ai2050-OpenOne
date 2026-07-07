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

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase944_activation_weighted_mlp_channel_causal_audit as p944  # noqa: E402
import phase201_stop_prose_component_atlas as p201  # noqa: E402
import phase204_global_trajectory_stop_execution_atlas as p204  # noqa: E402
import phase207_eos_positive_head_atlas as p207  # noqa: E402


PHASE = 208
SOURCE_PHASE = 207
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase208_decode_config_eos_boundary_audit")


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


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            vals.append(fval)
    return None if not vals else float(sum(vals) / len(vals))


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def token_text(tokenizer, token_id: int | None) -> str:
    if token_id is None:
        return ""
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return str(token_id)


def config_audit(model, tokenizer, max_steps: int) -> dict[str, Any]:
    generation_config = getattr(model, "generation_config", None)
    model_config = getattr(model, "config", None)
    return {
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_eos_token": tokenizer.eos_token,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "tokenizer_pad_token": tokenizer.pad_token,
        "tokenizer_bos_token_id": tokenizer.bos_token_id,
        "tokenizer_bos_token": tokenizer.bos_token,
        "generation_config_eos_token_id": getattr(generation_config, "eos_token_id", None),
        "generation_config_pad_token_id": getattr(generation_config, "pad_token_id", None),
        "generation_config_forced_eos_token_id": getattr(generation_config, "forced_eos_token_id", None),
        "generation_config_max_new_tokens": getattr(generation_config, "max_new_tokens", None),
        "model_config_eos_token_id": getattr(model_config, "eos_token_id", None),
        "model_config_pad_token_id": getattr(model_config, "pad_token_id", None),
        "manual_max_steps": int(max_steps),
        "stopping_criteria": "default generate EOS stopping for generate modes; manual greedy stops when emitted token id is in eos group",
    }


def special_prompts(tokenizer) -> list[dict[str, Any]]:
    empty_proxy = tokenizer.bos_token if tokenizer.bos_token else " "
    items = [
        ("empty_completion", empty_proxy),
        ("space_completion", " "),
        ("answer_boundary", "Answer:"),
        ("final_marker", "Final answer:"),
        ("end_now", "End the response now."),
        ("single_word_done", "Reply with exactly one word, then stop. Answer: red"),
    ]
    out = []
    for idx, (protocol, text) in enumerate(items):
        out.append(
            {
                "phase": PHASE,
                "model": None,
                "sample_id": f"special_{idx}",
                "source_sample_id": f"special_{idx}",
                "trajectory_id": f"special_{idx}|{protocol}",
                "relation": "special_decode",
                "language_pair": "special",
                "prompt_protocol": protocol,
                "object": "",
                "target_label": "",
                "prompt": text,
            }
        )
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=False)
            out.append(
                {
                    "phase": PHASE,
                    "model": None,
                    "sample_id": "special_chat_empty",
                    "source_sample_id": "special_chat_empty",
                    "trajectory_id": "special_chat_empty|chat_empty",
                    "relation": "special_decode",
                    "language_pair": "special",
                    "prompt_protocol": "chat_empty",
                    "object": "",
                    "target_label": "",
                    "prompt": text,
                }
            )
        except Exception:
            pass
    return out


def build_prompts(args: argparse.Namespace, tokenizer, holdout_by_pair: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    prompts = p207.build_prompt_records(args, tokenizer, holdout_by_pair)
    prompts.extend(special_prompts(tokenizer))
    for row in prompts:
        row["model"] = args.model
    return prompts


def metric_for_next(model, tokenizer, device: torch.device, text: str, sample: dict[str, Any], groups: dict[str, list[int]]) -> dict[str, Any]:
    encoded = tokenizer([text], return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = attention_mask.sum(dim=1).long() - 1
    with torch.inference_mode():
        result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = result.logits[0, int(last_pos[0].item())].detach().float().cpu()
    metrics = p204.metric_for_logits(tokenizer, logits, sample, groups)
    del result, input_ids, attention_mask
    return metrics


def manual_greedy(
    model,
    tokenizer,
    device: torch.device,
    prompts: list[dict[str, Any]],
    groups: dict[str, list[int]],
    max_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    token_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    eos_ids = set(int(x) for x in groups.get("eos") or [])
    for item in prompts:
        generated = ""
        emitted_ids: list[int] = []
        emitted_tokens: list[str] = []
        first_eos_step = None
        for step in range(1, int(max_steps) + 1):
            metrics = metric_for_next(model, tokenizer, device, str(item.get("prompt") or "") + generated, item, groups)
            next_id = int(metrics["top_token_id"])
            next_text = str(metrics.get("top_token") or "")
            emitted_ids.append(next_id)
            emitted_tokens.append(next_text)
            token_rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase208_manual_token_row",
                    "model": item.get("model"),
                    "trajectory_id": item.get("trajectory_id"),
                    "decode_mode": "manual_greedy",
                    "prompt_protocol": item.get("prompt_protocol"),
                    "step": step,
                    "emitted_token_id": next_id,
                    "emitted_token": next_text,
                    "emitted_is_eos": next_id in eos_ids,
                    **metrics,
                }
            )
            generated += next_text
            if next_id in eos_ids:
                first_eos_step = step
                break
        traj_rows.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase208_decode_trajectory_row",
                "model": item.get("model"),
                "trajectory_id": item.get("trajectory_id"),
                "source_sample_id": item.get("source_sample_id"),
                "relation": item.get("relation"),
                "language_pair": item.get("language_pair"),
                "prompt_protocol": item.get("prompt_protocol"),
                "decode_mode": "manual_greedy",
                "seed": None,
                "temperature": None,
                "prompt": item.get("prompt"),
                "generated": generated,
                "steps_generated": len(emitted_ids),
                "emitted_ids": emitted_ids,
                "emitted_tokens": emitted_tokens,
                "first_token_id": emitted_ids[0] if emitted_ids else None,
                "first_token": emitted_tokens[0] if emitted_tokens else None,
                "ended_with_eos": bool(first_eos_step is not None),
                "first_eos_step": first_eos_step,
            }
        )
    return token_rows, traj_rows


def generate_decode(
    model,
    tokenizer,
    device: torch.device,
    prompts: list[dict[str, Any]],
    groups: dict[str, list[int]],
    mode: str,
    max_steps: int,
    batch_size: int,
    seed: int | None = None,
    temperature: float | None = None,
) -> list[dict[str, Any]]:
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    rows: list[dict[str, Any]] = []
    eos_ids = set(int(x) for x in groups.get("eos") or [])
    effective_batch_size = 1 if tokenizer.pad_token_id is not None and tokenizer.pad_token_id in eos_ids else max(1, int(batch_size))
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    try:
        for start in range(0, len(prompts), effective_batch_size):
            batch = prompts[start : start + effective_batch_size]
            encoded = tokenizer([str(row.get("prompt") or "") for row in batch], return_tensors="pt", padding=True, add_special_tokens=False)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_steps),
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "return_dict_in_generate": False,
            }
            if mode == "generate_greedy":
                kwargs.update({"do_sample": False, "num_beams": 1})
            elif mode == "generate_beam":
                kwargs.update({"do_sample": False, "num_beams": 3, "early_stopping": True})
            elif mode == "generate_sample":
                kwargs.update({"do_sample": True, "num_beams": 1, "temperature": float(temperature or 1.0), "top_p": 0.95})
            else:
                raise ValueError(mode)
            with torch.inference_mode():
                output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            new_ids = output_ids[:, input_ids.shape[1] :].detach().cpu()
            for row_idx, item in enumerate(batch):
                raw_ids = [int(x) for x in new_ids[row_idx].tolist()]
                ids = raw_ids
                if tokenizer.pad_token_id is not None and int(tokenizer.pad_token_id) not in eos_ids:
                    ids = [token_id for token_id in raw_ids if token_id != int(tokenizer.pad_token_id)]
                first_eos_step = None
                kept_ids = []
                for idx, token_id in enumerate(ids, start=1):
                    kept_ids.append(token_id)
                    if token_id in eos_ids:
                        first_eos_step = idx
                        break
                texts = [token_text(tokenizer, token_id) for token_id in kept_ids]
                generated = tokenizer.decode(kept_ids, skip_special_tokens=False) if kept_ids else ""
                rows.append(
                    {
                        "phase": PHASE,
                        "source_phase": SOURCE_PHASE,
                        "row_kind": "phase208_decode_trajectory_row",
                        "model": item.get("model"),
                        "trajectory_id": item.get("trajectory_id"),
                        "source_sample_id": item.get("source_sample_id"),
                        "relation": item.get("relation"),
                        "language_pair": item.get("language_pair"),
                        "prompt_protocol": item.get("prompt_protocol"),
                        "decode_mode": mode,
                        "seed": seed,
                        "temperature": temperature,
                        "prompt": item.get("prompt"),
                        "generated": generated,
                        "steps_generated": len(kept_ids),
                        "emitted_ids": kept_ids,
                        "emitted_tokens": texts,
                        "first_token_id": kept_ids[0] if kept_ids else None,
                        "first_token": texts[0] if texts else None,
                        "ended_with_eos": bool(first_eos_step is not None),
                        "first_eos_step": first_eos_step,
                    }
                )
            del input_ids, attention_mask, output_ids, new_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = old_padding_side
    return rows


def summarize_decode(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ["model", "decode_mode", "prompt_protocol", "temperature"]
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "ended_with_eos": sum(1 for item in items if item.get("ended_with_eos")),
                "avg_steps": mean([item.get("steps_generated") for item in items]),
                "top_first_tokens": dict(Counter(str(item.get("first_token")) for item in items).most_common(8)),
                "top_generated_prefixes": dict(Counter(str(item.get("generated"))[:48] for item in items).most_common(6)),
            }
        )
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def compare_manual_generate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("decode_mode") in {"manual_greedy", "generate_greedy"}:
            by_key[(row.get("trajectory_id"), row.get("prompt_protocol"))][str(row.get("decode_mode"))] = row
    out = []
    for (trajectory_id, protocol), modes in by_key.items():
        manual = modes.get("manual_greedy")
        generate = modes.get("generate_greedy")
        if not manual or not generate:
            continue
        out.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase208_manual_generate_compare_row",
                "trajectory_id": trajectory_id,
                "prompt_protocol": protocol,
                "model": manual.get("model"),
                "first_token_match": manual.get("first_token_id") == generate.get("first_token_id"),
                "manual_first_token": manual.get("first_token"),
                "generate_first_token": generate.get("first_token"),
                "manual_ended_with_eos": manual.get("ended_with_eos"),
                "generate_ended_with_eos": generate.get("ended_with_eos"),
                "manual_steps": manual.get("steps_generated"),
                "generate_steps": generate.get("steps_generated"),
            }
        )
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Decode Config Audit and EOS Induction Boundary Search",
        "model": args.model,
        "prompt_protocols": parse_csv(args.prompt_protocols),
        "sampling_temperatures": [float(x) for x in parse_csv(args.sampling_temperatures)],
        "sampling_seeds": [int(x) for x in parse_csv(args.sampling_seeds)],
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase208_{args.model}_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload
    model = None
    tokenizer = None
    token_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        audit = config_audit(model, tokenizer, int(args.max_steps))
        groups = p201.token_groups(tokenizer)
        holdout_by_pair, meta = p944.build_holdout_samples(args, model, tokenizer, device)
        prompts = build_prompts(args, tokenizer, holdout_by_pair)
        if int(args.max_prompts) > 0:
            prompts = prompts[: int(args.max_prompts)]
        manual_tokens, manual_rows = manual_greedy(model, tokenizer, device, prompts, groups, int(args.max_steps))
        token_rows.extend(manual_tokens)
        decode_rows.extend(manual_rows)
        decode_rows.extend(generate_decode(model, tokenizer, device, prompts, groups, "generate_greedy", int(args.max_steps), int(args.batch_size)))
        decode_rows.extend(generate_decode(model, tokenizer, device, prompts, groups, "generate_beam", int(args.max_steps), int(args.batch_size)))
        for temp in [float(x) for x in parse_csv(args.sampling_temperatures)]:
            for seed in [int(x) for x in parse_csv(args.sampling_seeds)]:
                decode_rows.extend(
                    generate_decode(
                        model,
                        tokenizer,
                        device,
                        prompts,
                        groups,
                        "generate_sample",
                        int(args.max_steps),
                        int(args.batch_size),
                        seed=seed,
                        temperature=temp,
                    )
                )
        compare_rows = compare_manual_generate(decode_rows)
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    decode_summary_rows = summarize_decode(decode_rows)
    eos_positive_rows = [row for row in decode_rows if row.get("ended_with_eos")]
    compare_summary = {
        "rows": len(compare_rows),
        "first_token_matches": sum(1 for row in compare_rows if row.get("first_token_match")),
        "manual_eos": sum(1 for row in compare_rows if row.get("manual_ended_with_eos")),
        "generate_eos": sum(1 for row in compare_rows if row.get("generate_ended_with_eos")),
    }
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
        "config_audit": audit,
        "prompt_count": len({row.get("trajectory_id") for row in decode_rows}),
        "decode_rows": len(decode_rows),
        "manual_token_rows": len(token_rows),
        "eos_positive_count": len(eos_positive_rows),
        "decode_summary_rows": decode_summary_rows,
        "manual_generate_compare_summary": compare_summary,
        "boundary": "Decode audit only. Sampling EOS, if any, is stochastic and must not be merged with greedy ModelStopExecuted.",
    }
    write_json(out_dir / f"phase208_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase208_{args.model}_manual_token_rows.jsonl", token_rows)
    write_jsonl(out_dir / f"phase208_{args.model}_decode_rows.jsonl", decode_rows)
    write_jsonl(out_dir / f"phase208_{args.model}_manual_generate_compare_rows.jsonl", compare_rows)
    write_jsonl(out_dir / f"phase208_{args.model}_eos_positive_rows.jsonl", eos_positive_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "prompt_count": payload["prompt_count"],
                "decode_rows": len(decode_rows),
                "eos_positive_count": len(eos_positive_rows),
                "manual_generate_compare_summary": compare_summary,
                "top_decode_summary_rows": decode_summary_rows[:16],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase208_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    decode_summary_rows = []
    for summary in summaries:
        decode_summary_rows.extend(dict(row) for row in summary.get("decode_summary_rows") or [])
    payload = {
        "schema_version": "phase208_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "total_eos_positive_count": sum(int(summary.get("eos_positive_count") or 0) for summary in summaries),
        "model_summaries": summaries,
        "decode_summary_rows": decode_summary_rows,
    }
    write_json(out_dir / "phase208_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase208_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 208 decode config EOS boundary audit", ""]
    lines.append(f"Total EOS positives: {payload.get('total_eos_positive_count')}")
    lines.append("")
    lines.append("| model | mode | protocol | temp | rows | eos | avg steps | first tokens |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | --- |")
    for row in payload.get("decode_summary_rows") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('decode_mode')} | {row.get('prompt_protocol')} | {row.get('temperature')} | "
            f"{row.get('rows')} | {row.get('ended_with_eos')} | {finite(row.get('avg_steps')):.2f} | {row.get('top_first_tokens')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="decode_config_eos_boundary_audit")
    parser.add_argument("--phase944-round", default="activation_weighted_mlp_channel_causal_audit")
    parser.add_argument("--phase937-round", default="semantic_reuse_difference_state_atlas")
    parser.add_argument("--phase939-round", default="bilingual_specificity_tightening_audit")
    parser.add_argument("--phase940-round", default="semantic_boundary_bridge_audit")
    parser.add_argument("--phase943-round", default="consensus_coordinate_component_mapping_audit")
    parser.add_argument("--domains", default="")
    parser.add_argument("--relations", default="category,color,function")
    parser.add_argument("--max-objects-per-domain", type=int, default=8)
    parser.add_argument("--templates-per-language", type=int, default=2)
    parser.add_argument("--min-train-per-label", type=int, default=2)
    parser.add_argument("--min-specific-margin", type=float, default=0.05)
    parser.add_argument("--min-specific-gain", type=float, default=0.05)
    parser.add_argument("--min-phase940-bridge-gain", type=float, default=0.02)
    parser.add_argument("--max-specs-per-pair", type=int, default=12)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--min-train-specs", type=int, default=4)
    parser.add_argument("--min-holdout-specs", type=int, default=3)
    parser.add_argument("--language-pairs", default="en->en,zh->zh")
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-samples-per-pair", type=int, default=6)
    parser.add_argument("--max-prompts", type=int, default=96)
    parser.add_argument("--prompt-protocols", default="short_answer,stop_explicit,eos_instruction,final_answer,chat_eos")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sampling-temperatures", default="0.7,1.0")
    parser.add_argument("--sampling-seeds", default="11,23,37")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload.get("status"), "models": payload.get("models"), "eos_positive": payload.get("total_eos_positive_count")}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is used")
    eval_model(args)


if __name__ == "__main__":
    main()
