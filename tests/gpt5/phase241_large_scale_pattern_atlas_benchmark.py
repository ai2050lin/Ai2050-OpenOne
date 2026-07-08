#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402


PHASE = 241
SOURCE_PHASE = 240
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
RESULT_ROOT = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark")
ROUND_DEFAULT = "large_scale_pattern_atlas_benchmark"

VARIANTS = [
    "full",
    "short_answer_instruction",
    "one_word_strict",
    "explain_instruction",
    "no_answer_anchor",
    "target_seeded",
]

FACTS = [
    ("apple", "color", "red", ["red", "crimson"], "fruit"),
    ("banana", "color", "yellow", ["yellow"], "fruit"),
    ("grass", "color", "green", ["green"], "plant"),
    ("snow", "color", "white", ["white"], "weather"),
    ("coal", "color", "black", ["black"], "material"),
    ("lemon", "taste", "sour", ["sour", "tart"], "fruit"),
    ("sugar", "taste", "sweet", ["sweet"], "food"),
    ("hammer", "function", "hit", ["hit", "hammer", "drive"], "tool"),
    ("wheel", "part_of", "car", ["car", "vehicle"], "part"),
    ("leaf", "part_of", "plant", ["plant", "tree"], "plant"),
    ("glass", "material", "sand", ["sand", "silica"], "material"),
    ("bird", "category", "animal", ["animal"], "animal"),
    ("rose", "color", "red", ["red"], "plant"),
    ("ocean", "color", "blue", ["blue"], "place"),
    ("fire", "cause", "heat", ["heat", "burning"], "event"),
    ("rain", "property", "wet", ["wet"], "weather"),
]

ZH = {
    "red": ["红色", "红"],
    "yellow": ["黄色", "黄"],
    "green": ["绿色", "绿"],
    "white": ["白色", "白"],
    "black": ["黑色", "黑"],
    "blue": ["蓝色", "蓝"],
    "sour": ["酸"],
    "sweet": ["甜"],
    "snow": ["雪"],
    "apple": ["苹果"],
    "banana": ["香蕉"],
    "grass": ["草"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_unique_jsonl(path: Path, new_rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + new_rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or row.get("case_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_word(text: str) -> str:
    return re.sub(r"^[\s\"'`.,:;!?，。；：！？]+|[\s\"'`.,:;!?，。；：！？]+$", "", text.lower())


def first_word(text: str) -> str:
    parts = text.replace("\n", " ").split()
    return clean_word(parts[0]) if parts else ""


def aliases_for(target: str, extra: list[str] | None = None) -> list[str]:
    aliases = [target]
    aliases.extend(extra or [])
    aliases.extend(ZH.get(target, []))
    return sorted(set(str(x) for x in aliases if str(x)))


def build_prompt(family: str, mode: str, fact: tuple[str, str, str, list[str], str], sample_idx: int) -> tuple[str, str, list[str], str]:
    obj, rel, target, aliases, category = fact
    rel_text = rel.replace("_", " ")
    expected = mode
    if family == "content_knowledge":
        prompts = {
            "object_attribute": f"What is the {rel_text} of {obj}?\nAnswer with one word.\nAnswer:",
            "object_relation_value": f"For the object {obj}, give the value of relation {rel_text}.\nAnswer:",
            "category_membership": f"What broad category does {obj} belong to?\nAnswer with one word.\nAnswer:",
            "function_answer": f"What is {obj} mainly used to do?\nAnswer with one word.\nAnswer:",
            "part_whole": f"{obj} is part of what larger thing?\nAnswer with one word.\nAnswer:",
            "material_answer": f"What material is {obj} associated with?\nAnswer with one word.\nAnswer:",
            "location_fact": f"Where would you normally find {obj}?\nAnswer briefly.\nAnswer:",
            "causal_fact": f"What does {obj} most directly cause or indicate?\nAnswer with one word.\nAnswer:",
        }
        if mode == "category_membership":
            target, aliases = category, [category]
        elif mode == "function_answer" and rel != "function":
            target, aliases = "use", ["use", "used"]
        elif mode == "part_whole" and rel != "part_of":
            target, aliases = "thing", ["thing", "object"]
        elif mode == "material_answer" and rel != "material":
            target, aliases = "matter", ["matter", "material"]
        return prompts[mode], target, aliases_for(target, aliases), expected
    if family == "output_protocol":
        base = f"What is the {rel_text} of {obj}?"
        prompts = {
            "short_answer": f"{base}\nAnswer briefly.\nAnswer:",
            "one_word": f"{base}\nAnswer with exactly one word.\nAnswer:",
            "explain_answer": f"{base}\nAnswer with the answer first, then one short reason using because.\nAnswer:",
            "repeat_answer": f"{base}\nAnswer by repeating the answer twice separated by a comma.\nAnswer:",
            "list_answer": f"{base}\nAnswer as a two-item list containing only the answer and object.\nAnswer:",
            "json_answer": f"{base}\nReturn JSON with key answer.\nAnswer:",
            "table_answer": f"{base}\nReturn a tiny table with columns object and answer.\nAnswer:",
            "stop_after_answer": f"{base}\nAnswer with one word and then stop.\nAnswer:",
        }
        return prompts[mode], target, aliases_for(target, aliases), expected
    if family == "reasoning_constraint":
        prompts = {
            "because_reason": f"Why is {obj} associated with {target}?\nAnswer using because.\nAnswer:",
            "if_then": f"If the {rel_text} of {obj} is {target}, what is the {rel_text} of {obj}?\nAnswer with one word.\nAnswer:",
            "negation": f"If {obj} is not unrelated to {target}, is {target} still relevant?\nAnswer yes or no.\nAnswer:",
            "double_negation": f"It is not false that {obj} has relation {rel_text} to {target}. What is the value?\nAnswer:",
            "comparison": f"Between {obj} and an unknown object, which one has known {rel_text} {target}?\nAnswer with the object name.\nAnswer:",
            "counterfactual": f"If {obj} did not have {rel_text} {target}, would {target} still be the answer?\nAnswer yes or no.\nAnswer:",
            "scope_binding": f"Only {obj}, not the distractor, has {rel_text} {target}. What has {target}?\nAnswer:",
            "multi_hop_reasoning": f"{obj} has {rel_text} {target}. The requested value is the known relation value. What is it?\nAnswer:",
        }
        if mode in {"negation", "counterfactual"}:
            return prompts[mode], "yes" if mode == "negation" else "no", ["yes"] if mode == "negation" else ["no"], expected
        if mode in {"comparison", "scope_binding"}:
            return prompts[mode], obj, [obj], expected
        return prompts[mode], target, aliases_for(target, aliases), expected
    if family == "syntax_structure":
        prompts = {
            "answer_anchor": f"Question: What is the {rel_text} of {obj}?\nAnswer:",
            "colon_boundary": f"What is the {rel_text} of {obj}? Answer:",
            "period_stop": f"What is the {rel_text} of {obj}? Answer with one word followed by a period.\nAnswer:",
            "comma_repeat": f"What is the {rel_text} of {obj}? Repeat the answer twice separated by comma.\nAnswer:",
            "newline_boundary": f"What is the {rel_text} of {obj}?\n\nAnswer:",
            "question_form": f"Is the {rel_text} of {obj} {target}?\nAnswer yes or no.\nAnswer:",
            "imperative_form": f"Name the {rel_text} of {obj} in one word.\nAnswer:",
            "clause_embedding": f"The object that has {rel_text} {target} is {obj}. What is the {rel_text} of {obj}?\nAnswer:",
        }
        if mode == "question_form":
            return prompts[mode], "yes", ["yes"], expected
        return prompts[mode], target, aliases_for(target, aliases), expected
    if family == "language_action":
        prompts = {
            "answer": f"What is the {rel_text} of {obj}?\nAnswer:",
            "explain": f"Explain why {obj} relates to {target} in one sentence.\nAnswer:",
            "summarize": f"Summarize this fact in three words: {obj} has {rel_text} {target}.\nAnswer:",
            "translate": f"Translate this answer to Chinese: {target}\nAnswer:",
            "classify": f"Classify {obj} by broad type.\nAnswer:",
            "rewrite": f"Rewrite as a short fact: {obj} - {rel_text} - {target}.\nAnswer:",
            "compare": f"Compare {obj} with a random object and name the known answer {target}.\nAnswer:",
            "format_convert": f"Convert to key=value format: {obj} {rel_text} {target}\nAnswer:",
        }
        if mode == "translate":
            zh = ZH.get(target, [target])[0]
            return prompts[mode], zh, [zh], expected
        if mode == "classify":
            return prompts[mode], category, [category], expected
        return prompts[mode], target, aliases_for(target, aliases), expected
    if family == "cross_lingual":
        zh_target = ZH.get(target, [target])[0]
        zh_obj = ZH.get(obj, [obj])[0]
        prompts = {
            "EN_to_EN": f"What is the {rel_text} of {obj}?\nAnswer in English.\nAnswer:",
            "ZH_to_ZH": f"{zh_obj} 的答案是什么？请用中文回答。\n答案：",
            "EN_to_ZH": f"What is the {rel_text} of {obj}? Answer in Chinese.\nAnswer:",
            "ZH_to_EN": f"{zh_obj} 的英文答案是什么？Answer in English.\nAnswer:",
            "EN_to_FR": f"What is the {rel_text} of {obj}? Answer in French if possible.\nAnswer:",
            "FR_to_EN": f"Répondez en anglais: the {rel_text} of {obj} is what?\nAnswer:",
            "cross_lingual_reasoning": f"Use English reasoning but answer in Chinese: {obj} has {rel_text} {target}. What is the answer?\nAnswer:",
            "cross_lingual_negation": f"Answer in English: {zh_obj} is not unrelated to {target}. Is {target} relevant? yes or no.\nAnswer:",
        }
        if mode in {"ZH_to_ZH", "EN_to_ZH", "cross_lingual_reasoning"}:
            return prompts[mode], zh_target, [zh_target, *ZH.get(target, [])], expected
        if mode == "cross_lingual_negation":
            return prompts[mode], "yes", ["yes"], expected
        return prompts[mode], target, aliases_for(target, aliases), expected
    if family == "readout_competition":
        prompts = {
            "target_answer": f"What is the {rel_text} of {obj}?\nAnswer with one word.\nAnswer:",
            "because_reason": f"What is the {rel_text} of {obj}?\nAnswer with the answer first, then because.\nAnswer:",
            "period_stop": f"What is the {rel_text} of {obj}?\nAnswer with a word and period.\nAnswer:",
            "for_continuation": f"For {obj}, the {rel_text} is what?\nAnswer:",
            "the_continuation": f"The {rel_text} of {obj} is\nAnswer:",
            "be_continuation": f"{obj} can be described as having {rel_text}\nAnswer:",
            "answer_boundary": f"Question: What is the {rel_text} of {obj}?\nAnswer:\nAnswer:",
            "newline_boundary": f"What is the {rel_text} of {obj}?\n\n\nAnswer:",
        }
        return prompts[mode], target, aliases_for(target, aliases), expected
    if family == "state_drift":
        prompts = {
            "early_correct_late_drift": f"What is the {rel_text} of {obj}? Start with the answer, then continue naturally.\nAnswer:",
            "over_generation": f"What is the {rel_text} of {obj}? Answer in one word, but include any helpful context.\nAnswer:",
            "echo_drift": f"Repeat the question then answer: What is the {rel_text} of {obj}?\nAnswer:",
            "format_drift": f"Answer with one word, then convert it to a sentence: What is the {rel_text} of {obj}?\nAnswer:",
            "next_task_drift": f"What is the {rel_text} of {obj}?\nAnswer, then wait for the next task.\nAnswer:",
            "explain_takeover": f"What is the {rel_text} of {obj}? Do not explain.\nAnswer:",
            "continuation_takeover": f"The answer to the {rel_text} of {obj} is\nAnswer:",
            "boundary_takeover": f"Answer the question.\n\nWhat is the {rel_text} of {obj}?\n\nAnswer:",
        }
        return prompts[mode], target, aliases_for(target, aliases), expected
    if family == "closure":
        prompts = {
            "answer_correct": f"What is the {rel_text} of {obj}?\nAnswer:",
            "pattern_matched": f"What is the {rel_text} of {obj}?\nAnswer with exactly one word.\nAnswer:",
            "boundary_stable": f"What is the {rel_text} of {obj}?\nAnswer with one word and no newline.\nAnswer:",
            "done_state_stable": f"What is the {rel_text} of {obj}?\nAnswer once and stop.\nAnswer:",
            "model_stop_executed": f"What is the {rel_text} of {obj}?\nAnswer with only the answer.\nAnswer:",
            "no_drift": f"What is the {rel_text} of {obj}?\nAnswer only, no extra words.\nAnswer:",
            "eos_pressure": f"What is the {rel_text} of {obj}?\nAnswer one word, then end.\nAnswer:",
            "period_closed": f"What is the {rel_text} of {obj}?\nAnswer one word followed by a period.\nAnswer:",
        }
        return prompts[mode], target, aliases_for(target, aliases), expected
    raise ValueError(f"unknown family/mode: {family}/{mode}")


FAMILY_MODES = {
    "content_knowledge": ["object_attribute", "object_relation_value", "category_membership", "function_answer", "part_whole", "material_answer", "location_fact", "causal_fact"],
    "output_protocol": ["short_answer", "one_word", "explain_answer", "repeat_answer", "list_answer", "json_answer", "table_answer", "stop_after_answer"],
    "reasoning_constraint": ["because_reason", "if_then", "negation", "double_negation", "comparison", "counterfactual", "scope_binding", "multi_hop_reasoning"],
    "syntax_structure": ["answer_anchor", "colon_boundary", "period_stop", "comma_repeat", "newline_boundary", "question_form", "imperative_form", "clause_embedding"],
    "language_action": ["answer", "explain", "summarize", "translate", "classify", "rewrite", "compare", "format_convert"],
    "cross_lingual": ["EN_to_EN", "ZH_to_ZH", "EN_to_ZH", "ZH_to_EN", "EN_to_FR", "FR_to_EN", "cross_lingual_reasoning", "cross_lingual_negation"],
    "readout_competition": ["target_answer", "because_reason", "period_stop", "for_continuation", "the_continuation", "be_continuation", "answer_boundary", "newline_boundary"],
    "state_drift": ["early_correct_late_drift", "over_generation", "echo_drift", "format_drift", "next_task_drift", "explain_takeover", "continuation_takeover", "boundary_takeover"],
    "closure": ["answer_correct", "pattern_matched", "boundary_stable", "done_state_stable", "model_stop_executed", "no_drift", "eos_pressure", "period_closed"],
}


def build_cases(samples_per_mode: int, family_filter: set[str] | None = None, mode_limit: int | None = None) -> list[dict[str, Any]]:
    cases = []
    for family, modes in FAMILY_MODES.items():
        if family_filter and family not in family_filter:
            continue
        selected_modes = modes[: mode_limit or len(modes)]
        for mode in selected_modes:
            for sample_idx in range(int(samples_per_mode)):
                fact = FACTS[(sample_idx + len(mode) + len(family)) % len(FACTS)]
                prompt, target, aliases, expected = build_prompt(family, mode, fact, sample_idx)
                cases.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase241",
                        "created_at": utc_now(),
                        "case_id": f"phase241_{family}_{mode}_{sample_idx:04d}",
                        "family_id": family,
                        "mode_id": mode,
                        "prompt": prompt,
                        "target": target,
                        "target_aliases": aliases,
                        "expected_pattern": expected,
                        "sample_index": sample_idx,
                    }
                )
    return cases


def prompt_variants(base_prompt: str, target: str) -> dict[str, str]:
    variants = p239.prompt_variants(base_prompt, target)
    stem = p239.strip_answer_anchor(base_prompt)
    variants["no_answer_anchor"] = stem
    return {k: v for k, v in variants.items() if k in VARIANTS}


def classify_large_scale(output: str, aliases: list[str], family: str, mode: str, variant_id: str) -> dict[str, Any]:
    normalized = output.strip()
    low = normalized.lower()
    alias_hit, matched_alias = p239.contains_alias(normalized, aliases)
    token_count = len(normalized.replace("\n", " ").split())
    starts_alias = any(low.startswith(str(a).lower()) for a in aliases if str(a) and not re.search(r"[\u4e00-\u9fff]", str(a)))
    starts_alias = starts_alias or any(normalized.startswith(str(a)) for a in aliases if re.search(r"[\u4e00-\u9fff]", str(a)))
    has_because = "because" in low or "因为" in normalized
    has_answer_loop = "answer:" in low or "答案：" in normalized
    has_json_shape = "{" in normalized and "}" in normalized
    has_table_shape = "|" in normalized or "\t" in normalized
    has_list_shape = "\n" in normalized or "," in normalized or "1." in normalized or "-" in normalized
    has_period_end = normalized.endswith(".") or normalized.endswith("。")
    semantic_match = bool(alias_hit)
    if family == "cross_lingual" and aliases:
        semantic_match = semantic_match or any(str(a).lower() in low for a in aliases)
    protocol_match = False
    if mode in {"short_answer", "one_word", "stop_after_answer", "pattern_matched", "boundary_stable", "done_state_stable", "model_stop_executed", "no_drift", "eos_pressure"}:
        protocol_match = semantic_match and token_count <= 3 and not has_because and not has_answer_loop
    elif mode in {"explain_answer", "because_reason", "explain", "explain_takeover"}:
        protocol_match = semantic_match and has_because
    elif mode in {"repeat_answer", "comma_repeat"}:
        protocol_match = semantic_match and ("," in normalized or "，" in normalized)
    elif mode in {"list_answer"}:
        protocol_match = semantic_match and has_list_shape
    elif mode == "json_answer":
        protocol_match = semantic_match and has_json_shape
    elif mode == "table_answer":
        protocol_match = semantic_match and has_table_shape
    elif mode in {"period_stop", "period_closed"}:
        protocol_match = semantic_match and has_period_end
    elif mode in {"translate", "EN_to_ZH", "ZH_to_ZH", "cross_lingual_reasoning"}:
        protocol_match = semantic_match
    else:
        protocol_match = semantic_match and (token_count <= 16 or starts_alias)
    closure_signal = token_count <= 24 and not has_answer_loop
    over_generation = semantic_match and not protocol_match and token_count > 3
    if not normalized:
        failure_type = "empty_output"
        negative_category = "rollout_failure"
        mechanism_hint = "empty_generation"
    elif not semantic_match:
        failure_type = "semantic_target_failure"
        negative_category = "semantic_failure"
        mechanism_hint = "target_not_reached"
    elif not protocol_match and family == "output_protocol":
        failure_type = "protocol_failure"
        negative_category = "protocol_negative"
        mechanism_hint = "protocol_state_or_closure_failure"
    elif not protocol_match and family == "closure":
        failure_type = "closure_failure"
        negative_category = "closure_negative"
        mechanism_hint = "done_state_missing"
    elif not protocol_match and family == "readout_competition":
        failure_type = "readout_competition_failure"
        negative_category = "readout_negative"
        mechanism_hint = "competitor_regime_takeover"
    elif over_generation:
        failure_type = "rollout_failure"
        negative_category = "rollout_negative"
        mechanism_hint = "over_generation_or_drift"
    elif not protocol_match:
        failure_type = "protocol_or_format_failure"
        negative_category = "protocol_negative"
        mechanism_hint = "format_or_pattern_mismatch"
    else:
        failure_type = "none"
        negative_category = "none"
        mechanism_hint = "none"
    score = 0.40 * float(semantic_match) + 0.25 * float(starts_alias or semantic_match) + 0.25 * float(protocol_match) + 0.10 * float(closure_signal)
    return {
        "output_text": normalized,
        "output_token_count": token_count,
        "semantic_match": semantic_match,
        "answer_hit": semantic_match,
        "matched_alias": matched_alias,
        "starts_alias": starts_alias,
        "protocol_match": protocol_match,
        "closure_signal": closure_signal,
        "over_generation": over_generation,
        "has_because": has_because,
        "has_answer_loop": has_answer_loop,
        "calibrated_behavior_score": round(score, 4),
        "failure_type": failure_type,
        "negative_result": failure_type != "none",
        "negative_category": negative_category,
        "mechanism_hint": mechanism_hint,
        "should_enter_hook": failure_type in {"protocol_failure", "readout_competition_failure", "closure_failure", "rollout_failure"},
    }


def batch_next_token_logits(model: Any, tokenizer: Any, device: torch.device, prompts: list[str], batch_size: int) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            result = model(**encoded)
        attention = encoded["attention_mask"]
        for i in range(len(chunk)):
            last_pos = int(attention[i].sum().item()) - 1
            out.append(result.logits[i, last_pos].detach().float().cpu())
        del result, encoded
    return out


def batch_generate(model: Any, tokenizer: Any, device: torch.device, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    outs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        input_lens = [int(x.sum().item()) for x in encoded["attention_mask"]]
        encoded = {k: v.to(device) for k, v in encoded.items()}
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            generated = model.generate(**encoded, **kwargs)
        for i, input_len in enumerate(input_lens):
            outs.append(tokenizer.decode(generated[i, input_len:], skip_special_tokens=True).strip())
        del generated, encoded
    return outs


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    family_filter = set(args.families.split(",")) if args.families else None
    cases = build_cases(args.samples_per_mode, family_filter, args.mode_limit)
    jobs = []
    for case in cases:
        for variant_id, prompt in prompt_variants(case["prompt"], case["target"]).items():
            jobs.append({**case, "variant_id": variant_id, "prompt_variant": prompt})
    if args.max_jobs:
        jobs = jobs[: int(args.max_jobs)]
    run_id = f"phase241:{args.model}:{args.round_name}"
    model = None
    tokenizer = None
    behavior_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        prompts = [x["prompt_variant"] for x in jobs]
        logits_list = batch_next_token_logits(model, tokenizer, device, prompts, args.batch_size)
        outputs = batch_generate(model, tokenizer, device, prompts, args.max_new_tokens, args.batch_size)
        for idx, (job, logits, output) in enumerate(zip(jobs, logits_list, outputs), start=1):
            readout = p239.readout_metrics(tokenizer, logits, list(job["target_aliases"]))
            behavior = classify_large_scale(
                output,
                list(job["target_aliases"]),
                str(job["family_id"]),
                str(job["mode_id"]),
                str(job["variant_id"]),
            )
            common = {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "run_id": run_id,
                "model": args.model,
                **job,
            }
            row = {**common, **readout, **behavior}
            behavior_rows.append(row)
            readout_rows.append(
                {
                    **common,
                    "target_logit": readout["target_logit"],
                    "target_rank": readout["target_rank"],
                    "target_margin_vs_winner": readout["target_margin_vs_winner"],
                    "winning_regime": readout["winning_regime"],
                    "second_competitor": readout["second_competitor"],
                    "winning_regime_logit": readout["winning_regime_logit"],
                    "top_token": readout["top_token"],
                    "regime_scores": readout["regime_scores"],
                }
            )
            if row["negative_result"]:
                negative_rows.append(row)
            if idx % max(1, args.log_every) == 0:
                log(f"{args.model}: jobs={idx}/{len(jobs)} negatives={len(negative_rows)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics = metric_rows(args.model, behavior_rows)
    observations = observation_rows(behavior_rows)
    edges = graph_edges(args.model, metrics)
    trace_vectors = trace_vector_rows(args.model, behavior_rows)
    summary = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Large-scale pattern atlas benchmark",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "case_count": len(cases),
        "job_count": len(jobs),
        "behavior_rows": len(behavior_rows),
        "readout_rows": len(readout_rows),
        "negative_rows": len(negative_rows),
        "metric_rows": len(metrics),
        "observation_rows": len(observations),
        "graph_edges": len(edges),
        "mean_score": round(mean(safe_float(x["calibrated_behavior_score"]) for x in behavior_rows), 4) if behavior_rows else 0.0,
        "semantic_match_rate": round(sum(1 for x in behavior_rows if x["semantic_match"]) / max(1, len(behavior_rows)), 4),
        "protocol_match_rate": round(sum(1 for x in behavior_rows if x["protocol_match"]) / max(1, len(behavior_rows)), 4),
        "negative_rate": round(len(negative_rows) / max(1, len(behavior_rows)), 4),
        "negative_categories": dict(Counter(str(x["negative_category"]) for x in negative_rows).most_common()),
    }
    write_json(out_dir / f"phase241_{args.model}_summary.json", summary)
    write_jsonl(out_dir / f"phase241_{args.model}_case_rows.jsonl", cases)
    write_jsonl(out_dir / f"phase241_{args.model}_behavior_rows.jsonl", behavior_rows)
    write_jsonl(out_dir / f"phase241_{args.model}_readout_rows.jsonl", readout_rows)
    write_jsonl(out_dir / f"phase241_{args.model}_negative_result_rows.jsonl", negative_rows)
    write_jsonl(out_dir / f"phase241_{args.model}_mode_trace_vectors.jsonl", trace_vectors)
    write_jsonl(out_dir / f"phase241_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase241_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase241_{args.model}_graph_edges.jsonl", edges)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "jobs": len(jobs), "negative_rate": summary["negative_rate"]}, ensure_ascii=False, indent=2))
    return summary


def metric_rows(model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["family_id"], row["mode_id"])].append(row)
    for (family, mode), items in buckets.items():
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase241",
                "created_at": now,
                "metric_id": f"phase241:{model}:{family}:{mode}:large_scale_behavior",
                "scope": "mode",
                "model": model,
                "family_id": family,
                "mode_id": mode,
                "metric_name": "large_scale_behavior_signature",
                "metric_value": round(mean(safe_float(x["calibrated_behavior_score"]) for x in items), 4),
                "semantic_match_rate": round(sum(1 for x in items if x["semantic_match"]) / len(items), 4),
                "protocol_match_rate": round(sum(1 for x in items if x["protocol_match"]) / len(items), 4),
                "negative_rate": round(sum(1 for x in items if x["negative_result"]) / len(items), 4),
                "over_generation_rate": round(sum(1 for x in items if x["over_generation"]) / len(items), 4),
                "mean_target_margin_vs_winner": round(mean(safe_float(x["target_margin_vs_winner"]) for x in items), 4),
                "winner_regimes": dict(Counter(str(x["winning_regime"]) for x in items).most_common()),
                "negative_categories": dict(Counter(str(x["negative_category"]) for x in items if x["negative_result"]).most_common()),
                "rows": len(items),
            }
        )
    return out


def observation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    observations = []
    for row in rows:
        for name, value in {
            "calibrated_behavior_score": row["calibrated_behavior_score"],
            "semantic_match": float(row["semantic_match"]),
            "protocol_match": float(row["protocol_match"]),
            "negative_result": float(row["negative_result"]),
            "target_margin_vs_winner": row["target_margin_vs_winner"],
            "target_rank": row["target_rank"],
            "over_generation": float(row["over_generation"]),
        }.items():
            observations.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase241",
                    "created_at": now,
                    "observation_id": f"phase241:{row['model']}:{row['case_id']}:{row['variant_id']}:{name}",
                    "run_id": row["run_id"],
                    "case_id": row["case_id"],
                    "model": row["model"],
                    "family_id": row["family_id"],
                    "mode_id": row["mode_id"],
                    "level": "large_scale_behavior_readout",
                    "metric_name": name,
                    "metric_value": safe_float(value),
                    "metric_unit": "score",
                    "variant_id": row["variant_id"],
                    "target": row["target"],
                    "semantic_match": row["semantic_match"],
                    "protocol_match": row["protocol_match"],
                    "winner_regime": row["winning_regime"],
                    "second_competitor": row["second_competitor"],
                    "failure_type": row["failure_type"],
                    "negative_result": row["negative_result"],
                    "negative_category": row["negative_category"],
                    "mechanism_hint": row["mechanism_hint"],
                    "should_enter_hook": row["should_enter_hook"],
                }
            )
    return observations


def graph_edges(model: str, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    for row in metrics:
        winner = next(iter((row.get("winner_regimes") or {"none": 0}).keys()))
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase241",
                "created_at": now,
                "edge_id": f"phase241:{model}:{row['family_id']}:{row['mode_id']}:{winner}",
                "source": f"mode:{row['family_id']}:{row['mode_id']}",
                "target": f"regime:{winner}",
                "edge_type": "large_scale_readout_regime",
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "model": model,
                "evidence_type": "large_scale_behavior_readout",
                "effect_direction": "negative_dominant" if safe_float(row["negative_rate"]) > 0.5 else "positive_or_mixed",
                "effect_size": safe_float(row["negative_rate"]),
                "confidence": round(0.35 + min(0.35, safe_float(row.get("rows")) / 100.0) + min(0.20, abs(safe_float(row["mean_target_margin_vs_winner"])) / 10.0), 4),
                "supporting_phases": ["Phase241"],
                "status": "behavior_tested",
            }
        )
    return out


def trace_vector_rows(model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["family_id"], row["mode_id"])].append(row)
    for (family, mode), items in buckets.items():
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase241",
                "created_at": utc_now(),
                "trace_vector_id": f"phase241:{model}:{family}:{mode}",
                "model": model,
                "family_id": family,
                "mode_id": mode,
                "B_behavior": round(mean(safe_float(x["calibrated_behavior_score"]) for x in items), 4),
                "T_trigger": round(sum(1 for x in items if x["variant_id"] != "full" and x["semantic_match"]) / max(1, len(items)), 4),
                "G_gate_product": None,
                "R_residual": None,
                "C_competition": dict(Counter(str(x["winning_regime"]) for x in items).most_common(5)),
                "O_rollout": {
                    "over_generation_rate": round(sum(1 for x in items if x["over_generation"]) / len(items), 4),
                    "mean_output_tokens": round(mean(safe_float(x["output_token_count"]) for x in items), 4),
                },
                "K_closure": round(sum(1 for x in items if x["closure_signal"]) / len(items), 4),
                "negative_categories": dict(Counter(str(x["negative_category"]) for x in items if x["negative_result"]).most_common()),
                "hook_candidate_count": sum(1 for x in items if x["should_enter_hook"]),
                "rows": len(items),
            }
        )
    return out


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    behavior_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    vectors: list[dict[str, Any]] = []
    summaries = []
    cases_by_id: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        summaries.append(read_json(out_dir / f"phase241_{model}_summary.json"))
        for case in read_jsonl(out_dir / f"phase241_{model}_case_rows.jsonl"):
            cases_by_id[case["case_id"]] = case
        behavior_rows.extend(read_jsonl(out_dir / f"phase241_{model}_behavior_rows.jsonl"))
        readout_rows.extend(read_jsonl(out_dir / f"phase241_{model}_readout_rows.jsonl"))
        negative_rows.extend(read_jsonl(out_dir / f"phase241_{model}_negative_result_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase241_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase241_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase241_{model}_graph_edges.jsonl"))
        vectors.extend(read_jsonl(out_dir / f"phase241_{model}_mode_trace_vectors.jsonl"))
    summaries = [x for x in summaries if x]
    if not summaries or not behavior_rows:
        payload = {
            "phase": PHASE,
            "source_phase": SOURCE_PHASE,
            "title": "Cross-model large-scale pattern atlas benchmark",
            "status": "empty",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "schema_version": SCHEMA_VERSION,
            "models": [],
            "case_count": 0,
            "behavior_rows": 0,
            "readout_rows": 0,
            "negative_rows": 0,
            "observation_rows": 0,
            "metric_rows": 0,
            "graph_edges": 0,
            "trace_vectors": 0,
            "mean_score": 0.0,
            "semantic_match_rate": 0.0,
            "protocol_match_rate": 0.0,
            "negative_rate": 0.0,
            "negative_categories": {},
            "top_negative_modes": [],
            "family_failure_matrix": {},
            "readout_regime_matrix": {},
        }
        write_json(out_dir / "phase241_cross_model_summary.json", payload)
        print(json.dumps({"phase": PHASE, "status": "empty", "models": [], "behavior_rows": 0, "negative_rate": 0.0}, ensure_ascii=False, indent=2))
        return payload
    family_matrix = family_failure_matrix(behavior_rows)
    regime_matrix = readout_regime_matrix(behavior_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model large-scale pattern atlas benchmark",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "case_count": len(cases_by_id),
        "behavior_rows": len(behavior_rows),
        "readout_rows": len(readout_rows),
        "negative_rows": len(negative_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "trace_vectors": len(vectors),
        "mean_score": round(mean(safe_float(x["calibrated_behavior_score"]) for x in behavior_rows), 4) if behavior_rows else 0.0,
        "semantic_match_rate": round(sum(1 for x in behavior_rows if x["semantic_match"]) / max(1, len(behavior_rows)), 4),
        "protocol_match_rate": round(sum(1 for x in behavior_rows if x["protocol_match"]) / max(1, len(behavior_rows)), 4),
        "negative_rate": round(len(negative_rows) / max(1, len(behavior_rows)), 4),
        "negative_categories": dict(Counter(str(x["negative_category"]) for x in negative_rows).most_common()),
        "top_negative_modes": top_negative_modes(metrics),
        "family_failure_matrix": family_matrix,
        "readout_regime_matrix": regime_matrix,
    }
    write_json(out_dir / "phase241_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase241_large_scale_case_rows.jsonl", list(cases_by_id.values()))
    write_jsonl(out_dir / "phase241_large_scale_behavior_rows.jsonl", behavior_rows)
    write_jsonl(out_dir / "phase241_large_scale_readout_rows.jsonl", readout_rows)
    write_jsonl(out_dir / "phase241_negative_result_rows.jsonl", negative_rows)
    write_jsonl(out_dir / "phase241_mode_trace_vectors.jsonl", vectors)
    write_json(out_dir / "phase241_family_failure_matrix.json", family_matrix)
    write_json(out_dir / "phase241_readout_regime_matrix.json", regime_matrix)
    write_jsonl(out_dir / "phase241_cross_model_observations.jsonl", observations)
    write_jsonl(out_dir / "phase241_cross_model_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase241_cross_model_graph_edges.jsonl", edges)
    write_report(out_dir / "phase241_large_scale_summary.md", payload)
    update_atlas(payload, list(cases_by_id.values()), observations, metrics, edges)
    print(json.dumps({"phase": PHASE, "status": "complete", "models": payload["models"], "behavior_rows": len(behavior_rows), "negative_rate": payload["negative_rate"]}, ensure_ascii=False, indent=2))
    return payload


def family_failure_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["family_id"], row["model"])].append(row)
    for (family, model), items in buckets.items():
        out.setdefault(family, {})[model] = {
            "rows": len(items),
            "score": round(mean(safe_float(x["calibrated_behavior_score"]) for x in items), 4),
            "semantic_match_rate": round(sum(1 for x in items if x["semantic_match"]) / len(items), 4),
            "protocol_match_rate": round(sum(1 for x in items if x["protocol_match"]) / len(items), 4),
            "negative_rate": round(sum(1 for x in items if x["negative_result"]) / len(items), 4),
            "negative_categories": dict(Counter(str(x["negative_category"]) for x in items if x["negative_result"]).most_common()),
        }
    return out


def readout_regime_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["family_id"], row["mode_id"], row["model"])].append(row)
    for (family, mode, model), items in buckets.items():
        out.setdefault(family, {}).setdefault(mode, {})[model] = {
            "rows": len(items),
            "winner_regimes": dict(Counter(str(x["winning_regime"]) for x in items).most_common()),
            "second_competitors": dict(Counter(str(x["second_competitor"]) for x in items).most_common()),
            "mean_target_margin_vs_winner": round(mean(safe_float(x["target_margin_vs_winner"]) for x in items), 4),
        }
    return out


def top_negative_modes(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(metrics, key=lambda x: (safe_float(x.get("negative_rate")), -safe_float(x.get("metric_value"))), reverse=True)
    return rows[:30]


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase241 Large-Scale Pattern Atlas Benchmark", ""]
    for key in ["case_count", "behavior_rows", "readout_rows", "negative_rows", "mean_score", "semantic_match_rate", "protocol_match_rate", "negative_rate"]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(["", "## Negative Categories", ""])
    for key, value in payload["negative_categories"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Negative Modes", "", "| model | family | mode | rows | score | negative | protocol | winners |", "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"])
    for row in payload["top_negative_modes"][:30]:
        lines.append(
            f"| {row.get('model')} | {row.get('family_id')} | {row.get('mode_id')} | {row.get('rows')} | "
            f"{row.get('metric_value')} | {row.get('negative_rate')} | {row.get('protocol_match_rate')} | {row.get('winner_regimes')} |"
        )
    lines.extend(["", "## Caution", "", "This phase is large-scale behavior/readout mapping. It intentionally treats negative results as atlas data and does not claim mechanism closure."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_atlas(payload: dict[str, Any], cases: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    append_unique_jsonl(ATLAS_ROOT / "test_cases.jsonl", cases, "case_id")
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    progress = read_json(ATLAS_ROOT / "progress.json")
    if progress:
        progress["phase_id"] = "Phase241"
        progress["created_at"] = utc_now()
        progress.setdefault("global_progress", {})["pattern_family_atlas"] = 0.62
        progress.setdefault("global_progress", {})["general_language_mechanism_confidence"] = 0.50
        progress.setdefault("levels", {})["behavior"] = 0.68
        progress.setdefault("levels", {})["readout_competition"] = 0.56
        progress.setdefault("levels", {})["large_scale_negative_taxonomy"] = 0.35
        progress["next_phase"] = "Phase242_high_value_internal_trace_selection"
        progress["latest_phase"] = {
            "phase_id": "Phase241",
            "title": "大规模模式族行为与读出图谱基准",
            "case_count": payload["case_count"],
            "behavior_rows": payload["behavior_rows"],
            "negative_rows": payload["negative_rows"],
            "mean_score": payload["mean_score"],
            "semantic_match_rate": payload["semantic_match_rate"],
            "protocol_match_rate": payload["protocol_match_rate"],
            "negative_rate": payload["negative_rate"],
            "negative_categories": payload["negative_categories"],
        }
        write_json(ATLAS_ROOT / "progress.json", progress)
    summary_path = ATLAS_ROOT / "summary.md"
    old = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    marker = "## Phase241 Large-Scale Pattern Atlas Update"
    if marker in old:
        old = old.split(marker, 1)[0].rstrip()
    addition = (
        f"\n{marker}\n\n"
        f"- case_count: {payload['case_count']}\n"
        f"- behavior_rows: {payload['behavior_rows']}\n"
        f"- readout_rows: {payload['readout_rows']}\n"
        f"- negative_rows: {payload['negative_rows']}\n"
        f"- mean_score: {payload['mean_score']}\n"
        f"- semantic_match_rate: {payload['semantic_match_rate']}\n"
        f"- protocol_match_rate: {payload['protocol_match_rate']}\n"
        f"- negative_rate: {payload['negative_rate']}\n"
        f"- negative_categories: {payload['negative_categories']}\n"
        f"- top_negative_modes: {payload['top_negative_modes'][:5]}\n"
    )
    summary_path.write_text(old.rstrip() + "\n" + addition, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase241 large-scale pattern atlas benchmark")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--samples-per-mode", type=int, default=12)
    parser.add_argument("--mode-limit", type=int)
    parser.add_argument("--families", default="")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=256)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    args = parser.parse_args()
    if not args.summarize and not args.model:
        parser.error("--model is required unless --summarize is set")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize_round(args.round_name)
    else:
        eval_model(args)


if __name__ == "__main__":
    main()
