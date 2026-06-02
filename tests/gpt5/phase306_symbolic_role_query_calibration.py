from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hf_probe_env import load_probe_model, release_loaded
from phase289_contract_scan import tokenize
from phase301_passive_factor_closure import mean, select_bases


REPO_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[phase306] {message}", flush=True)


def symbolic_states(verb: str, entity_style: str) -> dict[str, str]:
    if entity_style == "ab":
        a, b = "A", "B"
    elif entity_style == "entity_ab":
        a, b = "ENTITY_A", "ENTITY_B"
    elif entity_style == "nonce":
        a, b = "dax", "wug"
    else:
        raise ValueError(f"unknown entity_style={entity_style}")
    return {
        "active_ab": f"{a} {verb} {b}",
        "active_ba": f"{b} {verb} {a}",
        "passive_ab_by": f"{b} was {verb} by {a}",
        "passive_ba_by": f"{a} was {verb} by {b}",
    }


def role_labels(state: str, answer_style: str) -> dict[str, str]:
    if answer_style == "letter":
        a, b = "A", "B"
    elif answer_style == "entity":
        a, b = "ENTITY_A", "ENTITY_B"
    else:
        raise ValueError(f"unknown answer_style={answer_style}")
    if state in {"active_ab", "passive_ab_by"}:
        return {"agent": a, "patient": b}
    if state in {"active_ba", "passive_ba_by"}:
        return {"agent": b, "patient": a}
    raise ValueError(f"unknown state={state}")


def sentence_group(state: str) -> str:
    return "active" if state.startswith("active") else "passive_by"


def order_group(state: str) -> str:
    return "ab" if "_ab" in state else "ba"


def symbolic_templates() -> list[dict[str, str]]:
    return [
        {
            "query_type": "agent",
            "template_id": "role_table_agent",
            "template": "Sentence: {sentence}\nRole table:\nAGENT =",
        },
        {
            "query_type": "patient",
            "template_id": "role_table_patient",
            "template": "Sentence: {sentence}\nRole table:\nPATIENT =",
        },
        {
            "query_type": "agent",
            "template_id": "forced_agent",
            "template": "Sentence: {sentence}\nChoose the AGENT from {opt1} and {opt2}.\nAGENT =",
        },
        {
            "query_type": "patient",
            "template_id": "forced_patient",
            "template": "Sentence: {sentence}\nChoose the PATIENT from {opt1} and {opt2}.\nPATIENT =",
        },
        {
            "query_type": "agent",
            "template_id": "json_agent",
            "template": "Sentence: {sentence}\nReturn JSON.\n{{\"AGENT\":\"",
        },
        {
            "query_type": "patient",
            "template_id": "json_patient",
            "template": "Sentence: {sentence}\nReturn JSON.\n{{\"PATIENT\":\"",
        },
        {
            "query_type": "agent",
            "template_id": "compact_agent",
            "template": "S: {sentence}\nAgt:",
        },
        {
            "query_type": "patient",
            "template_id": "compact_patient",
            "template": "S: {sentence}\nPat:",
        },
    ]


def answer_text(answer_style: str, label: str) -> str:
    if answer_style == "letter":
        return f" {label}"
    if answer_style == "entity":
        return f" {label}"
    raise ValueError(f"unknown answer_style={answer_style}")


def sequence_logprob(loaded: Any, prompt: str, answer: str, max_seq_len: int) -> dict[str, Any]:
    prompt_ids = loaded.tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = loaded.tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise ValueError(f"empty answer tokenization for {answer!r}")
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[-max_seq_len:]
        prompt_len = max(0, len(input_ids) - len(answer_ids))
    else:
        prompt_len = len(prompt_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=loaded.input_device)
    with torch.no_grad():
        logits = loaded.model(input_ids=input_tensor).logits[0].detach().float()
    token_logprobs: list[float] = []
    token_pieces: list[str] = []
    for idx in range(prompt_len, len(input_ids)):
        prev_idx = idx - 1
        if prev_idx < 0:
            continue
        target_id = int(input_ids[idx])
        log_probs = F.log_softmax(logits[prev_idx], dim=-1)
        token_logprobs.append(float(log_probs[target_id].cpu()))
        token_pieces.append(loaded.tokenizer.decode([target_id]))
    total = sum(token_logprobs)
    return {
        "answer_ids": answer_ids,
        "answer_pieces": token_pieces,
        "answer_logprob": total,
        "answer_mean_logprob": total / max(1, len(token_logprobs)),
        "num_answer_tokens": len(token_logprobs),
        "finite": all(torch.isfinite(torch.tensor(token_logprobs)).tolist()) if token_logprobs else False,
    }


def summarize(rows: list[dict[str, Any]], min_accuracy: float, min_margin: float) -> dict[str, Any]:
    by_state_groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_option_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_template_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key_state = (
            row["entity_style"],
            row["answer_style"],
            row["query_type"],
            row["template_id"],
            row["state"],
            row["sentence_group"],
        )
        by_state_groups[key_state].append(row)
        key_option = (
            row["entity_style"],
            row["answer_style"],
            row["query_type"],
            row["template_id"],
            row["option_order"],
        )
        by_option_groups[key_option].append(row)
        key_template = (
            row["entity_style"],
            row["answer_style"],
            row["query_type"],
            row["template_id"],
            row["sentence_group"],
        )
        by_template_groups[key_template].append(row)

    def pack(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "accuracy": mean([1.0 if item["correct_choice"] else 0.0 for item in items]),
            "mean_margin": mean([float(item["margin"]) for item in items]),
            "mean_correct_logprob": mean([float(item["correct_logprob"]) for item in items]),
            "mean_wrong_logprob": mean([float(item["wrong_logprob"]) for item in items]),
            "n": len(items),
        }

    by_state = [
        {
            "entity_style": es,
            "answer_style": ans,
            "query_type": q,
            "template_id": t,
            "state": s,
            "sentence_group": g,
            **pack(items),
        }
        for (es, ans, q, t, s, g), items in sorted(by_state_groups.items())
    ]
    by_option_order = [
        {
            "entity_style": es,
            "answer_style": ans,
            "query_type": q,
            "template_id": t,
            "option_order": o,
            **pack(items),
        }
        for (es, ans, q, t, o), items in sorted(by_option_groups.items())
    ]
    by_template_group = [
        {
            "entity_style": es,
            "answer_style": ans,
            "query_type": q,
            "template_id": t,
            "sentence_group": g,
            **pack(items),
        }
        for (es, ans, q, t, g), items in sorted(by_template_groups.items())
    ]

    reliable = []
    keys = sorted({(row["entity_style"], row["answer_style"], row["query_type"], row["template_id"]) for row in rows})
    for es, ans, q, t in keys:
        state_items = [row for row in by_state if row["entity_style"] == es and row["answer_style"] == ans and row["query_type"] == q and row["template_id"] == t]
        option_items = [row for row in by_option_order if row["entity_style"] == es and row["answer_style"] == ans and row["query_type"] == q and row["template_id"] == t]
        if not state_items:
            continue
        min_state_acc = min(float(row["accuracy"]) for row in state_items)
        min_state_margin = min(float(row["mean_margin"]) for row in state_items)
        min_option_acc = min(float(row["accuracy"]) for row in option_items) if option_items else 0.0
        reliable.append({
            "entity_style": es,
            "answer_style": ans,
            "query_type": q,
            "template_id": t,
            "min_state_accuracy": min_state_acc,
            "min_state_mean_margin": min_state_margin,
            "min_option_accuracy": min_option_acc,
            "passes": bool(min_state_acc >= min_accuracy and min_state_margin >= min_margin and min_option_acc >= min_accuracy),
        })

    return {
        "by_template_group": by_template_group,
        "by_state": by_state,
        "by_option_order": by_option_order,
        "reliable_templates": reliable,
        "num_reliable": sum(1 for row in reliable if row["passes"]),
        "nonfinite_rows": sum(1 for row in rows if not row.get("finite", True)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bases = select_bases(args.max_bases, args.seed)
    states = ["active_ab", "active_ba", "passive_ab_by", "passive_ba_by"]
    entity_styles = [item.strip() for item in args.entity_styles.split(",") if item.strip()]
    answer_styles = [item.strip() for item in args.answer_styles.split(",") if item.strip()]
    templates = symbolic_templates()
    loaded = None
    try:
        loaded = load_probe_model(args.model)
        log(
            f"model={args.model} class={type(loaded.model).__name__} bases={len(bases)} "
            f"entity_styles={entity_styles} answer_styles={answer_styles} templates={len(templates)}"
        )
        rows: list[dict[str, Any]] = []
        start = time.time()
        for base_idx, base in enumerate(bases, start=1):
            for entity_style in entity_styles:
                texts = symbolic_states(base.verb, entity_style)
                for state in states:
                    sentence = texts[state]
                    for answer_style in answer_styles:
                        labels = role_labels(state, answer_style)
                        label_a = "A" if answer_style == "letter" else "ENTITY_A"
                        label_b = "B" if answer_style == "letter" else "ENTITY_B"
                        option_orders = [
                            ("a_first", label_a, label_b),
                            ("b_first", label_b, label_a),
                        ]
                        for template in templates:
                            query_type = template["query_type"]
                            correct = labels[query_type]
                            wrong = labels["patient" if query_type == "agent" else "agent"]
                            for option_order, opt1, opt2 in option_orders:
                                prompt = template["template"].format(sentence=sentence, opt1=opt1, opt2=opt2)
                                correct_score = sequence_logprob(loaded, prompt, answer_text(answer_style, correct), args.max_seq_len)
                                wrong_score = sequence_logprob(loaded, prompt, answer_text(answer_style, wrong), args.max_seq_len)
                                margin = float(correct_score["answer_logprob"]) - float(wrong_score["answer_logprob"])
                                rows.append({
                                    "base": base.name,
                                    "verb": base.verb,
                                    "entity_style": entity_style,
                                    "answer_style": answer_style,
                                    "state": state,
                                    "sentence": sentence,
                                    "sentence_group": sentence_group(state),
                                    "order_group": order_group(state),
                                    "query_type": query_type,
                                    "template_id": template["template_id"],
                                    "template": template["template"],
                                    "option_order": option_order,
                                    "opt1": opt1,
                                    "opt2": opt2,
                                    "prompt": prompt,
                                    "correct": correct,
                                    "wrong": wrong,
                                    "correct_logprob": correct_score["answer_logprob"],
                                    "wrong_logprob": wrong_score["answer_logprob"],
                                    "correct_mean_logprob": correct_score["answer_mean_logprob"],
                                    "wrong_mean_logprob": wrong_score["answer_mean_logprob"],
                                    "correct_num_tokens": correct_score["num_answer_tokens"],
                                    "wrong_num_tokens": wrong_score["num_answer_tokens"],
                                    "correct_pieces": correct_score["answer_pieces"],
                                    "wrong_pieces": wrong_score["answer_pieces"],
                                    "margin": margin,
                                    "correct_choice": margin > 0,
                                    "finite": bool(correct_score["finite"] and wrong_score["finite"]),
                                })
            if base_idx % args.progress_every == 0:
                log(f"base {base_idx}/{len(bases)} rows={len(rows)} elapsed={time.time() - start:.1f}s")

        data = {
            "model": args.model,
            "class": type(loaded.model).__name__,
            "complete": True,
            "num_bases": len(bases),
            "states": states,
            "entity_styles": entity_styles,
            "answer_styles": answer_styles,
            "templates": templates,
            "num_rows": len(rows),
            "rows": rows,
            "summary": summarize(rows, args.min_accuracy, args.min_margin),
        }
        out_file = output_dir / f"{args.model}_phase306_symbolic_role_query_calibration.json"
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log(f"saved {out_file}")
        return data
    finally:
        if not args.hard_exit_after_model:
            release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase306_symbolic_role_query_calibration"))
    parser.add_argument("--max-bases", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--entity-styles", default="ab,entity_ab,nonce")
    parser.add_argument("--answer-styles", default="letter,entity")
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=306)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        data = run(args)
        log(
            f"done rows={data['num_rows']} reliable={data['summary']['num_reliable']} "
            f"nonfinite={data['summary']['nonfinite_rows']}"
        )
    finally:
        if args.hard_exit_after_model:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
