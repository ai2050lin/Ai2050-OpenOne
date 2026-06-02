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

from hf_probe_env import load_probe_model, release_loaded
from phase289_contract_scan import tokenize
from phase301_passive_factor_closure import mean, select_bases, state_texts
from phase303_role_query_closure import first_token_id, state_roles
from phase304_role_query_template_calibration import order_group, sentence_group


REPO_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[phase305] {message}", flush=True)


def option_templates() -> list[dict[str, str]]:
    return [
        {"query_type": "agent", "template_id": "agent_who_performed", "template": "sentence: {sentence}. question: who performed the action, the {opt1} or the {opt2}? answer: the"},
        {"query_type": "agent", "template_id": "agent_who_did", "template": "sentence: {sentence}. question: who did the action, the {opt1} or the {opt2}? answer: the"},
        {"query_type": "agent", "template_id": "agent_which_actor", "template": "sentence: {sentence}. question: which actor carried out the event, the {opt1} or the {opt2}? answer: the"},
        {"query_type": "agent", "template_id": "agent_choose_doer", "template": "sentence: {sentence}. choose the doer: the {opt1} or the {opt2}. answer: the"},
        {"query_type": "patient", "template_id": "patient_who_received", "template": "sentence: {sentence}. question: who received the action, the {opt1} or the {opt2}? answer: the"},
        {"query_type": "patient", "template_id": "patient_who_affected", "template": "sentence: {sentence}. question: who was affected by the action, the {opt1} or the {opt2}? answer: the"},
        {"query_type": "patient", "template_id": "patient_which_target", "template": "sentence: {sentence}. question: which target received the event, the {opt1} or the {opt2}? answer: the"},
        {"query_type": "patient", "template_id": "patient_choose_receiver", "template": "sentence: {sentence}. choose the receiver: the {opt1} or the {opt2}. answer: the"},
    ]


def score_margin(logits: torch.Tensor, tokenizer: Any, correct: str, wrong: str) -> dict[str, Any]:
    correct_id = first_token_id(tokenizer, f" {correct}")
    wrong_id = first_token_id(tokenizer, f" {wrong}")
    correct_logit = float(logits[correct_id])
    wrong_logit = float(logits[wrong_id])
    return {
        "correct": correct,
        "wrong": wrong,
        "correct_token_id": correct_id,
        "wrong_token_id": wrong_id,
        "correct_token_piece": tokenizer.decode([correct_id]),
        "wrong_token_piece": tokenizer.decode([wrong_id]),
        "correct_logit": correct_logit,
        "wrong_logit": wrong_logit,
        "margin": correct_logit - wrong_logit,
        "correct_choice": correct_logit > wrong_logit,
    }


def baseline_logits(loaded: Any, prompt: str, max_seq_len: int) -> torch.Tensor:
    seq_len = min(max(len(loaded.tokenizer.encode(prompt, add_special_tokens=True)), 8), max_seq_len)
    with torch.no_grad():
        out = loaded.model(**tokenize(loaded, prompt, seq_len))
    return out.logits[0, -1, :].detach().cpu().float().clone()


def summarize(rows: list[dict[str, Any]], min_accuracy: float, min_margin: float) -> dict[str, Any]:
    by_state_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_template_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_order_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_option_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state_groups[(row["query_type"], row["template_id"], row["state"], row["sentence_group"])].append(row)
        by_template_groups[(row["query_type"], row["template_id"], row["sentence_group"])].append(row)
        by_order_groups[(row["query_type"], row["template_id"], row["sentence_group"], row["order_group"])].append(row)
        by_option_groups[(row["query_type"], row["template_id"], row["option_order"])].append(row)

    def pack(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "accuracy": mean([1.0 if item["correct_choice"] else 0.0 for item in items]),
            "mean_margin": mean([float(item["margin"]) for item in items]),
            "n": len(items),
        }

    by_state = [
        {
            "query_type": q,
            "template_id": t,
            "state": s,
            "sentence_group": g,
            **pack(items),
        }
        for (q, t, s, g), items in sorted(by_state_groups.items())
    ]
    by_template_group = [
        {
            "query_type": q,
            "template_id": t,
            "sentence_group": g,
            **pack(items),
        }
        for (q, t, g), items in sorted(by_template_groups.items())
    ]
    by_option_order = [
        {
            "query_type": q,
            "template_id": t,
            "option_order": o,
            **pack(items),
        }
        for (q, t, o), items in sorted(by_option_groups.items())
    ]
    by_order = [
        {
            "query_type": q,
            "template_id": t,
            "sentence_group": g,
            "order_group": o,
            **pack(items),
        }
        for (q, t, g, o), items in sorted(by_order_groups.items())
    ]

    reliable = []
    for q, t in sorted({(row["query_type"], row["template_id"]) for row in rows}):
        state_items = [row for row in by_state if row["query_type"] == q and row["template_id"] == t]
        option_items = [row for row in by_option_order if row["query_type"] == q and row["template_id"] == t]
        if not state_items:
            continue
        min_state_acc = min(float(row["accuracy"]) for row in state_items)
        min_state_margin = min(float(row["mean_margin"]) for row in state_items)
        min_option_acc = min(float(row["accuracy"]) for row in option_items) if option_items else 0.0
        reliable.append({
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
        "by_order": by_order,
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
    templates = option_templates()
    loaded = None
    try:
        loaded = load_probe_model(args.model)
        log(f"model={args.model} class={type(loaded.model).__name__} bases={len(bases)} templates={len(templates)}")
        rows: list[dict[str, Any]] = []
        start = time.time()
        for base_idx, base in enumerate(bases, start=1):
            texts = state_texts(base)
            for state in states:
                sentence = texts[state]
                roles = state_roles(base, state)
                option_orders = [
                    ("agent_first", base.agent, base.patient),
                    ("patient_first", base.patient, base.agent),
                ]
                for template in templates:
                    query_type = template["query_type"]
                    correct = roles[query_type]
                    wrong = roles["patient" if query_type == "agent" else "agent"]
                    for option_order, opt1, opt2 in option_orders:
                        prompt = template["template"].format(sentence=sentence, opt1=opt1, opt2=opt2)
                        logits = baseline_logits(loaded, prompt, args.max_seq_len)
                        scored = score_margin(logits, loaded.tokenizer, correct, wrong)
                        rows.append({
                            "base": base.name,
                            "agent": base.agent,
                            "patient": base.patient,
                            "verb": base.verb,
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
                            "finite": bool(torch.isfinite(logits).all().item()),
                            **scored,
                        })
            if base_idx % args.progress_every == 0:
                log(f"base {base_idx}/{len(bases)} rows={len(rows)} elapsed={time.time() - start:.1f}s")

        data = {
            "model": args.model,
            "class": type(loaded.model).__name__,
            "complete": True,
            "num_bases": len(bases),
            "states": states,
            "templates": templates,
            "num_rows": len(rows),
            "rows": rows,
            "summary": summarize(rows, args.min_accuracy, args.min_margin),
        }
        out_file = output_dir / f"{args.model}_phase305_role_query_option_calibration.json"
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log(f"saved {out_file}")
        return data
    finally:
        if not args.hard_exit_after_model:
            release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase305_role_query_option_calibration"))
    parser.add_argument("--max-bases", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=305)
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
