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
from phase301_passive_factor_closure import PassiveBase, mean, select_bases, state_texts
from phase303_role_query_closure import first_token_id, state_roles


REPO_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[phase304] {message}", flush=True)


def query_templates() -> list[dict[str, str]]:
    return [
        {"query_type": "agent", "template_id": "agent_did_action", "template": "{sentence}. the one who did the action was the"},
        {"query_type": "agent", "template_id": "agent_performed_action", "template": "{sentence}. the one who performed the action was the"},
        {"query_type": "agent", "template_id": "agent_doer", "template": "{sentence}. the doer of the action was the"},
        {"query_type": "agent", "template_id": "agent_actor", "template": "{sentence}. the actor in this event was the"},
        {"query_type": "agent", "template_id": "agent_acted", "template": "{sentence}. the one who acted was the"},
        {"query_type": "agent", "template_id": "agent_done_by", "template": "{sentence}. the action was done by the"},
        {"query_type": "agent", "template_id": "agent_responsible", "template": "{sentence}. the one responsible for the action was the"},
        {"query_type": "agent", "template_id": "agent_carried_out", "template": "{sentence}. the one carrying out the action was the"},
        {"query_type": "patient", "template_id": "patient_affected", "template": "{sentence}. the one affected by the action was the"},
        {"query_type": "patient", "template_id": "patient_received", "template": "{sentence}. the one who received the action was the"},
        {"query_type": "patient", "template_id": "patient_recipient", "template": "{sentence}. the recipient of the action was the"},
        {"query_type": "patient", "template_id": "patient_acted_on", "template": "{sentence}. the one acted on was the"},
        {"query_type": "patient", "template_id": "patient_action_affected", "template": "{sentence}. the action affected the"},
        {"query_type": "patient", "template_id": "patient_target", "template": "{sentence}. the target of the action was the"},
        {"query_type": "patient", "template_id": "patient_happened_to", "template": "{sentence}. the one that the action happened to was the"},
        {"query_type": "patient", "template_id": "patient_receiving", "template": "{sentence}. the one receiving the action was the"},
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


def sentence_group(state: str) -> str:
    if state.startswith("active"):
        return "active"
    if state.startswith("passive"):
        return "passive_by"
    return "unknown"


def order_group(state: str) -> str:
    return "ab" if "_ab" in state else "ba"


def summarize(rows: list[dict[str, Any]], min_accuracy: float, min_margin: float) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    state_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    order_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["query_type"], row["template_id"], row["sentence_group"])
        groups[key].append(row)
        state_groups[(row["query_type"], row["template_id"], row["state"], row["sentence_group"])].append(row)
        order_groups[(row["query_type"], row["template_id"], row["sentence_group"], row["order_group"])].append(row)

    by_template_group = []
    for (query_type, template_id, sent_group), items in sorted(groups.items()):
        by_template_group.append({
            "query_type": query_type,
            "template_id": template_id,
            "sentence_group": sent_group,
            "accuracy": mean([1.0 if item["correct_choice"] else 0.0 for item in items]),
            "mean_margin": mean([float(item["margin"]) for item in items]),
            "n": len(items),
        })

    by_state = []
    for (query_type, template_id, state, sent_group), items in sorted(state_groups.items()):
        by_state.append({
            "query_type": query_type,
            "template_id": template_id,
            "state": state,
            "sentence_group": sent_group,
            "accuracy": mean([1.0 if item["correct_choice"] else 0.0 for item in items]),
            "mean_margin": mean([float(item["margin"]) for item in items]),
            "n": len(items),
        })

    by_order = []
    for (query_type, template_id, sent_group, order), items in sorted(order_groups.items()):
        by_order.append({
            "query_type": query_type,
            "template_id": template_id,
            "sentence_group": sent_group,
            "order_group": order,
            "accuracy": mean([1.0 if item["correct_choice"] else 0.0 for item in items]),
            "mean_margin": mean([float(item["margin"]) for item in items]),
            "n": len(items),
        })

    reliable = []
    template_keys = sorted({(row["query_type"], row["template_id"]) for row in rows})
    for query_type, template_id in template_keys:
        state_items = [row for row in by_state if row["query_type"] == query_type and row["template_id"] == template_id]
        order_items = [row for row in by_order if row["query_type"] == query_type and row["template_id"] == template_id]
        if not state_items:
            continue
        min_state_acc = min(float(row["accuracy"]) for row in state_items)
        min_state_margin = min(float(row["mean_margin"]) for row in state_items)
        min_order_acc = min(float(row["accuracy"]) for row in order_items) if order_items else 0.0
        reliable.append({
            "query_type": query_type,
            "template_id": template_id,
            "min_state_accuracy": min_state_acc,
            "min_state_mean_margin": min_state_margin,
            "min_order_accuracy": min_order_acc,
            "passes": bool(min_state_acc >= min_accuracy and min_state_margin >= min_margin),
        })

    return {
        "by_template_group": by_template_group,
        "by_state": by_state,
        "by_order": by_order,
        "reliable_templates": reliable,
        "num_reliable": sum(1 for row in reliable if row["passes"]),
        "nonfinite_rows": sum(1 for row in rows if not row.get("finite", True)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bases = select_bases(args.max_bases, args.seed)
    states = ["active_ab", "active_ba", "passive_ab_by", "passive_ba_by"]
    templates = query_templates()
    if args.query_types:
        allowed = {item.strip() for item in args.query_types.split(",") if item.strip()}
        templates = [item for item in templates if item["query_type"] in allowed]
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
                for template in templates:
                    query_type = template["query_type"]
                    prompt = template["template"].format(sentence=sentence)
                    logits = baseline_logits(loaded, prompt, args.max_seq_len)
                    correct = roles[query_type]
                    wrong = roles["patient" if query_type == "agent" else "agent"]
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
        out_file = output_dir / f"{args.model}_phase304_role_query_template_calibration.json"
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log(f"saved {out_file}")
        return data
    finally:
        if not args.hard_exit_after_model:
            release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase304_role_query_template_calibration"))
    parser.add_argument("--max-bases", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--query-types", default="")
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=304)
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
