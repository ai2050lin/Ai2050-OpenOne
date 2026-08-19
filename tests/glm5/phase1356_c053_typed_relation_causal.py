#!/usr/bin/env python3
"""Phase1356: frozen typed-donor causal test for a qualified C053 shared field."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1356, "C053"
CONTRACT = TESTS / "result/phase1353_c053_route_portfolio_contract"
FIELD = TESTS / "result/phase1355_c053_full_width_route_fields"
OUT = TESTS / "result/phase1356_c053_typed_relation_causal"
MODEL = "qwen3"
ARMS = ("baseline", "self", "same_family_true_donor", "different_family_true_donor",
        "same_family_false_donor")


def parents():
    final = core.load(FIELD / "analysis/final.json")
    audit = core.load(FIELD / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1356_c053_typed_causal" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1355 did not authorize causal work")
    field = core.load(FIELD / "analysis/quartet_field_summary.json")
    layer = field.get("shared_relation_selected_layer")
    if layer is None or not field.get("shared_relation_qualified"):
        raise RuntimeError("no qualified shared relation layer")
    return core.load(CONTRACT / "protocol/preregistration.json"), int(layer)


def prepare():
    protocol, layer = parents()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists():
        raise RuntimeError("Phase1356 manifest already exists")
    source = core.rows(CONTRACT / "material/b1_binary_cases.jsonl")
    compiled = core.rows(CONTRACT / "compiled/qwen3_B1_binary.jsonl")
    by_id = {x["case_id"]: y for x, y in zip(source, compiled)}
    lookup = {(x["partition"], x["surface"], x["target"], x["tested_family"]): x for x in source}
    family_words = defaultdict(list)
    for row in source:
        if row["surface"] == "ordinary":
            family_words[(row["partition"], row["target_family"])].append(row["target"])
    family_words = {key: sorted(set(value)) for key, value in family_words.items()}
    recipients = [x for x in source if x["partition"] in protocol["causal_gate"]["recipient_partitions"]
                  and x["surface"] == protocol["causal_gate"]["surface"] and not x["truth"]]
    entries = []
    for recipient in recipients:
        partition = recipient["partition"]
        tested = recipient["tested_family"]
        target_family = recipient["target_family"]
        valid_words = family_words[(partition, tested)]
        stable_index = int(recipient["case_id"].split("-")[-1]) % len(valid_words)
        correct_word = valid_words[stable_index]
        correct = lookup[(partition, "ordinary", correct_word, tested)]
        other_true_family = target_family
        different_word = family_words[(partition, other_true_family)][(stable_index + 1) % len(valid_words)]
        different = lookup[(partition, "ordinary", different_word, other_true_family)]
        held_families = sorted({x["target_family"] for x in source if x["partition"] == partition})
        false_family = next(f for f in held_families if f not in (tested, target_family))
        false_word = family_words[(partition, false_family)][(stable_index + 2) % len(valid_words)]
        false_donor = lookup[(partition, "ordinary", false_word, tested)]
        if not correct["truth"] or not different["truth"] or false_donor["truth"]:
            raise RuntimeError("donor truth typing failed")
        entries.append({
            "recipient": recipient["case_id"],
            "self": recipient["case_id"],
            "same_family_true_donor": correct["case_id"],
            "different_family_true_donor": different["case_id"],
            "same_family_false_donor": false_donor["case_id"],
            "partition": partition,
            "recipient_target_family": target_family,
            "recipient_tested_family": tested,
        })
    # The stable mapping is material-derived and fixed before any causal model output.
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "field_parent_sha256": core.sha(FIELD / "analysis/final.json"), "model": MODEL,
        "layer": layer, "site_role": protocol["causal_gate"]["site_role"], "arms": list(ARMS),
        "batch_recipients": 2, "recipient_count": len(entries), "entries": entries,
        "gate": protocol["causal_gate"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(path, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, indent=2))


def make_batch(rows, width, pad, device):
    ids = torch.full((len(rows), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[i, :len(value)] = value
        mask[i, :len(value)] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def margin(logits, candidate_ids):
    return float(logits[candidate_ids[0][0]].float() - logits[candidate_ids[1][0]].float())


@torch.inference_mode()
def run():
    protocol, selected_layer = parents()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    source = core.rows(CONTRACT / "material/b1_binary_cases.jsonl")
    compiled = core.rows(CONTRACT / "compiled/qwen3_B1_binary.jsonl")
    source_by_id = {x["case_id"]: x for x in source}
    compiled_by_id = {x["case_id"]: x for x in compiled}
    width = max(len(x["prompt_ids"]) for x in compiled)
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        layer = model.model.layers[selected_layer]
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for start in range(0, len(manifest["entries"]), manifest["batch_recipients"]):
            group = manifest["entries"][start:start + manifest["batch_recipients"]]
            donor_rows = []
            for entry in group:
                donor_rows.extend([compiled_by_id[entry[arm]] for arm in ARMS[1:]])
            dids, dmask, dpos = make_batch(donor_rows, width, pad, device)
            dkw = {"input_ids": dids, "attention_mask": dmask, "position_ids": dpos,
                   "use_cache": False, "output_hidden_states": True, "return_dict": True}
            if supports:
                dkw["logits_to_keep"] = 1
            donor_out = model(**dkw)
            donor_vectors = []
            for i, row in enumerate(donor_rows):
                donor_vectors.append(donor_out.hidden_states[selected_layer][i, row["tested_family_span"]].mean(0))

            target_rows, patch_rows, patch_positions, patch_vectors = [], [], [], []
            for local, entry in enumerate(group):
                recipient = compiled_by_id[entry["recipient"]]
                base = len(target_rows)
                target_rows.extend([recipient] * len(ARMS))
                for arm_index in range(1, len(ARMS)):
                    row_index = base + arm_index
                    vector = donor_vectors[local * 4 + arm_index - 1]
                    for position in recipient["tested_family_span"]:
                        patch_rows.append(row_index)
                        patch_positions.append(position)
                        patch_vectors.append(vector)
            tids, tmask, tpos = make_batch(target_rows, width, pad, device)
            rows_t = torch.tensor(patch_rows, dtype=torch.long, device=device)
            positions_t = torch.tensor(patch_positions, dtype=torch.long, device=device)
            vectors_t = torch.stack(patch_vectors)

            def hook(_module, args):
                hidden = args[0].clone()
                hidden[rows_t, positions_t] = vectors_t
                return (hidden,) + args[1:]

            handle = layer.register_forward_pre_hook(hook)
            try:
                tkw = {"input_ids": tids, "attention_mask": tmask, "position_ids": tpos,
                       "use_cache": False, "return_dict": True}
                if supports:
                    tkw["logits_to_keep"] = 1
                output = model(**tkw)
            finally:
                handle.remove()
            for local, entry in enumerate(group):
                recipient = compiled_by_id[entry["recipient"]]
                source_row = source_by_id[entry["recipient"]]
                base = local * len(ARMS)
                values = {arm: margin(output.logits[base + i, -1], recipient["candidate_ids"])
                          for i, arm in enumerate(ARMS)}
                baseline = values["baseline"]
                gains = {arm: values[arm] - baseline for arm in ARMS}
                records.append({**entry, "quartet_key": source_row["quartet_key"],
                                "margins": values, "gains": gains})
            del donor_out, output
        core.write_rows(OUT / "raw/qwen3_typed_causal.jsonl", records)
        correct = [x["gains"]["same_family_true_donor"] for x in records]
        different = [x["gains"]["different_family_true_donor"] for x in records]
        false = [x["gains"]["same_family_false_donor"] for x in records]
        self_diffs = [abs(x["gains"]["self"]) for x in records]
        advantage = [c - max(d, f) for c, d, f in zip(correct, different, false)]
        win = [c > d and c > f for c, d, f in zip(correct, different, false)]
        metrics = {
            "count": len(records), "correct_gain_median": statistics.median(correct),
            "correct_direction_fraction": sum(x > 0 for x in correct) / len(correct),
            "correct_over_wrong_median": statistics.median(advantage),
            "correct_over_wrong_win_fraction": sum(win) / len(win),
            "self_max_abs_diff": max(self_diffs),
            "partition_direction_fraction": {
                p: sum(x["gains"]["same_family_true_donor"] > 0 for x in records if x["partition"] == p)
                / sum(x["partition"] == p for x in records)
                for p in sorted({x["partition"] for x in records})
            },
        }
        gate = manifest["gate"]
        checks = {
            "gain": metrics["correct_gain_median"] >= gate["false_to_true_gain_min"],
            "direction": metrics["correct_direction_fraction"] >= gate["direction_fraction_min"],
            "selective_median": metrics["correct_over_wrong_median"] >= gate["correct_over_wrong_median_min"],
            "selective_win": metrics["correct_over_wrong_win_fraction"] >= gate["correct_over_wrong_win_min"],
            "self": metrics["self_max_abs_diff"] <= gate["self_max_abs_diff_max"],
        }
        summary = {"phase": PHASE, "campaign": CAMPAIGN, "model": MODEL, "layer": selected_layer,
                   "metrics": metrics, "checks": checks, "qualified": all(checks.values()),
                   "runtime": {"placement": placement, "quantization": quant,
                               "finished_at_utc": datetime.now(timezone.utc).isoformat()},
                   "claim_boundary": "typed whole-state donor selectivity at one frozen role/layer; no component minimality"}
        core.save(OUT / "analysis/qwen3_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize():
    summary = core.load(OUT / "analysis/qwen3_summary.json")
    final = {"phase": PHASE, "campaign": CAMPAIGN, "typed_causal_qualified": summary["qualified"],
             "authorization": "close_c053_with_typed_causal_candidate" if summary["qualified"]
             else "close_c053_at_causal_selectivity_boundary",
             "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "finalize"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run()
    else:
        finalize()


if __name__ == "__main__":
    main()
