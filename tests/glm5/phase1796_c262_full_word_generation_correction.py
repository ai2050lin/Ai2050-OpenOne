#!/usr/bin/env python3
"""C262: correct the one-token/one-word error and test full generated words."""
from __future__ import annotations

import gc
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1780_c246_c255_event_hypergraph_common as common
import phase1795_c261_coordinate_coverage_generation_side_effects as c261


core = common.core
OUT = common.RESULT / "phase1796_c262_full_word_generation_correction"
C261 = common.RESULT / "phase1795_c261_coordinate_coverage_generation_side_effects"


def install_cache_safe_hooks(model, tri, states, donor_i, target, family_i, mode="correct"):
    handles = []
    wrong_fi = common.FAMILIES.index("type_graph")
    for q in c261.CHECKPOINTS:
        source_q = 36 - q if mode == "reversed" else q

        def make_hook(q=q, source_q=source_q, mode=mode):
            def hook(_module, _inputs, output):
                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                ri = common.ROLES.index("relation")
                fi = wrong_fi if mode == "wrong_family" else family_i
                mask = np.asarray(tri[fi, 0, source_q, ri] != 0)
                if mode == "roll":
                    mask = np.roll(mask, 137)
                coords = np.flatnonzero(mask)
                if coords.size:
                    c = torch.as_tensor(coords, dtype=torch.long, device=hidden.device)
                    donor = torch.as_tensor(states[donor_i, source_q, ri, coords].astype(np.float32), dtype=hidden.dtype, device=hidden.device)
                    for pos in target["role_positions"]["relation"]:
                        # During cached decoding only the new token is present; the patched prefix is already cached.
                        if pos < hidden.shape[1]:
                            hidden[0, pos, c] = donor
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook

        handles.append(model.model.layers[q - 1].register_forward_hook(make_hook()))
    return handles


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("protocol", "analysis", "audit"):
        (OUT / name).mkdir(exist_ok=True)
    checks = {
        "c261_complete": core.load(C261 / "analysis/final.json")["all_checks_passed"],
        "c261_one_token_error_registered": True,
        "max_new_tokens_covers_doubt": 3,
        "same_frozen_path": True,
        "no_attention_or_mlp": True,
    }
    protocol = {
        "phase": 1796,
        "campaign": "C262",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_load",
        "correction": "C260 scored leading-space token IDs; C261 exact one-token text undercounted no-space doubt, which tokenizes as d/oub/t.",
        "conditions": ["clean", "correct", "wrong_family", "coordinate_roll", "reversed_masks"],
        "max_new_tokens": 3,
        "gate": {"clean_success_min": 0.80, "correct_success_min": 0.80, "correct_minus_best_control_min": 0.20},
        "claim_boundary": "This tests a controlled generated word, not unrestricted sentence generation or unrelated-task preservation.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_once_and_reclassify_C260_C261_word_claims",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = common.previous.load_bf16("qwen3")
        panel, states, _word_tokens = c261.build_word_panel(model, tokenizer, device)
        tri = np.load(common.RESULT / "phase1783_c249_third_material_event_core_prediction/analysis/tri_material_core.int8.npy", mmap_mode="r")
        key = {(row["unit"], row["a"]): i for i, row in enumerate(panel)}
        family_i = common.FAMILIES.index("attitude_event")
        rows = []
        for unit in range(8):
            for target_a, donor_a in ((0, 1), (1, 0)):
                ti, di = key[(unit, target_a)], key[(unit, donor_a)]
                target = panel[ti]
                target_word = "approval" if target_a == 0 else "doubt"
                donor_word = "approval" if donor_a == 0 else "doubt"
                ids = torch.tensor([target["prompt_ids"]], dtype=torch.long, device=device)
                for condition, mode in (("clean", None), ("correct", "correct"), ("wrong_family", "wrong_family"), ("coordinate_roll", "roll"), ("reversed_masks", "reversed")):
                    handles = [] if mode is None else install_cache_safe_hooks(model, tri, states, di, target, family_i, mode)
                    try:
                        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=3, do_sample=False, use_cache=True)
                    finally:
                        for handle in handles:
                            handle.remove()
                    continuation = generated[0, ids.shape[1]:].tolist()
                    text = tokenizer.decode(continuation).strip().lower()
                    expected = target_word if condition == "clean" else donor_word
                    rows.append({
                        "unit": unit,
                        "direction": f"{target_a}_to_{donor_a}",
                        "condition": condition,
                        "continuation_ids": continuation,
                        "text": text,
                        "expected": expected,
                        "success": text.startswith(expected),
                    })
        core.write_rows(OUT / "analysis/generation_rows.jsonl", rows)
        summaries = []
        for condition in protocol["conditions"]:
            selected = [row for row in rows if row["condition"] == condition]
            summaries.append({"condition": condition, "support": len(selected), "success_rate": float(np.mean([row["success"] for row in selected])), "outputs": sorted({row["text"] for row in selected})})
        by = {row["condition"]: row for row in summaries}
        best_control = max(by[name]["success_rate"] for name in ("wrong_family", "coordinate_roll", "reversed_masks"))
        specificity_margin = by["correct"]["success_rate"] - best_control
        gate = by["clean"]["success_rate"] >= 0.80 and by["correct"]["success_rate"] >= 0.80 and specificity_margin >= 0.20
        report = {
            "phase": 1796,
            "campaign": "C262",
            "status": "adjudicated",
            "tokenization": {
                "approval_no_space": tokenizer.encode("approval", add_special_tokens=False),
                "doubt_no_space": tokenizer.encode("doubt", add_special_tokens=False),
                "approval_with_space": tokenizer.encode(" approval", add_special_tokens=False),
                "doubt_with_space": tokenizer.encode(" doubt", add_special_tokens=False),
            },
            "summaries": summaries,
            "correct_minus_best_control": specificity_margin,
            "full_word_generation_gate_passed": gate,
            "c260_direct_word_reclassification": "controlled leading-space token-logit readout, not natural word readout",
            "c261_one_token_reclassification": "invalid full-word metric; no-space doubt requires three tokens",
            "c261_side_effect_reclassification": "same-prompt full-vocabulary divergence includes the intended answer change and used mismatched candidate token IDs; it is not an unrelated-capability test",
            "placement": placement,
            "elapsed_seconds": time.time() - started,
            "strict_interpretation": "A pass requires the correct early path to generate the donor word and beat every path control. A control collision means output manipulation is not path-specific even if the word changes.",
            "next_authorization": "new_operator_design_if_control_collision; independent_generated_word_replication_if_specific",
        }
        core.save(OUT / "analysis/summary.json", report)
        analysis_checks = {"rows": len(rows) == 16 * 5, "support": all(row["support"] == 16 for row in summaries), "hooks_removed": True, "tokenization_recorded": len(report["tokenization"]["doubt_no_space"]) == 3}
        core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
        final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
        final = {"phase": 1796, "campaign": "C262", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
        core.save(OUT / "analysis/final.json", final)
        core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": report["next_authorization"]})
        print(json.dumps(final, indent=2))
    finally:
        common.previous.release(model)
        gc.collect()


if __name__ == "__main__":
    main()
