#!/usr/bin/env python3
"""C187: complete the vocabulary x paraphrase response cells missing from C186."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1721_c187_vocabulary_paraphrase_failure_decomposition"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
C186 = RESULT / "phase1720_c186_new_material_response_ecology_prediction"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1720_c186_new_material_response_ecology_prediction as c186

PHASE, CAMPAIGN = 1721, "C187"
DIM, WIDTH = 2560, 224
ROLES = c186.ROLES


def selected_cross_rows(compiled, behavior):
    selected, missing = [], []
    for family in c186.RELATIONS:
        for unit, phrase_variant in ((0, 1), (3, 0)):
            matches = [row for row in behavior if row["family"] == family and row["unit"] == unit and row["phrase_variant"] == phrase_variant and row["order"] == 1]
            if len(matches) != 1:
                raise RuntimeError((family, unit, phrase_variant, matches))
            row = matches[0]
            cell = {"family": family, "unit": unit, "phrase_variant": phrase_variant, "row_index": row["row_index"], "case_id": row["case_id"], "correct": row["correct"]}
            (selected if row["correct"] else missing).append(cell)
    return selected, missing


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C186 / "audit/independent_final_audit.json")
    compiled = core.rows(C186 / "compiled/qwen3.jsonl")
    behavior = core.rows(C186 / "raw/behavior_index.jsonl")
    selected, missing = selected_cross_rows(compiled, behavior)
    checks = {
        "authorization": parent["all_checks_passed"] and "C187" in parent["authorization"],
        "selected": len(selected) == 13,
        "registered_missing": len(missing) == 1 and missing[0]["family"] == "reports_to" and missing[0]["unit"] == 3 and missing[0]["phrase_variant"] == 0,
        "all_selected_correct": all(row["correct"] for row in selected),
        "existing_cells": list(np.load(C186 / "raw/new_relation_role_response.float16.npy", mmap_mode="r").shape) == [14, 64, 6, 2560],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/selected_cross_cells.jsonl", selected)
    core.write_rows(OUT / "material/registered_missing_cells.jsonl", missing)
    core.write_rows(OUT / "compiled/qwen3_selected.jsonl", [compiled[row["row_index"]] for row in selected])
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "vocabulary_paraphrase_factorial_completion_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "existing_cells": "unit0 canonical and unit3 paraphrase from C186",
        "new_cells": "unit0 paraphrase and unit3 canonical; reports_to/unit3/canonical registered missing by behavior",
        "response": "relation q24 64 frozen source coordinates to q25 six-role x 2560 field",
        "analysis": "same-family target energy advantage against C180 discovery profiles for each 2x2 cell",
        "missing_policy": "no imputation; report cell support denominator",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cosine", "global route stop"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_13_cross_cell_responses",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "missing": missing}, indent=2))


@torch.inference_mode()
def run():
    rows = core.rows(OUT / "compiled/qwen3_selected.jsonl")
    selected = core.rows(OUT / "material/selected_cross_cells.jsonl")
    coordinates = core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:64]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(OUT / "raw/cross_cell_relation_response.float16.npy", mode="w+", dtype=np.float16, shape=(len(rows), 64, 6, DIM))
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        base = model.model
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def perturb(row, selected_coordinates, sign, epsilon):
            batch = [row] * len(selected_coordinates)
            ids, mask, positions, _lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            captured = {}
            def patch(_module, _args, value):
                state = c186.tensor(value); patched = state.clone()
                for local, coordinate in enumerate(selected_coordinates):
                    for position in row["role_positions"]["relation"]:
                        patched[local, position, int(coordinate)] += sign * epsilon
                return (patched,) + value[1:] if isinstance(value, tuple) else patched
            h1 = base.layers[23].register_forward_hook(patch)
            h2 = base.layers[24].register_forward_hook(lambda _m, _a, value: captured.__setitem__("state", c186.tensor(value).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            finally:
                h1.remove(); h2.remove()
            field = np.zeros((len(selected_coordinates), 6, DIM), np.float32)
            for local in range(len(selected_coordinates)):
                for role_i, role in enumerate(ROLES):
                    field[local, role_i] = captured["state"][local, row["role_positions"][role]].mean(0).float().cpu().numpy()
            return field

        for anchor_i, row in enumerate(rows):
            ids, mask, positions, _lengths = fixed_base.fixed_batch([row], pad, device, WIDTH)
            captured = {}
            hook = base.layers[23].register_forward_hook(lambda _m, _a, value: captured.__setitem__("state", c186.tensor(value).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            finally:
                hook.remove()
            source = captured["state"][0, row["role_positions"]["relation"]].mean(0).float().cpu().numpy()
            epsilon = 0.5 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))
            for start in range(0, 64, 16):
                cs = coordinates[start:start + 16]
                plus = perturb(row, cs, 1.0, epsilon)
                minus = perturb(row, cs, -1.0, epsilon)
                raw[anchor_i, start:start + len(cs)] = ((plus - minus) / (2 * epsilon)).astype(np.float16)
            raw.flush()
            print(f"[C187] {anchor_i + 1}/{len(rows)} {selected[anchor_i]['family']} unit{selected[anchor_i]['unit']} phrase{selected[anchor_i]['phrase_variant']}", flush=True)
        core.write_rows(OUT / "raw/response_index.jsonl", [{**cell, "anchor_index": i} for i, cell in enumerate(selected)])
        checks = {"shape": list(raw.shape) == [13, 64, 6, 2560], "finite": bool(np.isfinite(raw).all()), "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/run.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        raw.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def profile(values):
    energy = np.square(values, dtype=np.float64).sum(axis=(0, 1))
    return energy / max(energy.sum(), 1e-30)


def similarity(left, right):
    return float(1.0 - 0.5 * np.abs(left - right).sum())


def analyze():
    predictors = np.load(C186 / "protocol/c180_discovery_relation_target_profiles.float32.npy").astype(np.float64)
    families = core.load(C186 / "protocol/preregistration.json")["families"]
    predictor = {family: predictors[i] for i, family in enumerate(families)}
    existing_raw = np.load(C186 / "raw/new_relation_role_response.float16.npy", mmap_mode="r")
    existing_index = core.rows(C186 / "raw/response_anchor_index.jsonl")
    cross_raw = np.load(OUT / "raw/cross_cell_relation_response.float16.npy", mmap_mode="r")
    cross_index = core.rows(OUT / "raw/response_index.jsonl")
    missing = core.rows(OUT / "material/registered_missing_cells.jsonl")
    cells = {}
    for row in existing_index:
        unit = 0 if row["partition"] == "new_confirmation" else 3
        cells[(row["family"], unit, row["phrase_variant"])] = np.asarray(existing_raw[row["anchor_index"]], dtype=np.float32)
    for row in cross_index:
        cells[(row["family"], row["unit"], row["phrase_variant"])] = np.asarray(cross_raw[row["anchor_index"]], dtype=np.float32)
    rows = []
    for family in families:
        for unit in (0, 3):
            for phrase_variant in (0, 1):
                key = (family, unit, phrase_variant)
                if key not in cells:
                    rows.append({"family": family, "unit": unit, "phrase_variant": phrase_variant, "observed": False, "reason": "behavior_ineligible"})
                    continue
                actual = profile(cells[key])
                same = similarity(predictor[family], actual)
                wrong = [similarity(predictor[other], actual) for other in families if other != family]
                rows.append({"family": family, "unit": unit, "phrase_variant": phrase_variant, "observed": True, "same_similarity": same, "median_wrong_advantage": same - float(np.median(wrong)), "hard_wrong_advantage": same - float(np.max(wrong))})
    cell_summary = {}
    for unit in (0, 3):
        for phrase_variant in (0, 1):
            selected = [row for row in rows if row["unit"] == unit and row["phrase_variant"] == phrase_variant and row["observed"]]
            key = f"unit{unit}_phrase{phrase_variant}"
            cell_summary[key] = {"observed_families": len(selected), "median_same_similarity": float(np.median([row["same_similarity"] for row in selected])), "median_wrong_advantage": float(np.median([row["median_wrong_advantage"] for row in selected])), "median_hard_wrong_advantage": float(np.median([row["hard_wrong_advantage"] for row in selected])), "positive_family_count": int(sum(row["median_wrong_advantage"] > 0 for row in selected)), "positive_family_fraction": float(np.mean([row["median_wrong_advantage"] > 0 for row in selected]))}
    phrase_effects, vocabulary_effects = [], []
    for family in families:
        for unit in (0, 3):
            canonical = next(row for row in rows if row["family"] == family and row["unit"] == unit and row["phrase_variant"] == 0)
            paraphrase = next(row for row in rows if row["family"] == family and row["unit"] == unit and row["phrase_variant"] == 1)
            if canonical["observed"] and paraphrase["observed"]:
                phrase_effects.append({"family": family, "unit": unit, "advantage_change_paraphrase_minus_canonical": paraphrase["median_wrong_advantage"] - canonical["median_wrong_advantage"]})
        for phrase_variant in (0, 1):
            first = next(row for row in rows if row["family"] == family and row["unit"] == 0 and row["phrase_variant"] == phrase_variant)
            second = next(row for row in rows if row["family"] == family and row["unit"] == 3 and row["phrase_variant"] == phrase_variant)
            if first["observed"] and second["observed"]:
                vocabulary_effects.append({"family": family, "phrase_variant": phrase_variant, "advantage_change_unit3_minus_unit0": second["median_wrong_advantage"] - first["median_wrong_advantage"]})
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "vocabulary_paraphrase_failure_decomposed", "cell_summary": cell_summary, "median_paraphrase_effect": float(np.median([row["advantage_change_paraphrase_minus_canonical"] for row in phrase_effects])), "median_vocabulary_effect": float(np.median([row["advantage_change_unit3_minus_unit0"] for row in vocabulary_effects])), "phrase_effects": phrase_effects, "vocabulary_effects": vocabulary_effects, "rows": rows, "registered_missing": missing, "next_authorization": "run_C188_generic_scaffold_prediction_on_new_material_then_freeze_campaign_synthesis"}
    core.save(OUT / "analysis/factorial_atlas.json", report)
    checks = {"cells": len(rows) == 28, "observed": sum(row["observed"] for row in rows) == 27, "missing": len(missing) == 1, "finite": bool(np.isfinite([report["median_paraphrase_effect"], report["median_vocabulary_effect"]]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"cell_summary": cell_summary, "median_paraphrase_effect": report["median_paraphrase_effect"], "median_vocabulary_effect": report["median_vocabulary_effect"], "checks": checks}, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/factorial_atlas.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"cell_summary": report["cell_summary"], "median_paraphrase_effect": report["median_paraphrase_effect"], "median_vocabulary_effect": report["median_vocabulary_effect"], "registered_missing": report["registered_missing"]}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "run", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "run": run, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__": main()
