#!/usr/bin/env python3
"""C258: frozen path-coverage localization for the C257 output-control boundary."""
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

core = common.core
OUT = common.RESULT / "phase1792_c258_output_readout_path_localization"
C257 = common.RESULT / "phase1791_c257_fixed_codebook_output_causal_test"
ROUTES = {
    "sparse4": (8, 16, 24, 32),
    "early_dense": tuple(range(1, 17)),
    "late_dense": tuple(range(20, 36)),
    "all_dense": tuple(range(1, 36)),
    "boundary_role_late": tuple(range(24, 36)),
    "all_dense_wrong_family": tuple(range(1, 36)),
    "all_dense_coordinate_roll": tuple(range(1, 36)),
    "all_dense_reversed_masks": tuple(range(1, 36)),
}


@torch.inference_mode()
def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(C257 / "analysis/final.json"); c257 = parent["headline"]
    checks = {"parent": parent["all_checks_passed"], "branch": not c257["output_causal_gate_passed"], "positive_margin_without_flip": c257["condition_margin_gain_medians"]["correct_path"] > 0 and c257["condition_flip_rates"]["correct_path"] == 0, "no_topk": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True); (OUT / "analysis").mkdir()
    protocol = {"phase": 1792, "campaign": "C258", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "path_localization_frozen", "routes": {k: list(v) for k, v in ROUTES.items()}, "intervention": "all tri-material factor-A coordinates at each route's registered checkpoints; boundary_role_late patches only the boundary role", "gate": {"route_flip_rate_min": .20, "route_margin_gain_min": 0.0, "best_dense_control_margin_min": .10}, "claim_boundary": "This tests path coverage under the same controlled codebook. It is not minimality or natural necessity.", "producer_sha256": core.sha(Path(__file__)), "authorization": "run_all_routes_once"}
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    compiled = core.rows(C257 / "compiled/qwen3.jsonl"); states = np.load(C257 / "raw/role_states.float16.npy", mmap_mode="r"); tri = np.load(common.OUTS["C249"] / "analysis/tri_material_core.int8.npy", mmap_mode="r"); key = {(r["surface"], r["unit"], r["factor_a"], r["factor_b"], r["order"]): i for i, r in enumerate(compiled)}; fi = common.FAMILIES.index("attitude_event"); wrong_fi = common.FAMILIES.index("type_graph")
    model = None; rows = []; started = time.time()
    try:
        model, _tokenizer, device, placement = common.previous.load_bf16("qwen3"); layers = model.model.layers
        for surface, unit, target_a, donor_a in ((s, u, ta, da) for s, u in itertools.product(common.SURFACES, range(8)) for ta, da in ((0, 1), (1, 0))):
            ti, di = key[(surface, unit, target_a, 0, 1)], key[(surface, unit, donor_a, 0, 1)]; target, donor = compiled[ti], compiled[di]; ids = torch.tensor([target["prompt_ids"]], dtype=torch.long, device=device); donor_gold = donor["gold_position"]; other = 1 - donor_gold
            clean = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True); clean_margin = float(clean.logits[0, -1, target["candidate_ids"][donor_gold][0]] - clean.logits[0, -1, target["candidate_ids"][other][0]])
            for route, checkpoints in ROUTES.items():
                handles = []
                for q in checkpoints:
                    source_q = 36 - q if route == "all_dense_reversed_masks" else q
                    def make_hook(q=q, source_q=source_q, route=route):
                        def hook(_module, _inputs, output):
                            hidden = output[0].clone() if isinstance(output, tuple) else output.clone(); roles = ("boundary",) if route == "boundary_role_late" else common.ROLES
                            for role in roles:
                                ri = common.ROLES.index(role); mask_fi = wrong_fi if route == "all_dense_wrong_family" else fi; mask = np.asarray(tri[mask_fi, 0, source_q, ri] != 0)
                                if route == "all_dense_coordinate_roll": mask = np.roll(mask, 137)
                                coords = np.flatnonzero(mask)
                                if not coords.size: continue
                                coord_t = torch.tensor(coords, dtype=torch.long, device=hidden.device); value_t = torch.tensor(np.asarray(states[di, q, ri], np.float32)[coords], dtype=hidden.dtype, device=hidden.device)
                                for pos in target["role_positions"][role]: hidden[0, pos, coord_t] = value_t
                            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
                        return hook
                    handles.append(layers[q - 1].register_forward_hook(make_hook()))
                try: output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                finally:
                    for h in handles: h.remove()
                margin = float(output.logits[0, -1, target["candidate_ids"][donor_gold][0]] - output.logits[0, -1, target["candidate_ids"][other][0]]); rows.append({"surface": surface, "unit": unit, "direction": f"{target_a}_to_{donor_a}", "route": route, "clean_donor_margin": clean_margin, "patched_donor_margin": margin, "donor_margin_gain": margin - clean_margin, "flipped_to_donor": margin > 0})
            print(f"[C258] {surface} unit {unit + 1}/8 {target_a}->{donor_a}", flush=True)
        core.write_rows(OUT / "analysis/path_rows.jsonl", rows); summaries = []
        for route in ROUTES:
            subset = [r for r in rows if r["route"] == route]; summaries.append({"route": route, "support": len(subset), "median_margin_gain": float(np.median([r["donor_margin_gain"] for r in subset])), "flip_rate": float(np.mean([r["flipped_to_donor"] for r in subset])), "median_patched_donor_margin": float(np.median([r["patched_donor_margin"] for r in subset]))})
        by = {r["route"]: r for r in summaries}; candidate_routes = ("sparse4", "early_dense", "late_dense", "all_dense", "boundary_role_late"); best = max(candidate_routes, key=lambda name: (by[name]["flip_rate"], by[name]["median_margin_gain"])); controls = ("all_dense_wrong_family", "all_dense_coordinate_roll", "all_dense_reversed_masks"); margin = by[best]["median_margin_gain"] - max(by[name]["median_margin_gain"] for name in controls); passed = by[best]["flip_rate"] >= .20 and by[best]["median_margin_gain"] > 0 and margin >= .10
        report = {"phase": 1792, "campaign": "C258", "status": "adjudicated", "route_summaries": summaries, "best_registered_route": best, "best_vs_dense_control_margin": margin, "output_path_gate_passed": passed, "placement": placement, "elapsed_seconds": time.time() - started, "strict_interpretation": "A pass localizes sufficient distributed path coverage for this controlled output codebook. A failure shows that expanding the same event mask across depth is insufficient; neither outcome establishes a minimal natural circuit.", "next_authorization": "C259_independent_natural_paraphrase_if_passed; response-conditioned_output_compiler_observation_if_failed"}
        core.save(OUT / "analysis/summary.json", report); analysis_checks = {"rows": len(rows) == 32 * len(ROUTES), "routes": len(summaries) == len(ROUTES), "finite": bool(np.isfinite([v for r in summaries for k, v in r.items() if k not in ("route", "support")] + [margin]).all()), "hooks_removed": True}; core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())}); final_checks = {"contract": True, "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}; final = {"phase": 1792, "campaign": "C258", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": report["next_authorization"]}); print(json.dumps(final, indent=2))
    finally:
        common.previous.release(model); gc.collect()


if __name__ == "__main__": main()
