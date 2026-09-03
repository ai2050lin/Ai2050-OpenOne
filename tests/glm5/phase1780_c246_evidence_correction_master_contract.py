#!/usr/bin/env python3
"""C246: append-only evidence corrections and freeze the C246-C255 campaign."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1780_c246_c255_event_hypergraph_common as common

core = common.core
OUT = common.OUTS["C246"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OLD["C245"] / "audit/independent_final_audit.json")
    c235 = np.load(common.OLD["C235"] / "raw/full_fields.float16.npy", mmap_mode="r")
    c244 = np.load(common.OLD["C244"] / "raw/full_fields.float16.npy", mmap_mode="r")
    corrections = {
        "C235": {"shape": list(c235.shape), "float16_values": int(c235.size), "bytes": int(c235.nbytes)},
        "C244": {"shape": list(c244.shape), "float16_values": int(c244.size), "bytes": int(c244.nbytes)},
        "measurement_boundary": "C235/C236 preserve all token slots and coordinates; C237-C245 average tokens inside six researcher-defined role spans while retaining all 2560 coordinates.",
        "factorial_scale": "R_A=2 beta_A, R_B=2 beta_B, R_AB=4 beta_AB; raw event counts are not directly comparable across effects.",
        "causal_boundary": "Cross-prompt factorial contrasts are observational contrasts, not do-intervention responses.",
        "coordinate_boundary": "A physical activation coordinate is neither a model weight nor an independently identified neuron.",
    }
    checks = {
        "authorization": parent["all_checks_passed"],
        "c235_shape": list(c235.shape) == [640, 37, 128, 2560],
        "c235_values": c235.size == 7_759_462_400,
        "c235_bytes": c235.nbytes == 15_518_924_800,
        "c244_shape": list(c244.shape) == [240, 37, 128, 2560],
        "c244_values": c244.size == 2_909_798_400,
        "c244_bytes": c244.nbytes == 5_819_596_800,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1780, "campaign": "C246", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "master_contract_frozen", "corrections": corrections,
        "research_object": "prospective third-material validation of conditional signed events, full-token observation, typed composition, conditional causal adjudication, and cross-model abstract replication",
        "stages": ["C247 material", "C248 Qwen full field", "C249 prospective event core", "C250 full-token observation", "C251 typed composition", "C252 conditional causal branch", "C253 cross-model abstract graph", "C254 heatmap", "C255 theory adjudication"],
        "evidence_policy": "route-local failure; all observation branches continue; causal branch runs only after its frozen eligibility condition",
        "forbidden": ["attention states", "MLP states", "weights", "PCA", "Top-K discovery", "post-reveal threshold edits", "calling observation causal"],
        "required_controls": ["wrong family", "sign flip", "coordinate shift", "equal-count same-sign randomization", "candidate order", "surface", "token length"],
        "human_blind_naturalness": "registered missingness; no human judgment will be fabricated",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C247_material_contract",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/evidence_corrections.json", corrections)
    audit = {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    core.save(OUT / "analysis/final.json", {"phase": 1780, "campaign": "C246", "status": "closed", "all_checks_passed": True, "headline": corrections, "next_authorization": protocol["authorization"]})
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
