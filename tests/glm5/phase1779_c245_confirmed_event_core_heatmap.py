#!/usr/bin/env python3
"""Export the C245 confirmed signed event core at physical-coordinate resolution."""
from __future__ import annotations

import json

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.RESULT / "phase1779_c245_confirmed_event_core"
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c245_confirmed_event_core.json"
CHECKPOINTS = (0, 8, 16, 24, 36)
ROLES = ("relation", "boundary")


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    rules = np.load(OUT / "analysis/confirmed_rule_codes.int8.npy", mmap_mode="r")
    selected = rules[:, :, CHECKPOINTS][:, :, :, [common.ROLES.index(role) for role in ROLES]]
    importance = np.count_nonzero(selected, axis=(0, 1, 2, 3))
    default_coordinates = np.argsort(-importance)[:64].astype(int).tolist()
    rows = []
    for family_i, family in enumerate(common.FAMILIES):
        for effect_i, effect in enumerate(common.EFFECTS):
            for q in CHECKPOINTS:
                for role in ROLES:
                    role_i = common.ROLES.index(role)
                    values = np.asarray(rules[family_i, effect_i, q, role_i], np.int8)
                    rows.append({"source": "C245_confirmed_core", "family": family, "effect": effect, "checkpoint": q, "checkpoint_type": "embedding" if q == 0 else "hidden_state", "role": role, "confirmed_event_count": int(np.count_nonzero(values)), "label": f"{family}/{effect}/q{q}/{role}", "values": values.astype(int).tolist()})
    payload = {
        "schema": "c245_confirmed_event_core.v1", "phase": 1779, "campaign": "C245", "model": "Qwen3-4B",
        "dimensions": list(range(2560)), "default_coordinates": default_coordinates,
        "coordinate_semantics": "Every cell is the confirmed signed event code for one physical Qwen3 activation coordinate: -1 down, 0 absent, +1 up.",
        "claim_boundary": "The core is the same-sign intersection of C237 discovery stability and C244 independent controlled material. It is not a weight, semantic neuron, minimal path, or causal mechanism.",
        "summary": {"confirmed_events": final["headline"]["confirmed_events"], "old_rule_events": final["headline"]["old_rule_events"], "overall_retention": final["headline"]["overall_retention"], "embedding_events": final["headline"]["embedding_events"], "hidden_events": final["headline"]["hidden_events"], "cross_family_signed_jaccard_median": final["headline"]["cross_family_signed_jaccard_median"], "same_checkpoint_persistence_median": final["headline"]["same_checkpoint_persistence_median"]},
        "rows": rows,
    }
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    with ASSET.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
    checks = {"rows": len(rows) == 150, "dimensions": all(len(row["values"]) == 2560 for row in rows), "alphabet": set(value for row in rows for value in row["values"]) <= {-1, 0, 1}}
    result = {"phase": 1779, "campaign": "C245", "checks": checks, "all_checks_passed": all(checks.values()), "asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"), "bytes": ASSET.stat().st_size, "sha256": core.sha(ASSET)}
    core.save(OUT / "analysis/heatmap_manifest.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
