#!/usr/bin/env python3
"""Export the important C244 independent replication as an all-coordinate browser atlas."""
from __future__ import annotations

import itertools
import json

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.RESULT / "phase1778_c244_independent_event_replication"
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c244_independent_event_replication.json"
CHECKPOINTS = (0, 8, 16, 24, 36)
ROLES = ("relation", "boundary")


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    if not final["all_checks_passed"]:
        raise RuntimeError("C244 not closed")
    fields = np.load(OUT / "raw/full_fields.float16.npy", mmap_mode="r")
    index = core.rows(OUT / "raw/hidden_index.jsonl")
    key = {(row["family"], row["surface"], row["unit"], row["factor_a"], row["factor_b"], row["order"]): row for row in index}
    means = {}
    for family in common.FAMILIES:
        effect_rows = []
        for surface, unit, order in itertools.product(("field_memo", "public_hearing"), range(6), (1, -1)):
            if (surface == "field_memo") != (unit < 3):
                continue
            cells = {}
            for a, b in itertools.product((0, 1), repeat=2):
                row = key[(family, surface, unit, a, b, order)]
                state = np.asarray(fields[row["hidden_index"]], np.float32)
                aligned = np.empty((37, 6, 2560), np.float32)
                for role_i, role in enumerate(common.ROLES):
                    aligned[:, role_i] = state[:, row["role_positions"][role], :].mean(axis=1)
                cells[(a, b)] = aligned
            effect_rows.append(common.factorial_effect(cells))
        means[family] = np.mean(effect_rows, axis=0)
    importance = sum(np.sum(np.abs(value[:, CHECKPOINTS][:, :, [common.ROLES.index(role) for role in ROLES]]), axis=(0, 1, 2)) for value in means.values())
    default_coordinates = np.argsort(-importance)[:64].astype(int).tolist()
    rule = np.load(common.OUTS["C237"] / "analysis/rule_codes.int8.npy", mmap_mode="r")
    rows = []
    for family_i, family in enumerate(common.FAMILIES):
        for effect_i, effect in enumerate(common.EFFECTS):
            for q in CHECKPOINTS:
                for role in ROLES:
                    role_i = common.ROLES.index(role)
                    rows.append({
                        "source": "C244_independent",
                        "family": family,
                        "effect": effect,
                        "checkpoint": q,
                        "checkpoint_type": "embedding" if q == 0 else "hidden_state",
                        "role": role,
                        "stable_discovery_event_count": int(np.count_nonzero(rule[family_i, effect_i, q, role_i])),
                        "label": f"{family}/{effect}/q{q}/{role}",
                        "values": np.round(means[family][effect_i, q, role_i], 6).tolist(),
                    })
    family_results = {row["family"]: row for row in final["headline"]["event_replication"]["family_results"]}
    payload = {
        "schema": "c244_independent_event_replication.v1",
        "phase": 1778,
        "campaign": "C244",
        "model": "Qwen3-4B",
        "dimensions": list(range(2560)),
        "default_coordinates": default_coordinates,
        "coordinate_semantics": "Every column is a physical Qwen3 activation coordinate; q0 is the embedding and q8/q16/q24/q36 are complete HiddenStates.",
        "claim_boundary": "The view shows independent new-material means at two registered roles. Replicated signed events are not minimal neurons, weights, or causal paths.",
        "summary": {
            "behavior_accuracy": final["headline"]["behavior"]["global_accuracy"],
            "families_passed": [family for family, row in family_results.items() if row["passed"]],
            "target_families_passed": final["headline"]["event_replication"]["target_families_passed"],
            "candidate_order_signed_agreement_median": final["headline"]["event_replication"]["candidate_order_signed_agreement_median"],
            "cross_model_diagnostics_passed": final["headline"]["cross_model_scaffold_diagnostic"]["transforms_passed"],
        },
        "rows": rows,
    }
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    with ASSET.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    checks = {"rows": len(rows) == 150, "coordinates": all(len(row["values"]) == 2560 for row in rows), "embedding_and_hidden": {row["checkpoint_type"] for row in rows} == {"embedding", "hidden_state"}, "asset": ASSET.is_file()}
    result = {"phase": 1778, "campaign": "C244", "checks": checks, "all_checks_passed": all(checks.values()), "asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"), "bytes": ASSET.stat().st_size, "sha256": core.sha(ASSET)}
    core.save(OUT / "analysis/heatmap_manifest.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
