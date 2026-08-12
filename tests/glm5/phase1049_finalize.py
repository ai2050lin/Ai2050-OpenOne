#!/usr/bin/env python3
"""Aggregate Phase1049 held-out Q/K/V causal confirmation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import phase1049_qkv_read_path_protocol as protocol


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        for model in protocol.MODELS
    }
    for model, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model} protocol digest mismatch")
        if (
            not summary["baseline_logits_finite"]["all_finite"]
            or not summary["condition_logits_finite"]["all_finite"]
        ):
            raise RuntimeError(f"{model} nonfinite logits")

    minimum = int(
        prereg["causal_route_gate"]["minimum_models"]
    )
    union_models = [
        model
        for model, summary in summaries.items()
        if summary["analysis"]["selected_kv_union_route_passed"]
    ]
    all_models = [
        model
        for model, summary in summaries.items()
        if summary["analysis"][
            "selected_kv_all_postsource_route_passed"
        ]
    ]
    if len(union_models) >= minimum:
        route = "head_group_and_natural_rollout"
        rationale = (
            "The frozen natural read-band K/V union mediated the early "
            "fact-state effect with selected-fact specificity in at least "
            "two models."
        )
    elif len(all_models) >= minimum:
        route = "cumulative_depth_boundary"
        rationale = (
            "Only the broad post-source K/V path passed in at least two "
            "models; localize the cumulative depth boundary before any "
            "head-level claim."
        )
    else:
        route = "stop_attention_causal_block"
        rationale = (
            "Neither the frozen read bands nor the broad post-source K/V "
            "path passed in two models. Preserve Phase1048 as descriptive "
            "natural-read evidence and stop this intervention family."
        )

    artifacts = []
    for path in sorted(
        item for item in protocol.OUT_ROOT.rglob("*") if item.is_file()
    ):
        artifacts.append({
            "path": str(path.relative_to(protocol.OUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    aggregate: dict[str, Any] = {
        "schema_version": "phase1049_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "model_results": {
            model: {
                "clean_behavior": summary["clean_behavior"],
                "source_shift": summary["analysis"]["source_shift"],
                "condition_metrics": summary["analysis"][
                    "condition_metrics"
                ],
                "union_specificity": summary["analysis"][
                    "union_specificity"
                ],
                "all_postsource_specificity": summary["analysis"][
                    "all_postsource_specificity"
                ],
                "channel_interactions": summary["analysis"][
                    "channel_interactions"
                ],
                "selected_kv_union_route_passed": summary[
                    "analysis"
                ]["selected_kv_union_route_passed"],
                "selected_kv_all_postsource_route_passed": summary[
                    "analysis"
                ]["selected_kv_all_postsource_route_passed"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
            for model, summary in summaries.items()
        },
        "cross_model_gate": {
            "minimum_models": minimum,
            "frozen_union_passing_models": union_models,
            "all_postsource_passing_models": all_models,
            "frozen_union_passed": len(union_models) >= minimum,
            "all_postsource_passed": len(all_models) >= minimum,
        },
        "automatic_next_decision": {
            "route": route,
            "rationale": rationale,
            "should_continue_automatically": route != (
                "stop_attention_causal_block"
            ),
        },
        "interpretation_limits": [
            "A positive result identifies a projected causal transport "
            "path for an artificial early fact-state edit in this task.",
            "A negative result does not erase the natural Phase1048 "
            "attention atlas or prove that Attention is unused.",
            "The query-side Q reset measures propagation of the early "
            "source edit into Q; it is not a direct intervention on the "
            "natural query choice.",
            "No result identifies a single head, neuron, universal "
            "language mechanism, biological optimum, or new mathematics.",
        ],
        "artifact_manifest": {
            "file_count": len(artifacts),
            "total_bytes": sum(row["bytes"] for row in artifacts),
            "files": artifacts,
        },
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    print({
        "phase": protocol.PHASE,
        "union_models": union_models,
        "all_models": all_models,
        "next_route": route,
    })


if __name__ == "__main__":
    main()
