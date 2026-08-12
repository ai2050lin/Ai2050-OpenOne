#!/usr/bin/env python3
"""Post-hoc template robustness for Phase1092; never upgrades frozen evidence."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_protocol as protocol
import phase1092_natural_bilingual_attribute_finalize as analysis


def comparison(
    data,
    attribute: str,
    source_surface: str,
    target_surface: str,
    source_template: int,
    target_template: int,
    replicate: int,
) -> dict:
    content_source = analysis.bank(
        data, attribute, source_surface, "discovery", "content", replicate,
        template=source_template,
    )
    content_target = analysis.bank(
        data, attribute, target_surface, "confirmation", "content", replicate,
        template=target_template,
    )
    null_source = analysis.bank(
        data, attribute, source_surface, "discovery", "field_null", replicate,
        template=source_template,
    )
    null_target = analysis.bank(
        data, attribute, target_surface, "confirmation", "field_null", replicate,
        template=target_template,
    )
    labels = analysis.operation_names(attribute)
    content = analysis.exact_assignment(content_source, content_target, labels)
    null = analysis.exact_assignment(null_source, null_target, labels)
    identity_passed, identity_advantage = analysis.identity_pass(content, null)
    return {
        "replicate": replicate,
        "source_surface": source_surface,
        "target_surface": target_surface,
        "source_template": source_template,
        "target_template": target_template,
        "content_top1": content["top1_correct"],
        "field_null_top1": null["top1_correct"],
        "content_exact_p": content["exact_upper_tail_p"],
        "identity_advantage": identity_advantage,
        "identity_passed": identity_passed,
        "gram": analysis.gram_record(
            content_source, content_target, null_source, null_target
        ),
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    models = {name: analysis.load_model(name) for name in protocol.MODELS}
    by_model = {}
    for model_name, data in models.items():
        attributes = {}
        for attribute in protocol.ATTRIBUTES:
            cross_rows = []
            within_rows = []
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                for source_surface, target_surface in (("en", "zh"), ("zh", "en")):
                    for source_template in protocol.TEMPLATE_IDS:
                        for target_template in protocol.TEMPLATE_IDS:
                            cross_rows.append(comparison(
                                data, attribute, source_surface, target_surface,
                                source_template, target_template, replicate,
                            ))
                for surface in protocol.SURFACES:
                    for source_template, target_template in ((0, 1), (1, 0)):
                        within_rows.append(comparison(
                            data, attribute, surface, surface,
                            source_template, target_template, replicate,
                        ))
            behavior_passed = attribute in behavior["models"][model_name][
                "passing_attributes"
            ]
            attributes[attribute] = {
                "behavior_passed": behavior_passed,
                "cross_language": {
                    "identity_passing_count": sum(
                        int(row["identity_passed"]) for row in cross_rows
                    ),
                    "gram_passing_count": sum(
                        int(row["gram"]["passed"]) for row in cross_rows
                    ),
                    "comparison_count": len(cross_rows),
                    "all_identity_passed": all(
                        row["identity_passed"] for row in cross_rows
                    ),
                    "all_gram_passed": all(
                        row["gram"]["passed"] for row in cross_rows
                    ),
                    "rows": cross_rows,
                },
                "within_language_cross_template": {
                    "identity_passing_count": sum(
                        int(row["identity_passed"]) for row in within_rows
                    ),
                    "gram_passing_count": sum(
                        int(row["gram"]["passed"]) for row in within_rows
                    ),
                    "comparison_count": len(within_rows),
                    "all_identity_passed": all(
                        row["identity_passed"] for row in within_rows
                    ),
                    "all_gram_passed": all(
                        row["gram"]["passed"] for row in within_rows
                    ),
                    "rows": within_rows,
                },
            }
        by_model[model_name] = {"attributes": attributes}

    result = {
        "schema_version": "phase1092_posthoc_template_robustness.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "by_model": by_model,
        "evidence_upgrade_allowed": False,
        "interpretation": (
            "Template-specific comparisons diagnose whether an aggregate candidate "
            "survives all surface pairings. They were run after the frozen analysis "
            "and cannot change P1-P9."
        ),
    }
    result["posthoc_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "posthoc_template_robustness.json",
        result,
    )
    compact = {}
    for model_name, model in by_model.items():
        compact[model_name] = {
            attribute: {
                "behavior": row["behavior_passed"],
                "cross_id": (
                    row["cross_language"]["identity_passing_count"],
                    row["cross_language"]["comparison_count"],
                ),
                "cross_gram": (
                    row["cross_language"]["gram_passing_count"],
                    row["cross_language"]["comparison_count"],
                ),
                "within_id": (
                    row["within_language_cross_template"]["identity_passing_count"],
                    row["within_language_cross_template"]["comparison_count"],
                ),
            }
            for attribute, row in model["attributes"].items()
        }
    print({
        "phase": protocol.PHASE,
        "compact": compact,
        "posthoc_digest": result["posthoc_digest"],
    })


if __name__ == "__main__":
    main()
