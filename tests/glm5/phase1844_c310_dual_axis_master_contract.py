#!/usr/bin/env python3
"""C310: evidence correction and frozen C310-C335 master contract."""
from __future__ import annotations

import json
from pathlib import Path

import phase1844_c310_c335_dual_axis_common as common


def main() -> None:
    prior = common.core.load(common.old.OUTS["C309"] / "analysis/final.json")
    checks = {
        "prior_independent_audit": prior["all_checks_passed"],
        "campaigns_consecutive": [common.PHASES[f"C{i}"][0] for i in range(310, 336)] == list(range(1844, 1870)),
        "all_models_local_sequential": common.MODELS == ("qwen3", "glm4", "deepseek7b"),
        "all_2560_qwen_coordinates_primary": common.DIM == 2560,
        "no_attention_mlp_weight_readout": True,
    }
    protocol = {
        "status": "master_contract_frozen",
        "objective": "Test whether family-conditioned second-order response residuals support specific prediction, state-depth transport, distributed causal use, natural-surface transfer, cross-model response equivalence, and graph-depth transfer.",
        "evidence_corrections": [
            "C302 predicts a missing fourth factorial hidden state from the other three cells plus a family mean interaction residual; it does not yet identify a semantic operator.",
            "C302 uses six role means at 37 canonical checkpoints and all 2560 coordinates; it is not an all-token forecast.",
            "Six of six family-level mean gains do not imply that every one of the 192 sixth-material groups improves.",
            "C307 repeats anonymous task-conditioned role/depth topology; it is not functional isomorphism or cross-model coordinate identity.",
            "C306 rejects one broad patch interface only.",
        ],
        "corrected_sample_arithmetic": {
            "C315_base_configurations": "6 families x 2 surfaces x 8 units x 1 order = 96; six intervention conditions produce 576 forward evaluations",
            "C319_base_configurations": "6 families x 2 surfaces x 4 confirmation units = 48; 3 source widths x 4 coordinate widths x 2 polarities produce 1152 forward evaluations",
            "C321_cases": "6 families x 5 surfaces x 8 units x 4 factorial cells x 2 answer orders = 1920",
            "C326_panel": "6 families x 5 surfaces x 4 units x 4 factorial cells x 1 answer order = 480 per model",
            "C331_cases": "4 depths x 12 graph instances x 2 surfaces x 2 shortcut states x 2 answer orders = 384",
        },
        "branches": {
            "C310_C315": "specificity, atomic transport, dual-axis finite-difference square, full-coordinate atlas, coarse causal deletion",
            "C316_C320": "sign-amplitude coupling and multi-source/full-width causal dose map",
            "C321_C325": "five natural surfaces; human blind naturalness remains an explicit external no-test until real raters exist",
            "C326_C330": "three models loaded sequentially; compare model-native full coordinate axes only through anonymous role/depth response objects",
            "C331_C335": "controlled four-level graph plus renamed path controls, synthesis heatmap, and independent audit",
        },
        "failure_policy": "A failed strong gate retires only that claim. Registered observational, control, other-family, cross-model, and graph branches continue.",
        "forbidden": ["attention", "MLP", "weights", "PCA", "coordinate truncation as primary evidence", "fabricated human ratings", "post-reveal threshold edits"],
        "claim_boundary": "The campaign can discover conditional response regularities and qualified causal effects. It cannot by design establish a complete language code, unique circuit, or new mathematics.",
        "authorization": "C311_C335_all_registered_branches",
    }
    out = common.prepare("C310", protocol, checks)
    common.core.save(out / "protocol/campaign_map.json", {
        campaign: {"phase": phase, "slug": slug, "output": str(common.OUTS[campaign])}
        for campaign, (phase, slug) in common.PHASES.items()
    })
    headline = {
        "status": "audit_corrected_and_campaign_frozen",
        "corrected_overclaims": protocol["evidence_corrections"],
        "sample_arithmetic": protocol["corrected_sample_arithmetic"],
        "strict_interpretation": protocol["claim_boundary"],
    }
    common.close("C310", headline, {"map_has_26_campaigns": len(common.PHASES) == 26, "all_paths_unique": len(set(common.OUTS.values())) == 26}, "C311_interaction_residual_specificity")


if __name__ == "__main__":
    main()
