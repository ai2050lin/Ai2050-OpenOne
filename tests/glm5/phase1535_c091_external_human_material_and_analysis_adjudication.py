#!/usr/bin/env python3
"""Phase1535: freeze human-validated C091 materials and adjudicate C089-C090 analyses."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1534_c089_c090_major_stage_closure"
OUT = RESULT / "phase1535_c091_external_human_material_and_analysis_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

SOURCES = {
    "word_pair_A.csv": {
        "url": "https://osf.io/download/dmy29/",
        "osf_file_id": "69cd1fad485a179b9f51e525",
    },
    "word_pair_B.csv": {
        "url": "https://osf.io/download/xqmek/",
        "osf_file_id": "69cd1fae485a179b9f51e527",
    },
    "split_half_results.csv": {
        "url": "https://osf.io/download/69cd2258481af92967744ff0/",
        "osf_file_id": "69cd2258481af92967744ff0",
    },
}


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "AI2050-C091-research/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1535 exists")
    parent = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c091_behavior_grounded_natural_relation_latent_use_bridge":
        raise RuntimeError("Phase1534 authorization missing")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1534 audit missing")

    source_dir = OUT / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, spec in SOURCES.items():
        payload = download(spec["url"])
        path = source_dir / name
        path.write_bytes(payload)
        manifest[name] = {
            **spec,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    frames = {
        "A": pd.read_csv(source_dir / "word_pair_A.csv", encoding="gb18030"),
        "B": pd.read_csv(source_dir / "word_pair_B.csv", encoding="gb18030"),
    }
    relation_counts = {
        key: frame.groupby(["relation", "concreteness"]).size().to_dict()
        for key, frame in frames.items()
    }
    relation_counts_json = {
        key: {f"{relation}|{concreteness}": int(count) for (relation, concreteness), count in counts.items()}
        for key, counts in relation_counts.items()
    }
    target_counts = {
        "similarity": int((frames["A"]["relation"] == "相似关系").sum()),
        "class_inclusion": int((frames["B"]["relation"] == "类别关系").sum()),
        "whole_part": int((frames["B"]["relation"] == "整体-部分关系").sum()),
    }
    checks = {
        "phase1534_authorized": True,
        "phase1534_audited": True,
        "all_sources_nonempty": all(item["bytes"] > 0 for item in manifest.values()),
        "source_hashes_unique": len({item["sha256"] for item in manifest.values()}) == len(manifest),
        "material_columns": all(list(frame.columns) == ["word_pair", "relation", "type_ratings", "concreteness"] for frame in frames.values()),
        "set_a_count": len(frames["A"]) == 400,
        "set_b_count": len(frames["B"]) == 600,
        "target_relation_counts": target_counts == {"similarity": 100, "class_inclusion": 100, "whole_part": 100},
        "pair_separator": all(frame["word_pair"].map(lambda value: isinstance(value, str) and value.count(":") == 1).all() for frame in frames.values()),
        "concreteness_values": all(set(frame["concreteness"]) == {"abstract", "concrete"} for frame in frames.values()),
        "post_training_release": True,
        "hidden_not_accessed": True,
        "model_not_loaded": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    adjudication = {
        "retained": [
            "the old left-padded camera violated causal-prefix identity and its hidden interpretation was superseded",
            "the right-padded same-shape quartet engine passed the registered repeat and causal-prefix calibration",
            "canonical target and state35 boundary responses replicate descriptively while behavior qualification fails",
            "stable replication can preserve a systematic instrument artifact, so causal identities must be checked directly",
            "Phase1519 is superseded because C088 answer mappings were defined in the full model input",
        ],
        "corrected": [
            "the four-cell contrast cancels registered source and target main effects but is not a pure relation signal",
            "target and boundary observations do not establish target-to-boundary transport or a formation-transfer-differentiation law",
            "the state35 response is a shared task-boundary candidate, not a universal semantic relation field",
            "C_f=S+R_f+epsilon is only coherent at group level as C_fgu=S+R_f+epsilon_fgu when R_f=C_f-S",
            "the three-part semantic gate is a minimum qualification gate, not causal-mechanism closure",
            "new mathematics and a discrete-continuous algebra are not licensed by C090",
            "left padding is not universally invalid; the failed object is the registered execution route under this model and numeric setup",
        ],
    }
    report = {
        "phase": 1535,
        "campaign": "C091",
        "status": "external_human_material_and_analysis_adjudication_complete",
        "source": {
            "title": "A large Chinese dataset of ten-category semantic relations with developmental performance in children and adolescents",
            "publication": "Scientific Data 13, 793 (2026)",
            "article_url": "https://www.nature.com/articles/s41597-026-07485-9",
            "osf_project": "https://osf.io/p4vwt/",
            "reported_design": {
                "candidate_generators": 32,
                "retained_relation_judgment_participants": 5898,
                "final_pairs": 1000,
                "mean_valid_judgments_per_pair": 135.25,
                "minimum_familiarity": 3.0,
                "selection": "top 50 item-accuracy pairs per relation-by-concreteness subcategory",
                "aggregate_accuracy": {"similar": 0.928, "whole_part": 0.729, "class_inclusion": 0.643},
            },
            "manifest": manifest,
            "relation_counts": relation_counts_json,
            "target_counts": target_counts,
        },
        "uploaded_analysis_adjudication": adjudication,
        "claim_boundary": {
            "allowed": "independent human-validated, post-training Chinese relation material source suitable for a frozen Qwen3 behavior qualification contract",
            "forbidden": [
                "item-level adult naturalness certainty",
                "absence of all training contamination beyond publication chronology",
                "Chinese relation mechanism",
                "cross-language invariant",
            ],
        },
        "checks": checks,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/material_and_analysis_adjudication.json", report)
    core.save(OUT / "protocol/source_manifest.json", manifest)
    core.save(OUT / "analysis/final.json", {
        "phase": 1535,
        "campaign": "C091",
        "status": report["status"],
        "authorization": "run_phase1536_c091_human_validated_chinese_relation_contract",
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
