#!/usr/bin/env python3
"""Test whether Phase1093 relation geometry follows pair-incidence topology."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_finalize as common
import phase1092_natural_bilingual_attribute_protocol as prior_protocol
import phase1093_independent_relation_finalize as finalizer
import phase1093_independent_relation_protocol as protocol


EPSILON = 1e-12
BAND = (0.45, 0.62)


def relation_from_gram(gram: np.ndarray) -> np.ndarray:
    upper = gram[np.triu_indices(gram.shape[0], k=1)]
    upper = upper - upper.mean()
    norm = float(np.linalg.norm(upper))
    return upper / norm if norm > EPSILON else np.zeros_like(upper)


def incidence_bank(module, attribute: str, *, oriented: bool) -> np.ndarray:
    pairs = module.ATTRIBUTE_PAIRS[attribute]
    concepts = sorted({value for pair in pairs for value in pair})
    concept_index = {value: index for index, value in enumerate(concepts)}
    bank = np.zeros((len(pairs), len(concepts)), dtype=np.float64)
    for row, (left, right) in enumerate(pairs):
        bank[row, concept_index[left]] = -1.0 if oriented else 1.0
        bank[row, concept_index[right]] = 1.0
    return common.row_normalize(bank, centered=True)


def permutation_relations(reference: np.ndarray) -> np.ndarray:
    gram = reference @ reference.T
    rows = []
    for permutation in itertools.permutations(range(reference.shape[0])):
        indices = np.asarray(permutation, dtype=np.int64)
        rows.append(relation_from_gram(gram[np.ix_(indices, indices)]))
    return np.stack(rows)


def fit_row(bank: np.ndarray, reference: np.ndarray, permutations: np.ndarray) -> dict:
    observed = common.relation_vector(bank)
    target = common.relation_vector(reference)
    score = float(np.dot(observed, target))
    permutation_scores = permutations @ observed
    return {
        "cosine": score,
        "exact_upper_tail_p": float(np.mean(permutation_scores >= score)),
        "permutation_count": int(permutation_scores.shape[0]),
    }


def audit_models(models: dict, module) -> dict:
    references = {}
    permutations = {}
    for attribute in module.ATTRIBUTES:
        references[attribute] = {
            "oriented": incidence_bank(module, attribute, oriented=True),
            "unoriented": incidence_bank(module, attribute, oriented=False),
        }
        permutations[attribute] = {
            key: permutation_relations(value)
            for key, value in references[attribute].items()
        }

    by_model = {}
    for model_name, data in models.items():
        attributes = {}
        for attribute in module.ATTRIBUTES:
            rows = []
            for surface in module.SURFACES:
                for split in module.SPLITS:
                    for replicate in range(module.SIGNED_PROJECTION_REPLICATES):
                        row = {
                            "surface": surface,
                            "split": split,
                            "replicate": replicate,
                        }
                        for field in ("content", "field_null"):
                            bank = finalizer.band_bank(
                                data,
                                module,
                                attribute,
                                surface,
                                split,
                                field,
                                replicate,
                                BAND[0],
                                BAND[1],
                            )
                            row[field] = {
                                key: fit_row(
                                    bank,
                                    references[attribute][key],
                                    permutations[attribute][key],
                                )
                                for key in ("oriented", "unoriented")
                            }
                        rows.append(row)

            summaries = {}
            for topology in ("oriented", "unoriented"):
                content = np.asarray(
                    [row["content"][topology]["cosine"] for row in rows],
                    dtype=np.float64,
                )
                null = np.asarray(
                    [row["field_null"][topology]["cosine"] for row in rows],
                    dtype=np.float64,
                )
                p_values = np.asarray(
                    [
                        row["content"][topology]["exact_upper_tail_p"]
                        for row in rows
                    ],
                    dtype=np.float64,
                )
                summaries[topology] = {
                    "mean_content_fit": float(content.mean()),
                    "mean_field_null_fit": float(null.mean()),
                    "mean_content_over_null_advantage": float(
                        (content - null).mean()
                    ),
                    "median_content_exact_p": float(np.median(p_values)),
                    "content_fit_at_least_0_5": int(np.sum(content >= 0.5)),
                    "content_exceeds_null": int(np.sum(content > null)),
                    "row_count": int(content.shape[0]),
                }
            attributes[attribute] = {
                "rows": rows,
                "summary": summaries,
            }
        by_model[model_name] = {"attributes": attributes}
    return by_model


def main() -> None:
    current_models = finalizer.load_current_models()
    prior_models = finalizer.load_prior_models()
    preregistration = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    by_phase = {
        "phase1092": audit_models(prior_models, prior_protocol),
        "phase1093": audit_models(current_models, protocol),
    }

    output = {
        "schema_version": "phase1093_posthoc_incidence_audit.v2",
        "phase": 1093,
        "protocol_digest": preregistration["protocol_digest"],
        "normalized_band": list(BAND),
        "by_phase": by_phase,
        "interpretation": [
            "This is a post hoc confound audit and cannot upgrade a frozen gate.",
            "A high fit means the observed Gram may be explained by the directed or undirected pair-incidence graph.",
            "A high fit does not prove lexical overlap caused the geometry; it shows the current protocol cannot distinguish that explanation from semantic relation structure.",
        ],
    }
    output["incidence_audit_digest"] = protocol.digest(output)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "posthoc_incidence_audit.json",
        output,
    )

    compact = {
        phase_name: {
            model_name: {
                attribute: attribute_row["summary"]
                for attribute, attribute_row in model_row["attributes"].items()
            }
            for model_name, model_row in phase_rows.items()
        }
        for phase_name, phase_rows in by_phase.items()
    }
    print(
        {
            "phase": 1093,
            "compact": compact,
            "incidence_audit_digest": output["incidence_audit_digest"],
        }
    )


if __name__ == "__main__":
    main()
