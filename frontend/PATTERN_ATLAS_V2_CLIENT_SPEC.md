# Pattern Family Atlas v2 Client Spec

## Goal

Build the client around a light index first, detail-on-demand model:

```text
manifest.json
client_index.json
atlas_scores.jsonl
families.jsonl
```

The primary table is:

```text
path_signature_rows.jsonl
```

Large per-case details live under:

```text
case_details/{model}__{case_id}.json
```

## Views

1. Overview: global progress, latest phase, candidate counts.
2. Family Matrix: `family_id x model` score matrix from `atlas_scores.jsonl`.
3. Path Explorer: prompt -> state -> layer path -> components -> readout -> closure quality.
4. Component View: attention / MLP / residual scores and dominant layers.
5. Causal Audit: zero / half / mean-replace / random controls with side-effect flags.
6. Case Detail: raw JSON loaded from `detail_ref`.
7. Gap Matrix: family-model pressure cells from `phase274_coverage_matrix_rows.jsonl`.
8. Gap Queue: objective missing dimensions from `phase274_gap_rows.jsonl`.
9. Batch Planner: first executable queue from `phase274_selected_batch_rows.jsonl`.
10. Fill Results: Phase275 component and causal fill rows.
11. Causal Fill Audit: low-side-effect support and side-effect risk from `phase275_causal_fill_rows.jsonl`.

## Client Rules

- Initial load must not read observations or raw detail files.
- Detail files are loaded only after selecting a case.
- Every cell should expose `overall`, `physical path`, `component`, `causal`, and `closure` scores.
- `high_quality_candidate_not_closed` must be displayed as a candidate, not a closure result.
- Phase274 gap files are prioritization data, not new model evidence.
- `candidate_closure_verification` means the row already has some path evidence but still needs strict closure recheck.
- Phase275 fill files are physical-path evidence for selected queue rows, still not closure proof.
