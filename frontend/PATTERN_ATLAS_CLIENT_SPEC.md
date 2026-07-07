# Pattern Atlas Client Spec

`pattern_family_atlas_v1` is the fixed data package for the language pattern atlas. The client should treat it as the stable entry for pattern-family progress, mechanism graph status, and phase-by-phase updates.

Current package:

```text
tests/result/pattern_family_atlas/v1/
```

## Load Order

1. Load `manifest.json`.
2. Read `manifest.files`.
3. Load `schema.json` and `client_index.json`.
4. Load only the files required by the selected view.

The client should not infer file names from phase numbers. It should always start from `manifest.json`.

## Required Views

### Pattern Atlas Overview

Read:

```text
manifest.json
progress.json
families.jsonl
graph_nodes.jsonl
graph_edges.jsonl
```

Display:

- global atlas progress
- family cards with status and progress
- priority queue
- recent phase update summary
- small-model bias warning

### Family Detail

Read:

```text
families.jsonl
modes.jsonl
metrics.jsonl
graph_nodes.jsonl
graph_edges.jsonl
```

Display:

- modes under one family
- status per mode
- evidence score
- unresolved failure boundary
- next recommended phase

### Mode Detail

Read:

```text
modes.jsonl
observations.jsonl
metrics.jsonl
graph_edges.jsonl
```

Display:

- behavior result distribution
- trigger/state/readout/source/hook/closure evidence
- rollout trace if present
- closure flags

### Mechanism Graph

Read:

```text
graph_nodes.jsonl
graph_edges.jsonl
```

Display:

- pattern-family nodes
- mode nodes
- known evidence edges
- relation/evidence/status filters

### Model Compare

Read:

```text
observations.jsonl
metrics.jsonl
graph_edges.jsonl
```

Display:

- qwen3 / glm4 / deepseek7b comparison
- competitor distribution
- model-specific weak evidence
- small-model roughness warning

## Status Order

The client should sort status by `schema.status_order`:

```text
not_started
behavior_tested
trigger_mapped
state_mapped
readout_mapped
source_candidate
hook_supported
closure_candidate
closed
failed_or_deprioritized
```

## Evidence Score

The client should use the stored `evidence_score` when available. If it needs to recompute the score, use `schema.evidence_weights`:

```text
EvidenceScore =
  0.15 * behavior_consistency
+ 0.15 * state_consistency
+ 0.20 * readout_consistency
+ 0.25 * hook_causal_support
+ 0.15 * closure_support
+ 0.10 * cross_model_consistency
```

Do not show a mode as closed unless `closure_support` is present and the status is `closed`.

## Update Rule

Every new phase that changes the pattern atlas should update the same package path:

```text
tests/result/pattern_family_atlas/v1/
```

At minimum it must update:

```text
manifest.json
progress.json
runs.jsonl
observations.jsonl
metrics.jsonl
graph_nodes.jsonl
graph_edges.jsonl
summary.md
```

The client should support append-style growth in JSONL files. If a repeated ID appears, the latest `created_at` record wins.

## Relation To Atlas Graph v1

`ATLAS_GRAPH_FORMAT.md` remains the 3D mechanism graph format. `pattern_family_atlas_v1` is broader:

- it tracks pattern families and modes;
- it tracks progress and fixed test output files;
- it includes graph nodes and edges that can be converted into `atlas_graph_v1`.

The conversion rule is simple:

```text
graph_nodes.jsonl -> atlas_graph_v1.graph.nodes
graph_edges.jsonl -> atlas_graph_v1.graph.edges
progress.json     -> atlas_graph_v1.metrics
```

## Current Limitation

Phase235 defines the data contract and imports known evidence from earlier phases. It is not a new behavior benchmark. The next benchmark phase should fill `observations.jsonl` with cross-model case-level results.
