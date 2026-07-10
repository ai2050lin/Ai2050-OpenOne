# Pattern-Family Neuron Atlas v1

`pattern_family_neuron_atlas.v1` is the evidence-scoped bridge between model experiments and the 3D research client.

## Purpose

The package shows only physical units and path stages that have source artifacts. It does not publish synthetic placeholder neurons for unmapped pattern families.

```text
pattern family -> observed component path -> localized unit candidates -> readout
```

An ordered trace is not a causal path. A channel-group intervention is not single-unit causality.

## Entry Point

```text
/vis_data/pattern_family_neuron_atlas/v1/manifest.json
```

The client loads `manifest.json` and `families.json` first. It then loads one partition only:

```text
partitions/{family_id}/{model}.json
```

## Canonical Files

```text
manifest.json
families.json
neuron_index.json
neuron_nodes.jsonl
neuron_edges.jsonl
neuron_events.jsonl
neuron_interventions.jsonl
neuron_runs.jsonl
checksums.json
```

The JSONL files are the append/analysis representation. Partition JSON files are the client read model.

## Unit Identity

Every unit node requires:

```text
node_id
family_id
model
model_revision
layer
component
unit_kind
unit_index
run_id
```

The client must never use a bare unit index as a cross-model identity.

## Evidence States

```text
L2 natural observation
L4 localized component candidate
channel_group_not_single_unit
single_unit_causal
negative
missing
```

The current Phase325 package contains L2 observations and L4 localized candidates. Its `single_unit_causal_count` is zero.

## Edge Semantics

`observed_component_sequence` means events were recorded in forward order. It is explicitly non-causal.

`contains_localized_candidate` links a layer anchor to a physical candidate address. It is explicitly non-causal.

Future causal edges must include intervention, matched control, heldout replication, side effects, and an explicit causal scope.

## Build

```bash
python tests/gpt5/phase325_pattern_family_neuron_atlas.py
```

The builder writes the canonical package to:

```text
tests/gpt5/result/pattern_family_neuron_atlas/v1
```

and mirrors the same bytes to:

```text
frontend/public/vis_data/pattern_family_neuron_atlas/v1
```
