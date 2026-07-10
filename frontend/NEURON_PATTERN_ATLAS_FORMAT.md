# Pattern-Family Physical Path Atlas v1

`pattern_family_neuron_atlas.v1` is the evidence-scoped bridge between model experiments and the 3D research client.

## Purpose

The package shows only physical units, component-set members, and path stages that have source artifacts. It does not publish synthetic placeholder neurons for unmapped pattern families.

```text
pattern family -> observed component path -> unit candidates / component-set members -> readout
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

## Scene Integration

The physical-path atlas is an overlay on the existing DNN scene. It must not replace or reshape the base model.

```text
base DNN layer plane: z = (layer - (layer_count - 1) / 2) * 0.92
atlas candidate: projected onto the same z plane with a deterministic radial position
```

Turning the overlay off must leave the original camera, layer stack, model label, forward-pass animation, layer expansion, and base interactions unchanged. Atlas coordinates are display coordinates, not learned geometry.

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

Phase326 extends the package with `attention_head` and `mlp_product_group` component-set members. These remain distinct from `unit_candidate` nodes. Its `single_unit_causal_count` is zero.

Phase327 annotates the frozen Phase326 members with mechanism-level natural identity, position necessity, natural-state transplant, and complete-generation results. It adds `path.natural_retrieval_paths` to the content-knowledge partitions. These paths do not create new neuron nodes and remain `causal=false` because no upstream intervention was shown to alter the downstream frozen set.

Phase328 adds `path.upstream_residual_mediation_edges` for registered pooled query-residual interventions. A single-model mediation pass is displayed as a candidate state, not a causal edge. `causal=true` requires cross-model mediation together with natural top-1 answer unlock; the Phase328 published count is zero.

Phase329 adds `path.full_vocabulary_mediation_paths`. Each path records the residual observation layer, the correctly aligned next-block input layer, full-vocabulary blocker decline, tokenwise-versus-pooled comparison, carrier-member mediation, top-1 unlock, generation improvement, and the single-unit intervention gate. These paths are evidence overlays on existing physical members; they do not create neurons. Function and format tokens at answer onset are stored as surface-protocol competitors and are not labeled semantic blockers without a separate causal test.

Phase330 maps all nine registered families for all three models. Each family/model partition contains eight mechanisms, full-layer attention/MLP/residual path anchors at source/query/last roles, and 32 frozen component-set members. The global package adds `phase330_paths.jsonl`, `phase330_carrier_sets.jsonl`, and `phase330_claim_registry.jsonl`. A node may carry `phase330_registered_set_support` or `phase330_cross_model_readout_specific`; neither field means visible-behavior closure or single-neuron causality. The client preserves the original DNN geometry and renders these members only as a physical evidence overlay.

## Node Types

```text
unit_candidate: localized real-unit candidate from Phase286/287
component_set_member: frozen attention-head or MLP-product-group member from Phase326
```

`expanded_confirmation_pass=true` means a frozen component set passed the larger new-object and new-template necessity audit for that model and mechanism. It does not make every member individually causal.

## Edge Semantics

`observed_component_sequence` means events were recorded in forward order. It is explicitly non-causal.

`contains_localized_candidate` links a layer anchor to a physical candidate address. It is explicitly non-causal.

Future causal edges must include intervention, matched control, heldout replication, side effects, and an explicit causal scope.

## Build

```bash
python tests/gpt5/phase330_publish_global_atlas.py
```

The builder writes the canonical package to:

```text
tests/gpt5/result/pattern_family_neuron_atlas/v1
```

and mirrors the same bytes to:

```text
frontend/public/vis_data/pattern_family_neuron_atlas/v1
```
