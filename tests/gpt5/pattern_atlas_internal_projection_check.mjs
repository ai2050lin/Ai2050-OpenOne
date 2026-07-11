import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  dedupePatternAtlasUnits,
  patternAtlasPhysicalKey,
  patternAtlasUnitAddressLabel,
  selectBalancedPatternAtlasNodes,
} from '../../frontend/src/researchKernel/patternAtlasEvidence.js';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, '..', '..');
const atlasRoot = path.join(root, 'frontend', 'public', 'vis_data', 'pattern_family_neuron_atlas', 'v1');
const outputDir = path.join(root, 'tests', 'result', 'pattern_atlas_internal_projection');
const manifest = JSON.parse(fs.readFileSync(path.join(atlasRoot, 'manifest.json'), 'utf8'));

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const summaries = manifest.partitions.map((partitionRef) => {
  const partition = JSON.parse(fs.readFileSync(path.join(atlasRoot, partitionRef.path), 'utf8'));
  const snapshot = partition.model_snapshot || {};
  const nodes = partition.nodes || [];

  nodes.forEach((node) => {
    assert(Number.isInteger(Number(node.layer)), `${partitionRef.path}: invalid layer`);
    assert(['attention', 'mlp'].includes(node.component), `${partitionRef.path}: unsupported component ${node.component}`);
    assert(Number.isInteger(Number(node.unit_index)) && Number(node.unit_index) >= 0, `${partitionRef.path}: invalid unit index`);
    if (node.unit_kind === 'attention_head') {
      assert(Number(node.unit_index) < Number(snapshot.num_attention_heads), `${partitionRef.path}: attention head out of range`);
      assert(patternAtlasUnitAddressLabel(node).startsWith('H#'), `${partitionRef.path}: attention prefix mismatch`);
    } else if (node.unit_kind === 'mlp_product_neuron') {
      assert(Number(node.unit_index) < Number(snapshot.intermediate_size), `${partitionRef.path}: MLP neuron out of range`);
      assert(patternAtlasUnitAddressLabel(node).startsWith('N#'), `${partitionRef.path}: neuron prefix mismatch`);
    } else if (node.unit_kind === 'mlp_product_group') {
      assert(patternAtlasUnitAddressLabel(node).startsWith('G#'), `${partitionRef.path}: group prefix mismatch`);
    }
  });

  const focusCounts = {};
  for (const focus of ['key', 'natural', 'group', 'confirmed']) {
    const selected = selectBalancedPatternAtlasNodes(nodes, focus, 48);
    assert(selected.length <= 48, `${partitionRef.path}: selection exceeds display limit`);
    if (focus === 'natural') assert(selected.every((node) => node.natural_observed), `${partitionRef.path}: natural filter leak`);
    if (focus === 'group') assert(selected.every((node) => node.group_intervention_supported), `${partitionRef.path}: group filter leak`);
    if (focus === 'confirmed') assert(selected.every((node) => node.expanded_confirmation_pass), `${partitionRef.path}: confirmed filter leak`);
    focusCounts[focus] = selected.length;
  }

  const keyUnits = dedupePatternAtlasUnits(selectBalancedPatternAtlasNodes(nodes, 'key', 48));
  assert(new Set(keyUnits.map(patternAtlasPhysicalKey)).size === keyUnits.length, `${partitionRef.path}: duplicate physical units after dedupe`);

  return {
    family_id: partitionRef.family_id,
    model: partitionRef.model,
    node_count: nodes.length,
    selected_physical_unit_count: keyUnits.length,
    exact_attention_head_count: keyUnits.filter((node) => node.unit_kind === 'attention_head').length,
    exact_mlp_neuron_count: keyUnits.filter((node) => node.unit_kind === 'mlp_product_neuron').length,
    exact_mlp_group_count: keyUnits.filter((node) => node.unit_kind === 'mlp_product_group').length,
    single_unit_causal_count: Number(partition.metrics?.single_unit_causal_count || 0),
    focus_counts: focusCounts,
  };
});

const qwenContent = summaries.find((row) => row.family_id === 'content_knowledge' && row.model === 'qwen3');
assert(qwenContent?.exact_mlp_neuron_count > 0, 'Qwen3 content atlas has no exact MLP neuron candidate in the visible selection');

const result = {
  schema_version: 'pattern_atlas_internal_projection_validation.v1',
  status: 'passed',
  generated_at: new Date().toISOString(),
  partition_count: summaries.length,
  evidence_boundary: manifest.evidence_boundary,
  partitions: summaries,
};

fs.mkdirSync(outputDir, { recursive: true });
fs.writeFileSync(path.join(outputDir, 'validation.json'), `${JSON.stringify(result, null, 2)}\n`, 'utf8');
console.log(JSON.stringify(result, null, 2));
