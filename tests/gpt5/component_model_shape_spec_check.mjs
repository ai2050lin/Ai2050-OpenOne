import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  COMPONENT_MODEL_PHASES,
  getComponentModelSpec,
} from '../../frontend/src/blueprint/appleNeuron/componentModelSpec.js';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, '..', '..');
const outputDir = path.join(root, 'tests', 'result', 'component_model_shape_spec');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const expectedPhaseCount = 12;
const expectedShapes = new Set([
  'vector_portal',
  'normalization_tunnel',
  'multi_head_router',
  'neuron_expansion_field',
  'dual_path_merge',
]);

assert(Object.keys(COMPONENT_MODEL_PHASES).length === expectedPhaseCount, 'Component phase count mismatch');
assert(
  new Set(Object.values(COMPONENT_MODEL_PHASES).map((phase) => phase.shapeId)).size === expectedShapes.size,
  'Component shape family count mismatch'
);

for (const [phaseId, phase] of Object.entries(COMPONENT_MODEL_PHASES)) {
  assert(expectedShapes.has(phase.shapeId), `${phaseId}: unsupported shape ${phase.shapeId}`);
  const spec = getComponentModelSpec({ component: phase.component, phaseId, lang: 'zh', evidenceUnits: [] });
  assert(spec.shapeId === phase.shapeId, `${phaseId}: shape mapping mismatch`);
  assert(spec.title && spec.shape && spec.mechanism && spec.unitSummary && spec.boundary, `${phaseId}: incomplete explanation`);
}

const attentionSpec = getComponentModelSpec({
  component: 'attention',
  phaseId: 'attn_score',
  lang: 'zh',
  evidenceUnits: [
    { unit_kind: 'attention_head', unit_index: 22, display_priority: 2 },
    { unit_kind: 'attention_head', unit_index: 7, display_priority: 1 },
  ],
});
assert(attentionSpec.unitSummary.includes('H#22'), 'Attention explanation omits exact H# address');
assert(attentionSpec.exactUnitCount === 2, 'Attention exact unit count mismatch');

const ffnSpec = getComponentModelSpec({
  component: 'ffn',
  phaseId: 'ffn_act',
  lang: 'zh',
  evidenceUnits: [
    { unit_kind: 'mlp_product_neuron', unit_index: 774, display_priority: 2 },
    { unit_kind: 'mlp_product_group', unit_index: 28, display_priority: 1 },
  ],
});
assert(ffnSpec.unitSummary.includes('N#774'), 'FFN explanation omits exact N# address');
assert(ffnSpec.unitSummary.includes('G#28'), 'FFN explanation omits exact G# address');

for (const component of ['input', 'ln', 'residual']) {
  const phaseId = Object.keys(COMPONENT_MODEL_PHASES).find((id) => COMPONENT_MODEL_PHASES[id].component === component);
  const spec = getComponentModelSpec({ component, phaseId, lang: 'zh', evidenceUnits: [] });
  assert(spec.exactUnitCount === 0, `${component}: structural component must not claim exact neurons`);
  assert(!/H#\d|N#\d|G#\d/.test(spec.unitSummary), `${component}: structural component leaks physical unit labels`);
}

const result = {
  schema_version: 'component_model_shape_spec_validation.v1',
  status: 'passed',
  generated_at: new Date().toISOString(),
  phase_count: Object.keys(COMPONENT_MODEL_PHASES).length,
  shape_families: Array.from(expectedShapes),
  attention_example: attentionSpec.unitSummary,
  ffn_example: ffnSpec.unitSummary,
  evidence_boundary: 'Attention and MLP addresses are physical candidates; single-unit causal closure remains zero.',
};

fs.mkdirSync(outputDir, { recursive: true });
fs.writeFileSync(path.join(outputDir, 'validation.json'), `${JSON.stringify(result, null, 2)}\n`, 'utf8');
console.log(JSON.stringify(result, null, 2));
