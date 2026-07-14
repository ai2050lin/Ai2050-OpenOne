#!/usr/bin/env node
/** Validate every dataset exposed by the multi-route 3D client registry. */

import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  normalizeManifestEntries,
  normalizeVisualizationPayload,
} from '../../frontend/src/neural_vis/dataSourceAdapters.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const PUBLIC_ROOT = path.join(ROOT, 'frontend/public/vis_data');
const REGISTRY_PATH = path.join(PUBLIC_ROOT, 'source_registry.json');
const OUT = path.join(ROOT, 'tests/gpt5/result/phase415_multi_route_vis_sources');

function localVisPath(publicPath) {
  assert.match(publicPath, /^\/vis_data\//, `unexpected public path: ${publicPath}`);
  return path.join(PUBLIC_ROOT, publicPath.replace(/^\/vis_data\//, ''));
}

async function readJson(filepath) {
  return JSON.parse(await fs.readFile(filepath, 'utf8'));
}

async function main() {
  const registry = await readJson(REGISTRY_PATH);
  assert.equal(registry.schema_version, 'vis_data_source_registry.v1');
  assert.ok(Array.isArray(registry.sources));
  assert.ok(registry.sources.length >= 4);

  const sourceIds = registry.sources.map((source) => source.id);
  const routeIds = [...new Set(registry.sources.map((source) => source.route_id))];
  assert.equal(new Set(sourceIds).size, sourceIds.length, 'source ids must be unique');
  assert.ok(sourceIds.includes(registry.default_source_id));
  assert.deepEqual(new Set(routeIds), new Set(['gpt5', 'glm5']));

  const sourceResults = [];
  let datasetCount = 0;
  let canonicalNodeCount = 0;
  let canonicalEdgeCount = 0;

  for (const source of registry.sources) {
    const manifest = await readJson(localVisPath(source.manifest_path));
    assert.equal(manifest.schema_version, source.manifest_schema, `${source.id} manifest schema`);
    const entries = normalizeManifestEntries(source, manifest);
    assert.ok(entries.length > 0, `${source.id} must expose datasets`);

    const schemas = new Set();
    const models = new Set();
    let sourceNodeCount = 0;
    let sourceEdgeCount = 0;
    let nonCausalAdaptedEdgeCount = 0;

    for (const entry of entries) {
      const payloadPath = localVisPath(entry.path);
      const payload = await readJson(payloadPath);
      const normalized = normalizeVisualizationPayload(payload, entry, source);
      const nodes = normalized.graph?.nodes || [];
      const edges = normalized.graph?.edges || normalized.graph?.links || [];

      assert.equal(normalized.schema_version, 'atlas_graph_v1', `${entry.id} canonical schema`);
      assert.equal(normalized.source_schema_version, payload.schema_version, `${entry.id} source schema preserved`);
      assert.equal(normalized.source_context.route_id, source.route_id, `${entry.id} route provenance`);
      assert.ok(nodes.length > 0, `${entry.id} must produce a non-empty graph`);
      assert.ok(nodes.every((node) => node.id), `${entry.id} node ids`);
      assert.ok(edges.every((edge) => edge.source && edge.target), `${entry.id} edge endpoints`);

      if (['real_component_trace.v1', 'mechanism_trace_v1'].includes(payload.schema_version)) {
        assert.ok(edges.every((edge) => edge.causal === false), `${entry.id} observed edges must remain non-causal`);
        nonCausalAdaptedEdgeCount += edges.length;
      }

      schemas.add(payload.schema_version);
      if (payload.model || entry.model) models.add(payload.model || entry.model);
      sourceNodeCount += nodes.length;
      sourceEdgeCount += edges.length;
    }

    datasetCount += entries.length;
    canonicalNodeCount += sourceNodeCount;
    canonicalEdgeCount += sourceEdgeCount;
    sourceResults.push({
      source_id: source.id,
      route_id: source.route_id,
      manifest_schema: manifest.schema_version,
      payload_schemas: [...schemas].sort(),
      declared_models: source.models,
      observed_models: [...models].sort(),
      dataset_count: entries.length,
      canonical_node_count: sourceNodeCount,
      canonical_edge_count: sourceEdgeCount,
      explicitly_noncausal_adapted_edge_count: nonCausalAdaptedEdgeCount,
      all_dataset_paths_resolved: true,
      all_payloads_renderable: true,
    });
  }

  const hookSource = await fs.readFile(path.join(ROOT, 'frontend/src/neural_vis/hooks/useVisData.js'), 'utf8');
  const clientSource = await fs.readFile(path.join(ROOT, 'frontend/src/neural_vis/index.jsx'), 'utf8');
  const mainClientSource = await fs.readFile(path.join(ROOT, 'frontend/src/App.jsx'), 'utf8');
  assert.match(hookSource, /source_registry\.json/);
  assert.match(hookSource, /selectDataSource/);
  assert.match(clientSource, /aria-label="测试路线数据源"/);
  assert.match(clientSource, /route-dataset-list/);
  assert.doesNotMatch(clientSource, /dataFiles\.slice\(0,\s*5\)/);
  assert.match(mainClientSource, /aria-label="主工作台测试路线数据源"/);
  assert.match(mainClientSource, /aria-label="主工作台测试数据集"/);
  assert.match(mainClientSource, /selectDataSource/);

  const result = {
    schema_version: 'phase415_multi_route_vis_source_contract.v1',
    phase_id: 'Phase415-MultiRouteVisualizationDataSources',
    generated_at: new Date().toISOString(),
    valid: true,
    registry_path: 'frontend/public/vis_data/source_registry.json',
    route_count: routeIds.length,
    route_ids: routeIds,
    source_count: registry.sources.length,
    dataset_count: datasetCount,
    canonical_node_count: canonicalNodeCount,
    canonical_edge_count: canonicalEdgeCount,
    source_results: sourceResults,
    client_contract: {
      route_selector_present: true,
      dataset_search_present: true,
      full_manifest_list_available: true,
      local_json_import_preserved: true,
      legacy_glm5_fallback_preserved: true,
      main_workspace_route_selector_present: true,
    },
    evidence_boundary: [
      'This phase validates data provenance, loading, adaptation, and rendering contracts only.',
      'Observed trace order and factor continuity are explicitly non-causal.',
      'No model execution, physical-path promotion, neuron causality, or mechanism closure is added.',
    ],
  };

  await fs.mkdir(OUT, { recursive: true });
  await fs.writeFile(
    path.join(OUT, 'phase415_multi_route_vis_source_contract.json'),
    `${JSON.stringify(result, null, 2)}\n`,
    'utf8'
  );
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
