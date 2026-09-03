import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const frontendDir = path.resolve(scriptDir, '..');
const repoRoot = path.resolve(frontendDir, '..');
const resultsDir = path.join(repoRoot, 'results');
const publicVisDir = process.env.AI2050_RESEARCH_ASSET_ROOT
  ? path.resolve(process.env.AI2050_RESEARCH_ASSET_ROOT)
  : path.join(repoRoot, 'tests', 'glm5', 'result', 'client_visualization_assets');
const publicAtlasDir = path.join(publicVisDir, 'atlas');

function sanitizeFilename(value) {
  return value
    .replace(/\\/g, '/')
    .replace(/\.json$/i, '')
    .replace(/[^a-zA-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 180) + '.json';
}

function extractPhase(data, fallbackPath) {
  const explicit = Number(data?.model_info?.phase ?? data?.metrics?.source_phase);
  if (Number.isFinite(explicit)) return explicit;
  const match = fallbackPath.match(/phase(\d+)/i);
  return match ? Number(match[1]) : null;
}

function buildLabel(data, relPath, phase) {
  const parts = relPath.replace(/\\/g, '/').split('/');
  const runName = parts[0] || path.basename(relPath, '.json');
  const splitName = parts.length > 2 ? parts[1] : null;
  const readableRunName = runName
    .replace(/^glm5_/, '')
    .replace(/^gpt5_/, '')
    .replace(/_/g, ' ');
  const title = data?.title || readableRunName;
  const model = data?.model_info?.model || data?.model || splitName || 'atlas';
  const prefix = Number.isFinite(phase) ? `Phase ${phase}` : 'Atlas';
  return `${prefix} · ${model} · ${title}`;
}

async function walk(dir) {
  const entries = await import('node:fs/promises').then((fs) => fs.readdir(dir, { withFileTypes: true }));
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await walk(fullPath));
    } else if (entry.isFile() && entry.name.endsWith('atlas_graph.json')) {
      files.push(fullPath);
    }
  }
  return files;
}

async function main() {
  await mkdir(publicAtlasDir, { recursive: true });

  const sourceFiles = await walk(resultsDir);
  const manifestFiles = [];

  for (const sourcePath of sourceFiles) {
    const relPath = path.relative(resultsDir, sourcePath);
    let data;
    try {
      data = JSON.parse(await readFile(sourcePath, 'utf8'));
    } catch {
      continue;
    }
    if (data?.schema_version !== 'atlas_graph_v1') {
      continue;
    }

    const publicName = sanitizeFilename(relPath);
    const targetPath = path.join(publicAtlasDir, publicName);
    await copyFile(sourcePath, targetPath);

    const nodes = Array.isArray(data?.graph?.nodes) ? data.graph.nodes.length : 0;
    const edges = Array.isArray(data?.graph?.edges)
      ? data.graph.edges.length
      : Array.isArray(data?.graph?.links)
      ? data.graph.links.length
      : 0;
    const phase = extractPhase(data, relPath);

    manifestFiles.push({
      filename: `atlas/${publicName}`,
      label: buildLabel(data, relPath, phase),
      schema_version: 'atlas_graph_v1',
      phase,
      model: data?.model_info?.model || data?.model || null,
      title: data?.title || null,
      node_count: nodes,
      edge_count: edges,
      source: relPath.replace(/\\/g, '/'),
    });
  }

  manifestFiles.sort((a, b) => {
    const phaseDiff = (b.phase ?? -1) - (a.phase ?? -1);
    if (phaseDiff) return phaseDiff;
    return a.source.localeCompare(b.source);
  });

  const manifest = {
    schema_version: 'vis_data_manifest_v1',
    generated_at: new Date().toISOString(),
    files: manifestFiles,
  };

  await writeFile(
    path.join(publicVisDir, 'manifest.json'),
    JSON.stringify(manifest, null, 2) + '\n',
    'utf8'
  );

  console.log(`Wrote ${manifestFiles.length} atlas graph entries to ${path.relative(repoRoot, publicVisDir)}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
