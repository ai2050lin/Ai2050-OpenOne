import { copyFile, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const frontendDir = path.resolve(scriptDir, '..');
const repoRoot = path.resolve(frontendDir, '..');
const sourceDir = path.join(repoRoot, 'tests', 'result', 'pattern_family_atlas', 'v1');
const targetDir = path.join(frontendDir, 'public', 'vis_data', 'pattern_family_atlas', 'v1');

const requiredFiles = [
  'manifest.json',
  'schema.json',
  'client_index.json',
  'families.jsonl',
  'modes.jsonl',
  'test_cases.jsonl',
  'runs.jsonl',
  'observations.jsonl',
  'metrics.jsonl',
  'graph_nodes.jsonl',
  'graph_edges.jsonl',
  'progress.json',
  'summary.md',
];

async function exists(filePath) {
  try {
    await readFile(filePath);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  if (!(await exists(path.join(sourceDir, 'manifest.json')))) {
    throw new Error(`Pattern atlas source is missing: ${sourceDir}`);
  }

  await rm(targetDir, { recursive: true, force: true });
  await mkdir(targetDir, { recursive: true });

  const copied = [];
  for (const fileName of requiredFiles) {
    const sourcePath = path.join(sourceDir, fileName);
    if (!(await exists(sourcePath))) continue;
    await copyFile(sourcePath, path.join(targetDir, fileName));
    copied.push(fileName);
  }

  const sourceEntries = await readdir(sourceDir);
  for (const fileName of sourceEntries) {
    if (copied.includes(fileName)) continue;
    if (!/\.(json|jsonl|md)$/i.test(fileName)) continue;
    await copyFile(path.join(sourceDir, fileName), path.join(targetDir, fileName));
    copied.push(fileName);
  }

  const publicManifest = {
    schema_version: 'pattern_family_atlas_public_v1',
    generated_at: new Date().toISOString(),
    source: path.relative(repoRoot, sourceDir).replace(/\\/g, '/'),
    public_base: '/vis_data/pattern_family_atlas/v1',
    entrypoint: '/vis_data/pattern_family_atlas/v1/manifest.json',
    files: copied.sort(),
  };

  await writeFile(
    path.join(targetDir, 'public_manifest.json'),
    JSON.stringify(publicManifest, null, 2) + '\n',
    'utf8'
  );

  console.log(`Synced ${copied.length} pattern atlas files to ${path.relative(repoRoot, targetDir)}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
