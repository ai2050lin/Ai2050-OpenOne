import { copyFile, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const frontendDir = path.resolve(scriptDir, '..');
const repoRoot = path.resolve(frontendDir, '..');
const sourceDir = path.join(repoRoot, 'tests', 'result', 'pattern_family_atlas', 'v2');
const targetDir = path.join(frontendDir, 'public', 'vis_data', 'pattern_family_atlas', 'v2');

async function exists(filePath) {
  try {
    await readFile(filePath);
    return true;
  } catch {
    return false;
  }
}

async function copyTree(source, target, copied, base = sourceDir) {
  await mkdir(target, { recursive: true });
  for (const entry of await readdir(source, { withFileTypes: true })) {
    const from = path.join(source, entry.name);
    const to = path.join(target, entry.name);
    if (entry.isDirectory()) {
      await copyTree(from, to, copied, base);
    } else if (/\.(json|jsonl|md)$/i.test(entry.name)) {
      await copyFile(from, to);
      copied.push(path.relative(base, from).replace(/\\/g, '/'));
    }
  }
}

async function main() {
  if (!(await exists(path.join(sourceDir, 'manifest.json')))) {
    throw new Error(`Pattern atlas v2 source is missing: ${sourceDir}`);
  }
  await rm(targetDir, { recursive: true, force: true });
  await mkdir(targetDir, { recursive: true });
  const copied = [];
  await copyTree(sourceDir, targetDir, copied);
  const publicManifest = {
    schema_version: 'pattern_family_atlas_public_v2',
    generated_at: new Date().toISOString(),
    source: path.relative(repoRoot, sourceDir).replace(/\\/g, '/'),
    public_base: '/vis_data/pattern_family_atlas/v2',
    entrypoint: '/vis_data/pattern_family_atlas/v2/manifest.json',
    files: copied.sort(),
  };
  await writeFile(
    path.join(targetDir, 'public_manifest.json'),
    JSON.stringify(publicManifest, null, 2) + '\n',
    'utf8'
  );
  console.log(`Synced ${copied.length} pattern atlas v2 files to ${path.relative(repoRoot, targetDir)}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
