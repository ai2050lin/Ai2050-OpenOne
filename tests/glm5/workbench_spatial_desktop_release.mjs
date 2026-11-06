import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { extractFile } from '../../frontend/node_modules/@electron/asar/lib/asar.js';
import { Arch, build, Platform } from '../../frontend/node_modules/electron-builder/out/index.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const frontend = path.join(root, 'frontend');
const results = process.env.AI2050_DESKTOP_RESULTS || path.join(root, 'tests/glm5/result/workbench_spatial_desktop');
const browserProof = process.env.AI2050_DESKTOP_BROWSER_PROOF || path.join(root, 'tests/glm5/result/workbench_spatial_unified/playwright.json');
assert.equal(JSON.parse(await fs.readFile(browserProof, 'utf8')).passed, true);
const pkg = JSON.parse(await fs.readFile(path.join(frontend, 'package.json'), 'utf8'));
const lock = JSON.parse(await fs.readFile(path.join(frontend, 'package-lock.json'), 'utf8'));
assert.equal(lock.version, pkg.version);
assert.equal(lock.packages[''].version, pkg.version);
assert.deepEqual(pkg.dependencies || {}, {});

const frontendDist = process.env.AI2050_DESKTOP_FRONTEND_DIST || path.join(frontend, 'dist');
const output = process.env.AI2050_DESKTOP_BUILD_DIR || path.join(frontend, 'dist-installer', pkg.version);
const unpacked = path.join(output, 'win-unpacked');
const resources = pkg.build.extraResources.map(item => ({ ...item, from: path.resolve(frontend, item.from) }));
const { build: buildConfig, devDependencies, ...runtime } = pkg;
if (!process.argv.includes('--verify-only')) {
  const stage = await fs.mkdtemp(path.join(os.tmpdir(), 'ai2050-spatial-stage-'));
  try {
    await fs.cp(frontendDist, path.join(stage, 'dist'), { recursive: true });
    await fs.cp(path.join(frontend, 'electron'), path.join(stage, 'electron'), { recursive: true });
    await fs.writeFile(path.join(stage, 'package.json'), JSON.stringify({ ...runtime, packageManager: 'traversal@1' }, null, 2));
    process.env.CSC_IDENTITY_AUTO_DISCOVERY = 'false';
    process.env.ELECTRON_BUILDER_CACHE ||= path.join(process.env.LOCALAPPDATA, 'electron-builder/Cache');
    await build({
      projectDir: stage,
      config: {
        ...buildConfig,
        directories: { output, buildResources: path.join(frontend, 'electron') },
        npmRebuild: false,
        electronVersion: devDependencies.electron.replace('^', ''),
        electronDist: path.join(frontend, 'node_modules/electron/dist'),
        extraResources: resources,
        win: { ...buildConfig.win, signAndEditExecutable: false },
      },
      targets: Platform.WINDOWS.createTarget(['nsis'], Arch.x64),
      publish: 'never',
    });
  } finally {
    // The directory was created by this process inside the dedicated temporary directory.
    assert.equal(path.dirname(stage), path.resolve(os.tmpdir()));
    await fs.rm(stage, { recursive: true });
  }
}

async function sha256(file) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}

const archive = path.join(unpacked, 'resources/app.asar');
assert.equal(JSON.parse(extractFile(archive, 'package.json')).version, pkg.version);
let verifiedFiles = 0;
for (const folder of ['dist', 'electron']) {
  const sourceFolder = folder === 'dist' ? frontendDist : path.join(frontend, folder);
  for (const file of await fs.readdir(sourceFolder, { recursive: true, withFileTypes: true })) {
    if (!file.isFile()) continue;
    const source = path.join(file.parentPath, file.name);
    const key = path.join(folder, path.relative(sourceFolder, source));
    const actual = createHash('sha256').update(extractFile(archive, key)).digest('hex');
    assert.equal(actual, await sha256(source), key);
    verifiedFiles++;
  }
}
for (const resource of resources) {
  assert.equal(await sha256(path.join(unpacked, 'resources', resource.to)), await sha256(resource.from), resource.to);
}
const exe = await fs.open(path.join(unpacked, pkg.productName + '.exe'), 'r');
try {
  const dos = Buffer.alloc(64);
  await exe.read(dos, 0, dos.length, 0);
  assert.equal(dos.toString('ascii', 0, 2), 'MZ');
  const pe = Buffer.alloc(6);
  await exe.read(pe, 0, pe.length, dos.readUInt32LE(60));
  assert.equal(pe.toString('ascii', 0, 4), 'PE\0\0');
  assert.equal(pe.readUInt16LE(4), 0x8664);
} finally {
  await exe.close();
}
const setup = path.join(output, `AI2050-Setup-${pkg.version}-x64.exe`);
const setupHash = await sha256(setup);
const blockMap = setup + '.blockmap';
await fs.writeFile(path.join(output, 'SHA256SUMS.txt'),
  `${setupHash}  ${path.basename(setup)}\n${await sha256(blockMap)}  ${path.basename(blockMap)}\n`);
const report = {
  schema: 'package_integrity.v1', kind: 'engineering', passed: true, version: pkg.version,
  target: 'win-x64', setup, sha256: setupHash, bytes: (await fs.stat(setup)).size,
  archive_sha256: await sha256(archive), verified_files: verifiedFiles, verified_resources: resources.length,
  signed: false, created_at: new Date().toISOString(),
};
await fs.mkdir(results, { recursive: true });
await fs.writeFile(path.join(results, 'package-verification.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
