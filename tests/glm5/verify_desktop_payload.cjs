const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '../..');
const asar = require(path.join(root, 'frontend/node_modules/@electron/asar'));
const directory = path.join(root, 'release/desktop-1.0.8/win-unpacked');
const archive = path.join(directory, 'resources/app.asar');
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
const pkg = JSON.parse(asar.extractFile(archive, 'package.json'));
if (pkg.version !== '1.0.8') throw new Error('desktop version differs');
function files(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const file = path.join(dir, entry.name);
    return entry.isDirectory() ? files(file) : [file];
  });
}
const verified = [];
for (const folder of ['dist', 'electron']) {
  for (const file of files(path.join(root, 'frontend', folder))) {
    const rel = path.relative(path.join(root, 'frontend'), file).replaceAll('\\', '/');
    // Tests are intentionally excluded by electron-builder's default filters.
    if (/\.test\.[cm]?js$/.test(rel)) continue;
    const payload = asar.extractFile(archive, path.normalize(rel));
    if (sha(payload) !== sha(fs.readFileSync(file))) throw new Error('payload differs: ' + rel);
    verified.push(rel);
  }
}
const cfg = JSON.parse(fs.readFileSync(path.join(root, 'frontend/package.json'))).build;
for (const resource of cfg.extraResources) {
  const source = path.resolve(root, 'frontend', resource.from);
  const actual = path.join(directory, 'resources', resource.to);
  if (sha(fs.readFileSync(source)) !== sha(fs.readFileSync(actual))) throw new Error('node resource differs: ' + resource.to);
}
const report = { version: pkg.version, archive_sha256: sha(fs.readFileSync(archive)),
  verified_payload_files: verified.length, verified_node_resources: cfg.extraResources.length,
  installation_wizard_run: false, source_payload_equal: true };
fs.writeFileSync(path.join(root, 'tests/glm5/result/deployment_bundle_20261108/desktop-payload.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
