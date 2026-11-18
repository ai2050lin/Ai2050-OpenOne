const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '../..');
const asar = require(path.join(root, 'frontend/node_modules/@electron/asar'));
const directory = path.join(root, 'release/desktop-1.0.10/win-unpacked');
const archive = path.join(directory, 'resources/app.asar');
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const pkg = JSON.parse(asar.extractFile(archive, 'package.json'));
if (pkg.version !== '1.0.10') throw Error('version mismatch');
function files(dir) {return fs.readdirSync(dir, {withFileTypes:true}).flatMap(e => e.isDirectory()?files(path.join(dir,e.name)):[path.join(dir,e.name)]);}
let count=0;
for(const folder of ['dist','electron'])for(const file of files(path.join(root,'frontend',folder))){
  const rel=path.relative(path.join(root,'frontend'),file).replaceAll('\\','/');
  if(/\.test\.[cm]?js$/.test(rel))continue;
  if(sha(asar.extractFile(archive,path.normalize(rel)))!==sha(fs.readFileSync(file)))throw Error('payload mismatch '+rel);
  count++;
}
const cfg=JSON.parse(fs.readFileSync(path.join(root,'frontend/package.json'))).build;
for(const r of cfg.extraResources)if(sha(fs.readFileSync(path.resolve(root,'frontend',r.from)))!==sha(fs.readFileSync(path.join(directory,'resources',r.to))))throw Error('resource mismatch '+r.to);
const report={version:pkg.version,verified_payload_files:count,verified_node_resources:cfg.extraResources.length,
  asar_sha256:sha(fs.readFileSync(archive)),installation_wizard_run:false,source_payload_equal:true};
fs.writeFileSync(path.join(root,'tests/glm5/result/mlp_tail_acceptance/desktop-payload.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
