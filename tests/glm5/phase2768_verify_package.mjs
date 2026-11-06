/* Verify shipped bytes, package version, Windows target and SHA256. No installation. */
import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { extractFile } from '../../frontend/node_modules/@electron/asar/lib/asar.js';

const root=fileURLToPath(new URL('../../',import.meta.url));
const out=path.join(root,'frontend/dist-installer/1.0.2');
const app=path.join(out,'win-unpacked'),archive=path.join(app,'resources/app.asar');
const setup=path.join(out,'AI2050-Setup-1.0.2-x64.exe');
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
async function hashFile(file){const h=createHash('sha256');for await(const chunk of createReadStream(file))h.update(chunk);return h.digest('hex')}
const pkg=JSON.parse(extractFile(archive,'package.json'));
assert.equal(pkg.version,'1.0.2');
const checks=[];
for(const name of ['main.js','preload.js','nodeSupervisor.js','runtimeInstaller.js','aiAssistant.js']){
 const actual=extractFile(archive,'electron/'+name),source=await fs.readFile(path.join(root,'frontend/electron',name));
 assert.equal(digest(actual),digest(source));checks.push({path:'electron/'+name,sha256:digest(actual)});
}
async function walk(dir){const result=[];for(const item of await fs.readdir(dir,{withFileTypes:true})){const file=path.join(dir,item.name);if(item.isDirectory())result.push(...await walk(file));else result.push(file)}return result}
const frontend=path.join(root,'frontend');let frontendFiles=0;
for(const file of await walk(path.join(frontend,'dist'))){const name=path.relative(frontend,file);assert.equal(digest(extractFile(archive,name)),await hashFile(file));frontendFiles++}
const resources={
 'parameter_reader.py':'deploy/parameter_reader.py','node_agent.py':'deploy/node_agent.py','node_service.py':'deploy/node_service.py',
 'goodfire_runner.py':'shared/dist_protocol/goodfire_runner.py','goodfire_probe.py':'shared/dist_protocol/goodfire_probe.py',
 'runtime/requirements-win-cu130.lock':'deploy/runtime/requirements-win-cu130.lock',
};
for(const [name,source] of Object.entries(resources)){
 const sha=await hashFile(path.join(app,'resources/node',name));assert.equal(sha,await hashFile(path.join(root,source)));checks.push({path:'resources/node/'+name,sha256:sha});
}
const handle=await fs.open(path.join(app,'AI2050 语言机制研究平台.exe'),'r');
try{const dos=Buffer.alloc(64);await handle.read(dos,0,64,0);assert.equal(dos.toString('ascii',0,2),'MZ');const pe=Buffer.alloc(6);await handle.read(pe,0,6,dos.readUInt32LE(60));assert.equal(pe.toString('ascii',0,4),'PE\0\0');assert.equal(pe.readUInt16LE(4),0x8664)}finally{await handle.close()}
const setupSha=await hashFile(setup),blockmap=setup+'.blockmap',blockmapSha=await hashFile(blockmap);
await fs.writeFile(path.join(out,'SHA256SUMS.txt'),`${setupSha}  ${path.basename(setup)}\n${blockmapSha}  ${path.basename(blockmap)}\n`);
const report={kind:'package_integrity_verification',version:pkg.version,target:'win-x64',passed:true,installer:setup,bytes:(await fs.stat(setup)).size,sha256:setupSha,frontend_files_verified:frontendFiles,files:checks,installer_executed:false};
await fs.writeFile(path.join(root,'tests/glm5/result/phase2768_knowledge_loop/package-verification.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify({version:report.version,target:report.target,bytes:report.bytes,frontendFiles,resourceFiles:Object.keys(resources).length,passed:true}));
