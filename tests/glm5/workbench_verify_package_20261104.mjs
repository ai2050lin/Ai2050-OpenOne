/* Verify installer architecture, every built frontend file and shipped local node sources. */
import fs from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {extractFile} from '../../frontend/node_modules/@electron/asar/lib/asar.js';
const root=fileURLToPath(new URL('../../',import.meta.url)),front=path.join(root,'frontend');
const pkg=JSON.parse(await fs.readFile(path.join(front,'package.json'),'utf8')),out=path.join(front,'dist-installer',pkg.version),app=path.join(out,'win-unpacked');
const archive=path.join(app,'resources/app.asar'),setup=path.join(out,`AI2050-Setup-${pkg.version}-x64.exe`);
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
async function hashFile(file){const h=createHash('sha256');for await(const chunk of createReadStream(file))h.update(chunk);return h.digest('hex')}
assert.equal(JSON.parse(extractFile(archive,'package.json')).version,pkg.version);
const checks=[];
for(const name of ['main.js','preload.js','nodeSupervisor.js','runtimeInstaller.js','aiAssistant.js']){const bytes=extractFile(archive,'electron/'+name);assert.equal(hash(bytes),await hashFile(path.join(front,'electron',name)));checks.push({path:'electron/'+name,sha256:hash(bytes)})}
async function walk(dir){const result=[];for(const e of await fs.readdir(dir,{withFileTypes:true})){const f=path.join(dir,e.name);if(e.isDirectory())result.push(...await walk(f));else result.push(f)}return result}
let frontendFiles=0;for(const f of await walk(path.join(front,'dist'))){assert.equal(hash(extractFile(archive,path.relative(front,f))),await hashFile(f));frontendFiles++}
for(const item of pkg.build.extraResources){const target=path.join(app,'resources',item.to);const sha=await hashFile(target);assert.equal(sha,await hashFile(path.resolve(front,item.from)));checks.push({path:'resources/'+item.to,sha256:sha})}
const handle=await fs.open(path.join(app,pkg.productName+'.exe'),'r');
try{const dos=Buffer.alloc(64);await handle.read(dos,0,64,0);assert.equal(dos.toString('ascii',0,2),'MZ');const pe=Buffer.alloc(6);await handle.read(pe,0,6,dos.readUInt32LE(60));assert.equal(pe.toString('ascii',0,4),'PE\0\0');assert.equal(pe.readUInt16LE(4),0x8664)}finally{await handle.close()}
const sha256=await hashFile(setup),blockmap=setup+'.blockmap';
await fs.writeFile(path.join(out,'SHA256SUMS.txt'),`${sha256}  ${path.basename(setup)}\n${await hashFile(blockmap)}  ${path.basename(blockmap)}\n`);
const report={schema:'package_integrity.v1',kind:'engineering',passed:true,version:pkg.version,target:'win-x64',installer:setup,bytes:(await fs.stat(setup)).size,sha256,frontend_files_verified:frontendFiles,files:checks,installer_executed:false,signed:false,created_at:new Date().toISOString()};
await fs.writeFile(path.join(root,'tests/glm5/result/workbench_playwright_20261104/package-verification.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
