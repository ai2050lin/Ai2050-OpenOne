// Offline Windows client. Build exclusively from the locally installed Electron runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {build,Platform,Arch} from '../../frontend/node_modules/electron-builder/out/index.js';
import {extractFile} from '../../frontend/node_modules/@electron/asar/lib/asar.js';
import storage from '../../frontend/electron/storagePolicy.js';

const root=fileURLToPath(new URL('../../',import.meta.url)), front=path.join(root,'frontend');
const overview=process.argv.includes('--overview');
const folder=process.argv.includes('--compact')?'workbench_compact_rail':process.argv.includes('--polish')?'workbench_polish_20261120':overview?'client_overview_20261120':'test_records_20261119';
const proof=path.join(root,'tests/glm5/result',folder);
const temp=path.join(root,'tests/glm5_temp',folder);
storage.checkWrite(temp,3*2**30,'测试记录客户端打包');
const accepted=JSON.parse(await fs.readFile(path.join(proof,overview?'overview-acceptance.json':'history-acceptance.json'),'utf8'));
assert.equal(accepted.passed,true);
const stage=await fs.mkdtemp(path.join(temp,'desktop-build-'));
const pkg=JSON.parse(await fs.readFile(path.join(front,'package.json'),'utf8'));
const {build:config,devDependencies,...metadata}=pkg;
const output=path.join(overview?proof:temp,'desktop');
storage.checkWrite(output,2*2**30,'桌面客户端产物');
process.env.TEMP=stage;process.env.TMP=stage;process.env.TMPDIR=stage;
process.env.ELECTRON_BUILDER_CACHE=path.join(temp,'builder-cache');
process.env.CSC_IDENTITY_AUTO_DISCOVERY='false';
try {
  await fs.cp(path.join(front,'dist'),path.join(stage,'dist'),{recursive:true});
  await fs.cp(path.join(front,'electron'),path.join(stage,'electron'),{recursive:true});
  await fs.writeFile(path.join(stage,'package.json'),JSON.stringify({...metadata,packageManager:'traversal@1'}));
  const extra=config.extraResources.map(r=>({...r,from:path.resolve(front,r.from)}));
  await build({projectDir:stage,publish:'never',targets:Platform.WINDOWS.createTarget(['dir'],Arch.x64),config:{
    ...config,directories:{output,buildResources:path.join(front,'electron')},npmRebuild:false,
    electronVersion:devDependencies.electron.replace('^',''),electronDist:path.join(front,'node_modules/electron/dist'),
    extraResources:extra,win:{...config.win,signAndEditExecutable:false}}});
  const resources=path.join(output,'win-unpacked/resources'), archive=path.join(resources,'app.asar');
  assert.equal(JSON.parse(extractFile(archive,'package.json')).version,pkg.version);
  let count=0;
  for(const folder of ['electron','dist']){
    for(const file of await fs.readdir(path.join(front,folder),{recursive:true,withFileTypes:true})){
      if(!file.isFile())continue;
      const name=path.join(file.parentPath,file.name);
      assert.ok(extractFile(archive,path.relative(front,name)).equals(await fs.readFile(name)),name);
      count++;
    }
  }
  for(const r of extra)assert.ok((await fs.readFile(r.from)).equals(await fs.readFile(path.join(resources,r.to))));
  await fs.writeFile(path.join(proof,'desktop.json'),JSON.stringify({passed:true,version:pkg.version,
    files:count,resources:extra.length,asar_sha256:createHash('sha256').update(await fs.readFile(archive)).digest('hex'),signed:false}));
}finally{
  const resolved=await fs.realpath(stage);
  assert.equal(path.dirname(resolved),await fs.realpath(temp));assert.ok(path.basename(resolved).startsWith('desktop-build-'));
  await fs.rm(resolved,{recursive:true,force:true});
}

