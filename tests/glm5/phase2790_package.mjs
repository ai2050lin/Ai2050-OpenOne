/* Offline packaging using installed Electron and builder's built-in traversal collector.
 * Main process has zero production npm dependencies; Vite bundles browser code.
 * Staging isolates package-manager metadata from the source project.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {build,Platform,Arch} from '../../frontend/node_modules/electron-builder/out/index.js';
const root=fileURLToPath(new URL('../../',import.meta.url));
const front=path.join(root,'frontend'),stage=path.join(root,'tests/glm5_temp/phase2790-app');
const pkg=JSON.parse(await fs.readFile(path.join(front,'package.json'),'utf8'));
if(Object.keys(pkg.dependencies||{}).length)throw Error('Staging strategy requires zero production npm dependencies');
await fs.mkdir(stage,{recursive:true});
await fs.cp(path.join(front,'dist'),path.join(stage,'dist'),{recursive:true});
await fs.cp(path.join(front,'electron'),path.join(stage,'electron'),{recursive:true});
const config={...pkg.build,npmRebuild:false,electronVersion:pkg.devDependencies.electron.replace('^',''),
 electronDist:path.join(front,'node_modules/electron/dist'),
 directories:{output:path.join(front,'dist-installer/1.0.4'),buildResources:path.join(front,'electron')},
 extraResources:pkg.build.extraResources.map(x=>({...x,from:path.resolve(front,x.from)})),
 win:{...pkg.build.win,signAndEditExecutable:false}};
const {build:unused,devDependencies,...runtime}=pkg;
await fs.writeFile(path.join(stage,'package.json'),JSON.stringify({...runtime,packageManager:'traversal@1'},null,2));
process.env.CSC_IDENTITY_AUTO_DISCOVERY='false';
process.env.ELECTRON_BUILDER_CACHE='C:/Users/Admin/AppData/Local/electron-builder/Cache';
await build({projectDir:stage,config,targets:Platform.WINDOWS.createTarget(['nsis'],Arch.x64),publish:'never'});
