"""Verify the client's uploader against an isolated copy of the deployed 1.0.27 API."""
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from b35_recursive_release_check import ROOT, OUT


BASE = 'http://127.0.0.1:55139'
EXERCISE = """
import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {prepareResearchBundle,uploadResearchBundle} from './frontend/src/components/v8/researchBundleUpload.js';
const base='http://127.0.0.1:55139';
const bundle=await prepareResearchBundle(new Blob([await readFile('release/ai2050-research-qwen3-4b-b35.tar.gz')]));
const data=JSON.stringify({username:'upload-test',password:'isolated-example#73'});
const reg=await fetch(base+'/api/auth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:data});
assert.equal(reg.status,200);
const {token,user}=await reg.json();
assert.equal(user.role,'admin');
let writes=0;
let interrupted=true;
async function request(path,{method='GET',rawBody}={}) {
  if(method==='POST' && interrupted && writes===3) throw Error('simulated interruption');
  const response=await fetch(base+path,{method,headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},
    redirect:'error',body:rawBody});
  assert.ok(response.ok, path+' returned '+response.status);
  if(method==='POST') writes++;
  return response.json();
}
await assert.rejects(uploadResearchBundle(bundle,request,()=>{}),/simulated interruption/);
assert.equal(writes,3);
interrupted=false;
const resumed=await uploadResearchBundle(bundle,request,()=>{});
assert.equal(resumed.uploaded,97);
assert.equal((await uploadResearchBundle(bundle,request,()=>{})).uploaded,0);
const sha=resumed.studies[0].report_sha256;
const report=await request('/api/research-library/artifacts/'+sha);
assert.equal(report.d_ff,9728);
console.log(JSON.stringify({writes, resumed:resumed.uploaded, verified:resumed.total, report_sha256:sha,
  retries_skipped:true, feature_units:report.d_ff}));
"""


def main():
    with tempfile.TemporaryDirectory(prefix='bundle-protocol-smoke-',dir=ROOT/'tests/glm5_temp') as stage:
        folder = Path(stage)
        with tarfile.open(ROOT/'release/ai2050-center-1.0.27.tar.gz') as archive:
            archive.extractall(folder,filter='data')
        app = folder/'ai2050-center-1.0.27'
        env = {**os.environ,'AI2050_DIST_DIR':str(folder/'data'),'AI2050_NATIVE_DIR':str(folder/'native'),
               'AI2050_RESEARCH_DIR':str(folder/'research'),'AI2050_SNAPSHOT_DISABLE':'1','AI2050_LOCAL_TOOLS':'0',
               'AI2050_DIST_ADMIN_TOKEN':secrets.token_hex(24),'AI2050_REGISTRATION':'open','PYTHONUTF8':'1'}
        env.pop('PYTHONPATH',None)
        with (OUT/'bundle-protocol-server.log').open('w',encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable,'deploy/center_app.py','--host','127.0.0.1','--port','55139'],
                env=env,cwd=app,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                for _ in range(100):
                    try:
                        with urlopen(BASE+'/api/health',timeout=3) as res:
                            health = json.load(res)
                        break
                    except (URLError,TimeoutError):
                        if proc.poll() is not None:
                            raise
                        time.sleep(.25)
                else:
                    raise RuntimeError('Sandbox center did not start')
                assert health['version']=='1.0.27'
                result = subprocess.run(['node','--input-type=module','-e',EXERCISE],cwd=ROOT,
                                        capture_output=True,text=True,timeout=100)
                if result.returncode:
                    print(result.stderr,flush=True)
                    result.check_returncode()
                proof = json.loads(result.stdout)
                proof.update(health=health,endpoint=BASE,timestamp=datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds'))
                (OUT/'bundle-client-protocol-proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
                print(json.dumps(proof),flush=True)
            finally:
                proc.terminate()
                proc.wait(timeout=15)


if __name__ == '__main__':
    main()
