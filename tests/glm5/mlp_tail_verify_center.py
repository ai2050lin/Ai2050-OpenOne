"""Inspect final released bytes and serve deployed (not raw node) data on CPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
sys.stdout.reconfigure(encoding='utf-8')

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'tests/glm5/result/mlp_tail_acceptance'
VERSION=(ROOT/'VERSION').read_text().strip();NAME='ai2050-center-'+VERSION
archive=ROOT/'release'/f'{NAME}.tar.gz';target=OUT/'package-smoke';target.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:
    man=json.load(tar.extractfile(NAME+'/RELEASE_MANIFEST.json'))
    assert man['version']==VERSION
    for name,digest in man['files'].items():
        raw=tar.extractfile(NAME+'/'+name).read();assert hashlib.sha256(raw).hexdigest()==digest,name
        assert (ROOT/name).read_bytes()==raw,'stale packaged source '+name
    assert all(m.isfile() and (target/m.name).resolve().is_relative_to(target.resolve()) for m in tar.getmembers())
    tar.extractall(target,filter='data')
package=target/NAME
data=ROOT/'tests/glm5/result/mlp_tail_final_pack_acceptance/center'
env={**os.environ,'PYTHONPATH':str(package),'AI2050_SNAPSHOT_DISABLE':'1','AI2050_DIST_DIR':str(data),
     'AI2050_MLP_TAIL_DIR':str(data/'mlp_tail/MLP-TAIL-GOODFIRE-20261108-v1'),
     'AI2050_RESEARCH_DIR':str(OUT/'empty-research'),'AI2050_LOCAL_TOOLS':'0','AI2050_DEV':'1',
     'AI2050_FRONTEND_DIST':str(package/'frontend/dist'),'TEMP':str(OUT),'TMP':str(OUT)}
env['AI2050_DIST_ADMIN_TOKEN']='isolated-tail-package-check-only'
code='''
import json
from fastapi.testclient import TestClient
import center_app
with TestClient(center_app.create_app()) as client:
    health=client.get('/api/health').json()
    base='/api/research/mlp-tail'
    report=client.get(base).json()
    assert report['n_units']==38912
    assert report['admissibility']['scientific_admissibility']=='exploratory_observation_only'
    for b in range(32,36):
        r=client.get(f'{base}/layers/{b}');assert r.status_code==200
        assert len(r.json()['arrays']['A_std'])==9728
    r=client.get(base+'/units/35/9727');assert r.status_code==200
    assert len(r.json()['weights']['down'])==2560 and len(r.json()['examples'])==8
    r=client.get(base+'/relations/34/1244?kind=causal&factor=2');assert r.status_code==200
    assert r.json()['n_valid']==96 and len(r.json()['values'])==9728
    assert client.get('/api/models/local').status_code==403
    assert client.get('/api/autotest/queue').status_code==403
    assert client.get('/workbench').status_code==200
    print(json.dumps({'health':health,'units':38912,'native_vectors':True,'causal_paired_n':96,'local_tools_guarded':True}))
'''
r=subprocess.run([sys.executable,'-c',code],cwd=package/'deploy',env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
(OUT/'center-package-smoke.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
assert r.returncode==0,r.stdout+r.stderr
summary={'version':VERSION,'members_verified':len(man['files']),'all_source_bytes_equal':True,'packaged_api_smoke_passed':True,'deployed_display_data_used':True,'stdout':r.stdout}
(OUT/'center-package.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(summary)
