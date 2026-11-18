"""Verify install-version routing and authenticated research data in a disposable center."""

import hashlib
import json
import os
import sys
import tarfile

from package_kv_release import CENTER, OUT, ROOT, UPLOADS, VERSION, WORK

sys.path[:0]=[str(ROOT),str(ROOT/'deploy')]
from deploy.research_ingest_bundle import verified_objects
from deploy.storage_policy import checked_dir,check_write


def app():
    probe=checked_dir(WORK/'center-test',2**20,operation='隔离中心接口验证')
    os.environ.update(AI2050_DIST_DIR=str(probe/'data'),AI2050_FRONTEND_DIST=str(WORK/'dist'),
                      AI2050_RESEARCH_DIR=str(probe/'empty-catalog'),AI2050_SNAPSHOT_DISABLE='1',
                      AI2050_LOCAL_TOOLS='0',AI2050_DEV='1',AI2050_DIST_ADMIN_TOKEN='local-release-verification-only',
                      TEMP=str(probe),TMP=str(probe),TMPDIR=str(probe))
    from deploy.center_app import create_app
    return create_app()


def body(response,code=200):
    assert response.status_code==code,(response.status_code,response.text[:160])
    return response.json()


def probe(client):
    api='/api/research-library'
    health=body(client.get('/api/health'))
    assert health['version']==VERSION
    assert client.get('/workbench').content==(WORK/'dist/index.html').read_bytes()
    archive=index_count=0
    checks=[]
    with tarfile.open(OUT/CENTER,'r:gz') as t:
        p=f'ai2050-center-{VERSION}/'
        manifest=json.load(t.extractfile(p+'RELEASE_MANIFEST.json'))
        for name,sha in manifest['files'].items():
            if name.startswith('frontend/dist/'):
                res=client.get('/'+name.removeprefix('frontend/dist/'))
                assert res.status_code==200 and hashlib.sha256(res.content).hexdigest()==sha,name
                archive+=1
    headers={'X-Admin-Token':os.environ['AI2050_DIST_ADMIN_TOKEN'],'Content-Type':'application/json'}
    for filename in UPLOADS:
        with tarfile.open(OUT/filename,'r:gz') as package:
            spec=json.load(package.extractfile('MANIFEST.json')); items=list(verified_objects(package,spec))
        if index_count==0:
            first=client.post(api+'/ingest',content=items[0][1],headers={'Content-Type':'application/json'})
            assert first.status_code in (401,403),first.status_code
        for round_ in range(2):
            for sha,raw in items:
                response=body(client.post(api+'/ingest',content=raw,headers=headers))
                assert response['sha256']==sha
                if round_==0:assert response.get('new',True),response
                artifact=body(client.get(api+'/artifacts/'+sha))
                assert artifact==json.loads(raw)
        study,=spec['studies']
        trace=body(client.get('/api/traces/research-space',params={'run_id':study['run_id']}))
        found,=[r for r in trace['studies'] if r['run_id']==study['run_id']]
        for key in ('report_sha256','binding_sha256','workpackage_sha256','model_snapshot'):
            assert found[key]==study[key],key
        index_count+=1; checks.append(dict(run=study['run_id'],objects=len(items),sha=study['report_sha256']))
    assert client.get(api+'/artifacts/'+'0'*64).status_code==404
    return dict(health=health,web_http_files=archive,studies=checks,
                unauthorized_ingest_rejected=True,duplicate_ingest_verified=True,missing_object_returns_404=True)


def verify():
    from fastapi.testclient import TestClient
    report=probe(TestClient(app()))
    out=ROOT/'tests/glm5/result/deployment_20261117/local-api.json'
    check_write(out,2**14).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--serve']:
        import uvicorn
        uvicorn.run(app(),host='127.0.0.1',port=5048,access_log=False)
    else:
        verify()
