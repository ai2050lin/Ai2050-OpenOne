"""Exercise the center using existing real artifacts in an isolated store, with no GPU."""
import argparse
import base64
from contextlib import ExitStack
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import checked_dir, process_env
from shared.dist_protocol.test_archive import CHUNK_BYTES, digest

SNAP = '1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce'
RUN = 'record-a641dea2-e60b-4262-bd38-6d3aba1bd868'
PATH = '/api/research/workspace'
PRIVATE = PATH + '/private'
DATA = ROOT / 'tests/glm5/result/live_original_14b_v4'


def unpack(value, index):
    native = base64.b64decode(value)
    return struct.unpack('<f', b'\0\0' + native[index*2:index*2+2])[0]


def stage_archive(store):
    package = DATA / 'live-original-package'
    submission = json.loads((package / 'submission.json').read_bytes())
    plan = json.loads((DATA / 'live-original-plan.json').read_bytes())
    if 'plan' in plan:
        plan = plan['plan']
    actor = 'isolated-archive-qa'
    preflight = store.preflight(plan, actor)
    store.start({'run_id':RUN, 'preflight_id':preflight['preflight_id'], 'decision':'new_acquisition',
                 'reason':'Restore existing acquisition for isolated engineering QA'}, actor)
    for artifact in submission['final']['artifacts']:
        raw = (package / artifact['name']).read_bytes()
        upload = store.upload_init({'sha256':artifact['sha256'],'size':len(raw)}, actor)
        if not upload['available']:
            token = upload['upload_id']
            for i in range(0,len(raw),CHUNK_BYTES):
                store.upload_chunk(token,i,raw[i:i+CHUNK_BYTES],actor)
            store.upload_finish(token,actor)
    store.finalize(RUN,submission['final'],actor)
    receipt = store.get(RUN)['receipt']
    assert receipt['manifest_sha256'] == submission['manifest_sha256']
    return receipt


def check_http(client, auth, member, receipt):
    def get(path, status=200, header=None):
        response = client.get(path, headers=header)
        assert response.status_code == status, (path,response.status_code,response.text[:500])
        return response.json()

    catalog = get(PATH+'/catalog')
    target = next(model for model in catalog['models'] if model['model_snapshot']==SNAP)
    assert [target[k] for k in ('n_layers','n_heads','n_kv_heads','head_dim','d_ff')]==[40,40,8,128,17408]
    arch = get(PATH+f'/models/{SNAP}/architecture')
    assert arch['model_snapshot']==SNAP and arch['n_heads']==40 and arch['d_ff']==17408
    assert catalog['revision']==digest({k:v for k,v in catalog.items() if k!='revision'})
    assert 'b26' not in {a['id'] for a in catalog['archives']}
    get(PATH+'/archives/b26',404)
    get(PRIVATE+'/catalog',401)
    get(PRIVATE+'/archives/b26',403,member)
    assert 'b26' in {a['id'] for a in get(PRIVATE+'/catalog',header=auth)['archives']}
    original = get(PATH+'/archives/head/blocks/26?row=confirm-000-02')
    get(PATH+'/archives/head?revision='+'0'*64,409)
    assert original['position']==190 and original['tokenId']==510 and original['partialHead']==36
    assert len(base64.b64decode(original['fields']['head_z']))==128*2
    assert unpack(original['fields']['head_z'],0)==-1.8671875
    row = get(PRIVATE+'/archives/b26',header=auth)
    assert row['caseId']=='confirm-000-02' and row['run']==RUN and row['receiptSha256']==receipt['receipt_sha256']
    positions = [item['position'] for item in row['rows']]
    assert positions==[0,95,190]
    tail = get(PRIVATE+'/archives/b26/blocks/26?row=190',header=auth)
    mid = get(PRIVATE+'/archives/b26/blocks/26?row=95',header=auth)
    assert tail['tokenId']==510 and tail['snapshot']==SNAP
    assert len(base64.b64decode(tail['fields']['head_z']))==40*128*2
    assert len(base64.b64decode(tail['fields']['mlp_activation']))==17408*2
    assert unpack(tail['fields']['head_z'],36*128)==-1.859375
    assert unpack(tail['fields']['mlp_activation'],17407)==0.0299072265625
    assert unpack(mid['fields']['mlp_activation'],17407)==0.412109375
    get(PRIVATE+'/archives/b26/blocks/25?row=95',404,auth)
    assert client.get(PRIVATE+'/archives/b26',headers=auth).headers['cache-control']=='private, no-store'
    pilot = get(PATH+'/resources/goodfire-records')
    assert pilot['records']
    get(PATH+'/resources/dictionary')
    from deploy.research_library import library
    from deploy.datapack import research_entries
    lib = library()
    public = lib.ingest({'title':'Visible engineering check'},'isolated-qa')['sha256']
    hidden = lib.ingest({'title':'Unpublished engineering check'},'isolated-qa')['sha256']
    with lib.db() as db:
        db.execute('INSERT INTO mainline_publications VALUES(?,0)',(hidden,))
    get('/api/research-library/artifacts/'+public)
    get('/api/research-library/artifacts/'+hidden,404)
    entries, _ = research_entries()
    assert '/api/research-library/artifacts/'+public in entries
    assert '/api/research-library/artifacts/'+hidden not in entries
    return {'catalog_revision':catalog['revision'],'model_snapshot':SNAP,'heads':40,'mlp':17408,
            'positions':positions,'original_D0':-1.8671875,'rerun_D0':-1.859375,'pilot_records':len(pilot['records']),
            'archive_manifest_sha256':receipt['manifest_sha256'],'private_acl_and_adjacent_missing':True,
            'public_visibility_shared_by_api_and_pack':True}


def check_snapshot_guard(base):
    from fastapi import HTTPException
    from deploy import parameter_reader
    from server import live_analysis_service as live

    class Unusable:
        def __getattr__(self, _):
            raise AssertionError('Model was used before snapshot validation')

    with patch.object(live,'model_dir',return_value=base), patch.object(live,'_read_meta',return_value={}), \
         patch.object(parameter_reader,'model_snapshot',return_value={'sha256':SNAP}):
        live.live_load(live.LiveLoad(expected_snapshot='b'*64))
        live._THREAD.join(timeout=5)
        assert live._state['status']=='error' and live._state['model'] is None
        with patch.dict(live._state,{'status':'ready','dir':str(base),'model_snapshot':SNAP,'model':Unusable()}):
            try:
                live.live_run(live.LiveRun(text='snapshot guard test',expected_snapshot='b'*64))
            except HTTPException as error:
                assert error.status_code==409
            else:
                raise AssertionError('Unverified runtime accepted the request')


def check_evidence(client, admin, member):
    endpoint = PATH+'/evidence'
    restricted = PRIVATE+'/evidence'
    case = {'q':'confirm-000-02','source':'case','target':'B26/H36','position':190,'token_id':510,
            'snapshot':SNAP,'coverage':'native'}
    public_response = client.get(endpoint,params=case)
    assert public_response.status_code==200, public_response.text[:500]
    public = public_response.json()
    assert public['total']==1
    assert public['items'][0]['run_id']=='clt_b26_three_cues_v3'
    assert public['items'][0]['readouts'][0]['row']=='confirm-000-02'
    assert 'history' not in public['scope_totals'] and RUN not in public_response.text
    assert client.get(restricted).status_code==401
    assert client.get(restricted,headers=member).status_code==403
    response = client.get(restricted,params=case,headers=admin)
    assert response.status_code==200,response.text[:500]
    assert response.headers['cache-control']=='private, no-store'
    private = response.json()
    assert private['total']==2
    exact = next(row for row in private['items'] if row['run_id']==RUN)
    assert exact['output']=='silver' and exact['case_sha256'] and exact['review']=='unreviewed'
    assert exact['source_case']=={'run':'clt_b26_three_cues_v3','case':'confirm-000-02'}
    assert exact['readouts'][0]['archive']=='b26'
    assert private['scope_totals']['history']==4861 and private['scope_totals']['study']==37
    assert private['historical_links']==22
    assert private['coverage_counts']['reviewed']=={}
    assert client.get(restricted,params={**case,'token_id':509},headers=admin).json()['total']==0
    assert client.get(restricted,params={**case,'target':'B25/H36'},headers=admin).json()['total']==0
    fail = client.get(endpoint,params={'q':'gf-2961','source':'study'}).json()['items'][0]
    assert fail['id']=='gf-2961' and fail['earlier']=='gf-2959' and fail['novelty'] and fail['gate'] and fail['counter']
    negatives = client.get(endpoint,params={'source':'case','q':'clt_b26_competition_v1','counter':True}).json()
    assert negatives['total']==308 and not any(row.get('readouts') for row in negatives['items'])
    original = client.get(endpoint,params={'source':'case','q':'confirm-000-02','coverage':'native'}).json()
    assert original['total']==1 and original['items'][0].get('prompt')
    study = client.get(endpoint,params={'q':'study-archive-source-selection','source':'study'}).json()['items'][0]
    assert sorted(study['linked_runs'])==['clt_b26_competition_v1','clt_b26_three_cues_v3']
    assert 'native' in study['coverage']
    return {'public_totals':public['scope_totals'],'authorized_totals':private['scope_totals'],
            'coverage':private['coverage_counts'],'negative_cases_2990':negatives['total'],
            'exact_historical_links':private['historical_links'],
            'origins_distinguished':True,'position_and_token_exact':True,
            'archive_acl':True,'failed_behavior_and_retest_delta_retained':True}


def browser(app, work, out, password, script=None):
    import uvicorn
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        host = f'http://127.0.0.1:{listener.getsockname()[1]}'
        server = uvicorn.Server(uvicorn.Config(app,log_level='warning',lifespan='off'))
        thread = threading.Thread(target=server.run, kwargs={'sockets':[listener]}, daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.05)
            env = process_env(work)
            env['AI2050_WORKSPACE_QA_PASSWORD'] = password
            subprocess.run(['node',str(script or Path(__file__).with_suffix('.cjs')),host,str(work),str(out)],
                           env=env,check=True,timeout=110)
        finally:
            server.should_exit = True
            thread.join(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist',type=Path)
    parser.add_argument('--evidence',action='store_true')
    args = parser.parse_args()
    staging = checked_dir(ROOT/'tests/glm5_temp'/('workspace-evidence' if args.evidence else 'workspace-center'),300*2**20)
    out = checked_dir(ROOT/'tests/glm5/result'/('workspace_evidence' if args.evidence else 'workspace_center_cutover'),20*2**20)
    with tempfile.TemporaryDirectory(dir=staging,prefix='acceptance-',ignore_cleanup_errors=True) as directory, ExitStack() as stack:
        work = Path(directory)
        password = secrets.token_urlsafe(20)
        env = {'AI2050_DIST_DIR':str(work/'isolated-center'), 'AI2050_RESEARCH_DIR':str(work/'research'),
               'AI2050_WORKSPACE_DIR':str(work/'publish'), 'AI2050_DIST_ADMIN_TOKEN':secrets.token_hex(20),
               'AI2050_SNAPSHOT_DISABLE':'1', 'AI2050_LOCAL_TOOLS':'0'}
        if args.dist:
            env['AI2050_FRONTEND_DIST'] = str(args.dist.resolve())
        stack.enter_context(patch.dict(os.environ,env))
        from deploy.center_app import create_app
        from deploy.test_archive_service import archive, services
        from fastapi.testclient import TestClient
        from shared.dist_protocol import accounts

        app = create_app()
        receipt = stage_archive(archive())
        db = services()._db()
        stack.callback(db.close)
        accounts.create_user(db,'workspace-qa',password)
        accounts.create_user(db,'workspace-member',password,role='member',by_admin=True)
        admin = accounts.login(db,'workspace-qa',password)
        member = accounts.login(db,'workspace-member',password)
        with TestClient(app) as client:
            admin_auth = {'Authorization':'Bearer '+admin['token']}
            member_auth = {'Authorization':'Bearer '+member['token']}
            result = check_http(client,admin_auth,member_auth,receipt)
            if args.evidence:
                result['evidence_search'] = check_evidence(client,admin_auth,member_auth)
        check_snapshot_guard(work)
        result['runtime_snapshot_mismatch_rejected_before_model'] = True
        (out/'center-api.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print('Center API and no-weight snapshot guard checks passed.',flush=True)
        if args.dist:
            (work/'publish').mkdir()
            browser(app,work,out,password,Path(__file__).with_name('workspace_evidence_acceptance.cjs') if args.evidence else None)


if __name__=='__main__':
    main()
