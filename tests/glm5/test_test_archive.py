"""Isolated engineering fixtures; never submit these as real research evidence."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from shared.dist_protocol.test_archive import Archive, CHUNK_BYTES, CORE_ROLES, Conflict, digest, validate_plan
from deploy import test_archive_service as api
from deploy.test_archive_client import Client, cleanup_scratch
from deploy.storage_policy import check_write

TEMP = check_write(ROOT / 'tests/glm5_temp/test_archive_20261119', 200 * 2**20)
TEMP.mkdir(parents=True, exist_ok=True)


def plan():
    return {'schema': 'research_test_plan.v1', 'kind': 'real', 'title': 'isolated engineering fixture',
            'objective': 'test the archive contract, no model', 'principle': 'controlled fixture',
            'hypothesis': 'identity and receipt remain consistent', 'alternatives': 'duplicate and missing objects',
            'criteria': 'all required fixture files read back', 'domains': ['KN', 'RS'], 'family_ids': ['KN-source'],
            'dataset': {'id': 'fixture', 'revision': 'v1', 'content_sha256': 'b'*64,
                        'split_manifest_sha256': 'c'*64, 'case_count': 1, 'exposure': 'engineering only'},
            'acquisition': {'model_id': 'fixture-no-model', 'model_snapshot': 'a'*64,
                            'tokenizer_sha256': 'd'*64, 'inputs_sha256': 'e'*64, 'code_sha256': 'f'*64,
                            'environment_sha256': '1'*64, 'precision': 'fixture', 'seed': 19,
                            'generation': {}, 'capture': {'layers': 'all', 'readouts': ['final'], 'boundary': 'residual'}},
            'analysis': {'method_version': 'fixture-v1', 'code_sha256': '2'*64, 'scoring_sha256': '3'*64,
                         'selection': {'split': 'discovery'}, 'required_artifact_roles': ['native_field']}}


class TestArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=TEMP)
        self.base = Path(self.tmp.name)
        self.store = Archive(self.base / 'archive')
        self.actor = 'center-admin'
        def auth(token):
            if token != 'engineering-token':
                raise HTTPException(401, 'not authorized')
        service = SimpleNamespace(_DATA_DIR=self.base, _auth_admin=auth,
                                  _auth_call=lambda *_: (_ for _ in ()).throw(HTTPException(401)),
                                  _accounts=SimpleNamespace(require_admin=None))
        self.patches = [patch.object(api, 'services', return_value=service), patch.object(api, 'archive', return_value=self.store)]
        for p in self.patches:p.start()
        app = FastAPI();app.include_router(api.router)
        self.web = TestClient(app)
        self.headers = {'X-Admin-Token': 'engineering-token'}

    def tearDown(self):
        self.web.close()
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def request(self, path, doc=None, method=None, raw=None):
        method = method or ('POST' if doc is not None else 'GET')
        kwargs = {'headers': self.headers}
        kwargs.update({'content': raw} if raw is not None else {'json': doc} if doc is not None else {})
        response = self.web.request(method, '/api/test-archive' + path, **kwargs)
        return response

    def start(self, run_id='run-1', spec=None, decision='new_acquisition', source=''):
        p = self.store.preflight(spec or plan(), self.actor)
        return self.store.start({'run_id': run_id, 'preflight_id': p['preflight_id'], 'decision': decision,
                                 'source_run_id': source, 'reason': 'engineering fixture'}, self.actor)

    def upload(self, data):
        sha = hashlib.sha256(data).hexdigest()
        up = self.store.upload_init({'sha256': sha, 'size': len(data)}, self.actor)
        if not up['available']:
            for i in range(0, len(data), CHUNK_BYTES):
                self.store.upload_chunk(up['upload_id'], i, data[i:i+CHUNK_BYTES], self.actor)
            self.store.upload_finish(up['upload_id'], self.actor)
        return sha

    def complete(self, run_id='run-1', native=True, spec=None):
        self.start(run_id, spec=spec)
        roles = CORE_ROLES | ({'native_field'} if native else set())
        artifacts = []
        for role in roles:
            data = ('engineering-fixture:' + role).encode()
            artifacts.append({'name': role + '.bin', 'role': role, 'retention': 'core' if role in CORE_ROLES else 'important',
                              'availability': 'uploaded', 'size': len(data), 'sha256': self.upload(data)})
        doc = {'status': 'completed', 'summary': 'engineering fixture only', 'artifacts': artifacts}
        return self.store.finalize(run_id, doc, self.actor)

    def test_structured_run_search_binds_filters_to_the_same_run(self):
        spec = plan()
        spec['acquisition']['capture']['targets'] = ['B26/H36', 'L27/D8']
        self.complete('declared-head', spec=spec)
        second = plan()
        second['acquisition']['capture']['targets'] = ['B26/H3']
        second['title'] = 'B26/H36 keyword only'
        self.complete('keyword-only', native=False, spec=second)
        query = ('/runs?target=B26%2FH36&snapshot=' + 'a'*64 +
                 '&domain=KN&method=fixture-v1&archive_state=server_verified&artifact_role=native_field')
        response = self.request(query)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual([row['id'] for row in data['items']], ['declared-head'])
        self.assertEqual(data['applied_filters']['target'], 'B26/H36')
        self.assertEqual(self.request('/runs?target=B26%2FH3&artifact_role=native_field').json()['total'], 0)
        self.assertEqual(self.request('/runs?target=B26%2FH036').status_code, 422)
        self.assertEqual(self.request('/runs?q=B26%2FH36').json()['total'], 2)

    def test_authentication_and_no_public_raw_data(self):
        self.assertEqual(self.web.get('/api/test-archive/capabilities').status_code, 200)
        for method, path in [('GET', '/runs'), ('POST', '/preflight'), ('GET', '/objects/' + 'a'*64)]:
            self.assertEqual(self.web.request(method, '/api/test-archive' + path, json={}).status_code, 401)
        self.assertFalse((self.base / 'objects').exists())

    def test_analysis_key_independent_from_acquisition(self):
        a = plan(); b = copy.deepcopy(a);b['analysis']['code_sha256'] = '4'*64
        self.assertEqual(validate_plan(a)[0], validate_plan(b)[0])
        self.assertNotEqual(validate_plan(a)[1], validate_plan(b)[1])
        b['acquisition']['precision'] = 'other'
        self.assertNotEqual(validate_plan(a)[0], validate_plan(b)[0])

    def test_metadata_validation_and_body_limit(self):
        p = plan();p['acquisition']['model_snapshot'] = 'short'
        self.assertEqual(self.request('/preflight', p).status_code, 422)
        self.assertEqual(self.request('/preflight', raw=b'x'*(513*1024), method='POST').status_code, 413)
        p=plan();p['kind']='synthetic'
        self.assertEqual(self.request('/preflight', p).status_code, 422)

    def test_query_is_correlated_and_case_insensitive(self):
        self.start()
        result = self.request('/runs?q=ENGINEERING&domain=KN&model=fixture-no-model').json()
        self.assertEqual(result['total'], 1)
        self.assertEqual(self.request('/runs?domain=GR').json()['total'], 0)
        self.assertEqual(self.request('/runs?limit=9999').status_code, 422)
        self.assertEqual(result['coverage'], 'registered_records_only')

    def test_duplicate_race_requires_explicit_replication(self):
        checks = [self.store.preflight(plan(), self.actor) for _ in range(2)]
        def launch(i):
            try:
                return self.store.start({'run_id': f'race-{i}', 'preflight_id': checks[i]['preflight_id'],
                                        'decision': 'new_acquisition', 'reason': 'fixture'}, self.actor)['id']
            except Conflict:return 'conflict'
        with ThreadPoolExecutor(2) as pool:
            result = list(pool.map(launch, range(2)))
        self.assertEqual(result.count('conflict'), 1)
        self.start('independent', decision='independent_replication')

    def test_run_identity_is_idempotent_not_overwritten(self):
        p=self.store.preflight(plan(), self.actor)
        doc={'run_id':'same','preflight_id':p['preflight_id'],'decision':'new_acquisition','reason':'fixture'}
        self.assertEqual(self.store.start(doc,self.actor)['id'], self.store.start(doc,self.actor)['id'])
        with self.assertRaises(Conflict):self.store.start({**doc,'reason':'different'},self.actor)
        with self.assertRaises(ValueError):self.store.start({**doc,'run_id':'other'},'other-actor')

    def test_reuse_data_and_keep_replication_distinct(self):
        self.complete()
        check=self.store.preflight(plan(),self.actor)
        self.assertEqual(check['matches'][0]['suggestion'],'reuse_results')
        changed=plan();changed['analysis']['method_version']='fixture-v2'
        check=self.store.preflight(changed,self.actor)
        self.assertEqual(check['matches'][0]['suggestion'],'reuse_acquisition')
        run=self.start('analysis-v2',changed,'analysis_only','run-1')
        self.assertEqual(run['source_run_id'],'run-1')
        self.assertEqual(run['decision'],'analysis_only')

    def test_retired_fields_require_explicit_recapture(self):
        self.complete(native=False)
        changed=plan();changed['analysis']['method_version']='v2'
        check=self.store.preflight(changed,self.actor)
        self.assertEqual(check['matches'][0]['suggestion'],'recapture_required')
        with self.assertRaises(Conflict):self.start('no-field',changed,'analysis_only','run-1')

    def test_chunk_resume_replay_and_finish_retry(self):
        data=b'a'*(CHUNK_BYTES+17);value=hashlib.sha256(data).hexdigest()
        up=self.store.upload_init({'sha256':value,'size':len(data)},self.actor);uid=up['upload_id']
        self.store.upload_chunk(uid,0,data[:CHUNK_BYTES],self.actor)
        restarted=Archive(self.store.root)
        self.assertEqual(restarted.upload_status(uid,self.actor)['offset'],CHUNK_BYTES)
        self.assertTrue(restarted.upload_chunk(uid,0,data[:CHUNK_BYTES],self.actor)['replayed'])
        with self.assertRaises(Conflict):restarted.upload_finish(uid,self.actor)
        restarted.upload_chunk(uid,CHUNK_BYTES,data[CHUNK_BYTES:],self.actor)
        self.assertTrue(restarted.upload_finish(uid,self.actor)['verified'])
        self.assertTrue(restarted.upload_finish(uid,self.actor)['replayed'])
        self.assertEqual(restarted.download(value).read_bytes(),data)

    def test_bad_digest_does_not_poison_retry(self):
        value=hashlib.sha256(b'good').hexdigest()
        up=self.store.upload_init({'sha256':value,'size':4},self.actor)
        self.store.upload_chunk(up['upload_id'],0,b'evil',self.actor)
        with self.assertRaises(ValueError):self.store.upload_finish(up['upload_id'],self.actor)
        self.assertNotEqual(self.store.upload_init({'sha256':value,'size':4},self.actor)['upload_id'],up['upload_id'])

    def test_chunk_limit_offset_and_path_traversal(self):
        up=self.request('/uploads',{'sha256':'1'*64,'size':CHUNK_BYTES+1}).json()
        uid=up['upload_id']
        self.assertEqual(self.request('/uploads/'+uid+'?offset=0',method='PUT',raw=b'x'*(CHUNK_BYTES+1)).status_code,413)
        self.assertEqual(self.request('/uploads/'+uid).json()['offset'],0)
        self.assertEqual(self.request('/uploads/'+uid+'?offset=1',method='PUT',raw=b'x').status_code,409)
        self.assertEqual(self.request('/objects/not-a-hash').status_code,422)

    def test_incomplete_final_preserves_failed_run_and_blocks_cleanup(self):
        self.start()
        result=self.store.finalize('run-1',{'status':'interrupted','summary':'fixture stopped','artifacts':[]},self.actor)
        self.assertEqual(result['state'],'interrupted')
        self.assertEqual(result['archive_state'],'missing_artifacts')
        self.assertFalse(self.store.receipt('run-1')['cleanup_allowed'])
        with self.assertRaises(Conflict):self.store.finalize('run-1',{'status':'completed','summary':'overwrite','artifacts':[]},self.actor)

    def test_important_missing_artifact_is_rejected(self):
        self.start()
        response=self.request('/runs/run-1/finalize',{'status':'completed','summary':'fixture','artifacts':[
            {'name':'field.bin','role':'native_field','retention':'important','availability':'retired_not_archived','reason':'not saved'}]})
        self.assertEqual(response.status_code,422)
        self.assertEqual(self.store.get('run-1')['state'],'running')

    def test_artifact_recovery_preserves_manifest_history_and_execution_status(self):
        self.start()
        first={'status':'failed','summary':'fixture missing file','artifacts':[]}
        self.store.finalize('run-1',first,self.actor)
        revised={**first,'summary':'recovered source record; still failed'}
        self.store.finalize('run-1',revised,self.actor,parent_sha=digest(first))
        self.assertEqual(len(self.store.revisions('run-1')['items']),2)
        self.assertEqual(self.store.get('run-1')['state'],'failed')
        with self.assertRaises(Conflict):self.store.finalize('run-1',{**revised,'status':'completed'},self.actor,parent_sha=digest(revised))
        with self.assertRaises(Conflict):self.store.finalize('run-1',{**revised,'summary':'stale'},self.actor,parent_sha=digest(first))

    def test_corrupt_or_missing_object_invalidates_receipt(self):
        run=self.complete();receipt=self.store.receipt('run-1')
        self.assertTrue(receipt['readback_verified']);self.assertTrue(receipt['cleanup_allowed'])
        artifact=run['final']['artifacts'][0]
        self.store.object_path(artifact['sha256']).write_bytes(b'x'*artifact['size'])
        receipt=self.store.receipt('run-1')
        self.assertFalse(receipt['readback_verified']);self.assertFalse(receipt['cleanup_allowed'])

    def test_client_submission_and_precise_cleanup(self):
        self.start()
        client=Client.__new__(Client)
        def request(path,doc=None,method=None,raw=None):
            response=self.request(path,doc,method,raw)
            if response.status_code>=400:raise RuntimeError(str(response.status_code))
            return response.json()
        client.request=request
        files=[]
        for role in CORE_ROLES:
            name=role+'.json';(self.base/name).write_text('fixture',encoding='utf-8')
            files.append({'path':name,'role':role,'retention':'core'})
        (self.base/'cache.bin').write_bytes(b'disposable')
        (self.base/'unknown.bin').write_bytes(b'keep me')
        files.append({'path':'cache.bin','role':'derived','retention':'scratch','in_use':False,'rebuild':'fixture recipe'})
        manifest=self.base/'files.json';manifest.write_text(json.dumps({'schema':'research_test_files.v1','files':files}))
        receipt=client.submit('run-1',manifest,'completed','fixture only')
        self.assertTrue(receipt['cleanup_allowed'])
        cleanup=cleanup_scratch(manifest,receipt)
        self.assertEqual(cleanup['bytes_removed'],10)
        self.assertFalse((self.base/'cache.bin').exists())
        self.assertEqual((self.base/'unknown.bin').read_bytes(),b'keep me')
        self.assertTrue((self.base/'case_results.json').exists())

    def test_cleanup_refuses_unverified_or_active_files(self):
        manifest=self.base/'files.json';cache=self.base/'cache';cache.write_bytes(b'stay')
        manifest.write_text(json.dumps({'schema':'research_test_files.v1','files':[
            {'path':'cache','role':'derived','retention':'scratch','in_use':True,'rebuild':'fixture'}]}))
        with self.assertRaises(ValueError):cleanup_scratch(manifest,{'cleanup_allowed':False})
        with self.assertRaises(ValueError):cleanup_scratch(manifest,{'cleanup_allowed':True,'readback_verified':True,'artifacts':[]})
        self.assertTrue(cache.exists())

    def test_wrapper_checks_server_before_starting_any_child(self):
        spec=importlib.util.spec_from_file_location('routine_archive_test',ROOT/'tests/glm5/run_routine_test.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        script=ROOT/'tests/glm5/test_test_archive.py';questions=self.base/'questions.json';questions.write_text('[]')
        p=plan();p['entrypoint_sha256']=module.digest(script);p['dataset']['questions_file_sha256']=module.digest(questions);p['execution_args']=[]
        p['acquisition'].update(entrypoint_sha256=module.digest(script),questions_file_sha256=module.digest(questions),environment_sha256=digest(module.runtime_identity()))
        p['acquisition']['generation']['script_args']=[]
        path=self.base/'plan.json';path.write_text(json.dumps(p))
        args=SimpleNamespace(archive_plan=path,archive_decision='new_acquisition',archive_reason='fixture',script=script,questions=questions,peak_gib=.01,script_args=[],model_snapshot=None,tokenizer_sha256=None,seed=None,center='http://127.0.0.1',token_file=None)
        with patch.object(module,'Client') as fake,patch.object(module.subprocess,'Popen') as launch:
            fake.return_value.preflight.side_effect=RuntimeError('offline')
            with self.assertRaisesRegex(RuntimeError,'offline'):module.run(args)
            launch.assert_not_called()

    def test_wrapper_end_to_end_submits_outputs_and_only_removes_declared_cache(self):
        spec=importlib.util.spec_from_file_location('routine_archive_full',ROOT/'tests/glm5/run_routine_test.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        script=Path(__file__).resolve();questions=self.base/'questions.json';questions.write_text('[{"case_id":"fixture"}]')
        p=plan();p.update(entrypoint_sha256=module.digest(script),execution_args=['--runner-fixture'])
        p['dataset']['questions_file_sha256']=module.digest(questions)
        p['acquisition'].update(entrypoint_sha256=module.digest(script),questions_file_sha256=module.digest(questions),environment_sha256=digest(module.runtime_identity()))
        p['acquisition']['generation']['script_args']=['--runner-fixture']
        plan_path=self.base/'plan.json';plan_path.write_text(json.dumps(p))
        def request(path,doc=None,method=None,raw=None):
            response=self.request(path,doc,method,raw)
            if response.status_code>=400:raise RuntimeError(str(response.status_code))
            return response.json()
        class Bridge(Client):
            def __init__(self,*_):self.request=request
        def checked(path,*args,**kwargs):
            target=self.base/'records' if Path(path)==ROOT/'tests/glm5/result/routine' else path
            result=check_write(target,*args,**kwargs);result.mkdir(parents=True,exist_ok=True);return result
        args=SimpleNamespace(archive_plan=plan_path,archive_decision='new_acquisition',archive_reason='fixture',archive_source_run_id='',script=script,questions=questions,peak_gib=.05,script_args=['--runner-fixture'],model_snapshot=None,tokenizer_sha256=None,seed=None,center='http://127.0.0.1',token_file=None)
        with patch.object(module,'Client',Bridge),patch.object(module,'checked_dir',checked),patch.object(module,'storage_dir',lambda *parts:self.base.joinpath('storage',*parts)):
            callback = Mock(side_effect=RuntimeError('interrupted before launch'))
            with self.assertRaisesRegex(RuntimeError,'interrupted before launch'):
                module.run(args,on_registered=callback)
            registration,record = callback.call_args.args
            staged = json.loads((record/'review.json').read_text(encoding='utf-8'))
            self.assertFalse((Path(staged['scratch'])/'output.log').exists())
            self.assertEqual(module.run(args,registered_record=record),0)
        review=json.loads(next((self.base/'records').glob('*/review.json')).read_text(encoding='utf-8'))
        work=Path(review['scratch'])
        self.assertEqual(review['server_run_id'],registration['id'])
        self.assertEqual(review['archive_state'],'server_verified')
        self.assertFalse((work/'projection.tmp').exists())
        self.assertTrue((work/'case_results.jsonl').exists())
        self.assertTrue((work/'unclassified.bin').exists())


if __name__=='__main__':
    if sys.argv[1:]==['--runner-fixture']:
        work=Path(os.environ['AI2050_TEST_WORK_DIR'])
        (work/'conclusion.json').write_text('{"fixture_only":true}')
        (work/'case_results.jsonl').write_text('{"case_id":"fixture","output":"fixture","status":"pass"}\n')
        (work/'projection.tmp').write_bytes(b'disposable')
        (work/'unclassified.bin').write_bytes(b'must remain')
        (work/'artifacts.json').write_text(json.dumps({'schema':'research_test_files.v1','files':[
            {'path':'projection.tmp','role':'derived','retention':'scratch','in_use':False,'rebuild':'engineering fixture'}]}))
        print('engineering fixture 1/1 done; no model',flush=True)
    else:unittest.main()
