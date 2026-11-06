"""Protocol fixtures: task isolation, pinning, tampering and upload. No model inference."""
import base64
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'deploy')]
from shared.dist_protocol import natural_register_runner as N


def fixture():
    e={'alpha':6,'seed':27732026,'directions':{str(l):[1.]+[0.]*2559 for l in range(18,28)},
        'columns':{str(l):list(range(512)) for l in range(18,28)},'energy':{str(l):[1.]*9728 for l in range(18,28)}}
    items=[{'id':'fixture-'+reg,'family_id':'fixture-family','frame':0,'tokens':512,'shard_index':0,
        'register':reg,'facts':{'subject':'software fixture'},'text':'共同文本。'+req,'style_request':req,
        'token_ids':[i]*512,'prompt':'software fixture only','domain':'fixture'}
        for i,(reg,req) in enumerate([('acad','书面'),('colloq','口语')])]
    c={'schema':N.SCHEMA,'job_kind':N.JOB_KIND,'model_snapshot':'a'*64,'precision':'bf16','conditions':N.CONDITIONS,
        'items':items,'shards':1,'edit':e,'source_edit_sha256':N.digest(e),'max_new_tokens':64,'batch_size':2,
        'run_id':'software-fixture','canary_item':{'id':'fixed','token_ids':[0]*512}}
    c['canary_recipe_sha256']=N.digest(c['canary_item']);return c


def result(c):
    manifest={'schema':'fixture_native','files':[]}
    s={'protocol':N.SCHEMA,'status':'real','job_kind':N.JOB_KIND,'model_snapshot':c['model_snapshot'],'precision':'bf16',
        'source_edit_sha256':c['source_edit_sha256'],'canary_recipe_sha256':c['canary_recipe_sha256'],
        'weights_restored':True,'corpus_sha256':N.digest(c),'shard':'0/1','native_manifest_sha256':N.digest(manifest),
        'provenance':{'runner_sha256':N.file_sha(N.__file__)}}
    outputs={'schema':'natural_free_outputs.v1','kind':'real','outputs':[{'sample_id':r['id'],'condition':cond,'text':'软件测试。',
        'input_sha256':N.digest(r['token_ids']),'token_ids':[1,2]} for r in c['items'] for cond in N.CONDITIONS]}
    canary={'entries':{'fixed':{'0':{'head':[0.]*16,'norm':1.,'dim':2560}}}}
    return s,outputs,canary,manifest


class Pure(unittest.TestCase):
    def test_whole_family_shards_and_pair_guard(self):
        c=fixture();self.assertEqual(N.validate_corpus(c)['pairs'],1)
        bad=copy.deepcopy(c);bad['items'][1]['facts']={'different':True}
        with self.assertRaises(ValueError):N.validate_corpus(bad)
        bad=copy.deepcopy(c);bad['items'][1]['text']='different'
        with self.assertRaises(ValueError):N.validate_corpus(bad)
    def test_result_complete_and_no_grade_promotion(self):
        c=fixture();s,o,a,_=result(c);v=N.validate_result(c,'0/1',s,o,a)
        self.assertTrue(v['execution_validated']);self.assertFalse(v['mechanism_evidence_promoted'])
        bad=copy.deepcopy(o);bad['outputs'].pop()
        with self.assertRaises(ValueError):N.validate_result(c,'0/1',s,bad,a)
    def test_snapshot_tokens_and_restoration_must_match(self):
        c=fixture();s,o,a,_=result(c)
        for bad in [{**s,'model_snapshot':'b'*64},{**s,'weights_restored':False}]:
            with self.assertRaises(ValueError):N.validate_result(c,'0/1',bad,o,a)
        o['outputs'][0]['input_sha256']='0'*64
        with self.assertRaises(ValueError):N.validate_result(c,'0/1',s,o,a)


class Center(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault('AI2050_DIST_ADMIN_TOKEN','software-fixture-only')
        os.environ['AI2050_SNAPSHOT_DISABLE']='1'
        base=ROOT/'tests/glm5_temp';base.mkdir(exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=base)
        import distributed_service as D
        D._DATA_DIR=Path(self.tmp.name)/'dist';D._DATA_DIR.mkdir();D._conn=None;D._tls=threading.local();self.D=D
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app=FastAPI();app.include_router(D.router);self.client=TestClient(app)
        reg=self.client.post('/api/nodes/register',json={'name':'fixture-node'}).json();self.headers={'X-Node-Token':reg['node_token']}
        self.corpus=fixture();self.design=D._register_template({'tm_id':'NAT-FIXTURE-v1','version':1,'name':'software fixture',
            'dim':'ST','description':'software fixture','metric_pin':'fixture','model_pin':'a'*64,
            'corpus':self.corpus,'runner':Path(N.__file__).read_bytes().decode('utf-8')})
    def tearDown(self):
        self.client.close()
        for conn in [self.D._conn,getattr(self.D._tls,'conn',None)]:
            if conn:conn.close()
        self.D._conn=None;self.D._tls=threading.local();gc.collect()
        assert Path(self.tmp.name).resolve().is_relative_to((ROOT/'tests/glm5_temp').resolve())
        self.tmp.cleanup()
    def claim(self,kind='natural_register',rev='a'*64):
        return self.client.post('/api/tasks/claim',headers=self.headers,json={'tm_id':'NAT-FIXTURE-v1','job_kind':kind,'model_id':'fixture','model_rev':rev,'dtype':'bf16'})
    def test_opt_in_and_full_model_pin(self):
        self.assertEqual(self.claim('family_test').status_code,204)
        self.assertEqual(self.claim(rev='c'*64).status_code,204)
        task=self.claim();self.assertEqual(task.status_code,200);self.assertEqual(task.json()['job_kind'],'natural_register')
        self.assertEqual(self.claim().status_code,204)
    def test_frozen_upload_and_no_duplicate_dispatch(self):
        task=self.claim().json();s,o,a,m=result(self.corpus)
        files={name:N.canonical(doc) for name,doc in [('summary.json',s),('samples.json',o),('canary.json',a),('native-manifest.json',m)]}
        manifest=[{'path':k,'size':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in files.items()]
        meta={'task_id':task['task_id'],'tm_id':task['tm_id'],'design_sha':self.design,'model_id':'fixture',
            'model_rev':'a'*64,'precision':'bf16','kind':'real','summary':s,'manifest':manifest}
        init=self.client.post('/api/results/upload/init',headers=self.headers,json=meta);self.assertEqual(init.status_code,200,init.text)
        uid=init.json()['upload_id'];table=json.dumps(list(files),ensure_ascii=False).encode();blob=len(table).to_bytes(4,'big')+table+b''.join(files.values())
        self.client.post(f'/api/results/upload/{uid}/chunk',headers=self.headers,json={'seq':0,'data_b64':base64.b64encode(blob).decode()})
        fin=self.client.post(f'/api/results/upload/{uid}/finish',headers=self.headers);self.assertEqual(fin.status_code,200,fin.text)
        stored=json.loads(self.D._db().execute('SELECT summary_json FROM results WHERE sha=?',(fin.json()['sha'],)).fetchone()[0])
        self.assertFalse(stored['_natural_execution']['mechanism_evidence_promoted'])
        self.assertEqual(self.claim().status_code,204)
    def test_no_arbitrary_runner_in_natural_bundle(self):
        payload={'tm_id':'BAD','version':1,'name':'fixture','dim':'ST','description':'','metric_pin':'','model_pin':'',
            'corpus':self.corpus,'runner':'print(1)'}
        with self.assertRaises(ValueError):self.D._register_template(payload)


if __name__=='__main__':unittest.main()
