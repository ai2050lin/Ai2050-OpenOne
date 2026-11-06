"""Software correctness fixtures only: never LLM experiment evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'deploy'))
os.environ.setdefault('AI2050_DIST_ADMIN_TOKEN','phase2764-test-only')
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from deploy import research_library as R, parameter_reader as P, parameter_service as PS, node_control as C
import node_service as N


def weight_file(folder,dtype='BF16',shape=(3,5),values=None,name='model.layers.0.mlp.gate_proj.weight'):
    values=values or [float(i)-6 for i in range(shape[0]*shape[1])]
    if dtype=='BF16':raw=b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',v))[0]>>16) for v in values)
    else:raw=struct.pack('<'+{'F16':'e','F32':'f','F64':'d'}[dtype]*len(values),*values)
    head=json.dumps({name:{'dtype':dtype,'shape':list(shape),'data_offsets':[0,len(raw)]}}).encode()
    (folder/'model.safetensors').write_bytes(struct.pack('<Q',len(head))+head+raw)
    (folder/'config.json').write_text(json.dumps({'hidden_size':5,'intermediate_size':3,'num_hidden_layers':1,'num_attention_heads':1,'num_key_value_heads':1,'head_dim':5,'vocab_size':3}))
    return name


class Parameters(unittest.TestCase):
    def test_all_native_precisions_last_indices_and_bounds(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t)
            for dtype in ('BF16','F16','F32','F64'):
                name=weight_file(folder,dtype)
                result=P.read_slice(folder,name,1,2,64,64)
                self.assertEqual(result['values'],[[1.,2.,3.],[6.,7.,8.]])
                self.assertEqual(result['loaded_numel'],6)
                self.assertEqual(P.read_slice(folder,name,2,4,1,1)['values'],[[8.]])
            for args in ((-1,0,1,1),(3,0,1,1),(0,5,1,1),(0,0,65,1),(0,0,1,0)):
                with self.assertRaises(ValueError):P.read_slice(folder,name,*args)
            with self.assertRaises(FileNotFoundError):P.read_slice(folder,'../../secret',0,0,1,1)

    def test_corrupted_and_nonfinite_weights_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);name=weight_file(folder,'F32',values=[float('nan')]+list(range(14)))
            with self.assertRaises(ValueError):P.read_slice(folder,name,0,0,1,1)
            p=folder/'model.safetensors';p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):P.read_slice(folder,name,2,4,1,1)
            p.write_bytes(struct.pack('<Q',40*1024**2))
            with self.assertRaises(ValueError):P.header(p)

    def test_model_catalog_and_api_restrict_model_identity(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);weight_file(folder)
            models=P.catalog([folder,folder]);self.assertEqual(len(models),1)
            app=FastAPI();app.include_router(PS.router)
            with patch.object(PS,'roots',return_value=[folder]),TestClient(app) as client:
                meta=client.get('/api/parameters/models').json()
                self.assertNotIn('_path',meta['models'][0]);mid=meta['models'][0]['id']
                self.assertEqual(client.get('/api/parameters/slice',params={'model':mid,'name':'model.layers.0.mlp.gate_proj.weight','row':2,'col':4}).json()['values'],[[8.]])
                self.assertEqual(client.get('/api/parameters/slice',params={'model':'../../','name':'x'}).status_code,404)


class ResearchArchive(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.lib=R.Library(self.root/'archive')
    def tearDown(self):self.tmp.cleanup()

    def graph_doc(self):
        evidence=self.lib.ingest({'kind':'real','run_id':'fixture','result':'software fixture'},'fixture')['sha256']
        return {'schema':'research_graph.v1','kind':'real','model_snapshot':'a'*64,
                'feature_cards':[{'id':'f1','layer':3,'boundary':'mlp.down_proj.input','index':{'space':'mlp','unit':5},'method':'test-fixture.v1','run_id':'fixture','counterexamples':[],
                                  'evidence':{'level':'observation','artifact_sha256':evidence}}],
                'graph_edges':[{'id':'e1','source':'language:syntax','target':'f1','relation':'associated','run_id':'fixture','evidence':{'level':'negative','artifact_sha256':evidence}}]}

    def test_cas_immutable_idempotent_and_revisions_survive_source_deletion(self):
        research=self.root/'research';research.mkdir();p=research/'sample.research.json'
        p.write_text('{"kind":"real","value":1}')
        self.assertEqual(self.lib.scan(research)['new_revisions'],1)
        old=self.lib.summary()['recent'][0]['sha'];fp=self.lib.fingerprint()
        self.assertEqual(self.lib.scan(research)['new_revisions'],0)
        p.write_text('{"kind":"real","value":22}')
        self.assertEqual(self.lib.scan(research)['new_revisions'],1);p.unlink()
        self.lib.scan(research);self.assertEqual(self.lib.summary()['artifacts'],2)
        self.assertEqual(self.lib.artifact(old)['value'],1);self.assertNotEqual(fp,self.lib.fingerprint())
        self.lib.ingest({'kind':'real','value':1},'new-source')
        self.assertEqual(self.lib.summary()['artifacts'],2)

    def test_graph_provenance_references_and_no_automatic_promotion(self):
        doc=self.graph_doc();first=self.lib.ingest(doc,'explicit')
        self.lib.ingest(doc,'explicit-again')
        self.assertEqual(self.lib.summary()['graph_revisions'],{'feature':1,'edge':1})
        doc['feature_cards'][0]['evidence']['level']='heldout'
        second=self.lib.ingest(doc,'explicit');self.assertNotEqual(first['sha256'],second['sha256'])
        self.assertEqual(self.lib.artifact(first['sha256'])['feature_cards'][0]['evidence']['level'],'observation')
        self.assertEqual(len(self.lib.graph(limit=1)['objects']),1)
        self.assertEqual(self.lib.summary()['graph_revisions']['feature'],2)

    def test_missing_evidence_wrong_snapshot_synthetic_and_dangling_edges_rejected(self):
        base=self.graph_doc()
        for modify in (lambda d:d.update(kind='synthetic'),lambda d:d.update(model_snapshot='model-name'),
                       lambda d:d['feature_cards'][0].update(layer=-1),
                       lambda d:d['feature_cards'][0]['evidence'].update(artifact_sha256='b'*64),
                       lambda d:d['graph_edges'][0].update(target='unknown')):
            d=copy.deepcopy(base);modify(d)
            with self.assertRaises(ValueError):self.lib.ingest(d,'invalid')
        bad=self.root/'research';bad.mkdir();(bad/'bad.research.json').write_text('not-json')
        self.assertEqual(len(self.lib.scan(bad)['errors']),1)

    def test_concurrent_ingest_and_checksum_verification(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(4) as pool:
            results=list(pool.map(lambda _:self.lib.ingest({'result':3},'parallel'),range(12)))
        self.assertEqual(len({r['sha256'] for r in results}),1)
        sha=results[0]['sha256'];(self.lib.root/'objects'/sha[:2]/(sha+'.json')).write_text('{}')
        with self.assertRaises(ValueError):self.lib.artifact(sha)

    def test_offline_research_pack_contains_all_versions_and_graph_pages(self):
        from deploy import datapack as DP
        doc=self.graph_doc();self.lib.ingest(doc,'test')
        with patch.object(R,'library',return_value=self.lib):entries,notes=DP.research_entries()
        self.assertIn('/api/research-library/graph?limit=100&offset=0',entries)
        for r in self.lib.summary()['recent']:self.assertIn('/api/research-library/artifacts/'+r['sha'],entries)
        self.assertEqual(DP.pack_hint('/api/research-library/artifacts/'+'a'*64),'research-library')

    def test_legacy_nonfinite_report_preserves_original_bytes_and_missingness(self):
        folder=self.root/'research';folder.mkdir();raw=b'{"kind":"real","model":"legacy","stat":NaN}'
        (folder/'report.json').write_bytes(raw)
        self.assertEqual(self.lib.scan(folder)['errors'],[])
        doc=self.lib.artifact(self.lib.summary()['recent'][0]['sha'])
        self.assertEqual(doc['source_text'].encode('utf-8'),raw)
        self.assertEqual(doc['source_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(doc['report']['stat'],{'$nonfinite':'NaN'})


class ControlAndProgress(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.conn=sqlite3.connect(self.root/'center.db',check_same_thread=False);self.conn.row_factory=sqlite3.Row
        self.conn.executescript('''CREATE TABLE nodes(node_id TEXT PRIMARY KEY,name TEXT,last_seen REAL,owner_user_id TEXT);
            INSERT INTO nodes VALUES('mine','one',0,'owner'); INSERT INTO nodes VALUES('other','two',0,'other');
            CREATE TABLE templates(tm_id TEXT,design_sha TEXT,bundle_dir TEXT);
            CREATE TABLE results(sha TEXT,tm_id TEXT,model_id TEXT,model_rev TEXT,kind TEXT,summary_json TEXT,design_sha TEXT);
            CREATE TABLE audits(result_sha TEXT,status TEXT,reviewer TEXT,note TEXT,checks_json TEXT,created_at REAL);''')
        def auth(fn,token):
            if token!='Bearer owner':raise HTTPException(401,'login required')
            return {'user_id':'owner'}
        def node_auth(token):
            if token!='node-token':raise HTTPException(401,'bad node token')
            return dict(self.conn.execute("SELECT * FROM nodes WHERE node_id='mine'").fetchone())
        self.service=SimpleNamespace(_db=lambda:self.conn,_db_lock=threading.RLock(),_DATA_DIR=self.root,
                    _auth_call=auth,_accounts=SimpleNamespace(require_user=object(),require_admin=object()),_auth_node=node_auth)
        self.p=patch.object(R,'services',return_value=self.service);self.p.start()
        app=FastAPI();app.include_router(C.router);app.include_router(R.router);self.client=TestClient(app)
        self.headers={'Authorization':'Bearer owner'};self.nodeheaders={'X-Node-Token':'node-token'}
    def tearDown(self):self.client.close();self.p.stop();self.conn.close();self.tmp.cleanup()

    def heartbeat(self,grant='grant-1',ack=None):
        return self.client.post('/api/research-nodes/poll',headers=self.nodeheaders,json={'grant_id':grant,'state':'running','ack':ack or []}).json()
    def cmd(self,node='mine',action='pause'):
        return self.client.post(f'/api/research-nodes/{node}/commands',headers=self.headers,json={'action':action})

    def test_owner_only_whitelist_and_local_revoke(self):
        self.assertEqual(self.cmd().status_code,409);self.heartbeat()
        self.assertEqual(self.cmd('other').status_code,403)
        self.assertEqual(self.cmd(action='shell').status_code,422)
        self.assertEqual(self.client.get('/api/research-nodes').status_code,401)
        response=self.cmd();self.assertEqual(response.status_code,200);cid=response.json()['id']
        self.assertEqual(self.heartbeat()['commands'][0]['id'],cid)
        self.assertEqual(self.heartbeat('',[cid])['commands'],[])
        self.assertEqual(self.cmd(action='resume').status_code,409)
        self.assertEqual(self.client.get('/api/research-nodes',headers=self.headers).json()['nodes'][0]['history'][0]['status'],'applied')

    def test_expiry_new_grant_and_ack_idempotency(self):
        self.heartbeat();cid=self.cmd().json()['id']
        self.assertEqual(self.heartbeat('new-grant')['commands'],[])
        self.conn.execute('UPDATE node_commands SET expires=?',(time.time()-1,));self.conn.commit()
        self.assertEqual(self.heartbeat()['commands'],[])
        self.assertEqual(self.conn.execute('SELECT status FROM node_commands').fetchone()[0],'expired')
        self.conn.execute('UPDATE node_control_state SET updated=0');self.conn.commit()
        self.assertEqual(self.cmd().status_code,409)

    def test_progress_unique_shards_current_design_precision_and_audits(self):
        bundle=self.root/'bundle';bundle.mkdir();(bundle/'corpus.json').write_text(json.dumps({'family_id':'KN-ATTR','shards':2}))
        self.conn.execute('INSERT INTO templates VALUES(?,?,?)',('tpl','design',str(bundle)))
        cases=[('a','0/2','passed','ok','design','bf16','real'),('b','0/2','passed','ok','design','bf16','real'),
               ('c','1/2','rejected','ok','design','bf16','real'),('d','1/2','passed','drift','design','bf16','real'),
               ('e','1/3','passed','ok','design','bf16','real'),('f','1/2','passed','ok','old','bf16','real'),
               ('g','1/2','passed','ok','design','f32','real'),('h','1/2','passed','ok','design','bf16','synthetic')]
        for sha,shard,status,canary,design,precision,kind in cases:
            self.conn.execute('INSERT INTO results VALUES(?,?,?,?,?,?,?)',(sha,'tpl','qwen','a'*64,kind,json.dumps({'shard':shard,'precision':precision,'_canary':{'status':canary}}),design))
            self.conn.execute('INSERT INTO audits VALUES(?,?,?,?,?,?)',(sha,status,'reviewer','','{}',0))
        self.conn.commit();buckets=R.progress()['buckets'];self.assertEqual(len(buckets),2)
        bf=next(b for b in buckets if b['precision']=='bf16');self.assertEqual(bf['directions']['知识网络'],{'done':1,'planned':2})
        self.assertEqual(bf['directions']['语法体系']['planned'],0)
        self.conn.execute("UPDATE results SET model_rev='cfg-legacy'");self.conn.commit()
        for bucket in R.progress()['buckets']:
            self.assertFalse(bucket['snapshot_verified'])
            self.assertEqual(bucket['directions']['知识网络']['done'],0)

    def test_background_sync_retains_audit_revisions(self):
        self.conn.execute('INSERT INTO results VALUES(?,?,?,?,?,?,?)',('sha','tpl','qwen','snapshot','real','{}','design'))
        self.conn.commit()
        with patch.dict(os.environ,{'AI2050_RESEARCH_DIR':str(self.root/'research')}):
            R.sync();self.assertEqual(R.library().summary()['artifacts'],1)
            self.conn.execute('INSERT INTO audits VALUES(?,?,?,?,?,?)',('sha','rejected','reviewer','failed','{}',1));self.conn.commit()
            R.sync();self.assertEqual(R.library().summary()['artifacts'],2)
            self.assertEqual(R.library().summary()['sync']['errors'],[])

    def test_member_submission_archived_without_adding_graph_or_evidence_grade(self):
        doc={'schema':'research_submission.v1','kind':'real','run_id':'software-fixture','model_snapshot':'a'*64,'report':{'verdict':'negative'}}
        response=self.client.post('/api/research-library/submissions',json=doc,headers=self.headers)
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()['verification'],'unreviewed')
        self.assertEqual(R.library().summary()['graph_revisions'],{'feature':0,'edge':0})
        doc['graph_edges']=[{'id':'unverified'}]
        self.assertEqual(self.client.post('/api/research-library/submissions',json=doc,headers=self.headers).status_code,422)
        self.assertEqual(self.client.post('/api/research-library/submissions',json={},headers={}).status_code,401)

    def test_census_validation_and_dispute_revisions_are_retained(self):
        self.conn.executescript('''CREATE TABLE census_jobs(job_id TEXT,snapshot_sha256 TEXT,model_key TEXT);
            CREATE TABLE census_results(lease_id TEXT,sha TEXT,job_id TEXT,kind TEXT,status TEXT,summary_json TEXT,validation_json TEXT);
            INSERT INTO census_jobs VALUES('job','snapshot','model');
            INSERT INTO census_results VALUES('lease','sha','job','real','accepted','{}','{}');''')
        with patch.dict(os.environ,{'AI2050_RESEARCH_DIR':str(self.root/'research')}):
            R.sync();self.assertEqual(R.library().summary()['artifacts'],1)
            self.conn.execute("UPDATE census_results SET status='disputed'");self.conn.commit();R.sync()
            self.assertEqual(R.library().summary()['artifacts'],2)

    def test_computed_atlas_history_survives_snapshot_rebuild(self):
        folder=self.root/'snapshots';folder.mkdir();path=folder/'lang_atlas.json'
        path.write_text(json.dumps({'fingerprint':'a','data':{'nodes':['one']}}))
        with patch.dict(os.environ,{'AI2050_RESEARCH_DIR':str(self.root/'research')}):
            R.sync();first=R.library().summary()['recent'][0]['sha']
            path.write_text(json.dumps({'fingerprint':'b','data':{'nodes':['one','two']}}));R.sync()
            self.assertEqual(R.library().summary()['artifacts'],2)
            self.assertEqual(R.library().artifact(first)['data']['nodes'],['one'])


class NodeSession(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.svc=N.NodeService(Path(self.tmp.name),synthetic=True)
        self.svc.creds=lambda:{'server':'https://invalid.example','node_id':'mine','owner_bound':True}
        self.svc.cfg['participation_mode']='managed';self.svc.state='running';self.svc.grant_id='first'
    def tearDown(self):self.svc.shutdown();self.tmp.cleanup()
    def test_local_revoke_remote_pause_resume_and_old_command(self):
        self.svc._apply_remote_commands([{'id':'p','grant_id':'first','action':'pause'}])
        self.assertEqual(self.svc.state,'paused');self.assertEqual(self.svc.grant_id,'first')
        self.svc._apply_remote_commands([{'id':'r','grant_id':'first','action':'resume'}])
        self.assertEqual(self.svc.state,'running')
        self.svc.pause();self.svc._apply_remote_commands([{'id':'late','grant_id':'first','action':'resume'}])
        self.assertEqual(self.svc.state,'paused');self.assertFalse(self.svc.grant_id)
        self.svc.resume();self.assertNotEqual(self.svc.grant_id,'first')
        self.svc._apply_remote_commands([{'id':'bad','grant_id':self.svc.grant_id,'action':'shell'}])
        self.assertNotIn('bad',self.svc.command_seen)
        self.svc.stop();self.assertFalse(self.svc.status()['remote_granted'])
    def test_budgets_and_forbidden_config(self):
        self.svc.update_config({'limits':{'max_session_minutes':1}});self.svc.session_started=time.time()-61
        self.assertIn('时长',self.svc.budget_reason())
        self.svc.update_config({'limits':{'max_session_minutes':0,'max_work_gib':1/2**30}})
        work=Path(self.tmp.name)/'work';work.mkdir(exist_ok=True);(work/'data').write_bytes(b'xx')
        self.assertIn('存储',self.svc.budget_reason())
        for patch_ in ({'participation_mode':'shell'},{'limits':{'max_session_minutes':-1}},{'limits':{'max_work_gib':float('nan')}}):
            with self.assertRaises(ValueError):self.svc.update_config(patch_)

    def test_local_probe_records_supply_snapshot_identity(self):
        pid='GF-L3-U4-20261103-010000';folder=self.svc.probes_dir/pid;folder.mkdir(parents=True)
        (folder/'result.json').write_text(json.dumps({'env':{'dtype':'bf16'},'verdict':{'status':'not_passed'}}))
        (folder/'status.json').write_text(json.dumps({'params':{'snapshot':'b'*64},'status':'done'}))
        report=self.svc.get_probe(pid)
        self.assertEqual(report['research_identity']['model_snapshot'],'b'*64)
        self.assertEqual(report['research_identity']['run_id'],pid)

    def test_desktop_local_parameter_api_with_guard(self):
        import urllib.request
        import urllib.error
        folder=Path(self.tmp.name)/'models';folder.mkdir();name=weight_file(folder)
        self.svc.cfg['models_root']=str(folder)
        ref={};server=N.ThreadingHTTPServer(('127.0.0.1',0),N.make_handler(self.svc,'local-token',ref,()))
        ref['port']=server.server_address[1]
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f"http://127.0.0.1:{ref['port']}"
        def get(path):
            req=urllib.request.Request(base+path,headers={'X-Local-Token':'local-token'})
            with urllib.request.urlopen(req,timeout=3) as response:return json.load(response)
        try:
            with self.assertRaises(urllib.error.HTTPError) as denied:urllib.request.urlopen(base+'/parameters/models',timeout=3)
            self.assertEqual(denied.exception.code,401)
            mid=get('/parameters/models')['models'][0]['id']
            from urllib.parse import urlencode
            result=get('/parameters/slice?'+urlencode({'model':mid,'name':name,'row':2,'col':4,'rows':1,'cols':1}))
            self.assertEqual(result['values'],[[8.]])
        finally:server.shutdown();server.server_close();thread.join(3)


if __name__=='__main__':unittest.main()
