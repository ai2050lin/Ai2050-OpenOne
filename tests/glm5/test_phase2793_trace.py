"""Random tiny CPU software fixtures only, never scientific model results."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from shared.dist_protocol.mechanism_trace import TraceStore,Recorder,GpuLease,digest,save_json,unbits


class TraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        cls.torch=torch;torch.manual_seed(29)
        cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.)
        cfg._attn_implementation='sdpa';cls.model=Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    def setUp(self):
        base=ROOT/'tests/glm5_temp';base.mkdir(exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=base)
        self.root=Path(self.tmp.name);self.store=TraceStore(self.root/'index',self.root)
    def tearDown(self):
        assert self.root.resolve().is_relative_to((ROOT/'tests/glm5_temp').resolve());self.tmp.cleanup()
    def capture(self):
        torch=self.torch;ids=[2,4,7,9,11]
        with torch.inference_mode():
            baseline=self.model(input_ids=torch.tensor([ids]),logits_to_keep=1).logits
            with Recorder(self.model,[1,4],self.root/'native',deep=True) as r:
                out=self.model(input_ids=torch.tensor([ids]),output_hidden_states=True,logits_to_keep=1)
        self.assertTrue(torch.equal(baseline,out.logits))
        class Tok:
            def decode(self,ids):return ':'.join(map(str,ids))
        d,p=r.finish(out,{'id':'software-fixture','text':'fixture','prompt':'fixture','token_ids':ids},Tok(),'a'*64,'fixture',{'source':'software fixture'})
        self.store.register(p);return d,p
    def test_instrumentation_preserves_logits_and_full_units(self):
        d,_=self.capture();f=self.store.field(d['trace_id'],0,4,'mlp_activation')
        self.assertEqual(f['shape'],[64]);self.assertEqual(len(f['values']),64)
        self.assertLess(max(v['attention_relative_l2'] for v in d['reconstruction'].values()),.05)
    def test_query_position_is_real_not_last_position_alias(self):
        d,_=self.capture();first=self.store.attention_row(d['trace_id'],0,3,0);last=self.store.attention_row(d['trace_id'],0,3,4)
        self.assertEqual(first['kv_group'],1);self.assertEqual(first['weights'],[1.,0.,0.,0.,0.])
        self.assertNotEqual(last['weights'],first['weights']);self.assertAlmostEqual(sum(last['weights']),1.,places=2)
    def test_full_embedding_and_layer_boundaries(self):
        d,_=self.capture();x=self.store.hidden(d['trace_id'],0,3)
        self.assertEqual(x['boundary'],'embedding');self.assertEqual(len(x['values']),32)
        self.assertEqual(self.store.hidden(d['trace_id'],2,4)['layer'],2)
        with self.assertRaises(ValueError):self.store.hidden(d['trace_id'],3,0)
    def test_hash_and_manifest_tampering_rejected(self):
        d,p=self.capture();bad=copy.deepcopy(d);bad['sample_id']='changed';save_json(p,bad)
        with self.assertRaises(ValueError):self.store.register(p)
        raw=self.root/'native/block_00.npz';raw.write_bytes(b'bad')
        with self.assertRaises(ValueError):self.store.block(d['trace_id'],0)
    def test_out_of_range_or_unregistered_boundaries_rejected(self):
        d,_=self.capture()
        with self.assertRaises(ValueError):self.store.attention_row(d['trace_id'],0,4,1)
        with self.assertRaises(KeyError):self.store.field(d['trace_id'],0,2,'mlp_activation')
        with self.assertRaises(ValueError):self.store.field(d['trace_id'],0,4,'imaginary')
    def test_gpu_lease_cannot_be_reentered_by_another_owner(self):
        with GpuLease(self.root/'gpu.lock'):
            with self.assertRaises(RuntimeError):GpuLease(self.root/'gpu.lock').acquire()
        with GpuLease(self.root/'gpu.lock'):pass
    def test_cached_generation_uses_absolute_query_position(self):
        torch=self.torch
        with torch.inference_mode():
            prefix=self.model(input_ids=torch.tensor([[2,4,7]]),use_cache=True,logits_to_keep=1)
            with Recorder(self.model,[0],self.root/'step',position_offset=3) as rec:
                out=self.model(input_ids=torch.tensor([[9]]),past_key_values=prefix.past_key_values,use_cache=True,
                    output_hidden_states=True,logits_to_keep=1)
        class Tok:
            def decode(self,ids):return str(ids)
        d,p=rec.finish(out,{'id':'fixture-generation','text':'fixture','prompt':'fixture','token_ids':[2,4,7,9],
            'generation_step':1},Tok(),'a'*64,'fixture',{})
        self.store.register(p);row=self.store.attention_row(d['trace_id'],0,0,3)
        self.assertEqual(d['readout_positions'],[3]);self.assertEqual(len(row['weights']),4)
        self.assertGreater(sum(row['weights'][1:]),.1)
    def test_cpu_parameter_connection_sums_include_every_unit(self):
        from unittest.mock import patch
        from safetensors.torch import save_file
        import server.mechanism_trace_service as S
        d,_=self.capture();folder=self.root/'weights';folder.mkdir()
        save_file({k:v.detach().contiguous() for k,v in self.model.state_dict().items()},str(folder/'model.safetensors'))
        m={'id':'fixture-local','_path':folder,'config':self.model.config.to_dict()}
        with patch.object(S,'_store',self.store),patch.object(S,'trace_model',return_value=m):
            result=S.contributions(d['trace_id'],0,4,1,7,2)
        self.assertEqual(len(result['mlp_output_terms']),64);self.assertEqual(len(result['head_gate_terms']),4)
        for k in ['gate','value','attention_output_coordinate','mlp_output_coordinate']:
            self.assertAlmostEqual(result['observed'][k],result['reconstructed'][k],delta=.01)
        self.assertIn('not causal',result['edge_types']['head_to_mlp'])
    def test_read_only_endpoints_reject_invalid_queries(self):
        from unittest.mock import patch
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import server.mechanism_trace_service as S
        d,_=self.capture();app=FastAPI();app.include_router(S.router)
        with patch.object(S,'_store',self.store),TestClient(app) as client:
            self.assertEqual(client.get('/api/traces/catalog').status_code,200)
            good=client.get(f'/api/traces/{d["trace_id"]}/attention-row?query=0&head=3&block=0')
            self.assertEqual(good.json()['query_position'],0)
            self.assertEqual(client.get(f'/api/traces/{d["trace_id"]}/field?position=2&block=0').status_code,409)
            self.assertEqual(client.get(f'/api/traces/{d["trace_id"]}/attention-row?query=8&head=3&block=0').status_code,422)
    def test_live_long_input_limit_never_silently_truncates(self):
        from unittest.mock import patch
        import server.live_analysis_service as L
        from fastapi import HTTPException
        class Tok:
            def apply_chat_template(self,*a,**kw):return 'fixture'
            def encode(self,*a,**kw):return [2]*2049
        state={'status':'ready','model':self.model,'tokenizer':Tok()}
        self.assertEqual(L.MAX_TOKENS_CAP,2048)
        with patch.object(L,'_state',state):
            with self.assertRaises(HTTPException) as cm:L.live_run(L.LiveRun(text='software fixture',max_tokens=2048))
        self.assertEqual(cm.exception.status_code,422);self.assertIn('未截断',cm.exception.detail)

    def test_cached_generation_hidden_uses_absolute_manifest_position(self):
        from unittest.mock import patch
        import server.mechanism_trace_service as S
        self.test_cached_generation_uses_absolute_query_position()
        # The fixture above registers a one-token cached pass at absolute 3.
        with self.store.db() as db:
            rows=db.execute('SELECT id FROM traces').fetchall()
        identity=rows[-1]['id'];d,_=self.store.manifest(identity)
        if not d.get('parent_trace_id'):
            # Original minimal fixture has no parent identity; patch metadata
            # only in this reader test, never publish it as research evidence.
            original=self.store.manifest
            def patched(key):
                doc,path=original(key);return ({**doc,'parent_trace_id':'fixture-parent'},path)
            with patch.object(S,'_store',self.store),patch.object(self.store,'manifest',side_effect=patched):
                r=S.hidden_at(identity,0,3)
                self.assertEqual(len(r['values']),32)
                with self.assertRaises(KeyError):S.hidden_at(identity,0,0)


if __name__=='__main__':unittest.main()
