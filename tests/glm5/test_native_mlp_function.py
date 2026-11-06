"""Meaningful control and archival fixtures; no fixture is scientific evidence."""
import unittest,sys,tempfile,copy,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from deploy.research_library import Library
from unittest.mock import patch
class StudyControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=P.read(P.OUT/'corpus.json')
    def test_pair_differs_in_one_key_token_only_andreadout_causality(self):
        rows=self.c['items'];self.assertEqual(len(rows),3264)
        for a,b in zip(rows[::2],rows[1::2]):
            x=np.asarray(a['token_ids']);y=np.asarray(b['token_ids']);diff=np.flatnonzero(x!=y)
            self.assertEqual(len(diff),1);self.assertEqual(diff[0],a['readout_positions'][1]);self.assertLess(a['readout_positions'][0],diff[0])
            self.assertEqual(a['readout_positions'][2],1023);self.assertEqual(x[1023],y[1023]);self.assertEqual(a['pair_id'],b['pair_id'])
    def test_holdout_documents_andname_stems_do_not_leak(self):
        seen={s:{r['background_doc_id'] for r in self.c['items'] if r['split']==s} for s in ['discovery','confirm','lockbox']}
        self.assertFalse(seen['discovery']&seen['confirm']);self.assertFalse(seen['confirm']&seen['lockbox'])
        self.assertNotIn('enumeration',{r['family'] for r in self.c['items']})
    def test_final_embedding_andselector_baseline_control(self):
        for split in ['discovery','confirm','lockbox']:
            for family in P.FAMILIES:
                rr=[r for r in self.c['items'] if r['split']==split and r['family']==family]
                for key in ['K','M']:
                    group=[r for r in rr if r['selected_key']==key];self.assertEqual(sum(r['semantic'] for r in group)*2,len(group))
    def test_spatial_binding_rejects_wrong_snapshot_and_retains_revision(self):
        with tempfile.TemporaryDirectory(dir=P.ROOT/'tests/glm5_temp') as folder:
            lib=Library(folder);a=lib.ingest({'schema':'report','kind':'real','model_snapshot':'a'*64},'fixture')['sha256']
            g=lib.ingest({'schema':'research_graph.v1','kind':'real','model_snapshot':'a'*64,'feature_cards':[],'graph_edges':[]},'graph')['sha256']
            doc={'schema':'research_space_binding.v1','kind':'real','run_id':'fixture','model_snapshot':'a'*64,'report_sha256':a,'graph_sha256':g,'context':{},'traces':[]}
            good=lib.ingest(doc,'fixture');bad={**doc,'model_snapshot':'b'*64}
            with self.assertRaises(ValueError):lib.ingest(bad,'bad')
            from server import research_space_service as S
            with patch.object(S,'library',return_value=lib):self.assertEqual(S.binding('fixture')['binding_sha256'],good['sha256'])
    def test_space_index_includes_unlocated_results_andno_inferred_features(self):
        with tempfile.TemporaryDirectory(dir=P.ROOT/'tests/glm5_temp') as folder:
            lib=Library(folder);lib.ingest({'kind':'real','summary':'unlocated negative result'},'fixture-unlocated')
            from server import research_space_service as S
            with patch.object(S,'library',return_value=lib):d=S.research_space()
            self.assertEqual(d['counts']['artifacts'],1);self.assertEqual(d['objects'],[]);self.assertEqual(d['artifacts'][0]['source'],'fixture-unlocated')
    def test_zero_discordance_has_nonzero_uncertainty(self):
        from native_mlp_function_validate import paired_bounds
        r=paired_bounds([True]*256,[True]*256,.0125)
        self.assertLess(r['ci95'][0],0);self.assertGreater(r['ci95'][1],0)
        self.assertEqual(r['confidence'],.9875)
    def test_catalog_keeps_latest_annotation_but_original_remains_addressable(self):
        from shared.dist_protocol.mechanism_trace import TraceStore,digest,save_json
        from server.trace_catalog import catalog
        with tempfile.TemporaryDirectory(dir=P.ROOT/'tests/glm5_temp') as tmp:
            root=Path(tmp);store=TraceStore(root/'index',root)
            d={'schema':'mechanism_trace.v1','kind':'real','model_snapshot':'a'*64,'run_id':'fixture','sample_id':'one','files':[], 'readout_positions':[12,22], 'n_layers':2}
            d['trace_id']=digest(d);save_json(root/'original.json',d);store.register(root/'original.json')
            old=d['trace_id'];d.pop('trace_id');d['annotation_revision']=3;d['trace_id']=digest(d)
            save_json(root/'corrected.json',d);store.register(root/'corrected.json')
            rows=catalog(store,'fixture')['traces'];self.assertEqual(len(rows),1);self.assertEqual(rows[0]['trace_id'],d['trace_id']);self.assertEqual(rows[0]['readout_positions'],[12,22])
            self.assertEqual(store.manifest(old)[0]['trace_id'],old)
    def test_binding_requires_model_when_same_run_has_multiple_snapshots(self):
        from server import research_space_service as S
        with patch.object(S,'bindings',return_value=[{'run_id':'x','model_snapshot':'a'*64},{'run_id':'x','model_snapshot':'b'*64}]):
            with self.assertRaises(ValueError):S.binding('x')
            self.assertEqual(S.binding('x','b'*64)['model_snapshot'],'b'*64)
    def test_place_names_are_balanced_independently_of_fact_andkey(self):
        c=P.read(P.OUT/'place-polarity/polarity-corpus.json')['items']
        for cue in [0,1]:
            for frame in [0,1]:
                rows=[r for r in c if r['cue_form']==cue and r['frame']==frame and r['question_polarity']==0]
                for fact in [0,1]:
                    rr=[r for r in rows if r['fact']==fact]
                    self.assertEqual(sum(r['name_suffix_assignment'] for r in rr)*2,len(rr))
                    self.assertEqual(sum(r['selected_key']=='K' for r in rr)*2,len(rr))
    def test_invalid_old_attention_reconstruction_is_not_served_as_real(self):
        from server import mechanism_trace_service as S
        from fastapi import HTTPException
        from unittest.mock import Mock
        store=Mock();store.manifest.return_value=({'reconstruction':{'0':{'attention_relative_l2':.7}}},None)
        with patch.object(S,'_store',store):
            with self.assertRaises(HTTPException) as caught:S.attention_row('historical',0,0,0)
            self.assertEqual(caught.exception.status_code,409);store.attention_row.assert_not_called()
    def test_boolean_padding_mask_reconstruction_matches_actual_sdpa(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from native_mlp_function_operation_traces_v2 import Recorder
        torch.manual_seed(31)
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        config._attn_implementation='sdpa';model=Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        ids=torch.tensor([[0,0,3,4,5],[1,2,3,4,5]]);mask=torch.tensor([[0,0,1,1,1],[1,1,1,1,1]])
        with tempfile.TemporaryDirectory(dir=P.ROOT/'tests/glm5_temp') as tmp,torch.inference_mode():
            baseline=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1).logits
            with Recorder(model,[4],Path(tmp)/'native') as recorder:
                out=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1)
            self.assertTrue(torch.equal(baseline,out.logits))
            self.assertLess(max(r['attention_relative_l2'] for r in recorder.reconstruction.values()),.05)
            with np.load(Path(tmp)/'native/block_00.npz') as z:self.assertTrue(np.all(z['attention_rows'][...,:2]==0))
    def test_streamed_pack_preserves_unicode_values_and_completion(self):
        import gzip
        from deploy.research_pack_stream import gzip_records
        with tempfile.TemporaryDirectory(dir=P.ROOT/'tests/glm5_temp') as tmp:
            p=Path(tmp)/'fixture.gz';doc={'schema':'datapack_extra.v1','id':'research-library','version':'fixture','entries':{'/api/甲':{'text':'汉字\\"🙂','native':[1,-.5,0]},'/api/乙':{'status':'negative'}}}
            with gzip.open(p,'wt',encoding='utf-8') as f:json.dump(doc,f,ensure_ascii=False)
            records=[json.loads(x) for x in gzip.decompress(b''.join(gzip_records(p))).decode().splitlines()]
            self.assertEqual(records[0]['schema'],'datapack_extra_stream.v1');self.assertEqual(records[1]['data'],doc['entries']['/api/甲']);self.assertEqual(records[-1]['n_entries'],2)
    def test_truncated_stream_does_not_emit_completion(self):
        import gzip
        from deploy.research_pack_stream import iter_records
        with tempfile.TemporaryDirectory(dir=P.ROOT/'tests/glm5_temp') as tmp:
            p=Path(tmp)/'truncated.gz'
            with gzip.open(p,'wt',encoding='utf-8') as f:f.write('{"schema":"datapack_extra.v1","id":"research-library","version":"fixture","entries":{"/api/x":{"ok":true}')
            records=[]
            with self.assertRaises(ValueError):
                for row in iter_records(p):records.append(row)
            self.assertFalse(any(r.get('complete') for r in records))
    def test_streamed_pack_entry_crosses_many_decoder_chunks(self):
        import gzip
        from deploy.research_pack_stream import iter_records
        with tempfile.TemporaryDirectory(dir=P.ROOT/'tests/glm5_temp') as tmp:
            p=Path(tmp)/'long.gz';v={'negative_context':'ab中"\\'*500000}
            with gzip.open(p,'wt',encoding='utf-8') as f:json.dump({'schema':'datapack_extra.v1','id':'research-library','version':'fixture','entries':{'/api/x':v}},f,ensure_ascii=False)
            self.assertEqual(list(iter_records(p))[1]['data'],v)
if __name__=='__main__':unittest.main()
