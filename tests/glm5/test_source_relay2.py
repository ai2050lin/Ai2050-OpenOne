"""Guard causal visibility, family leakage, admission, and masked-route semantics."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay2_design as D
import source_relay2_math as M

class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.raw=D.raw_rows()
    def test_family_identity_and_balance(self):
        names=[]
        for split,rows in self.raw.items():
            self.assertEqual(len(rows),D.COUNTS[split]*12)
            names.append({x for r in rows for x in r['names']+r['sources']})
            for fam in {r['family'] for r in rows}:
                self.assertEqual({(r['source'],r['entity'],r['order']) for r in rows if r['family']==fam},
                    {(s,e,o) for s in range(3) for e in range(2) for o in range(2)})
        for i in range(3):
            for j in range(i):self.assertFalse(names[i]&names[j])
    def test_source_entity_and_spans(self):
        for rows in self.raw.values():
            for r in rows:
                self.assertEqual(r['gold'],r['values'][2*r['source']+r['entity']])
                a,b=r['query_source_span'];self.assertEqual(r['content'][a:b],r['sources'][r['source']])
                a,b=r['query_name_span'];self.assertEqual(r['content'][a:b],r['names'][r['entity']])
                for j,rec in r['records'].items():
                    a,b=rec['value_span'];self.assertEqual(r['content'][a:b],r['values'][j])
    def test_gate_missing_and_failed_cell(self):
        rows=self.raw['discovery'];items=[{'id':r['id'],'forced_correct':True,'normalized_correct':True,'terminated':True,'strict_correct':True,'top1_correct':True} for r in rows]
        self.assertTrue(D.gate(rows,items)['passed'])
        with self.assertRaises(AssertionError):D.gate(rows,items[:-1])
        for i,r in enumerate(rows):
            if r['source']==0 and r['entity']==0 and r['order']==0:items[i]['normalized_correct']=False
        self.assertFalse(D.gate(rows,items)['passed'])
    def test_new_query_frames_and_determinism(self):
        self.assertEqual({r['frame'] for r in self.raw['lockbox_frame']},{8,9,10,11})
        self.assertEqual(self.raw,D.raw_rows())
    def test_native_gqa_topology(self):
        cfg=D.read(D.MODELS[D.MODEL]/'config.json')
        for c in D.HEADS:self.assertEqual(c['kv_head'],c['head']//(cfg['num_attention_heads']//cfg['num_key_value_heads']))

class MathTests(unittest.TestCase):
    def test_right_padding_generation_mask(self):
        import torch
        from source_relay2_runtime import generation_key_mask
        lengths=torch.tensor([3,5])
        for step in [1,2,3]:
            mask=generation_key_mask(lengths,8,step)
            self.assertEqual(mask.sum(-1).tolist(),[3+step,5+step])
            self.assertFalse(mask[0,3:8].any());self.assertFalse(mask[1,5:8].any())
            self.assertTrue(mask[:,8:].all())
    def test_linear_planted_and_wrong_target(self):
        random=np.random.default_rng(143);x=random.normal(size=(600,12));w=random.normal(size=(12,8));y=x@w
        good=M.linear_fit(x[:400],y[:400]);bad=M.linear_fit(x[:400],random.normal(size=y[:400].shape))
        self.assertLess(np.mean((M.predict(x[400:],good)-y[400:])**2),.001)
        self.assertGreater(np.mean((M.predict(x[400:],bad)-y[400:])**2),1)
    def test_write_metric_matches_full_coordinates(self):
        random=np.random.default_rng(7);a=random.normal(size=(30,8));b=random.normal(size=(30,8));w=random.normal(size=(91,8))
        np.testing.assert_allclose(M.error_rows(a,b,w.T@w),np.square((a-b)@w.T).sum(1),rtol=1e-12)
    def test_masks_partition_correct_wrong(self):
        r={'tokens':list(range(30)),'target_record':3,'records':{str(j):{'span':[j*3,j*3+1,j*3+2]} for j in range(6)},'groups':{'records':list(range(18))}}
        self.assertFalse(set(M.block_tokens(r,'mask_correct'))&set(M.block_tokens(r,'mask_wrong')))
        self.assertEqual(set(M.block_tokens(r,'mask_correct'))|set(M.block_tokens(r,'mask_wrong')),set(M.block_tokens(r,'mask_records')))
        self.assertNotIn(29,M.block_tokens(r,'mask_allpast'))
    def test_family_repeats_not_independent(self):
        rows=[{'family':str(i//4)} for i in range(20)];x=np.arange(5).repeat(4)
        self.assertEqual(M.family_means(rows,x).tolist(),list(range(5)))
        self.assertEqual(M.holm([.01,.04]),[.02,.04])
    def test_advanced_index_preserves_head_order(self):
        a=np.arange(4*2*7*8).reshape(4,2,7,8);ix=np.array([1,3,6])
        z=a[0,:,ix].reshape(3,-1)
        for j,p in enumerate(ix):np.testing.assert_array_equal(z[j],a[0,:,p].reshape(-1))
    def test_rotary_matches_native_and_padding_invisible(self):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        random=np.random.default_rng(72);x=torch.tensor(random.normal(size=(3,9,128))).float();cos=torch.tensor(np.cos(random.normal(size=(3,9,64))).repeat(2,axis=-1)).bfloat16();sin=torch.tensor(np.sin(random.normal(size=(3,9,64))).repeat(2,axis=-1)).bfloat16()
        a,_=apply_rotary_pos_emb(x.bfloat16()[:,None],x.bfloat16()[:,None],cos,sin)
        self.assertTrue(torch.equal(a[:,0].float(),M.rotate_torch(x,cos.float(),sin.float())))
        valid=torch.ones((3,9),dtype=torch.bool);valid[:,:2]=False;q=x[:,-1];p,_=M.distribution_torch(q,x,cos.float(),sin.float(),valid)
        changed=x.clone();changed[:,:2]=1000;p2,_=M.distribution_torch(q,changed,cos.float(),sin.float(),valid)
        self.assertTrue(torch.equal(p,p2));self.assertTrue((p[:,:2]==0).all())

if __name__=='__main__':unittest.main()
