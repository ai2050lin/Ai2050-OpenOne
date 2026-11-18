"""Engineering controls, never model evidence."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D

class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=D.source_rows()
    def good(self,split='discovery'):
        return [{'id':r['id'],'forced_correct':True,'normalized_correct':True,'strict_correct':True,'terminated':True,'top1_correct':True} for r in self.rows[split]]
    def test_disjoint_new_name_families(self):
        sets=[{n for r in rs for n in r['names']+r['sources']} for rs in self.rows.values()]
        self.assertEqual(sum(map(len,sets)),len(set.union(*sets)))
        self.assertEqual([len(rs) for rs in self.rows.values()],[768,1152,1152])
    def test_source_entity_order_full_product(self):
        from collections import Counter
        for split,rs in self.rows.items():
            c=Counter((r['source'],r['entity'],r['order']) for r in rs)
            self.assertEqual(set(c.values()),{D.COUNTS[split]})
            for r in rs:self.assertEqual(r['gold'],r['values'][2*r['source']+r['entity']])
    def test_complete_gate(self):self.assertTrue(D.gate(self.rows['discovery'],self.good())['passed'])
    def test_missing_rows_rejected(self):
        with self.assertRaises(AssertionError):D.gate(self.rows['discovery'],self.good()[:-1])
    def test_cell_failure_not_hidden_by_pool(self):
        items=self.good()
        for i,r in enumerate(self.rows['discovery']):
            if r['source']==0 and r['entity']==0 and r['order']==0:items[i]['forced_correct']=False
        self.assertFalse(D.gate(self.rows['discovery'],items)['passed'])
    def test_unterminated_outputs_fail(self):
        items=self.good()
        for r in items:r['terminated']=False
        self.assertFalse(D.gate(self.rows['discovery'],items)['passed'])
    def test_normalization_does_not_extract_answer(self):
        self.assertEqual(D.normalized(' Cat. \n'),'cat')
        self.assertNotEqual(D.normalized('Entity: cat.'),'cat')
        self.assertNotEqual(D.normalized('cat or dog'),'cat')
    def test_frames_cross_distance_and_relation(self):
        for rs in self.rows.values():
            for f in {r['frame'] for r in rs}:
                self.assertEqual({(r['distant'],r['label']) for r in rs if r['frame']==f},{(0,'code'),(0,'tag'),(1,'code'),(1,'tag')})

class AlgorithmTests(unittest.TestCase):
    def test_cached_decode_matches_full_prefix_with_padding(self):
        import torch
        from transformers import Qwen3Config,GlmConfig,DynamicCache
        from transformers.models.qwen3 import modeling_qwen3 as Q
        from transformers.models.glm import modeling_glm as G
        torch.set_num_threads(4);torch.manual_seed(120)
        with torch.inference_mode():
            for Config,mod,prefix in [(Qwen3Config,Q,'Qwen3'),(GlmConfig,G,'Glm')]:
                cfg=Config(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16)
                cfg._attn_implementation='sdpa'
                layers=[getattr(mod,prefix+'DecoderLayer')(cfg,i).bfloat16().eval() for i in range(2)]
                rotary=getattr(mod,prefix+'RotaryEmbedding')(cfg,device='cpu')
                lengths=torch.tensor([8,6]);valid=torch.arange(8)[None]>=8-lengths[:,None]
                pos=(valid.long().cumsum(1)-1).clamp_min(0);x=torch.randn(2,8,64).bfloat16()
                mask=torch.ones(1,1,8,8,dtype=torch.bool).tril()&valid[:,None,None]
                cache=DynamicCache(config=cfg);h=x.clone();pe=rotary(h,pos)
                for layer in layers:h=layer(h,attention_mask=mask,position_embeddings=pe,past_key_values=cache,use_cache=True)
                token=torch.randn(2,1,64).bfloat16();h=token.clone();pe=rotary(h,lengths[:,None])
                v=torch.arange(9)[None]>=8-lengths[:,None]
                for layer in layers:h=layer(h,attention_mask=v[:,None,None],position_embeddings=pe,past_key_values=cache,use_cache=True)
                full=torch.cat([x,token],dim=1);pos=(v.long().cumsum(1)-1).clamp_min(0);pe=rotary(full,pos)
                mask=torch.ones(1,1,9,9,dtype=torch.bool).tril()&v[:,None,None]
                for layer in layers:full=layer(full,attention_mask=mask,position_embeddings=pe)
                error=(h[:,0].float()-full[:,-1].float()).norm()/full[:,-1].float().norm()
                self.assertLess(float(error),.02)
    def test_forecast_edit_preserves_other_gqa_heads(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from transformers import Qwen3Config
        from transformers.models.qwen3 import modeling_qwen3 as mod
        from source_formation_causal import Editor
        torch.set_num_threads(4);torch.manual_seed(91)
        cfg=Qwen3Config(hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=16)
        cfg._attn_implementation='sdpa';layer=mod.Qwen3DecoderLayer(cfg,0).bfloat16().eval()
        rt=SimpleNamespace(t=torch,mod=mod,hd=16)
        ed=Editor(rt,{'heads':[{'block':0,'head':1,'kv_head':0}],'mlps':[]})
        q=torch.randn(2,4,9,16).bfloat16();k=torch.randn(2,2,9,16).bfloat16();v=torch.randn(2,2,9,16).bfloat16()
        mask=torch.ones(2,1,9,9,dtype=torch.bool).tril()
        with torch.inference_mode():
            base,_=ed.attention(layer.self_attn,q,k,v,mask,scaling=.25)
            ed.arm='forecast_qk';ed.forecasts={0:(q[:,1,-1],k[:,0])}
            exact,_=ed.attention(layer.self_attn,q,k,v,mask,scaling=.25)
            self.assertTrue(torch.equal(base,exact))
            ed.forecasts={0:(q[:,1,-1],-k[:,0])}
            changed,_=ed.attention(layer.self_attn,q,k,v,mask,scaling=.25)
            self.assertTrue(torch.equal(base[:,:-1],changed[:,:-1]))
            self.assertTrue(torch.equal(base[:,-1,[0,2,3]],changed[:,-1,[0,2,3]]))
            self.assertFalse(torch.equal(base[:,-1,1],changed[:,-1,1]))
    def test_native_coordinate_linear_forecast(self):
        import numpy as np
        from source_formation_analysis import coordinates,ridge,predict
        rng=np.random.default_rng(44);x=rng.normal(size=(400,256)).astype('f');y=3*x[:,7:8]-2*x[:,33:34]
        idx,_=coordinates(x,y,16);self.assertIn(7,idx);self.assertIn(33,idx)
        fit=ridge(x[:300,idx],y[:300]);pred=predict(x[300:,idx],fit)
        self.assertLess(np.mean((pred-y[300:])**2)/np.var(y[300:]),.01)
        random=rng.normal(size=y.shape).astype('f');f=ridge(x[:300,idx],random[:300])
        self.assertGreater(np.mean((predict(x[300:,idx],f)-random[300:])**2)/np.var(random[300:]),.8)
    def test_predictor_preserves_token_axis(self):
        import numpy as np
        from source_formation_analysis import save_fit,predict_fit
        rg=np.random.default_rng(67);x=rg.normal(size=(40,8)).astype('f');y=x[:,[2,6]]
        params={};save_fit(params,'p',x,y,np.array([2,6]))
        three=x.reshape(4,10,8)
        self.assertTrue(np.allclose(predict_fit(params,'p',three),predict_fit(params,'p',x).reshape(4,10,2)))
    def test_norm_matching_realized_bf16(self):
        import torch
        from source_formation_causal import damped_output
        torch.manual_seed(11);base=torch.randn(24,256).bfloat16()
        a=torch.randn(24,256)*4;b=torch.randn(24,256)*.3
        target=.5*torch.minimum(a.norm(dim=1),b.norm(dim=1))
        x,nx,sx=damped_output(base,a,target);y,ny,sy=damped_output(base,b,target)
        self.assertLess(float(((nx-ny).abs()/target).max()),.02)
        self.assertTrue(bool(((sx>=0)&(sx<=1)&(sy>=0)&(sy<=1)).all()))
        z,nz,_=damped_output(base,a,torch.zeros(24));self.assertTrue(torch.equal(base,z));self.assertEqual(float(nz.max()),0)
    def test_past_memory_excludes_last(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from transformers.models.qwen3 import modeling_qwen3 as m
        from source_formation_analysis import attention
        rt=SimpleNamespace(mod=m);q=np.ones((2,128),np.float32);k=np.zeros((2,5,128),np.float32);v=np.zeros_like(k)
        v[:,-1]=1000
        pe=(torch.ones(2,5,128).bfloat16(),torch.zeros(2,5,128).bfloat16())
        ctx,p=attention(rt,q,k,v,pe,[3,5],True)
        self.assertEqual(float(abs(ctx).max()),0);self.assertEqual(float(p[:,-1].max()),0)
        self.assertEqual(float(p[0,:2].max()),0)
        ctx,_=attention(rt,q,k,v,pe,[3,5],False);self.assertGreater(float(ctx.min()),100)
    def test_source_entity_interaction_changes_only_joint_term(self):
        import numpy as np
        from source_formation_analysis import relation_features
        rs=D.source_rows()['discovery'][:12]
        # Use actual tokenized spans for this engineering feature test.
        compiled=D.read(D.OUT/'qwen14-materials.json')['splits']['discovery'][:12]
        a=relation_features(compiled,200,False);b=relation_features(compiled,200,True)
        self.assertTrue(np.array_equal(a,b[:,:-1]));self.assertTrue(np.array_equal(b[:,-1],a[:,0]*a[:,1]))
        self.assertEqual(int(b[:,-1].sum()),12)

if __name__=='__main__':unittest.main()
