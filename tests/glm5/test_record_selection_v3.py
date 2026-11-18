"""Isolated engineering tests, never evidence about trained Qwen behavior."""
import unittest
import numpy as np
import record_selection_core_v3 as R
import record_selection_design_v3 as D
from shared.dist_protocol import grammar

class DesignTests(unittest.TestCase):
    def test_qkv_source_replay_and_actual_unit_collection(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        torch.manual_seed(24);torch.set_num_threads(2)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=40,hidden_size=32,intermediate_size=64,num_hidden_layers=36,
            num_attention_heads=32,num_key_value_heads=8,head_dim=128)).to(torch.bfloat16).eval().requires_grad_(False)
        model.set_attn_implementation('sdpa');cap=R.Recorder(model)
        def cpu_forward(row):
            cap.base.positions=row['positions'];cap.base.on=True;cap.base.values={}
            ids=torch.tensor([row['tokens']]);n=ids.shape[1];mask=torch.ones((1,1,n,n),dtype=torch.bool).tril()
            with torch.inference_mode():out=model(ids,attention_mask={'full_attention':mask},use_cache=False,logits_to_keep=1)
            cap.base.on=False;return out.logits[0,0],cap.base.values.copy()
        cap.base.forward=cpu_forward
        try:
            logits,arrays=cap.run({'tokens':[1,2,3,4,5,6],'positions':[0,1,3,5],
                'spans':{'record1':[1],'record2':[2],'query_instruction':[3]}})
            self.assertEqual(arrays['head_context'].shape,(36,32,128))
            self.assertEqual(arrays['units'].shape,(36,64));self.assertLess(arrays['qkv_relative_error'].max(),.02)
            np.testing.assert_allclose(arrays['source_mass'].sum(-1),1,atol=1e-6)
            self.assertTrue(np.isfinite(arrays['source_context']).all())
        finally:cap.close()
    def test_grammar_balance_and_new_frames(self):
        for split,n in [('discovery',8),('confirm',16)]:
            for frame in range(n):
                s=D.spec(split,frame);self.assertEqual(grammar.validate(s),[])
                rows=grammar.generate(s,include_canary=False)['items'];self.assertEqual(len(rows),len(s['fillers'])*8)
                self.assertEqual(set(r['split'] for r in rows),{split})
    def test_native_head_and_unit_edits_preserve_other_positions(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        torch.manual_seed(18);torch.set_num_threads(2)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=40,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
            num_attention_heads=32,num_key_value_heads=8,head_dim=128)).eval();model.set_attn_implementation('sdpa')
        ids=torch.tensor([[1,2,3,4,5,6]]);mask=torch.ones((1,1,6,6),dtype=torch.bool).tril()
        def run():
            with torch.inference_mode():return model(ids,attention_mask={'full_attention':mask},use_cache=False).logits
        base=run()
        for kind,indices in [('heads',[1,9]),('units',[2,3,7])]:
            candidate={'kind':kind,'block':0,'indices':indices}
            with R.group_edit(model,candidate,0):zero=run()
            torch.testing.assert_close(base,zero,atol=0,rtol=0)
            with R.group_edit(model,candidate,1):edited=run()
            torch.testing.assert_close(base[:,:-1],edited[:,:-1],atol=0,rtol=0)
            self.assertGreater(float((base[:,-1]-edited[:,-1]).abs().max()),0)
            torch.testing.assert_close(base,run(),atol=0,rtol=0)
        with R.group_source_exclusion(model,{'kind':'heads','block':0,'indices':[1,9]},[1,2]):edited=run()
        self.assertGreater(float((base[:,-1]-edited[:,-1]).abs().max()),0)
        torch.testing.assert_close(base[:,:-1],edited[:,:-1],atol=0,rtol=0)

class NullTests(unittest.TestCase):
    def test_native_direction_matched_realized_bf16_norm(self):
        import torch
        from record_selection_match import matched_direction
        torch.manual_seed(31);base=torch.randn(1,2560).bfloat16();other=(base.float()+.1*torch.randn(1,2560)).bfloat16()
        for target in [1.,3.,10.,30.]:
            value,error,actual,original=matched_direction(base,other,target)
            self.assertLessEqual(error,.02);self.assertGreater(original,0)
            self.assertAlmostEqual(actual,float((value.float()-base.float()).norm()))
            cosine=torch.nn.functional.cosine_similarity((value.float()-base.float()).flatten(),(other.float()-base.float()).flatten(),dim=0)
            self.assertGreater(float(cosine),.98)
    def test_absolute_maxT_detects_planted_and_is_sign_invariant(self):
        from record_selection_analysis import null_test
        rng=np.random.default_rng(42);x=rng.normal(size=(128,32));x[:,0]+=2
        p,_,_=null_test(x);q,_,_=null_test(-x)
        self.assertLess(p[0],.0125);np.testing.assert_array_equal(p,q)
        pure=rng.normal(size=(128,32));p,_,_=null_test(pure)
        self.assertGreater(p.min(),.0125)
    def test_failed_cell_excludes_whole_paired_family(self):
        from record_selection_collect import eligible
        r=dict(frame=0,valid_position=-1,conflict=1,language=1)
        rows=[r]*8+[dict(r,frame=1)]*8
        self.assertEqual(eligible(rows,{'cells':{'0:-1:1:1':{'strict_accuracy':.8}}}),[False,True])
    def test_ridge_recovers_unseen_linear_and_noise_does_not(self):
        from record_selection_analysis import ridge_fit,ridge_predict
        rng=np.random.default_rng(8);x=rng.normal(size=(256,7));v=rng.normal(size=(128,7));w=np.arange(7)
        y=x@w;yt=v@w
        pred=ridge_predict(v,*ridge_fit(x,y));bad=ridge_predict(v,*ridge_fit(x,rng.permutation(y)))
        self.assertLess(np.mean((pred-yt)**2),.01);self.assertGreater(np.mean((bad-yt)**2),40)

if __name__=='__main__':unittest.main()
