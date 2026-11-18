"""Isolated engineering/synthetic tests; never registered as scientific evidence."""
import unittest
import numpy as np
import state_formation_core as C
import state_formation_math as M
import state_formation_design as D
from shared.dist_protocol import grammar

class MathTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(query=q,record=r,language=l,color=q*r) for q in [-1,1] for r in [-1,1] for l in [-1,1]]
        self.x=C.design(self.rows)
    def test_center_shift_found_and_wrong_correction_fails(self):
        b=np.zeros((1,8,4));b[0,6]=[3,4,0,0];b[0,3]=[1,0,0,0]
        h=M.predict(b,self.x,list(range(8)));v=np.ones((1,4));nc=np.zeros((1,8));noise=np.zeros((1,4))
        scores=M.center_predictions(h,b,v,nc,noise,self.x[:,6])
        self.assertEqual(float(M.accuracy(scores['decoded_center'],self.x[:,3])[0]),1)
        self.assertEqual(float(M.accuracy(scores['fixed'],self.x[:,3])[0]),.5)
        self.assertEqual(float(M.accuracy(scores['wrong_center'],self.x[:,3])[0]),.5)
        # Self-corrected predictions must not use true color supplied to oracle.
        changed=M.center_predictions(h,b,v,nc,noise,-self.x[:,6])
        np.testing.assert_array_equal(changed['decoded_center'],scores['decoded_center'])
        np.testing.assert_array_equal(changed['fixed'],scores['fixed'])
    def test_no_center_shift_no_improvement(self):
        b=np.zeros((1,8,4));b[0,6,1]=2;b[0,3,0]=1;h=M.predict(b,self.x,list(range(8)))
        scores=M.center_predictions(h,b,np.ones((1,4)),np.zeros((1,8)),np.zeros((1,4)),self.x[:,6])
        np.testing.assert_array_equal(scores['fixed'],scores['decoded_center'])
    def test_factorial_and_strong_baseline(self):
        b=np.zeros((2,8,6));b[:,6,0]=2;b[:,3,1]=1;b[:,7,2]=.7
        h=M.predict(b,self.x,[0,3,6,7]);np.testing.assert_allclose(M.contrast(h,self.x),b,atol=1e-12)
        self.assertGreater(np.mean((M.predict(b,self.x,[0,1,2,3])-h)**2),0)
    def test_all_grammars_legal_and_family_conditions_complete(self):
        for split,n in [('discovery',8),('confirm',16)]:
            for frame in range(n):
                spec=D.build_grammar(split,frame);self.assertEqual(grammar.validate(spec),[])
                corpus=grammar.generate(spec,include_canary=False)
                self.assertEqual(len(corpus['items']),len(spec['fillers'])*8)
                self.assertTrue(all(i['split']==split for i in corpus['items']))
    def test_template_cluster_uncertainty_not_hidden_by_many_rows(self):
        values=np.repeat([-1.,1.],256);frames=np.repeat([0,1],256)
        ci=M.bootstrap(values,frames)
        self.assertLess(ci['frame_interval'][0],ci['family_interval'][0])
        self.assertGreater(ci['frame_interval'][1],ci['family_interval'][1])

class NativeInterventionTests(unittest.TestCase):
    def test_original_qwen_architecture_identity_and_source_mask(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        torch.manual_seed(42);torch.set_num_threads(2)
        config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,tie_word_embeddings=True)
        model=Qwen3ForCausalLM(config).eval();model.set_attn_implementation('sdpa')
        ids=torch.tensor([[1,2,3,4,5,6]]);mask=torch.ones((1,1,6,6),dtype=torch.bool).tril()
        def run():
            with torch.inference_mode():return model(ids,attention_mask={'full_attention':mask},use_cache=False).logits
        captured={}
        handle=model.model.layers[0].post_attention_layernorm.register_forward_pre_hook(lambda mod,args:captured.update(r=args[0].clone()))
        base=run();state=captured['r'][:,-1].clone();handle.remove();candidate={'module':'attention','block':0}
        with C.residual_replay(model,candidate,state):replayed=run()
        torch.testing.assert_close(base,replayed,atol=0,rtol=0)
        with C.ablated(model,candidate,0):zero=run()
        torch.testing.assert_close(base,zero,atol=0,rtol=0)
        with C.ablated(model,candidate,1):edited=run()
        self.assertGreater(float((edited[:,-1]-base[:,-1]).abs().max()),0)
        torch.testing.assert_close(edited[:,:-1],base[:,:-1],atol=0,rtol=0)
        with C.exclude_source(model,candidate,[1,2]):masked=run()
        self.assertGreater(float((masked[:,-1]-base[:,-1]).abs().max()),0)
        torch.testing.assert_close(masked[:,:-1],base[:,:-1],atol=0,rtol=0)
        torch.testing.assert_close(run(),base,atol=0,rtol=0)

if __name__=='__main__':unittest.main()
