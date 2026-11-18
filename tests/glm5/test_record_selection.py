"""Isolated engineering tests, never evidence about trained Qwen behavior."""
import unittest
import numpy as np
import record_selection_core as R
import record_selection_design as D
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
            self.assertEqual(arrays['units'].shape,(3,64));self.assertLess(arrays['qkv_relative_error'].max(),.02)
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

if __name__=='__main__':unittest.main()
