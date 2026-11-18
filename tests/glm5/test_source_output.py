"""Isolated engineering checks; random models are not scientific evidence."""
import unittest
import numpy as np
import source_output_core as S
from source_output_core import C
import source_output_design as D
from shared.dist_protocol import grammar

class SourceOutputTests(unittest.TestCase):
    def test_stop_registration_cannot_bypass_an_admitted_branch(self):
        from unittest.mock import patch
        import source_output_stopped as stopped
        with patch.object(C,'checked'),patch.object(stopped,'check_analysis'),patch.object(stopped,'admitted',return_value=['status:digits']),patch.object(C,'save') as save:
            with self.assertRaises(AssertionError):stopped.main('calibration')
            save.assert_not_called()
        branches={t+':'+a:{'passed':False} for t in S.TASKS for a in S.ALPHABETS}
        def read(path):return {'rows':[{}]*24} if path.name.endswith('-materials.json') else {'branches':branches}
        with patch.object(C,'checked'),patch.object(stopped,'check_analysis'),patch.object(stopped,'admitted',return_value=[]),patch.object(C,'read',side_effect=read),patch.object(C,'sha',return_value='test'),patch.object(C,'save') as save:
            stopped.main('calibration');calls={x.args[0]:x.args[1] for x in save.call_args_list}
            self.assertEqual(calls['admission-calibration.json']['stopped_branches'],list(branches))
            self.assertEqual(calls['native-calibration-complete.json']['n'],0)
    def test_text_parser_and_unfiltered_causal_selection(self):
        from source_output_analysis import parsed
        from source_output_causal import selected
        for split in ['calibration','confirm']:
            rows=C.read(C.OUT/(split+'-materials.json'))['rows']
            for row in rows:self.assertEqual(parsed(row),(row['target_sign'],row['code_sign'],row['language']))
        active=[t+':'+a for t in S.TASKS for a in S.ALPHABETS];subset=selected(rows,active);self.assertEqual(len(subset),1152)
        for branch in active:
            part=[r for r in subset if S.branch(r)==branch];self.assertEqual(len(part),192)
            for key in ['target_sign','code_sign','language']:self.assertEqual(sum(r[key] for r in part),0)
        self.assertEqual(selected(rows,[]),[])
    def test_prediction_finds_planted_effect_and_rejects_null(self):
        from source_output_analysis import ridge,predict,null_p
        rng=np.random.default_rng(523);x=rng.normal(size=(300,8));truth=x[:,0]+2*x[:,1];fit=ridge(x[:150],truth[:150])
        self.assertLess(np.mean((predict(x[150:],fit)-truth[150:])**2),.01)
        null=rng.normal(size=150);null-=null.mean();self.assertGreater(null_p(null),.5);self.assertLess(null_p(np.ones(150)),.01)
        shuffled=ridge(x[:150],rng.permutation(truth[:150]));self.assertGreater(np.mean((predict(x[150:],shuffled)-truth[150:])**2),1)
    def test_grammars_and_factorial_counts(self):
        for split,count,families in [('calibration',6,8),('confirm',12,16)]:
            for f in range(count):
                for task in S.TASKS:
                    for alphabet in S.ALPHABETS:
                        spec=D.spec(split,f,task,alphabet);self.assertEqual(grammar.validate(spec),[])
                        items=grammar.generate(spec,include_canary=False)['items'];self.assertEqual(len(items),4*families)
    def test_gate_stops_entire_branch_and_checks_actual_top1(self):
        rows=[];scores=[]
        for task in S.TASKS:
            for alphabet in S.ALPHABETS:
                for s in [-1,1]:
                    for m in [-1,1]:
                        for j in range(48):
                            rows.append({'task':task,'alphabet':alphabet,'target_sign':s,'code_sign':m,'language':1 if j%2 else -1})
                            scores.append({'strict_correct':True,'first_top1':True,'tie':False})
        gate=S.behavior(rows,scores,{'minimum_cell_n':48});self.assertTrue(gate['passed'])
        for i in range(8):scores[i]['first_top1']=False
        gate=S.behavior(rows,scores,{'minimum_cell_n':48});self.assertFalse(gate['branches']['status:digits']['passed'])
        self.assertTrue(gate['branches']['status:letters']['passed'])
    def test_two_position_native_qkv_swiglu_and_future_mask(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        torch.manual_seed(41);torch.set_num_threads(2)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=40,hidden_size=32,intermediate_size=64,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128)).to(torch.bfloat16).eval().requires_grad_(False)
        model.set_attn_implementation('sdpa');cap=S.Capture(model)
        def cpu_forward(row):
            cap.base.positions=row['positions'];cap.base.on=True;cap.base.values={};ids=torch.tensor([row['tokens']]);n=len(row['tokens'])
            with torch.inference_mode():out=model(ids,attention_mask={'full_attention':torch.ones(1,1,n,n,dtype=torch.bool).tril()},use_cache=False,logits_to_keep=1)
            cap.base.on=False;return out.logits[0,0],cap.base.values.copy()
        cap.base.forward=cpu_forward
        try:
            row={'tokens':[1,2,3,4,5,6,7],'positions':[3,6],'spans':{'record1':[1],'record2':[2],'instruction':[3,4,5]}}
            _,a=cap.run(row);self.assertEqual(a['hidden'].shape,(37,2,32));self.assertEqual(a['units'].shape,(2,64))
            self.assertEqual(a['selected_source_context'].shape,(2,2,4,128));self.assertLess(a['qkv_relative_error'].max(),.02)
            self.assertEqual(float(np.abs(a['attention_prob'][0,:,4:]).max()),0)
            changed=dict(row,tokens=[1,2,3,4,8,9,10]);_,b=cap.run(changed);np.testing.assert_array_equal(a['hidden'][:,0],b['hidden'][:,0])
            self.assertGreater(float(np.abs(C.unbits(a['hidden'][:,-1])-C.unbits(b['hidden'][:,-1])).max()),0)
        finally:cap.close()

if __name__=='__main__':unittest.main()
