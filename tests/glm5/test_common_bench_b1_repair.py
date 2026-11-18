"""Engineering tests use synthetic data only, not research evidence."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import common_bench_b1_repair as R

class RepairTests(unittest.TestCase):
    def test_full_id_independence(self):
        ids=[f'bench-family-{i:06d}-same-suffix' for i in range(512)]
        self.assertEqual(len({R.seed_fingerprint(i,'random0') for i in ids}),512)
        self.assertFalse(np.array_equal(R.rng_for(ids[0],'random0').normal(size=10),R.rng_for(ids[0],'random1').normal(size=10)))

    def test_balance(self):
        rows=[{'id':str(i),'source':1,'language':1,'gold_color':'red'} for i in range(101)]
        a=R.target_map(rows);self.assertEqual(a,R.target_map(rows[::-1]))
        counts=np.bincount(list(a.values()),minlength=5)
        self.assertEqual(counts[0],0);self.assertLessEqual(max(counts[1:])-min(counts[1:]),1)

    def test_gate_top1_missing_cell_entity(self):
        rows=[];rec=[]
        for i,(s,l) in enumerate([(1,1),(1,-1),(-1,1),(-1,-1)]*10):
            rows.append({'id':str(i),'source':s,'language':l,'entities':['a','b'],'query':1,'frame':0})
            rec.append({'id':str(i),'base':0,'gold':0,'base_top1_correct':True})
        self.assertTrue(R.behavior_gate(rows,rec)[0]['passed'])
        rec[0]['base_top1_correct']=False
        for x in rec:x['base_top1_correct']=False
        self.assertFalse(R.behavior_gate(rows,rec)[0]['passed'])
        for x in rec:x['base_top1_correct']=True
        sel=[i for i,r in enumerate(rows) if r['source']==1]
        self.assertFalse(R.behavior_gate([rows[i] for i in sel],[rec[i] for i in sel])[0]['passed'])

    def test_decision_precedence(self):
        rates={'pooled':1,'cross_language':1,'cross_source':1,'cross_language_wrong_language':0}
        self.assertTrue(all(v=='supported' for v in R.decision(True,True,True,rates).values()))
        self.assertTrue(all(v=='descriptive_only_controls_failed' for v in R.decision(True,False,True,rates).values()))
        self.assertTrue(all(v=='behavior_ineligible' for v in R.decision(False,True,True,rates).values()))
        self.assertTrue(all(v=='invalid_execution' for v in R.decision(True,True,False,rates).values()))

    def test_actual_bf16_match(self):
        import torch
        from common_bench_b1_repair_run import effective_norm,match_norm
        g=torch.Generator().manual_seed(7)
        h=(torch.randn(8,2560,generator=g)*15).bfloat16()
        d=torch.randn(8,2560,generator=g)*3
        target=effective_norm(h,d)
        matched=match_norm(h,torch.randn(8,2560,generator=g),target)
        self.assertLess(float((effective_norm(h,matched)/target-1).abs().max()),.02)
        self.assertTrue(torch.equal((h.float()+torch.zeros_like(h)).to(h.dtype),h))

    def test_interval_not_perfect_certainty(self):
        lo,hi=R.cp_interval(100,100)
        self.assertLess(lo,1);self.assertEqual(hi,1)

if __name__=='__main__':unittest.main()
