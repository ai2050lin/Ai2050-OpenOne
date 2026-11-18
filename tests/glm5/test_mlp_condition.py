"""Isolated algorithm checks; never registered as scientific evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import unittest
import numpy as np
import mlp_condition_analysis as A
import mlp_condition_design as D

class ConditionalPredictionTests(unittest.TestCase):
    def test_recovers_context_reversing_planted_transfer_on_new_samples(self):
        rng=np.random.default_rng(71)
        def sample(n):
            base=np.column_stack([np.ones(n),rng.integers(0,2,size=(n,5))])
            source=rng.normal(size=n)
            target=(2*base[:,2]-1)*source*3+rng.normal(scale=.03,size=n)
            return base,source,target
        b,x,y=sample(1800);fit=A.ridge(A.feature(b,x,'conditional'),y)
        simple=A.ridge(A.feature(b,x,'source'),y)
        b,x,y=sample(600)
        error=np.mean((A.predict(A.feature(b,x,'conditional'),fit)-y)**2)
        self.assertLess(error,.01)
        self.assertGreater(np.mean((A.predict(A.feature(b,x,'source'),simple)-y)**2),5)

    def test_unrelated_target_does_not_generalize(self):
        rng=np.random.default_rng(87)
        base=np.column_stack([np.ones(3000),rng.integers(0,2,size=(3000,5))]);x=rng.normal(size=3000);y=rng.normal(size=3000)
        fit=A.ridge(A.feature(base[:1500],x[:1500],'conditional'),y[:1500])
        pred=A.predict(A.feature(base[1500:],x[1500:],'conditional'),fit)
        self.assertGreater(np.mean((pred-y[1500:])**2)/np.mean(y[1500:]**2),.95)

    def test_future_mapping_not_visible_at_early_readout(self):
        row={'identity':0,'role':1,'late':0,'mapping':0,'positions':[1,3],'tokens':[9,8,7,6]}
        alternate={**row,'mapping':1}
        left=A.nuisance([row],[6,8]);right=A.nuisance([alternate],[6,8])
        np.testing.assert_array_equal(left[0],right[0]);self.assertNotEqual(left[1,4],right[1,4])

    def test_factorial_material_keeps_length_and_token_bag_for_role(self):
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(D.MODEL,local_files_only=True)
        for split in ('discovery','holdout'):
            for family in D.materials(tok,split,4):
                self.assertEqual(len(family['rows']),16)
                self.assertEqual(len({tuple(r[k] for k in ('identity','role','late','mapping')) for r in family['rows']}),16)

if __name__=='__main__':unittest.main()
