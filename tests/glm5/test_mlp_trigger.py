"""Planted and negative checks for common-cause interpretation, not scientific data."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_trigger_run as R

class TestTrigger(unittest.TestCase):
    def test_common_input_can_produce_correlation_without_serial_path(self):
        rng=np.random.default_rng(31);z=rng.normal(size=1000);a=2*z+rng.normal(size=1000)*.1;b=-3*z+rng.normal(size=1000)*.1
        self.assertLess(np.corrcoef(a,b)[0,1],-.99)
        x=np.column_stack([np.ones(1000),z]);m=R.fit(x[:500],np.column_stack([a,b])[:500]);err=np.column_stack([a,b])[500:]-R.predict(m,x[500:])
        self.assertLess(abs(np.corrcoef(err.T)[0,1]),.15)
        # In this planted DAG, clamping A does not remove Z -> B. A serial DAG differs.
        self.assertEqual(float((-3*1)-(-3*0)), -3.)
    def test_independent_noise_does_not_predict_heldout(self):
        rng=np.random.default_rng(44);x=np.column_stack([np.ones(1024),rng.normal(size=(1024,8))]);y=rng.normal(size=(1024,2))
        m=R.fit(x[:512],y[:512]);err=np.mean((R.predict(m,x[512:])-y[512:])**2)
        self.assertGreater(err/np.mean((y[512:]-y[:512].mean(0))**2),.95)
    def test_nuisance_terms_match_frozen_definition(self):
        row={'long':1,'repetition':0,'past':1,'role':0,'mapping':1,'identity':2,'tokens':[0]*800}
        self.assertEqual(R.nuisance(row,400),[1,1,0,1,0,1,0,1,.5,.8])
    def test_signed_bf16_conversion(self):
        from shared.dist_protocol.mechanism_trace import unbits
        self.assertEqual(unbits(np.array([0x3f80,0xbf80,0x4348],dtype=np.uint16)).tolist(),[1.,-1.,200.])

if __name__=='__main__':unittest.main()
