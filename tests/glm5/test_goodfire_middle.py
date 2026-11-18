"""CPU-only engineering checks; planted data never enters research evidence."""
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
import goodfire_middle as M
import goodfire_middle_analysis as A


class MiddleAnalysisChecks(unittest.TestCase):
    def test_discovery_scaling_is_insensitive_to_heldout_values(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(8192,262));x[:,256]=1
        changed=x.copy();changed[4096:]+=100
        y,s=A.standardize(x);z,t=A.standardize(changed)
        np.testing.assert_array_equal(y[:4096],z[:4096]);self.assertEqual(s,t)
        np.testing.assert_array_equal(y[:,256],np.ones(8192))

    def test_constant_nuisance_columns_stay_finite(self):
        x=np.zeros((8192,262));x[:,256]=1;x[:,200]=9
        y,_=A.standardize(x)
        self.assertTrue(np.isfinite(y).all());self.assertTrue((y[:,200]==0).all())

    def test_planted_context_and_document_shift_control(self):
        rng=np.random.default_rng(51);x=np.zeros((8192,262),np.float32)
        x[:,:4]=rng.normal(size=(8192,4));x[:,256]=1
        x,_=A.standardize(x);truth=2*x[:,0]-3*x[:,1]+.7
        op=A.operator(x,256);fit=op@truth[:4096]
        wrong=op@np.roll(truth[:4096].reshape(512,8),137,axis=0).reshape(-1)
        prediction=x[4096:]@fit;shifted=x[4096:]@wrong
        err=np.mean((prediction-truth[4096:])**2);bad=np.mean((shifted-truth[4096:])**2)
        self.assertLess(err,.01);self.assertGreater(bad,5.)

    def test_composed_gate_value_can_recover_non_linear_product(self):
        rng=np.random.default_rng(201);x=np.zeros((8192,262),np.float32)
        x[:,:2]=rng.normal(size=(8192,2));x[:,256]=1;x,_=A.standardize(x)
        g=2*x[:,0];v=3*x[:,1];actual=(g/(1+np.exp(-g)))*v
        op=A.operator(x,256)
        pg=x[4096:]@(op@g[:4096]);pv=x[4096:]@(op@v[:4096])
        composed=(pg/(1+np.exp(-pg)))*pv;direct=x[4096:]@(op@actual[:4096])
        self.assertLess(np.mean((composed-actual[4096:])**2),.01*np.mean((direct-actual[4096:])**2))

    def test_storage_not_admitted_blocks_gpu_entry(self):
        with patch.object(M,'BASE',M.BASE/'nonexistent-admission-fixture'):
            with self.assertRaisesRegex(AssertionError,'Storage not admitted'):
                M.require_admission()

    def test_tied_unembedding_route(self):
        with patch.object(M,'SPEC',M.SPECS['qwen4']),patch.object(M,'ORIGINAL_WEIGHT',side_effect=lambda key:key):
            self.assertEqual(M.weight('lm_head.weight'),'model.embed_tokens.weight')
        with patch.object(M,'SPEC',M.SPECS['qwen14']),patch.object(M,'ORIGINAL_WEIGHT',side_effect=lambda key:key):
            self.assertEqual(M.weight('lm_head.weight'),'lm_head.weight')

    def test_full_b18_b28_identity_counts(self):
        self.assertEqual(M.BLOCKS,list(range(18,29)))
        self.assertEqual(11*M.SPECS['qwen4']['width'],107008)
        self.assertEqual(11*M.SPECS['qwen14']['width'],191488)
        self.assertEqual(len(M.C.POSITIONS),8)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):unittest.main(verbosity=2)
