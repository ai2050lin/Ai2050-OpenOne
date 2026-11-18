"""Scientific bookkeeping tests: interaction recovery and null discrimination."""
import unittest
import numpy as np
import torch
from mlp_puzzle_campaign import features,bootstrap_gain

class Tests(unittest.TestCase):
    def test_bf16_realized_control_must_not_be_rounded_again(self):
        torch.manual_seed(1)
        h=torch.randn(4096).bfloat16()
        proposal=(h.float()+torch.randn(4096)*.05).bfloat16()
        realized=proposal.float()-h.float()
        self.assertFalse(torch.equal((h+realized.bfloat16()),proposal))
        self.assertTrue(torch.equal((h.float()+realized).bfloat16(),proposal))
    def test_binding_requires_interaction(self):
        rows=[dict(query=q,record=r,language=l) for q in (-1,1) for r in (-1,1) for l in (-1,1)]
        x=np.stack([features(r) for r in rows]);np.testing.assert_allclose(x.T@x,8*np.eye(8))
        y=x[:,6]*3+x[:,3]*2
        coef=x.T@y/8
        np.testing.assert_allclose(x@coef,y)
        self.assertGreater(np.mean((x[:,:6]@coef[:6]-y)**2),8)
    def test_additive_null_not_interaction(self):
        x=np.stack([features(dict(query=q,record=r,language=l)) for q in (-1,1) for r in (-1,1) for l in (-1,1)])
        y=x[:,1]*2+x[:,2]*3+x[:,3]
        coef=x.T@y/8
        self.assertEqual(float(np.sum(coef[6:]**2)),0)
    def test_family_interval_sign(self):
        lower,_=bootstrap_gain(np.ones(512),np.ones(512)*2)
        self.assertGreater(lower,0)
        lower,_=bootstrap_gain(np.ones(512)*2,np.ones(512))
        self.assertLess(lower,0)

if __name__=='__main__':unittest.main()
