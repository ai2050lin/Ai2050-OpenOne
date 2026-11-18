"""Isolated algebra/position engineering checks, never model evidence."""
import unittest
import numpy as np
from b36_transition14_native_math import partitions,subset_mask,mobius

class Tests(unittest.TestCase):
    def test_partition_disjoint_complete_and_padding_untouched(self):
        p=partitions(5,9,12);self.assertEqual(p.sum(axis=1).tolist(),[5,3,1])
        self.assertEqual(subset_mask(p,7).tolist(),[True]*9+[False]*3)
        self.assertTrue(np.array_equal(subset_mask(p,3),p[0]|p[1]));self.assertFalse(subset_mask(p,0).any())
    def test_mobius_planted_interactions_and_additive_control(self):
        coefficients=np.array([1.,2.,3.,.5,-1.,0.,0.,.2]);values=np.array([sum(coefficients[t] for t in range(8) if t&s==t) for s in range(8)])
        np.testing.assert_allclose(mobius(values),coefficients,atol=1e-12)
        additive=np.array([1+2*bool(s&1)+3*bool(s&2)-bool(s&4) for s in range(8)],dtype=float)
        np.testing.assert_allclose(mobius(additive)[[3,5,6,7]],0,atol=1e-12)

if __name__=='__main__':unittest.main(verbosity=2)
