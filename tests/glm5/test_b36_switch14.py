"""CPU engineering tests; synthetic strings are not model evidence."""
import unittest
import b36_switch14_design as D


class Tests(unittest.TestCase):
    def test_prefix_visibility(self):
        self.assertEqual(D.candidate_sites('Steps: 31'),['after_first'])
        self.assertEqual(D.candidate_sites('Steps: 31,32'),['after_second'])
        self.assertEqual(D.candidate_sites('Steps: 31,32,33'),[])
        self.assertEqual(D.candidate_sites('Steps: 31,32\nValue: '),['report'])
        self.assertEqual(D.candidate_sites('Steps: 31,32\nValue: 3'),[])
        self.assertFalse(D.eligible('after_first','2'))
        self.assertFalse(D.eligible('report','\n'))
        self.assertTrue(D.eligible('after_second','   \n'))

    def test_classes(self):
        self.assertEqual(D.token_class(' ,'),'comma')
        self.assertEqual(D.token_class(' \nValue'),'newline')
        self.assertEqual(D.token_class(' 9'),'digit9')
        self.assertEqual(D.token_class('answer9'),'other')
        for depth,expected in [(2,'newline'),(3,'comma')]:
            self.assertEqual(D.target_class(dict(depth=depth,truth=[31,42,53][:depth]),'after_second'),expected)

    def test_new_visible_families(self):
        ff,paths=D.families();self.assertEqual(len(ff),96)
        self.assertEqual(len({D.S.digest(g['ops']) for g in ff}),96)
        self.assertEqual(len({D.S.digest([g['start'],g['ops'][:1]]) for g in ff}),96)
        self.assertTrue(all(len(g['ops'])==3 and min(g['truth'])>=10 and max(g['truth'])<=99 for g in ff))

    def test_readonly_tail_cache(self):
        import torch
        from b36_switch14_paths import ReadOnlyPast
        old=torch.arange(6).reshape(1,1,3,2);saved=old.clone();new=torch.ones((1,1,1,2),dtype=torch.int64)*9
        cache=ReadOnlyPast(36,(old,old+1));k,v=cache.update(new,new+1,36)
        self.assertTrue(torch.equal(old,saved));self.assertTrue(torch.equal(k[:,:,:3],old));self.assertTrue(torch.equal(k[:,:,-1:],new))
        k[:,:,:3]=0;self.assertTrue(torch.equal(old,saved))
        self.assertTrue(torch.equal(cache.update(new,new+1,36)[0][:,:,:3],saved))
        with self.assertRaises(AssertionError):cache.update(new,new,37)

    def test_lossless_logit_encoding_and_holm(self):
        import numpy as np
        from b36_switch14_analyze import holm,decode
        x=np.asarray([[16256,16384],[16256,16448]],dtype=np.uint16);encoded=x.copy();encoded[1:]^=encoded[0];encoded[1:]^=encoded[0]
        self.assertTrue(np.array_equal(x,encoded));self.assertTrue(np.array_equal(decode(x),np.asarray([[1,2],[1,3]],np.float32)))
        self.assertTrue(np.allclose(holm([.01,.04,.03]),[.03,.06,.06]))


if __name__=='__main__':unittest.main()
