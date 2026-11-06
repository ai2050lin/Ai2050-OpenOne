"""Tiny independent software fixtures, never research measurements."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from native_mlp_long_features import prefix_features,mcnemar
from native_mlp_long_validate import install
class NativeMlpTest(unittest.TestCase):
    def test_prefix_does_not_use_current_or_future_tokens(self):
        rng=np.random.default_rng(4);cat=rng.normal(size=(10,12)).astype(np.float32);a=np.arange(10);b=a.copy();b[5:]=0
        np.testing.assert_equal(prefix_features(a,cat)[:6],prefix_features(b,cat)[:6])
        self.assertFalse(np.array_equal(prefix_features(a,cat)[6],prefix_features(b,cat)[6]))
    def test_zero_prefix_and_exact_previous_token(self):
        cat=np.eye(12,dtype=np.float32);x=prefix_features([2,4,6],cat)
        np.testing.assert_equal(x[0],np.zeros(96));np.testing.assert_equal(x[2,48:60],cat[4]);np.testing.assert_equal(x[2,60:72],cat[2])
    def test_paired_exact_test_and_no_discordance(self):
        self.assertEqual(mcnemar([True]*20,[True]*20)['p'],1)
        m=mcnemar([False]*20,[True]*20);self.assertEqual(m['delta'],1);self.assertLess(m['p'],.00001)
        self.assertGreater(mcnemar([True]*20,[False]*20)['p'],.99)
    def test_intervention_changes_only_declared_query_and_restores(self):
        import torch
        from types import SimpleNamespace
        down=torch.nn.Identity();model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=down))]))
        x=torch.arange(24,dtype=torch.float32).reshape(1,4,6);handles=install(model,[[0,2]],.5,1);y=down(x)
        expected=x.clone();expected[:,1,2]*=.5;self.assertTrue(torch.equal(y,expected));self.assertTrue(torch.equal(x,torch.arange(24).reshape(1,4,6)))
        for h in handles:h.remove()
        self.assertTrue(torch.equal(down(x),x))
    def test_selected_pairs_are_document_disjoint_same_token_and_position(self):
        import native_mlp_long_materials as P
        d=P.read(P.OUT/'documents.json');raw=np.load(P.OUT/'input-tokens.npy',mmap_mode='r')
        for split,pp in d['pairs'].items():
            seen=set()
            for p in pp:
                self.assertGreaterEqual(p['position'],512);self.assertEqual(raw[p['a'],p['position']],raw[p['b'],p['position']])
                self.assertNotIn(p['a'],seen);self.assertNotIn(p['b'],seen);seen.update([p['a'],p['b']])
                self.assertEqual(d['docs'][p['a']]['split'],split);self.assertEqual(d['docs'][p['b']]['split'],split)
            self.assertEqual(len(seen),512)
    def test_context_endpoint_has_explicit_natural_scope_and_no_fake_contrast(self):
        from unittest.mock import patch
        from fastapi import HTTPException
        import server.mechanism_trace_service as service
        with patch.object(service,'context',return_value={'study_type':'native_mlp_long'}):
            with self.assertRaises(HTTPException) as e:service.profile('NMLP-LONG-20261104-v1',0,'mlp')
            self.assertEqual(e.exception.status_code,404)
if __name__=='__main__':unittest.main()
