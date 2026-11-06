"""Statistical false-certification and edit-operator controls for W6 v2."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2773_w6_loop as W
from phase2773_w6_paired import gap
from phase2773_resume import ids_only


class W6StatisticalContracts(unittest.TestCase):
    def test_no_change_large_sample_noninferior(self):
        d=W.paired_interval([True]*2200,[True]*2200)
        self.assertEqual(d['status'],'noninferior')
        self.assertLess(d['loss_interval'][1],.02)

    def test_no_change_small_sample_not_proof(self):
        self.assertEqual(W.paired_interval([True]*10,[True]*10)['status'],'inconclusive')

    def test_clear_regression_rejected(self):
        self.assertEqual(W.paired_interval([True]*100,[False]*100)['status'],'inferior')

    def test_unverified_labels_excluded(self):
        d=W.paired_interval([True]*2210,[True]*2200+[False]*10,[True]*2200+[False]*10)
        self.assertEqual(d['n'],2200)
        self.assertEqual(d['losses'],0)

    def test_gains_do_not_count_as_damage(self):
        d=W.paired_interval([False]*2200,[True]*2200)
        self.assertEqual(d['status'],'noninferior')
        self.assertLess(d['net_loss'],0)

    def test_empty_not_tested(self):
        self.assertEqual(W.paired_interval([],[])['status'],'not_tested')

    def test_parameter_operator_local_identity(self):
        rng=np.random.default_rng(17);matrix=rng.normal(size=(7,11));u=rng.normal(size=7);u/=np.linalg.norm(u)
        mask=np.zeros(11);mask[[2,4,9]]=1;activation=rng.normal(size=11)
        for alpha in [0,1,6,-1]:
            delta=-alpha*np.outer(u,(u@matrix)*mask)
            np.testing.assert_allclose(delta@activation,-alpha*u*((u@matrix)*mask@activation),atol=1e-12)
            self.assertLessEqual(np.linalg.matrix_rank(delta),1)
            np.testing.assert_allclose(delta[:,mask==0],0)
        reflected=matrix-6*np.outer(u,(u@matrix)*mask)
        np.testing.assert_allclose((u@reflected)[mask==1],-5*(u@matrix)[mask==1])

    def test_column_not_coordinate_semantics(self):
        a,f=W.render(93,'acad');c,g=W.render(93,'colloq')
        self.assertEqual(f,g)
        self.assertNotEqual(a,c)
        self.assertIn(f['item'],a)
        self.assertIn(f['item'],c)

    def test_bootstrap_negative_control(self):
        d=W.bootstrap_mean([0]*160)
        self.assertEqual(d['ci95'],[0.,0.])
        self.assertFalse(d['ci95'][0]>0)

    def test_duplicate_text_normalization(self):
        a={'question':'a b\n','choices':['x','y','z','w']}
        b={'question':'ab','choices':['x','y','z','w']}
        self.assertEqual(W.digest_question(a),W.digest_question(b))

    def test_paired_readout_cancels_shared_answer_prior(self):
        s=np.array([[-30.,-10.],[-23.,-10.],[-28.,-10.],[-22.,-10.]])
        np.testing.assert_allclose(gap(s),[7,6])
        shifted=s.copy();shifted[:,0]-=50
        np.testing.assert_allclose(gap(shifted),gap(s))

    def test_compatibility_repair_preserves_input_ids(self):
        ids=[1,2,3]
        self.assertIs(ids_only({'input_ids':ids,'attention_mask':[1,1,1]}),ids)
        self.assertIs(ids_only(ids),ids)


if __name__=='__main__':unittest.main()
