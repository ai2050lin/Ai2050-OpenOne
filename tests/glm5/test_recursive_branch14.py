"""Isolated algorithm/material checks; artificial arrays are never evidence."""
import unittest
import numpy as np
from scipy.special import expit
import recursive_branch14 as R
import recursive_branch14_analysis as A


class BranchChecks(unittest.TestCase):
    def test_family_contrasts_cancel_title_and_background(self):
        rows=[];values=[]
        for i in range(40):
            for role in ['operative','quoted','inventory']:
                for lexical in (0,1):
                    rows.append(dict(family=str(i),role=role,lexical=lexical))
                    values.append(100*i+13*lexical+(2 if role!='inventory' else 0))
        self.assertTrue(np.allclose(A.paired(rows,values),2))
        self.assertTrue(np.allclose(A.paired(rows,values,'operative'),0))
        self.assertTrue(np.allclose(A.paired(rows,values,'title'),13))
        self.assertEqual(A.inference(A.paired(rows,values))['n_families'],40)

    def test_planted_joint_product_beats_direct_linear_and_noise(self):
        rng=np.random.default_rng(4917);x=rng.normal(size=(800,8))
        g=x[:,0]*1.5-x[:,1];v=x[:,2]*2+x[:,3];a=g*expit(g)*v
        y=np.column_stack([g,v,a]);m=A.regression(x[:400],y[:400]);p=A.predict(m,x[400:])
        direct=np.mean((p[:,2]-y[400:,2])**2);joint=np.mean((A.compose(p)[:,2]-y[400:,2])**2)
        self.assertLess(joint,direct*.01)
        noise=A.regression(rng.normal(size=(400,8)),y[:400])
        baseline=np.mean((y[400:]-y[:400].mean(0))**2)
        noise_error=np.mean((A.predict(noise,rng.normal(size=(400,8)))-y[400:])**2)
        self.assertGreater(noise_error,baseline*.8)

    def test_material_family_and_operation_balance(self):
        rows=R.materials();self.assertEqual(len(rows),6144)
        self.assertEqual(len(R.pilot_rows(rows)),384)
        self.assertTrue(all(rows[i]['split']=='discovery' for i in R.pilot_rows(rows)))
        seen={}
        for r in rows:
            seen.setdefault(r['family'],set()).add(r['split'])
            self.assertLess(r['content_end'],r['answer_pos'])
            self.assertEqual(r['tokens'][r['content_end']],624)
            self.assertEqual(len(r['choice_ids']),3)
            self.assertLess(len(r['tokens']),R.LENGTH)
        self.assertEqual(len(seen),1024)
        self.assertTrue(all(len(x)==1 for x in seen.values()))
        for h in R.CANDIDATES:
          for split in R.SPLITS:
            rr=[r for r in rows if r['hypothesis']==h and r['split']==split]
            for frame in {r['frame'] for r in rr}:
                counts=[len({r['family'] for r in rr if r['frame']==frame and r['metadata']['operation']==op}) for op in ['reverse','rotate']]
                self.assertEqual(counts[0],counts[1])

    def test_predicted_features_ignore_future_options(self):
        r=R.materials()[0];alt=dict(r,tokens=r['tokens'][:r['content_end']+1]+[999]*20)
        self.assertTrue(np.array_equal(A.lexical([r]),A.lexical([alt])))
        self.assertTrue(np.array_equal(A.semantic([r]),A.semantic([alt])))

    def test_rare_wrong_choice_fails_behavior_gate(self):
        rows=[dict(hypothesis='structure',split='discovery',role='operative',lexical=0,gold=0)]*10
        scores=np.array([[2.,1.,0.]]*7+[[0.,1.,2.]]*3)
        self.assertFalse(R.behavior(rows,scores)[0]['passed'])
        self.assertTrue(R.behavior(rows,np.array([[2.,1.,0.]]*10))[0]['passed'])


if __name__=='__main__':unittest.main()
