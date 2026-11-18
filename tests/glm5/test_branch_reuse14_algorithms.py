"""Isolated synthetic engineering checks, never model research evidence."""
import unittest
import os
import numpy as np
from threadpoolctl import threadpool_limits
import branch_reuse14_analysis as A


class Algorithms(unittest.TestCase):
    def test_native_predictor_recovers_planted_map_but_not_unrelated_inputs(self):
        rng=np.random.default_rng(40);x=rng.normal(size=(400,20));w=rng.normal(size=(20,6));y=x@w+.01*rng.normal(size=(400,6))
        with threadpool_limits(limits=2):
            good=A.fit(x[:240],y[:240],alpha=1.)
            unrelated=rng.normal(size=x.shape);bad=A.fit(unrelated[:240],y[:240],alpha=100.)
        base=np.mean((y[240:]-y[:240].mean(0))**2)
        self.assertLess(np.mean((A.predict(good,x[240:])-y[240:])**2)/base,.01)
        self.assertGreater(np.mean((A.predict(bad,unrelated[240:])-y[240:])**2)/base,.8)

    def test_high_dimensional_dual_map_matches_primal(self):
        rng=np.random.default_rng(41);x=rng.normal(size=(40,90));y=rng.normal(size=(40,70));xx=rng.normal(size=(12,90))
        with threadpool_limits(limits=2):
            m=A.fit(x,y,alpha=10.);z=(x-m['xm'])/m['xs']
            primal=np.linalg.solve(z.T@z+10*np.eye(90),z.T@((y-m['ym'])/m['ys']))
        expected=((xx-m['xm'])/m['xs']@primal)*m['ys']+m['ym']
        np.testing.assert_allclose(A.predict(m,xx),expected,atol=1e-10,rtol=1e-10)

    def test_family_pairing_removes_title_offset_without_multiplying_n(self):
        rows=[];values=[]
        for i in range(30):
          for role in ['rule','record']:
           for title in [0,1]:
            rows.append(dict(family=str(i),branch='number',role=role,title=title))
            values.append(i+title*100+(2 if role=='rule' else 0))
        delta=A.paired(rows,np.array(values))
        self.assertEqual(len(delta),30);np.testing.assert_array_equal(delta,np.full(30,2))
        folds=A.fold_ids(rows)
        self.assertTrue(all(len(set(folds[i:i+4]))==1 for i in range(0,len(rows),4)))

    def test_recursive_prediction_does_not_read_later_truth(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(300,8));w1=rng.normal(size=(8,6));w2=rng.normal(size=(6,3))
        y=x@w1;z=y@w2
        with threadpool_limits(limits=2):m1=A.fit(x[:200],y[:200],alpha=1.);m2=A.fit(y[:200],z[:200],alpha=1.)
        pred=A.predict(m2,A.predict(m1,x[200:]))
        self.assertLess(np.mean((pred-z[200:])**2)/np.mean(z[200:]**2),.01)
        # Corrupt the seed, which a true free recursion must depend on.
        bad=A.predict(m2,A.predict(m1,rng.normal(size=x[200:].shape)))
        self.assertGreater(np.mean((bad-z[200:])**2)/np.mean(z[200:]**2),.8)

    @unittest.skipUnless(os.environ.get('BR14_CUDA_ENGINEERING')=='1','Explicit isolated CUDA engineering check')
    def test_cuda_native_edit_is_local_and_noop_is_exact(self):
        import branch_reuse14 as B
        import branch_reuse14_collect as C
        import branch_reuse14_intervene as I
        B.configure();C.bind()
        with B.GpuLease():
          import torch
          with torch.inference_mode():
            torch.manual_seed(44)
            layer=torch.nn.Linear(64,32,bias=False).to('cuda',torch.bfloat16)
            values=torch.randn(2,256,64,device='cuda',dtype=torch.bfloat16);output=layer(values)
            rows=[dict(answer_pos=41,content_start=4,content_end=30),dict(answer_pos=52,content_start=8,content_end=33)]
            noop=I.final_position(layer,(values,),output,rows,[2,7],1.)
            self.assertTrue(torch.equal(noop,output))
            edited=I.final_position(layer,(values,),output,rows,[2,7],0.)
            manual=values.clone();manual[0,41,[2,7]]=0;manual[1,52,[2,7]]=0
            self.assertTrue(torch.equal(edited,layer(manual)))
            mask=torch.ones((2,256),dtype=torch.bool,device='cuda');mask[0,41]=False;mask[1,52]=False
            self.assertTrue(torch.equal(edited[mask],output[mask]))
            body,stats=I.perturb(layer,(values,),output,rows,[2,7],0.)
            self.assertTrue(torch.equal(body[~stats['region']],output[~stats['region']]))


if __name__=='__main__':unittest.main()
