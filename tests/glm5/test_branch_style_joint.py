import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import branch_style_joint as C
import branch_style_analysis as A

class JointTest(unittest.TestCase):
    def test_family_split_and_factorial(self):
        spec=C.build_spec();self.assertEqual(C.grammar.validate(spec),[])
        corpus=C.grammar.generate(spec,include_canary=False)
        by={}
        for r in corpus['items']:by.setdefault(r['filler_id'],set()).add(r['split'])
        self.assertEqual(len(by),512);self.assertTrue(all(len(v)==1 for v in by.values()))
        self.assertEqual(len(corpus['items']),6144)

    def test_style_content_is_unchanged(self):
        spec=C.build_spec()
        for i in range(0,len(spec['frames']),4):
            a,b=spec['frames'][i],spec['frames'][i+2]
            self.assertEqual(a['text'].split('\n',1)[1],b['text'].split('\n',1)[1])

    def test_native_bit_roundtrip(self):
        import torch
        x=torch.tensor([-3.,-.1,0.,.25,50.],dtype=torch.bfloat16)
        np.testing.assert_array_equal(C.unbits(C.bits(x)),x.float().numpy())

    def test_ridge_planted_relation_and_shuffled_control(self):
        rng=np.random.default_rng(28);x=rng.normal(size=(250,8));w=rng.normal(size=(8,4));y=x@w
        good=A.ridge_fit(x[:150],y[:150],.01)
        bad=A.ridge_fit(x[:150],y[rng.permutation(150)],.01)
        self.assertLess(np.mean((A.ridge_predict(good,x[150:])-y[150:])**2),1e-5)
        self.assertGreater(np.mean((A.ridge_predict(bad,x[150:])-y[150:])**2),1.)

    def test_joint_interaction_beats_additive_on_planted_data(self):
        rng=np.random.default_rng(29);x=rng.normal(size=(600,1));t=rng.integers(0,2,(600,1));y=x*(2*t-1)
        additive=np.c_[x,t];joint=np.c_[x,t,x*t]
        b=A.ridge_fit(additive[:400],y[:400],.01);m=A.ridge_fit(joint[:400],y[:400],.01)
        self.assertGreater(np.mean((A.ridge_predict(b,additive[400:])-y[400:])**2),.5)
        self.assertLess(np.mean((A.ridge_predict(m,joint[400:])-y[400:])**2),1e-5)

    def test_family_bootstrap_counts_families_not_rows(self):
        family=[str(i) for i in range(20) for _ in range(12)]
        x=np.repeat(np.linspace(.5,1.,20),12);zero=np.zeros_like(x)
        r=A.bootstrap_difference(x,zero,family)
        self.assertEqual(r['families'],20);self.assertGreater(r['ci_97p5'][0],0)

    def test_kl_identity_and_direction(self):
        x=np.array([1.,2.,3.]);self.assertAlmostEqual(float(A.kl(x,x)),0.)
        self.assertGreater(float(A.kl(x,-x)),0.)

    def test_full_scorer_matches_known_bigram_probabilities(self):
        import torch
        from types import SimpleNamespace
        from branch_style_score_audit import score_full
        transition=torch.arange(64,dtype=torch.float32).reshape(8,8)%11
        class Model:
            device='cpu'
            def __call__(self,input_ids,logits_to_keep,**kwargs):
                return SimpleNamespace(logits=transition[input_ids][:,-logits_to_keep:])
        engine=SimpleNamespace(model=Model(),tok=SimpleNamespace(pad_token_id=0))
        rows=[{'tokens':[1,2], 'candidate_tokens':[[3],[4,5],[2,3,4],[6,7]]}, {'tokens':[4,2,1,3], 'candidate_tokens':[[1,2],[3],[4,5,6],[7,1]]}]
        got=score_full(engine,rows);lp=transition.log_softmax(-1)
        expected=[]
        for row in rows:
            scores=[]
            for candidate in row['candidate_tokens']:
                source=[row['tokens'][-1]]+candidate[:-1]
                scores.append(sum(float(lp[a,b]) for a,b in zip(source,candidate)))
            expected.append(scores)
        np.testing.assert_allclose(got,expected,atol=2e-6)

    def test_free_audit_distinguishes_orange_noun_from_color(self):
        from branch_style_freeall import inspect
        row={'task':1,'slots':{'X':'橙子','C':'黄色','EC':'yellow'}}
        r=inspect(row,'The orange is yellow.')
        self.assertTrue(r['joint_color_language'])

    def test_color_knowledge_does_not_imply_translation_compliance(self):
        from branch_style_freeall import inspect
        row={'task':1,'slots':{'X':'苹果','C':'蓝色','EC':'blue'}}
        r=inspect(row,'苹果是蓝色的。')
        self.assertFalse(r['output_language_compliant'])
        self.assertFalse(r['joint_color_language'])

    def test_chinese_audit_uses_queried_subject(self):
        from branch_style_freeall import inspect
        row={'task':0,'slots':{'X':'苹果','C':'绿色','EC':'green'}}
        self.assertTrue(inspect(row,'梨是蓝色的，苹果是绿色的。')['color_correct'])

if __name__=='__main__':unittest.main()
