import unittest,operator
import b36_generated14 as G
S=G.S
class Tests(unittest.TestCase):
    def test_components(self):
        p=lambda text,eos=True:G.parse(text,[4,7],[7,8,9,10],'A',eos)
        self.assertTrue(p('Steps: 4, 7\nFinal: A<|im_end|>')['joint'])
        self.assertTrue(p('Steps: 5, 7\nFinal: A')['calculation_error'])
        self.assertTrue(p('Steps: 4, 7\nFinal: B')['correct_steps_wrong_final'])
        self.assertFalse(p('Steps: 4, 7\nFinal: A',False)['joint'])
        self.assertFalse(p('I think the answer is A')['steps_parseable'])
        self.assertFalse(p('Explanation\nSteps: 4, 7\nFinal: A')['format_correct'])
    def test_new_material_truth_and_family_partition(self):
        plan=S.read(S.OUT/'plan.json');groups=plan['families'];self.assertEqual(len({g['id'] for g in groups}),108)
        prior=S.read(G.OLD/'arithmetic-families.json');oldids=set(prior['diagnostic_family_ids'])|{g['id'] for g in S.read(G.OLD/'expanded-plan.json')['families']}
        self.assertFalse(oldids.intersection(g['id'] for g in groups))
        for g in groups:
            self.assertGreaterEqual(g['start'],10);x=g['start'];h=[]
            for op,k in g['ops']:x={'add':operator.add,'sub':operator.sub,'mul':operator.mul}[op](x,k);h.append(x)
            self.assertEqual(h,g['truth'])
        self.assertEqual({r['family'] for r in plan['rows']},{g['id'] for g in groups[:12]})
    def test_code_truth(self):
        rows=S.read(S.OUT/'plan.json')['rows']
        for r in rows:
            if r['format']=='direct':self.assertEqual(r['expected'],str(r['truth'][-1]))
            else:self.assertEqual(r['values']['ABCD'.index(r['expected'])],r['truth'][-1])
if __name__=='__main__':unittest.main(verbosity=2)
