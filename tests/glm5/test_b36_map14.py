import unittest,operator,collections
import b36_map14 as M
S=M.S
class Tests(unittest.TestCase):
    def test_materials(self):
        p=S.read(S.OUT/'plan.json');self.assertEqual(len(p['families']),96);self.assertEqual(len(p['rows']),672)
        self.assertEqual(len({S.digest([g['start'],g['ops'][:1]]) for g in p['families']}),96)
        for g in p['families']:
            x=g['start'];truth=[]
            for kind,k in g['ops'][:g['depth']]:x={'add':operator.add,'sub':operator.sub,'mul':operator.mul}[kind](x,k);truth.append(x)
            self.assertEqual(truth,g['truth']);self.assertEqual(g['values'][g['slot']-1],x)
        counts=collections.Counter((g['depth'],g['slot']) for g in p['families']);self.assertEqual(set(counts.values()),{12})
    def test_rotation_and_bypass(self):
        p=S.read(S.OUT/'plan.json');groups=collections.defaultdict(dict)
        for r in p['rows']:groups[r['family']][r['arm']]=r
        for group in groups.values():
            self.assertEqual(len({tuple(r['values']) for r in group.values()}),1)
            self.assertEqual(len({r['slot'] for r in group.values()}),1)
            self.assertNotEqual(group['integrated0']['code'],group['integrated2']['code'])
            for arm in M.ARMS:
                r=group[arm];self.assertEqual(r['codes'][r['slot']-1],r['code'])
    def test_scoring_and_failure_separation(self):
        r=dict(arm='integrated0',depth=2,truth=[25,21],slot=2,values=[22,21,25,26],codes=list('ABCD'),code='B')
        score=lambda s,e=True:M.evaluate(s,r,e)['metrics']
        self.assertTrue(score('Steps: 25,21\nValue: 21\nSlot: 2\nCode: B<|im_end|>')['joint'])
        q=score('Steps: 25,21\nValue: 21\nSlot: 4\nCode: D');self.assertTrue(q['value_correct']);self.assertFalse(q['slot_correct']);self.assertTrue(q['code_consistent_with_generated_slot'])
        q=score('Steps: 25,21\nValue: 21\nSlot: 2\nCode: D');self.assertTrue(q['slot_correct']);self.assertFalse(q['code_correct']);self.assertFalse(q['code_consistent_with_generated_slot'])
        self.assertFalse(score('Steps: 25,21\nValue: 21\nSlot: 2\nCode: B',False)['joint'])
        self.assertFalse(score('The answer is B')['format_correct'])
if __name__=='__main__':unittest.main(verbosity=2)
