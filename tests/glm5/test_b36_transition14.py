"""Engineering tests, not model evidence."""
import unittest,re,collections,operator
import b36_transition14 as M
S=M.S

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ff,_=M.families()
    def test_visible_program_novelty_and_truth(self):
        old,_=M.prior_materials();oldrecipes={S.digest(g['ops'][:d]) for g in old for d in [2,3]};oldprefixes={S.digest([g['start'],g['ops'][:d]]) for g in old for d in [1,2,3]}
        self.assertEqual(len(self.ff),96);self.assertEqual(len({S.digest(g['ops']) for g in self.ff}),96)
        self.assertEqual(set(collections.Counter((g['depth'],g['slot']) for g in self.ff).values()),{12})
        for g in self.ff:
            self.assertNotIn(S.digest(g['ops']),oldrecipes)
            x=g['start'];truth=[]
            for d,(op,k) in enumerate(g['ops'],1):
                x={'add':operator.add,'sub':operator.sub,'mul':operator.mul}[op](x,k);truth.append(x)
                self.assertNotIn(S.digest([g['start'],g['ops'][:d]]),oldprefixes)
            self.assertEqual(truth,g['truth']);self.assertEqual(g['values'][g['slot']-1],truth[-1]);self.assertEqual(len(set(g['values'])),4)
            self.assertNotEqual(g['foil'],truth[-1]);self.assertEqual(len(str(g['foil'])),len(str(truth[-1])))
    def test_exact_table_control_and_field_controls(self):
        class FakeTokenizer:
            def apply_chat_template(self,items,**kwargs):return items[0]['content']
            def encode(self,s,**kwargs):return list(s.encode())
        g=self.ff[0];tok=FakeTokenizer();rows={a:M.prompt_for(g,a,tok) for a in M.ARMS}
        without=re.sub(r'\nOptions:\n(?:Row[^\n]+\n){3}Row[^\n]+','',rows['table']['prompt'])
        self.assertEqual(without,rows['no_table']['prompt'])
        self.assertEqual(rows['full']['tokens'][:20],rows['counted0']['tokens'][:20])
        self.assertNotEqual(rows['counted0']['code'],rows['counted2']['code'])
        self.assertEqual(rows['counted0']['values'],rows['counted2']['values'])
        self.assertNotIn('Steps',rows['no_steps']['fields'])
    def test_scoring_extra_update_and_terminal_error(self):
        row=dict(start=13,ops=[['sub',4],['sub',4]],depth=2,truth=[9,5],fields=['Steps','Value','Slot','Code'],slot=1,values=[5,6,9,13],codes=list('ABCD'),code='A')
        q=M.evaluate('Steps: 9,5,1\nValue: 1\nSlot: 1\nCode: A<|im_end|>',row,True)
        self.assertEqual(q['step_pattern'],'appended_repeated_last_update');self.assertFalse(q['metrics']['value_correct']);self.assertTrue(q['metrics']['code_correct']);self.assertFalse(q['metrics']['joint'])
        q=M.evaluate('Steps: 9,5\nValue: 5\nSlot: 1\nCode: A<|im_end|>',row,True);self.assertTrue(q['metrics']['joint'])
        self.assertFalse(M.evaluate('Steps: 9,5\nValue: 5\nSlot: 1\nCode: A',row,False)['metrics']['joint'])
        self.assertEqual(M.evaluate('Steps: 13,9,5\nValue: 5\nSlot: 1\nCode: A',row,True)['step_pattern'],'included_initial_assignment')
        self.assertIsNone(M.evaluate('Steps: -9,5\nValue: 5',row,True)['steps'])
    def test_history_cut_uses_actual_prefix_and_replacement(self):
        class CharacterTokenizer:
            def decode(self,ids,**kw):return ''.join(chr(v) for v in ids)
        text='Steps: 9, 5, 1\nValue: 1';p=M.history_prefix(text,list(map(ord,text)),2,CharacterTokenizer())
        self.assertEqual(p['text'],'Steps: 9, 5');self.assertEqual(p['values'],[9,5]);self.assertEqual(text[p['cut']:],', 1\nValue: 1')
        self.assertEqual(M.replace_steps(p['text'],[7,6]),'Steps: 7, 6')
        self.assertIsNone(M.history_prefix('Answer: 5',list(map(ord,'Answer: 5')),2,CharacterTokenizer()))

if __name__=='__main__':unittest.main(verbosity=2)
