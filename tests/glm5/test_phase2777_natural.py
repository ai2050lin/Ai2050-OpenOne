import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2777_materials as M
import phase2777_natural_loop as N
import phase2777_materials_v2 as M2
import phase2777_materials_v3 as M3


class NaturalMeasurement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.builder=M.Builder(AutoTokenizer.from_pretrained(N.W.MODEL,local_files_only=True))

    def test_all_frames_have_exact_active_lengths(self):
        for frame in range(4):
            for length in [384,512]:
                rows=self.builder.pair(2031,frame,length,'test')
                self.assertEqual([len(r['token_ids']) for r in rows],[length,length])
                self.assertEqual(rows[0]['matching']['common_neutral_prefix'],rows[1]['matching']['common_neutral_prefix'])
                self.assertEqual(rows[0]['fact'],rows[1]['fact'])
                self.assertTrue(all(r['text'].endswith(M.INSTRUCTION) for r in rows))

    def test_unseen_frames_change_sentence_constructions(self):
        a=self.builder.pair(2031,0,384,'test')[0]
        b=self.builder.pair(2031,2,384,'test')[0]
        self.assertNotEqual(a['text'],b['text'])
        self.assertIn('由',b['text'])

    def test_reference_traps_ignore_quoted_markers(self):
        for r in M.references():self.assertEqual(M.rubric(r['text'])['label'],r['label'])

    def test_single_opening_marker_not_style_control(self):
        s='说白了，本文根据记录进行核算。上述数量应当保持一致，因此需要依据现有事实形成明确说明。'
        self.assertNotEqual(M.rubric(s)['label'],'colloq')

    def test_tiny_output_is_not_qualified(self):
        self.assertFalse(M.rubric('咱们说白了吧。')['adequate'])

    def test_chinese_and_decimal_numbers_equivalent(self):
        for s,n in [('一百零三',103),('两百一十二',212),('四百三十六',436),('436',436)]:
            self.assertEqual(M.chinese_number(s),n)

    def test_content_requires_item_and_correct_quantity(self):
        f={'item':'文件夹','initial':120,'added':20,'removed':10,'quantity':130}
        self.assertTrue(M.content('文件夹现在剩下一百三十份。',f)['pass'])
        self.assertFalse(M.content('文件夹现在剩下140份。',f)['pass'])
        self.assertFalse(M.content('现在剩下130份。',f)['pass'])

    def test_family_cluster_does_not_count_frames_as_independent(self):
        rows=[{'family_id':f'f{i}','register':r} for i in range(3) for _ in range(2) for r in ['acad','colloq']]
        v=N.family_mean(rows,list(range(12)),'acad')
        self.assertEqual(len(v),3)
        np.testing.assert_allclose(v,[1,5,9])

    def test_weak_ni_is_inconclusive_not_preserved(self):
        self.assertEqual(N.ni([True]*10,[True]*10,np.ones(10,bool))['status'],'inconclusive')

    def test_strong_ni_and_clear_damage(self):
        self.assertEqual(N.ni([True]*2400,[True]*2400,np.ones(2400,bool))['status'],'noninferior')
        self.assertEqual(N.ni([True]*2400,[False]*2400,np.ones(2400,bool))['status'],'inferior')

    def test_cluster_null_control_has_no_positive_ci(self):
        self.assertEqual(N.interval(np.zeros(128))['ci'],[0.,0.])

    def test_explicit_request_pair_has_identical_source_facts(self):
        b=M2.Builder(self.builder.tok)
        for frame in range(4):
            for length in [384,512]:
                a,c=b.pair(2201,frame,length,'test')
                self.assertEqual(a['tokens'],length);self.assertEqual(c['tokens'],length)
                self.assertEqual(a['matching']['common_fact_source'],c['matching']['common_fact_source'])
                self.assertEqual(a['text'].replace(a['matching']['register_request'],''),
                    c['text'].replace(c['matching']['register_request'],''))
                self.assertTrue(a['fact']['verified_result_supplied'])

    def test_generic_each_one_is_not_inventory_regression(self):
        f={'item':'记录册','initial':320,'added':22,'removed':51,'quantity':291}
        self.assertTrue(M2.content('每一份登记册都已核验，现在剩余291份。',f)['pass'])

    def test_prospective_v3_pairs_only_change_the_equal_length_style_request(self):
        b=M3.Builder(self.builder.tok)
        for frame in [2,3]:
            for length in [384,512]:
                a,c=b.pair(3501,frame,length,'software_test_only')
                self.assertEqual(a['tokens'],length);self.assertEqual(c['tokens'],length)
                self.assertEqual(a['text'].replace(a['matching']['register_request'],''),
                    c['text'].replace(c['matching']['register_request'],''))
                self.assertTrue(a['text'].endswith(a['matching']['register_request']))

    def test_natural_rubric_requires_multiple_linguistic_families(self):
        mild=next(x for x in M3.references() if x['id']=='mild-factual-C')
        self.assertEqual(M3.rubric(mild['text'])['label'],'colloq')
        self.assertTrue(M3.rubric(mild['text'])['adequate'])
        self.assertNotEqual(M3.rubric('你看，根据记录，本次结余已经核验。因此上述结果可以作为正式说明。')['label'],'colloq')


if __name__=='__main__':unittest.main()
