import unittest
import numpy as np
import b36_semantic14 as S
from b36_semantic14_materials import canonical,history

class MaterialTests(unittest.TestCase):
    def test_family_is_invariant_to_relabelling_and_operator_swap(self):
        a=[1,2,3,4,5,0];b=[1,0,2,4,3,5];q=[3,0,4,5,1,2];inv=np.argsort(q)
        aa=[q[a[inv[i]]] for i in range(6)];bb=[q[b[inv[i]]] for i in range(6)]
        self.assertEqual(canonical(a,b),canonical(bb,aa))
        self.assertNotEqual(canonical(a,b),canonical(a,list(range(6))))
    def test_truth_matches_independent_matrix_composition(self):
        data=S.read(S.OUT/'graph-families.json')['graphs']
        for g in data:
            x=np.eye(6)[g['start']];states=[g['start']]
            for op in g['sequence']:
                mat=np.eye(6)[g[op]];x=x@mat;states.append(int(x.argmax()))
            self.assertEqual(history(g,g['sequence'],g['start']),states)
    def test_code_rotation_preserves_semantics_and_causal_prefix(self):
        rows=S.read(S.OUT/'pilot-plan.json')['rows'];groups={}
        for r in rows:groups.setdefault((r['family'],r['branch'],r['depth'],r['frame']),[]).append(r)
        for rr in groups.values():
            self.assertEqual(len({tuple(r['intermediate_states']) for r in rr}),1)
            self.assertEqual(len({tuple(r['prefix_tokens_before_format']) for r in rr}),1)
            code=[r for r in rr if r['format'].startswith('code')]
            self.assertEqual({r['expected_answer'] for r in code},set('ABCDEF'))
            self.assertTrue(all(r['content_end']<r['answer_pos'] for r in rr))

if __name__=='__main__':unittest.main(verbosity=2)
