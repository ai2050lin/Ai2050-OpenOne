"""Engineering-only admission and tie-handling checks."""
import unittest
from unittest.mock import patch
import b36_composition14 as B
import b36_composition14_collect as C
import b36_composition14_analysis as A


class AdmissionTests(unittest.TestCase):
    def test_ties_fail_even_if_gold_is_first(self):
        row=dict(branch='number',depth=1,mode='program',title=0,split='confirm',gold=0)
        self.assertEqual(C.behavior([row],[[2.,2.,1.]])['overall_accuracy'],0.)

    def test_one_failed_title_excludes_whole_stratum(self):
        rows=[dict(id=f'{depth}:{title}',branch='number',depth=depth,mode='program',title=title) for depth in [1,2] for title in [0,1]]
        cells=[dict(branch='number',depth=depth,mode='program',title=title,passed=not(depth==2 and title==1)) for depth in [1,2] for title in [0,1]]
        pilot=[dict(c,passed=True) for c in cells];saved=[]
        with patch.object(B,'configure'),patch.object(A,'admit'),patch.object(A,'load',return_value=(rows,dict(behavior=dict(cells=cells)))),patch.object(B,'read',return_value=dict(behavior=dict(cells=pilot))),patch.object(B,'file_sha',return_value='engineering-hash'),patch.object(B,'save',side_effect=lambda n,d:saved.append(d)):
            A.scope('confirm')
        self.assertEqual(saved[0]['eligible_row_ids'],['1:0','1:1']);self.assertFalse(saved[0]['joint_passed'])


if __name__=='__main__':unittest.main(verbosity=2)
