"""Isolated engineering negatives for scientific-admission checks, no model data."""
import copy
import itertools
import unittest
from conditional_lookup_stage_audit import verify_rows


def engineering_fixture():
    rows, scores = [], []
    for a, b, cue, late in itertools.product((0, 1), repeat=4):
        gold = 2 * cue + (a, b)[cue]
        identity = f'engineering-{a}{b}{cue}{late}'
        rows.append(dict(id=identity, family=0, r0=a, r1=b, b=cue, late=late,
                         target_record=gold, choice_ids=[10, 11, 12, 13],
                         gold_id=10 + gold, tokens=[1, 2, 20 + cue], entity_pos=1))
        values = [0., 0., 0., 0.]
        values[gold] = 2.
        scores.append(dict(id=identity, scores=values, gold_margin=2.,
                           top1=10 + gold, top1_correct=True, forced_correct=True))
    return rows, scores


class AdmissionNegatives(unittest.TestCase):
    def test_complete_balanced_fixture(self):
        rows, scores = engineering_fixture()
        self.assertEqual(len(verify_rows(rows, scores, 1)), 16)

    def test_missing_cell_is_not_partial_success(self):
        rows, scores = engineering_fixture()
        with self.assertRaises(AssertionError):
            verify_rows(rows[:-1], scores[:-1], 1)

    def test_duplicate_condition_rejected(self):
        rows, scores = engineering_fixture()
        rows[-1] = copy.deepcopy(rows[0])
        scores[-1] = copy.deepcopy(scores[0])
        rows[-1]['id'] = scores[-1]['id'] = 'distinct-id-same-cell'
        with self.assertRaises(AssertionError):
            verify_rows(rows, scores, 1)

    def test_future_cue_leak_rejected(self):
        rows, scores = engineering_fixture()
        for row in rows:
            if row['late'] and row['b']:
                row['tokens'][1] = 99
        with self.assertRaisesRegex(AssertionError, 'future cue'):
            verify_rows(rows, scores, 1)

    def test_score_reordering_rejected(self):
        rows, scores = engineering_fixture()
        scores.reverse()
        with self.assertRaises(AssertionError):
            verify_rows(rows, scores, 1)

    def test_wrong_margin_rejected(self):
        rows, scores = engineering_fixture()
        scores[0]['gold_margin'] = 3.
        with self.assertRaises(AssertionError):
            verify_rows(rows, scores, 1)


if __name__ == '__main__':
    unittest.main()
