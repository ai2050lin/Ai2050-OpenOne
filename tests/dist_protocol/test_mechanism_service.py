# -*- coding: utf-8 -*-
"""机制案例与方法校准报告端点（server/mechanism_service.py）：索引摘要、判据归一、原样返回、非法 id 与坏文件，
以及数据包研究戳随报告变化。"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'deploy'))

CF = {
    'criteria': {'schema': 'conflict_case.v1', 'K': 20},
    'set': 'CF-v1', 'model': 'qwen3-4b', 'model_snapshot': 'abc', 'n_facts': 4, 'n_gated': 3, 'n_heldout': 1,
    'priority_table': {'plain': {'context_win_rate': 0.5, 'race_mean': [0.0, 1.0]}, 'negated': {'context_win_rate': 0.7}},
    'negated_control': {'context_win_rate': 0.7, 'copying_candidate': True},
    'heads': {'necessity': {'drop_top': 0.4, 'necessary': True}, 'sufficiency_last_position': {'keep_fraction': 0.3, 'sufficient': False}},
}
QK = {'criteria': {'schema': 'calib_qk.v1'}, 'checks': {'a': True, 'b': True}, 'passed': True, 'model_sha256': 'def'}


class MechanismServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.old = os.environ.get('AI2050_RESEARCH_DIR')
        os.environ['AI2050_RESEARCH_DIR'] = str(self.tmp)
        for sub, cid, doc in (('mechanism', 'CF-v1-qwen3-4b', CF), ('calibration', 'C2-qk-gpt2', QK)):
            d = self.tmp / sub / cid
            d.mkdir(parents=True)
            (d / 'report.json').write_text(json.dumps(doc), encoding='utf-8')
        bad = self.tmp / 'calibration' / 'C9-bad'
        bad.mkdir()
        (bad / 'report.json').write_text('{not json', encoding='utf-8')
        from server import mechanism_service
        self.M = mechanism_service

    def tearDown(self):
        if self.old is None:
            os.environ.pop('AI2050_RESEARCH_DIR', None)
        else:
            os.environ['AI2050_RESEARCH_DIR'] = self.old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_index_summaries(self):
        cases = {c['id']: c for c in self.M.index()['cases']}
        self.assertEqual(set(cases), {'CF-v1-qwen3-4b', 'C2-qk-gpt2', 'C9-bad'})
        cf = cases['CF-v1-qwen3-4b']
        self.assertEqual(cf['kind'], 'mechanism')
        self.assertEqual(cf['schema'], 'conflict_case.v1')
        # 判据归一：必要性过、充分性不过、否定对照判为复制候选 → 不通过
        self.assertEqual(cf['checks'], {'head_necessity': True, 'head_sufficiency_last_position': False,
                                        'negated_control_not_copying': False})
        self.assertFalse(cf['passed'])
        self.assertEqual(cf['void_checks'], {})
        self.assertIsNone(cf['fidelity_void'])
        self.assertEqual(cf['head_circuit_keep_fraction'], 0.3)
        self.assertEqual(cf['cues'], ['negated', 'plain'])
        self.assertEqual(cf['has'], {'priority_table': True, 'head_dla': False, 'head_total_effect': False,
                                     'neurons': False, 'head_circuit_fidelity': True})
        qk = cases['C2-qk-gpt2']
        self.assertTrue(qk['passed'])
        self.assertEqual(qk['model_snapshot'], 'def')
        self.assertIn('error', cases['C9-bad'])

    def test_v1_void_and_v2_checks(self):
        doc = {**CF, 'heads': {**CF['heads'], 'heldout_mean_d': -4.2},
               'v2': {'by_cue': {'plain': {'C_context_writers': {'delta': -1, 'effect': True}, 'M_memory_writers': {'delta': 1, 'effect': False},
                                           'heldout_mean_d': -4.2}},
                      'hierarchy': {'holds': True}}}
        (self.tmp / 'mechanism' / 'CF-v1-qwen3-4b' / 'report.json').write_text(json.dumps(doc), encoding='utf-8')
        cf = {c['id']: c for c in self.M.index()['cases']}['CF-v1-qwen3-4b']
        # 平均 d < 0：v1 必要性 / 充分性不计通过也不计未通过
        self.assertEqual(set(cf['void_checks']), {'head_necessity', 'head_sufficiency_last_position'})
        self.assertTrue(cf['fidelity_void'])
        self.assertEqual(cf['checks'], {'negated_control_not_copying': False, 'v2_plain_C_context_writers': True,
                                        'v2_plain_M_memory_writers': False, 'v2_hierarchy_U_to_C': True})

    def test_case_returns_report_verbatim(self):
        r = self.M.case('CF-v1-qwen3-4b')
        self.assertEqual(r['report'], CF)
        self.assertEqual(r['kind'], 'mechanism')

    def test_bad_ids(self):
        from fastapi import HTTPException
        for bad, code in (('../secret', 400), ('nope', 404), ('C9-bad', 422)):
            with self.assertRaises(HTTPException) as cm:
                self.M.case(bad)
            self.assertEqual(cm.exception.status_code, code)

    def test_research_stamp_tracks_reports(self):
        import datapack as DP
        a = DP.research_stamp(self.tmp)
        (self.tmp / 'mechanism' / 'CF-v1-qwen3-4b' / 'report.json').write_text(json.dumps({**CF, 'n_gated': 2}), encoding='utf-8')
        self.assertNotEqual(a, DP.research_stamp(self.tmp))


if __name__ == '__main__':
    unittest.main()
