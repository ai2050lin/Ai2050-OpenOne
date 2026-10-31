# -*- coding: utf-8 -*-
"""dataset_progress.v1：每个数据集两条线、每条五个阶段；状态只取固定集合；数据源不可读不让整页失败；进主数据包。"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'deploy'), str(ROOT / 'server')]
os.environ.setdefault('AI2050_SNAPSHOT_DISABLE', '1')


class DatasetProgressTests(unittest.TestCase):
    def test_shape_and_status_vocabulary(self):
        from server import dataset_progress_service as P
        doc = P.build()
        self.assertEqual(doc['schema'], 'dataset_progress.v1')
        for track, stages in P.STAGES.items():
            self.assertEqual(len(stages), 5)
        for ds in doc['datasets']:
            for track, stages in P.STAGES.items():
                for s in stages:
                    self.assertIn(ds[track][s['id']]['status'], P.STATUS, (ds['id'], s['id']))
            self.assertIn('contribute', ds)
            for lay in ds.get('layers', {}).values():
                self.assertIn(lay['kind'], ('sequential', 'diverging', 'progress'))

    def test_broken_source_only_affects_its_rows(self):
        from server import dataset_progress_service as P
        orig = P._corpus_rows
        P._corpus_rows = lambda: (_ for _ in ()).throw(RuntimeError('boom'))
        try:
            doc = P.build()
        finally:
            P._corpus_rows = orig
        self.assertFalse(any(d['kind'] == 'corpus' for d in doc['datasets']))

    def test_in_datapack_plan(self):
        import datapack
        self.assertIn('/api/progress/datasets', [p for p, _ in datapack.entries_plan()])


if __name__ == '__main__':
    unittest.main()
