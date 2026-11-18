"""Windows storage gates and daily-test retention regression checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from deploy import datapack
from deploy.storage_policy import POLICY, GIB, StorageBlocked, check_write, staging_file


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / 'tests/glm5_temp'


class StorageTests(unittest.TestCase):
    def test_reserve_and_forecast_are_checked_before_creation(self):
        path = TEMP / 'storage-check-do-not-create' / 'nested'
        with patch('deploy.storage_policy.shutil.disk_usage') as usage:
            reserve = POLICY['windows_reserve_gib' if os.name == 'nt' else 'portable_reserve_gib'] * GIB
            usage.return_value.free = reserve + 10
            self.assertEqual(check_write(path, 10), path.resolve())
            with self.assertRaisesRegex(StorageBlocked, '空间不足'):
                check_write(path, 11)
        self.assertFalse(path.exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows drive policy')
    def test_existing_c_drive_and_d_drive_junction(self):
        c_root = Path(os.environ['SystemRoot'])
        with self.assertRaises(StorageBlocked) as direct:
            check_write(c_root)
        self.assertEqual(direct.exception.reason, 'system_drive')
        junction = ROOT / 'tests/glm5/result/state_formation_v1'
        if not junction.is_junction() or junction.resolve().drive != c_root.drive:
            self.skipTest('read-only fixture requires an existing D-to-C research junction')
        destination = junction / 'storage-policy-nonexistent' / 'test.npz'
        with self.assertRaises(StorageBlocked) as linked:
            check_write(destination)
        self.assertEqual(linked.exception.reason, 'system_drive')
        self.assertEqual(linked.exception.path, destination.resolve())
        if node := shutil.which('node'):
            result = subprocess.run([node, '-e', """
                const policy = require('./frontend/electron/storagePolicy');
                try { policy.checkWrite(process.argv[1]); process.exit(1); }
                catch (error) { if (error.reason !== 'system_drive') throw error; }
                """, str(destination)], cwd=ROOT, check=True, capture_output=True)
            self.assertEqual(result.returncode, 0)

    def test_pack_preserves_previous_archive_when_staging_budget_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as raw:
            path = Path(raw) / 'datapack.json.gz'
            path.write_bytes(b'old archive')
            with patch('deploy.storage_policy.shutil.disk_usage') as usage:
                usage.return_value.free = POLICY['windows_reserve_gib' if os.name == 'nt' else 'portable_reserve_gib'] * GIB
                with self.assertRaisesRegex(StorageBlocked, '空间不足'):
                    datapack._write_gz(path, {'new': True})
            self.assertEqual(path.read_bytes(), b'old archive')
            self.assertFalse(path.with_suffix('.gz.tmp').exists())

    def test_interrupted_pack_removes_only_its_temporary_file(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as raw:
            path = Path(raw) / 'existing.tar'
            path.write_bytes(b'old archive')
            with self.assertRaisesRegex(OSError, 'test interruption'):
                with staging_file(path) as partial:
                    partial.write_bytes(b'incomplete')
                    raise OSError('test interruption')
            self.assertEqual(path.read_bytes(), b'old archive')
            self.assertEqual(list(Path(raw).iterdir()), [path])

    def test_daily_run_retains_only_the_replay_record(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as raw:
            root = Path(raw)
            # The replay wrapper intentionally only accepts scripts in tests/glm5.
            questions = root / 'questions.json'
            questions.write_text('{"answer": 5}', encoding='utf-8')
            proc = subprocess.run([sys.executable, '-B', str(ROOT / 'tests/glm5/run_routine_test.py'),
                                   '--script', __file__, '--questions', str(questions), '--peak-gib', '.01',
                                   '--seed', '19', '--', '--fixture'],
                                  env={**os.environ, 'AI2050_STORAGE_ROOT': raw, 'PYTHONUTF8': '1'},
                                  capture_output=True, text=True, encoding='utf-8', check=True)
            record = Path(proc.stdout.splitlines()[-1].split(' · ', 1)[1]).parent
            try:
                review = json.loads((record / 'review.json').read_text(encoding='utf-8'))
                self.assertEqual(review['conclusion'], {'observed': 5})
                self.assertEqual(review['seed'], 19)
                self.assertEqual({f.name for f in record.iterdir()}, {'test.py', 'questions.json', 'review.json'})
                self.assertFalse(Path(review['scratch']).exists())
            finally:
                self.assertEqual(record.parent.resolve(), (ROOT / 'tests/glm5/result/routine').resolve())
                for name in ('test.py', 'questions.json', 'review.json'):
                    (record / name).unlink(missing_ok=True)
                record.rmdir()


if __name__ == '__main__':
    if sys.argv[1:] == ['--fixture']:
        work = Path(os.environ['AI2050_TEST_WORK_DIR'])
        questions = json.loads(Path(os.environ['AI2050_TEST_QUESTIONS']).read_text(encoding='utf-8'))
        (work / 'intermediate.npz').write_bytes(b'disposable')
        (work / 'conclusion.json').write_text(json.dumps({'observed': questions['answer']}), encoding='utf-8')
    else:
        unittest.main()
