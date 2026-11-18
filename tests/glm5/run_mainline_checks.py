"""Progress-logged protocol + publication regression. Uses a workspace-owned TEMP."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'deploy')]
OUT = ROOT / 'tests/glm5/result/mainline_sync_20261108'
OUT.mkdir(parents=True, exist_ok=True)
temp = OUT / 'temp'
temp.mkdir(exist_ok=True)
research = OUT / 'empty-research'
research.mkdir(exist_ok=True)
os.environ.update(TEMP=str(temp), TMP=str(temp), AI2050_SNAPSHOT_DISABLE='1')
os.environ['AI2050_RESEARCH_DIR'] = str(research)
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
tempfile.tempdir = str(temp)
sys.stdout.reconfigure(encoding='utf-8')
started = time.monotonic()
suite = unittest.TestSuite([
    unittest.defaultTestLoader.discover(str(ROOT / 'tests/dist_protocol')),
    unittest.defaultTestLoader.discover(str(ROOT / 'tests/glm5'), pattern='test_mainline_state.py'),
])
total = suite.countTestCases()


class Progress(unittest.TextTestResult):
    def startTest(self, test):
        self.current_test = str(test)
        super().startTest(test)

    def stopTest(self, test):
        super().stopTest(test)
        if self.testsRun % 10 == 0 or self.testsRun == total:
            elapsed = time.monotonic() - started
            print(f'\nprogress {self.testsRun}/{total} elapsed={elapsed:.1f}s eta={elapsed / self.testsRun * (total-self.testsRun):.1f}s', flush=True)


print(f'progress 0/{total} starting CPU protocol/publication checks', flush=True)
stop = threading.Event()


def heartbeat():
    while not stop.wait(60):
        print(f'progress heartbeat elapsed={time.monotonic()-started:.1f}s; checks still running, see last completed count above', flush=True)


threading.Thread(target=heartbeat, daemon=True).start()
result = unittest.TextTestRunner(verbosity=1, resultclass=Progress).run(suite)
stop.set()
summary = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
           'skipped': len(result.skipped), 'elapsed_seconds': time.monotonic()-started,
           'details': [{'test': str(t), 'traceback': tb} for t, tb in result.errors + result.failures]}
(OUT / 'backend.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
sys.exit(0 if result.wasSuccessful() else 1)
