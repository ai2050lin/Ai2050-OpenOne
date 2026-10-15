# -*- coding: utf-8 -*-
"""研发 · 自动测试（autotest.v1）作业队列：模型登记校验、批次展开、进度解析、行为门停止、取消、重启中断。
用一个假驱动代替 run_language_family.py（只打印约定的 @@progress / @@result 行），不需要中心服务与模型。"""
import json
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import autotest  # noqa: E402

FAKE = textwrap.dedent('''
    import json, sys, time
    mode = sys.argv[1]
    def p(step, label, **kw):
        print('@@progress ' + json.dumps({'step': step, 'label': label, **kw}, ensure_ascii=False), flush=True)
    p(1, '文法预览')
    if mode == 'gate':
        p(2, '试跑与行为门', detail='3/30 格达标'); sys.exit(3)
    if mode == 'crash':
        print('Traceback: boom', flush=True); sys.exit(1)
    if mode == 'slow':
        p(4, '分片采集'); time.sleep(30)
    p(3, '冻结正式模板', tm_id='TG-X')
    p(7, '报告')
    print('@@result ' + json.dumps({'tm_id': 'TG-X', 'bucket': 'm|r|bf16', 'findings': {'supported': 2}}), flush=True)
''')


class AutoTestQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fake = self.tmp / 'fake_driver.py'
        self.fake.write_text(FAKE, encoding='utf-8')
        self.weights = self.tmp / 'model'
        self.weights.mkdir()
        (self.weights / 'config.json').write_text(json.dumps({'architectures': ['QwenForCausalLM'], 'num_hidden_layers': 4,
                                                              'hidden_size': 64}), encoding='utf-8')
        (self.weights / 'model.safetensors').write_bytes(b'\0' * 16)
        self.mode = {}
        self.at = self._make()

    def _make(self, synthetic=True):
        def cmd(job, _env):
            return [sys.executable, str(self.fake), self.mode.get(job['family'], 'ok')]
        return autotest.AutoTest(self.tmp / 'data', ROOT, lambda: 'http://127.0.0.1:9', 'tok',
                                 allow_synthetic=lambda: synthetic, command_builder=cmd)

    def wait(self, pred, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if pred():
                return
            time.sleep(0.05)
        self.fail('timeout; jobs=' + json.dumps(self.at.jobs, ensure_ascii=False)[:600])

    def test_model_validation(self):
        with self.assertRaises(ValueError):
            self.at.upsert_model({'model_id': 'bad id', 'model_path': str(self.weights)})
        with self.assertRaises(ValueError):
            self.at.upsert_model({'model_id': 'm1', 'kind': 'local'})               # 缺权重目录
        with self.assertRaises(ValueError):
            self._make(synthetic=False).upsert_model({'model_id': 's', 'kind': 'synthetic'})
        self.at.upsert_model({'model_id': 'm1', 'model_path': str(self.weights), 'dtype': 'BF16'})
        m = self.at.list_models()[0]
        self.assertEqual(m['dtype'], 'bf16')
        self.assertTrue(m['check']['ok'])
        self.assertEqual(m['check']['layers'], 4)
        self.at.upsert_model({'model_id': 'm1', 'model_path': str(self.tmp / 'missing')})   # 同 id 覆盖
        self.assertEqual(len(self.at.models), 1)
        self.assertFalse(self.at.list_models()[0]['check']['ok'])

    def test_submit_validation(self):
        self.at.upsert_model({'model_id': 'gone', 'model_path': str(self.tmp / 'missing')})
        with self.assertRaises(ValueError):
            self.at.submit(['KN-ISA'], [], {}, ['KN-ISA'])
        with self.assertRaises(ValueError):
            self.at.submit(['NOPE'], ['gone'], {}, ['KN-ISA'])
        with self.assertRaises(ValueError):                                              # 权重目录不可用
            self.at.submit(['KN-ISA'], ['gone'], {}, ['KN-ISA'])
        with self.assertRaises(ValueError):
            self.at.submit(['KN-ISA'], ['unregistered'], {}, ['KN-ISA'])

    def test_batch_runs_all_combinations(self):
        self.at.upsert_model({'model_id': 'm1', 'model_path': str(self.weights)})
        self.at.upsert_model({'model_id': 's1', 'kind': 'synthetic'})
        self.mode['GATE'] = 'gate'
        self.mode['CRASH'] = 'crash'
        res = self.at.submit(['OK', 'GATE', 'CRASH'], ['m1', 's1'], {'pilot_per_cell': 99}, ['OK', 'GATE', 'CRASH'])
        self.assertEqual(len(res['created']), 6)
        again = self.at.submit(['OK'], ['m1'], {}, ['OK'])                              # 已在队列中的组合跳过
        self.assertEqual(again['created'], [])
        self.assertEqual(again['skipped_duplicates'], ['OK × m1'])
        self.wait(lambda: all(j['status'] in autotest.FINAL for j in self.at.jobs))
        by = {(j['family'], j['model_id']): j for j in self.at.jobs}
        ok = by[('OK', 'm1')]
        self.assertEqual(ok['status'], 'done')
        self.assertEqual((ok['tm_id'], ok['bucket']), ('TG-X', 'm|r|bf16'))
        self.assertEqual(ok['step'], len(autotest.STEPS))
        self.assertEqual(ok['options']['pilot_per_cell'], 20)                          # 选项被夹到合法范围
        self.assertEqual(by[('GATE', 's1')]['status'], 'stopped')
        self.assertEqual(by[('GATE', 's1')]['step'], 2)
        self.assertEqual(by[('CRASH', 'm1')]['status'], 'failed')
        self.assertIn('boom', by[('CRASH', 'm1')]['error'])
        order = [(j['model_id'], j['family']) for j in sorted(self.at.jobs, key=lambda j: j['started_at'])]
        self.assertEqual([m for m, _ in order], ['m1'] * 3 + ['s1'] * 3)                # 同一模型的作业连续执行
        self.assertTrue(any('文法预览' in ln or 'boom' in ln for ln in self.at.log_tail(by[('CRASH', 'm1')]['id'])))
        st = self.at.state()
        self.assertEqual(st['schema'], 'autotest.v1')
        self.assertEqual(len(st['steps']), 7)
        retry = self.at.retry(by[('CRASH', 'm1')]['id'])
        self.assertEqual(len(retry['created']), 1)
        self.wait(lambda: all(j['status'] in autotest.FINAL for j in self.at.jobs))
        self.assertGreaterEqual(self.at.clear_finished(), 7)
        self.assertEqual(self.at.jobs, [])

    def test_cancel_running_and_queued(self):
        self.at.upsert_model({'model_id': 'm1', 'model_path': str(self.weights)})
        self.mode['SLOW'] = 'slow'
        res = self.at.submit(['SLOW', 'OK'], ['m1'], {}, ['SLOW', 'OK'])
        slow, ok = res['created']
        self.wait(lambda: self.at._job(slow)['step'] == 4)
        self.at.cancel(ok)
        self.assertEqual(self.at._job(ok)['status'], 'cancelled')
        t0 = time.time()
        self.at.cancel(slow)
        self.wait(lambda: self.at._job(slow)['status'] == 'cancelled', timeout=10)
        self.assertLess(time.time() - t0, 10)
        with self.assertRaises(ValueError):
            self.at.cancel(slow)

    def test_restart_marks_running_as_interrupted(self):
        self.at.upsert_model({'model_id': 'm1', 'model_path': str(self.weights)})
        self.mode['SLOW'] = 'slow'
        jid = self.at.submit(['SLOW'], ['m1'], {}, ['SLOW'])['created'][0]
        self.wait(lambda: self.at._job(jid)['status'] == 'running')
        fresh = self._make()
        self.assertEqual(fresh._job(jid)['status'], 'interrupted')
        self.at.cancel(jid)
        self.wait(lambda: self.at._job(jid)['status'] == 'cancelled', timeout=10)

    def test_log_path_rejects_traversal(self):
        with self.assertRaises(KeyError):
            self.at.log_path('../../etc/passwd')


if __name__ == '__main__':
    unittest.main()
