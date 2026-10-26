# -*- coding: utf-8 -*-
"""中心数据迁移（center_data.v1）：scripts/export_center_data.py 导出 → deploy/import_center_data.py 合并导入。
检查：只导出当前模板版本的 real 结果；节点令牌被替换；导入不碰目标中心的账号；bundle_dir 与普查 model_path 被改写；
重复导入是幂等的。"""
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def make_center(data: Path) -> None:
    """用中心服务自己的建库逻辑建一个空中心（播种早期模板）。"""
    code = ("import sys,threading;sys.path[:0]=[r'%s',r'%s'];import distributed_service as D;"
            "from pathlib import Path;D._DATA_DIR=Path(r'%s');D._conn=None;D._tls=threading.local();D._db()") % (ROOT, ROOT / 'deploy', data)
    subprocess.run([PY, '-c', code], check=True, capture_output=True)


class CenterDataMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src, self.dst = self.tmp / 'src', self.tmp / 'dst'
        self.native_src, self.native_dst = self.tmp / 'nsrc', self.tmp / 'ndst'
        for d in (self.src, self.dst):
            d.mkdir()
            make_center(d)
        c = sqlite3.connect(self.src / 'dist.sqlite3')
        tm, ds = c.execute("SELECT tm_id, design_sha FROM templates ORDER BY tm_id LIMIT 1").fetchone()
        blob = b'\x00\x00\x00\x02[]'
        for i, design in enumerate((ds, 'old-design')):
            sha = f'{i:02d}' + 'a' * 62
            (self.src / 'objects' / sha[:2]).mkdir(parents=True, exist_ok=True)
            (self.src / 'objects' / sha[:2] / sha).write_bytes(blob)
            c.execute("INSERT INTO results(sha,tm_id,version,node_id,model_id,model_rev,kind,design_sha,summary_json,uploaded_at) "
                      "VALUES(?,?,?,?,?,?,?,?,?,?)", (sha, tm, 1, 'NODE-X', 'm', 'r', 'real', design, '{}', time.time()))
        c.execute("INSERT INTO results(sha,tm_id,version,node_id,model_id,model_rev,kind,design_sha,summary_json,uploaded_at) "
                  "VALUES(?,?,?,?,?,?,?,?,?,?)", ('ff' + 'b' * 62, tm, 1, 'NODE-X', 'm', 'r', 'smoke', ds, '{}', time.time()))
        c.execute("INSERT INTO nodes(node_id,token_hash,name,created_at,last_seen) VALUES('NODE-X','secret-hash','n',0,0)")
        c.commit(); c.close()
        self.tm = tm
        # 目标中心有一个账号
        d = sqlite3.connect(self.dst / 'dist.sqlite3')
        sys.path[:0] = [str(ROOT)]
        from shared.dist_protocol import accounts as A
        d.row_factory = sqlite3.Row
        A.ensure_schema(d)
        A.register(d, 'admin1', 'pw-admin-12345')
        d.close()
        # 普查
        run = self.native_src / '_census' / 'mk-test' / 'run1'
        run.mkdir(parents=True)
        tokdir = self.tmp / 'tok'
        tokdir.mkdir()
        (tokdir / 'tokenizer.json').write_text('{}', encoding='utf-8')
        (run / 'manifest.json').write_text(json.dumps({'schema': 'mlp_census.v1', 'model_key': 'mk-test', 'corpus_fp': 'fp1',
                                                       'n_tokens_counted': 10, 'model_path': str(tokdir)}), encoding='utf-8')
        buf = io.BytesIO(); np.savez(buf, n=np.array(10)); (run / 'L0.npz').write_bytes(buf.getvalue())
        corpus = self.native_src / '_census' / 'corpus' / 'fp1'
        corpus.mkdir(parents=True)
        (corpus / 'manifest.json').write_text('{}', encoding='utf-8')

    def export(self):
        out = self.tmp / 'out'
        env = dict(os.environ, AI2050_DIST_DIR=str(self.src), AI2050_NATIVE_DIR=str(self.native_src), PYTHONIOENCODING='utf-8')
        r = subprocess.run([PY, str(ROOT / 'scripts' / 'export_center_data.py'), '--out', str(out)], env=env, capture_output=True,
                           text=True, encoding='utf-8')
        self.assertEqual(r.returncode, 0, r.stderr)
        return next(out.glob('ai2050-data-*.tar'))

    def run_import(self, pack, *extra):
        r = subprocess.run([PY, str(ROOT / 'deploy' / 'import_center_data.py'), str(pack), '--data-dir', str(self.dst),
                            '--native-dir', str(self.native_dst), *extra], capture_output=True, text=True, encoding='utf-8',
                           env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        return r.stdout

    def test_export_import_merge(self):
        pack = self.export()
        out = self.run_import(pack)
        self.assertIn('已备份', out)
        d = sqlite3.connect(self.dst / 'dist.sqlite3')
        self.assertEqual(d.execute('SELECT COUNT(*) FROM users').fetchone()[0], 1)               # 账号未被动
        shas = [r[0] for r in d.execute('SELECT sha FROM results')]
        self.assertEqual(shas, ['00' + 'a' * 62])                                               # 只有当前版本的 real
        self.assertTrue((self.dst / 'objects' / '00' / ('00' + 'a' * 62)).is_file())
        tok = d.execute("SELECT token_hash FROM nodes WHERE node_id='NODE-X'").fetchone()[0]
        self.assertTrue(tok.startswith('imported-disabled-'))                                  # 节点令牌不可用
        man = json.loads((self.native_dst / '_census' / 'mk-test' / 'run1' / 'manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(Path(man['model_path']).joinpath('tokenizer.json').is_file())          # model_path 指向包内 tokenizer
        self.assertTrue((self.native_dst / '_census' / 'corpus' / 'fp1' / 'manifest.json').is_file())
        d.close()
        out2 = self.run_import(pack)                                                            # 幂等
        self.assertIn('"already_present": 1', out2)
        d = sqlite3.connect(self.dst / 'dist.sqlite3')
        self.assertEqual(d.execute('SELECT COUNT(*) FROM results').fetchone()[0], 1)
        d.close()

    def test_dry_run_changes_nothing(self):
        pack = self.export()
        self.run_import(pack, '--dry-run')
        d = sqlite3.connect(self.dst / 'dist.sqlite3')
        self.assertEqual(d.execute('SELECT COUNT(*) FROM results').fetchone()[0], 0)
        d.close()
        self.assertFalse((self.native_dst / '_census').exists())


if __name__ == '__main__':
    unittest.main()
