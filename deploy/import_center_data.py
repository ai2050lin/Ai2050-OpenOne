#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把中心研究数据包（center_data.v1，scripts/export_center_data.py 生成）合并导入本中心。只依赖标准库。

合并规则：只新增，不覆盖目标库已有的同主键记录（INSERT OR IGNORE）；不碰账号、会话、绑定码；
模板的 bundle_dir 改写为本中心数据目录下的路径；内容寻址文件与模板包只补缺失的；
普查运行写入 <native>/_census/<model_key>/<run>/，其 manifest 的 model_path 改为包内带来的 tokenizer 目录。
导入前自动备份数据库；请先停止中心服务，导入后再启动。

用法（在服务器上，以 root 运行）：
  systemctl stop ai2050-center
  python3 deploy/import_center_data.py ai2050-data-20261026.tar --data-dir /var/lib/ai2050/center --native-dir /var/lib/ai2050/native
  chown -R ai2050:ai2050 /var/lib/ai2050 && systemctl start ai2050-center
可加 --dry-run 只显示将要导入的数量。数据目录以 systemd 中的 AI2050_DIST_DIR / AI2050_NATIVE_DIR 为准。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
from pathlib import Path

TABLES = ('templates', 'nodes', 'results', 'tasks', 'audits', 'canary_refs')


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def open_pack(path: Path, tmp: Path) -> Path:
    if path.is_dir():
        return path
    with tarfile.open(path) as tar:
        for m in tar.getmembers():                      # 拒绝绝对路径与 .. 跳出
            if m.name.startswith(('/', '\\')) or '..' in Path(m.name).parts or m.issym() or m.islnk():
                raise SystemExit(f'包内路径不安全：{m.name}')
        tar.extractall(tmp)
    dirs = [p for p in tmp.iterdir() if p.is_dir()]
    if len(dirs) != 1:
        raise SystemExit('包结构不对：应只有一个顶层目录')
    return dirs[0]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description='AI2050 center data import (merge, no overwrite)')
    ap.add_argument('pack', help='ai2050-data-<日期>.tar 或解开的目录')
    ap.add_argument('--data-dir', default='/var/lib/ai2050/center', help='中心数据目录（AI2050_DIST_DIR）')
    ap.add_argument('--native-dir', default='/var/lib/ai2050/native', help='普查目录（AI2050_NATIVE_DIR）')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-verify', action='store_true', help='跳过逐文件 sha256 核对（大包较慢）')
    a = ap.parse_args()
    data, native = Path(a.data_dir), Path(a.native_dir)
    db_path = data / 'dist.sqlite3'
    if not db_path.is_file():
        raise SystemExit(f'找不到中心数据库 {db_path}：先启动一次中心服务让它建库，或检查 --data-dir')
    with tempfile.TemporaryDirectory(dir=str(data.parent)) as td:
        pack = open_pack(Path(a.pack), Path(td))
        man = json.loads((pack / 'PACK_MANIFEST.json').read_text(encoding='utf-8'))
        if man.get('schema') != 'center_data.v1':
            raise SystemExit('不是 center_data.v1 数据包')
        if not a.skip_verify:
            bad = [rel for rel, h in man['files'].items() if sha_file(pack / rel) != h]
            if bad:
                raise SystemExit(f'{len(bad)} 个文件 sha256 不符（如 {bad[:3]}），包可能损坏')
            print(f'[import] {len(man["files"])} 个文件 sha256 核对通过')
        src = sqlite3.connect(str(pack / 'export.sqlite'))
        dst = sqlite3.connect(str(db_path), timeout=60)
        plan = {}
        for t in TABLES:
            if not src.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone():
                continue
            scols = [r[1] for r in src.execute(f'PRAGMA table_info({t})')]
            dcols = [r[1] for r in dst.execute(f'PRAGMA table_info({t})')]
            cols = [c for c in scols if c in dcols]
            pk = [r[1] for r in dst.execute(f'PRAGMA table_info({t})') if r[5]]
            rows = src.execute(f'SELECT {",".join(cols)} FROM {t}').fetchall()
            exist = 0
            if pk:
                ki = [cols.index(k) for k in pk]
                for r in rows:
                    if dst.execute(f'SELECT 1 FROM {t} WHERE ' + ' AND '.join(f'{k}=?' for k in pk), [r[i] for i in ki]).fetchone():
                        exist += 1
            plan[t] = {'in_pack': len(rows), 'already_present': exist, 'cols': cols, 'rows': rows}
        print('[import] 计划：' + json.dumps({t: {k: v for k, v in p.items() if k in ('in_pack', 'already_present')} for t, p in plan.items()},
                                            ensure_ascii=False))
        print(f"[import] 普查：{[c['model_key'] + '/' + c['run'] for c in man.get('census', [])]}")
        src.close()                                     # 之后只用 plan 中的行；尽早关闭，便于清理临时目录
        if a.dry_run:
            dst.close()
            return 0
        backup = data / f'dist.sqlite3.before-import-{time.strftime("%Y%m%d%H%M%S")}'
        bk = sqlite3.connect(str(backup))
        dst.backup(bk)
        bk.close()
        print(f'[import] 数据库已备份 → {backup}')
        with dst:
            for t, p in plan.items():
                cols, rows = p['cols'], p['rows']
                if t == 'templates' and 'bundle_dir' in cols:
                    bi, ti, vi = cols.index('bundle_dir'), cols.index('tm_id'), cols.index('version')
                    rows = [tuple(str(data / 'bundles' / r[ti] / f'v{r[vi]}') if i == bi else x for i, x in enumerate(r)) for r in rows]
                dst.executemany(f'INSERT OR IGNORE INTO {t} ({",".join(cols)}) VALUES ({",".join("?" * len(cols))})', rows)
        dst.close()
        n_obj = n_bdl = 0
        for p in (pack / 'objects').rglob('*'):
            if p.is_file():
                q = data / 'objects' / p.parent.name / p.name
                if not q.exists():
                    q.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, q)
                    n_obj += 1
        if (pack / 'bundles').is_dir():
            for p in (pack / 'bundles').rglob('*'):
                if p.is_file():
                    q = data / p.relative_to(pack)
                    if not q.exists():
                        q.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(p, q)
                        n_bdl += 1
        if (pack / 'analysis_cache').is_dir():
            for p in (pack / 'analysis_cache').glob('*.json'):
                q = data / 'analysis_cache' / p.name
                if not q.exists():
                    q.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, q)
        for c in man.get('census', []):
            rel = Path('_census') / c['model_key'] / c['run']
            target = native / rel
            if target.exists():
                print(f'[import] 普查 {rel} 已存在，跳过')
            else:
                shutil.copytree(pack / 'native' / rel, target)
                mpath = target / 'manifest.json'
                m = json.loads(mpath.read_text(encoding='utf-8'))
                m['model_path'] = str((target / 'tokenizer').resolve())
                mpath.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
                print(f'[import] 普查 {rel} 已导入')
            cdir = native / '_census' / 'corpus' / c['corpus_fp']
            if not cdir.exists():
                shutil.copytree(pack / 'native' / '_census' / 'corpus' / c['corpus_fp'], cdir)
                print(f"[import] 语料 {c['corpus_fp']} 已导入")
        if (pack / 'snapshots').is_dir():
            sd = native / '_census' / 'snapshots'
            sd.mkdir(parents=True, exist_ok=True)
            for p in (pack / 'snapshots').glob('*.json'):
                shutil.copyfile(p, sd / p.name)
            print(f'[import] 模型快照 → {sd}（建普查作业时用 admin_tool.py census-create --snapshot-file）')
        print(f'[import] 完成：新增结果文件 {n_obj} 个、模板包文件 {n_bdl} 个。'
              f'记得 chown -R ai2050:ai2050 {data} {native}，再启动中心服务。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
