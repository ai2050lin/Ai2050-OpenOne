# -*- coding: utf-8 -*-
"""导出中心研究数据包（center_data.v1），用于把本机中心的研究数据合并进另一台中心（如公开服务器）。

只导出公开范围内、能进入合并与展示的数据：
  - 模板（templates）与其冻结包（bundles/<tm>/v<N>/）；
  - kind = real 且 design_sha 与当前模板一致的结果（results）及其内容寻址文件（objects/）；对应的已完成任务、审计、金丝雀参考；
  - 结果引用的节点（nodes）——节点令牌哈希替换为不可用的随机值，导入后这些节点不能再领取任务；
  - 自然语料普查（<native>/_census/<model_key>/<run>/）、对应语料（_census/corpus/<fp>/）与 tokenizer 文件；
  - 规律分析缓存（analysis_cache/），目标中心运行同版本代码时直接命中。
不导出：账号、会话、绑定码、自动测试作业、旧版本模板的结果、smoke / synthetic 结果、节点本地原生全场。

用法：python scripts/export_center_data.py [--out release] [--census-run 2ca5f0a756d36c8a-20000000]
导入：在目标服务器上 python deploy/import_center_data.py <包> --data-dir ... --native-dir ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = Path(os.environ.get('AI2050_DIST_DIR', str(ROOT / 'distributed_data')))
NATIVE = Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native'))
TABLES = ('templates', 'results', 'tasks', 'audits', 'canary_refs', 'nodes')


def copy_rows(src, dst, table, where='', args=()):
    cols = [r[1] for r in src.execute(f'PRAGMA table_info({table})')]
    rows = src.execute(f'SELECT {",".join(cols)} FROM {table} {where}', args).fetchall()
    dst.execute(f'CREATE TABLE IF NOT EXISTS {table} ({",".join(cols)})')
    dst.executemany(f'INSERT INTO {table} ({",".join(cols)}) VALUES ({",".join("?" * len(cols))})', rows)
    return rows, cols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(ROOT / 'release'))
    ap.add_argument('--census-run', action='append', default=None,
                    help='要导出的普查运行（<model_key>/<run> 或只写 run 名）；缺省导出 n_tokens 最大的一次')
    a = ap.parse_args()
    stamp = time.strftime('%Y%m%d')
    name = f'ai2050-data-{stamp}'
    work = Path(a.out) / name
    if work.exists():
        shutil.rmtree(work)
    (work / 'objects').mkdir(parents=True)
    src = sqlite3.connect(str(DIST / 'dist.sqlite3'))
    dst = sqlite3.connect(str(work / 'export.sqlite'))

    tpl_rows, tcols = copy_rows(src, dst, 'templates')
    cur = {r[tcols.index('tm_id')]: r[tcols.index('design_sha')] for r in tpl_rows}
    res_rows, rcols = copy_rows(src, dst, 'results', "WHERE kind='real'")
    keep = [r for r in res_rows if cur.get(r[rcols.index('tm_id')]) == r[rcols.index('design_sha')]]
    drop = {r[rcols.index('sha')] for r in res_rows} - {r[rcols.index('sha')] for r in keep}
    dst.executemany('DELETE FROM results WHERE sha=?', [(s,) for s in drop])
    shas = [r[rcols.index('sha')] for r in keep]
    q = ','.join('?' * len(shas))
    copy_rows(src, dst, 'tasks', f"WHERE status='done' AND result_sha IN ({q})", shas)
    copy_rows(src, dst, 'audits', f'WHERE result_sha IN ({q})', shas)
    copy_rows(src, dst, 'canary_refs')
    node_ids = sorted({r[rcols.index('node_id')] for r in keep})
    _, ncols = copy_rows(src, dst, 'nodes', f'WHERE node_id IN ({",".join("?" * len(node_ids))})', node_ids)
    if 'owner_user_id' in ncols:                  # 账号不随包导出：节点归属清空
        dst.execute('UPDATE nodes SET owner_user_id = NULL')
    for (nid,) in dst.execute('SELECT node_id FROM nodes').fetchall():   # 令牌哈希换成不可用的随机值
        dst.execute('UPDATE nodes SET token_hash=? WHERE node_id=?', ('imported-disabled-' + secrets.token_hex(16), nid))
    dst.commit()

    nbytes = 0
    for s in shas:
        p = DIST / 'objects' / s[:2] / s
        (work / 'objects' / s[:2]).mkdir(exist_ok=True)
        shutil.copyfile(p, work / 'objects' / s[:2] / s)
        nbytes += p.stat().st_size
    for r in tpl_rows:
        bd = Path(r[tcols.index('bundle_dir')])
        rel = Path('bundles') / r[tcols.index('tm_id')] / f"v{r[tcols.index('version')]}"
        if bd.is_dir():
            shutil.copytree(bd, work / rel, dirs_exist_ok=True)

    # 规律分析缓存（键含分片 sha 与 analysis.py 哈希；目标中心同版本代码可直接命中，免去大模板的首次计算）
    if (DIST / 'analysis_cache').is_dir():
        shutil.copytree(DIST / 'analysis_cache', work / 'analysis_cache', ignore=shutil.ignore_patterns('*.tmp'))

    # 普查
    runs = []
    for man in sorted((NATIVE / '_census').glob('*/*/manifest.json')):
        if man.parent.parent.name == 'corpus':
            continue
        m = json.loads(man.read_text(encoding='utf-8'))
        runs.append((m.get('n_tokens_counted', 0), man.parent, m))
    if a.census_run:
        runs = [r for r in runs if any(sel in (r[1].name, f'{r[1].parent.name}/{r[1].name}') for sel in a.census_run)]
    else:
        runs = sorted(runs, key=lambda r: -r[0])[:1]
    census = []
    for n, rdir, m in runs:
        rel = Path('native') / '_census' / rdir.parent.name / rdir.name
        shutil.copytree(rdir, work / rel)
        mp = Path(m['model_path'])
        mp = mp if mp.is_absolute() else ROOT / mp
        (work / rel / 'tokenizer').mkdir(exist_ok=True)
        for f in ('tokenizer.json', 'tokenizer_config.json', 'config.json'):
            if (mp / f).is_file():
                shutil.copyfile(mp / f, work / rel / 'tokenizer' / f)
        cfp = m['corpus_fp']
        cdst = work / 'native' / '_census' / 'corpus' / cfp
        if not cdst.exists():
            shutil.copytree(NATIVE / '_census' / 'corpus' / cfp, cdst)
        census.append({'model_key': rdir.parent.name, 'run': rdir.name, 'n_tokens': n, 'corpus_fp': cfp})
        snap_p = work / 'snapshots' / f'{rdir.parent.name}.json'
        if not snap_p.exists() and any(mp.glob('*.safetensors')):
            # 模型快照（权重与 tokenizer 文件 sha256）：目标中心没有权重时，用它建普查作业（admin_tool.py --snapshot-file）
            import sys
            sys.path.insert(0, str(ROOT))
            from shared.dist_protocol.runner_v3 import model_snapshot
            snap_p.parent.mkdir(exist_ok=True)
            snap_p.write_text(json.dumps(model_snapshot(mp), ensure_ascii=False, indent=1), encoding='utf-8')

    files = {}
    for p in sorted(work.rglob('*')):
        if p.is_file() and p.name != 'PACK_MANIFEST.json':
            h = hashlib.sha256()
            with p.open('rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b''):
                    h.update(chunk)
            files[p.relative_to(work).as_posix()] = h.hexdigest()
    manifest = {'schema': 'center_data.v1', 'name': name, 'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'counts': {t: dst.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in TABLES},
                'results_dropped_old_design': len(drop), 'object_bytes': nbytes, 'census': census, 'files': files,
                'note': '合并导入：只新增，不覆盖目标中心已有的同主键记录；不含账号与会话'}
    dst.close()
    (work / 'PACK_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    tar_path = Path(a.out) / f'{name}.tar'
    with tarfile.open(tar_path, 'w') as tar:              # 结果文件多为已压缩的 npz：不再 gzip，节省打包时间
        tar.add(work, arcname=name)
    shutil.rmtree(work)
    digest = hashlib.sha256()
    with tar_path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            digest.update(chunk)
    (Path(a.out) / f'{name}.tar.sha256').write_text(f'{digest.hexdigest()}  {tar_path.name}\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('name', 'counts', 'results_dropped_old_design', 'object_bytes', 'census')},
                     ensure_ascii=False, indent=1))
    print(f'{tar_path}  {tar_path.stat().st_size / 2**30:.2f} GB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
