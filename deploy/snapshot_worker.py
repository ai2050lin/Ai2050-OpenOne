# -*- coding: utf-8 -*-
"""中心后台计算子进程：快照（总览漏斗、覆盖矩阵、题库、记分板、谱图）、规律分析预热与普查自动合并。

与网站进程分开运行（低优先级），结果写到 <AI2050_DIST_DIR>/snapshots/<name>.json，网站进程只读文件，
计算再久也不会拖慢页面。每个快照记录数据指纹（结果、审计、模板、普查的计数与最新时间）；
指纹没变就不重算，所以重启后立即可用，只有数据变化时才重新计算。

由 deploy/center_app.py 启动时自动拉起（AI2050_SNAPSHOT_DISABLE=1 关闭）；也可单独运行：
    python deploy/snapshot_worker.py [--once]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

NAMES = ('catalog', 'coverage', 'pipeline_overview', 'scoreboard', 'atlas')


def snapshot_dir() -> Path:
    import distributed_service as D
    d = Path(D._DATA_DIR) / 'snapshots'
    d.mkdir(parents=True, exist_ok=True)
    return d


def fingerprint() -> str:
    """决定快照是否过期的数据指纹：结果、审计、模板、节点、普查结果的计数与最新时间。"""
    import distributed_service as D
    db = D._db()
    parts = []
    for sql in ("SELECT COUNT(*), MAX(uploaded_at) FROM results",
                "SELECT COUNT(*), MAX(created_at) FROM audits",
                "SELECT COUNT(*), GROUP_CONCAT(design_sha || status, ',') FROM templates",
                "SELECT COUNT(*) FROM nodes"):
        parts.append(list(db.execute(sql).fetchone()))
    try:
        parts.append(list(db.execute("SELECT COUNT(*), MAX(created_at) FROM census_results").fetchone()))
    except Exception:
        parts.append(None)
    return hashlib.sha256(json.dumps(parts, default=str).encode('utf-8')).hexdigest()[:20]


def write_snapshot(name: str, data, fp: str, seconds: float, error=None) -> None:
    path = snapshot_dir() / f'{name}.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'name': name, 'computed_at': time.time(), 'seconds': round(seconds, 1), 'fingerprint': fp,
                               'error': error, 'data': data}, ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(path)


def stored_fingerprint(name: str):
    path = snapshot_dir() / f'{name}.json'
    if not path.is_file():
        return None
    try:
        with path.open(encoding='utf-8') as fh:
            head = json.load(fh)
        return head.get('fingerprint') if head.get('data') is not None else None
    except (OSError, ValueError):
        return None


def run_once(force: bool = False, log=print) -> dict:
    import distributed_service as D
    import platform_service as PS
    out = {'computed': [], 'skipped': []}
    try:
        import census_service as CS
        merged = CS.auto_merge()                       # 普查：新完成的切片并入单元卡
        if merged:
            log(f'[snapshots] 普查合并：{merged}')
    except Exception as exc:
        log(f'[snapshots] census auto merge: {exc}')
    fp = fingerprint()
    builders = PS._builders()
    for name in NAMES:
        if not force and stored_fingerprint(name) == fp:
            out['skipped'].append(name)
            continue
        t0 = time.time()
        try:
            write_snapshot(name, builders[name](), fp, time.time() - t0)
            out['computed'].append(name)
            log(f'[snapshots] {name} {time.time() - t0:.0f}s')
        except Exception as exc:                       # 失败时保留上一次的文件
            log(f'[snapshots] {name} 失败：{type(exc).__name__}: {exc}')
    marker = snapshot_dir() / '_analyses.fingerprint'
    if not force and marker.is_file() and marker.read_text(encoding='utf-8').strip() == fp:
        out['analyses'] = 'skipped'                    # 数据没变：规律分析缓存已齐，不必逐模板检查
        return out
    try:
        from shared.dist_protocol import pipeline as P
        fresh = P.warm_analyses(D._db(), D._DATA_DIR)  # 实验响应页的规律分析：逐个模板 × 模型桶落盘
        if fresh:
            log(f'[snapshots] 规律分析预热：{fresh}')
        out['analyses'] = fresh
        marker.write_text(fp, encoding='utf-8')
    except Exception as exc:
        log(f'[snapshots] analysis warm-up: {exc}')
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='AI2050 center snapshot worker')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if hasattr(os, 'nice'):
        try:
            os.nice(10)                                # 让出 CPU 给网站进程
        except OSError:
            pass
    interval = float(os.environ.get('AI2050_SNAPSHOT_SECONDS', '300'))
    while True:
        run_once(force=a.force)
        if a.once:
            return 0
        time.sleep(interval)


if __name__ == '__main__':
    raise SystemExit(main())
