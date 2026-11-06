# -*- coding: utf-8 -*-
"""中心服务 · 数据管理（data_admin.v1）：管理员只读查看服务器上的整体数据与每个账号上传了什么。

  GET /api/admin/data/overview                 存储占用（数据目录各部分、磁盘剩余）、各表计数、按账号汇总的上传
  GET /api/admin/data/uploads                  上传记录（模式族结果 + 普查分片），按时间倒序，可按来源 / 账号 / 节点 / 类型 / 状态 / 关键词筛选
  GET /api/admin/data/uploads/{source}/{id}    单条上传的详情：文件清单、摘要、金丝雀、审计、副本或校验与复算比较

授权：管理员账号的会话（Authorization: Bearer）或中心管理令牌（X-Admin-Token）。只读：不删除、不修改任何数据；
计数只描述服务器上有什么，不表示证据晋级。
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query

import distributed_service as D  # deploy/ 在 sys.path 上（server.py 与 center_app.py 都会加入）
from shared.dist_protocol import accounts as A

router = APIRouter(prefix='/api/admin/data', tags=['admin'])

STORAGE_TTL = 120.0              # 目录遍历的缓存秒数（objects 可能有上万个文件）
_storage_cache: Dict[str, Any] = {'at': 0.0, 'key': '', 'data': None}
_storage_lock = threading.Lock()


def _require_admin(authorization: str, x_admin_token: str) -> None:
    tok = A.bearer(authorization) or x_admin_token
    D._auth_admin(tok)


def _census_ready() -> bool:
    """普查表由 census_service 首次访问时建立；没挂载普查路由的中心没有这些表。"""
    try:
        import census_service as CS
        CS.db()
        return True
    except Exception:
        return False


def _tree_size(p: Path) -> Dict[str, int]:
    files = size = 0
    if p.is_file():
        return {'files': 1, 'bytes': p.stat().st_size}
    stack = [p]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            files += 1
                            size += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return {'files': files, 'bytes': size}


STORAGE_PARTS = (
    ('db', '数据库', ('dist.sqlite3', 'dist.sqlite3-wal', 'dist.sqlite3-shm')),
    ('objects', '模式族结果（内容寻址）', ('objects',)),
    ('census', '普查作业与分片结果', ('census',)),
    ('bundles', '模板任务包', ('bundles',)),
    ('snapshots', '后台快照与数据包', ('snapshots',)),
)


def _disk(p: Path) -> Optional[Dict[str, int]]:
    try:
        u = shutil.disk_usage(p)
        return {'total': u.total, 'used': u.used, 'free': u.free}
    except OSError:
        return None


def storage(force: bool = False) -> Dict[str, Any]:
    root = D._DATA_DIR
    try:
        from census_service import native_root
        native = native_root()
    except Exception:
        native = None
    key = f'{root}|{native}'
    with _storage_lock:
        c = _storage_cache
        if not force and c['data'] is not None and c['key'] == key and time.time() - c['at'] < STORAGE_TTL:
            return c['data']
    parts, claimed = [], set()
    for pid, label, names in STORAGE_PARTS:
        files = size = 0
        for name in names:
            p = root / name
            claimed.add(name)
            if p.exists():
                s = _tree_size(p)
                files += s['files']; size += s['bytes']
        parts.append({'id': pid, 'label': label, 'files': files, 'bytes': size})
    files = size = 0
    if root.is_dir():
        for e in root.iterdir():
            if e.name not in claimed:
                s = _tree_size(e)
                files += s['files']; size += s['bytes']
    parts.append({'id': 'other', 'label': '其他（规律分析缓存、谱图、日志等）', 'files': files, 'bytes': size})
    out: Dict[str, Any] = {
        'data_dir': str(root), 'parts': parts, 'total_bytes': sum(p['bytes'] for p in parts),
        'disk': _disk(root) if root.exists() else None, 'computed_at': time.time(),
    }
    if native is not None and native.exists() and not str(native.resolve()).startswith(str(root.resolve())):
        census = native / '_census'
        out['native'] = {'dir': str(native), 'census_bytes': _tree_size(census)['bytes'] if census.exists() else 0,
                         'disk': _disk(native)}
    with _storage_lock:
        _storage_cache.update(at=time.time(), key=key, data=out)
    return out


def _group(db, sql: str) -> Dict[str, int]:
    return {str(r[0] if r[0] is not None else '—'): r[1] for r in db.execute(sql).fetchall()}


def counts(db, census: bool) -> Dict[str, Any]:
    now = time.time()
    one = lambda sql, *a: db.execute(sql, a).fetchone()[0] or 0  # noqa: E731
    out: Dict[str, Any] = {
        'users': {'total': one('SELECT COUNT(*) FROM users'), 'admin': one("SELECT COUNT(*) FROM users WHERE role='admin'"),
                  'disabled': one('SELECT COUNT(*) FROM users WHERE disabled=1')},
        'nodes': {'total': one('SELECT COUNT(*) FROM nodes'),
                  'online': one('SELECT COUNT(*) FROM nodes WHERE last_seen>=?', now - D.NODE_ONLINE_WINDOW),
                  'unowned': one('SELECT COUNT(*) FROM nodes WHERE owner_user_id IS NULL')},
        'templates': _group(db, 'SELECT status, COUNT(*) FROM templates GROUP BY status'),
        'tasks': _group(db, 'SELECT status, COUNT(*) FROM tasks GROUP BY status'),
        'results': {'total': one('SELECT COUNT(*) FROM results'), 'bytes': one('SELECT SUM(size) FROM results'),
                    'by_kind': _group(db, 'SELECT kind, COUNT(*) FROM results GROUP BY kind'),
                    'replicas': one('SELECT COUNT(*) FROM result_replicas'),
                    'audits': _group(db, 'SELECT status, COUNT(*) FROM audits GROUP BY status'),
                    'unaudited': one('SELECT COUNT(*) FROM results r LEFT JOIN audits a ON a.result_sha=r.sha WHERE a.result_sha IS NULL')},
    }
    if census:
        out['census'] = {
            'jobs': _group(db, 'SELECT status, COUNT(*) FROM census_jobs GROUP BY status'),
            'results': _group(db, 'SELECT status, COUNT(*) FROM census_results GROUP BY status'),
            'by_kind': _group(db, 'SELECT kind, COUNT(*) FROM census_results GROUP BY kind'),
            'bytes': one('SELECT SUM(l.bytes) FROM census_results c JOIN census_leases l ON l.lease_id=c.lease_id'),
            'tokens': one("SELECT SUM(n_tokens) FROM census_results WHERE status='accepted'"),
            'active_leases': one("SELECT COUNT(*) FROM census_leases WHERE status='claimed' AND lease_until>=?", now),
            'replica_checks': _group(db, "SELECT CASE ok WHEN 1 THEN 'ok' ELSE 'mismatch' END, COUNT(*) FROM census_replicas GROUP BY ok"),
        }
    return out


def by_user(db, census: bool) -> List[Dict[str, Any]]:
    """每个账号（另加一行「无主节点」）的节点与上传汇总；副本（字节相同的重复上传）单独计数。"""
    now = time.time()
    rows: Dict[str, Dict[str, Any]] = {}

    def row(uid: Optional[str]) -> Dict[str, Any]:
        k = uid or ''
        if k not in rows:
            rows[k] = {'user_id': uid, 'nodes': 0, 'online': 0, 'results': 0, 'results_bytes': 0, 'replicas': 0,
                       'census': 0, 'census_accepted': 0, 'census_rejected': 0, 'census_disputed': 0,
                       'census_bytes': 0, 'census_tokens': 0, 'last_upload': None, 'synthetic': 0}
        return rows[k]

    for u in db.execute('SELECT user_id, username, display_name, role, disabled FROM users').fetchall():
        row(u['user_id']).update(username=u['username'], display_name=u['display_name'], role=u['role'], disabled=bool(u['disabled']))
    for n in db.execute('SELECT owner_user_id, COUNT(*) n, SUM(CASE WHEN last_seen>=? THEN 1 ELSE 0 END) o FROM nodes '
                        'GROUP BY owner_user_id', (now - D.NODE_ONLINE_WINDOW,)).fetchall():
        r = row(n['owner_user_id']); r['nodes'] = n['n']; r['online'] = n['o'] or 0

    def bump(r: Dict[str, Any], ts: Optional[float]) -> None:
        if ts and (r['last_upload'] is None or ts > r['last_upload']):
            r['last_upload'] = ts

    for x in db.execute('SELECT n.owner_user_id uid, COUNT(*) c, SUM(r.size) b, MAX(r.uploaded_at) t, '
                        "SUM(CASE WHEN r.kind='synthetic' THEN 1 ELSE 0 END) s "
                        'FROM results r LEFT JOIN nodes n ON n.node_id=r.node_id GROUP BY n.owner_user_id').fetchall():
        r = row(x['uid']); r['results'] = x['c']; r['results_bytes'] = x['b'] or 0; r['synthetic'] += x['s'] or 0; bump(r, x['t'])
    for x in db.execute('SELECT n.owner_user_id uid, COUNT(*) c, MAX(p.uploaded_at) t FROM result_replicas p '
                        'LEFT JOIN nodes n ON n.node_id=p.node_id GROUP BY n.owner_user_id').fetchall():
        r = row(x['uid']); r['replicas'] = x['c']; bump(r, x['t'])
    if census:
        for x in db.execute('SELECT n.owner_user_id uid, c.status st, COUNT(*) k, SUM(l.bytes) b, SUM(c.n_tokens) tok, MAX(c.created_at) t, '
                            "SUM(CASE WHEN c.kind='synthetic' THEN 1 ELSE 0 END) s "
                            'FROM census_results c LEFT JOIN census_leases l ON l.lease_id=c.lease_id '
                            'LEFT JOIN nodes n ON n.node_id=c.node_id GROUP BY n.owner_user_id, c.status').fetchall():
            r = row(x['uid'])
            r['census'] += x['k']; r['census_bytes'] += x['b'] or 0; r['synthetic'] += x['s'] or 0
            if x['st'] in ('accepted', 'rejected', 'disputed'):
                r[f"census_{x['st']}"] += x['k']
            if x['st'] == 'accepted':
                r['census_tokens'] += x['tok'] or 0
            bump(r, x['t'])
    out = list(rows.values())
    for r in out:
        if r['user_id'] is None:
            r.update(username='', display_name='无主节点', role='', disabled=False)
        elif 'username' not in r:      # 节点的所有者已不在 users 表（不应出现，如实显示）
            r.update(username=r['user_id'], display_name='（账号不存在）', role='', disabled=False)
    out.sort(key=lambda r: (-(r['results_bytes'] + r['census_bytes']), -(r['last_upload'] or 0), r['username']))
    return [r for r in out if r['user_id'] is not None or r['nodes'] or r['results'] or r['census'] or r['replicas']]


@router.get('/overview')
def overview(refresh: bool = False, authorization: str = Header(default=''), x_admin_token: str = Header(default='')) -> dict:
    _require_admin(authorization, x_admin_token)
    db = D._db()
    census = _census_ready()
    return {'schema': 'data_admin.v1', 'storage': storage(force=refresh), 'counts': counts(db, census),
            'by_user': by_user(db, census), 'census_enabled': census, 'generated_at': time.time()}


def _owners(db) -> Dict[str, Dict[str, Any]]:
    out = {}
    for r in db.execute('SELECT n.node_id, n.name, n.owner_user_id, u.username, u.display_name FROM nodes n '
                        'LEFT JOIN users u ON u.user_id=n.owner_user_id').fetchall():
        out[r['node_id']] = dict(r)
    return out


def _result_items(db, kind: str) -> List[Dict[str, Any]]:
    q = ('SELECT r.sha, r.tm_id, r.node_id, r.model_id, r.model_rev, r.kind, r.size, r.uploaded_at, r.downloads, '
         't.version tm_version, a.status audit, r.summary_json FROM results r LEFT JOIN templates t ON t.tm_id=r.tm_id '
         'LEFT JOIN audits a ON a.result_sha=r.sha')
    args: list = []
    if kind:
        q += ' WHERE r.kind=?'; args.append(kind)
    items = []
    for r in db.execute(q, args).fetchall():
        try:
            s = json.loads(r['summary_json'] or '{}')
        except json.JSONDecodeError:
            s = {}
        canary = (s.get('_canary') or {}).get('status')
        items.append({'source': 'result', 'id': r['sha'], 'sha': r['sha'], 'at': r['uploaded_at'], 'node_id': r['node_id'],
                      'subject': r['tm_id'], 'subject_detail': s.get('shard') if isinstance(s.get('shard'), str) else '',
                      'model': r['model_id'], 'kind': r['kind'], 'bytes': r['size'] or 0, 'files': len(s.get('_manifest') or []),
                      'status': r['audit'] or 'unaudited', 'canary': canary, 'downloads': r['downloads'] or 0})
    for p in db.execute('SELECT p.rowid rid, p.sha, p.node_id, p.model_id, p.uploaded_at, p.note, r.tm_id, r.kind, r.size '
                        'FROM result_replicas p LEFT JOIN results r ON r.sha=p.sha' + (' WHERE r.kind=?' if kind else ''), args).fetchall():
        items.append({'source': 'replica', 'id': f"{p['sha']}:{p['rid']}", 'sha': p['sha'], 'at': p['uploaded_at'], 'node_id': p['node_id'],
                      'subject': p['tm_id'], 'subject_detail': '', 'model': p['model_id'], 'kind': p['kind'], 'bytes': p['size'] or 0,
                      'files': None, 'status': p['note'] or 'replica', 'canary': None, 'downloads': None})
    return items


def _census_items(db, kind: str) -> List[Dict[str, Any]]:
    q = ('SELECT c.lease_id, c.sha, c.job_id, c.shard, c.node_id, c.kind, c.status, c.n_tokens, c.created_at, l.bytes, j.model_key '
         'FROM census_results c LEFT JOIN census_leases l ON l.lease_id=c.lease_id LEFT JOIN census_jobs j ON j.job_id=c.job_id')
    args: list = []
    if kind:
        q += ' WHERE c.kind=?'; args.append(kind)
    return [{'source': 'census', 'id': r['lease_id'], 'sha': r['sha'], 'at': r['created_at'], 'node_id': r['node_id'],
             'subject': r['job_id'], 'subject_detail': r['shard'], 'model': r['model_key'], 'kind': r['kind'],
             'bytes': r['bytes'] or 0, 'files': 1, 'status': r['status'], 'canary': None, 'downloads': None,
             'n_tokens': r['n_tokens']} for r in db.execute(q, args).fetchall()]


@router.get('/uploads')
def uploads(source: str = Query('all', pattern='^(all|result|replica|census)$'), user: str = '', node: str = '',
            kind: str = '', status: str = '', q: str = '', limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
            authorization: str = Header(default=''), x_admin_token: str = Header(default='')) -> dict:
    """user：账号 user_id，或 '-' 表示无主节点。q 匹配 sha、模式族 / 作业、分片、模型、节点名与账号名（不分大小写）。"""
    _require_admin(authorization, x_admin_token)
    db = D._db()
    items: List[Dict[str, Any]] = []
    if source in ('all', 'result', 'replica'):
        items += [x for x in _result_items(db, kind) if source == 'all' or x['source'] == source]
    if source in ('all', 'census') and _census_ready():
        items += _census_items(db, kind)
    owners = _owners(db)
    needle = q.strip().lower()
    out = []
    for x in items:
        o = owners.get(x['node_id']) or {}
        x['node_name'] = o.get('name') or ''
        x['owner'] = {'user_id': o.get('owner_user_id'), 'username': o.get('username') or '',
                      'display_name': o.get('display_name') or ''} if o.get('owner_user_id') else None
        if user == '-' and x['owner']:
            continue
        if user and user != '-' and (not x['owner'] or x['owner']['user_id'] != user):
            continue
        if node and x['node_id'] != node:
            continue
        if status and x['status'] != status:
            continue
        if needle and not any(needle in str(v or '').lower() for v in (
                x['sha'], x['subject'], x['subject_detail'], x['model'], x['node_name'], x['node_id'],
                (x['owner'] or {}).get('username'), (x['owner'] or {}).get('display_name'))):
            continue
        out.append(x)
    out.sort(key=lambda x: -(x['at'] or 0))
    statuses: Dict[str, int] = {}
    for x in out:
        statuses[x['status']] = statuses.get(x['status'], 0) + 1
    return {'total': len(out), 'bytes': sum(x['bytes'] for x in out), 'statuses': statuses,
            'items': out[offset:offset + limit], 'offset': offset, 'limit': limit}


def _json(text: Optional[str]) -> Any:
    try:
        return json.loads(text or 'null')
    except json.JSONDecodeError:
        return None


def _node_info(db, node_id: str) -> Optional[Dict[str, Any]]:
    r = db.execute('SELECT n.node_id, n.name, n.gpu, n.model, n.dtype, n.trust, n.created_at, n.last_seen, n.owner_user_id, '
                   'u.username, u.display_name FROM nodes n LEFT JOIN users u ON u.user_id=n.owner_user_id WHERE n.node_id=?',
                   (node_id,)).fetchone()
    return dict(r) if r else None


@router.get('/uploads/{source}/{item_id}')
def upload_detail(source: str, item_id: str, authorization: str = Header(default=''), x_admin_token: str = Header(default='')) -> dict:
    _require_admin(authorization, x_admin_token)
    db = D._db()
    if source in ('result', 'replica'):
        sha = item_id.split(':', 1)[0]
        r = db.execute('SELECT r.*, t.version tm_version, t.name tm_name FROM results r LEFT JOIN templates t ON t.tm_id=r.tm_id '
                       'WHERE r.sha=?', (sha,)).fetchone()
        if not r:
            raise HTTPException(404, 'result not found')
        s = _json(r['summary_json']) or {}
        audit = db.execute('SELECT status, reviewer, note, created_at FROM audits WHERE result_sha=?', (sha,)).fetchone()
        reps = [dict(p) for p in db.execute('SELECT node_id, task_id, model_id, model_rev, precision, uploaded_at, note '
                                            'FROM result_replicas WHERE sha=? ORDER BY uploaded_at', (sha,)).fetchall()]
        task = db.execute('SELECT task_id, status, claimed_at, done_at, shard, work_unit_id FROM tasks WHERE result_sha=? '
                          'ORDER BY done_at LIMIT 1', (sha,)).fetchone()
        summary = {k: v for k, v in s.items() if not k.startswith('_')}
        text = json.dumps(summary, ensure_ascii=False)
        return {
            'source': 'result', 'sha': sha, 'tm_id': r['tm_id'], 'tm_name': r['tm_name'], 'tm_version': r['tm_version'],
            'kind': r['kind'], 'design_sha': r['design_sha'], 'model_id': r['model_id'], 'model_rev': r['model_rev'], 'seed': r['seed'],
            'bytes': r['size'], 'uploaded_at': r['uploaded_at'], 'downloads': r['downloads'],
            'object_present': (D._DATA_DIR / 'objects' / sha[:2] / sha).is_file(),
            'files': [{'path': f.get('path'), 'size': f.get('size'), 'sha256': f.get('sha256')} for f in s.get('_manifest') or []],
            'upload_meta': s.get('_upload_meta'), 'canary': s.get('_canary'), 'research_work_unit': s.get('_research_work_unit'),
            'audit': dict(audit) if audit else None, 'replicas': reps, 'task': dict(task) if task else None,
            'node': _node_info(db, r['node_id']),
            'summary': summary if len(text) <= 20000 else None, 'summary_keys': sorted(summary), 'summary_chars': len(text),
        }
    if source == 'census':
        if not _census_ready():
            raise HTTPException(404, 'census not enabled')
        r = db.execute('SELECT c.*, l.bytes, l.claimed_at, l.done_at, l.note lease_note, j.model_key, j.status job_status '
                       'FROM census_results c LEFT JOIN census_leases l ON l.lease_id=c.lease_id '
                       'LEFT JOIN census_jobs j ON j.job_id=c.job_id WHERE c.lease_id=?', (item_id,)).fetchone()
        if not r:
            raise HTTPException(404, 'census result not found')
        reps = [dict(p, metrics=_json(p['metrics_json'])) for p in db.execute(
            'SELECT result_a, result_b, sha_a, sha_b, ok, metrics_json, created_at FROM census_replicas '
            'WHERE result_a=? OR result_b=?', (item_id, item_id)).fetchall()]
        for p in reps:
            p.pop('metrics_json', None)
        try:
            import census_service as CS
            present = CS._result_path(r['job_id'], r['sha']).is_file()
        except Exception:
            present = False
        return {
            'source': 'census', 'lease_id': item_id, 'sha': r['sha'], 'job_id': r['job_id'], 'job_status': r['job_status'],
            'shard': r['shard'], 'model_key': r['model_key'], 'kind': r['kind'], 'status': r['status'], 'n_tokens': r['n_tokens'],
            'bytes': r['bytes'], 'claimed_at': r['claimed_at'], 'uploaded_at': r['created_at'], 'object_present': present,
            'validation': _json(r['validation_json']), 'summary': _json(r['summary_json']),
            'reviewed_by': r['reviewed_by'], 'review_note': r['review_note'], 'replica_checks': reps,
            'node': _node_info(db, r['node_id']),
        }
    raise HTTPException(404, 'unknown source')
