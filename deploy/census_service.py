# -*- coding: utf-8 -*-
"""中心服务 · 神经元普查分片（census_job.v1）路由：建作业、派发分片、分块上传、校验、复算比较与合并。

  GET  /api/census/jobs                      公开：作业列表与进度
  GET  /api/census/jobs/{job_id}             公开：作业规范与逐分片状态
  GET  /api/census/jobs/{job_id}/files/{n}   公开：任务包文件（runner、census.py、job.json、切片 token 文件）
  POST /api/census/jobs                      管理：从中心本机的模型目录与语料建作业（冻结 runner 副本）
  POST /api/census/jobs/{job_id}/status      管理：open / paused / closed
  POST /api/census/jobs/{job_id}/merge       管理：合并已完成切片，写入 <AI2050_NATIVE_DIR>/_census/<model_key>/<job_id>/
  POST /api/census/claim                     节点：按模型快照领取一个分片（租约 3h）
  PUT  /api/census/leases/{lease}/heartbeat  节点：续租
  POST /api/census/leases/{lease}/chunk      节点：上传一块原始字节（?seq=）
  POST /api/census/leases/{lease}/finish     节点：提交（sha256 + summary），中心立即校验
  POST /api/census/leases/{lease}/fail       节点：报告失败
  POST /api/census/results/{result}/review   管理：处理复算不一致（accept / reject）

纪律：任务包文件在建作业时冻结（含 runner 与 census.py 副本及其 sha256），节点按 sha256 核对后才执行；中心按切片 token
重算应计入的 token 数与位置集合并逐项校验；按比例抽取的分片由两个不同节点（尽量不同账号）复算，比较不一致的结果暂停合并，
等待管理员处理。校验通过只表示执行产物可信、可以合并，不是证据晋级。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import distributed_service as D  # deploy/ 在 sys.path 上（server.py 与 center_app.py 都会加入）
from shared.dist_protocol import census_job as CJ

router = APIRouter(prefix='/api/census', tags=['census'])

LEASE_SECONDS = 3 * 3600
MAX_CHUNK = 8 * 1024 * 1024
MAX_UPLOAD = 512 * 1024 * 1024
_REPO = Path(__file__).resolve().parent.parent
_PROTO = _REPO / 'shared' / 'dist_protocol'
_merge_lock = threading.Lock()


def native_root() -> Path:
    from server.neuron_census_service import native_root as _nr     # 与单元卡端点读同一目录
    return _nr()


def census_dir() -> Path:
    d = D._DATA_DIR / 'census'
    d.mkdir(parents=True, exist_ok=True)
    return d


def allow_synthetic() -> bool:
    return os.environ.get('AI2050_DIST_ALLOW_SYNTHETIC') == '1'


_schema_done = False


def db():
    global _schema_done
    conn = D._db()
    if not _schema_done:
        with D._db_lock:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS census_jobs(
                    job_id TEXT PRIMARY KEY, status TEXT NOT NULL, model_key TEXT, snapshot_sha256 TEXT,
                    job_sha256 TEXT, created_at REAL, merged_at REAL, merge_json TEXT);
                CREATE TABLE IF NOT EXISTS census_leases(
                    lease_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, shard TEXT NOT NULL, node_id TEXT NOT NULL,
                    status TEXT NOT NULL, claimed_at REAL, lease_until REAL, done_at REAL,
                    next_seq INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0, result_sha TEXT, note TEXT);
                CREATE INDEX IF NOT EXISTS idx_census_leases_shard ON census_leases(job_id, shard);
                CREATE TABLE IF NOT EXISTS census_results(
                    lease_id TEXT PRIMARY KEY, sha TEXT NOT NULL, job_id TEXT NOT NULL, shard TEXT NOT NULL, node_id TEXT NOT NULL,
                    kind TEXT NOT NULL, status TEXT NOT NULL, n_tokens INTEGER,
                    summary_json TEXT, validation_json TEXT, created_at REAL, reviewed_by TEXT, review_note TEXT);
                CREATE INDEX IF NOT EXISTS idx_census_results_shard ON census_results(job_id, shard);
                CREATE TABLE IF NOT EXISTS census_replicas(
                    job_id TEXT, shard TEXT, result_a TEXT, result_b TEXT, sha_a TEXT, sha_b TEXT, ok INTEGER,
                    metrics_json TEXT, created_at REAL, PRIMARY KEY(result_a, result_b));
            ''')
            conn.commit()
        _schema_done = True
    return conn


def _job_path(job_id: str) -> Path:
    if not job_id or any(c in job_id for c in '/\\.') or len(job_id) > 64:
        raise HTTPException(400, 'bad job id')
    return census_dir() / job_id


@lru_cache(maxsize=8)
def _job_cached(job_id: str, stamp: float) -> Dict[str, Any]:
    return json.loads((_job_path(job_id) / 'files' / 'job.json').read_text(encoding='utf-8'))


def load_job(job_id: str) -> Dict[str, Any]:
    p = _job_path(job_id) / 'files' / 'job.json'
    if not p.is_file():
        raise HTTPException(404, f'census job {job_id} not found')
    return _job_cached(job_id, p.stat().st_mtime)


@lru_cache(maxsize=4)
def _counted(job_id: str, s: int) -> np.ndarray:
    job = load_job(job_id)
    with np.load(_job_path(job_id) / 'files' / f'tokens_S{s}.npz') as z:
        sl = {k: z[k] for k in ('windows', 'src', 'win_off')}
    return CJ.counted_gidx(sl, job['skip_head'], job['separator_id'])


def _bundle(job: Dict[str, Any], shard: Dict[str, Any]) -> List[Dict[str, Any]]:
    fdir = _job_path(job['job_id']) / 'files'
    names = ['census.py', 'census_runner.py', 'job.json', f'tokens_S{shard["slice"]}.npz']
    out = []
    for n in names:
        p = fdir / n
        out.append({'name': n, 'size': p.stat().st_size, 'sha256': _sha_file(p),
                    'url': f'/api/census/jobs/{job["job_id"]}/files/{n}'})
    return out


@lru_cache(maxsize=256)
def _sha_file_cached(path: str, size: int, mtime: float) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha_file(p: Path) -> str:
    st = p.stat()
    return _sha_file_cached(str(p), st.st_size, st.st_mtime)


# ---------- 建作业（管理） ----------

class JobCreate(BaseModel):
    job_id: str = Field(pattern=r'^[A-Za-z0-9_-]{3,64}$')
    model_dir: str                       # 中心本机目录：至少有 config.json 与 tokenizer 文件；有权重时可算快照
    model_name: str = ''
    hf_repo: str = ''
    hf_revision: str = ''
    snapshot: Optional[dict] = None      # 中心没有权重时，由持有权重的机器算好的 model_snapshot（sha256 + files）
    corpus_fp: str
    tokens_per_slice: int = Field(1_000_000, ge=1024)
    max_slices: Optional[int] = None
    group_size: int = Field(4, ge=1)
    replica_rate: float = Field(0.1, ge=0.0, le=1.0)
    k: int = Field(32, ge=4, le=64)
    seq_len: int = Field(512, ge=64)
    skip_head: int = Field(16, ge=0)
    batch: int = Field(8, ge=1)
    n_layers: Optional[int] = None       # 缺省读 config.json
    d_ff: Optional[int] = None


def create_job(b: JobCreate) -> Dict[str, Any]:
    jdir = _job_path(b.job_id)
    if jdir.exists():
        raise HTTPException(409, f'census job {b.job_id} already exists')
    mdir = Path(b.model_dir)
    cfg_p = mdir / 'config.json'
    if not cfg_p.is_file() or not (mdir / 'tokenizer.json').is_file():
        raise HTTPException(422, 'model_dir 需要 config.json 与 tokenizer.json')
    cfg = json.loads(cfg_p.read_text(encoding='utf-8'))
    n_layers = b.n_layers or int(cfg.get('num_hidden_layers') or 0)
    d_ff = b.d_ff or int(cfg.get('intermediate_size') or 0)
    if not n_layers or not d_ff:
        raise HTTPException(422, '无法从 config.json 读出层数与 MLP 宽度')
    if b.snapshot:
        snap = b.snapshot
    else:
        from shared.dist_protocol.runner_v3 import model_snapshot
        snap = model_snapshot(mdir)
        if not any(n.endswith(('.safetensors', '.bin')) for n in snap['files']):
            raise HTTPException(422, 'model_dir 没有权重文件：请提供 snapshot')
    cdir = native_root() / '_census' / 'corpus' / b.corpus_fp
    if not (cdir / 'manifest.json').is_file():
        raise HTTPException(422, f'中心本机没有语料 {b.corpus_fp}（{cdir}）')
    cman = json.loads((cdir / 'manifest.json').read_text(encoding='utf-8'))
    streams = []
    for s in cman['sources']:
        with np.load(cdir / s['file']) as z:
            streams.append(z['tokens'])
    model = {'model_key': f"{snap['sha256'][:16]}-bf16", 'name': b.model_name or mdir.name, 'snapshot_sha256': snap['sha256'],
             'files': snap.get('files') or {}, 'hf_repo': b.hf_repo, 'hf_revision': b.hf_revision, 'dtype': 'bf16'}
    job, files = CJ.build_job(b.job_id, model=model, corpus_fp=b.corpus_fp, corpus_manifest=cman, streams=streams,
                              n_layers=n_layers, d_ff=d_ff, tokens_per_slice=b.tokens_per_slice, max_slices=b.max_slices,
                              seq_len=b.seq_len, skip_head=b.skip_head, k=b.k, group_size=b.group_size,
                              replica_rate=b.replica_rate, batch=b.batch)
    fdir = jdir / 'files'
    fdir.mkdir(parents=True)
    for n in ('census.py', 'census_runner.py'):
        shutil.copyfile(_PROTO / n, fdir / n)                      # 冻结副本：仓库代码之后的改动不影响本作业
    job['code'] = {n: _sha_file(fdir / n) for n in ('census.py', 'census_runner.py')}
    job['corpus_manifest_sha256'] = hashlib.sha256((cdir / 'manifest.json').read_bytes()).hexdigest()
    job['created_at'] = time.time()
    for n, data in files.items():
        (fdir / n).write_bytes(data)
    tdir = jdir / 'tokenizer'
    tdir.mkdir()
    for n in ('tokenizer.json', 'tokenizer_config.json', 'config.json'):
        if (mdir / n).is_file():
            shutil.copyfile(mdir / n, tdir / n)
    jbytes = json.dumps(job, ensure_ascii=False, indent=1).encode('utf-8')
    (fdir / 'job.json').write_bytes(jbytes)
    conn = db()
    with D._db_lock:
        conn.execute('INSERT INTO census_jobs(job_id,status,model_key,snapshot_sha256,job_sha256,created_at) VALUES(?,?,?,?,?,?)',
                     (b.job_id, 'open', model['model_key'], snap['sha256'], hashlib.sha256(jbytes).hexdigest(), time.time()))
        conn.commit()
    return {'job_id': b.job_id, 'n_shards': len(job['shards']), 'n_slices': job['n_slices'],
            'layer_groups': len(job['layer_groups']), 'model_key': model['model_key']}


@router.post('/jobs')
def jobs_create(b: JobCreate, x_admin_token: str = Header(default='')) -> dict:
    D._auth_admin(x_admin_token)
    return create_job(b)


class JobStatus(BaseModel):
    status: str = Field(pattern='^(open|paused|closed)$')


@router.post('/jobs/{job_id}/status')
def jobs_status(job_id: str, b: JobStatus, x_admin_token: str = Header(default='')) -> dict:
    D._auth_admin(x_admin_token)
    load_job(job_id)
    conn = db()
    with D._db_lock:
        conn.execute('UPDATE census_jobs SET status=? WHERE job_id=?', (b.status, job_id))
        conn.commit()
    return {'job_id': job_id, 'status': b.status}


# ---------- 进度（公开） ----------

def _expire(conn) -> None:
    now = time.time()
    with D._db_lock:
        conn.execute("UPDATE census_leases SET status='expired' WHERE status='claimed' AND lease_until<?", (now,))
        conn.commit()


def shard_states(job: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    conn = db()
    jid = job['job_id']
    res: Dict[str, List[dict]] = {}
    for r in conn.execute('SELECT sha,shard,node_id,status,kind FROM census_results WHERE job_id=?', (jid,)):
        res.setdefault(r['shard'], []).append(dict(r))
    act: Dict[str, int] = {}
    for r in conn.execute("SELECT shard,COUNT(*) AS n FROM census_leases WHERE job_id=? AND status='claimed' GROUP BY shard", (jid,)):
        act[r['shard']] = r['n']
    reps = {(r['shard']): bool(r['ok']) for r in conn.execute('SELECT shard,ok FROM census_replicas WHERE job_id=?', (jid,))}
    out = {}
    for sh in job['shards']:
        rs = res.get(sh['shard'], [])
        acc = [r for r in rs if r['status'] == 'accepted']
        disputed = [r for r in rs if r['status'] == 'disputed']
        need = sh['replicas']
        if disputed:
            state = 'disputed'
        elif len(acc) >= need:
            state = 'done'
        elif acc:
            state = 'replica_pending'
        elif act.get(sh['shard']):
            state = 'running'
        else:
            state = 'queued'
        out[sh['shard']] = {'state': state, 'accepted': len(acc), 'replicas': need, 'active': act.get(sh['shard'], 0),
                            'rejected': sum(r['status'] == 'rejected' for r in rs), 'replica_ok': reps.get(sh['shard'])}
    return out


def job_progress(job_id: str) -> Dict[str, Any]:
    job = load_job(job_id)
    row = db().execute('SELECT * FROM census_jobs WHERE job_id=?', (job_id,)).fetchone()
    st = shard_states(job)
    counts: Dict[str, int] = {}
    for v in st.values():
        counts[v['state']] = counts.get(v['state'], 0) + 1
    done_slices = [s['slice'] for s in job['slices']
                   if all(st[sh['shard']]['state'] == 'done' for sh in job['shards'] if sh['slice'] == s['slice'])]
    merge = json.loads(row['merge_json']) if row and row['merge_json'] else None
    return {'job_id': job_id, 'status': row['status'] if row else 'unknown', 'model': {k: job['model'].get(k) for k in
            ('model_key', 'name', 'hf_repo', 'hf_revision', 'snapshot_sha256')},
            'corpus_fp': job['corpus_fp'], 'n_shards': len(job['shards']), 'n_slices': job['n_slices'],
            'tokens_per_slice': job['tokens_per_slice'], 'layer_groups': job['layer_groups'], 'counts': counts,
            'slices_done': done_slices, 'tokens_done': int(sum(job['slices'][s]['expected_n'] for s in done_slices)),
            'merged': merge, 'replica_rate': job['replica_rate']}


@router.get('/jobs')
def jobs_list() -> dict:
    _expire(db())
    rows = db().execute('SELECT job_id FROM census_jobs ORDER BY created_at DESC').fetchall()
    return {'schema': 'census_jobs.v1', 'jobs': [job_progress(r['job_id']) for r in rows],
            'note': '分片状态：queued 待领取 / running 执行中 / replica_pending 等待第二个节点复算 / done 完成 / disputed 复算不一致待审'}


@router.get('/jobs/{job_id}')
def jobs_get(job_id: str) -> dict:
    job = load_job(job_id)
    st = shard_states(job)
    return {**job_progress(job_id), 'shards': [{**sh, **st[sh['shard']]} for sh in job['shards']],
            'slices': job['slices'], 'code': job.get('code'), 'note': job['note']}


@router.get('/jobs/{job_id}/files/{name}')
def jobs_file(job_id: str, name: str):
    if '/' in name or '\\' in name or name.startswith('.'):
        raise HTTPException(400, 'bad file name')
    p = _job_path(job_id) / 'files' / name
    if not p.is_file():
        raise HTTPException(404, 'file not found')
    return FileResponse(p, filename=name, media_type='application/octet-stream')


# ---------- 领取、上传、校验（节点） ----------

class ClaimIn(BaseModel):
    model_snapshot: str = ''
    job_id: str = ''
    mode: str = Field('real', pattern='^(real|synthetic)$')


def _owner(conn, node_id: str) -> Optional[str]:
    r = conn.execute('SELECT owner_user_id FROM nodes WHERE node_id=?', (node_id,)).fetchone()
    return r['owner_user_id'] if r else None


@router.post('/claim')
def claim(b: ClaimIn, response: Response, x_node_token: str = Header(default='')):
    n = D._auth_node(x_node_token)
    if b.mode == 'synthetic' and not allow_synthetic():
        raise HTTPException(403, 'synthetic 只在沙盒中心（AI2050_DIST_ALLOW_SYNTHETIC=1）可用')
    conn = db()
    _expire(conn)
    me_owner = _owner(conn, n['node_id'])
    sql = "SELECT job_id FROM census_jobs WHERE status='open'"
    args: list = []
    if b.job_id:
        sql += ' AND job_id=?'; args.append(b.job_id)
    if b.mode == 'real':
        sql += ' AND snapshot_sha256=?'; args.append(b.model_snapshot)
    now = time.time()
    with D._db_lock:
        for jr in conn.execute(sql + ' ORDER BY created_at', args).fetchall():
            job = load_job(jr['job_id'])
            res: Dict[str, list] = {}
            for r in conn.execute("SELECT shard,node_id,status FROM census_results WHERE job_id=? AND status IN ('accepted','disputed')",
                                  (job['job_id'],)):
                res.setdefault(r['shard'], []).append(r)
            leases: Dict[str, List[str]] = {}
            for r in conn.execute("SELECT shard,node_id FROM census_leases WHERE job_id=? AND status='claimed'", (job['job_id'],)):
                leases.setdefault(r['shard'], []).append(r['node_id'])
            first, second = [], []
            for sh in job['shards']:
                done_nodes = [r['node_id'] for r in res.get(sh['shard'], [])]
                if any(r['status'] == 'disputed' for r in res.get(sh['shard'], [])):
                    continue
                busy = leases.get(sh['shard'], [])
                if n['node_id'] in done_nodes or n['node_id'] in busy:
                    continue
                if len(done_nodes) + len(busy) >= sh['replicas']:
                    continue
                if done_nodes:
                    owners = {_owner(conn, x) for x in done_nodes}
                    if me_owner and me_owner in owners:
                        continue                      # 复算必须来自另一个账号
                    second.append(sh)
                else:
                    first.append(sh)
            pick = (first or second or [None])[0]
            if pick is None:
                continue
            lease_id = 'CL-' + uuid.uuid4().hex[:12]
            conn.execute('INSERT INTO census_leases(lease_id,job_id,shard,node_id,status,claimed_at,lease_until) VALUES(?,?,?,?,?,?,?)',
                         (lease_id, job['job_id'], pick['shard'], n['node_id'], 'claimed', now, now + LEASE_SECONDS))
            conn.execute('UPDATE nodes SET last_seen=? WHERE node_id=?', (now, n['node_id']))
            conn.commit()
            return {'lease_id': lease_id, 'job_id': job['job_id'], 'shard': pick['shard'], 'spec': pick,
                    'model': {k: job['model'].get(k) for k in ('model_key', 'name', 'hf_repo', 'hf_revision', 'snapshot_sha256')},
                    'files': _bundle(job, pick), 'lease_until': now + LEASE_SECONDS, 'mode': b.mode,
                    'chunk_bytes': MAX_CHUNK}
    response.status_code = 204
    return None


def _lease(lease_id: str, node_id: str):
    r = db().execute('SELECT * FROM census_leases WHERE lease_id=?', (lease_id,)).fetchone()
    if not r or r['node_id'] != node_id:
        raise HTTPException(404, 'lease not found')
    return r


def _upload_path(lease_id: str) -> Path:
    d = census_dir() / '_uploads'
    d.mkdir(exist_ok=True)
    return d / f'{lease_id}.part'


@router.put('/leases/{lease_id}/heartbeat')
def heartbeat(lease_id: str, x_node_token: str = Header(default='')) -> dict:
    n = D._auth_node(x_node_token)
    r = _lease(lease_id, n['node_id'])
    if r['status'] != 'claimed':
        raise HTTPException(409, f"lease status={r['status']}")
    until = time.time() + LEASE_SECONDS
    conn = db()
    with D._db_lock:
        conn.execute('UPDATE census_leases SET lease_until=? WHERE lease_id=?', (until, lease_id))
        conn.execute('UPDATE nodes SET last_seen=? WHERE node_id=?', (time.time(), n['node_id']))
        conn.commit()
    return {'lease_id': lease_id, 'lease_until': until}


@router.post('/leases/{lease_id}/chunk')
async def chunk(lease_id: str, request: Request, seq: int = Query(..., ge=0), x_node_token: str = Header(default='')) -> dict:
    n = D._auth_node(x_node_token)
    r = _lease(lease_id, n['node_id'])
    if r['status'] != 'claimed':
        raise HTTPException(409, f"lease status={r['status']}")
    data = await request.body()
    if len(data) > MAX_CHUNK:
        raise HTTPException(413, 'chunk too large')
    if seq != r['next_seq']:
        if seq < r['next_seq']:
            return {'seq': seq, 'duplicate': True, 'next_seq': r['next_seq']}
        raise HTTPException(409, f"expected seq {r['next_seq']}")
    if r['bytes'] + len(data) > MAX_UPLOAD:
        raise HTTPException(413, 'upload too large')
    p = _upload_path(lease_id)
    with p.open('ab' if seq else 'wb') as fh:
        fh.write(data)
    conn = db()
    with D._db_lock:
        conn.execute('UPDATE census_leases SET next_seq=?, bytes=? WHERE lease_id=?', (seq + 1, r['bytes'] + len(data), lease_id))
        conn.commit()
    return {'seq': seq, 'next_seq': seq + 1}


class FinishIn(BaseModel):
    sha256: str
    summary: dict


class FailIn(BaseModel):
    note: str = ''


@router.post('/leases/{lease_id}/fail')
def fail(lease_id: str, b: FailIn, x_node_token: str = Header(default='')) -> dict:
    n = D._auth_node(x_node_token)
    r = _lease(lease_id, n['node_id'])
    conn = db()
    with D._db_lock:
        if r['status'] == 'claimed':
            conn.execute("UPDATE census_leases SET status='failed', done_at=?, note=? WHERE lease_id=?", (time.time(), b.note[:500], lease_id))
            conn.commit()
    _upload_path(lease_id).unlink(missing_ok=True)
    return {'lease_id': lease_id, 'status': 'failed'}


def _result_path(job_id: str, sha: str) -> Path:
    d = _job_path(job_id) / 'results'
    d.mkdir(exist_ok=True)
    return d / f'{sha}.npz'


def _load_result(job_id: str, sha: str):
    return CJ.load_shard_npz(_result_path(job_id, sha).read_bytes())


@router.post('/leases/{lease_id}/finish')
def finish(lease_id: str, b: FinishIn, x_node_token: str = Header(default='')) -> dict:
    n = D._auth_node(x_node_token)
    r = _lease(lease_id, n['node_id'])
    if r['status'] == 'done' and r['result_sha'] == b.sha256:
        return {'lease_id': lease_id, 'idempotent_replay': True, 'sha': b.sha256}
    if r['status'] != 'claimed':
        raise HTTPException(409, f"lease status={r['status']}")
    p = _upload_path(lease_id)
    if not p.is_file():
        raise HTTPException(409, 'nothing uploaded')
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != b.sha256:
        raise HTTPException(409, 'sha256 mismatch: 上传字节与声明不一致')
    job = load_job(r['job_id'])
    sh = CJ.shard_spec(job, r['shard'])
    s = b.summary or {}
    kind = s.get('status') if s.get('status') in ('real', 'synthetic') else 'invalid'
    errors: List[str] = []
    if kind == 'synthetic' and not allow_synthetic():
        errors.append('synthetic 结果只在沙盒中心接受')
    if kind == 'invalid':
        errors.append('summary.status 必须是 real 或 synthetic')
    if s.get('job_id') != job['job_id'] or s.get('shard') != r['shard']:
        errors.append('summary 的作业或分片与租约不符')
    if kind == 'real' and s.get('model_snapshot') != job['model']['snapshot_sha256']:
        errors.append('模型快照与作业不符')
    code = job.get('code') or {}
    if s.get('runner_sha256') != code.get('census_runner.py') or s.get('census_sha256') != code.get('census.py'):
        errors.append('runner 或 census.py 与冻结任务包不符')
    validation: Dict[str, Any] = {'ok': False, 'errors': []}
    try:
        layers = CJ.load_shard_npz(data)
        validation = CJ.validate_shard(job, r['shard'], layers, _counted(job['job_id'], sh['slice']))
    except Exception as exc:  # 损坏的 npz
        validation = {'ok': False, 'errors': [f'无法解析结果：{type(exc).__name__}: {exc}'[:300]]}
    errors += validation['errors']
    status = 'accepted' if not errors else 'rejected'
    conn = db()
    replica = None
    with D._db_lock:
        if status == 'accepted':
            _result_path(job['job_id'], sha).write_bytes(data)
        conn.execute('INSERT OR REPLACE INTO census_results(lease_id,sha,job_id,shard,node_id,kind,status,n_tokens,summary_json,'
                     'validation_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                     (lease_id, sha, job['job_id'], r['shard'], n['node_id'], kind, status, s.get('n_tokens_counted'),
                      json.dumps(s, ensure_ascii=False)[:20000], json.dumps({'ok': not errors, 'errors': errors[:20]}, ensure_ascii=False),
                      time.time()))
        conn.execute('UPDATE census_leases SET status=?, done_at=?, result_sha=?, note=? WHERE lease_id=?',
                     ('done' if status == 'accepted' else 'rejected', time.time(), sha, '; '.join(errors)[:500], lease_id))
        conn.execute('UPDATE nodes SET last_seen=? WHERE node_id=?', (time.time(), n['node_id']))
        conn.commit()
        if status == 'accepted':
            # 每次上传各占一行（两个节点逐字节相同的结果 sha 相同，不能互相覆盖）
            others = conn.execute("SELECT lease_id,sha FROM census_results WHERE job_id=? AND shard=? AND status='accepted' "
                                  'AND lease_id<>?', (job['job_id'], r['shard'], lease_id)).fetchall()
            for o in others:
                cmp = CJ.compare_replicas(_load_result(job['job_id'], o['sha']), layers, job.get('replica_tol'))
                conn.execute('INSERT OR REPLACE INTO census_replicas(job_id,shard,result_a,result_b,sha_a,sha_b,ok,metrics_json,created_at) '
                             'VALUES(?,?,?,?,?,?,?,?,?)', (job['job_id'], r['shard'], o['lease_id'], lease_id, o['sha'], sha,
                                                         int(cmp['ok']), json.dumps(cmp), time.time()))
                if not cmp['ok']:
                    conn.execute("UPDATE census_results SET status='disputed' WHERE lease_id IN (?,?)", (o['lease_id'], lease_id))
                replica = cmp
            conn.commit()
    p.unlink(missing_ok=True)
    return {'lease_id': lease_id, 'result_id': lease_id, 'sha': sha, 'status': status, 'errors': errors[:20],
            'replica': None if replica is None else {'ok': replica['ok']}}


class ReviewIn(BaseModel):
    action: str = Field(pattern='^(accept|reject)$')
    note: str = ''


@router.post('/results/{result_id}/review')
def review(result_id: str, b: ReviewIn, x_admin_token: str = Header(default='')) -> dict:
    """处理复算不一致：管理员在对照两份结果后接受其中一份（另一份随之驳回），或驳回这一份。"""
    D._auth_admin(x_admin_token)
    conn = db()
    r = conn.execute('SELECT * FROM census_results WHERE lease_id=?', (result_id,)).fetchone()
    if not r:
        raise HTTPException(404, 'result not found')
    with D._db_lock:
        if b.action == 'accept':
            conn.execute("UPDATE census_results SET status='rejected', review_note=? WHERE job_id=? AND shard=? AND status='disputed' "
                         'AND lease_id<>?', (f'复算不一致，管理员接受了 {result_id}', r['job_id'], r['shard'], result_id))
            conn.execute("UPDATE census_results SET status='accepted', reviewed_by='admin', review_note=? WHERE lease_id=?", (b.note, result_id))
        else:
            conn.execute("UPDATE census_results SET status='rejected', reviewed_by='admin', review_note=? WHERE lease_id=?", (b.note, result_id))
        conn.commit()
    return {'result_id': result_id, 'action': b.action}


@router.get('/review')
def review_queue() -> dict:
    rows = db().execute("SELECT lease_id AS result_id,sha,job_id,shard,node_id,created_at FROM census_results WHERE status='disputed' "
                        'ORDER BY created_at').fetchall()
    reps = db().execute('SELECT * FROM census_replicas WHERE ok=0 ORDER BY created_at DESC LIMIT 100').fetchall()
    return {'disputed': [dict(r) for r in rows],
            'replica_checks': [{**{k: r[k] for k in ('job_id', 'shard', 'result_a', 'result_b', 'sha_a', 'sha_b', 'created_at')},
                                'metrics': json.loads(r['metrics_json'])} for r in reps]}


# ---------- 合并 ----------

def merge_job(job_id: str) -> Dict[str, Any]:
    """合并全部层组都已完成的切片；写入原生普查目录，单元卡端点（/api/neurons/*）随即可读。"""
    import io
    job = load_job(job_id)
    st = shard_states(job)
    slices = [s['slice'] for s in job['slices']
              if all(st[sh['shard']]['state'] == 'done' for sh in job['shards'] if sh['slice'] == s['slice'])]
    if 0 not in slices:
        raise HTTPException(409, '切片 0（含读出）尚未完成，不能合并')
    conn = db()
    picks = []
    for sh in job['shards']:
        if sh['slice'] not in slices:
            continue
        r = conn.execute("SELECT sha FROM census_results WHERE job_id=? AND shard=? AND status='accepted' ORDER BY created_at LIMIT 1",
                         (job_id, sh['shard'])).fetchone()
        picks.append((sh['shard'], r['sha']))
    with _merge_lock:
        merged = CJ.merge_job(job, ((sid, _load_result(job_id, sha)) for sid, sha in picks))
        out = native_root() / '_census' / job['model']['model_key'] / job_id
        tmp = out.with_name(out.name + '.tmp')
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        files = []
        for L in sorted(merged):
            buf = io.BytesIO()
            np.savez(buf, **merged[L])
            data = buf.getvalue()
            (tmp / f'L{L}.npz').write_bytes(data)
            files.append({'path': f'L{L}.npz', 'block': L, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        man = CJ.merged_manifest(job, merged, files, str((_job_path(job_id) / 'tokenizer').resolve()), len(picks))
        man['distributed']['slices'] = slices
        man['distributed']['shards'] = [{'shard': s, 'sha': h} for s, h in picks]
        mbytes = json.dumps(man, ensure_ascii=False, indent=1).encode('utf-8')
        (tmp / 'manifest.json').write_bytes(mbytes)
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
    info = {'dir': str(out), 'slices': slices, 'n_tokens': man['n_tokens_counted'], 'n_results': len(picks),
            'manifest_sha256': hashlib.sha256(mbytes).hexdigest(), 'merged_at': time.time()}
    with D._db_lock:
        conn.execute('UPDATE census_jobs SET merged_at=?, merge_json=? WHERE job_id=?', (info['merged_at'], json.dumps(info), job_id))
        conn.commit()
    try:  # 让单元卡端点看到新合并结果
        from server import neuron_census_service as NS
        for fn in (NS.census_manifest, NS.census_layer, NS.readout_index):
            fn.cache_clear()
    except Exception:
        pass
    return info


def auto_merge() -> List[str]:
    """后台定时调用：已完成的切片比上次合并多时重新合并（单元卡随之更新）。返回本次合并的作业。"""
    done = []
    for r in db().execute("SELECT job_id, merge_json FROM census_jobs WHERE status IN ('open','closed')").fetchall():
        try:
            prog = job_progress(r['job_id'])
            merged = set((json.loads(r['merge_json']) or {}).get('slices') or []) if r['merge_json'] else set()
            if 0 in prog['slices_done'] and set(prog['slices_done']) - merged:
                merge_job(r['job_id'])
                done.append(r['job_id'])
        except Exception as exc:  # 单个作业合并失败不影响其他作业；下次刷新重试
            print(f'[census] auto merge {r["job_id"]} failed: {type(exc).__name__}: {exc}', flush=True)
    return done


@router.post('/jobs/{job_id}/merge')
def jobs_merge(job_id: str, x_admin_token: str = Header(default='')) -> dict:
    D._auth_admin(x_admin_token)
    return merge_job(job_id)

