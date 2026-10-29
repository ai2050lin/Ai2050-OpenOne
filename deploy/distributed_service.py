# -*- coding: utf-8 -*-
"""AI2050 分布式研发平台服务（中心节点）—— design/distributed_platform_plan_v1.md S2-S6

双重运行方式:
  1) 挂载进主客户端后端:  server.py 里 `app.include_router(router)`（开发/演示用，:5001）
  2) CentOS 独立部署:     python -m server.distributed_service --host 0.0.0.0 --port 5010
                          （见 deploy/README_CENTOS.md 与 deploy/ai2050-distributed.service）

组成（对应方案编号）:
  S2 模板注册表   POST /api/templates（admin，design_sha=规范化 corpus+runner 哈希，预注册冻结）
  S3 节点与调度   /api/nodes/register · /api/tasks/claim（租约 6h）· /api/tasks/{id}/heartbeat
  S4 结果库       /api/results/upload/init|chunk|finish（sha256 内容寻址，design_sha 不符拒收；
                  敏感原始数据不上传——只收统计量/均值/summary，单文件上限 64MB）
  S5 公开只读     /api/distributed/summary · /api/templates · /api/results · /api/results/{sha} · /api/news
  S9 实时模型     /api/live/connect|status|disconnect（读本机 HF 模型 config+权重头，不加载权重；空间透镜「实时分析」） · /api/tasks（只读调度队列）
  S9c 模型库      /api/models/local|download|downloads（本机模型扫描 + huggingface_hub 后台下载；空间透镜·模型库弹窗）
  S6 聚合 AGG-v0/v1  /api/agg/{tm_id}（v0 分桶回显）· /api/agg/{tm_id}/v1（v1 桶内 η² 跨节点按因子汇总；跨桶平均仍禁止）
  S7 研究包       /api/kits · /api/kits/{kit_id}/bundle（语言模板×分析技术整包下载：
                  bundle 文件 + 满足输入契约的结果清单 + README；公开只读，design_sha 可本地复核）
  S8 纪律         节点 token 哈希存储；AI API key 与本服务无关（永不出本机）
  S10 协议 v3     （shared/dist_protocol，研发透镜·模板工作室/试跑与冻结/审计队列 + 总览/空间/路线消费）
                  /api/grammar/library|preview|register · /api/grammar/{tm_id}/report
                  /api/agg/{tm_id}/v2（充分统计量精确合并，金丝雀/审计/行为门过滤）
                  /api/coverage · /api/audit/queue · POST /api/audit/{sha} · /api/pipeline/overview · /api/scoreboard
                  /api/analysis/{tm_id}（逐条向量规律分析 + 3D 响应空间坐标）

存储（env AI2050_DIST_DIR，默认 <repo>/distributed_data）:
  dist.sqlite3            节点/模板/任务/结果索引
  objects/<aa>/<sha256>   内容寻址文件
  bundles/<tm_id>/v{N}/   模板包（corpus.json / runner.py / contract.json）
  news_cache.json         新闻缓存（arXiv 1h；失败回退内置列表）
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

try:  # 包内挂载（server.py / python -m server.distributed_service）
    from server.dist_templates_seed import NEWS_FALLBACK, seed_templates
except ImportError:  # 直接脚本运行
    from dist_templates_seed import NEWS_FALLBACK, seed_templates  # type: ignore

ROUTER_VERSION = 'distributed-v1(M1)+protocol-v3'
LEASE_SECONDS = 6 * 3600
NODE_ONLINE_WINDOW = 600.0
CHUNK_SIZE = 1_500_000          # base64 前，字节数
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
ARXIV_URL = ('https://export.arxiv.org/api/query?search_query='
             'all:%22mechanistic+interpretability%22'
             '&sortBy=submittedDate&sortOrder=descending&max_results=10')
NEWS_TTL = 3600.0
NEWS_FALLBACK_TTL = 600.0     # builtin-fallback 兜底列表的短缓存 TTL

_DATA_DIR = Path(os.environ.get('AI2050_DIST_DIR',
                                Path(__file__).resolve().parent.parent / 'distributed_data'))
_WORKSPACE = Path(__file__).resolve().parent.parent
_RESEARCH_OS = _WORKSPACE / 'ai2050_research_os'
# 管理令牌：生产必须显式设置 AI2050_DIST_ADMIN_TOKEN（center_app 入口强制）。
# 未设置时（本机开发/临时进程）生成一次性随机令牌并打印到日志——不使用固定默认值，
# 防止绕过 center_app 直接启动的服务带着可猜测的开发令牌上线（client_architecture_review P0-3）。
ADMIN_TOKEN = os.environ.get('AI2050_DIST_ADMIN_TOKEN', '')
if not ADMIN_TOKEN:
    ADMIN_TOKEN = secrets.token_hex(16)
    print('[distributed_service] AI2050_DIST_ADMIN_TOKEN 未设置：本次进程使用随机管理令牌 '
          f'{ADMIN_TOKEN}（管理员账号会话同样可用；写进环境变量可固定）', flush=True)
NEWS_OFFLINE = os.environ.get('AI2050_DIST_NEWS_OFFLINE', '') == '1'

router = APIRouter(prefix='/api', tags=['distributed'])

# dist_protocol v3（仓库内 shared/dist_protocol，随 git 分发；deploy/ 只需与仓库同目录部署）
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
try:
    from shared.dist_protocol import canary as _canary
    from shared.dist_protocol import grammar as _grammar
    from shared.dist_protocol import pipeline as _pipeline
    from shared.dist_protocol import autotest as _autotest
    _PROTO_ERR = ''
except Exception as _exc:  # numpy 缺失或仓库不完整：v3 端点返回 503，legacy 端点不受影响
    _canary = _grammar = _pipeline = _autotest = None  # type: ignore[assignment]
    _PROTO_ERR = f'{type(_exc).__name__}: {_exc}'
from shared.dist_protocol import accounts as _accounts  # noqa: E402  只依赖标准库：numpy 缺失时账号仍可用
_GRAMMAR_DIR = _WORKSPACE / 'shared' / 'dist_protocol' / 'grammars'
_RUNNER_V3 = _WORKSPACE / 'shared' / 'dist_protocol' / 'runner_v3.py'

_db_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_upload_lock = threading.Lock()   # finish 串行：同一上传重复提交时只入库一次
UPLOAD_TTL = 24 * 3600            # 未完成的上传在磁盘上保留的时长；完成记录保留同样久，供 finish 重放


# ---------- 存储层 ----------

_tls = threading.local()
_init_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    """每个线程一个 SQLite 连接（WAL 下读写并发安全）。同一个连接被多个请求线程同时使用会触发
    sqlite3.InterfaceError（客户端同时请求总览、题库、谱图时出现）；写操作仍由 _db_lock 串行。"""
    if _conn is None:
        with _init_lock:
            if _conn is None:
                _init_db()
    conn = getattr(_tls, 'conn', None)
    if conn is None:
        conn = sqlite3.connect(str(_DATA_DIR / 'dist.sqlite3'), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout=30000')
        _tls.conn = conn
    return conn


def _init_db() -> sqlite3.Connection:
    """建表、迁移、播种（只执行一次）。"""
    global _conn
    if _conn is None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        (_DATA_DIR / 'objects').mkdir(exist_ok=True)
        (_DATA_DIR / 'bundles').mkdir(exist_ok=True)
        _conn = sqlite3.connect(str(_DATA_DIR / 'dist.sqlite3'), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS nodes(
                node_id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, name TEXT,
                gpu TEXT, model TEXT, dtype TEXT, trust TEXT DEFAULT 'unverified',
                created_at REAL, last_seen REAL);
            CREATE TABLE IF NOT EXISTS templates(
                tm_id TEXT PRIMARY KEY, version INTEGER, name TEXT, dim TEXT,
                description TEXT, metric_pin TEXT, model_pin TEXT,
                design_sha TEXT NOT NULL, status TEXT DEFAULT 'open',
                bundle_dir TEXT, created_at REAL,
                research_adapter_json TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE IF NOT EXISTS tasks(
                task_id TEXT PRIMARY KEY, tm_id TEXT, node_id TEXT,
                status TEXT DEFAULT 'claimed', claimed_at REAL, lease_until REAL,
                done_at REAL, result_sha TEXT, note TEXT,
                work_unit_id TEXT, research_ref_json TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE IF NOT EXISTS results(
                sha TEXT PRIMARY KEY, tm_id TEXT, version INTEGER, node_id TEXT,
                model_id TEXT, model_rev TEXT, seed INTEGER, kind TEXT,
                design_sha TEXT, size INTEGER, summary_json TEXT,
                uploaded_at REAL, downloads INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS audits(
                result_sha TEXT PRIMARY KEY, status TEXT NOT NULL, reviewer TEXT,
                note TEXT, checks_json TEXT, created_at REAL);
            CREATE TABLE IF NOT EXISTS result_replicas(
                sha TEXT, node_id TEXT, task_id TEXT, model_id TEXT, model_rev TEXT,
                precision TEXT, uploaded_at REAL, note TEXT);
            CREATE TABLE IF NOT EXISTS canary_refs(
                model_key TEXT PRIMARY KEY, reference_json TEXT NOT NULL,
                source_sha TEXT, created_at REAL);
        ''')
        # Additive migration for local centers created before the Research OS bridge.
        for table, column, declaration in (
            ('templates', 'research_adapter_json', "TEXT NOT NULL DEFAULT '{}'"),
            ('tasks', 'work_unit_id', 'TEXT'),
            ('tasks', 'research_ref_json', "TEXT NOT NULL DEFAULT '{}'"),
            ('tasks', 'shard', 'TEXT'),
        ):
            columns = {row['name'] for row in _conn.execute(f'PRAGMA table_info({table})')}
            if column not in columns:
                _conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {declaration}')
        _conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_work_unit_unique '
                      'ON tasks(work_unit_id) WHERE work_unit_id IS NOT NULL')
        _conn.commit()
        _accounts.ensure_schema(_conn)              # users / sessions / login_failures / node_bind_codes + nodes.owner_user_id
        _seed_if_empty()
    return _conn


def _canonical_design_sha(tm_id: str, version: int, corpus: dict, runner: str,
                          research_adapter: Optional[dict] = None) -> str:
    design = {'tm_id': tm_id, 'version': version, 'corpus': corpus,
              'runner_sha256': hashlib.sha256(runner.encode('utf-8')).hexdigest()}
    # Preserve hashes of legacy templates. Research-aware adapters become part
    # of the immutable design identity when present.
    if research_adapter:
        design['research_adapter'] = research_adapter
    payload = json.dumps(
        design,
        sort_keys=True, ensure_ascii=False, separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _register_template(cur_row: dict) -> str:
    db = _db()
    design_sha = _canonical_design_sha(
        cur_row['tm_id'], cur_row['version'], cur_row['corpus'], cur_row['runner'],
        cur_row.get('research_adapter'))
    bundle_dir = _DATA_DIR / 'bundles' / cur_row['tm_id'] / f"v{cur_row['version']}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / 'corpus.json').write_text(
        json.dumps(cur_row['corpus'], ensure_ascii=False, indent=1), encoding='utf-8')
    (bundle_dir / 'runner.py').write_text(cur_row['runner'], encoding='utf-8')
    contract = {'tm_id': cur_row['tm_id'], 'version': cur_row['version'],
                'name': cur_row['name'], 'dim': cur_row['dim'],
                'design_sha': design_sha, 'metric_pin': cur_row['metric_pin'],
                'model_pin': cur_row['model_pin'], 'preregistered': True,
                'description': cur_row['description']}
    if cur_row.get('research_adapter'):
        contract['research_adapter'] = cur_row['research_adapter']
    (bundle_dir / 'contract.json').write_text(
        json.dumps(contract, ensure_ascii=False, indent=1), encoding='utf-8')
    with _db_lock:
        db.execute(
            'INSERT OR REPLACE INTO templates(tm_id,version,name,dim,description,'
            'metric_pin,model_pin,design_sha,status,bundle_dir,created_at,research_adapter_json) '
            'VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (cur_row['tm_id'], cur_row['version'], cur_row['name'], cur_row['dim'],
             cur_row['description'], cur_row['metric_pin'], cur_row['model_pin'],
             design_sha, 'open', str(bundle_dir), time.time(),
             json.dumps(cur_row.get('research_adapter') or {}, ensure_ascii=False, sort_keys=True)))
        db.commit()
    return design_sha


def _seed_if_empty() -> None:
    """对齐式播种（v2.1）：种子表=期望状态；按 (tm_id, 最高已注册版本) 对齐——
    已注册且版本 >= 种子版本则不动（bundle/design_sha 快照保持冻结），
    缺失的 tm_id 或种子里更高版本（runner 修复走升版）补注册。"""
    have = {r['tm_id']: r['v'] for r in _conn.execute(
        'SELECT tm_id, MAX(version) AS v FROM templates GROUP BY tm_id').fetchall()}
    added = 0
    for t in seed_templates():
        if t['tm_id'] in have and have[t['tm_id']] >= t['version']:
            continue
        _register_template(t)
        added += 1
    if added:
        print(f"[distributed] seeded {added} template rows (registry aligned)")


def _bundle_files(bundle_dir: str) -> Dict[str, str]:
    d = Path(bundle_dir)
    return {f.name: f.read_text(encoding='utf-8') for f in sorted(d.iterdir())
            if f.is_file() and f.suffix in ('.json', '.py')}


# ---------- 鉴权 ----------

def _auth_node(x_node_token: str) -> sqlite3.Row:
    if not x_node_token:
        raise HTTPException(401, 'missing X-Node-Token')
    h = hashlib.sha256(x_node_token.encode()).hexdigest()
    row = _db().execute(
        'SELECT * FROM nodes WHERE token_hash=?', (h,)).fetchone()
    if not row:
        raise HTTPException(401, 'unknown node token')
    return row


def _auth_admin(x_admin_token: str) -> None:
    """写操作授权：中心管理令牌，或管理员账号的会话令牌（客户端登录管理员账号后不必再填令牌）。"""
    if ADMIN_TOKEN and x_admin_token and hmac.compare_digest(x_admin_token, ADMIN_TOKEN):
        return
    user = _accounts.session_user(_db(), x_admin_token or '') if x_admin_token else None
    if user and user['role'] == 'admin':
        return
    raise HTTPException(401, 'bad admin token')


def _os_json(relative_path: str) -> Any:
    path = (_RESEARCH_OS / relative_path).resolve()
    if not path.is_relative_to(_RESEARCH_OS.resolve()) or not path.is_file():
        raise HTTPException(503, f'Research OS source unavailable: {relative_path}')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(503, f'Research OS source invalid: {relative_path}: {exc}')


def _rows_by_id(relative_path: str, key: str) -> dict[str, dict]:
    value = _os_json(relative_path)
    if not isinstance(value, list):
        raise HTTPException(503, f'Research OS registry must be a list: {relative_path}')
    return {str(row[key]): row for row in value if isinstance(row, dict) and row.get(key)}


class ResearchWorkUnitIn(BaseModel):
    schema_version: Literal['distributed_work_unit.v1']
    tm_id: str = Field(min_length=1, max_length=80)
    contract_id: str = Field(min_length=1, max_length=100)
    contract_sha256: str = Field(pattern='^[a-f0-9]{64}$')
    puzzle_id: str = Field(min_length=1, max_length=80)
    case_family_id: str = Field(min_length=1, max_length=160)
    split_id: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=160)
    model_snapshot_id: str = Field(min_length=1, max_length=240)
    tokenizer_sha256: str = Field(pattern='^[a-f0-9]{64}$')
    precision: str = Field(min_length=1, max_length=40)
    capture_tier: str = Field(pattern='^tier_[a-d]$')
    seed: int = Field(ge=0)
    required_artifacts: List[str] = Field(min_length=1)
    resource_request: dict = Field(default_factory=dict)
    work_unit_id: str = Field(default='', max_length=120)
    language_graph_target_id: Optional[str] = Field(default=None, pattern='^LG-[A-Z0-9-]+$')


def _validate_work_unit(payload: ResearchWorkUnitIn, template: sqlite3.Row) -> dict:
    if (not payload.required_artifacts
            or any(not isinstance(path, str) or not path.strip() for path in payload.required_artifacts)
            or len(payload.required_artifacts) != len(set(payload.required_artifacts))):
        raise HTTPException(422, 'required_artifacts must contain unique, non-empty artifact paths')
    contract_index = _rows_by_id('registry/contracts.json', 'id').get(payload.contract_id)
    if not contract_index:
        raise HTTPException(422, f"unknown contract_id: {payload.contract_id}")
    if contract_index.get('status') not in {'ready', 'running'} or contract_index.get('run_ready') is not True:
        raise HTTPException(409, 'Research OS contract is not currently runnable; only ready/running and run_ready contracts may be dispatched')
    if contract_index.get('contract_sha256') != payload.contract_sha256:
        raise HTTPException(422, 'contract_sha256 differs from canonical contracts.json')

    contract_path = (_RESEARCH_OS / str(contract_index.get('path', ''))).resolve()
    if not contract_path.is_relative_to(_RESEARCH_OS.resolve()) or not contract_path.is_file():
        raise HTTPException(503, 'canonical contract file is missing or outside Research OS')
    contract_bytes = contract_path.read_bytes()
    actual_sha = hashlib.sha256(contract_bytes).hexdigest()
    if actual_sha != payload.contract_sha256:
        raise HTTPException(409, 'canonical contract file hash does not match frozen registry hash')
    try:
        contract = json.loads(contract_bytes.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(503, 'canonical contract file cannot be decoded')

    puzzle = _rows_by_id('registry/puzzles.json', 'id').get(payload.puzzle_id)
    if not puzzle or payload.puzzle_id not in contract.get('puzzle_ids', []):
        raise HTTPException(422, 'puzzle_id must exist and be referenced by the frozen contract')
    if contract.get('campaign_id') != contract_index.get('campaign_id'):
        raise HTTPException(409, 'contract campaign identity differs from contracts registry')
    campaign = _rows_by_id('registry/campaigns.json', 'id').get(contract.get('campaign_id'))
    if not campaign or payload.puzzle_id not in campaign.get('puzzle_ids', []):
        raise HTTPException(422, 'puzzle_id is not part of the contract campaign')
    if payload.model_id not in contract.get('scope', {}).get('models', []):
        raise HTTPException(422, 'model_id is outside the frozen contract scope')
    if payload.precision not in contract.get('scope', {}).get('precision', []):
        raise HTTPException(422, 'precision is outside the frozen contract scope')
    gpu_hours = payload.resource_request.get('gpu_hours')
    budget_hours = contract.get('budget', {}).get('max_gpu_hours')
    if (isinstance(gpu_hours, bool) or not isinstance(gpu_hours, (int, float))
            or not math.isfinite(gpu_hours) or gpu_hours <= 0):
        raise HTTPException(422, 'resource_request.gpu_hours must be a finite positive number')
    if not isinstance(budget_hours, (int, float)) or gpu_hours > budget_hours:
        raise HTTPException(422, 'Work Unit GPU hours exceed the frozen contract budget')

    try:
        adapter = json.loads(template['research_adapter_json'] or '{}')
    except json.JSONDecodeError:
        adapter = {}
    if adapter.get('schema_version') != 'research_adapter.v1':
        raise HTTPException(409, f"template {payload.tm_id} has no frozen research_adapter.v1 binding")
    if adapter.get('contract_id') != payload.contract_id or adapter.get('contract_sha256') != actual_sha:
        raise HTTPException(409, 'template adapter is not bound to this exact canonical contract hash')
    if payload.language_graph_target_id:
        target_catalog = _os_json('registry/language_graph_targets.json')
        targets = {row.get('id'): row for row in target_catalog.get('targets', []) if isinstance(row, dict)}
        if payload.language_graph_target_id not in targets:
            raise HTTPException(422, f'unknown language_graph_target_id: {payload.language_graph_target_id}')
        if payload.language_graph_target_id not in adapter.get('language_graph_target_ids', []):
            raise HTTPException(422, 'language graph target is not authorized by the frozen template adapter (language_graph_target_ids)')

    allowlists = (
        ('puzzle_ids', payload.puzzle_id, 'puzzle'),
        ('case_family_ids', payload.case_family_id, 'case family'),
        ('split_ids', payload.split_id, 'split'),
        ('model_ids', payload.model_id, 'model'),
        ('model_snapshot_ids', payload.model_snapshot_id, 'model snapshot'),
        ('tokenizer_sha256s', payload.tokenizer_sha256, 'tokenizer hash'),
        ('precisions', payload.precision, 'precision'),
        ('capture_tiers', payload.capture_tier, 'capture tier'),
    )
    for field, value, label in allowlists:
        if value not in adapter.get(field, []):
            raise HTTPException(422, f'{label} is not authorized by the frozen template adapter ({field})')
    adapter_artifacts = set(adapter.get('required_artifacts', []))
    if not set(payload.required_artifacts).issubset(adapter_artifacts):
        raise HTTPException(422, 'required_artifacts exceed the frozen template adapter output contract')

    metrics = json.dumps(contract.get('metrics', []), ensure_ascii=False,
                         sort_keys=True, separators=(',', ':')).encode('utf-8')
    ref = payload.model_dump(exclude={'work_unit_id'}, exclude_none=True)
    stable_ref = dict(ref)
    digest = hashlib.sha256(json.dumps(stable_ref, ensure_ascii=False, sort_keys=True,
                                       separators=(',', ':')).encode('utf-8')).hexdigest()
    ref.update({
        'schema_version': 'distributed_work_unit.v1',
        'campaign_id': contract['campaign_id'],
        'work_package_id': contract['work_package_id'],
        'contract_id': payload.contract_id,
        'contract_sha256': actual_sha,
        'metric_version': 'sha256:' + hashlib.sha256(metrics).hexdigest(),
        'template_version': int(template['version']),
        'template_design_sha': template['design_sha'],
        'contract_gpu_hour_budget': budget_hours,
        'created_at': time.time(),
        'evidence_state': 'not_created',
    })
    work_unit_id = payload.work_unit_id or 'WU-' + digest[:20]
    ref['work_unit_id'] = work_unit_id
    return ref


# ---------- S3 节点与任务 ----------

class NodeReg(BaseModel):
    name: str
    gpu: str = ''
    model: str = ''
    dtype: str = ''
    trust: str = 'unverified'
    bind_code: str = ''          # 账号绑定码（客户端「参与测试」生成）：节点归属该账号


class ClaimReq(BaseModel):
    model_id: str = ''
    model_rev: str = ''
    tokenizer_sha256: str = ''
    gpu: str = ''
    dtype: str = ''
    seed: int = 0
    tm_id: str = ''  # 定向领取（可缺省=负载均衡轮询）


@router.post('/nodes/register')
def node_register(b: NodeReg, response: Response) -> dict:
    node_id = 'NODE-' + uuid.uuid4().hex[:4].upper()
    token = uuid.uuid4().hex + uuid.uuid4().hex[:8]
    db = _db()
    with _db_lock:
        owner = _accounts.consume_bind_code(db, b.bind_code, node_id) if b.bind_code else None
        if b.bind_code and owner is None:
            raise HTTPException(400, '绑定码无效、已过期或已被使用：在客户端「参与测试」重新生成')
        db.execute(
            'INSERT INTO nodes(node_id,token_hash,name,gpu,model,dtype,trust,created_at,last_seen,owner_user_id) '
            'VALUES(?,?,?,?,?,?,?,?,?,?)',
            (node_id, hashlib.sha256(token.encode()).hexdigest(), b.name,
             b.gpu, b.model, b.dtype, b.trust, time.time(), time.time(), owner))
        db.commit()
    response.status_code = 201
    return {'node_id': node_id, 'node_token': token, 'owner_bound': bool(owner),
            'note': 'token 仅本次返回，请保存在节点本地 credentials 文件'}


@router.get('/nodes/me')
def node_me(x_node_token: str = Header(default='')) -> dict:
    n = _auth_node(x_node_token)
    tasks = _db().execute(
        'SELECT task_id,tm_id,status,claimed_at,done_at,result_sha FROM tasks '
        'WHERE node_id=? ORDER BY claimed_at DESC LIMIT 20', (n['node_id'],)).fetchall()
    return {'node': dict(n), 'recent_tasks': [dict(t) for t in tasks]}


@router.get('/research/work-units')
def research_work_units(
        language_graph_target_id: str = Query(default='', max_length=80),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=500)) -> dict:
    """Read-only execution projection; canonical Run/Evidence remain in Research OS."""
    db = _db()
    contract_rows = _rows_by_id('registry/contracts.json', 'id')
    puzzle_rows = _rows_by_id('registry/puzzles.json', 'id')
    runnable = []
    invalid_runnable = []
    for row in contract_rows.values():
        if row.get('status') not in {'ready', 'running'} or row.get('run_ready') is not True:
            continue
        item = {'contract_id': row['id'], 'contract_sha256': row.get('contract_sha256', ''),
                'campaign_id': row.get('campaign_id', ''), 'work_package_id': row.get('work_package_id', ''),
                'status': row.get('status', ''), 'puzzle_ids': []}
        try:
            path = (_RESEARCH_OS / str(row.get('path', ''))).resolve()
            if not path.is_relative_to(_RESEARCH_OS.resolve()) or not path.is_file():
                raise ValueError('contract source is missing or outside Research OS')
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != item['contract_sha256']:
                raise ValueError('contract source hash differs from canonical Registry')
            contract = json.loads(content.decode('utf-8'))
            if contract.get('campaign_id') != item['campaign_id']:
                raise ValueError('contract campaign identity differs from canonical Registry')
            item['title'] = contract.get('title', row['id'])
            item['puzzle_ids'] = contract.get('puzzle_ids', [])
            runnable.append(item)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            invalid_runnable.append(row['id'])
    target_registry = _os_json('registry/language_graph_targets.json')
    registered_target_ids = {row.get('id') for row in target_registry.get('targets', []) if isinstance(row, dict)}
    if language_graph_target_id and language_graph_target_id not in registered_target_ids:
        raise HTTPException(422, f'unknown language_graph_target_id: {language_graph_target_id}')
    target_rows = db.execute(
        'SELECT task_id,status,result_sha,research_ref_json FROM tasks WHERE work_unit_id IS NOT NULL ORDER BY COALESCE(done_at,claimed_at) DESC,task_id DESC').fetchall()
    audit_map = _pipeline.audit_status_map(db) if _pipeline else {}
    target_progress = {
        'by_target': {}, 'total_work_units': len(target_rows), 'target_bound_work_units': 0,
        'unmapped_work_units': 0, 'unrecognized_target_work_units': 0,
        'result_uploaded_total': 0, 'awaiting_audit_total': 0,
        'audit_tracking_available': bool(_pipeline),
    }
    all_work_unit_ids = []
    target_work_unit_ids = {}
    for row in target_rows:
        all_work_unit_ids.append(row['task_id'])
        try:
            ref = json.loads(row['research_ref_json'] or '{}')
        except json.JSONDecodeError:
            ref = {}
        if not isinstance(ref, dict):
            ref = {}
        target_id = ref.get('language_graph_target_id') or ref.get('graph_target_id')
        if not target_id:
            target_progress['unmapped_work_units'] += 1
            continue
        if target_id not in registered_target_ids:
            target_progress['unrecognized_target_work_units'] += 1
            continue
        bucket = target_progress['by_target'].setdefault(target_id, {
            'work_units': 0, 'by_status': {}, 'result_uploaded': 0,
            'awaiting_audit': 0, 'audit_passed': None, 'evidence_levels': [],
        })
        bucket['work_units'] += 1
        target_work_unit_ids.setdefault(target_id, []).append(row['task_id'])
        bucket['by_status'][row['status']] = bucket['by_status'].get(row['status'], 0) + 1
        target_progress['target_bound_work_units'] += 1
        if row['result_sha']:
            bucket['result_uploaded'] += 1
            target_progress['result_uploaded_total'] += 1
            audit_st = (audit_map.get(row['result_sha']) or {}).get('status')
            if audit_st == 'passed':
                bucket['audit_passed'] = (bucket['audit_passed'] or 0) + 1
            elif not audit_st:
                bucket['awaiting_audit'] += 1
                target_progress['awaiting_audit_total'] += 1
    filtered_ids = target_work_unit_ids.get(language_graph_target_id, []) if language_graph_target_id else all_work_unit_ids
    page_total = len(filtered_ids)
    page_ids = filtered_ids[offset:offset + limit]
    if page_ids:
        placeholders = ','.join('?' for _ in page_ids)
        linked = db.execute(
            f'''SELECT t.task_id,t.tm_id,t.status,t.claimed_at,t.lease_until,t.done_at,t.result_sha,
                       t.note,t.work_unit_id,t.research_ref_json,n.name AS node_name,n.model AS node_model,
                       r.kind,r.model_id,r.model_rev,r.seed,r.size,r.summary_json
                FROM tasks t LEFT JOIN nodes n ON n.node_id=t.node_id
                LEFT JOIN results r ON r.sha=t.result_sha
                WHERE t.task_id IN ({placeholders})
                ORDER BY COALESCE(t.done_at,t.claimed_at) DESC,t.task_id DESC''', page_ids).fetchall()
    else:
        linked = []
    items = []
    for row in linked:
        ref = json.loads(row['research_ref_json'] or '{}')
        summary = {}
        if row['summary_json']:
            try:
                summary = json.loads(row['summary_json'])
            except json.JSONDecodeError:
                summary = {}
        manifest = summary.get('_manifest', [])
        ref.update({
            'task_id': row['task_id'], 'task_status': row['status'],
            'node_name': row['node_name'] or '', 'node_model': row['node_model'] or '',
            'claimed_at': row['claimed_at'], 'lease_until': row['lease_until'],
            'done_at': row['done_at'], 'result_sha': row['result_sha'] or '',
            'result_kind': row['kind'] or '', 'result_model_id': row['model_id'] or '',
            'result_model_rev': row['model_rev'] or '', 'result_seed': row['seed'],
            'result_size': row['size'],
            'artifacts': [{'path': x.get('path', ''), 'sha256': x.get('sha256', ''),
                           'size': x.get('size', 0)} for x in manifest if isinstance(x, dict)],
            # Upload/worker completion is an execution receipt, not audited evidence.
            # audit_status 只表示执行产物审计；Evidence 等级仍由 Research OS 登记。
            'audit_status': (audit_map.get(row['result_sha']) or {}).get('status') or '',
            'evidence_state': ('audit_' + (audit_map.get(row['result_sha']) or {}).get('status', ''))
                              if row['result_sha'] and audit_map.get(row['result_sha'])
                              else ('awaiting_audit' if row['result_sha'] else 'not_created'),
        })
        items.append(ref)
    status_counts = {row['status']: row['n'] for row in db.execute(
        'SELECT status,COUNT(*) AS n FROM tasks WHERE work_unit_id IS NOT NULL GROUP BY status')}
    total_tasks = db.execute('SELECT COUNT(*) AS n FROM tasks').fetchone()['n']
    total_linked = db.execute(
        'SELECT COUNT(*) AS n FROM tasks WHERE work_unit_id IS NOT NULL').fetchone()['n']
    return {
        'schema_version': 'research_distributed_view.v1',
        'server_time': time.time(),
        'sources': {'research_os': 'registry/contracts.json + registry/puzzles.json',
                    'execution': 'distributed SQLite tasks/results'},
        'counts': {
            'puzzles': len(puzzle_rows), 'contracts': len(contract_rows),
            'runnable_contracts': len(runnable), 'work_units': total_linked,
            'invalid_runnable_contracts': len(invalid_runnable),
            'unlinked_platform_tasks': max(0, total_tasks - total_linked),
            'linked_results': sum(1 for row in target_rows if row['result_sha']),
            'awaiting_audit': sum(1 for row in target_rows if row['result_sha'] and row['result_sha'] not in audit_map),
            'by_status': status_counts,
        },
        'runnable_contracts': runnable,
        'invalid_runnable_contract_ids': invalid_runnable,
        'puzzles': [{'puzzle_id': p['id'], 'name': p.get('name', p['id']),
                     'status': p.get('status', ''), 'blocker': p.get('blocker', ''),
                     'next_test_id': p.get('next_test_id', '')}
                    for p in puzzle_rows.values()],
        'work_units': items,
        'work_units_page': {
            'language_graph_target_id': language_graph_target_id or None,
            'offset': offset, 'limit': limit, 'total': page_total,
            'has_more': offset + len(page_ids) < page_total,
        },
        'language_graph_target_progress': target_progress,
        'policy': '节点完成与结果上传不会自动创建 Run、Artifact、Evidence 或提升图谱关系。',
    }


@router.post('/research/work-units')
def create_research_work_unit(b: ResearchWorkUnitIn,
                              x_admin_token: str = Header(default='')) -> dict:
    """Queue one idempotent, contract-bound slice; UI never receives the admin secret."""
    _auth_admin(x_admin_token)
    db = _db()
    template = db.execute('SELECT * FROM templates WHERE tm_id=? AND status=?',
                          (b.tm_id, 'open')).fetchone()
    if not template:
        raise HTTPException(404, f'open template not found: {b.tm_id}')
    ref = _validate_work_unit(b, template)
    with _db_lock:
        existing = db.execute('SELECT task_id,research_ref_json,status FROM tasks WHERE work_unit_id=?',
                              (ref['work_unit_id'],)).fetchone()
        if existing:
            previous = json.loads(existing['research_ref_json'] or '{}')
            comparable = dict(previous); comparable.pop('created_at', None)
            proposed = dict(ref); proposed.pop('created_at', None)
            if comparable != proposed:
                raise HTTPException(409, 'work_unit_id already exists with a different frozen payload')
            return {'task_id': existing['task_id'], 'work_unit_id': ref['work_unit_id'],
                    'status': existing['status'], 'idempotent_replay': True}
        rows = db.execute('SELECT research_ref_json FROM tasks WHERE work_unit_id IS NOT NULL').fetchall()
        allocated = 0.0
        for row in rows:
            prior = json.loads(row['research_ref_json'] or '{}')
            if prior.get('contract_id') == ref['contract_id']:
                allocated += float(prior.get('resource_request', {}).get('gpu_hours', 0) or 0)
        requested = float(ref['resource_request']['gpu_hours'])
        if allocated + requested > float(ref['contract_gpu_hour_budget']) + 1e-9:
            raise HTTPException(409, 'queued and completed Work Units would exceed the frozen contract GPU-hour budget')
        task_id = 'T-' + uuid.uuid4().hex[:10]
        db.execute(
            '''INSERT INTO tasks(task_id,tm_id,node_id,status,claimed_at,lease_until,
               work_unit_id,research_ref_json) VALUES(?,?,NULL,'queued',0,0,?,?)''',
            (task_id, b.tm_id, ref['work_unit_id'],
             json.dumps(ref, ensure_ascii=False, sort_keys=True)))
        db.commit()
    return {'task_id': task_id, 'work_unit_id': ref['work_unit_id'],
            'status': 'queued', 'contract_id': ref['contract_id'],
            'contract_sha256': ref['contract_sha256'], 'template_design_sha': ref['template_design_sha'],
            'note': 'queued; worker result will remain pending independent audit'}


# ---------- S8 调度队列只读视图（design/distributed_flow_plan_v1.md P2） ----------
# 公众可看队列全貌（谁在领什么、租约还剩多久、完成/失败计数）；
# 不含 token / 凭据；claim 本身仍是终端私有动作（需 X-Node-Token）。
@router.get('/tasks')
def tasks_public() -> dict:
    db = _db()
    now = time.time()
    with _db_lock:
        db.execute("UPDATE tasks SET status='expired' WHERE status='claimed' AND lease_until<?", (now,))
        db.commit()
    queued = db.execute(
        '''SELECT k.task_id,k.tm_id,k.work_unit_id,k.research_ref_json,
                  t.name AS template_name,t.version AS template_version
           FROM tasks k LEFT JOIN templates t ON t.tm_id=k.tm_id
           WHERE k.status='queued' ORDER BY k.rowid LIMIT 50''').fetchall()
    active = db.execute(
        '''SELECT k.task_id,k.tm_id,k.work_unit_id,k.research_ref_json,k.claimed_at,k.lease_until,
                  n.name AS node_name,n.gpu,n.model,n.dtype
           FROM tasks k LEFT JOIN nodes n ON n.node_id=k.node_id
           WHERE k.status='claimed' ORDER BY k.claimed_at DESC LIMIT 50''').fetchall()
    recent = db.execute(
        '''SELECT k.task_id,k.tm_id,k.work_unit_id,k.research_ref_json,k.status,k.claimed_at,k.done_at,k.note,k.result_sha,
                  n.name AS node_name,n.model
           FROM tasks k LEFT JOIN nodes n ON n.node_id=k.node_id
           WHERE k.status IN ('done','failed') ORDER BY COALESCE(k.done_at,k.claimed_at) DESC LIMIT 30''').fetchall()
    stats = {r['status']: r['n'] for r in db.execute(
        'SELECT status,COUNT(*) AS n FROM tasks GROUP BY status').fetchall()}
    by_tm = [dict(r) for r in db.execute(
        '''SELECT tm_id,
                  SUM(CASE WHEN status='claimed' THEN 1 ELSE 0 END) AS active,
                  SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
                  SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                  SUM(CASE WHEN status='expired' THEN 1 ELSE 0 END) AS expired
           FROM tasks GROUP BY tm_id ORDER BY tm_id''').fetchall()]
    def expose(row):
        value = dict(row)
        raw_ref = value.pop('research_ref_json', '{}')
        try:
            value['research_ref'] = json.loads(raw_ref or '{}')
        except json.JSONDecodeError:
            value['research_ref'] = {}
        return value
    return {'server_time': now, 'lease_hours': round(LEASE_SECONDS / 3600.0, 2),
            'stats': stats, 'by_template': by_tm,
            'queued': [expose(r) for r in queued],
            'active': [expose(r) for r in active],
            'recent': [expose(r) for r in recent]}


@router.post('/tasks/claim')
def task_claim(b: ClaimReq, x_node_token: str = Header(default='')) -> dict:
    n = _auth_node(x_node_token)
    db = _db()
    now = time.time()
    claimed = None
    with _db_lock:
        db.execute("UPDATE tasks SET status='expired' WHERE status='claimed' AND lease_until<?", (now,))
        active = {r['tm_id']: r['n'] for r in db.execute(
            "SELECT tm_id,COUNT(*) AS n FROM tasks WHERE status='claimed' GROUP BY tm_id").fetchall()}
        queued_sql = "SELECT * FROM tasks WHERE status='queued'"
        queued_args: list = []
        if b.tm_id:
            queued_sql += ' AND tm_id=?'; queued_args.append(b.tm_id)
        queued_sql += ' ORDER BY rowid LIMIT 100'
        for pending in db.execute(queued_sql, queued_args).fetchall():
            ref = json.loads(pending['research_ref_json'] or '{}')
            if (b.model_id != ref.get('model_id') or b.dtype != ref.get('precision')
                    or b.model_rev != ref.get('model_snapshot_id')
                    or b.tokenizer_sha256 != ref.get('tokenizer_sha256')
                    or b.seed != ref.get('seed')):
                continue
            tpl_candidate = db.execute('SELECT * FROM templates WHERE tm_id=? AND status=?',
                                       (pending['tm_id'], 'open')).fetchone()
            if not tpl_candidate:
                continue
            if (tpl_candidate['design_sha'] != ref.get('template_design_sha')
                    or int(tpl_candidate['version']) != int(ref.get('template_version', -1))):
                continue
            db.execute('UPDATE tasks SET status=?,node_id=?,claimed_at=?,lease_until=? WHERE task_id=? AND status=?',
                       ('claimed', n['node_id'], now, now + LEASE_SECONDS, pending['task_id'], 'queued'))
            if not pending['shard']:
                shard_ref = str((ref.get('resource_request') or {}).get('shard') or '') or _next_shard(db, tpl_candidate)
                db.execute('UPDATE tasks SET shard=? WHERE task_id=?', (shard_ref, pending['task_id']))
            claimed = (pending['task_id'], tpl_candidate, ref)
            break
        if claimed:
            db.execute('UPDATE nodes SET last_seen=?, model=? WHERE node_id=?',
                       (now, b.model_id or n['model'], n['node_id']))
            db.commit()
        else:
            claimed = None
        if claimed:
            task_id, tpl, research_ref = claimed
        else:
            if b.tm_id:
                tpl = db.execute(
                    "SELECT * FROM templates WHERE status='open' AND tm_id=? "
                    "AND COALESCE(research_adapter_json,'{}')='{}'", (b.tm_id,)).fetchone()
                if not tpl:
                    adapter_tpl = db.execute(
                        "SELECT research_adapter_json FROM templates WHERE status='open' AND tm_id=?",
                        (b.tm_id,)).fetchone()
                    if adapter_tpl and json.loads(adapter_tpl['research_adapter_json'] or '{}'):
                        raise HTTPException(409, 'template is bound to Research Work Units and cannot issue an unlinked legacy task')
                    raise HTTPException(404, f'template {b.tm_id} not found or not open')
            else:
                cand = db.execute(
                    "SELECT * FROM templates WHERE status='open' "
                    "AND COALESCE(research_adapter_json,'{}')='{}' ORDER BY tm_id").fetchall()
                if not cand:
                    raise HTTPException(204, 'no open template')
                cand = sorted(cand, key=lambda r: (active.get(r['tm_id'], 0), r['tm_id']))
                tpl = cand[0]
            task_id = 'T-' + uuid.uuid4().hex[:10]
            research_ref = {}
            shard = _next_shard(db, tpl, b.model_id, b.model_rev)
            db.execute(
                'INSERT INTO tasks(task_id,tm_id,node_id,status,claimed_at,lease_until,research_ref_json,shard) '
                'VALUES(?,?,?,?,?,?,?,?)',
                (task_id, tpl['tm_id'], n['node_id'], 'claimed', now, now + LEASE_SECONDS, '{}', shard))
            db.execute('UPDATE nodes SET last_seen=?, model=? WHERE node_id=?',
                       (now, b.model_id or n['model'], n['node_id']))
            db.commit()
    shard_row = db.execute('SELECT shard FROM tasks WHERE task_id=?', (task_id,)).fetchone()
    return {'task_id': task_id, 'tm_id': tpl['tm_id'], 'version': tpl['version'],
            'design_sha': tpl['design_sha'], 'name': tpl['name'], 'dim': tpl['dim'],
            'shard': (shard_row['shard'] if shard_row else None) or '',
            'lease_until': now + LEASE_SECONDS, 'work_unit_id': research_ref.get('work_unit_id', ''),
            'research_work_unit': research_ref,
            'files': _bundle_files(tpl['bundle_dir'])}


def _template_shards(tpl) -> int:
    try:
        corpus = json.loads((Path(tpl['bundle_dir']) / 'corpus.json').read_text(encoding='utf-8'))
        return max(1, int(corpus.get('shards', 1) or 1))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 1


def _shards_done_for_model(db: sqlite3.Connection, tpl, model_id: str, model_rev: str) -> set:
    """该模型快照在当前模板版本上已有结果的分片（多个被测模型共用一个模板时，各自补齐自己的分片）。"""
    done = set()
    if not model_id:
        return done
    for r in db.execute('SELECT summary_json FROM results WHERE tm_id=? AND design_sha=? AND model_id=? AND model_rev=?',
                        (tpl['tm_id'], tpl['design_sha'], model_id, model_rev)):
        try:
            done.add(str(json.loads(r['summary_json'] or '{}').get('shard') or ''))
        except json.JSONDecodeError:
            continue
    return done


def _next_shard(db: sqlite3.Connection, tpl, model_id: str = '', model_rev: str = '') -> Optional[str]:
    """v3 模板：优先选该模型尚无结果的分片，其中 done+claimed 次数最少者（并列取编号小者）；legacy 模板返回 None。"""
    shards = _template_shards(tpl)
    if shards <= 1:
        return '0/1' if _pipeline and _pipeline.is_v3(_pipeline.template_corpus(db, tpl['tm_id'])) else None
    used: Dict[str, int] = {}
    for r in db.execute("SELECT shard,COUNT(*) AS n FROM tasks WHERE tm_id=? AND status IN ('claimed','done') "
                        'AND shard IS NOT NULL GROUP BY shard', (tpl['tm_id'],)):
        used[r['shard']] = r['n']
    have = _shards_done_for_model(db, tpl, model_id, model_rev)
    return min((f'{i}/{shards}' for i in range(shards)),
               key=lambda k: (k in have, used.get(k, 0), int(k.split('/')[0])))


@router.put('/tasks/{task_id}/heartbeat')
def task_heartbeat(task_id: str, x_node_token: str = Header(default='')) -> dict:
    n = _auth_node(x_node_token)
    db = _db()
    with _db_lock:
        cur = db.execute('SELECT * FROM tasks WHERE task_id=? AND node_id=?',
                         (task_id, n['node_id'])).fetchone()
        if not cur:
            raise HTTPException(404, 'task not found')
        if cur['status'] != 'claimed':
            raise HTTPException(409, f"task status={cur['status']}")
        db.execute('UPDATE tasks SET lease_until=? WHERE task_id=?',
                   (time.time() + LEASE_SECONDS, task_id))
        db.execute('UPDATE nodes SET last_seen=? WHERE node_id=?', (time.time(), n['node_id']))
        db.commit()
    return {'task_id': task_id, 'lease_until': time.time() + LEASE_SECONDS}


class TaskDone(BaseModel):
    status: str = Field(pattern='^(done|failed)$')
    note: str = ''
    result_sha: str = ''


@router.post('/tasks/{task_id}/complete')
def task_complete(task_id: str, b: TaskDone, x_node_token: str = Header(default='')) -> dict:
    n = _auth_node(x_node_token)
    db = _db()
    with _db_lock:
        cur = db.execute('SELECT * FROM tasks WHERE task_id=? AND node_id=?',
                         (task_id, n['node_id'])).fetchone()
        if not cur:
            raise HTTPException(404, 'task not found')
        if cur['status'] not in {'claimed', 'done', 'failed'}:
            raise HTTPException(409, f"task status={cur['status']}")
        if cur['status'] in {'done', 'failed'} and cur['status'] != b.status:
            raise HTTPException(409, f"terminal task status={cur['status']} cannot transition to {b.status}")
        if cur['status'] == 'failed' and b.status == 'failed':
            return {'task_id': task_id, 'status': 'failed', 'idempotent_replay': True}
        if b.status == 'done':
            if not b.result_sha:
                raise HTTPException(422, 'done requires a content-addressed result_sha')
            result = db.execute('SELECT sha FROM results WHERE sha=?', (b.result_sha,)).fetchone()
            if not result:
                raise HTTPException(409, 'result_sha has not been accepted by the upload pipeline')
            if cur['result_sha'] != b.result_sha:
                raise HTTPException(409, 'task already points to a different immutable result')
            if cur['status'] == 'done' and cur['result_sha'] == b.result_sha:
                return {'task_id': task_id, 'status': 'done', 'idempotent_replay': True}
        db.execute('UPDATE tasks SET status=?, done_at=?, note=?, result_sha=? WHERE task_id=?',
                   (b.status, time.time(), b.note, b.result_sha, task_id))
        db.execute('UPDATE nodes SET last_seen=? WHERE node_id=?', (time.time(), n['node_id']))
        db.commit()
    return {'task_id': task_id, 'status': b.status}


# ---------- S2 模板（admin） ----------

class TemplateReg(BaseModel):
    tm_id: str
    version: int = 1
    name: str
    dim: str
    description: str = ''
    metric_pin: str = ''
    model_pin: str = ''
    corpus: dict
    runner: str
    research_adapter: dict = Field(default_factory=dict)


def _validate_research_adapter_for_registration(adapter: dict) -> None:
    if not adapter:
        return
    required = {
        'schema_version', 'input_contract', 'contract_id', 'contract_sha256',
        'puzzle_ids', 'case_family_ids', 'split_ids', 'model_ids',
        'model_snapshot_ids', 'tokenizer_sha256s', 'precisions',
        'capture_tiers', 'required_artifacts',
    }
    optional = {'language_graph_target_ids'}
    if not isinstance(adapter, dict) or required - set(adapter):
        raise HTTPException(422, 'research_adapter is missing required research_adapter.v1 fields')
    if set(adapter) - required - optional:
        raise HTTPException(422, f'research_adapter has unsupported fields: {sorted(set(adapter) - required - optional)}')
    if adapter.get('schema_version') != 'research_adapter.v1' or adapter.get('input_contract') != 'AI2050_RESEARCH_WORK_UNIT_PATH':
        raise HTTPException(422, 'research_adapter must declare v1 and consume AI2050_RESEARCH_WORK_UNIT_PATH')
    if not re_full_sha256(adapter.get('contract_sha256')):
        raise HTTPException(422, 'research_adapter.contract_sha256 must be lowercase SHA256')
    for key in required - {'schema_version', 'input_contract', 'contract_id', 'contract_sha256'}:
        values = adapter.get(key)
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
            raise HTTPException(422, f'research_adapter.{key} must be a non-empty string list')
        if len(values) != len(set(values)):
            raise HTTPException(422, f'research_adapter.{key} must not contain duplicates')
    if any(not re_full_sha256(value) for value in adapter['tokenizer_sha256s']):
        raise HTTPException(422, 'research_adapter.tokenizer_sha256s must contain lowercase SHA256 values')
    if any(value not in {'tier_a', 'tier_b', 'tier_c', 'tier_d'} for value in adapter['capture_tiers']):
        raise HTTPException(422, 'research_adapter.capture_tiers contains an unknown capture tier')
    if 'language_graph_target_ids' in adapter:
        target_ids = adapter['language_graph_target_ids']
        if (not isinstance(target_ids, list) or not target_ids
                or any(not isinstance(value, str) or not value.startswith('LG-') for value in target_ids)
                or len(target_ids) != len(set(target_ids))):
            raise HTTPException(422, 'research_adapter.language_graph_target_ids must be a unique non-empty LG-* string list')
        catalog = _os_json('registry/language_graph_targets.json')
        registered_ids = {row.get('id') for row in catalog.get('targets', []) if isinstance(row, dict)}
        if not set(target_ids).issubset(registered_ids):
            raise HTTPException(422, 'research_adapter.language_graph_target_ids contains an unknown target')
    contract_row = _rows_by_id('registry/contracts.json', 'id').get(adapter['contract_id'])
    if (not contract_row or contract_row.get('status') not in {'ready', 'running'}
            or contract_row.get('run_ready') is not True
            or contract_row.get('contract_sha256') != adapter['contract_sha256']):
        raise HTTPException(409, 'research_adapter must bind to an exact ready/run_ready Research OS contract')
    path = (_RESEARCH_OS / str(contract_row.get('path', ''))).resolve()
    if not path.is_relative_to(_RESEARCH_OS.resolve()) or not path.is_file():
        raise HTTPException(503, 'adapter contract source is missing or outside Research OS')
    if hashlib.sha256(path.read_bytes()).hexdigest() != adapter['contract_sha256']:
        raise HTTPException(409, 'adapter contract bytes differ from canonical SHA256')
    contract = json.loads(path.read_text(encoding='utf-8'))
    if not set(adapter['puzzle_ids']).issubset(set(contract.get('puzzle_ids', []))):
        raise HTTPException(422, 'adapter puzzle_ids exceed contract puzzle_ids')
    if not set(adapter['model_ids']).issubset(set(contract.get('scope', {}).get('models', []))):
        raise HTTPException(422, 'adapter model_ids exceed contract model scope')
    if not set(adapter['precisions']).issubset(set(contract.get('scope', {}).get('precision', []))):
        raise HTTPException(422, 'adapter precisions exceed contract precision scope')


def re_full_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in '0123456789abcdef' for ch in value)


@router.post('/templates')
def template_register(b: TemplateReg, x_admin_token: str = Header(default='')) -> dict:
    _auth_admin(x_admin_token)
    _validate_research_adapter_for_registration(b.research_adapter)
    design_sha = _register_template(b.model_dump())
    return {'tm_id': b.tm_id, 'version': b.version, 'design_sha': design_sha,
            'research_adapter': bool(b.research_adapter),
            'note': '预注册冻结：corpus/runner/research_adapter 变更必须升 version'}


class TemplateStatusIn(BaseModel):
    status: Literal['open', 'closed']
    note: str = Field(default='', max_length=500)


@router.post('/templates/{tm_id}/status')
def template_status(tm_id: str, b: TemplateStatusIn, x_admin_token: str = Header(default='')) -> dict:
    """关闭 / 重开模板（admin）。关闭后不再派发新任务、不进入谱图；已有结果保留。重新冻结（注册新版本）时自动重开。"""
    _auth_admin(x_admin_token)
    db = _db()
    row = db.execute('SELECT tm_id,version,status FROM templates WHERE tm_id=?', (tm_id,)).fetchone()
    if not row:
        raise HTTPException(404, f'unknown template {tm_id}')
    with _db_lock:
        db.execute('UPDATE templates SET status=? WHERE tm_id=?', (b.status, tm_id))
        db.commit()
    return {'tm_id': tm_id, 'version': row['version'], 'status': b.status, 'previous': row['status'], 'note': b.note}


@router.get('/templates')
def templates_list() -> dict:
    db = _db()
    rows = db.execute('SELECT tm_id,version,name,dim,description,metric_pin,model_pin,'
                      'design_sha,status,research_adapter_json FROM templates ORDER BY tm_id').fetchall()
    counts = {r['tm_id']: r['n'] for r in db.execute(
        'SELECT tm_id,COUNT(*) AS n FROM results GROUP BY tm_id').fetchall()}
    out = []
    for r in rows:
        d = dict(r)
        adapter = json.loads(d.pop('research_adapter_json') or '{}')
        d['research_adapter'] = bool(adapter)
        d['research_contract_id'] = adapter.get('contract_id', '')
        d['results'] = counts.get(r['tm_id'], 0)
        if _pipeline:
            corpus = _pipeline.template_corpus(db, r['tm_id'])
            study = [it for it in corpus.get('items', []) if not it.get('canary')]
            d['protocol'] = 'v3' if _pipeline.is_v3(corpus) else 'legacy'
            d['family_id'] = corpus.get('family_id') or ''
            d['module'] = corpus.get('module') or ''
            d['mode'] = corpus.get('mode') or ''
            d['shards'] = int(corpus.get('shards', 1) or 1)
            d['n_items'] = len(study)
            d['factors'] = corpus.get('factors') or []
            by_split: Dict[str, int] = {}
            for it in study:
                by_split[it.get('split', '-')] = by_split.get(it.get('split', '-'), 0) + 1
            d['by_split'] = by_split
        out.append(d)
    return {'templates': out}


@router.get('/templates/{tm_id}')
def template_detail(tm_id: str) -> dict:
    """D1-1（ui_decoupled_plan_v1）：单模板 contract + corpus 元数据。
    只输出协议键（analysis/factors/layers/items 计数），从 bundle corpus.json 提取——
    界面据此通用渲染任意模板，不出现任何 TM-xx 专属字段。"""
    db = _db()
    tpl = db.execute('SELECT tm_id,version,name,dim,description,metric_pin,model_pin,'
                     'design_sha,status,bundle_dir,research_adapter_json FROM templates WHERE tm_id=?', (tm_id,)).fetchone()
    if not tpl:
        raise HTTPException(404, f'template {tm_id} not found')
    d = {k: tpl[k] for k in ('tm_id', 'version', 'name', 'dim', 'description',
                             'metric_pin', 'model_pin', 'design_sha', 'status')}
    d['results'] = db.execute('SELECT COUNT(*) AS n FROM results WHERE tm_id=?',
                              (tm_id,)).fetchone()['n']
    adapter = json.loads(tpl['research_adapter_json'] or '{}')
    d['research_adapter'] = adapter if adapter else None
    meta: Dict[str, Any] = {'analysis': 'oneway', 'factors': [], 'factor_levels': {},
                            'layers': [], 'items': 0, 'fingerprint': False, 'conds': []}
    try:
        corpus = json.loads(_bundle_files(tpl['bundle_dir'])['corpus.json'])
        items = corpus.get('items') or []
        meta['analysis'] = corpus.get('analysis', 'oneway')
        meta['layers'] = list(corpus.get('layers') or [])
        meta['items'] = len(items)
        meta['factors'] = list(corpus.get('factors') or
                               (sorted(items[0]['cond']) if items and isinstance(items[0].get('cond'), dict) else []))
        # 因子水平与条件组合（展示用，协议键驱动：任意因子名通用）
        levels: Dict[str, list] = {}
        conds = set()
        for it in items:
            cond = it.get('cond') or {}
            if not isinstance(cond, dict):
                continue
            for k, v in cond.items():
                levels.setdefault(k, [])
                if v not in levels[k]:
                    levels[k].append(v)
            conds.add(json.dumps(cond, ensure_ascii=False, sort_keys=True))
        meta['factor_levels'] = {k: sorted(v) for k, v in levels.items()}
        meta['conds'] = sorted(conds)[:12]
        meta['fingerprint'] = bool(corpus.get('fingerprint')) or any(
            isinstance(it, dict) and 'fingerprint' in it for it in items)
    except Exception as e:  # bundle 缺损时不炸详情，标 degraded
        meta['degraded'] = str(e)[:200]
    d['meta'] = meta
    return d


@router.get('/templates/{tm_id}/bundle')
def template_bundle(tm_id: str, x_node_token: str = Header(default='')) -> dict:
    _auth_node(x_node_token)
    tpl = _db().execute('SELECT * FROM templates WHERE tm_id=?', (tm_id,)).fetchone()
    if not tpl:
        raise HTTPException(404, 'template not found')
    return {'tm_id': tm_id, 'version': tpl['version'], 'design_sha': tpl['design_sha'],
            'files': _bundle_files(tpl['bundle_dir'])}


# ---------- S4 结果库（内容寻址 + 分块上传） ----------

class UploadInit(BaseModel):
    task_id: str = ''
    tm_id: str
    design_sha: str
    model_id: str = ''
    model_rev: str = ''
    precision: str = ''
    tokenizer_sha256: str = ''
    seed: int = 0
    kind: str = Field(default='smoke', pattern='^(real|smoke|synthetic)$')
    summary: dict = {}
    manifest: List[dict]  # [{path, sha256, size}]


class ChunkIn(BaseModel):
    seq: int
    data_b64: str


@router.post('/results/upload/init')
def upload_init(b: UploadInit, x_node_token: str = Header(default='')) -> dict:
    n = _auth_node(x_node_token)
    db = _db()
    tpl = db.execute('SELECT * FROM templates WHERE tm_id=?', (b.tm_id,)).fetchone()
    if not tpl:
        raise HTTPException(404, f'template {b.tm_id} not found')
    if b.design_sha != tpl['design_sha']:
        raise HTTPException(422, f"design_sha mismatch: template={tpl['design_sha']} got={b.design_sha}"
                                 '（预注册纪律：模板冻结后语料/口径不可变，请升级模板 version）')
    if b.task_id:
        task = db.execute('SELECT * FROM tasks WHERE task_id=? AND node_id=?',
                          (b.task_id, n['node_id'])).fetchone()
        if not task or task['status'] != 'claimed':
            raise HTTPException(409, 'task_id is not an active lease owned by this node')
        if task['tm_id'] != b.tm_id:
            raise HTTPException(422, 'uploaded tm_id does not match the claimed task')
        ref = json.loads(task['research_ref_json'] or '{}')
        if ref:
            if (b.model_id != ref.get('model_id') or b.model_rev != ref.get('model_snapshot_id')
                    or b.precision != ref.get('precision')
                    or b.tokenizer_sha256 != ref.get('tokenizer_sha256')
                    or b.seed != ref.get('seed')):
                raise HTTPException(422, 'uploaded model snapshot, tokenizer, precision or seed differs from frozen Work Unit')
            provided = {str(item.get('path', '')) for item in b.manifest if isinstance(item, dict)}
            missing = sorted(set(ref.get('required_artifacts', [])) - provided)
            if missing:
                raise HTTPException(422, f'required Work Unit artifacts missing from upload manifest: {missing}')
    total = sum(int(f.get('size', 0)) for f in b.manifest)
    if total > MAX_TOTAL_BYTES:
        raise HTTPException(413, f'total {total} > {MAX_TOTAL_BYTES}')
    for f in b.manifest:
        if int(f.get('size', 0)) > MAX_FILE_BYTES:
            raise HTTPException(413, f"file {f.get('path')} too large（敏感原始数据不上传，见方案 S4）")
    _upload_gc()
    upload_id = 'U-' + uuid.uuid4().hex[:10]
    chunks_expected = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    d = _upload_dir(upload_id)
    d.mkdir(parents=True)
    (d / 'meta.json').write_text(json.dumps({'meta': b.model_dump(), 'node_id': n['node_id'],
                                             'chunks_expected': chunks_expected, 'created_at': time.time()},
                                            ensure_ascii=False), encoding='utf-8')
    return {'upload_id': upload_id, 'chunk_size': CHUNK_SIZE,
            'chunks_expected': chunks_expected}


# 上传中的分块落盘（<data>/uploads/<upload_id>/）：中心重启后节点可以接着传；同一 seq 重传会覆盖，
# 节点因网络中断重试分块是安全的。finish 成功后只留 done.json，重复的 finish 返回同一结果。
_UPLOAD_ID_RE = re.compile(r'^U-[0-9a-f]{10}$')


def _upload_dir(upload_id: str) -> Path:
    if not _UPLOAD_ID_RE.match(upload_id):
        raise HTTPException(404, 'upload not found')
    return Path(_DATA_DIR) / 'uploads' / upload_id


def _upload_load(upload_id: str, node_id: str) -> tuple:
    d = _upload_dir(upload_id)
    try:
        up = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise HTTPException(404, 'upload not found（超过 24 小时未完成的上传已清理，请重新上传）')
    if up['node_id'] != node_id:
        raise HTTPException(403, 'upload belongs to another node')
    return d, up


def _upload_gc() -> None:
    root = Path(_DATA_DIR) / 'uploads'
    if not root.is_dir():
        return
    cutoff = time.time() - UPLOAD_TTL
    for d in root.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


@router.post('/results/upload/{upload_id}/chunk')
def upload_chunk(upload_id: str, b: ChunkIn, x_node_token: str = Header(default='')) -> dict:
    n = _auth_node(x_node_token)
    d, up = _upload_load(upload_id, n['node_id'])
    if (d / 'done.json').is_file():
        raise HTTPException(409, 'upload already finished')
    if b.seq < 0 or b.seq > up['chunks_expected']:
        raise HTTPException(422, f"seq {b.seq} out of range（应在 0–{up['chunks_expected']}）")
    tmp = d / f'{b.seq:06d}.tmp'
    tmp.write_bytes(base64.b64decode(b.data_b64))
    tmp.replace(d / f'{b.seq:06d}.part')
    return {'upload_id': upload_id, 'received': len(list(d.glob('*.part')))}


@router.post('/results/upload/{upload_id}/finish')
def upload_finish(upload_id: str, x_node_token: str = Header(default='')) -> dict:
    n = _auth_node(x_node_token)
    with _upload_lock:
        d, up = _upload_load(upload_id, n['node_id'])
        done = d / 'done.json'
        if done.is_file():                     # 节点没收到上一次的响应而重试：返回同一结果，不重复入库
            return {**json.loads(done.read_text(encoding='utf-8')), 'idempotent_replay': True}
        try:
            out = _upload_register(n, up, sorted(d.glob('*.part')))
        except HTTPException as exc:
            if exc.status_code != 409:         # 内容不合格：丢弃这次上传；409（分块不全、租约失效）保留分块
                shutil.rmtree(d, ignore_errors=True)
            raise
        done.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
        for p in d.glob('*.part'):
            p.unlink(missing_ok=True)
        return out


def _upload_register(n, up: dict, parts: List[Path]) -> dict:
    meta = up['meta']
    actual_sequences = [int(p.stem) for p in parts]
    if not actual_sequences or actual_sequences != list(range(max(actual_sequences) + 1)):
        raise HTTPException(409, 'upload chunks are incomplete or contain unexpected sequence numbers')
    task = None
    task_ref = {}
    if meta.get('task_id'):
        task = _db().execute('SELECT * FROM tasks WHERE task_id=? AND node_id=?',
                             (meta['task_id'], n['node_id'])).fetchone()
        if not task or task['status'] != 'claimed':
            raise HTTPException(409, 'task lease is no longer active; uploaded bytes were not registered as a result')
        task_ref = json.loads(task['research_ref_json'] or '{}')
    blob = b''.join(p.read_bytes() for p in parts)
    # 传输格式（与 node_agent 约定）：4B 路径表长(big-endian) + 路径表 JSON + 按 manifest 顺序的文件内容
    if len(blob) < 4:
        raise HTTPException(422, 'empty upload blob')
    table_len = int.from_bytes(blob[:4], 'big')
    try:
        paths = json.loads(blob[4:4 + table_len].decode('utf-8'))
    except Exception:
        raise HTTPException(422, 'malformed upload blob (path table)')
    rest = blob[4 + table_len:]
    manifest = meta['manifest']
    expected_paths = [str(f.get('path', '')) for f in manifest]
    if paths != expected_paths:
        raise HTTPException(422, 'path table order or entries differ from manifest')
    raw_files: Dict[str, bytes] = {}
    offset = 0
    for i, f in enumerate(manifest):
        size = int(f['size'])
        content = rest[offset:offset + size]
        offset += size
        actual = hashlib.sha256(content).hexdigest()
        if actual != f['sha256']:
            raise HTTPException(422, f"sha256 mismatch for {f['path']}: "
                                     f"expected {f['sha256'][:12]} got {actual[:12]}")
        raw_files[f['path']] = content
    if offset != len(rest):
        raise HTTPException(422, 'blob payload length does not match manifest sizes')
    whole = hashlib.sha256(blob).hexdigest()
    obj_dir = _DATA_DIR / 'objects' / whole[:2]
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / whole).write_bytes(blob)
    summary_stored = dict(meta['summary'] or {})
    summary_stored['_manifest'] = meta['manifest']
    summary_stored['_upload_meta'] = {'precision': meta.get('precision', ''),
                                      'tokenizer_sha256': meta.get('tokenizer_sha256', '')}
    if _canary and 'canary.json' in raw_files:
        summary_stored['_canary'] = _evaluate_canary(meta, raw_files['canary.json'], whole)
    if task_ref:
        summary_stored['_research_work_unit'] = {
            'work_unit_id': task_ref.get('work_unit_id'),
            'contract_id': task_ref.get('contract_id'),
            'contract_sha256': task_ref.get('contract_sha256'),
            'puzzle_id': task_ref.get('puzzle_id'),
            'split_id': task_ref.get('split_id'),
            'capture_tier': task_ref.get('capture_tier'),
            'model_snapshot_id': task_ref.get('model_snapshot_id'),
            'tokenizer_sha256': task_ref.get('tokenizer_sha256'),
            'precision': task_ref.get('precision'),
            'seed': task_ref.get('seed'),
        }
    db = _db()
    replica_note = ''
    with _db_lock:
        existing = db.execute('SELECT node_id,model_id,model_rev,summary_json FROM results WHERE sha=?',
                              (whole,)).fetchone()
        if existing:
            # 内容寻址：字节完全相同的重复上传不覆盖原记录（保留首个上传者与金丝雀判定），
            # 另记一条副本——同快照下即「逐字节一致的独立重跑」；模型标识不同则标记可疑。
            prev_meta = (json.loads(existing['summary_json'] or '{}').get('_upload_meta') or {})
            same_model = (existing['model_id'] == meta['model_id'] and existing['model_rev'] == meta['model_rev']
                          and prev_meta.get('precision', '') == meta.get('precision', ''))
            replica_note = 'byte_identical_replicate' if same_model else 'identical_bytes_different_model'
            db.execute('INSERT INTO result_replicas(sha,node_id,task_id,model_id,model_rev,precision,uploaded_at,note) '
                       'VALUES(?,?,?,?,?,?,?,?)',
                       (whole, n['node_id'], meta.get('task_id', ''), meta['model_id'], meta['model_rev'],
                        meta.get('precision', ''), time.time(), replica_note))
        else:
            db.execute(
                'INSERT INTO results(sha,tm_id,version,node_id,model_id,model_rev,seed,'
                'kind,design_sha,size,summary_json,uploaded_at,downloads) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,0)',
                (whole, meta['tm_id'], None, n['node_id'], meta['model_id'], meta['model_rev'],
                 meta['seed'], meta['kind'], meta['design_sha'], len(blob),
                 json.dumps(summary_stored, ensure_ascii=False), time.time()))
        if meta.get('task_id'):
            db.execute("UPDATE tasks SET status='done', done_at=?, result_sha=? WHERE task_id=? "
                       "AND node_id=?",
                       (time.time(), whole, meta['task_id'], n['node_id']))
        db.execute('UPDATE nodes SET last_seen=? WHERE node_id=?', (time.time(), n['node_id']))
        db.commit()
    return {'sha': whole, 'tm_id': meta['tm_id'], 'kind': meta['kind'],
            'files': list(raw_files), 'replica': replica_note or None,
            'note': '已入库；内容寻址可去重' if not replica_note else f'与已有结果字节相同：{replica_note}'}


def _evaluate_canary(meta: dict, content: bytes, sha: str) -> dict:
    """金丝雀判定：同 (model_id, model_rev, precision) 首个 real 结果登记为参考，其后逐个比较。"""
    try:
        cand = json.loads(content.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {'status': 'missing', 'detail': 'canary.json 无法解析'}
    key = _canary.model_key(meta.get('model_id', ''), meta.get('model_rev', ''), meta.get('precision', ''))
    db = _db()
    row = db.execute('SELECT reference_json,source_sha FROM canary_refs WHERE model_key=?', (key,)).fetchone()
    ref = json.loads(row['reference_json']) if row else None
    status, detail = _canary.evaluate(ref, cand, meta.get('precision', ''))
    if status == 'reference' and meta.get('kind') in _pipeline.mergeable_kinds():
        with _db_lock:
            db.execute('INSERT OR IGNORE INTO canary_refs(model_key,reference_json,source_sha,created_at) VALUES(?,?,?,?)',
                       (key, json.dumps(cand, ensure_ascii=False), sha, time.time()))
            db.commit()
    elif status == 'reference':
        status, detail = 'unreferenced', {'status': 'unreferenced', 'detail': '非 real 结果不登记为参考'}
    out = dict(detail)
    out.update({'status': status, 'model_key': key, 'reference_sha': row['source_sha'] if row else sha})
    return out


@router.get('/results')
def results_list(tm_id: str = '', model_id: str = '', kind: str = '', limit: int = 50) -> dict:
    # D1-3（ui_decoupled_plan_v1）：summary 摘要字段平铺（协议键白名单），表格直接渲染
    digest_keys = ('status', 'analysis', 'layers', 'dim', 'eta2_by_factor', 'eta2_by_cond',
                   'per_layer', 'fingerprint', 'topk_jaccard', 'prefix_pull_mean',
                   'n_layer_head', 'cos_absmax_offdiag', 'protocol', 'shard', 'family_id',
                   'eta2_shard_digest', 'n_items')
    q = ('SELECT sha,tm_id,node_id,model_id,model_rev,seed,kind,size,uploaded_at,downloads,'
         'summary_json FROM results WHERE 1=1')
    args: list = []
    if tm_id:
        q += ' AND tm_id=?'; args.append(tm_id)
    if model_id:
        q += ' AND model_id=?'; args.append(model_id)
    if kind:
        q += ' AND kind=?'; args.append(kind)
    q += ' ORDER BY uploaded_at DESC LIMIT ?'; args.append(max(1, min(limit, 200)))
    rows = _db().execute(q, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            s = json.loads(d.pop('summary_json') or '{}')
        except Exception:
            s = {}
        d['summary_digest'] = {k: s[k] for k in digest_keys if k in s}
        d['summary_digest']['canary_status'] = (s.get('_canary') or {}).get('status')
        d['precision'] = (s.get('_upload_meta') or {}).get('precision') or s.get('precision', '')
        out.append(d)
    return {'results': out, 'total': len(out)}


@router.get('/results/{sha}')
def result_download(sha: str) -> dict:
    return result_detail(sha)


def result_detail(sha: str, count: bool = True) -> dict:
    """单个结果的摘要、文件（base64）与 cos 矩阵；count=False 时不计下载次数（离线数据包生成用）。"""
    db = _db()
    row = db.execute('SELECT * FROM results WHERE sha=?', (sha,)).fetchone()
    if not row:
        raise HTTPException(404, 'result not found')
    p = _DATA_DIR / 'objects' / sha[:2] / sha
    if not p.is_file():
        raise HTTPException(410, 'object missing')
    blob = p.read_bytes()
    if count:
        with _db_lock:
            db.execute('UPDATE results SET downloads=downloads+1 WHERE sha=?', (sha,))
            db.commit()
    cursor = 4 + int.from_bytes(blob[:4], 'big')
    paths = json.loads(blob[4:cursor].decode('utf-8'))
    rest = blob[cursor:]
    files: Dict[str, str] = {}
    meta = json.loads(row['summary_json'] or '{}')
    manifest = meta.get('_manifest') or []
    if len(paths) != len(manifest):
        raise HTTPException(410, 'stored object incompatible with manifest')
    off = 0
    for i, f in enumerate(manifest):
        size = int(f['size'])
        files[f['path']] = base64.b64encode(rest[off:off + size]).decode('ascii')
        off += size
    d = dict(row)
    d['summary'] = meta
    d['files'] = files
    # CellCosGrid 数据源（ui_decoupled_plan_v1 §2-4）：尽力从 means.npz 提取 cell 方向
    # cos 矩阵（numpy 可选依赖；失败时字段缺席，前端回退「见 npz」提示）
    try:
        import io
        import numpy as _np
        npz_name = next((n for n in files if n.endswith('.npz')), None)
        if npz_name:
            z = _np.load(io.BytesIO(base64.b64decode(files[npz_name])), allow_pickle=True)
            if 'cos' in z.files:
                cos = _np.asarray(z['cos'], dtype=float).tolist()
                keys = [str(k) for k in z['keys']] if 'keys' in z.files \
                    else [f'c{i}' for i in range(len(cos))]
                d['cos'] = {'keys': keys, 'matrix': cos}
    except Exception:
        pass
    return d


# ---------- S5 公开只读 ----------

@router.get('/distributed/summary')
def distributed_summary() -> dict:
    db = _db()
    now = time.time()
    nodes_total = db.execute('SELECT COUNT(*) AS n FROM nodes').fetchone()['n']
    nodes_online = db.execute(
        'SELECT node_id,name,gpu,model,dtype,last_seen FROM nodes WHERE last_seen>=? '
        'ORDER BY last_seen DESC LIMIT 12', (now - NODE_ONLINE_WINDOW,)).fetchall()
    tpl_rows = db.execute(
        '''SELECT t.tm_id,t.name,t.dim,t.status,
                  (SELECT COUNT(*) FROM results r WHERE r.tm_id=t.tm_id) AS results,
                  (SELECT COUNT(*) FROM tasks k WHERE k.tm_id=t.tm_id AND k.status='claimed') AS claims_active,
                  (SELECT COUNT(*) FROM tasks k WHERE k.tm_id=t.tm_id AND k.status='done') AS claims_done
           FROM templates t ORDER BY t.tm_id''').fetchall()
    results_total = db.execute('SELECT COUNT(*) AS n FROM results').fetchone()['n']
    downloads_total = db.execute('SELECT COALESCE(SUM(downloads),0) AS n FROM results').fetchone()['n']
    return {'version': ROUTER_VERSION, 'server_time': now,
            'nodes_online': len(nodes_online), 'nodes_total': nodes_total,
            'nodes': [dict(r) for r in nodes_online],
            'templates': [dict(r) for r in tpl_rows],
            'results_total': results_total, 'downloads_total': int(downloads_total),
            'agg_version': 'AGG-v0'}


@router.get('/agg/{tm_id}')
def agg_bucket(tm_id: str) -> dict:
    """AGG-v0：仅分桶回显（模型×kind×seed），不做跨桶平均。η² 跨节点汇总 = AGG-v1。"""
    rows = _db().execute(
        'SELECT sha,model_id,kind,seed,summary_json,uploaded_at FROM results '
        'WHERE tm_id=? ORDER BY uploaded_at', (tm_id,)).fetchall()
    buckets: Dict[str, list] = {}
    for r in rows:
        key = f"{r['model_id'] or 'unknown'}|{r['kind']}|seed{r['seed']}"
        buckets.setdefault(key, []).append(
            {'sha': r['sha'], 'summary': json.loads(r['summary_json'] or '{}'),
             'uploaded_at': r['uploaded_at']})
    return {'tm_id': tm_id, 'agg_version': 'AGG-v0',
            'warning': '跨桶简单平均被禁止（research_os 13.1）；本端点只做分桶回显',
            'buckets': buckets}


# ---------- S7 研究包（design/distributed_flow_plan_v1.md §2.2） ----------
# 研究包 = 语言模板（TM 注册表，服务端侧的实际可下载单元） × 分析技术 的一等组合。
# 分析技术注册表（服务端侧；与前端 distributedData.js ANALYSES 同源对齐）——
# input 契约在此换算为可对结果库判定的谓词，availability 由数据自动得出，不硬编码。

ANALYSIS_REGISTRY = [
    {'id': 'rsa-rdm', 'name': 'RSA / RDM 相似性结构', 'metric_version': 'client-v0(1−cos)',
     'evidence_level': 'observed', 'input': ['result.cos'],
     'note': '从 cell 方向 cos 矩阵现算 RDM（相异度=1−cos，0–2）'},
    {'id': 'eta2-decompose', 'name': 'η² 方差分解', 'metric_version': 'summary_digest-v2',
     'evidence_level': 'observed', 'input': ['result.summary.eta2_by_factor'],
     'note': '因子方差占比，summary_digest 白名单键'},
    {'id': 'cka-linear', 'name': 'CKA 中心核对齐', 'metric_version': 'pending-v0',
     'evidence_level': 'planned', 'input': ['result2.cos'],
     'note': '双结果对齐——需同模板≥2 份含 npz 的结果，P2 开放'},
]


def _result_paths(summary: dict) -> List[str]:
    return [str(f.get('path', '')) for f in ((summary or {}).get('_manifest') or [])]


def _result_satisfies(analysis_id: str, summary: dict) -> bool:
    """input 契约谓词：结果是否满足该分析技术的输入要求。"""
    paths = _result_paths(summary)
    if analysis_id == 'rsa-rdm':
        return any(p.endswith('.npz') for p in paths)
    if analysis_id == 'eta2-decompose':
        return 'eta2_by_factor' in (summary or {})
    if analysis_id == 'cka-linear':
        return False  # 双结果选择 P2 开放；单结果恒不满足
    return False


def _kit_tm_results(db: sqlite3.Connection, tm_id: str, analysis_id: str,
                    need: int = 1) -> List[dict]:
    """该模板下满足分析技术契约的结果清单（新→旧，≤50）。"""
    out: List[dict] = []
    rows = db.execute(
        'SELECT sha,model_id,kind,seed,summary_json,uploaded_at FROM results '
        'WHERE tm_id=? ORDER BY uploaded_at DESC LIMIT 200', (tm_id,)).fetchall()
    for r in rows:
        try:
            s = json.loads(r['summary_json'] or '{}')
        except Exception:
            s = {}
        if _result_satisfies(analysis_id, s):
            out.append({'sha': r['sha'], 'model_id': r['model_id'], 'kind': r['kind'],
                        'seed': r['seed'], 'uploaded_at': r['uploaded_at']})
            if len(out) >= 50:
                break
    return out if len(out) >= need else []


def _kit_readme(kit_id: str, tpl: sqlite3.Row, a: dict, manifest: List[dict]) -> str:
    lines = [
        f'# AI2050 研究包 {kit_id}',
        '',
        f'- 模板：{tpl["tm_id"]} v{tpl["version"]} · {tpl["name"]}（design_sha={tpl["design_sha"][:16]}…，预注册冻结）',
        f'- 分析技术：{a["name"]}（{a["id"]}）· 口径 {a["metric_version"]} · 证据级 {a["evidence_level"]}',
        f'- 满足输入契约的结果：{len(manifest)} 份（results_manifest.json）',
        '',
        '## 本地复现步骤',
        '',
        '1. **一键复现**：`python reproduce.py`（本包内含）——自动完成 design_sha 冻结校验 → '
        'runner 复现 → 输出对账（加 `--server http://<center>:5010` 联网逐文件比对结果库）'
        ' → 生成 reproduce_report.txt；真实模型口径加 `--model-path <HF目录>`。',
        '2. 手工校验冻结：用本包 corpus.json + runner.py 重算 design_sha（规范化 JSON+runner sha256，'
        '算法见 distributed_service.py _canonical_design_sha），须与 contract.json 的 design_sha 一致；',
        '3. 观察：结果清单中每份 sha 经 GET /api/results/{sha} 取回，文件逐一 sha256 校验；',
        f'4. 施加分析：{a["note"]}（口径 {a["metric_version"]}——跨口径比较被禁止，research_os 13.1）；',
        f'5. 跨节点 η²：包内 agg_v1 键 / GET /api/agg/{tpl["tm_id"]}/v1（桶内按因子汇总，离散度强制展示）。',
        '',
        '纪律：corpus/runner 不可变（变更须升模板 version）；结果内容寻址（sha256）；'
        '敏感原始数据不上传——包内只含统计量/均值/summary。',
    ]
    return '\n'.join(lines)


# 一键本地复现脚本（P3）：随研究包下发为 reproduce.py，与 bundle 同目录执行。
# r-string 下发：内层脚本的 \n 等转义须原样落盘。
REPRODUCE_SCRIPT = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI2050 研究包一键本地复现（S7-v1）——与本包（contract.json / corpus.json / runner.py）同目录执行
# 流程：design_sha 冻结校验 -> runner 复现 -> 输出与 results_manifest.json 逐文件对账 -> reproduce_report.txt
# 用法：python reproduce.py [--python PATH] [--model-path PATH] [--timeout SEC]
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def canonical_design_sha(contract, corpus, runner_bytes):
    payload = {'tm_id': contract.get('tm_id'), 'version': contract.get('version'),
               'corpus': corpus, 'runner_sha256': hashlib.sha256(runner_bytes).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def main():
    ap = argparse.ArgumentParser(description='AI2050 kit local reproduction')
    ap.add_argument('--python', default=sys.executable, help='执行 runner 的解释器（默认当前解释器）')
    ap.add_argument('--model-path', default='', help='HF 模型目录；留空=冒烟口径')
    ap.add_argument('--server', default='', help='中心节点地址（如 http://center:5010）；提供则对 results_manifest 逐结果联网对账')
    ap.add_argument('--timeout', type=float, default=1800.0)
    args = ap.parse_args()

    lines = []
    def log(s):
        print(s, flush=True)
        lines.append(s)

    log('# AI2050 reproduce · ' + HERE.name + ' · ' + time.strftime('%Y-%m-%d %H:%M:%S'))
    contract = json.loads((HERE / 'contract.json').read_text(encoding='utf-8'))
    corpus = json.loads((HERE / 'corpus.json').read_text(encoding='utf-8'))
    runner = (HERE / 'runner.py').read_bytes()
    design = canonical_design_sha(contract, corpus, runner)
    ok = design == contract.get('design_sha')
    log('[1] design_sha 冻结校验: ' + ('OK ' + design[:16] if ok else 'MISMATCH ' + design))
    if not ok:
        (HERE / 'reproduce_report.txt').write_text('\n'.join(lines), encoding='utf-8')
        sys.exit('design_sha mismatch — 研究包冻结校验失败，请上报中心节点')

    out = HERE / 'outputs'
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir()
    env = dict(os.environ)
    if args.model_path:
        env['AI2050_RUNNER_MODEL'] = args.model_path
    t0 = time.time()
    log('[2] runner 复现: ' + Path(args.python).name + ' runner.py --out outputs（timeout '
        + str(int(args.timeout)) + 's, model=' + ('real' if args.model_path else 'smoke') + '）')
    proc = subprocess.run([args.python, str(HERE / 'runner.py'), '--out', str(out)],
                          cwd=str(HERE), env=env, timeout=args.timeout,
                          capture_output=True, text=True)
    log('    exit=' + str(proc.returncode) + ' · ' + format(time.time() - t0, '.1f') + 's')
    if proc.returncode != 0 and proc.stderr:
        log('    stderr tail: ' + proc.stderr[-400:])
    log('    stdout tail: ' + (proc.stdout or '')[-200:].replace('\n', ' | '))

    log('[3] 输出对账（results_manifest.json 为结果级清单：sha=model×kind×seed 一次上传）')
    mf = HERE / 'results_manifest.json'
    entries = json.loads(mf.read_text(encoding='utf-8')) if mf.is_file() else []
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.iterdir()) if p.is_file()}
    for name in sorted(actual):
        log('    本地产出 ' + name + ' sha256=' + actual[name][:16])
    match = diff = checked = 0
    if args.server and entries:
        import base64 as b64
        import urllib.request
        server = args.server.rstrip('/')
        for e in entries:
            rsha = str(e.get('sha') or '')
            if not rsha:
                continue
            try:
                with urllib.request.urlopen(server + '/api/results/' + rsha, timeout=60) as resp:
                    payload = json.loads(resp.read().decode('utf-8'))
                files = payload.get('files') or {}
                same = 0
                for fname, fb64 in files.items():
                    fsha = hashlib.sha256(b64.b64decode(fb64)).hexdigest()
                    if fname in actual:
                        checked += 1
                        if fsha == actual[fname]:
                            same += 1
                            match += 1
                        else:
                            diff += 1
                            log('    ' + rsha[:12] + '/' + fname + ': DIFF（本地与中心节点字节不一致）')
                log('    结果 ' + rsha[:12] + '（' + str(e.get('model_id')) + '|'
                    + str(e.get('kind')) + '|seed' + str(e.get('seed')) + '）: '
                    + str(same) + '/' + str(len(files)) + ' 文件与本地 runner 输出一致')
            except Exception as ex:
                log('    结果 ' + rsha[:12] + ' 拉取失败: ' + str(ex)[:120])
    else:
        log('    （未提供 --server 或清单为空——跳过联网对账；真实模型重跑允许非逐字节一致）')
    log('[4] 结论: 联网对账 ' + str(match) + ' MATCH / ' + str(diff) + ' DIFF（checked '
        + str(checked) + '）· 纪律：统计量跨口径比较被禁止（research_os 13.1）')
    (HERE / 'reproduce_report.txt').write_text('\n'.join(lines), encoding='utf-8')
    log('report -> reproduce_report.txt')


if __name__ == '__main__':
    main()
'''


def _agg_v1_payload(db: sqlite3.Connection, tm_id: str) -> dict:
    """AGG-v1：桶内（模型×kind×seed）η² 按因子跨节点/seed 汇总，离散度强制展示。
    纪律：跨桶平均仍被禁止（research_os 13.1）；无 η² 数据的桶不进入输出。"""
    rows = db.execute(
        'SELECT sha,model_id,kind,seed,summary_json,uploaded_at FROM results '
        'WHERE tm_id=? ORDER BY uploaded_at', (tm_id,)).fetchall()
    buckets: Dict[str, list] = {}
    for r in rows:
        try:
            s = json.loads(r['summary_json'] or '{}')
        except Exception:
            continue
        e2 = s.get('eta2_by_factor')
        if not isinstance(e2, dict) or not e2:
            continue
        seed = r['seed'] if r['seed'] is not None else 0
        key = f"{r['model_id'] or 'unknown'}|{r['kind']}|seed{seed}"
        buckets.setdefault(key, []).append((r['sha'], e2))
    out = {}
    for key, items in buckets.items():
        factors = sorted({f for _, e2 in items for f in e2 if isinstance(e2.get(f), (int, float))})
        per = {}
        for f in factors:
            vals = [e2[f] for _, e2 in items if isinstance(e2.get(f), (int, float))]
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            per[f] = {'n': n, 'mean': round(mean, 6), 'min': round(min(vals), 6),
                      'max': round(max(vals), 6), 'std': round(var ** 0.5, 6)}
        out[key] = {'n_results': len(items), 'eta2_by_factor': per,
                    'shas': [sha for sha, _ in items]}
    return {'tm_id': tm_id, 'agg_version': 'AGG-v1',
            'warning': '仅桶内（模型×kind×seed）η² 按因子汇总；跨桶平均被禁止（research_os 13.1）；'
                       'std>0 的因子禁止合并解读',
            'buckets': out, 'buckets_total': len(out)}


@router.get('/agg/{tm_id}/v1')
def agg_v1(tm_id: str) -> dict:
    return _agg_v1_payload(_db(), tm_id)


# ---------- S10 dist_protocol v3：生成式模板 / 合并 / 覆盖 / 审计 / 总览 / 记分板 ----------

def _proto() -> None:
    if _pipeline is None:
        raise HTTPException(503, f'dist_protocol v3 不可用（{_PROTO_ERR}）；确认 numpy 已安装且 shared/dist_protocol 在仓库中')


@router.get('/grammar/library')
def grammar_library() -> dict:
    """内置模板族文法（shared/dist_protocol/grammars/*.json）——模板工作室的起点。"""
    _proto()
    out = []
    for path in sorted(_GRAMMAR_DIR.glob('*.json')):
        try:
            spec = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            out.append({'file': path.name, 'error': str(exc)})
            continue
        out.append({'file': path.name, 'family_id': spec.get('family_id'), 'module': spec.get('module', ''),
                    'name': spec.get('name', ''), 'description': spec.get('description', ''),
                    'factors': spec.get('factors', []), 'n_frames': len(spec.get('frames', [])),
                    'n_fillers': len(spec.get('fillers', [])), 'spec': spec})
    db = _db()
    frozen = {}
    for t in db.execute('SELECT tm_id,version,design_sha,bundle_dir FROM templates').fetchall():
        c = _pipeline.template_corpus(db, t['tm_id'])
        if _pipeline.is_v3(c):
            frozen.setdefault(c.get('family_id'), []).append(
                {'tm_id': t['tm_id'], 'version': t['version'], 'mode': c.get('mode'),
                 'design_sha': t['design_sha'], 'grammar_sha256': c.get('grammar_sha256')})
    for row in out:
        row['registered'] = frozen.get(row.get('family_id'), [])
    return {'schema': 'grammar_library.v1', 'grammars': out}


class GrammarPreviewIn(BaseModel):
    spec: dict
    dims: int = Field(default=2560, ge=1, le=65536)


@router.post('/grammar/preview')
def grammar_preview(b: GrammarPreviewIn, x_admin_token: str = Header(default='')) -> dict:
    """模板预览有真实计算成本（生成式展开+功效近似），与其它管理写操作同级鉴权。"""
    _auth_admin(x_admin_token)
    _proto()
    return _grammar.preview(b.spec, dims=b.dims)


class GrammarRegisterIn(BaseModel):
    spec: dict
    mode: Literal['pilot', 'final'] = 'final'
    pilot_per_cell: int = Field(default=4, ge=1, le=64)


@router.post('/grammar/register')
def grammar_register(b: GrammarRegisterIn, x_admin_token: str = Header(default='')) -> dict:
    """冻结生成式模板：pilot = 每格少量样本、单分片，用于行为试跑；final = 完整语料与分片。
    tm_id = TG-<family>（pilot 加 -P）；语料或 runner 变化时自动升 version（旧版本结果不再合并）。"""
    _auth_admin(x_admin_token)
    _proto()
    errors = _grammar.validate(b.spec)
    if errors:
        raise HTTPException(422, {'errors': errors})
    fam = b.spec['family_id']
    tm_id = f'TG-{fam}' + ('-P' if b.mode == 'pilot' else '')
    try:
        if b.mode == 'pilot':
            corpus = _grammar.generate(_grammar.pilot_spec(b.spec), per_cell_limit=b.pilot_per_cell)
        else:
            corpus = _grammar.generate(b.spec)
    except _grammar.GrammarError as exc:
        raise HTTPException(422, {'errors': exc.errors})
    corpus.pop('_warnings', None)
    corpus['mode'] = b.mode
    runner = _RUNNER_V3.read_text(encoding='utf-8')
    db = _db()
    cur = db.execute('SELECT version,design_sha FROM templates WHERE tm_id=?', (tm_id,)).fetchone()
    version = 1
    if cur:
        prev = _pipeline.template_corpus(db, tm_id)
        same = dict(prev); same.pop('tm_id', None); same.pop('version', None)
        if same == corpus and _canonical_design_sha(tm_id, cur['version'], prev, runner) == cur['design_sha']:
            return {'tm_id': tm_id, 'version': cur['version'], 'design_sha': cur['design_sha'],
                    'mode': b.mode, 'shards': int(corpus.get('shards', 1) or 1), 'idempotent_replay': True,
                    'n_items': sum(1 for it in corpus['items'] if not it.get('canary')),
                    'note': '同一文法与 runner 已冻结过：沿用现有版本（其他模型可继续补齐自己的分片）。'}
        version = int(cur['version']) + 1
    corpus = {'tm_id': tm_id, 'version': version, **corpus}
    factors = corpus['factors']
    design_sha = _register_template({
        'tm_id': tm_id, 'version': version,
        'name': b.spec.get('name', fam) + (' · 试跑' if b.mode == 'pilot' else ''),
        'dim': ' × '.join(factors) + f"；{sum(1 for it in corpus['items'] if not it.get('canary'))} 条 · {corpus['shards']} 分片",
        'description': b.spec.get('description', ''),
        'metric_pin': 'dist_protocol.v3 · suffstats.v1',
        'model_pin': '节点声明 model_id / model_rev / precision，按桶合并',
        'corpus': corpus, 'runner': runner,
    })
    return {'tm_id': tm_id, 'version': version, 'design_sha': design_sha, 'mode': b.mode,
            'shards': corpus['shards'], 'n_items': sum(1 for it in corpus['items'] if not it.get('canary')),
            'note': '已冻结：语料与 runner 不可变；变更文法后重新冻结会升 version，旧版本结果不再参与合并。'}


@router.get('/grammar/{tm_id}/report')
def grammar_report(tm_id: str) -> dict:
    """试跑/正式模板执行报告：逐格行为门、分片覆盖、排除原因（不含大载荷）。"""
    _proto()
    db = _db()
    corpus = _pipeline.template_corpus(db, tm_id)
    if not corpus:
        raise HTTPException(404, f'template {tm_id} not found')
    agg = _pipeline.aggregate(db, _DATA_DIR, tm_id, gate=False)
    buckets = []
    for bk in agg['buckets']:
        light = {k: v for k, v in bk.items() if k != 'agg'}
        light['cells_by_split'] = (bk.get('agg') or {}).get('cells_by_split', {})
        buckets.append(light)
    tasks = {r['status']: r['n'] for r in db.execute('SELECT status,COUNT(*) AS n FROM tasks WHERE tm_id=? GROUP BY status', (tm_id,))}
    smoke = db.execute("SELECT COUNT(*) AS n FROM results WHERE tm_id=? AND kind!='real'", (tm_id,)).fetchone()['n']
    return {'tm_id': tm_id, 'family_id': corpus.get('family_id'), 'mode': corpus.get('mode'),
            'prereg': corpus.get('prereg', {}), 'shards': corpus.get('shards', 1),
            'n_items': sum(1 for it in corpus.get('items', []) if not it.get('canary')),
            'tasks': tasks, 'non_real_results': smoke, 'buckets': buckets, 'gates': _pipeline.gate_chain(agg)}


@router.get('/agg/{tm_id}/v2')
def agg_v2(tm_id: str, bucket: str = '', gate: bool = True, layer: str = '', factor: str = '') -> dict:
    """AGG-v2：充分统计量跨节点精确合并（桶 = 模型快照 × 精度）。layer+factor 给出时附逐维 η²。"""
    _proto()
    full = (layer, factor) if layer and factor else None
    return _pipeline.aggregate(_db(), _DATA_DIR, tm_id, bucket=bucket or None, gate=gate, full=full)


@router.get('/analysis/{tm_id}')
def family_analysis(tm_id: str, bucket: str = '', gate: bool = True):
    """模式族规律分析（analysis.v1）：置换零分布、留出探针、白化方向、句式不变性、组合可加性、
    词身份与长度对照、3D 响应空间坐标与规律陈述。需要结果附带 items.npz。"""
    _proto()
    out = _pipeline.analyze_bucket(_db(), _DATA_DIR, tm_id, bucket=bucket or None, gate=gate, background=True, as_path=True)
    if isinstance(out, Path):                           # 已缓存：原样发送文件，不在网站进程里解析与重新编码
        from fastapi.responses import FileResponse
        return FileResponse(out, media_type='application/json')
    return out


@router.get('/coverage')
def coverage_matrix() -> dict:
    _proto()
    return _pipeline.coverage(_db(), _DATA_DIR)


@router.get('/audit/queue')
def audit_queue(status: str = Query(default='pending', pattern='^(pending|passed|rejected|needs_rerun|all)$'),
                limit: int = Query(default=100, ge=1, le=500)) -> dict:
    _proto()
    return _pipeline.audit_queue(_db(), status=status, limit=limit)


class AuditIn(BaseModel):
    status: Literal['passed', 'rejected', 'needs_rerun']
    reviewer: str = Field(default='', max_length=80)
    note: str = Field(default='', max_length=2000)


@router.post('/audit/{sha}')
def audit_record(sha: str, b: AuditIn, x_admin_token: str = Header(default='')) -> dict:
    """人工审计裁决（需管理令牌）：只决定执行产物能否进入合并；不创建 Evidence。"""
    _auth_admin(x_admin_token)
    _proto()
    db = _db()
    row = db.execute('SELECT * FROM results WHERE sha=?', (sha,)).fetchone()
    if not row:
        raise HTTPException(404, 'result not found')
    checks = _pipeline.audit_checks(db, row, _pipeline.summary_of(row))
    if b.status == 'passed' and any(c['status'] == 'fail' for c in checks):
        raise HTTPException(409, '自动检查存在 fail 项，不能标记为通过；请选择驳回或需重跑')
    return _pipeline.record_audit(db, _db_lock, sha, b.status, b.reviewer, b.note, checks)


@router.get('/pipeline/overview')
def pipeline_overview() -> dict:
    _proto()
    return _pipeline.overview(_db(), _DATA_DIR, node_online_window=NODE_ONLINE_WINDOW)


@router.get('/scoreboard')
def scoreboard() -> dict:
    _proto()
    return _pipeline.scoreboard(_db(), _DATA_DIR)


@router.get('/kits')
def kits_list() -> dict:
    """S7 枚举：模板 × 分析技术组合与可用性（由结果库按 input 契约自动判定）。"""
    db = _db()
    tpls = db.execute(
        'SELECT tm_id,name,version,design_sha,status FROM templates ORDER BY tm_id').fetchall()
    kits = []
    for t in tpls:
        for a in ANALYSIS_REGISTRY:
            got = _kit_tm_results(db, t['tm_id'], a['id'])
            kits.append({'kit_id': f"{t['tm_id']}__{a['id']}",
                         'tm_id': t['tm_id'], 'template_name': t['name'],
                         'template_version': t['version'], 'design_sha': t['design_sha'],
                         'analysis': a['id'], 'analysis_name': a['name'],
                         'metric_version': a['metric_version'],
                         'evidence_level': a['evidence_level'],
                         'status': 'available' if got else 'pending_results',
                         'results_total': len(got)})
    return {'registry_version': 'S7-v0', 'kits': kits,
            'analyses': ANALYSIS_REGISTRY,
            'note': 'kit_id={tm_id}__{analysis}；availability 由结果库按 input 契约判定'}


@router.get('/kits/{kit_id}/bundle')
def kit_bundle(kit_id: str) -> dict:
    """S7 整包下载：bundle 文件（corpus/runner/contract）+ 结果清单 + README。
    公开只读；不暴露节点 token 与任务私有载荷（S8 纪律）。"""
    if '__' not in kit_id:
        raise HTTPException(400, 'kit_id 格式：{tm_id}__{analysis}')
    tm_id, _, analysis_id = kit_id.partition('__')
    db = _db()
    tpl = db.execute('SELECT * FROM templates WHERE tm_id=?', (tm_id,)).fetchone()
    if not tpl:
        raise HTTPException(404, f'template {tm_id} not found')
    a = next((x for x in ANALYSIS_REGISTRY if x['id'] == analysis_id), None)
    if not a:
        raise HTTPException(404, f'analysis {analysis_id} not registered')
    manifest = _kit_tm_results(db, tm_id, analysis_id)
    if not manifest:
        raise HTTPException(409, f'kit {kit_id} pending_results：结果库暂无满足 input 契约的数据')
    files = _bundle_files(tpl['bundle_dir'])
    files['reproduce.py'] = REPRODUCE_SCRIPT     # P3：一键本地复现脚本随包下发
    agg_v1 = _agg_v1_payload(db, tm_id)          # P3：跨节点 η² 汇总纳入包
    with _db_lock:
        for m in manifest:  # 研究包下载计入各结果引用计数
            db.execute('UPDATE results SET downloads=downloads+1 WHERE sha=?', (m['sha'],))
        db.commit()
    return {'kit_id': kit_id, 'bundle_version': 'S7-v1',
            'template': {'tm_id': tm_id, 'version': tpl['version'], 'name': tpl['name'],
                         'dim': tpl['dim'], 'design_sha': tpl['design_sha'],
                         'metric_pin': tpl['metric_pin'], 'model_pin': tpl['model_pin']},
            'analysis': a, 'results_manifest': manifest, 'agg_v1': agg_v1,
            'files': files, 'README.md': _kit_readme(kit_id, tpl, a, manifest)}


# ---------- S9 实时模型连接（空间透镜「实时分析」模式） ----------
# 连接本机 HF 模型目录：读 config.json + safetensors 权重头（mmap header，不加载权重），
# 生成「全部参数」真实清单（每 tensor 的 name/shape/dtype/params + 分组汇总 + 总量）。
# 只读操作、秒级完成；激活级实时流（真实 forward）属 P2，需要 runner 侧加载权重。
class LiveConnectReq(BaseModel):
    model_path: str

_LIVE = {'connected': False, 'payload': None}   # 进程内单连接（本机工具语义）

_ST_DTYPE = {'BF16': 'bf16', 'F16': 'fp16', 'F32': 'fp32', 'F64': 'fp64',
             'I8': 'int8', 'I16': 'int16', 'I32': 'int32', 'I64': 'int64',
             'U8': 'uint8', 'U16': 'uint16', 'U32': 'uint32', 'U64': 'uint64',
             'BOOL': 'bool', 'F8_E4M3': 'fp8_e4m3', 'F8_E5M2': 'fp8_e5m2'}

def _st_header(path):
    """读 safetensors 文件头（前 8 字节长度 + JSON header），不触碰张量数据。"""
    with open(path, 'rb') as f:
        (n,) = struct.unpack('<Q', f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop('__metadata__', None)
    return hdr

def _scan_live_model(mp) -> dict:
    if not mp.is_dir():
        raise HTTPException(404, f'目录不存在: {mp}')
    cfg = {}
    if (mp / 'config.json').is_file():
        try:
            cfg = json.loads((mp / 'config.json').read_text(encoding='utf-8'))
        except Exception:
            cfg = {}
    shards = sorted(mp.glob('*.safetensors'))
    if not shards:
        raise HTTPException(400, '未找到 *.safetensors 权重文件——实时参数表当前支持 HF safetensors 格式（.bin 不受支持）')
    tensors, files = [], []
    for sp in shards:
        try:
            hdr = _st_header(sp)
        except Exception as e:
            raise HTTPException(400, f'权重头解析失败 {sp.name}: {e}')
        files.append({'name': sp.name, 'bytes': sp.stat().st_size})
        for name, meta in hdr.items():
            shape = [int(d) for d in (meta.get('shape') or [])]
            params = 1
            for d in shape:
                params *= d
            raw_dt = str(meta.get('dtype', '?'))
            tensors.append({'name': name, 'shape': shape,
                            'dtype': _ST_DTYPE.get(raw_dt, raw_dt.lower()), 'params': params,
                            'file': sp.name})
    tensors.sort(key=lambda t: t['name'])
    groups: dict = {}
    for t in tensors:                       # 分组：model.layers.N 各成一组，其余取前 2 段
        seg = t['name'].split('.')
        key = '.'.join(seg[:3]) if (len(seg) >= 3 and seg[0] == 'model' and seg[1] == 'layers') \
            else ('.'.join(seg[:2]) if len(seg) > 1 else seg[0])
        g = groups.setdefault(key, {'prefix': key, 'tensor_count': 0, 'params': 0, 'tensors': []})
        g['tensor_count'] += 1
        g['params'] += t['params']
        g['tensors'].append(t)
    dtypes: dict = {}
    for t in tensors:
        dtypes[t['dtype']] = dtypes.get(t['dtype'], 0) + t['params']
    return {'model_path': str(mp),
            'arch': (cfg.get('architectures') or [None])[0],
            'config': {'layers': cfg.get('num_hidden_layers'), 'hidden': cfg.get('hidden_size'),
                       'heads': cfg.get('num_attention_heads'), 'kv_heads': cfg.get('num_key_value_heads'),
                       'vocab': cfg.get('vocab_size'), 'tie_word_embeddings': cfg.get('tie_word_embeddings'),
                       'torch_dtype': cfg.get('torch_dtype')},
            'total_params': sum(t['params'] for t in tensors), 'dtypes': dtypes,
            'files': files, 'tensors': tensors, 'groups': list(groups.values()), 'connected_at': time.time()}

@router.post('/live/connect')
def live_connect(req: LiveConnectReq):
    mp = Path(req.model_path.strip()).expanduser()
    try:
        payload = _scan_live_model(mp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f'连接失败: {e}')
    _LIVE['connected'] = True
    _LIVE['payload'] = payload
    return payload

@router.get('/live/status')
def live_status():
    if not _LIVE['connected']:
        return {'connected': False}
    return {'connected': True, **_LIVE['payload']}   # 保留 connected 标志，前端据此判定 LIVE

@router.post('/live/disconnect')
def live_disconnect():
    _LIVE['connected'] = False
    _LIVE['payload'] = None
    _WT_T_CACHE.clear(); _WT_SCAN_CACHE['key'] = None   # 数值缓存随连接失效
    return {'connected': False}


# ---------- S9c 本地模型库（目录扫描 + HF 下载；空间透镜·实时分析·「模型库 / 下载」弹窗） ----------
# 扫描：AI2050_MODELS_DIR（默认 <repo>/models）一级子目录 + HF_HOME/hub 缓存快照，
#       只收「config.json + *.safetensors」齐备的目录（与 /live/connect 契约一致）。
# 下载：后台 subprocess 调 huggingface_hub CLI（--local-dir 平铺；排除 bin/pth/gguf/onnx，
#       只下 safetensors 口径）。任务状态仅进程内（产物落盘后扫描端点可见）；
#       HF_ENDPOINT 沿用服务端进程 env（国内可设镜像加速）。

class ModelDownloadReq(BaseModel):
    repo_id: str

_MODEL_CATALOG = [
    {'repo_id': 'Qwen/Qwen3-4B',            'name': 'Qwen3-4B',    'size': '≈8 GB · bf16',  'note': '研究线主模型 qwen3-4b'},
    {'repo_id': 'Qwen/Qwen3-14B',           'name': 'Qwen3-14B',   'size': '≈28 GB · bf16', 'note': '研究线 14B 对照'},
    {'repo_id': 'zai-org/glm-4-9b-chat-hf', 'name': 'GLM-4-9B',    'size': '≈18 GB · bf16', 'note': '研究线 glm4-9b（safetensors）'},
    {'repo_id': 'openai-community/gpt2',    'name': 'GPT-2 small', 'size': '≈0.5 GB',       'note': 'server.py 启动默认模型'},
]

def _models_root() -> Path:
    env = os.environ.get('AI2050_MODELS_DIR', '').strip()
    root = Path(env).expanduser() if env else Path(__file__).resolve().parent.parent / 'models'
    root.mkdir(parents=True, exist_ok=True)
    return root

def _model_dir_info(d: Path):
    if not (d / 'config.json').is_file():
        return None
    sts = sorted(d.glob('*.safetensors'))
    if not sts:
        return None
    try:
        cfg = json.loads((d / 'config.json').read_text(encoding='utf-8'))
    except Exception:
        cfg = {}
    return {'name': d.name, 'path': str(d),
            'arch': (cfg.get('architectures') or [None])[0],
            'layers': cfg.get('num_hidden_layers'), 'hidden': cfg.get('hidden_size'),
            'files': len(sts), 'bytes': sum(p.stat().st_size for p in sts)}

@router.get('/models/local')
def models_local():
    root = _models_root()
    out, seen = [], set()
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith('.'):
            info = _model_dir_info(d)
            if info:
                out.append(info); seen.add(str(d.resolve()))
    hf = Path(os.environ.get('HF_HOME') or os.path.expanduser('~/.cache/huggingface'))
    for snap in list((hf / 'hub').glob('models--*/snapshots/*/'))[:50]:
        if snap.is_dir() and str(snap.resolve()) not in seen:
            info = _model_dir_info(snap)
            if info:
                info['name'] = snap.parent.parent.name.replace('models--', '', 1).replace('--', '/')
                info['cached'] = True
                out.append(info); seen.add(str(snap.resolve()))
    return {'root': str(root), 'models': out, 'catalog': _MODEL_CATALOG}

_MODEL_DL: Dict[str, dict] = {}          # repo_id -> 任务状态（进程内；不含 proc 对象）
_MD_LOCK = threading.Lock()

def _md_watch(proc, st):
    for line in iter(proc.stdout.readline, ''):
        st['tail'] = (st['tail'] + [line.strip()])[-6:]
    proc.wait()
    st['status'] = 'done' if proc.returncode == 0 else 'failed'
    st['exit'] = proc.returncode
    st['ended'] = time.time()

@router.post('/models/download')
def models_download(req: ModelDownloadReq):
    repo_id = req.repo_id.strip()
    if '/' not in repo_id or any(c in repo_id for c in ' ;&|`$"\'' ):
        raise HTTPException(400, '非法 repo_id')
    with _MD_LOCK:
        st = _MODEL_DL.get(repo_id)
        if st and st['status'] == 'downloading':
            raise HTTPException(409, f'{repo_id} 正在下载中')
        target = _models_root() / repo_id.split('/')[-1]
        if (target / 'config.json').is_file() and list(target.glob('*.safetensors')):
            raise HTTPException(409, f'{repo_id} 已存在于 {target}')
        cmd = [sys.executable, '-m', 'huggingface_hub.commands.huggingface_cli',
               'download', repo_id, '--local-dir', str(target),
               '--exclude', '*.bin', '--exclude', '*.pth', '--exclude', '*.gguf', '--exclude', '*.onnx']
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf-8', errors='replace')
        except Exception as e:
            raise HTTPException(500, f'下载进程启动失败: {e}')
        st = {'repo_id': repo_id, 'status': 'downloading', 'dir': str(target),
              'started': time.time(), 'tail': [], 'pid': proc.pid}
        _MODEL_DL[repo_id] = st
    threading.Thread(target=_md_watch, args=(proc, st), daemon=True).start()
    return {'ok': True, 'repo_id': repo_id, 'dir': str(target), 'pid': proc.pid}

@router.get('/models/downloads')
def models_downloads():
    with _MD_LOCK:
        return {'downloads': [dict(st) for st in _MODEL_DL.values()]}


# ---------- S9b 权重数值（P2 第一步：结构 → 数值） ----------
# 懒加载真实权重数值：safetensors get_slice 按连续行块切片读（不整体加载权重）；
# 行数超限时在行域均匀抽 32 块 × 64 行，标量统计/热图/行范数全部降采样返回。
# 消费方：全部参数表（summary 数值列）、3D 层盒数值着色（scan）、参数热图（tensor 详情）。
_WT_T_CACHE = {}                                    # ('s'|'d', name) -> payload（连接会话内有效，disconnect 清空）
_WT_SCAN_CACHE = {'key': None, 'data': None}
_WT_BLOCK_ROWS, _WT_BLOCKS = 64, 32
_WT_HEAT_COLS, _WT_RN_POINTS = 64, 512


def _wt_conn_payload():
    if not _LIVE['connected'] or not _LIVE['payload']:
        raise HTTPException(409, '未连接本地模型——先 POST /api/live/connect')
    return _LIVE['payload']


def _wt_tmap(payload):
    return {t['name']: t for t in payload['tensors']}


_ST_ITEMSIZE = {                        # safetensors dtype → 单元素字节数（BF16 存为 2 字节）
    'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2,
    'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1,
}
_ST_HDR_CACHE = {}                      # path -> (header, data_base_offset, size, mtime)


def _st_np_dtype(dtype):
    import numpy as np
    return {
        'F64': np.float64, 'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16,
        'I64': np.int64, 'I32': np.int32, 'I16': np.int16, 'I8': np.int8, 'U8': np.uint8,
    }[dtype]


def _wt_header(path):
    """解析 safetensors header（8B 长度 + JSON）并返回 (header, data_base_offset)。

    注意：不要与上方 `_st_header` 混用——那个只返回 header（供模型扫描），
    本函数额外返回数据区起始偏移，且带 (size, mtime) 缓存。
    """
    st = os.stat(str(path))
    hit = _ST_HDR_CACHE.get(str(path))
    if hit and hit[2] == st.st_size and hit[3] == st.st_mtime:
        return hit[0], hit[1]
    with open(str(path), 'rb') as f:
        (n,) = struct.unpack('<Q', f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop('__metadata__', None)
    base = 8 + n
    _ST_HDR_CACHE[str(path)] = (hdr, base, st.st_size, st.st_mtime)
    return hdr, base


def _st_read_rows(path, name, hdr, base, r0, r1):
    """精确 seek 直读行区间 [r0, r1)，返回 float32 torch tensor [r1-r0, cols]（不 mmap）。"""
    import numpy as np
    import torch
    meta = hdr[name]
    dtype = meta['dtype']
    shape = meta['shape'] or []
    np_dt = _st_np_dtype(dtype)
    item = _ST_ITEMSIZE[dtype]
    cols = int(shape[-1]) if shape else 1
    row_bytes = cols * item
    off0 = int(meta['data_offsets'][0])
    want = (r1 - r0) * row_bytes
    with open(str(path), 'rb') as f:
        f.seek(base + off0 + r0 * row_bytes)
        raw = f.read(want)
    if len(raw) != want:
        raise IOError(f'safetensors 短读 {name} rows {r0}-{r1}: {len(raw)}/{want}')
    arr = np.frombuffer(raw, dtype=np_dt).reshape(r1 - r0, cols)
    if dtype == 'BF16':
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    elif arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _wt_blocks(path, name, tmeta):
    """yield (start_row, block[float32], sampled)：2D 按行块切片；1D 整读一次。

    实现改用「safetensors header + 精确 seek 直读」，**不再 mmap**。
    原因（2026-10-19 实测）：Windows 上 `safe_open(framework='pt')` 每次调用都会映射
    整个权重分片；逐层扫描（36 层 × 每层多个 tensor）时累积未释放的 mapping 会耗尽
    系统提交内存，抛 `OSError: 页面文件太小 (os error 1455)`——单次调用正常、连续
    调用必挂，导致 /live/weights/scan 与 /live/weights/summary 在真实模型上不可用。
    直读只取所需行块的字节，内存峰值 = 单个块（数 MB），且不依赖页面文件余量。
    数值与 safe_open 路径逐项一致（验证脚本 tests/gpt5_temp/_st_direct_read.py）。
    """
    shape = tmeta.get('shape') or []
    hdr, base = _wt_header(path)
    if name not in hdr:
        raise HTTPException(404, f'tensor 不在权重分片中: {name}')
    if len(shape) <= 1:
        yield 0, _st_read_rows(path, name, hdr, base, 0, 1).reshape(-1), False
        return
    rows = int(shape[0])
    need = _WT_BLOCKS * _WT_BLOCK_ROWS
    if rows <= need:
        starts = list(range(0, rows, _WT_BLOCK_ROWS))
        sampled = False
    else:
        step = (rows - need) / (_WT_BLOCKS - 1)
        starts = [int(round(i * step)) for i in range(_WT_BLOCKS)]
        sampled = True
    for s in starts:
        e = min(rows, s + _WT_BLOCK_ROWS)
        if e <= s:
            continue
        yield s, _st_read_rows(path, name, hdr, base, s, e), sampled


def _wt_scalar_stats(payload, tmap, name):
    import math
    hit = _WT_T_CACHE.get(('s', name))
    if hit:
        return hit
    n = 0; s1 = 0.0; s1a = 0.0; s2 = 0.0
    mn = None; mx = None; sampled = False; rows_seen = 0
    fp = Path(payload['model_path']) / tmap[name]['file']
    for _s, t, smp in _wt_blocks(fp, name, tmap[name]):
        sampled = sampled or smp
        n += int(t.numel())
        rows_seen += int(t.shape[0]) if t.dim() else 1
        s1 += float(t.sum()); s1a += float(t.abs().sum()); s2 += float((t * t).sum())
        tmin, tmax = float(t.min()), float(t.max())
        mn = tmin if mn is None else min(mn, tmin)
        mx = tmax if mx is None else max(mx, tmax)
    if not n:
        raise HTTPException(400, f'空 tensor: {name}')
    out = {'name': name, 'mean': round(s1 / n, 6), 'mean_abs': round(s1a / n, 6),
           'rms': round(math.sqrt(s2 / n), 6), 'min': round(mn, 6), 'max': round(mx, 6),
           'max_abs': round(max(abs(mn), abs(mx)), 6),
           'sampled': sampled, 'rows_examined': rows_seen}
    _WT_T_CACHE[('s', name)] = out
    return out


def _downsample(v, k):
    import torch
    t = torch.as_tensor(v, dtype=torch.float32)
    n = int(t.numel())
    if n <= k:
        return [round(float(x), 6) for x in t.tolist()]
    edges = [int(round(i * n / k)) for i in range(k + 1)]
    out = []
    for i in range(k):
        seg = t[edges[i]:max(edges[i + 1], edges[i] + 1)]
        out.append(round(float(seg.mean()), 6))
    return out


@router.get('/live/weights/scan')
def live_weights_scan():
    """每层代表权重（mlp.down_proj.weight）标量统计——3D 层盒数值着色数据源。"""
    import re
    payload = _wt_conn_payload()
    key = (payload['model_path'], tuple(f['name'] for f in payload['files']))
    if _WT_SCAN_CACHE['key'] == key:
        return _WT_SCAN_CACHE['data']
    tmap = _wt_tmap(payload)
    pat = re.compile(r'^(?:model\.)?layers\.(\d+)\.mlp\.down_proj\.weight$')
    found = {}
    for nm, tm in tmap.items():
        m = pat.match(nm)
        if m and len(tm.get('shape') or []) == 2:
            found[int(m.group(1))] = nm
    layers = []
    for l in sorted(found):
        st = _wt_scalar_stats(payload, tmap, found[l])
        layers.append({'layer': l, 'name': found[l], 'rms': st['rms'],
                       'mean_abs': st['mean_abs'], 'max_abs': st['max_abs'], 'sampled': st['sampled']})
    data = {'layers': layers, 'source': 'safetensors 懒加载真实权重（32×64 行块，超限均匀抽样）'}
    _WT_SCAN_CACHE['key'] = key
    _WT_SCAN_CACHE['data'] = data
    return data


class _WtNamesReq(BaseModel):
    names: List[str]


@router.post('/live/weights/summary')
def live_weights_summary(req: _WtNamesReq):
    """批量标量统计（≤24 名）——全部参数表数值列。"""
    payload = _wt_conn_payload()
    tmap = _wt_tmap(payload)
    out = {}
    for nm in req.names[:24]:
        if nm in tmap and tmap[nm].get('params', 0) >= 16:
            out[nm] = _wt_scalar_stats(payload, tmap, nm)
    return {'results': out, 'block': '32×64 行块（超限均匀抽样）'}


class _WtNameReq(BaseModel):
    name: str


class _WtSliceReq(BaseModel):
    name: str
    start: List[int]
    size: List[int]


@router.post('/live/tensor/slice')
def live_tensor_slice(req: _WtSliceReq):
    """Read an exact, bounded safetensors slice without loading the full tensor."""
    from safetensors import safe_open
    import math

    payload = _wt_conn_payload()
    tmap = _wt_tmap(payload)
    name = req.name.strip()
    if name not in tmap:
        raise HTTPException(404, f'tensor 不存在: {name}')

    meta = tmap[name]
    shape = [int(dim) for dim in (meta.get('shape') or [])]
    if len(req.start) != len(shape) or len(req.size) != len(shape):
        raise HTTPException(400, f'切片维度不匹配：tensor rank={len(shape)}')
    if any(int(value) < 0 for value in req.start):
        raise HTTPException(400, '切片起点不能小于 0')
    if any(int(value) <= 0 for value in req.size):
        raise HTTPException(400, '切片数量必须大于 0')
    if any(start >= dim for start, dim in zip(req.start, shape)):
        raise HTTPException(400, '切片起点超出 tensor 边界')

    end = [min(start + size, dim) for start, size, dim in zip(req.start, req.size, shape)]
    actual_shape = [stop - start for start, stop in zip(req.start, end)]
    loaded_numel = math.prod(actual_shape) if actual_shape else 1
    if loaded_numel > 4096:
        raise HTTPException(413, '单次切片最多读取 4096 个标量')

    file_path = Path(payload['model_path']) / meta['file']
    try:
        with safe_open(str(file_path), framework='pt', device='cpu') as f:
            if shape:
                index = tuple(slice(start, stop) for start, stop in zip(req.start, end))
                tensor = f.get_slice(name)[index]
                values = tensor.tolist()
            else:
                values = f.get_tensor(name).item()
    except Exception as e:
        raise HTTPException(400, f'读取 tensor 切片失败: {e}')

    return {
        'name': name,
        'dtype': meta.get('dtype'),
        'shape': shape,
        'start': req.start,
        'requested_size': req.size,
        'actual_shape': actual_shape,
        'values': values,
        'loaded_numel': loaded_numel,
        'source': 'safetensors_exact_slice',
    }


@router.post('/live/tensor')
def live_tensor_detail(req: _WtNameReq):
    """单 tensor 数值详情：标量统计 + |W| 行带×列带均值热图 + 行 L2 范数（全部降采样）。"""
    import torch
    payload = _wt_conn_payload()
    tmap = _wt_tmap(payload)
    nm = req.name.strip()
    if nm not in tmap:
        raise HTTPException(404, f'tensor 不存在: {nm}')
    ck = ('d', nm)
    hit = _WT_T_CACHE.get(ck)
    if hit:
        return hit
    tmeta = tmap[nm]
    bands, rnorms, sampled_any = [], [], False
    fp = Path(payload['model_path']) / tmeta['file']
    for _s, t, smp in _wt_blocks(fp, nm, tmeta):
        sampled_any = sampled_any or smp
        if t.dim() <= 1:
            bands.append(t.abs())
            rnorms.append(float(t.norm()))
        else:
            bands.append(t.abs().mean(dim=0))
            rnorms.extend(torch.norm(t, dim=1).tolist())
    rows = len(bands)
    heat = [_downsample(b.tolist(), _WT_HEAT_COLS) for b in bands]
    flat = [x for row in heat for x in row]
    st = _wt_scalar_stats(payload, tmap, nm)
    data = {'name': nm, 'shape': tmeta.get('shape'), 'dtype': tmeta.get('dtype'),
            'stats': st,
            'heatmap': {'rows': rows, 'cols': (_WT_HEAT_COLS if rows else 0), 'data': flat},
            'rownorm': {'n': len(rnorms), 'data': _downsample(rnorms, _WT_RN_POINTS), 'sampled': sampled_any},
            'block': '32×64 行块（超限均匀抽样）'}
    _WT_T_CACHE[ck] = data
    if len(_WT_T_CACHE) > 40:
        _WT_T_CACHE.pop(next(iter(_WT_T_CACHE)))
    return data


# ---------- 新闻（S5） ----------

def _fetch_arxiv() -> List[dict]:
    import urllib.request
    req = urllib.request.Request(ARXIV_URL, headers={'User-Agent': 'ai2050-dist/1.0'})
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = resp.read()
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(data)
    items = []
    for e in root.findall('a:entry', ns)[:10]:
        title = (e.findtext('a:title', default='', namespaces=ns) or '').strip().replace('\n', ' ')
        summary = (e.findtext('a:summary', default='', namespaces=ns) or '').strip().replace('\n', ' ')
        published = (e.findtext('a:published', default='', namespaces=ns) or '')[:10]
        link = ''
        for l in e.findall('a:link', ns):
            link = l.attrib.get('href', link)
        items.append({'date': published, 'title': title[:120], 'p': summary[:220],
                      'src': 'arxiv.org', 'tag': '论文', 'url': link})
    return items


@router.get('/news')
def news() -> dict:
    cache = _DATA_DIR / 'news_cache.json'
    if not NEWS_OFFLINE and cache.is_file():
        try:
            c = json.loads(cache.read_text(encoding='utf-8'))
            # 失败兜底（builtin-fallback）只短缓存，避免一次抓取抖动放大成 1 小时旧数据
            ttl = NEWS_FALLBACK_TTL if 'fallback' in str(c.get('source', '')) else NEWS_TTL
            if time.time() - c.get('fetched_at', 0) < ttl and c.get('items'):
                return {'source': c.get('source', 'cache'), 'items': c['items']}
        except Exception:
            pass
    items: List[dict] = []
    source = 'builtin'
    if not NEWS_OFFLINE:
        try:
            items = _fetch_arxiv()
            source = 'arxiv'
        except Exception:
            items = []
    if not items:
        items = NEWS_FALLBACK
        source = 'builtin-fallback'
    try:
        cache.write_text(json.dumps({'fetched_at': time.time(), 'source': source,
                                     'items': items}, ensure_ascii=False), encoding='utf-8')
    except Exception:
        pass
    return {'source': source, 'items': items}


# ---------- 研发 · 自动测试（autotest.v1）：被测模型 × 模式族 → 一键完整流程 ----------
# 作业在中心所在机器上顺序执行（scripts/run_language_family.py → node_agent → runner_v3），
# 驱动经 AI2050_AUTOTEST_CENTER（缺省 http://127.0.0.1:5001；独立运行时为本服务端口）回连中心。

class AutoModelReq(BaseModel):
    model_id: str
    kind: str = 'local'
    label: str = ''
    model_path: str = ''
    model_rev: str = ''
    dtype: str = 'bf16'
    gpu: str = ''
    note: str = ''


class AutoProbeReq(BaseModel):
    model_path: str


class AutoBatchReq(BaseModel):
    families: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)


_AUTOTEST = None
_AUTOTEST_LOCK = threading.Lock()


def _autotest_center() -> str:
    return os.environ.get('AI2050_AUTOTEST_CENTER', 'http://127.0.0.1:5001').rstrip('/')


def _at():
    global _AUTOTEST
    _proto()
    with _AUTOTEST_LOCK:
        if _AUTOTEST is None:
            _AUTOTEST = _autotest.AutoTest(
                _DATA_DIR, _WORKSPACE, _autotest_center, ADMIN_TOKEN,
                allow_synthetic=lambda: 'synthetic' in _pipeline.mergeable_kinds())
        return _AUTOTEST


def _library_families() -> List[str]:
    out = []
    for path in sorted(_GRAMMAR_DIR.glob('*.json')):
        try:
            fid = json.loads(path.read_text(encoding='utf-8')).get('family_id')
        except (OSError, json.JSONDecodeError):
            continue
        if fid:
            out.append(fid)
    return out


def _model_dump(b) -> dict:
    return b.model_dump() if hasattr(b, 'model_dump') else (b.dict() if hasattr(b, 'dict') else dict(vars(b)))


@router.get('/autotest/state')
def autotest_state(limit: int = Query(default=100)) -> dict:
    return _at().state(limit=max(1, min(limit, 500)))


@router.post('/autotest/models')
def autotest_model_upsert(b: AutoModelReq, x_admin_token: str = Header(default='')) -> dict:
    _auth_admin(x_admin_token)
    try:
        return {'ok': True, 'model': _at().upsert_model(_model_dump(b))}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.delete('/autotest/models/{model_id}')
def autotest_model_delete(model_id: str, x_admin_token: str = Header(default='')) -> dict:
    _auth_admin(x_admin_token)
    if not _at().delete_model(model_id):
        raise HTTPException(404, f'未登记的模型 {model_id}')
    return {'ok': True}


@router.post('/autotest/probe')
def autotest_probe(b: AutoProbeReq) -> dict:
    """检查本机权重目录（只读 config.json 与文件列表，不加载权重）。"""
    _proto()
    return _autotest.model_dir_info(b.model_path)


@router.post('/autotest/batches')
def autotest_submit(b: AutoBatchReq, x_admin_token: str = Header(default='')) -> dict:
    _auth_admin(x_admin_token)
    try:
        return _at().submit(b.families, b.models, b.options or {}, _library_families())
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post('/autotest/jobs/{job_id}/cancel')
def autotest_cancel(job_id: str, x_admin_token: str = Header(default='')) -> dict:
    _auth_admin(x_admin_token)
    try:
        job = _at().cancel(job_id)
    except KeyError:
        raise HTTPException(404, f'没有作业 {job_id}')
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return {'ok': True, 'status': job['status']}


@router.post('/autotest/jobs/{job_id}/retry')
def autotest_retry(job_id: str, x_admin_token: str = Header(default='')) -> dict:
    _auth_admin(x_admin_token)
    try:
        return _at().retry(job_id)
    except KeyError:
        raise HTTPException(404, f'没有作业 {job_id}')
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.post('/autotest/clear')
def autotest_clear(x_admin_token: str = Header(default='')) -> dict:
    _auth_admin(x_admin_token)
    return {'ok': True, 'removed': _at().clear_finished()}


@router.get('/autotest/jobs/{job_id}/log')
def autotest_log(job_id: str, tail: int = Query(default=400)) -> dict:
    at = _at()
    try:
        at.log_path(job_id)
    except KeyError:
        raise HTTPException(404, f'没有作业 {job_id}')
    return {'job_id': job_id, 'lines': at.log_tail(job_id, max(1, min(tail, 5000)))}


# ---------- 账号（accounts.v1）：客户端注册 / 登录；管理员账号的会话可代替管理令牌；节点可绑定到账号 ----------

class AuthRegisterReq(BaseModel):
    username: str
    password: str
    display_name: str = ''
    email: str = ''
    client: str = ''


class AuthLoginReq(BaseModel):
    username: str
    password: str
    client: str = ''


class AuthPasswordReq(BaseModel):
    old_password: str
    new_password: str


class AuthCreateUserReq(BaseModel):
    username: str
    password: str
    display_name: str = ''
    email: str = ''
    role: str = 'member'


class AuthUpdateUserReq(BaseModel):
    role: Optional[str] = None
    disabled: Optional[bool] = None
    display_name: Optional[str] = None
    email: Optional[str] = None


class AuthResetPasswordReq(BaseModel):
    new_password: str


class AuthDeleteUserReq(BaseModel):
    confirm: str


class AuthRegistrationReq(BaseModel):
    policy: str


def _auth_call(fn, *args, **kwargs):
    """在写锁内执行账号操作，并把 AuthError 转成 HTTP 错误。"""
    conn = _db()            # 先在锁外取连接：首次访问会建库并播种模板，播种自身要拿 _db_lock（锁不可重入）
    try:
        with _db_lock:
            return fn(conn, *args, **kwargs)
    except _accounts.AuthError as exc:
        raise HTTPException(exc.status, exc.message)


@router.get('/auth/status')
def auth_status(authorization: str = Header(default='')) -> dict:
    """客户端启动时调用：是否已有账号、是否开放注册；带有效会话时附上当前用户。"""
    db = _db()
    out = _accounts.status(db)
    out['user'] = _accounts.session_user(db, _accounts.bearer(authorization))
    return out


@router.post('/auth/register')
def auth_register(b: AuthRegisterReq) -> dict:
    return _auth_call(_accounts.register, b.username, b.password, display_name=b.display_name,
                      email=b.email, client=b.client)


@router.post('/auth/login')
def auth_login(b: AuthLoginReq) -> dict:
    return _auth_call(_accounts.login, b.username, b.password, client=b.client)


@router.get('/auth/me')
def auth_me(authorization: str = Header(default='')) -> dict:
    return {'user': _auth_call(_accounts.require_user, authorization)}


@router.post('/auth/logout')
def auth_logout(authorization: str = Header(default='')) -> dict:
    tok = _accounts.bearer(authorization)
    if tok:
        _auth_call(_accounts.logout, tok)
    return {'ok': True}


@router.post('/auth/password')
def auth_password(b: AuthPasswordReq, authorization: str = Header(default='')) -> dict:
    u = _auth_call(_accounts.require_user, authorization)
    _auth_call(_accounts.change_password, u['user_id'], b.old_password, b.new_password,
               keep_token=_accounts.bearer(authorization))
    return {'ok': True, 'note': '密码已修改；其他设备上的登录已失效。'}


@router.get('/auth/users')
def auth_users(authorization: str = Header(default='')) -> dict:
    _auth_call(_accounts.require_admin, authorization)
    info = _auth_call(_accounts.registration_info)
    return {'users': _auth_call(_accounts.list_users), 'registration': info['policy'], 'registration_source': info['source']}


@router.post('/auth/users')
def auth_create_user(b: AuthCreateUserReq, authorization: str = Header(default='')) -> dict:
    _auth_call(_accounts.require_admin, authorization)
    return {'user': _auth_call(_accounts.create_user, b.username, b.password, display_name=b.display_name,
                               email=b.email, role=b.role, by_admin=True)}


@router.post('/auth/users/{user_id}')
def auth_update_user(user_id: str, b: AuthUpdateUserReq, authorization: str = Header(default='')) -> dict:
    actor = _auth_call(_accounts.require_admin, authorization)
    return {'user': _auth_call(_accounts.update_user, actor, user_id, role=b.role, disabled=b.disabled,
                               display_name=b.display_name, email=b.email)}


@router.post('/auth/users/{user_id}/password')
def auth_reset_user_password(user_id: str, b: AuthResetPasswordReq, authorization: str = Header(default='')) -> dict:
    actor = _auth_call(_accounts.require_admin, authorization)
    return {'user': _auth_call(_accounts.admin_reset_password, actor, user_id, b.new_password),
            'note': '密码已重置；该账号的全部登录已失效，锁定已解除。'}


@router.post('/auth/users/{user_id}/logout')
def auth_revoke_user_sessions(user_id: str, authorization: str = Header(default='')) -> dict:
    actor = _auth_call(_accounts.require_admin, authorization)
    return _auth_call(_accounts.revoke_sessions, actor, user_id)


@router.post('/auth/users/{user_id}/delete')
def auth_delete_user(user_id: str, b: AuthDeleteUserReq, authorization: str = Header(default='')) -> dict:
    actor = _auth_call(_accounts.require_admin, authorization)
    return _auth_call(_accounts.delete_user, actor, user_id, b.confirm)


@router.post('/auth/registration')
def auth_set_registration(b: AuthRegistrationReq, authorization: str = Header(default='')) -> dict:
    actor = _auth_call(_accounts.require_admin, authorization)
    return _auth_call(_accounts.set_registration_policy, actor, b.policy)


@router.post('/auth/node-code')
def auth_node_code(authorization: str = Header(default='')) -> dict:
    """一次性节点绑定码（30 分钟）：node_agent register --bind-code <码>，节点即归属当前账号。"""
    u = _auth_call(_accounts.require_user, authorization)
    return _auth_call(_accounts.new_bind_code, u['user_id'])


@router.get('/auth/my-nodes')
def auth_my_nodes(authorization: str = Header(default='')) -> dict:
    u = _auth_call(_accounts.require_user, authorization)
    db = _db()
    rows = db.execute('SELECT node_id,name,gpu,model,dtype,created_at,last_seen FROM nodes WHERE owner_user_id=? '
                      'ORDER BY created_at DESC', (u['user_id'],)).fetchall()
    nodes = []
    for r in rows:
        n = db.execute('SELECT COUNT(*) FROM results WHERE node_id=?', (r['node_id'],)).fetchone()[0]
        nodes.append({**dict(r), 'results': n, 'online': bool(r['last_seen'] and r['last_seen'] >= time.time() - NODE_ONLINE_WINDOW)})
    return {'nodes': nodes}


# ---------- 题库 · 研究包 · 语言谱图（catalog.v1 / respack.v1 / atlas.v1） ----------
# 任何用户（客户端）都可以：读题库与状态 → 领取分片在本地跑 → 回传；下载研究包到本地 3D 观察与复算；
# 读跨模式族的语言谱图，并让配置好的 AI 模型据谱图摘要提出可检验假设（假设只是「未检验」，必须变成新模板运行）。

class AtlasAskReq(BaseModel):
    model: str = ''
    question: str = ''


def _library_specs() -> List[dict]:
    out = []
    for path in sorted(_GRAMMAR_DIR.glob('*.json')):
        try:
            out.append(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _zip_response(data: bytes, filename: str) -> Response:
    return Response(content=data, media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename="{filename}"'})


@router.get('/catalog')
def api_catalog() -> dict:
    """题库：类别 → 题目（模式族）→ 冻结版本、按模型的测试状态、领取命令、研究包下载地址；贡献者列表。"""
    _proto()
    return _pipeline.catalog(_db(), _DATA_DIR, _library_specs(), node_online_window=NODE_ONLINE_WINDOW,
                             center_hint=os.environ.get('AI2050_PUBLIC_URL', 'http://<center>:5010'))


@router.get('/export/{tm_id}')
def api_export(tm_id: str, bucket: str = Query(default=''), gate: bool = Query(default=True),
               items: bool = Query(default=True)):
    """研究包 respack.v1（zip）：corpus / agg / analysis（含 3D 坐标）/ suffstats / 逐条向量。"""
    _proto()
    try:
        data, name = _pipeline.export_pack(_db(), _DATA_DIR, tm_id, bucket=bucket or None, gate=gate, include_items=items)
    except KeyError:
        raise HTTPException(404, f'没有模板 {tm_id}')
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    return _zip_response(data, name)


def _atlas(gate: bool, model: str = '') -> dict:
    payload = _pipeline.atlas_payload(_db(), _DATA_DIR, gate=gate)
    if model:
        if model not in payload['models']:
            raise HTTPException(404, f'谱图中没有模型桶 {model}（可用：{", ".join(payload["models"]) or "无"}）')
        payload = {**payload, 'graphs': [g for g in payload['graphs'] if g['model'] == model]}
    return payload


@router.get('/atlas')
def api_atlas(model: str = Query(default=''), gate: bool = Query(default=True)) -> dict:
    """语言谱图 atlas.v1：每个模型的特征节点（含 3D 坐标）、跨族边、共享编码轴、概念假设、跨模型一致性、规律陈述。"""
    _proto()
    return _atlas(gate, model)


@router.get('/atlas/digest')
def api_atlas_digest(model: str = Query(default=''), question: str = Query(default='')) -> dict:
    from shared.dist_protocol import atlas as _atlas_mod
    _proto()
    payload = _atlas(True)
    return {'markdown': _atlas_mod.digest(payload, model or None),
            'prompt': _atlas_mod.ai_prompt(payload, question, model or None), 'system': _atlas_mod.AI_SYSTEM}


@router.get('/atlas/export')
def api_atlas_export(model: str = Query(default=''), gate: bool = Query(default=True)):
    _proto()
    data, name = _pipeline.atlas_pack(_atlas(gate), model or None)
    return _zip_response(data, name)


def _ai_complete(prompt: str, system: str):
    """用「研发 · Agent 工作流」配置的主模型回答。返回 (文本, 不可用原因)。"""
    try:
        import asyncio
        from server.ai_rnd_service import ModelConfig, call_ai_model, get_session  # type: ignore
    except Exception as exc:  # noqa: BLE001 - 独立部署的中心服务没有 AI 研发模块
        return None, f'AI 研发服务不可用（{type(exc).__name__}）：中心服务需挂载在后端 server.server 上'
    cfg = dict((getattr(get_session(), 'config', None) or {}).get('master_model') or {})
    if not str(cfg.get('api_key') or '').strip():
        return None, '未配置主模型 API Key（研发 · Agent 工作流 → 模型配置）'
    try:
        text = asyncio.run(call_ai_model(ModelConfig(**cfg), prompt, system))
    except Exception as exc:  # noqa: BLE001
        return None, f'AI 调用失败：{type(exc).__name__}: {exc}'
    if str(text).startswith('[ERROR]'):
        return None, str(text)
    return str(text), ''


@router.post('/atlas/interpret')
def api_atlas_interpret(b: AtlasAskReq, x_admin_token: str = Header(default='')) -> dict:
    """AI 解读谱图：主模型读谱图摘要，输出 ≤5 个可检验假设（JSON）。不可用时返回提示词，便于复制到任意 AI。
    结果记为「未检验」并保存到 <AI2050_DIST_DIR>/atlas/hypotheses.jsonl，不进入证据。"""
    from shared.dist_protocol import atlas as _atlas_mod
    _auth_admin(x_admin_token)
    _proto()
    payload = _atlas(True, b.model)
    prompt = _atlas_mod.ai_prompt(payload, b.question, b.model or None)
    text, reason = _ai_complete(prompt, _atlas_mod.AI_SYSTEM)
    if text is None:
        return {'available': False, 'reason': reason, 'prompt': prompt, 'system': _atlas_mod.AI_SYSTEM}
    parsed = _atlas_mod.parse_hypotheses(text)
    rec = {'id': 'H-' + uuid.uuid4().hex[:8], 'created_at': time.time(), 'model': b.model, 'question': b.question,
           'atlas_models': payload['models'], **parsed}
    path = _DATA_DIR / 'atlas' / 'hypotheses.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return {'available': True, **rec}


@router.get('/atlas/hypotheses')
def api_atlas_hypotheses(limit: int = Query(default=50)) -> dict:
    path = _DATA_DIR / 'atlas' / 'hypotheses.jsonl'
    rows = []
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines()[-max(1, min(limit, 500)):]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {'schema': 'atlas_hypotheses.v1', 'items': rows[::-1],
            'note': 'AI 提出的假设全部是「未检验」：只有写成新模板（文法 + concepts 预注册）并运行通过后才有证据。'}


# ---------- 独立部署入口（CentOS: python -m server.distributed_service） ----------

app = None
if __name__ == '__main__':
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(description='AI2050 distributed platform service')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=int(os.environ.get('AI2050_DIST_PORT', '5010')))
    args = ap.parse_args()
    os.environ.setdefault('AI2050_AUTOTEST_CENTER', f'http://127.0.0.1:{args.port}')
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    app = FastAPI(title='AI2050 Distributed Research Platform')
    app.add_middleware(
        CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
    app.include_router(router)

    @app.get('/')
    def _root() -> dict:
        return {'service': 'ai2050-distributed', 'version': ROUTER_VERSION}

    uvicorn.run(app, host=args.host, port=args.port)
