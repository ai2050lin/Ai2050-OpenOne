# -*- coding: utf-8 -*-
"""中心服务 · 开放平台路由（platform.v1）：公开配置、研究进展快照、贡献榜与参与者一键包。

  GET /api/platform/config          公开：站点配置（标题、注册方式、开放的任务类型、所需模型、数据许可）
  GET /api/platform/overview        公开：研究进展快照（模式族、谱图、神经元普查与解释、节点、贡献者），后台定时刷新
  GET /api/platform/snapshot/{name} 公开：重计算端点的缓存副本（pipeline_overview / scoreboard / atlas / coverage / catalog）
  GET /api/platform/contributors    公开：按账号统计的贡献（只显示显示名）
  GET /api/platform/datapack        公开：客户端离线数据包（deploy/datapack.py，后台子进程生成）
  GET /api/platform/datapack/extra/{id}  公开：附加包（模式族实验响应、单元卡）
  GET /api/platform/kit             参与者一键包（zip）：节点程序、启动脚本、配置（不带绑定码）
  GET /api/auth/kit                 登录用户的一键包：内嵌一次性节点绑定码（7 天），节点自动归属该账号
  GET /api/auth/my-contributions    登录用户：我的节点与贡献

公开范围：模式族合并结果、谱图、神经元卡、干预与解释检验结果、预注册文件。研究记录（AGI_CLAUDE_MEMO 等）与原生全场不经本服务公开。
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request, Response

import distributed_service as D

router = APIRouter(prefix='/api/platform', tags=['platform'])
# 需要会话令牌的两个端点放在 /api/auth 下：客户端只给 /api/auth/* 请求带 Authorization 头
auth_router = APIRouter(prefix='/api/auth', tags=['platform'])
# 语言谱图（lang_atlas.v1）：语言树状态、语言项详情、模型骨架、3D 叠加。都只读后台子进程写好的 lang_atlas 快照
lang_router = APIRouter(prefix='/api', tags=['platform'])

_REPO = Path(__file__).resolve().parent.parent
def _read_version() -> str:
    """版本号的唯一来源：仓库 / 发布包根目录的 VERSION 文件（scripts/build_center_release.py 每次打包时递增）。"""
    try:
        return (_REPO / 'VERSION').read_text(encoding='utf-8').strip() or '0.0.0'
    except OSError:
        return '0.0.0'


PLATFORM_VERSION = _read_version()
KIT_BIND_TTL = 7 * 86400
# 站点配置没有登记有效模型时的缺省参与模型（快照与 Hugging Face 版本已逐文件核对一致，见 Phase 255 / 1.0.x 部署记录）
DEFAULT_MODELS = [{'name': 'qwen3-4b', 'hf_repo': 'Qwen/Qwen3-4B', 'hf_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
                   'snapshot_sha256': '6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad', 'min_vram_gib': 12}]

_snap: Dict[str, Dict[str, Any]] = {}          # 只放网站进程里算的轻量快照（overview）；重计算快照由子进程写文件
_snap_lock = threading.Lock()
_worker = None
_file_cache: Dict[str, Any] = {}


def _builders() -> Dict[str, Callable[[], Any]]:
    return {
        'pipeline_overview': D.pipeline_overview,
        'coverage': D.coverage_matrix,
        'catalog': D.api_catalog,
        'atlas': lambda: D.api_atlas(model='', gate=True),
        'scoreboard': D.scoreboard,
        'overview': build_overview,
    }


def _refresh_one(name: str) -> None:
    t0 = time.time()
    try:
        data, err = _builders()[name](), None
    except Exception as exc:  # 快照失败时保留上一次结果并记录错误
        data, err = None, f'{type(exc).__name__}: {exc}'[:300]
    with _snap_lock:
        prev = _snap.get(name) or {}
        _snap[name] = {'data': data if data is not None else prev.get('data'), 'computed_at': time.time() if data is not None else prev.get('computed_at'),
                       'seconds': round(time.time() - t0, 1), 'error': err}


def _read_file_snapshot(name: str):
    """读后台子进程写好的快照文件（按修改时间缓存解析结果）。"""
    path = Path(D._DATA_DIR) / 'snapshots' / f'{name}.json'
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _file_cache.get(name)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return hit[1] if hit else None
    _file_cache[name] = (mtime, doc)
    return doc


def start_refresher() -> None:
    """拉起后台计算子进程（deploy/snapshot_worker.py，低优先级）：快照、规律分析预热、普查自动合并。
    计算放在独立进程里，网站进程不会被拖慢；结果写文件并带数据指纹，重启后直接可用。"""
    global _worker
    if os.environ.get('AI2050_SNAPSHOT_DISABLE') == '1':
        return
    if _worker is not None and _worker.poll() is None:
        return
    import subprocess
    import sys
    env = dict(os.environ, AI2050_DIST_DIR=str(D._DATA_DIR), PYTHONIOENCODING='utf-8')
    _worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve().parent / 'snapshot_worker.py')], env=env)


def stop_refresher() -> None:
    if _worker is not None and _worker.poll() is None:
        _worker.terminate()


def snapshot(name: str) -> Dict[str, Any]:
    if name not in _builders():
        raise HTTPException(404, f'unknown snapshot {name}')
    if name == 'overview':                               # 轻量：网站进程内计算，60 秒有效
        with _snap_lock:
            s = _snap.get(name)
        if s is None or s.get('data') is None or time.time() - (s.get('computed_at') or 0) > 60:
            _refresh_one(name)
            with _snap_lock:
                s = _snap.get(name)
        return {'pending': False, 'computed_at': s.get('computed_at'), 'seconds': s.get('seconds'), 'error': s.get('error'), 'data': s.get('data')}
    doc = _read_file_snapshot(name)
    if doc is None or doc.get('data') is None:
        start_refresher()
        return {'pending': True, 'note': '快照正在后台计算（首次启动可能需要几分钟），请稍后刷新。', 'data': None}
    return {'pending': False, 'computed_at': doc.get('computed_at'), 'seconds': doc.get('seconds'), 'error': doc.get('error'),
            'fingerprint': doc.get('fingerprint'), 'data': doc.get('data')}


# ---------- 配置 ----------

def _config_file() -> Dict[str, Any]:
    p = Path(os.environ.get('AI2050_PLATFORM_CONFIG', str(D._DATA_DIR / 'platform_config.json')))
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def public_url(request: Optional[Request] = None) -> str:
    env = os.environ.get('AI2050_PUBLIC_URL', '').rstrip('/')
    if env:
        return env
    if request is not None:
        return str(request.base_url).rstrip('/')
    return ''


def census_open_jobs() -> list:
    try:
        import census_service as CS
        out = []
        for r in CS.db().execute("SELECT job_id FROM census_jobs WHERE status='open' ORDER BY created_at").fetchall():
            p = CS.job_progress(r['job_id'])
            out.append({k: p[k] for k in ('job_id', 'model', 'n_shards', 'counts', 'tokens_per_slice')})
        return out
    except Exception:
        return []


@router.get('/config')
def config(request: Request) -> dict:
    from shared.dist_protocol import accounts as A
    cfg = _config_file()
    jobs = census_open_jobs()
    models, seen = [], set()
    listed = [m for m in (cfg.get('models') or []) if len(str(m.get('snapshot_sha256') or '')) == 64]
    for m in (listed or DEFAULT_MODELS) + [j['model'] for j in jobs]:   # 参与所需模型：配置文件登记（缺省 DEFAULT_MODELS）+ 开放的普查作业
        snap = str(m.get('snapshot_sha256') or '')
        if len(snap) == 64 and all(ch in '0123456789abcdef' for ch in snap) and snap not in seen:   # 占位值不下发
            seen.add(m['snapshot_sha256'])
            models.append({k: m.get(k) for k in ('name', 'hf_repo', 'hf_revision', 'snapshot_sha256', 'min_vram_gib')})
    return {
        'schema': 'platform_config.v1', 'version': PLATFORM_VERSION,
        'title': cfg.get('title', 'AI2050 · 开源 LLM 逆向研究平台'),
        'tagline': cfg.get('tagline', '一起绘制大语言模型的语言编码图谱：看进展、学方法、用自己的 GPU 跑一个分片。'),
        'public_url': public_url(request),
        'registration': A.registration_policy(),
        'task_kinds': cfg.get('task_kinds', ['family_test', 'census_shard']),
        'contact': cfg.get('contact', ''),
        'repo_url': cfg.get('repo_url', 'https://github.com/ai2050lin/Ai2050-OpenOne'),
        'license': {'code': 'Apache-2.0', 'data': 'CC BY 4.0'},
        'census_jobs': jobs,
        'models': models,
        'sandbox_synthetic': os.environ.get('AI2050_DIST_ALLOW_SYNTHETIC') == '1',
        'datapack': _datapack_meta(),                    # 客户端离线数据包的版本与大小；客户端据此提示「有新数据」
        'local_tools': os.environ.get('AI2050_LOCAL_TOOLS') == '1',   # 本机中心：工作台显示自动测试与模型资产
        'public_scope': ['模式族合并结果', '语言谱图', '神经元单元卡（普查摘要、两极片段、解释等级）', '干预与解释检验结果', '预注册文件'],
        'not_public': ['研究记录（memo）', '节点本地原生全场'],
    }


# ---------- 进展快照 ----------

def _families() -> Dict[str, Any]:
    db = D._db()
    tpl = db.execute("SELECT COUNT(*) AS n FROM templates WHERE status='open'").fetchone()['n']
    res = {r['kind']: r['n'] for r in db.execute('SELECT kind,COUNT(*) AS n FROM results GROUP BY kind')}
    models = [r['m'] for r in db.execute("SELECT DISTINCT model_id AS m FROM results WHERE kind='real' AND model_id<>''")]
    return {'templates_open': tpl, 'results_real': res.get('real', 0), 'results_synthetic': res.get('synthetic', 0), 'models': models}


def _neurons() -> Dict[str, Any]:
    try:
        from server import neuron_census_service as NS
        runs = NS.census_runs()
        explained: Dict[str, int] = {}
        for recs in NS.expl_index().values():
            g = NS.explain_axis(recs).split('（')[0].split('：')[0]
            explained[g] = explained.get(g, 0) + 1
        return {'census_runs': [{k: r.get(k) for k in ('model_key', 'run', 'n_tokens', 'n_layers')} for r in runs],
                'explanation_grades': explained}
    except Exception as exc:
        return {'census_runs': [], 'explanation_grades': {}, 'error': f'{type(exc).__name__}: {exc}'[:200]}


def contributors() -> list:
    db = D._db()
    rows = db.execute('SELECT user_id, COALESCE(NULLIF(display_name,\'\'), username) AS name, created_at FROM users WHERE disabled=0').fetchall()
    out = []
    for u in rows:
        nodes = [r['node_id'] for r in db.execute('SELECT node_id FROM nodes WHERE owner_user_id=?', (u['user_id'],))]
        fam = census = tokens = 0
        if nodes:
            q = ','.join('?' * len(nodes))
            fam = db.execute(f"SELECT COUNT(*) FROM results WHERE kind='real' AND node_id IN ({q})", nodes).fetchone()[0]
            try:
                r = db.execute(f"SELECT COUNT(*) AS c, COALESCE(SUM(n_tokens),0) AS t FROM census_results "
                               f"WHERE status='accepted' AND kind='real' AND node_id IN ({q})", nodes).fetchone()
                census, tokens = r['c'], r['t']
            except Exception:
                pass
        if fam or census:
            out.append({'name': u['name'], 'nodes': len(nodes), 'family_results': fam, 'census_shards': census, 'census_tokens': tokens})
    return sorted(out, key=lambda x: (-(x['family_results'] + x['census_shards']), x['name']))


def build_overview() -> Dict[str, Any]:
    db = D._db()
    now = time.time()
    online = db.execute('SELECT COUNT(*) AS n FROM nodes WHERE last_seen>=?', (now - D.NODE_ONLINE_WINDOW,)).fetchone()['n']
    total = db.execute('SELECT COUNT(*) AS n FROM nodes').fetchone()['n']
    users = db.execute('SELECT COUNT(*) AS n FROM users WHERE disabled=0').fetchone()['n']
    census = []
    try:
        import census_service as CS
        census = [CS.job_progress(r['job_id']) for r in CS.db().execute('SELECT job_id FROM census_jobs ORDER BY created_at DESC')]
    except Exception:
        pass
    atlas = (_read_file_snapshot('atlas') or {}).get('data')
    atlas_sum = None
    if atlas:
        atlas_sum = {'models': atlas.get('models', []),
                     'n_nodes': sum(len(g.get('nodes', [])) for g in atlas.get('graphs', [])),
                     'n_edges': sum(len(g.get('edges', [])) for g in atlas.get('graphs', [])),
                     'n_axes': sum(len(g.get('axes', g.get('shared_axes', []))) for g in atlas.get('graphs', []))}
    return {'schema': 'platform_overview.v1', 'server_time': now, 'families': _families(), 'atlas': atlas_sum,
            'neurons': _neurons(), 'census_jobs': census,
            'people': {'users': users, 'nodes_total': total, 'nodes_online': online, 'contributors': len(contributors())},
            'note': '各项计数来自不同数据表，彼此独立，不构成「测试 → 特征 → 机制」的转化率；任务完成与审计通过都不表示证据晋级。'}


@router.get('/overview')
def overview() -> dict:
    s = snapshot('overview')
    return {**(s['data'] or {}), 'computed_at': s['computed_at']}


def _datapack_meta():
    import datapack as DP
    m = DP.read_meta(D._DATA_DIR)
    if not m:
        return None
    return {**{k: m.get(k) for k in ('version', 'created_at', 'bytes', 'raw_bytes', 'n_entries')},
            'extras': DP.extras_meta(D._DATA_DIR)}         # 附加包（模式族实验响应、单元卡）：按需手动下载


@router.get('/datapack')
def datapack_download():
    """客户端离线数据包（gzip JSON，datapack.v1）。由后台计算子进程在数据变化时重建；还没生成时返回 404。"""
    from fastapi.responses import FileResponse
    p = Path(D._DATA_DIR) / 'datapack' / 'datapack.json.gz'
    if not p.is_file():
        raise HTTPException(404, '数据包还在后台生成，请稍后再试')
    meta = _datapack_meta() or {}
    return FileResponse(p, media_type='application/octet-stream', filename='ai2050-datapack.json.gz',
                        headers={'Content-Encoding': 'identity', 'X-Datapack-Version': str(meta.get('version') or ''),
                                 'Cache-Control': 'no-cache'})


@router.get('/datapack/extra/{extra_id}')
def datapack_extra_download(extra_id: str):
    """附加包（gzip JSON，datapack_extra.v1）：tm:<模式族> 或 cards:<模型>。清单见站点配置 datapack.extras。"""
    from fastapi.responses import FileResponse
    import datapack as DP
    info = DP.read_extras(D._DATA_DIR).get(extra_id)
    p = DP.extra_file(D._DATA_DIR, extra_id)
    if not info or not p.is_file():
        raise HTTPException(404, '没有这个附加包，或还在后台生成')
    return FileResponse(p, media_type='application/octet-stream', filename=p.name,
                        headers={'Content-Encoding': 'identity', 'X-Datapack-Version': str(info.get('version') or ''),
                                 'Cache-Control': 'no-cache'})


@router.get('/snapshot/{name}')
def snapshot_ep(name: str) -> dict:
    if name == 'overview':
        raise HTTPException(404, 'use /api/platform/overview')
    return snapshot(name)


@router.get('/contributors')
def contributors_ep() -> dict:
    return {'schema': 'contributors.v1', 'items': contributors(),
            'note': '只统计已绑定账号的节点；模式族结果按 real 计数，普查分片按通过校验的计数。'}


@auth_router.get('/my-contributions')
def my(authorization: str = Header(default='')) -> dict:
    from shared.dist_protocol import accounts as A
    u = D._auth_call(A.require_user, authorization)
    db = D._db()
    nodes = [dict(r) for r in db.execute('SELECT node_id,name,gpu,model,created_at,last_seen FROM nodes WHERE owner_user_id=? '
                                         'ORDER BY created_at DESC', (u['user_id'],))]
    for n in nodes:
        n['online'] = bool(n['last_seen'] and n['last_seen'] >= time.time() - D.NODE_ONLINE_WINDOW)
        n['family_results'] = db.execute("SELECT COUNT(*) FROM results WHERE node_id=?", (n['node_id'],)).fetchone()[0]
        try:
            n['census'] = [dict(r) for r in db.execute(
                'SELECT job_id,shard,status,n_tokens,created_at FROM census_results WHERE node_id=? ORDER BY created_at DESC LIMIT 50',
                (n['node_id'],))]
        except Exception:
            n['census'] = []
    return {'user': {k: u.get(k) for k in ('username', 'display_name', 'role')}, 'nodes': nodes}


# ---------- 参与者一键包 ----------

KIT_README = '''AI2050 参与者一键包
====================

这个包让你的电脑成为 AI2050 开放研究平台的一个计算节点：从中心领取冻结的测试任务，在本机 GPU 上运行，上传可合并的统计结果。

1. Windows：双击 start_windows.bat；Linux / macOS：在终端运行 bash start_linux.sh
2. 第一次运行会创建 Python 虚拟环境并安装 torch、transformers 等依赖（约 3 GB，需要网络）。
3. 程序会检查 GPU 与显存，询问本机模型目录；没有模型时可以按提示从 Hugging Face 下载作业指定的版本。
4. 之后程序自动循环：领取任务 → 核对任务包 sha256 → 运行 → 上传 → 中心校验。关掉窗口即可停止，未完成的任务会在租约到期后重新派发。

请不要修改任务包或节点程序：任务包按 sha256 核对；中心逐项校验结果，并把一部分分片派给另一个账号的节点复算比较。
被修改过的结果会被拒收，不会进入合并。

隐私：上传的只有统计量（直方图、均值、极值位置等），不包含你电脑上的任何其他文件。原生全场只保存在本机（~/.ai2050）。
配置：config.json（中心地址、绑定码、任务类型、显卡编号）。绑定码一次性有效，节点注册后即失效。
'''

START_BAT = r'''@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv\Scripts\python.exe (
  echo [AI2050] 创建 Python 虚拟环境 ...
  %PY% -m venv .venv || (echo 需要 Python 3.10 以上：https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 || (echo 依赖安装失败 & pause & exit /b 1)
)
.venv\Scripts\python.exe node_agent.py participate --config config.json
pause
'''

START_SH = '''#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
  echo "[AI2050] 创建 Python 虚拟环境 ..."
  $PY -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python node_agent.py participate --config config.json
'''

KIT_REQUIREMENTS = '''numpy>=1.26
torch>=2.6
transformers>=4.56
safetensors>=0.4
accelerate>=1.0
huggingface_hub>=0.34
'''


def build_kit(server: str, bind_code: str = '', user: str = '') -> bytes:
    cfg = {'schema': 'participant_config.v1', 'server': server, 'bind_code': bind_code, 'account': user,
           'task_kinds': _config_file().get('task_kinds', ['family_test', 'census_shard']),
           'device': 'cuda', 'model_dirs': {}, 'models_root': '', 'node_name': '',
           'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'note': 'bind_code 一次性有效（7 天内）；model_dirs 按作业的模型快照填写本机目录，留空时程序会询问。'}
    here = Path(D.__file__).resolve().parent
    agent = (here / 'node_agent.py').read_bytes()
    service = (here / 'node_service.py').read_bytes()       # node_agent.py serve：本机节点服务（桌面客户端控制）
    proto = here.parent / 'shared' / 'dist_protocol'           # Goodfire 神经元测试（本机）：runner 与统计模块
    probe_files = {n: (proto / n).read_bytes() for n in ('goodfire_runner.py', 'goodfire_probe.py')}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        def add(name: str, data, mode: int = 0o644):
            info = zipfile.ZipInfo(f'ai2050-node/{name}', date_time=time.localtime()[:6])
            info.external_attr = (mode & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data if isinstance(data, bytes) else data.encode('utf-8'))
        add('node_agent.py', agent)
        add('node_service.py', service)
        for n, data in probe_files.items():
            add(n, data)
        add('config.json', json.dumps(cfg, ensure_ascii=False, indent=1))
        add('requirements.txt', KIT_REQUIREMENTS)
        add('README.txt', KIT_README)
        add('start_windows.bat', START_BAT.replace('\n', '\r\n'))
        add('start_linux.sh', START_SH, 0o755)
    return buf.getvalue()


def _kit_response(data: bytes) -> Response:
    return Response(data, media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename="ai2050-node.zip"', 'Cache-Control': 'no-store'})


@router.get('/kit')
def kit(request: Request) -> Response:
    """匿名一键包：不带绑定码，节点的贡献不署名（之后也可以在 config.json 填绑定码重新注册）。"""
    return _kit_response(build_kit(public_url(request)))


@auth_router.get('/kit')
def kit_bound(request: Request, authorization: str = Header(default='')) -> Response:
    """登录用户的一键包：内嵌一次性绑定码，节点注册后自动归属该账号（令牌只走请求头，不放进 URL）。"""
    from shared.dist_protocol import accounts as A
    u = D._auth_call(A.require_user, authorization)
    bind = D._auth_call(A.new_bind_code, u['user_id'], ttl=KIT_BIND_TTL)['code']
    return _kit_response(build_kit(public_url(request), bind, u.get('display_name') or u.get('username') or ''))


# ---------- 语言谱图（lang_atlas.v1，pieces.py 生成，后台子进程写 snapshots/lang_atlas.json） ----------

def _lang_atlas() -> Optional[Dict[str, Any]]:
    doc = _read_file_snapshot('lang_atlas')
    return doc if doc and doc.get('data') else None


def _lang_pending() -> Dict[str, Any]:
    start_refresher()
    return {'pending': True, 'note': '语言谱图正在后台计算（规律分析预热完成后生成），请稍后刷新。'}


@lang_router.get('/lang/tree')
def lang_tree_ep() -> dict:
    """语言树（类型 → 类别 → 语言项）与每个语言项在各模型上的两路线状态。"""
    doc = _lang_atlas()
    if doc is None:
        return _lang_pending()
    return {**doc['data']['tree'], 'pending': False, 'computed_at': doc.get('computed_at')}


@lang_router.get('/lang/item/{item_id}')
def lang_item_ep(item_id: str) -> dict:
    """一个语言项：全部 piece、逐条规则陈述、判据与回查来源。"""
    doc = _lang_atlas()
    if doc is None:
        return _lang_pending()
    item = doc['data']['items'].get(item_id)
    if item is None:
        raise HTTPException(404, f'没有语言项 {item_id}')
    return {**item, 'pending': False, 'computed_at': doc.get('computed_at')}


@lang_router.get('/model/skeleton')
def model_skeleton_ep(model_key: str = '') -> dict:
    """模型骨架（层数、宽度等，来自结果摘要与普查清单）；不给 model_key 时列出全部模型。"""
    doc = _lang_atlas()
    if doc is None:
        return _lang_pending()
    sk = doc['data']['skeletons']
    if not model_key:
        return {'schema': 'model_skeletons.v1', 'pending': False, 'models': [sk[m] for m in sorted(sk)]}
    if model_key not in sk:
        raise HTTPException(404, f'没有模型 {model_key} 的骨架')
    return {**sk[model_key], 'pending': False}


@lang_router.get('/space/overlay')
def space_overlay_ep(model_key: str) -> dict:
    """3D 默认场景的叠加数据：该模型上每个语言项的逐层曲线（路线 T）。"""
    doc = _lang_atlas()
    if doc is None:
        return _lang_pending()
    ov = doc['data']['overlays'].get(model_key)
    if ov is None:
        raise HTTPException(404, f'没有模型 {model_key} 的叠加数据')
    return {**ov, 'pending': False, 'computed_at': doc.get('computed_at')}
