#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI2050 本机节点服务（contributor_client.v1 P1 / P3 模型下载，设计见 design/contributor_client_v1.md）

桌面客户端的界面只和这个服务通信；只有它访问中心，并且只做 注册 / 领取 / 续租 / 上传 / 读取站点配置，带退避。
任务循环复用 node_agent.py（普查分片、模式族分片），任务包照旧按 sha256 核对后才运行。

启动：python node_agent.py serve [--port 5030] [--synthetic]
     （或 python node_service.py ...；仅依赖标准库，GPU 运行时由 runner 所用的解释器提供）

本机接口安全：只监听 127.0.0.1；Host 头必须是本机地址（防 DNS 重绑定）；每次启动生成随机令牌，
写入 <home>/service.json，请求必须带 X-Local-Token；带 Origin 的请求默认拒绝（--allow-origin 显式放行）。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import node_agent as A  # noqa: E402

SCHEMA = 'node_service.v1'
VERSION = '1'
DEFAULT_PORT = 5030
TASK_KINDS = ('census_shard', 'family_test')
DEFAULT_LIMITS = {'hours': '', 'pause_on_battery': True, 'yield_gpu_util': 30, 'max_vram_gib': 0, 'min_free_disk_gib': 5}
HF_ENDPOINTS = ('https://huggingface.co', 'https://hf-mirror.com')   # 模型下载来源只能二选一；文件最终按模型快照核对
DEFAULT_CONFIG = {'server': '', 'task_kinds': list(TASK_KINDS), 'device': 'cuda:0', 'node_name': '',
                  'models_root': '', 'model_dirs': {}, 'hf_endpoint': HF_ENDPOINTS[0], 'limits': dict(DEFAULT_LIMITS)}
CONFIG_TTL = 1800.0          # 站点配置最多每 30 分钟读一次
GATE_RECHECK = 60.0          # 资源门不满足时每分钟复查一次
VRAM_FACTOR = 1.2            # 显存需求 ≈ 权重文件总大小 × 1.2（近似）


# ---------- 纯函数：解析、校验、门、退避（单元测试覆盖） ----------

def parse_nvidia_smi(text: str) -> list:
    """nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version
    --format=csv,noheader,nounits 的输出 → GPU 列表。"""
    out = []
    for line in (text or '').splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6 or not parts[0].isdigit():
            continue

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None
        total, used, util = num(parts[2]), num(parts[3]), num(parts[4])
        out.append({'index': int(parts[0]), 'name': parts[1],
                    'vram_gib': round(total / 1024, 1) if total is not None else None,
                    'used_gib': round(used / 1024, 1) if used is not None else None,
                    'util': util, 'driver': parts[5]})
    return out


def parse_window(spec: str):
    """'22:00-08:00' → (起, 止) 分钟数；空串 → None（全天）。格式不对抛 ValueError。"""
    spec = (spec or '').strip()
    if not spec:
        return None
    try:
        a, b = spec.split('-')
        mins = []
        for x in (a, b):
            h, m = x.strip().split(':')
            h, m = int(h), int(m)
            if not (0 <= h < 24 and 0 <= m < 60):
                raise ValueError
            mins.append(h * 60 + m)
    except ValueError:
        raise ValueError(f'时间窗格式应为 HH:MM-HH:MM：{spec}') from None
    if mins[0] == mins[1]:
        raise ValueError('时间窗起止不能相同（全天请留空）')
    return tuple(mins)


def in_window(spec: str, now_struct) -> bool:
    w = parse_window(spec)
    if w is None:
        return True
    t = now_struct.tm_hour * 60 + now_struct.tm_min
    a, b = w
    return a <= t < b if a < b else (t >= a or t < b)     # 跨午夜


def validate_config(cur: dict, patch: dict) -> dict:
    """在 cur 上合并 patch，只接受白名单字段；返回新配置，不合法抛 ValueError。"""
    allowed = set(DEFAULT_CONFIG)
    bad = set(patch) - allowed
    if bad:
        raise ValueError(f'不支持的字段：{", ".join(sorted(bad))}')
    new = json.loads(json.dumps(cur))
    for k, v in patch.items():
        if k == 'limits':
            if not isinstance(v, dict):
                raise ValueError('limits 必须是对象')
            badl = set(v) - set(DEFAULT_LIMITS)
            if badl:
                raise ValueError(f'不支持的限制：{", ".join(sorted(badl))}')
            lim = {**DEFAULT_LIMITS, **new.get('limits', {}), **v}
            parse_window(lim['hours'])
            if not isinstance(lim['pause_on_battery'], bool):
                raise ValueError('pause_on_battery 必须是 true / false')
            for key, hi in (('yield_gpu_util', 100), ('max_vram_gib', 1024), ('min_free_disk_gib', 100000)):
                x = lim[key]
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not (0 <= x <= hi):
                    raise ValueError(f'{key} 应为 0–{hi} 的数')
            new['limits'] = lim
        elif k == 'task_kinds':
            if not isinstance(v, list) or not set(v) <= set(TASK_KINDS):
                raise ValueError(f'task_kinds 只能取 {", ".join(TASK_KINDS)}')
            new[k] = list(dict.fromkeys(v))
        elif k == 'server':
            v = str(v).strip().rstrip('/')
            if v and not (v.startswith('https://') or v.startswith('http://')):
                raise ValueError('server 必须是 http(s):// 地址')
            new[k] = v
        elif k == 'device':
            if not (v == 'cpu' or (isinstance(v, str) and (v == 'cuda' or (v.startswith('cuda:') and v[5:].isdigit())))):
                raise ValueError('device 只能是 cuda、cuda:N 或 cpu')
            new[k] = v
        elif k == 'hf_endpoint':
            if v not in HF_ENDPOINTS:
                raise ValueError(f'hf_endpoint 只能取 {", ".join(HF_ENDPOINTS)}')
            new[k] = v
        elif k == 'models_root':
            v = str(v).strip()
            if v and not Path(v).is_absolute():
                raise ValueError('models_root 必须是绝对路径')
            new[k] = v
        elif k == 'model_dirs':
            if not isinstance(v, dict) or not all(isinstance(a, str) and isinstance(b, str) for a, b in v.items()):
                raise ValueError('model_dirs 必须是 {快照: 目录}')
            new[k] = v
        else:
            new[k] = str(v)
    return new


def gpu_index(device: str) -> int:
    return int(device.split(':')[1]) if device.startswith('cuda:') else 0


def gate_reasons(cfg: dict, *, now_struct, gpus: list, battery, disk_free_gib, synthetic: bool = False) -> list:
    """领取任务前的资源门；返回不满足的原因（空 = 可以领取）。"""
    lim = {**DEFAULT_LIMITS, **(cfg.get('limits') or {})}
    out = []
    if not in_window(lim['hours'], now_struct):
        out.append(f"不在允许的时间段（{lim['hours']}）")
    if lim['pause_on_battery'] and battery is True:
        out.append('正在使用电池供电')
    if disk_free_gib is not None and disk_free_gib < lim['min_free_disk_gib']:
        out.append(f"磁盘剩余 {disk_free_gib:.1f} GiB，低于 {lim['min_free_disk_gib']} GiB")
    if not synthetic and cfg.get('device', 'cuda').startswith('cuda'):
        g = next((x for x in gpus if x['index'] == gpu_index(cfg.get('device', 'cuda'))), None)
        if g is None:
            out.append('没有检测到 NVIDIA GPU（需要 NVIDIA 显卡与驱动）')
        elif lim['yield_gpu_util'] and g.get('util') is not None and g['util'] > lim['yield_gpu_util']:
            out.append(f"GPU 正被其他程序使用（利用率 {g['util']:.0f}% > {lim['yield_gpu_util']}%）")
    return out


def model_fits(weight_gib, cfg: dict, gpus: list) -> bool:
    """按权重大小估计显存需求，超过上限（max_vram_gib，0 = 显卡总显存）的模型不领取。"""
    if not weight_gib:
        return True
    lim = (cfg.get('limits') or {}).get('max_vram_gib') or 0
    if not lim:
        g = next((x for x in gpus if x['index'] == gpu_index(cfg.get('device', 'cuda'))), None)
        lim = (g or {}).get('vram_gib') or 0
    return not lim or weight_gib * VRAM_FACTOR <= lim


class Waiting(Exception):
    """本轮不能领取（例如没有可用模型）：状态记为等待并给出原因，按资源门的节奏复查。"""


class Backoff:
    """没有任务：60 s 起 ×2，上限 15 min；出错：30 s 起 ×2，上限 30 min；均带 ±20 % 抖动。"""
    PLAN = {'idle': (60.0, 900.0), 'error': (30.0, 1800.0)}

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.n = {'idle': 0, 'error': 0}

    def next(self, kind: str) -> float:
        base, cap = self.PLAN[kind]
        other = 'error' if kind == 'idle' else 'idle'
        self.n[other] = 0
        d = min(cap, base * (2 ** self.n[kind]))
        self.n[kind] += 1
        return d * self.rng.uniform(0.8, 1.2)

    def reset(self) -> None:
        self.n = {'idle': 0, 'error': 0}


class Cancelled(Exception):
    """下载被取消：已下载的部分保留在 .part 文件里，下次从断点继续。"""


_REPO_RE = re.compile(r'^[A-Za-z0-9][\w.-]*/[\w.-]+$')
_REV_RE = re.compile(r'^[\w.-]{1,64}$')


def safe_filename(name: str) -> bool:
    """只下载仓库顶层的普通文件名（model_snapshot 只看顶层）；拒绝路径分隔、上级目录、盘符与隐藏文件。"""
    return bool(name) and len(name) <= 200 and not re.search(r'[\\/:*?"<>|\x00-\x1f]', name) \
        and not name.startswith('.') and name not in ('..',)


def hf_file_list(endpoint: str, repo: str, revision: str, timeout: float = 60) -> list:
    """Hugging Face 仓库某个版本的顶层文件：[{name, size, sha256（LFS 文件才有）}]。来源由配置白名单限定（HF_ENDPOINTS）。"""
    if not _REPO_RE.match(repo or '') or not _REV_RE.match(revision or 'main'):
        raise ValueError(f'仓库或版本不合法：{repo}@{revision}')
    url = f"{endpoint}/api/models/{quote(repo, safe='/')}/revision/{quote(revision or 'main', safe='')}?blobs=true"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'读取文件列表失败：HTTP {e.code}' + ('（该仓库需要登录授权，暂不支持）' if e.code in (401, 403) else '')) from e
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f'连不上 {endpoint}：{e}') from e
    out = []
    for s in data.get('siblings') or []:
        name = s.get('rfilename') or ''
        if '/' in name or not safe_filename(name):
            continue
        lfs = s.get('lfs') or {}
        size = s.get('size') if s.get('size') is not None else lfs.get('size')
        out.append({'name': name, 'size': int(size) if size is not None else None, 'sha256': str(lfs.get('sha256') or '').lower()})
    return out


def download_file(url: str, dest: Path, *, size=None, sha256: str = '', on_bytes=None, cancel=None, timeout: float = 60) -> str:
    """断点续传下载到 dest（先写 dest.part）；返回文件 sha256。已知 sha256 时不一致则删除并报错。"""
    part = dest.with_name(dest.name + '.part')
    have = part.stat().st_size if part.exists() else 0
    if size is not None and have > size:
        part.unlink()
        have = 0
    h = hashlib.sha256()
    if have:
        with part.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b''):
                h.update(chunk)
        if on_bytes:
            on_bytes(have)
    if size is None or have < size:
        req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if have and r.status != 206:              # 服务器不支持续传：从头开始
                if on_bytes:
                    on_bytes(-have)
                have, h = 0, hashlib.sha256()
                part.unlink()
            with part.open('ab') as fh:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
                    h.update(b)
                    if on_bytes:
                        on_bytes(len(b))
    got_size = part.stat().st_size
    if size is not None and got_size != size:
        raise RuntimeError(f'{dest.name} 大小不符：{got_size} ≠ {size}（网络中断？重试会从断点继续）')
    digest = h.hexdigest()
    if sha256 and digest != sha256:
        part.unlink()
        raise RuntimeError(f'{dest.name} 的 sha256 与仓库记录不一致，已删除')
    part.replace(dest)
    return digest


def seed_snapshot_cache(entries: list) -> None:
    """下载时已算出的 sha256 写入 model_snapshot 的缓存（键与 node_agent.model_snapshot 相同），核对时不必再读一遍大文件。"""
    p = A._home() / 'snapshot_cache.json'
    try:
        cache = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        cache = {}
    for path, digest in entries:
        st = Path(path).stat()
        cache[f'{Path(path).resolve()}|{st.st_size}|{int(st.st_mtime)}'] = digest
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache), encoding='utf-8')


def weight_gib(model_dir: str) -> float:
    root = Path(model_dir)
    if not root.is_dir():
        return 0.0
    n = sum(p.stat().st_size for pat in ('*.safetensors', '*.bin', '*.pt', '*.gguf') for p in root.glob(pat) if p.is_file())
    return round(n / 2**30, 2)


# ---------- 系统探测（Windows + NVIDIA） ----------

def query_gpus() -> list:
    exe = shutil.which('nvidia-smi')
    if not exe and os.name == 'nt' and Path(r'C:\Windows\System32\nvidia-smi.exe').is_file():
        exe = r'C:\Windows\System32\nvidia-smi.exe'
    if not exe:
        return []
    try:
        r = subprocess.run([exe, '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia_smi(r.stdout) if r.returncode == 0 else []


def on_battery():
    """True = 正在用电池；False = 接电源；None = 无法判断或没有电池。"""
    if os.name != 'nt':
        return None
    import ctypes

    class SPS(ctypes.Structure):
        _fields_ = [('ACLineStatus', ctypes.c_ubyte), ('BatteryFlag', ctypes.c_ubyte), ('BatteryLifePercent', ctypes.c_ubyte),
                    ('SystemStatusFlag', ctypes.c_ubyte), ('BatteryLifeTime', ctypes.c_ulong), ('BatteryFullLifeTime', ctypes.c_ulong)]
    s = SPS()
    try:
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return None
    except (AttributeError, OSError):
        return None
    if s.BatteryFlag == 128:            # 没有电池
        return None
    return {0: True, 1: False}.get(s.ACLineStatus)


RUNTIME_PROBE = r'''
import json
d = {}
try:
    import torch
    d['torch'] = torch.__version__
    d['cuda_available'] = bool(torch.cuda.is_available())
    d['cuda'] = torch.version.cuda
    if d['cuda_available']:
        d['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception as e:
    d['torch_error'] = str(e)[:200]
try:
    import transformers
    d['transformers'] = transformers.__version__
except Exception as e:
    d['transformers_error'] = str(e)[:200]
print(json.dumps(d))
'''


def probe_runtime(python: str) -> dict:
    """runner 所用解释器里的计算组件（torch / CUDA / transformers）；在子进程中检测，服务本身不导入 torch。"""
    try:
        r = subprocess.run([python, '-c', RUNTIME_PROBE], capture_output=True, text=True, timeout=120,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        d = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else {}
        if not d:
            d = {'error': (r.stderr or '')[-300:]}
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        d = {'error': str(e)[:300]}
    d['python'] = python
    d['ok'] = bool(d.get('cuda_available')) and 'transformers' in d
    return d


def disk_free_gib(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        return round(shutil.disk_usage(str(path)).free / 2**30, 1)
    except OSError:
        return None


# ---------- 服务 ----------

class NodeService:
    def __init__(self, home: Path, *, synthetic: bool = False, python: str = sys.executable, probes=None):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.synthetic = synthetic
        self.python = python
        self.probes = {'gpus': query_gpus, 'battery': on_battery, 'runtime': probe_runtime,
                       'disk': disk_free_gib, **(probes or {})}
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.quit = threading.Event()
        self.logs = collections.deque(maxlen=2000)
        self.seq = 0
        self.state = 'stopped'
        self.reasons: list = []
        self.current = None
        self.next_at = None
        self.last_error = ''
        self.counts = {'done': 0, 'failed': 0, 'rejected': 0}
        self.backoff = Backoff()
        self.checks: dict = {}           # 快照 → {'status': checking|ok|mismatch|missing|downloading|error|cancelled, 'dir', ...}
        self.cancels: dict = {}          # 快照 → 下载取消事件
        self.pcfg, self.pcfg_at = None, 0.0
        self.runtime, self.runtime_at = None, 0.0
        self.cfg = self._load_config()
        self.thread = None
        A.LOG_HOOKS.append(self._on_log)
        A.PROGRESS_HOOKS.append(self._on_progress)
        A.RESULT_HOOKS.append(self._on_result)

    # --- 持久化 ---
    @property
    def config_path(self) -> Path:
        return self.home / 'config.json'

    @property
    def history_path(self) -> Path:
        return self.home / 'history.jsonl'

    def _load_config(self) -> dict:
        try:
            raw = json.loads(self.config_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raw = {}
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        try:
            return validate_config(cfg, {k: v for k, v in raw.items() if k in DEFAULT_CONFIG})
        except ValueError:
            return cfg

    def _save_config(self) -> None:
        A._save_config(self.config_path, self.cfg)

    def creds(self) -> dict:
        p = A.creds_path(self.home)
        try:
            c = json.loads(p.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
        return c if c.get('server') == self.cfg.get('server') and c.get('node_token') else {}

    # --- 钩子 ---
    def _on_log(self, msg: str) -> None:
        with self.lock:
            self.seq += 1
            self.logs.append({'seq': self.seq, 't': time.time(), 'msg': str(msg)})

    def _on_progress(self, p: dict) -> None:
        with self.lock:
            if self.current is not None:
                self.current['progress'] = p

    def _on_result(self, r: dict) -> None:
        r = {'t': time.time(), **r}
        with self.lock:
            st = r.get('status')
            self.counts['done' if st in ('done', 'accepted') else 'rejected' if st == 'rejected' else 'failed'] += 1
            try:
                with self.history_path.open('a', encoding='utf-8') as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + '\n')
            except OSError:
                pass

    # --- 中心 ---
    def platform_config(self, force: bool = False) -> dict:
        server = self.cfg.get('server')
        if not server:
            return {}
        if force or self.pcfg is None or time.time() - self.pcfg_at > CONFIG_TTL:
            try:
                _, self.pcfg = A.http_json(server + '/api/platform/config', 'GET')
            except RuntimeError as e:
                A.log(f'读取站点配置失败：{e}')
                if self.pcfg is None:
                    raise
            self.pcfg_at = time.time()
        return self.pcfg or {}

    def bind(self, bind_code: str = '') -> dict:
        server = self.cfg.get('server')
        if not server:
            raise ValueError('请先在配置里填写中心地址（server）')
        import platform as _pf
        gpus = self.probes['gpus']()
        _, resp = A.http_json(server + '/api/nodes/register', 'POST', {
            'name': self.cfg.get('node_name') or _pf.node() or 'node', 'gpu': (gpus[0]['name'] if gpus else ''),
            'model': '', 'dtype': 'bf16', 'bind_code': bind_code or ''})
        A.creds_path(self.home).write_text(json.dumps({'server': server, **resp}, ensure_ascii=False, indent=1), encoding='utf-8')
        A.log(f"节点已注册 {resp['node_id']}" + ('，已绑定到你的账号' if resp.get('owner_bound') else '（未绑定账号：贡献不会署名）'))
        return {'node_id': resp['node_id'], 'owner_bound': bool(resp.get('owner_bound'))}

    # --- 模型 ---
    def models(self) -> list:
        pc = self.pcfg or {}
        out = []
        for m in pc.get('models') or []:
            snap = m.get('snapshot_sha256')
            if not snap:
                continue
            d = self.cfg['model_dirs'].get(snap, '')
            chk = self.checks.get(snap) or ({'status': 'ok', 'dir': d} if d else {'status': 'missing'})
            out.append({'snapshot': snap, 'name': m.get('name') or snap[:12], 'hf_repo': m.get('hf_repo') or '',
                        'hf_revision': m.get('hf_revision') or '', 'dir': chk.get('dir', d), 'status': chk['status'],
                        'weight_gib': weight_gib(chk.get('dir') or d) if (chk.get('dir') or d) and chk['status'] != 'downloading' else None,
                        'min_vram_gib': m.get('min_vram_gib'), 'error': chk.get('error', ''),
                        'progress': {k: chk.get(k) for k in ('done', 'total', 'file', 'started')} if chk['status'] == 'downloading' else None})
        return out

    def check_model(self, snapshot: str, model_dir: str) -> dict:
        if not snapshot or not model_dir:
            raise ValueError('需要 snapshot 与 dir')
        if not Path(model_dir).is_dir():
            raise ValueError(f'目录不存在：{model_dir}')
        with self.lock:
            if (self.checks.get(snapshot) or {}).get('status') == 'checking':
                return self.checks[snapshot]
            if snapshot in self.cancels:
                raise ValueError('该模型正在下载')
            self.checks[snapshot] = {'status': 'checking', 'dir': model_dir}
        threading.Thread(target=self._verify, args=(snapshot, model_dir), daemon=True).start()
        return self.checks[snapshot]

    def _verify(self, snapshot: str, model_dir: str) -> None:
        try:
            got = A.model_snapshot(model_dir)['sha256']
        except OSError as e:
            got, err = '', str(e)
        else:
            err = ''
        with self.lock:
            if got == snapshot:
                self.checks[snapshot] = {'status': 'ok', 'dir': model_dir}
                self.cfg['model_dirs'][snapshot] = model_dir
                self._save_config()
                A.log(f'模型快照一致 ✓ {model_dir}')
            else:
                self.checks[snapshot] = {'status': 'mismatch', 'dir': model_dir, 'got': got, 'error': err}
                A.log(f'模型目录 {model_dir} 的快照与要求不一致，不能用于该模型的任务')

    def models_root(self) -> Path:
        return Path(self.cfg.get('models_root') or (self.home / 'models'))

    def download_model(self, snapshot: str) -> dict:
        """按中心站点配置的 hf_repo@hf_revision 下载到 models_root/<名称>，完成后按快照核对。断点续传。"""
        m = next((x for x in (self.pcfg or {}).get('models') or [] if x.get('snapshot_sha256') == snapshot), None)
        if not m:
            raise ValueError('中心没有列出这个模型快照（先「读取中心要求」）')
        if not m.get('hf_repo'):
            raise ValueError('中心没有给出这个模型的下载来源（hf_repo）')
        name = re.sub(r'[^\w.-]', '_', m.get('name') or snapshot[:16])
        target = self.models_root() / name
        endpoint = self.cfg.get('hf_endpoint') or HF_ENDPOINTS[0]
        with self.lock:
            if snapshot in self.cancels or (self.checks.get(snapshot) or {}).get('status') == 'checking':
                return self.checks[snapshot]
            cancel = threading.Event()
            self.cancels[snapshot] = cancel
            self.checks[snapshot] = {'status': 'downloading', 'dir': str(target), 'done': 0, 'total': None, 'file': '',
                                     'started': time.time()}
        threading.Thread(target=self._download, args=(snapshot, m, target, endpoint, cancel), daemon=True).start()
        return self.checks[snapshot]

    def _download(self, snapshot: str, m: dict, target: Path, endpoint: str, cancel) -> None:
        st = self.checks[snapshot]
        repo, rev = m['hf_repo'], m.get('hf_revision') or 'main'
        try:
            files = hf_file_list(endpoint, repo, rev)
            if not files:
                raise RuntimeError('仓库里没有文件')
            target.mkdir(parents=True, exist_ok=True)
            todo = [f for f in files if not ((target / f['name']).is_file() and f['size'] is not None
                                             and (target / f['name']).stat().st_size == f['size'])]
            total = sum(f['size'] or 0 for f in files)
            done = total - sum(f['size'] or 0 for f in todo)
            need = sum((f['size'] or 0) - ((target / (f['name'] + '.part')).stat().st_size
                                           if (target / (f['name'] + '.part')).is_file() else 0) for f in todo)
            free = self.probes['disk'](target)
            floor = (self.cfg.get('limits') or {}).get('min_free_disk_gib', DEFAULT_LIMITS['min_free_disk_gib'])
            if free is not None and free - need / 2**30 < floor:
                raise RuntimeError(f'磁盘空间不够：还需 {need / 2**30:.1f} GiB，{target.anchor or target} 剩余 {free:.1f} GiB'
                                   f'（至少保留 {floor} GiB）。可以在「模型目录」换一个盘')
            with self.lock:
                st.update(total=total, done=done)
            A.log(f'下载模型 {m.get("name") or repo}：{repo}@{rev[:12]} → {target}（{len(todo)}/{len(files)} 个文件，'
                  f'{need / 2**30:.1f} GiB，来源 {endpoint}）')

            def on_bytes(n):
                with self.lock:
                    st['done'] += n
            seeded = []
            for f in todo:
                with self.lock:
                    st['file'] = f['name']
                url = f"{endpoint}/{quote(repo, safe='/')}/resolve/{quote(rev, safe='')}/{quote(f['name'])}"
                digest = download_file(url, target / f['name'], size=f['size'], sha256=f['sha256'], on_bytes=on_bytes, cancel=cancel)
                seeded.append((target / f['name'], digest))
            if seeded:
                seed_snapshot_cache(seeded)
        except Cancelled:
            with self.lock:
                self.checks[snapshot] = {'status': 'cancelled', 'dir': str(target)}
                self.cancels.pop(snapshot, None)
            A.log('已取消下载（已下载的部分保留，下次继续）')
            return
        except (RuntimeError, ValueError, OSError) as e:
            with self.lock:
                self.checks[snapshot] = {'status': 'error', 'dir': str(target), 'error': str(e)[:300]}
                self.cancels.pop(snapshot, None)
            A.log(f'下载失败：{e}')
            return
        with self.lock:
            self.checks[snapshot] = {'status': 'checking', 'dir': str(target)}
            self.cancels.pop(snapshot, None)
        A.log('下载完成，核对模型快照…')
        self._verify(snapshot, str(target))

    def cancel_download(self, snapshot: str) -> dict:
        with self.lock:
            ev = self.cancels.get(snapshot)
        if ev is None:
            raise ValueError('这个模型没有在下载')
        ev.set()
        return self.checks.get(snapshot) or {}

    # --- 设备 ---
    def device(self, refresh: bool = False) -> dict:
        if refresh or self.runtime is None or time.time() - self.runtime_at > 600:
            self.runtime, self.runtime_at = self.probes['runtime'](self.python), time.time()
        return {'gpus': self.probes['gpus'](), 'runtime': self.runtime, 'battery': self.probes['battery'](),
                'disk_free_gib': self.probes['disk'](self.home), 'home': str(self.home), 'synthetic': self.synthetic,
                'platform': sys.platform}

    def gates(self) -> list:
        return gate_reasons(self.cfg, now_struct=time.localtime(), gpus=[] if self.synthetic else self.probes['gpus'](),
                            battery=self.probes['battery'](), disk_free_gib=self.probes['disk'](self.home),
                            synthetic=self.synthetic)

    # --- 控制 ---
    def status(self) -> dict:
        c = self.creds()
        with self.lock:
            return {'schema': SCHEMA, 'state': self.state, 'reasons': list(self.reasons), 'current': self.current,
                    'next_at': self.next_at, 'last_error': self.last_error, 'counts': dict(self.counts),
                    'server': self.cfg.get('server'), 'bound': bool(c), 'node_id': c.get('node_id'),
                    'owner_bound': bool(c.get('owner_bound')), 'synthetic': self.synthetic, 'log_seq': self.seq}

    def start(self) -> dict:
        if not self.creds():
            raise PermissionError('节点还没有注册：先绑定账号（或以匿名方式注册）')
        with self.lock:
            if self.state in ('stopped', 'error'):
                self.state = 'running'
                self.backoff.reset()
                self.last_error = ''
                A.log('开始贡献：' + ('沙盒联调（synthetic）' if self.synthetic else '按资源限制领取任务'))
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, daemon=True, name='node-loop')
                self.thread.start()
        self.wake.set()
        return self.status()

    def pause(self) -> dict:
        with self.lock:
            if self.state != 'stopped':
                self.state = 'paused'
                A.log('已暂停：当前任务做完后不再领取' if self.current else '已暂停')
        self.wake.set()
        return self.status()

    def resume(self) -> dict:
        with self.lock:
            if self.state == 'paused':
                self.state = 'running'
                self.backoff.reset()
                A.log('继续领取任务')
        self.wake.set()
        return self.status()

    def stop(self) -> dict:
        with self.lock:
            self.state = 'stopped'
            self.next_at = None
        n = A.abort_running()
        A.log('已停止' + (f'（结束了 {n} 个运行中的任务，中心会重新派发）' if n else ''))
        self.wake.set()
        return self.status()

    def shutdown(self) -> None:
        self.stop()
        self.quit.set()
        self.wake.set()
        for hooks, f in ((A.LOG_HOOKS, self._on_log), (A.PROGRESS_HOOKS, self._on_progress), (A.RESULT_HOOKS, self._on_result)):
            if f in hooks:
                hooks.remove(f)

    def update_config(self, patch: dict) -> dict:
        with self.lock:
            new = validate_config(self.cfg, patch)
            if new.get('server') != self.cfg.get('server'):
                self.pcfg, self.pcfg_at = None, 0.0
            self.cfg = new
            self._save_config()
        self.wake.set()
        return self.cfg

    # --- 任务循环 ---
    def _sleep(self, seconds: float) -> None:
        with self.lock:
            self.next_at = time.time() + seconds
        self.wake.clear()
        self.wake.wait(seconds)
        with self.lock:
            self.next_at = None

    def _loop(self) -> None:
        while not self.quit.is_set():
            with self.lock:
                st = self.state
            if st in ('stopped', 'paused', 'error'):
                self.wake.clear()
                self.wake.wait(5)
                continue
            reasons = self.gates()
            with self.lock:
                if self.state != st:
                    continue
                self.reasons = reasons
                self.state = 'waiting' if reasons else 'running'
            if reasons:
                self._sleep(GATE_RECHECK)
                continue
            try:
                did = self.round()
            except Waiting as w:
                with self.lock:
                    if self.state == 'running':
                        self.state, self.reasons = 'waiting', [str(w)]
                self._sleep(GATE_RECHECK)
                continue
            except Exception as e:  # noqa: BLE001 — 网络、中心 5xx/429 等：退避后再试
                with self.lock:
                    self.last_error = str(e)[:300]
                A.log(f'本轮出错：{e}')
                self._sleep(self.backoff.next('error'))
                continue
            if did:
                self.backoff.reset()
                continue
            d = self.backoff.next('idle')
            A.log(f'暂时没有任务，{int(d)} 秒后再看')
            self._sleep(d)

    def ready_models(self, gpus: list) -> dict:
        offered = {m['snapshot_sha256']: m for m in (self.pcfg or {}).get('models') or [] if m.get('snapshot_sha256')}
        out = {}
        for snap, d in self.cfg['model_dirs'].items():
            if snap in offered and Path(d).is_dir() and model_fits(weight_gib(d), self.cfg, gpus):
                out[snap] = (d, offered[snap])
        return out

    def round(self) -> bool:
        """一轮：按任务类型尝试领取一个任务并执行；返回是否执行了任务。"""
        creds = self.creds()
        if not creds:
            raise RuntimeError('节点凭据与当前中心地址不符，请重新绑定')
        pc = self.platform_config()
        offered = pc.get('task_kinds') or []
        kinds = [k for k in self.cfg['task_kinds'] if k in offered]
        server, token = creds['server'], creds['node_token']
        work = self.home / 'work'
        if self.synthetic:
            if 'census_shard' in kinds:
                return self._run('census_shard', None, lambda: A.run_census_shard(server, token, work / 'census', '', 'cpu',
                                                                               'synthetic', self.python))
            return False
        gpus = self.probes['gpus']()
        ready = self.ready_models(gpus)
        if not ready:
            raise Waiting('没有可用的模型：为中心要求的模型快照指定本机目录并核对（或模型超过显存上限）')
        device = self.cfg.get('device') or 'cuda:0'
        if 'census_shard' in kinds:
            for snap, (d, m) in ready.items():
                if self._run('census_shard', m, lambda: A.run_census_shard(server, token, work / 'census', d, device, 'real',
                                                                         self.python, snapshot=snap)):
                    return True
        if 'family_test' in kinds:
            g = next((x for x in gpus if x['index'] == gpu_index(device)), {})
            for snap, (d, m) in ready.items():
                ns = argparse.Namespace(python=self.python, run_timeout=6 * 3600.0, model_path=d,
                                        model_id=(m.get('name') or '').lower(), model_rev=A._model_rev(d),
                                        tokenizer_sha256='', gpu=g.get('name', ''), dtype='bf16', seed=0, tm_id='', server=server)
                if self._run('family_test', m, lambda: A.run_one(ns, creds, work / 'tasks')):
                    return True
        return False

    def _run(self, kind: str, model, fn) -> bool:
        with self.lock:
            if self.state != 'running':
                return False
            self.current = {'kind': kind, 'model': (model or {}).get('name') or ('synthetic' if self.synthetic else ''),
                            'started': time.time(), 'progress': None}
        try:
            return bool(fn())
        finally:
            with self.lock:
                self.current = None

    def history(self, limit: int = 200) -> list:
        try:
            lines = self.history_path.read_text(encoding='utf-8').splitlines()
        except OSError:
            return []
        out = []
        for ln in lines[-limit:]:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
        return out[::-1]

    def logs_after(self, after: int) -> list:
        with self.lock:
            return [x for x in self.logs if x['seq'] > after]


# ---------- 本机 HTTP 接口 ----------

def make_handler(svc: NodeService, token: str, port_ref: dict, allow_origins=()):
    class H(BaseHTTPRequestHandler):
        server_version = 'ai2050-node/' + VERSION

        def log_message(self, *a):          # 不把请求写进控制台
            pass

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            origin = self.headers.get('Origin')
            if origin and origin in allow_origins:
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _guard(self) -> bool:
            host = (self.headers.get('Host') or '').lower()
            port = port_ref['port']
            if host not in (f'127.0.0.1:{port}', f'localhost:{port}'):
                self._send(403, {'detail': 'host not allowed'})
                return False
            origin = self.headers.get('Origin')
            if origin and origin not in allow_origins:
                self._send(403, {'detail': 'origin not allowed'})
                return False
            if not secrets.compare_digest(self.headers.get('X-Local-Token') or '', token):
                self._send(401, {'detail': 'missing or wrong X-Local-Token'})
                return False
            return True

        def _body(self) -> dict:
            n = int(self.headers.get('Content-Length') or 0)
            if n > 1 << 20:
                raise ValueError('请求太大')
            raw = self.rfile.read(n) if n else b''
            d = json.loads(raw.decode('utf-8')) if raw else {}
            if not isinstance(d, dict):
                raise ValueError('请求体必须是 JSON 对象')
            return d

        def do_OPTIONS(self):  # noqa: N802 — 开发时放行的 Origin 预检
            origin = self.headers.get('Origin')
            if not origin or origin not in allow_origins:
                self._send(403, {'detail': 'origin not allowed'})
                return
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Local-Token')
            self.end_headers()

        def _dispatch(self, method: str) -> None:
            if not self._guard():
                return
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            route = (method, u.path.rstrip('/') or '/')
            try:
                if route == ('GET', '/status'):
                    return self._send(200, svc.status())
                if route == ('GET', '/device'):
                    return self._send(200, svc.device(refresh=q.get('refresh') == '1'))
                if route == ('GET', '/config'):
                    return self._send(200, svc.cfg)
                if route == ('PUT', '/config'):
                    return self._send(200, svc.update_config(self._body()))
                if route == ('GET', '/models'):
                    svc.platform_config(force=q.get('refresh') == '1')
                    root = svc.models_root()
                    return self._send(200, {'models': svc.models(), 'models_root': str(root), 'hf_endpoint': svc.cfg.get('hf_endpoint'),
                                            'hf_endpoints': list(HF_ENDPOINTS), 'models_free_gib': svc.probes['disk'](root)})
                if route == ('POST', '/models/download'):
                    return self._send(200, svc.download_model(str(self._body().get('snapshot') or '')))
                if route == ('POST', '/models/cancel'):
                    return self._send(200, svc.cancel_download(str(self._body().get('snapshot') or '')))
                if route == ('POST', '/models/check'):
                    b = self._body()
                    return self._send(200, svc.check_model(str(b.get('snapshot') or ''), str(b.get('dir') or '')))
                if route == ('POST', '/bind'):
                    return self._send(200, svc.bind(str(self._body().get('bind_code') or '')))
                if route == ('GET', '/logs'):
                    return self._send(200, {'items': svc.logs_after(int(q.get('after') or 0))})
                if route == ('GET', '/history'):
                    return self._send(200, {'items': svc.history(int(q.get('limit') or 200))})
                actions = {'/start': svc.start, '/pause': svc.pause, '/resume': svc.resume, '/stop': svc.stop}
                if method == 'POST' and route[1] in actions:
                    return self._send(200, actions[route[1]]())
                if route == ('POST', '/shutdown'):
                    self._send(200, {'ok': True})
                    threading.Thread(target=port_ref['shutdown'], daemon=True).start()
                    return None
                return self._send(404, {'detail': 'not found'})
            except PermissionError as e:
                return self._send(409, {'detail': str(e)})
            except ValueError as e:
                return self._send(400, {'detail': str(e)})
            except RuntimeError as e:          # 中心不可达或拒绝
                return self._send(502, {'detail': str(e)[:400]})

        def do_GET(self):  # noqa: N802
            self._dispatch('GET')

        def do_POST(self):  # noqa: N802
            self._dispatch('POST')

        def do_PUT(self):  # noqa: N802
            self._dispatch('PUT')
    return H


def serve(home: Path, port: int = DEFAULT_PORT, *, synthetic: bool = False, python: str = sys.executable,
          allow_origins=(), server_override: str = '', ready=None, probes=None):
    """启动本机服务（阻塞）；ready(httpd, svc, token) 在开始监听后回调（测试用）。"""
    svc = NodeService(home, synthetic=synthetic, python=python, probes=probes)
    if server_override:
        svc.update_config({'server': server_override})
    token = secrets.token_urlsafe(32)
    ref = {'port': port}
    httpd = ThreadingHTTPServer(('127.0.0.1', port), make_handler(svc, token, ref, tuple(allow_origins)))
    ref['port'] = httpd.server_address[1]
    info = svc.home / 'service.json'

    def shutdown():
        svc.shutdown()
        httpd.shutdown()
    ref['shutdown'] = shutdown
    info.write_text(json.dumps({'schema': SCHEMA, 'port': ref['port'], 'token': token, 'pid': os.getpid(),
                                'synthetic': synthetic, 'started': time.time()}), encoding='utf-8')
    A.log(f"本机节点服务 http://127.0.0.1:{ref['port']}（令牌见 {info}）" + (' · synthetic' if synthetic else ''))
    if ready:
        ready(httpd, svc, token)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        svc.shutdown()
        httpd.server_close()
        try:
            if json.loads(info.read_text(encoding='utf-8')).get('token') == token:
                info.unlink()
        except (OSError, ValueError):
            pass


def default_home() -> Path:
    return A._home() / 'node_service'


def add_arguments(ap) -> None:
    ap.add_argument('--home', default=str(default_home()), help='服务目录（配置、凭据、历史、工作区）')
    ap.add_argument('--port', type=int, default=DEFAULT_PORT, help='本机端口（0 = 自动选择）')
    ap.add_argument('--python', default=sys.executable, help='运行 runner 的解释器（需 torch / transformers）')
    ap.add_argument('--server-override', default='', help='设置中心地址')
    ap.add_argument('--allow-origin', action='append', default=[], help='开发时放行的网页来源，如 http://localhost:5173')
    ap.add_argument('--synthetic', action='store_true', help='协议联调：伪激活，只在沙盒中心可用；界面不能打开')


def run_from_args(args) -> None:
    serve(Path(args.home), args.port, synthetic=args.synthetic, python=args.python, allow_origins=args.allow_origin,
          server_override=args.server_override)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description='AI2050 本机节点服务')
    add_arguments(p)
    run_from_args(p.parse_args())
