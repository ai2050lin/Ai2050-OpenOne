# -*- coding: utf-8 -*-
"""研发 · 自动测试：被测模型登记 + 「模型 × 模式族」测试队列（autotest.v1）。

客户端在「研发 · 自动测试」勾选若干被测模型与模式族，点击运行后，中心为每个组合生成一个作业；
单个工作线程按顺序执行（一张 GPU 上同时只跑一个模型），每个作业调用
scripts/run_language_family.py --progress，经中心 HTTP 接口走完
预览 → 试跑冻结 → 行为门 → 正式冻结 → 分片采集 → 合并与规律分析 → 报告。

纪律：
- 被测模型必须是本机权重（HF 目录：config.json + *.safetensors）；只有 API 的模型拿不到内部状态，不能作为被测模型。
- 「合成（沙盒）」模型只在沙盒中心（AI2050_DIST_ALLOW_SYNTHETIC=1）可用，结果与报告标注 SYNTHETIC。
- 作业状态与日志落盘在中心数据目录；服务重启时运行中的作业记为 interrupted，不会自动重跑。
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

SCHEMA = 'autotest.v1'
STEPS = [
    (1, '文法预览'), (2, '试跑与行为门'), (3, '冻结正式模板'), (4, '分片采集'),
    (5, '合并'), (6, '规律分析'), (7, '报告'),
]
ACTIVE = ('queued', 'running')
FINAL = ('done', 'failed', 'stopped', 'cancelled', 'interrupted')
DTYPES = ('bf16', 'fp16', 'fp32')
_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._\-]{0,63}$')
PROGRESS = '@@progress '
RESULT = '@@result '
EXIT_GATE_STOP = 3      # 驱动约定：试跑行为门不达标时以 3 退出（不是故障）


def model_dir_info(path: str) -> dict:
    """检查本机 HF 模型目录；不加载权重。"""
    d = Path(path).expanduser()
    info = {'path': str(d), 'exists': d.is_dir(), 'ok': False}
    if not d.is_dir():
        info['error'] = '目录不存在'
        return info
    cfg_path = d / 'config.json'
    sts = sorted(d.glob('*.safetensors'))
    if not cfg_path.is_file():
        info['error'] = '缺少 config.json'
        return info
    if not sts:
        info['error'] = '缺少 *.safetensors 权重'
        return info
    try:
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    except Exception as exc:  # noqa: BLE001
        info['error'] = f'config.json 无法解析：{exc}'
        return info
    info.update(ok=True, arch=(cfg.get('architectures') or [None])[0], layers=cfg.get('num_hidden_layers'),
                hidden=cfg.get('hidden_size'), files=len(sts), bytes=sum(p.stat().st_size for p in sts))
    return info


def _clean_model(raw: dict, allow_synthetic: bool) -> dict:
    kind = str(raw.get('kind') or 'local')
    if kind not in ('local', 'synthetic'):
        raise ValueError('kind 只能是 local（本机权重）或 synthetic（沙盒合成）')
    if kind == 'synthetic' and not allow_synthetic:
        raise ValueError('合成模型只能在沙盒中心使用（AI2050_DIST_ALLOW_SYNTHETIC=1）')
    model_id = str(raw.get('model_id') or '').strip()
    if not _ID_RE.match(model_id):
        raise ValueError('model_id 只能包含字母、数字、. _ -，最长 64 位（它会出现在模型桶名中）')
    dtype = str(raw.get('dtype') or 'bf16').lower()
    if dtype not in DTYPES:
        raise ValueError(f'精度只能是 {"/".join(DTYPES)}')
    path = str(raw.get('model_path') or '').strip()
    if kind == 'local' and not path:
        raise ValueError('本机模型需要权重目录')
    rev = str(raw.get('model_rev') or '').strip()
    if rev and not _ID_RE.match(rev):
        raise ValueError('model_rev 只能包含字母、数字、. _ -')
    return {
        'model_id': model_id, 'kind': kind, 'label': str(raw.get('label') or model_id).strip()[:80],
        'model_path': path if kind == 'local' else '', 'model_rev': rev, 'dtype': dtype,
        'gpu': str(raw.get('gpu') or '').strip()[:80], 'note': str(raw.get('note') or '').strip()[:200],
    }


class AutoTest:
    """被测模型登记 + 作业队列。command_builder 可替换，便于测试。"""

    def __init__(self, data_dir: Path, workspace: Path, center_url: Callable[[], str], admin_token: str,
                 allow_synthetic: Callable[[], bool], python: str = sys.executable,
                 command_builder: Optional[Callable[[dict, dict], List[str]]] = None):
        self.dir = Path(data_dir) / 'autotest'
        self.logs = self.dir / 'logs'
        self.workspace = Path(workspace)
        self.center_url = center_url
        self.admin_token = admin_token
        self.allow_synthetic = allow_synthetic
        self.python = python
        self.command_builder = command_builder or self._default_command
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.proc: Optional[subprocess.Popen] = None
        self.current: Optional[str] = None
        self.worker: Optional[threading.Thread] = None
        self.logs.mkdir(parents=True, exist_ok=True)
        self.models: List[dict] = self._load('models.json', [])
        self.jobs: List[dict] = self._load('jobs.json', [])
        changed = False
        for j in self.jobs:                      # 上次进程退出时未结束的作业
            if j['status'] == 'running':
                j.update(status='interrupted', ended_at=time.time(), error='中心服务重启，作业中断；可点「重跑」')
                changed = True
        if changed:
            self._save_jobs()

    # ---------------- 存储 ----------------
    def _load(self, name, default):
        p = self.dir / name
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return default

    def _write(self, name, obj):
        p = self.dir / name
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, p)

    def _save_models(self):
        self._write('models.json', self.models)

    def _save_jobs(self):
        self.jobs = self.jobs[-500:]
        self._write('jobs.json', self.jobs)

    # ---------------- 被测模型 ----------------
    def list_models(self) -> List[dict]:
        out = []
        for m in self.models:
            m2 = dict(m)
            if m['kind'] == 'local':
                m2['check'] = model_dir_info(m['model_path'])
            else:
                m2['check'] = {'ok': self.allow_synthetic(), 'error': '' if self.allow_synthetic() else '当前不是沙盒中心'}
            out.append(m2)
        return out

    def upsert_model(self, raw: dict) -> dict:
        m = _clean_model(raw, self.allow_synthetic())
        with self.lock:
            for i, old in enumerate(self.models):
                if old['model_id'] == m['model_id']:
                    self.models[i] = {**old, **m, 'updated_at': time.time()}
                    break
            else:
                self.models.append({**m, 'created_at': time.time(), 'updated_at': time.time()})
            self._save_models()
        return m

    def delete_model(self, model_id: str) -> bool:
        with self.lock:
            n = len(self.models)
            self.models = [m for m in self.models if m['model_id'] != model_id]
            self._save_models()
            return len(self.models) < n

    # ---------------- 作业 ----------------
    def submit(self, families: List[str], model_ids: List[str], options: dict, known_families: List[str]) -> dict:
        families = [f for f in dict.fromkeys(families or []) if f]
        model_ids = [m for m in dict.fromkeys(model_ids or []) if m]
        if not families:
            raise ValueError('至少选择一个模式族')
        if not model_ids:
            raise ValueError('至少选择一个被测模型')
        unknown = [f for f in families if f not in known_families]
        if unknown:
            raise ValueError(f'文法库中没有：{", ".join(unknown)}')
        opts = {
            'skip_pilot': bool(options.get('skip_pilot', False)),
            'pilot_per_cell': max(1, min(20, int(options.get('pilot_per_cell', 3) or 3))),
            'min_pilot_pass': max(0.0, min(1.0, float(options.get('min_pilot_pass', 0.5)))),
        }
        with self.lock:
            by_id = {m['model_id']: m for m in self.models}
            missing = [m for m in model_ids if m not in by_id]
            if missing:
                raise ValueError(f'未登记的模型：{", ".join(missing)}')
            for mid in model_ids:
                m = by_id[mid]
                if m['kind'] == 'synthetic' and not self.allow_synthetic():
                    raise ValueError(f'{mid} 是合成模型，只能在沙盒中心运行')
                if m['kind'] == 'local':
                    chk = model_dir_info(m['model_path'])
                    if not chk['ok']:
                        raise ValueError(f'{mid} 的权重目录不可用：{chk.get("error")}（{m["model_path"]}）')
            batch = 'B-' + time.strftime('%m%d%H%M%S') + '-' + uuid.uuid4().hex[:4]
            dup, created = [], []
            for mid in model_ids:                 # 同一模型的模式族连续执行
                for fam in families:
                    if any(j['family'] == fam and j['model_id'] == mid and j['status'] in ACTIVE for j in self.jobs):
                        dup.append(f'{fam} × {mid}')
                        continue
                    m = by_id[mid]
                    job = {'id': 'J-' + uuid.uuid4().hex[:8], 'batch': batch, 'family': fam, 'model_id': mid,
                           'model': {k: m[k] for k in ('model_id', 'kind', 'label', 'model_path', 'model_rev', 'dtype', 'gpu')},
                           'options': opts, 'status': 'queued', 'step': 0, 'step_label': '排队中', 'detail': '',
                           'created_at': time.time(), 'started_at': None, 'ended_at': None, 'exit_code': None,
                           'error': '', 'result': None}
                    self.jobs.append(job)
                    created.append(job)
            self._save_jobs()
        self._ensure_worker()
        self.wake.set()
        return {'batch': batch, 'created': [j['id'] for j in created], 'skipped_duplicates': dup}

    def cancel(self, job_id: str) -> dict:
        with self.lock:
            job = self._job(job_id)
            if job['status'] == 'queued':
                job.update(status='cancelled', ended_at=time.time(), step_label='已取消')
                self._save_jobs()
                return job
            if job['status'] != 'running':
                raise ValueError(f'作业状态为 {job["status"]}，不能取消')
            job['cancel_requested'] = True
            self._save_jobs()
            proc = self.proc if self.current == job_id else None
        if proc is not None:
            _kill_tree(proc)
        return job

    def retry(self, job_id: str) -> dict:
        with self.lock:
            old = self._job(job_id)
            if old['status'] in ACTIVE:
                raise ValueError('作业仍在队列或运行中')
            if not any(m['model_id'] == old['model_id'] for m in self.models):
                raise ValueError(f'模型 {old["model_id"]} 已从登记中删除')
        return self.submit([old['family']], [old['model_id']], old['options'], [old['family']])

    def clear_finished(self) -> int:
        with self.lock:
            n = len(self.jobs)
            self.jobs = [j for j in self.jobs if j['status'] in ACTIVE]
            self._save_jobs()
            return n - len(self.jobs)

    def _job(self, job_id: str) -> dict:
        for j in self.jobs:
            if j['id'] == job_id:
                return j
        raise KeyError(job_id)

    def state(self, limit: int = 100) -> dict:
        with self.lock:
            jobs = [dict(j) for j in self.jobs[-limit:]][::-1]
            queued = sum(1 for j in self.jobs if j['status'] == 'queued')
            running = next((dict(j) for j in self.jobs if j['status'] == 'running'), None)
        for j in jobs:
            j['log_tail'] = self.log_tail(j['id'], 8) if j['status'] in ('running', 'failed', 'stopped', 'interrupted') else []
        return {'schema': SCHEMA, 'steps': [{'n': n, 'label': lb} for n, lb in STEPS], 'models': self.list_models(),
                'jobs': jobs, 'queue': {'queued': queued, 'running': running['id'] if running else None},
                'sandbox_synthetic': self.allow_synthetic(), 'center': self.center_url()}

    def log_path(self, job_id: str) -> Path:
        if not re.match(r'^J-[0-9a-f]{8}$', job_id):
            raise KeyError(job_id)
        return self.logs / f'{job_id}.log'

    def log_tail(self, job_id: str, n: int = 200) -> List[str]:
        try:
            lines = self.log_path(job_id).read_text(encoding='utf-8', errors='replace').splitlines()
        except (OSError, KeyError):
            return []
        return [ln for ln in lines if not ln.startswith(('@@progress', '@@result'))][-n:]

    # ---------------- 执行 ----------------
    def _ensure_worker(self):
        with self.lock:
            if self.worker and self.worker.is_alive():
                return
            self.worker = threading.Thread(target=self._loop, name='autotest-worker', daemon=True)
            self.worker.start()

    def _next(self) -> Optional[dict]:
        with self.lock:
            for j in self.jobs:
                if j['status'] == 'queued':
                    j.update(status='running', started_at=time.time(), step=0, step_label='启动中')
                    self.current = j['id']
                    self._save_jobs()
                    return j
        return None

    def _loop(self):
        while True:
            job = self._next()
            if job is None:
                self.wake.wait(timeout=5.0)
                self.wake.clear()
                continue
            try:
                self._run(job)
            except Exception as exc:  # noqa: BLE001 - 作业失败不能拖垮工作线程
                with self.lock:
                    job.update(status='failed', ended_at=time.time(), error=f'{type(exc).__name__}: {exc}')
                    self._save_jobs()
            finally:
                with self.lock:
                    self.current = None
                    self.proc = None

    def _default_command(self, job: dict, _env: dict) -> List[str]:
        m = job['model']
        o = job['options']
        cmd = [self.python, '-u', str(self.workspace / 'scripts' / 'run_language_family.py'), '--progress',
               '--family', job['family'], '--center', self.center_url(), '--model-id', m['model_id'],
               '--dtype', m['dtype'], '--pilot-per-cell', str(o['pilot_per_cell']),
               '--min-pilot-pass', str(o['min_pilot_pass']), '--node-name', 'autotest-' + m['model_id'],
               '--data-dir', str(self.dir / 'agent' / m['model_id'])]
        if m.get('model_rev'):
            cmd += ['--model-rev', m['model_rev']]
        if m.get('gpu'):
            cmd += ['--gpu', m['gpu']]
        if o['skip_pilot']:
            cmd.append('--skip-pilot')
        if m['kind'] == 'synthetic':
            cmd.append('--synthetic')
        else:
            cmd += ['--model-path', m['model_path']]
        return cmd

    def _run(self, job: dict):
        env = dict(os.environ)
        env['AI2050_DIST_ADMIN_TOKEN'] = self.admin_token     # 令牌走环境变量，不出现在命令行
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        cmd = self.command_builder(job, env)
        log_path = self.log_path(job['id'])
        kw = {}
        if os.name == 'nt':
            kw['creationflags'] = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
        else:
            kw['start_new_session'] = True
        with open(log_path, 'w', encoding='utf-8') as log:
            log.write(f'# {job["id"]} · {job["family"]} × {job["model_id"]} · {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            log.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding='utf-8', errors='replace', cwd=str(self.workspace), env=env, **kw)
            with self.lock:
                self.proc = proc
                cancel_now = job.get('cancel_requested')
            if cancel_now:
                _kill_tree(proc)
            for line in iter(proc.stdout.readline, ''):
                log.write(line)
                log.flush()
                self._parse(job, line.rstrip('\n'))
            code = proc.wait()
            proc.stdout.close()
        with self.lock:
            job['exit_code'] = code
            job['ended_at'] = time.time()
            if job.get('cancel_requested'):
                job.update(status='cancelled', step_label='已取消')
            elif code == 0 and job.get('result'):
                job.update(status='done', step=len(STEPS), step_label='完成')
            elif code == EXIT_GATE_STOP:
                job.update(status='stopped', step_label='试跑行为门未达标，已停止')
            else:
                tail = self.log_tail(job['id'], 3)
                job.update(status='failed', error=(tail[-1] if tail else f'退出码 {code}'))
            self._save_jobs()

    def _parse(self, job: dict, line: str):
        if line.startswith(PROGRESS):
            try:
                ev = json.loads(line[len(PROGRESS):])
            except json.JSONDecodeError:
                return
            with self.lock:
                job['step'] = int(ev.get('step') or job['step'])
                job['step_label'] = str(ev.get('label') or job['step_label'])
                job['detail'] = str(ev.get('detail') or '')
                for k in ('tm_id', 'bucket', 'pilot_tm_id'):
                    if ev.get(k):
                        job[k] = ev[k]
                self._save_jobs()
        elif line.startswith(RESULT):
            try:
                res = json.loads(line[len(RESULT):])
            except json.JSONDecodeError:
                return
            with self.lock:
                job['result'] = res
                job['tm_id'] = res.get('tm_id', job.get('tm_id'))
                job['bucket'] = res.get('bucket', job.get('bucket'))
                self._save_jobs()


def _kill_tree(proc: subprocess.Popen):
    """结束驱动及其子进程（node_agent / runner）。"""
    if proc.poll() is not None:
        return
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True, check=False)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass
