#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI2050 分布式节点 Agent —— design/distributed_platform_plan_v1.md C1（M1 版）

流程: register → [claim → 落盘模板包 → 本地执行 runner → 收集产物 → 分块上传 → complete] × N

用法（Windows/Linux 通用，仅依赖标准库）:
    python server/node_agent.py --server http://<中心>:5010 register --name gpu-box-1
    python server/node_agent.py --server http://<中心>:5010 run --once
    python server/node_agent.py --server http://<中心>:5010 run --interval 60 \
        --model-path D:/models/hf/qwen3-4b          # 设置后 runner 走真实采集（需 torch）

凭据保存在 <data-dir>/node_credentials.json（node_token 与 AI API key 无关；
执行器环境 AI2050_RUNNER_MODEL 决定 runner 真实/冒烟模式——smoke 结果会被诚实标记，不冒充实测）。

上传线格式（与中心约定）: 4B 路径表长(big-endian) + 路径表 JSON + 按 manifest 顺序的文件内容。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

CHUNK_SIZE = 1_500_000  # 与中心 CHUNK_SIZE 一致（base64 前）


def http_json(url: str, method: str = 'GET', payload=None, token: str = '',
              timeout: float = 30, ok_204: bool = False):
    req = urllib.request.Request(url, method=method)
    req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('X-Node-Token', token)
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
            body = r.read()
            return r.status, (json.loads(body.decode('utf-8')) if body else None)
    except urllib.error.HTTPError as e:
        body = e.read()
        if ok_204 and e.code == 204:
            return 204, None
        try:
            detail = json.loads(body.decode('utf-8')).get('detail', body[:200])
        except Exception:
            detail = body[:200].decode('utf-8', 'replace')
        raise RuntimeError(f'{method} {url} -> {e.code}: {detail}') from e
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as e:
        raise RuntimeError(f'{method} {url} -> network: {e.reason}') from e


# 本机节点服务（node_service.py）挂接的钩子：日志、runner 进度、任务结论；命令行下为空。
LOG_HOOKS: list = []
PROGRESS_HOOKS: list = []
RESULT_HOOKS: list = []
RUNNING: set = set()            # 正在运行的 runner 子进程（服务的「停止」会结束它们）


def _emit(hooks, *a) -> None:
    for h in list(hooks):
        try:
            h(*a)
        except Exception:
            pass


def _console(line: str, end: str = '\n') -> None:
    """打印到控制台；中文控制台（GBK）不能编码的字符替换掉，输出已关闭时忽略——控制台输出不能让任务失败。"""
    try:
        print(line, end=end, flush=True)
    except (UnicodeEncodeError, OSError, ValueError):
        try:
            enc = getattr(sys.stdout, 'encoding', None) or 'ascii'
            print(line.encode(enc, 'replace').decode(enc, 'replace'), end=end, flush=True)
        except Exception:
            pass


def log(msg: str) -> None:
    _console(f"[agent {time.strftime('%H:%M:%S')}] {msg}")
    _emit(LOG_HOOKS, msg)


def abort_running() -> int:
    """结束全部正在运行的 runner；返回结束的个数。未上传的租约到期后由中心重新派发。"""
    n = 0
    for p in list(RUNNING):
        if p.poll() is None:
            p.kill()
            n += 1
    return n


def _spawn(cmd, **kw):
    p = subprocess.Popen(cmd, **kw)
    RUNNING.add(p)
    return p


def creds_path(data_dir: Path) -> Path:
    return data_dir / 'node_credentials.json'


def load_creds(data_dir: Path) -> dict:
    p = creds_path(data_dir)
    if not p.is_file():
        raise SystemExit(f'no credentials at {p} — 先执行 register')
    return json.loads(p.read_text(encoding='utf-8'))


def cmd_register(args) -> None:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    status, resp = http_json(
        args.server.rstrip('/') + '/api/nodes/register', 'POST',
        {'name': args.name, 'gpu': args.gpu, 'model': args.model, 'dtype': args.dtype,
         'bind_code': args.bind_code})
    p = creds_path(data_dir)
    p.write_text(json.dumps({'server': args.server.rstrip('/'), **resp}, ensure_ascii=False,
                            indent=1), encoding='utf-8')
    log(f"registered node_id={resp['node_id']} token saved -> {p}"
        + ('（已绑定到你的账号）' if resp.get('owner_bound') else ''))


def pack_blob(files: dict) -> bytes:
    """文件字典 → 4B 路径表长 + 路径表 + 依序内容。"""
    paths = list(files.keys())
    table = json.dumps(paths, ensure_ascii=False).encode('utf-8')
    return len(table).to_bytes(4, 'big') + table + b''.join(files.values())


def execute_task(args, task: dict, work_root: Path) -> Path:
    """落盘模板包 → 跑 runner → 返回 outputs 目录。"""
    tdir = work_root / task['task_id']
    if tdir.exists():
        import shutil
        shutil.rmtree(tdir)
    (tdir / 'bundle').mkdir(parents=True)
    (tdir / 'outputs').mkdir()
    for name, content in task['files'].items():
        (tdir / 'bundle' / name).write_bytes(content.encode('utf-8'))  # 字节写入：禁 CRLF 翻译（保 design_sha 一致）
    research_ref = task.get('research_work_unit') or {}
    if research_ref:
        # Immutable task context is adjacent to the frozen template bundle. It
        # never changes the template bytes/design SHA and contains no secrets.
        ref_path = tdir / 'research_work_unit.json'
        ref_path.write_text(json.dumps(research_ref, ensure_ascii=False, sort_keys=True, indent=2),
                            encoding='utf-8')
    runner = tdir / 'bundle' / 'runner.py'
    env = dict(os.environ)
    if research_ref:
        env['AI2050_RESEARCH_WORK_UNIT_PATH'] = str(ref_path)
    if args.model_path:
        env['AI2050_RUNNER_MODEL'] = args.model_path
    # dist_protocol.v3：分片与精度由中心节点/节点参数下发给 runner（legacy runner 忽略这两个变量）
    if task.get('shard'):
        env['AI2050_SHARD'] = str(task['shard'])
    if args.dtype:
        env['AI2050_PRECISION'] = args.dtype
    suffix = f" work_unit={research_ref.get('work_unit_id')}" if research_ref else ''
    log(f"exec {task['tm_id']} ({task['name']}) runner with {Path(args.python).name}{suffix}")
    proc = _spawn([args.python, str(runner), '--out', str(tdir / 'outputs')], cwd=str(tdir), env=env,
                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    try:
        out, err = proc.communicate(timeout=args.run_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        raise RuntimeError(f'runner timeout after {args.run_timeout:.0f}s')
    finally:
        RUNNING.discard(proc)
    log(f"runner exit={proc.returncode} {out.strip().splitlines()[-1] if out.strip() else ''}")
    if proc.returncode != 0:
        raise RuntimeError(f'runner failed: {err[-500:]}')
    return tdir / 'outputs'


def upload_outputs(args, creds: dict, task: dict, out_dir: Path) -> str:
    files = {f.name: f.read_bytes() for f in sorted(out_dir.iterdir()) if f.is_file()}
    if not files:
        raise RuntimeError('runner produced no output files')
    summary = {}
    if 'summary.json' in files:
        try:
            summary = json.loads(files['summary.json'].decode('utf-8'))
        except Exception:
            pass
    # synthetic 只在沙盒中心（AI2050_DIST_ALLOW_SYNTHETIC=1）参与合并；正式中心照常不合并
    kind = summary.get('status') if summary.get('status') in ('real', 'synthetic') else 'smoke'
    manifest = [{'path': name, 'sha256': hashlib.sha256(b).hexdigest(), 'size': len(b)}
                for name, b in files.items()]
    base = args.server.rstrip('/') if not creds.get('server') else creds['server']
    st, init = http_json(base + '/api/results/upload/init', 'POST', {
        'task_id': task['task_id'], 'tm_id': task['tm_id'],
        'design_sha': task['design_sha'], 'model_id': args.model_id,
        'model_rev': args.model_rev, 'precision': args.dtype,
        'tokenizer_sha256': args.tokenizer_sha256,
        'seed': args.seed, 'kind': kind,
        'summary': summary, 'manifest': manifest,
    }, token=creds['node_token'])
    blob = pack_blob(files)
    chunks = [blob[i:i + CHUNK_SIZE] for i in range(0, len(blob), CHUNK_SIZE)] or [b'']
    log(f"upload kind={kind} {len(files)} files {len(blob)} bytes -> {len(chunks)} chunks")
    for seq, c in enumerate(chunks):
        http_json(base + f"/api/results/upload/{init['upload_id']}/chunk", 'POST',
                  {'seq': seq, 'data_b64': base64.b64encode(c).decode('ascii')},
                  token=creds['node_token'])
    st, fin = http_json(base + f"/api/results/upload/{init['upload_id']}/finish", 'POST',
                        {}, token=creds['node_token'])
    return fin['sha']


def run_one(args, creds: dict, work_root: Path) -> bool:
    """单次领任务循环；返回是否执行了任务。"""
    base = creds['server']
    st, task = http_json(base + '/api/tasks/claim', 'POST',
                         {'model_id': args.model_id, 'model_rev': args.model_rev,
                          'tokenizer_sha256': args.tokenizer_sha256,
                          'gpu': args.gpu, 'dtype': args.dtype, 'seed': args.seed,
                          'tm_id': getattr(args, 'tm_id', '') or ''},
                         token=creds['node_token'], ok_204=True)
    if st == 204 or not task:
        return False
    log(f"claimed {task['task_id']} {task['tm_id']} v{task['version']} "
        f"design_sha={task['design_sha'][:12]}" + (f" shard={task['shard']}" if task.get('shard') else ''))
    stop_hb = threading.Event()

    def hb():
        while not stop_hb.wait(300):
            try:
                http_json(base + f"/api/tasks/{task['task_id']}/heartbeat", 'PUT', {},
                          token=creds['node_token'])
            except Exception as e:
                log(f'heartbeat failed: {e}')
    threading.Thread(target=hb, daemon=True).start()
    try:
        out_dir = execute_task(args, task, work_root)
        sha = upload_outputs(args, creds, task, out_dir)
        log(f"result sha={sha[:16]} uploaded")
        http_json(base + f"/api/tasks/{task['task_id']}/complete", 'POST',
                  {'status': 'done', 'result_sha': sha}, token=creds['node_token'])
        _emit(RESULT_HOOKS, {'kind': 'family_test', 'task': task['task_id'], 'tm_id': task['tm_id'],
                             'shard': task.get('shard'), 'status': 'done', 'sha': sha})
    except Exception as e:
        log(f"task failed: {e}")
        _emit(RESULT_HOOKS, {'kind': 'family_test', 'task': task['task_id'], 'tm_id': task['tm_id'],
                             'shard': task.get('shard'), 'status': 'failed', 'note': str(e)[:300]})
        try:
            http_json(base + f"/api/tasks/{task['task_id']}/complete", 'POST',
                      {'status': 'failed', 'note': str(e)[:300]}, token=creds['node_token'])
        except Exception:
            pass
    finally:
        stop_hb.set()
    return True


def cmd_run(args) -> None:
    creds = load_creds(Path(args.data_dir))
    if args.server:
        creds['server'] = args.server.rstrip('/')
    work_root = Path(args.data_dir) / 'tasks'
    work_root.mkdir(parents=True, exist_ok=True)
    log(f"agent loop server={creds['server']} once={args.once} "
        f"model={'REAL:' + args.model_path if args.model_path else 'smoke'}")
    while True:
        try:
            did = run_one(args, creds, work_root)
        except Exception as e:
            log(f'cycle error: {e}')
            did = False
        if args.once:
            if not did:
                log('no task available (--once)')
            break
        time.sleep(args.interval if not did else 2.0)


def cmd_download(args) -> None:
    """研究包下载（M6-P1，design/distributed_flow_plan_v1.md §2.4）：
    GET /api/kits/{kit}/bundle → 落地 kit/{corpus,runner,contract}+results_manifest+README，
    并本地重算 design_sha 复核模板冻结（corpus+runner 规范化哈希）。"""
    base = args.server.rstrip('/')
    st, bundle = http_json(base + f'/api/kits/{args.kit}/bundle', 'GET', None, timeout=60)
    if st != 200 or not bundle:
        raise RuntimeError(f'bundle fetch failed: HTTP {st}')
    out = Path(args.out) / args.kit
    (out / 'bundle').mkdir(parents=True, exist_ok=True)
    files = bundle.get('files') or {}
    for name, content in files.items():
        (out / 'bundle' / name).write_bytes(content.encode('utf-8'))  # 字节写入：禁 CRLF 翻译（保 design_sha 一致）
    (out / 'results_manifest.json').write_text(
        json.dumps(bundle.get('results_manifest') or [], ensure_ascii=False, indent=1),
        encoding='utf-8')
    (out / 'README.md').write_text(bundle.get('README.md') or '', encoding='utf-8')
    # 冻结复核：corpus + runner 重算 design_sha，与 contract.json 对账
    contract = json.loads(files.get('contract.json', '{}'))
    corpus = json.loads(files.get('corpus.json', '{}'))
    runner = files.get('runner.py', '')
    design = hashlib.sha256(json.dumps(
        {'tm_id': contract.get('tm_id'), 'version': contract.get('version'),
         'corpus': corpus, 'runner_sha256': hashlib.sha256(runner.encode('utf-8')).hexdigest()},
        sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    ok = design == contract.get('design_sha') == bundle.get('template', {}).get('design_sha')
    log(f"kit={args.kit} -> {out} · bundle files={len(files)} · "
        f"results={len(bundle.get('results_manifest') or [])} · design_sha {'✓ 冻结一致' if ok else '✗ 不一致（勿用于复现！）'}")
    if not ok:
        raise SystemExit('design_sha mismatch — 研究包冻结校验失败，请上报中心节点')


def cmd_status(args) -> None:
    """本机 agent 状态（S8）：读本地凭据 → GET /api/nodes/me，打印节点信息 + 最近任务。"""
    creds = load_creds(Path(args.data_dir))
    server, token, node_id = creds['server'], creds['node_token'], creds['node_id']
    _, me = http_json(server + '/api/nodes/me', 'GET', token=token)
    n = me.get('node') or {}
    log(f"node={node_id} name={n.get('name')} gpu={n.get('gpu')} model={n.get('model')} "
        f"dtype={n.get('dtype')} trust={n.get('trust')} last_seen={n.get('last_seen')}")
    rows = me.get('recent_tasks') or []
    if not rows:
        log('recent_tasks: （无）')
    for t in rows:
        log(f"  {t['task_id']} {t['tm_id']} {t['status']}"
            + (f" -> {t['result_sha'][:12]}" if t.get('result_sha') else ''))


def cmd_queue(args) -> None:
    """调度队列只读快照（S8 /api/tasks，无需凭据）：全队列统计 + 活跃租约。"""
    _, q = http_json(args.server.rstrip('/') + '/api/tasks', 'GET')
    stats = q.get('stats') or {}
    log(f"server={args.server} · 活跃 {stats.get('claimed', 0)} · 完成 {stats.get('done', 0)} · "
        f"失败 {stats.get('failed', 0)} · 过期 {stats.get('expired', 0)} · 单租约 {q.get('lease_hours')}h")
    for a in (q.get('active') or []):
        remain = max(0.0, (a['lease_until'] - q['server_time']) / 3600.0)
        log(f"  {a['task_id']} {a['tm_id']} node={a.get('node_name') or '?'} "
            f"model={a.get('model') or '?'} 租约剩 {remain:.1f}h")
    if not (q.get('active') or []):
        log('  （当前无活跃租约）')


# ---------- 一键参与（participate）：参与者一键包的入口 ----------
# 流程：读 config.json → 首次注册（绑定码）→ 检查 GPU → 解析所需模型并核对快照 → 循环领取「普查分片 / 模式族测试」→ 运行 → 上传。
# 任务包文件逐个核对 sha256；普查 runner 与 census.py 还要与作业规范 job.json 中冻结的 sha256 一致，否则拒绝执行。

SNAPSHOT_PATTERNS = ('*.safetensors', '*.bin', '*.pt', '*.gguf', 'config.json', 'generation_config.json',
                     'tokenizer*', 'vocab*', 'merges.txt', 'special_tokens_map.json', '*.model', '*.tiktoken')


def _home() -> Path:
    return Path(os.environ.get('AI2050_HOME', str(Path.home() / '.ai2050')))


def model_snapshot(model_path) -> dict:
    """与 runner_v3 / census_runner 同一口径；大文件哈希按 (路径, 大小, 修改时间) 缓存。"""
    cache_p = _home() / 'snapshot_cache.json'
    try:
        cache = json.loads(cache_p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        cache = {}
    root, files, seen = Path(model_path), {}, set()
    for pat in SNAPSHOT_PATTERNS:
        for p in (sorted(root.glob(pat)) if root.is_dir() else []):
            if not p.is_file() or p.name in seen:
                continue
            seen.add(p.name)
            st = p.stat()
            key = f'{p.resolve()}|{st.st_size}|{int(st.st_mtime)}'
            if key not in cache:
                log(f'计算 {p.name} 的 sha256（{st.st_size / 2**30:.1f} GB，只算一次）')
                h = hashlib.sha256()
                with p.open('rb') as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b''):
                        h.update(chunk)
                cache[key] = h.hexdigest()
            files[p.name] = {'sha256': cache[key], 'size': st.st_size}
    try:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_p.write_text(json.dumps(cache), encoding='utf-8')
    except OSError:
        pass
    total = hashlib.sha256(json.dumps(files, sort_keys=True).encode('utf-8')).hexdigest()
    return {'sha256': total, 'files': files, 'n_files': len(files)}


def http_download(url: str, dest: Path, timeout: float = 120) -> None:
    tmp = dest.with_suffix(dest.suffix + '.part')
    with urllib.request.urlopen(url, timeout=timeout) as r, tmp.open('wb') as fh:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
    tmp.replace(dest)


def http_bytes(url: str, data: bytes, token: str, timeout: float = 300) -> dict:
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'application/octet-stream')
    req.add_header('X-Node-Token', token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8') or '{}')
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'POST {url} -> {e.code}: {e.read()[:200]!r}') from e


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _save_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding='utf-8')


def _gpu_info(device: str) -> dict:
    try:
        import torch
    except ImportError:
        return {'ok': False, 'reason': '没有安装 torch（运行 start 脚本会自动安装）'}
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            return {'ok': False, 'reason': '没有检测到可用的 CUDA GPU（需要 NVIDIA 显卡与驱动）'}
        i = int(device.split(':')[1]) if ':' in device else 0
        p = torch.cuda.get_device_properties(i)
        return {'ok': True, 'name': p.name, 'vram_gib': round(p.total_memory / 2**30, 1), 'torch': torch.__version__}
    return {'ok': True, 'name': 'cpu', 'vram_gib': 0, 'torch': torch.__version__}


def _resolve_model(cfg: dict, cfg_path: Path, m: dict, interactive: bool) -> str:
    """作业所需模型 → 本机目录（核对快照）。返回目录，或空串表示跳过该模型。"""
    want = m.get('snapshot_sha256') or ''
    dirs = cfg.setdefault('model_dirs', {})
    cand = dirs.get(want) or ''
    while True:
        if not cand and interactive:
            hint = f"{m.get('hf_repo')}@{(m.get('hf_revision') or 'main')[:12]}" if m.get('hf_repo') else ''
            print(f"\n需要模型 {m.get('name') or ''}（快照 {want[:16]}）。")
            tip = f'，或直接回车从 Hugging Face 下载 {hint}' if hint else ''
            ans = input(f'  输入本机模型目录{tip}（输入 s 跳过）：').strip().strip('"')
            if ans.lower() == 's':
                return ''
            if not ans and hint:
                root = Path(cfg.get('models_root') or (_home() / 'models'))
                target = root / (m.get('name') or want[:16])
                print(f'  将下载到 {target}（数 GB，可随时中断后重试）')
                if input('  确认下载？[y/N] ').strip().lower() != 'y':
                    return ''
                from huggingface_hub import snapshot_download
                snapshot_download(m['hf_repo'], revision=m.get('hf_revision') or None, local_dir=str(target))
                ans = str(target)
            cand = ans
        if not cand:
            return ''
        snap = model_snapshot(cand)
        if snap['sha256'] == want:
            dirs[want] = cand
            _save_config(cfg_path, cfg)
            log(f"模型 {m.get('name')} 快照一致 ✓ {cand}")
            return cand
        log(f"模型目录 {cand} 的快照 {snap['sha256'][:16]} 与要求 {want[:16]} 不一致"
            '（文件不同：可能是不同版本或不同下载来源），不能用于该模型的任务')
        if not interactive:
            return ''
        cand = ''


def _model_rev(model_dir: str) -> str:
    cfgp = Path(model_dir) / 'config.json'
    return 'cfg-' + hashlib.sha256(cfgp.read_bytes()).hexdigest()[:12] if cfgp.is_file() else 'unknown'


def run_census_shard(server: str, token: str, work: Path, model_dir: str, device: str, mode: str, py: str,
                     snapshot: str = '', job_id: str = '') -> bool:
    """领取并执行一个普查分片；返回是否领到了分片。"""
    st, lease = http_json(server + '/api/census/claim', 'POST', {'model_snapshot': snapshot, 'mode': mode, 'job_id': job_id},
                          token=token, ok_204=True)
    if st == 204 or not lease:
        return False
    lid = lease['lease_id']
    layers = lease['spec']['layers']
    log(f"领取普查分片 {lease['job_id']} / {lease['shard']}（层 {layers[0]}–{layers[-1]}，"
        f"应计入 {lease['spec']['expected_n']:,} token）")
    stop = threading.Event()

    def hb():
        while not stop.wait(300):
            try:
                http_json(server + f'/api/census/leases/{lid}/heartbeat', 'PUT', {}, token=token)
            except Exception as e:
                log(f'续租失败：{e}')
    threading.Thread(target=hb, daemon=True).start()
    tdir = work / lid
    try:
        cache = work / 'cache'
        cache.mkdir(parents=True, exist_ok=True)
        tdir.mkdir(parents=True, exist_ok=True)
        for f in lease['files']:
            cp = cache / f['sha256']
            if not cp.is_file() or sha256_file(cp) != f['sha256']:
                http_download(server + f['url'], cp)
            if sha256_file(cp) != f['sha256']:
                raise RuntimeError(f"任务包文件 {f['name']} sha256 不符，拒绝执行")
            (tdir / f['name']).write_bytes(cp.read_bytes())
        job = json.loads((tdir / 'job.json').read_text(encoding='utf-8'))
        for n, h in (job.get('code') or {}).items():
            if sha256_file(tdir / n) != h:
                raise RuntimeError(f'{n} 与作业冻结的 sha256 不符，拒绝执行')
        tokens = next(f['name'] for f in lease['files'] if f['name'].startswith('tokens_'))
        cmd = [py, str(tdir / 'census_runner.py'), '--job', str(tdir / 'job.json'), '--shard', lease['shard'],
               '--tokens', str(tdir / tokens), '--out', str(tdir / 'outputs'), '--device', device, '--mode', mode]
        if mode == 'real':
            cmd += ['--model', model_dir]
        proc = _spawn(cmd, cwd=str(tdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                      encoding='utf-8', errors='replace')
        tail = []
        for line in proc.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-20:]
            if line.startswith('@@progress '):
                p = json.loads(line[11:])
                _emit(PROGRESS_HOOKS, {'job': lease['job_id'], 'shard': lease['shard'], **p})
                _console(f"\r  进度 {p['done']}/{p['total']} 窗口 · {p['tokens']:,} token · {p['tok_per_s']:.0f} tok/s · "
                         f"{p['seconds']:.0f}s   ", end='')
        rc = proc.wait()
        proc.stdout.close()
        RUNNING.discard(proc)
        if rc != 0:
            raise RuntimeError('runner 失败：' + ' | '.join(tail[-5:]))
        _console('')
        data = (tdir / 'outputs' / 'shard.npz').read_bytes()
        summary = json.loads((tdir / 'outputs' / 'summary.json').read_text(encoding='utf-8'))
        size = int(lease.get('chunk_bytes') or 4_000_000)
        for seq, i in enumerate(range(0, max(len(data), 1), size)):
            http_bytes(server + f'/api/census/leases/{lid}/chunk?seq={seq}', data[i:i + size], token)
        _, fin = http_json(server + f'/api/census/leases/{lid}/finish', 'POST',
                           {'sha256': hashlib.sha256(data).hexdigest(), 'summary': summary}, token=token, timeout=600)
        res = {'kind': 'census_shard', 'task': lid, 'job': lease['job_id'], 'shard': lease['shard'],
               'status': fin.get('status') or 'unknown', 'sha': fin.get('sha', ''),
               'replica': (fin.get('replica') or {}).get('ok')}
        if fin.get('status') != 'accepted':
            res['note'] = '; '.join(fin.get('errors') or [])[:300]
        _emit(RESULT_HOOKS, res)
        if fin.get('status') == 'accepted':
            rep = fin.get('replica')
            log(f"上传完成，中心校验通过 ✓ {fin['sha'][:12]}"
                + ('' if rep is None else f"（复算比较：{'一致' if rep['ok'] else '不一致，待审'}）"))
        else:
            log(f"中心拒收：{'; '.join(fin.get('errors') or [])[:400]}")
    except Exception as e:
        log(f'普查分片失败：{e}')
        _emit(RESULT_HOOKS, {'kind': 'census_shard', 'task': lid, 'job': lease['job_id'], 'shard': lease['shard'],
                             'status': 'failed', 'note': str(e)[:300]})
        try:
            http_json(server + f'/api/census/leases/{lid}/fail', 'POST', {'note': str(e)[:400]}, token=token)
        except Exception:
            pass
    finally:
        stop.set()
        import shutil
        shutil.rmtree(tdir, ignore_errors=True)
    return True


def cmd_participate(args) -> None:
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.is_file() else {}
    server = (args.server_override or cfg.get('server') or '').rstrip('/')
    if not server:
        raise SystemExit('config.json 缺少 server（中心地址）')
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    interactive = sys.stdin.isatty() and not args.non_interactive
    mode = 'synthetic' if args.synthetic else 'real'
    device = cfg.get('device') or 'cuda'
    print('=' * 64 + f'\nAI2050 开放研究平台 · 参与节点\n中心：{server}\n' + '=' * 64, flush=True)
    gpu = _gpu_info(device) if mode == 'real' else {'ok': True, 'name': 'synthetic', 'vram_gib': 0}
    if not gpu['ok']:
        raise SystemExit(f"无法参与：{gpu['reason']}")
    if mode == 'real':
        log(f"GPU：{gpu['name']} · 显存 {gpu['vram_gib']} GiB · torch {gpu.get('torch')}")
    else:
        device = 'cpu'
    p = creds_path(data_dir)
    creds = json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {}
    if creds.get('server') != server:
        import platform as _pf
        _, resp = http_json(server + '/api/nodes/register', 'POST', {
            'name': cfg.get('node_name') or _pf.node() or 'node', 'gpu': gpu.get('name', ''), 'model': '', 'dtype': 'bf16',
            'bind_code': cfg.get('bind_code', '')})
        creds = {'server': server, **resp}
        p.write_text(json.dumps(creds, ensure_ascii=False, indent=1), encoding='utf-8')
        log(f"节点已注册 {resp['node_id']}" + ('，已绑定到你的账号' if resp.get('owner_bound') else '（未绑定账号：贡献不会署名）'))
        if cfg.get('bind_code'):
            cfg['bind_code'] = ''                     # 一次性，用过即清除
            _save_config(cfg_path, cfg)
    token = creds['node_token']
    _, pcfg = http_json(server + '/api/platform/config', 'GET')
    offered = pcfg.get('task_kinds') or []
    kinds = [k for k in (cfg.get('task_kinds') or offered) if k in offered]
    models = {m['snapshot_sha256']: m for m in (pcfg.get('models') or []) if m.get('snapshot_sha256')}
    ready = {}
    if mode == 'real':
        for want, m in models.items():
            d = _resolve_model(cfg, cfg_path, m, interactive)
            if d:
                ready[want] = d
        if not ready:
            raise SystemExit('没有可用的模型：请在 config.json 的 model_dirs 填写本机模型目录，或按提示下载。')
    names = ', '.join(models[k].get('name') or k[:12] for k in ready) or mode
    log(f"任务类型：{', '.join(kinds) or '无'} · 模型：{names}")
    work = data_dir / 'census'
    py = sys.executable
    idle = 0
    while True:
        did = False
        try:
            if 'census_shard' in kinds:
                if mode == 'synthetic':
                    did = run_census_shard(server, token, work, '', 'cpu', mode, py)
                else:
                    for snap, d in ready.items():
                        if run_census_shard(server, token, work, d, device, mode, py, snapshot=snap):
                            did = True
                            break
            if not did and 'family_test' in kinds and mode == 'real':
                for snap, d in ready.items():
                    ns = argparse.Namespace(python=py, run_timeout=6 * 3600.0, model_path=d,
                                            model_id=(models[snap].get('name') or '').lower(), model_rev=_model_rev(d),
                                            tokenizer_sha256='', gpu=gpu.get('name', ''), dtype='bf16', seed=0, tm_id='',
                                            server=server)
                    if run_one(ns, creds, data_dir / 'tasks'):
                        did = True
                        break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f'本轮出错：{e}')
        if args.once:
            if not did:
                log('当前没有可领取的任务')
            return
        if did:
            idle = 0
            continue
        idle += 1
        if idle == 1:
            log(f'暂时没有任务，{int(args.interval)} 秒后再看（Ctrl+C 退出）')
        time.sleep(args.interval)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):   # Windows 中文控制台（GBK）打印 ✓ 等字符时不崩溃
        try:
            stream.reconfigure(errors='replace')
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description='AI2050 distributed node agent')
    ap.add_argument('--server', default=os.environ.get('AI2050_DIST_SERVER', 'http://127.0.0.1:5010'))
    ap.add_argument('--data-dir', default=str(Path.home() / '.ai2050_agent'))
    sub = ap.add_subparsers(dest='cmd', required=True)

    rg = sub.add_parser('register', help='注册节点并保存凭据')
    rg.add_argument('--name', default='node-' + uuid.uuid4().hex[:4])
    rg.add_argument('--gpu', default='')
    rg.add_argument('--model', default='')
    rg.add_argument('--dtype', default='')
    rg.add_argument('--bind-code', default=os.environ.get('AI2050_BIND_CODE', ''),
                    help='账号绑定码（客户端 总览/题库「参与测试」生成，30 分钟内有效）：节点归属你的账号')
    rg.set_defaults(fn=cmd_register)

    rn = sub.add_parser('run', help='领任务执行循环')
    rn.add_argument('--once', action='store_true', help='只跑一轮（有任务即执行一次）')
    rn.add_argument('--interval', type=float, default=60.0)
    rn.add_argument('--python', default=sys.executable, help='执行 runner 的解释器')
    rn.add_argument('--run-timeout', type=float, default=3600.0)
    rn.add_argument('--model-path', default='', help='HF 模型目录；留空=冒烟模式')
    rn.add_argument('--model-id', default='', help='登记到结果库的模型标识')
    rn.add_argument('--model-rev', default='', help='精确模型快照 ID；Research Work Unit 认领时必须匹配')
    rn.add_argument('--tokenizer-sha256', default='', help='tokenizer 文件集合 SHA256；Research Work Unit 认领时必须匹配')
    rn.add_argument('--gpu', default='')
    rn.add_argument('--dtype', default='')
    rn.add_argument('--seed', type=int, default=0)
    rn.add_argument('--tm-id', default='', help='只领取指定模板（如 TG-KN-ISA）的任务')
    rn.set_defaults(fn=cmd_run)

    dl = sub.add_parser('download', help='下载研究包（模板 bundle + 结果清单 + README，本地可复现）')
    dl.add_argument('kit', help='研究包 id，格式 {tm_id}__{analysis}，如 TM-04__rsa-rdm（清单见 GET /api/kits）')
    dl.add_argument('--out', default='./kit', help='落地根目录（默认 ./kit/<kit_id>/）')
    dl.set_defaults(fn=cmd_download)

    st = sub.add_parser('status', help='本机 agent 状态（凭据 → GET /api/nodes/me）')
    st.set_defaults(fn=cmd_status)

    q = sub.add_parser('queue', help='调度队列只读快照（GET /api/tasks，无需凭据）')
    q.set_defaults(fn=cmd_queue)

    pp = sub.add_parser('participate', help='一键参与：按 config.json 注册、核对模型、循环领取并运行任务')
    pp.add_argument('--config', default='config.json')
    pp.add_argument('--server-override', default='', help='覆盖 config.json 中的中心地址')
    pp.add_argument('--interval', type=float, default=60.0)
    pp.add_argument('--once', action='store_true', help='只执行一轮')
    pp.add_argument('--non-interactive', action='store_true', help='不询问（缺模型时跳过）')
    pp.add_argument('--synthetic', action='store_true', help='协议联调：伪激活，只在沙盒中心可用')
    pp.set_defaults(fn=cmd_participate)

    sv = sub.add_parser('serve', help='本机节点服务：供桌面客户端控制的本机接口（只监听 127.0.0.1，见 node_service.py）')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import node_service
    node_service.add_arguments(sv)
    sv.set_defaults(fn=node_service.run_from_args)

    args = ap.parse_args()
    args.data_dir = str(Path(args.data_dir).resolve())  # 子进程 cwd 切换前先固定绝对路径
    args.fn(args)


if __name__ == '__main__':
    main()
