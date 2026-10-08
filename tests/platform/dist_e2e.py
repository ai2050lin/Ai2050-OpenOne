# -*- coding: utf-8 -*-
"""分布式平台 e2e 联调（M1 验收预演）:
register → claim → 落盘 bundle → runner 冒烟 → init/chunk/finish 上传 →
下载核验 → summary → agg → templates → news → 负路径（design_sha 不符拒收）。
真实模式（--real）: 二次 claim 用 models/hf/qwen3-4b 走 GPU 真实采集上传（kind=real）。
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:5010'
REPO = Path(r'D:\AI2050\Ai2050-OpenOne')
WS = REPO / 'tests' / 'dist_test_data' / 'agent_ws'
PY = str(REPO / '.venv' / 'Scripts' / 'python.exe')
REAL = '--real' in sys.argv
out_lines = []


def P(name, ok, extra=''):
    out_lines.append(('PASS' if ok else 'FAIL') + ' | ' + name + ((' | ' + extra) if extra else ''))


def req(path, method='GET', payload=None, token='', timeout=60, allow_204=False):
    r = urllib.request.Request(BASE + path, method=method)
    r.add_header('Content-Type', 'application/json')
    if token:
        r.add_header('X-Node-Token', token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(r, data=data, timeout=timeout) as resp:
            b = resp.read()
            return resp.status, (json.loads(b) if b else None)
    except urllib.error.HTTPError as e:
        b = e.read()
        try:
            return e.code, json.loads(b)
        except Exception:
            return e.code, {'raw': b.decode('utf-8', 'replace')[:200]}


def do_cycle(tag, model_env, tm_id='TM-04'):
    # 1 claim（定向 tm_id：顺带覆盖定向领取路径；v2 runner 的 summary 才带 analysis 协议键）
    st, task = req('/api/tasks/claim', 'POST', {'model_id': 'qwen3-4b', 'gpu': 'RTX 5080', 'dtype': 'bf16',
                                                'tm_id': tm_id},
                   token=TOKEN)
    P(f'[{tag}] claim({tm_id})', st == 200 and task and task.get('task_id') and task.get('tm_id') == tm_id,
      f"{task.get('tm_id')} design_sha={task.get('design_sha','')[:12]}")
    tdir = WS / task['task_id']
    (tdir / 'bundle').mkdir(parents=True, exist_ok=True)
    (tdir / 'outputs').mkdir(parents=True, exist_ok=True)
    for name, content in task['files'].items():
        (tdir / 'bundle' / name).write_text(content, encoding='utf-8')
    # 2 heartbeat
    st, hb = req(f"/api/tasks/{task['task_id']}/heartbeat", 'PUT', {}, token=TOKEN)
    P(f'[{tag}] heartbeat', st == 200 and hb.get('lease_until', 0) > time.time())
    # 3 runner
    env = dict(os.environ)
    env.pop('AI2050_RUNNER_MODEL', None)
    if model_env:
        env['AI2050_RUNNER_MODEL'] = model_env
    t0 = time.time()
    proc = subprocess.run([PY, str(tdir / 'bundle' / 'runner.py'), '--out', str(tdir / 'outputs')],
                          cwd=str(tdir), env=env, capture_output=True, text=True, timeout=1800)
    P(f'[{tag}] runner exit0', proc.returncode == 0, f"{time.time()-t0:.1f}s")
    summary = json.loads((tdir / 'outputs' / 'summary.json').read_text(encoding='utf-8'))
    kind = summary.get('status')
    P(f'[{tag}] runner status', kind == ('real' if model_env else 'smoke'),
      f"status={kind} err={summary.get('real_error', '')[:60]}")
    # 4 manifest + upload
    files = {f.name: f.read_bytes() for f in sorted((tdir / 'outputs').iterdir()) if f.is_file()}
    manifest = [{'path': n, 'sha256': hashlib.sha256(b).hexdigest(), 'size': len(b)} for n, b in files.items()]
    st, init = req('/api/results/upload/init', 'POST',
                   {'task_id': task['task_id'], 'tm_id': task['tm_id'], 'design_sha': task['design_sha'],
                    'model_id': 'qwen3-4b', 'model_rev': 'local', 'seed': 0, 'kind': kind,
                    'summary': summary, 'manifest': manifest}, token=TOKEN)
    P(f'[{tag}] upload init', st == 200, f"chunks={init.get('chunks_expected')}")
    table = json.dumps(list(files.keys()), ensure_ascii=False).encode()
    blob = len(table).to_bytes(4, 'big') + table + b''.join(files.values())
    for seq, i in enumerate(range(0, len(blob), 1_500_000)):
        st, _ = req(f"/api/results/upload/{init['upload_id']}/chunk", 'POST',
                    {'seq': seq, 'data_b64': base64.b64encode(blob[i:i + 1_500_000]).decode()}, token=TOKEN)
    st, fin = req(f"/api/results/upload/{init['upload_id']}/finish", 'POST', {}, token=TOKEN)
    P(f'[{tag}] upload finish', st == 200 and fin.get('sha'), f"sha={fin.get('sha', '')[:16]}")
    # 5 task complete
    st, _ = req(f"/api/tasks/{task['task_id']}/complete", 'POST',
                {'status': 'done', 'result_sha': fin['sha']}, token=TOKEN)
    P(f'[{tag}] complete', st == 200)
    return fin['sha'], files


# ===== 0 健康与模板表 =====
st, root = req('/')
P('health /', st == 200 and 'ai2050-distributed' in str(root), str(root)[:60])
st, tpls = req('/api/templates')
P('templates seeded (>=5, TM-04/05 included)', st == 200 and len(tpls.get('templates', [])) >= 3
  and {'TM-04', 'TM-05'} <= {t['tm_id'] for t in tpls.get('templates', [])},
  ' '.join(t['tm_id'] for t in tpls.get('templates', [])))

# ===== 1 注册 =====
st, reg = req('/api/nodes/register', 'POST',
              {'name': 'e2e-box', 'gpu': 'RTX 5080', 'model': 'qwen3-4b', 'dtype': 'bf16'})
TOKEN = reg['node_token']
P('register', st == 201 and reg.get('node_id', '').startswith('NODE-'), reg.get('node_id', ''))

# ===== 2 冒烟闭环 =====
sha1, files1 = do_cycle('smoke', None)
P('[smoke] has summary.json only (no means.npz)', 'means.npz' not in files1,
  ','.join(sorted(files1)))

# ===== 3 负路径：design_sha 篡改 → 422 =====
st, task2 = req('/api/tasks/claim', 'POST', {}, token=TOKEN)
bad = dict(task2)
st2, err = req('/api/results/upload/init', 'POST',
               {'task_id': task2['task_id'], 'tm_id': task2['tm_id'],
                'design_sha': 'f' * 64, 'kind': 'smoke', 'summary': {},
                'manifest': [{'path': 'x.json', 'sha256': 'a' * 64, 'size': 1}]}, token=TOKEN)
P('negative design_sha -> 422', st2 == 422, str(err.get('detail', ''))[:80])
req(f"/api/tasks/{task2['task_id']}/complete", 'POST', {'status': 'failed', 'note': 'negative test'}, token=TOKEN)

# ===== 4 真实模式（GPU 采集 qwen3-4b） =====
if REAL:
    MODEL = str(REPO / 'models' / 'hf' / 'qwen3-4b')
    sha2, files2 = do_cycle('real', MODEL)
    P('[real] has means.npz', 'means.npz' in files2, ','.join(sorted(files2)))

# ===== 5 下载核验 =====
st, dl = req(f'/api/results/{sha1}')
back = {k: base64.b64decode(v) for k, v in dl.get('files', {}).items()}
P('download roundtrip', st == 200 and back == files1,
  f"{len(back)} files, downloads={dl.get('downloads')}")

# ===== 6 公开面 =====
st, summ = req('/api/distributed/summary')
P('summary', st == 200 and summ['results_total'] >= 1 and summ['nodes_online'] >= 1,
  f"nodes_online={summ['nodes_online']} results={summ['results_total']} dl={summ['downloads_total']}")
st, agg = req('/api/agg/TM-01')
P('agg v0 bucket', st == 200 and len(agg.get('buckets', {})) >= 1,
  ' | '.join(list(agg.get('buckets', {}))[:3]))

# ===== 6b D1-1/D1-3（ui_decoupled_plan_v1：协议驱动 UI 数据缺口） =====
st, td = req('/api/templates/TM-04')
m = td.get('meta', {})
P('D1-1 template detail', st == 200 and m.get('analysis') == 'factorial'
  and m.get('factors') == ['topic', 'frame'] and m.get('items') == 16 and m.get('fingerprint') is True,
  f"analysis={m.get('analysis')} factors={m.get('factors')} items={m.get('items')} layers={m.get('layers')}")
st, td5 = req('/api/templates/TM-05')
P('D1-1 detail TM-05 factors', st == 200 and td5.get('meta', {}).get('factors') == ['entity', 'position'],
  str(td5.get('meta', {}).get('factors')))
st, _ = req('/api/templates/TM-99')
P('D1-1 negative 404', st == 404)
st, rl = req('/api/results?tm_id=TM-04')
r0 = (rl.get('results') or [{}])[0]
dg = r0.get('summary_digest', {})
P('D1-3 results digest (TM-04 v2 runner)', st == 200 and isinstance(dg, dict)
  and 'status' in dg and 'analysis' in dg and dg.get('analysis') == 'factorial',
  f"digest_keys={sorted(dg)[:6]}")

st, news = req('/api/news')
P('news fallback (offline)', st == 200 and len(news.get('items', [])) >= 5, news.get('source', ''))

# ===== 7 M3 端点（ui_decoupled_plan_v2：主后端 :5001 的 queue / workspace / object） =====
B1 = 'http://127.0.0.1:5001'


def req1(path, timeout=20):
    try:
        with urllib.request.urlopen(B1 + path, timeout=timeout) as resp:
            b = resp.read()
            return resp.status, (json.loads(b) if b else None)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception:
        return 0, {}


st, q = req1('/api/ai-rnd/queue')
P('M3 queue (phase_queue_v1)', st == 200 and q.get('count', 0) >= 30 and q.get('sealed', 0) >= 7
  and any(i.get('id') == 'Q06' and i.get('status') == 'sealed' for i in q.get('queue', [])),
  f"count={q.get('count')} sealed={q.get('sealed')} source={q.get('source')}")
st, wsj = req1('/api/ai-rnd/workspace?path=tests/deepseek')
P('M3 workspace list', st == 200 and len(wsj.get('dirs', [])) >= 5
  and 'result' in [d['name'] for d in wsj.get('dirs', [])],
  f"dirs={len(wsj.get('dirs', []))} files={len(wsj.get('files', []))}")
st, wsf = req1('/api/ai-rnd/workspace/file?path=tests/deepseek/result/../../../server/server.py')
P('M3 workspace escape blocked (403)', st == 403, str(st))
st, wso = req1('/api/ai-rnd/workspace?path=deploy/distributed_service.py')
P('M3 workspace outside roots (403)', st == 403, str(st))
st, objs = req1('/api/objects')
P('M3 objects list', st == 200 and objs.get('count', 0) >= 1
  and 'F#3734' in [o.get('id') for o in objs.get('objects', [])],
  f"count={objs.get('count')}")
st, ob = req1('/api/object/F%233734')
obm = (ob.get('object') or {}).get('metrics', [])
P('M3 object detail (registered+enriched)', st == 200 and ob.get('registered') is True
  and len(obm) >= 2 and len(ob.get('related_results', [])) >= 1
  and (ob.get('related_results') or [{}])[0].get('tm_id') == 'TM-04',
  f"metrics={len(obm)} related={len(ob.get('related_results', []))}")
st, _ = req1('/api/object/F%239999')
P('M3 object unregistered (404)', st == 404, str(st))

report = '\n'.join(out_lines)
(REPO / 'tests' / 'dist_test_data' / 'e2e_report.txt').write_text(report, encoding='utf-8')
print(report)
fails = [l for l in out_lines if l.startswith('FAIL')]
print(f"TOTAL {'PASS' if not fails else 'FAIL'}: {len(out_lines) - len(fails)}/{len(out_lines)}")
