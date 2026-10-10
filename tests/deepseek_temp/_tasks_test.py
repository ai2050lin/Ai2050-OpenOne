# -*- coding: utf-8 -*-
"""M6-P2 S8 调度队列只读端点测试：临时数据目录 + 真实 claim/heartbeat/complete 全链路，结果写报告文件。"""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(r'D:\AI2050\Ai2050-OpenOne')
TMP = Path(tempfile.mkdtemp(prefix='ai2050_tasks_test_'))
os.environ['AI2050_DIST_DIR'] = str(TMP)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'deploy'))

import distributed_service as ds       # noqa: E402
from fastapi import FastAPI            # noqa: E402
import uvicorn                         # noqa: E402

app = FastAPI()
app.include_router(ds.router)   # router 自带 prefix='/api'
srv = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=5097, log_level='error'))
threading.Thread(target=srv.run, daemon=True).start()
for _ in range(60):
    try:
        urllib.request.urlopen('http://127.0.0.1:5097/api/tasks', timeout=1)
        break
    except Exception:
        time.sleep(0.2)

BASE = 'http://127.0.0.1:5097/api'
out = []
def ck(name, cond, extra=''):
    out.append(f"{'PASS' if cond else 'FAIL'} {name} {extra}")
    if not cond:
        out.append('!!! FAIL !!!')

def call(path, method='GET', payload=None, token=''):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('X-Node-Token', token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=10) as r:
            return r.status, (json.loads(r.read().decode('utf-8')) if r.status != 204 else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', 'replace')[:120]

# 1) 空队列
st, q = call('/tasks')
ck('GET /api/tasks 200', st == 200)
ck('零任务 stats 全 0', q['stats'] == {}, str(q['stats']))
ck('租约时长 6h', q['lease_hours'] == 6.0)
ck('无需凭据（公开只读）', 'token' not in json.dumps(q).lower())

# 2) 注册节点 + claim → 活跃租约可见
st, reg = call('/nodes/register', 'POST',
               {'name': 'node-testq', 'gpu': 'RTX-test', 'model': 'Qwen3-4B', 'dtype': 'bf16'})
ck('register 201', st == 201)
tok, nid = reg['node_token'], reg['node_id']
st, cl = call('/tasks/claim', 'POST', {'model_id': 'Qwen3-4B'}, token=tok)
ck('claim 200', st == 200, cl.get('tm_id', '') if st == 200 else str(cl))
tm_id = cl.get('tm_id', '')
st, q = call('/tasks')
ck('活跃租约 1', len(q['active']) == 1, f"tm={q['active'][0]['tm_id'] if q['active'] else '-'}")
ck('active 行含 node_name/model', q['active'][0]['node_name'] == 'node-testq'
   and q['active'][0]['model'] == 'Qwen3-4B')
ck('stats.claimed=1', q['stats'].get('claimed') == 1)
ck('by_template 有该模板 active=1', any(r['tm_id'] == tm_id and r['active'] == 1 for r in q['by_template']))
ck('租约剩余为正', q['active'][0]['lease_until'] > q['server_time'])
ck('无 token 泄漏', 'node_token' not in json.dumps(q) and tok not in json.dumps(q))

# 3) heartbeat 续租 → complete → recent
st, _ = call(f"/tasks/{cl['task_id']}/heartbeat", 'PUT', token=tok)
ck('heartbeat 200', st == 200)
st, _ = call(f"/tasks/{cl['task_id']}/complete", 'POST',
             {'status': 'done', 'note': 'test-ok', 'result_sha': 'a' * 64}, token=tok)
ck('complete 200', st == 200)
st, q = call('/tasks')
ck('完成后活跃 0', len(q['active']) == 0)
ck('stats done=1', q['stats'].get('done') == 1, str(q['stats']))
ck('recent 含 task 与 result_sha', len(q['recent']) == 1
   and q['recent'][0]['task_id'] == cl['task_id'] and q['recent'][0]['result_sha'] == 'a' * 64)
ck('by_template done=1', any(r['tm_id'] == tm_id and r['done'] == 1 for r in q['by_template']))

# 4) 租约过期清移（把 lease_until 改到过去 → GET /tasks 应转为 expired）
db = ds._db()
db.execute('UPDATE tasks SET status=?, lease_until=? WHERE task_id=?', ('claimed', time.time() - 10, cl['task_id']))
db.commit()
st, q = call('/tasks')
ck('过期自动清移', q['stats'].get('expired') == 1 and len(q['active']) == 0, str(q['stats']))

Path(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_tasks_test_report.txt').write_text(
    '\n'.join(out) + f"\nTOTAL PASS={sum(1 for l in out if l.startswith('PASS'))} "
    f"FAIL={sum(1 for l in out if l.startswith('FAIL'))}\n", encoding='utf-8')
print('done')
