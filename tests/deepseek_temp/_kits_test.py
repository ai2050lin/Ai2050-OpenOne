# -*- coding: utf-8 -*-
"""M6-P1 S7 研究包端点测试：临时数据目录 + 内存 sqlite，全场景断言，结果写报告文件。"""
import json
import os
import sys
import tempfile
import threading
import time
import hashlib
import urllib.request
from pathlib import Path

REPO = Path(r'D:\AI2050\Ai2050-OpenOne')
TMP = Path(tempfile.mkdtemp(prefix='ai2050_kits_test_'))
os.environ['AI2050_DIST_DIR'] = str(TMP)
sys.path.insert(0, str(REPO))          # server.dist_templates_seed
sys.path.insert(0, str(REPO / 'deploy'))  # distributed_service

import distributed_service as ds       # noqa: E402
from fastapi import FastAPI            # noqa: E402
import uvicorn                         # noqa: E402

app = FastAPI()
app.include_router(ds.router)   # router 自带 prefix='/api'
srv = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=5099, log_level='error'))
threading.Thread(target=srv.run, daemon=True).start()
for _ in range(60):
    try:
        urllib.request.urlopen('http://127.0.0.1:5099/api/kits', timeout=1)
        break
    except Exception:
        time.sleep(0.2)

BASE = 'http://127.0.0.1:5099/api'
out = []
def ck(name, cond, extra=''):
    out.append(f"{'PASS' if cond else 'FAIL'} {name} {extra}")
    if not cond:
        out.append('!!! FAIL !!!')

def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as r:
            return r.status, json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', 'replace')[:120]

# 0) 种子模板存在
st, kits = get('/kits')
tms = sorted({k['tm_id'] for k in kits['kits']})
ck('GET /api/kits 200', st == 200, f'{len(kits["kits"])} kits · tms={tms[:4]}...')
ck('零结果全 pending', all(k['status'] == 'pending_results' for k in kits['kits']))

# 1) 注入一份同时满足 rsa-rdm（npz manifest）与 eta2（eta2_by_factor）的结果
db = ds._db()
tm = tms[0]
summary = {'status': 'real', 'eta2_by_factor': {'factorA': 0.42, 'factorB': 0.11},
           '_manifest': [{'path': 'means.npz', 'sha256': 'a' * 64, 'size': 2048}]}
db.execute('INSERT INTO results(sha,tm_id,version,node_id,model_id,kind,size,summary_json,uploaded_at) '
           'VALUES(?,?,?,?,?,?,?,?,?)',
           ('f' * 64, tm, 1, 'N-test', 'test-model', 'real', 2048,
            json.dumps(summary), time.time()))
db.commit()

st, kits = get('/kits')
km = {k['kit_id']: k for k in kits['kits']}
ck('rsa-rdm available', km[f'{tm}__rsa-rdm']['status'] == 'available')
ck('eta2 available', km[f'{tm}__eta2-decompose']['status'] == 'available')
ck('cka pending（单结果）', km[f'{tm}__cka-linear']['status'] == 'pending_results')
ck('其他模板仍 pending', km[f'{tms[1]}__rsa-rdm']['status'] == 'pending_results')

# 2) bundle 下载 + design_sha 本地复核
st, b = get(f'/kits/{tm}__rsa-rdm/bundle')
ck('bundle 200', st == 200, f"files={sorted((b.get('files') or {}).keys()) if st == 200 else b}")
if st == 200:
    contract = json.loads(b['files']['contract.json'])
    corpus = json.loads(b['files']['corpus.json'])
    runner = b['files']['runner.py']
    design = hashlib.sha256(json.dumps(
        {'tm_id': contract['tm_id'], 'version': contract['version'], 'corpus': corpus,
         'runner_sha256': hashlib.sha256(runner.encode()).hexdigest()},
        sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    ck('design_sha 本地复核一致', design == contract['design_sha'] == b['template']['design_sha'],
       design[:12])
    ck('README 生成', b['README.md'].startswith(f'# AI2050 研究包 {tm}__rsa-rdm'))
    ck('结果清单 1 条', len(b['results_manifest']) == 1 and b['results_manifest'][0]['sha'] == 'f' * 64)

# 3) 负路径
st, _ = get('/kits/NOPE__rsa-rdm/bundle')
ck('未知模板 404', st == 404)
st, _ = get('/kits')
bad = [k for k in kits['kits'] if k['kit_id'].endswith('__nope')]
st2, _ = get('/kits/' + tms[0] + '__nope/bundle')
ck('未知分析 404', st2 == 404)
st3, body3 = get(f'/kits/{tms[1]}__rsa-rdm/bundle')
ck('pending 模板 bundle 409', st3 == 409, str(body3)[:60])
st4, _ = get('/kits/badformat/bundle')
ck('坏 kit_id 400', st4 == 400)

Path(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_kits_test_report.txt').write_text(
    '\n'.join(out) + f'\nTOTAL PASS={sum(1 for l in out if l.startswith("PASS"))} '
    f'FAIL={sum(1 for l in out if l.startswith("FAIL"))}\n', encoding='utf-8')
print('done')
