# -*- coding: utf-8 -*-
"""M6-P3 一键复现测试：临时数据目录起服务 → 注入满足契约的结果 → 下载 bundle（含 reproduce.py / agg_v1）
→ 落盘真跑 reproduce.py（design_sha 校验 → runner 冒烟 → 对账报告）。结果写报告文件。"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(r'D:\AI2050\Ai2050-OpenOne')
TMP = Path(tempfile.mkdtemp(prefix='ai2050_repro_test_'))
os.environ['AI2050_DIST_DIR'] = str(TMP)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'deploy'))

import distributed_service as ds       # noqa: E402
from fastapi import FastAPI            # noqa: E402
import uvicorn                         # noqa: E402

app = FastAPI()
app.include_router(ds.router)   # router 自带 prefix='/api'
srv = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=5096, log_level='error'))
threading.Thread(target=srv.run, daemon=True).start()
for _ in range(60):
    try:
        urllib.request.urlopen('http://127.0.0.1:5096/api/kits', timeout=1)
        break
    except Exception:
        time.sleep(0.2)

BASE = 'http://127.0.0.1:5096/api'
PY = str(REPO / '.venv' / 'Scripts' / 'python.exe')
out = []
def ck(name, cond, extra=''):
    out.append(f"{'PASS' if cond else 'FAIL'} {name} {extra}")
    if not cond:
        out.append('!!! FAIL !!!')

def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=20) as r:
            return r.status, json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', 'replace')[:120]

# 1) 注入一份同时满足 rsa-rdm 与 eta2 契约的结果（TM-01）
db = ds._db()
summary = {'status': 'real', 'eta2_by_factor': {'topic': 0.42, 'frame': 0.11},
           '_manifest': [{'path': 'means.npz', 'sha256': 'a' * 64, 'size': 2048}]}
db.execute('INSERT INTO results(sha,tm_id,version,node_id,model_id,kind,seed,size,summary_json,uploaded_at) '
           'VALUES(?,?,?,?,?,?,?,?,?,?)',
           ('e' * 64, 'TM-01', 1, 'N-test', 'test-model', 'real', 0, 2048,
            json.dumps(summary), time.time()))
db.commit()

# 2) AGG-v1 端点
st, a1 = get('/agg/TM-01/v1')
ck('GET /agg/{tm}/v1 200', st == 200)
ck('AGG-v1 桶内 η² 汇总', st == 200 and a1.get('buckets_total') == 1
   and 'test-model|real|seed0' in a1.get('buckets', {}),
   str(list((a1.get('buckets') or {}).keys())) if st == 200 else str(a1))
if st == 200 and a1['buckets']:
    per = list(a1['buckets'].values())[0]['eta2_by_factor']
    ck('η² 因子 mean 正确', abs(per['topic']['mean'] - 0.42) < 1e-9 and per['topic']['n'] == 1
       and per['frame']['std'] == 0.0, f"topic={per.get('topic')} frame={per.get('frame')}")
    ck('纪律 warning 在场', '跨桶平均被禁止' in a1.get('warning', ''))
st, a0 = get('/agg/TM-01')
ck('AGG-v0 原端点不回归', st == 200 and a0.get('agg_version') == 'AGG-v0')

# 3) bundle：reproduce.py + agg_v1 + README
st, b = get('/kits/TM-01__rsa-rdm/bundle')
ck('bundle 200 · S7-v1', st == 200 and b.get('bundle_version') == 'S7-v1' if st == 200 else False, str(st))
ck('包内含 reproduce.py', st == 200 and 'reproduce.py' in b.get('files', {}))
ck('包内含 agg_v1', st == 200 and isinstance(b.get('agg_v1'), dict)
   and b['agg_v1'].get('agg_version') == 'AGG-v1')
ck('README 含一键复现步骤', st == 200 and 'python reproduce.py' in b.get('README.md', ''))

# 4) 落盘真跑 reproduce.py（design_sha → runner 冒烟 → 对账）
kit_dir = TMP / 'kit_tm01'
kit_dir.mkdir()
for name, content in b['files'].items():
    p = kit_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content.encode('utf-8'))  # 字节写入：禁 CRLF 翻译（runner sha 完整性）
(kit_dir / 'results_manifest.json').write_text(
    json.dumps(b['results_manifest'], ensure_ascii=False, indent=1), encoding='utf-8')
proc = subprocess.run([PY, str(kit_dir / 'reproduce.py')], cwd=str(kit_dir),
                      capture_output=True, text=True, timeout=600)
ck('reproduce.py exit0', proc.returncode == 0, (proc.stderr or '')[-150:] if proc.returncode else '')
rep = (kit_dir / 'reproduce_report.txt')
ck('reproduce_report.txt 生成', rep.is_file())
if rep.is_file():
    txt = rep.read_text(encoding='utf-8')
    ck('design_sha 冻结校验 OK', 'design_sha 冻结校验: OK' in txt)
    ck('runner 复现步骤在场', '[2] runner 复现' in txt)
    ck('对账步骤在场', '[3] 输出对账' in txt and '[4] 结论' in txt)

Path(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_repro_test_report.txt').write_text(
    '\n'.join(out) + f'\nTOTAL PASS={sum(1 for l in out if l.startswith("PASS"))} '
    f'FAIL={sum(1 for l in out if l.startswith("FAIL"))}\n', encoding='utf-8')
print('done')
