# -*- coding: utf-8 -*-
"""P0 安全改动冒烟：ADMIN_TOKEN 默认值 + /api/grammar/preview 鉴权（TestClient，不加载模型）。"""
import os
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # tests/gpt5_temp → 仓库根
os.chdir(BASE)                    # sqlite/_DATA_DIR 相对路径以仓库根为基准
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'deploy'))   # distributed_service 平铺导入同目录模块

ok = 0
total = 0

def check(name, cond, detail=''):
    global ok, total
    total += 1
    ok += bool(cond)
    print(('PASS' if cond else 'FAIL'), name, detail)

# ── 1) 未设置环境变量：默认值不再是可猜串，而是随机 32 hex ──
os.environ.pop('AI2050_DIST_ADMIN_TOKEN', None)
import importlib
import distributed_service as dsvc      # 与 server.py 相同的平铺导入（BASE/deploy 在 sys.path）
importlib.reload(dsvc)
check('admin_token.not_default', dsvc.ADMIN_TOKEN != 'ai2050-admin-dev',
      f"len={len(dsvc.ADMIN_TOKEN)}")
check('admin_token.random_hex32', len(dsvc.ADMIN_TOKEN) == 32 and all(c in '0123456789abcdef' for c in dsvc.ADMIN_TOKEN))

# ── 2) preview 鉴权：无令牌/旧开发令牌 401；正确令牌通过（非 401）──
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ['AI2050_DIST_ADMIN_TOKEN'] = 'p0-smoke-token-12345'
importlib.reload(dsvc)
check('admin_token.env_respected', dsvc.ADMIN_TOKEN == 'p0-smoke-token-12345')

app = FastAPI()
app.include_router(dsvc.router)
client = TestClient(app)

r = client.post('/api/grammar/preview', json={'spec': {}})
check('preview.no_token_401', r.status_code == 401, f"got {r.status_code}")

r = client.post('/api/grammar/preview', json={'spec': {}}, headers={'X-Admin-Token': 'ai2050-admin-dev'})
check('preview.old_dev_token_401', r.status_code == 401, f"got {r.status_code}")

r = client.post('/api/grammar/preview', json={'spec': {}}, headers={'X-Admin-Token': 'p0-smoke-token-12345'})
check('preview.good_token_not_401', r.status_code != 401, f"got {r.status_code}（鉴权通过，spec 校验层可自返回 4xx）")

# ── 3) 回归：公开读端点不受影响 ──
r = client.get('/api/catalog')
check('catalog.still_200', r.status_code == 200, f"got {r.status_code}")

r = client.get('/api/grammar/library')
check('grammar_library.still_200', r.status_code == 200, f"got {r.status_code}")

print(f"\n== p0 backend smoke: {ok}/{total} PASS ==")
sys.exit(0 if ok == total else 1)
