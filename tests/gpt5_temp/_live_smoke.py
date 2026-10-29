# -*- coding: utf-8 -*-
"""实时分析后端冒烟（真实前向，Qwen3-4B bf16 GPU）：status → run → block。
只挂 live_analysis_service 一个路由，不起完整 server.py（其 lifespan 会加载 AGI 引擎）。
运行：D:/AI2050/Ai2050-OpenOne/.venv/Scripts/python.exe tests/gpt5_temp/_live_smoke.py
"""
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'server'))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from live_analysis_service import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

ok = 0


def check(name, cond, extra=''):
    global ok
    if cond:
        ok += 1
        print(f'[PASS] {name} {extra}')
    else:
        print(f'[FAIL] {name} {extra}')
        raise SystemExit(f'smoke failed at {name}')


# 1) status 触发后台加载并轮询到 ready
t0 = time.time()
st = client.get('/api/live/status').json()
check('status.available', st.get('available') is True, f"model_name={st.get('model_name')}")
for _ in range(120):
    st = client.get('/api/live/status').json()
    if st.get('status') == 'ready':
        break
    if st.get('status') == 'error':
        raise SystemExit(f"load error: {st.get('error')}")
    time.sleep(2)
check('status.ready', st.get('status') == 'ready', f"{time.time() - t0:.0f}s device={st.get('device')}")
meta = st.get('meta') or {}
check('meta.qwen3-4b', meta.get('n_layers') == 36 and meta.get('n_heads') == 32 and meta.get('d_ff') == 9728, json.dumps(meta))

# 2) run：真实前向
r = client.post('/api/live/run', json={'text': 'The capital of France is', 'max_tokens': 8})
check('run.http200', r.status_code == 200, r.text[:200] if r.status_code != 200 else '')
doc = r.json()
check('run.schema', doc['schema'] == 'live_trace.v1', f"tokens={doc['tokens']}")
check('run.n_tokens', doc['n_tokens'] == len(doc['tokens']) and 0 < doc['n_tokens'] <= 8, f"truncated={doc['truncated']} n={doc['n_tokens']}")
T = doc['n_tokens']
check('run.layers', len(doc['layers']) == 36)
l0 = doc['layers'][0]
check('l0.heads', len(l0['heads']['out_norm']) == 32 and len(l0['heads']['out_norm'][0]) == T)
check('l0.attn_last', len(l0['heads']['attn_last']) == 32 and all(abs(sum(row) - 1) < 1e-2 for row in l0['heads']['attn_last']),
      f"worst={max(abs(sum(r) - 1) for r in l0['heads']['attn_last']):.4f}")
check('l0.top_pairs', len(l0['heads']['top_pairs']) == 32 and len(l0['heads']['top_pairs'][0]) == 3)
check('l0.mlp_top', len(l0['mlp']['top']) == T and len(l0['mlp']['top'][0]) == 16)
check('l0.resid_in', len(l0['resid_in']) == T)
out = doc['output']
check('out.last_top', len(out['last_top']) == 8, f"top1={out['last_top'][0][0]!r} p={out['last_top'][0][1]}")
check('out.per_pos_top1', len(out['per_pos_top1']) == T, ' '.join(repr(t) for t in out['per_pos_top1']))

# 中文输入
r2 = client.post('/api/live/run', json={'text': '天空为什么是蓝色的', 'max_tokens': 12})
check('run.zh', r2.status_code == 200 and r2.json()['n_tokens'] >= 1, f"n={r2.json().get('n_tokens')}")

# 3) block 参数缩略图
rb = client.get('/api/live/block', params={'block': 5})
check('block.http200', rb.status_code == 200, rb.text[:200] if rb.status_code != 200 else '')
bd = rb.json()
check('block.schema', bd['schema'] == 'live_block.v1' and bd['block'] == 5)
check('block.heads', len(bd['heads']) == 32 and len(bd['heads'][0]['q']) == 8 and len(bd['heads'][0]['q'][0]) == 8)
check('block.mlp', len(bd['mlp']['gate']) == 8 and len(bd['mlp']['gate'][0]) == 16)
rb2 = client.get('/api/live/block', params={'block': 99})
check('block.404', rb2.status_code == 404)

print(f'\nSMOKE OK {ok} checks')
