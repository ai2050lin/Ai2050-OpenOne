# -*- coding: utf-8 -*-
"""S9 实时模型连接端点测试（tests/deepseek_temp/_live_test.py）
临时目录手工构造 config.json + 迷你 safetensors 文件（手工写 header，不依赖 torch），
起 uvicorn 验证：connect（成功/404/400）→ status → disconnect → 负路径。
结果写 _live_test_report.txt（本机 bash shim stdout 不稳，一律写文件再 Read）。"""
import json
import struct
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(r'D:\AI2050\Ai2050-OpenOne')
TMP = Path(tempfile.mkdtemp(prefix='ai2050_live_'))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'deploy'))

# ── 构造迷你模型目录 ──────────────────────────────────────────────
model_dir = TMP / 'mini-model'
model_dir.mkdir()
(model_dir / 'config.json').write_text(json.dumps({
    'architectures': ['Qwen3ForCausalLM'], 'num_hidden_layers': 2, 'hidden_size': 8,
    'num_attention_heads': 2, 'num_key_value_heads': 1, 'vocab_size': 100,
    'tie_word_embeddings': False, 'torch_dtype': 'bfloat16'}), encoding='utf-8')

def write_st(path, tensors):
    """tensors: {name: (dtype, shape, nbytes)}；手工按 safetensors 格式落盘。"""
    hdr, off = {}, 0
    for name, (dt, shape, nb) in tensors.items():
        hdr[name] = {'dtype': dt, 'shape': shape, 'data_offsets': [off, off + nb]}
        off += nb
    hdr['__metadata__'] = {'format': 'pt'}          # 测 parser 的 __metadata__ 剔除
    hb = json.dumps(hdr, separators=(',', ':')).encode('utf-8')
    pad = (8 - len(hb) % 8) % 8                     # 8 字节对齐（真实格式要求）
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hb) + pad) + hb + b'\x20' * pad + b'\x00' * off)

def n(st_shape):
    p = 1
    for d in st_shape:
        p *= d
    return p * 2                                    # bf16=2B

write_st(model_dir / 'model-00001-of-00002.safetensors', {
    'model.embed_tokens.weight': ('BF16', [100, 8], n([100, 8])),
    'model.layers.0.self_attn.q_proj.weight': ('BF16', [8, 8], n([8, 8])),
    'model.layers.0.mlp.down_proj.weight': ('BF16', [8, 8], n([8, 8])),
    'model.layers.0.input_layernorm.weight': ('BF16', [8], n([8])),
})
write_st(model_dir / 'model-00002-of-00002.safetensors', {
    'model.layers.1.self_attn.q_proj.weight': ('BF16', [8, 8], n([8, 8])),
    'model.norm.weight': ('BF16', [8], n([8])),
})
empty_dir = TMP / 'empty'; empty_dir.mkdir()

# ── 起服务 ───────────────────────────────────────────────────────
import os
os.environ['AI2050_DIST_DIR'] = str(TMP / 'data')
import distributed_service as ds
from fastapi import FastAPI
import uvicorn
app = FastAPI()
app.include_router(ds.router)                       # router 自带 prefix='/api'
srv = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=5096, log_level='error'))
threading.Thread(target=srv.run, daemon=True).start()
time.sleep(1.5)
BASE = 'http://127.0.0.1:5096'

REPORT = []
def ck(name, ok, extra=''):
    REPORT.append(('PASS' if ok else 'FAIL') + ' | ' + name + ((' | ' + str(extra)) if extra else ''))

def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.status, json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode('utf-8'))
        except Exception:
            return e.code, {}

# ── 用例 ────────────────────────────────────────────────────────
st0, s0 = call('GET', '/api/live/status')
ck('初始 status 未连接', st0 == 200 and s0.get('connected') is False, st0)

st1, r1 = call('POST', '/api/live/connect', {'model_path': str(TMP / 'nope')})
ck('不存在目录 → 404', st1 == 404, st1)

st2, r2 = call('POST', '/api/live/connect', {'model_path': str(empty_dir)})
ck('无 safetensors → 400', st2 == 400 and 'safetensors' in str(r2.get('detail', '')), st2)

st3, m = call('POST', '/api/live/connect', {'model_path': str(model_dir)})
ck('合法目录 → 200', st3 == 200, st3)
ck('arch 读自 config.json', m.get('arch') == 'Qwen3ForCausalLM', m.get('arch'))
cfg = m.get('config') or {}
ck('config 字段', cfg.get('layers') == 2 and cfg.get('hidden') == 8 and cfg.get('kv_heads') == 1, cfg)
ck('weight 文件数 2', len(m.get('files') or []) == 2, m.get('files'))
ck('总参数量', m.get('total_params') == 100 * 8 + 8 * 8 + 8 + 8 * 8 + 8 * 8 + 8,
   m.get('total_params'))
ck('dtype 归一 bf16', list((m.get('dtypes') or {}).keys()) == ['bf16'], m.get('dtypes'))
groups = {g['prefix']: g for g in (m.get('groups') or [])}
ck('分组 layers.0', 'model.layers.0' in groups and groups['model.layers.0']['tensor_count'] == 3,
   list(groups))
ck('分组 embed', 'model.embed_tokens' in groups, list(groups))
ck('__metadata__ 被剔除', all('__metadata__' not in g['prefix'] for g in groups.values()), list(groups))
q = [t for t in groups.get('model.layers.0', {}).get('tensors', []) if t['name'].endswith('q_proj.weight')]
ck('tensor 明细 shape/dtype/params', q and q[0]['shape'] == [8, 8] and q[0]['dtype'] == 'bf16' and q[0]['params'] == 64, q)

st4, s4 = call('GET', '/api/live/status')
ck('status 恢复连接（含 groups）', st4 == 200 and s4.get('connected') is True and 'groups' in s4, st4)

st5, r5 = call('POST', '/api/live/disconnect')
ck('disconnect → 200', st5 == 200 and r5.get('connected') is False, st5)
st6, s6 = call('GET', '/api/live/status')
ck('disconnect 后 status 未连接', s6.get('connected') is False, s6)

st7, r7 = call('POST', '/api/live/connect', {'model_path': ''})
ck('空路径 → 400（cwd 无 safetensors）', st7 == 400, st7)

fails = [r for r in REPORT if r.startswith('FAIL')]
(TMP / 'report').mkdir(exist_ok=True)
out = REPO / 'tests' / 'deepseek_temp' / '_live_test_report.txt'
out.write_text('\n'.join(REPORT) + f'\n\nTOTAL PASS={len(REPORT) - len(fails)} FAIL={len(fails)}\nTMP={TMP}\n', encoding='utf-8')
print('DONE', len(REPORT))
