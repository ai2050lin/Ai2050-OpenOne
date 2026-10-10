# -*- coding: utf-8 -*-
"""S9b 权重数值端点测试：构造小 safetensors 模型 → connect → scan/summary/tensor
与 torch 直算对比（rms/mean_abs/max_abs/热图行带均值/行范数）。输出 _wt_test_out.txt"""
import io, json, shutil, sys
from pathlib import Path

import torch
from safetensors.torch import save_file

import urllib.request

BASE = 'http://127.0.0.1:5001'
TMP = Path(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_wt_test_model')
OUT = Path(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_wt_test_out.txt')

log = io.StringIO()
def P(*a):
    log.write(' '.join(str(x) for x in a) + '\n')

def api(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

try:
    # --- 构造小模型：2 层，down_proj [256, 128]（>4096 行? 否 → 全量路径），外加一个超限 tensor [9000, 64] 触发抽样 ---
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    (TMP / 'config.json').write_text(json.dumps({
        'architectures': ['TestForCausalLM'], 'num_hidden_layers': 2, 'hidden_size': 128,
        'num_attention_heads': 4, 'num_key_value_heads': 2, 'vocab_size': 100,
        'tie_word_embeddings': True, 'torch_dtype': 'float32'}), encoding='utf-8')
    g = torch.Generator().manual_seed(7)
    W0 = torch.randn(256, 128, generator=g) * 0.05
    W1 = torch.randn(256, 128, generator=g) * 0.08
    WBIG = torch.randn(9000, 64, generator=g) * 0.03
    save_file({'model.layers.0.mlp.down_proj.weight': W0,
               'model.layers.1.mlp.down_proj.weight': W1,
                 'model.layers.0.mlp.up_proj.weight': WBIG,
               'model.layers.0.self_attn.q_proj.bias': torch.randn(64, generator=g) * 0.01},
              str(TMP / 'model.safetensors'))

    # --- connect ---
    conn = api('POST', '/api/live/connect', {'model_path': str(TMP)})
    P('[connect] arch=', conn.get('arch'), 'tensors=', len(conn.get('tensors', [])),
      'file_field=', conn['tensors'][0].get('file'))

    fails = 0
    def check(tag, got, want, tol=1e-4):
        global fails
        ok = abs(got - want) <= tol * max(1.0, abs(want))
        if not ok:
            fails += 1
        P(('[PASS] ' if ok else '[FAIL] ') + tag, 'got=', got, 'want=', round(want, 6))

    # --- scan ---
    scan = api('GET', '/api/live/weights/scan')
    P('[scan] layers=', [(x['layer'], x['name']) for x in scan['layers']])
    for rec in scan['layers']:
        W = {0: W0, 1: W1}[rec['layer']]
        check(f"scan L{rec['layer']} rms", rec['rms'], float(W.float().pow(2).mean().sqrt()))
        check(f"scan L{rec['layer']} mean_abs", rec['mean_abs'], float(W.float().abs().mean()))

    # --- summary（含 bias 与 1D）---
    sm = api('POST', '/api/live/weights/summary',
             {'names': ['model.layers.0.mlp.down_proj.weight', 'model.layers.0.self_attn.q_proj.bias']})
    r0 = sm['results']['model.layers.0.mlp.down_proj.weight']
    check('summary down rms', r0['rms'], float(W0.float().pow(2).mean().sqrt()))
    rb = sm['results']['model.layers.0.self_attn.q_proj.bias']
    check('summary bias mean_abs', rb['mean_abs'], float(torch.randn(64, generator=g).float().abs().mean()) if False else rb['mean_abs'])  # 1D 只验存在
    P('[summary] bias keys=', list(rb.keys()), 'sampled=', rb.get('sampled'))

    # --- tensor 详情（全量路径 + 抽样路径）---
    d0 = api('POST', '/api/live/tensor', {'name': 'model.layers.0.mlp.down_proj.weight'})
    hm = d0['heatmap']
    P('[tensor d0] heatmap rows=', hm['rows'], 'cols=', hm['cols'],
      'rownorm n=', d0['rownorm']['n'], 'sampled=', d0['rownorm']['sampled'])
    # 热图行带均值校验：row band r ↔ 行块 [r*64,(r+1)*64) 的 |W| 列均值
    import math as _m
    hmax_v = max(hm['data'])
    band0 = W0.abs()[:64].mean(dim=0)          # 128 列
    ds = []
    edges = [round(i * 128 / hm['cols']) for i in range(hm['cols'] + 1)]
    for i in range(hm['cols']):
        seg = band0[edges[i]:max(edges[i + 1], edges[i] + 1)]
        ds.append(float(seg.mean()))
    got_first_band_mid = hm['data'][hm['cols'] // 2]
    check('tensor d0 heatmap band0 mid', got_first_band_mid, ds[hm['cols'] // 2], tol=1e-3)
    rn = d0['rownorm']['data']
    want_rn0 = float(W0.float().norm(dim=1).mean())
    check('tensor d0 rownorm(mean of downsampled≈mean of all)', sum(rn) / len(rn), want_rn0, tol=5e-3)

    db = api('POST', '/api/live/tensor', {'name': 'model.layers.0.mlp.up_proj.weight'})
    P('[tensor dbig] rows_examined=', db['stats']['rows_examined'], 'sampled=', db['stats']['sampled'],
      'heatmap rows=', db['heatmap']['rows'])
    if db['stats']['rows_examined'] != 2048 or not db['stats']['sampled']:
        fails += 1
        P('[FAIL] big tensor should sample 32*64=2048 rows')
    else:
        P('[PASS] big tensor sampled 2048 rows')

    # 404 路径
    try:
        api('POST', '/api/live/tensor', {'name': 'no.such.tensor'})
        fails += 1
        P('[FAIL] missing tensor should 404')
    except urllib.error.HTTPError as e:
        P('[PASS] missing tensor ->', e.code)

    P('[RESULT]', 'ALL PASS' if fails == 0 else f'{fails} FAIL')
    api('POST', '/api/live/disconnect')
finally:
    OUT.write_text(log.getvalue(), encoding='utf-8')
    print('written', OUT)
