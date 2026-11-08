# -*- coding: utf-8 -*-
"""关联机制工作台的冻结样本数据（Phase 346）：从节点本地的 MA-H2 discovery 原生全场按条读出。

只在本机读（AI2050_NATIVE_DIR，缺省 ~/.ai2050/native_fields）；路径在 /api/traces/ 下，公开中心按本机工具规则拦截
（需要管理令牌），原生全场不经中心公开。数组用 float16 小端字节的 base64 传输，客户端解码；数值是原生精度
（MA-H2 采集时 bf16 存为 float16），不做二次压缩。

  GET /api/traces/mainline/ma-h2/samples       全部 2,688 条的格、实体、资料色 K、常识色 P、token 数与读出位置
  GET /api/traces/mainline/ma-h2/sample/{i}    第 i 条：hs [6, 37, 2560]、mlp [36, 9728]（答案位置 A）、
                                               attn [36, 32, ntok]（答案位置对前缀）、lens [37, 3]（K / P / D 的逐层读出）
"""
from __future__ import annotations

import base64
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix='/api/traces/mainline/ma-h2', tags=['mainline_sample'])
SNAP = '6be1034f0b55a6df-bf16'


def native_dir() -> Path:
    root = os.environ.get('AI2050_NATIVE_DIR') or str(Path.home() / '.ai2050' / 'native_fields')
    return Path(root) / '_mainline' / 'MA-H2' / SNAP


@lru_cache(maxsize=1)
def _meta():
    d = native_dir()
    if not (d / 'manifest.json').is_file():
        raise HTTPException(404, f'本机没有 MA-H2 原生全场（{d}）：原生数据保存在采集节点，只在桌面客户端或本机后端可读')
    man = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
    meta = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
    return man, meta, np.load(d / 'ntok.npy'), np.load(d / 'tokpos.npy')


def _b64(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a, dtype='<f2')
    return {'shape': list(a.shape), 'dtype': 'float16', 'b64': base64.b64encode(a.tobytes()).decode()}


def samples():
    man, meta, ntok, tokpos = _meta()
    return {'id': 'MA-H2', 'split': 'discovery', 'model': 'Qwen3-4B', 'model_snapshot': man['model_snapshot'],
            'positions': man['positions'], 'mlp_boundary': man.get('mlp_boundary'), 'n': len(meta),
            'items': [{'i': i, 'cell': m['cell'], 'entity': m['entity'], 'K': m['K'], 'P': m['P'], 'ntok': int(ntok[i]),
                       'tokpos': tokpos[i].tolist()} for i, m in enumerate(meta)]}


def sample(i: int):
    man, meta, ntok, tokpos = _meta()
    if not 0 <= i < len(meta):
        raise HTTPException(404, f'没有第 {i} 条')
    d, n = native_dir(), int(ntok[i])
    hs = np.load(d / 'hs.npy', mmap_mode='r')[i]
    mlp = np.load(d / 'mlp.npy', mmap_mode='r')[i]
    att = np.load(d / 'attn.npy', mmap_mode='r')[i][:, :, :n]
    lens = np.load(d / 'lens.npy', mmap_mode='r')[i]
    m = meta[i]
    return {'i': i, 'cell': m['cell'], 'entity': m['entity'], 'K': m['K'], 'P': m['P'], 'ntok': n, 'tokpos': tokpos[i].tolist(),
            'positions': man['positions'], 'model_snapshot': man['model_snapshot'],
            'hs': _b64(hs), 'mlp': _b64(mlp), 'attn': _b64(att), 'lens': np.asarray(lens).round(4).tolist(),
            'source': 'MA-H2 discovery 原生全场（节点本地，bf16 存为 float16）'}


router.add_api_route('/samples', samples, methods=['GET'])
router.add_api_route('/sample/{i}', sample, methods=['GET'])
