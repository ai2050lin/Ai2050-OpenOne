# -*- coding: utf-8 -*-
"""语言图谱结构（lang_map.v1）的公开只读端点。数据是 shared/dist_protocol/lang_map/ 下随版本发布的文件，
启动后第一次请求时读入并核对 sha256；改结构须在仓库里改 structure.json 后重建、随中心版本发布（不经接口写入）。

  GET /api/lang/map                    上三层（领域 → 子领域 → 语言项）、横切维度、版本与 sha256
  GET /api/lang/map/words/{item_id}    一个语言项下的词层
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.dist_protocol import lang_map  # noqa: E402

router = APIRouter(prefix='/api/lang/map', tags=['lang_map'])


@lru_cache(maxsize=1)
def _data() -> dict:
    return lang_map.load()


def map_ep() -> dict:
    d = _data()
    return {**d['map'], 'manifest': d['manifest']}


def words_ep(item_id: str) -> dict:
    d = _data()
    words = d['words']['items'].get(item_id)
    known = {i['id'] for dom in d['map']['domains'] for s in dom['subs'] for i in s['items']}
    if words is None and item_id not in known:
        raise HTTPException(404, f'没有语言项 {item_id}')
    return {'item': item_id, 'version': d['manifest']['version'], 'words': words or []}


def item_ids() -> list:
    return sorted(i['id'] for dom in _data()['map']['domains'] for s in dom['subs'] for i in s['items'])


router.add_api_route('', map_ep, methods=['GET'])
router.add_api_route('/words/{item_id}', words_ep, methods=['GET'])
