# -*- coding: utf-8 -*-
"""P0.1 provenance：所有结果 json 必带的溯源五字段（G5-DP01 P0.1 / 审计 B2）。

五字段：git_sha / git_dirty / model_path / model_revision / seed
外加 tool 与 created_at 两个辅助字段。写入走 write_result()，落 results/<日期>/。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DEFAULT_MODEL = os.environ.get('AI2050_LIVE_MODEL_DIR') or 'D:/develop/model/Qwen/Qwen3-4B'


def _git(*args: str) -> str:
    try:
        r = subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                           text=True, timeout=15)
        return (r.stdout or '').strip()
    except Exception:
        return ''


def model_revision(model_path=None) -> str:
    """模型身份 = config.json 的 sha8（内容变即版本变，与路径无关）。"""
    p = Path(model_path or DEFAULT_MODEL)
    cfg = p / 'config.json'
    if cfg.exists():
        return 'sha8:' + hashlib.sha256(cfg.read_bytes()).hexdigest()[:8]
    return 'unknown'


def provenance(model_path=None, seed=None, tool: str = '') -> dict:
    dirty = _git('status', '--porcelain')
    return {
        'git_sha': _git('rev-parse', 'HEAD') or 'unknown',
        'git_dirty': bool(dirty),
        'model_path': str(Path(model_path or DEFAULT_MODEL)),
        'model_revision': model_revision(model_path),
        'seed': int(seed) if seed is not None else None,
        'tool': tool,
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }


def check_provenance(doc: dict) -> list:
    """验收口径（P0.1）：五字段齐全且非 unknown/缺失。返回问题清单，空 = 通过。"""
    prov = (doc or {}).get('provenance') or {}
    need = ['git_sha', 'git_dirty', 'model_path', 'model_revision', 'seed']
    missing = [k for k in need if k not in prov]
    bad = [k for k in ('git_sha', 'model_revision') if prov.get(k) in (None, '', 'unknown')]
    return [f'缺失字段 {k}' for k in missing] + [f'字段 {k} 取值无效' for k in bad]


def write_result(name: str, payload, model_path=None, seed=None, tool: str = '') -> Path:
    """把一份结果带 provenance 落 results/<日期>/，返回路径。文件带内容 sha8。"""
    doc = {
        'schema': 'result.v1',
        'name': name,
        'provenance': provenance(model_path=model_path, seed=seed, tool=tool),
        'payload': payload,
    }
    body = json.dumps(doc, ensure_ascii=False, indent=1)
    doc['sha8'] = hashlib.sha256(body.encode('utf-8')).hexdigest()[:8]
    day = RESULTS / datetime.now().strftime('%Y-%m-%d')
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"{datetime.now().strftime('%H%M%S')}_{name}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n',
                    encoding='utf-8', newline='')
    return path
