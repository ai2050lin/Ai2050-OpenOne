# -*- coding: utf-8 -*-
"""客户端离线数据包（datapack.v1）：公开研究数据打成一个 gzip JSON，客户端手动下载后存在本地，页面只读本地。

条目以「规范化请求路径」为键（路径 + 按键排序、解码后的查询参数），值就是该接口的 JSON 返回，
客户端用同样的规则查找，命中就不访问服务器。由后台计算子进程（snapshot_worker）在数据变化时重建；
网站进程只读文件（GET /api/platform/datapack），版本号写进 /api/platform/config。
不进数据包、仍在线读取的：单元卡（按需从 1.1 GB 普查数据里现取）、实验响应的规律分析与合并统计（体积大，客户端打开时下载一次并缓存）、
管理类接口（审计、节点、任务、普查复核、自动测试）与账号相关接口。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import parse_qsl, urlsplit

SCHEMA = 'datapack.v1'


def canonical(path: str) -> str:
    """规范化请求路径：与客户端 dataStore.canonical 相同的规则。"""
    u = urlsplit(path)
    q = sorted(parse_qsl(u.query, keep_blank_values=True))
    return u.path + ('?' + '&'.join(f'{k}={v}' for k, v in q) if q else '')


def pack_dir(data_dir: Path) -> Path:
    d = Path(data_dir) / 'datapack'
    d.mkdir(parents=True, exist_ok=True)
    return d


def research_stamp(research_root: Path) -> str:
    """研究文件（神经元选择性、解释检验、3D 空间）的修改时间戳：它们变了，数据包也要重建。"""
    parts = []
    for sub in ('neurons', 'interventions'):
        d = Path(research_root) / sub
        if d.is_dir():
            for p in sorted(d.glob('*.json')):
                parts.append((p.name, int(p.stat().st_mtime), p.stat().st_size))
    return hashlib.sha256(json.dumps(parts).encode('utf-8')).hexdigest()[:12]


def entries_plan() -> List[Tuple[str, Callable[[], Any]]]:
    """(请求路径, 取值函数) 列表。函数直接调用各路由的实现，与在线接口返回完全一致。"""
    import census_service as CS
    import distributed_service as D
    import platform_service as PS
    from server import neuron_census_service as NS

    plan: List[Tuple[str, Callable[[], Any]]] = [
        ('/api/platform/overview', lambda: {**PS.build_overview(), 'computed_at': time.time()}),
        ('/api/catalog', D.api_catalog),
        ('/api/templates', D.templates_list),
        ('/api/grammar/library', D.grammar_library),
        ('/api/atlas/hypotheses?limit=5', lambda: D.api_atlas_hypotheses(limit=5)),
        ('/api/census/jobs', CS.jobs_list),
        ('/api/platform/contributors', PS.contributors_ep),
        ('/api/neurons/index', NS.index),
        ('/api/neurons/space/index', NS.space_index),
    ]
    for name in ('catalog', 'coverage', 'pipeline_overview', 'scoreboard', 'atlas'):
        plan.append((f'/api/platform/snapshot/{name}', (lambda n=name: PS.snapshot(n))))
    idx = NS.index()
    model_keys = sorted({r['model_key'] for r in idx.get('census', [])})
    for mk in model_keys:
        plan.append((f'/api/neurons/explained?model_key={mk}', (lambda m=mk: NS.explained(model_key=m, grade=None))))
        for g in NS.GRADES:
            plan.append((f'/api/neurons/explained?model_key={mk}&grade={g}', (lambda m=mk, gg=g: NS.explained(model_key=m, grade=gg))))
    for r in idx.get('selectivity', []):
        plan.append((f"/api/neurons/selectivity?run={r['file']}&top=20", (lambda f=r['file']: NS.selectivity(run=f, top=20))))
    for s in NS.space_index().get('layers', []):
        plan.append((f"/api/neurons/space?model_key={s['model_key']}&layer={s['layer']}",
                     (lambda m=s['model_key'], l=s['layer']: NS.space(model_key=m, layer=l))))
    return plan


def build(data_dir: Path, version: str, log=print) -> Dict[str, Any]:
    entries, errors = {}, []
    for path, fn in entries_plan():
        try:
            entries[canonical(path)] = fn()
        except Exception as exc:                        # 单个条目失败不影响整包；客户端对缺的条目会在线读取一次
            errors.append(f'{path}: {type(exc).__name__}: {exc}'[:200])
    doc = {'schema': SCHEMA, 'version': version, 'created_at': time.time(), 'entries': entries, 'errors': errors}
    raw = json.dumps(doc, ensure_ascii=False, default=str, separators=(',', ':')).encode('utf-8')
    d = pack_dir(data_dir)
    tmp = d / 'datapack.json.gz.tmp'
    with gzip.open(tmp, 'wb', compresslevel=6) as fh:
        fh.write(raw)
    tmp.replace(d / 'datapack.json.gz')
    meta = {'schema': SCHEMA, 'version': version, 'created_at': doc['created_at'], 'n_entries': len(entries),
            'bytes': (d / 'datapack.json.gz').stat().st_size, 'raw_bytes': len(raw), 'errors': errors}
    (d / 'datapack.meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    log(f"[datapack] {len(entries)} 条，{meta['bytes'] / 2**20:.1f} MB（解压 {len(raw) / 2**20:.1f} MB）" + (f'，{len(errors)} 条失败' if errors else ''))
    return meta


def read_meta(data_dir: Path) -> Dict[str, Any] | None:
    p = Path(data_dir) / 'datapack' / 'datapack.meta.json'
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
