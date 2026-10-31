# -*- coding: utf-8 -*-
"""客户端离线数据包（datapack.v1）：公开研究数据打成 gzip JSON，客户端手动下载后存在本地，页面只读本地。

条目以「规范化请求路径」为键（路径 + 按键排序、解码后的查询参数），值就是该接口的 JSON 返回，
客户端用同样的规则查找。离线客户端只在线读取站点配置与下载数据包，不实时访问其他接口。

- 主数据包（GET /api/platform/datapack）：总览、题库、模板、谱图、记分板、普查与神经元索引、解释检验、选择性、
  3D 层空间、legacy 结果等，约 1–2 MB。
- 附加包（GET /api/platform/datapack/extra/{id}）：体积大的部分按需下载。
  · tm:<tm_id>     一个模式族的合并统计（门控开 / 关）与已算好的规律分析（实验响应 3D 页）。
  · cards:<model>  已解释、模板选择性前列与 L 层检验过的单元卡（原生神经元页）；其余单元不打包。
由后台计算子进程（snapshot_worker）在数据变化时重建；网站进程只读文件，版本与大小写进 /api/platform/config。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit

SCHEMA = 'datapack.v1'
EXTRA_SCHEMA = 'datapack_extra.v1'


def canonical(path: str) -> str:
    """规范化请求路径：与客户端 dataStore.canonical 相同的规则。"""
    u = urlsplit(path)
    q = sorted(parse_qsl(u.query, keep_blank_values=True))
    return u.path + ('?' + '&'.join(f'{k}={v}' for k, v in q) if q else '')


def pack_dir(data_dir: Path) -> Path:
    d = Path(data_dir) / 'datapack'
    d.mkdir(parents=True, exist_ok=True)
    return d


def extra_file(data_dir: Path, extra_id: str) -> Path:
    """附加包文件名：id 里只保留安全字符（tm:TG-GR-Q → tm_TG-GR-Q.json.gz）。"""
    return pack_dir(data_dir) / 'extra' / (re.sub(r'[^A-Za-z0-9._-]', '_', extra_id) + '.json.gz')


def research_stamp(research_root: Path) -> str:
    """研究文件（神经元选择性、解释检验、3D 空间）的内容戳：它们变了，数据包也要重建。
    按内容而不是修改时间计算，同一批文件在开发机与服务器上得到同一个戳（预先算好的数据包可直接沿用）。"""
    h = hashlib.sha256()
    for sub in ('neurons', 'interventions'):
        d = Path(research_root) / sub
        if d.is_dir():
            for p in sorted(d.glob('*.json')):
                h.update(p.name.encode('utf-8'))
                h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:12]


def entries_plan() -> List[Tuple[str, Callable[[], Any]]]:
    """主数据包的 (请求路径, 取值函数) 列表。函数直接调用各路由的实现，与在线接口返回完全一致。"""
    import census_service as CS
    import distributed_service as D
    import platform_service as PS
    from server import neuron_census_service as NS
    from shared.dist_protocol import pipeline as P

    plan: List[Tuple[str, Callable[[], Any]]] = [
        ('/api/platform/overview', lambda: {**PS.build_overview(), 'computed_at': time.time()}),
        ('/api/catalog', D.api_catalog),
        ('/api/templates', D.templates_list),
        ('/api/grammar/library', D.grammar_library),
        ('/api/atlas/hypotheses?limit=5', lambda: D.api_atlas_hypotheses(limit=5)),
        ('/api/atlas/hypotheses?limit=20', lambda: D.api_atlas_hypotheses(limit=20)),
        ('/api/census/jobs', CS.jobs_list),
        ('/api/platform/contributors', PS.contributors_ep),
        ('/api/neurons/index', NS.index),
        ('/api/neurons/space/index', NS.space_index),
    ]
    for name in ('catalog', 'coverage', 'pipeline_overview', 'scoreboard', 'atlas'):
        plan.append((f'/api/platform/snapshot/{name}', (lambda n=name: PS.snapshot(n))))
    lang = PS._lang_atlas()                            # 语言谱图（体积小，进主数据包）：还没生成时不放条目，下一版数据包再补
    if lang is not None:
        data = lang['data']
        plan.append(('/api/lang/tree', PS.lang_tree_ep))
        plan.append(('/api/model/skeleton', PS.model_skeleton_ep))
        for item_id in sorted(data['items']):
            plan.append((f'/api/lang/item/{item_id}', (lambda i=item_id: PS.lang_item_ep(i))))
        for mk in sorted(data['skeletons']):
            plan.append((f'/api/model/skeleton?model_key={mk}', (lambda m=mk: PS.model_skeleton_ep(m))))
        for mk in sorted(data['overlays']):
            plan.append((f'/api/space/overlay?model_key={mk}', (lambda m=mk: PS.space_overlay_ep(m))))
    idx = NS.index()
    model_keys = sorted({r['model_key'] for r in idx.get('census', [])})
    for mk in model_keys:
        plan.append((f'/api/neurons/tower?model_key={mk}', (lambda m=mk: NS.neuron_tower(model_key=m, run=None))))
        plan.append((f'/api/neurons/explained?model_key={mk}', (lambda m=mk: NS.explained(model_key=m, grade=None))))
        for g in NS.GRADES:
            plan.append((f'/api/neurons/explained?model_key={mk}&grade={g}', (lambda m=mk, gg=g: NS.explained(model_key=m, grade=gg))))
    for r in idx.get('selectivity', []):
        plan.append((f"/api/neurons/selectivity?run={r['file']}&top=20", (lambda f=r['file']: NS.selectivity(run=f, top=20))))
    for s in NS.space_index().get('layers', []):
        plan.append((f"/api/neurons/space?model_key={s['model_key']}&layer={s['layer']}",
                     (lambda m=s['model_key'], l=s['layer']: NS.space(model_key=m, layer=l))))
    db = D._db()                                       # legacy 模板（单节点摘要，体积小）：结果列表与 cos 矩阵
    legacy =[r['tm_id'] for r in db.execute("SELECT DISTINCT tm_id FROM results WHERE kind='real'").fetchall()
              if not P.is_v3(P.template_corpus(db, r['tm_id']))]
    for tm in legacy:
        plan.append((f'/api/results?tm_id={tm}&kind=real&limit=50',
                     (lambda t=tm: D.results_list(tm_id=t, model_id='', kind='real', limit=50))))
        for r in db.execute("SELECT sha FROM results WHERE tm_id=? AND kind='real' ORDER BY uploaded_at DESC LIMIT 50", (tm,)):
            plan.append((f"/api/results/{r['sha']}", (lambda s=r['sha']: _legacy_detail(D, s))))
    return plan


def _legacy_detail(D, sha: str) -> Dict[str, Any]:
    d = D.result_detail(sha, count=False)
    d.pop('files', None)                               # 原始文件不进数据包（需要时下载研究包）
    return d


def _write_gz(path: Path, doc: Dict[str, Any]) -> Tuple[int, int]:
    raw = json.dumps(doc, ensure_ascii=False, default=str, separators=(',', ':')).encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(tmp, 'wb', compresslevel=6) as fh:
        fh.write(raw)
    tmp.replace(path)
    return path.stat().st_size, len(raw)


def build(data_dir: Path, version: str, log=print) -> Dict[str, Any]:
    entries, errors = {}, []
    for path, fn in entries_plan():
        try:
            entries[canonical(path)] = fn()
        except Exception as exc:                        # 单个条目失败不影响整包；离线客户端对缺的条目显示「数据包里没有」
            errors.append(f'{path}: {type(exc).__name__}: {exc}'[:200])
    created = time.time()
    doc = {'schema': SCHEMA, 'version': version, 'created_at': created, 'entries': entries, 'errors': errors}
    d = pack_dir(data_dir)
    size, raw = _write_gz(d / 'datapack.json.gz', doc)
    meta = {'schema': SCHEMA, 'version': version, 'created_at': created, 'n_entries': len(entries),
            'bytes': size, 'raw_bytes': raw, 'errors': errors}
    (d / 'datapack.meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    log(f"[datapack] {len(entries)} 条，{size / 2**20:.1f} MB（解压 {raw / 2**20:.1f} MB）" + (f'，{len(errors)} 条失败' if errors else ''))
    return meta


def read_meta(data_dir: Path) -> Dict[str, Any] | None:
    p = Path(data_dir) / 'datapack' / 'datapack.meta.json'
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


# ---------------- 附加包 ----------------

def read_extras(data_dir: Path) -> Dict[str, Any]:
    p = pack_dir(data_dir) / 'extras.json'
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _write_extras(data_dir: Path, extras: Dict[str, Any]) -> None:
    p = pack_dir(data_dir) / 'extras.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(extras, ensure_ascii=False), encoding='utf-8')
    tmp.replace(p)


class RawFile:
    """附加包里原样写入的 JSON 文件（规律分析缓存可达几十 MB：不解析，直接拷贝进包，生成时不占内存）。"""

    def __init__(self, path: Path):
        self.path = Path(path)


def _template_inputs(tm_id: str):
    import distributed_service as D
    from shared.dist_protocol import pipeline as P
    db = D._db()
    aggs = {gate: P.aggregate(db, D._DATA_DIR, tm_id, gate=gate) for gate in (True, False)}
    buckets = sorted({b['bucket'] for a in aggs.values() for b in a['buckets'] if b.get('agg')})
    analyses = {(b, gate): P.analysis_cached_path(db, D._DATA_DIR, tm_id, b, gate) for b in buckets for gate in (True, False)}
    return aggs, analyses


def template_version(tm_id: str) -> str:
    """一个模式族附加包的版本：合并结果的分片 sha、门控排除、被排除结果与规律分析缓存文件名，加上合并与分析代码哈希。
    只由数据与代码决定，同一份数据在开发机与服务器上得到同一个版本（预先算好的附加包可直接沿用）。"""
    from shared.dist_protocol import pipeline as P
    aggs, analyses = _template_inputs(tm_id)
    parts = [P._code_hash(), P._analysis_code_hash()]
    for gate, a in sorted(aggs.items()):
        for b in a['buckets']:
            parts.append([gate, b['bucket'], b.get('primary_shas'), b.get('gate_excluded_cells'),
                          sorted((e.get('sha'), e.get('reason')) for e in b.get('excluded') or []), bool(b.get('agg'))])
    parts.append(sorted([k[0], k[1], p.name if p else None] for k, p in analyses.items()))
    return hashlib.sha256(json.dumps(parts, default=str).encode('utf-8')).hexdigest()[:16]


def template_entries(tm_id: str) -> Tuple[Dict[str, Any], List[str]]:
    """一个模式族的实验响应数据：合并统计（门控开 / 关）与已缓存的规律分析（每个模型桶、门控开 / 关）。
    规律分析只取已算好的缓存（RawFile，原样拷贝），不在这里计算；没有缓存的组合离线时显示「数据包里没有」。"""
    aggs, analyses = _template_inputs(tm_id)
    entries, notes = {}, []
    for gate, agg in aggs.items():
        entries[canonical(f'/api/agg/{tm_id}/v2?gate={str(gate).lower()}')] = agg
    for (b, gate), p in sorted(analyses.items()):
        if p is None:
            notes.append(f'{b} gate={gate}：规律分析尚未计算')
            continue
        entries[canonical(f'/api/analysis/{tm_id}?bucket={b}&gate={str(gate).lower()}')] = RawFile(p)
    return entries, notes


def _write_extra_gz(path: Path, head: Dict[str, Any], entries: Dict[str, Any]) -> Tuple[int, int]:
    """流式写附加包：逐条写入，RawFile 按块拷贝，不在内存里拼整份 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    raw = 0
    with gzip.open(tmp, 'wb', compresslevel=6) as fh:
        def put(b: bytes):
            nonlocal raw
            fh.write(b)
            raw += len(b)
        put(json.dumps(head, ensure_ascii=False, default=str)[:-1].encode('utf-8') + b',"entries":{')
        for i, (k, v) in enumerate(entries.items()):
            put((',' if i else '').encode('utf-8') + json.dumps(k, ensure_ascii=False).encode('utf-8') + b':')
            if isinstance(v, RawFile):
                with v.path.open('rb') as src:
                    for chunk in iter(lambda: src.read(1 << 20), b''):
                        put(chunk)
            else:
                put(json.dumps(v, ensure_ascii=False, default=str, separators=(',', ':')).encode('utf-8'))
        put(b'}}')
    tmp.replace(path)
    return path.stat().st_size, raw


def card_units(model_key: str) -> List[Tuple[int, int]]:
    """单元卡包里的单元：已检验解释的单元、模板选择性前列（各层各读出位置）、有触发点 / 条件化消融检验行的单元。"""
    from server import neuron_census_service as NS
    units = set()
    for r in NS.explained(model_key=model_key, grade=None).get('results', []):
        units.add((int(r['block']), int(r['unit'])))
    for s in NS.index().get('selectivity', []):
        if s.get('model_key') != model_key:
            continue
        for t in NS.selectivity(run=s['file'], top=20).get('tags', []):
            for k in ('top_by_d', 'top_by_contrib'):
                for r in t.get(k) or []:
                    units.add((int(t['block']), int(r['unit'])))
    for f in sorted((NS.research_root() / 'neurons').glob('L*-v*-*.json')):
        m = re.match(r'L(\d+)-v\d+-(trigger|conditional)\.json$', f.name)
        if not m:
            continue
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if d.get('model_key', model_key) != model_key:
            continue
        for r in d.get('rows', []):
            units.add((int(m.group(1)), int(r['unit'])))
    return sorted(units)


def card_entries(model_key: str, log=print) -> Tuple[Dict[str, Any], List[str]]:
    """单元卡：只为该模型 token 数最多的普查运行生成（与原生神经元页默认选中的运行相同）。"""
    from server import neuron_census_service as NS
    run = NS.pick_census(model_key, None)['run']
    entries, notes = {}, []
    units = card_units(model_key)
    t0 = time.time()
    for block in range(NS.census_manifest(model_key, run)['n_layers']):   # 逐层摘要（神经元主线的层阵列）
        doc = NS.neuron_layer(model_key=model_key, block=block, run=run)
        if doc.get('pending'):
            notes.append(f'L{block}: 逐层摘要尚未生成')
        else:
            entries[canonical(f'/api/neurons/layer?model_key={model_key}&block={block}&run={run}')] = doc
    for i, (block, unit) in enumerate(units):
        try:
            card = NS.unit_card(model_key=model_key, block=block, unit=unit, run=run, n_poles=12)
            entries[canonical(f'/api/neurons/unit?model_key={model_key}&block={block}&unit={unit}&run={run}')] = card
        except Exception as exc:
            notes.append(f'L{block}.U{unit}: {type(exc).__name__}: {exc}'[:160])
        if (i + 1) % 200 == 0:
            log(f'[datapack] 单元卡 {i + 1}/{len(units)}（{time.time() - t0:.0f}s）')
    return entries, notes


def circuit_source(data_dir: Path, model_key: str) -> Path:
    """跨层电路预计算源文件（研发机 scripts/build_circuit_pack.py 生成后上传到 <data_dir>/circuit_sources/）。"""
    return Path(data_dir) / 'circuit_sources' / (re.sub(r'[^A-Za-z0-9._-]', '_', model_key) + '.circuit.v1.json.gz')


def extras_plan() -> List[Dict[str, Any]]:
    """附加包清单：每个 v3 模式族一个，每个有普查结果的模型一个单元卡包与一个跨层电路包。"""
    import distributed_service as D
    from server import neuron_census_service as NS
    from shared.dist_protocol import pipeline as P
    out = []
    for t, corpus in P.templates_v3(D._db()):
        if P.is_v3(corpus):
            out.append({'id': f"tm:{t['tm_id']}", 'kind': 'template', 'tm_id': t['tm_id'],
                        'label': f"实验响应 · {t['tm_id']}"})
    mks = sorted({r['model_key'] for r in NS.index().get('census', [])})
    for mk in mks:
        out.append({'id': f'cards:{mk}', 'kind': 'cards', 'model_key': mk, 'label': f'单元卡 · {mk[:8]}'})
    for mk in mks:
        out.append({'id': f'circuit:{mk}', 'kind': 'circuit', 'model_key': mk, 'label': f'跨层电路 · {mk[:8]}'})
    return out


def build_extras(data_dir: Path, cards_version: str, force: bool = False, log=print,
                 guard=None) -> Dict[str, Any]:
    """按需重建附加包：模式族包的版本由 template_version 按该模式族的数据计算，单元卡包用 cards_version
    （研究文件与普查版本）；同版本的包跳过。guard（可选，后台子进程的崩溃记录）：skip(name) / start(name)。"""
    extras = read_extras(data_dir)
    plan = extras_plan()
    keep = {e['id'] for e in plan}
    for gone in [k for k in extras if k not in keep]:
        extra_file(data_dir, gone).unlink(missing_ok=True)
        extras.pop(gone, None)
    built = []
    for e in plan:
        src_circuit = None
        try:
            if e['kind'] == 'cards':
                version = cards_version
            elif e['kind'] == 'circuit':
                # 电路在研发机预计算（服务器没有权重）：源文件存在才打包，版本 = 源哈希 + 研究版本
                src_circuit = circuit_source(data_dir, e['model_key'])
                if not src_circuit.is_file():
                    log(f"[datapack] {e['id']} 无源文件（build_circuit_pack.py 产出后上传 circuit_sources/），跳过")
                    continue
                version = hashlib.sha256(src_circuit.read_bytes()).hexdigest()[:8] + '-' + cards_version[:8]
            else:
                version = template_version(e['tm_id'])
        except Exception as exc:
            log(f"[datapack] {e['id']} 版本计算失败：{type(exc).__name__}: {exc}")
            continue
        old = extras.get(e['id'])
        if not force and old and old.get('version') == version and extra_file(data_dir, e['id']).is_file():
            continue
        name = f"extra:{e['id']}"
        if guard and guard.skip(name):
            continue
        if guard:
            guard.start(name)
        try:
            if e['kind'] == 'template':
                entries, notes = template_entries(e['tm_id'])
            elif e['kind'] == 'circuit':
                with gzip.open(src_circuit, 'rt', encoding='utf-8') as fh:
                    entries = json.load(fh).get('entries', {})
                notes = [f'预计算源 {src_circuit.name}（研发机模型权重， scripts/build_circuit_pack.py）']
            else:
                entries, notes = card_entries(e['model_key'], log=log)
        except Exception as exc:
            log(f"[datapack] {e['id']} 失败：{type(exc).__name__}: {exc}")
            continue
        created = time.time()
        head = {'schema': EXTRA_SCHEMA, 'id': e['id'], 'version': version, 'created_at': created, 'notes': notes}
        size, raw = _write_extra_gz(extra_file(data_dir, e['id']), head, entries)
        extras[e['id']] = {**e, 'version': version, 'created_at': created, 'n_entries': len(entries),
                           'bytes': size, 'raw_bytes': raw, 'notes': notes[:20]}
        _write_extras(data_dir, extras)
        built.append(e['id'])
        log(f"[datapack] 附加包 {e['id']}：{len(entries)} 条，{size / 2**20:.1f} MB")
    _write_extras(data_dir, extras)
    if guard:
        guard.done()
    return {'built': built, 'n': len(extras)}


def extras_meta(data_dir: Path) -> List[Dict[str, Any]]:
    """站点配置里的附加包清单（客户端据此显示下载按钮与大小）。"""
    keys = ('id', 'kind', 'label', 'tm_id', 'model_key', 'version', 'created_at', 'bytes', 'raw_bytes', 'n_entries')
    return [{k: v.get(k) for k in keys if v.get(k) is not None} for v in read_extras(data_dir).values()]


def pack_hint(path: str) -> Optional[str]:
    """一条请求应在哪个附加包里（客户端有同样的规则，用于「下载该包」提示）。"""
    p = urlsplit(path)
    m = re.match(r'^/api/(?:agg/([^/]+)/v2|analysis/([^/]+))$', p.path)
    if m:
        return f'tm:{m.group(1) or m.group(2)}'
    if p.path in ('/api/neurons/unit', '/api/neurons/layer'):
        mk = dict(parse_qsl(p.query)).get('model_key')
        return f'cards:{mk}' if mk else None
    if p.path == '/api/neurons/circuit':
        mk = dict(parse_qsl(p.query)).get('model_key')
        return f'circuit:{mk}' if mk else None
    return None
