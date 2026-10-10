# -*- coding: utf-8 -*-
"""S9b 权重数值端点补丁：deploy/distributed_service.py
1) _scan_live_model 每 tensor 记录所在 shard 文件名（'file' 字段）
2) live_disconnect 清数值缓存
3) 追加 S9b 三个端点：GET /live/weights/scan、POST /live/weights/summary、POST /live/tensor
幂等：已含 _wt_conn_payload 则跳过。
"""
import io, sys

P = r'D:\AI2050\Ai2050-OpenOne\deploy\distributed_service.py'
raw = open(P, 'rb').read()
text = raw.decode('utf-8')
NL = '\r\n' if raw.count(b'\r\n') > (raw.count(b'\n') - raw.count(b'\r\n')) else '\n'

def T(s):
    return s.replace('\n', NL) if NL == '\r\n' else s

if '_wt_conn_payload' in text:
    print('ALREADY_PATCHED')
    sys.exit(0)

# ---- 1) tensors.append 加 file 字段 ----
old1 = T("            tensors.append({'name': name, 'shape': shape,\n"
         "                            'dtype': _ST_DTYPE.get(raw_dt, raw_dt.lower()), 'params': params})")
new1 = T("            tensors.append({'name': name, 'shape': shape,\n"
         "                            'dtype': _ST_DTYPE.get(raw_dt, raw_dt.lower()), 'params': params,\n"
         "                            'file': sp.name})")
assert text.count(old1) == 1, ('ANCHOR1', text.count(old1))
text = text.replace(old1, new1)

# ---- 2) disconnect 清数值缓存 ----
old2 = T("def live_disconnect():\n"
         "    _LIVE['connected'] = False\n"
         "    _LIVE['payload'] = None\n"
         "    return {'connected': False}")
new2 = T("def live_disconnect():\n"
         "    _LIVE['connected'] = False\n"
         "    _LIVE['payload'] = None\n"
         "    _WT_T_CACHE.clear(); _WT_SCAN_CACHE['key'] = None   # 数值缓存随连接失效\n"
         "    return {'connected': False}")
assert text.count(old2) == 1, ('ANCHOR2', text.count(old2))
text = text.replace(old2, new2)

# ---- 3) S9b 端点块（插在新闻节之前） ----
anchor3 = T("# ---------- 新闻（S5） ----------")
assert text.count(anchor3) == 1, ('ANCHOR3', text.count(anchor3))

BLOCK = '''# ---------- S9b 权重数值（P2 第一步：结构 → 数值） ----------
# 懒加载真实权重数值：safetensors get_slice 按连续行块切片读（不整体加载权重）；
# 行数超限时在行域均匀抽 32 块 × 64 行，标量统计/热图/行范数全部降采样返回。
# 消费方：全部参数表（summary 数值列）、3D 层盒数值着色（scan）、参数热图（tensor 详情）。
_WT_T_CACHE = {}                                    # ('s'|'d', name) -> payload（连接会话内有效，disconnect 清空）
_WT_SCAN_CACHE = {'key': None, 'data': None}
_WT_BLOCK_ROWS, _WT_BLOCKS = 64, 32
_WT_HEAT_COLS, _WT_RN_POINTS = 64, 512


def _wt_conn_payload():
    if not _LIVE['connected'] or not _LIVE['payload']:
        raise HTTPException(409, '未连接本地模型——先 POST /api/live/connect')
    return _LIVE['payload']


def _wt_tmap(payload):
    return {t['name']: t for t in payload['tensors']}


def _wt_blocks(path, name, tmeta):
    """yield (start_row, block[float32], sampled)：2D 按行块切片；1D 整读一次。"""
    import torch
    from safetensors import safe_open
    shape = tmeta.get('shape') or []
    with safe_open(str(path), framework='pt', device='cpu') as f:
        if len(shape) <= 1:
            yield 0, f.get_slice(name)[:].float(), False
            return
        rows = int(shape[0])
        need = _WT_BLOCKS * _WT_BLOCK_ROWS
        if rows <= need:
            starts = list(range(0, rows, _WT_BLOCK_ROWS))
            sampled = False
        else:
            step = (rows - need) / (_WT_BLOCKS - 1)
            starts = [int(round(i * step)) for i in range(_WT_BLOCKS)]
            sampled = True
        for s in starts:
            e = min(rows, s + _WT_BLOCK_ROWS)
            yield s, f.get_slice(name)[s:e].float(), sampled


def _wt_scalar_stats(payload, tmap, name):
    import math
    hit = _WT_T_CACHE.get(('s', name))
    if hit:
        return hit
    n = 0; s1 = 0.0; s1a = 0.0; s2 = 0.0
    mn = None; mx = None; sampled = False; rows_seen = 0
    fp = Path(payload['model_path']) / tmap[name]['file']
    for _s, t, smp in _wt_blocks(fp, name, tmap[name]):
        sampled = sampled or smp
        n += int(t.numel())
        rows_seen += int(t.shape[0]) if t.dim() else 1
        s1 += float(t.sum()); s1a += float(t.abs().sum()); s2 += float((t * t).sum())
        tmin, tmax = float(t.min()), float(t.max())
        mn = tmin if mn is None else min(mn, tmin)
        mx = tmax if mx is None else max(mx, tmax)
    if not n:
        raise HTTPException(400, f'空 tensor: {name}')
    out = {'name': name, 'mean': round(s1 / n, 6), 'mean_abs': round(s1a / n, 6),
           'rms': round(math.sqrt(s2 / n), 6), 'min': round(mn, 6), 'max': round(mx, 6),
           'sampled': sampled, 'rows_examined': rows_seen}
    _WT_T_CACHE[('s', name)] = out
    return out


def _downsample(v, k):
    import torch
    t = torch.as_tensor(v, dtype=torch.float32)
    n = int(t.numel())
    if n <= k:
        return [round(float(x), 6) for x in t.tolist()]
    edges = [int(round(i * n / k)) for i in range(k + 1)]
    out = []
    for i in range(k):
        seg = t[edges[i]:max(edges[i + 1], edges[i] + 1)]
        out.append(round(float(seg.mean()), 6))
    return out


@router.get('/live/weights/scan')
def live_weights_scan():
    """每层代表权重（mlp.down_proj.weight）标量统计——3D 层盒数值着色数据源。"""
    import re
    payload = _wt_conn_payload()
    key = (payload['model_path'], tuple(f['name'] for f in payload['files']))
    if _WT_SCAN_CACHE['key'] == key:
        return _WT_SCAN_CACHE['data']
    tmap = _wt_tmap(payload)
    pat = re.compile(r'^(?:model\\.)?layers\\.(\\d+)\\.mlp\\.down_proj\\.weight$')
    found = {}
    for nm, tm in tmap.items():
        m = pat.match(nm)
        if m and len(tm.get('shape') or []) == 2:
            found[int(m.group(1))] = nm
    layers = []
    for l in sorted(found):
        st = _wt_scalar_stats(payload, tmap, found[l])
        layers.append({'layer': l, 'name': found[l], 'rms': st['rms'],
                       'mean_abs': st['mean_abs'], 'max_abs': st['max_abs'], 'sampled': st['sampled']})
    data = {'layers': layers, 'source': 'safetensors 懒加载真实权重（32×64 行块，超限均匀抽样）'}
    _WT_SCAN_CACHE['key'] = key
    _WT_SCAN_CACHE['data'] = data
    return data


class _WtNamesReq(BaseModel):
    names: List[str]


@router.post('/live/weights/summary')
def live_weights_summary(req: _WtNamesReq):
    """批量标量统计（≤24 名）——全部参数表数值列。"""
    payload = _wt_conn_payload()
    tmap = _wt_tmap(payload)
    out = {}
    for nm in req.names[:24]:
        if nm in tmap and tmap[nm].get('params', 0) >= 16:
            out[nm] = _wt_scalar_stats(payload, tmap, nm)
    return {'results': out, 'block': '32×64 行块（超限均匀抽样）'}


class _WtNameReq(BaseModel):
    name: str


@router.post('/live/tensor')
def live_tensor_detail(req: _WtNameReq):
    """单 tensor 数值详情：标量统计 + |W| 行带×列带均值热图 + 行 L2 范数（全部降采样）。"""
    import torch
    payload = _wt_conn_payload()
    tmap = _wt_tmap(payload)
    nm = req.name.strip()
    if nm not in tmap:
        raise HTTPException(404, f'tensor 不存在: {nm}')
    ck = ('d', nm)
    hit = _WT_T_CACHE.get(ck)
    if hit:
        return hit
    tmeta = tmap[nm]
    bands, rnorms, sampled_any = [], [], False
    fp = Path(payload['model_path']) / tmeta['file']
    for _s, t, smp in _wt_blocks(fp, nm, tmeta):
        sampled_any = sampled_any or smp
        if t.dim() <= 1:
            bands.append(t.abs())
            rnorms.append(float(t.norm()))
        else:
            bands.append(t.abs().mean(dim=0))
            rnorms.extend(torch.norm(t, dim=1).tolist())
    rows = len(bands)
    heat = [_downsample(b.tolist(), _WT_HEAT_COLS) for b in bands]
    flat = [x for row in heat for x in row]
    st = _wt_scalar_stats(payload, tmap, nm)
    data = {'name': nm, 'shape': tmeta.get('shape'), 'dtype': tmeta.get('dtype'),
            'stats': st,
            'heatmap': {'rows': rows, 'cols': (_WT_HEAT_COLS if rows else 0), 'data': flat},
            'rownorm': {'n': len(rnorms), 'data': _downsample(rnorms, _WT_RN_POINTS), 'sampled': sampled_any},
            'block': '32×64 行块（超限均匀抽样）'}
    _WT_T_CACHE[ck] = data
    if len(_WT_T_CACHE) > 40:
        _WT_T_CACHE.pop(next(iter(_WT_T_CACHE)))
    return data


'''
text = text.replace(anchor3, BLOCK + anchor3)

open(P, 'wb').write(text.encode('utf-8'))
print('PATCHED', len(BLOCK), 'NL=', repr(NL))
