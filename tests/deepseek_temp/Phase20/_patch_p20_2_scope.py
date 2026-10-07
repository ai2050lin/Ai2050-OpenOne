# -*- coding: utf-8 -*-
"""补丁 P20-2：(a) PROBE 用**全量配对**、只缩网格（seal 前的 A0 读数才有意义）；
(b) 锚复现只在**全尺度**运行（SMOKE/PROBE 下网格缩幅 ⇒ 与冻结锚不可比，标 N/A）。"""
import io
P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase20\n2h1a13_quant_scheme_robustness.py'
s = io.open(P, encoding='utf-8').read()

old1 = """if SMOKE or PROBE:
    BP = 200
    PAIRS_ALL = PAIRS_ALL[:8]
    _keep = []
    for _p in PAIRS_ALL:
        for _x in (_p[0], _p[2]):
            if _x not in _keep:
                _keep.append(_x)
    _sel = [t for t in INST_ALL if t[0] in _keep]
    INST_ALL = _sel if len(_sel) >= 4 else INST_ALL[:8]
    DISC = [d for d in DISC if d[0] in set(x[0] for x in INST_ALL)]
    CONF = [c for c in CONF if c[0] in set(x[0] for x in INST_ALL)]
if PROBE:
    # 缩幅网格（仍覆盖全域范围；用于 seal 前的 A0 观测）
    _ps = [1, 2, 3, 4, 5] + list(range(6, 35, 2))
    PROFILE = [s for s in _ps]
    ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
if SMOKE:
    PROFILE = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]"""
new1 = """FULL_SCALE = (not SMOKE) and (not PROBE)
if SMOKE:
    # SMOKE 只求「跑通」：同时缩配对与网格
    BP = 200
    PAIRS_ALL = PAIRS_ALL[:8]
    _keep = []
    for _p in PAIRS_ALL:
        for _x in (_p[0], _p[2]):
            if _x not in _keep:
                _keep.append(_x)
    _sel = [t for t in INST_ALL if t[0] in _keep]
    INST_ALL = _sel if len(_sel) >= 4 else INST_ALL[:8]
    DISC = [d for d in DISC if d[0] in set(x[0] for x in INST_ALL)]
    CONF = [c for c in CONF if c[0] in set(x[0] for x in INST_ALL)]
    PROFILE = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
if PROBE:
    # 可行性探针：**保留全量配对与实例**（U_l 秩 = 5 才成立），只缩小网格与 BP
    BP = 200
    PROFILE = [1, 2, 3, 4, 5] + list(range(6, 35, 2))
    ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]"""
assert s.count(old1) == 1, ('old1', s.count(old1))
s = s.replace(old1, new1)

old2 = """    rec['E9_anchor'] = dict(ok=bool(ok), applies=(scheme == 'nf4'), detail=ad)
    w('  E9 锚复现[%s]: %s' % (scheme, ('OK' if ok else 'DRIFT') if scheme == 'nf4' else 'N/A(处理臂)'))
    if scheme == 'nf4' and not ok:"""
new2 = """    rec['E9_anchor'] = dict(ok=bool(ok), full_scale=bool(FULL_SCALE),
                            applies=bool(scheme == 'nf4' and FULL_SCALE), detail=ad,
                            note=('全尺度下适用' if FULL_SCALE else 'SMOKE/PROBE 缩幅网格 ⇒ 与冻结锚不可比，标注 N/A'))
    _app = rec['E9_anchor']['applies']
    w('  E9 锚复现[%s]: %s' % (scheme, ('OK' if ok else 'DRIFT') if _app
                              else 'N/A(%s)' % ('处理臂' if scheme != 'nf4' else '缩幅网格')))
    if _app and not ok:"""
assert s.count(old2) == 1, ('old2', s.count(old2))
s = s.replace(old2, new2)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
s2 = io.open(P, encoding='utf-8').read()
assert 'FULL_SCALE' in s2 and s2.count('FULL_SCALE') >= 3, s2.count('FULL_SCALE')
assert 'applies=bool(scheme == ' in s2
print('PATCH P20-2 OK')
