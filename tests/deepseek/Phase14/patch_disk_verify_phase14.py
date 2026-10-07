# -*- coding: utf-8 -*-
"""修正 disk_verify_phase14.py 的 4 处断言缺陷（本轮独立复核首次跑出的真缺陷）。

1) E 区 null 重放的剖面构造路径写错：用了 `A1_curves` 的 y（由 curve_from_rows 产出），
   而主脚本 `full_concentration` 实际走 `PM.mean(axis=2)/FULL_SWAP`。两条路径的**求和顺序**
   不同 ⇒ 18 个位点里 8 个 xhalf 差 1 ULP（jumps max|d| = 3.331e-16）⇒ 置换 null 95 分位
   无法逐位复现（A1 4.4e-16 / A8 2.6e-15）。改为走 PM 路径（实测与 stored jumps 逐位相同）。
2) F 区把「α=1 == FULL_SWAP_pairs」的**逐对恒等式**错误地套到 A8 上：该恒等式在设计上
   只覆盖 A1（主脚本 L685 `for s in _A1_SITES`）。A8 的 α=1 **不是满替换** ——
   实测端点 y(i,1) ≡ Phase 12 `recover(site_i)`（18/18，max|d| = 0.000e+00），
   18/18 支撑的逐对向量都**不**等于 FULL_SWAP_pairs（这正是「端点近饱和 ≠ 构造饱和」）。
3) F 区 `F30a.n_pairs` 硬编码 24，实际为 432（= 18 位点 × 24 对）。
4) H 区 `P1..P7 全 PASS` 与预注册事实相反（P4/P5/P6 判 FAIL）⇒ 改为核对**确切模式**。

逐处 assert count==1 + 回读复核 + py_compile。
"""
import io
import os
import hashlib

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase14\disk_verify_phase14.py'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase14\patch_disk_verify_phase14.txt'

REPL = []

# ---------- 1) E 区 A1 null：改走 PM 路径 ----------
REPL.append((
    "nx95, nj95, _t3nx, _t3nj = replay_null(jx1, jj1, C_A1['range_x'], C_A1['range_j'])",
    """# null 重放必须走主脚本的剖面构造路径（PM.mean(axis=2)/FULL_SWAP，见主脚本 L846-L850）。
# 若改用 A1_curves 的 y（curve_from_rows 产出），求和顺序不同 ⇒ 8/18 位点差 1 ULP
# （jumps max|d| = 3.331e-16）⇒ null 95 分位无法逐位复现。
PM1 = np.stack([np.array(R['A1_perpair'][str(s)], float) for s in SITES], 0)
Y1 = PM1.mean(axis=2) / FULL_SWAP
_xh_pm1 = [cross_alpha(C_A1['alpha_grid'], Y1[i], XHF) for i in range(len(SITES))]
_Jv_pm1 = [J_only(C_A1['alpha_grid'], Y1[i]) for i in range(len(SITES))]
jx1_pm = np.diff(np.array(_xh_pm1, float)).tolist()
jj1_pm = np.diff(np.array(_Jv_pm1, float)).tolist()
d_jpm1 = max(abs(a - b) for a, b in zip(jx1_pm, C_A1['jumps_x']))
d_jjm1 = max(abs(a - b) for a, b in zip(jj1_pm, C_A1['jumps_j']))
nx95, nj95, _t3nx, _t3nj = replay_null(jx1_pm, jj1_pm, C_A1['range_x'], C_A1['range_j'])"""
))

# ---------- 1b) E 区 A8 null：同样改走 PM 路径 ----------
REPL.append((
    """# A8 的 null 用的是 A8 剖面自身的 jumps（独立重算）
_jx8 = np.diff(np.array([a8_ind_x[str(i)] for i in _a8_sites], float)).tolist()
_jj8 = np.diff(np.array([a8_ind_J[str(i)] for i in _a8_sites], float)).tolist()
x95_8, j95_8, _t8x, _t8j = replay_null(_jx8, _jj8, C_A8['range_x'], C_A8['range_j'])""",
    """# A8 的 null 同样走 PM 路径（PM.mean(axis=2)/FULL_SWAP），与主脚本逐位一致
PM8 = np.stack([np.array(R['A8_perpair'][str(i)], float) for i in _a8_sites], 0)
Y8 = PM8.mean(axis=2) / FULL_SWAP
_xh_pm8 = [cross_alpha(C_A8['alpha_grid'], Y8[i], XHF) for i in range(len(_a8_sites))]
_Jv_pm8 = [J_only(C_A8['alpha_grid'], Y8[i]) for i in range(len(_a8_sites))]
_jx8 = np.diff(np.array(_xh_pm8, float)).tolist()
_jj8 = np.diff(np.array(_Jv_pm8, float)).tolist()
d_jpm8 = max(abs(a - b) for a, b in zip(_jx8, C_A8['jumps_x']))
d_jjm8 = max(abs(a - b) for a, b in zip(_jj8, C_A8['jumps_j']))
x95_8, j95_8, _t8x, _t8j = replay_null(_jx8, _jj8, C_A8['range_x'], C_A8['range_j'])"""
))

# ---------- 1c) E 区断言：补 PM 路径逐位自检 ----------
REPL.append((
    "    ('A1 置换 null 95 分位 max|d| == 0 (%.1e)' % dnull, dnull == 0.0),",
    """    ('A1 jumps 重算（PM 路径）max|d| == 0 (x %.1e / J %.1e)' % (d_jpm1, d_jjm1),
     d_jpm1 == 0.0 and d_jjm1 == 0.0),
    ('A1 置换 null 95 分位 max|d| == 0 (%.1e)' % dnull, dnull == 0.0),"""
))
REPL.append((
    "    ('A8 置换 null 95 分位 max|d| == 0 (%.1e)' % dnull8, dnull8 == 0.0),",
    """    ('A8 jumps 重算（PM 路径）max|d| == 0 (x %.1e / J %.1e)' % (d_jpm8, d_jjm8),
     d_jpm8 == 0.0 and d_jjm8 == 0.0),
    ('A8 置换 null 95 分位 max|d| == 0 (%.1e)' % dnull8, dnull8 == 0.0),"""
))

# ---------- 2/3) F 区：A8 口径分离 + F30a.n_pairs=432 ----------
REPL.append((
    """f30a8 = {}
for i in _a8_sites:
    a8_last = sorted(np.array(R['A8_perpair'][str(i)], float)[-1].tolist())
    f30a8['A8_i%d' % i] = (len(a8_last) == nP and
                           all(abs(a - b) <= 0 for a, b in zip(a8_last, FS_SET)))
mean_ident = {}
for s in SITES:
    mean_ident['A1_L%d' % s] = ceq(float(np.mean(np.array(R['A1_perpair'][str(s)], float)[-1])), FULL_SWAP, 1e-12)
for i in _a8_sites:
    mean_ident['A8_i%d' % i] = ceq(float(np.mean(np.array(R['A8_perpair'][str(i)], float)[-1])), FULL_SWAP, 1e-12)

sec('F_alpha1_perpair_identity', [
    ('A1 全 18 位点 alpha=1 逐对 == FULL_SWAP_pairs（元素级）', all(f30.values())),
    ('A8 全 18 支撑 alpha=1 逐对 == FULL_SWAP_pairs（元素级）', all(f30a8.values())),
    ('A1 逐对均值 == FULL_SWAP (bit)', all(mean_ident[k] for k in mean_ident if k.startswith('A1'))),
    ('A8 逐对均值 == FULL_SWAP (bit)', all(mean_ident[k] for k in mean_ident if k.startswith('A8'))),
    ('R14.A8_verdict.F30a_dev == 0.0', R['A8_verdict']['F30a_dev'] == 0.0),
    ('R14.floors.F30.F30a.n_pairs == 24', R['floors']['F30']['F30a']['n_pairs'] == 24),
])""",
    """# A8 的 alpha=1 **不是**满替换：逐对恒等式只覆盖 A1（主脚本 L685 `for s in _A1_SITES`）。
# 独立核验「口径分离」：A8 每个支撑的 alpha=1 逐对向量都**不**等于 FULL_SWAP_pairs，
# 且端点 y(i,1) 逐位等于 Phase 12 `recover(site_i)`（18/18）。
f30a8_neq = {}
a8_y1 = {}
for i in _a8_sites:
    a8_last = sorted(np.array(R['A8_perpair'][str(i)], float)[-1].tolist())
    f30a8_neq['A8_i%d' % i] = (len(a8_last) == nP and a8_last != FS_SET)
    a8_y1[i] = float(R['A8_curves'][str(i)]['y'][-1])
mean_ident = {}
for s in SITES:
    mean_ident['A1_L%d' % s] = ceq(float(np.mean(np.array(R['A1_perpair'][str(s)], float)[-1])), FULL_SWAP, 1e-12)
REC12 = {int(k): float(v) for k, v in E14['inherits']['recover_12_by_site'].items()}
d_a8rec = max(abs(a8_y1[i] - REC12[SITES[i]]) for i in _a8_sites)

sec('F_alpha1_perpair_identity', [
    ('A1 全 18 位点 alpha=1 逐对 == FULL_SWAP_pairs（元素级）', all(f30.values())),
    ('A8 全 18 支撑 alpha=1 **不**等于 FULL_SWAP_pairs（口径分离）', all(f30a8_neq.values())),
    ('A1 逐对均值 == FULL_SWAP (bit)', all(mean_ident.values())),
    ('A8 端点 y(i,1) == Phase12 recover(site_i) 18/18 (max|d| %.1e)' % d_a8rec, d_a8rec == 0.0),
    ('R14.A8_verdict.F30a_dev == 0.0', R['A8_verdict']['F30a_dev'] == 0.0),
    ('R14.floors.F30.F30a.n_pairs == 432 (18 位点 x 24 对)', R['floors']['F30']['F30a']['n_pairs'] == 432),
])"""
))

# ---------- 4) H 区：P1..P7 确切模式 ----------
REPL.append((
    "    ('P1..P7 全 PASS', all(PC[k]['pass_'] for k in PC)),",
    """    ('P1..P7 == 预注册事实（P1/P2/P3/P7 真，P4/P5/P6 假）',
     all(PC[k]['pass_'] is True for k in ('P1', 'P2', 'P3', 'P7')) and
     all(PC[k]['pass_'] is False for k in ('P4', 'P5', 'P6'))),"""
))

# ---------- 汇总行 ----------
REPL.append((
    "w('alpha=1 逐对恒等式：A1 %d/18 ; A8 %d/18' % (sum(f30.values()), sum(f30a8.values())))",
    """w('alpha=1 逐对恒等式：A1 %d/18 ; A8 非恒等 %d/18 ; A8 端点==recover max|d| %.1e'
  % (sum(f30.values()), sum(f30a8_neq.values()), d_a8rec))
w('null 重放 jumps 自检（PM 路径）：A1 x %.1e / J %.1e ; A8 x %.1e / J %.1e'
  % (d_jpm1, d_jjm1, d_jpm8, d_jjm8))"""
))

b0 = open(P, 'rb').read()
t = b0.decode('utf-8')
L = ['=== patch_disk_verify_phase14 ===', 'bytes %d -> ? ; %d 处替换' % (len(b0), len(REPL)), '']
for i, (old, new) in enumerate(REPL, 1):
    c = t.count(old)
    L.append('  [%d] count=%d %s' % (i, c, 'OK' if c == 1 else '**FAIL**'))
    assert c == 1, 'REPL %d count=%d' % (i, c)
    t = t.replace(old, new)
for i, (old, new) in enumerate(REPL, 1):
    assert t.count(new) == 1, 'new %d not unique' % i
    if old in new:
        # 该处新文本**有意保留**旧行（在其后追加自检行），故旧文本仍然存在一次
        assert t.count(old) == 1, 'old %d count != 1' % i
    else:
        assert t.count(old) == 0, 'old %d still present' % i
open(P, 'wb').write(t.encode('utf-8'))
b1 = open(P, 'rb').read()
L += ['', 'bytes %d -> %d (%+d)' % (len(b0), len(b1), len(b1) - len(b0)),
      'sha256 = ' + hashlib.sha256(b1).hexdigest(),
      '残旧断言自检：count(P1..P7 全 PASS)=%d ; count(n_pairs == 24)=%d ; count(A8 全 18 支撑 alpha=1 逐对)=%d'
      % (t.count('P1..P7 全 PASS'), t.count("n_pairs == 24"), t.count('A8 全 18 支撑 alpha=1 逐对')),
      'ALL OK']
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
