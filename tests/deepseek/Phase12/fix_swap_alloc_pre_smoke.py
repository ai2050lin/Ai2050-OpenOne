# -*- coding: utf-8 -*-
"""Phase 12 主脚本的预冒烟补丁（铁律 o：关键修改走 Python 补丁 + assert count==1 + 回读复核）"""
import io, os, hashlib

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase12\n2h1a5_swap_alloc.py'
t = io.open(P, encoding='utf-8').read()
orig_len = len(t)


def rep(old, new, tag):
    global t
    n = t.count(old)
    assert n == 1, 'PATCH %s: 期望 1 处，实际 %d 处' % (tag, n)
    t = t.replace(old, new, 1)
    print('[patch12] %s OK (count=1)' % tag)


# --- P1: 存 hd_ell（供体贴的真值，供 F12 核对） ---
rep("    d = dict(u6=u6, n6=n6, unit6=(u6 / max(n6, 1e-9)),\n"
    "             hR=CAP[rw][1].astype(np.float32), nhR=float(np.linalg.norm(CAP[rw][1])),\n"
    "             h_ell={}, nh_ell={}, d_ell={})",
    "    d = dict(u6=u6, n6=n6, unit6=(u6 / max(n6, 1e-9)),\n"
    "             hR=CAP[rw][1].astype(np.float32), nhR=float(np.linalg.norm(CAP[rw][1])),\n"
    "             h_ell={}, hd_ell={}, nh_ell={}, d_ell={})",
    'P1-init-hd_ell')

rep("        d['h_ell'][s] = hr\n"
    "        d['nh_ell'][s] = float(np.linalg.norm(hr))\n"
    "        d['d_ell'][s] = hd - hr",
    "        d['h_ell'][s] = hr\n"
    "        d['hd_ell'][s] = hd\n"
    "        d['nh_ell'][s] = float(np.linalg.norm(hr))\n"
    "        d['d_ell'][s] = hd - hr",
    'P1-store-hd_ell')

# --- P2: 重写 F12 块（原版含死代码与 mixed-type sorted） ---
OLD_F12 = """w('--- F12 自检：alpha=1 的替换向量必须精确等于供体贴残差 ---')
f12 = {}
for site in sorted(set(F3_SITES) and set([7, 20, 34] + [R_SITE])):
    devs = []
    for rw in list(VEC.keys())[:3]:
        V = VEC[rw]
        h0, _ = base_state(V, site)
        hd = V['hR_donor'] if site == R_SITE else None
        devs.append(0.0)   # 占位，下面用统一口径覆盖
    f12[str(site)] = 0.0
# 统一口径：对每个位点，用 diff 重建供体贴残差并比较
for site in [6, 7, 20, 34, R_SITE]:
    vals = []
    for rw in VEC:
        V = VEC[rw]
        if site == R_SITE:
            lhs = V['hR'] + 1.0 * V['d_R']
            rhs = V['hR_donor']
        else:
            if site not in V['d_ell']:
                continue
            lhs = V['h_ell'][site] + 1.0 * V['d_ell'][site]
            hd = None
            # 重建供体贴：base + diff 已经等价，故用差分残差直接核对
            rhs = lhs
        if rhs is None:
            continue
        den = max(float(np.linalg.norm(rhs)), 1e-9)
        vals.append(float(np.linalg.norm(lhs - rhs)) / den)
    f12[str(site)] = float(max(vals)) if vals else 0.0
w('  max relative reconstruction error : %s' %
  '  '.join('%s=%.3e' % (k, v) for k, v in f12.items()))
w('  (alpha=1 的语义由 diff_ell := h_ell(donor) - h_ell(recip) 的构造保证；R 位点用 hR_donor 直接核对)')
assert all(v < 1e-6 for v in f12.values()), 'F12 失败：满替换不是精确的供体贴残差'"""

NEW_F12 = """w('--- F12 自检：alpha=1 的替换向量必须等于供体贴残差（float32 重建误差范围内）---')
f12 = {}
for site in [6, 7, 20, 34, R_SITE]:
    vals = []
    for rw in VEC:
        V = VEC[rw]
        if site == R_SITE:
            lhs = V['hR'] + 1.0 * V['d_R']
            rhs = V['hR_donor']
        else:
            if site not in V['hd_ell']:
                continue
            lhs = V['h_ell'][site] + 1.0 * V['d_ell'][site]
            rhs = V['hd_ell'][site]
        den = max(float(np.linalg.norm(rhs)), 1e-9)
        vals.append(float(np.linalg.norm(lhs - rhs)) / den)
    f12[str(site)] = float(max(vals)) if vals else 0.0
w('  max relative reconstruction error : %s' %
  '  '.join('%s=%.3e' % (k, v) for k, v in f12.items()))
w('  (alpha=1 的语义由 diff := h(donor) - h(recip) 的构造保证；此处核对 float32 重建误差)')
assert all(v < 1e-5 for v in f12.values()), 'F12 失败：满替换不是精确的供体贴残差'"""

rep(OLD_F12, NEW_F12, 'P2-rewrite-F12')

# --- P3: confP 初始化提到 if 外 ---
rep("conf_out = {}\nif CF_SITES and CONF_P:",
    "conf_out = {}\nconfP = {}\nif CF_SITES and CONF_P:",
    'P3-init-confP')

rep("    G5 = grid_of('E5_conf_swap', 3)\n    confP = {}; conf_rec = {}",
    "    G5 = grid_of('E5_conf_swap', 3)\n    conf_rec = {}",
    'P3-drop-inner-confP')

# --- P4: 报告里的 F7'' 引号（避免 python 相邻字符串拼接歧义） ---
rep("w('  置换零假设 95%% 带 = [%s, %s] (F7'' 要求 |界| < 0.6) -> %s' % (",
    "w('  置换零假设 95%% 带 = [%s, %s] (F7(dbl-prime) 要求 |界| < 0.6) -> %s' % (",
    'P4-F7pp-label')

assert len(t) != orig_len
io.open(P, 'w', encoding='utf-8').write(t)

# --- 回读复核 ---
t2 = io.open(P, encoding='utf-8').read()
checks = {
    "d['hd_ell'][s] = hd": 1,
    "h_ell={}, hd_ell={}, nh_ell={}, d_ell={}": 1,
    "--- F12 自检：alpha=1 的替换向量必须等于供体贴残差（float32 重建误差范围内）---": 1,
    "rhs = V['hd_ell'][site]": 1,
    "assert all(v < 1e-5 for v in f12.values())": 1,
    "conf_out = {}\nconfP = {}\nif CF_SITES and CONF_P:": 1,
    "    conf_rec = {}": 1,
    "F7(dbl-prime) 要求": 1,
}
bad = []
for k, n in checks.items():
    c = t2.count(k)
    if c != n:
        bad.append((k, c, n))
print('[patch12] readback checks:', 'ALL OK' if not bad else bad)
print('[patch12] old_len=%d new_len=%d sha8=%s' % (orig_len, len(t2), hashlib.sha256(t2.encode('utf-8')).hexdigest()[:8]))
assert not bad
# 语法自检
import py_compile
py_compile.compile(P, doraise=True)
print('[patch12] py_compile OK')
