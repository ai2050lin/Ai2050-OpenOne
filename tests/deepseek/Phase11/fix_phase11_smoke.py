# -*- coding: utf-8 -*-
"""Phase 11 冒烟后修正：J_only / J_batch 的"其余斜率中位数"条件与 curve_stats 对齐。

冒烟证据：E3 网格被截为 4 点（含 alpha=0）=> x>=0.01 只剩 3 点 => 只有 2 个段斜率
=> np.delete 后 rest 长度 1 => `len(rest) > 1` 为假 => s_med 被强制 0.0 => J = inf
=> J_own 全 n/a。curve_stats 用的是 `len(s) > 1`（s 为全部段斜率，len(s)=len(rest)+1），
故两者的边界条件必须一致。纯实现缺陷，不改判据。
"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase11\n2h1a4_ownbasis_band.py'
t = io.open(P, encoding='utf-8').read()
n0 = len(t)

# --- 1. J_only ---
a = "    rest = np.delete(s, k_i)\n    s_med = float(np.median(rest)) if len(rest) > 1 else 0.0"
b = "    rest = np.delete(s, k_i)\n    s_med = float(np.median(rest)) if len(s) > 1 else 0.0"
assert t.count(a) == 1, 'J_only 锚点 count=%d' % t.count(a)
t = t.replace(a, b)

# --- 2. J_batch ---
c = "        rest = np.delete(S[i], ai[i])\n        med = float(np.median(rest)) if len(rest) > 1 else 0.0"
d = "        rest = np.delete(S[i], ai[i])\n        med = float(np.median(rest)) if len(S[i]) > 1 else 0.0"
assert t.count(c) == 1, 'J_batch 锚点 count=%d' % t.count(c)
t = t.replace(c, d)

# --- 3. 删除未被调用的 _stack_pairs（含无效分支，易误导） ---
e = ('def _stack_pairs(arm_pairs, sites):\n'
     '    """arm_pairs[str(s)] = [per_alpha_list]; 返回 (sites, n_alpha, n_pairs) 与 x 网格"""\n'
     '    xs = None\n'
     '    mats = []\n'
     '    for s in sites:\n'
     '        rows = arm_pairs[str(s)]\n'
     '        if xs is None:\n'
     '            xs = np.array([r[0] for r in rows]) if False else None\n'
     '        mats.append(np.array(rows, dtype=float))\n'
     '    return np.stack(mats, 0)\n\n\n')
assert t.count(e) == 1, '_stack_pairs 锚点 count=%d' % t.count(e)
t = t.replace(e, '')

io.open(P, 'w', encoding='utf-8').write(t)
print('patched %s : %d -> %d bytes' % (P, n0, len(t)))
# 落盘复核
t2 = io.open(P, encoding='utf-8').read()
print('  J_only  fixed :', 'if len(s) > 1 else 0.0' in t2)
print('  J_batch fixed :', 'if len(S[i]) > 1 else 0.0' in t2)
print('  _stack_pairs removed :', '_stack_pairs' not in t2)
assert 'if len(s) > 1 else 0.0' in t2 and 'if len(S[i]) > 1 else 0.0' in t2 and '_stack_pairs' not in t2
print('OK')
