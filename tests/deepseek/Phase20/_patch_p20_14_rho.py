# -*- coding: utf-8 -*-
"""补丁 14：修 gen_memo 中 `rho_b_all`（dict）被 `q()` 当标量渲染的缺陷（3 处）。"""
import io
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D20 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase20')
P = os.path.join(D20, 'gen_memo_phase20.py')
s = io.open(P, encoding='utf-8').read()
LOG = []


def rep(old, new, tag, count=1):
    global s
    n = s.count(old)
    assert n == count, '[%s] 期望 %d 处，实为 %d' % (tag, count, n)
    s = s.replace(old, new)
    LOG.append('  OK  %s' % tag)


# 新增标量取 rho 的辅助
OLD = ("def qd(pair, key, nd=4):\n"
       "    s = QP.get(pair)\n"
       "    if not s:\n"
       "        return 'NA'\n"
       "    v = (s.get(key) or {}).get('delta')\n"
       "    return f(v, nd) if isinstance(v, (int, float)) else 'NA'\n")
NEW = OLD + ("\n\n"
             "def qr(pair, nd=4):\n"
             "    \"\"\"rho_b_all 是 dict{rho,resid_med,resid_p90} ⇒ 取标量 rho（勿用 q()）。\"\"\"\n"
             "    s = QP.get(pair)\n"
             "    if not s:\n"
             "        return 'NA'\n"
             "    v = (s.get('rho_b_all') or {}).get('rho')\n"
             "    return f(v, nd) if isinstance(v, (int, float)) else 'NA'\n")
rep(OLD, NEW, 'qr 辅助')

rep("A('行为谱秩相关 **' + q(PID, 'rho_b_all') + ' / ' + q(PID2, 'rho_b_all') + '**，')",
    "A('行为谱秩相关 **' + qr(PID) + ' / ' + qr(PID2) + '**，')",
    '§0 rho 标量')

rep("      + ' | ' + q(k, 'rho_b_all') + ' | ' + f(p['rho_b_all']['resid_med'], 4) + ' / '",
    "      + ' | ' + qr(k) + ' | ' + f(p['rho_b_all']['resid_med'], 4) + ' / '",
    '§4 rho 标量')

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
io.open(os.path.join(D20, '_patch_p20_14_rho.log'), 'w', encoding='utf-8', newline='\n')\
    .write('\n'.join(LOG) + '\n')
print('\n'.join(LOG))
print('patched:', len(LOG))
