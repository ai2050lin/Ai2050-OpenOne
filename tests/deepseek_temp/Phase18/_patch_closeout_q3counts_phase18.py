# -*- coding: utf-8 -*-
"""Phase 18 收尾脚本勘误：joint_verdict 无 Q3_counts（只有 Q3_joint）；Q6_deep_counts 的键是 DEEP。"""
import io
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P = os.path.join(ROOT, 'tests', 'deepseek', 'Phase18')

# ---- disk_verify:384
p1 = os.path.join(P, 'disk_verify_phase18.py')
s = io.open(p1, encoding='utf-8').read()
old1 = "ck('Q3 计数 BRIDGE_OK == 3', JV['Q3_counts']['BRIDGE_OK'] == 3, str(JV['Q3_counts']))"
new1 = ("_n3 = sum(1 for a in AO if V[a]['Q3_label'] == 'BRIDGE_OK')\n"
        "ck('Q3 计数 BRIDGE_OK == 3', _n3 == 3, str(_n3))")
assert s.count(old1) == 1, ('disk_verify old1 count=%d' % s.count(old1))
s = s.replace(old1, new1)
io.open(p1, 'w', encoding='utf-8', newline='\n').write(s)
print('disk_verify patched')

# ---- gen_memo: Q3_counts -> 计算；Q6_deep_counts -> DEEP
p2 = os.path.join(P, 'gen_memo_phase18.py')
t = io.open(p2, encoding='utf-8').read()
old2 = "A('**%s**（BRIDGE %d/%d）。' % (JV['Q3_joint'], JV['Q3_counts']['BRIDGE_OK'], JV['Q3_counts']['n']))"
new2 = ("A('**%s**（BRIDGE %d/%d）。'\n"
        "  % (JV['Q3_joint'], sum(1 for a in ARMS if V[a]['Q3_label'] == 'BRIDGE_OK'), len(ARMS)))")
assert t.count(old2) == 1, ('gen_memo old2 count=%d' % t.count(old2))
t = t.replace(old2, new2)
old3 = "  % (JV['Q6_deep_joint'], JV['Q6_deep_counts'].get('COMB_DEEP', 0), JV['Q6_deep_counts']['n']))"
new3 = "  % (JV['Q6_deep_joint'], JV['Q6_deep_counts'].get('DEEP', 0), JV['Q6_deep_counts']['n']))"
assert t.count(old3) == 1, ('gen_memo old3 count=%d' % t.count(old3))
t = t.replace(old3, new3)
io.open(p2, 'w', encoding='utf-8', newline='\n').write(t)
print('gen_memo patched')

# ---- 复核
for fn, need in ((p1, '_n3'), (p2, "get('DEEP'")):
    c = io.open(fn, encoding='utf-8').read()
    print(fn.split(os.sep)[-1], '->', need, 'present:', need in c, '| Q3_counts left:', c.count('Q3_counts'))
