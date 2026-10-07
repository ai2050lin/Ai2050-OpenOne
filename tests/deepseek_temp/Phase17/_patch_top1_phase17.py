# -*- coding: utf-8 -*-
"""一次性补丁：把 closeout_phase17.py 中残留的 top1=... 手工表达式替换为 _top1_tri(3)。
（幻影编辑缺陷修复：Edit 报成功但未落盘 -> 改用 Python patch + assert count==1 + 回读复核。）"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase17\closeout_phase17.py'
s = io.open(P, encoding='utf-8').read()

OLD = "    top1='/'.join('%.3f' % float(RES['arms'][a]['E5_com_V']['top1_head_share_nb']) for a in AO),\n"
NEW = "    top1=_top1_tri(3),\n"

n = s.count(OLD)
print('count(OLD) =', n)
assert n == 1, 'expect exactly 1 occurrence, got %d' % n

s2 = s.replace(OLD, NEW)
assert s2 != s
io.open(P, 'w', encoding='utf-8', newline='\n').write(s2)

# 回读复核
t = io.open(P, encoding='utf-8').read()
assert "    top1=_top1_tri(3),\n" in t and OLD not in t, 'patch failed on re-read'
print('PATCH OK; top1 occurrences of _top1_tri(3):', t.count('_top1_tri(3)'))
