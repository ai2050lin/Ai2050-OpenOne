# -*- coding: utf-8 -*-
"""SMOKE 抓到的缺陷修补：R11['profile_abs'] 含非数字键 'R'。"""
import io, hashlib, py_compile

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase12\n2h1a5_swap_alloc.py'
t = io.open(P, encoding='utf-8').read()

OLD = "J_INJECT = {int(k): v['jump_ratio'] for k, v in R11['profile_abs'].items()}"
NEW = ("J_INJECT = {}\n"
       "for _k, _v in R11['profile_abs'].items():\n"
       "    try:\n"
       "        J_INJECT[int(_k)] = _v['jump_ratio']\n"
       "    except (TypeError, ValueError):\n"
       "        pass   # payload 里含 'R' 等非数字键（SMOKE 抓到）")

n = t.count(OLD)
assert n == 1, '期望 1 处，实际 %d' % n
t = t.replace(OLD, NEW, 1)
io.open(P, 'w', encoding='utf-8').write(t)

t2 = io.open(P, encoding='utf-8').read()
assert t2.count("J_INJECT[int(_k)] = _v['jump_ratio']") == 1
assert t2.count("J_INJECT = {int(k): v['jump_ratio']") == 0
py_compile.compile(P, doraise=True)
print('[fix12] OK ; sha8=%s ; py_compile OK' % hashlib.sha256(t2.encode('utf-8')).hexdigest()[:8])
