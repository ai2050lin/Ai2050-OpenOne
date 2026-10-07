# -*- coding: utf-8 -*-
import io
P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase17\disk_verify_phase17.py'
s = io.open(P, encoding='utf-8').read()

o1 = "    obs, _ = _com_interval({}, [])  # placeholder, obs supplied by caller\n"
assert s.count(o1) == 1
s = s.replace(o1, "")

OLD = ("ck('E4 生产与探针 A0 com_V 逐位一致（26.1501 vs 26.15）',\n"
       "   abs(V['A0_calib_qwen3-4b-nf4']['Q3_com_V'] - float(PROBE['com_all']['all'] if isinstance(PROBE.get('com_all'), dict)\n"
       "                                                     else PROBE.get('com_V') or 26.15005633170243)) <= 5e-4,\n"
       "   '%.6f' % V['A0_calib_qwen3-4b-nf4']['Q3_com_V'])\n")
NEW = ("# 跨实现交叉验证：用本脚本的区间求和重算探针（独立脚本）自报的 com_V，二者须与生产逐位一致\n"
       "_pr = {int(l): float(PROBE['w_all'][l]) for l in range(len(PROBE['w_all']))}\n"
       "_pc, _ = _com_interval(_pr, [int(x) for x in PROBE['reach'] if 0 <= int(x) < int(PROBE['L']) - 1])\n"
       "ck('探针 w_all 区间求和重算 == 探针自报 com_V', abs(_pc - PROBE['com_V']) <= 1e-9,\n"
       "   '%.9f vs %.9f' % (_pc, PROBE['com_V']))\n"
       "ck('探针 com_V == 生产 A0 com_V（跨实现逐位一致，E4 交叉验证）',\n"
       "   abs(PROBE['com_V'] - V['A0_calib_qwen3-4b-nf4']['Q3_com_V']) <= 1e-9,\n"
       "   '%.9f vs %.9f' % (PROBE['com_V'], V['A0_calib_qwen3-4b-nf4']['Q3_com_V']))\n")
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
t = io.open(P, encoding='utf-8').read()
assert 'com_all' not in t and '探针 w_all 区间求和重算' in t
print('PATCH OK')
