# -*- coding: utf-8 -*-
"""补丁 4：closeout_phase20.py
 (1) 探针件真实名 = `_armrec20_probe_<arm>.json`（main() 在 PROBE 下的落盘名），原只找 `_probe20_<arm>.json` ⇒ probe_sha8 会变 None。
 (2) `xhalf` 在配对里是 dict（n_common/max_abs_dxh/...），`_f()` 会 `%.4f % dict` ⇒ TypeError。改取 max_abs_dxh。
"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase20\closeout_phase20.py'
s = io.open(P, encoding='utf-8').read()
n = 0

old1 = ("for nm in ('A0_nf4', 'A0_bf16'):\n"
        "    p = os.path.join(P20T, '_probe20_%s.json' % nm)\n"
        "    if os.path.exists(p):\n"
        "        PB[nm] = open(p, 'rb').read()\n")
new1 = ("for nm in ('A0_nf4', 'A0_bf16'):\n"
        "    for cand in ('_probe20_%s.json' % nm, '_armrec20_probe_%s.json' % nm):\n"
        "        p = os.path.join(P20T, cand)\n"
        "        if os.path.exists(p):\n"
        "            PB[nm] = open(p, 'rb').read()\n"
        "            break\n")
assert s.count(old1) == 1, 'old1 %d' % s.count(old1)
s = s.replace(old1, new1); n += 1

old2 = ("    dxh0=_f(PID, 'xhalf', 4) if QPM.get(PID) else 'NA',\n"
        "    dxh1=_f(PID2, 'xhalf', 4) if QPM.get(PID2) else 'NA',\n")
new2 = ("    dxh0=(('%.4f' % QPM[PID]['xhalf']['max_abs_dxh'])\n"
        "          if (QPM.get(PID) and QPM[PID]['xhalf'].get('max_abs_dxh') is not None) else 'NA'),\n"
        "    dxh1=(('%.4f' % QPM[PID2]['xhalf']['max_abs_dxh'])\n"
        "          if (QPM.get(PID2) and QPM[PID2]['xhalf'].get('max_abs_dxh') is not None) else 'NA'),\n")
assert s.count(old2) == 1, 'old2 %d' % s.count(old2)
s = s.replace(old2, new2); n += 1

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
print('PATCHED %d spots' % n)
for probe in ("'_armrec20_probe_%s.json' % nm", "QPM[PID]['xhalf']['max_abs_dxh']", "_f(PID, 'xhalf'"):
    print('  chk %-45s -> %d' % (probe[:45], s.count(probe)))
