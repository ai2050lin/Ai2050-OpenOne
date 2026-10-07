# -*- coding: utf-8 -*-
"""修复 disk_verify_phase10.py 中被静默丢失的 5 处编辑（幻影编辑缺陷）。
逐处 assert count==1，写完回读复核。"""
import hashlib
P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase10\disk_verify_phase10.py'
t = open(P, encoding='utf-8').read()
sha0 = hashlib.sha256(t.encode('utf-8')).hexdigest()
rep = []
def one(old, new, tag):
    global t
    c = t.count(old)
    assert c == 1, 'REP %s count=%d' % (tag, c)
    t = t.replace(old, new, 1)
    rep.append(tag)

one("chk('smoke' in J['meta'] and os.path.isdir(os.path.join(T, 'smoke')),",
    "chk('smoke_dir' in J['meta'] and os.path.isdir(os.path.join(T, 'smoke')),", 'smoke_key')

one("Q2 = (rho_all <= -0.6) and (jmin >= 1.5*max(jmax_non, jmax_all))",
    "Q2 = (rho_all <= -0.6) and (jabs[6] >= 1.5*jabs[34])   # J(min L) >= 1.5 * J(max L)", 'Q2')

one("slr, Jr = slopes_J([max(x, 0.01) for x in xr], yr)",
    "slr, Jr = slopes_J(xr, yr)", 'J_of_R')

one("chk(close(pr['gamma'], b1, 1e-3), 'gamma 重算一致', '%.6f vs %.6f' % (pr['gamma'], b1))",
    "_lx = [math.log(a) for a, b in zip(xr, yr) if a >= 0.01 and b > 0]\n"
    "_ly = [math.log(b) for a, b in zip(xr, yr) if a >= 0.01 and b > 0]\n"
    "_g = (len(_lx)*sum(a*b for a, b in zip(_lx, _ly)) - sum(_lx)*sum(_ly)) / (len(_lx)*sum(a*a for a in _lx) - sum(_lx)**2)\n"
    "chk(close(pr['gamma'], _g, 1e-6), 'gamma 重算一致（幂律 log-log 斜率）', '%.6f vs %.6f' % (pr['gamma'], _g))",
    'gamma')

one("chk(b'2036' if False else True, 'MEMO 追加已复核（见 verify_append_phase10.txt）')",
    "chk(os.path.isfile(os.path.join(S, 'verify_append_phase10.txt')), 'append 复核文件存在（verify_append_phase10.txt）')",
    'append_file')

open(P, 'w', encoding='utf-8', newline='').write(t)
t2 = open(P, encoding='utf-8').read()
assert t2 == t, 'roundtrip mismatch'
print('patched:', rep)
print('bytes %d sha8 %s -> %s' % (len(t.encode('utf-8')), sha0[:8], hashlib.sha256(t2.encode('utf-8')).hexdigest()[:8]))
