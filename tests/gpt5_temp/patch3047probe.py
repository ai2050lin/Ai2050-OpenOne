# -*- coding: utf-8 -*-
import io

P = r'D:\AI2050\Ai2050-OpenOne\gpt5_temp\probe3047d.py'
s = io.open(P, encoding='utf-8').read()
old = "rel = e_sub / float(np.abs(dt).max())"
new = "rel = e_sub / float(np.abs(dt.cpu().numpy()).max())"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8').write(s)
print('ok')
