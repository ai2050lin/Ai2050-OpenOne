# -*- coding: utf-8 -*-
"""p3121_patch2: MC loop used the full ann matrix
as fancy row index; per-step class column is
needed."""
import compileall
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3121_omega_p119_repl_causality_'
     r'erase_polarity_recon.py')
LOG = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp\p3121_patch2_log.txt')
o = []
src = io.open(P, encoding='utf-8').read()
old = (
    "        mh = seq[:, 0].copy()\n"
    "        col = ANN[dcode].astype(np.int64)\n"
    "        for k in range(N_NEW):\n"
    "            step = S * (mh - MS) \\\n"
    "                + content[dcode][col, k] \\")
new = (
    "        mh = seq[:, 0].copy()\n"
    "        for k in range(N_NEW):\n"
    "            ck = ANN[dcode][:, k] \\\n"
    "                .astype(np.int64)\n"
    "            step = S * (mh - MS) \\\n"
    "                + content[dcode][ck, k] \\")
c = src.count(old)
o.append('old count=%d' % c)
if c == 1:
    src = src.replace(old, new)
    with io.open(P, 'w', encoding='utf-8') as f:
        f.write(src)
    back = io.open(P, encoding='utf-8').read()
    o.append('disk ok=%s'
             % ("ck = ANN[dcode][:, k]" in back))
compileall.compile_file(P, force=True, quiet=2)
o.append('compile done')
io.open(LOG, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
