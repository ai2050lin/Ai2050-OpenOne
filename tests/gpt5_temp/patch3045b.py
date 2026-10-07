# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3045_omega_p42_l20_axis_anatomy_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

old = ("a111_min = 1.0\n"
       "for mine, refn in ((UBAR3, 'ubar3'),\n"
       "                   (UBAR20, 'ubar20'),\n"
       "                   (ALPHAS3[0], 'alpha3'),\n"
       "                   (ALPHAS20[0], 'alpha20')):\n"
       "    rb = z44[refn]\n"
       "    cs = float(mine @ rb) / (\n"
       "        np.linalg.norm(mine)\n"
       "        * np.linalg.norm(rb))\n"
       "    a111_min = min(a111_min, cs)\n")
new = ("a111_min = 1.0\n"
       "for mine, refn in ((UBAR3, 'ubar3'),\n"
       "                   (UBAR20, 'ubar20'),\n"
       "                   (ALPHAS3, 'alpha3'),\n"
       "                   (ALPHAS20, 'alpha20')):\n"
       "    rb = z44[refn]\n"
       "    if rb.ndim == 2:\n"
       "        for r in range(rb.shape[0]):\n"
       "            cs = float(mine[r] @ rb[r]) \\\n"
       "                / (np.linalg.norm(mine[r])\n"
       "                   * np.linalg.norm(rb[r]))\n"
       "            a111_min = min(a111_min, cs)\n"
       "    else:\n"
       "        cs = float(mine @ rb) / (\n"
       "            np.linalg.norm(mine)\n"
       "            * np.linalg.norm(rb))\n"
       "        a111_min = min(a111_min, cs)\n")
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
out = ('patch3045b ok: a111 shape-aware; compile zero '
       'errors')
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3045b_result.txt', 'w',
        encoding='utf-8').write(out)
print(out)
