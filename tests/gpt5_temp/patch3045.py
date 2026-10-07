# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3045_omega_p42_l20_axis_anatomy_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

old1 = ("for bi in range(len(NEW_BODIES)):\n"
        "    s = NEW_BODIES[bi]\n"
        "    ids = [int(x) for x in tok(\n"
        "        s, add_special_tokens=False)"
        "['input_ids']]\n"
        "    t = word_tok[NEW_TARGETS[bi]]\n"
        "    assert ids.count(t) == 1, bi\n"
        "    assembled.append({'ids': ids,\n"
        "                      'pos': ids.index(t),\n"
        "                      'cond': 0, 'body': bi,\n"
        "                      'new': True})\n"
        "n_pr = len(assembled)\n"
        "n_old = len(BODIES) * len(PREFIXES)\n"
        "assert n_old == 32 and n_pr == 36\n")
new1 = ("for bi in range(len(NEW_BODIES)):\n"
        "    for ci in range(len(PREFIXES)):\n"
        "        s = (PREFIXES[ci] + ' ' + "
        "NEW_BODIES[bi]) \\\n"
        "            if PREFIXES[ci] else "
        "NEW_BODIES[bi]\n"
        "        ids = [int(x) for x in tok(\n"
        "            s, add_special_tokens=False)[\n"
        "            'input_ids']]\n"
        "        t = word_tok[NEW_TARGETS[bi]]\n"
        "        assert ids.count(t) == 1, (bi, ci)\n"
        "        assembled.append({'ids': ids,\n"
        "                          'pos': "
        "ids.index(t),\n"
        "                          'cond': ci,\n"
        "                          'body': bi,\n"
        "                          'new': True})\n"
        "n_pr = len(assembled)\n"
        "n_old = len(BODIES) * len(PREFIXES)\n"
        "assert n_old == 32 and n_pr == 48\n")
assert s.count(old1) == 1, ('asm', s.count(old1))
s = s.replace(old1, new1)

old2 = ("for b in range(len(BODIES)):\n"
        "    GBAR3[b] = G3[:, b].mean()\n")
new2 = ("for b in range(len(BODIES)):\n"
        "    GBAR3[b] = G3[:, b].mean()\n"
        "D20_new = np.zeros((12, HDIM_V))\n"
        "GBAR20N = np.zeros(len(NEW_BODIES))\n"
        "k_new = 0\n"
        "for b in range(len(NEW_BODIES)):\n"
        "    vals = []\n"
        "    for c in (1, 2, 3):\n"
        "        ic = idx_of[(c, b, True)]\n"
        "        ib = idx_of[(0, b, True)]\n"
        "        d = Vbase[20][ic] - Vbase[20][ib]\n"
        "        D20_new[k_new] = d\n"
        "        vals.append(float(\n"
        "            np.linalg.norm(d)))\n"
        "        k_new += 1\n"
        "    GBAR20N[b] = float(np.mean(vals))\n")
assert s.count(old2) == 1, ('gbar', s.count(old2))
s = s.replace(old2, new2)

old3 = ("    g = float(GBAR20_new[b]) \\\n"
        "        if False else float(GBAR20N[b])\n")
new3 = "    g = float(GBAR20N[b])\n"
assert s.count(old3) == 1, ('g', s.count(old3))
s = s.replace(old3, new3)

assert 'if False' not in s, 'if False left'
assert 'GBAR20_new' not in s, 'ghost name left'
io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)

out = ('patch3045 ok: new bodies 4-cond assembly, '
       'GBAR20N built, T3 g fixed; compile zero '
       'errors; len=%d' % len(s))
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3045_result.txt', 'w',
        encoding='utf-8').write(out)
print(out)
