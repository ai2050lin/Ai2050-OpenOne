# -*- coding: utf-8 -*-
"""p3121_patch3: Part D tail steps can have zero
syntax/fact tokens -> mean_dm None.  Guard the npz
save (None -> nan) and the min/max report; count
zero-sample steps."""
import compileall
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3121_omega_p119_repl_causality_'
     r'erase_polarity_recon.py')
LOG = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp\p3121_patch3_log.txt')
o = []
src = io.open(P, encoding='utf-8').read()


def rep(old, new, tag):
    global src
    c = src.count(old)
    if c == 1:
        src = src.replace(old, new)
        o.append('%s: applied' % tag)
    else:
        o.append('%s: SKIP count=%d' % (tag, c))


rep(
    "npz_out['pertab_dm_P'] = np.array(\n"
    "    [[r['syntax']['mean_dm'] for r in\n"
    "      pertab['P']],\n"
    "     [r['fact_strict']['mean_dm'] for r in\n"
    "      pertab['P']],\n"
    "     [r['other']['mean_dm'] for r in\n"
    "      pertab['P']]], dtype=np.float64)\n"
    "npz_out['pertab_dm_A1'] = np.array(\n"
    "    [[r['syntax']['mean_dm'] for r in\n"
    "      pertab['A1']],\n"
    "     [r['fact_strict']['mean_dm'] for r in\n"
    "      pertab['A1']],\n"
    "     [r['other']['mean_dm'] for r in\n"
    "      pertab['A1']]], dtype=np.float64)",
    "def _pt(rows, nm):\n"
    "    return [np.nan\n"
    "            if r[nm]['mean_dm'] is None\n"
    "            else r[nm]['mean_dm']\n"
    "            for r in rows]\n"
    "\n"
    "\n"
    "npz_out['pertab_dm_P'] = np.array(\n"
    "    [_pt(pertab['P'], 'syntax'),\n"
    "     _pt(pertab['P'], 'fact_strict'),\n"
    "     _pt(pertab['P'], 'other')],\n"
    "    dtype=np.float64)\n"
    "npz_out['pertab_dm_A1'] = np.array(\n"
    "    [_pt(pertab['A1'], 'syntax'),\n"
    "     _pt(pertab['A1'], 'fact_strict'),\n"
    "     _pt(pertab['A1'], 'other')],\n"
    "    dtype=np.float64)",
    'P3a-npz-nan-guard')

rep(
    "        'syntax_min_P': min(\n"
    "            r['syntax']['mean_dm']\n"
    "            for r in pertab['P']),\n"
    "        'syntax_max_P': max(\n"
    "            r['syntax']['mean_dm']\n"
    "            for r in pertab['P']),\n"
    "        'syntax_min_A1': min(\n"
    "            r['syntax']['mean_dm']\n"
    "            for r in pertab['A1']),\n"
    "        'syntax_max_A1': max(\n"
    "            r['syntax']['mean_dm']\n"
    "            for r in pertab['A1']),",
    "        'syntax_n_zero_P': sum(\n"
    "            1 for r in pertab['P']\n"
    "            if r['syntax']['mean_dm']\n"
    "            is None),\n"
    "        'syntax_n_zero_A1': sum(\n"
    "            1 for r in pertab['A1']\n"
    "            if r['syntax']['mean_dm']\n"
    "            is None),\n"
    "        'syntax_min_P': min(\n"
    "            (r['syntax']['mean_dm']\n"
    "             for r in pertab['P']\n"
    "             if r['syntax']['mean_dm']\n"
    "             is not None), default=None),\n"
    "        'syntax_max_P': max(\n"
    "            (r['syntax']['mean_dm']\n"
    "             for r in pertab['P']\n"
    "             if r['syntax']['mean_dm']\n"
    "             is not None), default=None),\n"
    "        'syntax_min_A1': min(\n"
    "            (r['syntax']['mean_dm']\n"
    "             for r in pertab['A1']\n"
    "             if r['syntax']['mean_dm']\n"
    "             is not None), default=None),\n"
    "        'syntax_max_A1': max(\n"
    "            (r['syntax']['mean_dm']\n"
    "             for r in pertab['A1']\n"
    "             if r['syntax']['mean_dm']\n"
    "             is not None), default=None),",
    'P3b-report-guard')

with io.open(P, 'w', encoding='utf-8') as f:
    f.write(src)
back = io.open(P, encoding='utf-8').read()
o.append('disk check: a=%s b=%s'
         % ("def _pt(rows, nm)" in back,
            "syntax_n_zero_P" in back))
compileall.compile_file(P, force=True, quiet=2)
o.append('compile done')
io.open(LOG, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
