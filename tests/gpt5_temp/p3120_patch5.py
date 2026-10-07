# -*- coding: utf-8 -*-
"""p3120_patch5:
1) MEMO title: restore brackets around the Phase
   3120 timestamp ([[NOW]] replace ate them).
2) closeout source: fix the same bug for future
   reruns.
3) disk verify: align dtype paths (float32 subtract
   then astype f64), xg astype, and t2 on centered
   xc/yc.
Every edit: count==1 assertion; compile checks for
both patched scripts; write-back; reread verify."""
import compileall
import io

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = ROOT + r'\research\gpt5\docs\AGI_GPT5_MEMO.md'
CLOSE = ROOT + (r'\tests\gpt5_temp'
                r'\phase3120_closeout.py')
DV = ROOT + (r'\tests\gpt5_temp'
             r'\p3120_disk_verify.py')
LOG = ROOT + (r'\tests\gpt5_temp'
              r'\p3120_patch5_log.txt')
o = []


def edit(path, old, new, tag):
    src = io.open(path, encoding='utf-8').read()
    c = src.count(old)
    if c == 1:
        src = src.replace(old, new)
        with io.open(path, 'w',
                     encoding='utf-8') as f:
            f.write(src)
        back = io.open(path, encoding='utf-8').read()
        o.append('%s: applied (disk ok=%s)'
                 % (tag, new in back))
    elif old in src and new in src:
        o.append('%s: already applied' % tag)
    else:
        o.append('%s: SKIP count=%d new_present=%s'
                 % (tag, c, new in src))


# 1) MEMO timestamp brackets
edit(MEMO,
     u'\u626d\u66f2** 2026-09-23 16:57\n',
     u'\u626d\u66f2** [2026-09-23 16:57]\n',
     'MEMO-timestamp')

# 2) closeout source (future reruns)
edit(CLOSE,
     "sec = sec.replace('[[NOW]]', NOW)",
     "sec = sec.replace('[[NOW]]', "
     "'[' + NOW + ']')",
     'CLOSEOUT-nowfix')

# 3) disk verify dtype/t2 alignment
edit(DV,
     "mP = z18['gt_cleanseq_clean__P']"
     ".astype(np.float64)\n"
     "mA = z18['gt_cleanseq_clean__A1']"
     ".astype(np.float64)",
     "mP = z18['gt_cleanseq_clean__P']"
     "   # keep float32\n"
     "mA = z18['gt_cleanseq_clean__A1']"
     "  # keep float32",
     'DV-f32keep')
edit(DV,
     "dmP = (mP[:, 1:] - mP[:, :-1])[:, 1:]\n"
     "dmA = (mA[:, 1:] - mA[:, :-1])[:, 1:]",
     "dmP = (mP[:, 1:] - mP[:, :-1]).astype(\n"
     "    np.float64)[:, 1:]\n"
     "dmA = (mA[:, 1:] - mA[:, :-1]).astype(\n"
     "    np.float64)[:, 1:]",
     'DV-dm-f32sub')
edit(DV,
     "    x = mD[:, :-1][:, 1:]",
     "    x = mD[:, :-1].astype("
     "np.float64)[:, 1:]",
     'DV-gate-x-f64')
edit(DV,
     "xg = gap[:, :-1][:, 1:]",
     "xg = gap[:, :-1].astype("
     "np.float64)[:, 1:]",
     'DV-gap-xg-f64')
edit(DV,
     "sp2 = spearman(XS[sel2], YS[sel2])\n"
     "r2t2, _ = ols_fit([XS[sel2]], YS[sel2])",
     "sp2 = spearman(xc[sel2], yc[sel2])\n"
     "r2t2, _ = ols_fit([xc[sel2]], yc[sel2])",
     'DV-t2-centered')

# compile checks on real disk
compileall.compile_file(DV, force=True, quiet=2)
compileall.compile_file(CLOSE, force=True, quiet=2)
o.append('compile done')

# reread verify for every edit
back_m = io.open(MEMO, encoding='utf-8').read()
o.append('verify MEMO bracket=%s'
         % (u'\u626d\u66f2** [2026-09-23 16:57]'
            in back_m))
back_c = io.open(CLOSE, encoding='utf-8').read()
o.append('verify closeout nowfix=%s'
         % ("'[' + NOW + ']'" in back_c))
back_d = io.open(DV, encoding='utf-8').read()
o.append('verify dv edits=%s'
         % ('# keep float32' in back_d
            and 'xc[sel2]' in back_d))
io.open(LOG, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
