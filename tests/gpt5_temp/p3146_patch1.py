# -*- coding: utf-8 -*-
"""p3146 patch1: fix E_TRIALS naming bug
('%g' % 1.0 -> '1' != '1.0') and the same
bug in gap_by_dose key construction.
Hard-code trial names instead."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
      r'\phase3146_omega_p144_taillocus_'
      r'histfield_pcresdose_cleanclip.py')

t = io.open(FP, encoding='utf-8').read()

old1 = """E_TRIALS = []
for dsc in PCRES_DOSES:
    for sg, sname in ((1.0, 'pos'),
                      (-1.0, 'neg')):
        E_TRIALS.append(
            ('e_pcres38_d%s_%s'
             % (('%g' % dsc)
                .replace('0.', 'd0.')
                .replace('1.', 'd1.'),
                sname),
             38, sg, dsc))
"""
new1 = """E_TRIALS = [
    ('e_pcres38_d0.25_pos', 38, 1.0,
     0.25),
    ('e_pcres38_d0.25_neg', 38, -1.0,
     0.25),
    ('e_pcres38_d0.5_pos', 38, 1.0, 0.5),
    ('e_pcres38_d0.5_neg', 38, -1.0,
     0.5),
    ('e_pcres38_d1.0_pos', 38, 1.0, 1.0),
    ('e_pcres38_d1.0_neg', 38, -1.0,
     1.0)]
"""
c1 = t.count(old1)
assert c1 == 1, ('e_trials', c1)
t = t.replace(old1, new1)

old2 = """gap_by_dose = {}
for dsc in PCRES_DOSES:
    ds = ('%g' % dsc)
    tpos = ('e_pcres38_d%s_pos'
            % ds.replace('0.', 'd0.')
            .replace('1.', 'd1.'))
    tneg = ('e_pcres38_d%s_neg'
            % ds.replace('0.', 'd0.')
            .replace('1.', 'd1.'))
    gap_by_dose[dsc] = _chg(tpos) \\
        - _chg(tneg)
"""
new2 = """gap_by_dose = {}
for dsc, tpos, tneg in (
        (0.25, 'e_pcres38_d0.25_pos',
         'e_pcres38_d0.25_neg'),
        (0.5, 'e_pcres38_d0.5_pos',
         'e_pcres38_d0.5_neg'),
        (1.0, 'e_pcres38_d1.0_pos',
         'e_pcres38_d1.0_neg')):
    gap_by_dose[dsc] = _chg(tpos) \\
        - _chg(tneg)
"""
c2 = t.count(old2)
assert c2 == 1, ('gap_dose', c2)
t = t.replace(old2, new2)

io.open(FP, 'w', encoding='utf-8').write(t)

chk = io.open(FP, encoding='utf-8').read()
assert "e_pcres38_d1.0_pos" in chk
assert "('%g' % dsc)" not in chk
assert chk.count("E_TRIALS = [") == 1
print('PATCH1 OK c1=%d c2=%d' % (c1, c2))
