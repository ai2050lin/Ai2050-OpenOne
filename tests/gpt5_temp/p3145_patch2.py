# -*- coding: utf-8 -*-
"""p3145 patch2: restore rows_scan usage in
jc_ clip branch (patch1 over-deleted)."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3145_omega_p143_'
      r'v1clip_pc1spec_headsign_'
      r'residcausal.py')
t = io.open(FP, encoding='utf-8').read()

old = """        _gen = []
        for b0 in range(0, NCAP,
                        GEN_BATCH):
            batch = _rows[b0:b0
                          + GEN_BATCH]
            _gen.extend(gen_batch_g2(
                batch,
                inj_vec=[(29, dv_pc1[b0:b0
                                    + len(batch)],
                          1.0, 'allstep'),
                         (29, dv_dv29[b0:b0
                                      + len(batch)],
                          1.0, 'allstep')],
                clip=clip_spec))
"""
c = t.count(old)
assert c == 1, ('jc-branch', c)
new = """        _gen = []
        for b0 in range(0, NCAP,
                        GEN_BATCH):
            batch = rows_scan[b0:b0
                              + GEN_BATCH]
            _gen.extend(gen_batch_g2(
                batch,
                inj_vec=[(29, dv_pc1[b0:b0
                                    + len(batch)],
                          1.0, 'allstep'),
                         (29, dv_dv29[b0:b0
                                      + len(batch)],
                          1.0, 'allstep')],
                clip=clip_spec))
"""
t = t.replace(old, new)
io.open(FP, 'w', encoding='utf-8').write(t)
chk = io.open(FP, encoding='utf-8').read()
assert '_rows' not in chk, '_rows still present'
assert 'rows_scan[b0:b0\n                              + GEN_BATCH]' in chk
print('PATCH2 OK c=%d' % c)
