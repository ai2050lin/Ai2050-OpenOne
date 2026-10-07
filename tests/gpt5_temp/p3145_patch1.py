# -*- coding: utf-8 -*-
"""p3145 patch1: fix cross-line dict index
bug (chg_bc/chg_dc/chg_jc) + remove dead
elif block + remove unused _inj_vec line."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3145_omega_p143_'
      r'v1clip_pc1spec_headsign_'
      r'residcausal.py')
t = io.open(FP, encoding='utf-8').read()

# fix 1: dead elif block in PART A
old1 = """    elif tn in pt44['bit_anchors_B'] \\
            if 'bit_anchors_B' in pt44 \\
            else False:
        pass
"""
c1 = t.count(old1)
assert c1 == 1, ('dead-elif', c1)
t = t.replace(old1, '')

# fix 2: cross-line dict index bug
old2 = """chg_bc = clip_res['bc_l%02d' % CLIP_MAIN]
['chg']
chg_dc = clip_res['dc_l%02d' % CLIP_MAIN]
['chg']
chg_jc = clip_res['jc_l%02d' % CLIP_MAIN]
['chg']
"""
c2 = t.count(old2)
assert c2 == 1, ('crossline', c2)
new2 = """_m38 = CLIP_MAIN
chg_bc = clip_res['bc_l%02d' % _m38]['chg']
chg_dc = clip_res['dc_l%02d' % _m38]['chg']
chg_jc = clip_res['jc_l%02d' % _m38]['chg']
"""
t = t.replace(old2, new2)

# fix 3: unused _inj_vec assignment
old3 = """        _rows = rows_scan
        _b12 = base12_P
        _inj_vec = None
"""
c3 = t.count(old3)
assert c3 == 1, ('injvec', c3)
t = t.replace(old3, '')

io.open(FP, 'w', encoding='utf-8').write(t)
chk = io.open(FP, encoding='utf-8').read()
assert "bit_anchors_B" not in chk
assert "_inj_vec" not in chk
assert "chg_bc = clip_res['bc_l%02d' % _m38]['chg']" in chk
assert "chg_dc = clip_res['dc_l%02d' % _m38]['chg']" in chk
assert "chg_jc = clip_res['jc_l%02d' % _m38]['chg']" in chk
print('PATCH1 OK c1=%d c2=%d c3=%d' % (c1, c2, c3))
