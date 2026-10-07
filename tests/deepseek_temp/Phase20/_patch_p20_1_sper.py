# -*- coding: utf-8 -*-
"""补丁 P20-1：修正 sper() 的键空间混用（W17 用 'w_all' 名，WV 用 'INC_ALL' 名）。"""
import io
P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase20\n2h1a13_quant_scheme_robustness.py'
s = io.open(P, encoding='utf-8').read()

old = """    def sper(wk, wsrc, bcomp, table=None):
        t = table or B
        return spearman([wsrc[wk][l] for l in RE_L], [abs(t[bcomp][l]) for l in RE_L])

    sp_wall_ball = sper('w_all', W17, 'INC_ALL')          # P18 口径（w 用 P17 冻结锚）
    sp_wall_ball_own = sper('w_all', WV, 'INC_ALL')        # 同口径（w 用本臂自己重算）
    sp_wmlp_bmlp_own = sper('w_mlp', WV, 'INC_MLP')"""
new = """    def sper_anchor(bcomp):
        \"\"\"P18 口径：w 用 P17 冻结锚谱（键名 w_all/w_mlp/w_attn）。\"\"\"
        return spearman([W17['w_all'][l] for l in RE_L], [abs(B[bcomp][l]) for l in RE_L])

    def sper_own(cw, cb, table=None):
        \"\"\"同口径：w 用**本臂自己重算**的谱（键名 = 组件名 INC_*）。\"\"\"
        t = table or B
        return spearman([WV[cw][l] for l in RE_L], [abs(t[cb][l]) for l in RE_L])

    sp_wall_ball = sper_anchor('INC_ALL')                  # P18 口径（w 用 P17 冻结锚）
    sp_wall_ball_own = sper_own('INC_ALL', 'INC_ALL')       # 同口径（w 用本臂自己重算）
    sp_wmlp_bmlp_own = sper_own('INC_MLP', 'INC_MLP')"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
s2 = io.open(P, encoding='utf-8').read()
assert 'sper_own' in s2 and s2.count('sper_anchor') == 2 and 'wsrc' not in s2
print('PATCH P20-1 OK')
