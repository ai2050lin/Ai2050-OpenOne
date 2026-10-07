# -*- coding: utf-8 -*-
"""Phase18 探针勘误 P-B：删去与 INC_MLP/INC_ATTN 重复的线性残差前向（改为离线由 rows 计算）。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase18\probe_feasibility_phase18.py'
s = io.open(P, encoding='utf-8').read()

old = """            # 线性残差（只对 INC_ALL 有定义）
            if cname == 'INC_ALL':
                pvm = proj(d_mlp.astype(np.float32), Ub) * float(alpha)
                pva = proj(d_attn.astype(np.float32), Ub) * float(alpha)
                lg_m = fwd_patch(TMPL % rw, s, h0 + pvm); n_fw += 1
                lg_a = fwd_patch(TMPL % rw, s, h0 + pva); n_fw += 1
                xm = score_of(lg_m, ds, ids_of(dw)[0]) - BASE[rw]['sd0']
                xa = score_of(lg_a, ds, ids_of(dw)[0]) - BASE[rw]['sd0']
                lin_res.append(abs(x1 - (xm + xa)) / max(abs(x1), 1e-9))
"""
new = """            # 线性残差改为**离线**从 INC_MLP / INC_ATTN 两条读数计算，避免重复前向
            # （勘误 P-B：原实现每条 ALL 前向后再补 2 条 MLP/ATTN 前向，纯属重复）。
"""
assert s.count(old) == 1, 'anchor count=%d' % s.count(old)
s = s.replace(old, new)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)

t = io.open(P, encoding='utf-8').read()
assert 'lin_res.append' not in t, 'old lin_res still present'
assert '勘误 P-B' in t
assert 'lin_res_mean' in t or 'lin_res' in t
print('PATCH P-B OK (redundant lin forwards removed)')
