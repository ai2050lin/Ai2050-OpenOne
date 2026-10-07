# -*- coding: utf-8 -*-
"""Phase18 主脚本勘误 M-A：
  (1) ALL_SITES 与 P17 的 w_all 索引对齐（0..L-2，不是 1..L-1）；
  (2) bridge_site 改为按臂解析（= 该臂 L*_own）；
  (3) E7 增加 r_lin 峰值诊断、share 的 (mlp+attn) 分母版、L*_own、REACH 向量份额。
"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase18\n2h1a11_behavioral_component_budget.py'
s = io.open(P, encoding='utf-8').read()


def rep(old, new, tag):
    global s
    assert s.count(old) == 1, '%s count=%d' % (tag, s.count(old))
    s = s.replace(old, new)


# (1) 索引对齐：P17 的 w_all 覆盖 layer 0..L-2
rep("    ALL_SITES = list(range(1, L))          # 全域位点：1..L-1（hidden_states 的 level 索引）\n",
    "    # 位点 = 层号 l（注入 hooks layers[l] 的输出 = HH[l+1]）；与 P17 的 w_all（layer 0..L-2）逐索引对齐。\n"
    "    ALL_SITES = list(range(0, L - 1))\n",
    'allsites')

# (2) bridge_site 按臂解析
rep("BRIDGE_SITE = int(EX['bridge_site'])\n", "", 'rm-global-bridge')
rep("    nb = [s for s in RE_L if abs(s - EX['p17_anchors'][arm_id]['com_V']) <= NBW]\n",
    "    nb = [s for s in RE_L if abs(s - EX['p17_anchors'][arm_id]['com_V']) <= NBW]\n"
    "    BRIDGE_SITE = int(EX['p16_anchors'][arm_id]['L_star_own'])   # 该臂自己的写入窗\n",
    'bridge-per-arm')

# (3) E7 增补
rep("""    rlin_nb = float(np.mean([rlin[l] for l in nb]))
    rlin_reach = float(np.mean([rlin[l] for l in RE_L]))
    rlin_at_L6 = rlin.get(6)
""",
    """    rlin_nb = float(np.mean([rlin[l] for l in nb]))
    rlin_reach = float(np.mean([rlin[l] for l in RE_L]))
    _rl = np.array([rlin[l] for l in ALL_SITES], float)
    _order = np.argsort(-_rl)
    rlin_argmax = int(ALL_SITES[int(_order[0])])
    rlin_peak = float(_rl[_order[0]])
    rlin_second = float(_rl[_order[1]]) if len(_order) > 1 else None
    rlin_peak_ratio = (rlin_peak / rlin_second) if (rlin_second and rlin_second > 1e-12) else None
    rlin_at_lstar = rlin.get(int(EX['p16_anchors'][arm_id]['L_star_own']))
    # 份额的第二种分母（(mlp+attn) 为分母，与置换零假设同口径）
    _nm = float(sum(abs(B['INC_MLP'][l]) for l in nb)); _na = float(sum(abs(B['INC_ATTN'][l]) for l in nb))
    share_ratio_mlp_attn_nb = (_nm / (_nm + _na)) if (_nm + _na) > 1e-12 else None
    _wv_re = float(sum(W['w_mlp'][l] for l in RE_L)); _wv_ra = float(sum(W['w_all'][l] for l in RE_L))
    share_mlp_vec_reach = (_wv_re / _wv_ra) if _wv_ra > 1e-12 else None
""",
    'rlin-diag')

rep("""        rlin_nb=rlin_nb, rlin_reach=rlin_reach, rlin_at_bridge=rlin.get(BRIDGE_SITE),
""",
    """        rlin_nb=rlin_nb, rlin_reach=rlin_reach, rlin_at_bridge=rlin.get(BRIDGE_SITE),
        rlin_argmax=rlin_argmax, rlin_peak=rlin_peak, rlin_peak_ratio=rlin_peak_ratio,
        rlin_at_lstar=rlin_at_lstar, L_star_own=int(EX['p16_anchors'][arm_id]['L_star_own']),
        share_ratio_mlp_attn_nb=share_ratio_mlp_attn_nb, share_mlp_vec_reach=share_mlp_vec_reach,
        bridge_site=BRIDGE_SITE,
""",
    'e7-extra')

# (4) Q8 用峰值/写入窗对照
rep("""    v['Q8_rlin_bridge'] = S['rlin_at_bridge']; v['Q8_rlin_nb'] = S['rlin_nb']
    v['Q8_rlin_reach'] = S['rlin_reach']
    v['Q8_label'] = ('SUPERADD_AT_WINDOW' if (S['rlin_at_bridge'] is not None
                                              and S['rlin_at_bridge'] > FL['RLIN_WINDOW_MIN']
                                              and S['rlin_nb'] < FL['RLIN_DEEP_MAX'])
                     else 'NO_WINDOW_CONTRAST')
""",
    """    v['Q8_rlin_bridge'] = S['rlin_at_bridge']; v['Q8_rlin_nb'] = S['rlin_nb']
    v['Q8_rlin_reach'] = S['rlin_reach']
    v['Q8_rlin_argmax'] = S['rlin_argmax']; v['Q8_L_star_own'] = S['L_star_own']
    v['Q8_rlin_peak_ratio'] = S['rlin_peak_ratio']
    v['Q8_label'] = ('SUPERADD_AT_WINDOW'
                     if (S['rlin_argmax'] == S['L_star_own']
                         and S['rlin_peak_ratio'] is not None
                         and S['rlin_peak_ratio'] >= FL['RLIN_PEAK_RATIO_MIN'])
                     else 'NO_WINDOW_CONTRAST')
""",
    'q8')

rep("    P['P6'] = dict(name='超可加性集中在写入窗',\n"
    "                   pass_=bool(JV['Q8_counts']['SUPERADD_AT_WINDOW'] >= 2),\n"
    "                   detail=dict(Q8_counts=JV['Q8_counts'],\n"
    "                               rlin_window={a: V[a]['Q8_rlin_bridge'] for a in arms},\n"
    "                               rlin_nb={a: V[a]['Q8_rlin_nb'] for a in arms},\n"
    "                               rlin_reach={a: V[a]['Q8_rlin_reach'] for a in arms}))\n",
    "    P['P6'] = dict(name='超可加性峰值落在写入窗',\n"
    "                   pass_=bool(JV['Q8_counts']['SUPERADD_AT_WINDOW'] >= 2),\n"
    "                   detail=dict(Q8_counts=JV['Q8_counts'],\n"
    "                               rlin_argmax={a: V[a]['Q8_rlin_argmax'] for a in arms},\n"
    "                               L_star_own={a: V[a]['Q8_L_star_own'] for a in arms},\n"
    "                               rlin_peak_ratio={a: V[a]['Q8_rlin_peak_ratio'] for a in arms},\n"
    "                               rlin_nb={a: V[a]['Q8_rlin_nb'] for a in arms},\n"
    "                               rlin_reach={a: V[a]['Q8_rlin_reach'] for a in arms},\n"
    "                               share_ratio_mlp_attn_nb={a: None for a in arms}))\n",
    'p6')

# (5) 补 share_ratio 进 verdict（供报告）
rep("    v['Q4_rlin_nb'] = S['rlin_nb']\n",
    "    v['Q4_rlin_nb'] = S['rlin_nb']\n"
    "    v['Q4_share_ratio_mlp_attn_nb'] = S['share_ratio_mlp_attn_nb']\n"
    "    v['Q4_share_mlp_vec_reach'] = S['share_mlp_vec_reach']\n",
    'q4-extra')

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
t = io.open(P, encoding='utf-8').read()
for tok in ('list(range(0, L - 1))', 'BRIDGE_SITE = int(EX[\'p16_anchors\'][arm_id][\'L_star_own\'])',
            'rlin_peak_ratio', 'share_ratio_mlp_attn_nb', 'RLIN_PEAK_RATIO_MIN'):
    assert tok in t, 'missing ' + tok
assert "list(range(1, L))" not in t
print('PATCH M-A OK')
