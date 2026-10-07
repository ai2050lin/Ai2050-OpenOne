# -*- coding: utf-8 -*-
"""Phase18 主脚本勘误 M-B：修 E7 打印的 %.3f/F3 类型冲突，与 MERGE 分支的 BRIDGE_SITE 未定义。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase18\n2h1a11_behavioral_component_budget.py'
s = io.open(P, encoding='utf-8').read()


def rep(old, new, tag):
    global s
    assert s.count(old) == 1, '%s count=%d' % (tag, s.count(old))
    s = s.replace(old, new)


rep("""    w('  E7 com_B(inc)=%s (全域 %s) ; com_V(P17)=%.3f (同域重算 %.3f)'
      % (F3(com_B['INC_ALL']), F3(com_B_full['INC_ALL']),
         float(EX['p17_anchors'][arm_id]['com_V']), F3(com_V_recomputed)))
    w('     邻域 nb=%s : share_mlp_beh=%.4f (向量 %.4f) ; share_top1_beh=%.4f ; rlin_nb=%.3f'
      % (nb, share_mlp_beh_nb, share_mlp_vec_nb, share_top1_beh_nb, rlin_nb))
""",
    """    w('  E7 com_B(inc)=%s (全域 %s) ; com_V(P17)=%s (同域重算 %s, drift %.3e)'
      % (F3(com_B['INC_ALL']), F3(com_B_full['INC_ALL']),
         F3(float(EX['p17_anchors'][arm_id]['com_V'])),
         F3(com_V_recomputed), abs(float(com_V_recomputed) - float(EX['p17_anchors'][arm_id]['com_V']))))
    w('     邻域 nb=%s : share_mlp_beh=%s (ratio %s ; 向量 %s) ; share_top1_beh=%s ; rlin_nb=%s'
      % (nb, F3(share_mlp_beh_nb, 4), F3(share_ratio_mlp_attn_nb, 4), F3(share_mlp_vec_nb, 4),
         F3(share_top1_beh_nb, 4), F3(rlin_nb, 4)))
""",
    'e7-print')

rep("""    w('     bridge: CUM@L%d=%s vs FULL_SWAP=%.3f rel=%s ; rlin@L6=%s'
      % (BRIDGE_SITE, F3(cum_bridge), full_swap, F3(bridge_rel, 4), F3(rlin_at_L6)))
""",
    """    w('     bridge: CUM@L%d=%s vs FULL_SWAP=%s rel=%s ; rlin@L*=%s ; argmax rlin=%s (比值 %s)'
      % (BRIDGE_SITE, F3(cum_bridge), F3(full_swap), F3(bridge_rel, 4), F3(rlin_at_lstar, 4),
         rlin_argmax, F3(rlin_peak_ratio, 3)))
""",
    'bridge-print')

rep("""                   floors=FL, bootstrap=EX['bootstrap'], components=COMPONENTS,
                   bridge_site=BRIDGE_SITE,
""",
    """                   floors=FL, bootstrap=EX['bootstrap'], components=COMPONENTS,
                   components_confirmation=COMPONENTS_CONF,
                   bridge_site={a: int(EX['p16_anchors'][a]['L_star_own']) for a in recs},
""",
    'merge-bridge')

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
t = io.open(P, encoding='utf-8').read()
assert 'bridge_site={a: int(EX[\'p16_anchors\'][a][\'L_star_own\']) for a in recs}' in t
assert 'bridge_site=BRIDGE_SITE,' not in t
print('PATCH M-B OK')
