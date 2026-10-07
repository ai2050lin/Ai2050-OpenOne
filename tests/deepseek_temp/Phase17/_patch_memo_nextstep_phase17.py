# -*- coding: utf-8 -*-
"""一次性补丁：给 gen_memo_phase17.py 增加 §11「下一步（死线）」（项目要求每 Phase MEMO 记录接续）。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase17\gen_memo_phase17.py'
s = io.open(P, encoding='utf-8').read()

ANCHOR = "# ============================================================ 附\n"
assert s.count(ANCHOR) == 1, 'anchor count=%d' % s.count(ANCHOR)

BLOCK = (
    "# ============================================================ 11 下一步\n"
    "A('### 11. 下一步（死线）')\n"
    "A('')\n"
    "A('**Phase 18 最高优先 = 逐层组件「行为」预算** —— 补上 H11 的因果缺口：本 Phase 的「深端写入无效」'\n"
    "  '是**相关性陈述**（`spearman(w_ℓ,J_ℓ)` = %s / %s / %s），要变成因果陈述，必须把「向量预算」换成「行为预算」：'\n"
    "  '在 REACH 的每个位点，把 `Δ_inc,ℓ` 的**向量**分解替换为**行为**分解（逐层组件对 `Δlogit(is-a)` 的贡献），'\n"
    "  '与 §7 的 MLP 主导（share_mlp_nb = %s / %s / %s）交叉验证 —— 若行为层同样 MLP 主导，'\n"
    "  '则「质心层由 MLP 写」升级为因果；若行为层以 attn 为主，则本 Phase 的向量份额读数是**几何假象**'\n"
    "  '（方向相消的另一种表现）。'\n"
    "  % (f(V[ARMS[0]]['Q6_spearman_wJ'], 3), f(V[ARMS[1]]['Q6_spearman_wJ'], 3), f(V[ARMS[2]]['Q6_spearman_wJ'], 3),\n"
    "     f(V[ARMS[0]]['Q5_share_mlp_nb'], 3), f(V[ARMS[1]]['Q5_share_mlp_nb'], 3), f(V[ARMS[2]]['Q5_share_mlp_nb'], 3)))\n"
    "A('')\n"
    "A('**并列**：①NF4 vs BF16 的 `w_ℓ` 口径差异（本 Phase 全部读数在 nf4 口径；H5/H7 的容差来自 nf4 噪声地板，'\n"
    "  '需在 A0 同尺度 bf16 下复算 `w_ℓ` 谱，确认峰值位置与 `com_V` 不随量化口径漂移）；'\n"
    "  '②邻域宽度 ±2 的敏感性（三臂邻域**恰好都是 %s**，需检验 ±1 / ±3 是否改变 MLP 主导结论）。'\n"
    "  % jd({sh(a): RES['arms'][a]['E5_com_V']['neighbourhood'] for a in ARMS}))\n"
    "A('')\n"
    "A('**仍挂账**：N2h1-α-1 权重级定位；N2h1-β 水果类崩塌解剖；N3-β→N3-ε；R1 对照补强；K4 处置；'\n"
    "  '**N 线 Phase 3–7 补登 Ledger**（Phase 8–17 已各 1 条）。')\n"
    "A('')\n"
    "\n"
)

s2 = s.replace(ANCHOR, BLOCK + ANCHOR)
assert s2 != s
io.open(P, 'w', encoding='utf-8', newline='\n').write(s2)

t = io.open(P, encoding='utf-8').read()
assert '### 11. 下一步（死线）' in t and 'Phase 18 最高优先' in t
print('PATCH OK; Phase 18 occurrences =', t.count('Phase 18'))
