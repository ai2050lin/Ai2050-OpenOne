# -*- coding: utf-8 -*-
"""
Phase 20 (N2h1-alpha-13) seal 生成器（**设计冻结**，不含任何本轮观测值）。

产物：tests/deepseek_temp/Phase20/N2h1a13_design_seal.json
原则：seal 在任何观测前写出 —— 因此**不嵌入**本轮 probe 的读数，只**声明** probe 的落点文件名；
      真正的 probed 读数由 probe 自己写盘，其 sha256 由收尾链记录（见 exec / MEMO）。
材料来源：Phase 16 execution（模板/类/实例/配对/PROFILE/ALPHAS/XHF）+ Phase 18 execution（组件/种子/门）。
冻结锚：Phase 18 result（行为质心族）+ Phase 16 result（com_layer 族）+ Phase 17 result（com_V）。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


E16 = json.load(io.open(os.path.join(P16T, 'execution_phase16.json'), encoding='utf-8'))
E18 = json.load(io.open(os.path.join(P18T, 'execution_phase18.json'), encoding='utf-8'))
R18 = json.load(io.open(os.path.join(P18T, 'result_phase18.json'), encoding='utf-8'))
R16 = json.load(io.open(os.path.join(P16T, 'result_phase16.json'), encoding='utf-8'))
R17 = json.load(io.open(os.path.join(P17T, 'result_phase17.json'), encoding='utf-8'))

AK = {'A0_nf4': 'A0_calib_qwen3-4b-nf4', 'A1_nf4': 'A1_glm4-9b-nf4'}

# ---------------- 臂（唯一自变量 = 量化口径）
ARMS = {}
for nm, akey in AK.items():
    cfg = E18['arms'][akey]
    ARMS[nm] = dict(model=cfg['model'], dir=cfg['dir'], quant='nf4', offload=False,
                    role=('校准臂：nf4 口径，必须逐位复现 P18/P16 冻结锚' if nm == 'A0_nf4'
                          else '校准臂：跨家族 nf4，必须逐位复现 P18/P16 冻结锚'),
                    expected=cfg['expected'], config_sha256=cfg['config_sha256'],
                    anchor_key=akey,
                    reach=[int(x) for x in R16['E7_reach'][akey]['reach']],
                    nb=[int(x) for x in R18['arms'][akey]['E7_summary']['nb']],
                    L_star_own=int(R16['E3_localize'][akey]['L_star_own']))
    ARMS[nm.replace('_nf4', '_bf16')] = dict(model=cfg['model'], dir=cfg['dir'], quant='bf16',
                                             offload=bool(nm == 'A1_nf4'),
                                             role=('核心检验臂：同模型同尺度，唯一改动 = bf16'
                                                   if nm == 'A0_nf4' else
                                                   '跨家族 holdout 检验臂：bf16（需 CPU offload）'),
                                             expected=cfg['expected'], config_sha256=cfg['config_sha256'],
                                             anchor_key=akey,
                                             reach=[int(x) for x in R16['E7_reach'][akey]['reach']],
                                             nb=[int(x) for x in R18['arms'][akey]['E7_summary']['nb']],
                                             L_star_own=int(R16['E3_localize'][akey]['L_star_own']))
ARM_ORDER = ['A0_nf4', 'A0_bf16', 'A1_nf4', 'A1_bf16']

# ---------------- 冻结锚（供校准臂逐位断言）
ANCHORS = {}
for nm, akey in AK.items():
    S18 = R18['arms'][akey]['E7_summary']
    ANCHORS[akey] = dict(
        com_B={k: float(v) for k, v in S18['com_B'].items()},
        com_B_full={k: float(v) for k, v in S18['com_B_full'].items()},
        comlayer_B_all=float(S18['comlayer_B_all']),
        comlayer_B_mlp=float(S18['comlayer_B_mlp']),
        comlayer_B_attn=float(S18['comlayer_B_attn']),
        share_mlp_beh_nb=float(S18['share_mlp_beh_nb']),
        share_top1_beh_nb=float(S18['share_top1_beh_nb']),
        com_V=float(R17['arms'][akey]['E5_com_V']['com_V']),
        cum_bridge=float(S18['cum_bridge']),
        full_swap=float(R16['arms'][akey]['E2_full_swap']['FULL_SWAP']),
        com_layer_x=float(R16['verdict'][akey]['Q4_com_x']),
        com_layer_j=float(R16['verdict'][akey]['Q4_com_j']),
        nb=[int(x) for x in S18['nb']],
        reach=[int(x) for x in S18['reach']],
        L_star_own=int(R16['E3_localize'][akey]['L_star_own']),
        spearman_wall_ball=float(S18['spearman_wall_ball']),
        gap=float(S18['com_V_p17'] - S18['com_B']['INC_ALL']),
    )

FLOORS = dict(
    P20_FID_ARCH=3.0e-2, P20_FID_BLK=1.0e-2,
    QUANT_TOL_COMV=2.0, QUANT_TOL_COMB=2.0, QUANT_TOL_COMLAYER=2.0,
    QUANT_TOL_SHARE=0.10, QUANT_TOL_XHALF=0.05, RHO_B_MIN=0.80,
    MLP_DOM_MIN=0.50, SHALLOWER_MIN=2.0, UNREACH_y=0.10, NULL_ALPHA=0.05,
    CALIB_TOL_COMV=1.0e-4, CALIB_TOL_COMB=1.0e-4, CALIB_TOL_COMLAYER=1.0e-4,
)

SEAL = dict(
    phase=20, line='N2h1-alpha-13', kind='design_seal',
    created_local=time.strftime('%Y-%m-%d %H:%M:%S'),
    title='行为量与写入窗剖面的跨精度稳健性（nf4 vs bf16）—— 把 P19 的向量侧检验补齐到 P18/P16 的行为侧',

    motivation=dict(
        gap_1='P19 只把跨精度检验做到**向量侧**：`w_ℓ` 谱与 `com_V` 在 nf4→bf16 下只移动 0.07–0.09 层（容差 2.0），秩相关 ≥0.992。',
        gap_2='但 P18 的结论（`com_B` 比 `com_V` 浅 2.5–6.6 层、行为 MLP 主导 0.663/0.960/0.747、`spearman(w_all,b_all)`=+0.90/+0.69/+0.87）与 P16 的结论（`com_layer(x)`/`com_layer(J)`）**全部在 nf4 口径下**，从未跨精度复算；P18 §8 H3 也只承认「激活级」。',
        gap_3='P17 `quant.why` 说明三臂统一取 nf4 是**可行性妥协**（bf16 下 Qwen3-14B 29.5GB 触发 RAM 硬杀）。A0/A1 在 bf16 下**可载**（P19 `loadability` 实测：A0-bf16 全 GPU、A1-bf16 需 offload；A2-bf16 segfault）。',
        fix='在**同一装置、同一材料、同一域**下，把 P18 的行为预算 `b_{c,ℓ}` 与 P16 的写入窗 α 剖面（`xhalf`/`J`/`com_layer`）在 nf4 与 bf16 两口径各复算一遍，唯一自变量 = 数值精度。',
        falsifiers='若 bf16 下 `com_B` 移动 >2.0 层、或行为谱秩相关 <0.80、或 MLP 行为主导翻转、或 `spearman(w,b)` 变负、或 `gap` 消失 ⇒ P18 的行为结论是**量化地板效应**，须整体降级。反之则 P18/P16 获得跨数值口径支撑。',
    ),

    material_source=dict(
        p16_execution=dict(path='tests\\deepseek_temp\\Phase16\\execution_phase16.json',
                           sha256=fsha(os.path.join(P16T, 'execution_phase16.json'))),
        p18_execution=dict(path='tests\\deepseek_temp\\Phase18\\execution_phase18.json',
                           sha256=fsha(os.path.join(P18T, 'execution_phase18.json'))),
        p18_result=dict(path='tests\\deepseek_temp\\Phase18\\result_phase18.json',
                        sha256=fsha(os.path.join(P18T, 'result_phase18.json'))),
        p16_result=dict(path='tests\\deepseek_temp\\Phase16\\result_phase16.json',
                        sha256=fsha(os.path.join(P16T, 'result_phase16.json'))),
        p17_result=dict(path='tests\\deepseek_temp\\Phase17\\result_phase17.json',
                        sha256=fsha(os.path.join(P17T, 'result_phase17.json'))),
    ),

    invariants=dict(
        what_changes='数值精度（bitsandbytes nf4 4bit vs torch.bfloat16 原生）—— 四臂中唯一自变量。',
        what_is_frozen=('模板 `%s是一种`；6 类词；41 实例；24 discovery 配对；17 confirmation 配对；'
                        'U_ℓ = 全 41 实例按类平均 → 类别质心差 SVD（秩 = n_classes−1 = 5）；'
                        '位点 = 1..L−2（与 P17 w_all 索引对齐）；REACH 域取 P16 冻结值（同域配对）；'
                        'nb 取 P17 冻结值；L*_own 取 P16 冻结值；PROFILE/ALPHAS/XHF 取 P16 冻结值；'
                        '质心一律**区间求和 + 相邻位点中点**；种子取 P18/P16 冻结值。'),
    ),

    panels=dict(
        B=dict(name='行为预算（P18 口径）',
               injection='INC_ALL: h_ℓ^R + P_{U_ℓ}(Δ_attn,ℓ + Δ_mlp,ℓ) ；INC_MLP / INC_ATTN / INC_TOP1（逐层最大 ‖P(Δ_head)‖ 的单头）/ CUM_ALL: h_ℓ^R + P_{U_ℓ}(d_ℓ), d_ℓ = HH[ℓ+1]^D − HH[ℓ+1]^R',
               readout='b_{c,ℓ} = mean_pairs [ score_of(logits_patched, ds, sid_d) − BASE[rw].sd0 ]',
               components='discovery: INC_ALL/INC_MLP/INC_ATTN/INC_TOP1/CUM_ALL × 1..L−2 ；confirmation: INC_ALL/INC_MLP/INC_ATTN × 1..L−2',
               derived='com_B(区间求和质心) / com_layer_B_* (对 b 取相邻差后 com_layer) / share_mlp_beh(nb) / spearman(w,b) / r_lin / 自谱 w_ℓ = mean_pairs‖P_{U_ℓ}(dv_c)‖'),
        P=dict(name='写入窗 α 剖面（P16 口径）',
               injection='dv = HH[ℓ+1]^D − HH[ℓ+1]^R（**原始差**，不投影）；注入 h_ℓ^R + α·dv，α ∈ ALPHAS',
               readout='Y(ℓ,α) = mean_disc dDonor / FULL_SWAP（FULL_SWAP 用**本臂**口径）',
               derived='xhalf(ℓ) = cross_alpha(ALPHAS, Y[ℓ], 0.5) / J(ℓ) = J_only(ALPHAS, Y[ℓ]) / com_layer(x), com_layer(J), span3 / ρ(ℓ)=Y(ℓ,1) 与 REACH 重算（诊断）'),
    ),

    measures=dict(
        E0='装置自检：determinism（≤1e-6）/ hook 效应（>1e-6）/ Q0 device / 维度 / o_proj',
        E1='capture：HH / o_proj 输入 / MLP 输出（41 实例）',
        E2='保真度：架构恒等式（HH[ℓ+1]−HH[ℓ] ≈ OPR[ℓ](O[ℓ]) + M[ℓ]）max rel ≤ P20_FID_ARCH；分块可加性 max rel ≤ P20_FID_BLK',
        E3='U_ℓ：全 41 实例按类平均 → 类别质心差 SVD，取秩 n_classes−1',
        E4='BASE / FULL_SWAP（同 P16/P18 口径）',
        E5='Panel B discovery 扫描（5 组件 × ALL_SITES × 24 对）',
        E6='Panel B confirmation 扫描（3 组件 × ALL_SITES × 17 对）',
        E7='Panel P α 剖面（PROFILE × ALPHAS × 24 对）→ xhalf / J / com_layer / ρ / REACH 重算',
        E8='置换零假设（com_B(all)/com_B(mlp)/邻域份额/com_layer(x)）+ 确认集 + 离流形诊断',
        E9='锚复现（**仅 nf4 校准臂**）：P18 行为族 + P16 com_layer 族 + P17 com_V 逐位断言',
        E10='汇总：com_B 族 / com_layer_B 族 / 份额 / 秩相关 / r_lin / 桥接 / Panel P 族 / 谱',
    ),

    floors=FLOORS,

    predictions=[
        dict(id='P1', name='装置与保真度（四臂）',
             criterion='四臂 Q0_apparatus 且 Q1=FID_ALL_PASS（arch ≤ 0.03 / blk ≤ 0.01）',
             falsified_by='任一臂 arch>3e-2 或 blk>1e-2'),
        dict(id='P2', name='nf4 校准臂逐位复现 P18/P16 冻结锚',
             criterion='A0_nf4/A1_nf4 的 com_B 族、comlayer_B 族、share_mlp_beh_nb、com_V、cum_bridge、full_swap、com_layer_x/j、nb、reach_len 全部 |Δ| ≤ 1e-4',
             falsified_by='任一锚不一致 ⇒ 装置漂移，本 Phase 跨精度结论全部降级'),
        dict(id='P3', name='holdout 主预测 1 —— 行为质心位置跨精度稳健',
             criterion='两对（A0/A1）|Δcom_B(INC_ALL)| ≤ QUANT_TOL_COMB=2.0 层',
             falsified_by='任一对 >2.0 层'),
        dict(id='P4', name='holdout 主预测 2 —— 行为谱形状跨精度一致',
             criterion='REACH 域 spearman(b_nf4, b_bf16) ≥ RHO_B_MIN=0.80（两对）',
             falsified_by='任一对 <0.80'),
        dict(id='P5', name='holdout 主预测 3 —— 行为 MLP 主导在 bf16 下保持',
             criterion='两对份额同侧且四臂 share_mlp_beh_nb > MLP_DOM_MIN=0.50，|Δ| ≤ 0.10',
             falsified_by='某臂 ≤0.50 或某对翻转或 |Δ|>0.10'),
        dict(id='P6', name='holdout 主预测 4 —— 行为质心仍浅于向量质心',
             criterion='四臂 gap = com_V − com_B(INC_ALL) ≥ SHALLOWER_MIN=2.0 层，且两对同侧',
             falsified_by='某臂 gap<2.0 或某对翻侧'),
        dict(id='P7', name='holdout 主预测 5 —— 同对象耦合两口径皆成立',
             criterion='四臂 spearman(w_all(P17 冻结), b_all) > 0',
             falsified_by='任一口径 ≤0'),
        dict(id='P8', name='Panel P —— com_layer(x)/com_layer(J) 跨精度稳健',
             criterion='两对 |Δcom_layer(x)| ≤ 2.0 且 |Δcom_layer(J)| ≤ 2.0 层',
             falsified_by='任一对任一量 >2.0'),
        dict(id='P9', name='Panel P —— 半饱和点跨精度稳健',
             criterion='两对 max|Δxhalf| ≤ QUANT_TOL_XHALF=0.05（沿用 P16 的 XH_FAITHFUL_TOL）',
             falsified_by='任一对 >0.05'),
        dict(id='P10', name='对照（描述性，不设方向性预测）',
             criterion='置换零假设/确认集/离流形诊断只报告与判定',
             falsified_by='不适用'),
    ],

    verdict_tree=dict(
        Q0='装置：apparatus（T=2 布局 ∧ 维度 ∧ o_proj ∧ determinism ≤1e-6 ∧ hook 有效）与 device 类别',
        Q1='保真度：FID_PASS / FID_PARTIAL / FID_FAIL',
        Q2='锚：ANCHOR_ALL_OK（仅 nf4 校准臂）/ ANCHOR_DRIFT',
        Q3='com_V 跨精度：|Δ| ≤ 2.0（两对）',
        Q4='com_B(all) 跨精度：|Δ| ≤ 2.0（两对）—— 主预测 P3',
        Q5='comlayer_B_all 跨精度：|Δ| ≤ 2.0（两对）',
        Q6='行为谱一致：ρ(b_all) ≥ 0.80（两对）—— 主预测 P4',
        Q7='份额：同侧 ∧ |Δ| ≤ 0.10（两对）',
        Q8='MLP 行为主导保留：四臂 share > 0.50',
        Q9='浅化关系保留：四臂 gap ≥ 2.0 且两对同侧',
        Q10='同对象耦合保留：四臂 spearman(w_all,b_all) > 0',
        Q11='Panel P com_layer 跨精度：|Δcom_layer(x)| ≤2.0 ∧ |Δcom_layer(J)| ≤2.0（两对）—— 主预测 P8',
        Q12='Panel P xhalf 跨精度：max|Δxhalf| ≤ 0.05（两对）—— 主预测 P9',
    ),

    honesty=[
        'H1：可行性 probe 只在 **A0** 上运行（nf4 与 bf16 两口径、缩幅网格）；A1 的任何量在 seal 冻结前**未被观测**。',
        'H2：A1 的 bf16 臂需 CPU offload（18.8GB > 14GiB 上限）⇒ 含「分片执行」第二源；若其读数超容差，须先在无 offload 的 A0 上排除分片效应（承 P19 H6）。',
        'H3：A2（Qwen3-14B）**不参与**：nf4 臂不入本 Phase（跨精度无配对），bf16 臂已由 P19 实测 segfault ⇒ 本 Phase 的跨精度稳健性只在 **qwen3-4b 与 glm4-9b** 上验证。',
        'H4：b 与 Panel P 的读数都是**激活级**干预（hook 注入），不是权重级实现证明（承 N2h1-α-1 挂账）。',
        'H5：`U_ℓ` 由**全部 41 实例**估计 ⇒ 存在轻微选择性泄漏；任何「泛化」只指**跨臂/跨口径**。',
        'H6：保真度门容差来自 nf4 量化噪声的**实测地板**，不是理论误差界；bf16 预期更小（P19：arch 1.258e-2 vs 1.617e-2）。',
        'H7：bf16 与 nf4 的差异含「量化误差 + 线性层 kernel 路径」两源（同 `eager` 已控注意力 kernel）。',
        'H8：`com_V_recomputed` 由本 Phase 自己的 `w_ℓ = mean_pairs‖P_U(dv)‖` 谱算出，**应与 P17/P19 记录一致**（这是对 P19 结论的独立复现，二者共用同一 P17 口径）。',
        'H9：Panel P 的 `com_layer` 定义在**冻结 REACH**上（同域配对）；bf16 自算的 REACH 只作诊断报告。',
        'H10：`r_lin` 是**比值型**诊断量（承 P18 E-rlin），浅端近零分母会放大它 ⇒ 只作对照，不入主判据。',
        'H11：nb 只有 2 个位点 ⇒ 邻域份额的置换零假设区分力弱（承 P18 H7）⇒ 零假设只作对照。',
        'H12：若 offload 臂（A1_bf16）读数与 A0_bf16 相反，须以 A0 为准并记录分片疑点，不强行合并结论。',
    ],
    why_not_a_HARKing_violation=(
        '探针只在 A0 上运行，且**首跑在 seal 写出之前**；seal 的字节在任何观测值写入之前固定，'
        'probe 读数以**独立文件**存在（seal 只声明其路径与用途，不嵌入数字）。'
        '主判据的关键分支落在 **A1**（跨家族 holdout），其 bf16 量在 seal 冻结前未被观测。'
        '所有阈值都由机制无关的理由给出（2.0 层 = P19 沿用；0.80 = 谱形状一致性的朴素下限；'
        '0.50 = 「过半」的朴素定义；0.05 = P16 已发表的跨精度 xhalf 容差）。'
        'P10 明确不设方向性预测。'),
    may_falsify_the_whole_line=(
        '若 bf16 下 `com_B` 位移 >2.0 层或行为谱秩相关 <0.80，则 P18 的「行为质心比向量质心浅」'
        '只是 nf4 量化地板效应 —— 这会同时削弱 P8–P18 的全部行为侧结论（向量侧已由 P19 支撑）。'),

    arms=ARMS, arm_order=ARM_ORDER, anchors=ANCHORS,
    template=E16['template'], classes=E16['classes'], instances_all=E16['instances_all'],
    pairs_all=E16['pairs_all'], discovery=E16['discovery'], confirmation=E16['confirmation'],
    quant_nf4=E18['quant'], quant_bf16=dict(
        scheme='torch.bfloat16 (native)',
        attn_implementation='eager', device_map='auto',
        max_memory=E18['quant']['max_memory'],
        why='除量化口径外逐项与 nf4 臂一致（同 attn_implementation/device_map/max_memory/low_cpu_mem_usage）。'),
    components=E18['components'], components_confirmation=E18['components_confirmation'],
    profile_sites=E16['profile_sites'], profile_sites_legacy=E16['profile_sites_legacy'],
    alphas=E16['alphas'], xh_frac=E16['xh_frac'],
    bootstrap=dict(BP=E18['bootstrap']['BP'], seed=E18['bootstrap']['seed'],
                   seeds=dict(comB_inc=E18['bootstrap']['seeds']['comB_inc'],
                              comB_mlp=E18['bootstrap']['seeds']['comB_mlp'],
                              share_mlp=E18['bootstrap']['seeds']['share_mlp'],
                              comlayer=20261210),
                   note='前三个种子逐字继承 P18；`comlayer` 为 Panel P 新量（com_layer(x) 的置换零假设）专用。'),
    neighbourhood_width=E18['neighbourhood_width'],

    probe_files=dict(
        note='可行性 probe 的落点（seal 只声明，不嵌入数字；其 sha256 由收尾链在 exec/MEMO 中记录）',
        A0_nf4='tests\\deepseek_temp\\Phase20\\_probe20_A0_nf4.json',
        A0_bf16='tests\\deepseek_temp\\Phase20\\_probe20_A0_bf16.json',
        mode='PROBE=1 python tests/deepseek/Phase20/n2h1a13_quant_scheme_robustness.py（缩幅网格：PROFILE 20 位点 × 7 α；仅 A0）',
    ),

    next_step=('Phase 21 候选（最高）：把跨精度检验推进到**逐层组件的向量实现级**——'
               '在 bf16 下复算 P8 的向量预算 share_v 与 N2h1-α-1 的权重级定位；'
               '并列：邻域宽度 ±2 敏感性；P17 P6 的 MEMO 改判；N 线 P3–P7 补登 Ledger。'),
)

OUT = os.path.join(P20T, 'N2h1a13_design_seal.json')
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(SEAL, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('WROTE %s  %d B  sha256=%s' % (OUT, len(b), hashlib.sha256(b).hexdigest()))
print('sha8', hashlib.sha256(b).hexdigest()[:8])
print('arms', ARM_ORDER)
print('anchors keys', sorted(ANCHORS.keys()))
for k, v in ANCHORS.items():
    print(' ', k, 'com_B(all)=%.6f comlayer_B_all=%.6f share=%.6f com_V=%.6f CLx=%.6f CLj=%.6f' %
          (v['com_B']['INC_ALL'], v['comlayer_B_all'], v['share_mlp_beh_nb'], v['com_V'],
           v['com_layer_x'], v['com_layer_j']))
print('floors', json.dumps(FLOORS, ensure_ascii=False))
print('alphas', SEAL['alphas'], 'xh_frac', SEAL['xh_frac'], 'PROFILE', len(SEAL['profile_sites']))
