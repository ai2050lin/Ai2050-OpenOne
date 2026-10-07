# -*- coding: utf-8 -*-
"""
Phase 18 (N2h1-alpha-11) 预注册 seal 生成器。
产物：tests/deepseek_temp/Phase18/N2h1a11_design_seal.json
口径：材料逐字节继承 Phase 16 exec（词表/实例/配对/模板/量化），锚从 P16 result 与 P17 result 现场读入。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
os.makedirs(P18T, exist_ok=True)


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


EX16P = os.path.join(P16T, 'execution_phase16.json')
R16P = os.path.join(P16T, 'result_phase16.json')
S16P = os.path.join(P16T, 'N2h1a9_design_seal.json')
A16P = os.path.join(P16T, 'N2h1a9_design_seal_amend1.json')
EX17P = os.path.join(P17T, 'execution_phase17.json')
R17P = os.path.join(P17T, 'result_phase17.json')
S17P = os.path.join(P17T, 'N2h1a10_design_seal.json')

EX16 = json.load(io.open(EX16P, encoding='utf-8'))
R16 = json.load(io.open(R16P, encoding='utf-8'))
EX17 = json.load(io.open(EX17P, encoding='utf-8'))
R17 = json.load(io.open(R17P, encoding='utf-8'))
ARMS = EX16['arms']
ARM_IDS = list(ARMS.keys())

ANCH16 = {}
ANCH17 = {}
for a in ARM_IDS:
    ANCH16[a] = dict(
        com_layer_x=float(R16['E5_concentration'][a]['new_stat']['x']['obs_com']),
        com_layer_j=float(R16['E5_concentration'][a]['new_stat']['j']['obs_com']),
        span3_x=float(R16['E5_concentration'][a]['new_stat']['x']['obs_span']),
        span3_j=float(R16['E5_concentration'][a]['new_stat']['j']['obs_span']),
        L_star_own=int(R16['E3_localize'][a]['L_star_own']),
        ell_reach=int(R16['E7_reach'][a]['ell_reach']),
        reach=[int(x) for x in R16['E7_reach'][a]['reach']],
        J_by_site=[float(v) for v in R16['E4_summary'][a]['J']],
        sites=[int(x) for x in R16['E4_summary'][a]['sites']],
    )
    ANCH17[a] = dict(
        com_V=float(R17['arms'][a]['E5_com_V']['com_V']),
        neighbourhood=[int(x) for x in R17['arms'][a]['E5_com_V']['neighbourhood']],
        share_mlp_nb=float(R17['arms'][a]['E5_com_V']['share_mlp_nb']),
        share_attn_nb=float(R17['arms'][a]['E5_com_V']['share_attn_nb']),
        median_reach=float(R17['arms'][a]['E5_com_V']['median_reach']),
    )

PA = json.load(io.open(os.path.join(P18T, '_probe_analysis_A0.json'), encoding='utf-8'))
PB = json.load(io.open(os.path.join(P18T, '_probe_feasibility_A0.json'), encoding='utf-8'))
PROBE_TXT = os.path.join(P18T, '_probe_feasibility_A0.txt')
PROBE_TXT2 = os.path.join(P18T, '_probe_analysis_A0.txt')

SEED_BASE = 20261003
SEED_NULL = {'comB_inc': SEED_BASE + 171, 'comB_mlp': SEED_BASE + 183, 'share_mlp': SEED_BASE + 195}

COMPONENTS = ['INC_ALL', 'INC_MLP', 'INC_ATTN', 'INC_TOP1', 'CUM_ALL']
COMPONENTS_CONF = ['INC_ALL', 'INC_MLP', 'INC_ATTN']

SEAL = {
    'phase': 18,
    'line': 'N2h1-alpha-11',
    'title': ('逐层组件「行为」预算：把 P17 的向量预算换成行为效应 —— 组件归属（MLP vs attn vs top-1 头）、'
              '同对象耦合、行为/向量质心差、以及写入窗的超可加性'),
    'kind': 'design_seal',
    'created_local': time.strftime('%Y-%m-%d %H:%M:%S'),
    'supersedes': None,

    'motivation': {
        'gap_1': ('Phase 17 报出邻域 [26,28] 的 **向量** MLP 份额 share_mlp_nb = 0.740 / 0.975 / 0.824，'
                  '但那是**几何量**：它只说明"供体-受体模块输出差向量在该子空间里的范数"以 MLP 为主，'
                  '不说明**行为**（is-a logit 增益）以 MLP 为主。'),
        'gap_2': ('Phase 17 §7 的 H11 明文留下缺口：'
                  '「若行为层同样 MLP 主导 => 升级为因果；若以 attn 为主 => P17 向量份额是几何假象」。'
                  'P17 的 P5 只做了向量口径，故 H11 未决。'),
        'gap_3': ('P17 的 P6（spearman(w_ell, J_ell) < 0，"深端有大量写入但对行为无效"）把'
                  '**单层增量写入** w_ell 与 **累积差注入的剂量-响应形状** J_ell 摆在一起比。'
                  '两者不是同一个对象（d_ell = sum_{l\'<=l} Delta_inc,l\'）。'
                  '要判定 P6 是真机制还是**对象错配**，必须用"同一对象"的行为量。'),
        'fix': ('① 注入物**逐字沿用 P17 的定义**（Delta_inc / Delta_mlp / Delta_attn / Delta_head_h* / d_l），'
                '但读数从"范数"改成**行为效应** b_{c,l} = mean_pairs [score_of(patched, ds, sid_d) - BASE.sd0]'
                '（与 Phase 8 的 T 臂逐字同口径）；② 组件归属在**同一个** P17 邻域 nb=[26,28] 上比；'
                '③ 定义行为质量质心 com_B 与 P17 的 com_V 同域（REACH）同口径（区间求和）对照；'
                '④ 计算**同对象**耦合 spearman(w_ell, |b_ell|) 与 **P17 口径** spearman(w_ell, J_ell) 并排；'
                '⑤ 线性残差 r_lin = |b_all - (b_mlp + b_attn)| / |b_all| 作为"层内增益"诊断；'
                '⑥ CUM_ALL ↔ P16 FULL_SWAP 桥接门，把本 Phase 与前 10 个 Phase 的装置对齐。'),
    },

    'material_source': {
        'from_execution_p16': 'tests\\deepseek_temp\\Phase16\\execution_phase16.json',
        'from_execution_p16_sha256': fsha(EX16P),
        'from_result_p16_sha256': fsha(R16P),
        'from_seal_p16_sha256': fsha(S16P),
        'from_amend1_p16_sha256': fsha(A16P),
        'from_execution_p17': 'tests\\deepseek_temp\\Phase17\\execution_phase17.json',
        'from_execution_p17_sha256': fsha(EX17P),
        'from_result_p17_sha256': fsha(R17P),
        'from_seal_p17_sha256': fsha(S17P),
        'inherited_keys': ['template', 'classes', 'instances_all', 'pairs_all', 'discovery',
                           'confirmation', 'quant', 'profile_sites'],
        'note': ('禁止在 Phase 18 中改动词表/实例/配对/量化口径/模板；改动即视为新研究线。'
                 'com_V / nb / share_mlp_nb 从 **P17 result 现场读入**并断言（不硬编码）。'),
    },

    'template': EX16['template'],
    'classes': EX16['classes'],
    'instances_all': EX16['instances_all'],
    'pairs_all': EX16['pairs_all'],
    'discovery': EX16['discovery'],
    'confirmation': EX16['confirmation'],
    'quant': EX16['quant'],
    'profile_sites': EX16['profile_sites'],
    'sup_id_semantics': ('逐臂由**该臂 tokenizer** 现场解析类别词 id，并断言 6/6 类别词恰为单 token 且 '
                         'decode 可逆（F1b）；禁止跨词表沿用任何硬编码 id（Phase 15 amend1 事故）。'),

    'capture_extension': {
        'hooks': ['attn 输出投影模块（o_proj/dense/out_proj 现场解析）的 **forward_pre_hook** -> o_l',
                  'MLP 模块的 **forward_hook** -> m_l'],
        'stored': 'CAP[word] = (HH[L+1, HID], O{l: OIN}, M{l: HID}, logits[V])',
        'head_block_forward': ('逐头贡献**不用权重矩阵**（nf4 的 weight 是打包 uint8），而是把 o_l 的'
                               '**头块掩码**后送进**模块自身前向**，一次 [NH+1, OIN] 批调用同时得到 NH 个头块输出与全量输出。'),
        'additivity_is_definition': ('`Delta_attn := sum_h Delta_head_h` 与 `Delta_inc := Delta_attn + Delta_mlp` '
                                     '是**定义**。**注意：行为量不继承这个可加性**（见 nonlinearity 节）。'),
        'fidelity_gates': ('(i) 架构恒等式 ||(h_{l+1}-h_l) - (attn_out_l + m_l)|| / ||h_{l+1}-h_l||；'
                           '(ii) 分块可加性 ||sum_h head_block - o_proj(v)|| / ||o_proj(v)||。'
                           '容差由探针实测的 nf4 量化噪声地板决定。'),
    },

    'site_convention': {
        'site': '位点 = 层号 l；注入 hooks layers[l] 的输出（即 HH[l+1]）。与 P16 的 d_ell 站号、'
                'P17 的 w_ell 索引**逐索引对齐**。',
        'grid': 'ALL_SITES = 1..L-2（P17 的 w_all 覆盖 layer 0..L-2；位点 0 与位点 L-1 不参与判定）。'
                'REACH 与 nb 均为其子集。',
        'why_not_reach_only': ('质心用**区间求和**（com_of_mass）：区间 [s_j, s_{j+1}) 会包含不在 REACH 里的层'
                               '（如 13/15/17...）。故必须在**全域**测量，否则 com_B 与 com_V 的质量支撑不一致。'),
    },

    'behavioral_budget': {
        'estimand': ('b_{c,l} = mean over discovery pairs of [ score_of(logits(h_l^R + P_{U_l}(Delta_c_l)), ds, sid_d) '
                     '- BASE[rw].sd0 ]，其中 score_of(v, sup, sid) = v[ID(sup)] - mean_{x != sup} v[ID(x)]，'
                     '且 v[sid] = -1e9（与 Phase 8 的 T 臂逐字同口径）。'),
        'components': {
            'INC_ALL': 'dv = Delta_attn_l + Delta_mlp_l（该层的完整增量写入）',
            'INC_MLP': 'dv = Delta_mlp_l',
            'INC_ATTN': 'dv = Delta_attn_l',
            'INC_TOP1': 'dv = Delta_head_{h*}_l，h* = 该层 ||P_{U_l}(Delta_head_h)|| 最大的头（**逐层**重新选）',
            'CUM_ALL': 'dv = d_l = HH[l+1]^D - HH[l+1]^R（P16 的对象；桥接臂，只做保真度门与对照）',
        },
        'U_est': 'U_l = 6 个类别质心差的 SVD，秩 = n_classes - 1 = 5（与 P16/P17 同口径，全 41 实例按类平均）。',
        'alpha': '固定 alpha=1（注入该层完整投影写入）。不设 alpha 网格：P16 已穷举过剂量-响应形状，'
                 '本 Phase 的未知在**组件**与**对象**，不在剂量。',
        'aggregates': {
            'com_B(c)': ('把 |b_{c,l}| 按 REACH 的相邻位点区间聚合 W_j = sum_{l in [s_j, s_{j+1})} |b|，'
                         '再取 sum_j W_j mid_j / sum_j W_j，mid_j = (s_j+s_{j+1})/2 —— 与 P17 的 com_V '
                         '**逐字节同口径**（同 mid、同域、同区间求和）。'),
            'com_layer(b)': 'stat_com_layer(diff(b over REACH), REACH) —— 与 P16 的 com_layer 同口径（行为剖面质心）。',
            'share_mlp_beh(nb)': 'sum_{l in nb} |b_mlp,l| / sum_{l in nb} |b_all,l|（与 P17 的 share_mlp_nb 结构逐项对应）。',
            'share_attn_beh(nb)': 'sum_{l in nb} |b_attn,l| / sum_{l in nb} |b_all,l|',
            'share_top1_beh(nb)': 'sum_{l in nb} |b_top1,l| / sum_{l in nb} |b_all,l|',
            'share_ratio_mlp_attn(nb)': 'sum|b_mlp| / (sum|b_mlp| + sum|b_attn|)（置换零假设所用的分母）。',
        },
    },

    'component_attribution': {
        'neighbourhood': 'nb = P17 邻域 = {l in REACH : |l - com_V| <= 2}（现场从 P17 result 读入并断言）。',
        'why_fixed': ('H11 的问题字面就是"P17 那个邻域的向量份额是否也是行为份额"。'
                      '若改用 com_B 的邻域，就成了另一个问题（且需要新的预注册）。'),
        'comparison': 'share_mlp_beh(nb) 与 share_mlp_vec(nb)（后者从 P17 的 w 谱现场重算）比。',
    },

    'depth_relation': {
        'primary': 'com_B(INC_ALL) 与 com_V 的差 gap = com_V - com_B（同域 REACH）。',
        'why': ('P17 的 com_V 是**向量质量**质心；com_B 是**行为质量**质心。'
                '若 gap >= SHALLOWER_MIN 则"向量位置 != 行为位置"在**同一对象族**上成立，'
                '这把 P17 的 P4（跨对象）加强为同族内部的分离。'),
        'secondary': 'com_layer(b_all) 与 P16 的 com_layer(xhalf)/com_layer(J) 并排（描述性）。',
    },

    'efficacy_coupling': {
        'same_object': 'spearman(w_all_l, |b_all_l|)、spearman(w_mlp_l, |b_mlp_l|)、spearman(w_attn_l, |b_attn_l|)（REACH 位点对齐）。',
        'p17_counterpart': 'spearman(w_all_l, J_l)（P17 P6 口径，J 从 P16 result 现场读入）。',
        'why': ('若 same_object 为正而 p17_counterpart 为负，则 P17 的"深端写入对行为无效"是'
                '**对象错配**（增量 vs 累积）的产物，须在 MEMO 中改判。'),
    },

    'nonlinearity': {
        'r_lin': 'r_lin,l = |b_all,l - (b_mlp,l + b_attn,l)| / max(|b_all,l|, eps)',
        'why_not_error': ('行为效应**不可加**（层是非线性的），故 r_lin 不是"误差"而是**层内增益/耦合**的诊断量。'
                          '向量预算的可加性由构造成立，与它无关。'),
        'prediction': 'r_lin 的峰值应落在该臂自己的写入窗 L*_own（Phase 13 判据：写入窗 = REACH 左端点，'
                      'P16 实测 L*_own = 6/3/4）。',
    },

    'controls': {
        'permutation_null_com': ('com_B 的置换零假设：保留 |b| 的**多重集**、随机重排到 REACH 位点上（BP 次），'
                                 '取 5/95 分位；双侧。com_B 是顺序敏感量，故该零假设**非退化**。'),
        'permutation_null_share': ('份额的置换零假设：**固定 nb**（位置子集），只把 MLP 质量随机重排到 REACH 位点上。'
                                   '注意：在**全集**上取份额会因置换不变而结构性退化（P16 教训），故必须锚在固定子集。'),
        'confirmation_set': '全部预算在 discovery（n=24）上算，在 confirmation（n=17）上复核 com_B 同判。',
        'bridge_gate': ('CUM_ALL 在 L*_own 上的 b 应与 P16 result 的 FULL_SWAP 一致（二者是同一件事：'
                        'alpha=1 的累积差注入 = 供体末位隐状态）。这是一条**跨 Phase 装置门**。'),
        'domain_consistency': 'com_B 与 com_V 必须同域（REACH）、同 mid、同区间求和。',
    },

    'bootstrap': {'BP': 2000, 'seed': SEED_BASE, 'seeds': dict(SEED_NULL),
                  'note': ('三条零假设线各用独立种子：comB_inc 驱动 com_B(all)、comB_mlp 驱动 com_B(mlp)、'
                           'share_mlp 驱动邻域份额。脚本**从 exec 读取**种子，杜绝元数据与实现漂移。')},

    'floors': {
        'P18_FID_ARCH': 3.0e-2,
        'P18_FID_BLK': 1.0e-2,
        'BRIDGE_TOL_CUM': 0.20,
        'MLP_DOM_MIN': 0.50,
        'SHALLOWER_MIN': 2.0,
        'DEEP_MEDIAN': 0.5,
        'NULL_ALPHA': 0.05,
        'CONF_TOL_COMB': 3.0,
        'RLIN_PEAK_RATIO_MIN': 3.0,
    },

    'predictions': {
        'P1': dict(name='装置自检 + 保真度门',
                   claim=('三臂 Q0 device 全 GPU、T=2 布局、determinism ~ 0、hook 有效；'
                          '架构恒等式残差 <= P18_FID_ARCH 且分块可加性 <= P18_FID_BLK）。'),
                   rationale='若恒等式在 nf4 下失效，则「层增量 = 注意力 + MLP」的分解前提不成立，全部读数作废。',
                   falsified_if='任一臂 arch 残差 > 3e-2 或 blocks 残差 > 1e-2。'),
        'P2': dict(name='锚逐位复现 + 桥接门',
                   claim=('(a) P16 五条（com_layer(x)/com_layer(J)/span3/L*_own/ell_reach/REACH）+ '
                          'P17 三条（com_V/nb/share_mlp_nb）现场重算并逐位断言；'
                          '(b) CUM_ALL@L*_own 与 P16 FULL_SWAP 的相对差 <= BRIDGE_TOL_CUM。'),
                   rationale='跨 Phase 复用层索引与域；P15/P16/P17 教训：量化噪声下一切跨 Phase 常量必须现场读入 + 断言。',
                   falsified_if='任一锚不一致，或任一臂的桥接相对差 > 0.20。'),
        'P3': dict(name='holdout 主预测 1 —— H11：组件归属是行为的',
                   claim=('在 P17 邻域 nb 上，share_mlp_beh(nb) > MLP_DOM_MIN=0.50 **且**与向量份额同侧'
                          '（ATTRIBUTION_CONSISTENT），在 >= 2/3 臂。'
                          '=> P17 的向量份额不是几何假象，H11 由"几何"升级为"几何+行为"。'),
                   rationale=('A0 探针：share_mlp_beh(nb)=0.663 对向量 0.740，同侧且过半。'
                              'A1/A2 在 seal 前**未观测**。'),
                   falsified_if='>= 2/3 臂的 share_mlp_beh(nb) <= 0.50，或 >= 2/3 臂与向量份额**不同侧**。'),
        'P4': dict(name='holdout 主预测 2 —— 行为质心比向量质心浅',
                   claim=('gap = com_V - com_B(INC_ALL) >= SHALLOWER_MIN=2.0 层，在 >= 2/3 臂。'
                          '=> 「向量质量位置」与「行为质量位置」即使在**同一对象族**上也不重合。'),
                   rationale=('A0 探针：com_B=21.085 vs com_V=26.150，gap=5.065。'
                              '机制上可预期：增量写入的行为效应在浅端有一大块（写入窗），'
                              '而向量范数随深度单调上升。'),
                   falsified_if='>= 2/3 臂的 gap < 2.0 层。'),
        'P5': dict(name='holdout 主预测 3 —— 同对象耦合为正',
                   claim=('spearman(w_all, |b_all|) > 0（在 REACH 位点上，**同对象**）在 >= 2/3 臂。'
                          '=> P17 的 P6（spearman(w,J) < 0）是**对象错配**（增量 vs 累积）的产物，'
                          '须在 MEMO 中改判"深端写入对行为无效"。'),
                   rationale=('A0 探针：同对象 = +0.901，而 P17 口径（w vs J）= -0.546。'
                              '两者符号相反且幅度都很大，指向对象差异而非噪声。A1/A2 未观测。'),
                   falsified_if='>= 2/3 臂的 spearman(w_all,|b_all|) <= 0。'),
        'P6': dict(name='超可加性峰值落在写入窗',
                   claim=('argmax_l r_lin(l) == L*_own（该臂自己的写入窗）且峰值/次大 >= RLIN_PEAK_RATIO_MIN=3，'
                          '在 >= 2/3 臂。'),
                   rationale=('A0 探针：r_lin@L6=0.754，次大 0.186（L26），比值 4.05。'
                              '机制：Phase 8-9 的软门在写入窗最陡，attn 与 MLP 分量在那里耦合最强。'),
                   falsified_if='>= 2/3 臂的 argmax r_lin != L*_own，或峰值比 < 3。'),
        'P7': dict(name='对照（置换零假设 + 确认集 + 离流形诊断）',
                   claim=('置换零假设（com_B(all)/com_B(mlp)/邻域份额）、confirmation 复核、'
                          'pert_rel 离流形诊断。**只报告与判定，不设方向性预测**。'),
                   rationale=('探针已显示 com_B(all) 与邻域份额**未越出**置换零假设、com_B(mlp) 越出高尾 =>'
                              '不同量有不同区分力，这本身必须报告而不能选着报。'),
                   falsified_if='不适用（描述性条款）。'),
    },

    'verdict_tree': {
        'Q0': 'device/apparatus gate：Q0_device + F1(T=2) + F1b(tokenizer) + determinism + hook 效应',
        'Q1': 'fidelity gate：arch / blocks 残差 -> FID_PASS / FID_FAIL',
        'Q2': 'anchor gate：P16 五条 + P17 三条 -> ANCHOR_OK / ANCHOR_DRIFT',
        'Q3': 'bridge gate：|CUM@L*_own - FULL_SWAP| / |FULL_SWAP| -> BRIDGE_OK / BRIDGE_DRIFT',
        'Q4': 'attribution：share_mlp_beh(nb) vs MLP_DOM_MIN -> MLP_DOMINANT_BEH / MLP_NOT_DOMINANT_BEH',
        'Q5': 'H11 agreement：share_mlp_beh(nb) 与 share_mlp_vec(nb) 是否同侧 -> ATTRIBUTION_CONSISTENT / DISCREPANT',
        'Q6': 'depth：gap = com_V - com_B 与 SHALLOWER_MIN -> SHALLOWER / ALIGNED / DEEPER；'
              '并报 com_B vs median(REACH) -> DEEP / SHALLOW',
        'Q7': 'coupling：spearman(w_all,|b_all|) 符号 -> EFFICACY_COUPLED / EFFICACY_DECOUPLED（并排报 P17 口径）',
        'Q8': 'nonlinearity：argmax r_lin 是否 == L*_own 且峰值比 >= 3 -> SUPERADD_AT_WINDOW / NO_WINDOW_CONTRAST',
        'Q9': 'controls：置换零假设尾 + confirmation 复核 + pert_rel 上界',
    },

    'honesty': {
        'H1': '可行性探针**只在 A0 上运行**；A1 与 A2 的任何量在 seal 冻结前均未被观测。',
        'H2': ('探针**没有**计算 A1/A2 的任何量；探针只记录 A0 的 b 谱、com_B 族、份额、r_lin、桥接、'
               '保真度残差、置换零假设与耗时。'),
        'H3': ('b 是**激活级**干预读数（hook 注入），**不是权重级实现证明**（承接 N2h1-alpha-1 挂账）。'
               '与 P17 的 w 同层级。'),
        'H4': 'U_l 由**全部 41 实例**估计（与 P16/P17 同口径）=> 存在轻微选择性泄漏；任何"泛化"只指**跨臂**。',
        'H5': '保真度门容差来自 nf4 量化噪声的**实测地板**（探针 A0：1.62e-2 / 3.59e-3），不是理论误差界。',
        'H6': 'com_B 与 com_V 必须同域（REACH）、同 mid、同区间求和；全域值只入附录。',
        'H7': ('nb 只有 **2 个位点**（[26,28]）=> 邻域份额的置换零假设**几乎没有区分力**'
               '（探针 A0：p5=0.205, p95=1.379）。故 Q4/Q5 **不用零假设做门**，只报点值与跨臂一致性；'
               '零假设结果照实报告并列为限界。'),
        'H8': ('com_B(INC_ALL) 在探针 A0 上**未越出**其置换零假设（obs 21.085 vs p95 21.462），'
               '而 com_B(mlp) 越出高尾（obs 23.807 vs p95 20.920）。两者都报告。'
               'P4 是关于 com_B 与 com_V 的**比较**，不是零假设检验。'),
        'H9': ('CUM_ALL 的 pert_rel 在深端达 0.45-0.59（离流形）=> 该臂**只作桥接保真度门与对照**，'
               '不作机制解读。'),
        'H10': ('b_all 不是 (b_mlp + b_attn) 的和（层非线性 + 头是 attn 的子集）=> r_lin 是诊断量；'
                '份额的分母用 |b_all|，置换零假设的分母用 |b_mlp|+|b_attn|，两者都报。'),
        'H11': ('A0 的量在 seal 前已知（探针）；本 Phase 的**真检验在 A1/A2**（与 P16/P17 的结构一致）。'
                'A0 与 A1/A2 的共同结构是同一个 seal、同一份代码、同一批阈值。'),
        'H12': ('探针测了位点 1..35（A0 的 L=36），而判定用的 ALL_SITES = 1..L-2 = 1..34：'
                '位点 35（末层）在探针里有读数（b=7.90）但**不入任何判定量**（w_all 的支撑是 0..L-2）。'),
    },

    'probe_evidence': {
        'arm': PA['arm'], 'model': PB['model'], 'L': PB['L'], 'HID': PB['HID'], 'NH': PB['NH'],
        'head_dim': PB['head_dim'], 'o_proj_in': PB['o_proj_in'],
        'n_pairs': PB['n_pairs'], 'fwd_s': PB['fwd_s'], 'capture_s': round(PB['capture_s'], 2),
        'sweep_s': round(PB['sweep_s'], 1), 'n_forwards': PB['n_forwards'],
        'fidelity_arch': PB['fidelity_arch'], 'fidelity_blocks': PB['fidelity_blocks'],
        'com_B': PA['com_B'], 'com_V_p17': PA['com_V_p17'], 'com_V_recomputed': PA['com_V_recomputed'],
        'gap': PA['gap'], 'median_reach': float(PA['reach'] and __import__('numpy').median(PA['reach'])),
        'share_mlp_beh_nb': PA['share_mlp_beh_nb'], 'share_mlp_vec_nb': PA['share_mlp_vec_nb'],
        'share_attn_beh_nb': PA['share_attn_beh_nb'], 'share_top1_beh_nb': PA['share_top1_beh_nb'],
        'share_mlp_beh_reach': PA['share_mlp_beh_reach'],
        'com_layer_b_all': PA['com_layer_b_all'], 'com_layer_b_mlp': PA['com_layer_b_mlp'],
        'com_layer_b_attn': PA['com_layer_b_attn'],
        'spearman': PA['spearman'], 'rlin_L6': PA['rlin_L6'], 'rlin_nb_mean': PA['rlin_nb_mean'],
        'rlin_reach_mean': PA['rlin_reach_mean'],
        'cum_bridge': PA['cum_bridge'], 'full_swap': PA['full_swap'], 'bridge_rel': PA['bridge_rel'],
        'null': PA['null'],
        'note': ('以上为 A0 探针的全部读数；原件 _probe_feasibility_A0.txt 与派生 _probe_analysis_A0.txt。'
                 '探针位点 = 1..35（全域），n_pairs = 24（全 discovery 集）。'),
        'not_computed_before_seal': ['A1 的任何量', 'A2 的任何量', 'CUM_ALL 的确认集',
                                     '置换零假设在 A1/A2 上的分位', 'com_B 的全域值（A1/A2）'],
    },

    'why_not_a_HARKing_violation': (
        '探针在 seal 前只测了 A0，其读数**全部抄录**在 probe_evidence 里，读者可逐项核对。'
        '本 Phase 的**主判据是跨臂的**：P3/P4/P5/P6 的关键分支落在 A1 与 A2（seal 前未观测）；'
        'A0 只充当校准/装置臂（与 P16/P17 的 A0 角色一致）。'
        '阈值全部由**机制无关**的理由给出：P18_FID_* 由实测噪声地板取 2x 余量；'
        'BRIDGE_TOL_CUM=0.20 远宽于探针实测 0.0065（跨模型余量）；MLP_DOM_MIN=0.50 是"过半"的朴素定义；'
        'SHALLOWER_MIN=2.0 沿用 P17 的邻域宽度；RLIN_PEAK_RATIO_MIN=3.0 是"峰值显著高于次大"的朴素定义。'
        'P7 明确**不设方向性预测**，因为探针已显示不同对照量有不同区分力。'
        'P6 的判据用**该臂自己的** L*_own（臂相对），而不是把 A0 的 6 套到所有臂。'),

    'may_falsify_the_whole_line': (
        '若 P3 FAIL（行为组件归属与向量归属不同侧）=> P17 的 share_mlp_nb 是**几何假象**，'
        'N2h1-alpha 的"写入端组件归属"整条支线必须撤回重写（P8 的 L6 结论也须重审）。'
        '若 P5 FAIL（同对象耦合非正）=> P17 的 P6 保留原判，"深端写入对行为无效"成立，'
        '本 Phase 只贡献"几何位置 != 行为位置"的分离。'
        '若 P4 FAIL（gap < 2 层）=> 「向量位置 = 行为位置」成立，P17 的 P4 分离被削弱为实例噪声。'),

    'anchor_values': {'p16': ANCH16, 'p17': ANCH17},
    'arm_order': ARM_IDS,
    'components': COMPONENTS,
    'components_confirmation': COMPONENTS_CONF,
    'neighbourhood_width': 2,
}

OUT = os.path.join(P18T, 'N2h1a11_design_seal.json')
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(SEAL, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('WROTE %s  %d B  sha256=%s' % (OUT, len(b), hashlib.sha256(b).hexdigest()))
print('material: exec16=%s result16=%s exec17=%s result17=%s'
      % (fsha(EX16P)[:8], fsha(R16P)[:8], fsha(EX17P)[:8], fsha(R17P)[:8]))
for a in ARM_IDS:
    print('  anchor %-24s com_V=%8.3f nb=%s share_mlp_nb=%.4f | L*=%s reach_n=%d'
          % (a, ANCH17[a]['com_V'], ANCH17[a]['neighbourhood'], ANCH17[a]['share_mlp_nb'],
             ANCH16[a]['L_star_own'], len(ANCH16[a]['reach'])))
print('  probe A0: com_B(all)=%.3f gap=%.3f share_beh(nb)=%.4f share_vec(nb)=%.4f sp(w,b)=%+.4f bridge=%.4f'
      % (PA['com_B']['INC_ALL'], PA['gap'], PA['share_mlp_beh_nb'], PA['share_mlp_vec_nb'],
         PA['spearman']['wall_ball'], PA['bridge_rel']))
