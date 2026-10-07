# -*- coding: utf-8 -*-
"""
Phase 17 (N2h1-alpha-10) 预注册 seal 生成器。
产物：tests/deepseek_temp/Phase17/N2h1a10_design_seal.json
口径：材料逐字节继承 Phase 16（template/classes/instances/pairs/quant 全部原样搬运）。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
os.makedirs(P17T, exist_ok=True)


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


EX16P = os.path.join(P16T, 'execution_phase16.json')
R16P = os.path.join(P16T, 'result_phase16.json')
S16P = os.path.join(P16T, 'N2h1a9_design_seal.json')
A16P = os.path.join(P16T, 'N2h1a9_design_seal_amend1.json')
EX16 = json.load(io.open(EX16P, encoding='utf-8'))
R16 = json.load(io.open(R16P, encoding='utf-8'))
ARMS = EX16['arms']
ARM_IDS = list(ARMS.keys())

# 从 P16 result 现场读出要作为锚的量（P2 复现对象）
ANCH = {}
for a in ARM_IDS:
    ANCH[a] = dict(
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

PROBE = json.load(io.open(os.path.join(P17T, '_probe_feasibility_A0.json'), encoding='utf-8'))
PROBE_TXT = os.path.join(P17T, '_probe_feasibility_A0.txt')

SEED_BASE = 20261003
SEED_NULL = {'comv_all': SEED_BASE + 71, 'comv_mlp': SEED_BASE + 83}

SEAL = {
    'phase': 17,
    'line': 'N2h1-alpha-10',
    'title': ('写入向量的位置与效力：向量质量质心 com_V（逐层组件预算的精确可加分解）、'
              '组件归属、以及它与行为增益 J 与行为质心 com_layer 的三向对照'),
    'kind': 'design_seal',
    'created_local': time.strftime('%Y-%m-%d %H:%M:%S'),
    'supersedes': None,

    'motivation': {
        'gap_1': ('Phase 16 的限界明写：`com_layer` 是**描述性位置量** —— 能说「变化在 L24 附近」，'
                  '**不能说「L24 做了什么」**。位置与组件之间没有桥。Phase 8 的组件预算只做了**一个层**'
                  '（L6），而 Phase 16 说质心在 A0 是 L23、A1 是 L18、A2 是 L8 —— 谁都没查过那些层。'),
        'gap_2': ('Phase 16 的 P6 否证把「`xhalf` 深尾集中 / `J` 浅端集中」的**物理深度表述整体撤回**，'
                  '但**没有给出替代的位置量**。撤回之后，"位置"在这个系统里到底是行为读数的性质、'
                  '还是写入向量的性质，完全悬空。'),
        'gap_3': ('Phase 10 说行为增益 J(ell) 随深度单调下降（spearman(J,depth) ~ -1）；'
                  '而**向量层面的写入质量** w_ell 从未被测过。若二者反向，则"深端有大量写入但对行为无效"'
                  '会成为一条全新的机制陈述；若同向，则 com_layer 可以当作 J 的代理。二者必择其一，'
                  '这是本 Phase 的核心可证伪点。'),
        'fix': ('① 扩展 capture 存 o_proj 输入与 MLP 输出，把 Phase 8 的**向量预算 share_v**（精确可加）'
                '从单层 L6 推广到**逐层**，得到向量质量谱 w_ell；② 定义向量质量质心 com_V（与 stat_com_layer '
                '同一 mid 口径，REACH 域），并与 P16 的两个行为质心三向对照；③ 组件归属（MLP vs attn vs top-1 头）'
                '在 com_V 邻域报告；④ 计算 spearman(w_ell, J_ell)；⑤ span_k 体系化到 k in {2,3,5}。'),
    },

    'material_source': {
        'from_execution': 'tests\\deepseek_temp\\Phase16\\execution_phase16.json',
        'from_execution_sha256': fsha(EX16P),
        'from_result_sha256': fsha(R16P),
        'from_seal_sha256': fsha(S16P),
        'from_amend1_sha256': fsha(A16P),
        'inherited_keys': ['template', 'classes', 'instances_all', 'pairs_all', 'discovery',
                           'confirmation', 'quant', 'profile_sites', 'alphas'],
        'note': '禁止在 Phase 17 中改动词表/实例/配对/量化口径/模板；改动即视为新研究线。',
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
        'why': '向量预算需要把某一层的增量拆成「注意力(逐头) + MLP」两部分。',
        'hooks': ['attn 输出投影模块（o_proj/dense/out_proj 现场解析）的 **forward_pre_hook** -> o_l（o_proj 的输入，即拼接的逐头输出）',
                  'MLP 模块的 **forward_hook** -> m_l（MLP 输出）'],
        'stored': 'CAP[word] = (HH[L+1, HID], O{l: OIN}, M{l: HID}, logits[V])',
        'head_block_forward': ('逐头贡献**不用权重矩阵**（nf4 的 `weight` 是打包的 uint8，不能直接矩阵乘；'
                               '且 dequantize 会引入额外口径），而是把 o_l 的**头块掩码**后送进**模块自身前向**，'
                               '一次 [NH+1, OIN] 批调用同时得到 NH 个头块输出与全量输出。'),
        'additivity_is_definition': ('`Delta_attn := sum_h Delta_head_h` 与 `Delta_inc := Delta_attn + Delta_mlp` '
                                     '是**定义**，不是测量 —— 向量预算的精确可加性由此**由构造成立**。'),
        'fidelity_gates': ('另设两个**保真度门**检验这一构造与模型实际计算一致：'
                           '(i) 架构恒等式 ||(h_{l+1}-h_l) - (attn_out_l + m_l)|| / ||h_{l+1}-h_l||；'
                           '(ii) 分块可加性 ||sum_h head_block - o_proj(v)|| / ||o_proj(v)||。'
                           '两者都受 nf4 量化噪声限制，容差由探针实测地板决定。'),
    },

    'budget': {
        'functional': 'P_{U_l}(v) = (v @ U_l^T) @ U_l ; ||P_{U_l}(v)|| 记为质量',
        'U_est': ('U_l = est_U(l+1)：6 个类别质心差的 SVD，秩 = n_classes - 1 = 5，'
                  '与 Phase 16 E3 同口径（全 41 实例按类平均）。'),
        'per_component': ['head0..head{NH-1}', 'mlp'],
        'denominator': '质量分母 = sum over {NH 个头, MLP}（与 Phase 8 同）；incoming/累计量只记账不入分母。',
        'forbidden': '**禁用效应份额**（Phase 8 铁律 (a)）：效应量不可加，份额判据一律用向量预算。',
    },

    'mass_profile': {
        'w_ell': 'mean over discovery pairs of ||P_{U_l}(Delta_inc_l)||',
        'component_mass': 'w_ell^{mlp} / w_ell^{attn} / w_ell^{top1head} 同式，分量替换 Delta_inc_l',
        'com_V': ('把逐层质量按 REACH 的相邻位点区间聚合 W_j = sum_{l in [s_j, s_{j+1})} w_l，'
                  '再取 com_V = sum_j W_j * mid_j / sum_j W_j，mid_j = (s_j + s_{j+1}) / 2 '
                  '—— 与 `stat_com_layer` **同一 mid 口径**，单位「层」。'),
        'domains': '主域 = REACH（与 P16 的 com_layer 同域）；全域 1..L-1 只作对照，不入判据。',
        'why_absolute_mass': ('行为效应随写入向量幅度增大，故与 Y(ell, alpha) 对应的是**绝对**质量，'
                              '不是相对（类别占比）。相对口径只作诊断。'),
    },

    'component_attribution': {
        'neighbourhood': 'com_V 邻域 = REACH 中 |l - com_V| <= 2 的位点',
        'metrics': ['share_mlp_nb = sum_{l in nb} w^{mlp}_l / sum_{l in nb} w_l',
                    'share_attn_nb = 1 - share_mlp_nb',
                    'top1_head_share_nb = max_h sum_{l in nb} w^{head h}_l / sum_{l in nb} w_l'],
        'com_V_by_component': 'com_V^{mlp} / com_V^{attn} / com_V^{top1head}',
    },

    'efficacy_relation': {
        'stat': 'spearman(w_ell, J_ell)，在 REACH 位点上取同站点对齐（J 由 P16 result 现场读入）',
        'why': 'Phase 10 已确立 J(ell) 随深度单调下降；w_ell 是新量。二者关系是本 Phase 的核心未知。',
    },

    'span_spectrum': {
        'ks': [2, 3, 5],
        'stat': 'stat_span_k(jumps, k)，jumps = 行为剖面（xhalf / J）在 REACH 上的相邻差',
        'coupling': ('报告 span 序（x 相对 J 更窄/更宽）与 com_layer 序（x 相对 J 更深/更浅）'
                     '在 3 臂上是否同号 -> SPAN_CENTROID_COUPLED / DECOUPLED。'),
    },

    'controls': {
        'permutation_null': ('com_V 的置换零假设：保留 w_ell 的**多重集**、随机重排到 REACH 位点上 '
                             '（BP 次），取 5/95 分位；双侧。注意 com_V 是顺序敏感量，故该零假设**非退化**。'),
        'confirmation_set': '预算在 discovery（n=24）上算，在 confirmation（n=17）上复核同判。',
        'domain_consistency': 'com_V 与 com_layer 必须同域（REACH）；跨域比较只在附录。',
    },

    'bootstrap': {'BP': 2000, 'seed': SEED_BASE, 'seeds': dict(SEED_NULL),
                  'note': ('com_V 的置换零假设分两条线各用一个独立种子：`comv_all` 驱动 com_V(all)、'
                           '`comv_mlp` 驱动 com_V(mlp)，协议相同（随机重排质量在 REACH 位点上的归属）。'
                           '脚本**从 exec 读取**种子，杜绝元数据与实现漂移（P16 教训）。')},

    'floors': {
        'P17_FID_ARCH': 3.0e-2,
        'P17_FID_BLK': 1.0e-2,
        'DEEP_MEDIAN': 0.5,
        'CENTROID_SEP_MIN': 4.0,
        'MLP_DOM_MIN': 0.50,
        'NULL_ALPHA': 0.05,
        'CONF_TOL_COMV': 3.0,
        'CONF_TOL_SPAN': 0.25,
    },

    'predictions': {
        'P1': dict(name='装置自检 + 保真度门',
                   claim=('三臂 Q0 device 全 GPU、T=2 布局、determinism ~ 0、hook 有效；'
                          '架构恒等式残差 <= P17_FID_ARCH 且分块可加性 <= P17_FID_BLK（全层全臂）。'),
                   rationale='若恒等式在 nf4 下失效，则「层增量 = 注意力 + MLP」这一分解前提不成立，后续全部读数作废。',
                   falsified_if='任一臂的 arch 残差 > 3e-2 或 blocks 残差 > 1e-2。'),
        'P2': dict(name='P16 锚逐位复现',
                   claim=('从 P16 result 现场读入 com_layer(x)/com_layer(J)/span3/L*_own/ell_reach/REACH，'
                          '断言与本 seal 记录的 anchor_values 逐位一致（装置门，非科学预测）。'),
                   rationale='跨 Phase 复用层索引与域；P15 教训：量化噪声下 argmax 会换窗，故一切跨 Phase 常量必须现场读入 + 断言。',
                   falsified_if='任一字段不一致。'),
        'P3': dict(name='holdout 主预测 —— 写入质量深端集中',
                   claim=('**A1 与 A2**（seal 前**未观测**）上 com_V >= median(REACH)，'
                          '即向量写入质量在可达域右半集中。'),
                   rationale=('单位置的探针（A0）显示 w_ell 在 L28-L34 达 30-45、在 L6 仅 12.8，'
                              'com_V=26.15 落在 REACH 中位 19 之上。若这是层栈共性，则两个未见臂同判；'
                              '若只是 A0 的性质，则本 Phase 头条垮掉。'),
                   falsified_if='A1 或 A2 的 com_V < median(REACH)。'),
        'P4': dict(name='三向桥接 + A2 判别臂',
                   claim=('com_V 与两个行为质心的距离 d_x = |com_V - com_layer(x)|、d_j = |com_V - com_layer(J)|。'
                          '**A2 为判别臂**（其 com_layer(x)=8.42 与 com_layer(J)=9.14 几乎重合，无法靠 d 的'
                          '大小区分坐标）=> 预注册：A2 的 com_V 落在深端且 min(d_x, d_j) >= CENTROID_SEP_MIN=4 层。'
                          '若成立 => 「向量位置 != 行为位置」，P16 的「描述性位置量」限界被加强为'
                          '「行为质心不能由向量质量质心替代」。'),
                   rationale=('A0 的 d_x=3.14、d_j=17.31（探针已知），两者差异巨大，故 A0 不具判别力；'
                              'A2 的行为质心在浅端而 A0 在深端，是天然的分辨点。'),
                   falsified_if='A2 的 min(d_x, d_j) < 4 层。'),
        'P5': dict(name='组件归属 —— MLP 主导',
                   claim='com_V 邻域（+-2 层）的 MLP 向量预算份额 share_mlp_nb > MLP_DOM_MIN=0.50，在 >=2/3 臂。',
                   rationale=('Phase 8 在 L6 得 MLP 向量预算 0.4717（最大单头仅 0.0742）；'
                              '深端 MLP 是否更主导是未测的。'),
                   falsified_if='>=2/3 臂的 share_mlp_nb <= 0.50。'),
        'P6': dict(name='效力关系 —— 写入质量与行为增益反向',
                   claim='spearman(w_ell, J_ell) < 0，在 >=2/3 臂（A0 为已知；A1/A2 为 holdout）。',
                   rationale=('J 随深度单调下降（Phase 10）；若 w_ell 随深度上升则反向。'
                              '反向的含义是"深端有大量写入但对行为无效"，是一条新机制陈述。'),
                   falsified_if='>=2/3 臂的 spearman(w,J) >= 0。'),
        'P7': dict(name='span_k 跨度谱体系化',
                   claim=('span_k 在 k in {2,3,5}、双坐标、三臂上给出完整谱，并判定 '
                          'span 序 与 com_layer 序 是否同号 -> SPAN_CENTROID_COUPLED / DECOUPLED。'
                          '本条**只报告与判定，不设方向性预测**（Phase 16 已见过 k=3 的 2/3，'
                          '再预测同一事实即为 HARKing）。'),
                   rationale='跨度谱是 P16 遗留的对照项；体系化后才知道「J 更窄」是否跨模型稳健。',
                   falsified_if='不适用（描述性条款；判定本身可被后续 Phase 否证）。'),
    },

    'verdict_tree': {
        'Q0': 'device/apparatus gate：Q0_device + F1(T=2) + F1b(tokenizer) + determinism + hook 效应',
        'Q1': 'fidelity gate：arch / blocks 两个残差与各自 floors 比较 -> FID_PASS / FID_FAIL',
        'Q2': 'anchor gate：P16 锚逐位 -> ANCHOR_OK / ANCHOR_DRIFT',
        'Q3': 'deep-mass：每臂 com_V 与 median(REACH) 比较 -> DEEP / SHALLOW；联合 DEEP_ALL / DEEP_PARTIAL / DEEP_NONE',
        'Q4': 'bridge：d_x / d_j 与 CENTROID_SEP_MIN -> TRANSFORM_ALIGNED（min d < 4） / POSITION_DECOUPLED（>=4）',
        'Q5': 'attribution：share_mlp_nb 与 MLP_DOM_MIN -> MLP_DOMINANT / MLP_NOT_DOMINANT',
        'Q6': 'efficacy：spearman(w,J) 符号 -> WRITE_EFFICACY_ANTICORR / WRITE_EFFICACY_COUPLED',
        'Q7': 'span：SPAN_CENTROID_COUPLED / DECOUPLED + 谱表',
    },

    'honesty': {
        'H1': '可行性探针**只在 A0 上运行**；A1 与 A2 的任何量在 seal 冻结前均未被观测。',
        'H2': ('探针**没有**计算 spearman(w_ell, J_ell)，也**没有**计算 A1/A2 的任何量；'
               '探针只记录了 A0 的 w_ell 谱、com_V 族、保真度残差、耗时与 share 汇总。'),
        'H3': ('`w_ell` 是**激活级**的写入向量分解（hook 出的 o_proj 输入与 MLP 输出），'
               '**不是权重级实现证明**（承接 N2h1-alpha-1 挂账）。'),
        'H4': ('U_l 由**全部 41 实例**估计（与 P16 E3 同口径）=> 存在轻微选择性泄漏；'
               '本 Phase 的任何"泛化"表述只指**跨臂**，不指跨词表/跨实例。'),
        'H5': '保真度门容差来自 nf4 量化噪声的**实测地板**（探针 A0：1.62e-2 / 3.59e-3），不是理论误差界。',
        'H6': 'com_V 与 com_layer 的域口径必须一致（REACH）；全域值只入附录，不入判据。',
        'H7': ('逐头分解在 bf16/4-bit 下只有 ~3.6e-3 的相对精度 => 头部质心之间小于该量级的差异'
               '不做机制解读。'),
        'H8': ('P6 的 A0 分支在 seal 前**未观测**（见 H2），但 A0 的 w_ell 单调性与 Phase 10 的 J 单调性'
               '各自公开 => 该预测在 A0 上属"由两条已知事实推出的推论"，其**真正的检验是 A1/A2**。'),
    },

    'probe_evidence': {
        'arm': PROBE['arm'], 'model': PROBE['model'], 'L': PROBE['L'], 'HID': PROBE['HID'],
        'NH': PROBE['NH'], 'head_dim': PROBE['head_dim'], 'o_proj_in': PROBE['o_proj_in'],
        'n_pairs': PROBE['n_pairs'], 'fwd_s': PROBE['fwd_s'],
        'capture_s': round(PROBE['capture_s'], 2), 'decomp_s': round(PROBE['decomp_s'], 3),
        'fidelity_arch': PROBE['fidelity_arch'], 'fidelity_blocks': PROBE['fidelity_blocks'],
        'com_V': PROBE['com_V'], 'com_V_mlp': PROBE['com_V_mlp'], 'com_V_attn': PROBE['com_V_attn'],
        'com_V_top': PROBE['com_V_top'], 'com_V_full': PROBE['com_V_full'],
        'd_x': abs(PROBE['com_V'] - PROBE['com_layer_p16_x']),
        'd_j': abs(PROBE['com_V'] - PROBE['com_layer_p16_j']),
        'share_attn': PROBE['share_attn'], 'share_mlp': PROBE['share_mlp'],
        'w_all': PROBE['w_all'], 'argmax_w_layer': int(max(range(len(PROBE['w_all'])),
                                                           key=lambda i: PROBE['w_all'][i])),
        'note': '以上为 A0 探针的全部读数；txt 原件 _probe_feasibility_A0.txt。',
        'not_computed_before_seal': ['spearman(w_ell, J_ell)', 'A1 的任何量', 'A2 的任何量',
                                     'share_mlp_nb（邻域口径）', '置换零假设分位', 'span 谱'],
    },

    'why_not_a_HARKing_violation': (
        '探针在 seal 前只测了 A0，其读数**全部抄录**在 probe_evidence 里，读者可逐项核对。'
        '本 Phase 的**主判据是跨臂的**：P3/P4/P6 的关键分支落在 A1 与 A2（seal 前未观测）；'
        'A0 只充当校准/装置臂（与 Phase 16 的 A0 角色一致）。'
        '阈值全部由**机制无关**的理由给出：P17_FID_* 由实测噪声地板取 2x 余量；'
        'DEEP_MEDIAN=0.5 是"落在域右半"的朴素定义；CENTROID_SEP_MIN=4.0 沿用 Phase 16 '
        '（2x 邻域宽度）；MLP_DOM_MIN=0.50 是"过半"的朴素定义。没有任何阈值是为迁就观测值而设。'
        'P7 明确**不设方向性预测**，正是因为 Phase 16 已发表过 k=3 的同一事实。'),

    'may_falsify_the_whole_line': (
        '若 P3 FAIL（A1 或 A2 的 com_V 落在可达域左半），则「向量写入质量深端集中」不是层栈共性，'
        '本 Phase 必须改判为否证叙述，并撤回任何把 com_V 当作"写入位置"的机制表述（保留的只有 A0 的单臂描述）。'
        '若 P6 FAIL（spearman(w,J) >= 0 在 >=2/3 臂），则撤回「写入质量与行为增益解耦」这一说法。'),

    'anchor_values': ANCH,
    'arm_order': ARM_IDS,
}

OUT = os.path.join(P17T, 'N2h1a10_design_seal.json')
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(SEAL, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('WROTE %s  %d B  sha256=%s' % (OUT, len(b), hashlib.sha256(b).hexdigest()))
print('material: exec16=%s result16=%s' % (fsha(EX16P)[:8], fsha(R16P)[:8]))
for a in ARM_IDS:
    print('  anchor %-24s com_layer x=%8.3f j=%8.3f  L*=%s ell_reach=%s'
          % (a, ANCH[a]['com_layer_x'], ANCH[a]['com_layer_j'],
             ANCH[a]['L_star_own'], ANCH[a]['ell_reach']))
