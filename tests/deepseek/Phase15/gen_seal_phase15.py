# -*- coding: utf-8 -*-
"""
Phase 15 (N2h1-alpha-8) 预注册设计冻结（seal 生成器）。
所有判据、网格、阈值必须在任何观测前冻结；本脚本只读 Phase 12 execution（面板）与 Phase 14 seal（继承已发表量），
不读任何 Phase 15 的观测数据。产物：tests/deepseek_temp/Phase15/N2h1a8_design_seal.json
"""
import io
import os
import json
import hashlib
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
P14T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
P12T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
if not os.path.isdir(P15):
    os.makedirs(P15)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def sha8(p):
    return sha(p)[:8]


SEAL14 = os.path.join(P14T, 'N2h1a7_design_seal.json')
EXEC12 = os.path.join(P12T, 'execution_phase12.json')
S14 = json.load(io.open(SEAL14, encoding='utf-8'))
E12 = json.load(io.open(EXEC12, encoding='utf-8'))
IP = S14['inheritance_anchors']['inherited_published']

# ---------- 面板与网格：逐字继承 Phase 12（保证剖面可比） ----------
CLASSES = list(E12['classes'])
SUP_ID = dict(E12['sup_id'])
DISCOVERY = [list(x) for x in E12['discovery']]
CONFIRMATION = [list(x) for x in E12['confirmation']]
INSTANCES_ALL = [list(x) for x in E12['instances_all']]
PAIRS_ALL = [list(x) for x in E12['pairs_all']]
TMPL = E12['template']                      # '%s是一种'
PROFILE_SITES = list(E12['profile_sites'])  # 18 个位点，与 Phase 12 同构
ALPHAS = list(E12['arms']['E2_swap_curve']['alpha_grid'])   # 14 点
W = 3                                        # 集中度窗口宽（jumps 数），Phase 12/13/14 口径
XHF = 0.5                                    # xhalf = y 达 0.5*max(y) 的 alpha

ARMS = [
    dict(id='A0_calib_qwen3-4b-nf4', model='qwen3-4b', dir='qwen3-4b',
         role='quantization-fidelity calibration: 与 Phase 12 bf16 已发表量逐位点比对',
         expected=dict(num_hidden_layers=36, hidden_size=2560, num_attention_heads=32,
                       num_key_value_heads=8, head_dim=128, tie_word_embeddings=True),
         primary=1),
    dict(id='A1_glm4-9b-nf4', model='glm4-9b-chat-hf', dir='glm4-9b-chat-hf',
         role='cross-family untied replication（GLM 家族，非 Qwen）',
         expected=dict(num_hidden_layers=40, hidden_size=4096, num_attention_heads=32,
                       num_key_value_heads=2, head_dim=128, tie_word_embeddings=False),
         primary=2),
    dict(id='A2_qwen3-14b-nf4', model='Qwen3-14B', dir='Qwen3-14B',
         role='same-family untied scale-up replication（Qwen3 家族，参数 3.5x）',
         expected=dict(num_hidden_layers=40, hidden_size=5120, num_attention_heads=40,
                       num_key_value_heads=8, head_dim=128, tie_word_embeddings=False),
         primary=3),
]

for a in ARMS:
    md = os.path.join(ROOT, 'models', 'hf', a['dir'])
    a['config_sha256'] = sha(os.path.join(md, 'config.json'))
    a['config_sha8'] = a['config_sha256'][:8]
    tk = os.path.join(md, 'tokenizer.json')
    a['tokenizer_sha256'] = sha(tk) if os.path.exists(tk) else None
    a['ckpt_gb'] = round(sum(os.path.getsize(os.path.join(md, f))
                             for f in os.listdir(md) if f.endswith('.safetensors')) / 1e9, 2)

QUANT = dict(scheme='bitsandbytes nf4 (4bit)',
             bnb_4bit_quant_type='nf4',
             bnb_4bit_compute_dtype='bfloat16',
             bnb_4bit_use_double_quant=True,
             device_map='auto', max_memory={'0': '14GiB', 'cpu': '24GiB'},
             attn_implementation='eager',
             why='bf16 下 qwen3-14b (29.5GB) 需 CPU 侧 >15GB，本机 RAM 可用 ~17GB -> 加载期被硬杀（无 Python 栈）；'
                 '为保持三臂「同一数值口径」统一改为 nf4。A0 臂专职量化对其结论的影响。')

LOCALIZE = dict(
    purpose='正面回应死线的「禁止沿用 L6/U6」：写入窗在本模型上重新定位，且逐层 U_ell 独立重建',
    cands=[1, 2, 3, 4, 5, 6, 7, 9, 12, 16, 20, 25, 30, 34, 38],
    criterion='B_cat 曲线的相邻层最大增量处（Phase 8 口径）；不预设 L6，允许 L* 因模型而异',
    vector='patch 该层输出末位为 h_ell_recip + P_{U_ell}(h_ell_donor - h_ell_recip)',
    U_est='est_U(ell+1, discovery)：6 类质心差的 SVD，秩 = n_classes - 1 = 5，逐层独立',
    note='profile 臂使用裸残差差 d_ell（不投影），因此剖面结论不依赖 U；定位臂的 U 仅用于写入窗描述。',
    extra_forwards='len(cands) * n_pairs（每臂），零额外 capture',
)

SEAL = dict(
    phase=15,
    name='N2h1-alpha-8 跨模型复算「统一剖面」',
    one_sentence='把 Phase 12/13/14 在 qwen3-4b（tied, bf16）上确立的唯一有效口径——单点位点替换族的 '
                 'xhalf(ell)/J(ell) 双坐标剖面 + 置换零假设校准——在 glm4-9b 与 qwen3-14b（均 untied、'
                 '跨家族/跨规模）上独立复算，判定「两坐标 argmax 相距 13」是层栈性质还是 qwen3-4b 特例，'
                 '以及置换 null 95 分位是否普遍高达 0.70-0.75（若如此则集中度判据整条线作废）。',
    motivation=[
        'Phase 13 在 qwen3-4b 上得到 MODE_X=14 / MODE_J=1（窗口索引差 13），但无法区分「层栈性质」与「该模型特例」。',
        'Phase 14 的置换零假设给出 null95_x=0.6998 > 观测 share_x=0.5745，即 xhalf 坐标的集中度判据在该模型上无区分力；'
        '该结论是否普遍（还是 4B 特有）只能跨模型判定。',
        'Phase 14 已证「第三条独立口径不存在」（位置前缀/逐层累积均退化到单点族），故跨模型复算必须使用这条唯一有效口径。',
    ],
    model='three arms: qwen3-4b (calibration) / glm4-9b-chat-hf / Qwen3-14B ; 全部 nf4 统一口径',
    object=dict(
        template=TMPL,
        positions='T=2（pos0 = 实例词，pos1 = 框架词「是一种」）；干预只重写 pos1 末位',
        readout='最终 LayerNorm 之后的 logits 末位；统计量 = 供体类分数提升 dDonor',
        normalization='每臂独立：Y(ell, alpha) = dDonor / FULL_SWAP_arm，FULL_SWAP_arm = 该臂 discovery 面板上'
                      '「供体自身前向」的 dDonor 均值（零额外前向）',
    ),
    quantization=QUANT,
    template=TMPL,
    panel=dict(classes=CLASSES, sup_id=SUP_ID, discovery=DISCOVERY, confirmation=CONFIRMATION,
               instances_all=INSTANCES_ALL, pairs_all=PAIRS_ALL,
               n_discovery=len(DISCOVERY), n_confirmation=len(CONFIRMATION)),
    profile_sites=PROFILE_SITES,
    alphas=ALPHAS,
    intervention=dict(
        kind='single-site replacement (swap)',
        vector='h_ell_recip(clean) + alpha * (h_ell_donor - h_ell_recip)  ; alpha in [0,1]',
        note='alpha=1 即满替换为供体贴残差；alpha=0 必须还原基线（F3 型自检）',
    ),
    dose_coordinate=dict(alpha='绝对剂量', y='dDonor / FULL_SWAP_arm'),
    statistics_definition=dict(
        xhalf='y 首次达到 0.5 * max(y) 所在的 alpha（cross_alpha 线性插值），网格不变量',
        J='峰值斜率 / 其余斜率中位数（J_only，alpha>=0.01 段），网格依赖量',
        recover='alpha=1 处的 y 值',
        spearman='Spearman(stat, depth)',
    ),
    localize_arm=LOCALIZE,
    arms={a['id']: {k: v for k, v in a.items() if k != 'id'} for a in ARMS},
    arm_order=[a['id'] for a in ARMS],
    metrics=['xhalf per site', 'J per site', 'recover per site', 'XH_RANGE', 'jumps_x', 'jumps_j',
             'top3_share_x (conc_hat W=3)', 'top3_share_j', 'argmax_w_x', 'argmax_w_j',
             'null95_x', 'null95_j', 'margin_x', 'margin_j',
             'Spearman(xhalf, depth)', 'Spearman(J, depth)', 'Spearman(recover, depth)',
             'L_star_own (localization)', 'FULL_SWAP_arm'],
    bootstrap=dict(BS=0, BP=2000, seed=20261002, scheme='permutation null: 重排 jumps_x / jumps_j 后重算 top3_share',
                   note='点估计剖面 = 24 pair 均值，无需 pair bootstrap（本 Phase 不报 CI，只报 null 95 分位与裕度）'),
    pre_registered_predictions=[
        dict(id='P1', desc='三臂装置自检全部通过：T=2 独占、双前向 max|dlogits|=0、o_proj_in == n_heads*head_dim、hook 冒烟效应 != 0',
             falsified_if='任一臂出现非 T=2 实例 / 确定性偏差 > 0 / 维度等式失败 / hook 效应为 0'),
        dict(id='P2', desc='A0（nf4）的 argmax_w_x 与 Phase 12 bf16 相同（=14 窗口索引），且 max|dXHALF| <= 0.05',
             falsified_if='argmax_w_x != 14 或 max|dXHALF| > 0.05'),
        dict(id='P3', desc='A1/A2 的 null95_x >= 0.60（null 高度在不同模型上都偏高）',
             falsified_if='任一臂 null95_x < 0.60'),
        dict(id='P4', desc='A1/A2 的 d_argmax = |argmax_w_x - argmax_w_j| >= 3（两坐标分辨率分离可复现）',
             falsified_if='任一臂 d_argmax < 3'),
        dict(id='P5', desc='A1/A2 的独立定位 L*_own 落在 B_cat 曲线的平台起点，且不为 6（L6 是 4B 的写入窗，非普遍常数）',
             falsified_if='L*_own == 6 或与相邻最大增量位置不符'),
        dict(id='P6', desc='A1/A2 的 Spearman(xhalf, depth) <= -0.5（xhalf 随深度单调变小，如 4B 的 rho_x）',
             falsified_if='任一臂 Spearman > -0.5'),
        dict(id='P7', desc='A1/A2 的 XH_RANGE 与 4B 的 0.1094 同量级（0.03 ~ 0.30）',
             falsified_if='任一臂 XH_RANGE 落在 [0.03, 0.30] 之外'),
    ],
    decision=dict(
        Q0=dict(desc='装置门：三臂 P1 全通过，否则该臂 ABORT', rule='per-arm'),
        Q1=dict(desc='量化保真门：A0 与 Phase 12 bf16 已发表量比 max|dXHALF|、argmax_w_x 一致性',
                pass_rule='max|dXHALF| <= 0.05 且 argmax_w_x 相同',
                labels=['NF4_FAITHFUL', 'NF4_DEVIANT']),
        Q2=dict(desc='两坐标 argmax 距离：d_argmax = |argmax_w_x - argmax_w_j|',
                reference_4B=abs(int(IP['MODE_X_13']) - int(IP['MODE_J_13'])),
                arm_label_rule='d_argmax >= 3 -> ARGS_GAP_GE3 ; 否则 ARGS_GAP_LT3',
                joint='A1 与 A2 均 GE3 -> ARGS_GAP_LAYERSTACK ; 均 LT3 -> ARGS_GAP_4B_SPECIFIC ; 否则 ARGS_GAP_MIXED'),
        Q3=dict(desc='xhalf 坐标判据是否整条线作废：null95_x 高度',
                arm_label_rule='null95_x >= 0.70 -> NULL_X_HIGH ; 否则 NULL_X_OK',
                joint='A1 与 A2 均 HIGH -> CONC_JUDGE_INVALID_X_ALL ; 否则 CONC_JUDGE_ALIVE_X'),
        Q4=dict(desc='观测是否超 null：margin_x = share_x - null95_x ; margin_j = share_j - null95_j',
                rule='margin > 0 -> ABOVE_NULL，否则 AT_OR_BELOW_NULL（必须与 null 同报，不得单独引用 share）'),
        Q5=dict(desc='剖面形状跨模型同构性：Spearman(xhalf, depth)、Spearman(J, depth)、XH_RANGE、recover 水平',
                rule='描述性，不设硬门；用于回答「剖面形状是否层栈性质」'),
    ),
    floors=dict(
        XH_FAITHFUL_TOL=0.05,
        ARGS_GAP_REF_4B=abs(int(IP['MODE_X_13']) - int(IP['MODE_J_13'])),
        ARGS_GAP_MIN=3,
        NULL_HIGH=0.70,
        RHO_X_MAX=-0.5,
        XH_RANGE_BAND=(0.03, 0.30),
        UNREACH_y=0.10,
    ),
    honesty=[
        '1. 本 Phase 的数值口径是 nf4（4bit 权重 + bf16 计算），不是 Phase 12-14 的 bf16。A0 臂是唯一的量化保真证据；'
        '若 A0 判 NF4_DEVIANT，则 A1/A2 的一切剖面结论降级为「nf4 口径下的描述」。',
        '2. 集中度统计量 top3_share 是极值型统计量，其 null 分布由 jumps 重排得到；任何 share 值不得脱离 null95 单独引用。',
        '3. 本 Phase 不做权重级验证，全部为激活级干预。',
        '4. 本 Phase 不测位置前缀族 / 逐层累积族（Phase 14 已证其退化）；只复算单点替换族这一条唯一有效口径。',
        '5. profile 臂不使用 U 子空间，故「禁止沿用 L6/U6」自动满足；定位臂的 U_ell 逐层独立重建。',
        '6. A1 与 A2 只有两个模型，无法分离「家族」与「规模」两个因素；结论只能表述为「在 untied 的 Qwen3-14B 与 GLM4-9B 上」。',
    ],
    may_falsify_the_whole_line=[
        '若 A1/A2 的 null95_x 也 >= 0.70，则 xhalf 坐标上的「集中度」判据自 Phase 12 起就是无区分力的度量，'
        'Phase 12/13/14 中一切基于 top3_share_x 的表述必须整体撤回（Phase 14 在 4B 上的观察得到跨模型确证）。',
        '若 A0 判 NF4_DEVIANT，则本 Phase 的三臂都不足以支撑跨模型结论，需回到 bf16 路线（须先解决 14B 的内存约束）。',
    ],
    artifacts=dict(
        script='tests/deepseek/Phase15/n2h1a8_cross_model_profile.py',
        result='tests/deepseek_temp/Phase15/result_phase15.json',
        report_prefix='tests/deepseek_temp/Phase15/n2h1a8_report_',
        seal='tests/deepseek_temp/Phase15/N2h1a8_design_seal.json',
        execution='tests/deepseek_temp/Phase15/execution_phase15.json',
    ),
    inheritance_anchors=dict(
        phase14_seal_sha256=sha(SEAL14),
        phase14_seal_sha8=sha8(SEAL14),
        phase12_execution_sha256=sha(EXEC12),
        phase12_execution_sha8=sha8(EXEC12),
        inherited_published=dict(
            FULL_SWAP_12=float(IP['FULL_SWAP']),
            XH_12_by_site={k: float(v) for k, v in IP['XH_12_by_site'].items()},
            J_swap_12_by_site={k: float(v) for k, v in IP['J_swap_12_by_site'].items()},
            recover_12_by_site={k: float(v) for k, v in IP['recover_12_by_site'].items()},
            XH_RANGE_12=float(IP['XH_RANGE_12']),
            SHARE_X_13=float(IP['SHARE_X_13']),
            SHARE_J_13=float(IP['SHARE_J_13']),
            MODE_X_13=int(IP['MODE_X_13']),
            MODE_J_13=int(IP['MODE_J_13']),
            Q_ELL_12={k: float(v) for k, v in IP['Q_ELL_12'].items()},
        ),
        note='已发表量只用于 A0 的量化保真校准与 Q2 的参照 d_argmax=13；不得用于「复现」A1/A2（不同模型）。',
    ),
    why_not_a_HARKing_violation='本 seal 在任何 Phase 15 观测前冻结；网格与面板逐字继承 Phase 12（保证可比），'
                                '唯一的新增自由度（nf4 口径）由预先登记并设门的 A0 臂承担；'
                                'Q1 的容差 0.05 由 Phase 12 的 XH_RANGE=0.1094 与 alpha 网格步长 0.05/0.1 论证得到，'
                                '不是观测后选取。',
    frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
)

OUT = os.path.join(P15, 'N2h1a8_design_seal.json')
io.open(OUT, 'w', encoding='utf-8', newline='\n').write(
    json.dumps(SEAL, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('SEAL -> %s' % OUT)
print('bytes=%d sha256=%s' % (len(b), hashlib.sha256(b).hexdigest()))
print('sha8=%s' % hashlib.sha256(b).hexdigest()[:8])
print('arms=%d ; profile_sites=%d ; alphas=%d ; W=%d ; XHF=%.2f' %
      (len(ARMS), len(PROFILE_SITES), len(ALPHAS), W, XHF))
print('d_argmax_ref_4B=%d' % SEAL['decision']['Q2']['reference_4B'])
for a in ARMS:
    print('  %-24s %-18s L=%d hid=%d heads=%d kv=%d tie=%s ckpt=%.2fGB cfg=%s' %
          (a['id'], a['model'], a['expected']['num_hidden_layers'], a['expected']['hidden_size'],
           a['expected']['num_attention_heads'], a['expected']['num_key_value_heads'],
           a['expected']['tie_word_embeddings'], a['ckpt_gb'], a['config_sha8']))
