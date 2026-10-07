# -*- coding: utf-8 -*-
"""生成 Phase 16 (N2h1-alpha-9) 预注册设计 seal。

纪律：
  - 实验材料（template/classes/instances/pairs/discovery/confirmation/quant）**逐字节继承**
    Phase 15 的 execution（防止「换一批词再跑」的隐性自由度）；
  - 新增量：profile_sites 下探到 1..5（使写入窗落入剖面域）、集中度重设计（com_layer/span_k）、
    可达性掩膜（rho >= UNREACH_y）；
  - 判据（P1..P7 / Q0..Q5）在任何观测前冻结。
"""
import io
import os
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
os.makedirs(P16T, exist_ok=True)

EX15 = json.load(io.open(os.path.join(P15T, 'execution_phase15.json'), encoding='utf-8'))
RES15 = json.load(io.open(os.path.join(P15T, 'result_phase15.json'), encoding='utf-8'))
AM1 = json.load(io.open(os.path.join(P15T, 'N2h1a8_design_seal_amend1.json'), encoding='utf-8'))


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


PROFILE_LEGACY = list(EX15['profile_sites'])
PROFILE_NEW = [1, 2, 3, 4, 5] + PROFILE_LEGACY

# 冻结锚（从 Phase 15 result 现场读出，保证锚不是手抄）
ANCH = {}
for a in EX15['arm_order']:
    ANCH[a] = dict(
        FULL_SWAP=RES15['E2_full_swap'][a]['FULL_SWAP'],
        L_star_own=RES15['E3_localize'][a]['L_star_own'],
        xhalf_by_site={str(s): RES15['E4_summary'][a]['xhalf'][i]
                       for i, s in enumerate(RES15['E4_summary'][a]['sites'])},
        J_by_site={str(s): RES15['E4_summary'][a]['J'][i]
                   for i, s in enumerate(RES15['E4_summary'][a]['sites'])},
        legacy_top3_x=RES15['E5_concentration'][a]['top3_x'],
        legacy_top3_j=RES15['E5_concentration'][a]['top3_j'],
        legacy_argmax_w_x=RES15['E5_concentration'][a]['argmax_w_x'],
        legacy_argmax_w_j=RES15['E5_concentration'][a]['argmax_w_j'],
        legacy_null95_x=RES15['E5_concentration'][a]['null_x']['null95'],
        legacy_null95_j=RES15['E5_concentration'][a]['null_j']['null95'],
        legacy_margin_x=RES15['E5_concentration'][a]['margin_x'],
        legacy_margin_j=RES15['E5_concentration'][a]['margin_j'],
    )

SEAL = dict(
    phase=16,
    line='N2h1-alpha-9',
    title=('写入窗原点化剖面：把写入窗 L*_own 变成剖面域左端点（下探 ell=1..5 + 可达性掩膜），'
           '并以「物理深度质心 com_layer / top-k 跨度 span_k」双量替换极值型 3-窗占比'),
    kind='design_seal',
    created_local=time.strftime('%Y-%m-%d %H:%M:%S'),
    supersedes=None,

    motivation=dict(
        gap_1=('Phase 15 的独立定位给出 L*_own = A0 6 / A1 3 / A2 4，而剖面域是 PROFILE=[6..34]。'
               'A1/A2 的写入窗**落在剖面域之外** ⇒「写入窗 vs 集中窗」关系在既有口径下**不可比较**，'
               'Phase 15 §8 写死的最高优先死线因此尚未真正被回答。'),
        gap_2=('Phase 15 的集中度判据 top3_share = max_w|sum(jm[j:j+W])|/range 是**极值型 3-窗占比**：'
               '等价于「最大 3 步净位移 / 全幅」。在 6 个「臂 × 坐标」格里只有 2 格超过置换 null 95 分位，'
               '且窗位由曲线形状（单调段端部）决定。'),
        gap_3=('置换零假设**保留 jump 的多重集** ⇒ 任何只依赖多重集的量（熵 / max / max-mean 比）'
               '在零假设下恒等于观测值，**结构性地不可检验**。Phase 15 未声明这一退化族。'),
        fix=('① 把 profile_sites 下探到 1..5，并以可达性掩膜 REACH={ell : rho(ell) >= UNREACH_y} 定义主域'
             '（rho = Y(ell, alpha=1) = dDonor/FULL_SWAP），使写入窗成为主域左端点；'
             '② 以**顺序敏感 + 尺度无关 + 定义在物理深度轴**的双量 (com_layer, span_k) 替换极值型量；'
             '③ 在同一置换零假设（BP=2000）下对 (com_layer, span_k) 做**双边**检验，并与旧量判决逐格对照。'),
    ),

    # ---- 实验材料：逐字节继承 Phase 15
    material_source=dict(
        from_execution=os.path.join('tests', 'deepseek_temp', 'Phase15', 'execution_phase15.json'),
        from_execution_sha256=sha(os.path.join(P15T, 'execution_phase15.json')),
        from_result_sha256=sha(os.path.join(P15T, 'result_phase15.json')),
        from_amend1_sha256=sha(os.path.join(P15T, 'N2h1a8_design_seal_amend1.json')),
        inherited_keys=['template', 'classes', 'instances_all', 'pairs_all',
                        'discovery', 'confirmation', 'quant'],
        note='禁止在 Phase 16 中改动词表/实例/配对/量化口径；改动即视为新研究线。',
    ),
    template=EX15['template'],
    classes=EX15['classes'],
    instances_all=EX15['instances_all'],
    pairs_all=EX15['pairs_all'],
    discovery=EX15['discovery'],
    confirmation=EX15['confirmation'],
    quant=EX15['quant'],
    sup_id_semantics=('逐臂由**该臂 tokenizer** 现场解析类别词 id，并断言 6/6 类别词恰为单 token 且 '
                      'decode 可逆（F1b）；禁止跨词表沿用任何硬编码 id（Phase 15 amend1 事故）。'),
    sup_id_ref_qwen=EX15['sup_id'],
    amend1_lesson=dict(kind=AM1['kind'], root_cause=AM1.get('defect_root_cause'),
                       evidence=AM1.get('evidence_from_device_gate')),

    # ---- 网格
    profile_sites=PROFILE_NEW,
    profile_sites_legacy=PROFILE_LEGACY,
    alphas=EX15['alphas'],
    localize=dict(EX15['localize'],
                  note='E3 独立写入窗定位：逐层独立 U_ell = est_U(ell+1)（6 类质心差 SVD，秩 5），'
                       'curve[ell] = 用 proj(hd-hr, U_ell) 替换后的 dDonor；L*_own = 相邻层最大增量处。'
                       'CANDS 与 Phase 15 完全相同（含 1..5），仅被截断到 < L-1。'),
    concentration=dict(
        W=EX15['W'],
        coord_x='xhalf(ell) = cross_alpha(ALPHAS, Y[ell], xh_frac)（每臂每站点自归一：目标 = 0.5 * 该站点 max）',
        coord_j='J(ell) = J_only(ALPHAS, Y[ell])（峰值斜率 / 其余斜率中位数，逐字沿用 Phase 12）',
        legacy='top3_share = max_w |sum(jm[j:j+W])| / (F.max()-F.min())',
        new=dict(
            com_layer=('com_layer(F) = sum_j |dj| * mid_j / sum_j |dj|，'
                       'mid_j = (sites[j]+sites[j+1])/2 为**物理层号**中点；'
                       '单位为「层」。定义在物理轴上 ⇒ 对网格加密/下探不变。'),
            com_layer_signed=('com_signed(F) = sum_j dj * mid_j / sum_j dj（有符号版，仅记录，不作判据）'),
            span_k=('span_k(F) = (max_idx - min_idx of the k 个最大 |dj|) / (n_jumps - 1)，k=3；'
                    '尺度无关、顺序敏感；小 = 主变集中在少数相邻步，大 = 摊开'),
        ),
        excluded_family=('**被排除的量族（结构性退化）**：任何只依赖 jump 多重集的量 —— 谱熵、'
                         'max|dj|/mean|dj|、Σ|dj|²/(Σ|dj|)² 等 —— 在「多重集随机排列」零假设下**恒等于观测值**，'
                         '双边检验必得 p=1，故一律不得作为集中度判据。'),
    ),
    reachability=dict(
        rho_def='rho(ell) = Y(ell, alpha=1) = dDonor(ell, alpha=1) / FULL_SWAP（FULL_SWAP 为 Phase 15 冻结值）',
        mask='REACH = {ell in profile_sites : rho(ell) >= UNREACH_y}',
        note=('主域 = REACH（写入窗应成为其左端点）；legacy 域 = [6..34] 用于 P2 冻结锚复现。'
              '被掩膜排除的位点必须**逐个报告**（位数 + rho 值），不得静默丢弃。'),
    ),
    bootstrap=dict(BP=int(EX15['bootstrap']['BP']), seed=int(EX15['bootstrap']['seed']),
                   seeds=dict(legacy_x=int(EX15['bootstrap']['seed']) + 13,
                              legacy_j=int(EX15['bootstrap']['seed']) + 29,
                              new_x=int(EX15['bootstrap']['seed']) + 41,
                              new_j=int(EX15['bootstrap']['seed']) + 53),
                   note='新旧统计量用**不同** rng 种子但同一置换协议（BP=2000，单边/双边分位同时记录）。'),

    inheritance=dict(
        from_phase=15,
        anchors=ANCH,
        floors_from_15=EX15['floors'],
        note=('Phase 15 的 legacy 值在此后视为**冻结锚**：Phase 16 重算 6..34 区间必须复现（P2）。'
              '若复现失败 ⇒ Q1_recon = RECON_DRIFT，全部跨 Phase 比较作废。'),
    ),

    floors=dict(
        UNREACH_y=float(EX15['floors']['UNREACH_y']),
        XH_FAITHFUL_TOL=float(EX15['floors']['XH_FAITHFUL_TOL']),
        RECON_TOL_XH=1e-3,
        RECON_TOL_J_REL=2e-2,
        CENTROID_SEP_MIN=4.0,
        CENTROID_AFTER_WIN_MIN=5.0,
        NEW_NONDEG_MIN=2,
        NULL_HIGH=float(EX15['floors']['NULL_HIGH']),
    ),

    predictions=dict(
        P1=dict(name='装置自检', crit='三臂 Q0_device = PASS（F1b_ok & T2_only & F4_dims_ok & F2_base_ok & determinism==0 & hook_effect>0）'),
        P2=dict(name='冻结锚逐位复现',
                crit='三臂 legacy(6..34)：max|xhalf_new - xhalf_15| <= RECON_TOL_XH '
                     '且 max_rel|J_new - J_15| <= RECON_TOL_J_REL '
                     '且 legacy argmax_w_x / argmax_w_j 与 Phase 15 整数完全相同'),
        P3=dict(name='写入窗入域且重算一致',
                crit='三臂 L*_own(本次) == L*_own(Phase 15)（6/3/4）且落在 [profile_sites[0], profile_sites[-1]]'),
        P4=dict(name='可达域左端点 == 写入窗',
                crit='三臂 ell_reach := min{ell in profile_sites : rho(ell) >= 0.5} 满足 ell_reach == L*_own（严格相等）'),
        P5=dict(name='新量不劣于旧量（重设计有效）',
                crit='6 格中 com_layer 双边显著格数 >= 旧量 top3_share 单边显著格数，且 com_layer 显著格数 >= NEW_NONDEG_MIN'),
        P6=dict(name='物理深度质心分离',
                crit='com_layer(xhalf) - com_layer(J) >= CENTROID_SEP_MIN (4.0 层) 在 3/3 臂成立'),
        P7=dict(name='xhalf 质心位于写入窗之后',
                crit='com_layer(xhalf) - L*_own >= CENTROID_AFTER_WIN_MIN (5.0 层) 在 >= 2/3 臂成立'),
    ),

    verdict_tree=dict(
        Q0='PASS / FAIL（装置）',
        Q1_recon="RECON_OK / RECON_DRIFT（P2）",
        Q2_reach="REACH_EQ_WRITEWIN(3/3) / REACH_OFFSET(部分) / REACH_FAIL(0/3)（P4）",
        Q3_domain="WIN_IN_DOMAIN / WIN_OUT_OF_DOMAIN（P3）",
        Q4_centroid="CENTROID_SEPARATED(3/3) / CENTROID_PARTIAL(2/3) / CENTROID_OVERLAP(<=1/3)（P6）",
        Q5_redesign="STAT_REDESIGN_EFFECTIVE / STAT_REDESIGN_EQUIVALENT（P5）",
    ),

    honesty=[
        '本 Phase 的集中度结论一律在**可达域 REACH** 上声明；不可达位点的位数与 rho 必须逐臂列出。',
        'com_layer 是**位置**统计量，不是显著性替代品；任何「集中/分散」断言必须同报双边分位带与观测。',
        '设计期已阅读 Phase 15 **已发表**的 6..34 曲线（既有公开数据）；本 Phase 的新增观测只有 ell=1..5。'
        'P5/P6/P7 的判据在含新增位点的**可达域**上评估，判据本身在观测前冻结。',
        'ell_reach 依赖 rho 的 0.5 阈值；若浅端 rho 非单调，严格相等（P4）可能不成立 —— 此时必须报 rho 全曲线，'
        '并降级为「首次跨越 0.5 的位点」的区间叙述，不得事后调阈值。',
        '仍为激活级单点替换，不含权重级实现证明（继承 Phase 9/10/15 同一条限界）。',
        '三臂仍为 nf4 口径（bf16 对本机 16GB GPU 不可行，Phase 15 已实测否决）；A0 是量化保真校准臂，'
        '其「跨模型」身份不成立，只用于锚复现。',
        'span_k 与 com_layer 的零假设是「同一 jump 多重集的随机排列」；该零假设**不检验**「jump 多重集本身'
        '是否来自某个机制」，只检验**位置**是否非随机。',
    ],
    may_falsify_the_whole_line=('若三臂的 com_layer(xhalf) 与 com_layer(J) 不再分离（P6 失败），'
                                '则 Phase 12/13/14 的「双坐标集中窗」叙事在**物理深度轴**上不成立，'
                                '此前关于「xhalf 深尾集中 / J 浅端集中」的全部表述必须撤回，'
                                '并改以「两坐标仅为同一曲线的两种形状统计量」重述。'),
)

OUT = os.path.join(P16T, 'N2h1a9_design_seal.json')
b = json.dumps(SEAL, ensure_ascii=False, indent=1).encode('utf-8')
io.open(OUT, 'wb').write(b)
h = hashlib.sha256(b).hexdigest()
print('SEAL -> %s' % OUT)
print('bytes=%d sha256=%s sha8=%s' % (len(b), h, h[:8]))
print('profile_sites n=%d -> %s' % (len(PROFILE_NEW), PROFILE_NEW))
print('legacy n=%d' % len(PROFILE_LEGACY))
print('arms=%s' % EX15['arm_order'])
for a in EX15['arm_order']:
    print('  %-22s FULL_SWAP=%.6f L*=%s' % (a, ANCH[a]['FULL_SWAP'], ANCH[a]['L_star_own']))
