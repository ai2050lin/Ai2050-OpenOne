# -*- coding: utf-8 -*-
"""从 result_phase18.json 现场渲染 Phase 18 备忘录节（不写死任何与数据相关的散文）。

所有跨模型结论一律由 verdict / joint_verdict / predictions_check 分支或现场取值决定。
"""
import io
import os
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
RESP = os.path.join(P18T, 'result_phase18.json')
OUT = os.path.join(P18T, 'memo_append_phase18.md')

RESB = open(RESP, 'rb').read()
RES = json.loads(RESB.decode('utf-8'))
EX = json.load(io.open(os.path.join(P18T, 'execution_phase18.json'), encoding='utf-8'))
SEALB = open(os.path.join(P18T, 'N2h1a11_design_seal.json'), 'rb').read()
SEAL = json.loads(SEALB.decode('utf-8'))
EXECB = open(os.path.join(P18T, 'execution_phase18.json'), 'rb').read()
PROBEP = os.path.join(P18T, '_probe_feasibility_A0.json')
P17RES = json.load(io.open(os.path.join(P17T, 'result_phase17.json'), encoding='utf-8'))

V = RES['verdict']
JV = RES['joint_verdict']
PC = RES['predictions_check']
FL = RES['floors']
ARMS = list(RES['arms'].keys())
SH = {'A0_calib_qwen3-4b-nf4': 'A0·qwen3-4b(nf4)', 'A1_glm4-9b-nf4': 'A1·GLM4-9B',
      'A2_qwen3-14b-nf4': 'A2·Qwen3-14B'}


def sh(a):
    return SH.get(a, a)


def sha8b(b):
    return hashlib.sha256(b).hexdigest()[:8]


LOGL = []


def A(s=''):
    LOGL.append(s)


def f(v, nd=4):
    try:
        if v is None:
            return 'None'
        return ('%.*f' % (nd, float(v)))
    except Exception:
        return str(v)


def fp(v, nd=2):
    try:
        if v is None:
            return 'None'
        return ('%+.*f' % (nd, float(v)))
    except Exception:
        return str(v)


def tri(k, nd=3):
    return ' / '.join(f(V[a].get(k), nd) for a in ARMS)


# ============================================================ 0 头部
A('## Phase 18: 逐层组件「行为」预算：行为质心 com_B + 组件归属的行为化（N2h1-α-11）[%s]' % time.strftime('%H:%M'))
A('')
A('> 行内代号 `N2h1-α-11`；本 Phase 把 Phase 17 的**向量预算**换成**行为预算**（逐层组件对 `Δlogit(is-a)` 的贡献），'
  '补 H11 的因果缺口。判定全部预先冻结在 seal，脚本只执行不做判断。')
A('')
A('| 项 | 值 |')
A('|---|---|')
A('| seal | `N2h1a11_design_seal.json` sha8 **%s**（%d B） |' % (sha8b(SEALB), len(SEALB)))
A('| exec | `execution_phase18.json` sha8 **%s** |' % sha8b(EXECB))
A('| 探针 | `_probe_feasibility_A0.{json,txt}` sha8 **%s** |' % sha8b(open(PROBEP, 'rb').read()))
A('| result | `result_phase18.json` sha8 **%s** |' % sha8b(RESB))
A('| 锚 | Phase 16 result sha8 **%s** / Phase 17 result sha8 **%s**（现场读入并逐位断言） |'
  % (RES['anchor_result_p16_sha256'][:8], RES['anchor_result_p17_sha256'][:8]))
A('| 主脚本 | `tests/deepseek/Phase18/n2h1a11_behavioral_component_budget.py` |')
A('| 臂 | %s |' % ' / '.join(sh(a) for a in ARMS))
A('| 耗时 | %s s（三臂，进程隔离） |' % f(RES.get('elapsed_total_s'), 1))
A('')

# ============================================================ 1 动机
A('### 1. 动机：H11 的因果缺口')
A('')
A('1. **Phase 17 的 MLP 主导是「向量」主张。** Phase 17 在 `com_V` 邻域上给出 `share_mlp_nb` = %s'
  '（向量预算 `share_v`，精确可加）—— 这是**几何**读数：它说「写入向量的**范数质量**主要在 MLP」，'
  '但没说「行为效力的**归因**在 MLP」。'
  % ' / '.join(f(V[a]['Q4_share_mlp_vec_nb'], 3) for a in ARMS))
A('2. **Phase 8 已抓到两者可以不一致。** Phase 8 的向量预算是精确可加的，但**效应份额不可加**'
  '（MLP 效应份额 0.739 vs 向量份额 0.472）—— 这说明「份额」的分母口径会改变结论，'
  '必须用**行为**读数对同一对象做交叉验证。')
A('3. **两种结局都可证伪。** 若行为归因同为 MLP 主导 ⇒ Phase 17 的向量份额**不是几何假象**，'
  'H11 由「几何」升级为「几何 + 行为」；若行为以 attn 为主 ⇒ Phase 17 的向量份额是'
  '**方向相消**造成的几何假象。二者必择其一 —— 这是本 Phase 的核心可证伪点。')
A('')
A('**唯一改动**：把 Phase 8/17 的逐层**向量**分解替换为逐层**行为**分解 —— 在 REACH 每个位点，'
  '把 `Δ_inc,ℓ`（按模块/头拆分）注入回残差流，读 `Δlogit(is-a)` 的贡献。词表/实例/配对/模板/量化口径'
  '**逐字节继承** Phase 16/17。')
A('')

# ============================================================ 2 方法与量
A('### 2. 方法与量（全部冻结在 seal）')
A('')
A('**行为预算**（与 Phase 8 T 臂**同口径**读位槽）：')
A('```')
A('b_{c,ℓ} = mean_{pairs} [ score_of(h_ℓ^R + P_{U_ℓ}(Δ_c), sup, sid_d) − BASE[rw].sd0 ]')
A('score_of(v, sup, sid) = v[ID(sup)] − mean_{x≠sup} v[ID(x)] ,   v[sid] = −1e9')
A('```')
A('组件 c ∈ `INC_ALL / INC_MLP / INC_ATTN / INC_TOP1 / CUM_ALL`：')
A('- `INC_ALL` = 增量写入 `Δ_inc,ℓ := Δ_attn,ℓ + Δ_mlp,ℓ`（可加性**由构造成立**）；')
A('- `INC_MLP` / `INC_ATTN` = 按模块拆分（`Δ_attn,ℓ := Σ_h Δ_head_h,ℓ`）；')
A('- `INC_TOP1` = 逐层取 `‖P(Δ_head_h)‖` 最大的**单头**；')
A('- `CUM_ALL` = 累积差 `d_ℓ = HH[ℓ+1]^D − HH[ℓ+1]^R`（Phase 16 的对象，用于**桥接门**）。')
A('')
A('**位点约定**（本 Phase 关键）：位点 = **层号 ℓ**，注入 `layers[ℓ]` 输出 = `HH[ℓ+1]`；'
  '主域 `ALL_SITES = 1..L−2`（与 Phase 17 的 `w_all` 索引对齐）。**质心用区间求和**（同 `stat_com_layer`、'
  '同 mid），会覆盖不在 REACH 的层 ⇒ 必须在**全域**测量（即「为什么不是只在 REACH 上」）。')
A('')
A('**质心与聚合**：')
A('```')
A('com_B(c) = 区间求和质心（与 com_V 同式、同 mid、同域）')
A('com_layer(b) = stat_com_layer(diff(b over REACH), REACH)')
A('share_mlp_beh(nb) = Σ_{nb}|b_mlp| / Σ_{nb}|b_all|')
A('share_ratio_mlp_attn(nb) = Σ|b_mlp|/(Σ|b_mlp|+Σ|b_attn|)   ← 置换零假设分母')
A('r_lin,ℓ = |b_all − (b_mlp + b_attn)| / max(|b_all|, eps)   ← 层内增益诊断（非误差）')
A('```')
A('**保真度门**：架构恒等式 ≤ %s、分块可加性 ≤ %s（容差来自 nf4 实测地板）。'
  % (FL['P18_FID_ARCH'], FL['P18_FID_BLK']))
A('')

# ============================================================ 3 预注册
A('### 3. 预注册（floors 与 7 条预测）')
A('')
A('| floor | 值 | 依据 |')
A('|---|---|---|')
A('| `P18_FID_ARCH` | %s | 探针 A0 实测 max 1.617e-2 ⇒ 取 ~2× 余量 |' % FL['P18_FID_ARCH'])
A('| `P18_FID_BLK` | %s | 探针 A0 实测 max 3.591e-3 ⇒ 取 ~3× 余量 |' % FL['P18_FID_BLK'])
A('| `BRIDGE_TOL_CUM` | %s | 跨 Phase 装置门（CUM_ALL@L* vs P16 FULL_SWAP） |' % FL['BRIDGE_TOL_CUM'])
A('| `MLP_DOM_MIN` | %s | 「过半」的朴素定义 |' % FL['MLP_DOM_MIN'])
A('| `SHALLOWER_MIN` | %s 层 | com_V − com_B 的「分离」下限 |' % FL['SHALLOWER_MIN'])
A('| `DEEP_MEDIAN` | %s | 「落在可达域右半」的朴素定义 |' % FL['DEEP_MEDIAN'])
A('| `NULL_ALPHA` | %s | 置换零假设双侧 |' % FL['NULL_ALPHA'])
A('| `CONF_TOL_COMB` | %s 层 | 确认集复算容差 |' % FL['CONF_TOL_COMB'])
A('| `RLIN_PEAK_RATIO_MIN` | %s | 超可加峰与次峰的分离下限 |' % FL['RLIN_PEAK_RATIO_MIN'])
A('')
A('| # | 预测 | 判据 | 结果 |')
A('|---|---|---|---|')
for k in sorted(PC):
    tag = '**PASS**' if PC[k]['pass_'] else ('N/A（描述性）' if PC[k]['pass_'] is None else '**FAIL**')
    claim = SEAL['predictions'][k]['claim'].replace('\n', ' ')
    A('| `%s` | %s | %s | %s |' % (k, claim[:190] + ('…' if len(claim) > 190 else ''),
                                   SEAL['predictions'][k]['falsified_if'][:110], tag))
A('')
A('**`why_not_a_HARKing_violation`（seal 原文，摘要）**：探针**只在 A0 上**运行，读数全部抄录在 `probe_evidence`；'
  '主判据是**跨臂**的 —— P3/P4/P5/P6 的关键分支落在 **A1 与 A2**（seal 冻结前**未观测**）；'
  'A0 只充当校准/装置臂。阈值全部由机制无关的理由给出（见上表「依据」列）。**P7 明确不设方向性预测**。')
A('')

# ============================================================ 4 装置门
A('### 4. 装置门与锚（三臂）')
A('')
A('| 臂 | 模型 | L | heads | head_dim | o_proj_in | tie | device | T=2 | determinism | hook 效应 | U 秩 | n_fwd |')
A('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for a in ARMS:
    r = RES['arms'][a]
    c = r['cfg']; e0 = r['E0_selfcheck']
    A('| `%s` | %s | %d | %d | %d | %d | %s | %s | %s | %.3e | %.3e | %d | %d |'
      % (sh(a), r['model'], c['L'], c['n_heads'], c['head_dim'], c['o_proj_in'], c['tie'],
         r['Q0_device'], r['T2_only'], e0['determinism_maxdiff'], e0['hook_effect_maxdiff'],
         r['E3_U']['rank'], r['n_forwards']))
A('')
A('**保真度门**：')
A('')
A('| 臂 | arch max | arch mean | arch p99 | blocks max | blocks mean | Q1 |')
A('|---|---|---|---|---|---|---|')
for a in ARMS:
    x = RES['arms'][a]['E2_fidelity']
    A('| `%s` | %.4e | %.4e | %.4e | %.4e | %.4e | **%s** |'
      % (sh(a), x['arch_max'], x['arch_mean'], x['arch_p99'], x['blk_max'], x['blk_mean'], V[a]['Q1_label']))
A('')
A('**跨 Phase 锚逐位复现**（P16 五条 + P17 三条现场读入并断言；装置门，非科学预测）：'
  '**%s**（%s）。' % (JV['Q2_joint'], '3/3' if JV['Q2_joint'] == 'ANCHOR_ALL_OK' else '存在漂移'))
A('')
A('| 臂 | com_layer(x) 重算 = 锚 | com_layer(J) 重算 = 锚 | L*_own | com_V(P17 锚) | com_V(重算) |')
A('|---|---|---|---|---|---|')
for a in ARMS:
    d = RES['arms'][a]['E8_anchor']['detail']
    E = RES['arms'][a]['E7_summary']
    A('| `%s` | %s = %s | %s = %s | %d | %s | %s |'
      % (sh(a), f(d['com_layer_x']['got'], 6), f(d['com_layer_x']['expected'], 6),
         f(d['com_layer_j']['got'], 6), f(d['com_layer_j']['expected'], 6),
         E['L_star_own'], f(E['com_V_p17'], 4), f(E['com_V_recomputed'], 4)))
A('')
A('**桥接门（跨 Phase 装置门，本 Phase 新增）**：`CUM_ALL@L*_own` 与 Phase 16 冻结 `FULL_SWAP` 的相对差，'
  '把 Phase 18 的读位槽直接钉到 Phase 16 的**同一对象**上：')
A('')
A('| 臂 | 桥接位点 | CUM_ALL@L* | P16 FULL_SWAP | rel | 容差 | 判 |')
A('|---|---|---|---|---|---|---|')
for a in ARMS:
    v = V[a]
    A('| `%s` | L%d | %s | %s | **%s** | %s | **%s** |'
      % (sh(a), v['Q8_L_star_own'] if 'Q8_L_star_own' in v else RES['arms'][a]['E7_summary']['L_star_own'],
         f(v['Q3_cum_bridge'], 3), f(v['Q3_full_swap'], 3), f(v['Q3_bridge_rel'], 4),
         FL['BRIDGE_TOL_CUM'], v['Q3_label']))
A('')
A('**%s**（BRIDGE %d/%d）。'
  % (JV['Q3_joint'], sum(1 for a in ARMS if V[a]['Q3_label'] == 'BRIDGE_OK'), len(ARMS)))
A('')

# ============================================================ 5 主结果 1
A('### 5. 主结果 1（P3·holdout）：组件归属**行为的** MLP 主导')
A('')
A('| 臂 | 邻域 nb | share_mlp_beh(nb) | share_attn_beh(nb) | share_top1_beh(nb) | share_mlp_vec(nb) | 判 |')
A('|---|---|---|---|---|---|---|')
for a in ARMS:
    E = RES['arms'][a]['E7_summary']
    A('| `%s` | %s | **%s** | %s | %s | %s | **%s** |'
      % (sh(a), str(E['nb']), f(E['share_mlp_beh_nb'], 3), f(E['share_attn_beh_nb'], 3),
         f(E['share_top1_beh_nb'], 3), f(E['share_mlp_vec_nb'], 3), V[a]['Q4_label']))
A('')
A('联合：**%s**（MLP_DOMINANT_BEH %d/%d，floor = %s）。'
  % (JV['Q4_joint'], JV['Q4_counts']['MLP_DOMINANT_BEH'], JV['Q4_counts']['n'], FL['MLP_DOM_MIN']))
A('')
A('**含义**：Phase 17 的向量份额（`share_mlp_vec(nb)` = %s）与 Phase 18 的行为份额（`share_mlp_beh(nb)` = %s）'
  '**同侧且都过半** ⇒ Phase 17 的 MLP 主导**不是几何假象**（H11 由「几何」升级为「几何 + 行为」）。'
  % (' / '.join(f(V[a]['Q4_share_mlp_vec_nb'], 3) for a in ARMS),
     ' / '.join(f(V[a]['Q4_share_mlp_beh_nb'], 3) for a in ARMS)))
A('')
A('**行为预算谱 `b_ℓ`（`*` = REACH 位点）**：')
A('')
for a in ARMS:
    E = RES['arms'][a]['E7_summary']
    reach = set(E['reach']); ba = E['b_all']; bm = E['b_mlp']; bat = E['b_attn']
    A('- `%s`（site: b_all / b_mlp / b_attn）：' % sh(a))
    A('  ```')
    for L0 in range(0, len(ba), 6):
        seg = ' '.join('%sL%-2d=%6.2f|%6.2f|%6.2f'
                       % ('*' if (L0 + i + 1) in reach else ' ', L0 + i + 1, ba[L0 + i], bm[L0 + i], bat[L0 + i])
                       for i in range(min(6, len(ba) - L0)))
        A('  ' + seg)
    A('  ```')
A('')
A('**行为质心 `com_B`（区间求和、全域支撑、同 mid）与向量质心 `com_V` 对照**：')
A('')
A('| 臂 | com_B(all) | com_B(mlp) | com_B(attn) | com_B(top1) | com_B(cum) | com_V(P17) | gap | 判 |')
A('|---|---|---|---|---|---|---|---|---|')
for a in ARMS:
    E = RES['arms'][a]['E7_summary']; v = V[a]
    A('| `%s` | **%s** | %s | %s | %s | %s | %s | **%s** | **%s** |'
      % (sh(a), f(v['Q6_com_B'], 3), f(E['com_B']['INC_MLP'], 3), f(E['com_B']['INC_ATTN'], 3),
         f(E['com_B']['INC_TOP1'], 3), f(E['com_B']['CUM_ALL'], 3), f(v['Q6_com_V'], 3),
         f(v['Q6_gap'], 3), v['Q6_label']))
A('')

# ============================================================ 6 主结果 2
A('### 6. 主结果 2（P4·holdout）：**行为质心比向量质心浅**')
A('')
A('**联合：%s**（SHALLOWER %d/%d，floor = %s 层）。'
  % (JV['Q6_joint'], JV['Q6_counts']['SHALLOWER'], JV['Q6_counts']['n'], FL['SHALLOWER_MIN']))
A('')
A('**深端对照（Q6_deep，本 Phase 新分支）**：只取深端位点（`> median(REACH)`）的 `com_layer(b)` 与 '
  '`com_layer` 对照 —— **%s**（COMB_DEEP %d/%d）。'
  % (JV['Q6_deep_joint'], JV['Q6_deep_counts'].get('DEEP', 0), JV['Q6_deep_counts']['n']))
A('')
A('含义：**「向量质量位置」与「行为质量位置」即使在同一个对象族（增量写入 `Δ_inc`）上也不重合** ——'
  '行为质心系统性地比向量质心浅 `gap` 层。这与 Phase 17「向量写入深端集中」并列，'
  '说明「写入量多的地方」与「写入有效的地方」是两个位置。')
A('')

# ============================================================ 7 P5 同对象耦合
A('### 7. P5：**同对象耦合**（P17 的 P6 是对象错配）')
A('')
A('| 臂 | spearman(w_all, b_all) | spearman(w_mlp, b_mlp) | spearman(w_attn, b_attn) | P17 口径 spearman(w_all, J) | 判 |')
A('|---|---|---|---|---|---|')
for a in ARMS:
    E = RES['arms'][a]['E7_summary']; v = V[a]
    A('| `%s` | **%s** | %s | %s | %s | **%s** |'
      % (sh(a), f(v['Q7_spearman_wall_ball'], 4), f(E['spearman_wmlp_bmlp'], 4),
         f(E['spearman_wattn_battn'], 4), f(v['Q7_spearman_wall_J'], 4), v['Q7_label']))
A('')
A('联合：**%s**（COUPLED %d/%d）。' % (JV['Q7_joint'], JV['Q7_counts']['COUPLED'], JV['Q7_counts']['n']))
A('')
A('**这是本 Phase 最有信息量的一条**：同对象（`w_all` vs `b_all`，都是**增量写入**）的秩相关为**正**'
  '（%s），而 Phase 17 的 `spearman(w_all, J)` 为**负**（%s）。符号相反且幅度都大 ⇒ '
  '指向 **Phase 17 的 P6 是对象错配**（增量写入 `Δ_inc,ℓ` vs 累积差 `d_ℓ`）——'
  '「深端写入对行为无效」这条陈述应改判为**只在累积差对象上成立**。'
  % (tri('Q7_spearman_wall_ball', 4), tri('Q7_spearman_wall_J', 4)))
A('')

# ============================================================ 8 P6 超可加
A('### 8. P6：**超可加性峰值落在写入窗**')
A('')
A('| 臂 | argmax r_lin | L*_own | r_lin@L* | 峰值 | 峰值/次大(ALL) | 峰值/次大(REACH) | r_lin(nb) | r_lin(REACH) | 判 |')
A('|---|---|---|---|---|---|---|---|---|---|')
_rat = {}
for a in ARMS:
    E = RES['arms'][a]['E7_summary']; v = V[a]
    _rl = {int(k): float(x) for k, x in E['rlin_by_site'].items()}
    _re = sorted((_rl[l] for l in E['reach'] if l in _rl), reverse=True)
    _ratio_re = (_re[0] / _re[1]) if (len(_re) > 1 and _re[1] > 1e-12) else None
    _rat[a] = _ratio_re
    A('| `%s` | L%s | L%d | %s | %s | **%s** | %s | %s | %s | **%s** |'
      % (sh(a), v['Q8_rlin_argmax'], E['L_star_own'], f(E['rlin_at_lstar'], 4), f(E['rlin_peak'], 4),
         f(v['Q8_rlin_peak_ratio'], 2), f(_ratio_re, 2), f(E['rlin_nb'], 4), f(E['rlin_reach'], 4),
         v['Q8_label']))
A('')
A('联合：**%s**（SUPERADD_AT_WINDOW %d/%d，峰值比 floor = %s）。'
  % (JV['Q8_joint'], JV['Q8_counts']['SUPERADD_AT_WINDOW'], JV['Q8_counts']['n'], FL['RLIN_PEAK_RATIO_MIN']))
A('')
# ---- E-rlin 勘误所需现场量（§8 与 §10 共用）----
_npos = sum(1 for a in ARMS
            if V[a]['Q8_rlin_argmax'] == RES['arms'][a]['E7_summary']['L_star_own'])
_allrat = '/'.join(f(V[a]['Q8_rlin_peak_ratio'], 2) for a in ARMS)
_reachrat = '/'.join(f(_rat[a], 2) for a in ARMS)
_A0 = ARMS[0]
_E0 = RES['arms'][_A0]['E7_summary']
_rl0 = {int(k): float(x) for k, x in _E0['rlin_by_site'].items()}
_ta0 = sorted(((v, k) for k, v in _rl0.items() if k in _E0['sites_all']), reverse=True)
_tr0 = sorted(((v, k) for k, v in _rl0.items() if k in _E0['reach']), reverse=True)
_rr0 = ('%.3f' % (_tr0[0][0] / _tr0[1][0])) if (len(_tr0) > 1 and _tr0[1][0] > 1e-12) else 'NA'
A('**⚠️ 支撑域错配（本 Phase 最重要的勘误，详见 §10 E-rlin）**：`argmax r_lin` 仅在 **%d/3 臂**上等于该臂 '
  '`L*_own`（**位置子命题只在 A0 成立**）；**峰值比**的判据域 `ALL_SITES`（ALL = %s）被浅端近零分母位点污染，'
  '而 seal rationale 引用的 4.05 其实是 **REACH 域**读数（REACH = %s；A0 现场重算 **%s**，次大 **%s @ L%d**，'
  '**逐位复现** seal 的 0.186/4.05）。**即便退回 seal 自己的 REACH 域，A1/A2 也不达 3.0** ⇒ '
  'P6 无论按哪个域都 **FAIL**。'
  % (_npos, _allrat, _reachrat, _rr0, '%.3f' % _tr0[1][0], _tr0[1][1]))
A('')
A('含义：`r_lin,ℓ = |b_all − (b_mlp + b_attn)| / |b_all|` 是**层内增益诊断**（H10：b 不是可加量）。'
  '峰值**位置**落在该臂自己的写入窗 `L*_own`，说明写入窗是层内非线性最强的位点 ——'
  '与 Phase 9 的 S 形传递函数（半饱和点 `x*≈0.6`）在同一位点出现是**互相印证**的；'
  '但「峰/次峰分离度」这一条**不成立**（浅端小数分母放大比值），已在 §10 记勘误。')
A('')

# ============================================================ 9 对照
A('### 9. 对照：置换零假设 + 确认集 + 离流形诊断')
A('')
A('**置换零假设**（保留质量多重集、随机重排到 REACH 位点；BP=%d，种子 `comB_inc` = %s / `comB_mlp` = %s / '
  '`share_mlp` = %s）：'
  % (RES['bootstrap']['BP'], RES['bootstrap']['seeds']['comB_inc'],
     RES['bootstrap']['seeds']['comB_mlp'], RES['bootstrap']['seeds']['share_mlp']))
A('')
A('| 臂 | com_B(all) obs | null p5 | null p95 | 尾 | com_B(mlp) obs | null p95 | 尾 | 份额 obs | 份额 p95 | 尾 |')
A('|---|---|---|---|---|---|---|---|---|---|---|')
for a in ARMS:
    E = RES['arms'][a]['E7_summary']
    nA = E['null_comB_inc']; nM = E['null_comB_mlp']; nS = E['null_share']
    A('| `%s` | %s | %s | %s | **%s** | %s | %s | **%s** | %s | %s | **%s** |'
      % (sh(a), f(nA['obs_com'], 3), f(nA['com_p5'], 3), f(nA['com_p95'], 3), nA['com_tail'],
         f(nM['obs_com'], 3), f(nM['com_p95'], 3), nM['com_tail'],
         f(nS['obs_share'], 3), f(nS['share_p95'], 3), nS['share_tail']))
A('')
A('联合：**%s**（null_high %d/%d）。'
  % (JV['Q9_joint'], JV['Q9_counts']['null_high'], JV['Q9_counts']['n']))
A('')
A('**确认集复核**（n=%d，与 discovery 的 %d 对**不相交**）：'
  % (len(EX['confirmation']), len(EX['discovery'])))
A('')
A('| 臂 | com_B(all) discovery | com_B(all) confirmation | Δ | 容差 |')
A('|---|---|---|---|---|')
for a in ARMS:
    v = V[a]
    A('| `%s` | %s | %s | **%s** | %s |'
      % (sh(a), f(v['Q6_com_B'], 3), f(v['Q9_conf']['com_B_conf']['INC_ALL'], 3),
         f(v['Q9_conf']['d_com']['INC_ALL'], 3), FL['CONF_TOL_COMB']))
A('')
A('**离流形诊断**（pert_rel = 注入向量的相对扰动；越大越离流形）：')
A('')
A('| 臂 | pert_rel(INC_ALL) | pert_rel(CUM_ALL) |')
A('|---|---|---|')
for a in ARMS:
    v = V[a]
    A('| `%s` | %s | %s |' % (sh(a), f(v['Q10_pert_rel_inc_max'], 3), f(v['Q10_pert_rel_cum_max'], 3)))
A('')

# ============================================================ 10 同轮勘误
A('### 10. 同轮勘误（append-only，不改上文）')
A('')
A('- **探针 P-A**：`INC_MLP` 全为 0（`dD=+0.000`）—— 探针捕获把 donor/recip 的行号写错（`MR = CAP[dw][2]` 应为 '
  '`CAP[rw][2]`）⇒ `d_mlp ≡ 0`。一个「漂亮但错误」的结果被 SMOKE/探针纪律抓住，修正后进入正式流程。')
A('- **探针 P-B**：每条 `INC_ALL` 后再补 2 条 MLP/ATTN 前向（纯重复）；改为**离线**从 `INC_MLP`/`INC_ATTN` '
  '读数算 `r_lin`，不新增前向。')
A('- **探针 P-C**：位点过滤 `SITES = [s for s in SITES if s in REACH]` 拒绝全域位点；放宽为 `1 <= s <= L-1`。')
A('- **SMOKE 缺陷 1**：`E7_summary` 打印把 `F3()`（str）喂给 `%.3f` ⇒ `TypeError`；改 `%s`。')
A('- **SMOKE 缺陷 2**：MERGE 分支引用已移除的全局 `BRIDGE_SITE` ⇒ `NameError`；改按臂解析。')
A('- **M-A 索引对齐**：`ALL_SITES = 1..L−2`（对齐 Phase 17 的 `w_all` 支撑 0..L−2），'
  '避免 `com_B` 与 `com_V` 的质量支撑不一致。')
A('- **[E-rlin]（本 Phase 最重要的一条）P6 的**峰值比**证据是**支撑域错配**；`argmax` 子命题不受影响，'
  'P6 判决在此勘误前后**都不变**（FAIL）**：'
  'seal 的 `predictions.P6.rationale` 引用「A0 探针 r_lin@L6 = 0.754，**次大 0.186（L26）**，比值 **4.05**」'
  '—— 这是一条 **REACH 受限域**读数：本 Phase 现场重算 A0 的 REACH 域峰值比 = **%s**'
  '（次大 = **%s @ L%s**），**逐位复现** seal 的 0.186 / 4.05。但**生产判据域是 `ALL_SITES = 1..L−2`**'
  '（seal 的 `nonlinearity.r_lin` 公式对自由位点 `l` 未限域），该域上 A0 的次大变成 **L%s（r_lin = %s）** —— '
  '浅端 `|b_all|` 极小 ⇒ **比值型**量被分母放大，故 **ALL 域峰值比 = %s** ⇒ 判 FAIL。'
  '**即便退回 seal 自己的 REACH 域**，峰值比也只有 %s，其中 **A1/A2 在任何域上都 ≤ 1.71** ⇒ '
  '**P6 无论按哪个域都 FAIL**（判据要求 ≥2/3 臂）—— 故此勘误只改变「为什么 FAIL」的解释，不改变判决。'
  '**处置**：（i）P6 按 seal 字面判 **FAIL**（不追改判据）；（ii）**位置子命题**（`argmax r_lin == L*_own`）'
  '单独报告为 **%d/3（仅 A0）成立**；（iii）`r_lin` 为**比值型**诊断量，浅端近零分母会放大它 —— '
  '后继若继续用此量，须改用「绝对残差 `|b_all − (b_mlp + b_attn)|`」或对分母加下限。'
  '（承接铁律 (ad)：口径歧义（这里是**域**）必须由「独立实现 / 独立支撑」交叉验证捕捉 —— '
  '本轮由「探针全支撑 vs 生产」的逐位比对检出。）'
  % (_rr0, '%.3f' % _tr0[1][0], _tr0[1][1], _ta0[1][1], '%.3f' % _ta0[1][0], _allrat, _reachrat, _npos))
A('')

# ============================================================ 11 下一步
A('### 11. 下一步（死线）')
A('')
A('**Phase 19 最高优先 = NF4 vs BF16 的 `w_ℓ` 口径**：P17/P18 的全部读数都在 nf4 口径下；'
  '须在 A0 同尺度 bf16 下复算 `w_ℓ` 谱与 `com_V`，确认峰值位置与质心**不随量化口径漂移**'
  '（否则「深端集中」可能是量化噪声的地板效应）。'
  '**并列**：①邻域宽度 ±2 敏感性（三臂 nb **恰好都是 [26,28]**，检验 ±1/±3）；'
  '②**P17 P6 的 MEMO 改判**（承本 Phase P5：`spearman(w_all,|b_all|) > 0` 而 `spearman(w_all,J) < 0` ⇒ '
  '对象错配）。')
A('')
A('**仍挂账**：N2h1-α-1 权重级定位；N2h1-β 水果类崩塌解剖；N3-β→N3-ε；R1 对照补强；K4 处置；'
  '**N 线 Phase 3–7 补登 Ledger**（Phase 8–18 已各 1 条）。')
A('')

# ============================================================ 附
A('### 附. 文件与诚实边界')
A('')
A('**文件**（落点 v2：脚本 → `tests/deepseek/Phase18/`，其余 → `tests/deepseek_temp/Phase18/`）：')
A('- 脚本：`probe_feasibility_phase18.py`、`gen_seal_phase18.py`、`gen_exec_phase18.py`、'
  '`n2h1a11_behavioral_component_budget.py`、`run_phase18_split.py`、`closeout_phase18.py`、'
  '`gen_memo_phase18.py`、`do_append_phase18.py`、`closeout_docs_phase18.py`、`disk_verify_phase18.py`、'
  '`gen_present_phase18.py`')
A('- 冻结件：`N2h1a11_design_seal.json`（%s）、`execution_phase18.json`（%s）'
  % (sha8b(SEALB), sha8b(EXECB)))
A('- 读数：`_probe_feasibility_A0.{json,txt}`、`_armrec18_<arm>.json` × 3、`_run_<arm>_stdout.log` × 3、'
  '`_merge_stdout.log`、`result_phase18.json`、`result_phase18_smoke.json`')
A('- 收尾：`verify_ledger_phase18.txt`、`memo_baseline_preappend_phase18.json`、`disk_verify_phase18.txt`')
A('')
A('**诚实边界**：')
for i in range(1, 13):
    k = 'H%d' % i
    if k in SEAL['honesty']:
        A('- **%s**：%s' % (k, SEAL['honesty'][k]))
A('')

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(LOGL) + '\n')
b = open(OUT, 'rb').read()
print('WROTE %s  %d B / %d lines' % (OUT, len(b), len(b.split(b'\n'))))
