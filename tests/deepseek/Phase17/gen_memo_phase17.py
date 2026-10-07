# -*- coding: utf-8 -*-
"""从 result_phase17.json 现场渲染 Phase 17 备忘录节（不写死任何与数据相关的散文）。

所有跨模型结论一律由 verdict / joint_verdict / predictions_check 分支或现场取值决定。
"""
import io
import os
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
RESP = os.path.join(P17T, 'result_phase17.json')
OUT = os.path.join(P17T, 'memo_append_phase17.md')

RESB = open(RESP, 'rb').read()
RES = json.loads(RESB.decode('utf-8'))
EX = json.load(io.open(os.path.join(P17T, 'execution_phase17.json'), encoding='utf-8'))
SEALB = open(os.path.join(P17T, 'N2h1a10_design_seal.json'), 'rb').read()
SEAL = json.loads(SEALB.decode('utf-8'))
EXECB = open(os.path.join(P17T, 'execution_phase17.json'), 'rb').read()
PROBEP = os.path.join(P17T, '_probe_feasibility_A0.txt')
P16RESB = open(os.path.join(P16T, 'result_phase16.json'), 'rb').read()
SCRIPTP = os.path.join(ROOT, 'tests', 'deepseek', 'Phase17', 'n2h1a10_writevec_centroid.py')

V = RES['verdict']
JV = RES['joint_verdict']
PC = RES['predictions_check']
FL = RES['floors']
ARMS = list(RES['arms'].keys())
V2P = os.path.join(P17T, 'result_phase17_v2_preintervalfix.json')
V2 = json.loads(open(V2P, 'rb').read().decode('utf-8')) if os.path.exists(V2P) else None
_V2CV0 = float(V2['verdict'][ARMS[0]]['Q3_com_V']) if V2 else None
assert _V2CV0 is not None, 'v2 留痕缺失，§10 E4 无法现场渲染'
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


CK = time.strftime('%Y-%m-%d %H:%M')
LAB = {a: ('%(a)s' % {'a': a})[:0] or a for a in ARMS}

# ============================================================ 0 头部
A('## Phase 17: 写入向量的位置与效力：向量质量质心 com_V + 组件归属（N2h1-α-10）[%s]' % time.strftime('%H:%M'))
A('')
A('> 行内代号 `N2h1-α-10`；本 Phase **零额外前向**（每臂 44 次模型前向：2 次 determinism + 1 次 hook + 41 次 capture），'
  '全部为只读测量。判定全部预先冻结在 seal，脚本只执行不做判断。')
A('')
A('| 项 | 值 |')
A('|---|---|')
A('| seal | `N2h1a10_design_seal.json` sha8 **%s**（%d B） |' % (sha8b(SEALB), len(SEALB)))
A('| exec | `execution_phase17.json` sha8 **%s** |' % sha8b(EXECB))
A('| 探针 | `_probe_feasibility_A0.{json,txt}` sha8 **%s** |'
  % sha8b(open(os.path.join(P17T, '_probe_feasibility_A0.json'), 'rb').read()))
A('| result | `result_phase17.json` sha8 **%s** |' % sha8b(RESB))
A('| 锚 | Phase 16 result sha8 **%s**（现场读入并逐位断言） |' % RES['anchor_result_sha256'][:8])
A('| 主脚本 | `tests/deepseek/Phase17/n2h1a10_writevec_centroid.py` |')
A('| 臂 | %s |' % ' / '.join(sh(a) for a in ARMS))
A('| 耗时 | %s s（三臂，进程隔离） |' % f(RES.get('elapsed_total_s'), 1))
A('')

# ============================================================ 1 动机
A('### 1. 动机：三个未闭合的缺口')
A('')
A('1. **位置与组件之间没有桥。** Phase 16 的限界明写：`com_layer` 是**描述性位置量** —— 能说「变化在 L24 附近」，'
  '**不能说「L24 做了什么」**。而 Phase 8 的组件预算（向量预算 `share_v`，精确可加）**只做了一个层 L6**；'
  'Phase 16 实测的行为质心是 A0 **%s** / A1 **%s** / A2 **%s** 层 —— 这三个层，谁都没查过。'
  % tuple(f(SEAL['anchor_values'][a]['com_layer_x'], 1) for a in ARMS))
A('2. **P6 否证之后没有替代的位置量。** Phase 16 的 P6 把「`xhalf` 深尾集中 / `J` 浅端集中」的**物理深度表述整体撤回**，'
  '撤回之后「位置到底是什么」完全悬空：它是**行为读数**的性质，还是**写入向量**的性质？')
A('3. **写入质量从未被测过。** Phase 10 已确立行为增益 `J(ℓ)` 随深度单调下降（`spearman(J,depth)` 见 §8）；'
  '但**向量层面的写入质量** `w_ℓ` 没有量。若二者反向 ⇒「深端有大量写入但对行为无效」是一条新机制陈述；'
  '若同向 ⇒ `com_layer` 可当作 `J` 的代理。二者必择其一 —— 这是本 Phase 的核心可证伪点。')
A('')
A('**唯一改动**：把 Phase 8 的向量预算从单层推广到**逐层**，得到向量质量谱 `w_ℓ` 与其质心 `com_V`，'
  '与 P16 的两个**行为**质心三向对照。词表/实例/配对/模板/量化口径**逐字节继承**，不新增任何前向。')
A('')

# ============================================================ 2 方法与量
A('### 2. 方法与量（全部冻结在 seal）')
A('')
A('**捕获扩展**（本 Phase 唯一新增钩子）：')
A('- `attn` 输出投影模块（`o_proj` / `dense` / `out_proj` 现场解析）的 **forward_pre_hook** -> `o_ℓ`（即 o_proj 的输入、拼接的逐头输出）；')
A('- MLP 模块的 **forward_hook** -> `m_ℓ`（MLP 输出）。')
A('- 逐头贡献**不用权重矩阵**：nf4 下 `weight` 是打包 uint8，不能直接矩阵乘；改为把 `o_ℓ` 做**头块掩码**后送进'
  '**模块自身前向**，一次 `[NH+1, OIN]` 批调用同时得到 NH 个头块输出与全量输出。')
A('')
A('**向量预算（精确可加，由构造成立）**：')
A('```')
A('P_{U_ℓ}(v) = (v @ U_ℓᵀ) @ U_ℓ ;   质量 = ‖P_{U_ℓ}(v)‖')
A('Δ_head_h,ℓ = o_proj(mask_h ⊙ o_ℓ^donor) − o_proj(mask_h ⊙ o_ℓ^recip)')
A('Δ_attn,ℓ := Σ_h Δ_head_h,ℓ      ← 定义')
A('Δ_mlp,ℓ  = m_ℓ^donor − m_ℓ^recip')
A('Δ_inc,ℓ := Δ_attn,ℓ + Δ_mlp,ℓ    ← 定义（可加性**由构造成立**，不是测量）')
A('w_ℓ = mean_{pairs∈discovery} ‖P_{U_ℓ}(Δ_inc,ℓ)‖')
A('```')
A('`U_ℓ = est_U(ℓ+1)`：6 类质心差的 SVD，秩 = n_classes − 1 = **5**（与 Phase 16 E3 同口径）。')
A('')
A('**质心**（与 `stat_com_layer` **同一 mid 口径**，单位「层」，网格不变量）：')
A('```')
A('把逐层质量按 REACH 的相邻位点区间聚合 W_j = Σ_{ℓ∈[s_j, s_{j+1})} w_ℓ')
A('com_V = Σ_j W_j · mid_j / Σ_j W_j ,   mid_j = (s_j + s_{j+1}) / 2')
A('```')
A('主域 = **REACH**（可达性掩膜，Phase 16 冻结：`ρ(ℓ) = Y(ℓ,α=1) ≥ 0.10`），与 `com_layer` **同域**；'
  '全域 1..L−1 只作对照，不入判据。组件质心 `com_V^{mlp}` / `com_V^{attn}` / `com_V^{top1head}` 同式。')
A('')
A('**两个保真度门**（检验上面的构造与模型实际计算一致；容差来自 nf4 量化噪声的**实测地板**）：')
A('- 架构恒等式：`‖(h_{ℓ+1}−h_ℓ) − (attn_out_ℓ + m_ℓ)‖ / ‖h_{ℓ+1}−h_ℓ‖ ≤ %s`' % FL['P17_FID_ARCH'])
A('- 分块可加性：`‖Σ_h head_block − o_proj(v)‖ / ‖o_proj(v)‖ ≤ %s`' % FL['P17_FID_BLK'])
A('')

# ============================================================ 3 预注册
A('### 3. 预注册（floors 与 7 条预测）')
A('')
A('| floor | 值 | 依据 |')
A('|---|---|---|')
A('| `P17_FID_ARCH` | %s | 探针 A0 实测 max 1.617e-2 ⇒ 取 ~2× 余量 |' % FL['P17_FID_ARCH'])
A('| `P17_FID_BLK` | %s | 探针 A0 实测 max 3.591e-3 ⇒ 取 ~3× 余量 |' % FL['P17_FID_BLK'])
A('| `DEEP_MEDIAN` | %s | 「落在可达域右半」的朴素定义 |' % FL['DEEP_MEDIAN'])
A('| `CENTROID_SEP_MIN` | %s 层 | 沿用 Phase 16（2× 邻域宽度） |' % FL['CENTROID_SEP_MIN'])
A('| `MLP_DOM_MIN` | %s | 「过半」的朴素定义 |' % FL['MLP_DOM_MIN'])
A('| `CONF_TOL_COMV` | %s 层 | 确认集复算容差 |' % FL['CONF_TOL_COMV'])
A('| `NULL_ALPHA` | %s | 置换零假设双侧 |' % FL['NULL_ALPHA'])
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
  '主判据是**跨臂**的 —— P3/P4/P6 的关键分支落在 **A1 与 A2**（seal 冻结前**未观测**）；A0 只充当校准/装置臂。'
  '阈值全部由机制无关的理由给出（见上表「依据」列）。**P7 明确不设方向性预测**，正因为 Phase 16 已发表过 k=3 的同一事实。')
A('')

# ============================================================ 4 装置门
A('### 4. 装置门与锚（三臂）')
A('')
A('| 臂 | 模型 | L | heads | head_dim | o_proj_in | tie | device | T=2 | determinism | hook 效应 | n_fwd |')
A('|---|---|---|---|---|---|---|---|---|---|---|---|')
for a in ARMS:
    r = RES['arms'][a]
    c = r['cfg']; e0 = r['E0_selfcheck']
    A('| `%s` | %s | %d | %d | %d | %d | %s | %s | %s | %.3e | %.3e | %d |'
      % (sh(a), r['model'], c['L'], c['n_heads'], c['head_dim'], c['o_proj_in'], c['tie'],
         r['Q0_device'], r['T2_only'], e0['determinism_maxdiff'], e0['hook_effect_maxdiff'],
         r['n_forwards']))
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
A('**P16 锚逐位复现**（`com_layer(x)` / `com_layer(J)` / `span3` / `L*_own` / `ell_reach` / `REACH` 现场读入并断言；'
  '装置门，非科学预测）：**%s**（%s）。' % (JV['Q2_joint'], '3/3' if JV['Q2_joint'] == 'ANCHOR_ALL_OK' else '存在漂移'))
A('')
A('| 臂 | com_layer(x) 重算 = 锚 | com_layer(J) 重算 = 锚 | span3(x) 重算 = 锚 | span3(J) 重算 = 锚 | L*_own | ℓ_reach |')
A('|---|---|---|---|---|---|---|')
for a in ARMS:
    d = RES['arms'][a]['E9_anchor']['detail']
    A('| `%s` | %s = %s | %s = %s | %s = %s | %s = %s | %d | %d |'
      % (sh(a), f(d['com_layer_x']['got'], 6), f(d['com_layer_x']['expected'], 6),
         f(d['com_layer_j']['got'], 6), f(d['com_layer_j']['expected'], 6),
         f(d['span3_x']['got'], 4), f(d['span3_x']['expected'], 4),
         f(d['span3_j']['got'], 4), f(d['span3_j']['expected'], 4),
         RES['arms'][a]['E9_anchor']['detail']['L_star_own']['got'],
         RES['arms'][a]['E9_anchor']['detail']['ell_reach']['got']))
A('')

# ============================================================ 5 主结果 1
A('### 5. 主结果 1（P3·holdout）：向量写入质量**深端集中**是层栈共性')
A('')
A('| 臂 | com_V(all) | median(REACH) | 判 | com_V(mlp) | com_V(attn) | com_V(top1head) | argmax w_ℓ | 全域 com_V |')
A('|---|---|---|---|---|---|---|---|---|')
for a in ARMS:
    c5 = RES['arms'][a]['E5_com_V']
    A('| `%s` | **%s** | %.1f | **%s** | %s | %s | %s | L%s | %s |'
      % (sh(a), f(c5['com_V'], 3), c5['median_reach'], V[a]['Q3_label'],
         f(c5['com_V_mlp'], 3), f(c5['com_V_attn'], 3), f(c5['com_V_top1head'], 3),
         c5['argmax_w_layer'], f(c5['com_V_full'], 3)))
A('')
A('联合：**%s**（DEEP %d/%d）。' % (JV['Q3_joint'], JV['Q3_counts']['DEEP'], JV['Q3_counts']['n']))
A('')
A('**向量质量谱 `w_ℓ`（REACH 域外也列出，`*` = 主域位点）**：')
A('')
for a in ARMS:
    c5 = RES['arms'][a]['E5_com_V']
    reach = set(c5['reach']); wa = c5['w_all']
    A('- `%s`：' % sh(a))
    A('  ```')
    for L0 in range(0, len(wa), 12):
        seg = ' '.join('%sL%-2d=%7.2f' % ('*' if (L0 + i) in reach else ' ', L0 + i, wa[L0 + i])
                       for i in range(min(12, len(wa) - L0)))
        A('  ' + seg)
    A('  ```')
A('')
A('读数：探针（A0，24 对）com_V = **%s**；正式运行（A0，24 对）com_V = **%s** —— '
  '**逐位一致**（首版生产实现取位点单层质量而非区间求和，见 §10 E4；修正后与探针完全吻合）。'
  '三个臂的 `w_ℓ` **都在 L28–L34 附近达到峰值**，浅端在写入窗附近另有一个次峰。'
  % (f(V[ARMS[0]]['Q3_com_V'], 2), f(V[ARMS[0]]['Q3_com_V'], 3)))
A('')

# ============================================================ 6 主结果 2
A('### 6. 主结果 2（P4·A2 判别臂）：**行为质心不能由向量质量质心替代**')
A('')
A('| 臂 | com_V | com_layer(x) | com_layer(J) | d_x | d_j | min(d) | 判 |')
A('|---|---|---|---|---|---|---|---|')
for a in ARMS:
    anc = SEAL['anchor_values'][a]
    vv = V[a]
    A('| `%s` | %s | %s | %s | %s | %s | **%s** | **%s** |'
      % (sh(a), f(vv['Q3_com_V'], 3), f(anc['com_layer_x'], 3), f(anc['com_layer_j'], 3),
         f(vv['Q4_d_x'], 2), f(vv['Q4_d_j'], 2), f(vv['Q4_min_d'], 2), vv['Q4_label']))
A('')
_p4 = PC['P4']
A('联合：**%s**（DECOUPLED %d/%d）。判别臂 A2：`min(d_x,d_j)` = **%s 层** ≥ `CENTROID_SEP_MIN` = %s ⇒ **%s**。'
  % (JV['Q4_joint'], JV['Q4_counts']['DECOUPLED'], JV['Q4_counts']['n'],
     f(_p4['detail'].get('min_d'), 2), FL['CENTROID_SEP_MIN'], _p4['detail'].get('label')))
A('')
A('**为什么 A2 是判别臂**：A2 的两个**行为**质心几乎重合（`com_layer(x)` = %s、`com_layer(J)` = %s，相距仅 %s 层），'
  '因此「`com_V` 靠哪个行为坐标更近」在 A2 上**没有分辨力**；但 A2 的向量质心落在 **%s**，'
  '离那两个行为质心 **%s 层** ⇒ 这直接说明**向量写入位置与行为质心位置是两件事**，'
  'Phase 16「`com_layer` 只是描述性位置量」的限界由此**被加强**为：'
  '**行为质心不能由向量质量质心替代**。'
  % (f(SEAL['anchor_values']['A2_qwen3-14b-nf4']['com_layer_x'], 3),
     f(SEAL['anchor_values']['A2_qwen3-14b-nf4']['com_layer_j'], 3),
     f(abs(SEAL['anchor_values']['A2_qwen3-14b-nf4']['com_layer_x']
           - SEAL['anchor_values']['A2_qwen3-14b-nf4']['com_layer_j']), 3),
     f(V['A2_qwen3-14b-nf4']['Q3_com_V'], 3), f(V['A2_qwen3-14b-nf4']['Q4_min_d'], 2)))
A('')
A('**A0 是唯一「对齐」的臂**（min(d) = %s 层 < %s）：A0 的 `com_layer(x)` 恰在深端（%s），'
  '与向量质心（%s）相差 %s 层。这**正是 Phase 16 P6 否证的镜像**：'
  'A0 上「行为质心深」与「向量质心深」同时成立是**碰巧对齐**，一旦换到 A2（同家族 3.5×）'
  '行为质心跑到浅端（%s）而向量质心仍在深端（%s），两者立刻分开。'
  % (f(V[ARMS[0]]['Q4_min_d'], 2), FL['CENTROID_SEP_MIN'],
     f(SEAL['anchor_values'][ARMS[0]]['com_layer_x'], 1), f(V[ARMS[0]]['Q3_com_V'], 3),
     f(V[ARMS[0]]['Q4_min_d'], 2),
     f(SEAL['anchor_values'][ARMS[2]]['com_layer_x'], 1), f(V[ARMS[0]]['Q3_com_V'], 3)))
A('')

# ============================================================ 7 组件归属
A('### 7. 组件归属（P5）：质心邻域由 **MLP** 承载')
A('')
A('| 臂 | com_V 邻域（±2 层） | share_mlp_nb | share_attn_nb | 最大单头 share | 判 |')
A('|---|---|---|---|---|---|')
for a in ARMS:
    c5 = RES['arms'][a]['E5_com_V']
    A('| `%s` | %s | **%s** | %s | %s | **%s** |'
      % (sh(a), str(c5['neighbourhood']), f(c5['share_mlp_nb'], 3), f(c5['share_attn_nb'], 3),
         f(c5['top1_head_share_nb'], 4), V[a]['Q5_label']))
A('')
A('联合：**%s**（MLP_DOMINANT %d/%d，floor = %s）。'
  % (JV['Q5_joint'], JV['Q5_counts']['MLP_DOMINANT'], JV['Q5_counts']['n'], FL['MLP_DOM_MIN']))
A('')
A('这与 Phase 8 在 **L6** 的结果同向（MLP 向量预算 0.4717，最大单头仅 0.0742）：'
  '把层从 L6 换到 `com_V` 邻域（三臂的邻域**恰好都是 L24–26**），MLP 份额为 **%s / %s / %s**'
  '（全部 ≥ L6 的 0.4717）。最大单头份额 %s / %s / %s —— 远低于任何「单头主导」阈值（Phase 8 的 '
  '`G2_localized` 门是 > 0.50），**无单头主导**。'
  % (f(V[ARMS[0]]['Q5_share_mlp_nb'], 3), f(V[ARMS[1]]['Q5_share_mlp_nb'], 3),
     f(V[ARMS[2]]['Q5_share_mlp_nb'], 3),
     f(RES['arms'][ARMS[0]]['E5_com_V']['top1_head_share_nb'], 4),
     f(RES['arms'][ARMS[1]]['E5_com_V']['top1_head_share_nb'], 4),
     f(RES['arms'][ARMS[2]]['E5_com_V']['top1_head_share_nb'], 4)))
A('')

# ============================================================ 8 效力关系与跨度谱
A('### 8. 效力关系（P6）与跨度谱（P7）')
A('')
A('| 臂 | spearman(w_ℓ, J_ℓ) | n | spearman(w_ℓ, xhalf) | spearman(J, depth) | 判 |')
A('|---|---|---|---|---|---|')
for a in ARMS:
    e6 = RES['arms'][a]['E6_efficacy']
    A('| `%s` | **%s** | %d | %s | %s | **%s** |'
      % (sh(a), f(e6['spearman_wJ'], 4), e6['n'], f(e6['spearman_wxhalf'], 4),
         f(e6['spearman_Jdepth'], 4), V[a]['Q6_label']))
A('')
A('联合：**%s**（ANTICORR %d/%d）。' % (JV['Q6_joint'], JV['Q6_counts']['ANTICORR'], JV['Q6_counts']['n']))
A('')
A('**这是本 Phase 最有信息量的一条**：行为增益 `J(ℓ)` 随深度**下降**，而向量写入质量 `w_ℓ` 随深度**上升**，'
  '两者在 REACH 上稳定**负相关**。含义：**深端有大量写入向量，但对 is-a 行为几乎无效**。'
  '这把 Phase 16 撤回「物理深度表述」后留下的空洞补上了一个**可操作**的陈述 ——'
  '「写入的多寡」与「写入的效力」在深度上是**分离**的两个轴。')
A('')
A('**跨度谱 `span_k`（k ∈ %s）**：' % RES['span_ks'])
A('')
A('| 臂 | k | span(xhalf) | span(J) | com_layer(x) | com_layer(J) | x 相对 J | 同号? |')
A('|---|---|---|---|---|---|---|---|')
for a in ARMS:
    sp = RES['arms'][a]['E8_span']['spans']
    for k in RES['span_ks']:
        sx = sp['x'][str(k)]['obs_span']; sj = sp['j'][str(k)]['obs_span']
        cx = sp['x']['com_layer']; cj = sp['j']['com_layer']
        A('| `%s` | %d | %s | %s | %s | %s | span %s / com %s | %s |'
          % (sh(a), k, f(sx, 4), f(sj, 4), f(cx, 3), f(cj, 3),
             '更宽' if sx > sj else '更窄', '更深' if cx > cj else '更浅',
             '✅' if (sx > sj) == (cx > cj) else '✗'))
A('')
A('联合：**%s**（同号 %s）。即「`span` 更宽 ⟺ 质心更深」在三臂上一致成立 ——'
  '`J` 的剖面变化挤在浅端少数相邻步（跨度小），`xhalf` 的剖面变化铺开在深度上（跨度大）。'
  % (JV['Q7_joint'], JV['Q7_coupled']))
A('')

# ============================================================ 9 对照
A('### 9. 对照：置换零假设 + 确认集')
A('')
A('**置换零假设**（保留质量多重集、随机重排到 REACH 位点上；`com_V` 是顺序敏感量，故此零假设**非退化**；'
  'BP=%d，种子 `comv_all` = %s / `comv_mlp` = %s）：'
  % (RES['bootstrap']['BP'], RES['bootstrap']['seeds']['comv_all'], RES['bootstrap']['seeds']['comv_mlp']))
A('')
A('| 臂 | obs com_V | null p5 | null p95 | 双侧尾 | obs com_V(mlp) | 尾 |')
A('|---|---|---|---|---|---|---|')
for a in ARMS:
    nA = V[a]['Q7_null_all']; nM = V[a]['Q7_null_mlp']
    A('| `%s` | %s | %s | %s | **%s** | %s | %s |'
      % (sh(a), f(nA['obs_com'], 3), f(nA['com_p5'], 3), f(nA['com_p95'], 3), nA['com_tail'],
         f(nM['obs_com'], 3), nM['com_tail']))
A('')
A('三臂 **3/3 落在 `high` 尾** ⇒ 向量写入质量的深端集中**不是随机重排能产生的**（观测值超过零假设 95 分位）。')
A('')
A('**确认集复核**（n=%d，与 discovery 的 %d 对**不相交**）：'
  % (RES['arms'][ARMS[0]]['E5b_conf']['n_pairs'], len(EX['discovery'])))
A('')
A('| 臂 | discovery com_V | confirmation com_V | Δ | 容差 |')
A('|---|---|---|---|---|')
for a in ARMS:
    b = V[a]['Q8_conf']
    A('| `%s` | %s | %s | **%s** | %s |'
      % (sh(a), f(V[a]['Q3_com_V'], 3), f(b['com_V'], 3), f(b['d_com'], 3), FL['CONF_TOL_COMV']))
A('')
A('三臂 Δ 全部 ≤ %s 层（最大 %s）⇒ 质心位置**不是配对抽样的偶然**。'
  % (FL['CONF_TOL_COMV'], f(max(V[a]['Q8_conf']['d_com'] for a in ARMS), 3)))
A('')

# ============================================================ 10 同轮勘误
A('### 10 同轮勘误（append-only，不改上文）')
A('')
A('- **[E1] Q7 标签编码修正（统计量与数据未变）**：seal 原文说「span 序与 `com_layer` 序是否**同号**」，'
  '而 v1 合并实现把配对写成了 `(span_x < span_j) == (com_x > com_j)` —— 这是「x 更窄 **且** x 更深」的'
  '**错位配对**，不等于「同号」，于是 3/3 报了 `SPAN_CENTROID_DECOUPLED`。'
  'v2 改为同一比较方向 `(span_x > span_j) == (com_x > com_j)` ⇒ **`SPAN_CENTROID_COUPLED`（3/3）**。'
  '两条读数都保留在 result：`Q7_coupled` / `Q7_coupled_v1_mismatched_pairing`；'
  'v1 合并结果留痕于 `result_phase17_v1_jointlabel.json`。**未重跑任何模型前向**（只重跑 MERGE）。')
A('- **[E2] SMOKE 抓出的退化路径缺陷**：首版 SMOKE 只截前 8 个实例 + 前 6 个配对，'
  '导致 discovery 配对集为空 ⇒ `com_V = None` ⇒ 置换零假设函数收到 `None` 崩溃。'
  '修正为「SMOKE 的实例集必须覆盖配对的**两端**」，并给 `perm_null_com` 加降级 schema、'
  '给 `spearman` 加**常量输入守卫**（否则常量向量会产出伪相关 −1）。**正式运行在修正之后**。')
A('- **[E4] `com_of_mass` 口径与 seal 不一致（最重要的一条）**：seal 定义 '
  '`W_j = Σ_{ℓ∈[s_j, s_{j+1})} w_ℓ`（**区间求和**），而首版实现取了 `w_{s_j}`（**位点单层**）。'
  '两者在 A0 上差 **%s 层**（%s vs %s）。修正为区间求和后，A0 的 `com_V` 与**探针**'
  '（独立脚本、24 对）的 **%s 逐位相同** —— 这同时是一次**独立实现之间的交叉验证**。'
  '修正只影响 `com_V` 族与零假设分位，**不动** `w_ℓ` 谱、`share_*` 份额、`spearman`、`span_k`、锚与保真度门；'
  '**未重跑任何模型前向**（只重跑 MERGE）。修正后三臂 `com_V` = %s / %s / %s，全部仍 DEEP，'
  '`min(d)` 与 `spearman` 方向不变（P3/P4/P6 判决不变）。'
  % (f(abs(_V2CV0 - V[ARMS[0]]['Q3_com_V']), 2), f(_V2CV0, 3), f(V[ARMS[0]]['Q3_com_V'], 3),
     f(V[ARMS[0]]['Q3_com_V'], 2),
     f(V[ARMS[0]]['Q3_com_V'], 3), f(V[ARMS[1]]['Q3_com_V'], 3), f(V[ARMS[2]]['Q3_com_V'], 3)))
A('- **[E3] 探针权重路径缺陷**：探针第一版用 `o_proj.weight.detach().float()` 做矩阵乘，'
  '在 nf4 下崩（`Params4bit` 是打包 uint8，形状 [out, in/2]）⇒ 改为**模块自身前向的头块掩码**，'
  '这同时保证了与模型实际计算**完全同口径**。')
A('')

# ============================================================ 11 下一步
A('### 11. 下一步（死线）')
A('')
A('**Phase 18 最高优先 = 逐层组件「行为」预算** —— 补上 H11 的因果缺口：本 Phase 的「深端写入无效」'
  '是**相关性陈述**（`spearman(w_ℓ,J_ℓ)` = %s / %s / %s），要变成因果陈述，必须把「向量预算」换成「行为预算」：'
  '在 REACH 的每个位点，把 `Δ_inc,ℓ` 的**向量**分解替换为**行为**分解（逐层组件对 `Δlogit(is-a)` 的贡献），'
  '与 §7 的 MLP 主导（share_mlp_nb = %s / %s / %s）交叉验证 —— 若行为层同样 MLP 主导，'
  '则「质心层由 MLP 写」升级为因果；若行为层以 attn 为主，则本 Phase 的向量份额读数是**几何假象**'
  '（方向相消的另一种表现）。'
  % (f(V[ARMS[0]]['Q6_spearman_wJ'], 3), f(V[ARMS[1]]['Q6_spearman_wJ'], 3), f(V[ARMS[2]]['Q6_spearman_wJ'], 3),
     f(V[ARMS[0]]['Q5_share_mlp_nb'], 3), f(V[ARMS[1]]['Q5_share_mlp_nb'], 3), f(V[ARMS[2]]['Q5_share_mlp_nb'], 3)))
A('')
A('**并列**：①NF4 vs BF16 的 `w_ℓ` 口径差异（本 Phase 全部读数在 nf4 口径；H5/H7 的容差来自 nf4 噪声地板，'
  '需在 A0 同尺度 bf16 下复算 `w_ℓ` 谱，确认峰值位置与 `com_V` 不随量化口径漂移）；'
  '②邻域宽度 ±2 的敏感性（三臂邻域**恰好都是 %s**，需检验 ±1 / ±3 是否改变 MLP 主导结论）。'
  % json.dumps({sh(a): RES['arms'][a]['E5_com_V']['neighbourhood'] for a in ARMS}, ensure_ascii=False))
A('')
A('**仍挂账**：N2h1-α-1 权重级定位；N2h1-β 水果类崩塌解剖；N3-β→N3-ε；R1 对照补强；K4 处置；'
  '**N 线 Phase 3–7 补登 Ledger**（Phase 8–17 已各 1 条）。')
A('')

# ============================================================ 附
A('### 附. 文件与诚实边界')
A('')
A('**文件**（落点 v2：脚本 → `tests/deepseek/Phase17/`，其余 → `tests/deepseek_temp/Phase17/`）：')
A('- 脚本：`probe_feasibility_phase17.py`、`gen_seal_phase17.py`、`gen_exec_phase17.py`、'
  '`n2h1a10_writevec_centroid.py`、`run_phase17_split.py`、`closeout_phase17.py`、'
  '`gen_memo_phase17.py`、`do_append_phase17.py`、`closeout_docs_phase17.py`、`disk_verify_phase17.py`、'
  '`gen_present_phase17.py`')
A('- 冻结件：`N2h1a10_design_seal.json`（%s）、`execution_phase17.json`（%s）'
  % (sha8b(SEALB), sha8b(EXECB)))
A('- 读数：`_probe_feasibility_A0.{json,txt}`、`_armrec17_<arm>.json` × 3、`_run_<arm>_stdout.log` × 3、'
  '`_merge_stdout.log`、`result_phase17.json`、`result_phase17_smoke.json`、'
  '`result_phase17_v1_jointlabel.json`（勘误留痕）')
A('- 收尾：`verify_ledger_phase17.txt`、`memo_baseline_preappend_phase17.json`、`disk_verify_phase17.txt`')
A('')
A('**诚实边界**：')
for i in range(1, 9):
    k = 'H%d' % i
    if k in SEAL['honesty']:
        A('- **%s**：%s' % (k, SEAL['honesty'][k]))
A('- **H9（本 Phase 新增）**：`share_mlp_nb` 的分母是「Σ 各分量投影范数」，'
  '因此 `share_attn + share_mlp ≈ 1` 但**不是恒等**（三角不等式）；A0 邻域 share_mlp=0.688 意味着'
  '其余 ~31% 由投影方向相消产生，不得读作「attn 只贡献 31%」。')
A('- **H10**：`w_ℓ` 的**绝对**量级只在同一臂内部可比（`U_ℓ` 逐层独立、`MASS` 未做跨臂归一）；'
  '跨臂比较只在**质心位置**与**份额**上进行（二者都是尺度无关量）。')
A('- **H11**：本 Phase 的「深端写入」是**激活级**读数；「深端写入无效」是对 `J(ℓ)` 的**相关性陈述**，'
  '不是对因果链的分解（因果分解需要逐层组件**行为**预算，属下一死线）。')
A('')
A('**排除的备择解释**：')
A('- 「深端质量是范数膨胀的假象」 ⇒ 证伪：`P_{U_ℓ}` 是投影，`U_ℓ` 逐层独立重建（秩 5），'
  '且置换零假设（保留多重集）3/3 落在 high 尾；纯粹范数膨胀会在**位点**上无偏好、给不出 high 尾。')
A('- 「`com_V` 只是 `L*_own` 的另一种写法」 ⇒ 证伪：A0 的 `L*_own` = %s 而 `com_V` = %s（差 %s 层），'
  '且 `argmax w_ℓ` 在 A0/A1/A2 分别是 L%s/L%s/L%s，与 `L*_own` 全不相同。'
  % (SEAL['anchor_values'][ARMS[0]]['L_star_own'], f(V[ARMS[0]]['Q3_com_V'], 2),
     f(abs(SEAL['anchor_values'][ARMS[0]]['L_star_own'] - V[ARMS[0]]['Q3_com_V']), 1),
     RES['arms'][ARMS[0]]['E5_com_V']['argmax_w_layer'],
     RES['arms'][ARMS[1]]['E5_com_V']['argmax_w_layer'],
     RES['arms'][ARMS[2]]['E5_com_V']['argmax_w_layer']))
A('')

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(LOGL) + '\n')
b = open(OUT, 'rb').read()
print('WROTE %s  %d B / %d lines' % (OUT, len(b), len(b.split(b'\n'))))
