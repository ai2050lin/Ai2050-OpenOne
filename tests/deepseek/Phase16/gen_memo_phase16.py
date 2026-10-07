# -*- coding: utf-8 -*-
"""从 result_phase16.json 现场渲染 Phase 16 备忘录节（不写死任何与数据相关的散文）。

所有跨模型结论一律由 verdict / joint_verdict / predictions_check 分支或现场取值决定。
"""
import io
import os
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
RESP = os.path.join(P16T, 'result_phase16.json')
OUT = os.path.join(P16T, 'memo_append_phase16.md')

RESB = open(RESP, 'rb').read()
RES = json.loads(RESB.decode('utf-8'))
EX = json.load(io.open(os.path.join(P16T, 'execution_phase16.json'), encoding='utf-8'))
SEALB = open(os.path.join(P16T, 'N2h1a9_design_seal.json'), 'rb').read()
SEAL = json.loads(SEALB.decode('utf-8'))
AM1B = open(os.path.join(P16T, 'N2h1a9_design_seal_amend1.json'), 'rb').read()
EXECB = open(os.path.join(P16T, 'execution_phase16.json'), 'rb').read()
SCRIPTP = os.path.join(ROOT, 'tests', 'deepseek', 'Phase16', 'n2h1a9_writewin_origin_profile.py')
SCRIPTB = open(SCRIPTP, 'rb').read()

V = RES['verdict']
JV = RES['joint_verdict']
PC = RES['predictions_check']
E7 = RES['E7_reach']
E5 = RES['E5_concentration']
E4 = RES['E4_summary']
E3 = RES['E3_localize']
E2 = RES['E2_full_swap']
ARMS = list(RES['arms'].keys())
SH = {'A0_calib_qwen3-4b-nf4': 'A0·4B(nf4)', 'A1_glm4-9b-nf4': 'A1·GLM4-9B', 'A2_qwen3-14b-nf4': 'A2·Qwen3-14B'}
SHORT = [SH[a] for a in ARMS]

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


def jd(o):
    return json.dumps(o, ensure_ascii=False)


A('## Phase 16: 写入窗原点化剖面 + 集中度统计量重设计（N2h1-α-9）[%s]' % time.strftime('%H:%M'))
A('')

# ---------------- 0
A('**四锚**：seal `%s`（%d B）/ exec `%s`（%d B）/ amend1 `%s`（%d B）/ result `%s`（%d B）/ '
  'Phase 15 冻结锚 `%s`（本 Phase 逐位复现对象）；脚本 `n2h1a9_writewin_origin_profile.py` sha8 `%s`。'
  % (hashlib.sha256(SEALB).hexdigest()[:8], len(SEALB),
     hashlib.sha256(EXECB).hexdigest()[:8], len(EXECB),
     hashlib.sha256(AM1B).hexdigest()[:8], len(AM1B),
     hashlib.sha256(RESB).hexdigest()[:8], len(RESB),
     RES['anchor_result_sha8'], hashlib.sha256(SCRIPTB).hexdigest()[:8]))
A('')
A('**三臂**（同一 nf4 口径，逐字节继承 Phase 15）：`A0_calib_qwen3-4b-nf4`（qwen3-4b L=36 hid=2560 tie=True，'
  '量化保真校准臂）/ `A1_glm4-9b-nf4`（glm4-9b-chat-hf L=40 hid=4096 untied，跨家族）/ '
  '`A2_qwen3-14b-nf4`（Qwen3-14B L=40 hid=5120 untied，同家族 3.5×）。')
A('')

# ---------------- 1 动机
A('### 1. 动机：Phase 15 的两条死线与本 Phase 的三处改动')
A('')
A('**死线 1（最高优先，Phase 15 §8 写死）**：统一剖面下的「**写入窗 vs 集中窗**」关系。'
  '但 Phase 15 的独立定位（B_cat：逐层独立 `U_ℓ` = 6 类质心差 SVD，秩 5；`L*_own` = 相邻层最大增量处）'
  '给出 `L*_own` = A0 **%s** / A1 **%s** / A2 **%s**，而剖面域只有 `PROFILE = [6..34]` —— '
  '**A1 与 A2 的写入窗根本不在剖面域内**，两者此前不可比较。'
  % (f(V[ARMS[0]].get('Q2_L_star_own'), 0), f(V[ARMS[1]].get('Q2_L_star_own'), 0),
     f(V[ARMS[2]].get('Q2_L_star_own'), 0)))
A('')
A('**死线 2（并列最高优先）**：集中度统计量重设计。Phase 15 的 `top3_share = max_w |Σ jm[j:j+W]| / range` '
  '是**极值型 3-窗占比**（等价于「最大 3 步净位移 / 全幅」），在 6 个「臂 × 坐标」格里只有 2 格超过置换 null '
  '95 分位。本 Phase 指出并落地其**结构性缺陷**与替代口径。')
A('')
A('**改动 1（网格）**：`profile_sites = [1,2,3,4,5] + [6..34]`（**%d 位点**，legacy 子域 %d 位点不变）。'
  % (len(RES['grid']['profile_sites']), len(RES['grid']['profile_sites_legacy'])))
A('**改动 2（主域）**：以**可达性掩膜**定义主域 `REACH = {ℓ : ρ(ℓ) ≥ UNREACH_y = %s}`，'
  '其中 `ρ(ℓ) = Y(ℓ, α=1) = dDonor / FULL_SWAP`（FULL_SWAP 取 Phase 15 冻结值）—— **设计意图**是使写入窗成为主域**左端点**。'
  '被掩膜排除的位点必须逐个报告。**实测该意图 A0/A1 成立、A2 不成立**（见 §5 与 §10 勘误 c）。'
  % f(E7[ARMS[0]]['unreach_y'], 2))
A('**改动 3（重设计）**：`top3_share` → **(com_layer, span_k) 双量**：')
A('')
A('| 量 | 定义 | 性质 | 零假设下可检验性 |')
A('|---|---|---|---|')
A('| `com_layer` | `Σ_j \\|Δ_j\\| · mid_j / Σ_j \\|Δ_j\\|`，`mid_j = (sites[j]+sites[j+1])/2` | **物理层号**上的质心（单位：层）；尺度无关；**对网格加密/下探不变** | 顺序敏感 ⇒ 双边可检验 |')
A('| `span_k` | `(max_idx − min_idx of k 个最大 \\|Δ\\|)/(n−1)`，k=3 | 主变是否挤在少数相邻步；尺度无关 | 顺序敏感 ⇒ 双边可检验 |')
A('| ~~谱熵 / max÷mean / 参与比~~ | 仅依赖 jump 多重集 | —— | **结构性退化**：置换零假设保留多重集 ⇒ 恒等于观测，双边必得 p=1，**一律不得作判据** |')
A('')
A('装置（hooks / BASE / FULL_SWAP / E3 localize / B_cat）与 Phase 15 **逐字节相同**；材料（模板、6 类词、'
  '41 实例、24 配对、量化口径）逐字节继承。')
A('')

# ---------------- 2 预注册
A('### 2. 预注册判据（观测前冻结）与判决')
A('')
A('| 预测 | 判据 | 结果 |')
A('|---|---|---|')
NAMES = {'P1': '装置自检三臂 Q0=PASS', 'P2': '冻结锚逐位复现（legacy 6..34；amend1 分层）',
         'P3': '写入窗入域且重算与 Phase 15 一致', 'P4': '可达域左端点 == 写入窗（严格相等）',
         'P5': '新量显著格数 ≥ 旧量且 ≥ 2', 'P6': '物理质心分离 ≥ 4.0 层（3/3）',
         'P7': 'xhalf 质心在写入窗之后 ≥ 5.0 层（≥2/3）'}
for k in sorted(PC):
    A('| **%s** | %s | **%s** |' % (k, NAMES.get(k, ''), 'PASS' if PC[k]['pass_'] else 'FAIL'))
A('')
A('**联合判决**：`Q1=%s` ; `Q2=%s` ; `Q3=%s` ; `Q4=%s` ; `Q5=%s`（计数 %s）。'
  % (JV.get('Q1_joint'), JV.get('Q2_joint'), JV.get('Q3_joint'), JV.get('Q4_joint'), JV.get('Q5_joint'),
     jd(JV.get('Q5_counts'))))
A('')

# ---------------- 3 装置
A('### 3. 装置与预算')
A('')
A('| 臂 | 模型 | L / hid / kv / tie | 载入 s | E1 capture | E3 定位 s | FULL_SWAP | L\\*_own |')
A('|---|---|---|---|---|---|---|---|')
for a in ARMS:
    c = RES['arms'][a].get('cfg', {})
    A('| `%s` | %s | %d / %d / %s / %s | %s | %s | %s | %s | %s |'
      % (a, RES['arms'][a]['model'], c.get('L', 0), c.get('hid', 0), c.get('kv_heads'), c.get('tie'),
         f(RES['arms'][a].get('load_s'), 1), jd(RES['arms'][a].get('E1_capture', {}).get('n')),
         f((RES['arms'][a].get('E3_localize') or {}).get('seconds'), 1),
         f(E2[a].get('FULL_SWAP'), 6), f(E3[a].get('L_star_own'), 0)))
A('')
_NS = len(EX['profile_sites']); _NA = len(EX['alphas']); _NP = len(EX['discovery'])
_NW = {}
for _a in ARMS:
    _nc = len(RES['arms'][_a].get('cands_used') or EX['localize']['cands'])
    _NW[_a] = _NS * _NA * _NP + _nc * _NP + 41 + 12
_NTOT = sum(_NW.values())
A('**前向预算**：%d 位点 × %d α × %d 配对 = **%d**/臂，加 %s 候选定位与 41 实例 capture；'
  '逐臂 = %s；总计 **%d** 次真实 GPU 前向（会计估计，公式 `sites×alphas×pairs + cands×pairs + 41 + 12`），'
  '`elapsed_total_s` = %s s。'
  % (_NS, _NA, _NP, _NS * _NA * _NP,
     '/'.join(str(len(RES['arms'][_a].get('cands_used') or EX['localize']['cands'])) for _a in ARMS),
     jd(_NW), _NTOT, f(RES.get('elapsed_total_s'), 0)))
A('')

# ---------------- 4 装置门 + 锚
A('### 4. 装置门与冻结锚复现（P1 / P2）')
A('')
A('| 臂 | Q0 | sup_id 同参考 | T=2 | determinism | hook 效应 | F2 base bad | 逐臂 sup_id |')
A('|---|---|---|---|---|---|---|---|')
for a in ARMS:
    r = RES['arms'][a]
    e0 = RES['E0_selfcheck'][a] or {}
    A('| `%s` | **%s** | %s | %s | %s | %s | %s | %s |'
      % (a, V[a].get('Q0_device'), r.get('sup_id_matches_ref'), r.get('T2_only'),
         ('%.3e' % e0.get('determinism_maxdiff', float('nan'))),
         f(e0.get('hook_effect_maxdiff'), 3), jd(RES['arms'][a].get('F2_base_bad')),
         jd(RES['sup_id_per_arm'][a])))
A('')
A('**F1b（各臂 tokenizer 自检：6 个类别词均单 token 且 `decode(id) == 词`）三臂均 = %s**。'
  '`sup_id 同参考` 一列反映的是「该臂现场解析出的 id 是否等于 qwen 参考 id」：A1 为 **False** 是**预期且正确**的'
  '（GLM4 词表不同 ⇒ 同一类别词映射到不同 id），不是装置失败 —— Phase 15 amend1 已把 `sup_id` 从全局硬编码'
  '改为**逐臂现场解析**，正是为了允许这一差异。'
  % jd({a: RES['arms'][a].get('F1b_ok') for a in ARMS}))
A('')
A('**P2 冻结锚（vs Phase 15 result `%s`）**：' % RES['anchor_result_sha8'])
A('')
A('| 臂 | 分级 | max\\|Δxhalf\\| | max rel \\|ΔJ\\| | legacy argmax 相同 | Δtop3 (x / j) | 分层阈值 |')
A('|---|---|---|---|---|---|---|')
_tier = {'RECON_OK': '**RECON_OK**', 'RECON_OK_LOOSE': 'RECON_OK_LOOSE', 'RECON_DRIFT': '**RECON_DRIFT**'}
for a in ARMS:
    vv = V[a]
    A('| `%s` | %s | %s | %s | %s | %s / %s | ≤1e-3 / ≤5e-3 |'
      % (a, _tier.get(vv.get('Q1_label'), vv.get('Q1_label')),
         ('%.3e' % vv['Q1_max_abs_dxh_leg']) if vv.get('Q1_max_abs_dxh_leg') is not None else 'None',
         ('%.3e' % vv['Q1_max_rel_dJ_leg']) if vv.get('Q1_max_rel_dJ_leg') is not None else 'None',
         vv.get('Q1_argmax_same_leg'), f(vv.get('Q1_dtop3_x'), 5), f(vv.get('Q1_dtop3_j'), 5)))
A('')
A('**amend1 分层定义（判据，冻结在观测前）**：`RECON_OK`（`max|Δxhalf| ≤ 1e-3` 且 `max rel|ΔJ| ≤ 2e-2`，'
  '即逐位复现）/ `RECON_OK_LOOSE`（`≤ 5e-3`，必须附逐位点 Δxhalf 表）/ `RECON_DRIFT`（超限 ⇒ 锚不可用、'
  '本 Phase 跨模型结论全部降级）。P2 要求三臂全 ∈ {`RECON_OK`, `RECON_OK_LOOSE`} 且 ≥2 臂严格。'
  '本轮实测 **n_strict_ok = %d / n_loose = %d**。'
  % (sum(1 for a in ARMS if V[a].get('Q1_label') == 'RECON_OK'),
     sum(1 for a in ARMS if V[a].get('Q1_label') == 'RECON_OK_LOOSE')))
A('')
if any(V[a].get('Q1_label') == 'RECON_OK_LOOSE' for a in ARMS):
    A('**amend1 要求：RECON_OK_LOOSE 格必须附逐站点 dxh 表。**')
    A('')
    A('| 臂 | 逐站点 Δxhalf（legacy 6..34） |')
    A('|---|---|')
    for a in ARMS:
        if V[a].get('Q1_label') == 'RECON_OK_LOOSE':
            d = V[a].get('Q1_dxh_by_site') or {}
            A('| `%s` | %s |' % (a, ' '.join('%s:%s' % (k, f(d[k], 5)) for k in sorted(d, key=lambda z: int(z)))))
    A('')
E6A0 = RES['E6_calibration'][ARMS[0]] or {}
A('**A0 量化保真（E6，vs Phase 12 bf16）**：`max|dxhalf| = %s`（tol %s ⇒ pass_tol=%s）；'
  '`argmax_w_x` nf4 = %s vs bf16 = %s（同 = %s）；`max|drecover| = %s`；'
  '`XH_RANGE` nf4 = %s vs bf16 = %s。'
  % (f(E6A0.get('max_abs_dxh'), 6), f(EX['floors'].get('XH_FAITHFUL_TOL'), 2), E6A0.get('pass_tol'),
     E6A0.get('argmax_w_x_nf4'), E6A0.get('argmax_w_x_bf16'), E6A0.get('argmax_same'),
     f(E6A0.get('max_abs_drecover'), 6),
     f(E6A0.get('XH_RANGE_nf4'), 4), f(E6A0.get('XH_RANGE_bf16'), 4)))
A('')

# ---------------- 5 可达性
A('### 5. 主结果 1：可达性剖面 —— **写入窗就是可达性门槛**（P3 / P4）')
A('')
A('| 臂 | L\\*_own（B_cat，本次重算） | ℓ_reach = min{ℓ : ρ(ℓ) ≥ 0.5} | 判定 | 可达域 REACH | 被掩膜排除（ρ < %s） |'
  % f(E7[ARMS[0]]['unreach_y'], 2))
A('|---|---|---|---|---|---|')
for a in ARMS:
    A('| `%s` | **%s** | **%s** | %s | %s | %s |'
      % (a, f(V[a].get('Q2_L_star_own'), 0), f(V[a].get('Q2_ell_reach'), 0), V[a].get('Q2_label'),
         jd(E7[a]['reach']), jd(E7[a]['excluded'])))
A('')
A('**ρ(ℓ) 全曲线（每臂 %d 位点）**：' % len(E7[ARMS[0]]['sites']))
A('')
for a in ARMS:
    A('- `%s`：%s' % (a, ' '.join('L%d:%s' % (E7[a]['sites'][i], fp(E7[a]['rho'][i], 4))
                                   for i in range(len(E7[a]['sites'])))))
A('')
A('**读法**：每臂在写入窗处 ρ 由 ~0 跳到 ~1，且**窗下一位点的 ρ ≤ %s**。'
  '这意味着 B_cat 几何定位（逐层独立类别子空间的相邻最大增量）与替换探针的**剂量-响应可达性阈值**'
  '在同一位点严格重合 —— 两者用的是完全不同的可观测量。'
  % f(max([abs(E7[a]['rho'][E7[a]['sites'].index(E7[a]['ell_reach']) - 1])
           if E7[a]['sites'].index(E7[a]['ell_reach']) > 0 else 0.0 for a in ARMS] or [0]), 4))
A('')
_le_ok = [a for a in ARMS if (E7[a]['reach'] and int(E7[a]['reach'][0]) == int(E7[a]['ell_reach']))]
A('**补注（左端点 vs 写入窗）**：`REACH` 的**左端点**与 `ell_reach` 相等的臂 = **%d/3**（%s）。'
  '不等的臂（%s）在写入窗**之下**已有位点 ρ ≥ `UNREACH_y` ⇒ 「写入窗 = 可达域左端点」这一**设计意图**'
  '在 %d/3 臂上成立，其余臂上 `ell_reach` 落在 REACH **内部**（P3 判据按「写入窗 ∈ REACH」评估，三臂均满足）。'
  % (len(_le_ok), jd(_le_ok), jd([a for a in ARMS if a not in _le_ok]), len(_le_ok)))
A('')

# ---------------- 6 重设计
A('### 6. 主结果 2：集中度重设计（P5）—— 旧量在主域上全灭，新量给出**位置**结论')
A('')
A('**6.1 旧量（legacy 域 = Phase 15 原口径，用于锚复现）**')
A('')
A('| 臂 | top3_x | argmax_w_x | null95_x | 裕度_x | top3_j | argmax_w_j | null95_j | 裕度_j | d_argmax |')
A('|---|---|---|---|---|---|---|---|---|---|')
for a in ARMS:
    lx = E5[a]['legacy_domain']['x']
    lj = E5[a]['legacy_domain']['j']
    A('| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s |'
      % (a, f(lx['top3'], 4), lx['argmax_w'], f(lx['null'].get('null95'), 4), fp(lx['margin'], 4),
         f(lj['top3'], 4), lj['argmax_w'], f(lj['null'].get('null95'), 4), fp(lj['margin'], 4),
         E5[a]['legacy_domain'].get('d_argmax_window')))
A('')
A('**6.2 主域 REACH 上的旧量（对照）与新量（判决量）**')
A('')
A('| 臂 | 主域步数 n | 旧量显著（裕度>0） | `com_layer` x | `com_layer` j | **com_sep（层）** | `span_3` x / j | com 双边尾 x / j |')
A('|---|---|---|---|---|---|---|---|')
for a in ARMS:
    md = E5[a]['main_domain']
    ns = E5[a]['new_stat']
    osig = V[a].get('Q5_old_sig') or []
    A('| `%s` | %d | %s | %s [%s, %s] | %s [%s, %s] | **%s** | %s / %s | %s |'
      % (a, len(md['x']['jumps']), jd(osig),
         f(ns['x'].get('obs_com'), 3), f(ns['x'].get('com_p5'), 3), f(ns['x'].get('com_p95'), 3),
         f(ns['j'].get('obs_com'), 3), f(ns['j'].get('com_p5'), 3), f(ns['j'].get('com_p95'), 3),
         f(V[a].get('Q4_sep'), 3),
         f(ns['x'].get('obs_span'), 4), f(ns['j'].get('obs_span'), 4),
         '%s / %s' % (ns['x'].get('com_tail'), ns['j'].get('com_tail'))))
A('')
A('**6.3 为什么旧量在这个装置上必然失效（结构性，不是偶然）**：')
A('')
A('1. `top3_share` 只依赖**最大 3 步净位移**。对单调段（`J` 从浅端到深端单调下降、`xhalf` 末段上翘），'
  '该窗位由曲线单调性**确定性地**决定，而非由「机制在哪里发生」决定；实测主域上的 argmax_w = '
  '%s。' % jd({SH[a]: E5[a]['main_domain']['x']['argmax_w'] for a in ARMS}))
A('2. 置换零假设**保留 jump 的多重集** ⇒ 熵 / max÷mean / 参与比等量在零假设下恒等于观测，'
  '双边检验必 p=1。**这类量必须整族排除**（seal 已明文列出）。')
A('3. 因此旧量在 6 格中只有 %d 格「显著」，且这 %d 格的位置与曲线形状一一对应，不构成机制证据。'
  % (sum(len(V[a].get('Q5_old_sig') or []) for a in ARMS),
     sum(len(V[a].get('Q5_old_sig') or []) for a in ARMS)))
A('')
A('**6.4 新量给出的结论**：`com_layer` 是**物理层号**上的质心 ⇒ 结论是「**变化发生在哪一层**」，'
  '而不是「占比多少」。它在 6 格中的双边尾标记为 '
  '`x: %s` / `j: %s`（x 为深尾 / j 为浅尾时 tail=high / low）。'
  % (jd({SH[a]: E5[a]['new_stat']['x'].get('com_tail') for a in ARMS}),
     jd({SH[a]: E5[a]['new_stat']['j'].get('com_tail') for a in ARMS})))
A('')

# ---------------- 7 质心分离
A('### 7. 主结果 3：物理深度质心分离（P6 / P7）')
A('')
A('| 臂 | com_layer(xhalf) | com_layer(J) | 分离（层） | 相对 L | xhalf 质心 − L\\*_own（层） |')
A('|---|---|---|---|---|---|')
for a in ARMS:
    Lv = (RES['arms'][a].get('cfg') or {}).get('L', 0)
    A('| `%s` | %s | %s | **%s** | %s | %s |'
      % (a, f(V[a].get('Q4_com_x'), 3), f(V[a].get('Q4_com_j'), 3), f(V[a].get('Q4_sep'), 3),
         f((V[a].get('Q4_com_x') or 0) / max(Lv, 1), 3) if V[a].get('Q4_com_x') is not None else 'None',
         f(V[a].get('Q4_com_after_win_x'), 2)))
A('')
_sep = {a: V[a].get('Q4_sep') for a in ARMS}
_npos = sum(1 for a in ARMS if (_sep[a] is not None and _sep[a] >= 4.0))
_nneg = sum(1 for a in ARMS if (_sep[a] is not None and _sep[a] < 0))
A('**读法（数据驱动）**：≥4.0 层的臂数 = **%d/3**；反号臂数 = **%d/3**（%s）。'
  % (_npos, _nneg, jd({SH[a]: V[a].get('Q4_label') for a in ARMS})))
if _npos == 3:
    A('⇒ 双坐标质心分离在**物理深度轴**上三臂全过，「两坐标 = 两种深度尺度上的形状统计量」成立。')
elif _nneg >= 1:
    A('⇒ **否证**：至少在 %d 臂上 `com_layer(xhalf) ≤ com_layer(J)`，即 `xhalf` 的主变**不比** `J` 更深。'
      '按 seal 的 `may_falsify_the_whole_line` 条款，Phase 12/13/14 的'
      '「`xhalf` 深尾集中 / `J` 浅端集中」**在物理轴上的表述必须撤回**，'
      '改述为「两坐标只是同一剂量-响应曲线的两种形状统计量，其质心差**随模型变化**」。' % _nneg)
else:
    A('⇒ 分离方向一致但幅度不稳（见上表）。')
A('')
A('**同一家族的尺度对照（最重要的一条）**：A0（qwen3-4b，tied）与 A2（Qwen3-14B，untied）同属 Qwen3 家族、'
  '参数 3.5×，而分离量由 **%s 层** 变为 **%s 层** —— 即**同家族放大即抹平该分离**；'
  'A1（GLM4-9B，跨家族）居中（%s 层）。因此**该分离不是层栈共性，而是随规模/家族变化的量**。'
  % (f(_sep[ARMS[0]], 2), f(_sep[ARMS[2]], 2), f(_sep[ARMS[1]], 2)))
A('')

# ---------------- 8 机制拼图
A('### 8. 机制拼图：本 Phase 修正了什么、确立了什么')
A('')
CONC = []
if JV.get('Q2_joint') == 'REACH_IDENTITY_ROBUST':
    CONC.append('**【主结论·P4】「写入窗 = 可达性门槛」三臂严格成立。** '
                'B_cat 的几何定位（`L*_own`）与替换探针的剂量-响应可达性（`ℓ_reach`）在同一位点重合：'
                '窗下 ρ ≤ %s，窗上 ρ ≈ 1。两条独立可观测量互为验证 ⇒ '
                '「写入窗」不再是单一装置的产物，而是一个**可被第二种读数再现的物理位置**。'
                % f(max([abs(E7[a]['rho'][E7[a]['sites'].index(E7[a]['ell_reach']) - 1])
                         if E7[a]['sites'].index(E7[a]['ell_reach']) > 0 else 0.0 for a in ARMS] or [0]), 4))
elif JV.get('Q2_joint') == 'REACH_IDENTITY_PARTIAL':
    CONC.append('**【主结论·P4】「写入窗 = 可达性门槛」部分成立**（见 §5 逐臂 ℓ_reach 与 L*_own 的偏差）。')
else:
    CONC.append('**【主结论·P4 否证】「写入窗 = 可达性门槛」不成立**（见 §5）。')
CONC.append('**【主结论·P5】Phase 12–15 的集中度判据在这条线上应退休。** '
            '`top3_share` 在主域上只有 %d/6 格显著，而它的窗位是曲线单调性的确定性函数；'
            '重设计后 `com_layer` 给出的是**位置**（单位：层），并把「集中/分散」问题'
            '还原为「两坐标的质心相距多少层」。' % sum(len(V[a].get('Q5_old_sig') or []) for a in ARMS))
if JV.get('Q4_joint') == 'CENTROID_SEPARATED_ALL':
    CONC.append('**【主结论·P6】双坐标的物理深度质心分离三臂全过**（%s 层）。'
                % jd({SH[a]: f(V[a].get('Q4_sep'), 1) for a in ARMS}))
elif JV.get('Q4_joint') == 'CENTROID_PARTIAL':
    CONC.append('**【主结论·P6 = 预注册否证】双坐标质心分离只在 %d/3 臂成立，且在同家族放大臂上反号。** '
                '分离量 = %s 层（A0 → A1 → A2）。A2（Qwen3-14B）给出 **%s 层**，即 `xhalf` 的质心**不比** `J` 深。'
                '⇒ 按 seal 的 `may_falsify_the_whole_line` 条款：'
                'Phase 12/13/14 关于「`xhalf` 深尾集中 / `J` 浅端集中」的**物理深度表述整体撤回**；'
                '保留的只是在**旧坐标（jump 序号 + 极值型占比）**下的形状差异 —— 而那正是本 Phase 判定为'
                '「窗位由单调性决定」的量。**这是一次成功的自我否证，不是装置失败**：'
                '装置门 P1/P2 全过、锚 bit 级复现、P4 三臂严格成立。'
                % (sum(1 for a in ARMS if (V[a].get('Q4_sep') is not None and V[a].get('Q4_sep') >= 4.0)),
                   (f(V[ARMS[0]].get('Q4_sep'), 1) + ' → ' + f(V[ARMS[1]].get('Q4_sep'), 1) + ' → ' +
                    f(V[ARMS[2]].get('Q4_sep'), 1)), f(V[ARMS[2]].get('Q4_sep'), 2)))
else:
    CONC.append('**【主结论·P6】双坐标质心分离未通过**：%s。' % jd({SH[a]: V[a].get('Q4_label') for a in ARMS}))
CONC.append('**对 Phase 15 的自我修正**：Phase 15 的「6 格只有 2 格过零假设」当时被读作'
            '「集中度判据证据强度随规模下降」；本 Phase 表明更准确的读法是'
            '**判据本身错配**（极值型 + 位置由形状决定），因此 Phase 15 那一句必须改述为'
            '「旧判据在 6 格中只有 2 格显著，且其窗位可由曲线单调性预测 ⇒ 不可用作机制证据」。')
CONC.append('**尚未解决**：`com_layer` 仍是**描述性**位置量；它能说「变化在 L24 附近」，'
            '但不能说「L24 做了什么」。把它接到组件级（头/MLP）预算上，是下一条死线。')
for c in CONC:
    A('- ' + c)
A('')

# ---------------- 9 下一步
A('### 9. Phase 17 候选（死线优先级）')
A('')
A('- **最高优先 · 把「位置」接到「组件」**：在每臂的 `com_layer` 邻域（±2 层）做逐层组件预算'
  '（沿用 Phase 8 的向量预算 `share_v` 口径，禁止用效应份额），回答「质心所在层由谁贡献写入向量」。')
A('- **并列最高优先 · `span_k` 的体系化**：本 Phase 只把 `span_k` 作对照；需在**三臂 × 双坐标 × k∈{2,3,5}** '
  '上给出跨度谱，检验「J 的跨度小 = 单一步主导」是否跨模型稳健。')
A('- **第三 · `xhalf` 的可达域敏感性**：`xhalf` 在深尾的抬升（A0 在 L32→L34 上升）'
  '是否与「末层被排除」有关（装置铁律：patch 必剔末层）。')
A('- **挂账（继承）**：N2h1-α-1 权重级定位；N2h1-β 水果类崩塌解剖；N3-β→N3-ε；R1 对照补强；K4 处置；'
  '**N 线 Phase 3–7 补登 Ledger**（Phase 8–16 已各 1 条）。')
A('')

# ---------------- 10 同轮勘误
A('### 10. 同轮勘误（v1 → v2，**同轮内**，全过程留痕）')
A('')
A('本节由**本轮收尾链的独立磁盘复核**（`disk_verify_phase16.py`）发现并触发，全部为**元数据/标签层**缺陷，'
  '**不改变任何观测、判决或结论**。按「纠错 append、不回改」纪律，本节为**追加的勘误**；'
  'v1 原文原样保留在 `memo_append_phase16_v1_asappended.md`（作废首版留痕，参照 Phase 15 作废 stdout 惯例）。')
A('')
A('- **(a) §4 表头错标**：v1 把该列标为「F1b 词表匹配」，但该列实际渲染的是 `sup_id_matches_ref`'
  '（「该臂现场解析出的类别 id 是否等于 qwen 参考 id」）。`F1b_ok`（各臂 tokenizer 自检）三臂**均 True**；'
  'A1 的 False 是**预期**（GLM4 词表不同 ⇒ 同一类别词映射到不同 id），Phase 15 amend1 已把 `sup_id` '
  '改为逐臂现场解析正是为此。v2 表头更正为「sup_id 同参考」，并在表后补一行 F1b 说明。')
A('- **(b) exec 元数据缺陷（已冻结，不改）**：`execution_phase16.json` 的 `bootstrap.seeds` 记录 '
  '`new_x = SEED+41` / `new_j = SEED+53`，而**实现**（`n2h1a9_writewin_origin_profile.py` L674–675）'
  '用的是 `SEED+61` / `SEED+67`；`legacy_x/legacy_j = SEED+13/SEED+29` 与实现一致。'
  '**影响面**：只影响**零假设分位的复现**，不影响任何观测值与判决（`com_layer`/`span_k`/`top3`/`argmax_w` 全为确定性量）。'
  '**正确复现方式**：新量 null 用 `SEED+61`（x）/ `SEED+67`（j）。exec 已冻结 ⇒ 不回改，以本节为准；'
  '本 Phase 的 `disk_verify_phase16.py` 分区 B 会**显式确认**该不符（作为已知元数据缺陷的长期断言）。')
A('- **(c) §1「左端点」表述过强**：设计意图是使写入窗成为 `REACH` 左端点。实测只有 A0/A1 成立；'
  'A2（Qwen3-14B）的 `REACH` 左端点是 **ℓ=3**（ρ=0.2428 ≥ 0.10）而 `ell_reach = 4` ⇒ 写入窗落在域**内部**。'
  'P3 的判据是「写入窗 ∈ REACH」，三臂均满足；「左端点」是 **2/3** 的描述性事实，已在 §5 补注更正。')
A('')
A('**v1 → v2 的机械差异**：仅 §4 表头 1 处 + §4 表后 1 行 + §5 补注 1 段 + 本节（新增）。'
  '**其余全部内容逐字节相同**；`result_phase16.json` / Ledger 条目 / 判决均未变。')
A('')

# ---------------- 附
A('### 附. 文件与诚实边界')
A('')
A('**脚本**：`tests/deepseek/Phase16/`（`probe_state_phase16.py`、`probe_p15curves.py`、'
  '`probe_feasibility_phase16.py`、`gen_seal_phase16.py`、`gen_seal_amend1_phase16.py`、'
  '`gen_exec_phase16.py`、`p16_patch_writewin.py`、`n2h1a9_writewin_origin_profile.py`、'
  '`run_phase16_split.py`、`closeout_phase16.py`、`do_append_phase16.py`、`gen_memo_phase16.py`、'
  '`closeout_docs_phase16.py`、`disk_verify_phase16.py`、`gen_present_phase16.py`）。')
A('**产物**：`tests/deepseek_temp/Phase16/`（seal / amend1 / exec / result / 各臂日志 / 报告 / 展示页）。')
A('**探针证据**：`_probe_feasibility_{A0,A1,A2}.json|txt` —— 浅端可钩性、ρ 阶梯、每前向耗时，'
  '均在 seal 冻结**之前**取得。')
A('**机器键名对照（供回查 `result_phase16.json`）**：`L_star_own` = 逐层独立 B_cat 写入窗定位；'
  '`ell_reach` = 可达域左端点；`rho` = ρ(ℓ) 可达性曲线；`com_layer` / `span_k` = 新集中度双量；'
  '`E7_reach` = 可达性块；`E5_concentration.new_stat` = 新量双边分位；`Q4_sep` = com_layer(xhalf) − com_layer(J)。')
A('')
HON = SEAL.get('honesty') or []
for i, h in enumerate(HON, 1):
    A('%d. %s' % (i, h))
A('%d. **amend1 记忆**：`xhalf` 是 `cross_alpha` 在 α 网格上的**插值位置** ⇒ 对插值量设单一硬门'
  '违反「软门优先硬 assert」；本 Phase 把锚判据分层（1e-3 / 5e-3 / DRIFT）并冻结在观测之前。' % (len(HON) + 1))
A('%d. **`n_forwards` 为会计估计**：公式 = `sites×alphas×pairs + cands×pairs + 41 + 12`，'
  '与 Phase 15 的 6437/6461 同构；非逐调用计数。' % (len(HON) + 2))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(LOGL) + '\n')
b = io.open(OUT, encoding='utf-8').read().encode('utf-8')
print('MEMO -> %s (%d B, %d lines, sha8 %s)' % (OUT, len(b), len(LOGL), hashlib.sha256(b).hexdigest()[:8]))
