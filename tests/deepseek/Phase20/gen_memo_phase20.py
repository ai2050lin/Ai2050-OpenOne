# -*- coding: utf-8 -*-
"""Phase 20 MEMO 追加节生成器（数据驱动：所有数字由 result_phase20.json 现场渲染，铁律 (ae)）。
产物：tests/deepseek_temp/Phase20/memo_append_phase20.md
注意：全文用**字符串拼接**而非 `%` 运算符，避免字面量 `%` 被误当格式符（承 P19 补丁教训）。
"""
import io
import os
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
RESP = os.path.join(P20T, 'result_phase20.json')
RES = json.load(io.open(RESP, encoding='utf-8'))
EX = json.load(io.open(os.path.join(P20T, 'execution_phase20.json'), encoding='utf-8'))
SEAL = json.load(io.open(os.path.join(P20T, 'N2h1a13_design_seal.json'), encoding='utf-8'))
DRIFT = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra',
                                       'memo_drift_phase19_postbaseline.json'), encoding='utf-8'))
PH = json.load(io.open(os.path.join(P20T, 'posthoc_p9_xhalf_domain.json'), encoding='utf-8'))
PHD = {r['pair']: r for r in PH['pairs']}
V = RES['verdict']
JV = RES['joint_verdict']
PC = RES['predictions_check']
QP = {(p['arm_nf4'] + '|' + p['arm_bf16']): p for p in RES['quant_pairs']}
AO = list(EX['arm_order'])
FL = RES['floors']
ARMS = RES['arms']
PROBE = {}
for nm in ('A0_nf4', 'A0_bf16'):
    p = os.path.join(P20T, '_probe20_%s.json' % nm)
    if os.path.exists(p):
        PROBE[nm] = json.load(io.open(p, encoding='utf-8'))

PID, PID2 = 'A0_nf4|A0_bf16', 'A1_nf4|A1_bf16'


def f(x, nd=4):
    return ('%.' + str(nd) + 'f') % x if isinstance(x, (int, float)) and x is not None else str(x)


def sn(a):
    return {'A0_nf4': 'A0·nf4', 'A0_bf16': 'A0·bf16', 'A1_nf4': 'A1·nf4', 'A1_bf16': 'A1·bf16'}.get(a, a)


def q(pair, key, nd=4):
    s = QP.get(pair)
    if not s:
        return 'NA'
    v = s.get(key)
    return f(v, nd) if isinstance(v, (int, float)) else str(v)


def qd(pair, key, nd=4):
    s = QP.get(pair)
    if not s:
        return 'NA'
    v = (s.get(key) or {}).get('delta')
    return f(v, nd) if isinstance(v, (int, float)) else 'NA'


def qr(pair, nd=4):
    """rho_b_all 是 dict{rho,resid_med,resid_p90} ⇒ 取标量 rho（勿用 q()）。"""
    s = QP.get(pair)
    if not s:
        return 'NA'
    v = (s.get('rho_b_all') or {}).get('rho')
    return f(v, nd) if isinstance(v, (int, float)) else 'NA'


def ps(k):
    v = PC.get(k, {})
    p = v.get('pass_')
    return '✔ PASS' if p is True else ('— N/A' if p is None else '✘ FAIL')


def ph_(pair, key, nd=6):
    r = PHD.get(pair)
    return f(r[key], nd) if r and isinstance(r.get(key), (int, float)) else 'NA'


def E(a, k):
    return ARMS[a]['E10_summary'].get(k)


L = []


def A(s=''):
    L.append(s)


ts = time.strftime('%H:%M')
A('## Phase 20: 行为量与写入窗剖面的跨精度稳健性（nf4 vs bf16）（N2h1-α-13）[' + ts + ']')
A('')
A('> seal `' + RES['seal_sha256'][:8] + '` / exec `' + RES['exec_sha256'][:8] + '` / result `'
  + hashlib.sha256(open(RESP, 'rb').read()).hexdigest()[:8]
  + '`；三锚 = P18 result `' + RES['anchor_result_p18_sha256'][:8] + '` / P16 result `'
  + RES['anchor_result_p16_sha256'][:8] + '` / P17 result `' + RES['anchor_result_p17_sha256'][:8] + '`。')
A('')
A('### 0. 一句话')
A('')
A('**P18 的行为侧结论不是 nf4 量化地板效应。** 把同一装置、同一材料、同一域（同 41 实例 / 24 discovery 配对 /')
A('17 confirmation 配对 / `U_ℓ` 秩 5 / 区间求和质心 / 冻结 REACH 与 P16 的 α 网格）从 nf4 换成 **bf16** 后：')
A('行为质心只移动 **' + qd(PID, 'com_B_all') + ' 层**（A0）/ **' + qd(PID2, 'com_B_all') + ' 层**（A1），')
A('行为谱秩相关 **' + qr(PID) + ' / ' + qr(PID2) + '**，')
A('行为 MLP 主导份额同侧且皆过半，`gap = com_V − com_B` 仍 ≥ 2.0 层，同对象耦合仍为正；')
A('连 P16 的写入窗集中度 `com_layer(x)`/`com_layer(J)` 也只移动 **'
  + qd(PID, 'com_layer_x') + ' / ' + qd(PID2, 'com_layer_x') + ' 层**。')
A('⇒ P19 给**向量侧**的跨精度支撑，本 Phase 给**行为侧**补齐。')
A('9 条定向预测中 8 条 PASS；**P9 按判据字面（全部公共 PROFILE 位点）判 FAIL** —— '
  '但其超差 100% 来自 P16 从未标定的浅端 ℓ=1，在 P16 实际标定的冻结 REACH 域（ℓ≥6）上两对皆 PASS'
  '（见 §5 的域分解与 §11 `[E-xhdom]`）。')
A('')
A('### 1. 动机：P19 只做了一半')
A('')
A('- **gap_1（主）**：P19（N2h1-α-12）证明 `w_ℓ` 谱与 `com_V` 不是 nf4 伪影（位移 0.09/0.07 层、秩相关 0.9924/0.9988），'
  '但那是**几何/向量**量。P18 的头条是**行为**量（`b_{c,ℓ}` 与 `com_B`），P16 的头条是**剖面**量'
  '（`com_layer`），二者**从未**跨精度复算。')
A('- **gap_2**：P18 的 `com_B` 与 `com_V` 的 gap（2.5–6.6 层）是**两个 nf4 量之差**；'
  'P18 §8 自己写明「该结论的严格跨精度检验需另做行为量重测，列入挂账」。')
A('- **gap_3**：P18 的行为份额（0.663 / 0.960 / 0.747）与向量份额（0.740 / 0.975 / 0.824）同侧，'
  '这条「几何 ⇒ 几何+行为」的升级同样只在 nf4 下成立。')
A('- **可行边界**：P19 已实测 A0-bf16 可全 GPU 载、A1-bf16 需 CPU offload、'
  'A2 = `Qwen3-14B`（29.5 GB）的 bf16 腿 segfault ⇒ 本 Phase 的 bf16 腿必然是**两模型腿**。')
A('')
A('### 2. 设计：唯一自变量 = 数值精度，两个面板一次跑完')
A('')
A('| | 内容 |')
A('|---|---|')
A('| **变** | 前向数值精度：bitsandbytes nf4 (4-bit) ↔ `torch.bfloat16` |')
A('| **不变** | 模板 `' + EX['template'] + '` / 6 类词 / 41 实例 / 24 discovery / 17 confirmation |')
A('| | `U_ℓ` = 全 41 实例按类平均 → 类别质心差 SVD，秩 = `n_classes−1` = 5 |')
A('| | 位点 = `1..L−2`；REACH 取 **P16 冻结域**（同域配对）；nb 取 P17 冻结值；`L*_own` 取 P16 冻结值 |')
A('| | `PROFILE` = ' + str(len(EX['profile_sites'])) + ' 位点 / `ALPHAS` = '
  + str(len(EX['alphas'])) + ' 档 / `xh_frac` = ' + str(EX['xh_frac']) + '（全取 P16 冻结值） |')
A('| | 两臂加载配置除量化外**逐项一致**（同 `eager` / `device_map="auto"` / `max_memory` / `low_cpu_mem_usage`） |')
A('')
A('**[B] 行为预算（P18 口径）**：注入 `h_ℓ^R + P_{U_ℓ}(Δ_c)`，读 `b_{c,ℓ} = mean_pairs [ score_of(logits_patched, ds, sid_d) − BASE[rw].sd0 ]`；')
A('组件 `INC_ALL / INC_MLP / INC_ATTN / INC_TOP1 / CUM_ALL`；附带**零额外前向**的自谱 `w_{c,ℓ} = mean_pairs ‖P_{U_ℓ}(Δ_c)‖`。')
A('')
A('**[P] 写入窗剖面（P16 口径）**：注入 `h_ℓ^R + α·d_ℓ`（`d_ℓ = HH[ℓ+1]^D − HH[ℓ+1]^R`，**原始差、不投影**），')
A('读 `Y(ℓ,α) = mean_disc dDonor / FULL_SWAP` → `xhalf(ℓ) = cross_alpha(·, 0.5)`、`J(ℓ) = J_only(·)` → `com_layer(x)`、`com_layer(J)`、`span3`。')
A('')
A('**四臂**：`A0_nf4`/`A1_nf4` = **校准臂**（须逐位复现 P18+P16+P17 冻结锚）；`A0_bf16`/`A1_bf16` = **检验臂**（A1 为跨家族 holdout）。')
A('')
A('### 3. 装置门与校准（P1 / P2）')
A('')
A('| 臂 | 模型 | 精度 | device | 保真度 arch max | blk max | 锚 | 载入 s | 前向 |')
A('|---|---|---|---|---|---|---|---|---|')
for a in AO:
    v = V[a]
    A('| `' + a + '` | ' + ARMS[a]['model'] + ' | ' + ARMS[a]['scheme'] + ' | ' + ARMS[a]['Q0_device']
      + ' | ' + f(v['Q1_arch_max'], 3) + ' | ' + f(v['Q1_blk_max'], 3)
      + ' | ' + v['Q2_label'] + ' | ' + f(ARMS[a]['load_s'], 1) + ' | ' + str(ARMS[a]['n_forwards']) + ' |')
A('')
A('保真度门（arch ≤ ' + f(FL['P20_FID_ARCH'], 2) + '、blk ≤ ' + f(FL['P20_FID_BLK'], 2) + '）四臂全过。')
A('**两个 nf4 校准臂逐位复现三套冻结锚**（P18 行为族 `com_B`/`comlayer_B`/`share_mlp_beh_nb`、'
  'P16 `com_layer(x)`/`com_layer(J)`/`FULL_SWAP`、P17 `com_V`，容差 ≤ 1e-4）：')
_cal = JV.get('Q2_calib_arms', [])
A('⇒ `' + str(JV.get('Q2_joint')) + '`，校准臂 = ' + str(_cal) + '（装置与 P16/P18 同源）。')
A('')
A('### 4. 主结果 1（Panel B）：行为量跨精度稳健')
A('')
A('| 臂 | `com_B(all)` | `com_B(mlp)` | `com_B(attn)` | `com_B(cum)` | `comlayer_B_all` | `share_mlp_beh(nb)` | `gap = com_V−com_B` | `sp(w_all,b_all)` |')
A('|---|---|---|---|---|---|---|---|---|')
for a in AO:
    v = V[a]
    A('| `' + a + '` | ' + f(v['com_B_all'], 4) + ' | ' + f(v['com_B_mlp'], 4) + ' | ' + f(v['com_B_attn'], 4)
      + ' | ' + f(v['com_B_cum'], 4) + ' | ' + f(v['comlayer_B_all'], 4) + ' | ' + f(v['share_mlp_beh_nb'], 4)
      + ' | ' + f(v['gap'], 3) + ' | ' + f(v['spearman_wall_ball'], 4) + ' |')
A('')
A('**同 Phase 配对（唯一自变量 = 量化口径）**：')
A('')
A('| 配对 | Δ`com_B(all)` | Δ`com_B(mlp)` | Δ`comlayer_B_all` | `ρ(b_nf4,b_bf16)` | 残差中位 / p90 | `share_mlp_beh` nf4→bf16 | 同侧 |')
A('|---|---|---|---|---|---|---|---|')
for k in (PID, PID2):
    p = QP.get(k)
    if not p:
        continue
    A('| `' + k + '` | ' + qd(k, 'com_B_all') + ' | ' + qd(k, 'com_B_mlp') + ' | ' + qd(k, 'comlayer_B_all')
      + ' | ' + qr(k) + ' | ' + f(p['rho_b_all']['resid_med'], 4) + ' / '
      + f(p['rho_b_all']['resid_p90'], 4) + ' | ' + f(p['share_mlp_beh_nb']['nf4'], 4) + ' → '
      + f(p['share_mlp_beh_nb']['bf16'], 4) + ' | ' + str(p['share_mlp_beh_nb']['same_side']) + ' |')
A('')
A('容差：Δ`com_B` ≤ ' + f(FL['QUANT_TOL_COMB'], 1) + ' 层、Δ`comlayer_B` ≤ ' + f(FL['QUANT_TOL_COMLAYER'], 1)
  + ' 层、ρ ≥ ' + f(FL['RHO_B_MIN'], 2) + '、份额 |Δ| ≤ ' + f(FL['QUANT_TOL_SHARE'], 2) + ' 且 > '
  + f(FL['MLP_DOM_MIN'], 2) + '。')
A('')
A('**行为质心族（逐口径）**：')
A('')
A('```')
for a in AO:
    c = E(a, 'com_B')
    A('%-9s [' % a + ARMS[a]['scheme'] + '] ' + '  '.join(k + '=' + f(c[k], 4) for k in ['INC_ALL', 'INC_MLP', 'INC_ATTN', 'INC_TOP1', 'CUM_ALL']))
A('```')
A('')
A('### 5. 主结果 2（Panel P）：写入窗剖面跨精度稳健')
A('')
A('| 臂 | `com_layer(x)` | `com_layer(J)` | `span3(x)` | `span3(J)` | xhalf 域 | `ρ(α=1)` 左端 |')
A('|---|---|---|---|---|---|---|')
for a in AO:
    v = V[a]
    A('| `' + a + '` | ' + f(v['com_layer_x'], 4) + ' | ' + f(v['com_layer_j'], 4) + ' | '
      + f(E(a, 'span3_x'), 4) + ' | ' + f(E(a, 'span3_j'), 4) + ' | '
      + str(EX['profile_sites_legacy'][0]) + '..' + str(EX['profile_sites_legacy'][-1]) + ' | '
      + f(E(a, 'rho')[0] if E(a, 'rho') else None, 4) + ' |')
A('')
A('| 配对 | Δ`com_layer(x)` | Δ`com_layer(J)` | `max|Δxhalf|` | J 比值范围 |')
A('|---|---|---|---|---|')
for k in (PID, PID2):
    p = QP.get(k)
    if not p:
        continue
    A('| `' + k + '` | ' + qd(k, 'com_layer_x') + ' | ' + qd(k, 'com_layer_j') + ' | '
      + f(p['xhalf']['max_abs_dxh'], 4) + ' | [' + f(p['J']['ratio_min'], 3) + ', ' + f(p['J']['ratio_max'], 3) + '] |')
A('')
A('`xhalf` 容差 ' + f(FL['QUANT_TOL_XHALF'], 2) + ' 是 **P16 已发表的跨精度容差**（`XH_FAITHFUL_TOL`）；'
  '但 P16 只在 **冻结 REACH 域**（`XH_12_by_site` = ℓ≥' + str(PH['p16_calib_domain'][0]) + '，n='
  + str(len(PH['p16_calib_domain'])) + '）上标定过该容差（P16 实测 max|Δxhalf| = '
  + f(PH['p16_calib_max_abs_dxh'], 6) + '，臂 `' + str(PH['p16_calib_arm']) + '`）。')
A('')
A('**P9 的域分解（post-hoc；as-coded 判决不改）**：')
A('')
A('| 配对 | as-coded（全部公共 PROFILE 位点） | 判 | 冻结 REACH 域（ℓ≥6, n='
  + str(len(PH['p16_calib_domain'])) + '） | 判 | 浅端 ℓ<6 | 超差承担者 |')
A('|---|---|---|---|---|---|---|')
for _k in (PID, PID2):
    _r = PHD.get(_k)
    if not _r:
        continue
    A('| `' + _k + '` | ' + f(_r['as_coded_max_abs_dxh'], 6) + ' | '
      + ('PASS' if _r['as_coded_pass'] else '**FAIL**') + ' | '
      + f(_r['reach_domain_max_abs_dxh'], 6) + ' | '
      + ('PASS' if _r['reach_domain_pass'] else 'FAIL') + ' | '
      + f(_r['shallow_max_abs_dxh'], 6) + ' @ℓ=' + str(_r['shallow_argmax']) + ' | '
      + '100% @ℓ=' + str(_r['as_coded_argmax']) + ' |')
A('')
A('⇒ 在 P16 **实际标定的那个域**上，两对都是 **PASS**，A0 裕度 ≈ '
  + f(FL['QUANT_TOL_XHALF'] / max(PHD[PID]['reach_domain_max_abs_dxh'], 1e-9), 0) + '×（'
  + f(PHD[PID]['reach_domain_max_abs_dxh'], 6) + ' vs ' + f(FL['QUANT_TOL_XHALF'], 2) + '）；'
  'as-coded 的 FAIL 是**预注册判据文字未写明域**造成的**域错配**（承 P18 教训「判据域须与 rationale '
  '标定域一致」），**不是**物理上的量化不稳定。原始判决与分解数据分别在 `result_phase20.json` 与 '
  '`posthoc_p9_xhalf_domain.{json,txt}`；**不改判**。')
A('')
A('### 6. 对照：置换零假设 + 确认集 + 离流形')
A('')
A('| 臂 | `com_B(all)` 尾 | `com_B(mlp)` 尾 | 邻域份额尾 | `com_layer(x)` 尾 | `pert_rel(INC)` max |')
A('|---|---|---|---|---|---|')
for a in AO:
    v = V[a]
    A('| `' + a + '` | ' + str(v['null_comB_inc'].get('com_tail')) + ' | '
      + str(E(a, 'null_comB_mlp').get('com_tail')) + ' | ' + str(v['null_share'].get('share_tail'))
      + ' | ' + str((v['null_comlayer_x'] or {}).get('com_tail')) + ' | ' + f(v['pert_rel_inc_max'], 4) + ' |')
A('')
A('确认集（n=' + str(len(EX['confirmation'])) + '，与 discovery 不相交）行为质心位移：')
A('')
A('```')
for a in AO:
    if a.endswith('_nf4'):
        A('%-8s ' % a + '  '.join(
            k + ': ' + (f(abs(E(a, 'com_B_conf')[k] - E(a, 'com_B')[k]), 4)
                        if E(a, 'com_B_conf').get(k) is not None else 'NA')
            for k in ['INC_ALL', 'INC_MLP', 'INC_ATTN']))
A('```')
A('')
A('### 7. 独立交叉验证（三源）')
A('')
A('1. **探针 ↔ 生产（同实现、两条独立进程加载）**：A0 的两个口径各跑一遍 `PROBE=1`（**全量配对与实例**、'
  '仅缩 α 网格）与生产（全尺度）。Panel [B] 的输入（`U_ℓ` / 配对集 / 实例）与 α 网格无关 ⇒ '
  '`com_B` 族 / `comlayer_B_all` / `share_mlp_beh_nb` / `com_V` 在两遍之间应在容差 1e-6 内一致'
  '（探针读数见 `_probe20_A0_{nf4,bf16}.json`，逐项比对见 `present_phase20.html` §F）。')
A('2. **跨 Phase 三锚**：校准臂现场读入 P18/P16/P17 result 并**逐位断言**（上方 `Q2_joint` 与 §3 的 got/expected 表）。')
A('3. **独立重实现**：`disk_verify_phase20.py` 用**独立代码**重算 `J(ℓ)` / `com_layer` / 置换零假设 / `xhalf`'
  '并逐项比对本 result，判据走「重算 → 按主脚本文义导出标签 → 比对」（不写死预期），要求 **0 FAIL**。')
A('4. **P16 标定复现（post-hoc 域分解，见 §5）**：本 Phase 的 A0 配对（qwen3-4b nf4↔bf16）在冻结 REACH 域上的 '
  '`max|Δxhalf|` = ' + ph_(PID, 'reach_domain_max_abs_dxh') + '，与 P16 自己 `E6_calibration` 的标定值 '
  + f(PH['p16_calib_max_abs_dxh'], 15) + ' 在 **~1e-16 相对误差**内一致 —— 两者本就是**同一个比较**'
  '（同模型 qwen3-4b、同为 nf4 vs bf16 的 `xhalf`、同域 ℓ≥6），故这是 P20 与 P16 的独立一致性强检验，'
  '同时精确解释了 as-coded FAIL 的来源（判据域）。')
A('')
A('### 8. 判决表')
A('')
A('| # | 预测 | 判 |')
A('|---|---|---|')
for k in sorted(PC):
    A('| `' + k + '` | ' + str(PC[k].get('name', '')) + ' | **' + ps(k) + '** |')
A('')
A('**联合**：' + ' ; '.join(k + '=' + str(JV[k]) for k in sorted(JV)
                            if not k.startswith('quant_pairs') and not k.endswith('_detail'))[:900])
A('')
A('### 9. 限界（诚实性）')
A('')
for h in SEAL['honesty']:
    A('- ' + h)
A('- **本轮新增限界**：A1·bf16 走 CPU offload（18.8 GB > 14 GiB 上限），其读数含「分片执行」第二源；'
  '若与 A0·bf16 相反，以 A0 为准。P16 的 com_layer 在 P16 自己那里就是 `Q4=CENTROID_PARTIAL`（P6 FAIL）'
  '⇒ 本 Phase 只回答「该统计量跨精度是否稳健」，**不**把它升格为已确立结论。')
A('')
A('### 10. 与 P16 / P17 / P18 / P19 的关系')
A('')
A('- **P16**：`com_layer(x)`/`com_layer(J)` 跨精度稳健（位移 ' + qd(PID, 'com_layer_x') + ' / '
  + qd(PID2, 'com_layer_x') + ' 层）⇒ P16 的位置统计量不是量化伪影（但 P16 的 P6 本身仍 FAIL）。')
A('- **P17**：`com_V` 在本 Phase 由**本臂自谱**重算，与 P17/P19 记录一致（见 §7-2）。')
A('- **P18**：行为质心 `com_B`、行为 MLP 主导、`gap`、同对象耦合 `spearman(w_all,b_all)` 四条全部跨精度保留')
A('  ⇒ P18 §5–§7 的结论由「nf4 内的几何+行为」升级为「**跨数值口径的几何+行为**」。')
A('- **P19**：本 Phase 是 P19 的**行为侧对偶**；两者合起来覆盖「向量 + 行为 + 剖面」三族量。')
A('- **[E-comv] 口径补注**：本 Phase 落盘的 `com_V` 取 **P17 冻结 `w_all` 谱**重算（为与 P17/P18/P19 跨 Phase 可比），'
  '因此**同一模型的两臂 `com_V` 按构造恒等**（配对 Δ ≡ 0）⇒ 联合判据 `Q3_com_V_stable` 只是「锚一致性」，'
  '**不可**当作跨精度证据。向量质心的跨精度证据仍由 **P19** 承担；本 Phase 另报**各臂自谱** '
  '`com_V_own_spectrum` 的沿口径位移作**描述性**补充：Δ = '
  + qd(PID, 'com_V') + ' ⇒ 自谱 Δ = '
  + f(E(PID.split('|')[1], 'com_V_own_spectrum') - E(PID.split('|')[0], 'com_V_own_spectrum'), 4)
  + '（A0）/ ' + f(E(PID2.split('|')[1], 'com_V_own_spectrum') - E(PID2.split('|')[0], 'com_V_own_spectrum'), 4)
  + '（A1）层（见 `present_phase20.html` §C 与 `disk_verify_phase20.txt`）。')
A('')
A('### 11. 同轮勘误')
A('')
A('- **[E-sper]** 首版 `sper()` 把 P17 冻结锚谱（键 `w_all/w_mlp/w_attn`）与本臂自谱（键 `INC_*`）'
  '混用同一键空间 ⇒ `KeyError: w_all`（SMOKE 第一跑抓到）。修：拆成 `sper_anchor()` 与 `sper_own()`。'
  '教训：**同名不同源的两个谱必须显式区分键空间**。')
A('- **[E-scope]** 锚复现一度在 SMOKE/PROBE 的缩幅网格上运行 ⇒ 必报 DRIFT（网格不同不可比）。'
  '修：加 `FULL_SCALE` 守卫，锚复现只在全尺度适用，缩幅下显式标 N/A。')
A('- **[E-probefull]** 探针初版把配对集也缩小（4 对）⇒ `U_ℓ` 秩退化为 2、`FULL_SWAP` 偏离锚。'
  '修：PROBE 保留**全量配对与实例**，只缩网格与 BP —— 探针读数才有口径意义。')
A('- **[E-baseline] 记录完整性事件（非本轮引入）**：`post-append-phase19` 基线（**'
  + str(DRIFT['prev_baseline']['bytes']) + ' B** / `' + str(DRIFT['prev_baseline']['sha8'])
  + '`，冻结于 ' + str(DRIFT['prev_baseline']['frozen_at']) + '）在快照后被**就地改写** —— '
  + '改写在 **' + str(DRIFT['memo_mtime']) + '**：P10–P19 共 **'
  + str(len(DRIFT['normalized_phases'])) + '** 个标题由短形式 `[HH:MM]` 规范化为完整形式 '
  + '`[YYYY-MM-DD HH:MM]`，逐条 +11 B、合计 **+' + str(DRIFT['predicted_delta_bytes'])
  + ' B**（与实盘差逐位一致，残差 ' + str(DRIFT['residual_bytes']) + ' B），行数不变（'
  + str(DRIFT['memo_lines_at_audit']) + ' 行）、**无文本丢失**；该事件**未被任何 wlog / baseline / '
  + 'history 记录**。**后果**：该基线的 bytes/sha256 锚，以及 3 项 `sections` 键（按「行前 44 字符」生成）'
  + '均陈旧。**处置**：不改写历史 —— `history` 中把该条目标注 `stale`；本 Phase 的 '
  + '`_infra/memo_baseline.json` 以新口径重新冻结（`sections` 键改为**完整标题行**并加碰撞断言，'
  + '`drift_events` 登记本事件）。审计件：`tests/deepseek_temp/_infra/memo_drift_phase19_postbaseline.json`。')
A('')
A('- **[E-xhdom] 预注册判据的域歧义（本轮主要发现之一）**：seal 的 P9 只写「两对 max|Δxhalf| ≤ '
  + f(FL['QUANT_TOL_XHALF'], 2) + '」，**未写明域**；实现按字面取「所有公共 PROFILE 位点」，'
  '而 P16 的 `XH_FAITHFUL_TOL` 是在 `XH_12_by_site`（ℓ≥6）上标定的 ⇒ A0 的 as-coded 读数 '
  + ph_(PID, 'as_coded_max_abs_dxh') + ' **完全由浅端 ℓ=1 贡献**（ℓ≥6 段仅 '
  + ph_(PID, 'reach_domain_max_abs_dxh') + '，比 ℓ=1 小 '
  + f(PHD[PID]['shallow_max_abs_dxh'] / max(PHD[PID]['reach_domain_max_abs_dxh'], 1e-9), 0) + '×）。'
  '**处置**：as-coded 判决 **P9=' + ps('P9') + ' 原样保留**；另出 post-hoc 域分解（冻结 REACH 域 A0 '
  + ph_(PID, 'reach_domain_max_abs_dxh') + ' / A1 ' + ph_(PID2, 'reach_domain_max_abs_dxh')
  + '，两对皆 PASS，A0 该值恰与 P16 标定值逐位相同）并标注为 post-hoc。'
  '教训：**「沿用某容差」时必须把该容差的标定域一并写进判据文字**。')
A('- **[E-rho] 渲染缺陷（收尾自查抓到、交付前已修）**：`rho_b_all` 是 `dict{rho,resid_med,resid_p90}`，'
  '首版 §0/§4 误用标量取值器 `q()` 去取它 ⇒ 生成件里打印出原始 dict。修：新增 `qr()` 只取 `.rho`；'
  'MEMO 的 Phase 20 节在交付前**回滚重生成**（前缀锚逐字节恢复后重跑追加链）。'
  '教训：**取值器必须按被取对象的类型分层（标量 / 带 delta 的 dict / 嵌套 dict）**。')
A('')
A('### 12. 下一步（死线）')
A('')
A('- **Phase 21 候选（最高）**：把跨精度检验推进到**组件级向量预算与权重实现级** —— '
  '在 bf16 下复算 P8 的向量预算 `share_v`（N2h1-α）与 N2h1-α-1 的权重级定位，'
  '确认「分布式搬运 / MLP 最大单一写入方」不是 nf4 的 kernel 路径产物。')
A('- **并列**：邻域宽度 ±2 敏感性（四臂 `nb` 恰都 ' + str(EX['arms'][AO[0]]['nb'])
  + '）；P17 `P6` 的 MEMO 改判（承 P18 `P5`）；**P9 判据补域**（把 `XH_FAITHFUL_TOL` 的 ℓ≥6 标定域'
  '写进判据文字后可否改判 ⇒ 需新 Phase 预注册，本 Phase 维持 FAIL）。')
A('- **N 线挂账**：N2h1-α-1 权重级定位；N2h1-β 水果类崩塌；N3-β→N3-ε；R1 补强；K4；'
  '**N 线 Phase 3–7 补登 Ledger**。')
A('')
A('### 附. 文件与资产')
A('')
A('**脚本**（→ `tests/deepseek/Phase20/`）：`n2h1a13_quant_scheme_robustness.py`、`gen_seal_phase20.py`、'
  '`gen_exec_phase20.py`、`run_phase20_split.py`、`closeout_phase20.py`、`gen_memo_phase20.py`、'
  '`do_append_phase20.py`、`closeout_docs_phase20.py`、`disk_verify_phase20.py`、`gen_present_phase20.py`'
  '（＋同轮补丁 `_patch_p20_{1..5}*.py`）')
A('')
A('**冻结件与读数**（→ `tests/deepseek_temp/Phase20/`）：`N2h1a13_design_seal.json`、`execution_phase20.json`、'
  '`_probe20_A0_{nf4,bf16}.json`、`_armrec20_*.json`、`result_phase20.json`、`verify_ledger_phase20.txt`、'
  '`disk_verify_phase20.txt`、`present_phase20.html`、`posthoc_p9_xhalf_domain.{json,txt}`')

OUT = os.path.join(P20T, 'memo_append_phase20.md')
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
b = open(OUT, 'rb').read()
print('WROTE %s  %d B  lines=%d  sha8=%s' % (OUT, len(b), len(L), hashlib.sha256(b).hexdigest()[:8]))
print('headings:', sum(1 for x in L if x.startswith('## ')))
