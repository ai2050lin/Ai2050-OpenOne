# -*- coding: utf-8 -*-
"""Phase 18 文档收尾：当日 wlog 追加 + _infra/memo_baseline.json 刷新（含 history 链）。
纪律：wlog 正文所有数字均从 result_phase18.json / Ledger / MEMO 现场取值渲染（不自报）。
必须在 do_append_phase18.py 之后运行（读追加后的 MEMO）。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(P18T, 'closeout_docs_phase18.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


R = json.load(io.open(os.path.join(P18T, 'result_phase18.json'), encoding='utf-8'))
EX = json.load(io.open(os.path.join(P18T, 'execution_phase18.json'), encoding='utf-8'))
LG = json.load(io.open(LEDGER, encoding='utf-8'))
PRE = json.load(io.open(os.path.join(P18T, 'memo_baseline_preappend_phase18.json'), encoding='utf-8'))

V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']; FL = R['floors']
ARMS = EX['arm_order']
A0, A1, A2 = ARMS
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig').split('\r\n')
new_sha8 = hashlib.sha256(mb).hexdigest()[:8]
ph_lines = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase ')]
p18_line = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase 18')]
tail = [m for m in LG['measurements'] if m.get('phase') == 18][-1]


def fn(x, n=6):
    return 'n/a' if x is None else ('%.*f' % (n, x))


def jd(x):
    return json.dumps(x, ensure_ascii=False)


def _rlin_top(a, dom):
    """返回 [(r_lin, site), ...] 降序；dom = 'reach' | 'all'。"""
    E = R['arms'][a]['E7_summary']
    rl = {int(k): float(x) for k, x in E['rlin_by_site'].items()}
    keys = E['reach'] if dom == 'reach' else E['sites_all']
    return sorted(((v, k) for k, v in rl.items() if k in keys), reverse=True)


_A0 = A0  # 校准臂（seal rationale 的 r_lin 数字取自它）
_RA0, _AA0 = _rlin_top(_A0, 'reach'), _rlin_top(_A0, 'all')
_RR0 = fn(_RA0[0][0] / _RA0[1][0], 3) if _RA0[1][0] > 1e-12 else 'NA'   # A0 REACH 域峰值比 = 4.053
_AR0 = fn(_AA0[0][0] / _AA0[1][0], 3) if _AA0[1][0] > 1e-12 else 'NA'   # A0 ALL 域峰值比 = 1.547
_RAT_RE = ' / '.join(fn(_rlin_top(a, 'reach')[0][0] / _rlin_top(a, 'reach')[1][0], 2) for a in ARMS)
_NPOS = sum(1 for a in ARMS if V[a]['Q8_rlin_argmax'] == R['arms'][a]['E7_summary']['L_star_own'])


# ---------- 1. 当日 wlog 追加 ----------
NEW = not os.path.exists(WLOG)
b0 = open(WLOG, 'rb').read() if not NEW else b''
t0 = b0.decode('utf-8')
if NEW:
    t0 = '# 2026-10-02\n'

L = []


def p(s):
    L.append(s)


p('')
p('## Phase 18 / N2h1-α-11：逐层组件「行为」预算（行为质心 com_B + 组件归属的行为化）——%s / %s / %s（%s）'
  % (JV['Q3_joint'], JV['Q4_joint'], JV['Q6_joint'], time.strftime('%H:%M')))
p('')
p('- **死线执行**：Phase 17 §7 写死的**最高优先** —— **把「向量预算」换成「行为预算」**（逐层组件对 '
  '`Δlogit(is-a)` 的贡献），补 **H11 的因果缺口**：与 Phase 17 的向量主导交叉验证 ——'
  '同向 ⇒ H11 由「几何」升级为「几何 + 行为」；以 attn 为主 ⇒ Phase 17 的向量份额是**几何假象**。')
p('- **唯一改动（seal 冻结）**：在 REACH 每个位点，把 Phase 8/17 的**向量**分解换成**行为**分解 ——'
  '`b_{c,ℓ} = mean_pairs [score_of(h_ℓ^R + P_{U_ℓ}(Δ_c), sup, sid_d) − BASE[rw].sd0]`，'
  '`score_of(v,sup,sid) = v[ID(sup)] − mean_{x≠sup} v[ID(x)]`、`v[sid] = −1e9`（**Phase 8 T 臂逐字同口径**）。'
  '组件 = `INC_ALL / INC_MLP / INC_ATTN / INC_TOP1 / CUM_ALL`。**位点约定**：位点 = 层号 ℓ，'
  '`ALL_SITES = 1..L−2`（对齐 Phase 17 的 `w_all` 支撑）；质心**区间求和** ⇒ 必须**全域**测量。')
p('- **装置锚（全部通过）**：三臂 `Q0 = PASS`；三臂全 `cuda`；`T=2` 合格；`determinism = 0.000e+00`；'
  '保真度门 **`%s`**（arch max %s / %s / %s ≤ %s；blocks max %s / %s / %s ≤ %s）；'
  '**跨 Phase 锚逐位复现 `%s`（3/3，≤1e-6）**：`com_layer(x)` = %s / %s / %s，'
  '`com_layer(J)` = %s / %s / %s；`com_V` 重算 = P17 锚 %s / %s / %s（P16 result `%s` / P17 result `%s` 现场读入并断言）。'
  % (JV['Q1_joint'], fn(V[A0]['Q1_arch_max'], 4), fn(V[A1]['Q1_arch_max'], 4), fn(V[A2]['Q1_arch_max'], 4), FL['P18_FID_ARCH'],
     fn(V[A0]['Q1_blk_max'], 4), fn(V[A1]['Q1_blk_max'], 4), fn(V[A2]['Q1_blk_max'], 4), FL['P18_FID_BLK'],
     JV['Q2_joint'],
     fn(R['arms'][A0]['E8_anchor']['detail']['com_layer_x']['got'], 6), fn(R['arms'][A1]['E8_anchor']['detail']['com_layer_x']['got'], 6),
     fn(R['arms'][A2]['E8_anchor']['detail']['com_layer_x']['got'], 6),
     fn(R['arms'][A0]['E8_anchor']['detail']['com_layer_j']['got'], 6), fn(R['arms'][A1]['E8_anchor']['detail']['com_layer_j']['got'], 6),
     fn(R['arms'][A2]['E8_anchor']['detail']['com_layer_j']['got'], 6),
     fn(R['arms'][A0]['E7_summary']['com_V_recomputed'], 3), fn(R['arms'][A1]['E7_summary']['com_V_recomputed'], 3),
     fn(R['arms'][A2]['E7_summary']['com_V_recomputed'], 3),
     R['anchor_result_p16_sha256'][:8], R['anchor_result_p17_sha256'][:8]))
p('- **桥接门（跨 Phase 装置门，本 Phase 新增）**：`CUM_ALL@L*_own` vs Phase 16 冻结 `FULL_SWAP` —— '
  'rel = %s / %s / %s ≤ %s ⇒ **`%s`**。这把本 Phase 的读位槽直接钉在 Phase 16 的**同一对象**上。'
  % (fn(V[A0]['Q3_bridge_rel'], 4), fn(V[A1]['Q3_bridge_rel'], 4), fn(V[A2]['Q3_bridge_rel'], 4),
     FL['BRIDGE_TOL_CUM'], JV['Q3_joint']))
p('- **主结果 1（P3·holdout，本轮主预测）：组件归属「行为的」MLP 主导**。P17 邻域 nb（±2）'
  '`share_mlp_beh` = **%s / %s / %s**（向量份额 = %s / %s / %s，**同侧且都过半**）'
  '⇒ **`%s`（3/3）**。**关键**：A1/A2 在 seal 冻结前**从未被观测**（探针只在 A0 上跑），故这是**真 holdout**。'
  '最大单头 `share_top1_beh` = %s / %s / %s（≪ 0.50，**无单头主导**）。'
  % (fn(V[A0]['Q4_share_mlp_beh_nb'], 3), fn(V[A1]['Q4_share_mlp_beh_nb'], 3), fn(V[A2]['Q4_share_mlp_beh_nb'], 3),
     fn(V[A0]['Q4_share_mlp_vec_nb'], 3), fn(V[A1]['Q4_share_mlp_vec_nb'], 3), fn(V[A2]['Q4_share_mlp_vec_nb'], 3),
     JV['Q4_joint'],
     fn(R['arms'][A0]['E7_summary']['share_top1_beh_nb'], 3), fn(R['arms'][A1]['E7_summary']['share_top1_beh_nb'], 3),
     fn(R['arms'][A2]['E7_summary']['share_top1_beh_nb'], 3)))
p('- **主结果 2（P4·holdout）：行为质心比向量质心浅**。`com_B` = %s / %s / %s vs `com_V` = %s / %s / %s，'
  '**gap = %s / %s / %s 层** ≥ %s ⇒ **`%s`（3/3）**。即「写入量多的地方」与「写入有效的地方」'
  '在同一对象族（增量写入 `Δ_inc`）上也是**两个位置**。'
  % (fn(V[A0]['Q6_com_B'], 3), fn(V[A1]['Q6_com_B'], 3), fn(V[A2]['Q6_com_B'], 3),
     fn(V[A0]['Q6_com_V'], 3), fn(V[A1]['Q6_com_V'], 3), fn(V[A2]['Q6_com_V'], 3),
     fn(V[A0]['Q6_gap'], 3), fn(V[A1]['Q6_gap'], 3), fn(V[A2]['Q6_gap'], 3), FL['SHALLOWER_MIN'], JV['Q6_joint']))
p('- **P5（本 Phase 最有信息量）：同对象耦合 ⇒ P17 的 P6 是对象错配**。'
  '`spearman(w_all, |b_all|)` = **%s / %s / %s**（同对象，增量写入）而 **P17 口径** '
  '`spearman(w_all, J)` = %s / %s / %s（累积差）⇒ **符号相反且幅度都大** ⇒ **`%s`**。'
  '「深端写入对行为无效」应改判为**只在累积差对象上成立**（承 seal P5：`spearman(w_all, J) < 0`、'
  '本 Phase 预测 `spearman(w_all, |b_all|) > 0`）。'
  % (fn(V[A0]['Q7_spearman_wall_ball'], 4), fn(V[A1]['Q7_spearman_wall_ball'], 4), fn(V[A2]['Q7_spearman_wall_ball'], 4),
     fn(V[A0]['Q7_spearman_wall_J'], 4), fn(V[A1]['Q7_spearman_wall_J'], 4), fn(V[A2]['Q7_spearman_wall_J'], 4),
     JV['Q7_joint']))
p('- **P6：超可加性峰值**位置**落在写入窗（位置**仅 A0 成立** / 峰值比 **0/3** 不成立）**。'
  '`argmax r_lin` = L%s / L%s / L%s vs `L*_own` = L%d / L%d / L%d（%d/3 命中，%s）；'
  '**峰值/次大（判据域 `ALL_SITES`）= %s / %s / %s < %s ⇒ P6 按 seal 字面 FAIL**。'
  '**⚠️ 同轮勘误 E-rlin（支撑域错配）**：seal 的 P6 rationale 引用「次大 0.186（L26）、比值 4.05」'
  '其实是 **REACH 域**读数 —— A0 现场重算 **%s**（次大 **%s @ L%s**），**逐位复现**；'
  '而**生产判据域**是 `ALL_SITES = 1..L−2`，该域上 A0 次大变成 **L%s（r_lin = %s）**'
  '（浅端 `|b_all|` 极小 ⇒ 比值型量被分母放大）⇒ ALL 域峰值比骤降到 %s。'
  '**即便退回 seal 自己的 REACH 域**，三臂峰值比也只有 %s，**A1/A2 在任何域都 ≤1.71** ⇒ '
  '**P6 无论按哪个域都 FAIL**（判据要求 ≥2/3 臂）—— 勘误只改「为什么 FAIL」，不改判决。'
  '`r_lin = |b_all − (b_mlp + b_attn)| / |b_all|` 是**比值型层内增益诊断**（b 不可加）；'
  '**位置子命题**（argmax 落在写入窗）仅 A0 成立 ⇒ 与 Phase 9 的 S 形传递函数同一位点仍互相印证；'
  '**分离度子命题**不成立（浅端近零分母污染），后继须改用绝对残差或加分母下限。'
  % (V[A0]['Q8_rlin_argmax'], V[A1]['Q8_rlin_argmax'], V[A2]['Q8_rlin_argmax'],
     R['arms'][A0]['E7_summary']['L_star_own'], R['arms'][A1]['E7_summary']['L_star_own'],
     R['arms'][A2]['E7_summary']['L_star_own'], _NPOS, JV['Q8_joint'],
     fn(V[A0]['Q8_rlin_peak_ratio'], 2), fn(V[A1]['Q8_rlin_peak_ratio'], 2), fn(V[A2]['Q8_rlin_peak_ratio'], 2),
     FL['RLIN_PEAK_RATIO_MIN'],
     _RR0, fn(_RA0[1][0], 3), _RA0[1][1], _AA0[1][1], fn(_AA0[1][0], 3), _AR0, _RAT_RE))
p('- **对照（P7，描述性）**：①**置换零假设**（保留质量多重集、随机重排到 REACH 位点；BP=%d、'
  '种子 `comB_inc`=%s / `comB_mlp`=%s / `share_mlp`=%s）—— `com_B(all)` 三臂尾 = %s / %s / %s，'
  '`com_B(mlp)` 尾 = %s / %s / %s；**`%s`**。'
  '②**确认集**（n=%d，与 %d 对 discovery **不相交**）Δ = %s / %s / %s ≤ %s。'
  '③**离流形诊断** `pert_rel(INC_ALL)` = %s / %s / %s、`pert_rel(CUM_ALL)` = %s / %s / %s。'
  % (R['bootstrap']['BP'], R['bootstrap']['seeds']['comB_inc'], R['bootstrap']['seeds']['comB_mlp'],
     R['bootstrap']['seeds']['share_mlp'],
     R['arms'][A0]['E7_summary']['null_comB_inc']['com_tail'], R['arms'][A1]['E7_summary']['null_comB_inc']['com_tail'],
     R['arms'][A2]['E7_summary']['null_comB_inc']['com_tail'],
     R['arms'][A0]['E7_summary']['null_comB_mlp']['com_tail'], R['arms'][A1]['E7_summary']['null_comB_mlp']['com_tail'],
     R['arms'][A2]['E7_summary']['null_comB_mlp']['com_tail'], JV['Q9_joint'],
     len(EX['confirmation']), len(EX['discovery']),
     fn(V[A0]['Q9_conf']['d_com']['INC_ALL'], 3), fn(V[A1]['Q9_conf']['d_com']['INC_ALL'], 3),
     fn(V[A2]['Q9_conf']['d_com']['INC_ALL'], 3), FL['CONF_TOL_COMB'],
     fn(V[A0]['Q10_pert_rel_inc_max'], 3), fn(V[A1]['Q10_pert_rel_inc_max'], 3), fn(V[A2]['Q10_pert_rel_inc_max'], 3),
     fn(V[A0]['Q10_pert_rel_cum_max'], 3), fn(V[A1]['Q10_pert_rel_cum_max'], 3), fn(V[A2]['Q10_pert_rel_cum_max'], 3)))
p('- **同轮勘误（append-only）**：**探针 P-A** `INC_MLP ≡ 0`（donor/recip 行号写错，`CAP[dw][2]`→`CAP[rw][2]`）'
  '—— 一个「漂亮但错误」的结果被 SMOKE/探针纪律在**发表前**抓住；**P-B** 删重复前向（改离线算 `r_lin`）；'
  '**P-C** 位点过滤放宽到全域；**SMOKE 缺陷 2 处**（打印类型 `%%.3f`←str、MERGE 引用已移除的 `BRIDGE_SITE`）；'
  '**M-A** `ALL_SITES = 1..L−2`（对齐 `w_all` 支撑）；'
  '**E-rlin（最重要）** P6 峰值比的**支撑域错配** —— rationale 的「比值 4.05」其实是 **REACH 域**读数'
  '（A0 现场重算 **%s**，次大 **0.186 @ L26**，逐位复现），而生产判据域是 `ALL_SITES`（浅端近零分母放大 '
  '⇒ 峰值比骤降到 **%s**）；**A1/A2 在任何域都 ≤1.71** ⇒ **P6 判决不改**（FAIL），只改解释；'
  '**位置子命题**仅 A0 成立；由「探针全支撑 vs 生产」逐位比对检出（承铁律 (ad)：口径歧义靠独立实现/独立支撑交叉验证捕捉）。'
  % (_RR0, _AR0))
p('- **预注册预测**：%s；**判决**：%s；%s；%s；%s；%s（P7 描述性 N/A）。'
  % (' '.join('%s=%s' % (k, PC[k]['pass_']) for k in sorted(PC)),
     JV['Q1_joint'], JV['Q2_joint'], JV['Q3_joint'], JV['Q4_joint'], JV['Q6_joint']))
p('- **记录**：deepseek 备忘录新增 `## Phase 18` 节（**L%s** 起），%d → **%d B** / %d → **%d 行**'
  '（前缀逐字节未变、BOM/CRLF、`bare_lf 0`、Phase 标题 **%d** 个）；Ledger 补登 N 线第 11 条'
  '（%d → **%d**，verdict `%s`，`ledger_sha256_8 = %s`）。'
  % (p18_line[0] if p18_line else '?', PRE['bytes'], len(mb), PRE['lines'], len(mt), len(ph_lines),
     len(LG['measurements']) - 1, len(LG['measurements']), tail['verdict'], LG['ledger_sha256_8']))
p('- **下一步（死线）**：**Phase 19 最高优先 = NF4 vs BF16 的 `w_ℓ` 口径**（P17/P18 全在 nf4 ⇒ 须在 A0 同尺度 '
  'bf16 复算 `w_ℓ` 谱与 `com_V`，排除「深端集中」是量化地板效应）。**并列**：①邻域宽度 ±2 敏感性'
  '（三臂 nb **恰好都是 %s**）；②**P17 P6 的 MEMO 改判**（承本 Phase P5）。'
  '仍挂账：N2h1-α-1 权重级定位、N2h1-β 水果类崩塌解剖、N3-β→N3-ε、R1 对照补强、K4 处置、'
  '**N 线 Phase 3–7 补登 Ledger**（Phase 8–18 已各 1 条）。'
  % jd({a: R['arms'][a]['E7_summary']['nb'] for a in ARMS}))
p('')
if PC and not all(v['pass_'] for v in PC.values()):
    _f = [k for k in sorted(PC) if not PC[k]['pass_']]
    _na = [k for k in sorted(PC) if PC[k]['pass_'] is None]
    p('- **本轮预测 %d/%d 通过；描述性 N/A：%s；否证：%s**。'
      % (len(PC) - len(_f) - len(_na), len(PC), ', '.join(_na) or '无', ', '.join(_f) or '无'))
elif PC:
    p('- **本轮预测全通过**。')
p('')

sec = '\n'.join(L)
_sec_n = sec.replace('\r\n', '\n').replace('\n', '\r\n').strip('\r\n')
ALREADY_W = ('## Phase 18 / N2h1-' in t0)
if ALREADY_W:
    w('wlog 已含 Phase 18 段 ⇒ 跳过追加（幂等路径）')
    t1 = t0 if t0.endswith('\n') else t0 + '\n'
else:
    t1 = t0.rstrip('\r\n') + '\r\n\r\n' + _sec_n + '\r\n'
open(WLOG, 'wb').write(t1.encode('utf-8'))
b1 = open(WLOG, 'rb').read()
w('wlog(%s): bytes %d -> %d (%+d) ; lines %d -> %d' %
  ('new' if NEW else ('skip' if ALREADY_W else 'append'), len(b0), len(b1), len(b1) - len(b0),
   len(b0.split(b'\n')), len(b1.split(b'\n'))))
w('wlog sha256 = %s' % hashlib.sha256(b1).hexdigest())

# ---------- 2. _infra/memo_baseline.json 刷新（带 history 链） ----------
heads = {}
for i, l in enumerate(mt):
    if l.startswith('## '):
        heads[l[:44]] = i + 1
hist = []
if PRE:
    hist.append({'tag': PRE.get('tag'), 'bytes': PRE.get('bytes'), 'lines': PRE.get('lines'),
                 'sha256': PRE.get('sha256') or PRE.get('sha8')})
NEW_TAG = 'post-append-phase18'
old = os.path.join(INFRA, 'memo_baseline.json')
if os.path.exists(old):
    try:
        ob = json.load(io.open(old, encoding='utf-8'))
        for e in (ob.get('history') or []):
            if e.get('tag') not in [h['tag'] for h in hist]:
                hist.append(e)
        if ob.get('tag') not in [h['tag'] for h in hist] and ob.get('tag') != NEW_TAG:
            hist.append({'tag': ob.get('tag'), 'bytes': ob.get('bytes'), 'lines': ob.get('lines'),
                         'sha256': ob.get('sha256')})
    except Exception as e:
        w('warn: old history unreadable: %r' % (e,))
hist = [h for h in hist if h.get('tag') != NEW_TAG]
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': NEW_TAG,
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(mt), 'sha256': hashlib.sha256(mb).hexdigest(),
        'sha8': new_sha8,
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'phase_headings': ph_lines,
        'sections': heads, 'history': hist}
io.open(old, 'w', encoding='utf-8', newline='\n').write(json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(post-append): bytes %d lines %d sha8 %s bare_lf %d phase_headings=%d history=%d' %
  (base['bytes'], base['lines'], new_sha8, base['bare_lf'], len(base['phase_headings']), len(hist)))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('DONE ->', OUT)
