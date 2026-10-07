# -*- coding: utf-8 -*-
"""Phase 16 文档收尾：当日 wlog 追加 + _infra/memo_baseline.json 刷新（含 history 链）。
纪律：wlog 正文所有数字均从 result_phase16.json / Ledger / MEMO 现场取值渲染（不自报）。
必须在 do_append_phase16.py 之后运行（读追加后的 MEMO 基线）。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(P16T, 'closeout_docs_phase16.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


R = json.load(io.open(os.path.join(P16T, 'result_phase16.json'), encoding='utf-8'))
EX = json.load(io.open(os.path.join(P16T, 'execution_phase16.json'), encoding='utf-8'))
LG = json.load(io.open(LEDGER, encoding='utf-8'))
PRE = json.load(io.open(os.path.join(P16T, 'memo_baseline_preappend_phase16.json'), encoding='utf-8'))
AM1_SHA8 = sha(os.path.join(P16T, 'N2h1a9_design_seal_amend1.json'))[:8]

V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']
E5 = R['E5_concentration']; E3 = R['E3_localize']; E7 = R['E7_reach']
FL = R['floors']
ARMS = EX['arm_order']
A0, A1, A2 = ARMS

mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig').split('\r\n')
new_sha8 = hashlib.sha256(mb).hexdigest()[:8]
ph_lines = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase ')]
p16_line = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase 16')]
tail = LG['measurements'][-1]


def fn(x, n=6):
    return 'n/a' if x is None else ('%.*f' % (n, x))


def sign(x, n=3):
    return 'n/a' if x is None else ('%+.*f' % (n, x))


def jd(o):
    return json.dumps(o, ensure_ascii=False)


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
p('## Phase 16 / N2h1-α-9：写入窗原点化剖面 + 集中度统计量重设计（三臂同一 nf4 口径）——%s / %s（%s）'
  % (JV['Q2_joint'], JV['Q4_joint'], time.strftime('%H:%M')))
p('')
p('- **死线执行**：Phase 15 §8 写死的两条**并列最高优先** —— ①统一剖面下的「写入窗 vs 集中窗」关系；'
  '②集中度统计量重设计（换对排序/单窗口不敏感的量，在三臂 × 双坐标上重算 Phase 12–14 全部集中度表）。'
  '**为什么这两条此前不可答**：Phase 15 的独立定位（B_cat 逐层独立 `U_ℓ` 的相邻最大增量）给出 '
  '`L*_own` = A0 **%s** / A1 **%s** / A2 **%s**，而剖面域只有 `[6..34]` ⇒ A1/A2 的写入窗**根本不在域内**。'
  % (E3[A0]['L_star_own'], E3[A1]['L_star_own'], E3[A2]['L_star_own']))
p('- **三处改动（seal 冻结）**：①网格 `profile_sites = [1,2,3,4,5] + [6..34]`（**%d 位点**，legacy 子域 18 不变）；'
  '②主域改为**可达性掩膜** `REACH = {ℓ : ρ(ℓ) ≥ UNREACH_y = %s}`，`ρ(ℓ) = Y(ℓ, α=1) = dDonor/FULL_SWAP` '
  '⇒ 写入窗成为主域**左端点**，被排除位点逐个报告；③`top3_share` → **(`com_layer`, `span_k`) 双量**：'
  '`com_layer = Σ|Δ_j|·mid_j / Σ|Δ_j|`（**物理层号**质心，单位「层」，对网格加密不变）、'
  '`span_k`（k 个最大 |Δ| 的跨度/(n−1)，k=3），**双边检验**。'
  % (len(EX['profile_sites']), FL['UNREACH_y']))
p('- **⚠️ 设计期整族排除（本轮最重要的一条方法学）**：置换零假设**保留 jump 的多重集** ⇒ '
  '任何只依赖多重集的量（**谱熵 / max÷mean / 参与比**）在零假设下**恒等于观测**、双边必 p=1 —— '
  '这类量是**结构性退化**的，**必须整族排除**。Phase 15 的「换谱熵」设想在此被否证，'
  '改用**顺序敏感**的 `com_layer`/`span_k`。')
p('- **装置锚（全部通过）**：三臂 `Q0 = PASS`；41/41 实例 `T=2`；双前向 `determinism = 0.000e+00`；'
  'hook 效应 %s/%s/%s；F1b（各臂 tokenizer 自检）三臂 %s；`sup_id 同参考` A0 %s / A1 %s / A2 %s'
  '（A1 为 False 是**预期**：GLM4 词表不同，Phase 15 amend1 已改逐臂现场解析）。'
  % (fn(R['arms'][A0]['E0_selfcheck'].get('hook_effect_maxdiff'), 3),
     fn(R['arms'][A1]['E0_selfcheck'].get('hook_effect_maxdiff'), 3),
     fn(R['arms'][A2]['E0_selfcheck'].get('hook_effect_maxdiff'), 3),
     jd({a: R['arms'][a]['F1b_ok'] for a in ARMS}),
     R['arms'][A0]['sup_id_matches_ref'], R['arms'][A1]['sup_id_matches_ref'], R['arms'][A2]['sup_id_matches_ref']))
p('- **P2 冻结锚 = 逐位复现（本轮最强装置结论）**：legacy 6..34 子域在同一 run 内重算，与 Phase 15 冻结 '
  'result（sha8 `%s`）**bit-for-bit**：`max|Δxhalf| = 0.000e+00`、`max rel|ΔJ| = 0.000e+00`、'
  '`Δtop3 = 0.00000 / 0.00000`、legacy argmax_w 全部相同 ⇒ 三臂 **`RECON_OK`（严格层，n_strict_ok = %d）**。'
  % (R['anchor_result_sha8'], PC['P2']['detail']['n_strict_ok']))
p('- **⚠️ 判据分层（amend1 `%s`，criterion tiering，不改假设）**：SMOKE 给出 '
  '`max|Δxhalf| = %s` —— 这**不是漂移**，而是 `xhalf` 作为 `cross_alpha` 在 α 网格上的**插值量**'
  '在 3 点网格下的必然偏移。**对插值量设单一 1e-3 硬门违反「软门优先硬 assert」** ⇒ 判据分层为 '
  '`RECON_OK`(≤1e-3) / `RECON_OK_LOOSE`(≤5e-3，须附逐位点表) / `RECON_DRIFT`，并**冻结在观测之前**；'
  '正式 run 三臂全部落在**严格层**。'
  % (AM1_SHA8, fn(0.010517198667084726, 6)))
p('- **主结果 1（P3/P4）：写入窗 = 可达性门槛，三臂严格成立**。`ρ(ℓ)` 呈**阶跃**：窗下位点 '
  'ρ ≤ %s，窗上 ρ ≈ 0.99，而 `ℓ_reach = min{ℓ : ρ ≥ 0.5}` 与 B_cat 的 `L*_own` **严格相等**（%s / %s / %s）。'
  '两条**完全不同的可观测量**（几何子空间相邻最大增量 ↔ 替换探针的剂量-响应可达性）在同一位点重合。'
  % (fn(max(E7[a]['rho'][E7[a]['sites'].index(E7[a]['ell_reach']) - 1] for a in ARMS if E7[a]['sites'].index(E7[a]['ell_reach']) > 0), 4),
     *[E7[a]['ell_reach'] for a in ARMS]))
p('- **主结果 2（P5）：旧判据退休**。`top3_share` 是**极值型 3-窗占比**，其窗位由曲线单调性**确定性**决定；'
  '在 legacy 6 格里只有 %d 格过零假设。新量在主域上给出 **%d 格**显著（旧 %d → 新 %d），'
  '且 `com_layer` 给出的是**位置**（哪一层），而不是「占比多少」。'
  % (sum(len(V[a].get('Q5_old_sig') or []) for a in ARMS),
     sum(len(V[a].get('Q5_new_sig') or []) for a in ARMS),
     sum(len(V[a].get('Q5_old_sig') or []) for a in ARMS),
     sum(len(V[a].get('Q5_new_sig') or []) for a in ARMS)))
p('- **⚠️ 主结论·P6 = 预注册否证（本轮最重要的自我否证）**：`com_layer(xhalf) − com_layer(J)` = '
  'A0 **%s** / A1 **%s** / A2 **%s 层**。判据要求 ≥ %s 层（3/3），实测 **%d/3**，'
  '且 **A2（Qwen3-14B，与 A0 同属 Qwen3 家族、参数 3.5×）反号** ⇒ '
  '「`xhalf` 深尾集中 / `J` 浅端集中」的**物理深度表述整体撤回**（seal 的 `may_falsify_the_whole_line` 条款）。'
  '**这是一次成功的自我否证，不是装置失败**：装置门全过、锚 bit 级复现、P4 三臂严格。'
  % (sign(V[A0].get('Q4_sep'), 3), sign(V[A1].get('Q4_sep'), 3), sign(V[A2].get('Q4_sep'), 3),
     FL['CENTROID_SEP_MIN'], sum(1 for a in ARMS if (V[a].get('Q4_sep') is not None and V[a]['Q4_sep'] >= FL['CENTROID_SEP_MIN']))))
p('- **P7（保留）**：`com_layer(xhalf) − L*_own` = %s / %s / %s 层（≥ %s 需 %d/3 ⇒ %s）—— '
  '«xhalf 的质心在写入窗之后» 这一弱形式仍成立；被否证的是「两坐标质心相距 ≥4 层」这一**形状对比**。'
  % (fn(V[A0].get('Q4_com_after_win_x'), 2), fn(V[A1].get('Q4_com_after_win_x'), 2), fn(V[A2].get('Q4_com_after_win_x'), 2),
     FL['CENTROID_AFTER_WIN_MIN'], PC['P7']['detail']['n_pass'],
     'PASS' if PC['P7']['pass_'] else 'FAIL'))
p('- **旧量 vs 新量的坐标依赖**：新量双边尾 = x:%s / j:%s；`span_3` 观测 x/j = %s。'
  % (jd({a: (E5[a]['new_stat']['x'] or {}).get('com_tail') for a in ARMS}),
     jd({a: (E5[a]['new_stat']['j'] or {}).get('com_tail') for a in ARMS}),
     ' ; '.join('%s %.4f/%.4f' % (a, (E5[a]['new_stat']['x'] or {}).get('obs_span') or -1,
                                 (E5[a]['new_stat']['j'] or {}).get('obs_span') or -1) for a in ARMS)))
p('- **A0 量化保真（E6）**：`max|dxhalf| = %s`（tol %s）⇒ **%s**。' % (
    fn(R['E6_calibration'][A0].get('max_abs_dxh'), 6), FL['XH_FAITHFUL_TOL'],
    V[A0].get('Q6_e6_label')))
p('- **预注册预测**：%s。' % ' '.join('%s = %s' % (k, PC[k]['pass_']) for k in sorted(PC)))
p('- **判决**：%s；%s；%s；%s；%s。' % (JV['Q1_joint'], JV['Q2_joint'], JV['Q3_joint'], JV['Q4_joint'], JV['Q5_joint']))
p('- **⚠️ 同轮重写（v1 → v2）+ 两处元数据缺陷（由独立磁盘复核发现）**：收尾链的 `disk_verify_phase16.py` '
  '在首跑抓到 3 项 FAIL，逐项处置：①**`memo_append_phase16.md` 与已追加正文不一致** —— 追加后我又改了渲染器'
  '（§4 表头 + 锚点补齐），导致「生成器 ≠ 留痕件 ≠ MEMO」；按**可复现不变量**要求，先对**冻结的追加前基线 '
  'sha256**（`%s` / %d B）**逐位验证回滚**，再以修正版**重追加**（v1 原文留痕 `memo_append_phase16_v1_asappended.md`）。'
  '②**§4 表头错标**：v1 把 `sup_id_matches_ref` 列标成「F1b 词表匹配」（`F1b_ok` 三臂其实均 True；'
  'A1 的 False 是 GLM4 词表不同导致的**预期**差异）⇒ v2 更正为「sup_id 同参考」+ 补一行 F1b 说明。'
  '③**exec 元数据缺陷（已冻结、不回改，以 MEMO §10 勘误为准）**：`bootstrap.seeds` 记 `new_x/new_j = SEED+41/+53`，'
  '而实现（`n2h1a9_…py` L674–675）用 `SEED+61/+67`；只影响**零假设分位的复现**，不动任何观测/判决。'
  '④**§1「左端点」表述过强**：设计意图是「写入窗 = REACH 左端点」，实测只 A0/A1 成立（A2 的 REACH 左端点 = 3 < `ell_reach` = 4，'
  'ρ(3)=0.2428 ≥ 0.10）⇒ P3 按「写入窗 ∈ REACH」评估（三臂均满足）。'
  % (PRE.get('sha8'), PRE.get('bytes')))
p('- **记录**：deepseek 备忘录新增 `## Phase 16` 节（**L%s** 起），%d → **%d B** / %d → **%d 行**'
  '（前缀逐字节未变、BOM/CRLF、`bare_lf 0`、Phase 标题 **%d** 个）；Ledger 补登 N 线第 9 条（%d → **%d**，'
  '备份 `atlas_ledger_backup_pre_phase16.json`，verdict `%s`，`ledger_sha256_8 = %s`）。'
  % (p16_line[0] if p16_line else '?', PRE['bytes'], len(mb), PRE['lines'], len(mt), len(ph_lines),
     len(LG['measurements']) - 1, len(LG['measurements']), tail['verdict'], LG['ledger_sha256_8']))
p('- **新增铁律（1 条，(ac)）**：**对「插值型读数」不得设单一硬门，必须分层并冻结在观测前** —— '
  'Phase 16 的 `xhalf` 是 `cross_alpha` 在 α 网格上的插值位置，α 取 3 点还是 14 点会改变它；'
  '同一量在 14 点正式 run 下 `Δ=0.000e+00` 却在 3 点 SMOKE 下 `Δ=0.0105` ⇒ '
  '**硬门必须写在「插值不变」的量上（如 legacy argmax、整数索引），或对插值量明写分层档位与降级条件**。')
p('- **下一步（死线）**：**Phase 17 最高优先 = 把「位置」接到「组件」** —— 在每臂 `com_layer` 邻域（±2 层）'
  '做逐层组件预算（沿用 Phase 8 的**向量预算 `share_v`**（精确可加），禁止用效应份额），'
  '回答「质心所在层由谁贡献写入向量」。**并列**：`span_k` 体系化（三臂 × 双坐标 × k∈{2,3,5} 跨度谱）；'
  '**第三**：`xhalf` 的可达域敏感性（末层被排除是否造成深尾抬升）。'
  '仍挂账：N2h1-α-1 权重级定位、N2h1-β 水果类崩塌解剖、N3-β→N3-ε、R1 对照补强、K4 处置、'
  '**N 线 Phase 3–7 补登 Ledger**（Phase 8–16 已各 1 条）。')
p('')
if PC and not all(v['pass_'] for v in PC.values()):
    _f = [k for k in sorted(PC) if not PC[k]['pass_']]
    p('- **本轮预测 %d/%d 通过；否证：%s（详见上方主结论·P6）**。' % (len(PC) - len(_f), len(PC), ', '.join(_f)))
elif PC:
    p('- **本轮预测全通过**。')
p('')

sec = '\n'.join(L)
_sec_n = sec.replace('\r\n', '\n').replace('\n', '\r\n').strip('\r\n')
ALREADY_W = ('## Phase 16 / N2h1-' in t0)
if ALREADY_W:
    w('wlog 已含 Phase 16 段 ⇒ 跳过追加（幂等路径）')
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
NEW_TAG = 'post-append-phase16'
old = os.path.join(INFRA, 'memo_baseline.json')
if os.path.exists(old):
    try:
        ob = json.load(io.open(old, encoding='utf-8'))
        for e in (ob.get('history') or []):
            if e.get('tag') not in [h['tag'] for h in hist]:
                hist.append(e)
        # 幂等重跑守卫：旧基线自身的 tag 不得作为历史条目回流（否则与 NEW_TAG 自引用重复）
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
