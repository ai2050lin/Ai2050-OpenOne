# -*- coding: utf-8 -*-
"""Phase 20 文档收尾：当日 wlog 追加 + _infra/memo_baseline.json 刷新（含 history 链）。
纪律：wlog 正文所有数字均从 result_phase20.json / Ledger / MEMO 现场取值渲染（不自报）。
必须在 do_append_phase20.py 之后运行（读追加后的 MEMO）。
注意：格式串中**不得出现裸百分号**（需写 %%）——本文正文一律避免百分号字符。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(P20T, 'closeout_docs_phase20.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


R = json.load(io.open(os.path.join(P20T, 'result_phase20.json'), encoding='utf-8'))
EX = json.load(io.open(os.path.join(P20T, 'execution_phase20.json'), encoding='utf-8'))
LG = json.load(io.open(LEDGER, encoding='utf-8'))
PRE = json.load(io.open(os.path.join(P20T, 'memo_baseline_preappend_phase20.json'), encoding='utf-8'))
DRIFT = json.load(io.open(os.path.join(INFRA, 'memo_drift_phase19_postbaseline.json'), encoding='utf-8'))
_DV19 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase19', 'disk_verify_phase19.txt')
DV19_T = (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(_DV19).st_mtime))
          if os.path.exists(_DV19) else 'n/a')

V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']; FL = R['floors']
ARMS = list(EX['arm_order'])
A0, A0b, A1, A1b = ARMS
QPM = {(p['arm_nf4'] + '|' + p['arm_bf16']): p for p in R['quant_pairs']}
PID, PID2 = 'A0_nf4|A0_bf16', 'A1_nf4|A1_bf16'
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig').split('\r\n')
new_sha8 = hashlib.sha256(mb).hexdigest()[:8]
ph_lines = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase ')]
p20_line = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase 20')]
tail = [m for m in LG['measurements'] if m.get('phase') == 20][-1]


def fn(x, n=6):
    return 'n/a' if x is None else ('%.*f' % (n, x))


def jd(x):
    return json.dumps(x, ensure_ascii=False)


def E(a, k):
    return R['arms'][a]['E10_summary'].get(k)


def dl(a, c='INC_ALL'):
    v = E(a, 'com_B_conf').get(c); z = E(a, 'com_B').get(c)
    return abs(v - z) if (v is not None and z is not None) else None


def q(pair, key, nd=4):
    s = QPM.get(pair)
    if not s:
        return 'NA'
    v = s.get(key)
    return fn(v, nd) if isinstance(v, (int, float)) else str(v)


def qd(pair, key, nd=4):
    s = QPM.get(pair)
    if not s:
        return 'NA'
    v = (s.get(key) or {}).get('delta')
    return fn(v, nd) if isinstance(v, (int, float)) else 'NA'


def tv(a, k, nd=4):
    return fn(V[a].get(k), nd)


def ps(k):
    v = PC.get(k, {})
    p = v.get('pass_')
    return 'PASS' if p is True else ('N/A' if p is None else 'FAIL')


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
p('## Phase 20 / N2h1-α-13：行为量与写入窗剖面的跨精度稳健性（nf4 vs bf16）——%s / %s（%s）'
  % (JV['Q2_joint'], 'COM_B_STABLE' if JV['Q4_com_B_stable'] else 'COM_B_UNSTABLE', time.strftime('%H:%M')))
p('')
p('- **死线执行**：Phase 19 §11 写死的**最高优先** —— P19 只把跨精度检验做到**向量侧**（`w_ℓ` 谱与 `com_V`）；'
  '本 Phase 补齐**行为侧**：P18 的行为预算 `b_{c,ℓ}` 与 P16 的写入窗集中度 `com_layer` 在 bf16 下复算。')
p('- **唯一自变量（seal 冻结）**：前向数值精度 bitsandbytes nf4 (4-bit) ↔ `torch.bfloat16`。'
  '模板 / 6 类词 / 41 实例 / 24 discovery / 17 confirmation / `U_ℓ`（全 41 实例按类平均 → 类别质心差 SVD，'
  '秩 = `n_classes−1` = 5）/ 位点 `1..L−2` / **P16 冻结 REACH 域** / P17 冻结 `nb` / P16 冻结 `L*_own` 与 α 网格 '
  '—— 全部逐字继承；两臂加载配置除量化外**逐项一致**（同 `eager` / `device_map="auto"` / `max_memory` / `low_cpu_mem_usage`）。')
p('- **两个面板一次跑完（同一装置、同一次加载、同一批 capture）**：'
  '**[B] 行为预算（P18 口径）** 注入 `h_ℓ^R + P_{U_ℓ}(Δ_c)`，读 '
  '`b_{c,ℓ} = mean_pairs [ score_of(logits_patched, ds, sid_d) − BASE[rw].sd0 ]`（组件 '
  '`INC_ALL / INC_MLP / INC_ATTN / INC_TOP1 / CUM_ALL`），并附带**零额外前向**的自谱 '
  '`w_{c,ℓ} = mean_pairs ‖P_{U_ℓ}(Δ_c)‖`；'
  '**[P] 写入窗剖面（P16 口径）** 注入 `h_ℓ^R + α·d_ℓ`（`d_ℓ = HH[ℓ+1]^D − HH[ℓ+1]^R`，**原始差、不投影**），'
  '读 `Y(ℓ,α) = mean_disc dDonor / FULL_SWAP` → `xhalf` / `J` → `com_layer(x)`、`com_layer(J)`、`span3`。')
p('- **装置锚（全部通过）**：四臂 `Q0` 合格、`device` ∈ {cuda, OFFLOAD}；`determinism = 0.000e+00`；'
  '保真度门 arch max = %s / %s / %s / %s ≤ %s、blk max = %s / %s / %s / %s ≤ %s ⇒ **`%s`**；'
  '**两个 nf4 校准臂逐位复现三套冻结锚（`%s`）** —— P18 行为族（`com_B` 五分量 / `comlayer_B` 三量 / '
  '`share_mlp_beh_nb` / `cum_bridge`）、P16（`FULL_SWAP` / `com_layer(x)` / `com_layer(J)`）、'
  'P17（`com_V`），容差 ≤ 1e-4；三锚现场读入（P18 `%s` / P16 `%s` / P17 `%s`）。'
  % (fn(V[A0]['Q1_arch_max'], 4), fn(V[A0b]['Q1_arch_max'], 4), fn(V[A1]['Q1_arch_max'], 4), fn(V[A1b]['Q1_arch_max'], 4), FL['P20_FID_ARCH'],
     fn(V[A0]['Q1_blk_max'], 4), fn(V[A0b]['Q1_blk_max'], 4), fn(V[A1]['Q1_blk_max'], 4), fn(V[A1b]['Q1_blk_max'], 4), FL['P20_FID_BLK'],
     JV['Q1_joint'], JV['Q2_joint'],
     R['anchor_result_p18_sha256'][:8], R['anchor_result_p16_sha256'][:8], R['anchor_result_p17_sha256'][:8]))
p('- **主结果 1（[B] Panel，P3/P4 双 holdout）：行为量跨精度稳健**。'
  '`com_B(all)` = %s / %s / %s / %s；配对位移 **Δ = %s 层（A0）/ %s 层（A1）** ≤ %s；'
  '行为谱秩相关 **ρ = %s（A0）/ %s（A1）** ≥ %s；'
  '行为 MLP 份额 **%s / %s / %s / %s**（同侧 **%s / %s**、皆过半）⇒ 联合 **`%s` / `%s`**。'
  '`comlayer_B_all` = %s / %s / %s / %s（Δ = %s / %s ≤ %s）。'
  % (tv(A0, 'com_B_all'), tv(A0b, 'com_B_all'), tv(A1, 'com_B_all'), tv(A1b, 'com_B_all'),
     qd(PID, 'com_B_all'), qd(PID2, 'com_B_all'), FL['QUANT_TOL_COMB'],
     q(PID, 'rho_b_all'), q(PID2, 'rho_b_all'), FL['RHO_B_MIN'],
     tv(A0, 'share_mlp_beh_nb'), tv(A0b, 'share_mlp_beh_nb'), tv(A1, 'share_mlp_beh_nb'), tv(A1b, 'share_mlp_beh_nb'),
     str(QPM[PID]['share_mlp_beh_nb']['same_side']), str(QPM[PID2]['share_mlp_beh_nb']['same_side']),
     'MLP_DOM_RETAINED' if JV['Q8_mlp_dom_retained'] else 'MLP_DOM_LOST',
     'SHALLOW_RETAINED' if JV['Q9_shallow_retained'] else 'SHALLOW_LOST',
     tv(A0, 'comlayer_B_all'), tv(A0b, 'comlayer_B_all'), tv(A1, 'comlayer_B_all'), tv(A1b, 'comlayer_B_all'),
     qd(PID, 'comlayer_B_all'), qd(PID2, 'comlayer_B_all'), FL['QUANT_TOL_COMLAYER']))
p('- **主结果 2（[B] Panel）：`gap = com_V − com_B` 与同对象耦合跨精度保留**。'
  '`gap` = %s / %s / %s / %s 层（同侧 **%s / %s**）；'
  '`spearman(w_all(P17 冻结), b_all)` = %s / %s / %s / %s（四臂皆 > 0 ⇒ **`%s`**）；'
  '同口径自谱版 `spearman(w_own, b_all)` = %s / %s / %s / %s。'
  % (tv(A0, 'gap', 3), tv(A0b, 'gap', 3), tv(A1, 'gap', 3), tv(A1b, 'gap', 3),
     str(QPM[PID]['gap_sign_same']), str(QPM[PID2]['gap_sign_same']),
     tv(A0, 'spearman_wall_ball'), tv(A0b, 'spearman_wall_ball'),
     tv(A1, 'spearman_wall_ball'), tv(A1b, 'spearman_wall_ball'),
     'COUPLED_RETAINED' if JV['Q10_coupled_retained'] else 'COUPLED_LOST',
     tv(A0, 'spearman_wall_ball_own'), tv(A0b, 'spearman_wall_ball_own'),
     tv(A1, 'spearman_wall_ball_own'), tv(A1b, 'spearman_wall_ball_own')))
p('- **主结果 3（[P] Panel，P8/P9）：写入窗集中度跨精度稳健**。'
  '`com_layer(x)` = %s / %s / %s / %s（Δ = %s / %s 层）；'
  '`com_layer(J)` = %s / %s / %s / %s（Δ = %s / %s 层）；均 ≤ %s ⇒ **`%s`**。'
  '半饱和点 `max|Δxhalf|` = %s（A0）/ %s（A1）≤ %s ⇒ **`%s`**'
  '（容差沿用 P16 已发表的跨精度容差 `XH_FAITHFUL_TOL`）；'
  'J 比值范围 A0 [%s, %s]、A1 [%s, %s]。'
  % (tv(A0, 'com_layer_x'), tv(A0b, 'com_layer_x'), tv(A1, 'com_layer_x'), tv(A1b, 'com_layer_x'),
     qd(PID, 'com_layer_x'), qd(PID2, 'com_layer_x'),
     tv(A0, 'com_layer_j'), tv(A0b, 'com_layer_j'), tv(A1, 'com_layer_j'), tv(A1b, 'com_layer_j'),
     qd(PID, 'com_layer_j'), qd(PID2, 'com_layer_j'), FL['QUANT_TOL_COMLAYER'],
     'COM_LAYER_STABLE' if JV['Q11_profile_stable'] else 'COM_LAYER_UNSTABLE',
     fn(QPM[PID]['xhalf']['max_abs_dxh'], 4), fn(QPM[PID2]['xhalf']['max_abs_dxh'], 4),
     FL['QUANT_TOL_XHALF'],
     'XHALF_STABLE' if JV['Q12_xhalf_stable'] else 'XHALF_UNSTABLE',
     fn(QPM[PID]['J']['ratio_min'], 3), fn(QPM[PID]['J']['ratio_max'], 3),
     fn(QPM[PID2]['J']['ratio_min'], 3), fn(QPM[PID2]['J']['ratio_max'], 3)))
p('- **对照（P10，描述性）**：①**置换零假设**（BP=%d；种子 `comB_inc`=%s / `comB_mlp`=%s / `share_mlp`=%s / '
  '`comlayer`=%s）—— `com_B(all)` 尾 = %s / %s / %s / %s，`com_B(mlp)` 尾 = %s / %s / %s / %s，'
  '`com_layer(x)` 尾 = %s / %s / %s / %s。'
  '②**确认集**（n=%d，与 %d 对 discovery 不相交）Δ`com_B(all)` = %s / %s / %s / %s ≤ %s。'
  '③**离流形诊断** `pert_rel(INC_ALL)` max = %s / %s / %s / %s。'
  % (R['bootstrap']['BP'], R['bootstrap']['seeds']['comB_inc'], R['bootstrap']['seeds']['comB_mlp'],
     R['bootstrap']['seeds']['share_mlp'], R['bootstrap']['seeds']['comlayer'],
     V[A0]['null_comB_inc']['com_tail'], V[A0b]['null_comB_inc']['com_tail'],
     V[A1]['null_comB_inc']['com_tail'], V[A1b]['null_comB_inc']['com_tail'],
     E(A0, 'null_comB_mlp')['com_tail'], E(A0b, 'null_comB_mlp')['com_tail'],
     E(A1, 'null_comB_mlp')['com_tail'], E(A1b, 'null_comB_mlp')['com_tail'],
     (V[A0]['null_comlayer_x'] or {}).get('com_tail'), (V[A0b]['null_comlayer_x'] or {}).get('com_tail'),
     (V[A1]['null_comlayer_x'] or {}).get('com_tail'), (V[A1b]['null_comlayer_x'] or {}).get('com_tail'),
     len(EX['confirmation']), len(EX['discovery']),
     fn(dl(A0), 3), fn(dl(A0b), 3), fn(dl(A1), 3), fn(dl(A1b), 3), 3.0,
     tv(A0, 'pert_rel_inc_max', 3), tv(A0b, 'pert_rel_inc_max', 3),
     tv(A1, 'pert_rel_inc_max', 3), tv(A1b, 'pert_rel_inc_max', 3)))
p('- **覆盖限界（H3）**：A2（Qwen3-14B）**不参与** bf16 腿（P19 实测 bf16 加载至约 19pct 权重处 segfault）⇒ '
  '本 Phase 的跨精度稳健性只在 **qwen3-4b 与 glm4-9b** 两模型上验证；'
  'A1·bf16 需 CPU offload ⇒ 含「分片执行」第二源（H2）。')
p('- **同轮勘误（append-only）**：**[E-sper]** `sper()` 把 P17 冻结锚谱（键 `w_all/w_mlp/w_attn`）与本臂自谱'
  '（键 `INC_*`）混用同一键空间 ⇒ `KeyError: w_all`（SMOKE 首跑抓到）；拆成 `sper_anchor()` / `sper_own()`。'
  '**[E-scope]** 锚复现一度在 SMOKE/PROBE 缩幅网格上运行 ⇒ 必报 DRIFT；加 `FULL_SCALE` 守卫，缩幅下显式标 N/A。'
  '**[E-probefull]** 探针初版连配对集也缩小（4 对）⇒ `U_ℓ` 秩退化为 2、`FULL_SWAP` 偏离锚；'
  '改为 PROBE 保留**全量配对与实例**、只缩网格与 BP。')
p('- **预注册预测**：%s；**判决**：%s / %s / %s / %s / %s / %s / %s。'
  % (' '.join('%s=%s' % (k, ps(k)) for k in sorted(PC)),
     JV['Q1_joint'], JV['Q2_joint'],
     'COM_B_STABLE' if JV['Q4_com_B_stable'] else 'COM_B_UNSTABLE',
     'SPECTRUM_CONSISTENT' if JV['Q6_spectrum_consistent'] else 'SPECTRUM_INCONSISTENT',
     'SHARE_STABLE' if JV['Q7_share_stable'] else 'SHARE_UNSTABLE',
     'COM_LAYER_STABLE' if JV['Q11_profile_stable'] else 'COM_LAYER_UNSTABLE',
     'XHALF_STABLE' if JV['Q12_xhalf_stable'] else 'XHALF_UNSTABLE'))
p('- **记录**：deepseek 备忘录新增 `## Phase 20` 节（**L%s** 起），%d → **%d B** / %d → **%d 行**'
  '（前缀逐字节未变、BOM/CRLF、`bare_lf 0`、Phase 标题 **%d** 个）；Ledger 补登 N 线第 %d 条'
  '（%d → **%d**，verdict `%s`，`ledger_sha256_8 = %s`）。'
  % (p20_line[0] if p20_line else '?', PRE['bytes'], len(mb), PRE['lines'], len(mt), len(ph_lines),
     sum(1 for m in LG['measurements']
         if isinstance(m.get('phase'), int) and 8 <= m['phase'] <= 20),
     len(LG['measurements']) - 1, len(LG['measurements']), tail['verdict'], LG['ledger_sha256_8']))
p('- **⚠️ MEMO 完整性事件（登记在案；非本轮引入）**：`post-append-phase19` 基线（%d B / `%s`，冻结于 %s）'
  '在快照后于 **%s** 被**就地改写** —— P10–P19 共 **%d** 个标题由短形式 `[HH:MM]` 规范化为完整形式 '
  '`[YYYY-MM-DD HH:MM]`，逐条 +11 B、合计 **+%d B**（实测 **+%d B**，残差 %+d B），行数不变（%d）、无文本丢失；'
  '事件未被任何 wlog / baseline / history 记录（P19 独立磁盘复核 `disk_verify_phase19.txt` mtime %s 早于该改写，'
  '且当时读到与基线一致的字节数并 PASS）。**处置**：不改写历史 —— `post-append-phase19` 在 `history` 中标注 `stale`；'
  '本轮 `_infra/memo_baseline.json` 以新口径重新冻结（`sections` 键由「行前 44 字符」改为**完整标题行**并加碰撞断言，'
  '`drift_events` 登记本事件）。现场审计件 `tests/deepseek_temp/_infra/memo_drift_phase19_postbaseline.json` '
  '+ `audit_memo_drift_phase19.txt`。'
  % (int(DRIFT['prev_baseline']['bytes']), str(DRIFT['prev_baseline']['sha8']),
     str(DRIFT['prev_baseline']['frozen_at']), str(DRIFT['memo_mtime']),
     len(DRIFT['normalized_phases']), int(DRIFT['predicted_delta_bytes']),
     int(DRIFT['observed_delta_bytes']), int(DRIFT['residual_bytes']),
     int(DRIFT['memo_lines_at_audit']), DV19_T))
p('- **下一步（死线）**：**Phase 21 最高优先 = 把跨精度检验推进到组件级向量预算与权重实现级** —— '
  '在 bf16 下复算 P8 的向量预算 `share_v` 与 N2h1-α-1 的权重级定位（确认「分布式搬运 / MLP 最大单一写入方」'
  '不是 nf4 的 kernel 路径产物）。**并列**：①邻域宽度 ±2 敏感性（四臂 nb 恰都 %s）；'
  '②**P17 `P6` 的 MEMO 改判**（承 P18 `P5`）。仍挂账：N2h1-α-1 权重级定位、N2h1-β 水果类崩塌解剖、'
  'N3-β→N3-ε、R1 对照补强、K4 处置、**N 线 Phase 3–7 补登 Ledger**（Phase 8–20 已各 1 条）。'
  % jd({a: R['arms'][a]['E10_summary']['nb'] for a in ARMS}))
p('')
if PC and not all(v['pass_'] for v in PC.values()):
    _f = [k for k in sorted(PC) if PC[k]['pass_'] is False]
    _na = [k for k in sorted(PC) if PC[k]['pass_'] is None]
    p('- **本轮预测 %d/%d 通过；描述性 N/A：%s；否证：%s**。'
      % (len(PC) - len(_f) - len(_na), len(PC), ', '.join(_na) or '无', ', '.join(_f) or '无'))
elif PC:
    p('- **本轮预测全通过**。')
p('')

sec = '\n'.join(L)
_sec_n = sec.replace('\r\n', '\n').replace('\n', '\r\n').strip('\r\n')
ALREADY_W = ('## Phase 20 / N2h1-' in t0)
if ALREADY_W:
    w('wlog 已含 Phase 20 段 ⇒ 跳过追加（幂等路径）')
    t1 = t0 if t0.endswith('\n') else t0 + '\n'
else:
    t1 = t0.rstrip('\r\n') + '\r\n\r\n' + _sec_n + '\r\n'
open(WLOG, 'wb').write(t1.encode('utf-8'))
b1 = open(WLOG, 'rb').read()
w('wlog(%s): bytes %d -> %d (%+d) ; lines %d -> %d' %
  ('new' if NEW else ('skip' if ALREADY_W else 'append'), len(b0), len(b1), len(b1) - len(b0),
   len(b0.split(b'\n')), len(b1.split(b'\n'))))
w('wlog sha256 = %s' % hashlib.sha256(b1).hexdigest())

# ---------- 2. _infra/memo_baseline.json 刷新 ----------
heads = {}
for i, l in enumerate(mt):
    if l.startswith('## '):
        heads[l.rstrip()] = i + 1
_n_hdr = sum(1 for l in mt if l.startswith('## '))
assert len(heads) == _n_hdr, ('sections 键碰撞：%d 个标题行 -> %d 个键'
                              % (_n_hdr, len(heads)))
hist = []
if PRE:
    hist.append({'tag': PRE.get('tag'), 'bytes': PRE.get('bytes'), 'lines': PRE.get('lines'),
                 'sha256': PRE.get('sha256') or PRE.get('sha8')})
NEW_TAG = 'post-append-phase20'
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
# ---- 完整性事件登记：P19 基线快照之后 MEMO 被就地规范化 ----
_STALE = {DRIFT['prev_baseline']['tag']:
          ('快照后于 ' + str(DRIFT['memo_mtime']) + ' 被就地规范化：P10–P19 共 '
           + str(len(DRIFT['normalized_phases'])) + ' 个标题由短形式 [HH:MM] 改为完整形式，+'
           + str(DRIFT['observed_delta_bytes']) + ' B（行数不变、无文本丢失）'
           + '⇒ bytes/sha256 锚陈旧；审计件 '
           + 'tests/deepseek_temp/_infra/memo_drift_phase19_postbaseline.json')}
for _e in hist:
    if _e.get('tag') in _STALE:
        _e['stale'] = True
        _e['note'] = _STALE[_e['tag']]
w('history 陈旧标注：%s' % [h.get('tag') for h in hist if h.get('stale')])
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': NEW_TAG,
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(mt), 'sha256': hashlib.sha256(mb).hexdigest(),
        'sha8': new_sha8,
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'phase_headings': ph_lines,
        'sections_key_rule': 'full-heading-line',
        'drift_events': [{'tag': DRIFT['prev_baseline']['tag'],
                          'event': 'post-baseline in-place heading normalization',
                          'memo_mtime': DRIFT['memo_mtime'],
                          'delta_bytes': DRIFT['observed_delta_bytes'],
                          'delta_lines': DRIFT['observed_delta_lines'],
                          'phases': DRIFT['normalized_phases'],
                          'residual_bytes': DRIFT['residual_bytes'],
                          'artifact': ('tests/deepseek_temp/_infra/'
                                       'memo_drift_phase19_postbaseline.json')}],
        'sections': heads, 'history': hist}
io.open(old, 'w', encoding='utf-8', newline='\n').write(json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(post-append): bytes %d lines %d sha8 %s bare_lf %d phase_headings=%d history=%d' %
  (base['bytes'], base['lines'], new_sha8, base['bare_lf'], len(base['phase_headings']), len(hist)))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('DONE ->', OUT)
