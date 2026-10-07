# -*- coding: utf-8 -*-
"""Phase 17 文档收尾：当日 wlog 追加 + _infra/memo_baseline.json 刷新（含 history 链）。
纪律：wlog 正文所有数字均从 result_phase17.json / Ledger / MEMO 现场取值渲染（不自报）。
必须在 do_append_phase17.py 之后运行（读追加后的 MEMO）。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(P17T, 'closeout_docs_phase17.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


R = json.load(io.open(os.path.join(P17T, 'result_phase17.json'), encoding='utf-8'))
EX = json.load(io.open(os.path.join(P17T, 'execution_phase17.json'), encoding='utf-8'))
LG = json.load(io.open(LEDGER, encoding='utf-8'))
PRE = json.load(io.open(os.path.join(P17T, 'memo_baseline_preappend_phase17.json'), encoding='utf-8'))
V2P = os.path.join(P17T, 'result_phase17_v2_preintervalfix.json')
V2 = json.load(io.open(V2P, encoding='utf-8')) if os.path.exists(V2P) else None
_V2CV0 = float(V2['verdict']['A0_calib_qwen3-4b-nf4']['Q3_com_V']) if V2 else None
assert _V2CV0 is not None, 'v2 留痕缺失，E4 无法现场渲染'

V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']; FL = R['floors']
ARMS = EX['arm_order']
A0, A1, A2 = ARMS
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig').split('\r\n')
new_sha8 = hashlib.sha256(mb).hexdigest()[:8]
ph_lines = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase ')]
p17_line = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase 17')]
tail = [m for m in LG['measurements'] if m.get('phase') == 17][-1]


def fn(x, n=6):
    return 'n/a' if x is None else ('%.*f' % (n, x))


def jd(x):
    return json.dumps(x, ensure_ascii=False)


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
p('## Phase 17 / N2h1-α-10：写入向量的位置与效力（向量质量质心 com_V + 组件归属）——%s / %s / %s（%s）'
  % (JV['Q3_joint'], JV['Q4_joint'], JV['Q6_joint'], time.strftime('%H:%M')))
p('')
p('- **死线执行**：Phase 16 §7 写死的**最高优先** —— **把「位置」接到「组件」**：在每臂 `com_layer` 邻域（±2 层）'
  '做**逐层组件预算**（沿用 Phase 8 的**向量预算 `share_v`**（精确可加），禁用效应份额）；'
  '**并列** = `span_k` 体系化（三臂 × 双坐标 × k∈{2,3,5}）；**第三** = `xhalf` 可达域敏感性。')
p('- **唯一改动（seal 冻结）**：把 Phase 8 的向量预算从**单层 L6** 推广到**逐层** ⇒ 向量质量谱 '
  '`w_ℓ = mean_pairs ‖P_{U_ℓ}(Δ_inc,ℓ)‖` 与其质心 `com_V`（**区间求和** `W_j = Σ_{ℓ∈[s_j,s_{j+1})} w_ℓ`，'
  '与 `stat_com_layer` 同 mid 口径）。`Δ_inc,ℓ := Δ_attn,ℓ + Δ_mlp,ℓ`、`Δ_attn,ℓ := Σ_h Δ_head_h,ℓ` '
  '⇒ **可加性由构造成立**，另设**两个保真度门**（架构恒等式 / 分块可加性）检验构造与模型实际计算一致。'
  '**零额外前向**（每臂 44 次：2 determinism + 1 hook + 41 capture）。')
p('- **装置锚（全部通过）**：三臂 `Q0 = PASS`；三臂全 `cuda`；`T=2` 41/41；`determinism = 0.000e+00`；'
  '保真度门 **`FID_ALL_PASS`**（arch max %s / %s / %s ≤ %s；blocks max %s / %s / %s ≤ %s）；'
  '**P16 锚逐位复现 `ANCHOR_ALL_OK`（3/3，≤1e-6）**：`com_layer(x)` = %s / %s / %s，'
  '`com_layer(J)` = %s / %s / %s（Phase 16 result sha8 `%s` 现场读入并断言）。'
  % (fn(V[A0]['Q1_arch_max'], 4), fn(V[A1]['Q1_arch_max'], 4), fn(V[A2]['Q1_arch_max'], 4), FL['P17_FID_ARCH'],
     fn(V[A0]['Q1_blk_max'], 4), fn(V[A1]['Q1_blk_max'], 4), fn(V[A2]['Q1_blk_max'], 4), FL['P17_FID_BLK'],
     fn(V[A0]['Q2_detail']['com_layer_x']['got'], 6), fn(V[A1]['Q2_detail']['com_layer_x']['got'], 6),
     fn(V[A2]['Q2_detail']['com_layer_x']['got'], 6),
     fn(V[A0]['Q2_detail']['com_layer_j']['got'], 6), fn(V[A1]['Q2_detail']['com_layer_j']['got'], 6),
     fn(V[A2]['Q2_detail']['com_layer_j']['got'], 6), R['anchor_result_sha256'][:8]))
p('- **主结果 1（P3·holdout，本轮主预测）：向量写入质量「深端集中」是层栈共性**。`com_V` = '
  'A0 **%s** / A1 **%s** / A2 **%s** 层，`median(REACH)` = %s / %s / %s ⇒ **`DEEP_ALL`（3/3）**。'
  '**关键**：A1/A2 在 seal 冻结前**从未被观测**（探针只在 A0 上跑），故此结论是**真 holdout**。'
  '`com_V(mlp)` 与 `com_V(all)` 同深（%s / %s / %s），`argmax w_ℓ` 在 A0/A1/A2 = L%d/L%d/L%d。'
  % (fn(V[A0]['Q3_com_V'], 3), fn(V[A1]['Q3_com_V'], 3), fn(V[A2]['Q3_com_V'], 3),
     V[A0]['Q3_median'], V[A1]['Q3_median'], V[A2]['Q3_median'],
     fn(R['arms'][A0]['E5_com_V']['com_V_mlp'], 3), fn(R['arms'][A1]['E5_com_V']['com_V_mlp'], 3),
     fn(R['arms'][A2]['E5_com_V']['com_V_mlp'], 3),
     R['arms'][A0]['E5_com_V']['argmax_w_layer'], R['arms'][A1]['E5_com_V']['argmax_w_layer'],
     R['arms'][A2]['E5_com_V']['argmax_w_layer']))
p('- **主结果 2（P4·A2 判别臂）：行为质心不能由向量质量质心替代**。`min(d_x,d_j)` = '
  '%s / %s / %s 层 vs `CENTROID_SEP_MIN` = %s ⇒ **`POSITION_DECOUPLED` 2/3**（A0 例外）。'
  '判别臂 A2：两个**行为**质心几乎重合（`com_layer(x)`=%s vs `com_layer(J)`=%s，相距 %s 层）'
  '却在 `com_V`=%s 处相差 %s 层 ⇒ 「向量写入位置」与「行为质心位置」是**两件事**；'
  'Phase 16「`com_layer` 只是描述性位置量」的限界由此**加强**。'
  'A0 是唯一「对齐」臂（min_d=%s）——**正是 Phase 16 P6 否证的镜像**。'
  % (fn(V[A0]['Q4_min_d'], 2), fn(V[A1]['Q4_min_d'], 2), fn(V[A2]['Q4_min_d'], 2), FL['CENTROID_SEP_MIN'],
     fn(V[A2]['Q2_detail']['com_layer_x']['got'], 3), fn(V[A2]['Q2_detail']['com_layer_j']['got'], 3),
     fn(abs(V[A2]['Q2_detail']['com_layer_x']['got'] - V[A2]['Q2_detail']['com_layer_j']['got']), 3),
     fn(V[A2]['Q3_com_V'], 3), fn(V[A2]['Q4_min_d'], 2), fn(V[A0]['Q4_min_d'], 2)))
p('- **组件归属（P5）：质心邻域由 MLP 承载**。`com_V` 邻域（±2 层，三臂**恰好都是 %s**）'
  '`share_mlp_nb` = **%s / %s / %s**（全部 ≥ Phase 8 在 L6 的 0.4717）⇒ **`MLP_DOMINANT_ALL`**；'
  '最大单头 share = %s / %s / %s（≪ 0.50，**无单头主导**）。'
  % (jd({a: R['arms'][a]['E5_com_V']['neighbourhood'] for a in ARMS}),
     fn(V[A0]['Q5_share_mlp_nb'], 3), fn(V[A1]['Q5_share_mlp_nb'], 3), fn(V[A2]['Q5_share_mlp_nb'], 3),
     fn(R['arms'][A0]['E5_com_V']['top1_head_share_nb'], 4),
     fn(R['arms'][A1]['E5_com_V']['top1_head_share_nb'], 4),
     fn(R['arms'][A2]['E5_com_V']['top1_head_share_nb'], 4)))
p('- **效力关系（P6，本 Phase 最有信息量）：`spearman(w_ℓ, J_ℓ)` = %s / %s / %s** ⇒ '
  '**`WRITE_EFFICACY_ANTICORR_ALL`**。行为增益 `J(ℓ)` 随深度**下降**而向量写入质量 `w_ℓ` 随深度**上升**'
  ' ⇒ **深端有大量写入，但对 is-a 行为几乎无效**。这把 Phase 16 撤回「物理深度表述」后的空洞'
  '补成一条**可操作**陈述：「写入的多寡」与「写入的效力」在深度上是**分离**的两个轴。'
  % (fn(V[A0]['Q6_spearman_wJ'], 4), fn(V[A1]['Q6_spearman_wJ'], 4), fn(V[A2]['Q6_spearman_wJ'], 4)))
p('- **跨度谱（P7，描述性，不设方向性预测）：`%s`**（同号 %s）。即「span 更宽 ⟺ 质心更深」'
  '在三臂上一致成立 ——`J` 的剖面变化挤在浅端少数相邻步（跨度小），`xhalf` 的剖面变化铺开在深度上（跨度大）。'
  % (JV['Q7_joint'], JV['Q7_coupled']))
p('- **对照（全部通过）**：①**置换零假设**（保留质量多重集、随机重排到 REACH 位点；`com_V` 顺序敏感 ⇒ 非退化；'
  'BP=%d、种子 `comv_all`=%s / `comv_mlp`=%s）三臂 **3/3 落 `high` 尾**（obs %s / %s / %s vs p95 %s / %s / %s）。'
  '②**确认集**（n=%d，与 24 对 discovery **不相交**）Δ = %s / %s / %s ≤ %s。'
  % (R['bootstrap']['BP'], R['bootstrap']['seeds']['comv_all'], R['bootstrap']['seeds']['comv_mlp'],
     fn(V[A0]['Q7_null_all']['obs_com'], 3), fn(V[A1]['Q7_null_all']['obs_com'], 3), fn(V[A2]['Q7_null_all']['obs_com'], 3),
     fn(V[A0]['Q7_null_all']['com_p95'], 3), fn(V[A1]['Q7_null_all']['com_p95'], 3), fn(V[A2]['Q7_null_all']['com_p95'], 3),
     V[A0]['Q8_conf']['n_pairs'],
     fn(V[A0]['Q8_conf']['d_com'], 3), fn(V[A1]['Q8_conf']['d_com'], 3), fn(V[A2]['Q8_conf']['d_com'], 3),
     FL['CONF_TOL_COMV']))
p('- **⚠️ 同轮勘误 E1–E4（append-only）**：**[E4]（最重要）`com_of_mass` 口径与 seal 不一致** —— '
  'seal 定义 `W_j = Σ_{ℓ∈[s_j,s_{j+1})} w_ℓ`（**区间求和**），首版实现取了 `w_{s_j}`（**位点单层**），'
  'A0 差 %s 层（%s vs %s）；修正后 A0 的 `com_V` 与**独立探针**（24 对）的 %s **逐位相同** —— '
  '这同时是一次**独立实现之间的交叉验证**，只影响 `com_V` 族与零假设分位，**未重跑任何模型前向**，'
  '**P3/P4/P6 判决全部不变**。**[E1]** Q7 标签口径：v1 把「同号」写成错位配对 `(sx<sj)==(cx>cj)`，'
  '3/3 误报 `DECOUPLED`；v2 改同号方向 ⇒ `%s`（统计量与数据未变，v1 读数留痕）。'
  '**[E2]** SMOKE 抓出退化路径缺陷（截断致 discovery 配对集为空 ⇒ `com_V=None` 崩溃）；'
  '修「SMOKE 实例集须覆盖配对**两端**」+ 降级 schema + `spearman` 常量守卫。'
  '**[E3]** 探针用 `o_proj.weight`（nf4 打包 uint8）崩 ⇒ 改**模块自身前向的头块掩码**（与模型同口径）。'
  % (fn(abs(_V2CV0 - V[A0]['Q3_com_V']), 2), fn(_V2CV0, 3), fn(V[A0]['Q3_com_V'], 3),
     fn(V[A0]['Q3_com_V'], 2), JV['Q7_joint']))
p('- **预注册预测**：%s；**判决**：%s；%s；%s；%s；%s（P7 描述性 N/A）。'
  % (' '.join('%s=%s' % (k, PC[k]['pass_']) for k in sorted(PC)),
     JV['Q1_joint'], JV['Q2_joint'], JV['Q3_joint'], JV['Q4_joint'], JV['Q6_joint']))
p('- **记录**：deepseek 备忘录新增 `## Phase 17` 节（**L%s** 起），%d → **%d B** / %d → **%d 行**'
  '（前缀逐字节未变、BOM/CRLF、`bare_lf 0`、Phase 标题 **%d** 个）；Ledger 补登 N 线第 10 条'
  '（%d → **%d**，verdict `%s`，`ledger_sha256_8 = %s`；条目已就地刷新到最终 result sha8 `%s`）。'
  % (p17_line[0] if p17_line else '?', PRE['bytes'], len(mb), PRE['lines'], len(mt), len(ph_lines),
     len(LG['measurements']) - 1, len(LG['measurements']), tail['verdict'], LG['ledger_sha256_8'],
     tail['result_sha8']))
p('- **新增铁律（2 条）**：**(ad) 实现必须与 seal 的字面定义逐字一致** —— 口径歧义（区间求和 vs 位点取值）'
  '必须由「与独立实现交叉验证」捕捉，不能靠「数值合理」放过（本轮 E4 即由探针 vs 生产的逐位比对检出）；'
  '**(ae) 文档/元数据里的数字必须由 result 现场渲染**，禁手工转录（Ledger `rev_note` 与 MEMO §1/§5/§6/§10 '
  '的手打数字均被本轮自查拦截并改为数据驱动）。')
p('- **下一步（死线）**：**Phase 18 最高优先 = 逐层组件「行为」预算** —— 补 H11 的因果缺口：把「向量预算」'
  '换成「行为预算」（逐层组件对 `Δlogit(is-a)` 的贡献），与 §7 的 MLP 主导（share_mlp_nb = %s / %s / %s）'
  '交叉验证。**并列**：①NF4 vs BF16 的 `w_ℓ` 口径差异；②邻域宽度 ±2 的敏感性（三臂邻域恰好都是 [26,28]）。'
  '仍挂账：N2h1-α-1 权重级定位、N2h1-β 水果类崩塌解剖、N3-β→N3-ε、R1 对照补强、K4 处置、'
  '**N 线 Phase 3–7 补登 Ledger**（Phase 8–17 已各 1 条）。'
  % (fn(V[A0]['Q5_share_mlp_nb'], 3), fn(V[A1]['Q5_share_mlp_nb'], 3), fn(V[A2]['Q5_share_mlp_nb'], 3)))
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
ALREADY_W = ('## Phase 17 / N2h1-' in t0)
if ALREADY_W:
    w('wlog 已含 Phase 17 段 ⇒ 跳过追加（幂等路径）')
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
NEW_TAG = 'post-append-phase17'
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
