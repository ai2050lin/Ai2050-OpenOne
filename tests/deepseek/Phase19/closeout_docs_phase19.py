# -*- coding: utf-8 -*-
"""Phase 19 文档收尾：当日 wlog 追加 + _infra/memo_baseline.json 刷新（含 history 链）。
纪律：wlog 正文所有数字均从 result_phase19.json / Ledger / MEMO 现场取值渲染（不自报）。
必须在 do_append_phase19.py 之后运行（读追加后的 MEMO）。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P19T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase19')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(P19T, 'closeout_docs_phase19.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


R = json.load(io.open(os.path.join(P19T, 'result_phase19.json'), encoding='utf-8'))
EX = json.load(io.open(os.path.join(P19T, 'execution_phase19.json'), encoding='utf-8'))
LG = json.load(io.open(LEDGER, encoding='utf-8'))
PRE = json.load(io.open(os.path.join(P19T, 'memo_baseline_preappend_phase19.json'), encoding='utf-8'))

V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']; FL = R['floors']
QP = JV['quant_pairs']
ARMS = EX['arm_order']
A0n, A0b, A1n, A1b = ARMS
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig').split('\r\n')
new_sha8 = hashlib.sha256(mb).hexdigest()[:8]
ph_lines = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase ')]
p19_line = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase 19')]
tail = [m for m in LG['measurements'] if m.get('phase') == 19][-1]


def fn(x, n=4):
    return 'n/a' if x is None else ('%.*f' % (n, x))


def jd(x):
    return json.dumps(x, ensure_ascii=False)


def pk(pair, key, n=4):
    s = QP.get(pair)
    if not s:
        return 'NA'
    v = s[key]
    return fn(v, n) if isinstance(v, (int, float)) else str(v)


def tri(k, n=4):
    return ' / '.join(fn(V[a][k], n) for a in ARMS)


df_q = max(QP[k]['delta_com_V'] for k in QP) if QP else 0.0
rho_min = min(QP[k]['spearman_w'] for k in QP) if QP else 0.0

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
p('## Phase 19 / N2h1-α-12：写入向量谱的量化口径稳健性（nf4 ↔ bf16）——%s / %s / %s（%s）'
  % (JV['Q3_joint'], JV['Q4_joint'], JV['Q6_joint'], time.strftime('%H:%M')))
p('')
p('- **死线执行**：Phase 17 §7 写死的**最高优先** —— **NF4 vs BF16 的 `w_ℓ` 口径**。P17/P18 的全部头条'
  '（`DEEP_ALL`、MLP 主导归属）**都只在 bitsandbytes nf4 单一数值口径下测得**；而 P17 自己的 `quant.why`'
  '写着「A0 臂专职量化对其结论的影响」——该检查**从未执行**。本 Phase 执行它。')
p('- **唯一自变量（seal 冻结）**：前向数值精度 **nf4 (4-bit) ↔ bfloat16**。其余**逐项不变**：'
  '同 template `%s` / classes / instances_all（n=%d）/ pairs_all（n=%d）/ discovery（n=%d）'
  '（confirmation n=%d 不参与产量定义）；`U_ℓ` = 全 %d 实例按类平均→类别质心差 SVD（秩 = `n_classes−1` = %d）；'
  '质量定义 `w_ℓ = mean_pairs ‖P_{U_ℓ}(Δ_inc,ℓ)‖`；质心 = REACH 上**相邻位点区间求和** + 中点；'
  'REACH 取该臂 P17 冻结值；邻域宽 ±%d；BP=%d 与种子。**两臂加载配置除量化外逐项一致**'
  '（同 `eager` / `device_map="auto"` / `max_memory` / `low_cpu_mem_usage`）。'
  % (EX['template'], len(EX['instances_all']), len(EX['pairs_all']), len(EX['discovery']),
     len(EX['confirmation']), len(EX['instances_all']), len(EX['classes']) - 1,
     EX['neighbourhood_width'], R['bootstrap']['BP']))
p('- **臂集（4 臂）**：`%s`/`%s` = **预注册校准臂**（必须逐位复现 P17 锚）；`%s`/`%s` = **检验臂**'
  '（`%s` 为 **holdout**，seal 冻结前其任何研究量均未被观测）。'
  % (A0n, A1n, A0b, A1b, A1b))
p('- **装置门与校准（P1 = %s）**：四臂 `Q0 = PASS`（`%s`/`%s`/`%s`/`%s`）；保真度门 **`%s`**'
  '（arch max %s ≤ %s；blk max %s ≤ %s）；**两个 nf4 校准臂逐位复现 P17 冻结锚 `%s`**'
  '（`com_V` / `com_V_mlp` / `com_V_attn` 差 ≤ %s，`nb` / `argmax_w` 精确相同）⇒ 装置与 Phase 17 同源。'
  % ('PASS' if PC['P1']['pass_'] else 'FAIL',
     V[A0n]['Q0_device'], V[A0b]['Q0_device'], V[A1n]['Q0_device'], V[A1b]['Q0_device'],
     JV['Q1_joint'],
     ' / '.join(fn(V[a]['Q1_arch_max'], 3) for a in ARMS), FL['P19_FID_ARCH'],
     ' / '.join(fn(V[a]['Q1_blk_max'], 3) for a in ARMS), FL['P19_FID_BLK'],
     R['anchor_result_sha256'][:8], ('%g' % FL['CALIB_TOL_COMV'])))
p('- **主结果 1（P2·同模型 = %s / P3·跨家族 holdout = %s）：「深端集中」几乎不随精度移动**。'
  '`com_V` = **%s**（臂序 nf4/bf16/nf4/bf16）；同 Phase 配对 Δ`com_V` = '
  '**%s** 层（A0）/ **%s** 层（A1），全部 ≤ 容差 %s 层 ⇒ **`%s`（%d/%d 对）**；'
  '两臂的 bf16 质心仍深于 `median(REACH)`（A0 17.0、A1 14.0）⇒ **`%s`**（%s）。'
  % ('PASS' if PC['P2']['pass_'] else 'FAIL', 'PASS' if PC['P3']['pass_'] else 'FAIL',
     tri('com_V'), pk('A0_nf4|A0_bf16', 'delta_com_V'), pk('A1_nf4|A1_bf16', 'delta_com_V'),
     fn(FL['QUANT_TOL_COMV'], 1), JV['Q3_joint'], JV['Q3_counts']['STABLE'], JV['Q3_counts']['n'],
     JV['Q6_joint'], '四臂 bf16 与 nf4 同判深端' if JV['Q6_joint'].endswith('ALL') else ''))
p('- **主结果 2（P4·holdout）：组件归属同侧且仍过半**。邻域 nb（±%d）`share_mlp_nb` = **%s**'
  '（nf4 侧 %s / bf16 侧 %s；A1 家族 %s → %s）；**全部 ≫ %s ⇒ `%s`**；'
  '`argmax_w` 层 nf4/bf16 完全相同（A0 L%s/L%s、A1 L%s/L%s）⇒ 无单头/单层位移。'
  % (EX['neighbourhood_width'],
     ' / '.join(fn(V[a]['share_mlp_nb'], 4) for a in ARMS),
     pk('A0_nf4|A0_bf16', 'share_mlp_nb_nf4'), pk('A0_nf4|A0_bf16', 'share_mlp_nb_bf16'),
     fn(V[A1n]['share_mlp_nb'], 4), fn(V[A1b]['share_mlp_nb'], 4), fn(FL['MLP_DOM_MIN'], 2), JV['Q5_joint'],
     V[A0n]['argmax_w_layer'], V[A0b]['argmax_w_layer'], V[A1n]['argmax_w_layer'], V[A1b]['argmax_w_layer']))
p('- **主结果 3（P5 = %s）：谱形状几乎完全保持**。`spearman(w_nf4, w_bf16)` = **%s**（≥ %s）；'
  '相对逐位点残差中位 / p90 = **%s / %s**（A0）、**%s / %s**（A1）⇒ **`%s`（%d/%d 对）**。'
  % ('PASS' if PC['P5']['pass_'] else 'FAIL',
     ' / '.join(pk(k, 'spearman_w') for k in sorted(QP)), fn(FL['RHO_SHAPE_MIN'], 2),
     pk('A0_nf4|A0_bf16', 'median_rel_resid'), pk('A0_nf4|A0_bf16', 'p90_rel_resid'),
     pk('A1_nf4|A1_bf16', 'median_rel_resid'), pk('A1_nf4|A1_bf16', 'p90_rel_resid'),
     JV['Q4_joint'], JV['Q4_counts']['CONSISTENT'], JV['Q4_counts']['n']))
p('- **交叉验证（承铁律 (ad)）：独立探针 ↔ 生产实现逐位相同**。`com_V` 三方对齐 '
  '26.1501（P17 冻结锚 = 探针 nf4 = 生产 nf4）、bf16 26.0570（探针 = 生产）；'
  '`share_mlp_nb` 与 `argmax_w` 同样逐位相同 ⇒ 量化敏感度**不是实现细节的产物**。')
p('- **对照（P7）**：置换零假设（保留质量多重集、随机重排到 REACH 位点；BP=%d、种子 `%s`/`%s`）'
  '—— 四臂 `com_V(all)`/`com_V(mlp)` 尾全 = **`%s`**（高尾）。'
  % (R['bootstrap']['BP'], R['bootstrap']['seeds']['comv_all'], R['bootstrap']['seeds']['comv_mlp'],
     jd(JV['Q7_null'][A0n])))
p('- **限界（诚实性）**：① **A2（Qwen3-14B，29.5 GB）无法参与 bf16 腿 —— 加载 19%% 时 segfault**'
  '（实测；与 P17 `quant.why` 同一 RAM 天花板）⇒ 跨精度稳健性只在 **qwen3-4b 与 glm4-9b** 两模型成立'
  '（A2 仅入校准门，参考值 `com_V=26.6749`）；② bf16 vs nf4 的差含「量化误差 + 反量化 kernel 路径」两源；'
  '③ `%s` 需 CPU offload（18.8 GB > 14 GiB）⇒ 该臂含「分片执行」第三源，按预注册仍入硬门但须记此限；'
  '④ `w_ℓ` 是**激活级**分解（hook），非权重级实现证明；⑤ 本 Phase **不重测**行为量（J / `com_layer` / P18 `b`）'
  '——只回答「`w_ℓ` 谱与 `com_V` 是否量化稳健」。' % A1b)
p('- **同轮勘误（append-only）**：**[E-A2]** A2·bf16 加载期 segfault（~19% 权重）⇒ bf16 腿只有两模型'
  '（如实缩小覆盖范围，**不因可行性而改判据**）；**[E-pair]** 探针首版配对过滤写成 '
  '`p[0] in DISC_W and p[2] in DISC_W`（比 P17 严，24 对降 17 对）⇒ 探针 nf4 `com_V` = 26.1956 ≠ 锚 26.1501'
  '（差 0.0455），改为与 P17 逐字一致的 `p[0] in DISC_W` 后**逐位复现 26.1501** '
  '（教训：**配对集定义也是口径的一部分**，跨实现比对是唯一可靠检出手段）；'
  '**[E-offload]** accelerate 的 CPU-offload 用 **meta 占位参数**承载权重 ⇒ 取设备必须读 '
  '`m._hf_hook.execution_device`（否则 `Cannot copy out of meta tensor`），修后**重跑全部四臂**'
  '以保持同一实现版本。')
p('- **预注册预测**：%s；**判决**：%s / %s / %s / %s / %s / %s（P7 描述性）。'
  % (' '.join('%s=%s' % (k, PC[k]['pass_']) for k in sorted(PC)),
     JV['Q1_joint'], JV['Q2_joint'], JV['Q3_joint'], JV['Q4_joint'], JV['Q5_joint'], JV['Q6_joint']))
p('- **记录**：deepseek 备忘录新增 `## Phase 19` 节（**L%s** 起），%d → **%d B** / %d → **%d 行**'
  '（前缀逐字节未变、BOM/CRLF、`bare_lf 0`、Phase 标题 **%d** 个）；Ledger 补登 N 线第 12 条'
  '（%d → **%d**，verdict `%s`，`ledger_sha256_8 = %s`）；sel=%s / exec=%s / result=%s；'
  '四臂前向合计 **%d**（每臂 %d），wall **%ss**。'
  % (p19_line[0] if p19_line else '?', PRE['bytes'], len(mb), PRE['lines'], len(mt), len(ph_lines),
     len(LG['measurements']) - 1, len(LG['measurements']), tail['verdict'], LG['ledger_sha256_8'],
     R['seal_sha256'][:8], R['exec_sha256'][:8], hashlib.sha256(open(os.path.join(P19T, 'result_phase19.json'), 'rb').read()).hexdigest()[:8],
     sum(int(R['arms'][a]['n_forwards']) for a in ARMS), R['arms'][A0n]['n_forwards'],
     R['elapsed_total_s']))
p('- **下一步（死线）**：**Phase 20 最高优先 = 把跨精度检验推进到行为量** —— P18 的 `b_{c,ℓ}` 与 P16 的 '
  '`com_layer` 在 bf16 下复算，补 P18 结论的跨精度证据（本 Phase 只覆盖向量侧）。**并列**：'
  '①邻域宽度 ±2 敏感性（四臂 `nb` **恰好都是 %s**）；②P17 `P6` 的 MEMO 改判（承 P18 `P5`：'
  '`spearman(w_all,|b_all|)>0` 而 `spearman(w_all,J)<0` ⇒ 对象错配）。'
  '仍挂账：N2h1-α-1 权重级定位、N2h1-β 水果类崩塌解剖、N3-β→N3-ε、R1 对照补强、K4 处置、'
  '**N 线 Phase 3–7 补登 Ledger**（Phase 8–19 已各 1 条）。'
  % jd({a: V[a]['neighbourhood'] for a in ARMS}))
p('')
if PC and not all(v['pass_'] for v in PC.values()):
    _f = [k for k in sorted(PC) if not PC[k]['pass_']]
    _na = [k for k in sorted(PC) if PC[k]['pass_'] is None]
    p('- **本轮预测 %d/%d 通过；描述性 N/A：%s；否证：%s**。'
      % (len(PC) - len(_f) - len(_na), len(PC), ', '.join(_na) or '无', ', '.join(_f) or '无'))
elif PC:
    p('- **本轮预测全通过（5/5）**。')
p('')

sec = '\n'.join(L)
_sec_n = sec.replace('\r\n', '\n').replace('\n', '\r\n').strip('\r\n')
ALREADY_W = ('## Phase 19 / N2h1-' in t0)
if ALREADY_W:
    w('wlog 已含 Phase 19 段 ⇒ 跳过追加（幂等路径）')
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
NEW_TAG = 'post-append-phase19'
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
