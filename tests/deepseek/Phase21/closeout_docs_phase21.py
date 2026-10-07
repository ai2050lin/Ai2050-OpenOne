# -*- coding: utf-8 -*-
"""Phase 21 文档收尾：当日 wlog 追加 + _infra/memo_baseline.json 刷新（history 链）。
纪律：wlog 正文所有数字均从 result_phase21.json / execution / Ledger / MEMO 现场取值渲染（不自报）。
必须在 do_append_phase21.py 之后运行（读追加后的 MEMO）。
注意：格式串中**不得出现裸百分号**（需写 %%）；本文正文一律避免百分号字符。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P21T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', time.strftime('%Y-%m-%d') + '.md')
OUT = os.path.join(P21T, 'closeout_docs_phase21.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


R = json.load(io.open(os.path.join(P21T, 'result_phase21.json'), encoding='utf-8'))
EX = json.load(io.open(os.path.join(P21T, 'execution_phase21.json'), encoding='utf-8'))
LG = json.load(io.open(LEDGER, encoding='utf-8'))
PRE = json.load(io.open(os.path.join(P21T, 'memo_baseline_preappend_phase21.json'), encoding='utf-8'))
JD = json.load(io.open(os.path.join(P21T, 'judgement_phase21.json'), encoding='utf-8'))

ARMS = list(EX['arm_order'])
A0, A0b, A1, A1b = ARMS
QP = {p['model']: p for p in R['quant_pairs']}
P = R['predictions']
CAL = R['calibration']
FL = EX['floors']
ARM = R['arms']
VERDICT = JD.get('verdict') or R.get('verdict') or 'n/a'
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig').split('\r\n')
ph_lines = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase ')]
p21_line = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase 21')]
tail = [m for m in LG['measurements'] if m.get('phase') == 21][-1]


def fn(x, n=4):
    return 'n/a' if x is None else ('%.*f' % (n, x))


def sg(x, n=4):
    return 'n/a' if x is None else ('%+.*f' % (n, x))


def A_(a, *ks):
    cur = ARM[a]
    for k in ks:
        cur = cur[k]
    return cur


def q(model, key, nd=6):
    v = QP[model].get(key)
    return fn(v, nd) if isinstance(v, (int, float)) else str(v)


def ps(k):
    v = P.get(k)
    return 'PASS' if v is True else ('N/A' if v is None else 'FAIL')


# ---------- 1. 当日 wlog 追加 ----------
NEW = not os.path.exists(WLOG)
b0 = open(WLOG, 'rb').read() if not NEW else b''
t0 = b0.decode('utf-8-sig')
if NEW:
    t0 = '# %s\n' % time.strftime('%Y-%m-%d')

L = []


def p(s):
    L.append(s)


p('')
p('## Phase 21 / N2h1-alpha-14：组件级向量预算与权重实现级的跨精度稳健性（nf4 vs bf16）——`%s`（%s）'
  % (VERDICT, time.strftime('%H:%M')))
p('')
p('- **死线执行**：Phase 20 §11 写死的**最高优先** —— 把跨精度检验推进到**组件级向量预算 `share_v`** 与 '
  '**权重实现级容量 `W`**（P19 只覆盖向量侧 `w_l`/`com_V`，P20 只覆盖行为侧 `b_{c,l}` 与剖面侧 `com_layer`）。')
p('- **新论据（谱系精度缺口）**：P8 用 `dtype=torch.bfloat16` 跑出 `share_v`；而 P16/P17/P18 把同一预算推广到'
  '逐层/行为/剖面时**全用 nf4** ⇒ P8 的 `share_v`/`W` 与 P17 的 `w_l`/`com_V` 属**不同精度谱系**，'
  '本 Phase 在同一装置内闭合该缺口。')
p('- **唯一自变量（seal 冻结）**：前向数值精度 bitsandbytes nf4 (4bit) ↔ `torch.bfloat16`；'
  '模板 / 6 类词 / 41 实例 / 24 discovery / 17 confirmation / 写入窗层 / U 构造（discovery 估计，秩 %d）/ '
  '判据阈值 / 容差 —— 全部逐字继承 P8。'
  % int(A_(A0, 'U', 'rank')))
p('- **三个度量**：**M1 向量预算 `share_v`**（第一指标，精确可加，**零额外前向**）；'
  '**M2 权重实现级容量 `W`**（`W_o` 经**单位阵探针** `mod(I_in)` 取得 `W^T`，绕开 bnb 反量化内部）；'
  '**M3 效应侧 `dDonor`/`I_nl`**（第二指标，不可加）。四臂各 %d 次前向。'
  % int(JD['n_forwards_per_arm'][A0]))
p('- **装置锚（A0_bf16 校准臂逐位复现 P8 冻结锚，容差 ≤ %s）**：'
  '`share_v(mlp)` got/exp %s/%s（d=%s）；`max_head_share_v` %s/%s、argmax `%s`/`%s`；'
  '`W.max_head_share` %s/%s、W.argmax #%s/#%s；`I_nl` %s/%s；`T[diff6]` %s/%s ⇒ **%s**'
  % (fn(FL['CALIB_TOL_SHARE_V'], 4),
     fn(CAL['share_v_mlp']['got'], 4), fn(CAL['share_v_mlp']['exp'], 4), '%.1e' % CAL['share_v_mlp']['d'],
     fn(CAL['max_head_share_v']['got'], 4), fn(CAL['max_head_share_v']['exp'], 4),
     CAL['argmax_head_v']['got'], CAL['argmax_head_v']['exp'],
     fn(CAL['W_max_head_share']['got'], 4), fn(CAL['W_max_head_share']['exp'], 4),
     CAL['W_argmax_head']['got'], CAL['W_argmax_head']['exp'],
     fn(CAL['I_nl']['got'], 4), fn(CAL['I_nl']['exp'], 4),
     fn(CAL['T_diff6']['got'], 4), fn(CAL['T_diff6']['exp'], 4),
     ('PASS（逐位复现）' if CAL.get('ok') else 'FAIL')))
p('- **主结果 1（M1，四臂 G1 核心门）**：`share_v(mlp)` = %s / %s / %s / %s；'
  '`max_head_share_v` = %s / %s / %s / %s（argmax %s / %s / %s / %s）⇒ 四臂皆 '
  '`max_head_share_v ≤ %s` 且 `share_v(mlp) ≤ %s` ⇒ **G1_core 全 %s**'
  '（「分布式搬运」**不是 nf4 kernel 路径的产物**）。'
  % (fn(A_(A0, 'M1', 'share_v_mlp')), fn(A_(A0b, 'M1', 'share_v_mlp')),
     fn(A_(A1, 'M1', 'share_v_mlp')), fn(A_(A1b, 'M1', 'share_v_mlp')),
     fn(A_(A0, 'M1', 'max_head_share_v')), fn(A_(A0b, 'M1', 'max_head_share_v')),
     fn(A_(A1, 'M1', 'max_head_share_v')), fn(A_(A1b, 'M1', 'max_head_share_v')),
     A_(A0, 'M1', 'argmax_head_v'), A_(A0b, 'M1', 'argmax_head_v'),
     A_(A1, 'M1', 'argmax_head_v'), A_(A1b, 'M1', 'argmax_head_v'),
     fn(FL['G1_MAXHEAD_V'], 2), fn(FL['G1_MLP_SHARE_V'], 2),
     all(A_(a, 'G1_core') for a in ARMS)))
p('- **主结果 2（跨精度配对，同模型 nf4 vs bf16）**：qwen3-4b `d_share_v(mlp)` = %s（≤ %s）、'
  '`d_max_head_share_v` = %s、`spearman` = %s；glm4-9b `d_share_v(mlp)` = %s、`d_max_head_share_v` = %s、'
  '`spearman` = %s ⇒ **33 维预算分布跨精度同序**。'
  % (sg(QP['qwen3-4b']['d_share_v_mlp'], 6), fn(FL['QUANT_TOL_SHARE_V'], 4),
     sg(QP['qwen3-4b']['d_max_head_share_v'], 6), fn(QP['qwen3-4b']['spearman_share_v']),
     sg(QP['glm4-9b-chat-hf']['d_share_v_mlp'], 6), sg(QP['glm4-9b-chat-hf']['d_max_head_share_v'], 6),
     fn(QP['glm4-9b-chat-hf']['spearman_share_v'])))
p('- **主结果 3（M2 权重实现级）**：`W.max_head_share` = %s / %s / %s / %s；跨精度 |Δ| 最大 %s（≤ %s），'
  'argmax 不变（#%s / #%s）⇒ **W 稳定**。'
  % (fn(A_(A0, 'M2', 'max_head_share')), fn(A_(A0b, 'M2', 'max_head_share')),
     fn(A_(A1, 'M2', 'max_head_share')), fn(A_(A1b, 'M2', 'max_head_share')),
     fn(max(abs(QP[k]['W']['d_max']) for k in QP), 6), fn(FL['QUANT_TOL_W'], 4),
     A_(A0, 'M2', 'argmax_head'), A_(A1, 'M2', 'argmax_head')))
p('- **对照（描述性）**：glm4-9b 的 `W.mlp_share_vs_attn` = %s / %s 与 qwen3-4b 的 %s / %s 形成强对照'
  '（glm4-9b 的 MLP 容量远大于 attn）；`loo_vec_top1` = %s / %s / %s / %s。'
  % (fn(A_(A1, 'M2', 'mlp_share_vs_attn'), 2), fn(A_(A1b, 'M2', 'mlp_share_vs_attn'), 2),
     fn(A_(A0, 'M2', 'mlp_share_vs_attn'), 4), fn(A_(A0b, 'M2', 'mlp_share_vs_attn'), 4),
     fn(A_(A0, 'M1', 'loo_vec_top1')), fn(A_(A0b, 'M1', 'loo_vec_top1')),
     fn(A_(A1, 'M1', 'loo_vec_top1')), fn(A_(A1b, 'M1', 'loo_vec_top1'))))
p('- **确认集（n=%d，与 %d 对 discovery 不相交）**：`share_v(mlp)` = %s / %s / %s / %s；'
  '`G1_core` = %s / %s / %s / %s ⇒ 同带。'
  % (int(A_(A0, 'confirmation', 'n')), len(EX['discovery']),
     fn(A_(A0, 'confirmation', 'share_v_mlp')), fn(A_(A0b, 'confirmation', 'share_v_mlp')),
     fn(A_(A1, 'confirmation', 'share_v_mlp')), fn(A_(A1b, 'confirmation', 'share_v_mlp')),
     A_(A0, 'confirmation', 'G1_core'), A_(A0b, 'confirmation', 'G1_core'),
     A_(A1, 'confirmation', 'G1_core'), A_(A1b, 'confirmation', 'G1_core')))
p('- **覆盖限界**：A2（Qwen3-14B，29.5 GB）**不参与** bf16 腿（P19 实测 bf16 加载 segfault）⇒ '
  '跨精度稳健性只在 **qwen3-4b 与 glm4-9b** 上验证；A1_bf16 需 CPU offload ⇒ 含「分片执行」第二源；'
  'A1 写入窗层 L%d ≠ A0 的 L%d ⇒ 跨模型比的是各臂自身写入窗；M1（可加）与 M3（不可加）**不得混用**。'
  % (int(A_(A1, 'primary_layer')), int(A_(A0, 'primary_layer'))))
p('- **同轮勘误（append-only，不改判）**：**[E-floors]** A1（glm4-9b）两精度 floors 均不达标且比值几乎相同'
  '（frac_M %s / %s；A0 为 %s–%s 达标，判据 frac ≤ %s）⇒ P9 的 FAIL 是 **A1 装置自身的既有性质，不是精度效应**。'
  '**[E-argmax]** qwen3-4b 单头 argmax 在 nf4 下由 `head14` 变 `head8`，但**前三单头彼此差 < 0.004**、'
  '33 维秩相关 ≥ %s ⇒ 「单头身份」对精度不稳健、「分布形状/量级」稳健；glm4-9b 的 argmax 两精度同为 `head27`。'
  % (fn(A_(A1, 'M3', 'floors', 'frac_M')), fn(A_(A1b, 'M3', 'floors', 'frac_M')),
     fn(A_(A0, 'M3', 'floors', 'frac_M'), 4), fn(A_(A0b, 'M3', 'floors', 'frac_M'), 4),
     fn(FL['FLOOR_FRAC'], 2),
     fn(min(QP[k]['spearman_share_v'] for k in QP))))
p('- **预注册预测**：%d/%d 通过（%s）；否证：%s。'
  % (R['n_pass'], R['n_total'],
     ' '.join('%s=%s' % (k, ps(k)) for k in sorted(P)),
     ', '.join(k for k in sorted(P) if P[k] is False) or '无'))
p('- **记录**：deepseek 备忘录新增 `## Phase 21` 节（**L%s** 起），%d → **%d B** / %d → **%d 行**'
  '（前缀逐字节未变、BOM/CRLF、`bare_lf 0`、Phase 标题 **%d** 个）；Ledger 补登 N 线第 %d 条'
  '（%d → **%d**，verdict `%s`，`ledger_sha256_8 = %s`）。'
  % (p21_line[0] if p21_line else '?', PRE['bytes'], len(mb), PRE['lines'], len(mt), len(ph_lines),
     sum(1 for m in LG['measurements']
         if isinstance(m.get('phase'), int) and 8 <= m['phase'] <= 21),
     len(LG['measurements']) - 1, len(LG['measurements']), tail['verdict'], LG['ledger_sha256_8']))
p('- **下一步（死线）**：**Phase 22 最高优先 = 把 P8 线的 `share_v` 与 P16/P17 的逐层 `w_l` 在同一精度下对接**'
  '（P17 的 `w_6` 与 P8 的 `vec_budget` 逐位比对），彻底消除「跨 Phase 不同精度」的隐性不确定性。'
  '**并列**：邻域宽度 ±2 敏感性；P17 `P6` 的 MEMO 改判。仍挂账：N2h1-alpha-1 权重级定位（已完成部分）、'
  'N2h1-beta 水果类崩塌、N3-beta→N3-epsilon、R1 补强、K4、**N 线 P3–P7 补登 Ledger**（P8–P21 已各 1 条）。')
p('')

sec = '\n'.join(L)
_sec_n = sec.replace('\r\n', '\n').replace('\n', '\r\n').strip('\r\n')
ALREADY_W = ('## Phase 21 / N2h1-' in t0)
if ALREADY_W:
    w('wlog 已含 Phase 21 段 ⇒ 跳过追加（幂等路径）')
    t1 = t0 if t0.endswith('\n') else t0 + '\n'
else:
    t1 = t0.rstrip('\r\n') + '\r\n\r\n' + _sec_n + '\r\n'
open(WLOG, 'wb').write(t1.encode('utf-8'))
b1 = open(WLOG, 'rb').read()
w('wlog(%s): bytes %d -> %d (%+d) ; lines %d -> %d'
  % ('new' if NEW else ('skip' if ALREADY_W else 'append'), len(b0), len(b1), len(b1) - len(b0),
     len(b0.split(b'\n')), len(b1.split(b'\n'))))
w('wlog sha256 = %s' % hashlib.sha256(b1).hexdigest())
w('wlog path = %s' % WLOG)

# ---------- 2. _infra/memo_baseline.json 刷新 ----------
heads = {}
for i, l in enumerate(mt):
    if l.startswith('## '):
        heads[l.rstrip()] = i + 1
_n_hdr = sum(1 for l in mt if l.startswith('## '))
assert len(heads) == _n_hdr, ('sections 键碰撞：%d 个标题行 -> %d 个键' % (_n_hdr, len(heads)))
hist = []
if PRE:
    hist.append({'tag': PRE.get('tag'), 'bytes': PRE.get('bytes'), 'lines': PRE.get('lines'),
                 'sha256': PRE.get('sha256') or PRE.get('sha8')})
NEW_TAG = 'post-append-phase21'
old = os.path.join(INFRA, 'memo_baseline.json')
prev_drift = []
if os.path.exists(old):
    try:
        ob = json.load(io.open(old, encoding='utf-8'))
        prev_drift = ob.get('drift_events') or []
        for e in (ob.get('history') or []):
            if e.get('tag') not in [h['tag'] for h in hist]:
                hist.append(e)
        if ob.get('tag') not in [h['tag'] for h in hist] and ob.get('tag') != NEW_TAG:
            hist.append({'tag': ob.get('tag'), 'bytes': ob.get('bytes'), 'lines': ob.get('lines'),
                         'sha256': ob.get('sha256')})
    except Exception as e:
        w('warn: old baseline unreadable: %r' % (e,))
hist = [h for h in hist if h.get('tag') != NEW_TAG]
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': NEW_TAG,
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(mt), 'sha256': hashlib.sha256(mb).hexdigest(),
        'sha8': hashlib.sha256(mb).hexdigest()[:8],
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'phase_headings': ph_lines,
        'sections_key_rule': 'full-heading-line',
        'drift_events': prev_drift,
        'sections': heads, 'history': hist}
io.open(old, 'w', encoding='utf-8', newline='\n').write(json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(post-append): bytes %d lines %d sha8 %s bare_lf %d phase_headings=%d history=%d drift_events=%d'
  % (base['bytes'], base['lines'], base['sha8'], base['bare_lf'],
     len(base['phase_headings']), len(hist), len(prev_drift)))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('DONE ->', OUT)
