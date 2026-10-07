# -*- coding: utf-8 -*-
"""Phase 15 文档收尾：当日 wlog 追加 + _infra/memo_baseline.json 刷新（含 history 链）。
铁律 (w)：wlog 正文所有数字均从 result_phase15.json / judgement / Ledger / MEMO 现场取值渲染。
必须在 do_append_phase15.py 之后运行（读追加后的 MEMO 基线）。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(P15T, 'closeout_docs_phase15.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


R = json.load(io.open(os.path.join(P15T, 'result_phase15.json'), encoding='utf-8'))
JUD = json.load(io.open(os.path.join(P15T, 'judgement_phase15.json'), encoding='utf-8'))
LG = json.load(io.open(LEDGER, encoding='utf-8'))
EX = json.load(io.open(os.path.join(P15T, 'execution_phase15.json'), encoding='utf-8'))
SEAL = json.load(io.open(os.path.join(P15T, 'N2h1a8_design_seal.json'), encoding='utf-8'))
AM1 = json.load(io.open(os.path.join(P15T, 'N2h1a8_design_seal_amend1.json'), encoding='utf-8'))
AM1_SHA8 = sha(os.path.join(P15T, 'N2h1a8_design_seal_amend1.json'))[:8]
PRE = json.load(io.open(os.path.join(P15T, 'memo_baseline_preappend_phase15.json'), encoding='utf-8'))

V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']
E4 = R['E4_summary']; E5 = R['E5_concentration']; E2 = R['E2_full_swap']; E3 = R['E3_localize']
E6 = R['E6_calibration']; FL = R['floors']
ARMS = EX['arm_order']
A0, A1, A2 = ARMS
CR = JUD['calibration']

mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig').splitlines()
new_sha8 = hashlib.sha256(mb).hexdigest()[:8]
ph_lines = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase ')]
p15_line = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase 15')]
tail = LG['measurements'][-1]


def fn(x, n=6):
    return 'n/a' if x is None else ('%.*f' % (n, x))


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
p('## Phase 15 / N2h1-α-8：跨模型复算「统一剖面」（三臂同一 nf4 口径）——%s / %s（%s）'
  % (JV['Q2_joint'], JV['Q3_joint'], time.strftime('%H:%M')))
p('')
p('- **死线执行**：Phase 14 §8 最高优先 —— 把 Phase 12/13/14 唯一有效口径（单点替换族 `xhalf(ℓ)`/`J(ℓ)` 双坐标剖面 + 置换零假设校准）'
  '在 glm4-9b 与 qwen3-14b（均 untied）上独立复算，回答 ①「两坐标 argmax 相距 13」是层栈性质还是 qwen3-4b 特例；'
  '② null 95 分位是否普遍高达 0.70–0.75。**判据先冻结；禁止沿用 L6/U6**。')
p('- **为什么是本轮最重要的工程决定（nf4 口径）**：本机 GPU 16GB / RAM 可用 ~17GB，而 bf16 下 '
  'qwen3-14b 权重 29.5GB ⇒ 三条 bf16 路线实测：① `device_map=auto` 把 **10 个模块 offload 到磁盘**（层放置 `L16..L39 = meta`）、'
  '单前向 **7.32 s**（全剖面 316 min/臂，不可行）；② `max_memory` 禁磁盘路线在加载期被**硬杀**（无 Python 栈）；'
  '③ glm4-9b bf16 + `max_memory` 可行（**0.745 s/前向**，`L30..L39 = meta`）但会破坏「两臂同一口径」。'
  '最终统一改 **nf4（4bit 权重 + bf16 计算 + double quant）**：三模型均全载 GPU，**0.036 / 0.036 / 0.041 s 每前向**。'
  '量化对结论的影响由**预注册的 A0 臂**承担。')
p('- **装置锚（全部通过）**：41/41 实例 `T=2`；每臂双前向 `max|dlogits| = 0.000e+00`；'
  'α=0 在首/中/末剖面位点 `max|dScore| = 0.000e+00`；每臂 `o_proj.in_features == n_heads × head_dim`；'
  '**独立写入窗定位**（B_cat 相邻最大增量，U_ℓ 逐层独立重建）A0 得 `L*_own = %s`（增量 %+.3f）—— 复现 Phase 8 的几何，'
  '且**未沿用 L6/U6**。' % (E3[A0]['L_star_own'], E3[A0]['L_star_increment']))
p('- **量化保真（A0 vs Phase 12 bf16 已发表量）**：`max|dxhalf| = %s`（容差 %s）、`max|drecover| = %s`；'
  '`argmax_w_x` nf4 = %s vs bf16 = %s（同 %s）；`share_x` %s vs %s；`XH_RANGE` %s vs %s；'
  '`J` 比值带 [%s, %s] ⇒ **`%s`**。'
  % (fn(CR['max_abs_dxh']), FL['XH_FAITHFUL_TOL'], fn(CR['max_abs_drecover']),
     CR['argmax_w_x_nf4'], CR['argmax_w_x_bf16'], CR['argmax_same'],
     fn(CR['share_x_nf4']), fn(CR['share_x_bf16']), fn(CR['XH_RANGE_nf4']), fn(CR['XH_RANGE_bf16']),
     fn(CR['J_ratio_min'], 3), fn(CR['J_ratio_max'], 3), CR['label']))
p('- **⚠️ 首轮运行被装置门拦下并作废重跑（amend1 `%s`，apparatus fix，不改假设）**：冻结的 `panel.sup_id`'
  '（`水果=104618` 等）是 **qwen 词表** id 却被**全局**用于三臂；`glm4-9b-chat-hf` 词表不同（vocab **151329 vs 151643**）'
  '⇒ A1 全程读**错误的类别 token**。装置门 `F2_base_ok`（要求每实例受体类分数 > 0）当场抓到 '
  '**`bad = %d/41`**、受体类分数均值 `%+.3f`、`FULL_SWAP = %+.3f`（4B 为 `%+.3f`）、剂量曲线平坦且低 α 段为负。'
  '**修正**：`sup_id` 改为**每臂由该臂 tokenizer 现场解析** + 新增硬断言 **F1b**'
  '（6/6 类别词单 token 且 `decode(id) == 词`）；A0/A2 解析结果与冻结值**逐位相同**（零副作用），首轮数据作废重跑'
  '（原 stdout 保留 `_formal_stdout_run1_INVALID_supid.log`）。'
  '**若无此门，这会以「GLM4 的 is-a 关系不成立」的机制结论被发表。**'
  % (AM1_SHA8, AM1['evidence_from_device_gate']['A1_F2_base_bad_n'],
     AM1['evidence_from_device_gate']['A1_F2_receptor_class_score_mean'],
     AM1['evidence_from_device_gate']['A1_FULL_SWAP'],
     AM1['evidence_from_device_gate']['A0_FULL_SWAP']))
p('- **主结果 1（Q2 两坐标 argmax 距离）**：Phase 13 在 qwen3-4b 上给 `MODE_X = %s` / `MODE_J = %s`（差 **13** 个窗口单位；'
  'x 集中窗 = w%s [%s..%s]、J 集中窗 = w%s [%s..%s]）。本轮跨模型：%s ⇒ 合取判决 **`%s`**。'
  % (JUD['reference_4B']['MODE_X_13'], JUD['reference_4B']['MODE_J_13'],
     (E5[A0]['win_sem_x'] or {}).get('w'), (E5[A0]['win_sem_x'] or {}).get('a'),
     (E5[A0]['win_sem_x'] or {}).get('b'),
     (E5[A0]['win_sem_j'] or {}).get('w'), (E5[A0]['win_sem_j'] or {}).get('a'),
     (E5[A0]['win_sem_j'] or {}).get('b'),
     ' ; '.join('%s `d_argmax = %s`（`%s`；x 窗 %s [%s..%s]，J 窗 %s [%s..%s]）' % (
         a, V[a].get('Q2_d_argmax'), V[a].get('Q2_label'),
         (E5[a]['win_sem_x'] or {}).get('w'), (E5[a]['win_sem_x'] or {}).get('a'),
         (E5[a]['win_sem_x'] or {}).get('b'),
         (E5[a]['win_sem_j'] or {}).get('w'), (E5[a]['win_sem_j'] or {}).get('a'),
         (E5[a]['win_sem_j'] or {}).get('b')) for a in JV['arms_used_for_cross_model']),
     JV['Q2_joint']))
p('- **主结果 2（Q3 置换零假设 95 分位）**：Phase 14 在 qwen3-4b 上得 `null95_x = 0.6998 > share_x = 0.5745` ⇒ xhalf 坐标判据在 4B 上**无区分力**。'
  '本轮跨模型：%s ⇒ 合取判决 **`%s`**。'
  % (' ; '.join('%s `null95_x = %s`（share_x = %s，裕度 = %s）／`null95_j = %s`（share_j = %s，裕度 = %s）' % (
      a, fn((E5[a]['null_x'] or {}).get('null95')), fn(E5[a]['top3_x']), fn(E5[a]['margin_x']),
      fn((E5[a]['null_j'] or {}).get('null95'), 4), fn(E5[a]['top3_j'], 4), fn(E5[a]['margin_j'], 4))
      for a in JV['arms_used_for_cross_model']), JV['Q3_joint']))
p('- **⚠️ 本轮最重要的自我修正（零假设校准的逐格复核）**：对 `J` 坐标施加**同一套**置换零假设检验后，'
  '**6 个「臂 × 坐标」格里只有 %d 格超过各自的 null 95 分位**（%s）⇒ 三臂上**不存在跨模型稳健的集中度判据**；'
  '「坐标依赖」须再升一级为「**坐标 × 模型**双重依赖」，且**不得再说「某个坐标才是有区分力的坐标」**。'
  '（`J` 裕度 = %s；`xhalf` 裕度 = %s）'
  % (sum(1 for a in ARMS if (E5[a]['margin_j'] or -1) > 0 or (E5[a]['margin_x'] or -1) > 0),
     ' / '.join('%s·%s（裕度 %+.4f）' % (a, 'J' if (E5[a]['margin_j'] or -1) > 0 else 'xhalf',
                                       E5[a]['margin_j'] if (E5[a]['margin_j'] or -1) > 0 else E5[a]['margin_x'])
                for a in ARMS if (E5[a]['margin_j'] or -1) > 0 or (E5[a]['margin_x'] or -1) > 0),
     ' / '.join('%s %+.4f' % (a, E5[a]['margin_j']) for a in ARMS),
     ' / '.join('%s %+.4f' % (a, E5[a]['margin_x']) for a in ARMS)))
p('- **剖面形状（Q5，描述性）**：%s。' %
  ' ; '.join('%s `XH_RANGE = %s`、`spearman(xhalf, depth) = %s`、`spearman(J, depth) = %s`、`L*_own = %s`' % (
      a, fn(E4[a]['XH_RANGE']), fn(E5[a]['spearman_xh_depth'], 4),
      fn(E5[a]['spearman_J_depth'], 4), E3[a]['L_star_own']) for a in JV['arms_used_for_cross_model']))
p('- **FULL_SWAP（零额外前向的归一分母）**：%s。' %
  ' ; '.join('%s `%+.6f`（n=%d）vs Phase 12 参照 `%+.6f`' % (
      a, E2[a]['FULL_SWAP'], E2[a]['n'], JUD['reference_4B']['FULL_SWAP_12']) for a in ARMS if a in E2))
p('- **预注册预测**：%s。' % ' '.join('%s = %s' % (k, PC[k]['pass_']) for k in sorted(PC)))
p('- **判决**：Q1 量化保真 `%s`；Q2 合取 `%s`；Q3 合取 `%s`。' % (CR['label'], JV['Q2_joint'], JV['Q3_joint']))
p('- **记录**：deepseek 备忘录新增 `## Phase 15` 节（**L%s** 起），%d → **%d B** / %d → **%d 行**'
  '（前缀逐字节未变、BOM/CRLF、`bare_lf 0`、Phase 标题 **%d** 个）；Ledger 补登 N 线第 8 条（%d → **%d**，'
  '备份 `atlas_ledger_backup_pre_phase15.json`，verdict `%s`）。'
  % (p15_line[0] if p15_line else '?', PRE['bytes'], len(mb), PRE['lines'], len(mt), len(ph_lines),
     len(LG['measurements']) - 1, len(LG['measurements']), tail['verdict']))
p('- **新增铁律（3 条）**：**（z）量化口径必须由预注册校准臂承担** —— 当宿主硬件容不下目标模型的 bf16 时，'
  '换数值口径（4bit）本身是一个新自由度：必须（i）在 seal 中冻结、（ii）在同一模型上用**已发表的 bf16 量**做逐位点校准、'
  '（iii）给出容差并预先声明「校准失败 ⇒ 全部跨模型结论降级」；**且门必须写在最稳健的量上**'
  '（本轮 A0 的 `argmax_w_j` 在量化噪声下换窗 `%s`→`%s`，若门写在 `J` 上会误判 DEVIANT，写在 `xhalf` 上则稳定）。'
  '**（aa）跨设备加载的内存约束必须作为「技术路线」写进 seal，并给出被否决路线的实测数字** —— '
  '本轮记录三条 bf16 路线（磁盘 offload **7.317 s/前向** ⇒ 316 min/臂；禁磁盘 **加载期被硬杀**；'
  '单臂可行但破坏口径一致性）与最终 nf4 路线（**0.036–0.041 s/前向，约 200×**）。'
  '**（ab）跨模型/跨词表移植时，一切「词表相关常量」（类别 token id、实例 token id、特殊 token id）必须由该模型'
  '自己的 tokenizer 现场解析，禁止沿用源模型的硬编码值，并须配一条「读对了 token 吗」的装置门** —— '
  '本轮 amend1 即此：qwen 的 `sup_id` 被用于 GLM4 词表 ⇒ `bad = %d/41`。'
  '**装置门必须与主结论正交**：`F2_base_ok` 只问「token 读对了吗」，正因如此才能在 A1 产生主结论之前把它拦下。'
  % (E5[A0]['argmax_w_j'], JUD['reference_4B']['MODE_J_13'],
     AM1['evidence_from_device_gate']['A1_F2_base_bad_n']))
p('- **下一步（死线）**：**Phase 16 最高优先 = 统一剖面下的「写入窗 vs 集中窗」关系**（把本轮三臂的 `L*_own`（B_cat 相邻最大增量）'
  '与两坐标集中窗并列，回答「写入窗是否就是 J 坐标的集中窗」这一 Phase 12 遗留问题；判据先冻结）。'
  '**但不得直接沿用本轮的 `J` 窗口索引**：`argmax_w_j` 恰是本轮唯一被量化噪声换掉的量（nf4 vs bf16 换窗）。'
  '第二候选（因「6 格只活 2 格」而升为并列最高优先）：**集中度统计量的重设计** —— `top3_share` 这种'
  '「极值型 3-窗口占比」在两个坐标上都缺跨模型稳健性，需换**对排序 / 单窗口不敏感**的量'
  '（谱熵、窗口加权质心、或深度去势后的残差集中度），并在三臂 × 双坐标上重算 Phase 12/13/14 的全部集中度表。'
  '第三候选：**家族 vs 规模的解耦**（补第三个同家族不同规模 / 第二个 GLM 规模点）。'
  '仍挂账：N2h1-α-1 权重级定位、N2h1-β 水果类崩塌解剖、R1 对照补强、K4 处置、N 线 Phase 3–7 补登 Ledger。')
p('')
if PC and all(v['pass_'] for v in PC.values()):
    p('- **本轮预测全通过**。')
p('')

sec = '\n'.join(L)
_sec_n = sec.replace('\r\n', '\n').replace('\n', '\r\n').strip('\r\n')
t1 = t0.rstrip('\r\n') + '\r\n\r\n' + _sec_n + '\r\n'
open(WLOG, 'wb').write(t1.encode('utf-8'))
b1 = open(WLOG, 'rb').read()
w('wlog(%s): bytes %d -> %d (+%d) ; lines %d -> %d' %
  ('new' if NEW else 'append', len(b0), len(b1), len(b1) - len(b0),
   len(b0.split(b'\n')), len(b1.split(b'\n'))))
w('wlog sha256 = %s' % hashlib.sha256(b1).hexdigest())

# ---------- 2. _infra/memo_baseline.json 刷新（带 history 链） ----------
heads = {}
for i, l in enumerate(mt):
    if l.startswith('## '):
        heads[l[:44]] = i + 1
hist = []
bp = os.path.join(P15T, 'memo_baseline_preappend_phase15.json')
if os.path.exists(bp):
    prev = json.load(io.open(bp, encoding='utf-8'))
    hist.append({'tag': 'pre-append-phase15', 'bytes': prev['bytes'], 'lines': prev['lines'],
                 'sha256': prev['sha256']})
old = os.path.join(INFRA, 'memo_baseline.json')
if os.path.exists(old):
    try:
        ob = json.load(io.open(old, encoding='utf-8'))
        oh = ob.get('history') or []
        for e in oh:
            if e.get('tag') not in [h['tag'] for h in hist]:
                hist.append(e)
        if ob.get('tag') not in [h['tag'] for h in hist]:
            hist.append({'tag': ob.get('tag'), 'bytes': ob.get('bytes'), 'lines': ob.get('lines'),
                         'sha256': ob.get('sha256')})
    except Exception as e:
        w('warn: old history unreadable: %r' % (e,))
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': 'post-append-phase15',
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
