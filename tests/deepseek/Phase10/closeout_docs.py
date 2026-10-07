# -*- coding: utf-8 -*-
"""Phase 10 文档收尾：当日 wlog 追加 + _infra/memo_baseline.json 刷新（含 history）。"""
import os, io, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P10T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase10')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
OUT = os.path.join(P10T, 'closeout_docs_phase10.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


# ---------- 1. 当日 wlog 追加 ----------
b0 = open(WLOG, 'rb').read()
t0 = b0.decode('utf-8')
sec = """
## Phase 10 / N2h1-α-3：软阈值的深度定位（22:06）

- **死线执行**：Phase 9 §8 最高优先——把同一 D1 剂量探针（`h_recip + α·P_U6(diff6)`）从 L6 输出逐层搬到 L34 输出，再加「最终 LayerNorm 之后」的读数位点 R，输出 `J(ℓ)` / `x*(ℓ)` / `y_sat(ℓ)` 剖面，判定「单层产生」vs「逐层累积」。
- **预注册**：seal `N2h1a3_design_seal.json`（`70bb8b92`）+ **观测前修正案** `..._amend1.json`（`de8a0717`，A1–A5，五项缺口全部来自冒烟证据）+ exec 冻结 `execution_phase10.json`（`49f3ebda`，面板 12 字段 × Phase8/Phase9 **双向逐元素**继承断言）。
- **主结果（发现集 n=24，GPU 253.0 s，零 OOM）**：
  - **J(ℓ) 随注入深度单调下降**（绝对剂量 5.41→1.15；相对剂量 14.26→1.03），Spearman −0.692（12 非 UNREACH 位点）/ **−0.872**（18 剖面位点）⇒ **软阈值是整条深层栈逐层累积的**，不是某一层做出来的。**全程无断崖**（最大相邻跌幅 4.83→2.85 = 1.7× < Q3 要求的 2×）⇒ 「单层产生」被否证 ⇒ **"层=软门"须改为"栈=软门"**。
  - **读数位点 R（否证探针）严格线性**：R 的相对剂量只有深度位点的 1/3（`r_R = 0.1513`），同网格下 UNREACH ⇒ amend1 A5 补扩展网格（α 到 16）。实测 `dDonor/α = 1.127` 常数、**R²_lin = 1.0000、γ = 1.00、J = 1.00**，推到相对剂量 2.58 仍无拐点 ⇒ **把"软阈值只是解嵌几何/类分数度量的读数假象"这一否证条款排除**。
  - **自基臂揭示最重要的限界**：类子空间随深度急剧旋转（`overlap(U_ℓ, U6)` 0.6892→**0.0298**），而**自基下 α=1 在 L7/L12/L20/L34 全部 ≈ full**（10.735/10.648/10.474/10.576 vs full 10.575）⇒ **绝对效应随深度塌掉（y_sat 1.016→0.027）主要是方向失配，不是下游衰减**。`V_ownbasis = BASIS_SENSITIVE`（2/4）：J 的**趋势**双基稳健，"J 的绝对高度"不是。
  - **饱和值与 r_ℓ 同比例**（y_sat/r_ℓ ≈ 1–2 全程）；相对剂量下 y_sat 在 L6–L20 基本平（0.82–1.01）。
- **判决**：修正族 **Q_abs = Q_rel = Q2_accumulate ⇒ Q_ROBUST**（两剂量坐标一致）；封存 P 族 = `P0_no_verdict`（绝对）/ `P4_once_formed`（相对）⇒ 封存两坐标不一致的**原因是 J 在相对坐标整体更高把窗口判据 [7,10] 抬过阈值**，属判据形式问题，非物理矛盾；**本 Phase 不改判封存结果，两者并列报告**。
- **内建跨 Phase 复现（比特级）**：E0（ℓ=6, α=1）= **10.574739583333335** 与 Phase 9 `full` **逐位相同**；E0b 口径锚点 +0.3335 与 Phase 9 D1a 一致；F3 四处（含 norm hook）0.000e+00；`mean‖P_U6(diff6)‖` 17.0613 与 Phase 9 相同（n6 drift=False）；**F1 通过**（E4 地板 0.1227 / maxabs 12.444 = 0.0099）。
- **确认集（n=17）**：同判 3/4（L7/L20/L34 ✅；L11 ❌，J=1.99 vs 发现集 2.04，压在 2.0 阈值门口）。
- **口径陷阱（新增）**：`r_ℓ = ‖u6‖/‖h_ℓ(recip)‖` ≠ Phase 9 的 `rbar = mean‖P_U6(diff5)‖/mean‖P_U6(diff6)‖`（0.4664 vs 0.2368，近 2×）；冒烟时差点当成 drift 去查。
- **记录**：deepseek 备忘录新增 `## Phase 10` 节（**2036 行**起），183,607 → **209,316 B** / 2033 → **2301 行**（前缀逐字节未变、BOM/CRLF 保持、bare_lf 0、Phase 标题 10 个）；Ledger `atlas_ledger.json` 补登 N 线第 3 条（292 → **293**，备份 `atlas_ledger_backup_pre_phase10.json`）。
- **下一步（死线）**：**Phase 11 候选（最高优先）= 自基全剖面 + 噪声带**——E3 自基臂铺到全部 18 个剖面位点给出**双基 J(ℓ) 剖面**，同时**记录逐对 dDonor（n=24）**用配对 bootstrap 给 J(ℓ)/x*(ℓ) 误差带（**零额外前向**，只改落盘），预注册判据＝Q2 的 Spearman 95% bootstrap 区间是否完全在 −0.6 以下。第二候选：用「逐层替换」而非「逐层注入」做层内贡献分配。其他挂账：N2h1-α-1 权重级定位、N2h1-β 水果类崩塌、N3-β/δ/γ/ε、跨模型写入端（GLM4-9b untied / qwen3-14b 的 r_ℓ 与 J(ℓ) 须独立复算）、R1 对照补强、K4 处置。
"""
t1 = t0.rstrip('\r\n') + '\n\n' + sec.strip('\n') + '\n'
open(WLOG, 'wb').write(t1.encode('utf-8'))
b1 = open(WLOG, 'rb').read()
w('wlog: bytes %d -> %d (+%d) ; lines %d -> %d' %
  (len(b0), len(b1), len(b1) - len(b0), len(b0.split(b'\n')), len(b1.split(b'\n'))))
w('wlog sha256 = %s' % hashlib.sha256(b1).hexdigest())

# ---------- 2. _infra/memo_baseline.json 刷新（带 history）----------
mb = open(MEMO, 'rb').read()
T = mb.decode('utf-8-sig')
lines = T.splitlines()
heads = {}
for i, l in enumerate(lines):
    if l.startswith('## '):
        heads[l[:44]] = i + 1
prev = {}
bp = os.path.join(P10T, 'memo_baseline_preappend_phase10.json')
if os.path.exists(bp):
    prev = json.load(io.open(bp, encoding='utf-8'))
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': 'post-append-phase10',
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(lines), 'sha256': hashlib.sha256(mb).hexdigest(),
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'phase_headings': [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')],
        'sections': heads,
        'history': (prev and [{'tag': 'pre-append-phase10', 'bytes': prev['bytes'], 'lines': prev['lines'],
                               'sha256': prev['sha256']}]) or []}
io.open(os.path.join(INFRA, 'memo_baseline.json'), 'w', encoding='utf-8').write(
    json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(post-append): bytes %d lines %d sha8 %s bare_lf %d phase_headings=%d' %
  (base['bytes'], base['lines'], base['sha256'][:8], base['bare_lf'], len(base['phase_headings'])))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
