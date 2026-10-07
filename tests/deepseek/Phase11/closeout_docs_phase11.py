# -*- coding: utf-8 -*-
"""Phase 11 文档收尾：当日 wlog 追加 + _infra/memo_baseline.json 刷新（含 history）。"""
import os, io, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P11T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
OUT = os.path.join(P11T, 'closeout_docs_phase11.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


# ---------- 1. 当日 wlog 追加 ----------
b0 = open(WLOG, 'rb').read()
t0 = b0.decode('utf-8')
sec = """
## Phase 11 / N2h1-α-4：自基全剖面 + 噪声带（23:45）

- **死线执行**：Phase 10 §8 最高优先——把 E3 自基臂由 4 个位点铺满**全部 18 个剖面位点**（`h_ℓ + α·P_U{ℓ}(diff_ℓ)`，同 7 点网格）得**双基 J(ℓ) 剖面**；并把 E1/E1b/E3 改为**逐对 dDonor（n=24）落盘**，用配对 bootstrap（B=2000）+ 置换零假设（B_perm=2000）给误差带（**零额外前向**，只改落盘）。
- **预注册**：seal `N2h1a4_design_seal.json`（`a9c59555`）+ **观测前修正案** `..._amend1.json`（`84897e36`，A1–A4）+ exec `execution_phase11.json`（`4573a8bd`，54 字段；面板 12 字段 × Phase8/Phase10 **双向逐元素**断言）。
- **主结果（发现集 n=24，GPU 329.5 s，零 OOM）**：
  - **装置自检全绿（含本轮新增三条）**：E0 = `10.574739583333335` **逐位复现 Phase 9**（F6）；E0b = +0.3335 完全一致（drift=False）；`n6 = 17.0613`（drift=False）；**F9（新增）E1 对 `result_phase10` 的每个 (site, α) 单元逐位复现，max|d| = `0.000e+00`，bad=[]**；**F8 E3(L6) ≡ E1(L6)，max|d| = 0**；**F10 逐对均值自洽，max|d| = `3.553e-15`**；F1 地板比 `0.1227/12.444 = 0.0099`。
  - **B1（噪声带，核心）**：绝对剂量 `rho = −0.8720`，95% 带 **[−0.9340, −0.7998]**；相对剂量 `rho = −0.9773`，带 **[−0.9876, −0.9092]** ⇒ **两坐标上界都 < −0.6**。
  - **B2（双基全剖面）**：18 位点自基剖面 `rho = −0.9897`（带 [−0.9979, −0.9525]），`spread = 5.587`（带 [4.08, 8.71]）⇒ **「J 随深度递减」不是 U6 固定基的产物**（Phase 10 的限界由 4 位点采样升级为全剖面陈述）。
  - **F7'（判据区分力）**：置换零假设 95% 带 **[−0.4655, +0.4696]**，|界| 0.47 < 0.6 ⇒ 阈值 −0.6 不在噪声可达域。
  - **B3（分辨力诊断）**：相邻位点 J 的 95% 区间 **16/17 重叠**（唯一不重叠是 (22,24)）⇒ **剖面趋势可信、逐位点 J 高度不可排序**。
  - **V_ownbasis = BASIS_SENSITIVE (6/18)**：类标签（离散量）双基不一致（浅端 L8/L9/L10 由 S_WEAK→S_STRONG、深端 L24–L34 由 UNREACH→GRADUAL），与 **B2（趋势连续量一致）并存** ⇒ 精确定位 Phase 10 的「趋势双基稳健、绝对高度不稳健」。
  - **V_readout = LINEAR**（R\* 扩展网格 α 到 16）⇒ 否证条款第二次排除；**V_abs = P0_no_verdict / Q_abs = Q_rel = Q2_accumulate ⇒ Q_ROBUST**（与 Phase 10 逐字一致）。
  - 确认集 n=17 同判 **3/4**（L11 J=1.99 vs 2.04，压在 2.0 门口，与 B3 吻合）。
- **判决**：**Q2_ESTABLISHED_WITH_BAND** —— 逐层累积在「双坐标 + 双基 + 噪声带」下全部成立，且判据有区分力。
- **记录**：deepseek 备忘录新增 `## Phase 11` 节（**L2305** 起），209,316 → **236,092 B** / 2301 → **2552 行**（前缀逐字节未变、BOM/CRLF、bare_lf 0、Phase 标题 11 个）；Ledger `atlas_ledger.json` 补登 N 线第 4 条（293 → **294**，备份 `atlas_ledger_backup_pre_phase11.json`，verdict `own_basis_full_profile_q2_accumulate__band_q2_established_with_band`）。
- **新增铁律 (p)(q)**：(p) 非线性统计量的判决必须**同时给误差带与零假设校准**（置换/打乱标签）；只报点估计或只报带都不够。(q) 探针族凡存在「应由构造决定的位点」（如 E3(L6) ≡ E1(L6)），必须写成**硬断言**——它免费区分实现缺陷与物理发现。
- **下一步（死线）**：**Phase 12 候选（最高优先）= 层内贡献分配（逐层替换而非逐层注入）**——把受体句 ℓ 处残差替换为供体残差，测单层贡献的 J；判据：少数层（≤3）贡献 ≥60% spread ⇒「栈=软门」细化为「少层主导」。**第二候选**：相邻位点**配对** bootstrap（`J_b(ℓ_i) − J_b(ℓ_{i+1})` 的 95% 带是否含 0），把 B3 从诊断升级为判决。其他挂账不变：N2h1-α-1 权重级定位、N2h1-β 水果类崩塌、N3-β/δ/γ/ε、跨模型写入端（GLM4-9b untied / qwen3-14b 的 r_ℓ 与 J(ℓ) 须**独立复算**）、R1 对照补强、K4 处置。
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
bp = os.path.join(P11T, 'memo_baseline_preappend_phase11.json')
if os.path.exists(bp):
    prev = json.load(io.open(bp, encoding='utf-8'))
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': 'post-append-phase11',
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(lines), 'sha256': hashlib.sha256(mb).hexdigest(),
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'phase_headings': [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')],
        'sections': heads,
        'history': (prev and [{'tag': 'pre-append-phase11', 'bytes': prev['bytes'], 'lines': prev['lines'],
                               'sha256': prev['sha256']}]) or []}
io.open(os.path.join(INFRA, 'memo_baseline.json'), 'w', encoding='utf-8').write(
    json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(post-append): bytes %d lines %d sha8 %s bare_lf %d phase_headings=%d' %
  (base['bytes'], base['lines'], base['sha256'][:8], base['bare_lf'], len(base['phase_headings'])))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
