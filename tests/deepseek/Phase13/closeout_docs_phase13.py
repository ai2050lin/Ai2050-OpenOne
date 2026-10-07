# -*- coding: utf-8 -*-
"""Phase 13 文档收尾：当日 wlog 追加（2026-10-02）+ _infra/memo_baseline.json 刷新（含 history 链）。"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P13T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(P13T, 'closeout_docs_phase13.txt')

o = []
def w(s=''):
    o.append(str(s)); print(s)


# ---------- 1. 当日 wlog 追加 ----------
NEW = not os.path.exists(WLOG)
b0 = open(WLOG, 'rb').read() if not NEW else b''
t0 = b0.decode('utf-8')
if NEW:
    t0 = '# 2026-10-02\n'
sec = """
## Phase 13 / N2h1-α-6：位点间配对 bootstrap —— 装置逐位复现（max|d|=0）、配对口径 10/17 vs 独立口径 2/17、集中度是**坐标系依赖**（01:11）

- **死线执行**：Phase 12 §8 最高优先 —— 对相邻位点做**配对** bootstrap（同一 `idx_b` 下 `Δ_b(ℓ_i) = F_b(ℓ_i) − F_b(ℓ_{i+1})`），把「哪些相邻对真的可分辨」从诊断升为**判决**，并决断 Phase 12 的 `G2_mid`（`top3_share_x` 带 `[0.3867, 0.8316]` 跨 0.60）。
- **零额外前向（本线首次）**：Phase 12 已把逐对矩阵落盘（`E2_pairs` 18 位点 × 14 α × 24 对、`E6_pairs` 14×24、`E5_pairs` 4×4×17），且 18 位点 × 14 个 α 的 `order` **完全一致** ⇒ 可重建 `PM_swap/PM_R/PM_conf`。分母 `FS_VEC` 由 `FULL_SWAP_pairs`（以受体词为键）按 `order` 取回，重建后 `mean = 10.7973958333333` 与 Phase 12 `FULL_SWAP` **逐位相等**。Bootstrap 生成器 `BRNG = default_rng(20261001)` 的**第一个消费点**就是主循环 `integers` ⇒ **完整重放 BRNG 流即可逐位复现**。**纯 CPU，2.11 s，不导入 torch（F21）。**
- **可行性探针前置（本轮治理改进）**：先跑 `_feas_probe.py`（**只重建 Phase 12 已发表量**，不算任何新统计量），证实「逐位复现」这一最大技术风险 ⇒ 才写 seal。**把 SMOKE 前置化**，避免 A0 硬断言在正式运行才崩。探针结果：`J_ci`(18×3) / `top3_share_x` / `rho_xhalf` / `R_ci` / **两个 2000 值置换零假设** 全部 `max|d| = 0.000e+00`。
- **预注册**：seal `N2h1a6_design_seal.json`（`808c4575`，17,533 B，23 键）+ exec `execution_phase13.json`（`7b5b73ef`，11 臂）。**先冻结 3 条可证伪预测**（依据全部来自 Phase 12 已发表产物：`xhalf.jumps` 中最大单跳是末位 `+0.0645`，占 `XH_RANGE` 58.9%）：P1 `xhalf` argmax 窗口 = 最深窗口，频次 ≥ 0.50；P2 `J_swap` argmax 窗口 ∈ {0,1,2}，频次 ≥ 0.50；P3 `spearman(|jumps_x|, |jumps_J|) ≤ 0.2`。**三条全部 PASS。**
- **装置锚（本线最强）**：重放 BRNG ⇒ Phase 12 的 `J_ci`（18 位点 × lo/hi/med）、`top3_share_x_ci`、`top3_share_recover_ci`、`rho_recover`、`rho_xhalf`、`R_ci`(recover/xhalf)、**两个 2000 值置换零假设** —— **全部 `max|d| = 0.000e+00`（BIT-EXACT）**；A0b 确认集带亦 BIT-EXACT。**这是本线第一次把「装置一致」写成可机检的逐位硬断言**，而且是在零前向的纯再分析里做到的。其余 floor：F15 `3.55e-15`、F16 `2.84e-14`、F17 `2.84e-14`、F19 `0.0`、F18 带含点估计 **17/17（两坐标）**。
- **主结果 1（结清 Phase 11 B3）**：配对口径 **`N_dec_J = 10/17`**（10 对的 `Δ̂` 全为正 = J 随深度显著下降），而 **独立区间口径只 `2/17`** ⇒ **Phase 11 的「16/17 重叠」是保守口径的产物**（边际不重叠是充分条件，重叠 ≠ 不可分辨）。`Δ_xhalf` 只 `4/17`（L9→L10、L14→L16、L22→L24 下降；**L32→L34 回升**）。**A3 紧化归因**：`cov > 0` 达 **17/17、反例 NONE**（硬断言「`cov>0 ⇒ sd_paired < sd_indep`」逐对成立）；`rho_pair ∈ [0.347, 0.894]`；**收紧比中位数 0.5475**。
- **主结果 2（决断 Phase 12 G2_mid，结论：坐标系依赖）**：给「集中度」加上**双坐标 + argmax 窗口定位**。
  - `xhalf`：`top3_share = 0.5745`，带 `[0.3867, 0.8316]`，**`P(≥0.60) = 0.3790`**、`P(≤0.40) = 0.0435`、`P(mid) = 0.5775`；**argmax 窗口 = w14 = L28→L34（最深）**，频次 **0.7505**。
  - `J_swap`：`top3_share = 0.7953`，带 `[0.5961, 0.8986]`，**`P(≥0.60) = 0.9730`**、`P(≤0.40) = 0.0000`；**argmax 窗口 = w1 = L7→L10（浅端）**，频次 **0.7390**（w0 0.2515 + w1 0.739 + w2 0.0095 ⇒ 全部集中在前两窗）。
  - **两窗口相距 13 个跳变位** ⇒ 冻结判据「少数几层承载 ≥ 60% 的 spread」在 `J` 坐标**成立**、在 `xhalf` 坐标**不成立**。`P13_verdict = CONCENTRATION_COORDINATE_DEPENDENT`（`G0p = True`，`coord_dep = True`）；`disc_verdict = PAIRED_TEST_INFORMATIVE`。
- **深尾反转被决断（回应 G4 符号矛盾）**：`L32→L34` 的 `xhalf` = **`−0.064481`，带 `[−0.078529, −0.051705]`（完全排除 0）** ⇒ **L34 反弹是真信号**，且它是整条剖面最大单跳；`L30→L32` 与 `L28→L30` 单个都 TIE ⇒ **「L30 谷」是两个不可分辨小步的叠加**。确认集（3 对，n=17）唯一可分辨的 `L20→L34` 与发现集**同号**（`Δxhalf = −0.031639`，带 `[−0.052958, −0.004599]`）⇒ **G4 的 `+0.80` 与发现集 `−0.7833` 都「对」，只是量了剖面的不同支**。
- **稳健性（两条必须一起记住）**：
  - **A8 α 网格留一**：`XH_RANGE ∈ [0.1007, 0.1119]`、`top3 ∈ [0.5526, 0.5759]`；**剔除 `α = 0.4` 时 `XH_RANGE` 从 `0.1094` 降到 `0.1007` —— 仍高于 0.10 阈值，但 `G0` 裕度由 `9.39%` 压缩到 `0.73%`（差 13 倍）** ⇒ Phase 12 G0「仅高阈值 9%」的脆弱性被独立证实，也是 Phase 14 网格加密的定量理由。
  - **A9 陡度统计量替代**：分母改 IQR 后 `spearman = 0.5707`，**`N_dec_J_alt = 5/17`（对比主口径 10/17）** ⇒ `N_dec` 是**统计量依赖的**，只能读作「在该定义下的可分辨对数」。
- **判决**：**`CONCENTRATION_COORDINATE_DEPENDENT`** —— Phase 12 的 `ALLOCATION_AMBIGUOUS` 不是抽样噪声，而是**坐标系依赖**；应改述为「在 `xhalf` 坐标不可决断；在 `J` 坐标少层主导成立（集中在浅端 L7→L10 的下降段）」。
- **记录**：deepseek 备忘录新增 `## Phase 13` 节（**L2860** 起），273,579 → **308,702 B** / 2858 → **3149 行**（前缀逐字节未变、BOM/CRLF、bare_lf 0、Phase 标题 **13** 个）；Ledger 补登 N 线第 6 条（295 → **296**，备份 `atlas_ledger_backup_pre_phase13.json`，verdict `paired_paired_test_informative__concentration_coordinate_dependent`）。
- **新增铁律 (t)(u)**：(t) **集中度 / 离散度型判据必须至少在两个独立坐标上同时报告，且必须报告 argmax 位置（不只是取值）** —— 单坐标的「未决」可能只是坐标系选择（本轮 `G2_mid` 即此例：`xhalf` 上 `P(≥0.60)=0.379` 未决，`J` 上 `0.973` 成立，窗口相距 13）。(u) **「不可排序 / 不可分辨」型结论必须写明所用区间口径（边际 vs 配对）** —— 边际不重叠是充分条件，重叠 ≠ 不可分辨；`cov>0` 时配对差口径必然更紧；凡引用「X/Y 重叠」必须附配对口径对照（本轮 2/17 → 10/17）。
- **下一步（死线）**：**Phase 14 最高优先 = 逐层累积代换（prefix swap）+ 双坐标集中度报告**（第三条独立口径；判据写成「`(share, argmax_window)` 二元组在两坐标上的一致性」）。**第二候选**：α 网格低端加密（0.0–0.2 段）。**第三候选**：跨模型写入端 × 集中度坐标依赖（glm4-9b untied / qwen3-14b 独立复算 `J(ℓ)`/`xhalf(ℓ)`/`proj_share_u6`/argmax 窗口）。其他挂账不变：N2h1-α-1 权重级定位、N2h1-β 水果类崩塌、N3-β/δ/γ/ε、R1 对照补强、K4 处置、G 线 Phase 3154。
- **过程备注（非实验内容）**：`do_append_phase13.py` 的**源文件预检**在追加前抓到 4 个锚点缺失（`CONCENTRATION_COORDINATE_DEPENDENT`/`PAIRED_TEST_INFORMATIVE`/`587e7880`/`1d95cd24`）⇒ **先修源文件再追加，一次成功，无需回滚** —— Phase 12 新增的教训 #12「自检锚点须先对追加源文件预检」**首次生效**。
"""
t1 = t0.rstrip('\r\n') + '\n\n' + sec.strip('\n') + '\n'
open(WLOG, 'wb').write(t1.encode('utf-8'))
b1 = open(WLOG, 'rb').read()
w('wlog(%s): bytes %d -> %d (+%d) ; lines %d -> %d' %
  ('new' if NEW else 'append', len(b0), len(b1), len(b1) - len(b0),
   len(b0.split(b'\n')), len(b1.split(b'\n'))))
w('wlog sha256 = %s' % hashlib.sha256(b1).hexdigest())

# ---------- 2. _infra/memo_baseline.json 刷新（带 history 链） ----------
mb = open(MEMO, 'rb').read()
T = mb.decode('utf-8-sig')
lines = T.splitlines()
heads = {}
for i, l in enumerate(lines):
    if l.startswith('## '):
        heads[l[:44]] = i + 1
hist = []
bp = os.path.join(P13T, 'memo_baseline_preappend_phase13.json')
if os.path.exists(bp):
    prev = json.load(io.open(bp, encoding='utf-8'))
    hist.append({'tag': 'pre-append-phase13', 'bytes': prev['bytes'], 'lines': prev['lines'],
                 'sha256': prev['sha256']})
old = os.path.join(INFRA, 'memo_baseline.json')
if os.path.exists(old):
    try:
        oh = json.load(io.open(old, encoding='utf-8')).get('history') or []
        for e in oh:
            if e.get('tag') not in [h['tag'] for h in hist]:
                hist.append(e)
    except Exception as e:
        w('warn: old history unreadable: %r' % (e,))
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': 'post-append-phase13',
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(lines), 'sha256': hashlib.sha256(mb).hexdigest(),
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'phase_headings': [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')],
        'sections': heads, 'history': hist}
io.open(old, 'w', encoding='utf-8').write(json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(post-append): bytes %d lines %d sha8 %s bare_lf %d phase_headings=%d history=%d' %
  (base['bytes'], base['lines'], base['sha256'][:8], base['bare_lf'], len(base['phase_headings']), len(hist)))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')
print('DONE ->', OUT)
