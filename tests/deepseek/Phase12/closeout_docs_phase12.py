# -*- coding: utf-8 -*-
"""Phase 12 文档收尾：当日 wlog 追加（2026-10-02，新建）+ _infra/memo_baseline.json 刷新（含 history 链）+ 清理中间备份。"""
import os, io, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P12T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(P12T, 'closeout_docs_phase12.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


# ---------- 1. 当日 wlog 追加（跨日：Phase 12 落在 2026-10-02）----------
NEW = not os.path.exists(WLOG)
b0 = open(WLOG, 'rb').read() if not NEW else b''
t0 = b0.decode('utf-8')
if NEW:
    t0 = '# 2026-10-02\n'
sec = """
## Phase 12 / N2h1-α-5：逐层残差替换 —— 端点态按构造充分、形状量给出逐层累积、集中度不可决断（00:20）

- **死线执行**：Phase 11 §8 最高优先 —— 把探针族由「**注入**固定 rank-5 类轴 u6」整体换成「**替换整段残差**」（`h_ℓ + α·diff_ℓ`，`diff_ℓ = h_ℓ(donor) − h_ℓ(recip)`，α∈[0,1]；α=1 时该位点残差**精确等于供体贴残差**）。18 位点 × 14 点 α 网格；满替换用供体自身前向 ⇒ **零额外前向**。目的：给 Phase 8–11 的「逐层累积（Q2）」做**换族交叉验证**，并回应 B3（位点不可排序）。
- **预注册**：seal `N2h1a5_design_seal.json`（`4280b23c`，24,725 B，21 顶层键）+ **正式运行前**修正案 `..._amend1.json`（`f13ea993`，A1–A6）+ exec `execution_phase12.json`（`67f38c53`，59 字段；**面板 12 字段 × Phase8/Phase10/Phase11 三向逐元素断言**）；继承锚 `4573a8bd`/`49f3ebda`/`45b4641a`（F13）。
- **amend1 的由来（SMOKE 暴露的设计级缺陷）**：`recover(α=1)` 按构造饱和 —— Phase 8–11 已证「L6 只注入 rank-5 类轴、α=1 即得 full_L6」，而满替换是**严格更大**的干预 ⇒ 端点量必然 ≥1 且无深度信息（实测 `span = 0.0048`）。冻结的 `G0`（要求 `span ≥ 0.30`）因此必然失败。**修**：主量改为形状量 `xhalf` / `J_swap`，端点量升格为**独立正向判据 G5**；G0/G1/G2 搬到 xhalf 口径；α 网格 12 → **14 点**（补 0.15/0.95）。amend1 落盘 `why_this_is_a_priori_not_a_data_peek` 与 `explicitly_unchanged`。
- **主结果（发现集 n=24，GPU 264.2 s，零 OOM）**：
  - **装置自检全绿（含两条新增构造决定点）**：E0 = `10.574739583333335` **逐位复现 Phase 9**（F6）；`n6 = 17.0613`（drift=False）；**F11（新增）读数位点 R 的满替换 ≡ FULL_SWAP，max|d| = `0.000e+00`**；**F12（新增）满替换 ≡ 供体贴残差（float32），max = `0.000e+00`**；F3 四处 `0.000e+00`；**F10 逐对均值自洽 max|d| = `5.329e-15`**；F1 地板比 `0.3582/18.805 = 0.0190`。
  - **`FULL_SWAP = +10.797396`**（n=24，min +5.406 / max +15.791）。
  - **G5 充分性 ✅ `LAST_POS_STATE_SUFFICIENT`**：`recover ∈ [0.9955, 1.0010]`，`min = 0.9955 ≥ 0.90`，**18/18 位点成立**；确认集 `min = 0.9711`。⇒ **末位残差在每一个位点都是「供体答案已成形」的充分统计量。**
  - **端点量退化（构造性）**：`span(recover) = 0.0048`、`rho(recover, depth) = +0.7936` ⇒ **深度信息不在端点里**。
  - **形状量有信号**：`J_swap` **25.35（L6）→ 1.06（L34）**（降 23.9×，log2 降 4.58）；`xhalf` 浅端平台 0.499→0.445、中段平台 ~0.44、**L30 触底 0.390**、**L34 反弹 0.465**（R 处 0.500）；`rho(xhalf, depth) = −0.7833`，95% 带 **[−0.9154, −0.5500]**。
  - **G0 ✅ 但边界脆弱**：`curve_ok_frac = 18/18 = 1.000`，`XH_RANGE = 0.1094`（阈值 0.10，**仅高 9%**）。
  - **G3 ✅ `G3_same_gradient`**：`rho(J_swap, J_inject) = 0.8741`（n=18）⇒ **换族后逐层累积存活**（首次跨探针族交叉验证）。
  - **G2 ⚠ `G2_mid`**：`top3_share_x = 0.5745`，95% 带 **[0.3867, 0.8316]**（**跨 0.60**）；`max_share_x = 0.5895`（> 0.40）⇒ **少层主导（≥0.60）与逐层累积（≤0.40）都判不出**。
  - **G4 ❌ `G4_fail`**：确认集 4 位点 `rho(xhalf) = +0.80`，与发现集 −0.7833 **符号相反** ⇒ 事后判定为**采样密度不足**（4 位点落在非单调剖面的不同支上），**不作否定证据**。
  - **G1 = `G1a_crystallized` 但为仪器伪影**：归一化 `XN=(xhalf−min)/(max−min)` 隐含假设剖面随深度**上升**；实测 rho 为负 ⇒ `XN[0]=1.0`、`first_reach` 全部退化返回首站点（`x_half = 6.00`、`span_10_90 = 0.00`）。**本 Phase 不引用 G1**，正确方向读法是 `xhalf` 随深度**下降**。
  - **F7' ✅**（置换零假设带 `[−0.4675, +0.4757]`，|界| 0.47 < 0.6 ⇒ 阈值有区分力）。
  - **饱和行为**：E2b 相对坐标下 L7/L12 于 `α_rel≈0.8` 饱和（+10.685/+10.802），L20 到 1.6 仍升（+11.871），**L34 到 1.6 达 +18.805（外推、未饱和，超 FULL_SWAP 74%）**；E3 超量 α>1 同结论（L6 降、L20 缓升、L34 升）。离流形 α（`pert_rel > 0.50`）= `[0.6, 0.7, 0.8, 0.9, 0.95, 1.0]` ⇒ **满替换本质上是流形外操作**（已写入 honesty）。
  - **两族坐标不兼容（实测确认）**：`proj_share_u6` 由 L6 **0.680** 单调降到 L34 **0.148**（R 处 0.1288）⇒ 浅端「全残差 ≈ 类轴」、深端「全残差主要是非类别内容」；故两族只比**秩**。
- **判决**：**`ALLOCATION_AMBIGUOUS`**（`G2_mid ∧ G3_same` 落入裁决表「其余」分支）—— 逐层累积获换族交叉确认，但「少层主导 vs 逐层累积」**不可决断**。
- **记录**：deepseek 备忘录新增 `## Phase 12` 节（**L2555** 起），236,092 → **273,579 B** / 2553 → **2858 行**（前缀逐字节未变、BOM/CRLF、bare_lf 0、Phase 标题 **12** 个）；Ledger `atlas_ledger.json` 补登 N 线第 5 条（294 → **295**，备份 `atlas_ledger_backup_pre_phase12.json`，verdict `swap_allocation_ambiguous__g5_last_pos_state_sufficient`）。
- **新增铁律 (r)(s)**：(r) **端点量若由构造决定饱和**（α=1 等价于某个已知满干预），必须降级为独立正向判据、另立形状量作主量 —— 否则「深度信息」会被端点饱和静默吃掉。(s) **归一化方向必须与物理方向一致** —— 「首次达到比例」型判据隐含了「被测剖面随深度上升」的假设，方向相反时会全部退化到首站点，造出**看起来正合预期的仪器伪影**（本轮 G1a）；必须先断言 `spearman(量, 深度)` 的符号。
- **下一步（死线）**：**Phase 13 最高优先 = 位点间配对 bootstrap**（同一 `idx_b` 下 `Δ_b(ℓ_i) = J_b(ℓ_i) − J_b(ℓ_{i+1})`，或 xhalf 的配对差；报 95% 带是否含 0），用于**决断 G2 的 `top3_share_x` 不可决断点**（配对差口径比独立区间更紧）；代价：需先把 E2 改为**逐对曲线**落盘（无需额外前向）。**第二候选**：α 网格低端加密（`XH_RANGE` 只高阈值 9%）+ **逐层累积代换（prefix swap）**作第三条独立口径。其他挂账不变：N2h1-α-1 权重级定位、N2h1-β 水果类崩塌、N3-β/δ/γ/ε、跨模型写入端（GLM4-9b untied / qwen3-14b 的 `r_ℓ`/`J(ℓ)`/`proj_share_u6` 须**独立复算**）、R1 对照补强、K4 处置。
- **过程备注（非实验内容）**：追加后自检发现锚点 `cross_alpha` 在正文中未出现 ⇒ 用 `rollback_append12.py` **按字节截断回基线**（前缀 sha 逐字节相等校验通过）→ 补写实现名 → 重新追加（`do_append_phase12.py`，三向断言全绿）。**回滚路径本身被实证可用**。
"""
t1 = t0.rstrip('\r\n') + '\n\n' + sec.strip('\n') + '\n'
open(WLOG, 'wb').write(t1.encode('utf-8'))
b1 = open(WLOG, 'rb').read()
w('wlog(%s): bytes %d -> %d (+%d) ; lines %d -> %d' %
  ('new' if NEW else 'append', len(b0), len(b1), len(b1) - len(b0), len(b0.split(b'\n')), len(b1.split(b'\n'))))
w('wlog sha256 = %s' % hashlib.sha256(b1).hexdigest())

# ---------- 2. _infra/memo_baseline.json 刷新（带 history 链）----------
mb = open(MEMO, 'rb').read()
T = mb.decode('utf-8-sig')
lines = T.splitlines()
heads = {}
for i, l in enumerate(lines):
    if l.startswith('## '):
        heads[l[:44]] = i + 1
hist = []
bp = os.path.join(P12T, 'memo_baseline_preappend_phase12.json')
if os.path.exists(bp):
    prev = json.load(io.open(bp, encoding='utf-8'))
    hist.append({'tag': 'pre-append-phase12', 'bytes': prev['bytes'], 'lines': prev['lines'], 'sha256': prev['sha256']})
old = os.path.join(INFRA, 'memo_baseline.json')
if os.path.exists(old):
    try:
        oh = json.load(io.open(old, encoding='utf-8')).get('history') or []
        for e in oh:
            if e.get('tag') not in [h['tag'] for h in hist]:
                hist.append(e)
    except Exception as e:
        w('warn: old history unreadable: %r' % (e,))
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': 'post-append-phase12',
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(lines), 'sha256': hashlib.sha256(mb).hexdigest(),
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'phase_headings': [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')],
        'sections': heads, 'history': hist}
io.open(old, 'w', encoding='utf-8').write(json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(post-append): bytes %d lines %d sha8 %s bare_lf %d phase_headings=%d history=%d' %
  (base['bytes'], base['lines'], base['sha256'][:8], base['bare_lf'], len(base['phase_headings']), len(hist)))

# ---------- 3. 清理中间备份（已追加的第一版，含失效锚点）----------
bk = os.path.join(P12T, '_memo_with12_backup.md')
if os.path.exists(bk):
    sz = os.path.getsize(bk)
    os.remove(bk)
    w('removed intermediate backup %s (%d bytes)' % (os.path.basename(bk), sz))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
