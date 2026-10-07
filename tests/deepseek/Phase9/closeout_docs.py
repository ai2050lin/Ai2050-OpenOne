# -*- coding: utf-8 -*-
"""Phase 9 收尾文档：备忘录基线刷新（含历史）+ 当日 wlog 追加。"""
import os, io, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P9T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase9')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
o = []
def w(s=''):
    o.append(str(s)); print(s)

# ---------- 1. 基线刷新（保留历史）----------
bp = os.path.join(INFRA, 'memo_baseline.json')
old = json.load(io.open(bp, encoding='utf-8')) if os.path.isfile(bp) else {}
mb = open(MEMO, 'rb').read()
T = mb.decode('utf-8-sig')
lines = T.splitlines()
heads = {}
for i, l in enumerate(lines):
    if l.startswith('## '):
        heads[l[:46]] = i + 1
new = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'),
       'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
       'bytes': len(mb), 'lines': len(lines), 'sha256': hashlib.sha256(mb).hexdigest(),
       'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
       'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
       'phase9_hdr_lines': [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 9:')],
       'phase_hdr_count': sum(1 for l in lines if l.startswith('## Phase ')),
       'sections': heads,
       'note': ('Phase 9 追加后基线。每次追加后刷新；对账方式=前缀 sha256 逐字节比较。'
                '历史事件：38 处裸 LF→CRLF + 1 空行（+40B，内容无损）。'
                '时钟事件：Phase 8 标题 [22:05] 晚于其产物 mtime 21:34:50，不可作因果排序依据。')}
hist = old.get('history', [])
if old.get('sha256') and old.get('sha256') != new['sha256']:
    hist.append({k: old.get(k) for k in ['frozen_at', 'bytes', 'lines', 'sha256', 'note']})
new['history'] = hist
io.open(bp, 'w', encoding='utf-8').write(json.dumps(new, ensure_ascii=False, indent=1))
w('baseline refreshed: bytes %d lines %d sha8 %s sections %d phase_hdr %d history %d' %
  (new['bytes'], new['lines'], new['sha256'][:8], len(heads), new['phase_hdr_count'], len(hist)))

# ---------- 2. 当日 wlog ----------
sec = []
sec.append('')
sec.append('## Phase 9 / N2h1-α-2：承诺层阈值增益的直接检验（21:49）')
sec.append('')
sec.append('- **死线执行**：Phase 8 §8 最高优先——α-缩放上游 `P_U(diff5)`（扩展网格找 cross-over α*），验证「亚阈值突变」。')
sec.append('- **预注册**：seal `N2h1a2_design_seal.json`（`e99ebd02`）+ **观测前修正案** `..._amend1.json`（补 D7 必要性对偶臂 + 曲线分类器缺口）+ execution 冻结 `execution_phase9.json`（`ece7ed8c`，11 臂，**面板逐字节继承 Phase 8**：`inherits_panel_sha256 45b4641a`）。')
sec.append('- **主结果（发现集 n=24，GPU 41.9 s，零 OOM）**：')
sec.append('  - **amp（L6 自增益）平坦**：剂量放大 **12.0×**（x 0.118→1.421），L6 输出的 U 轴增益只从 **1.2624 → 1.3458（+6.6%）**，J=1.42，幂律 γ=0.030 ⇒ **L6 内部没有阈值增益**（`amp_ref = 1/r̄ = 4.224` 只是范数比，不是增益）。')
sec.append('  - **行为曲线是 S 形**：D1 y = 0/.011/.035/.298/.846/1.000/1.016，陡段 2.19 vs 其余中位 0.40（J=5.41）；D2 同形（J=4.92，γ 1.94）；**半饱和点 x\\* ≈ 0.59–0.63（写向量占比）** ⇒ **非线性在 L6 之后的读出链**。')
sec.append('  - **上游末位 U 分量「充分但不必要」**：放大到写向量量级（x≈1.02）给 **95.5%** 效应；撤除只损失 **5.8%**（kill_frac 0.058，dD(0)=−0.624 vs REF −10.797），donor-class rank-1 比例 **0.458→0.458 不变**。与 Phase 6 的 patch@L5=+0.578 镜像对称。')
sec.append('  - **I_nl 重新归属**：用 D1 曲线作传递函数，单头 `share_v` → 预测 `f(share)·full` 与 Phase 8 实测同量级（head14 0.069/0.037、head11 0.067/0.055、head8 0.059/0.047、head15 0.055/0.032、head24 0.052/0.031），且**所有组件份额 ≪ x\\*≈0.6** ⇒ **I_nl=6.85 是软阈值的算术后果，不是"层内超可加"的独立证据**。')
sec.append('- **内建跨 Phase 复现（比特级）**：D1@α=1 = **10.574739583333335**，与 Phase 8 `T[\'diff6\']` **逐位相同** ⇒ U6 子空间与站点口径完全复现。另：F3 恒等自检两站点 **0.000e+00**；F5 面板继承断言通过。')
sec.append('- **诚实边界**：预注册判决树三级均 **H0**（H1 只差"跳变右端 y≥0.50"一条：J=5.41 达标、右端 0.298 未达标）；**F1 形式失败**（D4 随机 5 维地板 0.166 > 0.10）但被「随机方向在写方向上的几何投影」（E|cos|=3/8）**定量解释**（0.169 vs 预测 0.1665）；非线性只能定位到"L6 之后"。**判据设计缺陷不改判**（改判=事后挑选），留给 Phase 10 参数化重写。')
sec.append('- **记录**：deepseek 备忘录新增 `## Phase 9` 节（**1799 行**起），158,395 → **183,607 B**（前缀逐字节未变、BOM/CRLF 保持、bare_lf 0）；Ledger `atlas_ledger.json` 补登 N 线第 2 条（291 → **292**，含备份 `atlas_ledger_backup_pre_phase9.json`）。')
sec.append('- **时钟事件**：Phase 8 节标题 `[2026-10-01 22:05]` **晚于**其 `_formal_stdout.log` mtime 21:34:50，也晚于本次追加 21:49 ⇒ 标题时间戳不可作因果排序依据。')
sec.append('- **下一步（死线）**：**Phase 10 候选 = 软阈值的深度定位**（D1 剂量探针在 L7…L34 复用，输出 `x*(ℓ)` 与 `J(ℓ)` 剖面；预注册"单层产生"vs"逐层累积"）；曲线分类器改**参数化**（x\\*、陡度、饱和值）替换 H1–H4；R1 挂账对照补强；N2h1-β；跨模型写入端（GLM4 层位须独立复算）。')
sec.append('')
txt = '\n'.join(sec).replace('\r\n', '\n').replace('\n', '\r\n')
if os.path.isfile(WLOG):
    n_before = os.path.getsize(WLOG)
    with open(WLOG, 'ab') as f:
        f.write(txt.encode('utf-8'))
    w('wlog %d -> %d bytes' % (n_before, os.path.getsize(WLOG)))
else:
    io.open(WLOG, 'w', encoding='utf-8').write(txt)
    w('wlog created %d bytes' % os.path.getsize(WLOG))

OUT = os.path.join(P9T, 'closeout_docs.txt')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
