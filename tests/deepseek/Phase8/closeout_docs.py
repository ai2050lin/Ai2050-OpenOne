# -*- coding: utf-8 -*-
"""Phase 8 收尾文档：备忘录基线刷新（含历史）+ 双 wlog 当日日志追加。"""
import os, io, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
B = chr(92)
o = []
def w(s=''):
    o.append(str(s)); print(s)

# ---------- 1. 基线刷新 ----------
bp = os.path.join(INFRA, 'memo_baseline.json')
old = json.load(io.open(bp, encoding='utf-8')) if os.path.isfile(bp) else {}
mb = open(MEMO, 'rb').read()
T = mb.decode('utf-8-sig')
lines = T.splitlines()
heads = {}
for i, l in enumerate(lines):
    if l.startswith('## '):
        heads[l[:44]] = i + 1
new = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'),
       'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
       'bytes': len(mb), 'lines': len(lines), 'sha256': hashlib.sha256(mb).hexdigest(),
       'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
       'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
       'sections': heads,
       'note': 'Phase 8 追加后基线。每次追加后刷新；对账方式=前缀 sha256 逐字节比较。'}
hist = old.get('history', [])
if old.get('sha256') and old.get('sha256') != new['sha256']:
    hist.append({k: old.get(k) for k in ['frozen_at', 'bytes', 'lines', 'sha256', 'note']})
new['history'] = hist
io.open(bp, 'w', encoding='utf-8').write(json.dumps(new, ensure_ascii=False, indent=1))
w('baseline refreshed: bytes %d lines %d sha8 %s sections %d history %d' %
  (new['bytes'], new['lines'], new['sha256'][:8], len(heads), len(hist)))

# ---------- 2. 当日日志 ----------
sec = []
sec.append('')
sec.append('## Phase 8 / N2h1-alpha：写入算子组件预算（22:05）')
sec.append('')
sec.append('- **死线执行**：N2h1-α（最高优先）——qwen3-4b L6 写入算子的组件级归因，判据 K_g =「单头 share ≤ 30% 且 MLP ≤ 50% ⇒ 分布式搬运，停止追搬运工」。')
sec.append('- **预注册**：seal `N2h1a_design_seal.json`（922f6747）+ **观测前修正案** `..._amend1.json`（37c1a609）+ execution 冻结（45b4641a，含面板指纹）；落点 `tests/deepseek_temp/Phase8/`。')
sec.append('- **SMOKE 抓到的实质缺陷**：分量**效应不可加**（attn 1.58 + mlp 0.35 + diff5 0.33 = 2.26 ≪ diff6 11.46，超可加 5.1×）⇒ 用效应份额当判据会机械得出"分布式"（假阳性）。修正案把第一指标改为**向量预算 share_v**（`P_U(diff6)=P_U(diff5)+Σ_h P_U(Δa6_h)+P_U(Δm6)` 精确可加），阈值不变。')
sec.append('- **主结果（发现集 n=24，GPU 99.8 s）**：验证臂独立复算写入窗 **@L6 (+10.159，flag=OK)**；向量预算 **MLP share_v=0.4717**、**最大单头 head14 = 0.0742**（公平份额 3.03%）、最大效率份额 0.111；Z 零消融最大单头 0.100 / MLP 0.278；W 容量最大头 0.133。⇒ **G1 分布式搬运**。确认集（n=17）同带：单头 0.0765 / MLP 0.4511。')
sec.append('- **层性质新量**：`I_nl = |dDonor(diff6)| / Σ_c|dDonor(c)| = 6.85` ⇒ **写入窗是阈值型放大**；上游残差已占写入向量 23.2%（范数）却只贡献 2.7% 效应。')
sec.append('- **记录**：deepseek 备忘录新增 `## Phase 8` 节（1630 行起），142,312 → **158,395 B**（前缀逐字节未变，BOM/CRLF 保持）；Ledger `atlas_ledger.json` 补登 N 线第 1 条（290 → 291，含备份与 self-hash stale 说明）。')
sec.append('- **完整性事件**：21:22→21:25 备忘录被外部进程改写（38 处裸 LF→CRLF + 1 空行 = +40B/+1 行），**内容无损失**；已冻结基线 `tests/deepseek_temp/_infra/memo_baseline.json` 供对账（含历史）。')
sec.append('- **下一步（死线）**：Phase 9 候选＝**阈值增益直接检验**（α-缩放上游 `P_U(diff5)` 找 cross-over α*，验证"亚阈值突变"）；N2h1-β 水果类崩塌；跨模型写入端（glm4/qwen2.5，GLM4 层位须独立复算）；R1 挂账对照补强。')
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

# workspace wlog（若存在同名文件在 workspace 下）
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8', 'closeout_docs.txt')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE')
