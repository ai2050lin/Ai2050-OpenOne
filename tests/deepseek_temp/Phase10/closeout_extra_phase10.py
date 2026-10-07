# -*- coding: utf-8 -*-
"""Phase 10 收尾补记：wlog 追加 + MEMORY 两处外科替换。逐处 assert count==1 + 回读复核。"""
import hashlib, os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')

# ---------- 1) wlog 追加 ----------
add = (
"\n## Phase 10 收尾链补全 [2026-10-01]\n"
"- 独立磁盘复核：`tests/deepseek/Phase10/disk_verify_phase10.py` -> `disk_verify_phase10.txt`，**184 项 / 0 FAIL**。\n"
"  - 从冻结 `result_phase10.json` 确定性重算：J(l)（绝对/相对双坐标，18 位点，1e-9）、段斜率数组、logistic 拟合 (k_log, x*, R2_log)（18 位点 + R，1e-6）、Spearman（18 位点 -0.8720330237 / 12 位点 -0.6923076923）、Q 族布尔（Q2_accumulate）、x*_rel = x*_abs*r_l 不变量（18/18）、overlap 单调性、F1 地板比 0.009857、比特锚 E0=10.574739583333335。\n"
"  - 修正过程中复现「同一消息多条 Edit 静默丢失」：6 条丢 4 条（Q2 公式、J(R) 的 x 处理、gamma 定义、always-true 检查），改用 `fix_disk_verify.py`（Python 补丁，逐处 assert count==1）一次性修好。\n"
"- 技能同步：`patch_skills_phase10.py` -> `patch_skills_phase10.txt`。\n"
"  - `rdc-main-axis-probe/SKILL.md`：28,408 -> 33,841 B（sha8 2a339bdb），10 臂 -> 11 臂、26 坑 -> 36 坑，新增 N2h1-a-3 臂行 + N2h1-a-3 门 + 坑 32-36。\n"
"  - `rdc-phase-closeout/SKILL.md`：14,518 -> 15,778 B（sha8 173039a3，CRLF 保持），6 条 -> 8 条收尾教训（新增 IME 吞字、Read 行号 off-by-one），节标题含 Phase 10，实证表加「技能同步」行。\n"
"- 新增源码：`tests/deepseek/Phase10/disk_verify_phase10.py`、`tests/deepseek_temp/Phase10/fix_disk_verify.py`、`patch_skills_phase10.py`、`probe_result_struct{,2}.py`。\n"
)
b0 = open(WLOG, 'rb').read() if os.path.isfile(WLOG) else b''
with open(WLOG, 'ab') as f:
    f.write(add.replace('\n', '\r\n').encode('utf-8'))
b1 = open(WLOG, 'rb').read()
print('wlog: %d -> %d bytes (+%d), prefix_intact=%s' % (len(b0), len(b1), len(b1)-len(b0), b1[:len(b0)] == b0))

# ---------- 2) MEMORY 补记 ----------
t = open(WMEM, encoding='utf-8').read()
sha0 = hashlib.sha256(t.encode('utf-8')).hexdigest()

A1 = '备份 `atlas_ledger_backup_pre_phase10.json`）。'
A1N = (A1 + '\n'
 '    - **收尾链补全**：独立磁盘复核 `disk_verify_phase10.py` = **184 项 / 0 FAIL**（重算 J(ℓ) 双坐标 / logistic (k,x*,R2) / Spearman / Q 族 / `x*_rel=x*_abs·r_ℓ` 不变量 18/18 / overlap 单调 / F1=0.009857 / 比特锚）；技能同步完成（`rdc-main-axis-probe` 11 臂 36 坑、`rdc-phase-closeout` 8 条教训）。')
assert t.count(A1) == 1, 'A1 count=%d' % t.count(A1)
t = t.replace(A1, A1N, 1)

import re
# 在 Phase 10 铁律行末尾追加 (o)
m = re.search(r'（n）深部位点的固定基探针必须同时报 `overlap\(U_ℓ, U_ref\)`\*\*。', t)
assert m, 'A2 anchor not found'
t = t[:m.end()] + '**（o）同一消息内多条 Edit 会静默丢失**（Phase 10 实测 **6 条丢 4 条**：Q2 公式、J(R) 的 x 处理、gamma 定义、always-true 检查）⇒ **关键修改一律用 Python 补丁脚本逐处 `assert count==1`，改完回读复核**；`numpy` import 的丢失靠 `NameError` 才发现。' + t[m.end():]

open(WMEM, 'w', encoding='utf-8', newline='').write(t)
t2 = open(WMEM, encoding='utf-8').read()
for s in ['**184 项 / 0 FAIL**' if False else '184 项 / 0 FAIL', '（o）同一消息内多条 Edit 会静默丢失', '（n）深部位点的固定基探针']:
    c = t2.count(s)
    print('  MEMORY check %-40s count=%d %s' % (s[:40], c, 'OK' if c >= 1 else 'FAIL'))
    assert c >= 1, s
print('MEMORY bytes %d sha8 %s -> %s (limit 3000? len=%d chars)' % (
    len(t2.encode('utf-8')), sha0[:8], hashlib.sha256(t2.encode('utf-8')).hexdigest()[:8], len(t2)))
