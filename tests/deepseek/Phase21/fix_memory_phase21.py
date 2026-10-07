# -*- coding: utf-8 -*-
"""MEMORY.md P21 回填补齐：幂等（new 已在则跳过，old 在则替换），最后全量校验。"""
import os
import io
import hashlib

SRC = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase21\_patch_memory_report.txt'
o = []


def w(s=''):
    o.append(str(s))


REP = [
    (u'收尾链（**十三次 P8–P20**）：', u'收尾链（**十四次 P8–P21**）：'),
    (u'- Ledger n=**303**（P8–P20 各 1；**N 线 P3–P7 待补**）。基线 `post-append-phase20`=**488,772 B/4,674 行/20 标题**（`sha8 68eddd46`；旧 `post-append-phase19` 标 `stale`+`drift_events`；`sections` 键=完整标题行）。',
     u'- Ledger n=**304**（P8–P21 各 1；**N 线 P3–P7 待补**）。基线 `post-append-phase21`=**497,406 B/4,774 行/21 标题**（`sha8 4ba5e22f`；上一步 `post-append-phase20`=488,772 B/`68eddd46`；`sections` 键=完整标题行）。'),
    (u'勘误 `E-comv`/`E-xhdom`/`E-rho`。',
     u'勘误 `E-comv`/`E-xhdom`/`E-rho`。\n'
     u'- **P21**：组件级向量预算 `share_v`+权重实现级 `W` 跨精度（nf4↔bf16 × qwen3-4b/glm4-9b 四臂）**7/9**；A0_bf16 **逐位复现 P8 锚**（`share_v(mlp)` 0.4717、W 0.1329/#9、`I_nl` 6.846、`T[diff6]` 10.5747）；**G1_core 四臂全 True** ⇒「分布式搬运」**不是 nf4 kernel 路径产物**；Δ`share_v(mlp)` ≤**0.0234**、ρ≥**0.9920**、ΔW ≤**0.0110**。**谱系精度缺口（P8=bf16 vs P16–P18=nf4）由本 Phase 闭合**。'),
    (u'⑫ **P20** 只两模型+offload、**不**升格 P16 `com_layer`（`P6` FAIL）、P9 是**域歧义**非物理不稳定。',
     u'⑫ **P20** 只两模型+offload、**不**升格 P16 `com_layer`（`P6` FAIL）、P9 是**域歧义**非物理不稳定。'
     u'⑬ **P21** argmax 单头身份可在**并列带内**跨精度翻转（head14→head8，前三差<0.004）；glm4-9b floors 两精度同不达标 ⇒ **跨模型地板不可比**（非精度效应）。'),
    (u'**交付前自查⇒生成件缺陷可回滚重跑**。',
     u'**交付前自查⇒生成件缺陷可回滚重跑**。\n'
     u'- **P21 教训**：① **identity-probe 取模块权重**（`mod(I_in)` 得 `W^T`，绕开 bnb 4bit 反量化内部）；'
     u'② **argmax 并列内翻转** ⇒ 必并报「前三单头差 + 秩相关」，只报 argmax 会误判；'
     u'③ **跨模型地板不可比**（A1 单组件效应幅度小 ⇒ 相对地板被抬高）；'
     u'④ **复核脚本 `sha8` 与 64-hex 错配** = 生成件缺陷 ⇒ 修脚本重跑，**不动产物**。'),
    (u'- **Phase 21 = 把跨精度推进到组件级向量预算与权重实现级**（bf16 下复算 P8 `share_v` 与 N2h1-α-1 权重级定位）。',
     u'- **Phase 22 = 把 P8 的 `share_v` 与 P16/P17 的逐层 `w_ℓ` 在同一精度下逐位对接**（P17 `w_6` vs P8 `vec_budget`），彻底消除跨 Phase 精度不确定性。'),
    (u'`rdc-main-axis-probe`（15 臂+**61 坑**）、`rdc-phase-closeout`（**33 教训**/十三次链）',
     u'`rdc-main-axis-probe`（15 臂+**62 坑**）、`rdc-phase-closeout`（**34 教训**/十四次链）'),
]

raw = open(SRC, 'rb').read()
bom = raw[:3] == b'\xef\xbb\xbf'
t = raw.decode('utf-8-sig')
w('bytes(before)=%d chars=%d bom=%s sha8=%s' % (len(raw), len(t), bom, hashlib.sha256(raw).hexdigest()[:8]))
applied = []
skipped = []
err = []
for i, (a, b) in enumerate(REP):
    has_new = (b in t) or (b.split('\n')[-1] in t and b.split('\n')[-1].strip() and b.split('\n')[-1] in t)
    has_old = a in t
    w('  [%d] old=%d new_partial=%s' % (i, t.count(a), (b[:24] in t)))
    if a in t and b not in t:
        t = t.replace(a, b)
        applied.append(i)
    elif b in t:
        skipped.append(i)
    elif a not in t and b not in t:
        err.append(i)

out = (('\ufeff' if bom else '') + t).encode('utf-8')
if applied:
    open(SRC, 'wb').write(out)
    w('WROTE (applied=%s skipped=%s)' % (applied, skipped))
else:
    w('NO WRITE (applied=%s skipped=%s)' % (applied, skipped))
rb = open(SRC, 'rb').read()
r2 = rb.decode('utf-8-sig')
w('bytes(after)=%d chars=%d sha8=%s bare_lf=%d bom=%s'
  % (len(rb), len(r2), hashlib.sha256(rb).hexdigest()[:8],
     rb.count(b'\n') - rb.count(b'\r\n'), rb[:3] == b'\xef\xbb\xbf'))
w('err=%s' % err)
CHK = [u'十四次 P8–P21', u'n=**304**', u'4ba5e22f', u'- **P21**：组件级', u'**P21 教训**',
       u'**Phase 22 =', u'**62 坑**', u'**34 教训**', u'⑬ **P21** argmax']
for k in CHK:
    w('  verify %-16s count=%d' % (k, r2.count(k)))
NEG = [u'十三次', u'n=**303**', u'68eddd46 标', u'- **Phase 21 =']
for k in NEG:
    w('  neg    %-16s count=%d' % (k, r2.count(k)))
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('DONE applied=%s err=%s' % (applied, err))
raise SystemExit(1 if err else 0)
