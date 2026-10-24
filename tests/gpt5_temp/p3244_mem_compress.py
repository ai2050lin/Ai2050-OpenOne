# -*- coding: utf-8 -*-
"""Compress workspace MEMORY.md before the 3244 closeout block replace."""
P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
mem = open(P, encoding='utf-8').read()
n0 = len(mem)

REPS = [
    # discipline line
    ('- 每轮：freeze（DESIGN 全静态、hash 排除 created/design_sha8；禁引用 SMOKE 截断对象）→ SMOKE（P{phase}_SMOKE=1，必看数字）→ 正式 → verdict → closeout 五写（磁盘回读口径）→ 独立复核（verify FAIL 只修 verify 不动封存产物）→ present → 预注册下一 Phase。',
     '- 每轮：freeze（DESIGN 静态、hash 排除 created）→ SMOKE → 正式 → verdict → closeout 五写 → 独立复核（verify FAIL 只修 verify）→ present → 预注册下一 Phase。'),
    # G-line status lines
    ('- 26 特征×3 模型（qwen3-4b/14b/glm4-9b）全 scope：E2_held_out×20+、E3×3、E1×3；升级链 9；失败账本 F1–F20（F20=读出几何≠行为控制，3229/3230 双配方确证：GA_flip 0、cos(dR,Δw)=+0.0997/+0.0625 传递正交）。',
     '- 26 特征×3 模型全 scope：E2_held_out×20+、E3×3、E1×3；升级链 9；F1–F20（F20=读出几何≠行为控制）。'),
    ('- GAP-1/2/3 closed；GAP-4 机制链收官（3188 DECOUPLED→3189 SHARED→3190 LOO→3191 held-out→3192 池化 CARRIER→3232 类级 ROBUST，amp_ratio_c 6.57–12.21，spearman 0.79–0.83）；GAP-5 三段式（上游承载→末层线性读出→post-softmax 非线性）入册 gap v1.7–v1.10。',
     '- GAP-1/2/3 closed；GAP-4 机制链收官（3188–3192→3232，amp_ratio_c 6.57–12.21）；GAP-5 三段式入册 gap v1.7–v1.10。'),
    # phase lines in "下一步"
    ('- 3238 收官（census A）：ENT/SCLS 近秩1、USTAR 最扩散、复用非正交、hub #1 FTR-11；registry v1.33 `a30215fd` / ledger #390。',
     '- 3238 收官（census A）：hub #1 FTR-11；registry v1.33 `a30215fd` / ledger #390。'),
    ('- **3239 收官（fidelity min-loop）**：KL 缺口恢复 38.7%；R² OOV 0.179；DVU/DCC 正、ENT/SCLS 负。ledger #391 / freeze `3776313c` / res `5af7a0dd`。',
     '- **3239**：KL 缺口恢复 38.7%；DVU/DCC 正、ENT/SCLS 负。ledger #391 / freeze `3776313c` / res `5af7a0dd`。'),
    ('- **3240 收官（cls extrapolation）**：F0 d39=0.0；F2 全族正向（SCLS +4.47/ENT +3.85）转正未达；臂(b) 客户端化。ledger #392 post `0cf46b3a` / freeze `69c1a6f9` / res `ffb97506`。',
     '- **3240**：F2 全族正向转正未达；臂(b) 客户端化。ledger #392 post `0cf46b3a` / freeze `69c1a6f9` / res `ffb97506`。'),
    ('- **3241 收官（lexical bridge）**：类桥零效应；双桥 G1e 正向 SCLS +0.98/ENT +0.82 但 DVU/DCC 回退越界 → partial；权重侧不足转正。ledger #393 post `794ea771` / freeze `d986f7ae` / res `5f0648f7`。',
     '- **3241**：G1e 正向但 DVU/DCC 回退越界 → partial。ledger #393 post `794ea771` / freeze `d986f7ae` / res `5f0648f7`。'),
    ('- **3242 收官（xmodel transfer）**：迁移真实（KL 分离）；T3_dual 首个零回退正向（SCLS +0.46/ENT +0.40）但 <0.5 门 → partial；缺口=映射 OOV 外推。ledger #394 / freeze `02af8bd7` / res `eb673f70`。',
     '- **3242**：T3_dual 首个零回退正向（+0.46/+0.40）但 <0.5 门 → partial；缺口=映射 OOV 外推。ledger #394 / freeze `02af8bd7` / res `eb673f70`。'),
    ('- **3243 收官（map OOV extrap）**：T3_rep d42=0.0 位级复现；三变体全败（M1 中心化有害 +0.27 vs +0.46、M2 塌缩、M3 几何有行为无）→ base 映射处 seen-only 信息极限。ledger #395 post `4b5b12a8` / freeze `5ccf2bdc` / res `2734168a`。',
     '- **3243**：T3_rep d42=0.0；三变体全败 → base 映射处 seen-only 信息极限。ledger #395 post `4b5b12a8` / freeze `5ccf2bdc` / res `2734168a`。'),
    ('- 3238-align（工程整理）：registry 自动发现 + /atlas/gaps 端点 + 对齐文档；UI 线冻结 v1.10 未动。',
     '- 3238-align：registry 自动发现 + /atlas/gaps + 对齐文档；UI 线 v1.10 冻结。'),
    ('- 挂账不变：APPX-5 施加（重基线后）；N 线施加（ready）；5 appendix GPU/跨线项；UI 线冻结于 v1.10；水果类/K4 列表删除待用户确认。',
     '- 挂账：APPX-5（重基线后）；N 线施加；5 appendix GPU 项；UI 线 v1.10 冻结；水果类/K4 删除待确认。'),
]
for old, new in REPS:
    assert old in mem, 'NOT FOUND: ' + old[:40]
    mem = mem.replace(old, new)
assert '- **3244 已预注册**：(a) 实体桥 DVU/DCC 回退归因（零 GPU）；(b) u* 归因顺延。' in mem
assert len(mem) < 3000, len(mem)
with open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(mem)
print('compressed %d -> %d chars' % (n0, len(mem)))
