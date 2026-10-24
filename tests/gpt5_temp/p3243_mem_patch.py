# -*- coding: utf-8 -*-
"""3243 MEMORY patch v2: replace prereg line with closeout block + compress old lines."""
P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
mem = open(P, encoding='utf-8').read()
old = '- **3243 已预注册**：(a) 映射 OOV 外推改进臂（零 GPU）；(b) 实体桥回退归因顺延；(c) u* 归因顺延。'
new = ('- **3243 收官（map OOV extrap）**：T3_rep d42=0.0 位级复现；三变体全败（M1 中心化有害 +0.27 vs +0.46、M2 塌缩、M3 几何有行为无）→ base 映射处 seen-only 信息极限。ledger #395 post `4b5b12a8` / freeze `5ccf2bdc` / res `2734168a`。'
       '- **3244 已预注册**：(a) 实体桥 DVU/DCC 回退归因（零 GPU）；(b) u* 归因顺延。')
assert old in mem, 'old 3243 block not found'
mem = mem.replace(old, new)

reps = [
    # 3242 compress
    ('- **3242 收官（xmodel transfer）**：迁移真实（cos 0.25/0.12 vs 置换 0.14/0.06，KL 干净分离）；seen 映射 R² 0.81–0.84 但 OOV 外推负；T3_dual 首个零回退正向（SCLS +0.46/ENT +0.40）但 <0.5 门 → partial。缺口=映射 OOV 外推。ledger #394 / freeze `02af8bd7` / res `eb673f70`。',
     '- **3242 收官（xmodel transfer）**：迁移真实（KL 分离）；T3_dual 首个零回退正向（SCLS +0.46/ENT +0.40）但 <0.5 门 → partial；缺口=映射 OOV 外推。ledger #394 / freeze `02af8bd7` / res `eb673f70`。'),
    # 3240 compress
    ('- **3240 收官（cls extrapolation）**：F0 d39=0.0；F1 塌缩；F2 全族正向（SCLS +4.47/ENT +3.85）转正未达；臂(b) 客户端化（/atlas/fidelity + FidelityMode，playwright 11/11）。ledger #392 post `0cf46b3a` / freeze `69c1a6f9` / res `ffb97506`。',
     '- **3240 收官（cls extrapolation）**：F0 d39=0.0；F2 全族正向（SCLS +4.47/ENT +3.85）转正未达；臂(b) 客户端化。ledger #392 post `0cf46b3a` / freeze `69c1a6f9` / res `ffb97506`。'),
    # 3241 compress
    ('- **3241 收官（lexical bridge，预注册证伪为主）**：类桥零效应（ΔSCLS −0.001）；双桥 G1e 首次同时正向 SCLS +0.98/ENT +0.82 但 DVU/DCC 回退越界 → partial；真实身份优于置换（−0.77 vs −4.07）。权重侧不足转正。ledger #393 post `794ea771` / freeze `d986f7ae` / res `5f0648f7`。',
     '- **3241 收官（lexical bridge）**：类桥零效应；双桥 G1e 正向 SCLS +0.98/ENT +0.82 但 DVU/DCC 回退越界 → partial；权重侧不足转正。ledger #393 post `794ea771` / freeze `d986f7ae` / res `5f0648f7`。'),
    # 3239 compress
    ('- **3239 收官（fidelity min-loop）**：KL 缺口恢复 38.7%；R² seen 0.659/OOV 0.179；DVU/DCC 正、ENT/SCLS 负。ledger #391 / freeze `3776313c` / res `5af7a0dd`。',
     '- **3239 收官（fidelity min-loop）**：KL 缺口恢复 38.7%；R² OOV 0.179；DVU/DCC 正、ENT/SCLS 负。ledger #391 / freeze `3776313c` / res `5af7a0dd`。'),
]
for o, n in reps:
    assert o in mem, 'not found: %s' % o[:40]
    mem = mem.replace(o, n)
assert len(mem) < 3000, len(mem)
open(P, 'w', encoding='utf-8', newline='').write(mem)
print('MEMORY patched, len=%d' % len(mem))
