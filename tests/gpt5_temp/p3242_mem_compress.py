# -*- coding: utf-8 -*-
p = r'D:/AI2050/Ai2050-OpenOne/.workbuddy/memory/MEMORY.md'
mem = open(p, encoding='utf-8').read()
o1 = ('**3239 收官（fidelity min-loop）**：KL 缺口恢复 38.7%（B1 24.8%）；R² seen 0.659/OOV 0.179；'
      'DVU/DCC 正保真、ENT/SCLS 负（OOV 类列无信号）。ledger #391 / freeze `3776313c` / res `5af7a0dd`；registry v1.33 不动。')
n1 = ('**3239 收官（fidelity min-loop）**：KL 缺口恢复 38.7%；R² seen 0.659/OOV 0.179；DVU/DCC 正、'
      'ENT/SCLS 负。ledger #391 / freeze `3776313c` / res `5af7a0dd`。')
assert o1 in mem, 'o1'
mem = mem.replace(o1, n1)
o2 = ('**3240 收官（cls extrapolation）**：F0 复现 d39=0.0；F1 均匀先验塌缩；F2 子空间补全全族正向'
      '（SCLS +4.47/ENT +3.85）、R² 0.205、KL 1.0008，转正未达。ledger #392 post `0cf46b3a` / '
      'freeze `69c1a6f9` / res `ffb97506`；verify 33/0；臂(b) 客户端化完成（/atlas/fidelity 端点 + '
      'FidelityMode 面板，playwright 11/11）。')
n2 = ('**3240 收官（cls extrapolation）**：F0 d39=0.0；F1 塌缩；F2 全族正向（SCLS +4.47/ENT +3.85）'
      '转正未达；臂(b) 客户端化（/atlas/fidelity + FidelityMode，playwright 11/11）。ledger #392 '
      'post `0cf46b3a` / freeze `69c1a6f9` / res `ffb97506`。')
assert o2 in mem, 'o2'
mem = mem.replace(o2, n2)
o3 = ('- 3238-align（工程整理，非账本轮）：registry 自动发现 32 版 + /atlas/gaps 端点；atlas_tree 刷新；'
      'TM-ISA-DEMO-v1 bundle；对齐文档 `research/gpt5/docs/CLIENT_SERVER_ALIGNMENT_v1.md`；自检 15/15。'
      'UI 线冻结 v1.10 未动。')
n3 = '- 3238-align（工程整理）：registry 自动发现 + /atlas/gaps 端点 + 对齐文档；UI 线冻结 v1.10 未动。'
assert o3 in mem, 'o3'
mem = mem.replace(o3, n3)
open(p, 'w', encoding='utf-8', newline='').write(mem)
print('compressed, len=%d' % len(mem))
