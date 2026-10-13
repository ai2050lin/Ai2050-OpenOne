# -*- coding: utf-8 -*-
import io, hashlib

# daily log
p = r'D:\AI2050\Ai2050-OpenOne\memory'
p = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-10.md'
t = io.open(p, encoding='utf-8').read()
ADD = '''
## Phase 3200 — G5-E2 FTR-23 held_out 升格臂（零 GPU 登记臂）HOLDOUT_PREDICTIVE 6/6
- 协议 freeze 前细化（诚实登记）：FTR-23 无 OOV 面板方向向量 ⇒ 改评封存 3178 实体留出 null fold cos_out（16 fold × 2 关系 × 3 模型，3178 协议 verbatim）；门 bar 0.5 预冻结探针后钉定（保守）。
- 门 6/6 全过：med cos_out isa 0.7696/0.7564/0.7762、hasa 0.7429/0.7855/0.7885，min fold ≥0.6091——关系内几何实体留出高度可预测；关系特异残差=关系间偏移非关系内噪声。与 3199 合读：held-out 可预测性住在类均值/关系内结构，关系间偏移不承载跨面板预测。
- FTR-23 E1→E2_held_out（升级链 #9，E2×20/E1×3）；registry v1.22 9fd87433→v1.23 5f5b9e73；atlas_v1_25 7dc08f1e（782/782 清）。
- 对照锚：3178 标量 4dp 回读 drift==0 ×3（实测探针先于 freeze）+ 3183 dv 逐位 + 3157 tc<1e-9；design 2ea6e1f6、res f9a280fc/cba9d56c、final ae7a5837/dfd3b880、ledger #352（post 8964e20d）、MEMO 4b9359ab→989fd907。
- 3201 预注册：G5-E3 E 线收官审计臂（26 特征锚齐备性审计 + FTR-24 scope_limits append 方法论定论 + appendix 5 项盘点；0 矛盾 ⇒ E_LINE_CLOSED，v1.23→v1.24 + atlas_v1_26）。
'''
t = t + ADD
io.open(p, 'w', encoding='utf-8', newline='').write(t)
assert io.open(p, encoding='utf-8').read() == t
print('daily ->', hashlib.sha256(t.encode()).hexdigest()[:8])

# workspace MEMORY.md
m = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
t = io.open(m, encoding='utf-8').read()
BT = '`'
pairs = [
 ('## 2 G 线图谱现状（2026-10-10，registry v1.22 + gap v1.6 + atlas_v1_24.html）',
  '## 2 G 线图谱现状（2026-10-10，registry v1.23 + gap v1.6 + atlas_v1_25.html）'),
 ('atlas_v1_24 ' + BT + '74a11492' + BT + '（775 字段；',
  'atlas_v1_25 ' + BT + '7dc08f1e' + BT + '（782 字段；**3200 G5-E2 FTR-23 held_out 臂 '
  'HOLDOUT_PREDICTIVE 6/6（封存 3178 实体留出 null cos_out 中位 0.74-0.79 ≥0.5 全单元，'
  '关系内可预测/关系特异残差=关系间偏移），FTR-23 E1→E2_held_out 升级链 #9，'
  'registry v1.22→v1.23 5f5b9e73**；'),
 ('E2×19、E3×3、E1×3（FTR-04/FTR-23/FTR-24；',
  'E2×20、E3×3、E1×3（FTR-04 已闭 noise_level/FTR-24 OOV 面板 NOT_PREDICTIVE 在案；'),
 ('ledger n=351（post ' + BT + 'cffec403' + BT + '）。',
  'ledger n=352（post ' + BT + '8964e20d' + BT + '）。'),
 ('- **3200 已预注册 = G5-E2 FTR-23 held_out 升格臂（零 GPU，3194 同协议 ×3/3 门）**：PASS ⇒ v1.22→v1.23 + atlas_v1_25 + upgrade_log #9；亦负 ⇒ E1 不可预测性定论、转 appendix 清算。',
  '- **3200 已完成（G5-E2 HOLDOUT_PREDICTIVE 6/6，FTR-23 升 E2_held_out，升级链 #9）**：'
  'registry v1.23 5f5b9e73、atlas_v1_25 7dc08f1e、ledger #352 post 8964e20d、design 2ea6e1f6、'
  'res f9a280fc/cba9d56c、final ae7a5837/dfd3b880、MEMO 989fd907。\n'
  '- **3201 已预注册 = G5-E3 E 线收官审计臂（零 GPU）**：26 特征锚齐备性审计 + FTR-24 '
  'scope_limits append（held-out 可预测性几何分界定论）+ appendix 5 项盘点；0 矛盾 ⇒ '
  'E_LINE_CLOSED（v1.23→v1.24 + atlas_v1_26）。'),
]
for old, new in pairs:
    assert t.count(old) == 1, (t.count(old), old[:60])
    t = t.replace(old, new)
io.open(m, 'w', encoding='utf-8', newline='').write(t)
assert io.open(m, encoding='utf-8').read() == t
print('MEMORY.md ->', hashlib.sha256(t.encode()).hexdigest()[:8])
