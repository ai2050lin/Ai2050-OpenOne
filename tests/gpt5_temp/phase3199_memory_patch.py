# -*- coding: utf-8 -*-
import io, hashlib

m = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
t = io.open(m, encoding='utf-8').read()
BT = '`'
pairs = [
 ('## 2 G 线图谱现状（2026-10-10，registry v1.21 + gap v1.6 + atlas_v1_23.html）',
  '## 2 G 线图谱现状（2026-10-10，registry v1.22 + gap v1.6 + atlas_v1_24.html）'),
 ('atlas_v1_23 ' + BT + '9488777c' + BT + '（775 字段；',
  'atlas_v1_24 ' + BT + '74a11492' + BT + '（775 字段；**3199 G5-E1 FTR-24 held_out 臂 '
  'NOT_PREDICTIVE 0/3（med|cos_held,dvu| min 0.035-0.060 远低于 0.3，关系方向不承载谱外可预测性，'
  '与 3188/3189 互证解耦），counter_evidence #4 append、E1 维持，registry v1.21->v1.22 9fd87433**；'),
 ('ledger n=350（post ' + BT + '44b4ccc5' + BT + '）。',
  'ledger n=351（post ' + BT + 'cffec403' + BT + '）。'),
 ('- **3199 已预注册 = G5-E1 FTR-24 held_out 升格臂（零 GPU，3194 同协议）**：4-fold OOV 类留出 median |cos_held| ≥0.3 ×3/3 ⇒ E1→E2（registry v1.21→v1.22 + atlas_v1_24）；FTR-15 已于 3180 升 E2 故改选 FTR-24。',
  '- **3199 已完成（G5-E1 NOT_PREDICTIVE 0/3）**：registry v1.22 9fd87433、atlas_v1_24 74a11492、'
  'ledger #351 post cffec403、design cfbe183c、res 94442e48/b908731e、final 06644f1f/82009c4a、'
  'MEMO 4b9359ab。\n- **3200 已预注册 = G5-E2 FTR-23 held_out 升格臂（零 GPU，3194 同协议 ×3/3 门）**：'
  'PASS ⇒ v1.22→v1.23 + atlas_v1_25 + upgrade_log #9；亦负 ⇒ E1 不可预测性定论、转 appendix 清算。'),
]
for old, new in pairs:
    assert t.count(old) == 1, (t.count(old), old[:60])
    t = t.replace(old, new)
io.open(m, 'w', encoding='utf-8', newline='').write(t)
assert io.open(m, encoding='utf-8').read() == t
print('MEMORY.md ->', hashlib.sha256(t.encode()).hexdigest()[:8])
