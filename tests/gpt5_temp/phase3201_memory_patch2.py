# -*- coding: utf-8 -*-
"""Phase 3201 workspace MEMORY.md update (match actual content)."""
import hashlib
import io

M = r'D:/AI2050/Ai2050-OpenOne/.workbuddy/memory/MEMORY.md'
t = io.open(M, encoding='utf-8').read()
pre = hashlib.sha256(t.encode()).hexdigest()[:8]

def rep(old_s, new_s):
    global t
    c = t.count(old_s)
    assert c == 1, ('count', c, old_s[:60])
    t = t.replace(old_s, new_s)

rep('## 2 G 线图谱现状（2026-10-10，registry v1.23 + gap v1.6 + atlas_v1_25.html）',
    '## 2 G 线图谱现状（2026-10-10，registry v1.24 + gap v1.6 + atlas_v1_26.html；**3201 G5-E3 收官审计 CLEAN ⇒ E_LINE_CLOSED**：A1 计数 E2_held_out 3/E3 3/E1 3/E2_pred 17 + A2a/b/c 锚齐备 + A3 升级链 9 逐位符合，0 矛盾；FTR-24 scope_limits append held-out 几何分界定论（类均值/关系内结构承载、关系间偏移与关系方向不承载）；appendix 5 项全 GPU/跨线依赖零 GPU 候选集空；registry v1.23 5f5b9e73→v1.24 0d74b628、atlas_v1_26 fbc064d4 783/783、ledger #353 post ba3d1bcc）')
rep('- ledger n=352（post `8964e20d`）。',
    '- ledger n=353（post `ba3d1bcc`）。') if '- ledger n=352（post `8964e20d`）。' in t else None
rep('- **3201 已预注册 = G5-E3 E 线收官审计臂（零 GPU）**：26 特征锚齐备性审计 + FTR-24 scope_limits append（held-out 可预测性几何分界定论）+ appendix 5 项盘点；0 矛盾 ⇒ E_LINE_CLOSED（v1.23→v1.24 + atlas_v1_26）。',
    '''- **3201 已完成（G5-E3 AUDIT CLEAN ⇒ E_LINE_CLOSED）**：registry v1.24 0d74b628、atlas_v1_26 fbc064d4（783/783）、ledger #353 post ba3d1bcc、design 38fbdd1a、res 1dc22531/30b08ff8、final c0350500/25cad189、MEMO 4e6a48f4。
- **3202 已预注册 = G5-F1（零 GPU）**：默认 (a) FTR-04 glm4 弱特异性归因复读臂（回读封存 3181/3176 做单元级分解登记）；封存字段不足则降级 (b) GAP-4 残留机制复读臂（report-only，不改 gap 账本）。''')
io.open(M, 'w', encoding='utf-8', newline='').write(t)
back = io.open(M, encoding='utf-8').read()
assert back == t
print('MEMORY.md: pre %s -> post %s' % (pre, hashlib.sha256(back.encode()).hexdigest()[:8]))
