# -*- coding: utf-8 -*-
"""Phase 3202 daily log append + workspace MEMORY.md update."""
import hashlib
import io

D = r'D:/AI2050/Ai2050-OpenOne/.workbuddy/memory/2026-10-10.md'
old = io.open(D, encoding='utf-8').read()
pre = hashlib.sha256(old.encode()).hexdigest()[:8]
ADD = '''

## Phase 3202 — G5-F1 FTR-04 glm4 弱特异性单元级分解复读臂（REGISTERED）
- R1 3181 标量 4dp 回读 drift==0 ×3；R2 FTR-04 p3181 锚逐位一致；R3 loeo_gap 0.1181 ≥0.10 / margin_ratio 1.9661 ≥1.5 / 几何带 OK ⇒ **RECHECK CLEAN ⇒ REGISTERED**。
- 定论（FTR-04 scope_limits append）：glm4 弱特异性 = **读出不一致性**（margin 最高 5.2326 而 LOEO acc 最低 0.4800 ⇒ 类对间隔分布散），非几何缺陷（kappa/pr/e90 同带）——单元级支撑 3181 noise_level 定判，E1_repeatable 维持。
- registry v1.24 `0d74b628`→v1.25 `3bb33a24`；atlas_v1_27 `739faeb7` 784/784；ledger #354 post `4074e4bc`；design `822996d4`，final `f75b3a6d`/`dbcaf643`；MEMO `4e6a48f4→2a3a5a8f`。
- 诚实日志：3181 顶层 per_model（无 arm 包装）+ flatten 缺 meta.p3181_result 键（extra=1），均 FULL 数据写出前修复并同步 create 脚本（marker-splice 模式）。
- 3203 预注册（G5-F2）：默认 (a) GAP-4 残留机制复读臂（回读封存 3169/3172/3175/3188–3192，report-only）；回读失败降级 (b) appendix 协议草案。
'''
new = old + ADD
assert new.startswith(old)
io.open(D, 'w', encoding='utf-8', newline='').write(new)
back = io.open(D, encoding='utf-8').read()
assert back == new
print('daily: pre %s -> post %s' % (pre, hashlib.sha256(back.encode()).hexdigest()[:8]))

M = r'D:/AI2050/Ai2050-OpenOne/.workbuddy/memory/MEMORY.md'
t = io.open(M, encoding='utf-8').read()
pre_m = hashlib.sha256(t.encode()).hexdigest()[:8]

def rep(old_s, new_s):
    global t
    c = t.count(old_s)
    assert c == 1, ('count', c, old_s[:60])
    t = t.replace(old_s, new_s)

rep('## 2 G 线图谱现状（2026-10-10，registry v1.24 + gap v1.6 + atlas_v1_26.html；',
    '## 2 G 线图谱现状（2026-10-10，registry v1.25 + gap v1.6 + atlas_v1_27.html；**3202 G5-F1 FTR-04 glm4 弱特异性单元级分解复读 REGISTERED**：R1 4dp drift==0 ×3 + R2 锚逐位一致 + R3 loeo_gap 0.1181/margin_ratio 1.9661/几何带 OK ⇒ RECHECK CLEAN；定论：glm4 弱特异性=读出不一致性（margin 5.2326 最高而 LOEO acc 0.4800 最低），非几何缺陷，支撑 3181 noise_level，E1 维持；registry v1.24 0d74b628→v1.25 3bb33a24、atlas_v1_27 739faeb7 784/784、ledger #354 post 4074e4bc；3201 G5-E3 收官审计 CLEAN ⇒ E_LINE_CLOSED：')
rep('ledger n=353（post `ba3d1bcc`）。',
    'ledger n=354（post `4074e4bc`）。')
rep('- **3202 已预注册 = G5-F1（零 GPU）**：默认 (a) FTR-04 glm4 弱特异性归因复读臂（回读封存 3181/3176 做单元级分解登记）；封存字段不足则降级 (b) GAP-4 残留机制复读臂（report-only，不改 gap 账本）。',
    '''- **3202 已完成（G5-F1 REGISTERED，FTR-04 scope_limits append）**：registry v1.25 3bb33a24、atlas_v1_27 739faeb7（784/784）、ledger #354 post 4074e4bc、design 822996d4、res af81e9f4/480bd271、final f75b3a6d/dbcaf643、MEMO 2a3a5a8f。
- **3203 已预注册 = G5-F2（零 GPU）**：默认 (a) GAP-4 残留机制复读臂（回读封存 3169/3172/3175/3188–3192，report-only 提案）；回读失败降级 (b) appendix 协议草案（APPX-1/2/3 最小可执行协议，纯文档）。''')
io.open(M, 'w', encoding='utf-8', newline='').write(t)
back_m = io.open(M, encoding='utf-8').read()
assert back_m == t
print('MEMORY.md: pre %s -> post %s' % (pre_m, hashlib.sha256(back_m.encode()).hexdigest()[:8]))
