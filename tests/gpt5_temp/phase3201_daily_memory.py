# -*- coding: utf-8 -*-
"""Phase 3201 daily log append + workspace MEMORY.md update."""
import hashlib
import io

# ---- daily (append-only) ----
D = r'D:/AI2050/Ai2050-OpenOne/.workbuddy/memory/2026-10-10.md'
old = io.open(D, encoding='utf-8').read()
pre = hashlib.sha256(old.encode()).hexdigest()[:8]
ADD = '''

## Phase 3201 — G5-E3 E 线收官审计臂（E_LINE_CLOSED）
- 审计 A1–A3 全 OK（0 矛盾）：A1 值域+计数 E2_held_out 3/E3 3/E1 3/E2_predictive 17；A2a held_out 锚 FTR-23/25/26；A2b primary_intervention FTR-10/11/16；A2c counter_evidence FTR-04/14/24；A3 升级链 9 序列逐位符合 ⇒ **E_LINE_CLOSED**。
- A4：FTR-24 scope_limits append（held-out 几何分界定论 + E1 残余记录完备 + E2_held_out 口径注记），其余全字节保全；appendix 5 项全 GPU/跨线依赖，零 GPU 候选集空。
- registry v1.23 `5f5b9e73`→v1.24 `0d74b628`；atlas_v1_26 `fbc064d4` 783/783；ledger #353 post `ba3d1bcc`；design `38fbdd1a`，final `c0350500`/`25cad189`；MEMO `989fd907→4e6a48f4`。
- 诚实日志：生成链三层缺陷（三引号定界冲突/双反斜杠行续符、P17 全局改名波及新代码、HTML 丢 v1.20 span）均 FULL 数据写出前修复，脚本 `phase3201_create_from_3200.py` + `phase3201_fix_rename.py`。
- 3202 预注册（G5-F1）：默认 (a) FTR-04 glm4 弱特异性归因复读臂；封存字段不足则降级 (b) GAP-4 复读臂。
'''
new = old + ADD
assert new.startswith(old)
io.open(D, 'w', encoding='utf-8', newline='').write(new)
back = io.open(D, encoding='utf-8').read()
assert back == new
print('daily: pre %s -> post %s' % (pre, hashlib.sha256(back.encode()).hexdigest()[:8]))

# ---- workspace MEMORY.md (in-place section update) ----
M = r'D:/AI2050/Ai2050-OpenOne/.workbuddy/memory/MEMORY.md'
t = io.open(M, encoding='utf-8').read()
pre_m = hashlib.sha256(t.encode()).hexdigest()[:8]

def rep(old_s, new_s):
    global t
    c = t.count(old_s)
    assert c == 1, ('count', c, old_s[:60])
    t = t.replace(old_s, new_s)

rep("registry v1.5 `e1942d03`、atlas_v1_6 `b95e67a2`、ledger n=330 chain `5851a521`。",
    "registry v1.5 `e1942d03`、atlas_v1_6 `b95e67a2`、ledger n=330 chain `5851a521`。（G 线主图谱此后持续推进，最新见 §3/§4）")
rep("- **3178 A2 归因 FAIL×3（null p 0.019–0.026 ≪ r 0.092/0.095/0.129 ⇒ 关系特异残差确证，registry v1.5）**。",
    """- **3178 A2 归因 FAIL×3（null p 0.019–0.026 ≪ r 0.092/0.095/0.129 ⇒ 关系特异残差确证，registry v1.5）**。
- 3198 D 线收官（FTR-26 入册，registry v1.21）；3199 FTR-24 NOT_PREDICTIVE（v1.22）；3200 FTR-23 HOLDOUT_PREDICTIVE 升 E2_held_out（v1.23 `5f5b9e73`）；**3201 G5-E3 收官审计 CLEAN ⇒ E_LINE_CLOSED（registry v1.24 `0d74b628`、atlas_v1_26 `fbc064d4` 783/783、ledger #353 post `ba3d1bcc`）**。E 线定论：held-out 可预测性几何分界——类均值/关系内结构承载，关系间偏移与关系方向不承载（写入 FTR-24 scope_limits）。""")
rep("- **3179 已预注册 = G5-B4 FTR-23 立项入册**（零 GPU）：T_C 关系特异残差升格独立特征 E1_repeatable，registry v1.5→v1.6 + atlas_v1_7.html；锚 p3178 4f521d9b/1f166645 + p3177 d451030f/7b746eb2 + p3157 collect 95a25965/9552086d/23dd74eb；statement 草案见 MEMO 3178 节（freeze 前定稿）。",
    "- **3202 已预注册 = G5-F1**（零 GPU）：默认 (a) FTR-04 glm4 弱特异性归因复读臂（回读封存 3181/3176）；封存字段不足降级 (b) GAP-4 残留机制复读臂（report-only）。")
io.open(M, 'w', encoding='utf-8', newline='').write(t)
back_m = io.open(M, encoding='utf-8').read()
assert back_m == t
print('MEMORY.md: pre %s -> post %s' % (pre_m, hashlib.sha256(back_m.encode()).hexdigest()[:8]))
