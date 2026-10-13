# -*- coding: utf-8 -*-
"""Phase 3201 MEMO append (assert prefix)."""
import hashlib
import io

P = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'
old = io.open(P, encoding='utf-8').read()
pre_sha = hashlib.sha256(old.encode('utf-8')).hexdigest()[:8]

ADD = '''

## Phase 3201 — G5-E3 E 线收官审计臂（封存 registry v1.23 复读，零 GPU，登记臂）【2026-10-10】

**预注册钉定 + 审计判据 freeze 前实测细化（诚实登记）**：对 registry v1.23 全 26 特征做钉定审计 A1–A3（判据在预冻结探针封存 v1.23 anchor role 分布后钉定）：A1 evidence_level 值域 {E0_candidate,E1_repeatable,E2_predictive,E2_held_out,E3_causal_scoped} + 计数 E2_held_out==3 / E3==3 / E1==3 / E2_predictive==17；A2a FTR-23/25/26 各有 role=held_out 锚；A2b FTR-10/11/16 各有 role=primary_intervention 锚；A2c FTR-04/14/24 各有 ≥1 counter_evidence；A3 upgrade_log 节点序列 == 9 条钉定序列（N04→FTR-13、N05→FTR-12、N13→FTR-16、FTR-06、FTR-08、FTR-15、FTR-25、FTR-26、FTR-23）。门（freeze 前钉定）：A1–A3 审计 0 矛盾 且 A4 append 验证（FTR-24 scope_limits startswith + 25 legacy 特征 + upgrade_log + failures 字节保全）⇒ **E_LINE_CLOSED**（registry v1.23→v1.24 仅 FTR-24 scope_limits 变更 + atlas_v1_26.html）；否则逐条登记 gap report-only。对照锚：3178 标量 4dp 回读 drift==0 ×3（ANCH_3178 自 3200 承接）+ 3183 dv 逐位 + 3157 tc drift<1e-9。

**结果**：**AUDIT CLEAN（0 矛盾）⇒ E_LINE_CLOSED**——A1 值域+计数 OK（E2_held_out 3 / E3 3 / E1 3 / E2_predictive 17）、A2a/A2b/A2c 锚齐备全 OK、A3 升级链 9 序列逐位符合。**E 线（3198 D 线收官 → 3199 FTR-24 NOT_PREDICTIVE → 3200 FTR-23 HOLDOUT_PREDICTIVE → 3201 收官审计）完整闭环**。

**A4 收官定论（写入 FTR-24 scope_limits append，仅此一处变更）**：held-out 可预测性的几何分界——类均值/关系内结构承载（FTR-25/26 OOV 面板 med|cos|≥0.3、FTR-23 关系内实体留出 med cos 0.74–0.79），关系间偏移方向与关系方向不承载跨面板预测（FTR-24 dvu med|cos_held| 0.035–0.060 ≪0.3）；E1 残余可证伪记录完备（FTR-04 noise_level 已闭 3181、FTR-14 F16 gate fail 3177+3178 归因、FTR-24 counter_evidence #4）；口径注记：E2_held_out 为 E2_predictive 的 held-out 强化档（FTR-23/25/26），legacy 值字节保全不动。**appendix 盘点：5 项全部 GPU/跨线依赖，零 GPU 候选集为空**（APPX-1/2/3 GPU、APPX-4/5 跨线）。

**登记**：registry **v1.23 `5f5b9e73` → v1.24 `0d74b628`**（append-only：仅 FTR-24 scope_limits append；upgrade_log/failures F1–F19/25 legacy 特征字节保全断言）；atlas_v1_26.html `fbc064d4`（**783/783 字段，0/0/0/0**）。

**诚实日志（生成链缺陷，FULL 前修复）**：①拷贝补丁脚本 P9/P10 段三引号定界冲突（嵌入代码行尾单引号与三引号终止符粘连）→ 改双引号三连/raw 定界；P10 双反斜杠行续符失配 → 单反斜杠修正；P17 残留 G_MED 定义 → 补删除段。②P17 全局改名 reg23→reg24 过度波及新插入代码（preservation asserts 自比较恒真）→ `phase3201_fix_rename.py` 10 处定向恢复（逐处 assert count==1）。③HTML 头初版丢了 v1.20 span 与 flatten 键 meta.registry_v120_sha8 失配（missing=1）→ span 回补后 783/783。三层缺陷均先于 FULL 数据写出修复，SMOKE 阶段 A1–A3 与 FULL 一致 CLEAN。

**图谱现状（E 线收官定版）**：registry v1.24（26 特征：E2×20〔含 E2_held_out 强化档 ×3〕/E3×3/E1×3）、failures F1–F19、upgrade_log 9、gap v1.6 `0f6423e0`、atlas_v1_26 `fbc064d4`、ledger n=353。

**产物**：design `38fbdd1a`，res `1dc22531`/seal `30b08ff8`，final `c0350500`/`25cad189`，ledger #353（post `ba3d1bcc`）。

**3202 预注册（G5-F1，零 GPU，E 线后首轮新臂）**——候选两支，freeze 前择一钉定：**(a) FTR-04 glm4 弱特异性归因臂**：FTR-04 noise_level 已闭（3181）但 glm4 单元特异性读数偏弱，回读封存 3181/3176 产物做单元级分解登记（无新计算，纯复读臂）；**(b) GAP-4 残留机制复读臂**：谱外崩塌 2.54×（端口 29% + 残留 1.8×）在 registry v1.24/gap v1.6 口径下复读并出 gap 状态更新提案（report-only，不改 gap 账本）。默认执行 (a)；若封存产物单元级字段不足则降级 (b) 并诚实登记。
'''
new = old + ADD
assert new.startswith(old)
io.open(P, 'w', encoding='utf-8', newline='').write(new)
back = io.open(P, encoding='utf-8').read()
assert back == new
post_sha = hashlib.sha256(back.encode('utf-8')).hexdigest()[:8]
print('MEMO appended: pre %s -> post %s (+%d chars)' % (pre_sha, post_sha, len(ADD)))
