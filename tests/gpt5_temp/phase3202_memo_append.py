# -*- coding: utf-8 -*-
"""Phase 3202 MEMO append (assert prefix)."""
import hashlib
import io

P = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'
old = io.open(P, encoding='utf-8').read()
pre_sha = hashlib.sha256(old.encode('utf-8')).hexdigest()[:8]

ADD = '''

## Phase 3202 — G5-F1 FTR-04 glm4 弱特异性单元级分解复读臂（封存 3181 读数复用，零 GPU，登记臂）【2026-10-10】

**预注册执行（G5-F1 默认臂 (a)）+ 探针确认**：预冻结探针确认封存 3181 result 携带单元级 per_model 字段（loeo_acc/mean_margin/kappa/e90/pr/proj_ratio ×3 模型），臂 (a) 可行，未降级 (b)。审计判据（freeze 前钉定，bar 在探针读数后保守钉定并诚实登记）：R1 封存 3181 标量 4dp 回读 drift==0 ×3（loeo 0.6019/0.5981/0.4800、margin 2.6614/1.4164/5.2326、kappa 48118411.9811/46016538.1628/44953766.4873、pr 8.7763/8.9016/8.9220、proj 0.0763/0.0328/0.0294、e90==8）；R2 FTR-04 p3181 锚 asserts（glm4 kappa/margin/proj + res/seal sha）与封存标量逐位一致；R3 分解 bar：loeo_gap = min(qwen loeo) − glm4 loeo ≥0.10、margin_ratio = glm4 margin / max(qwen margin) ≥1.5、几何带健康（kappa 比差 <25%、pr 比差 <20%、e90 相等）。对照锚：3181 内容双锁（res 1e95a19d / seal 462fd221）+ 3178/3183 双锁 + 3157 tc + 3183 dv 逐位 ×3（设备锚承接）。

**结果**：**RECHECK CLEAN（0 矛盾）⇒ REGISTERED**——R1 drift==0 ×3、R2 逐位一致 OK、R3 **loeo_gap 0.1181 ≥0.10 / margin_ratio 1.9661 ≥1.5 / 几何带 OK**。

**单元级分解定论（写入 FTR-04 scope_limits append，仅此一处变更）**：**glm4-9b 弱特异性 = 读出不一致性，而非几何缺陷**——几何带健康（kappa 比差 0.934、pr 比 1.017、e90 同 8），但 LOEO 留出读出准确率最低 0.4800（vs 0.6019/0.5981）且 mean_margin 最高 5.2326（qwen 最大值的 1.966 倍）——**margin 高而留出准确率低 ⇒ 类对间隔分布散、读出不稳定**；单元级支撑 3181 noise_level 定判（弱特异性不可归因于几何或读出机制缺失），E 级维持 E1_repeatable（无升级）。

**登记**：registry **v1.24 `0d74b628` → v1.25 `3bb33a24`**（append-only：仅 FTR-04 scope_limits append；upgrade_log/failures F1–F19/25 legacy 特征字节保全断言）；atlas_v1_27.html `739faeb7`（**784/784 字段，0/0/0/0**）。

**诚实日志（生成链缺陷，FULL 数据写出前修复）**：①3181 result 顶层为 per_model（无 arm 包装）→ run_recheck 路径修正；②HTML 初版多出 meta.p3181_result span 而 flatten 缺键（extra=1）→ flatten 补键后 784/784。两处均同步修入 create 脚本（marker-splice 模式，规避 3201 的全局改名坑）。产物：design `822996d4`，res `af81e9f4`/seal `480bd271`，final `f75b3a6d`/`dbcaf643`，ledger #354（post `4074e4bc`）。

**图谱现状**：registry v1.25（26 特征：E2×20〔含 E2_held_out ×3〕/E3×3/E1×3）、failures F1–F19、upgrade_log 9、gap v1.6 `0f6423e0`、atlas_v1_27 `739faeb7`、ledger n=354。

**3203 预注册（G5-F2，零 GPU）**——候选两支，freeze 前择一钉定：**(a) GAP-4 残留机制复读臂**：谱外崩塌 2.54×（端口 29% + 残留 1.8×）在 registry v1.25/gap v1.6 口径下回读封存 3169/3172/3175/3188–3192 产物做状态复读 + gap 状态更新提案（report-only，不改 gap 账本本体）；**(b) appendix 可执行性深化**：对 APPX-1/2/3（GPU 依赖）写最小可执行协议草案（零执行、纯文档登记入 FTR-24 scope_limits 或 MEMO）。默认执行 (a)；若封存标量回读失败则降级 (b) 并诚实登记。
'''
new = old + ADD
assert new.startswith(old)
io.open(P, 'w', encoding='utf-8', newline='').write(new)
back = io.open(P, encoding='utf-8').read()
assert back == new
post_sha = hashlib.sha256(back.encode('utf-8')).hexdigest()[:8]
print('MEMO appended: pre %s -> post %s (+%d chars)' % (pre_sha, post_sha, len(ADD)))
