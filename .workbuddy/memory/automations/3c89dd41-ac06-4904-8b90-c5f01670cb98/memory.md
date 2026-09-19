# Automation Execution Memory — AGI_GLM5 续研周期

## 2026-09-18 09:17 运行
- 读取 MEMO 尾部确认最新 Phase = 2872，本轮接续执行 **Phase 2873 + 2874**。
- Phase 2873（attr_fusion，qwen3-4b，13.0s）：锚点 A0 精确复现 acc(B_cf)=0.8875；主判据 A1 = attr_no_class_gain（属性轴融合无类增量，增长率 −0.111）；A3/A4 = qwen4 属性谱类泄漏（候选观察）；A5 隐藏块零干扰但几何偏移 ρ=0.209。
- Phase 2874（attr_geom_crossmodel，Qwen3-14B + GLM4 零前向，12.2s）：三模型属性×类几何独立全部成立（Q1/Q2），qwen4 类泄漏不跨模型复现。
- 产物：tests/glm5/phase2873_attr_fusion.py、phase2874_attr_geom_crossmodel.py；结果在 result/rdc_query_construction_20260913/phase2873|2874/；SHA256 已登记入 MEMO。
- MEMO 已 append（标题 [2026-09-18 09:17]），真实磁盘 UTF-8 复核通过。
- 下轮入口：2875 主选 = 隐藏态属性几何层间扫描（h_l·D 轨迹 + ρ_l 曲线，定位几何断裂层带）。
- 备注：本轮脚本延续 tests/glm5/ 约定（依赖 rdc_construction_common 等框架与 MEMO 登记路径连续性）；自动化提示词中的 tests/gpt5/ 路径与前序 2867-2872 会话实际使用的 glm5 路径冲突，延续了既有约定。
