log = r'C:/Users/Admin/WorkBuddy/2026-09-17-01-30-05/.workbuddy/memory/2026-09-17.md'
body = """
## 会话（2026-09-18 续，Phase 2854：增益特殊性对照与重大勘误）
- **Phase 2854**（脚本 302f68f7，Gen3 43s 正式）：**final_verdict = anisotropic_background**。Z1=false（z 中位 −0.23，cdir 相对 16 随机方向/词无特殊性）；Z2=false（ratio 0.84-1.19，clamp 态与 base 态切线增益相同，无工作点移动）。
- **重大勘误：2853 T1=active_amplification 作废**——mlp_batch 缺 post_attention_layernorm（Qwen3MLP 无内部 LN，真实前向 mlp(LN2(h))），基线错配 mlp_raw vs mlp(LN2) 产生 5-51× 伪影（g_rand MAD 几十 + g_cdir 系统偏移的观测结构完全吻合 Δ0 伪影模型）。"饱和制动的透射放大晶格"命名作废。
- **机制终版：被动透传带（passive transmission band）**——LN2+MLP 复合路径 cdir 增益 ~±1（切线 −0.8~+1.8，经验 0.2-1.7），cdir 分量积累 = din 链式传播 × 透传增益；"陡增"真源 = L29-31 增益 >1 窗口。分布式画像定稿：无魔法头/链/消费者/重组器/正源层/cdir 特异放大器。
- **第五次负分母防零教训**：Gen2 ratio 用 np.maximum(gcb,1e-9) → 负分母返回 1e-9 → ratio ±4e8 病态、Z2 误判；修 abs 阈值直接除。教训制度化入 MEMO（含 LN 边界、清理脚本路径核对）。
- **操作事故与恢复**：清理脚本路径复制错 → 误删 phase2852 产物；立即重跑，result.json/npz SHA 与原登记逐位一致（9b04cac3/e8767687，确定性复现实证），仅 execution.json 更新（3642fb61）。
- MEMO 4483→4527 行（2854 节 @4487）磁盘复核；2854 产物 {execution 6c3ec81b, result 7347efad, specificity.npz b0f5ac17}。
- 2855 候选：① L29-31 增益窗口解剖（LN2/SwiGLU 分解 + 逐神经元 top-k）；② MA 词表扩展预研 + 双谱普查管线封装。
"""
with open(log, 'a', encoding='utf-8') as f:
    f.write(body)
lines = open(log, encoding='utf-8').read().splitlines()
hits = [i + 1 for i, l in enumerate(lines) if '增益特殊性对照' in l]
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2854_log_append.txt'
open(rep, 'w', encoding='utf-8').write('total=%d hits=%s\n' % (len(lines), hits))
print('log append done')
