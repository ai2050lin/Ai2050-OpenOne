log = r'C:/Users/Admin/WorkBuddy/2026-09-17-01-30-05/.workbuddy/memory/2026-09-17.md'
body = """
## 会话（2026-09-18 续，Phase 2853：MLP 透传增益谱）
- **Phase 2853**（脚本 70d3d82a，36.3s 一次干净通过，max_resid 0.0101）：**T1 = active_amplification**。g_jac（数值 Jacobian cdir 二次型，ε=1.0 预注册）L26-35 = [0.99, 0.68, 5.08, 12.10, 17.93, 11.16, 13.40, 23.68, **51.26(L34)**, 23.06]——深层 MLP 对 cdir 有 5-51× 本征小信号放大率（L28 起激活）。
- **g_jac vs g_emp 差 10-40 倍、r=−0.26**：甄别为工作点分离（clamp 位移把 mlp 推入 SwiGLU 饱和区；线性度比 1.98 亚线性独立证实）+ J 非对称 + din 非 cdir 分量非线性混合。经验透传被**饱和制动**为温和增益（g_emp 0.2-1.7）→ 逐层累积 −1.7（2852 析出曲线）。
- din_cdir 0.086→0.665 单调负增（输入偏移积累）；dout_cdir L35 转正（+0.07）——最深层的回拉力，与 2851 抑制性一致。
- **硬伤入账：缺随机方向对照**（g_jac 大值可能含 J 谱背景各向异性，cdir 特殊性未判 z-score）→ 2854 首选补全。
- 终版机制：**饱和制动的透射放大晶格（saturated transmission lattice）**——本征放大器阵列+饱和限幅+早层发起+全员透传。2846→2853 全链闭环：无魔法头/链/消费者/重组器/正源层，放大器硬件真实存在但被饱和制动。
- MEMO 4433→4483 行（2853 节 @4437）磁盘复核；产物 phase2853/mlp_transmission/{execution c1be60ce, result 13c3b7e3, transmission.npz 19e2d8f6}。
- 2854 候选：① 随机方向对照 + clamp 态工作点 g_jac'（分离工作点移动假说 + cdir 特殊性 z-score）；② MA 词表扩展预研 + 双谱普查管线封装。
"""
with open(log, 'a', encoding='utf-8') as f:
    f.write(body)
lines = open(log, encoding='utf-8').read().splitlines()
hits = [i + 1 for i, l in enumerate(lines) if 'MLP 透传增益谱' in l]
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2853_log_append.txt'
open(rep, 'w', encoding='utf-8').write('total=%d hits=%s\n' % (len(lines), hits))
print('log append done')
