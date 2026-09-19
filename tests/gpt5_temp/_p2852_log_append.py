log = r'C:/Users/Admin/WorkBuddy/2026-09-17-01-30-05/.workbuddy/memory/2026-09-17.md'
body = """
## 会话（2026-09-18，Phase 2852：析出正源定位）
- **Phase 2852**（脚本 b2040807，Gen1 49.5s 一次通过；头身份映射 bug 修正后 Gen2 50.2s 正式）：**final_verdict = layered_writers**。S1=true（69.88%，源层 L29/L26/L28）、S2=true（52.6%）、S3 恒等式闭合（rel_err 0.108%）。
- **机制反转发现：无正源层**——inc_layer 全剖面除 L13 clamp 写入（+0.01）外全部为负；负增量主力 = **MLP**（L30-34：mlp −1.15 vs attn +0.08），L26 起渐强、L29-31 峰（−0.30/层）。cdir"析出" = **深层 MLP 透射累积**（读入带偏移状态、输出透传放大），非 attn OV 通道（2851 否证）、非离散消费者（2850）、非深层写入（本 Phase）。S1/S2 的集中语义 = layered_transmitters（透传者非发起者）。
- 2851 K1 fail 完整解释：判据只认 attn（深层 ≈0），真凶 MLP 不在判据内——预注册正确否证 attn 透射假说，2852 补 MLP 通道。
- 口径勘误：2851 delta profile 为 abs 口径（+1.76）、2852 为 signed（−1.70），相容（词间符号高度一致）；后续 profile 默认双口径。2851 vs 2852 组件增量量级差 2-7 倍系 null 词不同 → 钳制强度不同（未受控）。
- MEMO 4383→4433 行（2852 节 @4387）磁盘复核；产物 phase2852/emergence_source/{execution 6b126527, result 9b04cac3, source.npz e8767687}。
- 2853 候选：① MLP 透传增益直接测量（J_mlp·cdir 回投份额逐层谱 + LN 分离）；② MA 词表扩展预研（80→200 词）+ 双谱普查管线封装。
"""
with open(log, 'a', encoding='utf-8') as f:
    f.write(body)
lines = open(log, encoding='utf-8').read().splitlines()
hits = [i + 1 for i, l in enumerate(lines) if '析出正源定位' in l]
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2852_log_append.txt'
open(rep, 'w', encoding='utf-8').write('total=%d hits=%s\n' % (len(lines), hits))
print('log append done')
