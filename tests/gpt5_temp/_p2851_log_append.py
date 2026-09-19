log = r'C:/Users/Admin/WorkBuddy/2026-09-17-01-30-05/.workbuddy/memory/2026-09-17.md'
body = """
## 会话（Phase 2846-2851 六连 Phase，MA2 全头因果普查战线收束 + 涌现解剖）
- **2846 全头因果普查**（首战，脚本 456f63b1）：1152 头×80 词，判决 C1/C2=true、C3=false；L13h30 独大 10.12%，指数尾全尺度（28/36 层 R²>幂律），top-64 头（5.6%）承载 51.6% 正载荷，份额⊥必要性 Spearman 0.1496。
- **2847 隐形冠军解剖**（67a3e2bf）：L13h30 双因果角色钉死——早层形成器（cdir 直写≈0、总写出 926×正交）+ 晚层放大器（直写 0.26-0.48）；联合可加性三区：形成器 0.65 < 晚簇 0.93 ≈ 门池 0.96。
- **2848 横向抑制否证**（138ddea8）：绝对口径 0/10 升起（2847 相对口径 4/10 系小分母伪影，勘误入账）；接口 H2=true（门相对下降+输入位移）。
- **2849 高阶门交互**（dce3595c）：G1/G2=false → 门通道仅承载 18% 损伤；词级 ratio 分母病态教训第三次（Σsingles≈0→1e28，改聚合比）。
- **2850 内容消费者三联否证**（bc5e3b01）：J1/J2/J3 全 false——无离散头间内容传递；损伤 = 全局状态几何扰动，cdirchg L30-35 陡增（0.04→0.63）于 L34 峰重组。
- **2851 L30-35 几何重组层解剖**（5c16d430，37.8s，Gen4 正式）：**透射放大假说否证** → diffuse_unresolved。K1 双 fail：frac_late 0.2033（signed）/0.5053（abs）均 <<0.6；top5 贡献头与 census 前缘 0/5 重叠；透射增益仅 0.03-0.06。**深层反直觉：L30-35 attn/mlp 增量 cdir 投影为负（−0.065/−0.59），cdir 析出非深层写入**——位移 cdir 分量 0.373→1.761 增长来自早中层写入+身份携带+整体旋转。分布式画像补完：无魔法头/链/消费者/**重组器**。
- 运行事故全录（均观测前修复）：Gen1 Edit 假落盘（沙箱缺陷，Grep 复核抓住）；Gen2 hs 索引多套一层 [1]；Gen3 max(x,1e-30) 负分母反噬 → 1e28 病态（2841 教训第四次变体，纪律升级：signed 分母禁用 max 防零，须显式 abs 阈值）。
- MEMO 4330→4383 行（2851 节 @4334），磁盘复核；产物 phase2851/emergence_anatomy/{execution 345690ea, result 08b6c6f4, emergence.npz 0769bbe8}。
- 2852 候选：① 析出正源定位（L14-29 逐层增量剖面 + LN 旋转效应检验）；② MA 词表扩展预研（80→200 词）+ 双谱普查管线封装。
"""
with open(log, 'a', encoding='utf-8') as f:
    f.write(body)
lines = open(log, encoding='utf-8').read().splitlines()
hits = [i + 1 for i, l in enumerate(lines) if '透射放大假说否证' in l]
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2851_log_append.txt'
open(rep, 'w', encoding='utf-8').write('total=%d hits=%s\n' % (len(lines), hits))
print('log append done')
