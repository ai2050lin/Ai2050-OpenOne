# -*- coding: utf-8 -*-
import io

MP = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\MEMORY.md')
mm = io.open(MP, encoding='utf-8').read()
pairs = (
    ('2998 Ω-F2 GLM4 版负：xdir 注入 L17-19 sep 免疫（0.027 vs 0.86，L19 反向）~98% 洗消；类分离由词 token 携带；跨模型门须无量纲。',
     '2998 Ω-F2 GLM4 版负：xdir 注入 L17-19 sep 免疫（0.027 vs 0.86）~98% 洗消；类分离由词 token 携带；门须无量纲。'),
    ('2999 扫描：无层位偏移带（max 0.128@L4），s_c>6 全局强洗消；弱带 L2-4<L7-13<qwen L15-17。',
     '2999 扫描：无层位偏移带（max 0.128@L4）s_c>6 全局洗消；弱带 L2-4<L7-13<qwen L15-17。'),
    ('3000 Ω-F 收官：qwen 处处传播（band 25/28、area 0.829、a9 vs 2945 bit 级 0.0）vs GLM4 area 0.036 band 2——刻度差 23.2×；',
     '3000 Ω-F 收官：qwen 处处传播（band 25/28、area 0.829、a9 bit 级 0.0）vs GLM4 0.036/2——刻度差 23.2×；'),
    ('3001 Ω-G1：GLM4 鲁棒=词 token 携带（word_eff 86% vs ctx 23%）+一般性洗消（随机方向比 xdir 更弱=非选择性）；L19 注入下一层即杀、L4 幅度存续但 xdir 分量正交化。',
     '3001 Ω-G1：GLM4 鲁棒=词 token 携带（word_eff 86% vs ctx 23%）+一般性洗消（随机方向更弱=非选择性）；L19 注入次层即杀、L4 存续但 xdir 分量正交化。'),
    ('- max=3001，下一个 3002（A 主选 qwen 同款词 token swap 加法分解（跨模型镜像）；B L4 存续/L20 即杀机制分界；C 2989 T3 加密+k 剂量；D Ω-E 错误吸引子）。',
     '- max=3001，下一个 3002（A 主选 qwen 词 token swap 分解镜像；B L4 存续/L20 即杀分界；C 2989 T3 加密+k 剂量；D Ω-E 错误吸引子）。'),
    ('2996 审计：2995 dirs 全层退化（单槽 hook）→T3 作废；真 K2 z +10~31、K1 share p=0——注册表双口径双模型在场。',
     '2996 审计：2995 dirs 全层退化（单槽 hook）→T3 作废；真 K2 z +10~31、K1 share p=0——双口径双模型在场。'),
    ('- 判据可达性先检（永真禁用）；margin n≳40+粒度；quasi-post-hoc 标注；置换 p 粒度×family 先验，大 family maxT；显著集重叠须 null 校准；退化统计量加非退化门；',
     '- 判据可达性先检（永真禁用）；margin n≳40+粒度；quasi-post-hoc 标注；置换 p 粒度×family 先验，大 family maxT；显著集重叠 null 校准；退化统计量加非退化门；'),
    ('- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链或上游全精度；round 按精度设门；大内积相对阈 1e-8；跨相位锚优先 max|Δ| vs 上 Phase 产物。',
     '- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链或上游全精度；round 按精度设门；大内积阈 1e-8；跨相位锚优先 max|Δ| vs 上 Phase 产物。'),
    ('- bf16 batch 组成敏感：跨相位锚 bit 级一致；条件独立 batch 不拼接；分母守卫取最大幅度；单置换+位置切分，禁双置换（2995）。',
     '- bf16 batch 组成敏感：跨相位锚 bit 级一致；条件独立 batch 不拼接；分母守卫取最大幅度；单置换+位置切分（2995）。'),
)
for a, b in pairs:
    if a in mm:
        mm = mm.replace(a, b, 1)
    else:
        print('MISS', a[:20])
io.open(MP, 'w', encoding='utf-8').write(mm)
mm2 = io.open(MP, encoding='utf-8').read()
assert '3001 Ω-G1' in mm2 and 'max=3001' in mm2
print('chars=%d' % len(mm2))
