# -*- coding: utf-8 -*-
"""Phase 3014 memory compression round 3."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
miss = []


def rep(old, new):
    global t
    if old in t:
        t = t.replace(old, new, 1)
    else:
        miss.append(old[:30])


rep('- 判据可达性先检（永真禁用）；margin n≳40+粒度；'
    'quasi-post-hoc 标注；置换 p 粒度×family 先验，大 family '
    'maxT；显著集重叠 null 校准；退化统计量加非退化门；maxT '
    '选拔层与 rho 结构层分账；镜像 −dirs 对照必配；功能主张'
    '三层分账，承重报效应量+剖面。',
    '- 判据可达性先检（永真禁用）；margin n≳40+粒度；quasi-'
    'post-hoc 标注；置换 p 粒度×family 先验，大 family maxT；'
    '显著集重叠 null 校准；退化统计量加非退化门；镜像 −dirs '
    '对照必配；功能主张三层分账报效应量。')
rep('- norm 分解（能量 vs cos）→口径标签（禁跨口径混用）→'
    '子空间检查→SVD 能量迁移→组内相关混淆检查→倍率同时报'
    '分子分母绝对剖面→剂量响应分单调/峰位两检验→消融差分='
    '直接+重平衡（符号可反）。',
    '- norm 分解（能量 vs cos）→口径标签（禁混用）→子空间'
    '检查→SVD 能量迁移→组内相关混淆→倍率报分子分母→剂量分'
    '单调/峰位→消融差分=直接+重平衡。')
rep('- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python '
    '补丁（本会话两遇）；replace 未命中→先 Grep。',
    '- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python '
    '补丁（本会话三遇+补丁锚 miss 两次）；replace 未命中→'
    '先 Grep 再跑。')
rep('- bf16 batch 组成敏感：跨相位锚 bit 级一致；条件独立 '
    'batch 不拼接；分母守卫取最大幅度；单置换+位置切分（2995）。',
    '- bf16 batch 组成敏感：跨相位锚 bit 级一致；条件独立 '
    'batch 不拼接；bf16 禁转 numpy（改 torch Generator）。')
rep('- numpy 标量入 json 转 int()/float()；import 遮蔽→别名。',
    '- numpy 标量入 json 转 int()/float()；dict 键容器禁直接 '
    'np.array；大小写敏感常量对齐（3014 JOINT bug）。')

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
chk = {'has3014': '3014 Ω-P2h' in t2,
       'max14': 'max=3014' in t2}
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem14c.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s chk=%s'
    % (len(t2), miss,
       {k: v for k, v in chk.items() if not v}))
print('ok')
