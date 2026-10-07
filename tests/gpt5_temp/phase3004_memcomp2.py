# -*- coding: utf-8 -*-
"""MEMORY.md second compression after 3004."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
orig = len(t)

pairs = [
    ('2972 语言主调制×词类特化→2973 fr 塌缩=重写；2977 '
     '双轴注入@L17→2978 剂量窗→2979 转正分布式→2980 h12 '
     '载体→2981 交互=h12 响应→2982 三要素否定（-0.51）→'
     '2983 正交重定向~85%；2984 归宿=局部化；2985 perp='
     '重写；2987 单 token 坍缩≠漂移；2988 普查+卡片 v2；'
     '2989 注册表 lang 分布式/cls L12；2990 L12#2 消融='
     '快照；2991 头谱反相关=路由关系属性。',
     '2972-2991：语言主调制×词类特化/fr塌缩=重写/双轴注'
     '入/剂量窗/转正分布式/h12载体/交互=h12响应/三要素否'
     '定(-0.51)/正交重定向~85%/归宿=局部化/perp=重写/单to'
     'ken坍缩≠漂移/普查卡片v2/注册表/L12#2消融=快照/头谱'
     '反相关=路由关系属性。'),
    ('3000 Ω-F 收官：qwen 处处传播（area 0.829）vs GLM4 '
     '0.036——23.2×；qwen L3 双符号塌缩。',
     '3000 Ω-F：qwen area 0.829 vs GLM4 0.036（23.2×）；'
     'L3 双符号塌缩。'),
    ('3001 Ω-G1：GLM4 鲁棒=词携带 86%+一般洗消；L19 次层'
     '即杀、L4 存续正交化。',
     '3001 Ω-G1：GLM4 鲁棒=词携带 86%+一般洗消（T3 数值'
     '作废见 3004）。'),
]
miss = []
for old, new in pairs:
    if old in t:
        t = t.replace(old, new, 1)
    else:
        miss.append(old[:30])
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem04b.txt', 'w',
        encoding='utf-8').write(
    'miss=%s len %d->%d ok=%s\n' % (
        miss, orig, len(t2), len(t2) <= 3000))
print('ok')
