# -*- coding: utf-8 -*-
"""Phase 3013 MEMORY patch (append + equal-length
compression, direct disk writes)."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
n0 = len(t)

pairs = [
    # header range bump
    ('## 机制链状态（2936-3011）', '## 机制链状态（2936-3013）'),
    # compress 2938-2970 chain
    ('2938-2970：子空间/词盲/线性壳/阈值/头集中/组反转/'
     '重平衡/承重/路由跳变/秩1/签名/载体h15/消融/塌缩/峰锁/'
     'fr晚；',
     '2938-2970：子空间/词盲/线性壳/阈值/头集中/组反转/'
     '承重/路由跳变/秩1/签名/载体h15/消融/塌缩/峰锁/fr晚；'),
    # compress 2972-2991 chain
    ('2972-2991：主调制×词类特化/fr塌缩=重写/双轴注入/'
     '剂量窗/转正/h12载体/交互=h12/三要素否定/正交重定向/'
     '局部化/perp=重写/坍缩≠漂移/普查/注册表/快照/头谱='
     '路由属性。',
     '2972-2991：主调制×词类特化/fr塌缩=重写/双轴注入/'
     '剂量窗/转正/h12载体/三要素否定/正交重定向/局部化/'
     'perp=重写/坍缩≠漂移/普查/注册表/头谱=路由属性。'),
    # compress 2992-3000
    ('2992 字典：符号率 0.405=快照。2993 逻辑签名长度稳健；'
     '2994 轴防御。2995-2999 Ω-F GLM4：重分级；xdir ~98% '
     '洗消。3000：qwen/GLM4 23.2×；L3 塌缩。',
     '2992 字典：符号率 0.405=快照。2993 逻辑签名长度稳健；'
     '2994 轴防御。2995-2999 Ω-F：重分级；xdir ~98% 洗消。'
     '3000：qwen/GLM4 23.2×。'),
    # compress 3001-3004
    ('3001/3002 Ω-G：词携带 GLM4 86% vs qwen 63%；中带 xdir '
     '特异~150×；3003 Ω-G3：中带=xdir 特异剂量线性符号不对称'
     '（−xdir 弱 34-45%）；方案 v5。3004 Ω-P4：算子共享低秩'
     '（.715/.826）共享向≠xdir=重定向+放大；**T3 轴序 bug '
     'bit 级**修正。',
     '3001/3002 Ω-G：词携带 GLM4 86% vs qwen 63%；中带 xdir '
     '特异~150×；3003：中带=xdir 特异剂量线性符号不对称；'
     '方案 v5。3004 Ω-P4：算子共享低秩（.715/.826）共享向≠'
     'xdir=重定向+放大；**T3 轴序 bug bit 级**修正。'),
    # compress 3005-3006
    ('3005：GLM4 r1 .189=逐细胞散射；**闭环：qwen 构造性 vs '
     'GLM4 破坏性**。3006 Base：**中带与 chat 同构=预训练'
     '涌现非对齐雕刻**（r1 .656/.784；尾 1.69×）。',
     '3005：GLM4 r1 .189=逐细胞散射；**闭环：qwen 构造性 vs '
     'GLM4 破坏性**。3006 Base：**中带与 chat 同构=预训练'
     '涌现**（r1 .656/.784；尾 1.69×）。'),
    # compress 3007-3008 + fix missing period
    ('3007 Ω-P2a：锁定比 0.457（11.74<25.70）；扰动 2/4 '
     'divergent 但 3/4 零分歧 3008 Ω-P2b：**held-out 轴生成'
     '态分离 logic>content（p 0.0002×2）**；KV 零化饱和→'
     '分级干预。',
     '3007 Ω-P2a：锁定比 0.457（11.74<25.70）；扰动 2/4 '
     'divergent 但 3/4 零分歧。3008 Ω-P2b：**held-out 轴'
     '生成态分离 logic>content（p 0.0002×2）**；KV 零化饱和。'),
    # compress 3009-3010
    ('3009 Ω-P2c：KV 饱和与强度无关→token 分歧因果工具退役；'
     '坐标解离 logic<content<sham。 3010 Ω-P2d：换读出保干预→'
     '逻辑位因果特异性确立（D=0.0297 p=0.0135 四尺度）；读出'
     '解离=c8 vs JS 反序=门控分布边界。',
     '3009 Ω-P2c：KV 饱和与强度无关→token 分歧因果工具退役；'
     '坐标解离 logic<content<sham。3010 Ω-P2d：换读出保干预→'
     '逻辑位因果特异性确立（D=0.0297 p=0.0135）；读出解离='
     '门控分布边界。'),
    # compress 3011-3012
    ('3011 Ω-P2e：**门控定位于 L3 KV**（D_l*=0.0137 '
     'p_maxT=1e-4；2×次峰 L31；K/V 双臂；早层写+中带调。 '
     '3012 Ω-P2f：**L3 门控=混合**——mean 回填不恢复（−0.026 '
     'p=0.507）、噪声 0.256<0.5≈1/4 容量+内容特异主体；'
     'content 位 KV 可互换（JS 0.00025）logic 位不可；手术='
     '可破坏不可伪造。',
     '3011 Ω-P2e：**门控定位于 L3 KV**（D_l*=0.0137 '
     'p_maxT=1e-4；2×次峰 L31；K/V 双臂）；早层写+中带调。'
     '3012 Ω-P2f：**门控=混合**——mean 回填不恢复（−0.026）、'
     '噪声 0.256<0.5≈1/4 容量+内容特异主体；content 位 KV 可'
     '互换 logic 位不可；手术=可破坏不可伪造。'),
    # next-step replacement
    ('- max=3012，下一个 3013（A 主选 logic 位 K,V 内容分解——'
     '低秩结构定位"不可统计替代"；B L31 次峰；C 反向手术剂量律；'
     'D GLM4-9B-Base 对照）。方案 v5。',
     '- max=3013，下一个 3014（A 主选反向手术剂量律——L3 KV '
     '部分擦除 s 网格 JS 曲线+信息/容量分离定量；B L31 次峰；'
     'C 情景性检验：同词异位 K,V 相似度；D GLM4-9B-Base 对照）。'
     '方案 v5。'),
]
miss = []
for a, b in pairs:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        miss.append(a[:20])

# append 3013 after the 3012 sentence (end of line 42)
anchor = ('content 位 KV 可'
          '互换 logic 位不可；手术=可破坏不可伪造。')
add = (' 3013 Ω-P2g：**门控内容=位置特异高维（情景式非类'
       '码）**——类均值回填也不恢复（−0.036 p=0.75）、LOO '
       '秩 k=8 仅 0.397、噪声 0.256>两均值臂；logic 类更紧'
       '（cos 0.914 vs 0.847）但均值≠内容。')
if anchor in t and '3013 Ω-P2g' not in t:
    t = t.replace(anchor, anchor + add, 1)
else:
    miss.append('APPEND-ANCHOR')

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open((r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\tmp_mem13.txt'), 'w',
        encoding='utf-8').write(
    'len=%d (was %d) miss=%s has3013=%s'
    % (len(t2), n0, miss, '3013 Ω-P2g' in t2))
print('ok')
