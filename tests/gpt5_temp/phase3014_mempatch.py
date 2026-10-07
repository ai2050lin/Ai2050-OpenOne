# -*- coding: utf-8 -*-
"""Phase 3014 memory patch: append 3014 with equal
compression of 3012/3013 entries."""
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


# compress 3012/3013 lines
rep('3012 Ω-P2f：**门控=混合**——mean 回填不恢复（−0.026）、'
    '噪声 0.256<0.5≈1/4 容量+内容特异主体；content 位 KV 可'
    '互换 logic 位不可；可破坏不可伪造。',
    '3012 Ω-P2f：**门控=混合**——mean 回填不恢复（−0.026）、'
    '噪声 0.256≈1/4 容量+内容特异主体；content 位 KV 可互换 '
    'logic 位不可。')
rep('3013 Ω-P2g：**门控内容=位置特异高维（情景式非类码）**'
    '——类均值回填不恢复（−0.036 p=0.75）、LOO 秩 k=8 仅 '
    '0.397、噪声>两均值臂；logic 类更紧（0.914 vs 0.847）'
    '但均值≠内容。',
    '3013 Ω-P2g：**门控内容=位置特异高维情景式非类码**——'
    '类均值回填不恢复（−0.036）、LOO 秩 k=8 仅 0.397、噪声>'
    '两均值臂；logic 更紧（0.914 vs 0.847）但均值≠内容。')

# append 3014
rep('但均值≠内容。',
    '但均值≠内容。 3014 Ω-P2h：**破坏剂量非单调且门控脆弱**'
    '——retain_joint(0.5)=1.011≥0.75（半擦除≥全擦除，半幅 KV'
    '毒信号）；**K/V 分岔**：KONLY 非单调（1.018/0.971/0.241，'
    'softmax 竞争）vs VONLY 单调（0.370/0.177/0.054，线性）'
    '——破坏=K 路由通道；sham 校准干净；npz8 跨 run4/5 位级。'
    '**手术不对称定标：破坏粗粒度易行，伪造须情景 K,V。**')

# header range + next step
rep('## 机制链状态（2936-3013）',
    '## 机制链状态（2936-3014）')
rep('- max=3013，下一个 3014（A 主选反向手术剂量律——L3 KV '
    '部分擦除 s 网格+信息/容量分离定量；B L31 次峰；C 情景性'
    '检验（同词异位 K,V 相似度）；D GLM4-9B-Base 对照）。',
    '- max=3014，下一个 3015（A 主选 K 路由机制定位——L3 logic '
    '位 K 的下游消费头/attention 分配与 softmax 熵；B L31 次峰；'
    'C 情景性检验（同词异位 K,V 相似度）；D GLM4-9B-Base 对照）。')

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
chk = {
    'has3014': '3014 Ω-P2h' in t2,
    'max14': 'max=3014' in t2,
    'no13step': 'max=3013' not in t2,
}
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem14.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s chk=%s'
    % (len(t2), miss,
       {k: v for k, v in chk.items() if not v}))
print('ok')
