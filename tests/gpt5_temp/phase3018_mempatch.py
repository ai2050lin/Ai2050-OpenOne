# -*- coding: utf-8 -*-
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
orig_len = len(t)

# --- compress 3013 entry (slice between markers) ---
i = t.index('3013 Ω-P2g')
j = t.index('3014 Ω-P2h')
t = t[:i] + '3013：门控=位置特异情景式非类码' \
    '（LOO k=8 仅 .397）。' + t[j:]

# --- compress 3016+3015 entries (3016 precedes
#     3015 on the line) ---
i = t.index('3016 Ω-P2j')
j = t.index('3017 Ω-P2k')
t = t[:i] + '3016：放大=分布式+深层收敛（单载体复原' \
    ' 8.5%，delta≠效应；lens 中层 29× 深层<1）' \
    '——门控=种子，命运由分布式深层读出定。 ' \
    '3015：K 消费=领先 g7+情景背景（share .390 ' \
    'p=1.0；V 擦除~1e-4=K 特异）；K 擦除→注意力' \
    '重定向；rho .45。 ' + t[j:]

# --- replace 3017 entry, append 3018 ---
i = t.index('3017 Ω-P2k')
j = t.index('\n', i)
t = t[:i] + '3017：吸收=混合——反平行带=中带 ' \
    'L5-25+27（L10 −.496 双反平行）非晚层；' \
    '‖e‖ 增 10×、rel 降 3.7×。 3018 Ω-P2l：' \
    '**抵消主导翻转符号**——credit 2.51 nats>' \
    '总衰减 1.28（share 1.86 CI 1.66-2.36）；' \
    'no-cancel 反事实 E_nc 4.83>R 3.60=无中带则 ' \
    'rel 反增 3.4×；MLP 承载（L10 2dM.e −.469 ' \
    '带 L8-20）；恒等门 0.0。' + t[j:]

# --- update 下一步 line ---
i = t.index('- max=3017')
j = t.index('\n', i)
t = t[:i] + '- max=3018，下一个 3019（A 主选 ' \
    'MLP 抵消带身份——哪些神经元/方向承载 ' \
    'credit，通用 vs logic 特异；B L31 次峰；' \
    'C 情景性检验；D 重定向终点）。' + t[j:]

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy'
        r'\2026-09-17-01-30-05\.workbuddy'
        r'\tmp_mem18.txt', 'w', encoding='utf-8').write(
    'len=%d(was %d) has3018=%s max18=%s no3017line=%s'
    % (len(t2), orig_len, '3018 Ω-P2l' in t2,
       'max=3018' in t2, 'max=3017' not in t2))
print('ok')
