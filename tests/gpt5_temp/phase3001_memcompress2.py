# -*- coding: utf-8 -*-
import io

MP = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\MEMORY.md')
mm = io.open(MP, encoding='utf-8').read()
pairs = (
    ('- LPF v5.3 机械可解释性；qwen3-4b（D:\\AI2050\\Ai2050-OpenOne\\models\\hf\\qwen3-4b）。',
     '- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）。'),
    ('- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14 connects；hash=去 ledger_sha256_8 后 dumps(sort_keys) sha256 前 8 位。',
     '- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14 connects；hash=去 ledger_sha256_8 后 dumps(sort_keys) sha256 前 8 位。'),
    ('- 剂量干预切片级部分还原；anchor_fail 预初始化；对比写明"vs 哪个条件"。',
     '- 剂量干预切片级部分还原；anchor_fail 预初始化；对比写明 vs 哪个条件。'),
    ('- 高维 per-(l,h) transpose(1,3,0,2)、管线恒等门（2970）；np.where 同形；跨 Phase 常量显式重建；跨产物核对键/路径/精度。',
     '- 高维 per-(l,h) transpose(1,3,0,2)、管线恒等门（2970）；np.where 同形；跨 Phase 常量显式重建；跨产物核对键/精度。'),
    ('- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件后 Read；反引号/管道被篡改（chr(96)）。',
     '- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改（chr(96)）。'),
    ('- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁；replace 未命中→先 Grep 确认锚串。',
     '- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁（本会话两遇）；replace 未命中→先 Grep。'),
    ('- numpy 标量入 json 转 int()/float()；import 遮蔽→别名；大 n 用 lgamma。',
     '- numpy 标量入 json 转 int()/float()；import 遮蔽→别名。'),
)
for a, b in pairs:
    if a in mm:
        mm = mm.replace(a, b, 1)
    else:
        print('MISS', a[:18])
io.open(MP, 'w', encoding='utf-8').write(mm)
mm2 = io.open(MP, encoding='utf-8').read()
assert '3001 Ω-G1' in mm2 and 'max=3001' in mm2
print('chars=%d' % len(mm2))
