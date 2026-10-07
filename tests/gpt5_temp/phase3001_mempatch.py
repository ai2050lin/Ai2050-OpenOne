# -*- coding: utf-8 -*-
import io

MP = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\MEMORY.md')
mm = io.open(MP, encoding='utf-8').read()

a = ('注记 qwen L3 双符号均塌缩=早层部分幅度性。'
     '核心：null 重编码全层分布式涌现，对单点'
     '操作化关闭；头级重要性=关系属性。')
b = ('注记 qwen L3 双符号均塌缩=早层部分幅度性。'
     '3001 Ω-G1：GLM4 鲁棒=词 token 携带'
     '（word_eff 86% vs ctx 23%）+一般性洗消'
     '（随机方向比 xdir 更弱=非选择性）；L19 '
     '注入下一层即杀、L4 幅度存续但 xdir 分量'
     '正交化。核心：null 重编码全层分布式涌现，'
     '对单点操作化关闭；头级重要性=关系属性。')
assert a in mm, 'anchor A miss'
mm = mm.replace(a, b, 1)

a2 = ('- max=3000，下一个 3001（A 主选 Ω-G：'
      'GLM4 鲁棒性来源，词 token swap 判携带 '
      'vs 早期带；B 2989 T3 加密+k 剂量；'
      'C Ω-E 错误吸引子；D qwen L3 幅度性/'
      'L15 方向性分界）。方案 v4：research'
      '\\gpt5\\docs\\plan_v4_micro_macro_merge.md。')
b2 = ('- max=3001，下一个 3002（A 主选 qwen '
      '同款词 token swap 加法分解（跨模型镜像）；'
      'B L4 存续/L20 即杀机制分界；C 2989 T3 '
      '加密+k 剂量；D Ω-E 错误吸引子）。方案 '
      'v4：research\\gpt5\\docs'
      '\\plan_v4_micro_macro_merge.md。')
assert a2 in mm, 'anchor B miss'
mm = mm.replace(a2, b2, 1)

io.open(MP, 'w', encoding='utf-8').write(mm)
mm2 = io.open(MP, encoding='utf-8').read()
assert '3001 Ω-G1' in mm2
assert 'max=3001' in mm2
print('chars=%d' % len(mm2))
