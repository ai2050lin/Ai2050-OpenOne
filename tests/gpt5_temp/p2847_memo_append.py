import os

memo = 'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'
sec = open('D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/p2847_memo_section.md',
           encoding='utf-8').read()
with open(memo, 'a', encoding='utf-8') as f:
    f.write(sec)

log = ('C:/Users/Admin/WorkBuddy/2026-09-17-01-30-05/.workbuddy/'
       'memory/2026-09-17.md')
entry = '''

## Phase 2847（2026-09-18 04:35）
- 隐形冠军解剖+双角色判决（Gen1 一次成功 119.4s，80 词）：F1=true/F2=true → dual_roles_confirmed。L13 h30=上游形成器：cdir 直写≈0（-0.0028）但总写出量 926 倍于 cdir 分量、自绑定门 0.527 vs 对照 0.057（9.3x）、钳制后 cdir 变化峰在 L34（深层重组）；晚簇=读出放大器（直写 0.26-0.48、损伤仅 3-4%）。
- F3=false 负结果新机制：早簇亚可加 0.65（joint5 18.4% < Σ单 28.2%）vs 门池 0.96 / 晚簇 0.93——形成器重叠协作，放大器独立可加。三区可加性谱系钉死。
- 横向抑制线索：L13 h30 钳制后 4/10 竞争方向对齐率上升（负 drop）——形成器或以压制竞争类别方式工作；dir4 -12.23 为小分母病态如实登记。
- 臂 C：2846 残差报告病因为代码级（每层只打印末词残差 clamp_resid[-NH:]），全局 0.5708 为真实逐条 max；本跑冠军头饱和率 0、max_b 8.2、残差 0.0096——饱和不在冠军集。
- SHA256：script 67a3e2bf / exec 41c0ada2 / result 1406fa8e / npz d7326fea。
- 接续 2848：横向抑制定量检验（钳制后竞争方向升起的因果验证）+ 形成器→放大器接口追踪（L13→L22 状态传递）。
'''
with open(log, 'a', encoding='utf-8') as f:
    f.write(entry)

n = len(open(memo, encoding='utf-8').readlines())
ok = 'Phase 2847' in open(memo, encoding='utf-8').read()
ok2 = 'Phase 2847' in open(log, encoding='utf-8').read()
print('memo lines %d, 2847 in memo %s, in log %s' % (n, ok, ok2))
