# -*- coding: utf-8 -*-
"""p3146 patch2: MEMORY.md line-level
replace of the max=3145 line with the
3146-completed line (matching the
established single-line format).
Idempotent."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne'
      r'\.workbuddy\memory\MEMORY.md')
t = io.open(FP, encoding='utf-8').read()

if 'max=3146' in t:
    print('MEMORY: 3146 line already '
          'present (idempotent skip)')
else:
    lines = t.split('\n')
    idx = None
    for i, l in enumerate(lines):
        if l.startswith('- max=3145'):
            idx = i
            break
    assert idx is not None, 'no 3145 line'
    new = (u"- max=3146，下一 3147："
           u"①bot15 坐标符号矩阵（TBOT15 ×±sgn ×d{1,2}+bot15 内 top5/bot10 细分——尾信号方向特异性检验）"
           u"②co50ex 超线性解剖（内部 k∈{2,5,10,15} 子集注入+d{0.5,1,2} 低剂量补齐——定位 0.992 近全破坏的坐标主源与阈值结构）"
           u"③neg 晚发翻转机制（neg 翻转行 tf 逐层 capture+w_dn 逐步投影谱——读出竞争 vs format 通道偏移）"
           u"④v1 微 clip 下探（α{0.05,0.10,0.15}+base 生成 v1 幅度逐步轨迹——无害阈值或定案连续依赖）。"
           u"**尾部信号=bot15 坐标主导（tail_locus_bot15，0.359>top10 0.242，IDEINT 富集排序对行为反向）；"
           u"L39 跳升=下游真实动态非 norm 伪象（l39_jump_downstream，raw 1.92 vs norm 1.77）+hist 不敏感（注入-响应层内性质，hist 谱与 base-hist 逐层一致）；"
           u"pcres gap 剂量单调（pcres_gap_mono 0.148/0.539/0.563）+neg 晚发分散翻转（neg_flip_late，med fstep 4，直方图 4/6 集中）；"
           u"v1 无干净 clip 窗口（v1_no_clean_window，α0.25 0.297/α0.5 0.664/deco 0.781 全 dirty 剂量单调）=承重轴不可去；"
           u"剂量矩阵 gbot/co36full/co50ex/tail 全 mono、head mixed（d2 回落 d4 跳升），co50ex d4 0.992 近全破坏超线性。**")
    lines[idx] = new
    io.open(FP, 'w',
            encoding='utf-8').write(
        '\n'.join(lines))

chk = io.open(FP,
              encoding='utf-8').read()
assert 'max=3146' in chk
assert 'max=3145' not in chk
assert 'tail_locus_bot15' in chk
assert 'v1_no_clean_window' in chk
print('MEMORY LINE REPLACED OK')
