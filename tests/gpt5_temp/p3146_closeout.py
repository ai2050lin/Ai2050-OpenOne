# -*- coding: utf-8 -*-
"""Phase 3146 closeout: idempotent five
writes. ledger 282 -> 283 + MEMO (with
3147 prereg) + daily log + MEMORY.md.
All writes append/replace guarded."""
import hashlib
import io
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D46 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3146'
       r'\omega_p144_taillocus_histfield_'
       r'pcresdose_cleanclip')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
DLOG = (ROOT + r'\.workbuddy\memory'
        r'\2026-09-30.md')
WMEM = (ROOT + r'\.workbuddy\memory'
        r'\MEMORY.md')

res = json.load(io.open(
    D46 + r'\result.json',
    encoding='utf-8'))
assert res['smoke'] is False
raw = io.open(
    D46 + r'\result.json', 'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]
assert sha8 == 'e5ed3181', sha8
V = res['verdict']
assert 'repro_bit_14' in V
assert 'repro_bit_ok' in V
assert 'tail_locus_bot15' in V
assert 'l39_jump_downstream' in V
assert 'pcres_gap_mono' in V
assert 'v1_no_clean_window' in V
assert 'xphase_ok' in V
print('res verified sha8=%s seal=%s'
      % (sha8, res['seal_sha8']))

# ---------------- 1. ledger ------------
led = json.load(io.open(
    LEDGER, encoding='utf-8'))
ent = None
for m in led['measurements']:
    if m.get('phase') == 3146:
        ent = m
        break
if ent is None:
    ent = {
        'phase': 3146,
        'name': res['name'],
        'date': '2026-09-30',
        'res_sha8': sha8,
        'seal_sha8': res['seal_sha8'],
        'verdict': V,
        'tags': V.split('|'),
        'summary': (
            'Omega-P144: 14/14 bit replay; '
            'tail locus = bot15 (0.359 > '
            'ttop10 0.242, enrichment-'
            'rank reversed); dose matrix '
            'gbot/co36full/co50ex/tail '
            'mono, head mixed (d2 dip); '
            'co50ex d4 0.992 near-total '
            'break; L39 jump = downstream '
            '(norm jump 1.77 vs raw 1.92, '
            'hist jump 1.92 kept); pcres '
            'gap mono 0.148/0.539/0.563 + '
            'neg flip late (med fstep 4); '
            'v1 no clean window (0.297/'
            '0.664/0.781 all dirty) - '
            'carrier axis non-removable'),
        'runtime_s': res['runtime_s']}
    led['measurements'].append(ent)
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f,
                  ensure_ascii=False,
                  indent=1)
    print('ledger: appended 3146 (n=%d)'
          % len(led['measurements']))
else:
    assert ent['res_sha8'] == sha8
    print('ledger: 3146 already present '
          '(idempotent skip)')
assert len(led['measurements']) == 283

# ---------------- 2. MEMO --------------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3146:' in memo:
    print('MEMO: 3146 section already '
          'present (idempotent skip)')
else:
    sec = u"""

## Phase 3146: 尾坐标bot15定位+L39下游跳升+pcres剂量单调+v1不可去（T4 第29 Phase）[08:20]

**执行**：phase3146_omega_p144_taillocus_histfield_pcresdose_cleanclip.py；SMOKE 679s 全链贯通（rev-3146a patch1 修 E_TRIALS 命名 '%g'%1.0='1' 键名 bug）→ **正式跑一次通过 18791s**（5h13m，21 D trials+6 E trials+4 captures+5 B4/T4 captures）。设计四组：①尾坐标细分（TTOP10=order_e[25:35] vs TBOT15=order_e[35:] sgn−1 @d4 A1/L17）+ 剂量矩阵 head/headpos/gbot/co36full/co50ex/tail × d{1,2,4} ②pc1 累积解剖（16 diverged 行 tf captures L20-39：base-hist vs pc1-hist 注入逐层谱 + final-norm 分解）③pcres @L38 ±×d{0.25,0.5,1.0}（gap 演化 + neg fstep 结构）④v1 干净窗口（base+{α0.25,α0.5,decode-only}@38 → joint/dv29×best）。

**锚链**：PART A 3145 sha8=9d9cc6d1/seal=46c0187b 全数值断言通过（bit5/amp/clip_res/spec/headtail/residcausal/field/resid）；dvec19 重捕获 sha8=6a0332a6（drift 0.00e+00，**第 4 次跨会话位级**）；field 自洽双 1.0000；xphase P/A1=1.0（128/128）；z35 软锚 0.0102；C4 resid 锚 4/4 精确命中；**bit 锚 14/14 全命中**（3142 三重第 4 次、3137 双锚第 3 次、3144 三锚第 2 次、3145 四锚第 2 次）。

### §1 发现1：尾部方向信号定位在富集分更低的 bottom15 坐标（×3 强调）
- @d4 A1/L17：**tbot15 0.3594 > ttop10 0.2422**（|Δ|=0.117>0.03）→ **tail_locus_bot15**：尾部效应主源是 IDEINT 富集排名 35-50 位的坐标（bot15 单独注入甚至超过 TAIL25 整体 0.3047——top10 坐标在尾部组内是稀释项/抑制项）。**IDEINT 富集排序对行为效应的预测在尾部反向**——高富集≠高行为贡献，与 3144 rank 0.77-0.91 chance 呼应。
- 剂量矩阵：gbot mono（0.117→0.156→0.227）、co36full mono（0.141→0.227→0.656）、co50ex mono 超线性（0.422→0.852→**0.992 近全破坏**）、tail mono（0.094→0.172→0.305）；head/headpos **mixed**（d2 回落 0.219→0.203 / 0.141→0.125 后 d4 跳升 0.688/0.891）——头部低剂量区有饱和/回落结构，d4 起才释放；head d4/d1 ratio 3.14 ≈ tail 3.25（尾部弱信号靠剂量补偿到同等增长率）。

### §2 发现2：L39 跳升是下游真实动态，非 final-norm 伪象（×3 强调）
- T5 final-norm 分解：raw dlogit(131401) L38 1.474→L39 2.832（jump **1.921**）；norm 后 0.332→0.589（jump 1.774）→ **l39_jump_downstream**（1.774 ∈ [1.921×0.85, 1.921×1.15]）：RMSNorm 压缩绝对幅度 4.4×但保持跳升比——L39 的 1.9× 跳升不是 norm 放缩伪象，是最后一层下游计算的真实放大。
- T4 hist 重放全层谱：pc1-hist 条件 dlogit 谱与 base-hist **逐层一致**（L29 1.201→L39 2.832 完全相同），onset L29 不变，jump_hist 1.921——**干扰 token 抬升对前缀历史不敏感**（历史重放不改变注入响应谱），pc1 累积效应是注入-响应的层内性质而非跨步历史积累；结合 3145 instant 行为无效应：累积发生在行为读出端（多步 logit 偏置相加），不在单步表征端。

### §3 发现3：pcres 方向区分度剂量单调；neg 翻转晚发分散（×3 强调）
- @L38 gap(d)=pos−neg：0.25→0.148、0.5→0.539、1.0→0.562 → **pcres_gap_mono**（0.5→1.0 段 +0.023 勉强单调）：方向区分度主要在 d0.25→0.5 区间建立，1.0 处 pos 接近饱和（0.984）。
- neg 翻转结构：neg_d0.25 翻 19 行、neg_d0.5 翻 31 行，med fstep **4.0** → **neg_flip_late**：负向注入的翻转分散在 decode 中后段（fstep 直方图集中于 4/6 交替），非早期翻转——负向 resid 方向不直接翻转答案 token，而是逐渐积累偏移使中间 format token 偏离后间接破坏。

### §4 发现4：v1 轴不可去——无任何干净干预窗口（×3 强调）
- B4 三窗口全 dirty 且剂量单调：α0.25 **0.297** / α0.50 **0.664** / decode-only **0.781**（CLEAN_GATE 0.05）→ **v1_no_clean_window**：部分幅度去除 25% 就破坏 30% 生成，decode-only（保留 prompt forward）也崩——**L29 WR 主 PC 是逐 decode 步都被读出的承重方向，任何时刻去除都会破坏生成动态**。形态支配假说（pc1 分量主导 blocking）在 v1 维度不可分离检验（与 3145 v1_causal_dirty 合并：承重轴结论三级强化）。

### §5 综合 + 3147 预注册
3146 四问四答：(1) 尾部信号=bot15 坐标主导（富集排序反向），剂量矩阵 4 mono + head mixed；(2) L39 跳升=下游真实动态（norm 排除）+ hist 不敏感（注入-响应层内性质）；(3) pcres gap 剂量单调（0.148→0.539→0.563）+ neg 晚发分散翻转；(4) v1 无干净窗口（0.297/0.664/0.781）——承重轴不可去。

3147（Ω-P145）预注册：
1. bot15 坐标符号矩阵：TBOT15 × ±sgn × d{1,2} + bot15 再细分（bot15 内 top5/bot10 by 富集分）——尾部信号的符号结构与坐标簇定位（尾信号是否也是方向特异的，还是双向对称的破坏效应）；
2. co50ex 超线性解剖：co50ex 内部 k-步进子集注入（k∈{2,5,10,15} 分组）+ co50ex d{0.5,1,2} 补齐低剂量——定位近全破坏（0.992）的坐标主源与超线性响应的阈值结构；
3. neg 晚发翻转机制：neg 翻转行 teacher-forced 逐层 capture（fstep 前缀）+ w_dn 投影逐步谱——检验晚发翻转=读出端逐步竞争（dh.wdn 逐层衰减）还是 format 通道偏移；
4. v1 不可去性下探：α{0.05,0.10,0.15} 微 clip 剂量曲线 + base 生成中 v1 幅度‖h·v̂‖逐 step 轨迹（无干预观测）——找行为无害阈值或定案承重轴连续依赖。

关键数字：bit 14/14；ttop10 0.2422/tbot15 0.3594/tail_d4 0.3047；co50ex 0.422→0.852→0.992；co36full 0.141→0.227→0.656；head 0.219→0.203→0.688 mixed；jump_raw 1.921/jump_norm 1.774/jump_hist 1.921；gap 0.148/0.539/0.563；neg med fstep 4.0（19/31 行）；windows 0.297/0.664/0.781。

锚：result sha8=e5ed3181，seal sha8=ed8c8ede，dvec19 sha8=6a0332a6，ledger n=283。
"""
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(sec)
    print('MEMO: 3146 section appended')

# ---------------- 3. daily log ---------
dl = ''
if os.path.exists(DLOG):
    dl = io.open(DLOG,
                 encoding='utf-8').read()
if 'Phase 3146' in dl:
    print('daily: 3146 already present '
          '(idempotent skip)')
else:
    line = (u"""
## Phase 3146（Ω-P144）闭环 [08:20]
- 正式跑 18791s 一次通过（SMOKE 679s，rev-3146a 修 E_TRIALS 命名 bug）。verdict 全绿：**14/14 bit 锚**（3142×4、3137×3、3144×2、3145×2 跨会话重放）+ tail_locus_bot15 + l39_jump_downstream + pcres_gap_mono + neg_flip_late + v1_no_clean_window。
- 关键：尾部信号=bot15 坐标（0.359>top10 0.242，富集排序反向）；L39 跳升=下游真实（norm jump 1.77 vs raw 1.92）+hist 不敏感；pcres gap 单调 0.148/0.539/0.563+neg 晚发（fstep 4）；v1 三窗口全 dirty（0.297/0.664/0.781）=承重轴不可去。
- closeout 五写：ledger n=283（res sha8=e5ed3181、seal ed8c8ede）+ MEMO（含 3147 预注册：bot15 符号矩阵/co50ex 超线性解剖/neg 晚发机制/v1 微 clip 下探）+ 本日志 + MEMORY.md。
""")
    with io.open(DLOG, 'a',
                 encoding='utf-8') as f:
        f.write(line)
    print('daily: appended')

# ---------------- 4. MEMORY.md ---------
wm = io.open(WMEM,
             encoding='utf-8').read()
NEWLINE = (u"- 3146（T4 第29）：14/14 bit 重放（最深锚链）；尾部信号=bot15 坐标主导（tail_locus_bot15 0.359>top10 0.242，IDEINT 富集排序对行为反向）；L39 跳升=下游真实动态非 norm 伪象（l39_jump_downstream）+hist 不敏感（注入-响应层内性质）；pcres gap 剂量单调（pcres_gap_mono）+neg 晚发分散（neg_flip_late fstep 4）；v1 无干净 clip 窗口（v1_no_clean_window 0.297/0.664/0.781）=承重轴不可去；co50ex d4 0.992 近全破坏超线性。\n")
if '3146（T4 第29）' in wm:
    print('MEMORY: 3146 line already '
          'present (idempotent skip)')
else:
    anchor = u'- 3145（T4 第28）：'
    i = wm.find(anchor)
    assert i >= 0, 'anchor 3145 line'
    wm = (wm[:i] + NEWLINE + wm[i:])
    with io.open(WMEM, 'w',
                 encoding='utf-8') as f:
        f.write(wm)
    print('MEMORY: 3146 line inserted')
print('CLOSEOUT DONE (3146)')
