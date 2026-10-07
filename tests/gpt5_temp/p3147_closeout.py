# -*- coding: utf-8 -*-
"""p3147 closeout: idempotent five writes
(ledger + MEMO + daily log + MEMORY.md +
final check)."""
import hashlib
import io
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D47 = (ROOT + r'\tests\glm5\result'
       + r'\rdc_query_construction_20260913'
       + r'\phase3147'
       + r'\omega_p145_tbsym_co50exk_'
       + r'negmech_v1micro')
RES_SHA = '63b9885d'
SEAL = '66a7b2d9'
RUNTIME = 13802.0
MEMO = (ROOT + r'\research\gpt5\docs'
        + r'\AGI_GPT5_MEMO.md')
DAILY = (ROOT + r'\.workbuddy\memory'
         + r'\2026-09-30.md')
MEM = (ROOT + r'\.workbuddy\memory'
       + r'\MEMORY.md')

raw = io.open(D47 + r'\result.json',
              'rb').read()
sha = hashlib.sha256(raw).hexdigest()[:8]
assert sha == RES_SHA, sha
r = json.loads(raw.decode('utf-8'))
assert r['verdict'].startswith('a_3146_ok')
assert 'repro_bit_ok' in r['verdict']
print('CHECK res sha8 %s ok' % sha)

# ---- 1. ledger ------------------------
LP = (ROOT + r'\research\gpt5\atlas'
      + r'\atlas_ledger.json')
led = json.load(io.open(LP,
                        encoding='utf-8'))
has47 = any(m.get('phase') == 3147
            for m in led['measurements'])
if not has47:
    led['measurements'].append({
        'phase': 3147,
        'name': r['name'],
        'seal_sha8': r['seal_sha8'],
        'result_sha8': sha,
        'verdict': r['verdict'],
        'created': r['created'],
        'runtime_s': r['runtime_s']})
    tmp = LP + '.tmp'
    with io.open(tmp, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f,
                  ensure_ascii=False,
                  indent=1)
    os.replace(tmp, LP)
led2 = json.load(io.open(LP,
                         encoding='utf-8'))
n = len(led2['measurements'])
assert n == 284, n
assert led2['measurements'][-1][
    'phase'] == 3147
print('CHECK ledger n=%d ok' % n)

# ---- 2. MEMO --------------------------
t = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3147:' not in t:
    memo47 = """

## Phase 3147: 尾符号交叉+co50ex阈值集中+format偏移+v1连续（T4 第30 Phase）[21:35]

**执行**：phase3147_omega_p145_tbsym_co50exk_negmech_v1micro.py；SMOKE 390s 全链贯通（rev-3147a patch1 修死代码/N1 ckpt/补 pcres 重放/双 pad；rev-3147b patch2 co50ex list→ndarray；rev-3147c patch3 co50ex=50 坐标非 25——**co50∩co36=0 是 3135 结论**，断言修正并加不相交校验）→ **正式跑一次通过 13802s**（3h50m，30 gen trials+9 capture stages）。设计四组：①尾符号矩阵（TAIL25 pos d{1,2}+TBOT15 ±×d{1,2}+tb5/tb10 细分 @d4）②co50ex 解剖（k∈{2,5,10,15} top/bot 子集 @d4+d0.5 低剂量补齐）③neg 晚发机制（neg_d0.5 重放+8 翻转行×fstep 前缀逐 step capture @L38/39）④v1 微 clip（α{0.05,0.10,0.15}+base 生成 |h·v̂| 逐步轨迹）。

**锚链**：PART A 3146 sha8=e5ed3181/seal=ed8c8ede 全数值断言通过（bit14/dvals21/shapes/windows/pcres/hist/field/resid）；xphase P/A1=1.0（128/128）；dvec19 重捕获 sha8=6a0332a6（drift 0.00e+00，**第 5 次跨会话位级**）；field 自洽双 1.0000；z35 软锚 0.0102；C4 resid 锚 4/4 精确命中（share38 0.6405/wdn_p39 2.4955/ra_v1 0.2287/pnorm38 55.263）；**bit 锚 11/11 全命中**（3142 三重第 5 次、3146 双锚+co50ex 三锚第 2 次、3145 inst+pcres 双锚第 3 次）+ 意外软锚 tailpos_d2 0.2578 位级复现 3145。

### §1 发现1：尾部符号剂量交叉——低剂量正向主导、高剂量负向反转（×3 强调）
- TAIL25 符号曲线：pos {d1 0.102, d2 **0.258**, d4 0.234} vs neg {d1 0.094, d2 0.172, d4 **0.305**} → **tail_sign_cross**：d2 处 pos>neg（+0.086）、d4 处 pos<neg（−0.070）——**尾部坐标注入的符号效应随剂量翻转**，非固定方向信号。bot15 内部更极端：@d2 pos 0.359 vs neg 0.094（正向 3.8×）；结合 3146 @d4 neg 0.359 > pos（未测，但 TAIL25 整体已交叉）——低剂量区正向（+）注入对尾部是强破坏、高剂量区负向（−）才主导。
- 机制含义：尾部坐标的行为效应是**符号-剂量耦合的非线性函数**，任何"尾信号=方向 X"的单向表述都不成立；3144"符号对称噪声"→3145"剂量单调+sym_break"→3147"符号交叉"——三级修正链完成，尾部信号的真实结构=**符号平衡点随剂量移动**。
- 细分：tb5 0.156 / tb10 0.211 @d4（**tail_sub_bot10**，bot10>top5——TAIL25 内部富集反向第 2 次确认）；tb5+tb10=0.367 ≈ tb15 0.359（子集联合近似可加）。

### §2 发现2：co50ex=S 形阈值曲线+主源超集中（top-10 顶全集 90%）（×3 强调）
- 剂量曲线 {0.5: **0.141**, 1: 0.422, 2: 0.852, 4: **0.992**} → **co50ex_threshold**：d0.5/d1 比 0.33<0.5（低剂量压低=阈值结构），d1→d2 段斜率最大（+0.43），d2→d4 饱和（+0.14）——**S 形响应**而非简单超线性；3146"d4 0.992 近全破坏"的前史是低剂量慢启动。
- k 曲线 @d4：top {2: **0.430**, 5: 0.508, 10: **0.891**, 15: **0.953**} vs bot {2: 0.367, 5: 0.242, 10: 0.391, 15: 0.438} → **co50ex_locus_top**：**top-2 坐标单独注入=d4 全集 @d1 的水平（0.430≈0.422）；top-10 达 0.891≈全集 @d2 的 0.852；top-15 达 0.953≈全集 0.992 的 96%**——近全破坏的主源超集中在 IDEINT@L17 富集 top 坐标（首位 order_ex[0]=2530）；bot 系列弱且非单调（bot5 0.242 < bot2 0.367）。
- **富集-行为关系分段结构**：co50ex 内部富集排序与行为**同向**（enrich_inverse_absent），而 TAIL25 内部**反向**（bot15>top10，3146）——富集排序不是全局单调预测子，在 co36 尾段失灵、在 co50ex 段有效；"富集反向"结论限定于 TAIL25 内部，不可外推。

### §3 发现3：neg 晚发=format 通道偏移，非读出竞争（×3 强调）
- n_neg_d0.5 重放 chg=0.2421875 **位级复现 3146**（neg_repro_3146）；8 翻转行 fsteps=[4,6,4,4,4,4,6,4]（med 4，与 3146 一致）。
- 44 序列逐 step capture（fstep 前缀±注入 @L38）：**w_dn 投影轨迹全程平稳微小**（|wdn|≤0.024，无衰减无翻转）而 **dlogit(131401) 从 step0 −0.412 爬升至 step2 +1.453**（前 2 步内完成 +1.87 增长，step3-6 维持 0.9-1.4 平台）；||dh39|| ~30-33 平稳 → **neg_late_format_shift**：负向 pcres 注入的晚发翻转机制=实体风格 token（' preoc'）logit 系统性抬升逐步挤占答案读出（format 通道偏移），**而非 w_dn 读出方向的竞争衰减**——3145 发现的"pc1 干扰=format token 抬升"在负向 resid 通道同构成立，format 偏移是 L38 残差注入的普遍出口。

### §4 发现4：v1 承重轴连续依赖定案+幅度全程被读出（×3 强调）
- 微 clip α{0.05: **0.070**, 0.10: 0.078, 0.15: 0.133} 全部 > CLEAN_GATE 0.05 → **v1_micro_continuous**：**不存在无害阈值**——去除 5% v1 分量即破坏 7% 生成；与 3146（α0.25→0.297、α0.5→0.664、decode-only 0.781）合并成连续剂量曲线 0.05/0.10/0.15/0.25/0.5→0.07/0.08/0.13/0.30/0.66——**承重轴连续依赖定案**（五点单调，无可分离窗口）。
- |h·v̂| base 生成逐步轨迹 med 9.5-27.8、ratio 0.831 → **v1_amp_stable**：v1 幅度在 12 个 decode 步全程被持续读出（无递增/递减段）——承重轴的观测面：每步生成动态都经过 L29 主 PC 方向。

### §5 综合 + 3148 预注册
3147 四问四答：(1) 尾部符号=剂量交叉（pos 主导低剂量、neg 主导高剂量），bot15 低剂量正向效应 3.8×；(2) co50ex=S 形阈值+主源超集中（top-2 顶全集 d1、top-10 顶全集 d2、top-15 达 96%），富集-行为分段（co50ex 同向/TAIL25 反向）；(3) neg 晚发=format 通道偏移（dlogit 131401 前两步 +1.87，wdn 平稳）——format 出口普适；(4) v1 无无害阈值（α0.05 即 7% 破坏）+幅度全程读出=连续依赖定案。

3148（Ω-P146）预注册：
1. 符号交叉机制定位：TAIL25 pos/neg 中间剂量 d{2.5,3,3.5} 补齐交叉点 + 交叉区 fstep/first 分布对比（pos 低剂量破坏 vs neg 高剂量破坏的时间结构——format 通道 vs 答案侧的切换）；
2. top-2 主源机制身份：order_ex[:2]={2530,3755} 单独+成对剂量曲线 d{0.5,1,2}+L17 写入头分解（逐头 swap 消融对两坐标贡献）+与 dvec29/dvec19 顶坐标重叠检验——超集中主源的写入侧身份；
3. format 偏移因果充分性：neg 注入+逐步 unembed 抵消（dh−=(dh·ŵ131401)ŵ131401 投影去除）重放——行为恢复则 format 偏移=翻转的充分原因；
4. v1 扰动对称性：正向放大 scale{1.25,1.5}+负向微缩 scale{0.90,0.95} @L29 allstep——行为敏感度对称性（承重轴的方向特异性 vs 纯幅度依赖）。

关键数字：bit 11/11；tail pos {0.102,0.258,0.234}/neg {0.094,0.172,0.305} cross；tb15 pos_d2 0.359/neg_d2 0.094；tb5 0.156/tb10 0.211；co50ex {0.141,0.422,0.852,0.992} threshold；ktop {0.430,0.508,0.891,0.953}/kbot {0.367,0.242,0.391,0.438}；neg 0.2422 repro+fsteps 4/6×8；wdn ≤0.024 vs dlog −0.412→1.453；v1 {0.070,0.078,0.133} continuous+amp 0.831 stable。

锚：result sha8=63b9885d，seal sha8=66a7b2d9，dvec19 sha8=6a0332a6，ledger n=284。
"""
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(memo47)
t2 = io.open(MEMO, encoding='utf-8').read()
assert '## Phase 3147:' in t2
assert '3148（Ω-P146）预注册' in t2
print('CHECK memo ok')

# ---- 3. daily log ---------------------
tl = io.open(DAILY, encoding='utf-8').read()
if 'P145）闭环' not in tl:
    line = ("- 3147（Ω-P145）闭环：SMOKE 390s"
            "（3 patch：死代码/ndarray/"
            "co50ex=50）→ 正式跑 13802s 一次"
            "通过；bit 11/11（3142 五 "
            "次/3146 三次/3145 三次）+"
            "dvec19 第5次位级；四发现："
            "尾部符号剂量交叉（pos 低剂量"
            "主导 0.258/neg 高剂量 0.305）"
            "+co50ex S 形阈值且主源超集中"
            "（top-10 0.891/top-15 0.953，"
            "富集-行为分段：co50ex 同向/"
            "TAIL25 反向）+neg 晚发=format"
            " 偏移（dlog131401 step2 1.45、"
            "wdn 平稳）+v1 连续依赖定案"
            "（α0.05 即 0.070>gate，amp "
            "stable 0.831）；ledger n=284"
            "（res 63b9885d/seal 66a7b2d9）"
            "；3148 预注册：交叉点补齐/"
            "top2 身份/unembed 抵消/v1 对"
            "称性。\\n")
    with io.open(DAILY, 'a',
                 encoding='utf-8') as f:
        f.write(line)
tl2 = io.open(DAILY,
              encoding='utf-8').read()
assert 'P145）闭环' in tl2
print('CHECK daily ok')

# ---- 4. MEMORY.md line replace --------
tm = io.open(MEM, encoding='utf-8').read()
lines = tm.split('\n')
idx = None
for i, l in enumerate(lines):
    if l.startswith('- max=3146，'):
        idx = i
        break
if idx is not None:
    new_line = ("- max=3147，下一 3148："
                "①符号交叉定位（TAIL25 pos/neg"
                " 中间剂量 d{2.5,3,3.5}+交叉区"
                " fstep/first 对比——format vs"
                " 答案侧切换）②top-2 主源身份"
                "（order_ex[:2]={2530,3755} "
                "单独+成对 d{0.5,1,2}+L17 写入"
                "头分解+与 dvec29/19 顶坐标重"
                "叠）③format 偏移因果（neg 注入"
                "+逐步 unembed 抵消重放——行为"
                "恢复=充分原因）④v1 扰动对称性"
                "（放大 scale{1.25,1.5}+微缩 "
                "scale{0.90,0.95}——方向特异性"
                " vs 幅度依赖）。**尾部符号=剂"
                "量交叉（tail_sign_cross，pos "
                "{0.102,0.258,0.234}/neg "
                "{0.094,0.172,0.305}，低剂量正"
                "向 bot15 3.8×=0.359/0.094）；"
                "co50ex=S 形阈值+主源超集中"
                "（co50ex_threshold "
                "{0.141,0.422,0.852,0.992}+"
                "co50ex_locus_top，top-2 "
                "0.430 顶全集 d1/top-10 "
                "0.891/top-15 0.953，富集-行"
                "为分段结构：co50ex 同向/"
                "TAIL25 反向 enrich_inverse_"
                "absent）；neg 晚发=format 通"
                "道偏移（neg_late_format_"
                "shift，dlog131401 −0.412→"
                "1.453 前两步、wdn 平稳 ≤"
                "0.024，neg_repro_3146 位级"
                "0.2422）；v1 连续依赖定案"
                "（v1_micro_continuous，"
                "α{0.05,0.10,0.15}→"
                "{0.070,0.078,0.133} 全>"
                "gate，v1_amp_stable 0.831）"
                "；bit 11/11+tailpos_d2 0.2578"
                " 复现 3145+ledger n=284（res "
                "63b9885d/seal 66a7b2d9）。**")
    lines[idx] = new_line
    tmp = MEM + '.tmp'
    with io.open(tmp, 'w',
                 encoding='utf-8') as f:
        f.write('\n'.join(lines))
    os.replace(tmp, MEM)
tm2 = io.open(MEM,
              encoding='utf-8').read()
assert '- max=3147，' in tm2
assert 'tail_sign_cross' in tm2
assert 'co50ex_threshold' in tm2
assert 'neg_late_format_shift' in tm2
assert 'v1_micro_continuous' in tm2
print('CHECK memory ok')
print('CLOSEOUT 3147 DONE (5 writes)')
