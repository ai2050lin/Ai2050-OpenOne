# -*- coding: utf-8 -*-
"""p3148 closeout: idempotent five writes
(ledger + MEMO + daily log + MEMORY.md +
final check)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D48 = (ROOT + r'\tests\glm5\result'
       + r'\rdc_query_construction_20260913'
       + r'\phase3148'
       + r'\omega_p146_xcross_'
       + r'headsrc_uncancel_v1sym')
RES_SHA = '29327d0e'
SEAL = '87842206'
MEMO = (ROOT + r'\research\gpt5\docs'
        + r'\AGI_GPT5_MEMO.md')
DAILY = (ROOT + r'\.workbuddy\memory'
         + r'\2026-09-30.md')
MEM = (ROOT + r'\.workbuddy\memory'
       + r'\MEMORY.md')

raw = io.open(D48 + r'\result.json',
              'rb').read()
sha = hashlib.sha256(raw).hexdigest()[:8]
assert sha == RES_SHA, sha
r = json.loads(raw.decode('utf-8'))
assert r['verdict'].startswith('a_3147_ok')
assert 'repro_bit_ok' in r['verdict']
assert 'tail_xcross_located' in r['verdict']
print('CHECK res sha8 %s ok' % sha)

# ---- 1. ledger ------------------------
LP = (ROOT + r'\research\gpt5\atlas'
      + r'\atlas_ledger.json')
led = json.load(io.open(LP,
                        encoding='utf-8'))
has48 = any(m.get('phase') == 3148
            for m in led['measurements'])
if not has48:
    led['measurements'].append({
        'phase': 3148,
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
assert n == 285, n
assert led2['measurements'][-1][
    'phase'] == 3148
print('CHECK ledger n=%d ok' % n)

# ---- 2. MEMO --------------------------
t = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3148:' not in t:
    memo48 = """

## Phase 3148: 符号交叉定位+主源分散+抵消否定+v1混合（T4 第31 Phase）[22:35]

**执行**：phase3148_omega_p146_xcross_headsrc_uncancel_v1sym.py；SMOKE 386s 全链贯通（patch1-3 骨架改造+rev-3148a patch1 修 flip_rows 8 行截断——3147 NFLIP_CAP 语义在修剪中丢失，前 8 行位级一致故 ckpt 复用零重算）→ **正式跑 8088.7s**（2h15m，53 gen trials+32×32 headab forwards，中途 N2 断言修一次 ckpt 续跑）。设计四组：①TAIL25 pos/neg 全剂量曲线 d{1,2,2.5,3,3.5,4}×±补齐交叉点+pos_d1/neg_d4 翻转行 fstep 结构 ②order_ex[:2]={2530,3755} 单独/成对剂量 d{0.5,1,2}+L17 逐头 o_proj 输入侧消融（32 头×32 行）+dvec29/19 顶 50 重叠 ③neg d0.5+L39 投影去除 ŵ131401 vs 随机方向 vs w_dn ④v1 放大 α{−0.25,−0.5} vs 削减 α{+0.25,+0.5} @L38 all。

**锚链**：PART A 3147 sha8=63b9885d/seal=66a7b2d9 全数值断言通过（bit11/part_s/part_x order_ex2/part_n flip_rows+fsteps/part_v micro）；xphase P/A1=1.0（128/128）；dvec19 重捕获 sha8=6a0332a6（**第 6 次跨会话位级**，drift 0.00e+00）；field 自洽双 1.0000；z35 软锚 0.0102；C4 resid 锚 4/4 精确；**bit 锚 18/18 全命中**（3142 三重第 6 次、3146 五锚第 3 次、3147 六锚+tail 五锚第 2 次）+ 软锚 tailpos_d2 0.2578 第 2 次位级 + n_neg_d0.5 0.2422 第 3 次位级 + flip_rows/fsteps 冻结表精确匹配。

### §1 发现1：符号交叉点精确定位 d*∈(3.0,3.5]，时间结构随剂量翻转（×3 强调）
- 全剂量曲线：pos {1: 0.102, 2: 0.258, **2.5: 0.266(峰)**, 3: 0.227, 3.5: 0.164(谷), 4: 0.234} vs neg {1: 0.094, 2: 0.172, 2.5: 0.156, 3: 0.188, 3.5: 0.258, 4: 0.305(单调)}；diff {+0.008, +0.086, **+0.109**, +0.039, **−0.094**, −0.070} → **tail_xcross_located (interval [3.0, 3.5])**：正向优势峰在 d2.5 而非 d2，交叉发生在 3.0-3.5 之间——pos 侧是**倒 U 形**（d2.5 峰→d3.5 谷→d4 回升），neg 侧近单调升——符号结构比"线性交叉"复杂，正向通道存在自身的高剂量塌陷。
- 时间结构**反转预期**（xcross_fstep_poslate）：pos_d1 翻转 13 行 med fstep **4.0**（晚发渐进）vs neg_d4 翻转 39 行 med fstep **0.0**（82% 首步即翻）——**剂量决定时间结构而非符号**：低剂量（两符号）=晚发渐进翻转，高剂量 neg=立即首 token 翻转；neg_d4 翻转率 30% vs pos_d1 10%。3147"neg_d0.5 晚发"不是 neg 特性而是**低剂量特性**；高剂量直接压制首 token 读出。
- 综合：尾部符号效应=（剂量×符号×时间）三元结构，d*∈(3.0,3.5] 是符号平衡点，低剂量晚发/高剂量即时的分界也在同一剂量带。

### §2 发现2：co50ex 主源=坐标族联合而非巨头坐标，写入侧中度分散（×3 强调）
- top-2 坐标单独/成对 @d2：solo1(2530) 0.094 / solo2(3755) 0.094 / pair 0.148 → **h_pair_sub**（resid −0.039）；pair 占全集 co50ex_d2 0.852 的仅 **17.4%**（share_top2_vs_co50ex_d2）——**行为主源不在 top-2 富集坐标**：IDEINT@L17 富集首位的 2530 单独效应只及全集 11%，3147 的 top-10 0.891 集中是**约 10 个中等坐标的联合效应**，非 1-2 个巨头。
- L17 写入头分解（o_proj 输入侧 32 头消融）：top6 头 {23: 0.0239, 21: 0.0193, 22: 0.0181, 27: 0.0161, 3: 0.0139, 13: 0.0112}，frac_top2=0.269 → **head_conc_moderate**——写入侧无主导头，top-2 头仅占正贡献 27%，与读出端"端口类"结构（3109/3139/3140）对称：**写入与读出都是分散端口**。
- 重叠检验：2530 在 dvec19 顶 50 的 rank 10、3755 不在；两坐标均不在 dvec29 顶 50 → **src_top_partial**——主源坐标与 L19 swap 场中度关联、与 L29 场无关，写入身份是 L17-L19 局部现象。
- 方法论：3147"top-2 顶全集 d1 水平"的表述需修正——top-2 @d1 0.430 接近全集 @d1 0.422 是**低剂量区小效应偶然对齐**，@d2 起差距拉开（0.148 vs 0.852）。

### §3 发现3：format 抵消否定——dlogit 抬升是共伴信号非翻转载体（×3 强调）
- u_neg_cancel（去除 ŵ131401 分量）= **0.820**（d=−0.578 反向加重）vs u_neg_rand=0.203（d=+0.039 无害）vs u_neg_wdn=0.992（d=−0.750 全毁）→ **uncancel_insufficient**。
- 三重对照的逻辑链：随机方向去除无害→投影去除操作本身干净（非扰动伪象）；答案轴去除全毁→读出通道确认方向特异；**format 方向去除反而加重破坏**→' preoc' token 方向的分量在正常生成中承担功能（可能承载文体连贯性信号），去掉它模拟了更强的 format 扰动而非抵消。
- flip-row 恢复率 cancel 0.00 vs rand 0.25 加强否定。**3147 §3 的"format 偏移=晚发翻转的充分原因"被因果实验否定**：dlogit(131401) 前两步 +1.87 的抬升与翻转共伴但非充分——翻转的真实载体在别处（候选：多路冗余读出/上游共同驱动/非 131401 单方向的分布式偏移）。3147 结论降级为描述性共伴。
- 关联 3145：pc1 通道的 format token 挤入同样只是描述面——两通道的 format 信号都不是因果充分载体，**format 信号=干扰的指纹而非干扰的手**。

### §4 发现4：v1 方向偏置的混合对称——放大效应为削减的 ~60%（×3 强调）
- V2：amp {−0.25: 0.188, −0.50: 0.383} vs clip {+0.25: 0.297(第3次位级), +0.50: 0.664(第3次位级)} → sym25=0.63 / sym50=0.58 → **v1_sym_mixed**。
- 两侧均单调且非对称：放大破坏 0.188/0.383 vs 削减 0.297/0.664——削减更伤→**v1 正方向（PC1 主轴）承载略多的行为关键信号**（方向偏置），但负方向放大也实质破坏（0.188>gate）→非纯方向特异。结合 3146/3147 连续依赖曲线：**承重轴=带方向偏置的连续幅度读出**——每步生成沿 v1 的读出对幅度连续敏感、对方向轻度不对称（正分量略更关键），与 3145"dv29 自身 −67 主导+v1 抵消 +7.9"的符号结构一致。

### §5 综合 + 3149 预注册
3148 四问四答：(1) 符号交叉定位 d*∈(3.0,3.5]，时间结构随剂量翻转（低剂量晚发/高剂量即时），pos 侧倒 U；(2) 主源=坐标族联合（top-2 仅 17%），写入侧 head_conc_moderate 分散、src_top_partial；(3) format 抵消否定（cancel 反向加重/rand 无害/wdn 全毁）——3147 §3 降级为描述性；(4) v1_sym_mixed（sym 0.58-0.63）=方向偏置连续读出。

3149（Ω-P147）预注册：
1. 翻转载体定位：neg_d0.5 dlogit 分解 per-format-token——对 8 翻转行做 logit 差（注入−base）按 top-20 投影 token 分解（前 2 步），找除 131401 外的共同抬升 token 集；对照 pos_d1 翻转行同分解——晚发/即时的载体差异；
2. pos_d1 晚发机制：13 翻转行 fstep 前缀逐步 capture @L38/39（3147 N3 方法）——w_dn 轨迹 vs dlogit 轨迹 vs ||dh||，检验 pos 晚发=读出竞争（wdn 衰减）还是 format（dlogit 增长）；
3. 坐标数-剂量互换：co50ex top-5/top-10 子集 @d{0.5,1,2}（3147 只测 @d4）——检验"更多坐标×更低剂量"是否等效于"少坐标×高剂量"（阈值结构的广度-强度互换）；
4. v1 微放大补齐：α{−0.05,−0.10,−0.15} @L38 all——与 3147 微削减 {0.070,0.078,0.133} 并排成对称曲线，量化方向偏置的剂量依赖（偏置恒定 or 随幅度增长）。

关键数字：bit 18/18；diff {+0.008,+0.086,+0.109,+0.039,−0.094,−0.070} cross [3.0,3.5]；pos_d1 med_fs 4.0(13 行)/neg_d4 med_fs 0.0(39 行,82% 即时)；solo {0.094,0.094}/pair 0.148 sub+share 0.174；head top {23:0.024,21:0.019} frac 0.269；dv19 rank2530=10；cancel 0.820/rand 0.203/wdn 0.992 insufficient；amp {0.188,0.383}/clip {0.297,0.664} sym 0.63/0.58 mixed。

锚：result sha8=29327d0e，seal sha8=87842206，dvec19 sha8=6a0332a6（第 6 次），ledger n=285。
"""
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(memo48)
t2 = io.open(MEMO, encoding='utf-8').read()
assert '## Phase 3148:' in t2
assert '3149（Ω-P147）预注册' in t2
print('CHECK memo ok')

# ---- 3. daily log ---------------------
tl = io.open(DAILY, encoding='utf-8').read()
if 'P146）闭环' not in tl:
    line = ("- 3148（Ω-P146）闭环：SMOKE 386s"
            "（3 patch+rev-3148a flip cap）"
            "→ 正式跑 8088.7s（中途修 1 次"
            "ckpt 续跑）；bit 18/18（3142 六"
            "次/3146 三次/3147+tail 二次）"
            "+dvec19 第6次位级；四发现："
            "符号交叉定位 d*∈(3.0,3.5]（pos"
            " 倒 U 峰 d2.5/neg 单调，时间结"
            "构随剂量翻转：pos_d1 晚发 med4"
            "/neg_d4 即时 82%）；主源=坐标"
            "族联合（top-2 仅 17.4%，写入头"
            "中度分散 frac 0.269，src_top_"
            "partial）；format 抵消否定"
            "（cancel 0.820 反向/rand 无害/"
            "wdn 全毁——3147 §3 降级描述"
            "性）；v1_sym_mixed（sym 0.63/"
            "0.58 方向偏置连续读出）；"
            "ledger n=285（res 29327d0e/"
            "seal 87842206）；3149 预注册："
            "载体定位/pos 晚发机制/坐标数-剂"
            "量互换/v1 微放大。\\n")
    with io.open(DAILY, 'a',
                 encoding='utf-8') as f:
        f.write(line)
tl2 = io.open(DAILY,
              encoding='utf-8').read()
assert 'P146）闭环' in tl2
print('CHECK daily ok')

# ---- 4. MEMORY.md line replace --------
tm = io.open(MEM, encoding='utf-8').read()
lines = tm.split('\n')
idx = None
for i, l in enumerate(lines):
    if l.startswith('- max=3147，'):
        idx = i
        break
if idx is not None:
    new_line = ("- max=3148，下一 3149："
                "①翻转载体定位（neg_d0.5 8 翻转"
                "行 logit 差 per-token 分解 top-20"
                " 前 2 步+pos_d1 对照——131401 之"
                "外的共同抬升集）②pos_d1 晚发机制"
                "（13 行 fstep 前缀 capture @L38/39"
                " w_dn vs dlogit vs ||dh||——读出"
                "竞争 vs format）③坐标数-剂量互换"
                "（co50ex top-5/10 @d{0.5,1,2}"
                "——广度-强度互换检验）④v1 微放大"
                "（α{−0.05,−0.10,−0.15} 与削减并"
                "排——方向偏置剂量依赖）。**符号交叉"
                "精确定位 d*∈(3.0,3.5]（tail_xcross_"
                "located，diff {+0.008,+0.086,"
                "+0.109,+0.039,−0.094,−0.070}，"
                "pos 倒 U 峰 d2.5 0.266 谷 d3.5 "
                "0.164/neg 单调 d4 0.305）；时间"
                "结构随剂量翻转（xcross_fstep_"
                "poslate：pos_d1 翻转晚发 med_fs "
                "4.0 n=13 vs neg_d4 即时 med_fs "
                "0.0 n=39 82% 首步——低剂量=渐进"
                "晚发/高剂量=即时压制，非符号特"
                "性）；主源=坐标族联合（h_pair_sub "
                "solo {0.094,0.094}/pair 0.148，"
                "share_top2 仅 17.4%——3147 top-2"
                " 表述修正为低剂量偶然对齐；写入"
                "头 head_conc_moderate frac 0.269 "
                "top 头 {23,21,22}，src_top_"
                "partial dv19 rank2530=10）；"
                "format 抵消否定（uncancel_"
                "insufficient：cancel 0.820 反向"
                "加重 d−0.578/rand 0.203 无害/"
                "wdn 0.992 全毁——3147 §3 降级"
                "描述性共伴，format 信号=指纹非"
                "手）；v1_sym_mixed（amp "
                "{0.188,0.383} vs clip "
                "{0.297,0.664}，sym 0.63/0.58"
                "——方向偏置连续幅度读出）；"
                "bit 18/18+dvec19 第 6 次位级"
                "+ledger n=285（res 29327d0e/"
                "seal 87842206）。**")
    lines[idx] = new_line
    tmp = MEM + '.tmp'
    with io.open(tmp, 'w',
                 encoding='utf-8') as f:
        f.write('\n'.join(lines))
    os.replace(tmp, MEM)
tm2 = io.open(MEM,
              encoding='utf-8').read()
assert '- max=3148，' in tm2
assert 'tail_xcross_located' in tm2
assert 'uncancel_insufficient' in tm2
assert 'v1_sym_mixed' in tm2
print('CHECK memory ok')
print('CLOSEOUT 3148 DONE (5 writes)')
