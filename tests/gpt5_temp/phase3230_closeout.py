# -*- coding: utf-8 -*-
"""Phase 3230 closeout: five writes (ledger #382 / MEMO / daily / MEMORY /
self-check), idempotent, disk-readback assertions."""
import hashlib
import json
import os
import sys

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PREV_POST = 'b21db7e3'
LED_N = 382
REPORT = []


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# ---------- 1. ledger ----------
LP = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
raw = open(LP, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
already = [e for e in led['measurements'] if e.get('phase') == 3230]
if not already:
    assert len(led['measurements']) == 381 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3230,
        "name": "g5g27_relcontrast_efficacy",
        "line": "G",
        "date": "2026-10-11",
        "model": "CPU matrix math on sealed caches (zero GPU)",
        "verdict": "dR_lever_fail|GA_flip_FAIL_0of4x2_null_band_0|GB_dir_PASS_x2_dD_monotone_mirror|recipes_decoupled_cos_dR_visA_-0.2463_-0.0547|transfer_orthogonal_cos_dR_dW_+0.0997_+0.0625",
        "evidence_level": "efficacy_test (gate fail -> no E-level change; F20 refinement routed to 3231)",
        "model_scope": "qwen3-4b + qwen3-14b (2/3); glm4 deferred",
        "prereg_id": "G5-G27",
        "superseded_by": None,
        "detail": ("G5-G27 relation-contrast (state) direction efficacy arm "
                   "(zero GPU, default (a) per 3229 prereg). dR = mean(h[isa "
                   "64 rows]) - mean(h[hasa 64 rows]) over the sealed 3157 "
                   "cache at slot NH-1 (post-norm pre-logit, C9 space "
                   "coherence; K1: prereg wording '3157 KOUT 缓存均值' read "
                   "at NH-1, KOUT-slot version sealed as dR_raw, cos(dR,"
                   "dR_raw)=0.8614/0.7081). |dR|=16.40/20.72 (~7-9x the row "
                   "RMS 2.16/2.19). Protocol verbatim from sealed 3228 "
                   "(freeze c51bfea5 / res 99693298): direct linear map "
                   "l=W_U@(h+dv), dv=unit(dR)*alpha*rms(h_row), alphas "
                   "{0.5,1,2}, nulls 10 pinned seeds + orth. Freeze "
                   "1d02dd97 BEFORE any dR number; gates frozen in-freeze. "
                   "Results: GB_dir PASS x2 (seen dD strictly monotone "
                   "+0.1772/+0.3545/+0.7089 (4b) and +0.1632/+0.3263/+0.6526 "
                   "(14b); dR_neg perfect mirror = linear-map sanity) - dR "
                   "pushes the margin in ITS OWN direction (qualitative "
                   "contrast with the anti-directional T2 u_isa); GA_flip "
                   "FAIL x2 (flip 0.0000 on 4/4 seen folds x 2 models at "
                   "all alphas, null band also 0; mechanical attribution: "
                   "cos(dR, unit(W_isa-W_hasa))=+0.0997/+0.0625 ~ "
                   "orthogonal, transfer capped, delta-m@a2 ~0.71/0.65 << "
                   "|d0| 3.97/4.69). Diagnostics: cos(dR,v_isa)=-0.2463/"
                   "-0.0547, cos(dR,v_hasa)=+0.0606/+0.1067 -> |cos|<0.5 "
                   "recipes DECOUPLED per the prereg reading - the 3228 "
                   "anti-direction finding is NOT explained by ctx-contrast "
                   "anti-parallelism; both direction families fail as "
                   "behavioral levers for different reasons. Device anchors: "
                   "3157 tc + 3183 drift 0 x2, C6v3 marg 0.9963/0.9941 "
                   "(centered-cos 1.0000), RMS scale 2.16/11.98 + 2.19/25.31. "
                   "Res 1b63b063 / npz d430d29e. Ledger read-only in arm "
                   "(n=381 post b21db7e3 before==after); registry changes "
                   "deferred to 3231. pre-append file sha8 %s." % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LP + '.bak_3230', 'wb').write(raw)
    with open(LP, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('ledger: appended #382 post=%s (pre_file=%s)' % (new_post, pre_file_sha8))
else:
    assert len(already) == 1, ('duplicate 3230 entries', len(already))
    new_post = led['ledger_sha256_8']
    REPORT.append('ledger: 3230 entry already present, post=%s (idempotent)' % new_post)
# self-check recompute (chaining convention)
chk = json.load(open(LP, encoding='utf-8'))
assert len(chk['measurements']) == LED_N, (len(chk['measurements']), LED_N)
stored = chk['ledger_sha256_8']
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == stored, ('ledger post recompute', rec, new_post, stored)
REPORT.append('ledger self-check: n=%d post=%s recompute OK (chain prev=%s)'
              % (LED_N, new_post, PREV_POST))

# ---------- 2. MEMO ----------
MP = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
memo = open(MP, encoding='utf-8').read()
SEC = '''

## Phase 3230 — G5-G27 关系对比方向效力臂（2026-10-11）

**执行模式**：零 GPU CPU 矩阵运算（默认 (a)，按 3229 预注册）；freeze **`1d02dd97`**（任何 dR 数字观测前封存；GA_flip/GB_dir 门冻结于 freeze 内；K1–K5 五条澄清）；res **`1b63b063`**（None 口径）、npz **`d430d29e`**、verdict **`dR_lever_fail`**（0 矛盾）；账本只读（n=381 post `b21db7e3` before==after）。

### 方向与协议
dR = mean(h[isa 64 行])−mean(h[hasa 64 行])（封存 3157 缓存，槽 **NH−1** post-norm pre-logit——与 3228 C9 注入位同空间；K1：预注册措辞「3157 KOUT 缓存均值」按 C9 空间一致性读作 NH−1，KOUT 槽版本封存为 dR_raw，cos(dR,dR_raw)=0.8614/0.7081）。协议逐字继承封存 3228（freeze c51bfea5/res 99693298）：直接线性图 l=W_U·(h+dv)、dv=unit(dR)·α·rms(h_row)、α{0.5,1,2}、null=10 钉种子+orth。|dR|=16.40/20.72（行态 RMS 2.16/2.19 的 7–9 倍）。

### 门结果（GA_flip / GB_dir 冻结定义）
- **GB_dir PASS ×2**：seen dD(+dR) 严格单调正向 0.5→1→2（4b +0.1772/+0.3545/+0.7089；14b +0.1632/+0.3263/+0.6526），dR_neg 完美镜像（线性图自洽）——**dR 按自身方向推 margin（与 u_isa 的反方向形成质的对照）**。
- **GA_flip FAIL ×2**：flip 0.0000 于 4/4 seen 折 × 2 模型 × 全 α（null 带亦 0）；机械归因：**cos(dR, W_isa−W_hasa)=+0.0997/+0.0625 ≈ 正交**——读出传递效率封顶，α=2 的 Δm≈0.71/0.65 ≪ |d0| 3.97/4.69。
- **taxonomy：dR_lever_fail（GB 过、GA 不过）→ 3231 决策臂路由 F20 refinement，不隐藏。**

### 诊断（report-only，预注册读法）
cos(dR,v_isa)=−0.2463/−0.0547、cos(dR,v_hasa)=+0.0606/+0.1067 → **|cos|<0.5：两配方解耦**——3228 的反方向发现不能归因 ctx-对比配方的反平行；替代图景：**两个方向族以不同方式非杠杆**（u_R 反方向；dR 同方向但近正交读出权重差）。状态空间里关系信号巨大（|dR|≈7–9×行态）而在判断读出上近乎不可见——「读出几何≠行为控制」（F20）获第二配方独立确证，GAP-5 三段式（上游编码承载行为）再加固。

### 设备锚
3157 tc + 3183 drift 0 ×2；C6v3 marg 0.9963/0.9941（centered-cos 1.0000）；RMS 刻度 2.16/11.98、2.19/25.31（C9 再证）；本臂不涉 Qn/u* 管线。

### 预注册 Phase 3231（两支候选）
- (a) **G5-G28 registry 决策臂**（默认）：按 3230 verdict 路由——F20 refinement append（配方普适化定论：dR 亦非杠杆 + 传递正交机制 + 解耦诊断）+ GAP-5 mechanism_note append（三层图景第二配方确证）+ FTR-24/25/26 scope_limits 注记；registry v1.31、gap v1.9；
- (b) **GD_amp 扩样臂**：n=4 → n=10 类（3169 全类崩塌幅度 + 3225 npz u* 逐类投影，纯算术）——裁决幅度承载假设。
'''
if '## Phase 3230 —' not in memo:
    with open(MP, 'a', encoding='utf-8', newline='') as f:
        f.write(SEC)
    REPORT.append('memo: Phase 3230 section appended')
else:
    REPORT.append('memo: Phase 3230 section already present (idempotent)')
memo_sha8 = fsha(MP)

# ---------- 3. daily ----------
DP = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-11.md')
DSEC = '''
### Phase 3230 — G5-G27 关系对比方向效力臂（2026-10-11）
freeze `1d02dd97`（GA_flip/GB_dir 门冻结在先，K1–K5 澄清）；零 GPU；res `1b63b063` / npz `d430d29e` / verdict `dR_lever_fail`（0 矛盾）。GB_dir PASS ×2（seen dD 严格单调正向 +0.3545/+0.3263 @α1，dR_neg 完美镜像）；GA_flip FAIL ×2（flip 全 0；cos(dR, W_isa−W_hasa)=+0.0997/+0.0625 传递正交，Δm@α2 0.71/0.65 ≪ |d0| 3.97/4.69）。诊断：cos(dR,v_isa)=−0.2463/−0.0547 → 两配方解耦——3228 反方向不能归因配方反平行；dR 同方向亦非杠杆 ⇒ F20 第二配方独立确证。设备锚全 drift 0，C6v3 0.9963/0.9941。账本 #382 post `%s`（pre-append file `%s`）。3231 预注册：(a) G5-G28 registry 决策臂（默认，F20 refinement + GAP-5 note，registry v1.31/gap v1.9）；(b) GD_amp 扩样臂。
''' % (new_post, pre_file_sha8)
daily = open(DP, encoding='utf-8').read() if os.path.exists(DP) else ''
if '### Phase 3230' not in daily:
    with open(DP, 'a', encoding='utf-8', newline='') as f:
        f.write(DSEC)
    REPORT.append('daily: Phase 3230 note appended')
else:
    REPORT.append('daily: Phase 3230 note already present (idempotent)')
daily_sha8 = fsha(DP)

# ---------- 4. MEMORY ----------
MEMP = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
mem = open(MEMP, encoding='utf-8').read()
OLD = "- **3230 已预注册**：(a) **G5-G27 关系对比方向效力臂**（默认）：dR = h(isa 面板)−h(hasa 面板)（3157 KOUT 缓存均值，非 ctx-对比配方 v_R）重测 flip 效力——检验反方向/零效力是否 T2 配方特有；(b) GD_amp 扩样臂（n=4→10 类，3169 全类崩塌幅度 + u* 逐类投影，纯算术）。"
NEW = ("- **3231 已预注册**：(a) **G5-G28 registry 决策臂**（默认）：按 3230 verdict 路由——F20 refinement append（配方普适化定论：状态对比方向 dR 亦非行为杠杆 + 传递正交机制 cos(dR,Δw)=0.10/0.06 + 解耦诊断 cos(dR,v_isa)=−0.2463/−0.0547）+ GAP-5 mechanism_note append（三层图景第二配方确证）+ FTR-24/25/26 scope 注记；registry v1.31、gap v1.9；(b) GD_amp 扩样臂（n=4→10 类，纯算术）。\n"
       "- 3230 收官：dR_lever_fail（freeze `1d02dd97`，res `1b63b063`，npz `d430d29e`），ledger #382 post `%s`；GB_dir PASS ×2（dD 严格单调正向 + 完美镜像——dR 按自身方向推 margin，与 u_isa 反方向质的对照）；GA_flip FAIL ×2（flip 全 0，|d0| 3.97/4.69 ≫ Δm@α2 0.71/0.65）；|dR|=16.40/20.72 ≈ 7–9× 行态——关系信号在状态空间巨大而读出近正交，「读出几何≠行为控制」第二配方确证。" % new_post)
cnt = mem.count(OLD)
assert cnt == 1, ('MEMORY anchor count', cnt)
mem = mem.replace(OLD, NEW)
with open(MEMP, 'w', encoding='utf-8', newline='') as f:
    f.write(mem)
REPORT.append('memory: 3230 prereg line replaced (anchor count=1)')
mem_sha8 = fsha(MEMP)

# ---------- 5. self-check readback ----------
memo2 = open(MP, encoding='utf-8').read()
daily2 = open(DP, encoding='utf-8').read()
mem2 = open(MEMP, encoding='utf-8').read()
assert '## Phase 3230 —' in memo2 and 'dR_lever_fail' in memo2
assert '### Phase 3230' in daily2 and new_post in daily2
assert '3230 收官' in mem2 and new_post in mem2 and '3231 已预注册' in mem2
REPORT.append('self-check: MEMO/daily/MEMORY readback OK')
REPORT.append('final: ledger_post=%s memo_sha8=%s daily_sha8=%s memory_sha8=%s'
              % (new_post, memo_sha8, daily_sha8, mem_sha8))
with open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'phase3230_closeout_out.txt'),
          'w', encoding='utf-8') as f:
    f.write('\n'.join(REPORT) + '\n')
print('CLOSEOUT OK')
