# -*- coding: utf-8 -*-
"""Phase 3231 closeout: five writes (ledger #383 / MEMO / daily / MEMORY /
self-check), idempotent, disk-readback assertions."""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PREV_POST = '3cba5317'
LED_N = 383
TS = '2026-10-13 11:40'
REPORT = []


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# ---------- 1. ledger ----------
LP = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
raw = open(LP, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
already = [e for e in led['measurements'] if e.get('phase') == 3231]
if not already:
    assert len(led['measurements']) == 382 and \
        led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3231,
        "name": "g5g28_registry_decision",
        "line": "G",
        "date": "2026-10-13",
        "model": "CPU JSON assembly on sealed artifacts (zero GPU)",
        "verdict": ("f20_refinement_registered|registry_v1.31|gap_v1.9|"
                    "byte_preservation_OK|verify_12of12_PASS"),
        "evidence_level": ("registration arm (no new observation -> no "
                           "E-level change; F20 refinement appended, "
                           "upgrade_log untouched at 11)"),
        "model_scope": "qwen3-4b + qwen3-14b (2/3, inherited); glm4 deferred",
        "prereg_id": "G5-G28",
        "superseded_by": None,
        "detail": (
            "G5-G28 registry decision arm (zero GPU, default (a) per 3230 "
            "prereg; K1 pure registration - no new observation, no gate "
            "re-opened). Freeze 45a130ea BEFORE any assembly (K1-K5 "
            "clarifications in-freeze). S1 integrity: d0_mean/dD(a=0.5,1,2)/"
            "GA_flip(4 seen folds via 3154 masks)/cos(dR,dR_raw) recomputed "
            "bit-exact from sealed 3230 npz d430d29e x2 models; GB monotone "
            "both True; GA flips all 0. S2 registry v1.31 (file ef5189e7 / "
            "content 4b6648c2, from v1.30 c11d13bd): F20 gains append-only "
            "'refinements' key (recipe-generalization second recipe: state-"
            "contrast dR also NOT a behavioral lever, GA_flip 0 x2 models x "
            "all alphas though GB_dir PASS; transfer-orthogonality "
            "cos(dR,unit(W_isa-W_hasa))=+0.0997/+0.0625; decoupling "
            "cos(dR,v_isa)=-0.2463/-0.0547); FTR-24/25/26 scope_limits "
            "appends; 23 other features + failures[:19] + F20 pre-existing "
            "fields + upgrade_log(11) byte-exact. S3 gap v1.9 (file "
            "9fca29bd / content 83dac915, from v1.8 7065ec4e): GAP-5 "
            "mechanism_note appended (three-stage picture second-recipe "
            "confirmation), legacy gaps x4 byte-exact. Independent verify "
            "12/12 PASS (incl. cos(dR,dw) recomputed from sha-pinned W_U "
            "slices: 0.099659/0.062541). Res 54e55c7a. Ledger read-only in "
            "arm (n=382 post 3cba5317 before==after); this entry #383 "
            "appended in closeout. pre-append file sha8 %s." % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LP + '.bak_3231', 'wb').write(raw)
    with open(LP, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('ledger: appended #383 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    assert len(already) == 1, ('duplicate 3231 entries', len(already))
    new_post = led['ledger_sha256_8']
    REPORT.append('ledger: 3231 entry already present, post=%s (idempotent)'
                  % new_post)
# self-check recompute (chaining convention)
chk = json.load(open(LP, encoding='utf-8'))
assert len(chk['measurements']) == LED_N, (len(chk['measurements']), LED_N)
stored = chk['ledger_sha256_8']
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == stored, ('ledger post recompute', rec, new_post,
                                   stored)
REPORT.append('ledger self-check: n=%d post=%s recompute OK (chain prev=%s)'
              % (LED_N, new_post, PREV_POST))

# ---------- 2. MEMO ----------
MP = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
memo = open(MP, encoding='utf-8').read()
SEC = '''

## Phase 3231: G5-G28 registry 决策臂 [2026-10-13 11:40]

**执行模式**：零 GPU 纯算术装配（默认 (a)，按 3230 预注册；K1 纯入册臂——无新观测、不重开任何门，只执行 3230 verdict 路由的 registry/gap append）。freeze **`45a130ea`**（任何装配前封存，K1–K5 澄清在案）；res **`54e55c7a`**；verdict **`f20_refinement_registered`**（0 矛盾）；账本只读（n=382 post `3cba5317` before==after），本 closeout 追加 #383。

### S1 完整性重算（封存 3230 npz `d430d29e`）
d0_mean / dD(α=0.5,1,2) / GA_flip(4 seen 折，3154 掩码) / cos(dR,dR_raw) 自封存 npz **逐位复现** ×2 模型；GB 单调 ×2 True；GA flip 全 0。装配建立在已封存事实之上，无任何新数字。

### S2 registry v1.30 → v1.31（file `ef5189e7` / content `4b6648c2`，源 v1.30 `c11d13bd`）
- **F20 refinement append**（新增 append-only `refinements` 键，既有字段逐位保全）：配方普适化定论——状态对比方向 dR=mean(h[isa])−mean(h[hasa])（3157 面板，NH−1 槽，|dR|=16.40/20.72 ≈ 行态 RMS 7–9×）在末层精确线性读出图上同样**不是行为杠杆**：GA_flip 0/4 折 ×2 模型 ×全 α（null 带亦 0），尽管 GB_dir PASS（dD 严格单调 +0.3545/+0.3263 @α1，dR_neg 完美镜像）。机制归因=**传递正交** cos(dR, unit(W_isa−W_hasa))=+0.0997/+0.0625，Δm@α2 0.71/0.65 ≪ |d0| 3.97/4.69；解耦诊断 cos(dR,v_isa)=−0.2463/−0.0547 排除配方反平行解释。**两个方向族以不同方式非杠杆（u_R 反方向；dR 同方向但读出近正交）——「读出几何≠行为控制」获第二配方独立确证。**
- **FTR-24/25/26 scope_limits append**：dR 第二配方 FAIL 是行为杠杆地位的进一步反证，不改变几何带 E 级（E2_held_out 等）与既有锚。
- 字节保全断言：23 特征逐位不变 + failures[:19] + upgrade_log(11) 不动 + 顶层字段不变。

### S3 gap v1.8 → v1.9（file `9fca29bd` / content `83dac915`，源 v1.8 `7065ec4e`）
GAP-5 mechanism_note append（三段式三层图景第二配方确证：上游编码承载行为 → 末层精确线性投影 → post-softmax 分布级非线性；行为杠杆须在上游机制中寻找）；legacy gaps ×4 逐位不变。

### 独立复核（verify.json，12/12 PASS）
V1–V3 身份/sha 锚、V4 smoke 隔离、V5–V6 源锚未漂移、V7 npz 重算逐位、**V8 cos(dR,Δw) 自 sha-pinned W_U 切片独立重算 0.099659/0.062541**、V9 registry 字节保全、V10 F20 文本数字 vs 封存值、V11 gap 字节保全、V12 账本头未动。V10 初跑 FAIL 系 verify 探针 en-dash（`7–9`）误查 ASCII 连字符，按纪律只修 verify 不动封存产物，复跑全 PASS。

### 预注册 Phase 3232（候选）
- (a) **GD_amp 扩样臂**（默认）：n=4 → n=10 类（3169 全类崩塌幅度 + 3225 npz u* 逐类投影，纯算术）——裁决 u*_amplitude_carrier 幅度承载假设的类数稳健性；
- (b) GAP-5 上游机制臂设计（FTR-22 残留挂账口径 closed@3175 已清，上游杠杆候选扫描需先立项设计）。
'''
if '## Phase 3231: G5-G28 registry 决策臂' not in memo:
    with open(MP, 'a', encoding='utf-8', newline='') as f:
        f.write(SEC)
    REPORT.append('memo: Phase 3231 section appended')
else:
    REPORT.append('memo: Phase 3231 section already present (idempotent)')
memo_sha8 = fsha(MP)

# ---------- 3. daily ----------
DP = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-13.md')
DSEC = '''
### Phase 3231 — G5-G28 registry 决策臂（2026-10-13 11:40）
freeze `45a130ea`（K1–K5 澄清在先）；零 GPU 纯装配；res `54e55c7a` / verdict `f20_refinement_registered`（0 矛盾）。S1 完整性：d0/dD/GA_flip/cos 自封存 3230 npz 逐位复现 ×2。registry v1.30→v1.31（file `ef5189e7`/content `4b6648c2`）：F20 refinement append（dR 第二配方非杠杆 + 传递正交 cos(dR,Δw)=+0.0997/+0.0625 + 解耦 cos(dR,v_isa)=−0.2463/−0.0547）+ FTR-24/25/26 scope 注记；gap v1.8→v1.9（file `9fca29bd`/content `83dac915`）：GAP-5 mechanism_note append。独立复核 12/12 PASS（V8 cos(dR,Δw) 自 W_U 切片独立重算；V10 初 FAIL 系探针 en-dash bug，只修 verify）。账本 #383 post `%s`（pre-append file `%s`）。3232 预注册：(a) GD_amp 扩样臂（默认，n=4→10 类）；(b) GAP-5 上游机制臂设计。
''' % (new_post, pre_file_sha8)
daily = open(DP, encoding='utf-8').read() if os.path.exists(DP) else ''
if '### Phase 3231' not in daily:
    with open(DP, 'a', encoding='utf-8', newline='') as f:
        f.write(DSEC)
    REPORT.append('daily: Phase 3231 note appended')
else:
    REPORT.append('daily: Phase 3231 note already present (idempotent)')
daily_sha8 = fsha(DP)

# ---------- 4. MEMORY ----------
MEMP = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
mem = open(MEMP, encoding='utf-8').read()
lines = mem.splitlines()
i_prereg = [i for i, l in enumerate(lines) if l.startswith('- **3231 已预注册**')]
i_close = [i for i, l in enumerate(lines) if l.startswith('- 3230 收官：')]
assert len(i_prereg) == 1 and len(i_close) == 1 and i_close[0] == i_prereg[0] + 1, \
    (i_prereg, i_close)
NEW_LINES = [
    ("- **3232 已预注册**：(a) **GD_amp 扩样臂**（默认）：n=4 → n=10 类"
     "（3169 全类崩塌幅度 + 3225 npz u* 逐类投影，纯算术）——裁决 "
     "u*_amplitude_carrier 幅度承载假设的类数稳健性；(b) GAP-5 上游机制臂"
     "设计（上游杠杆候选扫描先立项）。"),
    ("- 3231 收官：f20_refinement_registered（freeze `45a130ea`，res "
     "`54e55c7a`），ledger #383 post `%s`；registry v1.30→v1.31（`ef5189e7`"
     "/`4b6648c2`）F20 refinement append（dR 第二配方非杠杆：GA_flip 0 ×2 "
     "×全 α、传递正交 cos(dR,Δw)=+0.0997/+0.0625、解耦 cos(dR,v_isa)="
     "−0.2463/−0.0547）+ FTR-24/25/26 scope 注记，upgrade_log(11)/E 级不动；"
     "gap v1.8→v1.9（`9fca29bd`/`83dac915`）GAP-5 note append；独立复核 "
     "12/12 PASS（cos(dR,Δw) 自 W_U 切片独立重算 0.099659/0.062541）。"
     % new_post),
]
lines[i_prereg[0]:i_close[0] + 1] = NEW_LINES
with open(MEMP, 'w', encoding='utf-8', newline='') as f:
    f.write('\n'.join(lines) + '\n')
REPORT.append('memory: 3231 prereg+3230 closeout lines -> 3232 prereg + '
              '3231 closeout (anchor check OK)')
mem_sha8 = fsha(MEMP)

# ---------- 5. self-check readback ----------
memo2 = open(MP, encoding='utf-8').read()
daily2 = open(DP, encoding='utf-8').read()
mem2 = open(MEMP, encoding='utf-8').read()
assert '## Phase 3231: G5-G28 registry 决策臂' in memo2 and \
    'f20_refinement_registered' in memo2
assert '### Phase 3231' in daily2 and new_post in daily2
assert '3231 收官' in mem2 and new_post in mem2 and '3232 已预注册' in mem2
REPORT.append('self-check: MEMO/daily/MEMORY readback OK')
REPORT.append('final: ledger_post=%s memo_sha8=%s daily_sha8=%s '
              'memory_sha8=%s' % (new_post, memo_sha8, daily_sha8, mem_sha8))
with open(os.path.join(ROOT, 'tests', 'gpt5_temp',
                       'phase3231_closeout_out.txt'), 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(REPORT) + '\n')
print('CLOSEOUT OK')
