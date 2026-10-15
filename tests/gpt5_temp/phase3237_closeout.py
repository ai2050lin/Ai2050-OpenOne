# -*- coding: utf-8 -*-
"""Phase 3237 closeout: five writes (ledger #389 / MEMO / daily / MEMORY /
self-check), idempotent, disk-readback assertions."""
import hashlib
import json
import os
import datetime

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PREV_POST = '05a31fc4'
LED_N = 389
BASE = os.path.join(ROOT, 'tests', 'glm5', 'result',
                    'rdc_query_construction_20260913')
RD = os.path.join(BASE, 'phase3237', 'g5g31_registry_landing')
REPORT = []


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


res = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
FREEZE = res['freeze_sha8']
RES8 = res['res_sha8']
VERDICT = res['verdict']
REG_F = res['wrote']['registry_v132']['file_sha8']
GAP_F = res['wrote']['gap_v111']['file_sha8']
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
assert VERDICT == 'f21_upstream_absent_registered', VERDICT

# ---------- 1. ledger ----------
LP = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
raw = open(LP, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
already = [e for e in led['measurements'] if e.get('phase') == 3237]
if not already:
    assert len(led['measurements']) == 388 and \
        led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3237,
        "name": "g5g31_registry_landing",
        "line": "G",
        "date": "2026-10-14",
        "model": "none (zero-GPU registry landing arm)",
        "verdict": ("f21_upstream_absent_registered|registry_v1.32|"
                    "gap_v1.11|F21_upstream_lever_absent|GAP-5_FINAL_"
                    "three_stage|contradictions_0"),
        "evidence_level": ("registration arm (routes the sealed 3235 "
                           "UPSTREAM_ABSENT verdict, gates preregistered "
                           "in 3233 d74b6484; no new observation)"),
        "model_scope": "qwen3-4b + qwen3-14b (sealed 3235 scope)",
        "prereg_id": "G5-G31 (MEMO 3236 tail, deferred 3236 landing arm)",
        "superseded_by": None,
        "detail": (
            "Registry landing (zero GPU): (1) F21 NEW failure entry "
            "appended (F20 family extension: upstream lever absent — "
            "full-layer scan u_isa/u_hasa x2 directions, peak-band alpha "
            "fine scan, null 10+2; UL_flip FAIL L*=0 all folds 0 flips "
            "x2 models, UL_dose FAIL non-monotone, UL_amp FAIL "
            "spearman -0.20/+0.40 << 0.6, UL_spec PASS = no flip rather "
            "than wild flip; together with F20: readout end AND upstream "
            "end have NO single-direction behavioral lever => "
            "distributed carrying UPSTREAM_ABSENT closed); (2) GAP-5 "
            "mechanism_note FINAL append (three-stage closure: upstream "
            "distributed -> last-layer exact linear readout -> "
            "post-softmax nonlinearity) + NEW final_verdict key; (3) "
            "FTR-25/26 scope_limits appends. Arm S1 recomputed "
            "UL_flip/UL_dose/UL_amp from sealed 3235 result, matched "
            "sealed gates x2 models; 3235 res_sha8 recomputed OK. "
            "Independent verify 30/0 PASS (file shas, res/freeze "
            "recomputes, F21 evidence numbers vs sealed 3235, "
            "byte-preservation failures[:20] + features 24 exact + 2 "
            "scope appends + upgrade_log 11 + legacy gaps x4 + GAP-5 "
            "legacy fields, ledger untouched). 3235 contradictions "
            "boundary recorded verbatim in F21 (verify V7 probe mode "
            "leak, verdict unaffected). Registry v1.31 -> v1.32 (file "
            "%s), gap v1.10 -> v1.11 (file %s), upgrade_log untouched "
            "(11), E-levels unchanged. freeze %s (before any assembly), "
            "res %s. pre-append ledger file sha8 %s."
            % (REG_F, GAP_F, FREEZE, RES8, pre_file_sha8)),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LP + '.bak_3237', 'wb').write(raw)
    with open(LP, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('ledger: appended #389 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    assert len(already) == 1, ('duplicate 3237 entries', len(already))
    new_post = led['ledger_sha256_8']
    REPORT.append('ledger: 3237 entry already present, post=%s '
                  '(idempotent)' % new_post)
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

## Phase 3237: G5-G31 registry 落账臂（零 GPU） [%s]

**执行模式**：纯落账臂（预注册自 3236 tail 顺延；路由 3235 sealed verdict **UPSTREAM_ABSENT**，门控 3233 design `d74b6484` 预注册）；freeze **`%s`**（任何装配前）；res **`%s`**，verdict **`%s`**；closeout 追加 #389。零 GPU：仅对 sealed 3235 result 做一致性重算 + JSON 装配。

### 三项 append（append-only，byte-preservation 全过）

1. **F21 新失败条目**（failures 20→21，非 F20 refinement——F20=读出方向族非杠杆，F21=上游杠杆缺席，两者合并才得 UPSTREAM_ABSENT）：全层上游扫描（u_isa/u_hasa 双向 × 全 L 网格 + 峰带 α 细扫 + null 10+2，3234 sealed npz `48034bd9`）——**UL_flip FAIL**（L*={0,0}，全折 flip 0、per_fold_max_L 全 0.0 ×2 模型）、**UL_dose FAIL**（非单调）、**UL_amp FAIL**（spearman_best −0.20(4b)/+0.40(14b) ≪ 0.6）、**UL_spec PASS**（失败是不翻而非乱翻）。读出端（F20 双配方正交）+ 上游端（F21 全层缺席）⇒ 行为承载分布式定谳。3235 contradictions 边界（verify V7 探针 mode 泄漏、verdict 不受影响）逐字记录于 F21 evidence。
2. **GAP-5 mechanism_note 终版 append + final_verdict 新键**：三段式收官定谳——上游分布式承载（F20+F21）→ 末层 pre-softmax 精确线性读出（cos_S 1.0000）→ post-softmax 归一化非线性；legacy status 键 `localized_linear_presoftmax` 保持 byte-exact（append-only 纪律）。机制单位转向「载体子空间 × 幅度调度 × 组合规则」（u* 3232 类级 ROBUST 为首例）。
3. **FTR-25/26 scope_limits append**：u_isa/u_hasa 族两端均无行为杠杆；几何 E 级不变（否定性仅限 behavioral-lever 状态）。

### 完整性与复核

臂内 S1：UL_flip/UL_dose/UL_amp 从 sealed 3235 result 重算并与 sealed gates 匹配 ×2 模型；3235 res_sha8 重算 OK；registry v1.31（file `ef5189e7`）/ gap v1.10（file `378de1cd`）/ npz34 `48034bd9` file sha 断言过。独立复核 **30/0 PASS**（V1 文件 sha、V2/V3 res+freeze 重算、V5 F21 证据数值逐项对 sealed 3235、V6/V7 byte-preservation 全家族、V8 账本只读）。

**产物**：`atlas_registry_v1_32.json`（file `%s`）、`gap_ledger_v1_11.json`（file `%s`）、execution/result/run_log @ `phase3237/g5g31_registry_landing/`。upgrade_log 不动（11）、E 级不动、ledger 本臂只读（#389 于 closeout 追加）。

### 预注册 Phase 3238（候选）
- (a) **阶段 A 普查升级臂**（默认，零 GPU）：26 封存特征内在维度/子空间秩测量 + 复用矩阵族内校准 → hub 排序（复用度/因果承重/跨模型普适三判据）→ registry v1.33；
- (b) u* 内容归因类级扩展（零 GPU，顺延）；
- (c) DAS 式子空间交换干预设计臂（轻 GPU 预备，设计先行）。
''' % (NOW, FREEZE, RES8, VERDICT, REG_F, GAP_F)
if '## Phase 3237: G5-G31 registry 落账臂' not in memo:
    with open(MP, 'a', encoding='utf-8', newline='') as f:
        f.write(SEC)
    REPORT.append('memo: Phase 3237 section appended')
else:
    REPORT.append('memo: Phase 3237 section already present (idempotent)')
memo_sha8 = fsha(MP)

# ---------- 3. daily ----------
DP = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-14.md')
DSEC = '''

### Phase 3237 — G5-G31 registry 落账臂（零 GPU，%s）
路由 sealed 3235 UPSTREAM_ABSENT 入册：F21 新失败条目（上游杠杆缺席：UL_flip/UL_dose/UL_amp 三门 FAIL ×2 模型、UL_spec PASS；与 F20 合并=两端均无单方向杠杆 ⇒ 分布式承载定谳）、GAP-5 mechanism_note 终版 + final_verdict 新键（三段式收官：分布式承载→末层线性→post-softmax 非线性）、FTR-25/26 scope notes。registry v1.31→v1.32（file `%s`）、gap v1.10→v1.11（file `%s`），byte-preservation 全过，upgrade_log 不动（11），E 级不动。freeze `%s`、res `%s`。独立复核 30/0 PASS。账本 #389 post `%s`（pre file `%s`）。预注册 3238：阶段 A 普查升级臂（内在维度 + 复用矩阵校准 + hub 排序）。
''' % (NOW, REG_F, GAP_F, FREEZE, RES8, new_post, pre_file_sha8)
daily = open(DP, encoding='utf-8').read() if os.path.exists(DP) else ''
if '### Phase 3237' not in daily:
    with open(DP, 'a', encoding='utf-8', newline='') as f:
        f.write(DSEC)
    REPORT.append('daily: Phase 3237 note appended (2026-10-14.md)')
else:
    REPORT.append('daily: Phase 3237 note already present (idempotent)')
daily_sha8 = fsha(DP)

# ---------- 4. MEMORY ----------
MEMP = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
mem = open(MEMP, encoding='utf-8').read()
if '3238 已预注册' in mem and '3237 收官' in mem:
    REPORT.append('memory: already at 3238-prereg/3237-closeout state '
                  '(idempotent skip)')
else:
    lines = mem.splitlines()
    i_prereg = [i for i, l in enumerate(lines)
                if l.startswith('- **3237 已预注册**')]
    i_close = [i for i, l in enumerate(lines)
               if l.startswith('- 3236 收官：')]
    assert len(i_prereg) == 1 and len(i_close) == 1 and \
        i_close[0] == i_prereg[0] + 1, (i_prereg, i_close)
    NEW_LINES = [
        ('- **3238 已预注册**：(a) **阶段 A 普查升级臂**（默认，零 GPU）：'
         '26 封存特征内在维度/子空间秩 + 复用矩阵族内校准 → hub 排序'
         '（复用度/因果承重/跨模型普适）→ registry v1.33；(b) u* 内容归因'
         '类级扩展（顺延）；(c) DAS 式子空间交换设计臂（轻 GPU 预备）。'),
        ('- 3237 收官：f21_upstream_absent_registered（freeze `%s`，res '
         '`%s`），ledger #389 post `%s`；F21 入册（上游杠杆缺席，三门 '
         'FAIL ×2 模型）+ GAP-5 终版三段式收官 + final_verdict 新键；'
         'registry v1.32 `%s` / gap v1.11 `%s`；verify 30/0。'
         % (FREEZE, RES8, new_post, REG_F, GAP_F)),
    ]
    lines[i_prereg[0]:i_close[0] + 1] = NEW_LINES
    with open(MEMP, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(lines) + '\n')
    REPORT.append('memory: 3237 prereg+3236 closeout -> 3238 prereg + '
                  '3237 closeout (anchor check OK)')
mem_sha8 = fsha(MEMP)

# ---------- 5. self-check readback ----------
memo2 = open(MP, encoding='utf-8').read()
daily2 = open(DP, encoding='utf-8').read()
mem2 = open(MEMP, encoding='utf-8').read()
assert '## Phase 3237: G5-G31 registry 落账臂' in memo2 and RES8 in memo2
assert '### Phase 3237' in daily2 and new_post in daily2
assert '3237 收官' in mem2 and new_post in mem2 and '3238 已预注册' in mem2
assert len(mem2) < 3000, ('MEMORY too long', len(mem2))
REPORT.append('self-check: MEMO/daily/MEMORY readback OK (mem len=%d)'
              % len(mem2))
REPORT.append('final: ledger_post=%s memo_sha8=%s daily_sha8=%s '
              'memory_sha8=%s' % (new_post, memo_sha8, daily_sha8,
                                  mem_sha8))
with open(os.path.join(ROOT, 'tests', 'gpt5_temp',
                       'phase3237_closeout_out.txt'), 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(REPORT) + '\n')
print('CLOSEOUT OK')
