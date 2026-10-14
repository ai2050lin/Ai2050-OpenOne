# -*- coding: utf-8 -*-
"""Phase 3235 closeout: five writes (ledger #387 / MEMO / daily / MEMORY /
self-check), idempotent, disk-readback assertions.
Anchor values read dynamically from sealed result.json + verify.json."""
import hashlib
import json
import os
import datetime

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PREV_POST = 'd0dd6b77'
LED_N = 387
RD = os.path.join(ROOT, 'tests', 'glm5', 'result',
                  'rdc_query_construction_20260913', 'phase3235',
                  'g5g31_gap5_gate_decision')
REPORT = []


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


res = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
ver = json.load(open(os.path.join(RD, 'verify.json'), encoding='utf-8'))
FREEZE = res['freeze_sha8']
RES8 = res['res_sha8']
VERDICT = res['verdict']
NCON = len(res['contradictions'])
VP = ver.get('n_pass', '?')
VF = ver.get('n_fails', '?')
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
assert VERDICT == 'UPSTREAM_ABSENT', VERDICT

# ---------- 1. ledger ----------
LP = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
raw = open(LP, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
already = [e for e in led['measurements'] if e.get('phase') == 3235]
if not already:
    assert len(led['measurements']) == 386 and \
        led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3235,
        "name": "g5g31_gap5_gate_decision",
        "line": "G",
        "date": "2026-10-13",
        "model": "zero-GPU gate arithmetic on 3234 sealed matrices (qwen pair)",
        "verdict": ("UPSTREAM_ABSENT|UL_flip_0_all_L_folds_dirs_x2models|"
                    "UL_dose_nonmono|UL_amp_below|UL_spec_trivial_pass|"
                    "UL_loc_band[1,5]_report_only|contradictions_%d" % NCON),
        "evidence_level": ("gate decision arm (per 3233 prereg taxonomy: "
                           "ABSENT => upstream distributed-carrying "
                           "confirmed; F-ledger/registry + GAP-5 note "
                           "landing deferred to 3236 landing arm -> no "
                           "E-level change yet)"),
        "model_scope": "qwen pair (3234 sealed matrices)",
        "prereg_id": "G5-G31",
        "superseded_by": None,
        "detail": (
            "G5-G31 GAP-5 upstream-lever gate decision arm (default (a) "
            "per 3234 prereg, zero GPU). Gates from 3234 sealed npz "
            "(48034bd9), freeze %s before any gate arithmetic: UL_flip "
            "primary 0/4 folds x2 dirs x2 models (max-over-L delta vs "
            "null max = 0.0000 everywhere; sensitivity pooled-L* variant "
            "same), UL_dose non-monotone (u_isa both models; u_hasa mono "
            "but flip levels ~0), UL_amp best rho 0.2/0.4 < 0.6, UL_spec "
            "trivial pass (no flips to cross), UL_loc band [1,5] "
            "report-only no overlap (peaks at L0). C11 honest record: "
            "arm's redundant rates_j1 recompute had a mode-variable leak "
            "(u_isa iterated with leftover mode='hasa'); numerically "
            "coincided with total flips (all u_isa flips hasa-directed); "
            "gate basis = sealed total-flip rates, 4/4 recompute match, "
            "verdict unaffected. res %s. Independent verify %s/%s PASS "
            "(V5/V7 initial FAILs = probe bugs: V5 substring false-"
            "positive on gate-definition prose, V7 = the C11 leak). "
            "Registry v1.31 UNTOUCHED (ef5189e7), gap v1.10 UNTOUCHED "
            "(378de1cd); F-ledger F21 + GAP-5 mechanism_note landing "
            "preregistered as 3236. pre-append ledger file sha8 %s."
            % (FREEZE, RES8, VP, VF, pre_file_sha8)),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LP + '.bak_3235', 'wb').write(raw)
    with open(LP, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('ledger: appended #387 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    assert len(already) == 1, ('duplicate 3235 entries', len(already))
    new_post = led['ledger_sha256_8']
    REPORT.append('ledger: 3235 entry already present, post=%s (idempotent)'
                  % new_post)
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

## Phase 3235: G5-G31 GAP-5 上游杠杆门评估决策臂 [%s]

**执行模式**：零 GPU 门控算术（默认 (a)，按 3234 预注册；全部输入取自 3234 封存 npz `48034bd9`）；freeze **`%s`**（任何门控算术前）；res **`%s`**，verdict **`%s`**（矛盾 %d 项）；registry v1.31 未动（`ef5189e7`）、gap v1.10 未动（`378de1cd`）；closeout 追加 #387。

### 门评估结果（3233 设计门 verbatim 操作化，双模型）

| 门 | qwen3-4b | qwen3-14b | 结论 |
|---|---|---|---|
| UL_flip（≥0.15 ×≥3/4 折，vs null max） | 0/4 折 ×2 方向（delta 全 0.0000） | 0/4 折 ×2 方向 | **FAIL** |
| UL_dose（α 单调） | u_isa 非单调 | u_isa 非单调 | **FAIL** |
| UL_spec（cross ≤ 1/3 matched） | 平凡 PASS（无 flip 可 cross） | 同 | PASS（无信息） |
| UL_amp（spearman ≥0.6） | best 0.2 | best 0.4 | **FAIL** |
| UL_loc（峰带 ⊆ [1,5]，report-only） | 峰在 L0，不重叠 | 同 | 记录 |

**⇒ verdict `UPSTREAM_ABSENT`（按 3233 taxonomy：全网格 FAIL ⇒ 上游分布式承载确证）**——残差流任意层、任意方向（u_R/u*/dR/ΔW）norm-matched patch 均无法翻转 isa/hasa 判断（粗扫全 L 网格 ×4 折 delta 对 null 全为 0.0000），上游不存在单一方向行为杠杆；GAP-5 第三段（上游承载）定谳为**分布式**，与 F20（末层读出几何非杠杆）、3161/3163（写入端无单点头、全流分布式）构成三重互证。

**C11 诚实记录**：臂内冗余 `rates_j1` 重算存在 mode 变量泄漏（u_isa 迭代误用 UL_flip 循环残留的 mode='hasa'）；数值上与总 flip 率巧合相等（u_isa patch 的全部翻转恰为 hasa 向），门控主值 = 封存总 flip 率（与 UL_flip 语义一致，4/4 独立重算逐位匹配），**verdict 不受影响**。

### 独立复核（verify.json，%s/%s PASS）
身份哈希、3234 npz sha、UL_flip primary+sensitivity 自 J1 重算、UL_dose 总 flip 口径重算、UL_spec matched/cross 重算、UL_amp best-layer+spearman 重算（自 3234 result + 3228 E_cls + 3232 崩塌）、UL_loc 重叠重算、verdict 推导复算、gates 表一致性、registry/gap 未动断言。V5/V7 初跑 FAIL 均为探针口径问题（V5 子串误报 gate 定义文本、V7 即 C11 泄漏的暴露），封存产物未动。

### 预注册 Phase 3236（候选）
- (a) **G5-G31 落账臂**（默认，零 GPU，照 3231 模板）：F21（F20 族扩展：上游亦无杠杆）append → registry v1.32 + GAP-5 mechanism_note 终版（三段式收官：上游分布式承载→末层线性读出→post-softmax 非线性）→ gap v1.11；
- (b) u* 内容归因类级扩展（零 GPU，顺延）。
''' % (NOW, FREEZE, RES8, VERDICT, NCON, VP, VF)
if '## Phase 3235: G5-G31 GAP-5 上游杠杆门评估决策臂' not in memo:
    with open(MP, 'a', encoding='utf-8', newline='') as f:
        f.write(SEC)
    REPORT.append('memo: Phase 3235 section appended')
else:
    REPORT.append('memo: Phase 3235 section already present (idempotent)')
memo_sha8 = fsha(MP)

# ---------- 3. daily ----------
DP = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-13.md')
DSEC = '''
### Phase 3235 — G5-G31 GAP-5 上游杠杆门评估决策臂（%s）
零 GPU 门控算术（3234 封存矩阵）；freeze `%s`、res `%s`、verdict **`UPSTREAM_ABSENT`**（矛盾 %d）。UL_flip 0/4 折 ×2 方向 ×2 模型（vs null 全 0.0000）、UL_dose 非单调、UL_amp 0.2/0.4<0.6、UL_loc 峰在 L0 不落 [1,5] ⇒ 上游分布式承载确证（与 F20/3161/3163 三重互证）。C11：rates_j1 mode 泄漏（数值巧合、门 basis 无恙、verdict 不受影响）。独立复核 %s/%s PASS（V5/V7 初 FAIL=探针口径）。账本 #387 post `%s`（pre file `%s`）。registry v1.31 / gap v1.10 未动。3236 预注册：(a) 落账臂（F21→registry v1.32 + GAP-5 终版 note→gap v1.11，默认）；(b) u* 内容归因类级扩展。
''' % (NOW, FREEZE, RES8, NCON, VP, VF, new_post, pre_file_sha8)
daily = open(DP, encoding='utf-8').read() if os.path.exists(DP) else ''
if '### Phase 3235' not in daily:
    with open(DP, 'a', encoding='utf-8', newline='') as f:
        f.write(DSEC)
    REPORT.append('daily: Phase 3235 note appended')
else:
    REPORT.append('daily: Phase 3235 note already present (idempotent)')
daily_sha8 = fsha(DP)

# ---------- 4. MEMORY ----------
MEMP = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
mem = open(MEMP, encoding='utf-8').read()
if '3236 已预注册' in mem and '3235 收官' in mem and \
        '3235 已预注册' not in mem:
    REPORT.append('memory: already at 3236-prereg/3235-closeout state '
                  '(idempotent skip)')
else:
    lines = mem.splitlines()
    i_prereg = [i for i, l in enumerate(lines)
                if l.startswith('- **3235 已预注册**')]
    i_close = [i for i, l in enumerate(lines)
               if l.startswith('- 3234 收官：')]
    assert len(i_prereg) == 1 and len(i_close) == 1 and \
        i_close[0] == i_prereg[0] + 1, (i_prereg, i_close)
    NEW_LINES = [
        ('- **3236 已预注册**：(a) **G5-G31 落账臂**（默认，零 GPU，照 '
         '3231 模板）：F21（F20 族：上游亦无杠杆）append → registry v1.32 '
         '+ GAP-5 mechanism_note 终版（三段式收官）→ gap v1.11；(b) u* '
         '内容归因类级扩展（顺延）。'),
        ('- 3235 收官：UPSTREAM_ABSENT（freeze `%s`，res `%s`），ledger '
         '#387 post `%s`；零 GPU 门控算术：UL_flip 0/4×2dir×2model（vs '
         'null 全 0）、dose 非单调、amp<0.6、loc 不落带 ⇒ 上游分布式承载'
         '确证（F20/3161/3163 三重互证）；C11 rates_j1 mode 泄漏（数值巧合'
         '、verdict 不受影响）；独立复核 %s/%s PASS；registry v1.31 / gap '
         'v1.10 未动。' % (FREEZE, RES8, new_post, VP, VF)),
    ]
    lines[i_prereg[0]:i_close[0] + 1] = NEW_LINES
    with open(MEMP, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(lines) + '\n')
    REPORT.append('memory: 3235 prereg+3234 closeout lines -> 3236 prereg + '
                  '3235 closeout (anchor check OK)')
mem_sha8 = fsha(MEMP)

# ---------- 5. self-check readback ----------
memo2 = open(MP, encoding='utf-8').read()
daily2 = open(DP, encoding='utf-8').read()
mem2 = open(MEMP, encoding='utf-8').read()
assert '## Phase 3235: G5-G31 GAP-5 上游杠杆门评估决策臂' in memo2 \
    and RES8 in memo2
assert '### Phase 3235' in daily2 and new_post in daily2
assert '3235 收官' in mem2 and new_post in mem2 and '3236 已预注册' in mem2
assert len(mem2) < 3000, ('MEMORY too long', len(mem2))
REPORT.append('self-check: MEMO/daily/MEMORY readback OK (mem len=%d)'
              % len(mem2))
REPORT.append('final: ledger_post=%s memo_sha8=%s daily_sha8=%s '
              'memory_sha8=%s' % (new_post, memo_sha8, daily_sha8, mem_sha8))
with open(os.path.join(ROOT, 'tests', 'gpt5_temp',
                       'phase3235_closeout_out.txt'), 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(REPORT) + '\n')
print('CLOSEOUT OK')
