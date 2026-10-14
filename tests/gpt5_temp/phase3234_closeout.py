# -*- coding: utf-8 -*-
"""Phase 3234 closeout: five writes (ledger #386 / MEMO / daily / MEMORY /
self-check), idempotent, disk-readback assertions.
Anchor values read dynamically from sealed result.json + verify.json."""
import hashlib
import json
import os
import datetime

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PREV_POST = '86f04998'
LED_N = 386
RD = os.path.join(ROOT, 'tests', 'glm5', 'result',
                  'rdc_query_construction_20260913', 'phase3234',
                  'g5g30_gap5_upstream_lever_scan')
REPORT = []


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


res = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
ver = json.load(open(os.path.join(RD, 'verify.json'), encoding='utf-8'))
FREEZE = res['freeze_sha8']
RES8 = res['res_sha8']
NPZ8 = res['sealed_npz']['file_sha8']
VERDICT = res['verdict']
NCON = len(res['contradictions'])
VP = ver.get('n_pass', '?')
VF = ver.get('n_fails', '?')
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
assert VERDICT.startswith('upstream_lever_scan_sealed'), VERDICT

# ---------- 1. ledger ----------
LP = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
raw = open(LP, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
already = [e for e in led['measurements'] if e.get('phase') == 3234]
if not already:
    assert len(led['measurements']) == 385 and \
        led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3234,
        "name": "g5g30_gap5_upstream_lever_scan",
        "line": "G",
        "date": "2026-10-13",
        "model": "qwen3-4b (bf16 bitwise path) + qwen3-14b (nf4, recorded)",
        "verdict": ("UPSTREAM_LEVER_SCAN_SEALED|coarse_L0_35_x2dirs_"
                    "fine_band_x3alpha|nulls_10rand_2orth|UL_amp_u*_"
                    "channel|nf4_bridge_4b|gate_application_DEFERRED_"
                    "3235|contradictions_%d" % NCON),
        "evidence_level": ("compute+seal arm (matrices + nulls + amplitude "
                           "data sealed; gate evaluation per 3233 chain "
                           "deferred to 3235 -> no E-level change yet)"),
        "model_scope": "qwen pair; glm4 deferred (W_U absent from local hub)",
        "prereg_id": "G5-G30",
        "superseded_by": None,
        "detail": (
            "G5-G30 GAP-5 upstream-lever scan compute+seal arm (default "
            "(a) per 3233 prereg, GPU). Device anchors: A1 4b bf16 "
            "baseline hidden_states fp16 == sealed cache H bitwise "
            "(3154 all 208 rows x all slots + 3169 t=0 all 730 rows x "
            "all slots), A2 baseline margins == 3230 seen_m0 bitwise, "
            "C6v3 direct class readout argmax agreement 1.0000. 3154 "
            "prompt reconstruction verified (tags + d column, 0/208 "
            "mismatch both models). Sealed: coarse L-grid x {u_isa, "
            "u_hasa} flip matrices (J1 per layer per model), fine band "
            "x alpha {0.5,1,2} J1+marg, u* channel per-class med "
            "margin-shift (UL_amp data, 4 OOV spearman + 10cls "
            "report-only vs 3232 collapse), dR/dW peak flips, nulls "
            "10 random norm-matched seeds + 2 orthogonalized at peak "
            "layers, 4B nf4 bridge at peak layers (quantization "
            "perturbation quantified). Freeze %s (before any "
            "observation), res %s, npz %s. Gate application DEFERRED "
            "to 3235 per 3233 chain. Independent verify %s/%s PASS. "
            "Registry v1.31 UNTOUCHED (ef5189e7), gap v1.10 UNTOUCHED "
            "(378de1cd). pre-append ledger file sha8 %s."
            % (FREEZE, RES8, NPZ8, VP, VF, pre_file_sha8)),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LP + '.bak_3234', 'wb').write(raw)
    with open(LP, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('ledger: appended #386 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    assert len(already) == 1, ('duplicate 3234 entries', len(already))
    new_post = led['ledger_sha256_8']
    REPORT.append('ledger: 3234 entry already present, post=%s (idempotent)'
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

## Phase 3234: G5-G30 GAP-5 上游杠杆扫描 compute+seal 臂 [%s]

**执行模式**：GPU 计算+封存（默认 (a)，按 3233 预注册；无门控决策——门施加按 3233 链 DEFERRED 到 3235）；freeze **`%s`**（任何观测前）；res **`%s`**，verdict **`%s`**（矛盾 %d 项）；npz **`%s`**；registry v1.31 未动（`ef5189e7`）、gap v1.10 未动（`378de1cd`）；closeout 追加 #386。

### 装置锚（最强锚级，全过）
- **A1 逐位锚**：4b bf16 baseline 前向 hidden_states（fp16 化）与封存缓存**全槽逐位相等**——3154 全 208 行 × 全槽 + 3169 t=0 全 730 行 × 全槽 ⇒ 扫描装置与收集器完全同一；
- **A2 逐位锚**：baseline margin 与 3230 封存 seen_m0 **bitwise 相等**（drift 0.00e+00，两模型）；C6v3 直接类读出 argmax 一致率 **1.0000**（门 0.98）；
- **3154 prompt 重建验证**：MAT 表 + tags 六因子 + d 列（token 计数）0/208 错配 ×2 模型。

### 封存矩阵（npz）
- 粗扫：L ∈ {0..NL−1} × {u_isa, u_hasa} × 208 行（3154 面板）flip J1 逐层逐模型；
- 峰带细扫：band {L*−1, L*, L*+1} × α {0.5, 1, 2} × 两方向 × 730 行（3169 t=0）J1+marg；
- **u* 通道（UL_amp 数据）**：band 层逐类 median margin 偏移 ×10 类，4 OOV spearman vs 3228 E_cls + 10 类 report-only vs 3232 崩塌能量；
- dR/dW 峰层描述性 flip；峰层对照 10 随机 norm-matched（pinned seeds）+ 2 正交化（两面板）；
- **4B nf4 bridge**：4b 以 nf4 重跑峰层，量化扰动定量记录（bf16 常驻不可用先例）。

### 独立复核（verify.json，%s/%s PASS）
结果/冻结身份哈希（域口径分立：res 含 res_sha8=None、execution.json 不含自身键）、npz 文件 sha、J0 三锚逐位复算（cache-m0==3230 seen_m0、coarse_J0、fine_J0）、coarse flip_stats ×全部键自 J1 重算、pooled+Lstar 重算、fine flip/dD 自 J1+marg 重算、nulls 重算、UL_amp 双 spearman 重算（自封存 amps + 3228 E_cls + 3232 崩塌）、u* 逐类 med shift 重算、bridge 一致性、verdict/矛盾一致性、账本只读断言。

### 预注册 Phase 3235（候选）
- (a) **G5-G30 门评估 + registry/gap 决策臂**（默认，零/低 GPU，基于 3234 封存矩阵）：施加 UL_flip（峰层 flip_rate ≥0.15 ×≥3/4 折 ×2 模型，对照 null 分布）/UL_dose 单调/UL_spec ≤1/3/UL_amp rank-corr ≥0.6 + UL_loc report-only 记录（k* 带 [1,5]）；FOUND ⇒ registry upgrade_log；ABSENT ⇒ F 账本 append + GAP-5 mechanism_note（分布式承载定谳）；
- (b) u* 内容归因类级扩展（零 GPU，顺延）。
''' % (NOW, FREEZE, RES8, VERDICT, NCON, NPZ8, VP, VF)
if '## Phase 3234: G5-G30 GAP-5 上游杠杆扫描 compute+seal 臂' not in memo:
    with open(MP, 'a', encoding='utf-8', newline='') as f:
        f.write(SEC)
    REPORT.append('memo: Phase 3234 section appended')
else:
    REPORT.append('memo: Phase 3234 section already present (idempotent)')
memo_sha8 = fsha(MP)

# ---------- 3. daily ----------
DP = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-13.md')
DSEC = '''
### Phase 3234 — G5-G30 GAP-5 上游杠杆扫描 compute+seal 臂（%s）
GPU 计算+封存；freeze `%s`、res `%s`、npz `%s`、verdict `%s`（矛盾 %d）。装置锚 A1 逐位（3154 208 行+3169 730 行全槽）+A2 bitwise+C6v3=1.0000 全过；3154 prompt 重建 0/208 错配 ×2。封存：粗扫 L 网格×2 方向 J1、峰带 α 细扫 J1+marg、u* 通道逐类 med shift（UL_amp 双 spearman）、dR/dW 峰层 flip、nulls 10 随机+2 正交、4B nf4 bridge。门施加 DEFERRED 3235。独立复核 %s/%s PASS。账本 #386 post `%s`（pre file `%s`）。registry v1.31 / gap v1.10 未动。3235 预注册：(a) 门评估+决策臂（默认）；(b) u* 内容归因类级扩展。
''' % (NOW, FREEZE, RES8, NPZ8, VERDICT, NCON, VP, VF, new_post,
       pre_file_sha8)
daily = open(DP, encoding='utf-8').read() if os.path.exists(DP) else ''
if '### Phase 3234' not in daily:
    with open(DP, 'a', encoding='utf-8', newline='') as f:
        f.write(DSEC)
    REPORT.append('daily: Phase 3234 note appended')
else:
    REPORT.append('daily: Phase 3234 note already present (idempotent)')
daily_sha8 = fsha(DP)

# ---------- 4. MEMORY ----------
MEMP = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
mem = open(MEMP, encoding='utf-8').read()
if '3235 已预注册' in mem and '3234 收官' in mem and \
        '3234 已预注册' not in mem:
    REPORT.append('memory: already at 3235-prereg/3234-closeout state '
                  '(idempotent skip)')
else:
    lines = mem.splitlines()
    i_prereg = [i for i, l in enumerate(lines)
                if l.startswith('- **3234 已预注册**')]
    i_close = [i for i, l in enumerate(lines)
               if l.startswith('- 3233 收官：')]
    assert len(i_prereg) == 1 and len(i_close) == 1 and \
        i_close[0] == i_prereg[0] + 1, (i_prereg, i_close)
    NEW_LINES = [
        ('- **3235 已预注册**：(a) **G5-G30 门评估+决策臂**（默认，零/低 '
         'GPU，基于 3234 封存矩阵）：施加 UL_flip/dose/spec/amp 门 + '
         'UL_loc report-only；FOUND ⇒ registry upgrade_log；ABSENT ⇒ F '
         '账本 + GAP-5 mechanism_note（分布式承载定谳）；(b) u* 内容归因'
         '类级扩展（顺延）。'),
        ('- 3234 收官：upstream_lever_scan_sealed（freeze `%s`，res `%s`，'
         'npz `%s`），ledger #386 post `%s`；GPU compute+seal：装置锚 A1 '
         '逐位/A2 bitwise/C6v3=1.0000 全过，粗扫+峰带细扫+u* UL_amp+dR/dW+'
         'nulls+nf4 bridge 全封存，门施加 DEFERRED 3235；独立复核 %s/%s '
         'PASS；registry v1.31 / gap v1.10 未动。' % (FREEZE, RES8, NPZ8,
                                                      new_post, VP, VF)),
    ]
    lines[i_prereg[0]:i_close[0] + 1] = NEW_LINES
    with open(MEMP, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(lines) + '\n')
    REPORT.append('memory: 3234 prereg+3233 closeout lines -> 3235 prereg + '
                  '3234 closeout (anchor check OK)')
mem_sha8 = fsha(MEMP)

# ---------- 5. self-check readback ----------
memo2 = open(MP, encoding='utf-8').read()
daily2 = open(DP, encoding='utf-8').read()
mem2 = open(MEMP, encoding='utf-8').read()
assert '## Phase 3234: G5-G30 GAP-5 上游杠杆扫描 compute+seal 臂' in memo2 \
    and RES8 in memo2
assert '### Phase 3234' in daily2 and new_post in daily2
assert '3234 收官' in mem2 and new_post in mem2 and '3235 已预注册' in mem2
assert len(mem2) < 3000, ('MEMORY too long', len(mem2))
REPORT.append('self-check: MEMO/daily/MEMORY readback OK (mem len=%d)'
              % len(mem2))
REPORT.append('final: ledger_post=%s memo_sha8=%s daily_sha8=%s '
              'memory_sha8=%s' % (new_post, memo_sha8, daily_sha8, mem_sha8))
with open(os.path.join(ROOT, 'tests', 'gpt5_temp',
                       'phase3234_closeout_out.txt'), 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(REPORT) + '\n')
print('CLOSEOUT OK')
