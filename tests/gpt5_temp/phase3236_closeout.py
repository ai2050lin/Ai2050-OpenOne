# -*- coding: utf-8 -*-
"""Phase 3236 closeout: five writes (ledger #388 / MEMO / daily / MEMORY /
self-check), idempotent, disk-readback assertions."""
import hashlib
import json
import os
import datetime

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PREV_POST = '26a9fa05'
LED_N = 388
RD = os.path.join(ROOT, 'tests', 'glm5', 'result',
                  'rdc_query_construction_20260913', 'phase3236',
                  'g5d0_platform_pipeline_demo')
REPORT = []


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


res = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
ver = json.load(open(os.path.join(RD, 'verify.json'), encoding='utf-8'))
FREEZE = res['freeze_sha8']
RES8 = res['res_sha8']
VERDICT = res['verdict']
VP = ver.get('n_pass', '?')
VF = ver.get('n_fails', '?')
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
assert VERDICT == 'DEMO_PIPELINE_PARTIAL', VERDICT

# ---------- 1. ledger ----------
LP = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
raw = open(LP, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
already = [e for e in led['measurements'] if e.get('phase') == 3236]
if not already:
    assert len(led['measurements']) == 387 and \
        led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3236,
        "name": "g5d0_platform_pipeline_demo",
        "line": "G",
        "date": "2026-10-14",
        "model": "qwen3-4b bf16 (user-directed platform pipeline demo)",
        "verdict": ("DEMO_PIPELINE_PARTIAL|G1_sep_pos_PASS_x5tpl_"
                    "auc0.75-1.00|G2_stitch_0.500_FAIL(0.6)|G3_xlang_"
                    "0.557_PASS|G4_layer_report|contradictions_0"),
        "evidence_level": ("demo arm (end-to-end pipeline worked example; "
                           "demo-level gates, no E-level claim; landing "
                           "arm prereg 3236 -> shifted to 3237)"),
        "model_scope": "qwen3-4b only",
        "prereg_id": "G5-D0 (user-directed)",
        "superseded_by": None,
        "detail": (
            "Platform-pipeline demo (user-directed): one feature "
            "(F-DEMO-ISA-001 isa class-membership binding) walked through "
            "the full loop: task-list entry -> 5 language templates "
            "(ZH stmt/quest/ctx + EN stmt/quest) -> 160-row terminal test "
            "(4 entities x 4 classes x 5 templates x 2 langs, qwen3-4b "
            "bf16) -> stable features (F1 separation AUC 0.745-1.000, "
            "F2 cross-template spearman, F3 ZH-EN behavior 0.557, F4 "
            "atlas links cos(v,dR) 0.018-0.193 / cos(v,u*) ~0, F5 layer "
            "emergence curve) -> atlas assembly -> mechanism readout. "
            "Honest findings: (i) same-family templates stitch strongly "
            "(T1|T5 0.821, T2|T4 0.871) but class-continuation vs "
            "judgment families weakly (min 0.084) => mean 0.500 fails "
            "the 0.6 stitch gate: template families need calibration "
            "before pooling (3172 port-calibration 29 pct echo); (ii) u* "
            "projection ~0 for seen entities (u* is OOV-carrier, "
            "coherent with 3232); (iii) layer curve flat early then "
            "final-layer amplification x2-8 (L32->L35: 4.3->8.4 / "
            "1.7->13.3 / 5.0->10.4) => readout-amplification segment of "
            "GAP-5 visible in a single-feature demo. freeze %s (before "
            "any observation, JSON-roundtrip-normalized), res %s. "
            "Independent verify %s/%s PASS (2 probe bugs fixed: auc tie "
            "credit, sha8 self-key domain). Registry v1.31 UNTOUCHED, "
            "gap v1.10 UNTOUCHED. pre-append ledger file sha8 %s."
            % (FREEZE, RES8, VP, VF, pre_file_sha8)),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LP + '.bak_3236', 'wb').write(raw)
    with open(LP, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('ledger: appended #388 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    assert len(already) == 1, ('duplicate 3236 entries', len(already))
    new_post = led['ledger_sha256_8']
    REPORT.append('ledger: 3236 entry already present, post=%s '
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

## Phase 3236: G5-D0 平台全流程演示臂（用户指定） [%s]

**执行模式**：qwen3-4b bf16 单模型端到端演示（用户指定任务：用例子完整走一遍「任务清单→语言模版→终端测试→稳定特征→图谱拼图→机制破解」；原 3236 落账臂顺延为 3237 预注册）；freeze **`%s`**（任何观测前，JSON 往返归一化）；res **`%s`**，verdict **`%s`**；closeout 追加 #388。

### 流水线五步（一个特征的完整旅程：F-DEMO-ISA-001 类属绑定）

1. **任务清单条目**：特征 F-DEMO-ISA-001（isa 类属关系绑定），预设图谱对接点 = dR（3230 关系对比轴）/ u*（3225 崩塌载体）/ k*=3 层曲线（3152）；
2. **语言模版 ×5**：T1 中文陈述「{e}是一种」/ T2 中文问句「{e}是一种{c}吗？答案是」/ T3 英文陈述「The {e} is a kind of」/ T4 英文问句「Is the {e} a kind of {c}? Answer:」/ T5 中文语境「我喜欢{e}。{e}是一种」；实体 ×4（苹果/玫瑰/金鱼/锤子 + EN 对应）× 类 ×4 ⇒ **160 行测试矩阵**；
3. **终端测试**（qwen3-4b bf16，全层 37 槽捕获）：F1 分离度 AUC **0.745–1.000 全 5 模版为正**（G1 PASS）；F3 跨语言行为相关 **0.557**（G3 PASS）；
4. **稳定特征与拼图（核心发现）**：F2 跨模版 spearman——**同族模版强拼图**（T1|T5 0.821、T2|T4 0.871）、**跨族弱**（续接类 vs 判断类最低 0.084），均值 **0.500 未过 0.6 门（G2 FAIL）**⇒ 模版族间需先校准再合并（3172 端口校准 29 pct 的回声）——「多种语言模版如何拼图」的操作答案：**先族内收敛，再跨族校准**；F4 图谱对接：cos(v,dR)=0.018–0.193（续接类模版 > 判断类）、**cos(v,u*)≈0**（seen 实体无 u* 幅度——与 3232「u* 是 OOV 载体」自洽）；
5. **机制读出**：F5 层曲线——类读出分离度早层平坦（L0–12 ≈0–0.5）、后段爬升、**末层跳变 ×2–8**（L32→L35：4.3→8.4 / 1.7→13.3 / 5.0→10.4）⇒ GAP-5 三段式的「末层线性读出 + post-softmax 非线性」在单特征演示中肉眼可见。

**教学结论**：该例子展示了分布式逆向平台的最小闭环——一个特征从任务清单出发，经多模版测试获得行为稳定（AUC≈1）与几何指纹（v_T 方向），拼图时同族模版直接合并、跨族需校准，最后接入既有图谱节点（dR/u*/层曲线）并复现机制分段（早层绑定→末层放大）。G2 FAIL 本身是有效测量：模版族间的系统差就是「端口校准」要消除的对象。

### 独立复核（verify.json，%s/%s PASS）
身份哈希（freeze JSON 往返归一化域，不含自身 sha 键）、F1–F5 全部自封存 npz 独立重算、verdict 推导复算、账本/源 sha 只读断言；2 个探针 bug 修复（auc 平局计分、sha8 自键域），封存产物未动。

### 预注册 Phase 3237（候选）
- (a) **G5-G31 落账臂**（默认，零 GPU，照 3231 模板，自 3236 顺延）：F21（F20 族：上游亦无杠杆）append → registry v1.32 + GAP-5 mechanism_note 终版 → gap v1.11；
- (b) u* 内容归因类级扩展（零 GPU，顺延）。
''' % (NOW, FREEZE, RES8, VERDICT, VP, VF)
if '## Phase 3236: G5-D0 平台全流程演示臂' not in memo:
    with open(MP, 'a', encoding='utf-8', newline='') as f:
        f.write(SEC)
    REPORT.append('memo: Phase 3236 section appended')
else:
    REPORT.append('memo: Phase 3236 section already present (idempotent)')
memo_sha8 = fsha(MP)

# ---------- 3. daily ----------
DP = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-14.md')
DSEC = '''# 2026-10-14 工作日志

### Phase 3236 — G5-D0 平台全流程演示臂（用户指定，%s）
qwen3-4b bf16 端到端演示：任务清单条目（F-DEMO-ISA-001 类属绑定）→ 5 语言模版（ZH 陈述/问句/语境 + EN 陈述/问句）→ 160 行测试矩阵 → 稳定特征（F1 AUC 0.745–1.000 全正、F3 跨语言 0.557、F4 cos(v,dR) 0.018–0.193 / cos(v,u*)≈0、F5 层曲线末层跳变 ×2–8）→ 拼图（同族强 0.82/0.87、跨族弱 0.08–0.50，均值 0.500 未过 0.6 门 ⇒ 模版族需先校准再合并）→ 机制（早层绑定→末层放大，GAP-5 三段式单特征复现）。freeze `%s`、res `%s`、verdict `DEMO_PIPELINE_PARTIAL`（G2 FAIL 为有效测量）。独立复核 %s/%s PASS。账本 #388 post `%s`（pre file `%s`）。原 3236 落账臂顺延为 3237 预注册。
''' % (NOW, FREEZE, RES8, VP, VF, new_post, pre_file_sha8)
daily = open(DP, encoding='utf-8').read() if os.path.exists(DP) else ''
if '### Phase 3236' not in daily:
    with open(DP, 'a', encoding='utf-8', newline='') as f:
        f.write(DSEC)
    REPORT.append('daily: Phase 3236 note appended (2026-10-14.md)')
else:
    REPORT.append('daily: Phase 3236 note already present (idempotent)')
daily_sha8 = fsha(DP)

# ---------- 4. MEMORY ----------
MEMP = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
mem = open(MEMP, encoding='utf-8').read()
if '3237 已预注册' in mem and '3236 收官' in mem and \
        '3236 已预注册' not in mem:
    REPORT.append('memory: already at 3237-prereg/3236-closeout state '
                  '(idempotent skip)')
else:
    lines = mem.splitlines()
    i_prereg = [i for i, l in enumerate(lines)
                if l.startswith('- **3236 已预注册**')]
    i_close = [i for i, l in enumerate(lines)
               if l.startswith('- 3235 收官：')]
    assert len(i_prereg) == 1 and len(i_close) == 1 and \
        i_close[0] == i_prereg[0] + 1, (i_prereg, i_close)
    NEW_LINES = [
        ('- **3237 已预注册**：(a) **G5-G31 落账臂**（默认，零 GPU，照 '
         '3231 模板，自 3236 顺延）：F21（F20 族：上游亦无杠杆）append → '
         'registry v1.32 + GAP-5 mechanism_note 终版（三段式收官）→ gap '
         'v1.11；(b) u* 内容归因类级扩展（顺延）。'),
        ('- 3236 收官：DEMO_PIPELINE_PARTIAL（freeze `%s`，res `%s`），'
         'ledger #388 post `%s`；用户指定平台全流程演示臂：F-DEMO-ISA-001 '
         '×5 模版 ×160 行，F1 AUC 0.745–1.000、F3 跨语言 0.557、F2 拼图 '
         '0.500 未过门（同族强/跨族弱 ⇒ 族间校准，3172 回声）、F5 末层跳变 '
         '×2–8（GAP-5 读出段复现）；registry v1.31 / gap v1.10 未动。'
         % (FREEZE, RES8, new_post)),
    ]
    lines[i_prereg[0]:i_close[0] + 1] = NEW_LINES
    with open(MEMP, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(lines) + '\n')
    REPORT.append('memory: 3236 prereg+3235 closeout -> 3237 prereg + '
                  '3236 closeout (anchor check OK)')
mem_sha8 = fsha(MEMP)

# ---------- 5. self-check readback ----------
memo2 = open(MP, encoding='utf-8').read()
daily2 = open(DP, encoding='utf-8').read()
mem2 = open(MEMP, encoding='utf-8').read()
assert '## Phase 3236: G5-D0 平台全流程演示臂' in memo2 and RES8 in memo2
assert '### Phase 3236' in daily2 and new_post in daily2
assert '3236 收官' in mem2 and new_post in mem2 and '3237 已预注册' in mem2
assert len(mem2) < 3000, ('MEMORY too long', len(mem2))
REPORT.append('self-check: MEMO/daily/MEMORY readback OK (mem len=%d)'
              % len(mem2))
REPORT.append('final: ledger_post=%s memo_sha8=%s daily_sha8=%s '
              'memory_sha8=%s' % (new_post, memo_sha8, daily_sha8,
                                  mem_sha8))
with open(os.path.join(ROOT, 'tests', 'gpt5_temp',
                       'phase3236_closeout_out.txt'), 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(REPORT) + '\n')
print('CLOSEOUT OK')
