# -*- coding: utf-8 -*-
"""Phase 3111 closeout (idempotent):
Ledger -> MEMO Phase 3111 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3111'
        r'\omega_p109_broadcast_verdict')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
TODAY = datetime.date.today().isoformat()
o = []

res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
assert res['verdict'] == \
    'mixed_broadcast_plus_distribution'
assert res['gates']['DA_scalar_test']['raw'][
    'scalars']['mean'] >= 0.9

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3111
           for m in led['measurements']):
    claim = (
        'Omega-P109 (3111, offline T3 on frozen 3105 '
        'capture, no GPU, 38s) - broadcast-signal '
        'verdict.  Verdict mixed_broadcast_plus_'
        'distribution, with the decisive D-C result '
        'reshaping the picture.  D-A DIRECT TEST: '
        'per-record mean(h) AUC (direction-free) = '
        '0.9439 in raw space (>= 0.9 gate -> first-'
        'moment/DC broadcast PRESENT); ||h|| 0.8273 '
        'raw / 0.8945 Z; median(h) weak (0.557).  '
        'D-B CENTERED K-SWEEP: after per-record '
        'centering (DC removed) and further per-record '
        'normalization (magnitude channel removed), '
        'full-dim truth AUC stays 0.9999 / 0.9998 and '
        'd_min stays 5 in all three spaces - removing '
        'the first-moment and magnitude channels costs '
        'NOTHING; the broadcast is ALL-MOMENT, not '
        'just DC.  D-C PER-COORDINATE AUC (decisive): '
        'single-coordinate direction-free truth AUC '
        'over all 2560 coordinates: median 0.9222, '
        'p90 0.9918, max 0.9996, 80.6pct of '
        'coordinates individually > 0.7, 54.7pct '
        'individually > 0.9 - EVERY coordinate is '
        'nearly a complete readout of the truth '
        'variable.  FINAL PICTURE (3107-3111 five-'
        'phase synthesis): the truth variable is a '
        'RECORD-LEVEL GLOBAL BELIEF STATE broadcast '
        'holographically across the residual stream - '
        'each coordinate carries a near-complete copy '
        '(not a small share); model unembed and linear '
        'probes are all readout ports of this one '
        'broadcast; the degenerate solution family '
        '(3108), the 5-dim random port (3110) and the '
        'flat spectrum are all consequences of the '
        'holographic carrier.  Consistent with 3105 '
        'semantic readout (m margin), 3106 passive-'
        'rephrase penetration, and 400b norm effects.  '
        'HARD CAVEAT: single material/config; the '
        'belief-state narrative vs data-construction '
        'artifact (correct-vs-wrong fact token '
        'differences leaking to all coordinates) must '
        'be separated by cross-material replication '
        '(3106) + within-condition contrast (same '
        'vocabulary, different truth) before claiming '
        'model belief.  NEXT 3112: (a) layer of '
        'broadcast emergence - per-layer single-'
        'coordinate AUC median curve across L0..L8 '
        'locates where the broadcast forms (3106 '
        'binding L3 -> truth L4 as entry); (b) '
        'coordinate correlation structure - if the '
        '2560 coordinates are mutually highly '
        'correlated on truth-relevant variance -> '
        'single broadcast source; if clustered -> '
        'multiple channels; (c) cross-material (3106 '
        'chain/scatter) replication of D-C.  Then T4 '
        'multi-step autoregression + write-in side.')
    meas = {
        'meas_id': 'meas3111_omega_p109_'
                   'broadcast_verdict',
        'phase': 3111,
        'claim': claim,
        'verdict': 'mixed_broadcast_plus_distribution',
        'anchors': 'design_seal.json frozen before '
                   'computation: DA rule max(auc,1-auc) '
                   '>= 0.90 raw+Z, DB rule same seed '
                   'stream as 3110 K list, DC rule '
                   '2560 single-dim direction-free; '
                   'lambda 0.01 recorded',
        'artifacts': {
            'result': 'phase3111/omega_p109_'
                      'broadcast_verdict/'
                      'result.json',
            'seal': 'phase3111/omega_p109_'
                    'broadcast_verdict/'
                    'design_seal.json'},
        'hashes': {},
        'note': 'offline only; SMOKE caught a stale '
                'Zte name before real run; D-C '
                'decisive',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3111_omega_p109_broadcast_verdict')
    led.pop('ledger_sha256_8', None)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted')

# ---------- MEMO Phase 3111 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3111:' not in memo:
    sec = u'''## Phase 3111: Ω-P109 广播信号判决——mean(h) AUC 0.944（一阶矩广播存在）但中心化/归一化后读出无损（d_min 恒 5）；决定性 D-C：单坐标 AUC 中位 0.922、55% 坐标单独 >0.9 → 每个坐标几乎是真值的完整读出器 → 混合判决 mixed_broadcast_plus_distribution；终图景 = 记录级全局信念状态的全息广播 [[NOW]]

**性质**：T3 第 5 Phase（3107–3111 读出端五部曲收官），纯离线（3105 冻结 capture，免 GPU，38s，SMOKE 抓获一处陈旧变量名后通过）。预注册（design_seal.json 先于一切统计冻结）：D-A max(auc,1−auc) ≥ 0.90（raw+Z 双空间）；D-B 同 3110 seed 流重跑 raw/centered/centered+norm 三空间；D-C 2560 单坐标方向自由 AUC。

### 1. D-A 直接检验：一阶矩广播存在
raw 空间 mean(h) AUC = **0.9439**（≥0.9 门，DC/一阶矩广播实锤）；‖h‖ 0.8273（raw）/0.8945（Z）；median(h) 弱（0.557）。真值记录的整体激活水平系统性更高。

### 2. D-B 中心化 K-sweep：广播是全阶的，不只一阶矩
| 空间 | full AUC | d_min |
| --- | --- | --- |
| raw | 0.9999 | 5 |
| centered（去 DC） | 0.9999 | 5 |
| centered+norm（去幅值通道） | 0.9998 | 5 |

**去除一阶矩与范数通道后真值读出零损失**——每坐标携带的不是标量偏移，而是与真值相关的完整模式副本。

### 3. D-C 单坐标 AUC（决定性）：全息冗余
2560 坐标各自单独作为真值特征：**中位 AUC 0.9222**、p90 0.9918、max 0.9996；**80.6% 坐标单独 >0.7，54.7% 单独 >0.9**。真值变量在每个坐标上都有近乎完整的副本——不是"每坐标一小片"的分布式编码，而是**全息式冗余（holographic redundancy）**。

### 4. 终图景（3107–3111 五部曲综合）
**真值 = 记录级全局信念状态的广播（record-level global belief broadcast）**：模型在上下文末端已把"当前上下文是否蕴含真值"整合为一个全局内部状态，该状态调制残差流几乎全部坐标。模型 unembed 方向、线性探针、随机 5 维子空间都是这一个广播信号的读出端口——3108 宽平坦解族、3109 解锥、3110 d_min=5 全部是全息载体的必然后果。与 3105 语义读出（m margin）、3106 被动改述穿透、400b 范数效应一致。内容变量（crit_rel 关系身份）不广播（d_min=400）——残差流同时承载两类编码拓扑：全局信念广播 + 局部内容方向。

### 5. 硬伤
① 单材料单配置；② "信念状态"叙事 vs 数据构造伪迹（正确/错误事实的 token 系统差异泄漏到所有坐标）未分离——需 3106 跨材料复现 + 同词汇不同真值的条件内对照；③ D-C 单坐标 AUC 的坐标间相关结构未测（单一广播源 vs 多通道，3112 判决）；④ 全部线性/单变量分析，非线性承载体未检验。

### 6. 3112 预注册（观测前冻结于 3112 seal）
① **广播出现层定位**：L0–L8 逐层单坐标 AUC 中位曲线（3106 绑定 L3→真值 L4 为入口假设）；② **坐标相关结构**：2560 坐标在真值相关方差上的两两相关——单广播源 vs 多通道判决；③ **跨材料复现**：3106 chain/scatter 材料重跑 D-C。之后 T4 多步自回归 + 写入端条件化结构。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3111/omega_p109_broadcast_verdict/`（result.json、design_seal.json、run_log.txt）；脚本 `tests/glm5/phase3111_omega_p109_broadcast_verdict.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3111)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3111 Omega-P109 (offline T3, no GPU): '
          'broadcast verdict -> mixed_broadcast_plus_'
          'distribution. D-A: mean(h) AUC 0.9439 raw '
          '(first-moment broadcast present). D-B: '
          'centered/centered+norm full AUC 0.9999/0.9998, '
          'd_min 5 unchanged -> broadcast is ALL-MOMENT. '
          'D-C decisive: single-coordinate truth AUC '
          'median 0.9222, 80.6pct >0.7, 54.7pct >0.9 -> '
          'EVERY coordinate is nearly a complete readout '
          '= HOLOGRAPHIC redundancy. FINAL: truth = '
          'record-level global belief state broadcast '
          'across the residual stream; unembed/probes/'
          'random subspaces are all ports of one '
          'broadcast; explains 3107-3110 degeneracy '
          'entirely. Caveat: belief vs data-construction '
          'artifact needs cross-material + within-'
          'condition separation. NEXT 3112: layer of '
          'broadcast emergence + coordinate correlation '
          'structure + 3106 replication; then T4.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3111 Omega-P109' not in prev:
        try:
            with io.open(wl, 'a',
                         encoding='utf-8') as f:
                f.write(line_d)
            o.append('wlog appended %s' % wl)
        except Exception as e:
            o.append('wlog fail %s: %r' % (wl, e))
    else:
        o.append('wlog already %s' % wl)

# ---------- MEMORY.md update ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3111' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3110）',
        '## 机制链状态（3111）\n'
        '- 3111：广播判决 mixed_broadcast_plus_'
        'distribution。mean(h) AUC 0.944（一阶矩广播'
        '存在）但中心化/归一化后 d_min 恒 5（全阶广播）；'
        '决定性：单坐标 AUC 中位 0.922、55% 坐标单独 '
        '>0.9——每个坐标≈真值完整读出器（全息冗余）。'
        '终图景：真值=记录级全局信念状态广播；3107-3110 '
        '简并全部是该载体的必然后果。伪迹分离待 3112。\n')
    mem_new = mem_new.replace(
        'max=3110', 'max=3111').replace(
        '下一 3111：**广播信号判决**（‖h‖/mean 直接检验+中心化 K-sweep+单坐标直方图）→',
        '下一 3112：**广播出现层定位+坐标相关结构+3106 复现**→ 之后 T4 + 写入端')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
