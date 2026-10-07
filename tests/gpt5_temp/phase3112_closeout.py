# -*- coding: utf-8 -*-
"""Phase 3112 closeout (idempotent):
Ledger -> MEMO Phase 3112 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3112'
        r'\omega_p110_broadcast_emergence')
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
    'emerge_L6|few_channels|replicated'
assert res['emergence_layer_3105'] == 6
assert res['corr_structure'] == 'few_channels'
assert res['replicated_3106'] is True

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3112
           for m in led['measurements']):
    claim = (
        'Omega-P110 (3112, offline T3 on frozen 3105/'
        '3106 captures, no GPU, 24s) - broadcast '
        'emergence verdict.  Verdict emerge_L6|'
        'few_channels|replicated, answering the three '
        'preregistered questions of 3111.  Q1 '
        'EMERGENCE LAYER (3105 per-layer single-'
        'coordinate median AUC at last position): L0 '
        '0.5210 (chance) L1 0.5319 L2 0.6007 L3 0.6377 '
        'L4 0.7724 L5 0.8960 L6 0.9386 (FIRST >= 0.90 '
        '= emergence layer) L7 0.9472 (peak) L8 0.9222 '
        '(mild decline).  frac>0.7: 0.000/0.000/0.173/'
        '0.319/0.639/0.787/0.835/0.826/0.806.  The '
        'broadcast forms PROGRESSIVELY across depth '
        '(chance at L0-L1, start L2-L3, acceleration '
        'L4, crystallization L5-L6) - NOT a one-step '
        'event at binding L3 or truth injection L4.  '
        'Q2 COORDINATE CORRELATION STRUCTURE (2560x2560 '
        'Pearson on standardized states): lambda1 '
        'share 0.4791 -> few_channels band (0.20-0.50, '
        'just 0.021 under the 0.50 single-source '
        'threshold), PR 4.1 effective channels, '
        'mean|corr| 0.4203, and the DECISIVE PC1-truth '
        'AUC 0.9939 - the top principal component of '
        'coordinate covariation is itself a near-'
        'perfect truth readout.  NOT one single '
        'broadcast source but FEW STRONG CHANNELS; the '
        'holographic readout rides a low-dimensional '
        'covariation skeleton.  Q3 CROSS-MATERIAL '
        'REPLICATION (3106 chain/scatter): curve L0 '
        '0.6067 -> L4 0.6442 -> L6 0.7715 -> L7 0.7777 '
        '-> L8 0.7432 >= 0.70 gate -> replicated=True; '
        'same qualitative shape (gradual rise + mild '
        'end decline) but overall LOWER with no 0.90 '
        'spike (plateau ~0.78) - material affects '
        'broadcast STRENGTH, not formation MODE.  '
        'SYNTHESIS: the truth broadcast is built '
        'through depth (crystallizes L4-L6), carried '
        'by few strong covariation channels (PC1 '
        'AUC 0.994), replicates across materials; '
        'the write-in localization window narrows to '
        'L4-L6.  CAVEATS: (i) lambda1 0.479 vs 0.50 '
        'threshold gap 0.021 - few_channels vs '
        'single_source boundary not robust; (ii) 3106 '
        'plateau ~0.78 vs 3105 ~0.94 material gap '
        'unexplained; (iii) few_channels is a '
        'covariation description, not a causal wiring '
        'map - writing components unidentified; (iv) '
        'PC1-truth AUC near-perfect may partly reflect '
        'truth variance dominating total variance in '
        'this record set.  NEXT 3113: (a) artifact '
        'separation - within-condition contrast with '
        'token-matched swapped truth (belief vs data-'
        'construction artifact, 3111 caveat 2); (b) '
        'component-level write-in localization of L4-'
        'L6 (which attention heads / MLP write the '
        'broadcast; 3106 binding L3 -> truth L4 entry '
        'hypothesis); (c) then T4 multi-step '
        'autoregression + write-in side.')
    meas = {
        'meas_id': 'meas3112_omega_p110_'
                   'broadcast_emergence',
        'phase': 3112,
        'claim': claim,
        'verdict': 'emerge_L6|few_channels|replicated',
        'anchors': 'design_seal.json frozen before '
                   'computation: emergence_rule first '
                   'layer median AUC >= 0.90 at LAST '
                   'position; corr_rule lambda1 share '
                   '>= 0.50 single_source / 0.20-0.50 '
                   'few_channels / < 0.20 distributed, '
                   'PC1-truth AUC reported; '
                   'replication_rule 3106 last|L8 '
                   'median >= 0.70; direction-free '
                   'AUC TEST only; per-layer norm '
                   'frozen',
        'artifacts': {
            'result': 'phase3112/omega_p110_'
                      'broadcast_emergence/'
                      'result.json',
            'seal': 'phase3112/omega_p110_'
                    'broadcast_emergence/'
                    'design_seal.json'},
        'hashes': {},
        'note': 'offline only; two formal runs '
                'identical; emergence L6, PC1-truth '
                'AUC 0.9939 decisive for few-channel '
                'carrier',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3112_omega_p110_broadcast_emergence')
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

# ---------- MEMO Phase 3112 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3112:' not in memo:
    sec = u'''## Phase 3112: Ω-P110 广播出现层判决——3105 九层曲线 L6 首破 0.90（渐进形成非一步到位）；坐标共变 few_channels（λ1 share 0.479、PC1-truth AUC 0.994）；3106 跨材料复现 L8 0.743 [[NOW]]

**性质**：T3 第 6 Phase（3107–3112 读出端六部曲），纯离线（3105/3106 冻结 capture，免 GPU，24s，两次正式运行结果逐位一致）。预注册（design_seal.json 先于一切统计冻结）：emerge 规则=首个单坐标中位 AUC ≥0.90 的层（last 位置）；corr 规则=λ1 share ≥0.50 单源 / 0.20–0.50 few_channels / <0.20 distributed + PC1-truth AUC 报告；复现规则=3106 last|L8 中位 ≥0.70。回答 3111 预注册三问。

### 1. 广播出现层定位（3105 九层单坐标中位曲线）
| 层 | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| median AUC | 0.5210 | 0.5319 | 0.6007 | 0.6377 | 0.7724 | 0.8960 | **0.9386** | 0.9472 | 0.9222 |
| frac>0.7 | 0.000 | 0.000 | 0.173 | 0.319 | 0.639 | 0.787 | 0.835 | 0.826 | 0.806 |

**emerge = L6**（首个 ≥0.90，预注册规则命中），峰值 L7 0.9472，L8 轻微回落 0.9222。广播沿深度**渐进形成**：L0–L1 chance → L2–L3 起步 → L4 加速 → L5–L6 结晶——不是绑定 L3 或真值注入 L4 的一步到位事件，而是 L4–L6 深度区间逐层增强的过程性构建。

### 2. 坐标相关结构：few_channels，PC1 近完美携带真值
2560×2560 Pearson（标准化状态，全主记录）：**λ1 share = 0.4791** → few_channels 档（0.20–0.50，距 0.50 单源阈仅 0.021）、**PR = 4.1**（有效通道数）、mean|corr| = 0.4203、**PC1-truth AUC = 0.9939**——坐标共变的主成分本身几乎是完美的真值读出器。判决：不是严格单一广播源，而是**少数强通道**（~4 个有效方向）；全息读出由低维共变骨架承载。

### 3. 跨材料复现（3106 chain/scatter）
九层曲线 L0 0.6067 → L4 0.6442 → L6 0.7715 → L7 0.7777 → **L8 0.7432 ≥ 0.70 门 → replicated=True**。曲线同形（渐进上升 + 末端轻微回落）但整体更低、无 0.90 尖峰（平台 ~0.78）——材料影响广播**强度**，不改变形成**方式**。

### 4. 综合（3107–3112 六部曲）
真值广播是深度过程性构建的：在 L4–L6 逐层结晶（3105 emerge L6），由少数强共变通道承载（PC1 单独 AUC 0.994），跨材料复现成立（3106 同形较弱）。**写入端定位窗口收窄至 L4–L6**。

### 5. 硬伤
① λ1 share 0.479 与 0.50 单源阈仅差 0.021——few_channels vs single_source 的边界判决不稳健；② 3106 平台 ~0.78 vs 3105 ~0.94 的材料差来源未解释；③ few_channels 是共变描述而非因果接线图——写入组件未定位；④ PC1-truth AUC 近完美可能部分因真值方差在该记录集总方差中占比过高。

### 6. 3113 预注册（观测前冻结于 3113 seal）
① **伪迹分离强对照**：同词汇换真值（token 匹配的条件内对照），分离"信念状态"与"数据构造伪迹"（3111 硬伤②）；② **写入端组件级定位**：追踪 L4–L6 哪些 attention head / MLP 写入广播信号（3106 绑定 L3→真值 L4 入口假设）；③ 之后 T4 多步自回归 + 写入端条件化结构。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3112/omega_p110_broadcast_emergence/`（result.json、design_seal.json、run_log.txt）；脚本 `tests/glm5/phase3112_omega_p110_broadcast_emergence.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3112)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3112 Omega-P110 (offline T3, no GPU): '
          'emergence verdict emerge_L6|few_channels|'
          'replicated. 3105 per-layer single-coordinate '
          'median: L0 0.521 -> L4 0.772 -> L6 0.939 '
          '(FIRST >= 0.90 = emergence) -> peak L7 0.947 '
          '-> L8 0.922; broadcast forms PROGRESSIVELY '
          'across depth, not at binding L3 / injection '
          'L4. Coordinate correlation (2560): lambda1 '
          'share 0.479 -> few_channels (0.021 under '
          'single-source threshold), PR 4.1, PC1-truth '
          'AUC 0.994 (top PC nearly perfect truth '
          'readout). 3106 replication: L8 0.743 >= 0.70 '
          'gate -> replicated; same shape, weaker '
          'plateau ~0.78. Write-in window narrowed to '
          'L4-L6. NEXT 3113: artifact separation '
          '(token-matched swapped truth within-'
          'condition contrast) + component-level '
          'write-in localization L4-L6; then T4.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3112 Omega-P110' not in prev:
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
if 'max=3112' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3111）',
        '## 机制链状态（3112）\n'
        '- 3112：出现层判决 emerge_L6|few_channels|'
        'replicated。3105 九层单坐标中位 L0 0.521→L4 '
        '0.772→L6 0.939（emerge）→峰 L7 0.947→L8 '
        '0.922：广播沿深度渐进形成。坐标共变 few_'
        'channels：λ1 share 0.479、PR 4.1、PC1-truth '
        'AUC 0.994。3106 复现 L8 0.743（同形但更低，'
        '平台 ~0.78）。写入端窗口收窄至 L4-L6。\n')
    mem_new = mem_new.replace(
        'max=3111', 'max=3112').replace(
        '下一 3112：**广播出现层定位+坐标相关结构+3106 复现**→ '
        '之后 T4 + 写入端 之后 T4 + 写入端 '
        '之后 T4 多步自回归 + 写入端条件化结构。',
        '下一 3113：**伪迹分离强对照（同词汇换真值）+ '
        '写入端组件级定位（L4-L6 谁写入广播）**→ '
        '之后 T4 多步自回归。')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
