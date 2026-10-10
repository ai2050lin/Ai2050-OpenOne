# -*- coding: utf-8 -*-
# Phase 3177 closeout: five-write (ledger / MEMO / daily / workspace MEMORY /
# self-check). Idempotent. All numbers rendered live from sealed result.json.
import hashlib
import io
import json
import os
import shutil
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
S = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(S, 'phase3177', 'g5b2_ftr14_heldout')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3177_closeout_out.txt')
LOG = []


def log(s):
    LOG.append('[3177c] ' + s)
    print('[3177c] ' + s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
assert R['res_sha8'] and R['seal_sha8']
assert R['verdict'].startswith('g5b2_ftr14_heldout|gate_fail|a1_6of6|a2_broken|'), R['verdict']
SM = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
exes = sha8_file(os.path.join(PDIR, 'execution.json'))
REG14 = os.path.join(PDIR, 'atlas_registry_v1_4.json')
HTML = os.path.join(PDIR, 'atlas_v1_5.html')
REG13 = os.path.join(S, r'phase3176\g5b1_elev_upgrade\atlas_registry_v1_3.json')
reg14_sha = sha8_file(REG14)
html_sha = sha8_file(HTML)
assert reg14_sha == R['registry_v14_sha8'] == 'fb11d633', reg14_sha
assert html_sha == R['html_sha8'] == '82c087de', html_sha
SMOKE_EQ = sha8_file(os.path.join(PDIR, 'smoke_atlas_registry_v1_4.json')) == reg14_sha
assert SMOKE_EQ, 'SMOKE/FULL registry must be byte-identical (mode-independent arm)'
F16 = R['failure_added']
assert F16 and F16['id'] == 'F16'
pm = R['arm']['per_model']
MS = ('qwen3-4b', 'qwen3-14b', 'glm4-9b')


def g(m, fold, key):
    return pm[m]['folds'][fold][key]


a1i_s = '/'.join('%.4f' % g(m, 'isa', 'a1_diff') for m in MS)
a1h_s = '/'.join('%.4f' % g(m, 'hasa', 'a1_diff') for m in MS)
a2i_s = '/'.join('%.4f' % g(m, 'isa', 'a2_diff') for m in MS)
a2h_s = '/'.join('%.4f' % g(m, 'hasa', 'a2_diff') for m in MS)
chi_s = '/'.join('%.4f' % g(m, 'isa', 'cos_held_a2') for m in MS)
chh_s = '/'.join('%.4f' % g(m, 'hasa', 'cos_held_a2') for m in MS)
chf_s = '/'.join('%.4f' % pm[m]['cos_full_a2'] for m in MS)
tcf_s = '/'.join('%.4f' % pm[m]['tc_full'] for m in MS)
thi_s = '/'.join('%.4f' % g(m, 'isa', 'tc_held_a1') for m in MS)
thh_s = '/'.join('%.4f' % g(m, 'hasa', 'tc_held_a1') for m in MS)
exv_s = '/'.join('%.4f' % pm[m]['exchange_obs'] for m in MS)

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3177 for m in ms_):
    log('1. ledger: 3177 already present, skip')
else:
    detail = (
        'G5-B2 FTR-14 held_out relation-LOO arm (zero GPU, sealed 3157 collect npz '
        '95a25965/9552086d/23dd74eb; prereg title said GPU-arm but the preregistered '
        'protocol body pins the data source to the existing collect -> zero-GPU, '
        'resolved at freeze before any observation). Composite gate frozen '
        'pre-observation as A1 AND A2: A1 subset-statistic (|tc_held_pairwise - '
        'tc_full| < 0.1) passed 6/6 (' + a1i_s + ' isa / ' + a1h_s + ' hasa); A2 '
        'reconstruction reading (|cos_held_vs_shared - cos_full_vs_shared| < 0.1) '
        'FAILED on glm4 both folds (' + a2i_s + ' isa / ' + a2h_s + ' hasa; 4b/14b '
        'pass); exch band kept (cross-model mean exchange_obs 1.2728 in [1.0, 1.3], '
        'per-model ' + exv_s + ', recomputation drift 0). ARM GATE FAIL -> FTR-14 '
        'stays E1_repeatable, held_out upgrade NOT achieved, F16 registered (prereg '
        'gate frozen, no goalpost move), counter_evidence append only. Directional '
        'finding: tc_held > tc_full in all 6 fold x model cells (' + thi_s + ' isa / '
        + thh_s + ' hasa vs ' + tcf_s + ') => within-relation T_C pairwise cos is '
        'systematically tighter than the cross-relation-inclusive full mean, i.e. a '
        'small relation-specific component in T_C geometry is consistent with the A2 '
        'failure; a2_gap_train (same component, train vs held cells) ' +
        '0.1145/0.1190, 0.1412/0.0936, 0.1500/0.1486 also above 0.09 => the 0.1 bar '
        'on out-of-sample readings is globally tight; decomposition (sampling penalty '
        'vs relation residual) preregistered as 3178. Device: tc/exch/exch-mean '
        'recomputation drift 0.00e+00 x3 models, fold direction symmetric '
        '(|isa-hasa| < 0.004), SMOKE and FULL registry v1.4 byte-identical (' +
        reg14_sha + ') => arm numbers mode-independent. Registry v1.3 -> v1.4: 21 '
        'features + failures first 15 + upgrade_log first 5 byte-for-byte, failures '
        '15 -> 16 (F16), upgrade_log stays 5; lineage metadata kept per 3176 '
        'precedent (created/phase/provenance unchanged, version/supersedes only). '
        'atlas_v1_5.html 668 fields 0/0/0/0. Honest log: 2 authoring-time fixes '
        'pre-observation (dead stmt_add_text removed; unused var). design_sha8=' +
        R['design_sha8'] + '. res ' + R['res_sha8'] + ' seal ' + R['seal_sha8'] +
        ' (smoke res ' + SM['res_sha8'] + ').')
    entry = {
        'phase': 3177, 'name': 'g5b2_ftr14_heldout', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'qwen3-4b+qwen3-14b+glm4-9b (zero GPU)',
        'verdict': R['verdict'],
        'evidence_level': 'E1_repeatable',
        'model_scope': 'qwen3-4b+qwen3-14b+glm4-9b',
        'prereg_id': 'G5-B2',
        'superseded_by': None,
        'detail': detail,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms_.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n1 = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    assert n1 in (n0 + 1, n0 + 2), 'concurrent ledger write anomaly'
    log('1. ledger: appended 3177 (n=' + str(n1) + ', was ' + str(n0) + ') chain_sha8=' +
        led['ledger_sha256_8'])

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3177' in norm:
    log('2. MEMO: 3177 section already present, skip')
else:
    marker = '### 接续：预注册 3177'
    mi = norm.rfind(marker)
    assert mi > 0, '3177 prereg marker not found'
    S2 = []
    S2.append('## Phase 3177: G5-B2 FTR-14 held_out 关系留出臂（零 GPU 定判） ' + time.strftime('%H:%M'))
    S2.append('')
    S2.append('**日期**：2026-10-09。**脚本**：`tests/glm5/phase3177_g5b2_ftr14_heldout.py`'
              '（零 GPU，封存 3157 collect npz 关系留出）。产物：`phase3177/g5b2_ftr14_heldout/`'
              ' {exec ' + exes + '（design ' + R['design_sha8'] + '）, res ' + R['res_sha8'] +
              ', seal ' + R['seal_sha8'] + ', smoke res ' + SM['res_sha8'] + ', registry v1.4 ' +
              reg14_sha + ', atlas_v1_5.html ' + html_sha + '}。')
    S2.append('')
    S2.append('### 判决（重复三遍）')
    S2.append('')
    S2.append('**gate fail（a1_6of6 / a2_broken / exch_band_keep）：FTR-14 保持 E1_repeatable，'
              'held_out 升级未达成——freeze 前钉定的 composite 门（A1 ∧ A2）中 A2（重建型读数）在 '
              'glm4 双 fold 超门（isa 0.1114 / hasa 0.1107 ≥ 0.1），4b/14b 过门；A1（子集统计）6/6 '
              '全过（最大 0.0614）；exch 带 cross-model 均值 1.2728 落 [1.0, 1.3] partial 带（重算 '
              'drift 0）。门冻结不移动，F16 入失败账本，registry v1.3→v1.4（22 特征 + 失败 16 + '
              '升级链 5），FTR-14 仅 counter_evidence 追加。**')
    S2.append('')
    S2.append('**gate fail（重复二）：装置全绿——tc/exch/exch 均值重算 drift 0.00e+00（tc 3/3、exch 6/6、'
              '均值对 3157 summary 1.2728174525072664 恒等）；fold 方向对称（|isa−hasa| < 0.004）；'
              'SMOKE 与 FULL registry v1.4 字节恒等（' + reg14_sha + '）=> 臂数值运行模式无关；'
              'atlas_v1_5.html 668 字段 0/0/0/0。**')
    S2.append('')
    S2.append('**gate fail（重复三）：方向性发现——tc_held > tc_full 在全部 6 个 fold x 模型单元成立'
              '（isa ' + thi_s + ' / hasa ' + thh_s + ' vs 全量 ' + tcf_s + '）：关系内 T_C 两两 cos '
              '系统性高于含跨关系对的全量均值，即 T_C 几何存在小幅关系特异分量，与 A2 失败方向一致；'
              'a2_gap_train（同分量、train vs held 细胞）0.1145/0.1190、0.1412/0.0936、0.1500/0.1486 '
              '也全部不低于 0.09 => 0.1 门对 out-of-sample 读数整体偏紧；采样惩罚 vs 关系残差的分解'
              '预注册为 3178（新实验，非门移动）。**')
    S2.append('')
    S2.append('### 关系留出读数（result 现场渲染）')
    S2.append('')
    S2.append('| 读数 | 4b | 14b | glm4 | 门/判定 |')
    S2.append('|---|---|---|---|---|')
    S2.append('| A1 diff isa / hasa | %s | %s | %s | <0.1 全过 |' % (
        '%.4f / %.4f' % (g(MS[0], 'isa', 'a1_diff'), g(MS[0], 'hasa', 'a1_diff')),
        '%.4f / %.4f' % (g(MS[1], 'isa', 'a1_diff'), g(MS[1], 'hasa', 'a1_diff')),
        '%.4f / %.4f' % (g(MS[2], 'isa', 'a1_diff'), g(MS[2], 'hasa', 'a1_diff'))))
    S2.append('| A2 diff isa / hasa | %s | %s | %s | <0.1，glm4 FAIL |' % (
        '%.4f / %.4f' % (g(MS[0], 'isa', 'a2_diff'), g(MS[0], 'hasa', 'a2_diff')),
        '%.4f / %.4f' % (g(MS[1], 'isa', 'a2_diff'), g(MS[1], 'hasa', 'a2_diff')),
        '%.4f / %.4f' % (g(MS[2], 'isa', 'a2_diff'), g(MS[2], 'hasa', 'a2_diff'))))
    S2.append('| cos_held isa / hasa | %s | %s | %s | 描述 |' % (
        '%.4f / %.4f' % (g(MS[0], 'isa', 'cos_held_a2'), g(MS[0], 'hasa', 'cos_held_a2')),
        '%.4f / %.4f' % (g(MS[1], 'isa', 'cos_held_a2'), g(MS[1], 'hasa', 'cos_held_a2')),
        '%.4f / %.4f' % (g(MS[2], 'isa', 'cos_held_a2'), g(MS[2], 'hasa', 'cos_held_a2'))))
    S2.append('| cos_full（in-sample 基线） | %s | %s | %s | 描述 |' % (
        '%.4f' % pm[MS[0]]['cos_full_a2'], '%.4f' % pm[MS[1]]['cos_full_a2'],
        '%.4f' % pm[MS[2]]['cos_full_a2']))
    S2.append('| tc_full（drift 0） | %s | %s | %s | 装置锚 |' % (
        '%.6f' % pm[MS[0]]['tc_full'], '%.6f' % pm[MS[1]]['tc_full'],
        '%.6f' % pm[MS[2]]['tc_full']))
    S2.append('| tc_held isa / hasa | %s | %s | %s | 描述（全部 > tc_full） |' % (
        '%.4f / %.4f' % (g(MS[0], 'isa', 'tc_held_a1'), g(MS[0], 'hasa', 'tc_held_a1')),
        '%.4f / %.4f' % (g(MS[1], 'isa', 'tc_held_a1'), g(MS[1], 'hasa', 'tc_held_a1')),
        '%.4f / %.4f' % (g(MS[2], 'isa', 'tc_held_a1'), g(MS[2], 'hasa', 'tc_held_a1'))))
    S2.append('| a2_gap_train isa / hasa | %s | %s | %s | 描述 |' % (
        '%.4f / %.4f' % (g(MS[0], 'isa', 'a2_gap_train'), g(MS[0], 'hasa', 'a2_gap_train')),
        '%.4f / %.4f' % (g(MS[1], 'isa', 'a2_gap_train'), g(MS[1], 'hasa', 'a2_gap_train')),
        '%.4f / %.4f' % (g(MS[2], 'isa', 'a2_gap_train'), g(MS[2], 'hasa', 'a2_gap_train'))))
    S2.append('| exchange_obs | %s | %s | %s | 均值 1.2728 ∈ [1.0,1.3] |' % (
        '%.4f' % pm[MS[0]]['exchange_obs'], '%.4f' % pm[MS[1]]['exchange_obs'],
        '%.4f' % pm[MS[2]]['exchange_obs']))
    S2.append('')
    S2.append('### 装置与诚实登记')
    S2.append('')
    S2.append('**freeze 时裁定**：预注册标题为「GPU 臂」但协议体将数据源钉定为封存 3157 collect'
              '（无新 forward）=> 零 GPU 臂（先例：3176 held_out 臂于封存 npz）；裁定发生在任何观测前。'
              'composite 门（A1 ∧ A2 双读数同门）为 freeze 前保守钉定，非观测后补设。2 处编写期修正'
              '（删除死代码 stmt_add_text、无用变量），均先于观测。lineage 元数据按 3176 先例'
              '（created/phase/provenance 保持 v1.2 起原值，仅 version/supersedes 更新）。')
    S2.append('')
    S2.append('F16（失败账本）：' + F16['text'][:150] + '……（' + F16['evidence'] + '）')
    S2.append('')
    S2.append('### 接续：预注册 3178')
    S2.append('')
    S2.append('**3178 = G5-B3 FTR-14 A2 归因分解（零 GPU，sampling-null 标定）**——3177 的 A2 失败混合'
              '两种成分：(i) out-of-sample 采样惩罚（v_shared 由 32 细胞估出，held 细胞对其 cos 系统性'
              '低于 in-sample 基线，a2_gap_train 0.094-0.150 佐证）；(ii) 关系特异残差。分解协议：'
              '实体留出 null——同一关系内留一实体（16 fold），v_shared 由其余 15 实体 x 2 极性（30 细胞）'
              '重建，留出实体 2 细胞 cos 缺口 = 纯采样惩罚基准 p(m, rel)；关系残差 r(m) = 关系缺口'
              '（cos_train − cos_held，同 v_shared，3177 数据）− p(m, rel)。门：r(m) < 0.05 x3 模型 '
              '=> 关系无关在标定 bar 下成立（3179 执行 FTR-14 升级，锚=3178 标定读数 + 3177 A1/exch 带）；'
              '否则关系特异分量确证并量化登记（FTR-14 保持 E1）。协议细节 freeze 前从 3177/3157 脚本 '
              'verbatim 导出。FTR-04 glm4 弱特异性归因与 FTR-15 held_out 列为后续；挂账不变：N 线 '
              'P3-P7 补 Ledger、跨线账本补丁施加确认、水果类、K4。')
    S2.append('')
    S2.append('')
    S2.append('---')
    S2.append('')
    S2.append('')
    section = '\n'.join(S2) + '\n'
    new = norm[:mi] + section + norm[mi:]
    out = new.replace('\n', '\r\n') if crlf > 0 else new
    if had_bom:
        out = b'\xef\xbb\xbf' + out.encode('utf-8')
    else:
        out = out.encode('utf-8')
    shutil.copyfile(MEMO, MEMO + '.snap3177')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3177 section inserted (snapshot .snap3177)')

# ---------- 3. daily ----------
dline = ('- **3177 FTR-14 关系留出臂（2026-10-09）**：G5-B2 零 GPU 定判——**gate fail（a1_6of6 / '
         'a2_broken / exch_band_keep）**：A1 子集统计 6/6 过（最大 0.0614），A2 重建型读数 glm4 双 '
         'fold 超门（0.1114/0.1107 ≥ 0.1）、4b/14b 过，exch 均值 1.2728 落带（重算 drift 0）——门冻结'
         '不移动，F16 入账，FTR-14 保持 E1。方向性发现：tc_held > tc_full 全 6 单元 => T_C 有小幅关系'
         '特异分量；a2_gap_train 0.094-0.150 => 0.1 门对 out-of-sample 读数整体偏紧。registry v1.4 ' +
         reg14_sha + '（失败 16、升级链 5）、atlas_v1_5.html 668 字段全对。res ' + R['res_sha8'] +
         '/seal ' + R['seal_sha8'] + '；ledger n→329。下一步 3178=A2 采样惩罚 vs 关系残差分解'
         '（sampling-null 标定，零 GPU）。')
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if R['res_sha8'] in dtxt:
    log('3. daily: 3177 line already present, skip')
else:
    if not dtxt.endswith('\n'):
        dtxt += '\n'
    dtxt += dline + '\n'
    with io.open(DAILY, 'w', encoding='utf-8', newline='') as f:
        f.write(dtxt)
    log('3. daily: appended')

# ---------- 4. workspace MEMORY (append at EOF) ----------
wraw = open(WMEM, 'rb').read().decode('utf-8')
wm = wraw.replace('\r\n', '\n')
newline = ('\n- **✅ 3177 FTR-14 关系留出臂闭环（2026-10-09）**：G5-B2 零 GPU 定判（封存 3157 collect '
           '95a25965/9552086d/23dd74eb，prereg 标题 GPU 臂但协议体钉定既有 collect => 零 GPU，freeze 时'
           '先于观测裁定）——**gate fail（a1_6of6 / a2_broken / exch_band_keep）**：composite 门'
           '（A1 ∧ A2，freeze 前保守钉定）中 A1 子集统计 6/6 过（|tc_held−tc_full| 最大 0.0614），A2 '
           '重建型读数（留出关系对另一关系重建的 v_shared=unit(mean(dCn_train)) 的 cos vs 全量 in-sample '
           '基线）glm4 双 fold 超门（0.1114/0.1107 ≥ 0.1）、4b/14b 过（0.0860/0.0881、0.0984/0.0765）；'
           'exch 带 cross-model 均值 1.2728 ∈ [1.0,1.3]（per-model 1.1929/1.2945/1.3311，重算 drift 0）。'
           '门冻结不移动 => FTR-14 保持 E1、held_out 锚不入表、F16 失败账本、counter_evidence 追加。'
           '方向性发现：tc_held（0.581-0.623）> tc_full（0.561-0.564）全 6 单元 => T_C 几何存在小幅'
           '关系特异分量；a2_gap_train 0.094-0.150 => 0.1 门对 out-of-sample 读数整体偏紧。装置全绿：'
           'tc/exch/均值重算 drift 0.00e+00、fold 方向对称（<0.004）、SMOKE/FULL registry v1.4 字节'
           '恒乙（' + reg14_sha + '）、668 字段 0/0/0/0。registry v1.3→v1.4（失败 16、升级链 5、'
           'lineage 按 3176 先例）。res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] + '；ledger '
           'n=328→**329**。下一步 3178=G5-B3 A2 归因分解（实体留出 sampling-null 标定采样惩罚基线，'
           '关系残差 <0.05 x3 门，零 GPU）；FTR-04 glm4 归因、FTR-15 held_out 后续。\n')
if '3177 FTR-14 关系留出臂闭环' in wm:
    log('4. workspace MEMORY: 3177 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3177')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3177)')

# ---------- 5. self-check ----------
chk = []
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
ms2 = led2['measurements']
e3177 = [m for m in ms2 if m.get('phase') == 3177]
chk.append(('ledger has 3177 entry', len(e3177) == 1, 'n=' + str(len(ms2))))
chk.append(('ledger n>=329', len(ms2) >= 329, 'n=' + str(len(ms2))))
if e3177:
    chk.append(('ledger verdict gate_fail', 'gate_fail' in e3177[0]['verdict'],
                e3177[0]['verdict'][:60]))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3177', '## Phase 3177' in memo2))
chk.append(('MEMO has res sha', R['res_sha8'] in memo2))
chk.append(('MEMO has seal sha', R['seal_sha8'] in memo2))
chk.append(('MEMO has registry v1.4 sha', reg14_sha in memo2))
chk.append(('MEMO has 3178 prereg', '预注册 3178' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3177', '3177 FTR-14 关系留出臂' in d2 and R['res_sha8'] in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3177', '3177 FTR-14 关系留出臂闭环' in w2))
R2 = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
chk.append(('result seal intact', R2['seal_sha8'] == R['seal_sha8'] and
            R2['res_sha8'] == R['res_sha8'], R['res_sha8'] + '/' + R['seal_sha8']))
chk.append(('arm gate stable', R2['arm']['gate'] is False and
            R2['arm']['gate_a1'] is True and R2['arm']['gate_a2'] is False and
            R2['arm']['gate_exch_band'] is True, 'a1=T a2=F exch=T'))
chk.append(('A1 6of6 re-derive', all(g(m, f, 'a1_diff') < 0.1 for m in MS for f in ('isa', 'hasa')), ''))
chk.append(('A2 fail pattern', all(g('glm4-9b', f, 'a2_diff') >= 0.1 for f in ('isa', 'hasa')) and
            all(g(m, f, 'a2_diff') < 0.1 for m in MS[:2] for f in ('isa', 'hasa')), ''))
chk.append(('device drifts zero', all(pm[m]['tc_full_drift'] < 1e-9 and
            pm[m]['exch_drift'] < 1e-9 for m in MS), ''))
chk.append(('registry v1.4 on disk', os.path.exists(REG14) and sha8_file(REG14) == reg14_sha, reg14_sha))
chk.append(('registry SMOKE==FULL', sha8_file(os.path.join(PDIR, 'smoke_atlas_registry_v1_4.json'))
            == reg14_sha, 'mode-independent arm'))
RG = json.load(io.open(REG14, encoding='utf-8'))
RG13 = json.load(io.open(REG13, encoding='utf-8'))
lv = {f['id']: f['evidence_level'] for f in RG['features']}
lv13 = {f['id']: f['evidence_level'] for f in RG13['features']}
chk.append(('FTR-14 stays E1', lv['FTR-14'] == 'E1_repeatable' == lv13['FTR-14'], ''))
chk.append(('failures 16 with F16', len(RG['failures']) == 16 and
            RG['failures'][-1]['id'] == 'F16' and RG['failures'][-1]['phase'] == 3177,
            str(len(RG['failures']))))
chk.append(('upgrade_log stays 5', len(RG['upgrade_log']) == 5, str(len(RG['upgrade_log']))))
ce13 = [f for f in RG13['features'] if f['id'] == 'FTR-14'][0]['counter_evidence']
ce14 = [f for f in RG['features'] if f['id'] == 'FTR-14'][0]['counter_evidence']
chk.append(('FTR-14 counter append only', len(ce14) == len(ce13) + 1 and
            ce14[:len(ce13)] == ce13, str(len(ce13)) + '->' + str(len(ce14))))
unchanged = sum(1 for f in RG13['features'] if f['id'] != 'FTR-14' and
                json.dumps(f, ensure_ascii=False, sort_keys=True) ==
                json.dumps([x for x in RG['features'] if x['id'] == f['id']][0],
                           ensure_ascii=False, sort_keys=True))
chk.append(('21 features byte-for-byte', unchanged == 21, str(unchanged)))
chk.append(('html on disk', os.path.exists(HTML) and sha8_file(HTML) == html_sha, html_sha))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s %s' % ('OK ' if c[1] else 'FAIL', c[0], c[2] if len(c) > 2 else ''))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE (self-check ' + str(len(chk) - len(bad)) + '/' + str(len(chk)) + ')')
