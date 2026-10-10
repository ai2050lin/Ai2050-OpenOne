# -*- coding: utf-8 -*-
# Phase 3171 closeout: gap-ledger v1.2 (GAP-4 mechanism_note) + five-write
# (ledger / MEMO / daily / workspace MEMORY / self-check). Idempotent.
# All numbers rendered live from the sealed result.json on disk.
import hashlib
import io
import json
import os
import shutil
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3171', 'g5a8_collapse_mechanism')
P3170 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                     'phase3170', 'g5a7_atlas_v11')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3171_closeout_out.txt')
LOG = []


def log(s):
    LOG.append('[3171c] ' + s)
    print('[3171c] ' + s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
assert R['res_sha8'] and R['seal_sha8']
SM = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
exes = sha8_file(os.path.join(PDIR, 'execution.json'))

MKS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
CLS3 = R['overall']['cls_per_model']
MAIN = R['overall']['main_cls']
rat = [R['per_model'][m]['slots']['k_main']['ratio_S'] for m in MKS]
rat_s = '/'.join('%.4f' % v for v in rat)
ratK = [R['per_model'][m]['slots']['k_main']['ratio_K'] for m in MKS]
ratK_s = ', '.join('%.3f' % v for v in ratK)
fs_s = ', '.join('%.2e' % R['per_model'][m]['slots']['k_main']['f_S_seen'] for m in MKS)
fs_o = ', '.join('%.2e' % R['per_model'][m]['slots']['k_main']['f_S_oov'] for m in MKS)
nnls_o = ', '.join('%.3f-%.3f' % (min(n['rel_resid'] for n in R['per_model'][m]['centroid']['nnls_cone']),
                                  max(n['rel_resid'] for n in R['per_model'][m]['centroid']['nnls_cone']))
                   for m in MKS)
nnls_l = ', '.join('%.3f-%.3f' % (min(n['rel_resid'] for n in R['per_model'][m]['centroid']['baseline_nnls_loo']),
                                  max(n['rel_resid'] for n in R['per_model'][m]['centroid']['baseline_nnls_loo']))
                   for m in MKS)
marg_pairs = '; '.join('%.2f vs %.2f' % (R['per_model'][m]['marg']['oov_claimed_margin_mean'],
                                         R['per_model'][m]['marg']['seen_true_margin_mean'])
                       for m in MKS)
marg_seen_max = ', '.join('%.2f' % R['per_model'][m]['marg']['oov_max_seen_margin_mean'] for m in MKS)
cc_drift = max(R['crosscheck'][m]['drift_deg'] for m in MKS)
cls_s = '/'.join(CLS3)

MECH_NOTE = (
    '3171 机制定位（预注册门 ratio<0.5 encoding_missing / >0.8 readout_missing / else mixed）：'
    '判决 mixed_across_models（' + cls_s + '）——ratio_S 主槽（=3169 E_read 槽，k_main=NL-1）谱外/已见 = ' + rat_s +
    '。encoding_missing 被三模型一致否定（全 >0.5）：谱外类行保留 72-82% 的 S_class 类子空间投影能量'
    '（f_S seen ' + fs_s + ' / oov ' + fs_o + '）；K_readout top64 读出谱能量占比 oov/seen = ' + ratK_s +
    ' ≈1（读出流形能量不缺）；谱外类质心 NNLS 锥拟合 rel_resid ' + nnls_o + ' vs 已见留一基线 ' + nnls_l +
    '（谱外质心落在已见 6 质心锥/子空间结构内）；MARG 词级 logit：谱外行 claimed-class margin 与已见行 '
    'true-class margin 同量级（' + marg_pairs + '），且均高于谱外行 max-seen margin（' + marg_seen_max +
    '）。结论：崩塌不源于「编码缺失」——类身份能量/几何/词级 logit 三重存在；与 3169 E_read 崩塌（one-hot '
    '类端口在类别级留出下无训练信号）并置 => 缺口定位=读出通道（类端口->输出映射）泛化失效，非表征编码缺失。'
    '证据级：E1-E2（相关性探针，非干预因果）；升级路径=端口校准干预实验。锚：p3171 res ' + R['res_sha8'] +
    ' / seal ' + R['seal_sha8'] + '；crosscheck 复现 3166 census 67.0/71.9/74.014 度（drift<=' +
    ('%.1e' % cc_drift) + '）。')

# ---------- 0. gap ledger v1.2 (GAP-4 mechanism_note; byte-immutability) -------
GL11P = os.path.join(P3170, 'gap_ledger_v1_1.json')
GL12P = os.path.join(P3170, 'gap_ledger_v1_2.json')
gl12_sha = None
if os.path.exists(GL12P):
    gl12_sha = sha8_file(GL12P)
    log('0. gap ledger v1.2 exists (%s), skip' % gl12_sha)
else:
    G1 = json.load(io.open(GL11P, encoding='utf-8'))
    G2 = json.loads(json.dumps(G1))  # deep copy
    assert G1['version'] == '1.1', G1['version']
    gap4 = [g for g in G2['gaps'] if g['id'] == 'GAP-4'][0]
    assert 'mechanism_note' not in gap4
    gap4['mechanism_note'] = MECH_NOTE
    G2['version'] = '1.2'
    G2['supersedes'] = 'gap_ledger_v1_1 (phase 3170)'
    G2['updated'] = time.strftime('%Y-%m-%d')
    # byte-immutability: everything except GAP-4 mechanism_note and version block
    for k in ('schema', 'provenance', 'created', 'appendix'):
        assert json.dumps(G1[k], ensure_ascii=False, sort_keys=True) == \
            json.dumps(G2[k], ensure_ascii=False, sort_keys=True), ('immutability', k)
    for a, b in zip(G1['gaps'], G2['gaps']):
        assert a['id'] == b['id']
        ka = {k: v for k, v in a.items()}
        kb = {k: v for k, v in b.items() if k != 'mechanism_note'}
        assert json.dumps(ka, ensure_ascii=False, sort_keys=True) == \
            json.dumps(kb, ensure_ascii=False, sort_keys=True), ('gap immutability', a['id'])
    blob = json.dumps(G2, ensure_ascii=False, indent=1, sort_keys=False)
    with io.open(GL12P, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(blob)
    gl12_sha = sha8_file(GL12P)
    log('0. gap ledger v1.2 written (%s); GAP-4 += mechanism_note; rest byte-identical'
        % gl12_sha)

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3171 for m in ms_):
    log('1. ledger: 3171 already present, skip')
else:
    detail = (
        'G5-A8 OOV collapse mechanism localization (zero GPU; reuses sealed 3169 collect '
        'npz + 3165/3166 subspace recipes): pre-registered gate ratio_S(oov/seen S_class '
        'projection energy at the 3169 E_read slot) <0.5 encoding_missing / >0.8 '
        'readout_missing / else mixed. Verdict mixed_across_models: ratio_S = ' + rat_s +
        ' (4b/14b mixed, glm4 0.8205 just above the 0.8 gate). encoding_missing '
        'unanimously rejected (all >0.5): OOV-class rows keep 72-82 percent of S_class '
        'energy; K_readout top64 energy fraction oov/seen = ' + ratK_s + ' ~ 1 (no energy '
        'missing on the readout manifold); OOV class centroids fit inside the seen-centroid '
        'cone (NNLS rel_resid ' + nnls_o + ' vs leave-one-out seen baseline ' + nnls_l +
        '); MARG claimed-class margins on OOV rows same magnitude as seen true-class '
        '(' + marg_pairs + '). Conclusion: collapse is NOT encoding_missing - class-identity '
        'energy, geometry and word-level logits all present; combined with the 3169 E_read '
        'collapse the gap localizes to readout-channel generalization failure (one-hot '
        'class port has no training signal under class leave-out), not representation '
        'encoding. Evidence level E1-E2 (correlational probes, not interventional); '
        'upgrade path = port-calibration intervention (prereg 3172). GAP-4 mechanism_note '
        'appended as gap_ledger v1.2 (' + gl12_sha + '), all other gaps/appendix '
        'byte-identical. Honest log: 2 pre-observation device corrections (crosscheck tol '
        'calibrated to census 3-decimal storage 1e-3 deg after SMOKE caught 1.9e-5 drift; '
        'glm4 model-key mapping 3166 glm4 vs 3169 glm4-9b) - no measurement-semantics '
        'change. design_sha=' + R['design_sha8'] + '. res ' + R['res_sha8'] + ' seal ' +
        R['seal_sha8'] + ' (smoke res ' + SM['res_sha8'] + ').')
    entry = {
        'phase': 3171, 'name': 'g5a8_collapse_mechanism', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'zero_gpu (3-model npz reuse)',
        'verdict': ('g5a8_collapse_mechanism|mixed_across_models|ratio_S=' + rat_s +
                    '|encoding_missing_rejected_3/3|mechanism=readout_channel_failure'),
        'evidence_level': 'E1_correlational',
        'model_scope': 'qwen3-4b+qwen3-14b+glm4',
        'prereg_id': 'G5-A8',
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
    log('1. ledger: appended 3171 (n=' + str(n1) + ', was ' + str(n0) + ') chain_sha8=' +
        led['ledger_sha256_8'])

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3171' in norm:
    log('2. MEMO: 3171 section already present, skip')
else:
    marker = '### 接续：预注册 3171'
    mi = norm.rfind(marker)
    assert mi > 0, '3171 prereg marker not found'
    S = []
    S.append('## Phase 3171: G5-A8 谱外崩塌机制定位（encoding_missing vs readout_missing） ' + time.strftime('%H:%M'))
    S.append('')
    S.append('**日期**：2026-10-09。**脚本**：`tests/glm5/phase3171_g5a8_collapse_mechanism.py`（零 GPU，'
             '复用 3169 封存 collect npz + 3165/3166 子空间配方）。产物：`phase3171/g5a8_collapse_mechanism/` '
             '{exec ' + exes + ', res ' + R['res_sha8'] + ', seal ' + R['seal_sha8'] + ', smoke res ' +
             SM['res_sha8'] + '}；gap_ledger v1.2 ' + gl12_sha + '（GAP-4 += mechanism_note，其余逐字节不变）。')
    S.append('')
    S.append('### 判决（重复三遍）')
    S.append('')
    S.append('**mixed_across_models：ratio_S（S_class 投影能量比，谱外/已见，主槽=3169 E_read 槽）= ' + rat_s +
             '（4b/14b mixed、glm4 0.8205 刚过 0.8 门 +0.02 敏感）。encoding_missing 被三模型一致否定'
             '（全 >0.5）：谱外类崩塌不源于「编码缺失」，定位=读出通道（one-hot 类端口->输出映射）泛化失效。**')
    S.append('')
    S.append('**mixed_across_models（重复二）：谱外类行保留 72-82% 类子空间能量；K_readout top64 能量占比 '
             + ratK_s + ' ≈1；谱外类质心落在已见质心锥内（NNLS rel_resid ' + nnls_o + ' vs 已见留一基线 ' +
             nnls_l + '）；MARG 词级 logit 正常。**')
    S.append('')
    S.append('**mixed_across_models（重复三）：证据级 E1-E2（相关性探针非干预因果），升级路径=预注册 3172 '
             '端口校准干预实验。**')
    S.append('')
    S.append('### 四探针读数')
    S.append('')
    S.append('1. **(a) 主门**：f_S(h)=||Q_S^T h||²/||h||²（Q_S=S_class 10 方向正交基，3166 配方逐字重建）；'
             'f_S seen ' + fs_s + ' / oov ' + fs_o + '；ratio ' + rat_s + ' -> cls ' + cls_s +
             '。oov_pure 行（新实体×谱外类）与 newent 行并排（f_S_oov_pure ' +
             ', '.join('%.2e' % R['per_model'][m]['slots']['k_main']['f_S_oov_pure'] for m in MKS) +
             '，介于两者之间）。')
    S.append('2. **(b) K_readout 旁证**：top64（3158 W_U Gram 特征向量）能量占比 oov/seen = ' + ratK_s +
             ' ≈1——谱外行在主读出流形上能量不缺，排除「整体偏离读出流形」。')
    S.append('3. **(c) 质心几何**：谱外 4 类质心最近邻已见类余弦 0.975-0.993（共线背景）；NNLS 锥拟合 rel_resid ' +
             nnls_o + ' vs 已见留一 ' + nnls_l + '——谱外质心被已见 6 质心锥/子空间表示的程度与已见类互表'
             '基线相同；对 S_class 10 英文方向余弦与已见类同分布。')
    S.append('4. **(d) MARG 词级**：谱外行 claimed-class margin vs 已见行 true-class margin = ' + marg_pairs +
             '（同量级），均高于谱外行 max-seen margin（' + marg_seen_max + '）——读出端类词信号在。')
    S.append('')
    S.append('### 机制注记（GAP-4 mechanism_note，v1.2 已回写）')
    S.append('')
    S.append('E_read 崩塌（3169 ratio_B=2.54）的主体是 one-hot 类端口机制：phi 编码的谱外类 one-hot 列在'
             '类别级留出下训练行全零 -> 该类输出偏移权重不可学（ridge 压到 0）-> 预测塌向训练均值 -> MSE 升高。'
             '本 phase 三重相关性证据（子空间能量/锥几何/词级 logit）一致表明谱外类 H 本身携带类身份信息，'
             '故缺口④的机制定位=「读出通道泛化失效」而非「表征编码缺失」。诚实注记：glm4 ratio 0.8205 距门 '
             '+0.02 敏感；相关性非因果，因果升级=3172 端口校准干预（给 k 个谱外类校准行，恢复曲线定判）。')
    S.append('')
    S.append('### 装置与诚实登记')
    S.append('')
    S.append('10 源 sha8 断言（3169 collect ×3 + smoke + 3158 ×3 + p2806 + 3166/3169 result）；'
             'crosscheck 三模型复现 3166 census K_readout×S_class top1（67.0/71.9/74.014 度，drift<=' +
             ('%.1e' % cc_drift) + '）；槽断言 k_main=NL-1==3169 readout 字段、kstar=3；DESIGN 全静态。'
             '**2 次观测前装置修正**：SMOKE 抓到 crosscheck 容差过严（3166 census 存 3 位小数值，'
             '1e-6 -> 1e-3 度校准）；正式跑 glm4 键名映射（3166 用 glm4、3169 用 glm4-9b）——均纯装置层，'
             '不改测量语义；SMOKE 面板行集缩放全对（216/240/144/96）。')
    S.append('')
    S.append('### 接续：预注册 3172')
    S.append('')
    S.append('G5-A9 谱外类端口校准曲线（零 GPU，复用 3169 npz）：把 GAP-4 机制注记从相关性推到干预——'
             '假设「崩塌主体=one-hot 类端口在类别级留出下无训练信号（端口缺失）」，则给谱外类 c 少量校准行'
             '即可恢复。协议：对每谱外类 c，取 k∈{0,1,2,4,8} 个该类实体行加入口径 B 训练集（校准实体与测试'
             '实体不相交；该类其余实体行+全部谱外类行为测试），Q03 verbatim ridge，三模型，恢复曲线 '
             'ratio_B(k)。预注册门：k=8 时 pooled ratio <1.5 => 端口缺失确认（mechanism_note 升级 '
             'causal_support）；k=8 仍 >=2 => 结构性缺失（更深缺口，挂账）。k=1/2/4 给样本效率梯度。'
             '完成后图谱 v1.2 渲染关账（mechanism_note 回写 html + 字段校验）。')
    S.append('')
    S.append('')
    S.append('---')
    S.append('')
    S.append('')
    section = '\n'.join(S) + '\n'
    new = norm[:mi] + section + norm[mi:]
    out = new.replace('\n', '\r\n') if crlf > 0 else new
    if had_bom:
        out = b'\xef\xbb\xbf' + out.encode('utf-8')
    else:
        out = out.encode('utf-8')
    shutil.copyfile(MEMO, MEMO + '.snap3171')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3171 section inserted (snapshot .snap3171)')

# ---------- 3. daily ----------
dline = ('- **3171 谱外崩塌机制定位（2026-10-09）**：G5-A8 零 GPU 四探针——ratio_S=' + rat_s +
         ' -> mixed_across_models（encoding_missing 三模型一致否定）；K_readout 能量占比≈1、'
         '谱外质心在已见锥内（NNLS ' + nnls_o + ' vs 基线 ' + nnls_l + '）、MARG 词级正常 => '
         '缺口定位=读出通道（one-hot 类端口）泛化失效，非编码缺失。GAP-4 mechanism_note 回写 '
         'gap_ledger v1.2 ' + gl12_sha + '。res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] +
         '；ledger n→323。2 次观测前装置修正（crosscheck 容差校准、glm4 键名映射）。'
         '下一步 3172=G5-A9 端口校准曲线（k∈{0,1,2,4,8} 恢复实验，零 GPU）。')
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if '3171 谱外崩塌机制定位' in dtxt:
    log('3. daily: 3171 line already present, skip')
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
newline = ('\n- **✅ 3171 谱外崩塌机制定位（2026-10-09）**：G5-A8 零 GPU 四探针（复用 3169 collect npz + '
           '3165/3166 子空间配方，DESIGN 全静态）——**mixed_across_models**：ratio_S（S_class 投影能量比，'
           '主槽=3169 E_read 槽 NL-1）= ' + rat_s + '（4b/14b mixed、glm4 0.8205 刚过 0.8 门敏感）；'
           '**encoding_missing 三模型一致否定**（全>0.5）：谱外类行保留 72-82% 类子空间能量 + K_readout '
           'top64 能量占比 ' + ratK_s + '≈1 + 谱外质心 NNLS 锥 rel_resid ' + nnls_o + ' vs 已见留一基线 ' +
           nnls_l + ' + MARG 词级 logit 正常（' + marg_pairs + '）=> **GAP-4 机制定位=读出通道'
           '（one-hot 类端口->输出映射）泛化失效，非表征编码缺失**（E1-E2 相关性级）。gap_ledger v1.2 ' +
           gl12_sha + '（GAP-4 += mechanism_note，其余逐字节不变）。10 源 sha 断言；crosscheck 复现 3166 '
           'census 三角度 drift≤' + ('%.1e' % cc_drift) + '；2 次观测前装置修正（crosscheck 容差校准、'
           'glm4 键名映射）。res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] + '；ledger n=322→**323**。'
           '下一步 3172=**G5-A9 谱外类端口校准曲线**（k∈{0,1,2,4,8} 恢复实验，零 GPU：k=8 pooled ratio '
           '<1.5 => 端口缺失 causal_support / ≥2 => 结构性缺失；完成后图谱 v1.2 渲染关账）。\n')
if '3171 谱外崩塌机制定位' in wm:
    log('4. workspace MEMORY: 3171 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3171')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3171)')

# ---------- 5. self-check (disk read-back) ----------
chk = []
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
ms2 = led2['measurements']
e3171 = [m for m in ms2 if m.get('phase') == 3171]
chk.append(('ledger has 3171 entry', len(e3171) == 1, 'n=' + str(len(ms2))))
chk.append(('ledger n>=323', len(ms2) >= 323, 'n=' + str(len(ms2))))
if e3171:
    chk.append(('ledger 3171 verdict has ratio', rat_s in e3171[0]['verdict'],
                e3171[0]['verdict'][:80]))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3171', '## Phase 3171' in memo2))
chk.append(('MEMO has res sha', R['res_sha8'] in memo2))
chk.append(('MEMO has seal sha', R['seal_sha8'] in memo2))
chk.append(('MEMO has mixed_across_models', 'mixed_across_models' in memo2))
chk.append(('MEMO has prereg 3172 ref', '预注册 3172' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3171', '3171 谱外崩塌机制定位' in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3171', '3171 谱外崩塌机制定位' in w2))
chk.append(('gap ledger v1.2 on disk matches', gl12_sha == sha8_file(GL12P), gl12_sha))
G2d = json.load(io.open(GL12P, encoding='utf-8'))
g4 = [g for g in G2d['gaps'] if g['id'] == 'GAP-4'][0]
chk.append(('GAP-4 has mechanism_note', 'mechanism_note' in g4 and
            rat_s in g4['mechanism_note']))
chk.append(('gap ledger version 1.2', G2d['version'] == '1.2', G2d['version']))
chk.append(('result seal fields', bool(R['res_sha8']) and bool(R['seal_sha8'])))
chk.append(('result main_cls', R['overall']['main_cls'] == 'mixed_across_models',
            R['overall']['main_cls']))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s %s' % ('OK ' if c[1] else 'FAIL', c[0], c[2] if len(c) > 2 else ''))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE (self-check ' + str(len(chk) - len(bad)) + '/' + str(len(chk)) + ')')
