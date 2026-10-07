# -*- coding: utf-8 -*-
"""Phase 3141 closeout: ledger + MEMO
+ workspace log + MEMORY.md (idempotent)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D14 = os.path.join(
    RDIR, 'phase3141',
    'omega_p139_cinjrefine_wrjoint_'
    'coenrich_writeinduce')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (ROOT + r'\.workbuddy\memory'
        r'\2026-09-29.md')
WMEM = (ROOT + r'\.workbuddy\memory'
        r'\MEMORY.md')

RES_SHA = 'a93c8892'
SEAL_SHA = '7ccf7ffc'
VERDICT = ('a_3140_ok|repro_bit_9|'
           'repro_bit_ok|step1_peak_l19|'
           'step1_below_all|step1_dose_flat|'
           'peak_sharpened|dvec29_active|'
           'joint_blocking|'
           'co_enrich_causal_mixed|'
           'xphase_ok|v1fwd_bit_ok|'
           'writeinduce_flat|'
           'gen_retr_below|coverage_full')

# ---- verify result on disk ----
raw = io.open(os.path.join(D14,
                           'result.json'),
              'rb').read()
got_sha = hashlib.sha256(raw).hexdigest()[:8]
assert got_sha == RES_SHA, got_sha
resj = json.loads(raw.decode('utf-8'))
assert resj['verdict'] == VERDICT
assert str(resj['seal_sha8']) == SEAL_SHA
assert resj['smoke'] is False
print('result.json verified (sha8 %s)'
      % got_sha)

# ---- 1. ledger ----
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
_n0 = len(led['measurements'])
assert _n0 in (277, 278), _n0
entry = {
    'phase': 3141,
    'name': ('omega_p139_cinjrefine_'
             'wrjoint_coenrich_'
             'writeinduce'),
    'date': '2026-09-29',
    'kind': 'cinjrefine_wrjoint_'
            'coenrich_writeinduce',
    'verdict': VERDICT,
    'runtime_s': 11269.0,
    'hashes': {
        'result_sha256_8': RES_SHA,
        'seal_sha256_8': SEAL_SHA},
    'anchors': {
        'res14_sha8': '6d699902',
        'res37_sha8': '4f8055bb',
        'res36_sha8': '3903af46',
        'dvec_sha8': {
            '17': '5e4c3085',
            '29': 'ee9484b2',
            '33': '59fbe0d3',
            '38': 'aced803b'},
        'xphase': 'P1.0/A1.1.0',
        'repro_bit': '9/9'},
    'summary': (
        'step1: L23-28 allstep plateau '
        'collapses under single-step '
        'injection (0.06-0.14 vs 0.26-'
        '0.32) -> cumulative-write '
        'effect; L19 single-step peak '
        '0.1406 d2 (dose-mono, sharp '
        '2.00 vs 1.70). WR x dvec29 '
        'joint at L29: dvec29 dominant '
        '(0.2500/0.6406 dose-mono, '
        'first=1), joint 0.4766 < dvec '
        'alone -> joint_blocking, '
        'shared output bottleneck. '
        'co36 enrichment split: d1 hi '
        '0.2188 vs lo 0.0938 (2.3x) '
        'but d2 converges 1.18x -> '
        'mixed; enrichment ordering '
        'beats 3137 rank ordering at '
        'd1. Write-induced retrieval '
        'flat (gen 12 tok no gain, V1 '
        'fwd bit-replicates 3140) -> '
        'bank IDEINT identity is not '
        'session-writable; needs '
        'cross-template structure or '
        'weight-level memory. 9/9 bit '
        'anchors (5x 3140 + 1x wrpc1 + '
        '3x 3137 G); xphase P/A1 both '
        '1.0 (128/128).')}
_has = any(m.get('phase') == 3141
           for m in led['measurements'])
if not _has:
    led['measurements'].append(entry)
    io.open(LEDGER, 'w',
            encoding='utf-8').write(
        json.dumps(led, ensure_ascii=False,
                   indent=1))
led2 = json.load(io.open(LEDGER,
                         encoding='utf-8'))
assert len(led2['measurements']) == 278
assert led2['measurements'][-1][
    'phase'] == 3141
print('ledger n=278 OK')

# ---- 2. MEMO ----
block = (
    '\n\n## Phase 3141: step1 谱分离+WR'
    '叠加律+富集因果+写入诱导（T4 第24'
    'Phase）[14:55]\n\n'
    '**执行**：tests/glm5/'
    'phase3141_omega_p139_'
    'cinjrefine_wrjoint_coenrich_'
    'writeinduce.py，正式跑 11269s 一次'
    '通过。rev-3141a（PART A wr_trials '
    '取值 float 非 dict）。9/9 位级锚'
    '（5x3140 allstep/iinj17 + wrpc1_'
    'l29_d2 + 3x3137 G）；xphase P/A1 '
    '双 1.0（128/128）。result sha8='
    'a93c8892，seal sha8=7ccf7ffc，'
    'ledger n=278。\n\n'
    '### §1 C: cinj step1 精细化——'
    'L23-28 平台=累积效应\n'
    'allstep d2 对照 7 层全位级复现'
    '（L19 0.4922/L23 0.2891/L24 '
    '0.3203/L25 0.2734/L26 0.2891/'
    'L27 0.2812/L28 0.2578）。step1 '
    '（prompt+decode1，3135 语义）下 '
    'L23-28 平台崩溃：d2 全部降至 '
    '0.0625-0.1406（vs allstep '
    '0.2578-0.3203），剂量单调率仅 '
    '0.14（深层高剂量崩塌，L24 d4 '
    '0.0391<d2 0.1406）。唯 L19 单步'
    '敏感：d1 0.0703/d2 0.1406/d4 '
    '0.2422 单调，峰/中位比 2.00 vs '
    'allstep 1.70（sharpened）。\n\n'
    '**发现 1（×3）：cinj 谱两类结构'
    '——L19=单步敏感真峰（脉冲式）+ '
    'L23-28=累积依赖平台（逐步重复注'
    '入才改写行为，单步脉冲不足）。'
    '3140 的 L23-27 功能平台重新定性'
    '为累积写入效应，非瞬时读出端口。**\n\n'
    '### §2 D: WR PC1 x dvec29 联合='
    'blocking\n'
    'dvec29@L29 强活性：d1 0.2500/d2 '
    '0.6406 剂量单调、first=1（第一'
    '个解码步即翻转）。joint(pc1+dvec'
    ')d2=0.4766 < dvec29 单独 0.6406'
    '（p_ind 0.7136、p_add 0.8438 均'
    '远超）→ joint_blocking；d1 同'
    '（joint 0.2188 ≈ dvec 单独 '
    '0.2500）。wrpc1_l29_d2=0.2031 '
    'bit 复现。\n\n'
    '**发现 2（×3）：WR PC1（全局形态'
    '方向）与 dvec29（身份载体）在 '
    'L29 相互干扰而非叠加——joint 低'
    '于强分量单独，共享同一输出瓶颈；'
    '身份载体主导行为改写。3140 双通'
    '路模型修正为竞争型共享瓶颈。**\n\n'
    '### §3 E: co36 身份能量富集因果\n'
    'co36 50 坐标按 L17 IDEINT_P 能量'
    '份额排序分 high25/low25（high25'
    '=[2319,83,2309,3140,3099,...]，'
    '与 3137 rank 排序不同源）。d1: '
    'hi 0.2188 vs lo 0.0938（2.3x）'
    '——低剂量富集因果分离清晰；d2: '
    '0.2031 vs 0.1719（1.18x）收敛。'
    'tag=co_enrich_causal_mixed。3137 '
    '锚 co36 d1/d2、co50 d1 全 bit '
    '复现。\n\n'
    '**发现 3（×3）：身份能量富集度是'
    '比 co36_rank 更好的因果预测子'
    '（d1 尺度分离 2.3x vs rank 排序 '
    'flat 0.133/0.117）；高剂量饱和'
    '掩蔽结构差异——富集→因果的剂量'
    '窗口在低剂量区。**\n\n'
    '### §4 F: 写入诱导检索阴性\n'
    '84 unseen pairs（ents_n=28，'
    'chance 0.0357）。V1 fwd 位级复'
    '现 3140（0.0595/0.0357/0.0595）'
    '。生成 12 token（含答案+重述步）'
    '后检索无提升：gain 中位 0.0000'
    '（V1_L17 -0.0119、V2_L17 '
    '-0.0357），gen best 0.0595 '
    '<< 0.30。\n\n'
    '**发现 4（×3）：会话内写入（生成'
    '推理）不建立可检索身份方向——'
    'bank IDEINT 身份成分既非 prompt '
    '表面形式（3140 V2 flat）也非会'
    '话级写入（3141 gen flat）可达；'
    '指向跨模板结构（B/I/C/R 分解中 '
    'I 来自跨模板共享方差）或权重级'
    '（训练）记忆。3140 发现 5 进一步'
    '限定。**\n\n'
    '### §5 综合 + 3142 预注册\n'
    '3141 四问四答：(1) cinj 平台=累'
    '积效应，L19 真峰；(2) WR x 身份'
    '载体=blocking 竞争；(3) 富集度'
    '因果预测优于 rank 序（低剂量）；'
    '(4) 写入诱导阴性→身份读出需跨模'
    '板结构或权重记忆。\n\n'
    '3142（Ω-P140）预注册：\n'
    '1. L19 真峰机制解剖：dvec19 全新'
    '捕获（L19 位置身份载体，3135 冻'
    '结协议扩展）+ iinj/cinj x step1 '
    'x L17-21 精扫（step0.5 层距）定'
    '位单步敏感区边界；\n'
    '2. blocking 机制定位：注入顺序反'
    '转（dvec 先 pc1 后）+ 剂量矩阵 '
    'pc1 d{0.5,1,2} x dvec d{0.5,1,2}'
    '（L29）——区分读出竞争 vs 状态'
    '破坏；\n'
    '3. co36 富集四分位 x dose{0.5,1}'
    ' 曲线——定位富集→因果的剂量窗'
    '口边界；\n'
    '4. 跨模板结构检验（F 阴性归因）：'
    '新材料行 4 模板变体（V1/V2/T3/T4 '
    '重建）生成会话捕获→检验跨模板共'
    '享方差是否建立 I 成分并恢复检索。\n\n'
    '关键数字：repro 9/9；allstep d2 '
    'L19 0.4922/L24 0.3203/L28 0.2578；'
    'step1 d2 L19 0.1406/L24 0.1406；'
    'dvec29 0.2500/0.6406 first=1；'
    'joint d2 0.4766（p_add 0.8438）；'
    'co36 hi/lo d1 0.2188/0.0938、d2 '
    '0.2031/0.1719；retr fwd=gen≈chance。\n\n'
    '锚：result sha8=a93c8892，seal '
    'sha8=7ccf7ffc，ledger n=278。')
t = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3141' not in t:
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(block)
t2 = io.open(MEMO, encoding='utf-8').read()
assert t2.count('## Phase 3141') == 1
assert '3142（Ω-P140）预注册' in t2
print('MEMO ok')

# ---- 3. workspace log ----
add = ('\n\n## Phase 3141 (Omega-P139) '
       '闭环 [14:58]\n'
       '- 正式跑 11269s 一次通过。'
       'verdict: ' + VERDICT + '\n'
       '- 发现：L23-28 cinj 平台=累积'
       '效应（step1 崩溃 0.06-0.14 vs '
       'allstep 0.26-0.32）、L19 单步'
       '真峰 0.1406（sharp 2.00）；'
       'WR PC1 x dvec29 joint_blocking'
       '（dvec29 主导 0.6406，joint '
       '0.4766 < 单独）；co36 富集'
       '因果 d1 分离 2.3x（hi 0.2188/'
       'lo 0.0938）、d2 收敛；写入诱导'
       '阴性（gen 12 tok 无增益）。\n'
       '- 9/9 位级锚（3140x6 + 3137x3）；'
       'xphase P/A1 双 1.0。\n'
       '- closeout：ledger n=278、MEMO '
       '追加（T4 第24Phase）、wlog、'
       'MEMORY。result sha8=a93c8892 '
       'seal 7ccf7ffc。\n'
       '- 3142 预注册：L19 真峰解剖'
       '(dvec19 捕获+L17-21 精扫)、'
       'blocking 顺序/剂量矩阵、co36 '
       '四分位曲线、跨模板结构检验。')
tw = io.open(WLOG, encoding='utf-8').read()
if 'Phase 3141 (Omega-P139) 闭环' not in tw:
    with io.open(WLOG, 'a',
                 encoding='utf-8') as f:
        f.write(add)
tw2 = io.open(WLOG, encoding='utf-8').read()
assert tw2.count(
    'Phase 3141 (Omega-P139) 闭环') == 1
print('wlog ok')

# ---- 4. MEMORY.md ----
tm = io.open(WMEM, encoding='utf-8').read()
lines = tm.splitlines()
out_lines = []
changed = False
NEWLINE = ('- max=3141，下一 3142：'
           'L19 真峰机制解剖（dvec19 捕获'
           '+step1 L17-21 精扫）+ blocking '
           '顺序反转/剂量矩阵（读出竞争 vs '
           '状态破坏）+ co36 富集四分位剂'
           '量窗口 + 新材料 4 模板跨模板结'
           '构检验（写入诱导阴性归因）。'
           '**cinj 谱两类结构=L19 脉冲峰+'
           'L23-28 累积平台；WR x dvec='
           'blocking 共享瓶颈（身份载体主'
           '导）；富集度因果预测优于 rank '
           '序（低剂量 2.3x）；身份读出不'
           '可会话写入（fwd=gen≈chance）。**')
for l in lines:
    if l.startswith('- max='):
        l = NEWLINE
        changed = True
    out_lines.append(l)
if not changed:
    out_lines.append(NEWLINE)
# append/refresh 3141 result line after
# the 3140 line if missing
txt_new = '\n'.join(out_lines) + '\n'
if '3141（T4）' not in txt_new:
    marker = '\n- 3109：'
    idx = txt_new.find(marker)
    ins = ('\n- 3141（T4）：cinj step1 '
           '谱分离=L19 脉冲峰 0.1406+'
           'L23-28 累积平台（单步崩溃）；'
           'WR PC1 x dvec29 joint_blocking'
           '（dvec29 0.6406 主导，共享输'
           '出瓶颈）；co36 富集因果 d1 '
           '2.3x/低剂量窗口；写入诱导阴'
           '性=身份读出需跨模板结构或权'
           '重记忆。9/9 bit 锚。')
    if idx >= 0:
        txt_new = (txt_new[:idx] + ins
                   + txt_new[idx:])
io.open(WMEM, 'w',
        encoding='utf-8').write(txt_new)
tm2 = io.open(WMEM, encoding='utf-8').read()
assert 'max=3141' in tm2
assert '3141（T4）' in tm2
print('MEMORY.md ok (len %d)' % len(tm2))
print('CLOSEOUT 4/4 DONE')
