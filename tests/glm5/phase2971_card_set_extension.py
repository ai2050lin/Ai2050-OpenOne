# -*- coding: utf-8 -*-
"""Phase 2971: mechanism chain card set extension (2962-2970 nine rings -> 2961 card set).

ZERO forward, pure documentation. Modeled on phase2961 (primitive card compression):
- execution.json frozen BEFORE any artifact
- every number verbatim-traceable to sealed source result.json or its MEMO section
- verdict from frozen map only
"""
import hashlib
import io
import json
import os
import re
import time

BASE = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913')
SRC2961 = os.path.join(RES, 'phase2961', 'primitive_card_compression')
OUTD = os.path.join(RES, 'phase2971', 'card_set_extension')
SCRIPT = os.path.join(BASE, r'tests\glm5\phase2971_card_set_extension.py')
MEMO = os.path.join(BASE, r'research\gpt5\docs\AGI_GPT5_MEMO.md')

ARMS = {
    2962: 'word_class_signature_matrix',
    2963: 'frequency_controlled_band',
    2964: 'carrier_anatomy',
    2965: 'h15_functional_identity',
    2966: 'l34_routing_membership',
    2967: 'collapse_carrier_anatomy',
    2968: 'h15_peak_anatomy',
    2969: 'peak_word_attributes',
    2970: 'delay_carrier_localization',
}

MINUS = u'\u2212'  # MEMO sections use U+2212 for negative numbers

NEW_CARDS = [
 {
  'phase': 2962, 'ring': u'环24',
  'title': u'词类签名矩阵——读出坐标词盲复现与承重带边界带',
  'verdict': 'signature_mixed_pattern_registered',
  'layer_band': u'承重带（B gap）+ 路由边界带 L14-18 侧翼（ICC17）',
  'module': u'读出投影 sep @ u35 / 头路由分布 / B gap',
  'head_set': u'全 32 头路由分布（ICC，无 keep 门）',
  'readout': u'B gap（final-norm 输入投影 @ u35）+ S1/S2/S3 类差距',
  'dose_law': 'N.A.(非剂量 Phase)',
  'lin_r': 'N.A.(未入卡)',
  'mechanism': (u'读出 SVD 坐标层对词类关闭（2940 在 n=45 新类目复现）；'
                u'路由头分布仅边界带弱信号（ICC17 0.1163 p 0.0458，边界带 p 单独登记不与硬显著混池）；'
                u'功能词承重带差距 +1.23 为描述性——token-id/频率混淆强制在案（rho(tokid,B)=-0.605），'
                u'确认性检验留给频率受控预注册（2963 承接）。'),
  'key_numbers': ['0.1163', '0.0458', '1.23', '-0.605'],
  'vocab_note': (u'S1 用全英文功能/内容词表消语言混淆；此时语言轴未入环，'
                 u'2887 四语言结构尚未标注'),
 },
 {
  'phase': 2963, 'ring': u'环25',
  'title': u'频率受控复检——F/C 带差距为类效应与 Simpson 结构识破',
  'verdict': 'band_class_effect_beyond_frequency',
  'layer_band': u'承重带 L6-L12（F/C 类差距），全 36 层 CI 剖面',
  'module': u'o_proj 输入消融敏感度（2932 机器）+ Freedman-Lane',
  'head_set': 'N.A.(层带级统计)',
  'readout': u'raw CI 与 supp = raw_null - raw_func，协变量 token-id+频率',
  'dose_law': 'N.A.(非剂量)',
  'lin_r': u'层内 rho 中位 -0.5524（2936 口径复用）',
  'mechanism': (u'Freedman-Lane（协变量回归+残差置换+全模型重拟合）频率受控后 F/C 带差距 p 2e-4 '
                u'硬显著=类效应超越频率；全样本 rho(tokid,B)=-0.6464 强而 content 组内 -0.16 ns'
                u'=Simpson 结构（组均值差型混淆的正确 null）；名词内部无频率梯度（C≈R）；'
                u'协议恒等锚=重前向 5 词 B 比对 bit 级闭合（rel 7.76e-16）。'),
  'key_numbers': ['-0.6464', '-0.16', '2e-4', '7.76e-16', '1.46'],
  'vocab_note': (u'三词表 B gap 稳定 1.23/1.46/1.37——类效应对词表构成稳健，语言结构未标注'),
 },
 {
  'phase': 2964, 'ring': u'环26',
  'title': u'类效应载体解剖——L34/h15 单点定位与深层双带结构',
  'verdict': 'carrier_localized_layers_heads',
  'layer_band': u'唯一显著层 L34 主导 + L16/L30-32 次级双带（与 2932 承重带 L6-12 异带）',
  'module': u'全 36 层 o_proj 输入捕获 C[36,30,32] 头级贡献',
  'head_set': u'L34 头族 32（maxT）；top5 h15/h8/h21/h28/h11',
  'readout': u'头级 gap（final-norm 输入 @ u35）',
  'dose_law': 'N.A.',
  'lin_r': 'N.A.',
  'mechanism': (u'全新 30 词确认性复制（FL p 1.5e-03）；T2 maxT 族 36 唯一显著层 L34'
                u'（gap %s4.0，top-3 集中度 0.461）；T3 唯一显著头 h15；'
                u'rho(gap17, 2947 D_L17)=0.4296 中度同构，top5 与 2947-top5 空交集'
                u'（类效应头≠语言轴头），2953 早翻转头 h21 入 top5——'
                u'功能/内容差异是少数深层头专职分工，与语言轴头级部分解耦。' % MINUS),
  'key_numbers': [MINUS + '4.0', '0.4296', '1.37', '0.461'],
  'vocab_note': (u'全新 30 词（F15 tid 566-7241 / N15 tid 3241-26752）+ 3 个 2963 词重前向锚；'
                 u'语言结构未标注'),
 },
 {
  'phase': 2965, 'ring': u'环27',
  'title': u'h15 功能身份——最大单头因果贡献但消融后类效应存活',
  'verdict': 'h15_shared_carrier_effect_survives',
  'layer_band': u'L34（单头切片消融）',
  'module': u'o_proj 输入 pos-1 头切片置零 + batched 全前向（2932 标准件）',
  'head_set': u'h15（主）+ h8/h28 对照（top3 共享）',
  'readout': u'dirs_word[35] 投影 R + Freedman-Lane 类差距',
  'dose_law': 'N.A.(单点消融)',
  'lin_r': 'N.A.',
  'mechanism': (u'h15 = 最大单头因果载体（R 秩 1/32、rho -0.7953、u35 对齐秩 2/32），'
                u'但真实消融后类效应存活（FL p 2.1e-3，gap 仅缩 8.5%%）——共享载体、无单点必要头；'
                u'h8 选拔未显著但消融差分 0.1111 近同 h15（选拔量与消融差分口径分离，'
                u'maxT 选拔≠消融必要性，纪律 15 分账再确认）。'),
  'key_numbers': ['-0.7953', '2.1e-3', '8.5', '0.1111'],
  'vocab_note': u'2887 57 词语言 batch；语言结构未标注',
 },
 {
  'phase': 2966, 'ring': u'环28',
  'title': u'L34 路由成员判定——h15 biphasic 峰锁 s_c，B 单调塌缩',
  'verdict': 'l34_h15_independent_of_routing',
  'layer_band': u'L17 开关（A11 路由）×L34 载体（C15/B 响应）',
  'module': u'L17 xdir 注入（coef 1.0）+ 全层 o_proj 捕获 + v_proj',
  'head_set': u'h15（目标）+ 全头响应族 top5 h6/h31/h24/h2/h10',
  'readout': u'C15(s) 剂量曲线 + B(s) 带差分',
  'dose_law': u'biphasic（峰 0.7139 @ s=0.625 恰锁 s_c=0.6567）非 sigmoid 跟随',
  'lin_r': 'N.A.',
  'mechanism': (u'spearman(s,C15)=-0.5909 p 0.059 fail 单调门但曲线 biphasic 效应 0.43σ'
                u'——单调检验与峰位检验必须分开预注册（纪律 10 检验形状版）；'
                u'B 被语言方向注入单调抹平（rho 0.9364 p 1e-04，-1.365 近零），'
                u'h6 完美单调——词类载体与语言路由网络阈值解耦、带差分强耦合。'),
  'key_numbers': ['0.43', '0.6567', '0.9364', '-1.365', '1e-04'],
  'vocab_note': u'57 词语言 batch verbatim（2887 表）；语言结构未标注',
 },
 {
  'phase': 2967, 'ring': u'环29',
  'title': u'B 塌缩载体解剖——深层带八层分布式，L34 响应显著但未塌尽',
  'verdict': 'collapse_carrier_localized',
  'layer_band': u'L24-33 深层带八层分布式（318/1152 头显著）',
  'module': u'全层 o_proj 输入捕获 + 头族 maxT（层×头族）',
  'head_set': u'318/1152 有效头（maxT q<0.05），top12 全完美单调 |rho|=1.0',
  'readout': u'B(s) 端点比 + 逐头 rho(s, B_h)',
  'dose_law': u'端点比 0.5148 @ s=2（L34 显著响应 q 1e-3 但未塌尽）',
  'lin_r': 'N.A.',
  'mechanism': (u'B 抹平由 L24-33 八层分布式承载（top12 全 |rho|=1.0，无单点载体）；'
                u'L34"响应显著"与"塌缩彻底"是两个量——判据必须分层（显著层/载体层两档）；'
                u'同实现跨相位 bit 级复制锚第三次验证（a12-14 vs 2966 全 0.0）。'),
  'key_numbers': ['318', '1.0', '0.5148'],
  'vocab_note': u'57 词语言 batch；语言结构未标注',
 },
 {
  'phase': 2968, 'ring': u'环30',
  'title': u'biphasic 峰位表征——个体峰锁 s_c 分布中心，方向门控',
  'verdict': 'biphasic_locked_no_reverse',
  'layer_band': u'L34/h15（瞬态）×L17 开关',
  'module': u'双族设计 family A +xdir / family B -xdir（协议 2966 verbatim）',
  'head_set': u'h15 逐词 + h8/h21 对照',
  'readout': u'逐词 peak_loc 抛物线插值 + bootstrap CI',
  'dose_law': (u'40/57 词内峰、中位峰位 0.6746 vs s_c 0.6567（差 0.018）、'
               u'CI [0.5862,0.8375] 含 s_c'),
  'lin_r': 'N.A.',
  'mechanism': (u'双相性是个体性质（分布中心锁 L17 开关阈值，个体分布宽 0.29-1.61'
                u'是"分布中心锁"非逐词精确锁）；A11_L34_h15 增益峰与 C15 峰逐词 rho 0.9542'
                u'——瞬态过冲由路由增益个体驱动；反向 -xdir 下 h15 平坦（rho -0.4455 p 0.169）'
                u'sep 仅缓降——双相是关闭方向特有，方向门控。'),
  'key_numbers': ['0.6746', '0.6567', '0.9542', '0.169', '-0.4455'],
  'vocab_note': u'57 词语言 batch；语言结构未标注（2969 才发现四语言结构）',
 },
 {
  'phase': 2969, 'ring': u'环31',
  'title': u'峰位词属性——语言配对效应硬显著与 Simpson 结构',
  'verdict': 'peak_source_lang_within_pair_descriptive',
  'layer_band': u'L34/h15 峰位（逐词）',
  'module': u'配对符号翻转置换 + maxT（2968 峰位机器）',
  'head_set': 'N.A.(词级峰位统计)',
  'readout': u'逐词 peak_loc + (concept,lang) 配对差 d',
  'dose_law': 'N.A.',
  'lin_r': 'N.A.',
  'mechanism': (u'同 concept en/fr 配对 13/13 对 fr 峰位晚于 en（mean d +0.5132，p 2e-4）；'
                u'tid/频率组内无效（en rho 0.118 p 0.61 / L 0.052 p 0.84，maxT q 0.856）；'
                u'全样本 rho(tid,pk)=0.50 是组间 Simpson 结构（2963 规范复用）；'
                u'判据按 quasi-post-hoc 登记（2968 已并排展示峰位，纪律 9）。'),
  'key_numbers': ['0.5132', '2e-4', '0.118', '0.61', '0.856', '0.50'],
  'vocab_note': (u'Phase 2887 词表实为四语言（en/fr/de/es）——同 concept 多 L 词，'
                 u'(concept,lang) 配对口径敏感；语言结构自本 Phase 起标注'),
 },
 {
  'phase': 2970, 'ring': u'环32',
  'title': u'跨语言峰位延迟载体定位——深层带八层+分布式头群',
  'verdict': 'delay_carrier_heads_and_layers_localized',
  'layer_band': u'层级显著 [24,25,26,27,30,31,32,34]（与 2967 塌缩载体带重叠）',
  'module': u'2968 协议 verbatim family A + 全层头级贡献 C[36,11,57,32]',
  'head_set': u'359/1152 有效头（maxT q<0.05），top 全在 L19-32、符号双向',
  'readout': u'层级/头级各自 peak_loc → 配对语言差 d',
  'dose_law': u'层级 d 全正 0.56-0.83',
  'lin_r': 'N.A.',
  'mechanism': (u'跨语言峰位延迟载体=深层带八层+分布式头群，与 B 塌缩载体带重叠'
                u'——语言轴在深层带同时调制静态带差分（2967）与动态峰位时序（本环）；'
                u'rho(延迟,塌缩响应)=-0.486 中度负相关；配对口径恒等门'
                u'（d[34,15]=0.5132 vs 2969 diff 3.81e-06）抓住四语言表配对漂移'
                u'（11 对 vs 13 对）——跨产物统计管线必须设已知量恒等门；'
                u'transpose(1,3,0,2) 流顺序自检（错误 transpose 产假显著）。'),
  'key_numbers': ['0.5132', '3.81e-06', '0.861', '0.486'],
  'vocab_note': (u'2887 四语言表：配对必须峰词先行过滤（2969 口径，13 对）'
                 u'——cidx 全词覆盖后过滤=11 对是口径漂移'),
 },
]

SCHEMA = ['phase', 'ring', 'title', 'verdict', 'layer_band', 'module',
          'head_set', 'readout', 'dose_law', 'lin_r', 'mechanism',
          'key_numbers', 'source', 'vocab_note']
NA_OK = {'dose_law', 'lin_r', 'head_set'}

VERDICT_MAP = {
    'all_pass': 'primitive_cards_extended_chain_34',
    'anchor_fail': 'anchor_fail_all_void',
    'schema_fail': 'schema_incomplete_all_void',
}


def s8_binary(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:8]


def main():
    t0 = time.time()
    os.makedirs(OUTD, exist_ok=True)

    mem = io.open(MEMO, encoding='utf-8').read()
    sections = {}
    for ph in ARMS:
        m = re.search(r'## Phase %d:' % ph, mem)
        assert m, 'MEMO section for %d missing' % ph
        start = m.start()
        m2 = re.search(r'\n## Phase \d{4}:', mem[start + 10:])
        end = start + 10 + m2.start() if m2 else len(mem)
        sections[ph] = mem[start:end]

    sources = []
    for ph in ARMS:
        rel = 'phase%d/%s/result.json' % (ph, ARMS[ph])
        sources.append({'phase': ph, 'rel_path': rel,
                        'sha256_8': s8_binary(os.path.join(RES, rel.replace('/', os.sep)))})

    # ---- 1. freeze execution.json BEFORE anything else ----
    created = time.strftime('%Y-%m-%dT%H:%M:%S')
    execution = {
        'phase': 2971,
        'title': 'mechanism chain card set extension (2962-2970 nine rings)',
        'created': created,
        'model': 'qwen3-4b',
        'mode': ('ZERO forward, pure documentation: extend the 2961 primitive card set '
                 'with rings 24-32 (phases 2962-2970, word-class/language signature); '
                 'every number verbatim-traceable to the sealed source result.json or '
                 'its MEMO section; new field vocab_note registers the 2887 '
                 'four-language table structure annotation'),
        'card_schema': SCHEMA,
        'sources': sources,
        'anchors': {
            'a1': 'binary sha256-8 of each source result.json == MEMO section registration "result <h8>"',
            'a2': 'MEMO contains "## Phase 29NN:" header and the phase final_verdict string inside its section (9/9)',
            'a3': 'card phases == exactly {2936..2960} ∪ {2962..2970} (34 phases; '
                  '2961 is the card-set phase itself, not a ring card)',
            'a4': 'every key_number string occurs verbatim in the source result.json raw text OR the phase MEMO section text',
        },
        'tests': {
            'T1': 'schema completeness: 34 cards, all 14 fields non-empty (N.A. allowed), '
                  'new cards n_key_numbers >= 3, rings 24..32 sequential',
        },
        'verdict_map': VERDICT_MAP,
        'correction_note': ('run1 a3 criterion required a card for phase 2961 itself '
                            '(the card-set phase) — unreachable criterion, discipline-10 '
                            'recurrence; run2 fixed to {2936..2960} ∪ {2962..2970} and '
                            'old artifacts deleted before rerun'),
    }
    with io.open(os.path.join(OUTD, 'execution.json'), 'w', encoding='utf-8') as f:
        json.dump(execution, f, ensure_ascii=False, indent=1)

    # ---- 2. load 2961 cards + build new cards ----
    old = json.load(io.open(os.path.join(SRC2961, 'primitive_cards.json'),
                            encoding='utf-8'))
    assert old['phase'] == 2961 and len(old['cards']) == 25
    cards = []
    for c in old['cards']:
        c = dict(c)
        c['vocab_note'] = 'N.A.(语言轴未入环)'
        cards.append(c)
    for c, src in zip(NEW_CARDS, sources):
        c = dict(c)
        assert c['phase'] == src['phase']
        c['source'] = {'path': src['rel_path'], 'sha256_8': src['sha256_8']}
        cards.append(c)

    # ---- 3. anchors ----
    anchors = {}

    a1_rows = []
    for src in sources:
        reg = re.search(r'result ([0-9a-f]{8})', sections[src['phase']])
        ok = bool(reg) and reg.group(1) == src['sha256_8']
        a1_rows.append({'phase': src['phase'], 'sha8': src['sha256_8'],
                        'registered': reg.group(1) if reg else None, 'match': ok})
    anchors['a1'] = {'ok': all(r['match'] for r in a1_rows), 'rows': a1_rows}

    a2_rows = []
    for ph in ARMS:
        v = NEW_CARDS[[c['phase'] for c in NEW_CARDS].index(ph)]['verdict']
        ok = (v in sections[ph])
        a2_rows.append({'phase': ph, 'verdict': v, 'in_section': ok})
    anchors['a2'] = {'ok': all(r['in_section'] for r in a2_rows), 'rows': a2_rows}

    a3_ok = sorted(c['phase'] for c in cards) == (list(range(2936, 2961)) + list(range(2962, 2971)))
    anchors['a3'] = {'ok': a3_ok,
                     'n_cards': len(cards),
                     'phases_2936_2970_complete': a3_ok}

    a4_rows, a4_all = [], True
    for c, src in zip(cards[25:], sources):
        rp = os.path.join(RES, src['rel_path'].replace('/', os.sep))
        rtext = open(rp, 'rb').read()
        sec = sections[c['phase']].encode('utf-8')
        missing = []
        for kn in c['key_numbers']:
            kb = kn.encode('utf-8')
            if (kb not in rtext) and (kb not in sec):
                missing.append(kn)
        a4_all = a4_all and not missing
        a4_rows.append({'phase': c['phase'], 'missing': missing,
                        'n_key_numbers': len(c['key_numbers'])})
    anchors['a4'] = {'ok': a4_all, 'rows': a4_rows}

    # ---- 4. T1 schema completeness ----
    fields_missing_all, t1_rows = [], []
    rings_expected = [u'环%d' % i for i in range(24, 33)]
    rings_actual = [c['ring'] for c in cards[25:]]
    for i, c in enumerate(cards):
        missing = [k for k in SCHEMA
                   if k not in c or c[k] in ('', None)
                   or (isinstance(c[k], list) and not c[k])]
        fields_missing_all += ['card%d:%s' % (i, k) for k in missing]
    n_new_ok = all(len(c['key_numbers']) >= 3 for c in cards[25:])
    t1 = {
        'ok': (len(cards) == 34 and not fields_missing_all
               and n_new_ok and rings_actual == rings_expected),
        'n_cards': len(cards),
        'fields_missing': fields_missing_all,
        'new_cards_min_key_numbers_ok': n_new_ok,
        'rings_sequential': rings_actual == rings_expected,
    }

    # ---- 5. verdict (explicit in-branch assignment) ----
    if not (anchors['a1']['ok'] and anchors['a2']['ok']
            and anchors['a3']['ok'] and anchors['a4']['ok']):
        verdict = VERDICT_MAP['anchor_fail']
    elif not t1['ok']:
        verdict = VERDICT_MAP['schema_fail']
    else:
        verdict = VERDICT_MAP['all_pass']

    # ---- 6. outputs ----
    cards_json = {'phase': 2971, 'verdict': verdict, 'cards': cards}
    with io.open(os.path.join(OUTD, 'primitive_cards.json'), 'w',
                 encoding='utf-8') as f:
        json.dump(cards_json, f, ensure_ascii=False, indent=1)

    md = [u'# LPF v5.3 机制原语卡片集（34 卡，Phase 2971 扩充版）', '']
    md.append(u'- verdict: `%s`' % verdict)
    md.append(u'- 2961 前置/环1-23（25 卡）+ 2962-2970 环24-32（9 卡）；'
              u'新增 vocab_note 字段登记 2887 四语言表结构注记')
    md.append('')
    for c in cards:
        md.append(u'## 卡 %d · %s · %s' % (c['phase'], c['ring'], c['title']))
        md.append(u'- verdict: `%s`' % c['verdict'])
        for k in ('layer_band', 'module', 'head_set', 'readout',
                  'dose_law', 'lin_r'):
            md.append(u'- %s: %s' % (k, c[k]))
        md.append(u'- mechanism: %s' % c['mechanism'])
        md.append(u'- key_numbers: %s' % ', '.join(c['key_numbers']))
        md.append(u'- vocab_note: %s' % c['vocab_note'])
        md.append(u'- source: %s (sha256_8 %s)'
                  % (c['source']['path'], c['source']['sha256_8']))
        md.append('')
    with io.open(os.path.join(OUTD, 'primitive_cards.md'), 'w',
                 encoding='utf-8') as f:
        f.write('\n'.join(md))

    result = {
        'phase': 2971,
        'created': created,
        'runtime_s': round(time.time() - t0, 3),
        'anchors': anchors,
        'tests': {'T1': t1},
        'final_verdict': verdict,
        'outputs': ['primitive_cards.json', 'primitive_cards.md'],
        'n_cards': len(cards),
        'vocab_note_note': ('new field: 2887 four-language (en/fr/de/es) table '
                            'structure annotation; pairing = peak-word filter '
                            'first (2969 convention, 13 pairs)'),
    }
    with io.open(os.path.join(OUTD, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print('verdict:', verdict)
    print('anchors:', {k: v['ok'] for k, v in anchors.items()},
          'T1:', t1['ok'], 'n_cards:', len(cards))


if __name__ == '__main__':
    main()
